"""IAMCCS LongVid Pianosequenza v2 - upstream-parity final-chunk video latent engines.

GPL-3.0-compatible integration.  The IAMCCS project is GPL and these video
latent contracts are independently reimplemented from the behavior documented
and implemented in the upstream projects listed in THIRD_PARTY_PIANOSEQUENZA_V2.txt.

IMPORTANT SCOPE:
- chunk planner / 362-209-193 established geometry: untouched;
- chunk 1 output: untouched;
- chunk 2 output: untouched (its sampled AV latent is copied to CPU only);
- AudioCon and current audio masks/samples: authoritative and preserved;
- only the final chunk VIDEO latent transport / head conditioning changes;
- one normal H3 sample only.  There is no Pianosequenza second pass.
- v2.1 deduplicates terminal_reanchor twins inside the final chunk.
- v2.2 neutralizes the legacy LongVid VIDEO latent-tail half before Pianosequenza
  preparation, while preserving the existing IAMCCS AudioCon samples/mask.
- v2.2 also deduplicates terminal twins by encoded-latent equality when AddGuide
  metadata no longer carries the original guide id.
- v2.3 attempted to infer the editorial endpoint from Shotplan frame_count.
- v3.3 removes that metadata assumption entirely. It derives the actual
  handoff length from LongVid's runtime VIDEO mask and locates the exact copied
  editorial tail inside the cached sampled prior. No programme duration,
  segment count, overlap preset or project-specific frame count is hardcoded.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from typing import Any, Iterable

import torch

MARKER = "IAMCCS_LONGVID_PIANOSEQUENZA_V3_3_AGNOSTIC_AUTHORITY_ALIGNMENT"
LOG = logging.getLogger("IAMCCS.MiniMaxH3.PianosequenzaV2")
_CACHE: dict[str, dict[str, Any]] = {}
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
DRIFT_CONTROL_TAPER_STEPS = 4
_DRIFT_WRAPPER_KEY = "iamccs_pianosequenza_v2_drift_control"

MODES = {
    "pianosequenza_linear",
    "pianosequenza_drift",
    "pianosequenza_native",
    "pianosequenza_phase",
    "pianosequenza_frozen",
    "pianosequenza_hd",
}
# HD persists its exact editorial tail through the Native Checkpoint cache, not
# the legacy final-chunk-only in-memory penultimate cache below.
CACHE_MODES = (MODES - {"pianosequenza_hd"}) | {"pianosequenza_2stage"}


def normalize_mode(value: Any) -> str:
    raw = str(value or "hard_image").strip().lower()
    if raw == "latent_free":
        return "latent_free_closure"
    return raw


def _streams(latent: dict[str, Any], label: str = "latent"):
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if samples is None or not bool(getattr(samples, "is_nested", False)):
        raise RuntimeError(f"{label} must be a native MiniMax H3 nested AV latent")
    try:
        streams = list(samples.unbind())
    except Exception:
        streams = list(getattr(samples, "tensors", ()))
    if len(streams) != 2 or not all(torch.is_tensor(x) for x in streams):
        raise RuntimeError(f"{label} must contain exactly video+audio tensors")
    video, audio = streams
    if video.ndim != 5 or audio.ndim != 4:
        raise RuntimeError(f"{label} has invalid H3 AV dimensions")
    return video, audio


def _nested(video, audio):
    import comfy.nested_tensor
    return comfy.nested_tensor.NestedTensor((video, audio))


def _mask_streams(latent: dict[str, Any], video, audio):
    masks = latent.get("noise_mask") if isinstance(latent, dict) else None
    if masks is not None and bool(getattr(masks, "is_nested", False)):
        try:
            parts = list(masks.unbind())
        except Exception:
            parts = list(getattr(masks, "tensors", ()))
        if len(parts) == 2:
            return parts[0], parts[1]
    vm = torch.ones((video.shape[0], 1, video.shape[2], 1, 1), dtype=torch.float32, device=video.device)
    am = torch.ones((audio.shape[0], 1, 1, audio.shape[-1]), dtype=torch.float32, device=audio.device)
    return vm, am


def pixel_frames(latent_t: int) -> int:
    latent_t = int(latent_t)
    if latent_t < 0:
        raise ValueError("latent_t must be >= 0")
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(latent_t))


def latent_boundaries(latent_t: int):
    out = [0]
    acc = 0
    for k in range(int(latent_t)):
        acc += FRAME_PER_TOKEN[k % 5]
        out.append(acc)
    return out


def step_offsets(latent_t: int):
    return latent_boundaries(latent_t)[:-1]


def video_latent_t(frame_count: int) -> int:
    frame_count = int(frame_count)
    if frame_count < 5 or frame_count % 17 != 5:
        raise ValueError(f"H3 frame count must satisfy 17k+5, got {frame_count}")
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def _valid_guide_frames(requested: int) -> int:
    requested = int(requested)
    if requested < 5:
        return 1
    while requested % 17 != 5:
        requested -= 1
    return max(1, requested)


def _cache_key(shotplan: dict[str, Any]) -> str:
    sampling = shotplan.get("sampling") if isinstance(shotplan.get("sampling"), dict) else {}
    track = shotplan.get("guide_track") if isinstance(shotplan.get("guide_track"), dict) else {}
    events = []
    for item in track.get("events", []) if isinstance(track.get("events"), list) else []:
        if isinstance(item, dict) and str(item.get("kind", "")).lower() == "image":
            events.append((
                str(item.get("id", "")), str(item.get("source_path", "")),
                int(item.get("global_frame", 0) or 0), int(item.get("end_frame", 0) or 0),
            ))
    payload = {
        "w": int(shotplan.get("width", 0) or 0),
        "h": int(shotplan.get("height", 0) or 0),
        "duration": float(shotplan.get("requested_duration_seconds", 0.0) or 0.0),
        "segments": int(shotplan.get("total_segments", 0) or 0),
        "seed": int(sampling.get("seed", 0) or 0),
        "seed_policy": str(sampling.get("seed_policy", "")),
        "events": events,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _cpu_copy(latent: dict[str, Any]) -> dict[str, Any]:
    video, audio = _streams(latent, "sampled penultimate latent")
    out = {
        k: v for k, v in latent.items()
        if k not in {"samples", "noise_mask", "iamccs_pianosequenza_2stage_low_carry"}
    }
    out["samples"] = _nested(
        video.detach().to(device="cpu", copy=True),
        audio.detach().to(device="cpu", copy=True),
    )
    return out


def cache_penultimate(sampled: dict[str, Any], shotplan: dict[str, Any], chunk_index: int) -> str:
    mode = normalize_mode(shotplan.get("terminal_endpoint_mode"))
    if mode not in CACHE_MODES:
        return "off"
    chunks = shotplan.get("chunks", []) if isinstance(shotplan.get("chunks"), list) else []
    if len(chunks) < 2 or int(chunk_index) != len(chunks) - 2:
        return "not-penultimate"
    key = _cache_key(shotplan)
    _CACHE[key] = _cpu_copy(sampled)
    while len(_CACHE) > 4:
        _CACHE.pop(next(iter(_CACHE)))
    v, a = _streams(_CACHE[key], "cached penultimate latent")
    return f"cached_cpu key={key[:10]} video_t={int(v.shape[2])} audio_t={int(a.shape[-1])}"


def _legacy_handoff_from_target(target_latent, target_video, target_audio):
    """Recover the actual LongVid handoff from the already-prepared final latent.

    This is intentionally project-agnostic.  No duration, segment count,
    overlap frame value or known project geometry is used.  The legacy LongVid
    A/B path has already copied the exact editorial source tail into the final
    target prefix and written its 0->1 video denoise ramp.  That runtime state
    is therefore the strongest available authority for both handoff length and
    handoff pixels/latents.
    """
    if not bool(target_latent.get("iamccs_longvid_latent_tail_ab", False)):
        raise RuntimeError(
            "Pianosequenza 3.3 cannot resolve the LongVid handoff: "
            "iamccs_longvid_latent_tail_ab is absent."
        )
    video_mask, _audio_mask = _mask_streams(target_latent, target_video, target_audio)
    if not torch.is_tensor(video_mask) or video_mask.ndim != 5:
        raise RuntimeError("Pianosequenza 3.3 requires the LongVid VIDEO denoise mask")
    if int(video_mask.shape[2]) != int(target_video.shape[2]):
        raise RuntimeError("Pianosequenza 3.3 LongVid VIDEO mask/time geometry mismatch")

    profile = video_mask.detach().float().mean(dim=(0, 1, 3, 4)).cpu()
    if profile.numel() < 2:
        raise RuntimeError("Pianosequenza 3.3 final VIDEO mask is too short")
    eps = 1.0e-5
    constrained = torch.nonzero(profile < (1.0 - eps), as_tuple=False).reshape(-1)
    if constrained.numel() < 1:
        raise RuntimeError(
            "Pianosequenza 3.3 cannot find the LongVid 0->1 VIDEO handoff ramp in the final latent"
        )
    last_constrained = int(constrained[-1].item())
    # LongVid's proven A/B mask is linspace(0,1,N): the final handoff token is
    # exactly 1.0 and therefore is one token after the last value < 1.0.
    handoff_t = last_constrained + 2
    if handoff_t > int(target_video.shape[2]):
        raise RuntimeError("Pianosequenza 3.3 inferred handoff exceeds final VIDEO latent")

    prefix = profile[:handoff_t]
    if float(prefix[0].item()) > eps:
        raise RuntimeError("Pianosequenza 3.3 LongVid handoff ramp does not start at 0")
    if abs(float(prefix[-1].item()) - 1.0) > 5.0e-4:
        raise RuntimeError("Pianosequenza 3.3 LongVid handoff ramp does not end at 1")
    if handoff_t > 2 and bool(torch.any(prefix[1:] + eps < prefix[:-1])):
        raise RuntimeError("Pianosequenza 3.3 LongVid handoff ramp is not monotonic")
    if handoff_t < int(profile.numel()) and bool(torch.any(profile[handoff_t:] < (1.0 - eps))):
        raise RuntimeError("Pianosequenza 3.3 found a non-prefix VIDEO mask constraint")

    handoff_frames = pixel_frames(handoff_t)
    # pixel_frames(N) is the native H3 token clock.  This validation makes the
    # code work for any legal configured tail rather than a fixed 22f preset.
    if video_latent_t(handoff_frames) != handoff_t:
        raise RuntimeError("Pianosequenza 3.3 handoff token/frame roundtrip failed")
    handoff_tail = target_video[:1, :, :handoff_t].detach().clone()
    return handoff_tail, {
        "handoff_frames": int(handoff_frames),
        "handoff_tokens": int(handoff_t),
        "mask_start": float(prefix[0].item()),
        "mask_end": float(prefix[-1].item()),
    }


def _align_prior_video_to_runtime_handoff(previous_video, handoff_tail):
    """Locate LongVid's exact editorial handoff inside the raw sampled prior.

    The raw sampled latent may contain hidden technical suffix tokens.  Instead
    of guessing an endpoint from frame-count metadata, search for the actual
    tail LongVid copied into the final target.  This remains valid for arbitrary
    programme duration, arbitrary number of chunks and any legal tail length.
    """
    raw_t = int(previous_video.shape[2])
    tail_t = int(handoff_tail.shape[2])
    if tail_t < 1 or tail_t > raw_t:
        raise RuntimeError(
            f"Pianosequenza 3.3 invalid handoff extent: tail={tail_t}t raw={raw_t}t"
        )
    if tuple(previous_video.shape[3:]) != tuple(handoff_tail.shape[3:]):
        raise RuntimeError("Pianosequenza 3.3 handoff/prior spatial latent mismatch")

    # Matching on CPU float32 avoids VRAM spikes and tolerates harmless dtype
    # conversion while still requiring the copied latent tail to be genuinely
    # the same runtime handoff.
    raw_cpu = previous_video[:1].detach().to(device="cpu", dtype=torch.float32)
    tail_cpu = handoff_tail[:1].detach().to(device="cpu", dtype=torch.float32)
    candidates = []
    for start in range(0, raw_t - tail_t + 1):
        window = raw_cpu[:, :, start:start + tail_t]
        diff = (window - tail_cpu).abs()
        max_abs = float(diff.max().item())
        mse = float((diff * diff).mean().item())
        candidates.append((mse, max_abs, start))
    candidates.sort(key=lambda row: (row[0], row[1], -row[2]))
    best_mse, best_max, best_start = candidates[0]

    # The LongVid handoff is copied directly from this sampled latent.  A loose
    # floating tolerance permits dtype round-trips but rejects semantic matches.
    scale = max(1.0, float(tail_cpu.abs().max().item()))
    max_tol = 2.0e-4 * scale
    mse_tol = (5.0e-5 * scale) ** 2
    if best_max > max_tol or best_mse > mse_tol:
        raise RuntimeError(
            "Pianosequenza 3.3 could not locate LongVid's editorial handoff in the cached prior: "
            f"best_max_abs={best_max:.3e}, best_mse={best_mse:.3e}, "
            f"tolerance={max_tol:.3e}/{mse_tol:.3e}."
        )

    # Fail closed if a second location is effectively indistinguishable.  This
    # prevents a frozen/repeated latent from silently choosing the wrong time.
    ambiguous = []
    for mse, max_abs, start in candidates[1:]:
        if max_abs <= max_tol and mse <= mse_tol:
            ambiguous.append(start)
    if ambiguous:
        raise RuntimeError(
            "Pianosequenza 3.3 found an ambiguous repeated LongVid handoff in the cached prior: "
            f"best_start={best_start}t alternatives={ambiguous}. Refusing to guess."
        )

    endpoint_t = best_start + tail_t
    endpoint_frames = pixel_frames(endpoint_t)
    aligned = previous_video[:, :, :endpoint_t]
    if int(aligned.shape[2]) != endpoint_t:
        raise RuntimeError("Pianosequenza 3.3 runtime handoff alignment audit failed")
    return aligned, {
        "raw_frames": int(pixel_frames(raw_t)),
        "raw_tokens": int(raw_t),
        "endpoint_frames": int(endpoint_frames),
        "endpoint_tokens": int(endpoint_t),
        "hidden_suffix_frames": int(pixel_frames(raw_t) - endpoint_frames),
        "hidden_suffix_tokens": int(raw_t - endpoint_t),
        "match_start_token": int(best_start),
        "match_max_abs": float(best_max),
        "match_mse": float(best_mse),
    }


def _load_prior(shotplan: dict[str, Any], target_latent, target_video, target_audio):
    key = _cache_key(shotplan)
    cached = _CACHE.get(key)
    if cached is None:
        raise RuntimeError(
            "Pianosequenza v2 has no sampled penultimate latent. Run the LongVid from chunk 1; "
            "final-chunk-only execution is intentionally refused."
        )
    pv, pa = _streams(cached, "cached penultimate latent")
    if tuple(pv.shape[3:]) != tuple(target_video.shape[3:]):
        raise RuntimeError(
            f"Pianosequenza v2 spatial latent mismatch: previous {tuple(pv.shape[3:])} vs final {tuple(target_video.shape[3:])}"
        )

    handoff_tail, handoff = _legacy_handoff_from_target(target_latent, target_video, target_audio)
    aligned_cpu, matched = _align_prior_video_to_runtime_handoff(pv, handoff_tail.to(device=pv.device, dtype=pv.dtype))

    chunks = shotplan.get("chunks", []) if isinstance(shotplan.get("chunks"), list) else []
    source_index = len(chunks) - 2
    if source_index < 0:
        raise RuntimeError("Pianosequenza 3.3 requires a previous LongVid chunk")
    source_alignment = {
        "source_chunk": int(source_index) + 1,
        **handoff,
        **matched,
        "alignment": "runtime_authority_tail_match",
    }
    return (
        aligned_cpu.to(device=target_video.device, dtype=target_video.dtype),
        pa.to(device=target_audio.device, dtype=target_audio.dtype),
        key,
        source_alignment,
    )


def _guide_identity(kf: dict[str, Any]) -> str:
    if not isinstance(kf, dict):
        return ""
    for key in ("guide_id", "id", "slot_id", "source_id", "name", "label"):
        value = kf.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _latent_equivalent(a, b) -> bool:
    if not (torch.is_tensor(a) and torch.is_tensor(b)):
        return False
    if tuple(a.shape) != tuple(b.shape):
        return False
    if torch.equal(a, b):
        return True
    try:
        # Same source image encoded twice may differ by tiny dtype/cast noise.
        delta = (a.detach().float() - b.detach().float()).abs().max()
        return bool(float(delta.item()) <= 1e-6)
    except Exception:
        return False


def _collapse_terminal_reanchor_duplicates(kfs):
    """Keep one terminal image authority, even if AddGuide discarded textual IDs.

    First use the explicit ``__terminal_reanchor`` identity when present.  Then
    use the latest visual keyframe as the endpoint candidate and remove earlier
    keyframes carrying the same encoded latent.  This catches the observed
    @90 + @193 duplicate without relying on UI-only guide identifiers.
    """
    if not isinstance(kfs, (list, tuple)):
        return kfs, 0
    items = [dict(k) if isinstance(k, dict) else k for k in kfs]
    suffix = "__terminal_reanchor"
    terminal_roots: dict[str, float] = {}
    for kf in items:
        if not isinstance(kf, dict):
            continue
        gid = _guide_identity(kf)
        if not gid.endswith(suffix):
            continue
        root = gid[:-len(suffix)]
        pos = kf.get("resolved_frame_index")
        pos_f = float(pos) if isinstance(pos, (int, float)) else float("inf")
        terminal_roots[root] = max(pos_f, terminal_roots.get(root, float("-inf")))

    removed_indices: set[int] = set()
    for idx, kf in enumerate(items):
        if not isinstance(kf, dict):
            continue
        gid = _guide_identity(kf)
        if not gid or gid.endswith(suffix) or gid not in terminal_roots:
            continue
        pos = kf.get("resolved_frame_index")
        pos_f = float(pos) if isinstance(pos, (int, float)) else float("-inf")
        if pos_f < terminal_roots[gid]:
            removed_indices.add(idx)

    visual = []
    for idx, kf in enumerate(items):
        if not isinstance(kf, dict) or not torch.is_tensor(kf.get("latent")):
            continue
        pos = kf.get("resolved_frame_index")
        if isinstance(pos, (int, float)):
            visual.append((float(pos), idx, kf))
    if visual:
        _, terminal_idx, terminal_kf = max(visual, key=lambda row: row[0])
        terminal_latent = terminal_kf.get("latent")
        terminal_pos = float(terminal_kf.get("resolved_frame_index"))
        for pos, idx, kf in visual:
            if idx == terminal_idx or pos >= terminal_pos:
                continue
            if _latent_equivalent(kf.get("latent"), terminal_latent):
                removed_indices.add(idx)

    kept = [item for idx, item in enumerate(items) if idx not in removed_indices]
    return kept, len(removed_indices)


def _prepare_conditioning_for_pianosequenza(positive, protected_frames: int):
    """Remove competing head guides and deduplicate terminal reanchor twins."""
    if not isinstance(positive, list):
        return positive, 0, 0
    out = []
    removed = 0
    deduped = 0
    for entry in positive:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2 or not isinstance(entry[1], dict):
            out.append(entry)
            continue
        meta = dict(entry[1])
        kfs = meta.get("minimax_keyframes")
        if isinstance(kfs, (list, tuple)):
            trimmed = []
            for kf in kfs:
                if not isinstance(kf, dict):
                    trimmed.append(kf)
                    continue
                pos = kf.get("resolved_frame_index")
                if isinstance(pos, (int, float)) and 0 <= float(pos) < float(protected_frames):
                    removed += 1
                    continue
                trimmed.append(dict(kf))
            trimmed, deduped_here = _collapse_terminal_reanchor_duplicates(trimmed)
            deduped += int(deduped_here)
            meta["minimax_keyframes"] = trimmed
        repl = list(entry)
        repl[1] = meta
        out.append(repl)
    return out, removed, deduped


def _set_keyframes(positive, additions: list[dict[str, Any]]):
    import node_helpers
    if not isinstance(positive, list) or not positive or not isinstance(positive[0], (list, tuple)) or len(positive[0]) < 2:
        raise RuntimeError("Pianosequenza v2 requires standard ComfyUI CONDITIONING metadata")
    existing = positive[0][1].get("minimax_keyframes", []) if isinstance(positive[0][1], dict) else []
    keyframes = [dict(k) for k in existing if isinstance(k, dict)]
    keyframes.extend(additions)
    keyframes.sort(key=lambda k: float(k.get("resolved_frame_index", 0)))
    return node_helpers.conditioning_set_values(positive, {"minimax_keyframes": keyframes})


def _replace_video_keep_audio(target_latent, new_video, new_video_mask=None):
    target_video, target_audio = _streams(target_latent, "final target latent")
    old_video_mask, old_audio_mask = _mask_streams(target_latent, target_video, target_audio)
    if tuple(new_video.shape) != tuple(target_video.shape):
        raise RuntimeError("Pianosequenza v2 changed final video latent geometry")
    out = {k: v for k, v in target_latent.items() if k not in {"samples", "noise_mask"}}
    out["samples"] = _nested(new_video, target_audio)
    if new_video_mask is None:
        new_video_mask = old_video_mask
    out["noise_mask"] = _nested(new_video_mask, old_audio_mask)
    return out


def _neutralize_legacy_video_tail(target_latent):
    """Give Pianosequenza exclusive ownership of final-chunk VIDEO continuity.

    IAMCCS AudioCon remains untouched: target audio samples and its native
    hard-prefix/half-cosine release mask are retained exactly.  Only the video
    half produced by the legacy LongVid Latent Tail A/B path is reset to the
    fresh H3 target before the selected Pianosequenza method installs its own
    continuation contract.
    """
    video, audio = _streams(target_latent, "final target latent")
    legacy_present = bool(target_latent.get("iamccs_longvid_latent_tail_ab", False))
    if not legacy_present:
        return target_latent, "not_present"
    fresh_video = torch.zeros_like(video)
    free_video_mask = torch.ones(
        (video.shape[0], 1, video.shape[2], 1, 1),
        dtype=torch.float32,
        device=video.device,
    )
    out = _replace_video_keep_audio(target_latent, fresh_video, free_video_mask)
    out["iamccs_longvid_latent_tail_ab_video_neutralized"] = True
    out["iamccs_pianosequenza_video_authority"] = "exclusive"
    _, out_audio = _streams(out, "neutralized final target latent")
    if not torch.equal(out_audio.detach().cpu(), audio.detach().cpu()):
        raise RuntimeError("Pianosequenza v2.2 altered AudioCon while neutralizing legacy VIDEO tail")
    return out, "neutralized"


# -------------------------------------------------------------------------
# PIANOSEQUENZA LINEAR / DRIFT
# Behavioral parity with the direct-tail temporal-mask family used by finite
# H3 continuation.  Audio is intentionally restored to IAMCCS AudioCon.
# -------------------------------------------------------------------------

def _direct_tail_mask_video(target_latent, previous_video, guide_frames: int, gradient: bool):
    video, audio = _streams(target_latent, "final target latent")
    actual_frames = _valid_guide_frames(int(guide_frames))
    tokens = video_latent_t(actual_frames)
    if tokens > int(previous_video.shape[2]) or tokens > int(video.shape[2]):
        raise RuntimeError(f"Pianosequenza overlap {actual_frames}f/{tokens}t does not fit")
    out_video = video.clone()
    out_video[:, :, :tokens] = previous_video[:1, :, -tokens:].to(out_video)
    mask = torch.ones((1, 1, video.shape[2], 1, 1), dtype=torch.float32, device=video.device)
    ramp = (
        torch.linspace(0.0, 1.0, steps=tokens, dtype=mask.dtype, device=mask.device)
        if gradient else torch.zeros(tokens, dtype=mask.dtype, device=mask.device)
    )
    mask[:, :, :tokens] = ramp.view(1, 1, -1, 1, 1)
    return _replace_video_keep_audio(target_latent, out_video, mask), {
        "frames": actual_frames, "video_tokens": tokens,
        "mask_start": float(ramp[0].item()), "mask_end": float(ramp[-1].item()),
    }


def _schedule_values(sigmas: Any) -> tuple[float, ...]:
    values: Iterable[Any] = sigmas.detach().float().reshape(-1).cpu() if torch.is_tensor(sigmas) else (sigmas or ())
    clean = []
    for value in values:
        f = float(value)
        if math.isfinite(f) and f >= 0.0:
            clean.append(f)
    return tuple(sorted(set(clean), reverse=True))


def _next_schedule_sigma(current_sigma: float, sigmas: Any) -> float:
    current = float(current_sigma)
    if not math.isfinite(current) or current <= 0.0:
        return 0.0
    tolerance = max(1e-7, abs(current) * 1e-6)
    for candidate in _schedule_values(sigmas):
        if candidate < current - tolerance:
            return candidate
    return 0.0


def _matched_noise_ratio(current_sigma: float, sigmas: Any) -> float:
    current = float(current_sigma)
    if not math.isfinite(current) or current <= 0.0:
        return 0.0
    return max(0.0, min(1.0, _next_schedule_sigma(current, sigmas) / current))


def _temporal_prefix_weights(prefix_steps: int, taper_steps: int = DRIFT_CONTROL_TAPER_STEPS):
    count = int(prefix_steps)
    taper = min(int(taper_steps), count)
    if count < 1 or taper < 1:
        raise ValueError("Pianosequenza DRIFT prefix/taper must be positive")
    values = [1.0] * (count - taper)
    values.extend(float(taper - i - 1) / float(taper) for i in range(taper))
    return tuple(values)


class _DriftMaskState:
    def __init__(self, video_shape, audio_shape, sigmas, prefix_steps):
        self.video_shape = tuple(int(v) for v in video_shape)
        self.audio_shape = tuple(int(v) for v in audio_shape)
        self.sigmas = _schedule_values(sigmas)
        self.prefix_steps = int(prefix_steps)
        self.current_packed_mask = None
        self.current_video_mask = None
        self.current_audio_mask = None
        self.pianosequenza_stage_base_mask = None
        self.pianosequenza_stage_hard_lock = False

    def _update_masks(self, sigma, packed_mask):
        if self.pianosequenza_stage_hard_lock:
            output = packed_mask.clone()
            v_elements = math.prod(self.video_shape[1:])
            video_mask = output[..., :v_elements].reshape(self.video_shape)
            video_mask = torch.ceil(video_mask[:, :1].float() * 256.0) / 256.0
        else:
            current = float(torch.as_tensor(sigma).detach().float().reshape(-1)[0])
            ratio = _matched_noise_ratio(current, self.sigmas)
            output = packed_mask.clone()
            v_elements = math.prod(self.video_shape[1:])
            video_mask = output[..., :v_elements].reshape(self.video_shape)
            weights = torch.tensor(
                _temporal_prefix_weights(self.prefix_steps),
                device=video_mask.device, dtype=video_mask.dtype,
            ).mul_(ratio)
            video_mask[:, :, :self.prefix_steps] = weights.view(1, 1, self.prefix_steps, 1, 1)
            video_mask = torch.ceil(video_mask[:, :1].float() * 256.0) / 256.0
        self.current_packed_mask = output
        self.current_video_mask = video_mask
        v_elements = math.prod(self.video_shape[1:])
        audio = output[..., v_elements:]
        if int(audio.numel()) != math.prod(self.audio_shape):
            raise ValueError("Pianosequenza DRIFT packed audio mask geometry mismatch")
        self.current_audio_mask = audio.reshape(self.audio_shape)[:, :1].clone()
        return output

    def configure_pianosequenza_stage(self, video_shape, video_mask, audio_mask, hard_lock=False):
        """Bind DRIFT to the active low- or full-resolution 2-stage H3 grid."""
        shape = tuple(int(v) for v in video_shape)
        if len(shape) != 5:
            raise ValueError("Pianosequenza 2 Stage DRIFT requires a 5D H3 video latent")
        if audio_mask.ndim != len(self.audio_shape):
            raise ValueError("Pianosequenza 2 Stage received an invalid H3 audio mask")
        if audio_mask.shape[0] not in (1, self.audio_shape[0]) or audio_mask.shape[1] not in (1, self.audio_shape[1]):
            raise ValueError("Pianosequenza 2 Stage changed H3 audio batch/channel geometry")
        if any(ms not in (1, ss) for ms, ss in zip(audio_mask.shape[2:], self.audio_shape[2:])):
            raise ValueError("Pianosequenza 2 Stage changed H3 audio latent geometry")
        self.video_shape = shape
        self.pianosequenza_stage_hard_lock = bool(hard_lock)
        expanded_video = video_mask.expand(shape[0], shape[1], shape[2], shape[3], shape[4]).reshape(shape[0], 1, -1)
        expanded_audio = audio_mask.expand(self.audio_shape).reshape(shape[0], 1, -1)
        self.pianosequenza_stage_base_mask = torch.cat((expanded_video, expanded_audio), dim=-1)
        self.current_packed_mask = self.pianosequenza_stage_base_mask
        self.current_video_mask = video_mask
        self.current_audio_mask = audio_mask

    def denoise_mask_function(self, sigma, denoise_mask, extra_options=None):
        if not self.sigmas:
            self.sigmas = _schedule_values((extra_options or {}).get("sigmas", ()))
        return self._update_masks(sigma, denoise_mask)

    def apply_model_wrapper(self, executor, *args, **kwargs):
        if self.pianosequenza_stage_base_mask is not None:
            sigma = args[1] if len(args) > 1 else kwargs.get("t")
            if sigma is None:
                raise ValueError("Pianosequenza 2 Stage DRIFT model call has no sigma")
            self._update_masks(sigma, self.pianosequenza_stage_base_mask)
        if self.current_video_mask is not None:
            kwargs["denoise_mask"] = self.current_video_mask
        if self.current_audio_mask is not None:
            kwargs["audio_denoise_mask"] = self.current_audio_mask
        return executor(*args, **kwargs)


def _install_drift_model(model, latent, sigmas, prefix_steps: int):
    if model is None or not callable(getattr(model, "clone", None)):
        raise RuntimeError("Pianosequenza DRIFT requires a cloneable ComfyUI MODEL")
    video, audio = _streams(latent, "DRIFT target")
    patched = model.clone()
    options = getattr(patched, "model_options", None)
    if not isinstance(options, dict):
        raise RuntimeError("Pianosequenza DRIFT model has no model_options")
    if callable(options.get("denoise_mask_function")):
        raise RuntimeError("Pianosequenza DRIFT refuses to stack another dynamic denoise-mask patch")
    if not callable(getattr(patched, "set_model_denoise_mask_function", None)) or not callable(getattr(patched, "add_wrapper_with_key", None)):
        raise RuntimeError("Pianosequenza DRIFT requires current ComfyUI denoise-mask/model-wrapper APIs")
    from comfy.patcher_extension import WrappersMP
    state = _DriftMaskState(tuple(video.shape), tuple(audio.shape), sigmas, int(prefix_steps))
    patched.set_model_denoise_mask_function(state.denoise_mask_function)
    patched.add_wrapper_with_key(WrappersMP.APPLY_MODEL, _DRIFT_WRAPPER_KEY, state.apply_model_wrapper)
    patched.model_options[_DRIFT_WRAPPER_KEY] = state
    return patched


# -------------------------------------------------------------------------
# PIANOSEQUENZA NATIVE
# Native H3 latent-tail guide: fresh unmasked VIDEO target, one frame-zero
# keyframe containing the synchronized previous video tail.  IAMCCS AudioCon is
# left untouched by design.
# -------------------------------------------------------------------------

def _native_video_target_and_guide(target_latent, previous_video, overlap_frames: int):
    video, audio = _streams(target_latent, "final target latent")
    actual = _valid_guide_frames(overlap_frames)
    tokens = video_latent_t(actual)
    if tokens >= int(video.shape[2]) or tokens > int(previous_video.shape[2]):
        raise RuntimeError("PIANOSEQUENZA NATIVE overlap does not fit target/source")
    fresh_video = torch.zeros_like(video)
    full_mask = torch.ones((video.shape[0], 1, video.shape[2], 1, 1), dtype=torch.float32, device=video.device)
    fresh = _replace_video_keep_audio(target_latent, fresh_video, full_mask)
    guide = {
        "resolved_frame_index": 0,
        "latent": previous_video[:1, :, -tokens:].clone(),
    }
    return fresh, guide, {"frames": actual, "video_tokens": tokens}


# -------------------------------------------------------------------------
# PIANOSEQUENZA PHASE
# Canonical phase-aligned direct latent continuation.  The source end is the
# latest real source boundary and the context is extended backward to phase 0.
# -------------------------------------------------------------------------

def _latest_boundary_at_or_before(boundaries, desired_end_exclusive: int) -> int:
    for i in range(len(boundaries) - 1, 0, -1):
        if boundaries[i] <= int(desired_end_exclusive):
            return i
    return 0


def _phase_aligned_extended_context_slice(video_t: int, context_frames: int, desired_tail_frames: int = 0):
    video_t = int(video_t)
    context_frames = int(context_frames)
    boundaries = latent_boundaries(video_t)
    previous_frames = boundaries[-1]
    desired_end_exclusive = max(context_frames, previous_frames - max(0, int(desired_tail_frames)))
    ideal_last_frame = desired_end_exclusive - 1
    end_t = _latest_boundary_at_or_before(boundaries, desired_end_exclusive)
    if end_t <= 0:
        raise RuntimeError("PIANOSEQUENZA PHASE found no source H3 boundary")
    start_t = None
    for candidate in range(end_t - 1, -1, -1):
        if candidate % 5 != 0:
            continue
        if boundaries[end_t] - boundaries[candidate] >= context_frames:
            start_t = candidate
            break
    if start_t is None:
        raise RuntimeError("PIANOSEQUENZA PHASE source is too short for phase-0 context")
    offsets = [boundaries[k] - boundaries[start_t] for k in range(start_t, end_t)]
    canonical = step_offsets(end_t - start_t)
    if offsets != canonical:
        raise RuntimeError("PIANOSEQUENZA PHASE canonical offset audit failed")
    return {
        "start_t": start_t, "end_t": end_t, "context_steps": end_t - start_t,
        "offsets": offsets,
        "source_start_frame": boundaries[start_t], "source_end_frame": boundaries[end_t],
        "previous_frame_count": previous_frames,
        "actual_context_frames": boundaries[end_t] - boundaries[start_t],
        "context_extension_frames": boundaries[end_t] - boundaries[start_t] - context_frames,
        "ideal_handover_end_frame": ideal_last_frame,
        "cutoff_loss_frames": desired_end_exclusive - boundaries[end_t],
    }


def _phase_video_target_and_guides(target_latent, previous_video, context_frames: int = 22):
    video, audio = _streams(target_latent, "final target latent")
    sl = _phase_aligned_extended_context_slice(int(previous_video.shape[2]), context_frames, desired_tail_frames=0)
    source = previous_video[:1, :, sl["start_t"]:sl["end_t"]].clone()
    guides = [
        {"resolved_frame_index": int(offset), "latent": source[:, :, i:i+1]}
        for i, offset in enumerate(sl["offsets"])
    ]
    fresh_video = torch.zeros_like(video)
    full_mask = torch.ones((video.shape[0], 1, video.shape[2], 1, 1), dtype=torch.float32, device=video.device)
    fresh = _replace_video_keep_audio(target_latent, fresh_video, full_mask)
    return fresh, guides, sl


# -------------------------------------------------------------------------
# PIANOSEQUENZA FROZEN
# Exact frozen previous-video tail as target prefix; constant mask 0 over the
# inherited video prefix, 1 over new suffix.  Audio remains IAMCCS AudioCon.
# -------------------------------------------------------------------------

def _aligned_overlap_frames(requested: int, maximum: int) -> int:
    requested = min(max(0, int(requested)), int(maximum))
    if requested < 5:
        raise ValueError("Pianosequenza FROZEN overlap must be at least 5 frames")
    return 5 + 17 * ((requested - 5) // 17)


def _frozen_video_target(target_latent, previous_video, overlap_frames: int = 22, context_denoise: float = 0.0):
    video, audio = _streams(target_latent, "final target latent")
    target_frames = pixel_frames(int(video.shape[2]))
    source_frames = pixel_frames(int(previous_video.shape[2]))
    overlap = _aligned_overlap_frames(overlap_frames, min(source_frames, target_frames))
    tokens = video_latent_t(overlap)
    out_video = torch.zeros_like(video)
    out_video[:, :, :tokens] = previous_video[:, :, -tokens:].to(out_video)
    mask = torch.ones((video.shape[0], 1, video.shape[2], 1, 1), dtype=torch.float32, device=video.device)
    mask[:, :, :tokens] = float(context_denoise)
    return _replace_video_keep_audio(target_latent, out_video, mask), {
        "frames": overlap, "video_tokens": tokens, "context_denoise": float(context_denoise)
    }


def _prepare_pianosequenza_hd_before_sample(
    *, positive, target_latent, model, sigmas, shotplan, chunk_index: int, render_id: str = ""
):
    """Prepare the self-contained HD continuation contract.

    Chunk 1 has no inherited prefix. Every later chunk loads the previous
    editorial native latent tail persisted by Native Checkpoint, gives that
    prefix exclusive VIDEO ownership, installs the sigma-matched DRIFT model,
    and forwards the previous LOW-resolution tail to the HD progressive sampler.
    IAMCCS AudioCon samples/masks are never replaced here.
    """
    target_video, target_audio = _streams(target_latent, "PIANOSEQUENZA_HD target latent")
    chunks = shotplan.get("chunks", []) if isinstance(shotplan.get("chunks"), list) else []
    if int(chunk_index) <= 0:
        return positive, target_latent, model, (
            "PIANOSEQUENZA_HD | opening_chunk=yes | carry=none | drift=armed_for_continuations | "
            "audio=IAMCCS_AudioCon_untouched"
        )
    if not str(render_id or "").strip():
        raise RuntimeError(
            "PIANOSEQUENZA_HD continuation has no render_id. Start from chunk 1 and let Native Checkpoint queue the continuation."
        )

    from .iamccs_minimax_h3_continuity import load_longvid_sampled_latent_tail
    previous = load_longvid_sampled_latent_tail(str(render_id), int(chunk_index) - 1)
    if previous is None:
        raise RuntimeError(
            f"PIANOSEQUENZA_HD could not load sampled latent tail for chunk {int(chunk_index)}. "
            "Run the sequence from chunk 1; continuation-only execution is intentionally refused."
        )
    previous_video = previous.get("video_tail")
    if not torch.is_tensor(previous_video) or previous_video.ndim != 5:
        raise RuntimeError("PIANOSEQUENZA_HD cache has no valid previous VIDEO latent tail")
    if tuple(previous_video.shape[-2:]) != tuple(target_video.shape[-2:]):
        raise RuntimeError(
            "PIANOSEQUENZA_HD previous HIGH latent tail does not match the current target grid: "
            f"{tuple(previous_video.shape[-2:])} != {tuple(target_video.shape[-2:])}"
        )

    requested_tail = int(
        (shotplan.get("pianosequenza_hd_settings") or {}).get("tail_frames", previous.get("tail_frames", 22))
        if isinstance(shotplan.get("pianosequenza_hd_settings"), dict)
        else previous.get("tail_frames", 22)
    )
    overlap = _valid_guide_frames(requested_tail)
    cached_overlap = int(previous.get("tail_frames", overlap) or overlap)
    if cached_overlap != overlap:
        raise RuntimeError(
            f"PIANOSEQUENZA_HD tail mismatch: cached={cached_overlap}f requested={overlap}f"
        )

    positive, removed, deduped = _prepare_conditioning_for_pianosequenza(positive, overlap)
    prepared, legacy_video = _neutralize_legacy_video_tail(target_latent)
    prepared, details = _direct_tail_mask_video(prepared, previous_video, overlap, gradient=False)

    previous_low = previous.get("pianosequenza_2stage_low_video_tail")
    if torch.is_tensor(previous_low):
        if previous_low.ndim != 5:
            raise RuntimeError("PIANOSEQUENZA_HD cached LOW tail is not 5D")
        prepared["iamccs_pianosequenza_2stage_previous_low_tail"] = previous_low.detach().to(device="cpu", copy=True)

    active_model = _install_drift_model(model, prepared, sigmas, details["video_tokens"])

    pv, pa = _streams(prepared, "prepared PIANOSEQUENZA_HD target")
    if tuple(pv.shape) != tuple(target_video.shape):
        raise RuntimeError("PIANOSEQUENZA_HD changed VIDEO latent geometry")
    if not torch.equal(pa.detach().cpu(), target_audio.detach().cpu()):
        raise RuntimeError("PIANOSEQUENZA_HD changed IAMCCS AudioCon latent samples")

    report = (
        f"PIANOSEQUENZA_HD | chunk={int(chunk_index)+1}/{max(1, len(chunks))} | "
        f"previous_tail={details['frames']}f/{details['video_tokens']}t | drift=sigma_matched | "
        f"taper={min(DRIFT_CONTROL_TAPER_STEPS, details['video_tokens'])}t | "
        f"low_tail={'yes' if torch.is_tensor(previous_low) else 'no'} | "
        f"head_guides_removed={removed} | terminal_duplicates_removed={deduped} | "
        f"legacy_video_tail={legacy_video} | audio=IAMCCS_AudioCon_untouched"
    )
    return positive, prepared, active_model, report


def prepare_pianosequenza_before_sample(*, mode: str, positive, target_latent, model, sigmas, shotplan, chunk_index: int, render_id: str = ""):
    mode = normalize_mode(mode)
    if mode == "pianosequenza_hd":
        return _prepare_pianosequenza_hd_before_sample(
            positive=positive,
            target_latent=target_latent,
            model=model,
            sigmas=sigmas,
            shotplan=shotplan,
            chunk_index=int(chunk_index),
            render_id=str(render_id or ""),
        )
    if mode not in MODES:
        return positive, target_latent, model, "off"
    chunks = shotplan.get("chunks", []) if isinstance(shotplan.get("chunks"), list) else []
    if not chunks or int(chunk_index) != len(chunks) - 1:
        return positive, target_latent, model, f"armed:{mode};waiting-final-chunk"

    target_video, target_audio = _streams(target_latent, "final target latent")
    previous_video, previous_audio, cache_key, source_alignment = _load_prior(shotplan, target_latent, target_video, target_audio)

    overlap = int(source_alignment["handoff_frames"])
    positive, removed, deduped = _prepare_conditioning_for_pianosequenza(positive, overlap)
    prepared, legacy_video = _neutralize_legacy_video_tail(target_latent)
    active_model = model

    if mode == "pianosequenza_linear":
        prepared, details = _direct_tail_mask_video(prepared, previous_video, overlap, gradient=True)
        report = (
            f"PIANOSEQUENZA LINEAR | one-pass pre-sample | previous_tail={details['frames']}f/{details['video_tokens']}t | "
            f"video_mask={details['mask_start']:.3f}->{details['mask_end']:.3f} | head_guides_removed={removed} | terminal_duplicates_removed={deduped} | "
            "audio=IAMCCS_AudioCon_untouched"
        )

    elif mode == "pianosequenza_drift":
        prepared, details = _direct_tail_mask_video(prepared, previous_video, overlap, gradient=False)
        active_model = _install_drift_model(model, prepared, sigmas, details["video_tokens"])
        report = (
            f"PIANOSEQUENZA DRIFT | one-pass pre-sample | frozen_source_prefix={details['frames']}f/{details['video_tokens']}t | "
            f"sigma_matched=yes taper={min(DRIFT_CONTROL_TAPER_STEPS, details['video_tokens'])}t | head_guides_removed={removed} | terminal_duplicates_removed={deduped} | "
            "audio=IAMCCS_AudioCon_untouched"
        )

    elif mode == "pianosequenza_native":
        prepared, guide, details = _native_video_target_and_guide(prepared, previous_video, overlap)
        positive = _set_keyframes(positive, [guide])
        report = (
            f"PIANOSEQUENZA NATIVE | one-pass pre-sample | fresh_video_target=yes | "
            f"native_tail_guide@0={details['frames']}f/{details['video_tokens']}t | head_guides_removed={removed} | terminal_duplicates_removed={deduped} | "
            "hidden_overlap_is_generated_context | audio=IAMCCS_AudioCon_untouched"
        )

    elif mode == "pianosequenza_phase":
        prepared, guides, details = _phase_video_target_and_guides(prepared, previous_video, context_frames=overlap)
        positive = _set_keyframes(positive, guides)
        report = (
            f"PIANOSEQUENZA PHASE | one-pass pre-sample | phase0_context={details['actual_context_frames']}f/"
            f"{details['context_steps']}t (+{details['context_extension_frames']}f) | guides={len(guides)} | "
            f"cutoff_loss={details['cutoff_loss_frames']}f | head_guides_removed={removed} | terminal_duplicates_removed={deduped} | "
            "audio=IAMCCS_AudioCon_untouched"
        )

    elif mode == "pianosequenza_frozen":
        prepared, details = _frozen_video_target(prepared, previous_video, overlap, context_denoise=0.0)
        report = (
            f"PIANOSEQUENZA FROZEN | one-pass pre-sample | exact_previous_tail={details['frames']}f/"
            f"{details['video_tokens']}t | prefix_denoise={details['context_denoise']:.3f} | "
            f"head_guides_removed={removed} | terminal_duplicates_removed={deduped} | audio=IAMCCS_AudioCon_untouched"
        )
    else:
        raise RuntimeError(f"Unsupported Pianosequenza v2 mode {mode}")

    # Geometry and audio ownership audits: these are non-negotiable.
    pv, pa = _streams(prepared, "prepared Pianosequenza target")
    if tuple(pv.shape) != tuple(target_video.shape):
        raise RuntimeError("Pianosequenza v2 changed final chunk video latent geometry")
    if not torch.equal(pa.detach().cpu(), target_audio.detach().cpu()):
        raise RuntimeError("Pianosequenza v2 changed IAMCCS AudioCon latent samples")
    report += (
        f" | cache={cache_key[:10]} | source_chunk={source_alignment['source_chunk']} | "
        f"handoff={source_alignment['handoff_frames']}f/{source_alignment['handoff_tokens']}t(runtime) | "
        f"source_endpoint=authority_aligned(raw={source_alignment['raw_frames']}f/{source_alignment['raw_tokens']}t"
        f"->{source_alignment['endpoint_frames']}f/{source_alignment['endpoint_tokens']}t;"
        f"hidden_suffix={source_alignment['hidden_suffix_frames']}f/{source_alignment['hidden_suffix_tokens']}t;"
        f"match_start={source_alignment['match_start_token']}t;max_abs={source_alignment['match_max_abs']:.2e}) | "
        f"legacy_video_tail={legacy_video} | video_authority=pianosequenza_only | geometry=preserved"
    )
    return positive, prepared, active_model, report
