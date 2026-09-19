"""IAMCCS MiniMax H3 Terminal Latent Closure v1.

Refines only the terminal video-latent window of the already-sampled final
LongVid chunk. It never creates a new technical chunk, never switches task
mode, never decodes/re-encodes the source motion, and never re-denoises audio.
"""
from __future__ import annotations

import math
from typing import Any

import torch

MARKER = "IAMCCS_TERMINAL_LATENT_CLOSURE_V1"
PATCH_REVISION = "IAMCCS_TERMINAL_LATENT_CLOSURE_V1_1_PROTECTED_RESTORE"
CONTRACT_KEY = "iamccs_terminal_latent_closure_v1"
_FALLBACK_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)


def _frame_pattern() -> tuple[int, ...]:
    try:
        from comfy.ldm.minimax.model import FRAME_PER_TOKEN
        pattern = tuple(int(v) for v in FRAME_PER_TOKEN)
        if pattern and all(v > 0 for v in pattern):
            return pattern
    except Exception:
        pass
    return _FALLBACK_FRAME_PER_TOKEN


def _streams(latent: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(latent, dict):
        raise RuntimeError("Terminal Latent Closure requires a MiniMax H3 LATENT dictionary.")
    samples = latent.get("samples")
    if samples is None or not bool(getattr(samples, "is_nested", False)):
        raise RuntimeError("Terminal Latent Closure requires the native nested MiniMax H3 AV latent.")
    streams = list(samples.unbind())
    if len(streams) != 2 or not all(torch.is_tensor(item) for item in streams):
        raise RuntimeError("Terminal Latent Closure requires exactly video+audio latent streams.")
    video, audio = streams
    if video.ndim != 5 or int(video.shape[1]) != 24:
        raise RuntimeError(f"Unexpected MiniMax H3 video latent shape: {tuple(video.shape)}")
    if audio.ndim != 4:
        raise RuntimeError(f"Unexpected MiniMax H3 audio latent shape: {tuple(audio.shape)}")
    return video, audio


def _token_spans(token_count: int) -> list[tuple[int, int]]:
    pattern = _frame_pattern()
    spans: list[tuple[int, int]] = []
    cursor = 0
    for index in range(max(0, int(token_count))):
        length = pattern[index % len(pattern)]
        spans.append((cursor, cursor + length))
        cursor += length
    return spans


def _total_pixel_frames(token_count: int) -> int:
    spans = _token_spans(token_count)
    return spans[-1][1] if spans else 0


def attach_terminal_closure(
    latent: dict[str, Any],
    *,
    source_path: str,
    frame_idx: int,
    guide_id: str,
    chunk_index: int,
    total_chunks: int,
    window_frames: int = 39,
    denoise: float = 0.30,
    steps: int = 6,
) -> dict[str, Any]:
    """Attach a final-chunk-only refinement contract without touching AV samples."""
    video, _audio = _streams(latent)
    if int(total_chunks) < 1 or int(chunk_index) != int(total_chunks) - 1:
        raise RuntimeError("Terminal Latent Closure can only be attached to the final technical LongVid chunk.")
    source_path = str(source_path or "").strip()
    if not source_path:
        raise RuntimeError("Terminal Latent Closure needs the authored terminal image source path.")
    total_frames = _total_pixel_frames(int(video.shape[2]))
    target = int(frame_idx)
    if target < 0 or target >= total_frames:
        raise RuntimeError(
            f"Terminal Latent Closure endpoint {target} is outside the H3 latent clock ({total_frames} pixel frames)."
        )
    out = dict(latent)
    out[CONTRACT_KEY] = {
        "schema": "iamccs.minimax_h3.terminal_latent_closure",
        "schema_version": 1,
        "enabled": True,
        "source_path": source_path,
        "guide_id": str(guide_id or "terminal"),
        "frame_idx": target,
        "chunk_index": int(chunk_index),
        "total_chunks": int(total_chunks),
        "window_frames": max(22, min(56, int(window_frames))),
        "denoise": max(0.05, min(0.50, float(denoise))),
        "steps": max(4, min(8, int(steps))),
        "curve": "half_cosine",
        "audio_lock": "bit_exact",
        "source": MARKER,
    }
    return out


def get_terminal_closure_contract(latent: Any) -> dict[str, Any] | None:
    if not isinstance(latent, dict):
        return None
    value = latent.get(CONTRACT_KEY)
    if not isinstance(value, dict) or not bool(value.get("enabled")):
        return None
    return dict(value)


def build_terminal_closure_latent(
    sampled: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use sampled AV as truth and expose only a smooth terminal video mask."""
    video, audio = _streams(sampled)
    spans = _token_spans(int(video.shape[2]))
    total_frames = spans[-1][1] if spans else 0
    target = int(contract.get("frame_idx", -1))
    if target < 0 or target >= total_frames:
        raise RuntimeError(
            f"Terminal Latent Closure endpoint {target} is outside sampled latent clock ({total_frames})."
        )
    window_frames = max(22, min(56, int(contract.get("window_frames", 39))))
    window_start = max(0, target - window_frames + 1)
    active = [
        index for index, (start, end) in enumerate(spans)
        if start <= target and (end - 1) >= window_start
    ]
    if len(active) < 2:
        raise RuntimeError("Terminal Latent Closure window resolves to fewer than two H3 video tokens.")

    video_mask = torch.zeros(
        (int(video.shape[0]), 1, int(video.shape[2]), 1, 1),
        device=video.device,
        dtype=torch.float32,
    )
    values: list[float] = []
    for order, token_index in enumerate(active):
        x = float(order) / float(max(1, len(active) - 1))
        value = 0.5 - 0.5 * math.cos(math.pi * x)
        values.append(value)
        video_mask[:, :, token_index:token_index + 1, :, :] = float(value)

    # Audio is never sampled a second time. This is independent from AudioCon,
    # which already did its job at the incoming chunk junction.
    audio_mask = torch.zeros(
        (int(audio.shape[0]), 1, int(audio.shape[2]), int(audio.shape[-1])),
        device=audio.device,
        dtype=torch.float32,
    )

    import comfy.nested_tensor

    out = dict(sampled)
    out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))

    # The first token in the closure range has a half-cosine value of exactly
    # zero. Include it in the protected-prefix audit: this mathematically ties
    # the refinement to the original sampled motion instead of starting a new
    # local T2V sample at the closure boundary.
    protected_prefix_tokens = int(active[0]) + 1
    protected_suffix_start = int(active[-1]) + 1
    guard = {
        "video_shape": tuple(video.shape),
        "audio_shape": tuple(audio.shape),
        "prefix_tokens": protected_prefix_tokens,
        "suffix_start": protected_suffix_start,
        "prefix": video[:, :, :protected_prefix_tokens].detach().to(device="cpu", copy=True),
        "suffix": video[:, :, protected_suffix_start:].detach().to(device="cpu", copy=True),
        "audio": audio.detach().to(device="cpu", copy=True),
        "active_tokens": tuple(active),
        "mask_values": tuple(values),
        "window_start_frame": int(window_start),
        "target_frame": int(target),
        "total_frames": int(total_frames),
    }
    return out, guard


def finalize_terminal_closure(
    original_sampled: dict[str, Any],
    closed_sampled: dict[str, Any],
    guard: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Restore every protected AV region exactly, then audit bit-for-bit.

    H3/Comfy sampling can return tiny numerical changes even on rows whose
    denoise mask is zero. A zero mask means "do not author this region";
    production therefore restores those rows from the original sampled latent.
    """
    original_video, original_audio = _streams(original_sampled)
    closed_video, closed_audio = _streams(closed_sampled)

    if tuple(closed_video.shape) != tuple(guard.get("video_shape", ())):
        raise RuntimeError("Terminal Latent Closure changed the video latent shape.")
    if tuple(closed_audio.shape) != tuple(guard.get("audio_shape", ())):
        raise RuntimeError("Terminal Latent Closure changed the audio latent shape.")
    if not bool(torch.isfinite(closed_video).all()) or not bool(torch.isfinite(closed_audio).all()):
        raise RuntimeError("Terminal Latent Closure produced non-finite AV latent values.")

    prefix_tokens = max(0, min(int(guard.get("prefix_tokens", 0)), int(closed_video.shape[2])))
    suffix_start = max(0, min(int(guard.get("suffix_start", int(closed_video.shape[2]))), int(closed_video.shape[2])))

    def _drift(a: torch.Tensor, b: torch.Tensor) -> float:
        if a.numel() == 0:
            return 0.0
        aa = a.detach().to(device="cpu", dtype=torch.float32)
        bb = b.detach().to(device="cpu", dtype=torch.float32)
        return float((aa - bb).abs().max().item())

    raw_prefix_drift = (
        _drift(closed_video[:, :, :prefix_tokens], original_video[:, :, :prefix_tokens])
        if prefix_tokens > 0 else 0.0
    )
    raw_suffix_drift = (
        _drift(closed_video[:, :, suffix_start:], original_video[:, :, suffix_start:])
        if suffix_start < int(closed_video.shape[2]) else 0.0
    )
    raw_audio_drift = _drift(closed_audio, original_audio)

    restored_video = closed_video.clone()
    if prefix_tokens > 0:
        restored_video[:, :, :prefix_tokens] = original_video[:, :, :prefix_tokens].to(
            device=restored_video.device, dtype=restored_video.dtype
        )
    if suffix_start < int(restored_video.shape[2]):
        restored_video[:, :, suffix_start:] = original_video[:, :, suffix_start:].to(
            device=restored_video.device, dtype=restored_video.dtype
        )

    restored_audio = original_audio.to(device=closed_audio.device, dtype=closed_audio.dtype).clone()

    import comfy.nested_tensor
    restored_samples = comfy.nested_tensor.NestedTensor((restored_video, restored_audio))

    if prefix_tokens > 0:
        prefix_now = restored_video[:, :, :prefix_tokens].detach().to(device="cpu")
        prefix_truth = original_video[:, :, :prefix_tokens].detach().to(device="cpu")
        if not torch.equal(prefix_now, prefix_truth):
            raise RuntimeError("Terminal Latent Closure v1.1 could not restore its protected entry/prefix exactly.")
    if suffix_start < int(restored_video.shape[2]):
        suffix_now = restored_video[:, :, suffix_start:].detach().to(device="cpu")
        suffix_truth = original_video[:, :, suffix_start:].detach().to(device="cpu")
        if not torch.equal(suffix_now, suffix_truth):
            raise RuntimeError("Terminal Latent Closure v1.1 could not restore the hidden suffix exactly.")
    audio_now = restored_audio.detach().to(device="cpu")
    audio_truth = original_audio.detach().to(device="cpu")
    if not torch.equal(audio_now, audio_truth):
        raise RuntimeError("Terminal Latent Closure v1.1 could not restore H3 audio bit-exactly.")

    result = dict(original_sampled)
    result["samples"] = restored_samples
    if "noise_mask" in original_sampled:
        result["noise_mask"] = original_sampled["noise_mask"]
    else:
        result.pop("noise_mask", None)
    result.pop(CONTRACT_KEY, None)
    result["iamccs_terminal_closure_revision"] = "v1.1-protected-restore"

    active = tuple(guard.get("active_tokens", ()))
    values = tuple(float(v) for v in guard.get("mask_values", ()))
    report = (
        f"active_tokens={active[0] if active else -1}-{active[-1] if active else -1}/{len(active)} | "
        f"window={int(guard.get('window_start_frame', 0))}-{int(guard.get('target_frame', 0))}f | "
        f"mask={values[0] if values else 0.0:.3f}->{values[-1] if values else 0.0:.3f} | "
        f"protected_prefix={prefix_tokens}t | "
        f"restore_drift(prefix={raw_prefix_drift:.3e},suffix={raw_suffix_drift:.3e},audio={raw_audio_drift:.3e}) | "
        "protected=restored_bit_exact | audio=restored_bit_exact | "
        "IAMCCS_TERMINAL_LATENT_CLOSURE_V1_1_PROTECTED_RESTORE"
    )
    return result, report

def mask_curve_for_test(token_count: int, target_frame: int, window_frames: int = 39):
    spans = _token_spans(token_count)
    start = max(0, int(target_frame) - int(window_frames) + 1)
    active = [i for i, (a, b) in enumerate(spans) if a <= target_frame and (b - 1) >= start]
    values = [
        0.5 - 0.5 * math.cos(math.pi * (j / max(1, len(active) - 1)))
        for j in range(len(active))
    ]
    return active, values, start
