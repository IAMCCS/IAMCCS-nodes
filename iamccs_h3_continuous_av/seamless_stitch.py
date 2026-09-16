"""Pure helpers for context-aligned rendered AV stitching.

The generation/latent-continuation path deliberately does not depend on this
module.  It operates only on decoded IMAGE/AUDIO tensors after generation.
"""

from __future__ import annotations

import math
from typing import Any

import torch

try:
    from .latent_math import FPS
except ImportError:  # direct test import
    from latent_math import FPS


def _as_audio_waveform(audio: dict | None):
    if audio is None:
        return None, None
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("audio must be a ComfyUI AUDIO dict with waveform and sample_rate")
    waveform = audio["waveform"]
    if waveform.ndim != 3:
        raise ValueError(f"expected AUDIO waveform [batch,channels,samples], got {tuple(waveform.shape)}")
    return waveform, int(audio["sample_rate"])


def _match_audio_channels(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ca, cb = int(a.shape[1]), int(b.shape[1])
    if ca == cb:
        return a, b
    if ca == 1 and cb == 2:
        return a.repeat(1, 2, 1), b
    if ca == 2 and cb == 1:
        return a, b.repeat(1, 2, 1)
    raise ValueError(f"cannot stitch audio with {ca} and {cb} channels")


LUMINANCE_ANALYSIS_FRAMES = 8


def _rgb_luminance(images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4 or int(images.shape[-1]) < 3:
        raise ValueError("expected IMAGE tensor [frames,height,width,channels>=3]")
    rgb = images[..., :3].float()
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def estimate_luminance_gain(previous_context: torch.Tensor, next_context: torch.Tensor,
                            max_correction_percent: float = 10.0) -> dict[str, float | int | bool]:
    """Estimate a conservative global RGB gain from time-corresponding context frames.

    The measurement intentionally ignores near-black / near-white pixels and
    spatially subsamples the frame.  This makes it less sensitive to clipped
    highlights, letterboxing and tiny decode differences while still detecting
    the global brightness/exposure offset seen at H3 stitch boundaries.
    """
    if previous_context.shape != next_context.shape:
        raise ValueError(
            f"luminance context shapes differ: {tuple(previous_context.shape)} vs {tuple(next_context.shape)}"
        )
    frames = int(previous_context.shape[0])
    if frames <= 0:
        return {
            "luminance_analysis_frames": 0,
            "luminance_measured_gain": 1.0,
            "luminance_applied_gain": 1.0,
            "luminance_clamped": False,
        }

    # 4x spatial subsampling keeps the estimator cheap even for 1344x768 clips.
    prev_luma = _rgb_luminance(previous_context)[:, ::4, ::4]
    next_luma = _rgb_luminance(next_context.to(previous_context.device))[:, ::4, ::4]
    ratios = []
    for i in range(frames):
        a = prev_luma[i]
        b = next_luma[i]
        valid = (a > 0.04) & (a < 0.96) & (b > 0.04) & (b < 0.96)
        valid_count = int(valid.sum().item())
        min_valid = max(4, int(valid.numel() * 0.01))
        if valid_count < min_valid:
            continue
        a_mean = float(a[valid].mean().item())
        b_mean = float(b[valid].mean().item())
        if b_mean > 1e-6:
            ratios.append(a_mean / b_mean)

    if not ratios:
        measured = 1.0
    else:
        measured = float(torch.tensor(ratios, dtype=torch.float32).median().item())

    limit = max(0.0, float(max_correction_percent)) / 100.0
    low = max(0.01, 1.0 - limit)
    high = 1.0 + limit
    applied = min(high, max(low, measured))
    return {
        "luminance_analysis_frames": frames,
        "luminance_measured_gain": measured,
        "luminance_applied_gain": applied,
        "luminance_clamped": abs(applied - measured) > 1e-6,
    }


def apply_rgb_gain(images: torch.Tensor, gain: float) -> torch.Tensor:
    """Apply one global RGB gain while preserving any non-RGB channels."""
    g = float(gain)
    if images.numel() == 0 or abs(g - 1.0) < 1e-7:
        return images
    out = images.clone()
    out[..., :3] = (out[..., :3] * g).clamp(0.0, 1.0)
    return out


def apply_luminance_gain_fade(images: torch.Tensor, gain: float, fade_frames: int = 16,
                              inplace: bool = False) -> tuple[torch.Tensor, int]:
    """Start at ``gain`` and smoothly return to native brightness over N frames.

    ``inplace=True`` is reserved for the streaming Saved Chain path, where the
    decoded clip is disposable after encoding. This avoids cloning a multi-GB
    IMAGE tensor merely to correct a few boundary frames.
    """
    n = min(max(0, int(fade_frames)), int(images.shape[0]))
    if n <= 0 or images.numel() == 0 or abs(float(gain) - 1.0) < 1e-7:
        return images, 0
    out = images if inplace else images.clone()
    if n == 1:
        weights = torch.ones((1,), dtype=out.dtype, device=out.device)
    else:
        t = torch.linspace(0.0, 1.0, n, dtype=out.dtype, device=out.device)
        weights = 0.5 + 0.5 * torch.cos(math.pi * t)
    scales = 1.0 + (float(gain) - 1.0) * weights
    out[:n, ..., :3] = (out[:n, ..., :3] * scales.view(n, 1, 1, 1)).clamp(0.0, 1.0)
    return out, n


def blend_video_overlap(previous_tail: torch.Tensor, next_overlap: torch.Tensor) -> torch.Tensor:
    """Cosine blend two time-corresponding IMAGE tails of equal length."""
    if previous_tail.shape != next_overlap.shape:
        raise ValueError(f"video overlap shapes differ: {tuple(previous_tail.shape)} vs {tuple(next_overlap.shape)}")
    n = int(previous_tail.shape[0])
    if n == 0:
        return previous_tail
    next_overlap = next_overlap.to(previous_tail.device, previous_tail.dtype)
    if n == 1:
        alpha = torch.tensor([0.5], dtype=previous_tail.dtype, device=previous_tail.device)
    else:
        t = torch.linspace(0.0, 1.0, n, dtype=previous_tail.dtype, device=previous_tail.device)
        alpha = 0.5 - 0.5 * torch.cos(math.pi * t)
    alpha = alpha.view(n, 1, 1, 1)
    return previous_tail * (1.0 - alpha) + next_overlap * alpha


def blend_audio_overlap(previous_tail: torch.Tensor, next_overlap: torch.Tensor) -> torch.Tensor:
    """Short equal-gain/linear crossfade for corresponding AUDIO samples."""
    previous_tail, next_overlap = _match_audio_channels(previous_tail, next_overlap)
    if previous_tail.shape != next_overlap.shape:
        raise ValueError(f"audio overlap shapes differ: {tuple(previous_tail.shape)} vs {tuple(next_overlap.shape)}")
    n = int(previous_tail.shape[-1])
    if n == 0:
        return previous_tail
    next_overlap = next_overlap.to(previous_tail.device, previous_tail.dtype)
    if n == 1:
        alpha = torch.tensor([0.5], dtype=previous_tail.dtype, device=previous_tail.device)
    else:
        alpha = torch.linspace(0.0, 1.0, n, dtype=previous_tail.dtype, device=previous_tail.device)
    alpha = alpha.view(1, 1, n)
    return previous_tail * (1.0 - alpha) + next_overlap * alpha


def safe_tail_bridge_plan(handover: dict[str, Any] | None, max_bridge_frames: int = 2) -> dict[str, int | bool]:
    """Return conservative rendered frames that can bridge latent phase quantization loss.

    H3's phase-aligned latent handover may need to stop a few rendered frames
    before the detector's already-safe ideal end. Those 1-3 rendered frames are
    unusable as latent anchors but are still valid pixels. A stitch can therefore
    keep up to ``max_bridge_frames`` of them from the previous clip and skip the
    same number of early video frames in the next clip. Audio timing is unchanged.
    """
    cap = max(0, int(max_bridge_frames))
    out = {
        "safe_tail_bridge_frames": 0,
        "safe_tail_bridge_available_frames": 0,
        "safe_tail_bridge_capped": False,
        "safe_tail_bridge_start_frame": -1,
        "safe_tail_bridge_end_frame": -1,
    }
    if cap <= 0 or not isinstance(handover, dict) or not handover.get("available"):
        return out

    try:
        handover_end = int(handover.get("handover_end_frame", -1))
        ideal_end = int(handover.get("ideal_handover_end_frame", handover_end))
        cutoff_loss = int(handover.get("phase_aligned_cutoff_loss_frames", max(0, ideal_end - handover_end)))
        frame_count = int(handover.get("frame_count", 0) or 0)
    except Exception:
        return out

    if handover_end < 0:
        return out
    # Only frames explicitly inside both the conservative safe range and the
    # recorded phase-alignment loss are eligible. Never borrow from safety margin.
    conservative_gap = max(0, ideal_end - handover_end)
    available = min(conservative_gap, max(0, cutoff_loss))
    if frame_count > 0:
        available = min(available, max(0, frame_count - 1 - handover_end))
    bridge = min(cap, available)
    if bridge <= 0:
        return out

    out.update({
        "safe_tail_bridge_frames": int(bridge),
        "safe_tail_bridge_available_frames": int(available),
        "safe_tail_bridge_capped": bool(bridge < available),
        "safe_tail_bridge_start_frame": int(handover_end + 1),
        "safe_tail_bridge_end_frame": int(handover_end + bridge),
    })
    return out


def extract_safe_tail_bridge_images(full_images: torch.Tensor, handover: dict[str, Any] | None,
                                    max_bridge_frames: int = 2) -> tuple[torch.Tensor, dict[str, int | bool]]:
    """Extract the eligible rendered bridge frames from a full decoded previous clip."""
    if full_images.ndim != 4:
        raise ValueError("safe tail bridge expects IMAGE tensor [frames,height,width,channels]")
    plan = safe_tail_bridge_plan(handover, max_bridge_frames)
    n = int(plan["safe_tail_bridge_frames"])
    if n <= 0:
        return full_images[:0], plan
    start = int(plan["safe_tail_bridge_start_frame"])
    end = int(plan["safe_tail_bridge_end_frame"]) + 1
    if start < 0 or end > int(full_images.shape[0]):
        raise ValueError(
            f"safe tail bridge frames {start}..{end - 1} exceed decoded previous clip length {int(full_images.shape[0])}"
        )
    return full_images[start:end], plan

def frame_trimmed_audio(audio: dict | None, total_frames: int, head_frames: int, tail_frames: int,
                        fps: float = FPS) -> dict | None:
    """Apply the suite's exact rendered-frame -> audio-sample trim convention."""
    if audio is None:
        return None
    waveform, sr = _as_audio_waveform(audio)
    total = int(total_frames)
    head = max(0, int(head_frames))
    tail = max(0, int(tail_frames))
    if head + tail >= total:
        raise ValueError(f"audio trim removes whole clip: head={head}, tail={tail}, total={total}")
    head_samples = int(round(head / float(fps) * sr))
    kept_frames = total - head - tail
    want_samples = int(round(kept_frames / float(fps) * sr))
    if head_samples > waveform.shape[-1]:
        raise ValueError("audio shorter than requested head trim")
    out = waveform[..., head_samples:head_samples + want_samples]
    return {"waveform": out, "sample_rate": sr}


def fit_audio_length(waveform: torch.Tensor, target_samples: int) -> torch.Tensor:
    target = max(0, int(target_samples))
    current = int(waveform.shape[-1])
    if current == target:
        return waveform
    if current > target:
        return waveform[..., :target]
    if target == 0:
        return waveform[..., :0]
    if current == 0:
        shape = list(waveform.shape)
        shape[-1] = target
        return torch.zeros(shape, dtype=waveform.dtype, device=waveform.device)
    # A one/few-sample extension is less audible when holding the final sample
    # than when inserting a zero discontinuity.
    pad = waveform[..., -1:].expand(*waveform.shape[:-1], target - current)
    return torch.cat((waveform, pad), dim=-1)


def context_aligned_video_join(previous_images: torch.Tensor, next_images: torch.Tensor,
                               next_head_context_frames: int, next_tail_trim_frames: int,
                               crossfade_frames: int = 4, luminance_match: bool = False,
                               luminance_fade_frames: int = 16,
                               max_luminance_correction_percent: float = 10.0) -> tuple[torch.Tensor, dict[str, Any]]:
    """Join decoded clips without changing the hard-stitch timeline length.

    The last N frames of the existing timeline are blended against the last N
    *corresponding* frames inside the next clip's reused context head.  When
    enabled, a conservative global luminance gain is estimated from additional
    time-corresponding context frames, applied to the next side of the join, and
    faded back to the next clip's native brightness over a short body window.
    """
    if previous_images.ndim != 4 or next_images.ndim != 4:
        raise ValueError("video join expects IMAGE tensors [frames,height,width,channels]")
    if tuple(previous_images.shape[1:]) != tuple(next_images.shape[1:]):
        raise ValueError(
            f"video geometry mismatch {tuple(previous_images.shape[1:])} vs {tuple(next_images.shape[1:])}"
        )
    head = max(0, int(next_head_context_frames))
    tail = max(0, int(next_tail_trim_frames))
    total_next = int(next_images.shape[0])
    if head + tail >= total_next:
        raise ValueError(f"next clip trim removes whole clip: head={head}, tail={tail}, total={total_next}")

    next_end = total_next - tail if tail else total_next
    next_body = next_images[head:next_end].to(previous_images.device, previous_images.dtype)
    requested = max(0, int(crossfade_frames))
    n = min(requested, head, int(previous_images.shape[0]))

    luma_stats: dict[str, Any] = {
        "luminance_match_enabled": bool(luminance_match),
        "luminance_analysis_frames": 0,
        "luminance_measured_gain": 1.0,
        "luminance_applied_gain": 1.0,
        "luminance_clamped": False,
        "luminance_fade_frames": 0,
    }
    gain = 1.0
    if bool(luminance_match) and head > 0 and int(previous_images.shape[0]) > 0:
        analysis_n = min(LUMINANCE_ANALYSIS_FRAMES, head, int(previous_images.shape[0]))
        measured = estimate_luminance_gain(
            previous_images[-analysis_n:],
            next_images[head - analysis_n:head].to(previous_images.device, previous_images.dtype),
            max_correction_percent=float(max_luminance_correction_percent),
        )
        luma_stats.update(measured)
        gain = float(measured["luminance_applied_gain"])
        next_body, faded = apply_luminance_gain_fade(next_body, gain, int(luminance_fade_frames))
        luma_stats["luminance_fade_frames"] = faded

    if n <= 0:
        out = torch.cat((previous_images, next_body), dim=0)
        return out, {
            "video_crossfade_frames": 0,
            "next_kept_frames": int(next_body.shape[0]),
            **luma_stats,
        }

    previous_prefix = previous_images[:-n]
    previous_tail = previous_images[-n:]
    next_overlap = next_images[head - n:head].to(previous_images.device, previous_images.dtype)
    if bool(luminance_match):
        next_overlap = apply_rgb_gain(next_overlap, gain)
    blended = blend_video_overlap(previous_tail, next_overlap)
    out = torch.cat((previous_prefix, blended, next_body), dim=0)
    return out, {
        "video_crossfade_frames": n,
        "next_kept_frames": int(next_body.shape[0]),
        **luma_stats,
    }


def context_aligned_audio_join(previous_audio: dict | None, next_audio: dict | None,
                               previous_output_frames: int, next_total_frames: int,
                               next_head_context_frames: int, next_tail_trim_frames: int,
                               crossfade_ms: float = 15.0, fps: float = FPS) -> tuple[dict | None, dict[str, int]]:
    """Join AUDIO tensors with a short equal-gain de-click crossfade.

    The crossfade uses the time-corresponding end of the next clip's reused
    audio context. No samples are inserted or removed from the intended video
    timeline beyond normal frame->sample rounding correction.
    """
    if previous_audio is None and next_audio is None:
        return None, {"audio_crossfade_samples": 0}
    if previous_audio is None:
        next_w, sr = _as_audio_waveform(next_audio)
        trimmed = frame_trimmed_audio(next_audio, next_total_frames, next_head_context_frames, next_tail_trim_frames, fps)
        silence_samples = int(round(int(previous_output_frames) / float(fps) * sr))
        silence = torch.zeros((*next_w.shape[:-1], silence_samples), dtype=next_w.dtype, device=next_w.device)
        joined = torch.cat((silence, trimmed["waveform"]), dim=-1)
        target_frames = int(previous_output_frames) + int(next_total_frames) - int(next_head_context_frames) - int(next_tail_trim_frames)
        joined = fit_audio_length(joined, int(round(target_frames / float(fps) * sr)))
        return {"waveform": joined, "sample_rate": sr}, {"audio_crossfade_samples": 0}
    if next_audio is None:
        prev_w, sr = _as_audio_waveform(previous_audio)
        next_kept = int(next_total_frames) - int(next_head_context_frames) - int(next_tail_trim_frames)
        if next_kept < 0:
            raise ValueError("next clip trim is invalid")
        silence_samples = int(round(next_kept / float(fps) * sr))
        silence = torch.zeros((*prev_w.shape[:-1], silence_samples), dtype=prev_w.dtype, device=prev_w.device)
        joined = torch.cat((prev_w, silence), dim=-1)
        target_frames = int(previous_output_frames) + next_kept
        joined = fit_audio_length(joined, int(round(target_frames / float(fps) * sr)))
        return {"waveform": joined, "sample_rate": sr}, {"audio_crossfade_samples": 0}

    prev_w, prev_sr = _as_audio_waveform(previous_audio)
    next_w, next_sr = _as_audio_waveform(next_audio)
    if prev_sr != next_sr:
        raise ValueError(
            f"seamless audio join requires matching sample rates, got {prev_sr} and {next_sr}; use the same Audio VAE"
        )
    prev_w, next_w = _match_audio_channels(prev_w, next_w)
    sr = prev_sr
    head = max(0, int(next_head_context_frames))
    tail = max(0, int(next_tail_trim_frames))
    total_next = int(next_total_frames)
    if head + tail >= total_next:
        raise ValueError(f"next audio trim removes whole clip: head={head}, tail={tail}, total={total_next}")

    head_samples = int(round(head / float(fps) * sr))
    next_kept_frames = total_next - head - tail
    next_body_samples = int(round(next_kept_frames / float(fps) * sr))
    if head_samples > next_w.shape[-1]:
        raise ValueError("next audio is shorter than its declared context head")
    next_body = next_w[..., head_samples:head_samples + next_body_samples]

    requested = max(0, int(round(float(crossfade_ms) / 1000.0 * sr)))
    n = min(requested, head_samples, int(prev_w.shape[-1]))
    if n <= 0:
        joined = torch.cat((prev_w, next_body.to(prev_w.device, prev_w.dtype)), dim=-1)
    else:
        next_overlap = next_w[..., head_samples - n:head_samples].to(prev_w.device, prev_w.dtype)
        prev_prefix = prev_w[..., :-n]
        prev_tail = prev_w[..., -n:]
        blended = blend_audio_overlap(prev_tail, next_overlap)
        joined = torch.cat((prev_prefix, blended, next_body.to(prev_w.device, prev_w.dtype)), dim=-1)

    target_frames = int(previous_output_frames) + next_kept_frames
    target_samples = int(round(target_frames / float(fps) * sr))
    joined = fit_audio_length(joined, target_samples)
    return {"waveform": joined, "sample_rate": sr}, {
        "audio_crossfade_samples": n,
        "audio_crossfade_ms_effective": int(round(n / sr * 1000.0)) if sr else 0,
        "next_kept_frames": next_kept_frames,
    }


def resolve_saved_head_context(metadata: dict[str, Any], clip_index: int,
                               previous_handover: dict[str, Any] | None = None) -> tuple[int, str]:
    """Resolve a saved clip's reused head context with backward compatibility."""
    if int(clip_index) <= 1:
        return 0, "clip-1"
    raw = metadata.get("head_context_frames")
    if raw not in (None, ""):
        return max(0, int(raw)), "saved metadata"
    if isinstance(previous_handover, dict):
        for key in ("phase_aligned_context_frames", "phase_aware_context_frames"):
            if previous_handover.get(key) is not None:
                return max(0, int(previous_handover[key])), f"previous handover {key}"
    raise ValueError(
        f"clip {clip_index} has no saved head_context_frames and the previous clip has no usable context metadata"
    )
