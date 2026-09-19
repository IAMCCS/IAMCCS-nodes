# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pianosequenza 2 Stage progressive spatial sampler for MiniMax H3.

The LongVid timeline geometry is intentionally untouched.  Every IAMCCS macro
chunk remains one H3 sample with the same authored guide times.  The denoising
schedule alone is split spatially: broad motion is solved on a half-resolution
H3 video latent, a learned H3 3D latent lift restores the target spatial grid,
and the SAME sigma schedule is completed at target resolution.

The final LongVid chunk may additionally carry Pianosequenza DRIFT.  Existing
IAMCCS AudioCon/noise-mask ownership is preserved.
"""
from __future__ import annotations

import inspect
import logging
import math
import time
from typing import Any

import folder_paths
import torch
import torch.nn.functional as F

try:
    import latent_preview
except Exception:
    latent_preview = None


LOG = logging.getLogger("IAMCCS.MiniMaxH3.Pianosequenza2Stage")
LOW_CARRY_KEY = "iamccs_pianosequenza_2stage_low_carry"
PREVIOUS_LOW_TAIL_KEY = "iamccs_pianosequenza_2stage_previous_low_tail"
DRIFT_STATE_KEY = "iamccs_pianosequenza_v2_drift_control"
SEAM_DC_CLAMP = 0.05


def _result_item(value: Any, index: int = 0):
    try:
        return value[index]
    except (IndexError, KeyError, TypeError):
        result = getattr(value, "result", None)
        if result is None:
            raise
        return result[index]


def _streams(samples):
    if bool(getattr(samples, "is_nested", False)):
        return list(samples.unbind()), True
    return [samples], False


def _pack(streams, nested):
    if nested:
        import comfy.nested_tensor
        return comfy.nested_tensor.NestedTensor(tuple(streams))
    return streams[0]


def _move_stream_bundle(samples, device):
    """Move every stream of a Tensor/NestedTensor bundle to one device.

    ComfyUI packs MiniMax H3 VIDEO+AUDIO streams with torch.cat() immediately
    before sampling.  A progressive handoff may otherwise leave the video on
    the intermediate/upscaler device while the audio remains on the previous
    sampler device, which makes that pack fail before the high-resolution pass.
    Keep dtype unchanged and move only the relatively small latent/mask tensors.
    """
    if samples is None:
        return None
    streams, nested = _streams(samples)
    moved = []
    for stream in streams:
        if not torch.is_tensor(stream):
            raise TypeError("Pianosequenza 2 Stage expected tensor streams during device synchronization")
        moved.append(stream.to(device=device, non_blocking=True))
    return _pack(moved, nested)


def _bundle_devices(samples):
    if samples is None:
        return ()
    streams, _nested = _streams(samples)
    return tuple(str(stream.device) for stream in streams if torch.is_tensor(stream))


def _resize_keyframes(conditioning, height: int, width: int):
    """Resize H3 keyframe latents spatially only; never interpolate time."""
    out = []
    for tensor, metadata in conditioning:
        keyframes = metadata.get("minimax_keyframes") if isinstance(metadata, dict) else None
        if keyframes is None:
            out.append((tensor, metadata))
            continue
        metadata = metadata.copy()
        resized_keyframes = []
        for keyframe in keyframes:
            keyframe = dict(keyframe)
            latent = keyframe.get("latent")
            if torch.is_tensor(latent) and latent.ndim >= 4 and (
                int(latent.shape[-2]) != int(height) or int(latent.shape[-1]) != int(width)
            ):
                if latent.ndim == 5:
                    batch, channels, frames = latent.shape[:3]
                    resized = F.interpolate(
                        latent.float().permute(0, 2, 1, 3, 4).reshape(
                            batch * frames, channels, latent.shape[-2], latent.shape[-1]
                        ),
                        size=(int(height), int(width)),
                        mode="bilinear",
                        align_corners=False,
                    )
                    resized = resized.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)
                else:
                    resized = F.interpolate(
                        latent.float(), size=(int(height), int(width)), mode="bilinear", align_corners=False
                    )
                source_mean = latent.float().mean(dim=(-2, -1), keepdim=True)
                resized_mean = resized.mean(dim=(-2, -1), keepdim=True)
                keyframe["latent"] = (resized + (source_mean - resized_mean)).to(latent)
            resized_keyframes.append(keyframe)
        metadata["minimax_keyframes"] = resized_keyframes
        out.append((tensor, metadata))
    return out


def _validate_schedule(sigmas: torch.Tensor, transition_step: int):
    if not torch.is_tensor(sigmas) or sigmas.ndim != 1 or not sigmas.is_floating_point():
        raise ValueError("Pianosequenza 2 Stage requires a 1D floating-point sigma schedule")
    if int(sigmas.numel()) < 3:
        raise ValueError("Pianosequenza 2 Stage requires at least two denoising steps")
    if not bool(torch.isfinite(sigmas).all()) or bool((sigmas < 0).any()):
        raise ValueError("Pianosequenza 2 Stage sigmas must be finite and non-negative")
    total_steps = int(sigmas.numel()) - 1
    if not 1 <= int(transition_step) <= total_steps - 1:
        raise ValueError(
            f"Pianosequenza 2 Stage transition_step={transition_step} is invalid for {total_steps} steps"
        )
    if bool((sigmas[1:] > sigmas[:-1]).any()):
        raise ValueError("Pianosequenza 2 Stage sigmas must be non-increasing")
    if bool((sigmas[:-1] <= 0).any()):
        raise ValueError("Pianosequenza 2 Stage allows sigma=0 only at the final schedule item")


def _normalize_masks(latent: dict[str, Any], streams: list[torch.Tensor]):
    raw = latent.get("noise_mask")
    if raw is None:
        return None
    if not bool(getattr(raw, "is_nested", False)):
        raise ValueError("Pianosequenza 2 Stage H3 noise_mask must be nested VIDEO+AUDIO")
    masks = list(raw.unbind())
    if len(masks) != len(streams):
        raise ValueError("Pianosequenza 2 Stage mask stream count does not match AV latent")

    video = streams[0]
    if video.ndim != 5:
        raise ValueError("Pianosequenza 2 Stage requires a 5D H3 video latent")
    b, _c, t, h, w = video.shape
    video_mask = masks[0]
    if video_mask.ndim == 4:
        video_mask = video_mask[:, :, None]
    if video_mask.ndim != 5 or video_mask.shape[1] != 1:
        raise ValueError("Pianosequenza 2 Stage video mask must be [B,1,T|1,H|1,W|1]")
    if video_mask.shape[0] not in (1, b):
        raise ValueError("Pianosequenza 2 Stage video mask batch mismatch")
    if video_mask.shape[2] not in (1, t):
        raise ValueError("Pianosequenza 2 Stage video mask temporal mismatch")
    # H3 masks are often compact 1x1 spatial masks.  Expand only at the point
    # where the sampler needs the actual active spatial grid.
    if video_mask.shape[2] == 1 and t > 1:
        video_mask = video_mask.expand(video_mask.shape[0], 1, t, video_mask.shape[-2], video_mask.shape[-1])
    if video_mask.shape[-2:] != (h, w):
        if video_mask.shape[-2:] == (1, 1):
            video_mask = video_mask.expand(video_mask.shape[0], 1, t, h, w)
        else:
            video_mask = F.interpolate(
                video_mask.reshape(video_mask.shape[0] * t, 1, *video_mask.shape[-2:]).float(),
                size=(h, w), mode="bilinear", align_corners=False,
            ).reshape(video_mask.shape[0], 1, t, h, w)
    video_mask = video_mask.float().clamp(0.0, 1.0)

    auxiliary = []
    for stream, mask in zip(streams[1:], masks[1:]):
        if not torch.is_tensor(mask) or mask.ndim != stream.ndim:
            raise ValueError("Pianosequenza 2 Stage audio mask must match the H3 audio stream rank")
        if mask.shape[0] not in (1, stream.shape[0]) or mask.shape[1] not in (1, stream.shape[1]):
            raise ValueError("Pianosequenza 2 Stage audio mask batch/channel mismatch")
        if any(ms not in (1, ss) for ms, ss in zip(mask.shape[2:], stream.shape[2:])):
            raise ValueError("Pianosequenza 2 Stage audio mask is not broadcast-compatible")
        auxiliary.append(mask.float().clamp(0.0, 1.0))
    return video_mask, auxiliary


def _euler_step(state, denoised, sigma, sigma_next):
    step = ((sigma_next - sigma) / sigma).to(device=state.device, dtype=state.dtype)
    return state + (state - denoised.to(state)) * step


def _match_prefix_dc(lifted, previous, prefix_steps: int, clamp: float = SEAM_DC_CLAMP):
    if lifted.ndim != 5 or previous.ndim != 5:
        return lifted, None
    if lifted.shape[:2] != previous.shape[:2] or lifted.shape[-2:] != previous.shape[-2:]:
        return lifted, None
    count = min(max(0, int(prefix_steps)), int(lifted.shape[2]), int(previous.shape[2]))
    if count < 1:
        return lifted, None
    deltas = []
    for channel in range(int(lifted.shape[1])):
        difference = (
            lifted[:, channel, :count].float()
            - previous[:, channel, :count].to(lifted.device).float()
        )
        deltas.append(difference.reshape(-1).median())
    dc = torch.stack(deltas).clamp(-float(clamp), float(clamp))
    dc = dc.to(device=lifted.device, dtype=lifted.dtype).view(1, -1, 1, 1, 1)
    corrected = lifted.clone()
    corrected[:, :, :count].sub_(dc)
    return corrected, dc


def _selected_upscaler(shotplan: dict[str, Any]) -> tuple[str, str, str]:
    upscale = shotplan.get("upscale_settings") if isinstance(shotplan.get("upscale_settings"), dict) else {}
    settings = upscale.get("h3_latent_upres") if isinstance(upscale.get("h3_latent_upres"), dict) else {}
    model_name = str(settings.get("model_name", "") or "").strip()
    if not model_name:
        candidates = [
            name for name in folder_paths.get_filename_list("latent_upscale_models")
            if all(token in name.lower() for token in ("minimax", "h3", "upscaler", "3d"))
        ]
        candidates.sort(key=lambda name: (0 if "fp16" in name.lower() else 1, name.lower()))
        model_name = candidates[0] if candidates else ""
    if not model_name or not folder_paths.get_full_path("latent_upscale_models", model_name):
        raise ValueError(
            "Pianosequenza 2 Stage requires an installed MiniMax H3 3D latent upscaler selected in IAMCCS H3 Settings"
        )
    precision = str(settings.get("precision", "fp16") or "fp16").lower()
    if precision not in {"fp16", "bf16", "fp32"}:
        precision = "fp16"
    device = str(settings.get("device", "cuda") or "cuda").lower()
    if device not in {"cuda", "cpu"}:
        device = "cuda"
    return model_name, precision, device


def _learned_lift(video_latent: torch.Tensor, shotplan: dict[str, Any], width: int, height: int):
    import nodes

    klass = nodes.NODE_CLASS_MAPPINGS.get("MinimaxH3LatentUpscaler3D")
    if klass is None:
        raise RuntimeError(
            "Pianosequenza 2 Stage requires the installed MinimaxH3LatentUpscaler3D node"
        )
    model_name, precision, device = _selected_upscaler(shotplan)
    source = {"samples": video_latent}
    dynamic_mode = {"mode": "target dimensions", "width": int(width), "height": int(height)}
    # Pianosequenza progressive ownership: the H3 transformer remains under
    # ComfyUI model management across LOW -> lift -> HIGH.  Only the small
    # latent-upscaler is allowed to unload itself after the lift.  The previous
    # IAMCCS path called unload_all_models() around this node, which forced a
    # complete H3 reload immediately before the first high-resolution forward.
    execute_kwargs = {
        "latent": source,
        "model_name": model_name,
        "mode": dynamic_mode,
        "align": 32,
        "device": device,
        "precision": precision,
    }
    try:
        execute_parameters = inspect.signature(klass.execute).parameters
    except (TypeError, ValueError):
        execute_parameters = {}
    temporal_chunking = "enable_temporal_chunking" in execute_parameters
    isolated_unload = "force_unload" in execute_parameters
    if temporal_chunking:
        execute_kwargs["enable_temporal_chunking"] = True
    if isolated_unload:
        execute_kwargs["force_unload"] = True
    LOG.info(
        "Pianosequenza MEMORY-SAFE LIFT | H3_global_unload=disabled "
        "| temporal_chunking=%s | upscaler_force_unload=%s",
        temporal_chunking, isolated_unload,
    )
    result = klass.execute(**execute_kwargs)
    lifted = _result_item(result, 0)
    if not isinstance(lifted, dict) or not torch.is_tensor(lifted.get("samples")):
        raise RuntimeError("Pianosequenza 2 Stage latent upscaler returned no video latent")
    return lifted["samples"], model_name


def _normalize_stage_count(value: str) -> int:
    text = str(value or "2_stage").strip().lower()
    return 3 if text.startswith("3") else 2


def _manual_step_allocation(total_steps: int, stage_count: int, settings: dict[str, Any]) -> list[int]:
    values = [
        int(settings.get("manual_stage1_steps", 0) or 0),
        int(settings.get("manual_stage2_steps", 0) or 0),
        int(settings.get("manual_stage3_steps", 0) or 0),
    ][:stage_count]
    if sum(values) != int(total_steps):
        raise ValueError(
            f"Pianosequenza manual stage split must sum to total steps ({total_steps}); received {values}"
        )
    if any(v < 1 for v in values):
        raise ValueError("Pianosequenza manual stage split requires at least one step per active stage")
    return values


def _allocate_by_weights(total_steps: int, weights: list[float]) -> list[int]:
    count = len(weights)
    if total_steps < count:
        raise ValueError(f"Pianosequenza needs at least {count} denoising steps for {count} sampled stages")
    weight_sum = float(sum(weights))
    raw = [float(total_steps) * (float(w) / weight_sum) for w in weights]
    allocated = [max(1, int(math.floor(value))) for value in raw]
    while sum(allocated) < total_steps:
        candidates = sorted(
            ((raw[i] - math.floor(raw[i]), i) for i in range(count)),
            key=lambda item: (item[0], -item[1]),
            reverse=True,
        )
        for _fraction, idx in candidates:
            if sum(allocated) >= total_steps:
                break
            allocated[idx] += 1
    while sum(allocated) > total_steps:
        candidates = sorted(
            ((allocated[i] - raw[i], i) for i in range(count) if allocated[i] > 1),
            reverse=True,
        )
        if not candidates:
            raise ValueError("Pianosequenza could not normalize the stage allocation")
        allocated[candidates[0][1]] -= 1
    return allocated


def _stage_weights(stage_count: int, split_mode: str, profile: str, total_steps: int) -> list[float]:
    split_mode = str(split_mode or "auto").strip().lower()
    profile = str(profile or "balanced").strip().lower()
    if stage_count == 2:
        ratio_map = {
            "ratio_75_25": [0.75, 0.25],
            "ratio_80_20": [0.80, 0.20],
            "ratio_67_33": [0.67, 0.33],
        }
        profile_map = {
            "balanced": [0.75, 0.25],
            "low_authority": [0.80, 0.20],
            "high_refine": [0.67, 0.33],
            "fast_safe": [0.84, 0.16] if total_steps <= 6 else [0.80, 0.20],
            "pdd_safe": [0.75, 0.25],
            "toetao_safe": [0.67, 0.33] if total_steps <= 3 else [0.75, 0.25],
        }
        return ratio_map.get(split_mode) or profile_map.get(profile) or profile_map["balanced"]
    ratio_map = {
        "ratio_75_25": [0.55, 0.25, 0.20],
        "ratio_80_20": [0.60, 0.25, 0.15],
        "ratio_67_33": [0.45, 0.30, 0.25],
    }
    profile_map = {
        "balanced": [0.55, 0.25, 0.20],
        "low_authority": [0.60, 0.25, 0.15],
        "high_refine": [0.45, 0.30, 0.25],
        "fast_safe": [0.60, 0.25, 0.15],
        "pdd_safe": [0.55, 0.25, 0.20],
        "toetao_safe": [0.34, 0.33, 0.33] if total_steps <= 3 else [0.50, 0.30, 0.20],
    }
    return ratio_map.get(split_mode) or profile_map.get(profile) or profile_map["balanced"]


def _validate_sigma_slice(sigmas: torch.Tensor) -> None:
    if not torch.is_tensor(sigmas) or sigmas.ndim != 1 or not sigmas.is_floating_point():
        raise ValueError("Pianosequenza stage requires a 1D floating-point sigma schedule")
    if int(sigmas.numel()) < 2:
        raise ValueError("Pianosequenza stage requires at least one denoising step")
    if not bool(torch.isfinite(sigmas).all()) or bool((sigmas < 0).any()):
        raise ValueError("Pianosequenza sigmas must be finite and non-negative")
    if bool((sigmas[1:] > sigmas[:-1]).any()):
        raise ValueError("Pianosequenza sigmas must be non-increasing")


def _profile_contract(settings: dict[str, Any]) -> tuple[str, int, int, bool]:
    profile = str(settings.get("stage_profile", "") or "").strip().lower()
    if not profile:
        legacy_count = _normalize_stage_count(settings.get("stage_count", "2_stage"))
        profile = "3_stage_full" if legacy_count == 3 else "2_stage_full"
    contracts = {
        "2_stage_full": (2, 2, False),
        "2_stage_safe_delivery": (2, 1, True),
        "3_stage_safe_mid": (3, 2, True),
        "3_stage_full": (3, 3, False),
    }
    if profile not in contracts:
        raise ValueError(f"Unknown Pianosequenza stage profile: {profile}")
    spatial_count, sampled_count, final_lift_only = contracts[profile]
    return profile, spatial_count, sampled_count, final_lift_only


def _stage_step_allocation(
    sigmas: torch.Tensor,
    settings: dict[str, Any],
    acceleration_mode: str,
    sampled_stage_count: int,
) -> tuple[list[int], str]:
    total_steps = int(sigmas.numel()) - 1
    if total_steps < sampled_stage_count:
        raise ValueError(
            f"Pianosequenza requires at least {sampled_stage_count} denoising steps for this profile; received {total_steps}"
        )
    if sampled_stage_count == 1:
        return [total_steps], "safe_delivery:all_steps_low"

    split_mode = str(settings.get("split_mode", "auto") or "auto")
    profile = str(settings.get("auto_profile", "balanced") or "balanced")
    if split_mode == "manual":
        manual_settings = dict(settings)
        if sampled_stage_count == 2:
            # Safe 3-stage uses LOW+MID H3; the HIGH spatial stage is lift-only.
            manual_settings["manual_stage3_steps"] = 0
        allocation = _manual_step_allocation(total_steps, sampled_stage_count, manual_settings)
        reason = "manual"
    else:
        weights = _stage_weights(sampled_stage_count, split_mode, profile, total_steps)
        allocation = _allocate_by_weights(total_steps, weights)
        # For compact 2-stage schedules, keep at least two refinement NFEs once
        # six or more total steps are available. This preserves IAMCCS 4+2 at
        # FastH3-6 and 6+2 at 8-step schedules while remaining ratio-driven.
        if sampled_stage_count == 2 and total_steps >= 6 and allocation[1] < 2:
            allocation[0] -= 1
            allocation[1] += 1
        reason = f"{split_mode}:{profile}" if split_mode != "auto" else f"auto:{profile}"
    return allocation, reason


def _build_stage_layout(
    *,
    stage_count: int,
    low_latent_hw: tuple[int, int],
    high_latent_hw: tuple[int, int],
    low_pixels: tuple[int, int],
    high_pixels: tuple[int, int],
) -> list[dict[str, Any]]:
    low_w, low_h = low_latent_hw
    high_w, high_h = high_latent_hw
    low_px_w, low_px_h = low_pixels
    high_px_w, high_px_h = high_pixels
    if stage_count == 2:
        return [
            {"name": "LOW", "latent_w": low_w, "latent_h": low_h, "pixel_w": low_px_w, "pixel_h": low_px_h},
            {"name": "HIGH", "latent_w": high_w, "latent_h": high_h, "pixel_w": high_px_w, "pixel_h": high_px_h},
        ]
    mid_w = max(low_w + 1, int(round((low_w + high_w) / 2.0)))
    mid_h = max(low_h + 1, int(round((low_h + high_h) / 2.0)))
    mid_px_w = max(low_px_w + 1, int(round((low_px_w + high_px_w) / 2.0)))
    mid_px_h = max(low_px_h + 1, int(round((low_px_h + high_px_h) / 2.0)))
    return [
        {"name": "LOW", "latent_w": low_w, "latent_h": low_h, "pixel_w": low_px_w, "pixel_h": low_px_h},
        {"name": "MID", "latent_w": mid_w, "latent_h": mid_h, "pixel_w": mid_px_w, "pixel_h": mid_px_h},
        {"name": "HIGH", "latent_w": high_w, "latent_h": high_h, "pixel_w": high_px_w, "pixel_h": high_px_h},
    ]


def _resize_video_mask(mask: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
    return F.interpolate(
        mask.reshape(int(mask.shape[0]) * int(t), 1, int(mask.shape[-2]), int(mask.shape[-1])).float(),
        size=(int(h), int(w)), mode="bilinear", align_corners=False,
    ).reshape(int(mask.shape[0]), 1, int(t), int(h), int(w)).clamp(0.0, 1.0)


def _make_preview_callback(model, steps: int, stage_name: str):
    preview_callback = None
    preview_warned = False
    if latent_preview is not None:
        try:
            preview_callback = latent_preview.prepare_callback(model, steps)
        except Exception as exc:
            LOG.warning("Pianosequenza %s preview unavailable: %s", stage_name, exc)

    calls = {"n": 0}

    def callback(_step, x0, x, _total):
        nonlocal preview_warned
        if preview_callback is not None:
            try:
                preview_callback(_step, x0, x, _total)
            except Exception as exc:
                if not preview_warned:
                    LOG.warning("Pianosequenza %s preview callback disabled after error: %s", stage_name, exc)
                    preview_warned = True
        calls["n"] += 1
        return x0, x

    return calls, callback

def sample_pianosequenza_2stage(
    *,
    model,
    conditioning,
    sigmas: torch.Tensor,
    latent: dict[str, Any],
    seed: int,
    shotplan: dict[str, Any],
    disable_pbar: bool,
) -> tuple[dict[str, Any], str]:
    """IAMCCS Pianosequenza progressive spatial sampler with safe/full 2/3-stage profiles."""
    import comfy.model_management
    import comfy.model_sampling
    import comfy.sample
    import comfy.samplers

    stage_settings = shotplan.get("pianosequenza_2stage_settings") if isinstance(shotplan.get("pianosequenza_2stage_settings"), dict) else {}
    linked_resolution = bool(stage_settings.get("resolution_link_enabled", False))
    acceleration_mode = str(shotplan.get("acceleration", "native") or "native").strip().lower()
    stage_profile, spatial_stage_count, sampled_stage_count, final_lift_only = _profile_contract(stage_settings)
    stage_steps, allocator_reason = _stage_step_allocation(
        sigmas, stage_settings, acceleration_mode, sampled_stage_count
    )
    total_steps = int(sigmas.numel()) - 1
    if sum(stage_steps) != total_steps:
        raise RuntimeError(f"Pianosequenza allocator mismatch: total={total_steps}, stages={stage_steps}")
    LOG.info(
        "Pianosequenza STEP ALLOCATOR | stage_profile=%s | acceleration=%s | total=%d | sampled_stages=%d | steps=%s | final_lift_only=%s | reason=%s",
        stage_profile, acceleration_mode, total_steps, sampled_stage_count, stage_steps, final_lift_only, allocator_reason,
    )
    sampler = comfy.samplers.sampler_object("euler")

    model_sampling = model.get_model_object("model_sampling")
    if not isinstance(model_sampling, comfy.model_sampling.CONST):
        raise ValueError("Pianosequenza requires MiniMax H3 rectified-flow sampling")

    prepared = latent.copy()
    samples = comfy.sample.fix_empty_latent_channels(
        model,
        prepared["samples"],
        prepared.get("downscale_ratio_spacial"),
        prepared.get("downscale_ratio_temporal"),
    )
    streams, nested = _streams(samples)
    if not nested or len(streams) != 2:
        raise ValueError("Pianosequenza requires native MiniMax H3 VIDEO+AUDIO latent")
    video, audio = streams
    if video.ndim != 5 or int(video.shape[1]) != 24 or audio.ndim != 4:
        raise ValueError("Pianosequenza received invalid H3 AV latent geometry")

    b, c, t, full_h, full_w = video.shape
    target_width_px = int(stage_settings.get("high_width", shotplan.get("width", 0)) or 0)
    target_height_px = int(stage_settings.get("high_height", shotplan.get("height", 0)) or 0)
    low_width_px = int(stage_settings.get("low_width", 0) or 0)
    low_height_px = int(stage_settings.get("low_height", 0) or 0)
    if linked_resolution and target_width_px > 0 and target_height_px > 0 and low_width_px > 0 and low_height_px > 0:
        scale_w = float(low_width_px) / float(target_width_px)
        scale_h = float(low_height_px) / float(target_height_px)
        if not (0.0 < scale_w < 1.0 and 0.0 < scale_h < 1.0):
            raise ValueError(
                f"Pianosequenza linked resolution is invalid: low={low_width_px}x{low_height_px}, high={target_width_px}x{target_height_px}"
            )
        low_h = max(2, round(int(full_h) * scale_h / 2) * 2)
        low_w = max(2, round(int(full_w) * scale_w / 2) * 2)
    else:
        low_h = max(2, round(int(full_h) * 0.5 / 2) * 2)
        low_w = max(2, round(int(full_w) * 0.5 / 2) * 2)
        low_width_px = max(1, int(round(float(shotplan.get("width", 0) or 0) * 0.5)))
        low_height_px = max(1, int(round(float(shotplan.get("height", 0) or 0) * 0.5)))
        target_width_px = int(shotplan.get("width", 0) or 0)
        target_height_px = int(shotplan.get("height", 0) or 0)

    stage_layout = _build_stage_layout(
        stage_count=spatial_stage_count,
        low_latent_hw=(int(low_w), int(low_h)),
        high_latent_hw=(int(full_w), int(full_h)),
        low_pixels=(int(low_width_px), int(low_height_px)),
        high_pixels=(int(target_width_px), int(target_height_px)),
    )
    LOG.info(
        "Pianosequenza RESOLUTION | profile=%s | linked=%s | stages=%s",
        stage_profile, linked_resolution,
        ", ".join(
            f"{item['name']}={item['pixel_w']}x{item['pixel_h']} ({item['latent_w']}x{item['latent_h']} latent)"
            for item in stage_layout
        ),
    )

    working_device = comfy.model_management.intermediate_device()
    source_video = video.to(working_device)
    audio_streams = [audio.to(working_device)]
    stage_videos = [
        F.interpolate(
            source_video.float(),
            size=(int(t), int(spec["latent_h"]), int(spec["latent_w"])),
            mode="trilinear",
            align_corners=False,
        ).to(dtype=source_video.dtype)
        for spec in stage_layout
    ]

    masks = _normalize_masks(prepared, streams)
    stage_masks = [None] * spatial_stage_count
    auxiliary_masks = []
    if masks is not None:
        full_video_mask, auxiliary_masks = masks
        full_video_mask = full_video_mask.to(working_device)
        auxiliary_masks = [mask.to(working_device) for mask in auxiliary_masks]
        for idx, spec in enumerate(stage_layout):
            stage_masks[idx] = _resize_video_mask(
                full_video_mask, int(t), int(spec["latent_h"]), int(spec["latent_w"])
            ).to(working_device)

    drift_state = getattr(model, "model_options", {}).get(DRIFT_STATE_KEY)
    drift_continuation = callable(getattr(drift_state, "configure_pianosequenza_stage", None))

    previous_low_tail = prepared.get(PREVIOUS_LOW_TAIL_KEY)
    low_tail_tokens = 0
    if torch.is_tensor(previous_low_tail):
        if previous_low_tail.ndim != 5:
            raise ValueError("Pianosequenza previous low-resolution tail is not 5D")
        if tuple(previous_low_tail.shape[:2]) != tuple(stage_videos[0].shape[:2]) or tuple(previous_low_tail.shape[-2:]) != tuple(stage_videos[0].shape[-2:]):
            raise ValueError("Pianosequenza previous low-resolution tail does not match the current low grid")
        low_tail_tokens = int(previous_low_tail.shape[2])
        if low_tail_tokens < 1 or low_tail_tokens >= int(stage_videos[0].shape[2]):
            raise ValueError("Pianosequenza previous low-resolution tail has invalid temporal extent")
        stage_videos[0][:, :, :low_tail_tokens] = previous_low_tail.to(
            device=stage_videos[0].device, dtype=stage_videos[0].dtype
        )

    latent_format = model.get_model_object("latent_format")
    # Pianosequenza keeps one H3 patcher alive through the progressive
    # resolution transition.  Multiple clones are unnecessary here because
    # IAMCCS passes masks to the native sampler and configures the shared drift
    # state explicitly per stage.  Reusing the patcher also avoids a second
    # model-lifecycle owner at HIGH resolution.
    sampled_models = [model for _ in range(sampled_stage_count)]
    conditioning_by_stage = [
        _resize_keyframes(conditioning, stage_layout[i]["latent_h"], stage_layout[i]["latent_w"])
        for i in range(sampled_stage_count)
    ]

    current_latent = _pack([stage_videos[0]] + audio_streams, nested)
    current_noise_mask = _pack([stage_masks[0]] + auxiliary_masks, nested) if masks is not None else None
    current_noise = comfy.sample.prepare_noise(current_latent, int(seed), prepared.get("batch_index", None))
    low_carry = None
    seam_dc_max = 0.0
    report_parts = []
    step_cursor = 0
    final_out = None
    upscaler_name = "none"
    last_eval = {}

    for sampled_index in range(sampled_stage_count):
        spatial_index = sampled_index
        spec = stage_layout[spatial_index]
        step_count = int(stage_steps[sampled_index])
        stage_name = str(spec["name"])
        stage_model = sampled_models[sampled_index]
        stage_sigmas = sigmas[step_cursor : step_cursor + step_count + 1]
        _validate_sigma_slice(stage_sigmas)
        preview_calls, preview_forward = _make_preview_callback(stage_model, step_count, stage_name)
        transition = {}
        last_eval = {}
        has_next_sampled_stage = sampled_index + 1 < sampled_stage_count

        def stage_callback(_step, x0, x, _total, *, _budget=step_count, _needs_transition=has_next_sampled_stage):
            preview_forward(_step, x0, x, _total)
            if preview_calls["n"] > _budget:
                raise RuntimeError(f"Pianosequenza {stage_name} sampler exceeded its NFE budget")
            last_eval["state"] = x
            last_eval["x0"] = x0
            if _needs_transition and preview_calls["n"] == _budget:
                transition["state"] = x
                transition["x0"] = x0

        if drift_continuation and masks is not None:
            drift_state.configure_pianosequenza_stage(
                (b, c, t, int(spec["latent_h"]), int(spec["latent_w"])),
                stage_masks[spatial_index],
                auxiliary_masks[0],
                hard_lock=False,
            )

        LOG.info(
            "Pianosequenza %s START | profile=%s | steps=%d | latent=%dx%d | pixels=%dx%d | preview=forwarded",
            stage_name, stage_profile, step_count, int(spec["latent_w"]), int(spec["latent_h"]), int(spec["pixel_w"]), int(spec["pixel_h"]),
        )
        t0 = time.perf_counter()
        out = comfy.samplers.sample(
            stage_model,
            current_noise,
            conditioning_by_stage[sampled_index],
            conditioning_by_stage[sampled_index],
            1.0,
            stage_model.load_device,
            sampler,
            stage_sigmas,
            stage_model.model_options,
            latent_image=current_latent,
            denoise_mask=current_noise_mask,
            callback=stage_callback,
            disable_pbar=bool(disable_pbar),
            seed=int(seed),
        )
        LOG.info(
            "Pianosequenza %s COMPLETE | steps=%d | elapsed=%.1fs | preview_calls=%d",
            stage_name, step_count, time.perf_counter() - t0, preview_calls["n"],
        )
        if preview_calls["n"] != step_count or "x0" not in last_eval:
            raise RuntimeError(
                f"Pianosequenza {stage_name} expected {step_count} evaluations, received {preview_calls['n']}"
            )
        report_parts.append(f"{stage_name.lower()}={step_count}")

        last_x0_streams, _ = _streams(last_eval["x0"])
        if sampled_index == 0:
            low_carry = latent_format.process_out(last_x0_streams[0].float()).detach().to(
                device=working_device, dtype=comfy.model_management.intermediate_dtype()
            ).clone()

        if not has_next_sampled_stage:
            if final_lift_only:
                final_spec = stage_layout[-1]
                z_clean_external = latent_format.process_out(last_x0_streams[0].float()).to(working_device)
                LOG.info(
                    "Pianosequenza FINAL LIFT ONLY START | profile=%s | %s->%s | target=%dx%d | no HIGH H3 forward",
                    stage_profile, stage_name, final_spec["name"], int(final_spec["pixel_w"]), int(final_spec["pixel_h"]),
                )
                lift_t0 = time.perf_counter()
                lifted_external, upscaler_name = _learned_lift(
                    z_clean_external, shotplan, int(final_spec["pixel_w"]), int(final_spec["pixel_h"])
                )
                # Never unload H3 globally here.  The upscaler is asked to
                # release only itself (when its API supports force_unload), and
                # allocator cleanup is enough before delivery.
                comfy.model_management.soft_empty_cache()
                LOG.info(
                    "Pianosequenza FINAL LIFT ONLY COMPLETE | elapsed=%.1fs | final_high_h3=skipped",
                    time.perf_counter() - lift_t0,
                )
                lifted_external = lifted_external.to(working_device)
                if tuple(lifted_external.shape[-2:]) != (int(full_h), int(full_w)):
                    raise RuntimeError(
                        f"Pianosequenza final lift returned wrong canvas {tuple(lifted_external.shape[-2:])}; expected {(int(full_h), int(full_w))}"
                    )
                if masks is not None and not drift_continuation:
                    high_mask = stage_masks[-1].to(device=lifted_external.device, dtype=lifted_external.dtype)
                    lifted_external = torch.where(
                        high_mask == 0,
                        source_video.to(lifted_external),
                        lifted_external * high_mask + source_video.to(lifted_external) * (1.0 - high_mask),
                    )
                if drift_continuation:
                    prefix_steps = min(
                        int(getattr(drift_state, "prefix_steps", 0) or 0),
                        int(lifted_external.shape[2]),
                        int(source_video.shape[2]),
                    )
                    lifted_external, seam_dc = _match_prefix_dc(lifted_external, source_video, prefix_steps)
                    if seam_dc is not None:
                        seam_dc_max = float(seam_dc.float().abs().max().item())
                out_streams, out_nested = _streams(out)
                final_out = _pack(
                    [lifted_external.to(device=out_streams[0].device, dtype=out_streams[0].dtype)] + out_streams[1:],
                    out_nested,
                )
                report_parts.append("high=lift_only")
            else:
                final_out = out
            break

        if "state" not in transition or "x0" not in transition:
            raise RuntimeError(f"Pianosequenza {stage_name} transition state was not captured")
        trans_streams, _ = _streams(transition["state"])
        x0_streams, _ = _streams(transition["x0"])
        sigma_k = stage_sigmas[-2]
        sigma_next = stage_sigmas[-1]
        audio_next = [
            _euler_step(state.to(working_device), denoised.to(working_device), sigma_k, sigma_next)
            for state, denoised in zip(trans_streams[1:], x0_streams[1:])
        ]

        z0_external = latent_format.process_out(x0_streams[0].float()).to(working_device)
        next_spec = stage_layout[spatial_index + 1]
        LOG.info(
            "Pianosequenza %s->%s LIFT START | target=%dx%d",
            stage_name, next_spec["name"], int(next_spec["pixel_w"]), int(next_spec["pixel_h"]),
        )
        lift_t0 = time.perf_counter()
        z_next_external, upscaler_name = _learned_lift(
            z0_external, shotplan, int(next_spec["pixel_w"]), int(next_spec["pixel_h"])
        )
        # Pianosequenza memory-safe transition: do not call
        # unload_all_models().  Keeping the H3 patcher resident/managed avoids
        # the expensive full transformer reload that previously happened just
        # before HIGH NFE 1.
        comfy.model_management.soft_empty_cache()
        LOG.info(
            "Pianosequenza %s->%s LIFT COMPLETE | elapsed=%.1fs",
            stage_name, next_spec["name"], time.perf_counter() - lift_t0,
        )

        z_next = latent_format.process_in(z_next_external).to(working_device)
        clean_anchor = latent_format.process_in(stage_videos[spatial_index + 1]).to(z_next)
        if masks is not None and not drift_continuation:
            m_cast = stage_masks[spatial_index + 1].to(device=z_next.device, dtype=z_next.dtype)
            z_next = torch.where(
                m_cast == 0, clean_anchor, z_next * m_cast + clean_anchor * (1.0 - m_cast)
            )

        # Only the actual final spatial stage receives final-prefix DC matching.
        if drift_continuation and spatial_index + 1 == spatial_stage_count - 1:
            lifted_anchor = latent_format.process_out(z_next).to(stage_videos[spatial_index + 1])
            prefix_steps = min(
                int(getattr(drift_state, "prefix_steps", 0) or 0),
                int(lifted_anchor.shape[2]),
                int(stage_videos[spatial_index + 1].shape[2]),
            )
            lifted_anchor, seam_dc = _match_prefix_dc(
                lifted_anchor, stage_videos[spatial_index + 1], prefix_steps
            )
            if seam_dc is not None:
                seam_dc_max = float(seam_dc.float().abs().max().item())
                z_next = latent_format.process_in(lifted_anchor).to(z_next)

        video_noise = comfy.sample.prepare_noise(
            z_next,
            (int(seed) + sampled_index + 1) % (1 << 64),
            prepared.get("batch_index", None),
        ).to(z_next)
        video_state = model_sampling.noise_scaling(sigma_k, video_noise, z_next)
        next_streams = [_euler_step(video_state, z_next, sigma_k, sigma_next)] + audio_next

        if masks is not None:
            anchor_latent = _pack([stage_videos[spatial_index + 1].clone()] + audio_streams, nested)
            anchor_streams, _ = _streams(sampled_models[sampled_index + 1].model.process_latent_in(anchor_latent))
            noise_scale = float(getattr(model_sampling, "noise_scale", 1.0))
            resume_noise_streams = []
            for stream_idx, (state, anchor) in enumerate(zip(next_streams, anchor_streams)):
                sigma = sigma_next.to(device=state.device, dtype=state.dtype)
                clean = anchor.to(device=state.device, dtype=state.dtype)
                resumed_noise = (state - (1.0 - sigma) * clean) / (sigma * noise_scale)
                if stream_idx == 0:
                    guide_noise = comfy.sample.prepare_noise(
                        clean,
                        (int(seed) + sampled_index + 1) % (1 << 64),
                        prepared.get("batch_index", None),
                    ).to(clean)
                    resumed_noise = torch.where(
                        stage_masks[spatial_index + 1].to(device=state.device) == 0,
                        guide_noise,
                        resumed_noise,
                    )
                resume_noise_streams.append(resumed_noise)
            current_latent = anchor_latent
            current_noise = _pack(resume_noise_streams, nested)
            current_noise_mask = _pack([stage_masks[spatial_index + 1]] + auxiliary_masks, nested)
        else:
            resume_streams = [
                model_sampling.inverse_noise_scaling(sigma_next, item) for item in next_streams
            ]
            current_latent = sampled_models[sampled_index + 1].model.process_latent_out(
                _pack(resume_streams, nested)
            )
            current_noise = _pack([torch.zeros_like(item) for item in resume_streams], nested)
            current_noise_mask = None

        current_latent = _move_stream_bundle(current_latent, working_device)
        current_noise = _move_stream_bundle(current_noise, working_device)
        if current_noise_mask is not None:
            current_noise_mask = _move_stream_bundle(current_noise_mask, working_device)

        # Match Pianosequenza HD transition lifetime discipline.  The old
        # IAMCCS loop kept LOW output plus several full-resolution transition
        # aliases alive while the HIGH sampler was entering its first forward.
        # On a 12 GB card that can push ComfyUI into aggressive dynamic-offload
        # churn even though the same 1280 canvas samples normally on its own.
        conditioning_by_stage[sampled_index] = None
        stage_videos[spatial_index] = None
        if masks is not None:
            stage_masks[spatial_index] = None
        del trans_streams, x0_streams, audio_next, z0_external, z_next_external
        del z_next, clean_anchor, video_noise, video_state, next_streams
        if masks is not None:
            del anchor_latent, anchor_streams, resume_noise_streams, resumed_noise, guide_noise
            if not drift_continuation:
                del m_cast
        else:
            del resume_streams
        if drift_continuation and spatial_index + 1 == spatial_stage_count - 1:
            del lifted_anchor
        # `out` is the completed LOW/MID sampler result.  The resume state above
        # is now authoritative; retaining `out` across the next sampler call is
        # unnecessary and was another avoidable live latent bundle.
        del out
        comfy.model_management.soft_empty_cache()

        LOG.info(
            "Pianosequenza %s->%s device sync | latent=%s | noise=%s | mask=%s | working=%s",
            stage_name, next_spec["name"], _bundle_devices(current_latent), _bundle_devices(current_noise),
            _bundle_devices(current_noise_mask), str(working_device),
        )
        step_cursor += step_count

    if final_out is None:
        raise RuntimeError("Pianosequenza produced no final output latent")

    result = prepared.copy()
    result.pop(PREVIOUS_LOW_TAIL_KEY, None)
    result["samples"] = final_out.to(
        device=comfy.model_management.intermediate_device(),
        dtype=comfy.model_management.intermediate_dtype(),
    )
    if low_carry is not None:
        result[LOW_CARRY_KEY] = low_carry
    report = (
        f"Pianosequenza {stage_profile} | stages={'/'.join(report_parts)} | allocator={allocator_reason} "
        f"| final_high_h3={'skipped' if final_lift_only else 'sampled'} "
        f"| upscaler={upscaler_name} | terminal_drift={'yes' if drift_continuation else 'no'} "
        f"| seam_dc_max={seam_dc_max:.5f} | preview=forwarded_on_sampled_stages "
        "| runtime=pianosequenza_progressive | timeline_geometry=unchanged"
    )
    LOG.info(report)
    return result, report


def _pianosequenza_hd_step_split(total_steps: int) -> tuple[int, int]:
    """Recommended HD split: approximately 75% LOW / 25% HIGH.

    The engine still consumes one continuous sigma schedule and reuses the
    boundary denoiser evaluation, so these are NFE ownership counts rather than
    two independent denoise passes.
    """
    total = int(total_steps)
    if total < 2:
        raise ValueError("PIANOSEQUENZA_HD requires at least two denoising steps")
    high = max(1, min(total - 1, int(math.ceil(total * 0.25))))
    return total - high, high


def sample_pianosequenza_hd(
    *, model, conditioning, sigmas: torch.Tensor, latent: dict[str, Any], seed: int,
    shotplan: dict[str, Any], disable_pbar: bool,
) -> tuple[dict[str, Any], str]:
    """Self-contained HD progressive contract.

    PIANOSEQUENZA_HD intentionally ignores the separate Multi-Stage Spatial
    toggle/profile. It always performs exactly two spatial stages at 0.5x and
    1.0x, uses Euler on one continuous sigma schedule, and reserves roughly
    25% of the denoiser evaluations for the HIGH stage. Continuation DRIFT is
    installed before this function by the HD terminal preparation path.
    """
    total_steps = int(sigmas.numel()) - 1
    low_steps, high_steps = _pianosequenza_hd_step_split(total_steps)
    hd_plan = dict(shotplan)
    source_settings = (
        dict(shotplan.get("pianosequenza_2stage_settings"))
        if isinstance(shotplan.get("pianosequenza_2stage_settings"), dict)
        else {}
    )
    source_settings.update({
        "resolution_link_enabled": False,
        "stage_profile": "2_stage_full",
        "stage_count": "2_stage",
        "split_mode": "manual",
        "auto_profile": "balanced",
        "manual_stage1_steps": int(low_steps),
        "manual_stage2_steps": int(high_steps),
        "manual_stage3_steps": 0,
    })
    hd_plan["pianosequenza_2stage_settings"] = source_settings
    LOG.info(
        "PIANOSEQUENZA_HD START | total_steps=%d | low=%d | high=%d | lowres_scale=0.5 | "
        "same_sigma_schedule=yes | boundary_eval_reused=yes",
        total_steps, low_steps, high_steps,
    )
    result, report = sample_pianosequenza_2stage(
        model=model,
        conditioning=conditioning,
        sigmas=sigmas,
        latent=latent,
        seed=int(seed),
        shotplan=hd_plan,
        disable_pbar=bool(disable_pbar),
    )
    report = (
        f"PIANOSEQUENZA_HD | low={low_steps} | high={high_steps} | lowres_scale=0.5 | "
        f"drift={'yes' if callable(getattr(getattr(model, 'model_options', {}).get(DRIFT_STATE_KEY), 'configure_pianosequenza_stage', None)) else 'opening_chunk'} | "
        + report.replace("Pianosequenza 2_stage_full | ", "")
    )
    LOG.info(report)
    return result, report
