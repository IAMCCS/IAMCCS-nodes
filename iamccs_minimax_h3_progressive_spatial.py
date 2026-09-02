# SPDX-License-Identifier: GPL-3.0-or-later
"""IAMCCS-owned progressive spatial sampling for MiniMax H3.

This is an independent implementation of a general diffusion optimisation:
solve the broad spatial structure on a smaller video latent, increase only the
video resolution as the sigma schedule approaches detail formation, and keep
the audio latent untouched for the complete pass.

The implementation deliberately uses stock ComfyUI guider/sampler contracts,
PyTorch interpolation and an IAMCCS high-frequency re-entry step.  It does not
import, wrap or copy any third-party progressive sampler.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn.functional as F


LOG = logging.getLogger("IAMCCS.MiniMaxH3.ProgressiveSpatial")


@dataclass(frozen=True)
class ProgressiveSpatialProfile:
    name: str
    scales: tuple[float, ...]
    tolerance: float
    noise_amplitude: float
    noise_exponent: float
    reentry_gain: float


PROFILES = {
    # Two stages preserve the closest relationship to a normal full-resolution
    # pass and are the only profile promoted as a daily-production baseline.
    "iamccs_progressive_2stage": ProgressiveSpatialProfile(
        "IAMCCS Progressive 2-Stage",
        (0.50, 1.0),
        0.005,
        12.454,
        0.819,
        0.30,
    ),
    # Three stages save more early spatial work, but can change fine text,
    # small props or prompt adherence. It remains explicit and opt-in.
    "iamccs_progressive_3stage": ProgressiveSpatialProfile(
        "IAMCCS Progressive 3-Stage · Experimental",
        (1.0 / 3.0, 2.0 / 3.0, 1.0),
        0.005,
        12.454,
        0.819,
        0.24,
    ),
    # Same conservative spatial ladder while the MiniMax native PDD head bank
    # owns the eight-step denoise schedule. PDD validation remains upstream.
    "iamccs_progressive_pdd_2stage": ProgressiveSpatialProfile(
        "IAMCCS Progressive 2-Stage + Native PDD",
        (0.50, 1.0),
        0.005,
        12.454,
        0.819,
        0.26,
    ),
}


def is_progressive_spatial_mode(value: Any) -> bool:
    return str(value or "").strip().lower() in PROFILES


def _nested_parts(samples: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not bool(getattr(samples, "is_nested", False)):
        raise ValueError("IAMCCS Progressive Spatial requires MiniMax H3 nested AV latents")
    parts = list(samples.unbind())
    if len(parts) != 2:
        raise ValueError(f"IAMCCS Progressive Spatial expected video+audio latents, received {len(parts)} tensors")
    video, audio = parts
    if not torch.is_tensor(video) or video.ndim != 5 or int(video.shape[1]) != 24:
        raise ValueError(f"IAMCCS Progressive Spatial received an invalid H3 video latent shape: {getattr(video, 'shape', None)}")
    if not torch.is_tensor(audio):
        raise ValueError("IAMCCS Progressive Spatial received no H3 audio latent")
    return video, audio


def _nested(video: torch.Tensor, audio: torch.Tensor):
    import comfy.nested_tensor

    return comfy.nested_tensor.NestedTensor((video, audio))


def _legal_latent_size(value: float, full: int) -> int:
    target = max(2, min(int(full), int(round(float(full) * float(value)))))
    if target < int(full) and target % 2:
        target += 1
    return max(2, min(int(full), target))


def _resize_video(video: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if tuple(video.shape[-2:]) == (int(height), int(width)):
        return video
    source_dtype = video.dtype
    resized = F.interpolate(
        video.float(),
        size=(int(video.shape[-3]), int(height), int(width)),
        mode="trilinear",
        align_corners=False,
    )
    return resized.to(dtype=source_dtype)


def _threshold(scale: float, latent_height: int, latent_width: int, profile: ProgressiveSpatialProfile) -> float:
    frequency = max(1.0, float(scale) * float(min(latent_height, latent_width)) / 2.0)
    power = max(1e-8, float(profile.noise_amplitude) * math.pow(frequency, -float(profile.noise_exponent)))
    delta = max(1e-6, min(float(profile.tolerance), power * (1.0 + power) * 0.999))
    denominator = power * max(1e-8, 1.0 + power - delta)
    return 1.0 / (1.0 + math.sqrt(delta / denominator))


def _stage_boundaries(sigmas: torch.Tensor, profile: ProgressiveSpatialProfile, latent_height: int, latent_width: int) -> tuple[int, ...]:
    if not torch.is_tensor(sigmas) or sigmas.ndim != 1 or int(sigmas.numel()) < 3:
        raise ValueError("IAMCCS Progressive Spatial requires a one-dimensional sigma schedule with at least two steps")
    steps = int(sigmas.numel()) - 1
    boundaries: list[int] = []
    previous = 0
    cpu_sigmas = sigmas.detach().float().cpu().tolist()
    for stage_index, scale in enumerate(profile.scales[:-1]):
        target = _threshold(scale, latent_height, latent_width, profile)
        index = next((i for i in range(previous + 1, steps) if float(cpu_sigmas[i]) <= target), None)
        min_index = previous + 1
        remaining_transitions = len(profile.scales) - stage_index - 1
        max_index = steps - remaining_transitions
        if index is None:
            index = max(min_index, int(round(steps * (stage_index + 1) / len(profile.scales))))
        boundaries.append(max(min_index, min(max_index, int(index))))
        previous = boundaries[-1]
    boundaries.append(steps)
    return tuple(boundaries)


def _seeded_noise_like(tensor: torch.Tensor, seed: int) -> torch.Tensor:
    # A CPU generator is deterministic across CUDA memory-management modes;
    # only the resulting tensor is transferred to the active latent device.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
    generated = torch.randn(tuple(tensor.shape), generator=generator, dtype=torch.float32, device="cpu")
    return generated.to(device=tensor.device, dtype=tensor.dtype)


def _high_frequency_reentry(
    video: torch.Tensor,
    previous_height: int,
    previous_width: int,
    sigma: float,
    gain: float,
    seed: int,
) -> torch.Tensor:
    if float(gain) <= 0.0 or float(sigma) <= 0.0:
        return video
    fresh = _seeded_noise_like(video, seed)
    low = _resize_video(fresh, previous_height, previous_width)
    low = _resize_video(low, int(video.shape[-2]), int(video.shape[-1]))
    high = fresh - low
    rms = high.float().square().mean().sqrt().clamp_min(1e-6)
    return video + high * (float(sigma) * float(gain) / rms).to(dtype=video.dtype)


def sample_progressive_spatial(
    *,
    noise: Any,
    guider: Any,
    sigmas: torch.Tensor,
    latent: dict[str, Any],
    acceleration_mode: str,
    seed: int,
    disable_pbar: bool,
) -> tuple[dict[str, Any], str]:
    """Run the isolated progressive path and return a normal Comfy LATENT."""

    mode = str(acceleration_mode or "").strip().lower()
    profile = PROFILES.get(mode)
    if profile is None:
        raise ValueError(f"Unknown IAMCCS Progressive Spatial profile: {acceleration_mode}")
    if "noise_mask" in latent:
        raise ValueError("IAMCCS Progressive Spatial does not yet support noise_mask; use Native/PDD for masked sampling")

    import comfy.model_management
    import comfy.sample
    import comfy.samplers

    prepared = latent.copy()
    samples = comfy.sample.fix_empty_latent_channels(
        guider.model_patcher,
        prepared["samples"],
        prepared.get("downscale_ratio_spacial"),
        prepared.get("downscale_ratio_temporal"),
    )
    full_video, full_audio = _nested_parts(samples)
    full_height, full_width = int(full_video.shape[-2]), int(full_video.shape[-1])
    boundaries = _stage_boundaries(sigmas, profile, full_height, full_width)
    euler = comfy.samplers.sampler_object("euler")

    current = None
    start_index = 0
    stage_report: list[str] = []
    for stage_index, (scale, end_index) in enumerate(zip(profile.scales, boundaries)):
        target_height = _legal_latent_size(scale, full_height)
        target_width = _legal_latent_size(scale, full_width)
        if current is None:
            stage_video = _resize_video(full_video, target_height, target_width)
            stage_audio = full_audio
            stage_samples = _nested(stage_video, stage_audio)
            stage_latent = prepared.copy()
            stage_latent["samples"] = stage_samples
            stage_noise = noise.generate_noise(stage_latent)
        else:
            current_video, current_audio = _nested_parts(current)
            previous_height, previous_width = int(current_video.shape[-2]), int(current_video.shape[-1])
            stage_video = _resize_video(current_video, target_height, target_width)
            stage_video = _high_frequency_reentry(
                stage_video,
                previous_height,
                previous_width,
                float(sigmas[start_index].detach().cpu()),
                profile.reentry_gain,
                int(seed) + 10000 + stage_index,
            )
            # Audio is carried byte-for-byte across spatial transitions.
            stage_audio = current_audio
            stage_samples = _nested(stage_video, stage_audio)
            stage_noise = _nested(torch.zeros_like(stage_video), torch.zeros_like(stage_audio))

        stage_sigmas = sigmas[start_index : end_index + 1]
        if int(stage_sigmas.numel()) < 2:
            raise RuntimeError(
                f"IAMCCS Progressive Spatial produced an empty stage {stage_index + 1}: "
                f"sigma indices {start_index}:{end_index}"
            )
        LOG.info(
            "IAMCCS Progressive Spatial | stage=%d/%d | scale=%.3f | video_latent=%dx%d | sigma_steps=%d | audio=full",
            stage_index + 1,
            len(profile.scales),
            float(scale),
            target_width,
            target_height,
            int(stage_sigmas.numel()) - 1,
        )
        current = guider.sample(
            stage_noise,
            stage_samples,
            euler,
            stage_sigmas,
            denoise_mask=None,
            callback=None,
            disable_pbar=disable_pbar,
            seed=int(seed) + stage_index,
        )
        stage_report.append(f"{target_width}x{target_height}:{int(stage_sigmas.numel()) - 1}step")
        start_index = end_index

    out = prepared.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = current.to(comfy.model_management.intermediate_device())
    report = (
        f"{profile.name} (Euler, conservative delta={profile.tolerance:g}, "
        f"AV audio full, stages={' -> '.join(stage_report)})"
    )
    return out, report


class IAMCCS_MiniMaxH3ProgressiveSpatialSampler:
    """Standalone advanced-sampler replacement for controlled experiments."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
                "profile": (list(PROFILES), {"default": "iamccs_progressive_2stage"}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "STRING")
    RETURN_NAMES = ("output", "denoised_output", "report")
    FUNCTION = "sample"
    CATEGORY = "IAMCCS/Shotboard/Backends/MiniMax H3"

    def sample(self, noise, guider, sigmas, latent_image, profile):
        import comfy.utils

        output, report = sample_progressive_spatial(
            noise=noise,
            guider=guider,
            sigmas=sigmas,
            latent=latent_image,
            acceleration_mode=profile,
            seed=int(getattr(noise, "seed", 0)),
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
        )
        return output, output, report


NODE_CLASS_MAPPINGS = {
    "IAMCCS_MiniMaxH3ProgressiveSpatialSampler": IAMCCS_MiniMaxH3ProgressiveSpatialSampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_MiniMaxH3ProgressiveSpatialSampler": "IAMCCS MiniMax H3 Progressive Spatial Sampler",
}

