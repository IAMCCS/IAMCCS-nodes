"""Resolution-aware, temporally evolving film-grain finishing for IMAGE batches.

The implementation follows published film-grain rendering principles rather than
overlaying a repeated texture: grain is stochastic per frame, processed in linear
light, tone dependent, resolution aware, and gently correlated over time.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


PRESETS = (
    "65mm_4k_scan_subtle",
    "35mm_vision3_fine",
    "35mm_500t_texture",
    "16mm_fine_documentary",
    "custom_box_values",
)
BLEND_METHODS = ("density_exposure", "linear_additive", "soft_light_luma")


def _srgb_to_linear(value: torch.Tensor) -> torch.Tensor:
    return torch.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055).pow(2.4))


def _linear_to_srgb(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp_min(0.0)
    return torch.where(value <= 0.0031308, value * 12.92, 1.055 * value.pow(1.0 / 2.4) - 0.055)


def _normalized_field(
    height: int,
    width: int,
    pixel_size: float,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    scale = max(1.0, float(pixel_size))
    source_h = max(2, int(math.ceil(height / scale)))
    source_w = max(2, int(math.ceil(width / scale)))
    field = torch.randn((1, 1, source_h, source_w), generator=generator, device=device, dtype=torch.float32)
    if source_h != height or source_w != width:
        field = F.interpolate(field, size=(height, width), mode="bicubic", align_corners=False)
    # A weak second octave avoids electronic white-noise texture while keeping
    # the result fine enough for a 4K finishing pass.
    clump_h = max(2, int(math.ceil(height / max(1.0, scale * 2.75))))
    clump_w = max(2, int(math.ceil(width / max(1.0, scale * 2.75))))
    clump = torch.randn((1, 1, clump_h, clump_w), generator=generator, device=device, dtype=torch.float32)
    clump = F.interpolate(clump, size=(height, width), mode="bicubic", align_corners=False)
    field = field * 0.86 + clump * 0.14
    return (field - field.mean()) / field.std(unbiased=False).clamp_min(1e-6)


def _soft_light(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    low = base - (1.0 - 2.0 * blend) * base * (1.0 - base)
    d = torch.where(base <= 0.25, ((16.0 * base - 12.0) * base + 4.0) * base, base.sqrt())
    high = base + (2.0 * blend - 1.0) * (d - base)
    return torch.where(blend <= 0.5, low, high)


class IAMCCS_CineTemporalFilmGrain4K:
    """Fine animated grain intended after detail/upscale and before encoding."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "preset": (PRESETS, {"default": "65mm_4k_scan_subtle"}),
                "blend_method": (BLEND_METHODS, {"default": "density_exposure"}),
                "strength": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "grain_size_4k_px": ("FLOAT", {"default": 0.58, "min": 0.35, "max": 4.0, "step": 0.05, "display": "slider"}),
                "temporal_persistence": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 0.85, "step": 0.01, "display": "slider"}),
                "chroma_amount": ("FLOAT", {"default": 0.025, "min": 0.0, "max": 0.5, "step": 0.01, "display": "slider"}),
                "shadow_response": ("FLOAT", {"default": 0.48, "min": 0.0, "max": 2.0, "step": 0.02, "display": "slider"}),
                "midtone_response": ("FLOAT", {"default": 0.82, "min": 0.0, "max": 2.0, "step": 0.02, "display": "slider"}),
                "highlight_response": ("FLOAT", {"default": 0.24, "min": 0.0, "max": 2.0, "step": 0.02, "display": "slider"}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0x7FFFFFFFFFFFFFFF}),
                "frame_start": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("images", "grain_map", "report")
    FUNCTION = "apply"
    CATEGORY = "IAMCCS/Cine Nodes/Film Delivery"
    DESCRIPTION = (
        "Animated, resolution-aware film grain in linear light. Place after the final detail/upscale stage "
        "and before the encoder. Box values are render truth; preset selection only fills editable defaults in the UI."
    )

    def apply(
        self,
        images,
        preset,
        blend_method,
        strength,
        grain_size_4k_px,
        temporal_persistence,
        chroma_amount,
        shadow_response,
        midtone_response,
        highlight_response,
        seed,
        frame_start,
    ):
        if not torch.is_tensor(images) or images.ndim != 4 or images.shape[-1] not in (3, 4):
            raise ValueError("IAMCCS Cine Temporal Film Grain expects an IMAGE batch in BHWC RGB/RGBA format")
        if blend_method not in BLEND_METHODS:
            raise ValueError(f"Unknown grain blend method: {blend_method}")
        batch, height, width, channels = images.shape
        if batch == 0:
            return images, images.new_zeros((0, height, width)), "No frames received."
        if float(strength) <= 0.0:
            return (
                images.clone(),
                images.new_zeros((batch, height, width)),
                f"IAMCCS Cine Temporal Film Grain 4K | preset={preset} | bypassed=strength_zero | frames={batch}",
            )

        original_dtype = images.dtype
        # Keep only one output-sized allocation. Converting a complete long
        # fp16 batch to fp32 here would defeat the node's frame-wise memory
        # behavior; only the active frame is promoted for linear-light maths.
        output = images.clone()
        grain_map = torch.empty((batch, height, width), device=images.device, dtype=torch.float32)
        long_edge = max(height, width)
        resolved_size = max(0.35, float(grain_size_4k_px) * long_edge / 4096.0)
        persistence = min(0.85, max(0.0, float(temporal_persistence)))
        fresh_weight = math.sqrt(max(0.0, 1.0 - persistence * persistence))
        previous = None

        for frame_index in range(batch):
            generator = torch.Generator(device=images.device)
            generator.manual_seed((int(seed) + int(frame_start) + frame_index) & 0x7FFFFFFFFFFFFFFF)
            common = _normalized_field(height, width, resolved_size, generator, images.device)
            if previous is not None and persistence > 0.0:
                common = common * fresh_weight + previous * persistence
                common = (common - common.mean()) / common.std(unbiased=False).clamp_min(1e-6)
            previous = common
            common_hwc = common[0, 0, :, :, None]

            if float(chroma_amount) > 0.0:
                chroma = torch.cat([
                    _normalized_field(height, width, resolved_size * 1.08, generator, images.device)
                    for _ in range(3)
                ], dim=1)[0].permute(1, 2, 0)
                # Blue-biased chroma sensitivity is subtle and never replaces
                # the shared luminance grain structure.
                chroma[..., 2] *= 1.08
                noise = common_hwc * (1.0 - float(chroma_amount)) + chroma * float(chroma_amount)
            else:
                noise = common_hwc.expand(height, width, 3)

            frame_srgb = images[frame_index, ..., :3].to(torch.float32).clamp(0.0, 1.0)
            frame_linear = _srgb_to_linear(frame_srgb)
            luma = (frame_linear[..., 0] * 0.2126 + frame_linear[..., 1] * 0.7152 + frame_linear[..., 2] * 0.0722)
            shadow_w = ((0.50 - luma) / 0.50).clamp(0.0, 1.0)
            highlight_w = ((luma - 0.50) / 0.50).clamp(0.0, 1.0)
            mid_w = (1.0 - shadow_w - highlight_w).clamp(0.0, 1.0)
            tone = (
                shadow_w * float(shadow_response)
                + mid_w * float(midtone_response)
                + highlight_w * float(highlight_response)
            ).clamp(0.0, 2.0)
            sigma = float(strength) * 0.19 * tone[..., None]

            if blend_method == "density_exposure":
                processed_linear = frame_linear * torch.exp(noise * sigma - 0.5 * sigma.square())
                processed = _linear_to_srgb(processed_linear)
            elif blend_method == "linear_additive":
                processed = _linear_to_srgb(frame_linear + noise * sigma * 0.32)
            else:
                blend = (0.5 + noise * sigma * 1.9).clamp(0.0, 1.0)
                processed = _soft_light(frame_srgb, blend)

            output[frame_index, ..., :3] = processed.clamp(0.0, 1.0).to(original_dtype)
            grain_map[frame_index] = (common[0, 0].abs() / 3.0).clamp(0.0, 1.0)

        report = (
            f"IAMCCS Cine Temporal Film Grain 4K | preset={preset} | blend={blend_method} | "
            f"frames={batch} | {width}x{height} | strength={float(strength):.3f} | "
            f"grain_size_4k={float(grain_size_4k_px):.2f}px | resolved_size={resolved_size:.2f}px | "
            f"temporal_persistence={persistence:.2f} | chroma={float(chroma_amount):.2f} | "
            "linear_light=yes | repeated_texture=no"
        )
        return output, grain_map.to(original_dtype), report


NODE_CLASS_MAPPINGS = {"IAMCCS_CineTemporalFilmGrain4K": IAMCCS_CineTemporalFilmGrain4K}
NODE_DISPLAY_NAME_MAPPINGS = {"IAMCCS_CineTemporalFilmGrain4K": "IAMCCS Cine Temporal Film Grain · 4K Scan"}
