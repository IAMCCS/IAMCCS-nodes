# iamccs_wan_svipro_motion.py
# ===============================================================
# IAMCCS WanImageMotion
# Drop-in replacement for KJNodes WanImageToVideoSVIPro with motion control
#
# Copyright (C) 2025-2026 Carmine Cristallo Scalzi (IAMCCS)
#
# This file is part of IAMCCS-nodes and is distributed under the
# GNU General Public License v3.0 (GPL-3.0).
#
# This file contains logic derived from the following GPL-3.0 projects:
#   - ComfyUI-Wan-SVI2Pro-FLF (https://github.com/IAMCCS/ComfyUI-Wan-SVI2Pro-FLF)
#   - ComfyUI-KJNodes (https://github.com/kijai/ComfyUI-KJNodes)
#   - ComfyUI (https://github.com/comfyanonymous/ComfyUI)
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# ===============================================================

import torch
import logging
import comfy.model_management
import comfy.latent_formats
import comfy.clip_vision
import comfy.utils
import node_helpers


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    # x in [0,1]
    return x * x * (3.0 - 2.0 * x)


def _apply_soft_limiter(x: torch.Tensor, *, mode: str, limit: float) -> torch.Tensor:
    if mode == "hard":
        return x.clamp_(-limit, limit)
    if mode == "tanh":
        # Smooth limiter: prevents hard saturation artifacts.
        # For small values, tanh(x/limit) ≈ x/limit.
        return x.div_(limit).tanh_().mul_(limit)
    # Fallback
    return x.clamp_(-limit, limit)


def _center_diff(diff: torch.Tensor, *, mean_mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (diff_centered, diff_mean).

    mean_mode:
    - frame_scalar: legacy behavior (mean over channels+spatial).
    - per_channel: mean per channel (mean over spatial only). Helps reduce hue shifts.
    """

    if mean_mode == "per_channel":
        diff_mean = diff.mean(dim=(3, 4), keepdim=True)
    else:
        # legacy
        diff_mean = diff.mean(dim=(1, 3, 4), keepdim=True)
    diff_centered = diff - diff_mean
    return diff_centered, diff_mean


def _preset_params(safety_preset: str, motion_amplitude: float) -> dict:
    # "base": 1:1 with WanImageToVideoSVIProFLF / WanImageToVideoSVIPro.
    # No amplitude processing at all — pure latent concatenation, zero modification.
    base = {
        "enabled": False,
    }

    # "legacy": mirrors PainterI2V algorithm (frame-scalar diff centering, hard clamp ±6, no ramp).
    legacy = {
        "enabled": True,
        "mean_mode": "frame_scalar",
        "limiter_mode": "hard",
        "limiter_limit": 6.0,
        "ramp_frames": 0,
    }

    # "safe": per-channel stabilisation + tanh limiter + 2-frame ramp.
    # Reduces hue/saturation drift at any motion value.
    safe = {
        "enabled": True,
        "mean_mode": "per_channel",
        "limiter_mode": "tanh",
        "limiter_limit": 6.0,
        "ramp_frames": 2,
    }

    # "safer": same as safe but tighter limiter and longer ramp.
    # Recommended above motion=1.5 or for longer chains.
    safer = {
        "enabled": True,
        "mean_mode": "per_channel",
        "limiter_mode": "tanh",
        "limiter_limit": 5.5,
        "ramp_frames": 4,
    }

    if safety_preset == "base":
        return base
    if safety_preset == "legacy":
        return legacy
    if safety_preset == "safer":
        return safer
    # default: "safe"
    return safe


def _extract_prev_motion_tail(
    prev_samples,
    motion_latent_count: int,
    *,
    tail_pick_mode: str = "latest",
    skip_last_slots: int = 0,
    anchor_reference: torch.Tensor | None = None,
    anchor_blend: float = 0.0,
) -> tuple[torch.Tensor | None, int, int, int]:
    """Extract a FLF-normalized motion tail from prev_samples.

    Useful when prev_samples comes from a non-FLF generator and needs to be
    trimmed/aligned before being reused as SVI motion continuity.
    """

    if prev_samples is None or motion_latent_count == 0:
        return None, 0, 0, 0

    prev_latent = prev_samples["samples"].clone()
    t_prev = int(prev_latent.shape[2])
    skip_last_slots = max(0, int(skip_last_slots))
    usable_t = max(0, t_prev - skip_last_slots)

    if usable_t <= 0:
        return None, 0, t_prev, 0

    prev_usable = prev_latent[:, :, :usable_t]
    motion_count = min(int(motion_latent_count), int(prev_usable.shape[2]))
    if motion_count <= 0:
        return None, 0, t_prev, usable_t

    if tail_pick_mode == "first_usable":
        motion_tail = prev_usable[:, :, :motion_count]
    else:
        motion_tail = prev_usable[:, :, -motion_count:]

    anchor_blend = float(anchor_blend)
    if anchor_reference is not None and anchor_blend > 0.0:
        try:
            anchor_ref = anchor_reference[:, :, :1].to(device=motion_tail.device, dtype=motion_tail.dtype)
            if anchor_ref.shape[0] == 1 and motion_tail.shape[0] > 1:
                anchor_ref = anchor_ref.repeat(motion_tail.shape[0], 1, 1, 1, 1)
            if (
                anchor_ref.shape[1] == motion_tail.shape[1]
                and anchor_ref.shape[3] == motion_tail.shape[3]
                and anchor_ref.shape[4] == motion_tail.shape[4]
            ):
                anchor_ref = anchor_ref.expand(-1, -1, motion_count, -1, -1)
                motion_tail = torch.lerp(motion_tail, anchor_ref, min(anchor_blend, 1.0))
        except Exception:
            pass

    return motion_tail, motion_count, t_prev, usable_t


def _resolve_prev_tail_profile(
    profile: str,
    *,
    tail_pick_mode: str,
    skip_last_slots: int,
    anchor_blend: float,
) -> tuple[str, int, float]:
    """Resolve explicit prev tail controls from a high-level profile.

    direct:
    - use the widget values as-is

    mixed_svi_safe:
    - skip the last slot from prev_samples
    - still use the latest usable motion tail

    mixed_svi_aggressive:
    - skip the last slot from prev_samples
    - pick the earliest usable motion tail segment
    """

    profile = str(profile or "direct")
    if profile == "mixed_svi_safe":
        return "latest", max(1, int(skip_last_slots)), 0.0
    if profile == "mixed_svi_aggressive":
        return "first_usable", max(1, int(skip_last_slots)), 0.0
    return str(tail_pick_mode or "latest"), max(0, int(skip_last_slots)), float(anchor_blend)


def _common_upscale_nhwc_to(images: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    if int(images.shape[1]) == int(target_h) and int(images.shape[2]) == int(target_w):
        return images
    return comfy.utils.common_upscale(
        images.movedim(-1, 1),
        int(target_w),
        int(target_h),
        "bilinear",
        "center",
    ).movedim(1, -1)


def _prepare_vae_pixels(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4:
        raise ValueError(f"Expected NHWC IMAGE tensor, got shape {tuple(image.shape)}")
    pixels = image[:, :, :, :3].float()
    pixels = torch.nan_to_num(pixels, nan=0.0, posinf=1.0, neginf=0.0)
    return pixels.clamp(0.0, 1.0).contiguous()


def _normalize_encoded_latent(encoded) -> torch.Tensor:
    if isinstance(encoded, dict):
        encoded = encoded.get("samples", encoded)
    if not isinstance(encoded, torch.Tensor):
        raise TypeError(f"Unexpected VAE encode output type: {type(encoded)!r}")
    if encoded.ndim == 4:
        encoded = encoded.unsqueeze(2)
    if encoded.ndim != 5:
        raise ValueError(f"Unexpected encoded latent shape: {tuple(encoded.shape)}")
    return encoded.contiguous()


def _match_latent_batch(base_samples: torch.Tensor, other_samples: torch.Tensor) -> torch.Tensor:
    base_batch = int(base_samples.shape[0])
    other_batch = int(other_samples.shape[0])
    if base_batch == other_batch:
        return other_samples
    if other_batch == 1 and base_batch > 1:
        reps = [base_batch] + [1] * (other_samples.dim() - 1)
        return other_samples.repeat(*reps)
    return other_samples[:base_batch]


def _encode_bridge_image_to_latent(vae, image: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    pixels = _common_upscale_nhwc_to(image, int(target_h), int(target_w))
    pixels = _prepare_vae_pixels(pixels)
    encoded = vae.encode(pixels)
    return _normalize_encoded_latent(encoded)


class IAMCCS_WanImageMotion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "length": ("INT", {"default": 81, "min": 1, "max": 16384, "step": 4}),
                "anchor_samples": ("LATENT",),
                # Match FLF/SVI Pro semantics: typical 0-16.
                "motion_latent_count": ("INT", {"default": 1, "min": 0, "max": 16, "step": 1}),
                "motion": ("FLOAT", {"default": 1.15, "min": 1.0, "max": 2.0, "step": 0.05}),
                "motion_mode": (
                    [
                        "motion_only (prev_samples)",
                        "all_nonfirst (anchor+motion)",
                    ],
                    {"default": "motion_only (prev_samples)"},
                ),
                "add_reference_latents": ("BOOLEAN", {"default": False}),
                "latent_precision": (
                    [
                        "auto",
                        "fp16",
                        "fp32",
                        "normal",
                    ],
                    # FLF reference node allocates empty latent as fp32 by default.
                    {"default": "fp32"},
                ),
                "vram_profile": (
                    [
                        "normal",
                        "chunked_blocks_2",
                        "chunked_blocks_4",
                        "loop_per_frame (lowest_vram)",
                        "cpu_offload (slowest)",
                    ],
                    {"default": "normal"},
                ),
                "include_padding_in_motion": ("BOOLEAN", {"default": False}),
                # Keep this at the end to preserve existing widgets_values indexing in saved workflows.
                "safety_preset": (
                    [
                        "base",
                        "safe",
                        "safer",
                        "legacy",
                    ],
                    {
                        "default": "safe",
                        "tooltip": (
                            "base   : 1:1 with WanImageToVideoSVIProFLF — no amplitude processing at all. "
                            "safe   : per-channel stabilisation + tanh limiter + 2-frame ramp (active at any motion value, recommended default). "
                            "safer  : same as safe but tighter limiter + longer ramp (best above motion=1.5). "
                            "legacy : hard clamp ±6 + frame-scalar diff centering, mirrors PainterI2V algorithm."
                        ),
                    },
                ),
                # Append-only widget for backward compatibility:
                # older workflows will keep correct widget index alignment and this will default.
                "lock_start_slots": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 16,
                        "step": 1,
                        "tooltip": (
                            "How many initial latent slots to hard-lock to the start image via concat_mask. "
                            "Default 1 (≈ first 4 video frames) matches FLF-style start anchoring. "
                            "Set 0 to allow motion immediately from the first video frame."
                        ),
                    },
                ),
                # Append-only: diagnostic logging toggle (does not change outputs).
                "diagnostic_log": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "When enabled, prints compact diagnostics about concat_latent_image/mask and motion ranges. "
                            "Useful to debug 'static clip' or segment discontinuities."
                        ),
                    },
                ),
                # Append-only: allow keeping prev_samples wired (autolink) but ignoring it.
                "use_prev_samples": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "When False, prev_samples is ignored even if the wire is connected. "
                            "Use this on the first segment when autolink forces prev_samples to be connected."
                        ),
                    },
                ),
                # DC Drift Correction: corrects per-channel mean shift accumulated
                # across chained segments. Does NOT touch spatial structure or motion.
                # Algorithm:  drift = mean(motion_tail, dims=T,H,W) − mean(anchor, dims=T,H,W)
                #             motion_tail -= drift * latent_refresh
                "latent_refresh": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "DC drift correction strength (0=disabled, 1=full correction). "
                            "Corrects the per-channel mean shift between motion tail and anchor latent. "
                            "Does not touch spatial structure or motion. "
                            "Start with 0.5 for chains of 3+ segments. Enable diagnostic_log to verify."
                        ),
                    },
                ),
                # Soft-clamp on the DC drift itself.
                # Prevents over-correction when drift is very large (first segment or big motion).
                # 0 = disabled (no clamping, full drift correction applied).
                "delta_max": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.1,
                        "tooltip": (
                            "Soft-clamp on DC drift magnitude (0=disabled). "
                            "Caps how much drift correction can be applied per channel. "
                            "Use 0.5–1.0 to prevent over-correction on large motion segments."
                        ),
                    },
                ),
            },
            "optional": {
                "prev_samples": ("LATENT",),
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "apply"
    CATEGORY = "IAMCCS/Wan"

    _log = logging.getLogger("IAMCCS.WanImageMotion")

    def _pick_empty_latent_dtype(self, anchor_dtype: torch.dtype, latent_precision: str) -> torch.dtype:
        if latent_precision == "normal":
            # Backward-compat alias for older workflows.
            return anchor_dtype
        if latent_precision == "auto":
            # Prefer fp32 for 1:1 compatibility with the FLF reference node.
            return torch.float32
        if latent_precision == "fp32":
            return torch.float32
        if latent_precision == "fp16":
            return torch.float16
        return anchor_dtype

    def _apply_motion_amplitude(self, image_cond_latent: torch.Tensor, *, real_latents: int, anchor_latents: int,
                               motion_latents: int, motion_amplitude: float, motion_mode: str,
                               vram_profile: str, safety_preset: str = "safe") -> torch.Tensor:
        preset = _preset_params(safety_preset, motion_amplitude)

        # "base" preset: 1:1 with reference nodes — no processing at all.
        if not preset.get("enabled", True):
            return image_cond_latent

        if motion_amplitude is None or motion_amplitude <= 1.0:
            return image_cond_latent

        if image_cond_latent.shape[2] <= 1:
            return image_cond_latent

        base_latent = image_cond_latent[:, :, 0:1]  # first latent frame

        def _scale_slice_gpu(view_slice: torch.Tensor, *, gain: torch.Tensor | float) -> torch.Tensor:
            # view_slice: [B,C,T,H,W]
            # VRAM-optimized variant: keep only one full-sized temporary tensor.
            with torch.no_grad():
                diff = view_slice - base_latent
                diff_centered, diff_mean = _center_diff(diff, mean_mode=preset["mean_mode"])
                diff_centered.mul_(gain)
                out_local = diff_centered.add_(diff_mean).add_(base_latent)
                _apply_soft_limiter(out_local, mode=preset["limiter_mode"], limit=float(preset["limiter_limit"]))
                return out_local

        def _gain_weights(start: int, end: int) -> torch.Tensor | float:
            # Returns broadcastable gain weights for the slice.
            # gain = 1 + (motion-1)*w(t)
            ramp_frames = int(preset["ramp_frames"])
            if ramp_frames <= 0:
                return float(motion_amplitude)
            tcount = max(0, end - start)
            if tcount <= 0:
                return float(motion_amplitude)
            # ramp up only at the beginning of the boosted range
            ramp = min(ramp_frames, tcount)
            w = torch.ones((tcount,), device=image_cond_latent.device, dtype=image_cond_latent.dtype)
            if ramp > 0:
                # 0..1 over ramp
                x = torch.linspace(0.0, 1.0, steps=ramp, device=w.device, dtype=w.dtype)
                w[:ramp] = _smoothstep(x)
            gain = 1.0 + (motion_amplitude - 1.0) * w
            return gain.view(1, 1, tcount, 1, 1)

        def _apply_to_range(start: int, end: int) -> torch.Tensor:
            # Applies scaling to out[:, :, start:end] according to VRAM profile.
            # NOTE: caller controls what "real_latents" means. By default it excludes padding;
            # if include_padding_in_motion is enabled, caller may set real_latents=total_latents.
            if end <= start:
                return out

            if vram_profile == "normal":
                gain = _gain_weights(start, end)
                out[:, :, start:end] = _scale_slice_gpu(out[:, :, start:end], gain=gain)
                return out

            if vram_profile in ("chunked_blocks_2", "chunked_blocks_4"):
                block = 2 if vram_profile == "chunked_blocks_2" else 4
                t = start
                while t < end:
                    t2 = min(end, t + block)
                    gain = _gain_weights(t, t2)
                    out[:, :, t:t2] = _scale_slice_gpu(out[:, :, t:t2], gain=gain)
                    t = t2
                return out

            if vram_profile == "loop_per_frame (lowest_vram)":
                for t in range(start, end):
                    gain = _gain_weights(t, t + 1)
                    out[:, :, t:t+1] = _scale_slice_gpu(out[:, :, t:t+1], gain=gain)
                return out

            if vram_profile == "cpu_offload (slowest)":
                # Offload just the slice (and base frame) to CPU, then copy back.
                # This can help in extreme VRAM limits but is much slower.
                with torch.no_grad():
                    device = out.device
                    base_cpu = base_latent.detach().to("cpu")
                    slice_cpu = out[:, :, start:end].detach().to("cpu")
                    diff = slice_cpu - base_cpu
                    diff_centered, diff_mean = _center_diff(diff, mean_mode=preset["mean_mode"])
                    # gain weights are computed on the target device; rebuild on CPU
                    ramp_frames = int(preset["ramp_frames"])
                    tcount = max(0, end - start)
                    if ramp_frames > 0 and tcount > 0:
                        ramp = min(ramp_frames, tcount)
                        w = torch.ones((tcount,), device=diff_centered.device, dtype=diff_centered.dtype)
                        x = torch.linspace(0.0, 1.0, steps=ramp, device=w.device, dtype=w.dtype)
                        w[:ramp] = _smoothstep(x)
                        gain = (1.0 + (motion_amplitude - 1.0) * w).view(1, 1, tcount, 1, 1)
                    else:
                        gain = float(motion_amplitude)
                    diff_centered.mul_(gain)
                    out_cpu = diff_centered.add_(diff_mean).add_(base_cpu)
                    _apply_soft_limiter(out_cpu, mode=preset["limiter_mode"], limit=float(preset["limiter_limit"]))
                    out[:, :, start:end] = out_cpu.to(device)
                return out

            # Fallback
            gain = _gain_weights(start, end)
            out[:, :, start:end] = _scale_slice_gpu(out[:, :, start:end], gain=gain)
            return out

        # Avoid touching padding: operate only within [0:real_latents)
        real_latents = max(0, min(real_latents, image_cond_latent.shape[2]))
        if real_latents <= 1:
            return image_cond_latent

        out = image_cond_latent

        if motion_mode == "motion_only (prev_samples)":
            if motion_latents <= 0:
                return out

            start = anchor_latents
            end = min(anchor_latents + motion_latents, real_latents)
            if end <= start:
                return out

            return _apply_to_range(start, end)

        # "all_nonfirst (anchor+motion)"
        start = 1
        end = real_latents
        return _apply_to_range(start, end)

    def apply(self, positive, negative, length, anchor_samples, motion_latent_count, motion, motion_mode,
              add_reference_latents, latent_precision, vram_profile, include_padding_in_motion,
              safety_preset="safe", lock_start_slots=1, diagnostic_log=False, use_prev_samples=True,
              latent_refresh=0.0, delta_max=0.0,
              prev_samples=None, clip_vision_output=None, **_kwargs):
        with torch.no_grad():
            # Clone to prevent in-place motion amplitude writes from corrupting the caller's tensor.
            anchor_latent = anchor_samples["samples"].clone()

            B, C, T_anchor, H, W = anchor_latent.shape

            total_latents = (length - 1) // 4 + 1

            device = anchor_latent.device
            dtype = anchor_latent.dtype

            empty_latent_dtype = self._pick_empty_latent_dtype(dtype, latent_precision)
            empty_latent = torch.zeros(
                [B, 16, total_latents, H, W],
                device=comfy.model_management.intermediate_device(),
                dtype=empty_latent_dtype,
            )

            # --- 1. Estrazione motion tail da prev_samples ---
            motion_latent = None
            T_motion = 0
            has_prev = bool(use_prev_samples) and prev_samples is not None and motion_latent_count != 0

            if has_prev:
                motion_latent = prev_samples["samples"][:, :, -motion_latent_count:]
                T_motion = motion_latent.shape[2]

            # --- 2. DC Drift Correction ---
            # Corrects per-channel mean shift (DC drift) accumulated across chained segments.
            # Each KSampler output slightly shifts the per-channel mean of latents;
            # over multiple segments this accumulates → progressive visual degradation.
            #
            # Algorithm:
            #   drift  = mean(motion_tail, dims=T,H,W) − mean(anchor, dims=T,H,W)  [B,C,1,1,1]
            #   drift  = delta_max·tanh(drift/delta_max)  if delta_max > 0  (soft-clamp)
            #   motion_tail -= drift * latent_refresh
            #
            # This ONLY subtracts the mean offset per channel — spatial structure and motion
            # are completely preserved (translational invariance of mean subtraction).
            _lr = float(latent_refresh)
            _dm = float(delta_max)
            if motion_latent is not None and _lr > 0:
                with torch.no_grad():
                    # Per-channel mean over T, H, W → [B, C, 1, 1, 1]
                    motion_mean  = motion_latent.mean(dim=(2, 3, 4), keepdim=True)
                    anchor_mean  = anchor_latent.mean(dim=(2, 3, 4), keepdim=True)
                    dc_drift     = motion_mean - anchor_mean  # signed per-channel offset

                    if diagnostic_log:
                        # Log per-channel drift (16 channels) for full visibility
                        drift_vals = dc_drift[0, :, 0, 0, 0]  # [C]
                        drift_abs  = drift_vals.abs()
                        self._log.info(
                            "[WanImageMotion][dc_drift] BEFORE correction — "
                            "per-channel drift: max=%.4f mean=%.4f rms=%.4f "
                            "| motion_mean=[%.3f..%.3f] anchor_mean=[%.3f..%.3f]",
                            float(drift_abs.max()),
                            float(drift_vals.mean()),
                            float((drift_vals ** 2).mean().sqrt()),
                            float(motion_mean.min()), float(motion_mean.max()),
                            float(anchor_mean.min()), float(anchor_mean.max()),
                        )
                        for ci in range(min(16, int(drift_vals.shape[0]))):
                            self._log.info(
                                "[WanImageMotion][dc_drift]   ch%02d: drift=% .4f  motion_mean=% .4f  anchor_mean=% .4f",
                                ci, float(drift_vals[ci]),
                                float(motion_mean[0, ci, 0, 0, 0]),
                                float(anchor_mean[0, ci, 0, 0, 0]),
                            )

                    # Optional soft-clamp on the drift itself (prevents over-correction)
                    if _dm > 0:
                        dc_drift = _dm * torch.tanh(dc_drift / _dm)

                    # Apply correction: subtract drift * strength
                    correction = dc_drift * _lr
                    motion_latent = motion_latent - correction

                    if diagnostic_log:
                        new_mean = motion_latent.mean(dim=(2, 3, 4), keepdim=True)
                        residual = (new_mean - anchor_mean)[0, :, 0, 0, 0]
                        self._log.info(
                            "[WanImageMotion][dc_drift] AFTER  correction — "
                            "residual drift: max=%.4f mean=%.4f | correction_strength=%.2f delta_max=%.2f",
                            float(residual.abs().max()),
                            float(residual.mean()),
                            _lr, _dm,
                        )
                    else:
                        self._log.info(
                            "[WanImageMotion] DC drift correction active: strength=%.2f delta_max=%.2f "
                            "| drift_max=%.4f drift_mean=%.4f",
                            _lr, _dm,
                            float(dc_drift.abs().max()),
                            float(dc_drift.mean()),
                        )

            # --- 3. Costruzione del condizionamento ---
            if motion_latent is None:
                padding_size = total_latents - T_anchor
                image_cond_latent = anchor_latent
            else:
                padding_size = total_latents - T_anchor - T_motion
                image_cond_latent = torch.cat([anchor_latent, motion_latent], dim=2)

            padding_size = max(0, padding_size)
            if padding_size > 0:
                padding = torch.zeros(B, C, padding_size, H, W, dtype=dtype, device=device)
                padding = comfy.latent_formats.Wan21().process_out(padding)
                image_cond_latent = torch.cat([image_cond_latent, padding], dim=2)

            # FLF/SVI reference behavior: ensure exact temporal length.
            if image_cond_latent.shape[2] > total_latents:
                image_cond_latent = image_cond_latent[:, :, :total_latents]
            elif image_cond_latent.shape[2] < total_latents:
                # Safety: if something went off, truncate/pad has already handled it,
                # but keep a hard guard.
                image_cond_latent = image_cond_latent[:, :, :total_latents]

            # Apply motion amplitude before injecting into conditioning
            # IMPORTANT: do not apply motion scaling to *pure padding*.
            # Padding slots are zeros and should remain sampler-driven; modifying them can cause
            # near-static clips and segment disconnect artifacts.
            if include_padding_in_motion and not has_prev:
                self._log.info(
                    "[WanImageMotion] include_padding_in_motion=True but prev_samples is not connected; "
                    "skipping padding boost to avoid static/locked-looking clips."
                )

            if include_padding_in_motion and has_prev and (T_anchor + T_motion) > 1:
                effective_latents = total_latents
            else:
                effective_latents = min(total_latents, T_anchor + T_motion)

            motion_mode_effective = motion_mode

            # --- Logging (concise but informative) ---
            try:
                free_vram, total_vram = comfy.model_management.get_free_memory(device)
            except Exception:
                free_vram, total_vram = None, None

            self._log.info(
                "[WanImageMotion] length=%s -> total_latents=%s | motion=%s | mode=%s | vram_profile=%s | latent_precision=%s | add_reference_latents=%s | include_padding_in_motion=%s",
                length,
                total_latents,
                motion,
                motion_mode,
                vram_profile,
                latent_precision,
                add_reference_latents,
                include_padding_in_motion,
            )
            self._log.info(
                "[WanImageMotion] anchor: B=%s C=%s T=%s H=%s W=%s dtype=%s device=%s | prev=%s motion_latent_count=%s T_motion=%s | padding_size=%s",
                B,
                C,
                T_anchor,
                H,
                W,
                str(dtype).replace("torch.", ""),
                str(device),
                has_prev,
                motion_latent_count,
                T_motion,
                padding_size,
            )
            if prev_samples is not None and not use_prev_samples:
                self._log.info("[WanImageMotion] prev_samples connected but use_prev_samples=False — prev ignored.")
            if free_vram is not None:
                self._log.info("[WanImageMotion] free_vram=%s total_vram=%s", free_vram, total_vram)

            if motion_mode_effective == "motion_only (prev_samples)":
                motion_start = T_anchor
                motion_end = min(T_anchor + T_motion, effective_latents)
            else:
                motion_start = 1
                motion_end = effective_latents
            
            motion_frames_count = max(0, motion_end - motion_start)
            
            self._log.info(
                "[WanImageMotion] motion_range=[%s:%s] (effective_latents=%s) padding_included=%s",
                motion_start,
                motion_end,
                effective_latents,
                include_padding_in_motion,
            )
            
            if motion_frames_count == 0:
                self._log.warning(
                    "[WanImageMotion] ⚠️ WARNING: motion_range is EMPTY (no frames will be modified). "
                    "To apply motion boost, either enable 'include_padding_in_motion=True' or provide prev_samples with motion_latent_count > 0."
                )
            else:
                self._log.info(
                    "[WanImageMotion] ✓ Motion boost will be applied to %s frame(s) with amplitude=%.2f",
                    motion_frames_count,
                    motion,
                )
            
            image_cond_latent = self._apply_motion_amplitude(
                image_cond_latent,
                real_latents=effective_latents,
                anchor_latents=T_anchor,
                motion_latents=T_motion,
                motion_amplitude=motion,
                motion_mode=motion_mode_effective,
                vram_profile=vram_profile,
                safety_preset=safety_preset,
            )

            if diagnostic_log:
                try:
                    x = image_cond_latent
                    x_f = x.float()
                    t = int(x.shape[2])
                    x_min = float(x_f.min().item())
                    x_max = float(x_f.max().item())
                    x_mean = float(x_f.mean().item())
                    # std can be slightly expensive, but only runs when enabled
                    x_std = float(x_f.std(unbiased=False).item())
                    self._log.info(
                        "[WanImageMotion][diag] concat_latent_image stats: T=%s min=%.4f max=%.4f mean=%.4f std=%.4f",
                        t, x_min, x_max, x_mean, x_std,
                    )

                    # Per-slot mean abs diff vs first slot: helps spot 'everything identical'.
                    base = x_f[:, :, 0:1]
                    diffs = (x_f - base).abs().mean(dim=(0, 1, 3, 4))  # [T]
                    head_n = min(6, t)
                    tail_n = min(6, t)
                    head = [float(v) for v in diffs[:head_n].tolist()]
                    tail = [float(v) for v in diffs[-tail_n:].tolist()]
                    self._log.info(
                        "[WanImageMotion][diag] mean|Δ| vs t0: head=%s tail=%s",
                        head, tail,
                    )
                except Exception as e:
                    self._log.warning("[WanImageMotion][diag] failed to compute diagnostics: %s", e)

            # Conditioning tensors are always kept in fp32 for numerical stability.
            # WAN21.concat_cond casts them to xc.dtype (model inference dtype) via
            # cast_to_device anyway, so keeping them fp32 is safe regardless of latent_precision.
            mask = torch.ones((1, 1, empty_latent.shape[2], H, W), device=device, dtype=torch.float32)
            lock_start_slots_i = int(max(0, min(16, lock_start_slots)))
            if lock_start_slots_i > 0:
                mask[:, :, :lock_start_slots_i] = 0.0

            # Ensure image_cond_latent is fp32 for model accuracy.
            if image_cond_latent.dtype != torch.float32:
                image_cond_latent = image_cond_latent.float()

            # Device safety: keep conditioning tensors on the same device as the sampler latent.
            # Some workflows produce CPU latents; moving avoids implicit transfers / conditioning issues.
            cond_device = empty_latent.device
            if image_cond_latent.device != cond_device:
                if diagnostic_log:
                    self._log.info(
                        "[WanImageMotion][diag] moving conditioning tensors %s -> %s",
                        str(image_cond_latent.device),
                        str(cond_device),
                    )
                image_cond_latent = image_cond_latent.to(cond_device)
                mask = mask.to(cond_device)

            positive = node_helpers.conditioning_set_values(
                positive, {"concat_latent_image": image_cond_latent, "concat_mask": mask}
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"concat_latent_image": image_cond_latent, "concat_mask": mask}
            )

            if add_reference_latents:
                # Use the first anchor latent frame as a reference latent.
                # This is lightweight (single frame) and avoids VAE usage.
                ref_latent = anchor_latent[:, :, 0:1]
                positive = node_helpers.conditioning_set_values(
                    positive, {"reference_latents": [ref_latent]}, append=True
                )
                negative = node_helpers.conditioning_set_values(
                    negative, {"reference_latents": [torch.zeros_like(ref_latent)]}, append=True
                )

            # Apply CLIPVision semantic conditioning if provided.
            # Anchors the model's cross-attention to the subject's visual embedding,
            # reducing identity drift across chained segments.
            if clip_vision_output is not None:
                positive = node_helpers.conditioning_set_values(
                    positive, {"clip_vision_output": clip_vision_output}
                )
                negative = node_helpers.conditioning_set_values(
                    negative, {"clip_vision_output": clip_vision_output}
                )

            out_latent = {"samples": empty_latent}
            return (positive, negative, out_latent)


class WanImageMotionPro:
    """WanImageMotionPro

    Combines IAMCCS_WanImageMotion motion amplitude control with FLF-style
    (First/Last Frame) hard lock via optional end_samples.

    Behavior:
    - Start: anchor_samples + optional motion tail from prev_samples.
    - Motion: apply motion amplitude scaling (VRAM-aware) as in IAMCCS_WanImageMotion.
    - End: overwrite last temporal latent slots with end_samples, then lock them
      via concat_mask (FLF-style control).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Keep the same socket-style inputs as WanImageToVideoSVIProFLF
                # so existing FLF workflows can be migrated with minimal friction.
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "length": ("INT", {"default": 81, "min": 1, "max": 16384, "step": 4}),
                "anchor_samples": ("LATENT",),
                # Match FLF reference node range.
                "motion_latent_count": ("INT", {"default": 1, "min": 0, "max": 16, "step": 1}),
                "motion": ("FLOAT", {"default": 1.15, "min": 1.0, "max": 2.0, "step": 0.05}),
                "motion_mode": (
                    [
                        "motion_only (prev_samples)",
                        "all_nonfirst (anchor+motion)",
                    ],
                    {"default": "motion_only (prev_samples)"},
                ),
                "add_reference_latents": ("BOOLEAN", {"default": False}),
                "vram_profile": (
                    [
                        "normal",
                        "chunked_blocks_2",
                        "chunked_blocks_4",
                        "loop_per_frame (lowest_vram)",
                        "cpu_offload (slowest)",
                    ],
                    {"default": "normal"},
                ),
                "include_padding_in_motion": ("BOOLEAN", {"default": False}),
                # Keep this at the end to preserve existing widgets_values indexing in saved workflows.
                "safety_preset": (
                    [
                        "base",
                        "safe",
                        "safer",
                        "legacy",
                    ],
                    {
                        "default": "safe",
                        "tooltip": (
                            "base   : 1:1 with WanImageToVideoSVIProFLF — no amplitude processing at all. "
                            "safe   : per-channel stabilisation + tanh limiter + 2-frame ramp (active at any motion value, recommended default). "
                            "safer  : same as safe but tighter limiter + longer ramp (best above motion=1.5). "
                            "legacy : hard clamp ±6 + frame-scalar diff centering, mirrors PainterI2V algorithm."
                        ),
                    },
                ),
                # --- End-frame controls (keep at end to preserve widgets_values ordering) ---
                "use_end_frame": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "When False, end_samples is ignored even if the wire is connected. "
                            "Use this to safely bypass the end-frame encoder without disconnecting the node, "
                            "preventing ping-pong artifacts."
                        ),
                    },
                ),
                "end_transition_frames": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 32,
                        "step": 1,
                        "tooltip": (
                            "⚠️ DEPRECATED — keep at 0. "
                            "Values > 0 blend padding latents toward the end frame in latent space, "
                            "which produces static/frozen output because VAE latent interpolation is not linear. "
                            "0 = hard cut, identical to the original FLF node behavior."
                        ),
                    },
                ),
                "end_lock_slots": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 16,
                        "step": 1,
                        "tooltip": (
                            "How many final latent slots to hard-lock to end_samples. "
                            "Default 1 = only the very last slot (~last 4 video frames) is locked, "
                            "so the end frame appears exactly at the last frame defined by 'length'. "
                            "Increase only if a single locked slot is not enough for convergence. "
                            "Set 16 for 1:1 parity with the original FLF node (locks min(T_end, total_latents))."
                        ),
                    },
                ),
                # Append-only widget for backward compatibility.
                "lock_start_slots": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 16,
                        "step": 1,
                        "tooltip": (
                            "How many initial latent slots to hard-lock to the start image via concat_mask. "
                            "Default 1 (≈ first 4 video frames) matches FLF-style start anchoring. "
                            "Set 0 to allow motion immediately from the first video frame."
                        ),
                    },
                ),
                # Append-only: diagnostic logging toggle (does not change outputs).
                "diagnostic_log": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "When enabled, prints compact diagnostics about concat_latent_image/mask and motion/end-lock ranges. "
                            "Useful to debug 'static clip' or segment discontinuities."
                        ),
                    },
                ),
                # Append-only: allow keeping prev_samples wired (autolink) but ignoring it.
                "use_prev_samples": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "When False, prev_samples is ignored even if the wire is connected. "
                            "Use this on the first segment when autolink forces prev_samples to be connected."
                        ),
                    },
                ),
                # Append-only: end-frame overshoot — extends the internal generation window so the
                # hard-locked end frame lands beyond the visible output range, letting motion converge
                # toward the end image without freezing on it. Connect trim_slots output to a downstream
                # Cut Latent Frames or Get Latent Range node to remove the extra slots after sampling.
                # 0 = disabled (original behavior). 1 recommended when end-frame freeze breaks continuity.
                "end_overshoot_slots": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 8,
                        "step": 1,
                        "tooltip": (
                            "Extends the internal generation window by this many latent slots when the end lock "
                            "is active (use_end_frame=True + end_samples connected). "
                            "The end image is locked at the tail of the EXTENDED range, so the visible clip "
                            "converges toward the end frame without hard-freezing on it. "
                            "Use the trim_slots output to cut the overshoot frames after sampling. "
                            "0 = disabled, original behavior. 1 = +4 video frames (recommended). "
                            "2 = +8 video frames (looser convergence). "
                            "NOTE: If you have KJNodes WAN attention optimizations (FETA / wrapped_attention) enabled, "
                            "some versions may crash with an einops shape mismatch when overshoot > 0. "
                            "Workaround: set end_overshoot_slots=0 or disable/update that KJNodes optimization."
                        ),
                    },
                ),
                # DC Drift Correction: corrects per-channel mean shift accumulated
                # across chained segments. Does NOT touch spatial structure or motion.
                # Algorithm:  drift = mean(motion_tail, dims=T,H,W) − mean(anchor, dims=T,H,W)
                #             motion_tail -= drift * latent_refresh
                "latent_refresh": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "DC drift correction strength (0=disabled, 1=full correction). "
                            "Corrects the per-channel mean shift between motion tail and anchor latent. "
                            "Does not touch spatial structure or motion. "
                            "Start with 0.5 for chains of 3+ segments. Enable diagnostic_log to verify."
                        ),
                    },
                ),
                # Soft-clamp on the DC drift itself.
                # 0 = disabled. Use 0.5–1.0 to prevent over-correction on large motion segments.
                "delta_max": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.1,
                        "tooltip": (
                            "Soft-clamp on DC drift magnitude (0=disabled). "
                            "Caps how much drift correction can be applied per channel. "
                            "Use 0.5–1.0 to prevent over-correction on large motion segments."
                        ),
                    },
                ),
                # ---------------------------------------------------------------
                # clip_vision_mode — controls how start/end CLIP embeddings are merged.
                #
                # ⚠️  ROOT CAUSE OF CROSSFADE ARTIFACTS:
                # When clip_vision_start_image (image A) ≠ anchor_samples (image B),
                # the concatenated [A_tokens, B_tokens] embedding is applied via
                # GLOBAL cross-attention (clip_fea) to ALL temporal slots, including
                # the free (unmasked) middle slots → model blends A+B visually →
                # crossfade even when concat_latent_image start/end are both B.
                #
                # Mode guide:
                #   "auto"         : current behavior – concatenate start+end tokens.
                #                    Correct only for true A→B FLF (anchor=A, end=B,
                #                    clip_start=A, clip_end=B). Use "end_only" otherwise.
                #   "end_only"     : use ONLY clip_vision_end_image. Best mode for SVI
                #                    Pro continuation where anchor and end are the SAME
                #                    image but prev_samples drives motion from a different
                #                    scene. Prevents crossfade bleed from clip_start.
                #   "start_only"   : use ONLY clip_vision_start_image.
                #   "none"         : do NOT inject any clip_vision conditioning even if
                #                    sockets are connected. Emergency bypass.
                # ---------------------------------------------------------------
                "clip_vision_mode": (
                    [
                        "auto",
                        "end_only",
                        "start_only",
                        "none",
                    ],
                    {
                        "default": "auto",
                        "tooltip": (
                            "Controls how clip_vision_start_image / clip_vision_end_image are merged.\n"
                            "auto       : concatenate start+end tokens (original FLF behavior). "
                            "Correct ONLY when anchor=A and end=B (true A→B transition). "
                            "end_only   : use only clip_vision_end embedding. "
                            "RECOMMENDED for SVI Pro continuation where anchor=end=same image "
                            "but prev_samples comes from a different scene — prevents crossfade bleed. "
                            "start_only : use only clip_vision_start embedding. "
                            "none       : skip clip_vision injection entirely (bypass)."
                        ),
                    },
                ),
                # Append-only strict compatibility path for the original FLF+SVI behavior.
                # This bypasses MotionPro enhancements and builds conditioning like
                # WanImageToVideoSVIProFLF: full anchor latent + optional motion tail,
                # plain 1-channel slot mask, no clip vision, no overshoot, no motion scaling.
                "raw_mode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Strict compatibility mode for the original WanImageToVideoSVIProFLF behavior. "
                            "When enabled, WanImageMotionPro bypasses advanced features (motion amplitude, "
                            "DC drift correction, overshoot, clip vision routing, reference latents, slot caps) "
                            "and builds conditioning 1:1 like the original FLF+SVI node. "
                            "Use this to benchmark or to get the baseline first/last frame behavior back."
                        ),
                    },
                ),
                "prev_samples_profile": (
                    [
                        "direct",
                        "mixed_svi_safe",
                        "mixed_svi_aggressive",
                    ],
                    {
                        "default": "direct",
                        "tooltip": (
                            "High-level handling profile for prev_samples. "
                            "direct: use the explicit prev tail controls exactly as set. "
                            "mixed_svi_safe: intended for SVI-to-Pro chains; ignores the last prev slot and still uses the latest usable tail. "
                            "mixed_svi_aggressive: intended for harder SVI-to-Pro chains; ignores the last prev slot and takes the earliest usable tail segment instead of the latest."
                        ),
                    },
                ),
                "prev_skip_last_slots": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 16,
                        "step": 1,
                        "tooltip": (
                            "Ignore the last N temporal latent slots from prev_samples before extracting the motion tail. "
                            "Useful when prev_samples comes from a non-FLF generator and you want FLF-like continuation from a slightly earlier motion state."
                        ),
                    },
                ),
                "prev_tail_pick_mode": (
                    [
                        "latest",
                        "first_usable",
                    ],
                    {
                        "default": "latest",
                        "tooltip": (
                            "How to pick the motion tail from prev_samples after any tail trimming. "
                            "latest: use the last usable slots (default, original behavior). "
                            "first_usable: use the earliest usable slots after trimming, which can behave more like an FLF chain when SVI tails become unstable near the end."
                        ),
                    },
                ),
                "prev_anchor_blend": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Blend the extracted prev motion tail toward the first anchor slot before conditioning. "
                            "Helps when prev_samples comes from SVI and needs to behave more like a FLF-aligned predecessor."
                        ),
                    },
                ),
            },
            "optional": {
                # prev_samples is optional – mirrors original FLF node and IAMCCS_WanImageMotion.
                # apply() already handles None gracefully.
                "prev_samples": ("LATENT",),
                "end_samples": ("LATENT",),
                "clip_vision_start_image": ("CLIP_VISION_OUTPUT",),
                "clip_vision_end_image": ("CLIP_VISION_OUTPUT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("positive", "negative", "latent", "trim_slots")
    FUNCTION = "apply"
    CATEGORY = "IAMCCS/Wan/_Legacy"

    _log = logging.getLogger("IAMCCS.WanImageMotionPro")

    def _build_raw_mode_conditioning(
        self,
        *,
        positive,
        negative,
        length,
        anchor_samples,
        motion_latent_count,
        prev_samples,
        end_samples,
        prev_samples_profile="direct",
        prev_tail_pick_mode="latest",
        prev_skip_last_slots=0,
        prev_anchor_blend=0.0,
    ):
        # Intentional copy of the original WanImageToVideoSVIProFLF semantics.
        anchor_latent = anchor_samples["samples"].clone()
        B, C, T_anchor, H, W = anchor_latent.shape

        total_latents = (length - 1) // 4 + 1
        device = anchor_latent.device
        dtype = anchor_latent.dtype

        resolved_tail_pick_mode, resolved_skip_last_slots, resolved_anchor_blend = _resolve_prev_tail_profile(
            prev_samples_profile,
            tail_pick_mode=prev_tail_pick_mode,
            skip_last_slots=prev_skip_last_slots,
            anchor_blend=prev_anchor_blend,
        )

        motion_latent, motion_count, t_prev, usable_t = _extract_prev_motion_tail(
            prev_samples,
            motion_latent_count,
            tail_pick_mode=resolved_tail_pick_mode,
            skip_last_slots=resolved_skip_last_slots,
            anchor_reference=anchor_latent,
            anchor_blend=resolved_anchor_blend,
        )

        if motion_latent is None:
            base_latent = anchor_latent
        else:
            base_latent = torch.cat([anchor_latent, motion_latent], dim=2)

        T_base = base_latent.shape[2]
        padding_size = max(total_latents - T_base, 0)

        if padding_size > 0:
            padding = torch.zeros(B, C, padding_size, H, W, dtype=dtype, device=device)
            padding = comfy.latent_formats.Wan21().process_out(padding)
            image_cond_latent = torch.cat([base_latent, padding], dim=2)
        else:
            image_cond_latent = base_latent[:, :, :total_latents]

        if image_cond_latent.shape[2] != total_latents:
            image_cond_latent = image_cond_latent[:, :, :total_latents]

        end_t_fix = 0
        if end_samples is not None:
            end_latent = end_samples["samples"].clone()

            if end_latent.shape[0] == 1 and B > 1:
                end_latent = end_latent.repeat(B, 1, 1, 1, 1)

            if (
                end_latent.shape[1] == C
                and end_latent.shape[3] == H
                and end_latent.shape[4] == W
            ):
                T_end = end_latent.shape[2]
                end_t_fix = min(T_end, total_latents)

                if end_t_fix > 0:
                    image_cond_latent[:, :, -end_t_fix:] = end_latent[:, :, -end_t_fix:]
            else:
                end_t_fix = 0

        empty_latent = torch.zeros(
            [B, 16, total_latents, H, W],
            device=comfy.model_management.intermediate_device(),
        )

        mask = torch.ones((1, 1, total_latents, H, W), device=device, dtype=dtype)
        mask[:, :, :1] = 0.0

        if end_t_fix > 0:
            mask[:, :, -end_t_fix:] = 0.0

        positive = node_helpers.conditioning_set_values(
            positive, {"concat_latent_image": image_cond_latent, "concat_mask": mask}
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"concat_latent_image": image_cond_latent, "concat_mask": mask}
        )

        self._log.info(
            "[WanImageMotionPro][raw_mode] active: original FLF+SVI semantics | total_latents=%s end_t_fix=%s motion_latent_count=%s prev=%s prev_T=%s usable_prev_T=%s prev_samples_profile=%s prev_tail_pick_mode=%s prev_skip_last_slots=%s prev_anchor_blend=%.2f",
            total_latents,
            end_t_fix,
            motion_latent_count,
            prev_samples is not None,
            t_prev,
            usable_t,
            str(prev_samples_profile),
            str(resolved_tail_pick_mode),
            int(resolved_skip_last_slots),
            float(resolved_anchor_blend),
        )
        out_latent = {"samples": empty_latent}
        return (positive, negative, out_latent, 0)

    def _pick_empty_latent_dtype(self, anchor_dtype: torch.dtype, latent_precision: str) -> torch.dtype:
        # Keep 1:1 behavior with IAMCCS_WanImageMotion.
        if latent_precision == "normal":
            return anchor_dtype
        if latent_precision == "auto":
            return torch.float32
        if latent_precision == "fp32":
            return torch.float32
        if latent_precision == "fp16":
            return torch.float16
        return anchor_dtype

    def _apply_motion_amplitude(
        self,
        image_cond_latent: torch.Tensor,
        *,
        real_latents: int,
        anchor_latents: int,
        motion_latents: int,
        motion_amplitude: float,
        motion_mode: str,
        vram_profile: str,
        safety_preset: str = "safe",
    ) -> torch.Tensor:
        # Reuse the exact logic from IAMCCS_WanImageMotion (copy to keep node self-contained).
        preset = _preset_params(safety_preset, motion_amplitude)

        # "base" preset: 1:1 with reference nodes — no processing at all.
        if not preset.get("enabled", True):
            return image_cond_latent

        if motion_amplitude is None or motion_amplitude <= 1.0:
            return image_cond_latent

        if image_cond_latent.shape[2] <= 1:
            return image_cond_latent

        base_latent = image_cond_latent[:, :, 0:1]

        def _scale_slice_gpu(view_slice: torch.Tensor, *, gain: torch.Tensor | float) -> torch.Tensor:
            with torch.no_grad():
                diff = view_slice - base_latent
                diff_centered, diff_mean = _center_diff(diff, mean_mode=preset["mean_mode"])
                diff_centered.mul_(gain)
                out_local = diff_centered.add_(diff_mean).add_(base_latent)
                _apply_soft_limiter(out_local, mode=preset["limiter_mode"], limit=float(preset["limiter_limit"]))
                return out_local

        def _gain_weights(start: int, end: int) -> torch.Tensor | float:
            ramp_frames = int(preset["ramp_frames"])
            if ramp_frames <= 0:
                return float(motion_amplitude)
            tcount = max(0, end - start)
            if tcount <= 0:
                return float(motion_amplitude)
            ramp = min(ramp_frames, tcount)
            w = torch.ones((tcount,), device=image_cond_latent.device, dtype=image_cond_latent.dtype)
            if ramp > 0:
                x = torch.linspace(0.0, 1.0, steps=ramp, device=w.device, dtype=w.dtype)
                w[:ramp] = _smoothstep(x)
            gain = 1.0 + (motion_amplitude - 1.0) * w
            return gain.view(1, 1, tcount, 1, 1)

        def _apply_to_range(start: int, end: int) -> torch.Tensor:
            if end <= start:
                return out

            if vram_profile == "normal":
                gain = _gain_weights(start, end)
                out[:, :, start:end] = _scale_slice_gpu(out[:, :, start:end], gain=gain)
                return out

            if vram_profile in ("chunked_blocks_2", "chunked_blocks_4"):
                block = 2 if vram_profile == "chunked_blocks_2" else 4
                t = start
                while t < end:
                    t2 = min(end, t + block)
                    gain = _gain_weights(t, t2)
                    out[:, :, t:t2] = _scale_slice_gpu(out[:, :, t:t2], gain=gain)
                    t = t2
                return out

            if vram_profile == "loop_per_frame (lowest_vram)":
                for t in range(start, end):
                    gain = _gain_weights(t, t + 1)
                    out[:, :, t:t + 1] = _scale_slice_gpu(out[:, :, t:t + 1], gain=gain)
                return out

            if vram_profile == "cpu_offload (slowest)":
                with torch.no_grad():
                    device = out.device
                    base_cpu = base_latent.detach().to("cpu")
                    slice_cpu = out[:, :, start:end].detach().to("cpu")
                    diff = slice_cpu - base_cpu
                    diff_centered, diff_mean = _center_diff(diff, mean_mode=preset["mean_mode"])
                    ramp_frames = int(preset["ramp_frames"])
                    tcount = max(0, end - start)
                    if ramp_frames > 0 and tcount > 0:
                        ramp = min(ramp_frames, tcount)
                        w = torch.ones((tcount,), device=diff_centered.device, dtype=diff_centered.dtype)
                        x = torch.linspace(0.0, 1.0, steps=ramp, device=w.device, dtype=w.dtype)
                        w[:ramp] = _smoothstep(x)
                        gain = (1.0 + (motion_amplitude - 1.0) * w).view(1, 1, tcount, 1, 1)
                    else:
                        gain = float(motion_amplitude)
                    diff_centered.mul_(gain)
                    out_cpu = diff_centered.add_(diff_mean).add_(base_cpu)
                    _apply_soft_limiter(out_cpu, mode=preset["limiter_mode"], limit=float(preset["limiter_limit"]))
                    out[:, :, start:end] = out_cpu.to(device)
                return out

            gain = _gain_weights(start, end)
            out[:, :, start:end] = _scale_slice_gpu(out[:, :, start:end], gain=gain)
            return out

        real_latents = max(0, min(real_latents, image_cond_latent.shape[2]))
        if real_latents <= 1:
            return image_cond_latent

        out = image_cond_latent

        if motion_mode == "motion_only (prev_samples)":
            if motion_latents <= 0:
                return out

            start = anchor_latents
            end = min(anchor_latents + motion_latents, real_latents)
            if end <= start:
                return out

            return _apply_to_range(start, end)

        start = 1
        end = real_latents
        return _apply_to_range(start, end)

    def apply(
        self,
        positive,
        negative,
        length,
        anchor_samples,
        motion_latent_count,
        motion,
        motion_mode,
        add_reference_latents,
        latent_precision,
        vram_profile,
        include_padding_in_motion,
        safety_preset="safe",
        use_end_frame=True,
        end_transition_frames=4,
        end_lock_slots=1,
        lock_start_slots=1,
        diagnostic_log=False,
        use_prev_samples=True,
        end_overshoot_slots=0,
        latent_refresh=0.0,
        delta_max=0.0,
        clip_vision_mode="auto",
        raw_mode=False,
        prev_samples_profile="direct",
        prev_tail_pick_mode="latest",
        prev_skip_last_slots=0,
        prev_anchor_blend=0.0,
        prev_samples=None,
        end_samples=None,
        clip_vision_start_image=None,
        clip_vision_end_image=None,
        **_kwargs,
    ):
        with torch.no_grad():
            if raw_mode:
                return self._build_raw_mode_conditioning(
                    positive=positive,
                    negative=negative,
                    length=length,
                    anchor_samples=anchor_samples,
                    motion_latent_count=motion_latent_count,
                    prev_samples=prev_samples if use_prev_samples else None,
                    end_samples=end_samples if use_end_frame else None,
                    prev_samples_profile=prev_samples_profile,
                    prev_tail_pick_mode=prev_tail_pick_mode,
                    prev_skip_last_slots=prev_skip_last_slots,
                    prev_anchor_blend=prev_anchor_blend,
                )

            # Clone to prevent in-place motion amplitude writes from corrupting the caller's tensor.
            anchor_latent = anchor_samples["samples"].clone()

            B, C, T_anchor, H, W = anchor_latent.shape

            lock_start_slots_i = int(max(0, min(16, lock_start_slots)))

            # FLF core parity: when using a single-image start anchor, the core node constrains
            # only the actual start frame. A Wan VAEEncode of a still image often expands to more
            # than one temporal latent slot; keeping the full block here makes the bridge hold on
            # to A for too long and often degenerates into a dissolve. When FLF end control is
            # active, cap the start conditioning block to the number of locked start slots.
            anchor_slots_for_cond = T_anchor
            if use_end_frame and end_samples is not None and T_anchor > 1 and lock_start_slots_i <= 1:
                anchor_slots_for_cond = 1
                self._log.info(
                    "[WanImageMotionPro] FLF start normalization: T_anchor=%s -> using first %s slot for conditioning.",
                    T_anchor,
                    anchor_slots_for_cond,
                )

            anchor_cond_latent = anchor_latent[:, :, :anchor_slots_for_cond]

            total_latents = (length - 1) // 4 + 1

            # Overshoot: when end lock is active and end_overshoot_slots > 0, extend the internal
            # generation window so the end-locked zone lands beyond what the sampler actually produces
            # as "visible" output. trim_slots (4th output) tells the downstream crop node how many
            # latent slots to remove from the tail after sampling.
            _overshoot = (
                int(end_overshoot_slots)
                if (use_end_frame and end_samples is not None and int(end_overshoot_slots) > 0)
                else 0
            )
            total_latents_gen = total_latents + _overshoot

            if _overshoot > 0:
                self._log.warning(
                    "[WanImageMotionPro] end_overshoot_slots=%s enabled (total_latents=%s -> gen=%s). "
                    "If you use KJNodes WAN attention optimizations (FETA / wrapped_attention) and hit an einops shape mismatch, "
                    "set end_overshoot_slots=0 or disable/update that optimization.",
                    int(end_overshoot_slots),
                    total_latents,
                    total_latents_gen,
                )

            device = anchor_latent.device
            dtype = anchor_latent.dtype

            empty_latent_dtype = self._pick_empty_latent_dtype(dtype, latent_precision)
            empty_latent = torch.zeros(
                [B, 16, total_latents_gen, H, W],
                device=comfy.model_management.intermediate_device(),
                dtype=empty_latent_dtype,
            )

            # --- 1. Estrazione motion tail da prev_samples ---
            motion_latent = None
            T_motion = 0
            t_prev = 0
            usable_prev_t = 0
            resolved_tail_pick_mode, resolved_skip_last_slots, resolved_anchor_blend = _resolve_prev_tail_profile(
                prev_samples_profile,
                tail_pick_mode=prev_tail_pick_mode,
                skip_last_slots=prev_skip_last_slots,
                anchor_blend=prev_anchor_blend,
            )
            # In the original FLF node, prev_samples is a required socket.
            # If a workflow leaves it disconnected, ComfyUI may pass None.
            has_prev = bool(use_prev_samples) and prev_samples is not None and motion_latent_count != 0

            if has_prev:
                motion_latent, T_motion, t_prev, usable_prev_t = _extract_prev_motion_tail(
                    prev_samples,
                    motion_latent_count,
                    tail_pick_mode=resolved_tail_pick_mode,
                    skip_last_slots=resolved_skip_last_slots,
                    anchor_reference=anchor_latent,
                    anchor_blend=resolved_anchor_blend,
                )
                has_prev = motion_latent is not None and T_motion > 0

            # --- 2. DC Drift Correction ---
            # Same algorithm as IAMCCS_WanImageMotion: corrects per-channel mean shift.
            # drift = mean(motion_tail, T,H,W) − mean(anchor, T,H,W)  → [B,C,1,1,1]
            # motion_tail -= drift * latent_refresh  (spatial structure untouched)
            _lr = float(latent_refresh)
            _dm = float(delta_max)
            if motion_latent is not None and _lr > 0:
                with torch.no_grad():
                    motion_mean  = motion_latent.mean(dim=(2, 3, 4), keepdim=True)
                    anchor_mean  = anchor_latent.mean(dim=(2, 3, 4), keepdim=True)
                    dc_drift     = motion_mean - anchor_mean

                    if diagnostic_log:
                        drift_vals = dc_drift[0, :, 0, 0, 0]
                        drift_abs  = drift_vals.abs()
                        self._log.info(
                            "[WanImageMotionPro][dc_drift] BEFORE correction — "
                            "per-channel drift: max=%.4f mean=%.4f rms=%.4f "
                            "| motion_mean=[%.3f..%.3f] anchor_mean=[%.3f..%.3f]",
                            float(drift_abs.max()),
                            float(drift_vals.mean()),
                            float((drift_vals ** 2).mean().sqrt()),
                            float(motion_mean.min()), float(motion_mean.max()),
                            float(anchor_mean.min()), float(anchor_mean.max()),
                        )
                        for ci in range(min(16, int(drift_vals.shape[0]))):
                            self._log.info(
                                "[WanImageMotionPro][dc_drift]   ch%02d: drift=% .4f  motion_mean=% .4f  anchor_mean=% .4f",
                                ci, float(drift_vals[ci]),
                                float(motion_mean[0, ci, 0, 0, 0]),
                                float(anchor_mean[0, ci, 0, 0, 0]),
                            )

                    if _dm > 0:
                        dc_drift = _dm * torch.tanh(dc_drift / _dm)

                    correction = dc_drift * _lr
                    motion_latent = motion_latent - correction

                    if diagnostic_log:
                        new_mean = motion_latent.mean(dim=(2, 3, 4), keepdim=True)
                        residual = (new_mean - anchor_mean)[0, :, 0, 0, 0]
                        self._log.info(
                            "[WanImageMotionPro][dc_drift] AFTER  correction — "
                            "residual drift: max=%.4f mean=%.4f | correction_strength=%.2f delta_max=%.2f",
                            float(residual.abs().max()),
                            float(residual.mean()),
                            _lr, _dm,
                        )
                    else:
                        self._log.info(
                            "[WanImageMotionPro] DC drift correction active: strength=%.2f delta_max=%.2f "
                            "| drift_max=%.4f drift_mean=%.4f",
                            _lr, _dm,
                            float(dc_drift.abs().max()),
                            float(dc_drift.mean()),
                        )

            # --- 3. Costruzione del condizionamento ---
            if motion_latent is None:
                padding_size = total_latents_gen - anchor_slots_for_cond
                image_cond_latent = anchor_cond_latent
            else:
                padding_size = total_latents_gen - anchor_slots_for_cond - T_motion
                image_cond_latent = torch.cat([anchor_cond_latent, motion_latent], dim=2)

            padding_size = max(0, padding_size)
            if padding_size > 0:
                padding = torch.zeros(B, C, padding_size, H, W, dtype=dtype, device=device)
                padding = comfy.latent_formats.Wan21().process_out(padding)
                image_cond_latent = torch.cat([image_cond_latent, padding], dim=2)

            # FLF/SVI reference behavior: enforce exact temporal length (extended by overshoot).
            if image_cond_latent.shape[2] > total_latents_gen:
                image_cond_latent = image_cond_latent[:, :, :total_latents_gen]
            elif image_cond_latent.shape[2] < total_latents_gen:
                image_cond_latent = image_cond_latent[:, :, :total_latents_gen]

            # Pre-compute end_t_fix so we can exclude the end-locked zone from motion amplitude.
            # Motion should NOT touch slots that will be hard-locked to end_samples: scaling those
            # intermediate latents would generate noise that hurts the model's first→last interpolation.
            # Respect use_end_frame: if disabled, never cap the motion range.
            end_t_fix_early = 0
            if end_samples is not None and use_end_frame:
                _e = end_samples["samples"]
                if (
                    _e.shape[1] == C
                    and _e.shape[3] == H
                    and _e.shape[4] == W
                ):
                    # Cap to end_lock_slots so the motion range is not incorrectly
                    # shortened by the full VAE temporal size (which can be T>1 for a single image).
                    end_t_fix_early = min(_e.shape[2], total_latents_gen, end_lock_slots)

            # Motion boost applied before FLF overwrite.
            # Cap effective_latents so motion never reaches into the end-locked zone.
            # IMPORTANT: do not apply motion scaling to *pure padding*.
            # Padding slots are zeros and should remain sampler-driven.
            if include_padding_in_motion and not has_prev:
                self._log.info(
                    "[WanImageMotionPro] include_padding_in_motion=True but prev_samples is not connected; "
                    "skipping padding boost to avoid static/locked-looking clips."
                )

            if include_padding_in_motion and has_prev and (T_anchor + T_motion) > 1:
                effective_latents_base = total_latents_gen
            else:
                effective_latents_base = min(total_latents_gen, anchor_slots_for_cond + T_motion)
            effective_latents = max(1, min(effective_latents_base, total_latents_gen - end_t_fix_early))

            motion_mode_effective = motion_mode
            # NOTE: we intentionally do NOT auto-switch to all_nonfirst here.
            # Auto-switching would end up modifying padding slots on the first segment.

            try:
                free_vram, total_vram = comfy.model_management.get_free_memory(device)
            except Exception:
                free_vram, total_vram = None, None

            self._log.info(
                "[WanImageMotionPro] length=%s -> total_latents=%s (+overshoot=%s -> gen=%s) | motion=%s | mode=%s | vram_profile=%s | latent_precision=%s | add_reference_latents=%s | include_padding_in_motion=%s | use_end_frame=%s | end_transition_frames=%s | end_lock_slots=%s | lock_start_slots=%s",
                length,
                total_latents,
                _overshoot,
                total_latents_gen,
                motion,
                motion_mode,
                vram_profile,
                latent_precision,
                add_reference_latents,
                include_padding_in_motion,
                use_end_frame,
                end_transition_frames,
                end_lock_slots,
                lock_start_slots,
            )
            self._log.info(
                "[WanImageMotionPro] anchor: B=%s C=%s T=%s H=%s W=%s dtype=%s device=%s | prev=%s motion_latent_count=%s T_motion=%s prev_T=%s usable_prev_T=%s prev_samples_profile=%s prev_tail_pick_mode=%s prev_skip_last_slots=%s prev_anchor_blend=%.2f | padding_size=%s | end_samples=%s",
                B,
                C,
                T_anchor,
                H,
                W,
                str(dtype).replace("torch.", ""),
                str(device),
                has_prev,
                motion_latent_count,
                T_motion,
                t_prev,
                usable_prev_t,
                str(prev_samples_profile),
                str(resolved_tail_pick_mode),
                int(resolved_skip_last_slots),
                float(resolved_anchor_blend),
                padding_size,
                end_samples is not None,
            )
            if anchor_slots_for_cond != T_anchor:
                self._log.info(
                    "[WanImageMotionPro] start conditioning uses %s/%s anchor slots to match FLF first-frame behavior.",
                    anchor_slots_for_cond,
                    T_anchor,
                )
            if prev_samples is not None and not use_prev_samples:
                self._log.info("[WanImageMotionPro] prev_samples connected but use_prev_samples=False — prev ignored.")
            if free_vram is not None:
                self._log.info("[WanImageMotionPro] free_vram=%s total_vram=%s", free_vram, total_vram)

            if motion_mode_effective == "motion_only (prev_samples)":
                motion_start = anchor_slots_for_cond
                motion_end = min(anchor_slots_for_cond + T_motion, effective_latents)
            else:
                motion_start = 1
                motion_end = effective_latents

            motion_frames_count = max(0, motion_end - motion_start)
            self._log.info(
                "[WanImageMotionPro] motion_range=[%s:%s] (effective_latents=%s) padding_included=%s",
                motion_start,
                motion_end,
                effective_latents,
                include_padding_in_motion,
            )
            if motion_frames_count == 0:
                self._log.warning(
                    "[WanImageMotionPro] WARNING: motion_range is EMPTY (no frames will be modified). "
                    "Enable include_padding_in_motion or provide prev_samples with motion_latent_count > 0."
                )
            else:
                self._log.info(
                    "[WanImageMotionPro] Motion boost applies to %s frame(s) amplitude=%.2f",
                    motion_frames_count,
                    motion,
                )

            image_cond_latent = self._apply_motion_amplitude(
                image_cond_latent,
                real_latents=effective_latents,
                anchor_latents=anchor_slots_for_cond,
                motion_latents=T_motion,
                motion_amplitude=motion,
                motion_mode=motion_mode_effective,
                vram_profile=vram_profile,
                safety_preset=safety_preset,
            )

            if diagnostic_log:
                try:
                    x = image_cond_latent
                    x_f = x.float()
                    t = int(x.shape[2])
                    x_min = float(x_f.min().item())
                    x_max = float(x_f.max().item())
                    x_mean = float(x_f.mean().item())
                    x_std = float(x_f.std(unbiased=False).item())
                    self._log.info(
                        "[WanImageMotionPro][diag] concat_latent_image stats: T=%s min=%.4f max=%.4f mean=%.4f std=%.4f",
                        t, x_min, x_max, x_mean, x_std,
                    )

                    base = x_f[:, :, 0:1]
                    diffs = (x_f - base).abs().mean(dim=(0, 1, 3, 4))  # [T]
                    head_n = min(6, t)
                    tail_n = min(6, t)
                    head = [float(v) for v in diffs[:head_n].tolist()]
                    tail = [float(v) for v in diffs[-tail_n:].tolist()]
                    self._log.info(
                        "[WanImageMotionPro][diag] mean|Δ| vs t0: head=%s tail=%s",
                        head, tail,
                    )
                    self._log.info(
                        "[WanImageMotionPro][diag] end_t_fix_early=%s (motion cap) use_end_frame=%s end_lock_slots=%s | overshoot=%s total_latents_gen=%s",
                        end_t_fix_early, use_end_frame, end_lock_slots, _overshoot, total_latents_gen,
                    )
                except Exception as e:
                    self._log.warning("[WanImageMotionPro][diag] failed to compute diagnostics: %s", e)

            # FLF end lock: overwrite last slots with end_samples (if provided).
            # use_end_frame=False lets the user keep the wire connected without activating the lock,
            # prevents ping-pong artifacts when the end-frame encoder node is bypassed.
            end_t_fix = 0
            if end_samples is not None and use_end_frame:
                # Clone to prevent mutations from affecting the caller's tensor.
                end_latent = end_samples["samples"].clone()

                if end_latent.shape[0] == 1 and B > 1:
                    end_latent = end_latent.repeat(B, 1, 1, 1, 1)

                if (
                    end_latent.shape[1] == C
                    and end_latent.shape[3] == H
                    and end_latent.shape[4] == W
                ):
                    T_end = end_latent.shape[2]
                    # end_lock_slots caps the lock to the intended number of slots.
                    # The Wan VAE always encodes an image to T≥2 latent slots, so without
                    # this cap end_t_fix would be 2 even for a single frame, locking the
                    # last 8 video frames instead of only the last ~4 (= 1 latent slot).
                    end_t_fix = min(T_end, total_latents_gen, end_lock_slots)
                    if end_t_fix > 0:
                        self._log.info(
                            "[WanImageMotionPro] end lock: T_end=%s end_lock_slots=%s -> end_t_fix=%s (locking latent slots [%s:%s] = last ~%s video frames) | overshoot=%s trim_slots=%s",
                            T_end, end_lock_slots, end_t_fix,
                            total_latents_gen - end_t_fix, total_latents_gen,
                            end_t_fix * 4,
                            _overshoot, _overshoot,
                        )
                        image_cond_latent[:, :, -end_t_fix:] = end_latent[:, :, -end_t_fix:]

                        # end_transition_frames is DEPRECATED and intentionally disabled.
                        # Linear interpolation between padding latents and end_latent in VAE
                        # latent space does not produce meaningful intermediate frames —
                        # it generates static/corrupted conditioning that freezes the clip.
                        # The original FLF node uses a plain hard-lock (no blending).
                        if end_transition_frames > 0:
                            self._log.warning(
                                "[WanImageMotionPro] ⚠️  end_transition_frames=%s is DEPRECATED and has been DISABLED. "
                                "Linear latent interpolation before the end-lock produces static/frozen output. "
                                "Set end_transition_frames=0 to match original FLF behavior.",
                                end_transition_frames,
                            )
                else:
                    end_t_fix = 0
                    self._log.warning(
                        "[WanImageMotionPro] end_samples shape mismatch, skipping end lock. end=%s anchor=%s",
                        tuple(end_latent.shape),
                        tuple(anchor_latent.shape),
                    )
            elif end_samples is not None and not use_end_frame:
                self._log.info(
                    "[WanImageMotionPro] end_samples connected but use_end_frame=False — end lock skipped."
                )

            # Build a 4-channel temporal mask like the core WanFirstLastFrameToVideo node.
            # This matches Wan's native temporal conditioning layout better than a flat 1-channel
            # per-slot mask and avoids reducing FLF semantics to coarse whole-slot averaging.
            frame_mask = torch.ones((1, 1, total_latents_gen * 4, H, W), device=device, dtype=torch.float32)
            if lock_start_slots_i > 0:
                frame_mask[:, :, :lock_start_slots_i * 4] = 0.0
            if end_t_fix > 0:
                frame_mask[:, :, -(end_t_fix * 4):] = 0.0
            mask = frame_mask.view(1, frame_mask.shape[2] // 4, 4, H, W).transpose(1, 2).contiguous()

            # Ensure image_cond_latent is fp32 for model accuracy.
            if image_cond_latent.dtype != torch.float32:
                image_cond_latent = image_cond_latent.float()

            # Device safety: keep conditioning tensors on the same device as the sampler latent.
            cond_device = empty_latent.device
            if image_cond_latent.device != cond_device:
                if diagnostic_log:
                    self._log.info(
                        "[WanImageMotionPro][diag] moving conditioning tensors %s -> %s",
                        str(image_cond_latent.device),
                        str(cond_device),
                    )
                image_cond_latent = image_cond_latent.to(cond_device)
                mask = mask.to(cond_device)

            positive = node_helpers.conditioning_set_values(
                positive, {"concat_latent_image": image_cond_latent, "concat_mask": mask}
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"concat_latent_image": image_cond_latent, "concat_mask": mask}
            )

            if add_reference_latents:
                ref_latent = anchor_latent[:, :, 0:1]
                positive = node_helpers.conditioning_set_values(
                    positive, {"reference_latents": [ref_latent]}, append=True
                )
                negative = node_helpers.conditioning_set_values(
                    negative, {"reference_latents": [torch.zeros_like(ref_latent)]}, append=True
                )

            # Apply CLIPVision semantic conditioning if provided.
            # Merges start+end embeddings (same logic as WanFirstLastFrameToVideo).
            # Anchors cross-attention to subject/scene visual identity → reduces drift across segments.
            #
            # ⚠️  clip_vision_mode CRITICAL NOTE:
            # "auto" (concatenate) is CORRECT only for true A→B FLF (anchor=A, end=B).
            # If anchor=end=same image but clip_vision_start is a DIFFERENT image, use
            # "end_only" — otherwise the concatenated [start, end] tokens bias ALL free
            # temporal slots toward the start image via global clip_fea cross-attention.
            def _auto_prefers_end_only() -> bool:
                if not use_end_frame or end_samples is None:
                    return False
                try:
                    anchor_cmp = anchor_samples["samples"]
                    end_cmp = end_samples["samples"]
                except Exception:
                    return False
                if anchor_cmp is None or end_cmp is None or anchor_cmp.dim() != 5 or end_cmp.dim() != 5:
                    return False
                if (
                    anchor_cmp.shape[1] != end_cmp.shape[1]
                    or anchor_cmp.shape[3] != end_cmp.shape[3]
                    or anchor_cmp.shape[4] != end_cmp.shape[4]
                ):
                    return False
                compare_slots = min(anchor_cmp.shape[2], end_cmp.shape[2])
                if compare_slots <= 0:
                    return False
                return torch.allclose(
                    anchor_cmp[:, :, :compare_slots].float(),
                    end_cmp[:, :, :compare_slots].float(),
                    rtol=1e-4,
                    atol=1e-5,
                )

            _cv_out = None
            _mode = str(clip_vision_mode) if clip_vision_mode else "auto"

            if _mode != "none":
                if _mode == "end_only":
                    # Use only end image embedding. Best for SVI Pro continuation where
                    # prev_samples drives motion but anchor/end are the same scene.
                    if clip_vision_end_image is not None:
                        _cv_out = clip_vision_end_image
                    elif clip_vision_start_image is not None:
                        # Fallback: only start is connected → use it with a clear note.
                        _cv_out = clip_vision_start_image
                        self._log.info(
                            "[WanImageMotionPro] clip_vision_mode=end_only but only start image is connected → using start."
                        )

                elif _mode == "start_only":
                    # Use only start image embedding.
                    if clip_vision_start_image is not None:
                        _cv_out = clip_vision_start_image
                    elif clip_vision_end_image is not None:
                        _cv_out = clip_vision_end_image
                        self._log.info(
                            "[WanImageMotionPro] clip_vision_mode=start_only but only end image is connected → using end."
                        )

                else:
                    # "auto": preserve FLF concatenate behavior for true A→B transitions,
                    # but resolve to end_only for SVI continuation when anchor and end match.
                    if clip_vision_start_image is not None and clip_vision_end_image is not None:
                        if _auto_prefers_end_only():
                            _cv_out = clip_vision_end_image
                            self._log.info(
                                "[WanImageMotionPro] clip_vision_mode=auto resolved to end_only because anchor_samples and end_samples match. "
                                "This avoids clip_vision_start crossfade bleed during SVI continuation."
                            )
                        else:
                            # ⚠️  CROSSFADE WARNING: concatenated tokens bias ALL free slots.
                            # Only use this when anchor image == clip_vision_start and end image == clip_vision_end.
                            self._log.warning(
                                "[WanImageMotionPro] ⚠️  clip_vision_mode=auto: concatenating start+end CLIP tokens. "
                                "This creates crossfade artifacts if clip_vision_start image ≠ anchor_samples image. "
                                "If anchor and end_samples are the SAME image (SVI Pro continuation), "
                                "auto now resolves to end_only when the latents match; otherwise set clip_vision_mode=end_only manually."
                            )
                            _states = torch.cat(
                                [clip_vision_start_image.penultimate_hidden_states, clip_vision_end_image.penultimate_hidden_states],
                                dim=-2,
                            )
                            _cv_out = comfy.clip_vision.Output()
                            _cv_out.penultimate_hidden_states = _states
                    elif clip_vision_start_image is not None:
                        _cv_out = clip_vision_start_image
                    elif clip_vision_end_image is not None:
                        _cv_out = clip_vision_end_image

            if _cv_out is not None:
                positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": _cv_out})
                negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": _cv_out})

            out_latent = {"samples": empty_latent}
            return (positive, negative, out_latent, _overshoot)


class WanMotionProTrimmer:
    """WanMotionProTrimmer

    Companion node for WanImageMotionPro.
    Removes the overshoot latent slots from the sampler output when
    end_overshoot_slots > 0 was set on WanImageMotionPro.

    Wiring:
        WanImageMotionPro.latent  → KSampler → latent_in
        WanImageMotionPro.trim_slots            → trim_slots
        latent_out → downstream decode / next segment

    When trim_slots == 0 the node is a transparent pass-through (no copy).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_in": ("LATENT",),
                "trim_slots": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 64,
                        "step": 1,
                        "tooltip": (
                            "Number of latent slots to remove from the tail of latent_in. "
                            "Wire directly from the trim_slots output of WanImageMotionPro. "
                            "0 = pass-through, no modification."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latent", "video_frames")
    FUNCTION = "trim"
    CATEGORY = "IAMCCS/Wan"

    _log = logging.getLogger("IAMCCS.WanMotionProTrimmer")

    def trim(self, latent_in: dict, trim_slots: int):
        samples = latent_in["samples"]  # [B, C, T, H, W]
        T_in = samples.shape[2]
        slots = int(trim_slots)

        if slots <= 0:
            # Pass-through: do not copy the tensor.
            video_frames = (T_in - 1) * 4 + 1
            self._log.info(
                "[WanMotionProTrimmer] trim_slots=0 — pass-through. T=%s -> video_frames=%s",
                T_in, video_frames,
            )
            return (latent_in, video_frames)

        T_out = max(1, T_in - slots)
        trimmed = samples[:, :, :T_out].clone()
        video_frames = (T_out - 1) * 4 + 1
        self._log.info(
            "[WanMotionProTrimmer] T_in=%s trim_slots=%s -> T_out=%s video_frames=%s",
            T_in, slots, T_out, video_frames,
        )

        out = dict(latent_in)
        out["samples"] = trimmed
        return (out, video_frames)


WanImageMotionProAdvanced = WanImageMotionPro


class WanImageMotionPro:
    """WanImageMotionPro

    Combines IAMCCS_WanImageMotion motion amplitude control with FLF-style
    (First/Last Frame) hard lock via optional end_samples.

    Behavior:
    - Start: anchor_samples + optional motion tail from prev_samples.
    - Motion: apply motion amplitude scaling (VRAM-aware) as in IAMCCS_WanImageMotion.
    - End: overwrite last temporal latent slots with end_samples, then lock them
      via concat_mask (FLF-style control).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "length": ("INT", {"default": 81, "min": 1, "max": 16384, "step": 4}),
                "anchor_samples": ("LATENT",),
                "motion_latent_count": ("INT", {"default": 1, "min": 0, "max": 16, "step": 1}),
                "motion": ("FLOAT", {"default": 1.15, "min": 1.0, "max": 2.0, "step": 0.05}),
                "motion_mode": (
                    [
                        "motion_only (prev_samples)",
                        "all_nonfirst (anchor+motion)",
                    ],
                    {"default": "motion_only (prev_samples)"},
                ),
                "add_reference_latents": ("BOOLEAN", {"default": False}),
                "latent_precision": (
                    [
                        "auto",
                        "fp16",
                        "fp32",
                        "normal",
                    ],
                    {"default": "fp32"},
                ),
                "vram_profile": (
                    [
                        "normal",
                        "chunked_blocks_2",
                        "chunked_blocks_4",
                        "loop_per_frame (lowest_vram)",
                        "cpu_offload (slowest)",
                    ],
                    {"default": "normal"},
                ),
                "include_padding_in_motion": ("BOOLEAN", {"default": False}),
                "safety_preset": (
                    [
                        "base",
                        "safe",
                        "safer",
                        "legacy",
                    ],
                    {
                        "default": "safe",
                        "tooltip": (
                            "base   : 1:1 with WanImageToVideoSVIProFLF — no amplitude processing at all. "
                            "safe   : per-channel stabilisation + tanh limiter + 2-frame ramp (active at any motion value, recommended default). "
                            "safer  : same as safe but tighter limiter + longer ramp (best above motion=1.5). "
                            "legacy : hard clamp ±6 + frame-scalar diff centering, mirrors PainterI2V algorithm."
                        ),
                    },
                ),
                # --- End-frame controls (keep at end to preserve widgets_values ordering) ---
                "use_end_frame": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "When False, end_samples is ignored even if the wire is connected. "
                            "Use this to safely bypass the end-frame encoder without disconnecting the node, "
                            "preventing ping-pong artifacts."
                        ),
                    },
                ),
                "end_transition_frames": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 32,
                        "step": 1,
                        "tooltip": (
                            "⚠️ DEPRECATED — keep at 0. "
                            "Values > 0 blend padding latents toward the end frame in latent space, "
                            "which produces static/frozen output because VAE latent interpolation is not linear. "
                            "0 = hard cut, identical to the original FLF node behavior."
                        ),
                    },
                ),
                "end_lock_slots": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 16,
                        "step": 1,
                        "tooltip": (
                            "How many final latent slots to hard-lock to end_samples. "
                            "Default 1 = only the very last slot (~last 4 video frames) is locked, "
                            "so the end frame appears exactly at the last frame defined by 'length'. "
                            "Increase only if a single locked slot is not enough for convergence. "
                            "Set 16 for 1:1 parity with the original FLF node (locks min(T_end, total_latents))."
                        ),
                    },
                ),
                # Append-only widget for backward compatibility.
                "lock_start_slots": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 16,
                        "step": 1,
                        "tooltip": (
                            "How many initial latent slots to hard-lock to the start image via concat_mask. "
                            "Default 1 (≈ first 4 video frames) matches FLF-style start anchoring. "
                            "Set 0 to allow motion immediately from the first video frame."
                        ),
                    },
                ),
                # Append-only: diagnostic logging toggle (does not change outputs).
                "diagnostic_log": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "When enabled, prints compact diagnostics about concat_latent_image/mask and motion/end-lock ranges. "
                            "Useful to debug 'static clip' or segment discontinuities."
                        ),
                    },
                ),
                # Append-only: allow keeping prev_samples wired (autolink) but ignoring it.
                "use_prev_samples": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "When False, prev_samples is ignored even if the wire is connected. "
                            "Use this on the first segment when autolink forces prev_samples to be connected."
                        ),
                    },
                ),
                # Append-only: end-frame overshoot — extends the internal generation window so the
                # hard-locked end frame lands beyond the visible output range, letting motion converge
                # toward the end image without freezing on it. Connect trim_slots output to a downstream
                # Cut Latent Frames or Get Latent Range node to remove the extra slots after sampling.
                # 0 = disabled (original behavior). 1 recommended when end-frame freeze breaks continuity.
                "end_overshoot_slots": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 8,
                        "step": 1,
                        "tooltip": (
                            "Extends the internal generation window by this many latent slots when the end lock "
                            "is active (use_end_frame=True + end_samples connected). "
                            "The end image is locked at the tail of the EXTENDED range, so the visible clip "
                            "converges toward the end frame without hard-freezing on it. "
                            "Use the trim_slots output to cut the overshoot frames after sampling. "
                            "0 = disabled, original behavior. 1 = +4 video frames (recommended). "
                            "2 = +8 video frames (looser convergence). "
                            "NOTE: If you have KJNodes WAN attention optimizations (FETA / wrapped_attention) enabled, "
                            "some versions may crash with an einops shape mismatch when overshoot > 0. "
                            "Workaround: set end_overshoot_slots=0 or disable/update that KJNodes optimization."
                        ),
                    },
                ),
                # DC Drift Correction: corrects per-channel mean shift accumulated
                # across chained segments. Does NOT touch spatial structure or motion.
                # Algorithm:  drift = mean(motion_tail, dims=T,H,W) − mean(anchor, dims=T,H,W)
                #             motion_tail -= drift * latent_refresh
                "latent_refresh": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "DC drift correction strength (0=disabled, 1=full correction). "
                            "Corrects the per-channel mean shift between motion tail and anchor latent. "
                            "Does not touch spatial structure or motion. "
                            "Start with 0.5 for chains of 3+ segments. Enable diagnostic_log to verify."
                        ),
                    },
                ),
                # Soft-clamp on the DC drift itself.
                # 0 = disabled. Use 0.5–1.0 to prevent over-correction on large motion segments.
                "delta_max": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.1,
                        "tooltip": (
                            "Soft-clamp on DC drift magnitude (0=disabled). "
                            "Caps how much drift correction can be applied per channel. "
                            "Use 0.5–1.0 to prevent over-correction on large motion segments."
                        ),
                    },
                ),
                "clip_vision_mode": (
                    [
                        "auto",
                        "end_only",
                        "start_only",
                        "none",
                    ],
                    {
                        "default": "auto",
                        "tooltip": (
                            "Controls how clip_vision_start_image / clip_vision_end_image are merged.\n"
                            "auto       : concatenate start+end tokens (original FLF behavior). "
                            "Correct ONLY when anchor=A and end=B (true A→B transition). "
                            "end_only   : use only clip_vision_end embedding. "
                            "RECOMMENDED for SVI Pro continuation where anchor=end=same image "
                            "but prev_samples comes from a different scene — prevents crossfade bleed. "
                            "start_only : use only clip_vision_start embedding. "
                            "none       : skip clip_vision injection entirely (bypass)."
                        ),
                    },
                ),
                # Append-only strict compatibility path for the original FLF+SVI behavior.
                # This bypasses MotionPro enhancements and builds conditioning like
                # WanImageToVideoSVIProFLF: full anchor latent + optional motion tail,
                # plain 1-channel slot mask, no clip vision, no overshoot, no motion scaling.
                "raw_mode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Strict compatibility mode for the original WanImageToVideoSVIProFLF behavior. "
                            "When enabled, WanImageMotionPro bypasses advanced features (motion amplitude, "
                            "DC drift correction, overshoot, clip vision routing, reference latents, slot caps) "
                            "and builds conditioning 1:1 like the original FLF+SVI node. "
                            "Use this to benchmark or to get the baseline first/last frame behavior back."
                        ),
                    },
                ),
            },
            "optional": {
                "prev_samples": ("LATENT",),
                "end_samples": ("LATENT",),
                "clip_vision_start_image": ("CLIP_VISION_OUTPUT",),
                "clip_vision_end_image": ("CLIP_VISION_OUTPUT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("positive", "negative", "latent", "trim_slots")
    FUNCTION = "apply"
    CATEGORY = "IAMCCS/Wan"

    _log = logging.getLogger("IAMCCS.WanImageMotionPro")

    def _build_raw_mode_conditioning(
        self,
        *,
        positive,
        negative,
        length,
        anchor_samples,
        motion_latent_count,
        prev_samples,
        end_samples,
    ):
        # Intentional copy of the original WanImageToVideoSVIProFLF semantics.
        anchor_latent = anchor_samples["samples"].clone()
        B, C, T_anchor, H, W = anchor_latent.shape

        total_latents = (length - 1) // 4 + 1
        device = anchor_latent.device
        dtype = anchor_latent.dtype

        if prev_samples is None or motion_latent_count == 0:
            base_latent = anchor_latent
        else:
            prev_latent = prev_samples["samples"].clone()
            T_prev = prev_latent.shape[2]
            motion_count = min(motion_latent_count, T_prev)

            if motion_count > 0:
                motion_latent = prev_latent[:, :, -motion_count:]
                base_latent = torch.cat([anchor_latent, motion_latent], dim=2)
            else:
                base_latent = anchor_latent

        T_base = base_latent.shape[2]
        padding_size = max(total_latents - T_base, 0)

        if padding_size > 0:
            padding = torch.zeros(B, C, padding_size, H, W, dtype=dtype, device=device)
            padding = comfy.latent_formats.Wan21().process_out(padding)
            image_cond_latent = torch.cat([base_latent, padding], dim=2)
        else:
            image_cond_latent = base_latent[:, :, :total_latents]

        if image_cond_latent.shape[2] != total_latents:
            image_cond_latent = image_cond_latent[:, :, :total_latents]

        end_t_fix = 0
        if end_samples is not None:
            end_latent = end_samples["samples"].clone()

            if end_latent.shape[0] == 1 and B > 1:
                end_latent = end_latent.repeat(B, 1, 1, 1, 1)

            if (
                end_latent.shape[1] == C
                and end_latent.shape[3] == H
                and end_latent.shape[4] == W
            ):
                T_end = end_latent.shape[2]
                end_t_fix = min(T_end, total_latents)

                if end_t_fix > 0:
                    image_cond_latent[:, :, -end_t_fix:] = end_latent[:, :, -end_t_fix:]
            else:
                end_t_fix = 0

        empty_latent = torch.zeros(
            [B, 16, total_latents, H, W],
            device=comfy.model_management.intermediate_device(),
        )

        mask = torch.ones((1, 1, total_latents, H, W), device=device, dtype=dtype)
        mask[:, :, :1] = 0.0

        if end_t_fix > 0:
            mask[:, :, -end_t_fix:] = 0.0

        positive = node_helpers.conditioning_set_values(
            positive, {"concat_latent_image": image_cond_latent, "concat_mask": mask}
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"concat_latent_image": image_cond_latent, "concat_mask": mask}
        )

        self._log.info(
            "[WanImageMotionPro][raw_mode] active: original FLF+SVI semantics | total_latents=%s end_t_fix=%s motion_latent_count=%s prev=%s",
            total_latents,
            end_t_fix,
            motion_latent_count,
            prev_samples is not None,
        )
        out_latent = {"samples": empty_latent}
        return (positive, negative, out_latent, 0)

    def _pick_empty_latent_dtype(self, anchor_dtype: torch.dtype, latent_precision: str) -> torch.dtype:
        # Keep 1:1 behavior with IAMCCS_WanImageMotion.
        if latent_precision == "normal":
            return anchor_dtype
        if latent_precision == "auto":
            return torch.float32
        if latent_precision == "fp32":
            return torch.float32
        if latent_precision == "fp16":
            return torch.float16
        return anchor_dtype

    def _apply_motion_amplitude(
        self,
        image_cond_latent: torch.Tensor,
        *,
        real_latents: int,
        anchor_latents: int,
        motion_latents: int,
        motion_amplitude: float,
        motion_mode: str,
        vram_profile: str,
        safety_preset: str = "safe",
    ) -> torch.Tensor:
        # Reuse the exact logic from IAMCCS_WanImageMotion (copy to keep node self-contained).
        preset = _preset_params(safety_preset, motion_amplitude)

        # "base" preset: 1:1 with reference nodes — no processing at all.
        if not preset.get("enabled", True):
            return image_cond_latent

        if motion_amplitude is None or motion_amplitude <= 1.0:
            return image_cond_latent

        if image_cond_latent.shape[2] <= 1:
            return image_cond_latent

        base_latent = image_cond_latent[:, :, 0:1]

        def _scale_slice_gpu(view_slice: torch.Tensor, *, gain: torch.Tensor | float) -> torch.Tensor:
            with torch.no_grad():
                diff = view_slice - base_latent
                diff_centered, diff_mean = _center_diff(diff, mean_mode=preset["mean_mode"])
                diff_centered.mul_(gain)
                out_local = diff_centered.add_(diff_mean).add_(base_latent)
                _apply_soft_limiter(out_local, mode=preset["limiter_mode"], limit=float(preset["limiter_limit"]))
                return out_local

        def _gain_weights(start: int, end: int) -> torch.Tensor | float:
            ramp_frames = int(preset["ramp_frames"])
            if ramp_frames <= 0:
                return float(motion_amplitude)
            tcount = max(0, end - start)
            if tcount <= 0:
                return float(motion_amplitude)
            ramp = min(ramp_frames, tcount)
            w = torch.ones((tcount,), device=image_cond_latent.device, dtype=image_cond_latent.dtype)
            if ramp > 0:
                x = torch.linspace(0.0, 1.0, steps=ramp, device=w.device, dtype=w.dtype)
                w[:ramp] = _smoothstep(x)
            gain = 1.0 + (motion_amplitude - 1.0) * w
            return gain.view(1, 1, tcount, 1, 1)

        def _apply_to_range(start: int, end: int) -> torch.Tensor:
            if end <= start:
                return out

            if vram_profile == "normal":
                gain = _gain_weights(start, end)
                out[:, :, start:end] = _scale_slice_gpu(out[:, :, start:end], gain=gain)
                return out

            if vram_profile in ("chunked_blocks_2", "chunked_blocks_4"):
                block = 2 if vram_profile == "chunked_blocks_2" else 4
                t = start
                while t < end:
                    t2 = min(end, t + block)
                    gain = _gain_weights(t, t2)
                    out[:, :, t:t2] = _scale_slice_gpu(out[:, :, t:t2], gain=gain)
                    t = t2
                return out

            if vram_profile == "loop_per_frame (lowest_vram)":
                for t in range(start, end):
                    gain = _gain_weights(t, t + 1)
                    out[:, :, t:t + 1] = _scale_slice_gpu(out[:, :, t:t + 1], gain=gain)
                return out

            if vram_profile == "cpu_offload (slowest)":
                with torch.no_grad():
                    device = out.device
                    base_cpu = base_latent.detach().to("cpu")
                    slice_cpu = out[:, :, start:end].detach().to("cpu")
                    diff = slice_cpu - base_cpu
                    diff_centered, diff_mean = _center_diff(diff, mean_mode=preset["mean_mode"])
                    ramp_frames = int(preset["ramp_frames"])
                    tcount = max(0, end - start)
                    if ramp_frames > 0 and tcount > 0:
                        ramp = min(ramp_frames, tcount)
                        w = torch.ones((tcount,), device=diff_centered.device, dtype=diff_centered.dtype)
                        x = torch.linspace(0.0, 1.0, steps=ramp, device=w.device, dtype=w.dtype)
                        w[:ramp] = _smoothstep(x)
                        gain = (1.0 + (motion_amplitude - 1.0) * w).view(1, 1, tcount, 1, 1)
                    else:
                        gain = float(motion_amplitude)
                    diff_centered.mul_(gain)
                    out_cpu = diff_centered.add_(diff_mean).add_(base_cpu)
                    _apply_soft_limiter(out_cpu, mode=preset["limiter_mode"], limit=float(preset["limiter_limit"]))
                    out[:, :, start:end] = out_cpu.to(device)
                return out

            gain = _gain_weights(start, end)
            out[:, :, start:end] = _scale_slice_gpu(out[:, :, start:end], gain=gain)
            return out

        real_latents = max(0, min(real_latents, image_cond_latent.shape[2]))
        if real_latents <= 1:
            return image_cond_latent

        out = image_cond_latent

        if motion_mode == "motion_only (prev_samples)":
            if motion_latents <= 0:
                return out

            start = anchor_latents
            end = min(anchor_latents + motion_latents, real_latents)
            if end <= start:
                return out

            return _apply_to_range(start, end)

        start = 1
        end = real_latents
        return _apply_to_range(start, end)

    def apply(
        self,
        positive,
        negative,
        length,
        anchor_samples,
        motion_latent_count,
        motion,
        motion_mode,
        add_reference_latents,
        latent_precision,
        vram_profile,
        include_padding_in_motion,
        safety_preset="safe",
        use_end_frame=True,
        end_transition_frames=4,
        end_lock_slots=1,
        lock_start_slots=1,
        diagnostic_log=False,
        use_prev_samples=True,
        end_overshoot_slots=0,
        latent_refresh=0.0,
        delta_max=0.0,
        clip_vision_mode="auto",
        raw_mode=False,
        prev_samples=None,
        end_samples=None,
        clip_vision_start_image=None,
        clip_vision_end_image=None,
        **_kwargs,
    ):
        with torch.no_grad():
            if raw_mode:
                return self._build_raw_mode_conditioning(
                    positive=positive,
                    negative=negative,
                    length=length,
                    anchor_samples=anchor_samples,
                    motion_latent_count=motion_latent_count,
                    prev_samples=prev_samples if use_prev_samples else None,
                    end_samples=end_samples if use_end_frame else None,
                )

            # Clone to prevent in-place motion amplitude writes from corrupting the caller's tensor.
            anchor_latent = anchor_samples["samples"].clone()

            B, C, T_anchor, H, W = anchor_latent.shape

            lock_start_slots_i = int(max(0, min(16, lock_start_slots)))

            # FLF core parity: when using a single-image start anchor, the core node constrains
            # only the actual start frame. A Wan VAEEncode of a still image often expands to more
            # than one temporal latent slot; keeping the full block here makes the bridge hold on
            # to A for too long and often degenerates into a dissolve. When FLF end control is
            # active, cap the start conditioning block to the number of locked start slots.
            anchor_slots_for_cond = T_anchor
            if use_end_frame and end_samples is not None and T_anchor > 1 and lock_start_slots_i <= 1:
                anchor_slots_for_cond = 1
                self._log.info(
                    "[WanImageMotionPro] FLF start normalization: T_anchor=%s -> using first %s slot for conditioning.",
                    T_anchor,
                    anchor_slots_for_cond,
                )

            anchor_cond_latent = anchor_latent[:, :, :anchor_slots_for_cond]

            total_latents = (length - 1) // 4 + 1

            # Overshoot: when end lock is active and end_overshoot_slots > 0, extend the internal
            # generation window so the end-locked zone lands beyond what the sampler actually produces
            # as "visible" output. trim_slots (4th output) tells the downstream crop node how many
            # latent slots to remove from the tail after sampling.
            _overshoot = (
                int(end_overshoot_slots)
                if (use_end_frame and end_samples is not None and int(end_overshoot_slots) > 0)
                else 0
            )
            total_latents_gen = total_latents + _overshoot

            if _overshoot > 0:
                self._log.warning(
                    "[WanImageMotionPro] end_overshoot_slots=%s enabled (total_latents=%s -> gen=%s). "
                    "If you use KJNodes WAN attention optimizations (FETA / wrapped_attention) and hit an einops shape mismatch, "
                    "set end_overshoot_slots=0 or disable/update that optimization.",
                    int(end_overshoot_slots),
                    total_latents,
                    total_latents_gen,
                )

            device = anchor_latent.device
            dtype = anchor_latent.dtype

            empty_latent_dtype = self._pick_empty_latent_dtype(dtype, latent_precision)
            empty_latent = torch.zeros(
                [B, 16, total_latents_gen, H, W],
                device=comfy.model_management.intermediate_device(),
                dtype=empty_latent_dtype,
            )

            # --- 1. Estrazione motion tail da prev_samples ---
            motion_latent = None
            T_motion = 0
            has_prev = bool(use_prev_samples) and prev_samples is not None and motion_latent_count != 0

            if has_prev:
                motion_latent = prev_samples["samples"][:, :, -motion_latent_count:]
                T_motion = motion_latent.shape[2]

            # --- 2. DC Drift Correction ---
            # Same algorithm as IAMCCS_WanImageMotion: corrects per-channel mean shift.
            # drift = mean(motion_tail, T,H,W) − mean(anchor, T,H,W)  → [B,C,1,1,1]
            # motion_tail -= drift * latent_refresh  (spatial structure untouched)
            _lr = float(latent_refresh)
            _dm = float(delta_max)
            if motion_latent is not None and _lr > 0:
                with torch.no_grad():
                    motion_mean = motion_latent.mean(dim=(2, 3, 4), keepdim=True)
                    anchor_mean = anchor_latent.mean(dim=(2, 3, 4), keepdim=True)
                    dc_drift = motion_mean - anchor_mean

                    if diagnostic_log:
                        drift_vals = dc_drift[0, :, 0, 0, 0]
                        drift_abs = drift_vals.abs()
                        self._log.info(
                            "[WanImageMotionPro][dc_drift] BEFORE correction — "
                            "per-channel drift: max=%.4f mean=%.4f rms=%.4f "
                            "| motion_mean=[%.3f..%.3f] anchor_mean=[%.3f..%.3f]",
                            float(drift_abs.max()),
                            float(drift_vals.mean()),
                            float((drift_vals ** 2).mean().sqrt()),
                            float(motion_mean.min()),
                            float(motion_mean.max()),
                            float(anchor_mean.min()),
                            float(anchor_mean.max()),
                        )
                        for ci in range(min(16, int(drift_vals.shape[0]))):
                            self._log.info(
                                "[WanImageMotionPro][dc_drift]   ch%02d: drift=% .4f  motion_mean=% .4f  anchor_mean=% .4f",
                                ci,
                                float(drift_vals[ci]),
                                float(motion_mean[0, ci, 0, 0, 0]),
                                float(anchor_mean[0, ci, 0, 0, 0]),
                            )

                    if _dm > 0:
                        dc_drift = _dm * torch.tanh(dc_drift / _dm)

                    correction = dc_drift * _lr
                    motion_latent = motion_latent - correction

                    if diagnostic_log:
                        new_mean = motion_latent.mean(dim=(2, 3, 4), keepdim=True)
                        residual = (new_mean - anchor_mean)[0, :, 0, 0, 0]
                        self._log.info(
                            "[WanImageMotionPro][dc_drift] AFTER  correction — residual drift: max=%.4f mean=%.4f | correction_strength=%.2f delta_max=%.2f",
                            float(residual.abs().max()),
                            float(residual.mean()),
                            _lr,
                            _dm,
                        )
                    else:
                        self._log.info(
                            "[WanImageMotionPro] DC drift correction active: strength=%.2f delta_max=%.2f | drift_max=%.4f drift_mean=%.4f",
                            _lr,
                            _dm,
                            float(dc_drift.abs().max()),
                            float(dc_drift.mean()),
                        )

            # --- 3. Costruzione del condizionamento ---
            if motion_latent is None:
                padding_size = total_latents_gen - anchor_slots_for_cond
                image_cond_latent = anchor_cond_latent
            else:
                padding_size = total_latents_gen - anchor_slots_for_cond - T_motion
                image_cond_latent = torch.cat([anchor_cond_latent, motion_latent], dim=2)

            padding_size = max(0, padding_size)
            if padding_size > 0:
                padding = torch.zeros(B, C, padding_size, H, W, dtype=dtype, device=device)
                padding = comfy.latent_formats.Wan21().process_out(padding)
                image_cond_latent = torch.cat([image_cond_latent, padding], dim=2)

            # FLF/SVI reference behavior: enforce exact temporal length (extended by overshoot).
            if image_cond_latent.shape[2] > total_latents_gen:
                image_cond_latent = image_cond_latent[:, :, :total_latents_gen]
            elif image_cond_latent.shape[2] < total_latents_gen:
                image_cond_latent = image_cond_latent[:, :, :total_latents_gen]

            # Pre-compute end_t_fix so we can exclude the end-locked zone from motion amplitude.
            # Motion should NOT touch slots that will be hard-locked to end_samples: scaling those
            # intermediate latents would generate noise that hurts the model's first→last interpolation.
            # Respect use_end_frame: if disabled, never cap the motion range.
            end_t_fix_early = 0
            if end_samples is not None and use_end_frame:
                _e = end_samples["samples"]
                if (
                    _e.shape[1] == C
                    and _e.shape[3] == H
                    and _e.shape[4] == W
                ):
                    # Cap to end_lock_slots so the motion range is not incorrectly
                    # shortened by the full VAE temporal size (which can be T>1 for a single image).
                    end_t_fix_early = min(_e.shape[2], total_latents_gen, end_lock_slots)

            # Motion boost applied before FLF overwrite.
            # Cap effective_latents so motion never reaches into the end-locked zone.
            # IMPORTANT: do not apply motion scaling to *pure padding*.
            # Padding slots are zeros and should remain sampler-driven.
            if include_padding_in_motion and not has_prev:
                self._log.info(
                    "[WanImageMotionPro] include_padding_in_motion=True but prev_samples is not connected; "
                    "skipping padding boost to avoid static/locked-looking clips."
                )

            if include_padding_in_motion and has_prev and (T_anchor + T_motion) > 1:
                effective_latents_base = total_latents_gen
            else:
                effective_latents_base = min(total_latents_gen, anchor_slots_for_cond + T_motion)
            effective_latents = max(1, min(effective_latents_base, total_latents_gen - end_t_fix_early))

            motion_mode_effective = motion_mode
            # NOTE: we intentionally do NOT auto-switch to all_nonfirst here.
            # Auto-switching would end up modifying padding slots on the first segment.

            try:
                free_vram, total_vram = comfy.model_management.get_free_memory(device)
            except Exception:
                free_vram, total_vram = None, None

            self._log.info(
                "[WanImageMotionPro] length=%s -> total_latents=%s (+overshoot=%s -> gen=%s) | motion=%s | mode=%s | vram_profile=%s | latent_precision=%s | add_reference_latents=%s | include_padding_in_motion=%s | use_end_frame=%s | end_transition_frames=%s | end_lock_slots=%s | lock_start_slots=%s",
                length,
                total_latents,
                _overshoot,
                total_latents_gen,
                motion,
                motion_mode,
                vram_profile,
                latent_precision,
                add_reference_latents,
                include_padding_in_motion,
                use_end_frame,
                end_transition_frames,
                end_lock_slots,
                lock_start_slots,
            )
            self._log.info(
                "[WanImageMotionPro] anchor: B=%s C=%s T=%s H=%s W=%s dtype=%s device=%s | prev=%s motion_latent_count=%s T_motion=%s | padding_size=%s | end_samples=%s",
                B,
                C,
                T_anchor,
                H,
                W,
                str(dtype).replace("torch.", ""),
                str(device),
                has_prev,
                motion_latent_count,
                T_motion,
                padding_size,
                end_samples is not None,
            )
            if anchor_slots_for_cond != T_anchor:
                self._log.info(
                    "[WanImageMotionPro] start conditioning uses %s/%s anchor slots to match FLF first-frame behavior.",
                    anchor_slots_for_cond,
                    T_anchor,
                )
            if prev_samples is not None and not use_prev_samples:
                self._log.info("[WanImageMotionPro] prev_samples connected but use_prev_samples=False — prev ignored.")
            if free_vram is not None:
                self._log.info("[WanImageMotionPro] free_vram=%s total_vram=%s", free_vram, total_vram)

            if motion_mode_effective == "motion_only (prev_samples)":
                motion_start = anchor_slots_for_cond
                motion_end = min(anchor_slots_for_cond + T_motion, effective_latents)
            else:
                motion_start = 1
                motion_end = effective_latents

            motion_frames_count = max(0, motion_end - motion_start)
            self._log.info(
                "[WanImageMotionPro] motion_range=[%s:%s] (effective_latents=%s) padding_included=%s",
                motion_start,
                motion_end,
                effective_latents,
                include_padding_in_motion,
            )
            if motion_frames_count == 0:
                self._log.warning(
                    "[WanImageMotionPro] WARNING: motion_range is EMPTY (no frames will be modified). "
                    "Enable include_padding_in_motion or provide prev_samples with motion_latent_count > 0."
                )
            else:
                self._log.info(
                    "[WanImageMotionPro] Motion boost applies to %s frame(s) amplitude=%.2f",
                    motion_frames_count,
                    motion,
                )

            image_cond_latent = self._apply_motion_amplitude(
                image_cond_latent,
                real_latents=effective_latents,
                anchor_latents=anchor_slots_for_cond,
                motion_latents=T_motion,
                motion_amplitude=motion,
                motion_mode=motion_mode_effective,
                vram_profile=vram_profile,
                safety_preset=safety_preset,
            )

            if diagnostic_log:
                try:
                    x = image_cond_latent
                    x_f = x.float()
                    t = int(x.shape[2])
                    x_min = float(x_f.min().item())
                    x_max = float(x_f.max().item())
                    x_mean = float(x_f.mean().item())
                    x_std = float(x_f.std(unbiased=False).item())
                    self._log.info(
                        "[WanImageMotionPro][diag] concat_latent_image stats: T=%s min=%.4f max=%.4f mean=%.4f std=%.4f",
                        t,
                        x_min,
                        x_max,
                        x_mean,
                        x_std,
                    )

                    base = x_f[:, :, 0:1]
                    diffs = (x_f - base).abs().mean(dim=(0, 1, 3, 4))  # [T]
                    head_n = min(6, t)
                    tail_n = min(6, t)
                    head = [float(v) for v in diffs[:head_n].tolist()]
                    tail = [float(v) for v in diffs[-tail_n:].tolist()]
                    self._log.info(
                        "[WanImageMotionPro][diag] mean|Δ| vs t0: head=%s tail=%s",
                        head,
                        tail,
                    )
                    self._log.info(
                        "[WanImageMotionPro][diag] end_t_fix_early=%s (motion cap) use_end_frame=%s end_lock_slots=%s | overshoot=%s total_latents_gen=%s",
                        end_t_fix_early,
                        use_end_frame,
                        end_lock_slots,
                        _overshoot,
                        total_latents_gen,
                    )
                except Exception as e:
                    self._log.warning("[WanImageMotionPro][diag] failed to compute diagnostics: %s", e)

            # FLF end lock: overwrite last slots with end_samples (if provided).
            # use_end_frame=False lets the user keep the wire connected without activating the lock,
            # prevents ping-pong artifacts when the end-frame encoder node is bypassed.
            end_t_fix = 0
            if end_samples is not None and use_end_frame:
                # Clone to prevent mutations from affecting the caller's tensor.
                end_latent = end_samples["samples"].clone()

                if end_latent.shape[0] == 1 and B > 1:
                    end_latent = end_latent.repeat(B, 1, 1, 1, 1)

                if (
                    end_latent.shape[1] == C
                    and end_latent.shape[3] == H
                    and end_latent.shape[4] == W
                ):
                    T_end = end_latent.shape[2]
                    # end_lock_slots caps the lock to the intended number of slots.
                    # The Wan VAE always encodes an image to T≥2 latent slots, so without
                    # this cap end_t_fix would be 2 even for a single frame, locking the
                    # last 8 video frames instead of only the last ~4 (= 1 latent slot).
                    end_t_fix = min(T_end, total_latents_gen, end_lock_slots)
                    if end_t_fix > 0:
                        self._log.info(
                            "[WanImageMotionPro] end lock: T_end=%s end_lock_slots=%s -> end_t_fix=%s (locking latent slots [%s:%s] = last ~%s video frames) | overshoot=%s trim_slots=%s",
                            T_end,
                            end_lock_slots,
                            end_t_fix,
                            total_latents_gen - end_t_fix,
                            total_latents_gen,
                            end_t_fix * 4,
                            _overshoot,
                            _overshoot,
                        )
                        image_cond_latent[:, :, -end_t_fix:] = end_latent[:, :, -end_t_fix:]

                        # end_transition_frames is DEPRECATED and intentionally disabled.
                        # Linear interpolation between padding latents and end_latent in VAE
                        # latent space does not produce meaningful intermediate frames —
                        # it generates static/corrupted conditioning that freezes the clip.
                        # The original FLF node uses a plain hard-lock (no blending).
                        if end_transition_frames > 0:
                            self._log.warning(
                                "[WanImageMotionPro] ⚠️  end_transition_frames=%s is DEPRECATED and has been DISABLED. "
                                "Linear latent interpolation before the end-lock produces static/frozen output. "
                                "Set end_transition_frames=0 to match original FLF behavior.",
                                end_transition_frames,
                            )
                else:
                    end_t_fix = 0
                    self._log.warning(
                        "[WanImageMotionPro] end_samples shape mismatch, skipping end lock. end=%s anchor=%s",
                        tuple(end_latent.shape),
                        tuple(anchor_latent.shape),
                    )
            elif end_samples is not None and not use_end_frame:
                self._log.info(
                    "[WanImageMotionPro] end_samples connected but use_end_frame=False — end lock skipped."
                )

            # Build a 4-channel temporal mask like the core WanFirstLastFrameToVideo node.
            # This matches Wan's native temporal conditioning layout better than a flat 1-channel
            # per-slot mask and avoids reducing FLF semantics to coarse whole-slot averaging.
            frame_mask = torch.ones((1, 1, total_latents_gen * 4, H, W), device=device, dtype=torch.float32)
            if lock_start_slots_i > 0:
                frame_mask[:, :, :lock_start_slots_i * 4] = 0.0
            if end_t_fix > 0:
                frame_mask[:, :, -(end_t_fix * 4):] = 0.0
            mask = frame_mask.view(1, frame_mask.shape[2] // 4, 4, H, W).transpose(1, 2).contiguous()

            # Ensure image_cond_latent is fp32 for model accuracy.
            if image_cond_latent.dtype != torch.float32:
                image_cond_latent = image_cond_latent.float()

            # Device safety: keep conditioning tensors on the same device as the sampler latent.
            cond_device = empty_latent.device
            if image_cond_latent.device != cond_device:
                if diagnostic_log:
                    self._log.info(
                        "[WanImageMotionPro][diag] moving conditioning tensors %s -> %s",
                        str(image_cond_latent.device),
                        str(cond_device),
                    )
                image_cond_latent = image_cond_latent.to(cond_device)
                mask = mask.to(cond_device)

            positive = node_helpers.conditioning_set_values(
                positive, {"concat_latent_image": image_cond_latent, "concat_mask": mask}
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"concat_latent_image": image_cond_latent, "concat_mask": mask}
            )

            if add_reference_latents:
                ref_latent = anchor_latent[:, :, 0:1]
                positive = node_helpers.conditioning_set_values(
                    positive, {"reference_latents": [ref_latent]}, append=True
                )
                negative = node_helpers.conditioning_set_values(
                    negative, {"reference_latents": [torch.zeros_like(ref_latent)]}, append=True
                )

            # Apply CLIPVision semantic conditioning if provided.
            # Merges start+end embeddings (same logic as WanFirstLastFrameToVideo).
            # Anchors cross-attention to subject/scene visual identity → reduces drift across segments.
            #
            # ⚠️  clip_vision_mode CRITICAL NOTE:
            # "auto" (concatenate) is CORRECT only for true A→B FLF (anchor=A, end=B).
            # If anchor=end=same image but clip_vision_start is a DIFFERENT image, use
            # "end_only" — otherwise the concatenated [start, end] tokens bias ALL free
            # temporal slots toward the start image via global clip_fea cross-attention.
            _cv_out = None
            _mode = str(clip_vision_mode) if clip_vision_mode else "auto"

            if _mode != "none":
                if _mode == "end_only":
                    # Use only end image embedding. Best for SVI Pro continuation where
                    # prev_samples drives motion but anchor/end are the same scene.
                    if clip_vision_end_image is not None:
                        _cv_out = clip_vision_end_image
                    elif clip_vision_start_image is not None:
                        # Fallback: only start is connected → use it with a clear note.
                        _cv_out = clip_vision_start_image
                        self._log.info(
                            "[WanImageMotionPro] clip_vision_mode=end_only but only start image is connected → using start."
                        )

                elif _mode == "start_only":
                    # Use only start image embedding.
                    if clip_vision_start_image is not None:
                        _cv_out = clip_vision_start_image
                    elif clip_vision_end_image is not None:
                        _cv_out = clip_vision_end_image
                        self._log.info(
                            "[WanImageMotionPro] clip_vision_mode=start_only but only end image is connected → using end."
                        )

                else:
                    if clip_vision_start_image is not None and clip_vision_end_image is not None:
                        # ⚠️  CROSSFADE WARNING: concatenated tokens bias ALL free slots.
                        # Only use this when anchor image == clip_vision_start and end image == clip_vision_end.
                        self._log.warning(
                            "[WanImageMotionPro] ⚠️  clip_vision_mode=auto: concatenating start+end CLIP tokens. "
                            "This creates crossfade artifacts if clip_vision_start image ≠ anchor_samples image. "
                            "If anchor and end_samples are the SAME image (SVI Pro continuation), "
                            "set clip_vision_mode=end_only to prevent crossfade bleed from clip_vision_start."
                        )
                        _states = torch.cat(
                            [clip_vision_start_image.penultimate_hidden_states, clip_vision_end_image.penultimate_hidden_states],
                            dim=-2,
                        )
                        _cv_out = comfy.clip_vision.Output()
                        _cv_out.penultimate_hidden_states = _states
                    elif clip_vision_start_image is not None:
                        _cv_out = clip_vision_start_image
                    elif clip_vision_end_image is not None:
                        _cv_out = clip_vision_end_image

            if _cv_out is not None:
                positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": _cv_out})
                negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": _cv_out})

            out_latent = {"samples": empty_latent}
            return (positive, negative, out_latent, _overshoot)


class IAMCCS_WanImageMotionPro_Simple(WanImageMotionProAdvanced):
    """Simplified WanImageMotionPro UI.

    Keeps the useful motion/FLF controls while hiding legacy/debug and
    redundant expert toggles that make the node harder to use.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "length": ("INT", {"default": 81, "min": 1, "max": 16384, "step": 4}),
                "anchor_samples": ("LATENT",),
                "motion_latent_count": ("INT", {"default": 1, "min": 0, "max": 16, "step": 1}),
                "motion": ("FLOAT", {"default": 1.15, "min": 1.0, "max": 2.0, "step": 0.05}),
                "motion_mode": (
                    [
                        "motion_only (prev_samples)",
                        "all_nonfirst (anchor+motion)",
                    ],
                    {"default": "motion_only (prev_samples)"},
                ),
                "add_reference_latents": ("BOOLEAN", {"default": False}),
                "vram_profile": (
                    [
                        "normal",
                        "chunked_blocks_2",
                        "chunked_blocks_4",
                        "loop_per_frame (lowest_vram)",
                        "cpu_offload (slowest)",
                    ],
                    {"default": "normal"},
                ),
                "include_padding_in_motion": ("BOOLEAN", {"default": False}),
                "safety_preset": (
                    [
                        "base",
                        "safe",
                        "safer",
                        "legacy",
                    ],
                    {"default": "safe"},
                ),
                "use_end_frame": ("BOOLEAN", {"default": True}),
                "end_lock_slots": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
                "lock_start_slots": ("INT", {"default": 1, "min": 0, "max": 16, "step": 1}),
                "end_overshoot_slots": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1}),
                "latent_refresh": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "delta_max": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "clip_vision_mode": (
                    [
                        "auto",
                        "end_only",
                        "start_only",
                        "none",
                    ],
                    {"default": "auto"},
                ),
                "prev_skip_last_slots": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 16,
                        "step": 1,
                        "tooltip": (
                            "Ignore the last N temporal latent slots from prev_samples before extracting the motion tail. "
                            "Useful when the last part of a carried SVI tail is too rigid, frozen, or over-constrained."
                        ),
                    },
                ),
            },
            "optional": {
                "prev_samples": ("LATENT",),
                "end_samples": ("LATENT",),
                "clip_vision_start_image": ("CLIP_VISION_OUTPUT",),
                "clip_vision_end_image": ("CLIP_VISION_OUTPUT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("positive", "negative", "latent", "trim_slots")
    FUNCTION = "apply"
    CATEGORY = "IAMCCS/Wan"

    _log = logging.getLogger("IAMCCS.WanImageMotionPro")

    def apply(
        self,
        positive,
        negative,
        length,
        anchor_samples,
        motion_latent_count,
        motion,
        motion_mode,
        add_reference_latents,
        vram_profile,
        include_padding_in_motion,
        safety_preset="safe",
        use_end_frame=True,
        end_lock_slots=1,
        lock_start_slots=1,
        end_overshoot_slots=0,
        latent_refresh=0.0,
        delta_max=0.0,
        clip_vision_mode="auto",
        prev_skip_last_slots=0,
        prev_samples=None,
        end_samples=None,
        clip_vision_start_image=None,
        clip_vision_end_image=None,
        **_kwargs,
    ):
        return super().apply(
            positive=positive,
            negative=negative,
            length=length,
            anchor_samples=anchor_samples,
            motion_latent_count=motion_latent_count,
            motion=motion,
            motion_mode=motion_mode,
            add_reference_latents=add_reference_latents,
            latent_precision="fp32",
            vram_profile=vram_profile,
            include_padding_in_motion=include_padding_in_motion,
            safety_preset=safety_preset,
            use_end_frame=use_end_frame,
            end_transition_frames=0,
            end_lock_slots=end_lock_slots,
            lock_start_slots=lock_start_slots,
            diagnostic_log=False,
            use_prev_samples=True,
            end_overshoot_slots=end_overshoot_slots,
            latent_refresh=latent_refresh,
            delta_max=delta_max,
            clip_vision_mode=clip_vision_mode,
            raw_mode=False,
            prev_samples_profile="direct",
            prev_tail_pick_mode="latest",
            prev_skip_last_slots=prev_skip_last_slots,
            prev_anchor_blend=0.0,
            prev_samples=prev_samples,
            end_samples=end_samples,
            clip_vision_start_image=clip_vision_start_image,
            clip_vision_end_image=clip_vision_end_image,
        )


class IAMCCS_WanSVIToFLFBridgePro(WanImageMotionProAdvanced):
    """Bridge-oriented WAN node for SVI -> FLF -> SVI transitions.

    V1 adds an optional middle bridge image injected into the temporal conditioning
    path, allowing difficult SVI-to-FLF jumps to be split into two smaller visual
    transitions inside one generated bridge segment.
    """

    @classmethod
    def INPUT_TYPES(cls):
        base = WanImageMotionProAdvanced.INPUT_TYPES()
        required = dict(base["required"])
        required.update(
            {
                "bridge_position": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Where to inject the optional bridge image along the generated clip (0=start, 0.5=middle, 1=end).",
                    },
                ),
                "bridge_strength": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "How strongly the bridge image influences the selected middle latent slots.",
                    },
                ),
                "bridge_slots": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "tooltip": "How many temporal latent slots to affect around the bridge position.",
                    },
                ),
                "bridge_mode": (
                    [
                        "disabled",
                        "blend_middle",
                        "hard_middle",
                    ],
                    {
                        "default": "blend_middle",
                        "tooltip": (
                            "disabled: ignore bridge image. "
                            "blend_middle: blend the bridge latent into middle slots. "
                            "hard_middle: lock the middle slots to the bridge latent."
                        ),
                    },
                ),
            }
        )
        optional = dict(base.get("optional", {}))
        optional.update(
            {
                "bridge_image": ("IMAGE",),
                "vae": ("VAE",),
            }
        )
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("positive", "negative", "latent", "trim_slots")
    FUNCTION = "apply"
    CATEGORY = "IAMCCS/Wan/_Legacy"

    _log = logging.getLogger("IAMCCS.WanSVIToFLFBridgePro")

    def apply(
        self,
        positive,
        negative,
        length,
        anchor_samples,
        motion_latent_count,
        motion,
        motion_mode,
        add_reference_latents,
        latent_precision,
        vram_profile,
        include_padding_in_motion,
        safety_preset="safe",
        use_end_frame=True,
        end_transition_frames=0,
        end_lock_slots=1,
        lock_start_slots=1,
        diagnostic_log=False,
        use_prev_samples=True,
        end_overshoot_slots=0,
        latent_refresh=0.0,
        delta_max=0.0,
        clip_vision_mode="auto",
        raw_mode=False,
        prev_samples_profile="direct",
        prev_skip_last_slots=0,
        prev_tail_pick_mode="latest",
        prev_anchor_blend=0.0,
        bridge_position=0.5,
        bridge_strength=0.5,
        bridge_slots=1,
        bridge_mode="blend_middle",
        prev_samples=None,
        end_samples=None,
        clip_vision_start_image=None,
        clip_vision_end_image=None,
        bridge_image=None,
        vae=None,
        **_kwargs,
    ):
        bridge_enabled = (
            str(bridge_mode) != "disabled"
            and bridge_image is not None
            and vae is not None
            and float(bridge_strength) > 0.0
            and int(bridge_slots) > 0
            and not bool(raw_mode)
        )

        if not bridge_enabled:
            if str(bridge_mode) != "disabled" and (bridge_image is None or vae is None):
                self._log.info(
                    "[WanSVIToFLFBridgePro] bridge requested but bridge_image or vae is missing — falling back to WanImageMotionPro behavior."
                )
            if bool(raw_mode) and str(bridge_mode) != "disabled":
                self._log.info(
                    "[WanSVIToFLFBridgePro] raw_mode=True — bridge injection is skipped to preserve strict FLF compatibility."
                )
            return super().apply(
                positive,
                negative,
                length,
                anchor_samples,
                motion_latent_count,
                motion,
                motion_mode,
                add_reference_latents,
                latent_precision,
                vram_profile,
                include_padding_in_motion,
                safety_preset=safety_preset,
                use_end_frame=use_end_frame,
                end_transition_frames=end_transition_frames,
                end_lock_slots=end_lock_slots,
                lock_start_slots=lock_start_slots,
                diagnostic_log=diagnostic_log,
                use_prev_samples=use_prev_samples,
                end_overshoot_slots=end_overshoot_slots,
                latent_refresh=latent_refresh,
                delta_max=delta_max,
                clip_vision_mode=clip_vision_mode,
                raw_mode=raw_mode,
                prev_samples_profile=prev_samples_profile,
                prev_skip_last_slots=prev_skip_last_slots,
                prev_tail_pick_mode=prev_tail_pick_mode,
                prev_anchor_blend=prev_anchor_blend,
                prev_samples=prev_samples,
                end_samples=end_samples,
                clip_vision_start_image=clip_vision_start_image,
                clip_vision_end_image=clip_vision_end_image,
            )

        with torch.no_grad():
            anchor_latent = anchor_samples["samples"].clone()

            B, C, T_anchor, H, W = anchor_latent.shape

            lock_start_slots_i = int(max(0, min(16, lock_start_slots)))

            anchor_slots_for_cond = T_anchor
            if use_end_frame and end_samples is not None and T_anchor > 1 and lock_start_slots_i <= 1:
                anchor_slots_for_cond = 1
                self._log.info(
                    "[WanSVIToFLFBridgePro] FLF start normalization: T_anchor=%s -> using first %s slot for conditioning.",
                    T_anchor,
                    anchor_slots_for_cond,
                )

            anchor_cond_latent = anchor_latent[:, :, :anchor_slots_for_cond]

            total_latents = (length - 1) // 4 + 1
            _overshoot = (
                int(end_overshoot_slots)
                if (use_end_frame and end_samples is not None and int(end_overshoot_slots) > 0)
                else 0
            )
            total_latents_gen = total_latents + _overshoot

            device = anchor_latent.device
            dtype = anchor_latent.dtype

            empty_latent_dtype = self._pick_empty_latent_dtype(dtype, latent_precision)
            empty_latent = torch.zeros(
                [B, 16, total_latents_gen, H, W],
                device=comfy.model_management.intermediate_device(),
                dtype=empty_latent_dtype,
            )

            motion_latent = None
            T_motion = 0
            t_prev = 0
            usable_prev_t = 0
            resolved_tail_pick_mode, resolved_skip_last_slots, resolved_anchor_blend = _resolve_prev_tail_profile(
                prev_samples_profile,
                tail_pick_mode=prev_tail_pick_mode,
                skip_last_slots=prev_skip_last_slots,
                anchor_blend=prev_anchor_blend,
            )
            has_prev = bool(use_prev_samples) and prev_samples is not None and motion_latent_count != 0

            if has_prev:
                motion_latent, T_motion, t_prev, usable_prev_t = _extract_prev_motion_tail(
                    prev_samples,
                    motion_latent_count,
                    tail_pick_mode=resolved_tail_pick_mode,
                    skip_last_slots=resolved_skip_last_slots,
                    anchor_reference=anchor_latent,
                    anchor_blend=resolved_anchor_blend,
                )
                has_prev = motion_latent is not None and T_motion > 0

            _lr = float(latent_refresh)
            _dm = float(delta_max)
            if motion_latent is not None and _lr > 0:
                motion_mean = motion_latent.mean(dim=(2, 3, 4), keepdim=True)
                anchor_mean = anchor_latent.mean(dim=(2, 3, 4), keepdim=True)
                dc_drift = motion_mean - anchor_mean
                if _dm > 0:
                    dc_drift = _dm * torch.tanh(dc_drift / _dm)
                motion_latent = motion_latent - (dc_drift * _lr)

            if motion_latent is None:
                padding_size = total_latents_gen - anchor_slots_for_cond
                image_cond_latent = anchor_cond_latent
            else:
                padding_size = total_latents_gen - anchor_slots_for_cond - T_motion
                image_cond_latent = torch.cat([anchor_cond_latent, motion_latent], dim=2)

            padding_size = max(0, padding_size)
            if padding_size > 0:
                padding = torch.zeros(B, C, padding_size, H, W, dtype=dtype, device=device)
                padding = comfy.latent_formats.Wan21().process_out(padding)
                image_cond_latent = torch.cat([image_cond_latent, padding], dim=2)

            if image_cond_latent.shape[2] > total_latents_gen:
                image_cond_latent = image_cond_latent[:, :, :total_latents_gen]
            elif image_cond_latent.shape[2] < total_latents_gen:
                image_cond_latent = image_cond_latent[:, :, :total_latents_gen]

            end_t_fix_early = 0
            if end_samples is not None and use_end_frame:
                _e = end_samples["samples"]
                if _e.shape[1] == C and _e.shape[3] == H and _e.shape[4] == W:
                    end_t_fix_early = min(_e.shape[2], total_latents_gen, end_lock_slots)

            if include_padding_in_motion and has_prev and (T_anchor + T_motion) > 1:
                effective_latents_base = total_latents_gen
            else:
                effective_latents_base = min(total_latents_gen, anchor_slots_for_cond + T_motion)
            effective_latents = max(1, min(effective_latents_base, total_latents_gen - end_t_fix_early))

            motion_mode_effective = motion_mode

            image_cond_latent = self._apply_motion_amplitude(
                image_cond_latent,
                real_latents=effective_latents,
                anchor_latents=anchor_slots_for_cond,
                motion_latents=T_motion,
                motion_amplitude=motion,
                motion_mode=motion_mode_effective,
                vram_profile=vram_profile,
                safety_preset=safety_preset,
            )

            bridge_slots_i = max(1, min(int(bridge_slots), int(total_latents_gen)))
            bridge_strength_f = float(max(0.0, min(1.0, bridge_strength)))
            bridge_mid = round(float(max(0.0, min(1.0, bridge_position))) * max(0, total_latents_gen - 1))
            bridge_start = max(0, min(int(bridge_mid - ((bridge_slots_i - 1) // 2)), int(total_latents_gen - bridge_slots_i)))
            bridge_end = bridge_start + bridge_slots_i

            bridge_latent = _encode_bridge_image_to_latent(vae, bridge_image, H * 8, W * 8)
            bridge_latent = _match_latent_batch(anchor_latent, bridge_latent)
            bridge_slot = bridge_latent[:, :, :1].to(device=image_cond_latent.device, dtype=image_cond_latent.dtype)
            bridge_block = bridge_slot.expand(-1, -1, bridge_slots_i, -1, -1).contiguous()

            current_middle = image_cond_latent[:, :, bridge_start:bridge_end]
            if str(bridge_mode) == "hard_middle":
                image_cond_latent[:, :, bridge_start:bridge_end] = bridge_block
            else:
                image_cond_latent[:, :, bridge_start:bridge_end] = torch.lerp(
                    current_middle,
                    bridge_block,
                    bridge_strength_f,
                )

            if diagnostic_log:
                self._log.info(
                    "[WanSVIToFLFBridgePro] bridge_image injected: mode=%s slots=[%s:%s] strength=%.2f position=%.2f",
                    bridge_mode,
                    bridge_start,
                    bridge_end,
                    bridge_strength_f,
                    float(bridge_position),
                )

            end_t_fix = 0
            if end_samples is not None and use_end_frame:
                end_latent = end_samples["samples"].clone()
                if end_latent.shape[0] == 1 and B > 1:
                    end_latent = end_latent.repeat(B, 1, 1, 1, 1)
                if end_latent.shape[1] == C and end_latent.shape[3] == H and end_latent.shape[4] == W:
                    T_end = end_latent.shape[2]
                    end_t_fix = min(T_end, total_latents_gen, end_lock_slots)
                    if end_t_fix > 0:
                        image_cond_latent[:, :, -end_t_fix:] = end_latent[:, :, -end_t_fix:]
                        if end_transition_frames > 0:
                            self._log.warning(
                                "[WanSVIToFLFBridgePro] end_transition_frames=%s is deprecated and disabled.",
                                end_transition_frames,
                            )

            frame_mask = torch.ones((1, 1, total_latents_gen * 4, H, W), device=device, dtype=torch.float32)
            if lock_start_slots_i > 0:
                frame_mask[:, :, :lock_start_slots_i * 4] = 0.0
            if end_t_fix > 0:
                frame_mask[:, :, -end_t_fix * 4:] = 0.0

            bridge_frame_slice = slice(bridge_start * 4, bridge_end * 4)
            if str(bridge_mode) == "hard_middle":
                frame_mask[:, :, bridge_frame_slice] = 0.0
            else:
                target_mask = max(0.0, 1.0 - bridge_strength_f)
                frame_mask[:, :, bridge_frame_slice] = torch.minimum(
                    frame_mask[:, :, bridge_frame_slice],
                    torch.full_like(frame_mask[:, :, bridge_frame_slice], target_mask),
                )

            mask = frame_mask.view(1, frame_mask.shape[2] // 4, 4, H, W).transpose(1, 2).contiguous()

            if image_cond_latent.dtype != torch.float32:
                image_cond_latent = image_cond_latent.float()

            cond_device = empty_latent.device
            if image_cond_latent.device != cond_device:
                image_cond_latent = image_cond_latent.to(cond_device)
                mask = mask.to(cond_device)

            positive = node_helpers.conditioning_set_values(
                positive, {"concat_latent_image": image_cond_latent, "concat_mask": mask}
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"concat_latent_image": image_cond_latent, "concat_mask": mask}
            )

            if add_reference_latents:
                ref_latent = anchor_latent[:, :, 0:1]
                positive = node_helpers.conditioning_set_values(
                    positive, {"reference_latents": [ref_latent]}, append=True
                )
                negative = node_helpers.conditioning_set_values(
                    negative, {"reference_latents": [torch.zeros_like(ref_latent)]}, append=True
                )

            _cv_out = None
            _mode = str(clip_vision_mode) if clip_vision_mode else "auto"
            if _mode != "none":
                if _mode == "end_only":
                    _cv_out = clip_vision_end_image if clip_vision_end_image is not None else clip_vision_start_image
                elif _mode == "start_only":
                    _cv_out = clip_vision_start_image if clip_vision_start_image is not None else clip_vision_end_image
                else:
                    if clip_vision_start_image is not None and clip_vision_end_image is not None:
                        try:
                            anchor_cmp = anchor_samples["samples"]
                            end_cmp = end_samples["samples"] if (use_end_frame and end_samples is not None) else None
                            if end_cmp is not None:
                                compare_slots = min(anchor_cmp.shape[2], end_cmp.shape[2])
                                if compare_slots > 0 and torch.allclose(
                                    anchor_cmp[:, :, :compare_slots].float(),
                                    end_cmp[:, :, :compare_slots].float(),
                                    rtol=1e-4,
                                    atol=1e-5,
                                ):
                                    _cv_out = clip_vision_end_image
                                else:
                                    _states = torch.cat(
                                        [
                                            clip_vision_start_image.penultimate_hidden_states,
                                            clip_vision_end_image.penultimate_hidden_states,
                                        ],
                                        dim=-2,
                                    )
                                    _cv_out = comfy.clip_vision.Output()
                                    _cv_out.penultimate_hidden_states = _states
                            else:
                                _states = torch.cat(
                                    [
                                        clip_vision_start_image.penultimate_hidden_states,
                                        clip_vision_end_image.penultimate_hidden_states,
                                    ],
                                    dim=-2,
                                )
                                _cv_out = comfy.clip_vision.Output()
                                _cv_out.penultimate_hidden_states = _states
                        except Exception:
                            _cv_out = clip_vision_end_image
                    elif clip_vision_start_image is not None:
                        _cv_out = clip_vision_start_image
                    elif clip_vision_end_image is not None:
                        _cv_out = clip_vision_end_image

            if _cv_out is not None:
                positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": _cv_out})
                negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": _cv_out})

            out_latent = {"samples": empty_latent}
            return (positive, negative, out_latent, _overshoot)


class IAMCCS_WanSVIToFLFBridgePro_Simple(IAMCCS_WanSVIToFLFBridgePro):
    """Simplified bridge node for SVI -> FLF -> SVI use.

    Exposes only the controls that materially affect bridge behavior,
    while keeping the legacy/full node available for backward compatibility.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "length": ("INT", {"default": 81, "min": 1, "max": 16384, "step": 4}),
                "anchor_samples": ("LATENT",),
                "motion_latent_count": ("INT", {"default": 1, "min": 0, "max": 16, "step": 1}),
                "motion": ("FLOAT", {"default": 1.15, "min": 1.0, "max": 2.0, "step": 0.05}),
                "motion_mode": (
                    [
                        "motion_only (prev_samples)",
                        "all_nonfirst (anchor+motion)",
                    ],
                    {"default": "motion_only (prev_samples)"},
                ),
                "add_reference_latents": ("BOOLEAN", {"default": False}),
                "vram_profile": (
                    [
                        "normal",
                        "chunked_blocks_2",
                        "chunked_blocks_4",
                        "loop_per_frame (lowest_vram)",
                        "cpu_offload (slowest)",
                    ],
                    {"default": "normal"},
                ),
                "include_padding_in_motion": ("BOOLEAN", {"default": False}),
                "safety_preset": (
                    [
                        "base",
                        "safe",
                        "safer",
                        "legacy",
                    ],
                    {"default": "safe"},
                ),
                "use_end_frame": ("BOOLEAN", {"default": True}),
                "end_lock_slots": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
                "lock_start_slots": ("INT", {"default": 1, "min": 0, "max": 16, "step": 1}),
                "end_overshoot_slots": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1}),
                "latent_refresh": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "delta_max": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "clip_vision_mode": (
                    [
                        "auto",
                        "end_only",
                        "start_only",
                        "none",
                    ],
                    {"default": "end_only"},
                ),
                "prev_skip_last_slots": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 16,
                        "step": 1,
                        "tooltip": (
                            "Ignore the last N temporal latent slots from prev_samples before extracting the motion tail. "
                            "Useful when the carried SVI tail keeps too much of the previous scene and causes background crossfade."
                        ),
                    },
                ),
                "bridge_position": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                    },
                ),
                "bridge_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                    },
                ),
                "bridge_slots": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                    },
                ),
                "bridge_mode": (
                    [
                        "disabled",
                        "blend_middle",
                        "hard_middle",
                    ],
                    {"default": "hard_middle"},
                ),
            },
            "optional": {
                "prev_samples": ("LATENT",),
                "end_samples": ("LATENT",),
                "clip_vision_start_image": ("CLIP_VISION_OUTPUT",),
                "clip_vision_end_image": ("CLIP_VISION_OUTPUT",),
                "bridge_image": ("IMAGE",),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("positive", "negative", "latent", "trim_slots")
    FUNCTION = "apply"
    CATEGORY = "IAMCCS/Wan"

    _log = logging.getLogger("IAMCCS.WanSVIToFLFBridgePro")

    def apply(
        self,
        positive,
        negative,
        length,
        anchor_samples,
        motion_latent_count,
        motion,
        motion_mode,
        add_reference_latents,
        vram_profile,
        include_padding_in_motion,
        safety_preset="safe",
        use_end_frame=True,
        end_lock_slots=1,
        lock_start_slots=1,
        end_overshoot_slots=0,
        latent_refresh=0.0,
        delta_max=0.0,
        clip_vision_mode="end_only",
        prev_skip_last_slots=1,
        bridge_position=0.5,
        bridge_strength=1.0,
        bridge_slots=2,
        bridge_mode="hard_middle",
        prev_samples=None,
        end_samples=None,
        clip_vision_start_image=None,
        clip_vision_end_image=None,
        bridge_image=None,
        vae=None,
        **_kwargs,
    ):
        return super().apply(
            positive=positive,
            negative=negative,
            length=length,
            anchor_samples=anchor_samples,
            motion_latent_count=motion_latent_count,
            motion=motion,
            motion_mode=motion_mode,
            add_reference_latents=add_reference_latents,
            latent_precision="fp32",
            vram_profile=vram_profile,
            include_padding_in_motion=include_padding_in_motion,
            safety_preset=safety_preset,
            use_end_frame=use_end_frame,
            end_transition_frames=0,
            end_lock_slots=end_lock_slots,
            lock_start_slots=lock_start_slots,
            diagnostic_log=False,
            use_prev_samples=True,
            end_overshoot_slots=end_overshoot_slots,
            latent_refresh=latent_refresh,
            delta_max=delta_max,
            clip_vision_mode=clip_vision_mode,
            raw_mode=False,
            prev_samples_profile="direct",
            prev_skip_last_slots=prev_skip_last_slots,
            prev_tail_pick_mode="latest",
            prev_anchor_blend=0.0,
            bridge_position=bridge_position,
            bridge_strength=bridge_strength,
            bridge_slots=bridge_slots,
            bridge_mode=bridge_mode,
            prev_samples=prev_samples,
            end_samples=end_samples,
            clip_vision_start_image=clip_vision_start_image,
            clip_vision_end_image=clip_vision_end_image,
            bridge_image=bridge_image,
            vae=vae,
        )


NODE_CLASS_MAPPINGS = {
    "IAMCCS_WanImageMotion": IAMCCS_WanImageMotion,
    # Backward-compat alias: workflow JSONs saved with the _AdaIN name still load.
    "IAMCCS_WanImageMotion_AdaIN": IAMCCS_WanImageMotion,
    "WanImageMotionPro": WanImageMotionPro,
    "IAMCCS_WanImageMotionPro_AdaIN": WanImageMotionPro,
    # Hidden alias: present for backward-compat (saved workflows load correctly)
    # but intentionally absent from NODE_DISPLAY_NAME_MAPPINGS → never shows in the menu.
    "IAMCCS_WanImageMotionPro": WanImageMotionPro,
    "IAMCCS_WanImageMotionPro_Simple": IAMCCS_WanImageMotionPro_Simple,
    "IAMCCS_WanSVIToFLFBridgePro": IAMCCS_WanSVIToFLFBridgePro,
    "IAMCCS_WanSVIToFLFBridgePro_Simple": IAMCCS_WanSVIToFLFBridgePro_Simple,
    "WanMotionProTrimmer": WanMotionProTrimmer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_WanImageMotion": "IAMCCS WanImageMotion",
    "IAMCCS_WanImageMotion_AdaIN": "IAMCCS WanImageMotion",
    "WanImageMotionPro": "WanImageMotionPro",
    "IAMCCS_WanImageMotionPro_AdaIN": "WanImageMotionPro",
    "IAMCCS_WanImageMotionPro_Simple": "WanImageMotionPro Simple",
    "IAMCCS_WanSVIToFLFBridgePro_Simple": "WanSVIToFLFBridgePro",
    "WanMotionProTrimmer": "WanMotionProTrimmer",
}
