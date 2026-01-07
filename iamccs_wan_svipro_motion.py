# iamccs_wan_svipro_motion.py
# ===============================================================
# IAMCCS WanImageMotion
# Drop-in replacement for KJNodes WanImageToVideoSVIPro with motion control
# ===============================================================

import torch
import logging
import comfy.model_management
import comfy.latent_formats
import node_helpers


class IAMCCS_WanImageMotion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "length": ("INT", {"default": 81, "min": 1, "max": 16384, "step": 4}),
                "anchor_samples": ("LATENT",),
                "motion_latent_count": ("INT", {"default": 1, "min": 0, "max": 128, "step": 1}),
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
                    {"default": "auto"},
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
            },
            "optional": {
                "prev_samples": ("LATENT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "apply"
    CATEGORY = "IAMCCS/video"

    _log = logging.getLogger("IAMCCS.WanImageMotion")

    def _pick_empty_latent_dtype(self, anchor_dtype: torch.dtype, latent_precision: str) -> torch.dtype:
        if latent_precision == "normal":
            # Backward-compat alias for older workflows.
            return anchor_dtype
        if latent_precision == "fp32":
            return torch.float32
        if latent_precision == "fp16":
            return torch.float16
        return anchor_dtype

    def _apply_motion_amplitude(self, image_cond_latent: torch.Tensor, *, real_latents: int, anchor_latents: int,
                               motion_latents: int, motion_amplitude: float, motion_mode: str,
                               vram_profile: str) -> torch.Tensor:
        if motion_amplitude is None or motion_amplitude <= 1.0:
            return image_cond_latent

        if image_cond_latent.shape[2] <= 1:
            return image_cond_latent

        base_latent = image_cond_latent[:, :, 0:1]  # first latent frame

        def _scale_slice_gpu(view_slice: torch.Tensor) -> torch.Tensor:
            # view_slice: [B,C,T,H,W]
            # VRAM-optimized variant: keep only one full-sized temporary tensor.
            with torch.no_grad():
                diff = view_slice - base_latent
                diff_mean = diff.mean(dim=(1, 3, 4), keepdim=True)
                diff.sub_(diff_mean)
                diff.mul_(motion_amplitude)
                diff.add_(diff_mean)
                diff.add_(base_latent)
                diff.clamp_(-6, 6)
                return diff

        def _apply_to_range(start: int, end: int) -> torch.Tensor:
            # Applies scaling to out[:, :, start:end] according to VRAM profile.
            # NOTE: caller controls what "real_latents" means. By default it excludes padding;
            # if include_padding_in_motion is enabled, caller may set real_latents=total_latents.
            if end <= start:
                return out

            if vram_profile == "normal":
                out[:, :, start:end] = _scale_slice_gpu(out[:, :, start:end])
                return out

            if vram_profile in ("chunked_blocks_2", "chunked_blocks_4"):
                block = 2 if vram_profile == "chunked_blocks_2" else 4
                t = start
                while t < end:
                    t2 = min(end, t + block)
                    out[:, :, t:t2] = _scale_slice_gpu(out[:, :, t:t2])
                    t = t2
                return out

            if vram_profile == "loop_per_frame (lowest_vram)":
                for t in range(start, end):
                    out[:, :, t:t+1] = _scale_slice_gpu(out[:, :, t:t+1])
                return out

            if vram_profile == "cpu_offload (slowest)":
                # Offload just the slice (and base frame) to CPU, then copy back.
                # This can help in extreme VRAM limits but is much slower.
                with torch.no_grad():
                    device = out.device
                    base_cpu = base_latent.detach().to("cpu")
                    slice_cpu = out[:, :, start:end].detach().to("cpu")
                    diff = slice_cpu - base_cpu
                    diff_mean = diff.mean(dim=(1, 3, 4), keepdim=True)
                    diff.sub_(diff_mean)
                    diff.mul_(motion_amplitude)
                    diff.add_(diff_mean)
                    diff.add_(base_cpu)
                    diff.clamp_(-6, 6)
                    out[:, :, start:end] = diff.to(device)
                return out

            # Fallback
            out[:, :, start:end] = _scale_slice_gpu(out[:, :, start:end])
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
              prev_samples=None):
        with torch.no_grad():
            anchor_latent = anchor_samples["samples"]

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

            motion_latent = None
            T_motion = 0

            has_prev = prev_samples is not None and motion_latent_count != 0

            if prev_samples is None or motion_latent_count == 0:
                padding_size = total_latents - T_anchor
                image_cond_latent = anchor_latent
            else:
                motion_latent = prev_samples["samples"][:, :, -motion_latent_count:]
                T_motion = motion_latent.shape[2]
                padding_size = total_latents - T_anchor - T_motion
                image_cond_latent = torch.cat([anchor_latent, motion_latent], dim=2)

            padding_size = max(0, padding_size)
            padding = torch.zeros(1, C, padding_size, H, W, dtype=dtype, device=device)
            padding = comfy.latent_formats.Wan21().process_out(padding)
            image_cond_latent = torch.cat([image_cond_latent, padding], dim=2)

            # Apply motion amplitude before injecting into conditioning
            effective_latents = total_latents if include_padding_in_motion else min(total_latents, T_anchor + T_motion)

            motion_mode_effective = motion_mode
            # If there are no motion latents, "motion_only" would be a no-op.
            # When include_padding_in_motion=True, treat padding frames as motion targets.
            if motion_mode == "motion_only (prev_samples)" and T_motion == 0 and include_padding_in_motion:
                motion_mode_effective = "all_nonfirst (anchor+motion)"

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
            )

            mask = torch.ones((1, 1, empty_latent.shape[2], H, W), device=device, dtype=dtype)
            mask[:, :, :1] = 0.0

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

            out_latent = {"samples": empty_latent}
            return (positive, negative, out_latent)


NODE_CLASS_MAPPINGS = {
    "IAMCCS_WanImageMotion": IAMCCS_WanImageMotion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_WanImageMotion": "IAMCCS WanImageMotion",
}
