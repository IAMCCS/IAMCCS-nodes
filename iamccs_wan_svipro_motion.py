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
    # Keep changes non-invasive unless motion is pushed above the common safe zone.
    only_if_gt = 1.15

    legacy = {
        "enabled": True,
        "only_if_gt": float("inf"),
        "mean_mode": "frame_scalar",
        "limiter_mode": "hard",
        "limiter_limit": 6.0,
        "ramp_frames": 0,
    }

    safe = {
        "enabled": True,
        "only_if_gt": only_if_gt,
        "mean_mode": "per_channel",
        "limiter_mode": "tanh",
        "limiter_limit": 6.0,
        "ramp_frames": 2,
    }

    safer = {
        "enabled": True,
        "only_if_gt": only_if_gt,
        "mean_mode": "per_channel",
        "limiter_mode": "tanh",
        # slightly tighter limiter to avoid outliers at high motion
        "limiter_limit": 5.5,
        "ramp_frames": 4,
    }

    if safety_preset == "legacy":
        return legacy
    if safety_preset == "safer":
        return safer
    # default
    return safe


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
                        "safe",
                        "safer",
                        "legacy",
                    ],
                    {
                        "default": "safe",
                        "tooltip": (
                            "Safe preset reduces color/seam artifacts when motion > 1.15. "
                            "It applies per-channel stabilization, smooth limiter, and a short ramp. "
                            "Set legacy to use the original hard-clamp behavior."
                        ),
                    },
                ),
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
        if motion_amplitude is None or motion_amplitude <= 1.0:
            return image_cond_latent

        if image_cond_latent.shape[2] <= 1:
            return image_cond_latent

        base_latent = image_cond_latent[:, :, 0:1]  # first latent frame

        preset = _preset_params(safety_preset, motion_amplitude)
        # Non-invasive: if motion is within the usual safe zone, keep legacy behavior.
        # This preserves 1:1 results for typical workflows.
        if motion_amplitude <= preset["only_if_gt"]:
            preset = _preset_params("legacy", motion_amplitude)

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
              safety_preset="safe", prev_samples=None):
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
            effective_latents = total_latents if include_padding_in_motion else min(total_latents, T_anchor + T_motion)

            motion_mode_effective = motion_mode
            # When include_padding_in_motion=True, always extend motion boost to all non-anchor
            # frames (including zero-padded slots), regardless of whether prev_samples are set.
            # This ensures consistent full-range boost across all chunks in a chained workflow.
            if motion_mode == "motion_only (prev_samples)" and include_padding_in_motion:
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
                safety_preset=safety_preset,
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
                # Keep this at the end to preserve existing widgets_values indexing in saved workflows.
                "safety_preset": (
                    [
                        "safe",
                        "safer",
                        "legacy",
                    ],
                    {
                        "default": "safe",
                        "tooltip": (
                            "Safe preset reduces color/seam artifacts when motion > 1.15. "
                            "Set legacy to use original hard-clamp behavior."
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
                        "default": 4,
                        "min": 0,
                        "max": 32,
                        "step": 1,
                        "tooltip": (
                            "Number of latent frames before the end-lock zone to gradually blend toward "
                            "the end frame. 0 = hard cut (original behavior). Higher = smoother convergence."
                        ),
                    },
                ),
            },
            "optional": {
                # prev_samples is optional – mirrors original FLF node and IAMCCS_WanImageMotion.
                # apply() already handles None gracefully.
                "prev_samples": ("LATENT",),
                "end_samples": ("LATENT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "apply"
    CATEGORY = "IAMCCS/video"

    _log = logging.getLogger("IAMCCS.WanImageMotionPro")

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
        if motion_amplitude is None or motion_amplitude <= 1.0:
            return image_cond_latent

        if image_cond_latent.shape[2] <= 1:
            return image_cond_latent

        base_latent = image_cond_latent[:, :, 0:1]

        preset = _preset_params(safety_preset, motion_amplitude)
        if motion_amplitude <= preset["only_if_gt"]:
            preset = _preset_params("legacy", motion_amplitude)

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
        prev_samples=None,
        end_samples=None,
    ):
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

            motion_latent = None
            T_motion = 0
            # In the original FLF node, prev_samples is a required socket.
            # If a workflow leaves it disconnected, ComfyUI may pass None.
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
            if padding_size > 0:
                padding = torch.zeros(B, C, padding_size, H, W, dtype=dtype, device=device)
                padding = comfy.latent_formats.Wan21().process_out(padding)
                image_cond_latent = torch.cat([image_cond_latent, padding], dim=2)

            # FLF/SVI reference behavior: enforce exact temporal length.
            if image_cond_latent.shape[2] > total_latents:
                image_cond_latent = image_cond_latent[:, :, :total_latents]
            elif image_cond_latent.shape[2] < total_latents:
                image_cond_latent = image_cond_latent[:, :, :total_latents]

            # Pre-compute end_t_fix so we can exclude the end-locked zone from motion amplitude.
            # Motion should NOT touch slots that will be hard-locked to end_samples: scaling those
            # intermediate latents would generate noise that hurts the model's first→last interpolation.
            end_t_fix_early = 0
            if end_samples is not None:
                _e = end_samples["samples"]
                if (
                    _e.shape[1] == C
                    and _e.shape[3] == H
                    and _e.shape[4] == W
                ):
                    end_t_fix_early = min(_e.shape[2], total_latents)

            # Motion boost applied before FLF overwrite.
            # Cap effective_latents so motion never reaches into the end-locked zone.
            effective_latents_base = total_latents if include_padding_in_motion else min(total_latents, T_anchor + T_motion)
            effective_latents = max(1, min(effective_latents_base, total_latents - end_t_fix_early))

            motion_mode_effective = motion_mode
            # When include_padding_in_motion=True, always extend motion boost to all non-anchor
            # frames (including zero-padded slots), regardless of whether prev_samples are set.
            if motion_mode == "motion_only (prev_samples)" and include_padding_in_motion:
                motion_mode_effective = "all_nonfirst (anchor+motion)"

            try:
                free_vram, total_vram = comfy.model_management.get_free_memory(device)
            except Exception:
                free_vram, total_vram = None, None

            self._log.info(
                "[WanImageMotionPro] length=%s -> total_latents=%s | motion=%s | mode=%s | vram_profile=%s | latent_precision=%s | add_reference_latents=%s | include_padding_in_motion=%s | use_end_frame=%s | end_transition_frames=%s",
                length,
                total_latents,
                motion,
                motion_mode,
                vram_profile,
                latent_precision,
                add_reference_latents,
                include_padding_in_motion,
                use_end_frame,
                end_transition_frames,
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
            if free_vram is not None:
                self._log.info("[WanImageMotionPro] free_vram=%s total_vram=%s", free_vram, total_vram)

            if motion_mode_effective == "motion_only (prev_samples)":
                motion_start = T_anchor
                motion_end = min(T_anchor + T_motion, effective_latents)
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
                anchor_latents=T_anchor,
                motion_latents=T_motion,
                motion_amplitude=motion,
                motion_mode=motion_mode_effective,
                vram_profile=vram_profile,
                safety_preset=safety_preset,
            )

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
                    end_t_fix = min(T_end, total_latents)
                    if end_t_fix > 0:
                        image_cond_latent[:, :, -end_t_fix:] = end_latent[:, :, -end_t_fix:]

                        # Smooth transition zone: blend image_cond_latent toward the end frame over
                        # `end_transition_frames` slots immediately before the hard-locked zone.
                        # This prevents the abrupt jump / stop-motion artifact at chunk boundaries.
                        if end_transition_frames > 0:
                            # Use the first frame of end_latent as the blend target (entry to end seq).
                            end_ref = end_latent[:, :, 0:1].to(
                                device=image_cond_latent.device, dtype=image_cond_latent.dtype
                            )
                            trans_end = total_latents - end_t_fix       # exclusive upper bound
                            trans_start = max(1, trans_end - end_transition_frames)  # inclusive
                            trans_count = trans_end - trans_start
                            if trans_count > 0:
                                self._log.info(
                                    "[WanImageMotionPro] end_transition zone=[%s:%s] (%s frame(s)) blending toward end frame",
                                    trans_start, trans_end, trans_count,
                                )
                                # Smoothstep weights from 0 (far from end) → 1 (adjacent to locked zone)
                                x_vals = torch.linspace(
                                    0.0, 1.0,
                                    steps=trans_count + 2,
                                    device=image_cond_latent.device,
                                    dtype=image_cond_latent.dtype,
                                )[1:-1]  # exclude the 0 and 1 endpoints
                                for i in range(trans_count):
                                    alpha = _smoothstep(x_vals[i : i + 1]).item()
                                    t = trans_start + i
                                    image_cond_latent[:, :, t : t + 1] = (
                                        (1.0 - alpha) * image_cond_latent[:, :, t : t + 1]
                                        + alpha * end_ref
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

            # Mask: lock first slot + lock last end_t_fix slots.
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

            if add_reference_latents:
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
    "WanImageMotionPro": WanImageMotionPro,
    "IAMCCS_WanImageMotionPro": WanImageMotionPro,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_WanImageMotion": "IAMCCS WanImageMotion",
    "WanImageMotionPro": "WanImageMotionPro",
    "IAMCCS_WanImageMotionPro": "WanImageMotionPro",
}
