import logging

import comfy.latent_formats
import comfy.model_management
import node_helpers
import torch

from .iamccs_wan_svipro_motion import (
    _apply_soft_limiter,
    _center_diff,
    _preset_params,
    _smoothstep,
)


class WanImageMotionProLegacy:
    """Legacy FLF WanImageMotionPro kept 1:1 with the old D: implementation."""

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
                            "base   : 1:1 with WanImageToVideoSVIProFLF - no amplitude processing at all. "
                            "safe   : per-channel stabilisation + tanh limiter + 2-frame ramp (active at any motion value, recommended default). "
                            "safer  : same as safe but tighter limiter + longer ramp (best above motion=1.5). "
                            "legacy : hard clamp +/-6 + frame-scalar diff centering, mirrors PainterI2V algorithm."
                        ),
                    },
                ),
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
                            "DEPRECATED - keep at 0. "
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
                "lock_start_slots": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 16,
                        "step": 1,
                        "tooltip": (
                            "How many initial latent slots to hard-lock to the start image via concat_mask. "
                            "Default 1 (~ first 4 video frames) matches FLF-style start anchoring. "
                            "Set 0 to allow motion immediately from the first video frame."
                        ),
                    },
                ),
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
            },
            "optional": {
                "prev_samples": ("LATENT",),
                "end_samples": ("LATENT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("positive", "negative", "latent", "trim_slots")
    FUNCTION = "apply"
    CATEGORY = "IAMCCS/Wan"

    _log = logging.getLogger("IAMCCS.WanImageMotionPro")

    def _pick_empty_latent_dtype(self, anchor_dtype: torch.dtype, latent_precision: str) -> torch.dtype:
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
        preset = _preset_params(safety_preset, motion_amplitude)

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
        prev_samples=None,
        end_samples=None,
    ):
        with torch.no_grad():
            anchor_latent = anchor_samples["samples"].clone()
            B, C, T_anchor, H, W = anchor_latent.shape

            total_latents = (length - 1) // 4 + 1

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

            motion_latent = None
            T_motion = 0
            has_prev = bool(use_prev_samples) and prev_samples is not None and motion_latent_count != 0

            if (prev_samples is None) or (not use_prev_samples) or (motion_latent_count == 0):
                padding_size = total_latents_gen - T_anchor
                image_cond_latent = anchor_latent
            else:
                motion_latent = prev_samples["samples"][:, :, -motion_latent_count:]
                T_motion = motion_latent.shape[2]
                padding_size = total_latents_gen - T_anchor - T_motion
                image_cond_latent = torch.cat([anchor_latent, motion_latent], dim=2)

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
                end_tensor = end_samples["samples"]
                if (
                    end_tensor.shape[1] == C
                    and end_tensor.shape[3] == H
                    and end_tensor.shape[4] == W
                ):
                    end_t_fix_early = min(end_tensor.shape[2], total_latents_gen, end_lock_slots)

            if include_padding_in_motion and not has_prev:
                self._log.info(
                    "[WanImageMotionPro] include_padding_in_motion=True but prev_samples is not connected; "
                    "skipping padding boost to avoid static/locked-looking clips."
                )

            if include_padding_in_motion and has_prev and (T_anchor + T_motion) > 1:
                effective_latents_base = total_latents_gen
            else:
                effective_latents_base = min(total_latents_gen, T_anchor + T_motion)
            effective_latents = max(1, min(effective_latents_base, total_latents_gen - end_t_fix_early))

            motion_mode_effective = motion_mode

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
            if prev_samples is not None and not use_prev_samples:
                self._log.info("[WanImageMotionPro] prev_samples connected but use_prev_samples=False - prev ignored.")
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

            if diagnostic_log:
                try:
                    latent_float = image_cond_latent.float()
                    t = int(image_cond_latent.shape[2])
                    x_min = float(latent_float.min().item())
                    x_max = float(latent_float.max().item())
                    x_mean = float(latent_float.mean().item())
                    x_std = float(latent_float.std(unbiased=False).item())
                    self._log.info(
                        "[WanImageMotionPro][diag] concat_latent_image stats: T=%s min=%.4f max=%.4f mean=%.4f std=%.4f",
                        t, x_min, x_max, x_mean, x_std,
                    )

                    base = latent_float[:, :, 0:1]
                    diffs = (latent_float - base).abs().mean(dim=(0, 1, 3, 4))
                    head_n = min(6, t)
                    tail_n = min(6, t)
                    head = [float(v) for v in diffs[:head_n].tolist()]
                    tail = [float(v) for v in diffs[-tail_n:].tolist()]
                    self._log.info(
                        "[WanImageMotionPro][diag] mean|D| vs t0: head=%s tail=%s",
                        head, tail,
                    )
                    self._log.info(
                        "[WanImageMotionPro][diag] end_t_fix_early=%s (motion cap) use_end_frame=%s end_lock_slots=%s | overshoot=%s total_latents_gen=%s",
                        end_t_fix_early, use_end_frame, end_lock_slots, _overshoot, total_latents_gen,
                    )
                except Exception as exc:
                    self._log.warning("[WanImageMotionPro][diag] failed to compute diagnostics: %s", exc)

            end_t_fix = 0
            if end_samples is not None and use_end_frame:
                end_latent = end_samples["samples"].clone()

                if end_latent.shape[0] == 1 and B > 1:
                    end_latent = end_latent.repeat(B, 1, 1, 1, 1)

                if (
                    end_latent.shape[1] == C
                    and end_latent.shape[3] == H
                    and end_latent.shape[4] == W
                ):
                    T_end = end_latent.shape[2]
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

                        if end_transition_frames > 0:
                            self._log.warning(
                                "[WanImageMotionPro] end_transition_frames=%s is DEPRECATED and has been DISABLED. "
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
                    "[WanImageMotionPro] end_samples connected but use_end_frame=False - end lock skipped."
                )

            mask = torch.ones((1, 1, total_latents_gen, H, W), device=device, dtype=dtype)
            lock_start_slots_i = int(max(0, min(16, lock_start_slots)))
            if lock_start_slots_i > 0:
                mask[:, :, :lock_start_slots_i] = 0.0
            if end_t_fix > 0:
                mask[:, :, -end_t_fix:] = 0.0

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

            out_latent = {"samples": empty_latent}
            return (positive, negative, out_latent, _overshoot)


__all__ = ["WanImageMotionProLegacy"]