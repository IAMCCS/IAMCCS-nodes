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
            },
            "optional": {
                "prev_samples": ("LATENT",),
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
              safety_preset="safe", lock_start_slots=1, diagnostic_log=False, use_prev_samples=True, prev_samples=None):
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

            has_prev = bool(use_prev_samples) and prev_samples is not None and motion_latent_count != 0

            if (prev_samples is None) or (not use_prev_samples) or (motion_latent_count == 0):
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

            mask = torch.ones((1, 1, empty_latent.shape[2], H, W), device=device, dtype=dtype)
            lock_start_slots_i = int(max(0, min(16, lock_start_slots)))
            if lock_start_slots_i > 0:
                mask[:, :, :lock_start_slots_i] = 0.0

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
    CATEGORY = "IAMCCS/Wan"

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
            has_prev = bool(use_prev_samples) and prev_samples is not None and motion_latent_count != 0

            if (prev_samples is None) or (not use_prev_samples) or (motion_latent_count == 0):
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
                    end_t_fix_early = min(_e.shape[2], total_latents, end_lock_slots)

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
                effective_latents_base = total_latents
            else:
                effective_latents_base = min(total_latents, T_anchor + T_motion)
            effective_latents = max(1, min(effective_latents_base, total_latents - end_t_fix_early))

            motion_mode_effective = motion_mode
            # NOTE: we intentionally do NOT auto-switch to all_nonfirst here.
            # Auto-switching would end up modifying padding slots on the first segment.

            try:
                free_vram, total_vram = comfy.model_management.get_free_memory(device)
            except Exception:
                free_vram, total_vram = None, None

            self._log.info(
                "[WanImageMotionPro] length=%s -> total_latents=%s | motion=%s | mode=%s | vram_profile=%s | latent_precision=%s | add_reference_latents=%s | include_padding_in_motion=%s | use_end_frame=%s | end_transition_frames=%s | end_lock_slots=%s | lock_start_slots=%s",
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
                self._log.info("[WanImageMotionPro] prev_samples connected but use_prev_samples=False — prev ignored.")
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
                        "[WanImageMotionPro][diag] end_t_fix_early=%s (motion cap) use_end_frame=%s end_lock_slots=%s",
                        end_t_fix_early, use_end_frame, end_lock_slots,
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
                    end_t_fix = min(T_end, total_latents, end_lock_slots)
                    if end_t_fix > 0:
                        self._log.info(
                            "[WanImageMotionPro] end lock: T_end=%s end_lock_slots=%s -> end_t_fix=%s (locking latent slots [%s:%s] = last ~%s video frames)",
                            T_end, end_lock_slots, end_t_fix,
                            total_latents - end_t_fix, total_latents,
                            end_t_fix * 4,
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

            # Mask: lock first slot + lock last end_t_fix slots.
            mask = torch.ones((1, 1, total_latents, H, W), device=device, dtype=dtype)
            lock_start_slots_i = int(max(0, min(16, lock_start_slots)))
            if lock_start_slots_i > 0:
                mask[:, :, :lock_start_slots_i] = 0.0
            if end_t_fix > 0:
                mask[:, :, -end_t_fix:] = 0.0

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

            out_latent = {"samples": empty_latent}
            return (positive, negative, out_latent)


NODE_CLASS_MAPPINGS = {
    "IAMCCS_WanImageMotion": IAMCCS_WanImageMotion,
    "WanImageMotionPro": WanImageMotionPro,
    # Hidden alias: present for backward-compat (saved workflows load correctly)
    # but intentionally absent from NODE_DISPLAY_NAME_MAPPINGS → never shows in the menu.
    "IAMCCS_WanImageMotionPro": WanImageMotionPro,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_WanImageMotion": "IAMCCS WanImageMotion",
    "WanImageMotionPro": "WanImageMotionPro",
}
