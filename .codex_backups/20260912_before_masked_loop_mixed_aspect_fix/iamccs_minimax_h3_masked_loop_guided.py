# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later

"""Experimental one-latent Masked Loop Guided sampler for MiniMax H3.

This is deliberately isolated from the stable atomic generator.  It uses the
vendored MMH3Tools in-place loop: technical windows overlap inside one master
AV latent, so there is no decoded join, trim or crossfade between chunks.
"""

from __future__ import annotations

import logging

import folder_paths
import torch

from .iamccs_minimax_h3_atomic_backend import (
    H3_FPS,
    SUPERNODE_LINX_TYPE,
    _accelerate,
    _apply_fasth3_dense_lora,
    _apply_pdd_lora,
    _apply_secondary_lora,
    _apply_turbo_lora,
    _audit_lipsync_audio_lock,
    _chunk,
    _clean_vram_before_decode,
    _fused_turbo_settings,
    _is_fused_turbo_preview,
    _load_image,
    _release_conditioning_models,
    _resize_reference_image,
    _resolve_shotplan,
    _turbo_sampler,
    _turbo_settings,
)
from .iamccs_minimax_h3_pixel_refine_variant import _provider


LOG = logging.getLogger("IAMCCS.MiniMaxH3.MaskedLoopGuided")
CATEGORY = "IAMCCS/MiniMax H3/Experimental Continuity"


def _guide_batch(shotplan):
    contract = shotplan.get("masked_loop_guided")
    if not isinstance(contract, dict) or not bool(contract.get("enabled")):
        raise ValueError(
            "Masked Loop Guided sampler requires task_mode=longvid_masked_loop_guided in IAMCCS H3 Settings."
        )
    guides = contract.get("keyframes") if isinstance(contract.get("keyframes"), list) else []
    indices = contract.get("keyframe_indices") if isinstance(contract.get("keyframe_indices"), list) else []
    if len(guides) < 2 or len(guides) != len(indices):
        raise ValueError("Masked Loop Guided needs at least two matched Shotboard images and frame indices.")
    images = []
    for ordinal, guide in enumerate(guides, start=1):
        path = str(guide.get("source_path", "") or "").strip()
        image = _load_image(path)
        if not torch.is_tensor(image):
            raise ValueError(f"Masked Loop Guided image {ordinal} was not found: {path or 'empty path'}")
        image, _ = _resize_reference_image(image[:1], shotplan, f"masked_loop_guide_{ordinal}")
        images.append(image[:1].detach().to(device="cpu"))
    return torch.cat(images, dim=0), ",".join(str(int(value)) for value in indices)


class IAMCCS_MiniMaxH3MaskedLoopGuidedSampler:
    """Sample all technical windows in place and decode one continuous AV master."""

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers

        samplers = list(comfy.samplers.SAMPLER_NAMES)
        schedulers = list(comfy.samplers.SCHEDULER_NAMES)
        if "res_multistep" in samplers:
            samplers.remove("res_multistep")
            samplers.insert(0, "res_multistep")
        if "simple" in schedulers:
            schedulers.remove("simple")
            schedulers.insert(0, "simple")
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "latent": ("LATENT",),
                "video_vae": ("VAE",),
                "audio_vae": ("VAE",),
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "chunk_index": ("INT", {"forceInput": True}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "seed_stride": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
                "sampler_name": (samplers,),
                "scheduler": (schedulers,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01}),
            },
            # Kept for drop-in compatibility with GenerationBackendV2 graphs;
            # this mode owns continuity internally and therefore ignores it.
            "optional": {"motion_state": ("IAMCCS_H3_MOTION_CONTEXT",)},
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "IMAGE", "LATENT", "INT", "STRING")
    RETURN_NAMES = ("native_frames", "native_audio", "bridge_last_frame", "sampled_latent", "native_fps", "report")
    FUNCTION = "render"
    CATEGORY = CATEGORY

    def render(self, model, positive, latent, video_vae, audio_vae, cine_linx,
               chunk_index, seed, seed_stride, steps, sampler_name, scheduler,
               denoise, shift_video, shift_audio, motion_state=None):
        import nodes as comfy_nodes
        from comfy_extras.nodes_audio import VAEDecodeAudio
        from comfy_extras.nodes_custom_sampler import BasicGuider, BasicScheduler, KSamplerSelect, RandomNoise
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift

        shotplan = _resolve_shotplan(cine_linx)
        if str(shotplan.get("task_mode", "")).strip().lower() != "longvid_masked_loop_guided":
            raise ValueError(
                "This isolated sampler only accepts the MASKED LOOP GUIDED mode; stable IAMCCS modes keep using Generation V2."
            )
        if int(chunk_index) != 0 or len(shotplan.get("chunks", [])) != 1:
            raise ValueError("Masked Loop Guided is one master-latent execution and must receive chunk_index 0.")
        chunk = _chunk(shotplan, 0)
        contract = shotplan["masked_loop_guided"]
        loop = _provider("nodes_looping_sampler")
        if not loop.per_row_mask_is_continuous():
            raise RuntimeError(
                "Masked Loop Guided requires ComfyUI's continuous per-row MiniMax H3 mask support (#15375)."
            )
        if _is_fused_turbo_preview(shotplan):
            raise ValueError(
                "Fused Fast H3 preview is not enabled in the first Masked Loop Guided validation branch. "
                "Choose Native, PDD, FastH3 Dense or a compatible Turbo LoRA."
            )

        sampling = shotplan.get("sampling") if isinstance(shotplan.get("sampling"), dict) else {}
        seed = int(sampling.get("seed", seed))
        seed_stride = int(sampling.get("seed_stride", seed_stride))
        steps = int(sampling.get("steps", steps))
        sampler_name = str(sampling.get("sampler_name", sampler_name))
        scheduler = str(sampling.get("scheduler", scheduler))
        denoise = float(sampling.get("denoise", denoise))
        shift_video = float(sampling.get("shift_video", shift_video))
        shift_audio = float(sampling.get("shift_audio", shift_audio))
        actual_seed = (seed + int(chunk_index) * seed_stride) & 0xFFFFFFFFFFFFFFFF
        audio_lock_report = _audit_lipsync_audio_lock(shotplan, latent, 0)
        conditioning_cleanup = _release_conditioning_models(shotplan)

        turbo_model, turbo_report = _apply_turbo_lora(model, shotplan)
        fasth3_model, fasth3_report = _apply_fasth3_dense_lora(turbo_model, shotplan)
        pdd_model, pdd_report = _apply_pdd_lora(fasth3_model, shotplan)
        lora_model, secondary_report = _apply_secondary_lora(pdd_model, shotplan)
        turbo = _turbo_settings(shotplan)
        turbo_name = str(turbo.get("lora_name", "") or "").strip()
        turbo_enabled = bool(
            turbo.get("enabled", True)
            and str(turbo.get("mode", "off") or "off").lower() != "off"
            and turbo_name
            and folder_paths.get_full_path("loras", turbo_name)
        )
        if turbo_enabled:
            accelerated, acceleration_report = _accelerate(lora_model, shotplan)
            active_model = MiniMaxH3SigmaShift.execute(
                model=accelerated, shift_video=shift_video, shift_audio=shift_audio
            )[0]
        else:
            shifted = MiniMaxH3SigmaShift.execute(
                model=lora_model, shift_video=shift_video, shift_audio=shift_audio
            )[0]
            active_model, acceleration_report = _accelerate(shifted, shotplan)

        noise = RandomNoise.execute(noise_seed=actual_seed)[0]
        guider = BasicGuider.execute(model=active_model, conditioning=positive)[0]
        if turbo_enabled and str(turbo.get("sampler_mode", "audio_fixed")).lower() == "audio_fixed":
            sampler, sampler_report = _turbo_sampler(shotplan)
        else:
            sampler = KSamplerSelect.execute(sampler_name=sampler_name)[0]
            sampler_report = sampler_name
        sigmas = BasicScheduler.execute(
            model=active_model, scheduler=scheduler, steps=steps, denoise=denoise
        )[0]

        keyframes, keyframe_indices = _guide_batch(shotplan)
        LOG.info(
            "IAMCCS Masked Loop Guided start | total=%df | window=%df | overlap=%df | guides=%s | no outer joins",
            int(chunk.get("frame_count", 0)), int(contract["chunk_frames"]),
            int(contract["overlap_frames"]), keyframe_indices,
        )
        result = loop.MMH3LoopingSampler.execute(
            noise=noise,
            guider=guider,
            sampler=sampler,
            sigmas=sigmas,
            cond_set={"conds": [positive]},
            latent=latent,
            chunk_frames=int(contract["chunk_frames"]),
            overlap_frames=int(contract["overlap_frames"]),
            carry="mask",
            overlap_strength_video=float(contract.get("overlap_strength_video", 1.0)),
            overlap_strength_audio=float(contract.get("overlap_strength_audio", 1.0)),
            keyframes=keyframes,
            keyframe_indices=keyframe_indices,
            vae=video_vae,
        )
        sampled, chunks_rendered, loop_report = result[0], int(result[1]), str(result[2])

        cleanup_report = "disabled"
        if bool(shotplan.get("vram_clean_before_decode", True)):
            del noise, guider, sampler, sigmas, keyframes
            cleanup_report = _clean_vram_before_decode()
        native_frames = comfy_nodes.VAEDecode().decode(vae=video_vae, samples=sampled)[0]
        native_audio = VAEDecodeAudio.execute(vae=audio_vae, samples=sampled)[0]
        bridge_last_frame = native_frames[-1:].detach().clone()
        report = (
            f"IAMCCS Masked Loop Guided | one master latent | rendered_windows={chunks_rendered} | "
            f"frames={int(native_frames.shape[0])} | guides={keyframe_indices} | "
            f"window={int(contract['chunk_frames'])}f overlap={int(contract['overlap_frames'])}f | "
            f"sampler={sampler_report}+{scheduler} {steps} steps | "
            f"turbo={turbo_report} fasth3={fasth3_report} pdd={pdd_report} secondary={secondary_report} | "
            f"acceleration={acceleration_report} | audio_lock={audio_lock_report} | "
            f"pre_sample_cleanup={conditioning_cleanup} pre_decode_cleanup={cleanup_report} | "
            f"join=none/in-place\n{loop_report}"
        )
        return native_frames, native_audio, bridge_last_frame, sampled, H3_FPS, report


NODE_CLASS_MAPPINGS = {
    "IAMCCS_MiniMaxH3MaskedLoopGuidedSampler": IAMCCS_MiniMaxH3MaskedLoopGuidedSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_MiniMaxH3MaskedLoopGuidedSampler": "MiniMax H3 Masked Loop Guided · One Latent",
}
