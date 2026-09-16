# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later

"""Experimental one-master-latent Guided AV Loop sampler for MiniMax H3.

This is deliberately isolated from the stable atomic generator.  It uses the
vendored MMH3Tools in-place loop: technical windows overlap inside one master
AV latent, so there is no decoded join, trim or crossfade between chunks.
"""

from __future__ import annotations
from .iamccs_h3_seed_policy import chunk_seed, WindowNoise

import logging

import folder_paths
import torch

from .iamccs_minimax_h3_atomic_backend import (
    H3_FPS,
    SUPERNODE_LINX_TYPE,
    IAMCCS_MiniMaxH3AtomicConditioningBackend,
    _accelerate,
    _apply_fasth3_dense_lora,
    _apply_pdd_lora,
    _apply_secondary_lora,
    _apply_turbo_lora,
    _audit_lipsync_audio_lock,
    _clean_vram_before_decode,
    _effective_shotplan,
    _fused_turbo_settings,
    _is_fused_turbo_preview,
    _load_image,
    _release_conditioning_models,
    _resize_reference_image,
    _resolve_shotplan,
    _run_h3_conditioning_with_cpu_fallback,
    _turbo_sampler,
    _turbo_settings,
)
from .iamccs_minimax_h3_pixel_refine_variant import _provider


LOG = logging.getLogger("IAMCCS.MiniMaxH3.GuidedAVLoop")
CATEGORY = "IAMCCS/MiniMax H3/Experimental Continuity"

_GUIDED_AV_LOOP_MODES = {
    "guided_av_loop_experimental",
    # Load-only compatibility for the original isolated experimental graph.
    "longvid_masked_loop_guided",
}


def _fit_guide_to_master_canvas(image, shotplan, *, is_opener):
    """Put one still on the exact master grid before building an IMAGE batch.

    ComfyUI IMAGE batches cannot contain mixed spatial sizes.  The looping
    sampler already applies this same MiniMax policy internally (stretch the
    frame-zero opener, centre-crop later anchors), but it can only do so after
    receiving a valid batch.  Normalising here therefore removes only the
    batching ambiguity; it does not introduce a second guide policy.
    """
    if not torch.is_tensor(image) or image.ndim != 4:
        return image, "none"
    import comfy.utils

    target_w = max(16, int(shotplan.get("width", image.shape[2]) or image.shape[2]))
    target_h = max(16, int(shotplan.get("height", image.shape[1]) or image.shape[1]))
    source_h, source_w = int(image.shape[1]), int(image.shape[2])
    rgb = image[..., :3]
    if (source_h, source_w) == (target_h, target_w):
        return rgb, f"{source_w}x{source_h}:master"
    crop = "disabled" if bool(is_opener) else "center"
    fitted = comfy.utils.common_upscale(
        rgb.movedim(-1, 1), target_w, target_h, "lanczos", crop
    ).movedim(1, -1)
    policy = "stretch-opener" if bool(is_opener) else "centre-crop-anchor"
    return fitted, f"{source_w}x{source_h}->{target_w}x{target_h}:{policy}"


def _guide_batch(shotplan):
    contract = shotplan.get("masked_loop_guided")
    if not isinstance(contract, dict) or not bool(contract.get("enabled")):
        raise ValueError(
            "Guided AV Loop requires task_mode=guided_av_loop_experimental in IAMCCS H3 Settings."
        )
    guides = contract.get("keyframes") if isinstance(contract.get("keyframes"), list) else []
    indices = contract.get("keyframe_indices") if isinstance(contract.get("keyframe_indices"), list) else []
    if len(guides) < 2 or len(guides) != len(indices):
        raise ValueError("Guided AV Loop needs at least two matched Shotboard images and frame indices.")
    images = []
    resize_notes = []
    for ordinal, guide in enumerate(guides, start=1):
        path = str(guide.get("source_path", "") or "").strip()
        image = _load_image(path)
        if not torch.is_tensor(image):
            raise ValueError(f"Guided AV Loop image {ordinal} was not found: {path or 'empty path'}")
        image, pre_note = _resize_reference_image(image[:1], shotplan, f"masked_loop_guide_{ordinal}")
        image, fit_note = _fit_guide_to_master_canvas(
            image[:1], shotplan, is_opener=(int(indices[ordinal - 1]) == 0)
        )
        images.append(image[:1].detach().to(device="cpu"))
        resize_notes.append(f"g{ordinal}[{pre_note}; {fit_note}]")
    shapes = {(int(item.shape[1]), int(item.shape[2]), int(item.shape[3])) for item in images}
    if len(shapes) != 1:
        raise RuntimeError(
            "Guided AV Loop could not normalize all keyframes to one master canvas: "
            + ", ".join(str(shape) for shape in sorted(shapes))
        )
    LOG.info("IAMCCS Guided AV Loop keyframe canvas | %s", " | ".join(resize_notes))
    return torch.cat(images, dim=0), ",".join(str(int(value)) for value in indices)


_WINDOW_COND_KEY = "iamccs_masked_loop_window_conditions"


def _master_chunk(shotplan):
    chunks = shotplan.get("chunks") if isinstance(shotplan, dict) else None
    if not isinstance(chunks, list) or len(chunks) != 1 or not isinstance(chunks[0], dict):
        raise ValueError("Guided AV Loop requires exactly one validated master chunk.")
    return chunks[0]


def _without_window_condition_envelope(conditioning):
    """Return ordinary CONDITIONING plus an optional window-condition list."""
    if not isinstance(conditioning, (list, tuple)):
        return conditioning, None
    cleaned = []
    window_conditions = None
    for item in conditioning:
        if not isinstance(item, (list, tuple)) or len(item) < 2 or not isinstance(item[1], dict):
            cleaned.append(item)
            continue
        meta = dict(item[1])
        carried = meta.pop(_WINDOW_COND_KEY, None)
        if window_conditions is None and isinstance(carried, list) and carried:
            window_conditions = carried
        cleaned.append([item[0], meta])
    return cleaned, window_conditions


def _with_window_condition_envelope(conditioning, window_conditions):
    if not isinstance(conditioning, (list, tuple)) or not conditioning:
        raise RuntimeError("Guided AV Loop conditioner received an invalid CONDITIONING object.")
    wrapped = []
    for index, item in enumerate(conditioning):
        if not isinstance(item, (list, tuple)) or len(item) < 2 or not isinstance(item[1], dict):
            wrapped.append(item)
            continue
        meta = dict(item[1])
        if index == 0:
            meta[_WINDOW_COND_KEY] = window_conditions
        wrapped.append([item[0], meta])
    return wrapped


def _window_prompt_plan(shotplan, loop=None):
    """Build the prompt owned by each exact MMH3 technical window."""
    contract = shotplan.get("masked_loop_guided")
    if not isinstance(contract, dict) or not bool(contract.get("enabled")):
        return []
    loop = loop or _provider("nodes_looping_sampler")
    chunk = _master_chunk(shotplan)
    total_frames = int(chunk.get("frame_count", 0))
    chunk_frames = int(contract.get("chunk_frames", 0))
    overlap_frames = int(contract.get("overlap_frames", 22))
    _length, overlap_latents, planned_frames, _planned_t, windows = loop._plan(
        total_frames, chunk_frames, overlap_frames, "standard_static"
    )
    spans = loop._window_frame_spans(windows, planned_frames)
    ownership_overlap = loop.frame_at_latent(overlap_latents)
    guides = contract.get("keyframes") if isinstance(contract.get("keyframes"), list) else []
    guides = sorted(
        (dict(item) for item in guides if isinstance(item, dict)),
        key=lambda item: (int(item.get("global_frame", 0)), str(item.get("id", ""))),
    )
    structural = str(contract.get("structural_prompt", "") or "").strip()
    global_prompt = str(contract.get("global_prompt", shotplan.get("global_prompt", "")) or "").strip()
    result = []
    for window_index, (start_frame, end_frame) in enumerate(spans):
        owned = []
        for guide in guides:
            owner = loop._owner(spans, ownership_overlap, int(guide.get("global_frame", 0)))
            if owner is not None and int(owner[0]) == window_index:
                owned.append(guide)
        previous = None
        for guide in guides:
            if int(guide.get("global_frame", 0)) <= int(start_frame):
                previous = guide
            else:
                break
        selected = []
        if previous is not None and previous not in owned:
            selected.append(("continuity state", previous))
        selected.extend(("positioned guide", guide) for guide in owned)
        local_lines = []
        for role, guide in selected:
            text = str(guide.get("prompt", "") or "").strip()
            if text:
                local_lines.append(
                    f"{role} at {int(guide.get('global_frame', 0)) / H3_FPS:.2f}s: {text}"
                )
        scope = (
            f"Technical window {window_index + 1}: frames {int(start_frame)}-{int(end_frame)}. "
            "Follow only the local directions assigned to this window; do not anticipate later framing."
        )
        prompt = "\n\n".join(
            part for part in (structural, global_prompt, scope, "\n".join(local_lines)) if part
        ).strip()
        result.append({
            "index": window_index,
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "guide_ids": [str(guide.get("id", "")) for _role, guide in selected],
            "prompt": prompt,
        })
    return result


class IAMCCS_MiniMaxH3MaskedLoopConditioning(IAMCCS_MiniMaxH3AtomicConditioningBackend):
    """Atomic H3 conditioning plus one prompt payload per masked-loop window."""

    CATEGORY = CATEGORY

    def prepare(self, *args, **kwargs):
        output = list(super().prepare(*args, **kwargs))
        cine_linx = kwargs.get("cine_linx")
        clip = kwargs.get("clip")
        if cine_linx is None or clip is None:
            raise ValueError("Guided AV Loop conditioning requires clip and cine_linx inputs.")
        shotplan = _effective_shotplan(cine_linx, _resolve_shotplan(cine_linx))
        if str(shotplan.get("task_mode", "")).strip().lower() not in _GUIDED_AV_LOOP_MODES:
            raise ValueError("This isolated conditioner only accepts GUIDED AV LOOP · EXPERIMENTAL mode.")
        prompt_plan = _window_prompt_plan(shotplan)
        if not prompt_plan:
            raise RuntimeError("Guided AV Loop produced no technical-window prompt plan.")

        window_conditions = []
        placement_reports = []
        for item in prompt_plan:
            conditioning, placement = _run_h3_conditioning_with_cpu_fallback(
                clip,
                shotplan,
                lambda active_clip, text=item["prompt"]: active_clip.encode_from_tokens_scheduled(
                    active_clip.tokenize(text, images=[])
                ),
            )
            window_conditions.append(conditioning)
            placement_reports.append(placement)
        output[1] = _with_window_condition_envelope(output[1], window_conditions)
        output[10] = (
            str(output[10])
            + " | masked_loop_window_prompts="
            + "; ".join(
                f"w{item['index'] + 1}:{item['start_frame']}-{item['end_frame']}f guides={','.join(item['guide_ids']) or 'none'}"
                for item in prompt_plan
            )
            + f" | window_text_encoder={','.join(placement_reports)}"
        )
        LOG.info(
            "IAMCCS Guided AV Loop conditioning | %s",
            " | ".join(
                f"w{item['index'] + 1}={item['start_frame']}-{item['end_frame']}f:{','.join(item['guide_ids']) or 'no-local-guide'}"
                for item in prompt_plan
            ),
        )
        return tuple(output)


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
        if str(shotplan.get("task_mode", "")).strip().lower() not in _GUIDED_AV_LOOP_MODES:
            raise ValueError(
                "This isolated sampler only accepts GUIDED AV LOOP · EXPERIMENTAL; stable IAMCCS modes keep using their existing backends."
            )
        if int(chunk_index) != 0 or len(shotplan.get("chunks", [])) != 1:
            raise ValueError("Guided AV Loop is one master-latent execution and must receive chunk_index 0.")
        chunk = _master_chunk(shotplan)
        if str(shotplan.get("acceleration", "native")).startswith("iamccs_progressive_"):
            raise ValueError("Guided AV Loop uses masked windows and cannot run progressive spatial sampling. Choose Native, PDD, FastH3, SLA or a compatible Turbo LoRA.")
        contract = shotplan["masked_loop_guided"]
        loop = _provider("nodes_looping_sampler")
        if not loop.per_row_mask_is_continuous():
            raise RuntimeError(
                "Guided AV Loop requires ComfyUI's continuous per-row MiniMax H3 mask support (#15375)."
            )
        if _is_fused_turbo_preview(shotplan):
            raise ValueError(
                "Fused Fast H3 preview is not enabled in the Guided AV Loop experimental branch. "
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
        actual_seed = chunk_seed(sampling, chunk_index, seed, seed_stride)
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

        positive, window_conditions = _without_window_condition_envelope(positive)
        prompt_plan = _window_prompt_plan(shotplan, loop=loop)
        expected_windows = len(prompt_plan)
        if expected_windows > 1 and (
            not isinstance(window_conditions, list) or len(window_conditions) != expected_windows
        ):
            raise RuntimeError(
                "Guided AV Loop has multiple technical windows but received one shared prompt. "
                "Use IAMCCS MiniMax H3 Masked Loop Conditioning before this sampler; a shared prompt "
                "would leak later framing directions into earlier windows."
            )
        active_conditions = window_conditions if window_conditions else [positive]
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
            "IAMCCS Guided AV Loop start | one_master_AV=%df | internal_window=%df | overlap=%df | guides=%s | no outer chunks/joins",
            int(chunk.get("frame_count", 0)), int(contract["chunk_frames"]),
            int(contract["overlap_frames"]), keyframe_indices,
        )
        result = loop.MMH3LoopingSampler.execute(
            noise=WindowNoise(noise, sampling),
            guider=guider,
            sampler=sampler,
            sigmas=sigmas,
            cond_set={"conds": active_conditions},
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
            f"IAMCCS Guided AV Loop · EXPERIMENTAL | one persistent full AV master latent | rendered_internal_windows={chunks_rendered} | "
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
    "IAMCCS_MiniMaxH3MaskedLoopConditioning": IAMCCS_MiniMaxH3MaskedLoopConditioning,
    "IAMCCS_MiniMaxH3MaskedLoopGuidedSampler": IAMCCS_MiniMaxH3MaskedLoopGuidedSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_MiniMaxH3MaskedLoopConditioning": "GUIDED AV LOOP · Window Conditioning",
    "IAMCCS_MiniMaxH3MaskedLoopGuidedSampler": "GUIDED AV LOOP · EXPERIMENTAL · One Master AV Latent",
}
