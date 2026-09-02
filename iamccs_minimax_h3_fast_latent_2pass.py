# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fast direct-latent MiniMax H3 two-pass delivery.

This is an isolated delivery variant.  It does not replace the native R39/R40
generation path or the conservative R38B pixel/windowed route.
"""

from __future__ import annotations

import copy
import gc
import logging
from pathlib import Path
from typing import Any

import folder_paths
import torch

from .iamccs_minimax_h3_atomic_backend import (
    SUPERNODE_LINX_TYPE,
    IAMCCS_MiniMaxH3AtomicConditioningBackend,
    _resolve_shotplan,
)
from .iamccs_minimax_h3_latent_upres_variant import (
    _delivery_size,
    _grid_up,
    _replace_plan,
    _safe_render_id,
    _upres_settings,
    IAMCCS_MiniMaxH3LatentUpresSamplingR38,
)
from .iamccs_minimax_h3_pixel_refine_variant import (
    _finish_segment,
    _next_prompt,
    _rtx_finish_segment,
)


LOG = logging.getLogger("IAMCCS.MiniMaxH3.FastLatent2Pass")
CATEGORY = "IAMCCS/MiniMax H3/Quality Latent 2-Pass"
FAST_ROUTE = "h3_fast_latent_2pass"


def _result_item(value: Any, index: int = 0):
    """Read both legacy tuples and Comfy's current NodeOutput objects."""
    try:
        return value[index]
    except (IndexError, KeyError, TypeError):
        result = getattr(value, "result", None)
        if result is None:
            raise
        return result[index]


def _fast_stage_size(plan: dict[str, Any]) -> tuple[int, int]:
    """R41 reuses R38 tile width/height as the direct Stage-2 canvas.

    Those fields are not used for tiling on this route.  Reusing them avoids a
    positional widget/schema change in the proven Shotboard and Settings nodes.
    """
    upres = _upres_settings(plan)
    native_w = max(256, int(plan.get("width", 960) or 960))
    native_h = max(256, int(plan.get("height", 544) or 544))
    stage_w = _grid_up(int(upres.get("tile_width", 1504) or 1504))
    stage_h = _grid_up(int(upres.get("tile_height", 832) or 832))
    if stage_w < native_w or stage_h < native_h:
        raise ValueError(
            f"FAST LATENT 2-PASS Stage 2 must not downscale: native={native_w}x{native_h}, "
            f"stage2={stage_w}x{stage_h}. Apply a Fast Latent preset or increase the Stage-2 canvas."
        )
    return stage_w, stage_h


def _load_upscaler_class():
    import nodes

    klass = nodes.NODE_CLASS_MAPPINGS.get("MinimaxH3LatentUpscaler3D")
    if klass is None:
        raise RuntimeError(
            "FAST LATENT 2-PASS requires Comfyui_Minimax_h3_latent_Upscaler and its "
            "MinimaxH3LatentUpscaler3D node."
        )
    return klass


def _split_av(av_latent):
    from comfy_extras.nodes_lt import LTXVSeparateAVLatent

    result = LTXVSeparateAVLatent.execute(av_latent=av_latent)
    return _result_item(result, 0), _result_item(result, 1)


def _concat_av(video_latent, audio_latent):
    from comfy_extras.nodes_lt import LTXVConcatAVLatent

    result = LTXVConcatAVLatent.execute(video_latent=video_latent, audio_latent=audio_latent)
    return _result_item(result, 0)


def _upscale_video_latent(video_latent, plan, width, height):
    upres = _upres_settings(plan)
    model_name = str(upres.get("model_name", "") or "").strip()
    if not model_name or not folder_paths.get_full_path("latent_upscale_models", model_name):
        raise ValueError(
            "FAST LATENT 2-PASS needs an installed minimax_h3_latent_upscaler_3d checkpoint "
            "selected in IAMCCS H3 Settings."
        )
    precision = str(upres.get("precision", "bf16") or "bf16").lower()
    device = str(upres.get("device", "cuda") or "cuda").lower()
    klass = _load_upscaler_class()
    dynamic_mode = {"mode": "target dimensions", "width": int(width), "height": int(height)}
    LOG.info(
        "FAST LATENT 2-PASS learned upres start | model=%s | target=%dx%d | %s/%s",
        model_name, width, height, device, precision,
    )
    result = klass.execute(
        latent=video_latent,
        model_name=model_name,
        mode=dynamic_mode,
        align=32,
        device=device,
        precision=precision,
    )
    return _result_item(result, 0)


def _locked_av_latent(upscaled_video, original_audio):
    video = dict(upscaled_video)
    audio = dict(original_audio)
    video["noise_mask"] = torch.ones_like(video["samples"])
    # Stage 2 is a visual refinement.  The native H3 audio latent remains the
    # exact phonetic/timing authority and is never re-denoised.
    audio["noise_mask"] = torch.zeros_like(audio["samples"])
    return _concat_av(video, audio)


def _target_conditioning(model, clip, video_vae, audio_vae, cine_linx, segment_index,
                         stage_width, stage_height, stage1_conditioning,
                         bridge_frame=None, first_frame_override=None, last_frame_override=None,
                         ref_image_1=None, ref_image_2=None, ref_image_3=None, ref_image_4=None,
                         ref_video=None, ref_video_audio=None, ref_audio=None,
                         motion_state=None, render_id=""):
    plan = _resolve_shotplan(cine_linx)
    # Conditioning tensors that contain image/guide latents are spatially
    # authored.  Never reuse the native-resolution Stage-1 object on the
    # larger Stage-2 canvas.  Re-materialize the same named Shotboard mode at
    # the target grid; the upscaled sampled latent already carries the native
    # motion-context history, while denoise controls how much Stage 2 may move.
    target_plan = dict(plan)
    target_plan["width"] = int(stage_width)
    target_plan["height"] = int(stage_height)
    target_plan["upscale_enabled"] = False
    target_plan["upscale_mode"] = "off"
    target_linx = _replace_plan(cine_linx, target_plan)
    result = IAMCCS_MiniMaxH3AtomicConditioningBackend().prepare(
        model=model,
        clip=clip,
        video_vae=video_vae,
        audio_vae=audio_vae,
        cine_linx=target_linx,
        segment_index=int(segment_index),
        bridge_frame=bridge_frame,
        render_id=render_id,
        first_frame_override=first_frame_override,
        last_frame_override=last_frame_override,
        ref_image_1=ref_image_1,
        ref_image_2=ref_image_2,
        ref_image_3=ref_image_3,
        ref_image_4=ref_image_4,
        ref_video=ref_video,
        ref_video_audio=ref_video_audio,
        ref_audio=ref_audio,
    )
    return result[0], result[1], f"mode-matched target conditioning {stage_width}x{stage_height}; Stage-1 context retained in latent"


def _sample_stage2(model, conditioning, latent, cine_linx, segment_index):
    from comfy_extras.nodes_custom_sampler import BasicGuider, SamplerCustomAdvanced

    active_model, noise, sampler, sigmas, sampling_report = (
        IAMCCS_MiniMaxH3LatentUpresSamplingR38().prepare(model, cine_linx, int(segment_index))[:5]
    )
    guider = _result_item(BasicGuider.execute(model=active_model, conditioning=conditioning), 0)
    sampled = SamplerCustomAdvanced.execute(
        noise=noise,
        guider=guider,
        sampler=sampler,
        sigmas=sigmas,
        latent_image=latent,
    )
    return _result_item(sampled, 1), sampling_report


class IAMCCS_MiniMaxH3FastLatent2PassR41:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "video_vae": ("VAE",),
                "audio_vae": ("VAE",),
                "sampled_latent": ("LATENT",),
                "stage1_conditioning": ("CONDITIONING",),
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "native_frames": ("IMAGE",),
                "native_audio": ("AUDIO",),
                "resolved_render_id": ("STRING", {"forceInput": True}),
                "native_saved_report": ("STRING", {"forceInput": True}),
                "current_segment": ("INT", {"forceInput": True}),
                "total_segments": ("INT", {"forceInput": True}),
                "context_trim_frames": ("INT", {"forceInput": True}),
                "join_trim_frames": ("INT", {"forceInput": True}),
                "queue_next_segment": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "motion_state": ("IAMCCS_H3_MOTION_CONTEXT",),
                "bridge_frame": ("IMAGE",),
                "first_frame_override": ("IMAGE",),
                "last_frame_override": ("IMAGE",),
                "ref_image_1": ("IMAGE",),
                "ref_image_2": ("IMAGE",),
                "ref_image_3": ("IMAGE",),
                "ref_image_4": ("IMAGE",),
                "ref_video": ("IMAGE",),
                "ref_video_audio": ("AUDIO",),
                "ref_audio": ("AUDIO",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "report")
    FUNCTION = "finish"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def finish(self, model, clip, video_vae, audio_vae, sampled_latent, stage1_conditioning,
               cine_linx, native_frames, native_audio, resolved_render_id, native_saved_report,
               current_segment, total_segments, context_trim_frames, join_trim_frames,
               queue_next_segment=True, motion_state=None, bridge_frame=None,
               first_frame_override=None, last_frame_override=None,
               ref_image_1=None, ref_image_2=None, ref_image_3=None, ref_image_4=None,
               ref_video=None, ref_video_audio=None, ref_audio=None,
               prompt=None, extra_pnginfo=None):
        import comfy.model_management as mm
        from .iamccs_minimax_h3_shotboard import (
            _concat_videos,
            _concat_videos_overlap,
            _current_prompt,
            _encode_images,
            _enqueue,
            _trim_audio_frames,
            _write_segment_metadata,
        )

        plan = _resolve_shotplan(cine_linx)
        enabled = bool(plan.get("upscale_enabled")) and str(plan.get("upscale_mode", "off")) == FAST_ROUTE
        if not enabled:
            raise ValueError("This workflow is FAST LATENT 2-PASS. Enable that exact upscale route in IAMCCS H3 Settings.")
        if not native_saved_report:
            raise ValueError("FAST LATENT 2-PASS must run after the native H3 checkpoint.")

        index, total = int(current_segment), max(1, int(total_segments))
        if index < 0 or index >= total:
            raise ValueError("FAST LATENT 2-PASS received an invalid segment index.")
        run = _safe_render_id(resolved_render_id)
        root = Path(folder_paths.get_output_directory()) / "IAMCCS" / "MiniMaxH3" / "FAST_LATENT_2PASS" / run
        root.mkdir(parents=True, exist_ok=True)
        output = root / f"segment_{index + 1:04d}.mp4"
        if output.exists():
            raise FileExistsError(f"FAST LATENT 2-PASS refuses to overwrite {output}.")

        visible = int(native_frames.shape[0])
        join = max(0, int(join_trim_frames))
        frames = visible - (1 if join == 1 else 0)
        audio = _trim_audio_frames(native_audio, 1, 24) if join == 1 else native_audio
        trim = max(0, int(context_trim_frames))
        if isinstance(motion_state, dict) and motion_state.get("active"):
            trim = max(trim, int(motion_state.get("trim_frames", 0) or 0))

        stage_width, stage_height = _fast_stage_size(plan)
        delivery_width, delivery_height = _delivery_size(plan)
        upres = _upres_settings(plan)
        final_rtx = bool(upres.get("rtx_requested", False))
        if not final_rtx and (stage_width, stage_height) != (delivery_width, delivery_height):
            raise ValueError(
                "FAST LATENT 2-PASS without RTX requires Stage-2 and Delivery canvases to match. "
                f"Got {stage_width}x{stage_height} -> {delivery_width}x{delivery_height}."
            )

        LOG.info(
            "FAST LATENT 2-PASS start | segment=%d/%d | native=%dx%d | stage2=%dx%d | delivery=%dx%d | audio=locked",
            index + 1, total, int(plan.get("width", 0)), int(plan.get("height", 0)),
            stage_width, stage_height, delivery_width, delivery_height,
        )

        intermediate = None
        try:
            mm.unload_all_models()
            mm.soft_empty_cache()
            stage1_video, stage1_audio = _split_av(sampled_latent)
            upscaled_video = _upscale_video_latent(stage1_video, plan, stage_width, stage_height)
            del stage1_video
            locked_latent = _locked_av_latent(upscaled_video, stage1_audio)
            del upscaled_video
            mm.unload_all_models()
            mm.soft_empty_cache()

            target_model, target_conditioning, conditioning_report = _target_conditioning(
                model, clip, video_vae, audio_vae, cine_linx, index,
                stage_width, stage_height, stage1_conditioning,
                bridge_frame=bridge_frame,
                first_frame_override=first_frame_override,
                last_frame_override=last_frame_override,
                ref_image_1=ref_image_1, ref_image_2=ref_image_2,
                ref_image_3=ref_image_3, ref_image_4=ref_image_4,
                ref_video=ref_video, ref_video_audio=ref_video_audio, ref_audio=ref_audio,
                motion_state=motion_state, render_id=run,
            )
            mm.unload_all_models()
            mm.soft_empty_cache()
            refined, sampling_report = _sample_stage2(
                target_model, target_conditioning, locked_latent, cine_linx, index,
            )
            del locked_latent
            mm.unload_all_models()
            mm.soft_empty_cache()

            intermediate = _provider_stream_save(
                refined, video_vae, run, index, int(upres.get("pixel_groups", 1) or 1),
            )
            del refined
            if final_rtx:
                mm.unload_all_models()
                mm.soft_empty_cache()
                _rtx_finish_segment(
                    intermediate, output, audio, trim + (1 if join == 1 else 0), frames,
                    stage_width, stage_height, delivery_width, delivery_height, 24,
                    str(upres.get("rtx_quality", "ULTRA") or "ULTRA").upper(),
                )
            else:
                _finish_segment(
                    intermediate, output, audio, trim + (1 if join == 1 else 0), frames,
                    delivery_width, delivery_height, 24,
                )
        finally:
            mm.unload_all_models()
            gc.collect()
            mm.soft_empty_cache()

        _write_segment_metadata(output, frames, 24, "fast_latent_2pass_audio_locked")
        preview = output
        if index == total - 1:
            paths = [root / f"segment_{part + 1:04d}.mp4" for part in range(total)]
            preview = root / "final_film.mp4"
            if preview.exists():
                raise FileExistsError(f"FAST LATENT 2-PASS final film already exists: {preview}")
            if join > 1:
                _concat_videos_overlap(paths, preview, join, 24)
            else:
                _concat_videos(paths, preview)

        queued = False
        if bool(queue_next_segment) and index + 1 < total:
            live, extra, outputs, sensitive = _current_prompt()
            next_prompt = _next_prompt(live if live is not None else prompt, run, index + 1)
            _enqueue(next_prompt, extra_data=extra, outputs=outputs, sensitive=sensitive)
            queued = True
        report = (
            f"FAST LATENT 2-PASS saved {index + 1}/{total} | native={plan.get('width')}x{plan.get('height')} "
            f"-> stage2={stage_width}x{stage_height} -> delivery={delivery_width}x{delivery_height} | "
            f"audio=exact native lock | next queued={queued} | {preview}"
        )
        LOG.info("%s | %s | %s", report, conditioning_report, sampling_report)
        return {
            "ui": {
                "text": [report],
                "images": [{
                    "filename": preview.name,
                    "subfolder": preview.parent.relative_to(Path(folder_paths.get_output_directory())).as_posix(),
                    "type": "output",
                }],
                "animated": (True,),
            },
            "result": (str(preview), report),
        }


def _provider_stream_save(latent, video_vae, run, index, groups):
    # Reuse the audited streamed VAE/ffmpeg implementation already vendored by
    # the conservative R38B route; this avoids a full high-resolution IMAGE batch.
    from .iamccs_minimax_h3_pixel_refine_variant import _provider

    return _provider("nodes_save").MMH3StreamingSave.execute(
        latent=latent,
        vae=video_vae,
        groups_per_chunk=max(1, int(groups)),
        fps=24.0,
        filename_prefix=f"IAMCCS/MiniMaxH3/FAST_LATENT_2PASS/{run}/stage2_{index + 1:04d}",
        crf=16,
        save_metadata=False,
    )[0]


NODE_CLASS_MAPPINGS = {
    "IAMCCS_MiniMaxH3FastLatent2PassR41": IAMCCS_MiniMaxH3FastLatent2PassR41,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_MiniMaxH3FastLatent2PassR41": "MiniMax H3 · QUALITY LATENT 2-PASS · Streamed Delivery",
}
