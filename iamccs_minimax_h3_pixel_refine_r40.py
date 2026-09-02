# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later

"""R40 streamed pixel/refine delivery; isolated from the working R38B/R39 path."""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import folder_paths
import torch

from .iamccs_minimax_h3_atomic_backend import H3_FPS, SUPERNODE_LINX_TYPE, _resolve_shotplan
from .iamccs_minimax_h3_latent_upres_variant import _delivery_size, _grid_up, _safe_render_id, _upres_settings
from .iamccs_minimax_h3_pixel_refine_variant import (
    _finish_segment,
    _next_prompt,
    _pixel_upscale,
    _provider,
    _rtx_finish_segment,
)
from .iamccs_minimax_h3_seed_scout_variant import (
    R40_SHOT_LAB,
    IAMCCS_MiniMaxH3LatentUpresSamplingR40,
)


LOG = logging.getLogger("IAMCCS.MiniMaxH3.PixelRefineR40")
CATEGORY = "IAMCCS/MiniMax H3/R40 Shot Lab"


def _refine_r40(model, conditioning, latent, cine_linx, segment_index, shot_lab):
    from comfy_extras.nodes_custom_sampler import BasicGuider

    loop = _provider("nodes_looping_sampler")
    if not loop.per_row_mask_is_continuous():
        raise RuntimeError("R40 requires the continuous H3 per-row mask contract in the current ComfyUI core")
    plan = _resolve_shotplan(cine_linx)
    settings = _upres_settings(plan)
    active_model, noise, sampler, sigmas, sampling_report = (
        IAMCCS_MiniMaxH3LatentUpresSamplingR40().prepare(
            model=model,
            cine_linx=cine_linx,
            segment_index=segment_index,
            shot_lab=shot_lab,
        )
    )
    guider = BasicGuider.execute(model=active_model, conditioning=conditioning)[0]
    common = _provider("common")
    original_video, original_audio = common.unpack_av(latent, "R40 Stage-2 source")
    total_frames = common.latents_to_frames(int(original_video.shape[2]))
    length, overlap, _, _, windows = _provider("nodes_windows")._plan(
        total_frames,
        int(settings.get("window_frames", 136)),
        int(settings.get("window_overlap", 22)),
        "standard_static",
    )
    LOG.info(
        "R40 Stage-2 windows | %d passes | %df window / %df overlap | %df total | %s",
        len(windows),
        common.latents_to_frames(length),
        common.frame_at_latent(overlap) if overlap else 0,
        total_frames,
        sampling_report,
    )
    result = loop.MMH3LoopingSampler.execute(
        noise=noise,
        guider=guider,
        sampler=sampler,
        sigmas=sigmas,
        cond_set={"conds": [conditioning]},
        latent=latent,
        chunk_frames=int(settings.get("window_frames", 136)),
        overlap_frames=int(settings.get("window_overlap", 22)),
        carry="mask",
        overlap_strength_video=1.0,
        overlap_strength_audio=1.0,
        audio_denoise_mask=torch.zeros((1, 1, 1)),
    )[0]
    video, audio = common.unpack_av(result, "R40 refined")
    if tuple(video.shape) != tuple(original_video.shape):
        raise RuntimeError("R40 changed the video latent extent; refusing an altered duration")
    if original_audio is not None and not torch.equal(original_audio.cpu(), audio.cpu()):
        raise RuntimeError("R40 Stage-2 changed the locked H3 audio latent; refusing desynchronised delivery")
    return result, sampling_report


class IAMCCS_MiniMaxH3PixelRefineR40:
    """R39-equivalent face/upscale delivery with R40 downstream-only controls."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "video_vae": ("VAE",),
                "sampled_latent": ("LATENT",),
                "conditioning": ("CONDITIONING",),
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "shot_lab": (R40_SHOT_LAB,),
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

    def finish(
        self,
        model,
        video_vae,
        sampled_latent,
        conditioning,
        cine_linx,
        shot_lab,
        native_frames,
        native_audio,
        resolved_render_id,
        native_saved_report,
        current_segment,
        total_segments,
        context_trim_frames,
        join_trim_frames,
        queue_next_segment=True,
        prompt=None,
        extra_pnginfo=None,
    ):
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
        enabled = bool(plan.get("upscale_enabled")) and plan.get("upscale_mode", "off") != "off"
        if enabled and plan["upscale_mode"] != "h3_pixel_refine":
            raise ValueError("R40 wires Native / RTX -> H3 only. Choose h3_pixel_refine or Off")
        if plan.get("rife_mode", "off") != "off":
            raise ValueError("R40 streamed H3 delivery preserves native 24 fps; RIFE is not wired in this variant")
        if plan.get("face_detailer_settings", {}).get("enabled", False) and not sampled_latent.get("iamccs_r38b_face_applied"):
            raise ValueError("Face Detailer is enabled but its R38B face branch is not connected before R40 Stage 2")
        if not native_saved_report:
            raise ValueError("R40 must run after the native checkpoint and Motion Context state commit")
        index, total = int(current_segment), int(total_segments)
        if index < 0 or index >= total:
            raise ValueError("R40 received an invalid segment index")
        run = _safe_render_id(resolved_render_id)
        root = Path(folder_paths.get_output_directory()) / "IAMCCS" / "MiniMaxH3" / "R40" / run
        root.mkdir(parents=True, exist_ok=True)
        output = root / f"segment_{index + 1:04d}.mp4"
        if output.exists():
            raise FileExistsError(f"R40 segment already exists: {output}. Start a new render rather than overwriting it")
        visible = int(native_frames.shape[0])
        join = max(0, int(join_trim_frames))
        frames = visible - (1 if join == 1 else 0)
        audio = _trim_audio_frames(native_audio, 1, H3_FPS) if join == 1 else native_audio
        stage2_report = "native/off"
        if enabled:
            settings = _upres_settings(plan)
            width, height = _delivery_size(plan)
            final_rtx = bool(settings.get("rtx_requested", False))
            crop_width, crop_height = (width // 2, height // 2) if final_rtx else (width, height)
            stage_width, stage_height = _grid_up(crop_width), _grid_up(crop_height)
            if width % 2 or height % 2:
                raise ValueError("R40 H.264 delivery width and height must be even")
            if int(settings.get("window_overlap", 22)) >= int(settings.get("window_frames", 136)):
                raise ValueError("R40 window overlap must be smaller than its window")
            common = _provider("common")
            video, _ = common.unpack_av(sampled_latent, "R40 Stage 1")
            source_frames = common.latents_to_frames(int(video.shape[2]))
            trim = int(context_trim_frames)
            if source_frames < trim + visible:
                raise ValueError(f"R40 latent has {source_frames} frames, needs {trim}+{visible}")
            try:
                mm.unload_all_models()
                mm.soft_empty_cache()
                upscaled = _pixel_upscale(sampled_latent, video_vae, stage_width, stage_height, settings)
                mm.unload_all_models()
                mm.soft_empty_cache()
                refined, stage2_report = _refine_r40(
                    model, conditioning, upscaled, cine_linx, index, shot_lab
                )
                del upscaled
                mm.unload_all_models()
                mm.soft_empty_cache()
                intermediate = _provider("nodes_save").MMH3StreamingSave.execute(
                    latent=refined,
                    vae=video_vae,
                    groups_per_chunk=int(settings.get("pixel_groups", 1)),
                    fps=float(H3_FPS),
                    filename_prefix=f"IAMCCS/MiniMaxH3/R40/{run}/stage2_{index + 1:04d}",
                    crf=16,
                    save_metadata=False,
                )[0]
                del refined
                if final_rtx:
                    mm.unload_all_models()
                    mm.soft_empty_cache()
                    _rtx_finish_segment(
                        intermediate,
                        output,
                        audio,
                        trim + (1 if join == 1 else 0),
                        frames,
                        crop_width,
                        crop_height,
                        width,
                        height,
                        H3_FPS,
                        settings.get("rtx_quality", "ULTRA"),
                    )
                else:
                    _finish_segment(
                        intermediate,
                        output,
                        audio,
                        trim + (1 if join == 1 else 0),
                        frames,
                        width,
                        height,
                        H3_FPS,
                    )
            finally:
                mm.unload_all_models()
                gc.collect()
                mm.soft_empty_cache()
        else:
            _encode_images(native_frames[1:] if join == 1 else native_frames, audio, H3_FPS, output)
        _write_segment_metadata(
            output,
            frames,
            H3_FPS,
            "trim_silent_tail" if native_audio.get("iamccs_flf_locked_audio_handles", False) else "crossfade",
        )
        preview = output
        if index == total - 1:
            paths = [root / f"segment_{part + 1:04d}.mp4" for part in range(total)]
            preview = root / "final_film.mp4"
            if preview.exists():
                raise FileExistsError(f"R40 final film already exists: {preview}")
            if join > 1:
                _concat_videos_overlap(paths, preview, join, H3_FPS)
            else:
                _concat_videos(paths, preview)
        queued = False
        if bool(queue_next_segment) and index + 1 < total:
            live, extra, outputs, sensitive = _current_prompt()
            next_prompt = _next_prompt(live if live is not None else prompt, run, index + 1)
            _enqueue(next_prompt, extra_data=extra, outputs=outputs, sensitive=sensitive)
            queued = True
        report = (
            f"R40 saved {index + 1}/{total} | {frames}f | native audio preserved | "
            f"next queued={queued} | Stage-2={stage2_report} | {preview}"
        )
        LOG.info(report)
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


NODE_CLASS_MAPPINGS = {"IAMCCS_MiniMaxH3PixelRefineR40": IAMCCS_MiniMaxH3PixelRefineR40}
NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_MiniMaxH3PixelRefineR40": "MiniMax H3 R40 · Cached Scout → Isolated Stage-2 Delivery"
}
