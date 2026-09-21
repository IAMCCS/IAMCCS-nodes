# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later

"""Universal MiniMax H3 delivery backed by MMH3 Ultimate tiled sampling.

Unlike the R41 full-latent route, this delivery keeps only one temporal/spatial
piece in the H3 sampler at a time.  R41 remains available as the high-VRAM,
full-pass quality reference; this module is the bounded-memory default.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import folder_paths

from .iamccs_minimax_h3_atomic_backend import _resolve_shotplan
from .iamccs_minimax_h3_fast_latent_2pass import (
    IAMCCS_MiniMaxH3FastLatent2PassR41,
    _provider_stream_save,
    _result_item,
    _target_conditioning,
)
from .iamccs_minimax_h3_latent_upres_variant import (
    IAMCCS_MiniMaxH3LatentUpresSamplingR38,
    _delivery_size,
    _grid_up,
    _safe_render_id,
    _upres_settings,
)
from .iamccs_minimax_h3_pixel_refine_variant import (
    _finish_segment,
    _next_prompt,
    _rtx_finish_segment,
)
from .iamccs_minimax_h3_shotboard import (
    _concat_videos,
    _concat_videos_overlap,
    _current_prompt,
    _enqueue,
    _safe_name,
    _trim_audio_frames,
    _write_segment_metadata,
)


LOG = logging.getLogger("IAMCCS.MinimaxH3.UltimateTiled")
ULTIMATE_ROUTE = "h3_ultimate_tiled"


def _ultimate_class():
    import nodes

    klass = nodes.NODE_CLASS_MAPPINGS.get("MMH3UltimateUpscale")
    if klass is None:
        raise RuntimeError(
            "H3 ULTIMATE LATENT · TILED requires Comfyui-MMH3-UltimateUpscale. "
            "Install/enable that custom node and restart ComfyUI."
        )
    return klass


def _target_size(plan):
    delivery_w, delivery_h = _delivery_size(plan)
    upres = _upres_settings(plan)
    if bool(upres.get("rtx_requested", False)):
        return _grid_up((delivery_w + 1) // 2), _grid_up((delivery_h + 1) // 2)
    return _grid_up(delivery_w), _grid_up(delivery_h)


def _ultimate_params(plan, width, height):
    upres = _upres_settings(plan)
    model_name = str(upres.get("model_name", "") or "").strip()
    if not model_name or not folder_paths.get_full_path("latent_upscale_models", model_name):
        raise ValueError(
            "H3 ULTIMATE LATENT · TILED needs an installed MiniMax H3 3D latent "
            "upscaler checkpoint selected in IAMCCS Settings."
        )

    def grid(name, default, minimum=32):
        value = max(minimum, int(upres.get(name, default) or default))
        return max(minimum, round(value / 32) * 32)

    tile_w = min(width, grid("tile_width", 512))
    tile_h = min(height, grid("tile_height", 512))
    overlap_w = min(tile_w - 32, grid("overlap_width", 96))
    overlap_h = min(tile_h - 32, grid("overlap_height", 96))
    fade_w = min(overlap_w, grid("fade_width", 32))
    fade_h = min(overlap_h, grid("fade_height", 32))
    min_tile = min(tile_w, tile_h, grid("min_tile_size", 256))

    requested_chunk = max(17, int(upres.get("temporal_chunk", 102) or 102))
    chunk = max(17, round(requested_chunk / 17) * 17)
    requested_overlap = max(0, int(upres.get("temporal_overlap", 17) or 0))
    temporal_overlap = max(0, round(requested_overlap / 17) * 17)
    temporal_overlap = min(temporal_overlap, chunk - 17)

    latent = {
        "model_name": model_name,
        "width": int(width),
        "height": int(height),
        "device": str(upres.get("device", "cuda") or "cuda").lower(),
        "precision": str(upres.get("precision", "fp16") or "fp16").lower(),
    }
    temporal = {
        "chunk_length": chunk,
        "temporal_overlap": temporal_overlap,
        "anchor_strength": max(0.0, min(1.0, float(upres.get("anchor_strength", 0.999) or 0.999))),
    }
    spatial = {
        "tile_width": tile_w,
        "tile_height": tile_h,
        "spatial_w_overlap": overlap_w,
        "spatial_h_overlap": overlap_h,
        "fade_width": fade_w,
        "fade_height": fade_h,
        "min_tile_size": min_tile,
        "overlap_mode": str(upres.get("overlap_mode", "earlier") or "earlier"),
        "overlap_blend": str(upres.get("overlap_blend", "linear") or "linear"),
        "tile_size_mode": "specific_size",
        "masked_area_noise": 0.0,
        "brightness_match": False,
        "dynamic_fade": "off",
        "dynamic_fade_min": 32,
    }
    return latent, temporal, spatial


class IAMCCS_MiniMaxH3UltimateTiledDelivery(IAMCCS_MiniMaxH3FastLatent2PassR41):
    """Per-shot tiled H3 delivery with the native H3 audio as final authority."""

    OUTPUT_NODE = False
    CATEGORY = "IAMCCS/MiniMax H3/Universal Delivery"

    def finish(self, model, clip, video_vae, audio_vae, sampled_latent, stage1_conditioning,
               cine_linx, native_frames, native_audio, resolved_render_id, native_saved_report,
               current_segment, total_segments, context_trim_frames, join_trim_frames,
               queue_next_segment=True, motion_state=None, bridge_frame=None,
               first_frame_override=None, last_frame_override=None,
               ref_image_1=None, ref_image_2=None, ref_image_3=None, ref_image_4=None,
               ref_video=None, ref_video_audio=None, ref_audio=None,
               prompt=None, extra_pnginfo=None):
        import comfy.model_management as mm

        plan = _resolve_shotplan(cine_linx)
        if not bool(plan.get("upscale_enabled")) or str(plan.get("upscale_mode", "off")) != ULTIMATE_ROUTE:
            raise ValueError("Enable H3 ULTIMATE LATENT · TILED in IAMCCS H3 Settings.")
        if not native_saved_report:
            raise ValueError("H3 Ultimate Tiled must run after the native H3 checkpoint.")

        index, total = int(current_segment), max(1, int(total_segments))
        if index < 0 or index >= total:
            raise ValueError("H3 Ultimate Tiled received an invalid segment index.")
        run = _safe_render_id(resolved_render_id)
        root = Path(folder_paths.get_output_directory()) / "IAMCCS" / "MiniMaxH3" / "ULTIMATE_TILED" / run
        root.mkdir(parents=True, exist_ok=True)
        output = root / f"segment_{index + 1:04d}.mp4"
        if output.exists():
            raise FileExistsError(f"H3 Ultimate Tiled refuses to overwrite {output}.")

        visible = int(native_frames.shape[0])
        from .iamccs_minimax_h3_shotboard import _delivery_join_frames
        join = _delivery_join_frames(cine_linx, index, join_trim_frames)
        frames = visible - (1 if join == 1 else 0)
        audio = _trim_audio_frames(native_audio, 1, 24) if join == 1 else native_audio
        trim = max(0, int(context_trim_frames))
        if isinstance(motion_state, dict) and motion_state.get("active"):
            trim = max(trim, int(motion_state.get("trim_frames", 0) or 0))

        stage_w, stage_h = _target_size(plan)
        delivery_w, delivery_h = _delivery_size(plan)
        upres = _upres_settings(plan)
        latent_param, temporal_param, spatial_param = _ultimate_params(plan, stage_w, stage_h)
        final_rtx = bool(upres.get("rtx_requested", False))
        intermediate = None
        conditioning_report = sampling_report = ""
        try:
            mm.unload_all_models()
            mm.soft_empty_cache()
            target_model, conditioning, conditioning_report = _target_conditioning(
                model, clip, video_vae, audio_vae, cine_linx, index,
                stage_w, stage_h, stage1_conditioning,
                bridge_frame=bridge_frame, first_frame_override=first_frame_override,
                last_frame_override=last_frame_override, ref_image_1=ref_image_1,
                ref_image_2=ref_image_2, ref_image_3=ref_image_3, ref_image_4=ref_image_4,
                ref_video=ref_video, ref_video_audio=ref_video_audio, ref_audio=ref_audio,
                motion_state=motion_state, render_id=run,
            )
            active_model, noise, sampler, sigmas, sampling_report = (
                IAMCCS_MiniMaxH3LatentUpresSamplingR38().prepare(target_model, cine_linx, index)[:5]
            )
            refined = _result_item(_ultimate_class().execute(
                latent=sampled_latent,
                conditioning=conditioning,
                model=active_model,
                noise=noise,
                sampler=sampler,
                sigmas=sigmas,
                negative=None,
                cfg=1.0,
                latent_upscale_param=latent_param,
                temporal_split_param=temporal_param,
                spatial_split_param=spatial_param,
            ), 0)
            mm.unload_all_models()
            mm.soft_empty_cache()
            intermediate = _provider_stream_save(
                refined, video_vae, run, index,
                int(upres.get("pixel_groups", 1) or 1),
                route_folder="ULTIMATE_TILED",
            )
            del refined
            if final_rtx:
                _rtx_finish_segment(intermediate, output, audio, trim + (1 if join == 1 else 0), frames,
                                    stage_w, stage_h, delivery_w, delivery_h, 24,
                                    str(upres.get("rtx_quality", "ULTRA") or "ULTRA").upper())
            else:
                _finish_segment(intermediate, output, audio, trim + (1 if join == 1 else 0), frames,
                                delivery_w, delivery_h, 24)
        finally:
            mm.unload_all_models()
            gc.collect()
            mm.soft_empty_cache()

        source_workflow = extra_pnginfo.get("workflow") if isinstance(extra_pnginfo, dict) else None
        _write_segment_metadata(
            output, frames, 24, "h3_ultimate_tiled_native_audio",
            provenance={"render_id": run, "stage": "h3_ultimate_tiled", "segment_index": index,
                        "total_segments": total, "shotplan": plan},
            source_workflow=source_workflow,
        )
        preview = output
        if index == total - 1:
            paths = [root / f"segment_{part + 1:04d}.mp4" for part in range(total)]
            preview = root / "final_film.mp4"
            if preview.exists():
                raise FileExistsError(f"H3 Ultimate Tiled final film already exists: {preview}")
            if join > 1:
                _concat_videos_overlap(paths, preview, join, 24)
            else:
                _concat_videos(paths, preview)
            from .iamccs_minimax_h3_shotboard import _joined_frame_count
            _write_segment_metadata(
                preview, _joined_frame_count(paths, join if join > 1 else 0), 24,
                "crossfade" if join > 1 else "direct",
                provenance={"render_id": run, "stage": "h3_ultimate_tiled_master",
                            "total_segments": total, "segment_files": [path.name for path in paths],
                            "shotplan": plan},
                source_workflow=source_workflow,
            )

        queued = False
        if bool(queue_next_segment) and index + 1 < total:
            live, extra, outputs, sensitive = _current_prompt()
            next_prompt = _next_prompt(live if live is not None else prompt, run, index + 1)
            _enqueue(next_prompt, extra_data=extra, outputs=outputs, sensitive=sensitive)
            queued = True
        report = (
            f"H3 ULTIMATE LATENT · TILED saved {index + 1}/{total} | "
            f"native={plan.get('width')}x{plan.get('height')} -> tiled={stage_w}x{stage_h} "
            f"-> delivery={delivery_w}x{delivery_h} | temporal={temporal_param['chunk_length']}/"
            f"{temporal_param['temporal_overlap']} | tile={spatial_param['tile_width']}x"
            f"{spatial_param['tile_height']} overlap={spatial_param['spatial_w_overlap']}x"
            f"{spatial_param['spatial_h_overlap']} | final audio=native lock | next queued={queued} | {preview}"
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


NODE_CLASS_MAPPINGS = {
    "IAMCCS_MiniMaxH3UltimateTiledDelivery": IAMCCS_MiniMaxH3UltimateTiledDelivery,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_MiniMaxH3UltimateTiledDelivery": "IAMCCS H3 ULTIMATE LATENT · TILED",
}
