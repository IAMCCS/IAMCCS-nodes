# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later

"""Lazy delivery routing shared by the isolated R42 universal workflows.

R39/R40/R41 remain untouched.  These adapters make their proven delivery
implementations composable in one graph: only the path selected by the H3
Shotplan is requested by ComfyUI's lazy-input protocol.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import logging
import subprocess
import tempfile
from typing import Any

import folder_paths
import torch

LOG = logging.getLogger("IAMCCS.MinimaxH3.UniversalDelivery")

from .iamccs_minimax_h3_atomic_backend import SUPERNODE_LINX_TYPE, _resolve_shotplan
from .iamccs_minimax_h3_fast_latent_2pass import (
    FAST_ROUTE,
    IAMCCS_MiniMaxH3FastLatent2PassR41,
)
from .iamccs_minimax_h3_ultimate_tiled import (
    ULTIMATE_ROUTE,
    IAMCCS_MiniMaxH3UltimateTiledDelivery,
)
from .iamccs_minimax_h3_pixel_refine_variant import IAMCCS_MiniMaxH3PixelRefineR38B
from .iamccs_minimax_h3_pixel_refine_variant import (
    _next_prompt,
)
from .iamccs_minimax_h3_shotboard import (
    IAMCCS_MiniMaxH3NativeCheckpointSave,
    _concat_videos,
    _concat_videos_overlap,
    _current_prompt,
    _encode_images,
    _enqueue,
    _find_ffmpeg,
    _safe_name,
    _trim_audio_frames,
    _write_wav,
    _write_segment_metadata,
)


CATEGORY = "IAMCCS/MiniMax H3/Universal Delivery"
MASTER_ROUTES = {"ltx23"}
LTX_PER_CHUNK_ROUTE = "ltx23_per_chunk"
RTX_FINAL_ROUTE = "rtx_final"
PIXEL_TILED_ROUTE = "pixel_tiled_low_vram"
KNOWN_ROUTES = {"off", RTX_FINAL_ROUTE, PIXEL_TILED_ROUTE, "h3_pixel_refine", FAST_ROUTE, ULTIMATE_ROUTE, LTX_PER_CHUNK_ROUTE, *MASTER_ROUTES}


class IAMCCS_MiniMaxH3ExactLTXDeliverySizeR42:
    """Remove the protected LTX canvas and honour the authored delivery boxes.

    The LTX latent upsampler works on a legal processing canvas which can be
    larger than the requested output (for example 2208x1152 for a 2048x1080
    delivery).  The old R42 group exposed only that processing size, so its
    internal node labelled ``EXACT DELIVERY SIZE`` could not know the actual
    authored target.  This external adapter reads the target from Shotboard
    truth after LTX decode.  UHD requests stay untouched here because the
    following RTX node owns their final exact-size pass.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "cine_linx": (SUPERNODE_LINX_TYPE,),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "resize"
    CATEGORY = CATEGORY

    def resize(self, images, cine_linx):
        if not torch.is_tensor(images) or images.ndim != 4 or int(images.shape[0]) < 1:
            raise ValueError("R42 exact LTX delivery expects an IMAGE frame batch")
        plan = _resolve_shotplan(cine_linx)
        settings = plan.get("upscale_settings") if isinstance(plan.get("upscale_settings"), dict) else {}
        target_width = max(256, int(settings.get("target_width", plan.get("upscale_width", images.shape[2])) or images.shape[2]))
        target_height = max(256, int(settings.get("target_height", plan.get("upscale_height", images.shape[1])) or images.shape[1]))
        ltx_4k = bool(settings.get("ltx_4k_enabled", False)) and _route(cine_linx) in MASTER_ROUTES
        source_height, source_width = int(images.shape[1]), int(images.shape[2])
        if ltx_4k:
            return images, (
                f"R42 exact LTX delivery deferred to RTX 4K | "
                f"stage={source_width}x{source_height} -> target={target_width}x{target_height}"
            )
        if (source_width, source_height) == (target_width, target_height):
            return images, f"R42 exact LTX delivery already {target_width}x{target_height}"

        target_aspect = float(target_width) / float(target_height)
        source_aspect = float(source_width) / float(source_height)
        crop_width, crop_height = source_width, source_height
        if source_aspect > target_aspect:
            crop_width = max(2, min(source_width, round(source_height * target_aspect)))
        elif source_aspect < target_aspect:
            crop_height = max(2, min(source_height, round(source_width / target_aspect)))
        crop_left = max(0, (source_width - crop_width) // 2)
        crop_top = max(0, (source_height - crop_height) // 2)
        cropped = images[:, crop_top:crop_top + crop_height, crop_left:crop_left + crop_width, :]

        # Keep the resize bounded on long clips.  LTX delivery is already a
        # full IMAGE batch, but interpolation no longer adds another full
        # float32 copy of the programme to the peak working set.
        source_device, source_dtype = images.device, images.dtype
        completed = []
        for start in range(0, int(cropped.shape[0]), 4):
            piece = cropped[start:start + 4].permute(0, 3, 1, 2).to(dtype=torch.float32)
            piece = torch.nn.functional.interpolate(
                piece,
                size=(target_height, target_width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).permute(0, 2, 3, 1).clamp_(0.0, 1.0)
            completed.append(piece.to(device=source_device, dtype=source_dtype))
        resized = torch.cat(completed, dim=0)
        return resized, (
            f"R42 exact LTX delivery | protected={source_width}x{source_height} | "
            f"center_crop={crop_width}x{crop_height} | final={target_width}x{target_height}"
        )


def _stream_rtx_frames(frames, output, audio, width, height, fps, quality):
    """Stream native tensors through RTX VSR without a Full-HD batch or temp video."""
    import comfy.model_management as mm
    from .iamccs_rtx_vfx import _import_video_super_res

    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("NATIVE -> RTX FINAL requires ffmpeg")
    if output.exists():
        raise FileExistsError(f"NATIVE -> RTX FINAL refuses to overwrite: {output}")
    VideoSuperRes = _import_video_super_res()
    frame_count = int(frames.shape[0])
    source_height, source_width = int(frames.shape[1]), int(frames.shape[2])
    target_aspect = float(width) / float(height)
    source_aspect = float(source_width) / float(source_height)
    crop_width, crop_height = source_width, source_height
    if source_aspect > target_aspect:
        crop_width = max(2, min(source_width, round(source_height * target_aspect)))
    elif source_aspect < target_aspect:
        crop_height = max(2, min(source_height, round(source_width / target_aspect)))
    crop_top = max(0, (source_height - crop_height) // 2)
    crop_left = max(0, (source_width - crop_width) // 2)
    grid_width = ((int(width) + 7) // 8) * 8
    grid_height = ((int(height) + 7) // 8) * 8
    with tempfile.TemporaryDirectory(prefix="iamccs_native_rtx_") as temp, tempfile.TemporaryFile() as errors:
        wav = Path(temp) / "native_audio.wav"
        if not _write_wav(audio, wav):
            raise ValueError("NATIVE -> RTX FINAL requires native AUDIO")
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
            "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0", "-frames:v", str(frame_count),
            "-af", f"apad,atrim=duration={frame_count / fps:.9f},asetpts=PTS-STARTPTS",
            "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
        written = 0
        try:
            with VideoSuperRes(quality=getattr(VideoSuperRes.QualityLevel, quality), device=0) as effect:
                effect.output_width, effect.output_height = grid_width, grid_height
                effect.load()
                for frame in frames:
                    mm.throw_exception_if_processing_interrupted()
                    frame = frame[crop_top:crop_top + crop_height, crop_left:crop_left + crop_width, :3]
                    rgb = frame.to(device="cuda:0", dtype=torch.float32).permute(2, 0, 1).contiguous()
                    enhanced = torch.from_dlpack(effect.run(rgb).image).permute(1, 2, 0)
                    top = max(0, (int(enhanced.shape[0]) - int(height)) // 2)
                    left = max(0, (int(enhanced.shape[1]) - int(width)) // 2)
                    encoded = enhanced[top:top + height, left:left + width].clamp(0, 1).mul(255).round().to(
                        device="cpu", dtype=torch.uint8,
                    ).contiguous()
                    process.stdin.write(encoded.numpy().tobytes())
                    del rgb, enhanced, encoded
                    written += 1
                    if written % 24 == 0 or written == frame_count:
                        __import__("logging").getLogger("IAMCCS.MiniMaxH3.RTXFinal").info(
                            "NATIVE -> RTX FINAL progress | %d/%d | %dx%d", written, frame_count, width, height,
                        )
            process.stdin.close()
            code = process.wait()
            if code or written != frame_count:
                errors.seek(0)
                detail = errors.read().decode("utf8", "replace")[-1800:]
                raise RuntimeError(f"NATIVE -> RTX FINAL wrote {written}/{frame_count} frames: {detail}")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()


def _stream_pixel_safe_frames(frames, output, audio, width, height, fps, settings):
    """Encode one frame at a time; optional core Spandrel spatial tiling."""
    import comfy.model_management as mm
    import comfy.utils
    from PIL import Image

    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("Pixel Safe delivery requires ffmpeg")
    if output.exists():
        raise FileExistsError(f"Pixel Safe delivery refuses to overwrite: {output}")
    method = str(settings.get("method", "cpu_lanczos") or "cpu_lanczos")
    if method not in {"cpu_lanczos", "model_tiled"}:
        raise ValueError(f"Unknown Pixel Safe method: {method}")
    model = None
    if method == "model_tiled":
        from comfy_extras.nodes_upscale_model import UpscaleModelLoader
        model_name = str(settings.get("model_name", "") or "").strip()
        if not model_name:
            raise ValueError("Pixel Safe model_tiled needs a local model in ComfyUI/models/upscale_models")
        model = UpscaleModelLoader.execute(model_name)[0]
        mm.load_models_gpu([model.patcher], force_full_load=False)
    frame_count = int(frames.shape[0])
    source_height, source_width = int(frames.shape[1]), int(frames.shape[2])
    with tempfile.TemporaryDirectory(prefix="iamccs_pixel_safe_") as temp, tempfile.TemporaryFile() as errors:
        wav = Path(temp) / "native_audio.wav"
        if not _write_wav(audio, wav):
            raise ValueError("Pixel Safe delivery requires native AUDIO")
        if method == "cpu_lanczos":
            input_width, input_height = source_width, source_height
            ratio = min(source_width / width, source_height / height)
            crop_width = max(2, min(source_width, int(width * ratio) // 2 * 2))
            crop_height = max(2, min(source_height, int(height * ratio) // 2 * 2))
            vf = f"crop={crop_width}:{crop_height},scale={width}:{height}:flags=lanczos"
        else:
            input_width, input_height, vf = width, height, None
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{input_width}x{input_height}",
            "-r", str(fps), "-i", "pipe:0", "-i", str(wav),
            "-map", "0:v:0", "-map", "1:a:0", "-frames:v", str(frame_count),
        ]
        if vf:
            command.extend(["-vf", vf])
        command.extend([
            "-af", f"apad,atrim=duration={frame_count / fps:.9f},asetpts=PTS-STARTPTS",
            "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output),
        ])
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
        written = 0
        try:
            for frame in frames:
                mm.throw_exception_if_processing_interrupted()
                if model is None:
                    encoded = frame[..., :3].clamp(0, 1).mul(255).round().to(device="cpu", dtype=torch.uint8).contiguous()
                    payload = encoded.numpy().tobytes()
                else:
                    tile = max(128, min(2048, int(settings.get("tile_size", 512))))
                    overlap = max(0, min(tile // 3, int(settings.get("overlap", 64))))
                    source = frame[..., :3].permute(2, 0, 1).unsqueeze(0).to(model.patcher.load_device)
                    while True:
                        try:
                            result = comfy.utils.tiled_scale(
                                source, lambda patch: model(patch.float()), tile_x=tile, tile_y=tile,
                                overlap=overlap, upscale_amount=model.scale, output_device="cpu",
                            )
                            break
                        except Exception as exc:
                            mm.raise_non_oom(exc)
                            tile //= 2
                            if tile < 128:
                                raise
                            overlap = min(overlap, tile // 3)
                            mm.soft_empty_cache()
                    image = result.squeeze(0).permute(1, 2, 0).clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
                    pil = Image.fromarray(image, "RGB")
                    crop_ratio = min(pil.width / width, pil.height / height)
                    crop_w, crop_h = int(width * crop_ratio), int(height * crop_ratio)
                    left, top = (pil.width - crop_w) // 2, (pil.height - crop_h) // 2
                    pil = pil.crop((left, top, left + crop_w, top + crop_h)).resize((width, height), Image.Resampling.LANCZOS)
                    payload = pil.tobytes()
                    del source, result, image, pil
                process.stdin.write(payload)
                written += 1
            process.stdin.close()
            code = process.wait()
            if code or written != frame_count:
                errors.seek(0)
                detail = errors.read().decode("utf8", "replace")[-1800:]
                raise RuntimeError(f"Pixel Safe delivery wrote {written}/{frame_count} frames: {detail}")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()


def _route(cine_linx: Any) -> str:
    plan = _resolve_shotplan(cine_linx)
    if not bool(plan.get("upscale_enabled", False)):
        return "off"
    route = str(plan.get("upscale_mode", "off") or "off").strip().lower()
    if route not in KNOWN_ROUTES:
        raise ValueError(f"R42 universal delivery does not recognize upscale route: {route}")
    return route


class IAMCCS_MiniMaxH3UniversalRouteControlR42:
    """Choose the sole queue owner without evaluating a delivery branch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"cine_linx": (SUPERNODE_LINX_TYPE,)}}

    RETURN_TYPES = ("BOOLEAN", "BOOLEAN", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = (
        "native_checkpoint_queues_next",
        "per_segment_delivery_queues_next",
        "master_delivery_route",
        "selected_route",
        "report",
    )
    FUNCTION = "resolve"
    CATEGORY = CATEGORY

    def resolve(self, cine_linx):
        route = _route(cine_linx)
        master = route in MASTER_ROUTES
        per_segment = route in {"off", RTX_FINAL_ROUTE, PIXEL_TILED_ROUTE, "h3_pixel_refine", FAST_ROUTE, ULTIMATE_ROUTE, LTX_PER_CHUNK_ROUTE}
        report = (
            f"R42 universal route={route} | queue_owner="
            f"{'native master checkpoint' if master else 'selected per-segment delivery'} | lazy=yes"
        )
        return master, per_segment, master, route, report


class IAMCCS_MiniMaxH3UniversalNativeWindowedR42(IAMCCS_MiniMaxH3PixelRefineR38B):
    """Native/Windowed R38B branch, evaluated only when selected by R42."""

    OUTPUT_NODE = False
    CATEGORY = CATEGORY

    def finish(self, *args, **kwargs):
        cine_linx = kwargs.get("cine_linx")
        if cine_linx is None and len(args) >= 5:
            cine_linx = args[4]
        route = _route(cine_linx)
        if route not in {"off", "h3_pixel_refine"}:
            raise ValueError(f"R42 Native/Windowed branch was requested for incompatible route: {route}")
        kwargs["queue_next_segment"] = True
        return super().finish(*args, **kwargs)


class IAMCCS_MiniMaxH3UniversalFastR42(IAMCCS_MiniMaxH3FastLatent2PassR41):
    """Shared lazy H3 delivery socket: R41 full-pass or Ultimate tiled."""

    OUTPUT_NODE = False
    CATEGORY = CATEGORY

    def finish(self, *args, **kwargs):
        cine_linx = kwargs.get("cine_linx")
        if cine_linx is None and len(args) >= 7:
            cine_linx = args[6]
        route = _route(cine_linx)
        if route == ULTIMATE_ROUTE:
            kwargs["queue_next_segment"] = True
            return IAMCCS_MiniMaxH3UltimateTiledDelivery().finish(*args, **kwargs)
        if route != FAST_ROUTE:
            raise ValueError(f"R42 Fast branch was requested for incompatible route: {route}")
        kwargs["queue_next_segment"] = True
        return super().finish(*args, **kwargs)


class IAMCCS_MiniMaxH3UniversalRTXFinalR42:
    """One native H3 sample followed by streaming RTX VSR delivery.

    The route deliberately accepts the already checkpointed native IMAGE/AUDIO
    rather than MODEL, CONDITIONING or LATENT.  Consequently ComfyUI cannot
    evaluate a second H3 denoise pass for this delivery path.  Frames are fed
    to RTX VSR one at a time and directly encoded, so a Full-HD IMAGE batch is
    never retained in RAM or VRAM.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "native_frames": ("IMAGE",),
                "native_audio": ("AUDIO",),
                "resolved_render_id": ("STRING", {"forceInput": True}),
                "native_saved_report": ("STRING", {"forceInput": True}),
                "current_segment": ("INT", {"forceInput": True}),
                "total_segments": ("INT", {"forceInput": True}),
                "fps": ("INT", {"forceInput": True}),
                "join_trim_frames": ("INT", {"forceInput": True}),
                "queue_next_segment": ("BOOLEAN", {"default": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "report")
    FUNCTION = "finish"
    OUTPUT_NODE = False
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    def finish(
        self,
        cine_linx,
        native_frames,
        native_audio,
        resolved_render_id,
        native_saved_report,
        current_segment,
        total_segments,
        fps,
        join_trim_frames,
        queue_next_segment=True,
        prompt=None,
        extra_pnginfo=None,
    ):
        import gc
        import comfy.model_management as mm

        route = _route(cine_linx)
        if route not in {RTX_FINAL_ROUTE, PIXEL_TILED_ROUTE}:
            raise ValueError(f"Streaming pixel branch was requested for incompatible route: {route}")
        route_name = "RTX Final" if route == RTX_FINAL_ROUTE else "Pixel Safe"
        if not native_saved_report:
            raise ValueError(f"{route_name} must run after the native safety checkpoint")
        if not torch.is_tensor(native_frames) or native_frames.ndim != 4 or int(native_frames.shape[0]) < 1:
            raise ValueError(f"{route_name} expects the checkpointed native IMAGE frames")
        if not isinstance(native_audio, dict) or not torch.is_tensor(native_audio.get("waveform")):
            raise ValueError(f"{route_name} expects the checkpointed native AUDIO")

        plan = _resolve_shotplan(cine_linx)
        index, total = int(current_segment), int(total_segments)
        fps = max(1, int(fps))
        if index < 0 or total < 1 or index >= total:
            raise ValueError(f"{route_name} received invalid segment index {index + 1}/{total}")
        width = int(plan.get("upscale_width", 1920) or 1920)
        height = int(plan.get("upscale_height", 1080) or 1080)
        if width < 2 or height < 2 or width % 2 or height % 2:
            raise ValueError(f"{route_name} delivery width and height must be positive even numbers")
        settings = plan.get("h3_upres_settings", {}) if isinstance(plan.get("h3_upres_settings"), dict) else {}
        quality = str(settings.get("rtx_quality", "HIGH") or "HIGH").strip().upper()
        if route == RTX_FINAL_ROUTE and quality not in {"LOW", "MEDIUM", "HIGH", "ULTRA"}:
            raise ValueError(f"RTX Final quality is invalid: {quality}")

        run = _safe_name(str(resolved_render_id or "").strip(), "minimax_h3_render")
        route_label = "RTX_FINAL" if route == RTX_FINAL_ROUTE else "PIXEL_SAFE"
        root = Path(folder_paths.get_output_directory()) / "IAMCCS" / "MiniMaxH3" / route_label / run
        root.mkdir(parents=True, exist_ok=True)
        output = root / f"segment_{index + 1:04d}.mp4"
        if output.exists():
            raise FileExistsError(f"{route_name} refuses to overwrite existing segment: {output}")

        from .iamccs_minimax_h3_shotboard import _delivery_join_frames
        join = _delivery_join_frames(cine_linx, index, join_trim_frames)
        drop = 1 if join == 1 else 0
        visible_frames = native_frames[drop:] if drop else native_frames
        audio = _trim_audio_frames(native_audio, drop, fps) if drop else native_audio
        frames = int(visible_frames.shape[0])
        if frames < 1:
            raise ValueError(f"{route_name} has no visible frame after the authored join trim")
        native_h, native_w = int(visible_frames.shape[1]), int(visible_frames.shape[2])

        # Stream one native tensor at a time through RTX VSR directly into the
        # final ffmpeg pipe. No temporary compressed video and no upscaled
        # IMAGE batch are materialized.
        LOG = __import__("logging").getLogger("IAMCCS.MiniMaxH3.StreamingPixel")
        LOG.info(
            "NATIVE -> %s start | segment=%d/%d | native=%dx%d | delivery=%dx%d | frames=%d | method=%s | H3_resample=no | audio=locked",
            route_name, index + 1, total, native_w, native_h, width, height, frames,
            quality if route == RTX_FINAL_ROUTE else plan.get("upscale_settings", {}).get("pixel_tiled", {}).get("method", "cpu_lanczos"),
        )
        try:
            mm.unload_all_models()
            mm.soft_empty_cache()
            if route == RTX_FINAL_ROUTE:
                _stream_rtx_frames(visible_frames, output, audio, width, height, fps, quality)
            else:
                pixel_settings = plan.get("upscale_settings", {}).get("pixel_tiled", {})
                _stream_pixel_safe_frames(visible_frames, output, audio, width, height, fps, pixel_settings)
        finally:
            mm.unload_all_models()
            gc.collect()
            mm.soft_empty_cache()

        source_workflow = extra_pnginfo.get("workflow") if isinstance(extra_pnginfo, dict) else None
        _write_segment_metadata(
            output,
            frames,
            fps,
            "trim_silent_tail" if bool(native_audio.get("iamccs_flf_locked_audio_handles", False)) else "crossfade",
            provenance={"render_id": run, "stage": route_label.lower(), "segment_index": index,
                        "total_segments": total, "shotplan": plan},
            source_workflow=source_workflow,
        )
        preview = output
        if index == total - 1:
            paths = [root / f"segment_{i + 1:04d}.mp4" for i in range(total)]
            preview = root / "final_film.mp4"
            if preview.exists():
                raise FileExistsError(f"{route_name} refuses to overwrite existing master: {preview}")
            if join > 1:
                _concat_videos_overlap(paths, preview, join, fps)
            else:
                _concat_videos(paths, preview)
            from .iamccs_minimax_h3_shotboard import _joined_frame_count
            _write_segment_metadata(
                preview, _joined_frame_count(paths, join if join > 1 else 0), fps,
                "crossfade" if join > 1 else "direct",
                provenance={"render_id": run, "stage": f"{route_label.lower()}_master",
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
            f"NATIVE -> {route_label} saved {index + 1}/{total} | {native_w}x{native_h} -> {width}x{height} | "
            f"{frames}f | one H3 sample | native audio locked | next queued={queued} | {preview}"
        )
        LOG.info(report)
        return {"ui": {"text": [report]}, "result": (str(preview), report)}


class IAMCCS_MiniMaxH3UniversalNativeCheckpointR42(IAMCCS_MiniMaxH3NativeCheckpointSave):
    """Always save native; queue only when a one-film delivery needs all chunks."""

    # The legacy checkpoint class is an OUTPUT_NODE because older standalone
    # workflows terminate directly on it.  In the universal graph the real
    # terminal is UniversalPathRouterR42, which lazily requests exactly one
    # delivery branch.  Inheriting OUTPUT_NODE=True here made ComfyUI execute
    # this checkpoint independently as a second terminal, evaluating a stale
    # standard H3 path even when LatentGoAhead/Continuous AV was selected.
    # Keep the legacy class untouched and make only this universal subclass an
    # ordinary upstream node.
    OUTPUT_NODE = False
    CATEGORY = CATEGORY

    def checkpoint(self, *args, **kwargs):
        cine_linx = kwargs.get("cine_linx")
        # The workflow connects cine_linx explicitly.  Legacy positional calls
        # remain valid but cannot own R42 continuation without that contract.
        route = _route(cine_linx) if cine_linx is not None else "off"
        kwargs["queue_next_segment"] = route in MASTER_ROUTES
        if route in MASTER_ROUTES:
            # A master delivery consumes the complete programme.  Keep the
            # editable checkpoint policy for other routes, but never hand LTX
            # only the final isolated H3 chunk.
            kwargs["merge_segments"] = True
            kwargs["editor_delivery_policy"] = "legacy_master"
            current = int(kwargs.get("current_segment", args[2] if len(args) > 2 else 0))
            total = max(1, int(kwargs.get("total_segments", args[3] if len(args) > 3 else 1)))
            LOG.info(
                "R42 LTX master delivery armed | native checkpoint=%d/%d | LTX starts only after completed native master",
                current + 1, total,
            )
        return super().checkpoint(*args, **kwargs)


class IAMCCS_MiniMaxH3UniversalMasterSaveR42:
    """Save either the one-film LTX master or one editorial LTX shot.

    Optional future sockets are retained as an append-only extension point, but
    R42 accepts only the physically connected LTX route.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "master_ready": ("BOOLEAN", {"forceInput": True}),
                "resolved_render_id": ("STRING", {"forceInput": True}),
                "fps": ("INT", {"forceInput": True}),
            },
            "optional": {
                "master_audio": ("AUDIO", {"lazy": True}),
                "ltx_frames": ("IMAGE", {"lazy": True}),
                "wan_frames": ("IMAGE", {"lazy": True}),
                "h3_latent_frames": ("IMAGE", {"lazy": True}),
                "current_segment": ("INT", {"forceInput": True}),
                "total_segments": ("INT", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "IAMCCS/MiniMaxH3/UNIVERSAL_R42"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "report")
    FUNCTION = "save"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    @staticmethod
    def _selected_input(cine_linx):
        return {
            "ltx23": "ltx_frames",
            LTX_PER_CHUNK_ROUTE: "ltx_frames",
            "wan22_5b": "wan_frames",
            "h3_latent_upres": "h3_latent_frames",
        }.get(_route(cine_linx))

    def check_lazy_status(
        self,
        cine_linx,
        master_ready,
        resolved_render_id,
        fps,
        master_audio=None,
        ltx_frames=None,
        wan_frames=None,
        h3_latent_frames=None,
        **kwargs,
    ):
        if not bool(master_ready):
            return []
        selected = self._selected_input(cine_linx)
        if selected is None:
            return []
        missing = []
        if master_audio is None:
            missing.append("master_audio")
        if {"ltx_frames": ltx_frames, "wan_frames": wan_frames, "h3_latent_frames": h3_latent_frames}[selected] is None:
            missing.append(selected)
        return missing

    def save(
        self,
        cine_linx,
        master_ready,
        resolved_render_id,
        fps,
        master_audio=None,
        ltx_frames=None,
        wan_frames=None,
        h3_latent_frames=None,
        current_segment=None,
        total_segments=None,
        filename_prefix="IAMCCS/MiniMaxH3/UNIVERSAL_R42",
        prompt=None,
        extra_pnginfo=None,
    ):
        route = _route(cine_linx)
        if route not in MASTER_ROUTES and route != LTX_PER_CHUNK_ROUTE:
            return "", f"R42 master delivery bypassed: selected={route}"
        if not bool(master_ready):
            return "", f"R42 {route} waiting for completed native master"
        selected = {
            "ltx23": ltx_frames,
            LTX_PER_CHUNK_ROUTE: ltx_frames,
            "wan22_5b": wan_frames,
            "h3_latent_upres": h3_latent_frames,
        }[route]
        if not torch.is_tensor(selected) or selected.ndim != 4 or int(selected.shape[0]) < 1:
            raise ValueError(f"R42 selected {route}, but its lazy IMAGE branch is not connected")
        if not isinstance(master_audio, dict) or not torch.is_tensor(master_audio.get("waveform")):
            raise ValueError(f"R42 selected {route}, but the native master AUDIO is not connected")
        run = _safe_name(str(resolved_render_id or "").strip(), "minimax_h3_render")
        if route == LTX_PER_CHUNK_ROUTE:
            if current_segment is None or total_segments is None:
                raise ValueError("LTX per-shot requires current_segment and total_segments connections")
            index, total = int(current_segment), max(1, int(total_segments))
            if index < 0 or index >= total:
                raise ValueError(f"LTX per-shot received invalid segment index {index + 1}/{total}")
            root = (
                Path(folder_paths.get_output_directory())
                / Path(str(filename_prefix).replace("\\", "/"))
                / run
                / "LTX_PER_SHOT"
            )
            root.mkdir(parents=True, exist_ok=True)
            output = root / f"segment_{index + 1:04d}.mp4"
            if output.exists():
                raise FileExistsError(f"LTX per-shot refuses to overwrite existing segment: {output}")
            _encode_images(selected, master_audio, max(1, int(fps)), output)
            _write_segment_metadata(
                output, int(selected.shape[0]), max(1, int(fps)), "editorial_hard_cut",
                provenance={"render_id": run, "stage": "ltx_per_shot", "segment_index": index,
                            "total_segments": total, "shotplan": _resolve_shotplan(cine_linx)},
                source_workflow=extra_pnginfo.get("workflow") if isinstance(extra_pnginfo, dict) else None,
            )
            queued = False
            if index + 1 < total:
                live, extra, outputs, sensitive = _current_prompt()
                next_prompt = _next_prompt(live if live is not None else prompt, run, index + 1)
                _enqueue(next_prompt, extra_data=extra, outputs=outputs, sensitive=sensitive)
                queued = True
            report = (
                f"LTX per-shot delivery saved | segment={index + 1}/{total} | "
                f"frames={int(selected.shape[0])} | audio=segment locked | next queued={queued} | {output}"
            )
            LOG.info(report)
            return str(output), report

        root = Path(folder_paths.get_output_directory()) / Path(str(filename_prefix).replace("\\", "/"))
        root.mkdir(parents=True, exist_ok=True)
        output = root / f"{run}_{route}_full.mp4"
        if output.exists():
            raise FileExistsError(f"R42 refuses to overwrite existing delivery: {output}")
        _encode_images(selected, master_audio, max(1, int(fps)), output)
        _write_segment_metadata(
            output, int(selected.shape[0]), max(1, int(fps)), "master_audio_locked",
            provenance={"render_id": run, "stage": route, "shotplan": _resolve_shotplan(cine_linx)},
            source_workflow=extra_pnginfo.get("workflow") if isinstance(extra_pnginfo, dict) else None,
        )
        report = f"R42 master delivery saved | route={route} | frames={int(selected.shape[0])} | {output}"
        return str(output), report


class IAMCCS_MiniMaxH3UniversalPathRouterR42:
    """Request exactly one delivery path and make the universal graph an output."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "current_segment": ("INT", {"forceInput": True}),
                "total_segments": ("INT", {"forceInput": True}),
            },
            "optional": {
                "native_or_windowed_path": ("STRING", {"lazy": True}),
                "rtx_final_path": ("STRING", {"lazy": True}),
                "fast_path": ("STRING", {"lazy": True}),
                "master_path": ("STRING", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("video_path", "delivery_ready", "report")
    FUNCTION = "route"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    @staticmethod
    def _selected_input(cine_linx):
        route = _route(cine_linx)
        if route in {"off", "h3_pixel_refine"}:
            return "native_or_windowed_path"
        if route in {RTX_FINAL_ROUTE, PIXEL_TILED_ROUTE}:
            return "rtx_final_path"
        if route in {FAST_ROUTE, ULTIMATE_ROUTE}:
            return "fast_path"
        return "master_path"

    def check_lazy_status(
        self,
        cine_linx,
        current_segment,
        total_segments,
        native_or_windowed_path=None,
        rtx_final_path=None,
        fast_path=None,
        master_path=None,
        **kwargs,
    ):
        selected = self._selected_input(cine_linx)
        values = {
            "native_or_windowed_path": native_or_windowed_path,
            "rtx_final_path": rtx_final_path,
            "fast_path": fast_path,
            "master_path": master_path,
        }
        return [selected] if values[selected] is None else []

    def route(
        self,
        cine_linx,
        current_segment,
        total_segments,
        native_or_windowed_path=None,
        rtx_final_path=None,
        fast_path=None,
        master_path=None,
    ):
        route = _route(cine_linx)
        selected = self._selected_input(cine_linx)
        value = {
            "native_or_windowed_path": native_or_windowed_path,
            "rtx_final_path": rtx_final_path,
            "fast_path": fast_path,
            "master_path": master_path,
        }[selected]
        path = str(value or "").strip()
        index, total = max(0, int(current_segment)), max(1, int(total_segments))
        ready = bool(path)
        report = (
            f"R42 universal delivery | route={route} | segment={index + 1}/{total} | "
            f"ready={'yes' if ready else 'waiting for native master'} | {path or 'no path yet'}"
        )
        ui = {"text": [report]}
        if ready:
            authored = Path(path)
            if authored.is_file():
                try:
                    subfolder = authored.parent.relative_to(
                        Path(folder_paths.get_output_directory())
                    ).as_posix()
                except ValueError:
                    # ComfyUI cannot serve arbitrary filesystem paths through
                    # its output preview endpoint. The path remains visible in
                    # the report and no misleading preview is emitted.
                    subfolder = None
                if subfolder is not None:
                    ui.update({
                        "images": [{
                            "filename": authored.name,
                            "subfolder": subfolder,
                            "type": "output",
                        }],
                        "animated": (True,),
                    })
        return {"ui": ui, "result": (path, ready, report)}


class IAMCCS_MiniMaxH3UniversalEditorPolicyR42:
    """Adapt only the editor-facing copy of a one-film delivery plan.

    Generation truth is never changed.  Native/Windowed/Fast and explicit LTX
    per-shot routes keep their normal editorial-roll behavior, while LTX master
    delivery is exposed to the existing editor nodes as one completed programme
    (the same proven policy already used by LongVid).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"cine_linx": (SUPERNODE_LINX_TYPE,)}}

    RETURN_TYPES = (SUPERNODE_LINX_TYPE, "BOOLEAN", "STRING")
    RETURN_NAMES = ("editor_cine_linx", "one_film_delivery", "report")
    FUNCTION = "adapt"
    CATEGORY = CATEGORY

    def adapt(self, cine_linx):
        route = _route(cine_linx)
        one_film = route in MASTER_ROUTES
        if not one_film:
            return cine_linx, False, f"R42 editor policy | route={route} | publish=shot-by-shot"

        adapted = deepcopy(cine_linx)
        plan = _resolve_shotplan(adapted)
        if not isinstance(plan, dict):
            raise ValueError("R42 editor policy could not resolve the H3 shotplan")
        plan["task_mode"] = "longvid_r42_one_film_delivery"
        plan["r42_editor_original_task_mode"] = str(_resolve_shotplan(cine_linx).get("task_mode", ""))
        plan["r42_editor_delivery_route"] = route
        return adapted, True, f"R42 editor policy | route={route} | publish=one completed master"


NODE_CLASS_MAPPINGS = {
    "IAMCCS_MiniMaxH3ExactLTXDeliverySizeR42": IAMCCS_MiniMaxH3ExactLTXDeliverySizeR42,
    "IAMCCS_MiniMaxH3UniversalRouteControlR42": IAMCCS_MiniMaxH3UniversalRouteControlR42,
    "IAMCCS_MiniMaxH3UniversalNativeCheckpointR42": IAMCCS_MiniMaxH3UniversalNativeCheckpointR42,
    "IAMCCS_MiniMaxH3UniversalNativeWindowedR42": IAMCCS_MiniMaxH3UniversalNativeWindowedR42,
    "IAMCCS_MiniMaxH3UniversalRTXFinalR42": IAMCCS_MiniMaxH3UniversalRTXFinalR42,
    "IAMCCS_MiniMaxH3UniversalFastR42": IAMCCS_MiniMaxH3UniversalFastR42,
    "IAMCCS_MiniMaxH3UltimateTiledDelivery": IAMCCS_MiniMaxH3UltimateTiledDelivery,
    "IAMCCS_MiniMaxH3UniversalMasterSaveR42": IAMCCS_MiniMaxH3UniversalMasterSaveR42,
    "IAMCCS_MiniMaxH3UniversalPathRouterR42": IAMCCS_MiniMaxH3UniversalPathRouterR42,
    "IAMCCS_MiniMaxH3UniversalEditorPolicyR42": IAMCCS_MiniMaxH3UniversalEditorPolicyR42,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_MiniMaxH3ExactLTXDeliverySizeR42": "MiniMax H3 R42 · Exact LTX Delivery Size",
    "IAMCCS_MiniMaxH3UniversalRouteControlR42": "MiniMax H3 R42 · Universal Route + Queue Owner",
    "IAMCCS_MiniMaxH3UniversalNativeCheckpointR42": "MiniMax H3 R42 · Native Safety Checkpoint",
    "IAMCCS_MiniMaxH3UniversalNativeWindowedR42": "MiniMax H3 · Native / Safe Windowed Delivery",
    "IAMCCS_MiniMaxH3UniversalRTXFinalR42": "MiniMax H3 · Native → RTX Final · Streaming",
    "IAMCCS_MiniMaxH3UniversalFastR42": "MiniMax H3 · R42 Universal H3 Delivery",
    "IAMCCS_MiniMaxH3UltimateTiledDelivery": "IAMCCS H3 ULTIMATE LATENT · TILED",
    "IAMCCS_MiniMaxH3UniversalMasterSaveR42": "MiniMax H3 · LTX Master Delivery Save",
    "IAMCCS_MiniMaxH3UniversalPathRouterR42": "MiniMax H3 · Lazy Delivery Path Router",
    "IAMCCS_MiniMaxH3UniversalEditorPolicyR42": "MiniMax H3 R42 · Editor Shot / Master Policy",
}
