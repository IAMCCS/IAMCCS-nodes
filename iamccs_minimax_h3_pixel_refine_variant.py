# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""R38B post-native delivery. Does not change R37 conditioning or sampling."""

from __future__ import annotations

import copy
import gc
import importlib
import logging
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import folder_paths
import torch

from .iamccs_minimax_h3_atomic_backend import SUPERNODE_LINX_TYPE, _resolve_shotplan
from .iamccs_minimax_h3_latent_upres_variant import (
    _delivery_size, _grid_up, _safe_render_id, _upres_settings,
    IAMCCS_MiniMaxH3LatentUpresSamplingR38,
)

LOG = logging.getLogger("IAMCCS.MiniMaxH3.PixelRefineR38B")
CATEGORY = "IAMCCS/MiniMax H3/Safe Windowed Refine"
_PROVIDER_PACKAGE = "_iamccs_r38b_mmh3tools"


def _provider(module):
    # Do not execute upstream's package initializer: it patches the tokenizer.
    if _PROVIDER_PACKAGE not in sys.modules:
        package = types.ModuleType(_PROVIDER_PACKAGE)
        package.__path__ = [str(Path(__file__).parent / "vendor" / "mmh3tools_r38b" / "mmh3tools")]
        sys.modules[_PROVIDER_PACKAGE] = package
    return importlib.import_module(f"{_PROVIDER_PACKAGE}.{module}")


def _rtx_frames(px, width, height, method, rtx_quality="ULTRA", device=None, sub_batch=17):
    """Use the installed IAMCCS RTX runtime; consume each DLPack view immediately."""
    from .iamccs_rtx_vfx import _import_video_super_res

    if tuple(px.shape[1:3]) == (height, width):
        return px
    VideoSuperRes = _import_video_super_res()
    quality = getattr(VideoSuperRes.QualityLevel, rtx_quality)
    result = torch.empty((len(px), height, width, 3), device="cpu", dtype=px.dtype)
    with VideoSuperRes(quality=quality, device=0) as effect:
        effect.output_width, effect.output_height = int(width), int(height)
        effect.load()
        for i in range(len(px)):
            frame = px[i, ..., :3].to(device="cuda:0", dtype=torch.float32).permute(2, 0, 1).contiguous()
            enhanced = effect.run(frame)
            result[i].copy_(torch.from_dlpack(enhanced.image).permute(1, 2, 0).clamp(0, 1).to("cpu"))
    return result


def _pixel_upscale(latent, vae, width, height, settings):
    module = _provider("nodes_upscale")
    method = settings.get("pixel_method", "rtx_vsr")
    original = module.upscale_frames
    # The private provider instance is used only by this execution path.
    if method == "rtx_vsr":
        module.upscale_frames = _rtx_frames
    try:
        return module.MMH3ChunkedPixelUpscale.execute(
            latent=latent, vae=vae, width=width, height=height, method=method,
            groups_per_chunk=int(settings.get("pixel_groups", 1)),
            rtx_quality=settings.get("rtx_quality", "ULTRA"), offload_latents=True,
        )[0]
    finally:
        module.upscale_frames = original


def _refine(model, conditioning, latent, cine_linx, segment_index):
    from comfy_extras.nodes_custom_sampler import BasicGuider

    loop = _provider("nodes_looping_sampler")
    if not loop.per_row_mask_is_continuous():
        raise RuntimeError("R38B needs H3 per-row masks in ComfyUI (#15375); core was not modified automatically.")
    plan = _resolve_shotplan(cine_linx)
    settings = _upres_settings(plan)
    steps, denoise = int(settings.get("steps", 2)), float(settings.get("denoise", 0.25))
    if steps < 1 or not 0.0 <= denoise <= 1.0:
        raise ValueError("R38B refinement requires steps >= 1 and denoise in [0, 1].")
    # The workflow supplies the PRE-sampler model. Native sampling applies LoRA,
    # shifts and attention to a private clone, not to this graph output. Prepare
    # our own clone once using the existing delivery contract and authored values.
    active_model, noise, sampler, sigmas, _ = IAMCCS_MiniMaxH3LatentUpresSamplingR38().prepare(
        model, cine_linx, segment_index)
    # Refine the already rendered clip, not the original still guides again.
    # The provider removes guides per window without mutating incoming conditioning.
    guider = BasicGuider.execute(model=active_model, conditioning=conditioning)[0]
    common = _provider("common")
    original_video, original_audio = common.unpack_av(latent, "R38B source")
    total_frames = common.latents_to_frames(int(original_video.shape[2]))
    length, overlap, _, _, windows = _provider("nodes_windows")._plan(
        total_frames, int(settings.get("window_frames", 136)),
        int(settings.get("window_overlap", 22)), "standard_static")
    LOG.info("R38B effective windows | %d passes x %d steps | %df window / %df overlap | %df total | pixel groups=%s",
             len(windows), steps, common.latents_to_frames(length),
             common.frame_at_latent(overlap) if overlap else 0, total_frames, settings.get("pixel_groups", 1))
    result = loop.MMH3LoopingSampler.execute(
        noise=noise, guider=guider, sampler=sampler, sigmas=sigmas,
        cond_set={"conds": [conditioning]}, latent=latent,
        chunk_frames=int(settings.get("window_frames", 136)),
        overlap_frames=int(settings.get("window_overlap", 22)),
        carry="mask", overlap_strength_video=1.0, overlap_strength_audio=1.0,
        audio_denoise_mask=torch.zeros((1, 1, 1)),
    )[0]
    video, audio = common.unpack_av(result, "R38B refined")
    if tuple(video.shape) != tuple(original_video.shape):
        raise RuntimeError("R38B changed the video latent extent; refusing an altered duration.")
    if original_audio is not None and not torch.equal(original_audio.cpu(), audio.cpu()):
        raise RuntimeError("R38B audio lock changed during refinement; refusing a desynchronised result.")
    return result


def _finish_segment(intermediate, output, audio, trim, frames, width, height, fps):
    from .iamccs_minimax_h3_shotboard import _find_ffmpeg, _write_wav

    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("R38B requires the same ffmpeg used by the native checkpoint.")
    if output.exists():
        raise FileExistsError(f"R38B refuses to overwrite an existing segment: {output}")
    with tempfile.TemporaryDirectory(prefix="iamccs_r38b_mux_") as temp:
        wav = Path(temp) / "native_audio.wav"
        if not _write_wav(audio, wav):
            raise ValueError("R38B requires the native AUDIO output, not a new audio loader.")
        vf = f"trim=start_frame={trim}:end_frame={trim + frames},setpts=PTS-STARTPTS,crop={width}:{height}"
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                   "-i", str(intermediate), "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0",
                   "-vf", vf, "-frames:v", str(frames), "-r", str(fps),
                   "-af", f"apad,atrim=duration={frames / fps:.9f},asetpts=PTS-STARTPTS",
                   "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
                   "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output)]
        completed = subprocess.run(command, capture_output=True)
        if completed.returncode:
            raise RuntimeError("R38B delivery encode failed: " + completed.stderr.decode("utf8", "replace")[-2000:])


def _remove_audio_frame_slice(audio, frame_index: int, fps: float = 24.0):
    """Remove one video-frame-equivalent slice from native audio."""
    if not isinstance(audio, dict) or not torch.is_tensor(audio.get("waveform")):
        return audio
    result = dict(audio)
    waveform = result["waveform"]
    sample_rate = max(1, int(result.get("sample_rate", 32000) or 32000))
    start = max(0, int(round(float(frame_index) * sample_rate / float(fps))))
    end = max(start, int(round(float(frame_index + 1) * sample_rate / float(fps))))
    start = min(start, int(waveform.shape[-1]))
    end = min(end, int(waveform.shape[-1]))
    result["waveform"] = torch.cat(
        (waveform[..., :start], waveform[..., end:]), dim=-1
    )
    return result


def _concat_videos_incoming_overlap(paths, output, overlap_frames: int, fps: float = 24.0):
    """TimelineDirector-style assembly: incoming segment owns the overlap.

    Every continuation segment contains [incoming overlap + unique frames].
    Therefore each non-final segment loses its LAST overlap, while the complete
    incoming segment is retained.  No dissolve and no duplicated timeline.
    """
    from .iamccs_minimax_h3_shotboard import (
        _find_ffmpeg, _read_segment_frame_count, _require_identical_segment_canvas,
    )

    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg non trovato: impossibile fare incoming-overlap assembly")
    if len(paths) < 2:
        from .iamccs_minimax_h3_shotboard import _concat_videos
        return _concat_videos(paths, output)

    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Segmenti mancanti: " + ", ".join(missing))

    _require_identical_segment_canvas(paths)
    counts = [_read_segment_frame_count(path) for path in paths]
    overlap = max(1, int(overlap_frames))
    if any(count <= overlap for count in counts[:-1]):
        raise ValueError(
            f"Incoming overlap {overlap}f non valido per segmenti {counts}"
        )

    keep_counts = [
        count - overlap if index < len(counts) - 1 else count
        for index, count in enumerate(counts)
    ]
    output_frames = sum(keep_counts)
    exact_duration = output_frames / float(fps)

    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n"]
    for path in paths:
        command += ["-i", str(path)]

    filters = []
    video_labels = []
    audio_labels = []
    for index, keep in enumerate(keep_counts):
        vlabel = f"vin{index}"
        alabel = f"ain{index}"
        duration = keep / float(fps)
        filters.append(
            f"[{index}:v]trim=start_frame=0:end_frame={keep},"
            f"setpts=PTS-STARTPTS,format=yuv420p[{vlabel}]"
        )
        filters.append(
            f"[{index}:a]aresample=48000:async=1:first_pts=0,"
            f"aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"atrim=duration={duration:.9f},asetpts=PTS-STARTPTS[{alabel}]"
        )
        video_labels.append(vlabel)
        audio_labels.append(alabel)

    filters.append(
        "".join(f"[{label}]" for label in video_labels)
        + f"concat=n={len(video_labels)}:v=1:a=0[vjoined]"
    )
    filters.append(
        "".join(f"[{label}]" for label in audio_labels)
        + f"concat=n={len(audio_labels)}:v=0:a=1[ajoined]"
    )

    command += [
        "-filter_complex", ";".join(filters),
        "-map", "[vjoined]", "-map", "[ajoined]",
        "-frames:v", str(output_frames),
        "-r", f"{float(fps):.6f}",
        "-fps_mode", "cfr",
        "-c:v", "libx264", "-preset", "medium", "-crf", "16",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-t", f"{exact_duration:.9f}",
        "-movflags", "+faststart",
        str(output),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, stdin=subprocess.DEVNULL
    )
    if result.returncode != 0:
        raise RuntimeError(
            "incoming-overlap ffmpeg concat failed: "
            + (result.stderr.strip() or result.stdout.strip())
        )
    LOG.info(
        "R38B incoming-overlap concat complete | chunks=%d | overlap=%df | "
        "segments=%s | kept=%s | output_frames=%d | fps=%.3f",
        len(paths), overlap, counts, keep_counts, output_frames, float(fps),
    )


def _rtx_finish_segment(intermediate, output, audio, trim, frames, crop_width, crop_height,
                        width, height, fps, quality):
    """Final RTX pass from a file, one frame at a time; no UHD IMAGE batch."""
    import av
    import comfy.model_management as mm
    from .iamccs_rtx_vfx import _import_video_super_res
    from .iamccs_minimax_h3_shotboard import _find_ffmpeg, _write_wav

    ffmpeg = _find_ffmpeg()
    if not ffmpeg or output.exists():
        raise ValueError("RTX delivery needs ffmpeg and a new output filename.")
    VideoSuperRes = _import_video_super_res()
    with tempfile.TemporaryDirectory(prefix="iamccs_r38b_rtx_") as temp, tempfile.TemporaryFile() as errors:
        wav = Path(temp) / "native_audio.wav"
        if not _write_wav(audio, wav):
            raise ValueError("RTX delivery requires native audio.")
        command = [ffmpeg,"-hide_banner","-loglevel","error","-nostdin","-n",
            "-f","rawvideo","-pix_fmt","rgb24","-s",f"{width}x{height}","-r",str(fps),"-i","pipe:0",
            "-i",str(wav),"-map","0:v:0","-map","1:a:0","-frames:v",str(frames),
            "-af",f"apad,atrim=duration={frames/fps:.9f},asetpts=PTS-STARTPTS",
            "-c:v","libx264","-preset","medium","-crf","16","-pix_fmt","yuv420p",
            "-c:a","aac","-b:a","192k","-movflags","+faststart",str(output)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
        written = 0
        try:
            with VideoSuperRes(quality=getattr(VideoSuperRes.QualityLevel, quality), device=0) as effect, av.open(str(intermediate)) as video:
                effect.output_width, effect.output_height = _grid_up(width, 8), _grid_up(height, 8)
                effect.load()
                for index, frame in enumerate(video.decode(video=0)):
                    mm.throw_exception_if_processing_interrupted()
                    if index < trim:
                        continue
                    if written >= frames:
                        break
                    px = torch.from_numpy(frame.to_ndarray(format="rgb24"))
                    top, left = (px.shape[0]-crop_height)//2, (px.shape[1]-crop_width)//2
                    px = px[top:top+crop_height, left:left+crop_width]
                    rgb = px.to(device="cuda:0",dtype=torch.float32).permute(2,0,1).contiguous().div_(255)
                    result = torch.from_dlpack(effect.run(rgb).image).permute(1,2,0)
                    top, left = (result.shape[0]-height)//2, (result.shape[1]-width)//2
                    encoded = result[top:top+height,left:left+width].clamp(0,1).mul(255).round().to(device="cpu",dtype=torch.uint8).contiguous()
                    process.stdin.write(encoded.numpy().tobytes())
                    del px, rgb, result, encoded
                    written += 1
                    if written % 24 == 0 or written == frames:
                        LOG.info("R38B final RTX | %d/%d | %dx%d", written,frames,width,height)
            process.stdin.close()
            code = process.wait()
            if code or written != frames:
                errors.seek(0)
                raise RuntimeError(f"R38B RTX delivery wrote {written}/{frames} frames: " + errors.read().decode("utf8","replace")[-1500:])
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()


def _next_prompt(prompt, render_id, next_segment):
    result = copy.deepcopy(prompt)
    controllers = {"IAMCCS_MiniMaxH3AtomicModelRouter", "IAMCCS_MiniMaxH3AtomicConditioningBackend",
                   "IAMCCS_MiniMaxH3DirectorFLFParityModelRouter", "IAMCCS_MiniMaxH3DirectorFLFParityConditioning"}
    advanced = 0
    for node in result.values():
        inputs = node.setdefault("inputs", {})
        if node.get("class_type") in controllers or ("segment_index" in inputs and not isinstance(inputs["segment_index"], list)):
            inputs["segment_index"] = next_segment
            advanced += 1
        if "render_id" in inputs and not isinstance(inputs["render_id"], list):
            inputs["render_id"] = render_id
        if node.get("class_type") in {"IAMCCS_MiniMaxH3NativeCheckpointSave", "IAMCCS_MiniMaxH3AtomicConditioningBackend"}:
            inputs["render_id"] = render_id
    if not advanced:
        raise ValueError("R38B cannot find the native segment controllers; no next job was queued.")
    return result


class IAMCCS_MiniMaxH3PixelRefineR38B:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",), "video_vae": ("VAE",), "sampled_latent": ("LATENT",),
                "conditioning": ("CONDITIONING",), "cine_linx": (SUPERNODE_LINX_TYPE,),
                "native_frames": ("IMAGE",), "native_audio": ("AUDIO",),
                "resolved_render_id": ("STRING", {"forceInput": True}),
                "native_saved_report": ("STRING", {"forceInput": True}),
                "current_segment": ("INT", {"forceInput": True}), "total_segments": ("INT", {"forceInput": True}),
                "context_trim_frames": ("INT", {"forceInput": True}), "join_trim_frames": ("INT", {"forceInput": True}),
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

    def finish(self, model, video_vae, sampled_latent, conditioning, cine_linx, native_frames,
               native_audio, resolved_render_id, native_saved_report, current_segment, total_segments,
               context_trim_frames, join_trim_frames, queue_next_segment=True, prompt=None, extra_pnginfo=None):
        import comfy.model_management as mm
        from .iamccs_minimax_h3_shotboard import (
            _concat_videos, _concat_videos_overlap, _current_prompt, _encode_images, _enqueue,
            _trim_audio_frames, _write_segment_metadata,
        )

        plan = _resolve_shotplan(cine_linx)
        enabled = bool(plan.get("upscale_enabled")) and plan.get("upscale_mode", "off") != "off"
        if enabled and plan["upscale_mode"] != "h3_pixel_refine":
            raise ValueError("R38B wires Native / RTX → H3 only. Choose h3_pixel_refine or Off; existing other delivery workflows remain separate.")
        if plan.get("rife_mode", "off") != "off":
            raise ValueError("R38B preserves native 24 fps; RIFE is not wired in this variant.")
        if plan.get("face_detailer_settings", {}).get("enabled", False) and not sampled_latent.get("iamccs_r38b_face_applied"):
            raise ValueError("Face Detailer is enabled but not wired. Use the R38B FACE WINDOWS variant, or disable Face.")
        if not native_saved_report:
            raise ValueError("R38B must run after the native checkpoint and R37 latent commit.")
        index, total = int(current_segment), int(total_segments)
        if index < 0 or index >= total:
            raise ValueError("R38B received an invalid segment index.")
        run = _safe_render_id(resolved_render_id)

        # IAMCCS LongVid Latent Tail A/B v1.4 R38B cache owner.
        # IAMCCS_PHASE2_V4_EDITORIAL_HANDOFF
        # Cache the tail ending at the SAME editorial endpoint delivered to film.
        tail_cfg = (
            plan.get("longvid_latent_tail")
            if isinstance(plan.get("longvid_latent_tail"), dict)
            else {}
        )
        if bool(tail_cfg.get("enabled", False)) and index < total - 1:
            from .iamccs_minimax_h3_continuity import save_longvid_sampled_latent_tail
            chunks = plan.get("chunks") if isinstance(plan.get("chunks"), list) else []
            if index >= len(chunks) or not isinstance(chunks[index], dict):
                raise RuntimeError(
                    "LONGVID LATENT TAIL Phase 2 v4 cannot resolve current chunk plan."
                )
            handoff_chunk = chunks[index]
            tail_frames = max(5, int(tail_cfg.get("tail_frames", 22) or 22))
            unique_frames = max(1, int(handoff_chunk.get("unique_frames", 0) or 0))
            # seg1 has no incoming overlap; continuation segments deliver
            # [incoming overlap + unique] after the 1f RGB bridge is removed.
            handoff_endpoint_frames = int(unique_frames + (tail_frames if index > 0 else 0))
            saved_tail = save_longvid_sampled_latent_tail(
                run,
                index,
                sampled_latent,
                tail_frames=tail_frames,
                endpoint_frames=handoff_endpoint_frames,
            )
            if not saved_tail:
                raise RuntimeError(
                    "LONGVID LATENT TAIL Phase 2 v4 could not persist editorial-aligned tail."
                )
            LOG.info(
                "MiniMax H3 Latent Tail editorial handoff saved | "
                "render=%s | segment=%d/%d | tail=%df | unique=%df | "
                "endpoint=%df | rule=editorial_endpoint",
                run, index + 1, total, tail_frames, unique_frames, handoff_endpoint_frames,
            )

        root = Path(folder_paths.get_output_directory()) / "IAMCCS" / "MiniMaxH3" / "R38B" / run
        root.mkdir(parents=True, exist_ok=True)
        output = root / f"segment_{index + 1:04d}.mp4"
        if output.exists():
            raise FileExistsError(f"R38B segment already exists: {output}. Start a new render rather than overwriting it.")
        visible = int(native_frames.shape[0])
        tail_cfg = plan.get("longvid_latent_tail")
        latent_tail_segment = bool(
            index > 0
            and isinstance(tail_cfg, dict)
            and bool(tail_cfg.get("enabled", False))
        )
        latent_tail_overlap = (
            max(0, int(tail_cfg.get("tail_frames", 0) or 0))
            if latent_tail_segment else 0
        )
        if latent_tail_segment:
            if enabled:
                raise ValueError(
                    "LongVid latent-tail incoming-overlap assembly is currently "
                    "validated only at native delivery. Disable upscale for this test."
                )
            chunk = (plan.get("chunks") or [])[index]
            unique_frames = max(1, int(chunk.get("unique_frames", 0) or 0))
            expected = latent_tail_overlap + 1 + unique_frames
            if visible < expected:
                raise ValueError(
                    f"Latent-tail segment {index + 1} decoded {visible}f, "
                    f"needs at least {expected}f "
                    f"({latent_tail_overlap} overlap + 1 bridge + {unique_frames} unique)."
                )
            # Sample layout is:
            # [carried overlap][1f RGB bridge][unique continuation][grid padding].
            # Keep the real incoming overlap, remove only the duplicate RGB bridge.
            native_frames = native_frames[:expected, ...]
            bridge_index = latent_tail_overlap
            native_frames = torch.cat(
                (
                    native_frames[:bridge_index, ...],
                    native_frames[bridge_index + 1:, ...],
                ),
                dim=0,
            )
            audio = _remove_audio_frame_slice(native_audio, bridge_index, 24.0)
            frames = int(native_frames.shape[0])
            join = 0
            LOG.info(
                "R38B latent-tail segment prepared | segment=%d/%d | "
                "overlap=%df | bridge_removed_at=%df | unique=%df | saved=%df",
                index + 1, total, latent_tail_overlap, bridge_index,
                unique_frames, frames,
            )
        else:
            from .iamccs_minimax_h3_shotboard import _delivery_join_frames
            join = _delivery_join_frames(cine_linx, index, join_trim_frames)
            frames = visible - (1 if join == 1 else 0)
            audio = _trim_audio_frames(native_audio, 1, 24) if join == 1 else native_audio
        if enabled:
            settings = _upres_settings(plan)
            width, height = _delivery_size(plan)
            final_rtx = bool(settings.get("rtx_requested", False))
            crop_width, crop_height = (width // 2, height // 2) if final_rtx else (width, height)
            stage_width, stage_height = _grid_up(crop_width), _grid_up(crop_height)
            if width % 2 or height % 2:
                raise ValueError("R38B H.264 delivery width and height must be even.")
            if int(settings.get("window_overlap", 22)) >= int(settings.get("window_frames", 136)):
                raise ValueError("R38B window overlap must be smaller than its window.")
            common = _provider("common")
            video, _ = common.unpack_av(sampled_latent, "R38B Stage 1")
            source_frames = common.latents_to_frames(int(video.shape[2]))
            trim = int(context_trim_frames)
            # Visible Stage-1 pixels already have context removed. Never infer a
            # new audio offset from upscale windows or re-trim their AUDIO.
            if source_frames < trim + visible:
                raise ValueError(f"R38B latent has {source_frames} frames, needs {trim}+{visible}.")
            LOG.info("R38B Stage 2 AFTER native save | segment=%d/%d | source=%dx%d | target=%dx%d | native steps=%s | native turbo=%s | windows=%s/%s | audio=locked",
                     index + 1, total, video.shape[4] * 16, video.shape[3] * 16, width, height,
                     plan.get("sampling", {}).get("steps"), plan.get("turbo", {}),
                     settings.get("window_frames", 136), settings.get("window_overlap", 22))
            try:
                mm.unload_all_models()
                mm.soft_empty_cache()
                upscaled = _pixel_upscale(sampled_latent, video_vae, stage_width, stage_height, settings)
                mm.unload_all_models()
                mm.soft_empty_cache()
                refined = _refine(model, conditioning, upscaled, cine_linx, index)
                del upscaled
                mm.unload_all_models()
                mm.soft_empty_cache()
                intermediate = _provider("nodes_save").MMH3StreamingSave.execute(
                    latent=refined, vae=video_vae, groups_per_chunk=int(settings.get("pixel_groups", 1)),
                    fps=24.0, filename_prefix=f"IAMCCS/MiniMaxH3/R38B/{run}/stage2_{index + 1:04d}",
                    crf=16, save_metadata=False,
                )[0]
                del refined
                if final_rtx:
                    mm.unload_all_models()
                    mm.soft_empty_cache()
                    _rtx_finish_segment(intermediate, output, audio, trim + (1 if join == 1 else 0), frames,
                                        crop_width,crop_height,width,height,24,settings.get("rtx_quality","ULTRA"))
                else:
                    _finish_segment(intermediate, output, audio, trim + (1 if join == 1 else 0), frames, width, height, 24)
            finally:
                mm.unload_all_models()
                gc.collect()
                mm.soft_empty_cache()
        else:
            _encode_images(native_frames[1:] if join == 1 else native_frames, audio, 24, output)
        _write_segment_metadata(output, frames, 24,
                                "trim_silent_tail" if native_audio.get("iamccs_flf_locked_audio_handles", False) else "crossfade")
        preview = output
        if index == total - 1:
            paths = [root / f"segment_{i + 1:04d}.mp4" for i in range(total)]
            preview = root / "final_film.mp4"
            if preview.exists():
                raise FileExistsError(f"R38B final film already exists: {preview}")
            final_tail_cfg = plan.get("longvid_latent_tail")
            if (
                isinstance(final_tail_cfg, dict)
                and bool(final_tail_cfg.get("enabled", False))
                and int(final_tail_cfg.get("tail_frames", 0) or 0) > 0
            ):
                _concat_videos_incoming_overlap(
                    paths,
                    preview,
                    int(final_tail_cfg.get("tail_frames", 0) or 0),
                    24.0,
                )
            elif join > 1:
                _concat_videos_overlap(paths, preview, join, 24)
            else:
                _concat_videos(paths, preview)
        queued = False
        if bool(queue_next_segment) and index + 1 < total:
            live, extra, outputs, sensitive = _current_prompt()
            next_prompt = _next_prompt(live if live is not None else prompt, run, index + 1)
            _enqueue(next_prompt, extra_data=extra, outputs=outputs, sensitive=sensitive)
            queued = True
        report = f"R38B saved {index + 1}/{total} | {frames}f | native audio preserved | next queued={queued} | {preview}"
        LOG.info(report)
        return {"ui": {"text": [report], "images": [{"filename": preview.name,
                "subfolder": preview.parent.relative_to(Path(folder_paths.get_output_directory())).as_posix(), "type": "output"}],
                "animated": (True,)}, "result": (str(preview), report)}


NODE_CLASS_MAPPINGS = {"IAMCCS_MiniMaxH3PixelRefineR38B": IAMCCS_MiniMaxH3PixelRefineR38B}
NODE_DISPLAY_NAME_MAPPINGS = {"IAMCCS_MiniMaxH3PixelRefineR38B": "MiniMax H3 · Safe Windowed Pixel → H3 Delivery"}

from .iamccs_minimax_h3_face_delivery_variant import (
    IAMCCS_MiniMaxH3FaceDeliveryR38B,
    IAMCCS_MiniMaxH3WideCharacterDetailer12GB,
)
NODE_CLASS_MAPPINGS["IAMCCS_MiniMaxH3FaceDeliveryR38B"] = IAMCCS_MiniMaxH3FaceDeliveryR38B
NODE_DISPLAY_NAME_MAPPINGS["IAMCCS_MiniMaxH3FaceDeliveryR38B"] = "MiniMax H3 · Optional FaceRefine · Before Upscale"
NODE_CLASS_MAPPINGS["IAMCCS_MiniMaxH3WideCharacterDetailer12GB"] = IAMCCS_MiniMaxH3WideCharacterDetailer12GB
NODE_DISPLAY_NAME_MAPPINGS["IAMCCS_MiniMaxH3WideCharacterDetailer12GB"] = "IAMCCS WIDE CHARACTER DETAILER · 12 GB"
