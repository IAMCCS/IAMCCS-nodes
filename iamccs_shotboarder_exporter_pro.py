from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch

from .iamccs_rtx_vfx import (
    RTX_COMMON_RATIOS,
    RTX_DIVISIBLE_BY_VALUES,
    RTX_QUALITY_LEVELS,
    RTX_RESIZE_METHODS,
    RTX_RESIZE_TYPES,
    apply_rtx_vfx,
)


_log = logging.getLogger("IAMCCS.ShotboarderExporterPRO")
NODE_ID = "IAMCCS_shotboarder_aud+vid_exporter_PRO"


VIDEO_PROFILES: Dict[str, Dict[str, Any]] = {
    "h264_mp4": {
        "label": "H.264 / MP4 / delivery",
        "extension": "mp4",
        "input_depth": 8,
        "pix_fmt": "yuv420p",
        "video_args": ["-c:v", "libx264", "-preset", "medium", "-profile:v", "high", "-pix_fmt", "yuv420p"],
        "quality": True,
        "faststart": True,
        "lossless": False,
    },
    "h265_mp4": {
        "label": "H.265 / HEVC / MP4 / 10-bit",
        "extension": "mp4",
        "input_depth": 16,
        "pix_fmt": "yuv420p10le",
        "video_args": ["-c:v", "libx265", "-preset", "medium", "-tag:v", "hvc1", "-pix_fmt", "yuv420p10le"],
        "quality": True,
        "faststart": True,
        "lossless": False,
    },
    "prores_422_mov": {
        "label": "Apple ProRes 422 / MOV",
        "extension": "mov",
        "input_depth": 16,
        "pix_fmt": "yuv422p10le",
        "video_args": ["-c:v", "prores_ks", "-profile:v", "2", "-pix_fmt", "yuv422p10le"],
        "quality": False,
        "faststart": False,
        "lossless": False,
    },
    "prores_422_hq_mov": {
        "label": "Apple ProRes 422 HQ / MOV",
        "extension": "mov",
        "input_depth": 16,
        "pix_fmt": "yuv422p10le",
        "video_args": ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"],
        "quality": False,
        "faststart": False,
        "lossless": False,
    },
    "prores_4444_mov": {
        "label": "Apple ProRes 4444 / MOV",
        "extension": "mov",
        "input_depth": 16,
        "pix_fmt": "yuv444p10le",
        "video_args": ["-c:v", "prores_ks", "-profile:v", "4"],
        "quality": False,
        "faststart": False,
        "lossless": False,
        "alpha": True,
    },
    "dnxhr_hqx_mov": {
        "label": "Avid DNxHR HQX / MOV / 10-bit",
        "extension": "mov",
        "input_depth": 16,
        "pix_fmt": "yuv422p10le",
        "video_args": ["-c:v", "dnxhd", "-profile:v", "dnxhr_hqx", "-pix_fmt", "yuv422p10le"],
        "quality": False,
        "faststart": False,
        "lossless": False,
    },
    "v210_mov": {
        "label": "v210 10-bit 4:2:2 / MOV",
        "extension": "mov",
        "input_depth": 16,
        "pix_fmt": "yuv422p10le",
        "video_args": ["-c:v", "v210", "-pix_fmt", "yuv422p10le"],
        "quality": False,
        "faststart": False,
        "lossless": False,
    },
    "ffv1_mkv": {
        "label": "FFV1 lossless / MKV / archive",
        "extension": "mkv",
        "input_depth": 16,
        "pix_fmt": "gbrp16le",
        "video_args": [
            "-c:v", "ffv1", "-level", "3", "-coder", "1", "-context", "1",
            "-g", "1", "-slices", "16", "-slicecrc", "1", "-pix_fmt", "gbrp16le",
        ],
        "quality": False,
        "faststart": False,
        "lossless": True,
    },
}


AUDIO_PROFILES: Dict[str, Dict[str, Any]] = {
    "copy_source": {"label": "Source audio / direct copy", "codec": "copy", "args": ["-c:a", "copy"], "lossless": True, "direct": True},
    "aac_320": {"label": "AAC 320 kb/s", "codec": "aac", "args": ["-c:a", "aac", "-b:a", "320k"], "lossless": False},
    "aac_192": {"label": "AAC 192 kb/s", "codec": "aac", "args": ["-c:a", "aac", "-b:a", "192k"], "lossless": False},
    "pcm_s16le": {"label": "PCM signed 16-bit", "codec": "pcm_s16le", "args": ["-c:a", "pcm_s16le"], "lossless": True},
    "pcm_s24le": {"label": "PCM signed 24-bit", "codec": "pcm_s24le", "args": ["-c:a", "pcm_s24le"], "lossless": True},
    "pcm_s32le": {"label": "PCM signed 32-bit", "codec": "pcm_s32le", "args": ["-c:a", "pcm_s32le"], "lossless": True},
    "flac": {"label": "FLAC lossless", "codec": "flac", "args": ["-c:a", "flac", "-compression_level", "8"], "lossless": True},
    "alac": {"label": "Apple Lossless / ALAC", "codec": "alac", "args": ["-c:a", "alac"], "lossless": True},
}


def _nested_master_audio_candidates(cine_linx: Any) -> Iterable[Dict[str, Any]]:
    if not isinstance(cine_linx, dict):
        return []
    resources = cine_linx.get("resources") if isinstance(cine_linx.get("resources"), dict) else {}
    outputs = cine_linx.get("outputs") if isinstance(cine_linx.get("outputs"), dict) else {}
    payload = cine_linx.get("payload") if isinstance(cine_linx.get("payload"), dict) else {}
    containers = (cine_linx, resources, outputs, payload)
    candidates = []
    for container in containers:
        for key in (
            "master_audio_asset",
            "masterAudioAsset",
            "cine_audio_master_audio_asset",
            "master_audio",
            "master_excerpt",
        ):
            value = container.get(key)
            if isinstance(value, dict):
                candidates.append(value)
        multi = container.get("multiGeneration")
        if isinstance(multi, dict):
            for key in ("master_audio_asset", "masterAudioAsset", "master_audio", "masterExcerpt"):
                value = multi.get(key)
                if isinstance(value, dict):
                    candidates.append(value)
        manifest = container.get("manifest")
        if isinstance(manifest, dict):
            for key in ("master_audio_asset", "masterAudioAsset", "master_audio", "master_excerpt"):
                value = manifest.get(key)
                if isinstance(value, dict):
                    candidates.append(value)
    return candidates


def _resolve_master_audio_file(cine_linx: Any, override: str = "") -> Tuple[Path | None, Dict[str, Any]]:
    import folder_paths  # type: ignore

    override_text = str(override or "").strip()
    candidates = [{"audioFile": override_text}] if override_text else list(_nested_master_audio_candidates(cine_linx))
    roots = []
    for getter_name in ("get_input_directory", "get_output_directory", "get_temp_directory"):
        getter = getattr(folder_paths, getter_name, None)
        if callable(getter):
            try:
                root = str(getter())
                if root and root not in roots:
                    roots.append(root)
            except Exception:
                pass
    for item in candidates:
        if not isinstance(item, dict):
            continue
        raw = str(
            item.get("path")
            or item.get("audioFile")
            or item.get("audio_file")
            or item.get("filename")
            or item.get("fileName")
            or ""
        ).strip()
        if not raw:
            continue
        direct = Path(raw)
        if direct.is_absolute() and direct.exists():
            return direct.resolve(), item
        subfolder = str(item.get("subfolder") or item.get("audioSubfolder") or "").strip().replace("\\", "/")
        filename = raw.replace("\\", "/")
        if "/" in filename and not subfolder:
            parts = [part for part in filename.split("/") if part]
            filename = parts.pop() if parts else filename
            subfolder = "/".join(parts)
        upload_type = str(item.get("audioUploadType") or item.get("type") or item.get("file_type") or "input").strip().lower()
        ordered_roots = list(roots)
        if upload_type == "output" and len(roots) > 1:
            ordered_roots = [roots[1], *[root for root in roots if root != roots[1]]]
        for root in ordered_roots:
            candidate = Path(root) / subfolder / filename
            try:
                if Path(os.path.abspath(candidate)).is_file() and os.path.commonpath((os.path.abspath(root), os.path.abspath(candidate))) == os.path.abspath(root):
                    return candidate.resolve(), item
            except Exception:
                if candidate.is_file():
                    return candidate.resolve(), item
    return None, {}


def _master_audio_trim(asset: Dict[str, Any], fps: float, target_seconds: float) -> Tuple[float, float]:
    if not isinstance(asset, dict):
        return 0.0, 0.0
    trim_start = float(asset.get("renderTrimStartSeconds") or asset.get("render_trim_start_seconds") or 0.0)
    trim_end = float(asset.get("renderTrimEndSeconds") or asset.get("render_trim_end_seconds") or 0.0)
    if trim_start <= 0.0:
        pre_roll_frames = max(0, int(round(float(asset.get("preRollFrames") or asset.get("pre_roll_frames") or 0))))
        trim_start = pre_roll_frames / max(1.0, float(fps))
    if trim_end <= trim_start:
        nominal_frames = max(0, int(round(float(asset.get("nominalDurationFrames") or asset.get("nominal_duration_frames") or 0))))
        nominal_seconds = float(asset.get("nominalDurationSeconds") or asset.get("nominal_duration_seconds") or 0.0)
        nominal_seconds = nominal_seconds or (nominal_frames / max(1.0, float(fps)))
        if nominal_seconds > 0:
            trim_end = trim_start + nominal_seconds
    if trim_end <= trim_start and target_seconds > 0:
        trim_end = trim_start + target_seconds
    return max(0.0, trim_start), max(0.0, trim_end)


def _finite_seconds(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _editor_master_audio_edl(editor_audio_edl_json: Any, fps: float, target_seconds: float) -> Tuple[list[Dict[str, Any]], str]:
    """Return master-audio windows in the exact order assembled by the editor.

    The parked video take carries its own physical pre/post-roll window.  When a
    user reveals one of those handles, the rendered video is no longer aligned
    to one continuous span of the master file.  The compact editor handoff
    preserves each video's source trim so the exporter can rebuild that audio
    edit decision list without changing the editor renderer itself.
    """
    raw = str(editor_audio_edl_json or "").strip()
    if not raw:
        return [], "not_provided"
    try:
        payload = json.loads(raw)
    except Exception:
        return [], "invalid_json"
    if not isinstance(payload, dict):
        return [], "invalid_payload"
    clips = payload.get("clips")
    if not isinstance(clips, list):
        return [], "missing_clips"

    video_clips = [
        clip for clip in clips
        if isinstance(clip, dict) and str(clip.get("type") or "").lower() == "video"
    ]
    if not video_clips:
        return [], "no_video_clips"

    # Every standard take has a sequential Txx identity.  A manual/unmapped
    # video has no deterministic position in the AudioBoard master, so keep
    # the established direct-master fallback in that case.
    take_nominal: Dict[int, float] = {}
    for clip in video_clips:
        take_index = int(round(_finite_seconds(clip.get("takeIndex"), 0.0)))
        timeline_id = str(clip.get("timelineId") or "").strip().upper()
        if take_index < 1 or not timeline_id.startswith("T"):
            return [], "contains_unmapped_video"
        nominal = _finite_seconds(clip.get("nominalDuration"), 0.0)
        if nominal <= 0:
            nominal_frames = _finite_seconds(clip.get("nominalDurationFrames"), 0.0)
            nominal = nominal_frames / max(1.0, fps)
        if nominal <= 0:
            return [], "missing_nominal_duration"
        take_nominal[take_index] = max(take_nominal.get(take_index, 0.0), nominal)

    ordered_takes = sorted(take_nominal)
    if ordered_takes != list(range(1, ordered_takes[-1] + 1)):
        return [], "non_contiguous_take_map"

    nominal_start: Dict[int, float] = {}
    cursor = 0.0
    for take_index in ordered_takes:
        nominal_start[take_index] = cursor
        cursor += take_nominal[take_index]

    ordered_clips = sorted(
        video_clips,
        key=lambda clip: (
            _finite_seconds(clip.get("startTime"), 0.0),
            int(round(_finite_seconds(clip.get("takeIndex"), 0.0))),
            str(clip.get("id") or ""),
        ),
    )
    windows: list[Dict[str, Any]] = []
    for clip in ordered_clips:
        take_index = int(round(_finite_seconds(clip.get("takeIndex"), 0.0)))
        trim_start = max(0.0, _finite_seconds(clip.get("trimStart"), 0.0))
        trim_end = _finite_seconds(clip.get("trimEnd"), 0.0)
        if trim_end <= trim_start:
            trim_end = trim_start + max(0.0, _finite_seconds(clip.get("duration"), 0.0))
        if trim_end <= trim_start:
            return [], "invalid_trim_window"
        pre_roll = max(
            0.0,
            _finite_seconds(clip.get("preRoll"), _finite_seconds(clip.get("preRollFrames"), 0.0) / max(1.0, fps)),
        )
        physical_master_start = max(0.0, nominal_start[take_index] - pre_roll)
        windows.append({
            "clip_id": str(clip.get("id") or f"T{take_index:02d}"),
            "take_index": take_index,
            "start": physical_master_start + trim_start,
            "end": physical_master_start + trim_end,
            # These are local parked-take frames, matching the Video Editor
            # renderer's int(round(seconds * fps)) trim contract exactly.
            "input_frame_start": int(round(trim_start * max(1.0, fps))),
            "input_frame_end": int(round(trim_end * max(1.0, fps))),
        })

    window_duration = sum(max(0.0, item["end"] - item["start"]) for item in windows)
    target = max(0.0, _finite_seconds(target_seconds, 0.0))
    tolerance = 0.5 / max(1.0, fps)
    if not windows or abs(window_duration - target) > tolerance:
        return [], "window_duration_mismatch"
    return windows, "ok"


def _subtract_covered_window(start: float, end: float, covered: list[Tuple[float, float]]) -> list[Tuple[float, float]]:
    """Return the part of a master window not already emitted by an earlier take."""
    remaining: list[Tuple[float, float]] = []
    cursor = float(start)
    for covered_start, covered_end in covered:
        if covered_end <= cursor:
            continue
        if covered_start >= end:
            break
        if covered_start > cursor:
            remaining.append((cursor, min(end, covered_start)))
        cursor = max(cursor, covered_end)
        if cursor >= end:
            break
    if cursor < end:
        remaining.append((cursor, end))
    return [(part_start, part_end) for part_start, part_end in remaining if part_end > part_start]


def _add_covered_window(covered: list[Tuple[float, float]], start: float, end: float) -> list[Tuple[float, float]]:
    """Keep source coverage normalized so post/pre-roll overlaps are emitted once."""
    merged: list[Tuple[float, float]] = []
    pending_start, pending_end = float(start), float(end)
    for covered_start, covered_end in sorted([*covered, (pending_start, pending_end)]):
        if not merged or covered_start > merged[-1][1]:
            merged.append((covered_start, covered_end))
        else:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, covered_end))
    return merged


def _deduplicate_editor_roll_frames(images: Any, windows: list[Dict[str, Any]], fps: float) -> Tuple[Any, list[Dict[str, Any]], bool, str]:
    """Remove repeated roll frames from the sequential editor-render payload.

    The editor renderer emits parked clips one after another. If T01 exposes
    post-roll while T02 still begins at its nominal source point, that payload
    contains the same master interval twice. Keep the first occurrence, trim
    the duplicate frames from later takes, and retain matching master windows
    for the exporter audio path.
    """
    if not torch.is_tensor(images) or images.ndim < 1:
        return images, windows, False, "invalid_frame_payload"
    if not windows:
        return images, windows, False, "no_editor_windows"

    input_frames = int(images.shape[0])
    expected_frames = sum(
        max(0, int(item.get("input_frame_end", 0)) - int(item.get("input_frame_start", 0)))
        for item in windows
    )
    if expected_frames != input_frames:
        return images, windows, False, "frame_contract_mismatch"

    covered: list[Tuple[float, float]] = []
    selected_slices: list[Any] = []
    selected_windows: list[Dict[str, Any]] = []
    input_cursor = 0
    duplicate_frames = 0
    for item in windows:
        frame_start = int(item.get("input_frame_start", 0))
        frame_end = int(item.get("input_frame_end", frame_start))
        item_frames = max(0, frame_end - frame_start)
        source_start = float(item["start"])
        source_end = float(item["end"])
        uncovered = _subtract_covered_window(source_start, source_end, covered)
        for keep_start, keep_end in uncovered:
            local_start = int(round((keep_start - source_start) * max(1.0, fps)))
            local_end = int(round((keep_end - source_start) * max(1.0, fps)))
            local_start = max(0, min(item_frames, local_start))
            local_end = max(local_start, min(item_frames, local_end))
            if local_end <= local_start:
                continue
            selected_slices.append(images[input_cursor + local_start: input_cursor + local_end])
            selected_windows.append({
                **item,
                "start": keep_start,
                "end": keep_end,
                "input_frame_start": frame_start + local_start,
                "input_frame_end": frame_start + local_end,
            })
        kept_frames = sum(
            max(0, int(round((keep_end - keep_start) * max(1.0, fps))))
            for keep_start, keep_end in uncovered
        )
        duplicate_frames += max(0, item_frames - kept_frames)
        covered = _add_covered_window(covered, source_start, source_end)
        input_cursor += item_frames

    if input_cursor != input_frames or not selected_slices:
        return images, windows, False, "slice_contract_mismatch"
    if duplicate_frames <= 0:
        return images, windows, False, "no_roll_overlap"
    output = torch.cat(selected_slices, dim=0)
    return output, selected_windows, True, f"deduplicated_{duplicate_frames}_frames"


def _edl_requires_audio_assembly(windows: list[Dict[str, Any]], fps: float, target_seconds: float) -> bool:
    """A continuous 0..duration master can retain the old direct-copy path."""
    if not windows:
        return False
    tolerance = 0.5 / max(1.0, fps)
    if abs(float(windows[0]["start"])) > tolerance:
        return True
    cursor = float(windows[0]["end"])
    for item in windows[1:]:
        if abs(float(item["start"]) - cursor) > tolerance:
            return True
        cursor = float(item["end"])
    return abs(cursor - max(0.0, float(target_seconds))) > tolerance


def _edl_filtergraph(windows: list[Dict[str, Any]], target_seconds: float, sync_mode: str) -> str:
    """Build sample-accurate master windows for FFmpeg's audio filter graph."""
    chains = []
    labels = []
    for index, item in enumerate(windows):
        label = f"aedl{index}"
        labels.append(f"[{label}]")
        chains.append(
            f"[1:a:0]atrim=start={float(item['start']):.09f}:end={float(item['end']):.09f},"
            f"asetpts=PTS-STARTPTS[{label}]"
        )
    if len(labels) == 1:
        chains.append(f"{labels[0]}anull[aedl]")
    else:
        chains.append(f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[aedl]")
    if str(sync_mode or "trim_to_video") == "trim_to_video":
        chains.append(
            f"[aedl]apad=whole_dur={max(0.0, float(target_seconds)):.09f},"
            f"atrim=duration={max(0.0, float(target_seconds)):.09f}[aout]"
        )
    else:
        chains.append("[aedl]anull[aout]")
    return ";".join(chains)


def _find_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("FFmpeg non trovato nel PATH e imageio-ffmpeg non e disponibile.") from exc


def _as_audio_waveform(audio: Any) -> Tuple[torch.Tensor, int]:
    current = audio
    for _ in range(5):
        if isinstance(current, dict) and "waveform" in current:
            waveform = current.get("waveform")
            sample_rate = int(current.get("sample_rate", 0) or 0)
            break
        if isinstance(current, dict) and "audio" in current:
            current = current.get("audio")
            continue
        if isinstance(current, (list, tuple)) and current:
            current = current[0]
            continue
        waveform = getattr(current, "waveform", None)
        sample_rate = int(getattr(current, "sample_rate", 0) or 0)
        break
    else:
        waveform, sample_rate = None, 0

    if waveform is None or sample_rate <= 0:
        raise ValueError("L'input audio non contiene un waveform/sample_rate valido.")
    if not torch.is_tensor(waveform):
        waveform = torch.as_tensor(waveform)
    waveform = waveform.detach().to(device="cpu", dtype=torch.float32)
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2 or waveform.shape[0] <= 0 or waveform.shape[1] <= 0:
        raise ValueError(f"Waveform non valido: shape={tuple(waveform.shape)}")
    return waveform.clamp(-1.0, 1.0).contiguous(), sample_rate


def _frame_payload_info(images: Any, profile: Dict[str, Any]) -> Tuple[torch.Tensor, str, int, int, int]:
    if not torch.is_tensor(images):
        images = torch.as_tensor(images)
    images = images.detach()
    if images.ndim == 5 and images.shape[0] == 1:
        images = images[0]
    if images.ndim != 4:
        raise ValueError(f"VIDEO images non valido: shape={tuple(images.shape)}")
    frames, height, width, channels = [int(v) for v in images.shape]
    if frames <= 0 or height <= 0 or width <= 0:
        raise ValueError("Il VIDEO non contiene frame esportabili.")
    if channels == 1:
        images = images.repeat(1, 1, 1, 3)
        channels = 3
    elif channels == 2:
        images = torch.cat((images[..., :1], images[..., :1], images[..., 1:2]), dim=-1)
        channels = 3
    elif channels > 4:
        images = images[..., :4]
        channels = 4

    wants_alpha = bool(profile.get("alpha")) and channels == 4
    keep_channels = 4 if wants_alpha else 3
    use_16bit = int(profile.get("input_depth", 8)) >= 16
    input_pix_fmt = ("rgba64le" if keep_channels == 4 else "rgb48le") if use_16bit else ("rgba" if keep_channels == 4 else "rgb24")

    # Chroma-subsampled delivery codecs require even dimensions. VHS applies
    # replication padding for the same reason. A one-pixel edge extension is
    # preferable to failing after an expensive RTX pass.
    width = width + (width % 2)
    height = height + (height % 2)
    return images, input_pix_fmt, frames, width, height


def _iter_frame_payload(
    images: torch.Tensor,
    profile: Dict[str, Any],
    width: int,
    height: int,
) -> Iterable[bytes]:
    """Convert one frame at a time, keeping peak host memory bounded."""
    channels = int(images.shape[-1])
    wants_alpha = bool(profile.get("alpha")) and channels == 4
    keep_channels = 4 if wants_alpha else 3
    use_16bit = int(profile.get("input_depth", 8)) >= 16
    target_dtype = np.uint16 if use_16bit else np.uint8
    target_max = 65535.0 if use_16bit else 255.0

    for source_frame in images:
        frame = source_frame.detach().to(device="cpu", dtype=torch.float32)
        if channels == 1:
            frame = frame.repeat(1, 1, 3)
        elif channels == 2:
            frame = torch.cat((frame[..., :1], frame[..., :1], frame[..., 1:2]), dim=-1)
        frame = frame[..., :keep_channels].clamp(0.0, 1.0).numpy()
        pad_height = int(height) - int(frame.shape[0])
        pad_width = int(width) - int(frame.shape[1])
        if pad_height or pad_width:
            frame = np.pad(frame, ((0, pad_height), (0, pad_width), (0, 0)), mode="edge")
        yield np.rint(frame * target_max).astype(target_dtype, copy=False).tobytes(order="C")


def _rtx_frame_tensor(images: Any) -> torch.Tensor:
    """Return exporter frames in Comfy IMAGE layout without duplicating the batch."""
    if not torch.is_tensor(images):
        images = torch.as_tensor(images)
    frames = images.detach()
    if frames.ndim == 5 and frames.shape[0] == 1:
        frames = frames[0]
    if frames.ndim != 4:
        raise ValueError(f"RTX frame output requires IMAGE [frames,height,width,channels], got {tuple(frames.shape)}")
    channels = int(frames.shape[-1])
    if channels == 1:
        frames = frames.repeat(1, 1, 1, 3)
    elif channels == 2:
        frames = torch.cat((frames[..., :1], frames[..., :1], frames[..., 1:2]), dim=-1)
    elif channels > 3:
        frames = frames[..., :3]
    # Normal IMAGE/RTX output is already float RGB in [0,1]. Do not call
    # .to(cpu), clamp(), clone(), or contiguous() here: each would allocate a
    # second full-resolution video batch after FFmpeg has already completed.
    return frames


def _output_path(filename_prefix: str, extension: str, width: int, height: int) -> Path:
    import folder_paths  # type: ignore

    prefix = str(filename_prefix or "IAMCCS/Shotboarder_PRO").strip().replace("\\", "/")
    for suffix in (".mp4", ".mov", ".mkv"):
        if prefix.lower().endswith(suffix):
            prefix = prefix[: -len(suffix)]
            break
    folder, filename, counter, _subfolder, _resolved = folder_paths.get_save_image_path(
        prefix, folder_paths.get_output_directory(), width, height
    )
    return Path(folder) / f"{filename}_{counter:05d}.{extension}"


def _output_preview_descriptor(path: Path, fps: float, frames: int, width: int, height: int) -> Dict[str, Any]:
    import folder_paths  # type: ignore

    output_root = Path(folder_paths.get_output_directory()).resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(output_root)
    except ValueError:
        relative = Path(resolved.name)
    extension = resolved.suffix.lower()
    media_format = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
    }.get(extension, "video/*")
    return {
        "filename": relative.name,
        "subfolder": relative.parent.as_posix() if str(relative.parent) != "." else "",
        "type": "output",
        "format": media_format,
        "frame_rate": float(fps),
        "frame_count": int(frames),
        "width": int(width),
        "height": int(height),
        "has_audio": True,
    }


def _validate_profiles(video_key: str, audio_key: str) -> None:
    video = VIDEO_PROFILES[video_key]
    audio = AUDIO_PROFILES[audio_key]
    extension = video["extension"]
    codec = audio["codec"]
    if codec == "copy":
        return
    if extension == "mp4" and codec not in {"aac", "alac"}:
        raise ValueError("PCM/FLAC non e compatibile con il profilo MP4 selezionato. Usa MOV o MKV per audio lossless.")
    if extension == "mov" and codec == "flac":
        raise ValueError("FLAC non e un profilo audio MOV affidabile. Usa PCM/ALAC o il profilo MKV lossless.")


def _align_audio(waveform: torch.Tensor, sample_rate: int, frames: int, fps: float, policy: str) -> torch.Tensor:
    if policy != "trim_to_video":
        return waveform
    target = max(1, int(round(float(frames) / max(1.0, float(fps)) * sample_rate)))
    if waveform.shape[1] > target:
        return waveform[:, :target].contiguous()
    if waveform.shape[1] < target:
        padding = torch.zeros((waveform.shape[0], target - waveform.shape[1]), dtype=waveform.dtype)
        return torch.cat((waveform, padding), dim=1).contiguous()
    return waveform


def _rtx_choice(value: Any, choices: Iterable[str], fallback: str) -> str:
    """Keep old serialized exporter widgets from failing Comfy validation."""
    candidate = str(value or "").strip()
    return candidate if candidate in choices else str(fallback)


class IAMCCS_ShotboarderAudVidExporterPRO:
    """Professional video/audio exporter for Shotboard and AudioBoard masters."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "audio_source_mode": (["audio_input", "master_file_direct"], {"default": "audio_input"}),
                "filename_prefix": ("STRING", {"default": "IAMCCS/Shotboarder_PRO"}),
                "video_profile": (list(VIDEO_PROFILES.keys()), {"default": "prores_422_hq_mov"}),
                "audio_profile": (list(AUDIO_PROFILES.keys()), {"default": "pcm_s16le"}),
                "audio_sync": (["trim_to_video", "shortest"], {"default": "trim_to_video"}),
                "frame_rate_override": ("FLOAT", {"default": 24.0, "min": 0.0, "max": 240.0, "step": 0.01}),
                "video_quality": ("INT", {"default": 18, "min": 0, "max": 51, "step": 1}),
                "embed_metadata": ("BOOLEAN", {"default": True}),
                "write_sidecar": ("BOOLEAN", {"default": True}),
                "rtx_enabled": ("BOOLEAN", {"default": False}),
                # These fields are custom-UI selects. Keep their transport type
                # as STRING so workflows saved before the RTX panel existed do
                # not map legacy metadata_json (often "{}") into an enum and
                # fail before this node gets a chance to normalize it.
                "rtx_mode": ("STRING", {"default": "VSR Medium"}),
                "rtx_resize_type": ("STRING", {"default": "Keep Ratio"}),
                "rtx_scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.05}),
                "rtx_megapixels": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 64.0, "step": 0.01}),
                "rtx_width": ("INT", {"default": 1920, "min": 64, "max": 8192, "step": 8}),
                "rtx_height": ("INT", {"default": 1080, "min": 64, "max": 8192, "step": 8}),
                "rtx_divisible_by": ("STRING", {"default": "8"}),
                "rtx_device": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1}),
                "rtx_ratio_preset": ("STRING", {"default": "16:9"}),
                "rtx_resize_method": ("STRING", {"default": "Center Crop (Fill)"}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "cine_linx": ("IAMCCS_SUPERNODE_LINX",),
                "master_audio_file": ("STRING", {"default": "", "multiline": False}),
                "metadata_json": ("STRING", {"default": "{}", "multiline": True}),
                # Compact video-clip trim handoff from the Video Editor. This
                # is intentionally optional: normal direct exports remain
                # fully compatible with existing workflows.
                "editor_audio_edl_json": ("STRING", {"default": "", "multiline": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    # Keep the original three output slots stable. The appended IMAGE output
    # exposes the exact frame sequence used by the exporter, after optional
    # native RTX processing.
    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("video", "output_path", "report", "rtx_frames")
    FUNCTION = "export"
    OUTPUT_NODE = True
    CATEGORY = "IAMCCS/Cine/Export"

    def export(
        self,
        video,
        audio_source_mode,
        filename_prefix,
        video_profile,
        audio_profile,
        audio_sync,
        frame_rate_override,
        video_quality,
        embed_metadata,
        write_sidecar,
        rtx_enabled=False,
        rtx_mode="VSR Medium",
        rtx_resize_type="Keep Ratio",
        rtx_scale=2.0,
        rtx_megapixels=2.0,
        rtx_width=1920,
        rtx_height=1080,
        rtx_divisible_by="1",
        rtx_device=0,
        rtx_ratio_preset="16:9",
        rtx_resize_method="Center Crop (Fill)",
        audio=None,
        cine_linx=None,
        master_audio_file="",
        metadata_json="{}",
        editor_audio_edl_json="",
        prompt=None,
        extra_pnginfo=None,
    ):
        video_key = str(video_profile or "prores_422_hq_mov")
        source_mode = str(audio_source_mode or "audio_input")
        if source_mode not in {"audio_input", "master_file_direct"}:
            source_mode = "audio_input"
        audio_key = str(audio_profile or "pcm_s16le")
        if video_key not in VIDEO_PROFILES:
            video_key = "prores_422_hq_mov"
        if audio_key not in AUDIO_PROFILES:
            audio_key = "pcm_s16le"
        _validate_profiles(video_key, audio_key)

        profile = VIDEO_PROFILES[video_key]
        audio_config = AUDIO_PROFILES[audio_key]
        components = video.get_components()
        fps = float(frame_rate_override or 0.0) or float(components.frame_rate or 24.0)
        fps = max(1.0, min(240.0, fps))
        rtx_mode = _rtx_choice(rtx_mode, RTX_QUALITY_LEVELS, "VSR Medium")
        rtx_resize_type = _rtx_choice(rtx_resize_type, RTX_RESIZE_TYPES, "Keep Ratio")
        rtx_divisible_by = _rtx_choice(rtx_divisible_by, RTX_DIVISIBLE_BY_VALUES, "1")
        rtx_ratio_preset = _rtx_choice(rtx_ratio_preset, RTX_COMMON_RATIOS, "16:9")
        rtx_resize_method = _rtx_choice(rtx_resize_method, RTX_RESIZE_METHODS, "Center Crop (Fill)")
        source_images = components.images
        direct_audio_edl: list[Dict[str, Any]] = []
        direct_audio_edl_status = "not_applicable"
        roll_visual_dedup_active = False
        roll_visual_dedup_status = "not_applicable"
        if source_mode == "master_file_direct":
            raw_frame_count = int(source_images.shape[0]) if torch.is_tensor(source_images) and source_images.ndim >= 1 else 0
            direct_audio_edl, direct_audio_edl_status = _editor_master_audio_edl(
                editor_audio_edl_json,
                fps,
                raw_frame_count / max(1.0, fps),
            )
            if direct_audio_edl:
                source_images, direct_audio_edl, roll_visual_dedup_active, roll_visual_dedup_status = _deduplicate_editor_roll_frames(
                    source_images,
                    direct_audio_edl,
                    fps,
                )
        rtx_active = bool(rtx_enabled)
        if rtx_active:
            export_images = apply_rtx_vfx(
                source_images,
                mode=str(rtx_mode or "VSR Medium"),
                resize_type=str(rtx_resize_type or "Keep Ratio"),
                scale=float(rtx_scale or 2.0),
                megapixels=float(rtx_megapixels or 2.0),
                width=int(rtx_width or 1920),
                height=int(rtx_height or 1080),
                divisible_by=str(rtx_divisible_by or "1"),
                device=int(rtx_device or 0),
                ratio_preset=str(rtx_ratio_preset or "16:9"),
                resize_method=str(rtx_resize_method or "Center Crop (Fill)"),
            )
        else:
            export_images = source_images
        export_images, input_pix_fmt, frame_count, width, height = _frame_payload_info(export_images, profile)
        direct_path = None
        direct_asset: Dict[str, Any] = {}
        direct_trim_start = 0.0
        direct_trim_end = 0.0
        direct_audio_edl_active = False
        effective_audio_profile = audio_key
        effective_audio_lossless = bool(audio_config.get("lossless"))
        if source_mode == "master_file_direct":
            direct_path, direct_asset = _resolve_master_audio_file(cine_linx, master_audio_file)
            if direct_path is None:
                raise ValueError(
                    f"{NODE_ID}: master_file_direct selected but no readable master audio file was found. "
                    "Connect AudioBoard cine_linx with master_audio_asset or provide master_audio_file."
                )
            direct_trim_start, direct_trim_end = _master_audio_trim(direct_asset, fps, frame_count / max(1.0, fps))
            direct_audio_edl_active = bool(direct_audio_edl) and _edl_requires_audio_assembly(
                direct_audio_edl,
                fps,
                frame_count / max(1.0, fps),
            )
            waveform = None
            sample_rate = int(direct_asset.get("sampleRate") or direct_asset.get("sample_rate") or 0)
        else:
            if audio is None:
                raise ValueError(f"{NODE_ID}: audio_input selected but no AUDIO input is connected.")
            waveform, sample_rate = _as_audio_waveform(audio)
            waveform = _align_audio(waveform, sample_rate, frame_count, fps, str(audio_sync or "trim_to_video"))
        audio_channels = int(waveform.shape[0]) if waveform is not None else int(direct_asset.get("channels") or direct_asset.get("channel_count") or 0)
        audio_samples = int(waveform.shape[1]) if waveform is not None else 0
        output_path = _output_path(filename_prefix, profile["extension"], width, height)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = _find_ffmpeg()

        metadata: Dict[str, Any] = {
            "node": NODE_ID,
            "video_profile": video_key,
            "video_profile_label": profile["label"],
            "audio_profile": audio_key,
            "audio_profile_label": audio_config["label"],
            "audio_profile_effective": effective_audio_profile,
            "fps": fps,
            "frames": frame_count,
            "width": width,
            "height": height,
            "audio_sample_rate": sample_rate,
            "audio_channels": audio_channels,
            "audio_samples": audio_samples,
            "audio_sync": str(audio_sync or "trim_to_video"),
            "video_lossless_codec": bool(profile.get("lossless")),
            "audio_lossless_codec": bool(audio_config.get("lossless")),
            "audio_source_mode": source_mode,
            "master_audio_file": str(direct_path or ""),
            "master_audio_source_asset": direct_asset if direct_asset else {},
            "master_audio_trim_start_seconds": direct_trim_start,
            "master_audio_trim_end_seconds": direct_trim_end,
            "master_audio_edl_status": direct_audio_edl_status,
            "master_audio_edl_active": direct_audio_edl_active,
            "master_audio_edl_windows": direct_audio_edl,
            "roll_visual_dedup_active": roll_visual_dedup_active,
            "roll_visual_dedup_status": roll_visual_dedup_status,
            "rtx_enabled": rtx_active,
            "rtx_mode": str(rtx_mode or "VSR Medium"),
            "rtx_resize_type": str(rtx_resize_type or "Keep Ratio"),
            "rtx_scale": float(rtx_scale or 2.0),
            "rtx_megapixels": float(rtx_megapixels or 2.0),
            "rtx_width": int(rtx_width or 1920),
            "rtx_height": int(rtx_height or 1080),
            "rtx_divisible_by": str(rtx_divisible_by or "1"),
            "rtx_device": int(rtx_device or 0),
            "rtx_ratio_preset": str(rtx_ratio_preset or "16:9"),
            "rtx_resize_method": str(rtx_resize_method or "Center Crop (Fill)"),
        }
        if isinstance(cine_linx, dict):
            metadata["cine_linx_type"] = str(cine_linx.get("type", ""))
            metadata["cine_linx_mode"] = str(cine_linx.get("mode", ""))
        try:
            supplied_metadata = json.loads(str(metadata_json or "{}"))
            if isinstance(supplied_metadata, dict):
                metadata["user_metadata"] = supplied_metadata
        except Exception:
            metadata["user_metadata_text"] = str(metadata_json or "")
        if isinstance(extra_pnginfo, dict):
            metadata["extra_pnginfo_keys"] = sorted(str(key) for key in extra_pnginfo.keys())
        metadata["prompt_present"] = bool(prompt)

        with tempfile.TemporaryDirectory(prefix="iamccs_shotboarder_export_") as temp_dir:
            audio_path = Path(temp_dir) / "audio.f32le"
            if waveform is not None:
                audio_interleaved = waveform.transpose(0, 1).numpy().astype(np.float32, copy=False)
                audio_path.write_bytes(audio_interleaved.tobytes(order="C"))

            video_args = list(profile["video_args"])
            if profile.get("quality"):
                video_args += ["-crf", str(max(0, min(51, int(video_quality))))]
            if video_key == "prores_4444_mov" and input_pix_fmt == "rgba64le":
                video_args += ["-pix_fmt", "yuva444p10le"]
            elif video_key == "ffv1_mkv" and input_pix_fmt == "rgba64le":
                video_args += ["-pix_fmt", "rgba64le"]
            if profile.get("faststart"):
                video_args += ["-movflags", "+faststart"]

            metadata_args = []
            if bool(embed_metadata):
                metadata_args = [
                    "-metadata", "title=IAMCCS Shotboarder PRO export",
                    "-metadata", f"comment={NODE_ID} | {video_key} | {audio_key}",
                ]
            if source_mode == "master_file_direct":
                audio_filter_args = []
                if direct_audio_edl_active:
                    # A filter graph cannot stream-copy audio. AudioBoard
                    # masters are PCM WAV files, so PCM16 keeps the existing
                    # source samples lossless while allowing exact roll cuts.
                    audio_input = ["-i", str(direct_path)]
                    audio_filter_args = [
                        "-filter_complex",
                        _edl_filtergraph(direct_audio_edl, frame_count / max(1.0, fps), str(audio_sync or "trim_to_video")),
                    ]
                    audio_map = ["-map", "[aout]"]
                else:
                    audio_input = []
                    if direct_trim_start > 0.000001:
                        audio_input += ["-ss", f"{direct_trim_start:.09f}"]
                    audio_input += ["-i", str(direct_path)]
                    if direct_trim_end > direct_trim_start + 0.000001:
                        audio_input += ["-t", f"{direct_trim_end - direct_trim_start:.09f}"]
                    audio_map = ["-map", "1:a:0"]
                audio_codec_args = list(audio_config["args"])
                if direct_audio_edl_active and audio_key == "copy_source":
                    if str(profile.get("extension") or "").lower() in {"mov", "mkv"}:
                        audio_codec_args = ["-c:a", "pcm_s16le"]
                        effective_audio_profile = "pcm_s16le"
                        effective_audio_lossless = True
                    else:
                        # MP4 cannot hold PCM reliably across FFmpeg builds;
                        # retain the user-selected delivery container and make
                        # the unavoidable codec conversion explicit in report.
                        audio_codec_args = ["-c:a", "aac", "-b:a", "320k"]
                        effective_audio_profile = "aac_320"
                        effective_audio_lossless = False
                elif audio_key == "copy_source":
                    audio_codec_args = ["-c:a", "copy"]
                audio_input_format = audio_input
            else:
                audio_input_format = [
                    "-f", "f32le", "-ar", str(sample_rate), "-ac", str(int(waveform.shape[0])), "-i", str(audio_path),
                ]
                audio_map = ["-map", "1:a:0"]
                audio_codec_args = list(audio_config["args"])
            command = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", input_pix_fmt,
                "-s:v", f"{width}x{height}", "-r", f"{fps:.09f}", "-i", "pipe:0",
                *audio_input_format,
                *(audio_filter_args if source_mode == "master_file_direct" else []),
                "-map", "0:v:0", *audio_map,
                *video_args,
                *audio_codec_args,
                *metadata_args,
            ]
            if source_mode == "master_file_direct" and not direct_audio_edl_active and direct_trim_end <= direct_trim_start + 0.000001 and str(audio_sync or "trim_to_video") == "trim_to_video":
                command += ["-t", f"{frame_count / max(1.0, fps):.09f}"]
            if str(audio_sync or "trim_to_video") == "shortest":
                command += ["-shortest"]
            command += [str(output_path)]
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            write_error = None
            try:
                from comfy.utils import ProgressBar  # type: ignore
                progress = ProgressBar(frame_count)
            except Exception:
                progress = None
            try:
                assert process.stdin is not None
                for index, frame_data in enumerate(_iter_frame_payload(export_images, profile, width, height), start=1):
                    process.stdin.write(frame_data)
                    if progress is not None:
                        progress.update_absolute(index, frame_count)
            except (BrokenPipeError, OSError) as exc:
                write_error = exc
            finally:
                if process.stdin is not None:
                    try:
                        process.stdin.close()
                    except OSError:
                        pass
            detail = process.stderr.read().decode("utf-8", errors="replace") if process.stderr is not None else ""
            return_code = process.wait()
            if return_code != 0 or write_error is not None:
                try:
                    output_path.unlink(missing_ok=True)
                except Exception:
                    pass
                detail = (detail or str(write_error or "FFmpeg failed")).strip()
                raise RuntimeError(f"{NODE_ID}: FFmpeg export failed: {detail[-4000:]}")

        metadata["audio_profile_effective"] = effective_audio_profile
        metadata["audio_lossless_codec"] = effective_audio_lossless

        sidecar_path = output_path.with_suffix(".metadata.json")
        if bool(write_sidecar):
            sidecar_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        preview = _output_preview_descriptor(output_path, fps, frame_count, width, height)
        report = json.dumps({
            "node": NODE_ID,
            "output_path": str(output_path),
            "sidecar_path": str(sidecar_path) if bool(write_sidecar) else "",
            "video_profile": video_key,
            "audio_profile": audio_key,
            "audio_profile_effective": effective_audio_profile,
            "audio_edl_active": direct_audio_edl_active,
            "audio_edl_status": direct_audio_edl_status,
            "visual_roll_dedup_active": roll_visual_dedup_active,
            "visual_roll_dedup_status": roll_visual_dedup_status,
            "codec_contract": f"{profile['video_args']} + {audio_config['args']}",
            "video_lossless": bool(profile.get("lossless")),
            "audio_lossless": effective_audio_lossless,
            "duration_seconds": round(frame_count / fps, 6),
            "preview": preview,
            "truth": "One explicit audio source is muxed once. When a roll edit creates non-contiguous master windows, the exporter rebuilds that exact audio EDL before muxing.",
        }, ensure_ascii=False, indent=2)
        _log.info("[%s] wrote %s", NODE_ID, output_path)
        rtx_frames = _rtx_frame_tensor(export_images)
        return {
            "ui": {"gifs": [preview], "iamccs_exporter_preview": [preview]},
            "result": (video, str(output_path), report, rtx_frames),
        }


NODE_CLASS_MAPPINGS = {NODE_ID: IAMCCS_ShotboarderAudVidExporterPRO}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_ID: "IAMCCS Shotboarder Aud+Vid Exporter PRO"}
