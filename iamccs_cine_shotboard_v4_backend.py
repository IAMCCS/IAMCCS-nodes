from __future__ import annotations

import base64
import copy
import io
import json
import math
import os
import re
from typing import Any, Dict, List, Tuple

import folder_paths
import numpy as np
import nodes as comfy_nodes
import torch
import torch.nn.functional as F
from PIL import Image

from .iamccs_cine_nodes import (
    IAMCCS_CineFilmmakerBackend,
    _load_original_promptrelay_module,
    _round_ltx_frames,
    _cine_guide_strength,
    _safe_bool,
    _safe_float,
    _safe_int,
)


SUPERNODE_LINX_TYPE = "IAMCCS_SUPERNODE_LINX"
MOTION_GUIDE_TYPE = "MOTION_GUIDE_DATA"


def _json_loads(value: Any, fallback: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        text = str(value or "").strip()
        if not text:
            return fallback
        return json.loads(text)
    except Exception:
        return fallback


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _tag_timeline_source(timeline: Dict[str, Any], source: str) -> Dict[str, Any]:
    if not isinstance(timeline, dict):
        return {}
    tagged = dict(timeline)
    tagged["_iamccs_timeline_source"] = str(source or "unknown")
    return tagged


def _retake_active(timeline: Dict[str, Any]) -> bool:
    return _safe_bool(timeline.get("retakeMode", timeline.get("retake_mode", False)), False)


def _retake_video(timeline: Dict[str, Any]) -> Dict[str, Any]:
    value = timeline.get("retakeVideo", timeline.get("retake_video"))
    return value if isinstance(value, dict) else {}


def _normalize_resize_method(value: Any) -> str:
    method = str(value or "").strip().lower()
    aliases = {
        "keep proportion": "maintain aspect ratio",
        "keep_proportion": "maintain aspect ratio",
        "maintain_aspect_ratio": "maintain aspect ratio",
        "stretch": "stretch to fit",
        "stretch_to_fit": "stretch to fit",
        "center crop": "crop",
        "center_crop": "crop",
    }
    method = aliases.get(method, method)
    if method not in {"maintain aspect ratio", "stretch to fit", "pad", "pad green", "crop"}:
        return "crop"
    return method


def _input_root() -> str:
    try:
        return folder_paths.get_input_directory()
    except Exception:
        return os.getcwd()


def _resolve_media_path(path_value: Any) -> str:
    raw = str(path_value or "").strip()
    if not raw:
        return ""
    if os.path.isabs(raw) and os.path.exists(raw):
        return raw
    try:
        annotated = folder_paths.get_annotated_filepath(raw)
        if annotated and os.path.exists(annotated):
            return annotated
    except Exception:
        pass
    root = _input_root()
    candidate = os.path.abspath(os.path.join(root, raw))
    try:
        inside_root = os.path.commonpath([os.path.abspath(root), candidate]) == os.path.abspath(root)
    except ValueError:
        inside_root = False
    if inside_root and os.path.exists(candidate):
        return candidate
    basename = os.path.basename(raw)
    if not basename:
        return ""
    try:
        for current, _dirs, files in os.walk(root):
            if basename in files:
                return os.path.join(current, basename)
    except Exception:
        pass
    return ""


def _segment_media_path(seg: Dict[str, Any], prefer_video: bool = False) -> str:
    keys = (
        ("videoFile", "video_file", "sourceVideoFile", "movieFile", "imageFile", "image_file", "imageTruthPath", "path")
        if prefer_video
        else ("imageFile", "image_file", "imageTruthPath", "path", "videoFile", "video_file")
    )
    for key in keys:
        value = str(seg.get(key) or "").strip()
        if value:
            return value
    return ""


def _load_image_tensor(seg: Dict[str, Any]) -> torch.Tensor:
    file_ref = _segment_media_path(seg, prefer_video=False)
    file_path = _resolve_media_path(file_ref)
    if file_path:
        try:
            img = Image.open(file_path).convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).unsqueeze(0)
        except Exception:
            pass

    encoded = str(seg.get("imageB64", seg.get("image_b64", "")) or "")
    if encoded and not encoded.startswith("/view?"):
        if "," in encoded:
            encoded = encoded.split(",", 1)[1]
        try:
            img = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).unsqueeze(0)
        except Exception:
            pass
    return torch.zeros((1, 512, 512, 3), dtype=torch.float32)


def _video_dimensions(path_ref: Any) -> Tuple[int, int]:
    file_path = _resolve_media_path(path_ref)
    if not file_path:
        return 0, 0
    try:
        import av

        with av.open(file_path) as container:
            stream = container.streams.video[0]
            return int(stream.width or stream.codec_context.width or 0), int(stream.height or stream.codec_context.height or 0)
    except Exception:
        return 0, 0


def _load_video_frames(video_ref: Any, trim_start_frames: float, length_frames: float, frame_rate: float, resample_mode: str = "nearest") -> torch.Tensor:
    file_path = _resolve_media_path(video_ref)
    if not file_path:
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)

    try:
        import av
    except Exception:
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)

    target_fps = max(1.0, float(frame_rate or 24.0))
    start_sec = max(0.0, float(trim_start_frames or 0.0) / target_fps)
    length_frames = max(1.0, float(length_frames or 1.0))
    end_sec = start_sec + length_frames / target_fps
    frames: List[np.ndarray] = []
    source_fps = target_fps

    try:
        with av.open(file_path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            try:
                source_fps = float(stream.average_rate) if stream.average_rate else float(stream.base_rate)
            except Exception:
                source_fps = target_fps
            if source_fps <= 0:
                source_fps = target_fps
            if start_sec > 0:
                try:
                    seek_pts = int(max(0.0, start_sec - 0.5) / float(stream.time_base)) if stream.time_base else int(max(0.0, start_sec - 0.5) * av.time_base)
                    container.seek(seek_pts, stream=stream, backward=True)
                except Exception:
                    pass
            decoded = 0
            for frame in container.decode(stream):
                if frame.time is not None:
                    timestamp = float(frame.time)
                elif frame.pts is not None and stream.time_base is not None:
                    timestamp = float(frame.pts * stream.time_base)
                else:
                    timestamp = float(decoded / max(0.001, source_fps))
                decoded += 1
                if timestamp < start_sec - 0.01:
                    continue
                if timestamp >= end_sec:
                    break
                frames.append(frame.to_ndarray(format="rgb24"))
    except Exception:
        frames = []

    if not frames:
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)

    images = torch.from_numpy(np.asarray(frames, dtype=np.float32) / 255.0)
    target_count = max(1, int(round(length_frames)))
    return _resample_frames(images, target_count, resample_mode, source_fps=source_fps, target_fps=target_fps)


def _resample_frames(frames: torch.Tensor, target_count: int, mode: str, source_fps: float = 1.0, target_fps: float = 1.0) -> torch.Tensor:
    n = int(frames.shape[0])
    target_count = max(1, int(target_count))
    if n <= 0:
        raise ValueError("frame batch is empty")
    if n == target_count and abs(float(source_fps) - float(target_fps)) < 1e-6:
        return frames
    if n == 1:
        return frames.repeat(target_count, 1, 1, 1)
    positions = torch.linspace(0, n - 1, target_count, device=frames.device, dtype=torch.float32)
    if str(mode or "nearest") == "nearest":
        indices = torch.round(positions).long().clamp(0, n - 1)
        return frames.index_select(0, indices)
    low = torch.floor(positions).long().clamp(0, n - 1)
    high = torch.ceil(positions).long().clamp(0, n - 1)
    alpha = (positions - low.to(positions.dtype)).view(-1, 1, 1, 1)
    return (frames.index_select(0, low).to(torch.float32) * (1.0 - alpha) + frames.index_select(0, high).to(torch.float32) * alpha).to(frames.dtype)


def _resize_image(tensor: torch.Tensor, target_w: int, target_h: int, method: str, divisible_by: int) -> torch.Tensor:
    def snap(value: int, divisor: int) -> int:
        divisor = max(1, int(divisor or 1))
        return max(divisor, (int(value) // divisor) * divisor)

    target_w = snap(max(1, int(target_w)), divisible_by)
    target_h = snap(max(1, int(target_h)), divisible_by)
    method = _normalize_resize_method(method)
    n, h, w, c = tensor.shape
    if h == target_h and w == target_w:
        return tensor
    nchw = tensor.permute(0, 3, 1, 2)
    if method == "stretch to fit":
        resized = F.interpolate(nchw, size=(target_h, target_w), mode="bilinear", align_corners=False)
    elif method == "maintain aspect ratio":
        ratio = min(target_w / max(1, w), target_h / max(1, h))
        new_w = snap(max(1, int(w * ratio)), divisible_by)
        new_h = snap(max(1, int(h * ratio)), divisible_by)
        resized = F.interpolate(nchw, size=(new_h, new_w), mode="bilinear", align_corners=False)
    elif method in {"pad", "pad green"}:
        ratio = min(target_w / max(1, w), target_h / max(1, h))
        new_w = snap(max(1, int(w * ratio)), divisible_by)
        new_h = snap(max(1, int(h * ratio)), divisible_by)
        inner = F.interpolate(nchw, size=(new_h, new_w), mode="bilinear", align_corners=False)
        pad_l = max(0, (target_w - new_w) // 2)
        pad_t = max(0, (target_h - new_h) // 2)
        if method == "pad green":
            resized = torch.zeros((n, c, target_h, target_w), dtype=nchw.dtype, device=nchw.device)
            resized[:, 0, :, :] = 102.0 / 255.0
            resized[:, 1, :, :] = 1.0
            resized[:, 2, :, :] = 0.0
            resized[:, :, pad_t:pad_t + new_h, pad_l:pad_l + new_w] = inner
        else:
            resized = F.pad(inner, (pad_l, target_w - new_w - pad_l, pad_t, target_h - new_h - pad_t), mode="constant", value=0)
    elif method == "crop":
        ratio = max(target_w / max(1, w), target_h / max(1, h))
        new_w = max(1, int(w * ratio))
        new_h = max(1, int(h * ratio))
        inner = F.interpolate(nchw, size=(new_h, new_w), mode="bilinear", align_corners=False)
        left = max(0, (new_w - target_w) // 2)
        top = max(0, (new_h - target_h) // 2)
        resized = inner[:, :, top:top + target_h, left:left + target_w]
    else:
        resized = F.interpolate(nchw, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return resized.permute(0, 2, 3, 1)


def _compress_video_like(tensor: torch.Tensor, crf: int) -> torch.Tensor:
    crf = int(crf or 0)
    if crf <= 0 or not torch.is_tensor(tensor) or tensor.ndim != 4:
        return tensor
    try:
        import av

        n, h, w, _c = tensor.shape
        h2 = (int(h) // 2) * 2
        w2 = (int(w) // 2) * 2
        if h2 <= 0 or w2 <= 0:
            return tensor
        raw = (tensor[:, :h2, :w2, :] * 255.0).clamp(0, 255).byte().cpu().numpy()
        buffer = io.BytesIO()
        container = av.open(buffer, mode="w", format="mp4")
        stream = container.add_stream("libx264", rate=24)
        stream.width = w2
        stream.height = h2
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(max(0, min(51, crf))), "preset": "ultrafast"}
        for index in range(n):
            frame = av.VideoFrame.from_ndarray(raw[index], format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        buffer.seek(0)
        decoded = []
        with av.open(buffer, mode="r") as decoded_container:
            for frame in decoded_container.decode(video=0):
                decoded.append(frame.to_ndarray(format="rgb24"))
        if not decoded:
            return tensor
        out = tensor.clone()
        decoded_np = np.stack(decoded).astype(np.float32) / 255.0
        count = min(int(n), len(decoded))
        out[:count, :h2, :w2, :] = torch.from_numpy(decoded_np[:count]).to(device=tensor.device, dtype=tensor.dtype)
        return out
    except Exception as exc:
        print(f"[IAMCCS Shotboard V4 Backend] guide compression skipped: {exc}")
        return tensor


def _snap_to_divisor(value: int, divisor: int) -> int:
    divisor = max(1, int(divisor or 1))
    return max(divisor, (int(value) // divisor) * divisor)


def _resize_to_policy(tensor: torch.Tensor, width: int, height: int, method: str, divisible_by: int) -> torch.Tensor:
    source_h = int(tensor.shape[1])
    source_w = int(tensor.shape[2])
    width = int(width or 0)
    height = int(height or 0)
    if width > 0 and height > 0:
        return _resize_image(tensor, width, height, method, divisible_by)
    if width > 0:
        target_w = _snap_to_divisor(width, divisible_by)
        target_h = _snap_to_divisor(int(source_h * target_w / max(1, source_w)), divisible_by)
        return _resize_image(tensor, target_w, target_h, "stretch to fit", divisible_by)
    if height > 0:
        target_h = _snap_to_divisor(height, divisible_by)
        target_w = _snap_to_divisor(int(source_w * target_h / max(1, source_h)), divisible_by)
        return _resize_image(tensor, target_w, target_h, "stretch to fit", divisible_by)
    return _resize_image(tensor, source_w, source_h, "maintain aspect ratio", divisible_by)


def _active_take_package(resources: Dict[str, Any]) -> Dict[str, Any]:
    for value in (
        resources.get("cine_take_router_package"),
        resources.get("cine_multigeneration_take_package"),
        resources.get("cine_take_package"),
        resources.get("cine_multigeneration_active_take"),
        _json_loads(resources.get("cine_take_router_package_json"), {}),
        _json_loads(resources.get("cine_multigeneration_take_package_json"), {}),
        _json_loads(resources.get("cine_take_package_json"), {}),
    ):
        if isinstance(value, dict) and (_safe_int(value.get("take_index"), 0) > 0 or str(value.get("timeline_id") or "").strip()):
            return value
    return {}


def _materialize_active_visual_timeline(timeline: Dict[str, Any], resources: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a bridge-selected visual timeline when a full TakeRouter is absent."""
    package = _active_take_package(resources)
    take_index = max(1, _safe_int(package.get("take_index"), 1))
    timeline_id = str(package.get("timeline_id") or f"T{take_index:02d}").strip()
    multi = timeline.get("multiGeneration") if isinstance(timeline.get("multiGeneration"), dict) else {}
    visual_timelines = multi.get("visualTimelines") if isinstance(multi.get("visualTimelines"), dict) else {}
    active_visual = visual_timelines.get(timeline_id)
    if not isinstance(active_visual, dict) and isinstance(package.get("visual_timeline"), dict):
        active_visual = package.get("visual_timeline")
    if not isinstance(active_visual, dict) and isinstance(package.get("visual_segments"), list):
        active_visual = {"segments": package.get("visual_segments", [])}
    if not isinstance(active_visual, dict):
        return timeline

    routed = copy.deepcopy(timeline)
    for key, value in active_visual.items():
        routed[key] = copy.deepcopy(value)
    visual_segments = active_visual.get("segments") if isinstance(active_visual.get("segments"), list) else []
    if visual_segments:
        routed["segments"] = copy.deepcopy(visual_segments)
        routed["shots"] = copy.deepcopy(active_visual.get("shots", visual_segments))
        routed["shotClips"] = copy.deepcopy(active_visual.get("shotClips", visual_segments))
    audio_segments = package.get("audio_segments")
    if not isinstance(audio_segments, list):
        audio_segments = package.get("audioSegments")
    if not isinstance(audio_segments, list):
        audio_segments = active_visual.get("audioSegments", active_visual.get("audioClips", []))
    if isinstance(audio_segments, list):
        routed["audioSegments"] = copy.deepcopy(audio_segments)
        routed["audioClips"] = copy.deepcopy(audio_segments)
        routed["use_custom_audio"] = bool(audio_segments)
    duration_seconds = _safe_float(
        active_visual.get("duration_seconds", active_visual.get("durationSeconds", active_visual.get("duration"))),
        _safe_float(package.get("duration_seconds"), _safe_float(timeline.get("duration_seconds"), 0.0)),
    )
    fps = _safe_float(
        active_visual.get("frame_rate", active_visual.get("frameRate", active_visual.get("fps"))),
        _safe_float(timeline.get("frame_rate"), 24.0),
    )
    if duration_seconds > 0:
        routed["duration_seconds"] = float(duration_seconds)
        routed["duration_frames"] = max(1, int(round(duration_seconds * max(1.0, fps))))
        routed["normalDurationFrames"] = int(routed["duration_frames"])
    if fps > 0:
        routed["frame_rate"] = float(fps)
    if str(active_visual.get("global_prompt", active_visual.get("prompt", "")) or "").strip():
        routed["global_prompt"] = str(active_visual.get("global_prompt", active_visual.get("prompt", "")) or "")
    routed["activeTimelineId"] = timeline_id
    routed["activeTake"] = int(take_index)
    routed["active_take"] = int(take_index)
    return routed


def _apply_routed_duration_truth(timeline: Dict[str, Any]) -> Dict[str, Any]:
    """Make TakeRouter seconds authoritative over stale planner frame fields."""
    routed = copy.deepcopy(timeline)
    duration_seconds = _safe_float(
        routed.get("duration_seconds", routed.get("durationSeconds", routed.get("duration", 0))),
        0.0,
    )
    fps = _safe_float(routed.get("frame_rate", routed.get("frameRate", routed.get("fps", 24))), 24.0)
    if duration_seconds > 0 and fps > 0:
        frames = max(1, int(round(duration_seconds * fps)))
        routed["duration_frames"] = frames
        routed["normalDurationFrames"] = frames
    return routed


def _find_timeline(cine_linx: Dict[str, Any], explicit_timeline_data: Any = "") -> Dict[str, Any]:
    resources = cine_linx.get("resources", {}) if isinstance(cine_linx, dict) else {}
    outputs = cine_linx.get("outputs", {}) if isinstance(cine_linx, dict) else {}
    payload = resources.get("cine_payload") if isinstance(resources.get("cine_payload"), dict) else {}
    explicit = _json_loads(explicit_timeline_data, None)
    routed_value = resources.get("cine_take_router_timeline_data")
    routed_candidate = _json_loads(routed_value, None)
    if isinstance(routed_candidate, dict):
        return _tag_timeline_source(
            _materialize_active_visual_timeline(_apply_routed_duration_truth(routed_candidate), resources),
            "cine_linx.resources.cine_take_router_timeline_data",
        )

    for source, value in (
        # Prefer the complete timeline contract so AudioBoard DSP metadata is
        # retained. The compact backend view remains a compatibility fallback.
        ("cine_linx.resources.cine_v4_timeline_data_json", resources.get("cine_v4_timeline_data_json")),
        ("cine_linx.resources.cine_payload.timeline_data", payload.get("timeline_data")),
        ("cine_linx.outputs.timeline_data", outputs.get("timeline_data")),
        ("cine_linx.resources.cine_board_timeline_data", resources.get("cine_board_timeline_data")),
        ("cine_linx.resources.cine_dialogue_shotboard_timeline_json", resources.get("cine_dialogue_shotboard_timeline_json")),
        ("cine_linx.resources.cine_backend_timeline_data_json", resources.get("cine_backend_timeline_data_json")),
        ("cine_linx.outputs.backend_timeline_data", outputs.get("backend_timeline_data")),
        ("cine_linx.resources.cine_payload.backend_timeline_data", payload.get("backend_timeline_data")),
        ("backend_widget.timeline_data", explicit),
    ):
        candidate = value if isinstance(value, dict) else _json_loads(value, None)
        if isinstance(candidate, dict):
            return _tag_timeline_source(_materialize_active_visual_timeline(candidate, resources), source)
    return _tag_timeline_source({}, "empty")


def _segments_from_timeline(timeline: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    value = timeline.get(key)
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    return []


def _derive_prompt_parts(
    timeline: Dict[str, Any],
    local_prompts: str,
    segment_lengths: str,
    duration_frames: int = 0,
) -> Tuple[str, str, str]:
    if str(local_prompts or "").strip() and str(segment_lengths or "").strip():
        guide_strength = str(timeline.get("guide_strength", timeline.get("guideStrength", "")) or "")
        return str(local_prompts), str(segment_lengths), guide_strength
    prompts = []
    lengths: List[int] = []
    strengths = []
    visual_segments = [
        seg for seg in _segments_from_timeline(timeline, "segments")
        if str(seg.get("type", "image") or "image").lower() != "audio"
        and not _safe_bool(seg.get("placeholder", False), False)
    ]
    visual_segments.sort(key=lambda item: _safe_int(item.get("start", item.get("frame", 0)), 0))
    target_frames = int(duration_frames or 0)
    if target_frames <= 0:
        target_frames = _safe_int(timeline.get("normalDurationFrames", timeline.get("duration_frames", 0)), 0)
    if target_frames <= 0:
        target_frames = max(
            1,
            int(round(
                _safe_float(timeline.get("duration_seconds"), 5.0)
                * _safe_float(timeline.get("frame_rate"), 24.0)
            )),
        )
    target_frames = max(1, target_frames)
    cursor = 0
    pending_gap = 0
    for seg in visual_segments:
        start = max(0, _safe_int(seg.get("start", seg.get("frame", 0)), 0))
        if start >= target_frames:
            break
        if start > cursor:
            gap = min(start, target_frames) - cursor
            if lengths:
                lengths[-1] += max(0, gap)
            else:
                pending_gap += max(0, gap)
        seg_len = max(1, _safe_int(seg.get("length", seg.get("len", max(1, round(_safe_float(timeline.get("frame_rate"), 24.0))))), 1))
        clipped_len = max(1, min(start + seg_len, target_frames) - start)
        total_len = max(1, clipped_len + pending_gap)
        pending_gap = 0
        cursor = start + seg_len
        prompt = str(seg.get("prompt", seg.get("local_prompt", seg.get("relay_prompt", ""))) or "").strip()
        if not _safe_bool(seg.get("use_prompt", True), True):
            prompt = ""
        prompts.append(prompt)
        lengths.append(int(total_len))
        strength = seg.get("guideStrength", seg.get("guide_strength", seg.get("strength", seg.get("force", None))))
        if strength is not None:
            strengths.append(str(float(_safe_float(strength, 1.0))))
    if lengths and cursor < target_frames:
        lengths[-1] = int(lengths[-1] + target_frames - min(cursor, target_frames))
    if not prompts:
        return "", "", ",".join(strengths)
    # Keep empty local slots: the global prompt remains Relay context and must
    # not replace a slot just because that slot has no local text.
    return " | ".join(prompts), ",".join(str(int(length)) for length in lengths), ",".join(strengths)


def _pad_promptrelay_lengths_to_ltx(segment_lengths: str, raw_duration: int, ltxv_length: int) -> str:
    parts = [_safe_int(part, 0) for part in re.split(r"[,;\s]+", str(segment_lengths or "")) if str(part).strip()]
    if not parts:
        return str(segment_lengths or "")
    raw_duration = max(1, int(raw_duration or 0))
    ltxv_length = max(raw_duration, int(ltxv_length or raw_duration))
    total = sum(max(0, part) for part in parts)
    if total == raw_duration and ltxv_length > raw_duration:
        parts[-1] = max(1, int(parts[-1]) + (ltxv_length - raw_duration))
        return ",".join(str(int(part)) for part in parts)
    return str(segment_lengths or "")


def _build_guide_data(
    timeline: Dict[str, Any],
    start_frame: int,
    duration_frames: int,
    frame_rate: float,
    custom_width: int,
    custom_height: int,
    resize_method: str,
    divisible_by: int,
    img_compression: int,
    guide_strength: str,
    optional_latent: Any = None,
    multi_output: Any = None,
) -> Tuple[Dict[str, Any], int, int]:
    guide_data: Dict[str, Any] = {
        "images": [],
        "insert_frames": [],
        "strengths": [],
        "frame_rate": float(frame_rate),
        "labels": [],
        "reference_indices": [],
        "motion_forces": [],
        "source": "IAMCCS_ShotboardV4_visual_timeline",
        "image_sources": [],
    }
    derived_w = max(0, int(custom_width or 0))
    derived_h = max(0, int(custom_height or 0))
    segments = [
        seg for seg in _segments_from_timeline(timeline, "segments")
        if str(seg.get("type", "image") or "image").lower() in {"image", "video"}
        and not _safe_bool(seg.get("placeholder", False), False)
        and _safe_bool(seg.get("use_guide", seg.get("guide", True)), True)
        and (
            _segment_media_path(seg, prefer_video=str(seg.get("type", "")).lower() == "video")
            or str(seg.get("imageB64", seg.get("image_b64", "")) or "")
            or (
                torch.is_tensor(multi_output)
                and multi_output.ndim == 4
                and 1 <= _safe_int(seg.get("ref", seg.get("reference_index", seg.get("image_ref", 1))), 1) <= int(multi_output.shape[0])
            )
        )
        and _safe_int(seg.get("start", 0), 0) < int(start_frame) + int(duration_frames)
        and _safe_int(seg.get("start", 0), 0) + max(1, _safe_int(seg.get("length", 1), 1)) > int(start_frame)
    ]
    segments.sort(key=lambda item: _safe_int(item.get("start", 0), 0))
    strengths = []
    if str(guide_strength or "").strip():
        for chunk in str(guide_strength).split(","):
            try:
                strengths.append(float(chunk.strip()))
            except Exception:
                pass

    for index, raw_seg in enumerate(segments):
        seg = dict(raw_seg)
        seg_start = _safe_int(seg.get("start", 0), 0)
        seg_len = max(1, _safe_int(seg.get("length", 1), 1))
        offset = max(0, int(start_frame) - seg_start)
        seg_type = str(seg.get("type", "image") or "image").lower()
        if seg_type == "video":
            media_ref = _segment_media_path(seg, prefer_video=True)
            if offset > 0:
                seg["trimStart"] = float(seg.get("trimStart", seg.get("trim_start", 0)) or 0) + offset
                seg["length"] = max(1, seg_len - offset)
            tensor = _load_video_frames(media_ref, float(seg.get("trimStart", 0) or 0), float(seg.get("length", 1) or 1), float(frame_rate), str(seg.get("resampleMode", "nearest") or "nearest"))
        else:
            media_ref = _segment_media_path(seg, prefer_video=False)
            tensor = _load_image_tensor(seg)

        ref = max(1, _safe_int(seg.get("ref", seg.get("reference_index", seg.get("image_ref", index + 1))), index + 1))
        if not media_ref and torch.is_tensor(multi_output) and multi_output.ndim == 4 and ref <= int(multi_output.shape[0]):
            tensor = multi_output[ref - 1 : ref]
            media_ref = f"multi_output[{ref}]"

        fallback_strength = _safe_float(
            seg.get("guideStrength", seg.get("guide_strength", seg.get("strength", seg.get("force", 1.0)))),
            1.0,
        )
        strength = _cine_guide_strength(seg, fallback_strength)
        if strength <= 0.0:
            continue

        tensor = _resize_to_policy(tensor, int(custom_width or 0), int(custom_height or 0), resize_method, int(divisible_by or 32))
        tensor = _compress_video_like(tensor, int(img_compression or 0))
        if not guide_data["images"]:
            derived_h = int(tensor.shape[1])
            derived_w = int(tensor.shape[2])
        if _safe_bool(seg.get("isEndFrame", seg.get("is_end_frame", False)), False):
            insert_frame = max(0, seg_start + seg_len - 1 - int(start_frame))
        else:
            insert_frame = max(0, seg_start - int(start_frame))
        guide_data["images"].append(tensor)
        guide_data["insert_frames"].append(int(insert_frame))
        guide_data["strengths"].append(float(strength))
        guide_data["labels"].append(str(seg.get("label", f"guide_{index + 1}")))
        guide_data["reference_indices"].append(int(ref))
        guide_data["motion_forces"].append(float(_safe_float(seg.get("motion_force", seg.get("force", 0.0)), 0.0)))
        guide_data["image_sources"].append(str(media_ref or "imageB64"))

    if not guide_data["images"] and optional_latent is None:
        src_w = derived_w if derived_w > 0 else 768
        src_h = derived_h if derived_h > 0 else 512
        retake_video = _retake_video(timeline)
        if _retake_active(timeline) and retake_video:
            vw, vh = _video_dimensions(retake_video.get("imageFile") or retake_video.get("videoFile") or retake_video.get("fileName"))
            if vw > 0 and vh > 0:
                src_w, src_h = vw, vh
        if src_w <= 0 or src_h <= 0:
            for seg in _segments_from_timeline(timeline, "motionSegments"):
                vw, vh = _video_dimensions(_segment_media_path(seg, prefer_video=True))
                if vw > 0 and vh > 0:
                    src_w, src_h = vw, vh
                    break
        dummy = torch.zeros((1, max(1, int(src_h)), max(1, int(src_w)), 3), dtype=torch.float32)
        dummy = _resize_to_policy(dummy, int(custom_width or 0), int(custom_height or 0), resize_method, int(divisible_by or 32))
        guide_data["images"].append(dummy)
        guide_data["insert_frames"].append(0)
        guide_data["strengths"].append(0.0)
        guide_data["labels"].append("dummy_shape_anchor")
        guide_data["reference_indices"].append(0)
        guide_data["motion_forces"].append(0.0)
        guide_data["image_sources"].append("dummy_shape_anchor")
        derived_h = int(dummy.shape[1])
        derived_w = int(dummy.shape[2])

    if derived_w <= 0:
        derived_w = 768
    if derived_h <= 0:
        derived_h = 512
    return guide_data, derived_w, derived_h


def _empty_audio(pixel_frames: int, frame_rate: float) -> Dict[str, Any]:
    sample_rate = 44100
    total_samples = max(1, int(math.ceil(max(1, int(pixel_frames)) / max(1.0, float(frame_rate or 24.0)) * sample_rate)))
    return {"waveform": torch.zeros((1, 2, total_samples), dtype=torch.float32), "sample_rate": sample_rate}


def _audio_segments_for_mode(timeline: Dict[str, Any], override_audio: bool, duration_frames: int) -> Tuple[List[Dict[str, Any]], str]:
    if _retake_active(timeline) and _retake_video(timeline):
        retake = _retake_video(timeline)
        video_file = retake.get("imageFile") or retake.get("videoFile") or retake.get("fileName")
        if video_file:
            return [{
                "videoFile": video_file,
                "audioFile": video_file,
                "start": 0,
                "length": retake.get("videoDurationFrames", duration_frames),
                "trimStart": 0,
            }], "videoFile"
    if override_audio:
        return _segments_from_timeline(timeline, "motionSegments"), "videoFile"
    return _segments_from_timeline(timeline, "audioSegments"), "audioFile"


def _build_combined_audio(timeline: Dict[str, Any], start_frame: int, duration_frames: int, frame_rate: float, override_audio: bool) -> Dict[str, Any]:
    audio_out = _empty_audio(duration_frames, frame_rate)
    segments, file_key = _audio_segments_for_mode(timeline, override_audio, duration_frames)
    if not segments:
        return audio_out
    try:
        import av
    except Exception:
        return audio_out

    sample_rate = int(audio_out["sample_rate"])
    out_waveform = torch.zeros((2, audio_out["waveform"].shape[-1]), dtype=torch.float32)
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        media_ref = seg.get(file_key) or seg.get("audioFile") or seg.get("videoFile")
        file_path = _resolve_media_path(media_ref)
        buffer = None
        if file_path:
            try:
                with open(file_path, "rb") as handle:
                    buffer = io.BytesIO(handle.read())
            except Exception:
                buffer = None
        if buffer is None and file_key == "audioFile" and seg.get("audioB64"):
            encoded = str(seg.get("audioB64") or "")
            if "," in encoded:
                encoded = encoded.split(",", 1)[1]
            try:
                buffer = io.BytesIO(base64.b64decode(encoded))
            except Exception:
                buffer = None
        if buffer is None:
            continue
        try:
            chunks = []
            with av.open(buffer) as container:
                streams = list(container.streams.audio)
                if not streams:
                    continue
                resampler = av.AudioResampler(format="fltp", layout="stereo", rate=sample_rate)
                for frame in container.decode(streams[0]):
                    for resampled in resampler.resample(frame):
                        chunks.append(torch.from_numpy(resampled.to_ndarray()))
                for resampled in resampler.resample(None):
                    chunks.append(torch.from_numpy(resampled.to_ndarray()))
            if not chunks:
                continue
            waveform = torch.cat(chunks, dim=1).to(torch.float32)
            seg_start = float(seg.get("start", 0) or 0)
            seg_len = max(1.0, float(seg.get("length", 1) or 1))
            trim = max(0.0, float(seg.get("trimStart", seg.get("trim_start", 0)) or 0))
            if seg_start + seg_len <= start_frame:
                continue
            offset = max(0.0, float(start_frame) - seg_start)
            trim += offset
            seg_len = max(1.0, seg_len - offset)
            seg_start = max(0.0, seg_start - float(start_frame))
            src_start = max(0, int(trim / max(1.0, float(frame_rate)) * sample_rate))
            src_len = max(1, int(seg_len / max(1.0, float(frame_rate)) * sample_rate))
            src_end = min(int(waveform.shape[1]), src_start + src_len)
            if src_end <= src_start:
                continue
            clip_waveform = waveform[:, src_start:src_end]
            dst_start = int(seg_start / max(1.0, float(frame_rate)) * sample_rate)
            if dst_start >= int(out_waveform.shape[1]):
                continue
            dst_end = min(int(out_waveform.shape[1]), dst_start + int(clip_waveform.shape[1]))
            actual = dst_end - dst_start
            if actual > 0:
                out_waveform[:, dst_start:dst_end] += clip_waveform[:, :actual]
        except Exception as exc:
            print(f"[IAMCCS Shotboard V4 Backend] audio segment skipped: {exc}")
            continue
    return {"waveform": out_waveform.unsqueeze(0), "sample_rate": sample_rate}


def _encode_audio_latent(audio_vae: Any, audio_out: Dict[str, Any], timeline: Dict[str, Any], start_frame: int, duration_frames: int, frame_rate: float, inpaint_audio: bool, override_audio: bool, active_custom_audio: bool) -> Dict[str, Any]:
    if audio_vae is None:
        return {}
    if not active_custom_audio:
        return IAMCCS_CineFilmmakerBackend._empty_audio_latent(audio_vae, duration_frames, frame_rate)
    waveform = audio_out.get("waveform") if isinstance(audio_out, dict) else None
    if torch.is_tensor(waveform) and waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    if not torch.is_tensor(waveform) or waveform.ndim != 3:
        return IAMCCS_CineFilmmakerBackend._empty_audio_latent(audio_vae, duration_frames, frame_rate)
    try:
        if hasattr(audio_vae, "first_stage_model"):
            latent_samples = audio_vae.encode(waveform.movedim(1, -1))
        else:
            latent_samples = audio_vae.encode({"waveform": waveform, "sample_rate": int(audio_out.get("sample_rate", 44100))})
        if not torch.is_tensor(latent_samples) or latent_samples.numel() == 0:
            return IAMCCS_CineFilmmakerBackend._empty_audio_latent(audio_vae, duration_frames, frame_rate)
        batch, _channels, frames, height = latent_samples.shape
        if _retake_active(timeline):
            mask = torch.zeros((batch, frames, height), dtype=torch.float32, device=latent_samples.device)
            retake_start = float(timeline.get("retakeStart", timeline.get("retake_start", 0)) or 0)
            retake_len = float(timeline.get("retakeLength", timeline.get("retake_length", 0)) or 0)
            overlap_start = max(float(start_frame), retake_start)
            overlap_end = min(float(start_frame + duration_frames), retake_start + retake_len)
            if overlap_end > overlap_start:
                rel_start = overlap_start - float(start_frame)
                rel_len = overlap_end - overlap_start
                total_sec = max(1.0 / max(1.0, frame_rate), float(duration_frames) / max(1.0, frame_rate))
                start_idx = int(((rel_start / max(1.0, frame_rate)) / total_sec) * frames)
                end_idx = int((((rel_start + rel_len) / max(1.0, frame_rate)) / total_sec) * frames)
                mask[:, max(0, min(frames, start_idx)):max(0, min(frames, end_idx)), :] = 1.0
        else:
            mask = torch.ones((batch, frames, height), dtype=torch.float32, device=latent_samples.device)
            segments, file_key = _audio_segments_for_mode(timeline, override_audio, duration_frames)
            for seg in segments:
                if not isinstance(seg, dict) or not (seg.get(file_key) or seg.get("audioFile") or seg.get("videoFile")):
                    continue
                seg_start = float(seg.get("start", 0) or 0)
                seg_len = max(1.0, float(seg.get("length", 1) or 1))
                if seg_start + seg_len <= start_frame or seg_start >= start_frame + duration_frames:
                    continue
                offset = max(0.0, float(start_frame) - seg_start)
                seg_start = max(0.0, seg_start - float(start_frame))
                seg_len = max(1.0, seg_len - offset)
                total_sec = max(1.0 / max(1.0, frame_rate), float(duration_frames) / max(1.0, frame_rate))
                start_idx = int(((seg_start / max(1.0, frame_rate)) / total_sec) * frames)
                end_idx = int((((seg_start + seg_len) / max(1.0, frame_rate)) / total_sec) * frames)
                mask[:, max(0, min(frames, start_idx)):max(0, min(frames, end_idx)), :] = 0.0
            if not inpaint_audio:
                mask = torch.zeros((batch, frames, height), dtype=torch.float32, device=latent_samples.device)
        return {
            "samples": latent_samples,
            "type": "audio",
            "sample_rate": int(audio_out.get("sample_rate", 44100)),
            "noise_mask": mask,
        }
    except Exception as exc:
        print(f"[IAMCCS Shotboard V4 Backend] audio latent fallback: {exc}")
        return IAMCCS_CineFilmmakerBackend._empty_audio_latent(audio_vae, duration_frames, frame_rate)


def _build_motion_guide_data(timeline: Dict[str, Any], start_frame: int, duration_frames: int, frame_rate: float, resize_method: str, use_custom_motion: bool) -> Dict[str, Any]:
    out = {
        "segments": [],
        "frame_rate": float(frame_rate),
        "duration_frames": int(duration_frames),
        "resize_method": _normalize_resize_method(resize_method),
        "schema": "iamccs.motion_guide_data.video_timeline",
    }
    if not use_custom_motion:
        return out
    for raw_seg in _segments_from_timeline(timeline, "motionSegments"):
        seg = dict(raw_seg)
        start = _safe_int(seg.get("start", 0), 0)
        length = max(1, _safe_int(seg.get("length", 1), 1))
        if start >= start_frame + duration_frames or start + length <= start_frame:
            continue
        video_ref = _segment_media_path(seg, prefer_video=True)
        if not video_ref:
            continue
        offset = max(0, start_frame - start)
        new_start = max(0, start - start_frame)
        clipped_len = min(length - offset, duration_frames - new_start)
        if clipped_len <= 0:
            continue
        seg["start"] = int(new_start)
        seg["length"] = int(clipped_len)
        seg["trimStart"] = float(seg.get("trimStart", seg.get("trim_start", 0)) or 0) + float(offset)
        seg["videoFile"] = video_ref
        if "videoStrength" not in seg:
            seg["videoStrength"] = _safe_float(seg.get("strength", 1.0), 1.0)
        if "videoAttentionStrength" not in seg:
            seg["videoAttentionStrength"] = _safe_float(seg.get("attention_strength", 0.65), 0.65)
        if "resampleMode" not in seg:
            seg["resampleMode"] = str(seg.get("resample_mode", "nearest") or "nearest")
        out["segments"].append(seg)
    return out


class IAMCCS_CineShotboardV4Backend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "cine_linx": (SUPERNODE_LINX_TYPE,),
            },
            "optional": {
                "audio_vae": ("VAE",),
                "optional_latent": ("LATENT",),
                "timeline_data": ("STRING", {"default": "", "multiline": True}),
                "global_prompt": ("STRING", {"default": "", "multiline": True}),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "duration_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "frame_rate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01}),
                "use_custom_audio": ("BOOLEAN", {"default": False}),
                "use_custom_motion": ("BOOLEAN", {"default": True}),
                "inpaint_audio": ("BOOLEAN", {"default": True}),
                "override_audio": ("BOOLEAN", {"default": False}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "resize_method": (["maintain aspect ratio", "stretch to fit", "pad", "pad green", "crop"], {"default": "crop"}),
                "divisible_by": ("INT", {"default": 32, "min": 1, "max": 256, "step": 1}),
                "img_compression": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "epsilon": ("FLOAT", {"default": 0.001, "min": 0.0001, "max": 0.99, "step": 0.0001}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "LATENT", "LATENT", "GUIDE_DATA", MOTION_GUIDE_TYPE, "FLOAT", "AUDIO")
    RETURN_NAMES = ("model", "positive", "video_latent", "audio_latent", "guide_data", "motion_guide_data", "frame_rate", "combined_audio")
    FUNCTION = "execute"
    CATEGORY = "IAMCCS/Cine/Shotboard V4"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    @staticmethod
    def _encode_basic(clip: Any, text: str) -> Any:
        result = comfy_nodes.CLIPTextEncode().encode(clip, str(text or ""))
        return result[0] if isinstance(result, tuple) else result

    def execute(
        self,
        model,
        clip,
        cine_linx,
        audio_vae=None,
        optional_latent=None,
        timeline_data="",
        global_prompt="",
        start_frame=0,
        duration_frames=0,
        frame_rate=0.0,
        use_custom_audio=False,
        use_custom_motion=True,
        inpaint_audio=True,
        override_audio=False,
        custom_width=0,
        custom_height=0,
        resize_method="crop",
        divisible_by=32,
        img_compression=0,
        epsilon=0.001,
    ):
        linx = cine_linx if isinstance(cine_linx, dict) else {}
        resources = linx.get("resources", {}) if isinstance(linx.get("resources"), dict) else {}
        outputs = linx.get("outputs", {}) if isinstance(linx.get("outputs"), dict) else {}
        payload = resources.get("cine_payload") if isinstance(resources.get("cine_payload"), dict) else {}
        timeline = _find_timeline(linx, timeline_data)

        timeline_fps = _safe_float(
            timeline.get("frame_rate", timeline.get("frameRate", timeline.get("fps", 0))),
            0.0,
        )
        fps = timeline_fps if timeline_fps > 0 else (
            float(frame_rate)
            if float(frame_rate or 0.0) > 0
            else _safe_float(resources.get("cine_frame_rate", outputs.get("frame_rate", payload.get("frame_rate", 24))), 24.0)
        )
        fps = max(1.0, fps)
        start_from_timeline = timeline.get("normalStartFrame", timeline.get("normal_start_frame", timeline.get("start_frame", None)))
        start = int(start_from_timeline if start_from_timeline is not None else (start_frame or payload.get("start_frame", 0)) or 0)
        # Keep the raw timeline duration as the truth. V3 and the duration crop
        # both distinguish it from the LTX 8n+1 padded latent length.
        raw_duration = _safe_int(timeline.get("normalDurationFrames", timeline.get("normal_duration_frames", timeline.get("duration_frames", 0))), 0)
        if raw_duration <= 0:
            timeline_duration_seconds = _safe_float(
                timeline.get("duration_seconds", timeline.get("durationSeconds", 0)),
                0.0,
            )
            if timeline_duration_seconds > 0:
                raw_duration = max(1, int(round(timeline_duration_seconds * fps)))
        if raw_duration <= 0:
            duration_seconds = _safe_float(resources.get("cine_duration_seconds", payload.get("duration_seconds", 0)), 0.0)
            if duration_seconds > 0:
                raw_duration = max(1, int(round(duration_seconds * fps)))
        if raw_duration <= 0:
            raw_duration = int(duration_frames or 0)
        if raw_duration <= 0:
            raw_duration = _safe_int(outputs.get("max_frames", resources.get("cine_max_frames", payload.get("max_frames", 0))), 0)
        if raw_duration <= 0:
            duration_seconds = _safe_float(timeline.get("duration_seconds", resources.get("cine_duration_seconds", payload.get("duration_seconds", 5.0))), 5.0)
            raw_duration = max(1, int(round(duration_seconds * fps)))
        ltxv_length = _round_ltx_frames(raw_duration, "up_8n_plus_1")
        retake_active = _retake_active(timeline)
        timeline_prompt = str(timeline.get("global_prompt", timeline.get("prompt", "")) or "").strip()
        retake_timeline_prompt = str(timeline.get("retake_global_prompt", timeline.get("retakePrompt", timeline.get("retake_prompt", ""))) or "").strip()
        fallback_prompt = str(global_prompt or resources.get("cine_global_prompt", outputs.get("global_prompt", payload.get("global_prompt", ""))) or "")
        prompt_source = "empty"
        if retake_active and retake_timeline_prompt:
            prompt = retake_timeline_prompt
            prompt_source = "timeline.retake_global_prompt"
        elif timeline_prompt:
            prompt = timeline_prompt
            prompt_source = "timeline.global_prompt"
        elif fallback_prompt:
            prompt = fallback_prompt
            prompt_source = "backend_fallback.global_prompt"
        else:
            prompt = ""

        local_prompts = str(resources.get("cine_local_prompts", outputs.get("local_prompts", payload.get("local_prompts", ""))) or "")
        segment_lengths = str(resources.get("cine_segment_lengths", outputs.get("segment_lengths", payload.get("segment_lengths", ""))) or "")
        local_prompts, segment_lengths, guide_strength = _derive_prompt_parts(
            timeline,
            local_prompts,
            segment_lengths,
            duration_frames=ltxv_length,
        )
        segment_lengths = _pad_promptrelay_lengths_to_ltx(segment_lengths, raw_duration, ltxv_length)

        width = _safe_int(
            timeline.get("custom_width", timeline.get("image_width", 0)),
            0,
        ) or _safe_int(resources.get("cine_image_width", payload.get("image_width", 0)), 0) or int(custom_width or 0)
        height = _safe_int(
            timeline.get("custom_height", timeline.get("image_height", 0)),
            0,
        ) or _safe_int(resources.get("cine_image_height", payload.get("image_height", 0)), 0) or int(custom_height or 0)
        method = _normalize_resize_method(
            timeline.get("resize_method", timeline.get("image_resize_method", ""))
            or resources.get("cine_image_resize_method", payload.get("image_resize_method", ""))
            or resize_method
            or "crop"
        )
        divisor = max(1, _safe_int(
            timeline.get("divisible_by", timeline.get("image_multiple_of", 0)),
            0,
        ) or _safe_int(resources.get("cine_image_multiple_of", payload.get("image_multiple_of", 0)), 0) or int(divisible_by or 32))
        compression = max(0, _safe_int(
            timeline.get("img_compression", 0),
            0,
        ) if "img_compression" in timeline else _safe_int(resources.get("cine_img_compression", payload.get("img_compression", img_compression)), int(img_compression or 0)))

        guide_data, derived_w, derived_h = _build_guide_data(
            timeline,
            start,
            raw_duration,
            fps,
            width,
            height,
            method,
            divisor,
            compression,
            guide_strength,
            optional_latent=optional_latent,
            multi_output=resources.get("cine_multi_input"),
        )

        latent = optional_latent if isinstance(optional_latent, dict) else IAMCCS_CineFilmmakerBackend._empty_latent(derived_w, derived_h, ltxv_length)
        if local_prompts.strip() and segment_lengths.strip():
            try:
                promptrelay_nodes = _load_original_promptrelay_module()
                patched_model, positive = promptrelay_nodes._encode_relay(model, clip, latent, prompt, local_prompts, segment_lengths, float(epsilon or 0.001))
            except Exception as exc:
                print(f"[IAMCCS Shotboard V4 Backend] PromptRelay fallback: {exc}")
                patched_model = model.clone() if hasattr(model, "clone") else model
                positive = self._encode_basic(clip, prompt)
        else:
            patched_model = model.clone() if hasattr(model, "clone") else model
            positive = self._encode_basic(clip, prompt)

        effective_override_audio = _safe_bool(timeline.get("overrideAudio", timeline.get("override_audio", override_audio)), bool(override_audio))
        effective_inpaint_audio = _safe_bool(timeline.get("inpaintAudio", timeline.get("inpaint_audio", inpaint_audio)), bool(inpaint_audio))
        timeline_has_audio = bool(_segments_from_timeline(timeline, "audioSegments"))
        effective_custom_audio = bool(use_custom_audio or _safe_bool(timeline.get("use_custom_audio", False), False) or timeline_has_audio or effective_override_audio or retake_active)
        # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        # audio_out is built for raw_duration so it matches the user-chosen duration exactly.
        # IAMCCS_LTXVideoDurationCrop will trim the video to raw_duration; audio needs no trimming.
        # audio_for_latent is built for ltxv_length (the 8n+1 padded length) so the audio VAE
        # produces the correct latent size expected by the LTX sampler.
        def build_audio(target_frames: int) -> Dict[str, Any]:
            # Normal Shotboard audio follows the same AudioBoard DSP path as
            # V3. Motion/retake override modes remain V4-specific and keep the
            # local-window builder below so video audio stays aligned.
            if start == 0 and not effective_override_audio and not retake_active:
                return IAMCCS_CineFilmmakerBackend._build_combined_audio(
                    _json_dumps(timeline),
                    int(target_frames),
                    float(fps),
                )
            return _build_combined_audio(timeline, start, int(target_frames), fps, effective_override_audio)

        audio_out = build_audio(raw_duration)
        audio_for_latent = build_audio(ltxv_length)
        audio_latent = _encode_audio_latent(audio_vae, audio_for_latent, timeline, start, ltxv_length, fps, effective_inpaint_audio, effective_override_audio, effective_custom_audio)

        effective_custom_motion = bool(use_custom_motion and _safe_bool(timeline.get("use_custom_motion", timeline.get("motionTrackEnabled", True)), True))
        motion_guide_data = _build_motion_guide_data(timeline, start, raw_duration, fps, method, effective_custom_motion)
        resolved_ic_lora_name, resolved_ic_lora_strength = _timeline_ic_lora_settings(timeline, guide_data, "None", 1.0)
        resolved_image_attention = max(0.0, min(1.0, _timeline_float_setting(
            timeline,
            guide_data,
            ("image_attention_strength", "imageAttentionStrength"),
            1.0,
        )))
        if resolved_ic_lora_name != "None":
            guide_data["ic_lora_name"] = resolved_ic_lora_name
            guide_data["icLoraName"] = resolved_ic_lora_name
            guide_data["ic_lora_strength"] = float(resolved_ic_lora_strength)
            guide_data["icLoraStrength"] = float(resolved_ic_lora_strength)
            guide_data["image_attention_strength"] = float(resolved_image_attention)
            motion_guide_data["ic_lora_name"] = resolved_ic_lora_name
            motion_guide_data["ic_lora_strength"] = float(resolved_ic_lora_strength)
        guide_data["timeline_data"] = _json_dumps(timeline)
        guide_data["start_frame"] = int(start)
        guide_data["duration_frames"] = int(raw_duration)
        guide_data["resize_method"] = method
        guide_data["global_prompt"] = str(prompt)
        guide_data["local_prompts"] = str(local_prompts)
        guide_data["segment_lengths"] = str(segment_lengths)
        timeline_source = str(timeline.get("_iamccs_timeline_source", "unknown") or "unknown")
        retake_video = _retake_video(timeline)
        retake_video_file = str(
            retake_video.get("imageFile")
            or retake_video.get("videoFile")
            or retake_video.get("fileName")
            or ""
        )
        guide_data["iamccs_backend"] = {
            "schema": "iamccs.shotboard_v4.backend_report",
            "timeline_source": timeline_source,
            "shotboard_controls_backend": timeline_source.startswith("cine_linx."),
            "backend_widgets_are_fallback": True,
            "prompt_source": prompt_source,
            "effective_prompt": str(prompt),
            "timeline_prompt": timeline_prompt,
            "retake_timeline_prompt": retake_timeline_prompt,
            "fallback_prompt": fallback_prompt,
            "local_prompts": str(local_prompts),
            "segment_lengths": str(segment_lengths),
            "duration_frames": int(raw_duration),
            "ltxv_length": int(ltxv_length),
            "start_frame": int(start),
            "frame_rate": float(fps),
            "custom_width": int(width or derived_w or 0),
            "custom_height": int(height or derived_h or 0),
            "derived_width": int(derived_w),
            "derived_height": int(derived_h),
            "resize_method": method,
            "image_guides": len(guide_data.get("images", [])),
            "motion_segments": len(motion_guide_data.get("segments", [])),
            "visual_segments": len(_segments_from_timeline(timeline, "segments")),
            "audio_segments": len(_segments_from_timeline(timeline, "audioSegments")),
            "custom_audio": bool(effective_custom_audio),
            "inpaint_audio": bool(effective_inpaint_audio),
            "override_audio": bool(effective_override_audio),
            "custom_motion": bool(effective_custom_motion),
            "ic_lora_name": resolved_ic_lora_name,
            "ic_lora_strength": float(resolved_ic_lora_strength),
            "image_attention_strength": float(resolved_image_attention),
            "retake_mode": bool(retake_active),
            "retake_video_file": retake_video_file,
            "retake_start": _safe_int(timeline.get("retakeStart", timeline.get("retake_start", 0)), 0),
            "retake_length": _safe_int(timeline.get("retakeLength", timeline.get("retake_length", 0)), 0),
            "retake_strength": _safe_float(timeline.get("retakeStrength", timeline.get("retake_strength", 1.0)), 1.0),
            "video_to_video_enabled": bool(_safe_bool(timeline.get("videoToVideoEnabled", timeline.get("video_to_video_enabled", False)), False)),
            "clip_edit_mode": str(timeline.get("clipEditMode", timeline.get("clip_edit_mode", "")) or ""),
            "continuation_mode": str(timeline.get("continuationMode", timeline.get("continuation_mode", "")) or ""),
            "standard_video_mode": "timeline video segments are guides; retake_mode encodes the source video as the editable latent base",
        }
        return patched_model, positive, latent, audio_latent, guide_data, motion_guide_data, float(fps), audio_out


def _conditioning_get_entries(conditioning: Any) -> List[Dict[str, Any]]:
    for item in conditioning or []:
        meta = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 and isinstance(item[1], dict) else {}
        entries = meta.get("guide_attention_entries")
        if entries is not None:
            return list(entries)
    return []


def _set_conditioning_entries(conditioning: Any, entries: List[Dict[str, Any]]) -> Any:
    import node_helpers

    return node_helpers.conditioning_set_values(conditioning, {"guide_attention_entries": entries})


def _append_attention_entry(conditioning: Any, token_count: int, latent_shape: List[int], attention_strength: float) -> Any:
    entries = _conditioning_get_entries(conditioning)
    entries.append({
        "pre_filter_count": int(token_count),
        "strength": float(attention_strength),
        "pixel_mask": None,
        "latent_shape": list(latent_shape),
    })
    return _set_conditioning_entries(conditioning, entries)


def _clone_noise_mask(latent: Dict[str, Any], latent_image: torch.Tensor) -> torch.Tensor:
    mask = latent.get("noise_mask") if isinstance(latent, dict) else None
    if torch.is_tensor(mask):
        return mask.clone()
    batch, _channels, frames, _height, _width = latent_image.shape
    return torch.ones((batch, 1, frames, 1, 1), dtype=torch.float32, device=latent_image.device)


def _resize_latent_spatial(latent_image: torch.Tensor, noise_mask: torch.Tensor, width: int, height: int, method: str) -> Tuple[torch.Tensor, torch.Tensor]:
    import comfy.utils

    batch, channels, frames, old_h, old_w = latent_image.shape
    if int(width) == int(old_w) and int(height) == int(old_h):
        return latent_image, noise_mask
    latent_4d = latent_image.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, old_h, old_w)
    latent_4d = comfy.utils.common_upscale(latent_4d, int(width), int(height), str(method or "bicubic"), "disabled")
    latent_image = latent_4d.reshape(batch, frames, channels, int(height), int(width)).permute(0, 2, 1, 3, 4)
    if noise_mask is not None and (noise_mask.shape[-1] > 1 or noise_mask.shape[-2] > 1):
        mask_4d = noise_mask.permute(0, 2, 1, 3, 4).reshape(batch * frames, 1, old_h, old_w)
        mask_4d = comfy.utils.common_upscale(mask_4d, int(width), int(height), str(method or "bicubic"), "disabled")
        noise_mask = mask_4d.reshape(batch, frames, 1, int(height), int(width)).permute(0, 2, 1, 3, 4)
    return latent_image, noise_mask


def _snap_latent_grid(latent_image: torch.Tensor, noise_mask: torch.Tensor, downscale_factor: float, method: str) -> Tuple[torch.Tensor, torch.Tensor]:
    factor = int(max(1, round(float(downscale_factor or 1.0))))
    if factor <= 1:
        return latent_image, noise_mask
    _batch, _channels, _frames, height, width = latent_image.shape
    new_w = int(math.ceil(width / factor) * factor)
    new_h = int(math.ceil(height / factor) * factor)
    if new_w == width and new_h == height:
        return latent_image, noise_mask
    return _resize_latent_spatial(latent_image, noise_mask, new_w, new_h, method)


def _load_ic_lora(model: Any, ic_lora_name: str, strength_model: float) -> Tuple[Any, float, bool]:
    if model is None or not ic_lora_name or ic_lora_name == "None":
        return model, 1.0, False
    import comfy
    import comfy.sd

    lora_path = folder_paths.get_full_path_or_raise("loras", ic_lora_name)
    lora, metadata = comfy.utils.load_torch_file(lora_path, safe_load=True, return_metadata=True)
    try:
        downscale = float(metadata.get("reference_downscale_factor", 1.0))
    except Exception:
        downscale = 1.0
    if float(strength_model or 0.0) != 0.0:
        model, _clip = comfy.sd.load_lora_for_models(model, None, lora, float(strength_model), 0)
    return model, max(1.0, downscale), True


_IC_LORA_PLACEHOLDER_NAMES = {"", "none", "ic-lora", "iclora", "ic_lora", "3dreal", "3dreal_reference"}
_IC_LORA_ROLE_HINTS = {
    "3dreal_reference": (
        "3DREAL-strong.safetensors",
        "3DREAL-light.safetensors",
        "3dreal",
    ),
    "motion_reference": (
        "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors",
        "motion-track-control",
        "motion_track",
    ),
    "motion_brush": (
        "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors",
        "motion-track-control",
        "motion_track",
    ),
    "camera_reference": (
        "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
        "union-control",
        "camera",
    ),
    "ingredients": (
        "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors",
        "ingredients",
    ),
}


def _available_lora_names() -> List[str]:
    try:
        return [str(item) for item in folder_paths.get_filename_list("loras")]
    except Exception:
        return []


def _find_lora_by_hints(hints: Tuple[str, ...]) -> str:
    names = _available_lora_names()
    if not names:
        return "None"
    lower_map = {name.lower(): name for name in names}
    for hint in hints:
        key = str(hint or "").strip().lower()
        if key in lower_map:
            return lower_map[key]
    for hint in hints:
        key = str(hint or "").strip().lower()
        if not key:
            continue
        for name in names:
            if key in name.lower():
                return name
    return "None"


def _resolve_ic_lora_name(candidate: Any, role: Any = "") -> str:
    value = str(candidate or "").strip()
    role_value = str(role or "").strip().lower()
    lower = value.lower()
    if lower and lower not in _IC_LORA_PLACEHOLDER_NAMES:
        return value
    hints = _IC_LORA_ROLE_HINTS.get(role_value)
    if hints:
        return _find_lora_by_hints(hints)
    if lower in {"3dreal", "3dreal_reference"}:
        return _find_lora_by_hints(_IC_LORA_ROLE_HINTS["3dreal_reference"])
    return "None"


def _ic_lora_explicit_none(value: Any) -> bool:
    return str(value or "").strip().lower() == "none"


def _timeline_ic_lora_settings(timeline: Dict[str, Any], guide_data: Any, fallback_name: str, fallback_strength: float) -> Tuple[str, float]:
    candidates: List[Tuple[Any, Any]] = []
    strengths: List[Any] = []
    explicit_none = False
    if isinstance(timeline, dict):
        backend_settings = timeline.get("backend_settings") if isinstance(timeline.get("backend_settings"), dict) else {}
        explicit_none = any(_ic_lora_explicit_none(value) for value in (
            timeline.get("ic_lora_name"),
            timeline.get("icLoraName"),
            timeline.get("ic_lora"),
            backend_settings.get("ic_lora_name"),
            backend_settings.get("icLoraName"),
        ))
        candidates.extend([
            (timeline.get("ic_lora_name"), timeline.get("ic_lora_role")),
            (timeline.get("icLoraName"), timeline.get("icLoraRole")),
            (timeline.get("ic_lora"), timeline.get("ic_lora_role")),
            (backend_settings.get("ic_lora_name"), backend_settings.get("ic_lora_role")),
            (backend_settings.get("icLoraName"), backend_settings.get("icLoraRole")),
        ])
        strengths.extend([
            timeline.get("ic_lora_strength"),
            timeline.get("icLoraStrength"),
            backend_settings.get("ic_lora_strength"),
            backend_settings.get("icLoraStrength"),
        ])
        for seg in _segments_from_timeline(timeline, "motionSegments"):
            role = seg.get("ic_lora_role", seg.get("icLoraRole", seg.get("controlMode", seg.get("control_mode", ""))))
            candidates.extend([(seg.get("ic_lora_name"), role), (seg.get("icLoraName"), role), (seg.get("lora"), role), ("", role)])
            strengths.extend([seg.get("ic_lora_strength"), seg.get("icLoraStrength")])
    if isinstance(guide_data, dict):
        candidates.extend([(guide_data.get("ic_lora_name"), guide_data.get("ic_lora_role")), (guide_data.get("icLoraName"), guide_data.get("icLoraRole"))])
        strengths.extend([guide_data.get("ic_lora_strength"), guide_data.get("icLoraStrength")])
    if not explicit_none:
        candidates.append((fallback_name, ""))
    name = "None"
    for candidate, role in candidates:
        resolved = _resolve_ic_lora_name(candidate, role)
        if resolved != "None":
            name = resolved
            break
    strength = _safe_float(fallback_strength, 1.0)
    for candidate in strengths:
        if candidate is not None and str(candidate).strip() != "":
            strength = _safe_float(candidate, strength)
            break
    return name, float(strength)


def _timeline_float_setting(timeline: Dict[str, Any], guide_data: Any, keys: Tuple[str, ...], fallback: float) -> float:
    candidates: List[Any] = []
    if isinstance(timeline, dict):
        backend_settings = timeline.get("backend_settings") if isinstance(timeline.get("backend_settings"), dict) else {}
        for key in keys:
            candidates.append(timeline.get(key))
            candidates.append(backend_settings.get(key))
        for seg in _segments_from_timeline(timeline, "motionSegments"):
            for key in keys:
                candidates.append(seg.get(key))
    if isinstance(guide_data, dict):
        for key in keys:
            candidates.append(guide_data.get(key))
    for candidate in candidates:
        if candidate is not None and str(candidate).strip() != "":
            return _safe_float(candidate, fallback)
    return float(fallback)


def _encode_motion_video_guide(vae: Any, latent_width: int, latent_height: int, images: torch.Tensor, scale_factors: Tuple[int, int, int], latent_downscale_factor: float, crop: str, use_tiled_encode: bool, tile_size: int, tile_overlap: int, resize_method: str) -> torch.Tensor:
    time_scale, width_scale, height_scale = scale_factors
    keep = ((int(images.shape[0]) - 1) // int(time_scale)) * int(time_scale) + 1
    images = images[:keep]
    target_w = max(8, int(latent_width * width_scale / max(1.0, float(latent_downscale_factor))))
    target_h = max(8, int(latent_height * height_scale / max(1.0, float(latent_downscale_factor))))
    method = _normalize_resize_method(resize_method)
    if method == "maintain aspect ratio":
        method = "pad"
    pixels = _resize_image(images, target_w, target_h, method, 1)[:, :, :, :3]
    if use_tiled_encode:
        return vae.encode_tiled(pixels, tile_x=int(tile_size), tile_y=int(tile_size), overlap=int(tile_overlap))
    return vae.encode(pixels)


def _dilate_latent(samples: torch.Tensor, mask: torch.Tensor, horizontal_scale: int, vertical_scale: int, target_h: int, target_w: int) -> Tuple[torch.Tensor, torch.Tensor]:
    horizontal_scale = int(max(1, horizontal_scale))
    vertical_scale = int(max(1, vertical_scale))
    if horizontal_scale == 1 and vertical_scale == 1:
        return samples, mask
    out_shape = samples.shape[:3] + (samples.shape[3] * vertical_scale, samples.shape[4] * horizontal_scale)
    out = torch.zeros(out_shape, device=samples.device, dtype=samples.dtype, requires_grad=False)
    out[..., ::vertical_scale, ::horizontal_scale] = samples
    out_mask = torch.full((samples.shape[0], 1, samples.shape[2], out_shape[3], out_shape[4]), -1.0, device=samples.device, dtype=samples.dtype, requires_grad=False)
    out_mask[..., ::vertical_scale, ::horizontal_scale] = mask if mask is not None else 1.0
    if out.shape[3] == target_h and out.shape[4] == target_w:
        return out, out_mask
    fixed = torch.zeros(out.shape[:3] + (target_h, target_w), device=out.device, dtype=out.dtype, requires_grad=False)
    fixed_mask = torch.full((out_mask.shape[0], 1, out_mask.shape[2], target_h, target_w), -1.0, device=out_mask.device, dtype=out_mask.dtype, requires_grad=False)
    copy_h = min(target_h, out.shape[3])
    copy_w = min(target_w, out.shape[4])
    fixed[..., :copy_h, :copy_w] = out[..., :copy_h, :copy_w]
    fixed_mask[..., :copy_h, :copy_w] = out_mask[..., :copy_h, :copy_w]
    return fixed, fixed_mask


class IAMCCS_CineShotboardV4Guide:
    @classmethod
    def INPUT_TYPES(cls):
        try:
            loras = folder_paths.get_filename_list("loras")
        except Exception:
            loras = []
        if not loras:
            loras = ["None"]
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "guide_data": ("GUIDE_DATA",),
            },
            "optional": {
                "motion_guide_data": (MOTION_GUIDE_TYPE,),
                "model": ("MODEL",),
                "ic_lora_name": (["None"] + [name for name in loras if name != "None"], {"default": "None"}),
                "ic_lora_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01}),
                "upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "bislerp"], {"default": "bicubic"}),
                "image_attention_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "crop": (["disabled", "center"], {"default": "center"}),
                "auto_snap_ic_grid": ("BOOLEAN", {"default": True}),
                "use_tiled_encode": ("BOOLEAN", {"default": False}),
                "tile_size": ("INT", {"default": 256, "min": 64, "max": 512, "step": 32}),
                "tile_overlap": ("INT", {"default": 64, "min": 16, "max": 256, "step": 16}),
                "retake_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "MODEL", "FLOAT")
    RETURN_NAMES = ("positive", "negative", "latent", "model", "latent_downscale_factor")
    FUNCTION = "execute"
    CATEGORY = "IAMCCS/Cine/Shotboard V4"

    def execute(
        self,
        positive,
        negative,
        vae,
        latent,
        guide_data,
        motion_guide_data=None,
        model=None,
        ic_lora_name="None",
        ic_lora_strength=1.0,
        scale_by=1.0,
        upscale_method="bicubic",
        image_attention_strength=1.0,
        crop="center",
        auto_snap_ic_grid=True,
        use_tiled_encode=False,
        tile_size=256,
        tile_overlap=64,
        retake_mode=False,
    ):
        import comfy.utils
        import node_helpers
        from comfy_extras import nodes_lt

        motion_segments = (motion_guide_data or {}).get("segments", []) if isinstance(motion_guide_data, dict) else []
        resize_method = (guide_data or {}).get("resize_method") if isinstance(guide_data, dict) else None
        if not resize_method and isinstance(motion_guide_data, dict):
            resize_method = motion_guide_data.get("resize_method")
        resize_method = _normalize_resize_method(resize_method or ("crop" if crop == "center" else "stretch to fit"))
        timeline = _json_loads((guide_data or {}).get("timeline_data", "{}") if isinstance(guide_data, dict) else "{}", {})
        if isinstance(timeline, dict) and isinstance(motion_guide_data, dict) and motion_guide_data.get("segments") and not timeline.get("motionSegments"):
            timeline = dict(timeline)
            timeline["motionSegments"] = motion_guide_data.get("segments", [])
        ic_lora_name, ic_lora_strength = _timeline_ic_lora_settings(timeline, guide_data, ic_lora_name, ic_lora_strength)
        image_attention_strength = max(0.0, min(1.0, _timeline_float_setting(
            timeline,
            guide_data,
            ("image_attention_strength", "imageAttentionStrength"),
            float(image_attention_strength or 1.0),
        )))

        latent_downscale_factor = 1.0
        lora_loaded = False
        if model is not None and ic_lora_name != "None":
            model, latent_downscale_factor, lora_loaded = _load_ic_lora(model, ic_lora_name, ic_lora_strength)

        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"].clone()
        noise_mask = _clone_noise_mask(latent, latent_image)
        if float(scale_by or 1.0) != 1.0:
            _b, _c, _f, h, w = latent_image.shape
            latent_image, noise_mask = _resize_latent_spatial(latent_image, noise_mask, max(1, round(w * float(scale_by))), max(1, round(h * float(scale_by))), str(upscale_method or "bicubic"))
        if auto_snap_ic_grid and lora_loaded:
            latent_image, noise_mask = _snap_latent_grid(latent_image, noise_mask, latent_downscale_factor, str(upscale_method or "bicubic"))

        time_scale = int(scale_factors[0])
        _batch, _channels, latent_length, latent_height, latent_width = latent_image.shape
        initial_latent_length = int(latent_length)
        retake_active = bool(retake_mode) or _retake_active(timeline)

        if retake_active:
            target_width = int(latent_width * 32)
            target_height = int(latent_height * 32)
            retake_start = _safe_int(timeline.get("retakeStart", timeline.get("retake_start", 0)), 0)
            retake_len = _safe_int(timeline.get("retakeLength", timeline.get("retake_length", 0)), 0)
            retake_strength = _safe_float(timeline.get("retakeStrength", timeline.get("retake_strength", 1.0)), 1.0)
            start_frame = _safe_int((guide_data or {}).get("start_frame", 0), 0) if isinstance(guide_data, dict) else 0
            relative_start = max(0, retake_start - start_frame)
            l_start = min(latent_length, relative_start // time_scale)
            l_end = min(latent_length, int(math.ceil((relative_start + retake_len) / float(time_scale))))
            is_empty_latent = bool(latent_image.abs().max().item() < 1e-5)
            need_base_video = not (not is_empty_latent and l_start == 0 and l_end >= latent_length)
            retake_video = _retake_video(timeline)
            video_file = retake_video.get("imageFile") or retake_video.get("videoFile") or retake_video.get("fileName") or ""
            if not video_file and not retake_video and motion_segments:
                video_file = motion_segments[0].get("videoFile", "")
            if need_base_video and video_file:
                frames = _load_video_frames(video_file, start_frame, (latent_length - 1) * time_scale + 1, float((motion_guide_data or {}).get("frame_rate", (guide_data or {}).get("frame_rate", 24))), "nearest")
                retake_resize = "pad" if resize_method == "maintain aspect ratio" else resize_method
                pixels = _resize_image(frames, target_width, target_height, retake_resize, 1)
                keep = ((int(pixels.shape[0]) - 1) // time_scale) * time_scale + 1
                pixels = pixels[:keep, :, :, :3]
                if use_tiled_encode:
                    base_latent = vae.encode_tiled(pixels, tile_x=int(tile_size), tile_y=int(tile_size), overlap=int(tile_overlap))
                else:
                    base_latent = vae.encode(pixels)
                base_latent = base_latent.to(device=latent_image.device, dtype=latent_image.dtype)
                paste_len = min(int(base_latent.shape[2]), int(latent_length))
                if is_empty_latent:
                    latent_image[:, :, :paste_len] = base_latent[:, :, :paste_len]
                else:
                    if l_start > 0:
                        latent_image[:, :, :l_start] = base_latent[:, :, :l_start]
                    if l_end < paste_len:
                        latent_image[:, :, l_end:paste_len] = base_latent[:, :, l_end:paste_len]
            noise_mask = torch.zeros_like(noise_mask)
            if l_end > l_start:
                noise_mask[:, :, l_start:l_end] = float(retake_strength)
            crop_frames = max(0, int(latent_image.shape[2]) - initial_latent_length)
            positive = node_helpers.conditioning_set_values(positive, {"nghtdrp_guide_crop_latent_frames": crop_frames})
            negative = node_helpers.conditioning_set_values(negative, {"nghtdrp_guide_crop_latent_frames": crop_frames})
            return positive, negative, {"samples": latent_image, "noise_mask": noise_mask}, model, float(latent_downscale_factor)

        images = (guide_data or {}).get("images", []) if isinstance(guide_data, dict) else []
        insert_frames = (guide_data or {}).get("insert_frames", []) if isinstance(guide_data, dict) else []
        strengths = (guide_data or {}).get("strengths", []) if isinstance(guide_data, dict) else []
        for index, img_tensor in enumerate(images):
            if not torch.is_tensor(img_tensor):
                continue
            strength = float(strengths[index] if index < len(strengths) else 1.0)
            if strength <= 0.0:
                continue
            frame = int(insert_frames[index] if index < len(insert_frames) else 0)
            target_w = int(latent_width * 32)
            target_h = int(latent_height * 32)
            if int(img_tensor.shape[2]) != target_w or int(img_tensor.shape[1]) != target_h:
                img_tensor = comfy.utils.common_upscale(img_tensor.permute(0, 3, 1, 2), target_w, target_h, str(upscale_method or "bicubic"), "disabled").permute(0, 2, 3, 1)
            image_pixels, guide_latent = nodes_lt.LTXVAddGuide.encode(vae, latent_width, latent_height, img_tensor, scale_factors)
            frame_idx, latent_idx = nodes_lt.LTXVAddGuide.get_latent_index(positive, latent_length, len(image_pixels), int(frame), scale_factors)
            if latent_idx >= latent_length:
                continue
            max_frames = latent_length - latent_idx
            if guide_latent.shape[2] > max_frames:
                guide_latent = guide_latent[:, :, :max_frames]
            if guide_latent.shape[2] <= 0:
                continue
            tokens_added = guide_latent.shape[2] * guide_latent.shape[3] * guide_latent.shape[4]
            guide_shape = list(guide_latent.shape[2:])
            positive, negative, latent_image, noise_mask = nodes_lt.LTXVAddGuide.append_keyframe(
                positive, negative, frame_idx, latent_image, noise_mask, guide_latent, strength, scale_factors
            )
            if lora_loaded:
                positive = _append_attention_entry(positive, tokens_added, guide_shape, image_attention_strength)
                negative = _append_attention_entry(negative, tokens_added, guide_shape, image_attention_strength)

        guide_fps = float((motion_guide_data or {}).get("frame_rate", (guide_data or {}).get("frame_rate", 24)) if isinstance(motion_guide_data, dict) or isinstance(guide_data, dict) else 24)
        for seg in motion_segments:
            if not isinstance(seg, dict):
                continue
            video_file = _segment_media_path(seg, prefer_video=True)
            if not video_file:
                continue
            start = _safe_int(seg.get("start", 0), 0)
            length = max(1, _safe_int(seg.get("length", 1), 1))
            trim = max(0, _safe_int(seg.get("trimStart", seg.get("trim_start", 0)), 0))
            strength = _safe_float(seg.get("videoStrength", seg.get("strength", 1.0)), 1.0)
            attention = _safe_float(seg.get("videoAttentionStrength", seg.get("attention_strength", 0.65)), 0.65)
            if strength <= 0.0:
                continue
            frames = _load_video_frames(video_file, trim, length, guide_fps, str(seg.get("resampleMode", "nearest") or "nearest"))
            keep = ((int(frames.shape[0]) - 1) // time_scale) * time_scale + 1
            frames = frames[:keep]
            causal_fix = start == 0 or int(frames.shape[0]) == 1
            encode_frames = frames if causal_fix else torch.cat([frames[:1], frames], dim=0)
            guide_latent = _encode_motion_video_guide(vae, latent_width, latent_height, encode_frames, scale_factors, latent_downscale_factor, crop, use_tiled_encode, tile_size, tile_overlap, resize_method)
            guide_latent = guide_latent.to(device=latent_image.device, dtype=latent_image.dtype)
            if not causal_fix:
                guide_latent = guide_latent[:, :, 1:, :, :]
            frame_idx = start
            latent_idx = (frame_idx + time_scale - 1) // time_scale if frame_idx > 0 else 0
            if latent_idx >= latent_length:
                continue
            if start > 0 and guide_latent.shape[2] > 1:
                guide_latent = guide_latent[:, :, 1:, :, :]
                frame_idx += time_scale
                latent_idx += 1
                if latent_idx >= latent_length:
                    continue
            max_frames = latent_length - latent_idx
            if guide_latent.shape[2] > max_frames:
                guide_latent = guide_latent[:, :, :max_frames]
            if guide_latent.shape[2] <= 0:
                continue
            guide_shape = list(guide_latent.shape[2:])
            guide_mask = torch.ones((guide_latent.shape[0], 1, guide_latent.shape[2], guide_latent.shape[3], guide_latent.shape[4]), device=guide_latent.device, dtype=guide_latent.dtype)
            if start > 0:
                for ramp_index, ramp in enumerate((0.25, 0.65)):
                    if ramp_index < guide_mask.shape[2]:
                        guide_mask[:, :, ramp_index, :, :] = 1.0 + strength * (1.0 - ramp)
            ldf = int(max(1, round(float(latent_downscale_factor))))
            if ldf > 1:
                guide_latent, guide_mask = _dilate_latent(guide_latent, guide_mask, ldf, ldf, latent_height, latent_width)
            tokens_added = guide_latent.shape[2] * guide_latent.shape[3] * guide_latent.shape[4]
            positive, negative, latent_image, noise_mask = nodes_lt.LTXVAddGuide.append_keyframe(
                positive,
                negative,
                frame_idx,
                latent_image,
                noise_mask,
                guide_latent,
                strength,
                scale_factors,
                guide_mask=guide_mask,
                latent_downscale_factor=float(latent_downscale_factor),
                causal_fix=causal_fix,
            )
            if lora_loaded:
                positive = _append_attention_entry(positive, tokens_added, guide_shape, attention)
                negative = _append_attention_entry(negative, tokens_added, guide_shape, attention)

        crop_frames = max(0, int(latent_image.shape[2]) - initial_latent_length)
        positive = node_helpers.conditioning_set_values(positive, {"nghtdrp_guide_crop_latent_frames": crop_frames})
        negative = node_helpers.conditioning_set_values(negative, {"nghtdrp_guide_crop_latent_frames": crop_frames})
        return positive, negative, {"samples": latent_image, "noise_mask": noise_mask}, model, float(latent_downscale_factor)


def _conditioning_get_value(conditioning: Any, key: str, fallback: Any = None) -> Any:
    for item in conditioning or []:
        meta = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 and isinstance(item[1], dict) else {}
        if key in meta and meta.get(key) is not None:
            return meta.get(key)
    return fallback


def _crop_frame_count(conditioning: Any) -> int:
    explicit = _conditioning_get_value(conditioning, "nghtdrp_guide_crop_latent_frames", None)
    if explicit is not None:
        return max(0, _safe_int(explicit, 0))
    keyframe_idxs = _conditioning_get_value(conditioning, "keyframe_idxs", None)
    try:
        return int(torch.unique(keyframe_idxs[:, 0, :, 0]).shape[0]) if keyframe_idxs is not None else 0
    except Exception:
        return 0


class IAMCCS_CineShotboardV4CropGuides:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "IAMCCS/Cine/Shotboard V4"

    def execute(self, positive, negative, latent):
        import node_helpers

        latent_image = latent["samples"].clone()
        noise_mask = _clone_noise_mask(latent, latent_image)
        crop_frames = min(_crop_frame_count(positive), max(0, int(latent_image.shape[2]) - 1))
        if crop_frames > 0:
            latent_image = latent_image[:, :, :-crop_frames]
            noise_mask = noise_mask[:, :, :-crop_frames]
        clear_values = {
            "keyframe_idxs": None,
            "guide_attention_entries": None,
            "nghtdrp_guide_crop_latent_frames": None,
        }
        positive = node_helpers.conditioning_set_values(positive, clear_values)
        negative = node_helpers.conditioning_set_values(negative, clear_values)
        return positive, negative, {"samples": latent_image, "noise_mask": noise_mask}


# ---------------------------------------------------------------------------
# LTX Duration Crop — trims the trailing corrupted 8n+1 padding frames.
# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
# ---------------------------------------------------------------------------

def _ltx_crop_audio(audio: Any, frame_count: int, fps: float) -> Any:
    """Proportionally trim audio dict to match a cropped frame count.
    By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    """
    if not isinstance(audio, dict) or audio.get("waveform") is None:
        return audio
    waveform = audio.get("waveform")
    if not torch.is_tensor(waveform):
        return audio
    sample_rate = int(audio.get("sample_rate") or 44100)
    target_samples = max(1, int(round((max(1, int(frame_count)) / max(1.0, float(fps))) * sample_rate)))
    if int(waveform.shape[-1]) <= target_samples:
        return audio
    trimmed = dict(audio)
    trimmed["waveform"] = waveform[..., :target_samples].contiguous()
    return trimmed


class IAMCCS_LTXVideoDurationCrop:
    """Trims LTX 8n+1 padding artifacts from a generated video.

    LTX requires frame counts of the form 8n+1, so the sampler always
    generates ceil((raw-1)/8)*8+1 frames. The frames beyond raw_duration
    are outside the conditioning window and appear corrupted/noisy.

    This node crops the decoded VIDEO back to the original target stored in
    guide_data["duration_frames"], which is the pre-rounding value written
    by both IAMCCS_CineFilmmakerBackend (V3) and IAMCCS_CineShotboardV4Backend (V4).

    Place this node AFTER VAE decode / VideoCombine and BEFORE SaveVideo
    or IAMCCS_ShotboardVideoEditorV1.

    By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    """

    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
            },
            "optional": {
                "guide_data": ("GUIDE_DATA",),
                "manual_duration_frames": ("INT", {
                    "default": 0, "min": 0, "max": 100000, "step": 1,
                    "tooltip": (
                        "Manual override: if > 0, trim to this exact frame count. "
                        "If 0, reads automatically from guide_data['duration_frames']."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "execute"
    CATEGORY = "IAMCCS/Cine/Shotboard V4"

    def execute(self, video, guide_data=None, manual_duration_frames=0):
        # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        try:
            from fractions import Fraction as _Fraction
            from comfy_api.latest import InputImpl, Types as _ComfyTypes
        except Exception as exc:
            print(f"[IAMCCS LTXVideoDurationCrop] comfy_api.latest unavailable: {exc}. Video passed through unchanged.")
            return (video,)

        # Resolve target frame count: manual wins over guide_data
        # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        target_frames = max(0, int(manual_duration_frames or 0))
        if target_frames <= 0 and isinstance(guide_data, dict):
            target_frames = max(0, int(guide_data.get("duration_frames", 0) or 0))

        if target_frames <= 0:
            print(
                "[IAMCCS LTXVideoDurationCrop] No target duration found "
                "(guide_data missing or manual_duration_frames=0). Video passed through unchanged."
            )
            return (video,)

        # Extract VIDEO components
        try:
            comp = video.get_components()
        except Exception as exc:
            print(f"[IAMCCS LTXVideoDurationCrop] Cannot read VIDEO components: {exc}. Video passed through.")
            return (video,)

        images = comp.images  # [N, H, W, C]
        audio = comp.audio
        frame_rate = comp.frame_rate
        total_frames = int(images.shape[0])

        if total_frames <= target_frames:
            print(
                f"[IAMCCS LTXVideoDurationCrop] No trim needed: "
                f"video has {total_frames} frames, target={target_frames}. "
                "(LTX padding already aligned or guide_data duration >= video length.)"
            )
            return (video,)

        # Perform the trim
        # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        trimmed_images = images[:target_frames].contiguous()
        fps = float(frame_rate or 24.0)
        # _ltx_crop_audio is a no-op if audio is already <= target length.
        # Since backends now build audio_out for raw_duration, audio is already aligned.
        trimmed_audio = _ltx_crop_audio(audio, target_frames, fps)

        audio_trimmed = False
        if isinstance(trimmed_audio, dict) and isinstance(audio, dict):
            orig_wav = audio.get("waveform")
            new_wav = trimmed_audio.get("waveform")
            if torch.is_tensor(orig_wav) and torch.is_tensor(new_wav):
                audio_trimmed = int(new_wav.shape[-1]) < int(orig_wav.shape[-1])

        print(
            f"[IAMCCS LTXVideoDurationCrop] Trimmed LTX padding: "
            f"{total_frames} -> {target_frames} frames "
            f"(removed {total_frames - target_frames} corrupted padding frame(s), "
            f"ltxv_length={total_frames}, raw_duration={target_frames}, "
            f"audio_trimmed={audio_trimmed})."
        )

        out_video = InputImpl.VideoFromComponents(_ComfyTypes.VideoComponents(
            images=trimmed_images,
            audio=trimmed_audio,
            frame_rate=_Fraction(round(max(1.0, fps) * 1000), 1000),
        ))
        return (out_video,)


NODE_CLASS_MAPPINGS = {
    "IAMCCS_CineShotboardV4Backend": IAMCCS_CineShotboardV4Backend,
    "IAMCCS_CineShotboardV4Guide": IAMCCS_CineShotboardV4Guide,
    "IAMCCS_CineShotboardV4CropGuides": IAMCCS_CineShotboardV4CropGuides,
    "IAMCCS_LTXVideoDurationCrop": IAMCCS_LTXVideoDurationCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_CineShotboardV4Backend": "IAMCCS Cine Shotboard V4 Backend",
    "IAMCCS_CineShotboardV4Guide": "IAMCCS Cine Shotboard V4 Guide",
    "IAMCCS_CineShotboardV4CropGuides": "IAMCCS Cine Shotboard V4 Crop Guides",
    "IAMCCS_LTXVideoDurationCrop": "IAMCCS LTX Video Duration Crop",
}
