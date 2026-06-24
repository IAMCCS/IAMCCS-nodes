from __future__ import annotations

import copy
import json
from typing import Any, Dict, List


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
    return json.dumps(value, indent=2, ensure_ascii=False)


def _clone_linx(cine_linx: Any) -> Dict[str, Any]:
    if isinstance(cine_linx, dict):
        return copy.deepcopy(cine_linx)
    return {
        "type": SUPERNODE_LINX_TYPE,
        "mode": "iamccs_cine_motion_sketch",
        "resources": {},
        "outputs": {},
        "chain": [],
        "stages": [],
    }


def _resources(cine_linx: Dict[str, Any]) -> Dict[str, Any]:
    resources = cine_linx.setdefault("resources", {})
    if not isinstance(resources, dict):
        resources = {}
        cine_linx["resources"] = resources
    return resources


def _outputs(cine_linx: Dict[str, Any]) -> Dict[str, Any]:
    outputs = cine_linx.setdefault("outputs", {})
    if not isinstance(outputs, dict):
        outputs = {}
        cine_linx["outputs"] = outputs
    return outputs


def _payload(cine_linx: Dict[str, Any]) -> Dict[str, Any]:
    resources = _resources(cine_linx)
    payload = resources.get("cine_payload")
    if not isinstance(payload, dict):
        payload = {}
        resources["cine_payload"] = payload
    return payload


def _safe_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return int(fallback)


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(fallback)


def _refresh_index(cine_linx: Dict[str, Any]) -> None:
    resources = _resources(cine_linx)
    cine_linx["resource_keys"] = sorted(resources.keys())
    cine_linx["resource_types"] = {key: type(value).__name__ for key, value in resources.items()}


def _shotboard_segments(cine_linx: Dict[str, Any]) -> List[Dict[str, Any]]:
    payload = _payload(cine_linx)
    candidates = [
        payload.get("visual_segments"),
        payload.get("segments"),
        payload.get("timeline_segments"),
        payload.get("rows"),
    ]
    for candidate in candidates:
        if isinstance(candidate, list):
            out: List[Dict[str, Any]] = []
            for index, item in enumerate(candidate):
                if not isinstance(item, dict):
                    continue
                seg = copy.deepcopy(item)
                seg["id"] = str(seg.get("id") or seg.get("segment_id") or f"shot_{index + 1:03d}")
                seg["start"] = max(0, _safe_int(seg.get("start", seg.get("frame", 0)), 0))
                seg["length"] = max(1, _safe_int(seg.get("length", seg.get("frames", 1)), 1))
                seg["label"] = str(seg.get("label") or seg.get("name") or seg.get("id"))
                out.append(seg)
            if out:
                return sorted(out, key=lambda seg: _safe_int(seg.get("start"), 0))
    return []


def _duration_frames(cine_linx: Dict[str, Any], sketch: Dict[str, Any]) -> int:
    payload = _payload(cine_linx)
    candidates = [
        sketch.get("duration_frames"),
        payload.get("duration_frames"),
        payload.get("max_frames"),
        cine_linx.get("duration_frames"),
    ]
    for value in candidates:
        frames = _safe_int(value, 0)
        if frames > 0:
            return frames
    segments = _shotboard_segments(cine_linx)
    return max([_safe_int(seg.get("start"), 0) + _safe_int(seg.get("length"), 1) for seg in segments] or [1])


def _normalize_strokes(raw: Any, segments: List[Dict[str, Any]], frame_rate: float) -> List[Dict[str, Any]]:
    source = raw if isinstance(raw, list) else []
    segment_by_id = {str(seg.get("id")): seg for seg in segments}
    out: List[Dict[str, Any]] = []
    for index, item in enumerate(source):
        if not isinstance(item, dict):
            continue
        stroke = copy.deepcopy(item)
        segment_id = str(stroke.get("segment_id") or stroke.get("segmentId") or "")
        segment = segment_by_id.get(segment_id) if segment_id else None
        start = _safe_int(stroke.get("start_frame", stroke.get("start", segment.get("start", 0) if segment else 0)), 0)
        length = _safe_int(stroke.get("length", stroke.get("length_frames", segment.get("length", max(1, round(frame_rate))) if segment else max(1, round(frame_rate)))), 1)
        stroke["id"] = str(stroke.get("id") or f"stroke_{index + 1:03d}")
        stroke["segment_id"] = segment_id or str(segment.get("id")) if segment else ""
        stroke["track"] = str(stroke.get("track") or "camera_path")
        stroke["mode"] = str(stroke.get("mode") or "motion_track")
        stroke["scope"] = str(stroke.get("scope") or "slot_only")
        stroke["start_frame"] = max(0, start)
        stroke["length_frames"] = max(1, length)
        stroke["strength"] = max(0.0, min(1.0, _safe_float(stroke.get("strength", 0.75), 0.75)))
        stroke["attention_strength"] = max(0.0, min(1.0, _safe_float(stroke.get("attention_strength", 0.65), 0.65)))
        stroke["radius"] = max(1.0, _safe_float(stroke.get("radius", 28), 28))
        stroke["falloff"] = max(0.0, min(1.0, _safe_float(stroke.get("falloff", 0.35), 0.35)))
        stroke["easing"] = str(stroke.get("easing") or "ease_in_out")
        points = stroke.get("points")
        if not isinstance(points, list):
            points = []
        stroke["points"] = [
            [
                max(0.0, min(1.0, _safe_float(point[0], 0.0))),
                max(0.0, min(1.0, _safe_float(point[1], 0.0))),
            ]
            for point in points
            if isinstance(point, list) and len(point) >= 2
        ]
        out.append(stroke)
    return out


def _track_color(track: str) -> tuple:
    palette = {
        "camera_path": (0, 170, 255),
        "subject_path": (255, 80, 130),
        "object_path": (255, 198, 84),
        "background_lock": (80, 245, 170),
        "attention_mask": (160, 105, 255),
    }
    return palette.get(str(track or ""), (120, 220, 255))


def _point_xy(point: Any) -> tuple:
    if isinstance(point, dict):
        return (
            max(0.0, min(1.0, _safe_float(point.get("x"), 0.0))),
            max(0.0, min(1.0, _safe_float(point.get("y"), 0.0))),
        )
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        return (
            max(0.0, min(1.0, _safe_float(point[0], 0.0))),
            max(0.0, min(1.0, _safe_float(point[1], 0.0))),
        )
    return (0.0, 0.0)


def _stroke_points_px(stroke: Dict[str, Any], width: int, height: int) -> List[tuple]:
    points = stroke.get("points")
    if not isinstance(points, list):
        return []
    out = []
    for point in points:
        x, y = _point_xy(point)
        out.append((int(round(x * max(1, width - 1))), int(round(y * max(1, height - 1)))))
    return out


def _motion_parts_from_plan(plan: Dict[str, Any], fallback_strength: float = 0.75, fallback_attention: float = 0.65) -> List[Dict[str, Any]]:
    existing = plan.get("motionParts")
    if isinstance(existing, list) and existing:
        return [copy.deepcopy(part) for part in existing if isinstance(part, dict)]

    duration_frames = max(1, _safe_int(plan.get("duration_frames"), 1))
    strokes = plan.get("strokes") if isinstance(plan.get("strokes"), list) else []
    segments = plan.get("shotboard_segments") if isinstance(plan.get("shotboard_segments"), list) else []
    segments = sorted([seg for seg in segments if isinstance(seg, dict)], key=lambda seg: _safe_int(seg.get("start"), 0))
    segment_by_id = {str(seg.get("id") or seg.get("segment_id") or ""): seg for seg in segments}
    def scoped_range(stroke: Dict[str, Any]) -> tuple:
        start = max(0, _safe_int(stroke.get("start_frame"), 0))
        length = max(1, _safe_int(stroke.get("length_frames"), duration_frames))
        segment_id = str(stroke.get("segment_id") or "")
        segment = segment_by_id.get(segment_id)
        if segment:
            start = max(0, _safe_int(segment.get("start"), start))
            length = max(1, _safe_int(segment.get("length"), length))
        scope = str(stroke.get("scope") or "slot_only")
        if scope == "continue_to_next" and segment:
            try:
                idx = segments.index(segment)
            except ValueError:
                idx = -1
            if idx >= 0 and idx + 1 < len(segments):
                next_seg = segments[idx + 1]
                end = max(start + length, _safe_int(next_seg.get("start"), start) + _safe_int(next_seg.get("length"), 1))
                length = max(1, end - start)
        elif scope == "hold_last" and segment:
            later = [seg for seg in segments if _safe_int(seg.get("start"), 0) > start]
            end = _safe_int(later[0].get("start"), start + length) if later else duration_frames
            length = max(1, end - start)
        elif scope == "cut_reset":
            length = max(1, length)
        return start, length, scope
    if not strokes:
        return [{
            "id": "motion_part_001",
            "type": "motion_control",
            "start": 0,
            "length": duration_frames,
            "trimStart": 0,
            "videoStrength": max(0.0, min(1.0, _safe_float(fallback_strength, 0.75))),
            "videoAttentionStrength": max(0.0, min(1.0, _safe_float(fallback_attention, 0.65))),
            "resampleMode": "nearest",
            "source": "IAMCCS_MotionGuideBridge",
        }]

    parts_by_key: Dict[str, Dict[str, Any]] = {}
    for index, stroke in enumerate(strokes):
        if not isinstance(stroke, dict):
            continue
        start, length, scope = scoped_range(stroke)
        key = str(stroke.get("segment_id") or f"{start}:{length}")
        if scope != "slot_only":
            key = f"{key}:{scope}"
        part = parts_by_key.get(key)
        if not part:
            part = {
                "id": f"motion_part_{len(parts_by_key) + 1:03d}",
                "type": "motion_control",
                "segment_id": str(stroke.get("segment_id") or ""),
                "scope": scope,
                "start": start,
                "length": length,
                "trimStart": start,
                "videoStrength": max(0.0, min(1.0, _safe_float(stroke.get("strength"), fallback_strength))),
                "videoAttentionStrength": max(0.0, min(1.0, _safe_float(stroke.get("attention_strength"), fallback_attention))),
                "resampleMode": "nearest",
                "source": "IAMCCS_MotionGuideBridge",
                "tracks": [],
            }
            parts_by_key[key] = part
        part["start"] = min(_safe_int(part.get("start"), start), start)
        part_end = max(_safe_int(part.get("start"), start) + _safe_int(part.get("length"), length), start + length)
        part["length"] = max(1, part_end - _safe_int(part.get("start"), start))
        track = str(stroke.get("track") or "camera_path")
        if track not in part["tracks"]:
            part["tracks"].append(track)
    return sorted(parts_by_key.values(), key=lambda item: _safe_int(item.get("start"), 0))


def _stroke_scoped_range(plan: Dict[str, Any], stroke: Dict[str, Any]) -> tuple:
    duration_frames = max(1, _safe_int(plan.get("duration_frames"), 1))
    segments = plan.get("shotboard_segments") if isinstance(plan.get("shotboard_segments"), list) else []
    segments = sorted([seg for seg in segments if isinstance(seg, dict)], key=lambda seg: _safe_int(seg.get("start"), 0))
    segment_by_id = {str(seg.get("id") or seg.get("segment_id") or ""): seg for seg in segments}
    start = max(0, _safe_int(stroke.get("start_frame"), 0))
    length = max(1, _safe_int(stroke.get("length_frames"), duration_frames))
    scope = str(stroke.get("scope") or "slot_only")
    segment = segment_by_id.get(str(stroke.get("segment_id") or ""))
    if segment:
        start = max(0, _safe_int(segment.get("start"), start))
        length = max(1, _safe_int(segment.get("length"), length))
    if scope == "continue_to_next" and segment:
        try:
            idx = segments.index(segment)
        except ValueError:
            idx = -1
        if idx >= 0 and idx + 1 < len(segments):
            next_seg = segments[idx + 1]
            end = max(start + length, _safe_int(next_seg.get("start"), start) + _safe_int(next_seg.get("length"), 1))
            length = max(1, end - start)
    elif scope == "hold_last" and segment:
        later = [seg for seg in segments if _safe_int(seg.get("start"), 0) > start]
        end = _safe_int(later[0].get("start"), start + length) if later else duration_frames
        length = max(1, end - start)
    return start, length, scope


class IAMCCS_CineMotionSketch:
    """Shotboard-synced motion sketch layer for IC-LoRA control planning."""

    DEFAULT_DATA = _json_dumps({
        "schema": "iamccs.shotboard_v4.motion_sketch",
        "schema_version": 1,
        "frame_rate": 24.0,
        "control_family": "auto",
        "render_mode": "motion_track_control",
        "resize_method": "crop",
        "strokes": [],
        "view": {
            "selected_segment_id": "",
            "show_camera_path": True,
            "show_subject_path": True,
            "show_background_lock": True,
            "snap_to_shotboard": True,
        },
    })

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion_sketch_data": ("STRING", {
                    "default": cls.DEFAULT_DATA,
                    "multiline": True,
                    "tooltip": "Edited by IAMCCS CineMotionSketch UI. Stores camera/subject/object/background-lock strokes in normalized image coordinates.",
                }),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "control_family": (["auto", "motion_track", "union_control", "depth", "edge", "pose", "hybrid"], {"default": "auto"}),
                "sync_policy": (["shotboard_realtime", "metadata_only"], {"default": "shotboard_realtime"}),
            },
            "optional": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE, "STRING", "STRING")
    RETURN_NAMES = ("cine_linx", "motion_sketch_json", "report")
    FUNCTION = "sketch"
    OUTPUT_NODE = True
    CATEGORY = "IAMCCS/Cine/Shotboard V4"

    def sketch(self, motion_sketch_data, frame_rate, control_family, sync_policy, cine_linx=None, unique_id=None, extra_pnginfo=None):
        linx = _clone_linx(cine_linx)
        resources = _resources(linx)
        outputs = _outputs(linx)
        payload = _payload(linx)

        sketch = _json_loads(motion_sketch_data, {})
        if not isinstance(sketch, dict):
            sketch = {}
        segments = _shotboard_segments(linx)
        fps = max(1.0, _safe_float(frame_rate, _safe_float(sketch.get("frame_rate", 24.0), 24.0)))
        strokes = _normalize_strokes(sketch.get("strokes"), segments, fps)
        duration_frames = _duration_frames(linx, sketch)

        normalized = {
            "schema": "iamccs.shotboard_v4.motion_sketch",
            "schema_version": 1,
            "source_node": "IAMCCS_CineMotionSketch",
            "frame_rate": fps,
            "duration_frames": duration_frames,
            "control_family": str(control_family or sketch.get("control_family") or "auto"),
            "render_mode": str(sketch.get("render_mode") or "motion_track_control"),
            "resize_method": str(sketch.get("resize_method") or payload.get("resize_method") or "crop"),
            "sync_policy": str(sync_policy or "shotboard_realtime"),
            "shotboard_segments": segments,
            "strokes": strokes,
            "view": sketch.get("view") if isinstance(sketch.get("view"), dict) else {},
        }

        resources["cine_motion_sketch"] = normalized
        resources["cine_motion_sketch_json"] = _json_dumps(normalized)
        outputs["motion_sketch_json"] = resources["cine_motion_sketch_json"]
        payload["motion_sketch"] = normalized
        linx.setdefault("chain", []).append({"role": "motion_sketch", "name": "IAMCCS_CineMotionSketch"})
        _refresh_index(linx)

        report = {
            "ok": True,
            "node": "IAMCCS_CineMotionSketch",
            "segments": len(segments),
            "strokes": len(strokes),
            "duration_frames": duration_frames,
            "frame_rate": fps,
            "truth": "Motion strokes are stored in cine_linx and remain synced to Shotboard segment ids/start/length.",
        }
        return (linx, resources["cine_motion_sketch_json"], _json_dumps(report))


class IAMCCS_MotionGuideBridge:
    """Builds LTX IC-LoRA-style motion guide data from IAMCCS motion sketch metadata."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion_sketch_json": ("STRING", {"default": "{}", "multiline": True}),
                "default_video_strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "default_attention_strength": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "resample_mode": (["nearest", "linear"], {"default": "nearest"}),
                "resize_method": (["crop", "pad", "stretch to fit", "maintain aspect ratio"], {"default": "crop"}),
            },
            "optional": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "control_video_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional rendered control video path. When empty, the bridge emits a render plan but no active video segment.",
                }),
            },
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE, MOTION_GUIDE_TYPE, "STRING", "STRING")
    RETURN_NAMES = ("cine_linx", "motion_guide_data", "motion_render_plan_json", "report")
    FUNCTION = "bridge"
    OUTPUT_NODE = True
    CATEGORY = "IAMCCS/Cine/Shotboard V4"

    def bridge(
        self,
        motion_sketch_json,
        default_video_strength,
        default_attention_strength,
        resample_mode,
        resize_method,
        cine_linx=None,
        control_video_path="",
    ):
        linx = _clone_linx(cine_linx)
        resources = _resources(linx)
        outputs = _outputs(linx)
        payload = _payload(linx)

        sketch = _json_loads(motion_sketch_json, {})
        if not isinstance(sketch, dict) or not sketch:
            sketch = resources.get("cine_motion_sketch") if isinstance(resources.get("cine_motion_sketch"), dict) else {}
        if not isinstance(sketch, dict):
            sketch = {}

        fps = max(1.0, _safe_float(sketch.get("frame_rate", payload.get("frame_rate", 24.0)), 24.0))
        duration_frames = max(1, _safe_int(sketch.get("duration_frames", payload.get("duration_frames", payload.get("max_frames", 1))), 1))
        strokes = sketch.get("strokes") if isinstance(sketch.get("strokes"), list) else []
        family = str(sketch.get("control_family") or "auto")

        render_plan = {
            "schema": "iamccs.shotboard_v4.motion_control_render_plan",
            "schema_version": 1,
            "source_node": "IAMCCS_MotionGuideBridge",
            "frame_rate": fps,
            "duration_frames": duration_frames,
            "control_family": family,
            "render_mode": sketch.get("render_mode") or "motion_track_control",
            "resize_method": resize_method,
            "shotboard_segments": sketch.get("shotboard_segments") if isinstance(sketch.get("shotboard_segments"), list) else [],
            "strokes": strokes,
            "target": {
                "recommended_lora": "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors" if family in {"auto", "motion_track"} else "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
                "backend_contract": "Render strokes to an IMAGE batch/video and feed it to an IC-LoRA guide node before sampling.",
            },
        }
        motion_parts = _motion_parts_from_plan(render_plan, default_video_strength, default_attention_strength)
        for part in motion_parts:
            part["videoStrength"] = max(0.0, min(1.0, _safe_float(part.get("videoStrength"), default_video_strength)))
            part["videoAttentionStrength"] = max(0.0, min(1.0, _safe_float(part.get("videoAttentionStrength"), default_attention_strength)))
            part["resampleMode"] = str(part.get("resampleMode") or resample_mode or "nearest")
        render_plan["motionParts"] = motion_parts

        control_path = str(control_video_path or "").strip()
        active_segments: List[Dict[str, Any]] = []
        if control_path:
            for part in motion_parts:
                active_segments.append({
                "id": str(part.get("id") or f"motion_part_{len(active_segments) + 1:03d}"),
                "type": "motion_control",
                "start": max(0, _safe_int(part.get("start"), 0)),
                "length": max(1, _safe_int(part.get("length"), duration_frames)),
                "trimStart": max(0, _safe_int(part.get("trimStart"), 0)),
                "videoFile": control_path,
                "videoStrength": max(0.0, min(1.0, _safe_float(part.get("videoStrength"), default_video_strength))),
                "videoAttentionStrength": max(0.0, min(1.0, _safe_float(part.get("videoAttentionStrength"), default_attention_strength))),
                "resampleMode": str(part.get("resampleMode") or resample_mode or "nearest"),
                "source": "IAMCCS_MotionGuideBridge",
            })

        motion_guide_data = {
            "segments": active_segments,
            "motionParts": motion_parts,
            "frame_rate": fps,
            "duration_frames": duration_frames,
            "resize_method": resize_method,
            "render_plan": render_plan,
            "schema": "iamccs.motion_guide_data.iclora_compatible",
        }

        resources["cine_motion_parts"] = motion_parts
        resources["cine_motion_parts_json"] = _json_dumps(motion_parts)
        resources["cine_motion_render_plan"] = render_plan
        resources["cine_motion_render_plan_json"] = _json_dumps(render_plan)
        resources["cine_motion_guide_data"] = motion_guide_data
        resources["cine_motion_guide_data_json"] = _json_dumps(motion_guide_data)
        outputs["motion_render_plan_json"] = resources["cine_motion_render_plan_json"]
        outputs["motion_guide_data_json"] = resources["cine_motion_guide_data_json"]
        payload["motion_guide_data"] = motion_guide_data
        linx.setdefault("chain", []).append({"role": "motion_guide_bridge", "name": "IAMCCS_MotionGuideBridge"})
        _refresh_index(linx)

        report = {
            "ok": True,
            "node": "IAMCCS_MotionGuideBridge",
            "active_motion_segments": len(active_segments),
            "motionParts": len(motion_parts),
            "strokes": len(strokes),
            "has_control_video": bool(control_path),
            "contract": "IAMCCS uses motionParts as authoring truth; MOTION_GUIDE_DATA.segments is emitted only as the executable guide shape expected by LTX IC-LoRA guide backends.",
            "next_step": "Render motion_render_plan_json to control frames and connect IAMCCS Motion Parts IC-LoRA Apply.",
        }
        return (linx, motion_guide_data, resources["cine_motion_render_plan_json"], _json_dumps(report))


class IAMCCS_MotionSketchRenderer:
    """Renders MotionSketch strokes into IC-LoRA-friendly control frames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion_render_plan_json": ("STRING", {"default": "{}", "multiline": True}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 288, "min": 64, "max": 2048, "step": 8}),
                "max_render_frames": ("INT", {"default": 240, "min": 1, "max": 2048, "step": 1}),
                "line_width": ("INT", {"default": 8, "min": 1, "max": 96, "step": 1}),
                "trail_frames": ("INT", {"default": 12, "min": 0, "max": 240, "step": 1}),
                "background_level": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "glow": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("control_frames", "control_video_hint_json", "report")
    FUNCTION = "render"
    OUTPUT_NODE = True
    CATEGORY = "IAMCCS/Cine/Shotboard V4"

    def render(
        self,
        motion_render_plan_json,
        width,
        height,
        max_render_frames,
        line_width,
        trail_frames,
        background_level,
        glow,
    ):
        try:
            import numpy as np
            import torch
            from PIL import Image, ImageDraw, ImageFilter
        except Exception as exc:
            raise RuntimeError(f"IAMCCS MotionSketch renderer requires torch, numpy and PIL: {exc}") from exc

        plan = _json_loads(motion_render_plan_json, {})
        if not isinstance(plan, dict):
            plan = {}
        strokes = plan.get("strokes") if isinstance(plan.get("strokes"), list) else []
        duration_frames = max(1, _safe_int(plan.get("duration_frames"), 1))
        frame_count = min(max(1, _safe_int(max_render_frames, 240)), duration_frames)
        width = max(64, _safe_int(width, 512))
        height = max(64, _safe_int(height, 288))
        line_width = max(1, _safe_int(line_width, 8))
        trail_frames = max(0, _safe_int(trail_frames, 12))
        bg = int(round(max(0.0, min(1.0, _safe_float(background_level, 0.0))) * 255))
        glow_amount = max(0.0, min(1.0, _safe_float(glow, 0.35)))

        frames = []
        scale = duration_frames / float(frame_count)
        for frame_index in range(frame_count):
            source_frame = int(round(frame_index * scale))
            base = Image.new("RGB", (width, height), (bg, bg, bg))
            glow_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            line_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            glow_draw = ImageDraw.Draw(glow_layer)
            line_draw = ImageDraw.Draw(line_layer)

            for stroke in strokes:
                if not isinstance(stroke, dict):
                    continue
                points = _stroke_points_px(stroke, width, height)
                if len(points) < 2:
                    continue
                start, length, scope = _stroke_scoped_range(plan, stroke)
                end = start + length
                if source_frame < start - trail_frames or source_frame > end:
                    continue
                progress = (source_frame - start) / float(length)
                progress = max(0.0, min(1.0, progress))
                if scope == "hold_last" and source_frame >= end - trail_frames:
                    progress = 1.0
                visible_count = max(2, int(round(1 + progress * (len(points) - 1))))
                visible = points[:visible_count]
                color = _track_color(str(stroke.get("track") or "camera_path"))
                strength = max(0.05, min(1.0, _safe_float(stroke.get("strength"), 0.75)))
                alpha = int(round(255 * strength))
                if str(stroke.get("track")) == "background_lock":
                    alpha = int(round(190 * strength))

                glow_width = max(line_width + 8, int(round(line_width * 2.5)))
                if glow_amount > 0:
                    glow_draw.line(visible, fill=(*color, int(alpha * glow_amount)), width=glow_width, joint="curve")
                line_draw.line(visible, fill=(*color, alpha), width=line_width, joint="curve")
                for point in visible[-3:]:
                    radius = max(3, line_width // 2)
                    line_draw.ellipse(
                        (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
                        fill=(*color, alpha),
                    )

            if glow_amount > 0:
                glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=max(1, int(round(line_width * 0.75)))))
                base = Image.alpha_composite(base.convert("RGBA"), glow_layer)
            base = Image.alpha_composite(base.convert("RGBA"), line_layer).convert("RGB")
            frames.append(np.asarray(base, dtype=np.float32) / 255.0)

        if not frames:
            frames.append(np.zeros((height, width, 3), dtype=np.float32))
        batch = torch.from_numpy(np.stack(frames, axis=0))
        hint = {
            "schema": "iamccs.shotboard_v4.motion_control_frames",
            "frame_count": int(batch.shape[0]),
            "width": width,
            "height": height,
            "source_duration_frames": duration_frames,
            "frame_sampling": "uniform_from_shotboard_duration",
            "intended_backend": "Feed control_frames into the local LTX 2.3 IC-LoRA guide node as the control image sequence.",
        }
        report = {
            "ok": True,
            "node": "IAMCCS_MotionSketchRenderer",
            "frames": int(batch.shape[0]),
            "strokes": len(strokes),
            "size": [width, height],
            "note": "Rendered actual control frames from MotionSketch strokes; this is the first executable backend layer after authoring.",
        }
        return (batch, _json_dumps(hint), _json_dumps(report))


class IAMCCS_MotionPartsICLoRAApply:
    """Applies rendered motionParts control frames to an LTX latent through the IC-LoRA guide path."""

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import folder_paths
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
                "control_frames": ("IMAGE",),
                "motion_render_plan_json": ("STRING", {"default": "{}", "multiline": True}),
                "default_strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "default_attention_strength": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "latent_downscale_factor": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 10.0, "step": 1.0}),
                "crop": (["disabled", "center"], {"default": "center"}),
                "use_tiled_encode": ("BOOLEAN", {"default": False}),
                "tile_size": ("INT", {"default": 256, "min": 64, "max": 512, "step": 32}),
                "tile_overlap": ("INT", {"default": 64, "min": 16, "max": 256, "step": 16}),
            },
            "optional": {
                "guide_data": ("GUIDE_DATA",),
                "model": ("MODEL",),
                "ic_lora_name": (["None"] + [name for name in loras if name != "None"], {"default": "None"}),
                "ic_lora_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "image_attention_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01}),
                "upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "bislerp"], {"default": "bicubic"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "MODEL", "FLOAT", "STRING")
    RETURN_NAMES = ("positive", "negative", "latent", "model", "latent_downscale_factor", "report")
    FUNCTION = "apply"
    CATEGORY = "IAMCCS/Cine/Shotboard V4"

    @staticmethod
    def _conditioning_entries(conditioning):
        for item in conditioning:
            meta = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else {}
            entries = meta.get("guide_attention_entries") if isinstance(meta, dict) else None
            if entries is not None:
                return list(entries)
        return []

    @staticmethod
    def _set_conditioning_entries(conditioning, entries):
        import node_helpers
        return node_helpers.conditioning_set_values(conditioning, {"guide_attention_entries": entries})

    @classmethod
    def _append_attention_entry(cls, conditioning, token_count, latent_shape, attention_strength):
        entries = cls._conditioning_entries(conditioning)
        entries.append({
            "pre_filter_count": int(token_count),
            "strength": float(attention_strength),
            "pixel_mask": None,
            "latent_shape": list(latent_shape),
        })
        return cls._set_conditioning_entries(conditioning, entries)

    @staticmethod
    def _resample_frames(frames, target_count, mode):
        import torch
        n = int(frames.shape[0])
        target_count = max(1, int(target_count))
        if n <= 0:
            raise ValueError("control_frames is empty")
        if n == target_count:
            return frames
        if n == 1:
            return frames.repeat(target_count, 1, 1, 1)
        positions = torch.linspace(0, n - 1, target_count, device=frames.device, dtype=torch.float32)
        if str(mode or "nearest") == "nearest":
            idx = torch.round(positions).long().clamp(0, n - 1)
            return frames.index_select(0, idx)
        idx0 = torch.floor(positions).long().clamp(0, n - 1)
        idx1 = torch.ceil(positions).long().clamp(0, n - 1)
        alpha = (positions - idx0.to(positions.dtype)).view(-1, 1, 1, 1)
        f0 = frames.index_select(0, idx0).to(torch.float32)
        f1 = frames.index_select(0, idx1).to(torch.float32)
        return (f0 * (1.0 - alpha) + f1 * alpha).to(frames.dtype)

    @staticmethod
    def _dilate_latent(samples, mask, horizontal_scale, vertical_scale):
        import torch
        if horizontal_scale == 1 and vertical_scale == 1:
            return samples, mask
        out_shape = samples.shape[:3] + (
            samples.shape[3] * vertical_scale,
            samples.shape[4] * horizontal_scale,
        )
        out = torch.zeros(out_shape, device=samples.device, dtype=samples.dtype, requires_grad=False)
        out[..., ::vertical_scale, ::horizontal_scale] = samples
        mask_shape = (samples.shape[0], 1, samples.shape[2], out_shape[3], out_shape[4])
        out_mask = torch.full(mask_shape, -1.0, device=samples.device, dtype=samples.dtype, requires_grad=False)
        out_mask[..., ::vertical_scale, ::horizontal_scale] = mask if mask is not None else 1.0
        return out, out_mask

    @staticmethod
    def _load_ic_lora(model, ic_lora_name, ic_lora_strength, fallback_downscale):
        if model is None or not ic_lora_name or ic_lora_name == "None":
            return model, fallback_downscale, False
        import comfy
        import comfy.sd
        import folder_paths
        lora_path = folder_paths.get_full_path_or_raise("loras", ic_lora_name)
        lora, metadata = comfy.utils.load_torch_file(lora_path, safe_load=True, return_metadata=True)
        try:
            downscale = float(metadata.get("reference_downscale_factor", fallback_downscale))
        except Exception:
            downscale = fallback_downscale
        if float(ic_lora_strength) != 0.0:
            model, _ = comfy.sd.load_lora_for_models(model, None, lora, float(ic_lora_strength), 0)
        return model, max(1.0, float(downscale)), True

    def apply(
        self,
        positive,
        negative,
        vae,
        latent,
        control_frames,
        motion_render_plan_json,
        default_strength,
        default_attention_strength,
        latent_downscale_factor,
        crop,
        use_tiled_encode,
        tile_size,
        tile_overlap,
        guide_data=None,
        model=None,
        ic_lora_name="None",
        ic_lora_strength=1.0,
        image_attention_strength=1.0,
        scale_by=1.0,
        upscale_method="bicubic",
    ):
        import comfy.utils
        import node_helpers
        import torch
        from comfy_extras import nodes_lt

        plan = _json_loads(motion_render_plan_json, {})
        if not isinstance(plan, dict):
            plan = {}
        motion_parts = _motion_parts_from_plan(plan, default_strength, default_attention_strength)
        if not motion_parts:
            raise ValueError("No motionParts available for IC-LoRA apply.")
        if control_frames is None or int(control_frames.shape[0]) <= 0:
            raise ValueError("control_frames is empty.")

        latent_image = latent["samples"].clone()
        noise_mask = latent.get("noise_mask")
        if noise_mask is None:
            noise_mask = torch.ones((latent_image.shape[0], 1, latent_image.shape[2], 1, 1), device=latent_image.device, dtype=torch.float32)
        else:
            noise_mask = noise_mask.clone()

        model, latent_downscale_factor, lora_loaded = self._load_ic_lora(
            model, ic_lora_name, ic_lora_strength, max(1.0, _safe_float(latent_downscale_factor, 1.0))
        )

        scale_factors = vae.downscale_index_formula
        if float(scale_by or 1.0) != 1.0:
            batch, channels, frames, height, width = latent_image.shape
            scaled_width = max(1, round(width * float(scale_by)))
            scaled_height = max(1, round(height * float(scale_by)))
            latent_4d = latent_image.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
            latent_4d = comfy.utils.common_upscale(latent_4d, scaled_width, scaled_height, str(upscale_method or "bicubic"), "disabled")
            latent_image = latent_4d.reshape(batch, frames, channels, scaled_height, scaled_width).permute(0, 2, 1, 3, 4)
            if noise_mask is not None and (noise_mask.shape[-1] > 1 or noise_mask.shape[-2] > 1):
                mask_4d = noise_mask.permute(0, 2, 1, 3, 4).reshape(batch * frames, 1, height, width)
                mask_4d = comfy.utils.common_upscale(mask_4d, scaled_width, scaled_height, str(upscale_method or "bicubic"), "disabled")
                noise_mask = mask_4d.reshape(batch, frames, 1, scaled_height, scaled_width).permute(0, 2, 1, 3, 4)
        time_scale_factor, width_scale_factor, height_scale_factor = scale_factors
        _, _, latent_length, latent_height, latent_width = latent_image.shape
        initial_latent_length = int(latent_length)
        applied = 0
        skipped = 0
        image_guides_applied = 0

        images = guide_data.get("images", []) if isinstance(guide_data, dict) else []
        insert_frames = guide_data.get("insert_frames", []) if isinstance(guide_data, dict) else []
        strengths = guide_data.get("strengths", []) if isinstance(guide_data, dict) else []
        for idx, img_tensor in enumerate(images):
            try:
                frame = insert_frames[idx] if idx < len(insert_frames) else 0
                strength = max(0.0, min(1.0, _safe_float(strengths[idx] if idx < len(strengths) else 1.0, 1.0)))
                if strength <= 0.0:
                    continue
                if not torch.is_tensor(img_tensor):
                    skipped += 1
                    continue
                target_pix_w = int(latent_width * 32)
                target_pix_h = int(latent_height * 32)
                if img_tensor.shape[2] != target_pix_w or img_tensor.shape[1] != target_pix_h:
                    img_resized = comfy.utils.common_upscale(
                        img_tensor.permute(0, 3, 1, 2),
                        target_pix_w,
                        target_pix_h,
                        str(upscale_method or "bicubic"),
                        "disabled",
                    ).permute(0, 2, 3, 1)
                    img_tensor = img_resized
                image_pixels, guide_latent = nodes_lt.LTXVAddGuide.encode(vae, latent_width, latent_height, img_tensor, scale_factors)
                frame_idx, latent_idx = nodes_lt.LTXVAddGuide.get_latent_index(positive, latent_length, len(image_pixels), int(frame), scale_factors)
                if latent_idx >= latent_length:
                    skipped += 1
                    continue
                max_frames = latent_length - latent_idx
                if guide_latent.shape[2] > max_frames:
                    guide_latent = guide_latent[:, :, :max_frames]
                if guide_latent.shape[2] <= 0:
                    skipped += 1
                    continue
                tokens_added = guide_latent.shape[2] * guide_latent.shape[3] * guide_latent.shape[4]
                guide_orig_shape = list(guide_latent.shape[2:])
                positive, negative, latent_image, noise_mask = nodes_lt.LTXVAddGuide.append_keyframe(
                    positive, negative, frame_idx, latent_image, noise_mask, guide_latent, strength, scale_factors
                )
                if lora_loaded:
                    att = max(0.0, min(1.0, _safe_float(image_attention_strength, 1.0)))
                    positive = self._append_attention_entry(positive, tokens_added, guide_orig_shape, att)
                    negative = self._append_attention_entry(negative, tokens_added, guide_orig_shape, att)
                image_guides_applied += 1
            except Exception as exc:
                raise RuntimeError(f"IAMCCS Director-compatible image guide failed at index {idx}: {exc}") from exc

        for part in motion_parts:
            try:
                start_frame = max(0, _safe_int(part.get("start"), 0))
                length_frames = max(1, _safe_int(part.get("length"), control_frames.shape[0]))
                trim_start = max(0, _safe_int(part.get("trimStart"), start_frame))
                strength = max(0.0, min(1.0, _safe_float(part.get("videoStrength"), default_strength)))
                attention_strength = max(0.0, min(1.0, _safe_float(part.get("videoAttentionStrength"), default_attention_strength)))
                if strength <= 0.0:
                    skipped += 1
                    continue

                frame_start = min(trim_start, int(control_frames.shape[0]) - 1)
                frame_end = min(int(control_frames.shape[0]), frame_start + length_frames)
                frames = control_frames[frame_start:frame_end]
                frames = self._resample_frames(frames, length_frames, part.get("resampleMode", "nearest"))
                keep = ((frames.shape[0] - 1) // time_scale_factor) * time_scale_factor + 1
                frames = frames[:keep]
                causal_fix = start_frame == 0 or int(frames.shape[0]) == 1
                encode_frames = frames if causal_fix else torch.cat([frames[:1], frames], dim=0)

                ldf_float = max(1.0, float(latent_downscale_factor))
                ldf_int = int(max(1, round(ldf_float)))
                if ldf_int > 1:
                    # IC-LoRA reference guides are encoded at a lower latent grid and
                    # dilated back. Odd latent dimensions (for example 544px -> 17)
                    # must use ceil here; floor would dilate 8 -> 16 and fail when
                    # concatenated with a 17-high latent video.
                    import math
                    target_latent_w = int(math.ceil(float(latent_width) / float(ldf_int)))
                    target_latent_h = int(math.ceil(float(latent_height) / float(ldf_int)))
                    target_w = max(8, int(target_latent_w * width_scale_factor))
                    target_h = max(8, int(target_latent_h * height_scale_factor))
                else:
                    target_w = max(8, int(latent_width * width_scale_factor / ldf_float))
                    target_h = max(8, int(latent_height * height_scale_factor / ldf_float))
                pixels = comfy.utils.common_upscale(
                    encode_frames.movedim(-1, 1),
                    target_w,
                    target_h,
                    "bilinear",
                    crop="center" if crop == "center" else "disabled",
                ).movedim(1, -1)
                pixels = pixels[:, :, :, :3]
                if use_tiled_encode:
                    guide_latent = vae.encode_tiled(pixels, tile_x=tile_size, tile_y=tile_size, overlap=tile_overlap)
                else:
                    guide_latent = vae.encode(pixels)
                guide_latent = guide_latent.to(device=latent_image.device, dtype=latent_image.dtype)

                if not causal_fix:
                    guide_latent = guide_latent[:, :, 1:, :, :]

                frame_idx = start_frame
                latent_idx = (frame_idx + time_scale_factor - 1) // time_scale_factor if frame_idx > 0 else 0
                if latent_idx >= latent_length:
                    skipped += 1
                    continue
                if start_frame > 0 and guide_latent.shape[2] > 1:
                    guide_latent = guide_latent[:, :, 1:, :, :]
                    frame_idx += time_scale_factor
                    latent_idx += 1
                    if latent_idx >= latent_length:
                        skipped += 1
                        continue

                max_frames = latent_length - latent_idx
                if guide_latent.shape[2] > max_frames:
                    guide_latent = guide_latent[:, :, :max_frames]
                if guide_latent.shape[2] <= 0:
                    skipped += 1
                    continue

                guide_orig_shape = list(guide_latent.shape[2:])
                guide_mask = torch.ones((guide_latent.shape[0], 1, guide_latent.shape[2], guide_latent.shape[3], guide_latent.shape[4]), device=guide_latent.device, dtype=guide_latent.dtype)
                ldf = int(max(1, round(float(latent_downscale_factor))))
                if ldf > 1:
                    guide_latent, guide_mask = self._dilate_latent(guide_latent, guide_mask, ldf, ldf)
                    # After ceil+dilate, odd dimensions can overshoot by one cell.
                    # Match the active latent exactly before append_keyframe.
                    if guide_latent.shape[3] != latent_height or guide_latent.shape[4] != latent_width:
                        fixed_latent = torch.zeros(
                            guide_latent.shape[:3] + (latent_height, latent_width),
                            device=guide_latent.device,
                            dtype=guide_latent.dtype,
                            requires_grad=False,
                        )
                        fixed_mask = torch.full(
                            (guide_mask.shape[0], 1, guide_mask.shape[2], latent_height, latent_width),
                            -1.0,
                            device=guide_mask.device,
                            dtype=guide_mask.dtype,
                            requires_grad=False,
                        )
                        copy_h = min(int(latent_height), int(guide_latent.shape[3]))
                        copy_w = min(int(latent_width), int(guide_latent.shape[4]))
                        fixed_latent[..., :copy_h, :copy_w] = guide_latent[..., :copy_h, :copy_w]
                        fixed_mask[..., :copy_h, :copy_w] = guide_mask[..., :copy_h, :copy_w]
                        guide_latent, guide_mask = fixed_latent, fixed_mask

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
                positive = self._append_attention_entry(positive, tokens_added, guide_orig_shape, attention_strength)
                negative = self._append_attention_entry(negative, tokens_added, guide_orig_shape, attention_strength)
                applied += 1
            except Exception as exc:
                raise RuntimeError(f"IAMCCS motionPart IC-LoRA apply failed for {part}: {exc}") from exc

        exact_crop_frames = max(0, int(latent_image.shape[2]) - initial_latent_length)
        positive = node_helpers.conditioning_set_values(positive, {"nghtdrp_guide_crop_latent_frames": exact_crop_frames})
        negative = node_helpers.conditioning_set_values(negative, {"nghtdrp_guide_crop_latent_frames": exact_crop_frames})

        report = {
            "ok": True,
            "node": "IAMCCS_MotionPartsICLoRAApply",
            "backend_mode": "director_compatible_b",
            "image_guides_applied": image_guides_applied,
            "motionParts": len(motion_parts),
            "applied": applied,
            "skipped": skipped,
            "lora_loaded": bool(lora_loaded),
            "ic_lora_name": str(ic_lora_name or "None"),
            "latent_downscale_factor": float(latent_downscale_factor),
            "exact_crop_frames": exact_crop_frames,
            "truth": "Backend B: applies IAMCCS Shotboard GUIDE_DATA first, then timeline motionParts control frames, then emits crop metadata like the Director guide stage.",
        }
        return (positive, negative, {"samples": latent_image, "noise_mask": noise_mask}, model, float(latent_downscale_factor), _json_dumps(report))


class IAMCCS_CineDirectorGuideB(IAMCCS_MotionPartsICLoRAApply):
    """Director-compatible IAMCCS guide stage.

    This node keeps the IAMCCS Shotboard/cine_linx/motionParts authoring model,
    while executing the same essential guide-stage contract: Shotboard image
    GUIDE_DATA first, then motion guide parts, then crop metadata for the
    downstream crop guide node.
    """

    CATEGORY = "IAMCCS/Cine/Shotboard V4"


NODE_CLASS_MAPPINGS = {
    "IAMCCS_CineMotionSketch": IAMCCS_CineMotionSketch,
    "IAMCCS_MotionGuideBridge": IAMCCS_MotionGuideBridge,
    "IAMCCS_MotionSketchRenderer": IAMCCS_MotionSketchRenderer,
    "IAMCCS_MotionPartsICLoRAApply": IAMCCS_MotionPartsICLoRAApply,
    "IAMCCS_CineDirectorGuideB": IAMCCS_CineDirectorGuideB,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_CineMotionSketch": "IAMCCS Cine Motion Sketch",
    "IAMCCS_MotionGuideBridge": "IAMCCS Motion Guide Bridge",
    "IAMCCS_MotionSketchRenderer": "IAMCCS Motion Sketch Renderer",
    "IAMCCS_MotionPartsICLoRAApply": "IAMCCS Motion Parts IC-LoRA Apply",
    "IAMCCS_CineDirectorGuideB": "IAMCCS Cine Director Guide B",
}
