from __future__ import annotations

import copy
import json
from typing import Any, Dict, Iterable, List, Tuple

from .audio.audio_bus_out import SUPERNODE_LINX_TYPE


_MULTITIMELINE_KEYS = (
    "cine_take_router_timeline_data",
    "cine_take_router_package",
    "cine_take_router_package_json",
    "cine_multigeneration_index",
    "cine_multigeneration_index_json",
    "cine_multigeneration_take_package",
    "cine_multigeneration_take_package_json",
)


def _safe_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return int(fallback)


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        parsed = float(value)
        return parsed if parsed == parsed else float(fallback)
    except Exception:
        return float(fallback)


def _safe_json(value: Any, fallback: Any) -> Any:
    if isinstance(value, (dict, list)):
        return copy.deepcopy(value)
    try:
        return json.loads(str(value or ""))
    except Exception:
        return copy.deepcopy(fallback)


def _dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _round_ltx_frames(raw_frames: int) -> int:
    raw = max(1, int(raw_frames))
    if raw <= 1:
        return 1
    return ((raw - 1 + 7) // 8) * 8 + 1


def _first_number(item: Dict[str, Any], keys: Iterable[str], fallback: int) -> int:
    for key in keys:
        if key in item and item.get(key) is not None:
            return _safe_int(item.get(key), fallback)
    return int(fallback)


def _clamp_segment_list(items: Any, frame_limit: int, report: Dict[str, Any]) -> List[Any]:
    """Clip frame-based segments without changing source identity or trim IDs."""
    if not isinstance(items, list):
        return items
    out: List[Any] = []
    start_keys = ("start", "frame", "start_frame", "startFrame", "globalStart")
    length_keys = ("length", "len", "duration_frames", "durationFrames", "audioDurationFrames")
    end_keys = ("end_frame", "endFrame")
    limit = max(1, int(frame_limit))
    fps = max(1, _safe_int(report.get("frame_rate", 1), 1))
    for item in items:
        if not isinstance(item, dict):
            out.append(item)
            continue
        cloned = copy.deepcopy(item)
        start = max(0, _first_number(cloned, start_keys, 0))
        end_hint = _first_number(cloned, ("end_frame", "endFrame"), 0)
        length = _first_number(cloned, length_keys, 0)
        if length <= 0 and end_hint > start:
            length = end_hint - start
        if length <= 0:
            length = limit - start
        if start >= limit:
            report["dropped_segments"] = report.get("dropped_segments", 0) + 1
            continue
        clipped_length = max(1, min(int(length), limit - start))
        if clipped_length != int(length):
            report["clipped_segments"] = report.get("clipped_segments", 0) + 1

        # Keep the original keys, while always exposing the canonical length.
        cloned["length"] = int(clipped_length)
        for key in ("len", "duration_frames", "durationFrames", "audioDurationFrames"):
            if key in cloned:
                cloned[key] = int(clipped_length)
        for key in end_keys:
            if key in cloned:
                cloned[key] = int(start + clipped_length)
        if "duration_seconds" in cloned:
            cloned["duration_seconds"] = float(clipped_length / fps)
        if "durationSeconds" in cloned:
            cloned["durationSeconds"] = float(clipped_length / fps)
        out.append(cloned)
    return out


def _set_timing(data: Dict[str, Any], duration_seconds: float, fps: int, max_frames: int) -> None:
    if not isinstance(data, dict):
        return
    for key in ("duration_seconds", "durationSeconds"):
        if key in data:
            data[key] = float(duration_seconds)
    for key in ("frame_rate", "frameRate", "fps"):
        if key in data:
            data[key] = int(fps)
    for key in ("max_frames", "maxFrames", "duration_frames", "durationFrames"):
        if key in data:
            data[key] = int(max_frames)
    if "duration" in data:
        data["duration"] = float(duration_seconds)
    settings = data.get("settings")
    if isinstance(settings, dict):
        _set_timing(settings, duration_seconds, fps, max_frames)


def _clamp_visual_timeline(data: Any, target_frames: int, visual_frames: int, fps: int, duration: float, report: Dict[str, Any]) -> Any:
    if not isinstance(data, dict):
        return data
    cloned = copy.deepcopy(data)
    _set_timing(cloned, duration, fps, visual_frames)
    for key in ("segments", "rows", "visual_segments"):
        if isinstance(cloned.get(key), list):
            before = len(cloned[key])
            cloned[key] = _clamp_segment_list(cloned[key], visual_frames, report)
            report["visual_lists_clamped"] = report.get("visual_lists_clamped", 0) + 1
            report["visual_items_seen"] = report.get("visual_items_seen", 0) + before
    if isinstance(cloned.get("audioSegments"), list):
        cloned["audioSegments"] = _clamp_segment_list(cloned["audioSegments"], target_frames, report)
        report["audio_lists_clamped"] = report.get("audio_lists_clamped", 0) + 1
    return cloned


def _clamp_audio_tree(value: Any, frame_limit: int, report: Dict[str, Any]) -> Any:
    if isinstance(value, list):
        return _clamp_segment_list(value, frame_limit, report)
    if not isinstance(value, dict):
        return value
    cloned = copy.deepcopy(value)
    for key in ("segments", "all_segments", "shotboard_segments", "audioSegments"):
        if isinstance(cloned.get(key), list):
            cloned[key] = _clamp_segment_list(cloned[key], frame_limit, report)
    for key, child in list(cloned.items()):
        if isinstance(child, dict):
            cloned[key] = _clamp_audio_tree(child, frame_limit, report)
    return cloned


def _has_multitimeline(resources: Dict[str, Any]) -> bool:
    return any(key in resources and resources.get(key) not in (None, "", {}, []) for key in _MULTITIMELINE_KEYS)


class IAMCCS_AudioBoardDirectShotboardAdapter:
    """Make a direct AudioBoard branch obey the Shotboard duration contract.

    This node is intentionally a boundary adapter. It does not change the V3
    planner or the multigeneration router, and it bypasses itself when a
    multitimeline TakePackage is already present.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "duration_seconds": ("FLOAT", {"default": 20.0, "min": 0.01, "max": 36000.0, "step": 0.01}),
                "frame_rate": ("INT", {"default": 24, "min": 1, "max": 240, "step": 1}),
                "trim_audio_to_duration": ("BOOLEAN", {"default": True}),
                "protect_multitimeline": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE, "STRING")
    RETURN_NAMES = ("cine_linx", "report")
    FUNCTION = "adapt"
    CATEGORY = "IAMCCS/Cine/Audio"

    def adapt(self, cine_linx, duration_seconds, frame_rate, trim_audio_to_duration=True, protect_multitimeline=True):
        out_linx = copy.deepcopy(cine_linx) if isinstance(cine_linx, dict) else {
            "type": SUPERNODE_LINX_TYPE,
            "mode": "iamccs_audio_board_direct_adapter",
            "resources": {},
            "outputs": {},
        }
        resources = out_linx.setdefault("resources", {})
        if not isinstance(resources, dict):
            resources = {}
            out_linx["resources"] = resources
        outputs = out_linx.setdefault("outputs", {})
        if not isinstance(outputs, dict):
            outputs = {}
            out_linx["outputs"] = outputs

        duration = max(0.01, _safe_float(duration_seconds, 20.0))
        fps = max(1, _safe_int(frame_rate, 24))
        target_frames = max(1, int(round(duration * fps)))
        visual_frames = _round_ltx_frames(target_frames)
        report: Dict[str, Any] = {
            "node": "IAMCCS_AudioBoardDirectShotboardAdapter",
            "mode": "direct_audioboard_boundary",
            "target_duration_seconds": float(duration),
            "target_audio_frames": int(target_frames),
            "target_visual_frames_ltx_rounded": int(visual_frames),
            "frame_rate": int(fps),
            "trim_audio_to_duration": bool(trim_audio_to_duration),
            "bypassed_multitimeline": False,
        }

        if bool(protect_multitimeline) and _has_multitimeline(resources):
            report["mode"] = "multitimeline_passthrough"
            report["bypassed_multitimeline"] = True
            report["reason"] = "TakePackage or multigeneration router metadata is present"
            outputs["audio_board_direct_adapter_report"] = _dump_json(report)
            return out_linx, _dump_json(report)

        if bool(trim_audio_to_duration):
            audio_report: Dict[str, Any] = {}
            for key in ("cine_audio_tracks", "cine_audio_bus_out"):
                if key in resources:
                    resources[key] = _clamp_audio_tree(resources[key], target_frames, audio_report)
            for key in ("cine_audio_master_out_json", "cine_audio_track_out_jsons"):
                raw = resources.get(key)
                if isinstance(raw, list):
                    resources[key] = [_dump_json(_clamp_audio_tree(_safe_json(item, {}), target_frames, audio_report)) for item in raw]
                elif isinstance(raw, str) and raw.strip():
                    resources[key] = _dump_json(_clamp_audio_tree(_safe_json(raw, {}), target_frames, audio_report))
            report.update(audio_report)

        for key in ("cine_board_timeline_data", "cine_dialogue_shotboard_timeline_json", "timeline_data"):
            raw = resources.get(key)
            if isinstance(raw, str) and raw.strip():
                resources[key] = _dump_json(_clamp_visual_timeline(
                    _safe_json(raw, {}), target_frames, visual_frames, fps, duration, report,
                ))

        for key in ("cine_audio_timeline_json",):
            raw = resources.get(key)
            if isinstance(raw, str) and raw.strip():
                audio_data = _safe_json(raw, {})
                audio_data = _clamp_audio_tree(audio_data, target_frames, report)
                if isinstance(audio_data, dict):
                    _set_timing(audio_data, duration, fps, target_frames)
                resources[key] = _dump_json(audio_data)

        visual_raw = resources.get("cine_visual_segments_json")
        if isinstance(visual_raw, str) and visual_raw.strip():
            visual_data = _safe_json(visual_raw, [])
            resources["cine_visual_segments_json"] = _dump_json(
                _clamp_segment_list(visual_data, visual_frames, report)
            )

        payload = resources.get("cine_payload")
        if isinstance(payload, dict):
            payload = copy.deepcopy(payload)
            _set_timing(payload, duration, fps, visual_frames)
            if isinstance(payload.get("visual_segments"), list):
                payload["visual_segments"] = _clamp_segment_list(payload["visual_segments"], visual_frames, report)
            if isinstance(payload.get("audioSegments"), list) and bool(trim_audio_to_duration):
                payload["audioSegments"] = _clamp_segment_list(payload["audioSegments"], target_frames, report)
            resources["cine_payload"] = payload

        resources["cine_duration_seconds"] = float(duration)
        resources["cine_frame_rate"] = int(fps)
        resources["cine_max_frames"] = int(visual_frames)
        resources["cine_audio_board_direct_mode"] = True
        resources["cine_audio_board_direct_adapter"] = copy.deepcopy(report)
        outputs["duration_seconds"] = float(duration)
        outputs["frame_rate"] = int(fps)
        outputs["max_frames"] = int(visual_frames)
        outputs["audio_board_direct_adapter_report"] = _dump_json(report)
        out_linx["type"] = SUPERNODE_LINX_TYPE
        out_linx["mode"] = "iamccs_audio_board_direct_adapter"
        out_linx.setdefault("chain", []).append({
            "role": "audio_board_direct_adapter",
            "name": "IAMCCS_AudioBoardDirectShotboardAdapter",
            "duration_seconds": float(duration),
            "frame_rate": int(fps),
        })
        out_linx["resource_keys"] = sorted(resources.keys())
        out_linx["resource_types"] = {key: type(value).__name__ for key, value in resources.items()}
        return out_linx, _dump_json(report)


NODE_CLASS_MAPPINGS = {
    "IAMCCS_AudioBoardDirectShotboardAdapter": IAMCCS_AudioBoardDirectShotboardAdapter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_AudioBoardDirectShotboardAdapter": "IAMCCS AudioBoard Direct -> Shotboard Adapter",
}
