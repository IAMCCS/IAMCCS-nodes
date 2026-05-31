from __future__ import annotations

import copy
import json
import math
from fractions import Fraction
from typing import Any, Dict, List, Tuple

import torch
import torchaudio
from comfy_api.latest import InputImpl, Types


SUPERNODE_LINX_TYPE = "IAMCCS_SUPERNODE_LINX"
MAX_TRACK_OUTS = 5


def _safe_json_loads(value: Any, fallback: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        text = str(value or "").strip()
        if not text:
            return fallback
        return json.loads(text)
    except Exception:
        return fallback


def _json_dump(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def _clone_linx(cine_linx: Any, mode: str = "iamccs_multigeneration") -> Dict[str, Any]:
    if isinstance(cine_linx, dict):
        return copy.deepcopy(cine_linx)
    return {
        "type": SUPERNODE_LINX_TYPE,
        "mode": mode,
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


def _refresh_linx_index(cine_linx: Dict[str, Any]) -> None:
    resources = _resources(cine_linx)
    cine_linx["resource_keys"] = sorted(resources.keys())
    cine_linx["resource_types"] = {key: type(value).__name__ for key, value in resources.items()}


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


def _seconds_for_template(template: str, custom_chunk_seconds: Any) -> float:
    normalized = str(template or "20s").strip().lower()
    if normalized == "custom":
        return max(1.0, _safe_float(custom_chunk_seconds, 20.0))
    if normalized.endswith("s"):
        normalized = normalized[:-1]
    return max(1.0, _safe_float(normalized, 20.0))


def _segments(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        seg = copy.deepcopy(item)
        seg["id"] = str(seg.get("id") or f"multi_src_{index + 1:03d}")
        seg["type"] = "audio"
        seg["start"] = max(0, _safe_int(seg.get("start", 0), 0))
        seg["length"] = max(1, _safe_int(seg.get("length", seg.get("audioDurationFrames", 1)), 1))
        seg["track"] = max(0, _safe_int(seg.get("track", 0), 0))
        seg["trimStart"] = max(0, _safe_int(seg.get("trimStart", 0), 0))
        seg["audioDurationFrames"] = max(seg["trimStart"] + seg["length"], _safe_int(seg.get("audioDurationFrames", seg["length"]), seg["length"]))
        out.append(seg)
    return sorted(out, key=lambda seg: (int(seg.get("start", 0)), int(seg.get("track", 0))))


def _max_end(segments: List[Dict[str, Any]]) -> int:
    return max([_safe_int(seg.get("start", 0), 0) + _safe_int(seg.get("length", 1), 1) for seg in segments] or [0])


def _parse_track_jsons(track_jsons: Tuple[Any, ...]) -> List[Dict[str, Any]]:
    parsed: List[Dict[str, Any]] = []
    for index, value in enumerate(track_jsons):
        data = _safe_json_loads(value, {})
        if isinstance(data, dict) and data:
            data = copy.deepcopy(data)
            data.setdefault("track_index", index)
            data.setdefault("track_name", f"A{index + 1}")
            data["segments"] = _segments(data.get("segments"))
            parsed.append(data)
    return parsed


def _bus_manifest(
    cine_linx: Any,
    bus_manifest_json: Any,
    master_out_json: Any,
    track_jsons: Tuple[Any, ...],
) -> Dict[str, Any]:
    linx = cine_linx if isinstance(cine_linx, dict) else {}
    resources = linx.get("resources", {}) if isinstance(linx.get("resources", {}), dict) else {}
    manifest = _safe_json_loads(bus_manifest_json, {})
    if not isinstance(manifest, dict) or not manifest:
        manifest = resources.get("cine_audio_bus_out") if isinstance(resources.get("cine_audio_bus_out"), dict) else {}
    if not isinstance(manifest, dict):
        manifest = {}

    master = _safe_json_loads(master_out_json, {})
    if not isinstance(master, dict) or not master:
        master = manifest.get("master") if isinstance(manifest.get("master"), dict) else {}
    if not isinstance(master, dict) or not master:
        audio_tracks = resources.get("cine_audio_tracks") if isinstance(resources.get("cine_audio_tracks"), dict) else {}
        master = {
            "schema": "iamccs.audio_bus_out.master",
            "source": "IAMCCS_MultiTimelineBridge",
            "segments": _segments(audio_tracks.get("all_segments") or audio_tracks.get("segments")),
            "masterBus": audio_tracks.get("master_bus") if isinstance(audio_tracks.get("master_bus"), dict) else {},
            "duration_frames": _safe_int(audio_tracks.get("source_end_frames", audio_tracks.get("duration_frames", 0)), 0),
        }
    master = copy.deepcopy(master)
    master["segments"] = _segments(master.get("segments"))
    master["duration_frames"] = max(_safe_int(master.get("duration_frames", 0), 0), _max_end(master["segments"]))

    tracks = _parse_track_jsons(track_jsons)
    if not tracks and isinstance(manifest.get("tracks"), list):
        for index, item in enumerate(manifest.get("tracks") or []):
            if not isinstance(item, dict):
                continue
            track = copy.deepcopy(item)
            track.setdefault("track_index", index)
            track.setdefault("track_name", f"A{index + 1}")
            track["segments"] = _segments(track.get("segments"))
            track["duration_frames"] = max(_safe_int(track.get("duration_frames", 0), 0), _max_end(track["segments"]))
            tracks.append(track)
    if not tracks:
        for track_index in range(MAX_TRACK_OUTS):
            track_segments = [seg for seg in master["segments"] if _safe_int(seg.get("track", 0), 0) == track_index]
            tracks.append({
                "schema": "iamccs.audio_bus_out.track",
                "source": "IAMCCS_MultiTimelineBridge",
                "track_index": track_index,
                "track_name": f"A{track_index + 1}",
                "segments": track_segments,
                "duration_frames": _max_end(track_segments),
            })

    generation_index = manifest.get("generation_index") if isinstance(manifest.get("generation_index"), dict) else {}
    if not generation_index:
        generation_index = resources.get("cine_audio_generation_index") if isinstance(resources.get("cine_audio_generation_index"), dict) else {}
    if not generation_index:
        generation_index = _safe_json_loads(resources.get("cine_audio_generation_index_json"), {})
    if not isinstance(generation_index, dict):
        generation_index = {}

    return {
        "schema": "iamccs.audio_bus_out.manifest",
        "schema_version": 1,
        "source": "IAMCCS_MultiTimelineBridge",
        "master": master,
        "tracks": tracks[:MAX_TRACK_OUTS],
        "generation_index": generation_index,
    }


def _source_from_manifest(manifest: Dict[str, Any], source_bus: str) -> Dict[str, Any]:
    source = str(source_bus or "master_out")
    if source == "master_out":
        return copy.deepcopy(manifest.get("master") if isinstance(manifest.get("master"), dict) else {})
    if source.startswith("track_"):
        track_number = max(1, _safe_int(source.split("_", 1)[1], 1))
        tracks = manifest.get("tracks") if isinstance(manifest.get("tracks"), list) else []
        index = track_number - 1
        if 0 <= index < len(tracks) and isinstance(tracks[index], dict):
            return copy.deepcopy(tracks[index])
    return {}


def _slice_segments(
    segments: List[Dict[str, Any]],
    window_start: int,
    window_length: int,
    track_layout: str,
) -> List[Dict[str, Any]]:
    window_end = window_start + window_length
    sliced: List[Dict[str, Any]] = []
    for index, seg in enumerate(segments):
        seg_start = _safe_int(seg.get("start", 0), 0)
        seg_length = max(1, _safe_int(seg.get("length", 1), 1))
        seg_end = seg_start + seg_length
        overlap_start = max(seg_start, window_start)
        overlap_end = min(seg_end, window_end)
        if overlap_end <= overlap_start:
            continue
        offset = overlap_start - seg_start
        out = copy.deepcopy(seg)
        out["id"] = f"{seg.get('id', 'aud')}_take_{window_start}_{index}"
        out["start"] = overlap_start - window_start
        out["length"] = overlap_end - overlap_start
        out["trimStart"] = max(0, _safe_int(seg.get("trimStart", 0), 0) + offset)
        out["audioDurationFrames"] = max(out["trimStart"] + out["length"], _safe_int(seg.get("audioDurationFrames", seg_length), seg_length))
        out["sourceSegmentId"] = str(seg.get("id", ""))
        out["sourceGlobalStart"] = overlap_start
        out["sourceGlobalEnd"] = overlap_end
        out["sourceTrack"] = _safe_int(seg.get("track", 0), 0)
        out["multiGenerationClip"] = True
        if str(track_layout) == "collapse_to_lane_1":
            out["track"] = 0
        sliced.append(out)
    return sorted(sliced, key=lambda seg: (_safe_int(seg.get("start", 0), 0), _safe_int(seg.get("track", 0), 0)))


def _timeline_for_take(take: Dict[str, Any], fps: float, track_layout: str) -> Dict[str, Any]:
    segments = _segments(take.get("audioSegments"))
    track_count = 1
    if str(track_layout) == "preserve_bus_tracks":
        track_count = max([_safe_int(seg.get("track", 0), 0) + 1 for seg in segments] or [1])
    return {
        "schema": "iamccs.multigeneration.take_audio_timeline",
        "schema_version": 1,
        "timeline_id": str(take.get("timeline_id", "")),
        "take_index": _safe_int(take.get("take_index", 1), 1),
        "frame_rate": float(fps),
        "duration_frames": _safe_int(take.get("duration_frames", 0), 0),
        "duration_seconds": _safe_int(take.get("duration_frames", 0), 0) / max(1.0, float(fps)),
        "audioSegments": segments,
        "audioTrackCount": track_count,
        "audioBusMode": "all_tracks",
        "onlyFirstTrack": False,
        "use_custom_audio": any(str(seg.get("audioFile", "")).strip() or str(seg.get("audioB64", "")).strip() for seg in segments),
    }


def _take_sort_key(seg: Dict[str, Any], fallback: int) -> Tuple[int, int, int]:
    take_index = _safe_int(seg.get("multiTakeIndex", seg.get("take_index", fallback + 1)), fallback + 1)
    global_start = _safe_int(seg.get("sourceGlobalStart", seg.get("globalStart", seg.get("start", 0))), 0)
    track = _safe_int(seg.get("track", 0), 0)
    return take_index, global_start, track


def _segments_from_generation_index(generation_index: Any) -> List[Dict[str, Any]]:
    if not isinstance(generation_index, dict):
        return []
    out: List[Dict[str, Any]] = []
    for take in generation_index.get("takes") if isinstance(generation_index.get("takes"), list) else []:
        if isinstance(take, dict):
            out.extend(_segments(take.get("segments")))
    return out


def _collect_prechunked_segments(manifest: Dict[str, Any], source_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = list(source_segments)
    master = manifest.get("master") if isinstance(manifest.get("master"), dict) else {}
    candidates.extend(_segments(master.get("segments")))
    for track in manifest.get("tracks") if isinstance(manifest.get("tracks"), list) else []:
        if isinstance(track, dict):
            candidates.extend(_segments(track.get("segments")))
    candidates.extend(_segments_from_generation_index(manifest.get("generation_index")))

    out: List[Dict[str, Any]] = []
    seen = set()
    for seg in candidates:
        if not (bool(seg.get("multiGenerationClip")) or str(seg.get("timelineId", "") or "").startswith("T")):
            continue
        key = (
            str(seg.get("id", "")),
            str(seg.get("timelineId", "")),
            _safe_int(seg.get("multiTakeIndex", seg.get("take_index", 0)), 0),
            _safe_int(seg.get("track", 0), 0),
            _safe_int(seg.get("sourceGlobalStart", seg.get("globalStart", seg.get("start", 0))), 0),
            _safe_int(seg.get("length", 1), 1),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(copy.deepcopy(seg))
    return sorted(out, key=lambda item: _take_sort_key(item, 0))


def _takes_from_prechunked_segments(
    segments: List[Dict[str, Any]],
    fps: float,
    chunk_frames: int,
    chunk_seconds: float,
    source_bus: str,
    track_layout: str,
    visual_timelines: Any,
) -> List[Dict[str, Any]]:
    multi_segments = [
        copy.deepcopy(seg)
        for seg in segments
        if bool(seg.get("multiGenerationClip")) or str(seg.get("timelineId", "") or "").startswith("T")
    ]
    if not multi_segments:
        return []

    groups: Dict[int, List[Dict[str, Any]]] = {}
    for fallback, seg in enumerate(sorted(multi_segments, key=lambda item: _take_sort_key(item, fallback=0))):
        take_index = _safe_int(seg.get("multiTakeIndex", seg.get("take_index", fallback + 1)), fallback + 1)
        groups.setdefault(max(1, take_index), []).append(seg)

    takes: List[Dict[str, Any]] = []
    for order, take_index in enumerate(sorted(groups), start=1):
        group = groups[take_index]
        global_start = min([
            _safe_int(seg.get("sourceGlobalStart", seg.get("globalStart", seg.get("start", 0))), 0)
            for seg in group
        ] or [0])
        global_end = max([
            _safe_int(seg.get("sourceGlobalEnd", seg.get("globalEnd", _safe_int(seg.get("start", 0), 0) + _safe_int(seg.get("length", 1), 1))), 0)
            for seg in group
        ] or [global_start + chunk_frames])
        duration = max(1, global_end - global_start)
        audio_segments: List[Dict[str, Any]] = []
        for seg_index, seg in enumerate(group):
            out = copy.deepcopy(seg)
            source_global_start = _safe_int(out.get("sourceGlobalStart", out.get("globalStart", out.get("start", 0))), 0)
            out["id"] = str(out.get("id") or f"multi_take_{take_index:02d}_{seg_index + 1:02d}")
            out["start"] = max(0, source_global_start - global_start)
            out["length"] = max(1, _safe_int(out.get("length", 1), 1))
            out["sourceTrack"] = _safe_int(out.get("sourceTrack", out.get("track", 0)), 0)
            out["timelineId"] = str(out.get("timelineId") or f"T{take_index:02d}")
            out["multiTakeIndex"] = int(take_index)
            if str(track_layout) == "collapse_to_lane_1":
                out["track"] = 0
            audio_segments.append(out)
        timeline_id = str(group[0].get("timelineId") or f"T{take_index:02d}")
        take = {
            "schema": "iamccs.multigeneration.take",
            "schema_version": 1,
            "take_index": int(order),
            "source_take_index": int(take_index),
            "timeline_id": timeline_id,
            "source_bus": str(source_bus),
            "global_start_frames": int(global_start),
            "global_end_frames": int(global_start + duration),
            "local_start_frames": 0,
            "duration_frames": int(duration),
            "duration_seconds": duration / fps,
            "chunk_frames": int(chunk_frames),
            "chunk_seconds": float(chunk_seconds),
            "audioSegments": sorted(audio_segments, key=lambda item: (_safe_int(item.get("start", 0), 0), _safe_int(item.get("track", 0), 0))),
            "audioTrackCount": max([_safe_int(seg.get("track", 0), 0) + 1 for seg in audio_segments] or [1]),
            "visual_timeline_key": timeline_id,
            "visual_timeline": visual_timelines.get(timeline_id) if isinstance(visual_timelines, dict) else None,
            "prechunked": True,
        }
        take["take_audio_timeline"] = _timeline_for_take(take, fps, str(track_layout))
        takes.append(take)
    return takes


def _apply_active_take(
    cine_linx: Dict[str, Any],
    generation_index: Dict[str, Any],
    active_take: Dict[str, Any],
    track_layout: str,
) -> None:
    fps = _safe_float(generation_index.get("frame_rate", 24.0), 24.0)
    take_timeline = _timeline_for_take(active_take, fps, track_layout)
    duration_frames = _safe_int(take_timeline.get("duration_frames", 0), 0)
    duration_seconds = duration_frames / max(1.0, fps)
    resources = _resources(cine_linx)
    outputs = _outputs(cine_linx)
    payload = _payload(cine_linx)

    resources["cine_multigeneration_index"] = generation_index
    resources["cine_multigeneration_index_json"] = _json_dump(generation_index)
    resources["cine_multigeneration_active_take"] = active_take
    resources["cine_multigeneration_active_take_json"] = _json_dump(active_take)
    resources["cine_multigeneration_take_audio_timeline"] = take_timeline
    resources["cine_multigeneration_take_audio_timeline_json"] = _json_dump(take_timeline)
    resources["cine_duration_seconds"] = float(duration_seconds)
    resources["cine_max_frames"] = int(duration_frames)

    payload["multi_generation"] = generation_index
    payload["multi_generation_active_take"] = active_take
    payload["timeline_id"] = str(active_take.get("timeline_id", ""))
    payload["duration_seconds"] = float(duration_seconds)
    payload["max_frames"] = int(duration_frames)
    payload["audioSegments"] = take_timeline["audioSegments"]
    payload["audioTrackCount"] = take_timeline["audioTrackCount"]
    payload["use_custom_audio"] = bool(take_timeline["use_custom_audio"])
    payload["audioSyncMode"] = "timeline_audio"

    outputs["generation_index_json"] = _json_dump(generation_index)
    outputs["active_take_json"] = _json_dump(active_take)
    outputs["take_audio_timeline_json"] = _json_dump(take_timeline)
    outputs["duration_seconds"] = float(duration_seconds)
    outputs["max_frames"] = int(duration_frames)


def _make_concat_plan(index: Dict[str, Any], source_bus: str) -> Dict[str, Any]:
    takes = index.get("takes") if isinstance(index.get("takes"), list) else []
    return {
        "schema": "iamccs.multigeneration.concat_plan",
        "schema_version": 1,
        "source": "IAMCCS_MultiTimelineBridge",
        "source_bus": str(source_bus),
        "final_audio_policy": "restore_original_master_or_selected_bus_after_video_concat",
        "video_concat_policy": "hard_cut_in_take_order",
        "takes": [{
            "take_index": _safe_int(take.get("take_index", idx + 1), idx + 1),
            "timeline_id": str(take.get("timeline_id", f"T{idx + 1:02d}")),
            "global_start_frames": _safe_int(take.get("global_start_frames", 0), 0),
            "duration_frames": _safe_int(take.get("duration_frames", 0), 0),
            "expected_video_slot": f"video_take_{idx + 1:02d}",
        } for idx, take in enumerate(takes)],
    }


class IAMCCS_MultiTimelineBridge:
    """Build a sequenced take index from BusOut audio for chunked video-driven generation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chunk_template": (["10s", "15s", "20s", "25s", "custom"], {"default": "20s"}),
                "custom_chunk_seconds": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 300.0, "step": 0.25}),
                "source_bus": (["master_out", "track_1", "track_2", "track_3", "track_4", "track_5"], {"default": "master_out"}),
                "take_source_mode": (["auto_detect_multi_lanes", "chunk_source_bus"], {"default": "auto_detect_multi_lanes"}),
                "take_count_mode": (["auto_from_audio", "fixed_take_count"], {"default": "auto_from_audio"}),
                "fixed_take_count": ("INT", {"default": 3, "min": 1, "max": 64, "step": 1}),
                "max_takes": ("INT", {"default": 12, "min": 1, "max": 64, "step": 1}),
                "active_take": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "take_track_layout": (["collapse_to_lane_1", "preserve_bus_tracks"], {"default": "collapse_to_lane_1"}),
            },
            "optional": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "bus_manifest_json": ("STRING", {"default": "", "multiline": True}),
                "master_out_json": ("STRING", {"default": "", "multiline": True}),
                "track_1_json": ("STRING", {"default": "", "multiline": True}),
                "track_2_json": ("STRING", {"default": "", "multiline": True}),
                "track_3_json": ("STRING", {"default": "", "multiline": True}),
                "track_4_json": ("STRING", {"default": "", "multiline": True}),
                "track_5_json": ("STRING", {"default": "", "multiline": True}),
                "visual_timelines_json": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE, "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("cine_linx", "generation_index_json", "active_take_json", "concat_plan_json", "report")
    FUNCTION = "build"
    CATEGORY = "IAMCCS/Cine/Multigeneration"

    def build(
        self,
        chunk_template,
        custom_chunk_seconds,
        source_bus,
        take_source_mode,
        take_count_mode,
        fixed_take_count,
        max_takes,
        active_take,
        frame_rate,
        take_track_layout,
        cine_linx=None,
        bus_manifest_json="",
        master_out_json="",
        track_1_json="",
        track_2_json="",
        track_3_json="",
        track_4_json="",
        track_5_json="",
        visual_timelines_json="",
    ):
        fps = max(1.0, float(frame_rate))
        chunk_seconds = _seconds_for_template(chunk_template, custom_chunk_seconds)
        chunk_frames = max(1, int(round(chunk_seconds * fps)))
        max_take_count = max(1, _safe_int(max_takes, 12))
        manifest = _bus_manifest(
            cine_linx,
            bus_manifest_json,
            master_out_json,
            (track_1_json, track_2_json, track_3_json, track_4_json, track_5_json),
        )
        source = _source_from_manifest(manifest, source_bus)
        source_segments = _segments(source.get("segments"))
        source_duration = max(_safe_int(source.get("duration_frames", 0), 0), _max_end(source_segments))
        if source_duration <= 0:
            source_duration = chunk_frames * max(1, _safe_int(fixed_take_count, 3))

        if str(take_count_mode) == "fixed_take_count":
            take_count = max(1, min(max_take_count, _safe_int(fixed_take_count, 3)))
        else:
            take_count = max(1, min(max_take_count, int(math.ceil(source_duration / max(1, chunk_frames)))))

        visual_timelines = _safe_json_loads(visual_timelines_json, {})
        if not isinstance(visual_timelines, (dict, list)):
            visual_timelines = {}

        takes = []
        if str(take_source_mode) == "auto_detect_multi_lanes":
            prechunked_segments = _collect_prechunked_segments(manifest, source_segments)
            if prechunked_segments:
                prechunked_end = max([
                    _safe_int(seg.get("sourceGlobalEnd", seg.get("globalEnd", _safe_int(seg.get("start", 0), 0) + _safe_int(seg.get("length", 1), 1))), 0)
                    for seg in prechunked_segments
                ] or [0])
                source_duration = max(source_duration, prechunked_end)
            takes = _takes_from_prechunked_segments(
                prechunked_segments,
                fps,
                chunk_frames,
                chunk_seconds,
                str(source_bus),
                str(take_track_layout),
                visual_timelines,
            )

        if not takes:
            for index in range(take_count):
                global_start = index * chunk_frames
                remaining = max(0, source_duration - global_start)
                duration = chunk_frames if str(take_count_mode) == "fixed_take_count" else min(chunk_frames, remaining or chunk_frames)
                duration = max(1, int(duration))
                timeline_id = f"T{index + 1:02d}"
                audio_segments = _slice_segments(source_segments, global_start, duration, str(take_track_layout))
                take = {
                    "schema": "iamccs.multigeneration.take",
                    "schema_version": 1,
                    "take_index": index + 1,
                    "timeline_id": timeline_id,
                    "source_bus": str(source_bus),
                    "global_start_frames": int(global_start),
                    "global_end_frames": int(global_start + duration),
                    "local_start_frames": 0,
                    "duration_frames": int(duration),
                    "duration_seconds": duration / fps,
                    "chunk_frames": int(chunk_frames),
                    "chunk_seconds": float(chunk_seconds),
                    "audioSegments": audio_segments,
                    "audioTrackCount": max([_safe_int(seg.get("track", 0), 0) + 1 for seg in audio_segments] or [1]),
                    "visual_timeline_key": timeline_id,
                    "visual_timeline": visual_timelines.get(timeline_id) if isinstance(visual_timelines, dict) else None,
                    "prechunked": False,
                }
                take["take_audio_timeline"] = _timeline_for_take(take, fps, str(take_track_layout))
                takes.append(take)

        active_index = max(1, min(len(takes), _safe_int(active_take, 1))) - 1
        generation_index = {
            "schema": "iamccs.multigeneration.index",
            "schema_version": 1,
            "source": "IAMCCS_MultiTimelineBridge",
            "frame_rate": float(fps),
            "chunk_template": str(chunk_template),
            "chunk_seconds": float(chunk_seconds),
            "chunk_frames": int(chunk_frames),
            "source_bus": str(source_bus),
            "take_source_mode": str(take_source_mode),
            "source_duration_frames": int(source_duration),
            "source_duration_seconds": source_duration / fps,
            "take_count": int(len(takes)),
            "active_take": int(active_index + 1),
            "take_track_layout": str(take_track_layout),
            "takes": takes,
            "bus_generation_index": manifest.get("generation_index") if isinstance(manifest.get("generation_index"), dict) else {},
            "truth": "Audio remains BusOut/AudioBoard custom-audio metadata. Each take receives a local audio window for sequential video-driven generation, then video takes are hard-concatenated.",
        }
        concat_plan = _make_concat_plan(generation_index, str(source_bus))

        out_linx = _clone_linx(cine_linx)
        resources = _resources(out_linx)
        outputs = _outputs(out_linx)
        resources["cine_multigeneration_concat_plan"] = concat_plan
        resources["cine_multigeneration_concat_plan_json"] = _json_dump(concat_plan)
        resources["cine_multigeneration_bus_manifest"] = manifest
        _apply_active_take(out_linx, generation_index, takes[active_index], str(take_track_layout))
        outputs["concat_plan_json"] = _json_dump(concat_plan)
        out_linx["type"] = SUPERNODE_LINX_TYPE
        out_linx["mode"] = "iamccs_multigeneration"
        out_linx.setdefault("chain", []).append({
            "role": "multigeneration_bridge",
            "name": "IAMCCS_MultiTimelineBridge",
            "active_take": active_index + 1,
        })
        _refresh_linx_index(out_linx)

        report = _json_dump({
            "node": "IAMCCS_MultiTimelineBridge",
            "source_bus": str(source_bus),
            "chunk_seconds": float(chunk_seconds),
            "chunk_frames": int(chunk_frames),
            "take_count": len(takes),
            "active_take": active_index + 1,
            "source_segments": len(source_segments),
            "active_segments": len(takes[active_index].get("audioSegments", [])),
            "prechunked": bool(takes[active_index].get("prechunked", False)),
            "concat_policy": concat_plan["video_concat_policy"],
        })
        outputs["report"] = report
        return out_linx, _json_dump(generation_index), _json_dump(takes[active_index]), _json_dump(concat_plan), report


class IAMCCS_MultiTimelineTakePicker:
    """Pick one take from a MultiTimelineBridge index and expose it as active cine_linx audio."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generation_index_json": ("STRING", {"default": "", "multiline": True}),
                "take_index": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "take_track_layout": (["collapse_to_lane_1", "preserve_bus_tracks"], {"default": "collapse_to_lane_1"}),
            },
            "optional": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
            },
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("cine_linx", "active_take_json", "take_audio_timeline_json", "report")
    FUNCTION = "pick"
    CATEGORY = "IAMCCS/Cine/Multigeneration"

    def pick(self, generation_index_json, take_index, take_track_layout, cine_linx=None):
        generation_index = _safe_json_loads(generation_index_json, {})
        if not isinstance(generation_index, dict):
            generation_index = {}
        takes = generation_index.get("takes") if isinstance(generation_index.get("takes"), list) else []
        if not takes:
            empty = {
                "schema": "iamccs.multigeneration.take",
                "schema_version": 1,
                "take_index": 1,
                "timeline_id": "T01",
                "duration_frames": 1,
                "audioSegments": [],
            }
            takes = [empty]
            generation_index["takes"] = takes
            generation_index.setdefault("frame_rate", 24.0)
        active_index = max(1, min(len(takes), _safe_int(take_index, 1))) - 1
        active_take = copy.deepcopy(takes[active_index])
        generation_index["active_take"] = active_index + 1
        out_linx = _clone_linx(cine_linx)
        _apply_active_take(out_linx, generation_index, active_take, str(take_track_layout))
        out_linx["type"] = SUPERNODE_LINX_TYPE
        out_linx["mode"] = "iamccs_multigeneration_take"
        out_linx.setdefault("chain", []).append({
            "role": "multigeneration_take_picker",
            "name": "IAMCCS_MultiTimelineTakePicker",
            "active_take": active_index + 1,
        })
        _refresh_linx_index(out_linx)
        take_timeline = _timeline_for_take(active_take, _safe_float(generation_index.get("frame_rate", 24.0), 24.0), str(take_track_layout))
        report = _json_dump({
            "node": "IAMCCS_MultiTimelineTakePicker",
            "active_take": active_index + 1,
            "timeline_id": str(active_take.get("timeline_id", "")),
            "duration_frames": _safe_int(active_take.get("duration_frames", 0), 0),
            "audio_segments": len(active_take.get("audioSegments", []) if isinstance(active_take.get("audioSegments"), list) else []),
        })
        _outputs(out_linx)["report"] = report
        return out_linx, _json_dump(active_take), _json_dump(take_timeline), report


def _video_components(video: Any):
    if video is None:
        return None
    if not hasattr(video, "get_components"):
        raise ValueError("IAMCCS Video Hard Concat: input is not a Comfy VIDEO object.")
    return video.get_components()


def _normalize_audio_channels(waveform: torch.Tensor, channels: int) -> torch.Tensor:
    if waveform.ndim != 3:
        raise ValueError("IAMCCS Video Hard Concat: AUDIO waveform must be [batch, channels, samples].")
    if waveform.shape[1] == channels:
        return waveform
    if waveform.shape[1] == 1 and channels == 2:
        return waveform.repeat(1, 2, 1)
    if waveform.shape[1] < channels:
        pad = torch.zeros(
            waveform.shape[0],
            channels - waveform.shape[1],
            waveform.shape[2],
            dtype=waveform.dtype,
            device=waveform.device,
        )
        return torch.cat((waveform, pad), dim=1)
    return waveform[:, :channels, :]


def _concat_audio(audio_items: List[Tuple[Any, int, float]]) -> Dict[str, Any] | None:
    usable = [audio for audio, _, _ in audio_items if isinstance(audio, dict) and audio.get("waveform") is not None]
    if not usable:
        return None
    target_rate = int(usable[0].get("sample_rate") or 44100)
    max_channels = max(int(audio["waveform"].shape[1]) for audio in usable)
    pieces = []
    for audio, frame_count, fps in audio_items:
        expected_samples = max(1, int(math.ceil((max(0, frame_count) / max(1.0, fps)) * target_rate)))
        if isinstance(audio, dict) and audio.get("waveform") is not None:
            waveform = audio["waveform"]
            sample_rate = int(audio.get("sample_rate") or target_rate)
            if sample_rate != target_rate:
                waveform = torchaudio.functional.resample(waveform, sample_rate, target_rate)
            waveform = _normalize_audio_channels(waveform, max_channels)
            if waveform.shape[-1] < expected_samples:
                pad = torch.zeros(
                    waveform.shape[0],
                    waveform.shape[1],
                    expected_samples - waveform.shape[-1],
                    dtype=waveform.dtype,
                    device=waveform.device,
                )
                waveform = torch.cat((waveform, pad), dim=2)
            else:
                waveform = waveform[..., :expected_samples]
        else:
            waveform = torch.zeros(1, max_channels, expected_samples)
        pieces.append(waveform)
    return {"waveform": torch.cat(pieces, dim=2), "sample_rate": target_rate}


class IAMCCS_VideoHardConcat:
    """Hard-concatenate generated take videos into a final VIDEO object."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_1": ("VIDEO",),
                "audio_policy": (["concat_clip_audio", "use_master_audio", "first_video_audio", "silent"], {"default": "concat_clip_audio"}),
                "fps_mode": (["from_first_video", "override_fps"], {"default": "from_first_video"}),
                "override_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
            },
            "optional": {
                "video_2": ("VIDEO",),
                "video_3": ("VIDEO",),
                "video_4": ("VIDEO",),
                "video_5": ("VIDEO",),
                "master_audio": ("AUDIO",),
                "concat_plan_json": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("VIDEO", "IMAGE", "STRING")
    RETURN_NAMES = ("video", "frames", "report")
    FUNCTION = "concat"
    CATEGORY = "IAMCCS/Cine/Multigeneration"

    def concat(
        self,
        video_1,
        audio_policy,
        fps_mode,
        override_fps,
        video_2=None,
        video_3=None,
        video_4=None,
        video_5=None,
        master_audio=None,
        concat_plan_json="",
    ):
        videos = [video for video in (video_1, video_2, video_3, video_4, video_5) if video is not None]
        if not videos:
            raise ValueError("IAMCCS Video Hard Concat: at least video_1 is required.")

        components = [_video_components(video) for video in videos]
        first = components[0]
        first_shape = tuple(first.images.shape[1:3])
        first_device = first.images.device
        frame_batches = []
        frame_counts = []
        for index, comp in enumerate(components):
            if tuple(comp.images.shape[1:3]) != first_shape:
                raise ValueError(
                    "IAMCCS Video Hard Concat: all takes must share the same height and width. "
                    f"video_1={first_shape}, video_{index + 1}={tuple(comp.images.shape[1:3])}"
                )
            images = comp.images
            if images.device != first_device:
                images = images.to(first_device)
            frame_batches.append(images)
            frame_counts.append(int(images.shape[0]))

        frames = torch.cat(frame_batches, dim=0)
        if str(fps_mode) == "override_fps":
            fps = float(override_fps)
        else:
            fps = float(first.frame_rate)
        frame_rate = Fraction(round(max(1.0, fps) * 1000), 1000)

        audio = None
        if str(audio_policy) == "use_master_audio":
            audio = master_audio
        elif str(audio_policy) == "first_video_audio":
            audio = first.audio
        elif str(audio_policy) == "concat_clip_audio":
            audio = _concat_audio([
                (comp.audio, int(comp.images.shape[0]), float(comp.frame_rate))
                for comp in components
            ])

        video = InputImpl.VideoFromComponents(Types.VideoComponents(images=frames, audio=audio, frame_rate=frame_rate))
        concat_plan = _safe_json_loads(concat_plan_json, {})
        report = _json_dump({
            "node": "IAMCCS_VideoHardConcat",
            "policy": "hard_cut_tensor_concat",
            "take_count": len(videos),
            "frames_per_take": frame_counts,
            "total_frames": int(frames.shape[0]),
            "fps": float(frame_rate),
            "duration_seconds": int(frames.shape[0]) / max(1.0, float(frame_rate)),
            "audio_policy": str(audio_policy),
            "has_audio": audio is not None,
            "concat_plan_takes": len(concat_plan.get("takes", [])) if isinstance(concat_plan, dict) else 0,
        })
        return video, frames, report


NODE_CLASS_MAPPINGS = {
    "IAMCCS_MultiTimelineBridge": IAMCCS_MultiTimelineBridge,
    "IAMCCS_MultiTimelineTakePicker": IAMCCS_MultiTimelineTakePicker,
    "IAMCCS_VideoHardConcat": IAMCCS_VideoHardConcat,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_MultiTimelineBridge": "IAMCCS MultiTimeline Bridge",
    "IAMCCS_MultiTimelineTakePicker": "IAMCCS MultiTimeline Take Picker",
    "IAMCCS_VideoHardConcat": "IAMCCS Video Hard Concat",
}
