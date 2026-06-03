from __future__ import annotations

import copy
import json
from typing import Any, Dict, List


SUPERNODE_LINX_TYPE = "IAMCCS_SUPERNODE_LINX"
MAX_TRACK_OUTS = 5


def _safe_json_loads(value: Any, fallback: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value or ""))
    except Exception:
        return fallback


def _json_dump(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def _clone_linx(cine_linx: Any) -> Dict[str, Any]:
    return copy.deepcopy(cine_linx) if isinstance(cine_linx, dict) else {
        "type": SUPERNODE_LINX_TYPE,
        "mode": "iamccs_bus_out",
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


def _segments(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        seg = copy.deepcopy(item)
        seg["id"] = str(seg.get("id") or f"bus_seg_{index + 1:03d}")
        seg["track"] = max(0, _safe_int(seg.get("track", 0), 0))
        seg["start"] = max(0, _safe_int(seg.get("start", 0), 0))
        seg["length"] = max(1, _safe_int(seg.get("length", seg.get("audioDurationFrames", 1)), 1))
        seg["gain"] = max(0.0, min(4.0, _safe_float(seg.get("gain", seg.get("volume", 1.0)), 1.0)))
        seg["pan"] = max(-1.0, min(1.0, _safe_float(seg.get("pan", 0.0), 0.0)))
        seg["mute"] = bool(seg.get("mute", False))
        seg["solo"] = bool(seg.get("solo", False))
        out.append(seg)
    return sorted(out, key=lambda seg: (int(seg.get("track", 0)), int(seg.get("start", 0))))


def _settings_at(settings: Any, track_index: int) -> Dict[str, Any]:
    if isinstance(settings, list) and 0 <= track_index < len(settings) and isinstance(settings[track_index], dict):
        return copy.deepcopy(settings[track_index])
    return {}


def _segments_with_track_mix(segments: List[Dict[str, Any]], settings: Any, filter_muted: bool = False) -> List[Dict[str, Any]]:
    track_settings = settings if isinstance(settings, list) else []
    solo_tracks = {
        index for index, item in enumerate(track_settings)
        if isinstance(item, dict) and bool(item.get("solo", False))
    }
    solo_clips = any(bool(seg.get("solo", False)) for seg in segments)
    out: List[Dict[str, Any]] = []
    for seg in segments:
        item = copy.deepcopy(seg)
        track = max(0, _safe_int(item.get("track", 0), 0))
        track_state = _settings_at(track_settings, track)
        track_volume = max(0.0, min(2.0, _safe_float(track_state.get("volume", item.get("trackVolume", 1.0)), 1.0)))
        track_pan = max(-1.0, min(1.0, _safe_float(track_state.get("pan", item.get("trackPan", 0.0)), 0.0)))
        if "trackVolume" not in item:
            item["gain"] = max(0.0, min(4.0, _safe_float(item.get("gain", 1.0), 1.0) * track_volume))
            item["trackVolume"] = track_volume
        if "trackPan" not in item:
            item["pan"] = max(-1.0, min(1.0, _safe_float(item.get("pan", 0.0), 0.0) + track_pan))
            item["trackPan"] = track_pan
        if bool(track_state.get("mute", False)):
            item["mute"] = True
        if solo_tracks and track not in solo_tracks:
            item["mute"] = True
        if solo_clips and not bool(item.get("solo", False)):
            item["mute"] = True
        if filter_muted and bool(item.get("mute", False)):
            continue
        out.append(item)
    return out


def _is_only_first_mode(value: Any) -> bool:
    return str(value or "").strip().lower() in {"only_first", "shotboard_only_first", "first_track", "first"}


def _max_end(segments: List[Dict[str, Any]]) -> int:
    return max([_safe_int(seg.get("start", 0), 0) + _safe_int(seg.get("length", 1), 1) for seg in segments] or [0])


def _has_stereo(segments: List[Dict[str, Any]]) -> bool:
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        if str(seg.get("channelMode", "")).strip().lower() == "stereo":
            return True
        if _safe_int(seg.get("channelCount", 1), 1) > 1:
            return True
        if _safe_float(seg.get("stereoWidth", 1.0), 1.0) > 1.0:
            return True
    return False


def _generation_index_from_segments(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    indexed_segments = [
        copy.deepcopy(seg)
        for seg in segments
        if bool(seg.get("multiGenerationClip")) or str(seg.get("timelineId", "") or "").startswith("T")
    ]
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for fallback, seg in enumerate(sorted(indexed_segments, key=lambda item: (
        _safe_int(item.get("multiTakeIndex", item.get("take_index", 1)), 1),
        _safe_int(item.get("track", 0), 0),
        _safe_int(item.get("sourceGlobalStart", item.get("globalStart", item.get("start", 0))), 0),
    ))):
        take_index = max(1, _safe_int(seg.get("multiTakeIndex", seg.get("take_index", fallback + 1)), fallback + 1))
        groups.setdefault(take_index, []).append(seg)

    takes: List[Dict[str, Any]] = []
    for order, take_index in enumerate(sorted(groups), start=1):
        group = groups[take_index]
        track_index = _safe_int(group[0].get("track", order - 1), order - 1)
        timeline_id = str(group[0].get("timelineId") or f"T{take_index:02d}")
        start = min([_safe_int(seg.get("sourceGlobalStart", seg.get("globalStart", seg.get("start", 0))), 0) for seg in group] or [0])
        end = max([
            _safe_int(seg.get(
                "sourceGlobalEnd",
                seg.get("globalEnd", _safe_int(seg.get("sourceGlobalStart", seg.get("globalStart", seg.get("start", 0))), 0) + _safe_int(seg.get("length", 1), 1))
            ), 0)
            for seg in group
        ] or [0])
        local_start = min([_safe_int(seg.get("start", 0), 0) for seg in group] or [0])
        takes.append({
            "timeline_id": timeline_id,
            "take_index": int(order),
            "source_take_index": int(take_index),
            "track_index": int(track_index),
            "track_name": f"A{track_index + 1}",
            "mapping": f"{timeline_id.replace('T0', 'T')} - A{track_index + 1}",
            "audio_chunk_ids": [str(seg.get("id", "")) for seg in group],
            "source_segment_ids": sorted({str(seg.get("sourceSegmentId", "")) for seg in group if str(seg.get("sourceSegmentId", ""))}),
            "start_frames": int(start),
            "end_frames": int(end),
            "local_start_frames": int(local_start),
            "duration_frames": int(max(0, end - start)),
            "segments": group,
        })
    return {
        "schema": "iamccs.audio_bus_out.generation_index",
        "schema_version": 1,
        "source": "IAMCCS_BusOut",
        "take_count": len(takes),
        "lane_policy": "timeline_take_to_audio_lane",
        "mapping": [take["mapping"] for take in takes],
        "takes": takes,
        "truth": "Each multigeneration take is indexed to its respective AudioBoard lane: T1-A1, T2-A2, T3-A3 when the arranger template starts at A1.",
    }


class IAMCCS_BusOut:
    """Expose the AudioBoardArranger master bus and up to five stem tracks from cine_linx."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": ([
                    "from_arranger_bus",
                    "master_plus_stems_max_5",
                    "all_lanes_force",
                ], {"default": "from_arranger_bus"}),
            },
            "optional": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
            },
        }

    RETURN_TYPES = (
        SUPERNODE_LINX_TYPE,
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "cine_linx",
        "master_out",
        "track_1",
        "track_2",
        "track_3",
        "track_4",
        "track_5",
        "bus_manifest",
        "report",
    )
    FUNCTION = "bus_out"
    CATEGORY = "IAMCCS/Cine/Audio"

    def bus_out(self, mode, cine_linx=None):
        out_linx = _clone_linx(cine_linx)
        resources = _resources(out_linx)
        outputs = _outputs(out_linx)
        payload = _payload(out_linx)

        audio_tracks = resources.get("cine_audio_tracks") if isinstance(resources.get("cine_audio_tracks"), dict) else {}
        source_segments = _segments(audio_tracks.get("all_segments") if isinstance(audio_tracks, dict) else payload.get("audioSegments"))
        if not source_segments:
            source_segments = _segments(audio_tracks.get("segments") if isinstance(audio_tracks, dict) else payload.get("audioSegments"))
        if not source_segments:
            source_segments = _segments(payload.get("audioSegments"))
        master_bus = audio_tracks.get("master_bus") if isinstance(audio_tracks.get("master_bus"), dict) else payload.get("masterBus", {})
        if not isinstance(master_bus, dict):
            master_bus = {}
        master_mono = bool(audio_tracks.get("master_mono", payload.get("masterMono", False)))
        effect_graph = resources.get("cine_audio_effect_graph_json") or outputs.get("audio_effect_graph_json") or payload.get("audio_effect_graph_json") or ""
        track_settings = audio_tracks.get("all_track_settings") if isinstance(audio_tracks.get("all_track_settings"), list) else audio_tracks.get("track_settings")
        if not isinstance(track_settings, list):
            track_settings = payload.get("trackSettings", [])
        if not isinstance(track_settings, list):
            track_settings = []
        raw_bus_mode = str(audio_tracks.get("bus_mode") or payload.get("audioBusMode") or "all_tracks")
        shotboard_bus_mode = str(audio_tracks.get("shotboard_bus_mode") or raw_bus_mode or "all_tracks")
        shotboard_segments = _segments(audio_tracks.get("shotboard_segments"))
        selected_segments = source_segments
        selected_track_settings = copy.deepcopy(track_settings)
        if str(mode) == "from_arranger_bus":
            bus_mode = shotboard_bus_mode or raw_bus_mode or "all_tracks"
            if _is_only_first_mode(bus_mode) and shotboard_segments:
                selected_segments = shotboard_segments
                selected_track_settings = copy.deepcopy(track_settings[:1]) if isinstance(track_settings, list) else []
        elif str(mode) == "master_plus_stems_max_5":
            bus_mode = raw_bus_mode or "all_tracks"
        else:
            bus_mode = "all_lanes_force"

        selected_segments_all = _segments_with_track_mix(selected_segments, selected_track_settings, filter_muted=False)
        selected_segments_audible = _segments_with_track_mix(selected_segments, selected_track_settings, filter_muted=True)

        track_outs: List[Dict[str, Any]] = []
        track_jsons: List[str] = []
        for track_index in range(MAX_TRACK_OUTS):
            track_segments = [seg for seg in selected_segments_audible if _safe_int(seg.get("track", 0), 0) == track_index]
            track_out = {
                "schema": "iamccs.audio_bus_out.track",
                "schema_version": 1,
                "source": "IAMCCS_BusOut",
                "stem_role": "track_stem",
                "track_index": track_index,
                "track_name": f"A{track_index + 1}",
                "segments": track_segments,
                "effects": _settings_at(selected_track_settings, track_index),
                "effect_graph_json": effect_graph,
                "duration_frames": _max_end(track_segments),
                "is_stereo": _has_stereo(track_segments),
                "channel_mode": "stereo" if _has_stereo(track_segments) else "mono",
                "has_media": any(str(seg.get("audioFile", "")).strip() or str(seg.get("audioB64", "")).strip() for seg in track_segments),
            }
            track_outs.append(track_out)
            track_jsons.append(_json_dump(track_out))

        master_out = {
            "schema": "iamccs.audio_bus_out.master",
            "schema_version": 1,
            "source": "IAMCCS_BusOut",
            "stem_role": "master_mix",
            "mode": str(mode),
            "bus_mode": bus_mode,
            "shotboard_bus_mode": shotboard_bus_mode,
            "segments": selected_segments_audible,
            "all_segments": selected_segments_all,
            "masterBus": copy.deepcopy(master_bus),
            "masterMono": master_mono,
            "channel_mode": "mono" if master_mono else "stereo",
            "is_stereo": not master_mono,
            "trackSettings": copy.deepcopy(selected_track_settings[:MAX_TRACK_OUTS]),
            "effect_graph_json": effect_graph,
            "duration_frames": _max_end(selected_segments_audible),
            "track_count": MAX_TRACK_OUTS,
            "has_media": any(str(seg.get("audioFile", "")).strip() or str(seg.get("audioB64", "")).strip() for seg in selected_segments_audible),
        }
        manifest = {
            "schema": "iamccs.audio_bus_out.manifest",
            "schema_version": 1,
            "source": "IAMCCS_BusOut",
            "mode": str(mode),
            "bus_mode": bus_mode,
            "shotboard_bus_mode": shotboard_bus_mode,
            "master": master_out,
            "tracks": track_outs,
            "effect_graph_json": effect_graph,
        }
        generation_index = _generation_index_from_segments(selected_segments_audible)
        manifest["generation_index"] = generation_index
        master_json = _json_dump(master_out)
        manifest_json = _json_dump(manifest)
        report = _json_dump({
            "node": "IAMCCS_BusOut",
            "mode": str(mode),
            "bus_mode": bus_mode,
            "shotboard_bus_mode": shotboard_bus_mode,
            "master_segments": len(selected_segments_audible),
            "all_master_segments": len(selected_segments_all),
            "track_segments": [len(track.get("segments", [])) for track in track_outs],
            "generation_take_count": generation_index.get("take_count", 0),
            "generation_mapping": generation_index.get("mapping", []),
            "max_track_outputs": MAX_TRACK_OUTS,
            "truth": "Splits AudioBoardArranger cine_linx audio metadata into one master mix plus up to five per-track stems.",
        })

        resources["cine_audio_bus_out"] = manifest
        if effect_graph:
            resources["cine_audio_bus_effect_graph_json"] = effect_graph
        resources["cine_audio_master_out_json"] = master_json
        resources["cine_audio_track_out_jsons"] = track_jsons
        resources["cine_audio_generation_index"] = generation_index
        resources["cine_audio_generation_index_json"] = _json_dump(generation_index)
        outputs["audio_master_out_json"] = master_json
        for index, track_json in enumerate(track_jsons, start=1):
            outputs[f"audio_track_{index}_json"] = track_json
        outputs["audio_bus_manifest_json"] = manifest_json
        outputs["audio_generation_index_json"] = _json_dump(generation_index)
        outputs["report"] = report
        payload["audio_bus_out"] = manifest
        payload["audio_generation_index"] = generation_index
        out_linx["type"] = SUPERNODE_LINX_TYPE
        out_linx["mode"] = "iamccs_bus_out"
        out_linx.setdefault("chain", []).append({"role": "audio_bus_out", "name": "IAMCCS_BusOut"})
        _refresh_linx_index(out_linx)
        return (
            out_linx,
            master_json,
            track_jsons[0],
            track_jsons[1],
            track_jsons[2],
            track_jsons[3],
            track_jsons[4],
            manifest_json,
            report,
        )


NODE_CLASS_MAPPINGS = {
    "IAMCCS_BusOut": IAMCCS_BusOut,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_BusOut": "IAMCCS BusOut",
}
