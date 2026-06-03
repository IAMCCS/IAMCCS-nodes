from __future__ import annotations

import copy
import json
from typing import Any, Dict, List


SUPERNODE_LINX_TYPE = "IAMCCS_SUPERNODE_LINX"


def _safe_json_loads(value: Any, fallback: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value or ""))
    except Exception:
        return fallback


def _json_report(data: Dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def _clone_linx(cine_linx: Any) -> Dict[str, Any]:
    return copy.deepcopy(cine_linx) if isinstance(cine_linx, dict) else {
        "type": SUPERNODE_LINX_TYPE,
        "mode": "iamccs_audio_board_arranger",
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


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(fallback)


def _safe_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return int(fallback)


def _first_list(*values: Any) -> List[Any]:
    for value in values:
        if isinstance(value, list):
            return value
    return []


def _first_dict(*values: Any) -> Dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


class IAMCCS_AudioBoardArranger:
    """DAW-style audio lane arranger that writes Shotboard V3 audioSegments through cine_linx."""

    DEFAULT_DATA = json.dumps({
        "schema": "iamccs.audio_board_arranger",
        "schema_version": 1,
        "audioSegments": [],
        "audioTrackCount": 5,
        "masterAudioGain": 1.0,
        "masterAudioNormalize": False,
        "masterMono": False,
        "masterBus": {
            "limiter": True,
            "ceilingDb": -1.0,
            "compressor": 0.0,
            "width": 1.0,
            "reverbSend": 0.0,
            "delaySend": 0.0,
        },
        "audioBusMode": "all_tracks",
        "onlyFirstTrack": False,
        "audioSyncMode": "timeline_audio",
        "duration_seconds": 26.0,
        "frame_rate": 24.0,
        "status": {"edits": []},
        "view": {"timeZoom": 1.0, "trackHeight": 168},
    }, indent=2)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "arranger_data": ("STRING", {
                    "default": cls.DEFAULT_DATA,
                    "multiline": True,
                    "tooltip": "Edited by the IAMCCS AudioBoardArranger UI. Uses Shotboard V3 audioSegments schema.",
                }),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "sync_policy": ([
                    "audio_lanes_drive_custom_audio",
                    "metadata_only_report",
                ], {"default": "audio_lanes_drive_custom_audio"}),
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
    RETURN_NAMES = ("cine_linx", "audio_timeline_json", "report")
    FUNCTION = "arrange"
    OUTPUT_NODE = True
    CATEGORY = "IAMCCS/Cine/Audio"

    @staticmethod
    def _segments(data: Any) -> List[Dict[str, Any]]:
        raw = data.get("audioSegments", []) if isinstance(data, dict) else []
        if not isinstance(raw, list):
            return []
        out: List[Dict[str, Any]] = []
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            seg = dict(item)
            seg["id"] = str(seg.get("id") or f"aud_{index + 1:03d}")
            seg["type"] = "audio"
            seg["start"] = max(0, _safe_int(seg.get("start", 0), 0))
            seg["length"] = max(1, _safe_int(seg.get("length", seg.get("audioDurationFrames", 1)), 1))
            seg["track"] = max(0, _safe_int(seg.get("track", 0), 0))
            seg["trimStart"] = max(0, _safe_int(seg.get("trimStart", 0), 0))
            seg["audioDurationFrames"] = max(seg["length"], _safe_int(seg.get("audioDurationFrames", seg["length"]), seg["length"]))
            seg["gain"] = max(0.0, min(4.0, _safe_float(seg.get("gain", seg.get("volume", 1.0)), 1.0)))
            seg["pan"] = max(-1.0, min(1.0, _safe_float(seg.get("pan", 0.0), 0.0)))
            seg["fadeInFrames"] = max(0, _safe_int(seg.get("fadeInFrames", 0), 0))
            seg["fadeOutFrames"] = max(0, _safe_int(seg.get("fadeOutFrames", 0), 0))
            seg["normalizeAudio"] = bool(seg.get("normalizeAudio", False))
            seg["mute"] = bool(seg.get("mute", False))
            seg["solo"] = bool(seg.get("solo", False))
            seg["purpose"] = str(seg.get("purpose", "dialogue_or_music") or "dialogue_or_music")
            out.append(seg)
        return sorted(out, key=lambda item: (int(item.get("track", 0)), int(item.get("start", 0))))

    @staticmethod
    def _track_settings_for_count(track_settings: Any, count: int) -> List[Dict[str, Any]]:
        source = track_settings if isinstance(track_settings, list) else []
        out: List[Dict[str, Any]] = []
        for index in range(max(1, int(count))):
            item = copy.deepcopy(source[index]) if index < len(source) and isinstance(source[index], dict) else {}
            item["volume"] = max(0.0, min(2.0, _safe_float(item.get("volume", 1.0), 1.0)))
            item["pan"] = max(-1.0, min(1.0, _safe_float(item.get("pan", 0.0), 0.0)))
            item["mute"] = bool(item.get("mute", False))
            item["solo"] = bool(item.get("solo", False))
            item["normalize"] = bool(item.get("normalize", False))
            item["reverb"] = bool(item.get("reverb", False))
            item["reverbSend"] = max(0.0, min(1.0, _safe_float(item.get("reverbSend", 0.0), 0.0)))
            out.append(item)
        return out

    @staticmethod
    def _segments_with_track_mix(segments: List[Dict[str, Any]], track_settings: Any, track_count: int) -> List[Dict[str, Any]]:
        settings = IAMCCS_AudioBoardArranger._track_settings_for_count(track_settings, track_count)
        solo_tracks = {index for index, item in enumerate(settings) if bool(item.get("solo", False))}
        solo_clips = any(bool(seg.get("solo", False)) for seg in segments)
        out: List[Dict[str, Any]] = []
        for seg in segments:
            item = copy.deepcopy(seg)
            track = max(0, _safe_int(item.get("track", 0), 0))
            track_state = settings[track] if track < len(settings) else {}
            track_volume = max(0.0, min(2.0, _safe_float(track_state.get("volume", 1.0), 1.0)))
            track_pan = max(-1.0, min(1.0, _safe_float(track_state.get("pan", 0.0), 0.0)))
            clip_gain = max(0.0, min(4.0, _safe_float(item.get("gain", item.get("volume", 1.0)), 1.0)))
            clip_pan = max(-1.0, min(1.0, _safe_float(item.get("pan", 0.0), 0.0)))
            item["trackVolume"] = track_volume
            item["trackPan"] = track_pan
            item["trackMute"] = bool(track_state.get("mute", False))
            item["trackSolo"] = bool(track_state.get("solo", False))
            item["gain"] = max(0.0, min(4.0, clip_gain * track_volume))
            item["pan"] = max(-1.0, min(1.0, clip_pan + track_pan))
            if bool(track_state.get("normalize", False)):
                item["normalizeAudio"] = True
            if bool(track_state.get("reverb", False)) and _safe_float(item.get("reverbSend", 0.0), 0.0) <= 0.0:
                item["reverbSend"] = max(0.15, _safe_float(track_state.get("reverbSend", 0.0), 0.0))
            if bool(track_state.get("mute", False)):
                item["mute"] = True
            if solo_tracks and track not in solo_tracks:
                item["mute"] = True
            if solo_clips and not bool(item.get("solo", False)):
                item["mute"] = True
            out.append(item)
        return out

    @staticmethod
    def _merge_audio_into_timeline(timeline_text: Any, data: Dict[str, Any], fps: float) -> str:
        timeline = _safe_json_loads(timeline_text, {})
        if not isinstance(timeline, dict):
            timeline = {}
        timeline.setdefault("schema", "iamccs.cine.filmmaker_timeline")
        timeline["schema_version"] = max(2, _safe_int(timeline.get("schema_version", 2), 2))
        segments = IAMCCS_AudioBoardArranger._segments(data)
        timeline["audioSegments"] = segments
        timeline["audioTrackCount"] = max(1, _safe_int(data.get("audioTrackCount", 3), 3))
        timeline["masterAudioGain"] = max(0.0, min(2.0, _safe_float(data.get("masterAudioGain", 1.0), 1.0)))
        timeline["masterAudioNormalize"] = bool(data.get("masterAudioNormalize", False))
        timeline["masterMono"] = bool(data.get("masterMono", False))
        timeline["masterBus"] = data.get("masterBus") if isinstance(data.get("masterBus"), dict) else {}
        timeline["trackSettings"] = data.get("trackSettings") if isinstance(data.get("trackSettings"), list) else []
        timeline["audioBusMode"] = str(data.get("audioBusMode", "all_tracks") or "all_tracks")
        timeline["onlyFirstTrack"] = bool(data.get("onlyFirstTrack", False))
        timeline["audioSyncMode"] = str(data.get("audioSyncMode", "timeline_audio") or "timeline_audio")
        timeline["use_custom_audio"] = any(str(seg.get("audioFile", "")).strip() or str(seg.get("audioB64", "")).strip() for seg in segments)
        timeline["frame_rate"] = _safe_float(timeline.get("frame_rate", data.get("frame_rate", fps)), fps)
        max_end = max([_safe_int(seg.get("start", 0), 0) + _safe_int(seg.get("length", 1), 1) for seg in segments] or [0])
        visual_segments = timeline.get("segments", [])
        visual_end = max([
            _safe_int(seg.get("start", seg.get("frame", 0)), 0) + _safe_int(seg.get("length", seg.get("len", 1)), 1)
            for seg in visual_segments
            if isinstance(seg, dict) and str(seg.get("type", "image") or "image").lower() != "audio"
        ] or [0]) if isinstance(visual_segments, list) else 0
        current_duration = _safe_float(timeline.get("duration_seconds", data.get("duration_seconds", 0.0)), 0.0)
        if max_end > 0:
            segment_duration = max(max_end, visual_end) / max(1.0, float(fps))
            if current_duration > 0 and current_duration <= segment_duration * 1.35:
                timeline["duration_seconds"] = max(current_duration, segment_duration)
            else:
                timeline["duration_seconds"] = segment_duration
        audio_data = {
            "audioSegments": segments,
            "audioTrackCount": timeline["audioTrackCount"],
            "use_custom_audio": timeline["use_custom_audio"],
            "masterAudioGain": timeline["masterAudioGain"],
            "masterAudioNormalize": timeline["masterAudioNormalize"],
            "masterBus": timeline["masterBus"],
            "trackSettings": timeline["trackSettings"],
            "audioBusMode": timeline["audioBusMode"],
            "onlyFirstTrack": timeline["onlyFirstTrack"],
            "audioSyncMode": timeline["audioSyncMode"],
        }
        timeline["audio_data"] = json.dumps(audio_data, ensure_ascii=False)
        return json.dumps(timeline, ensure_ascii=False, indent=2)

    def arrange(self, arranger_data, frame_rate, sync_policy, cine_linx=None, unique_id=None, extra_pnginfo=None):
        data = _safe_json_loads(arranger_data, {})
        if not isinstance(data, dict):
            data = _safe_json_loads(self.DEFAULT_DATA, {})
        fps = max(1.0, float(frame_rate))
        upstream_linx = _clone_linx(cine_linx)
        upstream_resources = _resources(upstream_linx)
        upstream_outputs = _outputs(upstream_linx)
        upstream_payload = _payload(upstream_linx)
        if not self._segments(data):
            candidates = [
                upstream_outputs.get("audio_timeline_json"),
                upstream_resources.get("cine_audio_timeline_json"),
                upstream_payload.get("audio_timeline_json"),
            ]
            for candidate in candidates:
                parsed = _safe_json_loads(candidate, {})
                if isinstance(parsed, dict) and self._segments(parsed):
                    data = parsed
                    break
            if not self._segments(data):
                tracks = upstream_resources.get("cine_audio_tracks")
                if isinstance(tracks, dict) and isinstance(tracks.get("segments"), list):
                    data = copy.deepcopy(data) if isinstance(data, dict) else _safe_json_loads(self.DEFAULT_DATA, {})
                    data["audioSegments"] = [dict(seg) for seg in tracks.get("segments") if isinstance(seg, dict)]
                    data["audioTrackCount"] = max(1, _safe_int(tracks.get("track_count", data.get("audioTrackCount", 2)), 2))
                    data["duration_seconds"] = _safe_float(tracks.get("duration_seconds", data.get("duration_seconds", 0.0)), 0.0)
        raw_source_segments = self._segments(data)
        source_track_count = max(1, _safe_int(data.get("audioTrackCount", 3), 3))
        upstream_tracks = upstream_resources.get("cine_audio_tracks") if isinstance(upstream_resources.get("cine_audio_tracks"), dict) else {}
        upstream_timeline_data = _safe_json_loads(
            upstream_outputs.get("audio_timeline_json")
            or upstream_resources.get("cine_audio_timeline_json")
            or upstream_payload.get("audio_timeline_json"),
            {}
        )
        if not isinstance(upstream_timeline_data, dict):
            upstream_timeline_data = {}
        track_settings = copy.deepcopy(_first_list(
            data.get("trackSettings") if isinstance(data.get("trackSettings"), list) and data.get("trackSettings") else None,
            upstream_tracks.get("all_track_settings") if isinstance(upstream_tracks.get("all_track_settings"), list) and upstream_tracks.get("all_track_settings") else None,
            upstream_tracks.get("track_settings") if isinstance(upstream_tracks.get("track_settings"), list) and upstream_tracks.get("track_settings") else None,
            upstream_timeline_data.get("trackSettings") if isinstance(upstream_timeline_data.get("trackSettings"), list) and upstream_timeline_data.get("trackSettings") else None,
        ))
        master_bus = copy.deepcopy(_first_dict(
            data.get("masterBus") if isinstance(data.get("masterBus"), dict) and data.get("masterBus") else None,
            upstream_tracks.get("master_bus") if isinstance(upstream_tracks.get("master_bus"), dict) and upstream_tracks.get("master_bus") else None,
            upstream_timeline_data.get("masterBus") if isinstance(upstream_timeline_data.get("masterBus"), dict) and upstream_timeline_data.get("masterBus") else None,
        ))
        track_settings = self._track_settings_for_count(track_settings, source_track_count)
        source_segments = self._segments_with_track_mix(raw_source_segments, track_settings, source_track_count)
        bus_mode_raw = str(data.get("audioBusMode", "all_tracks") or "all_tracks").strip().lower()
        only_first = bool(data.get("onlyFirstTrack", False)) or bus_mode_raw in {"only_first", "shotboard_only_first", "first_track", "first"}
        bus_mode = "all_tracks"
        shotboard_mode = "only_first" if only_first else "all_tracks"
        segments = source_segments
        shotboard_segments = [seg for seg in source_segments if _safe_int(seg.get("track", 0), 0) == 0] if only_first else source_segments
        export_track_count = source_track_count
        shotboard_track_count = 1 if only_first else source_track_count
        export_track_settings = copy.deepcopy(track_settings)
        shotboard_track_settings = copy.deepcopy(track_settings[:1]) if only_first else copy.deepcopy(track_settings)
        full_data = copy.deepcopy(data)
        full_data["audioSegments"] = source_segments
        full_data["audioTrackCount"] = source_track_count
        full_data["audioBusMode"] = "only_first" if only_first else "all_tracks"
        full_data["onlyFirstTrack"] = only_first
        data = copy.deepcopy(data)
        data["audioSegments"] = segments
        data["audioTrackCount"] = export_track_count
        data["trackSettings"] = export_track_settings
        data["masterBus"] = master_bus
        data["audioBusMode"] = bus_mode
        data["onlyFirstTrack"] = False
        data["frame_rate"] = fps
        shotboard_data = copy.deepcopy(data)
        shotboard_data["audioSegments"] = shotboard_segments
        shotboard_data["audioTrackCount"] = shotboard_track_count
        shotboard_data["trackSettings"] = shotboard_track_settings
        shotboard_data["audioBusMode"] = "shotboard_only_first" if only_first else "all_tracks"
        shotboard_data["onlyFirstTrack"] = only_first
        audio_timeline = {
            "audioSegments": segments,
            "audioTrackCount": export_track_count,
            "masterAudioGain": max(0.0, min(2.0, _safe_float(data.get("masterAudioGain", 1.0), 1.0))),
            "masterAudioNormalize": bool(data.get("masterAudioNormalize", False)),
            "masterMono": bool(data.get("masterMono", False)),
            "masterBus": master_bus,
            "trackSettings": export_track_settings,
            "audioBusMode": bus_mode,
            "onlyFirstTrack": False,
            "shotboardAudioSegments": shotboard_segments,
            "shotboardAudioTrackCount": shotboard_track_count,
            "shotboardAudioMode": shotboard_mode,
            "audioSyncMode": str(data.get("audioSyncMode", "timeline_audio") or "timeline_audio"),
        }
        shotboard_audio_timeline = copy.deepcopy(audio_timeline)
        shotboard_audio_timeline["audioSegments"] = shotboard_segments
        shotboard_audio_timeline["audioTrackCount"] = shotboard_track_count
        shotboard_audio_timeline["trackSettings"] = shotboard_track_settings
        shotboard_audio_timeline["audioBusMode"] = "shotboard_only_first" if only_first else "all_tracks"
        shotboard_audio_timeline["onlyFirstTrack"] = only_first
        audio_timeline_json = json.dumps(shotboard_audio_timeline, ensure_ascii=False, indent=2)
        bus_audio_timeline_json = json.dumps(audio_timeline, ensure_ascii=False, indent=2)
        effect_graph = {
            "schema": "iamccs.audio_effect_graph",
            "schema_version": 1,
            "clips": [{
                "id": str(seg.get("id", "")),
                "track": _safe_int(seg.get("track", 0), 0),
                "start": _safe_int(seg.get("start", 0), 0),
                "length": _safe_int(seg.get("length", 1), 1),
                "linkedVisualId": str(seg.get("linkedVisualId", "") or ""),
                "effects": {
                    "gain": _safe_float(seg.get("gain", 1.0), 1.0),
                    "pan": _safe_float(seg.get("pan", 0.0), 0.0),
                    "hpfHz": _safe_float(seg.get("hpfHz", 0.0), 0.0),
                    "lpfHz": _safe_float(seg.get("lpfHz", 22000.0), 22000.0),
                    "eqLowDb": _safe_float(seg.get("eqLowDb", 0.0), 0.0),
                    "eqMidDb": _safe_float(seg.get("eqMidDb", 0.0), 0.0),
                    "eqHighDb": _safe_float(seg.get("eqHighDb", 0.0), 0.0),
                    "compressor": _safe_float(seg.get("compressor", 0.0), 0.0),
                    "noiseGateDb": _safe_float(seg.get("noiseGateDb", -60.0), -60.0),
                    "ducking": _safe_float(seg.get("ducking", 0.0), 0.0),
                    "reverbSend": _safe_float(seg.get("reverbSend", 0.0), 0.0),
                    "delaySend": _safe_float(seg.get("delaySend", 0.0), 0.0),
                    "stereoWidth": _safe_float(seg.get("stereoWidth", 1.0), 1.0),
                    "transient": _safe_float(seg.get("transient", 0.0), 0.0),
                    "denoise": _safe_float(seg.get("denoise", 0.0), 0.0),
                },
            } for seg in segments],
            "masterBus": audio_timeline["masterBus"],
            "masterMono": audio_timeline["masterMono"],
            "trackSettings": export_track_settings,
            "busMode": bus_mode,
        }
        effect_graph_json = json.dumps(effect_graph, ensure_ascii=False, indent=2)
        has_media = any(str(seg.get("audioFile", "")).strip() or str(seg.get("audioB64", "")).strip() for seg in segments)
        shotboard_has_media = any(str(seg.get("audioFile", "")).strip() or str(seg.get("audioB64", "")).strip() for seg in shotboard_segments)
        source_max_end = max([_safe_int(seg.get("start", 0), 0) + _safe_int(seg.get("length", 1), 1) for seg in segments] or [0])
        shotboard_max_end = max([_safe_int(seg.get("start", 0), 0) + _safe_int(seg.get("length", 1), 1) for seg in shotboard_segments] or [0])
        duration_frames = shotboard_max_end or source_max_end
        duration_s = (duration_frames / fps) if duration_frames else max(0.0, _safe_float(data.get("duration_seconds", 0.0), 0.0))

        out_linx = _clone_linx(cine_linx)
        resources = _resources(out_linx)
        outputs = _outputs(out_linx)
        payload = _payload(out_linx)
        upstream_timeline = resources.get("cine_board_timeline_data") or payload.get("timeline_data") or outputs.get("timeline_data")
        merged_timeline = self._merge_audio_into_timeline(upstream_timeline, shotboard_data, fps) if upstream_timeline else ""

        resources.update({
            "cine_audio_timeline_json": audio_timeline_json,
            "cine_audio_bus_timeline_json": bus_audio_timeline_json,
            "cine_audio_tracks": {
                "source": "IAMCCS_AudioBoardArranger",
                "segments": segments,
                "all_segments": source_segments,
                "shotboard_segments": shotboard_segments,
                "source_end_frames": int(source_max_end),
                "shotboard_end_frames": int(shotboard_max_end),
                "duration_frames": int(duration_frames),
                "track_count": audio_timeline["audioTrackCount"],
                "source_track_count": source_track_count,
                "shotboard_track_count": shotboard_track_count,
                "selected_tracks": list(range(max(1, source_track_count))),
                "shotboard_selected_tracks": [0] if only_first else list(range(max(1, source_track_count))),
                "bus_mode": bus_mode,
                "shotboard_bus_mode": shotboard_mode,
                "master_gain": audio_timeline["masterAudioGain"],
                "master_normalize": audio_timeline["masterAudioNormalize"],
                "master_mono": audio_timeline["masterMono"],
                "master_bus": audio_timeline["masterBus"],
                "track_settings": export_track_settings,
                "all_track_settings": copy.deepcopy(track_settings),
                "shotboard_track_settings": shotboard_track_settings,
            },
            "cine_audio_layers": {
                "arranger": full_data,
                "bus": data,
                "export": shotboard_data,
                "policy": str(sync_policy),
                "bus_mode": bus_mode,
                "shotboard_bus_mode": shotboard_mode,
            },
            "cine_audio_effect_graph_json": effect_graph_json,
            "cine_use_custom_audio": bool(shotboard_has_media and str(sync_policy) == "audio_lanes_drive_custom_audio"),
        })
        if duration_s > 0:
            resources["cine_duration_seconds"] = float(duration_s)
            if duration_frames > 0:
                resources["cine_max_frames"] = int(duration_frames)
                payload["max_frames"] = int(duration_frames)
            payload["duration_seconds"] = float(duration_s)
        if merged_timeline:
            resources["cine_board_timeline_data"] = merged_timeline
            outputs["timeline_data"] = merged_timeline
            payload["timeline_data"] = merged_timeline
        payload.update({
            "audio_board_arranger": True,
            "audioSegments": shotboard_segments,
            "audioTrackCount": shotboard_track_count,
            "trackSettings": shotboard_track_settings,
            "audioBusMode": "shotboard_only_first" if only_first else "all_tracks",
            "onlyFirstTrack": only_first,
            "use_custom_audio": bool(shotboard_has_media and str(sync_policy) == "audio_lanes_drive_custom_audio"),
            "audioSyncMode": audio_timeline["audioSyncMode"],
        })
        outputs.update({
            "audio_timeline_json": audio_timeline_json,
            "audio_bus_timeline_json": bus_audio_timeline_json,
            "audio_effect_graph_json": effect_graph_json,
            "duration_seconds": float(duration_s),
            "max_frames": int(duration_frames) if duration_frames > 0 else 0,
        })
        out_linx["type"] = SUPERNODE_LINX_TYPE
        out_linx["mode"] = "iamccs_audio_board_arranger"
        out_linx.setdefault("chain", []).append({"role": "audio_arranger", "name": "IAMCCS_AudioBoardArranger"})
        _refresh_linx_index(out_linx)
        report = _json_report({
            "node": "IAMCCS_AudioBoardArranger",
            "segments": len(segments),
            "source_segments": len(source_segments),
            "shotboard_segments": len(shotboard_segments),
            "tracks": audio_timeline["audioTrackCount"],
            "source_tracks": source_track_count,
            "shotboard_tracks": shotboard_track_count,
            "bus_mode": bus_mode,
            "shotboard_bus_mode": shotboard_mode,
            "has_media": bool(has_media),
            "duration_seconds": float(duration_s),
            "sync_policy": str(sync_policy),
            "truth": "External/generated audio that should drive video is exported as Shotboard V3 audioSegments/custom audio through cine_linx.",
        })
        resources["cine_report"] = report
        outputs["report"] = report

        # Keep the visual AudioBoard widget synchronized with the runtime lanes.
        # This is intentionally local to the executing workflow metadata: it lets the user
        # see real audioFile clips after queue, instead of only invisible cine_linx data.
        try:
            workflow = None
            if isinstance(extra_pnginfo, list) and extra_pnginfo and isinstance(extra_pnginfo[0], dict):
                workflow = extra_pnginfo[0].get("workflow")
            elif isinstance(extra_pnginfo, dict):
                workflow = extra_pnginfo.get("workflow")
            uid = unique_id[0] if isinstance(unique_id, list) and unique_id else unique_id
            if workflow and uid is not None:
                for item in workflow.get("nodes", []):
                    if str(item.get("id")) == str(uid):
                        widgets = list(item.get("widgets_values") or [])
                        while len(widgets) < 4:
                            widgets.append("")
                        widgets[0] = audio_timeline_json
                        widgets[3] = report
                        item["widgets_values"] = widgets
                        break
        except Exception as exc:
            print(f"[IAMCCS AudioBoardArranger] UI widget sync skipped: {exc}")

        return {
            "ui": {
                "text": [report],
                "iamccs_audio_board": [audio_timeline_json],
            },
            "result": (out_linx, audio_timeline_json, report),
        }


NODE_CLASS_MAPPINGS = {
    "IAMCCS_AudioBoardArranger": IAMCCS_AudioBoardArranger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_AudioBoardArranger": "IAMCCS AudioBoard Arranger",
}
