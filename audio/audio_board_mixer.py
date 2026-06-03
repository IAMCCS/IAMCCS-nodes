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


def _clone_linx(cine_linx: Any) -> Dict[str, Any]:
    return copy.deepcopy(cine_linx) if isinstance(cine_linx, dict) else {
        "type": SUPERNODE_LINX_TYPE,
        "mode": "iamccs_audio_board_mixer",
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


def _segments(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        seg = copy.deepcopy(item)
        seg["id"] = str(seg.get("id") or f"mix_seg_{index + 1:03d}")
        seg["track"] = max(0, _safe_int(seg.get("track", 0), 0))
        seg["start"] = max(0, _safe_int(seg.get("start", 0), 0))
        seg["length"] = max(1, _safe_int(seg.get("length", seg.get("audioDurationFrames", 1)), 1))
        seg["gain"] = max(0.0, min(4.0, _safe_float(seg.get("gain", seg.get("volume", 1.0)), 1.0)))
        seg["pan"] = max(-1.0, min(1.0, _safe_float(seg.get("pan", 0.0), 0.0)))
        out.append(seg)
    return sorted(out, key=lambda seg: (int(seg.get("track", 0)), int(seg.get("start", 0))))


def _default_track_settings(count: int, existing: Any) -> List[Dict[str, Any]]:
    settings = existing if isinstance(existing, list) else []
    out: List[Dict[str, Any]] = []
    for index in range(max(1, int(count))):
        base = copy.deepcopy(settings[index]) if index < len(settings) and isinstance(settings[index], dict) else {}
        base.setdefault("name", f"A{index + 1}")
        base["volume"] = max(0.0, min(2.0, _safe_float(base.get("volume", 1.0), 1.0)))
        base["pan"] = max(-1.0, min(1.0, _safe_float(base.get("pan", 0.0), 0.0)))
        base["mute"] = bool(base.get("mute", False))
        base["solo"] = bool(base.get("solo", False))
        chain = base.get("effectChain")
        base["effectChain"] = chain if isinstance(chain, list) else []
        out.append(base)
    return out


# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com


def _first_list(*values: Any) -> List[Any]:
    for value in values:
        if isinstance(value, list) and value:
            return value
    return []


class IAMCCS_AudioBoardMixer:
    """Extended AudioBoard mixer surface synchronized through cine_linx."""

    DEFAULT_DATA = json.dumps({
        "schema": "iamccs.audio_board_mixer",
        "schema_version": 1,
        "mirrorTrackSettings": [],
        "masterBus": {
            "limiter": True,
            "compressor": 0.0,
            "ceilingDb": -1.0,
            "width": 1.0,
            "reverbSend": 0.0,
            "delaySend": 0.0,
        },
        "view": {"style": "reaper_channel_strips", "meter_mode": "peak_rms"},
    }, indent=2)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mixer_data": ("STRING", {
                    "default": cls.DEFAULT_DATA,
                    "multiline": True,
                    "tooltip": "Edited by the IAMCCS AudioBoardMixer UI. Mirrors AudioBoard track volume, pan, inserts and master bus state.",
                }),
                "sync_policy": ([
                    "mixer_overrides_cine_linx",
                    "read_only_monitor",
                ], {"default": "mixer_overrides_cine_linx"}),
            },
            "optional": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
            },
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE, "STRING", "STRING")
    RETURN_NAMES = ("cine_linx", "mixer_json", "report")
    FUNCTION = "mix"
    CATEGORY = "IAMCCS/Cine/Audio"

    def mix(self, mixer_data, sync_policy="mixer_overrides_cine_linx", cine_linx=None):
        linx = _clone_linx(cine_linx)
        resources = _resources(linx)
        outputs = _outputs(linx)
        payload = _payload(linx)

        mixer = _safe_json_loads(mixer_data, {})
        if not isinstance(mixer, dict):
            mixer = _safe_json_loads(self.DEFAULT_DATA, {})

        audio_tracks = resources.get("cine_audio_tracks") if isinstance(resources.get("cine_audio_tracks"), dict) else {}
        source_segments = _segments(audio_tracks.get("all_segments"))
        if not source_segments:
            source_segments = _segments(audio_tracks.get("segments"))
        if not source_segments:
            source_segments = _segments(payload.get("audioSegments"))

        source_settings = _first_list(
            mixer.get("mirrorTrackSettings"),
            mixer.get("trackSettings"),
            audio_tracks.get("trackSettings"),
            audio_tracks.get("track_settings"),
            audio_tracks.get("all_track_settings"),
            payload.get("trackSettings"),
        )
        track_count = max(
            1,
            _safe_int(mixer.get("audioTrackCount"), 0),
            _safe_int(audio_tracks.get("track_count"), 0),
            _safe_int(payload.get("audioTrackCount"), 0),
            max([_safe_int(seg.get("track", 0), 0) + 1 for seg in source_segments] or [1]),
        )
        track_settings = _default_track_settings(track_count, source_settings)
        master_bus = copy.deepcopy(audio_tracks.get("master_bus") if isinstance(audio_tracks.get("master_bus"), dict) else {})
        master_bus.update(copy.deepcopy(mixer.get("masterBus") if isinstance(mixer.get("masterBus"), dict) else {}))
        master_gain = max(0.0, min(2.0, _safe_float(mixer.get("masterAudioGain", payload.get("masterAudioGain", 1.0)), 1.0)))

        manifest = {
            "schema": "iamccs.audio_board_mixer.manifest",
            "schema_version": 1,
            "source": "IAMCCS_AudioBoardMixer",
            "sync_policy": str(sync_policy or "mixer_overrides_cine_linx"),
            "track_count": track_count,
            "masterAudioGain": master_gain,
            "masterBus": master_bus,
            "trackSettings": track_settings,
            "segments": source_segments,
            "truth": "Mixer mirrors AudioBoard track volume/pan/inserts and writes them back into cine_linx metadata for downstream BusOut/save stages.",
        }

        if sync_policy != "read_only_monitor":
            audio_tracks["trackSettings"] = track_settings
            audio_tracks["track_settings"] = copy.deepcopy(track_settings)
            audio_tracks["all_track_settings"] = copy.deepcopy(track_settings)
            audio_tracks["master_bus"] = master_bus
            audio_tracks["mixer_manifest"] = manifest
            resources["cine_audio_tracks"] = audio_tracks
            resources["cine_audio_mixer"] = manifest
            payload["trackSettings"] = track_settings
            payload["masterBus"] = master_bus
            payload["masterAudioGain"] = master_gain
            payload["audioMixer"] = manifest
            outputs["audio_mixer_json"] = json.dumps(manifest, ensure_ascii=False)

        if isinstance(linx.get("chain"), list):
            linx["chain"].append({"node": "IAMCCS_AudioBoardMixer", "track_count": track_count, "sync_policy": sync_policy})
        _refresh_linx_index(linx)

        report = {
            "node": "IAMCCS_AudioBoardMixer",
            "sync_policy": sync_policy,
            "track_count": track_count,
            "segments": len(source_segments),
            "has_master_bus": bool(master_bus),
            "writes_cine_linx": sync_policy != "read_only_monitor",
        }
        return linx, json.dumps(manifest, indent=2, ensure_ascii=False), json.dumps(report, indent=2, ensure_ascii=False)
