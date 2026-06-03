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


def _clone_linx(cine_linx: Any) -> Dict[str, Any]:
    return copy.deepcopy(cine_linx) if isinstance(cine_linx, dict) else {
        "type": SUPERNODE_LINX_TYPE,
        "mode": "iamccs_control_aud_efx",
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


def _effect_graph_from_segments(segments: List[Dict[str, Any]], master_bus: Dict[str, Any]) -> Dict[str, Any]:
    clips = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        clips.append({
            "id": str(seg.get("id", "")),
            "name": str(seg.get("name") or seg.get("fileName") or "audio"),
            "track": int(_safe_float(seg.get("track", 0), 0)),
            "start": int(round(_safe_float(seg.get("start", 0), 0))),
            "length": int(round(_safe_float(seg.get("length", 1), 1))),
            "trimStart": int(round(_safe_float(seg.get("trimStart", 0), 0))),
            "audioFile": str(seg.get("audioFile", "")),
            "audioUploadType": str(seg.get("audioUploadType", "input") or "input"),
            "linkedVisualId": str(seg.get("linkedVisualId", "")),
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
        })
    return {
        "schema": "iamccs.audio_effect_graph",
        "schema_version": 1,
        "clips": clips,
        "masterBus": master_bus if isinstance(master_bus, dict) else {},
    }


class IAMCCS_ControlAudEfx:
    """Realtime control/inspection companion for AudioBoardArranger effect graphs."""

    DEFAULT_DATA = json.dumps({
        "schema": "iamccs.control_aud_efx",
        "schema_version": 1,
        "selectedClipId": "",
        "view": "eq_wave_spectrum",
        "metering": {"fftSize": 2048, "smoothing": 0.72},
    }, indent=2)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_data": ("STRING", {
                    "default": cls.DEFAULT_DATA,
                    "multiline": True,
                    "tooltip": "Edited by the IAMCCS ControlAudEfx UI. Connect cine_linx from AudioBoardArranger.",
                }),
            },
            "optional": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "cine_linx_from_arranger": (SUPERNODE_LINX_TYPE,),
            },
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("cine_linx", "effect_graph_json", "selected_clip_json", "report")
    FUNCTION = "control"
    CATEGORY = "IAMCCS/Cine/Audio"

    def control(self, control_data, cine_linx=None, cine_linx_from_arranger=None):
        data = _safe_json_loads(control_data, {})
        if not isinstance(data, dict):
            data = _safe_json_loads(self.DEFAULT_DATA, {})
        out_linx = _clone_linx(cine_linx_from_arranger if isinstance(cine_linx_from_arranger, dict) else cine_linx)
        resources = _resources(out_linx)
        outputs = _outputs(out_linx)
        payload = _payload(out_linx)
        audio_tracks = resources.get("cine_audio_tracks") if isinstance(resources.get("cine_audio_tracks"), dict) else {}
        arranger = audio_tracks.get("segments") if isinstance(audio_tracks.get("segments"), list) else payload.get("audioSegments")
        segments = arranger if isinstance(arranger, list) else []
        master_bus = audio_tracks.get("master_bus") if isinstance(audio_tracks.get("master_bus"), dict) else payload.get("masterBus", {})
        effect_graph = _effect_graph_from_segments(segments, master_bus if isinstance(master_bus, dict) else {})
        selected_id = str(data.get("selectedClipId") or payload.get("selectedClipId") or "")
        selected = next((clip for clip in effect_graph["clips"] if clip.get("id") == selected_id), effect_graph["clips"][0] if effect_graph["clips"] else {})
        effect_graph_json = json.dumps(effect_graph, ensure_ascii=False, indent=2)
        selected_clip_json = json.dumps(selected, ensure_ascii=False, indent=2)
        resources["cine_audio_effect_graph_json"] = effect_graph_json
        resources["cine_audio_control_efx"] = data
        outputs["audio_effect_graph_json"] = effect_graph_json
        outputs["selected_audio_clip_json"] = selected_clip_json
        payload["audio_effect_graph"] = effect_graph
        out_linx["type"] = SUPERNODE_LINX_TYPE
        out_linx["mode"] = "iamccs_control_aud_efx"
        out_linx.setdefault("chain", []).append({"role": "audio_effect_control", "name": "IAMCCS_ControlAudEfx"})
        _refresh_linx_index(out_linx)
        report = json.dumps({
            "node": "IAMCCS_ControlAudEfx",
            "clips": len(effect_graph["clips"]),
            "selected_clip": selected.get("id", ""),
            "truth": "Visualizes and exports the AudioBoardArranger DSP/effect graph without changing Shotboard timing.",
        }, ensure_ascii=False, indent=2)
        outputs["report"] = report
        resources["cine_report"] = report
        return out_linx, effect_graph_json, selected_clip_json, report


NODE_CLASS_MAPPINGS = {
    "IAMCCS_ControlAudEfx": IAMCCS_ControlAudEfx,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_ControlAudEfx": "IAMCCS ControlAudEfx",
}
