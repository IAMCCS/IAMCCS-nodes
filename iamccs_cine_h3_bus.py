# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later

"""Expose MiniMax H3 Shotboard audio lanes as native ComfyUI AUDIO outputs."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import folder_paths
import torch


SUPERNODE_LINX_TYPE = "IAMCCS_SUPERNODE_LINX"
CATEGORY = "IAMCCS/MiniMax H3/Audio"
MAX_AUDIO_LANES = 8


# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
def _as_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value or ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
def _audio_timeline(cine_linx: Any) -> dict[str, Any]:
    envelope = _as_dict(cine_linx)
    resources = _as_dict(envelope.get("resources"))
    outputs = _as_dict(envelope.get("outputs"))
    payload = _as_dict(resources.get("cine_payload"))
    for candidate in (
        resources.get("cine_audio_timeline_json"),
        resources.get("cine_audio_bus_timeline_json"),
        payload.get("timeline_data"),
        outputs.get("audio_timeline_json"),
        outputs.get("timeline_data"),
    ):
        timeline = _as_json_dict(candidate)
        if isinstance(timeline.get("audioSegments"), list):
            return copy.deepcopy(timeline)
    return {}


# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
def _source_index(segment: dict[str, Any], fallback: int) -> int:
    for key in ("audio_input", "audioInput", "source_audio_index", "sourceAudioIndex", "input_index"):
        raw = segment.get(key)
        if raw is None or not str(raw).strip():
            continue
        try:
            value = int(round(float(raw)))
        except (TypeError, ValueError, OverflowError):
            continue
        return max(0, value - 1 if value > 0 else value)
    return fallback


# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
def _canonicalize_source_sockets(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign one deterministic internal AUDIO socket to every real clip.

    AudioBoard lanes are editorial tracks, not Comfy graph sockets: several
    clips may live on one track.  Rebuild the socket mapping from the current
    clip order so stale ``audio_input`` values can never point at a vanished
    manual connection.
    """
    canonical: list[dict[str, Any]] = []
    next_socket = 1
    for original in segments:
        segment = copy.deepcopy(original)
        if bool(segment.get("placeholder", False)):
            canonical.append(segment)
            continue
        # These aliases are deliberately overwritten. They are a runtime
        # transport address, never a user-editable timeline setting.
        segment["audio_input"] = next_socket
        segment["audioInput"] = next_socket
        segment["source_audio_index"] = next_socket
        segment["sourceAudioIndex"] = next_socket
        canonical.append(segment)
        next_socket += 1
    return canonical


def _input_audio_path(segment: dict[str, Any]) -> tuple[str, Path]:
    filename = str(
        segment.get("audioFile")
        or segment.get("audio_file")
        or segment.get("fileName")
        or segment.get("filename")
        or ""
    ).strip()
    if not filename:
        raise ValueError(f"Audio lane {segment.get('id') or '?'} has no saved input filename")
    root = Path(folder_paths.get_input_directory()).resolve()
    candidate = (root / filename).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Audio lane filename escapes ComfyUI/input: {filename}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"Shotboard audio file not found in ComfyUI/input: {filename}")
    return filename, candidate


def _load_native_audio(path: Path) -> dict[str, Any]:
    """Use the same decoder and AUDIO shape as ComfyUI's Load Audio node."""
    from comfy_extras.nodes_audio import load as load_audio

    waveform, sample_rate = load_audio(str(path))
    return {"waveform": waveform.unsqueeze(0), "sample_rate": int(sample_rate)}


def _lane_manifest(segment: dict[str, Any], lane_index: int, filename: str, audio: dict[str, Any]) -> dict[str, Any]:
    waveform = audio["waveform"]
    return {
        "lane": lane_index + 1,
        "id": str(segment.get("id") or f"audio_lane_{lane_index + 1}"),
        "audioFile": filename,
        "audio_input": lane_index + 1,
        "linkedVisualId": str(segment.get("linkedVisualId") or segment.get("linked_visual_id") or ""),
        "start_seconds": float(segment.get("start_seconds", 0.0) or 0.0),
        "duration_seconds": float(segment.get("duration_seconds", 0.0) or 0.0),
        "trimStart": int(segment.get("trimStart", segment.get("trim_start", 0)) or 0),
        "track": int(segment.get("track", 0) or 0),
        "gain": float(segment.get("gain", segment.get("volume", 1.0)) or 0.0),
        "pan": float(segment.get("pan", 0.0) or 0.0),
        "fadeInFrames": int(segment.get("fadeInFrames", 0) or 0),
        "fadeOutFrames": int(segment.get("fadeOutFrames", 0) or 0),
        "mute": bool(segment.get("mute", False)),
        "solo": bool(segment.get("solo", False)),
        "sample_rate": int(audio["sample_rate"]),
        "waveform_shape": [int(value) for value in waveform.shape],
    }


class IAMCCS_CineH3AudioBus:
    """Rebuild Shotboard timeline source files as loader-equivalent AUDIO lanes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"cine_linx": (SUPERNODE_LINX_TYPE,)}}

    RETURN_TYPES = (SUPERNODE_LINX_TYPE,) + ("AUDIO",) * MAX_AUDIO_LANES + ("STRING",)
    RETURN_NAMES = ("cine_linx",) + tuple(f"audio_{index}" for index in range(1, MAX_AUDIO_LANES + 1)) + ("audio_metadata_json",)
    FUNCTION = "publish"
    CATEGORY = CATEGORY

    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    def publish(self, cine_linx):
        timeline = _audio_timeline(cine_linx)
        segments = [item for item in timeline.get("audioSegments", []) if isinstance(item, dict)]
        if not segments:
            raise ValueError("Cine H3 Audio Bus found no audioSegments in cine_linx")

        # A visual AudioBoard track can contain many independent audio clips.
        # The transport below is clip-addressed (one internal socket per clip),
        # while ``track`` remains untouched for the mix / editorial UI.
        canonical_segments = _canonicalize_source_sockets(segments)
        canonical_timeline = copy.deepcopy(timeline)
        canonical_timeline["audioSegments"] = canonical_segments

        # ``lanes`` keeps the original eight graph outputs for existing
        # workflows. ``resource_lanes`` is the native CineLinX transport and
        # intentionally has no clip-count cap.
        lanes: list[dict[str, Any] | None] = [None] * MAX_AUDIO_LANES
        resource_lanes: list[dict[str, Any]] = []
        manifest_lanes: list[dict[str, Any]] = []
        for segment in canonical_segments:
            if bool(segment.get("placeholder", False)):
                continue
            lane_index = int(segment["audio_input"]) - 1
            filename, path = _input_audio_path(segment)
            audio = _load_native_audio(path)
            resource_lanes.append(audio)
            if lane_index < MAX_AUDIO_LANES:
                lanes[lane_index] = audio
            manifest_lanes.append(_lane_manifest(segment, lane_index, filename, audio))

        manifest = {
            "schema": "iamccs.minimax_h3.audio_bus",
            "schema_version": 2,
            "source": "shotboard_audioSegments",
            "transport": "clip_addressed_cine_linx",
            "lanes": manifest_lanes,
            "timeline": canonical_timeline,
        }
        out_linx = copy.deepcopy(_as_dict(cine_linx))
        resources = out_linx.setdefault("resources", {})
        outputs = out_linx.setdefault("outputs", {})
        timeline_json = json.dumps(canonical_timeline, ensure_ascii=False)
        tracks = copy.deepcopy(_as_dict(resources.get("cine_audio_tracks")))
        tracks["shotboard_segments"] = copy.deepcopy(canonical_segments)
        tracks["segments"] = copy.deepcopy(canonical_segments)
        resources["cine_audio_tracks"] = tracks
        resources["cine_audio_timeline_json"] = timeline_json
        resources["cine_audio_bus_timeline_json"] = timeline_json
        # Native in-band transport: the mix node can now recover every clip
        # from cine_linx even when a workflow has no manual audio_4..audio_8
        # wires. Existing socket wires still override these values.
        resources["iamccs_cine_h3_audio_bus_audio_lanes"] = resource_lanes
        resources["iamccs_cine_h3_audio_bus_manifest"] = manifest
        resources["iamccs_cine_h3_audio_bus_metadata_json"] = json.dumps(manifest, ensure_ascii=False)
        outputs["audio_timeline_json"] = timeline_json
        outputs["cine_h3_audio_bus_metadata_json"] = json.dumps(manifest, ensure_ascii=False)
        out_linx.setdefault("chain", []).append({"role": "cine_h3_audio_bus", "name": "IAMCCS_CineH3AudioBus"})
        out_linx["resource_keys"] = sorted(resources.keys())
        out_linx["resource_types"] = {key: type(value).__name__ for key, value in resources.items()}
        return (out_linx, *lanes, json.dumps(manifest, ensure_ascii=False, indent=2))


NODE_CLASS_MAPPINGS = {
    "IAMCCS_CineH3AudioBus": IAMCCS_CineH3AudioBus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_CineH3AudioBus": "Cine H3 Audio Bus (Shotboard Lanes)",
}