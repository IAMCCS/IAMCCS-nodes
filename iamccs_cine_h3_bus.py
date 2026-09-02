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
H3_INPUT_CATEGORY = "IAMCCS/MiniMax H3/Architecture"
H3_CONTROL_PREPROCESSORS = (
    "already_preprocessed",
    "from_iamccs_settings",
    "dwpose",
    "depth_anything",
    "canny",
    "hed",
    "mlsd",
)
H3_CONTROL_AIO_NAMES = {
    "dwpose": "DWPreprocessor",
    "depth_anything": "DepthAnythingPreprocessor",
    "canny": "CannyEdgePreprocessor",
    "hed": "HEDPreprocessor",
    "mlsd": "M-LSDPreprocessor",
}


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


def _meaningful(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (dict, list, tuple, set)):
        return bool(value)
    return True


def _clone_transport_value(value: Any) -> Any:
    """Clone CineLinX structure without duplicating runtime media tensors."""
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        return {key: _clone_transport_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_transport_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_transport_value(item) for item in value)
    if isinstance(value, set):
        return {_clone_transport_value(item) for item in value}
    try:
        return copy.deepcopy(value)
    except Exception:
        # Comfy model/media handles are immutable transport references from the
        # bridge's point of view and may deliberately reject deepcopy.
        return value


def _merge_named_dict(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Merge CineLinX maps while protecting valid values from empty aliases."""
    out = _clone_transport_value(base)
    for raw_key, incoming_value in incoming.items():
        key = str(raw_key)
        existing = out.get(key)
        if isinstance(existing, dict) and isinstance(incoming_value, dict):
            out[key] = _merge_named_dict(existing, incoming_value)
        elif key == "iamccs_prompter_injections" and isinstance(existing, list) and isinstance(incoming_value, list):
            combined = [_clone_transport_value(item) for item in existing]
            seen = {json.dumps(item, sort_keys=True, ensure_ascii=False, default=str) for item in combined}
            for item in incoming_value:
                marker = json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
                if marker not in seen:
                    combined.append(_clone_transport_value(item))
                    seen.add(marker)
            out[key] = combined
        elif _meaningful(incoming_value) or not _meaningful(existing):
            out[key] = _clone_transport_value(incoming_value)
    return out


def _append_unique_dicts(target: list[dict[str, Any]], incoming: Any) -> None:
    seen = {json.dumps(item, sort_keys=True, ensure_ascii=False, default=str) for item in target}
    for item in incoming if isinstance(incoming, list) else []:
        if not isinstance(item, dict):
            continue
        marker = json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
        if marker not in seen:
            target.append(_clone_transport_value(item))
            seen.add(marker)


def merge_cine_h3_inputs(named_inputs: list[tuple[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Combine independent CineLinX producers into one modular H3 envelope."""
    active = [(name, value) for name, value in named_inputs if isinstance(value, dict)]
    if not active:
        raise ValueError("IAMCCS CineH3Input requires at least one connected CineLinX module")

    out: dict[str, Any] = {"type": SUPERNODE_LINX_TYPE}
    chains: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    collision_sources: dict[str, list[str]] = {}
    resource_owner: dict[str, str] = {}

    for input_name, envelope in active:
        _append_unique_dicts(chains, envelope.get("chain"))
        _append_unique_dicts(stages, envelope.get("stages"))
        for map_key in (
            "resources", "outputs", "policies", "slot_map", "contracts",
            "resource_sources", "output_sources",
        ):
            incoming = envelope.get(map_key)
            if not isinstance(incoming, dict):
                continue
            current = out.get(map_key) if isinstance(out.get(map_key), dict) else {}
            if map_key == "resources":
                for resource_key, resource_value in incoming.items():
                    if resource_key in current and _meaningful(resource_value):
                        previous = resource_owner.get(str(resource_key), "earlier_input")
                        collision_sources.setdefault(str(resource_key), [previous]).append(input_name)
                    if _meaningful(resource_value) or not _meaningful(current.get(resource_key)):
                        resource_owner[str(resource_key)] = input_name
            out[map_key] = _merge_named_dict(current, incoming)

    if chains:
        out["chain"] = chains
    if stages:
        out["stages"] = stages
    manifest = {
        "schema": "iamccs.minimax_h3.modular_input_bridge",
        "schema_version": 1,
        "connected_inputs": [name for name, _value in active],
        "resource_owners": resource_owner,
        "collisions": collision_sources,
        "merge_policy": "named deep merge; later meaningful values win; empty values never erase data",
    }
    resources = out.setdefault("resources", {})
    resources["iamccs_cine_h3_input_manifest"] = manifest
    out.setdefault("outputs", {})["iamccs_cine_h3_input_manifest"] = manifest
    out.setdefault("chain", []).append({"role": "cine_h3_input", "name": "IAMCCS_CineH3Input"})
    out["mode"] = "iamccs_cine_h3_modular_input"
    out["active_stage"] = "IAMCCS CineH3Input"
    out["active_stage_kind"] = "cine_h3_modular_merge"
    out["stage_count"] = len(out.get("stages") or [])
    out["resource_keys"] = sorted(resources)
    out["resource_types"] = {key: type(value).__name__ for key, value in resources.items()}
    return out, manifest


class IAMCCS_CineH3Input:
    """Fan-in bridge for modular H3 authoring, settings and audio modules."""

    @classmethod
    def INPUT_TYPES(cls):
        socket = (SUPERNODE_LINX_TYPE,)
        return {
            "optional": {
                "audioboard_arranger": socket,
                "iamccs_prompter": socket,
                "dialogue_tag_editor": socket,
                "iamccs_h3_settings": socket,
                "h3_fun_control_input": socket,
                "cine_h3_vision_info": socket,
                "cine_h3_audio_bus": socket,
                "cine_module_1": socket,
                "cine_module_2": socket,
            }
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE, "STRING")
    RETURN_NAMES = ("cine_linx", "merge_report")
    FUNCTION = "merge"
    CATEGORY = H3_INPUT_CATEGORY
    DESCRIPTION = (
        "Merge AudioBoard Arranger, IAMCCS Prompter, Dialogue Tag Editor, H3 Settings, "
        "H3 Fun Control media, Vision Info and additional CineLinX modules into one Shotboard input."
    )

    def merge(self, **kwargs):
        ordered_names = (
            "cine_module_1", "cine_module_2", "cine_h3_vision_info",
            "dialogue_tag_editor", "iamccs_prompter", "audioboard_arranger",
            "cine_h3_audio_bus", "h3_fun_control_input", "iamccs_h3_settings",
        )
        merged, manifest = merge_cine_h3_inputs([(name, kwargs.get(name)) for name in ordered_names])
        return merged, json.dumps(manifest, ensure_ascii=False, indent=2)


class IAMCCS_CineH3FunControlInput:
    """Attach preprocessed H3 Fun ControlNet media after the modular fan-in.

    Connect this node between IAMCCS CineH3Input (or H3 Settings) and the
    Shotboard.  Runtime tensors are kept by reference so a long pose/depth
    batch is not duplicated merely to travel through CineLinX.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "source_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001}),
                "preprocessor": (list(H3_CONTROL_PREPROCESSORS), {"default": "already_preprocessed"}),
                "preprocess_resolution": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64}),
            },
            "optional": {
                "control_video": ("IMAGE",),
                "mask": ("MASK",),
                "source_video": ("IMAGE",),
            },
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE, "STRING", "IMAGE")
    RETURN_NAMES = ("cine_linx", "report", "control_preview")
    FUNCTION = "inject"
    CATEGORY = H3_INPUT_CATEGORY
    DESCRIPTION = (
        "Transport preprocessed H3 control frames, or preprocess source_video through the installed "
        "ControlNet Aux implementation. The appended preview output exposes the exact frames sent to H3."
    )

    @staticmethod
    def _settings_kind(cine_linx):
        resources = _as_dict(cine_linx.get("resources"))
        outputs = _as_dict(cine_linx.get("outputs"))
        payload = _as_dict(resources.get("cine_payload"))
        for candidate in (
            resources.get("iamccs_minimax_h3_settings"),
            outputs.get("iamccs_minimax_h3_settings"),
            payload.get("iamccs_minimax_h3_settings"),
        ):
            candidate = _as_dict(candidate)
            settings = _as_dict(candidate.get("settings")) or candidate
            if settings.get("h3_controlnet_kind"):
                return str(settings["h3_controlnet_kind"]).strip().lower()
        return "pose_dwpose"

    @classmethod
    def _preprocess(cls, images, choice, resolution):
        import nodes

        node_name = H3_CONTROL_AIO_NAMES.get(choice)
        if not node_name:
            return images
        aio_cls = nodes.NODE_CLASS_MAPPINGS.get("AIO_Preprocessor")
        if aio_cls is None:
            raise RuntimeError(
                "IAMCCS H3 Control preprocessing requires comfyui_controlnet_aux / AIO_Preprocessor. "
                "Install or enable that package, or choose already_preprocessed."
            )
        result = aio_cls().execute(preprocessor=node_name, image=images, resolution=int(resolution))
        if isinstance(result, dict):
            result = result.get("result")
        prepared = result[0] if isinstance(result, (tuple, list)) and result else None
        if not torch.is_tensor(prepared) or prepared.ndim != 4:
            raise RuntimeError(f"ControlNet Aux {node_name} did not return an IMAGE batch")
        return prepared

    def inject(
        self,
        cine_linx,
        source_fps,
        preprocessor="already_preprocessed",
        preprocess_resolution=512,
        control_video=None,
        mask=None,
        source_video=None,
    ):
        if not isinstance(cine_linx, dict):
            raise ValueError("IAMCCS Cine H3 Fun Control Input requires a CineLinX input")
        selected_preprocessor = str(preprocessor or "already_preprocessed").strip().lower()
        if selected_preprocessor == "from_iamccs_settings":
            selected_preprocessor = {
                "pose_dwpose": "dwpose", "depth": "depth_anything", "canny": "canny",
                "hed": "hed", "mlsd": "mlsd", "inpaint": "already_preprocessed",
            }.get(self._settings_kind(cine_linx), "already_preprocessed")
        if selected_preprocessor not in H3_CONTROL_PREPROCESSORS:
            raise ValueError(f"Unknown IAMCCS H3 Control preprocessor: {selected_preprocessor}")
        if control_video is None and torch.is_tensor(source_video):
            control_video = self._preprocess(source_video, selected_preprocessor, preprocess_resolution)
        if control_video is None:
            # Deliberate no-media pass-through: final R39/R40 graphs can keep
            # this modular insertion point wired without evaluating a video
            # preprocessor while ControlNet is OFF. Enabling ControlNet still
            # hard-fails in the backend until real control frames are attached.
            out = dict(cine_linx)
            chains = list(out.get("chain")) if isinstance(out.get("chain"), list) else []
            chains.append({"role": "h3_fun_control_media", "name": "IAMCCS_CineH3FunControlInput", "state": "pass_through"})
            out["chain"] = chains
            out["active_stage"] = "IAMCCS Cine H3 Fun Control Input · pass-through"
            out["active_stage_kind"] = "h3_fun_control_media"
            return out, "H3 Fun Control media | pass-through | attach source_video or preprocessed control_video before enabling ControlNet", None
        if not torch.is_tensor(control_video) or control_video.ndim != 4 or int(control_video.shape[0]) < 1:
            raise ValueError("H3 Fun Control input must be a non-empty IMAGE frame batch")
        if mask is not None and (not torch.is_tensor(mask) or mask.ndim not in {3, 4}):
            raise ValueError("H3 Fun Control inpaint mask must be a MASK frame batch")
        if source_video is not None and (not torch.is_tensor(source_video) or source_video.ndim != 4):
            raise ValueError("H3 Fun Control source_video must be an IMAGE frame batch")

        # Shallow structural copies preserve tensor storage. The prior modular
        # bridge has already resolved metadata collisions before this injector.
        out = dict(cine_linx)
        resources = dict(_as_dict(out.get("resources")))
        outputs = dict(_as_dict(out.get("outputs")))
        chains = list(out.get("chain")) if isinstance(out.get("chain"), list) else []
        meta = {
            "schema": "iamccs.minimax_h3.fun_control_media",
            "schema_version": 1,
            "source_fps": float(source_fps),
            "frame_count": int(control_video.shape[0]),
            "height": int(control_video.shape[1]),
            "width": int(control_video.shape[2]),
            "has_mask": torch.is_tensor(mask),
            "has_source_video": torch.is_tensor(source_video),
            "preprocessor": selected_preprocessor,
            "preprocess_resolution": int(preprocess_resolution),
            "contract": "preview is the exact control IMAGE batch; Settings owns kind/model/strength/window",
        }
        resources["iamccs_minimax_h3_control_video"] = control_video
        resources["iamccs_minimax_h3_control_video_meta"] = meta
        if torch.is_tensor(mask):
            resources["iamccs_minimax_h3_control_mask"] = mask
        if torch.is_tensor(source_video):
            resources["iamccs_minimax_h3_control_source_video"] = source_video
        outputs["iamccs_minimax_h3_control_video_meta"] = meta
        chains.append({"role": "h3_fun_control_media", "name": "IAMCCS_CineH3FunControlInput"})
        out["resources"] = resources
        out["outputs"] = outputs
        out["chain"] = chains
        out["active_stage"] = "IAMCCS Cine H3 Fun Control Input"
        out["active_stage_kind"] = "h3_fun_control_media"
        out["resource_keys"] = sorted(resources)
        out["resource_types"] = {key: type(value).__name__ for key, value in resources.items()}
        report = (
            f"H3 Fun Control media | {meta['frame_count']}f @ {meta['source_fps']:.3f}fps | "
            f"{meta['width']}x{meta['height']} | mask={'yes' if meta['has_mask'] else 'no'} | "
            "connect output to the Shotboard historical cine_linx input"
        )
        report += f" | preprocessor={selected_preprocessor}"
        # Show a bounded representative contact preview directly on this node.
        # The full control batch still travels through CineLinX and the IMAGE
        # output; only the temporary UI preview is sampled to avoid hundreds of
        # PNG writes for long videos.
        try:
            import nodes

            count = int(control_video.shape[0])
            preview_indices = torch.linspace(0, count - 1, steps=min(4, count)).round().long()
            preview_batch = control_video.index_select(0, preview_indices.to(control_video.device))
            ui = nodes.PreviewImage().save_images(
                preview_batch,
                filename_prefix="IAMCCS_H3_ControlPreview",
                prompt=None,
                extra_pnginfo=None,
            ).get("ui", {})
        except Exception:
            ui = {"text": [report]}
        ui.setdefault("text", []).append(report)
        return {"ui": ui, "result": (out, report, control_video)}


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


def _shotplan_audio_mode(cine_linx: Any) -> str:
    """Resolve the authored H3 audio policy without importing backend nodes."""
    envelope = _as_dict(cine_linx)
    if envelope.get("schema") == "iamccs.minimax_h3.shotplan":
        return str(envelope.get("audio_mode", "h3_native_generated") or "h3_native_generated").lower()
    resources = _as_dict(envelope.get("resources"))
    outputs = _as_dict(envelope.get("outputs"))
    payload = _as_dict(resources.get("cine_payload"))
    for candidate in (
        resources.get("iamccs_minimax_h3_shotplan"),
        resources.get("minimax_h3_shotplan"),
        resources.get("shotplan"),
        outputs.get("shotplan"),
        payload.get("minimax_h3_shotplan"),
        payload.get("shotplan"),
    ):
        if isinstance(candidate, dict) and candidate.get("schema") == "iamccs.minimax_h3.shotplan":
            return str(candidate.get("audio_mode", "h3_native_generated") or "h3_native_generated").lower()
    return "h3_native_generated"


def _silent_transport_audio(sample_rate: int = 32000) -> dict[str, Any]:
    """One-sample typed placeholder; native H3 audio never consumes it."""
    return {
        "waveform": torch.zeros((1, 1, 1), dtype=torch.float32),
        "sample_rate": int(sample_rate),
        "iamccs_transport_only": True,
    }


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
        audio_mode = _shotplan_audio_mode(cine_linx)
        native_audio_bypass = not segments and audio_mode == "h3_native_generated"
        if not segments and not native_audio_bypass:
            raise ValueError(
                f"Cine H3 Audio Bus found no audioSegments while audio_mode={audio_mode}. "
                "Insert an AudioBoard clip or select h3_native_generated."
            )

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

        if native_audio_bypass:
            # ComfyUI still evaluates the permanently wired AUDIO sockets in
            # the unified workflow. Supply a typed transport placeholder;
            # AudioTimelineMix and AtomicAudioDrive both recognize the native
            # mode and never encode or condition from this value.
            lanes = [_silent_transport_audio() for _ in range(MAX_AUDIO_LANES)]

        manifest = {
            "schema": "iamccs.minimax_h3.audio_bus",
            "schema_version": 2,
            "source": "shotboard_audioSegments",
            "transport": "clip_addressed_cine_linx",
            "native_audio_bypass": native_audio_bypass,
            "audio_mode": audio_mode,
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
    "IAMCCS_CineH3Input": IAMCCS_CineH3Input,
    "IAMCCS_CineH3FunControlInput": IAMCCS_CineH3FunControlInput,
    "IAMCCS_CineH3AudioBus": IAMCCS_CineH3AudioBus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_CineH3Input": "IAMCCS CineH3Input · Modular Bridge",
    "IAMCCS_CineH3FunControlInput": "IAMCCS Cine H3 Fun Control Input · Pose / Depth / Edge",
    "IAMCCS_CineH3AudioBus": "Cine H3 Audio Bus (Shotboard Lanes)",
}
