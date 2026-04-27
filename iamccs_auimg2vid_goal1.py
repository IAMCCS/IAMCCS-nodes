from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional

import torch

try:
    from .iamccs_audio_extender import (
        IAMCCS_AudioExtensionMath,
        IAMCCS_AudioExtender,
        IAMCCS_AudioTimelineGate,
    )
    from .iamccs_ltx2_extension_module import (
        IAMCCS_LTX2_ExtensionModule,
        IAMCCS_LTX2_ExtensionModule_Disk,
        IAMCCS_LTX2_FirstLastLatentControl_Pro,
        IAMCCS_VideoCombineFromDir,
    )
    from .iamccs_ltx2_tools import IAMCCS_SegmentPlanner, IAMCCS_SourceRangeFromSegmentPlan
except ImportError:
    from iamccs_audio_extender import (  # type: ignore
        IAMCCS_AudioExtensionMath,
        IAMCCS_AudioExtender,
        IAMCCS_AudioTimelineGate,
    )
    from iamccs_ltx2_extension_module import (  # type: ignore
        IAMCCS_LTX2_ExtensionModule,
        IAMCCS_LTX2_ExtensionModule_Disk,
        IAMCCS_LTX2_FirstLastLatentControl_Pro,
        IAMCCS_VideoCombineFromDir,
    )
    from iamccs_ltx2_tools import IAMCCS_SegmentPlanner, IAMCCS_SourceRangeFromSegmentPlan  # type: ignore


SUPERNODE_CONTRACT_TYPE = "IAMCCS_SUPERNODE_CONTRACT"
SUPERNODE_LINX_TYPE = "IAMCCS_SUPERNODE_LINX"
PIPELINE_KIND = "au_img2vid"
HELPER_CATEGORY = "IAMCCS/Ltx2 Helper Modules"
HELPER_LABEL_PREFIX = "Ltx2HelperModules"


def _resolve_goal1_preset(preset: str) -> Dict[str, Any]:
    preset_key = str(preset or "custom").strip().lower()
    presets: Dict[str, Dict[str, Any]] = {
        "custom": {},
        "lowram_safe_videoclip": {
            "segment_seconds": 10.0,
            "planning_mode": "manual_segment_seconds",
            "content_profile": "videoclip",
            "overlap_frames": 9,
            "backend_binding": "ltx_audio_guided_lowram",
            "surface_profile": "compact",
        },
        "lowram_fast_videoclip": {
            "segment_seconds": 8.0,
            "planning_mode": "manual_segment_seconds",
            "content_profile": "videoclip",
            "overlap_frames": 9,
            "backend_binding": "ltx_audio_guided_lowram",
            "surface_profile": "compact",
        },
        "normal_quality_videoclip": {
            "segment_seconds": 20.0,
            "planning_mode": "manual_segment_seconds",
            "content_profile": "videoclip",
            "overlap_frames": 9,
            "backend_binding": "ltx_audio_guided_normal",
            "surface_profile": "progressive",
        },
        "lowram_monologue": {
            "segment_seconds": 12.0,
            "planning_mode": "manual_segment_seconds",
            "content_profile": "monologue",
            "overlap_frames": 9,
            "backend_binding": "ltx_audio_guided_lowram",
            "surface_profile": "compact",
        },
    }
    return dict(presets.get(preset_key, presets["custom"]))


def _resolve_goal1_runtime_profile(goal1_preset: str, backend_binding: str, runtime_profile: str) -> str:
    preset_key = str(goal1_preset or "custom").strip().lower()
    if preset_key.startswith("lowram_"):
        return "disk_low_ram"
    if preset_key.startswith("normal_"):
        return "native_canvas"
    binding = str(backend_binding or "").strip().lower()
    if "lowram" in binding:
        return "disk_low_ram"
    if binding:
        return "native_canvas"
    return str(runtime_profile)


def _resolve_goal1_filename_prefix(goal1_preset: str, filename_prefix: str) -> str:
    preset_key = str(goal1_preset or "custom").strip().lower()
    base_prefix = str(filename_prefix or "IAMCCS/Ltx2HelperModules")
    if preset_key == "custom":
        return base_prefix
    if base_prefix.startswith("IAMCCS/Ltx2HelperModules"):
        slug = preset_key.replace("-", "_").replace(" ", "_").upper()
        return f"IAMCCS/Ltx2HelperModules/{slug}"
    return base_prefix


def _resolve_goal1_refresh_defaults(goal1_preset: str) -> Dict[str, Any]:
    preset_key = str(goal1_preset or "custom").strip().lower()
    defaults: Dict[str, Dict[str, Any]] = {
        "custom": {
            "refresh_interval_segments": 0,
            "reanchor_every_segments": 0,
            "protected_head_frames": 9,
            "settling_frames": 9,
            "continuity_preset": "videoclip_audio_24fps",
            "reanchor_preset": "micro_crossfade_3",
            "reanchor_overlap_mode": "filmic_crossfade",
            "reanchor_safe_mode": "native_workflow_safe",
            "reanchor_start_frames_rule": "ltx2_nearest",
        },
        "lowram_safe_videoclip": {
            "refresh_interval_segments": 2,
            "reanchor_every_segments": 3,
            "protected_head_frames": 9,
            "settling_frames": 9,
            "first_anchor_strength": 0.35,
            "last_anchor_strength": 0.20,
            "first_lock_slots": 1,
            "last_lock_slots": 1,
            "end_transition_slots": 1,
            "continuity_preset": "videoclip_audio_24fps",
            "reanchor_preset": "micro_crossfade_3",
            "reanchor_overlap_mode": "filmic_crossfade",
            "reanchor_safe_mode": "native_workflow_safe",
            "reanchor_start_frames_rule": "ltx2_nearest",
        },
        "lowram_fast_videoclip": {
            "refresh_interval_segments": 3,
            "reanchor_every_segments": 4,
            "protected_head_frames": 9,
            "settling_frames": 8,
            "first_anchor_strength": 0.25,
            "last_anchor_strength": 0.10,
            "first_lock_slots": 1,
            "last_lock_slots": 1,
            "end_transition_slots": 0,
            "continuity_preset": "videoclip_audio_24fps",
            "reanchor_preset": "cut_bestofk_16",
            "reanchor_overlap_mode": "linear_blend",
            "reanchor_safe_mode": "native_workflow_safe",
            "reanchor_start_frames_rule": "ltx2_round_down",
        },
        "normal_quality_videoclip": {
            "refresh_interval_segments": 2,
            "reanchor_every_segments": 2,
            "protected_head_frames": 9,
            "settling_frames": 10,
            "first_anchor_strength": 0.45,
            "last_anchor_strength": 0.25,
            "first_lock_slots": 1,
            "last_lock_slots": 2,
            "end_transition_slots": 1,
            "continuity_preset": "videoclip_audio_24fps",
            "reanchor_preset": "cut_bestofk_16_luma",
            "reanchor_overlap_mode": "perceptual_crossfade",
            "reanchor_safe_mode": "native_workflow_safe",
            "reanchor_start_frames_rule": "ltx2_nearest",
        },
        "lowram_monologue": {
            "refresh_interval_segments": 3,
            "reanchor_every_segments": 4,
            "protected_head_frames": 9,
            "settling_frames": 8,
            "first_anchor_strength": 0.30,
            "last_anchor_strength": 0.15,
            "first_lock_slots": 1,
            "last_lock_slots": 1,
            "end_transition_slots": 1,
            "continuity_preset": "monologue_audio_24fps",
            "reanchor_preset": "micro_crossfade_3",
            "reanchor_overlap_mode": "filmic_crossfade",
            "reanchor_safe_mode": "native_workflow_safe",
            "reanchor_start_frames_rule": "ltx2_nearest",
        },
    }
    return dict(defaults.get(preset_key, defaults["custom"]))


def _parse_payload(payload: Any) -> Dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(payload, str):
        value = payload.strip()
        if not value:
            return {}
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {"raw": str(payload)}


def _json_payload(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=True, sort_keys=True)


def _parse_contract(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return dict(payload)
    if payload is None:
        return {}
    if isinstance(payload, str):
        stripped = payload.strip()
        if not stripped:
            return {}
        try:
            loaded = json.loads(stripped)
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            pass
    parsed: Dict[str, Any] = {}
    for item in str(payload).split(";"):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _build_contract(
    *,
    node_role: str,
    node_label: str,
    parent_contract: Any,
    payload_key: str,
    payload_value: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    parent = _parse_contract(parent_contract)
    contract = {
        "type": SUPERNODE_CONTRACT_TYPE,
        "pipeline_key": parent.get("pipeline_key", f"{PIPELINE_KIND}:{node_role}"),
        "pipeline_kind": PIPELINE_KIND,
        "role": node_role,
        "label": node_label,
        payload_key: payload_value,
        "parent": parent,
    }
    if extra:
        contract.update(extra)
    return contract


def _build_linx(
    *,
    existing_linx: Any,
    contract: Dict[str, Any],
    unique_id: Any,
    node_role: str,
    node_label: str,
    module_keys: Optional[list[str]] = None,
) -> Dict[str, Any]:
    if isinstance(existing_linx, dict):
        linx_payload = dict(existing_linx)
        nodes = list(existing_linx.get("nodes") or [])
    else:
        linx_payload = {}
        nodes = []
    linx_payload["type"] = SUPERNODE_LINX_TYPE
    linx_payload["pipeline_kind"] = PIPELINE_KIND
    linx_payload["root_contract"] = contract.get("pipeline_key", PIPELINE_KIND)
    linx_payload["nodes"] = nodes
    linx_payload["nodes"].append(
        {
            "id": str(unique_id or node_label or node_role),
            "role": node_role,
            "label": node_label,
            "module_keys": list(module_keys or []),
        }
    )
    return linx_payload


def _audio_duration_seconds(audio: Any) -> float:
    if not isinstance(audio, Mapping):
        raise ValueError("audio input is required when duration_source uses audio")
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate", 0) or 0)
    if waveform is None or sample_rate <= 0:
        raise ValueError("audio payload missing waveform/sample_rate")
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.tensor(waveform)
    return float(int(waveform.shape[-1])) / float(sample_rate)


def _parse_numeric_list(raw: Any) -> list[float]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        values = raw
    else:
        text = str(raw).strip()
        if not text:
            return []
        try:
            loaded = json.loads(text)
            if isinstance(loaded, list):
                values = loaded
            else:
                values = [part.strip() for part in text.split(",") if part.strip()]
        except Exception:
            values = [part.strip() for part in text.split(",") if part.strip()]
    parsed: list[float] = []
    for value in values:
        try:
            parsed.append(float(value))
        except Exception:
            continue
    return parsed


def _build_project_timeline_segments(
    *,
    total_frames: int,
    segment_count: int,
    unique_segment_frames: int,
    first_segment_raw_frames: int,
    continuation_raw_frames: int,
    last_segment_unique_frames: int,
    fps: float,
) -> list[Dict[str, Any]]:
    segments: list[Dict[str, Any]] = []
    for segment_index in range(max(1, int(segment_count))):
        start_frame = int(unique_segment_frames) * int(segment_index)
        is_last_segment = segment_index >= max(0, int(segment_count) - 1)
        unique_frames = int(last_segment_unique_frames) if is_last_segment else int(unique_segment_frames)
        end_frame = min(int(total_frames), start_frame + max(0, unique_frames))
        segment_role = "initial" if segment_index == 0 else ("final" if is_last_segment else "continuation")
        raw_frames = int(first_segment_raw_frames) if segment_index == 0 else int(continuation_raw_frames)
        segments.append(
            {
                "segment_index": int(segment_index),
                "segment_role": segment_role,
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "unique_frames": int(max(0, end_frame - start_frame)),
                "raw_frames": int(raw_frames),
                "start_seconds": float(start_frame) / float(max(fps, 0.001)),
                "end_seconds": float(end_frame) / float(max(fps, 0.001)),
            }
        )
    return segments


# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
class IAMCCS_ProjectTimelinePlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "goal1_preset": (["custom", "lowram_safe_videoclip", "lowram_fast_videoclip", "normal_quality_videoclip", "lowram_monologue"], {"default": "custom"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.001, "max": 240.0, "step": 0.01}),
                "segment_seconds": ("FLOAT", {"default": 10.0, "min": 0.01, "max": 3600.0, "step": 0.01}),
                "target_duration_seconds": ("FLOAT", {"default": 30.0, "min": 0.01, "max": 36000.0, "step": 0.01}),
                "duration_source": (["target_duration_seconds", "audio", "max_audio_target"], {"default": "target_duration_seconds"}),
                "planning_mode": (["manual_segment_seconds", "auto_profile"], {"default": "auto_profile"}),
                "content_profile": (["videoclip", "monologue"], {"default": "videoclip"}),
                "overlap_frames": ("INT", {"default": 9, "min": 0, "max": 4096, "step": 1}),
                "ltx_round_mode": (["up", "nearest", "down"], {"default": "up"}),
                "surface_profile": (["compact", "progressive", "debug_surface"], {"default": "compact"}),
                "backend_binding": (["ltx_audio_guided_lowram", "ltx_audio_guided_normal", "custom"], {"default": "ltx_audio_guided_lowram"}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "upstream_contract": (SUPERNODE_CONTRACT_TYPE,),
                "linx": (SUPERNODE_LINX_TYPE,),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", SUPERNODE_CONTRACT_TYPE, SUPERNODE_LINX_TYPE, "INT", "INT", "FLOAT", "FLOAT", "INT", "STRING", "STRING")
    RETURN_NAMES = (
        "project_timeline_payload",
        "contract",
        "linx",
        "total_frames",
        "segment_count",
        "total_seconds",
        "segment_seconds_out",
        "unique_segment_frames",
        "segment_ranges_json",
        "report",
    )
    FUNCTION = "build"
    CATEGORY = HELPER_CATEGORY

    def build(
        self,
        goal1_preset,
        fps,
        segment_seconds,
        target_duration_seconds,
        duration_source,
        planning_mode,
        content_profile,
        overlap_frames,
        ltx_round_mode,
        surface_profile,
        backend_binding,
        audio=None,
        upstream_contract=None,
        linx=None,
        unique_id=None,
    ):
        preset_values = _resolve_goal1_preset(goal1_preset)
        segment_seconds = float(preset_values.get("segment_seconds", segment_seconds))
        planning_mode = str(preset_values.get("planning_mode", planning_mode))
        content_profile = str(preset_values.get("content_profile", content_profile))
        overlap_frames = int(preset_values.get("overlap_frames", overlap_frames))
        backend_binding = str(preset_values.get("backend_binding", backend_binding))
        surface_profile = str(preset_values.get("surface_profile", surface_profile))

        effective_duration = float(target_duration_seconds)
        if duration_source == "audio":
            effective_duration = _audio_duration_seconds(audio)
        elif duration_source == "max_audio_target" and audio is not None:
            effective_duration = max(float(target_duration_seconds), _audio_duration_seconds(audio))

        planner_result = IAMCCS_SegmentPlanner().plan(
            effective_duration,
            fps,
            segment_seconds,
            planning_mode,
            content_profile,
            overlap_frames,
            ltx_round_mode,
            0,
        )

        effective_fps = float(planner_result[17])
        total_frames = int(planner_result[0])
        segment_count = int(planner_result[4])
        unique_segment_frames = int(planner_result[1])
        first_segment_raw_frames = int(planner_result[2])
        continuation_raw_frames = int(planner_result[3])
        last_segment_unique_frames = int(planner_result[6])
        segments = _build_project_timeline_segments(
            total_frames=total_frames,
            segment_count=segment_count,
            unique_segment_frames=unique_segment_frames,
            first_segment_raw_frames=first_segment_raw_frames,
            continuation_raw_frames=continuation_raw_frames,
            last_segment_unique_frames=last_segment_unique_frames,
            fps=effective_fps,
        )
        meter_markers_frames = [int(segment["start_frame"]) for segment in segments]
        meter_markers_frames.append(int(total_frames))

        project_timeline_payload_dict = {
            "timeline_family": "project_timeline",
            "pipeline_kind": PIPELINE_KIND,
            "goal1_preset": str(goal1_preset),
            "target_duration_seconds": float(effective_duration),
            "duration_source": str(duration_source),
            "surface_profile": str(surface_profile),
            "backend_binding": str(backend_binding),
            "fps": float(effective_fps),
            "total_seconds": float(effective_duration),
            "total_frames": int(total_frames),
            "segment_seconds": float(segment_seconds),
            "segment_count": int(segment_count),
            "unique_segment_frames": int(unique_segment_frames),
            "first_segment_raw_frames": int(first_segment_raw_frames),
            "continuation_raw_frames": int(continuation_raw_frames),
            "last_segment_unique_frames": int(last_segment_unique_frames),
            "recommended_overlap_frames": int(planner_result[18]),
            "recommended_audio_left_context_s": float(planner_result[19]),
            "recommended_extension_preset": str(planner_result[20]),
            "effective_planning_mode": str(planner_result[21]),
            "segments": segments,
            "meter_markers_frames": meter_markers_frames,
        }
        project_timeline_payload = _json_payload(project_timeline_payload_dict)
        segment_ranges_json = _json_payload({"segments": segments})
        contract = _build_contract(
            node_role="project_timeline",
            node_label="ProjectTimelinePlanner",
            parent_contract=upstream_contract,
            payload_key="project_timeline_payload",
            payload_value=project_timeline_payload,
            extra={"surface_profile": surface_profile, "backend_binding": backend_binding},
        )
        linx_payload = _build_linx(
            existing_linx=linx,
            contract=contract,
            unique_id=unique_id,
            node_role="project_timeline",
            node_label="ProjectTimelinePlanner",
            module_keys=["timeline_family", "segments", "meter_markers_frames"],
        )
        report = (
            f"project_timeline duration={effective_duration:.3f}s | total={total_frames}f | "
            f"segments={segment_count} | segment_seconds={segment_seconds:.3f}s | fps={effective_fps:.3f}"
        )
        return (
            project_timeline_payload,
            contract,
            linx_payload,
            int(total_frames),
            int(segment_count),
            float(effective_duration),
            float(segment_seconds),
            int(unique_segment_frames),
            segment_ranges_json,
            report,
        )


# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
class IAMCCS_Ltx2HelperModules_Planner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "goal1_preset": (["custom", "lowram_safe_videoclip", "lowram_fast_videoclip", "normal_quality_videoclip", "lowram_monologue"], {"default": "custom"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.001, "max": 240.0, "step": 0.01}),
                "segment_seconds": ("FLOAT", {"default": 10.0, "min": 0.01, "max": 3600.0, "step": 0.01}),
                "target_duration_seconds": ("FLOAT", {"default": 30.0, "min": 0.01, "max": 36000.0, "step": 0.01}),
                "duration_source": (["target_duration_seconds", "audio", "max_audio_target"], {"default": "target_duration_seconds"}),
                "planning_mode": (["manual_segment_seconds", "auto_profile"], {"default": "auto_profile"}),
                "content_profile": (["videoclip", "monologue"], {"default": "videoclip"}),
                "overlap_frames": ("INT", {"default": 9, "min": 0, "max": 4096, "step": 1}),
                "ltx_round_mode": (["up", "nearest", "down"], {"default": "up"}),
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "surface_profile": (["compact", "progressive", "debug_surface"], {"default": "compact"}),
                "backend_binding": (["ltx_audio_guided_lowram", "ltx_audio_guided_normal", "custom"], {"default": "ltx_audio_guided_lowram"}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "upstream_contract": (SUPERNODE_CONTRACT_TYPE,),
                "linx": (SUPERNODE_LINX_TYPE,),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", SUPERNODE_CONTRACT_TYPE, SUPERNODE_LINX_TYPE, "INT", "INT", "INT", "INT", "INT", "STRING", "INT", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = (
        "planner_payload",
        "contract",
        "linx",
        "segment_count",
        "continuation_loops",
        "current_segment_start_frames",
        "current_segment_unique_frames",
        "current_source_end",
        "report",
        "total_frames",
        "unique_segment_frames",
        "first_segment_raw_frames",
        "continuation_raw_frames",
        "fps_out",
    )
    FUNCTION = "build"
    CATEGORY = HELPER_CATEGORY

    def build(
        self,
        goal1_preset,
        fps,
        segment_seconds,
        target_duration_seconds,
        duration_source,
        planning_mode,
        content_profile,
        overlap_frames,
        ltx_round_mode,
        segment_index,
        surface_profile,
        backend_binding,
        audio=None,
        upstream_contract=None,
        linx=None,
        unique_id=None,
    ):
        preset_values = _resolve_goal1_preset(goal1_preset)
        segment_seconds = float(preset_values.get("segment_seconds", segment_seconds))
        planning_mode = str(preset_values.get("planning_mode", planning_mode))
        content_profile = str(preset_values.get("content_profile", content_profile))
        overlap_frames = int(preset_values.get("overlap_frames", overlap_frames))
        backend_binding = str(preset_values.get("backend_binding", backend_binding))
        surface_profile = str(preset_values.get("surface_profile", surface_profile))

        effective_duration = float(target_duration_seconds)
        if duration_source == "audio":
            effective_duration = _audio_duration_seconds(audio)
        elif duration_source == "max_audio_target" and audio is not None:
            effective_duration = max(float(target_duration_seconds), _audio_duration_seconds(audio))

        planner_result = IAMCCS_SegmentPlanner().plan(
            effective_duration,
            fps,
            segment_seconds,
            planning_mode,
            content_profile,
            overlap_frames,
            ltx_round_mode,
            segment_index,
        )
        source_range = IAMCCS_SourceRangeFromSegmentPlan().derive(
            planner_result[8],
            planner_result[9],
            planner_result[10],
            planner_result[11],
        )
        planner_payload_dict = {
            "pipeline_kind": PIPELINE_KIND,
            "goal1_preset": str(goal1_preset),
            "target_duration_seconds": float(effective_duration),
            "duration_source": str(duration_source),
            "surface_profile": str(surface_profile),
            "backend_binding": str(backend_binding),
            "fps": float(planner_result[17]),
            "total_frames": int(planner_result[0]),
            "unique_segment_frames": int(planner_result[1]),
            "first_segment_raw_frames": int(planner_result[2]),
            "continuation_raw_frames": int(planner_result[3]),
            "segment_count": int(planner_result[4]),
            "continuation_loops": int(planner_result[5]),
            "last_segment_unique_frames": int(planner_result[6]),
            "segment_index": int(planner_result[8]),
            "current_segment_raw_frames": int(planner_result[9]),
            "current_segment_unique_frames": int(planner_result[10]),
            "current_segment_start_frames": int(planner_result[11]),
            "current_segment_end_frames": int(planner_result[12]),
            "current_remaining_frames_after": int(planner_result[13]),
            "current_segment_start_s": float(planner_result[14]),
            "current_segment_end_s": float(planner_result[15]),
            "recommended_overlap_frames": int(planner_result[18]),
            "recommended_audio_left_context_s": float(planner_result[19]),
            "recommended_extension_preset": str(planner_result[20]),
            "effective_planning_mode": str(planner_result[21]),
            "planning_profile_report": str(planner_result[22]),
            "current_source_start": int(source_range[0]),
            "current_source_end": int(source_range[1]),
            "current_source_count": int(source_range[2]),
        }
        planner_payload = _json_payload(planner_payload_dict)
        contract = _build_contract(
            node_role="planner",
            node_label=f"{HELPER_LABEL_PREFIX} Planner",
            parent_contract=upstream_contract,
            payload_key="planner_payload",
            payload_value=planner_payload,
            extra={"surface_profile": surface_profile, "backend_binding": backend_binding},
        )
        linx_payload = _build_linx(
            existing_linx=linx,
            contract=contract,
            unique_id=unique_id,
            node_role="planner",
            node_label=f"{HELPER_LABEL_PREFIX} Planner",
        )
        report = (
            f"ltx2_helper_modules_planner preset={planner_payload_dict['goal1_preset']} | duration={effective_duration:.3f}s | total={planner_payload_dict['total_frames']}f | "
            f"segments={planner_payload_dict['segment_count']} | seg={planner_payload_dict['segment_index']} | "
            f"source=[{planner_payload_dict['current_source_start']}..{planner_payload_dict['current_source_end']}) | "
            f"preset={planner_payload_dict['recommended_extension_preset']}"
        )
        return (
            planner_payload,
            contract,
            linx_payload,
            int(planner_payload_dict["segment_count"]),
            int(planner_payload_dict["continuation_loops"]),
            int(planner_payload_dict["current_segment_start_frames"]),
            int(planner_payload_dict["current_segment_unique_frames"]),
            int(planner_payload_dict["current_source_end"]),
            report,
            int(planner_payload_dict["total_frames"]),
            int(planner_payload_dict["unique_segment_frames"]),
            int(planner_payload_dict["first_segment_raw_frames"]),
            int(planner_payload_dict["continuation_raw_frames"]),
            float(planner_payload_dict["fps"]),
        )


# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
class IAMCCS_Ltx2HelperModules_AudioTimeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "planner_payload": ("STRING", {"default": "{}", "multiline": True}),
                "audio": ("AUDIO",),
                "mode": (["left_context_only", "right_context_only", "symmetric_context", "no_overlap"], {"default": "left_context_only"}),
                "left_overlap_s": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 30.0, "step": 0.01}),
                "right_overlap_s": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "clamp_policy": (["soft_clamp", "strict"], {"default": "soft_clamp"}),
                "min_next_frames": ("INT", {"default": 1, "min": 1, "max": 100000, "step": 1}),
            },
            "optional": {
                "parent_contract": (SUPERNODE_CONTRACT_TYPE,),
                "linx": (SUPERNODE_LINX_TYPE,),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", SUPERNODE_CONTRACT_TYPE, SUPERNODE_LINX_TYPE, "AUDIO", "AUDIO", "BOOLEAN", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "audio_payload",
        "contract",
        "linx",
        "conditioning_audio",
        "segment_audio",
        "continue_generation",
        "effective_unique_frames",
        "trim_frames",
        "report",
    )
    FUNCTION = "build"
    CATEGORY = HELPER_CATEGORY

    def build(self, planner_payload, audio, mode, left_overlap_s, right_overlap_s, clamp_policy, min_next_frames, parent_contract=None, linx=None, unique_id=None):
        planner = _parse_payload(planner_payload)
        fps = float(planner.get("fps", 24.0))
        segment_index = int(planner.get("segment_index", 0))
        generated_frames = int(planner.get("current_segment_raw_frames", planner.get("continuation_raw_frames", 1)))
        extension_frames = int(planner.get("current_segment_unique_frames", planner.get("unique_segment_frames", generated_frames)))
        segment_start_frames = int(planner.get("current_segment_start_frames", 0))
        first_pass_unique_frames = int(planner.get("unique_segment_frames", extension_frames))
        segment_duration_s = float(extension_frames) / float(max(fps, 0.001))

        math_result = IAMCCS_AudioExtensionMath().compute(
            audio,
            fps,
            segment_index,
            generated_frames,
            extension_frames,
            True,
            first_pass_unique_frames=first_pass_unique_frames,
            cursor_frames_in=segment_start_frames,
        )
        extender_result = IAMCCS_AudioExtender().slice_segment(
            audio,
            fps,
            mode,
            left_overlap_s,
            right_overlap_s,
            "use_timeline_cursor",
            "snap_to_video_duration",
            clamp_policy,
            segment_index=segment_index,
            segment_duration_s=segment_duration_s,
            video_frames=generated_frames,
            generated_frames=generated_frames,
            extension_frames=extension_frames,
            timeline_cursor_frames=segment_start_frames,
            segment_start_frames=math_result[1],
            effective_unique_frames=math_result[3],
            first_pass_unique_frames=first_pass_unique_frames,
        )
        gate_result = IAMCCS_AudioTimelineGate().decide(
            math_result[7],
            math_result[3],
            min_next_frames,
            True,
            is_last_segment=math_result[8],
            cursor_frames_out=math_result[0],
        )

        audio_payload_dict = {
            "pipeline_kind": PIPELINE_KIND,
            "segment_index": segment_index,
            "fps": fps,
            "cursor_frames_out": int(math_result[0]),
            "segment_start_frames": int(math_result[1]),
            "segment_end_frames": int(math_result[2]),
            "effective_unique_frames": int(math_result[3]),
            "trim_frames": int(math_result[4]),
            "segment_start_s": float(math_result[5]),
            "segment_end_s": float(math_result[6]),
            "remaining_frames_after": int(math_result[7]),
            "is_last_segment": int(math_result[8]),
            "conditioning_duration_s": float(extender_result[3]),
            "segment_duration_s": float(extender_result[2]),
            "segment_start_sample": int(extender_result[4]),
            "segment_end_sample": int(extender_result[5]),
            "conditioning_start_sample": int(extender_result[6]),
            "conditioning_end_sample": int(extender_result[7]),
            "continue_generation": bool(gate_result[0]),
            "gate_reason": str(gate_result[4]),
        }
        audio_payload = _json_payload(audio_payload_dict)
        contract = _build_contract(
            node_role="audio",
            node_label=f"{HELPER_LABEL_PREFIX} AudioTimeline",
            parent_contract=parent_contract,
            payload_key="audio_payload",
            payload_value=audio_payload,
        )
        linx_payload = _build_linx(
            existing_linx=linx,
            contract=contract,
            unique_id=unique_id,
            node_role="audio",
            node_label=f"{HELPER_LABEL_PREFIX} AudioTimeline",
        )
        report = (
            f"ltx2_helper_modules_audio seg={segment_index} | effective={audio_payload_dict['effective_unique_frames']}f | "
            f"trim={audio_payload_dict['trim_frames']}f | continue={'yes' if audio_payload_dict['continue_generation'] else 'no'} | "
            f"gate={audio_payload_dict['gate_reason']}"
        )
        return (
            audio_payload,
            contract,
            linx_payload,
            extender_result[0],
            extender_result[1],
            bool(audio_payload_dict["continue_generation"]),
            int(audio_payload_dict["effective_unique_frames"]),
            int(audio_payload_dict["trim_frames"]),
            report,
        )


# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
class IAMCCS_Ltx2HelperModules_KeyframeTimeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "planner_payload": ("STRING", {"default": "{}", "multiline": True}),
                "keyframe_positions_mode": (["global_frames", "global_seconds", "global_ratio", "none"], {"default": "none"}),
                "keyframe_positions": ("STRING", {"default": "", "multiline": True}),
                "refresh_mode_policy": (["auto", "anchor_only", "tail_refresh", "flf_blend", "disabled"], {"default": "auto"}),
                "protected_head_frames": ("INT", {"default": 9, "min": 0, "max": 256, "step": 1}),
                "settling_frames_default": ("INT", {"default": 9, "min": 0, "max": 256, "step": 1}),
                "min_keyframe_spacing_frames": ("INT", {"default": 8, "min": 0, "max": 512, "step": 1}),
                "refresh_interval_segments": ("INT", {"default": 2, "min": 0, "max": 256, "step": 1}),
                "reanchor_every_segments": ("INT", {"default": 3, "min": 0, "max": 256, "step": 1}),
                "prefer_tail_refresh": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "parent_contract": (SUPERNODE_CONTRACT_TYPE,),
                "linx": (SUPERNODE_LINX_TYPE,),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", SUPERNODE_CONTRACT_TYPE, SUPERNODE_LINX_TYPE, "BOOLEAN", "BOOLEAN", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "keyframe_payload",
        "contract",
        "linx",
        "has_refresh",
        "is_reanchor_segment",
        "recommended_refresh_mode",
        "protected_head_frames_out",
        "preferred_tail_start",
        "report",
    )
    FUNCTION = "build"
    CATEGORY = HELPER_CATEGORY

    def build(
        self,
        planner_payload,
        keyframe_positions_mode,
        keyframe_positions,
        refresh_mode_policy,
        protected_head_frames,
        settling_frames_default,
        min_keyframe_spacing_frames,
        refresh_interval_segments,
        reanchor_every_segments,
        prefer_tail_refresh,
        parent_contract=None,
        linx=None,
        unique_id=None,
    ):
        planner = _parse_payload(planner_payload)
        goal1_preset = str(planner.get("goal1_preset", "custom"))
        defaults = _resolve_goal1_refresh_defaults(goal1_preset)
        fps = float(planner.get("fps", 24.0))
        total_frames = int(planner.get("total_frames", 0))
        segment_index = int(planner.get("segment_index", 0))
        segment_start_frame = int(planner.get("current_segment_start_frames", 0))
        segment_end_frame = int(planner.get("current_segment_end_frames", segment_start_frame))
        unique_frames = int(planner.get("current_segment_unique_frames", planner.get("unique_segment_frames", 0)))

        effective_protected_head = int(defaults.get("protected_head_frames", protected_head_frames))
        effective_settling = int(defaults.get("settling_frames", settling_frames_default))
        effective_refresh_interval = int(defaults.get("refresh_interval_segments", refresh_interval_segments))
        effective_reanchor_every = int(defaults.get("reanchor_every_segments", reanchor_every_segments))

        global_positions = _parse_numeric_list(keyframe_positions)
        converted_frames: list[int] = []
        for value in global_positions:
            if keyframe_positions_mode == "global_seconds":
                converted_frames.append(int(round(value * fps)))
            elif keyframe_positions_mode == "global_ratio":
                converted_frames.append(int(round(value * total_frames)))
            elif keyframe_positions_mode == "global_frames":
                converted_frames.append(int(round(value)))

        local_refresh_frames: list[int] = []
        for frame in sorted(converted_frames):
            local_frame = int(frame) - segment_start_frame
            if local_frame < effective_protected_head:
                continue
            if local_frame >= unique_frames:
                continue
            if local_refresh_frames and (local_frame - local_refresh_frames[-1]) < int(min_keyframe_spacing_frames):
                continue
            local_refresh_frames.append(local_frame)

        periodic_refresh = bool(effective_refresh_interval > 0 and segment_index > 0 and (segment_index % effective_refresh_interval) == 0)
        is_reanchor_segment = bool(effective_reanchor_every > 0 and segment_index > 0 and (segment_index % effective_reanchor_every) == 0)
        preferred_tail_start = max(effective_protected_head, unique_frames - max(effective_settling, 1))
        if periodic_refresh and preferred_tail_start not in local_refresh_frames:
            local_refresh_frames.append(preferred_tail_start)
            local_refresh_frames.sort()

        has_refresh = bool(local_refresh_frames or periodic_refresh or is_reanchor_segment)
        recommended_refresh_mode = str(refresh_mode_policy)
        if recommended_refresh_mode == "auto":
            if is_reanchor_segment:
                recommended_refresh_mode = "flf_blend"
            elif has_refresh and bool(prefer_tail_refresh):
                recommended_refresh_mode = "tail_refresh"
            elif has_refresh:
                recommended_refresh_mode = "anchor_only"
            else:
                recommended_refresh_mode = "disabled"

        keyframe_payload_dict = {
            "pipeline_kind": PIPELINE_KIND,
            "feature_family": "keyframe_refresh",
            "goal1_preset": goal1_preset,
            "segment_index": segment_index,
            "segment_start_frame": segment_start_frame,
            "segment_end_frame": segment_end_frame,
            "segment_role": "reanchor" if is_reanchor_segment else "continuity",
            "keyframe_positions_mode": str(keyframe_positions_mode),
            "local_refresh_frames": list(local_refresh_frames),
            "has_refresh": bool(has_refresh),
            "periodic_refresh": bool(periodic_refresh),
            "is_reanchor_segment": bool(is_reanchor_segment),
            "recommended_refresh_mode": recommended_refresh_mode,
            "protected_head_frames": int(effective_protected_head),
            "settling_frames": int(effective_settling),
            "trim_advisory": int(effective_settling if has_refresh else 0),
            "preferred_tail_start": int(preferred_tail_start),
            "continuity_preset_override": str(defaults.get("continuity_preset", planner.get("recommended_extension_preset", "custom"))),
            "reanchor_preset_override": str(defaults.get("reanchor_preset", "micro_crossfade_3")),
            "reanchor_overlap_mode_override": str(defaults.get("reanchor_overlap_mode", "filmic_crossfade")),
            "reanchor_safe_mode_override": str(defaults.get("reanchor_safe_mode", "native_workflow_safe")),
            "reanchor_start_frames_rule_override": str(defaults.get("reanchor_start_frames_rule", "ltx2_nearest")),
        }
        keyframe_payload = _json_payload(keyframe_payload_dict)
        contract = _build_contract(
            node_role="keyframe_timeline",
            node_label=f"{HELPER_LABEL_PREFIX} KeyframeTimeline",
            parent_contract=parent_contract,
            payload_key="keyframe_payload",
            payload_value=keyframe_payload,
        )
        linx_payload = _build_linx(
            existing_linx=linx,
            contract=contract,
            unique_id=unique_id,
            node_role="keyframe_timeline",
            node_label=f"{HELPER_LABEL_PREFIX} KeyframeTimeline",
        )
        report = (
            f"ltx2_helper_modules_keyframe seg={segment_index} | role={keyframe_payload_dict['segment_role']} | "
            f"refresh={recommended_refresh_mode} | local={keyframe_payload_dict['local_refresh_frames']}"
        )
        return (
            keyframe_payload,
            contract,
            linx_payload,
            bool(has_refresh),
            bool(is_reanchor_segment),
            recommended_refresh_mode,
            int(effective_protected_head),
            int(preferred_tail_start),
            report,
        )


# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
class IAMCCS_Ltx2HelperModules_RefreshPolicy:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "planner_payload": ("STRING", {"default": "{}", "multiline": True}),
                "keyframe_payload": ("STRING", {"default": "{}", "multiline": True}),
                "segment_policy": (["planner_driven", "continuity_only", "reanchor_mix", "reanchor_only"], {"default": "planner_driven"}),
                "continuity_preset": (["custom", "target_extension_ltx2", "videoclip_audio_24fps", "monologue_audio_24fps", "cut_bestofk_16", "cut_bestofk_16_luma", "cut_bestofk_32", "micro_crossfade_3"], {"default": "videoclip_audio_24fps"}),
                "reanchor_preset": (["custom", "target_extension_ltx2", "videoclip_audio_24fps", "monologue_audio_24fps", "cut_bestofk_16", "cut_bestofk_16_luma", "cut_bestofk_32", "micro_crossfade_3"], {"default": "micro_crossfade_3"}),
                "first_anchor_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "last_anchor_strength": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tail_refresh_strength": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01}),
                "reanchor_overlap_mode": (["cut", "linear_blend", "ease_in_out", "filmic_crossfade", "perceptual_crossfade"], {"default": "filmic_crossfade"}),
            },
            "optional": {
                "parent_contract": (SUPERNODE_CONTRACT_TYPE,),
                "linx": (SUPERNODE_LINX_TYPE,),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", SUPERNODE_CONTRACT_TYPE, SUPERNODE_LINX_TYPE, "BOOLEAN", "STRING", "STRING", "FLOAT", "FLOAT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "refresh_payload",
        "contract",
        "linx",
        "apply_refresh",
        "segment_role",
        "continuity_preset_override",
        "first_anchor_strength_effective",
        "last_anchor_strength_effective",
        "first_lock_slots",
        "last_lock_slots",
        "end_transition_slots",
        "report",
    )
    FUNCTION = "build"
    CATEGORY = HELPER_CATEGORY

    def build(
        self,
        planner_payload,
        keyframe_payload,
        segment_policy,
        continuity_preset,
        reanchor_preset,
        first_anchor_strength,
        last_anchor_strength,
        tail_refresh_strength,
        reanchor_overlap_mode,
        parent_contract=None,
        linx=None,
        unique_id=None,
    ):
        planner = _parse_payload(planner_payload)
        keyframe = _parse_payload(keyframe_payload)
        goal1_preset = str(planner.get("goal1_preset", "custom"))
        defaults = _resolve_goal1_refresh_defaults(goal1_preset)

        planner_reanchor = bool(keyframe.get("is_reanchor_segment", False))
        if segment_policy == "reanchor_only":
            segment_role = "reanchor"
        elif segment_policy == "continuity_only":
            segment_role = "continuity"
        else:
            segment_role = "reanchor" if planner_reanchor else "continuity"

        apply_refresh = bool(segment_role == "reanchor" or keyframe.get("has_refresh", False))
        continuity_preset_override = str(defaults.get("continuity_preset", continuity_preset))
        reanchor_preset_override = str(defaults.get("reanchor_preset", reanchor_preset))
        effective_preset_override = reanchor_preset_override if segment_role == "reanchor" else continuity_preset_override
        first_anchor_strength_effective = float(defaults.get("first_anchor_strength", first_anchor_strength)) if segment_role == "reanchor" else 0.0
        last_anchor_strength_effective = float(defaults.get("last_anchor_strength", last_anchor_strength)) if segment_role == "reanchor" else 0.0
        first_lock_slots_effective = int(defaults.get("first_lock_slots", 1)) if segment_role == "reanchor" else 0
        last_lock_slots_effective = int(defaults.get("last_lock_slots", 1)) if segment_role == "reanchor" else 0
        end_transition_slots_effective = int(defaults.get("end_transition_slots", 0)) if segment_role == "reanchor" else 0

        refresh_payload_dict = {
            "pipeline_kind": PIPELINE_KIND,
            "feature_family": "keyframe_refresh",
            "goal1_preset": goal1_preset,
            "segment_index": int(planner.get("segment_index", 0)),
            "segment_role": segment_role,
            "segment_policy": str(segment_policy),
            "apply_refresh": bool(apply_refresh),
            "recommended_refresh_mode": str(keyframe.get("recommended_refresh_mode", "disabled")),
            "continuity_preset_override": continuity_preset_override,
            "reanchor_preset_override": reanchor_preset_override,
            "continuity_preset_effective": effective_preset_override,
            "reanchor_overlap_mode_override": str(defaults.get("reanchor_overlap_mode", reanchor_overlap_mode)),
            "reanchor_safe_mode_override": str(defaults.get("reanchor_safe_mode", "native_workflow_safe")),
            "reanchor_start_frames_rule_override": str(defaults.get("reanchor_start_frames_rule", "ltx2_nearest")),
            "protected_head_frames": int(keyframe.get("protected_head_frames", defaults.get("protected_head_frames", 9))),
            "preferred_tail_start": int(keyframe.get("preferred_tail_start", 0)),
            "first_anchor_strength": float(first_anchor_strength),
            "last_anchor_strength": float(last_anchor_strength),
            "tail_refresh_strength": float(tail_refresh_strength),
            "first_anchor_strength_effective": float(first_anchor_strength_effective),
            "last_anchor_strength_effective": float(last_anchor_strength_effective),
            "first_lock_slots": int(first_lock_slots_effective),
            "last_lock_slots": int(last_lock_slots_effective),
            "end_transition_slots": int(end_transition_slots_effective),
            "hide_historical_preset_nodes": True,
        }
        refresh_payload = _json_payload(refresh_payload_dict)
        contract = _build_contract(
            node_role="refresh_policy",
            node_label=f"{HELPER_LABEL_PREFIX} RefreshPolicy",
            parent_contract=parent_contract,
            payload_key="refresh_payload",
            payload_value=refresh_payload,
        )
        linx_payload = _build_linx(
            existing_linx=linx,
            contract=contract,
            unique_id=unique_id,
            node_role="refresh_policy",
            node_label=f"{HELPER_LABEL_PREFIX} RefreshPolicy",
        )
        report = (
            f"ltx2_helper_modules_refresh seg={refresh_payload_dict['segment_index']} | role={segment_role} | "
            f"apply={'yes' if apply_refresh else 'no'} | preset={effective_preset_override}"
        )
        return (
            refresh_payload,
            contract,
            linx_payload,
            bool(apply_refresh),
            segment_role,
            effective_preset_override,
            float(first_anchor_strength_effective),
            float(last_anchor_strength_effective),
            int(first_lock_slots_effective),
            int(last_lock_slots_effective),
            int(end_transition_slots_effective),
            report,
        )


# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
class IAMCCS_Ltx2HelperModules_ReanchorLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "refresh_payload": ("STRING", {"default": "{}", "multiline": True}),
            },
            "optional": {
                "first_anchor_image": ("IMAGE",),
                "last_anchor_image": ("IMAGE",),
                "parent_contract": (SUPERNODE_CONTRACT_TYPE,),
                "linx": (SUPERNODE_LINX_TYPE,),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LATENT", "STRING", SUPERNODE_CONTRACT_TYPE, SUPERNODE_LINX_TYPE, "STRING")
    RETURN_NAMES = ("latent", "reanchor_payload", "contract", "linx", "report")
    FUNCTION = "build"
    CATEGORY = HELPER_CATEGORY

    def build(self, vae, latent, refresh_payload, first_anchor_image=None, last_anchor_image=None, parent_contract=None, linx=None, unique_id=None):
        refresh = _parse_payload(refresh_payload)
        segment_role = str(refresh.get("segment_role", "continuity"))
        first_strength = float(refresh.get("first_anchor_strength_effective", 0.0))
        last_strength = float(refresh.get("last_anchor_strength_effective", 0.0))
        first_lock_slots = int(refresh.get("first_lock_slots", 0))
        last_lock_slots = int(refresh.get("last_lock_slots", 0))
        end_transition_slots = int(refresh.get("end_transition_slots", 0))

        if segment_role != "reanchor" or (first_strength <= 0.0 and last_strength <= 0.0):
            out_latent = latent
            reanchor_payload_dict = {
                "pipeline_kind": PIPELINE_KIND,
                "segment_role": segment_role,
                "reanchor_applied": False,
            }
            report = "ltx2_helper_modules_reanchor latent passthrough"
        else:
            out_latent, inner_report = IAMCCS_LTX2_FirstLastLatentControl_Pro().execute(
                vae,
                latent,
                first_strength,
                last_strength,
                first_lock_slots,
                last_lock_slots,
                end_transition_slots,
                first_image=first_anchor_image,
                last_image=last_anchor_image,
                middle_frames=None,
            )
            reanchor_payload_dict = {
                "pipeline_kind": PIPELINE_KIND,
                "segment_role": segment_role,
                "reanchor_applied": True,
                "first_anchor_strength_effective": float(first_strength),
                "last_anchor_strength_effective": float(last_strength),
                "first_lock_slots": int(first_lock_slots),
                "last_lock_slots": int(last_lock_slots),
            }
            report = f"ltx2_helper_modules_reanchor {inner_report}"

        reanchor_payload = _json_payload(reanchor_payload_dict)
        contract = _build_contract(
            node_role="reanchor_latent",
            node_label=f"{HELPER_LABEL_PREFIX} ReanchorLatent",
            parent_contract=parent_contract,
            payload_key="reanchor_payload",
            payload_value=reanchor_payload,
        )
        linx_payload = _build_linx(
            existing_linx=linx,
            contract=contract,
            unique_id=unique_id,
            node_role="reanchor_latent",
            node_label=f"{HELPER_LABEL_PREFIX} ReanchorLatent",
        )
        return (out_latent, reanchor_payload, contract, linx_payload, report)


# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
class IAMCCS_Ltx2HelperModules_DiskExtension:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_dir": ("STRING", {"default": "iamccs_extension_disk/loop_seg0_extended"}),
                "new_dir": ("STRING", {"default": "iamccs_vae_frames/loop_audio_sync/current"}),
                "planner_payload": ("STRING", {"default": "{}", "multiline": True}),
                "refresh_payload": ("STRING", {"default": "{}", "multiline": True}),
                "output_dir": ("STRING", {"default": "iamccs_extension_disk/loop_extended"}),
                "start_dir": ("STRING", {"default": "iamccs_extension_disk/loop_start"}),
            },
            "optional": {
                "parent_contract": (SUPERNODE_CONTRACT_TYPE,),
                "linx": (SUPERNODE_LINX_TYPE,),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "INT", "STRING", SUPERNODE_CONTRACT_TYPE, SUPERNODE_LINX_TYPE, "STRING")
    RETURN_NAMES = (
        "extended_dir",
        "start_dir_out",
        "overlap_frames",
        "calculated_frames",
        "extension_frames",
        "disk_payload",
        "contract",
        "linx",
        "report",
    )
    FUNCTION = "build"
    CATEGORY = HELPER_CATEGORY

    def build(self, source_dir, new_dir, planner_payload, refresh_payload, output_dir, start_dir, parent_contract=None, linx=None, unique_id=None):
        planner = _parse_payload(planner_payload)
        refresh = _parse_payload(refresh_payload)
        segment_role = str(refresh.get("segment_role", "continuity"))
        preset_override = str(refresh.get("continuity_preset_effective", planner.get("recommended_extension_preset", "custom")))
        overlap_mode = str(refresh.get("reanchor_overlap_mode_override", "cut")) if segment_role == "reanchor" else "cut"
        safe_mode = str(refresh.get("reanchor_safe_mode_override", "none")) if segment_role == "reanchor" else "none"
        start_frames_rule = str(refresh.get("reanchor_start_frames_rule_override", "none")) if segment_role == "reanchor" else "none"
        overlap_frames = int(planner.get("recommended_overlap_frames", 9))

        extended_dir, start_dir_out, effective_overlap, calculated_frames, extension_frames, inner_report = IAMCCS_LTX2_ExtensionModule_Disk().process_extension_disk(
            source_dir,
            output_dir,
            start_dir,
            overlap_frames,
            "source",
            overlap_mode,
            True,
            "none",
            safe_mode,
            start_frames_rule,
            preset=preset_override,
            new_dir=new_dir,
            math_value_b=1,
        )
        disk_payload_dict = {
            "pipeline_kind": PIPELINE_KIND,
            "segment_role": segment_role,
            "preset_override": preset_override,
            "extended_dir": str(extended_dir),
            "start_dir": str(start_dir_out),
            "overlap_frames": int(effective_overlap),
            "calculated_frames": int(calculated_frames),
            "extension_frames": int(extension_frames),
        }
        disk_payload = _json_payload(disk_payload_dict)
        contract = _build_contract(
            node_role="disk_extension",
            node_label=f"{HELPER_LABEL_PREFIX} DiskExtension",
            parent_contract=parent_contract,
            payload_key="disk_payload",
            payload_value=disk_payload,
        )
        linx_payload = _build_linx(
            existing_linx=linx,
            contract=contract,
            unique_id=unique_id,
            node_role="disk_extension",
            node_label=f"{HELPER_LABEL_PREFIX} DiskExtension",
        )
        report = f"ltx2_helper_modules_disk role={segment_role} | preset={preset_override} | {inner_report}"
        return (
            str(extended_dir),
            str(start_dir_out),
            int(effective_overlap),
            int(calculated_frames),
            int(extension_frames),
            disk_payload,
            contract,
            linx_payload,
            report,
        )


# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
class IAMCCS_Ltx2HelperModules_RuntimeBridge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "planner_payload": ("STRING", {"default": "{}", "multiline": True}),
                "audio_payload": ("STRING", {"default": "{}", "multiline": True}),
                "runtime_profile": (["native_canvas", "disk_low_ram", "image_batch"], {"default": "native_canvas"}),
                "model_binding": (["external_canvas", "ltx_runtime", "custom"], {"default": "external_canvas"}),
                "prompt_summary": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "parent_contract": (SUPERNODE_CONTRACT_TYPE,),
                "linx": (SUPERNODE_LINX_TYPE,),
                "continuity_payload": ("STRING", {"default": "{}", "multiline": True}),
                "keyframe_payload": ("STRING", {"default": "{}", "multiline": True}),
                "refresh_payload": ("STRING", {"default": "{}", "multiline": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", SUPERNODE_CONTRACT_TYPE, SUPERNODE_LINX_TYPE, "STRING")
    RETURN_NAMES = ("backend_payload", "contract", "linx", "report")
    FUNCTION = "build"
    CATEGORY = HELPER_CATEGORY

    def build(self, planner_payload, audio_payload, runtime_profile, model_binding, prompt_summary, parent_contract=None, linx=None, continuity_payload="{}", keyframe_payload="{}", refresh_payload="{}", unique_id=None):
        planner = _parse_payload(planner_payload)
        audio = _parse_payload(audio_payload)
        continuity = _parse_payload(continuity_payload)
        keyframe = _parse_payload(keyframe_payload)
        refresh = _parse_payload(refresh_payload)
        goal1_preset = str(planner.get("goal1_preset", "custom"))
        effective_backend_binding = str(planner.get("backend_binding", model_binding))
        effective_runtime_profile = _resolve_goal1_runtime_profile(goal1_preset, effective_backend_binding, runtime_profile)
        backend_payload_dict = {
            "pipeline_kind": PIPELINE_KIND,
            "goal1_preset": goal1_preset,
            "runtime_profile": effective_runtime_profile,
            "model_binding": effective_backend_binding,
            "segment_index": int(planner.get("segment_index", 0)),
            "segment_count": int(planner.get("segment_count", 1)),
            "recommended_extension_preset": str(planner.get("recommended_extension_preset", "custom")),
            "effective_unique_frames": int(audio.get("effective_unique_frames", 0)),
            "continue_generation": bool(audio.get("continue_generation", False)),
            "has_continuity_payload": bool(continuity),
            "has_keyframe_payload": bool(keyframe),
            "has_refresh_payload": bool(refresh),
            "segment_role": str(refresh.get("segment_role", keyframe.get("segment_role", "continuity"))),
            "recommended_refresh_mode": str(keyframe.get("recommended_refresh_mode", "disabled")),
            "continuity_preset_effective": str(refresh.get("continuity_preset_effective", planner.get("recommended_extension_preset", "custom"))),
            "prompt_summary": str(prompt_summary or ""),
        }
        backend_payload = _json_payload(backend_payload_dict)
        contract = _build_contract(
            node_role="backend",
            node_label=f"{HELPER_LABEL_PREFIX} RuntimeBridge",
            parent_contract=parent_contract,
            payload_key="backend_payload",
            payload_value=backend_payload,
        )
        linx_payload = _build_linx(
            existing_linx=linx,
            contract=contract,
            unique_id=unique_id,
            node_role="backend",
            node_label=f"{HELPER_LABEL_PREFIX} RuntimeBridge",
        )
        report = (
            f"ltx2_helper_modules_runtime goal1={goal1_preset} | seg={backend_payload_dict['segment_index']}/{backend_payload_dict['segment_count']} | "
            f"runtime={effective_runtime_profile} | role={backend_payload_dict['segment_role']} | model_binding={effective_backend_binding} | continue={'yes' if backend_payload_dict['continue_generation'] else 'no'}"
        )
        return (backend_payload, contract, linx_payload, report)


# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
class IAMCCS_Ltx2HelperModules_Continuity:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_images": ("IMAGE",),
                "overlap_frames": ("INT", {"default": 9, "min": 1, "max": 256, "step": 1}),
                "overlap_side": (["source", "new_images"], {"default": "source"}),
                "overlap_mode": (["cut", "linear_blend", "ease_in_out", "filmic_crossfade", "perceptual_crossfade"], {"default": "cut"}),
                "enable_math": ("BOOLEAN", {"default": True}),
                "math_operation": (["none", "a-b", "a-1", "a+b", "a*b", "a/b", "min(a,b)", "max(a,b)"], {"default": "a-b"}),
                "safe_mode": (["none", "native_workflow_safe"], {"default": "none"}),
                "start_frames_rule": (["none", "ltx2_round_down", "ltx2_nearest"], {"default": "none"}),
                "preset": (["custom", "target_extension_ltx2", "videoclip_audio_24fps", "monologue_audio_24fps", "cut_bestofk_16", "cut_bestofk_16_luma", "cut_bestofk_32", "micro_crossfade_3"], {"default": "custom"}),
                "use_planner_preset": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "new_images": ("IMAGE",),
                "planner_payload": ("STRING", {"default": "{}", "multiline": True}),
                "refresh_payload": ("STRING", {"default": "{}", "multiline": True}),
                "parent_contract": (SUPERNODE_CONTRACT_TYPE,),
                "linx": (SUPERNODE_LINX_TYPE,),
                "math_value_b": ("INT", {"default": 1, "min": 0, "max": 256, "step": 1}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", SUPERNODE_CONTRACT_TYPE, SUPERNODE_LINX_TYPE, "IMAGE", "IMAGE", "IMAGE", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "continuity_payload",
        "contract",
        "linx",
        "source_images_passthrough",
        "start_images",
        "extended_images",
        "overlap_frames_out",
        "calculated_frames",
        "extension_frames",
        "report",
    )
    FUNCTION = "build"
    CATEGORY = HELPER_CATEGORY

    def build(
        self,
        source_images,
        overlap_frames,
        overlap_side,
        overlap_mode,
        enable_math,
        math_operation,
        safe_mode,
        start_frames_rule,
        preset,
        use_planner_preset,
        new_images=None,
        planner_payload="{}",
        refresh_payload="{}",
        parent_contract=None,
        linx=None,
        math_value_b=1,
        unique_id=None,
    ):
        planner = _parse_payload(planner_payload)
        refresh = _parse_payload(refresh_payload)
        goal1_preset = str(planner.get("goal1_preset", "custom"))
        effective_preset = str(planner.get("recommended_extension_preset", preset)) if bool(use_planner_preset) else str(preset)
        if refresh.get("continuity_preset_effective"):
            effective_preset = str(refresh.get("continuity_preset_effective"))
        effective_overlap_frames = int(planner.get("recommended_overlap_frames", overlap_frames)) if bool(use_planner_preset) else int(overlap_frames)
        effective_overlap_mode = str(refresh.get("reanchor_overlap_mode_override", overlap_mode)) if str(refresh.get("segment_role", "continuity")) == "reanchor" else str(overlap_mode)
        effective_safe_mode = str(refresh.get("reanchor_safe_mode_override", safe_mode)) if str(refresh.get("segment_role", "continuity")) == "reanchor" else str(safe_mode)
        effective_start_frames_rule = str(refresh.get("reanchor_start_frames_rule_override", start_frames_rule)) if str(refresh.get("segment_role", "continuity")) == "reanchor" else str(start_frames_rule)
        result = IAMCCS_LTX2_ExtensionModule().process_extension(
            source_images,
            effective_overlap_frames,
            overlap_side,
            effective_overlap_mode,
            enable_math,
            math_operation,
            effective_safe_mode,
            effective_start_frames_rule,
            "none",
            1.0,
            8,
            "none",
            0,
            1.0,
            0.5,
            effective_preset,
            new_images=new_images,
            math_value_b=math_value_b,
        )
        continuity_payload_dict = {
            "pipeline_kind": PIPELINE_KIND,
            "goal1_preset": goal1_preset,
            "segment_index": int(planner.get("segment_index", 0)),
            "segment_role": str(refresh.get("segment_role", "continuity")),
            "preset": effective_preset,
            "overlap_frames": int(result[3]),
            "calculated_frames": int(result[4]),
            "extension_frames": int(result[5]),
            "has_new_images": new_images is not None,
        }
        continuity_payload = _json_payload(continuity_payload_dict)
        contract = _build_contract(
            node_role="continuity",
            node_label=f"{HELPER_LABEL_PREFIX} Continuity",
            parent_contract=parent_contract,
            payload_key="continuity_payload",
            payload_value=continuity_payload,
        )
        linx_payload = _build_linx(
            existing_linx=linx,
            contract=contract,
            unique_id=unique_id,
            node_role="continuity",
            node_label=f"{HELPER_LABEL_PREFIX} Continuity",
        )
        report = (
            f"ltx2_helper_modules_continuity goal1={goal1_preset} | seg={continuity_payload_dict['segment_index']} | role={continuity_payload_dict['segment_role']} | preset={effective_preset} | "
            f"overlap={continuity_payload_dict['overlap_frames']} | extension={continuity_payload_dict['extension_frames']}"
        )
        return (
            continuity_payload,
            contract,
            linx_payload,
            result[0],
            result[1],
            result[2],
            int(result[3]),
            int(result[4]),
            int(result[5]),
            report,
        )


# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
class IAMCCS_Ltx2HelperModules_Finalize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames_dir": ("STRING", {"default": "iamccs_extension_disk/final_extended"}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}),
                "filename_prefix": ("STRING", {"default": "IAMCCS/Ltx2HelperModules"}),
                "crf": ("INT", {"default": 19, "min": 0, "max": 51, "step": 1}),
                "pix_fmt": (["yuv420p", "yuv444p"], {"default": "yuv420p"}),
                "trim_to_audio": ("BOOLEAN", {"default": True}),
                "use_planner_fps": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "planner_payload": ("STRING", {"default": "{}", "multiline": True}),
                "parent_contract": (SUPERNODE_CONTRACT_TYPE,),
                "linx": (SUPERNODE_LINX_TYPE,),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", SUPERNODE_CONTRACT_TYPE, SUPERNODE_LINX_TYPE, "STRING", "STRING")
    RETURN_NAMES = ("finalize_payload", "contract", "linx", "video_path", "report")
    FUNCTION = "build"
    CATEGORY = HELPER_CATEGORY
    OUTPUT_NODE = True

    def build(self, frames_dir, frame_rate, filename_prefix, crf, pix_fmt, trim_to_audio, use_planner_fps, audio=None, planner_payload="{}", parent_contract=None, linx=None, unique_id=None):
        planner = _parse_payload(planner_payload)
        goal1_preset = str(planner.get("goal1_preset", "custom"))
        effective_fps = float(planner.get("fps", frame_rate)) if bool(use_planner_fps) else float(frame_rate)
        effective_filename_prefix = _resolve_goal1_filename_prefix(goal1_preset, filename_prefix)
        video_path, combine_report = IAMCCS_VideoCombineFromDir().combine(
            frames_dir,
            effective_fps,
            effective_filename_prefix,
            crf,
            pix_fmt,
            trim_to_audio,
            audio=audio,
        )
        finalize_payload_dict = {
            "pipeline_kind": PIPELINE_KIND,
            "goal1_preset": goal1_preset,
            "frames_dir": str(frames_dir),
            "frame_rate": float(effective_fps),
            "filename_prefix": str(effective_filename_prefix),
            "video_path": str(video_path),
            "trim_to_audio": bool(trim_to_audio),
        }
        finalize_payload = _json_payload(finalize_payload_dict)
        contract = _build_contract(
            node_role="output",
            node_label=f"{HELPER_LABEL_PREFIX} Finalize",
            parent_contract=parent_contract,
            payload_key="finalize_payload",
            payload_value=finalize_payload,
        )
        linx_payload = _build_linx(
            existing_linx=linx,
            contract=contract,
            unique_id=unique_id,
            node_role="output",
            node_label=f"{HELPER_LABEL_PREFIX} Finalize",
        )
        report = f"ltx2_helper_modules_finalize goal1={goal1_preset} | fps={effective_fps:.3f} | path={video_path} | {combine_report}"
        return (finalize_payload, contract, linx_payload, video_path, report)


NODE_CLASS_MAPPINGS = {
    "IAMCCS_ProjectTimelinePlanner": IAMCCS_ProjectTimelinePlanner,
    "IAMCCS_Ltx2HelperModules_ProjectTimelinePlanner": IAMCCS_ProjectTimelinePlanner,
    "IAMCCS_Ltx2HelperModules_Planner": IAMCCS_Ltx2HelperModules_Planner,
    "IAMCCS_Ltx2HelperModules_AudioTimeline": IAMCCS_Ltx2HelperModules_AudioTimeline,
    "IAMCCS_Ltx2HelperModules_KeyframeTimeline": IAMCCS_Ltx2HelperModules_KeyframeTimeline,
    "IAMCCS_Ltx2HelperModules_RefreshPolicy": IAMCCS_Ltx2HelperModules_RefreshPolicy,
    "IAMCCS_Ltx2HelperModules_ReanchorLatent": IAMCCS_Ltx2HelperModules_ReanchorLatent,
    "IAMCCS_Ltx2HelperModules_DiskExtension": IAMCCS_Ltx2HelperModules_DiskExtension,
    "IAMCCS_Ltx2HelperModules_RuntimeBridge": IAMCCS_Ltx2HelperModules_RuntimeBridge,
    "IAMCCS_Ltx2HelperModules_Continuity": IAMCCS_Ltx2HelperModules_Continuity,
    "IAMCCS_Ltx2HelperModules_Finalize": IAMCCS_Ltx2HelperModules_Finalize,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_ProjectTimelinePlanner": "IAMCCS Project Timeline Planner",
    "IAMCCS_Ltx2HelperModules_ProjectTimelinePlanner": "IAMCCS_Ltx2HelperModules Project Timeline Planner",
    "IAMCCS_Ltx2HelperModules_Planner": "IAMCCS_Ltx2HelperModules Planner",
    "IAMCCS_Ltx2HelperModules_AudioTimeline": "IAMCCS_Ltx2HelperModules Audio Timeline",
    "IAMCCS_Ltx2HelperModules_KeyframeTimeline": "IAMCCS_Ltx2HelperModules Keyframe Timeline",
    "IAMCCS_Ltx2HelperModules_RefreshPolicy": "IAMCCS_Ltx2HelperModules Refresh Policy",
    "IAMCCS_Ltx2HelperModules_ReanchorLatent": "IAMCCS_Ltx2HelperModules Reanchor Latent",
    "IAMCCS_Ltx2HelperModules_DiskExtension": "IAMCCS_Ltx2HelperModules Disk Extension",
    "IAMCCS_Ltx2HelperModules_RuntimeBridge": "IAMCCS_Ltx2HelperModules Runtime Bridge",
    "IAMCCS_Ltx2HelperModules_Continuity": "IAMCCS_Ltx2HelperModules Continuity",
    "IAMCCS_Ltx2HelperModules_Finalize": "IAMCCS_Ltx2HelperModules Finalize",
}


IAMCCS_Ltx2HelperModules_ProjectTimelinePlanner = IAMCCS_ProjectTimelinePlanner
IAMCCS_AUIMG2VID_Planner = IAMCCS_Ltx2HelperModules_Planner
IAMCCS_AUIMG2VID_ProjectTimelinePlanner = IAMCCS_ProjectTimelinePlanner
IAMCCS_AUIMG2VID_AudioTimeline = IAMCCS_Ltx2HelperModules_AudioTimeline
IAMCCS_AUIMG2VID_KeyframeTimeline = IAMCCS_Ltx2HelperModules_KeyframeTimeline
IAMCCS_AUIMG2VID_RefreshPolicy = IAMCCS_Ltx2HelperModules_RefreshPolicy
IAMCCS_AUIMG2VID_ReanchorLatent = IAMCCS_Ltx2HelperModules_ReanchorLatent
IAMCCS_AUIMG2VID_DiskExtension = IAMCCS_Ltx2HelperModules_DiskExtension
IAMCCS_AUIMG2VID_RuntimeBridge = IAMCCS_Ltx2HelperModules_RuntimeBridge
IAMCCS_AUIMG2VID_Continuity = IAMCCS_Ltx2HelperModules_Continuity
IAMCCS_AUIMG2VID_Finalize = IAMCCS_Ltx2HelperModules_Finalize