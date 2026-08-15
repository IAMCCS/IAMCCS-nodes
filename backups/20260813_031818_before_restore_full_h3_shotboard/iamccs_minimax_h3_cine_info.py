# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later

"""MiniMax H3 reference transport for IAMCCS CineLinX.

The node deliberately keeps REF2VA reference media outside the Shotboard
timeline.  The Shotboard remains the source of prompt, duration and shot
timing, while this node publishes optional image/video/audio references and
their roles to the isolated H3 backend through the existing CineLinX cable.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import torch

from .iamccs_supernodes_linx import SUPERNODE_LINX_TYPE, build_stage_linx_payload
from .iamccs_minimax_h3_shotboard_core import build_shotplan, plan_json


CATEGORY = "IAMCCS/MiniMax H3"
STAGE_NAME = "iamccs_cine_info_h3"
RESOURCE_PREFIX = "iamccs_minimax_h3_"

IMAGE_ROLES = ["subject_identity", "keyframe", "composition", "style", "disabled"]
VIDEO_ROLES = ["off", "motion_camera", "temporal_structure", "video_edit", "continuation"]
AUDIO_ROLES = ["off", "voice_timbre", "rhythm_timing", "audio_reuse", "sound_reference"]
TASK_OVERRIDES = ["from_shotboard", "t2va", "i2va", "fl2va", "ref2va"]


def _resources(cine_linx: Any) -> dict[str, Any]:
    if not isinstance(cine_linx, dict):
        return {}
    resources = cine_linx.get("resources")
    return resources if isinstance(resources, dict) else {}


def _has_h3_plan(cine_linx: Any) -> bool:
    resources = _resources(cine_linx)
    for key in ("iamccs_minimax_h3_shotplan", "minimax_h3_shotplan", "shotplan"):
        value = resources.get(key)
        if isinstance(value, dict) and value.get("schema") == "iamccs.minimax_h3.shotplan":
            return True
    return False


def _h3_plan(cine_linx: Any) -> dict[str, Any]:
    resources = _resources(cine_linx)
    outputs = cine_linx.get("outputs", {}) if isinstance(cine_linx, dict) else {}
    payload = resources.get("cine_payload") if isinstance(resources.get("cine_payload"), dict) else {}
    for value in (
        resources.get("iamccs_minimax_h3_shotplan"),
        resources.get("minimax_h3_shotplan"),
        resources.get("shotplan"),
        outputs.get("shotplan") if isinstance(outputs, dict) else None,
        payload.get("minimax_h3_shotplan"),
        payload.get("shotplan"),
    ):
        if isinstance(value, dict) and value.get("schema") == "iamccs.minimax_h3.shotplan":
            return value
    raise ValueError("IAMCCS Cine Info H3 did not find a valid MiniMax H3 shotplan")


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    try:
        parsed = json.loads(str(value or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _routed_timeline(cine_linx: Any) -> dict[str, Any]:
    """Return only a TakeRouter-owned timeline; never infer a different take."""
    resources = _resources(cine_linx)
    for value in (
        resources.get("cine_take_router_timeline_data"),
        resources.get("cine_take_router_timeline_json"),
    ):
        timeline = _json_dict(value)
        if timeline:
            return timeline
    return {}


def _copy_runtime_contract(source: dict[str, Any], target: dict[str, Any]) -> None:
    """Preserve non-timeline H3 controls while rebuilding the selected take."""
    for key in (
        "prompter_injection",
        "performance_profile",
        "sampling",
        "turbo",
        "reference_resize",
        "upscale_settings",
        "control_contract",
        "edition",
    ):
        if key in source:
            target[key] = copy.deepcopy(source[key])

    previous_performance = source.get("performance") if isinstance(source.get("performance"), dict) else {}
    max_frames = max(
        [int(chunk.get("frame_count", 0) or 0) for chunk in target.get("chunks", []) if isinstance(chunk, dict)]
        or [0]
    )
    width = int(target.get("width", 960) or 960)
    height = int(target.get("height", 544) or 544)
    performance = copy.deepcopy(previous_performance)
    performance["max_chunk_frames"] = max_frames
    performance["relative_native_load_vs_960x544x124"] = round(
        (float(width) * float(height) * max(1, max_frames)) / (960.0 * 544.0 * 124.0),
        3,
    )
    if performance:
        target["performance"] = performance


def _pack_recompiled_plan(
    cine_linx: dict[str, Any],
    plan: dict[str, Any],
    timeline: dict[str, Any],
    report: str,
) -> dict[str, Any]:
    slots = plan.get("slots") if isinstance(plan.get("slots"), list) else []
    chunks = plan.get("chunks") if isinstance(plan.get("chunks"), list) else []
    audio_segments = timeline.get("audioSegments")
    if not isinstance(audio_segments, list):
        audio_segments = timeline.get("audio_segments")
    if not isinstance(audio_segments, list):
        audio_segments = []
    global_prompt = str(plan.get("global_prompt", "") or "")
    local_prompts = " | ".join(
        str(slot.get("prompt", "")).strip()
        for slot in slots
        if isinstance(slot, dict) and str(slot.get("prompt", "")).strip()
    )
    segment_lengths = ",".join(
        str(int(chunk.get("frame_count", 0) or 0))
        for chunk in chunks
        if isinstance(chunk, dict)
    )
    plan_text = plan_json(plan)
    prompt_map_text = json.dumps(plan.get("prompt_map", []), ensure_ascii=False, indent=2)
    timeline_text = json.dumps(timeline, ensure_ascii=False)
    previous_payload = _resources(cine_linx).get("cine_payload")
    payload = copy.deepcopy(previous_payload) if isinstance(previous_payload, dict) else {}
    payload.update({
        "backend_mode": "minimax_h3_multitimeline",
        "pipeline_kind": "minimax_h3",
        "global_prompt": global_prompt,
        "local_prompts": local_prompts,
        "segment_lengths": segment_lengths,
        "duration_seconds": float(plan.get("effective_duration_seconds", 0.0) or 0.0),
        "effective_duration_seconds": float(plan.get("effective_duration_seconds", 0.0) or 0.0),
        "frame_rate": int(plan.get("fps", 24) or 24),
        "width": int(plan.get("width", 0) or 0),
        "height": int(plan.get("height", 0) or 0),
        "timeline_data": copy.deepcopy(timeline),
        "visual_segments": copy.deepcopy(slots),
        "audioSegments": copy.deepcopy(audio_segments),
        "minimax_h3_shotplan": plan,
    })
    outputs = {
        "shotplan": plan,
        "shotplan_json": plan_text,
        "prompt_map_json": prompt_map_text,
        "total_segments": int(plan.get("total_segments", len(chunks)) or 0),
        "effective_duration": float(plan.get("effective_duration_seconds", 0.0) or 0.0),
        "global_prompt": global_prompt,
        "local_prompts": local_prompts,
        "segment_lengths": segment_lengths,
        "timeline_data": timeline_text,
        "audio_timeline_json": json.dumps({"audioSegments": audio_segments}, ensure_ascii=False),
        "report": report,
    }
    resources = {
        "cine_payload": payload,
        "cine_global_prompt": global_prompt,
        "cine_local_prompts": local_prompts,
        "cine_segment_lengths": segment_lengths,
        "cine_duration_seconds": float(plan.get("effective_duration_seconds", 0.0) or 0.0),
        "cine_frame_rate": int(plan.get("fps", 24) or 24),
        "cine_width": int(plan.get("width", 0) or 0),
        "cine_height": int(plan.get("height", 0) or 0),
        "cine_timeline_data_json": timeline_text,
        "cine_visual_segments_json": json.dumps(slots, ensure_ascii=False),
        "cine_audio_timeline_json": json.dumps({"audioSegments": audio_segments}, ensure_ascii=False),
        "iamccs_minimax_h3_shotplan": plan,
        "iamccs_minimax_h3_shotplan_json": plan_text,
        "iamccs_minimax_h3_prompt_map": copy.deepcopy(plan.get("prompt_map", [])),
        "iamccs_minimax_h3_prompt_map_json": prompt_map_text,
        "iamccs_minimax_h3_total_segments": int(plan.get("total_segments", len(chunks)) or 0),
        "iamccs_minimax_h3_effective_duration": float(plan.get("effective_duration_seconds", 0.0) or 0.0),
        "iamccs_minimax_h3_multitimeline_report": report,
    }
    out = build_stage_linx_payload(
        cine_linx,
        stage_name="MiniMax H3 MultiTimeline Recompile",
        stage_kind="minimax_h3_take_router_recompile",
        payload=payload,
        report=report,
        outputs=outputs,
        resources=resources,
        policies={
            "minimax_h3_timeline_truth": "cine_take_router_timeline_data",
            "minimax_h3_take_fallback": "forbidden",
        },
        downstream_stages=("IAMCCS Cine Info H3", "MiniMax H3 backend"),
        requires={"resources": ["cine_take_router_timeline_data", "iamccs_minimax_h3_shotplan"]},
    )
    out["mode"] = "minimax_h3_multitimeline"
    return out


def _recompile_routed_h3_plan(cine_linx: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Rebuild the H3 plan only when a strict TakeRouter timeline is present."""
    timeline = _routed_timeline(cine_linx)
    if not timeline:
        return cine_linx, "single_timeline"
    source = _h3_plan(cine_linx)
    global_prompt = str(timeline.get("global_prompt", timeline.get("prompt", source.get("global_prompt", ""))) or "")
    duration = float(
        timeline.get("duration_seconds", timeline.get("duration", source.get("requested_duration_seconds", 10.0)))
        or 10.0
    )
    rebuilt = build_shotplan(
        timeline_data=timeline,
        global_prompt=global_prompt,
        duration_seconds=duration,
        task_mode=str(source.get("task_mode", "auto_from_timeline") or "auto_from_timeline"),
        audio_mode=str(source.get("audio_mode", "h3_native_generated") or "h3_native_generated"),
        prompt_mapping=str(source.get("prompt_mapping", "global_plus_local") or "global_plus_local"),
        upscale_mode=str(source.get("upscale_mode", "off") or "off"),
        width=int(source.get("width", 960) or 960),
        height=int(source.get("height", 544) or 544),
        acceleration=str(source.get("acceleration", "native") or "native"),
        ref_image_size=str(source.get("ref_image_size", "match") or "match"),
        text_encoder_device=str(source.get("text_encoder_device", "auto") or "auto"),
        reference_roles=copy.deepcopy(source.get("reference_roles", [])),
        reference_video_role=str(source.get("reference_video_role", "off") or "off"),
        reference_audio_role=str(source.get("reference_audio_role", "off") or "off"),
        sol_conditioning=str(source.get("sol_conditioning", "exact_kv") or "exact_kv"),
        spectrum_profile=str(source.get("spectrum_profile", "conservative_3060") or "conservative_3060"),
        vram_clean_before_decode=bool(source.get("vram_clean_before_decode", True)),
        rife_mode=str(source.get("rife_mode", "off") or "off"),
        upscale_enabled=bool(source.get("upscale_enabled", False)),
    )
    _copy_runtime_contract(source, rebuilt)
    multi = timeline.get("multiGeneration") if isinstance(timeline.get("multiGeneration"), dict) else {}
    timeline_id = str(multi.get("activeTimelineId") or timeline.get("activeTimelineId") or "unknown")
    take_index = int(multi.get("activeTake") or timeline.get("activeTake") or 0)
    rebuilt["multitimeline"] = {
        "routed": True,
        "timeline_id": timeline_id,
        "take_index": take_index,
        "source": "IAMCCS_TakeRouter",
        "fallback": "forbidden",
    }
    report = (
        "MiniMax H3 MultiTimeline recompiled | "
        f"take={take_index} | timeline={timeline_id} | chunks={rebuilt.get('total_segments', 0)} | "
        f"duration={float(rebuilt.get('effective_duration_seconds', 0.0)):.3f}s | "
        "sampler/Turbo/resolution preserved"
    )
    return _pack_recompiled_plan(cine_linx, rebuilt, timeline, report), report


def _shape(value: Any) -> list[int]:
    if not torch.is_tensor(value):
        return []
    return [int(item) for item in value.shape]


def _audio_meta(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not torch.is_tensor(value.get("waveform")):
        return {"connected": False}
    waveform = value["waveform"]
    sample_rate = int(value.get("sample_rate", 32000) or 32000)
    samples = int(waveform.shape[-1]) if waveform.ndim else 0
    return {
        "connected": True,
        "shape": [int(item) for item in waveform.shape],
        "sample_rate": sample_rate,
        "duration_seconds": round(samples / max(1, sample_rate), 4),
    }


def _clean_previous_h3_info(cine_linx: dict[str, Any]) -> dict[str, Any]:
    """Remove a previous H3-info stage without copying tensor payloads."""
    cleaned = dict(cine_linx)
    resources = dict(_resources(cine_linx))
    for key in list(resources):
        if key.startswith(RESOURCE_PREFIX) and (
            key.startswith(f"{RESOURCE_PREFIX}ref_")
            or key in {
                f"{RESOURCE_PREFIX}cine_info",
                f"{RESOURCE_PREFIX}reference_manifest",
                f"{RESOURCE_PREFIX}reference_manifest_json",
            }
        ):
            resources.pop(key, None)
    cleaned["resources"] = resources
    return cleaned


class IAMCCS_CineInfoH3:
    """Attach MiniMax H3 REF2VA media to CineLinX, outside the timeline."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "task_override": (TASK_OVERRIDES, {"default": "from_shotboard"}),
                "reference_role_1": (IMAGE_ROLES, {"default": "subject_identity"}),
                "reference_role_2": (IMAGE_ROLES, {"default": "subject_identity"}),
                "reference_role_3": (IMAGE_ROLES, {"default": "composition"}),
                "reference_role_4": (IMAGE_ROLES, {"default": "style"}),
                "reference_video_role": (VIDEO_ROLES, {"default": "off"}),
                "reference_audio_role": (AUDIO_ROLES, {"default": "off"}),
                "ref_image_size": (["match", "max"], {"default": "match"}),
                "reference_resize_policy": (
                    ["canvas_crop", "canvas_pad", "total_pixels", "off"],
                    {"default": "canvas_crop"},
                ),
                "reference_resize_megapixels": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.05},
                ),
                "reference_resize_filter": (
                    ["area", "bilinear", "bicubic", "nearest-exact"],
                    {"default": "area"},
                ),
            },
            "optional": {
                "reference_image_1": ("IMAGE",),
                "reference_image_2": ("IMAGE",),
                "reference_image_3": ("IMAGE",),
                "reference_image_4": ("IMAGE",),
                "reference_video": ("IMAGE",),
                "reference_video_audio": ("AUDIO",),
                "reference_audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE,)
    RETURN_NAMES = ("cine_linx",)
    FUNCTION = "attach"
    CATEGORY = CATEGORY

    def attach(
        self,
        cine_linx,
        task_override,
        reference_role_1,
        reference_role_2,
        reference_role_3,
        reference_role_4,
        reference_video_role,
        reference_audio_role,
        ref_image_size,
        reference_resize_policy,
        reference_resize_megapixels,
        reference_resize_filter,
        reference_image_1=None,
        reference_image_2=None,
        reference_image_3=None,
        reference_image_4=None,
        reference_video=None,
        reference_video_audio=None,
        reference_audio=None,
    ):
        if not isinstance(cine_linx, dict):
            raise ValueError("IAMCCS Cine Info H3 requires a valid cine_linx input")
        if not _has_h3_plan(cine_linx):
            raise ValueError(
                "IAMCCS Cine Info H3 did not find a MiniMax H3 shotplan. "
                "Connect MiniMax H3 Shotboard to IAMCCS CineInfo, then connect its cine_linx output here."
            )
        cine_linx, timeline_mode = _recompile_routed_h3_plan(cine_linx)

        images = [reference_image_1, reference_image_2, reference_image_3, reference_image_4]
        roles = [reference_role_1, reference_role_2, reference_role_3, reference_role_4]
        image_items = []
        for index, (image, role) in enumerate(zip(images, roles), start=1):
            image_items.append({
                "slot": index,
                "label": f"<Picture {index}>",
                "role": str(role),
                "connected": bool(torch.is_tensor(image)),
                "shape": _shape(image),
            })

        manifest = {
            "schema": "iamccs.minimax_h3.cine_info",
            "schema_version": 1,
            "reference_source": "cine_info_h3_only",
            "task_override": str(task_override),
            "image_references": image_items,
            "video_reference": {
                "connected": bool(torch.is_tensor(reference_video)),
                "shape": _shape(reference_video),
                "role": str(reference_video_role),
                "audio": _audio_meta(reference_video_audio),
            },
            "audio_reference": {
                **_audio_meta(reference_audio),
                "role": str(reference_audio_role),
            },
            "ref_image_size": str(ref_image_size),
            "reference_resize": {
                "policy": str(reference_resize_policy),
                "megapixels": float(reference_resize_megapixels),
                "filter": str(reference_resize_filter),
                "multiple_of": 32,
                "downscale_only": True,
            },
        }
        active_images = sum(1 for item in image_items if item["connected"] and item["role"] != "disabled")
        active_video = bool(torch.is_tensor(reference_video) and str(reference_video_role) != "off")
        active_audio = bool(
            (isinstance(reference_audio, dict) and str(reference_audio_role) != "off")
            or isinstance(reference_video_audio, dict)
        )
        manifest["active_reference_count"] = active_images + int(active_video) + int(active_audio)
        manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2)
        report = (
            "IAMCCS Cine Info H3 | references outside Shotboard timeline | "
            f"task={task_override} | images={active_images}/4 | video={'on' if active_video else 'off'} | "
            f"audio={'on' if active_audio else 'off'} | ref_size={ref_image_size} | "
            f"resize={reference_resize_policy}:{float(reference_resize_megapixels):.2f}MP/{reference_resize_filter} | "
            f"timeline={timeline_mode}"
        )

        config = {
            "schema": manifest["schema"],
            "schema_version": manifest["schema_version"],
            "reference_source": manifest["reference_source"],
            "task_override": str(task_override),
            "reference_roles": [str(item) for item in roles],
            "reference_video_role": str(reference_video_role),
            "reference_audio_role": str(reference_audio_role),
            "ref_image_size": str(ref_image_size),
            "reference_resize": dict(manifest["reference_resize"]),
        }
        base = _clean_previous_h3_info(cine_linx)
        resources = {
            f"{RESOURCE_PREFIX}cine_info": config,
            f"{RESOURCE_PREFIX}reference_manifest": manifest,
            f"{RESOURCE_PREFIX}reference_manifest_json": manifest_json,
            f"{RESOURCE_PREFIX}ref_image_1": reference_image_1,
            f"{RESOURCE_PREFIX}ref_image_2": reference_image_2,
            f"{RESOURCE_PREFIX}ref_image_3": reference_image_3,
            f"{RESOURCE_PREFIX}ref_image_4": reference_image_4,
            f"{RESOURCE_PREFIX}ref_video": reference_video,
            f"{RESOURCE_PREFIX}ref_video_audio": reference_video_audio,
            f"{RESOURCE_PREFIX}ref_audio": reference_audio,
        }
        out_linx = build_stage_linx_payload(
            base,
            stage_name=STAGE_NAME,
            stage_kind="minimax_h3_reference_transport",
            payload=config,
            report=report,
            slot_map={"cine_linx": "MiniMax H3 backend cine_linx"},
            downstream_stages=("IAMCCS MiniMax H3 Atomic Model Router", "IAMCCS MiniMax H3 Atomic Conditioning"),
            policies={
                "reference_media_location": "cine_info_h3_not_shotboard_timeline",
                "shotboard_owns": "prompt_duration_timeline",
                "reference_precedence": "explicit_backend_socket_then_cine_info_h3",
            },
            outputs={"minimax_h3_reference_manifest_json": manifest_json, "report": report},
            resources=resources,
            requires={"resources": ["iamccs_minimax_h3_shotplan"]},
        )
        return (out_linx,)


NODE_CLASS_MAPPINGS = {
    "IAMCCS_CineInfoH3": IAMCCS_CineInfoH3,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_CineInfoH3": "IAMCCS Cine Info H3 - REF2VA Inputs",
}
