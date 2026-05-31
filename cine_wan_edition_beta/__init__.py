from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..iamccs_cine_nodes import (
    MAX_CINE_ITEMS,
    SUPERNODE_LINX_TYPE,
    IAMCCS_CineReferenceBoard,
    IAMCCS_CineShotboardPlannerV3,
    _cine_debug,
    _clamp,
    _iamccs_cine_resize_method,
    _json_report,
    _load_original_promptrelay_module,
    _safe_bool,
    _safe_float,
    _safe_int,
)


WAN_BETA_MODE = "iamccs_wan_shotboard_v3_beta"
WAN_BETA_SCHEMA = "iamccs.wan.shotboard_v3.beta"


def _resources(cine_linx: Any) -> Dict[str, Any]:
    if not isinstance(cine_linx, dict):
        return {}
    resources = cine_linx.get("resources")
    return resources if isinstance(resources, dict) else {}


def _outputs(cine_linx: Any) -> Dict[str, Any]:
    if not isinstance(cine_linx, dict):
        return {}
    outputs = cine_linx.get("outputs")
    return outputs if isinstance(outputs, dict) else {}


def _payload(cine_linx: Any) -> Dict[str, Any]:
    resources = _resources(cine_linx)
    payload = resources.get("cine_payload")
    if isinstance(payload, dict):
        return payload
    if isinstance(cine_linx, dict):
        for stage in cine_linx.get("stages") or []:
            if isinstance(stage, dict) and isinstance(stage.get("payload"), dict):
                return stage["payload"]
    return {}


def _split_prompt_parts(text: Any) -> List[str]:
    return [part.strip() for part in str(text or "").split("|") if part.strip()]


def _split_length_parts(text: Any) -> List[str]:
    return [part for part in re.split(r"[,;\s]+", str(text or "")) if part.strip()]


def _text_hash(text: Any) -> str:
    return hashlib.sha1(str(text or "").encode("utf-8", errors="ignore")).hexdigest()[:12]


def _reference_paths(image_paths: str) -> List[str]:
    paths: List[str] = []
    for raw in str(image_paths or "").replace(";", "\n").splitlines():
        item = raw.strip().strip('"')
        if item:
            paths.append(item)
    return paths


class IAMCCS_CineShotboardPlannerV3WANEdition_BETA(IAMCCS_CineShotboardPlannerV3):
    """BETA WAN edition of Shotboard V3.

    It does not execute a WAN model directly. It compiles a WAN-native cine_linx
    plan with FLF/SVI metadata plus true PromptRelay inputs for the dedicated
    IAMCCS_WanPromptRelayBridge_BETA node.
    """

    CATEGORY = "IAMCCS/Wan/BETA"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "global_prompt": ("STRING", {
                    "default": "cinematic WAN image-to-video shot, stable identity, coherent physical motion, detailed scene continuity",
                    "multiline": True,
                }),
                "timeline_data": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "BETA. Use Shotboard V3-style JSON: segments with start/length/ref/prompt/use_prompt/use_guide.",
                }),
                "frame_rate": ("INT", {"default": 16, "min": 1, "max": 120, "step": 1}),
                "wan_chunk_frames": ("INT", {"default": 81, "min": 1, "max": 120, "step": 1}),
                "duration_seconds": ("FLOAT", {"default": 5.0625, "min": 0.01, "max": 36000.0, "step": 0.01}),
                "frame_budget_mode": (["wan_chunk_frames", "duration_seconds"], {"default": "wan_chunk_frames"}),
                "promptrelay_epsilon": ("FLOAT", {"default": 0.001, "min": 0.000001, "max": 0.99, "step": 0.0001}),
                "guide_policy": (["every_checked_row", "safe_core_guides", "prompt_only"], {"default": "every_checked_row"}),
                "min_guide_gap_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.05}),
                "max_guides": ("INT", {"default": 8, "min": 0, "max": 50, "step": 1}),
                "default_force": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "wan_mode": (["flf_svi_promptrelay", "flf_only", "promptrelay_only", "svi_continuation"], {"default": "flf_svi_promptrelay"}),
                "flf_role_policy": (["start_end", "every_image_anchor", "manual_metadata"], {"default": "start_end"}),
                "image_paths": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional image paths, one per line. Used as WAN visual anchors and reference metadata.",
                }),
                "image_width": ("INT", {"default": 1280, "min": 64, "max": 8192, "step": 32}),
                "image_height": ("INT", {"default": 736, "min": 64, "max": 8192, "step": 32}),
                "image_resize_method": (["crop", "pad", "keep proportion", "stretch", ""], {"default": "crop"}),
                "image_multiple_of": ("INT", {"default": 32, "min": 1, "max": 512, "step": 1}),
                "img_compression": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "multi_input": ("IMAGE",),
            },
        }

    RETURN_TYPES = (
        SUPERNODE_LINX_TYPE,
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "INT",
        "FLOAT",
        "STRING",
        "IMAGE",
        "IMAGE",
    )
    RETURN_NAMES = (
        "cine_linx",
        "wan_flf_plan_json",
        "global_prompt",
        "local_prompts",
        "segment_lengths",
        "max_frames",
        "promptrelay_epsilon",
        "report",
        "multi_output",
        "image_1",
    )
    FUNCTION = "execute"

    @classmethod
    def IS_CHANGED(
        cls,
        global_prompt=None,
        timeline_data=None,
        frame_rate=None,
        wan_chunk_frames=None,
        duration_seconds=None,
        frame_budget_mode=None,
        promptrelay_epsilon=None,
        guide_policy=None,
        min_guide_gap_seconds=None,
        max_guides=None,
        default_force=None,
        wan_mode=None,
        flf_role_policy=None,
        image_paths=None,
        image_width=None,
        image_height=None,
        image_resize_method=None,
        image_multiple_of=None,
        img_compression=None,
        cine_linx=None,
        multi_input=None,
        **kwargs,
    ):
        return _text_hash(json.dumps({
            "node": cls.__name__,
            "global_prompt": str(global_prompt or ""),
            "timeline_data": str(timeline_data or ""),
            "frame_rate": int(frame_rate or 0),
            "wan_chunk_frames": int(wan_chunk_frames or 0),
            "duration_seconds": float(duration_seconds or 0.0),
            "frame_budget_mode": str(frame_budget_mode or ""),
            "promptrelay_epsilon": float(promptrelay_epsilon or 0.0),
            "guide_policy": str(guide_policy or ""),
            "min_guide_gap_seconds": float(min_guide_gap_seconds or 0.0),
            "max_guides": int(max_guides or 0),
            "default_force": float(default_force or 0.0),
            "wan_mode": str(wan_mode or ""),
            "flf_role_policy": str(flf_role_policy or ""),
            "image_paths": str(image_paths or ""),
            "image_width": int(image_width or 0),
            "image_height": int(image_height or 0),
            "image_resize_method": str(image_resize_method or ""),
            "image_multiple_of": int(image_multiple_of or 0),
            "img_compression": int(img_compression or 0),
        }, sort_keys=True, ensure_ascii=False))

    @classmethod
    def _effective_frame_budget(cls, duration_seconds: float, fps: int, wan_chunk_frames: int, mode: str) -> Tuple[int, float]:
        fps = max(1, int(fps))
        if str(mode or "wan_chunk_frames") == "duration_seconds":
            frames = max(1, int(round(float(duration_seconds) * fps)))
            frames = min(120, frames)
        else:
            frames = max(1, min(120, int(wan_chunk_frames or 81)))
        return int(frames), float(frames) / float(fps)

    @classmethod
    def _rows_from_v3_timeline(cls, timeline_data: str, duration_seconds: float, fps: int, default_force: float) -> List[Dict[str, Any]]:
        rows = cls._parse_rows(str(timeline_data or ""), float(duration_seconds), float(default_force))
        for idx, row in enumerate(rows):
            if row.get("frame") is None:
                row["frame"] = max(0, int(round(float(row.get("second", 0.0)) * fps)))
            row["frame"] = max(0, int(row.get("frame") or 0))
            row["ref"] = max(1, min(MAX_CINE_ITEMS, _safe_int(row.get("ref", idx + 1), idx + 1)))
            row["force"] = _clamp(row.get("force", default_force), 0.0, 1.0, default_force)
            row["guide_strength"] = _clamp(row.get("guide_strength", row.get("force", default_force)), 0.0, 1.0, default_force)
        return rows

    @classmethod
    def _compile_wan_relay(cls, rows: List[Dict[str, Any]], frame_budget: int) -> Tuple[str, str, List[int]]:
        frame_budget = max(1, int(frame_budget))
        candidates: List[Tuple[int, str, Dict[str, Any]]] = []
        for row in rows:
            if not bool(row.get("use_prompt", False)):
                continue
            prompt = cls._row_prompt(row)
            if not str(prompt or "").strip():
                continue
            start = max(0, min(frame_budget - 1, int(row.get("frame") or 0)))
            candidates.append((start, str(prompt).strip(), row))
        candidates.sort(key=lambda item: item[0])
        if not candidates:
            return "", "", []

        prompts: List[str] = []
        lengths: List[int] = []
        for idx, (start, prompt, _row) in enumerate(candidates):
            next_start = candidates[idx + 1][0] if idx + 1 < len(candidates) else frame_budget
            length = max(1, int(next_start) - int(start))
            prompts.append(prompt)
            lengths.append(length)
        diff = frame_budget - sum(lengths)
        if lengths:
            lengths[-1] = max(1, lengths[-1] + diff)
        return " | ".join(prompts), ",".join(str(int(v)) for v in lengths), lengths

    @classmethod
    def _visual_segments(
        cls,
        rows: List[Dict[str, Any]],
        frame_budget: int,
        fps: int,
        reference_paths: List[str],
        default_force: float,
        flf_role_policy: str,
    ) -> List[Dict[str, Any]]:
        frame_budget = max(1, int(frame_budget))
        visual_rows = [row for row in rows if int(row.get("frame") or 0) < frame_budget]
        visual_rows.sort(key=lambda row: int(row.get("frame") or 0))
        segments: List[Dict[str, Any]] = []
        for idx, row in enumerate(visual_rows):
            start = max(0, min(frame_budget - 1, int(row.get("frame") or 0)))
            next_start = int(visual_rows[idx + 1].get("frame") or frame_budget) if idx + 1 < len(visual_rows) else frame_budget
            length = max(1, min(frame_budget, next_start) - start)
            ref = max(1, _safe_int(row.get("ref", idx + 1), idx + 1))
            if str(flf_role_policy or "") == "every_image_anchor":
                role = "flf_anchor"
            elif str(flf_role_policy or "") == "manual_metadata":
                role = str(row.get("wan_role", row.get("role", "visual_anchor")) or "visual_anchor")
            else:
                role = "start_frame" if idx == 0 else "end_frame" if idx == len(visual_rows) - 1 else "bridge_frame"
            segments.append({
                "id": f"wan_beta_seg_{idx + 1:02d}",
                "schema": WAN_BETA_SCHEMA,
                "type": "image",
                "start": int(start),
                "length": int(length),
                "second": float(start) / float(max(1, fps)),
                "ref": int(ref),
                "imageFile": reference_paths[ref - 1] if ref <= len(reference_paths) else "",
                "label": str(row.get("label", f"wan_ref_{idx + 1}") or f"wan_ref_{idx + 1}"),
                "prompt": str(row.get("relay_prompt", "")),
                "camera": str(row.get("camera", "")),
                "transition": str(row.get("transition", "continuous_motion")),
                "wan_role": role,
                "use_flf": bool(row.get("use_guide", True)),
                "use_svi": True,
                "use_promptrelay": bool(row.get("use_prompt", False)),
                "guideStrength": float(_safe_float(row.get("guide_strength", row.get("force", default_force)), default_force)),
                "motion_force": float(_safe_float(row.get("motion_force", row.get("force", default_force)), default_force)),
            })
        return segments

    @staticmethod
    def _multi_from_linx(cine_linx: Any) -> Any:
        resources = _resources(cine_linx)
        for key in ("cine_multi_input", "multi_input"):
            value = resources.get(key)
            if torch.is_tensor(value):
                return value
        return None

    def execute(
        self,
        global_prompt,
        timeline_data,
        frame_rate,
        wan_chunk_frames,
        duration_seconds,
        frame_budget_mode,
        promptrelay_epsilon,
        guide_policy,
        min_guide_gap_seconds,
        max_guides,
        default_force,
        wan_mode,
        flf_role_policy,
        image_paths,
        image_width,
        image_height,
        image_resize_method,
        image_multiple_of,
        img_compression,
        cine_linx=None,
        multi_input=None,
    ):
        upstream_resources = _resources(cine_linx)
        if upstream_resources:
            upstream_timeline = upstream_resources.get("cine_board_timeline_data")
            if isinstance(upstream_timeline, str) and upstream_timeline.strip():
                timeline_data = upstream_timeline
            upstream_prompt = upstream_resources.get("cine_global_prompt")
            if isinstance(upstream_prompt, str) and upstream_prompt.strip() and not str(global_prompt or "").strip():
                global_prompt = upstream_prompt
            multi_input = multi_input if torch.is_tensor(multi_input) else self._multi_from_linx(cine_linx)

        fps = max(1, int(frame_rate or 16))
        frame_budget, effective_duration = self._effective_frame_budget(
            float(duration_seconds),
            fps,
            int(wan_chunk_frames or 81),
            str(frame_budget_mode or "wan_chunk_frames"),
        )
        image_resize_method = _iamccs_cine_resize_method(image_resize_method)
        if str(image_paths or "").strip():
            try:
                multi_input = IAMCCS_CineReferenceBoard().load_ltx_style_images(
                    image_paths,
                    int(image_width),
                    int(image_height),
                    image_resize_method,
                    int(image_multiple_of or 32),
                    int(img_compression or 0),
                )
            except Exception as exc:
                print(f"[IAMCCS WAN BETA] could not load image_paths: {exc}")

        if torch.is_tensor(multi_input):
            multi_output = multi_input
        else:
            multi_output = torch.zeros((1, 64, 64, 3))
        image_1 = multi_output[0:1] if torch.is_tensor(multi_output) and multi_output.shape[0] > 0 else torch.zeros((1, 64, 64, 3))

        rows = self._rows_from_v3_timeline(str(timeline_data or ""), effective_duration, fps, float(default_force))
        local_prompts, segment_lengths, lengths = self._compile_wan_relay(rows, frame_budget)
        promptrelay_enabled = bool(local_prompts.strip()) and str(wan_mode) in {"flf_svi_promptrelay", "promptrelay_only"}
        if not promptrelay_enabled:
            local_prompts = ""
            segment_lengths = ""
            lengths = []

        reference_paths = _reference_paths(str(image_paths or ""))
        visual_segments = self._visual_segments(rows, frame_budget, fps, reference_paths, float(default_force), str(flf_role_policy))
        guide_rows = self._select_guides(rows, str(guide_policy), float(min_guide_gap_seconds), int(max_guides))
        flf_timeline = self._flf_from_rows(guide_rows, effective_duration, fps, int(multi_output.shape[0]) if torch.is_tensor(multi_output) else 0)

        wan_flf_plan = {
            "schema": WAN_BETA_SCHEMA,
            "version": 1,
            "beta": True,
            "mode": str(wan_mode),
            "frame_budget": int(frame_budget),
            "fps": int(fps),
            "duration_seconds": float(effective_duration),
            "flf_role_policy": str(flf_role_policy),
            "visual_segments": visual_segments,
            "flf_timeline": flf_timeline,
            "promptrelay": {
                "enabled": bool(promptrelay_enabled),
                "backend": "IAMCCS_WanPromptRelayBridge_BETA",
                "global_prompt": str(global_prompt or ""),
                "local_prompts": local_prompts,
                "segment_lengths": segment_lengths,
                "pixel_lengths": lengths,
                "epsilon": float(promptrelay_epsilon),
            },
            "svi": {
                "enabled": str(wan_mode) in {"flf_svi_promptrelay", "flf_only", "svi_continuation"},
                "recommended_motion_node": "WanImageMotionProPlus",
                "recommended_bridge_node": "IAMCCS_WanSVIToFLFBridgeProPlus",
            },
        }
        wan_flf_plan_json = json.dumps(wan_flf_plan, ensure_ascii=False)
        visual_segments_json = json.dumps(visual_segments, ensure_ascii=False)

        payload = {
            "schema": WAN_BETA_SCHEMA,
            "beta": True,
            "backend_mode": WAN_BETA_MODE,
            "pipeline_kind": "wan_i2v_flf_svi_promptrelay_beta",
            "global_prompt": str(global_prompt or ""),
            "timeline_data": str(timeline_data or ""),
            "duration_seconds": float(effective_duration),
            "frame_rate": int(fps),
            "wan_chunk_frames": int(frame_budget),
            "max_frames": int(frame_budget),
            "wan_mode": str(wan_mode),
            "flf_role_policy": str(flf_role_policy),
            "promptrelay_enabled": bool(promptrelay_enabled),
            "promptrelay_backend": "IAMCCS_WanPromptRelayBridge_BETA",
            "promptrelay_epsilon": float(promptrelay_epsilon),
            "local_prompts": local_prompts,
            "segment_lengths": segment_lengths,
            "promptrelay_pixel_lengths": lengths,
            "rows": rows,
            "guide_rows": guide_rows,
            "visual_segments": visual_segments,
            "wan_flf_plan": wan_flf_plan,
        }
        resources = {
            "cine_payload": payload,
            "cine_global_prompt": str(global_prompt or ""),
            "cine_local_prompts": local_prompts,
            "cine_segment_lengths": segment_lengths,
            "cine_promptrelay_enabled": bool(promptrelay_enabled),
            "cine_promptrelay_epsilon": float(promptrelay_epsilon),
            "cine_max_frames": int(frame_budget),
            "cine_duration_seconds": float(effective_duration),
            "cine_frame_rate": int(fps),
            "cine_image_paths": str(image_paths or ""),
            "cine_image_width": int(image_width),
            "cine_image_height": int(image_height),
            "cine_multi_input": multi_output,
            "cine_image_1": image_1,
            "cine_flf_timeline": flf_timeline,
            "cine_visual_segments_json": visual_segments_json,
            "wan_beta_enabled": True,
            "wan_beta_schema": WAN_BETA_SCHEMA,
            "wan_global_prompt": str(global_prompt or ""),
            "wan_local_prompts": local_prompts,
            "wan_segment_lengths": segment_lengths,
            "wan_promptrelay_enabled": bool(promptrelay_enabled),
            "wan_promptrelay_epsilon": float(promptrelay_epsilon),
            "wan_max_frames": int(frame_budget),
            "wan_chunk_frames": int(frame_budget),
            "wan_frame_rate": int(fps),
            "wan_duration_seconds": float(effective_duration),
            "wan_flf_plan_json": wan_flf_plan_json,
            "wan_visual_segments_json": visual_segments_json,
        }
        report = _json_report({
            "node": "IAMCCS_CineShotboardPlannerV3WANEdition_BETA",
            "beta": True,
            "mode": WAN_BETA_MODE,
            "frame_budget": int(frame_budget),
            "fps": int(fps),
            "duration_seconds": float(effective_duration),
            "promptrelay_enabled": bool(promptrelay_enabled),
            "local_prompt_count": len(_split_prompt_parts(local_prompts)),
            "segment_lengths": segment_lengths,
            "visual_segments": len(visual_segments),
            "guide_rows": len(guide_rows),
            "truth": "BETA WAN edition. It prepares WAN FLF/SVI metadata and true PromptRelay inputs, but the model is patched only by IAMCCS_WanPromptRelayBridge_BETA.",
        })
        resources["cine_report"] = report
        resources["wan_report"] = report

        cine_out = {
            "type": SUPERNODE_LINX_TYPE,
            "pipeline_kind": "wan_i2v_flf_svi_promptrelay_beta",
            "mode": WAN_BETA_MODE,
            "beta": True,
            "chain": [{"role": "planner", "name": "IAMCCS Shotboard V3 WAN Edition BETA"}],
            "stages": [{"name": "WAN_BETA", "kind": "wan_shotboard_v3_beta", "payload": payload}],
            "policies": {
                "promptrelay_backend": "IAMCCS_WanPromptRelayBridge_BETA",
                "promptrelay_model_family": "wan",
                "commit_safety": "optional_module; safe to exclude folder from commit when __init__ optional loader remains",
            },
            "outputs": {
                "global_prompt": str(global_prompt or ""),
                "local_prompts": local_prompts,
                "segment_lengths": segment_lengths,
                "max_frames": int(frame_budget),
                "promptrelay_epsilon": float(promptrelay_epsilon),
                "duration_seconds": float(effective_duration),
                "frame_rate": int(fps),
                "promptrelay_enabled": bool(promptrelay_enabled),
                "wan_flf_plan_json": wan_flf_plan_json,
                "report": report,
            },
            "resources": resources,
            "resource_keys": sorted(resources.keys()),
            "resource_types": {key: type(value).__name__ for key, value in resources.items()},
        }

        print(
            "[IAMCCS WAN BETA] "
            f"planner frame_budget={frame_budget} fps={fps} relay={bool(promptrelay_enabled)} "
            f"locals={len(_split_prompt_parts(local_prompts))} visual_segments={len(visual_segments)}"
        )
        _cine_debug(
            "[IAMCCS WAN BETA] "
            f"global_hash={_text_hash(global_prompt)} local_hash={_text_hash(local_prompts)} "
            f"segment_lengths={segment_lengths or '<empty>'}"
        )
        return (
            cine_out,
            wan_flf_plan_json,
            str(global_prompt or ""),
            local_prompts,
            segment_lengths,
            int(frame_budget),
            float(promptrelay_epsilon),
            report,
            multi_output,
            image_1,
        )


class IAMCCS_WanPromptRelayBridge_BETA:
    """BETA true PromptRelay bridge for WAN models.

    The bridge is intentionally separate from the LTX Shotboard backend. It reads
    WAN BETA cine_linx metadata and calls ComfyUI-PromptRelay's original
    _encode_relay against the connected WAN model, clip and latent.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
            },
            "optional": {
                "global_prompt": ("STRING", {"forceInput": True}),
                "local_prompts": ("STRING", {"forceInput": True}),
                "segment_lengths": ("STRING", {"forceInput": True}),
                "epsilon": ("FLOAT", {"default": 0.001, "min": 0.000001, "max": 0.99, "step": 0.0001, "forceInput": True}),
                "clip": ("CLIP", {"lazy": True}),
                "latent": ("LATENT", {"lazy": True}),
                "relay_options": ("RELAY_OPTIONS",),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "BOOLEAN", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("model", "positive", "promptrelay_enabled", "local_prompts", "segment_lengths", "max_frames", "report")
    FUNCTION = "execute"
    CATEGORY = "IAMCCS/Wan/BETA"

    @classmethod
    def _relay_values(
        cls,
        cine_linx: Any,
        global_prompt: Optional[str] = None,
        local_prompts: Optional[str] = None,
        segment_lengths: Optional[str] = None,
        epsilon: Optional[float] = None,
    ) -> Tuple[bool, str, str, str, float, int, str]:
        resources = _resources(cine_linx)
        outputs = _outputs(cine_linx)
        payload = _payload(cine_linx)
        resolved_global = str(
            global_prompt
            if global_prompt is not None else
            resources.get("wan_global_prompt",
            resources.get("cine_global_prompt",
            outputs.get("global_prompt",
            payload.get("global_prompt", ""))))
            or ""
        )
        resolved_local = str(
            local_prompts
            if local_prompts is not None else
            resources.get("wan_local_prompts",
            resources.get("cine_local_prompts",
            outputs.get("local_prompts",
            payload.get("local_prompts", ""))))
            or ""
        )
        resolved_lengths = str(
            segment_lengths
            if segment_lengths is not None else
            resources.get("wan_segment_lengths",
            resources.get("cine_segment_lengths",
            outputs.get("segment_lengths",
            payload.get("segment_lengths", ""))))
            or ""
        )
        resolved_epsilon = _safe_float(
            epsilon
            if epsilon is not None else
            resources.get("wan_promptrelay_epsilon",
            resources.get("cine_promptrelay_epsilon",
            outputs.get("promptrelay_epsilon",
            payload.get("promptrelay_epsilon", 0.001)))),
            0.001,
        )
        max_frames = _safe_int(
            resources.get("wan_max_frames",
            resources.get("cine_max_frames",
            outputs.get("max_frames",
            payload.get("max_frames", 0)))),
            0,
        )
        active = bool(_split_prompt_parts(resolved_local))
        source = "explicit_inputs" if local_prompts is not None else "wan_cine_linx"
        return active, resolved_local, resolved_lengths, resolved_global, float(resolved_epsilon), int(max_frames), source

    def check_lazy_status(
        self,
        cine_linx,
        model,
        positive,
        global_prompt=None,
        local_prompts=None,
        segment_lengths=None,
        epsilon=None,
        clip=None,
        latent=None,
        relay_options=None,
    ):
        active, *_ = self._relay_values(cine_linx, global_prompt, local_prompts, segment_lengths, epsilon)
        needed = []
        if active:
            if clip is None:
                needed.append("clip")
            if latent is None:
                needed.append("latent")
        return needed

    def execute(
        self,
        cine_linx,
        model,
        positive,
        global_prompt=None,
        local_prompts=None,
        segment_lengths=None,
        epsilon=None,
        clip=None,
        latent=None,
        relay_options=None,
    ):
        active, local_prompts, segment_lengths, global_prompt, epsilon, max_frames, source = self._relay_values(
            cine_linx,
            global_prompt,
            local_prompts,
            segment_lengths,
            epsilon,
        )
        local_parts = _split_prompt_parts(local_prompts)
        length_parts = _split_length_parts(segment_lengths)
        relay_prompt_log = [
            {
                "index": idx,
                "segment_length": length_parts[idx] if idx < len(length_parts) else "<auto>",
                "prompt": prompt,
            }
            for idx, prompt in enumerate(local_parts)
        ]
        if not active:
            report = _json_report({
                "node": "IAMCCS_WanPromptRelayBridge_BETA",
                "beta": True,
                "promptrelay_enabled": False,
                "mode": "BYPASS_PASS_THROUGH",
                "source": source,
                "truth": "No WAN local prompts were found. The WAN model and positive conditioning are returned unchanged.",
            })
            return model, positive, False, "", "", int(max_frames), report

        if clip is None or latent is None:
            report = _json_report({
                "node": "IAMCCS_WanPromptRelayBridge_BETA",
                "beta": True,
                "promptrelay_enabled": False,
                "mode": "BYPASS_MISSING_INPUTS",
                "warning": "Relay is active but clip or latent is missing.",
                "local_prompt_count": len(local_parts),
                "truth": "Connect WAN clip and WAN latent to enable true PromptRelay patching.",
            })
            print("[IAMCCS WAN BETA] PromptRelay bypass: missing clip or latent")
            return model, positive, False, local_prompts, segment_lengths, int(max_frames), report

        try:
            promptrelay_nodes = _load_original_promptrelay_module()
            patched_model, conditioning = promptrelay_nodes._encode_relay(
                model,
                clip,
                latent,
                global_prompt,
                local_prompts,
                segment_lengths,
                float(epsilon),
                relay_options,
            )
        except Exception as exc:
            report = _json_report({
                "node": "IAMCCS_WanPromptRelayBridge_BETA",
                "beta": True,
                "promptrelay_enabled": False,
                "mode": "BYPASS_RELAY_ERROR",
                "error": str(exc),
                "local_prompt_count": len(local_parts),
                "global_hash": _text_hash(global_prompt),
                "local_hash": _text_hash(local_prompts),
                "truth": "The original PromptRelay _encode_relay raised an error, so the bridge returned the unpatched WAN model.",
            })
            print(f"[IAMCCS WAN BETA] PromptRelay error, bypassing: {exc}")
            return model, positive, False, local_prompts, segment_lengths, int(max_frames), report

        report = _json_report({
            "node": "IAMCCS_WanPromptRelayBridge_BETA",
            "beta": True,
            "promptrelay_enabled": True,
            "mode": "WAN_PROMPT_RELAY_ORIGINAL_1TO1",
            "source": source,
            "local_prompt_count": len(local_parts),
            "segment_count": len(length_parts),
            "max_frames": int(max_frames),
            "epsilon": float(epsilon),
            "global_hash": _text_hash(global_prompt),
            "local_hash": _text_hash(local_prompts),
            "local_prompts_used": relay_prompt_log,
            "truth": "True WAN PromptRelay path. Called ComfyUI-PromptRelay _encode_relay against the connected model, clip and latent.",
        })
        print(
            "[IAMCCS WAN BETA] "
            f"PromptRelay APPLIED locals={len(local_parts)} segments={len(length_parts)} "
            f"max_frames={int(max_frames)} global_hash={_text_hash(global_prompt)} local_hash={_text_hash(local_prompts)}"
        )
        for item in relay_prompt_log[:30]:
            compact = str(item["prompt"]).replace("\n", " ")
            if len(compact) > 300:
                compact = compact[:297] + "..."
            print(f"[IAMCCS WAN BETA] relay[{int(item['index']):02d}] length={item['segment_length']} prompt={compact!r}")
        return patched_model, conditioning, True, local_prompts, segment_lengths, int(max_frames), report


NODE_CLASS_MAPPINGS = {
    "IAMCCS_CineShotboardPlannerV3WANEdition_BETA": IAMCCS_CineShotboardPlannerV3WANEdition_BETA,
    "IAMCCS_WanPromptRelayBridge_BETA": IAMCCS_WanPromptRelayBridge_BETA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_CineShotboardPlannerV3WANEdition_BETA": "IAMCCS Cine Shotboard Planner V3 WAN Edition BETA",
    "IAMCCS_WanPromptRelayBridge_BETA": "IAMCCS WAN PromptRelay Bridge BETA",
}
