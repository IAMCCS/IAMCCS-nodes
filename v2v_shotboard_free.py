from __future__ import annotations

import json
import math
from typing import Any

from .engine_v2v.cine_v2v_node import SUPERNODE_LINX_TYPE


def _float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _int(value: Any, fallback: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(fallback)


def _json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(str(value or "{}"))
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


class IAMCCS_V2VShotboardFreeScail:
    """Easy full-screen control shell for existing SCAIL-2 graphs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_video_path": ("STRING", {"default": ""}),
                "source_image_path": ("STRING", {"default": ""}),
                "duration_seconds": ("FLOAT", {"default": 5.0625, "min": 0.01, "max": 3600.0, "step": 0.01}),
                "fps": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "trim_start_s": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.01}),
                "trim_end_s": ("FLOAT", {"default": 5.0625, "min": 0.01, "max": 3600.0, "step": 0.01}),
                "frame_load_cap": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 1}),
                "generation_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "generation_height": ("INT", {"default": 896, "min": 64, "max": 4096, "step": 8}),
                "scail_identity_mode": (["single_person", "multi_person_identity"], {"default": "single_person"}),
                "show_reference_background": ("BOOLEAN", {"default": True}),
                "interpolate_to_32fps": ("BOOLEAN", {"default": True}),
                "chunk_frames": ("INT", {"default": 81, "min": 9, "max": 1001, "step": 4}),
                "overlap_frames": ("INT", {"default": 5, "min": 1, "max": 997, "step": 4}),
                "generation_steps": ("INT", {"default": 6, "min": 1, "max": 100, "step": 1}),
                "generation_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.05}),
                "generation_seed": ("INT", {"default": 123, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "lora_name": ("STRING", {"default": "auto / graph"}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "global_prompt": ("STRING", {
                    "default": "preserve the source performance, stable identity, coherent motion, cinematic natural detail",
                    "multiline": True,
                }),
                "negative_prompt": ("STRING", {
                    "default": "identity drift, duplicated person, malformed anatomy, flicker, subtitles, text",
                    "multiline": True,
                }),
                "output_prefix": ("STRING", {"default": "IAMCCS/SCAIL2_EASY_SHOTBOARD"}),
                "timeline_data": ("STRING", {"default": "", "multiline": True}),
                "preview_stage": (["source", "reference", "pose", "mask", "intermediate", "output"], {"default": "source"}),
            }
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE, "STRING")
    RETURN_NAMES = ("cine_linx", "report")
    FUNCTION = "plan"
    CATEGORY = "IAMCCS/V2V/Easy"
    DESCRIPTION = "Full-screen SCAIL-2 Single/Multi control shell for existing backend nodes in the same workflow."

    def plan(
        self,
        source_video_path: str,
        source_image_path: str,
        duration_seconds: float,
        fps: float,
        trim_start_s: float,
        trim_end_s: float,
        frame_load_cap: int,
        generation_width: int,
        generation_height: int,
        scail_identity_mode: str,
        show_reference_background: bool,
        interpolate_to_32fps: bool,
        chunk_frames: int,
        overlap_frames: int,
        generation_steps: int,
        generation_cfg: float,
        generation_seed: int,
        lora_name: str,
        lora_strength: float,
        global_prompt: str,
        negative_prompt: str,
        output_prefix: str,
        timeline_data: str,
        preview_stage: str,
    ):
        duration = max(0.01, _float(duration_seconds, 5.0625))
        fps_value = max(1.0, _float(fps, 16.0))
        trim_start = min(duration, max(0.0, _float(trim_start_s, 0.0)))
        trim_end = min(duration, max(trim_start + 0.01, _float(trim_end_s, duration)))
        selected_duration = max(0.01, trim_end - trim_start)
        width = max(64, _int(generation_width, 512))
        height = max(64, _int(generation_height, 896))
        cap = max(1, _int(frame_load_cap, round(selected_duration * fps_value)))
        chunk = max(9, _int(chunk_frames, 81))
        chunk = ((chunk - 1) // 4) * 4 + 1
        overlap = max(1, min(chunk - 4, _int(overlap_frames, 5)))
        overlap = ((overlap - 1) // 4) * 4 + 1
        identity_mode = "multi_person_identity" if str(scail_identity_mode) == "multi_person_identity" else "single_person"
        profile = "scail2_multi_person_identity" if identity_mode == "multi_person_identity" else "scail2_single_person"
        replacement_mode = not bool(show_reference_background)
        segment_count = max(1, math.ceil(cap / max(1, chunk - overlap)))

        backend_settings = {
            "scail_identity_mode": identity_mode,
            "show_reference_background": bool(show_reference_background),
            "replacement_mode": replacement_mode,
            "interpolate_to_32fps": bool(interpolate_to_32fps),
            "scail_output_stage": "final_32fps_upscaled" if interpolate_to_32fps else "generated_16fps",
            "chunk_length": chunk,
            "overlap": overlap,
            "generation_steps": max(1, _int(generation_steps, 6)),
            "generation_cfg": max(0.0, _float(generation_cfg, 1.0)),
            "generation_seed": max(0, _int(generation_seed, 123)),
            "lora_name": str(lora_name or "auto / graph"),
            "lora_strength": _float(lora_strength, 1.0),
            "enable_sam31_preview": True,
            "preview_stage": str(preview_stage or "source"),
        }

        timeline = _json(timeline_data)
        timeline.update({
            "schema": "iamccs.v2v.shotboard.easy.scail",
            "schema_version": 1,
            "edition": "easy",
            "source_video_path": str(source_video_path or ""),
            "source_image_path": str(source_image_path or ""),
            "duration_seconds": selected_duration,
            "source_duration_seconds": duration,
            "fps": fps_value,
            "trim_start_s": trim_start,
            "trim_end_s": trim_end,
            "frame_load_cap": cap,
            "generation_width": width,
            "generation_height": height,
            "backend_family": "scail2",
            "backend_mode": "scail2",
            "backend_profile": profile,
            "backend_variant": profile,
            "backend_settings": backend_settings,
            "segment_count": segment_count,
            "chunk_length_frames": chunk,
            "chunk_overlap_frames": overlap,
            "global_prompt": str(global_prompt or ""),
            "negative_prompt": str(negative_prompt or ""),
            "output_prefix": str(output_prefix or "IAMCCS/SCAIL2_EASY_SHOTBOARD"),
        })

        outputs = {
            "duration_seconds": selected_duration,
            "source_duration_seconds": duration,
            "fps": fps_value,
            "trim_start_s": trim_start,
            "trim_end_s": trim_end,
            "frame_load_cap": cap,
            "generation_width": width,
            "generation_height": height,
            "source_video_path": str(source_video_path or ""),
            "source_image_path": str(source_image_path or ""),
            "backend_family": "scail2",
            "backend_mode": "scail2",
            "backend_profile": profile,
            "backend_variant": profile,
            "scail_identity_mode": identity_mode,
            "show_reference_background": bool(show_reference_background),
            "replacement_mode": replacement_mode,
            "interpolate_to_32fps": bool(interpolate_to_32fps),
            "scail_output_stage": "final_32fps_upscaled" if interpolate_to_32fps else "generated_16fps",
            "backend_settings": backend_settings,
            "global_prompt": str(global_prompt or ""),
            "negative_prompt": str(negative_prompt or ""),
            "output_prefix": str(output_prefix or "IAMCCS/SCAIL2_EASY_SHOTBOARD"),
            "timeline_json": json.dumps(timeline, ensure_ascii=False, separators=(",", ":")),
            "preview_stage": str(preview_stage or "source"),
        }
        report = (
            f"V2V Shotboard Easy SCAIL | mode={identity_mode} | size={width}x{height} | "
            f"trim={trim_start:.3f}-{trim_end:.3f}s | frames={cap} | chunk={chunk}/{overlap} | "
            f"reference_background={'show' if show_reference_background else 'hide'} | "
            f"output_fps={'32 (FILM x2)' if interpolate_to_32fps else '16 native'}"
        )
        outputs["report"] = report
        resources = {
            "v2v_payload": dict(outputs),
            "v2v_timeline": timeline,
            "v2v_timeline_json": outputs["timeline_json"],
            "v2v_backend_settings": backend_settings,
            "v2v_report": report,
        }
        cine_linx = {
            "type": SUPERNODE_LINX_TYPE,
            "pipeline_kind": "v2v",
            "edition": "easy",
            "backend_id": "IAMCCS_SCAIL2_V2V",
            "mode": "iamccs_scail2_easy_shotboard",
            "chain": [{"role": "planner", "name": "IAMCCS V2V Shotboard Easy - SCAIL Edition"}],
            "stages": [{
                "name": "SCAIL2_V2V",
                "kind": profile,
                "variant": profile,
                "settings": backend_settings,
                "payload": dict(outputs),
            }],
            "outputs": outputs,
            "resources": resources,
            "resource_keys": sorted(resources),
            "resource_types": {key: type(value).__name__ for key, value in resources.items()},
        }
        return cine_linx, report


NODE_CLASS_MAPPINGS = {
    "IAMCCS_V2VShotboardFreeScail": IAMCCS_V2VShotboardFreeScail,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_V2VShotboardFreeScail": "V2V Shotboard Easy - SCAIL Edition",
}
