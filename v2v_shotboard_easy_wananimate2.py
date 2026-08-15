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


def _chunk_4n1(value: Any, fallback: int = 81) -> int:
    chunk = max(5, _int(value, fallback))
    return ((chunk - 1) // 4) * 4 + 1


class IAMCCS_V2VShotboardEasyWanAnimate2:
    """Full-screen Easy control shell for an IAMCCS Wan-Animate-2 backend graph."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_video_path": ("STRING", {"default": ""}),
                "source_image_path": ("STRING", {"default": ""}),
                "duration_seconds": ("FLOAT", {"default": 2.7, "min": 0.01, "max": 3600.0, "step": 0.01}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "trim_start_s": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.01}),
                "trim_end_s": ("FLOAT", {"default": 2.7, "min": 0.01, "max": 3600.0, "step": 0.01}),
                "frame_load_cap": ("INT", {"default": 81, "min": 1, "max": 100000, "step": 1}),
                "generation_width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "generation_height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "chunk_frames": ("INT", {"default": 81, "min": 5, "max": 1025, "step": 4}),
                "generation_steps": ("INT", {"default": 6, "min": 1, "max": 100, "step": 1}),
                "generation_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.05}),
                "generation_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "seed_mode": (["fixed", "increment"], {"default": "fixed"}),
                "reference_image_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "pose_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pose_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "enable_context_windows": ("BOOLEAN", {"default": True}),
                "context_length_latents": ("INT", {"default": 21, "min": 1, "max": 1024, "step": 1}),
                "context_overlap_latents": ("INT", {"default": 8, "min": 0, "max": 1023, "step": 1}),
                "context_schedule": (
                    ["standard_static", "standard_uniform", "looped_uniform", "batched"],
                    {"default": "standard_static"},
                ),
                "context_fuse_method": (
                    ["pyramid", "relative", "flat", "overlap-linear"],
                    {"default": "pyramid"},
                ),
                "enable_pose_cache": ("BOOLEAN", {"default": True}),
                "cache_device": (["cpu", "gpu"], {"default": "cpu"}),
                "cache_dtype": (["int8", "int4", "default"], {"default": "int8"}),
                "reference_background_mode": (
                    ["keep_reference_background", "isolate_character"],
                    {"default": "keep_reference_background"},
                ),
                "output_background_mode": (
                    [
                        "native_generated",
                        "source_video_composite",
                        "reference_image_composite",
                        "custom_background_composite",
                    ],
                    {"default": "native_generated"},
                ),
                "live_chunk_preview": (
                    ["off", "first_frame", "middle_frame", "last_frame"],
                    {"default": "middle_frame"},
                ),
                "empty_cache_each_chunk": ("BOOLEAN", {"default": False}),
                "model_name": ("STRING", {"default": "auto / graph"}),
                "lora_name": ("STRING", {"default": "auto / graph"}),
                "apply_distill_lora": ("BOOLEAN", {"default": False}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "clip_name": ("STRING", {"default": "auto / graph"}),
                "clip_vision_name": ("STRING", {"default": "auto / graph"}),
                "vae_name": ("STRING", {"default": "auto / graph"}),
                "global_prompt": ("STRING", {
                    "default": "Character appearance: preserve the reference identity, clothing, proportions and detail. Background: preserve a stable coherent environment.",
                    "multiline": True,
                }),
                "pose_prompt": ("STRING", {
                    "default": "Follow the driving video motion, facial expression, gaze and body performance precisely.",
                    "multiline": True,
                }),
                "negative_prompt": ("STRING", {
                    "default": "identity drift, duplicated person, malformed anatomy, flicker, static motion, subtitles, text, low quality",
                    "multiline": True,
                }),
                "output_prefix": ("STRING", {"default": "IAMCCS/WAN_ANIMATE_2_EASY"}),
                "timeline_data": ("STRING", {"default": "", "multiline": True}),
                "preview_stage": (
                    ["source", "reference", "pose", "mask", "intermediate", "output"],
                    {"default": "source"},
                ),
            }
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE, "STRING")
    RETURN_NAMES = ("cine_linx", "report")
    FUNCTION = "plan"
    CATEGORY = "IAMCCS/V2V/Easy"
    DESCRIPTION = "Full-screen Wan-Animate-2 Easy shell mapped to a native/GGUF backend and IAMCCS extender."

    def plan(self, **kwargs):
        duration = max(0.01, _float(kwargs.get("duration_seconds"), 2.7))
        fps = max(1.0, _float(kwargs.get("fps"), 30.0))
        trim_start = min(duration, max(0.0, _float(kwargs.get("trim_start_s"), 0.0)))
        trim_end = min(duration, max(trim_start + 0.01, _float(kwargs.get("trim_end_s"), duration)))
        selected_duration = max(0.01, trim_end - trim_start)
        width = max(16, _int(kwargs.get("generation_width"), 832)) // 16 * 16
        height = max(16, _int(kwargs.get("generation_height"), 480)) // 16 * 16
        frame_cap = max(1, _int(kwargs.get("frame_load_cap"), round(selected_duration * fps)))
        chunk = _chunk_4n1(kwargs.get("chunk_frames"), 81)
        chunk_count = max(1, math.ceil(max(1, frame_cap - 1) / max(1, chunk - 1)))
        pose_start = min(1.0, max(0.0, _float(kwargs.get("pose_start_percent"), 0.0)))
        pose_end = min(1.0, max(pose_start, _float(kwargs.get("pose_end_percent"), 1.0)))

        backend_settings = {
            "chunk_length": chunk,
            "generation_steps": max(1, _int(kwargs.get("generation_steps"), 6)),
            "generation_cfg": max(0.0, _float(kwargs.get("generation_cfg"), 1.0)),
            "generation_seed": max(0, _int(kwargs.get("generation_seed"), 0)),
            "seed_mode": str(kwargs.get("seed_mode") or "fixed"),
            "reference_image_strength": max(0.0, _float(kwargs.get("reference_image_strength"), 1.0)),
            "pose_strength": max(0.0, _float(kwargs.get("pose_strength"), 1.0)),
            "pose_start_percent": pose_start,
            "pose_end_percent": pose_end,
            "enable_context_windows": bool(kwargs.get("enable_context_windows", True)),
            "context_length_latents": max(1, _int(kwargs.get("context_length_latents"), 21)),
            "context_overlap_latents": max(0, _int(kwargs.get("context_overlap_latents"), 8)),
            "context_schedule": str(kwargs.get("context_schedule") or "standard_static"),
            "context_fuse_method": str(kwargs.get("context_fuse_method") or "pyramid"),
            "enable_pose_cache": bool(kwargs.get("enable_pose_cache", True)),
            "cache_device": str(kwargs.get("cache_device") or "cpu"),
            "cache_dtype": str(kwargs.get("cache_dtype") or "int8"),
            "reference_background_mode": str(kwargs.get("reference_background_mode") or "keep_reference_background"),
            "output_background_mode": str(kwargs.get("output_background_mode") or "native_generated"),
            "live_chunk_preview": str(kwargs.get("live_chunk_preview") or "middle_frame"),
            "empty_cache_each_chunk": bool(kwargs.get("empty_cache_each_chunk", False)),
            "model_name": str(kwargs.get("model_name") or "auto / graph"),
            "lora_name": str(kwargs.get("lora_name") or "auto / graph"),
            "apply_distill_lora": bool(kwargs.get("apply_distill_lora", False)),
            "lora_strength": _float(kwargs.get("lora_strength"), 1.0),
            "clip_name": str(kwargs.get("clip_name") or "auto / graph"),
            "clip_vision_name": str(kwargs.get("clip_vision_name") or "auto / graph"),
            "vae_name": str(kwargs.get("vae_name") or "auto / graph"),
            "preview_stage": str(kwargs.get("preview_stage") or "source"),
        }

        timeline = _json(kwargs.get("timeline_data"))
        timeline.update({
            "schema": "iamccs.v2v.shotboard.easy.wananimate2",
            "schema_version": 1,
            "edition": "easy",
            "source_video_path": str(kwargs.get("source_video_path") or ""),
            "source_image_path": str(kwargs.get("source_image_path") or ""),
            "source_duration_seconds": duration,
            "duration_seconds": selected_duration,
            "fps": fps,
            "trim_start_s": trim_start,
            "trim_end_s": trim_end,
            "frame_load_cap": frame_cap,
            "generation_width": width,
            "generation_height": height,
            "backend_family": "wananimate2",
            "backend_mode": "wananimate2_native_extender",
            "backend_profile": "wananimate2_gguf_easy",
            "backend_settings": backend_settings,
            "chunk_count": chunk_count,
            "global_prompt": str(kwargs.get("global_prompt") or ""),
            "pose_prompt": str(kwargs.get("pose_prompt") or ""),
            "negative_prompt": str(kwargs.get("negative_prompt") or ""),
            "output_prefix": str(kwargs.get("output_prefix") or "IAMCCS/WAN_ANIMATE_2_EASY"),
        })

        outputs = {
            **timeline,
            "timeline_json": json.dumps(timeline, ensure_ascii=False, separators=(",", ":")),
            "preview_stage": backend_settings["preview_stage"],
        }
        report = (
            f"V2V Shotboard Easy Wan-Animate-2 | size={width}x{height} | "
            f"trim={trim_start:.3f}-{trim_end:.3f}s | estimated_frames={frame_cap} | "
            f"chunk={chunk} x {chunk_count} | background={backend_settings['output_background_mode']}"
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
            "backend_id": "IAMCCS_WAN_ANIMATE_2_V2V",
            "mode": "iamccs_wananimate2_easy_shotboard",
            "chain": [{"role": "planner", "name": "IAMCCS V2V Shotboard Easy - Wan Animate 2 Edition"}],
            "stages": [{
                "name": "WAN_ANIMATE_2_V2V",
                "kind": "wananimate2_native_extender",
                "variant": "gguf_easy",
                "settings": backend_settings,
                "payload": dict(outputs),
            }],
            "outputs": outputs,
            "resources": resources,
            "resource_keys": sorted(resources),
            "resource_types": {key: type(value).__name__ for key, value in resources.items()},
        }
        return cine_linx, report


class IAMCCS_WanAnimate2ShotboardBridge:
    """Expose the Wan-Animate-2 planner payload as typed graph values."""

    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    CATEGORY = "IAMCCS/V2V/Easy"
    FUNCTION = "map"
    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    RETURN_TYPES = (
        "FLOAT", "FLOAT", "INT", "INT", "INT", "INT", "INT", "FLOAT", "INT",
        "FLOAT", "FLOAT", "FLOAT", "FLOAT", "BOOLEAN", "INT", "INT", "BOOLEAN",
        "BOOLEAN", "STRING", "STRING", "STRING", "FLOAT", "STRING",
    )
    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    RETURN_NAMES = (
        "trim_start_s", "trim_duration_s", "target_frames", "width", "height",
        "chunk_frames", "generation_steps", "generation_cfg", "generation_seed",
        "reference_strength", "pose_strength", "pose_start_percent", "pose_end_percent",
        "context_windows", "context_length", "context_overlap", "pose_cache",
        "empty_cache_each_chunk", "positive_prompt", "negative_prompt", "pose_prompt",
        "effective_lora_strength", "output_prefix",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"cine_linx": (SUPERNODE_LINX_TYPE,)}}

    def map(self, cine_linx):
        # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        payload = cine_linx.get("outputs", {}) if isinstance(cine_linx, dict) else {}
        settings = payload.get("backend_settings", {}) if isinstance(payload, dict) else {}
        if not isinstance(settings, dict):
            settings = {}

        def value(name: str, fallback: Any = "") -> Any:
            return payload.get(name, fallback) if isinstance(payload, dict) else fallback

        def setting(name: str, fallback: Any) -> Any:
            return settings.get(name, fallback)

        lora_strength = _float(setting("lora_strength", 1.0), 1.0)
        effective_lora_strength = lora_strength if bool(setting("apply_distill_lora", False)) else 0.0
        # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        return (
            _float(value("trim_start_s", 0.0), 0.0),
            _float(value("duration_seconds", 2.7), 2.7),
            max(1, _int(value("frame_load_cap", 81), 81)),
            max(16, _int(value("generation_width", 832), 832)),
            max(16, _int(value("generation_height", 480), 480)),
            _chunk_4n1(setting("chunk_length", 81)),
            max(1, _int(setting("generation_steps", 6), 6)),
            max(0.0, _float(setting("generation_cfg", 1.0), 1.0)),
            max(0, _int(setting("generation_seed", 0), 0)),
            max(0.0, _float(setting("reference_image_strength", 1.0), 1.0)),
            max(0.0, _float(setting("pose_strength", 1.0), 1.0)),
            min(1.0, max(0.0, _float(setting("pose_start_percent", 0.0), 0.0))),
            min(1.0, max(0.0, _float(setting("pose_end_percent", 1.0), 1.0))),
            bool(setting("enable_context_windows", True)),
            max(1, _int(setting("context_length_latents", 21), 21)),
            max(0, _int(setting("context_overlap_latents", 8), 8)),
            bool(setting("enable_pose_cache", True)),
            bool(setting("empty_cache_each_chunk", False)),
            str(value("global_prompt", "")),
            str(value("negative_prompt", "")),
            str(value("pose_prompt", "")),
            effective_lora_strength,
            str(value("output_prefix", "IAMCCS/WAN_ANIMATE_2_EASY")),
        )


NODE_CLASS_MAPPINGS = {
    "IAMCCS_V2VShotboardEasyWanAnimate2": IAMCCS_V2VShotboardEasyWanAnimate2,
    "IAMCCS_WanAnimate2ShotboardBridge": IAMCCS_WanAnimate2ShotboardBridge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_V2VShotboardEasyWanAnimate2": "V2V Shotboard Easy - Wan Animate 2 Edition",
    "IAMCCS_WanAnimate2ShotboardBridge": "Wan Animate 2 Shotboard Backend Bridge",
}
