"""Autonomous IAMCCS Shotboard V3B for LTX-2.5 BBox control.

V3B deliberately does not subclass or execute Shotboard V3.  It owns its input
contract, timeline compilation, image loading, BBox project and CineLinX output.
The production V3/V4 code paths therefore remain completely untouched.
"""

from __future__ import annotations

import copy
import json
import math
from typing import Any, Dict, Iterable, Tuple

import torch

from .iamccs_cine_nodes import IAMCCS_CineReferenceBoard, SUPERNODE_LINX_TYPE


SCHEMA = "iamccs.ltx25.shotboard.v3b.bbox"


def _loads(value: Any, fallback: Any) -> Any:
    if isinstance(value, (dict, list)):
        return copy.deepcopy(value)
    try:
        return json.loads(str(value or ""))
    except Exception:
        return copy.deepcopy(fallback)


def _clamp(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        return max(lo, min(hi, float(value)))
    except Exception:
        return default


def _resource(cine_linx: Any, key: str, default: Any = None) -> Any:
    if not isinstance(cine_linx, dict):
        return default
    resources = cine_linx.get("resources")
    return resources.get(key, default) if isinstance(resources, dict) else default


def _normal_project(raw: Any) -> Dict[str, Any]:
    project = _loads(raw, {})
    if not isinstance(project, dict):
        project = {}
    objects = []
    for index, source in enumerate(project.get("objects") or []):
        if not isinstance(source, dict):
            continue
        keyframes = []
        for kf in source.get("keyframes") or []:
            if not isinstance(kf, dict) or not isinstance(kf.get("box"), (list, tuple)) or len(kf["box"]) != 4:
                continue
            box = [_clamp(v, -2.0, 3.0, 0.0) for v in kf["box"]]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            keyframes.append({"time": _clamp(kf.get("time", kf.get("frame", 0)), 0.0, 1.0, 0.0), "box": box})
        keyframes.sort(key=lambda item: item["time"])
        if not keyframes:
            continue
        objects.append({
            "id": str(source.get("id") or f"object_{index + 1}"),
            "name": str(source.get("name") or f"Object {index + 1}"),
            "prompt": str(source.get("prompt") or "").strip(),
            "color": str(source.get("color") or "#71e2ff"),
            "strength": _clamp(source.get("strength", 1.0), 0.0, 5.0, 1.0),
            "enabled": bool(source.get("enabled", True)),
            "start_time": _clamp(source.get("start_time", source.get("start_frame", 0)), 0.0, 1.0, 0.0),
            "end_time": _clamp(source.get("end_time", source.get("end_frame", 1)), 0.0, 1.0, 1.0),
            "keyframes": keyframes,
        })
    return {
        "schema": SCHEMA,
        "version": 1,
        "style_prompt": str(project.get("style_prompt") or "").strip(),
        "scene_prompt": str(project.get("scene_prompt") or "").strip(),
        "background_image": str(project.get("background_image") or project.get("bg_image_base64") or ""),
        "objects": objects,
    }


def compile_bbox_project(raw: Any, width: int, height: int, total_frames: int) -> Dict[str, Any]:
    """Compile V3B normalized time/boxes to the native animator pixel contract."""
    project = _normal_project(raw)
    width, height, total_frames = max(64, int(width)), max(64, int(height)), max(1, int(total_frames))
    compiled = {
        "version": 2,
        "style_prompt": project["style_prompt"],
        "scene_prompt": project["scene_prompt"],
        "bg_image_base64": project["background_image"],
        "objects": [],
    }
    for obj in project["objects"]:
        keyframes = []
        for kf in obj["keyframes"]:
            x1, y1, x2, y2 = kf["box"]
            keyframes.append({
                "frame": int(round(kf["time"] * (total_frames - 1))),
                "box": [int(round(x1 * width)), int(round(y1 * height)), int(round(x2 * width)), int(round(y2 * height))],
            })
        compiled["objects"].append({
            "id": obj["id"], "name": obj["name"], "prompt": obj["prompt"],
            "color": obj["color"], "strength": obj["strength"], "enabled": obj["enabled"],
            "start_frame": int(round(obj["start_time"] * (total_frames - 1))),
            "end_frame": int(round(obj["end_time"] * (total_frames - 1))),
            "keyframes": keyframes,
        })
    return compiled


def _node_result(value: Any) -> Tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    result = getattr(value, "result", None)
    if isinstance(result, tuple):
        return result
    if isinstance(result, list):
        return tuple(result)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _node_class(name: str):
    import nodes
    cls = nodes.NODE_CLASS_MAPPINGS.get(name)
    if cls is None:
        raise RuntimeError(
            f"Required node {name} is not installed. Install/update ComfyUI-LTX-BBox-Animator and ComfyUI-LTXVideo."
        )
    return cls


def _reference_paths(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        pass
    return [item.strip() for item in text.replace("\r", "\n").splitlines() if item.strip()]


def _round_ltx_frames(duration_seconds: float, fps: int) -> int:
    requested = max(1, int(round(float(duration_seconds) * max(1, int(fps)))))
    return max(1, int(math.ceil(max(0, requested - 1) / 8.0) * 8 + 1))


class IAMCCS_CineShotboardPlannerV3B:
    CATEGORY = "IAMCCS/Cine/LTX 2.5 BBox"
    RETURN_TYPES = (SUPERNODE_LINX_TYPE,)
    RETURN_NAMES = ("cine_linx",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "global_prompt": ("STRING", {"default": "A coherent cinematic shot with stable subjects and controlled motion.", "multiline": True}),
            "timeline_data": ("STRING", {"default": "", "multiline": True, "tooltip": "Owned and edited by the autonomous V3B UI."}),
            "duration_seconds": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 3600.0, "step": 0.1}),
            "frame_rate": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1}),
            "image_paths": ("STRING", {"default": "", "multiline": True, "tooltip": "Owned by the autonomous V3B image slots."}),
            "image_width": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 32}),
            "image_height": ("INT", {"default": 448, "min": 64, "max": 8192, "step": 32}),
            "image_resize_method": (["crop", "pad", "keep proportion", "stretch"], {"default": "crop"}),
            "image_multiple_of": ("INT", {"default": 32, "min": 1, "max": 512, "step": 1}),
            "img_compression": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            "bbox_enabled": ("BOOLEAN", {"default": True}),
            "bbox_project_json": ("STRING", {
                "default": json.dumps({
                    "schema": SCHEMA, "version": 1,
                    "style_prompt": "Cinematic natural light, steady eye-level camera, realistic textures and colors.",
                    "scene_prompt": "",
                    "objects": [],
                }),
                "multiline": True,
                "tooltip": "Owned by the autonomous V3B BBox Director.",
            }),
        }}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return json.dumps({key: str(value) for key, value in sorted(kwargs.items())}, sort_keys=True)

    def execute(self, global_prompt, timeline_data, duration_seconds, frame_rate, image_paths,
                image_width, image_height, image_resize_method="crop", image_multiple_of=32,
                img_compression=0, bbox_enabled=True, bbox_project_json="{}"):
        timeline = _loads(timeline_data, {})
        if not isinstance(timeline, dict):
            timeline = {}
        duration_seconds = max(0.1, float(timeline.get("duration_seconds", duration_seconds) or duration_seconds))
        frame_rate = max(1, int(timeline.get("frame_rate", frame_rate) or frame_rate))
        image_width = max(64, int(timeline.get("image_width", image_width) or image_width))
        image_height = max(64, int(timeline.get("image_height", image_height) or image_height))
        total_frames = _round_ltx_frames(duration_seconds, frame_rate)
        paths = _reference_paths(image_paths)
        segments = []
        for index, source in enumerate(timeline.get("segments") or []):
            if not isinstance(source, dict) or str(source.get("type", "image")).lower() != "image":
                continue
            item = copy.deepcopy(source)
            item["start"] = max(0, min(total_frames - 1, int(item.get("start", 0) or 0)))
            item["length"] = max(1, min(total_frames - item["start"], int(item.get("length", frame_rate) or frame_rate)))
            item["ref"] = max(1, int(item.get("ref", index + 1) or index + 1))
            if not str(item.get("imageFile") or item.get("imageTruthPath") or "").strip() and item["ref"] <= len(paths):
                item["imageFile"] = paths[item["ref"] - 1]
            segments.append(item)
        segments.sort(key=lambda item: (item["start"], item["ref"]))
        local_prompts = []
        lengths = []
        for item in segments:
            local_prompts.append(str(item.get("prompt", "") or "").strip())
            lengths.append(int(item["length"]))
        if lengths:
            visual_end = max(int(item["start"]) + int(item["length"]) for item in segments)
            if visual_end < total_frames:
                lengths[-1] += total_frames - visual_end

        if paths:
            multi_output = IAMCCS_CineReferenceBoard().load_ltx_style_images(
                "\n".join(paths), image_width, image_height, image_resize_method,
                int(image_multiple_of or 32), int(img_compression or 0),
            )
        else:
            multi_output = torch.zeros((1, 64, 64, 3))
        image_1 = multi_output[0:1] if torch.is_tensor(multi_output) and int(multi_output.shape[0]) else torch.zeros((1, 64, 64, 3))

        embedded_project = timeline.get("bboxControl", timeline.get("bbox_control")) if isinstance(timeline, dict) else None
        project = _normal_project(embedded_project if isinstance(embedded_project, dict) else bbox_project_json)
        timeline["schema"] = "iamccs.ltx25.shotboard.v3b.timeline"
        timeline["schema_version"] = 1
        timeline["duration_seconds"] = duration_seconds
        timeline["frame_rate"] = frame_rate
        timeline["image_width"] = image_width
        timeline["image_height"] = image_height
        timeline["segments"] = segments
        timeline["bboxControl"] = project
        timeline["bbox_control"] = project
        timeline_json = json.dumps(timeline, ensure_ascii=False)
        visual_json = json.dumps(segments, ensure_ascii=False)
        local_prompt_text = " | ".join(local_prompts)
        length_text = ",".join(str(value) for value in lengths)
        contract = {
            "schema": SCHEMA,
            "coordinates": "normalized_xyxy",
            "timeline": "normalized_0_1",
            "control": "black_hollow_white_bbox_center_dot_5_frame_trail",
            "regional_global_weights": [0.85, 0.15],
            "media_authority": "shotboard_v3b_image_slots_and_timeline",
        }
        payload = {
            "backend_mode": "cine_ltx25_shotboard_v3b_bbox",
            "global_prompt": str(global_prompt or ""),
            "timeline_data": timeline_json,
            "duration_seconds": duration_seconds,
            "frame_rate": frame_rate,
            "max_frames": total_frames,
            "image_paths": "\n".join(paths),
            "visual_segments": segments,
            "local_prompts": local_prompt_text,
            "segment_lengths": length_text,
            "bbox_project": project,
        }
        resources = {
            "cine_payload": payload,
            "cine_board_timeline_data": timeline_json,
            "cine_global_prompt": str(global_prompt or ""),
            "cine_local_prompts": local_prompt_text,
            "cine_segment_lengths": length_text,
            "cine_promptrelay_enabled": bool(local_prompt_text.strip()),
            "cine_max_frames": total_frames,
            "cine_duration_seconds": duration_seconds,
            "cine_frame_rate": frame_rate,
            "cine_image_paths": "\n".join(paths),
            "cine_image_width": image_width,
            "cine_image_height": image_height,
            "cine_image_resize_method": image_resize_method,
            "cine_image_multiple_of": int(image_multiple_of or 32),
            "cine_img_compression": int(img_compression or 0),
            "cine_multi_input": multi_output,
            "cine_image_1": image_1,
            "cine_visual_segments_json": visual_json,
            "cine_ltx25_bbox_enabled": bool(bbox_enabled),
            "cine_ltx25_bbox_project": project,
            "cine_ltx25_bbox_project_json": json.dumps(project, ensure_ascii=False),
            "cine_ltx25_bbox_contract": contract,
        }
        cine_linx = {
            "type": SUPERNODE_LINX_TYPE,
            "pipeline_kind": "ltx25_i2v_bbox",
            "mode": "cine_ltx25_shotboard_v3b_bbox",
            "chain": [{"role": "planner", "name": "Autonomous Shotboard V3B"}],
            "stages": [{"name": "IAMCCS_LTX25BBoxBackend", "kind": "ltx25_bbox_backend", "payload": payload}],
            "policies": {"authoring_source": "shotboard_v3b", "bbox_backend": "IAMCCS_LTX25BBoxBackend"},
            "outputs": {
                "timeline_data": timeline_json, "global_prompt": str(global_prompt or ""),
                "local_prompts": local_prompt_text, "segment_lengths": length_text,
                "max_frames": total_frames, "duration_seconds": duration_seconds, "frame_rate": frame_rate,
            },
            "resources": resources,
            "resource_keys": sorted(resources.keys()),
            "resource_types": {key: type(value).__name__ for key, value in resources.items()},
        }
        return (cine_linx,)


class IAMCCS_LTX25BBoxBackend:
    CATEGORY = "IAMCCS/Cine/LTX 2.5 BBox"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "cine_linx": (SUPERNODE_LINX_TYPE,),
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "video_vae": ("VAE",),
            "video_latent": ("LATENT",),
            "negative": ("CONDITIONING",),
            "regional_prompt_weight": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 2.0, "step": 0.05}),
            "global_prompt_weight": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 2.0, "step": 0.05}),
            "control_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "use_first_timeline_image": ("BOOLEAN", {"default": True}),
            "i2v_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
            "latent_downscale_factor": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 10.0, "step": 1.0}),
            "use_tiled_encode": ("BOOLEAN", {"default": False}),
            "tile_size": ("INT", {"default": 256, "min": 64, "max": 512, "step": 32}),
            "tile_overlap": ("INT", {"default": 64, "min": 16, "max": 256, "step": 16}),
        }}

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "IMAGE", "LTX_REGIONS", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "control_images", "regions", "global_prompt", "compiled_bbox_json", "report")
    FUNCTION = "apply"

    def apply(self, cine_linx, model, clip, video_vae, video_latent, negative,
              regional_prompt_weight=0.85, global_prompt_weight=0.15, control_strength=1.0,
              use_first_timeline_image=True, i2v_strength=0.7,
              latent_downscale_factor=1.0, use_tiled_encode=False, tile_size=256, tile_overlap=64):
        if not bool(_resource(cine_linx, "cine_ltx25_bbox_enabled", False)):
            raise ValueError("Shotboard V3B BBox control is disabled")
        width = int(_resource(cine_linx, "cine_image_width", 768) or 768)
        height = int(_resource(cine_linx, "cine_image_height", 448) or 448)
        total_frames = int(_resource(cine_linx, "cine_max_frames", _resource(cine_linx, "cine_frame_count", 121)) or 121)
        fps = float(_resource(cine_linx, "cine_frame_rate", 24) or 24)
        project = _resource(cine_linx, "cine_ltx25_bbox_project", {})
        working_latent = dict(video_latent)
        samples = working_latent.get("samples")
        if hasattr(samples, "clone"):
            working_latent["samples"] = samples.clone()
        first_image = _resource(cine_linx, "cine_image_1")
        # Respect the actual first image clip on the Shotboard timeline.  This
        # keeps slot order and clip order independent when the user rearranges
        # media by dragging it on the V3B timeline.
        timeline_segments = _loads(_resource(cine_linx, "cine_visual_segments_json", "[]"), [])
        image_segments = [
            segment for segment in timeline_segments
            if isinstance(segment, dict) and str(segment.get("type", "image")).lower() == "image"
        ] if isinstance(timeline_segments, list) else []
        if image_segments:
            first_segment = min(image_segments, key=lambda segment: int(segment.get("start", 0) or 0))
            ref_index = max(1, int(first_segment.get("ref", 1) or 1)) - 1
            multi_input = _resource(cine_linx, "cine_multi_input")
            try:
                if multi_input is not None and int(multi_input.shape[0]) > ref_index:
                    first_image = multi_input[ref_index:ref_index + 1]
            except Exception:
                pass
        image_paths = str(_resource(cine_linx, "cine_image_paths", "") or "").strip()
        i2v_applied = False
        if bool(use_first_timeline_image) and image_paths and first_image is not None:
            i2v_cls = _node_class("LTXVImgToVideoConditionOnly")
            i2v_node = i2v_cls()
            (working_latent,) = _node_result(i2v_node.generate(
                first_image, video_vae, working_latent, float(i2v_strength), False
            ))[:1]
            i2v_applied = True
        compiled = compile_bbox_project(project, width, height, total_frames)
        if not compiled["objects"]:
            raise ValueError("Shotboard V3B has no enabled BBox objects/keyframes")
        native_json = json.dumps(compiled, ensure_ascii=False)

        animator = _node_class("LTXRegionalBBoxAnimator")()
        control, positive, regions, global_prompt = _node_result(
            animator.generate(clip, width, height, total_frames, fps, native_json)
        )[:4]
        conditioning_cls = _node_class("LTXVConditioning")
        positive, negative = _node_result(conditioning_cls.execute(positive, negative, fps))[:2]
        regional = _node_class("LTXApplyRegionalConditioning")()
        (patched_model,) = _node_result(regional.apply(
            model, working_latent, regions, regional_prompt_weight, global_prompt_weight
        ))[:1]
        guide_cls = _node_class("LTXAddVideoICLoRAGuide")
        guided = _node_result(guide_cls.execute(
            positive, negative, video_vae, working_latent, control, 0, control_strength,
            latent_downscale_factor, "disabled", use_tiled_encode, tile_size, tile_overlap,
        ))
        positive_out, negative_out, latent_out = guided[:3]
        report = json.dumps({
            "ok": True, "schema": SCHEMA, "width": width, "height": height,
            "frames": total_frames, "fps": fps, "objects": len(compiled["objects"]),
            "regional_prompt_weight": regional_prompt_weight,
            "global_prompt_weight": global_prompt_weight,
            "control_strength": control_strength,
            "shotboard_first_image_i2v": i2v_applied,
            "i2v_strength": i2v_strength,
        }, ensure_ascii=False)
        return patched_model, positive_out, negative_out, latent_out, control, regions, global_prompt, native_json, report


NODE_CLASS_MAPPINGS = {
    "IAMCCS_CineShotboardPlannerV3B": IAMCCS_CineShotboardPlannerV3B,
    "IAMCCS_LTX25BBoxBackend": IAMCCS_LTX25BBoxBackend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_CineShotboardPlannerV3B": "IAMCCS Cine Shotboard V3B · LTX 2.5 BBox",
    "IAMCCS_LTX25BBoxBackend": "IAMCCS LTX 2.5 BBox Backend",
}
