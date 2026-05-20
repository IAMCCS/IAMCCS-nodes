from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import math
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import folder_paths
from PIL import Image, ImageOps


SUPERNODE_LINX_TYPE = "IAMCCS_SUPERNODE_LINX"
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".avif"}
_MOTION_WORDS = {
    "sea", "ocean", "water", "wave", "waves", "foam", "surface", "underwater",
    "bubble", "bubbles", "particle", "particles", "caustic", "caustics",
    "dolly", "push", "push-in", "camera", "glide", "travel", "travels", "move",
    "moves", "moving", "motion", "parallax", "drift", "drifts", "roll", "rolls",
    "slide", "slides", "flow", "flows",
}
_IDENTITY_WORDS = {
    "face", "siren", "sirena", "mermaid", "creature", "character", "eyes", "mouth",
    "skin", "portrait", "closeup", "close-up", "macro", "head", "woman",
}


def _clamp(value: Any, lo: float, hi: float, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        number = float(default)
    return max(float(lo), min(float(hi), number))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_json_loads(value: Any, fallback: Any) -> Any:
    if isinstance(value, (dict, list)):
        return copy.deepcopy(value)
    try:
        return json.loads(str(value or ""))
    except Exception:
        return copy.deepcopy(fallback)


def _sanitize(value: Any, fallback: str = "cine_resolution_parity") -> str:
    clean = re.sub(r'[<>:"/\\|?*\x00-\x1F]+', "_", str(value or fallback).strip())
    clean = re.sub(r"\s+", "_", clean).strip("._")
    return (clean[:80] or fallback)


def _split_paths(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item or "").strip()]
    raw = str(value or "").strip()
    if not raw:
        return []
    parsed = _safe_json_loads(raw, None)
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item or "").strip()]
    if "\n" in raw or "\r" in raw:
        return [item.strip() for item in raw.splitlines() if item.strip()]
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    return parts if len(parts) > 1 else [raw]


def _join_paths(paths: Iterable[str]) -> str:
    return "\n".join(str(path or "").strip() for path in paths if str(path or "").strip())


def _text_for_segment(seg: Dict[str, Any]) -> str:
    parts = [
        seg.get("label", ""),
        seg.get("prompt", ""),
        seg.get("camera", ""),
        seg.get("transition", ""),
        seg.get("note", ""),
        seg.get("relay_prompt", ""),
        seg.get("step_transition_prompt", ""),
    ]
    return " ".join(str(part or "").lower() for part in parts)


def _classify_segment(seg: Dict[str, Any]) -> str:
    text = _text_for_segment(seg)
    identity_hits = sum(1 for word in _IDENTITY_WORDS if word in text)
    motion_hits = sum(1 for word in _MOTION_WORDS if word in text)
    if identity_hits and identity_hits >= motion_hits:
        return "identity"
    if motion_hits:
        return "motion"
    return "neutral"


def _strength_from_segment(seg: Dict[str, Any], default: float = 1.0) -> float:
    for key in ("guide_strength", "guideStrength", "strength", "force", "motion_force"):
        if key in seg and seg.get(key) is not None:
            return _clamp(seg.get(key), 0.0, 1.0, default)
    return _clamp(default, 0.0, 1.0, 1.0)


def _set_strength(seg: Dict[str, Any], value: float) -> None:
    strength = float(_clamp(value, 0.0, 1.0, 1.0))
    for key in ("guideStrength", "guide_strength", "strength", "force", "imageLockStrength", "image_lock_strength"):
        if key in seg:
            seg[key] = strength
    if not any(key in seg for key in ("guideStrength", "guide_strength", "strength", "force")):
        seg["guideStrength"] = strength


def _resolve_image_path(path: str) -> Optional[str]:
    clean = str(path or "").strip()
    if not clean or clean.startswith("data:"):
        return None
    if os.path.isabs(clean):
        return os.path.abspath(os.path.expanduser(clean))
    return os.path.abspath(os.path.join(folder_paths.get_input_directory(), clean.replace("/", os.sep)))


def _open_image(path_or_data: str) -> Image.Image:
    clean = str(path_or_data or "").strip()
    if clean.startswith("data:image"):
        _, _, payload = clean.partition(",")
        if not payload:
            raise ValueError("Invalid image data URL")
        return Image.open(io.BytesIO(base64.b64decode(payload)))
    source = _resolve_image_path(clean)
    if not source or not os.path.isfile(source):
        raise FileNotFoundError(f"Reference image not found: {clean}")
    return Image.open(source)


def _resize_cover(image: Image.Image, width: int, height: int, resample: int) -> Image.Image:
    src_w, src_h = image.size
    if src_w <= 0 or src_h <= 0:
        return image.resize((width, height), resample)
    target_ratio = width / float(height)
    src_ratio = src_w / float(src_h)
    if src_ratio >= target_ratio:
        crop_h = src_h
        crop_w = int(round(crop_h * target_ratio))
    else:
        crop_w = src_w
        crop_h = int(round(crop_w / target_ratio))
    left = max(0, int(round((src_w - crop_w) / 2.0)))
    top = max(0, int(round((src_h - crop_h) / 2.0)))
    return image.crop((left, top, min(src_w, left + crop_w), min(src_h, top + crop_h))).resize((width, height), resample)


def _prefilter_image_to_target(
    source: str,
    target_width: int,
    target_height: int,
    reference_width: int,
    reference_height: int,
    method: str,
    package_tag: str,
) -> str:
    resample = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }.get(str(method or "bicubic").lower(), Image.Resampling.BICUBIC)

    with _open_image(source) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
        semantic = _resize_cover(image, max(64, reference_width), max(64, reference_height), resample)
        target = semantic.resize((max(64, target_width), max(64, target_height)), resample)

    out_root = os.path.join(folder_paths.get_input_directory(), "IAMCCS_resolution_parity")
    os.makedirs(out_root, exist_ok=True)
    source_name = os.path.basename(_resolve_image_path(source) or "reference.png")
    stem = _sanitize(os.path.splitext(source_name)[0], "ref")[:48]
    digest = hashlib.sha1(str(source).encode("utf-8", errors="ignore")).hexdigest()[:10]
    filename = f"{package_tag}_{stem}_{digest}_{target_width}x{target_height}.png"
    out_path = os.path.join(out_root, filename)
    target.save(out_path, "PNG", optimize=True)
    return "IAMCCS_resolution_parity/" + filename


def _rewrite_image_refs(obj: Any, path_map: Dict[str, str]) -> None:
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            if key in {"imageFile", "image_file", "path"} and isinstance(value, str) and value in path_map:
                obj[key] = path_map[value]
            else:
                _rewrite_image_refs(value, path_map)
    elif isinstance(obj, list):
        for item in obj:
            _rewrite_image_refs(item, path_map)


def _apply_strength_translation(segments: List[Dict[str, Any]], motion_multiplier: float, identity_multiplier: float) -> Dict[str, Any]:
    counts = {"motion": 0, "identity": 0, "neutral": 0}
    values = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        role = _classify_segment(seg)
        counts[role] = counts.get(role, 0) + 1
        old = _strength_from_segment(seg, 1.0)
        if role == "motion":
            new = old * motion_multiplier
        elif role == "identity":
            new = old * identity_multiplier
        else:
            new = old
        new = _clamp(new, 0.0, 1.0, old)
        _set_strength(seg, new)
        seg["resolution_parity_role"] = role
        seg["resolution_parity_original_strength"] = float(old)
        seg["resolution_parity_strength"] = float(new)
        values.append({"label": seg.get("label", ""), "role": role, "from": float(old), "to": float(new)})
    return {"counts": counts, "values": values}


class IAMCCS_CineResolutionParityTranslator:
    """Translate a proven lower-resolution shotboard into a higher-resolution run."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "enabled": ("BOOLEAN", {"default": True}),
                "reference_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 32}),
                "reference_height": ("INT", {"default": 576, "min": 64, "max": 8192, "step": 32}),
                "guide_prefilter": ("BOOLEAN", {"default": True}),
                "prefilter_method": (["bicubic", "lanczos", "bilinear", "nearest"], {"default": "bicubic"}),
                "motion_multiplier": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "0 = auto, computed as reference/target linear scale.",
                }),
                "identity_multiplier": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE, "STRING")
    RETURN_NAMES = ("cine_linx", "report")
    FUNCTION = "translate"
    CATEGORY = "IAMCCS/Cine/02 Single Generation"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    def translate(
        self,
        cine_linx,
        enabled=True,
        reference_width=1024,
        reference_height=576,
        guide_prefilter=True,
        prefilter_method="bicubic",
        motion_multiplier=0.0,
        identity_multiplier=1.05,
    ):
        if not isinstance(cine_linx, dict):
            return cine_linx, json.dumps({"ok": False, "error": "cine_linx is not a dictionary"}, indent=2)
        if not enabled:
            return cine_linx, json.dumps({"ok": True, "enabled": False, "note": "Resolution parity translator bypassed."}, indent=2)

        out = dict(cine_linx)
        original_resources = cine_linx.get("resources") if isinstance(cine_linx.get("resources"), dict) else {}
        resources = dict(original_resources)
        out["resources"] = resources
        outputs = dict(cine_linx.get("outputs") if isinstance(cine_linx.get("outputs"), dict) else {})
        out["outputs"] = outputs
        payload = resources.get("cine_payload")
        if not isinstance(payload, dict):
            payload = {}
        else:
            payload = copy.deepcopy(payload)
        resources["cine_payload"] = payload

        target_width = _safe_int(resources.get("cine_image_width", outputs.get("width", payload.get("image_width", 0))), 0)
        target_height = _safe_int(resources.get("cine_image_height", outputs.get("height", payload.get("image_height", 0))), 0)
        if target_width <= 0:
            target_width = _safe_int(payload.get("image_width"), 768)
        if target_height <= 0:
            target_height = _safe_int(payload.get("image_height"), 432)

        resources["cine_image_width"] = int(target_width)
        resources["cine_image_height"] = int(target_height)
        payload["image_width"] = int(target_width)
        payload["image_height"] = int(target_height)
        outputs["width"] = int(target_width)
        outputs["height"] = int(target_height)

        ref_w = max(64, _safe_int(reference_width, 1024))
        ref_h = max(64, _safe_int(reference_height, 576))
        scale_x = target_width / float(ref_w)
        scale_y = target_height / float(ref_h)
        linear_scale = math.sqrt(max(0.0001, scale_x * scale_y))
        auto_motion = 1.0 / linear_scale if linear_scale > 1.0 else 1.0
        motion_mul = float(auto_motion if _safe_float(motion_multiplier, 0.0) <= 0 else motion_multiplier)
        identity_mul = float(identity_multiplier)

        visual_segments = _safe_json_loads(resources.get("cine_visual_segments_json", payload.get("visual_segments", [])), [])
        if isinstance(visual_segments, dict):
            visual_segments = visual_segments.get("segments", [])
        if not isinstance(visual_segments, list):
            visual_segments = []

        strength_report = _apply_strength_translation(visual_segments, motion_mul, identity_mul)

        original_paths = _split_paths(resources.get("cine_image_paths", payload.get("image_paths", "")))
        path_map: Dict[str, str] = {}
        image_errors = []
        package_tag = f"parity_{int(time.time() * 1000)}"

        if guide_prefilter and target_width > 0 and target_height > 0:
            for path in original_paths:
                try:
                    path_map[path] = _prefilter_image_to_target(
                        path,
                        target_width,
                        target_height,
                        ref_w,
                        ref_h,
                        str(prefilter_method),
                        package_tag,
                    )
                except Exception as exc:
                    image_errors.append({"path": path, "error": str(exc)})
            for seg in visual_segments:
                if not isinstance(seg, dict):
                    continue
                image_file = str(seg.get("imageFile", seg.get("image_file", "")) or "").strip()
                if image_file and image_file not in path_map:
                    try:
                        path_map[image_file] = _prefilter_image_to_target(
                            image_file,
                            target_width,
                            target_height,
                            ref_w,
                            ref_h,
                            str(prefilter_method),
                            package_tag,
                        )
                    except Exception as exc:
                        image_errors.append({"path": image_file, "error": str(exc)})

        if path_map:
            translated_paths = [path_map.get(path, path) for path in original_paths]
            resources["cine_image_paths"] = _join_paths(translated_paths)
            payload["image_paths"] = resources["cine_image_paths"]
            _rewrite_image_refs(visual_segments, path_map)
            _rewrite_image_refs(payload.get("rows"), path_map)
            _rewrite_image_refs(payload.get("guide_rows"), path_map)

        resources["cine_visual_segments_json"] = json.dumps(visual_segments, ensure_ascii=False)
        payload["visual_segments"] = visual_segments
        payload["resolution_parity"] = {
            "enabled": True,
            "reference_width": int(ref_w),
            "reference_height": int(ref_h),
            "target_width": int(target_width),
            "target_height": int(target_height),
            "linear_scale": float(linear_scale),
            "motion_multiplier": float(motion_mul),
            "identity_multiplier": float(identity_mul),
            "guide_prefilter": bool(guide_prefilter),
            "prefilter_method": str(prefilter_method),
            "prefiltered_images": len(path_map),
        }
        resources["cine_resolution_parity"] = payload["resolution_parity"]

        stages = []
        for stage in out.get("stages", []) if isinstance(out.get("stages"), list) else []:
            if isinstance(stage, dict) and isinstance(stage.get("payload"), dict):
                stage_copy = dict(stage)
                stage_copy["payload"] = payload
                stages.append(stage_copy)
            else:
                stages.append(stage)
        if stages:
            out["stages"] = stages

        resources["cine_payload"] = payload
        out["resource_keys"] = sorted(resources.keys())
        out["resource_types"] = {key: type(value).__name__ for key, value in resources.items()}

        report = {
            "ok": True,
            "node": "IAMCCS_CineResolutionParityTranslator",
            "reference": [int(ref_w), int(ref_h)],
            "target": [int(target_width), int(target_height)],
            "linear_scale": round(float(linear_scale), 4),
            "motion_multiplier": round(float(motion_mul), 4),
            "identity_multiplier": round(float(identity_mul), 4),
            "guide_prefilter": bool(guide_prefilter),
            "prefiltered_images": len(path_map),
            "image_errors": image_errors,
            "strength_translation": strength_report,
        }
        return out, json.dumps(report, indent=2, ensure_ascii=False)
