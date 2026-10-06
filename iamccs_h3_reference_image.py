"""Lazy R42 still-image branch driven by IAMCCS Settings PRO and Prompter."""

from __future__ import annotations

import base64
import copy
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import folder_paths
from PIL import Image


def _settings_from_linx(value: Any) -> dict[str, Any]:
    """Find the named Settings PRO payload without depending on one bus layout."""
    seen: set[int] = set()

    def visit(item: Any) -> dict[str, Any] | None:
        if isinstance(item, dict):
            marker = id(item)
            if marker in seen:
                return None
            seen.add(marker)
            payload = item.get("iamccs_minimax_h3_settings")
            if isinstance(payload, dict):
                settings = payload.get("settings")
                if isinstance(settings, dict):
                    return settings
            if isinstance(item.get("settings"), dict) and (
                str(item.get("schema", "")).startswith("iamccs.minimax_h3")
                or "task_mode" in item["settings"]
            ):
                return item["settings"]
            for child in item.values():
                result = visit(child)
                if result is not None:
                    return result
        elif isinstance(item, (list, tuple)):
            for child in item:
                result = visit(child)
                if result is not None:
                    return result
        return None

    return copy.deepcopy(visit(value) or {})


def _project(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value or "{}"))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _decode_visual(entry: Any) -> torch.Tensor | None:
    if not isinstance(entry, dict):
        return None
    encoded = entry.get("data") or entry.get("base64") or entry.get("data_url")
    if not isinstance(encoded, str) or not encoded.strip():
        return None
    encoded = encoded.strip()
    if encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[-1]
    encoded += "=" * (-len(encoded) % 4)
    try:
        raw = base64.b64decode(encoded, validate=False)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _load_visual_path(entry: Any) -> torch.Tensor | None:
    if not isinstance(entry, dict):
        return None
    raw = str(entry.get("path") or "").strip()
    if not raw:
        return None
    input_root = Path(folder_paths.get_input_directory()).resolve()
    try:
        annotated = folder_paths.get_annotated_filepath(raw)
    except Exception:
        annotated = None
    candidate = Path(annotated).resolve() if annotated else (input_root / raw).resolve()
    try:
        candidate.relative_to(input_root)
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    try:
        image = Image.open(candidate).convert("RGB")
    except Exception:
        return None
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _prompter_references(project_data: Any) -> list[torch.Tensor]:
    project = _project(project_data)
    visuals = project.get("ai_visual_files") or project.get("visual_files") or []
    if not isinstance(visuals, list):
        return []
    result: list[torch.Tensor] = []
    for visual in visuals:
        image = _decode_visual(visual)
        if image is None:
            image = _load_visual_path(visual)
        if image is not None:
            result.append(image)
        if len(result) >= 4:
            break
    return result


def _task_mode(cine_linx: Any) -> str:
    settings = _settings_from_linx(cine_linx)
    return str(settings.get("task_mode") or "").strip().lower()


class IAMCCS_R42ReferenceImagePromptBridge:
    """Resolve the Sato prompt, exact 4K canvas and selected reference route."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "project_json": ("STRING", {"default": "{}", "forceInput": True}),
                "cine_linx": ("IAMCCS_SUPERNODE_LINX",),
            },
            "optional": {"direct_reference": ("IMAGE", {"lazy": True})},
        }

    RETURN_TYPES = (
        "STRING", "INT", "INT", "IMAGE", "BOOLEAN", "STRING", "FLOAT", "STRING",
        "IMAGE", "IMAGE", "IMAGE",
    )
    RETURN_NAMES = (
        "h3_prompt", "width", "height", "reference_image",
        "detail_enabled", "detail_model", "detail_strength", "branch_report",
        "reference_image_2", "reference_image_3", "reference_image_4",
    )
    FUNCTION = "adapt"
    CATEGORY = "IAMCCS/MiniMax H3/Reference"

    def check_lazy_status(self, prompt, project_json, cine_linx, direct_reference=None):
        settings = _settings_from_linx(cine_linx)
        source = str(settings.get("h3_reference_image_source") or "auto_prompter_then_direct")
        embedded = _prompter_references(project_json)
        if source == "direct_reference" and direct_reference is None:
            return ["direct_reference"]
        # AUTO also supports a text-only H3 still.  Requiring the optional
        # socket here made PAN evaluate a stale LoadImage even when the
        # Prompter project deliberately contained no visual reference.
        return []

    def adapt(self, prompt, project_json, cine_linx, direct_reference=None):
        settings = _settings_from_linx(cine_linx)
        task = str(settings.get("task_mode") or "").strip().lower()
        if task != "reference_image":
            raise ValueError(
                "R42 H3 Image Generator is controlled by IAMCCS Settings PRO: "
                f"select task_mode=reference_image (current={task or 'missing'})."
            )
        text = str(prompt or "").strip()
        if not text:
            raise ValueError("R42 H3 Image Generator: IAMCCS Prompter returned an empty final prompt.")

        preset = str(settings.get("h3_reference_image_preset") or "single_image")
        if preset == "character_sheet":
            suffix = (
                "One static character reference sheet on a single canvas: clearly separated, non-overlapping "
                "identity-consistent views with readable face, silhouette, wardrobe and material detail."
            )
        else:
            suffix = (
                "One static production reference image on a single canvas, with no temporal action, montage, "
                "multi-shot sequence or motion language."
            )
        if suffix.lower() not in text.lower():
            text = f"{text}\n\nOUTPUT CONTRACT:\n{suffix}"

        width = max(64, int(settings.get("width") or 3840))
        height = max(64, int(settings.get("height") or 2160))
        source = str(settings.get("h3_reference_image_source") or "auto_prompter_then_direct")
        embedded = _prompter_references(project_json)
        if source == "prompter_ref2v":
            references = embedded
            if not references:
                raise ValueError(
                    "Settings PRO selected PROMPTER REF2V, but IAMCCS Prompter has no attached visual. "
                    "Attach an image in Prompter, run AI, then Inject."
                )
            route = "prompter_ref2v+inject"
        elif source == "direct_reference":
            references = [direct_reference] if direct_reference is not None else []
            if not references:
                raise ValueError("Settings PRO selected DIRECT REFERENCE, but the direct image socket is empty.")
            route = "direct_reference"
        else:
            references = embedded if embedded else ([direct_reference] if direct_reference is not None else [])
            route = "prompter_ref2v+inject" if embedded else (
                "direct_reference" if direct_reference is not None else "text_only"
            )

        detail_enabled = bool(settings.get("h3_reference_image_detail_enabled", False))
        detail_model = str(settings.get("h3_reference_image_detail_model") or "None")
        if not detail_enabled:
            detail_model = "None"
        detail_strength = float(settings.get("h3_reference_image_detail_strength", 0.60))
        report = (
            f"R42 Universal H3 image | Settings PRO mode={task} | SatoDiveH3Image | "
            f"preset={preset} | exact={width}x{height} | reference={route} | "
            f"detail={'off' if detail_model == 'None' else detail_model}"
        )
        padded = (references + [None, None, None, None])[:4]
        return text, width, height, padded[0], detail_enabled, detail_model, detail_strength, report, padded[1], padded[2], padded[3]


class IAMCCS_H3OptionalSameSizeDetail:
    """Optional ESRGAN-style detail pass that returns to the exact input canvas."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "enabled": ("BOOLEAN", {"forceInput": True}),
            "model_name": ("STRING", {"forceInput": True}),
            "strength": ("FLOAT", {"forceInput": True}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "apply"
    CATEGORY = "IAMCCS/MiniMax H3/Reference"

    def apply(self, image, enabled, model_name, strength):
        name = str(model_name or "None")
        mix = max(0.0, min(1.0, float(strength)))
        if not enabled or name.lower() == "none" or mix <= 0.0:
            return image, "H3 image detail pass OFF · native Sato pixels preserved"
        from comfy_extras import nodes_upscale_model as upscale_nodes
        import comfy.utils

        model = upscale_nodes.UpscaleModelLoader().load_model(name)[0]
        detailed = upscale_nodes.ImageUpscaleWithModel().upscale(model, image)[0]
        _, height, width, _ = image.shape
        detailed = comfy.utils.common_upscale(
            detailed.movedim(-1, 1), width, height, "lanczos", "disabled"
        ).movedim(1, -1)
        output = torch.lerp(image, detailed.to(device=image.device, dtype=image.dtype), mix).clamp(0.0, 1.0)
        return output, f"H3 same-size detail · {name} · strength={mix:.2f} · canvas={width}x{height}"


class IAMCCS_H3ReferenceImageSaveR42:
    """Lazy output: request Sato only in Settings PRO reference_image mode."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": ("IAMCCS_SUPERNODE_LINX",),
                "filename_prefix": ("STRING", {"default": "IAMCCS/H3_REFERENCES/reference"}),
            },
            "optional": {"images": ("IMAGE", {"lazy": True})},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("reference_image", "report")
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "IAMCCS/MiniMax H3/Reference"

    def check_lazy_status(self, cine_linx, filename_prefix, images=None, **kwargs):
        if _task_mode(cine_linx) == "reference_image" and images is None:
            return ["images"]
        return []

    def save(self, cine_linx, filename_prefix, images=None, prompt=None, extra_pnginfo=None):
        task = _task_mode(cine_linx)
        if task != "reference_image":
            return {"ui": {}, "result": (None, f"R42 image branch idle · Settings PRO mode={task or 'missing'}")}
        if images is None or not torch.is_tensor(images) or len(images) < 1:
            raise ValueError("R42 H3 Image Generator produced no image tensor.")
        from nodes import SaveImage

        image = images[:1]
        saved = SaveImage().save_images(image, filename_prefix, prompt, extra_pnginfo)
        report = (
            f"H3 reference PNG | true R42 Universal contract | SatoDive | "
            f"exact={int(image.shape[2])}x{int(image.shape[1])}"
        )
        return {"ui": saved["ui"], "result": (image, report)}


def reference_image_plan(plan, prompt):
    """Compatibility helper retained for older tests and serialized workflows."""
    if not str(prompt).strip():
        raise ValueError("Reference Image: enter a global description of the image to generate.")
    result = copy.deepcopy(plan)
    result.update(task_mode="reference_image", requested_task_mode="reference_image")
    # The ShotPlanner enriches this contract with the Prompter/Shotboard paths
    # after the common video plan has been built.  Older R42 plans do not have
    # this namespace, so always create it here instead of making the caller
    # depend on a particular serialized workflow revision.
    result["reference_image"] = {
        "source_policy": "prompter_ref2v_then_references",
        "prompter_paths": [],
        "fallback_paths": [],
        "prompt": str(prompt).strip(),
    }
    return result
