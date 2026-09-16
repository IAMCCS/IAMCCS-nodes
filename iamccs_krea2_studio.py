"""IAMCCS Krea 2 Studio integration for ComfyUI.

The legacy v1 node remains available for old workflows.  New workflows use the
``iamccs.krea2studio.v12`` layout adapter, which translates the Advanced layer
stack one-to-one into Fedor's Krea2RegionalMultiLoRAV12 inputs.  It deliberately
does not implement a second regional-conditioning engine.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import node_helpers
import torch


LOGGER = logging.getLogger("IAMCCS.Krea2Studio")
CONTRACT_ID = "iamccs.krea2studio.v1"
V12_CONTRACT_ID = "iamccs.krea2studio.v12"
MAX_REGIONS = 16


def _disambiguate_regional_subject(description: str) -> str:
    """Resolve common noun/adjective ambiguity without rewriting user detail."""
    text = str(description or "").strip()
    return re.sub(
        r"^\s*(?:a\s+)?giant\b(?=\s*(?:[,.;]|$|with\b|wearing\b|standing\b|sitting\b))",
        "a gigantic humanoid giant character",
        text,
        count=1,
        flags=re.IGNORECASE,
    )


def _clamp(value: Any, minimum: float, maximum: float, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = fallback
    return max(minimum, min(maximum, number))


def _normalise_box(raw: Any) -> dict[str, float]:
    box = raw if isinstance(raw, dict) else {}
    x = _clamp(box.get("x", 0.0), 0.0, 1.0, 0.0)
    y = _clamp(box.get("y", 0.0), 0.0, 1.0, 0.0)
    width = _clamp(box.get("width", box.get("w", 1.0)), 0.01, 1.0, 1.0)
    height = _clamp(box.get("height", box.get("h", 1.0)), 0.01, 1.0, 1.0)
    width = min(width, 1.0 - x)
    height = min(height, 1.0 - y)
    return {"x": x, "y": y, "width": max(0.01, width), "height": max(0.01, height)}


def parse_regions(regions_json: str) -> list[dict[str, Any]]:
    """Parse and validate the public Krea Studio region contract."""
    try:
        payload = json.loads(regions_json or "[]")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Krea 2 Studio regions_json is invalid: {exc}") from exc
    if isinstance(payload, dict):
        payload = payload.get("regions", [])
    if not isinstance(payload, list):
        raise ValueError("Krea 2 Studio regions_json must contain a list of regions.")

    regions: list[dict[str, Any]] = []
    for index, item in enumerate(payload[:MAX_REGIONS]):
        if not isinstance(item, dict):
            continue
        enabled = bool(item.get("enabled", item.get("enable", True)))
        prompt = str(item.get("prompt", "") or "").strip()
        lora = str(item.get("lora", item.get("lora_name", "")) or "").strip()
        if lora.lower() == "none":
            lora = ""
        regions.append({
            "id": str(item.get("id", f"region_{index + 1}") or f"region_{index + 1}"),
            "name": str(item.get("name", f"Region {index + 1}") or f"Region {index + 1}"),
            "enabled": enabled,
            "prompt": prompt,
            "prompt_strength": _clamp(item.get("prompt_strength", 1.0), 0.0, 10.0, 1.0),
            "lora": lora,
            "lora_strength": _clamp(item.get("lora_strength", item.get("strength", 1.0)), -10.0, 10.0, 1.0),
            "feather": _clamp(item.get("feather", 0.08), 0.0, 0.5, 0.08),
            "priority": int(_clamp(item.get("priority", index), -100000, 100000, index)),
            "box": _normalise_box(item.get("box", item.get("bbox", {}))),
        })
    return regions


def build_v12_layout(
    scene_prompt: str,
    regions_json: str,
    canvas_width: int,
    canvas_height: int,
    spatial_anchor_lora: str,
    base_lora_strength: float = 1.0,
    spatial_anchor_strength: float = 0.001,
):
    """Translate the public IAMCCS document to the exact V12 row/box contract.

    Local V12 treats a row as spatially active only when it has a non-zero LoRA
    or an in-node reference, and it installs unified attention only when at
    least one LoRA matrix is present.  Text-only Advanced regions therefore use
    the installed Krea identity-edit LoRA at a deliberately negligible weight
    as an activation anchor.  The actual prompt ownership, token spans,
    attraction field and box routing are still all executed by V12 itself.
    """
    width = max(64, int(canvas_width))
    height = max(64, int(canvas_height))
    regions = parse_regions(regions_json)
    active = [region for region in regions if region["enabled"] and (region["prompt"] or region["lora"])]
    if not active:
        raise ValueError("Krea 2 Studio needs at least one enabled region with a prompt or LoRA.")

    anchor = str(spatial_anchor_lora or "").strip()
    if anchor.lower() == "none":
        anchor = ""
    prompt_only = [region for region in active if not region["lora"]]
    if prompt_only and not anchor:
        raise ValueError(
            "Krea 2 Studio V12 needs the installed krea2_identity_edit LoRA to activate "
            "Fedor spatial routing for prompt-only regions."
        )

    base = _clamp(base_lora_strength, -10.0, 10.0, 1.0)
    anchor_strength = max(1e-6, min(0.05, abs(float(spatial_anchor_strength or 0.001))))
    single_region_fallback = len(active) == 1
    builder_elements = []
    fedor_rows = []
    pixel_boxes = []
    debug_regions = []

    for index, region in enumerate(regions):
        box = region["box"]
        x0 = max(0, min(1000, round(box["x"] * 1000)))
        y0 = max(0, min(1000, round(box["y"] * 1000)))
        x1 = max(x0 + 1, min(1000, round((box["x"] + box["width"]) * 1000)))
        y1 = max(y0 + 1, min(1000, round((box["y"] + box["height"]) * 1000)))
        description = region["prompt"] or region["name"]
        builder_elements.append({
            "type": "obj",
            "bbox": [y0, x0, y1, x1],
            "desc": description,
        })
        pixel_boxes.append({
            "x": round(box["x"] * width),
            "y": round(box["y"] * height),
            "width": max(1, round(box["width"] * width)),
            "height": max(1, round(box["height"] * height)),
        })

        is_active = region["enabled"] and bool(region["prompt"] or region["lora"])
        uses_anchor = is_active and not region["lora"]
        routed_lora = anchor if uses_anchor else (region["lora"] or "None")
        routed_strength = (
            anchor_strength if uses_anchor
            else float(region["lora_strength"]) * base
        )
        routed_prompt = description
        if single_region_fallback and is_active:
            semantic_description = _disambiguate_regional_subject(description)
            routed_prompt = (
                f"{semantic_description}; mandatory distinct standalone regional subject; "
                "visually separate from the background, not fused with terrain, architecture "
                "or scenery; complete, clearly visible and contained within its assigned area"
            )
        fedor_rows.append({
            "name": region["name"],
            "prompt": routed_prompt,
            "lora": routed_lora,
            "strength": routed_strength,
            "enable": bool(is_active),
            "ref_image": "",
            "ref_enable": True,
            "portrait": False,
        })
        debug_regions.append({
            "index": index,
            "id": region["id"],
            "enabled": bool(is_active),
            "prompt": description,
            "routed_prompt": routed_prompt,
            "box": box,
            "routing": "v12_anchor" if uses_anchor else ("v12_lora" if is_active else "disabled"),
            "lora": routed_lora if is_active else "",
            "strength": routed_strength if is_active else 0.0,
        })

    global_prompt = str(scene_prompt or "").strip()
    if not global_prompt:
        global_prompt = "A coherent scene containing the assigned regional subjects"

    routed_global_prompt = global_prompt
    if single_region_fallback:
        routed_global_prompt = (
            "Background environment only, with no people, creatures or subject-shaped "
            f"structures: {global_prompt}"
        )

    builder_prompt = {
        # Keep subject text out of V12's un-owned global span.  The single-row
        # existence reinforcement lives entirely in routed_prompt above, so it
        # receives the same hard bbox ownership as the original regional text.
        "high_level_description": routed_global_prompt,
        "compositional_deconstruction": {
            "background": "Environment only. All enabled regional elements are mandatory, visible and spatially authoritative.",
            "elements": builder_elements,
        },
    }
    debug = {
        "contract": V12_CONTRACT_ID,
        "backend": "Krea2RegionalMultiLoRAV12",
        "engine": "v12_unified_spatial",
        "canvas": {"width": width, "height": height},
        "region_count": len(active),
        "row_count": len(regions),
        "anchor_lora": anchor,
        "anchor_strength": anchor_strength,
        "single_region_semantic_fallback": single_region_fallback,
        "routed_global_prompt": routed_global_prompt,
        "regions": debug_regions,
    }
    return (
        json.dumps(builder_prompt, ensure_ascii=False),
        json.dumps(fedor_rows, ensure_ascii=False),
        [pixel_boxes],
        json.dumps(debug, ensure_ascii=False),
    )


def _encode(clip: Any, text: str):
    if clip is None:
        raise RuntimeError("Krea 2 Studio received no CLIP/text encoder.")
    tokens = clip.tokenize(str(text or ""))
    return clip.encode_from_tokens_scheduled(tokens)


def _zero_conditioning(conditioning):
    result = []
    for tensor, metadata in conditioning:
        clean = dict(metadata or {})
        pooled = clean.get("pooled_output")
        if torch.is_tensor(pooled):
            clean["pooled_output"] = torch.zeros_like(pooled)
        result.append([torch.zeros_like(tensor), clean])
    return result


def _box_masks(regions: list[dict[str, Any]], width: int, height: int) -> dict[str, torch.Tensor]:
    """Build inward-feathered, priority-exclusive masks in canvas space."""
    px = (torch.arange(width, dtype=torch.float32) + 0.5) / float(width)
    py = (torch.arange(height, dtype=torch.float32) + 0.5) / float(height)
    grid_y, grid_x = torch.meshgrid(py, px, indexing="ij")
    claimed = torch.zeros((height, width), dtype=torch.bool)
    masks: dict[str, torch.Tensor] = {}

    # Higher priority is the visual top of the Advanced Layer Stack and wins
    # deterministic ownership wherever boxes overlap.
    for region in sorted(regions, key=lambda row: row["priority"], reverse=True):
        box = region["box"]
        x0, y0 = box["x"], box["y"]
        x1, y1 = x0 + box["width"], y0 + box["height"]
        hard = (grid_x >= x0) & (grid_x <= x1) & (grid_y >= y0) & (grid_y <= y1)
        feather = min(region["feather"], box["width"] * 0.3, box["height"] * 0.3)
        if feather > 1e-6:
            edge = torch.minimum(
                torch.minimum((grid_x - x0) / feather, (x1 - grid_x) / feather),
                torch.minimum((grid_y - y0) / feather, (y1 - grid_y) / feather),
            ).clamp(0.0, 1.0)
            soft = edge * hard.to(torch.float32)
        else:
            soft = hard.to(torch.float32)
        available = ~claimed
        masks[region["id"]] = (soft * available.to(torch.float32)).unsqueeze(0)
        claimed |= hard
    return masks


def _apply_optional_regional_loras(
    model: Any,
    clip: Any,
    regions: list[dict[str, Any]],
    canvas_width: int,
    canvas_height: int,
    seam_feather: float,
    base_strength: float,
):
    lora_regions = [
        region for region in sorted(regions, key=lambda row: row["priority"], reverse=True)
        if region["enabled"] and region["lora"] and abs(region["lora_strength"] * base_strength) > 1e-8
    ]
    if not lora_regions:
        return model, {"backend": "none", "adapters": []}

    # Resolve at execution time, after ComfyUI has registered every custom-node
    # package. This keeps IAMCCS-nodes import-safe when the optional MIT backend
    # is not installed and gives the application a clear actionable error only
    # when regional LoRAs are actually requested.
    import nodes

    backend_class = nodes.NODE_CLASS_MAPPINGS.get("Krea2RegionalMultiLoRA")
    if backend_class is None:
        raise RuntimeError(
            "Krea 2 Studio regional prompts are available, but regional LoRAs require "
            "Krea2RegionalMultiLoRA (Krea2-Multi-Character-Lora-Node-w-bounding-box)."
        )

    backend_rows = [
        {
            "name": region["name"],
            "lora": region["lora"],
            "strength": region["lora_strength"],
            "enable": True,
        }
        for region in lora_regions
    ]
    boxes = [
        {
            "x": region["box"]["x"],
            "y": region["box"]["y"],
            "width": region["box"]["width"],
            "height": region["box"]["height"],
        }
        for region in lora_regions
    ]
    result = backend_class().apply(
        model=model,
        clip=clip,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        regions_json=json.dumps(backend_rows),
        split_mode="bbox",
        seam_feather=seam_feather,
        blend_override=0.0,
        bboxes=boxes,
        base_strength=base_strength,
        include_background=True,
    )
    data = result[3] if len(result) > 3 and isinstance(result[3], dict) else {}
    return result[0], {"backend": "Krea2RegionalMultiLoRA", **data}


class IAMCCSKrea2StudioRegional:
    """Compile an Advanced regional-box document into Krea 2 conditioning."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "canvas_width": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 8}),
                "canvas_height": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 8}),
                "scene_prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "regions_json": ("STRING", {"multiline": True, "default": "[]"}),
                "seam_feather": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01}),
                "base_lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("model", "positive", "negative", "studio_data")
    FUNCTION = "compile"
    CATEGORY = "IAMCCS/Krea 2 Studio"
    DESCRIPTION = (
        "IAMCCS Krea 2 Studio v1. Compiles draggable Advanced regional boxes into "
        "priority-exclusive soft conditioning masks and optional per-region LoRA routing."
    )

    def compile(
        self,
        model,
        clip,
        canvas_width,
        canvas_height,
        scene_prompt,
        negative_prompt,
        regions_json,
        seam_feather,
        base_lora_strength,
    ):
        width = max(64, int(canvas_width))
        height = max(64, int(canvas_height))
        regions = parse_regions(regions_json)
        active = [region for region in regions if region["enabled"] and (region["prompt"] or region["lora"])]
        if not active:
            raise ValueError("Krea 2 Studio needs at least one enabled region with a prompt or LoRA.")

        global_text = str(scene_prompt or "").strip()
        if not global_text:
            global_text = "A coherent scene containing " + ", ".join(
                region["prompt"] or region["name"] for region in active
            )
        positive = list(_encode(clip, global_text))
        masks = _box_masks(active, width, height)
        for region in active:
            if not region["prompt"]:
                continue
            local = _encode(clip, region["prompt"])
            local = node_helpers.conditioning_set_values(local, {
                "mask": masks[region["id"]],
                # Krea 2 exposes a temporal latent dimension even for a still
                # image. ComfyUI 0.34's multidimensional bounds optimiser adds
                # a second batch/time axis and get_mask_aabb then receives a
                # 3-D slice. Keep the full conditioning area; the soft mask
                # still gates every regional token exactly as intended.
                "set_area_to_bounds": False,
                "mask_strength": region["prompt_strength"],
                "iamccs_krea2studio_region": region["id"],
            })
            positive.extend(local)

        negative_text = str(negative_prompt or "").strip()
        negative = _encode(clip, negative_text) if negative_text else _zero_conditioning(_encode(clip, global_text))
        patched_model, lora_data = _apply_optional_regional_loras(
            model,
            clip,
            active,
            width,
            height,
            _clamp(seam_feather, 0.0, 0.5, 0.08),
            _clamp(base_lora_strength, -10.0, 10.0, 1.0),
        )
        data = {
            "contract": CONTRACT_ID,
            "canvas": {"width": width, "height": height},
            "region_count": len(active),
            "regions": [
                {
                    "id": region["id"],
                    "name": region["name"],
                    "priority": region["priority"],
                    "box": region["box"],
                    "has_prompt": bool(region["prompt"]),
                    "has_lora": bool(region["lora"]),
                }
                for region in active
            ],
            "regional_lora": lora_data,
        }
        LOGGER.info(
            "Krea 2 Studio compiled %d regions at %dx%d; LoRA backend=%s",
            len(active), width, height, lora_data.get("backend", "none"),
        )
        return patched_model, positive, negative, json.dumps(data, ensure_ascii=False)


class IAMCCSKrea2StudioLayoutV12:
    """One-to-one layout adapter for Fedor's unified-spatial V12 node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "canvas_width": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 8}),
                "canvas_height": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 8}),
                "scene_prompt": ("STRING", {"multiline": True, "default": ""}),
                "regions_json": ("STRING", {"multiline": True, "default": "[]"}),
                "spatial_anchor_lora": ("STRING", {"default": "krea2_identity_edit_v1_2.safetensors"}),
                "base_lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "spatial_anchor_strength": ("FLOAT", {"default": 0.001, "min": 0.000001, "max": 0.05, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "BOUNDING_BOX", "STRING")
    RETURN_NAMES = ("builder_prompt", "fedor_regions_json", "bboxes", "studio_data")
    FUNCTION = "compile"
    CATEGORY = "IAMCCS/Krea 2 Studio"
    DESCRIPTION = (
        "IAMCCS Krea 2 Studio V12 layout adapter. Preserves layer order and "
        "boxes one-to-one, then delegates unified token ownership and spatial "
        "attention to Krea2RegionalMultiLoRAV12."
    )

    def compile(
        self,
        canvas_width,
        canvas_height,
        scene_prompt,
        regions_json,
        spatial_anchor_lora,
        base_lora_strength,
        spatial_anchor_strength,
    ):
        result = build_v12_layout(
            scene_prompt=scene_prompt,
            regions_json=regions_json,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            spatial_anchor_lora=spatial_anchor_lora,
            base_lora_strength=base_lora_strength,
            spatial_anchor_strength=spatial_anchor_strength,
        )
        LOGGER.info(
            "Krea 2 Studio delegated layout to Fedor V12 at %dx%d",
            max(64, int(canvas_width)), max(64, int(canvas_height)),
        )
        return result


NODE_CLASS_MAPPINGS = {
    "IAMCCSKrea2StudioRegional": IAMCCSKrea2StudioRegional,
    "IAMCCSKrea2StudioLayoutV12": IAMCCSKrea2StudioLayoutV12,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCSKrea2StudioRegional": "IAMCCS Krea 2 Studio Regional",
    "IAMCCSKrea2StudioLayoutV12": "IAMCCS Krea 2 Studio Layout (Fedor V12)",
}
