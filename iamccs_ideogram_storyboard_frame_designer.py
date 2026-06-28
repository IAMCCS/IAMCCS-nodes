import asyncio
import copy
import hashlib
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageOps

import folder_paths
import nodes as comfy_nodes

_GEMMA_ASSIST_DEFAULT_MODEL = "text_encoders\\gemma4_e4b_it_fp8_scaled.safetensors"
_GEMMA_ASSIST_CLIP_CACHE: Dict[Tuple[str, str], Any] = {}
_GEMMA_ASSIST_RUNNING = False
_GEMMA_ASSIST_ABORT_REQUESTED = False

# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com


def _ideoboard_input_signature(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _safe_json(raw: Any, fallback: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return copy.deepcopy(raw)
    text = str(raw or "").strip()
    if not text:
        return copy.deepcopy(fallback)
    candidates = [text]
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.lower().startswith("json"):
                part = part[4:].strip()
            if part:
                candidates.append(part)
    first, last = text.find("{"), text.rfind("}")
    if first >= 0 and last > first:
        candidates.append(text[first:last + 1])
    first, last = text.find("["), text.rfind("]")
    if first >= 0 and last > first:
        candidates.append(text[first:last + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return copy.deepcopy(fallback)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_hex(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    if not text.startswith("#"):
        text = f"#{text}"
    if len(text) == 4:
        text = f"#{text[1] * 2}{text[2] * 2}{text[3] * 2}"
    if len(text) != 7:
        return ""
    try:
        int(text[1:], 16)
    except Exception:
        return ""
    return text


def _palette_list(value: Any, fallback: List[str] | None = None) -> List[str]:
    fallback = list(fallback or [])
    if isinstance(value, list):
        items = value
    else:
        items = str(value or "").replace(";", ",").split(",")
    normalized: List[str] = []
    for item in items:
        color = _normalize_hex(item)
        if color and color not in normalized:
            normalized.append(color)
    return normalized or fallback


def _clamp_int(value: Any, minimum: int, maximum: int, default: int) -> int:
    try:
        cast = int(round(float(value)))
    except Exception:
        cast = int(default)
    return max(minimum, min(maximum, cast))


def _default_design() -> Dict[str, Any]:
    return json.loads(r'''{
    "schema": "iamccs.ideogram_storyboard_frame_designer",
    "schema_version": 1,
    "preset_key": "storyboard",
    "workflow_mode": "storyboard_grid",
    "grid_key": "story_2x3",
    "target_resolution_key": "hd_720",
    "json_export_mode": "json_perfect",
    "brief_to_json": {
        "brief": "",
        "instruction": "Enhance the current Ideogram JSON without changing layout, bbox coordinates, visible text, or panel count."
    },
    "gemma_assistant": {
        "enabled": false,
        "provider": "local_gemma",
        "mode": "full_json_enhance",
        "speed": "fast",
        "model": "text_encoders\\gemma4_e4b_it_fp8_scaled.safetensors",
        "selected_id": "",
        "target_field": "",
        "current_text": "",
        "brief": "",
        "request_ready": false
    },
    "mask_paint": {
        "brush_size": 48,
        "strokes": []
    },
    "canvas": {
        "width": 2560,
        "height": 2160,
        "aspect_label": "1280 x 720 - HD 16:9 target / Storyboard 2x3 canvas 2560x2160",
        "target_resolution_key": "hd_720",
        "target_width": 1280,
        "target_height": 720
    },
    "scene": {
        "high_level_description": "A six-panel 2x3 cinematic storyboard contact sheet set on a frozen exoplanet, following one exhausted astronaut in a strange partly unsealed expedition suit crossing black ice plains, ruined research vehicles, and distant alien towers under two visible planets in the sky.",
        "aesthetics": "extremely photoreal live-action science fiction cinema, tactile frozen surfaces, strange worn astronaut suit, partly open asymmetric suit collar, visible inner thermal layers, practical space gear, believable human fatigue, cinematic continuity",
        "lighting": "cold blue planetary daylight, low amber rim light on ice crystals, breath vapor, long shadows, readable eyes, crisp silhouettes, clean subject separation in every panel",
        "photo": "35mm anamorphic film stills, varied shot scale, wide exoplanet landscape, tight medium portrait, top-down equipment detail, medium action frame, long telephoto silhouette, macro visor reflection, fine analog grain",
        "medium": "photograph",
        "art_style": "",
        "color_palette": [
            "#06111A",
            "#557C8A",
            "#A8D8E8",
            "#E6E1C6",
            "#E87C45",
            "#07080A"
        ],
        "background": "Clean storyboard contact sheet with exactly two columns and three rows. Six separate 16:9 cinematic stills show different beats of the same frozen-planet sequence with thin panel borders, readable horizon lines, alien planets in the sky, and clear physical action."
    },
    "i2i": {
        "enabled": false,
        "denoise": 0.28,
        "low_sigma_start_step": 12,
        "scheduler_hint": "Storyboard grid generation. Use AutoCropGrid with 2 columns and 3 rows.",
        "source_mode": "canvas_composite"
    },
    "reference_mode": {
        "mode": "single",
        "label": "Single / Storyboard",
        "panel_count": 1,
        "target_index": 0,
        "locked_indices": []
    },
    "direct_prompt": {
        "enabled": false,
        "text": ""
    },
    "json_override": {
        "enabled": false,
        "text": ""
    },
    "items": [
        {
            "id": "panel_01",
            "kind": "obj",
            "label": "Panel 1 - Ice Plain Arrival",
            "text": "",
            "x": 0,
            "y": 0,
            "w": 500,
            "h": 333,
            "desc": "ICE PLAIN ARRIVAL: wide establishing shot of a frozen exoplanet plain at sunrise, black ice road crossing toward ruined alien towers, one small astronaut figure walking alone, two planets hanging in the pale sky, long shadow, cold blue atmosphere.",
            "color_palette": [
                "#557C8A",
                "#A8D8E8"
            ]
        },
        {
            "id": "panel_02",
            "kind": "obj",
            "label": "Panel 2 - Broken Suit Collar",
            "text": "",
            "x": 500,
            "y": 0,
            "w": 500,
            "h": 333,
            "desc": "BROKEN SUIT COLLAR: tight medium portrait of the same exhausted astronaut pulling a frost-covered breathing scarf across the mouth, strange expedition suit partly unsealed at the collar, inner thermal fabric visible, breath vapor around focused eyes.",
            "color_palette": [
                "#A8D8E8",
                "#E87C45"
            ]
        },
        {
            "id": "panel_03",
            "kind": "obj",
            "label": "Panel 3 - Frozen Nav Module",
            "text": "",
            "x": 0,
            "y": 333,
            "w": 500,
            "h": 334,
            "desc": "FROZEN NAV MODULE: top-down close detail of a cracked wrist computer, circular oxygen gauge, folded star map, metal sample case, and frost crystals on white-blue ice, precise object clarity.",
            "color_palette": [
                "#06111A",
                "#557C8A"
            ]
        },
        {
            "id": "panel_04",
            "kind": "obj",
            "label": "Panel 4 - Rover Drag",
            "text": "",
            "x": 500,
            "y": 333,
            "w": 500,
            "h": 334,
            "desc": "ROVER DRAG: medium wide action frame of the astronaut dragging a broken rover battery sled across an ice trench, bent posture, boots cutting into snow crust, torn suit panels and loose cables moving in the wind.",
            "color_palette": [
                "#557C8A",
                "#E6E1C6"
            ]
        },
        {
            "id": "panel_05",
            "kind": "obj",
            "label": "Panel 5 - Footprints To Antenna",
            "text": "",
            "x": 0,
            "y": 667,
            "w": 500,
            "h": 333,
            "desc": "FOOTPRINTS TO ANTENNA: long telephoto silhouette of the astronaut walking along a frozen ridge toward a collapsed research antenna, clear footprints through powder snow, giant ringed planet low over the horizon.",
            "color_palette": [
                "#A8D8E8",
                "#07080A"
            ]
        },
        {
            "id": "panel_06",
            "kind": "obj",
            "label": "Panel 6 - Visor Planet Reflection",
            "text": "",
            "x": 500,
            "y": 667,
            "w": 500,
            "h": 333,
            "desc": "VISOR PLANET REFLECTION: extreme close-up of a cracked astronaut visor, twin planets and broken towers reflected in the glass, frost on eyelashes, skin dust, sweat, sharp helmet texture, cinematic macro detail.",
            "color_palette": [
                "#07080A",
                "#A8D8E8"
            ]
        }
    ]
}''')


def _preset_library() -> Dict[str, Dict[str, Any]]:
    return {
        "storyboard": json.loads(r'''{
    "workflow_mode": "storyboard_grid",
    "grid_key": "story_2x3",
    "target_resolution_key": "hd_720",
    "canvas": {
        "width": 2560,
        "height": 2160,
        "aspect_label": "1280 x 720 - HD 16:9 target / Storyboard 2x3 canvas 2560x2160",
        "target_resolution_key": "hd_720",
        "target_width": 1280,
        "target_height": 720
    },
    "scene": {
        "high_level_description": "A six-panel 2x3 cinematic storyboard contact sheet set on a frozen exoplanet, following one exhausted astronaut in a strange partly unsealed expedition suit crossing black ice plains, ruined research vehicles, and distant alien towers under two visible planets in the sky.",
        "aesthetics": "extremely photoreal live-action science fiction cinema, tactile frozen surfaces, strange worn astronaut suit, partly open asymmetric suit collar, visible inner thermal layers, practical space gear, believable human fatigue, cinematic continuity",
        "lighting": "cold blue planetary daylight, low amber rim light on ice crystals, breath vapor, long shadows, readable eyes, crisp silhouettes, clean subject separation in every panel",
        "photo": "35mm anamorphic film stills, varied shot scale, wide exoplanet landscape, tight medium portrait, top-down equipment detail, medium action frame, long telephoto silhouette, macro visor reflection, fine analog grain",
        "medium": "photograph",
        "art_style": "",
        "color_palette": [
            "#06111A",
            "#557C8A",
            "#A8D8E8",
            "#E6E1C6",
            "#E87C45",
            "#07080A"
        ],
        "background": "Clean storyboard contact sheet with exactly two columns and three rows. Six separate 16:9 cinematic stills show different beats of the same frozen-planet sequence with thin panel borders, readable horizon lines, alien planets in the sky, and clear physical action."
    },
    "i2i": {
        "enabled": false,
        "denoise": 0.28,
        "low_sigma_start_step": 12,
        "scheduler_hint": "Storyboard grid generation. Use AutoCropGrid with 2 columns and 3 rows.",
        "source_mode": "canvas_composite"
    },
    "items": [
        {
            "id": "panel_01",
            "kind": "obj",
            "label": "Panel 1 - Ice Plain Arrival",
            "text": "",
            "x": 0,
            "y": 0,
            "w": 500,
            "h": 333,
            "desc": "ICE PLAIN ARRIVAL: wide establishing shot of a frozen exoplanet plain at sunrise, black ice road crossing toward ruined alien towers, one small astronaut figure walking alone, two planets hanging in the pale sky, long shadow, cold blue atmosphere.",
            "color_palette": [
                "#557C8A",
                "#A8D8E8"
            ]
        },
        {
            "id": "panel_02",
            "kind": "obj",
            "label": "Panel 2 - Broken Suit Collar",
            "text": "",
            "x": 500,
            "y": 0,
            "w": 500,
            "h": 333,
            "desc": "BROKEN SUIT COLLAR: tight medium portrait of the same exhausted astronaut pulling a frost-covered breathing scarf across the mouth, strange expedition suit partly unsealed at the collar, inner thermal fabric visible, breath vapor around focused eyes.",
            "color_palette": [
                "#A8D8E8",
                "#E87C45"
            ]
        },
        {
            "id": "panel_03",
            "kind": "obj",
            "label": "Panel 3 - Frozen Nav Module",
            "text": "",
            "x": 0,
            "y": 333,
            "w": 500,
            "h": 334,
            "desc": "FROZEN NAV MODULE: top-down close detail of a cracked wrist computer, circular oxygen gauge, folded star map, metal sample case, and frost crystals on white-blue ice, precise object clarity.",
            "color_palette": [
                "#06111A",
                "#557C8A"
            ]
        },
        {
            "id": "panel_04",
            "kind": "obj",
            "label": "Panel 4 - Rover Drag",
            "text": "",
            "x": 500,
            "y": 333,
            "w": 500,
            "h": 334,
            "desc": "ROVER DRAG: medium wide action frame of the astronaut dragging a broken rover battery sled across an ice trench, bent posture, boots cutting into snow crust, torn suit panels and loose cables moving in the wind.",
            "color_palette": [
                "#557C8A",
                "#E6E1C6"
            ]
        },
        {
            "id": "panel_05",
            "kind": "obj",
            "label": "Panel 5 - Footprints To Antenna",
            "text": "",
            "x": 0,
            "y": 667,
            "w": 500,
            "h": 333,
            "desc": "FOOTPRINTS TO ANTENNA: long telephoto silhouette of the astronaut walking along a frozen ridge toward a collapsed research antenna, clear footprints through powder snow, giant ringed planet low over the horizon.",
            "color_palette": [
                "#A8D8E8",
                "#07080A"
            ]
        },
        {
            "id": "panel_06",
            "kind": "obj",
            "label": "Panel 6 - Visor Planet Reflection",
            "text": "",
            "x": 500,
            "y": 667,
            "w": 500,
            "h": 333,
            "desc": "VISOR PLANET REFLECTION: extreme close-up of a cracked astronaut visor, twin planets and broken towers reflected in the glass, frost on eyelashes, skin dust, sweat, sharp helmet texture, cinematic macro detail.",
            "color_palette": [
                "#07080A",
                "#A8D8E8"
            ]
        }
    ]
}'''),
        "poster": {
            "canvas": {"width": 1024, "height": 1536, "aspect_label": "2:3 Poster"},
            "scene": {
                "high_level_description": "A striking cinematic poster image with a dominant focal subject, bold typography zones, and premium visual hierarchy.",
                "aesthetics": "premium theatrical poster design, dramatic scale, iconic silhouette, polished key art",
                "lighting": "high-contrast dramatic key art lighting with controlled glow, depth, and premium finish",
                "medium": "cinematic poster illustration",
                "background": "Graphic poster backdrop with atmospheric depth and clean open space for billing, taglines, and title treatment.",
                "color_palette": ["#111111", "#C84E2F", "#F5E6C8", "#4E6FAE"],
            },
        },
        "signage": {
            "canvas": {"width": 1536, "height": 864, "aspect_label": "16:9 Signage"},
            "scene": {
                "high_level_description": "A cinematic environment built around readable signage, branded surfaces, and strong in-world typography.",
                "aesthetics": "designed environmental graphics, high readability, urban production design, premium prop styling",
                "lighting": "motivated practical lighting that supports legibility on signs, surfaces, and surrounding space",
                "medium": "environment concept art",
                "background": "Architectural or environmental context that supports the sign as the hero graphic while keeping text readable.",
                "color_palette": ["#0E1A24", "#19A7CE", "#F6F1D1", "#D65A31"],
            },
        },
        "screen_ui": {
            "canvas": {"width": 1536, "height": 864, "aspect_label": "16:9 Screen UI"},
            "scene": {
                "high_level_description": "A diegetic screen composition with readable interface panels, cinematic reflections, and believable display design.",
                "aesthetics": "futuristic screen graphics, clean UI hierarchy, believable diegetic display design, premium sci-fi interface art",
                "lighting": "monitor-emissive lighting with controlled reflections and clean contrast for readable interface elements",
                "medium": "screen interface design",
                "background": "Physical screen housing or surrounding set elements that support the interface without obscuring key text zones.",
                "color_palette": ["#08121C", "#37D5D6", "#D9F3FF", "#F2A65A"],
            },
        },
        "title_card": {
            "canvas": {"width": 1536, "height": 864, "aspect_label": "16:9 Title Card"},
            "scene": {
                "high_level_description": "A cinematic title card frame designed around elegant typography, strong open space, and mood-driven art direction.",
                "aesthetics": "title card design, clean hierarchy, premium typography composition, deliberate open space",
                "lighting": "controlled atmosphere with subtle gradients and composition that preserves crisp title readability",
                "medium": "title card artwork",
                "background": "Minimal or atmospheric background treatment that elevates the title and avoids visual clutter.",
                "color_palette": ["#141414", "#8C1C13", "#F6E7CB", "#6C8EA3"],
            },
        },
    }


def _design_from_preset(key: Any) -> Dict[str, Any]:
    base = _default_design()
    preset_key = _clean_text(key).lower()
    preset = _preset_library().get(preset_key, _preset_library()[base["preset_key"]])
    design = copy.deepcopy(base)
    design["preset_key"] = preset_key if preset_key in _preset_library() else base["preset_key"]
    for field in ("workflow_mode", "grid_key", "target_resolution_key"):
        if field in preset:
            design[field] = copy.deepcopy(preset[field])
    design["canvas"].update(copy.deepcopy(preset.get("canvas") or {}))
    design["scene"].update(copy.deepcopy(preset.get("scene") or {}))
    if isinstance(preset.get("items"), list):
        design["items"] = copy.deepcopy(preset["items"])
    return design


def _item_from_bbox(entry: Dict[str, Any], index: int) -> Dict[str, Any]:
    # Ideogram bbox order is [ymin, xmin, ymax, xmax] on a 0-1000 grid.
    bbox = entry.get("bbox") if isinstance(entry.get("bbox"), list) else [100, 100, 800, 800]
    if len(bbox) != 4:
        bbox = [100, 100, 800, 800]
    ymin = _clamp_int(bbox[0], 0, 999, 100)
    xmin = _clamp_int(bbox[1], 0, 999, 100)
    ymax = _clamp_int(bbox[2], ymin + 1, 1000, 800)
    xmax = _clamp_int(bbox[3], xmin + 1, 1000, 800)
    x1, y1, x2, y2 = xmin, ymin, xmax, ymax
    kind = "text" if str(entry.get("type") or "obj").lower() == "text" else "obj"
    label = _clean_text(entry.get("label") or entry.get("text") or entry.get("type") or f"Element {index + 1}")
    return {
        "id": f"item_{index + 1:03d}",
        "kind": kind,
        "label": label or f"Element {index + 1}",
        "text": _clean_text(entry.get("text")) if kind == "text" else "",
        "x": x1,
        "y": y1,
        "w": max(20, x2 - x1),
        "h": max(20, y2 - y1),
        "desc": _clean_text(entry.get("desc") or entry.get("description") or label),
        "color_palette": _palette_list(entry.get("color_palette"), ["#FFE4B5", "#1A1A2E"]),
    }


def _from_ideogram_prompt(data: Dict[str, Any]) -> Dict[str, Any]:
    base = _default_design()
    style = data.get("style_description") if isinstance(data.get("style_description"), dict) else {}
    raw_comp = data.get("compositional_deconstruction")
    if isinstance(raw_comp, dict):
        comp = raw_comp
        items = comp.get("elements") if isinstance(comp.get("elements"), list) else []
    elif isinstance(raw_comp, list):
        comp = {}
        items = raw_comp
    else:
        comp = {}
        items = []
    base["scene"] = {
        "high_level_description": _clean_text(data.get("high_level_description")) or base["scene"]["high_level_description"],
        "aesthetics": _clean_text(style.get("aesthetics")) or base["scene"]["aesthetics"],
        "lighting": _clean_text(style.get("lighting")) or base["scene"]["lighting"],
        "photo": _clean_text(style.get("photo")),
        "medium": _clean_text(style.get("medium")) or base["scene"]["medium"],
        "art_style": _clean_text(style.get("art_style")),
        "color_palette": _palette_list(style.get("color_palette"), base["scene"]["color_palette"]),
        "background": _clean_text(comp.get("background")) or base["scene"]["background"],
    }
    clean_items = [item for item in items if isinstance(item, dict)]
    base["items"] = [_item_from_bbox(item, index) for index, item in enumerate(clean_items)] or base["items"]
    base["mask_paint"] = _normalize_mask_paint(data.get("mask_paint"), base.get("mask_paint"))
    return base


def _normalize_item(entry: Dict[str, Any], index: int) -> Dict[str, Any]:
    raw_kind = str(entry.get("kind") or entry.get("type") or "obj").lower()
    if raw_kind in {"text", "txt"}:
        kind = "text"
    elif raw_kind in {"image", "img", "reference", "source"}:
        kind = "image"
    elif raw_kind in {"mask", "inpaint_mask", "paint_mask"}:
        kind = "mask"
    else:
        kind = "obj"
    item = {
        "id": _clean_text(entry.get("id")) or f"item_{index + 1:03d}",
        "kind": kind,
        "label": _clean_text(entry.get("label") or entry.get("name") or f"Element {index + 1}"),
        "text": _clean_text(entry.get("text")),
        "x": _clamp_int(entry.get("x"), 0, 980, 100 + (index * 40)),
        "y": _clamp_int(entry.get("y"), 0, 980, 100 + (index * 30)),
        "w": _clamp_int(entry.get("w"), 20, 1000, 260),
        "h": _clamp_int(entry.get("h"), 20, 1000, 180),
        "desc": _clean_text(entry.get("desc") or entry.get("description") or f"Element {index + 1}"),
        "color_palette": _palette_list(entry.get("color_palette"), ["#FFE4B5", "#1A1A2E"]),
    }
    item["w"] = min(item["w"], 1000 - item["x"])
    item["h"] = min(item["h"], 1000 - item["y"])
    if kind != "text":
        item["text"] = ""
    if kind == "image":
        item["image_path"] = _clean_text(
            entry.get("image_path")
            or entry.get("imagePath")
            or entry.get("imageFile")
            or entry.get("path")
            or entry.get("file")
        )
        item["fit"] = str(entry.get("fit") or entry.get("resize_mode") or "cover").lower()
        if item["fit"] not in {"cover", "contain", "stretch"}:
            item["fit"] = "cover"
        item["opacity"] = max(0.0, min(1.0, float(entry.get("opacity", 1.0) or 1.0)))
    return item


def _normalize_i2i(raw: Any) -> Dict[str, Any]:
    data = raw if isinstance(raw, dict) else {}
    return {
        "enabled": bool(data.get("enabled", False)),
        "denoise": max(0.0, min(1.0, float(data.get("denoise", 0.28) or 0.28))),
        "low_sigma_start_step": _clamp_int(data.get("low_sigma_start_step"), 0, 1000, 12),
        "scheduler_hint": _clean_text(data.get("scheduler_hint"))
        or "Use SplitSigmasDenoise for denoise or split high/low sigmas by step for advanced i2i.",
        "source_mode": _clean_text(data.get("source_mode")) or "canvas_composite",
    }


def _normalize_mask_target(raw: Any) -> Dict[str, Any] | None:
    data = raw if isinstance(raw, dict) else {}
    kind = _clean_text(data.get("kind") or data.get("type")).lower()
    if kind == "canvas_full":
        return {
            "kind": "canvas_full",
            "x": 0,
            "y": 0,
            "w": 1000,
            "h": 1000,
        }
    if kind != "image_content":
        return None
    x = _clamp_int(data.get("x"), 0, 999, 0)
    y = _clamp_int(data.get("y"), 0, 999, 0)
    w = _clamp_int(data.get("w"), 1, max(1, 1000 - x), max(1, 1000 - x))
    h = _clamp_int(data.get("h"), 1, max(1, 1000 - y), max(1, 1000 - y))
    return {
        "kind": "image_content",
        "item_id": _clean_text(data.get("item_id") or data.get("itemId") or data.get("id")),
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "fit": _clean_text(data.get("fit") or "contain") or "contain",
    }


def _normalize_mask_paint(raw: Any, fallback: Any = None) -> Dict[str, Any]:
    data = raw if isinstance(raw, dict) else {}
    fallback_data = fallback if isinstance(fallback, dict) else {"brush_size": 48, "strokes": []}
    strokes: List[Dict[str, Any]] = []
    raw_strokes = data.get("strokes") if isinstance(data.get("strokes"), list) else fallback_data.get("strokes", [])
    for raw_stroke in raw_strokes if isinstance(raw_strokes, list) else []:
        if not isinstance(raw_stroke, dict):
            continue
        raw_points = raw_stroke.get("points") if isinstance(raw_stroke.get("points"), list) else []
        points: List[List[int]] = []
        for raw_point in raw_points:
            if not isinstance(raw_point, (list, tuple)) or len(raw_point) < 2:
                continue
            points.append([
                _clamp_int(raw_point[0], 0, 1000, 0),
                _clamp_int(raw_point[1], 0, 1000, 0),
            ])
        if not points:
            continue
        mode = _clean_text(raw_stroke.get("mode")).lower()
        shape = _clean_text(raw_stroke.get("shape") or "stroke").lower()
        if shape not in {"stroke", "lasso", "rect"}:
            shape = "stroke"
        stroke = {
            "mode": "erase" if mode in {"erase", "eraser"} else "paint",
            "shape": shape,
            "size": _clamp_int(raw_stroke.get("size"), 1, 240, data.get("brush_size", fallback_data.get("brush_size", 48))),
            "points": points,
        }
        target = _normalize_mask_target(raw_stroke.get("target"))
        if target:
            stroke["target"] = target
        strokes.append(stroke)
    return {
        "brush_size": _clamp_int(data.get("brush_size"), 1, 240, fallback_data.get("brush_size", 48)),
        "strokes": strokes,
    }


def _image_item_by_id(design: Dict[str, Any], item_id: str) -> Dict[str, Any] | None:
    wanted = _clean_text(item_id)
    if not wanted:
        return None
    for index, raw_item in enumerate(design.get("items") if isinstance(design.get("items"), list) else []):
        if not isinstance(raw_item, dict):
            continue
        item = _normalize_item(raw_item, index)
        if item.get("kind") == "image" and item.get("id") == wanted:
            return item
    return None


def _image_content_target_for_item(item: Dict[str, Any]) -> Dict[str, Any]:
    x = _clamp_int(item.get("x"), 0, 999, 0)
    y = _clamp_int(item.get("y"), 0, 999, 0)
    w = _clamp_int(item.get("w"), 1, max(1, 1000 - x), max(1, 1000 - x))
    h = _clamp_int(item.get("h"), 1, max(1, 1000 - y), max(1, 1000 - y))
    fit = _clean_text(item.get("fit") or "contain").lower() or "contain"
    if fit == "contain":
        resolved = _resolve_image_path(item.get("image_path", ""))
        try:
            if resolved:
                with Image.open(resolved) as im:
                    src_w, src_h = ImageOps.exif_transpose(im).size
                if src_w > 0 and src_h > 0 and w > 0 and h > 0:
                    box_aspect = w / h
                    image_aspect = src_w / src_h
                    if image_aspect > box_aspect:
                        content_h = w / image_aspect
                        y += int(round((h - content_h) / 2))
                        h = int(round(content_h))
                    else:
                        content_w = h * image_aspect
                        x += int(round((w - content_w) / 2))
                        w = int(round(content_w))
        except Exception:
            pass
    x = _clamp_int(x, 0, 999, 0)
    y = _clamp_int(y, 0, 999, 0)
    w = _clamp_int(w, 1, max(1, 1000 - x), max(1, 1000 - x))
    h = _clamp_int(h, 1, max(1, 1000 - y), max(1, 1000 - y))
    return {
        "kind": "image_content",
        "item_id": _clean_text(item.get("id")),
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "fit": fit,
    }


def _live_mask_target_for_design(design: Dict[str, Any], raw_target: Any) -> Dict[str, Any] | None:
    target = _normalize_mask_target(raw_target)
    if not target:
        return None
    if target.get("kind") == "canvas_full":
        return target
    item = _image_item_by_id(design, target.get("item_id", ""))
    if item:
        return _image_content_target_for_item(item)
    return target


def _normalize_reference_mode(raw: Any) -> Dict[str, Any]:
    data = raw if isinstance(raw, dict) else {}
    mode = _clean_text(data.get("mode") or "single").lower()
    if mode not in {"single", "character_diptych", "multi_ref_triptych"}:
        mode = "single"
    defaults = {
        "single": {
            "label": "Single / Storyboard",
            "panel_count": 1,
            "target_index": 0,
            "locked_indices": [],
        },
        "character_diptych": {
            "label": "Character Ref Diptych",
            "panel_count": 2,
            "target_index": 1,
            "locked_indices": [0],
        },
        "multi_ref_triptych": {
            "label": "Multi Ref Triptych",
            "panel_count": 3,
            "target_index": 2,
            "locked_indices": [0, 1],
        },
    }[mode]
    return {
        "mode": mode,
        "label": _clean_text(data.get("label")) or defaults["label"],
        "panel_count": _clamp_int(data.get("panel_count"), 1, 8, defaults["panel_count"]),
        "target_index": _clamp_int(data.get("target_index"), 0, 7, defaults["target_index"]),
        "locked_indices": [
            _clamp_int(value, 0, 7, 0)
            for value in (data.get("locked_indices") if isinstance(data.get("locked_indices"), list) else defaults["locked_indices"])
        ],
    }


def _reference_mask_and_crop(design: Dict[str, Any], width: int, height: int) -> Tuple[torch.Tensor, Tuple[int, int, int, int], str]:
    mode = _normalize_reference_mode(design.get("reference_mode"))
    panel_count = max(1, int(mode.get("panel_count") or 1))
    target_index = max(0, min(panel_count - 1, int(mode.get("target_index") or 0)))
    panel_w = max(1, width // panel_count)
    x = min(width - 1, target_index * panel_w)
    target_w = width - x if target_index == panel_count - 1 else panel_w
    mask = torch.zeros((height, width), dtype=torch.float32)
    if mode["mode"] == "single":
        mask[:, :] = 1.0
        crop = (0, 0, width, height)
    else:
        mask[:, x:x + target_w] = 1.0
        crop = (x, 0, target_w, height)
    report = f"reference_mode={mode['mode']} target_crop={crop[0]},{crop[1]},{crop[2]},{crop[3]}"
    return mask, crop, report


def _direct_prompt_from_scene(scene: Dict[str, Any]) -> str:
    parts = [
        scene.get("high_level_description"),
        scene.get("aesthetics"),
        scene.get("lighting"),
        scene.get("photo"),
        scene.get("medium"),
        scene.get("background"),
    ]
    return " ".join([_clean_text(part) for part in parts if _clean_text(part)])


def _manual_design_is_populated(raw: Any) -> bool:
    data = _safe_json(raw, {})
    if not isinstance(data, dict):
        return False
    if data.get("boards") and isinstance(data.get("boards"), dict):
        return True
    if data.get("high_level_description") and data.get("style_description") and data.get("compositional_deconstruction"):
        return True
    scene = data.get("scene") if isinstance(data.get("scene"), dict) else {}
    items = data.get("items") if isinstance(data.get("items"), list) else []
    base = _default_design()
    base_scene = base.get("scene", {})
    scene_text = " ".join(
        _clean_text(scene.get(key))
        for key in ("high_level_description", "aesthetics", "lighting", "photo", "medium", "background")
    )
    base_text = " ".join(
        _clean_text(base_scene.get(key))
        for key in ("high_level_description", "aesthetics", "lighting", "photo", "medium", "background")
    )
    if scene_text and scene_text != base_text:
        return True
    if items and json.dumps(items, sort_keys=True, ensure_ascii=False) != json.dumps(base.get("items", []), sort_keys=True, ensure_ascii=False):
        return True
    return False


def _should_use_incoming_ideoboard(ideoboard: Any, design_data: Any, stored_signature: str) -> bool:
    incoming_signature = _ideoboard_input_signature(ideoboard)
    if not incoming_signature:
        return False
    # Match KJ import_mode="always": a connected ideoboard/import_json is the
    # authoritative source when it changes. After execution the UI stores the
    # signature, so the same imported JSON does not re-apply endlessly.
    return incoming_signature != stored_signature


def _normalize_direct_prompt(raw: Any, scene: Dict[str, Any]) -> Dict[str, Any]:
    # Single/direct prompt is intentionally disabled. The node always conditions
    # from the structured Ideogram JSON assembled by the canvas/box fields.
    return {
        "enabled": False,
        "text": "",
    }


def _normalize_json_override(raw: Any) -> Dict[str, Any]:
    data = raw if isinstance(raw, dict) else {}
    return {
        "enabled": bool(data.get("enabled", False)),
        "text": _clean_text(data.get("text")),
    }


def _is_ideogram_prompt_json(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    comp = value.get("compositional_deconstruction")
    return isinstance(comp, (dict, list))


def _json_override_payload(design: Dict[str, Any]) -> Dict[str, Any] | None:
    override = _normalize_json_override(design.get("json_override"))
    if not override["enabled"] or not override["text"]:
        return None
    parsed = _safe_json(override["text"], {})
    return parsed if _is_ideogram_prompt_json(parsed) else None


def _design_with_json_override(design: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any] | None]:
    payload = _json_override_payload(design)
    if payload is None:
        return design, None
    converted = _from_ideogram_prompt(payload)
    for key in ("canvas", "i2i", "reference_mode", "workflow_mode", "preset_key", "json_export_mode", "brief_to_json", "gemma_assistant", "json_override", "mask_paint"):
        if key in design:
            converted[key] = copy.deepcopy(design[key])
    return converted, payload


def _design_from_runtime_source(source_data: Any, fallback_data: Any) -> Dict[str, Any]:
    parsed_source = _safe_json(source_data, {})
    design = _normalize_design(source_data)
    if _is_ideogram_prompt_json(parsed_source):
        fallback = _normalize_design(fallback_data)
        for key in ("canvas", "i2i", "reference_mode", "workflow_mode", "preset_key", "json_export_mode", "brief_to_json", "gemma_assistant", "mask_paint"):
            if key not in parsed_source and key in fallback:
                design[key] = copy.deepcopy(fallback[key])
    return design


def _conditioning_text_for_design(design: Dict[str, Any], prompt_json: str) -> str:
    return prompt_json


def _normalize_design(raw: Any) -> Dict[str, Any]:
    data = _safe_json(raw, _default_design())
    if isinstance(data, dict) and isinstance(data.get("boards"), dict):
        preset_key = _clean_text(data.get("active_preset_key") or data.get("preset_key")).lower()
        if preset_key not in _preset_library():
            preset_key = _default_design()["preset_key"]
        selected = data.get("boards", {}).get(preset_key)
        if isinstance(selected, dict):
            data = copy.deepcopy(selected)
            data["preset_key"] = preset_key
        else:
            data = _design_from_preset(preset_key)
    if isinstance(data, dict) and "compositional_deconstruction" in data and (
        "high_level_description" in data or "style_description" in data
    ):
        return _from_ideogram_prompt(data)
    base = _default_design()
    if not isinstance(data, dict):
        return base
    merged = copy.deepcopy(base)
    preset_key = _clean_text(data.get("preset_key")).lower()
    if preset_key in _preset_library():
        merged = _design_from_preset(preset_key)
    merged["canvas"].update(data.get("canvas") if isinstance(data.get("canvas"), dict) else {})
    merged["scene"].update(data.get("scene") if isinstance(data.get("scene"), dict) else {})
    merged["scene"]["color_palette"] = _palette_list(merged["scene"].get("color_palette"), base["scene"]["color_palette"])
    merged["i2i"] = _normalize_i2i(data.get("i2i") if isinstance(data.get("i2i"), dict) else base.get("i2i"))
    merged["json_export_mode"] = _clean_text(data.get("json_export_mode") or base.get("json_export_mode") or "json_perfect").lower()
    if merged["json_export_mode"] not in {"json_perfect", "standard"}:
        merged["json_export_mode"] = "json_perfect"
    merged["brief_to_json"] = _normalize_brief_to_json(data.get("brief_to_json") if isinstance(data.get("brief_to_json"), dict) else base.get("brief_to_json"))
    merged["gemma_assistant"] = _normalize_gemma_assistant(data.get("gemma_assistant") if isinstance(data.get("gemma_assistant"), dict) else base.get("gemma_assistant"))
    merged["mask_paint"] = _normalize_mask_paint(
        data.get("mask_paint") if isinstance(data.get("mask_paint"), dict) else base.get("mask_paint"),
        base.get("mask_paint"),
    )
    merged["reference_mode"] = _normalize_reference_mode(
        data.get("reference_mode") if isinstance(data.get("reference_mode"), dict) else base.get("reference_mode")
    )
    merged["json_override"] = _normalize_json_override(data.get("json_override") if isinstance(data.get("json_override"), dict) else base.get("json_override"))
    merged["direct_prompt"] = _normalize_direct_prompt(
        data.get("direct_prompt") if isinstance(data.get("direct_prompt"), dict) else {},
        merged["scene"],
    )
    if merged["reference_mode"]["mode"] != "single" or merged["json_export_mode"] == "json_perfect":
        merged["direct_prompt"]["enabled"] = False
    merged["items"] = [_normalize_item(item, index) for index, item in enumerate(data.get("items") if isinstance(data.get("items"), list) else [])] or merged.get("items") or base["items"]
    merged["canvas"]["width"] = _clamp_int(merged["canvas"].get("width"), 256, 16384, 1024)
    merged["canvas"]["height"] = _clamp_int(merged["canvas"].get("height"), 256, 16384, 1024)
    merged["preset_key"] = preset_key if preset_key in _preset_library() else base["preset_key"]
    workflow_mode = _clean_text(data.get("workflow_mode") or merged.get("workflow_mode") or base.get("workflow_mode")).lower()
    merged["workflow_mode"] = workflow_mode if workflow_mode in {
        "single_image",
        "image_refine",
        "storyboard_grid",
        "character_diptych",
        "multi_ref_triptych",
    } else base.get("workflow_mode", "storyboard_grid")
    merged["canvas"]["aspect_label"] = _clean_text(merged["canvas"].get("aspect_label")) or base["canvas"]["aspect_label"]
    return merged


def _bboxes_for_design(design: Dict[str, Any], width: int, height: int) -> List[List[Dict[str, int]]]:
    """KJ-compatible per-frame BoundingBox output for obj/text regions."""
    bbox_dicts: List[Dict[str, int]] = []
    for index, raw_item in enumerate(design.get("items") if isinstance(design.get("items"), list) else []):
        if not isinstance(raw_item, dict):
            continue
        item = _normalize_item(raw_item, index)
        if item.get("kind") in {"image", "mask"}:
            continue
        x = max(0, min(width - 1, round((item["x"] / 1000.0) * width)))
        y = max(0, min(height - 1, round((item["y"] / 1000.0) * height)))
        w = max(1, round((item["w"] / 1000.0) * width))
        h = max(1, round((item["h"] / 1000.0) * height))
        if x + w > width:
            w = max(1, width - x)
        if y + h > height:
            h = max(1, height - y)
        bbox_dicts.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h)})
    return [bbox_dicts] if bbox_dicts else []


def _paint_mask_for_design(design: Dict[str, Any], width: int, height: int) -> Tuple[torch.Tensor, str]:
    mask_img = Image.new("L", (int(width), int(height)), 0)
    draw = ImageDraw.Draw(mask_img)
    count = 0
    has_image_layer = False
    fallback_target_count = 0
    invalid_target_count = 0
    for index, raw_item in enumerate(design.get("items") if isinstance(design.get("items"), list) else []):
        if not isinstance(raw_item, dict):
            continue
        item = _normalize_item(raw_item, index)
        if item.get("kind") == "image":
            has_image_layer = True
        if item.get("kind") != "mask":
            continue
        x = int(round((item["x"] / 1000.0) * width))
        y = int(round((item["y"] / 1000.0) * height))
        w = max(1, int(round((item["w"] / 1000.0) * width)))
        h = max(1, int(round((item["h"] / 1000.0) * height)))
        x2 = min(width, x + w)
        y2 = min(height, y + h)
        shape = _clean_text(raw_item.get("shape") or raw_item.get("mask_shape") or "rect").lower()
        if shape in {"ellipse", "oval", "circle"}:
            draw.ellipse((x, y, x2, y2), fill=255)
        else:
            draw.rectangle((x, y, x2, y2), fill=255)
        count += 1
    paint = design.get("mask_paint") if isinstance(design.get("mask_paint"), dict) else {}
    strokes = paint.get("strokes") if isinstance(paint.get("strokes"), list) else []
    stroke_count = 0
    live_target_count = 0
    point_total = 0
    point_counts: List[int] = []
    scale = (float(width) + float(height)) / 2000.0
    for raw_stroke in strokes:
        if not isinstance(raw_stroke, dict):
            continue
        raw_points = raw_stroke.get("points") if isinstance(raw_stroke.get("points"), list) else []
        target = _live_mask_target_for_design(design, raw_stroke.get("target"))
        if target is None:
            if raw_stroke.get("target"):
                invalid_target_count += 1
            # Never discard painted strokes silently. The frontend image editor
            # stores points in normalized full-canvas space, so this fallback
            # keeps inpaint masks alive even if an older/foreign target payload
            # reaches the backend.
            target = {"kind": "canvas_full", "x": 0, "y": 0, "w": 1000, "h": 1000}
            fallback_target_count += 1
        if target is not None:
            live_target_count += 1
        points: List[Tuple[int, int]] = []
        for raw_point in raw_points:
            if not isinstance(raw_point, (list, tuple)) or len(raw_point) < 2:
                continue
            px = _clamp_int(raw_point[0], 0, 1000, 0)
            py = _clamp_int(raw_point[1], 0, 1000, 0)
            if target:
                canvas_x = target["x"] + int(round((px / 1000.0) * target["w"]))
                canvas_y = target["y"] + int(round((py / 1000.0) * target["h"]))
            else:
                canvas_x = px
                canvas_y = py
            points.append((int(round((canvas_x / 1000.0) * width)), int(round((canvas_y / 1000.0) * height))))
        if not points:
            continue
        point_total += len(points)
        point_counts.append(len(points))
        target_scale = scale
        if target:
            target_scale = ((target["w"] / 1000.0) * width + (target["h"] / 1000.0) * height) / 2000.0
        size = max(1, int(round(_clamp_int(raw_stroke.get("size"), 1, 240, paint.get("brush_size", 48)) * max(0.1, target_scale))))
        radius = max(1, size // 2)
        fill = 0 if _clean_text(raw_stroke.get("mode")).lower() in {"erase", "eraser"} else 255
        shape = _clean_text(raw_stroke.get("shape") or "stroke").lower()
        if shape == "rect" and len(points) >= 2:
            x1, y1 = points[0]
            x2, y2 = points[-1]
            draw.rectangle((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), fill=fill)
            stroke_count += 1
            continue
        if shape == "lasso" and len(points) >= 3:
            draw.polygon(points, fill=fill)
            stroke_count += 1
            continue
        def _draw_cap(cx: int, cy: int) -> None:
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=fill)

        if len(points) == 1:
            x, y = points[0]
            _draw_cap(x, y)
        else:
            for start, end in zip(points, points[1:]):
                draw.line((start, end), fill=fill, width=size)
            for idx, (x, y) in enumerate(points):
                if idx == 0 or idx == len(points) - 1 or idx % 8 == 0:
                    _draw_cap(x, y)
        stroke_count += 1
    arr = np.asarray(mask_img).astype(np.float32) / 255.0
    mask = torch.from_numpy(arr).unsqueeze(0)
    mean = float(arr.mean()) if arr.size else 0.0
    maximum = float(arr.max()) if arr.size else 0.0
    minimum = float(arr.min()) if arr.size else 0.0
    return mask, (
        f"paint_mask zones={count} brush_strokes={stroke_count} live_targets={live_target_count} points={point_total} "
        f"fallback_targets={fallback_target_count} invalid_targets={invalid_target_count} "
        f"shape=1x{int(height)}x{int(width)} mean={mean:.6f} min={minimum:.3f} max={maximum:.3f}"
    )


def _to_ideogram_prompt(design: Dict[str, Any]) -> Dict[str, Any]:
    scene = design.get("scene") if isinstance(design.get("scene"), dict) else {}
    items = design.get("items") if isinstance(design.get("items"), list) else []
    json_perfect = _clean_text(design.get("json_export_mode") or "json_perfect").lower() == "json_perfect"
    elements: List[Dict[str, Any]] = []
    for index, raw_item in enumerate(items):
        item = _normalize_item(raw_item if isinstance(raw_item, dict) else {}, index)
        if item.get("kind") in {"image", "mask"}:
            continue
        x1 = item["x"]
        y1 = item["y"]
        x2 = min(1000, x1 + item["w"])
        y2 = min(1000, y1 + item["h"])
        entry: Dict[str, Any] = {
            "type": "text" if item["kind"] == "text" else "obj",
            "bbox": [y1, x1, y2, x2],
        }
        if item["kind"] == "text":
            entry["text"] = item["text"] or item["label"]
        entry["desc"] = item["desc"]
        palette = _palette_list(item.get("color_palette"), [] if json_perfect else ["#FFE4B5", "#1A1A2E"])
        if palette or not json_perfect:
            entry["color_palette"] = palette
        elements.append(entry)

    prompt: Dict[str, Any] = {}
    high = _clean_text(scene.get("high_level_description"))
    if high or not json_perfect:
        prompt["high_level_description"] = high

    aesthetics = _clean_text(scene.get("aesthetics"))
    lighting = _clean_text(scene.get("lighting"))
    photo = _clean_text(scene.get("photo"))
    medium = _clean_text(scene.get("medium"))
    art_style = _clean_text(scene.get("art_style"))
    style_palette = _palette_list(scene.get("color_palette"), [] if json_perfect else ["#1A1A2E", "#FFE4B5"])
    if json_perfect:
        if aesthetics or lighting or photo or medium or art_style or style_palette:
            style: Dict[str, Any] = {"aesthetics": aesthetics, "lighting": lighting}
            if art_style and not photo:
                style["medium"] = medium
                style["art_style"] = art_style
            else:
                style["photo"] = photo
                style["medium"] = medium
            if style_palette:
                style["color_palette"] = style_palette
            prompt["style_description"] = style
    else:
        prompt["style_description"] = {
            "aesthetics": aesthetics,
            "lighting": lighting,
            "photo": photo,
            "medium": medium,
            "art_style": art_style,
            "color_palette": style_palette,
        }
    prompt["compositional_deconstruction"] = {
        "background": _clean_text(scene.get("background")),
        "elements": elements,
    }
    return prompt


def _normalize_brief_to_json(raw: Any) -> Dict[str, str]:
    data = raw if isinstance(raw, dict) else {}
    return {
        "brief": _clean_text(data.get("brief")),
        "instruction": _clean_text(data.get("instruction"))
        or "Enhance the current Ideogram JSON without changing layout, bbox coordinates, visible text, or panel count.",
    }


def _normalize_gemma_assistant(raw: Any) -> Dict[str, Any]:
    data = raw if isinstance(raw, dict) else {}
    mode = _clean_text(data.get("mode") or "full_json_enhance")
    if mode not in {"full_json_enhance", "selected_box_enhance", "brief_to_ideoboard", "prompt_critic", "field_enhance"}:
        mode = "full_json_enhance"
    speed = _clean_text(data.get("speed") or data.get("detail_mode") or "fast").lower()
    if speed not in {"fast", "detailed"}:
        speed = "fast"
    return {
        "enabled": bool(data.get("enabled", False)),
        "provider": "local_gemma",
        "mode": mode,
        "speed": speed,
        "model": _clean_text(data.get("model")) or _GEMMA_ASSIST_DEFAULT_MODEL,
        "selected_id": _clean_text(data.get("selected_id")),
        "target_field": _clean_text(data.get("target_field")),
        "current_text": _clean_text(data.get("current_text")),
        "brief": _clean_text(data.get("brief")),
        "request_ready": bool(data.get("request_ready", False)),
    }


def _gemma_json_request_for_design(design: Dict[str, Any], prompt_json: str) -> str:
    brief = _normalize_brief_to_json(design.get("brief_to_json"))
    assistant = _normalize_gemma_assistant(design.get("gemma_assistant"))
    mode = assistant["mode"] if assistant.get("request_ready") else "full_json_enhance"
    selected_id = assistant.get("selected_id") or ""
    direction = assistant.get("brief") or brief["brief"] or brief["instruction"]
    selected_item = {}
    for item in design.get("items", []) if isinstance(design.get("items"), list) else []:
        if _clean_text(item.get("id")) == selected_id:
            selected_item = item
            break

    contract = (
        "OUTPUT CONTRACT\n"
        "Return one valid JSON object only, no markdown.\n"
    )
    if mode == "prompt_critic":
        contract += (
            "Return {\"mode\":\"prompt_critic\",\"notes\":\"concise critique\",\"suggestions\":[\"specific improvement\", \"specific improvement\"]}.\n"
            "Do not rewrite the full prompt in critic mode.\n"
        )
    else:
        contract += (
            "Return {\"mode\":\"" + mode + "\",\"ideogram_prompt\":{...},\"notes\":\"short reason\"}.\n"
            "The ideogram_prompt must preserve the exact schema: high_level_description, style_description, compositional_deconstruction.\n"
            "Preserve every element count, type, visible text, and bbox coordinate exactly unless mode is brief_to_ideoboard and the user asks for a new board.\n"
            "For selected_box_enhance, improve only the selected element description inside the full returned ideogram_prompt.\n"
        )

    return (
        "You are an Ideogram 4 JSON prompt enhancer for IAMCCS FrameDesigner. Output JSON only.\n\n"
        "TASK\n"
        "Improve the supplied Ideogram 4 structured JSON for stronger cinematic visual quality, clearer subject descriptions, "
        "better material detail, lighting, camera language, and storyboard continuity.\n\n"
        "STRICT RULES\n"
        "- Use positive concrete visual language.\n"
        "- Do not add negative prompts, banned lists, or lists of things to avoid.\n"
        "- Bbox order is [ymin, xmin, ymax, xmax] on a 0-1000 grid. Do not convert to x/y order.\n"
        "- Keep color_palette arrays as hex colors when present.\n"
        "- Keep the result compatible with IAMCCS_IdeoTranslate.\n\n"
        f"MODE\n{mode}\n\n"
        f"SELECTED ELEMENT ID\n{selected_id or 'none'}\n\n"
        "SELECTED ELEMENT JSON\n"
        f"{json.dumps(selected_item, ensure_ascii=False, indent=2)}\n\n"
        f"USER BRIEF OR ENHANCEMENT DIRECTION\n{direction}\n\n"
        f"{contract}\n"
        "CURRENT IDEOGRAM JSON\n"
        f"{prompt_json}\n"
    )


def _gemma_assistant_system_prompt() -> str:
    return (
        "You are Gemma running as the IAMCCS FrameDesigner assistant for Ideogram 4 structured prompting. "
        "Return compact JSON only. No markdown. No commentary outside JSON. "
        "Use positive, specific visual language. Do not write negative prompts, banned lists, or 'do not' instructions. "
        "Preserve the Ideogram JSON structure when enhancing. Preserve bbox coordinates unless the user explicitly asks to build a new board. "
        "Bbox order is [ymin, xmin, ymax, xmax] on a 0-1000 layout grid. "
        "Prioritize cinematic clarity, subject identity, physical action, material detail, lighting, lens language, and readable composition. "
        "When the user gives a general natural-language brief, infer whether they want a single image, image refinement, or storyboard grid, then produce a complete FrameDesigner ideoboard JSON. "
        "When enhancing one field or one selected box, return only the improved text for that target plus brief notes. "
        "After the closing JSON brace, immediately end the answer with <end_of_turn>."
    )


def _gemma_assistant_user_prompt(
    design: Dict[str, Any],
    mode: str,
    brief: str,
    selected_id: str,
    target_field: str = "",
    current_text: str = "",
) -> str:
    normalized = _normalize_design(design)
    prompt_json = json.dumps(_to_ideogram_prompt(normalized), ensure_ascii=False, indent=2)
    selected_item = None
    for item in normalized.get("items", []):
        if _clean_text(item.get("id")) == selected_id:
            selected_item = item
            break
    selected_block = json.dumps(selected_item or {}, ensure_ascii=False, indent=2)
    field = _clean_text(target_field)
    text = _clean_text(current_text)
    if mode == "field_enhance":
        return (
            "ASSISTANT MODE\n"
            "field_enhance\n\n"
            "TARGET FIELD KEY\n"
            f"{field}\n\n"
            "CURRENT FIELD TEXT\n"
            f"{text}\n\n"
            "USER DIRECTION\n"
            f"{_clean_text(brief) or text or 'Improve this field for Ideogram 4.'}\n\n"
            "CURRENT SELECTED BOX\n"
            f"{selected_block}\n\n"
            "CURRENT IDEOGRAM PROMPT JSON FOR CONTEXT\n"
            f"{prompt_json}\n\n"
            "OUTPUT CONTRACT\n"
            "{"
            "\"mode\":\"field_enhance\","
            "\"field_key\":\"same target field key\","
            "\"selected_id\":\"selected box id if any\","
            "\"text\":\"improved replacement text only\","
            "\"notes\":\"short reason\""
            "}\n"
            "Improve only the target field. Do not rewrite unrelated fields. Use positive concrete visual language and preserve user intent. "
            "Keep the replacement text concise. End immediately after the JSON object with <end_of_turn>.\n"
        )
    return (
        "ASSISTANT MODE\n"
        f"{mode}\n\n"
        "USER DIRECTION\n"
        f"{_clean_text(brief) or 'Improve the current board for stronger Ideogram 4 results while preserving user intent.'}\n\n"
        "CURRENT SELECTED BOX\n"
        f"{selected_block}\n\n"
        "TARGET FIELD KEY, IF A FIELD BUTTON WAS USED\n"
        f"{field or 'none'}\n\n"
        "CURRENT FIELD TEXT, IF A FIELD BUTTON WAS USED\n"
        f"{text or 'none'}\n\n"
        "CURRENT FRAMEDESIGNER IDEOBOARD JSON\n"
        f"{json.dumps(normalized, ensure_ascii=False, indent=2)}\n\n"
        "CURRENT IDEOGRAM PROMPT JSON\n"
        f"{prompt_json}\n\n"
        "OUTPUT CONTRACT\n"
        "- For mode selected_box_enhance, output: {\"mode\":\"selected_box_enhance\",\"selected_id\":\"...\",\"desc\":\"improved positive visual description\",\"notes\":\"short reason\"}.\n"
        "- For mode prompt_critic, output: {\"mode\":\"prompt_critic\",\"notes\":\"concise critique\",\"suggestions\":[\"specific improvement\", \"specific improvement\"]}.\n"
        "- For mode full_json_enhance, output: {\"mode\":\"full_json_enhance\",\"ideogram_prompt\":{...},\"notes\":\"short reason\"}. Preserve all element bbox coordinates and count.\n"
        "- For mode brief_to_ideoboard, output: {\"mode\":\"brief_to_ideoboard\",\"ideoboard\":{...},\"notes\":\"short reason\"}. The ideoboard must use schema iamccs.ideogram_storyboard_frame_designer, canvas, scene, items, workflow_mode, grid_key, target_resolution_key.\n"
        "Use compact JSON. Avoid whitespace-heavy formatting. End immediately after the JSON object with <end_of_turn>.\n"
    )


def _strip_gemma_response_noise(raw: Any) -> str:
    text = _clean_text(raw)
    if not text:
        return ""
    for marker in ("<end_of_turn>", "<eos>", "</s>"):
        text = text.replace(marker, "")
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _extract_balanced_json_object(raw: str) -> Dict[str, Any]:
    text = _strip_gemma_response_noise(raw)
    if not text:
        return {}
    for start in [idx for idx, char in enumerate(text) if char == "{"]:
        depth = 0
        in_string = False
        escape = False
        for pos in range(start, len(text)):
            char = text[pos]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:pos + 1]
                    parsed = _safe_json(candidate, {})
                    if isinstance(parsed, dict):
                        return parsed
                    break
    return {}


def _extract_json_object(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    cleaned = _strip_gemma_response_noise(raw)
    parsed = _safe_json(cleaned, {})
    if isinstance(parsed, dict):
        return parsed
    parsed = _extract_balanced_json_object(cleaned)
    return parsed if isinstance(parsed, dict) else {}


def _apply_gemma_assistant_response(design: Dict[str, Any], response_payload: Dict[str, Any], selected_id: str, mode: str) -> Tuple[Dict[str, Any], str]:
    current = _normalize_design(design)
    payload = response_payload if isinstance(response_payload, dict) else {}
    notes = _clean_text(payload.get("notes") or payload.get("report"))
    response_mode = _clean_text(payload.get("mode") or mode)

    if response_mode == "prompt_critic":
        suggestions = payload.get("suggestions")
        if isinstance(suggestions, list) and suggestions:
            notes = (notes + " | " if notes else "") + " | ".join(_clean_text(x) for x in suggestions if _clean_text(x))
        return current, notes or "Gemma critic returned notes only."

    if response_mode == "field_enhance":
        return current, notes or "Gemma returned a field patch."

    if response_mode == "selected_box_enhance" or payload.get("desc"):
        target_id = _clean_text(payload.get("selected_id") or selected_id)
        desc = _clean_text(payload.get("desc") or payload.get("description"))
        if desc:
            for item in current.get("items", []):
                if _clean_text(item.get("id")) == target_id:
                    item["desc"] = desc
                    if item.get("kind") == "text" and payload.get("text") is not None:
                        item["text"] = _clean_text(payload.get("text"))
                    return current, notes or f"Enhanced selected box {target_id}."
        return current, notes or "Gemma returned no selected-box description to apply."

    ideoboard = payload.get("ideoboard") or payload.get("design_data") or payload.get("board")
    if isinstance(ideoboard, dict):
        next_design = _normalize_design(ideoboard)
        return next_design, notes or "Gemma returned a complete ideoboard."

    ideogram_prompt = payload.get("ideogram_prompt") or payload.get("prompt_json") or payload.get("prompt")
    if isinstance(ideogram_prompt, dict) and _is_ideogram_prompt_json(ideogram_prompt):
        converted = _from_ideogram_prompt(ideogram_prompt)
        for key in ("canvas", "i2i", "reference_mode", "workflow_mode", "preset_key", "json_export_mode", "gemma_assistant", "mask_paint"):
            if key in current:
                converted[key] = copy.deepcopy(current[key])
        return _normalize_design(converted), notes or "Gemma returned enhanced Ideogram prompt JSON."

    if _is_ideogram_prompt_json(payload):
        converted = _from_ideogram_prompt(payload)
        for key in ("canvas", "i2i", "reference_mode", "workflow_mode", "preset_key", "json_export_mode", "gemma_assistant", "mask_paint"):
            if key in current:
                converted[key] = copy.deepcopy(current[key])
        return _normalize_design(converted), notes or "Gemma returned raw Ideogram prompt JSON."

    return current, notes or "Gemma response was valid JSON but did not contain an applicable ideoboard, prompt, or box patch."


def _normalize_gemma_model_name(value: Any) -> str:
    model = _clean_text(value) or _GEMMA_ASSIST_DEFAULT_MODEL
    return model.replace("/", "\\")


def _gemma_model_candidates(value: Any) -> List[str]:
    model = _normalize_gemma_model_name(value)
    candidates = [model]
    lower = model.lower()
    if lower.startswith("text_encoders\\"):
        candidates.append(model.split("\\", 1)[1])
    else:
        candidates.append(f"text_encoders\\{model}")
    out: List[str] = []
    for candidate in candidates:
        if candidate and candidate not in out:
            out.append(candidate)
    return out


def _list_gemma_assist_models() -> List[str]:
    try:
        names = folder_paths.get_filename_list("text_encoders")
    except Exception:
        names = []
    cleaned: List[str] = []
    for name in names:
        text = _clean_text(name).replace("/", "\\")
        if text and text not in cleaned:
            cleaned.append(text)
    preferred = [name for name in cleaned if "gemma" in name.lower()]
    others = [name for name in cleaned if name not in preferred]
    return preferred + others


def _load_gemma_assist_clip(model: str, device: str = "default") -> Tuple[Any, str]:
    last_error: Exception | None = None
    for candidate in _gemma_model_candidates(model):
        key = (candidate, device)
        if key in _GEMMA_ASSIST_CLIP_CACHE:
            return _GEMMA_ASSIST_CLIP_CACHE[key], candidate
        try:
            clip = comfy_nodes.CLIPLoader().load_clip(candidate, "ideogram4", device)[0]
            _GEMMA_ASSIST_CLIP_CACHE[key] = clip
            return clip, candidate
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not load Gemma text encoder '{model}': {last_error}")


def _extract_node_output_text(output: Any) -> str:
    try:
        result = getattr(output, "result", None)
        if isinstance(result, (list, tuple)) and result:
            return _clean_text(result[0])
    except Exception:
        pass
    try:
        return _clean_text(output[0])
    except Exception:
        pass
    if isinstance(output, (list, tuple)) and output:
        return _clean_text(output[0])
    return _clean_text(output)


def _set_gemma_interrupt(value: bool) -> None:
    try:
        import comfy.model_management as model_management
        model_management.interrupt_current_processing(bool(value))
    except Exception:
        pass


def _gemma_token_budget(mode: str, speed: str, requested: Any = None) -> int:
    try:
        explicit = int(requested)
    except Exception:
        explicit = 0
    if explicit > 0:
        return max(64, min(1400, explicit))
    speed_key = _clean_text(speed).lower()
    if speed_key not in {"fast", "detailed"}:
        speed_key = "fast"
    budgets = {
        "field_enhance": {"fast": 96, "detailed": 160},
        "selected_box_enhance": {"fast": 140, "detailed": 240},
        "prompt_critic": {"fast": 160, "detailed": 260},
        "full_json_enhance": {"fast": 420, "detailed": 720},
        "brief_to_ideoboard": {"fast": 560, "detailed": 900},
    }
    return budgets.get(mode, budgets["full_json_enhance"])[speed_key]


def _call_local_gemma_generate(model: str, system: str, prompt: str, max_length: int = 1400) -> Tuple[str, str]:
    from comfy_extras.nodes_textgen import TextGenerate
    from server import PromptServer

    if _GEMMA_ASSIST_ABORT_REQUESTED:
        raise RuntimeError("Gemma assistant was stopped before generation started.")
    clip, loaded_model = _load_gemma_assist_clip(model)
    formatted_prompt = (
        f"<start_of_turn>system\n{system.strip()}<end_of_turn>\n"
        f"<start_of_turn>user\n{prompt.strip()}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    server_instance = getattr(PromptServer, "instance", None)
    old_prompt_id = getattr(server_instance, "last_prompt_id", None) if server_instance is not None else None
    old_node_id = getattr(server_instance, "last_node_id", None) if server_instance is not None else None
    had_prompt_id = hasattr(server_instance, "last_prompt_id") if server_instance is not None else False
    had_node_id = hasattr(server_instance, "last_node_id") if server_instance is not None else False
    if server_instance is not None:
        server_instance.last_prompt_id = "iamccs_gemma_assistant"
        server_instance.last_node_id = "iamccs_framedesigner_gemma"
    try:
        _set_gemma_interrupt(False)
        output = TextGenerate.execute(
            clip=clip,
            prompt=formatted_prompt,
            max_length=max(64, min(1400, int(max_length or 1400))),
            sampling_mode={"sampling_mode": "off"},
            thinking=False,
            use_default_template=False,
        )
    finally:
        if server_instance is not None:
            if had_prompt_id:
                server_instance.last_prompt_id = old_prompt_id
            else:
                try:
                    delattr(server_instance, "last_prompt_id")
                except Exception:
                    pass
            if had_node_id:
                server_instance.last_node_id = old_node_id
            else:
                try:
                    delattr(server_instance, "last_node_id")
                except Exception:
                    pass
    return _extract_node_output_text(output), loaded_model


def _field_patch_from_gemma_response(response_json: Dict[str, Any], raw_text: str, target_field: str, selected_id: str) -> Dict[str, str]:
    payload = response_json if isinstance(response_json, dict) else {}
    text = _clean_text(payload.get("text") or payload.get("replacement") or payload.get("desc") or payload.get("description"))
    if not text:
        text = _strip_gemma_response_noise(raw_text)
    if text.startswith("{") and text.endswith("}"):
        parsed = _extract_json_object(text)
        text = _clean_text(parsed.get("text") or parsed.get("replacement") or parsed.get("desc") or parsed.get("description"))
    return {
        "field_key": _clean_text(payload.get("field_key") or target_field),
        "selected_id": _clean_text(payload.get("selected_id") or selected_id),
        "text": text,
    }


def _register_framedesigner_gemma_route() -> None:
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return
    instance = getattr(PromptServer, "instance", None)
    routes = getattr(instance, "routes", None)
    if routes is None or getattr(instance, "_iamccs_framedesigner_gemma_route", False):
        return

    @routes.get("/iamccs/framedesigner/gemma_models")
    async def iamccs_framedesigner_gemma_models(request):
        models = _list_gemma_assist_models()
        default_model = _GEMMA_ASSIST_DEFAULT_MODEL
        if default_model not in models:
            for candidate in _gemma_model_candidates(default_model):
                if candidate in models:
                    default_model = candidate
                    break
        return web.json_response({
            "ok": True,
            "models": models,
            "default": default_model if default_model in models else (models[0] if models else _GEMMA_ASSIST_DEFAULT_MODEL),
        })

    @routes.post("/iamccs/framedesigner/gemma_abort")
    async def iamccs_framedesigner_gemma_abort(request):
        global _GEMMA_ASSIST_ABORT_REQUESTED
        _GEMMA_ASSIST_ABORT_REQUESTED = True
        _set_gemma_interrupt(True)
        return web.json_response({
            "ok": True,
            "running": bool(_GEMMA_ASSIST_RUNNING),
            "message": "Gemma assistant stop requested.",
        })

    @routes.post("/iamccs/framedesigner/gemma_assist")
    async def iamccs_framedesigner_gemma_assist(request):
        global _GEMMA_ASSIST_ABORT_REQUESTED, _GEMMA_ASSIST_RUNNING
        try:
            payload = await request.json()
            design = payload.get("design_data") if isinstance(payload, dict) else {}
            mode = _clean_text(payload.get("mode") if isinstance(payload, dict) else "") or "full_json_enhance"
            if mode not in {"full_json_enhance", "selected_box_enhance", "brief_to_ideoboard", "prompt_critic", "field_enhance"}:
                mode = "full_json_enhance"
            brief = _clean_text(payload.get("brief") if isinstance(payload, dict) else "")
            selected_id = _clean_text(payload.get("selected_id") if isinstance(payload, dict) else "")
            target_field = _clean_text(payload.get("target_field") if isinstance(payload, dict) else "")
            current_text = _clean_text(payload.get("current_text") if isinstance(payload, dict) else "")
            model = _normalize_gemma_model_name(payload.get("model") if isinstance(payload, dict) else "")
            speed = _clean_text(payload.get("speed") if isinstance(payload, dict) else "") or "fast"
            max_tokens = _gemma_token_budget(mode, speed, payload.get("max_tokens") if isinstance(payload, dict) else None)
            normalized = _normalize_design(design)
            system_prompt = _gemma_assistant_system_prompt()
            user_prompt = _gemma_assistant_user_prompt(normalized, mode, brief, selected_id, target_field, current_text)
            _GEMMA_ASSIST_ABORT_REQUESTED = False
            _GEMMA_ASSIST_RUNNING = True
            response_text, loaded_model = await asyncio.to_thread(
                _call_local_gemma_generate,
                model,
                system_prompt,
                user_prompt,
                max_tokens,
            )
            if _GEMMA_ASSIST_ABORT_REQUESTED:
                return web.json_response({"ok": False, "error": "Gemma assistant was stopped."}, status=409)
            response_json = _extract_json_object(response_text)
            if mode == "field_enhance":
                field_patch = _field_patch_from_gemma_response(response_json, response_text, target_field, selected_id)
                return web.json_response({
                    "ok": True,
                    "mode": mode,
                    "selected_id": selected_id,
                    "target_field": target_field,
                    "field_patch": field_patch,
                    "notes": _clean_text(response_json.get("notes")) or "Gemma enhanced the selected field.",
                    "raw_response": response_json or _strip_gemma_response_noise(response_text),
                    "raw_text": _strip_gemma_response_noise(response_text),
                    "model": loaded_model,
                    "speed": speed,
                    "max_tokens": max_tokens,
                })
            next_design, notes = _apply_gemma_assistant_response(normalized, response_json, selected_id, mode)
            prompt_json = json.dumps(_to_ideogram_prompt(next_design), ensure_ascii=False, indent=2)
            return web.json_response({
                "ok": True,
                "mode": mode,
                "selected_id": selected_id,
                "target_field": target_field,
                "design_data": next_design,
                "prompt_json": prompt_json,
                "notes": notes,
                "raw_response": response_json or _strip_gemma_response_noise(response_text),
                "raw_text": _strip_gemma_response_noise(response_text),
                "model": loaded_model,
                "speed": speed,
                "max_tokens": max_tokens,
            })
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        finally:
            _GEMMA_ASSIST_RUNNING = False
            _GEMMA_ASSIST_ABORT_REQUESTED = False
            _set_gemma_interrupt(False)

    instance._iamccs_framedesigner_gemma_route = True



def _resolve_image_path(path: str) -> str:
    clean = _clean_text(path).replace("\\", "/")
    if not clean:
        return ""
    if os.path.isabs(clean) and os.path.exists(clean):
        return clean
    candidates: List[str] = []
    try:
        candidates.append(folder_paths.get_annotated_filepath(clean))
    except Exception:
        pass
    for root_getter in (folder_paths.get_input_directory, folder_paths.get_output_directory, folder_paths.get_temp_directory):
        try:
            root = root_getter()
            candidates.append(os.path.join(root, clean))
        except Exception:
            pass
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return ""


def _fit_image_to_box(image: Image.Image, width: int, height: int, fit: str) -> Image.Image:
    width = max(1, int(width))
    height = max(1, int(height))
    fit = str(fit or "cover").lower()
    if fit == "stretch":
        return image.resize((width, height), Image.Resampling.LANCZOS)
    src_w, src_h = image.size
    if src_w <= 0 or src_h <= 0:
        return image.resize((width, height), Image.Resampling.LANCZOS)
    scale = max(width / src_w, height / src_h) if fit == "cover" else min(width / src_w, height / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    if fit == "contain":
        canvas = Image.new("RGB", (width, height), (0, 0, 0))
        canvas.paste(resized, ((width - new_w) // 2, (height - new_h) // 2))
        return canvas
    left = max(0, (new_w - width) // 2)
    top = max(0, (new_h - height) // 2)
    return resized.crop((left, top, left + width, top + height))


def _render_canvas_image(design: Dict[str, Any]) -> Tuple[torch.Tensor, str, bool]:
    canvas = design.get("canvas") if isinstance(design.get("canvas"), dict) else {}
    width = _clamp_int(canvas.get("width"), 256, 16384, 1024)
    height = _clamp_int(canvas.get("height"), 256, 16384, 1024)
    # i2i/inpaint source pixels must not inherit the UI/story style palette.
    # Only explicit image layers are rendered; the workspace background is mute.
    bg = Image.new("RGB", (width, height), (0, 0, 0))

    loaded = 0
    missing: List[str] = []
    for raw_item in design.get("items") if isinstance(design.get("items"), list) else []:
        if not isinstance(raw_item, dict):
            continue
        item = _normalize_item(raw_item, loaded)
        if item.get("kind") != "image":
            continue
        resolved = _resolve_image_path(item.get("image_path", ""))
        if not resolved:
            missing.append(item.get("image_path", ""))
            continue
        try:
            with Image.open(resolved) as im:
                source = ImageOps.exif_transpose(im).convert("RGB")
            x = int(round((item["x"] / 1000.0) * width))
            y = int(round((item["y"] / 1000.0) * height))
            w = max(1, int(round((item["w"] / 1000.0) * width)))
            h = max(1, int(round((item["h"] / 1000.0) * height)))
            fitted = _fit_image_to_box(source, w, h, item.get("fit", "cover"))
            opacity = float(item.get("opacity", 1.0) or 1.0)
            if opacity < 0.999:
                bg.paste(Image.blend(Image.new("RGB", fitted.size), fitted, opacity), (x, y))
            else:
                bg.paste(fitted, (x, y))
            loaded += 1
        except Exception:
            missing.append(item.get("image_path", ""))

    arr = np.asarray(bg).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr)[None,]
    report = f"i2i muted source image layers={loaded}"
    if missing:
        report += f" | missing={len(missing)}"
    return tensor, report, loaded > 0




def _fit_pil_to_box_generic(image: Image.Image, width: int, height: int, fit: str, fill=0) -> Image.Image:
    fit = str(fit or "cover").lower()
    if fit == "stretch":
        return image.resize((width, height), Image.Resampling.LANCZOS)
    src_w, src_h = image.size
    if src_w <= 0 or src_h <= 0:
        return image.resize((width, height), Image.Resampling.LANCZOS)
    scale = max(width / src_w, height / src_h) if fit == "cover" else min(width / src_w, height / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    if fit == "contain":
        canvas = Image.new(image.mode, (width, height), fill)
        canvas.paste(resized, ((width - new_w) // 2, (height - new_h) // 2))
        return canvas
    left = max(0, (new_w - width) // 2)
    top = max(0, (new_h - height) // 2)
    return resized.crop((left, top, left + width, top + height))


def _image_batch_to_pil_list(image: torch.Tensor) -> List[Image.Image]:
    if image is None or not hasattr(image, "shape") or len(image.shape) != 4:
        raise ValueError("IMAGE tensor must have shape [B,H,W,C]")
    result = []
    for frame in image.detach().cpu().clamp(0, 1).numpy():
        result.append(Image.fromarray((frame * 255.0).astype(np.uint8)).convert("RGB"))
    return result


def _mask_batch_to_pil_list(mask: torch.Tensor, count: int) -> List[Image.Image]:
    if mask is None or not hasattr(mask, "shape"):
        raise ValueError("MASK tensor is required for inpainting")
    m = mask.detach().cpu().float().clamp(0, 1)
    if len(m.shape) == 2:
        m = m[None, :, :]
    if len(m.shape) == 4:
        m = m[..., 0]
    if len(m.shape) != 3:
        raise ValueError("MASK tensor must have shape [H,W], [B,H,W], or [B,H,W,1]")
    result = []
    arr = m.numpy()
    for index in range(count):
        frame = arr[min(index, arr.shape[0] - 1)]
        result.append(Image.fromarray((frame * 255.0).astype(np.uint8)).convert("L"))
    return result


def _pil_images_to_tensor(images: List[Image.Image]) -> torch.Tensor:
    arrays = [np.asarray(img.convert("RGB")).astype(np.float32) / 255.0 for img in images]
    return torch.from_numpy(np.stack(arrays, axis=0))


def _pil_masks_to_tensor(masks: List[Image.Image]) -> torch.Tensor:
    arrays = [np.asarray(mask.convert("L")).astype(np.float32) / 255.0 for mask in masks]
    return torch.from_numpy(np.stack(arrays, axis=0))

def _image_tensor_to_canvas(source_image: torch.Tensor, width: int, height: int) -> Tuple[torch.Tensor, str, bool]:
    try:
        if source_image is None or not hasattr(source_image, "shape") or len(source_image.shape) != 4:
            raise ValueError("source_image must be an IMAGE tensor [N,H,W,C]")
        image = source_image[0].detach().cpu().clamp(0, 1).numpy()
        pil = Image.fromarray((image * 255.0).astype(np.uint8)).convert("RGB")
        fitted = _fit_image_to_box(pil, int(width), int(height), "cover")
        arr = np.asarray(fitted).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr)[None,]
        return tensor, f"direct source_image guide fitted to {int(width)}x{int(height)}", True
    except Exception as exc:
        fallback = Image.new("RGB", (max(1, int(width)), max(1, int(height))), "#111111")
        arr = np.asarray(fallback).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr)[None,]
        return tensor, f"source_image guide failed: {exc}", False


class IAMCCS_StoryboardFrameDesigner:
    """Storyboarding and in-frame text director for Ideogram-style structured prompting."""

    DEFAULT_DATA = json.dumps(_design_from_preset("storyboard"), ensure_ascii=False, indent=2)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "design_data": (
                    "STRING",
                    {
                        "default": cls.DEFAULT_DATA,
                        "multiline": True,
                        "tooltip": "Editable working copy used by the IAMCCS StoryboardFrame canvas.",
                    },
                ),
                "ideoboard_input_signature": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Internal guard that lets manual canvas edits survive while the same IdeoTranslate input remains connected.",
                    },
                ),
            },
            "optional": {
                "ideoboard": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "forceInput": True,
                        "tooltip": "Connect IAMCCS_IdeoTranslate.design_data_json or an ideoboard package here.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("positive", "prompt_json", "width", "height", "gemma_json_request")
    FUNCTION = "encode"
    CATEGORY = "IAMCCS/Cine/Ideogram"

    @staticmethod
    def _basic_encode(clip, text: str):
        result = comfy_nodes.CLIPTextEncode().encode(clip, str(text or ""))
        return result[0] if isinstance(result, tuple) else result

    def encode(self, clip, design_data, ideoboard_input_signature="", ideoboard=""):
        incoming_signature = _ideoboard_input_signature(ideoboard)
        stored_signature = _clean_text(ideoboard_input_signature)
        use_incoming = _should_use_incoming_ideoboard(ideoboard, design_data, stored_signature)
        source_data = ideoboard if use_incoming else design_data

        design = _design_from_runtime_source(source_data, design_data) if use_incoming else _normalize_design(source_data)
        design, prompt_override = _design_with_json_override(design)
        design_data_json = json.dumps(design, ensure_ascii=False, indent=2)
        prompt_json = json.dumps(prompt_override if prompt_override is not None else _to_ideogram_prompt(design), ensure_ascii=False, indent=2)
        gemma_json_request = _gemma_json_request_for_design(design, prompt_json)
        conditioning_text = _conditioning_text_for_design(design, prompt_json)
        conditioning = self._basic_encode(clip, conditioning_text)
        canvas_info = design.get("canvas", {}) if isinstance(design.get("canvas"), dict) else {}
        width = int(canvas_info.get("width") or 1024)
        height = int(canvas_info.get("height") or 1024)
        target_panel_width = _clamp_int(canvas_info.get("target_width"), 1, 16384, width)
        target_panel_height = _clamp_int(canvas_info.get("target_height"), 1, 16384, height)
        return {
            "ui": {
                "design_data": [design_data_json],
                "ideoboard_input_signature": [incoming_signature or stored_signature],
                "used_ideoboard_input": [use_incoming],
                "width": [width],
                "height": [height],
            },
            "result": (conditioning, prompt_json, width, height, gemma_json_request),
        }


IDEO_INFO_RETURN_TYPES = (
    "CONDITIONING",
    "STRING",
    "INT",
    "INT",
    "IMAGE",
    "BOOLEAN",
    "FLOAT",
    "INT",
    "STRING",
    "MASK",
    "INT",
    "INT",
    "INT",
    "INT",
    "STRING",
    "BOUNDING_BOX",
    "STRING",
    "INT",
    "INT",
    "MASK",
    "STRING",
)
IDEO_INFO_RETURN_NAMES = (
    "positive",
    "prompt_json",
    "width",
    "height",
    "i2i_canvas_image",
    "i2i_enabled",
    "i2i_denoise",
    "low_sigma_start_step",
    "i2i_report",
    "reference_noise_mask",
    "target_x",
    "target_y",
    "target_width",
    "target_height",
    "reference_mode",
    "bboxes",
    "gemma_json_request",
    "target_panel_width",
    "target_panel_height",
    "paint_mask",
    "debug_report",
)


def _make_ideo_linx(**values: Any) -> Dict[str, Any]:
    return {name: values.get(name) for name in IDEO_INFO_RETURN_NAMES}




class IAMCCS_StoryboardFrameDesignerV2(IAMCCS_StoryboardFrameDesigner):
    """StoryboardFrame V2 with editable image layers and i2i/SDEdit metadata outputs."""

    @classmethod
    def INPUT_TYPES(cls):
        spec = copy.deepcopy(super().INPUT_TYPES())
        spec["required"]["i2i_enabled"] = ("BOOLEAN", {"default": False})
        spec["required"]["i2i_denoise"] = ("FLOAT", {"default": 0.28, "min": 0.0, "max": 1.0, "step": 0.01})
        spec["required"]["low_sigma_start_step"] = ("INT", {"default": 12, "min": 0, "max": 1000, "step": 1})
        spec.setdefault("optional", {})
        spec["optional"]["source_image"] = (
            "IMAGE",
            {
                "tooltip": "Optional direct image guide. When connected, V2 uses this IMAGE tensor as the i2i canvas guide instead of file-based image layers.",
            },
        )
        return spec

    RETURN_TYPES = ("IDEO_LINX",)
    RETURN_NAMES = ("IDEO_LINX",)
    FUNCTION = "encode_v2"
    CATEGORY = "IAMCCS/Cine/Ideogram"

    def encode_v2(
        self,
        clip,
        design_data,
        ideoboard_input_signature="",
        i2i_enabled=False,
        i2i_denoise=0.28,
        low_sigma_start_step=12,
        ideoboard="",
        source_image=None,
    ):
        incoming_signature = _ideoboard_input_signature(ideoboard)
        stored_signature = _clean_text(ideoboard_input_signature)
        use_incoming = _should_use_incoming_ideoboard(ideoboard, design_data, stored_signature)
        source_data = ideoboard if use_incoming else design_data

        design = _design_from_runtime_source(source_data, design_data) if use_incoming else _normalize_design(source_data)
        design, prompt_override = _design_with_json_override(design)
        i2i = _normalize_i2i(design.get("i2i"))
        i2i["enabled"] = bool(i2i_enabled) or bool(i2i.get("enabled"))
        i2i["denoise"] = max(0.0, min(1.0, float(i2i_denoise if i2i_denoise is not None else i2i.get("denoise", 0.28))))
        i2i["low_sigma_start_step"] = _clamp_int(low_sigma_start_step, 0, 1000, i2i.get("low_sigma_start_step", 12))
        design["i2i"] = i2i

        design_data_json = json.dumps(design, ensure_ascii=False, indent=2)
        prompt_json = json.dumps(prompt_override if prompt_override is not None else _to_ideogram_prompt(design), ensure_ascii=False, indent=2)
        gemma_json_request = _gemma_json_request_for_design(design, prompt_json)
        conditioning_text = _conditioning_text_for_design(design, prompt_json)
        conditioning = self._basic_encode(clip, conditioning_text)
        canvas_info = design.get("canvas", {}) if isinstance(design.get("canvas"), dict) else {}
        width = int(canvas_info.get("width") or 1024)
        height = int(canvas_info.get("height") or 1024)
        target_panel_width = _clamp_int(canvas_info.get("target_width"), 1, 16384, width)
        target_panel_height = _clamp_int(canvas_info.get("target_height"), 1, 16384, height)
        if source_image is not None:
            image, report, has_image = _image_tensor_to_canvas(source_image, width, height)
        else:
            image, report, has_image = _render_canvas_image(design)
        bboxes = _bboxes_for_design(design, width, height)
        reference_mask, target_crop, reference_report = _reference_mask_and_crop(design, width, height)
        paint_mask, paint_mask_report = _paint_mask_for_design(design, width, height)
        report += f" | {reference_report}"
        report += f" | {paint_mask_report}"
        normalized_mask_paint = _normalize_mask_paint(design.get("mask_paint"), _default_design().get("mask_paint"))
        reference_mode_key = _normalize_reference_mode(design.get("reference_mode"))["mode"]
        effective_i2i = bool(i2i["enabled"] and (has_image or reference_mode_key != "single"))
        if i2i["enabled"] and not has_image:
            report += " | i2i requested but no valid image layer found"
        report += " | structured JSON conditioning active"
        report += " | use SplitSigmasDenoise.denoise or high/low sigmas start step"
        debug_report = json.dumps({
            "used_ideoboard_input": bool(use_incoming),
            "incoming_signature": incoming_signature,
            "width": int(width),
            "height": int(height),
            "i2i_enabled": bool(i2i["enabled"]),
            "effective_i2i": bool(effective_i2i),
            "i2i_denoise": float(i2i["denoise"]),
            "source_image_connected": source_image is not None,
            "has_image": bool(has_image),
            "bboxes": len(bboxes[0]) if bboxes else 0,
            "mask_paint_strokes": len(normalized_mask_paint.get("strokes") or []),
            "mask_paint_point_counts": [len(stroke.get("points") or []) for stroke in (normalized_mask_paint.get("strokes") or []) if isinstance(stroke, dict)],
            "mask_paint_target_kinds": [
                _clean_text((stroke.get("target") or {}).get("kind") if isinstance(stroke.get("target"), dict) else "")
                for stroke in (normalized_mask_paint.get("strokes") or [])
                if isinstance(stroke, dict)
            ][:12],
            "mask_paint_total_points": sum(len(stroke.get("points") or []) for stroke in (normalized_mask_paint.get("strokes") or []) if isinstance(stroke, dict)),
            "paint_mask_shape": list(paint_mask.shape) if hasattr(paint_mask, "shape") else [],
            "paint_mask_mean": float(paint_mask.detach().float().mean().item()) if hasattr(paint_mask, "detach") else 0.0,
            "paint_mask_max": float(paint_mask.detach().float().max().item()) if hasattr(paint_mask, "detach") else 0.0,
            "paint_mask": paint_mask_report,
            "reference_mode": reference_mode_key,
            "report": report,
        }, ensure_ascii=False, indent=2)

        return {
            "ui": {
                "design_data": [design_data_json],
                "ideoboard_input_signature": [incoming_signature or stored_signature],
                "used_ideoboard_input": [use_incoming],
                "width": [width],
                "height": [height],
                "i2i_report": [report],
            },
            "result": (
                _make_ideo_linx(
                    positive=conditioning,
                    prompt_json=prompt_json,
                    width=width,
                    height=height,
                    i2i_canvas_image=image,
                    i2i_enabled=effective_i2i,
                    i2i_denoise=float(i2i["denoise"]),
                    low_sigma_start_step=int(i2i["low_sigma_start_step"]),
                    i2i_report=report,
                    reference_noise_mask=reference_mask,
                    target_x=int(target_crop[0]),
                    target_y=int(target_crop[1]),
                    target_width=int(target_crop[2]),
                    target_height=int(target_crop[3]),
                    reference_mode=_normalize_reference_mode(design.get("reference_mode"))["mode"],
                    bboxes=bboxes,
                    gemma_json_request=gemma_json_request,
                    target_panel_width=int(target_panel_width),
                    target_panel_height=int(target_panel_height),
                    paint_mask=paint_mask,
                    debug_report=debug_report,
                ),
            ),
        }


class IAMCCS_IdeoInfo:
    """Break out a FrameDesigner IDEO_LINX bundle into the legacy technical sockets."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "IDEO_LINX": (
                    "IDEO_LINX",
                    {
                        "tooltip": "Connect the single IDEO_LINX output from IAMCCS StoryboardFrame V2. This node exposes the same technical outputs the FrameDesigner V2 previously had."
                    },
                ),
            }
        }

    RETURN_TYPES = IDEO_INFO_RETURN_TYPES
    RETURN_NAMES = IDEO_INFO_RETURN_NAMES
    FUNCTION = "unpack"
    CATEGORY = "IAMCCS/Cine/Ideogram"

    def unpack(self, IDEO_LINX):
        if not isinstance(IDEO_LINX, dict):
            raise ValueError("IDEO_INFO expected an IDEO_LINX bundle from FrameDesigner V2")
        missing = [name for name in IDEO_INFO_RETURN_NAMES if name not in IDEO_LINX]
        if missing:
            raise ValueError("IDEO_INFO missing IDEO_LINX field(s): " + ", ".join(missing))
        values = {name: IDEO_LINX[name] for name in IDEO_INFO_RETURN_NAMES}
        bboxes = values.get("bboxes")
        bbox_count = len(bboxes[0]) if isinstance(bboxes, list) and bboxes and isinstance(bboxes[0], list) else 0
        info_report = {
            "ideo_info": "passthrough",
            "positive_type": type(values.get("positive")).__name__,
            "prompt_json_chars": len(str(values.get("prompt_json") or "")),
            "width": int(values.get("width") or 0),
            "height": int(values.get("height") or 0),
            "bboxes": bbox_count,
            "target_panel_width": int(values.get("target_panel_width") or 0),
            "target_panel_height": int(values.get("target_panel_height") or 0),
            "i2i_enabled": bool(values.get("i2i_enabled")),
        }
        existing_report = str(values.get("debug_report") or "").strip()
        values["debug_report"] = (existing_report + "\n" if existing_report else "") + json.dumps(info_report, ensure_ascii=False)
        return tuple(values[name] for name in IDEO_INFO_RETURN_NAMES)


class IAMCCS_IdeoInpaintPrep:
    """Resize an image+mask pair to the Ideogram frame before InpaintModelConditioning."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "width": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 8}),
                "fit": (["cover", "contain", "stretch"], {"default": "cover"}),
                "mask_grow_px": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "mask_blur_px": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "invert_mask": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "report")
    FUNCTION = "prepare"
    CATEGORY = "IAMCCS/Cine/Ideogram"

    def prepare(self, image, mask, width=1024, height=1024, fit="cover", mask_grow_px=0, mask_blur_px=0, invert_mask=False):
        width = _clamp_int(width, 64, 16384, 1024)
        height = _clamp_int(height, 64, 16384, 1024)
        fit = str(fit or "cover").lower()
        if fit not in {"cover", "contain", "stretch"}:
            fit = "cover"
        source_images = _image_batch_to_pil_list(image)
        source_masks = _mask_batch_to_pil_list(mask, len(source_images))
        out_images: List[Image.Image] = []
        out_masks: List[Image.Image] = []
        mask_means: List[float] = []
        mask_maxes: List[float] = []
        grow = max(0, int(mask_grow_px))
        blur = max(0, int(mask_blur_px))
        for src, msk in zip(source_images, source_masks):
            out_images.append(_fit_pil_to_box_generic(src, width, height, fit, fill=(0, 0, 0)))
            m = _fit_pil_to_box_generic(msk, width, height, fit, fill=0).convert("L")
            if invert_mask:
                m = ImageOps.invert(m)
            if grow > 0:
                radius = grow * 2 + 1
                m = m.filter(ImageFilter.MaxFilter(radius))
            if blur > 0:
                m = m.filter(ImageFilter.GaussianBlur(blur))
            arr = np.asarray(m, dtype=np.float32) / 255.0
            mask_means.append(float(arr.mean()) if arr.size else 0.0)
            mask_maxes.append(float(arr.max()) if arr.size else 0.0)
            out_masks.append(m)
        mean = sum(mask_means) / max(1, len(mask_means))
        maximum = max(mask_maxes) if mask_maxes else 0.0
        report = (
            f"IDEO inpaint prep: {len(out_images)} frame(s), {width}x{height}, "
            f"fit={fit}, grow={grow}, blur={blur}, invert={bool(invert_mask)}, "
            f"mask_mean={mean:.6f}, mask_max={maximum:.3f}"
        )
        return (_pil_images_to_tensor(out_images), _pil_masks_to_tensor(out_masks), report)


class IAMCCS_IdeoMaskedPixels:
    """Prepare visibly editable inpaint pixels by replacing the masked area.

    Some Ideogram/Flux-style inpaint paths preserve too much of the original
    pixels when the masked area is fed unchanged. This node mirrors the useful
    part of external masked-noise workflows without adding third-party runtime
    dependencies: it creates an altered pixel input for the mask while passing
    the same mask forward to InpaintModelConditioning.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "fill_mode": (["solid_black", "solid_gray", "noise", "blurred_source"], {"default": "solid_black"}),
                "strength": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fill_color": ("STRING", {"default": "#000000"}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "mask", "mask_preview", "report")
    FUNCTION = "apply"
    CATEGORY = "IAMCCS/Cine/Ideogram"

    def apply(self, image, mask, fill_mode="solid_black", strength=0.95, fill_color="#000000", seed=1):
        source_images = _image_batch_to_pil_list(image)
        source_masks = _mask_batch_to_pil_list(mask, len(source_images))
        mode = str(fill_mode or "solid_black").lower()
        if mode not in {"solid_black", "solid_gray", "noise", "blurred_source"}:
            mode = "solid_black"
        strength = max(0.0, min(1.0, float(strength if strength is not None else 0.95)))
        color = _normalize_hex(fill_color) or "#000000"
        rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        out_images: List[Image.Image] = []
        out_masks: List[Image.Image] = []
        previews: List[Image.Image] = []
        mask_means: List[float] = []

        for src, msk in zip(source_images, source_masks):
            src = src.convert("RGB")
            msk = msk.convert("L")
            mask_arr = np.asarray(msk, dtype=np.float32) / 255.0
            mask_means.append(float(mask_arr.mean()))
            alpha = np.clip(mask_arr * strength, 0.0, 1.0)[..., None]
            src_arr = np.asarray(src, dtype=np.float32) / 255.0

            if mode == "solid_gray":
                fill_arr = np.full_like(src_arr, 0.5, dtype=np.float32)
            elif mode == "noise":
                fill_arr = rng.random(src_arr.shape, dtype=np.float32)
            elif mode == "blurred_source":
                fill_arr = np.asarray(src.filter(ImageFilter.GaussianBlur(24)), dtype=np.float32) / 255.0
            else:
                fill_arr = np.zeros_like(src_arr, dtype=np.float32)
                fill_arr[..., 0] = rgb[0] / 255.0
                fill_arr[..., 1] = rgb[1] / 255.0
                fill_arr[..., 2] = rgb[2] / 255.0

            mixed = np.clip(src_arr * (1.0 - alpha) + fill_arr * alpha, 0.0, 1.0)
            out_images.append(Image.fromarray((mixed * 255.0).round().astype(np.uint8), "RGB"))
            out_masks.append(msk)
            previews.append(Image.merge("RGB", (msk, msk, msk)))

        report = (
            f"IDEO masked pixels: frames={len(out_images)}, mode={mode}, "
            f"strength={strength:.2f}, mask_mean={sum(mask_means) / max(1, len(mask_means)):.4f}"
        )
        return (_pil_images_to_tensor(out_images), _pil_masks_to_tensor(out_masks), _pil_images_to_tensor(previews), report)


class IAMCCS_IdeogramJSONPreviewPass:
    """Transparent Ideogram JSON checkpoint/pass node.

    It intentionally preserves the incoming prompt string. Use it after
    FrameDesigner or external IdeoTranslate/Gemma JSON builders to inspect and
    route the exact structured JSON that will feed Ideogram.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "forceInput": True,
                        "tooltip": "Structured Ideogram JSON to preview/pass. The node does not rewrite it.",
                    },
                ),
                "width": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 16}),
            },
            "optional": {
                "bboxes": (
                    "BOUNDING_BOX",
                    {
                        "forceInput": True,
                        "tooltip": "Optional KJ/IAMCCS pixel-space BoundingBox output passed through unchanged.",
                    },
                ),
                "debug_report": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "forceInput": True,
                        "tooltip": "Optional IDEO_INFO debug report shown in the preview UI.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "BOUNDING_BOX", "INT", "INT", "STRING")
    RETURN_NAMES = ("prompt_json", "bboxes", "width", "height", "report")
    FUNCTION = "pass_json"
    CATEGORY = "IAMCCS/Cine/Ideogram"
    OUTPUT_NODE = True

    def pass_json(self, prompt_json, width=1024, height=1024, bboxes=None, debug_report=""):
        text = str(prompt_json or "").strip()
        report = "Ideogram JSON pass-through"
        try:
            parsed = json.loads(text) if text else {}
            if isinstance(parsed, dict):
                style = parsed.get("style_description") if isinstance(parsed.get("style_description"), dict) else {}
                comp = parsed.get("compositional_deconstruction") if isinstance(parsed.get("compositional_deconstruction"), dict) else {}
                elements = comp.get("elements") if isinstance(comp.get("elements"), list) else []
                report = (
                    "Ideogram JSON pass-through | "
                    f"high={bool(parsed.get('high_level_description'))} "
                    f"style_keys={','.join(style.keys()) or 'none'} "
                    f"elements={len(elements)}"
                )
            else:
                report = "Ideogram JSON pass-through | parsed JSON is not an object"
        except Exception as exc:
            report = f"Ideogram JSON pass-through | invalid JSON: {exc}"
        debug_text = str(debug_report or "").strip()
        if debug_text:
            report = f"{report} | debug_report=yes"
        preview = text if len(text) <= 12000 else text[:12000] + "\n... [truncated preview]"
        ui_text = [report, preview]
        if debug_text:
            ui_text.append(debug_text if len(debug_text) <= 12000 else debug_text[:12000] + "\n... [truncated debug]")
        return {
            "ui": {"text": ui_text},
            "result": (text, bboxes or [], int(width), int(height), report),
        }


NODE_CLASS_MAPPINGS = {
    "IAMCCS_StoryboardFrameDesigner": IAMCCS_StoryboardFrameDesigner,
    "IAMCCS_StoryboardFrameDesignerV2": IAMCCS_StoryboardFrameDesignerV2,
    "IAMCCS_IdeoInfo": IAMCCS_IdeoInfo,
    "IAMCCS_IdeoInpaintPrep": IAMCCS_IdeoInpaintPrep,
    "IAMCCS_IdeoMaskedPixels": IAMCCS_IdeoMaskedPixels,
    "IAMCCS_IdeogramJSONPreviewPass": IAMCCS_IdeogramJSONPreviewPass,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_StoryboardFrameDesigner": "IAMCCS StoryboardFrame + TextInFrame Director",
    "IAMCCS_StoryboardFrameDesignerV2": "IAMCCS StoryboardFrame V2 + Image Canvas i2i",
    "IAMCCS_IdeoInfo": "IDEO_INFO",
    "IAMCCS_IdeoInpaintPrep": "IAMCCS Ideo Inpaint Prep",
    "IAMCCS_IdeoMaskedPixels": "IAMCCS Ideo Masked Pixels",
    "IAMCCS_IdeogramJSONPreviewPass": "IAMCCS Ideogram JSON Preview / Pass",
}


_register_framedesigner_gemma_route()
