import copy
import hashlib
import json
from typing import Any, Dict, List

import nodes as comfy_nodes

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
    try:
        return json.loads(str(raw or ""))
    except Exception:
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
    return {
        "schema": "iamccs.ideogram_storyboard_frame_designer",
        "schema_version": 1,
        "preset_key": "storyboard",
        "canvas": {
            "width": 1024,
            "height": 1024,
            "aspect_label": "1:1 Ideogram Board",
        },
        "scene": {
            "high_level_description": "A cinematic frame designed for structured visual prompting, balancing subject staging, readable text, and art-directed composition.",
            "aesthetics": "cinematic concept frame with clear hierarchy, polished layout, and production design intent",
            "lighting": "soft directional key light with controlled contrast and legible detail across foreground and background",
            "photo": "",
            "medium": "digital illustration",
            "color_palette": ["#1A1A2E", "#FF6B35", "#FFE4B5", "#8B4513"],
            "background": "A controlled background environment that supports the focal subject and leaves clean negative space for text or secondary design elements.",
        },
        "items": [
            {
                "id": "item_001",
                "kind": "obj",
                "label": "Primary subject",
                "text": "",
                "x": 160,
                "y": 170,
                "w": 480,
                "h": 520,
                "desc": "Primary visual subject in the mid-frame with strong silhouette readability, production-design detail, and clear cinematic emphasis.",
                "color_palette": ["#8B4513", "#FFE4B5", "#1A1A2E"],
            },
            {
                "id": "item_002",
                "kind": "text",
                "label": "Title block",
                "text": "TITLE",
                "x": 650,
                "y": 120,
                "w": 220,
                "h": 120,
                "desc": "Readable in-frame text block with deliberate typography, good contrast, and clean spatial separation from the main subject.",
                "color_palette": ["#FF6B35", "#FFE4B5"],
            },
        ],
    }


def _preset_library() -> Dict[str, Dict[str, Any]]:
    return {
        "storyboard": {
            "canvas": {"width": 1536, "height": 864, "aspect_label": "16:9 Storyboard"},
            "scene": {
                "high_level_description": "A storyboard-ready cinematic frame with strong blocking, readable staging, and production-minded composition.",
                "aesthetics": "film storyboard realism, readable silhouettes, practical art direction, shot-design clarity",
                "lighting": "controlled cinematic lighting with clear value separation and readable focal hierarchy",
                "medium": "storyboard concept art",
                "background": "Production-aware environment blocking that supports shot continuity and leaves room for annotations or title elements.",
                "color_palette": ["#1C2430", "#B86A3B", "#E9D7B9", "#6B7A8F"],
            },
        },
        "poster": {
            "canvas": {"width": 1024, "height": 1536, "aspect_label": "2:3 Poster"},
            "scene": {
                "high_level_description": "A striking cinematic poster image with a dominant focal subject, bold typography zones, and premium visual hierarchy.",
                "aesthetics": "premium theatrical poster design, dramatic scale, iconic silhouette, polished key art",
                "lighting": "high-contrast dramatic key art lighting with controlled glow, depth, and premium finish",
                "medium": "cinematic poster illustration",
                "background": "Graphic poster backdrop with atmospheric depth and clear negative space for billing, taglines, and title treatment.",
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
                "high_level_description": "A cinematic title card frame designed around elegant typography, strong negative space, and mood-driven art direction.",
                "aesthetics": "title card design, clean hierarchy, premium typography composition, deliberate negative space",
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
    design["canvas"].update(copy.deepcopy(preset.get("canvas") or {}))
    design["scene"].update(copy.deepcopy(preset.get("scene") or {}))
    return design


def _item_from_bbox(entry: Dict[str, Any], index: int) -> Dict[str, Any]:
    bbox = entry.get("bbox") if isinstance(entry.get("bbox"), list) else [100, 100, 800, 800]
    if len(bbox) != 4:
        bbox = [100, 100, 800, 800]
    x1 = _clamp_int(bbox[0], 0, 999, 100)
    y1 = _clamp_int(bbox[1], 0, 999, 100)
    x2 = _clamp_int(bbox[2], x1 + 1, 1000, 800)
    y2 = _clamp_int(bbox[3], y1 + 1, 1000, 800)
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
    comp = data.get("compositional_deconstruction") if isinstance(data.get("compositional_deconstruction"), dict) else {}
    items = comp.get("elements") if isinstance(comp.get("elements"), list) else []
    base["scene"] = {
        "high_level_description": _clean_text(data.get("high_level_description")) or base["scene"]["high_level_description"],
        "aesthetics": _clean_text(style.get("aesthetics")) or base["scene"]["aesthetics"],
        "lighting": _clean_text(style.get("lighting")) or base["scene"]["lighting"],
        "photo": _clean_text(style.get("photo")),
        "medium": _clean_text(style.get("medium")) or base["scene"]["medium"],
        "color_palette": _palette_list(style.get("color_palette"), base["scene"]["color_palette"]),
        "background": _clean_text(comp.get("background")) or base["scene"]["background"],
    }
    base["items"] = [_item_from_bbox(item, index) for index, item in enumerate(items)] or base["items"]
    return base


def _normalize_item(entry: Dict[str, Any], index: int) -> Dict[str, Any]:
    item = {
        "id": _clean_text(entry.get("id")) or f"item_{index + 1:03d}",
        "kind": "text" if str(entry.get("kind") or entry.get("type") or "obj").lower() == "text" else "obj",
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
    if item["kind"] != "text":
        item["text"] = ""
    return item


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
    if isinstance(data, dict) and {
        "high_level_description",
        "style_description",
        "compositional_deconstruction",
    }.issubset(set(data.keys())):
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
    merged["items"] = [_normalize_item(item, index) for index, item in enumerate(data.get("items") if isinstance(data.get("items"), list) else [])] or base["items"]
    merged["canvas"]["width"] = _clamp_int(merged["canvas"].get("width"), 256, 4096, 1024)
    merged["canvas"]["height"] = _clamp_int(merged["canvas"].get("height"), 256, 4096, 1024)
    merged["preset_key"] = preset_key if preset_key in _preset_library() else base["preset_key"]
    merged["canvas"]["aspect_label"] = _clean_text(merged["canvas"].get("aspect_label")) or base["canvas"]["aspect_label"]
    return merged


def _to_ideogram_prompt(design: Dict[str, Any]) -> Dict[str, Any]:
    scene = design.get("scene") if isinstance(design.get("scene"), dict) else {}
    items = design.get("items") if isinstance(design.get("items"), list) else []
    elements: List[Dict[str, Any]] = []
    for index, raw_item in enumerate(items):
        item = _normalize_item(raw_item if isinstance(raw_item, dict) else {}, index)
        x1 = item["x"]
        y1 = item["y"]
        x2 = min(1000, x1 + item["w"])
        y2 = min(1000, y1 + item["h"])
        entry: Dict[str, Any] = {
            "type": "text" if item["kind"] == "text" else "obj",
            "bbox": [x1, y1, x2, y2],
            "desc": item["desc"],
            "color_palette": item["color_palette"],
        }
        if item["kind"] == "text":
            entry["text"] = item["text"] or item["label"]
        elements.append(entry)
    return {
        "high_level_description": _clean_text(scene.get("high_level_description")),
        "style_description": {
            "aesthetics": _clean_text(scene.get("aesthetics")),
            "lighting": _clean_text(scene.get("lighting")),
            "photo": _clean_text(scene.get("photo")),
            "medium": _clean_text(scene.get("medium")),
            "color_palette": _palette_list(scene.get("color_palette"), ["#1A1A2E", "#FFE4B5"]),
        },
        "compositional_deconstruction": {
            "background": _clean_text(scene.get("background")),
            "elements": elements,
        },
    }


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

    RETURN_TYPES = ("CONDITIONING", "STRING", "INT", "INT")
    RETURN_NAMES = ("positive", "prompt_json", "width", "height")
    FUNCTION = "encode"
    CATEGORY = "IAMCCS/Cine/Ideogram"

    @staticmethod
    def _basic_encode(clip, text: str):
        result = comfy_nodes.CLIPTextEncode().encode(clip, str(text or ""))
        return result[0] if isinstance(result, tuple) else result

    def encode(self, clip, design_data, ideoboard_input_signature="", ideoboard=""):
        incoming_signature = _ideoboard_input_signature(ideoboard)
        stored_signature = _clean_text(ideoboard_input_signature)
        use_incoming = bool(incoming_signature and incoming_signature != stored_signature)
        source_data = ideoboard if use_incoming else design_data

        design = _normalize_design(source_data)
        design_data_json = json.dumps(design, ensure_ascii=False, indent=2)
        prompt_json = json.dumps(_to_ideogram_prompt(design), ensure_ascii=False, indent=2)
        conditioning = self._basic_encode(clip, prompt_json)
        width = int(design.get("canvas", {}).get("width") or 1024)
        height = int(design.get("canvas", {}).get("height") or 1024)
        return {
            "ui": {
                "design_data": [design_data_json],
                "ideoboard_input_signature": [incoming_signature or stored_signature],
                "used_ideoboard_input": [use_incoming],
                "width": [width],
                "height": [height],
            },
            "result": (conditioning, prompt_json, width, height),
        }


NODE_CLASS_MAPPINGS = {
    "IAMCCS_StoryboardFrameDesigner": IAMCCS_StoryboardFrameDesigner,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_StoryboardFrameDesigner": "IAMCCS StoryboardFrame + TextInFrame Director",
}