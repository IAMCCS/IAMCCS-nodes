import copy
import json
from datetime import datetime, timezone
from typing import Any, Dict, List

# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
# By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com

from .iamccs_ideogram_storyboard_frame_designer import (
    _clean_text,
    _design_from_preset,
    _normalize_design,
    _palette_list,
    _preset_library,
    _to_ideogram_prompt,
)


def _safe_json(raw: Any, fallback: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return copy.deepcopy(raw)
    try:
        return json.loads(str(raw or ""))
    except Exception:
        return copy.deepcopy(fallback)


def _translated_board_name(prompt_json_in: Any, preset_key: str) -> str:
    data = _safe_json(prompt_json_in, {})
    if isinstance(data, dict):
        if isinstance(data.get("board_name"), str) and data.get("board_name").strip():
            return data["board_name"].strip()
        high_level = _clean_text(data.get("high_level_description"))
        if high_level:
            compact = "_".join(high_level.lower().split()[:6]).strip("_")
            if compact:
                return f"IAMCCS_{preset_key}_{compact}"
    return f"IAMCCS_{preset_key}_translated"


def _default_text_item(preset_key: str, existing_count: int) -> Dict[str, Any]:
    defaults = {
        "storyboard": {
            "label": "Scene note",
            "text": "SHOT NOTE",
            "x": 700,
            "y": 90,
            "w": 220,
            "h": 90,
            "desc": "Auto-generated note zone for storyboard annotation or shot labeling.",
            "color_palette": ["#E9D7B9", "#B86A3B"],
        },
        "poster": {
            "label": "Title zone",
            "text": "TITLE",
            "x": 240,
            "y": 1190,
            "w": 520,
            "h": 110,
            "desc": "Auto-generated poster title zone with clean theatrical hierarchy.",
            "color_palette": ["#F5E6C8", "#C84E2F"],
        },
        "signage": {
            "label": "Main sign text",
            "text": "SIGN NAME",
            "x": 320,
            "y": 165,
            "w": 360,
            "h": 85,
            "desc": "Auto-generated signage text block for the main readable in-world graphic.",
            "color_palette": ["#F6F1D1", "#19A7CE"],
        },
        "screen_ui": {
            "label": "Screen header",
            "text": "SYSTEM HEADER",
            "x": 280,
            "y": 60,
            "w": 420,
            "h": 60,
            "desc": "Auto-generated diegetic header strip for screen-UI compositions.",
            "color_palette": ["#D9F3FF", "#37D5D6"],
        },
        "title_card": {
            "label": "Main title",
            "text": "TITLE CARD",
            "x": 240,
            "y": 340,
            "w": 520,
            "h": 110,
            "desc": "Auto-generated title lockup for title-card compositions.",
            "color_palette": ["#F6E7CB", "#8C1C13"],
        },
    }
    entry = copy.deepcopy(defaults.get(preset_key, defaults["storyboard"]))
    entry.update(
        {
            "id": f"item_{existing_count + 1:03d}",
            "kind": "text",
        }
    )
    return entry


def _apply_preset_override(design: Dict[str, Any], preset_override: str, normalize_canvas_to_preset: bool) -> Dict[str, Any]:
    preset_key = _clean_text(preset_override).lower()
    if preset_key not in _preset_library():
        return design
    translated = _design_from_preset(preset_key)
    translated["scene"].update(copy.deepcopy(design.get("scene") or {}))
    translated["items"] = copy.deepcopy(design.get("items") or translated.get("items") or [])
    translated["preset_key"] = preset_key
    if not normalize_canvas_to_preset:
        translated["canvas"].update(copy.deepcopy(design.get("canvas") or {}))
    return translated


def _ensure_text_boxes(design: Dict[str, Any], enabled: bool) -> Dict[str, Any]:
    if not enabled:
        return design
    items = design.get("items") if isinstance(design.get("items"), list) else []
    if any(str(item.get("kind") or "").lower() == "text" for item in items if isinstance(item, dict)):
        return design
    updated = copy.deepcopy(design)
    updated_items = list(updated.get("items") or [])
    updated_items.append(_default_text_item(updated.get("preset_key") or "storyboard", len(updated_items)))
    updated["items"] = updated_items
    return updated


def _refresh_scene_palette(design: Dict[str, Any], enabled: bool) -> Dict[str, Any]:
    if not enabled:
        return design
    updated = copy.deepcopy(design)
    scene = updated.get("scene") if isinstance(updated.get("scene"), dict) else {}
    item_colors: List[str] = []
    for raw_item in updated.get("items") if isinstance(updated.get("items"), list) else []:
        if not isinstance(raw_item, dict):
            continue
        item_colors.extend(_palette_list(raw_item.get("color_palette"), []))
    if item_colors:
        deduped = []
        for color in item_colors:
            if color not in deduped:
                deduped.append(color)
        scene["color_palette"] = deduped[:6]
    updated["scene"] = scene
    return updated


def _build_ideoboard_package(design: Dict[str, Any], board_name: str) -> Dict[str, Any]:
    preset_key = _clean_text(design.get("preset_key") or "storyboard").lower()
    if preset_key not in _preset_library():
        preset_key = "storyboard"
    canvas = design.get("canvas") if isinstance(design.get("canvas"), dict) else {}
    return {
        "schema": "iamccs.ideoboard.package",
        "schema_version": 1,
        "board_name": board_name,
        "active_preset_key": preset_key,
        "boards": {
            preset_key: copy.deepcopy(design),
        },
        "metadata": {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "source_node": "IAMCCS_IdeoTranslate",
            "width": int(canvas.get("width") or 1024),
            "height": int(canvas.get("height") or 1024),
        },
        "assets": {
            "prompt_json": _to_ideogram_prompt(design),
        },
    }


def _summary_text(design: Dict[str, Any]) -> str:
    items = design.get("items") if isinstance(design.get("items"), list) else []
    text_count = sum(1 for item in items if isinstance(item, dict) and str(item.get("kind") or "").lower() == "text")
    object_count = sum(1 for item in items if isinstance(item, dict) and str(item.get("kind") or "obj").lower() != "text")
    canvas = design.get("canvas") if isinstance(design.get("canvas"), dict) else {}
    scene = design.get("scene") if isinstance(design.get("scene"), dict) else {}
    return (
        f"preset={design.get('preset_key') or 'storyboard'} | "
        f"size={int(canvas.get('width') or 1024)}x{int(canvas.get('height') or 1024)} | "
        f"objects={object_count} | texts={text_count} | "
        f"scene={_clean_text(scene.get('high_level_description'))[:120]}"
    )


class IAMCCS_IdeoTranslate:
    """Bridge node from LLM structured prompt JSON to IAMCCS ideoboard-compatible design data."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_json_in": (
                    "STRING",
                    {
                        "default": "{\n  \"high_level_description\": \"A cinematic frame\",\n  \"style_description\": {\n    \"aesthetics\": \"premium cinematic composition\",\n    \"lighting\": \"controlled dramatic lighting\",\n    \"photo\": \"35mm lens\",\n    \"medium\": \"cinematic concept art\",\n    \"color_palette\": [\"#1C2430\", \"#B86A3B\", \"#E9D7B9\"]\n  },\n  \"compositional_deconstruction\": {\n    \"background\": \"A controlled cinematic environment\",\n    \"elements\": []\n  }\n}",
                        "multiline": True,
                        "tooltip": "Structured JSON from the LLM Prompt Builder, prompt JSON, ideoboard package, or existing design_data.",
                    },
                ),
                "preset_override": (["auto", "storyboard", "poster", "signage", "screen_ui", "title_card"],),
                "normalize_canvas_to_preset": ("BOOLEAN", {"default": True}),
                "auto_generate_text_boxes": ("BOOLEAN", {"default": True}),
                "auto_palette_from_json": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = (
        "design_data_json",
        "ideoboard_package_json",
        "prompt_json_out",
        "translation_summary",
        "width",
        "height",
    )
    FUNCTION = "translate"
    CATEGORY = "IAMCCS/Cine/Ideogram"

    def translate(
        self,
        prompt_json_in: Any,
        preset_override: str = "auto",
        normalize_canvas_to_preset: bool = True,
        auto_generate_text_boxes: bool = True,
        auto_palette_from_json: bool = True,
    ):
        design = _normalize_design(prompt_json_in)
        if preset_override != "auto":
            design = _apply_preset_override(design, preset_override, bool(normalize_canvas_to_preset))
        design = _ensure_text_boxes(design, bool(auto_generate_text_boxes))
        design = _refresh_scene_palette(design, bool(auto_palette_from_json))

        board_name = _translated_board_name(prompt_json_in, _clean_text(design.get("preset_key") or "storyboard").lower())
        prompt_json = json.dumps(_to_ideogram_prompt(design), ensure_ascii=False, indent=2)
        design_data_json = json.dumps(design, ensure_ascii=False, indent=2)
        ideoboard_package_json = json.dumps(_build_ideoboard_package(design, board_name), ensure_ascii=False, indent=2)
        summary = _summary_text(design)
        width = int((design.get("canvas") or {}).get("width") or 1024)
        height = int((design.get("canvas") or {}).get("height") or 1024)
        return (design_data_json, ideoboard_package_json, prompt_json, summary, width, height)


NODE_CLASS_MAPPINGS = {
    "IAMCCS_IdeoTranslate": IAMCCS_IdeoTranslate,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_IdeoTranslate": "IAMCCS IdeoTranslate",
}
