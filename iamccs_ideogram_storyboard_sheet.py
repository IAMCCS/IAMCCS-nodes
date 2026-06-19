import copy
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _clamp_int(value: Any, minimum: int, maximum: int, default: int) -> int:
    try:
        cast = int(round(float(value)))
    except Exception:
        cast = int(default)
    return max(minimum, min(maximum, cast))


def _safe_json(raw: Any, fallback: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return copy.deepcopy(raw)
    try:
        return json.loads(str(raw or ""))
    except Exception:
        return copy.deepcopy(fallback)


def _palette(value: Any, fallback: List[str] | None = None) -> List[str]:
    fallback = list(fallback or ["#0A0A0D", "#1C2230", "#7B1E2B", "#B9A66D", "#D8D1C0"])
    source = value if isinstance(value, list) else str(value or "").replace(";", ",").split(",")
    out: List[str] = []
    for raw in source:
        color = str(raw or "").strip().upper()
        if not color:
            continue
        if not color.startswith("#"):
            color = f"#{color}"
        if len(color) == 4:
            color = f"#{color[1] * 2}{color[2] * 2}{color[3] * 2}"
        try:
            if len(color) == 7:
                int(color[1:], 16)
                if color not in out:
                    out.append(color)
        except Exception:
            pass
    return out or fallback


def _default_panels() -> List[Dict[str, Any]]:
    return [
        {
            "title": "1. Corridor",
            "shot": "extreme wide action shot",
            "action": "the protagonist runs ankle-deep through a flooded palace corridor of impossible doors",
            "camera": "low wide lens, off-center, the character small in frame",
        },
        {
            "title": "2. Key",
            "shot": "medium-wide action shot",
            "action": "the protagonist crawls under a rotten dinner table reaching for a tarnished brass key",
            "camera": "profile view, eyes on the key, no camera gaze",
        },
        {
            "title": "3. Stairs",
            "shot": "low-angle full-body action shot",
            "action": "the protagonist slips down a chessboard staircase toward a sunken theatre",
            "camera": "body in motion, one hand scraping the filthy wall",
        },
        {
            "title": "4. Mirror",
            "shot": "tense mirror action shot",
            "action": "the protagonist is pulled sideways through a cracked oval mirror",
            "camera": "twisted profile, reflection delayed and misaligned",
        },
        {
            "title": "5. Saucer",
            "shot": "overhead action shot",
            "action": "the protagonist crawls across a cracked tea saucer floating in black oily water",
            "camera": "top-down composition, red thread labyrinth around the body",
        },
        {
            "title": "6. Door",
            "shot": "final wide action shot",
            "action": "the protagonist forces open a black door into a rotten winter garden under a false moon",
            "camera": "seen from behind three-quarter angle, not looking at camera",
        },
    ]


def _default_data() -> Dict[str, Any]:
    return {
        "schema": "iamccs.ideogram.storyboard_sheet",
        "schema_version": 1,
        "title": "IAMCCS Dark Alice Storyboard",
        "layout": {
            "columns": 2,
            "rows": 3,
            "panel_width": 1024,
            "panel_height": 576,
            "gap": 0,
            "panel_order": "column_major",
            "orientation_guard": "wide landscape contact sheet, not portrait, not vertical",
        },
        "character_bible": (
            "same very thin adult Alice-like woman, dirty tangled red hair, pale sick skin, torn dark blue velvet coat, "
            "stained white collar, frayed red ribbon, tarnished brass key; she is inside the action, not posing, not looking into camera"
        ),
        "world_bible": (
            "dark surreal fantasy palace-world: flooded corridors, impossible doors, cracked mirrors, chessboard marble, red velvet, "
            "winter garden glass, fog, mold, stains, rust, mud, sickness, tactile photorealistic decay"
        ),
        "style_bible": (
            "extremely photorealistic dark surrealist fairytale cinema, Hungarian arthouse stillness, Russian poetic realism, "
            "Belgian symbolist production design, unsettling diseased Alice in Wonderland atmosphere, "
            "shot on vintage anamorphic lenses, 1970s fantasy film look, faded Eastmancolor print, optical diffusion, analog grain, "
            "matte painting aesthetic, 1970s European fantasy cinema, surreal fairy tale world, handcrafted practical creature effects, "
            "theatrical costumes, dreamlike medieval kingdom, impossible architecture, giant floating castles, storybook atmosphere, "
            "matte painting landscapes, vintage fantasy film, faded technicolor colors, soft optical diffusion, practical monster suits, "
            "analog cinema look, Eastmancolor film stock, atmospheric haze, magical realism, cult fantasy movie aesthetic, "
            "highly cinematic, lost 1974 European fantasy film, surreal dark fairy tale, practical monsters and masked creatures, "
            "giant puppets, floating castles on rocky cliffs, mysterious jesters, dreamlike medieval villages, "
            "faded blue and cyan color palette, soft focus lenses, atmospheric fog, matte painting backgrounds, handcrafted costumes, "
            "strange fantasy races, analog optical effects, cult fantasy cinema, forgotten storybook world, cinematic composition, "
            "Vintage European Surreal Fantasy Cinema 1965-1985"
        ),
        "lighting_bible": (
            "cold moonlit interiors, wet reflective floors, dim candle practicals, soft volumetric fog, deep blacks, restrained red accents, "
            "brass highlights, realistic sickly skin and dirty fabric texture"
        ),
        "camera_bible": (
            "ARRI Alexa 65 cinematic stills, anamorphic landscape 16:9 panels, action blocking, no front-facing portrait staging, "
            "no beauty-shot closeups, no centered foreground protagonist"
        ),
        "negative_bible": (
            "no portrait poster, no vertical sheet, no protagonist staring at viewer, no large foreground face, no clean beauty fashion image, "
            "no cheerful fairytale, no second protagonist"
        ),
        "color_palette": ["#0A0A0D", "#1C2230", "#443348", "#7B1E2B", "#B9A66D", "#D8D1C0"],
        "panels": _default_panels(),
    }


def _normalize_panel(panel: Any, index: int) -> Dict[str, str]:
    raw = panel if isinstance(panel, dict) else {}
    fallback = _default_panels()[index % len(_default_panels())]
    return {
        "title": _clean(raw.get("title")) or fallback["title"],
        "shot": _clean(raw.get("shot")) or fallback["shot"],
        "action": _clean(raw.get("action")) or fallback["action"],
        "camera": _clean(raw.get("camera")) or fallback["camera"],
    }


def _compact_title(text: Any, fallback: str = "IAMCCS Storyboard") -> str:
    cleaned = " ".join(_clean(text).replace("\n", " ").split())
    if not cleaned:
        return fallback
    first = cleaned.split(".")[0].strip()
    words = first.split()
    return " ".join(words[:8]) or fallback


def _style_field(style: Any, key: str, fallback: str = "") -> str:
    if isinstance(style, dict):
        value = style.get(key)
        if isinstance(value, list):
            return ", ".join(_clean(item) for item in value if _clean(item))
        return _clean(value) or fallback
    return fallback


def _element_desc(element: Any) -> str:
    if isinstance(element, dict):
        for key in ("desc", "description", "prompt", "text", "label", "title"):
            value = _clean(element.get(key))
            if value:
                return value
    return _clean(element)


def _panel_from_element(element: Any, index: int) -> Dict[str, str]:
    desc = _element_desc(element)
    fallback = _default_panels()[index % len(_default_panels())]
    title = ""
    if isinstance(element, dict):
        title = _clean(element.get("title")) or _clean(element.get("label"))
    if not title and desc:
        title = f"{index + 1}. {_compact_title(desc, 'Panel')}"
    return {
        "title": title or fallback["title"],
        "shot": _clean(element.get("shot")) if isinstance(element, dict) else fallback["shot"],
        "action": desc or fallback["action"],
        "camera": _clean(element.get("camera")) if isinstance(element, dict) else fallback["camera"],
    }


def _storyboard_from_prompt_json(data: Dict[str, Any], columns: int, rows: int, panel_width: int, panel_height: int, gap: int) -> Dict[str, Any]:
    style = data.get("style_description") if isinstance(data.get("style_description"), dict) else {}
    comp = data.get("compositional_deconstruction") if isinstance(data.get("compositional_deconstruction"), dict) else {}
    elements = comp.get("elements") if isinstance(comp.get("elements"), list) else []
    high_level = _clean(data.get("high_level_description"))
    background = _clean(comp.get("background"))
    panels = [_panel_from_element(item, index) for index, item in enumerate(elements)]
    return {
        "schema": "iamccs.ideogram.storyboard_sheet",
        "schema_version": 1,
        "title": _clean(data.get("title")) or _compact_title(high_level),
        "layout": {
            "columns": columns,
            "rows": rows,
            "panel_width": panel_width,
            "panel_height": panel_height,
            "gap": gap,
            "orientation_guard": "wide landscape contact sheet, not portrait, not vertical",
        },
        "character_bible": _clean(data.get("character_bible")) or "consistent protagonist and costume across all panels, inside the action, not posing",
        "world_bible": background or high_level,
        "style_bible": _style_field(style, "aesthetics", _clean(data.get("style"))),
        "lighting_bible": _style_field(style, "lighting", "controlled cinematic lighting"),
        "camera_bible": _style_field(style, "photo", "cinematic landscape 16:9 framing"),
        "negative_bible": _clean(data.get("negative_bible")) or "no vertical portrait sheet, no unrelated second protagonist, no front-facing fashion pose",
        "color_palette": _palette(style.get("color_palette") if isinstance(style, dict) else None),
        "panels": panels or _default_panels(),
    }


def _storyboard_from_design_data(data: Dict[str, Any], columns: int, rows: int, panel_width: int, panel_height: int, gap: int) -> Dict[str, Any]:
    scene = data.get("scene") if isinstance(data.get("scene"), dict) else {}
    canvas = data.get("canvas") if isinstance(data.get("canvas"), dict) else {}
    items = data.get("items") if isinstance(data.get("items"), list) else []
    panels = [_panel_from_element(item, index) for index, item in enumerate(items)]
    return {
        "schema": "iamccs.ideogram.storyboard_sheet",
        "schema_version": 1,
        "title": _clean(data.get("board_name")) or _compact_title(scene.get("high_level_description")),
        "layout": {
            "columns": columns,
            "rows": rows,
            "panel_width": panel_width or int(canvas.get("width") or 1024),
            "panel_height": panel_height or int(canvas.get("height") or 576),
            "gap": gap,
            "orientation_guard": "wide landscape contact sheet, not portrait, not vertical",
        },
        "character_bible": _clean(scene.get("character_bible")) or "consistent protagonist and costume across all panels, inside the action, not posing",
        "world_bible": _clean(scene.get("background")) or _clean(scene.get("high_level_description")),
        "style_bible": _clean(scene.get("aesthetics")) or _clean(scene.get("style")) or "cinematic storyboard contact sheet",
        "lighting_bible": _clean(scene.get("lighting")) or "controlled cinematic lighting",
        "camera_bible": _clean(scene.get("photo")) or "cinematic landscape 16:9 framing",
        "negative_bible": _clean(scene.get("negative_bible")) or "no vertical portrait sheet, no unrelated second protagonist",
        "color_palette": _palette(scene.get("color_palette")),
        "panels": panels or _default_panels(),
    }


def _coerce_storyboard_source(raw: Any, columns: int, rows: int, panel_width: int, panel_height: int, gap: int) -> Any:
    data = _safe_json(raw, {})
    if not isinstance(data, dict):
        return raw
    if data.get("schema") == "iamccs.ideogram.storyboard_sheet" or isinstance(data.get("panels"), list):
        return data
    if data.get("schema") == "iamccs.ideoboard.package":
        assets = data.get("assets") if isinstance(data.get("assets"), dict) else {}
        prompt_json = assets.get("prompt_json")
        if isinstance(prompt_json, dict):
            converted = _storyboard_from_prompt_json(prompt_json, columns, rows, panel_width, panel_height, gap)
            converted["title"] = _clean(data.get("board_name")) or converted["title"]
            return converted
        boards = data.get("boards") if isinstance(data.get("boards"), dict) else {}
        active = _clean(data.get("active_preset_key"))
        board = boards.get(active) if active else next(iter(boards.values()), None)
        if isinstance(board, dict):
            return _storyboard_from_design_data(board, columns, rows, panel_width, panel_height, gap)
    if isinstance(data.get("compositional_deconstruction"), dict) or data.get("high_level_description"):
        return _storyboard_from_prompt_json(data, columns, rows, panel_width, panel_height, gap)
    if isinstance(data.get("scene"), dict) or isinstance(data.get("items"), list):
        return _storyboard_from_design_data(data, columns, rows, panel_width, panel_height, gap)
    return data


def _normalize_data(raw: Any, columns: int, rows: int, panel_width: int, panel_height: int, gap: int) -> Dict[str, Any]:
    base = _default_data()
    data = _safe_json(raw, base)
    if not isinstance(data, dict):
        data = base
    merged = copy.deepcopy(base)
    merged.update({k: copy.deepcopy(v) for k, v in data.items() if k not in {"layout", "panels", "color_palette"}})
    layout = copy.deepcopy(base["layout"])
    if isinstance(data.get("layout"), dict):
        layout.update(data["layout"])
    layout["columns"] = _clamp_int(columns if columns is not None else layout.get("columns"), 1, 6, 2)
    layout["rows"] = _clamp_int(rows if rows is not None else layout.get("rows"), 1, 6, 3)
    layout["panel_width"] = _clamp_int(panel_width if panel_width is not None else layout.get("panel_width"), 256, 2048, 1024)
    layout["panel_height"] = _clamp_int(panel_height if panel_height is not None else layout.get("panel_height"), 256, 2048, 576)
    layout["gap"] = _clamp_int(gap if gap is not None else layout.get("gap"), 0, 128, 0)
    merged["layout"] = layout
    count = int(layout["columns"]) * int(layout["rows"])
    source_panels = data.get("panels") if isinstance(data.get("panels"), list) else base["panels"]
    merged["panels"] = [_normalize_panel(source_panels[i] if i < len(source_panels) else {}, i) for i in range(count)]
    merged["color_palette"] = _palette(data.get("color_palette"), base["color_palette"])
    return merged


def _panel_position(index: int, columns: int, rows: int, order: Any = "row_major") -> Tuple[int, int]:
    order_key = str(order or "row_major").strip().lower().replace("-", "_")
    if order_key in {"column", "columns", "column_major", "column_first"}:
        return index // rows, index % rows
    return index % columns, index // columns


def _panel_bbox(index: int, columns: int, rows: int, order: Any = "row_major") -> List[int]:
    """KJ/Ideogram bbox order: [ymin, xmin, ymax, xmax] on a 0-1000 grid."""
    col, row = _panel_position(index, columns, rows, order)
    xmin = round((col / columns) * 1000)
    ymin = round((row / rows) * 1000)
    xmax = round(((col + 1) / columns) * 1000)
    ymax = round(((row + 1) / rows) * 1000)
    return [ymin, xmin, ymax, xmax]


def _crop_manifest(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    layout = data["layout"]
    columns = int(layout["columns"])
    rows = int(layout["rows"])
    panel_w = int(layout["panel_width"])
    panel_h = int(layout["panel_height"])
    gap = int(layout["gap"])
    crops = []
    for index, panel in enumerate(data["panels"]):
        col, row = _panel_position(index, columns, rows, layout.get("panel_order"))
        crops.append(
            {
                "index": index + 1,
                "title": panel["title"],
                "x": col * (panel_w + gap),
                "y": row * (panel_h + gap),
                "width": panel_w,
                "height": panel_h,
                "column": col + 1,
                "row": row + 1,
            }
        )
    return crops


def _panel_desc(data: Dict[str, Any], panel: Dict[str, str], index: int) -> str:
    return (
        f"Panel {index + 1}, {panel['title']}: {panel['shot']}; {panel['action']}; {panel['camera']}. "
        f"Keep the same protagonist: {data['character_bible']}. She is embedded in the action, not a portrait."
    )


def _prompt_json(data: Dict[str, Any]) -> Dict[str, Any]:
    layout = data["layout"]
    columns = int(layout["columns"])
    rows = int(layout["rows"])
    panel_w = int(layout["panel_width"])
    panel_h = int(layout["panel_height"])
    sheet_w = columns * panel_w + max(0, columns - 1) * int(layout["gap"])
    sheet_h = rows * panel_h + max(0, rows - 1) * int(layout["gap"])
    elements = []
    for index, panel in enumerate(data["panels"]):
        desc = _panel_desc(data, panel, index)
        elements.append(
            {
                "type": "obj",
                "bbox": _panel_bbox(index, columns, rows, layout.get("panel_order")),
                "desc": desc,
                "color_palette": data["color_palette"],
            }
        )
    return {
        "high_level_description": (
            f"{data['title']}. A {layout.get('orientation_guard')} arranged as exactly {columns} columns by {rows} rows. "
            f"It contains EXACTLY {columns * rows} equal landscape 16:9 cinematic storyboard panels, no more and no fewer. "
            f"Do not create 12 panels, do not create a 3x4 grid, do not create a 4x3 grid, and do not add extra mini-panels. "
            f"Full sheet {sheet_w}x{sheet_h}; each panel {panel_w}x{panel_h}. "
            f"The same protagonist appears consistently but always inside scene action."
        ),
        "style_description": {
            "aesthetics": data["style_bible"],
            "lighting": data["lighting_bible"],
            "photo": data["camera_bible"],
            "medium": "photography",
            "color_palette": data["color_palette"],
        },
        "compositional_deconstruction": {
            "background": (
                f"One single storyboard contact sheet with thin black grid lines and exactly {columns * rows} panels only. "
                f"{data['world_bible']}. "
                f"Character continuity bible: {data['character_bible']}. Negative constraints: {data['negative_bible']}."
            ),
            "elements": elements,
        },
    }


def _kj_elements_data(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    layout = data["layout"]
    columns = int(layout["columns"])
    rows = int(layout["rows"])
    boxes: List[Dict[str, Any]] = []
    for index, panel in enumerate(data["panels"]):
        col, row = _panel_position(index, columns, rows, layout.get("panel_order"))
        boxes.append(
            {
                "x": col / columns,
                "y": row / rows,
                "w": 1 / columns,
                "h": 1 / rows,
                "type": "obj",
                "text": "",
                "desc": _panel_desc(data, panel, index),
                "palette": data["color_palette"][:5],
            }
        )
    return boxes


def _ideoboard_design_from_storyboard(data: Dict[str, Any]) -> Dict[str, Any]:
    layout = data["layout"]
    columns = int(layout["columns"])
    rows = int(layout["rows"])
    panel_w = int(layout["panel_width"])
    panel_h = int(layout["panel_height"])
    gap = int(layout["gap"])
    width = columns * panel_w + max(0, columns - 1) * gap
    height = rows * panel_h + max(0, rows - 1) * gap
    items: List[Dict[str, Any]] = []
    for index, panel in enumerate(data["panels"]):
        col, row = _panel_position(index, columns, rows, layout.get("panel_order"))
        items.append(
            {
                "id": f"panel_{index + 1:02d}",
                "kind": "obj",
                "label": panel["title"],
                "text": "",
                "x": round((col / columns) * 1000),
                "y": round((row / rows) * 1000),
                "w": round(1000 / columns),
                "h": round(1000 / rows),
                "desc": _panel_desc(data, panel, index),
                "color_palette": data["color_palette"][:5],
            }
        )
    return {
        "schema": "iamccs.ideogram_storyboard_frame_designer",
        "schema_version": 1,
        "preset_key": "storyboard",
        "canvas": {
            "width": width,
            "height": height,
            "aspect_label": f"{columns}x{rows} Ideogram StorySheet",
        },
        "scene": {
            "high_level_description": f"{data['title']}. Exactly {columns} columns by {rows} rows, {columns * rows} panels only.",
            "aesthetics": data["style_bible"],
            "lighting": data["lighting_bible"],
            "photo": data["camera_bible"],
            "medium": "photography",
            "color_palette": data["color_palette"],
            "background": f"{data['world_bible']}. Character continuity bible: {data['character_bible']}. Negative constraints: {data['negative_bible']}.",
            "character_bible": data["character_bible"],
            "negative_bible": data["negative_bible"],
        },
        "i2i": {
            "enabled": False,
            "denoise": 0.28,
            "low_sigma_start_step": 12,
            "source_mode": "storyboard_sheet",
        },
        "items": items,
    }


def _ideoboard_package_from_storyboard(data: Dict[str, Any], prompt: Dict[str, Any], design: Dict[str, Any]) -> Dict[str, Any]:
    canvas = design.get("canvas") if isinstance(design.get("canvas"), dict) else {}
    return {
        "schema": "iamccs.ideoboard.package",
        "schema_version": 1,
        "board_name": data["title"],
        "active_preset_key": "storyboard",
        "boards": {"storyboard": design},
        "metadata": {
            "source_node": "IAMCCS_IdeogramStoryboardSheet",
            "width": int(canvas.get("width") or 1024),
            "height": int(canvas.get("height") or 1024),
            "panel_count": len(data["panels"]),
        },
        "assets": {"prompt_json": prompt},
    }


def _flatten_prompt(prompt: Dict[str, Any]) -> str:
    style = prompt["style_description"]
    comp = prompt["compositional_deconstruction"]
    lines = [
        prompt["high_level_description"],
        f"Style: {style['aesthetics']}",
        f"Lighting: {style['lighting']}",
        f"Camera/format: {style['photo']}",
        f"Background/world: {comp['background']}",
        "Panels:",
    ]
    for item in comp["elements"]:
        lines.append(f"- {item['desc']}")
    return "\n".join(lines)


class IAMCCS_IdeogramStoryboardSheet:
    """Direct IAMCCS storyboard-sheet prompt builder for Ideogram contact sheets."""

    DEFAULT_DATA = json.dumps(_default_data(), ensure_ascii=False, indent=2)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "storyboard_data": (
                    "STRING",
                    {
                        "default": cls.DEFAULT_DATA,
                        "multiline": True,
                        "tooltip": "IAMCCS storyboard sheet JSON edited by the professional UI.",
                    },
                ),
                "columns": ("INT", {"default": 2, "min": 1, "max": 6, "step": 1}),
                "rows": ("INT", {"default": 3, "min": 1, "max": 6, "step": 1}),
                "panel_width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
                "panel_height": ("INT", {"default": 576, "min": 256, "max": 2048, "step": 8}),
                "grid_gap": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
            },
            "optional": {
                "ideotranslate_json": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Connect IAMCCS IdeoTranslate prompt_json_out, design_data_json, or ideoboard_package_json here.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "prompt_text",
        "prompt_json",
        "width",
        "height",
        "crop_manifest_json",
        "panel_prompts_json",
        "storyboard_data_json",
        "refine_prompts_text",
        "ideoboard_design_json",
        "ideoboard_package_json",
        "kj_elements_data_json",
        "kj_style_palette_data_json",
    )
    FUNCTION = "build"
    CATEGORY = "IAMCCS/Cine/Ideogram"

    def build(self, storyboard_data, columns=2, rows=3, panel_width=1024, panel_height=576, grid_gap=0, ideotranslate_json=""):
        source = ideotranslate_json if _clean(ideotranslate_json) else storyboard_data
        source = _coerce_storyboard_source(source, columns, rows, panel_width, panel_height, grid_gap)
        data = _normalize_data(source, columns, rows, panel_width, panel_height, grid_gap)
        prompt = _prompt_json(data)
        crops = _crop_manifest(data)
        panel_prompts = [
            {
                "index": item["index"],
                "title": item["title"],
                "prompt": prompt["compositional_deconstruction"]["elements"][item["index"] - 1]["desc"],
                "crop": item,
            }
            for item in crops
        ]
        layout = data["layout"]
        width = int(layout["columns"]) * int(layout["panel_width"]) + max(0, int(layout["columns"]) - 1) * int(layout["gap"])
        height = int(layout["rows"]) * int(layout["panel_height"]) + max(0, int(layout["rows"]) - 1) * int(layout["gap"])
        prompt_json = json.dumps(prompt, ensure_ascii=False, indent=2)
        design = _ideoboard_design_from_storyboard(data)
        package = _ideoboard_package_from_storyboard(data, prompt, design)
        design_json = json.dumps(design, ensure_ascii=False, indent=2)
        package_json = json.dumps(package, ensure_ascii=False, indent=2)
        kj_elements_json = json.dumps(_kj_elements_data(data), ensure_ascii=False, indent=2)
        kj_palette_json = json.dumps(data["color_palette"], ensure_ascii=False, indent=2)
        data_json = json.dumps(data, ensure_ascii=False, indent=2)
        crop_json = json.dumps(crops, ensure_ascii=False, indent=2)
        panel_json = json.dumps(panel_prompts, ensure_ascii=False, indent=2)
        refine_prompts = []
        for item in panel_prompts:
            refine_prompts.append(
                "REFINE this exact storyboard crop into a cleaner high-resolution cinematic still. "
                "Preserve the original composition, camera angle, framing, character identity, costume, pose, props, and scene continuity. "
                "Keep the protagonist inside the action of the scene, not as a foreground portrait and not looking into camera. "
                "Do not introduce a second protagonist. Do not change the shot design. "
                "Increase photorealistic texture, controlled film lighting, deep blacks, dirt, mold, stains, wet surfaces, fabric detail, and atmospheric depth. "
                f"{data['style_bible']}. Same protagonist: {data['character_bible']}. "
                f"Panel {item['index']}: {item['prompt']}"
            )
        refine_prompts_text = "\n".join(refine_prompts)
        return {
            "ui": {
                "storyboard_data": [data_json],
                "width": [width],
                "height": [height],
                "panel_count": [len(data["panels"])],
            },
            "result": (
                _flatten_prompt(prompt),
                prompt_json,
                width,
                height,
                crop_json,
                panel_json,
                data_json,
                refine_prompts_text,
                design_json,
                package_json,
                kj_elements_json,
                kj_palette_json,
            ),
        }


def _tensor_to_pil(frame: torch.Tensor) -> Image.Image:
    arr = (frame.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _load_font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, int(size))
        except Exception:
            pass
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    box = draw.textbbox((0, 0), text or "", font=font)
    return box[2] - box[0], box[3] - box[1]


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int, max_lines: int) -> List[str]:
    text = " ".join(str(text or "").split())
    if not text:
        return []
    words = text.split(" ")
    lines: List[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        width, _ = _text_size(draw, trial, font)
        if width <= max_width:
            current = trial
            continue
        if current:
            lines.append(current)
        current = word
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if len(lines) == max_lines and words:
        last = lines[-1]
        while _text_size(draw, last + "...", font)[0] > max_width and len(last) > 4:
            last = last[:-1].rstrip()
        lines[-1] = last + "..."
    return lines[:max_lines]


def _captions_from_json(raw: Any, count: int) -> Tuple[str, str, List[Dict[str, str]]]:
    title = "IAMCCS Storyboard"
    footer = ""
    panels: List[Dict[str, str]] = []
    data = _safe_json(raw, {})
    if isinstance(data, list):
        for item in data[:count]:
            item = item if isinstance(item, dict) else {}
            crop = item.get("crop") if isinstance(item.get("crop"), dict) else {}
            panels.append({
                "title": _clean(item.get("title")) or _clean(crop.get("title")),
                "caption": _clean(item.get("caption")) or _clean(item.get("action")) or _clean(item.get("prompt")),
            })
    elif isinstance(data, dict):
        title = _clean(data.get("title")) or title
        footer = _clean(data.get("footer")) or _clean(data.get("logline")) or _clean(data.get("end_message"))
        source_panels = data.get("panels") if isinstance(data.get("panels"), list) else []
        for item in source_panels[:count]:
            item = item if isinstance(item, dict) else {}
            panels.append({
                "title": _clean(item.get("title")),
                "caption": _clean(item.get("caption")) or _clean(item.get("action")) or _clean(item.get("prompt")),
            })
        if not panels:
            comp = data.get("compositional_deconstruction") if isinstance(data.get("compositional_deconstruction"), dict) else {}
            elements = comp.get("elements") if isinstance(comp.get("elements"), list) else []
            for idx, item in enumerate(elements[:count]):
                item = item if isinstance(item, dict) else {}
                desc = _clean(item.get("caption")) or _clean(item.get("desc")) or _clean(item.get("prompt")) or _clean(item.get("text"))
                explicit_title = _clean(item.get("title")) or _clean(item.get("label"))
                panel_title = explicit_title
                caption = desc
                if not panel_title and ":" in desc:
                    head, body = desc.split(":", 1)
                    head = _clean(head)
                    body = _clean(body)
                    if 2 <= len(head) <= 96 and body:
                        panel_title = head
                        caption = body
                panels.append({
                    "title": panel_title or f"{idx + 1}. Panel",
                    "caption": caption,
                })
    while len(panels) < count:
        idx = len(panels) + 1
        panels.append({"title": f"{idx}. Panel", "caption": ""})
    return title, footer, panels


class IAMCCS_StoryboardCaptionSheet:
    """Assemble a storyboard contact sheet with title, per-panel captions, and footer."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "storyboard_json": ("STRING", {"forceInput": True}),
                "columns": ("INT", {"default": 3, "min": 1, "max": 8, "step": 1}),
                "rows": ("INT", {"default": 3, "min": 1, "max": 8, "step": 1}),
                "cell_width": ("INT", {"default": 640, "min": 128, "max": 2048, "step": 8}),
                "image_height": ("INT", {"default": 360, "min": 96, "max": 2048, "step": 8}),
            },
            "optional": {
                "title_override": ("STRING", {"default": "", "multiline": False}),
                "footer_override": ("STRING", {"default": "", "multiline": False}),
                "caption_height": ("INT", {"default": 72, "min": 24, "max": 256, "step": 4}),
                "title_height": ("INT", {"default": 86, "min": 0, "max": 256, "step": 4}),
                "footer_height": ("INT", {"default": 120, "min": 0, "max": 256, "step": 4}),
                "gap": ("INT", {"default": 4, "min": 0, "max": 48, "step": 1}),
                "margin": ("INT", {"default": 16, "min": 0, "max": 96, "step": 1}),
                "font_size": ("INT", {"default": 18, "min": 8, "max": 80, "step": 1}),
                "title_size": ("INT", {"default": 44, "min": 12, "max": 120, "step": 1}),
                "background": ("STRING", {"default": "#F3F2EE", "multiline": False}),
                "paper": ("STRING", {"default": "#FFFFFF", "multiline": False}),
                "ink": ("STRING", {"default": "#1A1A1A", "multiline": False}),
                "muted_ink": ("STRING", {"default": "#555555", "multiline": False}),
                "line_color": ("STRING", {"default": "#C8C3B8", "multiline": False}),
                "fit": (["cover", "contain"], {"default": "cover"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("storyboard_sheet",)
    FUNCTION = "assemble"
    CATEGORY = "IAMCCS/Cine/Storyboard"

    def assemble(
        self,
        images,
        storyboard_json,
        columns=3,
        rows=3,
        cell_width=640,
        image_height=360,
        title_override="",
        footer_override="",
        caption_height=72,
        title_height=86,
        footer_height=120,
        gap=4,
        margin=16,
        font_size=18,
        title_size=44,
        background="#F3F2EE",
        paper="#FFFFFF",
        ink="#1A1A1A",
        muted_ink="#555555",
        line_color="#C8C3B8",
        fit="cover",
    ):
        count = min(int(images.shape[0]), int(columns) * int(rows))
        title, footer, panels = _captions_from_json(storyboard_json, count)
        title = _clean(title_override) or title
        footer = _clean(footer_override) or footer

        columns = max(1, int(columns))
        rows = max(1, int(rows))
        cell_width = max(64, int(cell_width))
        image_height = max(32, int(image_height))
        caption_height = max(0, int(caption_height))
        title_height = max(0, int(title_height))
        footer_height = max(0, int(footer_height))
        gap = max(0, int(gap))
        margin = max(0, int(margin))

        cell_height = image_height + caption_height
        sheet_w = margin * 2 + columns * cell_width + max(0, columns - 1) * gap
        sheet_h = margin * 2 + title_height + rows * cell_height + max(0, rows - 1) * gap + footer_height
        canvas = Image.new("RGB", (sheet_w, sheet_h), background)
        draw = ImageDraw.Draw(canvas)
        title_font = _load_font(title_size, True)
        header_font = _load_font(font_size + 2, True)
        caption_font = _load_font(font_size, False)
        footer_font = _load_font(max(10, font_size + 8), True)

        if title_height > 0:
            draw.rectangle([margin, margin, sheet_w - margin, margin + title_height], fill=paper, outline=line_color)
            draw.text((margin + 14, margin + max(6, (title_height - title_size) // 2)), title.upper(), fill=ink, font=title_font)

        grid_y = margin + title_height
        for idx in range(columns * rows):
            col = idx % columns
            row = idx // columns
            x = margin + col * (cell_width + gap)
            y = grid_y + row * (cell_height + gap)
            draw.rectangle([x, y, x + cell_width, y + cell_height], fill=paper, outline=line_color)
            if idx >= count:
                continue

            img = _tensor_to_pil(images[idx])
            src_w, src_h = img.size
            if fit == "contain":
                scale = min(cell_width / src_w, image_height / src_h)
            else:
                scale = max(cell_width / src_w, image_height / src_h)
            new_w = max(1, round(src_w * scale))
            new_h = max(1, round(src_h * scale))
            resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            left = max(0, (new_w - cell_width) // 2)
            top = max(0, (new_h - image_height) // 2)
            cropped = resized.crop((left, top, left + cell_width, top + image_height))
            canvas.paste(cropped, (x, y))

            caption_y = y + image_height
            draw.rectangle([x, caption_y, x + cell_width, y + cell_height], fill=paper, outline=line_color)
            panel = panels[idx]
            panel_title = _clean(panel.get("title")) or f"{idx + 1}. Panel"
            caption = _clean(panel.get("caption"))
            draw.text((x + 8, caption_y + 6), panel_title.upper(), fill=ink, font=header_font)
            lines = _wrap_text(draw, caption, caption_font, cell_width - 16, max(1, (caption_height - 34) // max(10, font_size + 2)))
            text_y = caption_y + 32
            for line in lines:
                draw.text((x + 8, text_y), line, fill=muted_ink, font=caption_font)
                text_y += font_size + 4

        if footer_height > 0:
            footer_y = sheet_h - margin - footer_height
            draw.rectangle([margin, footer_y, sheet_w - margin, sheet_h - margin], fill=paper, outline=line_color)
            for i, line in enumerate(_wrap_text(draw, footer, footer_font, sheet_w - margin * 2 - 28, 2)):
                draw.text((margin + 14, footer_y + 16 + i * (font_size + 14)), line, fill=ink, font=footer_font)

        return (_pil_to_tensor(canvas),)


NODE_CLASS_MAPPINGS = {
    "IAMCCS_IdeogramStoryboardSheet": IAMCCS_IdeogramStoryboardSheet,
    "IAMCCS_StoryboardCaptionSheet": IAMCCS_StoryboardCaptionSheet,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_IdeogramStoryboardSheet": "IAMCCS Ideogram Storyboard Sheet",
    "IAMCCS_StoryboardCaptionSheet": "IAMCCS Storyboard Caption Sheet",
}
