"""IAMCCS Ideogram Sheet Builder.

Backend intentionally mirrors Ideogram structured prompt semantics:
- Ideogram bbox order: [ymin, xmin, ymax, xmax]
- prompt output mirrors Ideogram4PromptBuilderKJ JSON semantics
- same import_json/import_mode behavior
"""

import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

from comfy_api.latest import io


_FONT_PATH = os.path.join(os.path.dirname(__file__), "fonts", "FreeMono.ttf")


def _hex_rgb(h):
    h = str(h or "").lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)) if len(h) == 6 else (255, 255, 255)


def _readable(rgb):
    r, g, b = rgb
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    if lum < 130:
        t = (130 - lum) / max(1, (255 - lum))
        r, g, b = round(r + (255 - r) * t), round(g + (255 - g) * t), round(b + (255 - b) * t)
    return (r, g, b)


def _font(size):
    try:
        return ImageFont.truetype(_FONT_PATH, size)
    except Exception:
        try:
            return ImageFont.load_default(size)
        except Exception:
            return ImageFont.load_default()


def _wrap(draw, text, font, max_w):
    lines = []
    for para in str(text or "").split("\n"):
        line = ""
        for word in para.split():
            test = word if not line else line + " " + word
            if line and draw.textlength(test, font=font) > max_w:
                lines.append(line)
                line = word
            else:
                line = test
        lines.append(line)
    return lines


def _render_preview(boxes, width, height, bg=None, brightness=50):
    if bg is not None:
        iw, ih = bg.size
        long_edge = max(iw, ih)
        scale = min(1.0, 1024 / long_edge) if long_edge > 0 else 1.0
        rw, rh = max(1, round(iw * scale)), max(1, round(ih * scale))
        base = bg.convert("RGB").resize((rw, rh), Image.LANCZOS)
        if brightness < 100:
            base = ImageEnhance.Brightness(base).enhance(max(0.0, brightness / 100.0))
        img = base.convert("RGBA")
    else:
        long_edge = max(width, height)
        scale = min(1.0, 1024 / long_edge) if long_edge > 0 else 1.0
        rw = max(1, round(width * scale))
        rh = max(1, round(height * scale))
        img = Image.new("RGBA", (rw, rh), (0, 0, 0, 255))
    overlay = Image.new("RGBA", (rw, rh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    fs = max(10, round(rh / 64))
    font = _font(fs)
    tag_font = _font(max(9, fs - 2))
    lh = fs + 2

    for i, box in enumerate(boxes):
        if not isinstance(box, dict) or box.get("nobbox"):
            continue
        palette = [c for c in (box.get("palette") or []) if c]
        r, g, b = _hex_rgb(palette[0]) if palette else (140, 140, 140)
        x1 = max(0, min(rw, round(float(box.get("x", 0)) * rw)))
        y1 = max(0, min(rh, round(float(box.get("y", 0)) * rh)))
        x2 = max(0, min(rw, round((float(box.get("x", 0)) + float(box.get("w", 0))) * rw)))
        y2 = max(0, min(rh, round((float(box.get("y", 0)) + float(box.get("h", 0))) * rh)))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        draw.rectangle([x1, y1, x2, y2], outline=(r, g, b, 255), width=2)
        if palette and (x2 - x1) > 2:
            sh = max(5, fs // 2)
            seg = (x2 - x1) / min(5, len(palette))
            for p, hexc in enumerate(palette[:5]):
                sx = x1 + round(p * seg)
                draw.rectangle([sx, y1, x1 + round((p + 1) * seg), y1 + sh], fill=_hex_rgb(hexc))
        tag = str(i + 1).zfill(2)
        tw = draw.textlength(tag, font=tag_font)
        draw.rectangle([x1, y1, x1 + tw + 6, y1 + fs + 2], fill=(r, g, b, 255))
        tagfill = (0, 0, 0, 255) if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else (255, 255, 255, 255)
        draw.text((x1 + 3, y1 + 1), tag, fill=tagfill, font=tag_font)
        body = str(box.get("desc", "") or "")
        if box.get("type") == "text" and box.get("text"):
            body = '"%s"%s' % (box["text"], " - " + body if body else "")
        if body and (x2 - x1) > 8:
            ty = y1 + fs + 5
            for line in _wrap(draw, body, font, x2 - x1 - 8):
                if ty > y2:
                    break
                draw.text((x1 + 4, ty), line, fill=_readable((r, g, b)) + (255,), font=font)
                ty += lh

    img = Image.alpha_composite(img, overlay).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _norm_bbox(box):
    def c(v):
        try:
            return max(0, min(1000, round(float(v) * 1000)))
        except Exception:
            return 0

    x = float(box.get("x", 0.0) or 0.0)
    y = float(box.get("y", 0.0) or 0.0)
    w = float(box.get("w", 0.0) or 0.0)
    h = float(box.get("h", 0.0) or 0.0)
    ymin, xmin, ymax, xmax = c(y), c(x), c(y + h), c(x + w)
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    return [ymin, xmin, ymax, xmax]


def _palette(colors):
    if isinstance(colors, dict):
        colors = colors.values()
    out = []
    for c in colors or []:
        text = str(c or "").strip().upper()
        if not text:
            continue
        if not text.startswith("#"):
            text = "#" + text
        if len(text) == 7:
            out.append(text)
    return out


def _dumps(v, lvl=0):
    pad, end = "    " * (lvl + 1), "    " * lvl
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        if not v:
            return "[]"
        if all(not isinstance(x, (dict, list)) for x in v):
            return "[" + ", ".join(_dumps(x, lvl) for x in v) + "]"
        return "[\n" + ",\n".join(pad + _dumps(x, lvl + 1) for x in v) + "\n" + end + "]"
    if isinstance(v, dict):
        if not v:
            return "{}"
        items = [pad + json.dumps(k, ensure_ascii=False) + ": " + _dumps(val, lvl + 1) for k, val in v.items()]
        return "{\n" + ",\n".join(items) + "\n" + end + "}"
    return json.dumps(v, ensure_ascii=False)


def _parse_json_list(s):
    if s:
        try:
            v = json.loads(str(s))
            if isinstance(v, list):
                return v
        except Exception:
            pass
    return []


def _caption_to_boxes(cap):
    cd = cap.get("compositional_deconstruction") or {}
    boxes = []
    for el in (cd.get("elements") or []):
        if not isinstance(el, dict):
            continue
        box = {
            "type": "text" if el.get("type") == "text" else "obj",
            "text": el.get("text", "") or "",
            "desc": el.get("desc", "") or "",
            "tips": "",
            "useTips": False,
            "palette": list(el.get("color_palette") or []),
        }
        bb = el.get("bbox")
        if isinstance(bb, (list, tuple)) and len(bb) == 4:
            ymin, xmin, ymax, xmax = bb
            box.update(x=xmin / 1000.0, y=ymin / 1000.0, w=(xmax - xmin) / 1000.0, h=(ymax - ymin) / 1000.0)
        else:
            box.update(x=0.03, y=0.03, w=0.22, h=0.14, nobbox=True)
        boxes.append(box)
    return boxes


def _caption_to_ideoboard_design(caption, boxes, width, height, grid_columns=1, grid_rows=1):
    style = caption.get("style_description") or {}
    comp = caption.get("compositional_deconstruction") or {}
    items = []
    for i, box in enumerate(boxes or []):
        if not isinstance(box, dict):
            continue
        x = max(0, min(1000, round(float(box.get("x", 0) or 0) * 1000)))
        y = max(0, min(1000, round(float(box.get("y", 0) or 0) * 1000)))
        w = max(20, min(1000 - x, round(float(box.get("w", 0) or 0) * 1000)))
        h = max(20, min(1000 - y, round(float(box.get("h", 0) or 0) * 1000)))
        kind = "text" if box.get("type") == "text" else "obj"
        items.append({
            "id": f"sheet_panel_{i + 1:02d}",
            "kind": kind,
            "label": str(box.get("title") or f"Panel {i + 1}"),
            "text": str(box.get("text") or "") if kind == "text" else "",
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "desc": str(box.get("desc") or ""),
            "color_palette": _palette(box.get("palette", []))[:5] or _palette(style.get("color_palette", []))[:5],
        })
    return {
        "schema": "iamccs.ideogram_storyboard_frame_designer",
        "schema_version": 1,
        "preset_key": "storyboard",
        "canvas": {"width": int(width), "height": int(height), "aspect_label": f"{int(width)}x{int(height)}"},
        "scene": {
            "high_level_description": str(caption.get("high_level_description") or ""),
            "aesthetics": str(style.get("aesthetics") or ""),
            "lighting": str(style.get("lighting") or ""),
            "photo": str(style.get("photo") or style.get("art_style") or ""),
            "medium": str(style.get("medium") or "photograph"),
            "color_palette": _palette(style.get("color_palette", [])),
            "background": str(comp.get("background") or ""),
        },
        "i2i": {
            "enabled": False,
            "denoise": 0.28,
            "low_sigma_start_step": 12,
            "scheduler_hint": "Use this sheet-derived layout as a positive i2i planning canvas.",
            "source_mode": "canvas_composite",
        },
        "items": items,
        "grid": {"columns": int(grid_columns or 1), "rows": int(grid_rows or 1), "order": "row_major"},
    }


class IAMCCS_IdeogramSheetBuilder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="IAMCCS_IdeogramSheetBuilder",
            display_name="IAMCCS Ideogram Sheet Builder",
            category="IAMCCS/Cine/Ideogram",
            search_aliases=["iamccs", "ideogram", "caption", "bbox", "prompt builder", "sheet builder"],
            is_experimental=True,
            description="Ideogram 4 sheet builder with IAMCCS UI.",
            inputs=[
                io.Int.Input("width", default=1024, min=64, max=16384, step=16),
                io.Int.Input("height", default=1024, min=64, max=16384, step=16),
                io.String.Input("high_level_description", multiline=True, default=""),
                io.String.Input("background", multiline=True, default=""),
                io.DynamicCombo.Input(
                    "style",
                    options=[
                        io.DynamicCombo.Option("none", []),
                        io.DynamicCombo.Option("photo", [io.String.Input("photo", default="")]),
                        io.DynamicCombo.Option("art_style", [io.String.Input("art_style", default="")]),
                    ],
                ),
                io.String.Input("aesthetics", default=""),
                io.String.Input("lighting", default=""),
                io.String.Input("medium", default=""),
                io.Image.Input("image", optional=True),
                io.String.Input("import_json", default="", optional=True, force_input=True),
                io.String.Input("style_palette_data", default="", socketless=True, advanced=True),
                io.String.Input("elements_data", default="", socketless=True, advanced=True),
                io.Int.Input("grid_columns", default=1, min=1, max=12, advanced=True),
                io.Int.Input("grid_rows", default=1, min=1, max=12, advanced=True),
                io.Int.Input("bg_brightness", default=25, min=0, max=100, socketless=True, advanced=True),
                io.Combo.Input("import_mode", options=["when empty", "always"], default="when empty"),
                io.BoundingBox.Input("bboxes", optional=True, force_input=True),
            ],
            outputs=[
                io.String.Output(display_name="prompt"),
                io.Image.Output(display_name="preview"),
                io.BoundingBox.Output(display_name="bboxes"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.String.Output(display_name="current_sheet_json"),
                io.Int.Output(display_name="grid_columns"),
                io.Int.Output(display_name="grid_rows"),
                io.String.Output(display_name="ideoboard_for_frame_v2"),
            ],
        )

    @classmethod
    def execute(
        cls,
        width,
        height,
        background,
        style,
        high_level_description="",
        aesthetics="",
        lighting="",
        medium="",
        style_palette_data="",
        elements_data="",
        import_json="",
        import_mode="when empty",
        grid_columns=1,
        grid_rows=1,
        bboxes=None,
        image=None,
        bg_brightness=25,
    ) -> io.NodeOutput:
        width = int(width)
        height = int(height)
        boxes = _parse_json_list(elements_data)
        boxes_seeded = False
        if not boxes and bboxes:
            if isinstance(bboxes, dict):
                frame = [bboxes]
            elif bboxes and isinstance(bboxes[0], (list, tuple)):
                frame = bboxes[0]
            else:
                frame = bboxes
            for bb in frame:
                if not isinstance(bb, dict):
                    continue
                boxes.append({
                    "x": bb.get("x", 0) / width,
                    "y": bb.get("y", 0) / height,
                    "w": bb.get("width", 0) / width,
                    "h": bb.get("height", 0) / height,
                    "type": "obj",
                    "text": "",
                    "desc": "",
                    "tips": "",
                    "useTips": False,
                    "palette": [],
                })
            boxes_seeded = bool(boxes)

        imported = None
        if import_json and str(import_json).strip():
            try:
                c = json.loads(str(import_json))
                if isinstance(c, dict):
                    imported = c
            except Exception:
                pass

        kind = style["style"] if isinstance(style, dict) and "style" in style else str(style or "none")
        used_import = imported is not None and (import_mode == "always" or not boxes)

        if used_import:
            caption = imported
            boxes = _caption_to_boxes(imported)
        else:
            caption = {}
            if str(high_level_description or "").strip():
                caption["high_level_description"] = high_level_description
            if kind != "none":
                sd = {"aesthetics": aesthetics, "lighting": lighting}
                if kind == "photo":
                    sd["photo"] = style.get("photo", "") if isinstance(style, dict) else ""
                    sd["medium"] = medium
                else:
                    sd["medium"] = medium
                    sd["art_style"] = style.get("art_style", "") if isinstance(style, dict) else ""
                palette = _palette(_parse_json_list(style_palette_data))
                if palette:
                    sd["color_palette"] = palette
                caption["style_description"] = sd

            elements = []
            for box in boxes:
                if not isinstance(box, dict):
                    continue
                etype = "text" if box.get("type") == "text" else "obj"
                elem = {"type": etype}
                if not box.get("nobbox"):
                    elem["bbox"] = _norm_bbox(box)
                if etype == "text":
                    elem["text"] = box.get("text", "")
                # Keep prompt output identical to the reference builder semantics:
                # tips are UI guidance only and never silently rewrite desc.
                elem["desc"] = str(box.get("desc", "") or "")
                palette = _palette(box.get("palette", []))
                if palette:
                    elem["color_palette"] = palette[:5]
                elements.append(elem)

            caption["compositional_deconstruction"] = {
                "background": background,
                "elements": elements,
            }

        bg = None
        if image is not None:
            try:
                bg = Image.fromarray((image[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
            except Exception:
                bg = None
        preview = _render_preview(boxes, width, height, bg, bg_brightness)

        bbox_dicts = []
        for box in boxes:
            if not isinstance(box, dict) or box.get("nobbox"):
                continue
            x, y = float(box.get("x", 0.0) or 0.0), float(box.get("y", 0.0) or 0.0)
            bw, bh = float(box.get("w", 0.0) or 0.0), float(box.get("h", 0.0) or 0.0)
            if bw < 0:
                x += bw
                bw = -bw
            if bh < 0:
                y += bh
                bh = -bh
            bbox_dicts.append({"x": round(x * width), "y": round(y * height), "width": round(bw * width), "height": round(bh * height)})
        bboxes_out = [bbox_dicts] if bbox_dicts else []

        sheet_json = _dumps(caption)

        ui = {"dims": [width, height]}
        if boxes_seeded:
            ui["boxes"] = [json.dumps(boxes)]
        if used_import:
            ui["caption"] = [_dumps(imported)]
        ideoboard_json = json.dumps(_caption_to_ideoboard_design(caption, boxes, width, height, grid_columns, grid_rows), ensure_ascii=False, indent=2)
        return io.NodeOutput(sheet_json, preview, bboxes_out, width, height, sheet_json, int(grid_columns), int(grid_rows), ideoboard_json, ui=ui)


NODE_CLASS_MAPPINGS = {
    "IAMCCS_IdeogramSheetBuilder": IAMCCS_IdeogramSheetBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_IdeogramSheetBuilder": "IAMCCS Ideogram Sheet Builder",
}
