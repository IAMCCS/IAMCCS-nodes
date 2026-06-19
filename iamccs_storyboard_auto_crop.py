import json

import torch
import torch.nn.functional as F


def _edges(total, parts):
    return [round(i * total / parts) for i in range(parts + 1)]


def _resize_image_batch(image, width, height):
    if image.shape[1] == height and image.shape[2] == width:
        return image
    nchw = image.movedim(-1, 1)
    resized = F.interpolate(nchw, size=(height, width), mode="bilinear", align_corners=False)
    return resized.movedim(1, -1).clamp(0, 1)


def _clamp_int(value, min_value, max_value, fallback):
    try:
        number = int(round(float(value)))
    except Exception:
        number = int(fallback)
    return max(int(min_value), min(int(max_value), number))


def _parse_json_maybe(value):
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    if not text:
        return {}
    candidates = [text]
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            stripped = part.strip()
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                candidates.append(stripped)
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        candidates.append(text[first:last + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def _cluster_count(values, tolerance=70):
    cleaned = sorted(float(v) for v in values if v is not None)
    if not cleaned:
        return 0
    groups = []
    for value in cleaned:
        if not groups or abs(value - groups[-1][-1]) > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return len(groups)


def _infer_grid_from_prompt_json(prompt_json, fallback_columns, fallback_rows):
    data = _parse_json_maybe(prompt_json)
    elements = []
    comp = data.get("compositional_deconstruction") if isinstance(data, dict) else None
    if isinstance(comp, list):
        elements = comp
    elif isinstance(comp, dict):
        elements = comp.get("elements") or comp.get("objects") or comp.get("bboxes") or []
    if not isinstance(elements, list):
        elements = []
    centers_x = []
    centers_y = []
    for entry in elements:
        if not isinstance(entry, dict):
            continue
        bbox = entry.get("bbox") or entry.get("box") or entry.get("bounding_box")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            y0, x0, y1, x1 = [float(v) for v in bbox]
        except Exception:
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        centers_x.append((x0 + x1) / 2.0)
        centers_y.append((y0 + y1) / 2.0)
    columns = _cluster_count(centers_x) or int(fallback_columns)
    rows = _cluster_count(centers_y) or int(fallback_rows)
    return max(1, columns), max(1, rows)


def _fit_crop_to_aspect(x0, y0, x1, y1, target_aspect):
    if target_aspect <= 0:
        return x0, y0, x1, y1
    width = max(1, int(x1 - x0))
    height = max(1, int(y1 - y0))
    current_aspect = width / float(height)
    if abs(current_aspect - target_aspect) < 0.001:
        return x0, y0, x1, y1
    if current_aspect > target_aspect:
        new_w = max(1, int(round(height * target_aspect)))
        inset = max(0, (width - new_w) // 2)
        x0 += inset
        x1 = x0 + new_w
    else:
        new_h = max(1, int(round(width / target_aspect)))
        inset = max(0, (height - new_h) // 2)
        y0 += inset
        y1 = y0 + new_h
    return int(x0), int(y0), int(x1), int(y1)


class IAMCCS_StoryboardAutoCropGrid:
    DISPLAY_NAME = "IAMCCS Storyboard Auto Crop Grid"
    CATEGORY = "IAMCCS/Cine/Ideogram"
    FUNCTION = "crop"
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("images", "count", "panel_width", "panel_height", "crop_manifest_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "columns": ("INT", {"default": 2, "min": 1, "max": 12, "step": 1}),
                "rows": ("INT", {"default": 3, "min": 1, "max": 12, "step": 1}),
                "panel_order": (["column_major", "row_major"], {"default": "column_major"}),
                "output_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16}),
                "output_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16}),
                "crop_margin_px": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
            }
        }

    def crop(self, image, columns=2, rows=3, panel_order="column_major", output_width=0, output_height=0, crop_margin_px=0):
        if image is None:
            raise ValueError("IAMCCS Storyboard Auto Crop Grid: missing image")

        columns = max(1, int(columns))
        rows = max(1, int(rows))
        margin = max(0, int(crop_margin_px))
        b, h, w, c = image.shape

        x_edges = _edges(w, columns)
        y_edges = _edges(h, rows)
        target_w = int(output_width) if int(output_width) > 0 else max(1, round(w / columns) - margin * 2)
        target_h = int(output_height) if int(output_height) > 0 else max(1, round(h / rows) - margin * 2)

        order = []
        if str(panel_order) == "row_major":
            for row in range(rows):
                for col in range(columns):
                    order.append((col, row))
        else:
            for col in range(columns):
                for row in range(rows):
                    order.append((col, row))

        crops = []
        manifest = {
            "source_width": int(w),
            "source_height": int(h),
            "columns": columns,
            "rows": rows,
            "panel_order": str(panel_order),
            "output_width": target_w,
            "output_height": target_h,
            "crop_margin_px": margin,
            "panels": [],
        }

        for frame_index in range(int(b)):
            frame = image[frame_index:frame_index + 1]
            for panel_index, (col, row) in enumerate(order, start=1):
                x0, x1 = x_edges[col], x_edges[col + 1]
                y0, y1 = y_edges[row], y_edges[row + 1]
                if margin:
                    x0 = min(max(x0 + margin, 0), w - 1)
                    y0 = min(max(y0 + margin, 0), h - 1)
                    x1 = max(min(x1 - margin, w), x0 + 1)
                    y1 = max(min(y1 - margin, h), y0 + 1)
                panel = frame[:, y0:y1, x0:x1, :]
                panel = _resize_image_batch(panel, target_w, target_h)
                crops.append(panel)
                manifest["panels"].append({
                    "frame": frame_index,
                    "panel": panel_index,
                    "col": col,
                    "row": row,
                    "x": int(x0),
                    "y": int(y0),
                    "width": int(x1 - x0),
                    "height": int(y1 - y0),
                })

        out = torch.cat(crops, dim=0)
        return (out, int(out.shape[0]), target_w, target_h, json.dumps(manifest, ensure_ascii=False, indent=2))


class IAMCCS_StoryboardAutoCropGridPRO(IAMCCS_StoryboardAutoCropGrid):
    DISPLAY_NAME = "IAMCCS Storyboard Auto Crop Grid PRO"
    CATEGORY = "IAMCCS/Cine/Ideogram"
    FUNCTION = "crop_pro"
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("images", "count", "panel_width", "panel_height", "crop_manifest_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "columns": ("INT", {"default": 2, "min": 1, "max": 12, "step": 1}),
                "rows": ("INT", {"default": 3, "min": 1, "max": 12, "step": 1}),
                "panel_order": (["column_major", "row_major"], {"default": "row_major"}),
                "fit_mode": (["crop_to_target_aspect", "resize_to_target", "preserve_cell_size"], {"default": "crop_to_target_aspect"}),
                "output_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16}),
                "output_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16}),
                "crop_margin_px": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
            },
            "optional": {
                "IDEO_LINX": ("IDEO_LINX",),
                "prompt_json": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "target_panel_width": ("INT", {"default": 0, "min": 0, "max": 16384, "forceInput": True}),
                "target_panel_height": ("INT", {"default": 0, "min": 0, "max": 16384, "forceInput": True}),
                "canvas_width": ("INT", {"default": 0, "min": 0, "max": 32768, "forceInput": True}),
                "canvas_height": ("INT", {"default": 0, "min": 0, "max": 32768, "forceInput": True}),
            },
        }

    def crop_pro(
        self,
        image,
        columns=2,
        rows=3,
        panel_order="row_major",
        fit_mode="crop_to_target_aspect",
        output_width=0,
        output_height=0,
        crop_margin_px=0,
        IDEO_LINX=None,
        prompt_json="",
        target_panel_width=0,
        target_panel_height=0,
        canvas_width=0,
        canvas_height=0,
    ):
        if image is None:
            raise ValueError("IAMCCS Storyboard Auto Crop Grid PRO: missing image")

        linx = IDEO_LINX if isinstance(IDEO_LINX, dict) else {}
        prompt_source = prompt_json or linx.get("prompt_json") or ""
        image_b, image_h, image_w, _ = image.shape

        linx_width = _clamp_int(linx.get("width"), 0, 32768, 0)
        linx_height = _clamp_int(linx.get("height"), 0, 32768, 0)
        canvas_w = _clamp_int(canvas_width or linx_width, 0, 32768, 0)
        canvas_h = _clamp_int(canvas_height or linx_height, 0, 32768, 0)
        target_w_in = _clamp_int(target_panel_width or linx.get("target_panel_width"), 0, 16384, 0)
        target_h_in = _clamp_int(target_panel_height or linx.get("target_panel_height"), 0, 16384, 0)

        columns = _clamp_int(columns, 1, 12, 2)
        rows = _clamp_int(rows, 1, 12, 3)

        if canvas_w > 0 and canvas_h > 0 and target_w_in > 0 and target_h_in > 0:
            inferred_columns = round(canvas_w / float(target_w_in))
            inferred_rows = round(canvas_h / float(target_h_in))
            if 1 <= inferred_columns <= 12 and 1 <= inferred_rows <= 12:
                columns = int(inferred_columns)
                rows = int(inferred_rows)
        else:
            columns, rows = _infer_grid_from_prompt_json(prompt_source, columns, rows)

        if target_w_in <= 0 and canvas_w > 0:
            target_w_in = max(1, round(canvas_w / float(max(1, columns))))
        if target_h_in <= 0 and canvas_h > 0:
            target_h_in = max(1, round(canvas_h / float(max(1, rows))))

        margin = max(0, int(crop_margin_px))
        x_edges = _edges(int(image_w), columns)
        y_edges = _edges(int(image_h), rows)

        if int(output_width) > 0:
            target_w = int(output_width)
        elif target_w_in > 0:
            target_w = int(target_w_in)
        else:
            target_w = max(1, round(int(image_w) / columns) - margin * 2)

        if int(output_height) > 0:
            target_h = int(output_height)
        elif target_h_in > 0:
            target_h = int(target_h_in)
        else:
            target_h = max(1, round(int(image_h) / rows) - margin * 2)

        target_aspect = target_w / float(max(1, target_h))
        order = []
        if str(panel_order) == "row_major":
            for row in range(rows):
                for col in range(columns):
                    order.append((col, row))
        else:
            for col in range(columns):
                for row in range(rows):
                    order.append((col, row))

        crops = []
        manifest = {
            "source_width": int(image_w),
            "source_height": int(image_h),
            "source_batch": int(image_b),
            "columns": int(columns),
            "rows": int(rows),
            "panel_order": str(panel_order),
            "fit_mode": str(fit_mode),
            "output_width": int(target_w),
            "output_height": int(target_h),
            "crop_margin_px": int(margin),
            "auto_source": "IDEO_LINX" if linx else ("prompt_json" if prompt_source else "widgets"),
            "canvas_width": int(canvas_w),
            "canvas_height": int(canvas_h),
            "target_panel_width": int(target_w_in),
            "target_panel_height": int(target_h_in),
            "panels": [],
        }

        for frame_index in range(int(image_b)):
            frame = image[frame_index:frame_index + 1]
            for panel_index, (col, row) in enumerate(order, start=1):
                x0, x1 = x_edges[col], x_edges[col + 1]
                y0, y1 = y_edges[row], y_edges[row + 1]
                if margin:
                    x0 = min(max(x0 + margin, 0), int(image_w) - 1)
                    y0 = min(max(y0 + margin, 0), int(image_h) - 1)
                    x1 = max(min(x1 - margin, int(image_w)), x0 + 1)
                    y1 = max(min(y1 - margin, int(image_h)), y0 + 1)
                cell_x0, cell_y0, cell_x1, cell_y1 = int(x0), int(y0), int(x1), int(y1)
                if str(fit_mode) == "crop_to_target_aspect":
                    x0, y0, x1, y1 = _fit_crop_to_aspect(cell_x0, cell_y0, cell_x1, cell_y1, target_aspect)
                else:
                    x0, y0, x1, y1 = cell_x0, cell_y0, cell_x1, cell_y1
                panel = frame[:, y0:y1, x0:x1, :]
                if str(fit_mode) == "preserve_cell_size" and int(output_width) <= 0 and int(output_height) <= 0 and target_w_in <= 0 and target_h_in <= 0:
                    panel_out = panel
                    out_w = int(panel.shape[2])
                    out_h = int(panel.shape[1])
                else:
                    panel_out = _resize_image_batch(panel, int(target_w), int(target_h))
                    out_w = int(target_w)
                    out_h = int(target_h)
                crops.append(panel_out)
                manifest["panels"].append({
                    "frame": int(frame_index),
                    "panel": int(panel_index),
                    "col": int(col),
                    "row": int(row),
                    "cell_x": int(cell_x0),
                    "cell_y": int(cell_y0),
                    "cell_width": int(cell_x1 - cell_x0),
                    "cell_height": int(cell_y1 - cell_y0),
                    "x": int(x0),
                    "y": int(y0),
                    "width": int(x1 - x0),
                    "height": int(y1 - y0),
                    "output_width": int(out_w),
                    "output_height": int(out_h),
                })

        out = torch.cat(crops, dim=0)
        return (out, int(out.shape[0]), int(out.shape[2]), int(out.shape[1]), json.dumps(manifest, ensure_ascii=False, indent=2))


NODE_CLASS_MAPPINGS = {
    "IAMCCS_StoryboardAutoCropGrid": IAMCCS_StoryboardAutoCropGrid,
    "IAMCCS_StoryboardAutoCropGridPRO": IAMCCS_StoryboardAutoCropGridPRO,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_StoryboardAutoCropGrid": "IAMCCS Storyboard Auto Crop Grid",
    "IAMCCS_StoryboardAutoCropGridPRO": "IAMCCS Storyboard Auto Crop Grid PRO",
}
