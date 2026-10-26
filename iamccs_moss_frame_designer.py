"""Independent MOSS layout compiler for the IAMCCS MOSS frame editor."""
import base64
import io
import json
import math
from pathlib import Path

import folder_paths
import numpy as np
from PIL import Image
import torch


def image_from_project(design):
    generated = design.get("generated", {})
    asset = generated.get("data_url", "")
    if asset:
        if not asset.startswith("data:image/png;base64,"):
            raise ValueError("Mboard generated image must be an embedded PNG.")
        source = io.BytesIO(base64.b64decode(asset.split(",", 1)[1], validate=True))
    elif generated.get("files"):
        file = generated["files"][0]
        directory = folder_paths.get_directory_by_type(file.get("type", "output"))
        if directory is None:
            raise ValueError("Unknown Mboard image directory type.")
        root = Path(directory).resolve()
        source = (root / file.get("subfolder", "") / file["filename"]).resolve()
        if not source.is_relative_to(root):
            raise ValueError("Mboard image must stay inside its ComfyUI media directory.")
    else:
        return None
    with Image.open(source) as image:
        return torch.from_numpy(np.array(image.convert("RGB"), dtype=np.float32) / 255.0).unsqueeze(0)


def image_linx(design, image=None):
    transport = design.get("h3", {})
    slot = int(transport.get("slot", 1))
    role = transport.get("role", "composition")
    if slot not in (1, 2, 3, 4) or role not in ("composition", "subject_identity", "style", "keyframe"):
        raise ValueError("Select an H3 picture slot 1–4 and a supported reference role.")
    resources = {"iamccs_moss_frame_project": design}
    if image is not None:
        resources.update({"rendered_images": image, f"iamccs_minimax_h3_ref_image_{slot}": image})
        roles = ["disabled"] * 4
        roles[slot - 1] = role
        resources["iamccs_minimax_h3_cine_info"] = {"reference_roles": roles}
        target = transport.get("target", "pictures")
        if target in ("swap_subject", "swap_environment"):
            encoded = design.get("generated", {}).get("data_url")
            if not encoded:
                buffer = io.BytesIO()
                Image.fromarray((image[0].detach().cpu().clamp(0, 1).numpy() * 255).round().astype("uint8")).save(buffer, format="PNG")
                encoded = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
            key = "iamccs_h3_swap_reference" if target == "swap_subject" else "iamccs_h3_edit_environment"
            resources[key] = {"data_url": encoded}
    return {"type": "IAMCCS_SUPERNODE_LINX", "resources": resources,
            "outputs": {"moss_image_ready": image is not None},
            "resource_keys": sorted(resources),
            "chain": [{"role": "moss_frame_designer", "name": "IAMCCS Frame Designer MOSS"}]}


DEFAULT_DESIGN = {
    "schema": "iamccs.moss.frame_designer",
    "schema_version": 1,
    "canvas": {"width": 1280, "height": 720},
    "scene": {
        "high_level_description": "A live-action cinematic photograph of exactly two adults standing apart on a railway platform. Both people are visible from head to toe, with a clear empty gap between them.",
        "lighting": "Soft overcast daylight, consistent natural shadows.",
        "photo": "Eye-level wide shot, natural perspective, detailed fabric and unretouched skin.",
        "background": "An empty weathered railway platform, muted stone architecture and distant tracks.",
    },
    "items": [
        {"id": "woman_left", "kind": "obj", "role": "character", "label": "Woman / left",
         "x": 80, "y": 100, "w": 300, "h": 820,
         "desc": "One adult woman with short dark hair, wearing a bright red knee-length coat, dark trousers and black boots. She stands facing the camera, hands relaxed at her sides. Entire body including both boots visible.",
         "color_palette": ["#b52d35"]},
        {"id": "man_right", "kind": "obj", "role": "character", "label": "Man / right",
         "x": 620, "y": 100, "w": 300, "h": 820,
         "desc": "One adult man with a short gray beard, wearing a cobalt blue jacket, beige trousers and brown shoes. He stands facing the camera, hands relaxed at his sides. Entire body including both shoes visible.",
         "color_palette": ["#2656b8"]},
    ],
}


def compile_layout(raw):
    design = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(design, dict):
        raise ValueError("MOSS design must be a JSON object with canvas, scene and items.")
    canvas = design.get("canvas", {})
    width, height = int(canvas.get("width", 1024)), int(canvas.get("height", 1024))
    if width <= 0 or height <= 0:
        raise ValueError("Canvas width and height must be positive.")
    scene = design.get("scene", {})
    caption = " ".join(str(scene.get(k, "")).strip() for k in (
        "high_level_description", "background", "aesthetics", "lighting", "photo", "medium", "art_style"
    ) if scene.get(k))
    items, elements, boxes, warnings = [], [], [], []
    for index, source in enumerate(design.get("items", [])):
        kind = source.get("kind", "obj")
        if kind in ("image", "mask"):
            warnings.append(f"Layer {index + 1}: {kind} is an editor reference, not image conditioning.")
            continue
        role = "text" if kind == "text" else source.get("role", "object")
        if role not in ("object", "character", "text"):
            raise ValueError(f"Region {index + 1}: role must be object, character or text.")
        x, y, w, h = [float(source.get(k, v)) for k, v in (("x", 0), ("y", 0), ("w", 1000), ("h", 1000))]
        if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
            raise ValueError(f"Region {index + 1}: use finite coordinates and positive dimensions.")
        x1, y1 = max(0, min(1000, x)), max(0, min(1000, y))
        x2, y2 = max(0, min(1000, x + w)), max(0, min(1000, y + h))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Region {index + 1}: box is outside the canvas.")
        if (x1, y1, x2, y2) != (x, y, x + w, y + h):
            warnings.append(f"Region {index + 1}: box clipped to the canvas.")
        left, top = math.floor(x1 * width / 1000), math.floor(y1 * height / 1000)
        right, bottom = math.ceil(x2 * width / 1000), math.ceil(y2 * height / 1000)
        desc = str(source.get("desc", "")).strip()
        style = str(source.get("style", "")).strip()
        if style:
            desc = " ".join(filter(None, (desc, f"Rendering treatment for this region: {style}")))
        if role == "text" and source.get("text_mode", "native") == "overlay":
            warnings.append(f"Region {index + 1}: exact text is applied after diffusion by MOSS Result.")
            continue
        item = {"label": role, "caption": desc,
                "bbox": [top - height // 2, left - width // 2, bottom - height // 2, right - width // 2]}
        el = {"type": "text" if role == "text" else "obj", "desc": desc, "bbox": [y1, x1, y2, x2]}
        if role == "text":
            item["text"] = str(source.get("text", ""))
            el["text"] = item["text"]
        palette = source.get("color_palette", [])
        if palette:
            item["palette"] = palette
            el["color_palette"] = palette
        items.append(item)
        elements.append(el)
        boxes.append({"x": left, "y": top, "width": right - left, "height": bottom - top,
                      "metadata": {"type": el["type"], "desc": desc, "text": el.get("text", ""), "palette": palette}})
    layout = {"summary": {"caption": caption}, "items": items,
              "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
    report = {"edition": "MOSS", "canvas": [width, height], "regions": len(items),
              "characters": sum(i["label"] == "character" for i in items), "warnings": warnings,
              "note": "Layout guidance regenerates the whole image; regions are not hard masks or identity references."}
    return json.dumps(layout, ensure_ascii=False, separators=(",", ":")), width, height, elements, [boxes], report


class IAMCCS_MOSSFrameDesigner:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"design_data": ("STRING", {
            "default": json.dumps(DEFAULT_DESIGN, ensure_ascii=False), "multiline": True,
            "tooltip": "Editable MOSS project. Import existing Frame Designer V2 projects using the editor."})}}

    RETURN_TYPES = ("STRING", "INT", "INT", "ARRAY", "BOUNDING_BOX", "STRING", "IAMCCS_SUPERNODE_LINX", "STRING")
    RETURN_NAMES = ("layout_prompt", "width", "height", "elements", "bboxes", "report", "cine_linx", "project_data")
    FUNCTION = "compile"
    CATEGORY = "IAMCCS/Cine/MOSS"
    DESCRIPTION = "Visual frame design with native MOSS character, object and text bounding boxes. Uses a separate project and compiler from Ideogram."

    def compile(self, design_data):
        design = json.loads(design_data)
        prompt, width, height, elements, boxes, report = compile_layout(design_data)
        report_json = json.dumps(report, ensure_ascii=False)
        return {"ui": {"moss_prompt": [prompt], "moss_report": [report_json]},
                "result": (prompt, width, height, elements, boxes, report_json,
                           image_linx(design, image_from_project(design)), design_data)}


NODE_CLASS_MAPPINGS = {"IAMCCS_MOSSFrameDesigner": IAMCCS_MOSSFrameDesigner}
NODE_DISPLAY_NAME_MAPPINGS = {"IAMCCS_MOSSFrameDesigner": "IAMCCS Frame Designer — MOSS Edition"}
