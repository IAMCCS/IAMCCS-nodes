import json
import textwrap

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def _to_pil(img):
    arr = (img.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _font(size, bold=False):
    try:
        name = "arialbd.ttf" if bold else "arial.ttf"
        return ImageFont.truetype(name, size)
    except Exception:
        return ImageFont.load_default()


def _captions(prompt_json, count):
    try:
        data = json.loads(str(prompt_json or "{}"))
        elements = data.get("compositional_deconstruction", {}).get("elements", [])
        out = []
        for i in range(count):
            desc = str((elements[i] or {}).get("desc", "") if i < len(elements) else "")
            out.append(desc)
        return out
    except Exception:
        return [""] * count


class IAMCCS_StoryboardPromptContactSheet:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "prompt_json": ("STRING", {"forceInput": True}),
                "columns": ("INT", {"default": 3, "min": 1, "max": 12}),
                "rows": ("INT", {"default": 3, "min": 1, "max": 12}),
                "caption_height": ("INT", {"default": 120, "min": 40, "max": 512}),
                "gap": ("INT", {"default": 6, "min": 0, "max": 64}),
                "title": ("STRING", {"default": "IAMCCS Storyboard"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("storyboard",)
    FUNCTION = "render"
    CATEGORY = "IAMCCS/Cine/Ideogram"

    def render(self, images, prompt_json, columns=3, rows=3, caption_height=120, gap=6, title="IAMCCS Storyboard"):
        if images is None:
            raise ValueError("Missing images")
        columns = max(1, int(columns))
        rows = max(1, int(rows))
        gap = max(0, int(gap))
        caption_height = max(40, int(caption_height))
        count = min(int(images.shape[0]), columns * rows)
        panels = [_to_pil(images[i]) for i in range(count)]
        if not panels:
            raise ValueError("No images in batch")
        pw, ph = panels[0].size
        title_h = 64 if str(title or "").strip() else 0
        cell_h = ph + caption_height
        out_w = columns * pw + (columns + 1) * gap
        out_h = title_h + rows * cell_h + (rows + 1) * gap
        canvas = Image.new("RGB", (out_w, out_h), (238, 234, 224))
        draw = ImageDraw.Draw(canvas)
        title_font = _font(34, True)
        head_font = _font(18, True)
        body_font = _font(15)
        if title_h:
            draw.text((gap * 2, 14), str(title), fill=(20, 20, 20), font=title_font)
        caps = _captions(prompt_json, count)
        for i, panel in enumerate(panels):
            row = i // columns
            col = i % columns
            x = gap + col * (pw + gap)
            y = title_h + gap + row * (cell_h + gap)
            canvas.paste(panel, (x, y))
            draw.rectangle([x, y + ph, x + pw, y + ph + caption_height], fill=(250, 249, 245), outline=(200, 196, 188))
            draw.text((x + 8, y + ph + 8), f"Panel {i + 1}", fill=(15, 15, 15), font=head_font)
            cap = caps[i].replace("\\n", " ")
            words = textwrap.wrap(cap, width=max(28, int(pw / 9)))
            ty = y + ph + 34
            for line in words[:5]:
                draw.text((x + 8, ty), line, fill=(55, 55, 55), font=body_font)
                ty += 18
        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        return (torch.from_numpy(arr).unsqueeze(0),)


NODE_CLASS_MAPPINGS = {"IAMCCS_StoryboardPromptContactSheet": IAMCCS_StoryboardPromptContactSheet}
NODE_DISPLAY_NAME_MAPPINGS = {"IAMCCS_StoryboardPromptContactSheet": "IAMCCS Storyboard Prompt Contact Sheet"}
