"""Optional deterministic lettering for MOSS Result; independent of diffusion."""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


def _font(size):
    # Pillow searches installed platform fonts; no workstation-specific paths.
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def _wrap(text, draw, font, width):
    lines = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split(" "):
            candidate = line + (" " if line else "") + word
            if line and draw.textlength(candidate, font=font) > width:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    return "\n".join(lines)


def compose_text_layers(images, design):
    layers = [item for item in design.get("items", [])
              if item.get("kind") == "text" and item.get("text_mode") == "overlay" and item.get("text")]
    if not layers:
        return images, []
    height, width = images.shape[1:3]
    prepared = []
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    for item in layers:
        values = [float(item.get(key, default)) for key, default in (("x", 0), ("y", 0), ("w", 1000), ("h", 1000))]
        if not all(math.isfinite(value) for value in values) or min(values[2:]) <= 0:
            raise ValueError("Text overlay requires finite coordinates and a positive box size.")
        x, y, w, h = values
        left, top = max(0, math.floor(x * width / 1000)), max(0, math.floor(y * height / 1000))
        right, bottom = min(width, math.ceil((x + w) * width / 1000)), min(height, math.ceil((y + h) * height / 1000))
        if right <= left or bottom <= top:
            raise ValueError("Text overlay box is outside the image.")
        box_width, box_height = right - left, bottom - top
        preferred = float(item.get("font_size", 0) or 0)
        if not math.isfinite(preferred) or preferred < 0:
            raise ValueError("Text size must be a finite positive number or zero for automatic sizing.")
        size = min(max(1, round(preferred or height * 0.045)), box_height, box_width)
        text = str(item["text"])
        while size >= 1:
            font = _font(size)
            stroke = max(1, round(size / 18))
            spacing = max(1, round(size / 6))
            wrapped = _wrap(text, probe, font, max(1, box_width - stroke * 4))
            bounds = probe.multiline_textbbox((0, 0), wrapped, font=font, spacing=spacing, stroke_width=stroke)
            if bounds[2] - bounds[0] <= box_width and bounds[3] - bounds[1] <= box_height:
                break
            size -= 1
        if size < 1:
            raise ValueError("Text cannot fit its box. Enlarge the box or shorten the lettering.")
        prepared.append((left, top, right, bottom, wrapped, font, stroke, spacing, bounds))
    frames = []
    for frame in images:
        image = Image.fromarray((frame.detach().cpu().clamp(0, 1).numpy() * 255).round().astype("uint8"))
        for left, top, right, bottom, wrapped, font, stroke, spacing, bounds in prepared:
            layer = Image.new("RGBA", (right - left, bottom - top))
            ImageDraw.Draw(layer).multiline_text((-bounds[0], -bounds[1]), wrapped, font=font, fill="white",
                                                stroke_width=stroke, stroke_fill="black", spacing=spacing)
            image.paste(layer, (left, top), layer)
        frames.append(torch.from_numpy(np.asarray(image).copy()).to(device=images.device, dtype=images.dtype) / 255)
    metadata = [{"id": item.get("id"), "text": str(item["text"]), "mode": "post_generation_overlay"} for item in layers]
    return torch.stack(frames), metadata
