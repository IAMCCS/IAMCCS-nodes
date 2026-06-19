import base64
import io
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw


def _blank_image(width: int, height: int) -> torch.Tensor:
    arr = np.zeros((max(1, height), max(1, width), 3), dtype=np.float32)
    return torch.from_numpy(arr).unsqueeze(0)


def _pil_to_image_tensor(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _image_tensor_to_pil(image: torch.Tensor) -> Image.Image:
    t = image.detach().cpu().float()
    if t.ndim == 4:
        t = t[0]
    arr = t.numpy()
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).round().astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        arr = arr[..., :3]
    return Image.fromarray(arr, "RGB")


def _decode_data_url(data_url: str) -> Image.Image | None:
    if not data_url or not isinstance(data_url, str):
        return None
    marker = "base64,"
    if marker not in data_url:
        return None
    try:
        payload = data_url.split(marker, 1)[1]
        raw = base64.b64decode(payload)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None


def _safe_int(value: Any, default: int, min_value: int = 1, max_value: int = 16384) -> int:
    try:
        out = int(float(value))
    except Exception:
        out = int(default)
    return max(min_value, min(max_value, out))


def _mask_from_strokes(data: Dict[str, Any], width: int, height: int) -> Tuple[torch.Tensor, str]:
    mask_img = Image.new("L", (int(width), int(height)), 0)
    draw = ImageDraw.Draw(mask_img)
    strokes = data.get("strokes") if isinstance(data.get("strokes"), list) else []
    total_points = 0
    painted = 0
    erased = 0

    for stroke in strokes:
        if not isinstance(stroke, dict):
            continue
        raw_points = stroke.get("points") if isinstance(stroke.get("points"), list) else []
        points: List[Tuple[int, int]] = []
        for p in raw_points:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                continue
            try:
                x = float(p[0])
                y = float(p[1])
            except Exception:
                continue
            # Frontend stores natural image pixel coordinates.
            x = max(0.0, min(float(width - 1), x))
            y = max(0.0, min(float(height - 1), y))
            points.append((int(round(x)), int(round(y))))
        if not points:
            continue
        total_points += len(points)
        mode = str(stroke.get("mode") or "paint").strip().lower()
        fill = 0 if mode in {"erase", "eraser"} else 255
        if fill:
            painted += 1
        else:
            erased += 1
        size = _safe_int(stroke.get("size"), _safe_int(data.get("brush_size"), 48, 1, 512), 1, 512)
        radius = max(1, size // 2)

        def cap(cx: int, cy: int) -> None:
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=fill)

        if len(points) == 1:
            cap(points[0][0], points[0][1])
        else:
            for a, b in zip(points, points[1:]):
                draw.line((a, b), fill=fill, width=size)
            for idx, (x, y) in enumerate(points):
                if idx == 0 or idx == len(points) - 1 or idx % 8 == 0:
                    cap(x, y)

    arr = np.asarray(mask_img).astype(np.float32) / 255.0
    mask = torch.from_numpy(arr).unsqueeze(0)
    report = (
        f"goyai_paint mask shape=1x{height}x{width} strokes={len(strokes)} "
        f"painted={painted} erased={erased} points={total_points} "
        f"mean={float(arr.mean()) if arr.size else 0.0:.6f} max={float(arr.max()) if arr.size else 0.0:.3f}"
    )
    return mask, report


class IAMCCS_GoyAICanvasPaint:
    """Small dedicated image+mask painter for Ideogram/i2i workflows.

    It intentionally avoids FrameDesigner layer gestures. The frontend writes a
    single JSON payload containing an imported image data URL and vector strokes;
    the backend returns ComfyUI IMAGE + MASK.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "paint_data": ("STRING", {"default": "", "multiline": True}),
                "width": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 576, "min": 1, "max": 8192, "step": 8}),
                "fallback": (["blank", "connected_image"], {"default": "connected_image"}),
            },
            "optional": {
                "source_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "report")
    FUNCTION = "paint"
    CATEGORY = "IAMCCS/Ideogram"

    def paint(self, paint_data: str = "", width: int = 1024, height: int = 576, fallback: str = "connected_image", source_image=None):
        data: Dict[str, Any] = {}
        if isinstance(paint_data, str) and paint_data.strip():
            try:
                data = json.loads(paint_data)
            except Exception:
                data = {}

        img = _decode_data_url(str(data.get("image_data") or ""))
        if img is not None:
            width = int(img.width)
            height = int(img.height)
            image_tensor = _pil_to_image_tensor(img)
            source = "embedded_image"
        elif source_image is not None and str(fallback) == "connected_image":
            pil = _image_tensor_to_pil(source_image)
            width = int(pil.width)
            height = int(pil.height)
            image_tensor = _pil_to_image_tensor(pil)
            source = "connected_image"
        else:
            width = _safe_int(data.get("width"), width, 1, 8192)
            height = _safe_int(data.get("height"), height, 1, 8192)
            image_tensor = _blank_image(width, height)
            source = "blank"

        mask, mask_report = _mask_from_strokes(data, int(width), int(height))
        report = f"IAMCCS GoyAIcanvas Paint source={source} image=1x{height}x{width} | {mask_report}"
        return image_tensor, mask, report
