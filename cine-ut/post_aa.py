from __future__ import annotations

import json
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

try:
    from comfy.utils import ProgressBar
except Exception:
    ProgressBar = None

try:
    import comfy.model_management as mm
except Exception:
    mm = None


LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)
EDGE_FADE_WIDTH = 0.10
EDGE_EPSILON = 1e-6

SOBEL_X = (
    (-1.0, 0.0, 1.0),
    (-2.0, 0.0, 2.0),
    (-1.0, 0.0, 1.0),
)

SOBEL_Y = (
    (-1.0, -2.0, -1.0),
    (0.0, 0.0, 0.0),
    (1.0, 2.0, 1.0),
)


def _report(data: Dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def _clamp_float(value: Any, lo: float, hi: float, fallback: float) -> float:
    try:
        number = float(value)
    except Exception:
        number = float(fallback)
    return max(float(lo), min(float(hi), number))


def _preset_values(preset: str) -> Dict[str, float]:
    presets = {
        "Preview": {"strength": 0.35, "threshold": 0.18, "radius": 1, "temporal": 0.00, "protect": 0.70, "chroma": 0.00},
        "Light": {"strength": 0.55, "threshold": 0.14, "radius": 1, "temporal": 0.06, "protect": 0.60, "chroma": 0.05},
        "Balanced": {"strength": 0.85, "threshold": 0.10, "radius": 1, "temporal": 0.12, "protect": 0.50, "chroma": 0.10},
        "Strong": {"strength": 1.15, "threshold": 0.08, "radius": 2, "temporal": 0.18, "protect": 0.42, "chroma": 0.18},
        "Crisp Lines": {"strength": 0.95, "threshold": 0.07, "radius": 1, "temporal": 0.10, "protect": 0.82, "chroma": 0.12},
        "Low VRAM": {"strength": 0.65, "threshold": 0.13, "radius": 1, "temporal": 0.04, "protect": 0.65, "chroma": 0.04},
    }
    return dict(presets.get(str(preset), presets["Balanced"]))


class IAMCCS_CinePPAntiAlias:
    """1:1 post-production anti-aliasing designed for image batches and video frames."""

    def __init__(self):
        self._sobel_cache: Dict[Tuple[str, str], Tuple[torch.Tensor, torch.Tensor]] = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "preset": (["Auto", "Preview", "Light", "Balanced", "Strong", "Crisp Lines", "Low VRAM"], {"default": "Auto"}),
                "hardware_mode": (["auto", "low_vram", "balanced", "quality", "cpu_safe"], {"default": "auto"}),
                "aa_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "edge_threshold": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "blur_radius": ("INT", {"default": 0, "min": 0, "max": 4, "step": 1}),
                "temporal_stability": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 0.75, "step": 0.01}),
                "detail_protection": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "chroma_cleanup": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "max_frames_per_chunk": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "apply"
    CATEGORY = "IAMCCS/Cine-PP/Postproduction"

    def _sobel(self, dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (str(dtype), str(device))
        if key not in self._sobel_cache:
            sx = torch.tensor(SOBEL_X, dtype=dtype, device=device).view(1, 1, 3, 3)
            sy = torch.tensor(SOBEL_Y, dtype=dtype, device=device).view(1, 1, 3, 3)
            self._sobel_cache[key] = (sx, sy)
        return self._sobel_cache[key]

    @staticmethod
    def _blur(rgb: torch.Tensor, radius: int) -> torch.Tensor:
        radius = max(0, int(radius))
        if radius <= 0:
            return rgb
        kernel = radius * 2 + 1
        padded = F.pad(rgb, (radius, radius, radius, radius), mode="replicate")
        return F.avg_pool2d(padded, kernel_size=kernel, stride=1, padding=0)

    def _auto_chunk(self, images: torch.Tensor, radius: int, hardware_mode: str, requested: int) -> int:
        frames = int(images.shape[0])
        if requested > 0:
            return max(1, min(frames, int(requested)))

        h, w, c = int(images.shape[1]), int(images.shape[2]), int(images.shape[3])
        megapixels = max(0.1, (h * w) / 1_000_000.0)
        mode = str(hardware_mode or "auto")

        if mode == "low_vram":
            base = 2
        elif mode == "quality":
            base = 12
        elif mode == "cpu_safe":
            base = 1
        else:
            base = 6

        chunk = max(1, int(base / max(1.0, megapixels / 2.0)))
        if radius >= 3:
            chunk = max(1, chunk // 2)

        if images.device.type == "cuda":
            try:
                free, _total = torch.cuda.mem_get_info(images.device)
                bytes_per_frame = h * w * c * max(4, images.element_size()) * 18
                budget_ratio = {"low_vram": 0.16, "balanced": 0.28, "quality": 0.42}.get(mode, 0.24)
                memory_chunk = max(1, int((free * budget_ratio) // max(1, bytes_per_frame)))
                chunk = min(chunk, memory_chunk)
            except Exception:
                pass

        return max(1, min(frames, chunk))

    def _effective_preset(
        self,
        images: torch.Tensor,
        preset: str,
        hardware_mode: str,
        aa_strength: float,
        edge_threshold: float,
        blur_radius: int,
        temporal_stability: float,
        detail_protection: float,
        chroma_cleanup: float,
    ) -> Tuple[str, Dict[str, float]]:
        selected = str(preset or "Auto")
        mode = str(hardware_mode or "auto")
        h, w = int(images.shape[1]), int(images.shape[2])
        pixels = h * w
        frames = int(images.shape[0])

        if selected == "Auto":
            if mode == "low_vram" or pixels >= 3840 * 2160 or frames >= 97:
                selected = "Low VRAM"
            elif mode == "quality" and pixels <= 1920 * 1080 and frames <= 49:
                selected = "Strong"
            else:
                selected = "Balanced"

        values = _preset_values(selected)
        values["strength"] = values["strength"] * _clamp_float(aa_strength, 0.0, 2.0, 1.0)
        if float(edge_threshold) >= 0.0:
            values["threshold"] = _clamp_float(edge_threshold, 0.0, 1.0, values["threshold"])
        if int(blur_radius or 0) > 0:
            values["radius"] = int(max(0, min(4, int(blur_radius))))
        if float(temporal_stability) >= 0.0:
            values["temporal"] = _clamp_float(temporal_stability, 0.0, 0.75, values["temporal"])
        if float(detail_protection) >= 0.0:
            values["protect"] = _clamp_float(detail_protection, 0.0, 1.0, values["protect"])
        if float(chroma_cleanup) >= 0.0:
            values["chroma"] = _clamp_float(chroma_cleanup, 0.0, 1.0, values["chroma"])

        if mode == "low_vram":
            values["radius"] = min(int(values["radius"]), 1)
            values["temporal"] = min(float(values["temporal"]), 0.08)
        elif mode == "quality":
            values["radius"] = min(4, max(int(values["radius"]), 1))

        return selected, values

    def _apply_chunk(self, chunk: torch.Tensor, values: Dict[str, float]) -> torch.Tensor:
        img = chunk.permute(0, 3, 1, 2).contiguous()
        if img.shape[1] not in (3, 4):
            raise ValueError("IAMCCS Cine-PP AntiAlias expects RGB or RGBA IMAGE tensors.")

        has_alpha = img.shape[1] == 4
        alpha = img[:, 3:4] if has_alpha else None
        rgb = img[:, :3].float()

        luma = (
            LUMA_WEIGHTS[0] * rgb[:, 0:1]
            + LUMA_WEIGHTS[1] * rgb[:, 1:2]
            + LUMA_WEIGHTS[2] * rgb[:, 2:3]
        )

        sx, sy = self._sobel(luma.dtype, luma.device)
        edges_x = F.conv2d(luma, sx, padding=1)
        edges_y = F.conv2d(luma, sy, padding=1)
        edges = torch.sqrt(edges_x.square() + edges_y.square())
        edge_max = edges.amax(dim=(1, 2, 3), keepdim=True).clamp_min(EDGE_EPSILON)
        edges = edges / edge_max

        threshold = float(values["threshold"])
        edge_mask = torch.clamp((edges - threshold) / EDGE_FADE_WIDTH, 0.0, 1.0)
        edge_mask = torch.clamp(edge_mask * float(values["strength"]), 0.0, 1.0)

        blur = self._blur(rgb, int(values["radius"]))
        detail_delta = (rgb - blur).abs().mean(dim=1, keepdim=True)
        detail_keep = torch.clamp((detail_delta - 0.025) / 0.18, 0.0, 1.0) * float(values["protect"])
        edge_mask = edge_mask * (1.0 - detail_keep * 0.75)

        aa_rgb = rgb * (1.0 - edge_mask) + blur * edge_mask

        chroma = float(values["chroma"])
        if chroma > 0.0:
            luma_blur = self._blur(luma, 1)
            chroma_blur_rgb = self._blur(rgb, 1)
            chroma_rgb = luma + (rgb - luma) * (1.0 - edge_mask * chroma) + (chroma_blur_rgb - luma_blur) * edge_mask * chroma
            aa_rgb = aa_rgb * (1.0 - edge_mask * chroma) + chroma_rgb * edge_mask * chroma

        result = torch.clamp(aa_rgb, 0.0, 1.0)
        if has_alpha and alpha is not None:
            result = torch.cat([result.to(alpha.dtype), alpha], dim=1)
        return result.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def _temporal_pass(current: torch.Tensor, previous: torch.Tensor | None, amount: float) -> Tuple[torch.Tensor, torch.Tensor | None]:
        if amount <= 0.0 or current.shape[0] <= 0:
            return current, current[-1:].detach() if current.shape[0] else previous

        out = current.clone()
        prev = previous
        for i in range(current.shape[0]):
            frame = current[i:i + 1]
            if prev is not None:
                motion = (frame[..., :3] - prev[..., :3]).abs().mean(dim=3, keepdim=True)
                stable = torch.clamp(1.0 - motion * 10.0, 0.0, 1.0)
                blend = stable * float(amount)
                out[i:i + 1, ..., :3] = frame[..., :3] * (1.0 - blend) + prev[..., :3] * blend
            prev = out[i:i + 1].detach()
        return out, prev

    @torch.inference_mode()
    def apply(
        self,
        images,
        preset,
        hardware_mode,
        aa_strength,
        edge_threshold,
        blur_radius,
        temporal_stability,
        detail_protection,
        chroma_cleanup,
        max_frames_per_chunk,
    ):
        if images is None:
            raise ValueError("Missing images input.")
        if len(images.shape) != 4:
            raise ValueError("IAMCCS Cine-PP AntiAlias expects IMAGE tensor shape [frames, height, width, channels].")

        original_device = images.device
        original_dtype = images.dtype
        work = images
        if str(hardware_mode or "") == "cpu_safe" and images.device.type != "cpu":
            work = images.cpu()

        selected, values = self._effective_preset(
            work,
            preset,
            hardware_mode,
            aa_strength,
            edge_threshold,
            blur_radius,
            temporal_stability,
            detail_protection,
            chroma_cleanup,
        )

        if float(values["strength"]) <= 0.0 or int(values["radius"]) <= 0:
            return images, _report({
                "node": "IAMCCS_CinePPAntiAlias",
                "bypassed": True,
                "reason": "aa_strength <= 0 or blur_radius <= 0",
                "output_1to1": True,
            })

        chunk_size = self._auto_chunk(work, int(values["radius"]), str(hardware_mode), int(max_frames_per_chunk))
        pbar = ProgressBar(int(work.shape[0])) if ProgressBar is not None else None

        output = torch.empty_like(work)
        previous = None
        for start in range(0, int(work.shape[0]), chunk_size):
            if mm is not None and hasattr(mm, "throw_exception_if_processing_interrupted"):
                mm.throw_exception_if_processing_interrupted()
            end = min(int(work.shape[0]), start + chunk_size)
            chunk = work[start:end]
            processed = self._apply_chunk(chunk, values)
            processed, previous = self._temporal_pass(processed, previous, float(values["temporal"]))
            output[start:end].copy_(processed.to(device=output.device, dtype=output.dtype))
            if pbar is not None:
                pbar.update(end - start)
            if mm is not None and hasattr(mm, "soft_empty_cache"):
                try:
                    mm.soft_empty_cache()
                except Exception:
                    pass

        if output.device != original_device or output.dtype != original_dtype:
            output = output.to(device=original_device, dtype=original_dtype)

        return output.clamp(0.0, 1.0), _report({
            "node": "IAMCCS_CinePPAntiAlias",
            "selected_preset": selected,
            "hardware_mode": str(hardware_mode),
            "frames": int(images.shape[0]),
            "height": int(images.shape[1]),
            "width": int(images.shape[2]),
            "chunk_size": int(chunk_size),
            "aa_strength": float(values["strength"]),
            "edge_threshold": float(values["threshold"]),
            "blur_radius": int(values["radius"]),
            "temporal_stability": float(values["temporal"]),
            "detail_protection": float(values["protect"]),
            "chroma_cleanup": float(values["chroma"]),
            "output_1to1": True,
        })


NODE_CLASS_MAPPINGS = {
    "IAMCCS_CinePPAntiAlias": IAMCCS_CinePPAntiAlias,
    "IAMCCS_CineUTAntiAlias": IAMCCS_CinePPAntiAlias,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_CinePPAntiAlias": "IAMCCS Cine-PP AntiAlias 1:1",
    "IAMCCS_CineUTAntiAlias": "IAMCCS Cine-PP AntiAlias 1:1 (legacy)",
}
