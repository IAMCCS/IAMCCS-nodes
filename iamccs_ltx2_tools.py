# iamccs_ltx2_tools.py
# ===============================================================
# IAMCCS LTX-2 helpers
# - FrameRate sync (INT + FLOAT)
# - Validator / autofix for LTX-2 constraints
# - Simple control-image preprocess helper (aux) for IC-LoRA style workflows
# ===============================================================

from __future__ import annotations

import logging
import math
from typing import Tuple

import torch


_F = torch.nn.functional


_log = logging.getLogger("IAMCCS.LTX2.Tools")


def _to_grayscale(image: torch.Tensor) -> torch.Tensor:
    # image: [B,H,W,C] float in [0,1]
    if image.shape[-1] == 1:
        return image
    r = image[..., 0:1]
    g = image[..., 1:2]
    b = image[..., 2:3]
    return r * 0.299 + g * 0.587 + b * 0.114


def _sobel_edges(gray: torch.Tensor) -> torch.Tensor:
    # gray: [B,H,W,1]
    x = gray.permute(0, 3, 1, 2)  # [B,1,H,W]
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gx = torch.nn.functional.conv2d(x, kx, padding=1)
    gy = torch.nn.functional.conv2d(x, ky, padding=1)
    mag = torch.sqrt(torch.clamp(gx * gx + gy * gy, min=0.0))
    mag = mag / (mag.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6))
    return mag.permute(0, 2, 3, 1)  # [B,H,W,1]


class IAMCCS_LTX2_FrameRateSync:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}),
                "int_mode": (["round", "floor", "ceil", "fixed"], {"default": "round"}),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT", "STRING")
    RETURN_NAMES = ("fps_int", "fps_float", "report")
    FUNCTION = "sync"
    CATEGORY = "IAMCCS/LTX-2"

    def sync(self, fps: float, int_mode: str):
        fps_in = float(fps)

        if int_mode == "floor":
            fps_int = int(math.floor(fps_in))
            fps_float = fps_in
        elif int_mode == "ceil":
            fps_int = int(math.ceil(fps_in))
            fps_float = fps_in
        elif int_mode == "fixed":
            # "fixed" = force perfect INT/FLOAT agreement by snapping the float output
            # to the derived integer (useful when downstream nodes require exact match).
            fps_int = int(round(fps_in))
            fps_float = float(fps_int)
        else:
            fps_int = int(round(fps_in))
            fps_float = fps_in

        fps_int = max(1, fps_int)

        delta = fps_in - float(fps_int)
        report = f"fps_in={fps_in:.3f} -> fps_int={fps_int}, fps_float={fps_float:.3f} (mode={int_mode}, delta={delta:+.3f})"
        if int_mode == "fixed" and abs(delta) > 1e-6:
            _log.warning("[IAMCCS_LTX2_FrameRateSync] fixed mode snapped float to int. %s", report)

        return (fps_int, fps_float, report)


class IAMCCS_LTX2_Validator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Wraps EmptyImage-like behavior
                "width": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 16}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1}),
                "color": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                # Duration widgets are both visible for UX.
                # The frontend JS keeps seconds <-> length in sync using IAMCCS_LTX2_FrameRateSync.
                "seconds": ("FLOAT", {"default": 4.0, "min": 0.01, "max": 3600.0, "step": 0.01}),
                "length": ("INT", {"default": 81, "min": 1, "max": 16385, "step": 1, "control_after_generate": True}),
                # Validation controls
                "autofix": ("BOOLEAN", {"default": True}),
                "length_fix": (["up", "down", "nearest"], {"default": "up"}),
            },
            # Optional pass-through: if provided, we will make the IMAGE batch frame-count
            # match the validated length (8n+1) by padding/cropping. This is the workflow-safe
            # way to guarantee the VAE encode constraint without adding extra nodes.
            "optional": {
                "images": ("IMAGE", {}),
                "images_mode": (["pad_repeat_last", "crop_end"], {"default": "pad_repeat_last"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("image", "length", "report")
    FUNCTION = "validate"
    CATEGORY = "IAMCCS/LTX-2"

    def _fix_multiple(self, value: int, multiple: int) -> Tuple[int, int]:
        value = int(value)
        if multiple <= 0:
            return value, 0
        rem = value % multiple
        if rem == 0:
            return value, 0
        fixed = value + (multiple - rem)
        return fixed, fixed - value

    def _frames_rule_fix(self, frames: int, mode: str) -> Tuple[int, int]:
        # LTX-2 guideline: frame count should be 8n + 1.
        frames = int(frames)
        if frames < 1:
            frames = 1

        rem = (frames - 1) % 8
        if rem == 0:
            return frames, 0

        down = frames - rem
        up = frames + (8 - rem)

        if mode == "down":
            fixed = max(1, down)
        elif mode == "nearest":
            fixed = up if (up - frames) <= (frames - down) else max(1, down)
        else:
            fixed = up

        return fixed, fixed - frames

    def validate(
        self,
        width: int,
        height: int,
        batch_size: int,
        color: int,
        seconds: float,
        length: int,
        autofix: bool,
        length_fix: str,
        images: torch.Tensor | None = None,
        images_mode: str = "pad_repeat_last",
    ):
        width_in = int(width)
        height_in = int(height)
        batch_in = int(batch_size)
        color_in = int(color)
        seconds_in = float(seconds)
        length_in = int(length)

        # Spatial rule:
        # - We keep this relatively permissive (16px multiple) because many valid LTX-2
        #   workflows use resolutions like 1280x720 (720 is not divisible by 32).
        # - If a downstream node truly requires 32, it should validate there.
        spatial_multiple = 16
        w_ok = (width_in % spatial_multiple) == 0
        h_ok = (height_in % spatial_multiple) == 0
        l_ok = ((length_in - 1) % 8) == 0

        width_fixed, pad_w = self._fix_multiple(width_in, spatial_multiple)
        height_fixed, pad_h = self._fix_multiple(height_in, spatial_multiple)
        length_fixed, pad_len = self._frames_rule_fix(length_in, length_fix)

        if not autofix:
            width_fixed, height_fixed, length_fixed = width_in, height_in, length_in
            pad_w, pad_h, pad_len = 0, 0, 0

        batch_fixed = max(1, batch_in)
        color_fixed = max(0, min(255, color_in))

        def _pad_repeat_last_frames(x: torch.Tensor, pad: int) -> torch.Tensor:
            if pad <= 0:
                return x
            last = x[-1:, ...].repeat(int(pad), 1, 1, 1)
            return torch.cat([x, last], dim=0)

        # If an IMAGE batch is provided, enforce that its frames match the validated length.
        # This is the actual guarantee needed by LTX VAE encode.
        if images is not None:
            img_in = images
            if not torch.is_floating_point(img_in):
                img_in = img_in.float() / 255.0
            # Expect ComfyUI IMAGE: [frames,H,W,C]
            if img_in.ndim != 4:
                raise ValueError("images must be a ComfyUI IMAGE tensor with shape [frames,H,W,C]")

            frames_in = int(img_in.shape[0])
            frames_out = int(length_fixed)
            if not autofix:
                frames_out = frames_in

            mode = str(images_mode or "pad_repeat_last")
            if mode not in ("pad_repeat_last", "crop_end"):
                mode = "pad_repeat_last"

            if frames_out == frames_in:
                img = img_in
            elif frames_out < frames_in:
                # Crop end to match target length
                img = img_in[:frames_out, ...]
            else:
                # Pad by repeating last frame
                img = _pad_repeat_last_frames(img_in, frames_out - frames_in)

            # Note: we do NOT resize spatially here; this node's width/height validation is
            # meant for parameter sanity and EmptyImage-like generation. Spatial resizing remains
            # the responsibility of the workflow.
        else:
            # EmptyImage-compatible IMAGE tensor: [B,H,W,3] float in [0,1]
            fill = float(color_fixed) / 255.0
            img = torch.full(
                (batch_fixed, int(height_fixed), int(width_fixed), 3),
                fill,
                dtype=torch.float32,
                device="cpu",
            )

        modified = (width_fixed != width_in) or (height_fixed != height_in) or (length_fixed != length_in) or (batch_fixed != batch_in) or (color_fixed != color_in)
        if images is not None and isinstance(images, torch.Tensor) and images.ndim == 4:
            modified = modified or (int(images.shape[0]) != int(img.shape[0]))
        implied_fps_str = "n/a"
        if seconds_in > 0:
            implied_fps = (float(length_fixed) - 1.0) / seconds_in
            implied_fps_str = f"{implied_fps:.3f}"
        images_note = ""
        if images is not None and isinstance(images, torch.Tensor) and images.ndim == 4:
            images_note = f" | images_frames: in={int(images.shape[0])} -> out={int(img.shape[0])} (mode={images_mode}, autofix={autofix})"

        report = (
            f"input: {width_in}x{height_in}, batch={batch_in}, color={color_in} | "
            f"seconds_input={seconds_in:.3f}, length_input={length_in} | "
            f"rules: w%{spatial_multiple}={w_ok}, h%{spatial_multiple}={h_ok}, length=8n+1={l_ok} | "
            f"output: {width_fixed}x{height_fixed}, length={length_fixed}, batch={batch_fixed}, color={color_fixed} | "
            f"implied_fps={implied_fps_str} | "
            f"autofix={autofix} (len_fix={length_fix}) | modified={modified} | "
            f"delta: +{pad_w}w, +{pad_h}h, {pad_len:+d} length"
            f"{images_note}"
        )

        if not (w_ok and h_ok and l_ok):
            _log.warning("[IAMCCS_LTX2_Validator] %s", report)

        return (img, int(length_fixed), report)


class IAMCCS_LTX2_TimeFrameCount:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Duration widgets are both visible for UX.
                # The frontend JS keeps seconds <-> length in sync using IAMCCS_LTX2_FrameRateSync.
                "seconds": ("FLOAT", {"default": 4.0, "min": 0.01, "max": 3600.0, "step": 0.01}),
                "length": ("INT", {"default": 81, "min": 1, "max": 16385, "step": 1, "control_after_generate": True}),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT", "STRING")
    RETURN_NAMES = ("length", "seconds", "report")
    FUNCTION = "timeframe"
    CATEGORY = "IAMCCS/LTX-2"

    def timeframe(self, seconds: float, length: int):
        seconds_in = float(seconds)
        length_in = int(length)
        length_fixed = max(1, length_in)
        # LTX-2 encode constraint: frames must be 8n + 1.
        # Round UP to avoid shortening the requested duration.
        pad = 0
        rem = (length_fixed - 1) % 8
        if rem != 0:
            pad = 8 - rem
            length_fixed = length_fixed + pad
        seconds_fixed = max(0.01, seconds_in)

        implied_fps_str = "n/a"
        if seconds_fixed > 0:
            implied_fps = (float(length_fixed) - 1.0) / seconds_fixed
            implied_fps_str = f"{implied_fps:.3f}"

        snap = "ok" if pad == 0 else f"up(+{pad})"
        report = (
            f"seconds_input={seconds_in:.3f}, length_input={length_in} -> "
            f"seconds={seconds_fixed:.3f}, length={length_fixed} | implied_fps={implied_fps_str} | ltx2_8n+1={snap}"
        )

        return (int(length_fixed), float(seconds_fixed), report)


class IAMCCS_LTX2_ImageBatchPadReflect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {}),
                "pad_x": ("INT", {"default": 16, "min": 0, "max": 512, "step": 1}),
                "pad_y": ("INT", {"default": 16, "min": 0, "max": 512, "step": 1}),
                "pad_mode": (["reflect", "replicate"], {"default": "reflect"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("images", "pad_x", "pad_y", "report")
    FUNCTION = "pad"
    CATEGORY = "IAMCCS/LTX-2"

    def pad(self, images: torch.Tensor, pad_x: int, pad_y: int, pad_mode: str):
        # images: [B,H,W,C] float
        if images is None:
            raise ValueError("images is required")

        b, h, w, c = images.shape
        px = max(0, int(pad_x))
        py = max(0, int(pad_y))

        # reflect requires pad < dim; clamp to avoid runtime errors
        px_eff = min(px, max(0, w - 1))
        py_eff = min(py, max(0, h - 1))

        if px_eff == 0 and py_eff == 0:
            return (images, 0, 0, f"PadReflect: no-op (input {w}x{h})")

        mode = str(pad_mode or "reflect")
        if mode not in ("reflect", "replicate"):
            mode = "reflect"

        x = images.permute(0, 3, 1, 2)  # [B,C,H,W]
        # pad format: (left, right, top, bottom)
        x = _F.pad(x, (px_eff, px_eff, py_eff, py_eff), mode=mode)
        out = x.permute(0, 2, 3, 1).contiguous()

        report = (
            f"PadReflect: mode={mode}, requested=({px},{py}), used=({px_eff},{py_eff}) | "
            f"{w}x{h} -> {int(out.shape[2])}x{int(out.shape[1])}"
        )
        return (out, int(px_eff), int(py_eff), report)


class IAMCCS_LTX2_ImageBatchCropByPad:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {}),
                "pad_x": ("INT", {"default": 16, "min": 0, "max": 512, "step": 1}),
                "pad_y": ("INT", {"default": 16, "min": 0, "max": 512, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "crop"
    CATEGORY = "IAMCCS/LTX-2"

    def crop(self, images: torch.Tensor, pad_x: int, pad_y: int):
        if images is None:
            raise ValueError("images is required")

        b, h, w, c = images.shape
        px = max(0, int(pad_x))
        py = max(0, int(pad_y))

        if px == 0 and py == 0:
            return (images, f"CropByPad: no-op (input {w}x{h})")

        # Clamp so we never invert the crop
        px_eff = min(px, max(0, (w - 1) // 2))
        py_eff = min(py, max(0, (h - 1) // 2))

        out = images[:, py_eff : h - py_eff, px_eff : w - px_eff, :]
        report = (
            f"CropByPad: requested=({px},{py}), used=({px_eff},{py_eff}) | "
            f"{w}x{h} -> {int(out.shape[2])}x{int(out.shape[1])}"
        )
        return (out, report)


class IAMCCS_LTX2_EnsureFrames8nPlus1:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {}),
                # LTX-2 encode constraint: frames must be 1 + 8*k.
                # "pad" is workflow-safe (never shortens), "crop" is deterministic.
                "mode": (["pad_repeat_last", "crop_end"], {"default": "pad_repeat_last"}),
                "fix": (["up", "down", "nearest"], {"default": "up"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("images", "frames", "report")
    FUNCTION = "ensure"
    CATEGORY = "IAMCCS/LTX-2"

    def _frames_rule_fix(self, frames: int, mode: str) -> tuple[int, int]:
        frames = int(frames)
        if frames < 1:
            frames = 1

        rem = (frames - 1) % 8
        if rem == 0:
            return frames, 0

        down = frames - rem
        up = frames + (8 - rem)

        if mode == "down":
            fixed = max(1, down)
        elif mode == "nearest":
            fixed = up if (up - frames) <= (frames - down) else max(1, down)
        else:
            fixed = up

        return fixed, fixed - frames

    def ensure(self, images: torch.Tensor, mode: str, fix: str):
        if images is None:
            raise ValueError("images is required")

        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("images must be a ComfyUI IMAGE tensor with shape [frames,H,W,C]")

        frames_in = int(images.shape[0])
        frames_fixed, delta = self._frames_rule_fix(frames_in, str(fix or "up"))

        if frames_fixed == frames_in:
            return (images, frames_in, f"EnsureFrames8n+1: ok ({frames_in})")

        mode = str(mode or "pad_repeat_last")
        if mode == "crop_end":
            # For crop mode, prefer shortening, regardless of requested 'fix'.
            rem = (frames_in - 1) % 8
            frames_fixed = frames_in if rem == 0 else max(1, frames_in - rem)
            out = images[:frames_fixed, ...]
            return (out, frames_fixed, f"EnsureFrames8n+1: crop_end {frames_in} -> {frames_fixed}")

        # pad_repeat_last (workflow-safe)
        if frames_fixed < frames_in:
            # If user selected fix=down/nearest and it resulted in fewer frames,
            # still keep behavior consistent with padding node: do a crop.
            out = images[:frames_fixed, ...]
            return (out, frames_fixed, f"EnsureFrames8n+1: crop_end {frames_in} -> {frames_fixed} (fix={fix})")

        pad = int(frames_fixed - frames_in)
        last = images[-1:, ...].repeat(pad, 1, 1, 1)
        out = torch.cat([images, last], dim=0)
        return (out, frames_fixed, f"EnsureFrames8n+1: pad_repeat_last {frames_in} -> {frames_fixed} (pad={pad}, fix={fix})")


class IAMCCS_LTX2_ControlPreprocess:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": ([
                    "none",
                    "grayscale",
                    "invert",
                    "threshold",
                    "edges_sobel",
                    "edges_canny (opencv_if_available)",
                ], {"default": "edges_sobel"}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "canny_low": ("INT", {"default": 100, "min": 0, "max": 1024, "step": 1}),
                "canny_high": ("INT", {"default": 200, "min": 0, "max": 2048, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "process"
    CATEGORY = "IAMCCS/LTX-2"

    def process(self, image: torch.Tensor, mode: str, threshold: float, canny_low: int, canny_high: int):
        if mode == "none":
            return (image, "mode=none")

        img = image
        # ComfyUI IMAGE is typically float in [0,1], but in practice some custom nodes
        # provide float in [0,255] or even [-1,1]. If we only clamp, we can destroy
        # the signal (all-white / crushed blacks). Normalize defensively.
        if not torch.is_floating_point(img):
            img = img.float() / 255.0
        else:
            # Heuristic 1: float in [0,255]
            # Using a threshold > 1 to avoid touching regular [0,1] images.
            max_v = float(img.detach().amax().item()) if img.numel() else 0.0
            if max_v > 1.5:
                img = img / 255.0

            # Heuristic 2: float in [-1,1]
            min_v = float(img.detach().amin().item()) if img.numel() else 0.0
            if min_v < -0.01 and max_v <= 1.01:
                img = (img + 1.0) * 0.5

        img = img.clamp(0.0, 1.0)

        if mode == "invert":
            out = 1.0 - img
            return (out, "mode=invert")

        gray = _to_grayscale(img)

        if mode == "grayscale":
            out = gray.repeat(1, 1, 1, 3)
            return (out, "mode=grayscale")

        if mode == "threshold":
            t = float(threshold)
            out1 = (gray > t).to(gray.dtype)
            out = out1.repeat(1, 1, 1, 3)
            return (out, f"mode=threshold t={t:.3f}")

        if mode == "edges_sobel":
            edges = _sobel_edges(gray)
            out = edges.repeat(1, 1, 1, 3)
            return (out, "mode=edges_sobel")

        # canny via opencv if available; fall back to sobel
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore

            # Work on CPU per-frame. Convert to uint8 grayscale.
            gray_cpu = (gray.detach().to("cpu") * 255.0).clamp(0, 255).to(torch.uint8)
            b, h, w, _ = gray_cpu.shape
            out_frames = []
            for i in range(b):
                g = gray_cpu[i, :, :, 0].numpy()
                e = cv2.Canny(g, int(canny_low), int(canny_high))
                out_frames.append(e)
            edges_np = np.stack(out_frames, axis=0).astype(np.float32) / 255.0
            edges = torch.from_numpy(edges_np).to(img.device, dtype=img.dtype).view(-1, h, w, 1)
            out = edges.repeat(1, 1, 1, 3)
            return (out, f"mode=edges_canny low={canny_low} high={canny_high}")
        except Exception:
            edges = _sobel_edges(gray)
            out = edges.repeat(1, 1, 1, 3)
            return (out, "mode=edges_canny fallback=edges_sobel")


NODE_CLASS_MAPPINGS = {
    "IAMCCS_LTX2_FrameRateSync": IAMCCS_LTX2_FrameRateSync,
    "IAMCCS_LTX2_Validator": IAMCCS_LTX2_Validator,
    "IAMCCS_LTX2_TimeFrameCount": IAMCCS_LTX2_TimeFrameCount,
    "IAMCCS_LTX2_EnsureFrames8nPlus1": IAMCCS_LTX2_EnsureFrames8nPlus1,
    "IAMCCS_LTX2_ControlPreprocess": IAMCCS_LTX2_ControlPreprocess,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_LTX2_FrameRateSync": "LTX-2 FrameRate Sync (int+float)",
    "IAMCCS_LTX2_Validator": "LTX-2 Validator (32px, 8n +1)",
    "IAMCCS_LTX2_TimeFrameCount": "LTX-2 TimeFrameCount",
    "IAMCCS_LTX2_EnsureFrames8nPlus1": "LTX-2 Ensure Frames (8n + 1)",
    "IAMCCS_LTX2_ControlPreprocess": "LTX-2 Control Preprocess (aux)",
}
