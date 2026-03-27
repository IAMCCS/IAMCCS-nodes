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

from .iamccs_flexible_inputs import FlexibleOptionalInputType, any_type


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


class IAMCCS_SegmentPlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "song_duration_s": ("FLOAT", {"default": 180.0, "min": 0.01, "max": 36000.0, "step": 0.01}),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.001, "max": 240.0, "step": 0.01}),
                "segment_duration_s": ("FLOAT", {"default": 10.0, "min": 0.01, "max": 3600.0, "step": 0.01}),
                "planning_mode": (["manual_segment_seconds", "auto_profile"], {"default": "manual_segment_seconds"}),
                "content_profile": (["videoclip", "monologue"], {"default": "videoclip"}),
                "overlap_frames": ("INT", {"default": 9, "min": 0, "max": 4096, "step": 1}),
                "ltx_round_mode": (["up", "nearest", "down"], {"default": "up"}),
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
            }
        }

    RETURN_TYPES = (
        "INT", "INT", "INT", "INT", "INT", "INT", "INT", "STRING",
        "INT", "INT", "INT", "INT", "INT", "INT", "FLOAT", "FLOAT", "STRING", "FLOAT",
        "INT", "FLOAT", "STRING", "STRING", "STRING",
    )
    RETURN_NAMES = (
        "total_frames",
        "unique_segment_frames",
        "first_segment_raw_frames",
        "continuation_raw_frames",
        "estimated_segments",
        "continuation_loops",
        "last_segment_unique_frames",
        "report",
        "segment_index_out",
        "current_segment_raw_frames",
        "current_segment_unique_frames",
        "current_segment_start_frames",
        "current_segment_end_frames",
        "current_remaining_frames_after",
        "current_segment_start_s",
        "current_segment_end_s",
        "current_segment_report",
        "fps_out",
        "recommended_overlap_frames",
        "recommended_audio_left_context_s",
        "recommended_extension_preset",
        "effective_planning_mode",
        "planning_profile_report",
    )
    FUNCTION = "plan"
    CATEGORY = "IAMCCS/LTX-2"

    def _fix_ltx_frames(self, frames: int, mode: str) -> int:
        frames = max(1, int(frames))
        rem = (frames - 1) % 8
        if rem == 0:
            return frames

        down = max(1, frames - rem)
        up = frames + (8 - rem)

        if mode == "down":
            return down
        if mode == "nearest":
            return up if (up - frames) <= (frames - down) else down
        return up

    def _recommend_profile(self, content_profile: str):
        profile = str(content_profile or "videoclip")
        if profile == "monologue":
            return {
                "base_target_s": 15.0,
                "min_segment_s": 8.0,
                "max_segment_s": 15.0,
                "overlap_frames": 13,
                "audio_left_context_s": 0.75,
                "extension_preset": "monologue_audio_24fps",
            }
        return {
            "base_target_s": 10.0,
            "min_segment_s": 5.0,
            "max_segment_s": 10.0,
            "overlap_frames": 9,
            "audio_left_context_s": 0.5,
            "extension_preset": "videoclip_audio_24fps",
        }

    def plan(self, song_duration_s: float, fps: float, segment_duration_s: float, planning_mode: str, content_profile: str, overlap_frames: int, ltx_round_mode: str, segment_index: int):
        song_duration_s = max(0.01, float(song_duration_s))
        fps = max(0.001, float(fps))
        segment_duration_s = max(0.01, float(segment_duration_s))
        overlap_frames = max(0, int(overlap_frames))
        segment_index = max(0, int(segment_index))
        planning_mode = str(planning_mode or "manual_segment_seconds")
        content_profile = str(content_profile or "videoclip")

        rec = self._recommend_profile(content_profile)
        effective_planning_mode = planning_mode
        if planning_mode == "auto_profile":
            target_s = float(rec["base_target_s"])
            estimated_segments_auto = max(1, int(math.ceil(song_duration_s / max(0.01, target_s))))
            auto_segment_s = song_duration_s / float(estimated_segments_auto)
            auto_segment_s = max(float(rec["min_segment_s"]), min(float(rec["max_segment_s"]), auto_segment_s))
            segment_duration_s = max(0.01, float(auto_segment_s))
            overlap_frames = int(rec["overlap_frames"])

        total_frames = max(1, int(round(song_duration_s * fps)))
        unique_segment_frames = max(1, int(round(segment_duration_s * fps)))

        first_segment_raw_frames = self._fix_ltx_frames(unique_segment_frames, str(ltx_round_mode))
        continuation_raw_frames = self._fix_ltx_frames(unique_segment_frames + overlap_frames, str(ltx_round_mode))

        estimated_segments = max(1, int(math.ceil(float(total_frames) / float(unique_segment_frames))))
        continuation_loops = max(0, estimated_segments - 1)
        remainder = total_frames - unique_segment_frames * max(0, estimated_segments - 1)
        last_segment_unique_frames = unique_segment_frames if remainder <= 0 else int(remainder)

        clamped_segment_index = min(segment_index, max(0, estimated_segments - 1))
        current_segment_start_frames = unique_segment_frames * clamped_segment_index
        current_segment_unique_frames = max(0, min(unique_segment_frames, total_frames - current_segment_start_frames))
        if current_segment_unique_frames <= 0:
            current_segment_unique_frames = last_segment_unique_frames if clamped_segment_index == max(0, estimated_segments - 1) else unique_segment_frames
            current_segment_start_frames = min(current_segment_start_frames, max(0, total_frames - current_segment_unique_frames))
        current_segment_end_frames = min(total_frames, current_segment_start_frames + current_segment_unique_frames)
        current_remaining_frames_after = max(0, total_frames - current_segment_end_frames)
        current_segment_raw_frames = first_segment_raw_frames if clamped_segment_index == 0 else continuation_raw_frames
        current_segment_start_s = float(current_segment_start_frames) / float(fps)
        current_segment_end_s = float(current_segment_end_frames) / float(fps)
        recommended_overlap_frames = int(rec["overlap_frames"])
        recommended_audio_left_context_s = float(rec["audio_left_context_s"])
        recommended_extension_preset = str(rec["extension_preset"])

        report = (
            f"song={song_duration_s:.3f}s @ {fps:.3f}fps -> total={total_frames}f | "
            f"segment={segment_duration_s:.3f}s -> unique={unique_segment_frames}f | "
            f"overlap={overlap_frames}f | first_raw={first_segment_raw_frames}f | "
            f"continuation_raw={continuation_raw_frames}f | segments={estimated_segments} | "
            f"loops={continuation_loops} | last_unique={last_segment_unique_frames}f | ltx_round={ltx_round_mode} | "
            f"planning={effective_planning_mode} | profile={content_profile}"
        )

        current_segment_report = (
            f"segment_index={clamped_segment_index} | raw={current_segment_raw_frames}f | unique={current_segment_unique_frames}f | "
            f"range=[{current_segment_start_frames}..{current_segment_end_frames}) | remaining_after={current_remaining_frames_after}f | "
            f"time=[{current_segment_start_s:.3f}s..{current_segment_end_s:.3f}s]"
        )

        planning_profile_report = (
            f"profile={content_profile} | planning_mode={effective_planning_mode} | segment_duration_s={segment_duration_s:.3f} | "
            f"recommended_overlap={recommended_overlap_frames}f | recommended_left_context={recommended_audio_left_context_s:.2f}s | "
            f"recommended_extension_preset={recommended_extension_preset}"
        )

        return (
            int(total_frames),
            int(unique_segment_frames),
            int(first_segment_raw_frames),
            int(continuation_raw_frames),
            int(estimated_segments),
            int(continuation_loops),
            int(last_segment_unique_frames),
            report,
            int(clamped_segment_index),
            int(current_segment_raw_frames),
            int(current_segment_unique_frames),
            int(current_segment_start_frames),
            int(current_segment_end_frames),
            int(current_remaining_frames_after),
            float(current_segment_start_s),
            float(current_segment_end_s),
            current_segment_report,
            float(fps),
            int(recommended_overlap_frames),
            float(recommended_audio_left_context_s),
            recommended_extension_preset,
            effective_planning_mode,
            planning_profile_report,
        )


class IAMCCS_SegmentPlanFromPlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fps": ("FLOAT", {"default": 24.0, "min": 0.001, "max": 240.0, "step": 0.01}),
                "total_frames": ("INT", {"default": 4320, "min": 1, "max": 10000000, "step": 1}),
                "unique_segment_frames": ("INT", {"default": 240, "min": 1, "max": 1000000, "step": 1}),
                "first_segment_raw_frames": ("INT", {"default": 241, "min": 1, "max": 1000000, "step": 1}),
                "continuation_raw_frames": ("INT", {"default": 249, "min": 1, "max": 1000000, "step": 1}),
                "estimated_segments": ("INT", {"default": 18, "min": 1, "max": 100000, "step": 1}),
                "last_segment_unique_frames": ("INT", {"default": 240, "min": 1, "max": 1000000, "step": 1}),
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "segment_index_out",
        "current_segment_raw_frames",
        "current_segment_unique_frames",
        "current_segment_start_frames",
        "current_segment_end_frames",
        "current_remaining_frames_after",
        "current_segment_start_s",
        "current_segment_end_s",
        "current_segment_report",
    )
    FUNCTION = "derive"
    CATEGORY = "IAMCCS/LTX-2"

    def derive(
        self,
        fps: float,
        total_frames: int,
        unique_segment_frames: int,
        first_segment_raw_frames: int,
        continuation_raw_frames: int,
        estimated_segments: int,
        last_segment_unique_frames: int,
        segment_index: int,
    ):
        fps = max(0.001, float(fps))
        total_frames = max(1, int(total_frames))
        unique_segment_frames = max(1, int(unique_segment_frames))
        first_segment_raw_frames = max(1, int(first_segment_raw_frames))
        continuation_raw_frames = max(1, int(continuation_raw_frames))
        estimated_segments = max(1, int(estimated_segments))
        last_segment_unique_frames = max(1, int(last_segment_unique_frames))
        segment_index = max(0, min(int(segment_index), estimated_segments - 1))

        current_segment_start_frames = unique_segment_frames * segment_index
        if segment_index >= estimated_segments - 1:
            current_segment_unique_frames = last_segment_unique_frames
        else:
            current_segment_unique_frames = unique_segment_frames

        current_segment_end_frames = min(total_frames, current_segment_start_frames + current_segment_unique_frames)
        current_remaining_frames_after = max(0, total_frames - current_segment_end_frames)
        current_segment_raw_frames = first_segment_raw_frames if segment_index == 0 else continuation_raw_frames
        current_segment_start_s = float(current_segment_start_frames) / float(fps)
        current_segment_end_s = float(current_segment_end_frames) / float(fps)
        current_segment_report = (
            f"segment_index={segment_index} | raw={current_segment_raw_frames}f | unique={current_segment_unique_frames}f | "
            f"range=[{current_segment_start_frames}..{current_segment_end_frames}) | remaining_after={current_remaining_frames_after}f | "
            f"time=[{current_segment_start_s:.3f}s..{current_segment_end_s:.3f}s]"
        )

        return (
            int(segment_index),
            int(current_segment_raw_frames),
            int(current_segment_unique_frames),
            int(current_segment_start_frames),
            int(current_segment_end_frames),
            int(current_remaining_frames_after),
            float(current_segment_start_s),
            float(current_segment_end_s),
            current_segment_report,
        )


class IAMCCS_SourceRangeFromSegmentPlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "current_segment_raw_frames": ("INT", {"default": 241, "min": 1, "max": 1000000, "step": 1}),
                "current_segment_unique_frames": ("INT", {"default": 240, "min": 1, "max": 1000000, "step": 1}),
                "current_segment_start_frames": ("INT", {"default": 0, "min": 0, "max": 10000000, "step": 1}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("range_start_index", "range_end_index", "range_count", "report")
    FUNCTION = "derive"
    CATEGORY = "IAMCCS/LTX-2"

    def derive(
        self,
        segment_index: int,
        current_segment_raw_frames: int,
        current_segment_unique_frames: int,
        current_segment_start_frames: int,
    ):
        segment_index = max(0, int(segment_index))
        current_segment_raw_frames = max(1, int(current_segment_raw_frames))
        current_segment_unique_frames = max(1, int(current_segment_unique_frames))
        current_segment_start_frames = max(0, int(current_segment_start_frames))

        overlap_delta = max(0, current_segment_raw_frames - current_segment_unique_frames)
        overlap_backtrack = max(0, overlap_delta - 1)
        if segment_index == 0:
            overlap_backtrack = 0

        range_start_index = max(0, current_segment_start_frames - overlap_backtrack)
        range_end_index = range_start_index + current_segment_raw_frames
        range_count = max(1, range_end_index - range_start_index)

        report = (
            f"segment_index={segment_index} | raw={current_segment_raw_frames}f | unique={current_segment_unique_frames}f | "
            f"start_unique={current_segment_start_frames}f | backtrack={overlap_backtrack}f | "
            f"range=[{range_start_index}..{range_end_index}) count={range_count}"
        )
        return (int(range_start_index), int(range_end_index), int(range_count), report)


class IAMCCS_SegmentSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "default_value": (any_type,),
                "mode": (["single_prompt", "by_segment"], {"default": "single_prompt"}),
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
            },
            "optional": FlexibleOptionalInputType(
                any_type,
                data={
                    "segment_01": (any_type,),
                    "segment_02": (any_type,),
                    "segment_03": (any_type,),
                    "segment_04": (any_type,),
                    "segment_05": (any_type,),
                    "segment_06": (any_type,),
                },
            ),
        }

    RETURN_TYPES = (any_type, "INT", "STRING")
    RETURN_NAMES = ("value", "active_slot", "report")
    FUNCTION = "select"
    CATEGORY = "IAMCCS/LTX-2"

    def select(self, default_value, mode: str, segment_index: int, **kwargs):
        if str(mode) == "single_prompt":
            return (default_value, 0, "mode=single_prompt -> default_value")

        slot = max(1, int(segment_index) + 1)
        key = f"segment_{slot:02d}"
        value = kwargs.get(key, None)
        if value is None:
            return (default_value, 0, f"mode=by_segment | segment_index={int(segment_index)} -> fallback=default_value")

        return (value, slot, f"mode=by_segment | segment_index={int(segment_index)} -> {key}")


class IAMCCS_TwoSegmentPlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total_frames": ("INT", {"default": 481, "min": 1, "max": 10000000, "step": 1}),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.001, "max": 240.0, "step": 0.01}),
                "first_segment_raw_frames": ("INT", {"default": 241, "min": 1, "max": 1000000, "step": 1}),
                "second_segment_raw_frames": ("INT", {"default": 249, "min": 1, "max": 1000000, "step": 1}),
            }
        }

    RETURN_TYPES = (
        "INT", "INT", "INT",
        "INT", "INT", "INT",
        "INT", "FLOAT", "STRING",
    )
    RETURN_NAMES = (
        "seg0_start_frames",
        "seg0_end_frames",
        "seg0_count_frames",
        "seg1_start_frames",
        "seg1_end_frames",
        "seg1_count_frames",
        "overlap_frames",
        "total_duration_s",
        "report",
    )
    FUNCTION = "plan"
    CATEGORY = "IAMCCS/LTX-2"

    def plan(self, total_frames: int, fps: float, first_segment_raw_frames: int, second_segment_raw_frames: int):
        total_frames = max(1, int(total_frames))
        fps = max(0.001, float(fps))
        first_segment_raw_frames = max(1, int(first_segment_raw_frames))
        second_segment_raw_frames = max(1, int(second_segment_raw_frames))

        seg0_start_frames = 0
        seg0_end_frames = min(total_frames, first_segment_raw_frames)
        seg0_count_frames = max(1, seg0_end_frames - seg0_start_frames)

        seg1_count_frames = min(total_frames, second_segment_raw_frames)
        seg1_end_frames = total_frames
        seg1_start_frames = max(0, seg1_end_frames - seg1_count_frames)

        overlap_frames = max(0, seg0_count_frames + seg1_count_frames - total_frames)
        total_duration_s = float(total_frames) / float(fps)

        report = (
            f"total={total_frames}f @ {fps:.3f}fps ({total_duration_s:.3f}s) | "
            f"seg0=[{seg0_start_frames}..{seg0_end_frames}) count={seg0_count_frames} | "
            f"seg1=[{seg1_start_frames}..{seg1_end_frames}) count={seg1_count_frames} | "
            f"overlap={overlap_frames}f"
        )
        return (
            int(seg0_start_frames),
            int(seg0_end_frames),
            int(seg0_count_frames),
            int(seg1_start_frames),
            int(seg1_end_frames),
            int(seg1_count_frames),
            int(overlap_frames),
            float(total_duration_s),
            report,
        )


class IAMCCS_ThreeSegmentPlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total_frames": ("INT", {"default": 721, "min": 1, "max": 10000000, "step": 1}),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.001, "max": 240.0, "step": 0.01}),
                "first_segment_raw_frames": ("INT", {"default": 241, "min": 1, "max": 1000000, "step": 1}),
                "continuation_raw_frames": ("INT", {"default": 249, "min": 1, "max": 1000000, "step": 1}),
            }
        }

    RETURN_TYPES = (
        "INT", "INT", "INT",
        "INT", "INT", "INT",
        "INT", "INT", "INT",
        "INT", "INT", "FLOAT", "STRING",
    )
    RETURN_NAMES = (
        "seg0_start_frames",
        "seg0_end_frames",
        "seg0_count_frames",
        "seg1_start_frames",
        "seg1_end_frames",
        "seg1_count_frames",
        "seg2_start_frames",
        "seg2_end_frames",
        "seg2_count_frames",
        "overlap_frames",
        "unique_segment_frames",
        "total_duration_s",
        "report",
    )
    FUNCTION = "plan"
    CATEGORY = "IAMCCS/LTX-2"

    def plan(self, total_frames: int, fps: float, first_segment_raw_frames: int, continuation_raw_frames: int):
        total_frames = max(1, int(total_frames))
        fps = max(0.001, float(fps))
        first_segment_raw_frames = max(1, int(first_segment_raw_frames))
        continuation_raw_frames = max(1, int(continuation_raw_frames))

        unique_segment_frames = max(1, first_segment_raw_frames - 1)
        overlap_frames = max(0, continuation_raw_frames - unique_segment_frames)
        overlap_backtrack = max(0, overlap_frames - 1)

        def segment_range(index: int):
            if index == 0:
                start_frames = 0
                count_frames = min(total_frames, first_segment_raw_frames)
            else:
                unique_start_frames = unique_segment_frames * index
                start_frames = max(0, unique_start_frames - overlap_backtrack)
                count_frames = max(1, min(continuation_raw_frames, total_frames - start_frames))
            end_frames = min(total_frames, start_frames + count_frames)
            count_frames = max(1, end_frames - start_frames)
            return int(start_frames), int(end_frames), int(count_frames)

        seg0_start_frames, seg0_end_frames, seg0_count_frames = segment_range(0)
        seg1_start_frames, seg1_end_frames, seg1_count_frames = segment_range(1)
        seg2_start_frames, seg2_end_frames, seg2_count_frames = segment_range(2)
        total_duration_s = float(total_frames) / float(fps)

        report = (
            f"total={total_frames}f @ {fps:.3f}fps ({total_duration_s:.3f}s) | "
            f"unique={unique_segment_frames}f | overlap={overlap_frames}f | "
            f"seg0=[{seg0_start_frames}..{seg0_end_frames}) count={seg0_count_frames} | "
            f"seg1=[{seg1_start_frames}..{seg1_end_frames}) count={seg1_count_frames} | "
            f"seg2=[{seg2_start_frames}..{seg2_end_frames}) count={seg2_count_frames}"
        )
        return (
            int(seg0_start_frames),
            int(seg0_end_frames),
            int(seg0_count_frames),
            int(seg1_start_frames),
            int(seg1_end_frames),
            int(seg1_count_frames),
            int(seg2_start_frames),
            int(seg2_end_frames),
            int(seg2_count_frames),
            int(overlap_frames),
            int(unique_segment_frames),
            float(total_duration_s),
            report,
        )


NODE_CLASS_MAPPINGS = {
    "IAMCCS_LTX2_FrameRateSync": IAMCCS_LTX2_FrameRateSync,
    "IAMCCS_LTX2_Validator": IAMCCS_LTX2_Validator,
    "IAMCCS_LTX2_TimeFrameCount": IAMCCS_LTX2_TimeFrameCount,
    "IAMCCS_LTX2_EnsureFrames8nPlus1": IAMCCS_LTX2_EnsureFrames8nPlus1,
    "IAMCCS_LTX2_ControlPreprocess": IAMCCS_LTX2_ControlPreprocess,
    "IAMCCS_SegmentPlanner": IAMCCS_SegmentPlanner,
    "IAMCCS_SegmentPlanFromPlanner": IAMCCS_SegmentPlanFromPlanner,
    "IAMCCS_SourceRangeFromSegmentPlan": IAMCCS_SourceRangeFromSegmentPlan,
    "IAMCCS_TwoSegmentPlanner": IAMCCS_TwoSegmentPlanner,
    "IAMCCS_ThreeSegmentPlanner": IAMCCS_ThreeSegmentPlanner,
    "IAMCCS_SegmentSwitch": IAMCCS_SegmentSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_LTX2_FrameRateSync": "LTX-2 FrameRate Sync (int+float)",
    "IAMCCS_LTX2_Validator": "LTX-2 Validator (32px, 8n +1)",
    "IAMCCS_LTX2_TimeFrameCount": "LTX-2 TimeFrameCount",
    "IAMCCS_LTX2_EnsureFrames8nPlus1": "LTX-2 Ensure Frames (8n + 1)",
    "IAMCCS_LTX2_ControlPreprocess": "LTX-2 Control Preprocess (aux)",
    "IAMCCS_SegmentPlanner": "Segment Planner (song -> LTX frames)",
    "IAMCCS_SegmentPlanFromPlanner": "Segment Plan From Planner (per index)",
    "IAMCCS_SourceRangeFromSegmentPlan": "Source Range From Segment Plan",
    "IAMCCS_TwoSegmentPlanner": "Two Segment Planner (stable 2SEG)",
    "IAMCCS_ThreeSegmentPlanner": "Three Segment Planner (stable 3SEG)",
    "IAMCCS_SegmentSwitch": "Segment Switch (by segment_index)",
}
