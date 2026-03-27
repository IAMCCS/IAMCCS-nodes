from __future__ import annotations

import logging
import math

import torch

from nodes import MAX_RESOLUTION

try:
    import comfy.utils  # type: ignore
    from comfy import model_management  # type: ignore
except Exception as exc:  # pragma: no cover - ComfyUI runtime import guard
    raise ImportError("IAMCCS image resize requires ComfyUI runtime modules") from exc


_log = logging.getLogger("IAMCCS.ImageResize")


class IAMCCS_ImageResizeBatchSafe:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    keep_proportion_modes = [
        "stretch",
        "resize",
        "pad",
        "pad_edge",
        "pad_edge_pixel",
        "crop",
        "pillarbox_blur",
        "total_pixels",
    ]
    crop_positions = ["center", "top", "bottom", "left", "right"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "upscale_method": (cls.upscale_methods,),
                "keep_proportion": (cls.keep_proportion_modes, {"default": "crop"}),
                "pad_color": ("STRING", {"default": "0, 0, 0"}),
                "crop_position": (cls.crop_positions, {"default": "center"}),
                "divisible_by": ("INT", {"default": 2, "min": 0, "max": 512, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),
                "device": (["cpu", "gpu"], {"default": "cpu"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "MASK")
    RETURN_NAMES = ("IMAGE", "width", "height", "mask")
    FUNCTION = "resize"
    CATEGORY = "IAMCCS/Utils"
    DESCRIPTION = "Batch-safe IMAGE resize for IAMCCS workflows. Mirrors the KJ resize modes used in IAMCCS flows without the final concat RAM spike."

    @staticmethod
    def _parse_pad_color(value: str, channels: int, dtype: torch.dtype) -> torch.Tensor:
        try:
            parts = [float(part.strip()) for part in str(value).split(",")]
        except Exception:
            parts = [0.0, 0.0, 0.0]
        if not parts:
            parts = [0.0, 0.0, 0.0]
        if len(parts) == 1:
            parts = parts * channels
        elif len(parts) < channels:
            parts = parts + [parts[-1]] * (channels - len(parts))
        parts = parts[:channels]
        if any(part > 1.0 for part in parts):
            parts = [part / 255.0 for part in parts]
        return torch.tensor(parts, dtype=dtype)

    @staticmethod
    def _resolve_device(device_name: str, upscale_method: str) -> torch.device:
        if device_name == "gpu":
            if upscale_method == "lanczos":
                raise ValueError("Lanczos is not supported on GPU in this resize node")
            return model_management.get_torch_device()
        return torch.device("cpu")

    @staticmethod
    def _fit_size(src_w: int, src_h: int, dst_w: int, dst_h: int, mode: str) -> tuple[int, int]:
        if mode == "total_pixels":
            total_pixels = max(1, int(dst_w) * int(dst_h))
            aspect_ratio = float(src_w) / float(max(1, src_h))
            new_h = max(1, int(math.sqrt(total_pixels / max(aspect_ratio, 1e-6))))
            new_w = max(1, int(math.sqrt(total_pixels * max(aspect_ratio, 1e-6))))
            return new_w, new_h
        if dst_w == 0 and dst_h == 0:
            return src_w, src_h
        if dst_w == 0:
            ratio = float(dst_h) / float(max(1, src_h))
            return max(1, round(src_w * ratio)), max(1, int(dst_h))
        if dst_h == 0:
            ratio = float(dst_w) / float(max(1, src_w))
            return max(1, int(dst_w)), max(1, round(src_h * ratio))
        ratio = min(float(dst_w) / float(max(1, src_w)), float(dst_h) / float(max(1, src_h)))
        return max(1, round(src_w * ratio)), max(1, round(src_h * ratio))

    @staticmethod
    def _apply_divisible(size_w: int, size_h: int, divisible_by: int) -> tuple[int, int]:
        if int(divisible_by) <= 1:
            return int(size_w), int(size_h)
        div = int(divisible_by)
        size_w = max(div, int(size_w) - (int(size_w) % div))
        size_h = max(div, int(size_h) - (int(size_h) % div))
        return size_w, size_h

    @staticmethod
    def _crop_box(src_w: int, src_h: int, dst_w: int, dst_h: int, crop_position: str) -> tuple[int, int, int, int]:
        src_aspect = float(src_w) / float(max(1, src_h))
        dst_aspect = float(max(1, dst_w)) / float(max(1, dst_h))
        if src_aspect > dst_aspect:
            crop_w = max(1, round(src_h * dst_aspect))
            crop_h = src_h
        else:
            crop_w = src_w
            crop_h = max(1, round(src_w / max(dst_aspect, 1e-6)))
        if crop_position == "top":
            x = (src_w - crop_w) // 2
            y = 0
        elif crop_position == "bottom":
            x = (src_w - crop_w) // 2
            y = src_h - crop_h
        elif crop_position == "left":
            x = 0
            y = (src_h - crop_h) // 2
        elif crop_position == "right":
            x = src_w - crop_w
            y = (src_h - crop_h) // 2
        else:
            x = (src_w - crop_w) // 2
            y = (src_h - crop_h) // 2
        return x, y, crop_w, crop_h

    @staticmethod
    def _pad_offsets(canvas_w: int, canvas_h: int, inner_w: int, inner_h: int, crop_position: str) -> tuple[int, int]:
        if crop_position == "top":
            return max(0, (canvas_w - inner_w) // 2), 0
        if crop_position == "bottom":
            return max(0, (canvas_w - inner_w) // 2), max(0, canvas_h - inner_h)
        if crop_position == "left":
            return 0, max(0, (canvas_h - inner_h) // 2)
        if crop_position == "right":
            return max(0, canvas_w - inner_w), max(0, (canvas_h - inner_h) // 2)
        return max(0, (canvas_w - inner_w) // 2), max(0, (canvas_h - inner_h) // 2)

    @staticmethod
    def _resize_nhwc(images: torch.Tensor, width: int, height: int, method: str) -> torch.Tensor:
        if int(images.shape[1]) == int(height) and int(images.shape[2]) == int(width):
            return images
        return comfy.utils.common_upscale(
            images.movedim(-1, 1),
            int(width),
            int(height),
            method,
            "disabled",
        ).movedim(1, -1)

    @staticmethod
    def _resize_mask(mask: torch.Tensor, width: int, height: int, method: str) -> torch.Tensor:
        if int(mask.shape[1]) == int(height) and int(mask.shape[2]) == int(width):
            return mask
        if method == "lanczos":
            mask_3 = mask.unsqueeze(1).repeat(1, 3, 1, 1)
            resized = comfy.utils.common_upscale(mask_3, int(width), int(height), method, "disabled")
            return resized[:, 0, :, :]
        resized = comfy.utils.common_upscale(mask.unsqueeze(1), int(width), int(height), method, "disabled")
        return resized[:, 0, :, :]

    def _process_subbatch(
        self,
        image: torch.Tensor,
        mask: torch.Tensor | None,
        width: int,
        height: int,
        keep_proportion: str,
        upscale_method: str,
        crop_position: str,
        pad_color: str,
        divisible_by: int,
        target_device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int, int]:
        out_image = image if image.device == target_device else image.to(target_device)
        out_mask = None if mask is None else (mask if mask.device == target_device else mask.to(target_device))
        _, src_h, src_w, channels = out_image.shape

        mode = str(keep_proportion or "crop")
        pad_mode = mode in {"pad", "pad_edge", "pad_edge_pixel", "pillarbox_blur"}

        if mode == "stretch":
            inner_w = src_w if width == 0 else int(width)
            inner_h = src_h if height == 0 else int(height)
            inner_w, inner_h = self._apply_divisible(inner_w, inner_h, divisible_by)
            resized = self._resize_nhwc(out_image, inner_w, inner_h, upscale_method)
            resized_mask = None if out_mask is None else self._resize_mask(out_mask, inner_w, inner_h, upscale_method)
            return resized, resized_mask, inner_w, inner_h

        if mode == "crop":
            crop_target_w = src_w if width == 0 else int(width)
            crop_target_h = src_h if height == 0 else int(height)
            crop_target_w, crop_target_h = self._apply_divisible(crop_target_w, crop_target_h, divisible_by)
            x, y, crop_w, crop_h = self._crop_box(src_w, src_h, crop_target_w, crop_target_h, crop_position)
            cropped = out_image[:, y:y + crop_h, x:x + crop_w, :]
            cropped_mask = None if out_mask is None else out_mask[:, y:y + crop_h, x:x + crop_w]
            resized = self._resize_nhwc(cropped, crop_target_w, crop_target_h, upscale_method)
            resized_mask = None if cropped_mask is None else self._resize_mask(cropped_mask, crop_target_w, crop_target_h, upscale_method)
            return resized, resized_mask, crop_target_w, crop_target_h

        inner_w, inner_h = self._fit_size(src_w, src_h, int(width), int(height), mode)
        inner_w, inner_h = self._apply_divisible(inner_w, inner_h, divisible_by)
        resized = self._resize_nhwc(out_image, inner_w, inner_h, upscale_method)
        resized_mask = None if out_mask is None else self._resize_mask(out_mask, inner_w, inner_h, upscale_method)

        if not pad_mode:
            return resized, resized_mask, inner_w, inner_h

        canvas_w = src_w if width == 0 else int(width)
        canvas_h = src_h if height == 0 else int(height)
        canvas_w = max(canvas_w, inner_w)
        canvas_h = max(canvas_h, inner_h)
        canvas_w, canvas_h = self._apply_divisible(canvas_w, canvas_h, divisible_by)

        pad_value = self._parse_pad_color(pad_color, channels, resized.dtype).to(torch.device("cpu"))
        canvas = torch.empty((resized.shape[0], canvas_h, canvas_w, channels), dtype=resized.dtype, device=torch.device("cpu"))
        for channel_index in range(channels):
            canvas[..., channel_index].fill_(float(pad_value[channel_index].item()))

        offset_x, offset_y = self._pad_offsets(canvas_w, canvas_h, inner_w, inner_h, crop_position)
        canvas[:, offset_y:offset_y + inner_h, offset_x:offset_x + inner_w, :] = resized.cpu()

        canvas_mask = None
        if resized_mask is not None:
            canvas_mask = torch.zeros((resized_mask.shape[0], canvas_h, canvas_w), dtype=resized_mask.dtype, device=torch.device("cpu"))
            canvas_mask[:, offset_y:offset_y + inner_h, offset_x:offset_x + inner_w] = resized_mask.cpu()
        return canvas, canvas_mask, canvas_w, canvas_h

    def resize(
        self,
        image,
        width,
        height,
        upscale_method,
        keep_proportion,
        pad_color,
        crop_position,
        divisible_by,
        unique_id,
        mask=None,
        device="cpu",
    ):
        if image is None:
            raise ValueError("image is required")
        if image.ndim != 4:
            raise ValueError("Expected IMAGE tensor in NHWC format [B,H,W,C]")

        batch = int(image.shape[0])
        target_device = self._resolve_device(str(device or "cpu"), str(upscale_method))
        subbatch = 24 if target_device.type == "cpu" else 8

        if batch <= subbatch:
            out_image, out_mask, out_w, out_h = self._process_subbatch(
                image,
                mask,
                int(width),
                int(height),
                str(keep_proportion),
                str(upscale_method),
                str(crop_position),
                str(pad_color),
                int(divisible_by),
                target_device,
            )
            if out_image.device.type != "cpu":
                out_image = out_image.cpu()
            if out_mask is not None and out_mask.device.type != "cpu":
                out_mask = out_mask.cpu()
            return out_image.contiguous(), int(out_w), int(out_h), out_mask

        out_image = None
        out_mask = None
        out_w = 0
        out_h = 0
        total_batches = (batch + subbatch - 1) // subbatch

        for batch_index, start in enumerate(range(0, batch, subbatch), start=1):
            end = min(start + subbatch, batch)
            sub_image = image[start:end]
            sub_mask = None if mask is None else mask[start:end]
            sub_out_image, sub_out_mask, out_w, out_h = self._process_subbatch(
                sub_image,
                sub_mask,
                int(width),
                int(height),
                str(keep_proportion),
                str(upscale_method),
                str(crop_position),
                str(pad_color),
                int(divisible_by),
                target_device,
            )

            sub_out_image = sub_out_image.cpu().contiguous()
            if out_image is None:
                out_image = torch.empty(
                    (batch, int(sub_out_image.shape[1]), int(sub_out_image.shape[2]), int(sub_out_image.shape[3])),
                    dtype=sub_out_image.dtype,
                    device=torch.device("cpu"),
                )
            out_image[start:end] = sub_out_image

            if sub_out_mask is not None:
                sub_out_mask = sub_out_mask.cpu().contiguous()
                if out_mask is None:
                    out_mask = torch.empty(
                        (batch, int(sub_out_mask.shape[1]), int(sub_out_mask.shape[2])),
                        dtype=sub_out_mask.dtype,
                        device=torch.device("cpu"),
                    )
                out_mask[start:end] = sub_out_mask

            _log.info(
                "[IAMCCS_ImageResizeBatchSafe] batch %s/%s processed (%s/%s frames)",
                batch_index,
                total_batches,
                end,
                batch,
            )

        if out_image is None:
            raise RuntimeError("Resize produced no output")
        return out_image, int(out_w), int(out_h), out_mask