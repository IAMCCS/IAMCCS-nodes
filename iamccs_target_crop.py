import json

import torch
import torch.nn.functional as F


def _to_int(value, default=0):
    try:
        return int(round(float(value)))
    except Exception:
        return int(default)


def _resize_image_batch(image, width, height):
    width = max(1, int(width))
    height = max(1, int(height))
    if image.shape[1] == height and image.shape[2] == width:
        return image
    nchw = image.movedim(-1, 1)
    resized = F.interpolate(nchw, size=(height, width), mode="bilinear", align_corners=False)
    return resized.movedim(1, -1).clamp(0, 1)


class IAMCCS_TargetCrop:
    DISPLAY_NAME = "IAMCCS Target Crop"
    CATEGORY = "IAMCCS/Cine/Ideogram"
    FUNCTION = "crop"
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "x", "y", "width", "height", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_x": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1}),
                "target_y": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1}),
                "target_width": ("INT", {"default": 1280, "min": 1, "max": 16384, "step": 1}),
                "target_height": ("INT", {"default": 720, "min": 1, "max": 16384, "step": 1}),
                "resize_to_target": ("BOOLEAN", {"default": False}),
            }
        }

    def crop(self, image, target_x=0, target_y=0, target_width=1280, target_height=720, resize_to_target=False):
        if image is None:
            raise ValueError("IAMCCS Target Crop: missing image")
        if not hasattr(image, "shape") or len(image.shape) != 4:
            raise ValueError("IAMCCS Target Crop expects IMAGE tensor [B,H,W,C]")

        _, source_h, source_w, _ = image.shape
        x = max(0, min(_to_int(target_x), int(source_w) - 1))
        y = max(0, min(_to_int(target_y), int(source_h) - 1))
        requested_w = max(1, _to_int(target_width, 1280))
        requested_h = max(1, _to_int(target_height, 720))
        x2 = max(x + 1, min(int(source_w), x + requested_w))
        y2 = max(y + 1, min(int(source_h), y + requested_h))
        cropped = image[:, y:y2, x:x2, :].contiguous()

        if bool(resize_to_target) and (cropped.shape[1] != requested_h or cropped.shape[2] != requested_w):
            cropped = _resize_image_batch(cropped, requested_w, requested_h)

        report = {
            "source_width": int(source_w),
            "source_height": int(source_h),
            "requested": {
                "x": int(target_x),
                "y": int(target_y),
                "width": requested_w,
                "height": requested_h,
            },
            "applied": {
                "x": int(x),
                "y": int(y),
                "width": int(x2 - x),
                "height": int(y2 - y),
            },
            "resize_to_target": bool(resize_to_target),
            "output_width": int(cropped.shape[2]),
            "output_height": int(cropped.shape[1]),
        }
        return (cropped, int(x), int(y), int(cropped.shape[2]), int(cropped.shape[1]), json.dumps(report, ensure_ascii=False, indent=2))


NODE_CLASS_MAPPINGS = {
    "IAMCCS_TargetCrop": IAMCCS_TargetCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_TargetCrop": "IAMCCS Target Crop",
}
