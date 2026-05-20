import json
import math
from typing import Any, Dict, Optional

try:
    import torch
except Exception:  # pragma: no cover - ComfyUI normally provides torch
    torch = None


SUPERNODE_LINX_TYPE = "IAMCCS_SUPERNODE_LINX"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return int(default)


def _safe_json_loads(value: Any) -> Any:
    try:
        return json.loads(str(value or ""))
    except Exception:
        return None


def _round_ltx_frames(frames: int, mode: str) -> int:
    value = max(1, int(frames))
    if str(mode) == "none":
        return value
    rounded = int(round((value - 1) / 8.0) * 8 + 1)
    if str(mode) == "nearest_8n_plus_1":
        return max(1, rounded)
    return max(1, int(math.ceil(max(0, value - 1) / 8.0) * 8 + 1))


def _cine_linx_resources(cine_linx: Any) -> Dict[str, Any]:
    if not isinstance(cine_linx, dict):
        return {}
    resources = cine_linx.get("resources")
    return resources if isinstance(resources, dict) else {}


def _cine_linx_outputs(cine_linx: Any) -> Dict[str, Any]:
    if not isinstance(cine_linx, dict):
        return {}
    outputs = cine_linx.get("outputs")
    return outputs if isinstance(outputs, dict) else {}


def _duration_from_cine_linx(cine_linx: Any) -> Optional[float]:
    resources = _cine_linx_resources(cine_linx)
    outputs = _cine_linx_outputs(cine_linx)
    payload = resources.get("cine_payload") if isinstance(resources.get("cine_payload"), dict) else {}
    for source in (resources, outputs, payload):
        for key in ("cine_duration_seconds", "duration_seconds", "duration", "duration_sec"):
            if isinstance(source, dict) and key in source:
                value = _safe_float(source.get(key), 0.0)
                if value > 0:
                    return value
    return None


def _frame_rate_from_cine_linx(cine_linx: Any) -> Optional[int]:
    resources = _cine_linx_resources(cine_linx)
    outputs = _cine_linx_outputs(cine_linx)
    payload = resources.get("cine_payload") if isinstance(resources.get("cine_payload"), dict) else {}
    for source in (resources, outputs, payload):
        for key in ("cine_frame_rate", "frame_rate", "fps"):
            if isinstance(source, dict) and key in source:
                value = _safe_int(source.get(key), 0)
                if value > 0:
                    return value
    return None


def _duration_from_timeline_data(timeline_data: Any) -> Optional[float]:
    data = _safe_json_loads(timeline_data)
    if not isinstance(data, dict):
        return None
    for key in ("duration_seconds", "duration", "duration_sec"):
        value = _safe_float(data.get(key), 0.0)
        if value > 0:
            return value
    payload = data.get("payload") if isinstance(data.get("payload"), dict) else {}
    for key in ("duration_seconds", "duration", "duration_sec"):
        value = _safe_float(payload.get(key), 0.0)
        if value > 0:
            return value
    return None


def _guide_count_from_timeline_data(timeline_data: Any) -> int:
    data = _safe_json_loads(timeline_data)
    if isinstance(data, dict):
        rows = data.get("guides") or data.get("keyframes") or data.get("rows") or data.get("segments") or []
    elif isinstance(data, list):
        rows = data
    else:
        rows = []
    count = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("use_guide") is False:
            continue
        ref = _safe_int(row.get("ref", row.get("reference_index", row.get("image_ref", 0))), 0)
        strength = _safe_float(row.get("strength", row.get("guide_strength", row.get("force", 1.0))), 1.0)
        if ref > 0 and strength > 0:
            count += 1
    return count


class IAMCCS_CineBoardDurationLock:
    """Lock production latent length to the shotboard duration, without guide-tail padding."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline_data": ("STRING", {"default": "", "multiline": True}),
                "duration_seconds": ("INT", {"default": 8, "min": 1, "max": 36000, "step": 1}),
                "frame_rate": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1}),
                "ltx_round_mode": (["up_8n_plus_1", "nearest_8n_plus_1", "none"], {"default": "up_8n_plus_1"}),
            },
            "optional": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
            },
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("model_length_frames", "target_frames", "duration_seconds_exact", "frame_rate_int", "report")
    FUNCTION = "compute"
    CATEGORY = "IAMCCS/Cine/00 Utilities"

    def compute(self, timeline_data, duration_seconds, frame_rate, ltx_round_mode, cine_linx=None):
        fps = max(1, _frame_rate_from_cine_linx(cine_linx) or _safe_int(frame_rate, 24))
        duration = (
            _duration_from_cine_linx(cine_linx)
            or _duration_from_timeline_data(timeline_data)
            or max(0.1, _safe_float(duration_seconds, 8.0))
        )
        raw_frames = max(1, int(round(float(duration) * fps)))
        target_frames = _round_ltx_frames(raw_frames, str(ltx_round_mode))
        guide_count = _guide_count_from_timeline_data(timeline_data)
        report = json.dumps(
            {
                "node": "IAMCCS_CineBoardDurationLock",
                "duration_seconds_exact": float(duration),
                "frame_rate": int(fps),
                "raw_frames": int(raw_frames),
                "target_frames": int(target_frames),
                "model_length_frames": int(target_frames),
                "guide_count_observed": int(guide_count),
                "guide_tail_padding_frames": 0,
                "ltx_round_mode": str(ltx_round_mode),
                "truth": "The production latent length is locked to the board duration. Guides do not extend final narrative duration.",
            },
            ensure_ascii=False,
            indent=2,
        )
        return int(target_frames), int(target_frames), float(duration), int(fps), report


class IAMCCS_CineLatentDurationCrop:
    """Safety crop for video latents when a workflow branch still carries padded tail frames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "target_frames": ("INT", {"default": 81, "min": 1, "max": 36000, "step": 1}),
                "ltx_time_factor": ("INT", {"default": 8, "min": 1, "max": 32, "step": 1}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "report")
    FUNCTION = "crop"
    CATEGORY = "IAMCCS/Cine/00 Utilities"

    def crop(self, latent, target_frames, ltx_time_factor):
        if not isinstance(latent, dict) or torch is None:
            return latent, json.dumps({"node": "IAMCCS_CineLatentDurationCrop", "changed": False, "reason": "invalid latent"})
        samples = latent.get("samples")
        if not torch.is_tensor(samples) or samples.ndim < 3:
            return latent, json.dumps({"node": "IAMCCS_CineLatentDurationCrop", "changed": False, "reason": "missing samples tensor"})
        target = max(1, _safe_int(target_frames, 1))
        time_factor = max(1, _safe_int(ltx_time_factor, 8))
        target_latent_frames = max(1, int(math.ceil(max(1, target - 1) / float(time_factor))) + 1)
        current_latent_frames = int(samples.shape[2]) if samples.ndim == 5 else int(samples.shape[0])
        next_latent = dict(latent)
        if samples.ndim == 5 and current_latent_frames > target_latent_frames:
            next_latent["samples"] = samples[:, :, :target_latent_frames, :, :].clone()
        elif samples.ndim != 5 and current_latent_frames > target_latent_frames:
            next_latent["samples"] = samples[:target_latent_frames].clone()
        if "noise_mask" in latent and torch.is_tensor(latent["noise_mask"]):
            mask = latent["noise_mask"]
            if mask.ndim == 5 and int(mask.shape[2]) > target_latent_frames:
                next_latent["noise_mask"] = mask[:, :, :target_latent_frames, :, :].clone()
            elif mask.ndim != 5 and int(mask.shape[0]) > target_latent_frames:
                next_latent["noise_mask"] = mask[:target_latent_frames].clone()
        report = json.dumps(
            {
                "node": "IAMCCS_CineLatentDurationCrop",
                "target_pixel_frames": int(target),
                "target_latent_frames": int(target_latent_frames),
                "current_latent_frames": int(current_latent_frames),
                "changed": bool(current_latent_frames > target_latent_frames),
            },
            ensure_ascii=False,
            indent=2,
        )
        return next_latent, report


NODE_CLASS_MAPPINGS = {
    "IAMCCS_CineBoardDurationLock": IAMCCS_CineBoardDurationLock,
    "IAMCCS_CineLatentDurationCrop": IAMCCS_CineLatentDurationCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_CineBoardDurationLock": "IAMCCS Cine Board Duration Lock",
    "IAMCCS_CineLatentDurationCrop": "IAMCCS Cine Latent Duration Crop",
}
