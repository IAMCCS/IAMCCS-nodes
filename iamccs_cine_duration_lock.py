import json
import math
from typing import Any, Dict, Optional

try:
    import torch
except Exception:  # pragma: no cover - ComfyUI normally provides torch
    torch = None

try:
    from comfy_api.latest import InputImpl, Types
except Exception:  # pragma: no cover - older ComfyUI builds may not expose VIDEO helpers
    InputImpl = None
    Types = None


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


def _clone_cine_linx(cine_linx: Any, role: str) -> Dict[str, Any]:
    if isinstance(cine_linx, dict):
        try:
            cloned = json.loads(json.dumps(cine_linx, default=str))
        except Exception:
            cloned = dict(cine_linx)
    else:
        cloned = {
            "type": SUPERNODE_LINX_TYPE,
            "schema": "iamccs.cine_linx",
            "version": 1,
            "resources": {},
            "outputs": {},
            "chain": [],
        }
    cloned.setdefault("type", SUPERNODE_LINX_TYPE)
    cloned.setdefault("schema", "iamccs.cine_linx")
    cloned.setdefault("version", 1)
    cloned.setdefault("resources", {})
    cloned.setdefault("outputs", {})
    cloned.setdefault("chain", [])
    if isinstance(cloned.get("chain"), list):
        cloned["chain"].append({"role": role, "node": role})
    return cloned


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


def _tail_trim_from_cine_linx(cine_linx: Any) -> Optional[int]:
    resources = _cine_linx_resources(cine_linx)
    outputs = _cine_linx_outputs(cine_linx)
    payload = resources.get("cine_payload") if isinstance(resources.get("cine_payload"), dict) else {}
    for source in (resources, outputs, payload):
        for key in (
            "shotboard_tail_trim_frames",
            "cine_tail_trim_frames",
            "final_tail_trim_frames",
            "render_tail_trim_frames",
        ):
            if isinstance(source, dict) and key in source:
                value = _safe_int(source.get(key), -1)
                if value >= 0:
                    return value
    return None


def _target_frames_from_cine_linx(cine_linx: Any) -> Optional[int]:
    resources = _cine_linx_resources(cine_linx)
    outputs = _cine_linx_outputs(cine_linx)
    payload = resources.get("cine_payload") if isinstance(resources.get("cine_payload"), dict) else {}
    nested = []
    for source in (resources, outputs, payload):
        if not isinstance(source, dict):
            continue
        for key in (
            "cine_multigeneration_take_package",
            "cine_take_package",
            "take_package",
            "cine_audio_tracks",
            "flfreal_compile",
        ):
            value = source.get(key)
            if isinstance(value, dict):
                nested.append(value)
    for source in (*nested, resources, outputs, payload):
        if not isinstance(source, dict):
            continue
        for key in (
            "duration_frames",
            "target_frames",
            "cine_target_frames",
            "source_end_frames",
            "visual_end_frames",
            "audio_end_frames",
        ):
            if key in source:
                value = _safe_int(source.get(key), 0)
                if value > 0:
                    return value
    duration = _duration_from_cine_linx(cine_linx)
    fps = _frame_rate_from_cine_linx(cine_linx)
    if duration and fps:
        return max(1, int(round(float(duration) * int(fps))))
    return None


def _normalize_audio_waveform(audio: Any):
    if torch is None or not isinstance(audio, dict):
        return None, 0
    waveform = audio.get("waveform")
    sample_rate = _safe_int(audio.get("sample_rate"), 0)
    if sample_rate <= 0 or not torch.is_tensor(waveform):
        return None, sample_rate
    if waveform.ndim == 1:
        waveform = waveform.view(1, 1, -1)
    elif waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim != 3:
        return None, sample_rate
    return waveform, sample_rate


def _trim_audio_to_duration(audio: Any, duration_seconds: float) -> Any:
    waveform, sample_rate = _normalize_audio_waveform(audio)
    if waveform is None or sample_rate <= 0:
        return audio
    target_samples = max(1, int(round(float(duration_seconds) * float(sample_rate))))
    if int(waveform.shape[-1]) <= target_samples:
        return audio
    out = dict(audio)
    out["waveform"] = waveform[:, :, :target_samples].clone()
    out["sample_rate"] = int(sample_rate)
    return out


def _trim_image_batch(frames: Any, trim_frames: int):
    if torch is None or frames is None or not torch.is_tensor(frames):
        return frames, 0, 0
    if frames.ndim < 1:
        return frames, 0, 0
    frame_count = int(frames.shape[0])
    trim = max(0, min(_safe_int(trim_frames, 0), max(0, frame_count - 1)))
    if trim <= 0:
        return frames, frame_count, 0
    return frames[: frame_count - trim].clone(), frame_count, trim


def _crop_image_batch_to_frames(frames: Any, target_frames: int):
    if torch is None or frames is None or not torch.is_tensor(frames):
        return frames, 0, 0
    if frames.ndim < 1:
        return frames, 0, 0
    frame_count = int(frames.shape[0])
    target = _safe_int(target_frames, 0)
    if target <= 0 or frame_count <= target:
        return frames, frame_count, 0
    target = max(1, target)
    return frames[:target].clone(), frame_count, frame_count - target


def _crop_video_object_to_frames(video: Any, target_frames: int, trim_audio: bool):
    if video is None or InputImpl is None or Types is None or not hasattr(video, "get_components"):
        return video, None, 0, 0, None
    try:
        components = video.get_components()
    except Exception:
        return video, None, 0, 0, None
    images = getattr(components, "images", None)
    audio = getattr(components, "audio", None)
    frame_rate = _safe_float(getattr(components, "frame_rate", 24.0), 24.0)
    cropped_images, original_frames, effective_crop = _crop_image_batch_to_frames(images, target_frames)
    if effective_crop <= 0 or cropped_images is None:
        return video, images, original_frames, 0, audio
    next_audio = audio
    if trim_audio:
        next_duration = float(int(cropped_images.shape[0])) / max(1.0, float(frame_rate))
        next_audio = _trim_audio_to_duration(audio, next_duration)
    next_video = InputImpl.VideoFromComponents(
        Types.VideoComponents(images=cropped_images, audio=next_audio, frame_rate=frame_rate)
    )
    return next_video, cropped_images, original_frames, effective_crop, next_audio


def _trim_video_object(video: Any, trim_frames: int, trim_audio: bool):
    if video is None or InputImpl is None or Types is None or not hasattr(video, "get_components"):
        return video, None, 0, 0, None
    try:
        components = video.get_components()
    except Exception:
        return video, None, 0, 0, None
    images = getattr(components, "images", None)
    audio = getattr(components, "audio", None)
    frame_rate = _safe_float(getattr(components, "frame_rate", 24.0), 24.0)
    trimmed_images, original_frames, effective_trim = _trim_image_batch(images, trim_frames)
    if effective_trim <= 0 or trimmed_images is None:
        return video, images, original_frames, 0, audio
    next_audio = audio
    if trim_audio:
        next_duration = float(int(trimmed_images.shape[0])) / max(1.0, float(frame_rate))
        next_audio = _trim_audio_to_duration(audio, next_duration)
    next_video = InputImpl.VideoFromComponents(
        Types.VideoComponents(images=trimmed_images, audio=next_audio, frame_rate=frame_rate)
    )
    return next_video, trimmed_images, original_frames, effective_trim, next_audio


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


class IAMCCS_CineShotboardTailTrimPolicy:
    """Attach an explicit Shotboard final-tail trim policy to cine_linx metadata."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "mode": (["enabled", "disabled"], {"default": "enabled"}),
                "tail_trim_frames": ("INT", {"default": 1, "min": 0, "max": 24, "step": 1}),
            }
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE, "STRING")
    RETURN_NAMES = ("cine_linx", "report")
    FUNCTION = "apply"
    CATEGORY = "IAMCCS/Cine/00 Utilities"

    def apply(self, cine_linx, mode, tail_trim_frames):
        trim = max(0, _safe_int(tail_trim_frames, 1)) if str(mode) == "enabled" else 0
        out_linx = _clone_cine_linx(cine_linx, "IAMCCS_CineShotboardTailTrimPolicy")
        resources = out_linx.setdefault("resources", {})
        outputs = out_linx.setdefault("outputs", {})
        resources["shotboard_tail_trim_frames"] = int(trim)
        resources["cine_tail_trim_frames"] = int(trim)
        outputs["shotboard_tail_trim_frames"] = int(trim)
        report_obj = {
            "node": "IAMCCS_CineShotboardTailTrimPolicy",
            "enabled": bool(trim > 0),
            "tail_trim_frames": int(trim),
            "truth": "Shotboard owns final generated-tail cleanup. Manual video-editor trim remains separate.",
        }
        report = json.dumps(report_obj, ensure_ascii=False, indent=2)
        outputs["shotboard_tail_trim_report"] = report_obj
        return out_linx, report


class IAMCCS_CineShotboardFinalFrameTrim:
    """Trim generated tail frames after final decode, before save/collector/editor."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trim_policy": (["auto_from_cine_linx", "force_value", "off"], {"default": "auto_from_cine_linx"}),
                "tail_trim_frames": ("INT", {"default": 1, "min": 0, "max": 24, "step": 1}),
                "trim_embedded_video_audio": (["yes", "no"], {"default": "yes"}),
            },
            "optional": {
                "video": ("VIDEO",),
                "frames": ("IMAGE",),
                "audio": ("AUDIO",),
                "cine_linx": (SUPERNODE_LINX_TYPE,),
            },
        }

    RETURN_TYPES = ("VIDEO", "IMAGE", "AUDIO", SUPERNODE_LINX_TYPE, "STRING")
    RETURN_NAMES = ("video", "frames", "audio", "cine_linx", "report")
    FUNCTION = "trim"
    CATEGORY = "IAMCCS/Cine/00 Utilities"

    def trim(
        self,
        trim_policy,
        tail_trim_frames,
        trim_embedded_video_audio,
        video=None,
        frames=None,
        audio=None,
        cine_linx=None,
    ):
        policy = str(trim_policy or "auto_from_cine_linx")
        if policy == "off":
            requested_trim = 0
        elif policy == "auto_from_cine_linx":
            requested_trim = _tail_trim_from_cine_linx(cine_linx)
            if requested_trim is None:
                requested_trim = _safe_int(tail_trim_frames, 1)
        else:
            requested_trim = _safe_int(tail_trim_frames, 1)
        requested_trim = max(0, requested_trim)
        requested_target_frames = _target_frames_from_cine_linx(cine_linx) if policy != "off" else None

        out_video, video_frames, video_original, video_duration_crop, embedded_audio = _crop_video_object_to_frames(
            video,
            int(requested_target_frames or 0),
            str(trim_embedded_video_audio) == "yes",
        )
        out_video, video_frames, video_original, video_trim, embedded_audio = _trim_video_object(
            out_video,
            requested_trim,
            str(trim_embedded_video_audio) == "yes",
        )
        out_frames = frames
        frame_original = 0
        frame_duration_crop = 0
        frame_trim = 0
        if frames is not None:
            out_frames, frame_original, frame_duration_crop = _crop_image_batch_to_frames(frames, int(requested_target_frames or 0))
            out_frames, frame_original_after_crop, frame_trim = _trim_image_batch(out_frames, requested_trim)
            if frame_original <= 0:
                frame_original = frame_original_after_crop
        elif video_frames is not None:
            out_frames = video_frames
            frame_original = video_original
            frame_trim = video_trim

        out_audio = audio
        out_linx = _clone_cine_linx(cine_linx, "IAMCCS_CineShotboardFinalFrameTrim")
        resources = out_linx.setdefault("resources", {})
        outputs = out_linx.setdefault("outputs", {})
        effective_trim = max(int(video_trim), int(frame_trim))
        if effective_trim > 0:
            resources["shotboard_tail_trim_frames_applied"] = int(effective_trim)
            outputs["shotboard_tail_trim_frames_applied"] = int(effective_trim)
        report_obj = {
            "node": "IAMCCS_CineShotboardFinalFrameTrim",
            "policy": policy,
            "target_frames_from_cine_linx": int(requested_target_frames or 0),
            "requested_tail_trim_frames": int(requested_trim),
            "duration_crop_frames": int(max(video_duration_crop, frame_duration_crop)),
            "effective_tail_trim_frames": int(effective_trim),
            "video_original_frames": int(video_original or 0),
            "frames_original_frames": int(frame_original or 0),
            "trimmed_embedded_video_audio": bool(str(trim_embedded_video_audio) == "yes" and video_trim > 0),
            "has_passthrough_audio": bool(audio is not None),
            "truth": "Use after final decode and before SaveVideo/collector to remove generated chroma-offset tail frames. This is distinct from creative editor trim.",
        }
        report = json.dumps(report_obj, ensure_ascii=False, indent=2)
        outputs["shotboard_tail_trim_report"] = report_obj
        return out_video, out_frames, out_audio, out_linx, report


NODE_CLASS_MAPPINGS = {
    "IAMCCS_CineBoardDurationLock": IAMCCS_CineBoardDurationLock,
    "IAMCCS_CineLatentDurationCrop": IAMCCS_CineLatentDurationCrop,
    "IAMCCS_CineShotboardTailTrimPolicy": IAMCCS_CineShotboardTailTrimPolicy,
    "IAMCCS_CineShotboardFinalFrameTrim": IAMCCS_CineShotboardFinalFrameTrim,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_CineBoardDurationLock": "IAMCCS Cine Board Duration Lock",
    "IAMCCS_CineLatentDurationCrop": "IAMCCS Cine Latent Duration Crop",
    "IAMCCS_CineShotboardTailTrimPolicy": "IAMCCS Cine Shotboard Tail Trim Policy",
    "IAMCCS_CineShotboardFinalFrameTrim": "IAMCCS Cine Shotboard Final Frame Trim",
}
