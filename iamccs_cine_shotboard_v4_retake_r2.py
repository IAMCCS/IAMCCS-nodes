from __future__ import annotations

import copy
import json
import math
from typing import Any, Dict

import nodes as comfy_nodes
import torch

from .iamccs_cine_shotboard_v4_backend import (
    SUPERNODE_LINX_TYPE,
    _build_combined_audio,
    _find_timeline,
    _json_dumps,
    _load_video_frames,
    _resolve_media_path,
    _retake_active,
    _retake_video,
    _safe_bool,
    _safe_float,
    _safe_int,
)


def _ltx_length(frame_count: int) -> int:
    """Return the smallest valid LTX video length (8n+1) >= frame_count."""

    frames = max(1, int(frame_count))
    if frames == 1:
        return 1
    return int(math.ceil((frames - 1) / 8.0) * 8 + 1)


def _retake_prompt(timeline: Dict[str, Any]) -> str:
    return str(
        timeline.get("retake_global_prompt")
        or timeline.get("retakePrompt")
        or timeline.get("retake_prompt")
        or timeline.get("global_prompt")
        or timeline.get("prompt")
        or ""
    ).strip()


def _negative_prompt(timeline: Dict[str, Any], fallback: str = "") -> str:
    return str(
        timeline.get("retake_negative_prompt")
        or timeline.get("negative_prompt")
        or fallback
        or "pc game, console game, video game, cartoon, childish, ugly"
    ).strip()


class IAMCCS_CineShotboardV4RetakeSourceR2:
    """Materialize a Shotboard V4 retake range for the official LTX 2.3 inpaint graph.

    The existing V4 backend produces a latent noise mask.  The official LTX 2.3
    in/outpainting workflow instead needs source pixels plus a pixel-space video
    mask before ``LTXVInpaintPreprocess``.  This bridge performs that conversion
    without changing the current V4 planner or backend.

    ``duration_frames`` remains the user's editorial duration.  Source frames and
    audio are padded to 8n+1 for LTX; ``guide_data`` retains the raw duration so an
    ``IAMCCS_LTXVideoDurationCrop`` node can remove the padding after stage two.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
            },
            "optional": {
                "spatial_mask": ("MASK",),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
        "AUDIO",
        "FLOAT",
        "INT",
        "GUIDE_DATA",
        "FLOAT",
        "STRING",
    )
    RETURN_NAMES = (
        "source_frames",
        "retake_mask",
        "source_audio",
        "frame_rate",
        "duration_frames",
        "guide_data",
        "retake_strength",
        "report",
    )
    FUNCTION = "execute"
    CATEGORY = "IAMCCS/Cine/Shotboard V4/Experimental R2"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    def execute(self, cine_linx, spatial_mask=None):
        linx = cine_linx if isinstance(cine_linx, dict) else {}
        timeline = _find_timeline(linx)
        if not timeline:
            raise ValueError("Shotboard V4 Retake R2: cine_linx has no V4 timeline data.")
        if not _retake_active(timeline):
            raise ValueError(
                "Shotboard V4 Retake R2: choose Edit = Retake in Shotboard V4 before queueing."
            )

        retake = _retake_video(timeline)
        source_ref = str(
            retake.get("imageFile")
            or retake.get("videoFile")
            or retake.get("fileName")
            or retake.get("path")
            or ""
        ).strip()
        source_path = _resolve_media_path(source_ref)
        if not source_path:
            raise ValueError(
                "Shotboard V4 Retake R2: source video is missing. Import/select a video in the V4 main lane."
            )

        fps = max(
            1.0,
            _safe_float(
                timeline.get("frame_rate", timeline.get("frameRate", timeline.get("fps", 24.0))),
                24.0,
            ),
        )
        timeline_start = max(
            0,
            _safe_int(timeline.get("normalStartFrame", timeline.get("normal_start_frame", 0)), 0),
        )
        raw_duration = _safe_int(
            timeline.get(
                "normalDurationFrames",
                timeline.get("normal_duration_frames", timeline.get("duration_frames", 0)),
            ),
            0,
        )
        if raw_duration <= 0:
            duration_seconds = _safe_float(
                timeline.get("duration_seconds", timeline.get("durationSeconds", 0.0)),
                0.0,
            )
            if duration_seconds > 0:
                raw_duration = int(round(duration_seconds * fps))
        if raw_duration <= 0:
            raw_duration = _safe_int(
                retake.get("length", retake.get("videoDurationFrames", 0)),
                0,
            )
        raw_duration = max(1, raw_duration)
        padded_duration = _ltx_length(raw_duration)

        source_clip_start = max(0, _safe_int(retake.get("start", 0), 0))
        source_trim = max(
            0,
            _safe_int(retake.get("trimStart", retake.get("trim_start", 0)), 0)
            + max(0, timeline_start - source_clip_start),
        )
        source_frames = _load_video_frames(
            source_ref,
            source_trim,
            raw_duration,
            fps,
            "nearest",
        )
        if int(source_frames.shape[0]) < raw_duration:
            source_frames = torch.cat(
                [
                    source_frames,
                    source_frames[-1:].repeat(raw_duration - int(source_frames.shape[0]), 1, 1, 1),
                ],
                dim=0,
            )
        source_frames = source_frames[:raw_duration, :, :, :3]
        if padded_duration > raw_duration:
            source_frames = torch.cat(
                [
                    source_frames,
                    source_frames[-1:].repeat(padded_duration - raw_duration, 1, 1, 1),
                ],
                dim=0,
            )

        height = int(source_frames.shape[1])
        width = int(source_frames.shape[2])
        outside_value = max(
            0.0,
            min(
                1.0,
                _safe_float(
                    timeline.get(
                        "retakeMaskInitValueVideo",
                        timeline.get("retake_mask_init_value_video", 0.0),
                    ),
                    0.0,
                ),
            ),
        )
        regenerate_video = _safe_bool(
            timeline.get(
                "retakeRegenerateVideo",
                timeline.get("retake_regenerate_video", True),
            ),
            True,
        )
        retake_start = _safe_int(
            timeline.get("retakeStart", timeline.get("retake_start", timeline_start)),
            timeline_start,
        )
        retake_length = _safe_int(
            timeline.get("retakeLength", timeline.get("retake_length", raw_duration)),
            raw_duration,
        )
        relative_start = max(0, retake_start - timeline_start)
        relative_end = min(raw_duration, relative_start + max(0, retake_length))
        if relative_end <= relative_start:
            raise ValueError(
                "Shotboard V4 Retake R2: the retake IN/OUT range is empty. Set both boundaries in Settings."
            )

        retake_mask = torch.full(
            (padded_duration, height, width),
            float(outside_value),
            dtype=torch.float32,
        )
        if regenerate_video:
            retake_mask[relative_start:relative_end] = 1.0
        # Padded tail is always protected; it exists only to satisfy LTX's 8n+1 contract.
        if padded_duration > raw_duration:
            retake_mask[raw_duration:] = 0.0

        if torch.is_tensor(spatial_mask):
            mask = spatial_mask.float()
            if mask.ndim == 4:
                mask = mask[..., 0]
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if tuple(mask.shape[-2:]) != (height, width):
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(1),
                    size=(height, width),
                    mode="nearest",
                ).squeeze(1)
            if int(mask.shape[0]) == 1:
                mask = mask.repeat(padded_duration, 1, 1)
            elif int(mask.shape[0]) < padded_duration:
                mask = torch.cat(
                    [mask, mask[-1:].repeat(padded_duration - int(mask.shape[0]), 1, 1)],
                    dim=0,
                )
            retake_mask = retake_mask * mask[:padded_duration].clamp(0.0, 1.0)

        # Reuse the established V4 audio reader, but shift its window by the source trim.
        # This avoids the old retake path's trimStart=0 lipsync offset.
        audio_timeline = copy.deepcopy(timeline)
        audio_retake = dict(retake)
        audio_retake["videoDurationFrames"] = source_trim + padded_duration
        audio_retake["length"] = source_trim + padded_duration
        audio_timeline["retakeVideo"] = audio_retake
        audio_timeline["retake_video"] = audio_retake
        source_audio = _build_combined_audio(
            audio_timeline,
            source_trim,
            padded_duration,
            fps,
            False,
        )

        retake_strength = max(
            0.0,
            min(
                1.0,
                _safe_float(
                    timeline.get("retakeStrength", timeline.get("retake_strength", 1.0)),
                    1.0,
                ),
            ),
        )
        guide_data = {
            "schema": "iamccs.shotboard_v4.retake_inpaint_r2",
            "duration_frames": int(raw_duration),
            "ltxv_length": int(padded_duration),
            "frame_rate": float(fps),
            "source_video": source_ref,
            "source_path": source_path,
            "source_trim_start": int(source_trim),
            "retake_start": int(retake_start),
            "retake_length": int(retake_length),
            "retake_end": int(retake_start + retake_length),
            "relative_retake_start": int(relative_start),
            "relative_retake_end": int(relative_end),
            "retake_strength": float(retake_strength),
            "retake_prompt": _retake_prompt(timeline),
            "timeline_data": _json_dumps(timeline),
        }
        report = json.dumps(guide_data, ensure_ascii=False, indent=2)
        return (
            source_frames,
            retake_mask,
            source_audio,
            float(fps),
            int(raw_duration),
            guide_data,
            float(retake_strength),
            report,
        )


class IAMCCS_CineShotboardV4RetakePromptEncodeR2:
    """Encode the V4 retake prompt while leaving LTX frame-rate conditioning explicit."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "cine_linx": (SUPERNODE_LINX_TYPE,),
            },
            "optional": {
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "pc game, console game, video game, cartoon, childish, ugly",
                        "multiline": True,
                    },
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "negative", "effective_prompt")
    FUNCTION = "execute"
    CATEGORY = "IAMCCS/Cine/Shotboard V4/Experimental R2"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    def execute(self, clip, cine_linx, negative_prompt=""):
        timeline = _find_timeline(cine_linx if isinstance(cine_linx, dict) else {})
        prompt = _retake_prompt(timeline)
        if not prompt:
            raise ValueError(
                "Shotboard V4 Retake R2: write a Retake prompt override (or a global prompt) before queueing."
            )
        negative = _negative_prompt(timeline, negative_prompt)
        positive_conditioning = comfy_nodes.CLIPTextEncode().encode(clip, prompt)[0]
        negative_conditioning = comfy_nodes.CLIPTextEncode().encode(clip, negative)[0]
        return positive_conditioning, negative_conditioning, prompt


NODE_CLASS_MAPPINGS = {
    "IAMCCS_CineShotboardV4RetakeSourceR2": IAMCCS_CineShotboardV4RetakeSourceR2,
    "IAMCCS_CineShotboardV4RetakePromptEncodeR2": IAMCCS_CineShotboardV4RetakePromptEncodeR2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_CineShotboardV4RetakeSourceR2": "IAMCCS Shotboard V4 Retake Source + Mask R2",
    "IAMCCS_CineShotboardV4RetakePromptEncodeR2": "IAMCCS Shotboard V4 Retake Prompt Encode R2",
}
