# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later

"""Isolated MiniMax H3 V2V/Object-Swap backend candidate (R22).

The source video and identity references deliberately live in ``CineInfoH3V2V``
instead of the Shotboard timeline.  Shotboard remains the authority for prompt,
duration and source ranges.  The actual H3 model, sampler, custom-audio drive,
generation backend and native checkpoint remain the existing IAMCCS nodes.

Guide images are explicit inputs.  This module does *not* wrap ControlNet Aux,
Depth Anything or DWPose: optional third-party preprocessors can be connected
upstream, while missing guides fail with a precise validation message.  This
keeps saved workflows deterministic and avoids hidden model downloads.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from .iamccs_minimax_h3_atomic_backend import (
    _resolve_shotplan,
    _run_h3_conditioning_with_cpu_fallback,
)
from .iamccs_minimax_h3_shotboard_core import H3_FPS, align_h3_frames
from .iamccs_supernodes_linx import build_stage_linx_payload


SUPERNODE_LINX_TYPE = "IAMCCS_SUPERNODE_LINX"
CATEGORY = "IAMCCS/MiniMax H3/V2V R22"
RESOURCE_PREFIX = "iamccs_minimax_h3_v2v_"
CONFIG_RESOURCE = f"{RESOURCE_PREFIX}config"
MANIFEST_RESOURCE = f"{RESOURCE_PREFIX}manifest"
STAGE_NAME = "iamccs_minimax_h3_v2v_info_r22"

GUIDE_MODES = ("raw_only", "raw_pose", "raw_depth", "raw_depth_pose")
GUIDE_MODE_OVERRIDES = ("from_shotboard",) + GUIDE_MODES
SOURCE_RANGE_POLICIES = ("timeline_segment", "sequential_requested", "repeat_from_offset")
SOURCE_RANGE_OVERRIDES = ("from_shotboard",) + SOURCE_RANGE_POLICIES
SOURCE_FIT_POLICIES = ("native_adapt", "canvas_pad", "canvas_crop", "stretch")
SOURCE_FIT_OVERRIDES = ("from_shotboard",) + SOURCE_FIT_POLICIES
AUDIO_PAIRING_POLICIES = ("pair_with_source_video", "standalone_reference", "off")
AUDIO_PAIRING_OVERRIDES = ("from_shotboard",) + AUDIO_PAIRING_POLICIES
SOURCE_END_POLICIES = ("hold_last_for_grid", "error")
SOURCE_END_OVERRIDES = ("from_shotboard",) + SOURCE_END_POLICIES
REF_IMAGE_SIZE_POLICIES = ("match", "max")
REF_IMAGE_SIZE_OVERRIDES = ("from_shotboard",) + REF_IMAGE_SIZE_POLICIES
IMAGE_ROLES = ("subject_identity", "wardrobe_object", "environment", "style", "disabled")
AUDIO_BEHAVIOR_NATIVE = "h3_native_generated"
AUDIO_BEHAVIOR_REFERENCE = "h3_ref2va_audio"
AUDIO_BEHAVIOR_LOCKED = "h3_custom_audio_drive"
AUDIO_BEHAVIOR_EXTERNAL = "external_audio_post"
AUDIO_BEHAVIOR_ALIASES = {
    "": AUDIO_BEHAVIOR_NATIVE,
    "native": AUDIO_BEHAVIOR_NATIVE,
    "native_generated": AUDIO_BEHAVIOR_NATIVE,
    "ref2va_audio": AUDIO_BEHAVIOR_REFERENCE,
    "audio_reference": AUDIO_BEHAVIOR_REFERENCE,
    "custom_audio": AUDIO_BEHAVIOR_LOCKED,
    "custom_audio_drive": AUDIO_BEHAVIOR_LOCKED,
    "audio_driven": AUDIO_BEHAVIOR_LOCKED,
    "external_post": AUDIO_BEHAVIOR_EXTERNAL,
    "post_audio": AUDIO_BEHAVIOR_EXTERNAL,
}


def _resources(cine_linx: Any) -> dict[str, Any]:
    if not isinstance(cine_linx, dict):
        return {}
    value = cine_linx.get("resources")
    return value if isinstance(value, dict) else {}


def _shape(value: Any) -> list[int]:
    return [int(item) for item in value.shape] if torch.is_tensor(value) else []


def _audio_meta(audio: Any) -> dict[str, Any]:
    if not isinstance(audio, Mapping) or not torch.is_tensor(audio.get("waveform")):
        return {"connected": False}
    waveform = audio["waveform"]
    sample_rate = max(1, int(audio.get("sample_rate", 32000) or 32000))
    return {
        "connected": True,
        "shape": _shape(waveform),
        "sample_rate": sample_rate,
        "duration_seconds": round(int(waveform.shape[-1]) / sample_rate, 6),
    }


def _materialize_optional_audio(audio: Any) -> dict[str, Any] | None:
    """Resolve a VHS LazyAudioMap once and normalise it for H3 REF2VA.

    VideoHelperSuite returns ``LazyAudioMap`` for both videos with audio and
    video-only containers.  The latter raises only when the mapping is first
    read.  Source audio is optional in R22, so failed/empty lazy materialisation
    becomes a disconnected source rather than aborting the V2V render.

    H3 reference audio is normalised to two channels before entering the native
    MiniMax conditioning node: mono becomes dual-mono, stereo is preserved, and
    deterministic multi-channel handling keeps the first two channels.
    """
    if audio is None:
        return None
    if not isinstance(audio, Mapping):
        raise ValueError("MiniMax H3 V2V source_audio must be an AUDIO mapping")
    try:
        waveform = audio.get("waveform")
        sample_rate = max(1, int(audio.get("sample_rate", 32000) or 32000))
    except Exception:
        # A video-only VHS LazyAudioMap raises while probing its absent stream.
        # Audio is optional, so publish no audio resource and continue raw V2V.
        return None
    if not torch.is_tensor(waveform) or waveform.ndim != 3:
        return None
    if int(waveform.shape[0]) < 1 or int(waveform.shape[1]) < 1 or int(waveform.shape[-1]) < 1:
        return None
    channels = int(waveform.shape[1])
    if channels == 1:
        waveform = waveform.repeat(1, 2, 1)
    elif channels > 2:
        waveform = waveform[:, :2, :]
    return {"waveform": waveform, "sample_rate": sample_rate}


def _finite_float(value: Any, default: float, minimum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        number = float(default)
    if not math.isfinite(number):
        number = float(default)
    return max(float(minimum), number) if minimum is not None else number


def _plan_v2v_settings(shotplan: dict[str, Any]) -> dict[str, Any]:
    value = shotplan.get("v2v_settings")
    return value if isinstance(value, dict) else {}


def _audio_behavior(shotplan: dict[str, Any]) -> str:
    requested = str(shotplan.get("audio_mode", AUDIO_BEHAVIOR_NATIVE) or "").strip().lower()
    behavior = AUDIO_BEHAVIOR_ALIASES.get(requested, requested)
    if behavior not in {
        AUDIO_BEHAVIOR_NATIVE,
        AUDIO_BEHAVIOR_REFERENCE,
        AUDIO_BEHAVIOR_LOCKED,
        AUDIO_BEHAVIOR_EXTERNAL,
    }:
        behavior = AUDIO_BEHAVIOR_NATIVE
    return behavior


def _resolve_choice(
    override: Any,
    plan_value: Any,
    allowed: Iterable[str],
    default: str,
    label: str,
) -> str:
    allowed_set = set(allowed)
    requested = str(override or "from_shotboard").strip().lower()
    if requested == "from_shotboard":
        requested = str(plan_value or default).strip().lower()
    if requested not in allowed_set:
        raise ValueError(f"MiniMax H3 V2V invalid {label}: {requested!r}; expected {sorted(allowed_set)}")
    return requested


def _resolve_v2v_config(shotplan: dict[str, Any], attached: dict[str, Any]) -> dict[str, Any]:
    plan = _plan_v2v_settings(shotplan)
    # The Shotboard owns source-range semantics.  When CineInfo delegates its
    # range policy, its numeric widget must not accidentally mask the offset
    # stored in the shot plan.  An explicit CineInfo range policy is the only
    # case in which its local offset becomes authoritative.
    attached_range_policy = str(attached.get("source_range_policy") or "from_shotboard").strip().lower()
    source_offset = (
        plan.get("source_offset_seconds", 0.0)
        if attached_range_policy == "from_shotboard"
        else attached.get("source_offset_seconds", plan.get("source_offset_seconds", 0.0))
    )
    requested_audio_pairing = _resolve_choice(
        attached.get("audio_pairing"),
        plan.get("requested_audio_pairing", plan.get("audio_pairing")),
        AUDIO_PAIRING_POLICIES,
        "pair_with_source_video",
        "audio pairing",
    )
    audio_behavior = _audio_behavior(shotplan)
    # One policy owns audio for the complete graph.  Only REF2VA-reference mode
    # may place source audio inside native MiniMax reference conditioning.
    # Locked/custom and external-post modes transport the slice to the R21
    # router but keep it out of the REF2VA payload, preventing dual conditioning.
    effective_audio_pairing = (
        "pair_with_source_video"
        if audio_behavior == AUDIO_BEHAVIOR_REFERENCE
        else "off"
    )
    return {
        "guide_mode": _resolve_choice(
            attached.get("guide_mode"), plan.get("guide_mode"), GUIDE_MODES, "raw_only", "guide mode"
        ),
        "source_range_policy": _resolve_choice(
            attached.get("source_range_policy"),
            plan.get("source_range_policy"),
            SOURCE_RANGE_POLICIES,
            "timeline_segment",
            "source range policy",
        ),
        "source_fit": _resolve_choice(
            attached.get("source_fit"), plan.get("source_fit"), SOURCE_FIT_POLICIES, "canvas_pad", "source fit"
        ),
        "audio_pairing": effective_audio_pairing,
        "requested_audio_pairing": requested_audio_pairing,
        "audio_behavior": audio_behavior,
        "source_end_policy": _resolve_choice(
            attached.get("source_end_policy"),
            plan.get("source_end_policy"),
            SOURCE_END_POLICIES,
            "hold_last_for_grid",
            "source end policy",
        ),
        "source_offset_seconds": _finite_float(
            source_offset, 0.0, 0.0
        ),
        "source_fps": _finite_float(attached.get("source_fps", 24.0), 24.0, 1.0),
        "ref_image_size": _resolve_choice(
            attached.get("ref_image_size"),
            shotplan.get("ref_image_size"),
            REF_IMAGE_SIZE_POLICIES,
            "match",
            "reference image size",
        ),
    }


def _validate_guides(config: dict[str, Any], depth_guide: Any, pose_guide: Any) -> None:
    mode = config["guide_mode"]
    missing: list[str] = []
    if mode in {"raw_depth", "raw_depth_pose"} and not torch.is_tensor(depth_guide):
        missing.append("depth_guide")
    if mode in {"raw_pose", "raw_depth_pose"} and not torch.is_tensor(pose_guide):
        missing.append("pose_guide")
    if missing:
        raise ValueError(
            "MiniMax H3 V2V guide mode "
            f"{mode!r} requires explicit preprocessed input(s): {', '.join(missing)}. "
            "Connect Depth Anything V2 and/or DWPose upstream, or select raw_only. "
            "IAMCCS does not hide third-party preprocessors inside this node."
        )


def _clean_previous_v2v(cine_linx: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(cine_linx)
    resources = dict(_resources(cine_linx))
    for key in tuple(resources):
        if key.startswith(RESOURCE_PREFIX):
            resources.pop(key, None)
    cleaned["resources"] = resources
    return cleaned


def _shotplan_chunk(shotplan: dict[str, Any], segment_index: int) -> dict[str, Any]:
    chunks = shotplan.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("MiniMax H3 V2V shotplan has no chunks")
    index = int(segment_index)
    if index < 0 or index >= len(chunks):
        raise IndexError(f"MiniMax H3 V2V segment_index={index} outside 0..{len(chunks) - 1}")
    chunk = chunks[index]
    if not isinstance(chunk, dict):
        raise ValueError(f"MiniMax H3 V2V chunk {index} is invalid")
    return chunk


def _requested_frames(chunk: dict[str, Any]) -> int:
    return max(1, int(chunk.get("requested_frame_count", chunk.get("frame_count", 5)) or 5))


def _segment_source_start(
    shotplan: dict[str, Any],
    chunk: dict[str, Any],
    segment_index: int,
    policy: str,
    offset_seconds: float,
) -> float:
    if policy == "repeat_from_offset":
        return float(offset_seconds)
    if policy == "sequential_requested":
        chunks = shotplan.get("chunks") if isinstance(shotplan.get("chunks"), list) else []
        prior = sum(_requested_frames(item) for item in chunks[: int(segment_index)] if isinstance(item, dict))
        return float(offset_seconds) + prior / H3_FPS
    slots = shotplan.get("slots") if isinstance(shotplan.get("slots"), list) else []
    slot_index = int(chunk.get("slot_index", segment_index) or 0)
    slot = slots[slot_index] if 0 <= slot_index < len(slots) and isinstance(slots[slot_index], dict) else {}
    explicit = slot.get("source_start_seconds", slot.get("start_seconds", chunk.get("timeline_start_seconds", 0.0)))
    return float(offset_seconds) + _finite_float(explicit, 0.0, 0.0)


def _frame_indices(
    *,
    source_frames: int,
    source_fps: float,
    start_seconds: float,
    requested_frames: int,
    aligned_frames: int,
    end_policy: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if source_frames < 1:
        raise ValueError("MiniMax H3 V2V source video is empty")
    source_fps = _finite_float(source_fps, 24.0, 1.0)
    start_seconds = _finite_float(start_seconds, 0.0, 0.0)
    requested_frames = max(1, int(requested_frames))
    aligned_frames = max(requested_frames, int(aligned_frames))
    positions = start_seconds * source_fps + torch.arange(aligned_frames, dtype=torch.float64) * (source_fps / H3_FPS)
    indices = torch.floor(positions + 0.5).to(dtype=torch.long)
    requested_max = int(indices[requested_frames - 1].item())
    if requested_max >= source_frames:
        available_seconds = source_frames / source_fps
        needed_seconds = start_seconds + requested_frames / H3_FPS
        raise ValueError(
            "MiniMax H3 V2V source is shorter than the requested visible range: "
            f"available={available_seconds:.3f}s, requested_end={needed_seconds:.3f}s, "
            f"source_fps={source_fps:.3f}. Shorten the Shotboard segment or supply a longer source."
        )
    overflow = indices >= source_frames
    overflow_count = int(overflow.sum().item())
    if overflow_count and end_policy == "error":
        raise ValueError(
            "MiniMax H3 V2V aligned 17k+5 conditioning extends beyond the source. "
            "Select hold_last_for_grid to pad only the non-delivered alignment tail."
        )
    # Independent Shotboard source ranges end at the final requested frame.
    # H3's alignment-only tail must hold that boundary even when later source
    # frames physically exist; otherwise chunk N silently consumes the first
    # action frames of chunk N+1. The strict `error` policy intentionally keeps
    # real tail addressing when the caller explicitly asks for it.
    grid_tail_hold_frames = 0
    if end_policy == "hold_last_for_grid" and aligned_frames > requested_frames:
        grid_tail_hold_frames = aligned_frames - requested_frames
        indices[requested_frames:] = indices[requested_frames - 1]
    indices.clamp_(0, source_frames - 1)
    return indices, {
        "first_source_index": int(indices[0].item()),
        "last_source_index": int(indices[-1].item()),
        "grid_tail_hold_frames": grid_tail_hold_frames,
        "source_tail_overflow_frames": overflow_count,
        "source_fps": source_fps,
        "target_fps": H3_FPS,
    }


def _select_frames(video: torch.Tensor, indices: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(video) or video.ndim != 4:
        raise ValueError(f"MiniMax H3 V2V {name} must be IMAGE [frames,height,width,channels]")
    if int(video.shape[0]) < 1:
        raise ValueError(f"MiniMax H3 V2V {name} is empty")
    safe = indices.to(device=video.device)
    return torch.index_select(video, 0, safe)


def _fit_frames(video: torch.Tensor, width: int, height: int, policy: str) -> torch.Tensor:
    if policy == "native_adapt":
        return video
    width = max(32, int(width))
    height = max(32, int(height))
    source_h = int(video.shape[1])
    source_w = int(video.shape[2])
    if source_w == width and source_h == height:
        return video
    tensor = video.movedim(-1, 1)
    if policy == "stretch":
        return F.interpolate(tensor, size=(height, width), mode="bicubic", align_corners=False, antialias=True).movedim(1, -1)
    ratio = min(width / source_w, height / source_h) if policy == "canvas_pad" else max(width / source_w, height / source_h)
    resized_w = max(1, int(round(source_w * ratio)))
    resized_h = max(1, int(round(source_h * ratio)))
    resized = F.interpolate(tensor, size=(resized_h, resized_w), mode="bicubic", align_corners=False, antialias=True)
    if policy == "canvas_crop":
        left = max(0, (resized_w - width) // 2)
        top = max(0, (resized_h - height) // 2)
        return resized[:, :, top : top + height, left : left + width].movedim(1, -1)
    canvas = torch.zeros((int(video.shape[0]), int(video.shape[-1]), height, width), dtype=resized.dtype, device=resized.device)
    left = max(0, (width - resized_w) // 2)
    top = max(0, (height - resized_h) // 2)
    canvas[:, :, top : top + resized_h, left : left + resized_w] = resized
    return canvas.movedim(1, -1)


def _slice_audio(
    audio: Any,
    *,
    start_seconds: float,
    requested_frames: int,
    aligned_frames: int,
) -> dict[str, Any] | None:
    if not isinstance(audio, Mapping) or not torch.is_tensor(audio.get("waveform")):
        return None
    waveform = audio["waveform"]
    if waveform.ndim != 3:
        raise ValueError("MiniMax H3 V2V source_audio waveform must be [batch,channels,samples]")
    sample_rate = max(1, int(audio.get("sample_rate", 32000) or 32000))
    start = max(0, int(round(float(start_seconds) * sample_rate)))
    visible_samples = max(1, int(round(requested_frames / H3_FPS * sample_rate)))
    aligned_samples = max(visible_samples, int(round(aligned_frames / H3_FPS * sample_rate)))
    if start + visible_samples > int(waveform.shape[-1]):
        raise ValueError(
            "MiniMax H3 V2V source audio is shorter than the requested visible segment: "
            f"need samples {start}:{start + visible_samples}, have {int(waveform.shape[-1])}."
        )
    # Only the requested programme range may read real source samples.  The
    # 17k+5-only tail is conditioning padding and must be silence; otherwise
    # the beginning of the next chunk (and possibly its speech) leaks backward
    # into the current REF2VA/custom-audio conditioning window.
    sliced = waveform[..., start : start + visible_samples]
    if aligned_samples > visible_samples:
        sliced = F.pad(sliced, (0, aligned_samples - visible_samples))
    return {
        "waveform": sliced,
        "sample_rate": sample_rate,
        "iamccs_pre_sliced": True,
        "iamccs_source_start_seconds": float(start_seconds),
        "iamccs_requested_frames": int(requested_frames),
        "iamccs_aligned_frames": int(aligned_frames),
        "iamccs_fps": H3_FPS,
    }


def _trim_audio_to_frames(audio: Any, frame_count: int, fps: int = H3_FPS) -> dict[str, Any]:
    if not isinstance(audio, Mapping) or not torch.is_tensor(audio.get("waveform")):
        raise ValueError("MiniMax H3 V2V exact delivery expects an AUDIO dictionary")
    sample_rate = max(1, int(audio.get("sample_rate", 32000) or 32000))
    samples = max(1, int(round(int(frame_count) / max(1, int(fps)) * sample_rate)))
    waveform = audio["waveform"]
    trimmed = waveform[..., :samples]
    if int(trimmed.shape[-1]) < samples:
        trimmed = F.pad(trimmed, (0, samples - int(trimmed.shape[-1])))
    return {"waveform": trimmed, "sample_rate": sample_rate}


def _reference_payload(
    *,
    reference_images: list[Any],
    reference_roles: list[str],
    raw_video: torch.Tensor,
    depth_video: torch.Tensor | None,
    pose_video: torch.Tensor | None,
    segment_audio: dict[str, Any] | None,
    guide_mode: str,
    audio_pairing: str,
) -> tuple[dict[str, torch.Tensor] | None, dict[str, torch.Tensor], dict[str, dict[str, Any]] | None, dict[str, dict[str, Any]] | None, list[dict[str, Any]]]:
    images: dict[str, torch.Tensor] = {}
    items: list[dict[str, Any]] = []
    for source_slot, (image, role) in enumerate(zip(reference_images, reference_roles), start=1):
        role = str(role or "subject_identity").strip().lower()
        if role == "disabled" or not torch.is_tensor(image):
            continue
        ordinal = len(images)
        images[f"ref_image_{ordinal}"] = image[:1]
        items.append({
            "label": f"<Picture {ordinal + 1}>",
            "kind": "image",
            "role": role,
            "source_slot": source_slot,
            "ordinal": ordinal + 1,
        })

    videos: dict[str, torch.Tensor] = {"ref_video_0": raw_video}
    video_audios: dict[str, dict[str, Any]] | None = None
    standalone_audios: dict[str, dict[str, Any]] | None = None
    if segment_audio is not None and audio_pairing == "pair_with_source_video":
        video_audios = {"ref_video_audio_0": segment_audio}
        items.append({"label": "<Audio 1>", "kind": "audio", "role": "source_video_soundtrack", "paired_video": "<Video 1>"})
    items.append({"label": "<Video 1>", "kind": "video", "role": "raw_motion_camera_environment"})

    next_video = 1
    if guide_mode in {"raw_depth", "raw_depth_pose"} and torch.is_tensor(depth_video):
        videos[f"ref_video_{next_video}"] = depth_video
        items.append({"label": f"<Video {next_video + 1}>", "kind": "video", "role": "depth_guide"})
        next_video += 1
    if guide_mode in {"raw_pose", "raw_depth_pose"} and torch.is_tensor(pose_video):
        videos[f"ref_video_{next_video}"] = pose_video
        items.append({"label": f"<Video {next_video + 1}>", "kind": "video", "role": "pose_guide"})

    if segment_audio is not None and audio_pairing == "standalone_reference":
        standalone_audios = {"ref_audio_0": segment_audio}
        items.append({"label": "<Audio 1>", "kind": "audio", "role": "standalone_source_audio_reference"})
    return images or None, videos, video_audios, standalone_audios, items


def _reference_header(items: list[dict[str, Any]], prompt: str) -> str:
    mapping = "; ".join(f"{item['label']}={item['role']}" for item in items)
    return f"[IAMCCS H3 V2V REFERENCE MAP] {mapping}\n\n{str(prompt or '').strip()}".strip()


class IAMCCS_CineInfoH3V2V:
    """Attach one source video and V2V references to MiniMax CineLinX."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "source_video": ("IMAGE",),
                "source_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001}),
                "guide_mode": (GUIDE_MODE_OVERRIDES, {"default": "from_shotboard"}),
                "source_range_policy": (SOURCE_RANGE_OVERRIDES, {"default": "from_shotboard"}),
                "source_offset_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.01}),
                "source_fit": (SOURCE_FIT_OVERRIDES, {"default": "from_shotboard"}),
                "source_end_policy": (SOURCE_END_OVERRIDES, {"default": "from_shotboard"}),
                "audio_pairing": (AUDIO_PAIRING_OVERRIDES, {"default": "from_shotboard"}),
                "ref_image_size": (REF_IMAGE_SIZE_OVERRIDES, {"default": "from_shotboard"}),
                "reference_role_1": (IMAGE_ROLES, {"default": "subject_identity"}),
                "reference_role_2": (IMAGE_ROLES, {"default": "wardrobe_object"}),
                "reference_role_3": (IMAGE_ROLES, {"default": "environment"}),
                "reference_role_4": (IMAGE_ROLES, {"default": "style"}),
            },
            "optional": {
                "source_audio": ("AUDIO",),
                "reference_image_1": ("IMAGE",),
                "reference_image_2": ("IMAGE",),
                "reference_image_3": ("IMAGE",),
                "reference_image_4": ("IMAGE",),
                "depth_guide": ("IMAGE", {"tooltip": "Preprocess upstream with Depth Anything V2 when the selected guide mode requires depth."}),
                "pose_guide": ("IMAGE", {"tooltip": "Preprocess upstream with DWPose when the selected guide mode requires pose."}),
            },
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE,)
    RETURN_NAMES = ("cine_linx",)
    FUNCTION = "attach"
    CATEGORY = CATEGORY

    def attach(
        self,
        cine_linx,
        source_video,
        source_fps,
        guide_mode,
        source_range_policy,
        source_offset_seconds,
        source_fit,
        source_end_policy,
        audio_pairing,
        ref_image_size,
        reference_role_1,
        reference_role_2,
        reference_role_3,
        reference_role_4,
        source_audio=None,
        reference_image_1=None,
        reference_image_2=None,
        reference_image_3=None,
        reference_image_4=None,
        depth_guide=None,
        pose_guide=None,
    ):
        if not isinstance(cine_linx, dict):
            raise ValueError("IAMCCS Cine Info H3 V2V requires a valid cine_linx input")
        if not torch.is_tensor(source_video) or source_video.ndim != 4 or int(source_video.shape[0]) < 1:
            raise ValueError("IAMCCS Cine Info H3 V2V requires a non-empty source_video IMAGE batch")
        shotplan = _resolve_shotplan(cine_linx)
        attached = {
            "guide_mode": guide_mode,
            "source_range_policy": source_range_policy,
            "source_offset_seconds": source_offset_seconds,
            "source_fit": source_fit,
            "source_end_policy": source_end_policy,
            "audio_pairing": audio_pairing,
            "source_fps": source_fps,
            "ref_image_size": ref_image_size,
        }
        resolved = _resolve_v2v_config(shotplan, attached)
        source_audio = (
            None
            if resolved["audio_behavior"] == AUDIO_BEHAVIOR_NATIVE
            else _materialize_optional_audio(source_audio)
        )
        _validate_guides(resolved, depth_guide, pose_guide)
        references = [reference_image_1, reference_image_2, reference_image_3, reference_image_4]
        roles = [reference_role_1, reference_role_2, reference_role_3, reference_role_4]
        picture_items = []
        for source_slot, (image, role) in enumerate(zip(references, roles), start=1):
            resolved_role = str(role or "subject_identity").strip().lower()
            if not torch.is_tensor(image) or resolved_role == "disabled":
                continue
            ordinal = len(picture_items) + 1
            picture_items.append({
                "source_slot": source_slot,
                "ordinal": ordinal,
                "label": f"<Picture {ordinal}>",
                "connected": True,
                "shape": _shape(image),
                "role": resolved_role,
            })
        manifest = {
            "schema": "iamccs.minimax_h3.v2v",
            "schema_version": 1,
            "task": "v2va_object_swap",
            "model_family": "ref2va",
            "source": {
                "label": "<Video 1>",
                "shape": _shape(source_video),
                "fps": resolved["source_fps"],
                "role": "raw_motion_camera_environment",
                "audio": _audio_meta(source_audio),
            },
            "pictures": picture_items,
            "guides": {
                "mode": resolved["guide_mode"],
                "depth_connected": bool(torch.is_tensor(depth_guide)),
                "pose_connected": bool(torch.is_tensor(pose_guide)),
                "policy": "explicit_preprocessed_inputs_no_hidden_wrappers",
            },
            "range": {
                "policy": resolved["source_range_policy"],
                "offset_seconds": resolved["source_offset_seconds"],
            },
            "source_fit": resolved["source_fit"],
            "source_end_policy": resolved["source_end_policy"],
            "audio_pairing": resolved["audio_pairing"],
            "requested_audio_pairing": resolved["requested_audio_pairing"],
            "audio_behavior": resolved["audio_behavior"],
            "ref_image_size": resolved["ref_image_size"],
        }
        config = {**resolved, "reference_roles": [str(role) for role in roles], "model_family": "ref2va", "task": "v2va_object_swap"}
        base = _clean_previous_v2v(cine_linx)
        resources = {
            CONFIG_RESOURCE: config,
            MANIFEST_RESOURCE: manifest,
            f"{RESOURCE_PREFIX}source_video": source_video,
            f"{RESOURCE_PREFIX}reference_image_1": reference_image_1,
            f"{RESOURCE_PREFIX}reference_image_2": reference_image_2,
            f"{RESOURCE_PREFIX}reference_image_3": reference_image_3,
            f"{RESOURCE_PREFIX}reference_image_4": reference_image_4,
            f"{RESOURCE_PREFIX}depth_guide": depth_guide,
            f"{RESOURCE_PREFIX}pose_guide": pose_guide,
        }
        if source_audio is not None:
            resources[f"{RESOURCE_PREFIX}source_audio"] = source_audio
        active_pictures = len(picture_items)
        report = (
            "IAMCCS CineInfo H3 V2V R22 | model=REF2VA | raw_source=1 | "
            f"pictures={active_pictures}/4 | guide={resolved['guide_mode']} | "
            f"range={resolved['source_range_policy']}+{resolved['source_offset_seconds']:.3f}s | "
            f"source={resolved['source_fps']:.3f}->24fps/{resolved['source_fit']} | "
            f"audio={resolved['audio_behavior']}/{resolved['audio_pairing']}"
        )
        output = build_stage_linx_payload(
            base,
            stage_name=STAGE_NAME,
            stage_kind="minimax_h3_v2v_reference_transport",
            payload=config,
            report=report,
            slot_map={"cine_linx": "MiniMax H3 V2V conditioning"},
            downstream_stages=("IAMCCS MiniMax H3 V2V Conditioning R22",),
            policies={
                "shotboard_owns": "prompt_duration_source_ranges",
                "source_media_location": "cine_info_h3_v2v_not_timeline",
                "raw_video_occurs_once": True,
                "optional_guides_are_explicit": True,
                "model_family": "ref2va",
            },
            outputs={"manifest_json": json.dumps(manifest, ensure_ascii=False, indent=2), "report": report},
            resources=resources,
            requires={"resources": ["iamccs_minimax_h3_shotplan"]},
        )
        return (output,)


class IAMCCS_MiniMaxH3V2VConditioningR22:
    """Slice/resample source media and build deterministic REF2VA conditioning."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "video_vae": ("VAE",),
                "audio_vae": ("VAE",),
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
            },
            "optional": {
                "prompt_override": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = (
        "MODEL", "CONDITIONING", "LATENT", "IMAGE", "AUDIO", "STRING", "STRING",
        "INT", "INT", "INT", "INT", "INT", "STRING",
    )
    RETURN_NAMES = (
        "model", "positive", "latent", "source_preview", "source_audio_segment", "manifest_json", "prompt",
        "current_segment", "total_segments", "requested_frames", "aligned_frames", "trim_tail_frames", "report",
    )
    FUNCTION = "prepare"
    CATEGORY = CATEGORY

    def prepare(self, model, clip, video_vae, audio_vae, cine_linx, segment_index, prompt_override=""):
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo

        shotplan = _resolve_shotplan(cine_linx)
        chunk = _shotplan_chunk(shotplan, segment_index)
        resources = _resources(cine_linx)
        attached = resources.get(CONFIG_RESOURCE)
        if not isinstance(attached, dict):
            raise ValueError("MiniMax H3 V2V conditioning requires IAMCCS CineInfoH3V2V upstream")
        config = _resolve_v2v_config(shotplan, attached)
        if str(attached.get("model_family", "ref2va")) != "ref2va":
            raise ValueError("MiniMax H3 V2V must route the REF2VA model family")
        source_video = resources.get(f"{RESOURCE_PREFIX}source_video")
        source_audio = resources.get(f"{RESOURCE_PREFIX}source_audio")
        depth_guide = resources.get(f"{RESOURCE_PREFIX}depth_guide")
        pose_guide = resources.get(f"{RESOURCE_PREFIX}pose_guide")
        _validate_guides(config, depth_guide, pose_guide)
        if not torch.is_tensor(source_video):
            raise ValueError("MiniMax H3 V2V source_video resource is missing")

        requested = _requested_frames(chunk)
        aligned = int(align_h3_frames(requested))
        source_start = _segment_source_start(
            shotplan,
            chunk,
            int(segment_index),
            config["source_range_policy"],
            config["source_offset_seconds"],
        )
        indices, index_report = _frame_indices(
            source_frames=int(source_video.shape[0]),
            source_fps=config["source_fps"],
            start_seconds=source_start,
            requested_frames=requested,
            aligned_frames=aligned,
            end_policy=config["source_end_policy"],
        )
        width = int(shotplan.get("width", 1344))
        height = int(shotplan.get("height", 768))
        raw = _fit_frames(_select_frames(source_video, indices, "source_video"), width, height, config["source_fit"])

        depth = None
        if config["guide_mode"] in {"raw_depth", "raw_depth_pose"}:
            if int(depth_guide.shape[0]) != int(source_video.shape[0]):
                raise ValueError("MiniMax H3 V2V depth_guide frame count must match source_video before segment slicing")
            depth = _fit_frames(_select_frames(depth_guide, indices, "depth_guide"), width, height, config["source_fit"])
        pose = None
        if config["guide_mode"] in {"raw_pose", "raw_depth_pose"}:
            if int(pose_guide.shape[0]) != int(source_video.shape[0]):
                raise ValueError("MiniMax H3 V2V pose_guide frame count must match source_video before segment slicing")
            pose = _fit_frames(_select_frames(pose_guide, indices, "pose_guide"), width, height, config["source_fit"])

        segment_audio = None
        if config["audio_behavior"] != AUDIO_BEHAVIOR_NATIVE and source_audio is not None:
            segment_audio = _slice_audio(
                source_audio,
                start_seconds=source_start,
                requested_frames=requested,
                aligned_frames=aligned,
            )
        references = [resources.get(f"{RESOURCE_PREFIX}reference_image_{index}") for index in range(1, 5)]
        roles = list(attached.get("reference_roles") or [])[:4]
        while len(roles) < 4:
            roles.append(("subject_identity", "wardrobe_object", "environment", "style")[len(roles)])
        ref_images, ref_videos, ref_video_audios, ref_audios, items = _reference_payload(
            reference_images=references,
            reference_roles=roles,
            raw_video=raw,
            depth_video=depth,
            pose_video=pose,
            segment_audio=segment_audio,
            guide_mode=config["guide_mode"],
            audio_pairing=config["audio_pairing"],
        )
        creative_prompt = str(prompt_override or "").strip() or str(chunk.get("prompt", "")).strip()
        prompt = _reference_header(items, creative_prompt)
        result, text_encoder_report = _run_h3_conditioning_with_cpu_fallback(
            clip,
            shotplan,
            lambda active_clip: MiniMaxH3ReferenceToVideo.execute(
                clip=active_clip,
                vae=video_vae,
                audio_vae=audio_vae,
                prompt=prompt,
                width=width,
                height=height,
                length=aligned,
                ref_image_size=config["ref_image_size"],
                ref_images=ref_images,
                ref_videos=ref_videos,
                ref_video_audios=ref_video_audios,
                ref_audios=ref_audios,
            ),
        )
        positive, latent = result[0], result[1]
        manifest = {
            "schema": "iamccs.minimax_h3.v2v.segment",
            "schema_version": 1,
            "task": "v2va_object_swap",
            "model_family": "ref2va",
            "segment_index": int(segment_index),
            "source_start_seconds": source_start,
            "requested_frames": requested,
            "aligned_frames": aligned,
            "delivery_trim_tail_frames": aligned - requested,
            "source_resample": index_report,
            "guide_mode": config["guide_mode"],
            "audio_behavior": config["audio_behavior"],
            "audio_pairing": config["audio_pairing"],
            "requested_audio_pairing": config["requested_audio_pairing"],
            "items": items,
            "raw_video_occurrences": sum(1 for item in items if item.get("role") == "raw_motion_camera_environment"),
        }
        if manifest["raw_video_occurrences"] != 1:
            raise RuntimeError("MiniMax H3 V2V internal contract violation: raw source must occur exactly once")
        report = (
            f"MiniMax H3 V2V R22 | segment={int(segment_index) + 1}/{len(shotplan['chunks'])} | "
            f"model=REF2VA | source={source_start:.3f}s/{config['source_fps']:.3f}->24fps | "
            f"frames={requested}->{aligned}->trim {aligned - requested} | guide={config['guide_mode']} | "
            f"refs={len(items)} | audio={config['audio_behavior']}/{config['audio_pairing']} | "
            f"tail_hold={index_report['grid_tail_hold_frames']} | "
            f"text_encoder={text_encoder_report}"
        )
        return (
            model,
            positive,
            latent,
            raw[:requested],
            segment_audio,
            json.dumps(manifest, ensure_ascii=False, indent=2),
            prompt,
            int(segment_index),
            int(len(shotplan["chunks"])),
            requested,
            aligned,
            aligned - requested,
            report,
        )


class IAMCCS_MiniMaxH3V2VExactDeliveryR22:
    """Remove only the 17k+5 alignment tail after H3 decode."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "images", "audio", "delivery_frames", "trimmed_tail_frames",
        "checkpoint_trim_head_frames", "report",
    )
    FUNCTION = "trim"
    CATEGORY = CATEGORY

    def trim(self, images, audio, cine_linx, segment_index):
        if not torch.is_tensor(images) or images.ndim != 4 or int(images.shape[0]) < 1:
            raise ValueError("MiniMax H3 V2V exact delivery expects non-empty IMAGE frames")
        shotplan = _resolve_shotplan(cine_linx)
        chunk = _shotplan_chunk(shotplan, segment_index)
        requested = _requested_frames(chunk)
        if int(images.shape[0]) < requested:
            raise ValueError(
                f"MiniMax H3 V2V decode returned {int(images.shape[0])} frames, below requested delivery {requested}"
            )
        trimmed_tail = max(0, int(images.shape[0]) - requested)
        output_images = images[:requested]
        output_audio = _trim_audio_to_frames(audio, requested, H3_FPS)
        report = (
            f"MiniMax H3 V2V exact delivery | segment={int(segment_index) + 1} | "
            f"decoded={int(images.shape[0])} | delivered={requested} | trimmed_tail={trimmed_tail} | fps={H3_FPS}"
        )
        # V2VA segments are independent REF2VA renders. Their edit is a hard
        # cut, so NativeCheckpoint must not remove an additional head frame.
        return output_images, output_audio, requested, trimmed_tail, 0, report


NODE_CLASS_MAPPINGS = {
    "IAMCCS_CineInfoH3V2V": IAMCCS_CineInfoH3V2V,
    "IAMCCS_MiniMaxH3V2VConditioningR22": IAMCCS_MiniMaxH3V2VConditioningR22,
    "IAMCCS_MiniMaxH3V2VExactDeliveryR22": IAMCCS_MiniMaxH3V2VExactDeliveryR22,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_CineInfoH3V2V": "IAMCCS Cine Info H3 V2V - Source + Pictures",
    "IAMCCS_MiniMaxH3V2VConditioningR22": "MiniMax H3 V2V Object Swap Conditioning (R22)",
    "IAMCCS_MiniMaxH3V2VExactDeliveryR22": "MiniMax H3 V2V Exact Delivery Trim (R22)",
}
