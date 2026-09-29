# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later

"""IAMCCS MiniMax H3 Extended AV continuation primitives.

This module is intentionally isolated from the stable R42/R43 paths.  It is
used only when a Shotboard plan explicitly selects ``fl2va_extended_av``.

Technical lineage
-----------------
The masked-prefix, shared AV-grid, same-time overlap and lineage ideas are
implemented as the IAMCCS EXTEND-style masked RAW AV engine. Upstream technical
lineage and license attribution are preserved in ``THIRD_PARTY_NOTICES.md``. IAMCCS
keeps the implementation local to its Shotboard/Atomic backend and adds:

* VRAM-oriented legal run/overlap profiles;
* fail-closed MiniMax H3 mask capability checks;
* full raw AV sidecars for later joint-timeline refinement;
* IAMCCS Shotboard lineage metadata and seam diagnostics.

The important invariant is that an overlap is *the same moment in time* on
both sides of a join.  It is never a dissolve between A(t) and B(t+dt).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch



EXTENDED_AV_MODE = "fl2va_extended_av"
EXTENDED_AV_SCHEMA = "iamccs.minimax_h3.extended_av.v2"
EXTENDED_AV_SIDECAR_SCHEMA = "iamccs.h3.extended_av.sidecar.v2"
EXTENDED_AV_SIDECAR_SUFFIX = ".extended_av.safetensors"

H3_FPS = 24
FRAME_PATTERN = (1, 4, 4, 4, 4)
SHARED_AV_GRID_BASE = 39
SHARED_AV_GRID_STEP = 51
H3_FRAME_BASE = 5
H3_FRAME_STEP = 17
AUDIO_LATENT_FPS = 40
DEFAULT_MASKED_AUDIO_RELEASE_TICKS = 8
DEFAULT_BOUNDARY_RUNWAY_FRAMES = 34

# Every full technical run is also on the shared video/audio grid.  This is
# stricter than merely being a legal H3 17k+5 run and avoids the ~fractional
# 40-Hz audio boundary described by the upstream implementation.
PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "safe_8_12gb": {
        "sample_window_frames": 192,
        "overlap_frames": 39,
        "joint_window_frames": 192,
        "joint_window_overlap": 39,
    },
    "balanced_12_16gb": {
        "sample_window_frames": 243,
        "overlap_frames": 39,
        "joint_window_frames": 243,
        "joint_window_overlap": 39,
    },
    "quality_20_24gb": {
        "sample_window_frames": 294,
        "overlap_frames": 90,
        "joint_window_frames": 294,
        "joint_window_overlap": 90,
    },
    "max_32gb_plus": {
        "sample_window_frames": 345,
        "overlap_frames": 141,
        "joint_window_frames": 345,
        "joint_window_overlap": 90,
    },
}


def shared_av_window_ok(frames: int) -> bool:
    frames = int(frames)
    return frames >= SHARED_AV_GRID_BASE and (frames - SHARED_AV_GRID_BASE) % SHARED_AV_GRID_STEP == 0


def shared_av_snap_down(frames: int) -> int | None:
    frames = int(frames)
    if frames < SHARED_AV_GRID_BASE:
        return None
    return SHARED_AV_GRID_BASE + ((frames - SHARED_AV_GRID_BASE) // SHARED_AV_GRID_STEP) * SHARED_AV_GRID_STEP


def shared_av_snap_up(frames: int) -> int:
    frames = max(SHARED_AV_GRID_BASE, int(frames))
    down = shared_av_snap_down(frames)
    assert down is not None
    return down if down == frames else down + SHARED_AV_GRID_STEP




def h3_run_ok(frames: int) -> bool:
    frames = int(frames)
    return frames >= H3_FRAME_BASE and (frames - H3_FRAME_BASE) % H3_FRAME_STEP == 0


def h3_snap_run_up(frames: int) -> int:
    """Smallest legal H3 17k+5 run that covers ``frames``."""
    frames = max(H3_FRAME_BASE, int(frames))
    rem = (frames - H3_FRAME_BASE) % H3_FRAME_STEP
    return frames if rem == 0 else frames + (H3_FRAME_STEP - rem)


def audio_total(frame_count: int) -> int:
    """Cumulative H3 audio ticks. Differences of totals avoid chain drift."""
    return int(round(int(frame_count) / float(H3_FPS) * AUDIO_LATENT_FPS))


def _pixel_starts(latent_steps: int) -> list[int]:
    starts: list[int] = []
    cursor = 0
    for index in range(max(0, int(latent_steps))):
        starts.append(cursor)
        cursor += FRAME_PATTERN[index % len(FRAME_PATTERN)]
    return starts


def frames_for_steps(latent_steps: int) -> int:
    return sum(FRAME_PATTERN[index % len(FRAME_PATTERN)] for index in range(max(0, int(latent_steps))))


def video_tokens_for_frames(frame_count: int) -> int:
    frame_count = int(frame_count)
    if frame_count < 1 or frame_count % 17 != 5:
        raise ValueError(f"Extended AV frame count must be on the H3 17k+5 grid, got {frame_count}.")
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def audio_ticks_for_frames(frame_count: int, *, require_exact: bool = False) -> int:
    value = float(frame_count) * 40.0 / float(H3_FPS)
    rounded = int(round(value))
    if require_exact and abs(value - rounded) > 1e-7:
        raise ValueError(
            f"Extended AV frame count {frame_count} does not land on an integer 40-Hz audio tick."
        )
    return max(1, rounded)


def core_masks_available() -> bool:
    """Whether current ComfyUI exposes native per-stream H3 denoise masks."""
    try:
        import comfy.ldm.minimax.model as mm
        import comfy.model_base as mb
    except Exception:
        return False
    return hasattr(mm, "mask_row_values") and "scale_latent_inpaint" in vars(mb.MiniMaxH3)


def resolve_contract(
    *,
    profile: str = "safe_8_12gb",
    pin_mode: str = "masked",
    mask_profile: str = "exact",
    custom_window_frames: int = 192,
    custom_overlap_frames: int = 39,
    handover_mode: str = "auto",
    level_lock_mode: str = "auto",
    joint_refine_mode: str = "off",
    joint_window_frames: int = 192,
    joint_window_overlap: int = 39,
    soft_audio_handover_ms: float = 15.0,
    boundary_polish_ms: float = 3.0,
    boundary_polish_strength: float = 1.0,
    boundary_runway_frames: int = DEFAULT_BOUNDARY_RUNWAY_FRAMES,
) -> dict[str, Any]:
    """Return the canonical, validated Extended-AV contract.

    Extended AV is an EXTEND-style *extend* role. The production path is
    therefore ``masked``: the previous take's delivered tail is written into
    the head of a fresh target AV latent and protected by native H3 masks.
    ``both`` remains loadable only for old experimental contracts but is not
    selected by the dedicated extend planner.
    """
    profile = str(profile or "safe_8_12gb").strip().lower()
    if profile == "custom":
        sample_window = int(custom_window_frames)
        overlap = int(custom_overlap_frames)
    else:
        if profile not in PROFILE_PRESETS:
            profile = "safe_8_12gb"
        sample_window = int(PROFILE_PRESETS[profile]["sample_window_frames"])
        overlap = int(PROFILE_PRESETS[profile]["overlap_frames"])
        joint_window_frames = int(PROFILE_PRESETS[profile]["joint_window_frames"])
        joint_window_overlap = int(PROFILE_PRESETS[profile]["joint_window_overlap"])

    if not shared_av_window_ok(sample_window):
        raise ValueError(
            "FL2VA Extended AV sample window must be on the shared H3 AV grid "
            f"39+51k (39, 90, 141, 192, 243, 294, 345), got {sample_window}."
        )
    if sample_window > 362:
        raise ValueError("FL2VA Extended AV sample window cannot exceed H3's 362-frame native limit.")
    if not shared_av_window_ok(overlap):
        raise ValueError(
            "FL2VA Extended AV overlap must be on the shared H3 AV grid "
            f"39+51k (39, 90, 141, ...), got {overlap}."
        )
    if overlap >= sample_window:
        raise ValueError("FL2VA Extended AV overlap must be smaller than the technical sample window.")

    # Universal endpoint protection. This is temporal geometry, not a VRAM
    # quality preset: every technical sample reserves a hidden suffix after the
    # editorial endpoint so H3 can exhaust end-of-window composition drift
    # outside the delivered movie. 34f = two native 17-frame H3 steps.
    requested_boundary_runway = max(0, int(boundary_runway_frames or 0))
    if requested_boundary_runway % H3_FRAME_STEP != 0:
        raise ValueError(
            "FL2VA Extended AV boundary runway must be a multiple of 17 frames, "
            f"got {requested_boundary_runway}."
        )
    continuation_capacity_before_runway = sample_window - overlap
    max_runway = max(0, continuation_capacity_before_runway - H3_FRAME_STEP)
    max_runway -= max_runway % H3_FRAME_STEP
    resolved_boundary_runway = min(requested_boundary_runway, max_runway)
    if requested_boundary_runway and resolved_boundary_runway < H3_FRAME_STEP:
        raise ValueError(
            "FL2VA Extended AV sample/overlap geometry leaves no room for both "
            "a hidden boundary runway and a visible continuation interval."
        )

    requested_pin_mode = str(pin_mode or "masked").strip().lower()
    if requested_pin_mode not in {"masked", "both"}:
        raise ValueError("FL2VA Extended AV accepts legacy pin_mode=masked|both, but EXTEND executes masked.")
    # Upstream role rule: an EXTEND is MASKED.  ``both`` belongs to the
    # arriving/prepend side where keyframe rows help steer into a destination;
    # it is retained only as a load-compatible legacy widget value and never
    # changes the dedicated long-take execution path.
    pin_mode = "masked"
    mask_profile = str(mask_profile or "exact").strip().lower()
    if mask_profile not in {"exact", "runway", "soft"}:
        raise ValueError("FL2VA Extended AV mask profile must be exact, runway or soft.")
    handover_mode = str(handover_mode or "auto").strip().lower()
    if handover_mode not in {"auto", "same_time_equal_power", "direct"}:
        raise ValueError("FL2VA Extended AV handover must be auto, same_time_equal_power or direct.")
    level_lock_mode = str(level_lock_mode or "auto").strip().lower()
    if level_lock_mode not in {"auto", "off"}:
        raise ValueError(
            "FL2VA Extended AV Phase 1 exposes level_lock=auto|off only. "
            "The external project's post-overlap global Level Lock is intentionally "
            "not selectable until its runtime path is wired and smoke-tested."
        )
    joint_refine_mode = str(joint_refine_mode or "off").strip().lower()
    if joint_refine_mode != "off":
        raise ValueError(
            "FL2VA Extended AV Phase 1 keeps Joint Refine disabled. The shared-AV "
            "window planner is implemented for the later workflow integration, "
            "but no setting may claim to run it before that sampler is wired."
        )

    # Validate the reserved future geometry even while the execution switch is
    # off. This makes saved contracts forward-compatible without presenting a
    # non-functional feature to current users.
    if not shared_av_window_ok(int(joint_window_frames)):
        raise ValueError("Extended AV joint window must be on the shared 39+51k AV grid.")
    if not shared_av_window_ok(int(joint_window_overlap)):
        raise ValueError("Extended AV joint overlap must be on the shared 39+51k AV grid.")
    if int(joint_window_overlap) >= int(joint_window_frames):
        raise ValueError("Extended AV joint overlap must be smaller than its joint window.")

    # Match the masked BEFORE-pin EXTEND semantics. Video may be hard/soft held.
    # Pinned audio itself is exact, but its FINAL ticks may release toward the
    # generated region; that transition lives inside scaffolding which is
    # trimmed from the delivered child.
    soft_audio_handover_ms = max(5.0, min(60.0, float(soft_audio_handover_ms)))
    boundary_polish_ms = max(0.0, min(12.0, float(boundary_polish_ms)))
    boundary_polish_strength = max(0.0, min(1.0, float(boundary_polish_strength)))

    shape = {
        "exact": {"video_ramp_frames": 0, "video_edge": 0.0, "video_hold": 0.0},
        "runway": {"video_ramp_frames": 17, "video_edge": 0.60, "video_hold": 0.0},
        "soft": {"video_ramp_frames": 0, "video_edge": 0.30, "video_hold": 0.30},
    }[mask_profile]

    return {
        "schema": EXTENDED_AV_SCHEMA,
        "enabled": True,
        "mode": EXTENDED_AV_MODE,
        "experimental": True,
        "profile": profile,
        "sample_window_frames": sample_window,
        "overlap_frames": overlap,
        "boundary_runway_mode": "auto_universal",
        "boundary_runway_frames": int(resolved_boundary_runway),
        "boundary_runway_requested_frames": int(requested_boundary_runway),
        "boundary_runway_policy": "hidden_tail_trimmed_never_inherited",
        "root_visible_capacity_frames": sample_window - int(resolved_boundary_runway),
        "legacy_unique_capacity_frames": sample_window - overlap,
        "unique_capacity_frames": sample_window - overlap - int(resolved_boundary_runway),
        "pin_mode": pin_mode,
        "legacy_requested_pin_mode": requested_pin_mode,
        "mask_profile": mask_profile,
        **shape,
        "audio_mask": "pinned_exact_with_internal_release",
        "masked_audio_release_ticks": DEFAULT_MASKED_AUDIO_RELEASE_TICKS,
        # Phase 9.2 delivery: reuse the child hidden audio head as a
        # time-corresponding Soft-AV handover source for the outgoing parent.
        # This is audio-only; the MASKED+EXACT video junction remains untouched.
        "soft_audio_handover_ms": soft_audio_handover_ms,
        "boundary_polish_ms": boundary_polish_ms,
        "boundary_polish_strength": boundary_polish_strength,
        "audio_master_policy": "hidden_context_soft_av_boundary_polish",
        "audio_grid_hz": 40,
        "role": "extend",
        "delivery_policy": "trim_pinned_head_then_butt_join",
        "seam_repair_default": "off_for_masked_exact",
        "handover_mode": "masked_trim",
        "legacy_requested_handover_mode": handover_mode,
        "pinned_same_time_context": True,
        "decoded_overlap": False,
        # Hard masked EXTEND has no decoded seam repair by default. Keep the
        # old widget value only as migration metadata; it no longer selects a
        # pixel handover in this dedicated branch.
        "level_lock_mode": "off",
        "legacy_requested_level_lock_mode": level_lock_mode,
        "level_lock_frames": 12,
        "level_lock_runtime": "deferred_external_style_repair",
        "joint_refine_mode": joint_refine_mode,
        "joint_refine_runtime": "planned_not_wired_phase1",
        "joint_window_frames": int(joint_window_frames),
        "joint_window_overlap": int(joint_window_overlap),
        "raw_av_sidecars": True,
        "lineage": True,
        "legacy_modes_untouched": True,
        "source_lineage": "iamccs_extend_style_masked_raw_av_lineage",
    }



def resolved_handover_mode(contract: dict[str, Any]) -> str:
    """Compatibility label for UI/logging; assembly no longer blends exact extends.

    EXTEND-style masked EXTEND semantics trim the pinned head from the child and butt
    join the delivered child after the parent. There is no decoded overlap to
    crossfade. Soft profiles may deliver part of their ramp via the handover
    geometry, but ownership still comes from the pin recipe, not a pixel blend.
    """
    requested = str((contract or {}).get("handover_mode", "auto") or "auto").strip().lower()
    if requested not in {"auto", "same_time_equal_power", "direct", "masked_trim"}:
        raise ValueError(f"Unknown Extended AV handover mode: {requested}")
    return "masked_trim"


def handover_frames(overlap_frames: int, contract: dict[str, Any]) -> int:
    """masked BEFORE-pin handover offset inside the preserved prefix."""
    covered = int(overlap_frames)
    prof = _video_mask_values(covered, contract, device=torch.device("cpu"), dtype=torch.float32).tolist()
    held = [i for i, value in enumerate(prof) if float(value) == 0.0]
    if not held:
        return covered
    starts = _pixel_starts(len(prof))
    ends = starts[1:] + [covered]
    return int(ends[held[-1]])


def _streams(latent: Any, label: str) -> tuple[torch.Tensor, torch.Tensor]:
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if hasattr(samples, "unbind"):
        items = list(samples.unbind())
    elif hasattr(samples, "tensors"):
        items = list(samples.tensors)
    elif isinstance(samples, (tuple, list)):
        items = list(samples)
    else:
        items = []
    if len(items) < 2:
        raise ValueError(f"{label} is not a MiniMax H3 AV latent.")
    video, audio = items[:2]
    if not torch.is_tensor(video) or video.ndim != 5:
        raise ValueError(f"{label} has an invalid H3 video latent.")
    if not torch.is_tensor(audio) or audio.ndim != 4:
        raise ValueError(f"{label} has an invalid H3 audio latent.")
    return video, audio


def _video_mask_values(overlap_frames: int, contract: dict[str, Any], *, device, dtype) -> torch.Tensor:
    steps = video_tokens_for_frames(overlap_frames)
    deep = float(contract.get("video_hold", 0.0))
    edge = float(contract.get("video_edge", deep))
    ramp_frames = max(0, int(contract.get("video_ramp_frames", 0)))
    if ramp_frames <= 0 or abs(edge - deep) < 1e-9:
        return torch.full((steps,), deep, dtype=dtype, device=device)

    starts = _pixel_starts(steps)
    ends = starts[1:] + [int(overlap_frames)]
    # Extend/before-pin: join is at the END of the pinned prefix.
    distances = [int(overlap_frames) - int(end) for end in ends]
    values = [
        edge + (deep - edge) * min(1.0, max(0.0, float(distance) / float(ramp_frames)))
        for distance in distances
    ]
    return torch.tensor(values, dtype=dtype, device=device).clamp_(0.0, 1.0)


def _sidecar_folder() -> Path:
    import folder_paths
    return Path(folder_paths.get_output_directory()) / "minimax_h3_shotboard" / "extended_av"


def sidecar_path(render_id: str, segment_index: int) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(render_id or "").strip())
    safe = safe or "minimax_h3_render"
    return _sidecar_folder() / f"{safe}_seg_{int(segment_index):04d}{EXTENDED_AV_SIDECAR_SUFFIX}"


def _identity(render_id: str, segment_index: int, video: torch.Tensor, audio: torch.Tensor) -> str:
    # Avoid hashing gigabytes of tensor bytes. Geometry + a few deterministic
    # boundary statistics are sufficient for an internal lineage identity;
    # the path still remains authoritative for loading.
    payload = {
        "render_id": str(render_id),
        "segment_index": int(segment_index),
        "video_shape": list(video.shape),
        "audio_shape": list(audio.shape),
        "video_mean": round(float(video.detach().float().mean().cpu()), 7),
        "audio_mean": round(float(audio.detach().float().mean().cpu()), 7),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def save_sidecar(
    render_id: str,
    segment_index: int,
    sampled_latent: dict[str, Any],
    *,
    contract: dict[str, Any],
    delivered_frames: int,
    pinned_head_frames: int = 0,
    padding_tail_frames: int = 0,
    boundary_runway_frames: int = 0,
    grid_padding_frames: int | None = None,
    parent_segment_index: int | None = None,
) -> Path:
    """Persist the FULL raw sampled AV latent plus EXTEND-style delivery geometry.

    The sidecar is a take, not the delivered MP4. Continuation children keep
    their masked prefix in RAW coordinates even though that scaffolding is
    trimmed from the file shown to the editor. Future pins therefore map from
    delivered coordinates back to RAW via ``pinned_head_frames``.
    """
    from safetensors.torch import save_file

    video, audio = _streams(sampled_latent, "sampled_latent")
    video_cpu = video[:1].detach().to(device="cpu", copy=True).contiguous()
    audio_cpu = audio[:1].detach().to(device="cpu", copy=True).contiguous()
    raw_frames = frames_for_steps(int(video_cpu.shape[2]))
    pinned_head = max(0, int(pinned_head_frames))
    padding_tail = max(0, int(padding_tail_frames))
    boundary_runway = max(0, int(boundary_runway_frames or 0))
    if grid_padding_frames is None:
        grid_padding = max(0, padding_tail - boundary_runway)
    else:
        grid_padding = max(0, int(grid_padding_frames or 0))
    if boundary_runway + grid_padding != padding_tail:
        raise ValueError(
            "Extended AV technical-tail geometry mismatch: "
            f"runway={boundary_runway} grid_pad={grid_padding} total_pad={padding_tail}."
        )
    delivered = max(0, int(delivered_frames))
    if pinned_head + delivered + padding_tail > raw_frames:
        raise ValueError(
            "Extended AV delivery geometry exceeds RAW latent: "
            f"head={pinned_head} delivered={delivered} pad={padding_tail} raw={raw_frames}."
        )
    self_id = _identity(render_id, segment_index, video_cpu, audio_cpu)
    parent_id = ""
    parent_join = 0
    source_start = 0
    overlap = int(contract.get("overlap_frames", 39) or 39)
    if parent_segment_index is not None and int(segment_index) > 0:
        parent = load_sidecar(render_id, int(parent_segment_index))
        if parent is None or not isinstance(parent.get("meta"), dict):
            raise ValueError("Extended AV child cannot resolve its parent sidecar.")
        pmeta = parent["meta"]
        parent_id = str(pmeta.get("self_id", "") or "")
        p_head = int(pmeta.get("pinned_head_frames", 0) or 0)
        p_delivered = int(pmeta.get("delivered_frames", 0) or 0)
        if p_delivered < overlap:
            raise ValueError("Extended AV parent delivered range is shorter than the requested pin window.")
        source_start = p_head + p_delivered - overlap
        if source_start % H3_FRAME_STEP != 0:
            raise ValueError(
                f"Extended AV parent tail starts at RAW frame {source_start}, off the 17-frame latent phase."
            )
        parent_join = source_start + handover_frames(overlap, contract)

    pin_recipe = []
    if parent_id:
        pin_recipe = [{
            "source_id": parent_id,
            "source_kind": "clip",
            "source_start": int(source_start),
            "source_frames": int(overlap),
            "place": "before",
            "mode": "masked",
            "mask_ramp_frames": int(contract.get("video_ramp_frames", 0) or 0),
            "mask_ramp_edge": float(contract.get("video_edge", 0.0) or 0.0),
            "mask_hold": float(contract.get("video_hold", 0.0) or 0.0),
        }]

    metadata = {
        "format": EXTENDED_AV_SIDECAR_SCHEMA,
        "self_id": self_id,
        "render_id": str(render_id),
        "segment_index": str(int(segment_index)),
        "parent_segment_index": "" if parent_segment_index is None else str(int(parent_segment_index)),
        "parent_id": parent_id,
        "relation": "extends" if parent_id else "root",
        "parent_join_frame": str(int(parent_join)),
        "fps": str(H3_FPS),
        "raw_frames": str(int(raw_frames)),
        "pinned_head_frames": str(int(pinned_head)),
        "pinned_tail_frames": "0",
        "padding_tail_frames": str(int(padding_tail)),
        "boundary_runway_frames": str(int(boundary_runway)),
        "grid_padding_frames": str(int(grid_padding)),
        "editorial_endpoint_raw_frame": str(int(pinned_head + delivered)),
        "delivered_frames": str(int(delivered)),
        "overlap_frames": str(int(overlap)),
        "pins": json.dumps(pin_recipe, ensure_ascii=False, separators=(",", ":")),
        "pin_mode": "masked",
        "mask_profile": str(contract.get("mask_profile", "exact")),
        "contract": json.dumps(contract, ensure_ascii=False, separators=(",", ":")),
    }
    path = sidecar_path(render_id, segment_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    save_file({"video": video_cpu, "audio": audio_cpu}, str(tmp), metadata=metadata)
    tmp.replace(path)
    return path


def load_sidecar(render_id: str, segment_index: int) -> dict[str, Any] | None:
    from safetensors import safe_open

    path = sidecar_path(render_id, segment_index)
    if not path.is_file():
        return None
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            meta = dict(handle.metadata() or {})
            if meta.get("format") != EXTENDED_AV_SIDECAR_SCHEMA:
                return None
            video = handle.get_tensor("video").clone()
            audio = handle.get_tensor("audio").clone()
    except Exception:
        return None
    return {"path": path, "meta": meta, "video": video, "audio": audio}


def _tail_from_sidecar(bundle: dict[str, Any], overlap_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice the tail of the DELIVERED parent, mapped back onto its RAW latent.

    This is the critical EXTEND rule missing from v1: continuation children
    carry pinned scaffolding in raw coordinates, so the next tail is NOT simply
    "the last N frames of the raw tensor". It starts at
    ``pinned_head + delivered - overlap``.
    """
    meta = bundle.get("meta") if isinstance(bundle, dict) else None
    video = bundle.get("video") if isinstance(bundle, dict) else None
    audio = bundle.get("audio") if isinstance(bundle, dict) else None
    if not isinstance(meta, dict) or not torch.is_tensor(video) or not torch.is_tensor(audio):
        raise ValueError("Extended AV sidecar is incomplete.")
    overlap = int(overlap_frames)
    if not shared_av_window_ok(overlap):
        raise ValueError("Extended AV tail request is off the shared AV masked-window grid.")
    pinned_head = int(meta.get("pinned_head_frames", 0) or 0)
    delivered = int(meta.get("delivered_frames", meta.get("visible_frames", 0)) or 0)
    if delivered < overlap:
        raise ValueError("Extended AV delivered parent is shorter than the requested masked window.")
    raw_start = pinned_head + delivered - overlap
    raw_end = raw_start + overlap
    if raw_start % H3_FRAME_STEP != 0:
        raise ValueError(
            f"Extended AV delivered-tail RAW start {raw_start}f is off phase-0; refusing an unsound latent slice."
        )
    start_step = (raw_start // H3_FRAME_STEP) * 5
    tail_steps = video_tokens_for_frames(overlap)
    end_step = start_step + tail_steps
    if end_step > int(video.shape[2]):
        raise ValueError("Extended AV video sidecar does not contain the delivered tail slice.")
    video_tail = video[:, :, start_step:end_step].contiguous()

    a_lo = audio_total(raw_start)
    a_hi = audio_total(raw_end)
    expected = int(round(overlap * AUDIO_LATENT_FPS / float(H3_FPS)))
    if a_hi - a_lo != expected:
        raise ValueError(
            f"Extended AV audio slice {a_lo}:{a_hi} does not cover the {expected} ticks required by {overlap} frames."
        )
    if a_hi > int(audio.shape[-1]):
        raise ValueError("Extended AV audio sidecar does not contain the delivered tail slice.")
    audio_tail = audio[..., a_lo:a_hi].contiguous()
    return video_tail, audio_tail


def _add_video_lineage_guides(conditioning: Any, video_tail: torch.Tensor, overlap_frames: int) -> Any:
    """Attach the same raw tail as native keyframe rows for pin_mode=both."""
    import node_helpers

    starts = _pixel_starts(int(video_tail.shape[2]))
    guides = [
        {
            "resolved_frame_index": int(frame),
            "latent": video_tail[:, :, index:index + 1].clone(),
            "iamccs_extended_av_origin": "lineage_pin",
        }
        for index, frame in enumerate(starts)
        if int(frame) < int(overlap_frames)
    ]
    # ComfyUI append=True already prepends the existing list internally. Pass
    # only the new lineage rows here; passing existing+new would duplicate every
    # authored keyframe on each continuation.
    return node_helpers.conditioning_set_values(
        conditioning,
        {"minimax_keyframes": guides},
        append=True,
    )


def apply_prefix(
    target_latent: dict[str, Any],
    conditioning: Any,
    previous: dict[str, Any],
    *,
    contract: dict[str, Any],
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Apply an EXTEND-style masked BEFORE pin to a fresh target AV latent."""
    if not core_masks_available():
        raise RuntimeError(
            "FL2VA Extended AV needs ComfyUI's native MiniMax H3 per-stream denoise masks "
            "(mask_row_values + MiniMaxH3.scale_latent_inpaint)."
        )
    if target_latent.get("noise_mask") is not None:
        raise ValueError(
            "FL2VA Extended AV masked extend requires a FRESH target AV latent; "
            "the target already carries a noise_mask."
        )
    from comfy.nested_tensor import NestedTensor

    target_video, target_audio = _streams(target_latent, "target_latent")
    overlap = int(contract.get("overlap_frames", 39))
    source_video, source_audio = _tail_from_sidecar(previous, overlap)
    video_steps = int(source_video.shape[2])
    audio_ticks = int(source_audio.shape[-1])
    if video_steps >= int(target_video.shape[2]) or audio_ticks >= int(target_audio.shape[-1]):
        raise ValueError("Extended AV pin would consume the complete target run.")
    if tuple(source_video.shape[3:]) != tuple(target_video.shape[3:]):
        raise ValueError("Extended AV parent/child video latent canvases differ.")
    if tuple(source_audio.shape[1:3]) != tuple(target_audio.shape[1:3]):
        raise ValueError("Extended AV parent/child audio latent geometry differs.")

    video = target_video.clone()
    audio = target_audio.clone()
    source_video = source_video.to(device=video.device, dtype=video.dtype)
    source_audio = source_audio.to(device=audio.device, dtype=audio.dtype)
    video[:, :, :video_steps] = source_video.expand(video.shape[0], -1, -1, -1, -1)
    video_only = bool(contract.get("video_only", False))
    if not video_only:
        audio[..., :audio_ticks] = source_audio.expand(audio.shape[0], -1, -1, -1)

    # Match upstream shapes exactly; do not rely on spatial broadcasting.
    video_mask = torch.ones((1, 1) + tuple(video.shape[2:]), dtype=torch.float32, device=video.device)
    profile = _video_mask_values(overlap, contract, device=video.device, dtype=video_mask.dtype)
    video_mask[:, :, :video_steps] = profile.view(1, 1, video_steps, 1, 1)

    audio_mask = torch.ones((1, 1) + tuple(audio.shape[2:]), dtype=torch.float32, device=audio.device)
    if not video_only:
        audio_mask[..., :audio_ticks] = 0.0
    release = max(0, min(int(contract.get("masked_audio_release_ticks", DEFAULT_MASKED_AUDIO_RELEASE_TICKS) or 0), audio_ticks))
    if video_only:
        release = 0
    if release > 0:
        # masked BEFORE-pin rule: release only at the INTERNAL boundary from
        # pinned audio toward generated audio. These ticks are scaffolding and
        # are trimmed from the delivered child.
        i = torch.arange(1, release + 1, device=audio_mask.device, dtype=audio_mask.dtype)
        ramp = 0.5 - 0.5 * torch.cos(torch.pi * i / float(release))
        audio_mask[..., audio_ticks - release:audio_ticks] = ramp.expand(
            audio_mask.shape[:-1] + (release,)
        ).clone()

    result = dict(target_latent)
    result["samples"] = NestedTensor((video, audio))
    result["noise_mask"] = NestedTensor((video_mask, audio_mask))
    trim_head = handover_frames(overlap, contract)
    result["iamccs_extended_av"] = {
        "schema": EXTENDED_AV_SCHEMA,
        "role": "extend",
        "place": "before",
        "overlap_frames": overlap,
        "video_tokens": video_steps,
        "audio_ticks": audio_ticks,
        "pin_mode": "masked",
        "mask_profile": str(contract.get("mask_profile", "exact")),
        "trim_head_frames": int(trim_head),
    }
    held = int((profile == 0).sum().item())
    return result, conditioning, {
        "overlap_frames": overlap,
        "video_tokens": video_steps,
        "audio_ticks": audio_ticks,
        "video_mask_min": float(profile.min().item()),
        "video_mask_max": float(profile.max().item()),
        "video_exact_tokens": held,
        "audio_release_ticks": int(release),
        "trim_head_frames": int(trim_head),
        "pinned_same_time_context": True,
        "decoded_overlap": False,
        "role": "extend",
        "place": "before",
    }


def latent_seam_report(
    current_sampled: dict[str, Any],
    previous: dict[str, Any],
    *,
    overlap_frames: int,
) -> dict[str, Any]:
    """Cheap latent-domain continuity report, for the EXTEND seam audit."""
    child_video, child_audio = _streams(current_sampled, "current_sampled")
    parent_video, parent_audio = _tail_from_sidecar(previous, int(overlap_frames))
    n_video = min(int(parent_video.shape[2]), int(child_video.shape[2]))
    n_audio = min(int(parent_audio.shape[-1]), int(child_audio.shape[-1]))
    if n_video < 2:
        return {"measurable": False}

    a = parent_video[:, :, :n_video].detach().float().cpu().flatten()
    b = child_video[:, :, :n_video].detach().float().cpu().flatten()
    denom = float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
    fidelity = float(torch.dot(a, b) / denom) if denom > 1e-12 else 1.0

    v = child_video.detach().float().cpu()
    diffs = ((v[:, :, 1:] - v[:, :, :-1]) ** 2).mean(dim=(0, 1, 3, 4)).sqrt()
    boundary = max(0, n_video - 1)
    scan_stop = min(int(diffs.numel()), boundary + 8)
    scan = diffs[boundary:scan_stop]
    baseline = diffs[max(0, boundary - 8):boundary]
    baseline_value = float(baseline.median().item()) if int(baseline.numel()) else float(diffs.median().item())
    seam_ratio = float(scan.max().item()) / max(1e-8, baseline_value) if int(scan.numel()) else 1.0

    audio_fidelity = None
    if n_audio > 0:
        aa = parent_audio[..., :n_audio].detach().float().cpu().flatten()
        bb = child_audio[..., :n_audio].detach().float().cpu().flatten()
        aden = float(torch.linalg.vector_norm(aa) * torch.linalg.vector_norm(bb))
        audio_fidelity = float(torch.dot(aa, bb) / aden) if aden > 1e-12 else 1.0

    return {
        "measurable": True,
        "video_fidelity_cosine": fidelity,
        "audio_fidelity_cosine": audio_fidelity,
        "latent_seam_ratio": seam_ratio,
        "overlap_frames": int(overlap_frames),
        "video_tokens": n_video,
        "audio_ticks": n_audio,
        "verdict": "seamless" if seam_ratio < 1.45 else ("soft_bump" if seam_ratio < 1.9 else "hard_cut"),
    }


def write_seam_report(render_id: str, segment_index: int, report: dict[str, Any]) -> Path:
    path = sidecar_path(render_id, segment_index).with_suffix(".seam.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def equal_power_weight_expression(overlap_frames: int) -> str:
    """FFmpeg blend expression for the incoming rendering of the same time."""
    n = max(2, int(overlap_frames))
    # N is zero-based inside the overlap; use N+1 so the handover starts
    # moving immediately and reaches the incoming rendering at the last frame.
    w = f"pow(sin((N+1)*PI/(2*{n})),2)"
    return f"A*(1-({w}))+B*({w})"


def joint_window_plan(total_frames: int, window_frames: int, overlap_frames: int) -> list[dict[str, int]]:
    """Plan overlapping shared-AV windows for a future one-master latent refine.

    The sampler wiring is deliberately separate from the extension path; this
    function establishes the canonical temporal geometry and is consumed only
    when ``joint_refine_mode`` is explicitly enabled by a later workflow.
    """
    total = max(1, int(total_frames))
    window = int(window_frames)
    overlap = int(overlap_frames)
    if not shared_av_window_ok(window) or not shared_av_window_ok(overlap) or overlap >= window:
        raise ValueError("Extended AV joint window/overlap are not on a legal shared AV grid.")
    if total <= window:
        return [{"index": 0, "start_frame": 0, "end_frame": total}]
    stride = window - overlap
    result: list[dict[str, int]] = []
    start = 0
    index = 0
    while start < total:
        end = min(total, start + window)
        if end - start < SHARED_AV_GRID_BASE and result:
            # Fold a tiny tail into the last legal window rather than create an
            # out-of-distribution micro-window.
            result[-1]["end_frame"] = total
            break
        result.append({"index": index, "start_frame": start, "end_frame": end})
        if end >= total:
            break
        start += stride
        index += 1
    return result
