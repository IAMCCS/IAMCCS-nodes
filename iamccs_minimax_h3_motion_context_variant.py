# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later

"""Isolated IAMCCS MiniMax H3 Motion Context backend variant.

The proven R21/R22 atomic backend is intentionally not modified here.  This
module adapts CineLinX and the IAMCCS segment queue to the upstream standalone
Motion Context implementation, then delegates sampling to the existing atomic
generator.  A lazy result switch guarantees that only the selected generation
branch executes in workflows that expose both backends.
"""

from __future__ import annotations

import copy
import inspect
import json
import logging
import re
from typing import Any

import torch
import torch.nn.functional as F

from .iamccs_minimax_h3_atomic_backend import (
    H3_FPS,
    SUPERNODE_LINX_TYPE,
    IAMCCS_MiniMaxH3GenerationBackendV2,
    _audio_slice,
    _load_image,
    _load_timeline_audio,
    _resolve_shotplan,
)


LOG = logging.getLogger("IAMCCS.MiniMaxH3.MotionContextVariant")
CATEGORY = "IAMCCS/MiniMax H3/Motion Context Variant"

IAMCCS_MC_PATCH_REV = "R42-anchor-latent-v2-2026-09-16"
LOG.warning(
    "IAMCCS Motion Context Variant LOADED | patch=%s | file=%s",
    IAMCCS_MC_PATCH_REV,
    __file__,
)
PROVIDER = "ComfyUI-H3-Motion-Context-Auto-Chain-addon"


def _provider_node(name: str):
    import nodes as comfy_nodes

    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(name)
    if cls is None:
        raise RuntimeError(
            f"IAMCCS Motion Context R37 requires {PROVIDER} V0.1.2 or newer; "
            f"provider node '{name}' is unavailable. Update/enable the addon and restart ComfyUI."
        )
    return cls


def _provider_chain_config(chain_config: dict[str, Any], **fallbacks) -> dict[str, Any]:
    """Return an isolated H3_CHAIN contract compatible with newer Auto-Chain builds.

    Some provider releases read ``effective_trim_frames`` directly from
    ``chain_config``.  IAMCCS historically also supported older releases where
    context/trim values were separate arguments.  Normalize the bridge here and
    never give a third-party provider our original dict by reference.
    """
    if not isinstance(chain_config, dict):
        raise TypeError(
            "IAMCCS Motion Context provider chain_config must be a dict, "
            f"got {type(chain_config).__name__}"
        )

    config = copy.deepcopy(chain_config)
    repaired: list[str] = []

    if "effective_trim_frames" not in config:
        trim = fallbacks.get("trim_frames", fallbacks.get("context_length", 0))
        config["effective_trim_frames"] = max(0, int(trim or 0))
        repaired.append("effective_trim_frames")

    if "fps" not in config:
        config["fps"] = float(fallbacks.get("fps", 24.0) or 24.0)
        repaired.append("fps")

    if "clip_index" not in config:
        if "save_clip_index" in config:
            config["clip_index"] = int(config["save_clip_index"])
        elif "load_clip_index" in config:
            config["clip_index"] = int(config["load_clip_index"]) + 1
        else:
            config["clip_index"] = 1
        repaired.append("clip_index")

    if repaired:
        LOG.warning(
            "IAMCCS repaired incomplete H3_CHAIN provider contract: %s",
            ", ".join(repaired),
        )

    return config


def _provider_call(name: str, method: str, *, chain_config: dict[str, Any], **kwargs):
    """Call both legacy and current Auto-Chain provider APIs safely.

    Older provider builds exposed ``latent_path``/``clip_index`` widgets and
    separate context-length arguments.  Newer builds collapse those values into
    one ``H3_CHAIN`` input and require ``effective_trim_frames`` inside it.

    The provider always receives an isolated copy so a Load/Save implementation
    cannot mutate the config later reused by Motion Context.
    """
    instance = _provider_node(name)()
    function = getattr(instance, method)
    parameters = inspect.signature(function).parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    payload = {
        key: value
        for key, value in kwargs.items()
        if accepts_kwargs or key in parameters
    }
    if accepts_kwargs or "chain_config" in parameters:
        payload["chain_config"] = _provider_chain_config(chain_config, **kwargs)
    return function(**payload)


def _chunk(plan: dict[str, Any], index: int) -> dict[str, Any]:
    chunks = plan.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("IAMCCS Motion Context R37 received a shotplan without chunks")
    index = int(index)
    if index < 0 or index >= len(chunks):
        raise IndexError(f"segment_index={index} outside 0..{len(chunks) - 1}")
    return chunks[index]


def _variant_active(cine_linx: Any) -> bool:
    plan = _resolve_shotplan(cine_linx)
    task_mode = str(plan.get("task_mode", "") or "").strip().lower()
    contract = plan.get("motion_context_auto_chain")
    motion_context_enabled = bool(isinstance(contract, dict) and contract.get("enabled"))
    declared_variant = str(plan.get("backend_variant", "") or "").strip().lower()

    # LongVid Motion Context must never fall through to the legacy/T2V branch.
    # The executable contract is the authority; acceleration is only a sampler
    # choice and cannot prove that a Motion Context chain was compiled.
    motion_context_modes = {"longvid_motion_context", "longvid_continuous_guided"}
    if task_mode in motion_context_modes and not motion_context_enabled:
        raise RuntimeError(
            "IAMCCS Motion Context contract is missing or disabled. "
            "Recompile the Shotboard/Settings plan; refusing to fall back to T2V."
        )
    if motion_context_enabled and declared_variant not in ("", "motion_context_auto_chain_v1"):
        raise RuntimeError(
            "IAMCCS Motion Context contract conflicts with backend_variant="
            f"'{declared_variant}'. Refusing an ambiguous backend route."
        )
    if declared_variant == "motion_context_auto_chain_v1" and not motion_context_enabled:
        raise RuntimeError(
            "IAMCCS Motion Context backend was selected without an enabled "
            "motion_context_auto_chain contract."
        )
    return motion_context_enabled or str(plan.get("acceleration", "") or "").strip().lower() == "comfy_kitchen"


def _motion_context_active(cine_linx: Any) -> bool:
    plan = _resolve_shotplan(cine_linx)
    contract = plan.get("motion_context_auto_chain")
    return bool(isinstance(contract, dict) and contract.get("enabled"))


def _replace_plan(cine_linx: Any, plan: dict[str, Any]) -> dict[str, Any]:
    """Replace every known CineLinX shotplan mirror by name, never position."""
    if isinstance(cine_linx, dict) and cine_linx.get("schema") == "iamccs.minimax_h3.shotplan":
        return copy.deepcopy(plan)
    result = copy.deepcopy(cine_linx)
    resources = result.get("resources") if isinstance(result.get("resources"), dict) else {}
    outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
    payload = resources.get("cine_payload") if isinstance(resources.get("cine_payload"), dict) else {}
    for key in ("iamccs_minimax_h3_shotplan", "minimax_h3_shotplan", "shotplan"):
        if key in resources:
            resources[key] = copy.deepcopy(plan)
    if "shotplan" in outputs:
        outputs["shotplan"] = copy.deepcopy(plan)
    for key in ("minimax_h3_shotplan", "shotplan"):
        if key in payload:
            payload[key] = copy.deepcopy(plan)
    if not any(key in resources for key in ("iamccs_minimax_h3_shotplan", "minimax_h3_shotplan", "shotplan")):
        resources["iamccs_minimax_h3_shotplan"] = copy.deepcopy(plan)
    resources["cine_payload"] = payload
    result["resources"] = resources
    result["outputs"] = outputs
    return result


def _safe_run_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "iamccs_h3")).strip("._")
    return name or "iamccs_h3"


def _chain_config(
    render_id: str,
    segment_index: int,
    *,
    effective_trim_frames: int = 0,
    fps: float = 24.0,
) -> dict[str, Any]:
    index = int(segment_index)
    run_name = _safe_run_name(render_id)
    return {
        "chain_id": run_name,
        "run_name": run_name,
        "latent_prefix": f"h3_context/iamccs_{run_name}_clip",
        "load_clip_index": max(0, index),
        "save_clip_index": index + 1,
        "reset": index == 0,
        # Auto-Chain addon v0.1.9+ owns its context span through H3_CHAIN.
        # Older IAMCCS adapters only supplied the filesystem fields above,
        # which made chunk 2 fail before sampling with KeyError.
        "effective_trim_frames": max(0, int(effective_trim_frames)),
        "fps": float(fps),
        "clip_index": index + 1,
    }


def _trim_chain_config(trim_frames: int, fps: float, segment_index: int) -> dict[str, Any]:
    """Minimal chain contract needed by provider v0.1.9's trim node.

    Trimming happens after sampling and does not need the run's filesystem
    prefix.  Keeping this local also lets old graphs use the same adapter
    without adding a new render-id socket to the generation node.
    """
    return {
        "effective_trim_frames": max(0, int(trim_frames)),
        "fps": float(fps),
        "clip_index": int(segment_index) + 1,
    }


def _fit_audio(audio: Any, frames: int, fps: float) -> Any:
    if not isinstance(audio, dict) or not torch.is_tensor(audio.get("waveform")):
        return audio
    sample_rate = max(1, int(audio.get("sample_rate", 32000)))
    wanted = max(1, int(round(int(frames) / float(fps) * sample_rate)))
    waveform = audio["waveform"]
    if int(waveform.shape[-1]) > wanted:
        waveform = waveform[..., :wanted]
    elif int(waveform.shape[-1]) < wanted:
        waveform = F.pad(waveform, (0, wanted - int(waveform.shape[-1])))
    result = dict(audio)
    result["waveform"] = waveform
    result["sample_rate"] = sample_rate
    return result


def _remove_audio_frame_range(
    audio: Any,
    start_frame: int,
    end_frame: int,
    fps: float,
) -> Any:
    """Remove from audio the same interval removed from final decoded video."""
    if not isinstance(audio, dict) or not torch.is_tensor(audio.get("waveform")):
        return audio
    sample_rate = max(1, int(audio.get("sample_rate", 32000)))
    waveform = audio["waveform"]
    total_samples = int(waveform.shape[-1])
    start_sample = max(
        0,
        min(total_samples, int(round(float(start_frame) / float(fps) * sample_rate))),
    )
    end_sample = max(
        start_sample,
        min(total_samples, int(round(float(end_frame) / float(fps) * sample_rate))),
    )
    if end_sample <= start_sample:
        return audio
    result = dict(audio)
    result["waveform"] = torch.cat(
        (waveform[..., :start_sample], waveform[..., end_sample:]),
        dim=-1,
    )
    result["sample_rate"] = sample_rate
    return result


def _freeze_safe_final_tail(
    frames: Any,
    fps: float,
    plan: dict[str, Any],
) -> tuple[Any, dict[str, Any] | None]:
    """Remove only a genuinely static generated tail from the final Guided chunk.

    Motion is measured adaptively against the earlier part of the same clip.
    The exact last decoded frame is always preserved, so the authored final
    Shotboard destination remains the delivery endpoint.
    """
    if not torch.is_tensor(frames) or frames.ndim != 4:
        return frames, None

    total = int(frames.shape[0])
    if total < 16:
        return frames, None

    contract = plan.get("motion_context_auto_chain")
    if not isinstance(contract, dict):
        contract = {}

    if not bool(contract.get("freeze_safe_tail_trim", True)):
        return frames, None

    probe_frames = max(
        12,
        int(contract.get("freeze_safe_probe_frames", round(float(fps) * 2.0)) or round(float(fps) * 2.0)),
    )
    min_static_frames = max(
        6,
        int(contract.get("freeze_safe_min_static_frames", round(float(fps) * 0.40)) or round(float(fps) * 0.40)),
    )
    keep_final_frames = max(
        1,
        int(contract.get("freeze_safe_keep_final_frames", 1) or 1),
    )
    absolute_threshold = max(
        0.0001,
        float(contract.get("freeze_safe_abs_motion_threshold", 0.0015) or 0.0015),
    )
    baseline_ratio = max(
        0.01,
        min(0.80, float(contract.get("freeze_safe_baseline_ratio", 0.18) or 0.18)),
    )

    probe_frames = min(probe_frames, total - 3)
    if probe_frames < min_static_frames:
        return frames, None

    with torch.no_grad():
        rgb = frames[..., :3].detach().to(dtype=torch.float32).movedim(-1, 1)
        h, w = int(rgb.shape[-2]), int(rgb.shape[-1])
        scale = min(1.0, 64.0 / max(1, h), 64.0 / max(1, w))
        if scale < 1.0:
            rgb = F.interpolate(
                rgb,
                size=(max(8, int(round(h * scale))), max(8, int(round(w * scale)))),
                mode="area",
            )

        diffs = (rgb[1:] - rgb[:-1]).abs().mean(dim=(1, 2, 3))
        if int(diffs.numel()) < min_static_frames:
            return frames, None

        baseline_end = max(4, int(diffs.numel()) - probe_frames)
        baseline = float(torch.median(diffs[:baseline_end]).item())
        threshold = max(absolute_threshold, baseline * baseline_ratio)

        # If the whole clip is essentially static, there is no trustworthy
        # boundary between intended stillness and a generated freeze.
        if baseline <= absolute_threshold * 1.5:
            return frames, None

        low_run = 0
        for value in reversed(diffs[-probe_frames:].tolist()):
            if float(value) <= threshold:
                low_run += 1
            else:
                break

    static_frames = low_run + 1 if low_run else 0
    if static_frames < min_static_frames:
        return frames, None

    static_frames = min(static_frames, total - 2)
    keep_final_frames = min(keep_final_frames, static_frames)

    remove_start = total - static_frames
    remove_end = total - keep_final_frames
    removed = remove_end - remove_start
    if removed <= 0:
        return frames, None

    trimmed = torch.cat((frames[:remove_start], frames[remove_end:]), dim=0)
    return trimmed, {
        "removed_frames": int(removed),
        "static_frames": int(static_frames),
        "remove_start_frame": int(remove_start),
        "remove_end_frame": int(remove_end),
        "kept_final_frames": int(keep_final_frames),
        "baseline_motion": float(baseline),
        "threshold": float(threshold),
        "original_frames": int(total),
        "delivered_frames": int(trimmed.shape[0]),
    }


def _sampling_from_plan(plan: dict[str, Any], **fallbacks) -> dict[str, Any]:
    """Resolve R37 sampler controls from the Shotboard Queue-time truth.

    Early R37 workflows retained sampler widgets on the isolated generation
    node.  Later Settings/Shotboard revisions made ``plan['sampling']`` the
    visible authoring truth, but the R37 branch still consumed those stale
    node widgets.  Keep the inputs as compatibility fallbacks while honoring
    the compiled plan whenever it is present.
    """
    sampling = plan.get("sampling")
    if not isinstance(sampling, dict):
        sampling = {}
    return {
        "seed": int(sampling.get("seed", fallbacks["seed"])),
        "seed_stride": int(sampling.get("seed_stride", fallbacks["seed_stride"])),
        "steps": int(sampling.get("steps", fallbacks["steps"])),
        "sampler_name": str(sampling.get("sampler_name", fallbacks["sampler_name"])),
        "scheduler": str(sampling.get("scheduler", fallbacks["scheduler"])),
        "denoise": float(sampling.get("denoise", fallbacks["denoise"])),
        "shift_video": float(sampling.get("shift_video", fallbacks["shift_video"])),
        "shift_audio": float(sampling.get("shift_audio", fallbacks["shift_audio"])),
    }


def _visual_keyframe_payload(keyframe: Any) -> Any:
    """Return the visual payload carried by a MiniMax H3 keyframe.

    ComfyUI 0.34+ consumes the temporary ``image`` field inside
    MiniMaxH3ImageToVideo and stores the encoded anchor as ``latent``.
    Older/custom H3 implementations may still expose ``image``.  Support both
    representations so destination validation follows the actual conditioning
    contract instead of one historical metadata shape.
    """
    if not isinstance(keyframe, dict):
        return None
    latent_payload = keyframe.get("latent")
    if latent_payload is not None:
        return latent_payload
    image_payload = keyframe.get("image")
    if image_payload is not None:
        return image_payload
    return None


def _terminal_visual_anchors(conditioning: Any, endpoint: int) -> list[dict[str, Any]]:
    """Collect visual H3 anchors exactly at ``endpoint``."""
    anchors: list[dict[str, Any]] = []
    if not isinstance(conditioning, (list, tuple)):
        return anchors
    for item in conditioning:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        metadata = item[1]
        if not isinstance(metadata, dict):
            continue
        keyframes = metadata.get("minimax_keyframes", [])
        if not isinstance(keyframes, (list, tuple)):
            continue
        for keyframe in keyframes:
            if (
                isinstance(keyframe, dict)
                and int(keyframe.get("resolved_frame_index", -1)) == int(endpoint)
                and _visual_keyframe_payload(keyframe) is not None
            ):
                anchors.append(keyframe)
    return anchors


def _same_visual_anchor(source: dict[str, Any], candidate: dict[str, Any]) -> bool:
    """True when Motion Context retained the exact authored visual anchor.

    The upstream Motion Context implementation shallow-copies preserved
    keyframe dictionaries.  The encoded visual payload object therefore remains
    identical even though the dictionary object itself changes.
    """
    if not isinstance(source, dict) or not isinstance(candidate, dict):
        return False
    if int(candidate.get("resolved_frame_index", -1)) != int(source.get("resolved_frame_index", -2)):
        return False
    source_payload = _visual_keyframe_payload(source)
    candidate_payload = _visual_keyframe_payload(candidate)
    return source_payload is not None and candidate_payload is source_payload


def _guide_uses_native_tail(guide: Any, context_offset_frames: int) -> bool:
    """True when the previous native AV tail already owns this visual beat.

    A visual slot can span a technical chunk boundary.  Re-encoding that same
    still at the first visible frame after the 22-frame pinned head creates a
    cut exactly where Motion Context is meant to be seamless.  Future guides
    remain real positioned anchors; only the carried active guide is skipped.
    """
    return bool(
        int(context_offset_frames) > 0
        and isinstance(guide, dict)
        and str(guide.get("kind", "") or "").strip().lower() == "image"
        and bool(guide.get("continued_from_previous_chunk"))
    )


def _apply_positioned_guides(
    conditioning,
    latent,
    video_vae,
    audio_vae,
    plan: dict[str, Any],
    chunk: dict[str, Any],
    context_offset_frames: int,
):
    """Materialize each authored R37 guide exactly once for this chunk."""
    # Both public Motion Context modes use the same positioned-guide contract.
    # Previously ``longvid_continuous_guided`` compiled the guide events but
    # silently skipped them here.  Continuation chunks then had only the prior
    # AV tail plus their terminal image, which could resurrect an older slot
    # before finally converging on the requested destination.
    task_mode = str(plan.get("task_mode", "") or "").lower()
    if task_mode not in {"longvid_motion_context", "longvid_continuous_guided"}:
        return conditioning, []
    guide_events = chunk.get("guides")
    if not isinstance(guide_events, list):
        return conditioning, []

    contract = plan.get("motion_context_auto_chain")
    guide_boundary = ""
    if isinstance(contract, dict):
        guide_boundary = str(contract.get("guide_boundary", "") or "").strip().lower()
    if guide_boundary == "safe_handoff":
        # The planner must hand an authored image to the next technical
        # interval at its opening.  An image guide in the visible middle of a
        # Motion Context sample is exactly the topology that produced the
        # full-frame flashes seen in the first chunk.  Refuse stale plans so a
        # queued workflow cannot silently run the old positioned-guide path.
        visible_frames = int(chunk.get("visible_frame_count", chunk.get("frame_count", 0)) or 0)
        interior = [
            str(guide.get("id", "guide"))
            for guide in guide_events
            if isinstance(guide, dict)
            and str(guide.get("kind", "") or "").strip().lower() == "image"
            and 0 < int(guide.get("local_frame", 0) or 0) < visible_frames
        ]
        if interior:
            raise RuntimeError(
                "Motion Context Shotboard plan is stale: image guide(s) "
                f"{', '.join(interior)} occur inside a visible sample. "
                "Reopen/apply the Shotboard so guides become handoff boundaries."
            )

    from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide

    applied: list[str] = []
    custom_audio_drive = str(plan.get("audio_mode", "") or "").lower() == "h3_custom_audio_drive"
    for guide in guide_events:
        if not isinstance(guide, dict):
            continue
        kind = str(guide.get("kind", "") or "").strip().lower()
        guide_id = str(guide.get("id", "guide") or "guide").strip()
        if _guide_uses_native_tail(guide, context_offset_frames):
            applied.append(f"native-tail:{guide_id}")
            continue
        source_path = str(guide.get("source_path", "") or "").strip()
        frame_index = max(0, int(guide.get("local_frame", 0) or 0)) + max(0, int(context_offset_frames))
        if kind == "image":
            image = _load_image(source_path)
            if image is None:
                raise ValueError(f"R37 LongVid image guide '{guide_id}' has no source image")
            conditioning = MiniMaxH3AddGuide.execute(
                positive=conditioning,
                latent=latent,
                frame_idx=frame_index,
                vae=video_vae,
                image=image,
            )[0]
            applied.append(f"image:{guide_id}@{frame_index}")
        elif kind == "audio":
            if custom_audio_drive:
                # AtomicAudioDrive already owns the rebased waveform.  Adding
                # it again as a guide would duplicate phonetic conditioning.
                applied.append(f"audio-lock-only:{guide_id}@{frame_index}")
                continue
            audio = _load_timeline_audio(source_path)
            if audio is None:
                raise ValueError(f"R37 LongVid audio guide '{guide_id}' has no source audio")
            source_offset_seconds = max(0.0, float(guide.get("source_offset_frames", 0) or 0) / H3_FPS)
            duration_seconds = max(1.0 / H3_FPS, float(guide.get("duration_frames", 1) or 1) / H3_FPS)
            audio = _audio_slice(audio, source_offset_seconds, duration_seconds)
            conditioning = MiniMaxH3AddGuide.execute(
                positive=conditioning,
                latent=latent,
                frame_idx=frame_index,
                audio_vae=audio_vae,
                audio=audio,
            )[0]
            applied.append(f"audio:{guide_id}@{frame_index}")
        else:
            raise ValueError(f"Unsupported R37 LongVid guide kind '{kind}' for '{guide_id}'")
    return conditioning, applied


class IAMCCS_MiniMaxH3MotionContextConditionR37:
    """Load the prior IAMCCS AV latent and apply the upstream V0.1.2 core."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "video_vae": ("VAE",),
                "latent": ("LATENT",),
                "audio_vae": ("VAE",),
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "segment_index": ("INT", {"forceInput": True}),
                "render_id": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "INT", "STRING")
    RETURN_NAMES = ("conditioning", "trim_frames", "report")
    FUNCTION = "apply"
    CATEGORY = CATEGORY

    def apply(self, conditioning, video_vae, latent, audio_vae, cine_linx, segment_index, render_id=""):
        plan = _resolve_shotplan(cine_linx)
        chunk = _chunk(plan, segment_index)
        if not _motion_context_active(cine_linx):
            return conditioning, 0, "R37 Motion Context bypass: selected branch uses ComfyKitchen only"

        contract = plan.get("motion_context_auto_chain") or {}
        context_frames = int(contract.get("context_frames", 22) or 22)
        audio_context_frames = int(contract.get("audio_context_frames", 24) or 0)
        expected_trim = int(chunk.get("motion_context_trim_frames", 0) or 0)
        conditioning, applied_guides = _apply_positioned_guides(
            conditioning,
            latent,
            video_vae,
            audio_vae,
            plan,
            chunk,
            expected_trim,
        )
        guide_report = ",".join(applied_guides) if applied_guides else "none"
        if applied_guides:
            LOG.info(
                "MiniMax H3 R37 positioned guides materialized | chunk=%d/%d | %s",
                int(segment_index) + 1,
                len(plan.get("chunks", [])),
                ", ".join(applied_guides),
            )
        if int(segment_index) <= 0 or expected_trim <= 0:
            mode_label = (
                "LONG CONTINUOUS GUIDED interval 1"
                if str(plan.get("task_mode", "") or "").lower() == "longvid_continuous_guided"
                else "R37 Motion Context clip 1"
            )
            return conditioning, 0, (
                f"{mode_label}: upstream pass-through (no previous latent) | "
                f"positioned_guides={guide_report}"
            )

        config = _chain_config(
            render_id,
            segment_index,
            effective_trim_frames=expected_trim,
            fps=float(plan.get("fps", 24) or 24),
        )
        context_latent = _provider_call(
            "MiniMaxH3AutoChainLoadLatent",
            "load",
            chain_config=config,
            latent_path=config["latent_prefix"],
            clip_index=config["load_clip_index"],
            reset=False,
        )[0]
        if context_latent is None:
            raise RuntimeError(
                "IAMCCS Motion Context R37 could not load the previous native AV latent. "
                f"render_id={render_id!r}, previous_segment={int(segment_index)}. "
                "Do not purge h3_context while a chain is running."
            )
        # Capture the actual terminal anchor before the provider installs its
        # native head.  ComfyUI 0.34+ stores ImageToVideo visual anchors as
        # encoded ``latent`` payloads (the temporary ``image`` key is popped
        # during conditioning), while older/custom builds may still expose
        # ``image``.  Validate the semantic visual anchor rather than one
        # historical metadata field.
        destination = str(chunk.get("last_image", "") or "")
        terminal_anchors: list[dict[str, Any]] = []
        if str(plan.get("task_mode", "")) == "longvid_continuous_guided":
            endpoint = int(chunk["frame_count"]) - 1
            terminal_anchors = _terminal_visual_anchors(conditioning, endpoint)
            available_visual_indices = sorted({
                int(k.get("resolved_frame_index", -1))
                for item in conditioning
                if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], dict)
                for k in item[1].get("minimax_keyframes", [])
                if isinstance(k, dict) and _visual_keyframe_payload(k) is not None
            })
            LOG.warning(
                "IAMCCS MC DESTINATION AUDIT | patch=%s | segment=%d | destination=%r | "
                "expected_frame=%d | terminal_anchor_count=%d | visual_keyframes=%s",
                IAMCCS_MC_PATCH_REV,
                int(segment_index) + 1,
                destination,
                endpoint,
                len(terminal_anchors),
                available_visual_indices,
            )
            if not destination or not terminal_anchors:
                raise ValueError(
                    f"Continuous Guided segment {int(segment_index) + 1}: "
                    f"missing Shotboard destination anchor {destination!r} at frame {endpoint}; "
                    f"visual keyframes present at {available_visual_indices or 'none'}; "
                    f"patch={IAMCCS_MC_PATCH_REV}"
                )
        conditioned, actual_trim = _provider_call(
            "MiniMaxH3AutoChainMotionContext",
            "apply",
            chain_config=config,
            conditioning=conditioning,
            vae=video_vae,
            latent=latent,
            context_length=str(context_frames),
            audio_context_length=audio_context_frames,
            context_latent=context_latent,
            audio_vae=audio_vae,
        )
        if terminal_anchors:
            retained = [
                k
                for item in conditioned
                if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], dict)
                for k in item[1].get("minimax_keyframes", [])
                if isinstance(k, dict)
            ]
            if not all(
                any(_same_visual_anchor(source, candidate) for candidate in retained)
                for source in terminal_anchors
            ):
                raise RuntimeError("Motion Context changed or removed the Shotboard destination anchor")
            LOG.info("Continuous Guided destination verified | segment=%d | image=%s | frame=%d | native_head_trim=%d | dissolve=not_requested_by_context",
                     int(segment_index) + 1, destination, endpoint, int(actual_trim))
        if int(actual_trim) != expected_trim:
            raise RuntimeError(
                "IAMCCS Motion Context trim contract changed underneath the planner: "
                f"provider={actual_trim}, planned={expected_trim}. Refusing a shifted join."
            )
        report_label = (
            "LONG CONTINUOUS GUIDED native AV hand-off"
            if str(plan.get("task_mode", "") or "").lower() == "longvid_continuous_guided"
            else "R37 upstream Motion Context applied"
        )
        report = (
            f"{report_label} | segment={int(segment_index) + 1}/{len(plan['chunks'])} | "
            f"video_tail={context_frames}f | audio_tail={audio_context_frames}f | trim={actual_trim}f | "
            f"positioned_guides={guide_report}"
        )
        LOG.info(report)
        return conditioned, int(actual_trim), report


class IAMCCS_MiniMaxH3MotionContextGenerationR37:
    """R37 generator: optional CK attention, legacy sampler delegation, exact trim."""

    @classmethod
    def INPUT_TYPES(cls):
        spec = copy.deepcopy(IAMCCS_MiniMaxH3GenerationBackendV2.INPUT_TYPES())
        spec["required"]["trim_frames"] = ("INT", {"forceInput": True})
        return spec

    RETURN_TYPES = IAMCCS_MiniMaxH3GenerationBackendV2.RETURN_TYPES
    RETURN_NAMES = IAMCCS_MiniMaxH3GenerationBackendV2.RETURN_NAMES
    FUNCTION = "render"
    CATEGORY = CATEGORY

    def render(self, model, positive, latent, video_vae, audio_vae, cine_linx, chunk_index,
               seed, seed_stride, steps, sampler_name, scheduler, denoise,
               shift_video, shift_audio, trim_frames, motion_state=None):
        plan = copy.deepcopy(_resolve_shotplan(cine_linx))
        chunk = _chunk(plan, chunk_index)
        sampling = _sampling_from_plan(
            plan,
            seed=seed,
            seed_stride=seed_stride,
            steps=steps,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
            shift_video=shift_video,
            shift_audio=shift_audio,
        )
        seed = sampling["seed"]
        seed_stride = sampling["seed_stride"]
        steps = sampling["steps"]
        sampler_name = sampling["sampler_name"]
        scheduler = sampling["scheduler"]
        denoise = sampling["denoise"]
        shift_video = sampling["shift_video"]
        shift_audio = sampling["shift_audio"]
        acceleration = str(plan.get("acceleration", "native") or "native").lower()
        attention_report = "authored acceleration delegated"
        if acceleration == "comfy_kitchen":
            from comfy_extras.nodes_model_advanced import ModelAttentionBackend

            model = ModelAttentionBackend.execute(
                model=model, attention="comfy kitchen attention"
            ).result[0]
            # The old backend remains immutable and therefore receives native;
            # CK has already been applied per-model above.
            plan["acceleration"] = "native"
            attention_report = "ComfyKitchen INT8 per-model attention"
        adapted_linx = _replace_plan(cine_linx, plan)
        result = IAMCCS_MiniMaxH3GenerationBackendV2().render(
            model=model,
            positive=positive,
            latent=latent,
            video_vae=video_vae,
            audio_vae=audio_vae,
            cine_linx=adapted_linx,
            chunk_index=chunk_index,
            seed=seed,
            seed_stride=seed_stride,
            steps=steps,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
            shift_video=shift_video,
            shift_audio=shift_audio,
            motion_state=motion_state,
        )
        frames, audio, _, sampled_latent, fps, base_report = result
        trim_frames = max(0, int(trim_frames))
        if trim_frames:
            frames, audio = _provider_call(
                "MiniMaxH3AutoChainMotionContextTrim",
                "trim",
                chain_config=_trim_chain_config(trim_frames, float(fps), int(chunk_index)),
                images=frames,
                trim_frames=trim_frames,
                audio=audio,
                fps=float(fps),
                match_tail=True,
            )
        visible_frames = int(chunk.get("visible_frame_count", int(frames.shape[0])) or int(frames.shape[0]))
        if int(frames.shape[0]) < visible_frames:
            raise RuntimeError(
                f"R37 decoded {int(frames.shape[0])} visible frames, planner requires {visible_frames}."
            )
        frames = frames[:visible_frames]
        audio = _fit_audio(audio, visible_frames, float(fps))

        freeze_info = None
        task_mode = str(plan.get("task_mode", "") or "").strip().lower()
        is_final_chunk = int(chunk_index) + 1 >= len(plan.get("chunks", []))
        if task_mode == "longvid_continuous_guided" and is_final_chunk:
            frames, freeze_info = _freeze_safe_final_tail(frames, float(fps), plan)
            if freeze_info:
                audio = _remove_audio_frame_range(
                    audio,
                    freeze_info["remove_start_frame"],
                    freeze_info["remove_end_frame"],
                    float(fps),
                )
                audio = _fit_audio(audio, int(frames.shape[0]), float(fps))
                LOG.info(
                    "IAMCCS freeze-safe final tail | patch=%s | detected=%df | removed=%df | "
                    "kept_final=%df | original=%df | delivered=%df | "
                    "baseline=%.6f | threshold=%.6f",
                    IAMCCS_MC_PATCH_REV,
                    int(freeze_info["static_frames"]),
                    int(freeze_info["removed_frames"]),
                    int(freeze_info["kept_final_frames"]),
                    int(freeze_info["original_frames"]),
                    int(freeze_info["delivered_frames"]),
                    float(freeze_info["baseline_motion"]),
                    float(freeze_info["threshold"]),
                )

        delivered_frames = int(frames.shape[0])
        bridge_last = frames[-1:].detach().clone()
        freeze_report = (
            f" | freeze_safe_tail={int(freeze_info['removed_frames'])}f"
            if freeze_info else " | freeze_safe_tail=0f"
        )
        report = (
            f"{base_report} | R37={attention_report} | upstream_motion_context_trim={trim_frames}f | "
            f"planned_visible={visible_frames}f | delivered={delivered_frames}f"
            f"{freeze_report} | sampler_truth=shotboard:{steps}x{sampler_name}+{scheduler}"
        )
        return frames, audio, bridge_last, sampled_latent, fps, report


class IAMCCS_MiniMaxH3MotionContextStateCommitR37:
    """Persist the sampler AV latent after IAMCCS resolves the numbered render ID."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "audio": ("AUDIO",),
                "sampled_latent": ("LATENT",),
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "segment_index": ("INT", {"forceInput": True}),
                "resolved_render_id": ("STRING", {"forceInput": True}),
                "checkpoint_report": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("frames", "audio", "resolved_render_id", "report")
    FUNCTION = "commit"
    CATEGORY = CATEGORY

    def commit(self, frames, audio, sampled_latent, cine_linx, segment_index,
               resolved_render_id, checkpoint_report):
        if not _motion_context_active(cine_linx):
            return frames, audio, resolved_render_id, checkpoint_report
        config = _chain_config(resolved_render_id, segment_index)
        path = _provider_call(
            "MiniMaxH3AutoChainSaveLatent",
            "save",
            chain_config=config,
            latent=sampled_latent,
            filename_prefix=config["latent_prefix"],
            clip_index=config["save_clip_index"],
        )[0]
        report = f"{checkpoint_report} | R37 Motion Context AV latent committed: {path}"
        LOG.info("IAMCCS Motion Context state committed | segment=%d | %s", int(segment_index) + 1, path)
        return frames, audio, resolved_render_id, report


class IAMCCS_MiniMaxH3BackendLazySwitchR37:
    """Select one six-output generator branch without evaluating the other."""

    @classmethod
    def INPUT_TYPES(cls):
        lazy_image = ("IMAGE", {"lazy": True})
        lazy_audio = ("AUDIO", {"lazy": True})
        lazy_latent = ("LATENT", {"lazy": True})
        lazy_int = ("INT", {"lazy": True})
        lazy_string = ("STRING", {"lazy": True})
        return {
            "required": {"cine_linx": (SUPERNODE_LINX_TYPE,)},
            "optional": {
                "legacy_frames": lazy_image,
                "legacy_audio": lazy_audio,
                "legacy_bridge": lazy_image,
                "legacy_latent": lazy_latent,
                "legacy_fps": lazy_int,
                "legacy_report": lazy_string,
                "variant_frames": lazy_image,
                "variant_audio": lazy_audio,
                "variant_bridge": lazy_image,
                "variant_latent": lazy_latent,
                "variant_fps": lazy_int,
                "variant_report": lazy_string,
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "IMAGE", "LATENT", "INT", "STRING")
    RETURN_NAMES = ("frames", "audio", "bridge", "sampled_latent", "fps", "report")
    FUNCTION = "select"
    CATEGORY = CATEGORY

    @staticmethod
    def _names(cine_linx):
        prefix = "variant" if _variant_active(cine_linx) else "legacy"
        return [f"{prefix}_{suffix}" for suffix in ("frames", "audio", "bridge", "latent", "fps", "report")]

    def check_lazy_status(self, cine_linx, **kwargs):
        return [name for name in self._names(cine_linx) if kwargs.get(name) is None]

    def select(self, cine_linx, **kwargs):
        names = self._names(cine_linx)
        missing = [name for name in names if kwargs.get(name) is None]
        if missing:
            raise ValueError("R37 lazy backend branch is incomplete: " + ", ".join(missing))
        branch = "motion-context/CK variant" if names[0].startswith("variant") else "legacy atomic"
        LOG.info("IAMCCS MiniMax H3 lazy backend selected: %s", branch)
        return tuple(kwargs[name] for name in names)


NODE_CLASS_MAPPINGS = {
    "IAMCCS_MiniMaxH3MotionContextConditionR37": IAMCCS_MiniMaxH3MotionContextConditionR37,
    "IAMCCS_MiniMaxH3MotionContextGenerationR37": IAMCCS_MiniMaxH3MotionContextGenerationR37,
    "IAMCCS_MiniMaxH3MotionContextStateCommitR37": IAMCCS_MiniMaxH3MotionContextStateCommitR37,
    "IAMCCS_MiniMaxH3BackendLazySwitchR37": IAMCCS_MiniMaxH3BackendLazySwitchR37,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_MiniMaxH3MotionContextConditionR37": "MiniMax H3 · Multi-Shot Upstream Motion Context",
    "IAMCCS_MiniMaxH3MotionContextGenerationR37": "MiniMax H3 · Multi-Shot Motion Context + ComfyKitchen",
    "IAMCCS_MiniMaxH3MotionContextStateCommitR37": "MiniMax H3 · Native AV Context Commit",
    "IAMCCS_MiniMaxH3BackendLazySwitchR37": "MiniMax H3 · Standard / Motion Context Lazy Switch",
}
