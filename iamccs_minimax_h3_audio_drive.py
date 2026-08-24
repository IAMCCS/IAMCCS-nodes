# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later

"""Atomic MiniMax H3 audio routing for the isolated R21 backend.

The node has one mutually-exclusive policy per Shotboard chunk:

* ``h3_native_generated`` leaves the joint AV latent untouched;
* ``h3_ref2va_audio`` leaves it untouched and exposes a chunk audio reference;
* ``h3_custom_audio_drive`` VAE-encodes the chunk audio and locks that audio
  stream with a zero noise mask before sampling;
* ``external_audio_post`` leaves the latent untouched and exposes an exact
  timeline-length audio slice for post-generation mux/concat.

No live IAMCCS module imports this staging file.  Registration is deliberately
self-contained so the file can be smoke-tested before an R21 integration.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from typing import Any

import torch
import torch.nn.functional as F


SUPERNODE_LINX_TYPE = "IAMCCS_SUPERNODE_LINX"
CATEGORY = "IAMCCS/MiniMax H3/Atomic Backend"
H3_FPS = 24

BEHAVIOR_NATIVE = "h3_native_generated"
BEHAVIOR_REFERENCE = "h3_ref2va_audio"
BEHAVIOR_LOCKED = "h3_custom_audio_drive"
BEHAVIOR_EXTERNAL = "external_audio_post"
BEHAVIORS = (
    BEHAVIOR_NATIVE,
    BEHAVIOR_REFERENCE,
    BEHAVIOR_LOCKED,
    BEHAVIOR_EXTERNAL,
)

_BEHAVIOR_ALIASES = {
    "native": BEHAVIOR_NATIVE,
    "native_generated": BEHAVIOR_NATIVE,
    BEHAVIOR_NATIVE: BEHAVIOR_NATIVE,
    "reference": BEHAVIOR_REFERENCE,
    "reference_only": BEHAVIOR_REFERENCE,
    "h3_ref2va_reference": BEHAVIOR_REFERENCE,
    BEHAVIOR_REFERENCE: BEHAVIOR_REFERENCE,
    "locked": BEHAVIOR_LOCKED,
    "locked_audio": BEHAVIOR_LOCKED,
    "locked_audio_drive": BEHAVIOR_LOCKED,
    "h3_locked_audio_drive": BEHAVIOR_LOCKED,
    "custom_audio": BEHAVIOR_LOCKED,
    "custom_audio_drive": BEHAVIOR_LOCKED,
    "audio_driven": BEHAVIOR_LOCKED,
    "force_audio_latent": BEHAVIOR_LOCKED,
    BEHAVIOR_LOCKED: BEHAVIOR_LOCKED,
    "external": BEHAVIOR_EXTERNAL,
    "external_post": BEHAVIOR_EXTERNAL,
    BEHAVIOR_EXTERNAL: BEHAVIOR_EXTERNAL,
}


def _resources(cine_linx: Any) -> dict[str, Any]:
    if not isinstance(cine_linx, dict):
        return {}
    resources = cine_linx.get("resources")
    return resources if isinstance(resources, dict) else {}


def _resolve_shotplan(value: Any) -> dict[str, Any]:
    """Resolve the private H3 plan from a direct plan or CineLinX envelope."""
    if isinstance(value, dict) and value.get("schema") == "iamccs.minimax_h3.shotplan":
        return value
    if not isinstance(value, dict):
        raise ValueError("MiniMax H3 Audio Drive requires a valid cine_linx input")

    resources = _resources(value)
    outputs = value.get("outputs") if isinstance(value.get("outputs"), dict) else {}
    payload = resources.get("cine_payload") if isinstance(resources.get("cine_payload"), dict) else {}
    for candidate in (
        resources.get("iamccs_minimax_h3_shotplan"),
        resources.get("minimax_h3_shotplan"),
        resources.get("shotplan"),
        outputs.get("shotplan"),
        payload.get("minimax_h3_shotplan"),
        payload.get("shotplan"),
    ):
        if isinstance(candidate, dict) and candidate.get("schema") == "iamccs.minimax_h3.shotplan":
            return candidate
    raise ValueError("CineLinX does not contain an IAMCCS MiniMax H3 shotplan")


def _resolve_chunk(cine_linx: Any, segment_index: int) -> tuple[dict[str, Any], dict[str, Any], int]:
    shotplan = _resolve_shotplan(cine_linx)
    chunks = shotplan.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("MiniMax H3 shotplan has no chunks")
    index = int(segment_index)
    if index < 0 or index >= len(chunks):
        raise IndexError(f"segment_index={index} outside 0..{len(chunks) - 1}")
    chunk = chunks[index]
    if not isinstance(chunk, dict):
        raise ValueError(f"MiniMax H3 chunk {index} is not a dictionary")
    return shotplan, chunk, index


def _effective_task(cine_linx: Any, chunk: dict[str, Any]) -> str:
    config = _resources(cine_linx).get("iamccs_minimax_h3_cine_info")
    if isinstance(config, dict):
        override = str(config.get("task_override", "from_shotboard") or "from_shotboard").lower()
        if override in {"t2va", "i2va", "fl2va", "ref2va"}:
            return override
    return str(chunk.get("task_mode", "t2va") or "t2va").lower()


def _canonical_behavior(value: Any, *, allow_from_plan: bool = False) -> str:
    clean = str(value or "").strip().lower()
    if allow_from_plan and clean in {"", "auto", "from_cine_linx", "from_shotboard"}:
        return "from_cine_linx"
    behavior = _BEHAVIOR_ALIASES.get(clean)
    if behavior is None:
        raise ValueError(
            f"Unsupported MiniMax H3 audio behavior '{value}'. "
            f"Expected one of: {', '.join(BEHAVIORS)}"
        )
    return behavior


def _resolve_behavior(cine_linx: Any, requested: Any) -> tuple[str, str, bool]:
    """Resolve a policy and reject contradictory non-native plan overrides.

    R20 plans do not yet contain the locked value.  An explicit R21 locked or
    reference request is therefore allowed to migrate a legacy native plan,
    while two different active non-native policies are treated as a conflict.
    """
    shotplan = _resolve_shotplan(cine_linx)
    plan_behavior = _canonical_behavior(shotplan.get("audio_mode", BEHAVIOR_NATIVE))
    requested_behavior = _canonical_behavior(requested, allow_from_plan=True)
    if requested_behavior == "from_cine_linx":
        return plan_behavior, plan_behavior, False
    if plan_behavior != BEHAVIOR_NATIVE and requested_behavior != plan_behavior:
        raise ValueError(
            "Conflicting MiniMax H3 audio policies: "
            f"cine_linx requests '{plan_behavior}', node requests '{requested_behavior}'. "
            "Select from_cine_linx or make the Shotboard and Audio Drive policies agree."
        )
    migrated_legacy_native = plan_behavior == BEHAVIOR_NATIVE and requested_behavior != plan_behavior
    return requested_behavior, plan_behavior, migrated_legacy_native


def _is_audio(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and torch.is_tensor(value.get("waveform"))
        and value["waveform"].ndim == 3
    )


def _resource_audio(cine_linx: Any, behavior: str) -> tuple[dict[str, Any] | None, str]:
    resources = _resources(cine_linx)
    if behavior == BEHAVIOR_REFERENCE:
        keys = (
            "iamccs_minimax_h3_ref_audio",
            "iamccs_minimax_h3_ref_video_audio",
        )
    else:
        keys = (
            "iamccs_minimax_h3_custom_audio",
        )
    for key in keys:
        value = resources.get(key)
        if _is_audio(value):
            return value, f"cine_linx.resources.{key}"
    return None, "none"


def _select_audio_source(
    cine_linx: Any,
    behavior: str,
    source_audio: Any,
    segment_index: int,
) -> tuple[dict[str, Any] | None, str]:
    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    # The timeline mixer has already applied trims and rebased this exact chunk
    # on H3's aligned clock. Prefer it over a second slice from the master.
    if behavior == BEHAVIOR_LOCKED:
        selected = _resources(cine_linx).get("iamccs_minimax_h3_chunk_audio")
        selected_meta = selected.get("_iamccs") if isinstance(selected, Mapping) else None
        if (
            _is_audio(selected)
            and bool(selected.get("iamccs_pre_sliced", False))
            and isinstance(selected_meta, Mapping)
            and int(selected_meta.get("chunk_index", -1)) == int(segment_index)
        ):
            return selected, "cine_linx.resources.iamccs_minimax_h3_chunk_audio"
    if source_audio is not None:
        if not _is_audio(source_audio):
            raise ValueError("source_audio must be a ComfyUI AUDIO dictionary with waveform [B,C,S]")
        return source_audio, "source_audio socket"
    return _resource_audio(cine_linx, behavior)


def _chunk_timing(shotplan: dict[str, Any], chunk: dict[str, Any]) -> tuple[float, float, int]:
    fps = max(1, int(shotplan.get("fps", H3_FPS) or H3_FPS))
    start_seconds = float(
        chunk.get("timeline_start_seconds", chunk.get("start_seconds", 0.0)) or 0.0
    )
    default_duration = max(1, int(chunk.get("frame_count", 124) or 124)) / fps
    duration_seconds = float(chunk.get("duration_seconds", default_duration) or default_duration)
    if not math.isfinite(start_seconds) or start_seconds < 0.0:
        raise ValueError(f"Invalid MiniMax H3 chunk start time: {start_seconds}")
    if not math.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise ValueError(f"Invalid MiniMax H3 chunk duration: {duration_seconds}")
    return start_seconds, duration_seconds, fps


def _slice_audio_for_chunk(
    audio: dict[str, Any],
    start_seconds: float,
    duration_seconds: float,
    *,
    pad_to_duration: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Slice a timeline-master AUDIO and optionally silence-pad the chunk.

    Locked custom audio is a timing authority, including its trailing silence:
    it must cover the complete legal H3 chunk before VAE encoding.  Reference
    audio keeps its real available length because it is conditioning rather
    than a latent lock.  External-post audio is also padded because it bypasses
    the H3 audio sampler entirely.
    """
    if not _is_audio(audio):
        raise ValueError("MiniMax H3 audio source must contain waveform [B,C,S]")
    waveform = audio["waveform"]
    sample_rate = int(audio.get("sample_rate", 32000) or 32000)
    if sample_rate <= 0:
        raise ValueError(f"Invalid AUDIO sample_rate={sample_rate}")

    start_sample = max(0, int(round(start_seconds * sample_rate)))
    requested_samples = max(1, int(round(duration_seconds * sample_rate)))
    if start_sample >= int(waveform.shape[-1]):
        raise ValueError(
            "Audio source does not cover the selected MiniMax H3 chunk: "
            f"chunk starts at {start_seconds:.3f}s, source ends at "
            f"{int(waveform.shape[-1]) / sample_rate:.3f}s"
        )
    end_sample = min(int(waveform.shape[-1]), start_sample + requested_samples)
    sliced = waveform[:1, :, start_sample:end_sample]
    available_samples = int(sliced.shape[-1])
    padded_samples = 0
    if pad_to_duration and available_samples < requested_samples:
        padded_samples = requested_samples - available_samples
        sliced = F.pad(sliced, (0, padded_samples))

    result = {
        "waveform": sliced,
        "sample_rate": sample_rate,
    }
    metadata = {
        "start_seconds": start_seconds,
        "duration_seconds": duration_seconds,
        "sample_rate": sample_rate,
        "start_sample": start_sample,
        "requested_samples": requested_samples,
        "source_samples": available_samples,
        "output_samples": int(sliced.shape[-1]),
        "padded_samples": padded_samples,
    }
    return result, metadata


def _normalize_audio_channels(
    audio: dict[str, Any],
    target_channels: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Match a waveform to the native H3 audio latent channel count.

    MiniMax H3 uses a stereo latent.  ``VAEEncodeAudio`` preserves the source
    waveform channel count, so a mono source would otherwise encode to
    ``[B, 32, 1, T]``.  Core ``LTXVConcatAVLatent.fit_audio`` then pads the
    missing stereo lane with a *generative* mask, leaving exactly half of the
    channel-major H3 audio token block unlocked.  Promote mono to dual-mono
    before encoding so both latent lanes are supplied and locked.

    The general rules are deterministic for defensive compatibility:

    * fewer channels repeat cyclically up to the target count;
    * more channels are split into contiguous groups and averaged;
    * an exact match is passed through unchanged.
    """
    if not _is_audio(audio):
        raise ValueError("MiniMax H3 audio channel normalization requires a valid AUDIO dictionary")
    target_channels = int(target_channels)
    if target_channels < 1:
        raise ValueError(f"Invalid native MiniMax H3 audio channel count: {target_channels}")

    waveform = audio["waveform"]
    source_channels = int(waveform.shape[1])
    if source_channels < 1:
        raise ValueError("MiniMax H3 audio source has no waveform channels")

    if source_channels == target_channels:
        normalized = waveform
        action = "identity"
    elif source_channels < target_channels:
        repeats = int(math.ceil(target_channels / source_channels))
        normalized = waveform.repeat(1, repeats, 1)[:, :target_channels, :]
        action = "mono_to_dual_mono" if source_channels == 1 and target_channels == 2 else "cyclic_repeat"
    else:
        # Contiguous group averaging preserves channel order and gives every
        # source channel exactly one deterministic destination.
        groups = []
        for index in range(target_channels):
            start = (index * source_channels) // target_channels
            stop = ((index + 1) * source_channels) // target_channels
            stop = max(start + 1, stop)
            groups.append(waveform[:, start:stop, :].mean(dim=1, keepdim=True))
        normalized = torch.cat(groups, dim=1)
        action = "contiguous_group_downmix"

    result = dict(audio)
    result["waveform"] = normalized
    return result, {
        "source_channels": source_channels,
        "target_channels": target_channels,
        "output_channels": int(normalized.shape[1]),
        "action": action,
    }


def _validate_joint_av_latent(av_latent: Any) -> tuple[Any, Any]:
    if not isinstance(av_latent, dict):
        raise ValueError("Locked MiniMax H3 audio drive requires a LATENT dictionary")
    samples = av_latent.get("samples")
    if samples is None or not bool(getattr(samples, "is_nested", False)):
        raise ValueError(
            "Locked MiniMax H3 audio drive requires a joint two-stream AV latent; "
            "connect the LATENT output of MiniMaxH3ImageToVideo or MiniMaxH3ReferenceToVideo"
        )
    streams = samples.unbind()
    if len(streams) != 2:
        raise ValueError(f"Expected two AV latent streams (video, audio), found {len(streams)}")
    return streams[0], streams[1]


def _first(result: Any) -> Any:
    try:
        return result[0]
    except (TypeError, IndexError, KeyError) as exc:
        raise RuntimeError("Unexpected ComfyUI core-node return value") from exc


def _lock_audio_stream(
    av_latent: dict[str, Any],
    chunk_audio: dict[str, Any],
    audio_vae: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replace the AV audio stream through ComfyUI's supported core nodes."""
    _validate_joint_av_latent(av_latent)
    if audio_vae is None:
        raise ValueError("h3_custom_audio_drive requires the MiniMax H3 audio VAE")

    from comfy_extras.nodes_audio import VAEEncodeAudio
    from comfy_extras.nodes_lt import LTXVConcatAVLatent, LTXVSeparateAVLatent

    separated = LTXVSeparateAVLatent.execute(av_latent=av_latent)
    video_latent = separated[0]
    native_audio_latent = separated[1]
    native_audio_samples = native_audio_latent.get("samples")
    if not torch.is_tensor(native_audio_samples) or native_audio_samples.ndim != 4:
        raise ValueError("MiniMax H3 native audio stream must have shape [B,C,channels,T]")
    target_channels = int(native_audio_samples.shape[2])
    normalized_audio, channel_report = _normalize_audio_channels(chunk_audio, target_channels)

    encoded_audio = _first(VAEEncodeAudio.execute(vae=audio_vae, audio=normalized_audio))
    encoded_samples = encoded_audio.get("samples") if isinstance(encoded_audio, dict) else None
    if not torch.is_tensor(encoded_samples) or encoded_samples.ndim != 4:
        raise RuntimeError("VAEEncodeAudio did not return an audio latent")
    encoded_channels = int(encoded_samples.shape[2])
    if encoded_channels != target_channels:
        raise RuntimeError(
            "MiniMax H3 custom audio channel normalization failed: "
            f"encoded latent has {encoded_channels} channel lane(s), native latent requires {target_channels}"
        )

    # Use an exact audio-latent mask rather than a generic 32x32 image mask.
    # The chunk was silence-padded before encoding, so every encoded sample --
    # including silence -- is authoritative and must remain noise-free.
    locked_audio = dict(encoded_audio)
    locked_audio["noise_mask"] = torch.zeros_like(encoded_samples, dtype=torch.float32)

    # Concat's automatic fit is only entered when its first input is still a
    # nested AV latent. Because this graph explicitly separates first, fit
    # against the separated native audio stream here, then concatenate the two
    # plain streams. The H3 target length remains authoritative.  Any tiny
    # codec-grid shortfall is zero-padded by the core fit and then locked as
    # silence; custom dialogue must never acquire a generative tail.
    fitted_samples, fitted_mask = LTXVConcatAVLatent.fit_audio(
        native_audio_samples,
        locked_audio["samples"],
        locked_audio.get("noise_mask"),
    )
    if tuple(fitted_samples.shape) != tuple(native_audio_samples.shape):
        raise RuntimeError(
            "MiniMax H3 custom audio fit produced the wrong latent shape: "
            f"{tuple(fitted_samples.shape)} != {tuple(native_audio_samples.shape)}"
        )
    # ``fit_audio`` intentionally marks a padded tail as generative for its
    # generic LTX use case. Locked H3 dialogue has already been padded with
    # silence, so override that generic policy and lock the entire fitted grid.
    fitted_mask = torch.zeros_like(fitted_samples, dtype=torch.float32)
    locked_audio = dict(locked_audio)
    locked_audio["samples"] = fitted_samples
    locked_audio["noise_mask"] = fitted_mask

    locked_av = _first(
        LTXVConcatAVLatent.execute(
            video_latent=video_latent,
            audio_latent=locked_audio,
        )
    )
    video_stream, audio_stream = _validate_joint_av_latent(locked_av)
    joint_mask = locked_av.get("noise_mask") if isinstance(locked_av, dict) else None
    if joint_mask is None or not bool(getattr(joint_mask, "is_nested", False)):
        raise RuntimeError("MiniMax H3 custom audio lock lost its joint AV noise mask")
    mask_streams = joint_mask.unbind()
    if len(mask_streams) != 2 or tuple(mask_streams[1].shape) != tuple(audio_stream.shape):
        raise RuntimeError(
            "MiniMax H3 custom audio mask shape does not match the final audio stream: "
            f"{tuple(mask_streams[1].shape) if len(mask_streams) > 1 else 'missing'} "
            f"!= {tuple(audio_stream.shape)}"
        )
    final_audio_mask = mask_streams[1]
    unlocked_tokens = int(torch.count_nonzero(final_audio_mask > 0).item())
    total_tokens = int(final_audio_mask.numel())
    locked_fraction = 1.0 - (unlocked_tokens / total_tokens if total_tokens else 0.0)
    if unlocked_tokens != 0:
        raise RuntimeError(
            "MiniMax H3 custom audio lock is incomplete: "
            f"{unlocked_tokens}/{total_tokens} final latent values remain generative"
        )
    metadata = {
        "video_latent_shape": list(video_stream.shape),
        "source_audio_latent_shape": list(encoded_samples.shape),
        "fitted_audio_latent_shape": list(audio_stream.shape),
        "source_audio_channels": channel_report["source_channels"],
        "target_audio_channels": channel_report["target_channels"],
        "encoded_audio_channels": encoded_channels,
        "channel_normalization": channel_report["action"],
        "zero_noise_mask": True,
        "locked_fraction": locked_fraction,
        "unlocked_latent_values": unlocked_tokens,
        "fit_rule": "trim overlong; lock silence-padded shortfall",
    }
    return locked_av, metadata


class IAMCCS_MiniMaxH3AtomicAudioDrive:
    """Exclusive native/reference/locked/post audio router for one H3 chunk."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "av_latent": ("LATENT",),
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "segment_index": ("INT", {"forceInput": True}),
                "audio_behavior": (
                    ["from_cine_linx", *BEHAVIORS],
                    {"default": "from_cine_linx"},
                ),
            },
            "optional": {
                "source_audio": ("AUDIO",),
                "audio_vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("LATENT", "AUDIO", "AUDIO", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = (
        "latent",
        "reference_audio",
        "locked_audio_slice",
        "external_post_audio",
        "resolved_behavior",
        "report",
    )
    FUNCTION = "route"
    CATEGORY = CATEGORY

    def route(
        self,
        av_latent,
        cine_linx,
        segment_index,
        audio_behavior,
        source_audio=None,
        audio_vae=None,
    ):
        shotplan, chunk, index = _resolve_chunk(cine_linx, segment_index)
        behavior, plan_behavior, migrated = _resolve_behavior(cine_linx, audio_behavior)
        task = _effective_task(cine_linx, chunk)
        shotboard_mode = str(shotplan.get("task_mode", "") or "").strip().lower()
        lipsync_contract = shotplan.get("lipsync") if isinstance(shotplan.get("lipsync"), dict) else {}
        lipsync_requested = bool(lipsync_contract.get("enabled", False)) or shotboard_mode in {
            "ref2vid_lipsync", "lipsync_ref2vid", "longvid_ref2vid_lipsync",
            "longvid_guided_lipsync", "longvid_audio_drive_lipsync",
        }
        if lipsync_requested and behavior != BEHAVIOR_LOCKED:
            raise ValueError(
                "MiniMax H3 LipSync requires h3_custom_audio_drive so the AudioBoard chunk is the exact pre-sampler audio authority. "
                "Select the LipSync mode from IAMCCS settings; it sets this route automatically."
            )
        guided_audio_drive = shotboard_mode == "longvid_guides" and behavior == BEHAVIOR_LOCKED
        start_seconds, duration_seconds, fps = _chunk_timing(shotplan, chunk)

        report: dict[str, Any] = {
            "schema": "iamccs.minimax_h3.audio_drive.r21",
            "behavior": behavior,
            "plan_behavior": plan_behavior,
            "legacy_native_override": migrated,
            "task": task,
            "shotboard_mode": shotboard_mode,
            "lipsync_audio_authority": lipsync_requested or guided_audio_drive,
            "guided_audio_drive": guided_audio_drive,
            "segment_index": index,
            "segment_number": index + 1,
            "total_segments": len(shotplan.get("chunks", [])),
            "timeline": {
                "start_seconds": start_seconds,
                "duration_seconds": duration_seconds,
                "fps": fps,
            },
            "latent_passthrough": behavior != BEHAVIOR_LOCKED,
            "audio_source": "none",
        }

        if behavior == BEHAVIOR_NATIVE:
            report["action"] = "native H3 AV latent passed unchanged"
            return av_latent, None, None, None, behavior, json.dumps(report, ensure_ascii=False, indent=2)

        audio, source_name = _select_audio_source(cine_linx, behavior, source_audio, index)
        if audio is None:
            if behavior == BEHAVIOR_REFERENCE:
                # REF audio is optional for video-only V2V sources. The raw
                # video and picture references remain valid REF2VA inputs, so
                # keep the untouched joint latent and let H3 generate audio.
                report["action"] = "REF2VA source has no audio stream; native H3 audio fallback"
                report["audio_source"] = "none_video_only_fallback"
                report["fallback"] = "h3_native_generated_audio"
                return av_latent, None, None, None, behavior, json.dumps(report, ensure_ascii=False, indent=2)
            raise ValueError(
                f"MiniMax H3 audio behavior '{behavior}' requires source_audio or an AUDIO resource from IAMCCS CineInfo H3"
            )
        report["audio_source"] = source_name

        pre_sliced = bool(audio.get("iamccs_pre_sliced", False))
        chunk_audio, slice_report = _slice_audio_for_chunk(
            audio,
            0.0 if pre_sliced else start_seconds,
            duration_seconds,
            pad_to_duration=behavior in {BEHAVIOR_LOCKED, BEHAVIOR_EXTERNAL},
        )
        slice_report["iamccs_pre_sliced"] = pre_sliced
        if pre_sliced:
            slice_report["source_start_seconds"] = float(audio.get("iamccs_source_start_seconds", start_seconds))
            slice_report["timeline_start_bypassed"] = start_seconds
        report["slice"] = slice_report

        if behavior == BEHAVIOR_REFERENCE:
            if not task.startswith("ref2va"):
                raise ValueError(
                    "h3_ref2va_audio is a reference-conditioning policy and requires a REF2VA task; "
                    f"the active chunk task is '{task}'"
                )
            report["action"] = "AV latent passed unchanged; chunk audio routed as REF2VA reference"
            return av_latent, chunk_audio, None, None, behavior, json.dumps(report, ensure_ascii=False, indent=2)

        if behavior == BEHAVIOR_LOCKED:
            locked_latent, lock_report = _lock_audio_stream(av_latent, chunk_audio, audio_vae)
            report["action"] = "audio stream VAE-encoded and locked before H3 sampling"
            report["latent_passthrough"] = False
            report["lock"] = lock_report
            return locked_latent, None, chunk_audio, None, behavior, json.dumps(report, ensure_ascii=False, indent=2)

        if behavior == BEHAVIOR_EXTERNAL:
            report["action"] = "AV latent passed unchanged; exact chunk audio routed after H3"
            return av_latent, None, None, chunk_audio, behavior, json.dumps(report, ensure_ascii=False, indent=2)

        raise AssertionError(f"Unhandled MiniMax H3 audio behavior: {behavior}")


def _fit_audio_to_duration(audio: Any, duration_seconds: float) -> tuple[dict[str, Any], dict[str, Any]]:
    if not _is_audio(audio):
        raise ValueError("MiniMax H3 audio output must be a ComfyUI AUDIO dictionary")
    sample_rate = max(1, int(audio.get("sample_rate", 32000) or 32000))
    requested_samples = max(1, int(round(max(0.0, float(duration_seconds)) * sample_rate)))
    waveform = audio["waveform"]
    before = int(waveform.shape[-1])
    if before > requested_samples:
        waveform = waveform[..., :requested_samples]
        action = "trim"
    elif before < requested_samples:
        waveform = F.pad(waveform, (0, requested_samples - before))
        action = "silence_pad"
    else:
        action = "identity"
    return {
        "waveform": waveform,
        "sample_rate": sample_rate,
    }, {
        "action": action,
        "before_samples": before,
        "after_samples": int(waveform.shape[-1]),
        "sample_rate": sample_rate,
        "duration_seconds": float(duration_seconds),
    }


class IAMCCS_MiniMaxH3AudioOutputPolicyR21:
    """Select the soundtrack that is actually written for one rendered chunk.

    Native and REF2VA-reference modes keep H3's decoded audio. Custom-drive and
    external-post modes use the original source slice, avoiding an unnecessary
    lossy AudioVAE round trip in the saved master.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "h3_audio": ("AUDIO",),
                "video_frames": ("IMAGE",),
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "segment_index": ("INT", {"forceInput": True}),
            },
            "optional": {
                "locked_audio_slice": ("AUDIO",),
                "external_post_audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("master_audio", "resolved_behavior", "report")
    FUNCTION = "select"
    CATEGORY = CATEGORY

    def select(
        self,
        h3_audio,
        video_frames,
        cine_linx,
        segment_index,
        locked_audio_slice=None,
        external_post_audio=None,
    ):
        shotplan, chunk, index = _resolve_chunk(cine_linx, segment_index)
        behavior, _, _ = _resolve_behavior(cine_linx, "from_cine_linx")
        fps = max(1, int(shotplan.get("fps", H3_FPS) or H3_FPS))
        if not torch.is_tensor(video_frames) or video_frames.ndim != 4 or int(video_frames.shape[0]) < 1:
            raise ValueError("MiniMax H3 Audio Output Policy requires non-empty video_frames")
        duration_seconds = int(video_frames.shape[0]) / float(fps)

        source_name = "h3_generated_audio"
        selected = h3_audio
        if behavior == BEHAVIOR_LOCKED:
            if not _is_audio(locked_audio_slice):
                raise ValueError("Custom audio drive is active but locked_audio_slice is not connected")
            selected = locked_audio_slice
            source_name = "original_custom_audio_slice"
        elif behavior == BEHAVIOR_EXTERNAL:
            if not _is_audio(external_post_audio):
                raise ValueError("External audio post is active but external_post_audio is not connected")
            selected = external_post_audio
            source_name = "external_post_audio_slice"

        fitted, fit_report = _fit_audio_to_duration(selected, duration_seconds)
        report = {
            "schema": "iamccs.minimax_h3.audio_output.r21",
            "behavior": behavior,
            "source": source_name,
            "segment_index": index,
            "segment_number": index + 1,
            "video_frames": int(video_frames.shape[0]),
            "fps": fps,
            "fit": fit_report,
            "note": (
                "REF2VA reference audio remains generative; custom-drive/external-post save the original source slice"
            ),
        }
        return fitted, behavior, json.dumps(report, ensure_ascii=False, indent=2)


class IAMCCS_MiniMaxH3ExactDeliveryDurationR21:
    """Restore the native programme duration after an 8n+1 LTX round trip."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "delivery_frames": ("IMAGE",),
                "audio": ("AUDIO",),
                "native_reference_frames": ("IMAGE",),
                "delivery_fps": ("INT", {"forceInput": True}),
            },
            "optional": {
                "native_fps": ("INT", {"default": H3_FPS, "min": 1, "max": 240, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("delivery_frames", "audio", "report")
    FUNCTION = "fit"
    CATEGORY = CATEGORY

    def fit(self, delivery_frames, audio, native_reference_frames, delivery_fps, native_fps=H3_FPS):
        if not torch.is_tensor(delivery_frames) or delivery_frames.ndim != 4 or int(delivery_frames.shape[0]) < 1:
            raise ValueError("Exact Delivery Duration requires non-empty delivery_frames")
        if not torch.is_tensor(native_reference_frames) or native_reference_frames.ndim != 4 or int(native_reference_frames.shape[0]) < 1:
            raise ValueError("Exact Delivery Duration requires non-empty native_reference_frames")
        delivery_fps = max(1, int(delivery_fps))
        native_fps = max(1, int(native_fps))
        native_count = int(native_reference_frames.shape[0])
        target_count = max(1, int(round(native_count * delivery_fps / float(native_fps))))
        before = int(delivery_frames.shape[0])
        if before > target_count:
            frames = delivery_frames[:target_count]
            action = "crop_tail"
        elif before < target_count:
            hold = delivery_frames[-1:].repeat(target_count - before, 1, 1, 1)
            frames = torch.cat([delivery_frames, hold], dim=0)
            action = "hold_last_frame"
        else:
            frames = delivery_frames
            action = "identity"
        duration_seconds = target_count / float(delivery_fps)
        fitted_audio, audio_report = _fit_audio_to_duration(audio, duration_seconds)
        report = {
            "schema": "iamccs.minimax_h3.exact_delivery_duration.r21",
            "native_frames": native_count,
            "native_fps": native_fps,
            "delivery_frames_before": before,
            "delivery_frames_after": int(frames.shape[0]),
            "delivery_fps": delivery_fps,
            "duration_seconds": duration_seconds,
            "video_action": action,
            "audio": audio_report,
        }
        return frames, fitted_audio, json.dumps(report, ensure_ascii=False, indent=2)


NODE_CLASS_MAPPINGS = {
    "IAMCCS_MiniMaxH3AtomicAudioDrive": IAMCCS_MiniMaxH3AtomicAudioDrive,
    "IAMCCS_MiniMaxH3AudioOutputPolicyR21": IAMCCS_MiniMaxH3AudioOutputPolicyR21,
    "IAMCCS_MiniMaxH3ExactDeliveryDurationR21": IAMCCS_MiniMaxH3ExactDeliveryDurationR21,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_MiniMaxH3AtomicAudioDrive": "MiniMax H3 Atomic Audio Drive (R21)",
    "IAMCCS_MiniMaxH3AudioOutputPolicyR21": "MiniMax H3 Audio Output Policy (R21)",
    "IAMCCS_MiniMaxH3ExactDeliveryDurationR21": "MiniMax H3 Exact Delivery Duration (R21)",
}
