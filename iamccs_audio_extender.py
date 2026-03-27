from __future__ import annotations

import importlib.util
import logging
import os
import sys
from collections.abc import Mapping
from typing import Any, Dict, Tuple

import torch


_log = logging.getLogger("IAMCCS.Audio")


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_typ = AnyType("*")


_UNIFIED_TTS_TEXT_NODE_CLASS = None


def _find_tts_audio_suite_dir() -> str:
    custom_nodes_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates = [
        os.path.join(custom_nodes_dir, "TTS-Audio-Suite"),
        os.path.join(custom_nodes_dir, "tts_audio_suite"),
    ]
    for candidate in candidates:
        module_path = os.path.join(candidate, "nodes", "unified", "tts_text_node.py")
        if os.path.isfile(module_path):
            return candidate
    raise FileNotFoundError(
        "TTS Audio Suite not found. Expected one of: "
        f"{', '.join(candidates)}"
    )


def _load_unified_tts_text_node_class():
    global _UNIFIED_TTS_TEXT_NODE_CLASS
    if _UNIFIED_TTS_TEXT_NODE_CLASS is not None:
        return _UNIFIED_TTS_TEXT_NODE_CLASS

    suite_dir = _find_tts_audio_suite_dir()
    module_path = os.path.join(suite_dir, "nodes", "unified", "tts_text_node.py")
    module_name = "iamccs_tts_audio_suite_unified_tts_text_node"

    if suite_dir not in sys.path:
        sys.path.insert(0, suite_dir)

    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load UnifiedTTSTextNode from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    if not hasattr(module, "UnifiedTTSTextNode"):
        raise ImportError("UnifiedTTSTextNode class not found in TTS Audio Suite")

    _UNIFIED_TTS_TEXT_NODE_CLASS = module.UnifiedTTSTextNode
    return _UNIFIED_TTS_TEXT_NODE_CLASS


def _unwrap_audio(audio: Any) -> Any:
    current = audio
    for _ in range(6):
        if current is None:
            return None
        if isinstance(current, Mapping):
            if "waveform" in current and "sample_rate" in current:
                return {
                    "waveform": current["waveform"],
                    "sample_rate": current["sample_rate"],
                }
            if "audio" in current:
                current = current.get("audio")
                continue
            if len(current) == 1:
                current = next(iter(current.values()))
                continue
            return current
        if isinstance(current, (list, tuple)):
            if not current:
                return None
            current = current[0]
            continue
        if hasattr(current, "waveform") and hasattr(current, "sample_rate"):
            return {
                "waveform": getattr(current, "waveform"),
                "sample_rate": getattr(current, "sample_rate"),
            }
        return current
    return current


def _normalize_waveform(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 3:
        return waveform
    if waveform.ndim == 2:
        return waveform.unsqueeze(0)
    if waveform.ndim == 1:
        return waveform.view(1, 1, -1)
    raise ValueError(f"Unsupported AUDIO waveform rank: {waveform.ndim}")


def _validate_audio(audio: Dict[str, Any]) -> Tuple[torch.Tensor, int]:
    audio = _unwrap_audio(audio)
    if not isinstance(audio, dict):
        raise ValueError("audio must be an AUDIO dict")
    if "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("audio must contain 'waveform' and 'sample_rate'")
    waveform = audio["waveform"]
    if not torch.is_tensor(waveform):
        raise ValueError("audio['waveform'] must be a tensor")
    waveform = _normalize_waveform(waveform)
    sample_rate = int(audio["sample_rate"])
    if sample_rate <= 0:
        raise ValueError("audio sample_rate must be > 0")
    if waveform.shape[-1] <= 0:
        raise ValueError("audio waveform is empty")
    return waveform, sample_rate


def _clamp_slice(start_sample: int, end_sample: int, total_samples: int) -> Tuple[int, int, bool]:
    orig_start = int(start_sample)
    orig_end = int(end_sample)
    start_sample = max(0, min(int(start_sample), total_samples))
    end_sample = max(start_sample, min(int(end_sample), total_samples))
    return start_sample, end_sample, (start_sample != orig_start or end_sample != orig_end)


def _audio_dict_from_slice(audio: Dict[str, Any], waveform_3d: torch.Tensor, start_sample: int, end_sample: int) -> Dict[str, Any]:
    out = dict(audio)
    out["waveform"] = waveform_3d[:, :, start_sample:end_sample]
    return out


def _clone_audio_dict(audio: Dict[str, Any], waveform_3d: torch.Tensor, sample_rate: int) -> Dict[str, Any]:
    out = dict(audio)
    out["waveform"] = waveform_3d
    out["sample_rate"] = int(sample_rate)
    return out


def _make_silent_audio(sample_rate: int, total_samples: int, channels: int = 1, dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
    waveform = torch.zeros((1, max(1, int(channels)), max(0, int(total_samples))), dtype=dtype)
    return {
        "waveform": waveform,
        "sample_rate": int(sample_rate),
    }


def _ensure_waveform_channels(waveform: torch.Tensor, channels: int) -> torch.Tensor:
    current_channels = int(waveform.shape[1])
    if current_channels == channels:
        return waveform
    if current_channels == 1 and channels > 1:
        return waveform.repeat(1, channels, 1)
    if channels == 1 and current_channels > 1:
        return waveform.mean(dim=1, keepdim=True)
    if current_channels > channels:
        return waveform[:, :channels, :]
    pad = waveform[:, -1:, :].repeat(1, channels - current_channels, 1)
    return torch.cat([waveform, pad], dim=1)


def _pad_audio_to_samples(audio: Dict[str, Any], total_samples: int) -> Dict[str, Any]:
    waveform, sample_rate = _validate_audio(audio)
    target_samples = max(0, int(total_samples))
    current_samples = int(waveform.shape[-1])
    if current_samples >= target_samples:
        return _clone_audio_dict(audio, waveform[:, :, :target_samples], sample_rate)
    pad = torch.zeros(
        (waveform.shape[0], waveform.shape[1], target_samples - current_samples),
        dtype=waveform.dtype,
        device=waveform.device,
    )
    return _clone_audio_dict(audio, torch.cat([waveform, pad], dim=-1), sample_rate)


def _overlay_audio_segment(
    timeline_audio: Dict[str, Any] | None,
    segment_audio: Dict[str, Any],
    start_sample: int,
    insert_mode: str,
    crossfade_samples: int,
) -> Dict[str, Any]:
    segment_waveform, sample_rate = _validate_audio(segment_audio)
    segment_waveform = segment_waveform.detach().clone()
    segment_len = int(segment_waveform.shape[-1])
    if segment_len <= 0:
        raise ValueError("segment_audio is empty")

    start_sample = max(0, int(start_sample))
    end_sample = start_sample + segment_len

    if timeline_audio is None:
        base_template = dict(segment_audio)
        timeline_waveform = torch.zeros(
            (1, int(segment_waveform.shape[1]), end_sample),
            dtype=segment_waveform.dtype,
            device=segment_waveform.device,
        )
    else:
        timeline_waveform, timeline_sample_rate = _validate_audio(timeline_audio)
        if int(timeline_sample_rate) != int(sample_rate):
            raise ValueError(
                f"timeline sample_rate ({timeline_sample_rate}) does not match segment sample_rate ({sample_rate})"
            )
        base_template = dict(timeline_audio)
        timeline_waveform = timeline_waveform.detach().clone()

    channels = max(int(timeline_waveform.shape[1]), int(segment_waveform.shape[1]))
    timeline_waveform = _ensure_waveform_channels(timeline_waveform, channels)
    segment_waveform = _ensure_waveform_channels(segment_waveform, channels)

    current_total = int(timeline_waveform.shape[-1])
    if end_sample > current_total:
        pad = torch.zeros(
            (1, channels, end_sample - current_total),
            dtype=timeline_waveform.dtype,
            device=timeline_waveform.device,
        )
        timeline_waveform = torch.cat([timeline_waveform, pad], dim=-1)

    updated = timeline_waveform.clone()

    if str(insert_mode) == "replace":
        updated[:, :, start_sample:end_sample] = segment_waveform
    elif str(insert_mode) == "crossfade":
        updated[:, :, start_sample:end_sample] = segment_waveform
        overlap_available = max(0, min(int(crossfade_samples), segment_len, max(0, current_total - start_sample)))
        if overlap_available > 0:
            fade = torch.linspace(0.0, 1.0, steps=overlap_available, dtype=updated.dtype, device=updated.device).view(1, 1, overlap_available)
            existing = timeline_waveform[:, :, start_sample:start_sample + overlap_available]
            incoming = segment_waveform[:, :, :overlap_available]
            updated[:, :, start_sample:start_sample + overlap_available] = existing * (1.0 - fade) + incoming * fade
    else:
        updated[:, :, start_sample:end_sample] += segment_waveform

    return _clone_audio_dict(base_template, updated, sample_rate)


def _mix_audio_tracks(audio_a: Dict[str, Any], audio_b: Dict[str, Any], normalize: bool) -> Dict[str, Any]:
    waveform_a, sample_rate_a = _validate_audio(audio_a)
    waveform_b, sample_rate_b = _validate_audio(audio_b)
    if int(sample_rate_a) != int(sample_rate_b):
        raise ValueError(
            f"speaker sample rates do not match ({sample_rate_a} vs {sample_rate_b})"
        )

    total_samples = max(int(waveform_a.shape[-1]), int(waveform_b.shape[-1]))
    channels = max(int(waveform_a.shape[1]), int(waveform_b.shape[1]))
    waveform_a = _ensure_waveform_channels(waveform_a, channels)
    waveform_b = _ensure_waveform_channels(waveform_b, channels)

    if int(waveform_a.shape[-1]) < total_samples:
        waveform_a = torch.cat(
            [
                waveform_a,
                torch.zeros((1, channels, total_samples - int(waveform_a.shape[-1])), dtype=waveform_a.dtype, device=waveform_a.device),
            ],
            dim=-1,
        )
    if int(waveform_b.shape[-1]) < total_samples:
        waveform_b = torch.cat(
            [
                waveform_b,
                torch.zeros((1, channels, total_samples - int(waveform_b.shape[-1])), dtype=waveform_b.dtype, device=waveform_b.device),
            ],
            dim=-1,
        )

    mixed = waveform_a + waveform_b
    if bool(normalize):
        peak = float(mixed.abs().max().item()) if mixed.numel() > 0 else 0.0
        if peak > 1.0:
            mixed = mixed / peak

    return _clone_audio_dict(audio_a, mixed, sample_rate_a)


def _normalize_speaker_token(token: str) -> str:
    return str(token).strip().lower().replace(" ", "")


def _parse_aliases(raw_aliases: str) -> set[str]:
    aliases = set()
    for part in str(raw_aliases or "").split(","):
        normalized = _normalize_speaker_token(part)
        if normalized:
            aliases.add(normalized)
    return aliases


def _parse_dialogue_script(script: str, speaker_1_aliases: str, speaker_2_aliases: str) -> list[dict[str, Any]]:
    aliases_1 = _parse_aliases(speaker_1_aliases) | {"1", "a", "s1", "spk1", "speaker1", "char1", "left"}
    aliases_2 = _parse_aliases(speaker_2_aliases) | {"2", "b", "s2", "spk2", "speaker2", "char2", "right"}
    entries: list[dict[str, Any]] = []

    for line_number, raw_line in enumerate(str(script or "").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split("|", 2)
        if len(parts) != 3:
            raise ValueError(
                "dialogue_script format must be 'speaker|start_seconds|text' per line. "
                f"Invalid line {line_number}: {raw_line}"
            )

        speaker_token = _normalize_speaker_token(parts[0])
        start_seconds_raw = parts[1].strip().replace(",", ".")
        text = parts[2].strip()
        if not text:
            raise ValueError(f"dialogue_script line {line_number} has empty text")

        try:
            start_seconds = float(start_seconds_raw)
        except ValueError as exc:
            raise ValueError(f"dialogue_script line {line_number} has invalid start time: {parts[1]}") from exc

        if start_seconds < 0.0:
            raise ValueError(f"dialogue_script line {line_number} start time must be >= 0")

        if speaker_token in aliases_1:
            speaker_index = 1
        elif speaker_token in aliases_2:
            speaker_index = 2
        else:
            raise ValueError(
                f"dialogue_script line {line_number} speaker '{parts[0]}' is unknown. "
                "Use a speaker_1 alias or speaker_2 alias."
            )

        entries.append({
            "line_number": line_number,
            "speaker_index": speaker_index,
            "start_seconds": float(start_seconds),
            "text": text,
        })

    if not entries:
        raise ValueError("dialogue_script is empty")

    entries.sort(key=lambda item: (item["start_seconds"], item["line_number"]))
    return entries


def _round_seconds_to_samples(seconds: float, sample_rate: int) -> int:
    return int(round(float(seconds) * float(sample_rate)))


class IAMCCS_AudioExtensionMath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "fps": ("FLOAT", {
                    "default": 25.0,
                    "min": 0.001,
                    "max": 240.0,
                    "step": 0.01,
                    "tooltip": "Timeline fps used by the stitched video.",
                }),
                "segment_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "step": 1,
                    "tooltip": "Zero-based generation pass index.",
                }),
                "generated_frames": ("INT", {
                    "default": 249,
                    "min": 1,
                    "max": 100000,
                    "step": 1,
                    "tooltip": "Raw frames produced by the current pass.",
                }),
                "extension_frames": ("INT", {
                    "default": 249,
                    "min": 1,
                    "max": 100000,
                    "step": 1,
                    "tooltip": "Unique frames actually added to the stitched timeline.",
                }),
                "use_cursor_input": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If true, trust cursor_frames_in as the canonical stitched timeline cursor.",
                }),
            },
            "optional": {
                "first_pass_unique_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "step": 1,
                    "tooltip": "Optional explicit unique-frame count for pass 0. 0 means use generated_frames.",
                }),
                "cursor_frames_in": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000000,
                    "step": 1,
                    "tooltip": "Cumulative unique-frame cursor before this pass.",
                }),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "FLOAT", "FLOAT", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "cursor_frames_out",
        "segment_start_frames",
        "segment_end_frames",
        "effective_unique_frames",
        "trim_frames",
        "segment_start_s",
        "segment_end_s",
        "remaining_frames_after",
        "is_last_segment",
        "report",
    )
    FUNCTION = "compute"
    CATEGORY = "IAMCCS/Audio"

    def compute(
        self,
        audio,
        fps,
        segment_index,
        generated_frames,
        extension_frames,
        use_cursor_input,
        first_pass_unique_frames=0,
        cursor_frames_in=0,
    ):
        waveform, sample_rate = _validate_audio(audio)
        total_samples = int(waveform.shape[-1])
        track_duration_s = float(total_samples) / float(sample_rate)
        track_total_frames = max(0, int(round(track_duration_s * float(fps))))

        first_unique = int(first_pass_unique_frames) if int(first_pass_unique_frames) > 0 else int(generated_frames)
        unique_frames = first_unique if int(segment_index) == 0 else int(extension_frames)

        if bool(use_cursor_input):
            cursor_frames = max(0, int(cursor_frames_in))
        else:
            if int(segment_index) == 0:
                cursor_frames = 0
            else:
                cursor_frames = max(0, int(first_unique) + (int(segment_index) - 1) * int(extension_frames))

        remaining_frames = max(0, int(track_total_frames) - int(cursor_frames))
        effective_unique_frames = min(max(0, int(unique_frames)), remaining_frames)
        trim_frames = max(0, int(unique_frames) - int(effective_unique_frames))

        segment_start_frames = int(cursor_frames)
        segment_end_frames = int(cursor_frames) + int(effective_unique_frames)
        cursor_frames_out = segment_end_frames
        remaining_frames_after = max(0, int(track_total_frames) - int(cursor_frames_out))
        is_last_segment = 1 if effective_unique_frames > 0 and remaining_frames_after == 0 else 0

        segment_start_s = float(segment_start_frames) / float(fps)
        segment_end_s = float(segment_end_frames) / float(fps)

        report = (
            f"track_total={track_total_frames}f ({track_duration_s:.3f}s) | "
            f"cursor_in={cursor_frames}f | segment_index={int(segment_index)} | "
            f"generated={int(generated_frames)}f | extension={int(extension_frames)}f | "
            f"effective={effective_unique_frames}f | trim={trim_frames}f | "
            f"segment=[{segment_start_s:.3f}s..{segment_end_s:.3f}s] | "
            f"cursor_out={cursor_frames_out}f | remaining_after={remaining_frames_after}f | "
            f"last_segment={'yes' if is_last_segment else 'no'}"
        )
        _log.info("[AudioExtensionMath] %s", report)

        return (
            cursor_frames_out,
            segment_start_frames,
            segment_end_frames,
            effective_unique_frames,
            trim_frames,
            segment_start_s,
            segment_end_s,
            remaining_frames_after,
            is_last_segment,
            report,
        )


class IAMCCS_AudioExtender:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "fps": ("FLOAT", {
                    "default": 25.0,
                    "min": 0.001,
                    "max": 240.0,
                    "step": 0.01,
                    "tooltip": "Timeline fps used by the stitched video.",
                }),
                "mode": (["left_context_only", "right_context_only", "symmetric_context", "no_overlap"], {
                    "default": "left_context_only",
                    "tooltip": "How much audio context to include around the unique segment region.",
                }),
                "left_overlap_s": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 30.0,
                    "step": 0.01,
                    "tooltip": "Context overlap before the unique segment region.",
                }),
                "right_overlap_s": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 30.0,
                    "step": 0.01,
                    "tooltip": "Context overlap after the unique segment region.",
                }),
                "link_mode": (["independent_segments", "use_generated_frames", "use_extension_frames", "use_timeline_cursor"], {
                    "default": "use_timeline_cursor",
                    "tooltip": "How the audio chunk is linked to the generation timeline.",
                }),
                "snap_mode": (["none", "snap_to_video_duration", "snap_to_samples"], {
                    "default": "snap_to_video_duration",
                    "tooltip": "How nominal segment duration is resolved when not using explicit timeline math.",
                }),
                "clamp_policy": (["soft_clamp", "strict"], {
                    "default": "soft_clamp",
                    "tooltip": "soft_clamp trims edge segments to the audio bounds; strict raises if the requested range exceeds the track.",
                }),
            },
            "optional": {
                "segment_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "step": 1,
                    "tooltip": "Fallback segment index for independent/fixed timing modes.",
                }),
                "segment_duration_s": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.0,
                    "max": 100000.0,
                    "step": 0.01,
                    "tooltip": "Nominal per-segment duration when not using explicit linked math.",
                }),
                "video_frames": ("INT", {
                    "default": 249,
                    "min": 1,
                    "max": 100000,
                    "step": 1,
                    "tooltip": "Nominal video frame count for the current generation pass.",
                }),
                "generated_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "step": 1,
                    "tooltip": "Raw frames produced by the current pass.",
                }),
                "extension_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "step": 1,
                    "tooltip": "Unique frames actually added to the stitched timeline.",
                }),
                "timeline_cursor_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000000,
                    "step": 1,
                    "tooltip": "Explicit stitched timeline cursor in unique frames.",
                }),
                "segment_start_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000000,
                    "step": 1,
                    "tooltip": "Explicit unique segment start frame from AudioExtensionMath.",
                }),
                "effective_unique_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "step": 1,
                    "tooltip": "Explicit unique frame count from AudioExtensionMath.",
                }),
                "first_pass_unique_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "step": 1,
                    "tooltip": "Optional explicit unique-frame count for pass 0 when deriving linked ranges statelessly.",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "FLOAT", "FLOAT", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "conditioning_audio",
        "segment_audio",
        "segment_duration_s_out",
        "conditioning_duration_s_out",
        "segment_start_sample",
        "segment_end_sample",
        "conditioning_start_sample",
        "conditioning_end_sample",
        "report",
    )
    FUNCTION = "slice_segment"
    CATEGORY = "IAMCCS/Audio"

    def _resolve_frame_window(
        self,
        *,
        fps: float,
        link_mode: str,
        snap_mode: str,
        segment_index: int,
        segment_duration_s: float,
        video_frames: int,
        generated_frames: int,
        extension_frames: int,
        timeline_cursor_frames: int,
        segment_start_frames: int,
        effective_unique_frames: int,
        first_pass_unique_frames: int,
    ) -> Tuple[int, int, str]:
        video_duration_s = float(video_frames) / float(fps)
        nominal_duration_s = video_duration_s if snap_mode == "snap_to_video_duration" else float(segment_duration_s)
        nominal_frames = max(0, int(round(nominal_duration_s * float(fps))))

        if link_mode == "use_timeline_cursor":
            start_frames = max(0, int(segment_start_frames if segment_start_frames > 0 or timeline_cursor_frames == 0 else timeline_cursor_frames))
            if int(segment_start_frames) == 0:
                start_frames = max(0, int(timeline_cursor_frames))
            unique_frames = int(effective_unique_frames) if int(effective_unique_frames) > 0 else max(0, int(extension_frames))
            return start_frames, unique_frames, "explicit_timeline_cursor"

        if link_mode == "use_extension_frames":
            first_unique = int(first_pass_unique_frames) if int(first_pass_unique_frames) > 0 else max(0, int(generated_frames))
            unique_frames = int(effective_unique_frames) if int(effective_unique_frames) > 0 else (first_unique if int(segment_index) == 0 else max(0, int(extension_frames)))
            if int(segment_start_frames) > 0 or int(timeline_cursor_frames) > 0:
                start_frames = max(0, int(segment_start_frames if int(segment_start_frames) > 0 else timeline_cursor_frames))
                return start_frames, unique_frames, "extension_linked_with_cursor"
            if int(segment_index) == 0:
                return 0, unique_frames, "extension_linked_stateless"
            start_frames = max(0, first_unique + (int(segment_index) - 1) * max(0, int(extension_frames)))
            return start_frames, unique_frames, "extension_linked_stateless"

        if link_mode == "use_generated_frames":
            unique_frames = max(0, int(generated_frames)) if int(generated_frames) > 0 else max(0, nominal_frames)
            start_frames = max(0, int(timeline_cursor_frames)) if int(timeline_cursor_frames) > 0 else int(segment_index) * unique_frames
            return start_frames, unique_frames, "generated_frames"

        unique_frames = max(0, nominal_frames)
        start_frames = int(segment_index) * unique_frames
        return start_frames, unique_frames, "independent_segments"

    def slice_segment(
        self,
        audio,
        fps,
        mode,
        left_overlap_s,
        right_overlap_s,
        link_mode,
        snap_mode,
        clamp_policy,
        segment_index=0,
        segment_duration_s=10.0,
        video_frames=249,
        generated_frames=0,
        extension_frames=0,
        timeline_cursor_frames=0,
        segment_start_frames=0,
        effective_unique_frames=0,
        first_pass_unique_frames=0,
    ):
        waveform, sample_rate = _validate_audio(audio)
        total_samples = int(waveform.shape[-1])

        if float(fps) <= 0.0:
            raise ValueError("fps must be > 0")

        start_frames, unique_frames, source_rule = self._resolve_frame_window(
            fps=float(fps),
            link_mode=str(link_mode),
            snap_mode=str(snap_mode),
            segment_index=int(segment_index),
            segment_duration_s=float(segment_duration_s),
            video_frames=int(video_frames),
            generated_frames=int(generated_frames),
            extension_frames=int(extension_frames),
            timeline_cursor_frames=int(timeline_cursor_frames),
            segment_start_frames=int(segment_start_frames),
            effective_unique_frames=int(effective_unique_frames),
            first_pass_unique_frames=int(first_pass_unique_frames),
        )

        if unique_frames <= 0:
            raise ValueError("Resolved unique segment duration is empty. Check extension/generated/timeline inputs.")

        segment_start_s = float(start_frames) / float(fps)
        segment_end_s = float(start_frames + unique_frames) / float(fps)

        conditioning_start_s = segment_start_s
        conditioning_end_s = segment_end_s
        if mode in ("left_context_only", "symmetric_context"):
            conditioning_start_s -= float(left_overlap_s)
        if mode in ("right_context_only", "symmetric_context"):
            conditioning_end_s += float(right_overlap_s)

        segment_start_sample = _round_seconds_to_samples(segment_start_s, sample_rate)
        segment_end_sample = _round_seconds_to_samples(segment_end_s, sample_rate)
        conditioning_start_sample = _round_seconds_to_samples(conditioning_start_s, sample_rate)
        conditioning_end_sample = _round_seconds_to_samples(conditioning_end_s, sample_rate)

        if str(clamp_policy) == "strict":
            if segment_start_sample < 0 or segment_end_sample > total_samples:
                raise ValueError("Segment region exceeds audio bounds in strict mode")
            if conditioning_start_sample < 0 or conditioning_end_sample > total_samples:
                raise ValueError("Conditioning region exceeds audio bounds in strict mode")
            segment_clamped = False
            conditioning_clamped = False
        else:
            segment_start_sample, segment_end_sample, segment_clamped = _clamp_slice(segment_start_sample, segment_end_sample, total_samples)
            conditioning_start_sample, conditioning_end_sample, conditioning_clamped = _clamp_slice(conditioning_start_sample, conditioning_end_sample, total_samples)

        if segment_end_sample <= segment_start_sample:
            raise ValueError("Resolved segment audio slice is empty")
        if conditioning_end_sample <= conditioning_start_sample:
            raise ValueError("Resolved conditioning audio slice is empty")

        segment_audio = _audio_dict_from_slice(audio, waveform, segment_start_sample, segment_end_sample)
        conditioning_audio = _audio_dict_from_slice(audio, waveform, conditioning_start_sample, conditioning_end_sample)

        segment_duration_s_out = float(segment_end_sample - segment_start_sample) / float(sample_rate)
        conditioning_duration_s_out = float(conditioning_end_sample - conditioning_start_sample) / float(sample_rate)

        effective_left_context_s = max(0.0, float(segment_start_sample - conditioning_start_sample) / float(sample_rate))
        effective_right_context_s = max(0.0, float(conditioning_end_sample - segment_end_sample) / float(sample_rate))

        report = (
            f"link={link_mode} ({source_rule}) | mode={mode} | sr={sample_rate} | fps={float(fps):.3f} | "
            f"segment_frames=[{start_frames}..{start_frames + unique_frames}) unique={unique_frames}f | "
            f"segment_time=[{segment_start_s:.3f}s..{segment_end_s:.3f}s] | "
            f"segment_samples=[{segment_start_sample}..{segment_end_sample}) | "
            f"conditioning_samples=[{conditioning_start_sample}..{conditioning_end_sample}) | "
            f"left_ctx={effective_left_context_s:.3f}s right_ctx={effective_right_context_s:.3f}s | "
            f"segment_dur={segment_duration_s_out:.3f}s conditioning_dur={conditioning_duration_s_out:.3f}s | "
            f"clamped_segment={'yes' if segment_clamped else 'no'} clamped_conditioning={'yes' if conditioning_clamped else 'no'}"
        )
        _log.info("[AudioExtender] %s", report)

        return (
            conditioning_audio,
            segment_audio,
            segment_duration_s_out,
            conditioning_duration_s_out,
            int(segment_start_sample),
            int(segment_end_sample),
            int(conditioning_start_sample),
            int(conditioning_end_sample),
            report,
        )


NODE_CLASS_MAPPINGS = {
    "IAMCCS_AudioExtensionMath": IAMCCS_AudioExtensionMath,
    "IAMCCS_AudioExtender": IAMCCS_AudioExtender,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_AudioExtensionMath": "Audio Extension Math (timeline sync)",
    "IAMCCS_AudioExtender": "Audio Extender (segment + overlap)",
}


class IAMCCS_AudioTimelineGate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "remaining_frames_after": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000000,
                    "step": 1,
                    "tooltip": "Remaining track-aligned frames after the current segment.",
                }),
                "effective_unique_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000000,
                    "step": 1,
                    "tooltip": "Actual usable unique frames of the current segment.",
                }),
                "min_next_frames": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100000,
                    "step": 1,
                    "tooltip": "Minimum remaining frames required to justify creating another segment.",
                }),
                "strict_stop_on_last": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If true, stop immediately when nothing remains after the current segment.",
                }),
            },
            "optional": {
                "is_last_segment": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1,
                    "step": 1,
                    "tooltip": "Optional explicit last-segment flag from AudioExtensionMath.",
                }),
                "cursor_frames_out": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000000,
                    "step": 1,
                    "tooltip": "Optional passthrough cursor for downstream routing.",
                }),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "should_continue",
        "should_stop",
        "is_last_segment_out",
        "cursor_frames_out_passthrough",
        "report",
    )
    FUNCTION = "decide"
    CATEGORY = "IAMCCS/Audio"

    def decide(
        self,
        remaining_frames_after,
        effective_unique_frames,
        min_next_frames,
        strict_stop_on_last,
        is_last_segment=0,
        cursor_frames_out=0,
    ):
        remaining_frames_after = max(0, int(remaining_frames_after))
        effective_unique_frames = max(0, int(effective_unique_frames))
        min_next_frames = max(1, int(min_next_frames))
        explicit_last = 1 if int(is_last_segment) > 0 else 0

        computed_last = 1 if remaining_frames_after == 0 and effective_unique_frames > 0 else 0
        last_out = 1 if explicit_last or computed_last else 0

        if bool(strict_stop_on_last) and last_out:
            should_continue = 0
        else:
            should_continue = 1 if remaining_frames_after >= min_next_frames else 0

        should_stop = 0 if should_continue else 1

        report = (
            f"remaining_after={remaining_frames_after}f | effective={effective_unique_frames}f | "
            f"min_next={min_next_frames}f | explicit_last={explicit_last} | computed_last={computed_last} | "
            f"continue={should_continue} | stop={should_stop}"
        )
        _log.info("[AudioTimelineGate] %s", report)

        return (
            int(should_continue),
            int(should_stop),
            int(last_out),
            int(cursor_frames_out),
            report,
        )


NODE_CLASS_MAPPINGS.update({
    "IAMCCS_AudioTimelineGate": IAMCCS_AudioTimelineGate,
})


NODE_DISPLAY_NAME_MAPPINGS.update({
    "IAMCCS_AudioTimelineGate": "Audio Timeline Gate (continue/stop)",
})


class IAMCCS_AudioTimelineAssembler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segment_audio": ("AUDIO",),
                "insert_mode": (["replace", "crossfade"], {"default": "replace"}),
                "crossfade_samples": ("INT", {"default": 0, "min": 0, "max": 480000, "step": 1}),
                "pad_to_total_duration": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "timeline_audio": ("AUDIO",),
                "segment_start_sample": ("INT", {"default": 0, "min": 0, "max": 2000000000, "step": 1}),
                "segment_start_frames": ("INT", {"default": 0, "min": 0, "max": 100000000, "step": 1}),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.001, "max": 240.0, "step": 0.01}),
                "total_duration_s": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("AUDIO", "FLOAT", "INT", "INT", "STRING")
    RETURN_NAMES = ("timeline_audio", "timeline_duration_s", "insert_start_sample", "insert_end_sample", "report")
    FUNCTION = "assemble"
    CATEGORY = "IAMCCS/Audio"

    def _resolve_insert_start_sample(self, segment_start_sample: int, segment_start_frames: int, fps: float, sample_rate: int) -> tuple[int, str]:
        start_sample = max(0, int(segment_start_sample))
        if start_sample > 0:
            return start_sample, "segment_start_sample"
        if int(segment_start_frames) > 0:
            if float(fps) <= 0.0:
                raise ValueError("fps must be > 0 when using segment_start_frames")
            return max(0, _round_seconds_to_samples(float(segment_start_frames) / float(fps), sample_rate)), "segment_start_frames"
        return 0, "default_zero"

    def _ensure_channels(self, waveform: torch.Tensor, channels: int) -> torch.Tensor:
        current_channels = int(waveform.shape[1])
        if current_channels == channels:
            return waveform
        if current_channels == 1 and channels > 1:
            return waveform.repeat(1, channels, 1)
        if channels == 1 and current_channels > 1:
            return waveform.mean(dim=1, keepdim=True)
        if current_channels > channels:
            return waveform[:, :channels, :]
        pad = waveform[:, -1:, :].repeat(1, channels - current_channels, 1)
        return torch.cat([waveform, pad], dim=1)

    def assemble(
        self,
        segment_audio,
        insert_mode,
        crossfade_samples,
        pad_to_total_duration,
        timeline_audio=None,
        segment_start_sample=0,
        segment_start_frames=0,
        fps=24.0,
        total_duration_s=0.0,
    ):
        segment_waveform, sample_rate = _validate_audio(segment_audio)
        segment_waveform = segment_waveform.detach().clone()
        segment_len = int(segment_waveform.shape[-1])
        if segment_len <= 0:
            raise ValueError("segment_audio is empty")

        start_sample, source_rule = self._resolve_insert_start_sample(
            int(segment_start_sample), int(segment_start_frames), float(fps), int(sample_rate)
        )
        end_sample = int(start_sample + segment_len)

        if timeline_audio is None:
            base_template = dict(segment_audio)
            timeline_waveform = torch.zeros((1, int(segment_waveform.shape[1]), 0), dtype=segment_waveform.dtype)
            current_total = 0
            timeline_source = "empty"
        else:
            timeline_waveform, timeline_sample_rate = _validate_audio(timeline_audio)
            if int(timeline_sample_rate) != int(sample_rate):
                raise ValueError(
                    f"timeline_audio sample_rate ({timeline_sample_rate}) does not match segment_audio sample_rate ({sample_rate})"
                )
            base_template = dict(timeline_audio)
            timeline_waveform = timeline_waveform.detach().clone()
            current_total = int(timeline_waveform.shape[-1])
            timeline_source = "input"

        channels = max(int(segment_waveform.shape[1]), int(timeline_waveform.shape[1]) if timeline_waveform.numel() > 0 else 0, 1)
        if timeline_waveform.numel() == 0:
            timeline_waveform = torch.zeros((1, channels, 0), dtype=segment_waveform.dtype)
        else:
            timeline_waveform = self._ensure_channels(timeline_waveform, channels)
        segment_waveform = self._ensure_channels(segment_waveform, channels)

        target_samples = end_sample
        if bool(pad_to_total_duration) and float(total_duration_s) > 0.0:
            target_samples = max(target_samples, _round_seconds_to_samples(float(total_duration_s), sample_rate))

        if target_samples > int(timeline_waveform.shape[-1]):
            pad = torch.zeros((1, channels, target_samples - int(timeline_waveform.shape[-1])), dtype=timeline_waveform.dtype)
            timeline_waveform = torch.cat([timeline_waveform, pad], dim=-1)

        updated = timeline_waveform.clone()
        updated[:, :, start_sample:end_sample] = segment_waveform

        overlap_available = max(0, min(int(crossfade_samples), segment_len, max(0, current_total - start_sample)))
        if str(insert_mode) == "crossfade" and overlap_available > 0:
            fade = torch.linspace(0.0, 1.0, steps=overlap_available, dtype=updated.dtype).view(1, 1, overlap_available)
            existing = timeline_waveform[:, :, start_sample:start_sample + overlap_available]
            incoming = segment_waveform[:, :, :overlap_available]
            updated[:, :, start_sample:start_sample + overlap_available] = existing * (1.0 - fade) + incoming * fade

        timeline_duration_s = float(updated.shape[-1]) / float(sample_rate)
        report = (
            f"timeline_source={timeline_source} | insert_mode={insert_mode} | start={start_sample} | end={end_sample} | "
            f"segment_len={segment_len} | total_samples={int(updated.shape[-1])} | sr={sample_rate} | "
            f"crossfade_used={overlap_available} | start_source={source_rule} | duration={timeline_duration_s:.3f}s"
        )
        _log.info("[AudioTimelineAssembler] %s", report)

        return (
            _clone_audio_dict(base_template, updated, sample_rate),
            float(timeline_duration_s),
            int(start_sample),
            int(end_sample),
            report,
        )


NODE_CLASS_MAPPINGS.update({
    "IAMCCS_AudioTimelineAssembler": IAMCCS_AudioTimelineAssembler,
})


NODE_DISPLAY_NAME_MAPPINGS.update({
    "IAMCCS_AudioTimelineAssembler": "Audio Timeline Assembler (full track)",
})