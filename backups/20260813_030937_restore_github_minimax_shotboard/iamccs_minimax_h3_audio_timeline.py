# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later

"""CPU-only AudioBoard/CineLinX mixdown for MiniMax H3 R21/R30.

This module deliberately does not load media files.  It binds ordinary
ComfyUI ``AUDIO`` sockets to AudioBoard ``audioSegments``
metadata, mixes all audible lanes, and rebases independent I2VA/REF2VA clips
onto the *aligned* H3 chunk clock.  That rebase is important: a five-second
120-frame box becomes 124 frames on H3's 17k+5 grid, so the second source must
start at 5.1667 seconds in the generated master rather than at 5.0000 seconds.

The resulting master can be connected to ``IAMCCS_MiniMaxH3AtomicAudioDrive``;
the selected chunk output is also exposed for inspection and future direct
chunk-local routing.
"""

from __future__ import annotations

import copy
import json
import math
from typing import Any

import torch
import torch.nn.functional as F


SUPERNODE_LINX_TYPE = "IAMCCS_SUPERNODE_LINX"
CATEGORY = "IAMCCS/MiniMax H3/Atomic Backend"
H3_FPS = 24
MAX_AUDIO_INPUTS = 8


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value or ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        result = float(default)
    return result if math.isfinite(result) else float(default)


def _integer(value: Any, default: int = 0) -> int:
    return int(round(_number(value, default)))


def _resources(cine_linx: Any) -> dict[str, Any]:
    return _dict(_dict(cine_linx).get("resources"))


def _outputs(cine_linx: Any) -> dict[str, Any]:
    return _dict(_dict(cine_linx).get("outputs"))


def _payload(cine_linx: Any) -> dict[str, Any]:
    return _dict(_resources(cine_linx).get("cine_payload"))


def _resolve_shotplan(cine_linx: Any) -> dict[str, Any]:
    if isinstance(cine_linx, dict) and cine_linx.get("schema") == "iamccs.minimax_h3.shotplan":
        return cine_linx
    resources = _resources(cine_linx)
    outputs = _outputs(cine_linx)
    payload = _payload(cine_linx)
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


def _candidate_audio_timelines(cine_linx: Any) -> list[dict[str, Any]]:
    resources = _resources(cine_linx)
    outputs = _outputs(cine_linx)
    payload = _payload(cine_linx)
    tracks = _dict(resources.get("cine_audio_tracks"))
    candidates: list[Any] = [
        {"audioSegments": tracks.get("shotboard_segments"), "trackSettings": tracks.get("shotboard_track_settings")},
        {"audioSegments": tracks.get("segments"), "trackSettings": tracks.get("track_settings")},
        resources.get("cine_audio_timeline_json"),
        resources.get("cine_audio_bus_timeline_json"),
        resources.get("cine_board_timeline_data"),
        payload.get("timeline_data"),
        payload,
        outputs.get("audio_timeline_json"),
        outputs.get("timeline_data"),
    ]
    result: list[dict[str, Any]] = []
    for candidate in candidates:
        parsed = _json(candidate)
        if isinstance(parsed, dict):
            result.append(parsed)
            nested = _json(parsed.get("audio_data"))
            if isinstance(nested, dict):
                result.append(nested)
    return result


def _audio_contract(cine_linx: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    for timeline in _candidate_audio_timelines(cine_linx):
        segments = timeline.get("audioSegments")
        if isinstance(segments, list) and any(isinstance(item, dict) for item in segments):
            return [copy.deepcopy(item) for item in segments if isinstance(item, dict)], timeline
    return [], {}


def _is_audio(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and torch.is_tensor(value.get("waveform"))
        and value["waveform"].ndim == 3
        and int(value["waveform"].shape[-1]) > 0
        and _integer(value.get("sample_rate"), 0) > 0
    )


def _resample(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    if source_rate == target_rate:
        return waveform
    target_samples = max(1, int(round(int(waveform.shape[-1]) * target_rate / source_rate)))
    flat = waveform.reshape(-1, 1, waveform.shape[-1]).to(dtype=torch.float32)
    resized = F.interpolate(flat, size=target_samples, mode="linear", align_corners=False)
    return resized.reshape(*waveform.shape[:-1], target_samples)


def _channels(waveform: torch.Tensor, count: int) -> torch.Tensor:
    waveform = waveform[:1].to(device="cpu", dtype=torch.float32)
    if int(waveform.shape[1]) == count:
        return waveform
    if count == 1:
        return waveform.mean(dim=1, keepdim=True)
    if int(waveform.shape[1]) == 1:
        return waveform.repeat(1, count, 1)
    if int(waveform.shape[1]) > count:
        return waveform[:, :count]
    repeats = int(math.ceil(count / int(waveform.shape[1])))
    return waveform.repeat(1, repeats, 1)[:, :count]


def _source_index(segment: dict[str, Any], fallback: int) -> int:
    for key in ("audio_input", "audioInput", "source_audio_index", "sourceAudioIndex", "input_index"):
        if segment.get(key) is not None and str(segment.get(key)).strip():
            value = _integer(segment.get(key), 0)
            # User-facing indices are one-based. Explicit zero is tolerated as audio_1.
            return max(0, value - 1 if value > 0 else value)
    return fallback


def _segment_seconds(segment: dict[str, Any], fps: float) -> tuple[float, float, float]:
    if segment.get("start_seconds") is not None:
        start = max(0.0, _number(segment.get("start_seconds"), 0.0))
    else:
        start = max(0, _integer(segment.get("start", segment.get("frame", 0)), 0)) / fps
    if segment.get("duration_seconds") is not None:
        duration = max(1.0 / fps, _number(segment.get("duration_seconds"), 1.0 / fps))
    else:
        length = max(1, _integer(segment.get("length", segment.get("audioDurationFrames", 1)), 1))
        duration = length / fps
    trim_start = max(0, _integer(segment.get("trimStart", segment.get("trim_start", 0)), 0)) / fps
    return start, duration, trim_start


def _slot_interval(shotplan: dict[str, Any], chunk: dict[str, Any]) -> tuple[float, float, str]:
    fps = max(1.0, _number(shotplan.get("fps"), H3_FPS))
    slot_index = max(0, _integer(chunk.get("slot_index"), _integer(chunk.get("index"), 0)))
    slots = [item for item in _list(shotplan.get("slots")) if isinstance(item, dict)]
    slot = slots[slot_index] if slot_index < len(slots) else {}
    slot_id = str(chunk.get("slot_id") or slot.get("id") or "")
    start = max(0.0, _number(slot.get("start_seconds"), _number(chunk.get("source_start_seconds"), 0.0)))
    requested_frames = max(
        1,
        _integer(
            slot.get("requested_frame_count"),
            _integer(chunk.get("requested_frame_count"), _integer(chunk.get("frame_count"), 1)),
        ),
    )
    return start, start + requested_frames / fps, slot_id


def _chunk_interval(shotplan: dict[str, Any], chunk: dict[str, Any]) -> tuple[float, float]:
    fps = max(1.0, _number(shotplan.get("fps"), H3_FPS))
    start = max(0.0, _number(chunk.get("timeline_start_seconds"), 0.0))
    duration = max(1.0 / fps, _number(chunk.get("duration_seconds"), _integer(chunk.get("frame_count"), 1) / fps))
    return start, duration


def _linked_to_slot(segment: dict[str, Any], slot_id: str) -> bool:
    linked = str(
        segment.get("linkedVisualId")
        or segment.get("linked_visual_id")
        or segment.get("slot_id")
        or segment.get("slotId")
        or ""
    )
    return bool(linked and slot_id and linked == slot_id)


def _segment_audible(segment: dict[str, Any], contract: dict[str, Any]) -> bool:
    if bool(segment.get("mute", False)):
        return False
    track = max(0, _integer(segment.get("track"), 0))
    settings = [item for item in _list(contract.get("trackSettings")) if isinstance(item, dict)]
    solo_tracks = {index for index, item in enumerate(settings) if bool(item.get("solo", False))}
    solo_clips = any(bool(item.get("solo", False)) for item in _list(contract.get("audioSegments")) if isinstance(item, dict))
    if solo_tracks and track not in solo_tracks:
        return False
    if solo_clips and not bool(segment.get("solo", False)):
        return False
    return not (track < len(settings) and bool(settings[track].get("mute", False)))


def _segment_gain_pan(segment: dict[str, Any], contract: dict[str, Any]) -> tuple[float, float]:
    gain = max(0.0, _number(segment.get("gain", segment.get("volume", 1.0)), 1.0))
    pan = max(-1.0, min(1.0, _number(segment.get("pan"), 0.0)))
    # AudioBoardArranger writes sourceClipGain/trackVolume when it has already
    # baked track gain into ``gain``. Only raw Shotboard metadata needs the
    # track settings applied here.
    if segment.get("sourceClipGain") is None and segment.get("trackVolume") is None:
        track = max(0, _integer(segment.get("track"), 0))
        settings = [item for item in _list(contract.get("trackSettings")) if isinstance(item, dict)]
        if track < len(settings):
            state = settings[track]
            gain *= max(0.0, _number(state.get("volume"), 1.0))
            gain *= 10.0 ** (_number(state.get("gainDb"), 0.0) / 20.0)
            pan = max(-1.0, min(1.0, pan + _number(state.get("pan"), 0.0)))
    return gain, pan


def _apply_gain_pan(waveform: torch.Tensor, gain: float, pan: float, channels: int) -> torch.Tensor:
    waveform = _channels(waveform, channels) * float(gain)
    if channels < 2 or abs(pan) < 1e-8:
        return waveform
    # Constant-power balance. Existing stereo content retains both channels.
    angle = (float(pan) + 1.0) * math.pi / 4.0
    waveform = waveform.clone()
    waveform[:, 0] *= math.cos(angle) * math.sqrt(2.0)
    waveform[:, 1] *= math.sin(angle) * math.sqrt(2.0)
    return waveform


def _fade(waveform: torch.Tensor, fade_in: int, fade_out: int) -> torch.Tensor:
    length = int(waveform.shape[-1])
    result = waveform.clone()
    if fade_in > 0 and length > 0:
        count = min(length, int(fade_in))
        result[..., :count] *= torch.linspace(0.0, 1.0, count, dtype=result.dtype)
    if fade_out > 0 and length > 0:
        count = min(length, int(fade_out))
        result[..., -count:] *= torch.linspace(1.0, 0.0, count, dtype=result.dtype)
    return result


def _limit(waveform: torch.Tensor, mode: str) -> tuple[torch.Tensor, float, float]:
    before = float(waveform.abs().max().item()) if waveform.numel() else 0.0
    if mode == "peak_normalize" and before > 0.98:
        waveform = waveform * (0.98 / before)
    elif mode == "soft_clip":
        waveform = torch.tanh(waveform)
    after = float(waveform.abs().max().item()) if waveform.numel() else 0.0
    return waveform, before, after


def mix_audio_timeline(
    cine_linx: Any,
    segment_index: int,
    audio_inputs: list[dict[str, Any] | None],
    *,
    target_sample_rate: int = 32000,
    mapping_policy: str = "slot_locked_i2v",
    coverage_policy: str = "require_each_chunk",
    headroom: str = "peak_normalize",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return updated CineLinX, H3-aligned master, selected chunk and report."""
    shotplan = _resolve_shotplan(cine_linx)
    chunks = [item for item in _list(shotplan.get("chunks")) if isinstance(item, dict)]
    if not chunks:
        raise ValueError("MiniMax H3 shotplan contains no chunks")
    selected_index = int(segment_index)
    if selected_index < 0 or selected_index >= len(chunks):
        raise IndexError(f"segment_index={selected_index} outside 0..{len(chunks) - 1}")
    sources = [value if _is_audio(value) else None for value in audio_inputs[:MAX_AUDIO_INPUTS]]
    if not any(sources):
        raise ValueError("Connect at least one valid ComfyUI AUDIO input")
    rate = max(8000, min(192000, int(target_sample_rate)))
    segments, contract = _audio_contract(cine_linx)
    contract = copy.deepcopy(contract)
    contract["audioSegments"] = segments
    mapping_policy = str(mapping_policy or "slot_locked_i2v")
    if mapping_policy not in {"slot_locked_i2v", "absolute_timeline"}:
        raise ValueError(f"Unsupported audio mapping policy: {mapping_policy}")
    if coverage_policy not in {"require_each_chunk", "allow_silence"}:
        raise ValueError(f"Unsupported audio coverage policy: {coverage_policy}")
    if headroom not in {"none", "peak_normalize", "soft_clip"}:
        raise ValueError(f"Unsupported headroom mode: {headroom}")

    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    # FLF chunks are bridges, not the original visual slots. In custom-drive
    # mode, retain AudioBoard's editorial placement across those bridges.
    flf_audio_driven = (
        str(shotplan.get("continuation_mode", "")) == "flf_image_center_bridges"
        and str(shotplan.get("audio_mode", "")) == "h3_custom_audio_drive"
    )
    effective_mapping_policy = (
        "absolute_timeline"
        if flf_audio_driven and mapping_policy == "slot_locked_i2v"
        else mapping_policy
    )

    # Without AudioBoard metadata, deterministic one-input-per-chunk remains a
    # useful minimal contract for two consecutive I2VA shots.
    fallback_mapping = False
    if not segments:
        fallback_mapping = True
        fps = max(1.0, _number(shotplan.get("fps"), H3_FPS))
        segments = []
        for index, chunk in enumerate(chunks):
            if index >= len(sources) or sources[index] is None:
                continue
            slot_start, slot_end, slot_id = _slot_interval(shotplan, chunk)
            segments.append({
                "id": f"fallback_audio_{index + 1}",
                "audio_input": index + 1,
                "linkedVisualId": slot_id,
                "start_seconds": slot_start,
                "duration_seconds": max(1.0 / fps, slot_end - slot_start),
                "track": 0,
                "gain": 1.0,
                "pan": 0.0,
            })
        contract["audioSegments"] = segments

    mapped: list[tuple[dict[str, Any], int]] = []
    missing_inputs: list[dict[str, Any]] = []
    for fallback, segment in enumerate(segments):
        # Empty AudioBoard planning cards do not consume a waveform socket.
        if bool(segment.get("placeholder", False)) or not _segment_audible(segment, contract):
            continue
        source_index = _source_index(segment, fallback)
        if source_index >= len(sources) or sources[source_index] is None:
            missing_inputs.append({"segment_id": str(segment.get("id") or fallback), "audio_input": source_index + 1})
            continue
        mapped.append((segment, source_index))
    if missing_inputs:
        raise ValueError("AudioBoard segments reference missing AUDIO sockets: " + json.dumps(missing_inputs, ensure_ascii=False))

    source_channels = [int(source["waveform"].shape[1]) for source in sources if source is not None]
    output_channels = max(1, min(2, max(source_channels or [1])))
    if any(abs(_number(segment.get("pan"), 0.0)) > 1e-8 for segment, _ in mapped):
        output_channels = 2
    chunk_audio: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    uncovered: list[int] = []

    for chunk_index, chunk in enumerate(chunks):
        generated_start, generated_duration = _chunk_interval(shotplan, chunk)
        slot_start, slot_end, slot_id = _slot_interval(shotplan, chunk)
        requested_samples = max(1, int(round(generated_duration * rate)))
        mixed = torch.zeros((1, output_channels, requested_samples), dtype=torch.float32)
        used = 0
        for segment, source_index in mapped:
            segment_start, segment_duration, source_trim = _segment_seconds(
                segment,
                max(1.0, _number(shotplan.get("fps"), H3_FPS)),
            )
            segment_end = segment_start + segment_duration
            linked = _linked_to_slot(segment, slot_id)
            overlaps_slot = segment_end > slot_start and segment_start < slot_end
            if effective_mapping_policy == "slot_locked_i2v":
                if not (linked or (not str(segment.get("linkedVisualId") or segment.get("linked_visual_id") or "") and overlaps_slot)):
                    continue
                destination_seconds = max(0.0, segment_start - slot_start)
                clipped_at_slot = max(0.0, slot_start - segment_start)
            else:
                chunk_end = generated_start + generated_duration
                if segment_end <= generated_start or segment_start >= chunk_end:
                    continue
                destination_seconds = max(0.0, segment_start - generated_start)
                clipped_at_slot = max(0.0, generated_start - segment_start)

            source = sources[source_index]
            assert source is not None
            source_waveform = _resample(
                _channels(source["waveform"], output_channels),
                _integer(source.get("sample_rate"), rate),
                rate,
            )
            source_start = max(0, int(round((source_trim + clipped_at_slot) * rate)))
            desired = max(1, int(round(max(0.0, segment_duration - clipped_at_slot) * rate)))
            available = max(0, int(source_waveform.shape[-1]) - source_start)
            take = min(desired, available)
            destination = max(0, int(round(destination_seconds * rate)))
            take = min(take, max(0, requested_samples - destination))
            if take <= 0:
                continue
            gain, pan = _segment_gain_pan(segment, contract)
            piece = _apply_gain_pan(source_waveform[..., source_start:source_start + take], gain, pan, output_channels)
            fps = max(1.0, _number(shotplan.get("fps"), H3_FPS))
            fade_in = int(round(max(0, _integer(segment.get("fadeInFrames"), 0)) / fps * rate))
            fade_out = int(round(max(0, _integer(segment.get("fadeOutFrames"), 0)) / fps * rate))
            piece = _fade(piece, fade_in, fade_out)
            mixed[..., destination:destination + take] += piece
            used += 1
            assignments.append({
                "chunk_index": chunk_index,
                "slot_id": slot_id,
                "segment_id": str(segment.get("id") or ""),
                "track": max(0, _integer(segment.get("track"), 0)),
                "audio_input": source_index + 1,
                "destination_seconds": destination / rate,
                "mixed_seconds": take / rate,
                "linked": linked,
            })
        mixed, peak_before, peak_after = _limit(mixed, headroom)
        if flf_audio_driven:
            # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
            # Locked waveforms cannot obey a prompt-only silence request.
            # Reserve the Shotboard's explicit internal dialogue handles here.
            head_seconds = max(0.0, _number(chunk.get("audio_handoff_silence_head_seconds"), 0.0))
            tail_seconds = max(0.0, _number(chunk.get("audio_handoff_silence_tail_seconds"), 0.0))
            head_samples = min(requested_samples, int(round(head_seconds * rate)))
            tail_samples = min(requested_samples, int(round(tail_seconds * rate)))
            if head_samples:
                mixed[..., :head_samples] = 0
            if tail_samples:
                mixed[..., requested_samples - tail_samples:] = 0
            # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
            # This marker selects a trim-only audio stitch after H3 decoding.
        if used == 0:
            uncovered.append(chunk_index)
        chunk_audio.append({
            "waveform": mixed,
            "sample_rate": rate,
            "_iamccs": {
                "chunk_index": chunk_index,
                "timeline_start_seconds": generated_start,
                "duration_seconds": generated_duration,
                "sources": used,
                "peak_before": peak_before,
                "peak_after": peak_after,
                "flf_locked_audio_handles": flf_audio_driven,
            },
        })

    if uncovered and coverage_policy == "require_each_chunk":
        raise ValueError(f"No audible AUDIO segment mapped to H3 chunk(s): {uncovered}")

    master_samples = max(
        1,
        max(
            int(round((_chunk_interval(shotplan, chunk)[0] + _chunk_interval(shotplan, chunk)[1]) * rate))
            for chunk in chunks
        ),
    )
    master_waveform = torch.zeros((1, output_channels, master_samples), dtype=torch.float32)
    for index, chunk in enumerate(chunks):
        start, _ = _chunk_interval(shotplan, chunk)
        offset = max(0, int(round(start * rate)))
        waveform = chunk_audio[index]["waveform"]
        take = min(int(waveform.shape[-1]), master_samples - offset)
        if take > 0:
            master_waveform[..., offset:offset + take] += waveform[..., :take]
    master_gain = max(0.0, _number(contract.get("masterAudioGain"), 1.0))
    master_waveform *= master_gain
    master_waveform, master_peak_before, master_peak_after = _limit(master_waveform, headroom)
    master = {"waveform": master_waveform, "sample_rate": rate}
    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    selected = dict(chunk_audio[selected_index])
    selected["iamccs_pre_sliced"] = True
    selected["iamccs_source_start_seconds"] = float(
        _chunk_interval(shotplan, chunks[selected_index])[0]
    )

    report = {
        "schema": "iamccs.minimax_h3.audio_timeline_mix.r21",
        "mapping_policy": effective_mapping_policy,
        "requested_mapping_policy": mapping_policy,
        "coverage_policy": coverage_policy,
        "headroom": headroom,
        "fallback_one_input_per_chunk": fallback_mapping,
        "chunks": len(chunks),
        "audio_segments": len(segments),
        "mapped_assignments": assignments,
        "uncovered_chunks": uncovered,
        "sample_rate": rate,
        "channels": output_channels,
        "master_samples": master_samples,
        "master_duration_seconds": master_samples / rate,
        "master_peak_before": master_peak_before,
        "master_peak_after": master_peak_after,
        "selected_chunk": selected_index,
        "flf_locked_audio_handles": flf_audio_driven,
        # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        "alignment": "AudioBoard slot time rebased to aligned H3 chunk time" if effective_mapping_policy == "slot_locked_i2v" else "FLF bridge audio follows the editorial timeline" if flf_audio_driven else "absolute timeline",
    }

    # CineLinX can carry reference IMAGE/AUDIO tensors. A full deepcopy would
    # duplicate those buffers and can create a needless RAM/VRAM spike, so only
    # the envelope containers are copied; existing resource objects stay shared.
    if isinstance(cine_linx, dict):
        out_linx = dict(cine_linx)
        out_linx["resources"] = dict(_resources(cine_linx))
        out_linx["outputs"] = dict(_outputs(cine_linx))
        out_linx["chain"] = list(_list(cine_linx.get("chain")))
    else:
        out_linx = {
            "type": SUPERNODE_LINX_TYPE,
            "resources": {},
            "outputs": {},
            "chain": [],
        }
    resources = out_linx["resources"]
    outputs = out_linx["outputs"]
    # The installed R21 Audio Drive selects ``custom_audio`` for locked T2VA /
    # I2VA. Keep the older ``driven_audio`` alias for staged/backward readers.
    resources["iamccs_minimax_h3_custom_audio"] = master
    resources["iamccs_minimax_h3_driven_audio"] = master
    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    resources["iamccs_minimax_h3_chunk_audio"] = selected
    resources["iamccs_minimax_h3_audio_timeline_report"] = report
    outputs["minimax_h3_audio_timeline_report"] = json.dumps(report, ensure_ascii=False, indent=2)
    out_linx.setdefault("chain", []).append({
        "role": "minimax_h3_audio_timeline_mix",
        "name": "IAMCCS_MiniMaxH3AudioTimelineMixR21",
    })
    return out_linx, master, selected, report


class IAMCCS_MiniMaxH3AudioTimelineMixR21:
    """Bind up to eight AUDIO sockets to AudioBoard lanes and H3 chunks."""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {f"audio_{index}": ("AUDIO",) for index in range(1, MAX_AUDIO_INPUTS + 1)}
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "mapping_policy": (
                    ["slot_locked_i2v", "absolute_timeline"],
                    {"default": "slot_locked_i2v"},
                ),
                "coverage_policy": (
                    ["require_each_chunk", "allow_silence"],
                    {"default": "require_each_chunk"},
                ),
                "target_sample_rate": ("INT", {"default": 32000, "min": 8000, "max": 192000, "step": 1000}),
                "headroom": (["peak_normalize", "soft_clip", "none"], {"default": "peak_normalize"}),
            },
            "optional": optional,
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE, "AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = ("cine_linx", "master_audio", "chunk_audio", "report")
    FUNCTION = "mix"
    CATEGORY = CATEGORY

    def mix(
        self,
        cine_linx,
        segment_index,
        mapping_policy,
        coverage_policy,
        target_sample_rate,
        headroom,
        **kwargs,
    ):
        audio_inputs = [kwargs.get(f"audio_{index}") for index in range(1, MAX_AUDIO_INPUTS + 1)]
        out_linx, master, chunk, report = mix_audio_timeline(
            cine_linx,
            segment_index,
            audio_inputs,
            target_sample_rate=int(target_sample_rate),
            mapping_policy=str(mapping_policy),
            coverage_policy=str(coverage_policy),
            headroom=str(headroom),
        )
        return out_linx, master, chunk, json.dumps(report, ensure_ascii=False, indent=2)


NODE_CLASS_MAPPINGS = {
    "IAMCCS_MiniMaxH3AudioTimelineMixR21": IAMCCS_MiniMaxH3AudioTimelineMixR21,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_MiniMaxH3AudioTimelineMixR21": "MiniMax H3 Audio Timeline Mix (R21/R30)",
}
