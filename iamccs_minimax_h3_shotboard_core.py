# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pure MiniMax H3 timeline planner used by the standalone ComfyUI nodes.

The planner deliberately models only concepts that the stock MiniMax H3
conditioning nodes understand: T2VA, first/last keyframes, Ref2VA references,
Ref2VA audio, the fixed 24 fps clock, and the 17k+5 temporal grid.
"""

from __future__ import annotations

import json
import math
from typing import Any


H3_FPS = 24
H3_MIN_TRAINED_FRAMES = 124
H3_MAX_TRAINED_FRAMES = 362
H3_MIN_FRAMES = 5
H3_MIN_RESOLUTION = 256
H3_MAX_RESOLUTION = 5760
H3_CANVAS_MULTIPLE = 32
H3_NATIVE_MAX_PIXELS = 768 * 1344


def align_h3_frames(value: int) -> int:
    """Round up to MiniMax H3's 17k+5 temporal grid."""
    frames = max(5, int(value))
    remainder = frames % 17
    if remainder != 5:
        frames += (5 - remainder) % 17
    return frames


def _float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "off", "no"}
    return bool(value)


def parse_timeline(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    raw = _text(value)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"MiniMax H3 Shotboard timeline JSON non valido: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("MiniMax H3 Shotboard timeline deve essere un oggetto JSON")
    return parsed


def _first_value(source: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = source.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def _slot_prompt(slot: dict[str, Any]) -> str:
    return _text(_first_value(slot, ("prompt", "local_prompt", "relay_prompt", "text")))


def _slot_audio_prompt(slot: dict[str, Any]) -> str:
    return _text(_first_value(slot, ("audio_prompt", "sound_prompt", "audioPrompt")))


def _slot_image(slot: dict[str, Any]) -> str:
    value = _first_value(
        slot,
        (
            "imageTruthPath",
            "image_truth_path",
            "imageFile",
            "image_file",
            "image_path",
            "first_image",
            "first_frame",
            "image",
            "path",
        ),
    )
    return _text(value)


def _slot_explicit_last(slot: dict[str, Any]) -> str:
    return _text(_first_value(slot, ("last_image", "last_frame", "target_image", "end_image")))


def _normalise_transition(value: Any, index: int) -> str:
    raw = _text(value).lower().replace("-", "_").replace(" ", "_")
    if index == 0:
        return "start"
    if raw in {"hard_cut", "cut", "hard", "new_shot", "reset"}:
        return "hard_cut"
    return "h3_keyframe_chain"


def _timeline_rows(timeline: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("segments", "rows", "slots", "shots"):
        value = timeline.get(key)
        if isinstance(value, list):
            return [dict(row) for row in value if isinstance(row, dict)]
    nested = timeline.get("timeline")
    if isinstance(nested, dict):
        return _timeline_rows(nested)
    return []


def _timeline_image_paths(timeline: dict[str, Any]) -> list[str]:
    value = timeline.get("image_paths")
    if value is None and isinstance(timeline.get("timeline"), dict):
        value = timeline["timeline"].get("image_paths")
    if isinstance(value, list):
        return [_text(item) for item in value if _text(item)]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [_text(item) for item in parsed if _text(item)]
        return [_text(item) for item in raw.replace(";", "\n").splitlines() if _text(item)]
    return []


def _is_frame_timeline(timeline: dict[str, Any]) -> bool:
    schema = _text(timeline.get("schema")).lower()
    return (
        "filmmaker_timeline" in schema
        or "minimax_h3.shotboard_timeline" in schema
        or "frame_rate" in timeline
        or "fps" in timeline
    )


def _duration_seconds(slot: dict[str, Any], timeline: dict[str, Any], fallback: float) -> float:
    explicit = _first_value(slot, ("duration_seconds", "length_seconds", "duration"))
    if explicit is not None:
        return max(0.01, _float(explicit, fallback))
    length = slot.get("length")
    if length is not None:
        raw = max(0.01, _float(length, fallback))
        return raw / H3_FPS if _is_frame_timeline(timeline) else raw
    return max(0.01, fallback)


def _start_seconds(slot: dict[str, Any], timeline: dict[str, Any], fallback: float) -> float:
    explicit = _first_value(slot, ("start_seconds", "second", "time_seconds"))
    if explicit is not None:
        return max(0.0, _float(explicit, fallback))
    if slot.get("start") is not None:
        raw = max(0.0, _float(slot.get("start"), fallback))
        return raw / H3_FPS if _is_frame_timeline(timeline) else raw
    return max(0.0, fallback)


def _normalise_slots(timeline: dict[str, Any], duration_seconds: float, fallback_duration: float) -> list[dict[str, Any]]:
    raw_rows = _timeline_rows(timeline)
    image_paths = _timeline_image_paths(timeline)
    slots: list[dict[str, Any]] = []
    cursor = 0.0
    for index, row in enumerate(raw_rows):
        row_type = _text(row.get("type", "image")).lower()
        if row_type in {"audio", "motion", "video"} or _bool(row.get("placeholder"), False):
            continue
        duration = _duration_seconds(row, timeline, fallback_duration)
        start = _start_seconds(row, timeline, cursor)
        requested_frames = max(H3_MIN_FRAMES, int(round(duration * H3_FPS)))
        if requested_frames > H3_MAX_TRAINED_FRAMES:
            raise ValueError(
                f"Il box '{_text(_first_value(row, ('label', 'name'))) or index + 1}' richiede "
                f"{requested_frames} frame: riduci il trimming sulla timeline a massimo "
                f"{H3_MAX_TRAINED_FRAMES} frame. Il planner non divide automaticamente i box."
            )
        frame_count = align_h3_frames(requested_frames)
        if frame_count > H3_MAX_TRAINED_FRAMES:
            raise ValueError(
                f"Il box {index + 1} diventa {frame_count} frame dopo l'allineamento H3 17k+5: "
                f"riduci il trimming a massimo {H3_MAX_TRAINED_FRAMES} frame."
            )
        image = _slot_image(row)
        if not image:
            try:
                ref_index = int(row.get("ref", 0)) - 1
            except (TypeError, ValueError):
                ref_index = -1
            if 0 <= ref_index < len(image_paths):
                image = image_paths[ref_index]
        use_keyframe = row_type != "text" and _bool(
            row.get("use_keyframe", row.get("use_guide", True)),
            True,
        )
        slot = {
            "id": _text(row.get("id")) or f"shot_{index + 1}",
            "label": _text(_first_value(row, ("label", "name"))) or f"Shot {index + 1:02d}",
            "type": "image" if image and use_keyframe else "text",
            "start_seconds": start,
            "requested_frame_count": requested_frames,
            "frame_count": frame_count,
            "duration_seconds": frame_count / H3_FPS,
            "image": image if use_keyframe else "",
            "explicit_last_image": _slot_explicit_last(row),
            "prompt": _slot_prompt(row),
            "audio_prompt": _slot_audio_prompt(row),
            "transition": _normalise_transition(row.get("transition", row.get("continuity")), len(slots)),
            "use_keyframe": bool(image and use_keyframe),
        }
        slots.append(slot)
        cursor = max(cursor, start + frame_count / H3_FPS)

    slots.sort(key=lambda item: (float(item["start_seconds"]), str(item["id"])))
    for index, slot in enumerate(slots):
        if index == 0:
            slot["transition"] = "start"

    if slots:
        return slots

    total = max(0.01, _float(duration_seconds, fallback_duration))
    requested_frames = max(H3_MIN_FRAMES, int(round(total * H3_FPS)))
    if requested_frames > H3_MAX_TRAINED_FRAMES:
        raise ValueError(
            f"La timeline senza box richiede {requested_frames} frame: il massimo H3 per un singolo "
            f"chunk è {H3_MAX_TRAINED_FRAMES}. Aggiungi box e regolane il trimming."
        )
    frame_count = align_h3_frames(requested_frames)
    return [
        {
            "id": "shot_1",
            "label": "Shot 01",
            "type": "text",
            "start_seconds": 0.0,
            "requested_frame_count": requested_frames,
            "frame_count": frame_count,
            "duration_seconds": frame_count / H3_FPS,
            "image": "",
            "explicit_last_image": "",
            "prompt": "",
            "audio_prompt": "",
            "transition": "start",
            "use_keyframe": False,
        }
    ]


def _timeline_h3_bridges(timeline: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the dedicated MiniMax bridge contract, when the UI supplied it."""
    for key in ("h3_bridges", "h3Bridges"):
        value = timeline.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, dict)]
    nested = timeline.get("timeline")
    if isinstance(nested, dict):
        return _timeline_h3_bridges(nested)
    return []


def _normalise_flf_bridge_slots(
    timeline: dict[str, Any],
    slots: list[dict[str, Any]],
    duration_seconds: float,
) -> list[dict[str, Any]]:
    """Convert N image anchors into N-1 MiniMax first/last-frame chunks.

    The Shotboard renders the local prompt from the centre of one image box to
    the centre of the next.  Those centre distances determine the *relative*
    duration of the FLF chunks, while the first and last centres are normalised
    to the full requested timeline duration.  Consequently two image anchors
    on a ten-second board still produce one ten-second FLF chunk; with three or
    more anchors, resizing or moving a box changes the proportional timing of
    the adjacent chunks without losing the requested total duration.
    """
    anchors = [slot for slot in slots if _text(slot.get("image"))]
    if len(anchors) < 2:
        return slots

    ui_bridges = _timeline_h3_bridges(timeline)
    bridge_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for bridge in ui_bridges:
        pair = (
            _text(_first_value(bridge, ("from_segment_id", "fromSegmentId", "from_id"))),
            _text(_first_value(bridge, ("to_segment_id", "toSegmentId", "to_id"))),
        )
        if pair[0] and pair[1]:
            bridge_by_pair[pair] = bridge

    centres = [
        float(slot["start_seconds"]) + float(slot["requested_frame_count"]) / H3_FPS / 2.0
        for slot in anchors
    ]
    gaps = [max(1.0 / H3_FPS, centres[index + 1] - centres[index]) for index in range(len(centres) - 1)]
    gap_total = sum(gaps) or float(len(gaps))
    requested_total = max(H3_MIN_FRAMES, int(round(max(0.01, _float(duration_seconds, 10.0)) * H3_FPS)))
    if requested_total > H3_MAX_TRAINED_FRAMES * len(gaps):
        raise ValueError(
            f"La timeline FLF richiede {requested_total} frame ma {len(gaps)} ponti H3 possono contenerne "
            f"al massimo {H3_MAX_TRAINED_FRAMES * len(gaps)}. Aggiungi keyframe o riduci la durata."
        )

    raw_lengths = [requested_total * gap / gap_total for gap in gaps]
    requested_lengths = [max(H3_MIN_FRAMES, int(math.floor(value))) for value in raw_lengths]
    remainder = requested_total - sum(requested_lengths)
    order = sorted(
        range(len(raw_lengths)),
        key=lambda index: raw_lengths[index] - math.floor(raw_lengths[index]),
        reverse=remainder > 0,
    )
    step = 1 if remainder > 0 else -1
    for offset in range(abs(remainder)):
        index = order[offset % len(order)]
        if step < 0 and requested_lengths[index] <= H3_MIN_FRAMES:
            continue
        requested_lengths[index] += step

    bridge_slots: list[dict[str, Any]] = []
    cursor = 0.0
    for index, (first, last) in enumerate(zip(anchors, anchors[1:])):
        requested_frames = requested_lengths[index]
        if requested_frames > H3_MAX_TRAINED_FRAMES:
            raise ValueError(
                f"Il ponte FLF '{first['label']} -> {last['label']}' richiede {requested_frames} frame: "
                f"avvicina i centri dei box o aggiungi un keyframe (massimo {H3_MAX_TRAINED_FRAMES})."
            )
        frame_count = align_h3_frames(requested_frames)
        if frame_count > H3_MAX_TRAINED_FRAMES:
            raise ValueError(
                f"Il ponte FLF '{first['label']} -> {last['label']}' diventa {frame_count} frame dopo "
                f"l'allineamento H3 17k+5: riduci leggermente la durata relativa del ponte."
            )
        ui_bridge = bridge_by_pair.get((_text(first.get("id")), _text(last.get("id"))), {})
        local_prompt = _text(_first_value(ui_bridge, ("prompt", "local_prompt", "relay_prompt"))) or _text(first.get("prompt"))
        audio_prompt = _text(_first_value(ui_bridge, ("audio_prompt", "sound_prompt"))) or _text(first.get("audio_prompt"))
        bridge_slots.append(
            {
                "id": _text(ui_bridge.get("id")) or f"flf_bridge_{index + 1}",
                "label": _text(ui_bridge.get("label")) or f"{first['label']} -> {last['label']}",
                "type": "image",
                "start_seconds": cursor,
                "requested_frame_count": requested_frames,
                "frame_count": frame_count,
                "duration_seconds": frame_count / H3_FPS,
                "image": _text(first.get("image")),
                "explicit_last_image": _text(last.get("image")),
                "prompt": local_prompt,
                "audio_prompt": audio_prompt,
                "transition": "start" if index == 0 else "h3_keyframe_chain",
                "use_keyframe": True,
                "from_anchor_id": _text(first.get("id")),
                "to_anchor_id": _text(last.get("id")),
                "visual_start_frame": int(round(centres[index] * H3_FPS)),
                "visual_end_frame": int(round(centres[index + 1] * H3_FPS)),
            }
        )
        cursor += frame_count / H3_FPS
    return bridge_slots


def _compose_prompt(
    *,
    global_prompt: str,
    local_prompt: str,
    audio_prompt: str,
    prompt_mapping: str,
) -> str:
    sections: list[str] = []
    mapping = _text(prompt_mapping).lower()
    if mapping == "per_shot":
        mapping = "global_plus_local"
    if mapping != "local_only" and global_prompt:
        sections.append(global_prompt)
    if mapping != "global_only" and local_prompt:
        sections.append(local_prompt)
    if audio_prompt:
        sections.append(f"Audio: {audio_prompt}")
    return "\n\n".join(section for section in sections if section).strip()


def _keyframe_alignment_prompt(task: str, frame_count: int, has_first: bool, has_last: bool) -> str:
    """Build only the structural H3 keyframe alignment text.

    Creative content remains entirely user-controlled.  The final timestamp is
    derived from the aligned H3 frame count so timeline trimming is the single
    source of truth.
    """
    final_seconds = max(H3_MIN_FRAMES, int(frame_count)) / H3_FPS
    if task == "i2va" and has_first:
        # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        return (
            "Picture 1 defines the complete target frame at 0.00 seconds. "
            "Keep Picture 1's camera distance, framing, lens axis and composition locked for the full shot; "
            "do not introduce a push-in, zoom, crop or reframe unless the creative prompt explicitly requests one."
        )
    if task == "fl2va" and has_first and has_last:
        return (
            "Picture 1 defines the complete opening frame at 0.00 seconds. "
            f"Picture 2 defines the complete final frame at {final_seconds:.2f} seconds."
        )
    if task == "fl2va" and has_last:
        return f"Picture 1 defines the complete final frame at {final_seconds:.2f} seconds."
    return ""


def _audio_handoff_prompt(frame_count: int, *, is_first_chunk: bool, is_final_chunk: bool) -> str:
    """Keep generated speech clear of an edit/overlap boundary.

    H3 synthesises each chunk's audio independently.  Reserving matching
    ambience handles on both sides of every internal boundary prevents two
    different deliveries from colliding inside a subsequent AV overlap.
    A single standalone chunk remains unrestricted.
    """
    if is_first_chunk and is_final_chunk:
        return ""
    duration = max(H3_MIN_FRAMES, int(frame_count)) / H3_FPS
    dialogue_deadline = max(0.0, duration - 1.0)
    instructions = ["[AUDIO HANDOFF]"]
    if not is_first_chunk:
        instructions.append(
            "During the first 1.00 second of this chunk, use no dialogue, words, cries or new vocalisation. "
            "Continue the preceding location ambience, physical action and quiet natural breathing before any new line starts."
        )
    if not is_final_chunk:
        instructions.append(
            f"Complete every spoken line and shout no later than {dialogue_deadline:.2f} seconds. "
            "During the final 1.00 second of this chunk, use no dialogue, words, cries or new vocalisation. "
            "Keep only continuous location ambience, physical action sounds and quiet natural breathing; "
            "do not begin the next line before the following chunk."
        )
    return " ".join(instructions)


def _chunk_task(task_mode: str, audio_mode: str, has_first: bool, has_last: bool) -> str:
    requested = _text(task_mode).lower()
    if requested in {"ref2va", "ref2va_audio", "ref2va_reference", "ref2vid_lipsync", "lipsync_ref2vid", "v2va_object_swap"}:
        return "ref2va"
    if audio_mode == "h3_ref2va_audio":
        return "ref2va"
    if requested == "t2va":
        return "t2va"
    if requested in {"i2v", "i2va"}:
        return "i2va"
    if requested in {"flf", "fflf", "fl2va"}:
        return "fl2va"
    if has_first and has_last:
        return "fl2va"
    if has_first:
        return "i2va"
    if has_last:
        return "fl2va"
    return "t2va"


def _timeline_audio_rows(timeline: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the actual editorial audio lane, preserving its source timing."""
    for key in ("audioSegments", "audio_segments"):
        value = timeline.get(key)
        if isinstance(value, list):
            return [dict(row) for row in value if isinstance(row, dict)]
    nested = timeline.get("timeline")
    if isinstance(nested, dict):
        return _timeline_audio_rows(nested)
    return []


def _audio_source(row: dict[str, Any]) -> str:
    return _text(
        _first_value(
            row,
            (
                "audioTruthPath",
                "audio_truth_path",
                "audioFile",
                "audio_file",
                "sourceAudioFile",
                "source_audio_file",
                "file",
                "path",
            ),
        )
    )


def _audio_duration_seconds(row: dict[str, Any], timeline: dict[str, Any], fallback: float) -> float:
    """Return the placed AudioBoard duration, never the untrimmed source length.

    ``audioDurationFrames`` describes the uploaded media file. ``length`` is
    the actual range drawn on the editorial timeline. LongVid must use the
    latter when it exists: a 38-second music file trimmed to a 10-second lane
    must not silently create three H3 chunks.
    """
    if row.get("length") is not None:
        raw = max(1.0, _float(row.get("length"), 1.0))
        return raw / H3_FPS if _is_frame_timeline(timeline) else raw
    if row.get("duration_seconds") is not None:
        return max(1.0 / H3_FPS, _float(row.get("duration_seconds"), fallback))
    for key in ("audioDurationFrames", "audio_duration_frames", "length_frames"):
        if row.get(key) is not None:
            return max(1.0 / H3_FPS, _float(row.get(key), 1.0) / H3_FPS)
    return _duration_seconds(row, timeline, fallback)


def _timeline_duration_seconds(timeline: dict[str, Any], fallback: float) -> float:
    for key in ("duration_seconds", "durationSeconds", "duration"):
        if timeline.get(key) is not None:
            return max(0.0, _float(timeline.get(key), fallback))
    nested = timeline.get("timeline")
    if isinstance(nested, dict):
        return _timeline_duration_seconds(nested, fallback)
    return max(0.0, fallback)


def _lipsync_audio_slots(timeline: dict[str, Any]) -> list[dict[str, Any]]:
    """Use AudioBoard clips as independent Ref2Vid performance slots.

    A Ref2Vid LipSync board can intentionally have no main image boxes when a
    single external CineInfoH3 reference image is used for the whole film.  In
    that case each trimmed AudioBoard clip is still a real, separately queued
    performance take instead of collapsing into one arbitrary fallback slot.
    """
    slots: list[dict[str, Any]] = []
    for index, row in enumerate(_timeline_audio_rows(timeline)):
        source_path = _audio_source(row)
        if not source_path or _bool(row.get("placeholder"), False):
            continue
        start = _start_seconds(row, timeline, 0.0)
        requested_frames = max(H3_MIN_FRAMES, int(round(_audio_duration_seconds(row, timeline, 1.0 / H3_FPS) * H3_FPS)))
        if requested_frames > H3_MAX_TRAINED_FRAMES:
            raise ValueError(
                f"Il clip audio LipSync '{_text(_first_value(row, ('label', 'name', 'title'))) or index + 1}' richiede "
                f"{requested_frames} frame: dividi l'audio in clip di massimo {H3_MAX_TRAINED_FRAMES} frame."
            )
        frame_count = align_h3_frames(requested_frames)
        if frame_count > H3_MAX_TRAINED_FRAMES:
            raise ValueError(
                f"Il clip audio LipSync {index + 1} diventa {frame_count} frame dopo l'allineamento H3 17k+5; "
                "riduci di pochi frame il trimming."
            )
        slots.append({
            "id": _text(row.get("id")) or f"lipsync_audio_{index + 1}",
            "label": _text(_first_value(row, ("label", "name", "title"))) or f"LipSync audio {index + 1:02d}",
            "type": "text",
            "start_seconds": start,
            "requested_frame_count": requested_frames,
            "frame_count": frame_count,
            "duration_seconds": frame_count / H3_FPS,
            "image": "",
            "explicit_last_image": "",
            "prompt": "",
            "audio_prompt": "",
            "transition": "start" if not slots else "hard_cut",
            "use_keyframe": False,
        })
    return slots


def _lipsync_audio_event(
    slot: dict[str, Any],
    timeline: dict[str, Any],
    audio_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Pick the AudioBoard clip with the largest overlap for a visual slot."""
    slot_start = max(0.0, float(slot.get("start_seconds", 0.0)))
    slot_end = slot_start + max(1, int(slot.get("frame_count", H3_MIN_FRAMES))) / H3_FPS
    best: tuple[float, int, dict[str, Any], float, float] | None = None
    for index, row in enumerate(audio_rows):
        source_path = _audio_source(row)
        if not source_path or _bool(row.get("placeholder"), False):
            continue
        audio_start = _start_seconds(row, timeline, 0.0)
        audio_duration = _audio_duration_seconds(row, timeline, 1.0 / H3_FPS)
        audio_end = audio_start + audio_duration
        overlap_start = max(slot_start, audio_start)
        overlap_end = min(slot_end, audio_end)
        overlap = max(0.0, overlap_end - overlap_start)
        # For an audio-driven slot (created above) the times are equal; the
        # explicit tie-break keeps the first AudioBoard lane deterministic.
        candidate = (overlap, -index, row, overlap_start, audio_start)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None or best[0] <= 0.0:
        return None
    _, _, row, overlap_start, audio_start = best
    trim_start = max(0, int(round(_float(row.get("trimStart", row.get("trim_start", 0)), 0.0))))
    source_offset = trim_start + max(0, int(round((overlap_start - audio_start) * H3_FPS)))
    return {
        "id": _text(row.get("id")) or f"lipsync_audio_{best[1] * -1 + 1}",
        "source": "timeline_main_audio_slot",
        "source_path": _audio_source(row),
        "source_offset_frames": source_offset,
        # Ref2VA expects a conditioning clip matching the legal emitted H3
        # length. The backend pads only an H3 alignment tail when necessary.
        "duration_frames": max(1, int(slot.get("frame_count", H3_MIN_FRAMES))),
        "timeline_overlap_frames": max(1, int(round(best[0] * H3_FPS))),
        "label": _text(_first_value(row, ("label", "name", "title"))) or "AudioBoard LipSync source",
    }


def _lipsync_prompt(creative_prompt: str) -> str:
    """Keep transcript/lyrics out of the text contract for Ref2Vid LipSync."""
    contract = (
        "[REF2VID LIPSYNC]\n"
        "<Picture 1> is the static identity and visual reference. <Audio 1> is the complete spoken or sung performance timing source. "
        "Synchronize mouth shapes, phonemes, silence, breaths and facial acting to <Audio 1>. "
        "Do not request, infer, transcribe or place lyrics, dialogue text, captions or a script in the prompt. "
        "Keep the reference identity and camera framing stable unless the creative direction explicitly changes them."
    )
    return "\n\n".join(part for part in (contract, creative_prompt) if part).strip()


def _longvid_guide_plan(
    *,
    timeline: dict[str, Any],
    global_prompt: str,
    duration_seconds: float,
    prompt_mapping: str,
    resolved_width: int,
    resolved_height: int,
    audio_mode: str,
    acceleration: str,
    ref_image_size: str,
    text_encoder_device: str,
    roles: list[str],
    reference_video_role: str,
    reference_audio_role: str,
    sol_conditioning: str,
    spectrum_profile: str,
    vram_clean_before_decode: bool,
    rife_mode: str,
    active_upscale_mode: str,
    upscale_enabled: bool,
    voice_reference_picture_index: int,
) -> dict[str, Any]:
    """Compile the long-video timeline into stock ``MiniMaxH3AddGuide`` events.

    H3 itself only samples one legal 17k+5 clip at a time.  R31 therefore
    preserves one global, 24fps editorial clock and projects every main-lane
    image/audio slot into each legal H3 chunk it intersects.  The atomic
    backend consumes the resulting local events with the stock AddGuide node.
    """
    # The live Shotboard serializes its canonical edited boxes in ``rows``.
    # Older exports can also retain a stale ``segments`` mirror. LongVid must
    # honour what is visibly present on the editor timeline, otherwise an
    # edit such as deleting a guide can leave a mismatched duration/prompt
    # shadow in the generated guide plan.
    editor_rows = timeline.get("rows")
    visual_rows = (
        [dict(row) for row in editor_rows if isinstance(row, dict)]
        if isinstance(editor_rows, list) and editor_rows
        else _timeline_rows(timeline)
    )
    image_paths = _timeline_image_paths(timeline)
    visual_guides: list[dict[str, Any]] = []
    slots: list[dict[str, Any]] = []
    max_timeline_frame = 0

    for index, row in enumerate(visual_rows):
        row_type = _text(row.get("type", "image")).lower()
        if row_type in {"audio", "motion", "video", "text"} or _bool(row.get("placeholder"), False):
            continue
        if not _bool(row.get("use_keyframe", row.get("use_guide", True)), True):
            continue
        image = _slot_image(row)
        if not image:
            try:
                ref_index = int(row.get("ref", 0)) - 1
            except (TypeError, ValueError):
                ref_index = -1
            if 0 <= ref_index < len(image_paths):
                image = image_paths[ref_index]
        if not image:
            continue
        start_seconds = _start_seconds(row, timeline, 0.0)
        frame = max(0, int(round(start_seconds * H3_FPS)))
        duration = _duration_seconds(row, timeline, 1.0 / H3_FPS)
        label = _text(_first_value(row, ("label", "name"))) or f"Guide {len(visual_guides) + 1:02d}"
        guide = {
            "id": _text(row.get("id")) or f"longvid_image_{len(visual_guides) + 1}",
            "kind": "image",
            "source": "timeline_main_visual_slot",
            "source_path": image,
            "global_frame": frame,
            "start_seconds": frame / H3_FPS,
            "label": label,
            "prompt": _slot_prompt(row),
        }
        visual_guides.append(guide)
        slots.append(
            {
                "id": guide["id"],
                "label": label,
                "type": "image_guide",
                "start_seconds": guide["start_seconds"],
                "requested_frame_count": max(1, int(round(duration * H3_FPS))),
                "frame_count": 1,
                "duration_seconds": duration,
                "image": image,
                "prompt": guide["prompt"],
                "audio_prompt": "",
                "transition": "timeline_guide",
                "use_keyframe": True,
            }
        )
        max_timeline_frame = max(max_timeline_frame, frame + 1)

    audio_guides: list[dict[str, Any]] = []
    for index, row in enumerate(_timeline_audio_rows(timeline)):
        if _bool(row.get("placeholder"), False):
            continue
        source_path = _audio_source(row)
        if not source_path:
            continue
        start_seconds = _start_seconds(row, timeline, 0.0)
        start_frame = max(0, int(round(start_seconds * H3_FPS)))
        duration = _audio_duration_seconds(row, timeline, 1.0 / H3_FPS)
        duration_frames = max(1, int(round(duration * H3_FPS)))
        trim_start = max(0, int(round(_float(row.get("trimStart", row.get("trim_start", 0)), 0.0))))
        label = _text(_first_value(row, ("label", "name", "title"))) or f"Audio guide {len(audio_guides) + 1:02d}"
        audio_guides.append(
            {
                "id": _text(row.get("id")) or f"longvid_audio_{len(audio_guides) + 1}",
                "kind": "audio",
                "source": "timeline_main_audio_slot",
                "source_path": source_path,
                "global_frame": start_frame,
                "duration_frames": duration_frames,
                "source_offset_frames": trim_start,
                "start_seconds": start_frame / H3_FPS,
                "label": label,
            }
        )
        max_timeline_frame = max(max_timeline_frame, start_frame + duration_frames)

    requested_frames = max(
        H3_MIN_FRAMES,
        int(round(max(duration_seconds, _timeline_duration_seconds(timeline, duration_seconds)) * H3_FPS)),
        max_timeline_frame,
    )
    chunks: list[dict[str, Any]] = []
    prompt_map: list[dict[str, Any]] = []
    cursor = 0
    while cursor < requested_frames:
        remaining = requested_frames - cursor
        frame_count = align_h3_frames(min(H3_MAX_TRAINED_FRAMES, remaining))
        # A near-boundary rounding can only increase to the next valid grid;
        # clamp the requested portion, never the legal H3 sample length.
        if frame_count > H3_MAX_TRAINED_FRAMES:
            frame_count = H3_MAX_TRAINED_FRAMES
        chunk_index = len(chunks)
        chunk_end = cursor + frame_count
        local_guides: list[dict[str, Any]] = []
        for guide in visual_guides:
            global_frame = int(guide["global_frame"])
            if cursor <= global_frame < chunk_end:
                local_guides.append({**guide, "local_frame": global_frame - cursor})
        for guide in audio_guides:
            guide_start = int(guide["global_frame"])
            guide_end = guide_start + int(guide["duration_frames"])
            overlap_start = max(cursor, guide_start)
            overlap_end = min(chunk_end, guide_end)
            if overlap_start < overlap_end:
                local_guides.append(
                    {
                        **guide,
                        "local_frame": overlap_start - cursor,
                        "duration_frames": overlap_end - overlap_start,
                        "source_offset_frames": int(guide["source_offset_frames"]) + (overlap_start - guide_start),
                    }
                )
        local_guides.sort(key=lambda item: (int(item["local_frame"]), str(item["kind"]), str(item["id"])))
        local_prompt_lines = [
            f"Timeline guide at {float(item['local_frame']) / H3_FPS:.2f}s: {item['prompt']}"
            for item in local_guides
            if item.get("kind") == "image" and _text(item.get("prompt"))
        ]
        creative_prompt = _compose_prompt(
            global_prompt=_text(global_prompt),
            local_prompt="\n".join(local_prompt_lines),
            audio_prompt="",
            prompt_mapping=prompt_mapping,
        )
        prompt = "\n\n".join(
            part
            for part in (
                "[LONGVID TIMELINE GUIDES]\n"
                "Main-timeline image and audio guides are pinned at their stated local times. "
                "Preserve them exactly at those positions; generate the intervening motion naturally.",
                creative_prompt,
            )
            if part
        ).strip()
        chunk = {
            "index": chunk_index,
            "slot_index": chunk_index,
            "slot_id": f"longvid_chunk_{chunk_index + 1}",
            "slot_label": f"LongVid {chunk_index + 1:03d}",
            "task_mode": "t2va",
            "frame_count": frame_count,
            "requested_frame_count": min(H3_MAX_TRAINED_FRAMES, remaining),
            "fps": H3_FPS,
            "duration_seconds": frame_count / H3_FPS,
            "timeline_start_frame": cursor,
            "timeline_start_seconds": cursor / H3_FPS,
            "overlap_frames": 0,
            "trim_head_frames": 0,
            "join_mode": "hard_cut",
            "unique_frames": frame_count,
            "first_image": "",
            "last_image": "",
            "prompt": prompt,
            "creative_prompt": creative_prompt,
            "alignment_prompt": "",
            "audio_handoff_prompt": "",
            "audio_handoff_silence_seconds": 0.0,
            "audio_handoff_silence_head_seconds": 0.0,
            "audio_handoff_silence_tail_seconds": 0.0,
            "local_prompt": "\n".join(local_prompt_lines),
            "audio_prompt": "",
            "transition": "longvid_hard_cut",
            "uses_bridge_first_frame": False,
            "uses_explicit_first_keyframe": False,
            "uses_explicit_last_keyframe": False,
            "flf_anchor_contract": "longvid_positioned_guides",
            "frame_source": "longvid_global_timeline",
            "guides": local_guides,
        }
        chunks.append(chunk)
        prompt_map.append(
            {
                "chunk_index": chunk_index,
                "slot_index": chunk_index,
                "slot_label": chunk["slot_label"],
                "task_mode": "t2va",
                "start_seconds": chunk["timeline_start_seconds"],
                "duration_seconds": chunk["duration_seconds"],
                "prompt": prompt,
                "guide_count": len(local_guides),
            }
        )
        cursor = chunk_end

    reference_image_paths = list(dict.fromkeys(str(item["source_path"]) for item in visual_guides))[:4]
    guide_track = {
        "schema": "iamccs.minimax_h3.guide_track",
        "schema_version": 1,
        "mode": "longvid_guides",
        "clock": {"fps": H3_FPS, "origin": "timeline_start"},
        "visual_guide_count": len(visual_guides),
        "audio_guide_count": len(audio_guides),
        "events": visual_guides + audio_guides,
        "backend": "r31_stock_minimax_h3_add_guide",
        "source": "shotboard_main_slots",
    }
    return {
        "schema": "iamccs.minimax_h3.shotplan",
        "schema_version": 9,
        "backend_revision": "r31",
        "source_timeline_schema": _text(timeline.get("schema")),
        "fps": H3_FPS,
        "width": resolved_width,
        "height": resolved_height,
        "task_mode": "longvid_guides",
        "generation_mode": "longvid_guides",
        "continuation_mode": "longvid_timeline_guides_hard_cuts",
        "audio_mode": audio_mode,
        "prompt_mapping": prompt_mapping,
        "flf_join_mode": "h3_keyframe_cut",
        "flf_overlap_frames": 0,
        "audio_handoff_policy": {
            "scope": "timeline_audio_guides_are_pinned_per_chunk",
            "speech_free_head_seconds_after_first": 0.0,
            "speech_free_tail_seconds": 0.0,
            "final_chunk_restricted": False,
            "purpose": "R31 injects timeline audio directly through MiniMaxH3AddGuide",
        },
        "acceleration": acceleration,
        "ref_image_size": ref_image_size,
        "text_encoder_device": text_encoder_device,
        "reference_roles": roles,
        "reference_image_paths": reference_image_paths,
        "reference_video_role": _text(reference_video_role).lower() or "off",
        "reference_audio_role": _text(reference_audio_role).lower() or "off",
        "voice_reference_picture_index": max(0, min(4, int(_float(voice_reference_picture_index, 0)))),
        "sol_conditioning": sol_conditioning,
        "spectrum_profile": spectrum_profile,
        "vram_clean_before_decode": _bool(vram_clean_before_decode, True),
        "rife_mode": rife_mode,
        "upscale_enabled": bool(_bool(upscale_enabled, False)),
        "upscale_mode": active_upscale_mode,
        "chunk_policy": "longvid_global_clock_chunked_at_362_frames",
        "flf_anchor_mode": False,
        "i2v_hard_cut_mode": False,
        "ref2v_hard_cut_mode": False,
        "legacy_explicit_last": False,
        "chunk_max_frames": H3_MAX_TRAINED_FRAMES,
        "global_prompt": _text(global_prompt),
        "slots": slots,
        "segments": slots,
        "chunks": chunks,
        "prompt_map": prompt_map,
        "guide_track": guide_track,
        "total_segments": len(chunks),
        "total_shots": len(slots),
        "total_keyframes": len(visual_guides),
        "total_unique_frames": cursor,
        "effective_duration_seconds": cursor / H3_FPS,
        "requested_duration_seconds": requested_frames / H3_FPS,
        "temporal_grid": "17k+5",
        "trained_frame_range": [H3_MIN_TRAINED_FRAMES, H3_MAX_TRAINED_FRAMES],
        "resolution_contract": {
            "multiple": H3_CANVAS_MULTIPLE,
            "min_axis": H3_MIN_RESOLUTION,
            "max_axis": H3_MAX_RESOLUTION,
            "aspect_ratio_range": [0.4, 2.5],
            "native_max_pixels": H3_NATIVE_MAX_PIXELS,
            "above_native_canvas": resolved_width * resolved_height > H3_NATIVE_MAX_PIXELS,
        },
    }


def build_shotplan(
    *,
    timeline_data: Any,
    global_prompt: str,
    duration_seconds: float,
    task_mode: str = "auto",
    audio_mode: str = "h3_native_generated",
    prompt_mapping: str = "global_plus_local",
    flf_join_mode: str = "h3_keyframe_cut",
    flf_overlap_frames: int = 9,
    upscale_mode: str = "off",
    width: int = 1344,
    height: int = 768,
    acceleration: str = "native",
    ref_image_size: str = "match",
    text_encoder_device: str = "auto",
    reference_roles: list[str] | tuple[str, ...] | None = None,
    reference_video_role: str = "off",
    reference_audio_role: str = "off",
    voice_reference_picture_index: int = 0,
    sol_conditioning: str = "exact_kv_and_rows",
    spectrum_profile: str = "low_vram",
    vram_clean_before_decode: bool = True,
    rife_mode: str = "off",
    upscale_enabled: bool = False,
    chunk_profile: str | None = None,
    continuation_mode: str | None = None,
    generation_mode: str | None = None,
    chunk_seconds: float | None = None,
) -> dict[str, Any]:
    """Translate a Shotboard timeline into executable MiniMax H3 chunks.

    ``generation_mode`` and ``chunk_seconds`` remain accepted only so boards
    saved by the first standalone prototype can still be opened.
    """
    timeline = parse_timeline(timeline_data)
    if generation_mode and task_mode == "auto":
        legacy = {
            "fl2va_first_last": "auto",
            "ref2va_audio": "ref2va_audio",
            "ref2va_reference": "ref2va_reference",
            "t2va": "t2va",
        }
        task_mode = legacy.get(str(generation_mode), str(generation_mode))
    resolved_width = int(width)
    resolved_height = int(height)
    if not (H3_MIN_RESOLUTION <= resolved_width <= H3_MAX_RESOLUTION):
        raise ValueError(f"width H3 deve essere tra {H3_MIN_RESOLUTION} e {H3_MAX_RESOLUTION}")
    if not (H3_MIN_RESOLUTION <= resolved_height <= H3_MAX_RESOLUTION):
        raise ValueError(f"height H3 deve essere tra {H3_MIN_RESOLUTION} e {H3_MAX_RESOLUTION}")
    if resolved_width % H3_CANVAS_MULTIPLE or resolved_height % H3_CANVAS_MULTIPLE:
        raise ValueError("width e height H3 devono essere multipli di 32")
    ratio = resolved_width / resolved_height
    if not (0.4 <= ratio <= 2.5):
        raise ValueError("aspect ratio H3 deve essere compreso tra 2:5 e 5:2")

    acceleration = _text(acceleration).lower() or "native"
    if acceleration not in {
        "auto_3060", "low_vram_auto", "native", "h3_sage", "sage", "sage_sol", "sol_low_vram",
        "adaptive_safe", "sol_adaptive_safe", "sol_adaptive_balanced", "spectrum", "sage_spectrum",
    }:
        raise ValueError(f"accelerazione H3 non valida: {acceleration}")
    ref_image_size = _text(ref_image_size).lower() or "match"
    if ref_image_size not in {"match", "max"}:
        raise ValueError(f"ref_image_size H3 non valido: {ref_image_size}")
    text_encoder_device = _text(text_encoder_device).lower() or "auto"
    if text_encoder_device not in {"cpu_safe_12gb", "auto"}:
        raise ValueError(f"device text encoder H3 non valido: {text_encoder_device}")
    # Old boards remain loadable, but CPU is now an OOM-only fallback handled
    # by the atomic conditioning backend rather than a forced placement mode.
    if text_encoder_device == "cpu_safe_12gb":
        text_encoder_device = "auto"
    sol_conditioning = _text(sol_conditioning).lower() or "exact_kv_and_rows"
    if sol_conditioning not in {"exact_kv", "exact_kv_and_rows"}:
        raise ValueError(f"Sol-Attn conditioning non valido: {sol_conditioning}")
    spectrum_profile = _text(spectrum_profile).lower() or "low_vram"
    if spectrum_profile not in {"conservative_3060", "low_vram", "conservative_quality", "quality", "aggressive"}:
        raise ValueError(f"profilo Spectrum non valido: {spectrum_profile}")
    rife_mode = _text(rife_mode).lower() or "off"
    if rife_mode not in {"off", "rife_48fps", "rife_60fps"}:
        raise ValueError(f"modalita RIFE non valida: {rife_mode}")
    active_upscale_mode = _text(upscale_mode).lower() or "off"
    if active_upscale_mode not in {"off", "ltx23", "wan22_5b"}:
        raise ValueError(f"upscale H3 non valido: {active_upscale_mode}")
    if not _bool(upscale_enabled, False):
        active_upscale_mode = "off"
    roles = [
        _text(role).lower() or "subject_identity"
        for role in list(reference_roles or [])[:4]
    ]
    while len(roles) < 4:
        roles.append(("subject_identity", "subject_identity", "composition", "style")[len(roles)])

    requested_task_mode = _text(task_mode).lower() or "auto_from_timeline"
    # R31 is deliberately explicit.  ``auto`` continues to resolve exactly
    # as old boards did, so adding positioned guides can never change an
    # existing FLF/I2VA/REF2VA render route.
    if requested_task_mode in {"longvid_guides", "longvid", "long_video_guides"}:
        return _longvid_guide_plan(
            timeline=timeline,
            global_prompt=global_prompt,
            duration_seconds=max(0.01, _float(duration_seconds, 10.0)),
            prompt_mapping=prompt_mapping,
            resolved_width=resolved_width,
            resolved_height=resolved_height,
            audio_mode=audio_mode,
            acceleration=acceleration,
            ref_image_size=ref_image_size,
            text_encoder_device=text_encoder_device,
            roles=roles,
            reference_video_role=reference_video_role,
            reference_audio_role=reference_audio_role,
            sol_conditioning=sol_conditioning,
            spectrum_profile=spectrum_profile,
            vram_clean_before_decode=vram_clean_before_decode,
            rife_mode=rife_mode,
            active_upscale_mode=active_upscale_mode,
            upscale_enabled=upscale_enabled,
            voice_reference_picture_index=voice_reference_picture_index,
        )

    fallback_duration = min(H3_MAX_TRAINED_FRAMES / H3_FPS, max(H3_MIN_FRAMES / H3_FPS, 10.0))
    slots = _normalise_slots(timeline, duration_seconds, fallback_duration)
    lipsync_requested = requested_task_mode in {"ref2vid_lipsync", "lipsync_ref2vid"}
    lipsync_audio_rows = _timeline_audio_rows(timeline) if lipsync_requested else []
    # A LipSync performance can use one CineInfoH3 image connected outside the
    # Shotboard and several AudioBoard clips inside it. Preserve every audio
    # clip as its own hard-cut take when there are no image boxes to define the
    # take boundaries.
    if lipsync_requested and not any(_text(slot.get("image")) for slot in slots):
        audio_slots = _lipsync_audio_slots(timeline)
        if audio_slots:
            slots = audio_slots
    auto_task_mode = requested_task_mode in {"auto", "auto_from_timeline"}
    explicit_flf_mode = requested_task_mode in {"flf", "fflf", "fl2va"}
    explicit_i2v_mode = requested_task_mode in {"i2v", "i2va"}
    explicit_ref2v_mode = requested_task_mode in {
        "ref2va", "ref2va_audio", "ref2va_reference", "ref2vid_lipsync", "lipsync_ref2vid", "v2va_object_swap",
    }
    image_slots = [slot for slot in slots if _text(slot.get("image"))]
    legacy_explicit_last = len(image_slots) == 1 and bool(_text(image_slots[0].get("explicit_last_image")))
    flf_anchor_mode = bool(
        (explicit_flf_mode and len(image_slots) >= 2)
        or (auto_task_mode and len(image_slots) >= 2)
    )
    if flf_anchor_mode:
        timeline_duration = _float(timeline.get("duration_seconds"), duration_seconds)
        slots = _normalise_flf_bridge_slots(timeline, slots, timeline_duration)
    i2v_hard_cut_mode = bool(explicit_i2v_mode and len(image_slots) > 1)
    # Ref2VA does not accept the previous chunk's final frame as temporal
    # conditioning. Multiple timeline slots are independent reference-guided
    # renders and therefore meet with a clean cut. A synthetic overlap would
    # imply continuity the model never saw and can introduce edit morphing.
    # REF2VA audio-reference conditioning also routes the actual chunks to the
    # REF2VA model family.  It cannot accept temporal carry-over, even when a
    # legacy board still has FL2VA selected as its visual task label.
    ref2v_hard_cut_mode = bool(
        (explicit_ref2v_mode or _text(audio_mode).lower() == "h3_ref2va_audio")
        and len(slots) > 1
    )
    resolved_join_mode = _text(flf_join_mode).lower() or "h3_keyframe_cut"
    if resolved_join_mode not in {"h3_keyframe_cut", "wan_overlap_blend"}:
        raise ValueError(f"Modalita di concat FLF H3 non valida: {flf_join_mode}")
    resolved_overlap_frames = max(1, min(24, int(_float(flf_overlap_frames, 9))))

    chunks: list[dict[str, Any]] = []
    prompt_map: list[dict[str, Any]] = []
    unique_frames_total = 0

    for slot_index, slot in enumerate(slots):
        frame_count = int(slot["frame_count"])
        hard_cut_start = slot_index > 0 and (
            slot["transition"] == "hard_cut" or i2v_hard_cut_mode or ref2v_hard_cut_mode
        )
        next_slot = slots[slot_index + 1] if slot_index + 1 < len(slots) else None
        next_is_cut = bool(
            next_slot
            and (next_slot["transition"] == "hard_cut" or i2v_hard_cut_mode or ref2v_hard_cut_mode)
        )
        next_anchor = ""
        if next_slot and not next_is_cut:
            next_anchor = _text(next_slot.get("image"))
        explicit_last = _text(slot.get("explicit_last_image"))
        terminal_anchor = explicit_last or next_anchor
        chunk_index = len(chunks)
        first_path = _text(slot.get("image"))
        # Stable FL2VA is an N-keyframe / N-1-chunk contract:
        #
        #   chunk 1 = Picture A -> Picture B
        #   chunk 2 = Picture B -> Picture C
        #
        # The shared, user-authored Picture B must remain the opening keyframe
        # of chunk 2.  Replacing it with the sampled final frame from chunk 1
        # turns a deterministic FLF boundary into an experimental actual-output
        # chain; it can also make chunk 2 lose both of its intended anchors.
        # Native last-frame capture is retained for preview, diagnostics and the
        # isolated R19A research workflow, but never feeds this stable planner.
        bridge_first = False
        last_path = terminal_anchor
        has_first = bool(first_path or bridge_first)
        overlap = 0
        if chunk_index > 0 and not hard_cut_start and has_first:
            overlap = 1 if resolved_join_mode == "h3_keyframe_cut" else resolved_overlap_frames
        chunk_task = _chunk_task(task_mode, audio_mode, has_first, bool(last_path))
        start_frame = unique_frames_total
        creative_prompt = _compose_prompt(
            global_prompt=_text(global_prompt),
            local_prompt=_text(slot.get("prompt")),
            audio_prompt=_text(slot.get("audio_prompt")),
            prompt_mapping=prompt_mapping,
        )
        alignment_prompt = _keyframe_alignment_prompt(
            chunk_task,
            frame_count,
            has_first,
            bool(last_path),
        )
        audio_handoff_prompt = _audio_handoff_prompt(
            frame_count,
            is_first_chunk=slot_index == 0,
            is_final_chunk=slot_index + 1 >= len(slots),
        )
        lipsync_audio = _lipsync_audio_event(slot, timeline, lipsync_audio_rows) if lipsync_requested else None
        if lipsync_requested:
            if lipsync_audio is None:
                raise ValueError(
                    f"Ref2Vid LipSync richiede un clip AudioBoard sovrapposto allo slot '{slot['label']}'. "
                    "Importa l'audio nella lane principale o allinea il suo trimming allo slot."
                )
            # The source audio is the timing authority. FLF's editorial
            # speech-free boundaries would actively damage lip sync, so this
            # mode owns the complete source clip without generated handoffs.
            audio_handoff_prompt = ""
        prompt = "\n\n".join(
            part for part in (alignment_prompt, creative_prompt, audio_handoff_prompt) if part
        ).strip()
        if lipsync_requested:
            prompt = _lipsync_prompt(creative_prompt)
        chunk = {
            "index": chunk_index,
            "slot_index": slot_index,
            "slot_id": slot["id"],
            "slot_label": slot["label"],
            "task_mode": chunk_task,
            "frame_count": frame_count,
            "requested_frame_count": int(slot["requested_frame_count"]),
            "fps": H3_FPS,
            "duration_seconds": frame_count / H3_FPS,
            "timeline_start_frame": start_frame,
            "timeline_start_seconds": start_frame / H3_FPS,
            "overlap_frames": overlap,
            "trim_head_frames": overlap,
            "join_mode": resolved_join_mode if overlap else "none",
            "unique_frames": frame_count - overlap,
            "first_image": first_path,
            "last_image": last_path,
            "prompt": prompt,
            "creative_prompt": creative_prompt,
            "alignment_prompt": alignment_prompt,
            "audio_handoff_prompt": audio_handoff_prompt,
            # Legacy field keeps its original meaning: speech-free tail.
            "audio_handoff_silence_seconds": 0.0 if lipsync_requested or slot_index + 1 >= len(slots) else 1.0,
            "audio_handoff_silence_head_seconds": 0.0 if lipsync_requested or slot_index == 0 else 1.0,
            "audio_handoff_silence_tail_seconds": 0.0 if lipsync_requested or slot_index + 1 >= len(slots) else 1.0,
            "local_prompt": slot["prompt"],
            "audio_prompt": slot["audio_prompt"],
            "transition": "hard_cut" if hard_cut_start else "keyframe_adjacency",
            "uses_bridge_first_frame": bridge_first,
            "uses_explicit_first_keyframe": bool(first_path),
            "uses_explicit_last_keyframe": bool(last_path),
            "flf_anchor_contract": (
                "shared_planned_timeline_keyframe"
                if flf_anchor_mode and not hard_cut_start
                else "independent_or_hard_cut"
            ),
            "frame_source": "timeline_segment_trim",
        }
        if lipsync_audio is not None:
            chunk["lipsync_audio"] = lipsync_audio
            chunk["lipsync_contract"] = "ref2vid_static_picture_plus_audioboard_performance"
        chunks.append(chunk)
        prompt_map.append(
            {
                "chunk_index": chunk_index,
                "slot_index": slot_index,
                "slot_label": slot["label"],
                "task_mode": chunk_task,
                "start_seconds": chunk["timeline_start_seconds"],
                "duration_seconds": chunk["duration_seconds"],
                "prompt": prompt,
            }
        )
        unique_frames_total += frame_count - overlap

    reference_image_paths = _timeline_image_paths(timeline)[:4]
    image_count = len(reference_image_paths) or sum(1 for slot in slots if slot.get("image"))
    return {
        "schema": "iamccs.minimax_h3.shotplan",
        "schema_version": 10,
        "source_timeline_schema": _text(timeline.get("schema")),
        "fps": H3_FPS,
        "width": resolved_width,
        "height": resolved_height,
        "task_mode": task_mode,
        "generation_mode": task_mode,
        "continuation_mode": (
            "ref2vid_lipsync_audioboard_hard_cuts"
            if lipsync_requested
            else (
                "i2v_hard_cuts"
                if i2v_hard_cut_mode
                else (
                    "ref2va_independent_hard_cuts"
                    if ref2v_hard_cut_mode
                    else ("flf_shared_planned_keyframes" if flf_anchor_mode else "timeline_keyframe_adjacency")
                )
            )
        ),
        "audio_mode": audio_mode,
        "prompt_mapping": prompt_mapping,
        "flf_join_mode": resolved_join_mode,
        "flf_overlap_frames": resolved_overlap_frames,
        "audio_handoff_policy": {
            "scope": "source_audio_is_the_performance_timing_authority" if lipsync_requested else "both_sides_of_every_internal_independent_chunk_boundary",
            "speech_free_head_seconds_after_first": 0.0 if lipsync_requested else 1.0,
            "speech_free_tail_seconds": 0.0 if lipsync_requested else 1.0,
            "final_chunk_restricted": False,
            "purpose": "preserve direct AudioBoard lip-sync timing" if lipsync_requested else "keep dialogue and new vocalisations outside AV edit and overlap handles",
        },
        "acceleration": acceleration,
        "ref_image_size": ref_image_size,
        "text_encoder_device": text_encoder_device,
        "reference_roles": roles,
        "reference_image_paths": reference_image_paths,
        "reference_video_role": _text(reference_video_role).lower() or "off",
        "reference_audio_role": _text(reference_audio_role).lower() or "off",
        # 0 = off; 1-4 = pairs the ref_audio voice to that <Picture N> (Muse Minimax Director voice-clone convention).
        "voice_reference_picture_index": max(0, min(4, int(_float(voice_reference_picture_index, 0)))),
        "sol_conditioning": sol_conditioning,
        "spectrum_profile": spectrum_profile,
        "vram_clean_before_decode": _bool(vram_clean_before_decode, True),
        "rife_mode": rife_mode,
        "upscale_enabled": bool(_bool(upscale_enabled, False)),
        "upscale_mode": active_upscale_mode,
        "chunk_policy": (
            "one_visual_or_audioboard_slot_one_ref2vid_lipsync_hard_cut_chunk"
            if lipsync_requested
            else (
                "n_keyframes_n_minus_one_flf_bridges"
                if flf_anchor_mode
                else (
                    "one_i2v_box_one_hard_cut_chunk"
                    if i2v_hard_cut_mode
                    else (
                        "one_ref2va_prompt_box_one_hard_cut_chunk"
                        if ref2v_hard_cut_mode
                        else "one_timeline_box_one_h3_chunk"
                    )
                )
            )
        ),
        "flf_anchor_mode": flf_anchor_mode,
        "i2v_hard_cut_mode": i2v_hard_cut_mode,
        "ref2v_hard_cut_mode": ref2v_hard_cut_mode,
        "lipsync": {
            "enabled": lipsync_requested,
            "schema": "iamccs.minimax_h3.ref2vid_lipsync",
            "schema_version": 1,
            "image_source": "CineInfoH3 reference image preferred; main visual slot fallback",
            "audio_source": "main AudioBoard slot per performance chunk",
            "lyrics_in_prompt": False,
        },
        "legacy_explicit_last": legacy_explicit_last,
        "chunk_max_frames": H3_MAX_TRAINED_FRAMES,
        "global_prompt": _text(global_prompt),
        "slots": slots,
        "segments": slots,
        "chunks": chunks,
        "prompt_map": prompt_map,
        "total_segments": len(chunks),
        "total_shots": len(slots),
        "total_keyframes": image_count,
        "total_unique_frames": unique_frames_total,
        "effective_duration_seconds": unique_frames_total / H3_FPS,
        "requested_duration_seconds": sum(float(slot["requested_frame_count"]) / H3_FPS for slot in slots),
        "temporal_grid": "17k+5",
        "trained_frame_range": [H3_MIN_TRAINED_FRAMES, H3_MAX_TRAINED_FRAMES],
        "resolution_contract": {
            "multiple": H3_CANVAS_MULTIPLE,
            "min_axis": H3_MIN_RESOLUTION,
            "max_axis": H3_MAX_RESOLUTION,
            "aspect_ratio_range": [0.4, 2.5],
            "native_max_pixels": H3_NATIVE_MAX_PIXELS,
            "above_native_canvas": resolved_width * resolved_height > H3_NATIVE_MAX_PIXELS,
        },
    }


def plan_json(plan: dict[str, Any]) -> str:
    return json.dumps(plan, ensure_ascii=False, indent=2)
