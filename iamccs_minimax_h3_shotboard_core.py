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
    # An explicitly emptied current field is not permission to restore an
    # older local_prompt/relay_prompt mirror.
    if "prompt" in slot:
        return _text(slot["prompt"])
    return _text(_first_value(slot, ("prompt", "local_prompt", "relay_prompt", "text")))


def _slot_audio_prompt(slot: dict[str, Any]) -> str:
    return _text(_first_value(slot, ("audio_prompt", "sound_prompt", "audioPrompt")))


def _slot_image(slot: dict[str, Any]) -> str:
    if "imageFile" in slot:
        return _text(slot["imageFile"])
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
    # ``rows`` is the live editor truth.  ``segments`` is a compatibility
    # mirror and can retain deleted slots/prompts after a visual edit.  Reading
    # it first made a visible two-shot I2VA board compile four hidden chunks.
    # Prefer the live rows whenever they are present; older workflows that
    # only contain segments/slots/shots remain fully supported.
    for key in ("rows", "segments", "slots", "shots"):
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
        if not image and "imageFile" not in row:
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
    # First preserve the authored frame distances for prompt/timeline math.
    authored_lengths = [max(H3_MIN_FRAMES, int(math.floor(value))) for value in raw_lengths]
    remainder = requested_total - sum(authored_lengths)
    order = sorted(
        range(len(raw_lengths)),
        key=lambda index: raw_lengths[index] - math.floor(raw_lengths[index]),
        reverse=remainder > 0,
    )
    authored_step = 1 if remainder > 0 else -1
    for offset in range(abs(remainder)):
        index = order[offset % len(order)]
        if authored_step < 0 and authored_lengths[index] <= H3_MIN_FRAMES:
            continue
        authored_lengths[index] += authored_step

    # A shared FLF boundary is emitted by both neighbouring samples and then
    # removed once at delivery.  Allocate that one frame per join before
    # snapping to H3's legal 17k+5 grid.  Snapping each bridge independently
    # upward (the old behaviour) turned a 10-second board into 10.33 seconds
    # and made the extra tail look like a frozen handoff.
    target_sample_total = requested_total + max(0, len(gaps) - 1)
    raw_lengths = [target_sample_total * gap / gap_total for gap in gaps]

    def _floor_grid(value: float) -> int:
        frames = max(H3_MIN_FRAMES, int(math.floor(value)))
        return frames - ((frames - H3_MIN_FRAMES) % 17)

    sample_lengths = [_floor_grid(value) for value in raw_lengths]
    # Every legal length differs by 17 frames. Choose the nearest legal total
    # and distribute its steps where the authored interval is currently most
    # under-represented. This keeps the visual timing ratio while minimizing
    # the unavoidable H3 grid error.
    base_total = sum(sample_lengths)
    step_count = int(round((target_sample_total - base_total) / 17.0))
    if step_count > 0:
        order = sorted(
            range(len(raw_lengths)),
            key=lambda index: raw_lengths[index] - sample_lengths[index],
            reverse=True,
        )
        for offset in range(step_count):
            index = order[offset % len(order)]
            sample_lengths[index] += 17
    elif step_count < 0:
        order = sorted(
            range(len(raw_lengths)),
            key=lambda index: sample_lengths[index] - raw_lengths[index],
            reverse=True,
        )
        for offset in range(-step_count):
            index = order[offset % len(order)]
            if sample_lengths[index] > H3_MIN_FRAMES:
                sample_lengths[index] -= 17

    bridge_slots: list[dict[str, Any]] = []
    cursor = 0.0
    for index, (first, last) in enumerate(zip(anchors, anchors[1:])):
        requested_frames = authored_lengths[index]
        sample_frames = sample_lengths[index]
        if requested_frames > H3_MAX_TRAINED_FRAMES:
            raise ValueError(
                f"Il ponte FLF '{first['label']} -> {last['label']}' richiede {requested_frames} frame: "
                f"avvicina i centri dei box o aggiungi un keyframe (massimo {H3_MAX_TRAINED_FRAMES})."
            )
        frame_count = align_h3_frames(sample_frames)
        if frame_count > H3_MAX_TRAINED_FRAMES:
            raise ValueError(
                f"Il ponte FLF '{first['label']} -> {last['label']}' diventa {frame_count} frame dopo "
                f"l'allineamento H3 17k+5: riduci leggermente la durata relativa del ponte."
            )
        ui_bridge = bridge_by_pair.get((_text(first.get("id")), _text(last.get("id"))), {})
        # Bridge text is a derived UI mirror. Live rows own both explicit edits
        # and deliberate empty prompts; a stale bridge must never win.
        if isinstance(timeline.get('rows'), list):
            local_prompt = _text(first.get('prompt'))
            audio_prompt = _text(first.get('audio_prompt'))
        else:
            local_prompt = _slot_prompt(ui_bridge) if ui_bridge else _text(first.get('prompt'))
            audio_prompt = _slot_audio_prompt(ui_bridge) if ui_bridge else _text(first.get('audio_prompt'))
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


def _keyframe_alignment_prompt(
    task: str,
    frame_count: int,
    has_first: bool,
    has_last: bool,
    *,
    hard_cut_start: bool = False,
) -> str:
    """Build only the structural H3 keyframe alignment text.

    Creative content remains entirely user-controlled.  The final timestamp is
    derived from the aligned H3 frame count so timeline trimming is the single
    source of truth.
    """
    final_seconds = max(H3_MIN_FRAMES, int(frame_count)) / H3_FPS
    if task == "i2va" and has_first:
        # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        hard_cut_contract = (
            "This is a new independent shot after an editorial hard cut. "
            "Do not reconstruct or continue the preceding shot's camera move, subject scale, composition, scenery or action. "
            if hard_cut_start
            else ""
        )
        return hard_cut_contract + (
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
            "Continue only the sounds specified by the authored prompt before any new line starts."
        )
    if not is_final_chunk:
        instructions.append(
            f"Complete every spoken line and shout no later than {dialogue_deadline:.2f} seconds. "
            "During the final 1.00 second of this chunk, use no dialogue, words, cries or new vocalisation. "
            "Keep only the sounds specified by the authored prompt; "
            "do not begin the next line before the following chunk."
        )
    return " ".join(instructions)


def _chunk_task(task_mode: str, audio_mode: str, has_first: bool, has_last: bool) -> str:
    requested = _text(task_mode).lower()
    if requested in {"v2va_controlnet", "controlnet_v2v", "h3_fun_controlnet"}:
        # The official ComfyUI H3 Fun implementation patches a T2VA model and
        # supplies temporal structure through control_video. It is not a
        # REF2VA semantic-reference route.
        return "t2va"
    if requested in {"ref2va", "ref2va_audio", "ref2va_reference", "ref2vid_lipsync", "lipsync_ref2vid", "v2va_object_swap", "v2va_face_swap"}:
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
    """Apply the same reference-audio prompt contract used by working H3 flows."""
    authored = str(creative_prompt or "").strip()
    lower = authored.lower()
    # IAMCCS Prompter can already emit MiniMax's official six-section REF
    # grammar. Never put a proprietary header in front of that valid prompt.
    if "subject_definitions:" in lower and "detailed_description:" in lower:
        if "<audio 1>" not in lower:
            raise ValueError(
                "Ref2Vid LipSync full-reference prompt must define and use <Audio 1>. "
                "Use IAMCCS Prompter REF2VA/Audio Driven or add the AudioBoard performance as <Audio 1>."
            )
        return authored
    detail = authored or (
        "[Shot 1] <Subject 1> (S1) remains clearly visible and performs the supplied <Audio 1>. "
        "Keep the mouth unobstructed and the camera stable enough to read articulation."
    )
    if not detail.lstrip().lower().startswith("[shot 1]"):
        detail = f"[Shot 1] {detail}"
    # This is the successful v20/official Full-Reference relationship: Audio 1
    # is visible to Qwen as a reference and the exact same signal is also
    # locked into the sampler latent. Creative words are never fabricated.
    return (
        "subject_definitions:\n"
        "<Subject 1> is the visible performer in <Picture 1>; preserve identity, face and framing.\n"
        "<Audio 1> is the complete copied spoken or sung performance for <Subject 1> (S1).\n\n"
        "summary:\n"
        "[reference generation + audio reuse] <Subject 1> performs the complete supplied <Audio 1> with visible lip synchronization.\n\n"
        "retention_analysis:\n"
        "<Subject 1>: fully_preserved - preserve the identity and visible facial performance from <Picture 1>.\n"
        "<Audio 1>: fully_copy - reuse <Audio 1> 1:1 as the complete final performance timing track.\n\n"
        "detailed_description:\n"
        f"{detail}\n"
        "Synchronize <Subject 1> (S1)'s mouth shapes, phonemes, silence, breaths and facial acting to <Audio 1>. "
        "Every user-supplied spoken line or lyric must remain verbatim inside <d>[Language] ...</d>; never infer missing words.\n\n"
        "overall_soundscape:\n"
        "<Audio 1> is reused as the authoritative synchronized performance track.\n\n"
        "non_diegetic_music:\n"
        "N/A"
    ).strip()


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
    lipsync: bool = False,
    motion_context_tail_frames: int = 0,
    motion_context_audio: bool = True,
    motion_context_window_frames: int = H3_MAX_TRAINED_FRAMES,
    motion_context_guide_boundary: str = "safe_handoff",
) -> dict[str, Any]:
    """Compile the long-video timeline into stock ``MiniMaxH3AddGuide`` events.

    H3 itself only samples one legal 17k+5 clip at a time.  R31 therefore
    preserves one global, 24fps editorial clock and projects every main-lane
    image/audio slot into each legal H3 chunk it intersects.  The atomic
    backend consumes the resulting local events with the stock AddGuide node.
    """
    # LipSync uses stock ReferenceToVideo for identity, AddGuide for positioned
    # AudioBoard conditioning, and AudioDrive to lock that same rebased chunk.
    motion_context_tail_frames = int(motion_context_tail_frames or 0)
    if motion_context_tail_frames not in {0, 22, 39, 56}:
        motion_context_tail_frames = 22
    motion_context_enabled = motion_context_tail_frames > 0
    # A positioned AddGuide in the middle of a native AV sample is a known
    # source of full-frame flashes on the INT8/ComfyKitchen path.  The safe
    # default makes authored image guides technical handoff points: a chunk
    # ends before the next guide and the next chunk receives it at its local
    # opening.  ``positioned`` remains available for old diagnostic recipes.
    guide_boundary_mode = _text(motion_context_guide_boundary).lower() or "safe_handoff"
    if guide_boundary_mode not in {"safe_handoff", "positioned"}:
        guide_boundary_mode = "safe_handoff"
    plan_mode = (
        "longvid_motion_context"
        if motion_context_enabled
        else ("longvid_ref2vid_lipsync" if lipsync else "longvid_guides")
    )
    chunk_task = "ref2va" if lipsync else "t2va"
    guided_audio_drive = bool(not lipsync and audio_mode == "h3_custom_audio_drive")
    guide_prompt_header = (
        "[LONG MULTI-SHOT MOTION CONTEXT AUTO CHAIN]\n"
        "The previous chunk's native video/audio latent is pinned at the head of each continuation chunk. Preserve motion direction, identity, scene state and audio continuity across technical H3 joins while following the positioned Shotboard shot guides. Different image guides remain authored shot anchors, not a guaranteed continuous camera morph."
        if motion_context_enabled else
        "[LONGVID REF2VID LIPSYNC]\n"
        "<Picture 1> is the persistent visual identity. AudioBoard clips are pinned at their exact local timeline positions and are the performance timing source. "
        "Synchronize mouth shapes, phonemes, breaths, silence and facial acting to those positioned audio guides. "
        "Write every user-supplied spoken line or lyric verbatim as <d>[Language] ...</d>; never infer missing words."
        if lipsync else (
            "[LONGVID GUIDED AUDIO DRIVE]\n"
            "Main-timeline image guides remain positional H3 guides. The visible subject speaks or sings the exact supplied AudioBoard performance with natural, precise lip synchronization. "
            "The matching locked audio latent is the sole phonetic timing authority: synchronize mouth shapes, phonemes, breaths and facial acting to it, and keep the mouth closed during silence. "
            "Write every user-supplied spoken line or lyric verbatim as <d>[Language] ...</d>; never infer missing words."
            if guided_audio_drive else
            "[LONGVID TIMELINE GUIDES]\nMain-timeline image and audio guides are pinned at their stated local times. Preserve them exactly at those positions; generate the intervening motion naturally."
        )
    )
    # The live Shotboard serializes its canonical edited boxes in rows.
    # Older exports can also retain a stale ``segments`` mirror. LongVid must
    # honour what is visibly present on the editor timeline, otherwise an
    # edit such as deleting a guide can leave a mismatched duration/prompt
    # shadow in the generated guide plan.
    editor_rows = timeline.get("rows")
    visual_rows = (
        [dict(row) for row in editor_rows if isinstance(row, dict)]
        if isinstance(editor_rows, list)
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
        duration_frames = max(1, int(round(duration * H3_FPS)))
        end_frame = frame + duration_frames
        label = _text(_first_value(row, ("label", "name"))) or f"Guide {len(visual_guides) + 1:02d}"
        guide = {
            "id": _text(row.get("id")) or f"longvid_image_{len(visual_guides) + 1}",
            "kind": "image",
            "source": "timeline_main_visual_slot",
            "source_path": image,
            "global_frame": frame,
            "duration_frames": duration_frames,
            "end_frame": end_frame,
            "start_seconds": frame / H3_FPS,
            "end_seconds": end_frame / H3_FPS,
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
                "requested_frame_count": duration_frames,
                "frame_count": 1,
                "duration_seconds": duration,
                "image": image,
                "prompt": guide["prompt"],
                "audio_prompt": "",
                "transition": _normalise_transition(row.get("transition"), len(visual_guides)),
                "use_keyframe": True,
            }
        )
        max_timeline_frame = max(max_timeline_frame, end_frame)

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

    if (lipsync or guided_audio_drive) and not audio_guides:
        raise ValueError(
            "LongVid Guided Audio Drive requires at least one imported main AudioBoard clip. "
            "Place dialogue/music in an AudioBoard lane and anchor it to the intended timeline position."
        )

    requested_frames = max(
        H3_MIN_FRAMES,
        int(round(max(duration_seconds, _timeline_duration_seconds(timeline, duration_seconds)) * H3_FPS)),
        max_timeline_frame,
    )
    chunks: list[dict[str, Any]] = []
    prompt_map: list[dict[str, Any]] = []
    cursor = 0
    # Motion Context consumes part of the model's legal 362-frame window.
    # The visible timeline capacity is reduced by the selected pinned tail;
    # the sampler still receives a legal 17k+5 frame count.  The first chunk
    # has no previous latent and is aligned independently, then cropped to the
    # same visible cadence as every continuation chunk.
    # R37 and the original R41/R42 workflows default to the complete 362-frame
    # H3 window.  A smaller window is still an explicit technical choice; it
    # never changes authored slot lengths or guide positions.
    requested_motion_window = align_h3_frames(
        max(
            H3_MIN_FRAMES,
            min(
                H3_MAX_TRAINED_FRAMES,
                int(_float(motion_context_window_frames, H3_MAX_TRAINED_FRAMES)),
            ),
        )
    )
    requested_motion_window = min(H3_MAX_TRAINED_FRAMES, requested_motion_window)
    visible_capacity = (
        max(H3_MIN_FRAMES, requested_motion_window - motion_context_tail_frames)
        if motion_context_enabled
        else H3_MAX_TRAINED_FRAMES
    )
    while cursor < requested_frames:
        remaining = requested_frames - cursor
        if motion_context_enabled:
            visible_frame_count = min(visible_capacity, remaining)
            # Reserve a trained-length sample for the last technical window.
            # A 408-frame timeline with 187-frame capacity previously ended
            # in a 34-frame fragment (56 including context).
            minimum_tail = min(visible_capacity, max(1, H3_MIN_TRAINED_FRAMES - motion_context_tail_frames))
            if remaining > visible_capacity and remaining - visible_capacity < minimum_tail:
                visible_frame_count = remaining - minimum_tail
            trim_frames = motion_context_tail_frames if chunks else 0
            # In the first chunk there is no provider trim yet, but the
            # sampled latent still needs a complete native context tail for
            # the next chunk.  Keep it hidden from delivery and retain it in
            # sampled_latent for AutoChainLoadLatent.
            hidden_tail = motion_context_tail_frames if not chunks else trim_frames

            if guide_boundary_mode == "safe_handoff":
                future_boundaries = sorted(
                    {
                        int(guide.get("global_frame", 0) or 0)
                        for guide in visual_guides
                        if cursor < int(guide.get("global_frame", 0) or 0) < cursor + visible_frame_count
                    }
                )
                if future_boundaries:
                    candidate = future_boundaries[0] - cursor
                    # Never create an empty technical interval.  A short
                    # authored interval is valid on H3 (5 + 17k frames), and
                    # continuation chunks still receive their full tail.
                    if candidate >= H3_MIN_FRAMES:
                        visible_frame_count = candidate
                        if remaining > visible_frame_count:
                            # The next guide is now the exact timeline start
                            # of the following chunk; the old final-window
                            # rebalance must not move this authored boundary.
                            minimum_tail = 0

            frame_count = align_h3_frames(visible_frame_count + hidden_tail)
        else:
            frame_count = align_h3_frames(min(H3_MAX_TRAINED_FRAMES, remaining))
            visible_frame_count = frame_count
            trim_frames = 0
        # A near-boundary rounding can only increase to the next valid grid;
        # clamp the requested portion, never the legal H3 sample length.
        if frame_count > H3_MAX_TRAINED_FRAMES:
            frame_count = H3_MAX_TRAINED_FRAMES
        chunk_index = len(chunks)
        chunk_end = cursor + visible_frame_count
        local_guides: list[dict[str, Any]] = []
        for guide in visual_guides:
            # Preserve the proven R37 interval contract: a dragged Shotboard
            # slot remains the visual authority for its full authored extent.
            # If an explicit smaller technical window intersects that slot,
            # rebase the same guide after the carried AV head; slot timing is
            # never shortened or replaced by a generated T2V interval.
            guide_start = int(guide["global_frame"])
            guide_end = int(guide.get("end_frame", guide_start + max(1, int(guide.get("duration_frames", 1)))))
            overlap_start = max(cursor, guide_start)
            overlap_end = min(chunk_end, guide_end)
            if overlap_start < overlap_end:
                local_guides.append(
                    {
                        **guide,
                        "local_frame": overlap_start - cursor,
                        "intersection_start_frame": overlap_start,
                        "intersection_end_frame": overlap_end,
                        "continued_from_previous_chunk": guide_start < cursor,
                    }
                )
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
            f"Timeline guide at {(float(item['local_frame']) + trim_frames) / H3_FPS:.2f}s: {item['prompt']}"
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
                guide_prompt_header,
                creative_prompt,
            )
            if part
        ).strip()
        chunk = {
            "index": chunk_index,
            "slot_index": chunk_index,
            "slot_id": f"longvid_chunk_{chunk_index + 1}",
            "slot_label": f"LongVid {chunk_index + 1:03d}",
            "task_mode": chunk_task,
            "frame_count": frame_count,
            "requested_frame_count": visible_frame_count if motion_context_enabled else min(H3_MAX_TRAINED_FRAMES, remaining),
            **({
                "visible_frame_count": visible_frame_count,
            "motion_context_trim_frames": trim_frames,
            "motion_context_tail_frames": motion_context_tail_frames,
            "motion_context_hidden_tail_frames": (
                motion_context_tail_frames if motion_context_enabled and not chunks else 0
            ),
            "motion_context_guide_boundary": guide_boundary_mode,
            } if motion_context_enabled else {}),
            "fps": H3_FPS,
            "duration_seconds": frame_count / H3_FPS,
            **({"visible_duration_seconds": visible_frame_count / H3_FPS} if motion_context_enabled else {}),
            "timeline_start_frame": cursor,
            "timeline_start_seconds": cursor / H3_FPS,
            "overlap_frames": 0,
            "trim_head_frames": 0,
            "join_mode": "hard_cut",
            "unique_frames": visible_frame_count,
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
            "transition": (
                "longvid_start"
                if motion_context_enabled and cursor == 0
                else ("motion_context_continuation" if motion_context_enabled else "longvid_hard_cut")
            ),
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
                "task_mode": chunk_task,
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
        "mode": plan_mode,
        "clock": {"fps": H3_FPS, "origin": "timeline_start"},
        "visual_guide_count": len(visual_guides),
        "audio_guide_count": len(audio_guides),
        "events": visual_guides + audio_guides,
        "backend": (
            "r37_iamccs_motion_context_upstream_v012"
            if motion_context_enabled
            else "r31_stock_minimax_h3_add_guide"
        ),
        "lipsync": bool(lipsync or guided_audio_drive),
        "guided_audio_drive": guided_audio_drive,
        "source": "shotboard_main_slots",
    }
    return {
        "schema": "iamccs.minimax_h3.shotplan",
        "schema_version": 9,
        "backend_revision": "r37-motion-context-variant" if motion_context_enabled else "r31",
        "source_timeline_schema": _text(timeline.get("schema")),
        "fps": H3_FPS,
        "width": resolved_width,
        "height": resolved_height,
        "task_mode": plan_mode,
        "generation_mode": plan_mode,
        "continuation_mode": (
            "upstream_motion_context_native_av_latent"
            if motion_context_enabled
            else ("longvid_ref2vid_lipsync_guides_hard_cuts" if lipsync else ("longvid_guided_audio_drive_hard_cuts" if guided_audio_drive else "longvid_timeline_guides_hard_cuts"))
        ),
        **({
        "backend_variant": "motion_context_auto_chain_v1",
        "motion_context_auto_chain": {
            "enabled": motion_context_enabled,
            "provider": "ComfyUI-H3-Motion-Context-Auto-Chain-addon",
            "minimum_provider_version": "0.1.2",
            "context_frames": motion_context_tail_frames,
            "audio_context_frames": 24 if motion_context_audio else 0,
            "continue_audio": bool(motion_context_audio),
            "latent_transport": "native_av_safetensors",
            "visible_chunk_capacity": visible_capacity,
            "model_max_frames": H3_MAX_TRAINED_FRAMES,
        },
        } if motion_context_enabled else {}),
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
        "chunk_policy": (
            f"longvid_motion_context_visible_{visible_capacity}_window_{requested_motion_window}_model_max_{H3_MAX_TRAINED_FRAMES}"
            if motion_context_enabled
            else "longvid_global_clock_chunked_at_362_frames"
        ),
        "lipsync": {
            "enabled": bool(lipsync or guided_audio_drive),
            "contract": "ref2va_identity_plus_positioned_audioboard_guides" if lipsync else ("t2va_positioned_image_guides_plus_v20_zero_mask_audio_lock" if guided_audio_drive else "off"),
            "audio_authority": "locked_audioboard_chunk" if (lipsync or guided_audio_drive) else "native_guides",
            "reference_model": bool(lipsync),
            "dialogue_tags_present": bool(chunks) and all("<d>" in str(item.get("prompt", "")).lower() and "</d>" in str(item.get("prompt", "")).lower() for item in chunks),
            "dialogue_tag_contract": "user_supplied_verbatim_only",
        },
        "flf_anchor_mode": False,
        "i2v_hard_cut_mode": False,
        "ref2v_hard_cut_mode": False,
        "legacy_explicit_last": False,
        "chunk_max_frames": requested_motion_window if motion_context_enabled else H3_MAX_TRAINED_FRAMES,
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


def _masked_loop_guided_plan(base: dict[str, Any], *, duration_seconds: float,
                             window_frames: int, overlap_frames: int) -> dict[str, Any]:
    """Collapse a positioned-guide plan into one in-place looping latent.

    Unlike the legacy LongVid and Motion Context routes this contract has no
    outer segment queue, no decoded-frame join and no trim/crossfade.  The
    looping sampler owns the technical windows inside one full-length AV
    latent, while Shotboard image positions remain absolute guide indices.
    """
    guide_track = base.get("guide_track") if isinstance(base.get("guide_track"), dict) else {}
    events = guide_track.get("events") if isinstance(guide_track.get("events"), list) else []
    image_guides = [
        dict(item) for item in events
        if isinstance(item, dict) and str(item.get("kind", "")).strip().lower() == "image"
    ]
    if len(image_guides) < 2:
        raise ValueError(
            "GUIDED AV LOOP · EXPERIMENTAL requires at least two chronological Shotboard image guides."
        )
    image_guides.sort(key=lambda item: (int(item.get("global_frame", 0)), str(item.get("id", ""))))
    indices = [int(item.get("global_frame", 0)) for item in image_guides]
    if indices[0] < 0 or any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError(
            "GUIDED AV LOOP image guides must have unique, strictly increasing timeline positions."
        )

    requested_frames = max(
        H3_MIN_FRAMES,
        int(round(max(0.01, float(duration_seconds)) * H3_FPS)),
        max(int(item.get("end_frame", item.get("global_frame", 0) + 1)) for item in image_guides),
    )
    total_frames = align_h3_frames(requested_frames)
    if total_frames > 3600:
        raise ValueError(
            f"GUIDED AV LOOP needs {total_frames} frames, above the current 3600-frame looping ceiling. "
            "Shorten the timeline or split it into editorial reels."
        )
    for item in image_guides:
        if int(item.get("global_frame", 0)) >= total_frames:
            raise ValueError(
                f"GUIDED AV LOOP guide '{item.get('label') or item.get('id')}' lies outside the "
                f"{total_frames}-frame master latent."
            )

    window = align_h3_frames(max(H3_MIN_TRAINED_FRAMES, min(H3_MAX_TRAINED_FRAMES, int(window_frames))))
    window = min(H3_MAX_TRAINED_FRAMES, window)
    overlap = int(overlap_frames)
    if overlap not in {22, 39, 56}:
        overlap = 22
    if overlap >= window:
        raise ValueError("GUIDED AV LOOP overlap must be smaller than its technical window.")

    authored_prompts = [
        str(item.get("prompt", "")).strip() for item in image_guides
        if str(item.get("prompt", "")).strip()
    ]
    local_prompt = "\n".join(
        f"Guide at {int(item['global_frame']) / H3_FPS:.2f}s: {str(item.get('prompt', '')).strip()}"
        for item in image_guides if str(item.get("prompt", "")).strip()
    )
    global_prompt = str(base.get("global_prompt", "") or "").strip()
    structural_prompt = (
        "[GUIDED AV LOOP · EXPERIMENTAL]\nGenerate one continuous evolving take. The Shotboard images are exact "
        "positioned visual guides on the same global timeline. Preserve motion direction and scene state "
        "through every internal masked overlap; do not stage a cut, freeze, dissolve, reset or re-establish "
        "the shot at a technical window boundary."
    )
    # The dedicated conditioner maps local prompts to the technical window
    # that owns their guide.  Never concatenate future local directions into
    # this fallback prompt: doing so made e.g. a later "wide shot" command act
    # during the first window before its image guide was reached.
    prompt = "\n\n".join(part for part in (structural_prompt, global_prompt) if part).strip()

    chunk = {
        "index": 0,
        "slot_index": 0,
        "slot_id": "guided_av_loop_master",
        "slot_label": "Guided AV Loop master",
        "task_mode": "t2va",
        "frame_count": total_frames,
        "requested_frame_count": requested_frames,
        "fps": H3_FPS,
        "duration_seconds": total_frames / H3_FPS,
        "timeline_start_frame": 0,
        "timeline_start_seconds": 0.0,
        "overlap_frames": 0,
        "trim_head_frames": 0,
        "join_mode": "in_place_masked_loop",
        "unique_frames": total_frames,
        "first_image": "",
        "last_image": "",
        "prompt": prompt,
        "creative_prompt": "\n".join([global_prompt, *authored_prompts]).strip(),
        "local_prompt": local_prompt,
        "audio_prompt": "",
        "transition": "single_master_latent",
        "uses_bridge_first_frame": False,
        "uses_explicit_first_keyframe": False,
        "uses_explicit_last_keyframe": False,
        "flf_anchor_contract": "absolute_keyframes_inside_one_masked_loop_latent",
        "frame_source": "masked_loop_global_timeline",
        "guides": image_guides,
    }
    base.update({
        "schema_version": max(10, int(base.get("schema_version", 0) or 0)),
        "backend_revision": "guided-av-loop-experimental-v3-full-master-av",
        "backend_variant": "iamccs_guided_av_loop_experimental_v3",
        "task_mode": "guided_av_loop_experimental",
        "generation_mode": "guided_av_loop_experimental",
        "requested_task_mode": "guided_av_loop_experimental",
        "continuation_mode": "single_master_latent_in_place_masked_overlap",
        "chunk_policy": f"one_master_{total_frames}f_window_{window}f_overlap_{overlap}f",
        "chunk_max_frames": window,
        "chunks": [chunk],
        "prompt_map": [{
            "chunk_index": 0,
            "slot_index": 0,
            "slot_label": chunk["slot_label"],
            "task_mode": "t2va",
            "start_seconds": 0.0,
            "duration_seconds": total_frames / H3_FPS,
            "prompt": prompt,
            "guide_count": len(image_guides),
        }],
        "total_segments": 1,
        "total_unique_frames": total_frames,
        "effective_duration_seconds": total_frames / H3_FPS,
        "requested_duration_seconds": requested_frames / H3_FPS,
        "masked_loop_guided": {
            "enabled": True,
            "provider": "vendored MMH3Tools looping sampler",
            "conditioning_contract": "one_prompt_per_technical_window",
            "structural_prompt": structural_prompt,
            "global_prompt": global_prompt,
            "carry": "mask",
            "latent_transport": "one_persistent_full_native_av_master",
            "handover_analysis": "window_overlap_is_the_exact_reused_context",
            "recompose_only_reused_context": True,
            "chunk_frames": window,
            "overlap_frames": overlap,
            "overlap_strength_video": 1.0,
            "overlap_strength_audio": 1.0,
            "keyframe_indices": indices,
            "keyframes": image_guides,
            "outer_queue": False,
            "join": "none; chunks are written back into the master latent in place",
            "experimental": True,
        },
        "mode_contract": {
            "mode": "guided_av_loop_experimental",
            "public_name": "GUIDED AV LOOP · EXPERIMENTAL",
            "subtitle": "ONE MASTER AV LATENT · INTERNAL MASKED WINDOWS",
            "backend": "isolated_masked_loop_sampler",
            "legacy_backend_untouched": True,
            "core_requirements": ["continuous_per_row_mask", "any_frame_keyframe_guides"],
        },
    })
    base["guide_track"] = {
        **guide_track,
        "mode": "guided_av_loop_experimental",
        "backend": "iamccs_guided_av_loop_experimental_v3",
        "events": events,
    }
    return base


def build_shotplan(
    *,
    timeline_data: Any,
    global_prompt: str,
    duration_seconds: float,
    task_mode: str = "auto",
    audio_mode: str = "h3_native_generated",
    prompt_mapping: str = "global_plus_local",
    flf_join_mode: str = "h3_keyframe_cut",
    flf_overlap_frames: int = 0,
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
    motion_context_tail_frames: int = 22,
    motion_context_audio: bool = True,
    motion_context_window_frames: int = H3_MAX_TRAINED_FRAMES,
    keyframe_joint_latent_new: bool = False,
) -> dict[str, Any]:
    """Translate a Shotboard timeline into executable MiniMax H3 chunks.

    ``generation_mode`` and ``chunk_seconds`` remain accepted only so boards
    saved by the first standalone prototype can still be opened.
    """
    if task_mode == "keyframe_joint_native" and keyframe_joint_latent_new:
        arguments = dict(locals())
        arguments.update(task_mode="latent_go_ahead", keyframe_joint_latent_new=False)
        result = build_shotplan(**arguments)
        result["requested_task_mode"] = "keyframe_joint_native"
        result["mode_contract"].update(
            mode="keyframe_joint_latent_new", latent_new=True,
            destination="authored_image_guide", history="original_generated_av_tail",
            sample_count=len(result["chunks"]), context_handover=True)
        return result
    if task_mode == "latent_go_ahead":
        arguments = dict(locals())
        arguments.update(task_mode="fl2va", flf_join_mode="h3_keyframe_cut", flf_overlap_frames=0)
        result = build_shotplan(**arguments)
        for chunk in result['chunks']:
            chunk['audio_handoff_prompt'] = ''
            chunk['prompt'] = '\n\n'.join(str(p) for p in (chunk.get('alignment_prompt'), chunk.get('creative_prompt')) if p)
        for entry, chunk in zip(result.get('prompt_map', []), result['chunks']):
            entry['prompt'] = chunk['prompt']
        result["task_mode"] = "latent_go_ahead"
        result["requested_task_mode"] = "latent_go_ahead"
        result["mode_contract"] = {"mode":"latent_go_ahead", "experimental":True,
            "history":"original_av_latents_at_past_coordinates", "pixel_crossfade_frames":0}
        return result
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
        "auto_3060", "low_vram_auto", "native", "h3_sage", "h3_sla", "sage", "sage_sol", "sol_low_vram",
        "adaptive_safe", "sol_adaptive_safe", "sol_adaptive_balanced", "spectrum", "sage_spectrum",
        "comfy_kitchen", "h3_exact", "pdd_native_8step", "fasth3_dense_6step", "matlowai_fused_turbo_manual_sigma",
        "iamccs_progressive_2stage", "iamccs_progressive_3stage", "iamccs_progressive_pdd_2stage",
    }:
        raise ValueError(f"accelerazione H3 non valida: {acceleration}")
    ref_image_size = _text(ref_image_size).lower() or "match"
    if ref_image_size not in {"match", "max"}:
        raise ValueError(f"ref_image_size H3 non valido: {ref_image_size}")
    requested_text_encoder_device = _text(text_encoder_device).lower() or "gpu_auto"
    text_encoder_device = {
        "auto": "gpu_auto",
        "cpu_safe_12gb": "cpu_direct",
    }.get(requested_text_encoder_device, requested_text_encoder_device)
    if text_encoder_device not in {"gpu_auto", "cpu_direct"}:
        raise ValueError(f"device text encoder H3 non valido: {requested_text_encoder_device}")
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
    if active_upscale_mode not in {
        "off",
        "rtx_final",
        "ltx23",
        "ltx23_per_chunk",
        "wan22_5b",
        "h3_latent_upres",
        "h3_pixel_refine",
        "h3_fast_latent_2pass",
        "h3_ultimate_tiled",
    }:
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
    # Compatibility: the former serialized mode id remains the stable IAMCCS
    # FL2VA Continuous AV route.  A new, explicitly experimental mode enables
    # automatic freeze-tail trimming without altering saved workflows.
    herrgotts_requested = requested_task_mode in {
        "longvid_masked_loop_guided", "masked_loop_guided", "long_masked_loop_guided",
    }
    experimental_loop_requested = requested_task_mode in {
        "guided_av_loop_experimental", "longvid_guided_av_loop_experimental",
    }
    continuous_av_requested = herrgotts_requested or experimental_loop_requested
    # R31 is deliberately explicit.  ``auto`` continues to resolve exactly
    # as old boards did, so adding positioned guides can never change an
    # existing FLF/I2VA/REF2VA render route.
    longvid_lipsync_requested = requested_task_mode in {
        "longvid_guided_lipsync", "longvid_audio_drive_lipsync",
        # Legacy aliases formerly selected an unstable REF2VA+AddGuide hybrid.
        # Keep them loadable, but compile them into the safe T2VA/AddGuide
        # audio-drive topology so old workflows cannot recreate the mosaic.
        "longvid_ref2vid_lipsync", "longvid_lipsync", "longvid_ref2va_lipsync",
    }
    motion_context_requested = requested_task_mode in {
        "longvid_motion_context", "longvid_motion_context_auto_chain", "motion_context_auto_chain",
    }
    if experimental_loop_requested:
        # One Queue execution, one persistent full AV master latent. The
        # looping sampler divides it into internal masked windows; these are
        # never exposed as Shotboard FL2VA chunks or outer queue segments.
        base = _longvid_guide_plan(
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
            lipsync=False,
            motion_context_tail_frames=0,
            motion_context_audio=motion_context_audio,
            motion_context_window_frames=motion_context_window_frames,
        )
        return _masked_loop_guided_plan(
            base,
            duration_seconds=max(0.01, _float(duration_seconds, 10.0)),
            window_frames=motion_context_window_frames,
            overlap_frames=motion_context_tail_frames,
        )
    joint_keyframes_requested = requested_task_mode == "keyframe_joint_native"
    if joint_keyframes_requested:
        if acceleration.startswith("iamccs_progressive"):
            raise ValueError("KEYFRAME JOINT requires single-stage native sampling; turn off progressive sampling.")
        requested_joint_frames = int(round(float(duration_seconds) * H3_FPS))
        if requested_joint_frames > H3_MAX_TRAINED_FRAMES:
            raise ValueError(
                f"KEYFRAME JOINT requires one H3 sample: maximum {H3_MAX_TRAINED_FRAMES} frames "
                f"({H3_MAX_TRAINED_FRAMES / H3_FPS:.2f}s). Shorten the timeline; "
                "this mode never silently splits into FLF clips."
            )
    if requested_task_mode in {"longvid_guides", "longvid", "long_video_guides", "keyframe_joint_native"} or longvid_lipsync_requested or motion_context_requested:
        # Direct callers receive the same audio authority as the ShotPlanner UI.
        if longvid_lipsync_requested:
            audio_mode = "h3_custom_audio_drive"
        plan = _longvid_guide_plan(
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
            # Safe LongVid lip-sync is Guided FL2VA/T2VA audio drive, never the
            # old REF2VA-reference + duplicated AddGuide image topology.
            lipsync=False,
            motion_context_tail_frames=(motion_context_tail_frames if motion_context_requested else 0),
            motion_context_audio=motion_context_audio,
            motion_context_window_frames=motion_context_window_frames,
        )
        if joint_keyframes_requested:
            chunks = plan.get("chunks", [])
            events = plan.get("guide_track", {}).get("events", [])
            image_events = [event for event in events if event.get("kind") == "image"]
            if len(chunks) != 1 or len(image_events) < 2:
                raise ValueError("KEYFRAME JOINT needs at least two enabled image guides within one H3 window.")
            plan["requested_task_mode"] = "keyframe_joint_native"
            plan["mode_contract"] = {
                "mode": "keyframe_joint_native",
                "backend": "stock_h3_addguide_joint_sampling",
                "sample_count": 1,
                "pixel_crossfade_frames": 0,
                "context_handover": False,
                "experimental": True,
                "motion_guaranteed": False,
            }
            plan["chunk_policy"] = "one_joint_native_sample_no_silent_split"
        if motion_context_requested:
            plan["requested_task_mode"] = requested_task_mode
            plan["mode_contract"] = {
                "mode": "longvid_motion_context",
                "backend": "isolated_r37_variant",
                "legacy_backend_untouched": True,
                "requires": "ComfyUI-H3-Motion-Context-Auto-Chain-addon>=0.1.2",
            }
        if longvid_lipsync_requested:
            plan["requested_task_mode"] = requested_task_mode
            plan["mode_migration"] = {
                "from": requested_task_mode,
                "to": "longvid_guides+h3_custom_audio_drive",
                "reason": "safe_guided_lipsync_without_ref2va_hybrid",
            }
        return plan

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
    # LONG CONTINUOUS GUIDED is deliberately compiled by the FL2VA interval
    # planner, not by Long Multi-Shot's positioned AddGuide planner.  Every
    # consecutive pair of authored pictures is one destination interval; from
    # interval two onward the upstream Motion Context provider owns the opening
    # frames with the previous *sampled native AV latent*.
    continuous_guided_requested = requested_task_mode in {
        "longvid_continuous_guided", "long_continuous_guided",
    } or continuous_av_requested
    legacy_motion_context_requested = continuous_guided_requested and not continuous_av_requested
    if continuous_guided_requested and not continuous_av_requested:
        # Canonicalize only this new isolated branch; all historical mode IDs
        # retain their serialized spelling for compatibility.
        requested_task_mode = "longvid_continuous_guided"
        task_mode = requested_task_mode
    explicit_flf_mode = requested_task_mode in {"flf", "fflf", "fl2va"} or continuous_guided_requested
    explicit_i2v_mode = requested_task_mode in {"i2v", "i2va"}
    explicit_ref2v_mode = requested_task_mode in {
        "ref2va", "ref2va_audio", "ref2va_reference", "ref2vid_lipsync", "lipsync_ref2vid", "v2va_object_swap", "v2va_face_swap",
    }
    image_slots = [slot for slot in slots if _text(slot.get("image"))]
    if continuous_guided_requested:
        if len(image_slots) < 2:
            raise ValueError(
                "LONG CONTINUOUS GUIDED requires at least two chronological Shotboard image guides."
            )
        starts = [float(slot.get("start_seconds", 0.0)) for slot in image_slots]
        if any(next_start <= start for start, next_start in zip(starts, starts[1:])):
            raise ValueError(
                "LONG CONTINUOUS GUIDED image guides must have unique, strictly increasing timeline positions."
            )
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
    # The user-visible overlap is a delivery effect, not a Motion Context
    # requirement.  Keep it truly off unless WAN overlap blend is explicitly
    # selected.  The keyframe-cut path below still trims exactly one duplicated
    # shared boundary frame; that is not a dissolve and is intentionally not
    # exposed as an overlap amount.
    authored_overlap_frames = max(0, min(24, int(_float(flf_overlap_frames, 0))))
    # Decoded overlap is legal only for the ordinary FLF keyframe-adjacency
    # delivery path.  Hard-cut I2V/REF2VA and latent-handover continuation
    # already have their own boundary contract; blending their decoded clips
    # would reintroduce a previous shot or create the long frozen dissolves the
    # user explicitly disabled.  Keep a saved WAN selection harmless outside
    # its compatible route instead of silently changing generation semantics.
    overlap_allowed = bool(
        flf_anchor_mode
        and not continuous_guided_requested
        and not i2v_hard_cut_mode
        and not ref2v_hard_cut_mode
    )
    resolved_join_mode = (
        resolved_join_mode if overlap_allowed else "h3_keyframe_cut"
    )
    resolved_overlap_frames = (
        authored_overlap_frames
        if overlap_allowed and resolved_join_mode == "wan_overlap_blend"
        else 0
    )

    chunks: list[dict[str, Any]] = []
    prompt_map: list[dict[str, Any]] = []
    unique_frames_total = 0

    for slot_index, slot in enumerate(slots):
        requested_frame_count = int(slot["requested_frame_count"])
        motion_trim_frames = 0
        frame_count = int(slot["frame_count"])
        if legacy_motion_context_requested and slot_index > 0:
            motion_trim_frames = max(5, int(motion_context_tail_frames or 22))
            frame_count = align_h3_frames(requested_frame_count + motion_trim_frames)
            if frame_count > H3_MAX_TRAINED_FRAMES:
                raise ValueError(
                    f"LONG CONTINUOUS GUIDED interval '{slot['label']}' needs {frame_count} sampled frames "
                    f"({requested_frame_count} visible + {motion_trim_frames} Motion Context frames), above "
                    f"MiniMax H3's {H3_MAX_TRAINED_FRAMES}-frame limit. Move the guides closer or add a guide."
                )
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
        # Match the proven Context Loop First-Scene Image Gate exactly.
        # Scene 1 is a normal A -> B FL2VA interval.  A continuation scene must
        # not tokenize the shared B image again: its opening is owned by the
        # previous sampled AV tail injected by Motion Context, while only the
        # next authored destination remains as the final keyframe.  Re-sending
        # B as Picture 1 creates competing opening conditions and causes the
        # reset/freeze/dissolve behaviour seen in the failed smoke.
        first_path = (
            ""
            if continuous_guided_requested and slot_index > 0
            else _text(slot.get("image"))
        )
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
        if chunk_index > 0 and not hard_cut_start and has_first and not continuous_guided_requested:
            overlap = 1 if resolved_join_mode == "h3_keyframe_cut" else resolved_overlap_frames
        chunk_task = _chunk_task(task_mode, audio_mode, has_first, bool(last_path))
        start_frame = unique_frames_total
        unique_frames = frame_count - motion_trim_frames if continuous_guided_requested else frame_count - overlap
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
            hard_cut_start=hard_cut_start,
        )
        audio_handoff_prompt = _audio_handoff_prompt(
            frame_count,
            is_first_chunk=slot_index == 0,
            is_final_chunk=slot_index + 1 >= len(slots),
        )
        locked_audio_hard_cut = bool(
            _text(audio_mode).lower() == "h3_custom_audio_drive"
            and (i2v_hard_cut_mode or ref2v_hard_cut_mode or hard_cut_start or next_is_cut)
        )
        if locked_audio_hard_cut:
            # AudioBoard already owns the exact per-slot timing.  A generated
            # ambience/action handoff would contradict an editorial hard cut
            # and can make H3 visually continue or reconstruct the prior take.
            audio_handoff_prompt = ""
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
            "requested_frame_count": requested_frame_count,
            **({
                "visible_frame_count": unique_frames,
                "visible_duration_seconds": unique_frames / H3_FPS,
                "motion_context_trim_frames": motion_trim_frames,
                "motion_context_tail_frames": max(5, int(motion_context_tail_frames or 22)),
            } if continuous_guided_requested else {}),
            "fps": H3_FPS,
            "duration_seconds": frame_count / H3_FPS,
            "timeline_start_frame": start_frame,
            "timeline_start_seconds": start_frame / H3_FPS,
            "overlap_frames": overlap,
            "trim_head_frames": motion_trim_frames if continuous_guided_requested else overlap,
            "join_mode": (
                "phase_aligned_full_av"
                if continuous_av_requested and slot_index > 0
                else ("motion_context_native_av" if motion_trim_frames else (resolved_join_mode if overlap else "none"))
            ),
            "unique_frames": unique_frames,
            "first_image": first_path,
            "last_image": last_path,
            "prompt": prompt,
            "creative_prompt": creative_prompt,
            "alignment_prompt": alignment_prompt,
            "audio_handoff_prompt": audio_handoff_prompt,
            # Legacy field keeps its original meaning: speech-free tail.
            "audio_handoff_silence_seconds": 0.0 if lipsync_requested or locked_audio_hard_cut or slot_index + 1 >= len(slots) else 1.0,
            "audio_handoff_silence_head_seconds": 0.0 if lipsync_requested or locked_audio_hard_cut or slot_index == 0 else 1.0,
            "audio_handoff_silence_tail_seconds": 0.0 if lipsync_requested or locked_audio_hard_cut or slot_index + 1 >= len(slots) else 1.0,
            "local_prompt": slot["prompt"],
            "audio_prompt": slot["audio_prompt"],
            "transition": (
                "motion_context_continuation"
                if motion_trim_frames
                else ("hard_cut" if hard_cut_start else "keyframe_adjacency")
            ),
            "uses_bridge_first_frame": bridge_first,
            "uses_explicit_first_keyframe": bool(first_path),
            "uses_explicit_last_keyframe": bool(last_path),
            "flf_anchor_contract": (
                "previous_full_av_latent_plus_authored_destination"
                if continuous_av_requested and slot_index > 0
                else (
                    "previous_native_av_head_plus_authored_destination"
                    if legacy_motion_context_requested and motion_trim_frames
                    else (
                        "authored_opening_and_destination"
                        if continuous_guided_requested
                        else (
                            "shared_planned_timeline_keyframe"
                            if flf_anchor_mode and not hard_cut_start
                            else "independent_or_hard_cut"
                        )
                    )
                )
            ),
            "frame_source": (
                "iamccs_continuous_av_flf_interval"
                if continuous_av_requested
                else ("continuous_flf_interval" if continuous_guided_requested else "timeline_segment_trim")
            ),
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
        unique_frames_total += unique_frames

    reference_image_paths = _timeline_image_paths(timeline)[:4]
    image_count = len(reference_image_paths) or sum(1 for slot in slots if slot.get("image"))
    return {
        "schema": "iamccs.minimax_h3.shotplan",
        "schema_version": 10,
        **({
            "backend_revision": "iamccs-fl2va-continuous-av-v1",
            "backend_variant": (
                "iamccs_guided_av_loop_experimental_v1"
                if experimental_loop_requested else "iamccs_fl2va_continuous_av_v1"
            ),
            "herrgotts_direct_av_chain": {
                "enabled": True,
                "provider": "IAMCCS internal GPLv3 continuation engine",
                "source_attribution": "Herrgotts H3 Infinite Continuation Suite v1.2.1",
                "context_frames": str(max(5, int(motion_context_tail_frames or 22))),
                "handover_mode": "auto" if experimental_loop_requested else "manual",
                "alignment_mode": "phase_aligned_extended",
                "handover_preset": "Balanced",
                "video_crossfade_frames": resolved_overlap_frames,
                "audio_crossfade_ms": 15.0,
                "max_safe_tail_bridge_frames": 2,
                "latent_transport": "direct_full_native_av",
                "preserve_authored_intermediate_keyframes": not experimental_loop_requested,
                "manual_landing_tail_frames": 34 if experimental_loop_requested else 0,
                "freeze_tail_analysis": "automatic" if experimental_loop_requested else "diagnostic_only",
                "recompose_only_reused_context": bool(experimental_loop_requested),
                "semantics": "n_shotboard_guides_n_minus_one_flf_intervals",
            },
            "iamccs_continuous_av": {
                "enabled": True,
                "context_frames": str(max(5, int(motion_context_tail_frames or 22))),
                "handover_mode": "auto" if experimental_loop_requested else "manual",
                "alignment_mode": "phase_aligned_extended",
                "handover_preset": "Balanced",
                "video_crossfade_frames": resolved_overlap_frames,
                "audio_crossfade_ms": 15.0,
                "max_safe_tail_bridge_frames": 2,
                "latent_transport": "direct_full_native_av",
                "preserve_authored_intermediate_keyframes": not experimental_loop_requested,
                "manual_landing_tail_frames": 34 if experimental_loop_requested else 0,
                "freeze_tail_analysis": "automatic" if experimental_loop_requested else "diagnostic_only",
                "recompose_only_reused_context": bool(experimental_loop_requested),
            },
            **({
                "iamccs_guided_av_loop": {
                    "enabled": True,
                    "context_frames": str(max(5, int(motion_context_tail_frames or 22))),
                    "handover_mode": "auto",
                    "alignment_mode": "phase_aligned_extended",
                    "handover_preset": "Balanced",
                    "video_crossfade_frames": resolved_overlap_frames,
                    "audio_crossfade_ms": 15.0,
                    "max_safe_tail_bridge_frames": 2,
                    "latent_transport": "direct_full_native_av",
                    "preserve_authored_intermediate_keyframes": False,
                    "manual_landing_tail_frames": 34,
                    "freeze_tail_analysis": "automatic",
                    "recompose_only_reused_context": True,
                }
            } if experimental_loop_requested else {}),
            "mode_contract": {
                "mode": requested_task_mode,
                "public_name": (
                    "GUIDED AV LOOP · EXPERIMENTAL"
                    if experimental_loop_requested else "FL2VA CONTINUOUS AV"
                ),
                "subtitle": "PHASE-ALIGNED LATENT HANDOVER",
                "backend": "iamccs_internal_continuous_av_engine",
                "former_masked_loop_removed": True,
                "legacy_backend_untouched": True,
                "requires": "ComfyUI MiniMax H3 runtime",
            },
        } if continuous_av_requested else ({
            "backend_revision": "long-continuous-guided-v1",
            "backend_variant": "motion_context_auto_chain_v1",
            "motion_context_auto_chain": {
                "enabled": True,
                "provider": "ComfyUI-H3-Motion-Context-Auto-Chain-addon",
                "minimum_provider_version": "0.1.2",
                "context_frames": max(5, int(motion_context_tail_frames or 22)),
                "audio_context_frames": 24 if motion_context_audio else 0,
                "continue_audio": bool(motion_context_audio),
                "latent_transport": "native_av_safetensors",
                "semantics": "fl2va_interval_previous_av_head_plus_next_last_frame",
                "guide_boundary": "safe_handoff",
                "interior_image_guides": "rejected",
                "model_max_frames": H3_MAX_TRAINED_FRAMES,
            },
            "mode_contract": {
                "mode": "longvid_continuous_guided",
                "backend": "isolated_motion_context_flf_intervals",
                "legacy_backend_untouched": True,
                "requires": "ComfyUI-H3-Motion-Context-Auto-Chain-addon>=0.1.2",
            },
        } if continuous_guided_requested else {})),
        "source_timeline_schema": _text(timeline.get("schema")),
        "fps": H3_FPS,
        "width": resolved_width,
        "height": resolved_height,
        "task_mode": task_mode,
        "generation_mode": task_mode,
        "continuation_mode": (
            "herrgotts_phase_aligned_full_av_handover"
            if herrgotts_requested
            else ("direct_native_av_latent_plus_next_shotboard_destination"
            if continuous_guided_requested
            else (
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
            )))
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
            "n_guides_n_minus_one_herrgotts_direct_av_intervals"
            if herrgotts_requested
            else ("n_guides_n_minus_one_continuous_flf_intervals"
            if continuous_guided_requested
            else (
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
            )))
        ),
        "flf_anchor_mode": flf_anchor_mode,
        "i2v_hard_cut_mode": i2v_hard_cut_mode,
        "ref2v_hard_cut_mode": ref2v_hard_cut_mode,
        "lipsync": {
            "enabled": lipsync_requested,
            "schema": "iamccs.minimax_h3.ref2vid_lipsync",
            "schema_version": 2,
            "image_source": "CineInfoH3 reference image preferred; main visual slot fallback",
            "audio_source": "main AudioBoard slot per performance chunk",
            "audio_pipeline": "same_audioboard_chunk_as_ref_audio_1_and_zero_denoise_audio_latent",
            "dialogue_tags_present": bool(chunks) and all("<d>" in str(item.get("prompt", "")).lower() and "</d>" in str(item.get("prompt", "")).lower() for item in chunks),
            "dialogue_tag_contract": "user_supplied_verbatim_only",
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
