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
        return "Picture 1 defines the complete target frame at 0.00 seconds."
    if task == "fl2va" and has_first and has_last:
        return (
            "Picture 1 defines the complete opening frame at 0.00 seconds. "
            f"Picture 2 defines the complete final frame at {final_seconds:.2f} seconds."
        )
    if task == "fl2va" and has_last:
        return f"Picture 1 defines the complete final frame at {final_seconds:.2f} seconds."
    return ""


def _chunk_task(task_mode: str, audio_mode: str, has_first: bool, has_last: bool) -> str:
    requested = _text(task_mode).lower()
    if requested in {"ref2va", "ref2va_audio", "ref2va_reference"}:
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


def build_shotplan(
    *,
    timeline_data: Any,
    global_prompt: str,
    duration_seconds: float,
    task_mode: str = "auto",
    audio_mode: str = "h3_native_generated",
    prompt_mapping: str = "global_plus_local",
    upscale_mode: str = "off",
    width: int = 1344,
    height: int = 768,
    acceleration: str = "native",
    ref_image_size: str = "match",
    text_encoder_device: str = "cpu_safe_12gb",
    reference_roles: list[str] | tuple[str, ...] | None = None,
    reference_video_role: str = "off",
    reference_audio_role: str = "off",
    sol_conditioning: str = "exact_kv",
    spectrum_profile: str = "conservative_3060",
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
    if acceleration not in {"auto_3060", "native", "h3_sage", "sage", "sage_sol", "spectrum", "sage_spectrum"}:
        raise ValueError(f"accelerazione H3 non valida: {acceleration}")
    ref_image_size = _text(ref_image_size).lower() or "match"
    if ref_image_size not in {"match", "max"}:
        raise ValueError(f"ref_image_size H3 non valido: {ref_image_size}")
    text_encoder_device = _text(text_encoder_device).lower() or "cpu_safe_12gb"
    if text_encoder_device not in {"cpu_safe_12gb", "auto"}:
        raise ValueError(f"device text encoder H3 non valido: {text_encoder_device}")
    sol_conditioning = _text(sol_conditioning).lower() or "exact_kv"
    if sol_conditioning not in {"exact_kv", "exact_kv_and_rows"}:
        raise ValueError(f"Sol-Attn conditioning non valido: {sol_conditioning}")
    spectrum_profile = _text(spectrum_profile).lower() or "conservative_3060"
    if spectrum_profile not in {"conservative_3060", "conservative_quality", "aggressive"}:
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

    fallback_duration = min(H3_MAX_TRAINED_FRAMES / H3_FPS, max(H3_MIN_FRAMES / H3_FPS, 10.0))
    slots = _normalise_slots(timeline, duration_seconds, fallback_duration)

    chunks: list[dict[str, Any]] = []
    prompt_map: list[dict[str, Any]] = []
    unique_frames_total = 0

    for slot_index, slot in enumerate(slots):
        frame_count = int(slot["frame_count"])
        hard_cut_start = slot_index > 0 and slot["transition"] == "hard_cut"
        next_slot = slots[slot_index + 1] if slot_index + 1 < len(slots) else None
        next_is_cut = bool(next_slot and next_slot["transition"] == "hard_cut")
        next_anchor = ""
        if next_slot and not next_is_cut:
            next_anchor = _text(next_slot.get("image"))
        explicit_last = _text(slot.get("explicit_last_image"))
        terminal_anchor = explicit_last or next_anchor
        chunk_index = len(chunks)
        first_path = _text(slot.get("image"))
        bridge_first = bool(chunk_index > 0 and not hard_cut_start and not first_path)
        last_path = terminal_anchor
        has_first = bool(first_path or bridge_first)
        overlap = 1 if chunk_index > 0 and not hard_cut_start and has_first else 0
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
        prompt = "\n\n".join(part for part in (alignment_prompt, creative_prompt) if part).strip()
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
            "unique_frames": frame_count - overlap,
            "first_image": first_path,
            "last_image": last_path,
            "prompt": prompt,
            "creative_prompt": creative_prompt,
            "alignment_prompt": alignment_prompt,
            "local_prompt": slot["prompt"],
            "audio_prompt": slot["audio_prompt"],
            "transition": "hard_cut" if hard_cut_start else "keyframe_adjacency",
            "uses_bridge_first_frame": bridge_first,
            "uses_explicit_first_keyframe": bool(first_path),
            "uses_explicit_last_keyframe": bool(last_path),
            "frame_source": "timeline_segment_trim",
        }
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

    image_count = sum(1 for slot in slots if slot.get("image"))
    return {
        "schema": "iamccs.minimax_h3.shotplan",
        "schema_version": 5,
        "source_timeline_schema": _text(timeline.get("schema")),
        "fps": H3_FPS,
        "width": resolved_width,
        "height": resolved_height,
        "task_mode": task_mode,
        "generation_mode": task_mode,
        "continuation_mode": "timeline_keyframe_adjacency",
        "audio_mode": audio_mode,
        "prompt_mapping": prompt_mapping,
        "acceleration": acceleration,
        "ref_image_size": ref_image_size,
        "text_encoder_device": text_encoder_device,
        "reference_roles": roles,
        "reference_video_role": _text(reference_video_role).lower() or "off",
        "reference_audio_role": _text(reference_audio_role).lower() or "off",
        "sol_conditioning": sol_conditioning,
        "spectrum_profile": spectrum_profile,
        "vram_clean_before_decode": _bool(vram_clean_before_decode, True),
        "rife_mode": rife_mode,
        "upscale_enabled": bool(_bool(upscale_enabled, False)),
        "upscale_mode": active_upscale_mode,
        "chunk_policy": "one_timeline_box_one_h3_chunk",
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
