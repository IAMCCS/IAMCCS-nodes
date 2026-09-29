# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pure per-chunk conditioning schedule for FL2VA Extended AV."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


SCHEMA = "iamccs.h3.evolving.v1"


def _split_phase_contract(text: str) -> dict[str, str]:
    raw = str(text or "").strip()
    if not raw:
        return {"onset": "", "resolved": "", "sustain": "", "raw": ""}

    tag_patterns = {
        "onset": r"\[ONSET(?:_|\s+)ONCE\]",
        "resolved": r"\[RESOLVED(?:_|\s+)STATE\]",
        "sustain": r"\[(?:THEN(?:_|\s+))?SUSTAIN\]",
    }
    matches = []
    for kind, pattern in tag_patterns.items():
        for match in re.finditer(pattern, raw, flags=re.IGNORECASE):
            matches.append((match.start(), match.end(), kind))
    if not matches:
        return {"onset": "", "resolved": "", "sustain": "", "raw": raw}
    matches.sort(key=lambda item: item[0])
    values = {"onset": "", "resolved": "", "sustain": ""}
    prefix = raw[:matches[0][0]].strip(" ;\n\t")
    for index, (_start, tag_end, kind) in enumerate(matches):
        content_end = matches[index + 1][0] if index + 1 < len(matches) else len(raw)
        content = raw[tag_end:content_end].strip(" ;\n\t")
        if content:
            values[kind] = content
    if prefix and not values["sustain"]:
        values["sustain"] = prefix
    return {**values, "raw": raw}


def _sanitize_positive_continuation(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    patterns = [
        r"^after\s+[^,.;:]+?\s+(?:has|have)\s+finished[,;:\-]*\s*",
        r"^once\s+[^,.;:]+?\s+(?:ends?|finishes?)[,;:\-]*\s*",
        r"\bwithout\s+repeating\s+[^,.;:]+(?:\s+before\s+\d+(?:[.,]\d+)?\s*seconds?)?",
        r"\bdo\s+not\s+repeat\s+[^,.;:]+",
        r"\bno\s+more\s+[^,.;:]+",
        r"\bbefore\s+\d+(?:[.,]\d+)?\s*seconds?\b",
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"^[,;:\-\s]+|[,;:\-\s]+$", "", cleaned).strip()
    return cleaned


def _beat_text_for_chunk(action: str, *, continued: bool) -> str:
    phase = _split_phase_contract(action)
    resolved = _sanitize_positive_continuation(phase["resolved"])
    sustain = _sanitize_positive_continuation(phase["sustain"])
    if not continued:
        parts = []
        if phase["onset"]:
            parts.append(f"[ONSET ONCE] {phase['onset']}")
        if resolved:
            parts.append(f"[RESOLVED STATE] {resolved}")
        if sustain:
            parts.append(f"[THEN SUSTAIN] {sustain}")
        if not parts:
            return _sanitize_positive_continuation(phase["raw"]) or phase["raw"]
        return "; ".join(parts)

    carried_parts = []
    for candidate in (resolved, sustain):
        if candidate and candidate.lower() not in {item.lower() for item in carried_parts}:
            carried_parts.append(candidate)
    carried = "; ".join(carried_parts) or _sanitize_positive_continuation(phase["raw"])
    if not carried:
        return ""
    return (
        "[CARRIED STATE FROM THE PREVIOUS CHUNK] Preserve only the already-established "
        f"state, motion, direction and momentum: {carried}"
    )


def validate_schedule(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    if str(value.get("schema", "")).strip() != SCHEMA:
        raise ValueError(f"Unsupported H3 conditioning schedule: {value.get('schema')}")
    policy = str(value.get("policy", "continuous")).strip().lower()
    if policy not in {"continuous", "evolving"}:
        raise ValueError(f"Unsupported H3 conditioning policy: {policy}")
    beats = value.get("beats", [])
    if not isinstance(beats, list):
        raise ValueError("H3 conditioning beats must be a list.")
    normalized = []
    for index, beat in enumerate(beats):
        if not isinstance(beat, dict):
            raise ValueError("Each H3 conditioning beat must be an object.")
        start = int(beat.get("start_frame", beat.get("startFrame", 0)) or 0)
        end = int(beat.get("end_frame", beat.get("endFrame", start + 1)) or start + 1)
        if start < 0 or end <= start:
            raise ValueError(f"Invalid H3 conditioning beat range at index {index}.")
        normalized.append({**beat, "id": str(beat.get("id") or f"beat_{index + 1}"), "start_frame": start, "end_frame": end})
    calls = value.get("reference_calls", [])
    if not isinstance(calls, list):
        raise ValueError("H3 conditioning reference_calls must be a list.")
    return {
        **value,
        "schema": SCHEMA,
        "policy": policy,
        "global_context": str(value.get("global_context", "") or "").strip(),
        "beats": normalized,
        "reference_calls": [dict(item) for item in calls if isinstance(item, dict)],
    }


def resolve_chunk_conditioning(schedule: dict[str, Any] | None, *, start_frame: int, visible_frames: int, context_prefix_frames: int, fps: int = 24) -> dict[str, Any] | None:
    if not schedule or schedule["policy"] != "evolving":
        return None
    end_frame = start_frame + visible_frames
    active = [beat for beat in schedule["beats"] if beat["start_frame"] < end_frame and beat["end_frame"] > start_frame]
    lines = []
    bindings = []
    for beat in active:
        overlap_start = max(start_frame, beat["start_frame"])
        overlap_end = min(end_frame, beat["end_frame"])
        local_start = (context_prefix_frames + overlap_start - start_frame) / fps
        local_end = (context_prefix_frames + overlap_end - start_frame) / fps
        action = str(beat.get("action") or beat.get("prompt") or beat.get("text") or "").strip()
        details = [action]
        for key in ("body", "gaze", "camera", "emotion", "object_state", "environment_state", "transition_in", "transition_out", "sound"):
            value = beat.get(key)
            if value not in (None, "", {}, []):
                details.append(f"{key.replace('_', ' ')}: {json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value}")
        text = "; ".join(part for part in details if part)
        if text:
            continued = beat["start_frame"] < start_frame
            beat_text = _beat_text_for_chunk(text, continued=continued)
            if beat_text:
                lines.append(f"Timeline {local_start:.2f}s to {local_end:.2f}s: {beat_text}")
        bindings.append({
            "beat_id": beat["id"],
            "global_start_frame": overlap_start,
            "global_end_frame": overlap_end,
            "sample_local_start_frame": context_prefix_frames + overlap_start - start_frame,
            "sample_local_end_frame": context_prefix_frames + overlap_end - start_frame,
            "continued_from_previous_chunk": beat["start_frame"] < start_frame,
        })

    calls = []
    for call in schedule["reference_calls"]:
        at_frame = int(call.get("at_frame", call.get("atFrame", -1)) or -1)
        if start_frame <= at_frame < end_frame:
            calls.append({**call, "sample_local_frame": context_prefix_frames + at_frame - start_frame})

    global_context = schedule["global_context"]
    local_text = "\n".join(lines)
    prompt = "\n\n".join(part for part in (global_context, local_text) if part).strip()
    revision_source = {"schema": SCHEMA, "start": start_frame, "end": end_frame, "prompt": prompt, "bindings": bindings, "reference_calls": calls}
    revision = hashlib.sha256(json.dumps(revision_source, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"prompt": prompt, "local_prompt": local_text, "beat_bindings": bindings, "reference_calls": calls, "conditioning_revision": revision}
