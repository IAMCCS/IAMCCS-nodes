from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, List, Tuple


SUPERNODE_LINX_TYPE = "IAMCCS_SUPERNODE_LINX"


DEFAULT_DIALOGUE = {
    "schema": "iamccs.dialogue_script_planner",
    "schema_version": 1,
    "settings": {
        "engine_profile": "stepaudio_editx",
        "timeline_mode": "speaker_stems_for_overlap",
        "default_line_seconds": 2.6,
        "default_gap_seconds": 0.15,
    },
    "speakers": [
        {
            "id": "A",
            "name": "Alice",
            "voice": "voices_examples/female/female_02.wav",
            "reference_text": "The examination and testimony of the experts, enabled the commission to conclude, that five shots may have been fired.",
            "emotion_ref": "happy",
        },
        {
            "id": "B",
            "name": "Bob",
            "voice": "voices_examples/Clint_Eastwood CC3 (enhanced2).wav",
            "reference_text": "You know, after 55 years of doing it, you kind of get an idea of when you're on key and when you're off key.",
            "emotion_ref": "serious",
        },
    ],
    "lines": [
        {
            "id": "line_001",
            "speaker": "Alice",
            "start": 0.0,
            "duration": 2.6,
            "overlap_after": 0.25,
            "emotion": "happy",
            "style": "warm",
            "text": "I thought the room would be empty by now.",
        },
        {
            "id": "line_002",
            "speaker": "Bob",
            "start": 2.35,
            "duration": 2.7,
            "overlap_after": 0.0,
            "emotion": "calm",
            "style": "serious",
            "text": "It is never empty when somebody is still listening.",
        },
    ],
}


def _safe_json(value: Any, fallback: Any) -> Any:
    if isinstance(value, (dict, list)):
        return copy.deepcopy(value)
    try:
        return json.loads(str(value or ""))
    except Exception:
        return copy.deepcopy(fallback)


def _float(value: Any, fallback: float = 0.0) -> float:
    try:
        parsed = float(value)
        if parsed != parsed:
            return fallback
        return parsed
    except Exception:
        return fallback


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _srt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    whole = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        whole += 1
        millis -= 1000
    if whole >= 60:
        minutes += 1
        whole -= 60
    if minutes >= 60:
        hours += 1
        minutes -= 60
    return f"{hours:02d}:{minutes:02d}:{whole:02d},{millis:03d}"


def _line_to_srt(index: int, start: float, end: float, text: str) -> str:
    end = max(start + 0.04, end)
    return f"{index}\n{_srt_time(start)} --> {_srt_time(end)}\n{text}\n"


def _speaker_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "speaker").strip().lower()).strip("_") or "speaker"


def _speaker_lookup(speakers: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for speaker in speakers:
        name = str(speaker.get("name") or speaker.get("id") or "").strip()
        sid = str(speaker.get("id") or name).strip()
        if name:
            lookup[_speaker_key(name)] = speaker
        if sid:
            lookup[_speaker_key(sid)] = speaker
    return lookup


def _format_line_for_engine(line: Dict[str, Any], speaker: Dict[str, Any], engine_profile: str) -> str:
    text = _clean_text(line.get("text"))
    if not text:
        return ""
    name = str(speaker.get("name") or line.get("speaker") or "Speaker").strip()
    emotion = str(line.get("emotion") or "").strip()
    style = str(line.get("style") or "").strip()
    para = str(line.get("paralinguistic") or "").strip()

    if engine_profile == "stepaudio_editx":
        tags: List[str] = []
        if para and para != "none":
            tags.append(f"[{para}]")
        if emotion and emotion != "none":
            tags.append(f"<emotion:{emotion}>")
        if style and style != "none":
            tags.append(f"<style:{style}>")
        return f"[{name}] {text} {' '.join(tags)}".strip()

    if engine_profile == "indextts2":
        emotion_ref = str(line.get("emotion_ref") or speaker.get("emotion_ref") or emotion or "").strip()
        tag = f"[{name}:{emotion_ref}]" if emotion_ref and emotion_ref != "none" else f"[{name}]"
        return f"{tag} {text}"

    if engine_profile == "chatterbox":
        return f"[{name}] {text}"

    return f"{name}: {text}"


def _normalize_lines(data: Dict[str, Any], default_line_seconds: float, default_gap_seconds: float) -> List[Dict[str, Any]]:
    raw_lines = data.get("lines", [])
    if not isinstance(raw_lines, list):
        raw_lines = []
    lines: List[Dict[str, Any]] = []
    cursor = 0.0
    for index, item in enumerate(raw_lines):
        if not isinstance(item, dict):
            continue
        line = dict(item)
        duration = max(0.08, _float(line.get("duration", default_line_seconds), default_line_seconds))
        has_start = line.get("start", "") not in (None, "")
        start = max(0.0, _float(line.get("start", cursor), cursor)) if has_start else cursor
        overlap_after = max(0.0, _float(line.get("overlap_after", 0.0), 0.0))
        line["id"] = str(line.get("id") or f"line_{index + 1:03d}")
        line["speaker"] = str(line.get("speaker") or "Speaker").strip() or "Speaker"
        line["start"] = start
        line["duration"] = duration
        line["end"] = start + duration
        line["overlap_after"] = overlap_after
        line["text"] = _clean_text(line.get("text"))
        if line["text"]:
            lines.append(line)
        cursor = max(cursor, start + duration + default_gap_seconds - overlap_after)
    return sorted(lines, key=lambda item: (_float(item.get("start"), 0.0), str(item.get("speaker", ""))))


def _flatten_lines(lines: List[Dict[str, Any]], gap: float) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    cursor = 0.0
    for line in sorted(lines, key=lambda item: (_float(item.get("start"), 0.0), str(item.get("id", "")))):
        out = dict(line)
        duration = max(0.08, _float(out.get("duration"), 1.0))
        out["start"] = cursor
        out["end"] = cursor + duration
        flattened.append(out)
        cursor = out["end"] + max(0.0, gap)
    return flattened


class IAMCCS_DialogueScriptPlanner:
    """Dialogue script planner for TTS engines with per-speaker SRT stems and overlap metadata."""

    DEFAULT_DIALOGUE_DATA = json.dumps(DEFAULT_DIALOGUE, indent=2, ensure_ascii=False)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dialogue_data": ("STRING", {
                    "default": cls.DEFAULT_DIALOGUE_DATA,
                    "multiline": True,
                    "tooltip": "Edited by the DialogueScriptPlanner UI. Stores speakers, lines, timing, emotions and overlap.",
                }),
                "engine_profile": ([
                    "stepaudio_editx",
                    "chatterbox",
                    "indextts2",
                    "plain",
                ], {"default": "stepaudio_editx"}),
                "timeline_mode": ([
                    "speaker_stems_for_overlap",
                    "flatten_for_single_tts",
                    "preserve_overlap_in_master_srt",
                ], {"default": "speaker_stems_for_overlap"}),
                "default_line_seconds": ("FLOAT", {"default": 2.6, "min": 0.1, "max": 60.0, "step": 0.05}),
                "default_gap_seconds": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 10.0, "step": 0.05}),
            },
            "optional": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
            },
        }

    RETURN_TYPES = (
        SUPERNODE_LINX_TYPE,
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "cine_linx",
        "master_srt",
        "tagged_text",
        "speaker_1_srt",
        "speaker_2_srt",
        "speaker_3_srt",
        "speaker_4_srt",
        "dialogue_json",
        "report",
    )
    FUNCTION = "plan"
    CATEGORY = "IAMCCS/Cine/Audio"

    def plan(
        self,
        dialogue_data: str,
        engine_profile: str,
        timeline_mode: str,
        default_line_seconds: float,
        default_gap_seconds: float,
        cine_linx: Any = None,
    ) -> Tuple[Dict[str, Any], str, str, str, str, str, str, str, str]:
        data = _safe_json(dialogue_data, DEFAULT_DIALOGUE)
        if not isinstance(data, dict):
            data = copy.deepcopy(DEFAULT_DIALOGUE)

        speakers = data.get("speakers", [])
        if not isinstance(speakers, list) or not speakers:
            speakers = copy.deepcopy(DEFAULT_DIALOGUE["speakers"])
        speakers = [dict(item) for item in speakers if isinstance(item, dict)]
        lookup = _speaker_lookup(speakers)

        lines = _normalize_lines(data, float(default_line_seconds), float(default_gap_seconds))
        export_lines = _flatten_lines(lines, float(default_gap_seconds)) if timeline_mode == "flatten_for_single_tts" else lines

        master_parts: List[str] = []
        tagged_parts: List[str] = []
        speaker_srt: Dict[str, List[str]] = {}
        stem_plan: Dict[str, List[Dict[str, Any]]] = {}

        for index, line in enumerate(export_lines, start=1):
            key = _speaker_key(line.get("speaker"))
            speaker = lookup.get(key) or {"name": str(line.get("speaker") or "Speaker")}
            formatted = _format_line_for_engine(line, speaker, engine_profile)
            if not formatted:
                continue
            start = _float(line.get("start"), 0.0)
            end = _float(line.get("end"), start + _float(line.get("duration"), 1.0))
            master_parts.append(_line_to_srt(index, start, end, formatted))
            tagged_parts.append(formatted)

            speaker_name = str(speaker.get("name") or line.get("speaker") or "Speaker")
            speaker_srt.setdefault(speaker_name, []).append(_line_to_srt(len(speaker_srt.get(speaker_name, [])) + 1, start, end, formatted))
            stem_plan.setdefault(speaker_name, []).append({
                "line_id": line.get("id"),
                "speaker": speaker_name,
                "start": start,
                "end": end,
                "duration": max(0.0, end - start),
                "emotion": line.get("emotion", ""),
                "style": line.get("style", ""),
                "overlap_after": line.get("overlap_after", 0.0),
                "text": line.get("text", ""),
            })

        ordered_speakers = [str(item.get("name") or item.get("id") or "") for item in speakers]
        ordered_speakers = [name for name in ordered_speakers if name]
        for name in speaker_srt.keys():
            if name not in ordered_speakers:
                ordered_speakers.append(name)
        stem_outputs = ["".join(speaker_srt.get(name, [])) for name in ordered_speakers[:4]]
        while len(stem_outputs) < 4:
            stem_outputs.append("")

        payload = {
            "schema": "iamccs.dialogue_script_planner",
            "schema_version": 1,
            "engine_profile": engine_profile,
            "timeline_mode": timeline_mode,
            "speakers": speakers,
            "lines": lines,
            "export_lines": export_lines,
            "stem_plan": stem_plan,
            "truth": "Use speaker stem SRT outputs for real overlap. A single TTS branch can switch voices, but it cannot reliably produce simultaneous dialogue.",
        }
        master_srt = "".join(master_parts).strip()
        tagged_text = "\n".join(tagged_parts).strip()
        dialogue_json = json.dumps(payload, indent=2, ensure_ascii=False)
        report = json.dumps({
            "node": "IAMCCS_DialogueScriptPlanner",
            "engine_profile": engine_profile,
            "timeline_mode": timeline_mode,
            "lines": len(lines),
            "speakers": ordered_speakers[:4],
            "has_overlap": any(_float(line.get("overlap_after"), 0.0) > 0.0 for line in lines)
                or any(_float(a.get("end"), 0.0) > _float(b.get("start"), 0.0)
                       for a, b in zip(lines, lines[1:])),
            "routes": {
                "single_voice_or_serial_dialogue": "Use master_srt or tagged_text into Unified TTS Text/SRT.",
                "true_overlap": "Use speaker_1_srt..speaker_4_srt into separate TTS branches, then mix in AudioBoardArranger or an audio mixer.",
                "indextts2_emotion": "Use indextts2 profile and emotion_ref/character emotion tags; emotion refs must exist as engine-compatible refs or aliases.",
            },
        }, indent=2, ensure_ascii=False)

        out_linx = copy.deepcopy(cine_linx) if isinstance(cine_linx, dict) else {
            "type": SUPERNODE_LINX_TYPE,
            "mode": "iamccs_dialogue_script_planner",
            "resources": {},
            "outputs": {},
            "chain": [],
        }
        resources = out_linx.setdefault("resources", {})
        if not isinstance(resources, dict):
            resources = {}
            out_linx["resources"] = resources
        outputs = out_linx.setdefault("outputs", {})
        if not isinstance(outputs, dict):
            outputs = {}
            out_linx["outputs"] = outputs
        resources["dialogue_script_planner"] = payload
        outputs["dialogue_master_srt"] = master_srt
        outputs["dialogue_tagged_text"] = tagged_text
        outputs["dialogue_stem_srt"] = {f"speaker_{index + 1}": stem_outputs[index] for index in range(4)}
        out_linx.setdefault("chain", []).append({"role": "dialogue_script_planner", "name": "IAMCCS_DialogueScriptPlanner"})
        out_linx["resource_keys"] = sorted(resources.keys())

        return (
            out_linx,
            master_srt,
            tagged_text,
            stem_outputs[0],
            stem_outputs[1],
            stem_outputs[2],
            stem_outputs[3],
            dialogue_json,
            report,
        )


NODE_CLASS_MAPPINGS = {
    "IAMCCS_DialogueScriptPlanner": IAMCCS_DialogueScriptPlanner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_DialogueScriptPlanner": "IAMCCS DialogueScript Planner",
}
