from __future__ import annotations

import copy
import json
import os
import re
from typing import Any, Dict, List, Tuple


SUPERNODE_LINX_TYPE = "IAMCCS_SUPERNODE_LINX"


def _path_has_model_files(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    for _root, _dirs, files in os.walk(path):
        if any(str(name).lower().endswith((".safetensors", ".pt", ".bin", ".gguf", ".ckpt")) for name in files):
            return True
    return False


def _engine_profile_installed(engine_profile: str) -> bool:
    profile = str(engine_profile or "").strip().lower()
    if not profile:
        return False
    models_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models"))
    candidates = {
        "tts_audio_suite_chatterbox": [
            os.path.join(models_root, "TTS", "chatterbox"),
            os.path.join(models_root, "chatterbox"),
        ],
        "chatterbox": [
            os.path.join(models_root, "TTS", "chatterbox"),
            os.path.join(models_root, "chatterbox"),
        ],
        "indextts2": [os.path.join(models_root, "TTS", "IndexTTS")],
        "qwen3_tts": [os.path.join(models_root, "Qwen3-TTS")],
        "f5tts": [os.path.join(models_root, "TTS", "F5-TTS")],
        "vibevoice": [os.path.join(models_root, "vibevoice")],
        "sonic": [os.path.join(models_root, "sonic")],
        "plain": [models_root],
        "longcat": [models_root],
    }
    paths = candidates.get(profile)
    if not paths:
        return True
    return any(_path_has_model_files(path) for path in paths)


def _detect_default_engine_profile() -> str:
    for profile in ("tts_audio_suite_chatterbox", "indextts2", "qwen3_tts", "f5tts", "plain"):
        if _engine_profile_installed(profile):
            return profile
    return "plain"


def _resolve_engine_profile(requested: Any) -> str:
    profile = str(requested or "").strip().lower()
    if profile and _engine_profile_installed(profile):
        return profile
    return _detect_default_engine_profile()


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


def _speaker_stem_start_frames(lines: List[Dict[str, Any]], frame_rate: float) -> Dict[str, int]:
    starts: Dict[str, int] = {}
    fps = max(1.0, float(frame_rate))
    for line in lines:
        if not isinstance(line, dict):
            continue
        key = str(line.get("speaker") or line.get("speaker_name") or "").strip()
        if not key:
            continue
        start_frame = max(0, int(round(_float(line.get("start", 0.0), 0.0) * fps)))
        previous = starts.get(key)
        starts[key] = start_frame if previous is None else min(previous, start_frame)
    return starts

def _strip_inline_tags_for_speech(value: Any) -> str:
    text = re.sub(r"\[[^\]]+\]", " ", str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\{[^}]+\}", " ", text)
    return _clean_text(text)


def _speech_word_count(value: Any) -> int:
    text = _strip_inline_tags_for_speech(value)
    return len([part for part in re.split(r"\s+", text) if part.strip()])


def _estimate_line_seconds(value: Any, wpm: float, min_seconds: float, tail_padding: float = 0.22) -> float:
    words = _speech_word_count(value)
    if words <= 0:
        return max(0.08, float(min_seconds))
    return max(float(min_seconds), (float(words) / max(1.0, float(wpm))) * 60.0 + float(tail_padding))


def _dialogue_clone_linx(cine_linx: Any, mode: str) -> Dict[str, Any]:
    out = copy.deepcopy(cine_linx) if isinstance(cine_linx, dict) else {
        "type": SUPERNODE_LINX_TYPE,
        "mode": mode,
        "resources": {},
        "outputs": {},
        "chain": [],
    }
    out["type"] = SUPERNODE_LINX_TYPE
    out["mode"] = mode
    if not isinstance(out.get("resources"), dict):
        out["resources"] = {}
    if not isinstance(out.get("outputs"), dict):
        out["outputs"] = {}
    if not isinstance(out.get("chain"), list):
        out["chain"] = []
    return out


def _dialogue_payload(cine_linx: Dict[str, Any]) -> Dict[str, Any]:
    resources = cine_linx.setdefault("resources", {})
    payload = resources.get("cine_payload")
    if not isinstance(payload, dict):
        payload = {}
        resources["cine_payload"] = payload
    return payload


def _dialogue_refresh_linx(cine_linx: Dict[str, Any]) -> None:
    resources = cine_linx.setdefault("resources", {})
    cine_linx["resource_keys"] = sorted(resources.keys())
    cine_linx["resource_types"] = {key: type(value).__name__ for key, value in resources.items()}


def _tag_value(value: Any, prefix: str) -> str:
    text = str(value or "").strip()
    if not text or text == "none":
        return ""
    return f"<{prefix}:{text}>"


def _para_value(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text == "none":
        return ""
    return f"<{text}>"


def _format_dialogue_text(line: Dict[str, Any], speaker: Dict[str, Any], engine_profile: str, inline_edit_mode: str) -> str:
    text = _clean_text(line.get("text"))
    if not text:
        return ""
    speaker_id = str(speaker.get("id") or line.get("speaker") or "A").strip() or "A"
    speaker_name = str(speaker.get("name") or speaker_id).strip()
    language = str(line.get("language") or speaker.get("language") or "").strip()
    seed = str(line.get("seed") or "").strip()
    prefix_parts = [speaker_name]
    if language:
        prefix_parts.append(language)
    if seed:
        prefix_parts.append(f"seed:{seed}")
    prefix = "[" + "|".join(prefix_parts) + "]"

    if str(engine_profile).lower() in {"plain", "longcat"}:
        return f"{speaker_name}: {text}"

    if str(engine_profile).lower() in {"chatterbox", "tts_audio_suite_chatterbox"} and inline_edit_mode == "metadata_only":
        return f"{prefix} {text}"

    tags = []
    emotion = _tag_value(line.get("emotion"), "emotion")
    style = _tag_value(line.get("style"), "style")
    speed = _tag_value(line.get("speed"), "speed")
    para = _para_value(line.get("paralinguistic"))
    for tag in (para, emotion, style, speed):
        if tag:
            tags.append(tag)
    if bool(line.get("restore_voice", False)):
        tags.append("<restore>")
    return f"{prefix} {text} {''.join(tags)}".strip()


def _build_dialogue_export(data: Dict[str, Any], frame_rate: float, speech_wpm: float, min_line_seconds: float, default_gap_seconds: float, output_mode: str, inline_edit_mode: str) -> Dict[str, Any]:
    global_prompt = str(data.get("global_prompt") or data.get("prompt") or "").strip()
    if not global_prompt:
        global_prompt = "cinematic field and reverse-field dialogue, hard cut coverage, one dominant speaking face per shot, visible mouth movement, natural audio-driven performance, silent listener reaction, stable identities, coherent eyelines"
    speakers = data.get("speakers") if isinstance(data.get("speakers"), list) else []
    if not speakers:
        speakers = copy.deepcopy(DEFAULT_DIALOGUE["speakers"])
    speakers = [dict(item) for item in speakers if isinstance(item, dict)]
    for index, speaker in enumerate(speakers):
        speaker["id"] = str(speaker.get("id") or chr(65 + index))
        speaker["name"] = str(speaker.get("name") or speaker["id"])

    lookup = _speaker_lookup(speakers)
    raw_lines = data.get("lines") if isinstance(data.get("lines"), list) else []
    if not raw_lines:
        raw_lines = copy.deepcopy(DEFAULT_DIALOGUE["lines"])

    lines: List[Dict[str, Any]] = []
    cursor = 0.0
    for index, raw in enumerate(raw_lines):
        if not isinstance(raw, dict):
            continue
        line = dict(raw)
        speaker_token = str(line.get("speaker") or speakers[index % len(speakers)].get("id") or "A").strip()
        speaker = lookup.get(_speaker_key(speaker_token)) or speakers[index % len(speakers)]
        text = _clean_text(line.get("text") or line.get("dialogue"))
        if not text:
            continue
        duration = _float(line.get("duration"), 0.0)
        if duration <= 0:
            duration = _estimate_line_seconds(text, float(speech_wpm), float(min_line_seconds))
        overlap_after = max(0.0, _float(line.get("overlap_after"), 0.0))
        explicit_start = line.get("start", None)
        start = max(0.0, _float(explicit_start, cursor)) if explicit_start not in (None, "") else cursor
        end = max(start + 0.08, start + duration)
        line.update({
            "id": str(line.get("id") or f"line_{index + 1:03d}"),
            "speaker": str(speaker.get("id") or speaker.get("name") or speaker_token),
            "speaker_name": str(speaker.get("name") or speaker_token),
            "start": round(start, 3),
            "duration": round(duration, 3),
            "end": round(end, 3),
            "overlap_after": round(overlap_after, 3),
            "text": text,
            "estimated": bool(_float(raw.get("duration"), 0.0) <= 0),
            "track": max(0, _float(line.get("track"), speakers.index(speaker) if speaker in speakers else index % 2)),
            "ref": int(_float(line.get("ref"), 1 if index % 2 == 0 else 2)),
        })
        lines.append(line)
        cursor = max(cursor, end + float(default_gap_seconds) - overlap_after)

    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    single_track_mode = str(output_mode) == "flatten_for_single_track"
    if single_track_mode:
        cursor = 0.0
        flattened = []
        for line in lines:
            out = dict(line)
            out["start"] = round(cursor, 3)
            out["end"] = round(cursor + float(out["duration"]), 3)
            out["track"] = 0
            flattened.append(out)
            cursor = out["end"] + float(default_gap_seconds)
        export_lines = flattened
    else:
        export_lines = lines
    settings_data = data.get("settings") if isinstance(data.get("settings"), dict) else {}
    zero_start_stems = bool(settings_data.get("speaker_stems_zero_start", False)) and not single_track_mode
    # Speaker stem media should be generated on its own local timebase so the WAV
    # does not contain leading silence. The AudioBoard lane start keeps the real
    # timeline position unless the user explicitly enables A+B @ 0.
    stem_media_offsets: Dict[str, float] = {}
    for line in export_lines:
        key = str(line.get("speaker") or line.get("speaker_name") or "A")
        start = max(0.0, _float(line.get("start"), 0.0))
        stem_media_offsets[key] = min(stem_media_offsets.get(key, start), start)

    engine_profile = _resolve_engine_profile(data.get("settings", {}).get("engine_profile"))
    master_srt_parts: List[str] = []
    tagged_parts: List[str] = []
    stem_srt: Dict[str, List[str]] = {}
    stem_text: Dict[str, List[str]] = {}
    audio_segments: List[Dict[str, Any]] = []
    shotboard_segments: List[Dict[str, Any]] = []
    total_frames = 0
    for index, line in enumerate(export_lines, start=1):
        speaker = lookup.get(_speaker_key(line.get("speaker"))) or lookup.get(_speaker_key(line.get("speaker_name"))) or {"id": line.get("speaker"), "name": line.get("speaker_name")}
        formatted = _format_dialogue_text(line, speaker, engine_profile, inline_edit_mode)
        start = float(line["start"])
        end = float(line["end"])
        master_srt_parts.append(_line_to_srt(index, start, end, formatted))
        tagged_parts.append(formatted)
        key = str(speaker.get("id") or line.get("speaker") or "A")
        stem_media_offset = stem_media_offsets.get(key, 0.0)
        stem_start = max(0.0, start - stem_media_offset)
        stem_end = max(stem_start + 0.08, end - stem_media_offset)
        stem_srt.setdefault(key, []).append(_line_to_srt(len(stem_srt.get(key, [])) + 1, stem_start, stem_end, formatted))
        stem_text.setdefault(key, []).append(formatted)
        visual_start_frames = int(round(start * float(frame_rate)))
        start_frames = int(round(stem_start * float(frame_rate))) if zero_start_stems else visual_start_frames
        length_frames = max(1, int(round((end - start) * float(frame_rate))))
        total_frames = max(total_frames, visual_start_frames + length_frames, start_frames + length_frames)
        audio_segments.append({
            "id": f"dlg_{index:03d}_{_speaker_key(key)}",
            "type": "audio",
            "name": f"{key} {index:02d}",
            "track": 0 if single_track_mode else int(_float(line.get("track"), 0)),
            "start": start_frames,
            "length": length_frames,
            "audioDurationFrames": length_frames,
            "gain": 1.0,
            "pan": 0.0,
            "purpose": "dialogue_pending_tts",
            "speaker": key,
            "speakerName": str(speaker.get("name") or key),
            "dialogueText": str(line.get("text") or ""),
            "ttsText": formatted,
            "emotion": str(line.get("emotion") or "none"),
            "style": str(line.get("style") or "none"),
            "pendingTTS": True,
            "source": "IAMCCS_DialogueTagEditor",
        })
        shotboard_segments.append({
            "id": f"shot_{index:03d}_{_speaker_key(key)}",
            "type": "image",
            "start": visual_start_frames,
            "length": length_frames,
            "ref": int(line.get("ref", 1)),
            "label": str(line.get("label") or f"{key}_{index:02d}"),
            "prompt": str(line.get("local_prompt") or line.get("shot_prompt") or f"hard cut, Speaker {key} close-up, Speaker {key} speaks clearly, visible mouth movement, coherent eyeline"),
            "dialogue": f'{key}: "{line.get("text", "")}"',
            "audio_or_dialogue": f'{key}: "{line.get("text", "")}"',
            "dialogue_pin": True,
            "use_prompt": True,
            "use_guide": True,
            "transition": "hard_cut" if index > 1 else "opening_cut",
        })

    ordered_keys = [str(s.get("id") or s.get("name") or i) for i, s in enumerate(speakers)]
    for key in stem_srt.keys():
        if key not in ordered_keys:
            ordered_keys.append(key)

    duration_seconds = float(total_frames) / max(1.0, float(frame_rate)) if total_frames else 0.0
    local_prompt_parts = [str(seg.get("prompt") or "").strip() for seg in shotboard_segments if isinstance(seg, dict) and str(seg.get("prompt") or "").strip()]
    segment_length_parts = [str(int(seg.get("length", 1))) for seg in shotboard_segments if isinstance(seg, dict)]
    output_track_count = 1 if single_track_mode else max(1, len(ordered_keys))
    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    visual_timeline = {
        "schema": "iamccs.cine.dialogue_tag_editor_timeline",
        "schema_version": 1,
        "frame_rate": float(frame_rate),
        "duration_seconds": duration_seconds,
        "global_prompt": global_prompt,
        "prompt": global_prompt,
        "director_local_prompts": " | ".join(local_prompt_parts),
        "local_prompts": " | ".join(local_prompt_parts),
        "director_segment_lengths": ",".join(segment_length_parts),
        "segment_lengths": ",".join(segment_length_parts),
        "promptrelay_enabled": bool(local_prompt_parts),
        "segments": shotboard_segments,
        "audioSegments": audio_segments,
        "audioTrackCount": output_track_count,
        "audioSyncMode": "timeline_audio",
        "use_custom_audio": False,
    }
    payload = {
        "schema": "iamccs.dialogue_tag_editor",
        "schema_version": 1,
        "global_prompt": global_prompt,
        "local_prompts": local_prompt_parts,
        "segment_lengths": segment_length_parts,
        "settings": {
            **(data.get("settings") if isinstance(data.get("settings"), dict) else {}),
            "engine_profile": engine_profile,
            "output_mode": output_mode,
            "inline_edit_mode": inline_edit_mode,
            "speech_wpm": float(speech_wpm),
            "frame_rate": float(frame_rate),
        },
        "speakers": speakers,
        "lines": lines,
        "export_lines": export_lines,
        "speaker_stem_start_frames": ({key: 0 for key in ordered_keys} if zero_start_stems else _speaker_stem_start_frames(export_lines, frame_rate)) if not single_track_mode else {},
        "master_srt": "".join(master_srt_parts).strip(),
        "tagged_text": "\n".join(tagged_parts).strip(),
        "speaker_srt": {key: "".join(stem_srt.get(key, [])).strip() for key in ordered_keys},
        "speaker_text": {key: "\n".join(stem_text.get(key, [])).strip() for key in ordered_keys},
        # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        "audio_board_template": {
            "schema": "iamccs.audio_board_arranger",
            "schema_version": 1,
            "audioSegments": audio_segments,
            "audioTrackCount": output_track_count,
            "audioSyncMode": "timeline_audio",
            "duration_seconds": duration_seconds,
            "frame_rate": float(frame_rate),
            "speakerStemStartFrames": ({key: 0 for key in ordered_keys} if zero_start_stems else _speaker_stem_start_frames(export_lines, frame_rate)) if not single_track_mode else {},
            "speakerStemsZeroStart": zero_start_stems,
            "masterAudioGain": 1.0,
            "masterAudioNormalize": False,
        },
        "shotboard_timeline": visual_timeline,
        "truth": "DialogueTagEditor writes dialogue/timing/tag metadata into cine_linx. TTS engines remain external and replaceable.",
    }
    return payload


class IAMCCS_DialogueTagEditor:
    """App-style dialogue/tag planner that writes one cine_linx payload for TTS, AudioBoard and Shotboard."""

    DEFAULT_DATA = json.dumps({"schema": "iamccs.dialogue_tag_editor", "schema_version": 2, "global_prompt": "cinematic field and reverse-field dialogue, hard cut coverage, one dominant speaking face per shot, visible mouth movement, natural audio-driven performance, silent listener reaction, stable identities, coherent eyelines", "settings": {"engine_profile": "tts_audio_suite_chatterbox", "output_mode": "speaker_stems_for_overlap", "speaker_stems_zero_start": False, "inline_edit_mode": "metadata_only", "default_gap_seconds": 0.12, "text_theme": "light_boxes", "font_zoom": 1.0}, "speakers": [{"id": "A", "name": "Man A", "voice": "speaker_a_low_tense", "reference_text": "Keep your voice low. We do not know who is listening.", "language": "en"}, {"id": "B", "name": "Man B", "voice": "speaker_b_controlled_whisper", "reference_text": "Good. Now we finally have something worth protecting.", "language": "en"}], "lines": [{"id": "line_001", "speaker": "A", "text": "You said the signal was dead. Then why is that receiver still blinking?", "emotion": "tense", "style": "low", "paralinguistic": "Breathing", "overlap_after": 0.18, "ref": 1, "track": 0, "local_prompt": "hard cut, Man A close-up, Man A speaks clearly, visible mouth movement, tense controlled delivery, Man B listens quietly"}, {"id": "line_002", "speaker": "B", "text": "Because someone on the other side wants us to think we are alone.", "emotion": "serious", "style": "whisper", "paralinguistic": "none", "overlap_after": 0.12, "ref": 2, "track": 1, "local_prompt": "hard cut, Man B close-up, Man B speaks clearly, visible mouth movement, guarded quiet answer, Man A listens quietly"}, {"id": "line_003", "speaker": "A", "text": "If we open that door, we may be giving them exactly what they came for.", "emotion": "fearful", "style": "dry", "paralinguistic": "Sigh", "overlap_after": 0.1, "ref": 1, "track": 0, "local_prompt": "hard cut, Man A tighter close-up, Man A speaks clearly, visible mouth movement, fear held under discipline"}, {"id": "line_004", "speaker": "B", "text": "Then we do not open it. We make them knock twice.", "emotion": "coldness", "style": "authority", "paralinguistic": "none", "overlap_after": 0.0, "ref": 2, "track": 1, "local_prompt": "hard cut, Man B close-up, Man B speaks clearly, visible mouth movement, decisive controlled authority"}]}, indent=2, ensure_ascii=False)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dialogue_data": ("STRING", {
                    "default": cls.DEFAULT_DATA,
                    "multiline": True,
                    "tooltip": "Edited by the IAMCCS Dialogue Tag Editor app UI.",
                }),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "speech_wpm": ("FLOAT", {"default": 130.0, "min": 60.0, "max": 260.0, "step": 1.0}),
                "min_line_seconds": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 30.0, "step": 0.05}),
                "default_gap_seconds": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 10.0, "step": 0.01}),
                "output_mode": (["speaker_stems_for_overlap", "flatten_for_single_track"], {"default": "speaker_stems_for_overlap"}),
                "inline_edit_mode": (["metadata_only", "tts_audio_suite_inline_tags"], {"default": "metadata_only"}),
            },
            "optional": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
            },
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE,)
    RETURN_NAMES = ("cine_linx",)
    FUNCTION = "build"
    CATEGORY = "IAMCCS/Cine/Audio"

    def build(self, dialogue_data, frame_rate, speech_wpm, min_line_seconds, default_gap_seconds, output_mode, inline_edit_mode, cine_linx=None):
        data = _safe_json(dialogue_data, _safe_json(self.DEFAULT_DATA, {}))
        if not isinstance(data, dict):
            data = _safe_json(self.DEFAULT_DATA, {})
        data.setdefault("settings", {})
        data["settings"]["engine_profile"] = _resolve_engine_profile(data.get("settings", {}).get("engine_profile"))
        data["settings"]["output_mode"] = str(output_mode)
        data["settings"]["inline_edit_mode"] = str(inline_edit_mode)
        payload = _build_dialogue_export(data, float(frame_rate), float(speech_wpm), float(min_line_seconds), float(default_gap_seconds), str(output_mode), str(inline_edit_mode))

        out_linx = _dialogue_clone_linx(cine_linx, "iamccs_dialogue_tag_editor")
        resources = out_linx["resources"]
        outputs = out_linx["outputs"]
        cine_payload = _dialogue_payload(out_linx)
        shotboard_timeline = payload.get("shotboard_timeline") if isinstance(payload.get("shotboard_timeline"), dict) else {}
        audio_board_template = payload.get("audio_board_template") if isinstance(payload.get("audio_board_template"), dict) else {}
        speaker_srt = payload.get("speaker_srt") if isinstance(payload.get("speaker_srt"), dict) else {}
        local_prompts = payload.get("local_prompts") if isinstance(payload.get("local_prompts"), list) else []
        segment_lengths = payload.get("segment_lengths") if isinstance(payload.get("segment_lengths"), list) else []
        master_srt = str(payload.get("master_srt") or "")
        tagged_text = str(payload.get("tagged_text") or "")
        timeline_json = json.dumps(shotboard_timeline, ensure_ascii=False, indent=2)
        speaker_srt_json = json.dumps(speaker_srt, ensure_ascii=False, indent=2)
        audio_board_template_json = json.dumps(audio_board_template, ensure_ascii=False, indent=2)
        duration_seconds = float(shotboard_timeline.get("duration_seconds", 0.0) or 0.0)
        resources.update({
            "dialogue_tag_editor": payload,
            "dialogue_script_planner": payload,
            "cine_dialogue_json": json.dumps(payload, ensure_ascii=False, indent=2),
            "cine_dialogue_master_srt": master_srt,
            "cine_dialogue_tagged_text": tagged_text,
            "cine_dialogue_speaker_srt_json": speaker_srt_json,
            "cine_dialogue_audio_board_template_json": audio_board_template_json,
            "cine_dialogue_shotboard_timeline_json": timeline_json,
            "cine_duration_seconds": duration_seconds,
            "cine_global_prompt": payload.get("global_prompt", ""),
            "cine_local_prompts": " | ".join(str(item) for item in local_prompts),
            "cine_segment_lengths": ",".join(str(item) for item in segment_lengths),
            "cine_promptrelay_enabled": bool(local_prompts),
            "cine_board_timeline_data": timeline_json,
        })
        outputs.update({
            "dialogue_json": resources.get("cine_dialogue_json", "{}"),
            "dialogue_master_srt": master_srt,
            "dialogue_tagged_text": tagged_text,
            "dialogue_speaker_srt_json": speaker_srt_json,
            "audio_board_template_json": audio_board_template_json,
            "timeline_data": timeline_json,
            "duration_seconds": duration_seconds,
            "global_prompt": payload.get("global_prompt", ""),
            "local_prompts": " | ".join(str(item) for item in local_prompts),
            "segment_lengths": ",".join(str(item) for item in segment_lengths),
        })
        cine_payload.update({
            "dialogue_tag_editor": True,
            "dialogue": payload,
            "duration_seconds": outputs["duration_seconds"],
            "timeline_data": outputs["timeline_data"],
        })
        out_linx.setdefault("chain", []).append({"role": "dialogue_tag_editor", "name": "IAMCCS_DialogueTagEditor"})
        _dialogue_refresh_linx(out_linx)
        return (out_linx,)


class IAMCCS_DialogueAudioBoardBridge:
    """Bridge DialogueTagEditor cine_linx metadata into AudioBoard-compatible lanes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "lane_mode": (["speaker_stems", "per_line_clips", "single_dialogue_track"], {"default": "speaker_stems"}),
                "write_policy": (["audio_lanes_to_cine_linx", "metadata_only"], {"default": "audio_lanes_to_cine_linx"}),
            },
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE,)
    RETURN_NAMES = ("cine_linx",)
    FUNCTION = "bridge"
    CATEGORY = "IAMCCS/Cine/Audio"

    def bridge(self, cine_linx, frame_rate, lane_mode, write_policy):
        out_linx = _dialogue_clone_linx(cine_linx, "iamccs_dialogue_audio_board_bridge")
        resources = out_linx["resources"]
        outputs = out_linx["outputs"]
        cine_payload = _dialogue_payload(out_linx)
        raw = resources.get("dialogue_tag_editor") or resources.get("dialogue_script_planner")
        payload = raw if isinstance(raw, dict) else _safe_json(raw, {})
        if not isinstance(payload, dict) or "audio_board_template" not in payload:
            payload = _safe_json(resources.get("cine_dialogue_json"), {})
        if not isinstance(payload, dict):
            payload = {}
        template = payload.get("audio_board_template") if isinstance(payload.get("audio_board_template"), dict) else {}
        segments = list(template.get("audioSegments", [])) if isinstance(template.get("audioSegments"), list) else []
        speakers = payload.get("speakers") if isinstance(payload.get("speakers"), list) else []
        if str(lane_mode) == "single_dialogue_track":
            for seg in segments:
                if isinstance(seg, dict):
                    seg["track"] = 0
        elif str(lane_mode) == "speaker_stems":
            speaker_tracks = {str(s.get("id") or s.get("name") or i): i for i, s in enumerate(speakers)}
            for seg in segments:
                if isinstance(seg, dict):
                    seg["track"] = int(speaker_tracks.get(str(seg.get("speaker") or ""), seg.get("track", 0)))
                    seg["stemMode"] = "speaker_stem"
        else:
            for index, seg in enumerate(segments):
                if isinstance(seg, dict):
                    seg["track"] = int(seg.get("track", index % max(1, len(speakers) or 2)))
                    seg["stemMode"] = "per_line_clip"

        track_count = max(1, max([int(seg.get("track", 0)) for seg in segments if isinstance(seg, dict)] or [0]) + 1)
        duration_frames = max([int(seg.get("start", 0)) + int(seg.get("length", 1)) for seg in segments if isinstance(seg, dict)] or [0])
        duration_seconds = float(duration_frames) / max(1.0, float(frame_rate)) if duration_frames else float(template.get("duration_seconds", 0.0) or 0.0)
        audio_board = {
            "schema": "iamccs.audio_board_arranger",
            "schema_version": 1,
            "audioSegments": segments,
            "audioTrackCount": track_count,
            "audioSyncMode": "timeline_audio",
            "duration_seconds": duration_seconds,
            "frame_rate": float(frame_rate),
            "masterAudioGain": 1.0,
            "masterAudioNormalize": False,
            "bridgeStatus": {
                "source": "IAMCCS_DialogueAudioBoardBridge",
                "lane_mode": str(lane_mode),
                "pending_tts": True,
                "note": "Segments are dialogue placeholders until TTS audio files are generated or attached.",
            },
        }
        resources["cine_audio_timeline_json"] = json.dumps(audio_board, ensure_ascii=False, indent=2)
        resources["cine_audio_layers"] = {
            "arranger": audio_board,
            "policy": str(write_policy),
            "source": "IAMCCS_DialogueAudioBoardBridge",
        }
        resources["cine_audio_tracks"] = {
            "source": "IAMCCS_DialogueAudioBoardBridge",
            "segments": segments,
            "track_count": track_count,
            "duration_frames": duration_frames,
            "duration_seconds": duration_seconds,
            "pending_tts": True,
        }
        resources["cine_duration_seconds"] = max(float(resources.get("cine_duration_seconds", 0.0) or 0.0), duration_seconds)
        outputs["audio_timeline_json"] = resources["cine_audio_timeline_json"]
        outputs["duration_seconds"] = resources["cine_duration_seconds"]
        cine_payload.update({
            "audio_board_arranger": True,
            "audioSegments": segments,
            "audioTrackCount": track_count,
            "audioSyncMode": "timeline_audio",
            "use_custom_audio": False,
            "duration_seconds": outputs["duration_seconds"],
        })
        timeline = payload.get("shotboard_timeline") if isinstance(payload.get("shotboard_timeline"), dict) else _safe_json(resources.get("cine_dialogue_shotboard_timeline_json"), {})
        if isinstance(timeline, dict):
            timeline["audioSegments"] = segments
            timeline["audioTrackCount"] = track_count
            timeline["audioSyncMode"] = "timeline_audio"
            timeline["use_custom_audio"] = False
            timeline["duration_seconds"] = max(float(timeline.get("duration_seconds", 0.0) or 0.0), duration_seconds)
            timeline_data = json.dumps(timeline, ensure_ascii=False, indent=2)
            resources["cine_board_timeline_data"] = timeline_data
            outputs["timeline_data"] = timeline_data
            cine_payload["timeline_data"] = timeline_data
        out_linx.setdefault("chain", []).append({"role": "dialogue_audio_board_bridge", "name": "IAMCCS_DialogueAudioBoardBridge"})
        _dialogue_refresh_linx(out_linx)
        return (out_linx,)

NODE_CLASS_MAPPINGS = {
    "IAMCCS_DialogueTagEditor": IAMCCS_DialogueTagEditor,
    "IAMCCS_DialogueAudioBoardBridge": IAMCCS_DialogueAudioBoardBridge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_DialogueTagEditor": "IAMCCS Dialogue Tag Editor",
    "IAMCCS_DialogueAudioBoardBridge": "IAMCCS Dialogue AudioBoard Bridge",
}
