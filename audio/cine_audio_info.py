from __future__ import annotations

import copy
import json
import math
import os
import re
import time
import wave
from typing import Any, Dict, List, Tuple

try:
    import folder_paths
except Exception:  # pragma: no cover - ComfyUI provides this at runtime.
    folder_paths = None

SUPERNODE_LINX_TYPE = "IAMCCS_SUPERNODE_LINX"


def _safe_json(value: Any, fallback: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value or ""))
    except Exception:
        return fallback


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(fallback)


def _safe_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return int(fallback)


def _clone_linx(cine_linx: Any) -> Dict[str, Any]:
    return copy.deepcopy(cine_linx) if isinstance(cine_linx, dict) else {
        "type": SUPERNODE_LINX_TYPE,
        "mode": "iamccs_cine_audio_info",
        "resources": {},
        "outputs": {},
        "chain": [],
        "stages": [],
    }


def _resources(cine_linx: Dict[str, Any]) -> Dict[str, Any]:
    resources = cine_linx.setdefault("resources", {})
    if not isinstance(resources, dict):
        resources = {}
        cine_linx["resources"] = resources
    return resources


def _outputs(cine_linx: Dict[str, Any]) -> Dict[str, Any]:
    outputs = cine_linx.setdefault("outputs", {})
    if not isinstance(outputs, dict):
        outputs = {}
        cine_linx["outputs"] = outputs
    return outputs


def _payload(cine_linx: Dict[str, Any]) -> Dict[str, Any]:
    resources = _resources(cine_linx)
    payload = resources.setdefault("cine_payload", {})
    if not isinstance(payload, dict):
        payload = {}
        resources["cine_payload"] = payload
    return payload


def _refresh_linx(cine_linx: Dict[str, Any]) -> None:
    resources = _resources(cine_linx)
    cine_linx["type"] = SUPERNODE_LINX_TYPE
    cine_linx["mode"] = "iamccs_cine_audio_info"
    cine_linx["resource_keys"] = sorted(resources.keys())
    cine_linx["resource_types"] = {key: type(value).__name__ for key, value in resources.items()}


def _dialogue_payload(resources: Dict[str, Any], outputs: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    raw = resources.get("dialogue_tag_editor") or resources.get("dialogue_script_planner")
    if isinstance(raw, dict):
        return raw
    for key in ("cine_dialogue_json", "dialogue_json"):
        parsed = _safe_json(resources.get(key) or outputs.get(key) or payload.get(key), {})
        if isinstance(parsed, dict) and parsed:
            return parsed
    parsed = payload.get("dialogue")
    return parsed if isinstance(parsed, dict) else {}


def _speaker_order(dialogue: Dict[str, Any]) -> List[str]:
    speakers = dialogue.get("speakers") if isinstance(dialogue.get("speakers"), list) else []
    out = []
    for index, speaker in enumerate(speakers):
        if not isinstance(speaker, dict):
            continue
        out.append(str(speaker.get("id") or speaker.get("name") or index))
    return out or ["A", "B"]


def _dialogue_speaker_stem_start_frames(dialogue: Dict[str, Any], fps: float) -> Dict[str, int]:
    explicit = dialogue.get("speaker_stem_start_frames") if isinstance(dialogue.get("speaker_stem_start_frames"), dict) else {}
    if explicit:
        out = {}
        for key, value in explicit.items():
            name = str(key or "").strip()
            if not name:
                continue
            out[name] = max(0, _safe_int(value, 0))
        if out:
            return out
    template = dialogue.get("audio_board_template") if isinstance(dialogue.get("audio_board_template"), dict) else {}
    explicit_template = template.get("speakerStemStartFrames") if isinstance(template.get("speakerStemStartFrames"), dict) else {}
    if explicit_template:
        out = {}
        for key, value in explicit_template.items():
            name = str(key or "").strip()
            if not name:
                continue
            out[name] = max(0, _safe_int(value, 0))
        if out:
            return out
    lines = dialogue.get("export_lines") if isinstance(dialogue.get("export_lines"), list) else dialogue.get("lines")
    out = {}
    if isinstance(lines, list):
        for line in lines:
            if not isinstance(line, dict):
                continue
            key = str(line.get("speaker") or line.get("speaker_name") or "").strip()
            if not key:
                continue
            start_frame = max(0, int(round(_safe_float(line.get("start", 0.0), 0.0) * max(1.0, float(fps)))))
            previous = out.get(key)
            out[key] = start_frame if previous is None else min(previous, start_frame)
    return out


def _segments_from_linx(resources: Dict[str, Any], outputs: Dict[str, Any], payload: Dict[str, Any], dialogue: Dict[str, Any]) -> List[Dict[str, Any]]:
    tracks = resources.get("cine_audio_tracks")
    if isinstance(tracks, dict) and isinstance(tracks.get("segments"), list):
        return [dict(seg) for seg in tracks.get("segments") if isinstance(seg, dict)]
    if isinstance(payload.get("audioSegments"), list):
        return [dict(seg) for seg in payload.get("audioSegments") if isinstance(seg, dict)]
    template = dialogue.get("audio_board_template") if isinstance(dialogue.get("audio_board_template"), dict) else {}
    if isinstance(template.get("audioSegments"), list):
        return [dict(seg) for seg in template.get("audioSegments") if isinstance(seg, dict)]
    parsed = _safe_json(outputs.get("audio_timeline_json") or resources.get("cine_audio_timeline_json"), {})
    if isinstance(parsed, dict) and isinstance(parsed.get("audioSegments"), list):
        return [dict(seg) for seg in parsed.get("audioSegments") if isinstance(seg, dict)]
    return []


def _timeline_from_linx(resources: Dict[str, Any], outputs: Dict[str, Any], payload: Dict[str, Any], dialogue: Dict[str, Any]) -> Dict[str, Any]:
    for value in (
        resources.get("cine_board_timeline_data"),
        outputs.get("timeline_data"),
        payload.get("timeline_data"),
        dialogue.get("shotboard_timeline") if isinstance(dialogue, dict) else None,
        resources.get("cine_dialogue_shotboard_timeline_json"),
    ):
        parsed = _safe_json(value, {})
        if isinstance(parsed, dict) and parsed:
            return parsed
    return {
        "schema": "iamccs.cine.filmmaker_timeline",
        "schema_version": 2,
        "segments": [],
        "audioSegments": [],
        "audioTrackCount": 2,
        "audioSyncMode": "timeline_audio",
    }


def _max_end_frames(segments: List[Dict[str, Any]]) -> int:
    end = 0
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        end = max(end, _safe_int(seg.get("start", 0), 0) + max(1, _safe_int(seg.get("length", seg.get("audioDurationFrames", 1)), 1)))
    return int(end)


def _srt_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    whole = int(seconds)
    millis = int(round((seconds - whole) * 1000.0))
    if millis >= 1000:
        whole += 1
        millis -= 1000
    h = whole // 3600
    m = (whole % 3600) // 60
    s = whole % 60
    return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"


def _line_to_srt(index: int, start: float, end: float, text: str) -> str:
    return f"{index}\n{_srt_timestamp(start)} --> {_srt_timestamp(end)}\n{text.strip()}\n\n"


def _strip_inline_tts_tags(text: str) -> str:
    # In plain dialogue mode, keep only words meant to be spoken. Metadata stays in cine_linx.
    clean = re.sub(r"<[^>]+>", "", str(text or ""))
    clean = re.sub(r"\[[^\]]+\]", "", clean)
    return re.sub(r"\s+", " ", clean).strip()


def _line_text(line: Dict[str, Any], mode: str) -> str:
    text = str(line.get("ttsText") or line.get("text") or line.get("dialogueText") or "").strip()
    speaker = str(line.get("speaker") or line.get("speakerName") or "").strip()
    if mode == "plain_dialogue":
        return _strip_inline_tts_tags(text)
    if speaker and not text.startswith("[") and mode in {"speaker_tags", "tts_audio_suite_tags"}:
        text = f"[{speaker}|en] {text}"
    if mode == "tts_audio_suite_tags":
        emotion = str(line.get("emotion") or "none").strip()
        style = str(line.get("style") or "none").strip()
        para = str(line.get("paralinguistic") or line.get("para") or "none").strip()
        if emotion and emotion.lower() != "none" and "<emotion:" not in text:
            text += f"<emotion:{emotion}>"
        if style and style.lower() != "none" and "<style:" not in text:
            text += f"<style:{style}>"
        if para and para.lower() != "none" and f"<{para}>" not in text:
            text += f"<{para}>"
    return text


def _export_speaker_srts(dialogue: Dict[str, Any], segments: List[Dict[str, Any]], fps: float, mode: str) -> Dict[str, str]:
    order = _speaker_order(dialogue)
    grouped: Dict[str, List[str]] = {key: [] for key in order}
    lines = dialogue.get("export_lines") if isinstance(dialogue.get("export_lines"), list) else dialogue.get("lines")
    if isinstance(lines, list) and lines:
        counters: Dict[str, int] = {}
        for line in lines:
            if not isinstance(line, dict):
                continue
            key = str(line.get("speaker") or line.get("speakerName") or order[0])
            if key not in grouped:
                grouped[key] = []
            start = _safe_float(line.get("start", 0.0), 0.0)
            duration = _safe_float(line.get("duration", 0.0), 0.0)
            end = _safe_float(line.get("end", start + duration), start + max(0.8, duration))
            counters[key] = counters.get(key, 0) + 1
            grouped[key].append(_line_to_srt(counters[key], start, max(end, start + 0.2), _line_text(line, mode)))
    else:
        counters: Dict[str, int] = {}
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            key = str(seg.get("speaker") or seg.get("speakerName") or order[0])
            if key not in grouped:
                grouped[key] = []
            start = _safe_int(seg.get("start", 0), 0) / max(1.0, fps)
            length = max(1, _safe_int(seg.get("length", 1), 1)) / max(1.0, fps)
            counters[key] = counters.get(key, 0) + 1
            grouped[key].append(_line_to_srt(counters[key], start, start + length, _line_text(seg, mode)))
    return {key: "".join(parts).strip() for key, parts in grouped.items()}


def _export_srt(dialogue: Dict[str, Any], segments: List[Dict[str, Any]], fps: float, mode: str) -> Tuple[str, str]:
    srt = str(dialogue.get("master_srt") or "").strip()
    tagged = str(dialogue.get("tagged_text") or "").strip()
    if srt and mode != "plain_dialogue":
        return srt, tagged

    lines = dialogue.get("export_lines") if isinstance(dialogue.get("export_lines"), list) else dialogue.get("lines")
    parts = []
    text_parts = []
    if isinstance(lines, list) and lines:
        for index, line in enumerate(lines, start=1):
            if not isinstance(line, dict):
                continue
            start = _safe_float(line.get("start", 0.0), 0.0)
            duration = _safe_float(line.get("duration", 0.0), 0.0)
            end = _safe_float(line.get("end", start + duration), start + max(0.8, duration))
            text = _line_text(line, mode)
            parts.append(_line_to_srt(index, start, max(end, start + 0.2), text))
            text_parts.append(text)
    elif segments:
        for index, seg in enumerate(segments, start=1):
            start = _safe_int(seg.get("start", 0), 0) / max(1.0, fps)
            length = max(1, _safe_int(seg.get("length", 1), 1)) / max(1.0, fps)
            text = _line_text(seg, mode)
            parts.append(_line_to_srt(index, start, start + length, text))
            text_parts.append(text)
    return "".join(parts).strip(), "\n".join(text_parts).strip()


def _input_root() -> str:
    if folder_paths is not None:
        try:
            return folder_paths.get_input_directory()
        except Exception:
            pass
    return os.path.abspath(os.path.join(os.getcwd(), "input"))


def _sanitize(value: Any, fallback: str) -> str:
    clean = re.sub(r'[^A-Za-z0-9._-]+', "_", str(value or fallback).strip())
    clean = clean.strip("._")
    return clean[:80] or fallback


def _extract_audio(audio: Any) -> Tuple[Any, int]:
    """Accept standard Comfy AUDIO plus common wrapper shapes used by TTS nodes."""
    if audio is None:
        return None, 0
    if isinstance(audio, (list, tuple)):
        if len(audio) == 1:
            return _extract_audio(audio[0])
        if len(audio) >= 2:
            first, second = audio[0], audio[1]
            if isinstance(first, dict):
                wave, sr = _extract_audio(first)
                return wave, sr or _safe_int(second, 0)
            return first, _safe_int(second, 0)
    if hasattr(audio, "waveform"):
        return getattr(audio, "waveform", None), _safe_int(getattr(audio, "sample_rate", 0), 0)
    if not isinstance(audio, dict):
        return None, 0
    for key in ("audio", "samples", "result"):
        if key in audio:
            wave, sr = _extract_audio(audio.get(key))
            if wave is not None:
                return wave, sr or _safe_int(audio.get("sample_rate", 0), 0)
    inner = audio.get("waveform")
    if isinstance(inner, dict) and "waveform" in inner:
        return inner.get("waveform"), _safe_int(inner.get("sample_rate", audio.get("sample_rate", 0)), 0)
    return inner, _safe_int(audio.get("sample_rate", 0), 0)


def _tensor_to_channels(waveform: Any) -> Any:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch is required to save ComfyUI AUDIO tensors") from exc
    if not torch.is_tensor(waveform):
        raise RuntimeError(f"Unsupported AUDIO waveform type: {type(waveform).__name__}")
    wave_t = waveform.detach().cpu().to(torch.float32)
    while wave_t.dim() > 3:
        wave_t = wave_t[0]
    if wave_t.dim() == 3:
        wave_t = wave_t[0]
    if wave_t.dim() == 1:
        wave_t = wave_t.unsqueeze(0)
    if wave_t.dim() != 2:
        raise RuntimeError(f"Unsupported AUDIO waveform shape: {tuple(wave_t.shape)}")
    # Expected ComfyUI shape is channels x samples. If it looks transposed, fix it.
    if wave_t.shape[0] > 8 and wave_t.shape[1] <= 8:
        wave_t = wave_t.transpose(0, 1)
    if wave_t.shape[0] > 2:
        wave_t = wave_t[:2]
    if wave_t.shape[0] < 1 or wave_t.shape[1] < 1:
        raise RuntimeError("AUDIO waveform is empty")
    return wave_t.clamp(-1.0, 1.0)


def _save_audio_to_input(audio: Any, subfolder: str, prefix: str) -> Tuple[str, float, int, int]:
    waveform, sample_rate = _extract_audio(audio)
    if waveform is None:
        return "", 0.0, 0, 0
    channels = _tensor_to_channels(waveform)
    sample_rate = max(1, int(sample_rate or 44100))
    duration = float(channels.shape[-1]) / float(sample_rate)
    root = _input_root()
    rel_dir = _sanitize(subfolder, "IAMCCS_generated_audio")
    out_dir = os.path.join(root, rel_dir)
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{_sanitize(prefix, 'cine_dialogue')}_{int(time.time() * 1000)}.wav"
    abs_path = os.path.join(out_dir, filename)

    pcm = (channels.transpose(0, 1).numpy() * 32767.0).round().clip(-32768, 32767).astype("<i2")
    with wave.open(abs_path, "wb") as handle:
        handle.setnchannels(int(channels.shape[0]))
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    rel_path = "/".join([rel_dir, filename])
    return rel_path, duration, int(sample_rate), int(channels.shape[-1])


def _build_audio_timeline(segments: List[Dict[str, Any]], track_count: int, duration_seconds: float, fps: float) -> Dict[str, Any]:
    return {
        "schema": "iamccs.audio_board_arranger",
        "schema_version": 1,
        "audioSegments": segments,
        "audioTrackCount": max(1, int(track_count)),
        "audioSyncMode": "timeline_audio",
        "duration_seconds": float(duration_seconds),
        "frame_rate": float(fps),
        "masterAudioGain": 1.0,
        "masterAudioNormalize": False,
        "audioBusMode": "all_tracks",
        "onlyFirstTrack": False,
    }


class IAMCCS_CineAudioInfo:
    """Bidirectional cine_linx audio bridge for TTS, AudioBoard lanes and Shotboard custom audio."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": ([
                    "export_tts_srt",
                    "export_speaker_stems",
                    "inject_generated_audio",
                    "inject_speaker_stems",
                    "prepare_audio_board",
                    "inspect",
                ], {"default": "export_tts_srt"}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "tts_text_mode": ([
                    "tts_audio_suite_tags",
                    "speaker_tags",
                    "plain_dialogue",
                ], {"default": "tts_audio_suite_tags"}),
                "lane_injection_mode": ([
                    "slice_master_by_existing_lanes",
                    "speaker_full_timeline_clips",
                    "single_master_clip",
                    "attach_to_first_lane",
                ], {"default": "slice_master_by_existing_lanes"}),
                "save_subfolder": ("STRING", {"default": "IAMCCS_generated_audio", "multiline": False}),
                "file_prefix": ("STRING", {"default": "dialogue_tts_master", "multiline": False}),
            },
            "optional": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "generated_audio": ("AUDIO",),
                "generated_audio_b": ("AUDIO",),
                "adjusted_srt": ("STRING", {"forceInput": True}),
                "adjusted_srt_b": ("STRING", {"forceInput": True}),
                "timing_report": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE, "STRING", "STRING", "STRING", "FLOAT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("cine_linx", "tts_srt", "tts_text", "audio_timeline_json", "duration_seconds", "report", "speaker_a_srt", "speaker_b_srt")
    FUNCTION = "run"
    CATEGORY = "IAMCCS/Cine/Audio"

    def run(self, mode, frame_rate, tts_text_mode, lane_injection_mode, save_subfolder, file_prefix, cine_linx=None, generated_audio=None, generated_audio_b=None, adjusted_srt="", adjusted_srt_b="", timing_report=""):
        fps = max(1.0, float(frame_rate))
        out_linx = _clone_linx(cine_linx)
        resources = _resources(out_linx)
        outputs = _outputs(out_linx)
        payload = _payload(out_linx)
        dialogue = _dialogue_payload(resources, outputs, payload)
        segments = _segments_from_linx(resources, outputs, payload, dialogue)
        timeline = _timeline_from_linx(resources, outputs, payload, dialogue)
        tts_srt, tts_text = _export_srt(dialogue, segments, fps, str(tts_text_mode))
        speaker_srts = _export_speaker_srts(dialogue, segments, fps, str(tts_text_mode))
        speaker_order = _speaker_order(dialogue)
        speaker_a_srt = speaker_srts.get(speaker_order[0], "") if speaker_order else ""
        speaker_b_srt = speaker_srts.get(speaker_order[1], "") if len(speaker_order) > 1 else ""
        if str(mode) == "export_speaker_stems":
            tts_srt = speaker_a_srt
            tts_text = "\n".join(part.strip() for part in [speaker_a_srt, speaker_b_srt] if part.strip())

        duration_frames = _max_end_frames(segments)
        duration_seconds = float(duration_frames) / fps if duration_frames else _safe_float(timeline.get("duration_seconds", outputs.get("duration_seconds", 0.0)), 0.0)
        injected_file = ""
        audio_duration = 0.0
        sample_rate = 0

        if str(mode) == "inject_speaker_stems":
            source_segments = [dict(seg) for seg in segments if isinstance(seg, dict)]
            file_a, dur_a, sr_a, _samples_a = _save_audio_to_input(generated_audio, save_subfolder, f"{file_prefix}_A")
            file_b, dur_b, sr_b, _samples_b = _save_audio_to_input(generated_audio_b, save_subfolder, f"{file_prefix}_B")
            files_by_speaker = {}
            if speaker_order:
                files_by_speaker[str(speaker_order[0])] = (file_a, dur_a, sr_a)
            if len(speaker_order) > 1:
                files_by_speaker[str(speaker_order[1])] = (file_b, dur_b, sr_b)
            audio_duration = max(float(dur_a), float(dur_b))
            sample_rate = int(sr_a or sr_b or 0)
            if audio_duration > 0:
                audio_frames = max(1, int(math.ceil(audio_duration * fps)))
                duration_frames = max(duration_frames, audio_frames)
                duration_seconds = max(duration_seconds, audio_duration)
                if str(lane_injection_mode) == "speaker_full_timeline_clips" or not segments:
                    speaker_start_frames = _dialogue_speaker_stem_start_frames(dialogue, fps)
                    for source_seg in source_segments:
                        key = str(source_seg.get("speaker") or source_seg.get("speakerName") or "")
                        if not key:
                            continue
                        start_frame = max(0, _safe_int(source_seg.get("start", 0), 0))
                        previous = speaker_start_frames.get(key)
                        speaker_start_frames[key] = start_frame if previous is None else min(previous, start_frame)
                    generated = []
                    for idx, key in enumerate(speaker_order[:2] or ["A", "B"]):
                        file_path, dur, _sr = files_by_speaker.get(str(key), ("", 0.0, 0))
                        if not file_path:
                            continue
                        frames = max(1, int(math.ceil(float(dur or audio_duration) * fps)))
                        start_frame = max(0, _safe_int(speaker_start_frames.get(str(key), 0), 0))
                        generated.append({
                            "id": f"dialogue_tts_{str(key).lower()}_stem",
                            "type": "audio",
                            "name": f"Dialogue {key} Stem",
                            "track": idx,
                            "start": start_frame,
                            "length": frames,
                            "audioDurationFrames": frames,
                            "audioFile": file_path,
                            "fileName": os.path.basename(file_path),
                            "trimStart": 0,
                            "gain": 1.0,
                            "pan": 0.0,
                            "purpose": "dialogue_tts_speaker_stem",
                            "speaker": str(key),
                            "pendingTTS": False,
                            "source": "IAMCCS_CineAudioInfo",
                        })
                    segments = generated
                else:
                    updated = []
                    for seg in segments:
                        clone = dict(seg)
                        key = str(clone.get("speaker") or clone.get("speakerName") or "")
                        file_path, dur, _sr = files_by_speaker.get(key, ("", 0.0, 0))
                        if file_path:
                            audio_frames = max(1, int(math.ceil(float(dur or audio_duration) * fps)))
                            start = max(0, _safe_int(clone.get("start", 0), 0))
                            length = max(1, _safe_int(clone.get("length", clone.get("audioDurationFrames", 1)), 1))
                            if start + length > audio_frames:
                                length = max(1, audio_frames - start)
                            clone.update({
                                "audioFile": file_path,
                                "fileName": os.path.basename(file_path),
                                "trimStart": start,
                                "length": length,
                                "audioDurationFrames": audio_frames,
                                "pendingTTS": False,
                                "source": "IAMCCS_CineAudioInfo",
                            })
                        updated.append(clone)
                    segments = updated

        if str(mode) == "inject_generated_audio":
            injected_file, audio_duration, sample_rate, _samples = _save_audio_to_input(generated_audio, save_subfolder, file_prefix)
            if injected_file:
                audio_frames = max(1, int(math.ceil(audio_duration * fps)))
                duration_frames = max(duration_frames, audio_frames)
                duration_seconds = max(duration_seconds, audio_duration)
                if str(lane_injection_mode) == "single_master_clip" or not segments:
                    segments = [{
                        "id": "dialogue_tts_master",
                        "type": "audio",
                        "name": "Dialogue TTS Master",
                        "track": 0,
                        "start": 0,
                        "length": audio_frames,
                        "audioDurationFrames": audio_frames,
                        "audioFile": injected_file,
                        "fileName": os.path.basename(injected_file),
                        "gain": 1.0,
                        "pan": 0.0,
                        "purpose": "dialogue_tts_master",
                        "source": "IAMCCS_CineAudioInfo",
                    }]
                elif str(lane_injection_mode) == "attach_to_first_lane":
                    first = dict(segments[0])
                    first.update({
                        "audioFile": injected_file,
                        "fileName": os.path.basename(injected_file),
                        "length": audio_frames,
                        "audioDurationFrames": audio_frames,
                        "trimStart": 0,
                        "pendingTTS": False,
                        "source": "IAMCCS_CineAudioInfo",
                    })
                    segments = [first] + [dict(seg) for seg in segments[1:]]
                else:
                    updated = []
                    for seg in segments:
                        clone = dict(seg)
                        start = max(0, _safe_int(clone.get("start", 0), 0))
                        length = max(1, _safe_int(clone.get("length", clone.get("audioDurationFrames", 1)), 1))
                        if start + length > audio_frames:
                            length = max(1, audio_frames - start)
                        clone.update({
                            "audioFile": injected_file,
                            "fileName": os.path.basename(injected_file),
                            "trimStart": start,
                            "length": length,
                            "audioDurationFrames": audio_frames,
                            "pendingTTS": False,
                            "source": "IAMCCS_CineAudioInfo",
                        })
                        updated.append(clone)
                    segments = updated

        track_count = max(1, max([_safe_int(seg.get("track", 0), 0) for seg in segments] or [0]) + 1)
        if str(mode) == "prepare_audio_board":
            # No media is injected here; this is the clean handoff into AudioBoardArranger.
            for seg in segments:
                seg.setdefault("pendingTTS", True)

        has_media = any(str(seg.get("audioFile", "")).strip() or str(seg.get("audioB64", "")).strip() for seg in segments)
        audio_timeline = _build_audio_timeline(segments, track_count, duration_seconds, fps)
        audio_timeline_json = json.dumps(audio_timeline, ensure_ascii=False, indent=2)

        timeline["audioSegments"] = segments
        timeline["audioTrackCount"] = track_count
        timeline["audioSyncMode"] = "timeline_audio"
        timeline["use_custom_audio"] = bool(has_media)
        if duration_seconds > 0:
            timeline["duration_seconds"] = max(_safe_float(timeline.get("duration_seconds", 0.0), 0.0), duration_seconds)
        timeline["audio_data"] = json.dumps({
            "audioSegments": segments,
            "audioTrackCount": track_count,
            "use_custom_audio": bool(has_media),
            "masterAudioGain": 1.0,
            "masterAudioNormalize": False,
            "audioSyncMode": "timeline_audio",
        }, ensure_ascii=False)
        timeline_data = json.dumps(timeline, ensure_ascii=False, indent=2)

        resources.update({
            "cine_audio_info": {
                "mode": str(mode),
                "lane_injection_mode": str(lane_injection_mode),
                "tts_text_mode": str(tts_text_mode),
                "generated_audio_file": injected_file,
                "speaker_stem_files": dict(([(str(speaker_order[0]), file_a if "file_a" in locals() else "")] if speaker_order else []) + ([(str(speaker_order[1]), file_b if "file_b" in locals() else "")] if len(speaker_order) > 1 else [])),
                "generated_audio_duration_seconds": float(audio_duration),
                "generated_audio_sample_rate": int(sample_rate),
                "has_media": bool(has_media),
            },
            "cine_dialogue_master_srt": str(adjusted_srt or "").strip() or tts_srt,
            "cine_dialogue_speaker_a_srt": str(adjusted_srt or "").strip() or speaker_a_srt,
            "cine_dialogue_speaker_b_srt": str(adjusted_srt_b or "").strip() or speaker_b_srt,
            "cine_dialogue_tagged_text": tts_text,
            "cine_audio_timeline_json": audio_timeline_json,
            "cine_audio_tracks": {
                "source": "IAMCCS_CineAudioInfo",
                "segments": segments,
                "all_segments": segments,
                "shotboard_segments": segments,
                "track_count": track_count,
                "duration_frames": int(_max_end_frames(segments)),
                "duration_seconds": float(duration_seconds),
                "has_media": bool(has_media),
            },
            "cine_audio_layers": {
                "arranger": audio_timeline,
                "export": audio_timeline,
                "policy": "cine_audio_info",
                "source": "IAMCCS_CineAudioInfo",
            },
            "cine_board_timeline_data": timeline_data,
            "cine_use_custom_audio": bool(has_media),
            "cine_duration_seconds": float(duration_seconds),
        })
        if timing_report:
            resources["cine_tts_timing_report"] = str(timing_report)
        outputs.update({
            "dialogue_master_srt": resources["cine_dialogue_master_srt"],
            "dialogue_tagged_text": tts_text,
            "audio_timeline_json": audio_timeline_json,
            "timeline_data": timeline_data,
            "duration_seconds": float(duration_seconds),
        })
        payload.update({
            "audio_info": True,
            "audioSegments": segments,
            "audioTrackCount": track_count,
            "audioSyncMode": "timeline_audio",
            "use_custom_audio": bool(has_media),
            "duration_seconds": float(duration_seconds),
            "timeline_data": timeline_data,
        })
        out_linx.setdefault("chain", []).append({"role": "cine_audio_info", "name": "IAMCCS_CineAudioInfo", "mode": str(mode)})
        _refresh_linx(out_linx)

        report = json.dumps({
            "node": "IAMCCS_CineAudioInfo",
            "mode": str(mode),
            "segments": len(segments),
            "tracks": track_count,
            "has_media": bool(has_media),
            "audio_file": injected_file,
            "duration_seconds": float(duration_seconds),
            "tts_srt_chars": len(tts_srt),
            "truth": "CineAudioInfo exports dialogue SRT to TTS and injects generated AUDIO back into Shotboard-compatible AudioBoard lanes through cine_linx.",
        }, ensure_ascii=False, indent=2)

        resources["cine_report"] = report
        outputs["report"] = report
        return out_linx, resources["cine_dialogue_master_srt"], tts_text, audio_timeline_json, float(duration_seconds), report, resources["cine_dialogue_speaker_a_srt"], resources["cine_dialogue_speaker_b_srt"]


NODE_CLASS_MAPPINGS = {
    "IAMCCS_CineAudioInfo": IAMCCS_CineAudioInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_CineAudioInfo": "IAMCCS CineAudioInfo",
}
