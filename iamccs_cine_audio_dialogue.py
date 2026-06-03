from __future__ import annotations

import copy
import json
import math
import os
import re
import sys
import tempfile
from typing import Any, Dict, List, Tuple

import torch


SUPERNODE_LINX_TYPE = "IAMCCS_SUPERNODE_LINX"


def _safe_json_loads(value: Any, fallback: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value or ""))
    except Exception:
        return fallback


def _json_report(data: Dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def _clone_linx(cine_linx: Any) -> Dict[str, Any]:
    return copy.deepcopy(cine_linx) if isinstance(cine_linx, dict) else {
        "type": SUPERNODE_LINX_TYPE,
        "mode": "iamccs_cine_audio_dialogue",
        "resources": {},
        "outputs": {},
        "chain": [],
        "stages": [],
    }


def _resources(cine_linx: Dict[str, Any]) -> Dict[str, Any]:
    resources = cine_linx.setdefault("resources", {})
    return resources if isinstance(resources, dict) else {}


def _outputs(cine_linx: Dict[str, Any]) -> Dict[str, Any]:
    outputs = cine_linx.setdefault("outputs", {})
    return outputs if isinstance(outputs, dict) else {}


def _refresh_linx_index(cine_linx: Dict[str, Any]) -> None:
    resources = _resources(cine_linx)
    cine_linx["resource_keys"] = sorted(resources.keys())
    cine_linx["resource_types"] = {key: type(value).__name__ for key, value in resources.items()}


def _payload(cine_linx: Dict[str, Any]) -> Dict[str, Any]:
    resources = _resources(cine_linx)
    payload = resources.get("cine_payload")
    if not isinstance(payload, dict):
        payload = {}
        resources["cine_payload"] = payload
    return payload


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


def _normalise_label(value: Any, fallback: str) -> str:
    text = re.sub(r"[^\w.-]+", "_", str(value or "").strip(), flags=re.UNICODE).strip("_")
    return text or str(fallback)


def _extract_quoted_text(text: Any) -> List[str]:
    pattern = r'"([^"]*)"|\'([^\']*)\'|\u201c([^\u201d]*)\u201d|\u2018([^\u2019]*)\u2019'
    out: List[str] = []
    for match in re.findall(pattern, str(text or "")):
        quoted = next((part for part in match if part), "").strip()
        if quoted:
            out.append(quoted)
    return out


def _count_spoken_words(text: Any) -> int:
    words = str(text or "").split()
    return len([word for word in words if word.strip()])


def _estimate_speech_seconds(spoken_text: Any, wpm: float, additional_time: float = 0.0) -> float:
    word_count = _count_spoken_words(spoken_text)
    if word_count == 0 and float(additional_time) == 0.0:
        return 0.0
    return (float(word_count) / max(1.0, float(wpm)) * 60.0) + float(additional_time)


def _speech_wpm(speed: str, custom_wpm: float) -> float:
    lookup = {
        "slow_100_wpm": 100.0,
        "average_130_wpm": 130.0,
        "fast_160_wpm": 160.0,
    }
    return lookup.get(str(speed), max(1.0, float(custom_wpm or 130.0)))


def _audio_waveform(audio: Any) -> Tuple[torch.Tensor, int]:
    current = audio
    for _ in range(6):
        if isinstance(current, dict):
            if "waveform" in current and "sample_rate" in current:
                waveform = current["waveform"]
                if not torch.is_tensor(waveform):
                    raise ValueError("audio['waveform'] must be a tensor")
                if waveform.ndim == 1:
                    waveform = waveform.view(1, 1, -1)
                elif waveform.ndim == 2:
                    waveform = waveform.unsqueeze(0)
                elif waveform.ndim != 3:
                    raise ValueError(f"Unsupported audio waveform rank: {waveform.ndim}")
                return waveform, int(current["sample_rate"])
            if "audio" in current:
                current = current.get("audio")
                continue
        if isinstance(current, (list, tuple)) and current:
            current = current[0]
            continue
        break
    raise ValueError("Expected ComfyUI AUDIO dict")


def _gain(audio_waveform: torch.Tensor, db: float) -> torch.Tensor:
    return audio_waveform * (10.0 ** (float(db) / 20.0))


def _ensure_channels(waveform: torch.Tensor, channels: int) -> torch.Tensor:
    current = int(waveform.shape[1])
    if current == channels:
        return waveform
    if current == 1 and channels > 1:
        return waveform.repeat(1, channels, 1)
    if channels == 1 and current > 1:
        return waveform.mean(dim=1, keepdim=True)
    if current > channels:
        return waveform[:, :channels, :]
    return torch.cat([waveform, waveform[:, -1:, :].repeat(1, channels - current, 1)], dim=1)


def _pad_to(waveform: torch.Tensor, samples: int) -> torch.Tensor:
    if int(waveform.shape[-1]) >= int(samples):
        return waveform[:, :, : int(samples)]
    pad = torch.zeros(
        (waveform.shape[0], waveform.shape[1], int(samples) - int(waveform.shape[-1])),
        dtype=waveform.dtype,
        device=waveform.device,
    )
    return torch.cat([waveform, pad], dim=-1)


def _resample_waveform(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    if int(source_rate) == int(target_rate):
        return waveform
    try:
        import torchaudio.functional as TAF
        return TAF.resample(waveform, orig_freq=int(source_rate), new_freq=int(target_rate))
    except Exception as exc:
        raise RuntimeError(f"Cannot resample audio from {source_rate}Hz to {target_rate}Hz: {exc}") from exc


def _visual_segments(cine_linx: Dict[str, Any], timeline_data: str = "") -> List[Dict[str, Any]]:
    resources = _resources(cine_linx)
    payload = _payload(cine_linx)
    candidates = [
        resources.get("cine_visual_segments_json"),
        payload.get("visual_segments"),
        timeline_data,
    ]
    for candidate in candidates:
        data = _safe_json_loads(candidate, None)
        if isinstance(data, dict):
            data = data.get("segments", data.get("visual_segments", []))
        if isinstance(data, list):
            return [dict(item) for item in data if isinstance(item, dict)]
    return []


def _speaker_from_text(text: str, default: str) -> Tuple[str, str]:
    raw = str(text or "").strip()
    match = re.match(r"^\s*([ABab12]|speaker\s*[12])\s*[:|-]\s*(.+)$", raw)
    if not match:
        return default, raw
    token = match.group(1).lower().replace(" ", "")
    speaker = "B" if token in {"b", "2", "speaker2"} else "A"
    return speaker, match.group(2).strip()


def _round_up_8n_plus_1(frames: int) -> int:
    frames = max(1, int(frames))
    rem = (frames - 1) % 8
    return frames if rem == 0 else frames + (8 - rem)


class IAMCCS_CineSpeechLength:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": 'Put spoken words inside quotes, for example: "I am ready."'}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1}),
                "additional_time": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.1}),
            },
            "optional": {
                "text_input": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("slow_frame_count", "average_frame_count", "fast_frame_count", "text")
    FUNCTION = "calculate_speech"
    CATEGORY = "IAMCCS/Cine/Audio"

    def calculate_speech(self, text, fps, additional_time=0.0, text_input=None):
        active_text = text_input if isinstance(text_input, str) and text_input.strip() else text
        quoted_text = " ".join(_extract_quoted_text(active_text))
        word_count = _count_spoken_words(quoted_text)

        def calc_frames(wpm: float) -> int:
            if word_count == 0 and float(additional_time) == 0.0:
                return 0
            seconds = (word_count / float(wpm) * 60.0) + float(additional_time)
            return int(math.ceil(seconds * int(fps)))

        return calc_frames(100.0), calc_frames(130.0), calc_frames(160.0), str(active_text or "")


class IAMCCS_CineDialogueDurationPlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "speech_speed": (["average_130_wpm", "slow_100_wpm", "fast_160_wpm", "custom_wpm"], {"default": "average_130_wpm"}),
                "custom_wpm": ("FLOAT", {"default": 130.0, "min": 40.0, "max": 320.0, "step": 1.0}),
                "additional_time_s": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.05}),
                "panel_padding_s": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 10.0, "step": 0.05}),
                "min_panel_s": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 60.0, "step": 0.05}),
                "max_panel_s": ("FLOAT", {"default": 25.0, "min": 0.5, "max": 600.0, "step": 0.1}),
                "speaker_split_mode": (["detect_a_b_prefix", "alternate", "all_to_a"], {"default": "detect_a_b_prefix"}),
                "timeline_update_mode": (["report_only", "write_audio_cues_to_cine_linx", "extend_visual_segments_in_cine_linx"], {"default": "write_audio_cues_to_cine_linx"}),
            },
            "optional": {
                "dialogue_text": ("STRING", {"multiline": True, "forceInput": True}),
                "timeline_data": ("STRING", {"multiline": True, "forceInput": True}),
            },
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE, "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("cine_linx", "dialogue_cues_json", "tts_text_a", "tts_text_b", "panel_plan_json", "segment_lengths", "report")
    FUNCTION = "plan"
    CATEGORY = "IAMCCS/Cine/Audio"

    def _cues_from_text(self, text: str, fps: float, wpm: float, additional_time_s: float, padding_s: float, split_mode: str) -> List[Dict[str, Any]]:
        cues: List[Dict[str, Any]] = []
        current_time = 0.0
        alternate = "A"
        for raw_line in str(text or "").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            speaker = alternate
            start_s = current_time
            spoken = line
            parts = line.split("|", 2)
            if len(parts) == 3:
                speaker, maybe_time, spoken = parts[0].strip().upper()[:1] or "A", parts[1], parts[2].strip()
                start_s = _safe_float(maybe_time, current_time)
            elif len(parts) == 2:
                speaker, spoken = parts[0].strip().upper()[:1] or "A", parts[1].strip()
            elif split_mode == "detect_a_b_prefix":
                speaker, spoken = _speaker_from_text(line, alternate)
            elif split_mode == "all_to_a":
                speaker = "A"
            quoted = _extract_quoted_text(spoken)
            spoken = quoted[0] if quoted else spoken.strip().strip('"').strip("'")
            seconds = _estimate_speech_seconds(spoken, wpm, additional_time_s) + float(padding_s)
            cues.append({
                "speaker": "B" if speaker == "B" else "A",
                "start_s": round(start_s, 3),
                "estimated_duration_s": round(seconds, 3),
                "estimated_frames": int(math.ceil(seconds * float(fps))),
                "text": spoken,
                "source": "dialogue_text",
            })
            current_time = max(current_time, start_s + seconds)
            alternate = "B" if alternate == "A" else "A"
        return cues

    def _cues_from_segments(self, cine_linx: Dict[str, Any], timeline_data: str, fps: float, wpm: float, additional_time_s: float, padding_s: float, split_mode: str) -> List[Dict[str, Any]]:
        cues: List[Dict[str, Any]] = []
        alternate = "A"
        for index, seg in enumerate(sorted(_visual_segments(cine_linx, timeline_data), key=lambda item: _safe_int(item.get("start", item.get("frame", 0)), 0))):
            if str(seg.get("type", "image") or "image").strip().lower() == "audio":
                continue
            raw_text = " ".join(str(seg.get(key, "") or "") for key in ("dialogue", "audio_or_dialogue", "prompt", "local_prompt", "relay_prompt", "note"))
            quoted = _extract_quoted_text(raw_text)
            if not quoted:
                continue
            start_frames = max(0, _safe_int(seg.get("start", seg.get("frame", 0)), 0))
            panel_frames = max(1, _safe_int(seg.get("length", seg.get("len", round(float(fps)))), round(float(fps))))
            for phrase in quoted:
                speaker = str(seg.get("speaker", "") or "").strip().upper()[:1]
                if speaker not in {"A", "B"}:
                    speaker = alternate if split_mode != "all_to_a" else "A"
                seconds = _estimate_speech_seconds(phrase, wpm, additional_time_s) + float(padding_s)
                cues.append({
                    "speaker": speaker,
                    "start_s": round(start_frames / float(fps), 3),
                    "panel_index": index,
                    "panel_start_frames": start_frames,
                    "panel_frames": panel_frames,
                    "estimated_duration_s": round(seconds, 3),
                    "estimated_frames": int(math.ceil(seconds * float(fps))),
                    "text": phrase,
                    "source": "visual_segments",
                    "label": str(seg.get("label", seg.get("name", f"segment_{index + 1}")) or f"segment_{index + 1}"),
                })
                alternate = "B" if alternate == "A" else "A"
        return cues

    def plan(
        self,
        cine_linx,
        fps,
        speech_speed,
        custom_wpm,
        additional_time_s,
        panel_padding_s,
        min_panel_s,
        max_panel_s,
        speaker_split_mode,
        timeline_update_mode,
        dialogue_text=None,
        timeline_data=None,
    ):
        out_linx = _clone_linx(cine_linx)
        wpm = _speech_wpm(str(speech_speed), float(custom_wpm))
        text = str(dialogue_text or "").strip()
        cues = self._cues_from_text(text, float(fps), wpm, float(additional_time_s), float(panel_padding_s), str(speaker_split_mode)) if text else []
        if not cues:
            cues = self._cues_from_segments(out_linx, str(timeline_data or ""), float(fps), wpm, float(additional_time_s), float(panel_padding_s), str(speaker_split_mode))

        panel_plan: List[Dict[str, Any]] = []
        for cue in cues:
            current_panel_s = float(cue.get("panel_frames", 0) or 0) / float(fps) if cue.get("panel_frames") else 0.0
            needed_s = max(float(min_panel_s), min(float(max_panel_s), float(cue["estimated_duration_s"])))
            recommended_s = max(current_panel_s, needed_s)
            panel_plan.append({
                **cue,
                "current_panel_s": round(current_panel_s, 3),
                "recommended_panel_s": round(recommended_s, 3),
                "recommended_panel_frames": int(math.ceil(recommended_s * float(fps))),
                "needs_more_time": bool(current_panel_s > 0 and recommended_s > current_panel_s + 0.001),
            })

        tts_a = "\n".join(item["text"] for item in cues if item.get("speaker") == "A")
        tts_b = "\n".join(item["text"] for item in cues if item.get("speaker") == "B")
        segment_lengths = ",".join(str(item["recommended_panel_frames"]) for item in panel_plan)

        resources = _resources(out_linx)
        outputs = _outputs(out_linx)
        payload = _payload(out_linx)
        resources["cine_dialogue_cues_json"] = json.dumps(cues, ensure_ascii=False, indent=2)
        resources["cine_dialogue_panel_plan_json"] = json.dumps(panel_plan, ensure_ascii=False, indent=2)
        resources["cine_dialogue_tts_text_a"] = tts_a
        resources["cine_dialogue_tts_text_b"] = tts_b
        resources["cine_dialogue_estimated_segment_lengths"] = segment_lengths
        payload["dialogue_duration_planner"] = {
            "wpm": float(wpm),
            "cue_count": len(cues),
            "mode": str(timeline_update_mode),
        }
        outputs["dialogue_cues_json"] = resources["cine_dialogue_cues_json"]

        extension_report: Dict[str, Any] = {"enabled": False}
        if str(timeline_update_mode) == "extend_visual_segments_in_cine_linx" and panel_plan:
            segments = _visual_segments(out_linx, str(timeline_data or ""))
            panel_by_index = {
                int(item["panel_index"]): item
                for item in panel_plan
                if isinstance(item.get("panel_index"), int)
            }
            sorted_pairs = sorted(enumerate(segments), key=lambda pair: _safe_int(pair[1].get("start", pair[1].get("frame", 0)), 0))
            cursor = 0
            changed = False
            total_frames = 0
            updates = []
            for original_index, seg in sorted_pairs:
                if not isinstance(seg, dict):
                    continue
                seg_type = str(seg.get("type", "image") or "image").strip().lower()
                original_start = max(0, _safe_int(seg.get("start", seg.get("frame", 0)), 0))
                original_length = max(1, _safe_int(seg.get("length", seg.get("len", round(float(fps)))), round(float(fps))))
                new_start = max(original_start, cursor)
                new_length = original_length
                plan_item = panel_by_index.get(original_index)
                if plan_item and seg_type != "audio":
                    new_length = max(original_length, int(plan_item["recommended_panel_frames"]))
                    seg["dialogue_estimated_frames"] = int(plan_item["estimated_frames"])
                    seg["dialogue_recommended_frames"] = int(plan_item["recommended_panel_frames"])
                    seg["dialogue_needs_more_time"] = bool(plan_item["needs_more_time"])
                if int(seg.get("start", seg.get("frame", 0)) or 0) != new_start:
                    seg["start"] = int(new_start)
                    changed = True
                if int(seg.get("length", seg.get("len", 0)) or 0) != new_length:
                    seg["length"] = int(new_length)
                    changed = True
                cursor = int(new_start + new_length)
                total_frames = max(total_frames, cursor)
                updates.append({
                    "index": original_index,
                    "type": seg_type,
                    "old_start": original_start,
                    "new_start": new_start,
                    "old_length": original_length,
                    "new_length": new_length,
                    "changed": bool(new_start != original_start or new_length != original_length),
                })
            for item in panel_plan:
                idx = item.get("panel_index")
                if isinstance(idx, int) and 0 <= idx < len(segments):
                    segments[idx]["dialogue_estimated_frames"] = int(item["estimated_frames"])
            resources["cine_visual_segments_json"] = json.dumps(segments, ensure_ascii=False, indent=2)
            payload["visual_segments"] = segments
            if changed and total_frames > 0:
                duration_s = float(total_frames) / float(fps)
                rounded_frames = _round_up_8n_plus_1(total_frames)
                resources["cine_duration_seconds"] = float(duration_s)
                resources["cine_max_frames"] = int(rounded_frames)
                outputs["duration_seconds"] = float(duration_s)
                outputs["max_frames"] = int(rounded_frames)
                payload["duration_seconds"] = float(duration_s)
                payload["max_frames"] = int(rounded_frames)
            extension_report = {
                "enabled": True,
                "changed": bool(changed),
                "total_frames": int(total_frames),
                "duration_seconds": float(total_frames) / float(fps) if total_frames else 0.0,
                "max_frames_8n_plus_1": int(_round_up_8n_plus_1(total_frames)) if total_frames else 0,
                "updates": updates,
            }

        _refresh_linx_index(out_linx)
        report = _json_report({
            "node": "IAMCCS_CineDialogueDurationPlanner",
            "shotboard_touched": False,
            "wpm": float(wpm),
            "cue_count": len(cues),
            "mode": str(timeline_update_mode),
            "extension": extension_report,
            "tts_a_lines": len([x for x in tts_a.splitlines() if x.strip()]),
            "tts_b_lines": len([x for x in tts_b.splitlines() if x.strip()]),
        })
        return (
            out_linx,
            resources["cine_dialogue_cues_json"],
            tts_a,
            tts_b,
            resources["cine_dialogue_panel_plan_json"],
            segment_lengths,
            report,
        )


class IAMCCS_CineAudioDurationProbe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "target_panel_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 600.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("audio_seconds", "audio_frames", "samples", "sample_rate", "delta_vs_target_s", "report")
    FUNCTION = "probe"
    CATEGORY = "IAMCCS/Cine/Audio"

    def probe(self, audio, fps, target_panel_seconds):
        waveform, sample_rate = _audio_waveform(audio)
        samples = int(waveform.shape[-1])
        seconds = samples / float(sample_rate)
        frames = int(math.ceil(seconds * float(fps)))
        delta = seconds - float(target_panel_seconds)
        return seconds, frames, samples, int(sample_rate), delta, _json_report({
            "node": "IAMCCS_CineAudioDurationProbe",
            "audio_seconds": seconds,
            "audio_frames": frames,
            "samples": samples,
            "sample_rate": int(sample_rate),
            "target_panel_seconds": float(target_panel_seconds),
            "delta_vs_target_s": delta,
        })


class IAMCCS_CineDialogueTimingReconciler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "dialogue_audio": ("AUDIO",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "target_duration_s": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 600.0, "step": 0.01}),
                "update_mode": (["report_only", "store_audio_duration", "extend_cine_duration_if_needed"], {"default": "store_audio_duration"}),
                "tail_padding_s": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 30.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE, "FLOAT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("cine_linx", "dialogue_audio_seconds", "dialogue_audio_frames", "delta_vs_target_s", "report")
    FUNCTION = "reconcile"
    CATEGORY = "IAMCCS/Cine/Audio"

    def reconcile(self, cine_linx, dialogue_audio, fps, target_duration_s, update_mode, tail_padding_s):
        out_linx = _clone_linx(cine_linx)
        waveform, sample_rate = _audio_waveform(dialogue_audio)
        seconds = int(waveform.shape[-1]) / float(sample_rate)
        frames = int(math.ceil(seconds * float(fps)))
        target = float(target_duration_s)
        delta = seconds - target if target > 0 else seconds
        resources = _resources(out_linx)
        payload = _payload(out_linx)
        resources["cine_dialogue_audio_duration_s"] = float(seconds)
        resources["cine_dialogue_audio_frames"] = int(frames)
        payload["dialogue_audio_duration_s"] = float(seconds)
        if str(update_mode) == "extend_cine_duration_if_needed":
            current = _safe_float(resources.get("cine_duration_seconds", payload.get("duration_seconds", target)), target)
            new_duration = max(current, seconds + float(tail_padding_s))
            resources["cine_duration_seconds"] = float(new_duration)
            payload["duration_seconds"] = float(new_duration)
        _refresh_linx_index(out_linx)
        return out_linx, float(seconds), int(frames), float(delta), _json_report({
            "node": "IAMCCS_CineDialogueTimingReconciler",
            "mode": str(update_mode),
            "audio_seconds": float(seconds),
            "audio_frames": int(frames),
            "target_duration_s": target,
            "delta_vs_target_s": float(delta),
        })


class IAMCCS_CineWooshFoleyChunkPlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "max_duration_s": ("FLOAT", {"default": 25.0, "min": 0.1, "max": 120.0, "step": 0.1}),
                "chunk_seconds": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 8.0, "step": 0.1}),
                "overlap_seconds": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.1}),
                "seed": ("INT", {"default": 801, "min": 0, "max": 2147483647, "step": 1}),
                "prompt_style": ("STRING", {"default": "cinematic foley, realistic movement, room tone, environmental ambience, object impacts", "multiline": True}),
            },
            "optional": {
                "visual_summary": ("STRING", {"multiline": True, "forceInput": True}),
            },
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE, "STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("cine_linx", "chunk_ranges_json", "chunk_count", "global_foley_prompt", "report")
    FUNCTION = "plan"
    CATEGORY = "IAMCCS/Cine/Audio"

    def plan(self, cine_linx, fps, max_duration_s, chunk_seconds, overlap_seconds, seed, prompt_style, visual_summary=None):
        out_linx = _clone_linx(cine_linx)
        resources = _resources(out_linx)
        payload = _payload(out_linx)
        duration = _safe_float(resources.get("cine_duration_seconds", payload.get("duration_seconds", max_duration_s)), max_duration_s)
        duration = min(max(0.1, duration), float(max_duration_s))
        chunk = min(8.0, max(1.0, float(chunk_seconds)))
        overlap = max(0.0, min(float(overlap_seconds), chunk - 0.1))
        step = max(0.1, chunk - overlap)
        ranges = []
        start = 0.0
        index = 0
        while start < duration - 0.001 and index < 32:
            end = min(duration, start + chunk)
            ranges.append({
                "index": index,
                "start_s": round(start, 3),
                "end_s": round(end, 3),
                "start_frame": int(round(start * float(fps))),
                "end_frame": int(round(end * float(fps))),
                "duration_s": round(end - start, 3),
                "seed": int(seed) + index,
            })
            if end >= duration:
                break
            start += step
            index += 1
        prompt = str(prompt_style or "").strip()
        if str(visual_summary or "").strip():
            prompt = f"{prompt}, {str(visual_summary).strip()}"
        resources["cine_woosh_foley_chunks_json"] = json.dumps(ranges, ensure_ascii=False, indent=2)
        resources["cine_woosh_foley_prompt"] = prompt
        payload["woosh_foley"] = {"chunk_count": len(ranges), "chunk_seconds": chunk, "overlap_seconds": overlap}
        _refresh_linx_index(out_linx)
        return out_linx, resources["cine_woosh_foley_chunks_json"], int(len(ranges)), prompt, _json_report({
            "node": "IAMCCS_CineWooshFoleyChunkPlanner",
            "duration_s": duration,
            "chunk_count": len(ranges),
            "chunk_seconds": chunk,
            "overlap_seconds": overlap,
            "max_single_woosh_seconds": 8.0,
        })


class IAMCCS_CineFinalAudioMixer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dialogue_audio": ("AUDIO",),
                "dialogue_gain_db": ("FLOAT", {"default": 0.0, "min": -60.0, "max": 24.0, "step": 0.1}),
                "foley_gain_db": ("FLOAT", {"default": -6.0, "min": -60.0, "max": 24.0, "step": 0.1}),
                "normalize_peak": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "foley_audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("final_audio", "report")
    FUNCTION = "mix"
    CATEGORY = "IAMCCS/Cine/Audio"

    def mix(self, dialogue_audio, dialogue_gain_db, foley_gain_db, normalize_peak, foley_audio=None):
        dialogue_wave, sr = _audio_waveform(dialogue_audio)
        dialogue_wave = _gain(dialogue_wave.detach().clone(), float(dialogue_gain_db))
        if foley_audio is None:
            mixed = dialogue_wave
            foley_samples = 0
        else:
            foley_wave, foley_sr = _audio_waveform(foley_audio)
            if int(foley_sr) != int(sr):
                raise ValueError(f"Sample rates do not match: dialogue={sr}, foley={foley_sr}")
            foley_wave = _gain(foley_wave.detach().clone(), float(foley_gain_db))
            total = max(int(dialogue_wave.shape[-1]), int(foley_wave.shape[-1]))
            channels = max(int(dialogue_wave.shape[1]), int(foley_wave.shape[1]))
            dialogue_wave = _ensure_channels(_pad_to(dialogue_wave, total), channels)
            foley_wave = _ensure_channels(_pad_to(foley_wave, total), channels)
            mixed = dialogue_wave + foley_wave
            foley_samples = int(foley_wave.shape[-1])
        peak = float(mixed.abs().max().item()) if mixed.numel() else 0.0
        if bool(normalize_peak) and peak > 1.0:
            mixed = mixed / peak
        out = {"waveform": mixed, "sample_rate": int(sr)}
        return out, _json_report({
            "node": "IAMCCS_CineFinalAudioMixer",
            "sample_rate": int(sr),
            "dialogue_samples": int(dialogue_wave.shape[-1]),
            "foley_samples": int(foley_samples),
            "peak_before_normalize": peak,
            "normalized": bool(normalize_peak and peak > 1.0),
        })


class IAMCCS_CineEmotionButtons:
    EMOTIONS = {
        "calm": "calm, intimate, controlled breathing, soft eyes",
        "fear": "fearful, tense, shallow breathing, guarded expression",
        "anger": "angry, clipped diction, tight jaw, contained intensity",
        "sad": "sad, fragile voice, lowered gaze, restrained emotion",
        "joy": "warm joy, brighter voice, open face, light movement",
        "shock": "shocked, broken rhythm, widened eyes, sudden breath",
        "whisper": "whispered, close-mic intimacy, low breath, delicate diction",
        "urgent": "urgent, fast intent, focused eyes, pressured delivery",
        "wonder": "quiet wonder, softened voice, attentive gaze, slow breath",
        "resolve": "resolved, steady voice, grounded posture, clear intention",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "selected_emotions": ("STRING", {"default": "", "multiline": True}),
                "intensity": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "target": (["performance_and_voice", "voice_only", "visual_performance"], {"default": "performance_and_voice"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("emotion_prompt", "voice_prompt", "visual_prompt", "report")
    FUNCTION = "build"
    CATEGORY = "IAMCCS/Cine/Prompt Helpers"

    def build(self, selected_emotions, intensity, target):
        raw = str(selected_emotions or "").strip()
        parts = []
        for token in re.split(r"[,;\n]+", raw):
            key = token.strip().lower().replace(" ", "_")
            if not key:
                continue
            parts.append(self.EMOTIONS.get(key, token.strip()))
        prompt = ", ".join(dict.fromkeys(parts))
        voice = prompt if str(target) in {"performance_and_voice", "voice_only"} else ""
        visual = prompt if str(target) in {"performance_and_voice", "visual_performance"} else ""
        if prompt:
            prompt = f"{prompt}, emotional intensity {float(intensity):.2f}"
        return prompt, voice, visual, _json_report({
            "node": "IAMCCS_CineEmotionButtons",
            "target": str(target),
            "intensity": float(intensity),
            "button_ui": "web/iamccs_cine_audio_dialogue_ui.js",
        })


class IAMCCS_CineDialoguePromptKit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subject_identity": ("STRING", {"default": "the same main character, stable identity", "multiline": True}),
                "place": ("STRING", {"default": "a grounded cinematic location", "multiline": True}),
                "visual_style": ("STRING", {"default": "cinematic realism, coherent lighting, physical camera motion", "multiline": True}),
                "camera_language": ("STRING", {"default": "controlled camera movement, readable blocking", "multiline": True}),
                "story_action": ("STRING", {"default": "the character performs one clear continuous action", "multiline": True}),
                "emotion_prompt": ("STRING", {"default": "", "multiline": True}),
                "dialogue_a": ("STRING", {"default": "", "multiline": True}),
                "dialogue_b": ("STRING", {"default": "", "multiline": True}),
                "duration_seconds": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 600.0, "step": 0.1}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "beat_count": ("INT", {"default": 3, "min": 1, "max": 8, "step": 1}),
            },
            "optional": {
                "local_prompt_1": ("STRING", {"multiline": True, "forceInput": True}),
                "local_prompt_2": ("STRING", {"multiline": True, "forceInput": True}),
                "local_prompt_3": ("STRING", {"multiline": True, "forceInput": True}),
                "local_prompt_4": ("STRING", {"multiline": True, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("global_prompt", "local_prompts", "segment_lengths", "dialogue_script", "shotboard_timeline_json", "markdown_preview", "report")
    FUNCTION = "build"
    CATEGORY = "IAMCCS/Cine/Prompt Helpers"

    def build(
        self,
        subject_identity,
        place,
        visual_style,
        camera_language,
        story_action,
        emotion_prompt,
        dialogue_a,
        dialogue_b,
        duration_seconds,
        frame_rate,
        beat_count,
        local_prompt_1="",
        local_prompt_2="",
        local_prompt_3="",
        local_prompt_4="",
    ):
        locals_in = [str(x or "").strip() for x in (local_prompt_1, local_prompt_2, local_prompt_3, local_prompt_4)]
        local_prompts = [x for x in locals_in if x]
        while len(local_prompts) < int(beat_count):
            local_prompts.append(str(story_action or "the scene continues with coherent cinematic motion"))
        local_prompts = local_prompts[: int(beat_count)]
        total_frames = max(1, int(round(float(duration_seconds) * float(frame_rate))))
        base = total_frames // max(1, int(beat_count))
        lengths = [base] * int(beat_count)
        lengths[-1] += total_frames - sum(lengths)
        global_prompt = ", ".join(part.strip() for part in (
            subject_identity,
            place,
            visual_style,
            camera_language,
            story_action,
            emotion_prompt,
            "dialogue is synchronized when present",
        ) if str(part or "").strip())
        dialogue_lines = []
        if str(dialogue_a or "").strip():
            dialogue_lines.append(f"A|0.0|{str(dialogue_a).strip()}")
        if str(dialogue_b or "").strip():
            dialogue_lines.append(f"B|0.0|{str(dialogue_b).strip()}")
        segments = []
        cursor = 0
        for idx, (prompt, length) in enumerate(zip(local_prompts, lengths), start=1):
            segments.append({
                "id": f"prompt_{idx}",
                "type": "text",
                "start": int(cursor),
                "length": int(length),
                "label": f"prompt_{idx}",
                "prompt": prompt,
                "use_prompt": True,
                "use_guide": False,
            })
            cursor += int(length)
        timeline = {
            "schema": "iamccs.cine.dialogue_prompt_kit",
            "frame_rate": float(frame_rate),
            "duration_seconds": float(duration_seconds),
            "segments": segments,
        }
        markdown = "\n".join([
            "# IAMCCS Dialogue Prompt Kit",
            "",
            "## Global Prompt",
            global_prompt,
            "",
            "## Local Prompts",
            " | ".join(local_prompts),
            "",
            "## Dialogue",
            "\n".join(dialogue_lines),
        ])
        return (
            global_prompt,
            " | ".join(local_prompts),
            ",".join(str(int(x)) for x in lengths),
            "\n".join(dialogue_lines),
            json.dumps(timeline, ensure_ascii=False, indent=2),
            markdown,
            _json_report({
                "node": "IAMCCS_CineDialoguePromptKit",
                "beat_count": int(beat_count),
                "duration_seconds": float(duration_seconds),
                "frame_rate": float(frame_rate),
                "dialogue_a": bool(str(dialogue_a or "").strip()),
                "dialogue_b": bool(str(dialogue_b or "").strip()),
            }),
        )


class IAMCCS_BoardMaker_DialogueFoley:
    """Dialogue-first board writer with foley metadata carried through cine_linx."""

    DEFAULT_DIALOGUE_DATA = json.dumps({
        "lines": [
            {
                "speaker": "A",
                "seconds": 4.0,
                "ref": 1,
                "label": "campo_A",
                "framing": "medium close-up",
                "dialogue": "Non guardare la finestra. Se capisce che l'abbiamo visto, siamo finiti.",
                "voice": "low tense voice, almost whispered",
                "local_prompt": "man A in field shot, tense medium close-up, restrained fear, coherent eyeline",
                "foley": "soft breathing, cloth rustle, distant room tone",
            },
            {
                "speaker": "B",
                "seconds": 3.5,
                "ref": 2,
                "label": "controcampo_B",
                "framing": "reverse angle",
                "dialogue": "L'ha gia capito. Sta aspettando che uno di noi faccia il primo passo.",
                "voice": "controlled low voice, guarded",
                "local_prompt": "man B in reverse field shot, listening then answering, controlled fear",
                "foley": "small chair creak, quiet breath, low room pressure",
            },
            {
                "speaker": "A",
                "seconds": 3.5,
                "ref": 1,
                "label": "ritorno_A",
                "framing": "close-up",
                "dialogue": "Allora il primo passo lo fai tu, lentamente, verso l'uscita.",
                "voice": "firm whisper, urgent but contained",
                "local_prompt": "return to man A close-up, slight push-in, decision in the eyes",
                "foley": "subtle foot shift, jacket movement, controlled breath",
            },
            {
                "speaker": "B",
                "seconds": 4.0,
                "ref": 2,
                "label": "chiusura_B",
                "framing": "reverse close-up",
                "dialogue": "No. Se esco da solo, tu resti qui a spiegargli dove siamo.",
                "voice": "dry, bitter, quietly direct",
                "local_prompt": "man B reverse close-up, hard pause before the last words, tense eye line",
                "foley": "distant door tension, faint wood creak, breath held",
            },
        ]
    }, indent=2, ensure_ascii=False)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "board_name": ("STRING", {"default": "dialogue_foley_board"}),
                "global_prompt": ("STRING", {
                    "default": "Two men in a tense field and reverse-field dialogue scene, stable identity, coherent eyelines, cinematic realism, restrained camera movement.",
                    "multiline": True,
                }),
                "scene_context": ("STRING", {
                    "default": "A quiet interior at night. The two men speak in low voices while something unseen waits beyond the room.",
                    "multiline": True,
                }),
                "foley_prompt": ("STRING", {
                    "default": "realistic cinematic foley, tense room tone, male breathing, cloth movement, subtle chair creaks, distant door pressure, no music",
                    "multiline": True,
                }),
                "dialogue_data": ("STRING", {
                    "default": cls.DEFAULT_DIALOGUE_DATA,
                    "multiline": True,
                    "tooltip": "Edited by the IAMCCS Dialogue/Foley BoardMaker UI. JSON lines: speaker, seconds, ref, label, dialogue, voice, local_prompt, foley.",
                }),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "image_width": ("INT", {"default": 1280, "min": 64, "max": 8192, "step": 32}),
                "image_height": ("INT", {"default": 736, "min": 64, "max": 8192, "step": 32}),
                "default_force": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "guide_policy": (["every_checked_row", "safe_core_guides", "prompt_only"], {"default": "every_checked_row"}),
                "woosh_chunk_seconds": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 8.0, "step": 0.1}),
                "woosh_overlap_seconds": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.1}),
            },
            "optional": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
            },
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE,)
    RETURN_NAMES = ("cine_linx",)
    FUNCTION = "build"
    CATEGORY = "IAMCCS/Cine/01 Prompting"

    @staticmethod
    def _clean(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip())

    @classmethod
    def _inject_speech_token(cls, text: Any, dialogue: Any) -> str:
        value = cls._clean(text)
        speech = cls._clean(dialogue)
        for token in ("<speech1>", "{speech1}", "[speech1]", "<Transcript1>", "{Transcript1}", "[Transcript1]"):
            value = value.replace(token, speech)
        return value

    @classmethod
    def _dialogue_lines(cls, dialogue_data: Any) -> List[Dict[str, Any]]:
        data = _safe_json_loads(dialogue_data, {})
        raw = data if isinstance(data, list) else data.get("lines", []) if isinstance(data, dict) else []
        if not isinstance(raw, list):
            raw = []
        lines: List[Dict[str, Any]] = []
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            speaker = cls._clean(item.get("speaker", "A")).upper()[:1]
            if speaker not in {"A", "B"}:
                speaker = "A" if index % 2 == 0 else "B"
            dialogue = cls._clean(item.get("dialogue", item.get("text", "")))
            local_prompt = cls._clean(item.get("local_prompt", item.get("prompt", "")))
            if not dialogue and not local_prompt:
                continue
            seconds = max(0.25, _safe_float(item.get("seconds", item.get("duration", 3.0)), 3.0))
            lines.append({
                "speaker": speaker,
                "seconds": seconds,
                "ref": max(1, _safe_int(item.get("ref", 1 if speaker == "A" else 2), 1 if speaker == "A" else 2)),
                "label": _normalise_label(cls._clean(item.get("label", "")), f"{'campo_A' if speaker == 'A' else 'controcampo_B'}_{index + 1}"),
                "framing": cls._clean(item.get("framing", "medium close-up" if speaker == "A" else "reverse angle")),
                "dialogue": dialogue,
                "voice": cls._clean(item.get("voice", item.get("voice_direction", ""))),
                "local_prompt": local_prompt or f"speaker {speaker} dialogue performance, coherent eyeline, natural facial movement",
                "foley": cls._clean(item.get("foley", "")),
            })
        return lines

    @staticmethod
    def _chunks(duration: float, chunk: float, overlap: float) -> List[Dict[str, float]]:
        ranges = []
        start = 0.0
        idx = 1
        chunk = max(1.0, min(8.0, float(chunk)))
        overlap = max(0.0, min(float(overlap), chunk - 0.1))
        while start < duration - 0.001 and idx <= 64:
            end = min(duration, start + chunk)
            ranges.append({"index": idx, "start_s": round(start, 3), "end_s": round(end, 3), "duration_s": round(end - start, 3)})
            if end >= duration:
                break
            start = max(start + 0.1, end - overlap)
            idx += 1
        return ranges

    def build(
        self,
        board_name,
        global_prompt,
        scene_context,
        foley_prompt,
        dialogue_data,
        frame_rate,
        image_width,
        image_height,
        default_force,
        guide_policy,
        woosh_chunk_seconds,
        woosh_overlap_seconds,
        cine_linx=None,
    ):
        fps = max(1.0, float(frame_rate))
        lines = self._dialogue_lines(dialogue_data)
        if not lines:
            lines = self._dialogue_lines(self.DEFAULT_DIALOGUE_DATA)
        duration = sum(float(item["seconds"]) for item in lines)
        cursor_frames = 0
        segments: List[Dict[str, Any]] = []
        rows: List[Dict[str, Any]] = []
        dialogue_tracks: List[Dict[str, Any]] = []
        local_prompts: List[str] = []
        segment_lengths: List[int] = []
        for index, item in enumerate(lines):
            length = max(1, int(round(float(item["seconds"]) * fps)))
            start_s = cursor_frames / fps
            prompt_parts = [
                item["local_prompt"],
                item["framing"],
                str(scene_context or "").strip(),
                f'{item["speaker"]}: "{item["dialogue"]}"' if item["dialogue"] else "",
                item["voice"],
            ]
            local_prompt = ", ".join(part for part in (self._clean(p) for p in prompt_parts) if part)
            local_prompts.append(local_prompt)
            segment_lengths.append(length)
            dialogue_text = f'{item["speaker"]}: "{item["dialogue"]}"' if item["dialogue"] else ""
            segment = {
                "id": f"dlg_{index + 1:02d}_{item['speaker'].lower()}",
                "type": "image",
                "start": int(cursor_frames),
                "length": int(length),
                "ref": int(item["ref"]),
                "label": item["label"],
                "prompt": local_prompt,
                "note": item["foley"],
                "camera": item["framing"],
                "transition": "hard_cut" if index else "opening_cut",
                "guideStrength": float(default_force),
                "imageLockStrength": float(default_force),
                "defaultForceSource": float(default_force),
                "forceCustom": True,
                "use_guide": True,
                "use_prompt": True,
                "dialogue_pin": bool(item["dialogue"]),
                "speaker": item["speaker"],
                "dialogue": dialogue_text,
                "audio_or_dialogue": dialogue_text,
                "voice_direction": item["voice"],
                "foley_prompt": item["foley"],
                "step_transition_enabled": False,
                "step_transition_type": "off",
                "step_transition_prompt": "",
                "step_transition_duration": 0,
                "step_transition_arrival": "auto",
                "step_transition_auto_fit": True,
            }
            segments.append(segment)
            rows.append({
                "second": round(start_s, 3),
                "frame": int(cursor_frames),
                "ref": int(item["ref"]),
                "force": float(default_force),
                "image_lock_strength": float(default_force),
                "label": item["label"],
                "camera": item["framing"],
                "transition": segment["transition"],
                "note": item["foley"],
                "relay_prompt": local_prompt,
                "use_guide": True,
                "use_prompt": True,
                "dialogue_pin": bool(item["dialogue"]),
                "speaker": item["speaker"],
                "dialogue": dialogue_text,
                "voice_direction": item["voice"],
                "duration": float(item["seconds"]),
            })
            dialogue_tracks.append({
                "speaker": item["speaker"],
                "start_s": round(start_s, 3),
                "duration_s": round(float(item["seconds"]), 3),
                "ref": int(item["ref"]),
                "label": item["label"],
                "dialogue": item["dialogue"],
                "voice_direction": item["voice"],
            })
            cursor_frames += length

        timeline = {
            "schema": "iamccs.cine.filmmaker_timeline",
            "schema_version": 2,
            "flfrealMode": "iamccs_enhanced",
            "flfreal_mode": "iamccs_enhanced",
            "global_prompt_only": False,
            "use_global_prompt_only": False,
            "promptrelay_enabled": True,
            "use_custom_audio": False,
            "audioSyncMode": "timeline_audio",
            "generationStrategy": "single_timeline",
            "duration_seconds": round(duration, 3),
            "frame_rate": fps,
            "image_width": int(image_width),
            "image_height": int(image_height),
            "image_resize_method": "crop",
            "image_multiple_of": 32,
            "promptrelay_epsilon": 0.6,
            "audioTrackCount": 2,
            "segments": segments,
            "audioSegments": [],
            "rows": rows,
            "director_local_prompts": " | ".join(local_prompts),
            "director_segment_lengths": ",".join(str(x) for x in segment_lengths),
            "local_prompts": " | ".join(local_prompts),
            "segment_lengths": ",".join(str(x) for x in segment_lengths),
            "dialogue_tracks": dialogue_tracks,
            "foley_prompt": str(foley_prompt or ""),
        }
        timeline_data = json.dumps(timeline, ensure_ascii=False, indent=2)
        board = {
            "metadata": {
                "schema": "iamccs.cine.dialogue_foley_board",
                "schema_version": 1,
                "node_type": "IAMCCS_BoardMaker_DialogueFoley",
                "board_name": str(board_name or "dialogue_foley_board"),
                "image_storage": "manual_after_import",
                "notes": "Dialogue/Foley board. Import into Shotboard V3, then add reference images manually.",
            },
            "global_prompt": str(global_prompt or ""),
            "prompt": str(global_prompt or ""),
            "timeline_data": timeline_data,
            "timeline": timeline,
            "rows": rows,
            "settings": {
                "duration_seconds": round(duration, 3),
                "frame_rate": fps,
                "guide_policy": str(guide_policy),
                "min_guide_gap_seconds": 0.0,
                "max_guides": len(rows),
                "default_force": float(default_force),
                "promptrelay_epsilon": 0.6,
                "ltx_round_mode": "up_8n_plus_1",
                "image_width": int(image_width),
                "image_height": int(image_height),
                "image_resize_method": "crop",
                "image_multiple_of": 32,
                "img_compression": 0,
            },
            "duration_seconds": round(duration, 3),
            "frame_rate": fps,
            "image_width": int(image_width),
            "image_height": int(image_height),
            "image_paths": "",
            "images": [],
        }
        board_json = json.dumps(board, ensure_ascii=False, indent=2)
        tts_a = "\n".join(item["dialogue"] for item in dialogue_tracks if item["speaker"] == "A")
        tts_b = "\n".join(item["dialogue"] for item in dialogue_tracks if item["speaker"] == "B")
        dialogue_script = "\n".join(f'{item["speaker"]}: "{item["dialogue"]}"' for item in dialogue_tracks)
        chunks = self._chunks(duration, float(woosh_chunk_seconds), float(woosh_overlap_seconds))

        out_linx = _clone_linx(cine_linx)
        resources = _resources(out_linx)
        outputs = _outputs(out_linx)
        payload = _payload(out_linx)
        payload.update({
            "boardmaker_dialogue_foley": True,
            "board_json": board_json,
            "timeline_data": timeline_data,
            "global_prompt": str(global_prompt or ""),
            "duration_seconds": round(duration, 3),
            "frame_rate": fps,
            "dialogue_tracks": dialogue_tracks,
            "foley_prompt": str(foley_prompt or ""),
            "woosh_chunks": chunks,
        })
        resources.update({
            "cine_board_json": board_json,
            "cine_board_timeline_data": timeline_data,
            "cine_payload": payload,
            "cine_global_prompt": str(global_prompt or ""),
            "cine_duration_seconds": round(duration, 3),
            "cine_frame_rate": fps,
            "cine_image_width": int(image_width),
            "cine_image_height": int(image_height),
            "cine_visual_segments_json": json.dumps(segments, ensure_ascii=False),
            "cine_dialogue_tracks": dialogue_tracks,
            "cine_script_layers": {"dialogue_script": dialogue_script, "tts_text_a": tts_a, "tts_text_b": tts_b},
            "cine_audio_tracks": {"foley_prompt": str(foley_prompt or ""), "woosh_chunks": chunks},
            "cine_helper_patches": {"source": "IAMCCS_BoardMaker_DialogueFoley", "import_into": "Shotboard V3"},
            "cine_group_metadata": {"board_name": str(board_name or ""), "line_count": len(lines)},
        })
        outputs.update({
            "board_json": board_json,
            "timeline_data": timeline_data,
            "dialogue_script": dialogue_script,
            "tts_text_a": tts_a,
            "tts_text_b": tts_b,
            "foley_prompt": str(foley_prompt or ""),
            "duration_seconds": round(duration, 3),
        })
        out_linx["type"] = SUPERNODE_LINX_TYPE
        out_linx["mode"] = "iamccs_boardmaker_dialogue_foley"
        out_linx.setdefault("chain", []).append({"role": "board_writer", "name": "IAMCCS_BoardMaker_DialogueFoley"})
        out_linx.setdefault("stages", []).append({"name": "dialogue_foley_board", "kind": "board_writer", "payload": payload})
        _refresh_linx_index(out_linx)

        report = _json_report({
            "node": "IAMCCS_BoardMaker_DialogueFoley",
            "line_count": len(lines),
            "duration_seconds": round(duration, 3),
            "frame_rate": fps,
            "outputs": ["cine_linx"],
            "cine_info_3": "Connect cine_linx to IAMCCS_CineInfo3 to expose board_json, timeline_data, dialogue_script, TTS text A/B and foley data.",
            "truth": "Small architecture node: writes a Shotboard V3-importable dialogue/foley board and carries the same data through cine_linx.",
        })
        resources["cine_report"] = report
        outputs["report"] = report
        _refresh_linx_index(out_linx)
        return (out_linx,)


class IAMCCS_CineInfo3:
    """Expose BoardMaker dialogue/foley metadata from cine_linx without cluttering the writer node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = (
        "board_json",
        "timeline_data",
        "dialogue_script",
        "tts_text_a",
        "tts_text_b",
        "foley_prompt",
        "woosh_chunks_json",
        "duration_seconds",
        "report",
    )
    FUNCTION = "extract"
    CATEGORY = "IAMCCS/Cine/Info"

    def extract(self, cine_linx):
        linx = _clone_linx(cine_linx)
        resources = _resources(linx)
        outputs = _outputs(linx)
        payload = _payload(linx)
        scripts = resources.get("cine_script_layers") if isinstance(resources.get("cine_script_layers"), dict) else {}
        audio_tracks = resources.get("cine_audio_tracks") if isinstance(resources.get("cine_audio_tracks"), dict) else {}

        board_json = str(outputs.get("board_json") or resources.get("cine_board_json") or payload.get("board_json") or "")
        timeline_data = str(outputs.get("timeline_data") or resources.get("cine_board_timeline_data") or payload.get("timeline_data") or "")
        dialogue_script = str(outputs.get("dialogue_script") or scripts.get("dialogue_script") or "")
        tts_text_a = str(outputs.get("tts_text_a") or scripts.get("tts_text_a") or resources.get("cine_dialogue_tts_text_a") or "")
        tts_text_b = str(outputs.get("tts_text_b") or scripts.get("tts_text_b") or resources.get("cine_dialogue_tts_text_b") or "")
        foley_prompt = str(outputs.get("foley_prompt") or audio_tracks.get("foley_prompt") or payload.get("foley_prompt") or resources.get("cine_woosh_foley_prompt") or "")
        woosh_chunks = audio_tracks.get("woosh_chunks") or payload.get("woosh_chunks") or _safe_json_loads(resources.get("cine_woosh_foley_chunks_json"), [])
        duration = _safe_float(outputs.get("duration_seconds", resources.get("cine_duration_seconds", payload.get("duration_seconds", 0.0))), 0.0)
        woosh_chunks_json = json.dumps(woosh_chunks if isinstance(woosh_chunks, list) else [], ensure_ascii=False, indent=2)
        report = _json_report({
            "node": "IAMCCS_CineInfo3",
            "source_mode": linx.get("mode", ""),
            "resource_keys": sorted(resources.keys()),
            "has_board_json": bool(board_json),
            "has_timeline_data": bool(timeline_data),
            "duration_seconds": duration,
            "woosh_chunk_count": len(_safe_json_loads(woosh_chunks_json, [])),
        })
        return board_json, timeline_data, dialogue_script, tts_text_a, tts_text_b, foley_prompt, woosh_chunks_json, float(duration), report


class IAMCCS_CineSpeech1PromptCompiler:
    """Compile BoardMaker dialogue rows by replacing <speech1> inside local prompts."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "speech_token": ("STRING", {"default": "<speech1>"}),
                "append_scene_context": ("BOOLEAN", {"default": True}),
                "append_voice_style": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE, "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("cine_linx", "global_prompt", "local_prompts", "segment_lengths", "dialogue_script", "foley_prompt", "report")
    FUNCTION = "compile"
    CATEGORY = "IAMCCS/Cine/01 Prompting"

    @staticmethod
    def _text(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip())

    @classmethod
    def _replace_token(cls, text: Any, speech: Any, token: str) -> str:
        value = cls._text(text)
        speech_value = cls._text(speech)
        tokens = [str(token or "<speech1>"), "<speech1>", "{speech1}", "[speech1]", "<Transcript1>", "{Transcript1}", "[Transcript1]"]
        for item in dict.fromkeys(tokens):
            value = value.replace(item, speech_value)
        return value

    @staticmethod
    def _timeline_from_linx(linx: Dict[str, Any]) -> Dict[str, Any]:
        resources = _resources(linx)
        outputs = _outputs(linx)
        payload = _payload(linx)
        for value in (
            outputs.get("timeline_data"),
            resources.get("cine_board_timeline_data"),
            payload.get("timeline_data"),
        ):
            data = _safe_json_loads(value, {})
            if isinstance(data, dict):
                return data
        return {}

    @staticmethod
    def _dialogue_from_linx(linx: Dict[str, Any], timeline: Dict[str, Any]) -> List[Dict[str, Any]]:
        resources = _resources(linx)
        payload = _payload(linx)
        raw = resources.get("cine_dialogue_tracks")
        if not isinstance(raw, list):
            raw = payload.get("dialogue_tracks")
        if not isinstance(raw, list):
            raw = timeline.get("dialogue_tracks")
        if not isinstance(raw, list):
            raw = timeline.get("dialogue_lines")
        if not isinstance(raw, list):
            raw = []
        rows = timeline.get("rows") if isinstance(timeline.get("rows"), list) else []
        result = []
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            row = rows[index] if index < len(rows) and isinstance(rows[index], dict) else {}
            result.append({
                "speaker": str(item.get("speaker", row.get("speaker", "A" if index % 2 == 0 else "B"))).upper()[:1],
                "dialogue": str(item.get("dialogue", row.get("dialogue", ""))).strip().strip('"'),
                "start_s": _safe_float(item.get("start_s", row.get("second", 0.0)), 0.0),
                "duration_s": _safe_float(item.get("duration_s", row.get("duration", item.get("seconds", 3.0))), 3.0),
                "ref": _safe_int(item.get("ref", row.get("ref", 1)), 1),
                "label": str(item.get("label", row.get("label", f"line_{index + 1}"))),
                "voice_direction": str(item.get("voice_direction", item.get("voice", row.get("voice_direction", "")))),
            })
        if result:
            return result
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            dialogue = str(row.get("dialogue", "")).strip().strip('"')
            result.append({
                "speaker": str(row.get("speaker", "A" if index % 2 == 0 else "B")).upper()[:1],
                "dialogue": dialogue,
                "start_s": _safe_float(row.get("second", 0.0), 0.0),
                "duration_s": _safe_float(row.get("duration", 3.0), 3.0),
                "ref": _safe_int(row.get("ref", 1), 1),
                "label": str(row.get("label", f"line_{index + 1}")),
                "voice_direction": str(row.get("voice_direction", "")),
            })
        return result

    def compile(self, cine_linx, speech_token="<speech1>", append_scene_context=True, append_voice_style=True):
        linx = _clone_linx(cine_linx)
        resources = _resources(linx)
        outputs = _outputs(linx)
        payload = _payload(linx)
        timeline = self._timeline_from_linx(linx)
        rows = timeline.get("rows") if isinstance(timeline.get("rows"), list) else []
        segments = timeline.get("segments") if isinstance(timeline.get("segments"), list) else []
        dialogue = self._dialogue_from_linx(linx, timeline)
        global_prompt = str(outputs.get("global_prompt") or resources.get("cine_global_prompt") or payload.get("global_prompt") or timeline.get("cine_global_prompt") or "")
        scene_context = str(payload.get("scene_context") or timeline.get("scene_context") or "")
        audio_resource = resources.get("cine_audio_tracks") if isinstance(resources.get("cine_audio_tracks"), dict) else {}
        foley_prompt = str(outputs.get("foley_prompt") or audio_resource.get("foley_prompt") or "")
        if not foley_prompt:
            foley_prompt = str(payload.get("foley_prompt") or timeline.get("foley_prompt") or "")
        fps = max(1.0, _safe_float(timeline.get("frame_rate", resources.get("cine_frame_rate", payload.get("frame_rate", 24))), 24.0))

        compiled_prompts: List[str] = []
        segment_lengths: List[int] = []
        compiled_tracks: List[Dict[str, Any]] = []
        for index, item in enumerate(dialogue):
            row = rows[index] if index < len(rows) and isinstance(rows[index], dict) else {}
            segment = segments[index] if index < len(segments) and isinstance(segments[index], dict) else {}
            base_prompt = row.get("relay_prompt") or row.get("local_prompt") or row.get("prompt") or segment.get("prompt") or ""
            compiled = self._replace_token(base_prompt, item.get("dialogue", ""), str(speech_token or "<speech1>"))
            extras = []
            if bool(append_scene_context) and scene_context:
                extras.append(scene_context)
            if bool(append_voice_style) and item.get("voice_direction"):
                extras.append(str(item.get("voice_direction")))
            if extras:
                compiled = ", ".join(part for part in [compiled, *[self._text(v) for v in extras]] if part)
            compiled_prompts.append(compiled)
            duration_s = max(0.01, _safe_float(item.get("duration_s"), 3.0))
            frames = max(1, int(round(duration_s * fps)))
            segment_lengths.append(frames)
            item = dict(item)
            item["local_prompt_compiled"] = compiled
            item["duration_s"] = round(duration_s, 3)
            item["frames"] = frames
            compiled_tracks.append(item)
            if index < len(rows) and isinstance(rows[index], dict):
                rows[index]["relay_prompt"] = compiled
                rows[index]["local_prompt"] = compiled
                rows[index]["prompt"] = compiled
                rows[index]["use_prompt"] = True
                rows[index]["dialogue"] = f'{item.get("speaker", "")}: "{item.get("dialogue", "")}"'.strip()
            if index < len(segments) and isinstance(segments[index], dict):
                segments[index]["prompt"] = compiled
                segments[index]["dialogue"] = f'{item.get("speaker", "")}: "{item.get("dialogue", "")}"'.strip()
                segments[index]["length"] = frames

        if rows:
            timeline["rows"] = rows
        if segments:
            timeline["segments"] = segments
        local_prompts = " | ".join(compiled_prompts)
        segment_lengths_s = ",".join(str(item) for item in segment_lengths)
        dialogue_script = "\n".join(f'{item.get("speaker", "")}: "{item.get("dialogue", "")}"' for item in compiled_tracks)
        timeline.update({
            "cine_global_prompt": global_prompt,
            "global_prompt": global_prompt,
            "promptrelay_enabled": bool(local_prompts.strip()),
            "director_local_prompts": local_prompts,
            "director_segment_lengths": segment_lengths_s,
            "local_prompts": local_prompts,
            "segment_lengths": segment_lengths_s,
            "dialogue_tracks": compiled_tracks,
            "foley_prompt": foley_prompt,
        })
        timeline_data = json.dumps(timeline, ensure_ascii=False, indent=2)
        board = _safe_json_loads(outputs.get("board_json") or resources.get("cine_board_json") or payload.get("board_json"), {})
        if isinstance(board, dict):
            board["global_prompt"] = global_prompt
            board["prompt"] = global_prompt
            board["timeline_data"] = timeline_data
            board["timeline"] = timeline
            board["rows"] = rows
            board_json = json.dumps(board, ensure_ascii=False, indent=2)
        else:
            board_json = ""

        scripts = {"dialogue_script": dialogue_script}
        for speaker in ("A", "B"):
            scripts[f"tts_text_{speaker.lower()}"] = "\n".join(item.get("dialogue", "") for item in compiled_tracks if item.get("speaker") == speaker)
        resources.update({
            "cine_board_json": board_json,
            "cine_board_timeline_data": timeline_data,
            "cine_global_prompt": global_prompt,
            "cine_local_prompts": local_prompts,
            "cine_segment_lengths": segment_lengths_s,
            "cine_dialogue_tracks": compiled_tracks,
            "cine_script_layers": scripts,
            "cine_speech1_compiled": True,
        })
        audio_tracks = resources.get("cine_audio_tracks") if isinstance(resources.get("cine_audio_tracks"), dict) else {}
        audio_tracks["foley_prompt"] = foley_prompt
        resources["cine_audio_tracks"] = audio_tracks
        payload.update({
            "board_json": board_json,
            "timeline_data": timeline_data,
            "global_prompt": global_prompt,
            "local_prompts": local_prompts,
            "segment_lengths": segment_lengths_s,
            "dialogue_tracks": compiled_tracks,
            "foley_prompt": foley_prompt,
        })
        outputs.update({
            "board_json": board_json,
            "timeline_data": timeline_data,
            "global_prompt": global_prompt,
            "local_prompts": local_prompts,
            "segment_lengths": segment_lengths_s,
            "dialogue_script": dialogue_script,
            "foley_prompt": foley_prompt,
        })
        linx["mode"] = "iamccs_speech1_compiled"
        linx.setdefault("chain", []).append({"role": "speech1_compiler", "name": "IAMCCS_CineSpeech1PromptCompiler", "token": str(speech_token or "<speech1>")})
        report = _json_report({
            "node": "IAMCCS_CineSpeech1PromptCompiler",
            "token": str(speech_token or "<speech1>"),
            "compiled_lines": len(compiled_tracks),
            "local_prompts": len([p for p in compiled_prompts if p]),
            "segment_lengths": segment_lengths_s,
            "truth": "This is the inject node: BoardMaker writes dialogue and raw <speech1> local prompts; this node compiles them into Shotboard/TTS/Woosh cine_linx.",
        })
        resources["cine_report"] = report
        outputs["report"] = report
        _refresh_linx_index(linx)
        return linx, global_prompt, local_prompts, segment_lengths_s, dialogue_script, foley_prompt, report


class IAMCCS_CineDialogueLineRouter:
    """Split dialogue/foley board metadata into queue-ready line prompts."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "line_fallback": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("line_1_text", "line_2_text", "line_3_text", "line_4_text", "foley_prompt", "report")
    FUNCTION = "route"
    CATEGORY = "IAMCCS/Cine/Audio"

    def route(self, cine_linx, line_fallback=""):
        linx = _clone_linx(cine_linx)
        resources = _resources(linx)
        tracks = resources.get("cine_dialogue_tracks")
        if not isinstance(tracks, list):
            tracks = _safe_json_loads(_payload(linx).get("dialogue_tracks"), [])
        if not isinstance(tracks, list):
            tracks = []
        lines = [str(item.get("dialogue", "") if isinstance(item, dict) else "").strip() for item in tracks[:4]]
        fallback_lines = [part.strip() for part in str(line_fallback or "").splitlines() if part.strip()]
        while len(lines) < 4:
            lines.append(fallback_lines[len(lines)] if len(fallback_lines) > len(lines) else "")
        audio_tracks = resources.get("cine_audio_tracks") if isinstance(resources.get("cine_audio_tracks"), dict) else {}
        foley_prompt = str(audio_tracks.get("foley_prompt") or _payload(linx).get("foley_prompt") or "")
        return (
            lines[0],
            lines[1],
            lines[2],
            lines[3],
            foley_prompt,
            _json_report({
                "node": "IAMCCS_CineDialogueLineRouter",
                "line_count": len([line for line in lines if line]),
                "foley_prompt": bool(foley_prompt),
            }),
        )


class IAMCCS_CineTimelineAudioMixer:
    """Place up to four dialogue line audios on the BoardMaker timeline and mix foley."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "target_sample_rate": ("INT", {"default": 48000, "min": 8000, "max": 192000, "step": 1000}),
                "dialogue_gain_db": ("FLOAT", {"default": 0.0, "min": -60.0, "max": 24.0, "step": 0.1}),
                "foley_gain_db": ("FLOAT", {"default": -9.0, "min": -60.0, "max": 24.0, "step": 0.1}),
                "normalize_peak": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "line_1_audio": ("AUDIO",),
                "line_2_audio": ("AUDIO",),
                "line_3_audio": ("AUDIO",),
                "line_4_audio": ("AUDIO",),
                "foley_audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = ("final_audio", "dialogue_only", "report")
    FUNCTION = "mix"
    CATEGORY = "IAMCCS/Cine/Audio"

    def _tracks(self, cine_linx: Dict[str, Any]) -> List[Dict[str, Any]]:
        resources = _resources(cine_linx)
        tracks = resources.get("cine_dialogue_tracks")
        if not isinstance(tracks, list):
            tracks = _safe_json_loads(_payload(cine_linx).get("dialogue_tracks"), [])
        return [dict(item) for item in tracks if isinstance(item, dict)]

    def _place(self, canvas: torch.Tensor, audio: Any, start_s: float, gain_db: float, target_rate: int) -> int:
        if audio is None:
            return 0
        wave, sr = _audio_waveform(audio)
        wave = _resample_waveform(wave.detach().cpu(), int(sr), int(target_rate))
        wave = _ensure_channels(wave, int(canvas.shape[1]))
        wave = _gain(wave, float(gain_db))
        start = max(0, int(round(float(start_s) * int(target_rate))))
        end = min(int(canvas.shape[-1]), start + int(wave.shape[-1]))
        if end <= start:
            return 0
        canvas[:, :, start:end] += wave[:, :, : end - start]
        return int(end - start)

    def mix(
        self,
        cine_linx,
        target_sample_rate,
        dialogue_gain_db,
        foley_gain_db,
        normalize_peak,
        line_1_audio=None,
        line_2_audio=None,
        line_3_audio=None,
        line_4_audio=None,
        foley_audio=None,
    ):
        linx = _clone_linx(cine_linx)
        tracks = self._tracks(linx)
        duration = _safe_float(_resources(linx).get("cine_duration_seconds", _payload(linx).get("duration_seconds", 15.0)), 15.0)
        rate = int(target_sample_rate)
        channels = 2
        total_samples = max(1, int(math.ceil(duration * rate)))
        dialogue_canvas = torch.zeros((1, channels, total_samples), dtype=torch.float32)
        final_canvas = torch.zeros_like(dialogue_canvas)
        audios = [line_1_audio, line_2_audio, line_3_audio, line_4_audio]
        placements = []
        for index, audio in enumerate(audios):
            start_s = _safe_float(tracks[index].get("start_s", 0.0), 0.0) if index < len(tracks) else 0.0
            samples = self._place(dialogue_canvas, audio, start_s, float(dialogue_gain_db), rate)
            placements.append({"line": index + 1, "start_s": round(start_s, 3), "samples": samples})
        final_canvas += dialogue_canvas
        foley_samples = self._place(final_canvas, foley_audio, 0.0, float(foley_gain_db), rate)
        peak = float(final_canvas.abs().max().item()) if final_canvas.numel() else 0.0
        if bool(normalize_peak) and peak > 1.0:
            final_canvas = final_canvas / peak
        return (
            {"waveform": final_canvas, "sample_rate": rate},
            {"waveform": dialogue_canvas, "sample_rate": rate},
            _json_report({
                "node": "IAMCCS_CineTimelineAudioMixer",
                "duration_seconds": duration,
                "target_sample_rate": rate,
                "placements": placements,
                "foley_samples": foley_samples,
                "peak_before_normalize": peak,
                "normalized": bool(normalize_peak and peak > 1.0),
            }),
        )



from .audio.audio_board_arranger import IAMCCS_AudioBoardArranger
from .audio.audio_bus_out import IAMCCS_BusOut
class IAMCCS_CineVideoToWooshInputs:
    """Convert a generated Comfy VIDEO to IMAGE frames and WOOSH_VIDEO for foley generation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "max_duration_s": ("FLOAT", {"default": 15.0, "min": 1.0, "max": 60.0, "step": 0.5}),
                "max_frames": ("INT", {"default": 144, "min": 16, "max": 801, "step": 1}),
                "max_side": ("INT", {"default": 384, "min": 128, "max": 1280, "step": 32}),
            }
        }

    RETURN_TYPES = ("IMAGE", "WOOSH_VIDEO", "FLOAT", "STRING")
    RETURN_NAMES = ("image_batch", "woosh_video", "frame_rate", "report")
    FUNCTION = "convert"
    CATEGORY = "IAMCCS/Cine/Audio"

    @staticmethod
    def _thin_and_resize(frames: torch.Tensor, max_frames: int, max_side: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if not torch.is_tensor(frames):
            frames = torch.from_numpy(frames)
        original_shape = tuple(frames.shape)
        if frames.ndim != 4:
            raise ValueError(f"Expected video frames as [T,H,W,C], got {original_shape}")
        total = int(frames.shape[0])
        max_frames = max(1, int(max_frames))
        if total > max_frames:
            indices = torch.linspace(0, total - 1, steps=max_frames).round().long()
            frames = frames.index_select(0, indices)
        h = int(frames.shape[1])
        w = int(frames.shape[2])
        max_side = max(64, int(max_side))
        if max(h, w) > max_side:
            scale = float(max_side) / float(max(h, w))
            new_h = max(16, int(round(h * scale / 16.0)) * 16)
            new_w = max(16, int(round(w * scale / 16.0)) * 16)
            as_float = frames.float()
            if as_float.max().item() > 1.5:
                as_float = as_float / 255.0
            nchw = as_float.permute(0, 3, 1, 2).contiguous()
            resized = torch.nn.functional.interpolate(nchw, size=(new_h, new_w), mode="bilinear", align_corners=False)
            frames = (resized.permute(0, 2, 3, 1).clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        elif frames.dtype != torch.uint8:
            as_float = frames.float()
            if as_float.max().item() <= 1.5:
                as_float = as_float * 255.0
            frames = as_float.clamp(0.0, 255.0).round().to(torch.uint8)
        report = {
            "original_shape": original_shape,
            "original_frames": total,
            "output_shape": tuple(frames.shape),
            "output_frames": int(frames.shape[0]),
            "max_frames": max_frames,
            "max_side": max_side,
        }
        return frames.cpu(), report

    def convert(self, video, max_duration_s, max_frames, max_side):
        try:
            source = video.get_stream_source()
        except Exception:
            source = None
        temp_path = None
        if isinstance(source, (str, os.PathLike)) and os.path.exists(source):
            video_path = str(source)
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp.close()
            temp_path = tmp.name
            video.save_to(temp_path)
            video_path = temp_path
        custom_nodes_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        woosh_pkg = os.path.join(custom_nodes_root, "ComfyUI-Woosh", "Woosh")
        if os.path.isdir(woosh_pkg) and woosh_pkg not in sys.path:
            sys.path.insert(0, woosh_pkg)
        try:
            from woosh.utils.videoio import extract_video_frames
            frames, rate, _pts = extract_video_frames(video_path, start_time=0, end_time=float(max_duration_s))
        finally:
            if temp_path:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
        frames, prep_report = self._thin_and_resize(frames, int(max_frames), int(max_side))
        if frames.dtype != torch.float32:
            frames_float = frames.float() / 255.0
        else:
            frames_float = frames.clamp(0.0, 1.0)
        if frames_float.ndim != 4:
            raise ValueError(f"Expected video frames as [T,H,W,C], got {tuple(frames_float.shape)}")
        return (
            frames_float.cpu(),
            {"frames": frames.cpu(), "rate": float(rate)},
            float(rate),
            _json_report({
                "node": "IAMCCS_CineVideoToWooshInputs",
                "low_vram": True,
                "frames": int(frames.shape[0]),
                "frame_rate": float(rate),
                "max_duration_s": float(max_duration_s),
                **prep_report,
            }),
        )


NODE_CLASS_MAPPINGS = {
    "IAMCCS_BoardMaker_DialogueFoley": IAMCCS_BoardMaker_DialogueFoley,
    "IAMCCS_CineInfo3": IAMCCS_CineInfo3,
    "IAMCCS_CineSpeech1PromptCompiler": IAMCCS_CineSpeech1PromptCompiler,
    "IAMCCS_CineDialogueLineRouter": IAMCCS_CineDialogueLineRouter,
    "IAMCCS_CineTimelineAudioMixer": IAMCCS_CineTimelineAudioMixer,
    "IAMCCS_AudioBoardArranger": IAMCCS_AudioBoardArranger,
    "IAMCCS_BusOut": IAMCCS_BusOut,
    "IAMCCS_CineVideoToWooshInputs": IAMCCS_CineVideoToWooshInputs,
    "IAMCCS_CineSpeechLength": IAMCCS_CineSpeechLength,
    "IAMCCS_CineDialogueDurationPlanner": IAMCCS_CineDialogueDurationPlanner,
    "IAMCCS_CineAudioDurationProbe": IAMCCS_CineAudioDurationProbe,
    "IAMCCS_CineDialogueTimingReconciler": IAMCCS_CineDialogueTimingReconciler,
    "IAMCCS_CineWooshFoleyChunkPlanner": IAMCCS_CineWooshFoleyChunkPlanner,
    "IAMCCS_CineFinalAudioMixer": IAMCCS_CineFinalAudioMixer,
    "IAMCCS_CineEmotionButtons": IAMCCS_CineEmotionButtons,
    "IAMCCS_CineDialoguePromptKit": IAMCCS_CineDialoguePromptKit,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_BoardMaker_DialogueFoley": "IAMCCS BoardMaker Dialogue Foley",
    "IAMCCS_CineInfo3": "IAMCCS Cine Info 3",
    "IAMCCS_CineSpeech1PromptCompiler": "IAMCCS Speech1 Prompt Compiler",
    "IAMCCS_CineDialogueLineRouter": "IAMCCS Dialogue Line Router",
    "IAMCCS_CineTimelineAudioMixer": "IAMCCS Timeline Audio Mixer",
    "IAMCCS_AudioBoardArranger": "IAMCCS AudioBoard Arranger",
    "IAMCCS_BusOut": "IAMCCS BusOut",
    "IAMCCS_CineVideoToWooshInputs": "IAMCCS Video To Woosh Inputs",
    "IAMCCS_CineSpeechLength": "IAMCCS Speech Length Calculator",
    "IAMCCS_CineDialogueDurationPlanner": "IAMCCS Dialogue Duration Planner",
    "IAMCCS_CineAudioDurationProbe": "IAMCCS Audio Duration Probe",
    "IAMCCS_CineDialogueTimingReconciler": "IAMCCS Dialogue Timing Reconciler",
    "IAMCCS_CineWooshFoleyChunkPlanner": "IAMCCS Woosh Foley Chunk Planner",
    "IAMCCS_CineFinalAudioMixer": "IAMCCS Final Audio Mixer",
    "IAMCCS_CineEmotionButtons": "IAMCCS Emotion Buttons",
    "IAMCCS_CineDialoguePromptKit": "IAMCCS Dialogue Prompt Kit",
}
