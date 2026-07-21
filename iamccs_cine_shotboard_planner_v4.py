import copy
import json
from typing import Any, Dict, List, Tuple

from .iamccs_cine_nodes import IAMCCS_CineShotboardPlannerV3


V4_SCHEMA = "iamccs.shotboard_v4.video_timeline"


def _json_loads(value: Any, fallback: Any) -> Any:
    if isinstance(value, (dict, list)):
        return copy.deepcopy(value)
    try:
        text = str(value or "").strip()
        if not text:
            return copy.deepcopy(fallback)
        return json.loads(text)
    except Exception:
        return copy.deepcopy(fallback)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _as_bool(value: Any, fallback: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return fallback
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return fallback


def _safe_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return int(fallback)


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(fallback)


def _first_list(data: Dict[str, Any], keys: Tuple[str, ...]) -> List[Dict[str, Any]]:
    for key in keys:
        value = data.get(key)
        if isinstance(value, list):
            return [copy.deepcopy(item) for item in value if isinstance(item, dict)]
    return []


def _first_value(data: Dict[str, Any], keys: Tuple[str, ...], fallback: Any = None) -> Any:
    for key in keys:
        if key in data and data.get(key) is not None:
            return data.get(key)
    return fallback


def _choice(value: Any, allowed: Tuple[str, ...], fallback: str) -> str:
    text = str(value or "").strip()
    return text if text in allowed else fallback


def _prompt_blocks_from_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for index, seg in enumerate(segments):
        prompt = str(seg.get("prompt") or seg.get("local_prompt") or seg.get("relay_prompt") or "").strip()
        if not prompt:
            continue
        blocks.append({
            "id": str(seg.get("promptBlockId") or seg.get("prompt_block_id") or f"prompt_{index + 1:03d}"),
            "clipId": str(seg.get("id") or ""),
            "start": max(0, _safe_int(seg.get("start", 0), 0)),
            "length": max(1, _safe_int(seg.get("length", 1), 1)),
            "prompt": prompt,
            "enabled": _as_bool(seg.get("use_prompt", seg.get("prompt_enabled")), True),
        })
    return blocks


def _source_audio_segments_from_video_clips(segments: List[Dict[str, Any]], motion_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    source_segments: List[Dict[str, Any]] = []
    candidates = list(segments) + list(motion_segments)
    for index, seg in enumerate(candidates):
        clip_type = str(seg.get("type") or "").strip().lower()
        if clip_type not in {"video", "motion_video", "motion_control", "camera_reference"}:
            continue
        video_file = str(seg.get("videoFile") or seg.get("video_file") or seg.get("imageFile") or seg.get("path") or "").strip()
        if not video_file:
            continue
        source_segments.append({
            "id": str(seg.get("audioSourceId") or seg.get("audio_source_id") or f"source_audio_{index + 1:03d}"),
            "clipId": str(seg.get("id") or ""),
            "type": "source_video_audio",
            "audioFile": video_file,
            "videoFile": video_file,
            "start": max(0, _safe_int(seg.get("start", 0), 0)),
            "length": max(1, _safe_int(seg.get("length", 1), 1)),
            "trimStart": max(0, _safe_int(seg.get("trimStart", seg.get("trim_start", 0)), 0)),
            "track": max(0, _safe_int(seg.get("audioTrack", seg.get("track", 0)), 0)),
            "source": "video_clip_voice_lock",
        })
    return source_segments


def _normalize_clip(item: Dict[str, Any], index: int, default_type: str) -> Dict[str, Any]:
    clip = copy.deepcopy(item)
    clip_id = str(
        clip.get("id")
        or clip.get("shotId")
        or clip.get("shot_id")
        or clip.get("clipId")
        or clip.get("clip_id")
        or f"{default_type}_{index + 1:03d}"
    )
    clip["id"] = clip_id
    if "shotId" not in clip and default_type in {"shot", "image", "video", "text"}:
        clip["shotId"] = clip_id
    if "shot_id" not in clip and default_type in {"shot", "image", "video", "text"}:
        clip["shot_id"] = clip_id

    if "start" not in clip and "frame" in clip:
        clip["start"] = clip.get("frame")
    clip["start"] = max(0, _safe_int(clip.get("start", 0), 0))
    clip["length"] = max(1, _safe_int(clip.get("length", clip.get("len", 1)), 1))
    clip["trimStart"] = max(0, _safe_int(clip.get("trimStart", clip.get("trim_start", 0)), 0))

    image_file = str(clip.get("imageFile") or clip.get("image_file") or clip.get("imageTruthPath") or clip.get("path") or "")
    video_file = str(clip.get("videoFile") or clip.get("video_file") or clip.get("sourceVideoFile") or clip.get("movieFile") or "")
    audio_file = str(clip.get("audioFile") or clip.get("audio_file") or "")
    if image_file and not clip.get("imageFile"):
        clip["imageFile"] = image_file
    if video_file and not clip.get("videoFile"):
        clip["videoFile"] = video_file
    if audio_file and not clip.get("audioFile"):
        clip["audioFile"] = audio_file

    clip_type = str(clip.get("type") or "").strip().lower()
    if not clip_type:
        if video_file:
            clip_type = "video"
        elif audio_file:
            clip_type = "audio"
        else:
            clip_type = default_type
    if clip_type == "shot":
        clip_type = "image" if image_file and not video_file else "video" if video_file else "text"
    clip["type"] = clip_type
    media_path = str(video_file or image_file or clip.get("path") or "").strip()
    if clip_type == "video" and media_path:
        # Some timeline backends load main-lane video clips from imageFile.
        clip["imageFile"] = media_path
        clip["image_file"] = media_path
        clip["videoFile"] = media_path
        clip["video_file"] = media_path
        clip["path"] = media_path
    if clip_type in {"motion_video", "motion_control", "camera_reference"} and media_path:
        clip["videoFile"] = media_path
        clip["video_file"] = media_path
        clip["path"] = media_path
    clip["trim_start"] = clip["trimStart"]
    source_role = str(clip.get("sourceRole") or clip.get("source_role") or "").strip()
    if not source_role:
        if clip_type == "video":
            source_role = "editorial_video"
        elif clip_type == "motion_video":
            source_role = "motion_reference"
        elif clip_type == "audio":
            source_role = "timeline_audio"
        else:
            source_role = "guide_frame"
    clip["sourceRole"] = source_role
    clip["source_role"] = source_role
    clip["editMode"] = str(clip.get("editMode") or clip.get("edit_mode") or "trim_extend")
    clip["edit_mode"] = clip["editMode"]
    clip["inPoint"] = clip["trimStart"]
    clip["in_point"] = clip["trimStart"]
    clip["outPoint"] = clip["trimStart"] + clip["length"]
    clip["out_point"] = clip["outPoint"]
    clip["sourceVoiceLock"] = _as_bool(clip.get("sourceVoiceLock", clip.get("source_voice_lock")), False)
    clip["source_voice_lock"] = clip["sourceVoiceLock"]
    return clip


def _normalize_timeline_data(timeline_data: Any, global_prompt: str, duration_seconds: float, frame_rate: int) -> Dict[str, Any]:
    raw = _json_loads(timeline_data, {})
    if not isinstance(raw, dict):
        raw = {}

    fps = max(1, _safe_int(raw.get("frame_rate", raw.get("frameRate", raw.get("fps", frame_rate))), frame_rate))
    duration = max(0.01, _safe_float(raw.get("duration_seconds", raw.get("durationSeconds", raw.get("duration", duration_seconds))), duration_seconds))

    shot_clips = _first_list(raw, ("shotClips", "shots", "visualClips", "visual_clips", "clips", "segments"))
    shot_clips = [_normalize_clip(item, index, "image") for index, item in enumerate(shot_clips)]

    motion_clips = _first_list(raw, ("motionClips", "motion_clips", "motionSegments", "icVideoClips", "ic_video_clips", "icVideoSegments"))
    motion_clips = [_normalize_clip(item, index, "motion_video") for index, item in enumerate(motion_clips)]
    for clip in motion_clips:
        if clip.get("type") in {"image", "video"}:
            clip["type"] = "motion_video"
        if "attention_strength" not in clip:
            clip["attention_strength"] = _safe_float(clip.get("videoAttentionStrength", clip.get("strength", 1.0)), 1.0)
        if "resampleMode" not in clip:
            clip["resampleMode"] = str(clip.get("resample_mode", "uniform") or "uniform")

    audio_clips = _first_list(raw, ("audioClips", "audio_clips", "audioSegments"))
    audio_clips = [_normalize_clip(item, index, "audio") for index, item in enumerate(audio_clips)]

    camera_clips = _first_list(raw, ("cameraClips", "camera_clips", "cameraSegments"))
    camera_clips = [_normalize_clip(item, index, "camera_reference") for index, item in enumerate(camera_clips)]

    reference_sheets = _first_list(raw, ("referenceSheets", "reference_sheets", "ingredientsSheets", "ingredients_sheets"))
    reference_sheets = [_normalize_clip(item, index, "reference_sheet") for index, item in enumerate(reference_sheets)]

    use_custom_audio = _as_bool(raw.get("use_custom_audio", raw.get("useCustomAudio")), False) or any(
        bool(item.get("audioFile") or item.get("audioB64")) for item in audio_clips
    )
    use_custom_motion = _as_bool(raw.get("use_custom_motion", raw.get("useCustomMotion")), False) or bool(motion_clips or camera_clips)
    time_units = _choice(
        _first_value(raw, ("timeUnits", "time_units", "display_mode", "displayMode"), "seconds"),
        ("seconds", "frames"),
        "seconds",
    )
    magnet_enabled = _as_bool(
        _first_value(raw, ("magnetEnabled", "magnet_enabled", "snappingEnabled", "snapEnabled"), True),
        True,
    )
    audio_input_enabled = _as_bool(
        _first_value(raw, ("audioInputEnabled", "audio_input_enabled", "audioTrackInputEnabled", "use_custom_audio", "useCustomAudio"), use_custom_audio),
        bool(use_custom_audio),
    )
    source_voice_lock = _as_bool(
        _first_value(raw, ("sourceVoiceLock", "source_voice_lock", "voiceLock", "voice_lock", "lock_source_voice"), False),
        False,
    )
    video_to_video_enabled = _as_bool(
        _first_value(raw, ("videoToVideoEnabled", "video_to_video_enabled"), None),
        any(str(item.get("type") or "").lower() == "video" for item in shot_clips) or bool(motion_clips) or bool(raw.get("retakeVideo")),
    )
    clip_edit_mode = _choice(
        _first_value(raw, ("clipEditMode", "clip_edit_mode"), "timeline_trim_split_extend"),
        ("timeline_trim_split_extend", "trim_only", "split_at_playhead", "extend_source", "retake_range", "combine_takes"),
        "timeline_trim_split_extend",
    )
    retake_mode = _as_bool(raw.get("retakeMode", raw.get("retake_mode")), False)
    if retake_mode and clip_edit_mode == "timeline_trim_split_extend":
        clip_edit_mode = "retake_range"
    continuation_mode = _choice(
        _first_value(raw, ("continuationMode", "continuation_mode"), "source_video"),
        ("source_video", "last_frame", "guide_frame", "retake_range", "manual_cut"),
        "source_video",
    )
    if retake_mode and continuation_mode == "source_video":
        continuation_mode = "retake_range"
    guide_frame_policy = _choice(
        _first_value(raw, ("guideFramePolicy", "guide_frame_policy"), "prompt_behavior_guide_geography"),
        ("prompt_behavior_guide_geography", "first_last_keyframes", "prompt_only", "source_video_motion", "manual_keyframes"),
        "prompt_behavior_guide_geography",
    )
    explicit_prompt_blocks = _first_list(raw, ("promptBlocks", "prompt_blocks", "textSegments", "text_segments"))
    prompt_blocks = explicit_prompt_blocks if explicit_prompt_blocks else _prompt_blocks_from_segments(shot_clips)
    explicit_source_audio = _first_list(raw, ("sourceAudioSegments", "source_audio_segments", "voiceLockSegments", "voice_lock_segments"))
    source_audio_segments = explicit_source_audio
    if not source_audio_segments and audio_input_enabled and source_voice_lock:
        source_audio_segments = _source_audio_segments_from_video_clips(shot_clips, motion_clips)
    if audio_input_enabled and source_voice_lock and not audio_clips and source_audio_segments:
        audio_clips = [_normalize_clip(item, index, "audio") for index, item in enumerate(source_audio_segments)]
        use_custom_audio = True

    timeline = copy.deepcopy(raw)
    timeline.update({
        "schema": V4_SCHEMA,
        "schema_version": 1,
        "ui_base": "IAMCCS_CineShotboardPlannerV3",
        "frame_rate": int(fps),
        "duration_seconds": float(duration),
        "duration_frames": max(1, int(round(float(duration) * int(fps)))),
        "global_prompt": str(raw.get("global_prompt", raw.get("globalPrompt", global_prompt)) or global_prompt or ""),
        "segments": shot_clips,
        "shots": shot_clips,
        "shotClips": shot_clips,
        "motionSegments": motion_clips,
        "motionClips": motion_clips,
        "audioSegments": audio_clips,
        "audioClips": audio_clips,
        "cameraSegments": camera_clips,
        "cameraClips": camera_clips,
        "referenceSheets": reference_sheets,
        "promptBlocks": prompt_blocks,
        "prompt_blocks": prompt_blocks,
        "sourceAudioSegments": source_audio_segments,
        "source_audio_segments": source_audio_segments,
        "retakeMode": bool(retake_mode),
        "retakeVideo": raw.get("retakeVideo", raw.get("retake_video")),
        "retakeStart": max(0, _safe_int(raw.get("retakeStart", raw.get("retake_start", 0)), 0)),
        "retakeLength": max(0, _safe_int(raw.get("retakeLength", raw.get("retake_length", 0)), 0)),
        "retakeStrength": _safe_float(raw.get("retakeStrength", raw.get("retake_strength", 1.0)), 1.0),
        "retakePrompt": str(raw.get("retakePrompt", raw.get("retake_prompt", "")) or ""),
        "retake_global_prompt": str(raw.get("retake_global_prompt", raw.get("retakeGlobalPrompt", "")) or ""),
        "mainTrackEnabled": _as_bool(raw.get("mainTrackEnabled", raw.get("main_track_enabled")), True),
        "audioTrackEnabled": _as_bool(raw.get("audioTrackEnabled", raw.get("audio_track_enabled")), bool(audio_clips) or use_custom_audio),
        "motionTrackEnabled": _as_bool(raw.get("motionTrackEnabled", raw.get("motion_track_enabled")), bool(motion_clips)),
        "videoToVideoEnabled": bool(video_to_video_enabled),
        "video_to_video_enabled": bool(video_to_video_enabled),
        "audioInputEnabled": bool(audio_input_enabled),
        "audio_input_enabled": bool(audio_input_enabled),
        "sourceVoiceLock": bool(source_voice_lock),
        "source_voice_lock": bool(source_voice_lock),
        "magnetEnabled": bool(magnet_enabled),
        "magnet_enabled": bool(magnet_enabled),
        "timeUnits": time_units,
        "time_units": time_units,
        "display_mode": time_units,
        "displayMode": time_units,
        "clipEditMode": clip_edit_mode,
        "clip_edit_mode": clip_edit_mode,
        "continuationMode": continuation_mode,
        "continuation_mode": continuation_mode,
        "guideFramePolicy": guide_frame_policy,
        "guide_frame_policy": guide_frame_policy,
        "propHeight": max(1, _safe_int(raw.get("propHeight", raw.get("prop_height", 90)), 90)),
        "globalPropHeight": max(1, _safe_int(raw.get("globalPropHeight", raw.get("global_prop_height", 60)), 60)),
        "showFilenames": _as_bool(raw.get("showFilenames", raw.get("show_filenames")), True),
        "normalStartFrame": max(0, _safe_int(raw.get("normalStartFrame", raw.get("normal_start_frame", 0)), 0)),
        "normalDurationFrames": max(1, _safe_int(raw.get("normalDurationFrames", raw.get("normal_duration_frames", max(1, int(round(float(duration) * int(fps)))))), max(1, int(round(float(duration) * int(fps)))))),
        "inpaint_audio": _as_bool(raw.get("inpaint_audio", raw.get("inpaintAudio")), False),
        "inpaintAudio": _as_bool(raw.get("inpaintAudio", raw.get("inpaint_audio")), False),
        "override_audio": _as_bool(raw.get("override_audio", raw.get("overrideAudio")), False),
        "overrideAudio": _as_bool(raw.get("overrideAudio", raw.get("override_audio")), False),
        "use_custom_audio": bool(use_custom_audio),
        "use_custom_motion": bool(use_custom_motion),
        "timelineControlContract": {
            "video_lane": "main",
            "motion_lane": "ic_lora_reference",
            "audio_lane": "audio",
            "supports_video_scrub": True,
            "supports_video_trim": True,
            "supports_split_at_playhead": True,
            "supports_source_voice_lock": True,
            "supports_audio_inpaint": True,
            "supports_retake_range": True,
            "supports_in_out_points": True,
            "time_units": time_units,
            "magnet_enabled": bool(magnet_enabled),
            "clip_edit_mode": clip_edit_mode,
            "retake_mode": bool(retake_mode),
            "retake_has_video": isinstance(raw.get("retakeVideo", raw.get("retake_video")), dict),
            "guide_frame_policy": guide_frame_policy,
        },
        "authoring_aliases": {
            "segments": "Executable backend visual clips",
            "shots": "IAMCCS authoring alias for visual clips",
            "shotClips": "IAMCCS video-editing alias for visual clips",
            "motionClips": "IAMCCS authoring alias for motionSegments",
            "audioClips": "IAMCCS authoring alias for audioSegments",
            "cameraClips": "IAMCCS authoring alias for cameraSegments",
            "promptBlocks": "IAMCCS editorial prompt blocks derived from visual clips",
            "sourceAudioSegments": "IAMCCS video-audio voice-lock source segments",
        },
    })
    return timeline


def _backend_timeline_view(timeline: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema": "iamccs.shotboard_v4.video_timeline_backend",
        "schema_version": 1,
        "mainTrackEnabled": bool(timeline.get("mainTrackEnabled", True)),
        "audioTrackEnabled": bool(timeline.get("audioTrackEnabled", bool(timeline.get("audioSegments", [])))),
        "motionTrackEnabled": bool(timeline.get("motionTrackEnabled", bool(timeline.get("motionSegments", [])))),
        "videoToVideoEnabled": bool(timeline.get("videoToVideoEnabled", timeline.get("video_to_video_enabled", False))),
        "audioInputEnabled": bool(timeline.get("audioInputEnabled", timeline.get("audio_input_enabled", False))),
        "sourceVoiceLock": bool(timeline.get("sourceVoiceLock", timeline.get("source_voice_lock", False))),
        "magnetEnabled": bool(timeline.get("magnetEnabled", timeline.get("magnet_enabled", True))),
        "timeUnits": str(timeline.get("timeUnits", timeline.get("display_mode", "seconds")) or "seconds"),
        "display_mode": str(timeline.get("display_mode", timeline.get("timeUnits", "seconds")) or "seconds"),
        "clipEditMode": str(timeline.get("clipEditMode", timeline.get("clip_edit_mode", "timeline_trim_split_extend")) or "timeline_trim_split_extend"),
        "continuationMode": str(timeline.get("continuationMode", timeline.get("continuation_mode", "source_video")) or "source_video"),
        "guideFramePolicy": str(timeline.get("guideFramePolicy", timeline.get("guide_frame_policy", "prompt_behavior_guide_geography")) or "prompt_behavior_guide_geography"),
        "propHeight": int(timeline.get("propHeight", 90) or 90),
        "globalPropHeight": int(timeline.get("globalPropHeight", 60) or 60),
        "showFilenames": bool(timeline.get("showFilenames", True)),
        "frame_rate": int(timeline.get("frame_rate", 24) or 24),
        "duration_seconds": float(timeline.get("duration_seconds", 0.0) or 0.0),
        "duration_frames": int(timeline.get("duration_frames", 0) or 0),
        "global_prompt": str(timeline.get("global_prompt", "") or ""),
        "negative_prompt": str(timeline.get("negative_prompt", "") or ""),
        "retake_global_prompt": str(timeline.get("retake_global_prompt", "") or ""),
        "normalStartFrame": int(timeline.get("normalStartFrame", 0) or 0),
        "normalDurationFrames": int(timeline.get("normalDurationFrames", timeline.get("duration_frames", 0)) or 0),
        "segments": copy.deepcopy(timeline.get("segments", [])),
        "motionSegments": copy.deepcopy(timeline.get("motionSegments", [])),
        "audioSegments": copy.deepcopy(timeline.get("audioSegments", [])),
        "promptBlocks": copy.deepcopy(timeline.get("promptBlocks", [])),
        "sourceAudioSegments": copy.deepcopy(timeline.get("sourceAudioSegments", [])),
        "cameraSegments": copy.deepcopy(timeline.get("cameraSegments", [])),
        "referenceSheets": copy.deepcopy(timeline.get("referenceSheets", [])),
        "retakeMode": bool(timeline.get("retakeMode", False)),
        "retakeVideo": timeline.get("retakeVideo"),
        "retakeStart": int(timeline.get("retakeStart", 0) or 0),
        "retakeLength": int(timeline.get("retakeLength", 0) or 0),
        "retakeStrength": float(timeline.get("retakeStrength", 1.0) or 1.0),
        "retakePrompt": str(timeline.get("retakePrompt", "") or ""),
        "inpaint_audio": bool(timeline.get("inpaint_audio", False)),
        "inpaintAudio": bool(timeline.get("inpaintAudio", timeline.get("inpaint_audio", False))),
        "override_audio": bool(timeline.get("override_audio", False)),
        "overrideAudio": bool(timeline.get("overrideAudio", timeline.get("override_audio", False))),
        "use_custom_audio": bool(timeline.get("use_custom_audio", False)),
        "use_custom_motion": bool(timeline.get("use_custom_motion", False)),
        "timelineControlContract": copy.deepcopy(timeline.get("timelineControlContract", {})),
    }


class IAMCCS_CineShotboardPlannerV4(IAMCCS_CineShotboardPlannerV3):
    """Shotboard V4 planner shell.

    V4 intentionally starts from the stable V3 timeline/meter UI and backend
    contract, then emits a neutral multi-track contract for future video,
    IC-LoRA, retake, camera and audio-inpaint backends.
    """

    CATEGORY = "IAMCCS/Cine/02 Single Generation VIP"

    @classmethod
    def INPUT_TYPES(cls):
        data = super().INPUT_TYPES()
        data["required"]["global_prompt"] = ("STRING", {
            "default": "one continuous cinematic shot with coherent motion, stable identity, video-aware timing and cinematic-grade camera continuity",
            "multiline": True,
        })
        data["required"]["timeline_data"] = ("STRING", {
            "default": "",
            "multiline": True,
            "tooltip": "Edited by Shotboard Planner V4. Accepts IAMCCS shots/shotClips/motionClips/audioClips aliases and exports backend-compatible segments/motionSegments/audioSegments.",
        })
        data["required"]["duration_seconds"] = ("FLOAT", {"default": 20.0, "min": 0.01, "max": 36000.0, "step": 0.01})
        data["required"]["guide_policy"] = (["every_checked_row", "safe_core_guides", "prompt_only"], {"default": "every_checked_row"})
        return data

    def execute(self, global_prompt, timeline_data, duration_seconds, frame_rate, guide_policy, min_guide_gap_seconds, max_guides, default_force, promptrelay_epsilon, ltx_round_mode, image_paths, image_width, image_height, image_resize_method="crop", image_multiple_of=32, img_compression=0, cine_linx=None):
        normalized = _normalize_timeline_data(timeline_data, global_prompt, float(duration_seconds), int(frame_rate))
        normalized_text = _json_dumps(normalized)

        (out_linx,) = super().execute(
            global_prompt,
            normalized_text,
            duration_seconds,
            frame_rate,
            guide_policy,
            min_guide_gap_seconds,
            max_guides,
            default_force,
            promptrelay_epsilon,
            ltx_round_mode,
            image_paths,
            image_width,
            image_height,
            image_resize_method,
            image_multiple_of,
            img_compression,
            cine_linx=cine_linx,
        )

        if not isinstance(out_linx, dict):
            return (out_linx,)

        resources = out_linx.setdefault("resources", {})
        outputs = out_linx.setdefault("outputs", {})
        payload = resources.get("cine_payload") if isinstance(resources.get("cine_payload"), dict) else {}

        visual_from_v3 = _json_loads(resources.get("cine_visual_segments_json", "[]"), [])
        audio_from_v3 = _json_loads(resources.get("cine_audio_timeline_json", outputs.get("audio_timeline_json", "{}")), {})
        if isinstance(visual_from_v3, list) and visual_from_v3:
            normalized["segments"] = visual_from_v3
            normalized["shots"] = visual_from_v3
            normalized["shotClips"] = visual_from_v3
        if isinstance(audio_from_v3, dict) and isinstance(audio_from_v3.get("audioSegments"), list):
            normalized["audioSegments"] = audio_from_v3.get("audioSegments", [])
            normalized["audioClips"] = audio_from_v3.get("audioSegments", [])
            normalized["use_custom_audio"] = bool(normalized["audioSegments"])
        elif isinstance(audio_from_v3, list):
            normalized["audioSegments"] = audio_from_v3
            normalized["audioClips"] = audio_from_v3
            normalized["use_custom_audio"] = bool(audio_from_v3)

        backend_timeline = _backend_timeline_view(normalized)
        v4_json = _json_dumps(normalized)
        backend_timeline_json = _json_dumps(backend_timeline)

        payload.update({
            "backend_mode": "cine_shotboard_v4_video_timeline_adapter",
            "shotboard_version": 4,
            "shotboard_v4_schema": V4_SCHEMA,
            "timeline_data": v4_json,
            "v4_timeline_data": v4_json,
            "backend_timeline_data": backend_timeline_json,
            "segments": normalized.get("segments", []),
            "shots": normalized.get("shots", []),
            "shotClips": normalized.get("shotClips", []),
            "motionSegments": normalized.get("motionSegments", []),
            "motionClips": normalized.get("motionClips", []),
            "promptBlocks": normalized.get("promptBlocks", []),
            "sourceAudioSegments": normalized.get("sourceAudioSegments", []),
            "cameraSegments": normalized.get("cameraSegments", []),
            "referenceSheets": normalized.get("referenceSheets", []),
            "use_custom_motion": bool(normalized.get("use_custom_motion", False)),
            "inpaint_audio": bool(normalized.get("inpaint_audio", False)),
            "override_audio": bool(normalized.get("override_audio", False)),
            "video_to_video_enabled": bool(normalized.get("video_to_video_enabled", False)),
            "audio_input_enabled": bool(normalized.get("audio_input_enabled", False)),
            "source_voice_lock": bool(normalized.get("source_voice_lock", False)),
            "magnet_enabled": bool(normalized.get("magnet_enabled", True)),
            "time_units": str(normalized.get("time_units", "seconds") or "seconds"),
            "clip_edit_mode": str(normalized.get("clip_edit_mode", "timeline_trim_split_extend") or "timeline_trim_split_extend"),
            "continuation_mode": str(normalized.get("continuation_mode", "source_video") or "source_video"),
            "guide_frame_policy": str(normalized.get("guide_frame_policy", "prompt_behavior_guide_geography") or "prompt_behavior_guide_geography"),
            "timeline_control_contract": normalized.get("timelineControlContract", {}),
        })
        resources["cine_payload"] = payload
        resources["cine_v4_timeline_data_json"] = v4_json
        resources["cine_backend_timeline_data_json"] = backend_timeline_json
        resources["cine_shots_json"] = _json_dumps(normalized.get("shots", []))
        resources["cine_motion_segments_json"] = _json_dumps(normalized.get("motionSegments", []))
        resources["cine_prompt_blocks_json"] = _json_dumps(normalized.get("promptBlocks", []))
        resources["cine_source_audio_segments_json"] = _json_dumps(normalized.get("sourceAudioSegments", []))
        resources["cine_camera_segments_json"] = _json_dumps(normalized.get("cameraSegments", []))
        resources["cine_reference_sheets_json"] = _json_dumps(normalized.get("referenceSheets", []))
        resources["cine_use_custom_motion"] = bool(normalized.get("use_custom_motion", False))
        resources["cine_inpaint_audio"] = bool(normalized.get("inpaint_audio", False))
        resources["cine_override_audio"] = bool(normalized.get("override_audio", False))
        resources["cine_timeline_control_contract_json"] = _json_dumps(normalized.get("timelineControlContract", {}))

        outputs["timeline_data"] = v4_json
        outputs["backend_timeline_data"] = backend_timeline_json
        outputs["motion_segments_json"] = resources["cine_motion_segments_json"]
        outputs["prompt_blocks_json"] = resources["cine_prompt_blocks_json"]
        outputs["source_audio_segments_json"] = resources["cine_source_audio_segments_json"]
        outputs["camera_segments_json"] = resources["cine_camera_segments_json"]
        outputs["reference_sheets_json"] = resources["cine_reference_sheets_json"]

        out_linx["mode"] = "cine_shotboard_v4_video_timeline_adapter"
        out_linx["pipeline_kind"] = "shotboard_v4_video_timeline"
        if out_linx.get("chain") and isinstance(out_linx["chain"][0], dict):
            out_linx["chain"][0]["name"] = "Cine Shotboard Planner V4"
        if out_linx.get("stages") and isinstance(out_linx["stages"][0], dict):
            out_linx["stages"][0]["kind"] = "cine_shotboard_v4_video_timeline_adapter"
            out_linx["stages"][0]["payload"] = payload
        policies = out_linx.setdefault("policies", {})
        policies["shotboard_v4_base"] = "shotboard_v3_timeline_meter_ui"
        policies["video_timeline_contract"] = "timeline_data: segments/motionSegments/audioSegments"
        policies["main_video_compatibility"] = "type=video clips export imageFile and videoFile for backend compatibility"
        policies["shotboard_v4_editorial_contract"] = "main video lane + IC/motion lane + audio lane + prompt blocks + source video audio lock"
        policies["reverse_engineering_rule"] = "feature_parity_must_be_based_on_working_nodes_workflows_or_verified_docs"

        report = {
            "node": "IAMCCS_CineShotboardPlannerV4",
            "base": "IAMCCS_CineShotboardPlannerV3",
            "schema": V4_SCHEMA,
            "ui_base": "V3 timeline meter",
            "segments": len(normalized.get("segments", [])),
            "motionSegments": len(normalized.get("motionSegments", [])),
            "audioSegments": len(normalized.get("audioSegments", [])),
            "promptBlocks": len(normalized.get("promptBlocks", [])),
            "sourceAudioSegments": len(normalized.get("sourceAudioSegments", [])),
            "cameraSegments": len(normalized.get("cameraSegments", [])),
            "referenceSheets": len(normalized.get("referenceSheets", [])),
            "use_custom_audio": bool(normalized.get("use_custom_audio", False)),
            "use_custom_motion": bool(normalized.get("use_custom_motion", False)),
            "inpaint_audio": bool(normalized.get("inpaint_audio", False)),
            "override_audio": bool(normalized.get("override_audio", False)),
            "video_to_video_enabled": bool(normalized.get("video_to_video_enabled", False)),
            "audio_input_enabled": bool(normalized.get("audio_input_enabled", False)),
            "source_voice_lock": bool(normalized.get("source_voice_lock", False)),
            "magnet_enabled": bool(normalized.get("magnet_enabled", True)),
            "time_units": str(normalized.get("time_units", "seconds") or "seconds"),
            "clip_edit_mode": str(normalized.get("clip_edit_mode", "timeline_trim_split_extend") or "timeline_trim_split_extend"),
            "continuation_mode": str(normalized.get("continuation_mode", "source_video") or "source_video"),
            "guide_frame_policy": str(normalized.get("guide_frame_policy", "prompt_behavior_guide_geography") or "prompt_behavior_guide_geography"),
            "truth": "V4 is a new-file planner shell. It preserves V3 UI/backend compatibility while emitting a neutral video timeline contract.",
        }
        resources["cine_v4_report_json"] = _json_dumps(report)
        outputs["v4_report"] = resources["cine_v4_report_json"]

        out_linx["resource_keys"] = sorted(resources.keys())
        out_linx["resource_types"] = {key: type(value).__name__ for key, value in resources.items()}
        return (out_linx,)
