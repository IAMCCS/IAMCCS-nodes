import copy
import json
import math
from typing import Any, Dict, List, Tuple

from .engine_v2v.cine_v2v_node import (
    IAMCCS_CineInfoV2VBackendRouter,
    SUPERNODE_LINX_TYPE,
    V2V_BACKEND_TEMPLATES,
)
from .iamccs_cine_shotboard_planner_v4 import IAMCCS_CineShotboardPlannerV4


V5_SCHEMA = "iamccs.shotboard_v5.v2v_timeline"


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


def _safe_int(value: Any, fallback: int = 0) -> int:
    try:
        number = float(value)
    except Exception:
        return int(fallback)
    if not math.isfinite(number):
        return int(fallback)
    return int(round(number))


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return float(fallback)
    if not math.isfinite(number):
        return float(fallback)
    return float(number)


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


def _first_value(data: Dict[str, Any], keys: Tuple[str, ...], fallback: Any = None) -> Any:
    for key in keys:
        if key in data and data.get(key) is not None:
            return data.get(key)
    return fallback


def _choice(value: Any, allowed: Tuple[str, ...], fallback: str) -> str:
    text = str(value or "").strip()
    return text if text in allowed else fallback


def _first_list(data: Dict[str, Any], keys: Tuple[str, ...]) -> List[Dict[str, Any]]:
    for key in keys:
        value = data.get(key)
        if isinstance(value, list):
            return [copy.deepcopy(item) for item in value if isinstance(item, dict)]
    return []


def _media_path(item: Dict[str, Any], prefer_video: bool = False) -> str:
    if not isinstance(item, dict):
        return ""
    video_keys = ("videoFile", "video_file", "sourceVideoFile", "movieFile", "path")
    image_keys = ("imageFile", "image_file", "imageTruthPath", "referenceImage", "path")
    keys = video_keys + image_keys if prefer_video else image_keys + video_keys
    for key in keys:
        value = str(item.get(key) or "").strip()
        if value:
            return value
    return ""


def _prompt_blocks_from_timeline(timeline: Dict[str, Any]) -> List[Dict[str, Any]]:
    explicit = _first_list(timeline, ("promptBlocks", "prompt_blocks", "textSegments", "text_segments"))
    if explicit:
        return explicit
    blocks: List[Dict[str, Any]] = []
    for index, seg in enumerate(_first_list(timeline, ("shotClips", "shots", "visualClips", "clips", "segments"))):
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


def _source_audio_segments_from_timeline(timeline: Dict[str, Any]) -> List[Dict[str, Any]]:
    explicit = _first_list(timeline, ("sourceAudioSegments", "source_audio_segments", "voiceLockSegments", "voice_lock_segments"))
    if explicit:
        return explicit
    source_segments: List[Dict[str, Any]] = []
    candidates = []
    for key in ("shotClips", "segments", "shots", "motionSegments", "motionClips"):
        candidates.extend(_first_list(timeline, (key,)))
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


def _first_media_from_timeline(timeline: Dict[str, Any], prefer_video: bool = False) -> str:
    keys = (
        ("shotClips", "segments", "shots", "visualClips", "motionSegments", "motionClips", "controlTracks")
        if prefer_video
        else ("shotClips", "segments", "shots", "visualClips", "identityTracks", "referenceSheets")
    )
    for key in keys:
        for item in timeline.get(key, []) if isinstance(timeline.get(key), list) else []:
            path = _media_path(item, prefer_video=prefer_video)
            if path:
                return path
    retake = timeline.get("retakeVideo") if isinstance(timeline.get("retakeVideo"), dict) else {}
    if prefer_video:
        path = _media_path(retake, prefer_video=True)
        if path:
            return path
    return ""


def _template_for_profile(profile: str) -> Dict[str, Any]:
    if profile in V2V_BACKEND_TEMPLATES:
        return copy.deepcopy(V2V_BACKEND_TEMPLATES[profile])
    if profile == "ltx23_v2v_loop_antidegrade":
        template = copy.deepcopy(V2V_BACKEND_TEMPLATES.get("ltx23_v2v_infinite_lipsync", {}))
        template.update({
            "label": "LTX Loop Antidegrade",
            "family": "ltx",
            "mode": "ltx_simple",
            "notes": "V5 loop/continuation profile. Uses the LTX V2V bus contract with overlap and antidegrade mode metadata.",
        })
        return template
    return copy.deepcopy(V2V_BACKEND_TEMPLATES.get("ltx23_v2v_infinite_lipsync", {}))


def _mode_contract(v2v_mode: str, scail_identity_mode: str, wan_background_lock: bool) -> Dict[str, str]:
    mode = str(v2v_mode or "ltx_v2v")
    if mode == "ltx_loop":
        return {
            "family": "ltx",
            "backend_mode": "ltx_simple",
            "profile": "ltx23_v2v_loop_antidegrade",
            "variant": "ltx_simple",
            "ui_mode": "ltx_loop",
        }
    if mode == "scail_single":
        return {
            "family": "scail2",
            "backend_mode": "scail2",
            "profile": "scail2_single_person",
            "variant": "scail2_single_person",
            "ui_mode": "scail_single",
        }
    if mode == "scail_multi":
        return {
            "family": "scail2",
            "backend_mode": "scail2",
            "profile": "scail2_multi_person_identity",
            "variant": "scail2_multi_person_identity",
            "ui_mode": "scail_multi",
        }
    if mode == "wananimate_bg_locked":
        return {
            "family": "wananimate",
            "backend_mode": "wananimate",
            "profile": "wananimate_extension",
            "variant": "wananimate_bg_locked" if wan_background_lock else "wananimate_extension",
            "ui_mode": "wananimate_bg_locked",
        }
    if mode == "pose_transfer":
        return {
            "family": "pose_transfer",
            "backend_mode": "pose_transfer",
            "profile": "flux_klein_pose_transfer",
            "variant": "flux_klein_pose_transfer",
            "ui_mode": "pose_transfer",
        }
    if str(scail_identity_mode or "") == "multi_person_identity":
        return _mode_contract("scail_multi", scail_identity_mode, wan_background_lock)
    return {
        "family": "ltx",
        "backend_mode": "ltx_simple",
        "profile": "ltx23_v2v_infinite_lipsync",
        "variant": "ltx_simple",
        "ui_mode": "ltx_v2v",
    }


def _normalize_v5_layers(timeline: Dict[str, Any], contract: Dict[str, str], output_stage: str, preview_stage: str) -> Dict[str, Any]:
    shot_clips = _first_list(timeline, ("shotClips", "shots", "visualClips", "clips", "segments"))
    mode_layers = _first_list(timeline, ("modeLayers", "mode_layers", "v2vModeLayers", "v2v_mode_layers"))
    identity_tracks = _first_list(timeline, ("identityTracks", "identity_tracks", "characters", "characterTracks"))
    control_tracks = _first_list(timeline, ("controlTracks", "control_tracks", "poseTracks", "maskTracks", "backgroundTracks"))
    takes = _first_list(timeline, ("takes", "videoTakes", "renderTakes"))
    if not mode_layers and shot_clips:
        for clip in shot_clips:
            mode_layers.append({
                "id": f"mode_{clip.get('id', clip.get('shotId', len(mode_layers) + 1))}",
                "clipId": clip.get("id") or clip.get("shotId") or "",
                "mode": contract["ui_mode"],
                "backend_family": contract["family"],
                "backend_profile": contract["profile"],
                "backend_variant": contract["variant"],
                "start": _safe_int(clip.get("start", 0), 0),
                "length": max(1, _safe_int(clip.get("length", 1), 1)),
                "output_stage": output_stage,
                "preview_stage": preview_stage,
                "enabled": True,
            })
    return {
        "shotClips": shot_clips,
        "modeLayers": mode_layers,
        "identityTracks": identity_tracks,
        "controlTracks": control_tracks,
        "takes": takes,
    }


class IAMCCS_CineShotboardPlannerV5V2V(IAMCCS_CineShotboardPlannerV4):
    """Shotboard V5 V2V adapter.

    V5 keeps the V4 timeline/editor contract, then adds a V2V mode layer and
    emits the same resources consumed by the existing IAMCCS V2V routers.
    """

    CATEGORY = "IAMCCS/Cine/Shotboard V5"

    @classmethod
    def INPUT_TYPES(cls):
        data = super().INPUT_TYPES()
        data["required"]["global_prompt"] = ("STRING", {
            "default": "cinematic video-to-video production timeline with editable clips, coherent continuation, stable identity, clear geography and audio continuity",
            "multiline": True,
        })
        data["required"]["timeline_data"] = ("STRING", {
            "default": "",
            "multiline": True,
            "tooltip": "Edited by Shotboard Planner V5. Emits V4 timeline data plus V2V modeLayers/controlTracks/takes and a router-compatible V2V payload.",
        })
        optional = data.setdefault("optional", {})
        optional.update({
            "source_video_path": ("STRING", {"default": ""}),
            "source_image_path": ("STRING", {"default": ""}),
            "trim_start_s": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.01}),
            "trim_end_s": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.01}),
            "frame_load_cap": ("INT", {"default": 241, "min": 1, "max": 100000, "step": 1}),
            "generation_width": ("INT", {"default": 1280, "min": 64, "max": 8192, "step": 8}),
            "generation_height": ("INT", {"default": 720, "min": 64, "max": 8192, "step": 8}),
            "v2v_mode": (["ltx_v2v", "ltx_loop", "scail_single", "scail_multi", "wananimate_bg_locked", "pose_transfer"], {"default": "ltx_v2v"}),
            "vram_profile": (["normal_vram", "low_vram"], {"default": "normal_vram"}),
            "segment_seconds": ("FLOAT", {"default": 10.0, "min": 0.01, "max": 3600.0, "step": 0.01}),
            "planning_mode": (["manual_segment_seconds", "explicit_preset_seconds"], {"default": "explicit_preset_seconds"}),
            "segment_preset": (["5sec", "10sec", "15sec", "20sec", "videoclip", "monologue"], {"default": "5sec"}),
            "overlap_frames": ("INT", {"default": 9, "min": 0, "max": 4096, "step": 1}),
            "time_units": (["seconds", "frames"], {"default": "seconds"}),
            "magnet_enabled": ("BOOLEAN", {"default": True}),
            "audio_input_enabled": ("BOOLEAN", {"default": True}),
            "source_voice_lock": ("BOOLEAN", {"default": True}),
            "video_to_video_enabled": ("BOOLEAN", {"default": True}),
            "inpaint_audio": ("BOOLEAN", {"default": True}),
            "clip_edit_mode": (["timeline_trim_split_extend", "trim_only", "split_at_playhead", "extend_source", "retake_range", "combine_takes"], {"default": "timeline_trim_split_extend"}),
            "continuation_mode": (["source_video", "last_frame", "guide_frame", "retake_range", "manual_cut"], {"default": "source_video"}),
            "guide_frame_policy": (["prompt_behavior_guide_geography", "first_last_keyframes", "prompt_only", "source_video_motion", "manual_keyframes"], {"default": "prompt_behavior_guide_geography"}),
            "output_stage": (["final", "draft", "sam_preview", "mask_preview", "pose_preview", "reference_preview", "result_image"], {"default": "final"}),
            "preview_stage": (["final", "draft", "sam_preview", "mask_preview", "pose_preview", "reference_preview", "result_image", "source"], {"default": "final"}),
            "render_scope": (["selected_clip", "selected_range", "all_clips", "dirty_clips"], {"default": "selected_clip"}),
            "audio_vae_name": ("STRING", {"default": "ltx-2.3-22b-dev_audio_vae.safetensors"}),
            "audio_vae_device": (["main_device", "cpu"], {"default": "main_device"}),
            "audio_vae_dtype": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
            "pose_mode": (["none", "dwpose_openpose", "source_pose_only", "image_pose_transfer"], {"default": "dwpose_openpose"}),
            "dwpose_enabled": ("BOOLEAN", {"default": True}),
            "dwpose_strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.5, "step": 0.01}),
            "scail_identity_mode": (["single_person", "multi_person_identity"], {"default": "single_person"}),
            "scail_output_stage": (["final_32fps_upscaled", "generated_16fps", "both"], {"default": "final_32fps_upscaled"}),
            "enable_sam31_preview": ("BOOLEAN", {"default": True}),
            "wan_background_lock": ("BOOLEAN", {"default": True}),
            "wan_character_mask_mode": (["sam31_identity_mask", "uploaded_character_mask", "none"], {"default": "sam31_identity_mask"}),
            "wan_control_preview": ("BOOLEAN", {"default": True}),
            "pose_transfer_image_path": ("STRING", {"default": ""}),
            "pose_transfer_video_path": ("STRING", {"default": ""}),
            "pose_transfer_result_path": ("STRING", {"default": ""}),
            "pose_transfer_result_mode": (["preview_only", "use_as_reference", "export_result"], {"default": "use_as_reference"}),
            "output_prefix": ("STRING", {"default": "IAMCCS/SHOTBOARD_V5_V2V"}),
            "negative_prompt": ("STRING", {"default": "cartoon, ugly, unstable anatomy, flicker, broken motion, identity drift, subtitles, text", "multiline": True}),
        })
        return data

    def execute(
        self,
        global_prompt,
        timeline_data,
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
        image_resize_method="crop",
        image_multiple_of=32,
        img_compression=0,
        cine_linx=None,
        source_video_path="",
        source_image_path="",
        trim_start_s=0.0,
        trim_end_s=0.0,
        frame_load_cap=241,
        generation_width=1280,
        generation_height=720,
        v2v_mode="ltx_v2v",
        vram_profile="normal_vram",
        segment_seconds=10.0,
        planning_mode="explicit_preset_seconds",
        segment_preset="5sec",
        overlap_frames=9,
        time_units="seconds",
        magnet_enabled=True,
        audio_input_enabled=True,
        source_voice_lock=True,
        video_to_video_enabled=True,
        inpaint_audio=True,
        clip_edit_mode="timeline_trim_split_extend",
        continuation_mode="source_video",
        guide_frame_policy="prompt_behavior_guide_geography",
        output_stage="final",
        preview_stage="final",
        render_scope="selected_clip",
        audio_vae_name="ltx-2.3-22b-dev_audio_vae.safetensors",
        audio_vae_device="main_device",
        audio_vae_dtype="bf16",
        pose_mode="dwpose_openpose",
        dwpose_enabled=True,
        dwpose_strength=0.75,
        scail_identity_mode="single_person",
        scail_output_stage="final_32fps_upscaled",
        enable_sam31_preview=True,
        wan_background_lock=True,
        wan_character_mask_mode="sam31_identity_mask",
        wan_control_preview=True,
        pose_transfer_image_path="",
        pose_transfer_video_path="",
        pose_transfer_result_path="",
        pose_transfer_result_mode="use_as_reference",
        output_prefix="IAMCCS/SHOTBOARD_V5_V2V",
        negative_prompt="cartoon, ugly, unstable anatomy, flicker, broken motion, identity drift, subtitles, text",
    ):
        (out_linx,) = super().execute(
            global_prompt,
            timeline_data,
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
        base_timeline = _json_loads(outputs.get("timeline_data") or resources.get("cine_v4_timeline_data_json") or timeline_data, {})
        if not isinstance(base_timeline, dict):
            base_timeline = {}

        fps_value = max(1.0, _safe_float(base_timeline.get("frame_rate", frame_rate), _safe_float(frame_rate, 24.0)))
        duration = max(0.01, _safe_float(base_timeline.get("duration_seconds", duration_seconds), _safe_float(duration_seconds, 10.0)))
        trim_start = max(0.0, _safe_float(base_timeline.get("trim_start_s", trim_start_s), _safe_float(trim_start_s, 0.0)))
        raw_trim_end = _safe_float(base_timeline.get("trim_end_s", trim_end_s), _safe_float(trim_end_s, 0.0))
        trim_end = raw_trim_end if raw_trim_end > trim_start else duration
        trim_end = min(max(trim_end, trim_start + 0.01), duration)
        effective_duration = max(0.01, trim_end - trim_start)
        width = max(64, _safe_int(base_timeline.get("generation_width", generation_width), _safe_int(generation_width, 1280)))
        height = max(64, _safe_int(base_timeline.get("generation_height", generation_height), _safe_int(generation_height, 720)))
        segment_duration = max(0.01, _safe_float(segment_seconds, 10.0))
        overlap = max(0, _safe_int(overlap_frames, 9))
        cap = max(1, _safe_int(frame_load_cap, int(round(effective_duration * fps_value))))
        estimated_segments = max(1, int(math.ceil(effective_duration / segment_duration)))

        contract = _mode_contract(v2v_mode, scail_identity_mode, bool(wan_background_lock))
        backend_template = _template_for_profile(contract["profile"])
        layers = _normalize_v5_layers(base_timeline, contract, str(output_stage), str(preview_stage))
        source_video = str(source_video_path or base_timeline.get("source_video_path") or _first_media_from_timeline(base_timeline, True) or "")
        source_image = str(source_image_path or base_timeline.get("source_image_path") or _first_media_from_timeline(base_timeline, False) or "")
        pose_image = str(pose_transfer_image_path or base_timeline.get("pose_transfer_image_path") or source_image)
        pose_video = str(pose_transfer_video_path or base_timeline.get("pose_transfer_video_path") or source_video)
        time_units_value = _choice(
            _first_value(base_timeline, ("timeUnits", "time_units", "display_mode", "displayMode"), time_units),
            ("seconds", "frames"),
            "seconds",
        )
        magnet_value = _as_bool(
            _first_value(base_timeline, ("magnetEnabled", "magnet_enabled", "snappingEnabled", "snapEnabled"), magnet_enabled),
            bool(magnet_enabled),
        )
        audio_input_value = _as_bool(
            _first_value(base_timeline, ("audioInputEnabled", "audio_input_enabled", "use_custom_audio", "useCustomAudio"), audio_input_enabled),
            bool(audio_input_enabled),
        )
        source_voice_value = _as_bool(
            _first_value(base_timeline, ("sourceVoiceLock", "source_voice_lock", "voiceLock", "voice_lock", "lock_source_voice"), source_voice_lock),
            bool(source_voice_lock),
        )
        v2v_enabled_value = _as_bool(
            _first_value(base_timeline, ("videoToVideoEnabled", "video_to_video_enabled"), video_to_video_enabled),
            bool(video_to_video_enabled or source_video),
        )
        inpaint_audio_value = _as_bool(
            _first_value(base_timeline, ("inpaintAudio", "inpaint_audio"), inpaint_audio),
            bool(inpaint_audio),
        )
        override_audio_value = _as_bool(_first_value(base_timeline, ("overrideAudio", "override_audio"), False), False)
        clip_edit_mode_value = _choice(
            _first_value(base_timeline, ("clipEditMode", "clip_edit_mode"), clip_edit_mode),
            ("timeline_trim_split_extend", "trim_only", "split_at_playhead", "extend_source", "retake_range", "combine_takes"),
            "timeline_trim_split_extend",
        )
        continuation_mode_value = _choice(
            _first_value(base_timeline, ("continuationMode", "continuation_mode"), continuation_mode),
            ("source_video", "last_frame", "guide_frame", "retake_range", "manual_cut"),
            "source_video",
        )
        guide_frame_policy_value = _choice(
            _first_value(base_timeline, ("guideFramePolicy", "guide_frame_policy"), guide_frame_policy),
            ("prompt_behavior_guide_geography", "first_last_keyframes", "prompt_only", "source_video_motion", "manual_keyframes"),
            "prompt_behavior_guide_geography",
        )
        prompt_blocks = _prompt_blocks_from_timeline(base_timeline)
        source_audio_segments = _source_audio_segments_from_timeline(base_timeline) if audio_input_value and source_voice_value else _first_list(base_timeline, ("sourceAudioSegments", "source_audio_segments"))
        timeline_control_contract = {
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
            "time_units": time_units_value,
            "magnet_enabled": bool(magnet_value),
            "clip_edit_mode": clip_edit_mode_value,
            "continuation_mode": continuation_mode_value,
            "guide_frame_policy": guide_frame_policy_value,
        }

        preview_channels = {
            "source": True,
            "pose": bool(str(pose_mode) != "none"),
            "sam31": bool(enable_sam31_preview and contract["backend_mode"] in {"scail2", "wananimate"}),
            "controlnet": bool(contract["backend_mode"] in {"wananimate", "pose_transfer"} and wan_control_preview),
            "result": bool(contract["backend_mode"] == "pose_transfer"),
            "taeltx": False,
        }
        backend_settings = {
            "backend_variant": contract["variant"],
            "ui_mode": contract["ui_mode"],
            "render_scope": str(render_scope or "selected_clip"),
            "time_units": time_units_value,
            "magnet_enabled": bool(magnet_value),
            "audio_input_enabled": bool(audio_input_value),
            "source_voice_lock": bool(source_voice_value),
            "video_to_video_enabled": bool(v2v_enabled_value),
            "inpaint_audio": bool(inpaint_audio_value),
            "override_audio": bool(override_audio_value),
            "clip_edit_mode": clip_edit_mode_value,
            "continuation_mode": continuation_mode_value,
            "guide_frame_policy": guide_frame_policy_value,
            "promptBlocks": prompt_blocks,
            "sourceAudioSegments": source_audio_segments,
            "timelineControlContract": timeline_control_contract,
            "scail_identity_mode": "multi_person_identity" if contract["ui_mode"] == "scail_multi" else str(scail_identity_mode or "single_person"),
            "scail_output_stage": str(scail_output_stage or "final_32fps_upscaled"),
            "enable_sam31_preview": bool(enable_sam31_preview),
            "wan_background_lock": bool(wan_background_lock),
            "wan_character_mask_mode": str(wan_character_mask_mode or "sam31_identity_mask"),
            "wan_control_preview": bool(wan_control_preview),
            "pose_transfer_image_path": pose_image,
            "pose_transfer_video_path": pose_video,
            "pose_transfer_result_path": str(pose_transfer_result_path or base_timeline.get("pose_transfer_result_path") or ""),
            "pose_transfer_result_mode": str(pose_transfer_result_mode or "use_as_reference"),
            "output_stage": str(output_stage or "final"),
            "preview_stage": str(preview_stage or output_stage or "final"),
            "modeLayers": layers["modeLayers"],
            "identityTracks": layers["identityTracks"],
            "controlTracks": layers["controlTracks"],
            "takes": layers["takes"],
        }
        v5_timeline = copy.deepcopy(base_timeline)
        v5_timeline.update({
            "schema": V5_SCHEMA,
            "schema_version": 1,
            "shotboard_version": 5,
            "ui_base": "IAMCCS_CineShotboardPlannerV4",
            "duration_seconds": float(duration),
            "source_duration_seconds": float(duration),
            "fps": float(fps_value),
            "frame_rate": float(fps_value),
            "trim_start_s": float(trim_start),
            "trim_end_s": float(trim_end),
            "generation_width": int(width),
            "generation_height": int(height),
            "frame_load_cap": int(cap),
            "v2v_mode": contract["ui_mode"],
            "videoToVideoEnabled": bool(v2v_enabled_value),
            "video_to_video_enabled": bool(v2v_enabled_value),
            "audioInputEnabled": bool(audio_input_value),
            "audio_input_enabled": bool(audio_input_value),
            "sourceVoiceLock": bool(source_voice_value),
            "source_voice_lock": bool(source_voice_value),
            "magnetEnabled": bool(magnet_value),
            "magnet_enabled": bool(magnet_value),
            "timeUnits": time_units_value,
            "time_units": time_units_value,
            "display_mode": time_units_value,
            "displayMode": time_units_value,
            "clipEditMode": clip_edit_mode_value,
            "clip_edit_mode": clip_edit_mode_value,
            "continuationMode": continuation_mode_value,
            "continuation_mode": continuation_mode_value,
            "guideFramePolicy": guide_frame_policy_value,
            "guide_frame_policy": guide_frame_policy_value,
            "inpaint_audio": bool(inpaint_audio_value),
            "inpaintAudio": bool(inpaint_audio_value),
            "override_audio": bool(override_audio_value),
            "overrideAudio": bool(override_audio_value),
            "backend_family": contract["family"],
            "backend_mode": contract["backend_mode"],
            "backend_profile": contract["profile"],
            "backend_variant": contract["variant"],
            "backend_settings": backend_settings,
            "backend_template": backend_template,
            "source_video_path": source_video,
            "source_image_path": source_image,
            "segment_seconds": float(segment_duration),
            "segment_preset": str(segment_preset),
            "planning_mode": str(planning_mode),
            "overlap_frames": int(overlap),
            "ltx_round_mode": str(ltx_round_mode),
            "vram_profile": str(vram_profile),
            "output_stage": str(output_stage or "final"),
            "preview_stage": str(preview_stage or output_stage or "final"),
            "render_scope": str(render_scope or "selected_clip"),
            "modeLayers": layers["modeLayers"],
            "identityTracks": layers["identityTracks"],
            "controlTracks": layers["controlTracks"],
            "takes": layers["takes"],
            "promptBlocks": prompt_blocks,
            "prompt_blocks": prompt_blocks,
            "sourceAudioSegments": source_audio_segments,
            "source_audio_segments": source_audio_segments,
            "timelineControlContract": timeline_control_contract,
            "production_modes": {
                "shotboard": "timeline_authoring",
                "audioboard": "audio_segments_and_lipsync_source",
                "video_editor": "takes_and_final_cut",
                "v2v": "active_backend_mode_layer",
            },
        })
        v2v_timeline = copy.deepcopy(v5_timeline)
        v2v_timeline.update({
            "schema": "iamccs.v2v.shotboard.timeline",
            "schema_version": 5,
        })
        v2v_outputs = {
            "duration_seconds": float(effective_duration),
            "fps": float(fps_value),
            "segment_duration_s": float(segment_duration),
            "planning_mode": str(planning_mode),
            "segment_preset": str(segment_preset),
            "overlap_frames": int(overlap),
            "ltx_round_mode": str(ltx_round_mode),
            "source_video_path": source_video,
            "source_image_path": source_image,
            "trim_start_s": float(trim_start),
            "trim_end_s": float(trim_end),
            "frame_load_cap": int(cap),
            "generation_width": int(width),
            "generation_height": int(height),
            "vram_profile": str(vram_profile),
            "backend_family": contract["family"],
            "backend_mode": contract["backend_mode"],
            "backend_profile": contract["profile"],
            "backend_template": backend_template,
            "backend_settings": backend_settings,
            "backend_variant": contract["variant"],
            "video_to_video_enabled": bool(v2v_enabled_value),
            "audio_input_enabled": bool(audio_input_value),
            "source_voice_lock": bool(source_voice_value),
            "inpaint_audio": bool(inpaint_audio_value),
            "override_audio": bool(override_audio_value),
            "magnet_enabled": bool(magnet_value),
            "time_units": time_units_value,
            "clip_edit_mode": clip_edit_mode_value,
            "continuation_mode": continuation_mode_value,
            "guide_frame_policy": guide_frame_policy_value,
            "prompt_blocks": prompt_blocks,
            "source_audio_segments": source_audio_segments,
            "timeline_control_contract": timeline_control_contract,
            "scail_identity_mode": backend_settings["scail_identity_mode"],
            "scail_output_stage": backend_settings["scail_output_stage"],
            "enable_sam31_preview": bool(enable_sam31_preview),
            "wan_background_lock": bool(wan_background_lock),
            "wan_character_mask_mode": backend_settings["wan_character_mask_mode"],
            "wan_control_preview": bool(wan_control_preview),
            "pose_transfer_image_path": pose_image,
            "pose_transfer_video_path": pose_video,
            "pose_transfer_result_path": backend_settings["pose_transfer_result_path"],
            "pose_transfer_result_mode": backend_settings["pose_transfer_result_mode"],
            "output_stage": str(output_stage or "final"),
            "preview_stage": str(preview_stage or output_stage or "final"),
            "dwpose_enabled": bool(dwpose_enabled),
            "dwpose_strength": float(_safe_float(dwpose_strength, 0.75)),
            "taeltx_preview_enabled": False,
            "taeltx_preview_max_frames": 17,
            "taeltx_preview_fps": 8,
            "preview_channels": preview_channels,
            "global_prompt": str(v5_timeline.get("global_prompt") or global_prompt or ""),
            "negative_prompt": str(negative_prompt or ""),
            "timeline_json": _json_dumps(v2v_timeline),
            "output_prefix": str(output_prefix or "IAMCCS/SHOTBOARD_V5_V2V"),
            "estimated_segments": int(estimated_segments),
            "segment_index": 0,
            "audio_vae_name": str(audio_vae_name or "ltx-2.3-22b-dev_audio_vae.safetensors"),
            "audio_vae_device": str(audio_vae_device or "main_device"),
            "audio_vae_dtype": str(audio_vae_dtype or "bf16"),
            "shotboard_v5_schema": V5_SCHEMA,
            "render_scope": str(render_scope or "selected_clip"),
        }
        report = (
            f"Shotboard V5 V2V | ui_mode={contract['ui_mode']} | backend={contract['profile']} | "
            f"duration={effective_duration:.3f}s @ {fps_value:.3f}fps | size={width}x{height} | "
            f"clip_layers={len(layers['modeLayers'])} | scope={render_scope}"
        )
        v2v_outputs["report"] = report

        outputs.update({
            "shotboard_version": 5,
            "pipeline_kind": "shotboard_v5_v2v",
            "timeline_data": _json_dumps(v5_timeline),
            "v5_timeline_data": _json_dumps(v5_timeline),
            "v2v_timeline_json": _json_dumps(v2v_timeline),
            "v2v_payload": v2v_outputs,
            "v5_report": report,
        })
        resources.update({
            "cine_v5_timeline_data_json": _json_dumps(v5_timeline),
            "cine_v5_report_json": _json_dumps({
                "node": "IAMCCS_CineShotboardPlannerV5V2V",
                "schema": V5_SCHEMA,
                "ui_base": "Shotboard V4",
                "v2v_mode": contract["ui_mode"],
                "backend_profile": contract["profile"],
                "modeLayers": len(layers["modeLayers"]),
                "identityTracks": len(layers["identityTracks"]),
                "controlTracks": len(layers["controlTracks"]),
                "takes": len(layers["takes"]),
                "promptBlocks": len(prompt_blocks),
                "sourceAudioSegments": len(source_audio_segments),
                "video_to_video_enabled": bool(v2v_enabled_value),
                "audio_input_enabled": bool(audio_input_value),
                "source_voice_lock": bool(source_voice_value),
                "inpaint_audio": bool(inpaint_audio_value),
                "magnet_enabled": bool(magnet_value),
                "time_units": time_units_value,
                "clip_edit_mode": clip_edit_mode_value,
                "continuation_mode": continuation_mode_value,
                "guide_frame_policy": guide_frame_policy_value,
                "truth": "V5 is a new-file planner shell. It preserves V4 timeline behavior while emitting router-compatible V2V resources.",
            }),
            "v2v_payload": dict(v2v_outputs),
            "v2v_timeline": v2v_timeline,
            "v2v_timeline_json": _json_dumps(v2v_timeline),
            "v2v_report": report,
            "v2v_backend_template": backend_template,
            "v2v_backend_settings": backend_settings,
            "cine_v5_prompt_blocks_json": _json_dumps(prompt_blocks),
            "cine_v5_source_audio_segments_json": _json_dumps(source_audio_segments),
            "cine_v5_timeline_control_contract_json": _json_dumps(timeline_control_contract),
        })
        out_linx["pipeline_kind"] = "shotboard_v5_v2v"
        out_linx["backend_id"] = f"IAMCCS_{contract['family'].upper()}_V5_V2V"
        out_linx["mode"] = "iamccs_shotboard_v5_v2v"
        out_linx.setdefault("chain", []).append({"role": "v2v_mode_adapter", "name": "IAMCCS Cine Shotboard Planner V5 V2V"})
        out_linx.setdefault("stages", []).append({
            "name": "SHOTBOARD_V5_V2V",
            "kind": contract["profile"],
            "variant": contract["variant"],
            "template": backend_template,
            "settings": backend_settings,
            "payload": dict(v2v_outputs),
        })
        policies = out_linx.setdefault("policies", {})
        policies["shotboard_v5_base"] = "shotboard_v4_timeline_meter_ui"
        policies["v2v_router_contract"] = "resources.v2v_payload -> IAMCCS_CineInfoV2VBackendRouter"
        policies["ui_language"] = "clips/layers/tracks/takes; backend may still export segments for compatibility"
        policies["shotboard_v5_editorial_contract"] = "main video lane + prompt blocks + source video audio lock + audio inpaint + IC/motion lane"
        out_linx["resource_keys"] = sorted(resources.keys())
        out_linx["resource_types"] = {key: type(value).__name__ for key, value in resources.items()}
        return (out_linx,)


class IAMCCS_CineShotboardV5V2VRouter(IAMCCS_CineInfoV2VBackendRouter):
    CATEGORY = "IAMCCS/Cine/Shotboard V5"


NODE_CLASS_MAPPINGS = {
    "IAMCCS_CineShotboardPlannerV5V2V": IAMCCS_CineShotboardPlannerV5V2V,
    "IAMCCS_CineShotboardV5V2VRouter": IAMCCS_CineShotboardV5V2VRouter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_CineShotboardPlannerV5V2V": "IAMCCS Cine Shotboard Planner V5 V2V",
    "IAMCCS_CineShotboardV5V2VRouter": "IAMCCS Cine Shotboard V5 V2V Router",
}
