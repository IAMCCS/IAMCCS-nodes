import json
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Tuple


SUPERNODE_LINX_TYPE = "IAMCCS_SUPERNODE_LINX"
V2V_MEDIA_BUS_TYPE = "IAMCCS_V2V_MEDIA_BUS"
V2V_CONTROL_BUS_TYPE = "IAMCCS_V2V_CONTROL_BUS"
V2V_PREVIEW_BUS_TYPE = "IAMCCS_V2V_PREVIEW_BUS"
_AUDIO_VAE_CACHE: Dict[str, Any] = {}
_LTX_AUDIO_ENCODE_GUARD_INSTALLED = False

V2V_BACKEND_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "ltx23_v2v_infinite_lipsync": {
        "label": "LTX Simple",
        "family": "ltx",
        "mode": "ltx_simple",
        "workflow_path": "C:/Users/carmc/Documents/IAMCCS_WORKFLOWS/NEW_LTX23/V2V/v2v real/DEFINITIVI/LTX23_V2V_INFINITE_NORMAL_VRAM_LIPSYNC_SHOTBOARD_PLANNER_V2V_CINELINX_AUDIO_GUARD.json",
        "core_nodes": ["LTX V2V sampler chain", "IAMCCS Shotboard Planner V2V", "LTX audio VAE guard"],
        "required_assets": ["source_video_path", "source_image_path", "audio_vae_name"],
        "preview_channels": ["source", "pose"],
        "output_stages": [{"name": "final", "prefix": "IAMCCS/LTX23_V2V_SHOTBOARD"}],
        "default_resolution": [1280, 720],
        "notes": "Current production LTX V2V backend with DW pose, lipsync/audio guard and segment planning.",
    },
    "scail2_single_person": {
        "label": "SCAIL-2 Single Person",
        "family": "scail2",
        "mode": "scail2",
        "workflow_path": "C:/Users/carmc/Documents/xorkflow da copiare/SCAIL2_NEW/civitai/FINAL SCAIL + WANANIMATE PER PATREON/IAMCCS_SCAIL2_SINGLE_PERSON.json",
        "core_nodes": ["IAMCCS_ScailExtends", "SAM3_TrackPreview", "VHS_LoadVideo", "LoadImage", "VHS_VideoCombine"],
        "required_assets": ["source_video_path", "source_image_path", "sam31_model", "clip_vision", "vae", "sampler"],
        "backend_inputs": ["pose_video", "pose_video_mask", "reference_image", "reference_image_mask", "clip_vision_output", "width", "height", "replacement_mode"],
        "preview_channels": ["source", "sam31_object_ids", "pose_mask", "reference"],
        "output_stages": [
            {"name": "generated_16fps", "prefix": "IAMCCS/SCAIL2_SINGLE_16FPS"},
            {"name": "final_32fps_upscaled", "prefix": "IAMCCS/SCAIL2_SINGLE_32FPS"},
        ],
        "default_resolution": [512, 896],
        "notes": "Single-person SCAIL replacement pipeline with SAM 3.1 tracking preview and 16/32 FPS outputs.",
    },
    "scail2_multi_person_identity": {
        "label": "SCAIL-2 Multi Identity",
        "family": "scail2",
        "mode": "scail2",
        "workflow_path": "C:/Users/carmc/Documents/xorkflow da copiare/SCAIL2_NEW/civitai/FINAL SCAIL + WANANIMATE PER PATREON/IAMCCS_SCAIL2_MULTI_PERSON_IDENTITY.json",
        "core_nodes": ["IAMCCS_ScailExtends", "IAMCCS_ScailIdentityTracker", "SCAIL2ColoredMask", "SAM3_TrackPreview"],
        "required_assets": ["source_video_path", "source_image_path", "sam3.1_multiplex_fp16", "clip_vision", "vae", "sampler"],
        "backend_inputs": ["pose_video", "pose_video_mask", "reference_image", "reference_image_mask", "clip_vision_output", "width", "height", "replacement_mode"],
        "preview_channels": ["source", "sam31_object_ids", "identity_masks", "colored_masks", "reference"],
        "output_stages": [
            {"name": "generated_16fps", "prefix": "IAMCCS/SCAIL2_MULTI_16FPS"},
            {"name": "final_32fps_upscaled", "prefix": "IAMCCS/SCAIL2_MULTI_32FPS"},
        ],
        "default_resolution": [512, 896],
        "notes": "Multi-person SCAIL identity pipeline with explicit identity tracker and colored SAM masks.",
    },
    "wananimate_extension": {
        "label": "WanAnimate Extension",
        "family": "wananimate",
        "mode": "wananimate",
        "workflow_path": "C:/Users/carmc/Documents/xorkflow da copiare/SCAIL2_NEW/civitai/FINAL SCAIL + WANANIMATE PER PATREON/IAMCCS_WAN-ANIMATE_EXTENSION-240626.json",
        "core_nodes": ["IAMCCS_WanAnimateExtends", "IAMCCS_ScailIdentityTracker", "SAM3_TrackToMask", "DrawViTPose", "DrawMaskOnImage"],
        "required_assets": ["source_video_path", "source_image_path", "vitpose", "sam31_model", "wan_model", "wan_vae"],
        "backend_inputs": ["reference_image", "face_video", "pose_video", "background_video", "character_mask", "width", "height"],
        "preview_channels": ["source", "vitpose", "sam31_mask", "character_mask", "background_video"],
        "output_stages": [{"name": "final_upscaled", "prefix": "IAMCCS/WAN_ANIMATE_SAM31_IDENTITY_SEGMENTATION_WANANIMATE_EXTENDS_FINAL"}],
        "default_resolution": [832, 480],
        "notes": "WanAnimate identity animation pipeline. Background lock expects a generated background_video plus character_mask.",
    },
    "flux_klein_pose_transfer": {
        "label": "FLUX Klein Pose Transfer",
        "family": "pose_transfer",
        "mode": "pose_transfer",
        "workflow_path": "C:/Users/carmc/Documents/xorkflow da copiare/SCAIL2_NEW/civitai/FINAL SCAIL + WANANIMATE PER PATREON/FLUX KLEIN POSE TRANSFER.json",
        "core_nodes": ["AIO_Preprocessor", "UNETLoader", "CLIPLoader", "VAELoader", "EmptyFlux2LatentImage", "KSampler", "VAEDecode"],
        "required_assets": ["pose_transfer_image_path", "pose_transfer_video_path", "flux_unet", "flux_clip", "vae"],
        "backend_inputs": ["reference_image", "driver_video", "pose_preview", "result_image"],
        "preview_channels": ["reference", "driver_video", "pose_preview", "result"],
        "output_stages": [{"name": "result_image", "prefix": "IAMCCS/POSE_TRANSFER_SHOTBOARD"}],
        "default_resolution": [768, 1024],
        "notes": "Canvas-style image+video pose transfer mode. Produces a result image/preview used as V2V reference or handoff asset.",
    },
}


def _clamp(value: Any, low: float, high: float, fallback: float) -> float:
    try:
        number = float(value)
    except Exception:
        number = fallback
    if not math.isfinite(number):
        number = fallback
    return max(low, min(high, number))


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


def _json_loads(text: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    try:
        data = json.loads(str(text or "").strip() or "{}")
    except Exception:
        return dict(fallback)
    return data if isinstance(data, dict) else dict(fallback)


def _backend_template(profile: str) -> Dict[str, Any]:
    template = V2V_BACKEND_TEMPLATES.get(str(profile or ""))
    if template is None:
        template = V2V_BACKEND_TEMPLATES["ltx23_v2v_infinite_lipsync"]
    return json.loads(json.dumps(template))


def _coerce_ltx_audio_encode_input(waveform, expected_channels: int = 2):
    if not hasattr(waveform, "ndim") or waveform.ndim != 3:
        return waveform
    shape = tuple(int(dim) for dim in waveform.shape)
    expected_channels = max(1, int(expected_channels or 2))
    # This Comfy build calls AudioVAE.encode(waveform.movedim(1, -1)),
    # while the LTX AudioVAE implementation checks waveform.shape[1].
    # Therefore encode must receive [B, C, S]. If we receive [B, S, C],
    # transpose it back here before the original implementation runs.
    if shape[1] > 8 and shape[-1] <= expected_channels:
        waveform = waveform.transpose(1, 2).contiguous()
        shape = tuple(int(dim) for dim in waveform.shape)
    if shape[1] == 1 and expected_channels > 1:
        waveform = waveform.repeat(1, expected_channels, 1)
    elif shape[1] > expected_channels:
        waveform = waveform[:, :expected_channels, :]
    return waveform


def _install_ltx_audio_encode_guard() -> None:
    global _LTX_AUDIO_ENCODE_GUARD_INSTALLED
    if _LTX_AUDIO_ENCODE_GUARD_INSTALLED:
        return
    try:
        from comfy.ldm.lightricks.vae.audio_vae import AudioVAE
    except Exception as exc:
        print(f"[IAMCCS_V2V_AUDIO_DEBUG] AudioVAE encode guard deferred: {exc!r}")
        return
    original = getattr(AudioVAE, "encode", None)
    if original is None or getattr(original, "_iamccs_v2v_audio_guard", False):
        _LTX_AUDIO_ENCODE_GUARD_INSTALLED = True
        return

    def guarded_encode(self, audio, *args, **kwargs):
        import torch

        expected_channels = 2
        model_dtype = None
        try:
            expected_channels = int(self.autoencoder.encoder.in_channels)
        except Exception:
            pass
        try:
            model_dtype = next(self.autoencoder.parameters()).dtype
        except Exception:
            model_dtype = None
        before = tuple(int(dim) for dim in audio.shape) if hasattr(audio, "shape") else None
        fixed = _coerce_ltx_audio_encode_input(audio, expected_channels)
        after = tuple(int(dim) for dim in fixed.shape) if hasattr(fixed, "shape") else None
        if before != after:
            print(f"[IAMCCS_V2V_AUDIO_DEBUG] AudioVAE.encode_guard input_shape={before} fixed_shape={after}")
        device_type = getattr(getattr(fixed, "device", None), "type", None)
        autocast_dtype = model_dtype if model_dtype in (torch.float16, torch.bfloat16) else None
        if autocast_dtype is not None:
            print(f"[IAMCCS_V2V_AUDIO_DEBUG] AudioVAE.encode_guard autocast device={device_type} dtype={autocast_dtype}")
        context = (
            torch.autocast(device_type=device_type, dtype=autocast_dtype)
            if autocast_dtype is not None and device_type in {"cuda", "cpu"}
            else nullcontext()
        )
        with context:
            return original(self, fixed, *args, **kwargs)

    guarded_encode._iamccs_v2v_audio_guard = True
    guarded_encode._iamccs_v2v_audio_original = original
    AudioVAE.encode = guarded_encode
    _LTX_AUDIO_ENCODE_GUARD_INSTALLED = True
    print("[IAMCCS_V2V_AUDIO_DEBUG] AudioVAE.encode_guard installed")


def _outputs(cine_linx: Any) -> Dict[str, Any]:
    if not isinstance(cine_linx, dict):
        return {}
    data = cine_linx.get("outputs")
    return data if isinstance(data, dict) else {}


def _resources(cine_linx: Any) -> Dict[str, Any]:
    if not isinstance(cine_linx, dict):
        return {}
    data = cine_linx.get("resources")
    return data if isinstance(data, dict) else {}


def _fix_ltx_frames(frames: int, mode: str) -> int:
    frames = max(1, int(frames))
    rem = (frames - 1) % 8
    if rem == 0:
        return frames
    down = max(1, frames - rem)
    up = frames + (8 - rem)
    if mode == "down":
        return down
    if mode == "nearest":
        return up if (up - frames) <= (frames - down) else down
    return up


def _normalize_segment_preset(segment_preset: str) -> str:
    value = str(segment_preset or "15sec")
    if value == "videoclip":
        return "10sec"
    if value == "monologue":
        return "15sec"
    if value in {"5sec", "10sec", "15sec", "20sec"}:
        return value
    return "15sec"


def _recommend_profile(segment_preset: str) -> Dict[str, Any]:
    preset = _normalize_segment_preset(segment_preset)
    if preset == "5sec":
        return {"segment_seconds": 5.0, "overlap_frames": 9, "audio_left_context_s": 0.25, "extension_preset": "videoclip_audio_24fps"}
    if preset == "20sec":
        return {"segment_seconds": 20.0, "overlap_frames": 9, "audio_left_context_s": 1.0, "extension_preset": "monologue_audio_24fps"}
    if preset == "15sec":
        return {"segment_seconds": 15.0, "overlap_frames": 9, "audio_left_context_s": 0.75, "extension_preset": "monologue_audio_24fps"}
    return {"segment_seconds": 10.0, "overlap_frames": 9, "audio_left_context_s": 0.5, "extension_preset": "videoclip_audio_24fps"}


def _recommend_profile_for_segment_seconds(segment_duration_s: float):
    duration = float(segment_duration_s)
    for preset, target_seconds in (("5sec", 5.0), ("10sec", 10.0), ("15sec", 15.0), ("20sec", 20.0)):
        if abs(duration - target_seconds) <= 0.1:
            return preset, _recommend_profile(preset)
    return None, None


def _plan_segment(
    song_duration_s: float,
    fps: float,
    segment_duration_s: float,
    planning_mode: str,
    segment_preset: str,
    overlap_frames: int,
    ltx_round_mode: str,
    segment_index: int,
) -> Dict[str, Any]:
    song_duration_s = max(0.01, float(song_duration_s))
    fps = max(0.001, float(fps))
    segment_duration_s = max(0.01, float(segment_duration_s))
    overlap_frames = max(0, int(overlap_frames))
    segment_index = max(0, int(segment_index))
    planning_mode = str(planning_mode or "manual_segment_seconds")
    if planning_mode == "auto_profile":
        planning_mode = "explicit_preset_seconds"
    if planning_mode not in {"manual_segment_seconds", "explicit_preset_seconds"}:
        planning_mode = "manual_segment_seconds"
    segment_preset = _normalize_segment_preset(segment_preset)

    rec = _recommend_profile(segment_preset)
    auto_duration_profile = None
    effective_planning_mode = planning_mode
    if planning_mode == "explicit_preset_seconds":
        segment_duration_s = max(0.01, float(rec["segment_seconds"]))
    else:
        auto_duration_profile, auto_rec = _recommend_profile_for_segment_seconds(segment_duration_s)
        if auto_rec is not None:
            rec = auto_rec

    total_frames = max(1, int(round(song_duration_s * fps)))
    unique_segment_frames = max(1, int(round(segment_duration_s * fps)))
    first_segment_raw_frames = _fix_ltx_frames(unique_segment_frames, str(ltx_round_mode))
    continuation_raw_frames = _fix_ltx_frames(unique_segment_frames + overlap_frames, str(ltx_round_mode))
    estimated_segments = max(1, int(math.ceil(float(total_frames) / float(unique_segment_frames))))
    continuation_loops = max(0, estimated_segments - 1)
    remainder = total_frames - unique_segment_frames * max(0, estimated_segments - 1)
    last_segment_unique_frames = unique_segment_frames if remainder <= 0 else int(remainder)
    last_segment_raw_frames = first_segment_raw_frames if estimated_segments <= 1 else _fix_ltx_frames(last_segment_unique_frames + overlap_frames, str(ltx_round_mode))

    clamped_segment_index = min(segment_index, max(0, estimated_segments - 1))
    current_segment_start_frames = unique_segment_frames * clamped_segment_index
    current_segment_unique_frames = max(0, min(unique_segment_frames, total_frames - current_segment_start_frames))
    if current_segment_unique_frames <= 0:
        current_segment_unique_frames = last_segment_unique_frames if clamped_segment_index == max(0, estimated_segments - 1) else unique_segment_frames
        current_segment_start_frames = min(current_segment_start_frames, max(0, total_frames - current_segment_unique_frames))
    current_segment_end_frames = min(total_frames, current_segment_start_frames + current_segment_unique_frames)
    current_remaining_frames_after = max(0, total_frames - current_segment_end_frames)
    if clamped_segment_index == 0:
        current_segment_raw_frames = first_segment_raw_frames
    elif clamped_segment_index >= estimated_segments - 1:
        current_segment_raw_frames = last_segment_raw_frames
    else:
        current_segment_raw_frames = continuation_raw_frames
    current_segment_start_s = float(current_segment_start_frames) / float(fps)
    current_segment_end_s = float(current_segment_end_frames) / float(fps)

    report = (
        f"song={song_duration_s:.3f}s @ {fps:.3f}fps -> total={total_frames}f | "
        f"segment={segment_duration_s:.3f}s -> unique={unique_segment_frames}f | "
        f"overlap={overlap_frames}f | first_raw={first_segment_raw_frames}f | "
        f"continuation_raw={continuation_raw_frames}f | segments={estimated_segments} | "
        f"loops={continuation_loops} | last_unique={last_segment_unique_frames}f | ltx_round={ltx_round_mode} | "
        f"planning={effective_planning_mode} | segment_preset={segment_preset}"
    )
    current_segment_report = (
        f"segment_index={clamped_segment_index} | raw={current_segment_raw_frames}f | unique={current_segment_unique_frames}f | "
        f"range=[{current_segment_start_frames}..{current_segment_end_frames}) | remaining_after={current_remaining_frames_after}f | "
        f"time=[{current_segment_start_s:.3f}s..{current_segment_end_s:.3f}s]"
    )
    planning_profile_report = (
        f"segment_preset={segment_preset} | planning_mode={effective_planning_mode} | segment_duration_s={segment_duration_s:.3f} | "
        f"recommended_overlap={int(rec['overlap_frames'])}f | recommended_left_context={float(rec['audio_left_context_s']):.2f}s | "
        f"recommended_extension_preset={str(rec['extension_preset'])} | auto_duration_profile={auto_duration_profile or 'none'}"
    )
    return {
        "total_frames": int(total_frames),
        "unique_segment_frames": int(unique_segment_frames),
        "first_segment_raw_frames": int(first_segment_raw_frames),
        "continuation_raw_frames": int(continuation_raw_frames),
        "estimated_segments": int(estimated_segments),
        "continuation_loops": int(continuation_loops),
        "last_segment_unique_frames": int(last_segment_unique_frames),
        "planner_report": report,
        "segment_index_out": int(clamped_segment_index),
        "current_segment_raw_frames": int(current_segment_raw_frames),
        "current_segment_unique_frames": int(current_segment_unique_frames),
        "current_segment_start_frames": int(current_segment_start_frames),
        "current_segment_end_frames": int(current_segment_end_frames),
        "current_remaining_frames_after": int(current_remaining_frames_after),
        "current_segment_start_s": float(current_segment_start_s),
        "current_segment_end_s": float(current_segment_end_s),
        "current_segment_report": current_segment_report,
        "fps_out": float(fps),
        "recommended_overlap_frames": int(rec["overlap_frames"]),
        "recommended_audio_left_context_s": float(rec["audio_left_context_s"]),
        "recommended_extension_preset": str(rec["extension_preset"]),
        "effective_planning_mode": effective_planning_mode,
        "planning_profile_report": planning_profile_report,
        "effective_segment_duration_s": float(segment_duration_s),
        "effective_segment_preset": segment_preset,
    }


def _source_range_from_plan(
    segment_index: int,
    current_segment_raw_frames: int,
    current_segment_unique_frames: int,
    current_segment_start_frames: int,
) -> Dict[str, Any]:
    segment_index = max(0, int(segment_index))
    current_segment_raw_frames = max(1, int(current_segment_raw_frames))
    current_segment_unique_frames = max(1, int(current_segment_unique_frames))
    current_segment_start_frames = max(0, int(current_segment_start_frames))

    overlap_delta = max(0, current_segment_raw_frames - current_segment_unique_frames)
    overlap_backtrack = max(0, overlap_delta - 1)
    if segment_index == 0:
        overlap_backtrack = 0

    range_start_index = max(0, current_segment_start_frames - overlap_backtrack)
    range_end_index = range_start_index + current_segment_raw_frames
    range_count = max(1, range_end_index - range_start_index)
    report = (
        f"segment_index={segment_index} | raw={current_segment_raw_frames}f | unique={current_segment_unique_frames}f | "
        f"start_unique={current_segment_start_frames}f | backtrack={overlap_backtrack}f | "
        f"range=[{range_start_index}..{range_end_index}) count={range_count}"
    )
    return {
        "range_start_index": int(range_start_index),
        "range_end_index": int(range_end_index),
        "range_count": int(range_count),
        "source_range_report": report,
    }


def _load_ltx_audio_vae_kj_compat(vae_name: str, device_name: str, weight_dtype: str):
    import os
    import torch
    import folder_paths
    from comfy import model_management
    from comfy.sd import VAE
    from comfy.utils import load_torch_file

    _install_ltx_audio_encode_guard()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(str(weight_dtype), torch.bfloat16)
    if str(device_name) == "cpu":
        device = torch.device("cpu")
    else:
        device = model_management.get_torch_device()
    cache_key = f"kjclone|{vae_name}|{device}|{dtype}"
    cached = _AUDIO_VAE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # Mirrors ComfyUI-KJNodes VAELoaderKJ.load_vae for the LTX audio VAE path.
    # Crucially, KJ's AudioVAE branch does not force .to(device, dtype), which
    # avoids CPU waveform vs CUDA weight mismatches in LTXVAudioVAEEncode.
    if str(vae_name) == "pixel_space":
        sd = {"pixel_space_vae": torch.tensor(1.0)}
        metadata = None
    else:
        vae_path = folder_paths.get_full_path_or_raise("vae", str(vae_name))
        sd, metadata = load_torch_file(vae_path, return_metadata=True)

    if "vocoder.conv_post.weight" in sd or "vocoder.vocoder.conv_post.weight" in sd:
        from comfy.ldm.lightricks.vae.audio_vae import AudioVAE

        try:
            vae = AudioVAE(sd, metadata)
        except TypeError as exc:
            if "positional arguments" not in str(exc):
                raise
            remapped_sd = {}
            for key, value in sd.items():
                if key.startswith("audio_vae."):
                    remapped_sd[f"autoencoder.{key[len('audio_vae.'):]}"] = value
                else:
                    remapped_sd[key] = value
            vae = AudioVAE(metadata=metadata)
            vae.load_state_dict(remapped_sd, strict=False)
    else:
        vae = VAE(sd=sd, device=device, dtype=dtype, metadata=metadata)
        vae.throw_exception_if_invalid()

    _AUDIO_VAE_CACHE[cache_key] = vae
    return vae


def _ensure_custom_node_path(folder_name: str) -> None:
    try:
        custom_nodes_root = Path(__file__).resolve().parents[2]
        candidate = custom_nodes_root / folder_name
        if candidate.exists():
            text = str(candidate)
            if text not in sys.path:
                sys.path.insert(0, text)
    except Exception:
        pass


def _ensure_iamccs_root_path() -> None:
    try:
        iamccs_root = Path(__file__).resolve().parents[1]
        text = str(iamccs_root)
        if text not in sys.path:
            sys.path.insert(0, text)
    except Exception:
        pass


def _load_vhs_video_internal(video_name: str, width: int, height: int, frame_load_cap: int):
    _ensure_custom_node_path("ComfyUI-VideoHelperSuite")
    try:
        from videohelpersuite.load_video_nodes import LoadVideoPath, LoadVideoUpload
    except Exception as exc:
        raise RuntimeError(f"VideoHelperSuite is required for internal V2V video loading: {exc!r}") from exc

    video_name = str(video_name or "").strip()
    if not video_name:
        raise ValueError("source_video_path is empty")
    kwargs = {
        "video": video_name,
        "force_rate": 0,
        "custom_width": int(width),
        "custom_height": int(height),
        "frame_load_cap": int(frame_load_cap),
        "skip_first_frames": 0,
        "select_every_nth": 1,
        "format": "AnimateDiff",
        "unique_id": "iamccs_v2v_internal_video",
    }
    is_path = os.path.isabs(video_name) or "/" in video_name or "\\" in video_name
    loader = LoadVideoPath() if is_path else LoadVideoUpload()
    return loader.load_video(**kwargs)


def _load_image_internal(image_name: str):
    try:
        from nodes import LoadImage
    except Exception as exc:
        raise RuntimeError(f"ComfyUI LoadImage is required for internal V2V image loading: {exc!r}") from exc
    image_name = str(image_name or "").strip()
    if not image_name:
        raise ValueError("source_image_path is empty")
    return LoadImage().load_image(image_name)


def _normalize_audio_for_ltx(audio):
    import torch

    if not isinstance(audio, dict):
        return audio
    waveform = audio.get("waveform")
    if not torch.is_tensor(waveform):
        return audio
    original_shape = tuple(int(dim) for dim in waveform.shape)
    if waveform.ndim == 1:
        waveform = waveform.view(1, 1, -1)
    elif waveform.ndim == 2:
        # Accept both [C, S] and [S, C]. Comfy AUDIO must be [B, C, S].
        if int(waveform.shape[0]) > 8 and int(waveform.shape[1]) <= 8:
            waveform = waveform.transpose(0, 1).contiguous()
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim == 3:
        # Some loaders/nodes return [B, S, C]. LTXVAudioVAEEncode calls
        # movedim(1, -1), so we must normalize to [B, C, S] first.
        if int(waveform.shape[1]) > 8 and int(waveform.shape[2]) <= 8:
            waveform = waveform.transpose(1, 2).contiguous()
    else:
        raise ValueError(f"Unsupported AUDIO waveform rank for LTX audio VAE: {waveform.ndim}")
    channels = int(waveform.shape[1])
    if channels <= 0:
        raise ValueError("AUDIO waveform has no channels")
    if channels == 1:
        waveform = waveform.repeat(1, 2, 1)
    elif channels > 2:
        waveform = waveform[:, :2, :]
    out = dict(audio)
    out["waveform"] = waveform.contiguous()
    out["sample_rate"] = int(out.get("sample_rate", 44100) or 44100)
    print(
        "[IAMCCS_V2V_AUDIO_DEBUG] normalize_for_ltx "
        f"input_shape={original_shape} output_shape={tuple(int(dim) for dim in out['waveform'].shape)} "
        f"sample_rate={out['sample_rate']}"
    )
    return out


def _debug_audio(label: str, audio):
    try:
        waveform = audio.get("waveform") if isinstance(audio, dict) else None
        sample_rate = audio.get("sample_rate") if isinstance(audio, dict) else None
        shape = tuple(int(dim) for dim in waveform.shape) if hasattr(waveform, "shape") else None
        print(f"[IAMCCS_V2V_AUDIO_DEBUG] {label} shape={shape} sample_rate={sample_rate}")
    except Exception as exc:
        print(f"[IAMCCS_V2V_AUDIO_DEBUG] {label} debug_failed={exc!r}")


def _slice_images_internal(images, start_index: int, end_index: int, count: int):
    import torch

    if not torch.is_tensor(images) or images.ndim != 4:
        raise ValueError("source video images must be an IMAGE tensor [N,H,W,C]")
    total = int(images.shape[0])
    if total <= 0:
        raise ValueError("source video images are empty")
    start = max(0, int(start_index))
    end = max(start, int(end_index))
    if start >= total:
        start = max(0, total - max(1, int(count)))
        end = total
    end = min(max(start + 1, end), total)
    selected = images[start:end].clone().contiguous()
    if int(selected.shape[0]) <= 0:
        raise ValueError(f"internal source image range is empty: [{start}..{end}) total={total}")
    return selected, int(selected.shape[0])


def _audio_segment_internal(audio, fps: float, plan: Dict[str, Any]):
    _ensure_iamccs_root_path()
    try:
        from iamccs_audio_extender import IAMCCS_AudioExtender
    except Exception as exc:
        raise RuntimeError(f"IAMCCS_AudioExtender is required for internal V2V audio slicing: {exc!r}") from exc
    return IAMCCS_AudioExtender().slice_segment(
        audio,
        float(fps),
        "no_overlap",
        0.0,
        0.0,
        "use_timeline_cursor",
        "snap_to_video_duration",
        "soft_clamp",
        segment_index=int(plan["segment_index_out"]),
        segment_duration_s=float(plan["effective_segment_duration_s"]),
        video_frames=int(plan["current_segment_raw_frames"]),
        generated_frames=int(plan["current_segment_raw_frames"]),
        extension_frames=int(plan["current_segment_unique_frames"]),
        timeline_cursor_frames=int(plan["current_segment_start_frames"]),
        segment_start_frames=int(plan["current_segment_start_frames"]),
        effective_unique_frames=int(plan["current_segment_unique_frames"]),
        first_pass_unique_frames=int(plan["unique_segment_frames"]),
    )


def _fallback_mask_from_image(image):
    import torch

    if torch.is_tensor(image) and image.ndim == 4:
        return torch.zeros((int(image.shape[0]), int(image.shape[1]), int(image.shape[2])), dtype=image.dtype, device=image.device)
    return torch.zeros((1, 64, 64), dtype=torch.float32)


def _white_mask_from_image(image):
    import torch

    if torch.is_tensor(image) and image.ndim == 4:
        return torch.ones((int(image.shape[0]), int(image.shape[1]), int(image.shape[2])), dtype=image.dtype, device=image.device)
    return torch.ones((1, 64, 64), dtype=torch.float32)


def _mask_image_from_image(image, value: float = 1.0):
    import torch

    if torch.is_tensor(image) and image.ndim == 4:
        return torch.ones_like(image[..., :3]) * float(value)
    return torch.ones((1, 64, 64, 3), dtype=torch.float32) * float(value)


def _payload_from_cine_linx(cine_linx: Any) -> Dict[str, Any]:
    if isinstance(cine_linx, dict) and (
        "backend_mode" in cine_linx
        or "backend_family" in cine_linx
        or "backend_profile" in cine_linx
        or "output_stage" in cine_linx
        or "preview_stage" in cine_linx
    ):
        return cine_linx
    out = _outputs(cine_linx)
    resources = _resources(cine_linx)
    payload = resources.get("v2v_payload") if isinstance(resources.get("v2v_payload"), dict) else out
    return payload if isinstance(payload, dict) else {}


def _load_planner_media(payload: Dict[str, Any], source_images=None, source_audio=None):
    import torch

    source_video_path = str(payload.get("source_video_path") or "IMG_4145 2.mp4")
    source_image_path = str(payload.get("source_image_path") or "QWEN2509_FIRST_FRAME_DWPOSE_OPENPOSE_CONTROL_00001_.png")
    width = int(max(64, _safe_int(payload.get("generation_width"), 1280)))
    height = int(max(64, _safe_int(payload.get("generation_height"), 720)))
    frame_cap = int(max(1, _safe_int(payload.get("frame_load_cap"), 241)))
    loaded_images = source_images
    loaded_audio = source_audio
    source_video_info = {}
    if loaded_images is None or loaded_audio is None:
        loaded_images, _frame_count, video_audio, source_video_info = _load_vhs_video_internal(
            source_video_path,
            width,
            height,
            frame_cap,
        )
        if loaded_audio is None:
            loaded_audio = video_audio
    loaded_audio = _normalize_audio_for_ltx(loaded_audio)
    try:
        source_image, source_mask = _load_image_internal(source_image_path)
    except Exception:
        source_image = loaded_images[:1] if torch.is_tensor(loaded_images) and loaded_images.ndim == 4 else torch.zeros((1, height, width, 3), dtype=torch.float32)
        source_mask = _fallback_mask_from_image(source_image)
    return loaded_images, loaded_audio, source_video_info, source_image, source_mask


def _bus_payload(control_bus: Any) -> Dict[str, Any]:
    if isinstance(control_bus, dict):
        payload = control_bus.get("payload")
        if isinstance(payload, dict):
            return payload
    return {}


def _bus_control(control_bus: Any) -> Dict[str, Any]:
    return control_bus if isinstance(control_bus, dict) else {}


def _bus_media(media_bus: Any) -> Dict[str, Any]:
    return media_bus if isinstance(media_bus, dict) else {}


def _bus_value(bus: Dict[str, Any], key: str, fallback: Any = None) -> Any:
    return bus.get(key, fallback) if isinstance(bus, dict) else fallback


class IAMCCS_ShotboardPlannerV2V:
    """All-in-one V2V planning panel with one cine_linx output."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_video_path": ("STRING", {"default": "IMG_4145 2.mp4"}),
                "source_image_path": ("STRING", {"default": "QWEN2509_FIRST_FRAME_DWPOSE_OPENPOSE_CONTROL_00001_.png"}),
                "duration_seconds": ("FLOAT", {"default": 10.0, "min": 0.01, "max": 36000.0, "step": 0.01}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}),
                "trim_start_s": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.01}),
                "trim_end_s": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.01}),
                "frame_load_cap": ("INT", {"default": 241, "min": 1, "max": 100000, "step": 1}),
                "generation_width": ("INT", {"default": 1280, "min": 64, "max": 8192, "step": 8}),
                "generation_height": ("INT", {"default": 720, "min": 64, "max": 8192, "step": 8}),
                "segment_seconds": ("FLOAT", {"default": 10.0, "min": 0.01, "max": 3600.0, "step": 0.01}),
                "planning_mode": (["manual_segment_seconds", "explicit_preset_seconds"], {"default": "explicit_preset_seconds"}),
                "segment_preset": (["5sec", "10sec", "15sec", "20sec", "videoclip", "monologue"], {"default": "5sec"}),
                "overlap_frames": ("INT", {"default": 9, "min": 0, "max": 4096, "step": 1}),
                "ltx_round_mode": (["up", "nearest", "down"], {"default": "up"}),
                "vram_profile": (["normal_vram", "low_vram"], {"default": "normal_vram"}),
                "backend_profile": ([
                    "ltx23_v2v_infinite_lipsync",
                    "ltx23_v2v_loop_antidegrade",
                    "scail2_single_person",
                    "scail2_multi_person_identity",
                    "wananimate_extension",
                    "flux_klein_pose_transfer",
                ], {"default": "ltx23_v2v_infinite_lipsync"}),
                "audio_vae_name": ("STRING", {"default": "ltx-2.3-22b-dev_audio_vae.safetensors"}),
                "audio_vae_device": (["main_device", "cpu"], {"default": "main_device"}),
                "audio_vae_dtype": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "pose_mode": (["none", "dwpose_openpose", "source_pose_only", "image_pose_transfer"], {"default": "dwpose_openpose"}),
                "dwpose_enabled": ("BOOLEAN", {"default": True}),
                "dwpose_strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.5, "step": 0.01}),
                "taeltx_preview_enabled": ("BOOLEAN", {"default": False}),
                "taeltx_preview_max_frames": ("INT", {"default": 17, "min": 1, "max": 257, "step": 1}),
                "taeltx_preview_fps": ("INT", {"default": 8, "min": 1, "max": 60, "step": 1}),
                "global_prompt": ("STRING", {"default": "cinematic video-to-video continuation, coherent motion, stable identity, natural lipsync", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "cartoon, ugly, unstable anatomy, flicker, broken motion, identity drift, subtitles, text", "multiline": True}),
                "timeline_data": ("STRING", {"default": "", "multiline": True}),
                "output_prefix": ("STRING", {"default": "IAMCCS/LTX23_V2V_SHOTBOARD"}),
                "backend_family": (["ltx", "scail2", "wananimate", "pose_transfer"], {"default": "ltx"}),
                "backend_mode": (["ltx_simple", "scail2", "wananimate", "pose_transfer"], {"default": "ltx_simple"}),
                "backend_variant": ([
                    "ltx_simple",
                    "scail2_single_person",
                    "scail2_multi_person_identity",
                    "wananimate_bg_locked",
                    "flux_klein_pose_transfer",
                ], {"default": "ltx_simple"}),
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
                "output_stage": (["final", "draft", "sam_preview", "mask_preview", "pose_preview", "reference_preview", "result_image"], {"default": "final"}),
                "preview_stage": (["final", "draft", "sam_preview", "mask_preview", "pose_preview", "reference_preview", "result_image", "source"], {"default": "final"}),
            }
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE,)
    RETURN_NAMES = ("cine_linx",)
    FUNCTION = "plan"
    CATEGORY = "IAMCCS/V2V"

    def plan(
        self,
        source_video_path: str,
        source_image_path: str,
        duration_seconds: float,
        fps: float,
        trim_start_s: float,
        trim_end_s: float,
        frame_load_cap: int,
        generation_width: int,
        generation_height: int,
        segment_seconds: float,
        planning_mode: str,
        segment_preset: str,
        overlap_frames: int,
        ltx_round_mode: str,
        vram_profile: str,
        backend_profile: str,
        audio_vae_name: str,
        audio_vae_device: str,
        audio_vae_dtype: str,
        pose_mode: str,
        dwpose_enabled: bool,
        dwpose_strength: float,
        taeltx_preview_enabled: bool,
        taeltx_preview_max_frames: int,
        taeltx_preview_fps: int,
        global_prompt: str,
        negative_prompt: str,
        timeline_data: str,
        output_prefix: str,
        backend_family: str = "ltx",
        backend_mode: str = "ltx_simple",
        backend_variant: str = "ltx_simple",
        scail_identity_mode: str = "single_person",
        scail_output_stage: str = "final_32fps_upscaled",
        enable_sam31_preview: bool = True,
        wan_background_lock: bool = True,
        wan_character_mask_mode: str = "sam31_identity_mask",
        wan_control_preview: bool = True,
        pose_transfer_image_path: str = "",
        pose_transfer_video_path: str = "",
        pose_transfer_result_path: str = "",
        pose_transfer_result_mode: str = "use_as_reference",
        output_stage: str = "final",
        preview_stage: str = "final",
    ) -> Tuple[Any, ...]:
        duration = max(0.01, _safe_float(duration_seconds, 10.0))
        fps_value = max(1.0, _safe_float(fps, 24.0))
        trim_start = max(0.0, _safe_float(trim_start_s, 0.0))
        trim_end = _safe_float(trim_end_s, duration)
        if trim_end <= trim_start:
            trim_start = 0.0
            trim_end = duration
        trim_end = min(max(trim_end, trim_start + 0.01), duration)
        effective_duration = max(0.01, trim_end - trim_start)
        segment_duration = max(0.01, _safe_float(segment_seconds, 10.0))
        overlap = max(0, _safe_int(overlap_frames, 9))
        cap = max(1, _safe_int(frame_load_cap, int(round(effective_duration * fps_value))))
        width = max(64, _safe_int(generation_width, 1280))
        height = max(64, _safe_int(generation_height, 720))
        preview_frames = max(1, _safe_int(taeltx_preview_max_frames, 17))
        preview_fps = max(1, _safe_int(taeltx_preview_fps, 8))
        pose_strength = _clamp(dwpose_strength, 0.0, 1.5, 0.75)
        estimated_segments = max(1, int(math.ceil(effective_duration / segment_duration)))
        mode_value = str(backend_mode or "ltx_simple")
        family_value = str(backend_family or "ltx")
        profile_value = str(backend_profile or "ltx23_v2v_infinite_lipsync")
        variant_value = str(backend_variant or "")
        scail_identity_value = str(scail_identity_mode or "single_person")
        scail_output_value = str(scail_output_stage or "final_32fps_upscaled")
        wan_mask_value = str(wan_character_mask_mode or "sam31_identity_mask")
        pose_result_mode_value = str(pose_transfer_result_mode or "use_as_reference")
        output_stage_value = str(output_stage or "final")
        preview_stage_value = str(preview_stage or output_stage_value or "final")
        if mode_value == "scail2":
            family_value = "scail2"
            if variant_value == "scail2_multi_person_identity" or scail_identity_value == "multi_person_identity":
                profile_value = "scail2_multi_person_identity"
                scail_identity_value = "multi_person_identity"
            else:
                profile_value = "scail2_single_person"
                scail_identity_value = "single_person"
        elif mode_value == "wananimate":
            family_value = "wananimate"
            profile_value = "wananimate_extension"
            variant_value = "wananimate_bg_locked" if bool(wan_background_lock) else "wananimate_extension"
        elif mode_value == "pose_transfer":
            family_value = "pose_transfer"
            profile_value = "flux_klein_pose_transfer"
            variant_value = "flux_klein_pose_transfer"
        elif mode_value == "ltx_simple":
            family_value = "ltx"
            if not profile_value.startswith("ltx"):
                profile_value = "ltx23_v2v_infinite_lipsync"
            variant_value = "ltx_simple"

        backend_template = _backend_template(profile_value)

        timeline = _json_loads(timeline_data, {})
        source_video_path = str(source_video_path or timeline.get("source_video_path") or "IMG_4145 2.mp4")
        source_image_path = str(source_image_path or timeline.get("source_image_path") or "QWEN2509_FIRST_FRAME_DWPOSE_OPENPOSE_CONTROL_00001_.png")
        pose_transfer_image_path = str(pose_transfer_image_path or timeline.get("pose_transfer_image_path") or source_image_path)
        pose_transfer_video_path = str(pose_transfer_video_path or timeline.get("pose_transfer_video_path") or source_video_path)
        pose_transfer_result_path = str(pose_transfer_result_path or timeline.get("pose_transfer_result_path") or "")
        preview_channels = {
            "source": True,
            "pose": bool(str(pose_mode) != "none"),
            "sam31": bool(enable_sam31_preview and mode_value in {"scail2", "wananimate"}),
            "controlnet": bool(mode_value in {"wananimate", "pose_transfer"} and wan_control_preview),
            "result": bool(mode_value == "pose_transfer"),
            "taeltx": False,
        }
        backend_settings = {
            "backend_variant": variant_value,
            "scail_identity_mode": scail_identity_value,
            "scail_output_stage": scail_output_value,
            "enable_sam31_preview": bool(enable_sam31_preview),
            "wan_background_lock": bool(wan_background_lock),
            "wan_character_mask_mode": wan_mask_value,
            "wan_control_preview": bool(wan_control_preview),
            "pose_transfer_image_path": pose_transfer_image_path,
            "pose_transfer_video_path": pose_transfer_video_path,
            "pose_transfer_result_path": pose_transfer_result_path,
            "pose_transfer_result_mode": pose_result_mode_value,
            "output_stage": output_stage_value,
            "preview_stage": preview_stage_value,
        }
        timeline.update({
            "schema": "iamccs.v2v.shotboard.timeline",
            "schema_version": 4,
            "source_video_path": source_video_path,
            "source_image_path": source_image_path,
            "duration_seconds": effective_duration,
            "source_duration_seconds": duration,
            "fps": fps_value,
            "trim_start_s": trim_start,
            "trim_end_s": trim_end,
            "frame_load_cap": cap,
            "generation_width": width,
            "generation_height": height,
            "segment_seconds": segment_duration,
            "overlap_frames": overlap,
            "planning_mode": str(planning_mode),
            "segment_preset": str(segment_preset),
            "ltx_round_mode": str(ltx_round_mode),
            "vram_profile": str(vram_profile),
            "backend_family": family_value,
            "backend_mode": mode_value,
            "backend_profile": profile_value,
            "backend_template": backend_template,
            "backend_settings": backend_settings,
            "backend_variant": variant_value,
            "scail_identity_mode": scail_identity_value,
            "scail_output_stage": scail_output_value,
            "enable_sam31_preview": bool(enable_sam31_preview),
            "wan_background_lock": bool(wan_background_lock),
            "wan_character_mask_mode": wan_mask_value,
            "wan_control_preview": bool(wan_control_preview),
            "pose_transfer_image_path": pose_transfer_image_path,
            "pose_transfer_video_path": pose_transfer_video_path,
            "pose_transfer_result_path": pose_transfer_result_path,
            "pose_transfer_result_mode": pose_result_mode_value,
            "output_stage": output_stage_value,
            "preview_stage": preview_stage_value,
            "audio_vae_name": str(audio_vae_name or "ltx-2.3-22b-dev_audio_vae.safetensors"),
            "audio_vae_device": str(audio_vae_device or "main_device"),
            "audio_vae_dtype": str(audio_vae_dtype or "bf16"),
            "pose_mode": str(pose_mode),
            "dwpose_enabled": bool(dwpose_enabled),
            "dwpose_strength": pose_strength,
            "taeltx_preview_enabled": False,
            "taeltx_preview_max_frames": preview_frames,
            "taeltx_preview_fps": preview_fps,
            "preview_channels": preview_channels,
            "global_prompt": str(global_prompt or ""),
            "negative_prompt": str(negative_prompt or ""),
            "output_prefix": str(output_prefix or ""),
        })

        outputs = {
            "duration_seconds": float(effective_duration),
            "fps": float(fps_value),
            "segment_duration_s": float(segment_duration),
            "planning_mode": str(planning_mode),
            "segment_preset": str(segment_preset),
            "overlap_frames": int(overlap),
            "ltx_round_mode": str(ltx_round_mode),
            "source_video_path": source_video_path,
            "source_image_path": source_image_path,
            "trim_start_s": float(trim_start),
            "trim_end_s": float(trim_end),
            "frame_load_cap": int(cap),
            "generation_width": int(width),
            "generation_height": int(height),
            "vram_profile": str(vram_profile),
            "backend_family": family_value,
            "backend_mode": mode_value,
            "backend_profile": profile_value,
            "backend_template": backend_template,
            "backend_settings": backend_settings,
            "backend_variant": variant_value,
            "scail_identity_mode": scail_identity_value,
            "scail_output_stage": scail_output_value,
            "enable_sam31_preview": bool(enable_sam31_preview),
            "wan_background_lock": bool(wan_background_lock),
            "wan_character_mask_mode": wan_mask_value,
            "wan_control_preview": bool(wan_control_preview),
            "pose_transfer_image_path": pose_transfer_image_path,
            "pose_transfer_video_path": pose_transfer_video_path,
            "pose_transfer_result_path": pose_transfer_result_path,
            "pose_transfer_result_mode": pose_result_mode_value,
            "output_stage": output_stage_value,
            "preview_stage": preview_stage_value,
            "dwpose_enabled": bool(dwpose_enabled),
            "dwpose_strength": float(pose_strength),
            "taeltx_preview_enabled": False,
            "taeltx_preview_max_frames": int(preview_frames),
            "taeltx_preview_fps": int(preview_fps),
            "preview_channels": preview_channels,
            "global_prompt": str(global_prompt or ""),
            "negative_prompt": str(negative_prompt or ""),
            "timeline_json": json.dumps(timeline, ensure_ascii=False, indent=2),
            "output_prefix": str(output_prefix or "IAMCCS/LTX23_V2V_SHOTBOARD"),
            "estimated_segments": int(estimated_segments),
            "segment_index": 0,
            "audio_vae_name": str(audio_vae_name or "ltx-2.3-22b-dev_audio_vae.safetensors"),
            "audio_vae_device": str(audio_vae_device or "main_device"),
            "audio_vae_dtype": str(audio_vae_dtype or "bf16"),
        }
        report = (
            f"V2V Shotboard | mode={mode_value} | variant={variant_value} | backend={profile_value} | vram={vram_profile} | "
            f"duration={effective_duration:.3f}s @ {fps_value:.3f}fps | cap={cap} | "
            f"resolution={width}x{height} | segment={segment_duration:.3f}s | "
            f"segments~{estimated_segments} | overlap={overlap}f"
        )
        outputs["report"] = report

        resources = {
            "v2v_payload": dict(outputs),
            "v2v_timeline": timeline,
            "v2v_timeline_json": outputs["timeline_json"],
            "v2v_report": report,
            "v2v_backend_template": backend_template,
            "v2v_backend_settings": backend_settings,
        }
        cine_linx = {
            "type": SUPERNODE_LINX_TYPE,
            "pipeline_kind": "v2v",
            "backend_id": f"IAMCCS_{family_value.upper()}_V2V",
            "mode": f"iamccs_{mode_value}_shotboard",
            "chain": [{"role": "planner", "name": "IAMCCS Shotboard Planner V2V"}],
            "stages": [{
                "name": f"{family_value.upper()}_V2V",
                "kind": profile_value,
                "variant": variant_value,
                "template": backend_template,
                "settings": backend_settings,
                "payload": dict(outputs),
            }],
            "outputs": outputs,
            "resources": resources,
            "resource_keys": sorted(resources.keys()),
            "resource_types": {key: type(value).__name__ for key, value in resources.items()},
        }
        return (cine_linx,)


class IAMCCS_CineInfoV2V:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"cine_linx": (SUPERNODE_LINX_TYPE,)},
            "optional": {
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "load_audio_vae": ("BOOLEAN", {"default": False}),
                "source_images": ("IMAGE",),
                "source_audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = (
        "FLOAT",
        "INT",
        "INT",
        "STRING",
        "STRING",
        "STRING",
        "INT",
        "STRING",
        "INT",
        "VAE",
        "IMAGE",
        "AUDIO",
        "VHS_VIDEOINFO",
        "IMAGE",
        "IMAGE",
        "AUDIO",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "fps",
        "generation_width",
        "generation_height",
        "global_prompt",
        "negative_prompt",
        "output_prefix",
        "continuation_loops",
        "planner_report",
        "segment_index_out",
        "audio_vae",
        "source_video_images",
        "source_audio",
        "source_video_info",
        "source_image",
        "current_source_images",
        "current_segment_audio",
        "current_audio_report",
        "report",
    )
    FUNCTION = "extract"
    CATEGORY = "IAMCCS/V2V"

    def extract(self, cine_linx, segment_index=0, load_audio_vae=False, source_images=None, source_audio=None):
        out = _outputs(cine_linx)
        resources = _resources(cine_linx)
        payload = resources.get("v2v_payload") if isinstance(resources.get("v2v_payload"), dict) else out
        report = str(payload.get("report", out.get("report", "")) or "")
        timeline_json = str(payload.get("timeline_json", resources.get("v2v_timeline_json", "{}")) or "{}")
        timeline_payload = _json_loads(timeline_json, {})
        source_video_path = str(payload.get("source_video_path") or timeline_payload.get("source_video_path") or "IMG_4145 2.mp4")
        source_image_path = str(payload.get("source_image_path") or timeline_payload.get("source_image_path") or "QWEN2509_FIRST_FRAME_DWPOSE_OPENPOSE_CONTROL_00001_.png")
        duration = float(_safe_float(payload.get("duration_seconds"), 10.0))
        fps = float(_safe_float(payload.get("fps"), 24.0))
        segment_duration = float(_safe_float(payload.get("segment_duration_s"), 10.0))
        planning_mode = str(payload.get("planning_mode", "explicit_preset_seconds") or "explicit_preset_seconds")
        segment_preset = str(payload.get("segment_preset", "5sec") or "5sec")
        overlap = int(max(0, _safe_int(payload.get("overlap_frames"), 9)))
        ltx_round = str(payload.get("ltx_round_mode", "up") or "up")
        plan = _plan_segment(duration, fps, segment_duration, planning_mode, segment_preset, overlap, ltx_round, _safe_int(segment_index, 0))
        source_range = _source_range_from_plan(
            int(plan["segment_index_out"]),
            int(plan["current_segment_raw_frames"]),
            int(plan["current_segment_unique_frames"]),
            int(plan["current_segment_start_frames"]),
        )
        audio_vae = _load_ltx_audio_vae_kj_compat(
            str(payload.get("audio_vae_name", "ltx-2.3-22b-dev_audio_vae.safetensors") or "ltx-2.3-22b-dev_audio_vae.safetensors"),
            str(payload.get("audio_vae_device", "main_device") or "main_device"),
            str(payload.get("audio_vae_dtype", "bf16") or "bf16"),
        )
        source_frame_count = 0
        source_video_info = {}
        loaded_audio = source_audio
        loaded_images = source_images
        _debug_audio("extract.source_audio_input", loaded_audio)
        if loaded_images is None or loaded_audio is None:
            loaded_images, source_frame_count, video_audio, source_video_info = _load_vhs_video_internal(
                source_video_path,
                int(max(64, _safe_int(payload.get("generation_width"), 1280))),
                int(max(64, _safe_int(payload.get("generation_height"), 720))),
                int(max(1, _safe_int(payload.get("frame_load_cap"), 241))),
            )
            _debug_audio("extract.video_audio_loaded", video_audio)
            if loaded_audio is None:
                loaded_audio = video_audio
        else:
            try:
                source_frame_count = int(getattr(loaded_images, "shape", [0])[0])
            except Exception:
                source_frame_count = 0
        loaded_audio = _normalize_audio_for_ltx(loaded_audio)
        _debug_audio("extract.loaded_audio_normalized", loaded_audio)

        try:
            source_image, source_mask = _load_image_internal(source_image_path)
        except Exception:
            source_image = loaded_images[:1]
            source_mask = _fallback_mask_from_image(source_image)

        current_source_images, current_source_count = _slice_images_internal(
            loaded_images,
            int(source_range["range_start_index"]),
            int(source_range["range_end_index"]),
            int(source_range["range_count"]),
        )
        audio_outputs = _audio_segment_internal(loaded_audio, fps, plan)
        _debug_audio("extract.current_segment_audio_raw", audio_outputs[1])
        current_segment_audio = _normalize_audio_for_ltx(audio_outputs[1])
        _debug_audio("extract.current_segment_audio_normalized", current_segment_audio)
        current_audio_report = str(audio_outputs[8])
        return (
            fps,
            int(max(64, _safe_int(payload.get("generation_width"), 1280))),
            int(max(64, _safe_int(payload.get("generation_height"), 720))),
            str(payload.get("global_prompt", "") or ""),
            str(payload.get("negative_prompt", "") or ""),
            str(payload.get("output_prefix", "IAMCCS/LTX23_V2V_SHOTBOARD") or "IAMCCS/LTX23_V2V_SHOTBOARD"),
            int(plan["continuation_loops"]),
            str(plan["planner_report"]),
            int(plan["segment_index_out"]),
            audio_vae,
            loaded_images,
            loaded_audio,
            source_video_info,
            source_image,
            current_source_images,
            current_segment_audio,
            current_audio_report,
            report,
        )


class IAMCCS_CineInfoV2VBackendRouter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"cine_linx": (SUPERNODE_LINX_TYPE,)},
            "optional": {
                "source_images": ("IMAGE",),
                "source_audio": ("AUDIO",),
                "result_image": ("IMAGE",),
                "pose_video_mask_image": ("IMAGE",),
                "reference_image_mask_image": ("IMAGE",),
                "face_video": ("IMAGE",),
                "pose_video": ("IMAGE",),
                "background_video": ("IMAGE",),
                "character_mask": ("MASK",),
            },
        }

    RETURN_TYPES = (V2V_MEDIA_BUS_TYPE, V2V_CONTROL_BUS_TYPE, V2V_PREVIEW_BUS_TYPE, "STRING")
    RETURN_NAMES = ("media_bus", "control_bus", "preview_bus", "report")
    FUNCTION = "route"
    CATEGORY = "IAMCCS/V2V"

    def route(
        self,
        cine_linx,
        source_images=None,
        source_audio=None,
        result_image=None,
        pose_video_mask_image=None,
        reference_image_mask_image=None,
        face_video=None,
        pose_video=None,
        background_video=None,
        character_mask=None,
    ):
        payload = _payload_from_cine_linx(cine_linx)
        mode = str(payload.get("backend_mode") or "ltx_simple")
        media_payload = dict(payload)
        if mode == "pose_transfer":
            pose_image_path = str(payload.get("pose_transfer_image_path") or payload.get("source_image_path") or "")
            pose_video_path = str(payload.get("pose_transfer_video_path") or payload.get("source_video_path") or "")
            if pose_image_path:
                media_payload["source_image_path"] = pose_image_path
            if pose_video_path:
                media_payload["source_video_path"] = pose_video_path
        loaded_images, loaded_audio, source_video_info, source_image, source_mask = _load_planner_media(media_payload, source_images, source_audio)
        result_image = result_image if result_image is not None else source_image
        face_video = face_video if face_video is not None else loaded_images
        pose_video = pose_video if pose_video is not None else loaded_images
        background_video = background_video if background_video is not None else loaded_images
        character_mask = character_mask if character_mask is not None else _white_mask_from_image(loaded_images)
        pose_video_mask_image = pose_video_mask_image if pose_video_mask_image is not None else _mask_image_from_image(loaded_images, 1.0)
        reference_image_mask_image = reference_image_mask_image if reference_image_mask_image is not None else _mask_image_from_image(source_image, 1.0)

        media_bus = {
            "type": V2V_MEDIA_BUS_TYPE,
            "source_video_images": loaded_images,
            "source_audio": loaded_audio,
            "source_video_info": source_video_info,
            "source_image": source_image,
            "source_mask": source_mask,
            "result_image": result_image,
            "face_video": face_video,
            "pose_video": pose_video,
            "background_video": background_video,
            "character_mask": character_mask,
            "pose_video_mask_image": pose_video_mask_image,
            "reference_image_mask_image": reference_image_mask_image,
        }
        control_bus = {
            "type": V2V_CONTROL_BUS_TYPE,
            "payload": payload,
            "backend_mode": mode,
            "backend_family": str(payload.get("backend_family") or "ltx"),
            "backend_profile": str(payload.get("backend_profile") or ""),
            "backend_variant": str(payload.get("backend_variant") or ""),
            "fps": float(_safe_float(payload.get("fps"), 24.0)),
            "generation_width": int(max(16, _safe_int(payload.get("generation_width"), 1280))),
            "generation_height": int(max(16, _safe_int(payload.get("generation_height"), 720))),
            "positive_prompt": str(payload.get("global_prompt", "") or ""),
            "negative_prompt": str(payload.get("negative_prompt", "") or ""),
            "output_prefix": str(payload.get("output_prefix", "IAMCCS/V2V_SHOTBOARD") or "IAMCCS/V2V_SHOTBOARD"),
            "backend_settings": payload.get("backend_settings") if isinstance(payload.get("backend_settings"), dict) else {},
        }
        preview_bus = {
            "type": V2V_PREVIEW_BUS_TYPE,
            "preview_channels": payload.get("preview_channels") if isinstance(payload.get("preview_channels"), dict) else {},
            "backend_template": payload.get("backend_template") if isinstance(payload.get("backend_template"), dict) else {},
            "result_path": str(payload.get("pose_transfer_result_path") or ""),
            "source_video_path": str(payload.get("source_video_path") or ""),
            "source_image_path": str(payload.get("source_image_path") or ""),
        }
        report = (
            f"V2V Backend Router | mode={control_bus['backend_mode']} | profile={control_bus['backend_profile']} | "
            f"size={control_bus['generation_width']}x{control_bus['generation_height']} | fps={control_bus['fps']:.3f}"
        )
        return (media_bus, control_bus, preview_bus, report)


class IAMCCS_V2VBusToScail:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "media_bus": (V2V_MEDIA_BUS_TYPE,),
                "control_bus": (V2V_CONTROL_BUS_TYPE,),
            },
            "optional": {
                "pose_video_mask_image": ("IMAGE",),
                "reference_image_mask_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("FLOAT", "INT", "INT", "STRING", "STRING", "STRING", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "AUDIO", "VHS_VIDEOINFO", "BOOLEAN", "STRING")
    RETURN_NAMES = ("fps", "generation_width", "generation_height", "positive_prompt", "negative_prompt", "output_prefix", "pose_video", "reference_image", "pose_video_mask_image", "reference_image_mask_image", "source_audio", "source_video_info", "replacement_mode", "report")
    FUNCTION = "convert"
    CATEGORY = "IAMCCS/V2V"

    def convert(self, media_bus, control_bus, pose_video_mask_image=None, reference_image_mask_image=None):
        media = _bus_media(media_bus)
        control = _bus_control(control_bus)
        payload = _bus_payload(control_bus)
        settings = control.get("backend_settings") if isinstance(control.get("backend_settings"), dict) else {}
        pose_video = _bus_value(media, "source_video_images")
        reference_image = _bus_value(media, "source_image")
        pose_video_mask_image = pose_video_mask_image if pose_video_mask_image is not None else _bus_value(media, "pose_video_mask_image", _mask_image_from_image(pose_video, 1.0))
        reference_image_mask_image = reference_image_mask_image if reference_image_mask_image is not None else _bus_value(media, "reference_image_mask_image", _mask_image_from_image(reference_image, 1.0))
        replacement_mode = str(settings.get("scail_identity_mode") or payload.get("scail_identity_mode") or "single_person") != "disabled"
        report = f"SCAIL bus adapter | profile={control.get('backend_profile')} | replacement={replacement_mode}"
        return (
            float(control.get("fps", 24.0)),
            int(control.get("generation_width", 512)),
            int(control.get("generation_height", 896)),
            str(control.get("positive_prompt", "")),
            str(control.get("negative_prompt", "")),
            str(control.get("output_prefix", "IAMCCS/SCAIL2_SHOTBOARD")),
            pose_video,
            reference_image,
            pose_video_mask_image,
            reference_image_mask_image,
            _bus_value(media, "source_audio"),
            _bus_value(media, "source_video_info", {}),
            bool(replacement_mode),
            report,
        )


class IAMCCS_V2VBusToWanAnimate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "media_bus": (V2V_MEDIA_BUS_TYPE,),
                "control_bus": (V2V_CONTROL_BUS_TYPE,),
            },
            "optional": {
                "face_video": ("IMAGE",),
                "pose_video": ("IMAGE",),
                "background_video": ("IMAGE",),
                "character_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("FLOAT", "INT", "INT", "STRING", "STRING", "STRING", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "MASK", "AUDIO", "VHS_VIDEOINFO", "STRING")
    RETURN_NAMES = ("fps", "generation_width", "generation_height", "positive_prompt", "negative_prompt", "output_prefix", "reference_image", "face_video", "pose_video", "background_video", "character_mask", "source_audio", "source_video_info", "report")
    FUNCTION = "convert"
    CATEGORY = "IAMCCS/V2V"

    def convert(self, media_bus, control_bus, face_video=None, pose_video=None, background_video=None, character_mask=None):
        media = _bus_media(media_bus)
        control = _bus_control(control_bus)
        source_frames = _bus_value(media, "source_video_images")
        face_video = face_video if face_video is not None else _bus_value(media, "face_video", source_frames)
        pose_video = pose_video if pose_video is not None else _bus_value(media, "pose_video", source_frames)
        background_video = background_video if background_video is not None else _bus_value(media, "background_video", source_frames)
        character_mask = character_mask if character_mask is not None else _bus_value(media, "character_mask", _white_mask_from_image(source_frames))
        report = f"WanAnimate bus adapter | profile={control.get('backend_profile')} | bg_lock={_bus_payload(control_bus).get('wan_background_lock')}"
        return (
            float(control.get("fps", 24.0)),
            int(control.get("generation_width", 832)),
            int(control.get("generation_height", 480)),
            str(control.get("positive_prompt", "")),
            str(control.get("negative_prompt", "")),
            str(control.get("output_prefix", "IAMCCS/WANANIMATE_SHOTBOARD")),
            _bus_value(media, "source_image"),
            face_video,
            pose_video,
            background_video,
            character_mask,
            _bus_value(media, "source_audio"),
            _bus_value(media, "source_video_info", {}),
            report,
        )


class IAMCCS_V2VBusToPoseTransfer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "media_bus": (V2V_MEDIA_BUS_TYPE,),
                "control_bus": (V2V_CONTROL_BUS_TYPE,),
            },
            "optional": {
                "result_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("FLOAT", "INT", "INT", "STRING", "STRING", "STRING", "IMAGE", "IMAGE", "IMAGE", "AUDIO", "VHS_VIDEOINFO", "STRING")
    RETURN_NAMES = ("fps", "generation_width", "generation_height", "positive_prompt", "negative_prompt", "output_prefix", "driver_video", "reference_image", "result_image", "source_audio", "source_video_info", "report")
    FUNCTION = "convert"
    CATEGORY = "IAMCCS/V2V"

    def convert(self, media_bus, control_bus, result_image=None):
        media = _bus_media(media_bus)
        control = _bus_control(control_bus)
        result_image = result_image if result_image is not None else _bus_value(media, "result_image", _bus_value(media, "source_image"))
        report = f"PoseTransfer bus adapter | profile={control.get('backend_profile')} | prefix={control.get('output_prefix')}"
        return (
            float(control.get("fps", 24.0)),
            int(control.get("generation_width", 768)),
            int(control.get("generation_height", 1024)),
            str(control.get("positive_prompt", "")),
            str(control.get("negative_prompt", "")),
            str(control.get("output_prefix", "IAMCCS/POSE_TRANSFER_SHOTBOARD")),
            _bus_value(media, "source_video_images"),
            _bus_value(media, "source_image"),
            result_image,
            _bus_value(media, "source_audio"),
            _bus_value(media, "source_video_info", {}),
            report,
        )


class IAMCCS_V2VBusToLTX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "media_bus": (V2V_MEDIA_BUS_TYPE,),
                "control_bus": (V2V_CONTROL_BUS_TYPE,),
            },
            "optional": {
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
            },
        }

    RETURN_TYPES = ("FLOAT", "INT", "INT", "STRING", "STRING", "STRING", "INT", "STRING", "INT", "VAE", "IMAGE", "AUDIO", "VHS_VIDEOINFO", "IMAGE", "IMAGE", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("fps", "generation_width", "generation_height", "global_prompt", "negative_prompt", "output_prefix", "continuation_loops", "planner_report", "segment_index_out", "audio_vae", "source_video_images", "source_audio", "source_video_info", "source_image", "current_source_images", "current_segment_audio", "current_audio_report", "report")
    FUNCTION = "convert"
    CATEGORY = "IAMCCS/V2V"

    def convert(self, media_bus, control_bus, segment_index=0):
        media = _bus_media(media_bus)
        payload = _bus_payload(control_bus)
        duration = float(_safe_float(payload.get("duration_seconds"), 10.0))
        fps = float(_safe_float(payload.get("fps"), 24.0))
        segment_duration = float(_safe_float(payload.get("segment_duration_s"), 10.0))
        planning_mode = str(payload.get("planning_mode", "explicit_preset_seconds") or "explicit_preset_seconds")
        segment_preset = str(payload.get("segment_preset", "5sec") or "5sec")
        overlap = int(max(0, _safe_int(payload.get("overlap_frames"), 9)))
        ltx_round = str(payload.get("ltx_round_mode", "up") or "up")
        plan = _plan_segment(duration, fps, segment_duration, planning_mode, segment_preset, overlap, ltx_round, _safe_int(segment_index, 0))
        source_range = _source_range_from_plan(
            int(plan["segment_index_out"]),
            int(plan["current_segment_raw_frames"]),
            int(plan["current_segment_unique_frames"]),
            int(plan["current_segment_start_frames"]),
        )
        loaded_images = _bus_value(media, "source_video_images")
        loaded_audio = _bus_value(media, "source_audio")
        source_image = _bus_value(media, "source_image")
        current_source_images, _current_source_count = _slice_images_internal(
            loaded_images,
            int(source_range["range_start_index"]),
            int(source_range["range_end_index"]),
            int(source_range["range_count"]),
        )
        audio_outputs = _audio_segment_internal(loaded_audio, fps, plan)
        current_segment_audio = _normalize_audio_for_ltx(audio_outputs[1])
        audio_vae = _load_ltx_audio_vae_kj_compat(
            str(payload.get("audio_vae_name", "ltx-2.3-22b-dev_audio_vae.safetensors") or "ltx-2.3-22b-dev_audio_vae.safetensors"),
            str(payload.get("audio_vae_device", "main_device") or "main_device"),
            str(payload.get("audio_vae_dtype", "bf16") or "bf16"),
        )
        return (
            fps,
            int(max(64, _safe_int(payload.get("generation_width"), 1280))),
            int(max(64, _safe_int(payload.get("generation_height"), 720))),
            str(payload.get("global_prompt", "") or ""),
            str(payload.get("negative_prompt", "") or ""),
            str(payload.get("output_prefix", "IAMCCS/LTX23_V2V_SHOTBOARD") or "IAMCCS/LTX23_V2V_SHOTBOARD"),
            int(plan["continuation_loops"]),
            str(plan["planner_report"]),
            int(plan["segment_index_out"]),
            audio_vae,
            loaded_images,
            loaded_audio,
            _bus_value(media, "source_video_info", {}),
            source_image,
            current_source_images,
            current_segment_audio,
            str(audio_outputs[8]),
            str(payload.get("report", "")),
        )


class IAMCCS_CineInfoV2VScail:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"cine_linx": (SUPERNODE_LINX_TYPE,)},
            "optional": {
                "source_images": ("IMAGE",),
                "source_audio": ("AUDIO",),
                "pose_video_mask_image": ("IMAGE",),
                "reference_image_mask_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = (
        "FLOAT",
        "INT",
        "INT",
        "STRING",
        "STRING",
        "STRING",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "AUDIO",
        "VHS_VIDEOINFO",
        "BOOLEAN",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "fps",
        "generation_width",
        "generation_height",
        "positive_prompt",
        "negative_prompt",
        "output_prefix",
        "pose_video",
        "reference_image",
        "pose_video_mask_image",
        "reference_image_mask_image",
        "source_audio",
        "source_video_info",
        "replacement_mode",
        "backend_profile",
        "report",
    )
    FUNCTION = "extract"
    CATEGORY = "IAMCCS/V2V"

    def extract(self, cine_linx, source_images=None, source_audio=None, pose_video_mask_image=None, reference_image_mask_image=None):
        payload = _payload_from_cine_linx(cine_linx)
        loaded_images, loaded_audio, source_video_info, source_image, _source_mask = _load_planner_media(payload, source_images, source_audio)
        if pose_video_mask_image is None:
            pose_video_mask_image = _mask_image_from_image(loaded_images, 1.0)
        if reference_image_mask_image is None:
            reference_image_mask_image = _mask_image_from_image(source_image, 1.0)
        settings = payload.get("backend_settings") if isinstance(payload.get("backend_settings"), dict) else {}
        replacement_mode = str(settings.get("scail_identity_mode") or payload.get("scail_identity_mode") or "single_person") != "disabled"
        report = (
            f"SCAIL CineInfo | profile={payload.get('backend_profile')} | "
            f"mode={payload.get('backend_mode')} | source={payload.get('source_video_path')} | ref={payload.get('source_image_path')}"
        )
        return (
            float(_safe_float(payload.get("fps"), 24.0)),
            int(max(32, _safe_int(payload.get("generation_width"), 512))),
            int(max(32, _safe_int(payload.get("generation_height"), 896))),
            str(payload.get("global_prompt", "") or ""),
            str(payload.get("negative_prompt", "") or ""),
            str(payload.get("output_prefix", "IAMCCS/SCAIL2_SHOTBOARD") or "IAMCCS/SCAIL2_SHOTBOARD"),
            loaded_images,
            source_image,
            pose_video_mask_image,
            reference_image_mask_image,
            loaded_audio,
            source_video_info,
            bool(replacement_mode),
            str(payload.get("backend_profile", "scail2_single_person") or "scail2_single_person"),
            report,
        )


class IAMCCS_CineInfoV2VWanAnimate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"cine_linx": (SUPERNODE_LINX_TYPE,)},
            "optional": {
                "source_images": ("IMAGE",),
                "source_audio": ("AUDIO",),
                "face_video": ("IMAGE",),
                "pose_video": ("IMAGE",),
                "background_video": ("IMAGE",),
                "character_mask": ("MASK",),
            },
        }

    RETURN_TYPES = (
        "FLOAT",
        "INT",
        "INT",
        "STRING",
        "STRING",
        "STRING",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "MASK",
        "AUDIO",
        "VHS_VIDEOINFO",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "fps",
        "generation_width",
        "generation_height",
        "positive_prompt",
        "negative_prompt",
        "output_prefix",
        "reference_image",
        "face_video",
        "pose_video",
        "background_video",
        "character_mask",
        "source_audio",
        "source_video_info",
        "backend_profile",
        "report",
    )
    FUNCTION = "extract"
    CATEGORY = "IAMCCS/V2V"

    def extract(self, cine_linx, source_images=None, source_audio=None, face_video=None, pose_video=None, background_video=None, character_mask=None):
        payload = _payload_from_cine_linx(cine_linx)
        loaded_images, loaded_audio, source_video_info, source_image, _source_mask = _load_planner_media(payload, source_images, source_audio)
        face_video = face_video if face_video is not None else loaded_images
        pose_video = pose_video if pose_video is not None else loaded_images
        background_video = background_video if background_video is not None else loaded_images
        character_mask = character_mask if character_mask is not None else _white_mask_from_image(loaded_images)
        report = (
            f"WanAnimate CineInfo | profile={payload.get('backend_profile')} | "
            f"bg_lock={payload.get('wan_background_lock')} | source={payload.get('source_video_path')} | ref={payload.get('source_image_path')}"
        )
        return (
            float(_safe_float(payload.get("fps"), 24.0)),
            int(max(16, _safe_int(payload.get("generation_width"), 832))),
            int(max(16, _safe_int(payload.get("generation_height"), 480))),
            str(payload.get("global_prompt", "") or ""),
            str(payload.get("negative_prompt", "") or ""),
            str(payload.get("output_prefix", "IAMCCS/WANANIMATE_SHOTBOARD") or "IAMCCS/WANANIMATE_SHOTBOARD"),
            source_image,
            face_video,
            pose_video,
            background_video,
            character_mask,
            loaded_audio,
            source_video_info,
            str(payload.get("backend_profile", "wananimate_extension") or "wananimate_extension"),
            report,
        )


class IAMCCS_CineInfoV2VPoseTransfer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"cine_linx": (SUPERNODE_LINX_TYPE,)},
            "optional": {
                "source_images": ("IMAGE",),
                "source_audio": ("AUDIO",),
                "result_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = (
        "FLOAT",
        "INT",
        "INT",
        "STRING",
        "STRING",
        "STRING",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "AUDIO",
        "VHS_VIDEOINFO",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "fps",
        "generation_width",
        "generation_height",
        "positive_prompt",
        "negative_prompt",
        "output_prefix",
        "driver_video",
        "reference_image",
        "result_image",
        "source_audio",
        "source_video_info",
        "backend_profile",
        "report",
    )
    FUNCTION = "extract"
    CATEGORY = "IAMCCS/V2V"

    def extract(self, cine_linx, source_images=None, source_audio=None, result_image=None):
        payload = _payload_from_cine_linx(cine_linx)
        settings = payload.get("backend_settings") if isinstance(payload.get("backend_settings"), dict) else {}
        pose_image_path = str(settings.get("pose_transfer_image_path") or payload.get("pose_transfer_image_path") or payload.get("source_image_path") or "")
        pose_video_path = str(settings.get("pose_transfer_video_path") or payload.get("pose_transfer_video_path") or payload.get("source_video_path") or "")
        pose_payload = dict(payload)
        if pose_image_path:
            pose_payload["source_image_path"] = pose_image_path
        if pose_video_path:
            pose_payload["source_video_path"] = pose_video_path
        loaded_images, loaded_audio, source_video_info, source_image, _source_mask = _load_planner_media(pose_payload, source_images, source_audio)
        result_image = result_image if result_image is not None else source_image
        report = (
            f"PoseTransfer CineInfo | profile={payload.get('backend_profile')} | "
            f"driver={pose_video_path or payload.get('source_video_path')} | ref={pose_image_path or payload.get('source_image_path')}"
        )
        return (
            float(_safe_float(payload.get("fps"), 24.0)),
            int(max(16, _safe_int(payload.get("generation_width"), 768))),
            int(max(16, _safe_int(payload.get("generation_height"), 1024))),
            str(payload.get("global_prompt", "") or ""),
            str(payload.get("negative_prompt", "") or ""),
            str(payload.get("output_prefix", "IAMCCS/POSE_TRANSFER_SHOTBOARD") or "IAMCCS/POSE_TRANSFER_SHOTBOARD"),
            loaded_images,
            source_image,
            result_image,
            loaded_audio,
            source_video_info,
            str(payload.get("backend_profile", "flux_klein_pose_transfer") or "flux_klein_pose_transfer"),
            report,
        )


class IAMCCS_V2VActiveBackendSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"cine_linx": (SUPERNODE_LINX_TYPE,)},
            "optional": {
                "control_bus": (V2V_CONTROL_BUS_TYPE,),
                "ltx": ("*", {"lazy": True}),
                "scail": ("*", {"lazy": True}),
                "scail_multi": ("*", {"lazy": True}),
                "wananimate": ("*", {"lazy": True}),
                "pose_transfer": ("*", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("selected", "report")
    FUNCTION = "switch"
    CATEGORY = "IAMCCS/V2V"

    @staticmethod
    def _selected_key(cine_linx, control_bus=None):
        payload = _payload_from_cine_linx(cine_linx)
        control = _bus_control(control_bus) if control_bus is not None else {}
        mode = str(control.get("backend_mode") or payload.get("backend_mode") or "ltx_simple").strip().lower()
        family = str(control.get("backend_family") or payload.get("backend_family") or "").strip().lower()
        profile = str(control.get("backend_profile") or payload.get("backend_profile") or "").strip().lower()
        combined = f"{mode} {family} {profile}"
        if "pose" in combined or "flux_klein" in combined:
            return "pose_transfer"
        if "wan" in combined:
            return "wananimate"
        if "scail_multi" in combined or "multi_person_identity" in combined:
            return "scail_multi"
        if "scail" in combined:
            return "scail"
        return "ltx"

    def check_lazy_status(
        self,
        cine_linx,
        control_bus=None,
        ltx=None,
        scail=None,
        scail_multi=None,
        wananimate=None,
        pose_transfer=None,
        **kwargs,
    ):
        selected = self._selected_key(cine_linx, control_bus)
        available = {
            "ltx": ltx,
            "scail": scail,
            "scail_multi": scail_multi,
            "wananimate": wananimate,
            "pose_transfer": pose_transfer,
        }
        if available.get(selected) is None:
            return [selected]
        return []

    def switch(
        self,
        cine_linx,
        control_bus=None,
        ltx=None,
        scail=None,
        scail_multi=None,
        wananimate=None,
        pose_transfer=None,
    ):
        selected = self._selected_key(cine_linx, control_bus)
        values = {
            "ltx": ltx,
            "scail": scail,
            "scail_multi": scail_multi,
            "wananimate": wananimate,
            "pose_transfer": pose_transfer,
        }
        value = values.get(selected)
        if value is None:
            raise ValueError(f"IAMCCS_V2VActiveBackendSwitch: selected backend '{selected}' is not connected")
        report = f"V2V Active Backend Switch | selected={selected}"
        return (value, report)


class IAMCCS_V2VStageSwitch:
    STAGE_KEYS = (
        "final",
        "draft",
        "sam_preview",
        "mask_preview",
        "pose_preview",
        "reference_preview",
        "result_image",
        "source",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "stage_kind": (["output_stage", "preview_stage"], {"default": "output_stage"}),
            },
            "optional": {
                "control_bus": (V2V_CONTROL_BUS_TYPE,),
                "final": ("*", {"lazy": True}),
                "draft": ("*", {"lazy": True}),
                "sam_preview": ("*", {"lazy": True}),
                "mask_preview": ("*", {"lazy": True}),
                "pose_preview": ("*", {"lazy": True}),
                "reference_preview": ("*", {"lazy": True}),
                "result_image": ("*", {"lazy": True}),
                "source": ("*", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("selected", "report")
    FUNCTION = "switch"
    CATEGORY = "IAMCCS/V2V"

    @staticmethod
    def _normalize_stage(stage):
        value = str(stage or "final").strip().lower()
        aliases = {
            "final_32fps_upscaled": "final",
            "final_upscaled": "final",
            "final_video": "final",
            "generated_16fps": "draft",
            "direct_seg0": "draft",
            "seg0": "draft",
            "sam": "sam_preview",
            "sam31": "sam_preview",
            "sam3.1": "sam_preview",
            "mask": "mask_preview",
            "pose": "pose_preview",
            "reference": "reference_preview",
            "result": "result_image",
        }
        value = aliases.get(value, value)
        return value if value in IAMCCS_V2VStageSwitch.STAGE_KEYS else "final"

    @classmethod
    def _selected_key(cls, cine_linx, control_bus=None, stage_kind="output_stage"):
        payload = _payload_from_cine_linx(cine_linx)
        control = _bus_control(control_bus) if control_bus is not None else {}
        control_payload = _bus_payload(control_bus) if control_bus is not None else {}
        settings = payload.get("backend_settings") if isinstance(payload.get("backend_settings"), dict) else {}
        key = "preview_stage" if str(stage_kind or "").strip().lower() == "preview_stage" else "output_stage"
        stage = (
            control.get(key)
            or control_payload.get(key)
            or payload.get(key)
            or settings.get(key)
        )
        if stage is None and key == "preview_stage":
            stage = control.get("output_stage") or control_payload.get("output_stage") or payload.get("output_stage")
        return cls._normalize_stage(stage)

    def check_lazy_status(
        self,
        cine_linx,
        stage_kind="output_stage",
        control_bus=None,
        final=None,
        draft=None,
        sam_preview=None,
        mask_preview=None,
        pose_preview=None,
        reference_preview=None,
        result_image=None,
        source=None,
        **kwargs,
    ):
        selected = self._selected_key(cine_linx, control_bus, stage_kind)
        available = {
            "final": final,
            "draft": draft,
            "sam_preview": sam_preview,
            "mask_preview": mask_preview,
            "pose_preview": pose_preview,
            "reference_preview": reference_preview,
            "result_image": result_image,
            "source": source,
        }
        if available.get(selected) is None:
            return [selected]
        return []

    def switch(
        self,
        cine_linx,
        stage_kind="output_stage",
        control_bus=None,
        final=None,
        draft=None,
        sam_preview=None,
        mask_preview=None,
        pose_preview=None,
        reference_preview=None,
        result_image=None,
        source=None,
    ):
        selected = self._selected_key(cine_linx, control_bus, stage_kind)
        values = {
            "final": final,
            "draft": draft,
            "sam_preview": sam_preview,
            "mask_preview": mask_preview,
            "pose_preview": pose_preview,
            "reference_preview": reference_preview,
            "result_image": result_image,
            "source": source,
        }
        value = values.get(selected)
        if value is None:
            raise ValueError(f"IAMCCS_V2VStageSwitch: selected stage '{selected}' is not connected")
        report = f"V2V Stage Switch | kind={stage_kind} | selected={selected}"
        return (value, report)


class IAMCCS_AudioNormalizeForLTX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"audio": ("AUDIO",)},
            "optional": {
                "debug_label": ("STRING", {"default": "ltx_audio"}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "report")
    FUNCTION = "normalize"
    CATEGORY = "IAMCCS/V2V"

    def normalize(self, audio, debug_label="ltx_audio"):
        normalized = _normalize_audio_for_ltx(audio)
        _debug_audio(f"AudioNormalizeForLTX.{debug_label}", normalized)
        waveform = normalized.get("waveform") if isinstance(normalized, dict) else None
        shape = tuple(int(dim) for dim in waveform.shape) if hasattr(waveform, "shape") else None
        sample_rate = normalized.get("sample_rate") if isinstance(normalized, dict) else None
        report = f"{debug_label}: shape={shape} sample_rate={sample_rate}"
        return (normalized, report)


NODE_CLASS_MAPPINGS = {
    "IAMCCS_ShotboardPlannerV2V": IAMCCS_ShotboardPlannerV2V,
    "IAMCCS_CineInfoV2V": IAMCCS_CineInfoV2V,
    "IAMCCS_CineInfoV2VBackendRouter": IAMCCS_CineInfoV2VBackendRouter,
    "IAMCCS_V2VBusToLTX": IAMCCS_V2VBusToLTX,
    "IAMCCS_V2VBusToScail": IAMCCS_V2VBusToScail,
    "IAMCCS_V2VBusToWanAnimate": IAMCCS_V2VBusToWanAnimate,
    "IAMCCS_V2VBusToPoseTransfer": IAMCCS_V2VBusToPoseTransfer,
    "IAMCCS_V2VActiveBackendSwitch": IAMCCS_V2VActiveBackendSwitch,
    "IAMCCS_V2VStageSwitch": IAMCCS_V2VStageSwitch,
    "IAMCCS_CineInfoV2VScail": IAMCCS_CineInfoV2VScail,
    "IAMCCS_CineInfoV2VWanAnimate": IAMCCS_CineInfoV2VWanAnimate,
    "IAMCCS_CineInfoV2VPoseTransfer": IAMCCS_CineInfoV2VPoseTransfer,
    "IAMCCS_AudioNormalizeForLTX": IAMCCS_AudioNormalizeForLTX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_ShotboardPlannerV2V": "IAMCCS Shotboard Planner V2V",
    "IAMCCS_CineInfoV2V": "IAMCCS CineInfo V2V",
    "IAMCCS_CineInfoV2VBackendRouter": "IAMCCS CineInfo V2V Backend Router",
    "IAMCCS_V2VBusToLTX": "IAMCCS V2V Bus To LTX",
    "IAMCCS_V2VBusToScail": "IAMCCS V2V Bus To SCAIL",
    "IAMCCS_V2VBusToWanAnimate": "IAMCCS V2V Bus To WanAnimate",
    "IAMCCS_V2VBusToPoseTransfer": "IAMCCS V2V Bus To Pose Transfer",
    "IAMCCS_V2VActiveBackendSwitch": "IAMCCS V2V Active Backend Switch",
    "IAMCCS_V2VStageSwitch": "IAMCCS V2V Stage Switch",
    "IAMCCS_CineInfoV2VScail": "IAMCCS CineInfo V2V SCAIL",
    "IAMCCS_CineInfoV2VWanAnimate": "IAMCCS CineInfo V2V WanAnimate",
    "IAMCCS_CineInfoV2VPoseTransfer": "IAMCCS CineInfo V2V Pose Transfer",
    "IAMCCS_AudioNormalizeForLTX": "IAMCCS Audio Normalize For LTX",
}

_install_ltx_audio_encode_guard()
