import copy
import gc
import json
import math
import os
import shutil

import torch

import comfy.model_management as model_management
import comfy.samplers
import comfy.utils
import nodes as comfy_nodes
from comfy_extras.nodes_hunyuan import LatentUpscaleModelLoader
from comfy_extras.nodes_custom_sampler import CFGGuider, KSamplerSelect, RandomNoise, SamplerCustomAdvanced
from comfy_extras.nodes_lt import (
    EmptyLTXVLatentVideo,
    LTXVConcatAVLatent,
    LTXVConditioning,
    LTXVCropGuides,
    LTXVImgToVideoInplace,
    LTXVPreprocess,
    LTXVScheduler,
    LTXVSeparateAVLatent,
    ModelSamplingLTXV,
)
from comfy_extras.nodes_lt_upsampler import LTXVLatentUpsampler
from comfy_extras.nodes_lt_audio import LTXVAudioVAEEncode
from comfy_extras.nodes_mask import SolidMask

from .iamccs_supernodes_linx import SUPERNODE_LINX_TYPE, build_stage_linx_payload, linx_output, linx_policy, linx_resource


_SAMPLER_NAMES = tuple(comfy.samplers.SAMPLER_NAMES)
_REFERENCE_MANUAL_SIGMAS = "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
_DEFAULT_GENERATED_FPS = 25.0
_REFERENCE_AUDIO_IMG2VID_FPS = 25.0
_REFERENCE_AUDIO_IMG2VID_WIDTH = 1280
_REFERENCE_AUDIO_IMG2VID_HEIGHT = 720
_STALE_AUDIO_IMG2VID_WIDTH = 768
_STALE_AUDIO_IMG2VID_HEIGHT = 512
_AUDIO_IMG2VID_CONTRACT_VERSION = "2026-05-03_workflow_kjresize_downround_v5"
_LEGACY_AUDIO_IMG2VID_CONTRACT_VERSION = "2026-05-03_legacy_exact_workflows_v1"
# Code by Carmine Cristallo Scalzi AI research (IAMCCS) patreon.com/IAMCCS
_REFERENCE_AUDIO_IMG2VID_DIVISIBLE_BY = 32
_REFERENCE_AUDIO_IMG2VID_NEGATIVE = "closed mouth, smiling without speaking, singing concert, dancing, exaggerated head movement, blurry mouth"
_REFERENCE_AUDIO_IMG2VID_TAIL_STRENGTH = 0.9
_REFERENCE_AUDIO_IMG2VID_TAIL_PREPROCESS_CRF = 28
_LEGACY_EXACT_TAIL_STRENGTH = 1.0
_LEGACY_EXACT_TAIL_PREPROCESS_CRF = 33
_PLANNER_AUDIO_MODES = (
    "melband_vocals_duration_math",
    "raw_audio_only",
)
_MEDIA_MODES = (
    "auto_from_generation_mode",
    "input_audio",
    "input_audio_img2vid",
    "input_audio_t2v",
    "generated_audio_img2vid",
    "generated_audio_t2v",
    "img2vid_pure",
    "t2v_pure",
)
_AUDIO_IMAGE_GENERATION_TYPE = "audio+image2video"
_AUDIO_IMAGE_LEGACY_GENERATION_TYPES = {
    "aud+img2video_simple",
    "aud+img2video_2_segments",
    "aud+img2video_infinite",
}
_GENERATION_TYPE_MODES = (
    _AUDIO_IMAGE_GENERATION_TYPE,
    "text+audio2video",
    "img2video",
    "text2video",
)
_RENDER_BACKEND_MODES = (
    "auto",
    "single_best",
    "ti2v_incremental_advanced",
    "legacy backend",
    "legacy_single",
    "legacy_two_segments",
    "legacy_loop",
    "two_segments_normal_vram",
    "three_segments_normal_vram",
    "loop_normal_vram",
    "loop_low_ram_disk",
)
_LEGACY_AUDIO_IMG2VID_BACKENDS = {"legacy_single", "legacy_two_segments", "legacy_loop"}
_MODULAR_DECODE_MODES = (
    "inherit_render_backend",
    "normal_tiled_iamccs",
    "normal_tiled_vhs",
    "low_ram",
    "low_ram_disk",
    "very_low_ram",
    "very_low_ram_disk",
    "high_vram",
    "custom_mode",
)
_VAE_DECODE_MODES = (
    "normal_tiled_iamccs",
    "low_ram",
    "very_low_ram",
    "high_vram",
)

try:
    import folder_paths  # type: ignore

    _LATENT_UPSCALE_MODEL_NAMES = tuple(folder_paths.get_filename_list("latent_upscale_models"))
    _MELBAND_MODEL_NAMES = tuple(
        name
        for name in folder_paths.get_filename_list("diffusion_models")
        if "melband" in str(name).lower() or "roformer" in str(name).lower()
    )
except Exception:
    _LATENT_UPSCALE_MODEL_NAMES = (
        "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
        "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    )
    _MELBAND_MODEL_NAMES = ("MelBandRoformer_fp32.safetensors",)

if not _MELBAND_MODEL_NAMES:
    _MELBAND_MODEL_NAMES = ("MelBandRoformer_fp32.safetensors",)

_MELBAND_MODEL_NAMES = tuple(sorted(
    (str(name) for name in _MELBAND_MODEL_NAMES if str(name)),
    key=lambda name: (
        0 if name == "MelBandRoformer_fp32.safetensors" else
        1 if ("melband" in name.lower() and "fp32" in name.lower()) else
        2,
        name.lower(),
    ),
))


def _resolve_melband_model_name(requested=None):
    requested_name = str(requested or "").strip()
    names = tuple(str(name) for name in (_MELBAND_MODEL_NAMES or ()) if str(name))
    preferred = None
    for name in names:
        if name == "MelBandRoformer_fp32.safetensors":
            preferred = name
            break
    if preferred is None:
        for name in names:
            lowered = name.lower()
            if "melband" in lowered and "fp32" in lowered:
                preferred = name
                break
    if requested_name and "fp16" not in requested_name.lower():
        return requested_name
    return preferred or requested_name or "MelBandRoformer_fp32.safetensors"


def _reference_ltx23_upscale_model_name():
    preferred = "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
    names = tuple(str(name) for name in (_LATENT_UPSCALE_MODEL_NAMES or ()) if str(name))
    for name in names:
        if name == preferred:
            return name
    for name in names:
        if "ltx-2.3-spatial-upscaler-x2-1.0" in name.lower():
            return name
    return names[0] if names else preferred


def _node_class(name):
    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(name)
    if cls is None:
        raise RuntimeError(f"Required node class is not loaded: {name}")
    return cls


def _resolve_output_path(path_value):
    out_dir = str(path_value or "").strip() or "iamccs_gc_auimg2vid"
    if os.path.isabs(out_dir):
        return out_dir
    try:
        from folder_paths import get_output_directory  # type: ignore

        base_out = get_output_directory()
    except Exception:
        base_out = os.getcwd()
    return os.path.join(base_out, out_dir)


def _ensure_clean_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


def _audio_duration_seconds(audio):
    if not isinstance(audio, dict):
        raise ValueError("audio must be a ComfyUI AUDIO dict")
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate", 0) or 0)
    if waveform is None or sample_rate <= 0:
        raise ValueError("audio is missing waveform/sample_rate")
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.tensor(waveform)
    total_samples = int(waveform.shape[-1])
    return float(total_samples) / float(sample_rate)


def _sanitize_audio_for_ffmpeg(audio):
    if audio is None:
        return None, "audio_sanitize=off(no_audio)"
    if not isinstance(audio, dict):
        return audio, "audio_sanitize=skipped(non_dict_audio)"
    waveform = audio.get("waveform")
    if waveform is None:
        return audio, "audio_sanitize=skipped(no_waveform)"
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.tensor(waveform)
    finite_mask = torch.isfinite(waveform)
    non_finite_count = int((~finite_mask).sum().item()) if waveform.numel() else 0
    sanitized_waveform = torch.nan_to_num(waveform.float(), nan=0.0, posinf=0.0, neginf=0.0)
    pre_clamp_peak = float(sanitized_waveform.abs().max().item()) if sanitized_waveform.numel() else 0.0
    clipped = pre_clamp_peak > 1.0
    if clipped:
        sanitized_waveform = sanitized_waveform.clamp(-1.0, 1.0)
    if non_finite_count == 0 and not clipped:
        return audio, "audio_sanitize=clean"
    sanitized_audio = dict(audio)
    sanitized_audio["waveform"] = sanitized_waveform.to(device=waveform.device)
    report = (
        f"audio_sanitize={'fixed' if (non_finite_count or clipped) else 'clean'} | "
        f"non_finite={non_finite_count} | peak_before_clamp={pre_clamp_peak:.6f} | clipped={'yes' if clipped else 'no'}"
    )
    return sanitized_audio, report


def _parse_payload(payload):
    if isinstance(payload, dict):
        return dict(payload)
    data = {}
    for raw_part in str(payload or "").split(";"):
        part = raw_part.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def _to_int(data, key, default):
    try:
        return int(float(data.get(key, default)))
    except Exception:
        return int(default)


def _to_float(data, key, default):
    try:
        value = data.get(key, default)
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default if default is None else float(default)


def _to_text(data, key, default):
    value = data.get(key, default)
    return str(value if value is not None else default)


def _to_bool(data, key, default):
    value = data.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value if value is not None else default).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return bool(default)


def _debug_verbose_enabled(debug_verbose=False, linx=None):
    if _to_bool({"value": debug_verbose}, "value", False):
        return True
    inherited = linx_output(linx, "debug_verbose", linx_resource(linx, "debug_verbose", False))
    return _to_bool({"value": inherited}, "value", False)


def _append_debug_report(report, debug_report):
    debug_text = str(debug_report or "").strip()
    if not debug_text:
        return str(report or "")
    return f"{report}\n{debug_text}"


def _same_as_default(value, default):
    if isinstance(default, float):
        try:
            return abs(float(value) - float(default)) <= 1e-9
        except Exception:
            return False
    return value == default


def _inherit_widget_value(widget_value, widget_default, linx, resource_key):
    inherited = linx_resource(linx, resource_key, None)
    if inherited is None:
        return widget_value
    if widget_value is None or _same_as_default(widget_value, widget_default):
        return inherited
    return widget_value


def _input_or_linx(explicit_value, linx, resource_key):
    if explicit_value is not None:
        return explicit_value
    return linx_resource(linx, resource_key, None)


def _require_runtime_value(value, name):
    if value is None:
        raise ValueError(f"{name} is required either as a local input or via linx inheritance")
    return value


def _first_line(text):
    return str(text or "").splitlines()[0].strip()


def _pipeline_debug_new(name, enabled=False):
    return {"name": str(name), "lines": [], "enabled": bool(enabled)}


def _pipeline_debug_preview(value, limit=120):
    if isinstance(value, torch.Tensor):
        shape = "x".join(str(x) for x in tuple(value.shape))
        return f"tensor(shape={shape}, dtype={value.dtype}, device={value.device})"
    if isinstance(value, dict):
        samples = value.get("samples")
        if isinstance(samples, torch.Tensor):
            shape = "x".join(str(x) for x in tuple(samples.shape))
            return f"latent(samples={shape}, dtype={samples.dtype}, device={samples.device})"
        if "waveform" in value:
            waveform = value.get("waveform")
            sample_rate = value.get("sample_rate", "?")
            return f"audio(waveform={_pipeline_debug_preview(waveform, limit)}, sample_rate={sample_rate})"
        return f"dict(keys={','.join(str(k) for k in list(value.keys())[:12])})"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}(len={len(value)})"
    if isinstance(value, (str, int, float, bool)) or value is None:
        text = str(value)
    else:
        text = value.__class__.__name__
    text = " ".join(text.split())
    return text if len(text) <= int(limit) else text[: int(limit) - 3] + "..."


def _pipeline_debug_step(trace, step_name, **fields):
    if not isinstance(trace, dict) or not bool(trace.get("enabled", False)):
        return ""
    lines = trace.setdefault("lines", [])
    idx = len(lines) + 1
    parts = [f"{key}={_pipeline_debug_preview(value)}" for key, value in fields.items()]
    line = f"{idx:02d}. {step_name}"
    if parts:
        line += " | " + " | ".join(parts)
    lines.append(line)
    print(f"[IAMCCS PipelineDebug:{trace.get('name', 'pipeline')}] {line}")
    return line


def _pipeline_debug_text(trace):
    if not isinstance(trace, dict) or not bool(trace.get("enabled", False)):
        return ""
    lines = trace.get("lines") if isinstance(trace, dict) else None
    if not lines:
        return ""
    return "Pipeline debug trace:\n" + "\n".join(str(line) for line in lines)


def _planner_chip(duration_seconds, planned, planning_mode, segment_preset):
    profile_label = str(planned[20]) if str(planning_mode) == "explicit_preset_seconds" else str(segment_preset)
    return (
        f"{profile_label} | duration {float(duration_seconds):.2f}s | total {int(planned[0])}f | "
        f"segments {int(planned[4])} | first {int(planned[2])}f | loop {int(planned[3])}f | overlap {int(planned[18])}f"
    )


def _duration_hint_from_linx(linx):
    if not isinstance(linx, dict):
        return None
    outputs = linx.get("outputs") or {}
    if isinstance(outputs, dict):
        try:
            value = outputs.get("estimated_duration_seconds")
            if value is not None:
                return float(value)
        except Exception:
            pass
    return None


def _duration_hint_from_payload(payload):
    data = _parse_payload(payload)
    if not data:
        return None
    return _to_float(data, "total_duration_seconds", None)


def _normalize_planner_mode(value):
    text = str(value or "manual_segment_seconds")
    if text == "auto_profile":
        return "explicit_preset_seconds"
    if text in {"manual_segment_seconds", "explicit_preset_seconds"}:
        return text
    return "manual_segment_seconds"


def _normalize_segment_preset(value):
    text = str(value or "10sec")
    if text == "videoclip":
        return "10sec"
    if text == "monologue":
        return "15sec"
    if text in {"5sec", "10sec", "15sec", "20sec"}:
        return text
    return "10sec"


def _segment_preset_from_seconds(seconds):
    value = float(seconds or 10.0)
    choices = ((5.0, "5sec"), (10.0, "10sec"), (15.0, "15sec"), (20.0, "20sec"))
    return min(choices, key=lambda item: abs(value - item[0]))[1]


_PLANNER_ROUTE_MODES = (
    "single generation (duration only)",
    "choose segment count (audio / segments)",
    "choose seconds per segment (auto count)",
)
_PLANNER_AUDIO_IMG2VID_BACKENDS = (
    "modern pipeline",
    "legacy exact pipeline",
)
_PLANNER_AUDIO_IMG2VID_MODES = (
    "single generation",
    "2 segments",
    "3 segments",
    "loop / 4+ segments",
)


def _normalize_audio_img2vid_backend_choice(value):
    text = str(value or "modern").strip().lower()
    if "legacy" in text:
        return "legacy"
    return "modern"


def _normalize_audio_img2vid_mode_choice(value, route_mode=None, segment_count=None):
    text = str(value or "").strip().lower()
    if "single" in text or text in {"1", "one"}:
        return "single"
    if text in {"2", "2_segments", "two_segments"} or "2 segment" in text:
        return "2_segments"
    if text in {"3", "3_segments", "three_segments"} or "3 segment" in text:
        return "3_segments"
    if "loop" in text or "auto" in text or "infinite" in text:
        return "loop"
    route_key = _normalize_planner_route_mode(route_mode)
    if route_key == "single_generation":
        return "single"
    count = max(1, int(segment_count or 2))
    if count == 1:
        return "single"
    if count == 2:
        return "2_segments"
    if count == 3:
        return "3_segments"
    return "loop"


def _audio_img2vid_backend_from_planner(backend_choice, mode_choice):
    family = _normalize_audio_img2vid_backend_choice(backend_choice)
    mode = _normalize_audio_img2vid_mode_choice(mode_choice)
    if family == "legacy":
        if mode == "single":
            return "legacy_single"
        if mode == "2_segments":
            return "legacy_two_segments"
        return "legacy_loop"
    if mode == "single":
        return "single_best"
    if mode == "2_segments":
        return "two_segments_normal_vram"
    if mode == "3_segments":
        return "three_segments_normal_vram"
    return "loop_normal_vram"


def _normalize_planner_route_mode(value):
    text = str(value or "").strip().lower()
    if "single" in text or text in {"1", "one", "single_generation"}:
        return "single_generation"
    if "seconds per segment" in text or "auto count" in text or text in {"fixed_segment_seconds", "choose_seconds_per_segment_auto_count"}:
        return "fixed_segment_seconds"
    return "fixed_segment_count"


def _decode_settings(decode_backend):
    if str(decode_backend) == "normal_high_disk":
        return {
            "tile": False,
            "tiling_mode": "auto",
            "tile_size": 1024,
            "overlap": 0,
            "cleanup_between_frames": False,
        }
    return {
        "tile": True,
        "tiling_mode": "auto",
        "tile_size": 512,
        "overlap": 64,
        "cleanup_between_frames": True,
    }


def _normalize_audio_preprocess_mode(mode):
    mapping = {
        "melband_vocals": "melband_vocals_duration_math",
        "melband_vocals_duration_math": "melband_vocals_duration_math",
        "raw_audio": "raw_audio_only",
        "raw_audio_only": "raw_audio_only",
    }
    return mapping.get(str(mode or "melband_vocals_duration_math"), "melband_vocals_duration_math")


def _normalize_backend_mode(mode):
    mapping = {
        "auto_from_plan": "auto",
        "auto": "auto",
        "single_workflow1_best": "single_best",
        "single_best": "single_best",
        "ti2v_incremental": "ti2v_incremental_advanced",
        "ti2v_advanced": "ti2v_incremental_advanced",
        "ti2v_incremental_advanced": "ti2v_incremental_advanced",
        "advanced_incremental": "ti2v_incremental_advanced",
        "legacy backend": "legacy_single",
        "legacy_backend": "legacy_single",
        "legacy_github": "legacy_single",
        "legacy_github_audimg": "legacy_single",
        "legacy_github_single": "legacy_single",
        "legacy single": "legacy_single",
        "legacy_single": "legacy_single",
        "legacy_github_two_segments": "legacy_two_segments",
        "legacy 2 segments": "legacy_two_segments",
        "legacy_two_segments": "legacy_two_segments",
        "legacy_github_loop": "legacy_loop",
        "legacy loop": "legacy_loop",
        "legacy_loop": "legacy_loop",
        "two_segments_normal_vram": "two_segments_normal_vram",
        "three_segments_normal_vram": "three_segments_normal_vram",
        "3_segments_normal_vram": "three_segments_normal_vram",
        "loop_normal_vram": "loop_normal_vram",
        "loop_low_ram_disk": "loop_low_ram_disk",
    }
    value = mapping.get(str(mode or "auto"), "auto")
    if value in _RENDER_BACKEND_MODES:
        return value
    return "auto"


def _normalize_modular_decode_mode(mode, backend_mode=None):
    mapping = {
        "inherit_render_backend": "inherit_from_backend",
        "inherit_from_render": "inherit_from_backend",
        "inherit_from_backend": "inherit_from_backend",
        "low_ram": "low_ram_disk",
        "low_ram_disk": "low_ram_disk",
        "very_low_ram": "very_low_ram_disk",
        "very_low_ram_disk": "very_low_ram_disk",
        "normal": "normal_tiled_vhs",
        "normal_tiled": "normal_tiled_vhs",
        "normal_tiled_iamccs": "normal_tiled_vhs",
        "normal_tiled_vhs": "normal_tiled_vhs",
        "normal_tiled_vhs_ready": "normal_tiled_vhs",
        "high": "high_vram_direct",
        "high_vram": "high_vram_direct",
        "high_vram_direct": "high_vram_direct",
        "custom_mode": "custom_mode",
    }
    normalized = mapping.get(str(mode or "inherit_from_backend"), "inherit_from_backend")
    if normalized != "inherit_from_backend":
        return normalized
    if _normalize_backend_mode(backend_mode) == "loop_low_ram_disk":
        return "low_ram_disk"
    return "normal_tiled_vhs"


def _modular_decode_to_vae_mode(modular_decode):
    mapping = {
        "inherit_from_backend": "normal_tiled_vhs",
        "low_ram": "low_ram_disk",
        "low_ram_disk": "low_ram_disk",
        "very_low_ram": "very_low_ram_disk",
        "very_low_ram_disk": "very_low_ram_disk",
        "normal": "normal_tiled_vhs",
        "normal_tiled": "normal_tiled_vhs",
        "normal_tiled_iamccs": "normal_tiled_vhs",
        "normal_tiled_vhs": "normal_tiled_vhs",
        "normal_tiled_vhs_ready": "normal_tiled_vhs",
        "high": "high_vram",
        "high_vram_direct": "high_vram",
        "custom_mode": "normal_tiled_vhs",
    }
    return mapping.get(str(modular_decode), "normal_tiled_vhs")


def _resolve_generation_type(generation_type, generation_mode, backend_mode, media_mode):
    requested = str(generation_type or _AUDIO_IMAGE_GENERATION_TYPE)
    mapping = {
        _AUDIO_IMAGE_GENERATION_TYPE: ("img2vid", "auto", "input_audio_img2vid"),
        "aud+img2video_simple": ("img2vid", "single_best", "input_audio_img2vid"),
        "aud+img2video_2_segments": ("img2vid", "two_segments_normal_vram", "input_audio_img2vid"),
        "aud+img2video_infinite": ("img2vid", "auto", "input_audio_img2vid"),
        "text+audio2video": ("t2v", "single_best", "input_audio_t2v"),
        "img2video": ("img2vid", "single_best", "img2vid_pure"),
        "text2video": ("t2v", "single_best", "t2v_pure"),
    }
    if requested not in mapping:
        requested = _AUDIO_IMAGE_GENERATION_TYPE
    resolved_generation, resolved_backend, resolved_media = mapping[requested]
    requested_backend = _normalize_backend_mode(backend_mode)
    if requested in {"img2video", "text2video"} and requested_backend in {"single_best", "ti2v_incremental_advanced"}:
        resolved_backend = requested_backend
    if requested == _AUDIO_IMAGE_GENERATION_TYPE and requested_backend in _LEGACY_AUDIO_IMG2VID_BACKENDS:
        resolved_backend = requested_backend
    return requested, resolved_generation, resolved_backend, resolved_media


def _resolve_media_mode(media_mode, generation_mode):
    requested_mode = str(media_mode or "auto_from_generation_mode")
    requested_generation = "t2v" if str(generation_mode or "img2vid") == "t2v" else "img2vid"
    alias_map = {
        "img2vid_generated_audio": "generated_audio_img2vid",
        "t2v_generated_audio": "generated_audio_t2v",
        "pure_img2vid": "img2vid_pure",
        "pure_t2v": "t2v_pure",
    }
    requested_mode = alias_map.get(requested_mode, requested_mode)

    if requested_mode == "input_audio_t2v":
        return {
            "mode": "input_audio_t2v",
            "generation_mode": "t2v",
            "uses_input_audio": True,
            "generates_audio": False,
            "exports_audio": True,
        }
    if requested_mode == "input_audio_img2vid":
        return {
            "mode": "input_audio_img2vid",
            "generation_mode": "img2vid",
            "uses_input_audio": True,
            "generates_audio": False,
            "exports_audio": True,
        }
    if requested_mode in {"generated_audio_t2v", "t2v_pure"}:
        return {
            "mode": requested_mode,
            "generation_mode": "t2v",
            "uses_input_audio": False,
            "generates_audio": True,
            "exports_audio": True,
        }
    if requested_mode in {"generated_audio_img2vid", "img2vid_pure"}:
        return {
            "mode": requested_mode,
            "generation_mode": "img2vid",
            "uses_input_audio": False,
            "generates_audio": True,
            "exports_audio": True,
        }

    return {
        "mode": "input_audio" if requested_mode == "input_audio" else f"input_audio_{requested_generation}",
        "generation_mode": requested_generation,
        "uses_input_audio": True,
        "generates_audio": False,
        "exports_audio": True,
    }


def _linx_models_only(existing_linx):
    if not isinstance(existing_linx, dict):
        return existing_linx
    source_resources = existing_linx.get("resources") or {}
    if not isinstance(source_resources, dict):
        source_resources = {}
    keep_keys = ("model", "clip", "vae", "audio_vae", "second_stage_model")
    resources = {
        key: source_resources.get(key)
        for key in keep_keys
        if source_resources.get(key) is not None
    }
    linx_payload = {
        "type": SUPERNODE_LINX_TYPE,
        "inheritance_scope": "models_only_no_audio",
    }
    if resources:
        linx_payload["resources"] = resources
        linx_payload["resource_keys"] = sorted(resources.keys())
        source_map = existing_linx.get("resource_sources") or {}
        if isinstance(source_map, dict):
            linx_payload["resource_sources"] = {
                key: source_map.get(key, "upstream_models_only")
                for key in resources.keys()
            }
    return linx_payload


def _pure_media_no_external_audio(media_mode, generation_type):
    return str(media_mode or "") in {"img2vid_pure", "t2v_pure"} or str(generation_type or "") in {"img2video", "text2video"}


def _ltx_compatible_frame_count(frame_count, round_mode="up"):
    target = max(1, int(math.ceil(float(frame_count or 1))))
    if target <= 1:
        return 1
    down = max(1, ((target - 1) // 8) * 8 + 1)
    up = max(1, int(math.ceil(max(0, target - 1) / 8.0)) * 8 + 1)
    mode = str(round_mode or "up").strip().lower()
    if mode == "down":
        return down
    if mode == "nearest":
        return down if abs(target - down) <= abs(up - target) else up
    return up


def _generated_duration_plan_payload(duration_seconds, fps):
    duration = max(0.1, float(duration_seconds or 10.0))
    fps_value = max(0.001, float(fps or _DEFAULT_GENERATED_FPS))
    requested_frames = max(1, int(math.ceil(duration * fps_value)))
    total_frames = _ltx_compatible_frame_count(requested_frames, "nearest")
    rounded_duration = float(total_frames) / fps_value
    return (
        f"pipeline_kind=au_img2vid_exec; total_duration_seconds={rounded_duration:.6f}; fps={fps_value:.6f}; "
        f"segment_seconds={rounded_duration:.6f}; requested_duration_seconds={duration:.6f}; "
        f"planning_mode=manual_segment_seconds; segment_preset=generated_reference; "
        f"content_profile=generated_reference; overlap_frames=9; ltx_round_mode=nearest; total_frames={int(total_frames)}; "
        f"target_frame_count={int(total_frames)}; target_frame_count_source=render_generated_duration; "
        f"requested_frames={int(requested_frames)}; generated_media_duration=true; generated_media_ltx_rounded=true"
    )


def _motion_guidance_strength(base_strength, motion_intensity):
    base_value = max(0.0, float(base_strength))
    intensity_value = max(0.01, float(motion_intensity or 1.0))
    return max(0.0, min(1.0, base_value / intensity_value))


def _resolve_backend_route(requested_backend_mode, planner_segment_count, modular_decode):
    backend_mode = _normalize_backend_mode(requested_backend_mode)
    planner_segment_count = max(1, int(planner_segment_count or 1))
    modular_decode = _normalize_modular_decode_mode(modular_decode, backend_mode)
    decode_is_disk = modular_decode in {"low_ram_disk", "very_low_ram_disk"}

    if backend_mode == "ti2v_incremental_advanced":
        return backend_mode, 1, True, False, modular_decode
    if backend_mode == "legacy_single":
        return backend_mode, 1, True, False, modular_decode
    if backend_mode == "legacy_two_segments":
        return backend_mode, 2, False, not decode_is_disk, modular_decode
    if backend_mode == "legacy_loop":
        return backend_mode, max(2, planner_segment_count), False, not decode_is_disk, modular_decode
    if backend_mode == "single_best":
        return backend_mode, 1, True, False, modular_decode
    if backend_mode == "two_segments_normal_vram":
        if decode_is_disk:
            return "loop_low_ram_disk", 2, False, False, modular_decode
        return backend_mode, 2, False, True, modular_decode
    if backend_mode == "three_segments_normal_vram":
        if decode_is_disk:
            return "loop_low_ram_disk", 3, False, False, modular_decode
        return backend_mode, 3, False, True, modular_decode
    if backend_mode == "loop_normal_vram":
        if decode_is_disk:
            return "loop_low_ram_disk", max(2, planner_segment_count), False, False, modular_decode
        return backend_mode, max(2, planner_segment_count), False, True, modular_decode
    if backend_mode == "loop_low_ram_disk":
        return backend_mode, max(2, planner_segment_count), False, False, modular_decode if decode_is_disk else "low_ram_disk"

    if planner_segment_count <= 1:
        return "single_best", 1, True, False, modular_decode
    if decode_is_disk:
        return "loop_low_ram_disk", planner_segment_count, False, False, modular_decode
    if planner_segment_count == 2:
        return "two_segments_normal_vram", 2, False, True, modular_decode
    if planner_segment_count == 3:
        return "three_segments_normal_vram", 3, False, True, modular_decode
    return "loop_normal_vram", planner_segment_count, False, True, modular_decode


def _render_status(duration_seconds, planner_data, modular_decode, continuity_mode, anti_drift_mode, second_stage_enabled, audio_concat_enabled):
    return (
        f"duration {float(duration_seconds):.2f}s | fps {float(planner_data['fps']):.2f} | total {int(_to_int(planner_data, 'total_frames', 0))}f | "
        f"segments {int(_to_int(planner_data, 'segment_count', 0))} | first {int(_to_int(planner_data, 'first_segment_raw_frames', 0))}f | "
        f"loop {int(_to_int(planner_data, 'continuation_raw_frames', 0))}f | overlap {int(_to_int(planner_data, 'recommended_overlap_frames', 0))}f | "
        f"vae {modular_decode} | stage2 {'on' if second_stage_enabled else 'off'} | audio_concat {'on' if audio_concat_enabled else 'off'} | continuity {continuity_mode} | anti_drift {anti_drift_mode}"
    )


def _downstream_stage_hints(stage_mode):
    mapping = {
        "finalize_only": ["finalize"],
        "upscale_ready": ["upscale", "finalize"],
        "detailer_ready": ["detailer", "finalize"],
        "upscale_then_detailer": ["upscale", "detailer", "finalize"],
    }
    return mapping.get(str(stage_mode), ["finalize"])


def _manual_sigmas(sigmas_text):
    values = []
    for item in str(sigmas_text or "").replace("\n", ",").split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        values = [float(item.strip()) for item in _REFERENCE_MANUAL_SIGMAS.split(",")]
    return torch.tensor(values, dtype=torch.float32)


def _node_output_tuple(value):
    current = value
    while isinstance(current, tuple) and len(current) == 1:
        current = current[0]
    if isinstance(current, tuple):
        return current
    if isinstance(current, list):
        return tuple(current)
    args = getattr(current, "args", None)
    if isinstance(args, (tuple, list)):
        return tuple(args)
    nested = getattr(current, "value", None)
    if nested is not None and nested is not current:
        return _node_output_tuple(nested)
    return (current,)


def _invoke_node(node, method_names, positional_variants=None, keyword_variants=None):
    positional_variants = positional_variants or []
    keyword_variants = keyword_variants or []
    last_error = None
    for method_name in method_names:
        fn = getattr(node, method_name, None)
        if fn is None:
            continue
        for kwargs in keyword_variants:
            try:
                return _node_output_tuple(fn(**kwargs))
            except TypeError as exc:
                last_error = exc
            except Exception as exc:
                last_error = exc
        for args in positional_variants:
            try:
                return _node_output_tuple(fn(*args))
            except TypeError as exc:
                last_error = exc
            except Exception as exc:
                last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"No callable method found on node {type(node).__name__}")


def _compute_audio_duration_seconds(audio):
    try:
        node = _node_class("Audio Duration (mtb)")()
        outputs = _invoke_node(
            node,
            ("execute", "get_duration", "duration"),
            positional_variants=[(audio,)],
            keyword_variants=[{"audio": audio}],
        )
        duration_ms = int(outputs[0])
        return float(duration_ms) * 0.001, "mtb"
    except Exception:
        return _audio_duration_seconds(audio), "waveform"


def _compute_audio_duration_frames_with_tail(audio, fps):
    fps_value = max(0.001, float(fps))
    try:
        node = _node_class("Audio Duration (mtb)")()
        outputs = _invoke_node(
            node,
            ("execute", "get_duration", "duration"),
            positional_variants=[(audio,)],
            keyword_variants=[{"audio": audio}],
        )
        duration_ms = int(outputs[0])
        return max(1, int((float(duration_ms) * 0.001 * fps_value) + 1.0)), "mtb_plus_tail"
    except Exception:
        seconds = _audio_duration_seconds(audio)
        return max(1, int((float(seconds) * fps_value) + 1.0)), "waveform_plus_tail"


def _extract_melband_vocals(audio, melband_model_name):
    if audio is None:
        raise ValueError("MelBand vocals extraction requires an audio input")
    model_name = _resolve_melband_model_name(melband_model_name)
    loader = _node_class("MelBandRoFormerModelLoader")()
    melband_model = _invoke_node(
        loader,
        ("execute", "load", "load_model", "loadmodel"),
        positional_variants=[(model_name,)],
        keyword_variants=[{"model_name": model_name}, {"model": model_name}],
    )[0]
    sampler = _node_class("MelBandRoFormerSampler")()
    vocals = _invoke_node(
        sampler,
        ("execute", "sample", "process"),
        positional_variants=[(melband_model, audio)],
        keyword_variants=[{"model": melband_model, "audio": audio}],
    )[0]
    return vocals, model_name


def _prepare_planner_audio(audio, audio_preprocess_mode, melband_model_name):
    mode = _normalize_audio_preprocess_mode(audio_preprocess_mode)
    model_name = _resolve_melband_model_name(melband_model_name)
    model_resolution_report = ""
    raw_seconds, raw_source = _compute_audio_duration_seconds(audio)
    result = {
        "raw_audio": audio,
        "conditioning_audio_single": audio,
        "conditioning_audio_segmented": audio,
        "ltx_conditioning_audio": audio,
        "ltx_conditioning_source": "raw_audio",
        "duration_audio": audio,
        "duration_seconds": raw_seconds,
        "duration_source": raw_source,
        "duration_frames_with_tail": None,
        "duration_frames_source": "unknown",
        "conditioning_duration_audio": audio,
        "conditioning_duration_seconds": raw_seconds,
        "conditioning_duration_source": raw_source,
        "preprocess_report": (
            f"audio_preprocess=raw_audio_only | global_duration_source={raw_source} | "
            f"conditioning_duration_source={raw_source}"
        ),
        "melband_enabled": False,
    }
    if mode != "melband_vocals_duration_math":
        return result

    try:
        vocals, model_name = _extract_melband_vocals(audio, model_name)
        duration_seconds, duration_source = _compute_audio_duration_seconds(vocals)
        result.update({
            "conditioning_audio_single": vocals,
            "conditioning_duration_audio": vocals,
            "conditioning_duration_seconds": duration_seconds,
            "conditioning_duration_source": duration_source,
            "ltx_conditioning_audio": audio,
            "ltx_conditioning_source": "raw_audio_with_melband_duration_math",
            "preprocess_report": (
                f"audio_preprocess=melband_vocals_duration_math | model={model_name}{model_resolution_report} | "
                f"global_duration_source={raw_source} | conditioning_duration_source={duration_source} | "
                f"global_duration_seconds={raw_seconds:.6f} | conditioning_duration_seconds={duration_seconds:.6f} | "
                "ltx_audio_vae_source=raw_audio"
            ),
            "melband_enabled": True,
        })
    except Exception as exc:
        result["preprocess_report"] = (
            f"audio_preprocess=raw_audio_fallback | reason={exc} | global_duration_source={raw_source} | "
            f"conditioning_duration_source={raw_source}"
        )
    return result


def _scheduler_sigmas(model, scheduler_name="simple", steps=8, denoise=1.0):
    try:
        node = _node_class("BasicScheduler")()
        outputs = _invoke_node(
            node,
            ("get_sigmas", "execute"),
            positional_variants=[
                (model, scheduler_name, int(steps), float(denoise)),
                (scheduler_name, int(steps), float(denoise), model),
            ],
            keyword_variants=[
                {"model": model, "scheduler": scheduler_name, "steps": int(steps), "denoise": float(denoise)},
                {"scheduler": scheduler_name, "steps": int(steps), "denoise": float(denoise), "model": model},
            ],
        )
        return outputs[0]
    except Exception:
        return _manual_sigmas(_REFERENCE_MANUAL_SIGMAS)


def _ltx_scheduler_sigmas(steps, max_shift, base_shift, sigma_terminal, latent):
    return LTXVScheduler.execute(
        max(1, int(steps or 20)),
        float(max_shift),
        float(base_shift),
        True,
        float(sigma_terminal),
        latent,
    )[0]


def _is_single_best_reference_audio_img2vid(generation_type):
    return str(generation_type or "") in {_AUDIO_IMAGE_GENERATION_TYPE, *_AUDIO_IMAGE_LEGACY_GENERATION_TYPES}


def _is_text_audio2video(generation_type):
    return str(generation_type or "") == "text+audio2video"


def _effective_sampler_name(sampler_name, generation_mode, media_mode, generation_type=None):
    # Regression guard: frontend sampler widgets are the source of truth.
    # Generation-type defaults belong in the UI; the backend must not silently
    # remap t2v/i2v sampler choices after the user changes them.
    return str(sampler_name or "euler")


def _effective_cfg(cfg, generation_mode, media_mode):
    return float(cfg)


def _effective_ltx_steps(steps, generation_mode, media_mode):
    return max(1, int(steps or 8))


def _main_sigmas(
    uses_input_audio,
    steps,
    max_shift,
    base_shift,
    sigma_terminal,
    latent,
    generation_mode,
    media_mode,
    manual_sigmas,
    generation_type=None,
    scheduler_model=None,
):
    if uses_input_audio:
        if _is_single_best_reference_audio_img2vid(generation_type) and scheduler_model is not None:
            effective_steps = max(1, int(steps or 8))
            sigmas = _scheduler_sigmas(scheduler_model, "simple", effective_steps, 1.0)
            return sigmas, f"basic_scheduler(simple,{effective_steps} steps,denoise=1.0)"
        sigmas = _manual_sigmas(manual_sigmas)
        return sigmas, f"manual({int(sigmas.numel())} values)"
    effective_steps = _effective_ltx_steps(steps, generation_mode, media_mode)
    sigmas = _ltx_scheduler_sigmas(effective_steps, max_shift, base_shift, sigma_terminal, latent)
    return sigmas, f"ltx_scheduler({int(sigmas.numel()) - 1} steps,terminal={float(sigma_terminal):.3f})"


def _decode_images_in_memory(
    video_latent,
    vae,
    tile_size=512,
    overlap=64,
    temporal_size=256,
    temporal_overlap=32,
    cleanup_before_decode=False,
    tiling_mode="manual",
):
    decode_node = _node_class("IAMCCS_VAEDecodeTiledSafe")()
    return decode_node.decode(
        video_latent,
        vae,
        True,
        str(tiling_mode or "manual"),
        int(tile_size),
        int(overlap),
        int(temporal_size),
        int(temporal_overlap),
        bool(cleanup_before_decode),
        False,
        0,
    )[0]


def _decode_images_for_vae_mode(
    video_latent,
    vae,
    resolved_decode_mode,
    tile_size=512,
    overlap=64,
    temporal_size=256,
    temporal_overlap=32,
    cleanup_before_decode=False,
):
    mode = str(resolved_decode_mode or "normal_tiled_vhs")
    if mode in {"normal_tiled_vhs", "custom_mode"}:
        images = _decode_images_in_memory(
            video_latent,
            vae,
            int(tile_size),
            int(overlap),
            int(temporal_size),
            int(temporal_overlap),
            bool(cleanup_before_decode),
            "manual",
        )
        return images, (
            "decode_node=IAMCCS_VAEDecodeTiledSafe | "
            f"mode=normal_tiled_vhs | tiling_mode=manual | tile_size={int(tile_size)} | overlap={int(overlap)} | "
            f"temporal_size={int(temporal_size)} | temporal_overlap={int(temporal_overlap)} | "
            f"cleanup_before_decode={'on' if bool(cleanup_before_decode) else 'off'}"
        )
    if mode == "high_vram":
        if bool(cleanup_before_decode):
            _soft_cleanup()
        images = comfy_nodes.VAEDecode().decode(vae, video_latent)[0]
        return images, f"decode_node=VAEDecode | mode=high_vram | cleanup_before_decode={'on' if bool(cleanup_before_decode) else 'off'}"
    raise ValueError(f"Unsupported in-memory VAE decode mode: {mode}")


def _load_images_from_dir_for_output(frames_dir):
    import numpy as np
    from PIL import Image

    path = str(frames_dir or "").strip()
    if not path or not os.path.isdir(path):
        return torch.zeros((1, 1, 1, 3), dtype=torch.float32)

    files = []
    for name in sorted(os.listdir(path)):
        lower = name.lower()
        if lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
            files.append(os.path.join(path, name))
    if not files:
        return torch.zeros((1, 1, 1, 3), dtype=torch.float32)

    images = []
    for file_path in files:
        with Image.open(file_path) as image:
            rgb = image.convert("RGB")
            array = np.asarray(rgb).astype("float32") / 255.0
            images.append(torch.from_numpy(array))
    return torch.stack(images, dim=0)


def _coerce_frame_count(value, default=0):
    try:
        count = int(float(value))
    except Exception:
        return int(default)
    return count if count > 0 else int(default)


def _resolve_linx_frame_target(linx):
    if bool(linx_output(linx, "disable_vae_frame_align", False) or linx_resource(linx, "disable_vae_frame_align", False)):
        return 0, "disabled_by_render"
    for key in (
        "vae_target_frame_count",
        "render_target_frame_count",
        "target_frame_count",
        "render_frame_count",
        "audio_duration_frames_with_tail",
        "total_frames",
    ):
        count = _coerce_frame_count(linx_output(linx, key, None), 0)
        if count > 0:
            return count, f"linx_output:{key}"
        count = _coerce_frame_count(linx_resource(linx, key, None), 0)
        if count > 0:
            return count, f"linx_resource:{key}"
    payload = str(linx_resource(linx, "planner_payload", "") or linx_output(linx, "plan_payload", "") or "")
    payload_data = _parse_payload(payload)
    for key in ("target_frame_count", "audio_duration_frames_with_tail", "total_frames"):
        count = _to_int(payload_data, key, 0)
        if count > 0:
            return count, f"planner_payload:{key}"
    return 0, "none"


def _align_images_to_frame_count(images, target_frame_count):
    target = _coerce_frame_count(target_frame_count, 0)
    if images is None or target <= 0:
        return images, "frame_align=off"
    current = int(images.shape[0])
    if current == target:
        return images, f"frame_align=matched target={target}"
    if current <= 0:
        return images, f"frame_align=skipped_empty target={target}"
    if current < target:
        pad = images[-1:].repeat(target - current, 1, 1, 1)
        return torch.cat([images, pad], dim=0), f"frame_align=padded {current}->{target}"
    return images[:target], f"frame_align=trimmed {current}->{target}"


def _resize_image_to(image, width, height):
    if image is None:
        raise ValueError("image is required for resize")
    resized = comfy.utils.common_upscale(image.movedim(-1, 1), int(width), int(height), "lanczos", "center")
    return resized.movedim(1, -1)


def _round_up_to_multiple(value, divisor):
    value = int(value)
    divisor = int(divisor)
    if divisor <= 1:
        return max(1, value)
    return max(divisor, ((max(1, value) + divisor - 1) // divisor) * divisor)


def _round_down_to_multiple(value, divisor):
    value = int(value)
    divisor = int(divisor)
    if divisor <= 1:
        return max(1, value)
    rounded = value - (value % divisor)
    return max(divisor, rounded)


def _resize_image_like_reference_audio_img2vid(image, width, height):
    if image is None:
        raise ValueError("image is required for reference audio+image resize")
    requested_width = int(width)
    requested_height = int(height)
    try:
        resize_cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("ImageResizeKJv2")
        if resize_cls is not None:
            resized, resized_width, resized_height, _ = resize_cls().resize(
                image,
                requested_width,
                requested_height,
                "crop",
                "lanczos",
                _REFERENCE_AUDIO_IMG2VID_DIVISIBLE_BY,
                "0, 0, 0",
                "top",
                None,
                device="cpu",
            )
            return (
                resized,
                int(resized_width),
                int(resized_height),
                f"ImageResizeKJv2(requested={requested_width}x{requested_height},actual={int(resized_width)}x{int(resized_height)},method=lanczos,mode=crop,crop_position=top,divisible_by={_REFERENCE_AUDIO_IMG2VID_DIVISIBLE_BY})",
            )
    except Exception as exc:
        fallback_reason = f"fallback_after={type(exc).__name__}:{exc}"
    else:
        fallback_reason = "fallback_after=ImageResizeKJv2_missing"

    target_width = _round_down_to_multiple(requested_width, _REFERENCE_AUDIO_IMG2VID_DIVISIBLE_BY)
    target_height = _round_down_to_multiple(requested_height, _REFERENCE_AUDIO_IMG2VID_DIVISIBLE_BY)
    old_height = int(image.shape[-3])
    old_width = int(image.shape[-2])
    old_aspect = float(old_width) / float(max(1, old_height))
    new_aspect = float(target_width) / float(max(1, target_height))
    if old_aspect > new_aspect:
        crop_width = max(1, round(old_height * new_aspect))
        crop_height = old_height
        crop_x = max(0, (old_width - crop_width) // 2)
        crop_y = 0
    else:
        crop_width = old_width
        crop_height = max(1, round(old_width / new_aspect))
        crop_x = 0
        crop_y = 0
    cropped = image.narrow(-2, crop_x, crop_width).narrow(-3, crop_y, crop_height)
    resized = comfy.utils.common_upscale(cropped.movedim(-1, 1), target_width, target_height, "lanczos", "disabled")
    return (
        resized.movedim(1, -1),
        int(target_width),
        int(target_height),
        f"manual_reference_resize(requested={requested_width}x{requested_height},actual={target_width}x{target_height},method=lanczos,mode=crop,crop_position=top,divisible_by={_REFERENCE_AUDIO_IMG2VID_DIVISIBLE_BY},{fallback_reason})",
    )


def _normalize_reference_audio_img2vid_requested_resolution(width, height):
    requested_width = int(width)
    requested_height = int(height)
    # Code by Carmine Cristallo Scalzi AI research (IAMCCS)
    # patreon.com/IAMCCS
    #
    # Regression guard for old SuperNode saves: 1280x736 was the generic LTX
    # default, but the working A+I2V reference workflow feeds ImageResizeKJv2
    # with 1280x720. KJ then applies its own divisible_by=32 down-rounding.
    if requested_width == _REFERENCE_AUDIO_IMG2VID_WIDTH and requested_height == 736:
        return _REFERENCE_AUDIO_IMG2VID_WIDTH, _REFERENCE_AUDIO_IMG2VID_HEIGHT, "stale_supernode_1280x736_normalized_to_reference_1280x720"
    if requested_width == _STALE_AUDIO_IMG2VID_WIDTH and requested_height == _STALE_AUDIO_IMG2VID_HEIGHT:
        return _REFERENCE_AUDIO_IMG2VID_WIDTH, _REFERENCE_AUDIO_IMG2VID_HEIGHT, "stale_supernode_768x512_normalized_to_reference_1280x720"
    return requested_width, requested_height, ""


def _pixel_dims_from_latent(latent, vae):
    samples = latent["samples"]
    _, _, _, latent_h, latent_w = samples.shape
    _, width_scale_factor, height_scale_factor = vae.downscale_index_formula
    return int(latent_w * width_scale_factor), int(latent_h * height_scale_factor)


def _images_to_dir(images, output_dir, prefix, image_format, jpg_quality, clear_existing):
    try:
        from PIL import Image  # type: ignore
    except Exception as e:
        raise RuntimeError(f"PIL (Pillow) is required to save decoded frames: {e!r}")

    out_dir = _resolve_output_path(output_dir)
    os.makedirs(out_dir, exist_ok=True)
    if bool(clear_existing):
        pfx = f"{prefix}_"
        for name in os.listdir(out_dir):
            if name.startswith(pfx):
                try:
                    os.remove(os.path.join(out_dir, name))
                except Exception:
                    pass

    image_format = str(image_format).lower()
    ext = ".jpg" if image_format == "jpg" else ".png"
    batch = images.shape[0]
    for index in range(batch):
        frame = images[index].detach().cpu().clamp(0.0, 1.0)
        array = (frame.numpy() * 255.0).round().astype("uint8")
        path = os.path.join(out_dir, f"{prefix}_{index:06d}{ext}")
        img = Image.fromarray(array)
        if image_format == "jpg":
            img.save(path, format="JPEG", quality=int(jpg_quality))
        else:
            img.save(path, format="PNG")
    return out_dir, int(batch)


def _frame_files_in_dir(path):
    if not path or not os.path.isdir(path):
        return []
    names = []
    for name in os.listdir(path):
        lower = name.lower()
        if lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
            names.append(os.path.join(path, name))
    names.sort()
    return names


def _align_frame_dir_to_frame_count(path, target_frame_count, prefix="frame"):
    target = _coerce_frame_count(target_frame_count, 0)
    files = _frame_files_in_dir(path)
    current = len(files)
    if target <= 0:
        return current, "frame_dir_align=off"
    if current == target:
        return current, f"frame_dir_align=matched target={target}"
    if current <= 0:
        return current, f"frame_dir_align=skipped_empty target={target}"
    if current > target:
        for file_path in files[target:]:
            try:
                os.remove(file_path)
            except Exception:
                pass
        return target, f"frame_dir_align=trimmed {current}->{target}"

    last_path = files[-1]
    ext = os.path.splitext(last_path)[1] or ".png"
    for index in range(current, target):
        dst = os.path.join(str(path), f"{prefix}_{index:06d}{ext}")
        shutil.copy2(last_path, dst)
    return target, f"frame_dir_align=padded {current}->{target}"


def _json_safe_metadata(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_metadata(item) for item in value]
    return str(value)


def _prompt_excerpt(text, limit=180):
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= int(limit):
        return cleaned
    return cleaned[: max(0, int(limit) - 3)] + "..."


def _save_video_metadata_sidecar(video_path, metadata):
    if not str(video_path or "").strip() or not metadata:
        return ""
    sidecar_path = os.path.splitext(str(video_path))[0] + ".metadata.json"
    with open(sidecar_path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe_metadata(metadata), handle, indent=2, ensure_ascii=True)
    return sidecar_path


def _format_segment_overlay_text(template, segment_index, segment_count, raw_frames, unique_frames, effective_frames, total_duration_seconds):
    text = str(template or "").strip() or "seg {segment_number}/{segment_count}"
    try:
        return text.format(
            segment_index=int(segment_index),
            segment_number=int(segment_index) + 1,
            segment_count=int(segment_count),
            raw_frames=int(raw_frames),
            unique_frames=int(unique_frames),
            effective_frames=int(effective_frames),
            total_duration_seconds=float(total_duration_seconds),
        )
    except Exception:
        return text


def _overlay_text_on_frame_dir(path, text):
    files = _frame_files_in_dir(path)
    if not files or not str(text or "").strip():
        return 0
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Pillow is required for segment overlay text: {e!r}")

    overlay_text = str(text)
    updated = 0
    for file_path in files:
        with Image.open(file_path).convert("RGB") as image:
            draw = ImageDraw.Draw(image, "RGBA")
            width, height = image.size
            font = ImageFont.load_default()
            bbox = draw.multiline_textbbox((0, 0), overlay_text, font=font, spacing=4)
            text_w = int(bbox[2] - bbox[0])
            text_h = int(bbox[3] - bbox[1])
            pad = max(10, int(round(min(width, height) * 0.015)))
            x = pad
            y = pad
            box = (x - pad // 2, y - pad // 2, x + text_w + pad, y + text_h + pad)
            draw.rounded_rectangle(box, radius=10, fill=(0, 0, 0, 160))
            draw.multiline_text((x, y), overlay_text, font=font, fill=(255, 240, 120, 255), spacing=4)
            image.save(file_path)
            updated += 1
    return updated


def _load_guidance_image_from_dir(path, fallback_image=None, pick_mode="latest"):
    files = _frame_files_in_dir(path)
    if not files:
        return fallback_image
    target_path = files[-1] if str(pick_mode) == "latest" else files[0]
    try:
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return fallback_image

    image = Image.open(target_path).convert("RGB")
    array = np.asarray(image).astype("float32") / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _resolve_stage2_model(primary_model, second_stage_model, second_stage_payload):
    data = _parse_payload(second_stage_payload)
    policy = _to_text(data, "stage2_model_policy", "stage2_model_if_connected")
    if second_stage_model is None:
        return primary_model
    if policy in {"stage2_model_if_connected", "replace_stage1_if_connected", "prefer_stage2_else_primary"}:
        return second_stage_model
    return primary_model


def _stage2_payload_from_exec_widgets(
    second_stage_mode,
    stage2_model_policy,
    second_stage_upscale_model,
    second_stage_reinject_strength,
    second_stage_cfg,
    second_stage_manual_sigmas,
):
    scale_mode = "x2_latent_upscale_beta" if str(second_stage_mode) == "latent_upscale_refine_x2_beta" else "same_resolution_refine"
    return (
        f"second_stage_mode={second_stage_mode}; "
        f"stage2_model_policy={stage2_model_policy}; "
        f"second_stage_upscale_model={second_stage_upscale_model}; "
        f"second_stage_scale_mode={scale_mode}; "
        f"second_stage_reinject_strength={float(second_stage_reinject_strength)}; "
        f"second_stage_cfg={float(second_stage_cfg)}; "
        f"second_stage_manual_sigmas={second_stage_manual_sigmas}; "
        f"second_stage_steps={max(0, len([x for x in str(second_stage_manual_sigmas).replace(chr(10), ',').split(',') if x.strip()]) - 1)}"
    )


def _apply_reference_stage2_defaults(
    generation_mode,
    media_mode,
    second_stage_mode,
    second_stage_scale_mode,
    second_stage_reinject_strength,
    second_stage_cfg,
    second_stage_manual_sigmas,
    second_stage_step_count,
):
    return (
        second_stage_mode,
        second_stage_scale_mode,
        second_stage_reinject_strength,
        second_stage_cfg,
        second_stage_manual_sigmas,
        second_stage_step_count,
        "",
    )


def _continuity_mode_from_payload(continuity_payload, fallback_mode, fallback_interval, fallback_strength):
    data = _parse_payload(continuity_payload)
    if not data:
        return {
            "mode": str(fallback_mode),
            "interval": int(fallback_interval),
            "strength": float(fallback_strength),
        }

    payload_policy = _to_text(data, "anchor_policy", fallback_mode)
    payload_source = _to_text(data, "anchor_source_mode", "")
    resolved_mode = str(fallback_mode)
    if payload_source == "prev_tail_plus_anchor":
        resolved_mode = "periodic_tail_then_source_refresh"
    elif payload_policy == "body_anchor_refresh":
        resolved_mode = "tail_then_source_refresh"
    elif payload_policy == "periodic_anchor_refresh" or payload_source == "periodic_keyframe_refresh":
        resolved_mode = "periodic_source_refresh"
    elif payload_policy in {"identity_first", "start_anchor_only"}:
        resolved_mode = "always_source_refresh"
    elif payload_policy == "off":
        resolved_mode = "off"

    return {
        "mode": resolved_mode,
        "interval": _to_int(data, "refresh_interval", fallback_interval),
        "strength": _to_float(data, "anchor_guidance_strength", fallback_strength),
    }


def _use_source_anchor(segment_index, continuity_mode, refresh_interval):
    mode = str(continuity_mode)
    if segment_index == 0:
        return True
    if mode == "always_source_refresh":
        return True
    if mode == "periodic_source_refresh" and int(refresh_interval) > 0:
        return (int(segment_index) + 1) % int(refresh_interval) == 0
    return False


def _use_tail_then_source_anchor(segment_index, continuity_mode, refresh_interval):
    mode = str(continuity_mode)
    if int(segment_index) <= 0:
        return False
    if mode == "tail_then_source_refresh":
        return True
    if mode == "periodic_tail_then_source_refresh" and int(refresh_interval) > 0:
        return (int(segment_index) + 1) % int(refresh_interval) == 0
    return False


def _clone_latent(latent):
    if latent is None:
        return None
    cloned = copy.deepcopy(latent)
    samples = cloned.get("samples")
    if isinstance(samples, torch.Tensor):
        cloned["samples"] = samples.detach().clone()
    noise_mask = cloned.get("noise_mask")
    if isinstance(noise_mask, torch.Tensor):
        cloned["noise_mask"] = noise_mask.detach().clone()
    return cloned


def _apply_iamccs_audio_latent_zero_mask(audio_latent):
    audio_samples = audio_latent.get("samples") if isinstance(audio_latent, dict) else None
    if audio_samples is None:
        raise ValueError("audio latent missing samples for IAMCCS audio+image mask")
    audio_zero_mask = torch.zeros_like(audio_samples, dtype=torch.float32, device=audio_samples.device)
    masked_audio_latent = audio_latent.copy()
    masked_audio_latent["noise_mask"] = audio_zero_mask
    return masked_audio_latent, audio_zero_mask


def _apply_workflow_solid_mask_to_audio_latent(audio_latent, width, height):
    # Match the working IAMCCS A+I2V workflows: ImageResizeKJv2 runtime
    # dimensions feed SolidMask, then SolidMask feeds SetLatentNoiseMask.
    mask_width = max(1, int(width))
    mask_height = max(1, int(height))
    audio_mask = SolidMask.execute(0.0, mask_width, mask_height)[0]
    masked_audio_latent = comfy_nodes.SetLatentNoiseMask().set_mask(audio_latent, audio_mask)[0]
    return masked_audio_latent, audio_mask, mask_width, mask_height


def _apply_legacy_backend_mask_to_audio_latent(audio_latent):
    audio_mask = SolidMask.execute(0.0, 1024, 1024)[0]
    masked_audio_latent = comfy_nodes.SetLatentNoiseMask().set_mask(audio_latent, audio_mask)[0]
    return masked_audio_latent, audio_mask


def _is_legacy_audio_img2vid_backend(backend_mode):
    return _normalize_backend_mode(backend_mode) in _LEGACY_AUDIO_IMG2VID_BACKENDS


def _is_legacy_single_audio_img2vid_backend(backend_mode):
    return _normalize_backend_mode(backend_mode) == "legacy_single"


def _latent_time_scale_factor(vae):
    try:
        scale = int(getattr(vae, "downscale_index_formula", (8, 1, 1))[0])
    except Exception:
        scale = 8
    return max(1, scale)


def _protected_prefix_latent_frames(vae, protect_video_frames):
    if int(protect_video_frames) <= 0:
        return 0
    return int(math.ceil(float(protect_video_frames) / float(_latent_time_scale_factor(vae))))


def _apply_tail_reference_adain(latents, reference, factor, protect_video_frames, vae, per_frame=False):
    strength = float(factor or 0.0)
    if latents is None or reference is None or strength <= 0.0:
        return latents

    samples = latents.get("samples")
    reference_samples = reference.get("samples") if isinstance(reference, dict) else None
    if not isinstance(samples, torch.Tensor) or not isinstance(reference_samples, torch.Tensor):
        return latents

    tail_start = _protected_prefix_latent_frames(vae, protect_video_frames)
    if tail_start >= int(samples.shape[2]):
        return latents

    target_tail = _clone_latent(latents)
    target_tail["samples"] = samples[:, :, tail_start:, :, :].detach().clone()
    reference_tail = _clone_latent(reference)

    if bool(per_frame):
        target_frames = int(target_tail["samples"].shape[2])
        reference_frames = int(reference_tail["samples"].shape[2])
        if reference_frames == 1 and target_frames > 1:
            reference_tail["samples"] = reference_tail["samples"].repeat(1, 1, target_frames, 1, 1)
        elif reference_frames > target_frames:
            reference_tail["samples"] = reference_tail["samples"][:, :, -target_frames:, :, :]

    normalized_tail = _node_class("LTXVAdainLatent")().batch_normalize(
        target_tail,
        reference_tail,
        strength,
        bool(per_frame),
    )[0]
    result = _clone_latent(latents)
    result["samples"][:, :, tail_start:, :, :] = normalized_tail["samples"]
    return result


def _soft_cleanup():
    gc.collect()
    try:
        model_management.soft_empty_cache()
    except Exception:
        pass


def _hard_unload_all_models():
    gc.collect()
    try:
        model_management.unload_all_models()
    except Exception:
        pass
    try:
        model_management.soft_empty_cache()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _accelerate_exec_model_if_available(model):
    # Diffusion-model checkpoints should pass through unchanged; the GGUF helper
    # is intentionally disabled for the SuperNode single-reference path.
    return model, "gguf_accelerator=disabled reason=diffusion_model_path"


class IAMCCS_GC_AUIMG2VIDExecutablePlanner:
    CATEGORY = "IAMCCS/GoyAIcanvas/TestBackends"
    FUNCTION = "plan"
    RETURN_TYPES = ("STRING", "FLOAT", "INT", "INT", "INT", "FLOAT", "STRING", SUPERNODE_LINX_TYPE, "STRING")
    RETURN_NAMES = (
        "plan_payload",
        "planned_duration_seconds",
        "total_frames",
        "segment_count",
        "recommended_overlap_frames",
        "recommended_audio_left_context_s",
        "planner_chip",
        "linx",
        "report",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "fps": ("FLOAT", {"default": 25.0, "min": 0.001, "max": 240.0, "step": 0.01}),
                "segment_seconds": ([5.0, 10.0, 15.0, 20.0, 30.0, 60.0], {"default": 10.0}),
                "planning_mode": (["manual_segment_seconds", "explicit_preset_seconds"], {"default": "manual_segment_seconds"}),
                "segment_preset": (["5sec", "10sec", "15sec", "20sec"], {"default": "10sec"}),
                "overlap_frames": ("INT", {"default": 9, "min": 0, "max": 4096, "step": 1}),
                "ltx_round_mode": (["up", "nearest", "down"], {"default": "up"}),
                "audio_preprocess_mode": (_PLANNER_AUDIO_MODES, {"default": "melband_vocals_duration_math"}),
                "melband_model_name": (_MELBAND_MODEL_NAMES, {"default": _MELBAND_MODEL_NAMES[0] if _MELBAND_MODEL_NAMES else "MelBandRoformer_fp32.safetensors"}),
                "audio_img2vid_backend": (_PLANNER_AUDIO_IMG2VID_BACKENDS, {"default": "modern pipeline"}),
                "audio_img2vid_mode": (_PLANNER_AUDIO_IMG2VID_MODES, {"default": "single generation"}),
                "route_mode": (_PLANNER_ROUTE_MODES, {"default": "choose segment count (audio / segments)"}),
                "segment_count": ([1, 2, 3, 4, 5, 6, 8, 10, 12], {"default": 2}),
                "single_duration_seconds": ([5.0, 10.0, 15.0, 20.0, 30.0, 60.0], {"default": 10.0}),
            }
            ,
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "linx": (SUPERNODE_LINX_TYPE,),
                "audio_concat_payload": ("STRING",),
                "debug_verbose": ("BOOLEAN", {"default": False}),
            }
        }

    def plan(self, audio, fps, segment_seconds, planning_mode, segment_preset, overlap_frames, ltx_round_mode, audio_preprocess_mode, melband_model_name, audio_img2vid_backend="modern", audio_img2vid_mode="single", route_mode="choose segment count (audio / segments)", segment_count=2, single_duration_seconds=10.0, debug_verbose=False, model=None, clip=None, vae=None, audio_vae=None, linx=None, audio_concat_payload=""):
        debug_verbose = _debug_verbose_enabled(debug_verbose, linx)
        pipeline_debug = _pipeline_debug_new("exec_planner", debug_verbose)
        _pipeline_debug_step(
            pipeline_debug,
            "input_received",
            audio=audio,
            fps=fps,
            segment_seconds=segment_seconds,
            planning_mode=planning_mode,
            segment_preset=segment_preset,
            overlap_frames=overlap_frames,
            ltx_round_mode=ltx_round_mode,
            audio_preprocess_mode=audio_preprocess_mode,
            melband_model_name=melband_model_name,
            audio_img2vid_backend=audio_img2vid_backend,
            audio_img2vid_mode=audio_img2vid_mode,
            route_mode=route_mode,
            segment_count=segment_count,
            single_duration_seconds=single_duration_seconds,
            linx_keys=",".join(str(k) for k in ((linx or {}).get("resource_keys") or [])),
        )
        planning_mode = _normalize_planner_mode(planning_mode)
        segment_preset = _normalize_segment_preset(segment_preset)
        audio_preprocess_mode = _normalize_audio_preprocess_mode(audio_preprocess_mode)
        audio_img2vid_backend = _normalize_audio_img2vid_backend_choice(audio_img2vid_backend)
        audio_img2vid_mode = _normalize_audio_img2vid_mode_choice(audio_img2vid_mode, route_mode, segment_count)
        audio_img2vid_render_backend = _audio_img2vid_backend_from_planner(audio_img2vid_backend, audio_img2vid_mode)
        _pipeline_debug_step(
            pipeline_debug,
            "planner_widgets_normalized",
            planning_mode=planning_mode,
            segment_preset=segment_preset,
            audio_preprocess_mode=audio_preprocess_mode,
            audio_img2vid_backend=audio_img2vid_backend,
            audio_img2vid_mode=audio_img2vid_mode,
            audio_img2vid_render_backend=audio_img2vid_render_backend,
        )
        audio_plan = _prepare_planner_audio(audio, audio_preprocess_mode, melband_model_name)
        _pipeline_debug_step(
            pipeline_debug,
            "audio_preprocessed",
            raw_audio=audio_plan["raw_audio"],
            conditioning_audio_single=audio_plan["conditioning_audio_single"],
            conditioning_audio_segmented=audio_plan["conditioning_audio_segmented"],
            ltx_conditioning_audio=audio_plan["ltx_conditioning_audio"],
            ltx_conditioning_source=audio_plan["ltx_conditioning_source"],
            duration_audio=audio_plan["duration_audio"],
            duration_seconds=audio_plan["duration_seconds"],
            melband_enabled=audio_plan["melband_enabled"],
            preprocess_report=audio_plan["preprocess_report"],
        )
        audio_duration_frames_with_tail, audio_duration_frames_source = _compute_audio_duration_frames_with_tail(audio_plan["duration_audio"], fps)
        audio_plan["duration_frames_with_tail"] = int(audio_duration_frames_with_tail)
        audio_plan["duration_frames_source"] = str(audio_duration_frames_source)
        duration_seconds = _duration_hint_from_payload(audio_concat_payload)
        _pipeline_debug_step(
            pipeline_debug,
            "duration_resolved",
            audio_concat_payload_present=bool(audio_concat_payload),
            audio_concat_duration=duration_seconds,
            audio_duration_seconds=audio_plan["duration_seconds"],
            audio_duration_frames_with_tail=audio_duration_frames_with_tail,
            audio_duration_frames_source=audio_duration_frames_source,
        )
        if duration_seconds is None:
            duration_seconds = float(audio_plan["duration_seconds"])

        if audio_img2vid_mode == "single":
            route_mode = "single generation (duration only)"
            segment_count = 1
        else:
            route_mode = "choose segment count (audio / segments)"
            if audio_img2vid_mode == "2_segments":
                segment_count = 2
            elif audio_img2vid_mode == "3_segments":
                segment_count = 3
            else:
                segment_count = max(4, int(segment_count or 4))

        route_key = _normalize_planner_route_mode(route_mode)
        auto_report = ""
        if route_key == "single_generation":
            duration_seconds = max(0.1, float(single_duration_seconds or segment_seconds or 10.0))
            segment_seconds = duration_seconds
            planning_mode = "manual_segment_seconds"
            segment_preset = _segment_preset_from_seconds(segment_seconds)
        else:
            auto_planner = _node_class("IAMCCS_AudioSegmentAutoPlanner")()
            split_mode = "choose seconds per segment (auto count)" if route_key == "fixed_segment_seconds" else "choose segment count (audio / segments)"
            auto_planned = auto_planner.plan_audio_segments(
                audio_plan["duration_audio"],
                fps,
                split_mode,
                segment_count,
                segment_seconds,
                overlap_frames,
                ltx_round_mode,
            )
            duration_seconds = float(auto_planned[0])
            segment_seconds = float(auto_planned[1])
            planning_mode = str(auto_planned[2])
            segment_preset = str(auto_planned[3])
            overlap_frames = int(auto_planned[4])
            auto_report = str(auto_planned[12])

        planner = _node_class("IAMCCS_SegmentPlanner")()
        planned = planner.plan(
            duration_seconds,
            fps,
            segment_seconds,
            planning_mode,
            segment_preset,
            overlap_frames,
            ltx_round_mode,
            0,
        )
        target_frame_count = int(planned[0]) if route_key == "single_generation" else int(audio_duration_frames_with_tail)
        target_frame_count_source = "planner_single_duration" if route_key == "single_generation" else "planner_audio_duration_with_tail"
        _pipeline_debug_step(
            pipeline_debug,
            "segment_plan_created",
            duration_seconds=duration_seconds,
            total_frames=int(planned[0]),
            segment_count=int(planned[4]),
            first_segment_raw_frames=int(planned[2]),
            continuation_raw_frames=int(planned[3]),
            recommended_overlap_frames=int(planned[18]),
            recommended_audio_left_context_s=float(planned[19]),
            recommended_extension_preset=planned[20],
            route_mode=route_mode,
            route_key=route_key,
            auto_planner_report=auto_report,
        )
        payload = (
            f"pipeline_kind=au_img2vid_exec; total_duration_seconds={duration_seconds:.6f}; fps={float(fps):.6f}; "
            f"segment_seconds={float(segment_seconds):.6f}; planning_mode={planning_mode}; segment_preset={segment_preset}; content_profile={segment_preset}; "
            f"audio_duration_frames_with_tail={int(audio_duration_frames_with_tail)}; audio_duration_frames_source={audio_duration_frames_source}; "
            f"target_frame_count={int(target_frame_count)}; target_frame_count_source={target_frame_count_source}; "
            f"audio_img2vid_backend_family={audio_img2vid_backend}; audio_img2vid_mode={audio_img2vid_mode}; "
            f"audio_img2vid_render_backend={audio_img2vid_render_backend}; "
            f"overlap_frames={int(overlap_frames)}; ltx_round_mode={ltx_round_mode}; total_frames={int(planned[0])}; "
            f"unique_segment_frames={int(planned[1])}; first_segment_raw_frames={int(planned[2])}; continuation_raw_frames={int(planned[3])}; "
            f"segment_count={int(planned[4])}; continuation_loops={int(planned[5])}; last_segment_unique_frames={int(planned[6])}; "
            f"recommended_overlap_frames={int(planned[18])}; recommended_audio_left_context_s={float(planned[19]):.6f}; "
            f"recommended_extension_preset={planned[20]}"
        )
        planner_chip = _planner_chip(duration_seconds, planned, planning_mode, segment_preset)
        planner_debug_report = _pipeline_debug_text(pipeline_debug)
        report = (
            f"Executable planner. duration={duration_seconds:.3f}s @ {float(fps):.3f}fps | "
            f"segments={int(planned[4])} | first_raw={int(planned[2])}f | continuation_raw={int(planned[3])}f | "
            f"recommended_overlap={int(planned[18])}f | left_context={float(planned[19]):.3f}s | "
            f"audio_frames_with_tail={int(audio_duration_frames_with_tail)}f ({audio_duration_frames_source}) | "
            f"audio_preprocess_mode={audio_preprocess_mode} | melband_model={melband_model_name} | "
            f"a2i_backend={audio_img2vid_backend}/{audio_img2vid_mode}->{audio_img2vid_render_backend} | "
            f"melband_enabled={bool(audio_plan['melband_enabled'])} | {audio_plan['preprocess_report']}"
        )
        report = _append_debug_report(report, planner_debug_report)
        linx = build_stage_linx_payload(
            linx,
            "exec_planner",
            "planning",
            {
                "pipeline_kind": "au_img2vid_exec",
                "fps": float(fps),
                "segment_seconds": float(segment_seconds),
                "planning_mode": str(planning_mode),
                "segment_preset": str(segment_preset),
                "content_profile": str(segment_preset),
                "recommended_extension_preset": str(planned[20]),
                "audio_concat_duration_override": float(duration_seconds),
                "route_mode": str(route_mode),
                "route_key": str(route_key),
                "audio_img2vid_backend_family": str(audio_img2vid_backend),
                "audio_img2vid_mode": str(audio_img2vid_mode),
                "audio_img2vid_render_backend": str(audio_img2vid_render_backend),
                "audio_preprocess_mode": str(audio_preprocess_mode),
                "melband_model_name": str(melband_model_name),
                "pipeline_debug_steps": len(pipeline_debug.get("lines") or []),
                "debug_verbose": bool(debug_verbose),
            },
            report,
            requires={
                "inputs": {"audio": "AUDIO", "model": "MODEL", "clip": "CLIP", "vae": "VAE", "audio_vae": "VAE"},
                "optional_linx": {SUPERNODE_LINX_TYPE: "merge upstream resources"},
            },
            slot_map={
                "plan_payload": {"type": "STRING", "role": "exec_plan"},
                "planner_chip": {"type": "STRING", "role": "planner_chip"},
                "linx": {"type": SUPERNODE_LINX_TYPE, "role": "stage_linx"},
            },
            downstream_stages=["render", "finalize"],
            policies={
                "recommended_overlap_frames": int(planned[18]),
                "recommended_audio_left_context_s": float(planned[19]),
            },
            outputs={
                "plan_payload": payload,
                "planned_duration_seconds": float(duration_seconds),
                "total_frames": int(planned[0]),
                "target_frame_count": int(target_frame_count),
                "target_frame_count_source": str(target_frame_count_source),
                "audio_img2vid_backend_family": str(audio_img2vid_backend),
                "audio_img2vid_mode": str(audio_img2vid_mode),
                "audio_img2vid_render_backend": str(audio_img2vid_render_backend),
                "audio_duration_frames_with_tail": int(audio_duration_frames_with_tail),
                "segment_count": int(planned[4]),
                "first_segment_raw_frames": int(planned[2]),
                "continuation_raw_frames": int(planned[3]),
                "recommended_overlap_frames": int(planned[18]),
                "recommended_audio_left_context_s": float(planned[19]),
                "planner_chip": planner_chip,
                "pipeline_debug": list(pipeline_debug.get("lines") or []),
                "debug_verbose": bool(debug_verbose),
            },
            resources={
            "audio": audio,
            "audio_raw": audio_plan["raw_audio"],
            "audio_conditioning_single": audio_plan["conditioning_audio_single"],
            "audio_conditioning_segmented": audio_plan["conditioning_audio_segmented"],
            "audio_ltx_conditioning": audio_plan["ltx_conditioning_audio"],
            "audio_ltx_conditioning_source": audio_plan["ltx_conditioning_source"],
            "audio_duration_source": audio_plan["duration_audio"],
            "target_frame_count": int(target_frame_count),
            "target_frame_count_source": str(target_frame_count_source),
            "audio_duration_frames_with_tail": int(audio_duration_frames_with_tail),
                "audio_duration_frames_source": str(audio_duration_frames_source),
                "model": model,
                "clip": clip,
                "vae": vae,
                "audio_vae": audio_vae,
                "fps": float(fps),
                "planner_payload": payload,
                "audio_preprocess_mode": str(audio_preprocess_mode),
                "melband_model_name": str(melband_model_name),
                "audio_preprocess_report": audio_plan["preprocess_report"],
                "melband_enabled": bool(audio_plan["melband_enabled"]),
                "debug_verbose": bool(debug_verbose),
            },
        )
        return (
            payload,
            float(duration_seconds),
            int(planned[0]),
            int(planned[4]),
            int(planned[18]),
            float(planned[19]),
            planner_chip,
            linx,
            report,
        )


class IAMCCS_GC_AUIMG2VIDExecutableRender:
    CATEGORY = "IAMCCS/GoyAIcanvas/TestBackends"
    FUNCTION = "render"
    RETURN_TYPES = ("STRING", "STRING", "INT", "FLOAT", SUPERNODE_LINX_TYPE, "STRING")
    RETURN_NAMES = ("frames_dir", "start_dir", "segments_rendered", "estimated_duration_seconds", "linx", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generation_type": (_GENERATION_TYPE_MODES, {"default": _AUDIO_IMAGE_GENERATION_TYPE}),
                "ui_preset": (["custom", "low_ram_safe", "balanced", "high_quality", "fast_preview", "motion_controlled", "loop_lipsync_safe", "img2vid_generated_audio", "t2v_generated_audio", "img2vid_pure", "t2v_pure", "loop_img2vid_pure_normal_vram", "loop_t2v_pure_normal_vram", "loop_img2vid_pure_low_ram"], {"default": "custom"}),
                "generated_media_duration_seconds": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 120.0, "step": 0.1}),
                "generated_media_fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 240.0, "step": 0.01}),
                "generation_mode": (["img2vid", "t2v"], {"default": "img2vid"}),
                "backend_mode": (_RENDER_BACKEND_MODES, {"default": "auto"}),
                "positive_text": ("STRING", {"default": "cinematic motion, detailed scene", "multiline": True}),
                "negative_text": ("STRING", {"default": "blurry, low quality, artifacts", "multiline": True}),
                "width": ("INT", {"default": 1280, "min": 64, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 720, "min": 64, "max": 8192, "step": 32}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 200, "step": 1}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (_SAMPLER_NAMES, {"default": "lcm" if "lcm" in _SAMPLER_NAMES else "euler"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                "max_shift": ("FLOAT", {"default": 2.05, "min": 0.0, "max": 100.0, "step": 0.01}),
                "base_shift": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 100.0, "step": 0.01}),
                "sigma_terminal": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.99, "step": 0.01}),
                "manual_sigmas": ("STRING", {"default": _REFERENCE_MANUAL_SIGMAS, "multiline": True}),
                "image_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "image_compression": ("INT", {"default": 33, "min": 0, "max": 100, "step": 1}),
                "audio_context_mode": (["left_context_only", "right_context_only", "symmetric_context", "no_overlap"], {"default": "left_context_only"}),
                "audio_left_context_s": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 30.0, "step": 0.01}),
                "audio_right_context_s": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "stitch_preset": (["custom", "lossless_refresh_24fps", "lossless_refresh_strong_24fps", "videoclip_audio_24fps", "monologue_audio_24fps", "target_extension_ltx2", "cut_bestofk_16", "cut_bestofk_16_luma", "cut_bestofk_32", "micro_crossfade_3"], {"default": "custom"}),
                "overlap_side": (["source", "new_images"], {"default": "source"}),
                "overlap_mode": (["cut", "linear_blend", "ease_in_out", "filmic_crossfade"], {"default": "cut"}),
                "start_frames_rule": (["none", "ltx2_round_down", "ltx2_nearest"], {"default": "none"}),
                "color_match_mode": (["none", "luma_only", "per_channel"], {"default": "none"}),
                "color_match_strength": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
                "continuity_anchor_mode": (["off", "tail_only", "periodic_tail_only", "periodic_tail_then_source_refresh", "tail_then_source_refresh", "periodic_source_refresh", "always_source_refresh"], {"default": "off"}),
                "anchor_refresh_interval": ("INT", {"default": 2, "min": 1, "max": 128, "step": 1}),
                "anchor_image_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "anti_drift_mode": (["off", "rolling_adain", "dual_reference_adain"], {"default": "off"}),
                "anti_drift_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "identity_persistence_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "vae_mode": (_MODULAR_DECODE_MODES, {"default": "inherit_render_backend"}),
                "downstream_stage_mode": (["finalize_only", "upscale_ready", "detailer_ready", "upscale_then_detailer"], {"default": "finalize_only"}),
                "output_root": ("STRING", {"default": "iamccs_gc_auimg2vid/exec_run"}),
                "segment_overlay_mode": (["off", "segment_label", "custom_text"], {"default": "off"}),
                "segment_overlay_text": ("STRING", {"default": "seg {segment_number}/{segment_count}", "multiline": True}),
                "second_stage_mode": (["off", "latent_refine_3step", "latent_upscale_refine_x2_beta"], {"default": "off"}),
                "stage2_model_policy": (["stage2_model_if_connected", "replace_stage1_if_connected", "prefer_stage2_else_primary", "keep_stage1_model"], {"default": "stage2_model_if_connected"}),
                "second_stage_upscale_model": (_LATENT_UPSCALE_MODEL_NAMES, {"default": _reference_ltx23_upscale_model_name()}),
                "second_stage_reinject_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "second_stage_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "second_stage_manual_sigmas": ("STRING", {"default": "0.909375, 0.725, 0.421875, 0.0", "multiline": True}),
                "media_mode": (_MEDIA_MODES, {"default": "auto_from_generation_mode"}),
                "vram_flush": ("BOOLEAN", {"default": False}),
                "motion_intensity": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05}),
            },
            "optional": {
                "image": ("IMAGE", {"lazy": True}),
                "linx": (SUPERNODE_LINX_TYPE, {"lazy": True}),
                "audio": ("AUDIO", {"lazy": True}),
                "model": ("MODEL", {"lazy": True}),
                "clip": ("CLIP", {"lazy": True}),
                "vae": ("VAE", {"lazy": True}),
                "audio_vae": ("VAE", {"lazy": True}),
                "plan_payload": ("STRING",),
                "refresh_image": ("IMAGE", {"lazy": True}),
                "second_stage_linx": (SUPERNODE_LINX_TYPE,),
                "stage2_model": ("MODEL",),
                "show_manual_sigmas": ("BOOLEAN", {"default": False}),
                "debug_verbose": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    def check_lazy_status(
        self,
        generation_type=_AUDIO_IMAGE_GENERATION_TYPE,
        generation_mode="img2vid",
        backend_mode="auto",
        media_mode="auto_from_generation_mode",
        second_stage_mode="off",
        image=None,
        linx=None,
        audio=None,
        model=None,
        clip=None,
        vae=None,
        audio_vae=None,
        second_stage_linx=None,
        stage2_model=None,
        **kwargs,
    ):
        generation_type, generation_mode, backend_mode, media_mode = _resolve_generation_type(
            generation_type,
            generation_mode,
            backend_mode,
            media_mode,
        )
        media_settings = _resolve_media_mode(media_mode, generation_mode)
        needed = []

        if str(media_settings["generation_mode"]) != "t2v" and image is None:
            needed.append("image")

        if bool(media_settings["uses_input_audio"]):
            if linx is None:
                needed.append("linx")
                return needed
            for name, value in (
                ("audio", audio),
                ("model", model),
                ("clip", clip),
                ("vae", vae),
                ("audio_vae", audio_vae),
            ):
                if value is None and linx_resource(linx, name, None) is None:
                    needed.append(name)
        else:
            for name, value in (
                ("model", model),
                ("clip", clip),
                ("vae", vae),
                ("audio_vae", audio_vae),
            ):
                if value is None:
                    needed.append(name)

        return needed

    def _planner_settings(self, plan_payload, linx, fps, segment_seconds, planning_mode, segment_preset, overlap_frames, ltx_round_mode):
        data = _parse_payload(plan_payload)
        if isinstance(linx, dict):
            outputs = linx.get("outputs") or {}
            if isinstance(outputs, dict):
                for key, value in outputs.items():
                    data.setdefault(str(key), value)
        if not data:
            return {
                "duration_seconds": None,
                "fps": float(fps),
                "segment_seconds": float(segment_seconds),
                "planning_mode": _normalize_planner_mode(planning_mode),
                "segment_preset": _normalize_segment_preset(segment_preset),
                "overlap_frames": int(overlap_frames),
                "ltx_round_mode": str(ltx_round_mode),
                "audio_img2vid_backend_family": "modern",
                "audio_img2vid_mode": "single",
                "audio_img2vid_render_backend": "single_best",
            }
        backend_family = _normalize_audio_img2vid_backend_choice(_to_text(data, "audio_img2vid_backend_family", "modern"))
        backend_mode_choice = _normalize_audio_img2vid_mode_choice(
            _to_text(data, "audio_img2vid_mode", ""),
            _to_text(data, "route_mode", ""),
            _to_int(data, "segment_count", 2),
        )
        render_backend = _normalize_backend_mode(
            _to_text(data, "audio_img2vid_render_backend", _audio_img2vid_backend_from_planner(backend_family, backend_mode_choice))
        )
        return {
            "duration_seconds": _to_float(data, "total_duration_seconds", None),
            "fps": _to_float(data, "fps", fps),
            "segment_seconds": _to_float(data, "segment_seconds", segment_seconds),
            "planning_mode": _normalize_planner_mode(_to_text(data, "planning_mode", planning_mode)),
            "segment_preset": _normalize_segment_preset(_to_text(data, "segment_preset", _to_text(data, "content_profile", segment_preset))),
            "overlap_frames": _to_int(data, "overlap_frames", overlap_frames),
            "ltx_round_mode": _to_text(data, "ltx_round_mode", ltx_round_mode),
            "audio_img2vid_backend_family": backend_family,
            "audio_img2vid_mode": backend_mode_choice,
            "audio_img2vid_render_backend": render_backend,
        }

    def render(
        self,
        generation_mode,
        backend_mode,
        positive_text,
        negative_text,
        width,
        height,
        steps,
        cfg,
        sampler_name,
        seed,
        max_shift,
        base_shift,
        sigma_terminal,
        manual_sigmas,
        image_strength,
        image_compression,
        audio_context_mode,
        audio_left_context_s,
        audio_right_context_s,
        stitch_preset,
        overlap_side,
        overlap_mode,
        start_frames_rule,
        color_match_mode,
        color_match_strength,
        continuity_anchor_mode,
        anchor_refresh_interval,
        anchor_image_strength,
        anti_drift_mode,
        anti_drift_strength,
        identity_persistence_strength,
        vae_mode,
        downstream_stage_mode,
        output_root,
        segment_overlay_mode,
        segment_overlay_text,
        second_stage_mode,
        stage2_model_policy,
        second_stage_upscale_model,
        second_stage_reinject_strength,
        second_stage_cfg,
        second_stage_manual_sigmas,
        media_mode,
      vram_flush,
      motion_intensity,
      show_manual_sigmas=False,
      debug_verbose=False,
      ui_preset="custom",
        generated_media_duration_seconds=10.0,
        generated_media_fps=25.0,
        generation_type=_AUDIO_IMAGE_GENERATION_TYPE,
        image=None,
        linx=None,
        audio=None,
        model=None,
        clip=None,
        vae=None,
        audio_vae=None,
        plan_payload="",
        refresh_image=None,
        second_stage_linx=None,
        stage2_model=None,
        unique_id=None,
    ):
        debug_verbose = _debug_verbose_enabled(debug_verbose, linx)
        pipeline_debug = _pipeline_debug_new("exec_render", debug_verbose)
        _pipeline_debug_step(
            pipeline_debug,
            "input_received",
            generation_type=generation_type,
            generation_mode=generation_mode,
            backend_mode=backend_mode,
            media_mode=media_mode,
            ui_preset=ui_preset,
            image_connected=image is not None,
            audio_connected=audio is not None,
            linx_keys=",".join(str(k) for k in ((linx or {}).get("resource_keys") or [])),
        )
        generation_type, generation_mode, backend_mode, media_mode = _resolve_generation_type(
            generation_type,
            generation_mode,
            backend_mode,
            media_mode,
        )
        _pipeline_debug_step(
            pipeline_debug,
            "generation_type_resolved",
            generation_type=generation_type,
            generation_mode=generation_mode,
            backend_mode=backend_mode,
            media_mode=media_mode,
        )
        reference_audio_img2vid = _is_single_best_reference_audio_img2vid(generation_type)
        legacy_audio_img2vid_backend = reference_audio_img2vid and _is_legacy_audio_img2vid_backend(backend_mode)
        requested_media_probe = str(media_mode or "auto_from_generation_mode")
        media_probe = _resolve_media_mode(requested_media_probe, generation_mode)
        input_audio_probe = _input_or_linx(audio, linx, "audio")
        if requested_media_probe == "auto_from_generation_mode" and input_audio_probe is None:
            fallback_media_probe = "generated_audio_t2v" if str(generation_mode) == "t2v" else "generated_audio_img2vid"
            media_probe = _resolve_media_mode(fallback_media_probe, generation_mode)
        _pipeline_debug_step(
            pipeline_debug,
            "media_probe",
            requested_media=requested_media_probe,
            resolved_media=media_probe.get("mode"),
            uses_input_audio=bool(media_probe.get("uses_input_audio")),
            generates_audio=bool(media_probe.get("generates_audio")),
            fallback_without_audio=(requested_media_probe == "auto_from_generation_mode" and input_audio_probe is None),
        )

        if not bool(media_probe["uses_input_audio"]):
            linx = _linx_models_only(linx)
            _pipeline_debug_step(pipeline_debug, "linx_pruned_for_non_audio_mode", linx_keys=",".join(str(k) for k in ((linx or {}).get("resource_keys") or [])))

        model = _require_runtime_value(_input_or_linx(model, linx, "model"), "model")
        clip = _require_runtime_value(_input_or_linx(clip, linx, "clip"), "clip")
        vae = _require_runtime_value(_input_or_linx(vae, linx, "vae"), "vae")
        audio_vae = _require_runtime_value(_input_or_linx(audio_vae, linx, "audio_vae"), "audio_vae")
        uses_input_audio_probe = bool(media_probe["uses_input_audio"])
        fps_default = _REFERENCE_AUDIO_IMG2VID_FPS if reference_audio_img2vid else (24.0 if uses_input_audio_probe else float(generated_media_fps or 25.0))
        fps_value = float(linx_resource(linx, "fps", fps_default) or fps_default) if uses_input_audio_probe else max(1.0, float(generated_media_fps or fps_default))
        _pipeline_debug_step(
            pipeline_debug,
            "resources_resolved",
            model=model,
            clip=clip,
            vae=vae,
            audio_vae=audio_vae,
            fps=fps_value,
            fps_source="planner_linx" if uses_input_audio_probe else "render_widget",
        )
        plan_payload = "" if not bool(media_probe["uses_input_audio"]) else str(plan_payload or linx_resource(linx, "planner_payload") or linx_output(linx, "plan_payload") or "")
        if not plan_payload and not bool(media_probe["uses_input_audio"]):
            plan_payload = _generated_duration_plan_payload(generated_media_duration_seconds, fps_value)
            generated_plan_data = _parse_payload(plan_payload)
            _pipeline_debug_step(
                pipeline_debug,
                "generated_duration_plan_created",
                duration_seconds=generated_media_duration_seconds,
                fps=fps_value,
                requested_frames=_to_int(generated_plan_data, "requested_frames", 0),
                total_frames=_to_int(generated_plan_data, "total_frames", 0),
                ltx_rounded=_to_bool(generated_plan_data, "generated_media_ltx_rounded", False),
            )
        if not plan_payload:
            _pipeline_debug_step(pipeline_debug, "error_missing_plan_payload", uses_input_audio=bool(media_probe["uses_input_audio"]))
            raise ValueError("plan_payload is required via Exec Planner/linx unless Media Mode is generated-audio/pure")

        payload_backend_data = _parse_payload(plan_payload)
        planner_backend_raw = _to_text(payload_backend_data, "audio_img2vid_render_backend", "").strip()
        if reference_audio_img2vid and planner_backend_raw:
            backend_mode = _normalize_backend_mode(planner_backend_raw)
        backend_mode = _normalize_backend_mode(backend_mode)
        legacy_audio_img2vid_backend = reference_audio_img2vid and _is_legacy_audio_img2vid_backend(backend_mode)
        if reference_audio_img2vid:
            _pipeline_debug_step(
                pipeline_debug,
                "audio_img_contract_loaded",
                contract_version=_AUDIO_IMG2VID_CONTRACT_VERSION,
                default_resolution=f"{_REFERENCE_AUDIO_IMG2VID_WIDTH}x{_REFERENCE_AUDIO_IMG2VID_HEIGHT}",
                resolution_source="frontend_width_height_to_ImageResizeKJv2_actual_outputs",
                resize_divisible_by=_REFERENCE_AUDIO_IMG2VID_DIVISIBLE_BY,
                examples="1280x720 matches the working workflow input; 1920x1080 follows ImageResizeKJv2 actual width/height",
                planner_backend_raw=planner_backend_raw or "none",
                audio_mask_contract="legacy_1024_solidmask" if legacy_audio_img2vid_backend else "workflow_solidmask_from_ImageResizeKJv2_width_height",
            )
        modular_decode = _normalize_modular_decode_mode(
            _inherit_widget_value(vae_mode, "inherit_render_backend", linx, "decode_mode"),
            backend_mode,
        )
        _pipeline_debug_step(pipeline_debug, "decode_policy_resolved", backend_mode=backend_mode, modular_decode=modular_decode, vae_mode_widget=vae_mode)
        output_root = str(_inherit_widget_value(output_root, "iamccs_gc_auimg2vid/exec_run", linx, "output_root"))
        audio_concat_payload = str(linx_output(linx, "audio_concat_payload", "") or "")
        continuity_payload = str(linx_output(linx, "continuity_payload", "") or "")
        requested_media_mode = str(media_mode or "auto_from_generation_mode")
        media_settings = _resolve_media_mode(requested_media_mode, generation_mode)
        linx_audio = linx_resource(linx, "audio", None)
        input_audio = linx_audio if linx_audio is not None else audio
        input_audio_source = "planner_linx" if linx_audio is not None else ("direct" if audio is not None else "missing")
        if requested_media_mode == "auto_from_generation_mode" and input_audio is None:
            fallback_media_mode = "generated_audio_t2v" if str(generation_mode) == "t2v" else "generated_audio_img2vid"
            media_settings = _resolve_media_mode(fallback_media_mode, generation_mode)
        generation_mode = str(media_settings["generation_mode"])
        effective_media_mode = str(media_settings["mode"])
        uses_input_audio = bool(media_settings["uses_input_audio"])
        generates_audio = bool(media_settings["generates_audio"])
        exports_audio = bool(media_settings.get("exports_audio", uses_input_audio or generates_audio))
        _pipeline_debug_step(
            pipeline_debug,
            "media_route_resolved",
            requested_media=requested_media_mode,
            effective_media=effective_media_mode,
            generation_mode=generation_mode,
            uses_input_audio=uses_input_audio,
            generates_audio=generates_audio,
            exports_audio=exports_audio,
            input_audio_source=input_audio_source,
        )
        if str(generation_mode) != "t2v" and image is None:
            _pipeline_debug_step(pipeline_debug, "error_missing_image", generation_mode=generation_mode, media_mode=effective_media_mode)
            raise ValueError("image is required for img2video generation types")
        audio = _require_runtime_value(input_audio, "audio") if uses_input_audio else None
        raw_audio = (linx_resource(linx, "audio_raw", None) or audio) if uses_input_audio else None
        segmented_audio = (linx_resource(linx, "audio_conditioning_segmented", None) or raw_audio) if uses_input_audio else None
        melband_single_audio = linx_resource(linx, "audio_conditioning_single", None) if uses_input_audio else None
        ltx_conditioning_audio = linx_resource(linx, "audio_ltx_conditioning", None) if uses_input_audio else None
        ltx_conditioning_source = str(
            linx_resource(linx, "audio_ltx_conditioning_source", "")
            or ""
        ) if uses_input_audio else ""
        planner_audio_preprocess_report = str(
            linx_resource(linx, "audio_preprocess_report", "")
            or ""
        ) if uses_input_audio else ""
        if uses_input_audio and reference_audio_img2vid:
            # BEST AUDIO2IMG2VIDEO_SINGLE sends MelBand vocals into LTXVAudioVAEEncode.
            # If an older/simpler workflow forgot to route the planner vocals, recover
            # vocals from raw audio here without changing the user's MelBand choice.
            recovered_melband_report = ""
            if melband_single_audio is None and raw_audio is not None:
                recovered_audio_plan = _prepare_planner_audio(
                    raw_audio,
                    "melband_vocals_duration_math",
                    "MelBandRoformer_fp32.safetensors",
                )
                recovered_melband_audio = recovered_audio_plan.get("conditioning_audio_single")
                if recovered_melband_audio is not None:
                    melband_single_audio = recovered_melband_audio
                    melband_enabled = True
                    recovered_melband_report = " | render_recovered_melband_vocals=yes"
            single_conditioning_audio = melband_single_audio or segmented_audio or ltx_conditioning_audio or raw_audio
        else:
            single_conditioning_audio = (ltx_conditioning_audio or raw_audio or segmented_audio or melband_single_audio) if uses_input_audio else None
        audio_preprocess_report = (planner_audio_preprocess_report or "audio_preprocess=unknown") if uses_input_audio else "audio_preprocess=generated_audio_empty_latent"
        melband_enabled = bool(linx_resource(linx, "melband_enabled", False)) if uses_input_audio else False
        if uses_input_audio and reference_audio_img2vid and melband_single_audio is not None:
            melband_enabled = True
        planner_audio_preprocess_mode = _normalize_audio_preprocess_mode(
            linx_resource(
                linx,
                "audio_preprocess_mode",
                "melband_vocals_duration_math" if melband_enabled else "raw_audio_only",
            )
        )
        planner_melband_model_name = _resolve_melband_model_name(
            linx_resource(linx, "melband_model_name", _MELBAND_MODEL_NAMES[0] if _MELBAND_MODEL_NAMES else "MelBandRoformer_fp32.safetensors")
        )
        if uses_input_audio and single_conditioning_audio is None:
            _pipeline_debug_step(pipeline_debug, "error_missing_audio_conditioning", raw_audio=raw_audio is not None, segmented_audio=segmented_audio is not None)
            raise ValueError("input-audio generation selected, but no audio conditioning reached Exec Render")
        if uses_input_audio:
            if reference_audio_img2vid and melband_single_audio is not None:
                conditioning_audio_source = "best_workflow:audio_conditioning_single_melband_vocals" + recovered_melband_report
            elif reference_audio_img2vid and segmented_audio is not None:
                conditioning_audio_source = "reference_canvas:audio_conditioning_segmented_fallback"
            elif ltx_conditioning_audio is not None:
                conditioning_audio_source = f"linx_audio_ltx_conditioning:{ltx_conditioning_source or 'raw_audio'}"
            elif raw_audio is not None:
                conditioning_audio_source = f"{input_audio_source}_raw_audio"
            elif segmented_audio is not None:
                conditioning_audio_source = "linx_audio_conditioning_segmented"
            elif melband_single_audio is not None:
                conditioning_audio_source = "linx_audio_conditioning_single_fallback"
            else:
                conditioning_audio_source = "missing"
        else:
            conditioning_audio_source = "generated_or_pure_audio_latent"
        _pipeline_debug_step(
            pipeline_debug,
            "audio_route",
            raw_audio=raw_audio,
            single_conditioning_audio=single_conditioning_audio,
            segmented_audio=segmented_audio,
            melband_single_audio=melband_single_audio,
            ltx_conditioning_audio=ltx_conditioning_audio,
            ltx_conditioning_source=ltx_conditioning_source,
            conditioning_audio_source=conditioning_audio_source,
            melband_enabled=melband_enabled,
            audio_preprocess_report=audio_preprocess_report,
        )
        print(
            "[IAMCCS SuperNodes Render] "
            f"audio_route generation_type={generation_type} generation_mode={generation_mode} "
            f"backend={backend_mode} media={effective_media_mode} uses_input_audio={uses_input_audio} "
            f"generates_audio={generates_audio} exports_audio={exports_audio} input_audio={input_audio_source} "
            f"raw_audio={'yes' if raw_audio is not None else 'no'} conditioning_audio={conditioning_audio_source} "
            f"melband={melband_enabled} resource_keys={','.join(str(k) for k in ((linx or {}).get('resource_keys') or []))}"
        )
        motion_intensity = max(0.25, min(4.0, float(motion_intensity or 1.0)))
        _pipeline_debug_step(pipeline_debug, "motion_and_stage2_widgets", motion_intensity=motion_intensity, second_stage_mode=second_stage_mode, stage2_model_policy=stage2_model_policy)
        second_stage_payload = _stage2_payload_from_exec_widgets(
            second_stage_mode,
            stage2_model_policy,
            second_stage_upscale_model,
            second_stage_reinject_strength,
            second_stage_cfg,
            second_stage_manual_sigmas,
        )
        second_stage_model = stage2_model
        if second_stage_model is None and isinstance(second_stage_linx, dict):
            second_stage_model = linx_resource(second_stage_linx, "second_stage_model", None)
        if second_stage_model is None:
            second_stage_model = linx_resource(linx, "second_stage_model", None)

        planner_settings = self._planner_settings(
            plan_payload,
            linx,
            fps_value,
            10.0,
            "manual_segment_seconds",
            "15sec",
            9,
            "up",
        )
        if reference_audio_img2vid:
            _pipeline_debug_step(
                pipeline_debug,
                "reference_audio_img2vid_route",
                fps=planner_settings["fps"],
                duration_rule="25fps keeps reference frame math; other fps use LTX 8n+1 rounding",
                image_preprocess="best_workflow:ImageResizeKJv2_lanczos_crop_top_before_ltx_preprocess",
                audio_mask="legacy_backend:SolidMask_1024_to_SetLatentNoiseMask" if legacy_audio_img2vid_backend else "workflow:SolidMask_from_ImageResizeKJv2_width_height_to_SetLatentNoiseMask",
                audio_vae_source=conditioning_audio_source,
                contract_version=_AUDIO_IMG2VID_CONTRACT_VERSION,
            )
        total_duration_seconds = None
        if total_duration_seconds is None:
            total_duration_seconds = planner_settings["duration_seconds"]
        if total_duration_seconds is None:
            total_duration_seconds = _duration_hint_from_payload(audio_concat_payload)
        if total_duration_seconds is None:
            total_duration_seconds = _duration_hint_from_linx(linx)
        if total_duration_seconds is None and uses_input_audio:
            total_duration_seconds = _audio_duration_seconds(audio)
        if total_duration_seconds is None:
            payload_data = _parse_payload(plan_payload)
            total_frames_hint = _to_int(payload_data, "total_frames", 0)
            total_duration_seconds = float(total_frames_hint) / max(0.001, float(fps_value)) if total_frames_hint > 0 else 0.0

        planner_node = _node_class("IAMCCS_SegmentPlanner")()
        audio_math_node = _node_class("IAMCCS_AudioExtensionMath")()
        audio_extender_node = _node_class("IAMCCS_AudioExtender")()
        audio_gate_node = _node_class("IAMCCS_AudioTimelineGate")()
        extension_node = _node_class("IAMCCS_LTX2_ExtensionModule_Disk")()
        start_inject_node = _node_class("IAMCCS_StartDirToVideoLatent")()
        vae_decode_node = _node_class("IAMCCS_VAEDecodeToDisk")()
        generated_audio_latent_node = _node_class("LTXVEmptyLatentAudio")() if generates_audio else None
        generated_audio_decode_node = _node_class("LTXVAudioVAEDecode")() if generates_audio else None
        vram_flush_node = _node_class("IAMCCS_VRAMFlushLatent")() if bool(vram_flush) else None

        positive_text = str(positive_text or "").strip()
        negative_text = str(negative_text or "").strip()
        negative_source = "frontend"
        if not positive_text:
            _pipeline_debug_step(pipeline_debug, "error_empty_prompt", positive_text=positive_text)
            raise ValueError("positive_text is empty; text/video generation would ignore the prompt")
        prompt_excerpt = _prompt_excerpt(positive_text)
        _pipeline_debug_step(pipeline_debug, "prompt_route", positive=prompt_excerpt, negative_chars=len(negative_text), negative_source=negative_source)
        print(
            "[IAMCCS SuperNodes Render] "
            f"prompt_route generation_type={generation_type} media={effective_media_mode} "
            f"positive=\"{prompt_excerpt}\""
        )
        positive = comfy_nodes.CLIPTextEncode().encode(clip, positive_text)[0]
        negative = comfy_nodes.CLIPTextEncode().encode(clip, negative_text)[0]
        _pipeline_debug_step(pipeline_debug, "prompt_encoded", positive=positive, negative=negative)

        run_root = _resolve_output_path(output_root)
        run_name = f"run_{unique_id or 'manual'}_{int(seed)}"
        run_dir = os.path.join(run_root, run_name)
        segments_dir = os.path.join(run_dir, "segments")
        extended_dir = os.path.join(run_dir, "extended")
        start_dir = os.path.join(run_dir, "start")
        _ensure_clean_dir(run_dir)
        os.makedirs(segments_dir, exist_ok=True)
        _pipeline_debug_step(pipeline_debug, "run_dirs_ready", run_dir=run_dir, segments_dir=segments_dir, extended_dir=extended_dir, start_dir=start_dir)

        fps_value = float(planner_settings["fps"])
        segment_seconds_value = float(planner_settings["segment_seconds"])
        overlap_frames_value = int(planner_settings["overlap_frames"])
        segment_count = None
        cursor_frames = 0
        rendered_segments = 0
        segment_reports = []

        planner_head = planner_node.plan(
            total_duration_seconds,
            fps_value,
            segment_seconds_value,
            planner_settings["planning_mode"],
            planner_settings["segment_preset"],
            overlap_frames_value,
            planner_settings["ltx_round_mode"],
            0,
        )
        planner_segment_count = int(planner_head[4])
        recommended_left_context = float(planner_head[19])
        if float(audio_left_context_s) <= 0.0:
            audio_left_context_s = recommended_left_context
        requested_backend_mode = backend_mode
        backend_mode, segment_count, use_single_best, use_in_memory_loop, modular_decode = _resolve_backend_route(
            requested_backend_mode,
            planner_segment_count,
            modular_decode,
        )
        legacy_audio_img2vid_backend = reference_audio_img2vid and _is_legacy_audio_img2vid_backend(backend_mode)
        legacy_planner_adapter = bool(legacy_audio_img2vid_backend)
        ti2v_incremental_backend = (
            str(backend_mode) == "ti2v_incremental_advanced"
            and not bool(uses_input_audio)
            and str(generation_type) in {"img2video", "text2video"}
        )
        _pipeline_debug_step(
            pipeline_debug,
            "planner_route",
            total_duration=total_duration_seconds,
            fps=fps_value,
            segment_seconds=segment_seconds_value,
            planner_segment_count=planner_segment_count,
            backend_requested=requested_backend_mode,
            backend_resolved=backend_mode,
            segment_count=segment_count,
            use_single_best=use_single_best,
            use_in_memory_loop=use_in_memory_loop,
            modular_decode=modular_decode,
            recommended_left_context=recommended_left_context,
            legacy_planner_adapter=legacy_planner_adapter,
            planner_audio_img2vid_backend=planner_settings.get("audio_img2vid_backend_family"),
            planner_audio_img2vid_mode=planner_settings.get("audio_img2vid_mode"),
            planner_render_backend=planner_settings.get("audio_img2vid_render_backend"),
        )

        planner_report_line = (
            f"Planner settings used. mode={planner_settings['planning_mode']} | "
            f"segment_preset={planner_settings['segment_preset']} | "
            f"segment_seconds={float(planner_settings['segment_seconds']):.3f}s | "
            f"overlap={int(planner_settings['overlap_frames'])}f | "
            f"ltx_round={planner_settings['ltx_round_mode']} | "
            f"a2i_backend={planner_settings.get('audio_img2vid_backend_family', 'modern')}/"
            f"{planner_settings.get('audio_img2vid_mode', 'single')}->"
            f"{planner_settings.get('audio_img2vid_render_backend', backend_mode)}"
        )
        if str(planner_settings["planning_mode"]) == "explicit_preset_seconds":
            planner_report_line += " | explicit_preset_seconds overrides segment_seconds with the selected 5/10/15/20 second preset"
        if legacy_planner_adapter:
            planner_report_line += " | legacy planner adapter: using legacy-compatible fps/segment_seconds/overlap/ltx_round/segment_count only"

        conditioned_positive, conditioned_negative = LTXVConditioning.execute(positive, negative, fps_value)
        decode_settings = _decode_settings("low_ram_disk")
        internal_decode_image_format = "jpg" if legacy_audio_img2vid_backend else "png"
        internal_decode_jpg_quality = 95 if legacy_audio_img2vid_backend else 100
        continuity_anchor_mode = str(continuity_anchor_mode or "off")
        if continuity_anchor_mode == "off":
            continuity_settings = {
                "mode": "off",
                "interval": int(anchor_refresh_interval),
                "strength": 0.0,
            }
        else:
            continuity_settings = _continuity_mode_from_payload(
                continuity_payload,
                continuity_anchor_mode,
                anchor_refresh_interval,
                anchor_image_strength,
            )
        anti_drift_mode = str(anti_drift_mode or "off")
        anti_drift_strength = max(0.0, float(anti_drift_strength))
        identity_persistence_strength = max(0.0, float(identity_persistence_strength))
        if str(continuity_settings["mode"]) == "off":
            anti_drift_mode = "off"
            anti_drift_strength = 0.0
            identity_persistence_strength = 0.0
        _pipeline_debug_step(
            pipeline_debug,
            "continuity_resolved",
            anchor_mode=continuity_settings["mode"],
            anchor_interval=continuity_settings["interval"],
            anchor_strength=continuity_settings["strength"],
            anti_drift_mode=anti_drift_mode,
            anti_drift_strength=anti_drift_strength,
            identity_persistence_strength=identity_persistence_strength,
        )
        refresh_source_image = refresh_image if refresh_image is not None else image
        stage2_model_active = _resolve_stage2_model(model, second_stage_model, second_stage_payload)
        stage2_data = _parse_payload(second_stage_payload)
        second_stage_mode = _to_text(stage2_data, "second_stage_mode", "off")
        second_stage_scale_mode = _to_text(stage2_data, "second_stage_scale_mode", "same_resolution_refine")
        second_stage_upscale_model_name = _to_text(
            stage2_data,
            "second_stage_upscale_model",
            _reference_ltx23_upscale_model_name(),
        )
        second_stage_reinject_strength = _to_float(stage2_data, "second_stage_reinject_strength", 0.0)
        second_stage_cfg = _to_float(stage2_data, "second_stage_cfg", 1.0)
        second_stage_manual_sigmas = _to_text(stage2_data, "second_stage_manual_sigmas", "0.909375, 0.725, 0.421875, 0.0")
        second_stage_step_count = _to_int(stage2_data, "second_stage_steps", 3)
        (
            second_stage_mode,
            second_stage_scale_mode,
            second_stage_reinject_strength,
            second_stage_cfg,
            second_stage_manual_sigmas,
            second_stage_step_count,
            reference_stage2_auto,
        ) = _apply_reference_stage2_defaults(
            generation_mode,
            effective_media_mode,
            second_stage_mode,
            second_stage_scale_mode,
            second_stage_reinject_strength,
            second_stage_cfg,
            second_stage_manual_sigmas,
            second_stage_step_count,
        )
        if (
            str(second_stage_mode) != "off"
            and _to_text(stage2_data, "stage2_model_policy", "stage2_model_if_connected") == "stage2_model_if_connected"
            and second_stage_model is None
            and not ti2v_incremental_backend
        ):
            second_stage_mode = "off"
            second_stage_scale_mode = "same_resolution_refine"
            reference_stage2_auto = ""
        ti2v_incremental_final_width = None
        ti2v_incremental_final_height = None
        if ti2v_incremental_backend:
            second_stage_mode = "latent_upscale_refine_x2_beta"
            second_stage_scale_mode = "x2_latent_upscale_beta"
            second_stage_upscale_model_name = _reference_ltx23_upscale_model_name()
            second_stage_reinject_strength = 0.0 if str(generation_mode) == "t2v" else (1.0 if float(second_stage_reinject_strength) <= 0.0 else float(second_stage_reinject_strength))
            second_stage_cfg = float(second_stage_cfg or 1.0)
            second_stage_manual_sigmas = second_stage_manual_sigmas or "0.909375, 0.725, 0.421875, 0.0"
            reference_stage2_auto = "ti2v_incremental_advanced"
        second_stage_model_source = "stage2_model" if second_stage_model is not None and stage2_model_active is second_stage_model else "stage1_model"
        _pipeline_debug_step(
            pipeline_debug,
            "second_stage_resolved",
            mode=second_stage_mode,
            scale_mode=second_stage_scale_mode,
            reinject_strength=second_stage_reinject_strength,
            cfg=second_stage_cfg,
            steps=second_stage_step_count,
            model_source=second_stage_model_source,
            model_connected=second_stage_model is not None,
            auto_reference=reference_stage2_auto,
        )
        identity_reference_latent = None
        rolling_reference_latent = None

        if use_single_best:
            payload_data = _parse_payload(plan_payload)
            frame_round_mode = _to_text(payload_data, "ltx_round_mode", "up")
            planner_total_frames_raw = max(1, _to_int(payload_data, "total_frames", int(planner_head[0]) or 1))
            planner_total_frames = _ltx_compatible_frame_count(planner_total_frames_raw, frame_round_mode)
            audio_total_frames_with_tail_raw = max(
                0,
                _to_int(
                    payload_data,
                    "audio_duration_frames_with_tail",
                    _to_int(
                        {"audio_duration_frames_with_tail": linx_output(linx, "audio_duration_frames_with_tail", None)},
                        "audio_duration_frames_with_tail",
                        0,
                    ),
                ),
            )
            audio_total_frames_with_tail = _ltx_compatible_frame_count(audio_total_frames_with_tail_raw, frame_round_mode) if audio_total_frames_with_tail_raw else 0
            total_frames_raw = max(1, planner_total_frames_raw, audio_total_frames_with_tail_raw)
            total_frames = _ltx_compatible_frame_count(total_frames_raw, frame_round_mode)
            if reference_audio_img2vid:
                reference_uses_direct_frame_math = abs(float(fps_value) - float(_REFERENCE_AUDIO_IMG2VID_FPS)) <= 0.001
                reference_planner_frames_raw = max(1, int((float(total_duration_seconds) * float(fps_value)) + 1.0))
                reference_audio_frames_raw = 0
                reference_audio_frames_source = "missing"
                reference_duration_audio = raw_audio or audio or single_conditioning_audio
                if reference_duration_audio is not None:
                    reference_audio_frames_raw, reference_audio_frames_source = _compute_audio_duration_frames_with_tail(
                        reference_duration_audio,
                        fps_value,
                    )
                planner_total_frames_raw = int(reference_planner_frames_raw)
                audio_total_frames_with_tail_raw = int(reference_audio_frames_raw)
                total_frames_raw = max(1, audio_total_frames_with_tail_raw or planner_total_frames_raw)
                if reference_uses_direct_frame_math:
                    frame_round_mode = "reference_25fps_formula_no_ltx_round"
                    planner_total_frames = int(reference_planner_frames_raw)
                    audio_total_frames_with_tail = int(reference_audio_frames_raw)
                    total_frames = int(total_frames_raw)
                else:
                    compatible_round_mode = _to_text(payload_data, "ltx_round_mode", frame_round_mode) or "up"
                    frame_round_mode = f"{compatible_round_mode}_ltx_8n_plus_1"
                    planner_total_frames = _ltx_compatible_frame_count(reference_planner_frames_raw, compatible_round_mode)
                    audio_total_frames_with_tail = _ltx_compatible_frame_count(reference_audio_frames_raw, compatible_round_mode) if reference_audio_frames_raw else 0
                    total_frames = _ltx_compatible_frame_count(total_frames_raw, compatible_round_mode)
                _pipeline_debug_step(
                    pipeline_debug,
                    "single_reference_audio_img2vid_duration",
                    fps=fps_value,
                    duration_seconds=total_duration_seconds,
                    planner_formula_frames=reference_planner_frames_raw,
                    audio_formula_frames=reference_audio_frames_raw,
                    audio_formula_source=reference_audio_frames_source,
                    ltx_8n_plus_1=not reference_uses_direct_frame_math,
                    chosen_total_frames=total_frames,
                )
            _pipeline_debug_step(
                pipeline_debug,
                "single_duration_protection",
                frame_round_mode=frame_round_mode,
                planner_total_frames_raw=planner_total_frames_raw,
                planner_total_frames=planner_total_frames,
                audio_total_frames_with_tail_raw=audio_total_frames_with_tail_raw,
                audio_total_frames_with_tail=audio_total_frames_with_tail,
                chosen_total_frames_raw=total_frames_raw,
                chosen_total_frames=total_frames,
            )
            if reference_audio_img2vid:
                effective_image_strength = float(image_strength)
                _pipeline_debug_step(
                    pipeline_debug,
                    "strict_reference_single_enabled",
                    contract="legacy_exact_single" if legacy_audio_img2vid_backend else "workflow_1_AU_IMG2VID_SINGLE",
                    image_strength=effective_image_strength,
                    motion_intensity_ignored=motion_intensity,
                    model_sampling="legacy_single:no_ModelSamplingLTXV" if legacy_audio_img2vid_backend else "none_like_reference_canvas",
                    scheduler="ManualSigmas" if legacy_audio_img2vid_backend else "BasicScheduler(simple)",
                    sampler_node="SamplerCustomAdvanced" if legacy_audio_img2vid_backend else "IAMCCS_SamplerAdvancedVersion1",
                )
            else:
                effective_image_strength = _motion_guidance_strength(image_strength, motion_intensity)
            guidance_image = None
            resize_report = "image_resize=not_used"
            latent_width = int(width)
            latent_height = int(height)
            if ti2v_incremental_backend:
                ti2v_incremental_final_width = _round_up_to_multiple(int(width), 32)
                ti2v_incremental_final_height = _round_up_to_multiple(int(height), 32)
                latent_width = max(64, int(round(float(ti2v_incremental_final_width) / 2.0)))
                latent_height = max(64, int(round(float(ti2v_incremental_final_height) / 2.0)))
                width = int(latent_width)
                height = int(latent_height)
                if str(generation_mode) != "t2v":
                    guidance_image = _resize_image_to(image, int(ti2v_incremental_final_width), int(ti2v_incremental_final_height))
                    resize_report = (
                        "ti2v_incremental_advanced:first_pass_half_res | "
                        f"final={int(ti2v_incremental_final_width)}x{int(ti2v_incremental_final_height)} | "
                        f"first={int(width)}x{int(height)}"
                    )
                _pipeline_debug_step(
                    pipeline_debug,
                    "ti2v_incremental_resolution",
                    final_width=ti2v_incremental_final_width,
                    final_height=ti2v_incremental_final_height,
                    first_pass_width=width,
                    first_pass_height=height,
                    second_stage="latent_upscale_x2",
                )
            if str(generation_mode) != "t2v" and reference_audio_img2vid and legacy_audio_img2vid_backend:
                (
                    legacy_requested_width,
                    legacy_requested_height,
                    legacy_resolution_fix,
                ) = _normalize_reference_audio_img2vid_requested_resolution(width, height)
                guidance_image, latent_width, latent_height, resize_report = _resize_image_like_reference_audio_img2vid(
                    image,
                    legacy_requested_width,
                    legacy_requested_height,
                )
                if legacy_resolution_fix:
                    resize_report = f"{resize_report} | {legacy_resolution_fix}"
                width = int(latent_width)
                height = int(latent_height)
                _pipeline_debug_step(
                    pipeline_debug,
                    "single_legacy_exact_image_resized",
                    source_image=image,
                    resized_image=guidance_image,
                    requested_width=int(legacy_requested_width),
                    requested_height=int(legacy_requested_height),
                    actual_width=int(width),
                    actual_height=int(height),
                    resize_report=resize_report,
                    contract_version=_LEGACY_AUDIO_IMG2VID_CONTRACT_VERSION,
                )
            elif str(generation_mode) != "t2v" and reference_audio_img2vid:
                # The frontend is the source of truth for resolution; the lipsync fix is
                # isolated to the audio-shaped noise mask below.
                (
                    reference_requested_width,
                    reference_requested_height,
                    reference_resolution_fix,
                ) = _normalize_reference_audio_img2vid_requested_resolution(width, height)
                guidance_image, latent_width, latent_height, resize_report = _resize_image_like_reference_audio_img2vid(
                    image,
                    reference_requested_width,
                    reference_requested_height,
                )
                if reference_resolution_fix:
                    resize_report = f"{resize_report} | {reference_resolution_fix}"
                width = int(latent_width)
                height = int(latent_height)
                _pipeline_debug_step(
                    pipeline_debug,
                    "single_reference_best_image_resized",
                    source_image=image,
                    resized_image=guidance_image,
                    requested_width=reference_requested_width,
                    requested_height=reference_requested_height,
                    actual_width=width,
                    actual_height=height,
                    resize_report=resize_report,
                )
            video_latent = EmptyLTXVLatentVideo.execute(int(width), int(height), total_frames, 1)[0]
            _pipeline_debug_step(pipeline_debug, "single_video_latent_created", video_latent=video_latent, width=width, height=height, frames=total_frames)
            if str(generation_mode) != "t2v":
                if guidance_image is None:
                    guidance_image = image
                preprocessed_image = LTXVPreprocess.execute(guidance_image, int(image_compression))[0]
                video_latent = LTXVImgToVideoInplace.execute(vae, preprocessed_image, video_latent, float(effective_image_strength), False)[0]
                _pipeline_debug_step(pipeline_debug, "single_image_conditioning_applied", image=guidance_image, image_compression=image_compression, image_strength=effective_image_strength, image_resize=resize_report, video_latent=video_latent)
            else:
                _pipeline_debug_step(pipeline_debug, "single_image_conditioning_skipped", generation_mode=generation_mode)

            if generates_audio:
                audio_latent = generated_audio_latent_node.execute(int(total_frames), max(1, int(round(fps_value))), 1, audio_vae)[0]
                _pipeline_debug_step(pipeline_debug, "single_audio_latent_generated", audio_latent=audio_latent, frames=total_frames, fps=fps_value)
            else:
                audio_latent = LTXVAudioVAEEncode.execute(single_conditioning_audio, audio_vae)[0]
                _pipeline_debug_step(pipeline_debug, "single_audio_latent_encoded", conditioning_audio=single_conditioning_audio, audio_latent=audio_latent)
                if reference_audio_img2vid or _is_text_audio2video(generation_type):
                    if legacy_audio_img2vid_backend:
                        audio_latent, audio_mask = _apply_legacy_backend_mask_to_audio_latent(audio_latent)
                        mask_source = "legacy_backend:SolidMask_1024_to_SetLatentNoiseMask"
                        mask_width = 1024
                        mask_height = 1024
                    else:
                        audio_latent, audio_mask, mask_width, mask_height = _apply_workflow_solid_mask_to_audio_latent(
                            audio_latent,
                            width,
                            height,
                        )
                        mask_source = "workflow:SolidMask_from_ImageResizeKJv2_width_height_to_SetLatentNoiseMask"
                    _pipeline_debug_step(
                        pipeline_debug,
                        "single_input_audio_noise_mask_applied",
                        mask=audio_mask,
                        mask_shape=f"solid_mask_{int(mask_width)}x{int(mask_height)}",
                        mask_source=mask_source,
                        audio_latent=audio_latent,
                    )
            _soft_cleanup()
            av_latent = LTXVConcatAVLatent.execute(video_latent, audio_latent)[0]
            effective_sampler = _effective_sampler_name(sampler_name, generation_mode, effective_media_mode, generation_type)
            effective_cfg_value = _effective_cfg(cfg, generation_mode, effective_media_mode)
            if reference_audio_img2vid:
                model_sampling_report = "legacy_exact:no_ModelSamplingLTXV_in_source_workflows" if legacy_audio_img2vid_backend else "strict_reference:no_ModelSamplingLTXV"
                model_for_segment = model
                if legacy_audio_img2vid_backend:
                    sigmas = _manual_sigmas(manual_sigmas or _REFERENCE_MANUAL_SIGMAS)
                    sigmas_report = f"legacy_exact_manual_sigmas({int(sigmas.numel())} values)"
                else:
                    sigmas = _scheduler_sigmas(model, "simple", max(1, int(steps or 8)), 1.0)
                    sigmas_report = f"strict_reference_basic_scheduler(simple,{max(1, int(steps or 8))} steps,denoise=1.0)"
                route_extra_report = "legacy_backend_mask=SolidMask_1024" if legacy_audio_img2vid_backend else "mask=workflow_solidmask_from_resize_dims"
                _pipeline_debug_step(
                    pipeline_debug,
                    "single_av_latent_concat",
                    av_latent=av_latent,
                    sampler=effective_sampler,
                    cfg=effective_cfg_value,
                    model_sampling=model_sampling_report,
                    route_contract="legacy_exact_single" if legacy_audio_img2vid_backend else "strict_workflow1_canvas_reference",
                )
            else:
                accelerated_model, accelerator_report = _accelerate_exec_model_if_available(model)
                model_sampling_report = f"ModelSamplingLTXV(max_shift={float(max_shift):.3f},base_shift={float(base_shift):.3f})"
                model_for_segment = ModelSamplingLTXV.execute(accelerated_model, float(max_shift), float(base_shift), av_latent)[0]
                sigmas, sigmas_report = _main_sigmas(
                    uses_input_audio,
                    steps,
                    max_shift,
                    base_shift,
                    sigma_terminal,
                    av_latent,
                    generation_mode,
                    effective_media_mode,
                    manual_sigmas,
                    generation_type,
                    model_for_segment,
                )
                route_extra_report = accelerator_report
                _pipeline_debug_step(
                    pipeline_debug,
                    "single_av_latent_concat",
                    av_latent=av_latent,
                    sampler=effective_sampler,
                    cfg=effective_cfg_value,
                    accelerator=accelerator_report,
                    model_sampling=model_sampling_report,
                )
            guider = CFGGuider.execute(model_for_segment, conditioned_positive, conditioned_negative, float(effective_cfg_value))[0]
            sampler = KSamplerSelect.execute(str(effective_sampler))[0]
            sampler_node_name = "SamplerCustomAdvanced"
            reference_sampler_cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("IAMCCS_SamplerAdvancedVersion1") if reference_audio_img2vid else None
            if reference_sampler_cls is not None:
                sampler_node_name = "IAMCCS_SamplerAdvancedVersion1(disable_progress=True,cleanup=True)"
            _pipeline_debug_step(
                pipeline_debug,
                "single_sampler_prepared",
                sampler=effective_sampler,
                sampler_node=sampler_node_name,
                sigmas=sigmas,
                sigmas_report=sigmas_report,
                seed=seed,
            )
            noise = RandomNoise.execute(int(seed))[0]
            if reference_sampler_cls is not None:
                sampled_av = reference_sampler_cls().sample(
                    noise,
                    guider,
                    sampler,
                    sigmas,
                    av_latent,
                    True,
                    True,
                )[0]
            else:
                sampled_av = SamplerCustomAdvanced.sample(
                    noise,
                    guider,
                    sampler,
                    sigmas,
                    av_latent,
                )[0]
            sampled_video, sampled_audio_latent = LTXVSeparateAVLatent.execute(sampled_av)
            if reference_audio_img2vid:
                if legacy_audio_img2vid_backend:
                    crop_guides_report = "legacy_exact:skipped_no_LTXVCropGuides_in_source_workflows"
                else:
                    crop_guides_report = "skipped_best_workflow_no_LTXVCropGuides_after_sampler"
            else:
                sampled_video = LTXVCropGuides.execute(conditioned_positive, conditioned_negative, sampled_video)[2]
                crop_guides_report = "applied"
            _pipeline_debug_step(
                pipeline_debug,
                "single_sampled",
                sampler_node=sampler_node_name,
                crop_guides=crop_guides_report,
                sampled_video=sampled_video,
                sampled_audio_latent=sampled_audio_latent,
            )

            stage2_model_active = _resolve_stage2_model(model, second_stage_model, second_stage_payload)
            stage2_data = _parse_payload(second_stage_payload)
            second_stage_mode = _to_text(stage2_data, "second_stage_mode", "off")
            second_stage_scale_mode = _to_text(stage2_data, "second_stage_scale_mode", "same_resolution_refine")
            second_stage_upscale_model_name = _to_text(
                stage2_data,
                "second_stage_upscale_model",
                _reference_ltx23_upscale_model_name(),
            )
            second_stage_reinject_strength = _to_float(stage2_data, "second_stage_reinject_strength", 0.0)
            second_stage_cfg = _to_float(stage2_data, "second_stage_cfg", 1.0)
            second_stage_manual_sigmas = _to_text(stage2_data, "second_stage_manual_sigmas", "0.909375, 0.725, 0.421875, 0.0")
            second_stage_step_count = _to_int(stage2_data, "second_stage_steps", 3)
            (
                second_stage_mode,
                second_stage_scale_mode,
                second_stage_reinject_strength,
                second_stage_cfg,
                second_stage_manual_sigmas,
                second_stage_step_count,
                reference_stage2_auto,
            ) = _apply_reference_stage2_defaults(
                generation_mode,
                effective_media_mode,
                second_stage_mode,
                second_stage_scale_mode,
                second_stage_reinject_strength,
                second_stage_cfg,
                second_stage_manual_sigmas,
                second_stage_step_count,
            )
            if (
                str(second_stage_mode) != "off"
                and _to_text(stage2_data, "stage2_model_policy", "stage2_model_if_connected") == "stage2_model_if_connected"
                and second_stage_model is None
                and not ti2v_incremental_backend
            ):
                second_stage_mode = "off"
                second_stage_scale_mode = "same_resolution_refine"
                reference_stage2_auto = ""
            if ti2v_incremental_backend:
                second_stage_mode = "latent_upscale_refine_x2_beta"
                second_stage_scale_mode = "x2_latent_upscale_beta"
                second_stage_upscale_model_name = _reference_ltx23_upscale_model_name()
                second_stage_reinject_strength = 0.0 if str(generation_mode) == "t2v" else (1.0 if float(second_stage_reinject_strength) <= 0.0 else float(second_stage_reinject_strength))
                second_stage_cfg = float(second_stage_cfg or 1.0)
                second_stage_manual_sigmas = second_stage_manual_sigmas or "0.909375, 0.725, 0.421875, 0.0"
                reference_stage2_auto = "ti2v_incremental_advanced"
            second_stage_model_source = "stage2_model" if second_stage_model is not None and stage2_model_active is second_stage_model else "stage1_model"
            _pipeline_debug_step(
                pipeline_debug,
                "single_second_stage_resolved",
                mode=second_stage_mode,
                scale_mode=second_stage_scale_mode,
                reinject_strength=second_stage_reinject_strength,
                cfg=second_stage_cfg,
                steps=second_stage_step_count,
                model_source=second_stage_model_source,
            )

            if str(second_stage_mode) in {"latent_refine_3step", "latent_upscale_refine", "latent_upscale_refine_x2_beta"}:
                stage2_positive, stage2_negative, cropped_video_latent = LTXVCropGuides.execute(
                    conditioned_positive,
                    conditioned_negative,
                    sampled_video,
                )
                if str(second_stage_scale_mode) == "x2_latent_upscale_beta":
                    upscale_model = LatentUpscaleModelLoader.execute(str(second_stage_upscale_model_name))[0]
                    stage2_video_latent = LTXVLatentUpsampler().upsample_latent(
                        cropped_video_latent,
                        upscale_model,
                        vae,
                    )[0]
                else:
                    stage2_video_latent = cropped_video_latent
                if float(second_stage_reinject_strength) > 0.0 and str(generation_mode) != "t2v":
                    reinject_width = int(ti2v_incremental_final_width or width)
                    reinject_height = int(ti2v_incremental_final_height or height)
                    resized_guidance_image = _resize_image_to(image, reinject_width, reinject_height)
                    preprocessed_guidance = LTXVPreprocess.execute(resized_guidance_image, int(image_compression))[0]
                    reinjected_video_latent = LTXVImgToVideoInplace.execute(
                        vae,
                        preprocessed_guidance,
                        stage2_video_latent,
                        float(second_stage_reinject_strength),
                        False,
                    )[0]
                else:
                    reinjected_video_latent = stage2_video_latent
                latent_stage2 = LTXVConcatAVLatent.execute(reinjected_video_latent, sampled_audio_latent)[0]
                if vram_flush_node is not None:
                    latent_stage2 = vram_flush_node.run(latent_stage2)[0]
                model_stage2 = ModelSamplingLTXV.execute(stage2_model_active, float(max_shift), float(base_shift), latent_stage2)[0]
                guider_stage2 = CFGGuider.execute(model_stage2, stage2_positive, stage2_negative, float(second_stage_cfg))[0]
                sampler_stage2 = KSamplerSelect.execute(str(effective_sampler))[0]
                sigmas_stage2 = _manual_sigmas(second_stage_manual_sigmas)
                noise_stage2 = RandomNoise.execute(int(seed))[0]
                sampled_stage2_av = SamplerCustomAdvanced.sample(
                    noise_stage2,
                    guider_stage2,
                    sampler_stage2,
                    sigmas_stage2,
                    latent_stage2,
                )[0]
                sampled_video, sampled_audio_latent = LTXVSeparateAVLatent.execute(sampled_stage2_av)
                _pipeline_debug_step(
                    pipeline_debug,
                    "single_second_stage_applied",
                    mode=second_stage_mode,
                    scale_mode=second_stage_scale_mode,
                    sigmas_stage2=sigmas_stage2,
                    sampled_video=sampled_video,
                    sampled_audio_latent=sampled_audio_latent,
                )
            else:
                _pipeline_debug_step(pipeline_debug, "single_second_stage_skipped", mode=second_stage_mode)

            if uses_input_audio:
                rendered_audio = raw_audio
                _pipeline_debug_step(pipeline_debug, "single_audio_export_input_passthrough", rendered_audio=rendered_audio)
            elif generates_audio and exports_audio:
                rendered_audio = generated_audio_decode_node.execute(sampled_audio_latent, audio_vae)[0]
                _pipeline_debug_step(pipeline_debug, "single_audio_export_generated", rendered_audio=rendered_audio)
            else:
                rendered_audio = None
                _pipeline_debug_step(pipeline_debug, "single_audio_export_off", exports_audio=exports_audio, generates_audio=generates_audio)

            disable_vae_frame_align = bool(reference_audio_img2vid)
            single_debug_report = _pipeline_debug_text(pipeline_debug)
            report = (
                f"duration {float(total_duration_seconds):.2f}s | fps {float(fps_value):.2f} | total {int(total_frames)}f | segments 1 | "
                f"backend_requested={requested_backend_mode} | backend_resolved={backend_mode} | media_mode={effective_media_mode} | generation_mode {generation_mode} | "
                f"stage2 {('auto_' + str(reference_stage2_auto) + '_refine') if reference_stage2_auto else ('on' if str(second_stage_mode) != 'off' else 'off')} | decode_mode={modular_decode} | "
                f"conditioning {'generated_audio_empty_latent' if generates_audio else conditioning_audio_source} | audio_export={'yes' if rendered_audio is not None else 'no'} | "
                f"melband_enabled={melband_enabled}\n"
                f"Prompt route. positive=\"{prompt_excerpt}\"\n"
                f"Planner settings used. mode={planner_settings['planning_mode']} | segment_preset={planner_settings['segment_preset']} | "
                f"segment_seconds={float(planner_settings['segment_seconds']):.3f}s | overlap={int(planner_settings['overlap_frames'])}f | "
                f"ltx_round={planner_settings['ltx_round_mode']}\n"
                f"Single duration protection. planner_total={int(planner_total_frames)}f | audio_total_with_tail={int(audio_total_frames_with_tail)}f | chosen_total={int(total_frames)}f\n"
                f"Audio preprocess. {audio_preprocess_report}\n"
                f"Single route details. sampler={effective_sampler} | cfg={float(effective_cfg_value):.3f} | sigmas={sigmas_report} | sampler_node={sampler_node_name} | cleanup_before_sampling=soft_cleanup | model_sampling={model_sampling_report} | motion_intensity={'ignored_strict_reference' if reference_audio_img2vid else f'{motion_intensity:.2f}'} | vram_flush={'on' if bool(vram_flush) else 'off'} | vae_frame_align={'off_official_audio_img2vid' if disable_vae_frame_align else 'on'} | {route_extra_report}\n"
                f"Executable AU+IMG2VID render completed. single generation backend={'ti2v_incremental_advanced' if ti2v_incremental_backend else ('legacy_single' if legacy_audio_img2vid_backend else ('strict_workflow1_canvas_reference' if reference_audio_img2vid else 'workflow1_best'))} | latent handed to VAE stage"
            )
            report = _append_debug_report(report, single_debug_report)
            render_linx = build_stage_linx_payload(
                linx,
                "exec_render",
                "render",
                {
                    "pipeline_kind": "au_img2vid_exec",
                    "backend_mode": str(backend_mode),
                    "media_mode": str(effective_media_mode),
                    "generation_mode": str(generation_mode),
                    "fps": float(fps_value),
                    "modular_decode": str(modular_decode),
                  "motion_intensity": float(motion_intensity),
                  "generated_media_duration_seconds": float(generated_media_duration_seconds),
                  "generated_media_fps": float(fps_value),
                  "generation_type": str(generation_type),
                    "debug_verbose": bool(debug_verbose),
                    "target_frame_count": int(total_frames),
                    "target_frame_count_source": "render_single_planner_audio" if uses_input_audio else "render_single_generated_duration",
                    "disable_vae_frame_align": disable_vae_frame_align,
                    "vram_flush": bool(vram_flush),
                    "segment_count": 1,
                    "segments_rendered": 1,
                    "second_stage_mode": str(second_stage_mode),
                    "second_stage_scale_mode": str(second_stage_scale_mode),
                    "second_stage_steps": int(second_stage_step_count),
                    "second_stage_model_source": str(second_stage_model_source),
                    "pipeline_debug_steps": len(pipeline_debug.get("lines") or []),
                },
                report,
                unique_id=unique_id,
                requires={
                    "resources": {"model": "MODEL", "clip": "CLIP", "vae": "VAE", "audio_vae": "VAE", "planner_payload": "STRING", "fps": "FLOAT"},
                    "input_audio_modes": {"audio": "AUDIO", "audio_ltx_conditioning": "AUDIO", "audio_conditioning_single": "AUDIO", "audio_conditioning_segmented": "AUDIO"},
                    "pure_or_generated_modes": {"audio": "optional final audio only"},
                },
                slot_map={
                    "frames_dir": {"type": "STRING", "role": "rendered_frames_dir"},
                    "start_dir": {"type": "STRING", "role": "rendered_start_dir"},
                    "linx": {"type": SUPERNODE_LINX_TYPE, "role": "stage_linx"},
                },
                downstream_stages=_downstream_stage_hints(downstream_stage_mode),
                policies={
                    "decode_mode": str(modular_decode),
                    "second_stage_mode": str(second_stage_mode),
                },
                outputs={
                    "frames_dir": "",
                    "start_dir": "",
                    "segments_rendered": 1,
                    "estimated_duration_seconds": float(total_duration_seconds),
                    "render_status": _first_line(report),
                    "backend_mode": str(backend_mode),
                    "render_backend_mode": str(backend_mode),
                    "decode_mode": str(modular_decode),
                    "media_mode": str(effective_media_mode),
                    "generation_type": str(generation_type),
                    "target_frame_count": int(total_frames),
                    "target_frame_count_source": "render_single_planner_audio" if uses_input_audio else "render_single_generated_duration",
                    "disable_vae_frame_align": disable_vae_frame_align,
                    "pipeline_debug": list(pipeline_debug.get("lines") or []),
                    "debug_verbose": bool(debug_verbose),
                },
                resources={
                    "audio": rendered_audio if exports_audio else None,
                    "model": model,
                    "clip": clip,
                    "vae": vae,
                    "audio_vae": audio_vae,
                    "fps": float(fps_value),
                    "decode_mode": str(modular_decode),
                    "output_root": str(output_root),
                    "planner_payload": str(plan_payload),
                    "media_mode": str(effective_media_mode),
                    "generation_mode": str(generation_mode),
                    "target_frame_count": int(total_frames),
                    "target_frame_count_source": "render_single_planner_audio" if uses_input_audio else "render_single_generated_duration",
                    "disable_vae_frame_align": disable_vae_frame_align,
                    "debug_verbose": bool(debug_verbose),
                    "video_latent": sampled_video,
                    "rendered_images": None,
                    "second_stage_model": stage2_model_active,
                    "second_stage_payload": second_stage_payload,
                },
            )
            return ("", "", 1, float(total_duration_seconds), render_linx, report)

        extension_node_mem = _node_class("IAMCCS_LTX2_ExtensionModule")() if use_in_memory_loop else None
        start_inject_images_node = _node_class("IAMCCS_StartImagesToVideoLatent")() if use_in_memory_loop else None
        source_anchor_inject_node = _node_class("IAMCCS_StartImagesToVideoLatent")()
        current_extended_images = None
        current_start_images = None
        in_memory_stitch_preset = str(stitch_preset or "custom")
        if in_memory_stitch_preset in {"lossless_refresh_24fps", "lossless_refresh_strong_24fps", "videoclip_audio_24fps", "monologue_audio_24fps"}:
            in_memory_stitch_preset = "custom"
        effective_sampler = _effective_sampler_name(sampler_name, generation_mode, effective_media_mode, generation_type)
        effective_cfg_value = _effective_cfg(cfg, generation_mode, effective_media_mode)
        if reference_audio_img2vid and str(generation_mode) != "t2v" and legacy_audio_img2vid_backend:
            original_image = image
            (
                legacy_requested_width,
                legacy_requested_height,
                legacy_resolution_fix,
            ) = _normalize_reference_audio_img2vid_requested_resolution(width, height)
            resized_image, latent_width, latent_height, resize_report = _resize_image_like_reference_audio_img2vid(
                image,
                legacy_requested_width,
                legacy_requested_height,
            )
            width = int(latent_width)
            height = int(latent_height)
            image = resized_image
            loop_resize_report = f"legacy_exact_resize_before_LTXVPreprocess | {resize_report}"
            if legacy_resolution_fix:
                loop_resize_report = f"{loop_resize_report} | {legacy_resolution_fix}"
            refresh_resize_report = "legacy_github:refresh_source=main_image"
            if refresh_image is None:
                refresh_source_image = resized_image
            else:
                refresh_source_image, _, _, refresh_resize_report = _resize_image_like_reference_audio_img2vid(
                    refresh_source_image,
                    width,
                    height,
                )
            _pipeline_debug_step(
                pipeline_debug,
                "loop_legacy_image_prepared",
                source_image=original_image,
                resized_image=resized_image,
                requested_width=int(legacy_requested_width),
                requested_height=int(legacy_requested_height),
                actual_width=int(latent_width),
                actual_height=int(latent_height),
                resize_report=loop_resize_report,
                refresh_resize_report=refresh_resize_report,
                contract_version=_LEGACY_AUDIO_IMG2VID_CONTRACT_VERSION,
            )
        elif reference_audio_img2vid and str(generation_mode) != "t2v":
            (
                loop_requested_width,
                loop_requested_height,
                loop_resolution_fix,
            ) = _normalize_reference_audio_img2vid_requested_resolution(width, height)
            original_image = image
            resized_image, latent_width, latent_height, loop_resize_report = _resize_image_like_reference_audio_img2vid(
                image,
                loop_requested_width,
                loop_requested_height,
            )
            if loop_resolution_fix:
                loop_resize_report = f"{loop_resize_report} | {loop_resolution_fix}"
            width = int(latent_width)
            height = int(latent_height)
            image = resized_image
            if refresh_image is None:
                refresh_source_image = resized_image
                refresh_resize_report = "refresh_source=main_image"
            else:
                refresh_source_image, _, _, refresh_resize_report = _resize_image_like_reference_audio_img2vid(
                    refresh_source_image,
                    width,
                    height,
                )
            _pipeline_debug_step(
                pipeline_debug,
                "loop_reference_image_resized",
                source_image=original_image,
                resized_image=resized_image,
                requested_width=loop_requested_width,
                requested_height=loop_requested_height,
                actual_width=width,
                actual_height=height,
                resize_report=loop_resize_report,
                refresh_resize_report=refresh_resize_report,
                contract_version=_AUDIO_IMG2VID_CONTRACT_VERSION,
            )
        _pipeline_debug_step(
            pipeline_debug,
            "loop_backend_ready",
            segment_count=segment_count,
            use_in_memory_loop=use_in_memory_loop,
            sampler=effective_sampler,
            cfg=effective_cfg_value,
            decode_mode=modular_decode,
            stitch_preset=stitch_preset,
        )

        for segment_index in range(segment_count):
            plan_segment = planner_node.plan(
                total_duration_seconds,
                fps_value,
                segment_seconds_value,
                planner_settings["planning_mode"],
                planner_settings["segment_preset"],
                overlap_frames_value,
                planner_settings["ltx_round_mode"],
                segment_index,
            )
            current_segment_raw_frames = int(plan_segment[9])
            current_segment_unique_frames = int(plan_segment[10])
            generated_frames_for_timeline = current_segment_unique_frames if segment_index == 0 else current_segment_raw_frames

            if uses_input_audio:
                math_out = audio_math_node.compute(
                    audio,
                    fps_value,
                    segment_index,
                    generated_frames_for_timeline,
                    current_segment_unique_frames,
                    True,
                    0,
                    cursor_frames,
                )
                cursor_frames_out = int(math_out[0])
                segment_start_frames = int(math_out[1])
                effective_unique_frames = int(math_out[3])
                remaining_frames_after = int(math_out[7])
                is_last_segment = int(math_out[8])

                workflow_melband_route = bool(
                    reference_audio_img2vid
                    and not legacy_audio_img2vid_backend
                    and planner_audio_preprocess_mode == "melband_vocals_duration_math"
                    and melband_enabled
                )
                post_extender_melband = False
                if legacy_audio_img2vid_backend:
                    extender_source_audio = segmented_audio or raw_audio or melband_single_audio
                    extender_source_label = "legacy:audio_conditioning_segmented_or_raw"
                    segment_audio_context_mode = audio_context_mode
                    segment_audio_left_context_s = float(audio_left_context_s)
                    segment_audio_right_context_s = float(audio_right_context_s)
                elif workflow_melband_route and int(segment_index) == 0:
                    extender_source_audio = raw_audio or segmented_audio
                    extender_source_label = "workflow:seg0_raw_full_audio"
                    post_extender_melband = True
                elif workflow_melband_route:
                    extender_source_audio = melband_single_audio or segmented_audio or raw_audio
                    extender_source_label = "workflow:full_melband_vocals"
                else:
                    extender_source_audio = segmented_audio or raw_audio or melband_single_audio
                    extender_source_label = "raw_audio_only:raw_or_segmented_audio"
                if extender_source_audio is None:
                    raise ValueError("input-audio segment render selected, but no audio reached IAMCCS_AudioExtender")

                if legacy_audio_img2vid_backend:
                    pass
                elif reference_audio_img2vid and int(segment_index) == 0:
                    segment_audio_context_mode = "no_overlap"
                    segment_audio_left_context_s = 0.0
                    segment_audio_right_context_s = 0.0
                else:
                    segment_audio_context_mode = audio_context_mode
                    segment_audio_left_context_s = float(audio_left_context_s)
                    segment_audio_right_context_s = float(audio_right_context_s)

                conditioning_slice = audio_extender_node.slice_segment(
                    extender_source_audio,
                    fps_value,
                    segment_audio_context_mode,
                    segment_audio_left_context_s,
                    segment_audio_right_context_s,
                    "use_timeline_cursor",
                    "snap_to_video_duration",
                    "soft_clamp",
                    segment_index,
                    float(segment_seconds_value),
                    current_segment_unique_frames,
                    generated_frames_for_timeline,
                    current_segment_unique_frames,
                    cursor_frames,
                    segment_start_frames,
                    effective_unique_frames,
                    current_segment_unique_frames,
                )[0]
                if post_extender_melband:
                    try:
                        conditioning_audio, segment_melband_model_name = _extract_melband_vocals(
                            conditioning_slice,
                            planner_melband_model_name,
                        )
                    except Exception as exc:
                        raise RuntimeError(f"segment 0 MelBand vocals extraction failed after AudioExtender: {exc}") from exc
                    conditioning_audio_source = (
                        f"workflow:seg0_raw_audio_extender_then_melband_vocals:{segment_melband_model_name}"
                    )
                else:
                    conditioning_audio = conditioning_slice
                    if legacy_audio_img2vid_backend:
                        conditioning_audio_source = "legacy:audio_extender_then_audio_vae"
                    elif workflow_melband_route:
                        conditioning_audio_source = "workflow:full_melband_vocals_then_audio_extender"
                    else:
                        conditioning_audio_source = "raw_audio_only:audio_extender_then_audio_vae"
            else:
                segment_start_frames = int(cursor_frames)
                effective_unique_frames = int(current_segment_unique_frames)
                cursor_frames_out = min(int(planner_head[0]), int(cursor_frames) + int(effective_unique_frames))
                remaining_frames_after = max(0, int(planner_head[0]) - int(cursor_frames_out))
                is_last_segment = 1 if int(segment_index) >= int(segment_count) - 1 or int(remaining_frames_after) <= 0 else 0
                conditioning_audio = None
                conditioning_slice = None
                conditioning_audio_source = "generated_audio_empty_latent"
                extender_source_label = "none"
                segment_audio_context_mode = "none"
                segment_audio_left_context_s = 0.0
                segment_audio_right_context_s = 0.0
            _pipeline_debug_step(
                pipeline_debug,
                f"segment_{segment_index}_timeline",
                raw_frames=current_segment_raw_frames,
                unique_frames=current_segment_unique_frames,
                generated_frames_for_timeline=generated_frames_for_timeline,
                segment_start_frames=segment_start_frames,
                effective_unique_frames=effective_unique_frames,
                remaining_frames_after=remaining_frames_after,
                is_last_segment=is_last_segment,
                audio_conditioning_route=conditioning_audio_source,
                audio_extender_input=extender_source_label,
                audio_context_mode=segment_audio_context_mode,
                audio_left_context_s=segment_audio_left_context_s,
                audio_right_context_s=segment_audio_right_context_s,
                conditioning_slice=conditioning_slice,
                conditioning_audio=conditioning_audio,
            )

            video_latent = EmptyLTXVLatentVideo.execute(int(width), int(height), current_segment_raw_frames, 1)[0]
            is_t2v = generation_mode == "t2v"
            uses_source_anchor = False if is_t2v else _use_source_anchor(segment_index, continuity_settings["mode"], continuity_settings["interval"])
            uses_tail_source_anchor = False if is_t2v else _use_tail_then_source_anchor(segment_index, continuity_settings["mode"], continuity_settings["interval"])
            if reference_audio_img2vid:
                if legacy_audio_img2vid_backend:
                    effective_image_strength = float(image_strength)
                    effective_anchor_strength = float(continuity_settings["strength"])
                else:
                    effective_image_strength = float(image_strength)
                    effective_anchor_strength = float(continuity_settings["strength"])
            else:
                effective_image_strength = _motion_guidance_strength(image_strength, motion_intensity)
                effective_anchor_strength = _motion_guidance_strength(continuity_settings["strength"], motion_intensity)
            init_mode = "t2v_empty" if is_t2v and segment_index == 0 else "tail"
            tail_refresh_report = "none"
            anchor_refresh_report = "off"
            tail_inject_strength = None
            tail_preprocess_crf = None
            if uses_source_anchor:
                anchor_image = refresh_source_image
                if segment_index > 0 and continuity_settings["mode"] == "periodic_source_refresh":
                    anchor_image = _load_guidance_image_from_dir(start_dir, fallback_image=refresh_source_image, pick_mode="latest")
                preprocessed_image = LTXVPreprocess.execute(anchor_image, int(image_compression))[0]
                source_strength = float(effective_image_strength if segment_index == 0 else effective_anchor_strength)
                video_latent = LTXVImgToVideoInplace.execute(vae, preprocessed_image, video_latent, source_strength, False)[0]
                init_mode = "source"
                anchor_refresh_report = f"source_head:{source_strength:.2f}"
            elif segment_index > 0:
                tail_inject_strength = (
                    float(_LEGACY_EXACT_TAIL_STRENGTH)
                    if legacy_audio_img2vid_backend
                    else float(_REFERENCE_AUDIO_IMG2VID_TAIL_STRENGTH)
                    if reference_audio_img2vid
                    else float(effective_image_strength)
                )
                tail_preprocess_crf = (
                    int(_LEGACY_EXACT_TAIL_PREPROCESS_CRF)
                    if legacy_audio_img2vid_backend
                    else int(_REFERENCE_AUDIO_IMG2VID_TAIL_PREPROCESS_CRF)
                    if reference_audio_img2vid
                    else int(image_compression)
                )
                if use_in_memory_loop:
                    video_latent, _, tail_refresh_report = start_inject_images_node.inject(
                        current_start_images,
                        vae,
                        video_latent,
                        "all",
                        max(1, overlap_frames_value),
                        0,
                        tail_inject_strength,
                        True,
                        tail_preprocess_crf,
                    )
                else:
                    video_latent, _, tail_refresh_report = start_inject_node.inject(
                        start_dir,
                        vae,
                        video_latent,
                        "all",
                        max(1, overlap_frames_value),
                        0,
                        tail_inject_strength,
                        True,
                        tail_preprocess_crf,
                    )
                if uses_tail_source_anchor:
                    protected_tail_slots = _protected_prefix_latent_frames(vae, overlap_frames_value)
                    anchor_insert_pixel_frame = protected_tail_slots * _latent_time_scale_factor(vae)
                    source_strength = float(effective_anchor_strength)
                    video_latent, _, anchor_refresh_report = source_anchor_inject_node.inject(
                        refresh_source_image,
                        vae,
                        video_latent,
                        "from_start",
                        1,
                        int(anchor_insert_pixel_frame),
                        source_strength,
                        False,
                        0,
                    )
                    init_mode = "tail+source_refresh"
            _pipeline_debug_step(
                pipeline_debug,
                f"segment_{segment_index}_video_init",
                video_latent=video_latent,
                init_mode=init_mode,
                source_anchor=uses_source_anchor,
                tail_source_anchor=uses_tail_source_anchor,
                image_strength=effective_image_strength,
                anchor_strength=effective_anchor_strength,
                tail_refresh=tail_refresh_report,
                anchor_refresh=anchor_refresh_report,
                tail_inject_strength=tail_inject_strength,
                tail_preprocess_crf=tail_preprocess_crf,
            )

            if generates_audio:
                audio_latent = generated_audio_latent_node.execute(
                    int(current_segment_raw_frames),
                    max(1, int(round(fps_value))),
                    1,
                    audio_vae,
                )[0]
            else:
                audio_latent = LTXVAudioVAEEncode.execute(conditioning_audio, audio_vae)[0]
                if reference_audio_img2vid:
                    if legacy_audio_img2vid_backend:
                        audio_latent, audio_mask = _apply_legacy_backend_mask_to_audio_latent(audio_latent)
                        mask_width = 1024
                        mask_height = 1024
                        mask_source = "legacy_github:SolidMask_1024_to_SetLatentNoiseMask"
                    else:
                        audio_latent, audio_mask, mask_width, mask_height = _apply_workflow_solid_mask_to_audio_latent(
                            audio_latent,
                            width,
                            height,
                        )
                        mask_source = "workflow:SolidMask_from_ImageResizeKJv2_width_height_to_SetLatentNoiseMask"
                    _pipeline_debug_step(
                        pipeline_debug,
                        f"segment_{segment_index}_reference_audio_noise_mask_applied",
                        mask=audio_mask,
                        mask_shape=f"solid_mask_{int(mask_width)}x{int(mask_height)}",
                        mask_source=mask_source,
                        audio_latent=audio_latent,
                    )
            _pipeline_debug_step(
                pipeline_debug,
                f"segment_{segment_index}_audio_latent",
                mode="generated" if generates_audio else "encoded_input",
                audio_latent=audio_latent,
            )
            segment_positive = conditioned_positive
            segment_negative = conditioned_negative
            _hard_unload_all_models()
            av_latent = LTXVConcatAVLatent.execute(video_latent, audio_latent)[0]
            if reference_audio_img2vid and legacy_audio_img2vid_backend:
                model_for_segment = model
                loop_model_sampling_report = "legacy_exact:no_ModelSamplingLTXV_in_source_workflows"
            elif reference_audio_img2vid:
                model_for_segment = model
                loop_model_sampling_report = "reference_audio_img2vid:no_ModelSamplingLTXV"
            else:
                model_for_segment = ModelSamplingLTXV.execute(model, float(max_shift), float(base_shift), av_latent)[0]
                loop_model_sampling_report = f"ModelSamplingLTXV(max_shift={float(max_shift):.3f},base_shift={float(base_shift):.3f})"
            guider = CFGGuider.execute(model_for_segment, segment_positive, segment_negative, float(effective_cfg_value))[0]
            sampler = KSamplerSelect.execute(str(effective_sampler))[0]
            if legacy_audio_img2vid_backend:
                sigmas = _manual_sigmas(manual_sigmas or _REFERENCE_MANUAL_SIGMAS)
                sigmas_report = f"legacy_exact_manual_sigmas({int(sigmas.numel())} values)"
            else:
                sigmas, sigmas_report = _main_sigmas(
                    uses_input_audio,
                    steps,
                    max_shift,
                    base_shift,
                    sigma_terminal,
                    av_latent,
                    generation_mode,
                    effective_media_mode,
                    manual_sigmas,
                    generation_type,
                    model_for_segment,
                )
            _pipeline_debug_step(
                pipeline_debug,
                f"segment_{segment_index}_sampler_prepared",
                av_latent=av_latent,
                sampler=effective_sampler,
                cfg=effective_cfg_value,
                sigmas=sigmas,
                sigmas_report=sigmas_report,
                seed=int(seed) + segment_index,
                model_sampling=loop_model_sampling_report,
                sampler_node="SamplerCustomAdvanced" if legacy_audio_img2vid_backend else ("IAMCCS_SamplerAdvancedVersion1(disable_progress=True,cleanup=True)" if reference_audio_img2vid and comfy_nodes.NODE_CLASS_MAPPINGS.get("IAMCCS_SamplerAdvancedVersion1") is not None else "SamplerCustomAdvanced"),
            )
            noise = RandomNoise.execute(int(seed) + segment_index)[0]
            reference_sampler_cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("IAMCCS_SamplerAdvancedVersion1") if reference_audio_img2vid and not legacy_audio_img2vid_backend else None
            if reference_sampler_cls is not None:
                sampled_av = reference_sampler_cls().sample(
                    noise,
                    guider,
                    sampler,
                    sigmas,
                    av_latent,
                    True,
                    True,
                )[0]
                loop_sampler_node_name = "IAMCCS_SamplerAdvancedVersion1(disable_progress=True,cleanup=True)"
            else:
                sampled_av = SamplerCustomAdvanced.sample(noise, guider, sampler, sigmas, av_latent)[0]
                loop_sampler_node_name = "SamplerCustomAdvanced"
            sampled_video, sampled_audio_latent = LTXVSeparateAVLatent.execute(sampled_av)
            if reference_audio_img2vid and legacy_audio_img2vid_backend:
                crop_guides_report = "legacy_exact:skipped_no_LTXVCropGuides_in_source_workflows"
            elif reference_audio_img2vid:
                crop_guides_report = "skipped_reference_audio_img2vid"
            else:
                segment_positive, segment_negative, sampled_video = LTXVCropGuides.execute(
                    segment_positive,
                    segment_negative,
                    sampled_video,
                )
                crop_guides_report = "applied"
            _pipeline_debug_step(
                pipeline_debug,
                f"segment_{segment_index}_sampled",
                sampler_node=loop_sampler_node_name,
                crop_guides=crop_guides_report,
                sampled_video=sampled_video,
                sampled_audio_latent=sampled_audio_latent,
            )

            stage_mode = str(second_stage_mode)
            stage2_applied = False
            if stage_mode in {"latent_refine_3step", "latent_upscale_refine", "latent_upscale_refine_x2_beta"}:
                guidance_source = "refresh"
                guidance_image = refresh_source_image
                if uses_tail_source_anchor:
                    guidance_source = "refresh_after_tail"
                elif not uses_source_anchor:
                    guidance_image = _load_guidance_image_from_dir(start_dir, fallback_image=refresh_source_image, pick_mode="latest")
                    guidance_source = "tail"
                stage2_positive, stage2_negative, cropped_video_latent = LTXVCropGuides.execute(
                    conditioned_positive,
                    conditioned_negative,
                    sampled_video,
                )
                if str(second_stage_scale_mode) == "x2_latent_upscale_beta":
                    upscale_model = LatentUpscaleModelLoader.execute(str(second_stage_upscale_model_name))[0]
                    stage2_video_latent = LTXVLatentUpsampler().upsample_latent(
                        cropped_video_latent,
                        upscale_model,
                        vae,
                    )[0]
                else:
                    stage2_video_latent = cropped_video_latent
                reinject_strength = float(second_stage_reinject_strength)
                if uses_tail_source_anchor and segment_index > 0:
                    reinject_strength = 0.0
                elif uses_source_anchor and segment_index > 0:
                    reinject_strength = min(reinject_strength, float(effective_anchor_strength))
                reinject_strength = _motion_guidance_strength(reinject_strength, motion_intensity)
                if reinject_strength > 0.0 and not is_t2v:
                    stage2_width, stage2_height = _pixel_dims_from_latent(stage2_video_latent, vae)
                    resized_guidance_image = _resize_image_to(guidance_image, stage2_width, stage2_height)
                    preprocessed_guidance = LTXVPreprocess.execute(resized_guidance_image, int(image_compression))[0]
                    reinjected_video_latent = LTXVImgToVideoInplace.execute(
                        vae,
                        preprocessed_guidance,
                        stage2_video_latent,
                        reinject_strength,
                        False,
                    )[0]
                else:
                    reinjected_video_latent = stage2_video_latent
                latent_stage2 = LTXVConcatAVLatent.execute(reinjected_video_latent, sampled_audio_latent)[0]
                if vram_flush_node is not None:
                    latent_stage2 = vram_flush_node.run(latent_stage2)[0]
                model_stage2 = ModelSamplingLTXV.execute(stage2_model_active, float(max_shift), float(base_shift), latent_stage2)[0]
                guider_stage2 = CFGGuider.execute(model_stage2, stage2_positive, stage2_negative, float(second_stage_cfg))[0]
                sampler_stage2 = KSamplerSelect.execute(str(effective_sampler))[0]
                sigmas_stage2 = _manual_sigmas(second_stage_manual_sigmas)
                noise_stage2 = RandomNoise.execute(int(seed) + segment_index)[0]
                sampled_stage2_av = SamplerCustomAdvanced.sample(
                    noise_stage2,
                    guider_stage2,
                    sampler_stage2,
                    sigmas_stage2,
                    latent_stage2,
                )[0]
                sampled_video, sampled_audio_latent = LTXVSeparateAVLatent.execute(sampled_stage2_av)
                stage2_applied = True
                _pipeline_debug_step(
                    pipeline_debug,
                    f"segment_{segment_index}_second_stage_applied",
                    mode=second_stage_mode,
                    scale_mode=second_stage_scale_mode,
                    guidance_source=guidance_source,
                    reinject_strength=reinject_strength,
                    sigmas_stage2=sigmas_stage2,
                    sampled_video=sampled_video,
                    sampled_audio_latent=sampled_audio_latent,
                )
            else:
                guidance_source = "none"
                _pipeline_debug_step(pipeline_debug, f"segment_{segment_index}_second_stage_skipped", mode=second_stage_mode)
            stage2_segment_report = (
                f"{('auto_' + str(reference_stage2_auto)) if reference_stage2_auto else 'on'}({int(second_stage_step_count)}step,{second_stage_model_source},{second_stage_scale_mode})"
                if stage2_applied
                else "off"
            )

            anti_drift_report = "off"
            if int(segment_index) > 0 and anti_drift_mode != "off":
                anti_drift_parts = []
                if anti_drift_mode in {"rolling_adain", "dual_reference_adain"} and rolling_reference_latent is not None and anti_drift_strength > 0.0:
                    sampled_video = _apply_tail_reference_adain(
                        sampled_video,
                        rolling_reference_latent,
                        anti_drift_strength,
                        overlap_frames_value,
                        vae,
                        False,
                    )
                    anti_drift_parts.append(f"rolling:{anti_drift_strength:.2f}")
                if anti_drift_mode == "dual_reference_adain" and identity_reference_latent is not None and identity_persistence_strength > 0.0:
                    sampled_video = _apply_tail_reference_adain(
                        sampled_video,
                        identity_reference_latent,
                        identity_persistence_strength,
                        overlap_frames_value,
                        vae,
                        False,
                    )
                    anti_drift_parts.append(f"identity:{identity_persistence_strength:.2f}")
                if anti_drift_parts:
                    anti_drift_report = "+".join(anti_drift_parts)
            _pipeline_debug_step(
                pipeline_debug,
                f"segment_{segment_index}_anti_drift",
                anti_drift_mode=anti_drift_mode,
                anti_drift_report=anti_drift_report,
                rolling_reference=rolling_reference_latent is not None,
                identity_reference=identity_reference_latent is not None,
            )

            if identity_reference_latent is None:
                identity_reference_latent = _clone_latent(sampled_video)
            rolling_reference_latent = _clone_latent(sampled_video)
            _hard_unload_all_models()

            if use_in_memory_loop:
                decoded_images, loop_decode_report = _decode_images_for_vae_mode(
                    sampled_video,
                    vae,
                    _modular_decode_to_vae_mode(modular_decode),
                    512,
                    64,
                )
                frames_saved = int(decoded_images.shape[0])
                overlay_report = f"overlay=off(in-memory) decode={loop_decode_report}"
                if segment_index == 0:
                    ext_out = extension_node_mem.process_extension(
                        decoded_images,
                        overlap_frames_value,
                        overlap_side,
                        overlap_mode,
                        True,
                        "none",
                        "none",
                        start_frames_rule,
                        color_match_mode,
                        float(color_match_strength),
                        8,
                        "none",
                        0,
                        1.0,
                        0.5,
                        in_memory_stitch_preset,
                        None,
                        1,
                    )
                    current_extended_images = ext_out[2]
                    current_start_images = ext_out[1]
                else:
                    ext_out = extension_node_mem.process_extension(
                        current_extended_images,
                        overlap_frames_value,
                        overlap_side,
                        overlap_mode,
                        True,
                        "none",
                        "none",
                        start_frames_rule,
                        color_match_mode,
                        float(color_match_strength),
                        8,
                        "none",
                        0,
                        1.0,
                        0.5,
                        in_memory_stitch_preset,
                        decoded_images,
                        1,
                    )
                    current_extended_images = ext_out[2]
                    current_start_images = ext_out[1]
            else:
                segment_dir = os.path.join(segments_dir, f"seg_{segment_index:03d}")
                decode_dir, frames_saved, _ = vae_decode_node.decode_to_disk(
                    sampled_video,
                    vae,
                    segment_dir,
                    "frame",
                    internal_decode_image_format,
                    int(internal_decode_jpg_quality),
                    bool(decode_settings["tile"]),
                    decode_settings["tiling_mode"],
                    int(decode_settings["tile_size"]),
                    int(decode_settings["overlap"]),
                    False,
                    os.path.join(run_dir, "seam_debug"),
                    bool(decode_settings["cleanup_between_frames"]),
                    True,
                    0,
                )

                overlay_report = "overlay=off"
                if str(segment_overlay_mode) != "off":
                    if str(segment_overlay_mode) == "custom_text":
                        overlay_text = _format_segment_overlay_text(
                            segment_overlay_text,
                            segment_index,
                            segment_count,
                            current_segment_raw_frames,
                            current_segment_unique_frames,
                            effective_unique_frames,
                            total_duration_seconds,
                        )
                    else:
                        overlay_text = _format_segment_overlay_text(
                            "seg {segment_number}/{segment_count}\nraw {raw_frames}f uniq {unique_frames}f eff {effective_frames}f",
                            segment_index,
                            segment_count,
                            current_segment_raw_frames,
                            current_segment_unique_frames,
                            effective_unique_frames,
                            total_duration_seconds,
                        )
                    overlay_frames = _overlay_text_on_frame_dir(decode_dir, overlay_text)
                    overlay_report = f"overlay={overlay_frames} frames"

                if segment_index == 0:
                    ext_out = extension_node.process_extension_disk(
                        decode_dir,
                        extended_dir,
                        start_dir,
                        overlap_frames_value,
                        overlap_side,
                        overlap_mode,
                        True,
                        "none",
                        "none",
                        start_frames_rule,
                        stitch_preset,
                        "",
                        1,
                    )
                else:
                    ext_out = extension_node.process_extension_disk(
                        extended_dir,
                        extended_dir,
                        start_dir,
                        overlap_frames_value,
                        overlap_side,
                        overlap_mode,
                        True,
                        "none",
                        "none",
                        start_frames_rule,
                        stitch_preset,
                        decode_dir,
                        1,
                    )

            _pipeline_debug_step(
                pipeline_debug,
                f"segment_{segment_index}_decoded_and_stitched",
                decode_backend="in_memory" if use_in_memory_loop else "disk",
                frames_saved=int(frames_saved),
                overlay_report=overlay_report,
                extension_report=ext_out[5],
                extended_dir=extended_dir if not use_in_memory_loop else "in-memory",
                start_dir=start_dir if not use_in_memory_loop else "in-memory",
            )
            gate_out = audio_gate_node.decide(
                remaining_frames_after,
                effective_unique_frames,
                1,
                True,
                is_last_segment,
                cursor_frames_out,
            )
            cursor_frames = cursor_frames_out
            rendered_segments += 1
            segment_reports.append(
                f"seg{segment_index}: raw={current_segment_raw_frames}f unique={current_segment_unique_frames}f effective={effective_unique_frames}f saved={int(frames_saved)}f init={init_mode} tail={tail_refresh_report} anchor={anchor_refresh_report} stage2={stage2_segment_report} guidance={guidance_source} anti_drift={anti_drift_report} {overlay_report} | {ext_out[5]}"
            )
            _pipeline_debug_step(
                pipeline_debug,
                f"segment_{segment_index}_gate",
                gate_decision=int(gate_out[0]),
                cursor_frames=cursor_frames,
                rendered_segments=rendered_segments,
                is_last_segment=is_last_segment,
            )
            _soft_cleanup()
            if int(gate_out[0]) == 0:
                break

        final_frames_dir = extended_dir if not use_in_memory_loop else ""
        final_start_dir = start_dir if not use_in_memory_loop else ""
        final_report_hint = final_frames_dir if final_frames_dir else "(in-memory images)"
        rendered_audio = raw_audio if uses_input_audio else None
        loop_audio_export_note = "input_audio_passthrough" if uses_input_audio else "off"
        if generates_audio and exports_audio:
            loop_audio_export_note = "off(loop_generated_audio_not_muxed)"
        _pipeline_debug_step(
            pipeline_debug,
            "loop_completed",
            rendered_segments=rendered_segments,
            segment_count=segment_count,
            final_frames_dir=final_frames_dir,
            final_start_dir=final_start_dir,
            audio_export=rendered_audio is not None,
            audio_export_note=loop_audio_export_note,
        )
        disable_vae_frame_align = bool(reference_audio_img2vid)
        loop_debug_report = _pipeline_debug_text(pipeline_debug)
        plan_data_for_frame_target = _parse_payload(plan_payload)
        render_target_frame_count = (
            _to_int(plan_data_for_frame_target, "target_frame_count", 0)
            or _to_int(plan_data_for_frame_target, "audio_duration_frames_with_tail", 0)
            or _to_int(plan_data_for_frame_target, "total_frames", 0)
        )
        render_target_frame_count_source = "planner_payload" if render_target_frame_count > 0 else "render_actual_frames"
        if render_target_frame_count <= 0 and final_frames_dir:
            render_target_frame_count = len(_frame_files_in_dir(final_frames_dir))
        if render_target_frame_count <= 0 and current_extended_images is not None:
            render_target_frame_count = int(current_extended_images.shape[0])

        report = (
            _render_status(
                total_duration_seconds,
                {
                    "fps": fps_value,
                    "total_frames": _to_int(_parse_payload(plan_payload), "total_frames", 0),
                    "segment_count": int(segment_count),
                    "first_segment_raw_frames": _to_int(_parse_payload(plan_payload), "first_segment_raw_frames", 0),
                    "continuation_raw_frames": _to_int(_parse_payload(plan_payload), "continuation_raw_frames", 0),
                    "recommended_overlap_frames": overlap_frames_value,
                },
                modular_decode,
                continuity_settings["mode"],
                anti_drift_mode,
                str(second_stage_mode) != "off",
                bool(linx_output(linx, "audio_concat_enabled", False) or audio_concat_payload),
            )
            + "\n"
            + planner_report_line
            + "\n"
            + f"Prompt route. positive=\"{prompt_excerpt}\"\n"
            + f"Audio preprocess. melband_enabled={melband_enabled} | {audio_preprocess_report}\n"
            + f"Render route. backend_requested={requested_backend_mode} | backend_resolved={backend_mode} | media_mode={effective_media_mode} | decode_mode={modular_decode} | generation_mode={generation_mode} | sampler={effective_sampler} | cfg={float(effective_cfg_value):.3f} | sigmas={sigmas_report} | audio_export={'yes' if rendered_audio is not None else 'no'} | audio_export_note={loop_audio_export_note} | motion_intensity={motion_intensity:.2f} | vram_flush={'on' if bool(vram_flush) else 'off'} | vae_frame_align={'off_iamccs_audio_img2vid' if disable_vae_frame_align else 'on'}\n"
            + f"Executable AU+IMG2VID render completed. segments_rendered={rendered_segments}/{segment_count} | "
            f"frames_dir={final_report_hint} | start_dir={final_start_dir or '(in-memory start_images)'}\n"
            + "\n".join(segment_reports)
        )
        report = _append_debug_report(report, loop_debug_report)
        render_linx = build_stage_linx_payload(
            linx,
            "exec_render",
            "render",
            {
                "pipeline_kind": "au_img2vid_exec",
                "backend_mode": str(backend_mode),
                "media_mode": str(effective_media_mode),
                "generation_mode": str(generation_mode),
                "fps": fps_value,
                "modular_decode": str(modular_decode),
                "continuity_anchor_mode": str(continuity_settings["mode"]),
                "anchor_refresh_interval": int(continuity_settings["interval"]),
                "manual_sigmas": str(manual_sigmas),
                "anti_drift_mode": str(anti_drift_mode),
                "anti_drift_strength": float(anti_drift_strength),
                "identity_persistence_strength": float(identity_persistence_strength),
              "motion_intensity": float(motion_intensity),
              "generated_media_duration_seconds": float(generated_media_duration_seconds),
              "generated_media_fps": float(fps_value),
              "target_frame_count": int(render_target_frame_count),
              "target_frame_count_source": str(render_target_frame_count_source),
              "disable_vae_frame_align": disable_vae_frame_align,
              "vram_flush": bool(vram_flush),
              "debug_verbose": bool(debug_verbose),
                "second_stage_mode": str(second_stage_mode),
                "second_stage_scale_mode": str(second_stage_scale_mode),
                "second_stage_upscale_model_name": str(second_stage_upscale_model_name),
                "second_stage_steps": int(second_stage_step_count),
                "second_stage_model_source": str(second_stage_model_source),
                "downstream_stage_mode": str(downstream_stage_mode),
                "segment_count": int(segment_count),
                "segments_rendered": int(rendered_segments),
                "pipeline_debug_steps": len(pipeline_debug.get("lines") or []),
                "debug_verbose": bool(debug_verbose),
            },
            report,
            unique_id=unique_id,
            requires={
                "resources": {"model": "MODEL", "clip": "CLIP", "vae": "VAE", "audio_vae": "VAE", "planner_payload": "STRING", "fps": "FLOAT"},
                "input_audio_modes": {"audio": "AUDIO", "audio_ltx_conditioning": "AUDIO", "audio_conditioning_single": "AUDIO", "audio_conditioning_segmented": "AUDIO"},
                "pure_or_generated_modes": {"audio": "optional final audio only"},
            },
            slot_map={
                "frames_dir": {"type": "STRING", "role": "rendered_frames_dir"},
                "start_dir": {"type": "STRING", "role": "rendered_start_dir"},
                "linx": {"type": SUPERNODE_LINX_TYPE, "role": "stage_linx"},
            },
            downstream_stages=_downstream_stage_hints(downstream_stage_mode),
            policies={
                "decode_mode": str(modular_decode),
                "stitch_preset": str(stitch_preset),
                "color_match_mode": str(color_match_mode),
                "color_match_strength": float(color_match_strength),
                "audio_context_mode": str(audio_context_mode),
                "continuity_anchor_mode": str(continuity_settings["mode"]),
                "anchor_refresh_interval": int(continuity_settings["interval"]),
                "anti_drift_mode": str(anti_drift_mode),
                "second_stage_mode": str(second_stage_mode),
                "second_stage_scale_mode": str(second_stage_scale_mode),
                "second_stage_steps": int(second_stage_step_count),
                "second_stage_model_source": str(second_stage_model_source),
            },
            outputs={
                "frames_dir": final_frames_dir,
                "start_dir": final_start_dir,
                "segments_rendered": int(rendered_segments),
                "estimated_duration_seconds": float(total_duration_seconds),
                "render_status": _first_line(report),
                "backend_mode": str(backend_mode),
                "render_backend_mode": str(backend_mode),
                "decode_mode": str(modular_decode),
                "media_mode": str(effective_media_mode),
                "generation_type": str(generation_type),
                "target_frame_count": int(render_target_frame_count),
                "target_frame_count_source": str(render_target_frame_count_source),
                "disable_vae_frame_align": disable_vae_frame_align,
                "pipeline_debug": list(pipeline_debug.get("lines") or []),
                "debug_verbose": bool(debug_verbose),
            },
            resources={
                "audio": rendered_audio if exports_audio else None,
                "model": model,
                "clip": clip,
                "vae": vae,
                "audio_vae": audio_vae,
                "fps": float(fps_value),
                "decode_mode": str(modular_decode),
                "output_root": str(output_root),
                "planner_payload": str(plan_payload),
                "media_mode": str(effective_media_mode),
                "generation_mode": str(generation_mode),
                "target_frame_count": int(render_target_frame_count),
                "target_frame_count_source": str(render_target_frame_count_source),
                "disable_vae_frame_align": disable_vae_frame_align,
                "debug_verbose": bool(debug_verbose),
                "second_stage_model": stage2_model_active,
                "second_stage_payload": second_stage_payload,
                "anti_drift_mode": str(anti_drift_mode),
                "rendered_images": current_extended_images if use_in_memory_loop else None,
                "start_images": current_start_images if use_in_memory_loop else None,
            },
        )
        return (final_frames_dir, final_start_dir, int(rendered_segments), float(total_duration_seconds), render_linx, report)


class IAMCCS_GC_AUIMG2VIDExecutableFinalize:
    CATEGORY = "IAMCCS/GoyAIcanvas/TestBackends"
    FUNCTION = "finalize"
    RETURN_TYPES = ("STRING", SUPERNODE_LINX_TYPE, "STRING")
    RETURN_NAMES = ("video_path", "linx", "report")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames_dir": ("STRING",),
                "audio": ("AUDIO",),
                "frame_rate": ("FLOAT", {"default": _DEFAULT_GENERATED_FPS, "min": 1.0, "max": 240.0, "step": 0.01}),
                "filename_prefix": ("STRING", {"default": "IAMCCS/GC_AUIMG2VID_EXEC"}),
                "crf": ("INT", {"default": 19, "min": 0, "max": 51, "step": 1}),
                "pix_fmt": (["yuv420p", "yuv444p"], {"default": "yuv420p"}),
                "trim_to_audio": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "linx": (SUPERNODE_LINX_TYPE,),
                "debug_verbose": ("BOOLEAN", {"default": False}),
            },
        }

    def finalize(self, frames_dir, audio, frame_rate, filename_prefix, crf, pix_fmt, trim_to_audio, debug_verbose=False, linx=None):
        debug_verbose = _debug_verbose_enabled(debug_verbose, linx)
        pipeline_debug = _pipeline_debug_new("exec_finalize", debug_verbose)
        _pipeline_debug_step(
            pipeline_debug,
            "input_received",
            frames_dir=frames_dir,
            audio=audio,
            frame_rate=frame_rate,
            filename_prefix=filename_prefix,
            crf=crf,
            pix_fmt=pix_fmt,
            trim_to_audio=trim_to_audio,
            linx_keys=",".join(str(k) for k in ((linx or {}).get("resource_keys") or [])),
        )
        combine_node = _node_class("IAMCCS_VideoCombineFromDir")()
        video_path, report = combine_node.combine(
            frames_dir,
            float(frame_rate),
            filename_prefix,
            int(crf),
            pix_fmt,
            bool(trim_to_audio),
            audio,
        )
        _pipeline_debug_step(pipeline_debug, "vhs_combine", video_path=video_path, combine_report=report)
        report = _append_debug_report(report, _pipeline_debug_text(pipeline_debug))
        finalize_linx = build_stage_linx_payload(
            linx,
            "exec_finalize",
            "finalize",
            {
                "filename_prefix": str(filename_prefix),
                "frame_rate": float(frame_rate),
                "trim_to_audio": bool(trim_to_audio),
                "pipeline_debug_steps": len(pipeline_debug.get("lines") or []),
            },
            report,
            slot_map={
                "video_path": {"type": "STRING", "role": "final_video_path"},
                "linx": {"type": SUPERNODE_LINX_TYPE, "role": "stage_linx"},
            },
            downstream_stages=[],
            outputs={
                "video_path": video_path,
                "frames_dir": str(frames_dir),
                "pipeline_debug": list(pipeline_debug.get("lines") or []),
                "debug_verbose": bool(debug_verbose),
            },
            resources={
                "debug_verbose": bool(debug_verbose),
            },
        )
        return (video_path, finalize_linx, report)


class IAMCCS_GC_AUIMG2VIDExecutableVAE:
    CATEGORY = "IAMCCS/GoyAIcanvas/TestBackends"
    FUNCTION = "decode_and_combine"
    RETURN_TYPES = ("STRING", SUPERNODE_LINX_TYPE, "STRING", "IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("video_path", "linx", "report", "images", "audio_passthrough", "frames_dir_out")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_rate": ("FLOAT", {"default": _DEFAULT_GENERATED_FPS, "min": 1.0, "max": 240.0, "step": 0.01}),
                "decode_mode": (_VAE_DECODE_MODES, {"default": "normal_tiled_iamccs"}),
                "filename_prefix": ("STRING", {"default": "IAMCCS/GC_AUIMG2VID_EXEC"}),
                "output_root": ("STRING", {"default": "iamccs_gc_auimg2vid/final_vae"}),
                "frames_subdir": ("STRING", {"default": "frames"}),
                "image_format": (["png", "jpg"], {"default": "jpg"}),
                "jpg_quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
                "tiled_tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "tiled_overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 1}),
                "tiled_temporal_size": ("INT", {"default": 256, "min": 8, "max": 4096, "step": 4}),
                "tiled_temporal_overlap": ("INT", {"default": 32, "min": 0, "max": 1024, "step": 4}),
                "cleanup_before_decode": ("BOOLEAN", {"default": False}),
                "crf": ("INT", {"default": 19, "min": 0, "max": 51, "step": 1}),
                "pix_fmt": (["yuv420p", "yuv444p"], {"default": "yuv420p"}),
                "trim_to_audio": ("BOOLEAN", {"default": True}),
                "save_metadata": ("BOOLEAN", {"default": False}),
                "vram_flush": ("BOOLEAN", {"default": False}),
                "ui_preset": (["custom", "balanced", "high_quality", "fast_preview", "low_ram_safe", "very_low_ram_decode"], {"default": "custom"}),
            },
            "optional": {
                "audio": ("AUDIO", {"lazy": True}),
                "video_latent": ("LATENT",),
                "vae": ("VAE",),
                "frames_dir": ("STRING",),
                "linx": (SUPERNODE_LINX_TYPE,),
                "debug_verbose": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def check_lazy_status(
        self,
        frame_rate=_DEFAULT_GENERATED_FPS,
        decode_mode="normal_tiled_iamccs",
        filename_prefix="IAMCCS/GC_AUIMG2VID_EXEC",
        output_root="iamccs_gc_auimg2vid/final_vae",
        frames_subdir="frames",
        image_format="jpg",
        jpg_quality=95,
        tiled_tile_size=512,
        tiled_overlap=64,
        tiled_temporal_size=256,
        tiled_temporal_overlap=32,
        cleanup_before_decode=False,
        crf=19,
        pix_fmt="yuv420p",
        trim_to_audio=True,
        save_metadata=False,
        vram_flush=False,
        ui_preset="custom",
        debug_verbose=False,
        audio=None,
        video_latent=None,
        vae=None,
        frames_dir="",
        linx=None,
        **kwargs,
    ):
        media_mode_hint = str(linx_output(linx, "media_mode", "") or linx_resource(linx, "media_mode", "") or "")
        generation_type_hint = str(linx_output(linx, "generation_type", "") or "")
        if _pure_media_no_external_audio(media_mode_hint, generation_type_hint):
            return []
        if linx_resource(linx, "audio", None) is not None:
            return []
        return ["audio"] if audio is None else []

    def decode_and_combine(
        self,
        frame_rate,
        decode_mode,
        filename_prefix,
        output_root,
        frames_subdir,
        image_format,
        jpg_quality,
        tiled_tile_size,
        tiled_overlap,
        tiled_temporal_size,
        tiled_temporal_overlap,
        cleanup_before_decode,
        crf,
        pix_fmt,
        trim_to_audio,
        save_metadata,
        vram_flush=False,
        ui_preset="custom",
        debug_verbose=False,
        audio=None,
        video_latent=None,
        vae=None,
        frames_dir="",
        linx=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        debug_verbose = _debug_verbose_enabled(debug_verbose, linx)
        pipeline_debug = _pipeline_debug_new("exec_vae", debug_verbose)
        _pipeline_debug_step(
            pipeline_debug,
            "input_received",
            decode_mode=decode_mode,
            frame_rate=frame_rate,
            audio_connected=audio is not None,
            video_latent_connected=video_latent is not None,
            vae_connected=vae is not None,
            frames_dir=frames_dir,
            linx_keys=",".join(str(k) for k in ((linx or {}).get("resource_keys") or [])),
        )
        media_mode_hint = str(linx_output(linx, "media_mode", "") or linx_resource(linx, "media_mode", "") or "")
        generation_type_hint = str(linx_output(linx, "generation_type", "") or "")
        pure_no_audio = _pure_media_no_external_audio(media_mode_hint, generation_type_hint)
        if pure_no_audio:
            audio = linx_resource(linx, "audio", None)
            audio_source = "linx_generated" if audio is not None else "pure_mode_silent"
        else:
            linx_audio = linx_resource(linx, "audio", None)
            audio = linx_audio if linx_audio is not None else audio
            audio_source = "linx_input_passthrough" if linx_audio is not None else ("direct_input" if audio is not None else "missing_input")
        if audio is None and not pure_no_audio:
            _pipeline_debug_step(pipeline_debug, "error_missing_audio", pure_no_audio=pure_no_audio, media_mode=media_mode_hint, generation_type=generation_type_hint)
            audio = _require_runtime_value(audio, "audio")
        audio, audio_sanitize_report = _sanitize_audio_for_ffmpeg(audio)
        if audio_sanitize_report not in {"audio_sanitize=clean", "audio_sanitize=off(no_audio)"}:
            _pipeline_debug_step(
                pipeline_debug,
                "audio_sanitized_for_ffmpeg",
                audio_source=audio_source,
                sanitize_report=audio_sanitize_report,
                audio=audio,
            )
        _pipeline_debug_step(
            pipeline_debug,
            "audio_resolved",
            pure_no_external_audio=pure_no_audio,
            media_mode_hint=media_mode_hint,
            generation_type_hint=generation_type_hint,
            audio_source=audio_source,
            audio_sanitize=audio_sanitize_report,
            audio=audio,
        )
        vae = _input_or_linx(vae, linx, "vae")
        if video_latent is None:
            video_latent = _input_or_linx(None, linx, "video_latent")
        rendered_images = _input_or_linx(None, linx, "rendered_images")
        _pipeline_debug_step(
            pipeline_debug,
            "video_source_resolved",
            vae=vae,
            video_latent=video_latent,
            rendered_images=rendered_images,
            frames_dir=frames_dir,
        )
        frame_rate = float(_inherit_widget_value(frame_rate, _DEFAULT_GENERATED_FPS, linx, "fps"))
        decode_mode = _normalize_modular_decode_mode(
            _inherit_widget_value(decode_mode, "inherit_render_backend", linx, "decode_mode"),
            linx_output(linx, "backend_mode", "auto"),
        )
        output_root = str(_inherit_widget_value(output_root, "iamccs_gc_auimg2vid/final_vae", linx, "output_root"))
        run_root = _resolve_output_path(output_root)
        target_frames_dir = os.path.join(run_root, str(frames_subdir or "frames"))
        actual_frames_dir = str(frames_dir or "").strip()
        target_frame_count, target_frame_count_source = _resolve_linx_frame_target(linx)
        decode_report = ""
        frame_align_report = "frame_align=off"
        resolved_decode_mode = _modular_decode_to_vae_mode(decode_mode)
        render_backend_mode = str(linx_output(linx, "backend_mode", "auto") or "auto")
        images_out = None
        vram_flush_enabled = bool(vram_flush)
        _pipeline_debug_step(
            pipeline_debug,
            "decode_route_resolved",
            decode_requested=decode_mode,
            decode_resolved=resolved_decode_mode,
            render_backend=render_backend_mode,
            output_root=output_root,
            target_frames_dir=target_frames_dir,
            tile_size=tiled_tile_size,
            overlap=tiled_overlap,
            temporal_size=tiled_temporal_size,
            temporal_overlap=tiled_temporal_overlap,
            cleanup_before_decode=cleanup_before_decode,
            target_frame_count=target_frame_count,
            target_frame_count_source=target_frame_count_source,
        )
        if vram_flush_enabled:
            _hard_unload_all_models()
            _pipeline_debug_step(pipeline_debug, "pre_decode_vram_flush", vram_flush=True)

        if rendered_images is not None:
            images_out, frame_align_report = _align_images_to_frame_count(rendered_images, target_frame_count)
            actual_frames_dir, frames_saved = _images_to_dir(
                images_out,
                target_frames_dir,
                "frame",
                image_format,
                int(jpg_quality),
                True,
            )
            decode_report = f"decode_mode={decode_mode} used in-memory rendered images | {frame_align_report} -> saved {frames_saved} frames to {actual_frames_dir}"
            _pipeline_debug_step(pipeline_debug, "decode_from_rendered_images", frames_saved=frames_saved, actual_frames_dir=actual_frames_dir, images_out=images_out, frame_align_report=frame_align_report)
        elif video_latent is not None and vae is not None:
            if str(resolved_decode_mode) in {"low_ram_disk", "very_low_ram_disk"}:
                vae_decode_node = _node_class("IAMCCS_VAEDecodeToDisk")()
                if bool(cleanup_before_decode):
                    _soft_cleanup()
                    _pipeline_debug_step(pipeline_debug, "cleanup_before_disk_decode", cleanup_before_decode=True)
                actual_frames_dir, _, _ = vae_decode_node.decode_to_disk(
                    video_latent,
                    vae,
                    target_frames_dir,
                    "frame",
                    image_format,
                    int(jpg_quality),
                    True,
                    "manual",
                    int(tiled_tile_size),
                    int(tiled_overlap),
                    False,
                    os.path.join(run_root, "seam_debug"),
                    True,
                    True,
                    0,
                )
                frames_saved, frame_align_report = _align_frame_dir_to_frame_count(actual_frames_dir, target_frame_count, "frame")
                decode_report = (
                    f"decode_node=IAMCCS_VAEDecodeToDisk | mode={resolved_decode_mode} | "
                    f"tiling_mode=manual | tile_size={int(tiled_tile_size)} | overlap={int(tiled_overlap)} | "
                    f"temporal_size={int(tiled_temporal_size)} | temporal_overlap={int(tiled_temporal_overlap)} | "
                    f"cleanup_before_decode={'on' if bool(cleanup_before_decode) else 'off'} | "
                    f"image_format={image_format} | jpg_quality={int(jpg_quality)} | {frame_align_report} -> {actual_frames_dir}"
                )
                images_out = _load_images_from_dir_for_output(actual_frames_dir)
                _pipeline_debug_step(pipeline_debug, "decode_to_disk", frames_saved=frames_saved, actual_frames_dir=actual_frames_dir, images_out=images_out, decode_report=decode_report, frame_align_report=frame_align_report)
            else:
                decoded_images, vae_decode_report = _decode_images_for_vae_mode(
                    video_latent,
                    vae,
                    resolved_decode_mode,
                    int(tiled_tile_size),
                    int(tiled_overlap),
                    int(tiled_temporal_size),
                    int(tiled_temporal_overlap),
                    bool(cleanup_before_decode),
                )
                images_out, frame_align_report = _align_images_to_frame_count(decoded_images, target_frame_count)
                actual_frames_dir, frames_saved = _images_to_dir(
                    images_out,
                    target_frames_dir,
                    "frame",
                    image_format,
                    int(jpg_quality),
                    True,
                )
                decode_report = f"{vae_decode_report} -> saved {frames_saved} frames to {actual_frames_dir}"
                if frame_align_report != "frame_align=off":
                    decode_report = f"{vae_decode_report} | {frame_align_report} -> saved {frames_saved} frames to {actual_frames_dir}"
                _pipeline_debug_step(pipeline_debug, "decode_in_memory", frames_saved=frames_saved, actual_frames_dir=actual_frames_dir, images_out=images_out, decode_report=decode_report, frame_align_report=frame_align_report)
        elif actual_frames_dir:
            upstream_decode = "unknown"
            if str(resolved_decode_mode) == "normal_tiled_vhs":
                upstream_decode = "IAMCCS_VAEDecodeTiledSafe"
            elif str(resolved_decode_mode) == "high_vram":
                upstream_decode = "VAEDecode"
            elif str(resolved_decode_mode) in {"low_ram_disk", "very_low_ram_disk"}:
                upstream_decode = "IAMCCS_VAEDecodeToDisk"
            decode_report = (
                f"decode_node=predecoded_frames | requested={decode_mode} | resolved={resolved_decode_mode} | "
                f"upstream_decode_node={upstream_decode} | frames_dir={actual_frames_dir}"
            )
            frames_saved, frame_align_report = _align_frame_dir_to_frame_count(actual_frames_dir, target_frame_count, "frame")
            if frame_align_report != "frame_dir_align=off":
                decode_report = f"{decode_report} | {frame_align_report}"
            images_out = _load_images_from_dir_for_output(actual_frames_dir)
            _pipeline_debug_step(pipeline_debug, "decode_predecoded_frames", frames_saved=frames_saved, actual_frames_dir=actual_frames_dir, images_out=images_out, upstream_decode=upstream_decode, frame_align_report=frame_align_report)
        else:
            _pipeline_debug_step(pipeline_debug, "error_missing_video_source", video_latent=video_latent, vae=vae, frames_dir=actual_frames_dir)
            raise ValueError("Executable VAE requires either video_latent+vae or frames_dir")

        if vram_flush_enabled:
            _soft_cleanup()
            _pipeline_debug_step(pipeline_debug, "post_decode_soft_cleanup", vram_flush=True)

        print(
            "[IAMCCS SuperNodes VAE] "
            f"decode_route requested={decode_mode} resolved={resolved_decode_mode} "
            f"backend={render_backend_mode} | {decode_report}"
        )

        combine_node = _node_class("IAMCCS_VideoCombineFromDir")()
        video_path, combine_report = combine_node.combine(
            actual_frames_dir,
            float(frame_rate),
            filename_prefix,
            int(crf),
            pix_fmt,
            bool(trim_to_audio) and audio is not None,
            audio,
        )
        combine_report = f"finalize_node=IAMCCS_VideoCombineFromDir | {combine_report}"
        _pipeline_debug_step(
            pipeline_debug,
            "vhs_combine",
            video_path=video_path,
            frame_rate=frame_rate,
            crf=crf,
            pix_fmt=pix_fmt,
            trim_to_audio=bool(trim_to_audio) and audio is not None,
            audio_mux=audio is not None,
            audio_sanitize=audio_sanitize_report,
            target_frame_count=target_frame_count,
            target_frame_count_source=target_frame_count_source,
            frame_align_report=frame_align_report,
            combine_report=combine_report,
        )
        metadata_report = "metadata=off"
        if bool(save_metadata):
            metadata_payload = {
                "prompt": prompt,
                "extra_pnginfo": extra_pnginfo,
                "iamccs_supernode": {
                    "node": "IAMCCS-SuperNodes AU+IMG2VID Exec VAE",
                    "frame_rate": float(frame_rate),
                    "decode_mode": str(decode_mode),
                    "filename_prefix": str(filename_prefix),
                    "output_root": str(output_root),
                    "frames_subdir": str(frames_subdir),
                    "image_format": str(image_format),
                    "jpg_quality": int(jpg_quality),
                    "tiled_tile_size": int(tiled_tile_size),
                    "tiled_overlap": int(tiled_overlap),
                    "tiled_temporal_size": int(tiled_temporal_size),
                    "tiled_temporal_overlap": int(tiled_temporal_overlap),
                    "cleanup_before_decode": bool(cleanup_before_decode),
                    "crf": int(crf),
                    "pix_fmt": str(pix_fmt),
                    "trim_to_audio": bool(trim_to_audio),
                    "target_frame_count": int(target_frame_count),
                    "target_frame_count_source": str(target_frame_count_source),
                    "frame_align_report": str(frame_align_report),
                    "audio_sanitize": str(audio_sanitize_report),
                    "save_metadata": bool(save_metadata),
                    "vram_flush": bool(vram_flush_enabled),
                    "ui_preset": str(ui_preset),
                    "frames_dir": str(actual_frames_dir),
                    "video_path": str(video_path),
                    "resource_keys": list((linx or {}).get("resource_keys") or []),
                },
            }
            metadata_path = _save_video_metadata_sidecar(video_path, metadata_payload)
            if metadata_path:
                metadata_report = f"metadata={metadata_path}"
        _pipeline_debug_step(pipeline_debug, "metadata", save_metadata=save_metadata, metadata_report=metadata_report)
        vae_debug_report = _pipeline_debug_text(pipeline_debug)
        report = (
            f"Executable VAE. backend_mode={render_backend_mode} | decode_requested={decode_mode} | "
            f"decode_resolved={resolved_decode_mode} | frame_rate={float(frame_rate):.3f} | "
            f"target_frame_count={int(target_frame_count)} ({target_frame_count_source}) | {frame_align_report} | "
            f"tile_size={int(tiled_tile_size)} | overlap={int(tiled_overlap)} | "
            f"temporal_size={int(tiled_temporal_size)} | temporal_overlap={int(tiled_temporal_overlap)} | "
            f"cleanup_before_decode={'on' if bool(cleanup_before_decode) else 'off'} | "
            f"audio_mux={'on' if audio is not None else 'off'} | audio_source={audio_source} | {audio_sanitize_report} | vram_flush={'on' if vram_flush_enabled else 'off'}\n"
            f"{decode_report} | {combine_report} | {metadata_report}"
        )
        report = _append_debug_report(report, vae_debug_report)
        vae_linx = build_stage_linx_payload(
            linx,
            "exec_vae",
            "vae_finalize",
            {
                "decode_mode": str(decode_mode),
                "frame_rate": float(frame_rate),
                "frames_dir": str(actual_frames_dir),
                "tiled_tile_size": int(tiled_tile_size),
                "tiled_overlap": int(tiled_overlap),
                "tiled_temporal_size": int(tiled_temporal_size),
                "tiled_temporal_overlap": int(tiled_temporal_overlap),
                "cleanup_before_decode": bool(cleanup_before_decode),
                "debug_verbose": bool(debug_verbose),
                "target_frame_count": int(target_frame_count),
                "target_frame_count_source": str(target_frame_count_source),
                "frame_align_report": str(frame_align_report),
                "save_metadata": bool(save_metadata),
                "vram_flush": bool(vram_flush_enabled),
                "ui_preset": str(ui_preset),
                "pipeline_debug_steps": len(pipeline_debug.get("lines") or []),
            },
            report,
            requires={
                "resources": {"audio": "AUDIO from input-audio modes or generated pure-mode linx", "vae": "VAE", "fps": "FLOAT"},
                "one_of": {"video_source": ["video_latent", "rendered_images", "frames_dir"]},
            },
            slot_map={
                "video_path": {"type": "STRING", "role": "final_video_path"},
                "linx": {"type": SUPERNODE_LINX_TYPE, "role": "stage_linx"},
            },
            downstream_stages=[],
            outputs={
                "video_path": video_path,
                "frames_dir": str(actual_frames_dir),
                "audio_passthrough": "present" if audio is not None else "missing",
                "metadata_saved": bool(save_metadata),
                "target_frame_count": int(target_frame_count),
                "target_frame_count_source": str(target_frame_count_source),
                "frame_align_report": str(frame_align_report),
                "pipeline_debug": list(pipeline_debug.get("lines") or []),
                "debug_verbose": bool(debug_verbose),
            },
            resources={
                "audio": audio,
                "audio_passthrough": audio,
                "audio_sanitize_report": str(audio_sanitize_report),
                "vae": vae,
                "fps": float(frame_rate),
                "target_frame_count": int(target_frame_count),
                "target_frame_count_source": str(target_frame_count_source),
                "frame_align_report": str(frame_align_report),
                "decode_mode": str(resolved_decode_mode),
                "output_root": str(output_root),
                "rendered_images": images_out,
                "debug_verbose": bool(debug_verbose),
            },
        )
        if images_out is None:
            images_out = _load_images_from_dir_for_output(actual_frames_dir)
        return (video_path, vae_linx, report, images_out, audio, str(actual_frames_dir))




