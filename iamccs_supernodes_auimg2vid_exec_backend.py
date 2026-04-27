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
    LTXVSeparateAVLatent,
    ModelSamplingLTXV,
)
from comfy_extras.nodes_lt_upsampler import LTXVLatentUpsampler
from comfy_extras.nodes_lt_audio import LTXVAudioVAEEncode
from comfy_extras.nodes_mask import SolidMask

from .iamccs_supernodes_linx import SUPERNODE_LINX_TYPE, build_stage_linx_payload, linx_output, linx_policy, linx_resource


_SAMPLER_NAMES = tuple(comfy.samplers.SAMPLER_NAMES)
_REFERENCE_MANUAL_SIGMAS = "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"

try:
    import folder_paths  # type: ignore

    _LATENT_UPSCALE_MODEL_NAMES = tuple(folder_paths.get_filename_list("latent_upscale_models"))
except Exception:
    _LATENT_UPSCALE_MODEL_NAMES = (
        "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
    )


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
    text = str(value or "15sec")
    if text == "videoclip":
        return "10sec"
    if text == "monologue":
        return "15sec"
    if text in {"10sec", "15sec", "20sec"}:
        return text
    return "15sec"


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


def _modular_decode_to_vae_mode(modular_decode):
    mapping = {
        "low_ram": "low_ram_disk",
        "normal": "normal_tiled",
        "high": "high_vram",
        "custom_mode": "custom_mode",
    }
    return mapping.get(str(modular_decode), "low_ram_disk")


def _render_status(duration_seconds, planner_data, modular_decode, continuity_mode, anti_drift_mode, second_stage_enabled, audio_concat_enabled):
    return (
        f"duration {float(duration_seconds):.2f}s | fps {float(planner_data['fps']):.2f} | total {int(_to_int(planner_data, 'total_frames', 0))}f | "
        f"segments {int(_to_int(planner_data, 'segment_count', 0))} | first {int(_to_int(planner_data, 'first_segment_raw_frames', 0))}f | "
        f"loop {int(_to_int(planner_data, 'continuation_raw_frames', 0))}f | overlap {int(_to_int(planner_data, 'recommended_overlap_frames', 0))}f | "
        f"decode {modular_decode} | stage2 {'on' if second_stage_enabled else 'off'} | audio_concat {'on' if audio_concat_enabled else 'off'} | continuity {continuity_mode} | anti_drift {anti_drift_mode}"
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
    for item in str(sigmas_text or "").split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        values = [float(item.strip()) for item in _REFERENCE_MANUAL_SIGMAS.split(",")]
    return torch.tensor(values, dtype=torch.float32)


def _resize_image_to(image, width, height):
    if image is None:
        raise ValueError("image is required for resize")
    resized = comfy.utils.common_upscale(image.movedim(-1, 1), int(width), int(height), "lanczos", "center")
    return resized.movedim(1, -1)


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


def _json_safe_metadata(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_metadata(item) for item in value]
    return str(value)


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
    policy = _to_text(data, "stage2_model_policy", "replace_stage1_if_connected")
    if second_stage_model is None:
        return primary_model
    if policy in {"replace_stage1_if_connected", "prefer_stage2_else_primary"}:
        return second_stage_model
    return primary_model


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
    if payload_policy in {"periodic_anchor_refresh", "body_anchor_refresh"} or payload_source in {"periodic_keyframe_refresh", "prev_tail_plus_anchor"}:
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
                "fps": ("FLOAT", {"default": 24.0, "min": 0.001, "max": 240.0, "step": 0.01}),
                "segment_seconds": ("FLOAT", {"default": 10.0, "min": 0.01, "max": 3600.0, "step": 0.01}),
                "planning_mode": (["manual_segment_seconds", "explicit_preset_seconds"], {"default": "manual_segment_seconds"}),
                "segment_preset": (["10sec", "15sec", "20sec"], {"default": "15sec"}),
                "overlap_frames": ("INT", {"default": 9, "min": 0, "max": 4096, "step": 1}),
                "ltx_round_mode": (["up", "nearest", "down"], {"default": "up"}),
            }
            ,
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "linx": (SUPERNODE_LINX_TYPE,),
                "audio_concat_payload": ("STRING",),
            }
        }

    def plan(self, audio, fps, segment_seconds, planning_mode, segment_preset, overlap_frames, ltx_round_mode, model=None, clip=None, vae=None, audio_vae=None, linx=None, audio_concat_payload=""):
        planning_mode = _normalize_planner_mode(planning_mode)
        segment_preset = _normalize_segment_preset(segment_preset)
        duration_seconds = _duration_hint_from_payload(audio_concat_payload)
        if duration_seconds is None:
            duration_seconds = _audio_duration_seconds(audio)
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
        payload = (
            f"pipeline_kind=au_img2vid_exec; total_duration_seconds={duration_seconds:.6f}; fps={float(fps):.6f}; "
            f"segment_seconds={float(segment_seconds):.6f}; planning_mode={planning_mode}; segment_preset={segment_preset}; content_profile={segment_preset}; "
            f"overlap_frames={int(overlap_frames)}; ltx_round_mode={ltx_round_mode}; total_frames={int(planned[0])}; "
            f"unique_segment_frames={int(planned[1])}; first_segment_raw_frames={int(planned[2])}; continuation_raw_frames={int(planned[3])}; "
            f"segment_count={int(planned[4])}; continuation_loops={int(planned[5])}; last_segment_unique_frames={int(planned[6])}; "
            f"recommended_overlap_frames={int(planned[18])}; recommended_audio_left_context_s={float(planned[19]):.6f}; "
            f"recommended_extension_preset={planned[20]}"
        )
        planner_chip = _planner_chip(duration_seconds, planned, planning_mode, segment_preset)
        report = (
            f"Executable planner. duration={duration_seconds:.3f}s @ {float(fps):.3f}fps | "
            f"segments={int(planned[4])} | first_raw={int(planned[2])}f | continuation_raw={int(planned[3])}f | "
            f"recommended_overlap={int(planned[18])}f | left_context={float(planned[19]):.3f}s"
        )
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
            },
            report,
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
                "segment_count": int(planned[4]),
                "first_segment_raw_frames": int(planned[2]),
                "continuation_raw_frames": int(planned[3]),
                "recommended_overlap_frames": int(planned[18]),
                "recommended_audio_left_context_s": float(planned[19]),
                "planner_chip": planner_chip,
            },
            resources={
                "audio": audio,
                "model": model,
                "clip": clip,
                "vae": vae,
                "audio_vae": audio_vae,
                "fps": float(fps),
                "planner_payload": payload,
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
                "image": ("IMAGE",),
                "ui_preset": (["low_ram_safe", "balanced", "high_quality", "fast_preview", "custom"], {"default": "balanced"}),
                "positive_text": ("STRING", {"default": "cinematic motion, detailed scene", "multiline": True}),
                "negative_text": ("STRING", {"default": "blurry, low quality, artifacts", "multiline": True}),
                "width": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 32}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200, "step": 1}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (_SAMPLER_NAMES, {"default": "euler"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                "max_shift": ("FLOAT", {"default": 2.05, "min": 0.0, "max": 100.0, "step": 0.01}),
                "base_shift": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 100.0, "step": 0.01}),
                "sigma_terminal": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.99, "step": 0.01}),
                "manual_sigmas": ("STRING", {"default": _REFERENCE_MANUAL_SIGMAS, "multiline": True}),
                "image_strength": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "image_compression": ("INT", {"default": 35, "min": 0, "max": 100, "step": 1}),
                "audio_context_mode": (["left_context_only", "right_context_only", "symmetric_context", "no_overlap"], {"default": "left_context_only"}),
                "audio_left_context_s": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 30.0, "step": 0.01}),
                "audio_right_context_s": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "stitch_preset": (["custom", "videoclip_audio_24fps", "monologue_audio_24fps", "target_extension_ltx2", "cut_bestofk_16", "cut_bestofk_16_luma", "cut_bestofk_32", "micro_crossfade_3"], {"default": "videoclip_audio_24fps"}),
                "overlap_side": (["source", "new_images"], {"default": "source"}),
                "overlap_mode": (["cut", "linear_blend", "ease_in_out", "filmic_crossfade"], {"default": "cut"}),
                "start_frames_rule": (["none", "ltx2_round_down", "ltx2_nearest"], {"default": "ltx2_round_down"}),
                "continuity_anchor_mode": (["off", "periodic_source_refresh", "always_source_refresh"], {"default": "off"}),
                "anchor_refresh_interval": ("INT", {"default": 2, "min": 1, "max": 128, "step": 1}),
                "anchor_image_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "anti_drift_mode": (["off", "rolling_adain", "dual_reference_adain"], {"default": "off"}),
                "anti_drift_strength": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01}),
                "identity_persistence_strength": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01}),
                "modular_decode": (["low_ram", "normal", "high", "custom_mode"], {"default": "low_ram"}),
                "downstream_stage_mode": (["finalize_only", "upscale_ready", "detailer_ready", "upscale_then_detailer"], {"default": "finalize_only"}),
                "output_root": ("STRING", {"default": "iamccs_gc_auimg2vid/exec_run"}),
                "segment_overlay_mode": (["off", "segment_label", "custom_text"], {"default": "off"}),
                "segment_overlay_text": ("STRING", {"default": "seg {segment_number}/{segment_count}", "multiline": True}),
            },
            "optional": {
                "linx": (SUPERNODE_LINX_TYPE,),
                "audio": ("AUDIO",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "plan_payload": ("STRING",),
                "refresh_image": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

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
            }
        return {
            "duration_seconds": _to_float(data, "total_duration_seconds", None),
            "fps": _to_float(data, "fps", fps),
            "segment_seconds": _to_float(data, "segment_seconds", segment_seconds),
            "planning_mode": _normalize_planner_mode(_to_text(data, "planning_mode", planning_mode)),
            "segment_preset": _normalize_segment_preset(_to_text(data, "segment_preset", _to_text(data, "content_profile", segment_preset))),
            "overlap_frames": _to_int(data, "overlap_frames", overlap_frames),
            "ltx_round_mode": _to_text(data, "ltx_round_mode", ltx_round_mode),
        }

    def render(
        self,
        image,
        ui_preset,
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
        continuity_anchor_mode,
        anchor_refresh_interval,
        anchor_image_strength,
        anti_drift_mode,
        anti_drift_strength,
        identity_persistence_strength,
        modular_decode,
        downstream_stage_mode,
        output_root,
        segment_overlay_mode,
        segment_overlay_text,
        linx=None,
        audio=None,
        model=None,
        clip=None,
        vae=None,
        audio_vae=None,
        plan_payload="",
        refresh_image=None,
        unique_id=None,
    ):
        del ui_preset
        audio = _require_runtime_value(_input_or_linx(audio, linx, "audio"), "audio")
        model = _require_runtime_value(_input_or_linx(model, linx, "model"), "model")
        clip = _require_runtime_value(_input_or_linx(clip, linx, "clip"), "clip")
        vae = _require_runtime_value(_input_or_linx(vae, linx, "vae"), "vae")
        audio_vae = _require_runtime_value(_input_or_linx(audio_vae, linx, "audio_vae"), "audio_vae")
        plan_payload = str(plan_payload or linx_resource(linx, "planner_payload") or linx_output(linx, "plan_payload") or "")
        if not plan_payload:
            raise ValueError("plan_payload is required either as a local input or via linx inheritance")

        fps_value = float(linx_resource(linx, "fps", 24.0) or 24.0)
        modular_decode = str(_inherit_widget_value(modular_decode, "low_ram", linx, "decode_mode"))
        output_root = str(_inherit_widget_value(output_root, "iamccs_gc_auimg2vid/exec_run", linx, "output_root"))
        audio_concat_payload = str(linx_output(linx, "audio_concat_payload", "") or "")
        continuity_payload = str(linx_output(linx, "continuity_payload", "") or "")
        second_stage_payload = str(linx_output(linx, "second_stage_payload", "") or "")
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
        total_duration_seconds = None
        if total_duration_seconds is None:
            total_duration_seconds = planner_settings["duration_seconds"]
        if total_duration_seconds is None:
            total_duration_seconds = _duration_hint_from_payload(audio_concat_payload)
        if total_duration_seconds is None:
            total_duration_seconds = _duration_hint_from_linx(linx)
        if total_duration_seconds is None:
            total_duration_seconds = _audio_duration_seconds(audio)

        planner_node = _node_class("IAMCCS_SegmentPlanner")()
        audio_math_node = _node_class("IAMCCS_AudioExtensionMath")()
        audio_extender_node = _node_class("IAMCCS_AudioExtender")()
        audio_gate_node = _node_class("IAMCCS_AudioTimelineGate")()
        extension_node = _node_class("IAMCCS_LTX2_ExtensionModule_Disk")()
        start_inject_node = _node_class("IAMCCS_StartDirToVideoLatent")()
        vae_decode_node = _node_class("IAMCCS_VAEDecodeToDisk")()

        positive = comfy_nodes.CLIPTextEncode().encode(clip, positive_text)[0]
        negative = comfy_nodes.CLIPTextEncode().encode(clip, negative_text)[0]

        run_root = _resolve_output_path(output_root)
        run_name = f"run_{unique_id or 'manual'}_{int(seed)}"
        run_dir = os.path.join(run_root, run_name)
        segments_dir = os.path.join(run_dir, "segments")
        extended_dir = os.path.join(run_dir, "extended")
        start_dir = os.path.join(run_dir, "start")
        _ensure_clean_dir(run_dir)
        os.makedirs(segments_dir, exist_ok=True)

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
        segment_count = int(planner_head[4])
        recommended_left_context = float(planner_head[19])
        if float(audio_left_context_s) <= 0.0:
            audio_left_context_s = recommended_left_context

        planner_report_line = (
            f"Planner settings used. mode={planner_settings['planning_mode']} | "
            f"segment_preset={planner_settings['segment_preset']} | "
            f"segment_seconds={float(planner_settings['segment_seconds']):.3f}s | "
            f"overlap={int(planner_settings['overlap_frames'])}f | "
            f"ltx_round={planner_settings['ltx_round_mode']}"
        )
        if str(planner_settings["planning_mode"]) == "explicit_preset_seconds":
            planner_report_line += " | explicit_preset_seconds overrides segment_seconds with the selected 10/15/20 second preset"

        conditioned_positive, conditioned_negative = LTXVConditioning.execute(positive, negative, fps_value)
        decode_settings = _decode_settings("low_ram_disk")
        internal_decode_image_format = "jpg"
        internal_decode_jpg_quality = 95
        continuity_settings = _continuity_mode_from_payload(
            continuity_payload,
            continuity_anchor_mode,
            anchor_refresh_interval,
            anchor_image_strength,
        )
        anti_drift_mode = str(anti_drift_mode or "off")
        anti_drift_strength = max(0.0, float(anti_drift_strength))
        identity_persistence_strength = max(0.0, float(identity_persistence_strength))
        refresh_source_image = refresh_image if refresh_image is not None else image
        stage2_model_active = _resolve_stage2_model(model, second_stage_model, second_stage_payload)
        stage2_data = _parse_payload(second_stage_payload)
        second_stage_mode = _to_text(stage2_data, "second_stage_mode", "off")
        second_stage_upscale_model_name = _to_text(
            stage2_data,
            "second_stage_upscale_model",
            _LATENT_UPSCALE_MODEL_NAMES[0] if _LATENT_UPSCALE_MODEL_NAMES else "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        )
        second_stage_reinject_strength = _to_float(stage2_data, "second_stage_reinject_strength", 1.0)
        second_stage_cfg = _to_float(stage2_data, "second_stage_cfg", 1.0)
        second_stage_manual_sigmas = _to_text(stage2_data, "second_stage_manual_sigmas", "0.909375, 0.725, 0.421875, 0.0")
        identity_reference_latent = None
        rolling_reference_latent = None

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

            conditioning_audio = audio_extender_node.slice_segment(
                audio,
                fps_value,
                audio_context_mode,
                float(audio_left_context_s),
                float(audio_right_context_s),
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

            video_latent = EmptyLTXVLatentVideo.execute(int(width), int(height), current_segment_raw_frames, 1)[0]
            uses_source_anchor = _use_source_anchor(segment_index, continuity_settings["mode"], continuity_settings["interval"])
            if uses_source_anchor:
                anchor_image = refresh_source_image
                if segment_index > 0 and continuity_settings["mode"] == "periodic_source_refresh":
                    anchor_image = _load_guidance_image_from_dir(start_dir, fallback_image=refresh_source_image, pick_mode="latest")
                preprocessed_image = LTXVPreprocess.execute(anchor_image, int(image_compression))[0]
                source_strength = float(image_strength if segment_index == 0 else continuity_settings["strength"])
                video_latent = LTXVImgToVideoInplace.execute(vae, preprocessed_image, video_latent, source_strength, False)[0]
            else:
                video_latent = start_inject_node.inject(
                    start_dir,
                    vae,
                    video_latent,
                    "all",
                    max(1, overlap_frames_value),
                    0,
                    float(image_strength),
                    True,
                    int(image_compression),
                )[0]

            audio_latent = LTXVAudioVAEEncode.execute(conditioning_audio, audio_vae)[0]
            audio_mask = SolidMask.execute(0.0, 1024, 1024)[0]
            audio_latent = comfy_nodes.SetLatentNoiseMask().set_mask(audio_latent, audio_mask)[0]
            segment_positive = conditioned_positive
            segment_negative = conditioned_negative
            _hard_unload_all_models()
            av_latent = LTXVConcatAVLatent.execute(video_latent, audio_latent)[0]
            model_for_segment = ModelSamplingLTXV.execute(model, float(max_shift), float(base_shift), av_latent)[0]
            guider = CFGGuider.execute(model_for_segment, segment_positive, segment_negative, float(cfg))[0]
            sampler = KSamplerSelect.execute(str(sampler_name))[0]
            sigmas = _manual_sigmas(manual_sigmas)
            noise = RandomNoise.execute(int(seed) + segment_index)[0]
            sampled_av = SamplerCustomAdvanced.sample(noise, guider, sampler, sigmas, av_latent)[1]
            sampled_video, sampled_audio_latent = LTXVSeparateAVLatent.execute(sampled_av)
            segment_positive, segment_negative, sampled_video = LTXVCropGuides.execute(
                segment_positive,
                segment_negative,
                sampled_video,
            )

            stage_mode = str(second_stage_mode)
            stage2_applied = False
            if stage_mode == "latent_upscale_refine":
                guidance_source = "refresh"
                guidance_image = refresh_source_image
                if not uses_source_anchor:
                    guidance_image = _load_guidance_image_from_dir(start_dir, fallback_image=refresh_source_image, pick_mode="latest")
                    guidance_source = "tail"
                stage2_positive, stage2_negative, cropped_video_latent = LTXVCropGuides.execute(
                    conditioned_positive,
                    conditioned_negative,
                    sampled_video,
                )
                upscale_model = LatentUpscaleModelLoader.execute(str(second_stage_upscale_model_name))[0]
                upsampled_video_latent = LTXVLatentUpsampler().upsample_latent(
                    cropped_video_latent,
                    upscale_model,
                    vae,
                )[0]
                upsample_width, upsample_height = _pixel_dims_from_latent(upsampled_video_latent, vae)
                resized_guidance_image = _resize_image_to(guidance_image, upsample_width, upsample_height)
                preprocessed_guidance = LTXVPreprocess.execute(resized_guidance_image, int(image_compression))[0]
                reinject_strength = float(second_stage_reinject_strength)
                if uses_source_anchor and segment_index > 0:
                    reinject_strength = min(reinject_strength, float(continuity_settings["strength"]))
                reinjected_video_latent = LTXVImgToVideoInplace.execute(
                    vae,
                    preprocessed_guidance,
                    upsampled_video_latent,
                    reinject_strength,
                    False,
                )[0]
                latent_stage2 = LTXVConcatAVLatent.execute(reinjected_video_latent, sampled_audio_latent)[0]
                model_stage2 = ModelSamplingLTXV.execute(stage2_model_active, float(max_shift), float(base_shift), latent_stage2)[0]
                guider_stage2 = CFGGuider.execute(model_stage2, stage2_positive, stage2_negative, float(second_stage_cfg))[0]
                sampler_stage2 = KSamplerSelect.execute("euler")[0]
                sigmas_stage2 = _manual_sigmas(second_stage_manual_sigmas)
                noise_stage2 = RandomNoise.execute(int(seed) + segment_index)[0]
                sampled_stage2_av = SamplerCustomAdvanced.sample(
                    noise_stage2,
                    guider_stage2,
                    sampler_stage2,
                    sigmas_stage2,
                    latent_stage2,
                )[1]
                sampled_video = LTXVSeparateAVLatent.execute(sampled_stage2_av)[0]
                stage2_applied = True
            else:
                guidance_source = "none"

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

            if identity_reference_latent is None:
                identity_reference_latent = _clone_latent(sampled_video)
            rolling_reference_latent = _clone_latent(sampled_video)
            _hard_unload_all_models()

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
                f"seg{segment_index}: raw={current_segment_raw_frames}f unique={current_segment_unique_frames}f effective={effective_unique_frames}f saved={int(frames_saved)}f anchor={'src' if uses_source_anchor else 'tail'} stage2={'on' if stage2_applied else 'off'} guidance={guidance_source} anti_drift={anti_drift_report} {overlay_report} | {ext_out[5]}"
            )
            _soft_cleanup()
            if int(gate_out[0]) == 0:
                break

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
            + f"Executable AU+IMG2VID render completed. segments_rendered={rendered_segments}/{segment_count} | "
            f"frames_dir={extended_dir} | start_dir={start_dir}\n"
            + "\n".join(segment_reports)
        )
        render_linx = build_stage_linx_payload(
            linx,
            "exec_render",
            "render",
            {
                "pipeline_kind": "au_img2vid_exec",
                "fps": fps_value,
                "modular_decode": str(modular_decode),
                "continuity_anchor_mode": str(continuity_settings["mode"]),
                "anchor_refresh_interval": int(continuity_settings["interval"]),
                "manual_sigmas": str(manual_sigmas),
                "anti_drift_mode": str(anti_drift_mode),
                "anti_drift_strength": float(anti_drift_strength),
                "identity_persistence_strength": float(identity_persistence_strength),
                "second_stage_mode": str(second_stage_mode),
                "second_stage_upscale_model_name": str(second_stage_upscale_model_name),
                "downstream_stage_mode": str(downstream_stage_mode),
                "segment_count": int(segment_count),
                "segments_rendered": int(rendered_segments),
            },
            report,
            unique_id=unique_id,
            slot_map={
                "frames_dir": {"type": "STRING", "role": "rendered_frames_dir"},
                "start_dir": {"type": "STRING", "role": "rendered_start_dir"},
                "linx": {"type": SUPERNODE_LINX_TYPE, "role": "stage_linx"},
            },
            downstream_stages=_downstream_stage_hints(downstream_stage_mode),
            policies={
                "decode_mode": _modular_decode_to_vae_mode(modular_decode),
                "stitch_preset": str(stitch_preset),
                "audio_context_mode": str(audio_context_mode),
                "continuity_anchor_mode": str(continuity_settings["mode"]),
                "anchor_refresh_interval": int(continuity_settings["interval"]),
                "anti_drift_mode": str(anti_drift_mode),
                "second_stage_mode": str(second_stage_mode),
            },
            outputs={
                "frames_dir": extended_dir,
                "start_dir": start_dir,
                "segments_rendered": int(rendered_segments),
                "estimated_duration_seconds": float(total_duration_seconds),
                "render_status": _first_line(report),
            },
            resources={
                "audio": audio,
                "model": model,
                "clip": clip,
                "vae": vae,
                "audio_vae": audio_vae,
                "fps": float(fps_value),
                "decode_mode": _modular_decode_to_vae_mode(modular_decode),
                "output_root": str(output_root),
                "planner_payload": str(plan_payload),
                "second_stage_model": stage2_model_active,
                "anti_drift_mode": str(anti_drift_mode),
            },
        )
        return (extended_dir, start_dir, int(rendered_segments), float(total_duration_seconds), render_linx, report)


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
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}),
                "filename_prefix": ("STRING", {"default": "IAMCCS/GC_AUIMG2VID_EXEC"}),
                "crf": ("INT", {"default": 19, "min": 0, "max": 51, "step": 1}),
                "pix_fmt": (["yuv420p", "yuv444p"], {"default": "yuv420p"}),
                "trim_to_audio": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "linx": (SUPERNODE_LINX_TYPE,),
            },
        }

    def finalize(self, frames_dir, audio, frame_rate, filename_prefix, crf, pix_fmt, trim_to_audio, linx=None):
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
        finalize_linx = build_stage_linx_payload(
            linx,
            "exec_finalize",
            "finalize",
            {
                "filename_prefix": str(filename_prefix),
                "frame_rate": float(frame_rate),
                "trim_to_audio": bool(trim_to_audio),
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
            },
        )
        return (video_path, finalize_linx, report)


class IAMCCS_GC_AUIMG2VIDExecutableVAE:
    CATEGORY = "IAMCCS/GoyAIcanvas/TestBackends"
    FUNCTION = "decode_and_combine"
    RETURN_TYPES = ("STRING", SUPERNODE_LINX_TYPE, "STRING")
    RETURN_NAMES = ("video_path", "linx", "report")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ui_preset": (["low_ram_safe", "balanced", "high_quality", "fast_preview", "custom"], {"default": "balanced"}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}),
                "decode_mode": (["low_ram", "normal", "high", "custom_mode"], {"default": "low_ram"}),
                "filename_prefix": ("STRING", {"default": "IAMCCS/GC_AUIMG2VID_EXEC"}),
                "output_root": ("STRING", {"default": "iamccs_gc_auimg2vid/final_vae"}),
                "frames_subdir": ("STRING", {"default": "frames"}),
                "image_format": (["png", "jpg"], {"default": "jpg"}),
                "jpg_quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
                "tiled_tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "tiled_overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 1}),
                "crf": ("INT", {"default": 19, "min": 0, "max": 51, "step": 1}),
                "pix_fmt": (["yuv420p", "yuv444p"], {"default": "yuv420p"}),
                "trim_to_audio": ("BOOLEAN", {"default": True}),
                "save_metadata": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "video_latent": ("LATENT",),
                "vae": ("VAE",),
                "frames_dir": ("STRING",),
                "linx": (SUPERNODE_LINX_TYPE,),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def decode_and_combine(
        self,
        ui_preset,
        frame_rate,
        decode_mode,
        filename_prefix,
        output_root,
        frames_subdir,
        image_format,
        jpg_quality,
        tiled_tile_size,
        tiled_overlap,
        crf,
        pix_fmt,
        trim_to_audio,
        save_metadata,
        audio=None,
        video_latent=None,
        vae=None,
        frames_dir="",
        linx=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        del ui_preset
        audio = _require_runtime_value(_input_or_linx(audio, linx, "audio"), "audio")
        vae = _input_or_linx(vae, linx, "vae")
        frame_rate = float(_inherit_widget_value(frame_rate, 24.0, linx, "fps"))
        decode_mode = str(_inherit_widget_value(decode_mode, "low_ram", linx, "decode_mode"))
        output_root = str(_inherit_widget_value(output_root, "iamccs_gc_auimg2vid/final_vae", linx, "output_root"))
        run_root = _resolve_output_path(output_root)
        target_frames_dir = os.path.join(run_root, str(frames_subdir or "frames"))
        actual_frames_dir = str(frames_dir or "").strip()
        decode_report = ""
        resolved_decode_mode = _modular_decode_to_vae_mode(decode_mode)

        if video_latent is not None and vae is not None:
            if str(resolved_decode_mode) == "low_ram_disk":
                vae_decode_node = _node_class("IAMCCS_VAEDecodeToDisk")()
                actual_frames_dir, _, _ = vae_decode_node.decode_to_disk(
                    video_latent,
                    vae,
                    target_frames_dir,
                    "frame",
                    image_format,
                    int(jpg_quality),
                    True,
                    "auto",
                    512,
                    64,
                    False,
                    os.path.join(run_root, "seam_debug"),
                    True,
                    True,
                    0,
                )
                decode_report = f"decode_mode=low_ram -> {actual_frames_dir}"
            else:
                actual_decode_mode = str(resolved_decode_mode)
                if actual_decode_mode == "custom_mode":
                    actual_decode_mode = "high_vram"
                if actual_decode_mode == "normal_tiled":
                    decoded_images = comfy_nodes.VAEDecodeTiled().decode(vae, video_latent, int(tiled_tile_size), int(tiled_overlap))[0]
                else:
                    decoded_images = comfy_nodes.VAEDecode().decode(vae, video_latent)[0]
                actual_frames_dir, frames_saved = _images_to_dir(
                    decoded_images,
                    target_frames_dir,
                    "frame",
                    image_format,
                    int(jpg_quality),
                    True,
                )
                decode_report = f"decode_mode={decode_mode} -> saved {frames_saved} frames to {actual_frames_dir}"
        elif actual_frames_dir:
            decode_report = f"decode_mode={decode_mode} bypassed because frames_dir was provided directly: {actual_frames_dir}"
        else:
            raise ValueError("Executable VAE requires either video_latent+vae or frames_dir")

        combine_node = _node_class("IAMCCS_VideoCombineFromDir")()
        video_path, combine_report = combine_node.combine(
            actual_frames_dir,
            float(frame_rate),
            filename_prefix,
            int(crf),
            pix_fmt,
            bool(trim_to_audio),
            audio,
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
                    "crf": int(crf),
                    "pix_fmt": str(pix_fmt),
                    "trim_to_audio": bool(trim_to_audio),
                    "save_metadata": bool(save_metadata),
                    "frames_dir": str(actual_frames_dir),
                    "video_path": str(video_path),
                    "resource_keys": list((linx or {}).get("resource_keys") or []),
                },
            }
            metadata_path = _save_video_metadata_sidecar(video_path, metadata_payload)
            if metadata_path:
                metadata_report = f"metadata={metadata_path}"
        report = f"{decode_report} | {combine_report} | {metadata_report}"
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
                "save_metadata": bool(save_metadata),
            },
            report,
            slot_map={
                "video_path": {"type": "STRING", "role": "final_video_path"},
                "linx": {"type": SUPERNODE_LINX_TYPE, "role": "stage_linx"},
            },
            downstream_stages=[],
            outputs={
                "video_path": video_path,
                "frames_dir": str(actual_frames_dir),
                "metadata_saved": bool(save_metadata),
            },
            resources={
                "audio": audio,
                "vae": vae,
                "fps": float(frame_rate),
                "decode_mode": str(resolved_decode_mode),
                "output_root": str(output_root),
            },
        )
        return (video_path, vae_linx, report)