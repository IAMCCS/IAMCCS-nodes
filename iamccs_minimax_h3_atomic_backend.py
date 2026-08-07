# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later

"""Atomic MiniMax H3 execution backend for IAMCCS Shotboard.

This module is intentionally separate from every IAMCCS V3/V4 backend.  It
consumes the MiniMax Shotboard plan and makes task, model, conditioning,
acceleration and delivery routing agree for each chunk.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

import folder_paths


SHOTPLAN_TYPE = "IAMCCS_MINIMAX_H3_SHOTPLAN"
SUPERNODE_LINX_TYPE = "IAMCCS_SUPERNODE_LINX"
CATEGORY = "IAMCCS/MiniMax H3/Atomic Backend"
H3_FPS = 24
LOG = logging.getLogger("IAMCCS.MiniMaxH3.AtomicBackend")


def _node_class(name: str):
    import nodes as comfy_nodes

    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(name)
    if cls is None:
        raise RuntimeError(
            f"MiniMax H3 backend requires node '{name}'. "
            "Install/enable the corresponding custom-node pack and restart ComfyUI."
        )
    return cls


def _resolve_shotplan(value: Any) -> dict[str, Any]:
    """Unwrap the private H3 plan carried by CineLinX; tolerate legacy plans."""
    if isinstance(value, dict) and value.get("schema") == "iamccs.minimax_h3.shotplan":
        return value
    if not isinstance(value, dict):
        raise ValueError("Invalid IAMCCS MiniMax H3 cine_linx")
    resources = value.get("resources") if isinstance(value.get("resources"), dict) else {}
    outputs = value.get("outputs") if isinstance(value.get("outputs"), dict) else {}
    payload = resources.get("cine_payload") if isinstance(resources.get("cine_payload"), dict) else {}
    for candidate in (
        resources.get("iamccs_minimax_h3_shotplan"),
        resources.get("minimax_h3_shotplan"),
        resources.get("shotplan"),
        outputs.get("shotplan"),
        payload.get("minimax_h3_shotplan"),
        payload.get("shotplan"),
    ):
        if isinstance(candidate, dict) and candidate.get("schema") == "iamccs.minimax_h3.shotplan":
            return candidate
    raise ValueError("CineLinX does not contain an IAMCCS MiniMax H3 shotplan")


def _chunk(cine_linx: dict[str, Any], segment_index: int) -> dict[str, Any]:
    shotplan = _resolve_shotplan(cine_linx)
    chunks = shotplan.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("MiniMax H3 shotplan has no chunks")
    index = int(segment_index)
    if index < 0 or index >= len(chunks):
        raise IndexError(f"segment_index={index} outside 0..{len(chunks) - 1}")
    return chunks[index]


def _task_family(task: str) -> str:
    return "ref2va" if str(task or "").lower().startswith("ref2va") else "fl2va"


def _resolve_image_path(value: str) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    direct = Path(os.path.expandvars(os.path.expanduser(raw)))
    if direct.is_file():
        return direct
    try:
        annotated = folder_paths.get_annotated_filepath(raw)
        if annotated and os.path.isfile(annotated):
            return Path(annotated)
    except Exception:
        pass
    candidate = Path(folder_paths.get_input_directory()) / raw
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"MiniMax H3 reference image not found: {raw}")


def _load_image(value: str) -> torch.Tensor | None:
    path = _resolve_image_path(value)
    if path is None:
        return None
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _black(width: int, height: int) -> torch.Tensor:
    return torch.zeros((1, max(1, int(height)), max(1, int(width)), 3), dtype=torch.float32)


def _interpolate_image(image: torch.Tensor, width: int, height: int, method: str) -> torch.Tensor:
    mode = str(method or "area").lower()
    if mode not in {"nearest-exact", "bilinear", "bicubic", "area"}:
        mode = "area"
    nchw = image.movedim(-1, 1)
    kwargs: dict[str, Any] = {"size": (int(height), int(width)), "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
        kwargs["antialias"] = True
    return F.interpolate(nchw, **kwargs).movedim(1, -1)


def _resize_reference_image(image, shotplan: dict[str, Any], label: str):
    """Downsize an IMAGE batch before native H3 conditioning.

    The attached Turbo workflow uses ImageScaleToTotalPixels on both endpoint
    images.  Here the policy lives in the shotplan and also covers timeline
    keyframes, bridge frames, REF2VA stills and reference video batches.
    """
    if not torch.is_tensor(image) or image.ndim != 4:
        return image, f"{label}=none"
    settings = shotplan.get("reference_resize")
    if not isinstance(settings, dict):
        settings = {}
    policy = str(settings.get("policy", "canvas_crop") or "canvas_crop").lower()
    method = str(settings.get("filter", "area") or "area")
    multiple = max(1, int(settings.get("multiple_of", 32) or 32))
    downscale_only = bool(settings.get("downscale_only", True))
    source_h, source_w = int(image.shape[1]), int(image.shape[2])
    if policy == "off":
        return image, f"{label}={source_w}x{source_h}:off"

    if policy in {"canvas_crop", "canvas_pad"}:
        target_w = max(multiple, int(shotplan.get("width", 960)) // multiple * multiple)
        target_h = max(multiple, int(shotplan.get("height", 544)) // multiple * multiple)
        scale = max(target_w / source_w, target_h / source_h) if policy == "canvas_crop" else min(target_w / source_w, target_h / source_h)
        scaled_w = max(multiple, int(round(source_w * scale)))
        scaled_h = max(multiple, int(round(source_h * scale)))
        resized = _interpolate_image(image, scaled_w, scaled_h, method)
        if policy == "canvas_crop":
            left = max(0, (scaled_w - target_w) // 2)
            top = max(0, (scaled_h - target_h) // 2)
            resized = resized[:, top:top + min(target_h, scaled_h), left:left + min(target_w, scaled_w), :]
            if int(resized.shape[1]) != target_h or int(resized.shape[2]) != target_w:
                resized = _interpolate_image(resized, target_w, target_h, method)
        else:
            if scaled_w != target_w or scaled_h != target_h:
                canvas = torch.zeros(
                    (int(resized.shape[0]), target_h, target_w, int(resized.shape[3])),
                    dtype=resized.dtype,
                    device=resized.device,
                )
                x = max(0, (target_w - scaled_w) // 2)
                y = max(0, (target_h - scaled_h) // 2)
                copy_w, copy_h = min(target_w, scaled_w), min(target_h, scaled_h)
                canvas[:, y:y + copy_h, x:x + copy_w, :] = resized[:, :copy_h, :copy_w, :]
                resized = canvas
        return resized, f"{label}={source_w}x{source_h}->{int(resized.shape[2])}x{int(resized.shape[1])}:{policy}/{method}"

    if policy == "total_pixels":
        target_pixels = max(0.1, float(settings.get("megapixels", 0.5) or 0.5)) * 1_000_000.0
        if downscale_only and source_w * source_h <= target_pixels:
            return image, f"{label}={source_w}x{source_h}:kept"
        scale = math.sqrt(target_pixels / max(1.0, float(source_w * source_h)))
        if downscale_only:
            scale = min(1.0, scale)
        target_w = max(multiple, int(round(source_w * scale / multiple)) * multiple)
        target_h = max(multiple, int(round(source_h * scale / multiple)) * multiple)
        resized = _interpolate_image(image, target_w, target_h, method)
        return resized, f"{label}={source_w}x{source_h}->{target_w}x{target_h}:{float(settings.get('megapixels', 0.5)):.2f}MP/{method}"

    raise ValueError(f"Unknown MiniMax H3 reference resize policy: {policy}")


def _audio_slice(audio: dict[str, Any] | None, start_seconds: float, duration_seconds: float):
    if not isinstance(audio, dict) or not torch.is_tensor(audio.get("waveform")):
        return None
    waveform = audio["waveform"]
    if waveform.ndim != 3:
        raise ValueError("MiniMax H3 AUDIO waveform must be [B,C,S]")
    sample_rate = int(audio.get("sample_rate", 32000))
    start = max(0, int(round(float(start_seconds) * sample_rate)))
    count = max(1, int(round(float(duration_seconds) * sample_rate)))
    sliced = waveform[:1, :, start:min(waveform.shape[-1], start + count)]
    if sliced.shape[-1] < count:
        sliced = torch.nn.functional.pad(sliced, (0, count - sliced.shape[-1]))
    return {"waveform": sliced, "sample_rate": sample_rate}


def _unique_plan_image_paths(shotplan: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for slot in shotplan.get("slots", []):
        if not isinstance(slot, dict):
            continue
        for key in ("image", "explicit_last_image"):
            path = str(slot.get(key) or "").strip()
            if path and path not in result:
                result.append(path)
            if len(result) >= 4:
                return result
    return result


def _reference_roles(shotplan: dict[str, Any]) -> list[str]:
    roles = [str(item or "subject_identity").strip().lower() for item in shotplan.get("reference_roles", [])[:4]]
    defaults = ["subject_identity", "subject_identity", "composition", "style"]
    while len(roles) < 4:
        roles.append(defaults[len(roles)])
    return roles


def _reference_header(items: list[dict[str, str]], prompt: str) -> str:
    """Add a compact routing header only when the user did not author one.

    A prompt already using the official full-reference sections owns its label
    semantics and is never rewritten by the backend.
    """
    lower = str(prompt or "").lower()
    if "subject_definitions:" in lower or not items:
        return str(prompt or "").strip()
    lines = ["Technical reference routing:"]
    for item in items:
        lines.append(f"{item['label']} supplies {item['role'].replace('_', ' ')} only.")
    lines.append("Keep these reference labels and roles stable throughout the target clip.")
    return "\n".join(lines) + "\n\n" + str(prompt or "").strip()


def _place_text_encoder(clip, shotplan: dict[str, Any]):
    """Retarget the very large Qwen3-VL encoder before it is evaluated.

    The Q2 GGUF file is compact on disk, but individual layers are expanded
    while ComfyUI encodes the prompt.  Loading the complete encoder on a
    A 12 GiB Low VRAM configuration leaves no room for those temporary tensors and fails before
    H3 sampling starts.  ComfyUI's native retargeter clones the CLIP wrapper
    and pins both load and offload devices to CPU without mutating the loader.
    """
    mode = str(shotplan.get("text_encoder_device", "cpu_safe_12gb") or "cpu_safe_12gb").lower()
    if mode == "auto":
        return clip, "auto"
    if mode != "cpu_safe_12gb":
        raise ValueError(f"Unsupported MiniMax H3 text encoder device mode: {mode}")

    from comfy_extras.nodes_multigpu import SelectCLIPDeviceNode

    safe_clip = SelectCLIPDeviceNode.execute(clip=clip, device="cpu")[0]
    load_device = getattr(getattr(safe_clip, "patcher", None), "load_device", None)
    if getattr(load_device, "type", str(load_device)) != "cpu":
        raise RuntimeError(
            "MiniMax H3 CPU-safe mode could not place Qwen3-VL on CPU. "
            "Update ComfyUI or insert Select CLIP Device=cpu before conditioning."
        )
    return safe_clip, "cpu_safe_12gb(cpu)"


def _apply_sage(model):
    sage_cls = _node_class("PathchSageAttentionKJ")
    return sage_cls().patch(model=model, sage_attention="auto", allow_compile=False)[0]


def _apply_h3_memory_efficient_sage(model):
    sage_cls = _node_class("MiniMaxH3MemoryEfficientSageAttentionPatch")
    return sage_cls.execute(model=model)[0]


def _apply_sol(model, conditioning_mode: str):
    sol_cls = _node_class("SolAttnPatch")
    return sol_cls.execute(
        model=model,
        tau=1.3,
        start_percent=0.20,
        end_percent=0.90,
        min_tokens=4096,
        int8_qk=True,
        sink_conditioning=str(conditioning_mode),
        morton=False,
        morton_curve="2d_frame",
        int8_pv=True,
        verbose=False,
        use_tma=False,
        dense_blocks="",
    )[0]


def _apply_spectrum(model, profile: str):
    spectrum_cls = _node_class("SpectrumApplyMiniMaxH3")
    profile = str(profile or "conservative_3060").lower()
    aggressive = profile == "aggressive"
    quality = profile == "conservative_quality"
    # max_history=5 is the minimum valid history for degree=4 and is the only
    # practical default for a 1280x736 / 243-frame single branch on 32 GiB RAM.
    max_history = 8 if quality else 5
    return spectrum_cls().apply(
        model=model,
        enabled=True,
        blend_weight=0.75 if aggressive else 0.50,
        degree=4,
        ridge_lambda=0.10,
        window_size=2.0,
        flex_window=3.0 if aggressive else 0.75,
        warmup_steps=5,
        tail_actual_steps=1,
        max_history=max_history,
        debug=False,
        history_storage="system_ram",
    )[0]


def _turbo_settings(shotplan: dict[str, Any]) -> dict[str, Any]:
    settings = shotplan.get("turbo")
    return settings if isinstance(settings, dict) else {"mode": "off", "enabled": False}


def _apply_turbo_lora(model, shotplan: dict[str, Any]):
    settings = _turbo_settings(shotplan)
    mode = str(settings.get("mode", "off") or "off").lower()
    if mode == "off" or not bool(settings.get("enabled", mode != "off")):
        return model, "off"
    lora_name = str(settings.get("lora_name", "") or "").strip()
    if not lora_name:
        return model, "base H3 fallback (Turbo LoRA not selected)"
    try:
        lora_path = folder_paths.get_full_path("loras", lora_name)
    except Exception:
        lora_path = None
    if not lora_path:
        return model, f"base H3 fallback (optional Turbo LoRA missing: {lora_name})"
    strength = float(settings.get("strength", 1.0) or 1.0)

    # drbaph's pruned conversion intentionally removes incompatible AdaLN
    # pairs and is designed for ComfyUI's native model-only loader. Larry's
    # original/full LoRA uses the custom loader, which can also re-inject the
    # full-width time-conditioning update on pruned bases.
    lower_name = lora_name.lower()
    if "pruned" in lower_name or "drbaph" in lower_name:
        import nodes as comfy_nodes

        patched = comfy_nodes.LoraLoaderModelOnly().load_lora_model_only(
            model=model,
            lora_name=lora_name,
            strength_model=strength,
        )[0]
        return patched, f"native model-only LoRA {lora_name}@{strength:.2f}"

    try:
        turbo_cls = _node_class("MiniMaxH3TurboLoRA")
        patched = turbo_cls().apply_lora(model=model, lora_name=lora_name, strength=strength)[0]
        return patched, f"Larry Turbo LoRA {lora_name}@{strength:.2f}"
    except Exception as custom_exc:
        try:
            import nodes as comfy_nodes

            patched = comfy_nodes.LoraLoaderModelOnly().load_lora_model_only(
                model=model,
                lora_name=lora_name,
                strength_model=strength,
            )[0]
            return patched, f"native fallback LoRA {lora_name}@{strength:.2f} (custom loader: {custom_exc})"
        except Exception:
            raise RuntimeError(
                f"Could not apply MiniMax H3 Turbo LoRA '{lora_name}'. "
                "Update ComfyUI-MiniMax-H3-Turbo and verify the selected base/LoRA pair."
            ) from custom_exc


def _turbo_sampler(shotplan: dict[str, Any]):
    settings = _turbo_settings(shotplan)
    mode = str(settings.get("sampler_mode", "audio_fixed") or "audio_fixed").lower()
    if mode != "audio_fixed":
        return None, "stock"
    sampler_cls = _node_class("MiniMaxH3TurboSampler")
    sampler = sampler_cls().get_sampler()[0]
    return sampler, "MiniMaxH3TurboSampler(audio-fixed AV schedules)"


def _accelerate(model, shotplan: dict[str, Any]):
    mode = str(shotplan.get("acceleration", "native") or "native").lower()
    if mode == "auto_3060":
        try:
            return _apply_h3_memory_efficient_sage(model), "Low VRAM Auto -> H3 Memory-Efficient Sage"
        except Exception as h3_exc:
            try:
                return _apply_sage(model), f"Low VRAM Auto -> generic Sage (H3 patch unavailable: {h3_exc})"
            except Exception as generic_exc:
                return model, f"Low VRAM Auto -> native (H3 Sage: {h3_exc}; generic Sage: {generic_exc})"
    if mode == "native":
        return model, "native"
    if mode == "sage":
        return _apply_sage(model), "SageAttention(auto)"
    if mode == "h3_sage":
        return _apply_h3_memory_efficient_sage(model), "MiniMax H3 Memory-Efficient Sage"
    if mode == "sage_sol":
        patched = _apply_sage(model)
        patched = _apply_sol(patched, str(shotplan.get("sol_conditioning", "exact_kv")))
        return patched, f"Sage+Sol({shotplan.get('sol_conditioning', 'exact_kv')})"
    if mode == "spectrum":
        profile = str(shotplan.get("spectrum_profile", "conservative_3060"))
        return _apply_spectrum(model, profile), f"Spectrum({profile},system_ram)"
    if mode == "sage_spectrum":
        profile = str(shotplan.get("spectrum_profile", "conservative_3060"))
        sage_model = _apply_sage(model)
        return _apply_spectrum(sage_model, profile), f"SageAttention + Spectrum({profile},system_ram)"
    raise ValueError(f"Unknown MiniMax H3 acceleration mode: {mode}")


def _clean_vram_before_decode() -> str:
    try:
        import comfy.model_management as mm

        mm.unload_all_models()
        try:
            mm.cleanup_models()
        except Exception:
            pass
        mm.soft_empty_cache()
        gc.collect()
        return "diffusion/text models unloaded; CUDA cache cleared"
    except Exception as exc:
        LOG.warning("MiniMax H3 pre-decode VRAM cleanup warning: %s", exc)
        gc.collect()
        return f"cleanup warning: {exc}"


def _release_cpu_conditioning_models(shotplan: dict[str, Any]) -> str:
    """Drop CPU Qwen/VAE pages after conditioning, before H3 is loaded.

    The conditioning tensors are already materialized at this point. Keeping
    the 9.6 GiB Qwen encoder resident would leave too little system RAM for a
    Q4 H3 model and Spectrum history on a 32 GiB workstation.
    """
    mode = str(shotplan.get("text_encoder_device", "cpu_safe_12gb") or "cpu_safe_12gb").lower()
    if mode != "cpu_safe_12gb":
        return "conditioning models kept"
    try:
        import comfy.model_management as mm

        mm.unload_all_models()
        try:
            mm.cleanup_models()
        except Exception:
            pass
        mm.soft_empty_cache()
    except Exception as exc:
        LOG.warning("MiniMax H3 pre-sampler conditioning cleanup warning: %s", exc)
        return f"conditioning cleanup warning: {exc}"
    gc.collect()
    if os.name == "nt":
        try:
            import ctypes

            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.psapi.EmptyWorkingSet(handle)
            return "CPU text encoder/VAE unloaded; Windows working set trimmed"
        except Exception as exc:
            LOG.warning("MiniMax H3 working-set trim warning: %s", exc)
    return "CPU text encoder/VAE unloaded"


class IAMCCS_MiniMaxH3AtomicModelRouter:
    """Lazy model switch: only the branch required by the active H3 task loads."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
            },
            "optional": {
                "fl2va_model": ("MODEL", {"lazy": True}),
                "ref2va_model": ("MODEL", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING", "STRING")
    RETURN_NAMES = ("model", "model_family", "report")
    FUNCTION = "select"
    CATEGORY = CATEGORY

    @staticmethod
    def _input_name(cine_linx, segment_index):
        task = str(_chunk(cine_linx, segment_index).get("task_mode", "t2va"))
        return "ref2va_model" if _task_family(task) == "ref2va" else "fl2va_model"

    def check_lazy_status(self, cine_linx, segment_index, fl2va_model=None, ref2va_model=None, **kwargs):
        selected = self._input_name(cine_linx, segment_index)
        if selected == "ref2va_model" and ref2va_model is None:
            return ["ref2va_model"]
        if selected == "fl2va_model" and fl2va_model is None:
            return ["fl2va_model"]
        return []

    def select(self, cine_linx, segment_index, fl2va_model=None, ref2va_model=None):
        chunk = _chunk(cine_linx, segment_index)
        task = str(chunk.get("task_mode", "t2va"))
        family = _task_family(task)
        model = ref2va_model if family == "ref2va" else fl2va_model
        if model is None:
            raise ValueError(
                f"Atomic H3 mode selected {family}, but its MODEL input is not connected. "
                f"Connect the {family}_model branch."
            )
        return model, family, f"Atomic model route | task={task} | selected={family} | lazy_branch=yes"


class IAMCCS_MiniMaxH3AtomicConditioningBackend:
    """Build matching T2VA/I2VA/FL2VA/REF2VA conditioning and AV latent."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "video_vae": ("VAE",),
                "audio_vae": ("VAE",),
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
            },
            "optional": {
                "bridge_frame": ("IMAGE",),
                "first_frame_override": ("IMAGE",),
                "last_frame_override": ("IMAGE",),
                "ref_image_1": ("IMAGE",),
                "ref_image_2": ("IMAGE",),
                "ref_image_3": ("IMAGE",),
                "ref_image_4": ("IMAGE",),
                "ref_video": ("IMAGE",),
                "ref_video_audio": ("AUDIO",),
                "ref_audio": ("AUDIO",),
                "prompt_override": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = (
        "MODEL", "CONDITIONING", "LATENT", "IMAGE", "IMAGE", "STRING",
        "STRING", "INT", "INT", "INT", "STRING",
    )
    RETURN_NAMES = (
        "model", "positive", "latent", "first_frame", "planned_last_frame",
        "reference_manifest_json", "prompt", "current_segment", "total_segments",
        "trim_head_frames", "report",
    )
    FUNCTION = "prepare"
    CATEGORY = CATEGORY

    def prepare(
        self,
        model,
        clip,
        video_vae,
        audio_vae,
        cine_linx,
        segment_index,
        bridge_frame=None,
        first_frame_override=None,
        last_frame_override=None,
        ref_image_1=None,
        ref_image_2=None,
        ref_image_3=None,
        ref_image_4=None,
        ref_video=None,
        ref_video_audio=None,
        ref_audio=None,
        prompt_override="",
    ):
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo

        shotplan = _resolve_shotplan(cine_linx)
        chunk = _chunk(shotplan, segment_index)
        clip, text_encoder_report = _place_text_encoder(clip, shotplan)
        task = str(chunk.get("task_mode", "t2va")).lower()
        width = int(shotplan.get("width", 960))
        height = int(shotplan.get("height", 544))
        frames = int(chunk.get("frame_count", 124))
        prompt = str(prompt_override or "").strip() or str(chunk.get("prompt", "")).strip()

        planned_first = _load_image(str(chunk.get("first_image", "")))
        planned_last = _load_image(str(chunk.get("last_image", "")))
        first = first_frame_override[:1] if torch.is_tensor(first_frame_override) else planned_first
        last = last_frame_override[:1] if torch.is_tensor(last_frame_override) else planned_last
        external_images = [ref_image_1, ref_image_2, ref_image_3, ref_image_4]

        if first is None and bool(chunk.get("uses_bridge_first_frame")) and torch.is_tensor(bridge_frame):
            first = bridge_frame[:1]
        if first is None and task in {"i2va", "fl2va"} and torch.is_tensor(ref_image_1):
            first = ref_image_1[:1]
        if last is None and task == "fl2va" and torch.is_tensor(ref_image_2):
            last = ref_image_2[:1]

        resize_reports: list[str] = []
        if torch.is_tensor(first):
            first, resized = _resize_reference_image(first[:1], shotplan, "first")
            resize_reports.append(resized)
        if torch.is_tensor(last):
            last, resized = _resize_reference_image(last[:1], shotplan, "last")
            resize_reports.append(resized)
        resized_external_images = []
        for index, image in enumerate(external_images, start=1):
            if torch.is_tensor(image):
                image, resized = _resize_reference_image(image[:1], shotplan, f"ref{index}")
                resize_reports.append(resized)
            resized_external_images.append(image)
        external_images = resized_external_images

        manifest: list[dict[str, str]] = []
        if task == "t2va":
            result = MiniMaxH3ImageToVideo.execute(
                clip=clip, vae=video_vae, prompt=prompt,
                width=width, height=height, length=frames,
                first_frame=None, last_frame=None,
            )
        elif task == "i2va":
            if first is None:
                raise ValueError("I2VA requires one opening image in the Shotboard or ref_image_1")
            result = MiniMaxH3ImageToVideo.execute(
                clip=clip, vae=video_vae, prompt=prompt,
                width=width, height=height, length=frames,
                first_frame=first, last_frame=None,
            )
            manifest.append({"label": "<Picture 1>", "role": "opening_keyframe"})
        elif task == "fl2va":
            if first is None or last is None:
                raise ValueError("FL2VA requires both opening and final images; connect ref_image_1/ref_image_2 or place two adjacent Shotboard keyframes")
            result = MiniMaxH3ImageToVideo.execute(
                clip=clip, vae=video_vae, prompt=prompt,
                width=width, height=height, length=frames,
                first_frame=first, last_frame=last,
            )
            manifest.extend([
                {"label": "<Picture 1>", "role": "opening_keyframe"},
                {"label": "<Picture 2>", "role": "final_keyframe"},
            ])
        elif task.startswith("ref2va"):
            roles = _reference_roles(shotplan)
            plan_paths = _unique_plan_image_paths(shotplan)
            refs: dict[str, torch.Tensor] = {}
            for index in range(4):
                role = roles[index]
                if role == "disabled":
                    continue
                image = external_images[index]
                if not torch.is_tensor(image) and index < len(plan_paths):
                    image = _load_image(plan_paths[index])
                    if torch.is_tensor(image):
                        image, resized = _resize_reference_image(image[:1], shotplan, f"ref{index + 1}")
                        resize_reports.append(resized)
                if not torch.is_tensor(image):
                    continue
                ordinal = len(refs) + 1
                refs[f"ref_image_{ordinal}"] = image[:1]
                manifest.append({"label": f"<Picture {ordinal}>", "role": role})

            video_role = str(shotplan.get("reference_video_role", "off") or "off").lower()
            audio_role = str(shotplan.get("reference_audio_role", "off") or "off").lower()
            ref_videos = None
            ref_video_audios = None
            if torch.is_tensor(ref_video) and video_role != "off":
                ref_video, resized = _resize_reference_image(ref_video, shotplan, "ref_video")
                resize_reports.append(resized)
                ref_videos = {"ref_video_1": ref_video}
                if isinstance(ref_video_audio, dict):
                    ref_video_audios = {"ref_video_audio_1": ref_video_audio}
                    manifest.append({"label": "<Audio 1>", "role": "synchronized_video_audio"})
                manifest.append({"label": "<Video 1>", "role": video_role})

            ref_audios = None
            active_audio = ref_audio
            if active_audio is None and isinstance(ref_video_audio, dict) and video_role == "off":
                active_audio = ref_video_audio
            if isinstance(active_audio, dict) and (
                audio_role != "off" or str(shotplan.get("audio_mode", "")) == "h3_ref2va_audio"
            ):
                sliced = _audio_slice(
                    active_audio,
                    float(chunk.get("timeline_start_seconds", 0.0)),
                    float(chunk.get("duration_seconds", frames / H3_FPS)),
                )
                ref_audios = {"ref_audio_1": sliced}
                audio_ordinal = 2 if ref_video_audios else 1
                manifest.append({"label": f"<Audio {audio_ordinal}>", "role": audio_role if audio_role != "off" else "driven_audio"})

            if not refs and not ref_videos and not ref_audios:
                raise ValueError("REF2VA requires at least one enabled image, video, or audio reference")
            prompt = _reference_header(manifest, prompt)
            result = MiniMaxH3ReferenceToVideo.execute(
                clip=clip,
                vae=video_vae,
                audio_vae=audio_vae,
                prompt=prompt,
                width=width,
                height=height,
                length=frames,
                ref_image_size=str(shotplan.get("ref_image_size", "match")),
                ref_images=refs or None,
                ref_videos=ref_videos,
                ref_video_audios=ref_video_audios,
                ref_audios=ref_audios,
            )
        else:
            raise ValueError(f"Unsupported atomic H3 task: {task}")

        positive, latent = result[0], result[1]
        report = (
            f"Atomic H3 conditioning | segment={int(segment_index) + 1}/{len(shotplan['chunks'])} | "
            f"task={task} | frames={frames} | first={'yes' if first is not None else 'no'} | "
            f"last={'yes' if last is not None else 'no'} | refs={len(manifest)} | "
            f"ref_size={shotplan.get('ref_image_size', 'match')} | "
            f"pre_resize={';'.join(resize_reports) if resize_reports else 'none'} | "
            f"text_encoder={text_encoder_report}"
        )
        return (
            model,
            positive,
            latent,
            first if first is not None else _black(width, height),
            last if last is not None else _black(width, height),
            json.dumps({"task": task, "items": manifest}, ensure_ascii=False, indent=2),
            prompt,
            int(segment_index),
            int(len(shotplan["chunks"])),
            int(chunk.get("trim_head_frames", 0)),
            report,
        )


class IAMCCS_MiniMaxH3GenerationBackendV2:
    """Sigma shift, selected acceleration, sampling, VRAM clean and AV decode."""

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers

        samplers = list(comfy.samplers.SAMPLER_NAMES)
        schedulers = list(comfy.samplers.SCHEDULER_NAMES)
        if "res_multistep" in samplers:
            samplers.remove("res_multistep")
            samplers.insert(0, "res_multistep")
        if "simple" in schedulers:
            schedulers.remove("simple")
            schedulers.insert(0, "simple")
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "latent": ("LATENT",),
                "video_vae": ("VAE",),
                "audio_vae": ("VAE",),
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "chunk_index": ("INT", {"forceInput": True}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "seed_stride": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
                "sampler_name": (samplers,),
                "scheduler": (schedulers,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "IMAGE", "LATENT", "INT", "STRING")
    RETURN_NAMES = ("native_frames", "native_audio", "bridge_last_frame", "sampled_latent", "native_fps", "report")
    FUNCTION = "render"
    CATEGORY = CATEGORY

    def render(
        self,
        model,
        positive,
        latent,
        video_vae,
        audio_vae,
        cine_linx,
        chunk_index,
        seed,
        seed_stride,
        steps,
        sampler_name,
        scheduler,
        denoise,
        shift_video,
        shift_audio,
    ):
        import nodes as comfy_nodes
        from comfy_extras.nodes_audio import VAEDecodeAudio
        from comfy_extras.nodes_custom_sampler import BasicGuider, BasicScheduler, KSamplerSelect, RandomNoise, SamplerCustomAdvanced
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift

        shotplan = _resolve_shotplan(cine_linx)
        chunk = _chunk(shotplan, chunk_index)
        sampling = shotplan.get("sampling") if isinstance(shotplan.get("sampling"), dict) else {}
        # Shotboard schema v3 owns the sampling contract.  The node widgets are
        # retained only so older workflow JSON files remain executable.
        seed = int(sampling.get("seed", seed))
        seed_stride = int(sampling.get("seed_stride", seed_stride))
        steps = int(sampling.get("steps", steps))
        sampler_name = str(sampling.get("sampler_name", sampler_name))
        scheduler = str(sampling.get("scheduler", scheduler))
        denoise = float(sampling.get("denoise", denoise))
        shift_video = float(sampling.get("shift_video", shift_video))
        shift_audio = float(sampling.get("shift_audio", shift_audio))
        sampling_source = str(sampling.get("source", "backend_legacy_fallback"))
        actual_seed = (int(seed) + int(chunk_index) * int(seed_stride)) & 0xFFFFFFFFFFFFFFFF
        conditioning_cleanup = _release_cpu_conditioning_models(shotplan)

        turbo = _turbo_settings(shotplan)
        turbo_requested = str(turbo.get("mode", "off") or "off").lower() != "off" and bool(turbo.get("enabled", True))
        turbo_lora_name = str(turbo.get("lora_name", "") or "").strip()
        try:
            turbo_file_available = bool(turbo_lora_name and folder_paths.get_full_path("loras", turbo_lora_name))
        except Exception:
            turbo_file_available = False
        turbo_enabled = turbo_requested and turbo_file_available
        if turbo_requested and not turbo_file_available:
            steps = max(16, int(steps))
        if turbo_enabled and steps < 4:
            raise ValueError("MiniMax H3 Turbo requires at least 4 sampling steps")
        turbo_model, turbo_report = _apply_turbo_lora(model, shotplan)
        if turbo_enabled:
            # Match the proven Turbo graph: base -> LoRA -> attention patch ->
            # H3 sigma shift -> guider/scheduler/sampler.
            accelerated, acceleration_report = _accelerate(turbo_model, shotplan)
            turbo_sampler_mode = str(turbo.get("sampler_mode", "audio_fixed") or "audio_fixed").lower()
            scheduler = "simple"
            if turbo_sampler_mode == "audio_fixed":
                # Larry's AV sampler intentionally runs audio on shift 3 while
                # video remains on shift 12. Exposing 4-6 here would lie about
                # the effective schedule because the custom sampler uses 3.
                shift_audio = 3.0
            else:
                sampler_name = "res_multistep"
            active_model = MiniMaxH3SigmaShift.execute(
                model=accelerated,
                shift_video=float(shift_video),
                shift_audio=float(shift_audio),
            )[0]
        else:
            shifted = MiniMaxH3SigmaShift.execute(
                model=turbo_model,
                shift_video=float(shift_video),
                shift_audio=float(shift_audio),
            )[0]
            active_model, acceleration_report = _accelerate(shifted, shotplan)

        noise = RandomNoise.execute(noise_seed=actual_seed)[0]
        guider = BasicGuider.execute(model=active_model, conditioning=positive)[0]
        sampler_report = str(sampler_name)
        if turbo_enabled and str(turbo.get("sampler_mode", "audio_fixed")).lower() == "audio_fixed":
            sampler, sampler_report = _turbo_sampler(shotplan)
        else:
            sampler = KSamplerSelect.execute(sampler_name=str(sampler_name))[0]
        sigmas = BasicScheduler.execute(
            model=active_model,
            scheduler=str(scheduler),
            steps=int(steps),
            denoise=float(denoise),
        )[0]
        sampled = SamplerCustomAdvanced.execute(
            noise=noise,
            guider=guider,
            sampler=sampler,
            sigmas=sigmas,
            latent_image=latent,
        )[0]

        cleanup_report = "disabled"
        if bool(shotplan.get("vram_clean_before_decode", True)):
            del noise, guider, sampler, sigmas
            cleanup_report = _clean_vram_before_decode()

        native_frames = comfy_nodes.VAEDecode().decode(vae=video_vae, samples=sampled)[0]
        native_audio = VAEDecodeAudio.execute(vae=audio_vae, samples=sampled)[0]
        if not torch.is_tensor(native_frames) or native_frames.ndim != 4 or native_frames.shape[0] < 1:
            raise RuntimeError("MiniMax H3 video VAE returned no frames")
        bridge_last_frame = native_frames[-1:].detach().clone()

        report = (
            f"Atomic H3 generation | chunk={int(chunk_index) + 1}/{len(shotplan['chunks'])} | "
            f"task={chunk.get('task_mode')} | seed={actual_seed} | {steps} steps | "
            f"{sampler_report}+{scheduler} | turbo={turbo_report} | acceleration={acceleration_report} | "
            f"controls={sampling_source} | shifts={shift_video:.2f}/{shift_audio:.2f} | denoise={denoise:.2f} | "
            f"pre_sample_cleanup={conditioning_cleanup} | "
            f"pre_decode_cleanup={cleanup_report} | native_last_frame=captured"
        )
        return native_frames, native_audio, bridge_last_frame, sampled, H3_FPS, report


class IAMCCS_MiniMaxH3SequentialLTXLoaderV2:
    """Load the LTX stack only after native H3 frames exist.

    ComfyUI is free to execute independent model loaders ahead of the sampler.
    In a combined H3 -> LTX graph that can make the LTX UNET/CLIP/VAE occupy
    RAM while MiniMax is still sampling.  The IMAGE dependency is an explicit
    execution barrier: H3 must finish its native decode before any LTX file is
    opened.  We also evict the H3 model immediately before the LTX load.
    """

    @staticmethod
    def _input_spec(node_name: str, input_name: str, preferred: str | None = None):
        groups = _node_class(node_name).INPUT_TYPES()
        spec = groups.get("required", {}).get(input_name) or groups.get("optional", {}).get(input_name)
        if spec is None:
            raise RuntimeError(f"Node '{node_name}' has no input '{input_name}'")
        values = spec[0]
        metadata = dict(spec[1]) if len(spec) > 1 and isinstance(spec[1], dict) else {}
        if preferred and isinstance(values, (list, tuple)) and preferred in values:
            metadata["default"] = preferred
        return (values, metadata) if metadata else (values,)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger_frames": ("IMAGE",),
                "unet_name": cls._input_spec(
                    "UnetLoaderGGUFAdvanced", "unet_name", "ltx-2.3-22b-dev-Q4_K_S.gguf"
                ),
                "text_encoder_name": cls._input_spec(
                    "DualCLIPLoader", "clip_name1", "gemma_3_12B_it_fp8_e4m3fn.safetensors"
                ),
                "text_projection_name": cls._input_spec(
                    "DualCLIPLoader", "clip_name2", "ltx-2.3_text_projection_bf16.safetensors"
                ),
                "video_vae_name": cls._input_spec(
                    "VAELoaderKJ", "vae_name", "ltx-2.3-22b-dev_video_vae.safetensors"
                ),
                "audio_vae_name": cls._input_spec(
                    "VAELoaderKJ", "vae_name", "ltx-2.3-22b-dev_audio_vae.safetensors"
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "video_vae", "audio_vae", "report")
    FUNCTION = "load_after_h3"
    CATEGORY = CATEGORY

    def load_after_h3(
        self,
        trigger_frames,
        unet_name,
        text_encoder_name,
        text_projection_name,
        video_vae_name,
        audio_vae_name,
        clip_device="cpu",
        video_vae_device="main_device",
        video_vae_dtype="bf16",
        audio_vae_device="cpu",
        audio_vae_dtype="bf16",
    ):
        if not torch.is_tensor(trigger_frames) or trigger_frames.ndim != 4 or trigger_frames.shape[0] < 1:
            raise ValueError("Sequential LTX loader requires decoded native H3 frames")

        import comfy.model_management as model_management

        LOG.info(
            "H3 native decode complete (%d frames); unloading H3 before LTX 2.3 load",
            int(trigger_frames.shape[0]),
        )
        model_management.unload_all_models()
        gc.collect()
        model_management.soft_empty_cache()

        model = _node_class("UnetLoaderGGUFAdvanced")().load_unet(
            unet_name,
            dequant_dtype="default",
            patch_dtype="default",
            patch_on_device=False,
        )[0]
        clip = _node_class("DualCLIPLoader")().load_clip(
            text_encoder_name,
            text_projection_name,
            "ltxv",
            clip_device,
        )[0]
        vae_loader = _node_class("VAELoaderKJ")()
        video_vae = vae_loader.load_vae(video_vae_name, video_vae_device, video_vae_dtype)[0]
        audio_vae = vae_loader.load_vae(audio_vae_name, audio_vae_device, audio_vae_dtype)[0]
        report = (
            f"Sequential LTX load | barrier=after {int(trigger_frames.shape[0])} native H3 frames | "
            f"H3 evicted=yes | unet={unet_name} | clip_device={clip_device} | "
            f"video_vae={video_vae_device}/{video_vae_dtype} | "
            f"audio_vae={audio_vae_device}/{audio_vae_dtype}"
        )
        return model, clip, video_vae, audio_vae, report


class IAMCCS_MiniMaxH3PostUpscaleControlV2:
    """Resolve the selected post-upscale branch from the Shotboard CineLinX.

    The node deliberately does not load LTX or Wan itself.  It provides the
    chunk-aware values used by the real workflow branches so ComfyUI's lazy
    delivery router can avoid loading either model while upscale is disabled.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "native_frames": ("IMAGE",),
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT", "FLOAT", "INT", "BOOLEAN", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = (
        "native_frames",
        "upscale_prompt",
        "target_width",
        "target_height",
        "duration_seconds",
        "upscale_seed",
        "sage_enabled",
        "wan_denoise",
        "selected_mode",
        "report",
    )
    FUNCTION = "prepare"
    CATEGORY = CATEGORY

    def prepare(self, cine_linx, native_frames, segment_index=0):
        if not torch.is_tensor(native_frames) or native_frames.ndim != 4:
            raise ValueError("MiniMax H3 post-upscale expects native IMAGE frames")
        shotplan = _resolve_shotplan(cine_linx)
        chunks = shotplan.get("chunks") if isinstance(shotplan.get("chunks"), list) else []
        index = int(segment_index)
        if chunks and not 0 <= index < len(chunks):
            raise IndexError(f"segment_index={index} outside 0..{len(chunks) - 1}")
        chunk = chunks[index] if chunks else {}
        settings = shotplan.get("upscale_settings")
        if not isinstance(settings, dict):
            settings = {}

        enabled = bool(shotplan.get("upscale_enabled", False))
        selected_mode = str(shotplan.get("upscale_mode", "off") or "off").lower() if enabled else "off"
        if selected_mode not in {"off", "ltx23", "wan22_5b"}:
            raise ValueError(f"Unknown MiniMax H3 post-upscale mode: {selected_mode}")

        native_width = int(shotplan.get("width", int(native_frames.shape[2])) or int(native_frames.shape[2]))
        native_height = int(shotplan.get("height", int(native_frames.shape[1])) or int(native_frames.shape[1]))
        target_width = max(256, int(settings.get("target_width", native_width * 2) or native_width * 2))
        target_height = max(256, int(settings.get("target_height", native_height * 2) or native_height * 2))
        prompt = str(settings.get("prompt") or chunk.get("prompt") or shotplan.get("global_prompt") or "high quality cinematic video").strip()
        duration_seconds = float(
            chunk.get("duration_seconds")
            or (max(1, int(native_frames.shape[0])) / H3_FPS)
        )
        sampling = shotplan.get("sampling") if isinstance(shotplan.get("sampling"), dict) else {}
        base_seed = int(sampling.get("seed", 0) or 0)
        seed_stride = int(sampling.get("seed_stride", 1) or 1)
        seed_offset = int(settings.get("seed_offset", 10000) or 0)
        upscale_seed = (base_seed + index * seed_stride + seed_offset) & 0xFFFFFFFFFFFFFFFF
        sage_enabled = bool(settings.get("sage", True))
        wan_denoise = min(1.0, max(0.0, float(settings.get("wan_denoise", 0.2) or 0.0)))
        report = (
            f"H3 post-upscale control | selected={selected_mode} | chunk={index + 1}/{max(1, len(chunks))} | "
            f"native={native_width}x{native_height} -> target={target_width}x{target_height} | "
            f"duration={duration_seconds:.3f}s | seed={upscale_seed} | sage={'on' if sage_enabled else 'off'} | "
            f"wan_denoise={wan_denoise:.2f} | lazy=yes"
        )
        return (
            native_frames,
            prompt,
            target_width,
            target_height,
            duration_seconds,
            upscale_seed,
            sage_enabled,
            wan_denoise,
            selected_mode,
            report,
        )


class IAMCCS_MiniMaxH3DeliveryRouterV2:
    """Lazy upscale selection plus optional RIFE, preserving the native bridge."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "native_frames": ("IMAGE",),
                "native_audio": ("AUDIO",),
                "bridge_last_frame": ("IMAGE",),
            },
            "optional": {
                "ltx23_upscaled_frames": ("IMAGE", {"lazy": True}),
                "wan22_upscaled_frames": ("IMAGE", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "IMAGE", "INT", "STRING", "STRING")
    RETURN_NAMES = ("delivery_frames", "audio", "bridge_last_frame", "delivery_fps", "upscale_applied", "report")
    FUNCTION = "route"
    CATEGORY = CATEGORY

    @staticmethod
    def _selected_upscale_input(cine_linx):
        shotplan = _resolve_shotplan(cine_linx)
        if not bool(shotplan.get("upscale_enabled", False)):
            return None
        mode = str(shotplan.get("upscale_mode", "off") or "off").lower()
        if mode == "ltx23":
            return "ltx23_upscaled_frames"
        if mode == "wan22_5b":
            return "wan22_upscaled_frames"
        return None

    def check_lazy_status(
        self,
        cine_linx,
        native_frames,
        native_audio,
        bridge_last_frame,
        ltx23_upscaled_frames=None,
        wan22_upscaled_frames=None,
        **kwargs,
    ):
        selected = self._selected_upscale_input(cine_linx)
        if selected == "ltx23_upscaled_frames" and ltx23_upscaled_frames is None:
            return [selected]
        if selected == "wan22_upscaled_frames" and wan22_upscaled_frames is None:
            return [selected]
        return []

    @staticmethod
    def _rife(frames: torch.Tensor, mode: str):
        if mode == "off":
            return frames, H3_FPS, "RIFE off"
        rife_cls = _node_class("RIFE VFI")
        if mode == "rife_48fps":
            output = rife_cls().vfi(
                ckpt_name="rife49.pth",
                frames=frames,
                clear_cache_after_n_frames=5,
                multiplier=2,
                fast_mode=True,
                ensemble=True,
                scale_factor=1.0,
            )[0]
            return output, 48, "RIFE 2x -> 48 fps"
        if mode == "rife_60fps":
            dense = rife_cls().vfi(
                ckpt_name="rife49.pth",
                frames=frames,
                clear_cache_after_n_frames=5,
                multiplier=5,
                fast_mode=True,
                ensemble=True,
                scale_factor=1.0,
            )[0]
            # RIFE creates the exact 120 fps temporal lattice. Taking every
            # second frame produces a true 60 fps lattice without changing the
            # H3 generation clock or the native last-frame bridge.
            output = dense[::2]
            if (int(dense.shape[0]) - 1) % 2:
                output = torch.cat([output, dense[-1:]], dim=0)
            return output, 60, "RIFE 5x lattice -> 60 fps decimation"
        raise ValueError(f"Unknown RIFE mode: {mode}")

    def route(
        self,
        cine_linx,
        native_frames,
        native_audio,
        bridge_last_frame,
        ltx23_upscaled_frames=None,
        wan22_upscaled_frames=None,
    ):
        shotplan = _resolve_shotplan(cine_linx)
        selected = self._selected_upscale_input(shotplan)
        if selected == "ltx23_upscaled_frames":
            if not torch.is_tensor(ltx23_upscaled_frames):
                raise ValueError("Shotboard enabled LTX 2.3 upscale, but the lazy LTX output is not connected")
            delivery = ltx23_upscaled_frames
            upscale_report = "LTX 2.3"
        elif selected == "wan22_upscaled_frames":
            if not torch.is_tensor(wan22_upscaled_frames):
                raise ValueError("Shotboard enabled Wan 2.2 5B upscale, but the lazy Wan output is not connected")
            delivery = wan22_upscaled_frames
            upscale_report = "Wan 2.2 5B"
        else:
            delivery = native_frames
            upscale_report = "off/native"

        post_upscale_cleanup = "not required"
        if upscale_report != "off/native":
            # Cleanup belongs here, after the lazy branch has actually been
            # selected and decoded.  An OUTPUT_NODE inside an LTX/Wan
            # subgraph is not lazy in ComfyUI and would force that entire
            # branch to execute even when Shotboard upscale is disabled.
            import comfy.model_management as model_management

            model_management.unload_all_models()
            gc.collect()
            model_management.soft_empty_cache()
            post_upscale_cleanup = "models unloaded after selected upscale"

        rife_mode = str(shotplan.get("rife_mode", "off") or "off").lower()
        delivery, delivery_fps, rife_report = self._rife(delivery, rife_mode)
        report = (
            f"H3 delivery route | upscale={upscale_report} | {rife_report} | "
            f"bridge=last native H3 frame before upscale/RIFE | lazy_upscale=yes | {post_upscale_cleanup}"
        )
        return delivery, native_audio, bridge_last_frame, int(delivery_fps), upscale_report, report


NODE_CLASS_MAPPINGS = {
    "IAMCCS_MiniMaxH3AtomicModelRouter": IAMCCS_MiniMaxH3AtomicModelRouter,
    "IAMCCS_MiniMaxH3AtomicConditioningBackend": IAMCCS_MiniMaxH3AtomicConditioningBackend,
    "IAMCCS_MiniMaxH3GenerationBackendV2": IAMCCS_MiniMaxH3GenerationBackendV2,
    "IAMCCS_MiniMaxH3SequentialLTXLoaderV2": IAMCCS_MiniMaxH3SequentialLTXLoaderV2,
    "IAMCCS_MiniMaxH3PostUpscaleControlV2": IAMCCS_MiniMaxH3PostUpscaleControlV2,
    "IAMCCS_MiniMaxH3DeliveryRouterV2": IAMCCS_MiniMaxH3DeliveryRouterV2,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_MiniMaxH3AtomicModelRouter": "MiniMax H3 Atomic FL2VA / REF2VA Model Router",
    "IAMCCS_MiniMaxH3AtomicConditioningBackend": "MiniMax H3 Atomic Shotboard Conditioning",
    "IAMCCS_MiniMaxH3GenerationBackendV2": "MiniMax H3 Generation V2 (Acceleration + Clean Decode)",
    "IAMCCS_MiniMaxH3SequentialLTXLoaderV2": "MiniMax H3 -> LTX Sequential Loader V2",
    "IAMCCS_MiniMaxH3PostUpscaleControlV2": "MiniMax H3 Post-Upscale Control V2 (LTX / Wan)",
    "IAMCCS_MiniMaxH3DeliveryRouterV2": "MiniMax H3 Delivery V2 (Lazy Upscale + RIFE)",
}
