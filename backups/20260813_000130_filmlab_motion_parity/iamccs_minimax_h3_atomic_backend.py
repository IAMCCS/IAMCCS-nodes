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

import comfy.utils
import folder_paths


SHOTPLAN_TYPE = "IAMCCS_MINIMAX_H3_SHOTPLAN"
SUPERNODE_LINX_TYPE = "IAMCCS_SUPERNODE_LINX"
CATEGORY = "IAMCCS/MiniMax H3/Atomic Backend"
H3_FPS = 24
LOG = logging.getLogger("IAMCCS.MiniMaxH3.AtomicBackend")


def _context_owned_fl2va_prompt(prompt: str) -> str:
    """Drop the stale opening-anchor instruction when AV context owns frame zero."""
    opening = "Picture 1 defines the complete opening frame at 0.00 seconds. "
    final = "Picture 2 defines the complete final frame"
    text = str(prompt or "")
    if text.startswith(opening):
        text = text[len(opening):]
    return text.replace(final, "The supplied final keyframe defines the complete final frame", 1)


def _is_true_4k_delivery(width: Any, height: Any) -> bool:
    try:
        target_width = max(0, int(width))
        target_height = max(0, int(height))
    except (TypeError, ValueError, OverflowError):
        return False
    return max(target_width, target_height) >= 3840 and min(target_width, target_height) >= 1600


def _ceil_multiple(value: float, multiple: int = 32) -> int:
    return max(multiple, int(math.ceil(float(value) / multiple) * multiple))


def _resolve_ltx_4k_stage(
    native_width: int,
    native_height: int,
    delivery_width: int,
    delivery_height: int,
    requested: bool,
) -> tuple[bool, int, int, str]:
    """Protect native detail while resolving the optional LTX -> RTX chain.

    The LTX workflow uses a 2x latent upsampler, so its pre-LTX input is half
    of ``stage_target``.  A blind delivery/2 rule therefore downsampled native
    H3 frames before enhancement.  This resolver guarantees that the implicit
    pre-input is never smaller than the actual native frame in either axis.
    """
    def direct_stage(reason: str) -> tuple[bool, int, int, str]:
        # LTX's latent upsampler is 2x. If delivery is below native*2, render a
        # protected processing intermediate first; DeliveryRouterV2 performs
        # the final exact-size downscale after decoding.
        direct_fraction = max(
            1.0,
            (native_width * 2.0) / max(1, delivery_width),
            (native_height * 2.0) / max(1, delivery_height),
        )
        stage_width = _ceil_multiple(delivery_width * direct_fraction, 32)
        stage_height = _ceil_multiple(delivery_height * direct_fraction, 32)
        stage_width = max(stage_width, _ceil_multiple(native_width * 2, 32))
        stage_height = max(stage_height, _ceil_multiple(native_height * 2, 32))
        protected = stage_width != delivery_width or stage_height != delivery_height
        suffix = (
            f"; protected LTX processing={stage_width}x{stage_height}, exact delivery follows"
            if protected
            else "; direct LTX processing already preserves native pixels"
        )
        return False, stage_width, stage_height, reason + suffix

    if not requested:
        return direct_stage("RTX 4K not requested")
    if not _is_true_4k_delivery(delivery_width, delivery_height):
        return direct_stage("RTX 4K rejected: delivery target is below UHD/DCI class")
    if delivery_width < native_width * 2 or delivery_height < native_height * 2:
        return direct_stage("RTX 4K bypassed: direct protected LTX preserves more native detail")

    stage_fraction = max(
        0.5,
        (native_width * 2.0) / max(1, delivery_width),
        (native_height * 2.0) / max(1, delivery_height),
    )
    if stage_fraction >= 0.999:
        return direct_stage("RTX 4K bypassed: no useful second-stage scale remains")

    stage_width = min(delivery_width, _ceil_multiple(delivery_width * stage_fraction, 32))
    stage_height = min(delivery_height, _ceil_multiple(delivery_height * stage_fraction, 32))
    stage_width = max(stage_width, _ceil_multiple(native_width * 2, 32))
    stage_height = max(stage_height, _ceil_multiple(native_height * 2, 32))
    if stage_width >= delivery_width and stage_height >= delivery_height:
        return direct_stage("RTX 4K bypassed: protected LTX stage already equals delivery")
    return (
        True,
        stage_width,
        stage_height,
        f"protected LTX stage; implicit pre-input >= {native_width}x{native_height}",
    )


def _resize_frames_exact_cpu(images: torch.Tensor, target_width: int, target_height: int):
    """Finish a protected LTX intermediate at the exact delivery canvas."""
    if not torch.is_tensor(images) or images.ndim != 4:
        return images, "exact delivery resize unavailable"
    source_height = int(images.shape[1])
    source_width = int(images.shape[2])
    if source_width == int(target_width) and source_height == int(target_height):
        return images, f"exact delivery already {source_width}x{source_height}"

    source = images.detach().to(device="cpu")
    frame_count = int(source.shape[0])
    channels = int(source.shape[-1])
    output = torch.empty(
        (frame_count, int(target_height), int(target_width), channels),
        dtype=torch.float16,
        device="cpu",
    )
    chunk_size = min(8, frame_count)
    for start in range(0, frame_count, chunk_size):
        end = min(frame_count, start + chunk_size)
        resized = F.interpolate(
            source[start:end].to(dtype=torch.float32).movedim(-1, 1),
            size=(int(target_height), int(target_width)),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).movedim(1, -1).clamp_(0.0, 1.0)
        output[start:end].copy_(resized.to(dtype=torch.float16))
        del resized
    del source
    gc.collect()
    return output, f"protected exact delivery {source_width}x{source_height}->{target_width}x{target_height}"


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


def _cine_info_h3(cine_linx: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(cine_linx, dict):
        return {}, {}
    resources = cine_linx.get("resources") if isinstance(cine_linx.get("resources"), dict) else {}
    config = resources.get("iamccs_minimax_h3_cine_info")
    return (config if isinstance(config, dict) else {}), resources


def _effective_task(cine_linx: Any, chunk: dict[str, Any]) -> str:
    config, _ = _cine_info_h3(cine_linx)
    override = str(config.get("task_override", "from_shotboard") or "from_shotboard").lower()
    if override in {"t2va", "i2va", "fl2va", "ref2va"}:
        return override
    return str(chunk.get("task_mode", "t2va") or "t2va").lower()


def _effective_shotplan(cine_linx: Any, shotplan: dict[str, Any]) -> dict[str, Any]:
    config, _ = _cine_info_h3(cine_linx)
    if not config:
        return shotplan
    result = dict(shotplan)
    for key in ("reference_roles", "reference_video_role", "reference_audio_role", "ref_image_size"):
        if key in config:
            result[key] = config[key]
    if isinstance(config.get("reference_resize"), dict):
        resize_override = dict(config["reference_resize"])
        resize_policy = str(resize_override.get("policy", "from_shotboard") or "from_shotboard").lower()
        if resize_policy not in {"from_shotboard", "inherit", "shotboard"}:
            result["reference_resize"] = resize_override
    result["reference_source"] = str(config.get("reference_source", "cine_info_h3_only"))
    return result


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


def _flf_bridge_path(render_id: str) -> Path | None:
    safe_render_id = str(render_id or "").strip()
    if not safe_render_id:
        return None
    return Path(folder_paths.get_output_directory()) / "minimax_h3_shotboard" / "bridges" / f"last_frame_{safe_render_id}.png"


def _load_flf_bridge(render_id: str) -> torch.Tensor | None:
    path = _flf_bridge_path(render_id)
    if path is None or not path.is_file():
        return None
    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    return _load_image(str(path))


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
    """Optionally pre-size an IMAGE batch before native H3 conditioning.

    Quality profiles leave this off so ComfyUI performs one Lanczos fit inside
    the native MiniMax conditioner. Explicit Low VRAM policies also cover
    timeline keyframes, bridge frames, REF2VA stills and reference-video
    batches.
    """
    if not torch.is_tensor(image) or image.ndim != 4:
        return image, f"{label}=none"
    settings = shotplan.get("reference_resize")
    if not isinstance(settings, dict):
        settings = {}
    policy = str(settings.get("policy", "off") or "off").lower()
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
    for path in shotplan.get("reference_image_paths", []):
        clean = str(path or "").strip()
        if clean and clean not in result:
            result.append(clean)
        if len(result) >= 4:
            return result
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
    """Keep Qwen3-VL on ComfyUI's automatic GPU-first placement.

    ``cpu_safe_12gb`` is retained only as a legacy workflow value.  It no
    longer pins the encoder to CPU because doing so turns prompt encoding into
    the dominant runtime cost.  A CPU clone is created only by the OOM retry
    path below, after a real CUDA allocation failure.
    """
    mode = str(shotplan.get("text_encoder_device", "auto") or "auto").lower()
    if mode not in {"auto", "cpu_safe_12gb"}:
        raise ValueError(f"Unsupported MiniMax H3 text encoder device mode: {mode}")
    if mode == "cpu_safe_12gb":
        return clip, "auto(gpu-first; migrated legacy cpu_safe_12gb)"
    return clip, "auto(gpu-first)"


def _text_encoder_dynamic_reserve_mb(clip) -> int:
    """Leave activation headroom when the 32B multimodal encoder uses a small GPU.

    ComfyUI's generic CLIP loader has no MiniMax-H3-specific activation estimate.
    With a 12 GB card it can therefore stage the complete ~9.6 GB Qwen3-VL GGUF,
    leaving too little room for the vision tower and two FL2VA image streams.  A
    temporary memory estimate makes CoreModelPatcher keep only part of the
    weights resident while the remaining layers stream from CPU.  This is still
    GPU-first execution; it is not the much slower all-CPU fallback.
    """
    patcher = getattr(clip, "patcher", None)
    device = getattr(patcher, "load_device", None)
    if getattr(device, "type", str(device)) != "cuda" or not torch.cuda.is_available():
        return 0
    try:
        total_gib = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    except Exception:
        return 0
    if total_gib <= 13.0:
        return 3584
    if total_gib <= 17.0:
        return 2560
    return 0


def _install_text_encoder_memory_estimate(clip, reserve_mb: int):
    """Temporarily add/raise CLIP's activation estimate; return a restore callback."""
    stage = getattr(clip, "cond_stage_model", None)
    if stage is None or reserve_mb <= 0:
        return lambda: None

    instance_dict = getattr(stage, "__dict__", {})
    had_instance_value = "memory_estimation_function" in instance_dict
    previous_instance_value = instance_dict.get("memory_estimation_function")
    previous = getattr(stage, "memory_estimation_function", None)
    reserve_bytes = int(reserve_mb) * 1024 * 1024

    def estimate(tokens, device=None):
        base = 0
        if callable(previous):
            try:
                base = int(previous(tokens, device=device))
            except TypeError:
                base = int(previous(tokens))
        return max(base, reserve_bytes)

    stage.memory_estimation_function = estimate

    def restore():
        if had_instance_value:
            stage.memory_estimation_function = previous_instance_value
        else:
            try:
                delattr(stage, "memory_estimation_function")
            except AttributeError:
                pass

    return restore


def _is_cuda_oom(exc: BaseException) -> bool:
    oom_type = getattr(torch, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "cuda out of memory",
            "cuda error: out of memory",
            "cudamalloc",
            "cuda memory allocation",
        )
    )


def _clear_conditioning_cuda_after_oom() -> str:
    """Release the failed GPU attempt before the one allowed CPU retry."""
    notes: list[str] = []
    try:
        import comfy.model_management as mm

        mm.unload_all_models()
        notes.append("unload_all_models")
        try:
            mm.cleanup_models()
            notes.append("cleanup_models")
        except Exception as exc:
            notes.append(f"cleanup warning: {exc}")
        mm.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        notes.append("empty CUDA cache")
    except Exception as exc:
        notes.append(f"cleanup warning: {exc}")
    gc.collect()
    return ", ".join(notes)


def _run_h3_conditioning_with_cpu_fallback(clip, shotplan: dict[str, Any], execute_fn):
    """Encode GPU-first with activation headroom; retry on CPU only after CUDA OOM."""
    active_clip, placement_report = _place_text_encoder(clip, shotplan)
    reserve_mb = _text_encoder_dynamic_reserve_mb(active_clip)
    restore_estimate = _install_text_encoder_memory_estimate(active_clip, reserve_mb)
    if reserve_mb:
        placement_report += f"+dynamic_reserve={reserve_mb}MB"
        LOG.info(
            "MiniMax H3 low-VRAM text encode reserve active: %d MB kept for Qwen3-VL activations; weights remain GPU-first with dynamic offload",
            reserve_mb,
        )
    try:
        try:
            return execute_fn(active_clip), placement_report
        finally:
            restore_estimate()
    except Exception as exc:
        if not _is_cuda_oom(exc):
            raise

        oom_exc = exc
        cleanup_report = _clear_conditioning_cuda_after_oom()
        LOG.warning(
            "MiniMax H3 text conditioning exhausted CUDA memory; retrying once on CPU after %s",
            cleanup_report,
        )

    from comfy_extras.nodes_multigpu import SelectCLIPDeviceNode

    cpu_clip = SelectCLIPDeviceNode.execute(clip=clip, device="cpu")[0]
    load_device = getattr(getattr(cpu_clip, "patcher", None), "load_device", None)
    if getattr(load_device, "type", str(load_device)) != "cpu":
        raise RuntimeError(
            "MiniMax H3 could not activate its CPU fallback after a CUDA OOM. "
            "Update ComfyUI and retry."
        ) from oom_exc
    return execute_fn(cpu_clip), f"auto->cpu_fallback(cuda_oom; {cleanup_report})"


def _apply_h3_memory_efficient_sage(model):
    sage_cls = _node_class("MiniMaxH3MemoryEfficientSageAttentionPatch")
    return sage_cls.execute(model=model)[0]


def _apply_h3_low_vram_exact(model):
    attention_cls = _node_class("MiniMaxLowVRAMAttention")
    patched = attention_cls.execute(model=model, head_chunks=4)[0]
    feed_forward_cls = _node_class("MiniMaxChunkFeedForward")
    return feed_forward_cls.execute(model=patched, chunks=2, seq_threshold=4096)[0]


def _apply_h3_sage_low_vram(model):
    return _apply_h3_low_vram_exact(_apply_h3_memory_efficient_sage(model))


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


def _apply_adaptive_cache(model, preset: str):
    cache_cls = _node_class("MiniMaxH3AdaptiveCache")
    return cache_cls().patch(model=model, preset=preset, cache_device="auto")[0]


def _apply_spectrum(model, profile: str):
    spectrum_cls = _node_class("SpectrumApplyMiniMaxH3")
    profile = str(profile or "low_vram").lower()
    aggressive = profile == "aggressive"
    quality = profile in {"conservative_quality", "quality"}
    degree = 4 if quality else 1
    warmup_steps = 5 if quality else 1
    max_history = 8 if quality else 2
    return spectrum_cls().apply(
        model=model,
        enabled=True,
        blend_weight=0.75 if aggressive else 0.50,
        degree=degree,
        ridge_lambda=0.10,
        window_size=2.0,
        flex_window=3.0 if aggressive else 0.75,
        warmup_steps=warmup_steps,
        tail_actual_steps=1,
        max_history=max_history,
        debug=False,
        history_storage="system_ram",
        bootstrap_first_forecast=not quality,
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

    # Kijai's Lightx2v conversion is a standard Comfy model-only LoRA.  Do not
    # route it through Larry's custom sampler package: the proven high-quality
    # graph uses the native loader at a moderate strength (normally 0.7).
    lower_name = lora_name.lower()
    if "lightx2v" in lower_name:
        import nodes as comfy_nodes

        patched = comfy_nodes.LoraLoaderModelOnly().load_lora_model_only(
            model=model,
            lora_name=lora_name,
            strength_model=strength,
        )[0]
        return patched, f"native Lightx2v LoRA {lora_name}@{strength:.2f}"

    # drbaph's pruned conversion intentionally removes incompatible AdaLN
    # pairs and is designed for ComfyUI's native model-only loader. Larry's
    # original/full LoRA uses the custom loader, which can also re-inject the
    # full-width time-conditioning update on pruned bases.
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
    if mode in {"auto_3060", "low_vram_auto"}:
        try:
            return _apply_h3_sage_low_vram(model), "Low VRAM Auto -> H3 Sage + exact attention/FFN chunks"
        except Exception as h3_exc:
            try:
                return _apply_h3_memory_efficient_sage(model), f"Low VRAM Auto -> H3 Sage (exact chunks unavailable: {h3_exc})"
            except Exception as sage_exc:
                return model, f"Low VRAM Auto -> native (H3 stack: {h3_exc}; H3 Sage: {sage_exc})"
    if mode == "native":
        return model, "native"
    if mode in {"sage", "h3_sage"}:
        return _apply_h3_sage_low_vram(model), "MiniMax H3 Sage + exact attention/FFN chunks"
    if mode in {"sage_sol", "sol_low_vram"}:
        patched = _apply_h3_low_vram_exact(model)
        patched = _apply_sol(patched, str(shotplan.get("sol_conditioning", "exact_kv")))
        return patched, f"Sol({shotplan.get('sol_conditioning', 'exact_kv')}) + exact Low VRAM chunks"
    if mode in {"adaptive_safe", "sol_adaptive_safe", "sol_adaptive_balanced"}:
        patched = _apply_h3_low_vram_exact(model)
        if mode.startswith("sol_"):
            patched = _apply_sol(patched, str(shotplan.get("sol_conditioning", "exact_kv_and_rows")))
        preset = "balanced" if mode.endswith("balanced") else "safe"
        patched = _apply_adaptive_cache(patched, preset)
        prefix = "Sol + " if mode.startswith("sol_") else ""
        return patched, f"{prefix}Adaptive Cache {preset} + exact Low VRAM chunks"
    if mode == "spectrum":
        profile = str(shotplan.get("spectrum_profile", "low_vram"))
        return _apply_spectrum(model, profile), f"Spectrum({profile},system_ram)"
    if mode == "sage_spectrum":
        profile = str(shotplan.get("spectrum_profile", "low_vram"))
        sage_model = _apply_h3_sage_low_vram(model)
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


def _release_conditioning_models(shotplan: dict[str, Any]) -> str:
    """Strict barrier after conditioning and immediately before H3 sampling.

    Positive conditioning and the AV latent are already materialized when the
    generation node runs.  Qwen3-VL (and any conditioning-time VAE residency)
    can therefore be unloaded before the H3 model is requested.
    """
    try:
        import comfy.model_management as mm

        mm.unload_all_models()
        try:
            mm.cleanup_models()
        except Exception:
            pass
        mm.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        LOG.warning("MiniMax H3 pre-sampler conditioning cleanup warning: %s", exc)
        return f"conditioning cleanup warning: {exc}"
    gc.collect()
    if os.name == "nt":
        try:
            import ctypes

            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.psapi.EmptyWorkingSet(handle)
            report = "conditioning models unloaded; CUDA cache cleared; Windows working set trimmed"
            LOG.info("MiniMax H3 pre-sampler barrier: %s", report)
            return report
        except Exception as exc:
            LOG.warning("MiniMax H3 working-set trim warning: %s", exc)
    report = "conditioning models unloaded; CUDA cache cleared"
    LOG.info("MiniMax H3 pre-sampler barrier: %s", report)
    return report


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
                "motion_context": ("IAMCCS_H3_MOTION_CONTEXT",),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING", "STRING")
    RETURN_NAMES = ("model", "model_family", "report")
    FUNCTION = "select"
    CATEGORY = CATEGORY

    @staticmethod
    def _input_name(cine_linx, segment_index, motion_context=None):
        del motion_context
        chunk = _chunk(cine_linx, segment_index)
        task = _effective_task(cine_linx, chunk)
        return "ref2va_model" if _task_family(task) == "ref2va" else "fl2va_model"

    def check_lazy_status(self, cine_linx, segment_index, fl2va_model=None, ref2va_model=None, motion_context=None, **kwargs):
        selected = self._input_name(cine_linx, segment_index, motion_context)
        if selected == "ref2va_model" and ref2va_model is None:
            return ["ref2va_model"]
        if selected == "fl2va_model" and fl2va_model is None:
            return ["fl2va_model"]
        return []

    def select(self, cine_linx, segment_index, fl2va_model=None, ref2va_model=None, motion_context=None):
        chunk = _chunk(cine_linx, segment_index)
        task = _effective_task(cine_linx, chunk)
        selected = self._input_name(cine_linx, segment_index, motion_context)
        family = "ref2va" if selected == "ref2va_model" else _task_family(task)
        model = ref2va_model if family == "ref2va" else fl2va_model
        if model is None:
            raise ValueError(
                f"Atomic H3 mode selected {family}, but its MODEL input is not connected. "
                f"Connect the {family}_model branch."
            )
        # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
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
                "render_id": ("STRING", {"default": ""}),
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
                "motion_context": ("IAMCCS_H3_MOTION_CONTEXT",),
            },
        }

    RETURN_TYPES = (
        "MODEL", "CONDITIONING", "LATENT", "IMAGE", "IMAGE", "STRING",
        "STRING", "INT", "INT", "INT", "STRING", "IAMCCS_H3_MOTION_CONTEXT",
    )
    RETURN_NAMES = (
        "model", "positive", "latent", "first_frame", "planned_last_frame",
        "reference_manifest_json", "prompt", "current_segment", "total_segments",
        "trim_head_frames", "report", "motion_state",
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
        render_id="",
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
        motion_context=None,
    ):
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo

        shotplan = _effective_shotplan(cine_linx, _resolve_shotplan(cine_linx))
        chunk = _chunk(shotplan, segment_index)
        task = _effective_task(cine_linx, chunk)
        width = int(shotplan.get("width", 960))
        height = int(shotplan.get("height", 544))
        visible_frames = max(5, int(chunk.get("frame_count", 124)))
        while visible_frames % 17 != 5:
            visible_frames += 1
        frames = visible_frames
        motion_state = {
            "active": False,
            "trim_frames": 0,
            "export_frames": visible_frames,
            "sample_frames": visible_frames,
            "segment_index": int(segment_index),
            "method": "planned_fl2va_keyframes",
        }
        # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        # Prompter changes are already baked into the chunk before execution.
        prompt = str(chunk.get("prompt", "")).strip() or str(prompt_override or "").strip()

        planned_first = _load_image(str(chunk.get("first_image", "")))
        planned_last = _load_image(str(chunk.get("last_image", "")))
        del bridge_frame, render_id, motion_context
        # Director FL2V parity: both planned endpoints remain hard keyframes.
        first = planned_first
        if first is None and torch.is_tensor(first_frame_override):
            first = first_frame_override[:1]
        last = planned_last if planned_last is not None else (last_frame_override[:1] if torch.is_tensor(last_frame_override) else None)
        h3_info, h3_resources = _cine_info_h3(cine_linx)
        socket_images = [ref_image_1, ref_image_2, ref_image_3, ref_image_4]
        resource_images = [h3_resources.get(f"iamccs_minimax_h3_ref_image_{index}") for index in range(1, 5)]
        external_images = [
            socket if torch.is_tensor(socket) else resource
            for socket, resource in zip(socket_images, resource_images)
        ]
        if not torch.is_tensor(ref_video):
            ref_video = h3_resources.get("iamccs_minimax_h3_ref_video")
        if not isinstance(ref_video_audio, dict):
            ref_video_audio = h3_resources.get("iamccs_minimax_h3_ref_video_audio")
        if not isinstance(ref_audio, dict):
            ref_audio = h3_resources.get("iamccs_minimax_h3_ref_audio")

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
            result, text_encoder_report = _run_h3_conditioning_with_cpu_fallback(
                clip,
                shotplan,
                lambda active_clip: MiniMaxH3ImageToVideo.execute(
                    clip=active_clip, vae=video_vae, prompt=prompt,
                    width=width, height=height, length=frames,
                    first_frame=None, last_frame=None,
                ),
            )
        elif task == "i2va":
            if first is None:
                raise ValueError("I2VA requires one opening image in the Shotboard or ref_image_1")
            result, text_encoder_report = _run_h3_conditioning_with_cpu_fallback(
                clip,
                shotplan,
                lambda active_clip: MiniMaxH3ImageToVideo.execute(
                    clip=active_clip, vae=video_vae, prompt=prompt,
                    width=width, height=height, length=frames,
                    first_frame=first, last_frame=None,
                ),
            )
            manifest.append({"label": "<Picture 1>", "role": "opening_keyframe"})
        elif task == "fl2va":
            if last is None or first is None:
                raise ValueError("FL2VA requires both opening and final images; connect ref_image_1/ref_image_2 or place two adjacent Shotboard keyframes")
            result, text_encoder_report = _run_h3_conditioning_with_cpu_fallback(
                clip,
                shotplan,
                lambda active_clip: MiniMaxH3ImageToVideo.execute(
                    clip=active_clip, vae=video_vae, prompt=prompt,
                    width=width, height=height, length=frames,
                    first_frame=first, last_frame=last,
                ),
            )
            manifest.extend([
                {"label": "<Picture 1>", "role": "opening_keyframe"},
                {"label": "<Picture 2>", "role": "final_keyframe"},
            ])
        elif task.startswith("ref2va"):
            roles = _reference_roles(shotplan)
            plan_paths = [] if str(h3_info.get("reference_source", "")) == "cine_info_h3_only" else _unique_plan_image_paths(shotplan)
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
            voice_pairing_note = ""
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
                audio_label = f"<Audio {audio_ordinal}>"
                manifest.append({"label": audio_label, "role": audio_role if audio_role != "off" else "driven_audio"})
                voice_picture_index = int(shotplan.get("voice_reference_picture_index", 0) or 0)
                if voice_picture_index > 0:
                    voice_picture_label = f"<Picture {voice_picture_index}>"
                    if any(item["label"] == voice_picture_label for item in manifest):
                        manifest.append({"label": audio_label, "role": f"voice_timbre_for_{voice_picture_label}"})
                        voice_pairing_note = (
                            f"The voice in {audio_label} is the timbre reference for the character shown in "
                            f"{voice_picture_label}; keep that character's dialogue in this voice."
                        )
                    else:
                        LOG.warning(
                            "MiniMax H3 voice cloning: voice_reference_picture_index=%d has no matching %s in this segment",
                            voice_picture_index,
                            voice_picture_label,
                        )
                        voice_pairing_note = ""
                else:
                    voice_pairing_note = ""

            if not refs and not ref_videos and not ref_audios:
                raise ValueError("REF2VA requires at least one enabled image, video, or audio reference")
            prompt = _reference_header(manifest, prompt)
            if voice_pairing_note:
                prompt = f"{prompt}\n\n{voice_pairing_note}"
            result, text_encoder_report = _run_h3_conditioning_with_cpu_fallback(
                clip,
                shotplan,
                lambda active_clip: MiniMaxH3ReferenceToVideo.execute(
                    clip=active_clip,
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
                ),
            )
        else:
            raise ValueError(f"Unsupported atomic H3 task: {task}")

        positive, latent = result[0], result[1]
        motion_report = "director_fl2v_planned_keyframes"
        LOG.info(
            "MiniMax H3 conditioning complete | task=%s | text_encoder=%s | pre-sampler unload barrier is next",
            task,
            text_encoder_report,
        )
        report = (
            f"Atomic H3 conditioning | segment={int(segment_index) + 1}/{len(shotplan['chunks'])} | "
            f"task={task} | frames={frames} | first={'yes' if first is not None else 'no'} | "
            f"last={'yes' if last is not None else 'no'} | refs={len(manifest)} | "
            f"ref_size={shotplan.get('ref_image_size', 'match')} | "
            f"ref_source={shotplan.get('reference_source', 'backend_sockets_or_legacy_timeline')} | "
            f"pre_resize={';'.join(resize_reports) if resize_reports else 'none'} | "
            f"text_encoder={text_encoder_report} | motion_context={motion_report}"
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
            motion_state,
        )


class IAMCCS_MiniMaxH3DirectorFLFParityModelRouter(IAMCCS_MiniMaxH3AtomicModelRouter):
    """Director-equivalent FL2VA router under an IAMCCS node identity."""


class IAMCCS_MiniMaxH3DirectorFLFParityConditioning(IAMCCS_MiniMaxH3AtomicConditioningBackend):
    """Director-equivalent FL2VA conditioning under an IAMCCS node identity."""


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
            },
            "optional": {"motion_state": ("IAMCCS_H3_MOTION_CONTEXT",)},
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
        motion_state=None,
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
        context_active = isinstance(motion_state, dict) and bool(motion_state.get("active"))
        actual_seed = (
            int(seed)
            if context_active
            else (int(seed) + int(chunk_index) * int(seed_stride))
        ) & 0xFFFFFFFFFFFFFFFF
        seed_contract = "director_base_seed" if context_active else "shotboard_seed_stride"
        conditioning_cleanup = _release_conditioning_models(shotplan)

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
        if turbo_enabled:
            turbo_mode = str(turbo.get("mode", "off") or "off").lower()
            minimum_steps = 6 if turbo_mode == "ckpt500_6_8" else 8
            if int(steps) < minimum_steps:
                LOG.warning(
                    "MiniMax H3 Turbo steps corrected from %d to %d for %s (%s)",
                    int(steps),
                    minimum_steps,
                    turbo_mode,
                    turbo_lora_name,
                )
                steps = minimum_steps
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
        if isinstance(native_audio, dict):
            native_audio = dict(native_audio)
            native_audio["iamccs_flf_locked_audio_handles"] = bool(
                str(shotplan.get("continuation_mode", "")) == "flf_image_center_bridges"
                and str(shotplan.get("audio_mode", "")) == "h3_custom_audio_drive"
            )
        if not torch.is_tensor(native_frames) or native_frames.ndim != 4 or native_frames.shape[0] < 1:
            raise RuntimeError("MiniMax H3 video VAE returned no frames")
        if isinstance(motion_state, dict) and bool(motion_state.get("active")):
            trim_frames = max(0, int(motion_state.get("trim_frames", 0)))
            export_frames = max(5, int(motion_state.get("export_frames", native_frames.shape[0])))
            native_frames = native_frames[trim_frames:trim_frames + export_frames, ...]
            if isinstance(native_audio, dict) and torch.is_tensor(native_audio.get("waveform")):
                sample_rate = max(1, int(native_audio.get("sample_rate", 32000)))
                audio_start = int(round(trim_frames * sample_rate / H3_FPS))
                audio_length = int(round(export_frames * sample_rate / H3_FPS))
                native_audio = dict(native_audio)
                native_audio["waveform"] = native_audio["waveform"][..., audio_start:audio_start + audio_length]
            if int(native_frames.shape[0]) != export_frames:
                raise RuntimeError("MiniMax H3 temporal context produced an incomplete visible export")
        bridge_last_frame = native_frames[-1:].detach().clone()

        report = (
            f"Atomic H3 generation | chunk={int(chunk_index) + 1}/{len(shotplan['chunks'])} | "
            f"task={chunk.get('task_mode')} | seed={actual_seed} ({seed_contract}) | {steps} steps | "
            f"{sampler_report}+{scheduler} | turbo={turbo_report} | acceleration={acceleration_report} | "
            f"controls={sampling_source} | shifts={shift_video:.2f}/{shift_audio:.2f} | denoise={denoise:.2f} | "
            f"pre_sample_cleanup={conditioning_cleanup} | "
            f"pre_decode_cleanup={cleanup_report} | native_last_frame=captured | "
            f"motion_context={'on' if isinstance(motion_state, dict) and motion_state.get('active') else 'off'}"
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


def _materialize_conditioning_on_cpu(value):
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: _materialize_conditioning_on_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_materialize_conditioning_on_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_materialize_conditioning_on_cpu(item) for item in value)
    return value


def _release_ltx_text_stage() -> str:
    import comfy.model_management as model_management

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_management.unload_all_models()
    try:
        model_management.cleanup_models()
    except Exception:
        pass
    model_management.soft_empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    if os.name == "nt":
        try:
            import ctypes

            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.psapi.EmptyWorkingSet(handle)
            return "unload_all_models + cleanup_models + CUDA cache + Windows working-set trim"
        except Exception as exc:
            LOG.warning("LTX text-stage working-set trim warning: %s", exc)
    return "unload_all_models + cleanup_models + CUDA cache"


class IAMCCS_MiniMaxH3LTXConditioningStageV3:
    """Materialize LTX text conditioning, then destroy Gemma and projection."""

    @classmethod
    def INPUT_TYPES(cls):
        input_spec = IAMCCS_MiniMaxH3SequentialLTXLoaderV2._input_spec
        return {
            "required": {
                "trigger_frames": ("IMAGE",),
                "positive_text": ("STRING", {"default": "high quality 4k", "multiline": True}),
                "negative_text": (
                    "STRING",
                    {
                        "default": "pc game, console game, video game, cartoon, childish, ugly",
                        "multiline": True,
                    },
                ),
                "text_encoder_name": input_spec(
                    "DualCLIPLoader", "clip_name1", "gemma_3_12B_it_fp8_e4m3fn.safetensors"
                ),
                "text_projection_name": input_spec(
                    "DualCLIPLoader", "clip_name2", "ltx-2.3_text_projection_bf16.safetensors"
                ),
                "text_encoder_device": (["default", "cpu"], {"default": "default"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "negative", "report")
    FUNCTION = "encode_and_release"
    CATEGORY = CATEGORY

    def encode_and_release(
        self,
        trigger_frames,
        positive_text,
        negative_text,
        text_encoder_name,
        text_projection_name,
        text_encoder_device="default",
    ):
        if not torch.is_tensor(trigger_frames) or trigger_frames.ndim != 4 or trigger_frames.shape[0] < 1:
            raise ValueError("LTX conditioning stage requires decoded native H3 frames")

        _release_ltx_text_stage()
        LOG.info(
            "LTX phase A start | native_frames=%d | Gemma/projection only | device=%s",
            int(trigger_frames.shape[0]),
            text_encoder_device,
        )
        clip = _node_class("DualCLIPLoader")().load_clip(
            text_encoder_name,
            text_projection_name,
            "ltxv",
            text_encoder_device,
        )[0]
        encoder = _node_class("CLIPTextEncode")()
        try:
            positive = encoder.encode(clip=clip, text=str(positive_text or ""))[0]
            negative = encoder.encode(clip=clip, text=str(negative_text or ""))[0]
            positive = _materialize_conditioning_on_cpu(positive)
            negative = _materialize_conditioning_on_cpu(negative)
        finally:
            del encoder, clip
            cleanup_report = _release_ltx_text_stage()

        report = (
            f"LTX phase A complete | conditioning=materialized on CPU | "
            f"Gemma/projection destroyed=yes | {cleanup_report}"
        )
        LOG.info(report)
        return positive, negative, report


class IAMCCS_MiniMaxH3LTXDenoiseStackLoaderV3:
    """Load LTXAV, VAEs and latent upsampler after Phase A has released text models."""

    @classmethod
    def INPUT_TYPES(cls):
        input_spec = IAMCCS_MiniMaxH3SequentialLTXLoaderV2._input_spec
        upscale_models = folder_paths.get_filename_list("latent_upscale_models")
        upscale_meta = {}
        preferred_upscaler = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
        if preferred_upscaler in upscale_models:
            upscale_meta["default"] = preferred_upscaler
        upscale_spec = (upscale_models, upscale_meta) if upscale_meta else (upscale_models,)
        return {
            "required": {
                "conditioning_ready": ("CONDITIONING",),
                "trigger_frames": ("IMAGE",),
                "unet_name": input_spec(
                    "UnetLoaderGGUFAdvanced", "unet_name", "ltx-2.3-22b-dev-Q4_K_S.gguf"
                ),
                "video_vae_name": input_spec(
                    "VAELoaderKJ", "vae_name", "ltx-2.3-22b-dev_video_vae.safetensors"
                ),
                "audio_vae_name": input_spec(
                    "VAELoaderKJ", "vae_name", "ltx-2.3-22b-dev_audio_vae.safetensors"
                ),
                "upscale_model_name": upscale_spec,
            }
        }

    RETURN_TYPES = ("MODEL", "VAE", "VAE", "LATENT_UPSCALE_MODEL", "STRING")
    RETURN_NAMES = ("model", "video_vae", "audio_vae", "upscale_model", "report")
    FUNCTION = "load_after_conditioning"
    CATEGORY = CATEGORY

    def load_after_conditioning(
        self,
        conditioning_ready,
        trigger_frames,
        unet_name,
        video_vae_name,
        audio_vae_name,
        upscale_model_name,
        video_vae_device="main_device",
        video_vae_dtype="bf16",
        audio_vae_device="cpu",
        audio_vae_dtype="bf16",
    ):
        if not isinstance(conditioning_ready, list) or not conditioning_ready:
            raise ValueError("LTX denoise stage requires materialized conditioning from Phase A")
        if not torch.is_tensor(trigger_frames) or trigger_frames.ndim != 4 or trigger_frames.shape[0] < 1:
            raise ValueError("LTX denoise stage requires decoded native H3 frames")

        cleanup_report = _release_ltx_text_stage()
        LOG.info("LTX phase B start | Gemma/projection absent | loading denoise stack")
        model = _node_class("UnetLoaderGGUFAdvanced")().load_unet(
            unet_name,
            dequant_dtype="default",
            patch_dtype="default",
            patch_on_device=False,
        )[0]
        vae_loader = _node_class("VAELoaderKJ")()
        video_vae = vae_loader.load_vae(video_vae_name, video_vae_device, video_vae_dtype)[0]
        audio_vae = vae_loader.load_vae(audio_vae_name, audio_vae_device, audio_vae_dtype)[0]
        upscale_model = _node_class("LatentUpscaleModelLoader").execute(upscale_model_name)[0]
        report = (
            f"LTX phase B ready | text models absent=yes | unet={unet_name} | "
            f"video_vae={video_vae_name} | audio_vae={audio_vae_name} | "
            f"upscaler={upscale_model_name} | pre_load_cleanup={cleanup_report}"
        )
        LOG.info(report)
        return model, video_vae, audio_vae, upscale_model, report


class IAMCCS_MiniMaxH3OptionalLTXDetailerLoRA:
    """Apply one user-selected LTX finishing LoRA without hardcoded filenames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": False}),
                "lora_name": ("STRING", {"default": ""}),
                "strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "apply"
    CATEGORY = CATEGORY

    def apply(self, model, enabled=False, lora_name="", strength=0.6):
        name = str(lora_name or "").strip()
        strength = min(2.0, max(0.0, float(strength or 0.0)))
        if not bool(enabled) or not name or strength == 0.0:
            return model, "LTX detailer LoRA off"
        try:
            path = folder_paths.get_full_path("loras", name)
        except Exception:
            path = None
        if not path:
            LOG.warning("Optional LTX detailer LoRA is unavailable: %s", name)
            return model, f"LTX detailer unavailable ({name}); base LTX path retained"

        import nodes as comfy_nodes

        patched = comfy_nodes.LoraLoaderModelOnly().load_lora_model_only(
            model=model,
            lora_name=name,
            strength_model=strength,
        )[0]
        compatibility_note = ""
        if "ic-lora-detailer" in name.lower():
            compatibility_note = (
                " | compatibility load only: the official IC Detailer reaches its full effect "
                "with the LTX IC conditioning/guiding-latent workflow"
            )
            LOG.warning(
                "LTX IC Detailer %s was loaded as a model-only finishing LoRA; "
                "use the dedicated IC-guided LTX pipeline for its full behavior",
                name,
            )
        return patched, f"LTX detailer LoRA {name}@{strength:.2f}{compatibility_note}"


class IAMCCS_MiniMaxH3RTX4KPost:
    """Optional final RTX VSR pass, requested lazily by the delivery router."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "cine_linx": (SUPERNODE_LINX_TYPE,),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images_4k", "report")
    FUNCTION = "upscale"
    CATEGORY = CATEGORY

    def upscale(self, images, cine_linx):
        if not torch.is_tensor(images) or images.ndim != 4 or images.shape[0] < 1:
            raise ValueError("RTX 4K post expects an IMAGE frame batch")
        shotplan = _resolve_shotplan(cine_linx)
        settings = shotplan.get("upscale_settings") if isinstance(shotplan.get("upscale_settings"), dict) else {}
        target_width = max(256, int(settings.get("target_width", 3840) or 3840))
        target_height = max(256, int(settings.get("target_height", 2160) or 2160))
        if not bool(settings.get("ltx_4k_enabled", False)) or not _is_true_4k_delivery(target_width, target_height):
            return images, "RTX VSR 4K off"

        quality = str(settings.get("ltx_4k_quality", "ULTRA") or "ULTRA").upper()
        if quality not in {"ULTRA", "HIGH", "MEDIUM", "LOW"}:
            quality = "ULTRA"
        source_height = int(images.shape[1])
        source_width = int(images.shape[2])
        scale = max(target_width / max(1, source_width), target_height / max(1, source_height))
        if not 1.0 <= scale <= 4.0:
            raise ValueError(
                f"RTX VSR scale {scale:.3f} is outside the installed node's 1x-4x range "
                f"({source_width}x{source_height} -> {target_width}x{target_height})"
            )

        # NVIDIA Video Effects accepts RGB float32 frames only.  Never convert
        # the full video at once: a 241-frame 1080p input plus its 4K float32
        # output can exceed 28 GiB before LTX caches are counted.  Feed small
        # batches to NvVFX and retain the completed 4K video as CPU float16.
        source_device = str(images.device)
        source_dtype = str(images.dtype)
        if int(images.shape[-1]) < 3:
            raise ValueError(
                f"RTX 4K post expects at least three RGB channels, got shape {tuple(images.shape)}"
            )

        import comfy.model_management as model_management

        model_management.unload_all_models()
        try:
            model_management.cleanup_models()
        except Exception:
            pass
        model_management.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        source = images[..., :3].detach().to(device="cpu")
        frame_count = int(source.shape[0])
        chunk_size = min(8, frame_count)
        upscaled = torch.empty(
            (frame_count, target_height, target_width, 3),
            device="cpu",
            dtype=torch.float16,
        )
        LOG.info(
            "MiniMax H3 RTX VSR chunked start | %s/%s -> cpu/float16 4K | frames=%d | chunk=%d",
            source_device,
            source_dtype,
            frame_count,
            chunk_size,
        )
        rtx_cls = _node_class("RTXVideoSuperResolution")
        progress = comfy.utils.ProgressBar(frame_count)
        for start in range(0, frame_count, chunk_size):
            end = min(frame_count, start + chunk_size)
            rtx_input = source[start:end].to(dtype=torch.float32).contiguous()
            rtx_input = torch.nan_to_num(
                rtx_input,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp_(0.0, 1.0)
            chunk_output = rtx_cls.execute(
                images=rtx_input,
                scale=float(scale),
                quality=quality,
            )[0]
            if int(chunk_output.shape[2]) != target_width or int(chunk_output.shape[1]) != target_height:
                chunk_output = F.interpolate(
                    chunk_output.permute(0, 3, 1, 2),
                    size=(target_height, target_width),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                ).permute(0, 2, 3, 1).clamp(0.0, 1.0)
            upscaled[start:end].copy_(chunk_output.to(device="cpu", dtype=torch.float16))
            del rtx_input, chunk_output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            progress.update_absolute(end, frame_count)
            LOG.info("MiniMax H3 RTX VSR progress | %d/%d frames", end, frame_count)
        del source
        gc.collect()
        return (
            upscaled,
            f"RTX VSR {quality} | {source_width}x{source_height} -> {target_width}x{target_height} | "
            f"scale={scale:.3f} | chunk={chunk_size} | output=cpu/float16",
        )


class IAMCCS_MiniMaxH3PostUpscaleControlV2:
    """Resolve the selected post-upscale branch from the Shotboard CineLinX.

    The node deliberately does not load LTX or Wan itself.  It provides the
    chunk-aware values used by the real workflow branches so ComfyUI's lazy
    delivery router can avoid loading either model while upscale is disabled.
    """

    @staticmethod
    def _ltx_looper_available():
        """Treat temporal looping as an optional LTX capability, never a route prerequisite."""
        try:
            from nodes import NODE_CLASS_MAPPINGS
            return "LTXVLoopingSampler" in NODE_CLASS_MAPPINGS
        except Exception:
            return False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "native_frames": ("IMAGE",),
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
            },
            "optional": {
                # True preserves every existing per-chunk workflow.  The new
                # master-first graph connects the explicit readiness output.
                "master_ready": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = (
        "IMAGE", "STRING", "INT", "INT", "FLOAT", "INT", "BOOLEAN", "FLOAT", "STRING", "STRING",
        "BOOLEAN", "STRING", "FLOAT", "BOOLEAN", "STRING", "BOOLEAN",
        "INT", "INT", "INT", "INT", "INT",
        "INT", "INT", "FLOAT", "FLOAT", "FLOAT", "INT", "INT", "INT",
        "BOOLEAN",
    )
    RETURN_NAMES = (
        "native_frames",
        "upscale_prompt",
        "stage_target_width",
        "stage_target_height",
        "duration_seconds",
        "upscale_seed",
        "sage_enabled",
        "wan_denoise",
        "selected_mode",
        "report",
        "ltx_detailer_enabled",
        "ltx_detailer_lora_name",
        "ltx_detailer_strength",
        "ltx_4k_enabled",
        "ltx_4k_quality",
        "ltx_seam_safe",
        "ltx_encode_temporal_size",
        "ltx_encode_temporal_overlap",
        "ltx_decode_temporal_size",
        "ltx_decode_temporal_overlap",
        "ltx_decode_spatial_overlap",
        "ltx_looper_temporal_tile_size",
        "ltx_looper_temporal_overlap",
        "ltx_looper_guiding_strength",
        "ltx_looper_overlap_strength",
        "ltx_looper_cond_image_strength",
        "ltx_looper_horizontal_tiles",
        "ltx_looper_vertical_tiles",
        "ltx_looper_spatial_overlap",
        "ltx_looper_available",
    )
    FUNCTION = "prepare"
    CATEGORY = CATEGORY

    def prepare(self, cine_linx, native_frames, segment_index=0, master_ready=True):
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

        enabled = bool(shotplan.get("upscale_enabled", False)) and bool(master_ready)
        selected_mode = str(shotplan.get("upscale_mode", "off") or "off").lower() if enabled else "off"
        if selected_mode not in {"off", "ltx23", "wan22_5b"}:
            raise ValueError(f"Unknown MiniMax H3 post-upscale mode: {selected_mode}")

        native_width = int(shotplan.get("width", int(native_frames.shape[2])) or int(native_frames.shape[2]))
        native_height = int(shotplan.get("height", int(native_frames.shape[1])) or int(native_frames.shape[1]))
        delivery_target_width = max(256, int(settings.get("target_width", native_width * 2) or native_width * 2))
        delivery_target_height = max(256, int(settings.get("target_height", native_height * 2) or native_height * 2))
        ltx_4k_requested = bool(settings.get("ltx_4k_enabled", False)) and selected_mode == "ltx23"
        if selected_mode == "ltx23":
            ltx_4k_enabled, stage_target_width, stage_target_height, ltx_4k_resolution_report = _resolve_ltx_4k_stage(
                native_width,
                native_height,
                delivery_target_width,
                delivery_target_height,
                ltx_4k_requested,
            )
        else:
            ltx_4k_enabled = False
            stage_target_width = delivery_target_width
            stage_target_height = delivery_target_height
            ltx_4k_resolution_report = "LTX path not selected"
        prompt_source = (
            settings.get("prompt")
            or (shotplan.get("global_prompt") if bool(master_ready) else chunk.get("prompt"))
            or shotplan.get("global_prompt")
            or "high quality cinematic video"
        )
        prompt = str(prompt_source).strip()
        duration_seconds = (
            max(1, int(native_frames.shape[0])) / H3_FPS
            if bool(master_ready)
            else float(chunk.get("duration_seconds") or (max(1, int(native_frames.shape[0])) / H3_FPS))
        )
        sampling = shotplan.get("sampling") if isinstance(shotplan.get("sampling"), dict) else {}
        base_seed = int(sampling.get("seed", 0) or 0)
        seed_stride = int(sampling.get("seed_stride", 1) or 1)
        seed_offset = int(settings.get("seed_offset", 10000) or 0)
        upscale_seed = (
            base_seed + (0 if bool(master_ready) else index * seed_stride) + seed_offset
        ) & 0xFFFFFFFFFFFFFFFF
        sage_enabled = bool(settings.get("sage", True))
        wan_denoise = min(1.0, max(0.0, float(settings.get("wan_denoise", 0.2) or 0.0)))
        ltx_detailer_enabled = bool(settings.get("ltx_detailer_enabled", False))
        ltx_detailer_lora_name = str(settings.get("ltx_detailer_lora_name", "") or "").strip()
        ltx_detailer_strength = min(2.0, max(0.0, float(settings.get("ltx_detailer_strength", 0.6) or 0.0)))
        ltx_4k_quality = str(settings.get("ltx_4k_quality", "ULTRA") or "ULTRA").upper()
        if ltx_4k_quality not in {"ULTRA", "HIGH", "MEDIUM", "LOW"}:
            ltx_4k_quality = "ULTRA"
        ltx_seam_safe = bool(settings.get("ltx_seam_safe", True))
        encode_temporal_size = int(settings.get("ltx_vae_encode_temporal_size", 500 if ltx_seam_safe else 64))
        encode_temporal_overlap = int(settings.get("ltx_vae_encode_temporal_overlap", 4 if ltx_seam_safe else 8))
        decode_temporal_size = int(settings.get("ltx_vae_decode_temporal_size", 64 if ltx_seam_safe else 16))
        decode_temporal_overlap = int(settings.get("ltx_vae_decode_temporal_overlap", 4 if ltx_seam_safe else 1))
        decode_spatial_overlap = int(settings.get("ltx_vae_decode_spatial_overlap", 4 if ltx_seam_safe else 1))
        looper = settings.get("ltx_looper") if isinstance(settings.get("ltx_looper"), dict) else {}
        looper_tile_size = max(24, min(1000, int(looper.get("temporal_tile_size", 80) or 80)))
        looper_overlap = max(16, min(80, int(looper.get("temporal_overlap", 24) or 24)))
        looper_overlap = min(looper_overlap, max(16, looper_tile_size - 8))
        looper_guiding_strength = min(1.0, max(0.0, float(looper.get("guiding_strength", 1.0) or 0.0)))
        looper_overlap_strength = min(1.0, max(0.0, float(looper.get("overlap_strength", 0.5) or 0.0)))
        looper_cond_image_strength = min(1.0, max(0.0, float(looper.get("cond_image_strength", 1.0) or 0.0)))
        looper_horizontal_tiles = max(1, min(6, int(looper.get("horizontal_tiles", 1) or 1)))
        looper_vertical_tiles = max(1, min(6, int(looper.get("vertical_tiles", 1) or 1)))
        looper_spatial_overlap = max(1, min(8, int(looper.get("spatial_overlap", 1) or 1)))
        looper_available = selected_mode == "ltx23" and self._ltx_looper_available()
        report = (
            f"H3 post-upscale control | selected={selected_mode} | "
            f"source={'native_full_master' if bool(master_ready) else f'chunk_{index + 1}'} | "
            f"native={native_width}x{native_height} -> LTX stage={stage_target_width}x{stage_target_height} "
            f"-> delivery={delivery_target_width}x{delivery_target_height} | "
            f"duration={duration_seconds:.3f}s | seed={upscale_seed} | sage={'on' if sage_enabled else 'off'} | "
            f"detailer={'on' if ltx_detailer_enabled else 'off'}:{ltx_detailer_lora_name or 'none'}@{ltx_detailer_strength:.2f} | "
            f"seam_safe={'on' if ltx_seam_safe else 'off'} | rtx_4k={'on' if ltx_4k_enabled else 'off'}:{ltx_4k_quality} "
            f"| looper={'available' if looper_available else 'not_available_standard_ltx'} "
            f"{looper_tile_size}/{looper_overlap}@{looper_overlap_strength:.2f} "
            f"({ltx_4k_resolution_report}) | "
            f"wan_denoise={wan_denoise:.2f} | lazy=yes"
        )
        return (
            native_frames,
            prompt,
            stage_target_width,
            stage_target_height,
            duration_seconds,
            upscale_seed,
            sage_enabled,
            wan_denoise,
            selected_mode,
            report,
            ltx_detailer_enabled,
            ltx_detailer_lora_name,
            ltx_detailer_strength,
            ltx_4k_enabled,
            ltx_4k_quality,
            ltx_seam_safe,
            encode_temporal_size,
            encode_temporal_overlap,
            decode_temporal_size,
            decode_temporal_overlap,
            decode_spatial_overlap,
            looper_tile_size,
            looper_overlap,
            looper_guiding_strength,
            looper_overlap_strength,
            looper_cond_image_strength,
            looper_horizontal_tiles,
            looper_vertical_tiles,
            looper_spatial_overlap,
            looper_available,
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
                "master_ready": ("BOOLEAN", {"default": True}),
                "ltx23_upscaled_frames": ("IMAGE", {"lazy": True}),
                "ltx23_4k_frames": ("IMAGE", {"lazy": True}),
                "wan22_upscaled_frames": ("IMAGE", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "IMAGE", "INT", "STRING", "STRING")
    RETURN_NAMES = ("delivery_frames", "audio", "bridge_last_frame", "delivery_fps", "upscale_applied", "report")
    FUNCTION = "route"
    CATEGORY = CATEGORY

    @staticmethod
    def _selected_upscale_input(cine_linx, master_ready=True, native_frames=None):
        if not bool(master_ready):
            return None
        shotplan = _resolve_shotplan(cine_linx)
        if not bool(shotplan.get("upscale_enabled", False)):
            return None
        mode = str(shotplan.get("upscale_mode", "off") or "off").lower()
        if mode == "ltx23":
            settings = shotplan.get("upscale_settings") if isinstance(shotplan.get("upscale_settings"), dict) else {}
            target_width = max(256, int(settings.get("target_width", 0) or 0))
            target_height = max(256, int(settings.get("target_height", 0) or 0))
            native_width = int(shotplan.get("width", 0) or 0)
            native_height = int(shotplan.get("height", 0) or 0)
            if torch.is_tensor(native_frames) and native_frames.ndim == 4:
                native_width = int(native_frames.shape[2])
                native_height = int(native_frames.shape[1])
            native_width = max(1, native_width)
            native_height = max(1, native_height)
            effective_4k, _, _, _ = _resolve_ltx_4k_stage(
                native_width,
                native_height,
                target_width,
                target_height,
                bool(settings.get("ltx_4k_enabled", False)),
            )
            if effective_4k:
                return "ltx23_4k_frames"
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
        master_ready=True,
        ltx23_upscaled_frames=None,
        ltx23_4k_frames=None,
        wan22_upscaled_frames=None,
        **kwargs,
    ):
        selected = self._selected_upscale_input(cine_linx, master_ready, native_frames)
        if selected == "ltx23_upscaled_frames" and ltx23_upscaled_frames is None:
            return [selected]
        if selected == "ltx23_4k_frames" and ltx23_4k_frames is None:
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
        master_ready=True,
        ltx23_upscaled_frames=None,
        ltx23_4k_frames=None,
        wan22_upscaled_frames=None,
    ):
        shotplan = _resolve_shotplan(cine_linx)
        selected = self._selected_upscale_input(shotplan, master_ready, native_frames)
        settings = shotplan.get("upscale_settings") if isinstance(shotplan.get("upscale_settings"), dict) else {}
        delivery_target_width = max(256, int(settings.get("target_width", int(native_frames.shape[2])) or int(native_frames.shape[2])))
        delivery_target_height = max(256, int(settings.get("target_height", int(native_frames.shape[1])) or int(native_frames.shape[1])))
        exact_delivery_report = "not required"
        if selected == "ltx23_upscaled_frames":
            if not torch.is_tensor(ltx23_upscaled_frames):
                raise ValueError("Shotboard enabled LTX 2.3 upscale, but the lazy LTX output is not connected")
            delivery, exact_delivery_report = _resize_frames_exact_cpu(
                ltx23_upscaled_frames,
                delivery_target_width,
                delivery_target_height,
            )
            upscale_report = "LTX 2.3"
        elif selected == "ltx23_4k_frames":
            if not torch.is_tensor(ltx23_4k_frames):
                raise ValueError("Shotboard enabled RTX VSR 4K after LTX, but the lazy 4K output is not connected")
            delivery, exact_delivery_report = _resize_frames_exact_cpu(
                ltx23_4k_frames,
                delivery_target_width,
                delivery_target_height,
            )
            upscale_report = "LTX 2.3 + RTX VSR 4K"
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
            f"bridge=last native H3 frame before upscale/RIFE | exact={exact_delivery_report} | "
            f"lazy_upscale=yes | {post_upscale_cleanup}"
        )
        return delivery, native_audio, bridge_last_frame, int(delivery_fps), upscale_report, report


NODE_CLASS_MAPPINGS = {
    "IAMCCS_MiniMaxH3AtomicModelRouter": IAMCCS_MiniMaxH3AtomicModelRouter,
    "IAMCCS_MiniMaxH3AtomicConditioningBackend": IAMCCS_MiniMaxH3AtomicConditioningBackend,
    "IAMCCS_MiniMaxH3DirectorFLFParityModelRouter": IAMCCS_MiniMaxH3DirectorFLFParityModelRouter,
    "IAMCCS_MiniMaxH3DirectorFLFParityConditioning": IAMCCS_MiniMaxH3DirectorFLFParityConditioning,
    "IAMCCS_MiniMaxH3GenerationBackendV2": IAMCCS_MiniMaxH3GenerationBackendV2,
    "IAMCCS_MiniMaxH3SequentialLTXLoaderV2": IAMCCS_MiniMaxH3SequentialLTXLoaderV2,
    "IAMCCS_MiniMaxH3LTXConditioningStageV3": IAMCCS_MiniMaxH3LTXConditioningStageV3,
    "IAMCCS_MiniMaxH3LTXDenoiseStackLoaderV3": IAMCCS_MiniMaxH3LTXDenoiseStackLoaderV3,
    "IAMCCS_MiniMaxH3OptionalLTXDetailerLoRA": IAMCCS_MiniMaxH3OptionalLTXDetailerLoRA,
    "IAMCCS_MiniMaxH3RTX4KPost": IAMCCS_MiniMaxH3RTX4KPost,
    "IAMCCS_MiniMaxH3PostUpscaleControlV2": IAMCCS_MiniMaxH3PostUpscaleControlV2,
    "IAMCCS_MiniMaxH3DeliveryRouterV2": IAMCCS_MiniMaxH3DeliveryRouterV2,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_MiniMaxH3AtomicModelRouter": "MiniMax H3 Atomic FL2VA / REF2VA Model Router",
    "IAMCCS_MiniMaxH3AtomicConditioningBackend": "MiniMax H3 Atomic Shotboard Conditioning",
    "IAMCCS_MiniMaxH3DirectorFLFParityModelRouter": "MiniMax H3 Director FLF Parity Router",
    "IAMCCS_MiniMaxH3DirectorFLFParityConditioning": "MiniMax H3 Director FLF Parity Conditioning",
    "IAMCCS_MiniMaxH3GenerationBackendV2": "MiniMax H3 Generation V2 (Acceleration + Clean Decode)",
    "IAMCCS_MiniMaxH3SequentialLTXLoaderV2": "MiniMax H3 -> LTX Sequential Loader V2",
    "IAMCCS_MiniMaxH3LTXConditioningStageV3": "MiniMax H3 -> LTX Phase A - Gemma Conditioning + Destroy",
    "IAMCCS_MiniMaxH3LTXDenoiseStackLoaderV3": "MiniMax H3 -> LTX Phase B - Denoise Stack After Barrier",
    "IAMCCS_MiniMaxH3OptionalLTXDetailerLoRA": "MiniMax H3 Optional LTX Detailer LoRA",
    "IAMCCS_MiniMaxH3RTX4KPost": "MiniMax H3 RTX VSR 4K Post",
    "IAMCCS_MiniMaxH3PostUpscaleControlV2": "MiniMax H3 Post-Upscale Control V2 (LTX / Wan)",
    "IAMCCS_MiniMaxH3DeliveryRouterV2": "MiniMax H3 Delivery V2 (Lazy Upscale + RIFE)",
}
