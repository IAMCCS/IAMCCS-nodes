import gc
import json
import os
import platform
import time
import logging
from typing import Any, Callable, Optional

import torch

import comfy.model_management as mm

from .iamccs_flexible_inputs import any_type

try:
    from comfy.patcher_extension import CallbacksMP  # type: ignore
except Exception:  # pragma: no cover
    CallbacksMP = None  # type: ignore


log = logging.getLogger("IAMCCS.HwSupporter")


_SAGE_ATTN_MODES = [
    "disabled",
    "auto",
    "sageattn_qk_int8_pv_fp16_cuda",
    "sageattn_qk_int8_pv_fp16_triton",
    "sageattn_qk_int8_pv_fp8_cuda",
    "sageattn_qk_int8_pv_fp8_cuda++",
    "sageattn3",
    "sageattn3_per_block_mean",
]

_TORCH_COMPILE_MODES = [
    "off",
    "auto",
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
]

_PROFILE_PRESETS = [
    "auto",
    "12GB_VRAM_32GB_RAM",
    "low_vram",
    "balanced",
    "max_speed",
]


def _bytes_to_gb(num_bytes: int | float | None) -> float | None:
    if num_bytes is None:
        return None
    try:
        return float(num_bytes) / (1024.0**3)
    except Exception:
        return None


def _safe_get_total_ram_gb() -> float | None:
    try:
        import psutil  # type: ignore

        return _bytes_to_gb(psutil.virtual_memory().total)
    except Exception:
        return None


def _safe_get_cuda_device_info() -> dict:
    info: dict = {
        "torch_cuda_available": bool(torch.cuda.is_available()),
    }
    if not torch.cuda.is_available():
        return info

    try:
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.update(
            {
                "cuda_device_index": int(idx),
                "cuda_device_name": getattr(props, "name", None),
                "cuda_total_vram_gb": _bytes_to_gb(getattr(props, "total_memory", None)),
                "cuda_cc_major": int(getattr(props, "major", -1)),
                "cuda_cc_minor": int(getattr(props, "minor", -1)),
            }
        )
    except Exception as e:
        info["cuda_device_info_error"] = repr(e)

    try:
        # mem_get_info returns free,total
        free_b, total_b = torch.cuda.mem_get_info()
        info["cuda_free_vram_gb"] = _bytes_to_gb(free_b)
        info["cuda_total_vram_gb_runtime"] = _bytes_to_gb(total_b)
    except Exception:
        pass

    return info


def _set_reserved_vram(reserved_gb: float) -> int:
    reserved_bytes = int(max(0.0, float(reserved_gb)) * (1024**3))
    mm.EXTRA_RESERVED_VRAM = reserved_bytes
    return reserved_bytes


def _try_get_nvml_total_used_gb() -> tuple[float | None, float | None, str | None]:
    """Best-effort: return (total_gb, used_gb, warning)."""
    try:
        import pynvml  # type: ignore

        try:
            pynvml.nvmlInit()
        except Exception:
            # Already initialized or init failed; continue anyway.
            pass

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_gb = float(mem.total) / (1024.0**3)
        used_gb = float(mem.used) / (1024.0**3)
        return total_gb, used_gb, None
    except Exception as e:
        return None, None, f"pynvml not available/failed: {e!r}"


def _try_get_torch_total_used_gb() -> tuple[float | None, float | None, str | None]:
    """Fallback: approximate used VRAM from torch.cuda.mem_get_info()."""
    if not torch.cuda.is_available():
        return None, None, "CUDA not available"
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        total_gb = float(total_b) / (1024.0**3)
        used_gb = max(0.0, float(total_b - free_b) / (1024.0**3))
        return total_gb, used_gb, None
    except Exception as e:
        return None, None, f"torch.cuda.mem_get_info failed: {e!r}"


def _auto_reserved_vram_gb(headroom_gb: float, auto_max_reserved_gb: float = 0.0) -> tuple[float | None, list[str]]:
    """ReservedVRAMSetter-style: reserved = used + headroom, optional max cap."""
    warnings: list[str] = []
    headroom_gb = float(headroom_gb)
    auto_max_reserved_gb = float(auto_max_reserved_gb)

    total_gb, used_gb, w = _try_get_nvml_total_used_gb()
    if w:
        warnings.append(w)
    if total_gb is None or used_gb is None:
        total_gb, used_gb, w2 = _try_get_torch_total_used_gb()
        if w2:
            warnings.append(w2)

    if total_gb is None or used_gb is None:
        return None, warnings

    reserved = max(0.0, used_gb + headroom_gb)
    if auto_max_reserved_gb > 0.0:
        reserved = min(reserved, auto_max_reserved_gb)

    # Never exceed total VRAM (can happen due to rounding or transient reporting).
    try:
        if total_gb is not None:
            reserved = min(reserved, max(0.0, float(total_gb) - 0.05))
    except Exception:
        pass
    return float(round(reserved, 2)), warnings


def _gpu_cleanup(unload_all: bool = True, soft_empty_cache: bool = True) -> None:
    gc.collect()
    if unload_all:
        try:
            mm.unload_all_models()
        except Exception:
            pass

    if soft_empty_cache:
        try:
            mm.soft_empty_cache()
        except Exception:
            pass

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _apply_fp16_accumulation(setting: str) -> tuple[bool | None, str | None]:
    """Returns (value_set, warning)."""
    if not hasattr(torch.backends, "cuda"):
        return None, "torch.backends.cuda not available"
    if not hasattr(torch.backends.cuda, "matmul"):
        return None, "torch.backends.cuda.matmul not available"
    if not hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
        return None, "allow_fp16_accumulation not available (needs newer PyTorch)"

    if setting == "on":
        torch.backends.cuda.matmul.allow_fp16_accumulation = True
        return True, None
    if setting == "off":
        torch.backends.cuda.matmul.allow_fp16_accumulation = False
        return False, None

    # auto
    try:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
            return True, None
    except Exception as e:
        return None, f"failed to set allow_fp16_accumulation: {e!r}"

    return None, "CUDA not available"


def _apply_tf32(setting: str) -> tuple[bool | None, str | None]:
    if setting == "on":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            return True, None
        except Exception as e:
            return None, f"failed to enable TF32: {e!r}"

    if setting == "off":
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            return False, None
        except Exception as e:
            return None, f"failed to disable TF32: {e!r}"

    # auto
    try:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            return True, None
    except Exception as e:
        return None, f"failed to set TF32 in auto: {e!r}"

    return None, "CUDA not available"


def _try_make_sage_attention_override(mode: str, allow_compile: bool) -> tuple[Optional[Callable[..., Any]], Optional[str]]:
    if mode == "disabled":
        return None, None

    # KJNodes-compatible implementation: keep behavior/shape handling 1:1.
    try:
        from sageattention import sageattn
    except Exception as e:
        return None, f"sageattention not importable: {e!r}"

    try:
        from comfy.ldm.modules.attention import wrap_attn
    except Exception as e:
        return None, f"ComfyUI attention wrapper import failed: {e!r}"

    try:
        sage_func: Callable[..., Any]

        if mode == "auto":
            def _sage_func_auto(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                return sageattn(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
            sage_func = _sage_func_auto

        elif mode == "sageattn_qk_int8_pv_fp16_cuda":
            from sageattention import sageattn_qk_int8_pv_fp16_cuda

            def _sage_func_int8_fp16_cuda(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                return sageattn_qk_int8_pv_fp16_cuda(
                    q,
                    k,
                    v,
                    is_causal=is_causal,
                    attn_mask=attn_mask,
                    pv_accum_dtype="fp32",
                    tensor_layout=tensor_layout,
                )
            sage_func = _sage_func_int8_fp16_cuda

        elif mode == "sageattn_qk_int8_pv_fp16_triton":
            from sageattention import sageattn_qk_int8_pv_fp16_triton

            def _sage_func_int8_fp16_triton(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                return sageattn_qk_int8_pv_fp16_triton(
                    q,
                    k,
                    v,
                    is_causal=is_causal,
                    attn_mask=attn_mask,
                    tensor_layout=tensor_layout,
                )
            sage_func = _sage_func_int8_fp16_triton

        elif mode == "sageattn_qk_int8_pv_fp8_cuda":
            from sageattention import sageattn_qk_int8_pv_fp8_cuda

            def _sage_func_int8_fp8_cuda(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                return sageattn_qk_int8_pv_fp8_cuda(
                    q,
                    k,
                    v,
                    is_causal=is_causal,
                    attn_mask=attn_mask,
                    pv_accum_dtype="fp32+fp32",
                    tensor_layout=tensor_layout,
                )
            sage_func = _sage_func_int8_fp8_cuda

        elif mode == "sageattn_qk_int8_pv_fp8_cuda++":
            from sageattention import sageattn_qk_int8_pv_fp8_cuda

            def _sage_func_int8_fp8_cuda_pp(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                return sageattn_qk_int8_pv_fp8_cuda(
                    q,
                    k,
                    v,
                    is_causal=is_causal,
                    attn_mask=attn_mask,
                    pv_accum_dtype="fp32+fp16",
                    tensor_layout=tensor_layout,
                )
            sage_func = _sage_func_int8_fp8_cuda_pp

        elif "sageattn3" in mode:
            from sageattn3 import sageattn3_blackwell

            def _sage_func_sageattn3(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD", **kwargs):
                q, k, v = [x.transpose(1, 2) if tensor_layout == "NHD" else x for x in (q, k, v)]
                out = sageattn3_blackwell(q, k, v, is_causal=is_causal, attn_mask=attn_mask, per_block_mean=(mode == "sageattn3_per_block_mean"))
                return out.transpose(1, 2) if tensor_layout == "NHD" else out
            sage_func = _sage_func_sageattn3

        else:
            return None, f"unknown sageattention mode: {mode}"
    except Exception as e:
        return None, f"failed to build sageattention mode '{mode}': {e!r}"

    if not allow_compile:
        try:
            sage_func = torch.compiler.disable()(sage_func)
        except Exception:
            # Older torch may not have torch.compiler; keep running.
            pass

    @wrap_attn
    def attention_sage(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        in_dtype = v.dtype
        if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
            q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
        if skip_reshape:
            b, _, _, dim_head = q.shape
            tensor_layout = "HND"
        else:
            b, _, dim_head = q.shape
            dim_head //= heads
            q, k, v = map(
                lambda t: t.view(b, -1, heads, dim_head),
                (q, k, v),
            )
            tensor_layout = "NHD"
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
        out = sage_func(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout).to(in_dtype)
        if tensor_layout == "HND":
            if not skip_output_reshape:
                out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        else:
            if skip_output_reshape:
                out = out.transpose(1, 2)
            else:
                out = out.reshape(b, -1, heads * dim_head)
        return out

    def attention_override_sage(func, *args, **kwargs):
        return attention_sage.__wrapped__(*args, **kwargs)

    return attention_override_sage, None


def _maybe_torch_compile_model(model, mode: str) -> tuple[bool, str | None]:
    if mode == "off":
        return False, None

    if not hasattr(torch, "compile"):
        return False, "torch.compile not available"

    if not torch.cuda.is_available():
        return False, "CUDA not available (torch.compile in this node is CUDA-oriented)"

    # Heuristic: avoid compile if ComfyUI is in lowvram/offload mode.
    try:
        if hasattr(mm, "should_use_fp16"):
            pass
    except Exception:
        pass

    if mode == "auto":
        mode = "reduce-overhead"

    try:
        inner = getattr(model, "model", None)
        if inner is None:
            return False, "model has no .model attribute"

        dm = getattr(inner, "diffusion_model", None)
        if dm is None:
            return False, "model.model has no diffusion_model"

        inner.diffusion_model = torch.compile(dm, mode=mode)
        return True, None
    except Exception as e:
        return False, f"torch.compile failed: {e!r}"


def _recommend_for_profile(profile: str, cuda_total_vram_gb: float | None) -> dict:
    # Conservative defaults.
    rec = {
        "reserved_vram_gb": 1.0,
        "sage_attention": "auto",
        "allow_sage_compile": False,
        "torch_compile_mode": "off",
        "fp16_accumulation": "auto",
        "tf32": "auto",
        "notes": [],
    }

    if profile == "12GB_VRAM_32GB_RAM":
        # Windows users sometimes hit hard-crashes in Triton/MLIR paths.
        # Keep the preset conservative and stable.
        rec.update(
            {
                "reserved_vram_gb": 1.25,
                "torch_compile_mode": "off",
                "sage_attention": "sageattn_qk_int8_pv_fp16_cuda" if platform.system() == "Windows" else "auto",
                "notes": [
                    "Prefer tiled VAE decode for high-res.",
                    "Avoid torch.compile when using aggressive offload/lowvram.",
                    "On Windows, prefer SageAttention CUDA mode over Triton if you see libtriton crashes.",
                ],
            }
        )
        return rec

    if profile == "low_vram":
        rec.update(
            {
                "reserved_vram_gb": 1.5,
                "sage_attention": "auto",
                "torch_compile_mode": "off",
                "notes": [
                    "Use smaller batch/tiles and consider CPU offload.",
                    "torch.compile often increases peak memory on first run.",
                ],
            }
        )
        return rec

    if profile == "max_speed":
        rec.update(
            {
                "reserved_vram_gb": 0.5,
                "torch_compile_mode": "reduce-overhead",
                "allow_sage_compile": True,
                "notes": [
                    "Speed-focused: first run may be slower/higher VRAM due to compilation.",
                ],
            }
        )
        return rec

    if profile == "balanced":
        rec.update(
            {
                "reserved_vram_gb": 1.0,
                "torch_compile_mode": "off",
            }
        )
        return rec

    # auto
    if cuda_total_vram_gb is not None:
        if cuda_total_vram_gb <= 12.5:
            rec.update(
                {
                    "reserved_vram_gb": 1.25,
                    "torch_compile_mode": "off",
                }
            )
        elif cuda_total_vram_gb <= 16.5:
            rec.update(
                {
                    "reserved_vram_gb": 1.0,
                    "torch_compile_mode": "off",
                }
            )
        else:
            rec.update(
                {
                    "reserved_vram_gb": 0.75,
                    "torch_compile_mode": "reduce-overhead",
                    "allow_sage_compile": True,
                }
            )

    return rec


class IAMCCS_HwSupporter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "profile": (_PROFILE_PRESETS, {"default": "auto"}),
                "apply_reserved_vram": ("BOOLEAN", {"default": True}),
                "reserved_vram_mode": (
                    ["manual", "auto_used_plus"],
                    {
                        "default": "manual",
                        "tooltip": "manual: uses reserved_vram_gb | auto_used_plus: sets EXTRA_RESERVED_VRAM = (used_vram + auto_headroom_gb), like ReservedVRAMSetter",
                    },
                ),
                "reserved_vram_gb": (
                    "FLOAT",
                    {
                        "default": 1.25,
                        "min": 0.0,
                        "max": 24.0,
                        "step": 0.25,
                        "tooltip": "Reserved VRAM in GB (manual mode). If profile=auto and this is 0, the node chooses a conservative value.",
                    },
                ),
                "reserved_vram_auto_headroom_gb": (
                    "FLOAT",
                    {"default": 1.0, "min": -2.0, "max": 24.0, "step": 0.25, "tooltip": "(auto_used_plus) headroom added on top of currently used VRAM"},
                ),
                "reserved_vram_auto_max_gb": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 48.0, "step": 0.25, "tooltip": "(auto_used_plus) max reserved cap, 0 = no cap"},
                ),
                "sage_attention": (_SAGE_ATTN_MODES, {"default": "auto"}),
                "allow_sageattention_torch_compile": ("BOOLEAN", {"default": False}),
                "torch_compile_mode": (
                    _TORCH_COMPILE_MODES,
                    {
                        "default": "off",
                        "tooltip": "auto = try torch.compile with a safe mode (reduce-overhead). On Windows this may still be unstable depending on Torch/driver; if you see crashes, set off.",
                    },
                ),
                "fp16_accumulation": (["auto", "on", "off"], {"default": "auto"}),
                "tf32": (["auto", "on", "off"], {"default": "auto"}),
                "clean_gpu_before": ("BOOLEAN", {"default": False}),
                "console_log": ("BOOLEAN", {"default": True, "tooltip": "Print a short summary (applied + warnings) in the server console."}),
                "include_hardware_report": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If enabled, probes current hardware and embeds an additional recommendations report into report_json (useful for printing/debug).",
                    },
                ),
            },
            "optional": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "report_json")
    FUNCTION = "apply"
    CATEGORY = "IAMCCS/HW"

    def apply(
        self,
        model,
        profile,
        apply_reserved_vram,
        reserved_vram_mode,
        reserved_vram_gb,
        reserved_vram_auto_headroom_gb,
        reserved_vram_auto_max_gb,
        sage_attention,
        allow_sageattention_torch_compile,
        torch_compile_mode,
        fp16_accumulation,
        tf32,
        clean_gpu_before,
        console_log,
        include_hardware_report,
        clip=None,
        vae=None,
    ):
        t0 = time.time()

        if clean_gpu_before:
            _gpu_cleanup(unload_all=True, soft_empty_cache=True)

        cuda_info = _safe_get_cuda_device_info()
        total_ram_gb = _safe_get_total_ram_gb()

        cuda_total = cuda_info.get("cuda_total_vram_gb")
        rec = _recommend_for_profile(profile, cuda_total)

        # Keep the raw user request for backend verification.
        torch_compile_mode_requested = str(torch_compile_mode or "auto")
        sage_attention_requested = str(sage_attention or "auto")
        reserved_vram_mode_requested = str(reserved_vram_mode or "manual")

        # Effective settings
        reserved_vram_mode = str(reserved_vram_mode or "manual")
        reserved_effective_gb = float(reserved_vram_gb)
        if profile == "auto" and reserved_effective_gb <= 0.0:
            reserved_effective_gb = float(rec["reserved_vram_gb"])

        sage_attention_effective = str(sage_attention_requested)
        if sage_attention_effective == "auto":
            sage_attention_effective = str(rec.get("sage_attention", "auto"))

        # torch_compile_mode semantics:
        # - off/default/reduce-overhead/... are explicit requests
        # - auto means "use a safe heuristic" (Windows defaults to off)
        # Profile recommendations are exposed in report_json but do not override user selection.
        compile_mode_effective = str(torch_compile_mode_requested)

        # Early log (similar timing to other patch nodes): print immediately when node starts.
        if console_log:
            try:
                start_msg = (
                    f"[IAMCCS_HwSupporter] start profile={profile} "
                    f"reserved_vram_mode={reserved_vram_mode} reserved_vram_gb={reserved_effective_gb} "
                    f"(auto_headroom={float(reserved_vram_auto_headroom_gb)} auto_max={float(reserved_vram_auto_max_gb)}) "
                    f"sage_attention={sage_attention_requested}->{sage_attention_effective} "
                    f"torch_compile={torch_compile_mode_requested}"
                )
                log.info("%s", start_msg)
                try:
                    print(start_msg, flush=True)
                except Exception:
                    pass
            except Exception:
                pass

        report: dict = {
            "profile": profile,
            "requested": {
                "torch_compile_mode": torch_compile_mode_requested,
                "sage_attention": sage_attention_requested,
                "reserved_vram_mode": reserved_vram_mode_requested,
                "reserved_vram_gb": float(reserved_vram_gb),
                "apply_reserved_vram": bool(apply_reserved_vram),
            },
            "platform": {
                "os": platform.system(),
                "os_release": platform.release(),
                "python": platform.python_version(),
                "torch": getattr(torch, "__version__", None),
            },
            "hardware": {
                **cuda_info,
                "system_ram_gb": total_ram_gb,
            },
            "applied": {},
            "warnings": [],
            "recommendations": rec,
            "vae_tiling_suggestion": None,
            "notes": [
                "Allocator tuning via PYTORCH_CUDA_ALLOC_CONF usually must be set before starting ComfyUI.",
                "If you use --lowvram/--medvram/offload, avoid torch.compile for stability.",
            ],
        }

        if include_hardware_report:
            try:
                from .iamccs_hw_probe import recommend_settings

                report["hardware_probe"] = recommend_settings()
            except Exception as e:
                report["warnings"].append(f"failed to run hardware_probe: {e!r}")

        # Suggest VAE tiled decode parameters (useful for 12GB class GPUs).
        try:
            vram_gb = cuda_info.get("cuda_total_vram_gb")
            compression = 8
            if vae is not None:
                try:
                    if hasattr(vae, "spacial_compression_decode"):
                        compression = int(vae.spacial_compression_decode())
                except Exception:
                    compression = 8

            tile_s, overlap_s = IAMCCS_VAEDecodeTiledSafe._auto_tile_params(vram_gb, compression)
            report["vae_tiling_suggestion"] = {
                "tiling_mode": "auto",
                "tile": True,
                "tile_size": int(tile_s),
                "overlap": int(overlap_s),
                "temporal_size": 24,
                "temporal_overlap": 4,
                "vram_gb": vram_gb,
                "compression": compression,
            }
        except Exception as e:
            report["warnings"].append(f"failed to compute vae_tiling_suggestion: {e!r}")

        # Reserve VRAM
        if apply_reserved_vram:
            try:
                if reserved_vram_mode == "auto_used_plus":
                    auto_reserved, warns = _auto_reserved_vram_gb(
                        headroom_gb=float(reserved_vram_auto_headroom_gb),
                        auto_max_reserved_gb=float(reserved_vram_auto_max_gb),
                    )
                    for w in warns:
                        report["warnings"].append(w)
                    if auto_reserved is None:
                        # Fallback to manual-ish conservative value.
                        reserved_bytes = _set_reserved_vram(reserved_effective_gb)
                        report["warnings"].append("auto_used_plus failed; fell back to reserved_vram_gb")
                        report["applied"]["reserved_vram_gb"] = _bytes_to_gb(reserved_bytes)
                    else:
                        reserved_bytes = _set_reserved_vram(float(auto_reserved))
                        report["applied"]["reserved_vram_gb"] = _bytes_to_gb(reserved_bytes)
                        report["applied"]["reserved_vram_auto_used_plus"] = {
                            "headroom_gb": float(reserved_vram_auto_headroom_gb),
                            "auto_max_reserved_gb": float(reserved_vram_auto_max_gb),
                            "effective_reserved_gb": float(auto_reserved),
                        }
                else:
                    reserved_bytes = _set_reserved_vram(reserved_effective_gb)
                    report["applied"]["reserved_vram_gb"] = _bytes_to_gb(reserved_bytes)

                if console_log:
                    try:
                        rv = report.get("applied", {}).get("reserved_vram_gb")
                        msg = f"[IAMCCS_HwSupporter] reserved_vram_gb(applied)={rv}"
                        log.info("%s", msg)
                        print(msg, flush=True)
                    except Exception:
                        pass
            except Exception as e:
                report["warnings"].append(f"failed to set EXTRA_RESERVED_VRAM: {e!r}")

        # Patch attention
        model_out = model
        try:
            model_out = model.clone()
        except Exception as e:
            report["warnings"].append(f"model.clone() failed; patching in-place: {e!r}")
            model_out = model

        # Windows-safe defaults: Triton crashes are usually fatal (0x80000003).
        if platform.system() == "Windows":
            if sage_attention_effective == "auto":
                # Prefer CUDA kernel path in common cases.
                sage_attention_effective = "sageattn_qk_int8_pv_fp16_cuda"
                report["warnings"].append(
                    "Windows: sage_attention=auto remapped to sageattn_qk_int8_pv_fp16_cuda to avoid Triton-related crashes"
                )
            if "triton" in str(sage_attention_effective):
                report["warnings"].append(
                    "Windows: Triton-based sageattention modes may hard-crash (libtriton.pyd); consider using *_cuda mode or disabled"
                )

        if sage_attention_effective != "disabled":
            override, err = _try_make_sage_attention_override(sage_attention_effective, allow_sageattention_torch_compile)
            if override is None:
                if err:
                    report["warnings"].append(err)
                    if console_log:
                        try:
                            msg = f"[IAMCCS_HwSupporter] sage_attention NOT applied: {err}"
                            log.warning("%s", msg)
                            print(msg, flush=True)
                        except Exception:
                            pass
            else:
                try:
                    model_out.model_options.setdefault("transformer_options", {})
                    # ComfyUI variants differ in which key they consult. Set a few common ones.
                    transformer_options = model_out.model_options["transformer_options"]
                    keys_set: list[str] = []

                    transformer_options["optimized_attention_override"] = override
                    keys_set.append("optimized_attention_override")

                    # Compatibility fallbacks (safe if unused).
                    transformer_options["optimized_attention"] = override
                    keys_set.append("optimized_attention")
                    transformer_options["attention_override"] = override
                    keys_set.append("attention_override")

                    report["applied"]["sage_attention"] = sage_attention_effective
                    report["applied"]["allow_sageattention_torch_compile"] = bool(allow_sageattention_torch_compile)
                    report["applied"]["sage_attention_keys_set"] = keys_set

                    if console_log:
                        try:
                            msg = f"[IAMCCS_HwSupporter] sage_attention(applied)={sage_attention_effective} keys={keys_set}"
                            log.info("%s", msg)
                            print(msg, flush=True)
                        except Exception:
                            pass
                except Exception as e:
                    report["warnings"].append(f"failed to set optimized_attention_override: {e!r}")

        # fp16 accumulation
        val, warn = _apply_fp16_accumulation(fp16_accumulation)
        report["applied"]["fp16_accumulation"] = val
        if warn:
            report["warnings"].append(warn)

        # KJNodes-style: also pin the setting per-model via callbacks when available.
        try:
            if CallbacksMP is not None and hasattr(model_out, "add_callback") and hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                if fp16_accumulation == "on":
                    if not hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                        report["warnings"].append(
                            "fp16_accumulation=on requested but allow_fp16_accumulation is unavailable (requires newer PyTorch)"
                        )
                    else:
                        def _enable_fp16_accum(_model):
                            torch.backends.cuda.matmul.allow_fp16_accumulation = True

                        def _disable_fp16_accum(_model):
                            torch.backends.cuda.matmul.allow_fp16_accumulation = False

                        model_out.add_callback(CallbacksMP.ON_PRE_RUN, _enable_fp16_accum)
                        model_out.add_callback(CallbacksMP.ON_CLEANUP, _disable_fp16_accum)
                        report["applied"]["fp16_accumulation_callbacks"] = True
                elif fp16_accumulation == "off":
                    if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                        def _disable_fp16_accum(_model):
                            torch.backends.cuda.matmul.allow_fp16_accumulation = False

                        model_out.add_callback(CallbacksMP.ON_PRE_RUN, _disable_fp16_accum)
                        report["applied"]["fp16_accumulation_callbacks"] = True
        except Exception as e:
            report["warnings"].append(f"failed to attach fp16_accumulation callbacks: {e!r}")

        # tf32
        val, warn = _apply_tf32(tf32)
        report["applied"]["tf32"] = val
        if warn:
            report["warnings"].append(warn)

        # Apply behavior for auto.
        # IMPORTANT: auto should never silently block an explicit desire to compile; pick a safe default mode.
        if compile_mode_effective == "auto":
            compile_mode_effective = "reduce-overhead"
            if platform.system() == "Windows":
                report["warnings"].append(
                    "Windows: torch.compile in auto mode selects reduce-overhead (may still be unstable; set off if you see hard-crashes/stalls)."
                )

        if platform.system() == "Windows" and compile_mode_effective != "off":
            report["warnings"].append(
                "Windows: torch.compile requested explicitly. This may crash depending on Torch/Inductor/driver; if you see hard-crashes or stalls, set torch_compile_mode=off."
            )
        compiled, warn = _maybe_torch_compile_model(model_out, compile_mode_effective)
        report["applied"]["torch_compile_mode"] = compile_mode_effective
        report["applied"]["torch_compiled"] = bool(compiled)
        if warn:
            report["warnings"].append(warn)

        # Extra verification fields
        report["effective"] = {
            "torch_compile_mode": compile_mode_effective,
            "sage_attention": sage_attention_effective,
            "reserved_vram_mode": reserved_vram_mode,
        }

        report["timing_ms"] = int((time.time() - t0) * 1000)

        if console_log:
            try:
                applied = report.get("applied", {})
                attn = applied.get("sage_attention")
                compiled = applied.get("torch_compiled")
                reserved = applied.get("reserved_vram_gb")
                warn_count = len(report.get("warnings") or [])
                summary = (
                    f"[IAMCCS_HwSupporter] profile={profile} "
                    f"reserved_vram_gb={reserved} sage_attention={attn} torch_compile={compiled} "
                    f"(mode={applied.get('torch_compile_mode')} requested={torch_compile_mode_requested}) "
                    f"warnings={warn_count} ({report.get('timing_ms')}ms)"
                )

                # Print to both the ComfyUI logger and stdout.
                # Some setups filter INFO logs; print() remains visible in console.
                log.info("%s", summary)
                try:
                    print(summary, flush=True)
                except Exception:
                    pass

                if warn_count:
                    for w in (report.get("warnings") or [])[:8]:
                        log.warning("[IAMCCS_HwSupporter] %s", w)
                        try:
                            print(f"[IAMCCS_HwSupporter] {w}", flush=True)
                        except Exception:
                            pass
                    if warn_count > 8:
                        log.warning("[IAMCCS_HwSupporter] (more warnings omitted: %d)", warn_count - 8)
                        try:
                            print(f"[IAMCCS_HwSupporter] (more warnings omitted: {warn_count - 8})", flush=True)
                        except Exception:
                            pass
            except Exception:
                pass

        return model_out, clip, vae, json.dumps(report, ensure_ascii=False, indent=2)


class IAMCCS_HwSupporterAny:
    """HW Supporter variant with flexible ANY in/out.

    Useful when you want to apply global knobs (reserved VRAM, cleanup, TF32/fp16 accumulation)
    without needing MODEL/CLIP/VAE wiring.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": (any_type, {}),
                "profile": (_PROFILE_PRESETS, {"default": "auto"}),
                "apply_reserved_vram": ("BOOLEAN", {"default": True}),
                "reserved_vram_mode": (["manual", "auto_used_plus"], {"default": "manual"}),
                "reserved_vram_gb": ("FLOAT", {"default": 1.25, "min": 0.0, "max": 24.0, "step": 0.25}),
                "reserved_vram_auto_headroom_gb": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 24.0, "step": 0.25}),
                "reserved_vram_auto_max_gb": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 48.0, "step": 0.25}),
                "fp16_accumulation": (["auto", "on", "off"], {"default": "auto"}),
                "tf32": (["auto", "on", "off"], {"default": "auto"}),
                "clean_gpu_before": ("BOOLEAN", {"default": False}),
                "console_log": ("BOOLEAN", {"default": True}),
                "include_hardware_report": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (any_type, "STRING")
    RETURN_NAMES = ("output", "report_json")
    FUNCTION = "apply_any"
    CATEGORY = "IAMCCS/HW"

    def apply_any(
        self,
        input,
        profile,
        apply_reserved_vram,
        reserved_vram_mode,
        reserved_vram_gb,
        reserved_vram_auto_headroom_gb,
        reserved_vram_auto_max_gb,
        fp16_accumulation,
        tf32,
        clean_gpu_before,
        console_log,
        include_hardware_report,
    ):
        t0 = time.time()

        if clean_gpu_before:
            _gpu_cleanup(unload_all=True, soft_empty_cache=True)

        cuda_info = _safe_get_cuda_device_info()
        total_ram_gb = _safe_get_total_ram_gb()
        cuda_total = cuda_info.get("cuda_total_vram_gb")
        rec = _recommend_for_profile(profile, cuda_total)

        reserved_vram_mode = str(reserved_vram_mode or "manual")
        reserved_effective_gb = float(reserved_vram_gb)
        if profile == "auto" and reserved_effective_gb <= 0.0:
            reserved_effective_gb = float(rec["reserved_vram_gb"])

        report: dict = {
            "profile": profile,
            "platform": {
                "os": platform.system(),
                "os_release": platform.release(),
                "python": platform.python_version(),
                "torch": getattr(torch, "__version__", None),
            },
            "hardware": {**cuda_info, "system_ram_gb": total_ram_gb},
            "applied": {},
            "warnings": [],
            "recommendations": rec,
        }

        if include_hardware_report:
            try:
                from .iamccs_hw_probe import recommend_settings

                report["hardware_probe"] = recommend_settings()
            except Exception as e:
                report["warnings"].append(f"failed to run hardware_probe: {e!r}")

        if apply_reserved_vram:
            try:
                if reserved_vram_mode == "auto_used_plus":
                    auto_reserved, warns = _auto_reserved_vram_gb(
                        headroom_gb=float(reserved_vram_auto_headroom_gb),
                        auto_max_reserved_gb=float(reserved_vram_auto_max_gb),
                    )
                    for w in warns:
                        report["warnings"].append(w)
                    if auto_reserved is None:
                        reserved_bytes = _set_reserved_vram(reserved_effective_gb)
                        report["warnings"].append("auto_used_plus failed; fell back to reserved_vram_gb")
                        report["applied"]["reserved_vram_gb"] = _bytes_to_gb(reserved_bytes)
                    else:
                        reserved_bytes = _set_reserved_vram(float(auto_reserved))
                        report["applied"]["reserved_vram_gb"] = _bytes_to_gb(reserved_bytes)
                        report["applied"]["reserved_vram_auto_used_plus"] = {
                            "headroom_gb": float(reserved_vram_auto_headroom_gb),
                            "auto_max_reserved_gb": float(reserved_vram_auto_max_gb),
                            "effective_reserved_gb": float(auto_reserved),
                        }
                else:
                    reserved_bytes = _set_reserved_vram(reserved_effective_gb)
                    report["applied"]["reserved_vram_gb"] = _bytes_to_gb(reserved_bytes)
            except Exception as e:
                report["warnings"].append(f"failed to set EXTRA_RESERVED_VRAM: {e!r}")

        val, warn = _apply_fp16_accumulation(fp16_accumulation)
        report["applied"]["fp16_accumulation"] = val
        if warn:
            report["warnings"].append(warn)

        val, warn = _apply_tf32(tf32)
        report["applied"]["tf32"] = val
        if warn:
            report["warnings"].append(warn)

        report["timing_ms"] = int((time.time() - t0) * 1000)

        if console_log:
            try:
                warn_count = len(report.get("warnings") or [])
                summary = (
                    f"[IAMCCS_HwSupporterAny] profile={profile} "
                    f"reserved_vram_gb={report.get('applied', {}).get('reserved_vram_gb')} "
                    f"warnings={warn_count} ({report.get('timing_ms')}ms)"
                )
                log.info("%s", summary)
                try:
                    print(summary, flush=True)
                except Exception:
                    pass
            except Exception:
                pass

        return (input, json.dumps(report, ensure_ascii=False, indent=2))


class IAMCCS_VRAMCleanup:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unload_all_models": ("BOOLEAN", {"default": True}),
                "soft_empty_cache": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "run"
    CATEGORY = "IAMCCS/HW"

    def run(self, unload_all_models, soft_empty_cache, model=None, clip=None, vae=None):
        _gpu_cleanup(unload_all=bool(unload_all_models), soft_empty_cache=bool(soft_empty_cache))
        return model, clip, vae


class IAMCCS_VAEDecodeTiledSafe:
    @staticmethod
    def _auto_tile_params(vram_gb: float | None, compression: int) -> tuple[int, int]:
        # Conservative defaults for SD/SDXL VAE decode.
        # Note: tile_size refers to final image-space pixels (like many tiled-decode UIs).
        # We also ensure divisibility by (compression * 8) to reduce odd edge cases.
        # NOTE: More conservative than many image-only defaults, since video VAEs
        # tend to spike memory during decode.
        if vram_gb is None:
            tile_size = 384
        elif vram_gb <= 6.5:
            tile_size = 256
        elif vram_gb <= 8.5:
            tile_size = 320
        elif vram_gb <= 12.5:
            tile_size = 384
        elif vram_gb <= 16.5:
            tile_size = 512
        elif vram_gb <= 24.5:
            tile_size = 640
        else:
            tile_size = 768

        snap = max(64, int(compression) * 8)
        tile_size = max(64, (int(tile_size) // snap) * snap)
        overlap = min(96, max(32, tile_size // 8))
        overlap = (int(overlap) // 32) * 32
        if overlap < 0:
            overlap = 0
        if tile_size < overlap * 4:
            overlap = tile_size // 4
        return int(tile_size), int(overlap)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "tile": ("BOOLEAN", {"default": True}),
                "tiling_mode": (["auto", "manual"], {"default": "auto"}),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                "temporal_size": (
                    "INT",
                    {
                        "default": 24,
                        "min": 8,
                        "max": 4096,
                        "step": 4,
                        "tooltip": "Only for video VAEs: frames to decode per chunk (lower = less VRAM, slower).",
                    },
                ),
                "temporal_overlap": (
                    "INT",
                    {
                        "default": 4,
                        "min": 4,
                        "max": 4096,
                        "step": 4,
                        "tooltip": "Only for video VAEs: overlapped frames between chunks.",
                    },
                ),
                "cleanup_before_decode": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "IAMCCS/HW"

    def decode(self, samples, vae, tile, tiling_mode, tile_size, overlap, temporal_size, temporal_overlap, cleanup_before_decode):
        if cleanup_before_decode:
            # IMPORTANT: do not unload all models here.
            # Unloading can cause large stalls because ComfyUI will re-load/offload models again.
            _gpu_cleanup(unload_all=False, soft_empty_cache=True)

        # Detect video VAE compression if supported.
        temporal_compression = None
        try:
            if hasattr(vae, "temporal_compression_decode"):
                temporal_compression = vae.temporal_compression_decode()
        except Exception:
            temporal_compression = None

        if temporal_compression is not None:
            temporal_size = max(2, int(temporal_size) // int(temporal_compression))
            temporal_overlap = max(1, min(temporal_size // 2, int(temporal_overlap) // int(temporal_compression)))
        else:
            temporal_size = None
            temporal_overlap = None

        compression = 8
        try:
            if hasattr(vae, "spacial_compression_decode"):
                compression = int(vae.spacial_compression_decode())
        except Exception:
            compression = 8

        if tiling_mode == "auto":
            cuda_info = _safe_get_cuda_device_info()
            vram_gb = cuda_info.get("cuda_total_vram_gb")
            tile_size, overlap = self._auto_tile_params(vram_gb, compression)

        # Keep overlap sane.
        if tile_size < overlap * 4:
            overlap = tile_size // 4
        if temporal_size is not None and temporal_overlap is not None:
            if temporal_size < temporal_overlap * 2:
                temporal_overlap = temporal_overlap // 2

        latents = samples.get("samples") if isinstance(samples, dict) else None
        if latents is None:
            raise ValueError("Invalid LATENT input: expected dict with key 'samples'")

        def _is_cpu_allocator_oom(err: BaseException) -> bool:
            msg = str(err)
            msg_l = msg.lower()
            return (
                "defaultcpuallocator" in msg_l
                or "alloc_cpu.cpp" in msg_l
                or "not enough memory" in msg_l
                or "you tried to allocate" in msg_l
            )

        def _decode_video_by_time_slices(ts: int, ov: int):
            # Avoid ComfyUI's decode_tiled_3d/tiled_scale_multidim CPU accumulation buffers,
            # which can OOM even when the final output might fit.
            if not hasattr(latents, "shape"):
                return None
            if len(latents.shape) != 5:
                return None

            b, c, t, h, w = latents.shape
            if t <= 1:
                return None

            images_out = None
            for ti in range(int(t)):
                frame_latents = latents[:, :, ti, :, :]
                if tile and hasattr(vae, "decode_tiled"):
                    frame_img = vae.decode_tiled(
                        frame_latents,
                        tile_x=int(ts) // compression,
                        tile_y=int(ts) // compression,
                        overlap=int(ov) // compression,
                        tile_t=None,
                        overlap_t=None,
                    )
                else:
                    frame_img = vae.decode(frame_latents)

                # Expected ComfyUI image tensor: [B, H, W, C]
                if images_out is None:
                    if len(frame_img.shape) != 4:
                        raise RuntimeError(
                            f"Unexpected VAE decoded frame shape: {tuple(frame_img.shape)} (expected 4D [B,H,W,C])"
                        )
                    images_out = torch.empty(
                        (int(b) * int(t), int(frame_img.shape[1]), int(frame_img.shape[2]), int(frame_img.shape[3])),
                        device=frame_img.device,
                        dtype=frame_img.dtype,
                    )

                start = int(ti) * int(b)
                end = start + int(b)
                images_out[start:end] = frame_img

            return images_out

        def _try_decode(ts: int, ov: int, tt: int | None, ot: int | None):
            if tile and hasattr(vae, "decode_tiled"):
                return vae.decode_tiled(
                    latents,
                    tile_x=int(ts) // compression,
                    tile_y=int(ts) // compression,
                    overlap=int(ov) // compression,
                    tile_t=tt,
                    overlap_t=ot,
                )
            return vae.decode(latents)

        # OOM-safe decode: try current params, then shrink tiles/temporal chunk.
        images = None
        last_err: Exception | None = None
        attempts = [
            (int(tile_size), int(overlap), temporal_size, temporal_overlap),
            (max(256, int(tile_size) // 2), max(0, int(overlap) // 2), (max(8, int(temporal_size) // 2) if temporal_size is not None else None), (max(4, int(temporal_overlap) // 2) if temporal_overlap is not None else None)),
        ]
        for i, (ts, ov, tt, ot) in enumerate(attempts, start=1):
            try:
                images = _try_decode(ts, ov, tt, ot)
                if i > 1:
                    log.warning(
                        "[IAMCCS_VAEDecodeTiledSafe] decode succeeded after fallback attempt %d (tile=%s overlap=%s tile_t=%s overlap_t=%s)",
                        i,
                        ts,
                        ov,
                        tt,
                        ot,
                    )
                break
            except RuntimeError as e:
                last_err = e
                msg = str(e).lower()
                is_oom = ("out of memory" in msg) or ("cuda" in msg and "memory" in msg)
                if not is_oom:
                    # Special case: CPU allocator OOM during video decode.
                    if _is_cpu_allocator_oom(e):
                        try:
                            images = _decode_video_by_time_slices(ts, ov)
                            if images is not None:
                                log.warning(
                                    "[IAMCCS_VAEDecodeTiledSafe] CPU OOM avoided by decoding video per-frame (tile=%s overlap=%s).",
                                    ts,
                                    ov,
                                )
                                break
                        except Exception as e2:
                            last_err = e2
                    raise

                # If this is CPU allocator OOM, try per-frame decode before shrinking tiles.
                if _is_cpu_allocator_oom(e):
                    try:
                        images = _decode_video_by_time_slices(ts, ov)
                        if images is not None:
                            log.warning(
                                "[IAMCCS_VAEDecodeTiledSafe] CPU OOM avoided by decoding video per-frame (tile=%s overlap=%s).",
                                ts,
                                ov,
                            )
                            break
                    except Exception as e2:
                        last_err = e2
                        # Fall through to the existing retry logic.
                log.warning(
                    "[IAMCCS_VAEDecodeTiledSafe] OOM during decode attempt %d; shrinking tiles and retrying. Error: %s",
                    i,
                    e,
                )
                _gpu_cleanup(unload_all=False, soft_empty_cache=True)
            except Exception as e:
                last_err = e
                break

        if images is None:
            if last_err:
                raise last_err
            raise RuntimeError("VAE decode failed: unknown error")

        # Combine batches if needed.
        try:
            if len(images.shape) == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        except Exception:
            pass

        return (images,)


class IAMCCS_VAEDecodeToDisk:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "output_dir": (
                    "STRING",
                    {
                        "default": "iamccs_vae_frames",
                        "tooltip": "Folder (relative to ComfyUI output dir if relative). Frames will be written here.",
                    },
                ),
                "prefix": ("STRING", {"default": "frame"}),
                "image_format": (["png", "jpg"], {"default": "png"}),
                "jpg_quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
                "tile": ("BOOLEAN", {"default": True}),
                "tiling_mode": (["auto", "manual"], {"default": "auto"}),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                "cleanup_between_frames": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If enabled, runs a light cache cleanup after each frame decode (slower, lower peak VRAM).",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("frames_dir", "frames_saved")
    FUNCTION = "decode_to_disk"
    CATEGORY = "IAMCCS/HW"

    def decode_to_disk(
        self,
        samples,
        vae,
        output_dir: str,
        prefix: str,
        image_format: str,
        jpg_quality: int,
        tile: bool,
        tiling_mode: str,
        tile_size: int,
        overlap: int,
        cleanup_between_frames: bool,
    ):
        try:
            from PIL import Image  # type: ignore
        except Exception as e:
            raise RuntimeError(f"PIL (Pillow) is required for IAMCCS_VAEDecodeToDisk: {e!r}")

        # Resolve output path.
        out_dir = str(output_dir or "iamccs_vae_frames").strip()
        if not out_dir:
            out_dir = "iamccs_vae_frames"

        # If user passes a relative path, put it under ComfyUI output directory.
        try:
            from folder_paths import get_output_directory  # type: ignore

            base_out = get_output_directory()
        except Exception:
            base_out = os.getcwd()

        if not os.path.isabs(out_dir):
            out_dir = os.path.join(base_out, out_dir)
        os.makedirs(out_dir, exist_ok=True)

        # Detect video VAE compression if supported.
        compression = 8
        try:
            if hasattr(vae, "spacial_compression_decode"):
                compression = int(vae.spacial_compression_decode())
        except Exception:
            compression = 8

        # Auto tile sizing.
        if str(tiling_mode) == "auto":
            cuda_info = _safe_get_cuda_device_info()
            vram_gb = cuda_info.get("cuda_total_vram_gb")
            tile_size, overlap = IAMCCS_VAEDecodeTiledSafe._auto_tile_params(vram_gb, compression)

        latents = samples.get("samples") if isinstance(samples, dict) else None
        if latents is None or not hasattr(latents, "shape"):
            raise ValueError("Invalid LATENT input: expected dict with key 'samples'")

        def _decode_latent_frame(frame_latents):
            if bool(tile) and hasattr(vae, "decode_tiled"):
                return vae.decode_tiled(
                    frame_latents,
                    tile_x=int(tile_size) // compression,
                    tile_y=int(tile_size) // compression,
                    overlap=int(overlap) // compression,
                    tile_t=None,
                    overlap_t=None,
                )
            return vae.decode(frame_latents)

        def _save_image_tensor(img_t, path: str):
            # Expected ComfyUI image tensor: [B, H, W, C]
            if len(img_t.shape) != 4:
                raise RuntimeError(f"Unexpected decoded image shape: {tuple(img_t.shape)} (expected 4D [B,H,W,C])")

            img_cpu = img_t.detach().to("cpu")
            img_cpu = torch.clamp(img_cpu, 0.0, 1.0)
            # Save each batch item.
            b = int(img_cpu.shape[0])
            for bi in range(b):
                arr = (img_cpu[bi].numpy() * 255.0).round().astype("uint8")
                im = Image.fromarray(arr)
                if image_format == "jpg":
                    im.save(path.replace("{b}", f"{bi:02d}"), format="JPEG", quality=int(jpg_quality))
                else:
                    im.save(path.replace("{b}", f"{bi:02d}"), format="PNG")

        frames_saved = 0

        # Video latent: [B, C, T, H, W]
        if len(latents.shape) == 5:
            b, c, t, h, w = latents.shape
            if int(t) <= 0:
                raise ValueError("Invalid video latent: T must be > 0")

            for ti in range(int(t)):
                frame_latents = latents[:, :, ti, :, :]
                frame_img = _decode_latent_frame(frame_latents)
                filename = f"{prefix}_{ti:05d}_b{{b}}.{image_format}"
                _save_image_tensor(frame_img, os.path.join(out_dir, filename))
                frames_saved += int(frame_img.shape[0])

                if bool(cleanup_between_frames):
                    _gpu_cleanup(unload_all=False, soft_empty_cache=True)
        else:
            # Image latent: assume [B, C, H, W]
            img = _decode_latent_frame(latents)
            filename = f"{prefix}_00000_b{{b}}.{image_format}"
            _save_image_tensor(img, os.path.join(out_dir, filename))
            frames_saved += int(img.shape[0])

        return (out_dir, int(frames_saved))
