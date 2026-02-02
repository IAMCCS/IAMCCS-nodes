import platform
import time
from typing import Any, Dict


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
    try:
        import torch

        if not torch.cuda.is_available():
            return None, None, "CUDA not available"
        free_b, total_b = torch.cuda.mem_get_info()
        total_gb = float(total_b) / (1024.0**3)
        used_gb = max(0.0, float(total_b - free_b) / (1024.0**3))
        return total_gb, used_gb, None
    except Exception as e:
        return None, None, f"torch.cuda.mem_get_info failed: {e!r}"


def _estimate_reserved_vram_gb(headroom_gb: float, auto_max_reserved_gb: float = 0.0) -> tuple[float | None, list[str]]:
    """Estimate effective reserved VRAM as (used + headroom), like ReservedVRAMSetter."""
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
        reserved = min(reserved, max(0.0, float(total_gb) - 0.05))
    except Exception:
        pass
    return float(round(reserved, 2)), warnings


def _bytes_to_gb(x: float | int | None) -> float | None:
    if x is None:
        return None
    try:
        return float(x) / (1024.0**3)
    except Exception:
        return None


def _safe_total_ram_gb() -> float | None:
    # Prefer psutil if available
    try:
        import psutil  # type: ignore

        return _bytes_to_gb(psutil.virtual_memory().total)
    except Exception:
        pass

    # Windows fallback via ctypes
    if platform.system() == "Windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
            return _bytes_to_gb(mem.ullTotalPhys)
        except Exception:
            return None

    return None


def _safe_cuda_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "cuda_available": False,
        "cuda_device": None,
        "cuda_device_name": None,
        "cuda_total_vram_gb": None,
        "cuda_capability": None,
    }

    try:
        import torch

        if not torch.cuda.is_available():
            return info

        idx = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(idx)
        cap = None
        try:
            cap = torch.cuda.get_device_capability(idx)
        except Exception:
            cap = None

        info.update(
            {
                "cuda_available": True,
                "cuda_device": int(idx),
                "cuda_device_name": getattr(prop, "name", None),
                "cuda_total_vram_gb": _bytes_to_gb(getattr(prop, "total_memory", None)),
                "cuda_capability": cap,
            }
        )

        return info
    except Exception:
        return info


def _snap_int(value: int, step: int, min_value: int, max_value: int) -> int:
    if step <= 0:
        return max(min_value, min(max_value, int(value)))
    v = int(value)
    v = (v // step) * step
    return max(min_value, min(max_value, v))


def _recommend_vae_decode(
    vram_gb: float | None,
    width: int | None,
    height: int | None,
    frames: int | None,
    fps: float | None,
) -> Dict[str, Any]:
    """Heuristic recommendations for IAMCCS_VAEDecodeTiledSafe.

    Goal: reduce peak VRAM during decode for long clips/high resolution.
    """

    # Base tile size by VRAM (image-space pixels)
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

    # Adjust by resolution and length.
    mp = None
    if width and height and width > 0 and height > 0:
        mp = (float(width) * float(height)) / 1_000_000.0

    fcount = int(frames) if frames and frames > 0 else None
    fps_val = float(fps) if fps and fps > 0 else None

    # Penalize very high resolutions.
    if mp is not None:
        if mp >= 1.5:
            tile_size -= 128
        elif mp >= 1.0:
            tile_size -= 64

    # Penalize long clips (decode memory spikes scale with temporal chunking).
    if fcount is not None:
        if fcount >= 320:
            tile_size -= 64
        elif fcount >= 200:
            tile_size -= 32

    # Slight penalty for very high fps (tends to correlate with longer/denser outputs)
    if fps_val is not None and fps_val >= 48:
        tile_size -= 32

    tile_size = _snap_int(tile_size, step=64, min_value=192, max_value=1024)
    overlap = _snap_int(max(32, tile_size // 8), step=32, min_value=0, max_value=160)
    if tile_size < overlap * 4:
        overlap = _snap_int(tile_size // 4, step=32, min_value=0, max_value=160)

    # Temporal chunking: key lever for long videos.
    # IMPORTANT: keep temporal_size >= 64 (widget-scale) for quality.
    # Video VAEs often apply temporal compression internally, so too-small values
    # can produce visible chunk boundary artifacts.
    if vram_gb is None:
        temporal_size = 64
    elif vram_gb <= 8.5:
        temporal_size = 64
    elif vram_gb <= 12.5:
        temporal_size = 64
    elif vram_gb <= 16.5:
        temporal_size = 96
    else:
        temporal_size = 128

    if fcount is not None:
        # Can't exceed total frames. If frames < 64, use frames.
        temporal_size = min(temporal_size, max(8, fcount))
        if fcount >= 64:
            temporal_size = max(64, temporal_size)

    temporal_size = _snap_int(temporal_size, step=4, min_value=8, max_value=4096)

    # Quality-oriented overlap: don't go too low.
    temporal_overlap = 16 if temporal_size >= 64 else 8
    temporal_overlap = min(temporal_overlap, max(4, temporal_size // 2))
    temporal_overlap = _snap_int(temporal_overlap, step=4, min_value=4, max_value=max(4, temporal_size // 2))

    return {
        "tile": True,
        "tiling_mode": "manual",
        "tile_size": int(tile_size),
        "overlap": int(overlap),
        "temporal_size": int(temporal_size),
        "temporal_overlap": int(temporal_overlap),
        "cleanup_before_decode": True,
        "context": {
            "width": width,
            "height": height,
            "frames": frames,
            "fps": fps,
            "megapixels": mp,
        },
    }


def recommend_settings(width: int | None = None, height: int | None = None, frames: int | None = None, fps: float | None = None) -> Dict[str, Any]:
    """Hardware probe + recommended IAMCCS settings.

    This is intended for UI-side "auto apply" and for embedding into node reports.
    """

    cuda = _safe_cuda_info()
    ram_gb = _safe_total_ram_gb()

    vram = cuda.get("cuda_total_vram_gb")
    cap = cuda.get("cuda_capability")
    is_windows = platform.system() == "Windows"

    # Profile selection (conservative)
    if vram is None:
        profile = "auto"
    elif vram <= 8.5:
        profile = "low_vram"
    elif vram <= 16.5:
        profile = "balanced"
    else:
        profile = "max_speed"

    # TF32 heuristic: mainly Ampere+ (8.0+) or Ada (8.9)
    tf32 = "off"
    if cap and isinstance(cap, (tuple, list)) and len(cap) >= 2:
        major = int(cap[0])
        if major >= 8:
            tf32 = "auto"

    # Reserved VRAM headroom heuristic
    if vram is None:
        headroom = 1.0
    elif vram <= 8.5:
        headroom = 1.25
    elif vram <= 12.5:
        headroom = 1.5
    elif vram <= 16.5:
        headroom = 1.75
    else:
        headroom = 2.0

    # Estimate effective reserved VRAM (used + headroom) so UI can display it.
    reserved_est_gb, reserved_warns = _estimate_reserved_vram_gb(headroom_gb=float(headroom), auto_max_reserved_gb=0.0)


    # SageAttention heuristic
    if not cuda.get("cuda_available"):
        sage_attention = "disabled"
    else:
        # Keep Windows stable by preferring CUDA kernel path
        sage_attention = "sageattn_qk_int8_pv_fp16_cuda" if is_windows else "auto"

    # torch.compile heuristic (Windows tends to be brittle)
    torch_compile_mode = "off" if is_windows else "auto"

    recommendations = {
        "hw_supporter": {
            "profile": profile,
            "apply_reserved_vram": True,
            "reserved_vram_mode": "auto_used_plus",
            "reserved_vram_auto_headroom_gb": float(headroom),
            "reserved_vram_auto_max_gb": 0.0,
            # Not used by auto_used_plus at runtime, but useful as a visible "effective" number in the UI.
            "reserved_vram_gb": float(reserved_est_gb) if reserved_est_gb is not None else 0.0,
            "reserved_vram_effective_gb": float(reserved_est_gb) if reserved_est_gb is not None else None,
            "reserved_vram_probe_warnings": reserved_warns,
            "sage_attention": sage_attention,
            "allow_sageattention_torch_compile": False,
            "torch_compile_mode": torch_compile_mode,
            "fp16_accumulation": "auto",
            "tf32": tf32,
            "clean_gpu_before": False,
            "include_hardware_report": True,
        },
        "vae_decode": _recommend_vae_decode(vram, width, height, frames, fps),
        "gguf_accelerator": None,
        "sampler": None,
    }

    # GGUF accelerator heuristics.
    # The goal is to reduce per-step patch movement overhead while keeping OOM risk reasonable.
    if vram is None:
        gguf = {
            "mode": "auto_oom_safe",
            "patch_on_device": True,
            "move_patches_now": True,
            "move_policy": "all_or_nothing",
            "leave_free_vram_mb": 1024,
            "min_free_vram_mb": 1500,
            "oom_fallback": True,
        }
    elif vram <= 8.5:
        gguf = {
            "mode": "auto_oom_safe",
            "patch_on_device": True,
            "move_patches_now": True,
            "move_policy": "partial_small_first",
            "leave_free_vram_mb": 1500,
            "min_free_vram_mb": 1500,
            "oom_fallback": True,
        }
    elif vram <= 16.5:
        gguf = {
            "mode": "auto_oom_safe",
            "patch_on_device": True,
            "move_patches_now": True,
            "move_policy": "all_or_nothing",
            "leave_free_vram_mb": 1200,
            "min_free_vram_mb": 1500,
            "oom_fallback": True,
        }
    else:
        gguf = {
            "mode": "manual",
            "patch_on_device": True,
            "move_patches_now": True,
            "move_policy": "all_or_nothing",
            "leave_free_vram_mb": 1024,
            "min_free_vram_mb": 0,
            "oom_fallback": True,
        }

    recommendations["gguf_accelerator"] = gguf

    # Sampler heuristics (works for IAMCCS_SamplerAdvancedVersion1).
    # disable_progress reduces UI overhead on long denoise loops; cleanup helps low VRAM.
    if vram is not None and vram <= 8.5:
        sampler = {"disable_progress": True, "cleanup": True}
    else:
        sampler = {"disable_progress": True, "cleanup": False}
    recommendations["sampler"] = sampler

    return {
        "timestamp": int(time.time()),
        "platform": {
            "os": platform.system(),
            "os_release": platform.release(),
            "python": platform.python_version(),
        },
        "hardware": {
            **cuda,
            "system_ram_gb": ram_gb,
        },
        "recommendations": recommendations,
        "notes": [
            "These are heuristics: best values depend on resolution, frame count, and model.",
            *([f"Reserved VRAM estimate warning: {w}" for w in reserved_warns] if reserved_warns else []),
        ],
    }
