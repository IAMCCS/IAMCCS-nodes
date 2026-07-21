"""Optional NVIDIA RTX Video Super Resolution engine for IAMCCS exporters.

This is a direct VideoSuperRes integration. It does not instantiate or call
the Deno Comfy node, so the exporter can use the same native RTX VFX runtime
without adding a graph dependency.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F


RTX_QUALITY_LEVELS = [
    "VSR Medium",
    "VSR High",
    "VSR Low",
    "VSR Ultra",
    "High Bitrate Medium",
    "High Bitrate High",
    "High Bitrate Low",
    "High Bitrate Ultra",
    "Denoise Medium",
    "Denoise High",
    "Denoise Low",
    "Denoise Ultra",
    "Deblur Medium",
    "Deblur High",
    "Deblur Low",
    "Deblur Ultra",
]

RTX_RESIZE_TYPES = ["Scale", "Keep Ratio", "Manual", "Preset Ratio", "Same Size"]
RTX_DIVISIBLE_BY_VALUES = ["1", "8", "16", "32", "64", "128"]
RTX_RESIZE_METHODS = ["Center Crop (Fill)", "Fit (Letterbox/Pillarbox)"]
RTX_COMMON_RATIOS = [
    "1:1", "4:5", "5:4", "3:4", "4:3", "2:3", "3:2", "16:9", "9:16",
    "16:10", "10:16", "21:9", "9:21",
]
RTX_DEFAULT_DIVISIBLE_BY = "8"
RTX_INSTALL_GUIDE_URL = "https://deno2026.github.io/comfyui-deno-custom-nodes/rtx-vfx-install/"
_RUNTIME_MARKER_NAME = "DENO_RTX_VFX_runtime_path.txt"


def _safe_divisible_by(value) -> int:
    try:
        value = int(value)
    except Exception:
        return int(RTX_DEFAULT_DIVISIBLE_BY)
    if str(value) not in RTX_DIVISIBLE_BY_VALUES:
        return int(RTX_DEFAULT_DIVISIBLE_BY)
    return value


def _round_up(value: float, multiple: int) -> int:
    return int(math.ceil(max(float(value), float(multiple)) / multiple) * multiple)


def _compute_aligned_ratio_dims(ratio_preset: str, megapixels: float, divisible_by: int) -> Tuple[int, int]:
    ratio_x, ratio_y = (int(part) for part in str(ratio_preset).split(":", 1))
    total_pixels = max(0.01, float(megapixels)) * 1_000_000.0
    alignment = int(divisible_by)
    base_width = math.sqrt(total_pixels * ratio_x / ratio_y)
    base_height = math.sqrt(total_pixels * ratio_y / ratio_x)

    def round_down(value: float) -> int:
        return max(alignment, int(math.floor(float(value) / alignment) * alignment))

    width_candidates = sorted({_round_up(base_width, alignment), round_down(base_width)})
    height_candidates = sorted({_round_up(base_height, alignment), round_down(base_height)})
    candidates = set()
    for width_candidate in width_candidates:
        exact_height = width_candidate * ratio_y / ratio_x
        candidates.add((width_candidate, _round_up(exact_height, alignment)))
        candidates.add((width_candidate, round_down(exact_height)))
    for height_candidate in height_candidates:
        exact_width = height_candidate * ratio_x / ratio_y
        candidates.add((_round_up(exact_width, alignment), height_candidate))
        candidates.add((round_down(exact_width), height_candidate))

    preferred_dimensions = [512, 720, 768, 1024, 1088, 1536, 1920]

    def candidate_score(dims: Tuple[int, int]) -> Tuple[float, float, float, float]:
        width, height = dims
        area_error = abs((width * height) - total_pixels) / total_pixels
        width_error = abs(width - base_width) / base_width
        height_error = abs(height - base_height) / base_height
        ratio_error = abs((width / height) - (ratio_x / ratio_y)) / (ratio_x / ratio_y)
        preference_error = min(abs(width - preferred) for preferred in preferred_dimensions) + min(
            abs(height - preferred) for preferred in preferred_dimensions
        )
        return (width_error + height_error, preference_error, area_error, ratio_error)

    return min(candidates, key=candidate_score)


def _aligned_megapixel_size(source_width: int, source_height: int, megapixels: float, divisible_by: int) -> Tuple[int, int]:
    alignment = _safe_divisible_by(divisible_by)
    target_area = max(0.01, float(megapixels)) * 1_000_000.0
    source_aspect = float(source_width) / float(source_height)
    source_area = max(1.0, float(source_width * source_height))
    scale = math.sqrt(target_area / source_area)
    base_width = max(float(alignment), float(source_width) * scale)
    base_height = max(float(alignment), float(source_height) * scale)

    def round_down(value: float) -> int:
        return max(alignment, int(math.floor(float(value) / alignment) * alignment))

    def round_nearest(value: float) -> int:
        return max(alignment, int(math.floor((float(value) / alignment) + 0.5) * alignment))

    candidates = set()
    for width_rounder in (round_down, round_nearest, lambda value: _round_up(value, alignment)):
        width_candidate = width_rounder(base_width)
        exact_height = width_candidate / source_aspect
        for height_rounder in (round_down, round_nearest, lambda value: _round_up(value, alignment)):
            candidates.add((width_candidate, height_rounder(exact_height)))
    for height_rounder in (round_down, round_nearest, lambda value: _round_up(value, alignment)):
        height_candidate = height_rounder(base_height)
        exact_width = height_candidate * source_aspect
        for width_rounder in (round_down, round_nearest, lambda value: _round_up(value, alignment)):
            candidates.add((width_rounder(exact_width), height_candidate))

    def candidate_score(dims: Tuple[int, int]) -> Tuple[float, float, float]:
        width, height = dims
        area_error = abs((width * height) - target_area) / target_area
        ratio_error = abs((width / height) - source_aspect) / source_aspect
        distance_error = abs(width - base_width) / base_width + abs(height - base_height) / base_height
        return (ratio_error, area_error, distance_error)

    return min(candidates, key=candidate_score)


def _same_size_only(mode: str) -> bool:
    return str(mode).startswith("Denoise ") or str(mode).startswith("Deblur ")


def _target_size(
    source_width: int,
    source_height: int,
    mode: str,
    resize_type: str,
    scale: float,
    megapixels: float,
    width: int,
    height: int,
    divisible_by: int,
    ratio_preset: str,
) -> Tuple[int, int]:
    alignment = _safe_divisible_by(divisible_by)
    if _same_size_only(mode) or resize_type == "Same Size":
        return source_width, source_height
    if resize_type == "Scale":
        return (
            _round_up(float(source_width) * float(scale), alignment),
            _round_up(float(source_height) * float(scale), alignment),
        )
    if resize_type == "Keep Ratio":
        return _aligned_megapixel_size(source_width, source_height, megapixels, alignment)
    if resize_type == "Preset Ratio":
        return _compute_aligned_ratio_dims(ratio_preset, megapixels, alignment)
    return (_round_up(int(width), alignment), _round_up(int(height), alignment))


def _fit_frame_to_target_aspect(frame, target_width: int, target_height: int, resize_method: str):
    source_channels, source_height, source_width = frame.shape
    source_aspect = float(source_width) / float(source_height)
    target_aspect = float(target_width) / float(target_height)
    if abs(source_aspect - target_aspect) < 0.0001:
        return frame.contiguous()
    if resize_method == "Center Crop (Fill)":
        if source_aspect > target_aspect:
            crop_width = max(1, min(int(source_width), int(round(float(source_height) * target_aspect))))
            crop_x = max(0, (int(source_width) - crop_width) // 2)
            return frame[:, :, crop_x:crop_x + crop_width].contiguous()
        crop_height = max(1, min(int(source_height), int(round(float(source_width) / target_aspect))))
        crop_y = max(0, (int(source_height) - crop_height) // 2)
        return frame[:, crop_y:crop_y + crop_height, :].contiguous()
    if source_aspect > target_aspect:
        padded_height = max(int(source_height), int(math.ceil(float(source_width) / target_aspect)))
        pad_total = padded_height - int(source_height)
        return F.pad(frame, (0, 0, pad_total // 2, pad_total - (pad_total // 2)), mode="constant", value=0.0).contiguous()
    padded_width = max(int(source_width), int(math.ceil(float(source_height) * target_aspect)))
    pad_total = padded_width - int(source_width)
    return F.pad(frame, (pad_total // 2, pad_total - (pad_total // 2), 0, 0), mode="constant", value=0.0).contiguous()


def _runtime_marker_candidates():
    package_dir = Path(__file__).resolve().parent
    custom_nodes_dir = package_dir.parent
    yield package_dir / "tools" / _RUNTIME_MARKER_NAME
    yield custom_nodes_dir / "comfyui-deno-custom-nodes" / "tools" / _RUNTIME_MARKER_NAME
    yield custom_nodes_dir / "deno-custom-nodes" / "tools" / _RUNTIME_MARKER_NAME


def _read_runtime_path() -> Path | None:
    override = os.environ.get("IAMCCS_RTX_VFX_RUNTIME_PATH", "").strip().strip('"')
    if override:
        path = Path(os.path.expandvars(override))
        if (path / "nvvfx").is_dir():
            return path
    expected = f"py{sys.version_info[0]}{sys.version_info[1]}"
    for marker in _runtime_marker_candidates():
        try:
            raw = marker.read_text(encoding="utf-8").strip().strip('"')
        except OSError:
            continue
        if not raw:
            continue
        path = Path(os.path.expandvars(raw))
        if (path / "nvvfx").is_dir() and any(part.lower() == expected.lower() for part in path.parts):
            return path
    return None


def _norm_path(path: Path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except (OSError, ValueError):
        return False
    return True


def _prefer_runtime_path() -> Path | None:
    runtime_path = _read_runtime_path()
    if runtime_path is None:
        return None
    key = _norm_path(runtime_path)
    sys.path[:] = [entry for entry in sys.path if _norm_path(Path(entry or os.curdir)) != key]
    sys.path.insert(0, str(runtime_path))
    return runtime_path


def _current_nvvfx_package_path() -> Path | None:
    module = sys.modules.get("nvvfx")
    if module is None:
        return None
    module_paths = getattr(module, "__path__", None)
    if module_paths:
        for path in module_paths:
            return Path(path)
    module_file = getattr(module, "__file__", None)
    return Path(module_file).parent if module_file else None


def _loaded_nvvfx_module_paths() -> dict[str, str]:
    result = {}
    for module_name in ("nvvfx", "nvvfx._ext", "nvvfx._lib_loader"):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        module_file = getattr(module, "__file__", None)
        module_path = getattr(module, "__path__", None)
        result[module_name] = str(module_file or ";".join(str(path) for path in module_path or []) or "loaded")
    return result


def _loaded_broadcast_vfx_module_paths() -> list[str]:
    if sys.platform != "win32":
        return []
    try:
        import ctypes
        from ctypes import wintypes

        hmodule = getattr(wintypes, "HMODULE", wintypes.HANDLE)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.EnumProcessModulesEx.argtypes = [wintypes.HANDLE, ctypes.POINTER(hmodule), wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.DWORD]
        psapi.EnumProcessModulesEx.restype = wintypes.BOOL
        psapi.GetModuleFileNameExW.argtypes = [wintypes.HANDLE, hmodule, wintypes.LPWSTR, wintypes.DWORD]
        psapi.GetModuleFileNameExW.restype = wintypes.DWORD
        handle = kernel32.GetCurrentProcess()
        modules = (hmodule * 2048)()
        needed = wintypes.DWORD()
        if not psapi.EnumProcessModulesEx(handle, modules, ctypes.sizeof(modules), ctypes.byref(needed), 0x03):
            return []
        count = min(int(needed.value / ctypes.sizeof(hmodule)), len(modules))
        paths = []
        for module in modules[:count]:
            buffer = ctypes.create_unicode_buffer(32768)
            length = psapi.GetModuleFileNameExW(handle, module, buffer, len(buffer))
            if length:
                paths.append(buffer.value)
        return [path for path in paths if "\\programdata\\nvidia\\ngx\\models\\nvbcast\\" in _norm_path(Path(path)) or "\\nvbcast\\versions\\" in _norm_path(Path(path))]
    except Exception:
        return []


def _runtime_status() -> str:
    runtime = _read_runtime_path()
    loaded = _current_nvvfx_package_path()
    native = ", ".join(f"{key}={value}" for key, value in _loaded_nvvfx_module_paths().items()) or "none"
    broadcast = "; ".join(_loaded_broadcast_vfx_module_paths()[:5]) or "none"
    return f"Runtime: {runtime or 'not prepared'}. Loaded nvvfx: {loaded or 'unknown'}. Native: {native}. Broadcast VFX: {broadcast}."


def _import_video_super_res():
    runtime_path = _prefer_runtime_path()
    loaded_path = _current_nvvfx_package_path()
    if runtime_path is not None and loaded_path is not None and not _is_relative_to(loaded_path, runtime_path / "nvvfx"):
        raise RuntimeError("NVIDIA RTX VFX is already loaded from another runtime. Close ComfyUI completely and restart it. " + _runtime_status())
    if "nvvfx._ext" in _loaded_nvvfx_module_paths() and loaded_path is None:
        raise RuntimeError("NVIDIA RTX VFX native extension is partially loaded in this process. Close ComfyUI completely and restart it. " + _runtime_status())
    try:
        from nvvfx import VideoSuperRes
    except Exception as exc:
        raise RuntimeError(
            "NVIDIA RTX VFX could not be imported by IAMCCS. Install the Deno RTX VFX runtime for this ComfyUI Python, "
            f"then restart ComfyUI. See {RTX_INSTALL_GUIDE_URL}. Original error: {type(exc).__name__}: {exc}. "
            + _runtime_status()
        ) from exc
    return VideoSuperRes


def _quality_attr(mode: str) -> str:
    return {
        "VSR Low": "LOW", "VSR Medium": "MEDIUM", "VSR High": "HIGH", "VSR Ultra": "ULTRA",
        "High Bitrate Low": "HIGHBITRATE_LOW", "High Bitrate Medium": "HIGHBITRATE_MEDIUM",
        "High Bitrate High": "HIGHBITRATE_HIGH", "High Bitrate Ultra": "HIGHBITRATE_ULTRA",
        "Denoise Low": "DENOISE_LOW", "Denoise Medium": "DENOISE_MEDIUM", "Denoise High": "DENOISE_HIGH", "Denoise Ultra": "DENOISE_ULTRA",
        "Deblur Low": "DEBLUR_LOW", "Deblur Medium": "DEBLUR_MEDIUM", "Deblur High": "DEBLUR_HIGH", "Deblur Ultra": "DEBLUR_ULTRA",
    }[mode]


def _safe_cuda_device_index(device: int) -> int:
    try:
        value = int(device)
    except Exception:
        return 0
    try:
        count = int(torch.cuda.device_count())
    except Exception:
        count = 0
    return 0 if value < 0 or (count and value >= count) else value


def apply_rtx_vfx(
    images: torch.Tensor,
    mode: str = "VSR Medium",
    resize_type: str = "Keep Ratio",
    scale: float = 2.0,
    megapixels: float = 2.0,
    width: int = 1920,
    height: int = 1080,
    divisible_by: str = RTX_DEFAULT_DIVISIBLE_BY,
    device: int = 0,
    ratio_preset: str = "16:9",
    resize_method: str = "Center Crop (Fill)",
) -> torch.Tensor:
    """Apply Deno RTX Video Super Resolution semantics directly to IMAGE frames."""
    if not torch.cuda.is_available():
        raise RuntimeError("IAMCCS RTX upscale requires CUDA, but this ComfyUI Python does not see CUDA.")
    if not torch.is_tensor(images):
        images = torch.as_tensor(images)
    if images.ndim == 5 and images.shape[0] == 1:
        images = images[0]
    if images.ndim != 4:
        raise ValueError(f"IAMCCS RTX upscale expects IMAGE [batch,height,width,channels], got {tuple(images.shape)}")
    batch, source_height, source_width, channels = images.shape
    if int(channels) < 3:
        raise ValueError("IAMCCS RTX upscale requires RGB frames with at least 3 channels.")

    mode = mode if mode in RTX_QUALITY_LEVELS else "VSR Medium"
    resize_type = resize_type if resize_type in RTX_RESIZE_TYPES else "Keep Ratio"
    resize_method = resize_method if resize_method in RTX_RESIZE_METHODS else "Center Crop (Fill)"
    ratio_preset = ratio_preset if ratio_preset in RTX_COMMON_RATIOS else "16:9"
    alignment = _safe_divisible_by(divisible_by)
    target_width, target_height = _target_size(
        int(source_width), int(source_height), mode, resize_type, float(scale), float(megapixels), int(width), int(height), alignment, ratio_preset
    )
    VideoSuperRes = _import_video_super_res()
    quality = getattr(VideoSuperRes.QualityLevel, _quality_attr(mode))
    device_index = _safe_cuda_device_index(device)
    cuda_device = torch.device(f"cuda:{device_index}")
    out_device = images.device
    out_dtype = images.dtype
    output = torch.empty((int(batch), int(target_height), int(target_width), 3), device=out_device, dtype=out_dtype)

    with torch.inference_mode():
        try:
            effect = VideoSuperRes(quality=quality, device=device_index)
        except Exception as exc:
            gpu_name = torch.cuda.get_device_name(device_index) if torch.cuda.is_available() else f"CUDA device {device_index}"
            raise RuntimeError(
                f"IAMCCS RTX Video Super Resolution could not be created for {gpu_name} (device {device_index}), mode {mode}. "
                "Check the NVIDIA RTX GPU/driver and restart ComfyUI if another native RTX runtime was loaded. "
                f"{_runtime_status()} Original error: {type(exc).__name__}: {exc}"
            ) from exc
        with effect:
            effect.output_width = int(target_width)
            effect.output_height = int(target_height)
            effect.load()
            for index in range(int(batch)):
                frame = images[index, :, :, :3].to(device=cuda_device, dtype=torch.float32).permute(2, 0, 1).contiguous()
                if not _same_size_only(mode):
                    frame = _fit_frame_to_target_aspect(frame, int(target_width), int(target_height), resize_method)
                result = effect.run(frame)
                enhanced = torch.from_dlpack(result.image).clone().permute(1, 2, 0).contiguous()
                output[index].copy_(enhanced.clamp(0.0, 1.0).to(device=out_device, dtype=out_dtype))
                del frame, enhanced
    return output
