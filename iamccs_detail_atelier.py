from __future__ import annotations

import json
import math
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path


VRAM_PRESETS = ("8GB", "12GB", "16GB", "24GB")
QUALITY_PRESETS = ("preview", "balanced", "quality")
DETAIL_ATELIER_LINX_TYPE = "IAMCCS_DETAIL_ATELIER_LINX"
HARDWARE_MODES = ("auto", "manual")


@dataclass(frozen=True)
class DetailAtelierPreset:
    target_long_edge: int
    temporal_tile_size: int
    temporal_overlap: int
    temporal_overlap_cond_strength: float
    guiding_strength: float
    cond_image_strength: float
    horizontal_tiles: int
    vertical_tiles: int
    spatial_overlap: int
    vae_tile_size: int
    vae_overlap: int
    vae_temporal_size: int
    vae_temporal_overlap: int
    reserved_vram_gb: float
    cleanup_before_decode: bool
    run_final_upscale: bool
    run_interpolation: bool
    final_upscale_mode: str
    notes: str


_PRESETS: dict[str, dict[str, DetailAtelierPreset]] = {
    "8GB": {
        "preview": DetailAtelierPreset(768, 32, 16, 0.45, 0.85, 1.0, 1, 1, 1, 256, 32, 96, 16, 1.5, True, False, False, "off", "Fastest safe pass for 8GB cards."),
        "balanced": DetailAtelierPreset(960, 40, 16, 0.50, 0.90, 1.0, 1, 1, 1, 256, 32, 128, 16, 1.5, True, True, False, "rtx_x2_optional", "8GB default: detail modestly, upscale after."),
        "quality": DetailAtelierPreset(1280, 40, 16, 0.55, 0.95, 1.0, 1, 1, 1, 256, 32, 128, 16, 2.0, True, True, False, "rtx_x2", "Slow but still conservative for 8GB."),
    },
    "12GB": {
        "preview": DetailAtelierPreset(960, 40, 16, 0.45, 0.90, 1.0, 1, 1, 1, 384, 48, 128, 16, 2.0, True, False, False, "off", "Preview for 12GB cards."),
        "balanced": DetailAtelierPreset(1280, 48, 16, 0.50, 1.00, 1.0, 1, 1, 1, 384, 48, 128, 16, 2.0, True, True, False, "rtx_x2", "Recommended 12GB daily preset."),
        "quality": DetailAtelierPreset(1536, 56, 24, 0.55, 1.00, 1.0, 1, 1, 1, 384, 48, 128, 16, 2.5, True, True, True, "rtx_x2", "Final-ish 12GB preset; interpolation is optional."),
    },
    "16GB": {
        "preview": DetailAtelierPreset(1280, 48, 16, 0.45, 0.95, 1.0, 1, 1, 1, 384, 48, 128, 16, 2.5, True, False, False, "off", "Quick check for 16GB cards."),
        "balanced": DetailAtelierPreset(1536, 56, 24, 0.50, 1.00, 1.0, 1, 1, 1, 512, 64, 192, 24, 3.0, False, True, False, "rtx_x2", "Recommended 16GB default."),
        "quality": DetailAtelierPreset(1920, 64, 24, 0.60, 1.00, 1.0, 1, 1, 1, 512, 64, 192, 24, 3.0, False, True, True, "rtx_x2", "High detail before final upscale."),
    },
    "24GB": {
        "preview": DetailAtelierPreset(1536, 56, 24, 0.45, 1.00, 1.0, 1, 1, 1, 512, 64, 192, 24, 3.0, False, False, False, "off", "Fast preview on high VRAM."),
        "balanced": DetailAtelierPreset(1920, 64, 24, 0.55, 1.00, 1.0, 1, 1, 1, 512, 64, 256, 32, 3.5, False, True, False, "rtx_x2", "24GB default for clean final passes."),
        "quality": DetailAtelierPreset(2160, 80, 32, 0.60, 1.00, 1.0, 2, 1, 2, 512, 64, 256, 32, 4.0, False, True, True, "rtx_x2", "Experimental high-end pass; spatial tiling may be slow."),
    },
}


def _round_to_multiple(value: float, multiple: int, mode: str = "nearest") -> int:
    value = max(float(multiple), float(value))
    if mode == "down":
        rounded = math.floor(value / multiple) * multiple
    elif mode == "up":
        rounded = math.ceil(value / multiple) * multiple
    else:
        rounded = round(value / multiple) * multiple
    return max(multiple, int(rounded))


def _scaled_dims(width: int, height: int, long_edge: int, multiple: int) -> tuple[int, int]:
    width = int(width)
    height = int(height)
    long_edge = int(long_edge)
    if width <= 0 or height <= 0:
        return (0, 0)
    scale = float(long_edge) / float(max(width, height))
    out_w = _round_to_multiple(width * scale, multiple)
    out_h = _round_to_multiple(height * scale, multiple)
    return (out_w, out_h)


def _preset_for(vram_preset: str, quality_mode: str) -> tuple[str, str, DetailAtelierPreset]:
    vram = str(vram_preset if vram_preset in _PRESETS else "12GB")
    quality = str(quality_mode if quality_mode in _PRESETS[vram] else "balanced")
    return vram, quality, _PRESETS[vram][quality]


def _safe_hardware_info() -> dict:
    info = {
        "cuda_available": False,
        "cuda_device_name": None,
        "cuda_total_vram_gb": None,
        "system_ram_gb": None,
        "warnings": [],
    }

    try:
        import torch

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            prop = torch.cuda.get_device_properties(idx)
            total = getattr(prop, "total_memory", None)
            info["cuda_available"] = True
            info["cuda_device_name"] = getattr(prop, "name", None)
            if total:
                info["cuda_total_vram_gb"] = float(total) / (1024.0**3)
    except Exception as e:
        info["warnings"].append(f"cuda_probe_failed={e!r}")

    try:
        import psutil  # type: ignore

        info["system_ram_gb"] = float(psutil.virtual_memory().total) / (1024.0**3)
    except Exception as e:
        info["warnings"].append(f"ram_probe_failed={e!r}")

    return info


def _vram_preset_from_gb(vram_gb: float | None, fallback: str) -> str:
    if vram_gb is None:
        return fallback if fallback in VRAM_PRESETS else "12GB"
    if vram_gb <= 8.5:
        return "8GB"
    if vram_gb <= 12.5:
        return "12GB"
    if vram_gb <= 16.5:
        return "16GB"
    return "24GB"


def _latent_shape_info(latents, vae=None) -> dict:
    out = {
        "latent_frames": None,
        "latent_height": None,
        "latent_width": None,
        "time_scale_factor": 8,
        "estimated_frames": None,
    }
    try:
        samples = latents.get("samples") if isinstance(latents, dict) else None
        shape = getattr(samples, "shape", None)
        if shape is not None and len(shape) >= 5:
            out["latent_frames"] = int(shape[2])
            out["latent_height"] = int(shape[3])
            out["latent_width"] = int(shape[4])
    except Exception:
        pass

    try:
        formula = getattr(vae, "downscale_index_formula", None)
        if formula and len(formula) >= 1:
            out["time_scale_factor"] = max(1, int(formula[0]))
    except Exception:
        pass

    if out["latent_frames"] is not None:
        out["estimated_frames"] = int(out["latent_frames"]) * int(out["time_scale_factor"])
    return out


def _estimate_temporal_chunks(latent_frames: int | None, temporal_tile_size: int, temporal_overlap: int, time_scale_factor: int) -> dict:
    if not latent_frames or latent_frames <= 0:
        return {"chunks": None, "latent_tile": None, "latent_overlap": None, "latent_step": None}
    scale = max(1, int(time_scale_factor or 8))
    latent_tile = max(1, int(temporal_tile_size) // scale)
    latent_overlap = max(0, int(temporal_overlap) // scale)
    if latent_overlap >= latent_tile:
        latent_overlap = max(0, latent_tile - 1)
    latent_step = max(1, latent_tile - latent_overlap)
    if int(latent_frames) <= latent_tile:
        chunks = 1
    else:
        chunks = int(math.ceil((int(latent_frames) - latent_tile) / float(latent_step))) + 1
    return {
        "chunks": int(chunks),
        "latent_tile": int(latent_tile),
        "latent_overlap": int(latent_overlap),
        "latent_step": int(latent_step),
    }


def _auto_adjust_loop_values(
    *,
    selected_vram: str,
    quality: str,
    temporal_tile_size: int,
    temporal_overlap: int,
    horizontal_tiles: int,
    vertical_tiles: int,
    spatial_overlap: int,
    prefer_low_ram: bool,
    hardware: dict,
    latent_info: dict,
) -> tuple[str, dict, list[str]]:
    decisions: list[str] = []
    detected_vram = _vram_preset_from_gb(hardware.get("cuda_total_vram_gb"), selected_vram)
    effective_vram = detected_vram
    if detected_vram != selected_vram:
        decisions.append(f"vram_preset:auto {selected_vram}->{detected_vram}")

    ram_gb = hardware.get("system_ram_gb")
    low_ram = bool(prefer_low_ram)
    if isinstance(ram_gb, (int, float)) and ram_gb <= 40.0 and effective_vram in {"8GB", "12GB"}:
        low_ram = True
        decisions.append("low_ram:auto_enabled")

    max_tile_by_vram = {
        "8GB": 40,
        "12GB": 48,
        "16GB": 64,
        "24GB": 80,
    }
    min_tile_by_vram = {
        "8GB": 32,
        "12GB": 40,
        "16GB": 48,
        "24GB": 56,
    }
    if low_ram:
        max_tile_by_vram["8GB"] = 32
        max_tile_by_vram["12GB"] = 48

    max_tile = max_tile_by_vram.get(effective_vram, 48)
    min_tile = min_tile_by_vram.get(effective_vram, 40)
    if int(temporal_tile_size) > max_tile:
        decisions.append(f"temporal_tile_size:cap {temporal_tile_size}->{max_tile}")
        temporal_tile_size = max_tile
    if int(temporal_tile_size) < min_tile and quality != "preview":
        decisions.append(f"temporal_tile_size:floor {temporal_tile_size}->{min_tile}")
        temporal_tile_size = min_tile

    max_overlap = 16 if effective_vram in {"8GB", "12GB"} else 24
    if int(temporal_overlap) > max_overlap:
        decisions.append(f"temporal_overlap:cap {temporal_overlap}->{max_overlap}")
        temporal_overlap = max_overlap
    if int(temporal_overlap) >= int(temporal_tile_size):
        new_overlap = max(0, int(temporal_tile_size) // 2)
        decisions.append(f"temporal_overlap:repair {temporal_overlap}->{new_overlap}")
        temporal_overlap = new_overlap

    latent_area = None
    if latent_info.get("latent_width") and latent_info.get("latent_height"):
        latent_area = int(latent_info["latent_width"]) * int(latent_info["latent_height"])
    if low_ram or effective_vram in {"8GB", "12GB"}:
        if horizontal_tiles != 1 or vertical_tiles != 1 or spatial_overlap != 1:
            decisions.append("spatial_tiling:low_ram_force_1x1")
        horizontal_tiles = 1
        vertical_tiles = 1
        spatial_overlap = 1
    elif latent_area is not None and latent_area > 4096 and effective_vram == "16GB":
        horizontal_tiles = max(1, int(horizontal_tiles))
        vertical_tiles = max(1, int(vertical_tiles))
        spatial_overlap = max(1, int(spatial_overlap))

    chunk_info = _estimate_temporal_chunks(
        latent_info.get("latent_frames"),
        int(temporal_tile_size),
        int(temporal_overlap),
        int(latent_info.get("time_scale_factor") or 8),
    )
    if chunk_info.get("chunks") and chunk_info["chunks"] > 12 and effective_vram in {"8GB", "12GB"}:
        decisions.append(f"long_clip:estimated_chunks={chunk_info['chunks']}")

    values = {
        "temporal_tile_size": int(temporal_tile_size),
        "temporal_overlap": int(temporal_overlap),
        "horizontal_tiles": int(horizontal_tiles),
        "vertical_tiles": int(vertical_tiles),
        "spatial_overlap": int(spatial_overlap),
        "low_ram_effective": bool(low_ram),
        "chunk_info": chunk_info,
    }
    return effective_vram, values, decisions


def _load_ltxv_looping_sampler_class():
    module_name = "_iamccs_ltxvideo_runtime.looping_sampler"
    cached = sys.modules.get(module_name)
    if cached is not None and hasattr(cached, "LTXVLoopingSampler"):
        return cached.LTXVLoopingSampler

    custom_nodes_dir = Path(__file__).resolve().parent.parent
    ltx_dir = custom_nodes_dir / "ComfyUI-LTXVideo"
    looping_path = ltx_dir / "looping_sampler.py"
    if not looping_path.exists():
        raise ImportError(f"ComfyUI-LTXVideo looping_sampler.py not found at {looping_path}")

    package_name = "_iamccs_ltxvideo_runtime"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(ltx_dir)]
        package.__file__ = str(ltx_dir / "__init__.py")
        package.__package__ = package_name
        sys.modules[package_name] = package

    spec = importlib.util.spec_from_file_location(module_name, looping_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load ComfyUI-LTXVideo looping sampler from {looping_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.LTXVLoopingSampler


class IAMCCS_DetailAtelier:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vram_preset": (VRAM_PRESETS, {"default": "12GB"}),
                "quality_mode": (QUALITY_PRESETS, {"default": "balanced"}),
                "source_width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
                "source_height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
                "target_override_long_edge": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "dimension_multiple": ("INT", {"default": 16, "min": 8, "max": 128, "step": 8}),
                "prefer_low_ram": ("BOOLEAN", {"default": True}),
                "allow_interpolation": ("BOOLEAN", {"default": True}),
                "allow_final_upscale": ("BOOLEAN", {"default": True}),
                "pretty_json": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = (
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "FLOAT",
        "FLOAT",
        "FLOAT",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "FLOAT",
        "BOOLEAN",
        "BOOLEAN",
        "BOOLEAN",
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "target_long_edge",
        "target_width",
        "target_height",
        "temporal_tile_size",
        "temporal_overlap",
        "temporal_overlap_cond_strength",
        "guiding_strength",
        "cond_image_strength",
        "horizontal_tiles",
        "vertical_tiles",
        "spatial_overlap",
        "vae_tile_size",
        "vae_overlap",
        "vae_temporal_size",
        "vae_temporal_overlap",
        "reserved_vram_gb",
        "cleanup_before_decode",
        "run_final_upscale",
        "run_interpolation",
        "final_upscale_mode",
        "report_json",
        "report",
    )
    FUNCTION = "plan"
    CATEGORY = "IAMCCS/LTX-2/Detail Atelier"

    def plan(
        self,
        vram_preset: str,
        quality_mode: str,
        source_width: int,
        source_height: int,
        target_override_long_edge: int,
        dimension_multiple: int,
        prefer_low_ram: bool,
        allow_interpolation: bool,
        allow_final_upscale: bool,
        pretty_json: bool,
    ):
        vram = str(vram_preset if vram_preset in _PRESETS else "12GB")
        quality = str(quality_mode if quality_mode in _PRESETS[vram] else "balanced")
        preset = _PRESETS[vram][quality]

        target_long_edge = int(target_override_long_edge) if int(target_override_long_edge) > 0 else preset.target_long_edge
        target_width, target_height = _scaled_dims(
            int(source_width),
            int(source_height),
            target_long_edge,
            max(8, int(dimension_multiple)),
        )

        vae_tile_size = preset.vae_tile_size
        vae_overlap = preset.vae_overlap
        vae_temporal_size = preset.vae_temporal_size
        vae_temporal_overlap = preset.vae_temporal_overlap
        cleanup_before_decode = bool(preset.cleanup_before_decode)

        if bool(prefer_low_ram):
            if vram == "8GB":
                vae_tile_size = min(vae_tile_size, 256)
                vae_overlap = min(vae_overlap, 32)
                cleanup_before_decode = True
            elif vram == "12GB":
                vae_tile_size = min(vae_tile_size, 384)
                vae_overlap = min(vae_overlap, 48)
                cleanup_before_decode = True

        run_final_upscale = bool(allow_final_upscale and preset.run_final_upscale)
        run_interpolation = bool(allow_interpolation and preset.run_interpolation)
        final_upscale_mode = preset.final_upscale_mode if run_final_upscale else "off"

        data = {
            "node": "IAMCCS_DetailAtelier",
            "vram_preset": vram,
            "quality_mode": quality,
            "source": {
                "width": int(source_width),
                "height": int(source_height),
            },
            "target": {
                "long_edge": int(target_long_edge),
                "width": int(target_width),
                "height": int(target_height),
                "dimension_multiple": int(dimension_multiple),
            },
            "looper": {
                "temporal_tile_size": preset.temporal_tile_size,
                "temporal_overlap": preset.temporal_overlap,
                "temporal_overlap_cond_strength": preset.temporal_overlap_cond_strength,
                "guiding_strength": preset.guiding_strength,
                "cond_image_strength": preset.cond_image_strength,
                "horizontal_tiles": preset.horizontal_tiles,
                "vertical_tiles": preset.vertical_tiles,
                "spatial_overlap": preset.spatial_overlap,
            },
            "decode": {
                "vae_tile_size": int(vae_tile_size),
                "vae_overlap": int(vae_overlap),
                "vae_temporal_size": int(vae_temporal_size),
                "vae_temporal_overlap": int(vae_temporal_overlap),
                "cleanup_before_decode": bool(cleanup_before_decode),
            },
            "post": {
                "reserved_vram_gb": preset.reserved_vram_gb,
                "run_final_upscale": run_final_upscale,
                "run_interpolation": run_interpolation,
                "final_upscale_mode": final_upscale_mode,
            },
            "strategy": "detail first, upscale after",
            "notes": preset.notes,
        }
        report_json = json.dumps(data, ensure_ascii=False, indent=2 if bool(pretty_json) else None)
        report = (
            f"{vram} {quality}: target_edge={target_long_edge}"
            f"{f' -> {target_width}x{target_height}' if target_width and target_height else ''}, "
            f"looper={preset.temporal_tile_size}/{preset.temporal_overlap}, "
            f"decode={vae_tile_size}/{vae_overlap}/{vae_temporal_size}/{vae_temporal_overlap}, "
            f"cleanup={cleanup_before_decode}, upscale={final_upscale_mode}, rife={run_interpolation}. "
            "Strategy: detail first, upscale after."
        )

        return (
            int(target_long_edge),
            int(target_width),
            int(target_height),
            int(preset.temporal_tile_size),
            int(preset.temporal_overlap),
            float(preset.temporal_overlap_cond_strength),
            float(preset.guiding_strength),
            float(preset.cond_image_strength),
            int(preset.horizontal_tiles),
            int(preset.vertical_tiles),
            int(preset.spatial_overlap),
            int(vae_tile_size),
            int(vae_overlap),
            int(vae_temporal_size),
            int(vae_temporal_overlap),
            float(preset.reserved_vram_gb),
            bool(cleanup_before_decode),
            bool(run_final_upscale),
            bool(run_interpolation),
            final_upscale_mode,
            report_json,
            report,
        )


class IAMCCS_DetailAtelierSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Diffusion model used for the detail pass."}),
                "vae": ("VAE", {"tooltip": "Video VAE used by LTXVideo."}),
                "noise": ("NOISE", {"tooltip": "Noise object; the seed is reused internally."}),
                "sampler": ("SAMPLER", {"tooltip": "Sampler selected for the detail pass."}),
                "sigmas": ("SIGMAS", {"tooltip": "Sigma schedule for the detail pass."}),
                "guider": ("GUIDER", {"tooltip": "CFG/STG guider for the detail pass."}),
                "latents": ("LATENT", {"tooltip": "Input video latents to detail."}),
                "vram_preset": (VRAM_PRESETS, {"default": "12GB"}),
                "quality_mode": (QUALITY_PRESETS, {"default": "balanced"}),
                "prefer_low_ram": ("BOOLEAN", {"default": True}),
                "allow_spatial_tiling": ("BOOLEAN", {"default": True}),
                "adain_factor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "guiding_start_step": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "guiding_end_step": ("INT", {"default": 1000, "min": 0, "max": 1000}),
                "optional_cond_image_indices": ("STRING", {"default": "0"}),
                "per_tile_seed_offsets": ("STRING", {"default": "0"}),
                "pretty_json": ("BOOLEAN", {"default": True}),
                "hardware_mode": (HARDWARE_MODES, {"default": "auto"}),
                "oom_retry": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "atelier_advanced": (DETAIL_ATELIER_LINX_TYPE, {}),
                "optional_cond_images": ("IMAGE", {}),
                "optional_guiding_latents": ("LATENT", {}),
                "optional_positive_conditionings": ("CONDITIONING", {}),
                "optional_negative_index_latents": ("LATENT", {}),
                "optional_normalizing_latents": ("LATENT", {}),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING", "STRING")
    RETURN_NAMES = ("denoised_output", "report_json", "report")
    FUNCTION = "detail"
    CATEGORY = "IAMCCS/LTX-2/Detail Atelier"

    def detail(
        self,
        model,
        vae,
        noise,
        sampler,
        sigmas,
        guider,
        latents,
        vram_preset: str,
        quality_mode: str,
        prefer_low_ram: bool,
        allow_spatial_tiling: bool,
        adain_factor: float,
        guiding_start_step: int,
        guiding_end_step: int,
        optional_cond_image_indices: str,
        per_tile_seed_offsets: str,
        pretty_json: bool,
        hardware_mode: str = "auto",
        oom_retry: bool = True,
        atelier_advanced=None,
        optional_cond_images=None,
        optional_guiding_latents=None,
        optional_positive_conditionings=None,
        optional_negative_index_latents=None,
        optional_normalizing_latents=None,
    ):
        requested_vram, quality, requested_preset = _preset_for(vram_preset, quality_mode)
        hardware = _safe_hardware_info()
        if str(hardware_mode or "auto") == "auto":
            vram = _vram_preset_from_gb(hardware.get("cuda_total_vram_gb"), requested_vram)
            _, _, preset = _preset_for(vram, quality)
        else:
            vram, _, preset = requested_vram, quality, requested_preset

        advanced = atelier_advanced if isinstance(atelier_advanced, dict) and atelier_advanced.get("enabled", True) else {}
        latent_info = _latent_shape_info(latents, vae)

        def adv_int(name: str, default: int) -> int:
            try:
                value = int(advanced.get(name, 0))
            except Exception:
                value = 0
            return int(value if value > 0 else default)

        def adv_float(name: str, default: float) -> float:
            try:
                value = float(advanced.get(name, -1.0))
            except Exception:
                value = -1.0
            return float(value if value >= 0.0 else default)

        temporal_tile_size = adv_int("temporal_tile_size", preset.temporal_tile_size)
        temporal_overlap = adv_int("temporal_overlap", preset.temporal_overlap)
        guiding_strength = adv_float("guiding_strength", preset.guiding_strength)
        temporal_overlap_cond_strength = adv_float(
            "temporal_overlap_cond_strength",
            preset.temporal_overlap_cond_strength,
        )
        cond_image_strength = adv_float("cond_image_strength", preset.cond_image_strength)

        horizontal_tiles = adv_int("horizontal_tiles", preset.horizontal_tiles)
        vertical_tiles = adv_int("vertical_tiles", preset.vertical_tiles)
        spatial_overlap = adv_int("spatial_overlap", preset.spatial_overlap)
        if not bool(allow_spatial_tiling):
            horizontal_tiles = 1
            vertical_tiles = 1
            spatial_overlap = 1
        spatial_mode = str(advanced.get("spatial_tiling_mode", "inherit") or "inherit")
        if spatial_mode == "force_off":
            horizontal_tiles = 1
            vertical_tiles = 1
            spatial_overlap = 1
        if bool(prefer_low_ram) and vram in {"8GB", "12GB"}:
            horizontal_tiles = 1
            vertical_tiles = 1
            spatial_overlap = min(spatial_overlap, 1)

        auto_decisions: list[str] = []
        auto_values = {
            "chunk_info": _estimate_temporal_chunks(
                latent_info.get("latent_frames"),
                int(temporal_tile_size),
                int(temporal_overlap),
                int(latent_info.get("time_scale_factor") or 8),
            )
        }
        if str(hardware_mode or "auto") == "auto":
            vram, auto_values, auto_decisions = _auto_adjust_loop_values(
                selected_vram=requested_vram,
                quality=quality,
                temporal_tile_size=int(temporal_tile_size),
                temporal_overlap=int(temporal_overlap),
                horizontal_tiles=int(horizontal_tiles),
                vertical_tiles=int(vertical_tiles),
                spatial_overlap=int(spatial_overlap),
                prefer_low_ram=bool(prefer_low_ram),
                hardware=hardware,
                latent_info=latent_info,
            )
            temporal_tile_size = int(auto_values["temporal_tile_size"])
            temporal_overlap = int(auto_values["temporal_overlap"])
            horizontal_tiles = int(auto_values["horizontal_tiles"])
            vertical_tiles = int(auto_values["vertical_tiles"])
            spatial_overlap = int(auto_values["spatial_overlap"])

        adain = adv_float("adain_factor", float(adain_factor))
        guide_start = adv_int("guiding_start_step", int(guiding_start_step))
        guide_end = adv_int("guiding_end_step", int(guiding_end_step))
        cond_indices = str(advanced.get("optional_cond_image_indices") or optional_cond_image_indices or "0")
        seed_offsets = str(advanced.get("per_tile_seed_offsets") or per_tile_seed_offsets or "0")

        looping_sampler_cls = _load_ltxv_looping_sampler_class()
        looping_sampler = looping_sampler_cls()
        retry_applied = False

        def run_loop(ts: int, ov: int, ht: int, vt: int, so: int):
            return looping_sampler.sample(
                model=model,
                vae=vae,
                noise=noise,
                sampler=sampler,
                sigmas=sigmas,
                guider=guider,
                latents=latents,
                guiding_strength=float(guiding_strength),
                adain_factor=float(adain),
                temporal_tile_size=int(ts),
                temporal_overlap=int(ov),
                temporal_overlap_cond_strength=float(temporal_overlap_cond_strength),
                horizontal_tiles=int(ht),
                vertical_tiles=int(vt),
                spatial_overlap=int(so),
                optional_cond_images=optional_cond_images,
                cond_image_strength=float(cond_image_strength),
                optional_guiding_latents=optional_guiding_latents,
                optional_negative_index_latents=optional_negative_index_latents,
                optional_negative_index_strength=1.0,
                optional_positive_conditionings=optional_positive_conditionings,
                guiding_start_step=int(guide_start),
                guiding_end_step=int(guide_end),
                optional_cond_image_indices=str(cond_indices),
                optional_normalizing_latents=optional_normalizing_latents,
                per_tile_seed_offsets=str(seed_offsets),
            )

        try:
            result = run_loop(temporal_tile_size, temporal_overlap, horizontal_tiles, vertical_tiles, spatial_overlap)
        except RuntimeError as e:
            is_oom = "out of memory" in str(e).lower() or "cuda oom" in str(e).lower()
            if not (bool(oom_retry) and str(hardware_mode or "auto") == "auto" and is_oom):
                raise
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
            except Exception:
                pass
            retry_applied = True
            old_ts, old_ov = int(temporal_tile_size), int(temporal_overlap)
            temporal_tile_size = max(32, int(temporal_tile_size) - 16)
            temporal_overlap = min(16, max(8, int(temporal_tile_size) // 3))
            horizontal_tiles = 1
            vertical_tiles = 1
            spatial_overlap = 1
            auto_decisions.append(f"oom_retry:{old_ts}/{old_ov}-> {temporal_tile_size}/{temporal_overlap}, tiles=1x1")
            auto_values["chunk_info"] = _estimate_temporal_chunks(
                latent_info.get("latent_frames"),
                int(temporal_tile_size),
                int(temporal_overlap),
                int(latent_info.get("time_scale_factor") or 8),
            )
            result = run_loop(temporal_tile_size, temporal_overlap, horizontal_tiles, vertical_tiles, spatial_overlap)

        denoised = result[0] if isinstance(result, tuple) else result

        data = {
            "node": "IAMCCS_DetailAtelierSampler",
            "display": "IAMCCS Detail Atelier",
            "hardware_mode": str(hardware_mode or "auto"),
            "requested_vram_preset": requested_vram,
            "effective_vram_preset": vram,
            "quality_mode": quality,
            "strategy": "orchestrated internal LTXV loop; detail only, no decode/upscale unification",
            "advanced_linx_enabled": bool(advanced),
            "hardware": hardware,
            "latent": latent_info,
            "estimated_chunks": auto_values.get("chunk_info"),
            "auto_decisions": auto_decisions,
            "oom_retry_applied": bool(retry_applied),
            "looper": {
                "temporal_tile_size": int(temporal_tile_size),
                "temporal_overlap": int(temporal_overlap),
                "guiding_strength": float(guiding_strength),
                "temporal_overlap_cond_strength": float(temporal_overlap_cond_strength),
                "cond_image_strength": float(cond_image_strength),
                "horizontal_tiles": int(horizontal_tiles),
                "vertical_tiles": int(vertical_tiles),
                "spatial_overlap": int(spatial_overlap),
                "adain_factor": float(adain),
                "guiding_start_step": int(guide_start),
                "guiding_end_step": int(guide_end),
                "optional_cond_image_indices": str(cond_indices),
                "per_tile_seed_offsets": str(seed_offsets),
            },
            "notes": preset.notes,
        }
        report_json = json.dumps(data, ensure_ascii=False, indent=2 if bool(pretty_json) else None)
        chunks = auto_values.get("chunk_info", {}).get("chunks") if isinstance(auto_values.get("chunk_info"), dict) else None
        report = (
            f"IAMCCS Detail Atelier {vram}/{quality} ({hardware_mode}): internal looper "
            f"{temporal_tile_size}/{temporal_overlap}, "
            f"tiles={horizontal_tiles}x{vertical_tiles}, overlap={spatial_overlap}"
            f"{f', chunks≈{chunks}' if chunks else ''}."
        )
        return (denoised, report_json, report)


class IAMCCS_DetailAtelierAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "temporal_tile_size": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 8}),
                "temporal_overlap": ("INT", {"default": 0, "min": 0, "max": 80, "step": 8}),
                "guiding_strength": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "temporal_overlap_cond_strength": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "cond_image_strength": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "horizontal_tiles": ("INT", {"default": 0, "min": 0, "max": 6}),
                "vertical_tiles": ("INT", {"default": 0, "min": 0, "max": 6}),
                "spatial_overlap": ("INT", {"default": 0, "min": 0, "max": 8}),
                "spatial_tiling_mode": (["inherit", "force_off"], {"default": "inherit"}),
                "adain_factor": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "guiding_start_step": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "guiding_end_step": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "optional_cond_image_indices": ("STRING", {"default": ""}),
                "per_tile_seed_offsets": ("STRING", {"default": ""}),
                "pretty_json": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = (DETAIL_ATELIER_LINX_TYPE, "STRING", "STRING")
    RETURN_NAMES = ("atelier_advanced", "report_json", "report")
    FUNCTION = "build"
    CATEGORY = "IAMCCS/LTX-2/Detail Atelier"

    def build(
        self,
        enabled: bool,
        temporal_tile_size: int,
        temporal_overlap: int,
        guiding_strength: float,
        temporal_overlap_cond_strength: float,
        cond_image_strength: float,
        horizontal_tiles: int,
        vertical_tiles: int,
        spatial_overlap: int,
        spatial_tiling_mode: str,
        adain_factor: float,
        guiding_start_step: int,
        guiding_end_step: int,
        optional_cond_image_indices: str,
        per_tile_seed_offsets: str,
        pretty_json: bool,
    ):
        cfg = {
            "enabled": bool(enabled),
            "temporal_tile_size": int(temporal_tile_size),
            "temporal_overlap": int(temporal_overlap),
            "guiding_strength": float(guiding_strength),
            "temporal_overlap_cond_strength": float(temporal_overlap_cond_strength),
            "cond_image_strength": float(cond_image_strength),
            "horizontal_tiles": int(horizontal_tiles),
            "vertical_tiles": int(vertical_tiles),
            "spatial_overlap": int(spatial_overlap),
            "spatial_tiling_mode": str(spatial_tiling_mode or "inherit"),
            "adain_factor": float(adain_factor),
            "guiding_start_step": int(guiding_start_step),
            "guiding_end_step": int(guiding_end_step),
            "optional_cond_image_indices": str(optional_cond_image_indices or ""),
            "per_tile_seed_offsets": str(per_tile_seed_offsets or ""),
        }
        report_json = json.dumps(cfg, ensure_ascii=False, indent=2 if bool(pretty_json) else None)
        active = [k for k, v in cfg.items() if k != "enabled" and v not in (0, -1.0, "", "inherit")]
        report = "Detail Atelier Advanced: " + ("disabled" if not enabled else ("overrides " + ", ".join(active) if active else "enabled, no overrides"))
        return (cfg, report_json, report)
