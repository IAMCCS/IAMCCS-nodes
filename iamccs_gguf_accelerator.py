from __future__ import annotations

import logging
from typing import Any, Tuple

import torch


_log = logging.getLogger("IAMCCS.GGUF.Accelerator")


def _move_to_device_recursive(obj: Any, device: torch.device) -> Tuple[Any, int, int]:
    """Recursively move torch.Tensor objects inside obj to device.

    Returns (new_obj, tensor_count_moved, bytes_moved).
    """
    if isinstance(obj, torch.Tensor):
        try:
            # If already on target device, do nothing.
            if obj.device == device:
                return obj, 0, 0
            moved = obj.to(device, non_blocking=True)
            bytes_moved = moved.element_size() * moved.numel()
            return moved, 1, bytes_moved
        except Exception:
            # If move fails (rare), keep original.
            return obj, 0, 0

    if isinstance(obj, tuple):
        out_items: list[Any] = []
        moved_count = 0
        moved_bytes = 0
        changed = False
        for item in obj:
            new_item, c, b = _move_to_device_recursive(item, device)
            out_items.append(new_item)
            moved_count += c
            moved_bytes += b
            changed = changed or (new_item is not item)
        return (tuple(out_items) if changed else obj), moved_count, moved_bytes

    if isinstance(obj, list):
        out_list: list[Any] = []
        moved_count = 0
        moved_bytes = 0
        changed = False
        for item in obj:
            new_item, c, b = _move_to_device_recursive(item, device)
            out_list.append(new_item)
            moved_count += c
            moved_bytes += b
            changed = changed or (new_item is not item)
        return (out_list if changed else obj), moved_count, moved_bytes

    if isinstance(obj, dict):
        out_dict: dict[Any, Any] = {}
        moved_count = 0
        moved_bytes = 0
        changed = False
        for k, v in obj.items():
            new_v, c, b = _move_to_device_recursive(v, device)
            out_dict[k] = new_v
            moved_count += c
            moved_bytes += b
            changed = changed or (new_v is not v)
        return (out_dict if changed else obj), moved_count, moved_bytes

    return obj, 0, 0


def _normalize_device(value: Any, fallback: torch.device) -> torch.device:
    if value is None:
        return fallback
    try:
        return torch.device(value)
    except Exception:
        return fallback


def _cuda_device_index(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    if device.index is not None:
        return int(device.index)
    try:
        return int(torch.cuda.current_device())
    except Exception:
        return 0


def _cuda_mem_info_mb(device: torch.device) -> tuple[float, float] | None:
    if not torch.cuda.is_available() or device.type != "cuda":
        return None
    try:
        idx = _cuda_device_index(device)
        if idx is None:
            return None
        free_b, total_b = torch.cuda.mem_get_info(idx)
        return (float(free_b) / (1024 * 1024), float(total_b) / (1024 * 1024))
    except Exception:
        return None


def _cuda_gc() -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


class IAMCCS_GGUF_accelerator:
    """GGUF accelerator: forces patch_on_device to reduce per-step CPU↔GPU patch movement.

    Intended for GGUF UNet models with LoRA patches where repeatedly moving patch tensors to GPU
    can dominate runtime on low VRAM setups.

    Notes:
    - This node does not change sampling parameters.
    - It can increase VRAM usage depending on how many/large LoRA patches are present.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (["auto_oom_safe", "manual"], {
                    "default": "auto_oom_safe",
                    "tooltip": "auto_oom_safe: tries patch_on_device+eager move, falls back to offload on OOM | manual: use toggles below"
                }),
                "patch_on_device": ("BOOLEAN", {"default": True}),
                "move_patches_now": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If enabled, attempts to pre-move patch tensors to the model load_device to reduce runtime transfers. Can increase VRAM usage."
                }),
                "min_free_vram_mb": ("INT", {
                    "default": 1500,
                    "min": 0,
                    "max": 65536,
                    "step": 64,
                    "tooltip": "(auto_oom_safe) If free VRAM is below this, we disable patch_on_device to reduce OOM risk. 0 disables the check."
                }),
                "oom_fallback": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If a CUDA OOM happens while moving patches, automatically switches to offload and continues."
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "accelerate"
    CATEGORY = "IAMCCS/Optimize"

    def accelerate(self, model, mode: str, patch_on_device: bool, move_patches_now: bool, min_free_vram_mb: int, oom_fallback: bool):
        # Clone to avoid mutating upstream graph state.
        try:
            model_out = model.clone()
        except Exception:
            model_out = model

        applied = []

        mode = str(mode or "auto_oom_safe")

        # Decide devices.
        load_device = _normalize_device(
            getattr(model_out, "load_device", None),
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        offload_device = _normalize_device(getattr(model_out, "offload_device", None), torch.device("cpu"))

        # auto mode chooses patch strategy based on VRAM headroom.
        chosen_patch_on_device = bool(patch_on_device)
        chosen_move_now = bool(move_patches_now)

        if mode == "auto_oom_safe" and load_device.type == "cuda":
            info = _cuda_mem_info_mb(load_device)
            if info is not None and int(min_free_vram_mb) > 0:
                free_mb, total_mb = info
                if free_mb < float(min_free_vram_mb):
                    chosen_patch_on_device = False
                    chosen_move_now = False
                    applied.append(f"auto:disable_patch_on_device(free≈{free_mb:.0f}MiB<min{int(min_free_vram_mb)}MiB)")

        # 1) Primary knob used by ComfyUI-GGUF's GGUFModelPatcher.
        if hasattr(model_out, "patch_on_device"):
            try:
                setattr(model_out, "patch_on_device", bool(chosen_patch_on_device))
                applied.append("model.patch_on_device")
            except Exception:
                pass

        # 2) Defensive: some wrappers might store a nested patcher.
        for attr in ("patcher", "model_patcher", "_patcher"):
            inner = getattr(model_out, attr, None)
            if inner is not None and hasattr(inner, "patch_on_device"):
                try:
                    setattr(inner, "patch_on_device", bool(chosen_patch_on_device))
                    applied.append(f"model.{attr}.patch_on_device")
                except Exception:
                    pass

        # Decide target device for patches.
        target_device = load_device if bool(chosen_patch_on_device) else offload_device

        moved_tensors = 0
        moved_bytes = 0

        def _try_move_patches() -> None:
            nonlocal moved_tensors, moved_bytes
            patches = getattr(model_out, "patches")
            new_patches, moved_tensors, moved_bytes = _move_to_device_recursive(patches, target_device)
            if new_patches is not patches:
                setattr(model_out, "patches", new_patches)

        # If patch_on_device is disabled, try to ensure patches live on offload_device (CPU)
        # to reduce persistent VRAM usage.
        if not bool(chosen_patch_on_device) and hasattr(model_out, "patches"):
            try:
                target_device = offload_device
                _try_move_patches()
                applied.append("model.patches(move_to_offload_device)")
            except Exception:
                pass

        # 3) Optional: eagerly move patch tensors to load_device.
        # This mirrors what GGUF's `move_patch_to_device` would do later, but doing it once
        # can eliminate huge per-layer overhead.
        if bool(chosen_patch_on_device) and bool(chosen_move_now) and hasattr(model_out, "patches"):
            try:
                target_device = load_device
                _cuda_gc()
                _try_move_patches()
                applied.append("model.patches(move_to_load_device)")
            except RuntimeError as e:
                msg = str(e).lower()
                is_oom = ("out of memory" in msg) or ("cuda" in msg and "memory" in msg)
                if bool(oom_fallback) and is_oom:
                    _log.warning("OOM while moving patches to CUDA; falling back to offload. Error: %s", e)
                    _cuda_gc()
                    try:
                        setattr(model_out, "patch_on_device", False)
                        applied.append("fallback:disable_patch_on_device")
                    except Exception:
                        pass
                    # Also attempt to move patches back to CPU/offload_device.
                    try:
                        target_device = offload_device
                        _try_move_patches()
                        applied.append("fallback:move_patches_to_offload_device")
                    except Exception:
                        pass
                else:
                    _log.warning("Failed to move patches to device: %s", e)
            except Exception as e:
                _log.warning("Failed to move patches to device: %s", e)

        mb = moved_bytes / (1024 * 1024) if moved_bytes else 0.0
        mem_info = _cuda_mem_info_mb(load_device) if load_device.type == "cuda" else None
        mem_str = ""
        if mem_info is not None:
            free_mb, total_mb = mem_info
            mem_str = f" | cuda_free≈{free_mb:.0f}/{total_mb:.0f} MiB"
        report = (
            f"mode={mode} | patch_on_device={bool(getattr(model_out, 'patch_on_device', chosen_patch_on_device))} | "
            f"move_patches_now={bool(chosen_move_now)} | load_device={load_device} | offload_device={offload_device}{mem_str} | "
            f"applied={applied or ['(none)']} | moved_tensors={moved_tensors} | moved≈{mb:.1f} MiB"
        )

        return (model_out, report)
