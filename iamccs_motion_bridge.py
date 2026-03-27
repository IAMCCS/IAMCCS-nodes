# iamccs_motion_bridge.py
# ===============================================================
# IAMCCS MotionBridge — Save/Load SVI segment tails for re-patching
#
# Copyright (C) 2025-2026 Carmine Cristallo Scalzi (IAMCCS)
# GNU General Public License v3.0 (GPL-3.0)
# ===============================================================
"""
Two nodes for persisting and reusing SVI segment continuity data:

  IAMCCS_MotionBridgeSave
    · LATENT passthrough  → slices the last N latent frames (tail)
    · IMAGE  passthrough  → extracts the last video frame
    · Saves both into  output/motion_bridges/<name>.safetensors
    · Optionally writes output/motion_bridges/<name>_lastframe.png

  IAMCCS_MotionBridgeLoad
    · Reads a saved bridge file (dropdown of available files)
    · Returns LATENT tail  →  wire to prev_samples of WanImageMotion*
    · Returns IMAGE        →  wire to VAEEncode → anchor_samples
                              (overridable: connect optional image input)

Typical re-patch workflow
─────────────────────────
  Workflow A (original chain):
    KSampler_seg2 ──► LATENT ──► MotionBridgeSave("seg2_bridge") ──► VAEDecode
                                         (saves tail + last_frame)

  Workflow B (re-patch seg2):
    MotionBridgeLoad("seg2_bridge")
    ├── latent ──► prev_samples  ──┐
    └── image  ──► VAEEncode ──────┴──► WanImageMotionPro
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger("IAMCCS.MotionBridge")

# ── output directory ────────────────────────────────────────────────────────

def _bridge_dir() -> Path:
    """Returns (and creates) the motion_bridges output directory."""
    try:
        import folder_paths  # type: ignore[import]
        base = Path(folder_paths.get_output_directory())
    except Exception:
        base = Path("output")
    d = base / "motion_bridges"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _list_bridges() -> list[str]:
    """Returns sorted list of .safetensors bridge file stems."""
    try:
        files = sorted(p.stem for p in _bridge_dir().glob("*.safetensors"))
    except Exception:
        files = []
    return files if files else ["(none)"]


# ── safetensors helpers ──────────────────────────────────────────────────────

def _save_safetensors(path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
    try:
        from safetensors.torch import save_file  # type: ignore[import]
        save_file(tensors, str(path), metadata=metadata)
    except ImportError:
        # Fallback: torch.save (less portable but always available)
        log.warning("[MotionBridge] safetensors not available, falling back to torch.save")
        torch.save({"tensors": tensors, "metadata": metadata}, str(path))


def _load_safetensors(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    try:
        from safetensors.torch import load_file, safe_open  # type: ignore[import]
        tensors = load_file(str(path))
        metadata: dict[str, str] = {}
        with safe_open(str(path), framework="pt") as f:
            metadata = f.metadata() or {}
        return tensors, metadata
    except ImportError:
        data = torch.load(str(path), map_location="cpu", weights_only=False)
        return data["tensors"], data.get("metadata", {})


# ── PNG helpers ──────────────────────────────────────────────────────────────

def _save_png(path: Path, image_tensor: torch.Tensor) -> None:
    """Save a single HWC float32 [0,1] tensor as PNG."""
    try:
        from PIL import Image  # type: ignore[import]
        arr = (image_tensor.cpu().float().clamp(0, 1).numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(str(path))
        log.info("[MotionBridge] Saved preview PNG: %s", path)
    except Exception as e:
        log.warning("[MotionBridge] Could not save PNG preview: %s", e)


def _load_png(path: Path) -> torch.Tensor | None:
    """Load a PNG as float32 HWC tensor [0,1], or None on failure."""
    try:
        from PIL import Image  # type: ignore[import]
        arr = np.array(Image.open(str(path)).convert("RGB")).astype(np.float32) / 255.0
        return torch.from_numpy(arr)
    except Exception as e:
        log.warning("[MotionBridge] Could not load PNG: %s", e)
        return None


# ════════════════════════════════════════════════════════════════════════════
# IAMCCS_MotionBridgeSave
# ════════════════════════════════════════════════════════════════════════════

class IAMCCS_MotionBridgeSave:
    """
    Pure-sink node: saves SVI segment tail (latent + last frame) to disk only.

    No outputs — nothing is held in ComfyUI's executor cache after this node runs.
    The input tensors are moved to CPU before saving and VRAM is flushed
    when flush_vram=True, so downstream KSamplers start with a clean VRAM budget.

    Wire the segment's KSampler latent and VAEDecode images into this node as a
    side-branch.  The downstream segment receives its prev_samples / source_images
    via direct connections bypassing this node entirely.

    Typical wiring (side-branch, no passthrough):
        KSampler_seg2 ──► latent ──► IAMCCS_MotionBridgeSave  (saves to disk)
        VAEDecode_seg2 ──► images ─┘
        KSampler_seg2 ──────────────────────────────────────► WanImageMotionPro.prev_samples
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "images": ("IMAGE",),
                "name": (
                    "STRING",
                    {
                        "default": "segment_bridge",
                        "tooltip": (
                            "Bridge file name (no extension). "
                            "Use descriptive names like 'scene01_seg2_to_seg3'. "
                            "Saved to output/motion_bridges/<name>.safetensors"
                        ),
                    },
                ),
                "tail_frames": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 16,
                        "step": 1,
                        "tooltip": (
                            "How many latent frames to slice from the tail of 'latent'. "
                            "Must match 'motion_latent_count' on WanImageMotion* in the next workflow."
                        ),
                    },
                ),
                "save_preview_png": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Also save the last frame as <name>_lastframe.png alongside the .safetensors. "
                            "Useful for visual inspection and as a fallback LoadImage source."
                        ),
                    },
                ),
                "flush_vram": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "After saving, move input tensors to CPU and call ComfyUI soft_empty_cache "
                            "+ torch.cuda.empty_cache(). Frees CUDA reserved blocks before the next "
                            "KSampler runs. Disable only if you need the tensors to stay on GPU for "
                            "another branch that runs immediately after this node."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "save"
    CATEGORY = "IAMCCS/Wan"
    OUTPUT_NODE = True

    def save(
        self,
        latent: dict,
        images: torch.Tensor,
        name: str,
        tail_frames: int,
        save_preview_png: bool,
        flush_vram: bool = True,
    ):
        samples = latent["samples"]  # [B, C, T, H, W]
        T = samples.shape[2]
        n = int(max(1, min(tail_frames, T)))

        # Always copy to CPU for disk serialisation — never write from GPU memory directly.
        tail = samples[:, :, -n:].clone().cpu()
        # Extract last video frame — images is [B, H, W, C] (ComfyUI convention)
        last_frame = images[-1].clone().cpu().contiguous()  # [H, W, C]

        # Build safetensors payload
        tensors: dict[str, torch.Tensor] = {
            "latent_tail": tail,
            "last_frame": last_frame,
        }
        meta: dict[str, str] = {
            "tail_frames": str(n),
            "B": str(tail.shape[0]),
            "C": str(tail.shape[1]),
            "T_tail": str(tail.shape[2]),
            "H": str(tail.shape[3]),
            "W": str(tail.shape[4]),
            "dtype": str(tail.dtype),
        }

        out_dir = _bridge_dir()
        st_path = out_dir / f"{name}.safetensors"
        _save_safetensors(st_path, tensors, meta)
        log.info(
            "[MotionBridgeSave] Saved '%s' | tail: %s | last_frame: %s",
            st_path.name,
            list(tail.shape),
            list(last_frame.shape),
        )

        if save_preview_png:
            png_path = out_dir / f"{name}_lastframe.png"
            _save_png(png_path, last_frame)

        # Explicit cleanup: delete CPU copies immediately after saving.
        del tail, last_frame, tensors

        if flush_vram:
            # Move the executor's input tensors to CPU so the CUDA allocator
            # can reclaim their VRAM before the next KSampler runs.
            # This is safe: if any other node still holds a reference to these
            # objects they will transparently find the data on CPU.
            if samples.is_cuda:
                latent["samples"] = samples.cpu()
            del samples
            if images.is_cuda:
                # images is passed by value (tensor), we can't mutate the caller's
                # ref directly — but we can trigger CUDA release via del + cache flush.
                del images

            try:
                import comfy.model_management as mm
                mm.soft_empty_cache()
            except Exception:
                pass
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            log.info("[MotionBridgeSave] VRAM flush complete.")

        # Pure sink — no outputs held in ComfyUI executor cache.
        return ()


# ════════════════════════════════════════════════════════════════════════════
# IAMCCS_MotionBridgeLoad
# ════════════════════════════════════════════════════════════════════════════

class IAMCCS_MotionBridgeLoad:
    """
    Loads a saved SVI segment bridge from disk.

    Outputs:
        latent  → wire to prev_samples of WanImageMotion / WanImageMotionPro
        image   → wire to VAEEncode → anchor_samples
                  (overridden by the optional 'image_override' input if connected)

    The optional 'image_override' input lets you use an external PNG or
    any other IMAGE node instead of the saved last_frame — useful when you
    want to manually edit the junction frame before re-patching.
    """

    @classmethod
    def INPUT_TYPES(cls):
        bridges = _list_bridges()
        return {
            "required": {
                "bridge_name": (
                    bridges,
                    {
                        "tooltip": (
                            "Bridge file to load (from output/motion_bridges/). "
                            "Refresh the ComfyUI page to see newly saved files."
                        ),
                    },
                ),
            },
            "optional": {
                "image_override": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "If connected, this image is used instead of the saved last_frame. "
                            "Useful when you want to manually edit the junction frame "
                            "(e.g. inpaint it) before feeding it to VAEEncode → anchor_samples."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "INT")
    RETURN_NAMES = ("latent", "image", "tail_frames")
    FUNCTION = "load"
    CATEGORY = "IAMCCS/Wan"

    def load(
        self,
        bridge_name: str,
        image_override: torch.Tensor | None = None,
    ):
        if bridge_name == "(none)":
            raise ValueError(
                "[MotionBridgeLoad] No bridge files found in output/motion_bridges/. "
                "Run MotionBridgeSave first."
            )

        st_path = _bridge_dir() / f"{bridge_name}.safetensors"
        if not st_path.exists():
            raise FileNotFoundError(
                f"[MotionBridgeLoad] Bridge file not found: {st_path}"
            )

        tensors, meta = _load_safetensors(st_path)

        tail = tensors["latent_tail"]  # [B, C, T, H, W]
        tail_frames = int(meta.get("tail_frames", str(tail.shape[2])))

        log.info(
            "[MotionBridgeLoad] Loaded '%s' | tail: %s | tail_frames=%s",
            bridge_name,
            list(tail.shape),
            tail_frames,
        )

        # IMAGE: override wins; fallback to saved last_frame
        if image_override is not None:
            image_out = image_override
            log.info("[MotionBridgeLoad] Using image_override (external image)")
        else:
            last_frame = tensors.get("last_frame")
            if last_frame is None:
                raise KeyError(
                    f"[MotionBridgeLoad] 'last_frame' not found in bridge file '{bridge_name}'. "
                    "Re-save with a newer version of MotionBridgeSave."
                )
            # last_frame is [H, W, C] — ComfyUI IMAGE is [B, H, W, C]
            if last_frame.dim() == 3:
                last_frame = last_frame.unsqueeze(0)
            image_out = last_frame.float()
            log.info("[MotionBridgeLoad] Using saved last_frame %s", list(image_out.shape))

        latent_out = {"samples": tail}
        return (latent_out, image_out, tail_frames)


# ════════════════════════════════════════════════════════════════════════════
# IAMCCS_LatentTailSlice  — lightweight stateless utility
# ════════════════════════════════════════════════════════════════════════════

class IAMCCS_LatentTailSlice:
    """
    Stateless utility: slices the last (or first) N latent frames from any LATENT.

    Use cases:
    · Extract prev_samples tail without saving to disk (in-workflow re-use)
    · Trim a LATENT to a precise temporal window
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "n_frames": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 64,
                        "step": 1,
                        "tooltip": "Number of latent frames to extract.",
                    },
                ),
                "from_end": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "True  → slice last N frames  (tail, for prev_samples). "
                            "False → slice first N frames (head, for anchor context)."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latent", "n_frames")
    FUNCTION = "slice_tail"
    CATEGORY = "IAMCCS/Wan"

    def slice_tail(self, latent: dict, n_frames: int, from_end: bool):
        samples = latent["samples"]  # [B, C, T, H, W]
        T = samples.shape[2]
        n = int(max(1, min(n_frames, T)))

        if from_end:
            sliced = samples[:, :, -n:].clone()
        else:
            sliced = samples[:, :, :n].clone()

        log.info(
            "[LatentTailSlice] T=%s n=%s from_end=%s → out shape %s",
            T, n, from_end, list(sliced.shape),
        )
        return ({"samples": sliced}, n)


# ── Node registry ────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "IAMCCS_MotionBridgeSave": IAMCCS_MotionBridgeSave,
    "IAMCCS_MotionBridgeLoad": IAMCCS_MotionBridgeLoad,
    "IAMCCS_LatentTailSlice":  IAMCCS_LatentTailSlice,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_MotionBridgeSave": "Motion Bridge Save 🎬💾",
    "IAMCCS_MotionBridgeLoad": "Motion Bridge Load 🎬📂",
    "IAMCCS_LatentTailSlice":  "Latent Tail Slice ✂️",
}
