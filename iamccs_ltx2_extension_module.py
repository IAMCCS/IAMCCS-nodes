# iamccs_ltx2_extension_module.py
# ===============================================================
# IAMCCS LTX-2 Extension Module
# All-in-one node for LTX-2 video extension workflows
# Combines: Image batch extension, overlap management, math operations
# ===============================================================

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import wave
from collections.abc import Mapping
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


_log = logging.getLogger("IAMCCS.LTX2.ExtensionModule")


def _resolve_output_path(path_value: str) -> str:
    out_dir = str(path_value or "").strip()
    if not out_dir:
        out_dir = "iamccs_extension_disk"
    if not os.path.isabs(out_dir):
        try:
            from folder_paths import get_output_directory  # type: ignore

            base_out = get_output_directory()
        except Exception:
            base_out = os.getcwd()
        out_dir = os.path.join(base_out, out_dir)
    return out_dir


def _list_frame_files(directory: str) -> list[str]:
    if not directory or not os.path.isdir(directory):
        return []
    files = []
    for name in os.listdir(directory):
        name_l = name.lower()
        if name_l.endswith(".png") or name_l.endswith(".jpg") or name_l.endswith(".jpeg") or name_l.endswith(".webp"):
            files.append(os.path.join(directory, name))
    files.sort()
    return files


def _clean_directory(directory: str):
    if os.path.isdir(directory):
        for name in os.listdir(directory):
            path = os.path.join(directory, name)
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except Exception:
                pass
    os.makedirs(directory, exist_ok=True)


def _promote_staged_directory(staged_dir: str, target_dir: str):
    _clean_directory(target_dir)
    for name in os.listdir(staged_dir):
        shutil.move(os.path.join(staged_dir, name), os.path.join(target_dir, name))
    shutil.rmtree(staged_dir, ignore_errors=True)


def _copy_frame(src_path: str, dst_path: str):
    shutil.copy2(src_path, dst_path)


def _blend_frame_pair(src_path: str, dst_path: str, out_path: str, mode: str, alpha: float):
    from PIL import Image  # type: ignore

    src_img = Image.open(src_path).convert("RGB")
    dst_img = Image.open(dst_path).convert("RGB")
    src_np = np.asarray(src_img, dtype=np.float32) / 255.0
    dst_np = np.asarray(dst_img, dtype=np.float32) / 255.0

    a = float(max(0.0, min(1.0, alpha)))
    if mode == "linear_blend":
        blended = (1.0 - a) * src_np + a * dst_np
    elif mode == "ease_in_out":
        eased = 3.0 * a * a - 2.0 * a * a * a
        blended = (1.0 - eased) * src_np + eased * dst_np
    elif mode == "filmic_crossfade":
        gamma = 2.2
        src_lin = np.power(np.clip(src_np, 0.0, 1.0), gamma)
        dst_lin = np.power(np.clip(dst_np, 0.0, 1.0), gamma)
        mix = (1.0 - a) * src_lin + a * dst_lin
        blended = np.power(np.clip(mix, 0.0, 1.0), 1.0 / gamma)
    else:
        blended = (1.0 - a) * src_np + a * dst_np

    out = (np.clip(blended, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(out).save(out_path)


def _build_ext(path_a: str, path_b: str) -> str:
    ext = os.path.splitext(path_a)[1] or os.path.splitext(path_b)[1]
    ext = ext.lower()
    if ext not in (".png", ".jpg", ".jpeg", ".webp"):
        ext = ".png"
    return ext


def _load_images_from_files(files: list[str]) -> torch.Tensor:
    from PIL import Image  # type: ignore

    images = []
    base_size = None
    load_errors = []
    for path in files:
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            load_errors.append(f"{os.path.basename(path)}: {exc}")
            continue
        if base_size is None:
            base_size = img.size
        elif img.size != base_size:
            img = img.resize(base_size, Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        images.append(torch.from_numpy(arr))
    if not images:
        if files and load_errors:
            preview = "; ".join(load_errors[:3])
            raise ValueError(f"No images loaded from files. Sample errors: {preview}")
        raise ValueError("No images loaded from files")
    return torch.stack(images, dim=0).contiguous()


class IAMCCS_LTX2_ExtensionModule:
    """
    All-in-one extension module for LTX-2 video generation workflows.
    Combines image batch extension with overlap, math operations, and frame calculations.
    
    Features:
    - Automatic overlap frame calculation with configurable modes
    - Multiple blending modes (linear, ease_in_out, filmic, perceptual)
    - Built-in math operations for frame calculations
    - Start images extraction for next generation pass
    - Compatible with iterative video extension workflows
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Source images (from previous generation or initial frames)
                "source_images": ("IMAGE", {
                    "tooltip": "The source images to extend (from previous generation)"
                }),
                
                # Overlap configuration
                "overlap_frames": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 256,
                    "step": 1,
                    "tooltip": "Number of overlapping frames between batches"
                }),
                
                "overlap_side": (["source", "new_images"], {
                    "default": "source",
                    "tooltip": "Which side to take overlap frames from"
                }),
                
                "overlap_mode": ([
                    "cut",
                    "linear_blend",
                    "ease_in_out",
                    "filmic_crossfade",
                    "perceptual_crossfade"
                ], {
                    "default": "linear_blend",
                    "tooltip": "Blending method for overlapping frames"
                }),
                
                # Math operations for frame calculations
                "enable_math": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable math calculations for frame adjustments"
                }),
                
                "math_operation": (["none", "a-b", "a-1", "a+b", "a*b", "a/b", "min(a,b)", "max(a,b)"], {
                    "default": "a-b",
                    "tooltip": "Math operation to perform on overlap value"
                }),

                "safe_mode": (["none", "native_workflow_safe"], {
                    "default": "none",
                    "tooltip": "Compatibility: mimic the original workflow behavior (start_images extracted as images[-overlap_frames:-1])"
                }),

                "start_frames_rule": (["none", "ltx2_round_down", "ltx2_nearest"], {
                    "default": "none",
                    "tooltip": "Optional: force start_images frame count to LTX rule (1 + 8*x) for VideoVAE encode"
                }),

                # Quality upgrades (default: none = keep current behavior)
                "color_match_mode": (["none", "luma_only", "per_channel"], {
                    "default": "none",
                    "tooltip": "Optional: match color/exposure of new_images to the tail of source_images before merging"
                }),
                "color_match_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "0=no effect, 1=full match (only used if color_match_mode != none)"
                }),
                "color_reference_window": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 256,
                    "step": 1,
                    "tooltip": "How many frames from the tail/head to use for stats matching"
                }),

                "seam_search_mode": (["none", "best_of_k"], {
                    "default": "none",
                    "tooltip": "Optional: search inside new_images for a better seam start (reduces rewind/odd restarts)"
                }),
                "k_search": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                    "tooltip": "How many candidate offsets to test (0 disables). Used only if seam_search_mode=best_of_k"
                }),
                "metric_weight_color": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Weight for color/luma continuity metric"
                }),
                "metric_weight_edges": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Weight for edge continuity metric"
                }),

                # Stitching presets (UI convenience). Frontend JS will also update widgets live.
                # Default is 'custom' to preserve existing workflows.
                "preset": ([
                    "custom",
                    "target_extension_ltx2",
                    "videoclip_audio_24fps",
                    "monologue_audio_24fps",
                    "cut_bestofk_16",
                    "cut_bestofk_16_luma",
                    "cut_bestofk_32",
                    "micro_crossfade_3",
                ], {
                    "default": "custom",
                    "tooltip": "Preset that auto-configures overlap/blend/seam search options (and updates widgets live). Choose 'custom' to keep manual settings."
                }),
            },
            "optional": {
                # New images (from current generation pass)
                "new_images": ("IMAGE", {
                    "tooltip": "The newly generated images to extend with"
                }),
                
                # Optional math operands
                "math_value_b": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 256,
                    "step": 1,
                    "tooltip": "Second operand for math operations (b)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "source_images",
        "start_images",
        "extended_images",
        "overlap_frames",
        "calculated_frames",
        "extension_frames",
        "report"
    )
    FUNCTION = "process_extension"
    CATEGORY = "IAMCCS/LTX-2"

    def _validate_ltx2_frames(self, frames: int) -> Tuple[bool, int]:
        """
        Validate if frame count follows LTX-2 rule (8n+1).
        Returns (is_valid, nearest_valid)
        """
        if frames < 1:
            return False, 1
        
        remainder = (frames - 1) % 8
        if remainder == 0:
            return True, frames
        
        # Find nearest valid value
        down = frames - remainder
        up = frames + (8 - remainder)
        nearest = up if (up - frames) <= (frames - down) else max(1, down)
        
        return False, nearest

    def _execute_math(self, operation: str, a: int, b: int) -> int:
        """Execute simple math operation safely"""
        try:
            if operation == "none" or operation == "":
                return a
            elif operation == "a-b":
                return max(0, a - b)
            elif operation == "a-1":
                return max(0, a - 1)
            elif operation == "a+b":
                return a + b
            elif operation == "a*b":
                return a * b
            elif operation == "a/b":
                return int(a / b) if b != 0 else a
            elif operation == "min(a,b)":
                return min(a, b)
            elif operation == "max(a,b)":
                return max(a, b)
            else:
                return a
        except Exception as e:
            _log.warning(f"Math operation failed: {e}, returning a={a}")
            return a

    def _blend_images(
        self,
        blend_src: torch.Tensor,
        blend_dst: torch.Tensor,
        mode: str
    ) -> torch.Tensor:
        """
        Blend two image batches using specified mode.
        Both inputs should have same shape: [N, H, W, C]
        """
        overlap = blend_src.shape[0]
        device = blend_src.device
        dtype = blend_src.dtype
        
        if mode == "cut":
            # No blending, just return destination
            return blend_dst
        
        elif mode == "linear_blend":
            # Simple linear interpolation
            alpha = torch.linspace(0, 1, overlap + 2, device=device, dtype=dtype)[1:-1]
            alpha = alpha.view(-1, 1, 1, 1)
            return (1 - alpha) * blend_src + alpha * blend_dst
        
        elif mode == "ease_in_out":
            # Smooth easing curve
            t = torch.linspace(0, 1, overlap + 2, device=device, dtype=dtype)[1:-1]
            eased_t = 3 * t * t - 2 * t * t * t
            eased_t = eased_t.view(-1, 1, 1, 1)
            return (1 - eased_t) * blend_src + eased_t * blend_dst
        
        elif mode == "filmic_crossfade":
            # Gamma-corrected blend for more natural transitions
            gamma = 2.2
            alpha = torch.linspace(0, 1, overlap + 2, device=device, dtype=dtype)[1:-1]
            alpha = alpha.view(-1, 1, 1, 1)
            
            linear_src = torch.pow(blend_src.clamp(0, 1), gamma)
            linear_dst = torch.pow(blend_dst.clamp(0, 1), gamma)
            blended = (1 - alpha) * linear_src + alpha * linear_dst
            return torch.pow(blended, 1.0 / gamma)
        
        elif mode == "perceptual_crossfade":
            # Blend in LAB color space for perceptually uniform transitions
            try:
                import kornia
                alpha = torch.linspace(0, 1, overlap + 2, device=device, dtype=dtype)[1:-1]
                alpha = alpha.view(-1, 1, 1, 1)
                
                # Convert to LAB space
                src_nchw = blend_src.movedim(-1, 1)
                dst_nchw = blend_dst.movedim(-1, 1)
                lab_src = kornia.color.rgb_to_lab(src_nchw)
                lab_dst = kornia.color.rgb_to_lab(dst_nchw)
                
                # Blend in LAB
                blended_lab = (1 - alpha) * lab_src + alpha * lab_dst
                
                # Convert back to RGB
                blended_rgb = kornia.color.lab_to_rgb(blended_lab)
                return blended_rgb.movedim(1, -1)
            except ImportError:
                _log.warning("Kornia not available, falling back to linear blend")
                return self._blend_images(blend_src, blend_dst, "linear_blend")
        
        else:
            # Fallback to linear
            return self._blend_images(blend_src, blend_dst, "linear_blend")

    def _apply_ltx2_frame_rule(self, frames: int, rule: str, max_allowed: int) -> int:
        """Apply LTX (1 + 8*x) rule to a frame count. rule='none' keeps value."""
        frames = int(frames)
        max_allowed = max(1, int(max_allowed))
        frames = max(1, min(frames, max_allowed))

        if rule == "none" or rule == "":
            return frames

        remainder = (frames - 1) % 8
        if remainder == 0:
            return frames

        down = max(1, frames - remainder)
        up = frames + (8 - remainder)

        if rule == "ltx2_round_down":
            return max(1, min(down, max_allowed))

        # nearest
        candidates = []
        if down <= max_allowed:
            candidates.append(down)
        if up <= max_allowed:
            candidates.append(up)
        if not candidates:
            return max(1, min(down, max_allowed))
        # choose min |delta|, prefer down on tie
        candidates.sort(key=lambda v: (abs(v - frames), v > frames))
        return int(candidates[0])

    def _compute_mean_std_per_channel(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-channel mean/std across batch+spatial dims for NHWC images."""
        if images.numel() == 0:
            raise ValueError("Empty image tensor")
        mean = images.mean(dim=(0, 1, 2))
        var = images.var(dim=(0, 1, 2), unbiased=False)
        std = torch.sqrt(var.clamp_min(1e-8))
        return mean, std

    def _match_color_exposure(
        self,
        new_images: torch.Tensor,
        source_images: torch.Tensor,
        mode: str,
        strength: float,
        reference_window: int,
    ) -> torch.Tensor:
        """Match new_images color/exposure to the tail of source_images. Images are NHWC in [0,1]."""
        if mode == "none" or strength <= 0.0:
            return new_images

        src_count = int(source_images.shape[0])
        new_count = int(new_images.shape[0])
        w = max(1, int(reference_window))
        src_ref = source_images[max(0, src_count - w):src_count]
        new_ref = new_images[: min(w, new_count)]

        if mode == "per_channel":
            src_mean, src_std = self._compute_mean_std_per_channel(src_ref)
            new_mean, new_std = self._compute_mean_std_per_channel(new_ref)
            scale = (src_std / new_std).view(1, 1, 1, -1)
            shift = (src_mean - (src_std / new_std) * new_mean).view(1, 1, 1, -1)
            matched = (new_images * scale + shift).clamp(0, 1)
        elif mode == "luma_only":
            # Match exposure/contrast on luma, apply same affine to all channels
            weights = torch.tensor([0.2126, 0.7152, 0.0722], device=new_images.device, dtype=new_images.dtype)
            src_y = (src_ref * weights.view(1, 1, 1, -1)).sum(dim=-1)
            new_y = (new_ref * weights.view(1, 1, 1, -1)).sum(dim=-1)
            src_mean = src_y.mean()
            src_std = src_y.std(unbiased=False).clamp_min(1e-6)
            new_mean = new_y.mean()
            new_std = new_y.std(unbiased=False).clamp_min(1e-6)
            scale = (src_std / new_std)
            shift = (src_mean - scale * new_mean)
            matched = (new_images * scale + shift).clamp(0, 1)
        else:
            return new_images

        s = float(max(0.0, min(1.0, strength)))
        return ((1.0 - s) * new_images + s * matched).clamp(0, 1)

    def _downsample_nhwc(self, images: torch.Tensor, size: int = 64) -> torch.Tensor:
        """Downsample NHWC images to size x size in NCHW."""
        nchw = images.movedim(-1, 1)
        return F.interpolate(nchw, size=(size, size), mode="bilinear", align_corners=False)

    def _sobel_mag(self, gray_nchw: torch.Tensor) -> torch.Tensor:
        """Sobel magnitude for NCHW grayscale tensor."""
        device = gray_nchw.device
        dtype = gray_nchw.dtype
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device, dtype=dtype).view(1, 1, 3, 3)
        ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=device, dtype=dtype).view(1, 1, 3, 3)
        gx = F.conv2d(gray_nchw, kx, padding=1)
        gy = F.conv2d(gray_nchw, ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-8)

    def _best_of_k_offset(
        self,
        source_tail: torch.Tensor,
        new_images: torch.Tensor,
        blend_overlap: int,
        k_search: int,
        w_color: float,
        w_edges: float,
    ) -> int:
        """Pick an offset into new_images that best matches source_tail over the overlap window."""
        if k_search <= 0:
            return 0

        new_count = int(new_images.shape[0])
        max_offset = min(int(k_search), max(0, new_count - blend_overlap))
        if max_offset <= 0:
            return 0

        # Prepare features (downsample + luma + edges)
        weights = torch.tensor([0.2126, 0.7152, 0.0722], device=new_images.device, dtype=new_images.dtype)
        src_win = source_tail[-blend_overlap:]
        src_ds = self._downsample_nhwc(src_win, size=64)
        src_luma = (src_ds * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
        src_edges = self._sobel_mag(src_luma)

        best_offset = 0
        best_score = None

        for off in range(0, max_offset + 1):
            cand = new_images[off: off + blend_overlap]
            if int(cand.shape[0]) != blend_overlap:
                continue
            cand_ds = self._downsample_nhwc(cand, size=64)
            cand_luma = (cand_ds * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
            cand_edges = self._sobel_mag(cand_luma)

            color_mse = (src_luma - cand_luma).pow(2).mean()
            edge_mse = (src_edges - cand_edges).pow(2).mean()
            score = float(w_color) * color_mse + float(w_edges) * edge_mse

            if best_score is None or score < best_score:
                best_score = score
                best_offset = off

        return int(best_offset)

    def process_extension(
        self,
        source_images: torch.Tensor,
        overlap_frames: int,
        overlap_side: str,
        overlap_mode: str,
        enable_math: bool,
        math_operation: str,
        safe_mode: str,
        start_frames_rule: str,
        color_match_mode: str,
        color_match_strength: float,
        color_reference_window: int,
        seam_search_mode: str,
        k_search: int,
        metric_weight_color: float,
        metric_weight_edges: float,
        preset: str = "custom",
        new_images: Optional[torch.Tensor] = None,
        math_value_b: int = 1,
    ):
        # Initialize
        source_count = int(source_images.shape[0])
        overlap_frames_in = int(overlap_frames)

        preset = str(preset or "custom")
        if preset != "custom":
            # NOTE: These presets are meant for stitching segments where crossfade is undesirable.
            # Frontend updates the widgets live; backend enforces the same mapping so renders match.
            preset_map: Dict[str, Dict[str, Any]] = {
                "videoclip_audio_24fps": {
                    "overlap_frames": 9,
                    "overlap_mode": "cut",
                    "overlap_side": "source",
                    "seam_search_mode": "best_of_k",
                    "k_search": 16,
                    "color_match_mode": "luma_only",
                    "color_match_strength": 0.25,
                    "color_reference_window": 8,
                },
                "monologue_audio_24fps": {
                    "overlap_frames": 13,
                    "overlap_mode": "cut",
                    "overlap_side": "source",
                    "seam_search_mode": "best_of_k",
                    "k_search": 16,
                    "color_match_mode": "luma_only",
                    "color_match_strength": 0.15,
                    "color_reference_window": 8,
                },
                # Prova 1: no crossfade, cut seam + best_of_k
                "cut_bestofk_16": {
                    "overlap_frames": 10,
                    "overlap_mode": "cut",
                    "overlap_side": "new_images",
                    "seam_search_mode": "best_of_k",
                    "k_search": 16,
                    "color_match_mode": "none",
                    "color_match_strength": 0.0,
                    "color_reference_window": 8,
                },
                # Prova 2: cut seam + luma match
                "cut_bestofk_16_luma": {
                    "overlap_frames": 10,
                    "overlap_mode": "cut",
                    "overlap_side": "new_images",
                    "seam_search_mode": "best_of_k",
                    "k_search": 16,
                    "color_match_mode": "luma_only",
                    "color_match_strength": 0.25,
                    "color_reference_window": 8,
                },
                # Prova 3: stronger seam search window
                "cut_bestofk_32": {
                    "overlap_frames": 16,
                    "overlap_mode": "cut",
                    "overlap_side": "new_images",
                    "seam_search_mode": "best_of_k",
                    "k_search": 32,
                    "color_match_mode": "none",
                    "color_match_strength": 0.0,
                    "color_reference_window": 8,
                },
                # Alternative: very short perceptual crossfade (minimizes visible dissolve)
                "micro_crossfade_3": {
                    "overlap_frames": 3,
                    "overlap_mode": "perceptual_crossfade",
                    "overlap_side": "source",
                    "seam_search_mode": "none",
                    "k_search": 0,
                    "color_match_mode": "none",
                    "color_match_strength": 0.0,
                    "color_reference_window": 8,
                },
            }

            cfg = preset_map.get(preset)
            if cfg is not None:
                overlap_frames_in = int(cfg.get("overlap_frames", overlap_frames_in))
                overlap_mode = str(cfg.get("overlap_mode", overlap_mode))
                overlap_side = str(cfg.get("overlap_side", overlap_side))
                seam_search_mode = str(cfg.get("seam_search_mode", seam_search_mode))
                k_search = int(cfg.get("k_search", k_search))
                color_match_mode = str(cfg.get("color_match_mode", color_match_mode))
                color_match_strength = float(cfg.get("color_match_strength", color_match_strength))
                color_reference_window = int(cfg.get("color_reference_window", color_reference_window))
        
        # Validate inputs (match KJNodes semantics: if overlap is too large, just passthrough)
        if source_count < 1:
            raise ValueError("source_images batch is empty")

        if overlap_frames_in < 1:
            overlap_frames_in = 1

        if overlap_frames_in >= source_count:
            report = (
                f"Source: {source_count} frames | "
                f"Overlap (effective): {overlap_frames_in} frames | "
                f"Start images: {source_count} frames | "
                f"Extended: {source_count} frames | "
                f"Extension delta: +0 frames | "
                f"Blend mode: {overlap_mode} | "
                f"Overlap side: {overlap_side}"
            )
            _log.info(f"[LTX2_ExtensionModule] {report}")
            return (
                source_images,
                source_images,
                source_images,
                overlap_frames_in,
                source_count,
                0,
                report,
            )

        # Initialize output
        # If no new_images are provided, behave as a "prep" node:
        # - extended_images == source_images
        # - start_images extracted from the (current) batch
        extended_images = source_images
        extension_frames_count = 0
        
        # Process extension if new images are provided
        if new_images is not None:
            assert new_images is not None
            new_count = int(new_images.shape[0])

            # Validate shapes
            if source_images.shape[1:3] != new_images.shape[1:3]:
                raise ValueError(
                    f"Source and new images must have same resolution: "
                    f"{tuple(source_images.shape[1:3])} vs {tuple(new_images.shape[1:3])}"
                )

            # Overlap used for blending (matches ImageBatchExtendWithOverlap)
            blend_overlap = min(overlap_frames_in, source_count, new_count)

            # Option 5: Best-of-K seam search (choose a better start inside new_images)
            chosen_offset = 0
            if seam_search_mode == "best_of_k" and int(k_search) > 0 and blend_overlap > 0:
                chosen_offset = self._best_of_k_offset(
                    source_images[-blend_overlap:],
                    new_images,
                    blend_overlap=blend_overlap,
                    k_search=int(k_search),
                    w_color=float(metric_weight_color),
                    w_edges=float(metric_weight_edges),
                )
                if chosen_offset > 0:
                    new_images = new_images[chosen_offset:]
                    new_count = int(new_images.shape[0])
                    blend_overlap = min(overlap_frames_in, source_count, new_count)

            # Option 3: Color/Exposure match (apply before blending)
            if color_match_mode != "none" and float(color_match_strength) > 0.0:
                new_images = self._match_color_exposure(
                    new_images=new_images,
                    source_images=source_images,
                    mode=str(color_match_mode),
                    strength=float(color_match_strength),
                    reference_window=int(color_reference_window),
                )

            prefix = source_images[:-blend_overlap]
            if overlap_side == "source":
                blend_src = source_images[-blend_overlap:]
                blend_dst = new_images[:blend_overlap]
            else:  # new_images
                blend_src = new_images[:blend_overlap]
                blend_dst = source_images[-blend_overlap:]

            suffix = new_images[blend_overlap:]

            if overlap_mode == "cut":
                # Match KJNodes cut semantics
                if overlap_side == "new_images":
                    extended_images = torch.cat((source_images, new_images[blend_overlap:]), dim=0)
                else:
                    extended_images = torch.cat((source_images[:-blend_overlap], new_images), dim=0)
            else:
                blended = self._blend_images(blend_src, blend_dst, overlap_mode)
                extended_images = torch.cat((prefix, blended, suffix), dim=0)

            extension_frames_count = int(extended_images.shape[0] - source_count)

        # Compute start_images from the CURRENT batch for the NEXT iteration.
        # When chaining multiple generations, using the post-merge batch (extended_images)
        # avoids graph cycles and removes the need for external math/range nodes.
        base_count = int(extended_images.shape[0])
        safe_mode = str(safe_mode or "none")
        if safe_mode == "native_workflow_safe":
            # Match the original graph: start_images is computed with
            # start = total - overlap_frames, end = total - 1 (exclusive)
            if base_count <= 1:
                start_images = extended_images[:1]
                start_index = 0
                start_end = int(start_images.shape[0])
            else:
                start_end = base_count - 1
                start_index = max(0, start_end - overlap_frames_in)
                start_images = extended_images[start_index:start_end]
        else:
            start_index = max(0, base_count - overlap_frames_in)

            calculated_frames = overlap_frames_in
            if enable_math and math_operation != "none":
                calculated_frames = self._execute_math(math_operation, overlap_frames_in, math_value_b)

            max_start_frames = max(1, base_count - start_index)
            calculated_frames = max(1, min(int(calculated_frames), max_start_frames))

            # Optional: enforce LTX (1+8*x) rule for VideoVAE encode
            calculated_frames = self._apply_ltx2_frame_rule(calculated_frames, str(start_frames_rule), max_start_frames)

            start_end = min(base_count, start_index + calculated_frames)
            start_images = extended_images[start_index:start_end]
        
        # Generate report
        report = (
            f"Source: {source_count} frames | "
            f"Overlap (effective): {overlap_frames_in} frames | "
            f"Start range (from current batch): start_index={start_index}, num_frames={start_images.shape[0]} | "
            f"Math: {math_operation if enable_math else 'disabled'} | "
            f"Start frames rule: {start_frames_rule} | "
            f"Safe mode: {safe_mode} | "
            f"Preset: {preset} | "
            f"Extended: {int(extended_images.shape[0]) if extended_images is not None else 0} frames | "
            f"Extension delta: +{extension_frames_count} frames | "
            f"Blend mode: {overlap_mode} | "
            f"Overlap side: {overlap_side} | "
            f"Color match: {color_match_mode} | "
            f"Seam search: {seam_search_mode}"
        )
        
        _log.info(f"[LTX2_ExtensionModule] {report}")
        
        return (
            source_images,           # Pass through source
            start_images,            # Start images for next pass
            extended_images,         # Extended result
            overlap_frames_in,       # Original overlap value
            int(start_images.shape[0]),  # Actual start-frame count
            extension_frames_count,  # How many frames were added
            report                   # Detailed report
        )


class IAMCCS_LTX2_ExtensionModule_Disk:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_dir": ("STRING", {"default": "iamccs_vae_frames/seg0", "tooltip": "Directory containing the current accumulated frames."}),
                "output_dir": ("STRING", {"default": "iamccs_extension_disk/extended", "tooltip": "Directory where the stitched sequence will be written."}),
                "start_dir": ("STRING", {"default": "iamccs_extension_disk/start", "tooltip": "Directory where start/overlap frames for the next pass will be written."}),
                "overlap_frames": ("INT", {"default": 9, "min": 1, "max": 256, "step": 1}),
                "overlap_side": (["source", "new_images"], {"default": "source"}),
                "overlap_mode": (["cut", "linear_blend", "ease_in_out", "filmic_crossfade"], {"default": "cut"}),
                "enable_math": ("BOOLEAN", {"default": True}),
                "math_operation": (["none", "a-b", "a-1", "a+b", "a*b", "a/b", "min(a,b)", "max(a,b)"], {"default": "none"}),
                "safe_mode": (["none", "native_workflow_safe"], {"default": "none"}),
                "start_frames_rule": (["none", "ltx2_round_down", "ltx2_nearest"], {"default": "none"}),
                "preset": (["custom", "target_extension_ltx2", "videoclip_audio_24fps", "monologue_audio_24fps", "cut_bestofk_16", "cut_bestofk_16_luma", "cut_bestofk_32", "micro_crossfade_3"], {"default": "custom"}),
            },
            "optional": {
                "new_dir": ("STRING", {"default": "", "tooltip": "Optional directory containing the new generated frames for the current pass."}),
                "math_value_b": ("INT", {"default": 1, "min": 0, "max": 256, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("extended_dir", "start_dir", "overlap_frames", "calculated_frames", "extension_frames", "report")
    FUNCTION = "process_extension_disk"
    CATEGORY = "IAMCCS/LTX-2"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # This node mutates output directories on disk. Always rerun to avoid stale
        # stitched sequences or start-frame folders being reused from cache.
        return float("nan")

    def _apply_ltx2_frame_rule(self, frames: int, rule: str, max_allowed: int) -> int:
        frames = int(frames)
        max_allowed = max(1, int(max_allowed))
        frames = max(1, min(frames, max_allowed))
        if rule == "none" or rule == "":
            return frames
        remainder = (frames - 1) % 8
        if remainder == 0:
            return frames
        down = max(1, frames - remainder)
        up = frames + (8 - remainder)
        if rule == "ltx2_round_down":
            return max(1, min(down, max_allowed))
        candidates = []
        if down <= max_allowed:
            candidates.append(down)
        if up <= max_allowed:
            candidates.append(up)
        if not candidates:
            return max(1, min(down, max_allowed))
        candidates.sort(key=lambda v: (abs(v - frames), v > frames))
        return int(candidates[0])

    def _execute_math(self, operation: str, a: int, b: int) -> int:
        if operation == "none" or operation == "":
            return a
        if operation == "a-b":
            return max(0, a - b)
        if operation == "a-1":
            return max(0, a - 1)
        if operation == "a+b":
            return a + b
        if operation == "a*b":
            return a * b
        if operation == "a/b":
            return int(a / b) if b != 0 else a
        if operation == "min(a,b)":
            return min(a, b)
        if operation == "max(a,b)":
            return max(a, b)
        return a

    def process_extension_disk(
        self,
        source_dir: str,
        output_dir: str,
        start_dir: str,
        overlap_frames: int,
        overlap_side: str,
        overlap_mode: str,
        enable_math: bool,
        math_operation: str,
        safe_mode: str,
        start_frames_rule: str,
        preset: str = "custom",
        new_dir: str = "",
        math_value_b: int = 1,
    ):
        preset = str(preset or "custom")
        preset_map: Dict[str, Dict[str, Any]] = {
            "videoclip_audio_24fps": {
                "overlap_frames": 9,
                "overlap_mode": "cut",
                "overlap_side": "source",
                "math_operation": "none",
                "safe_mode": "none",
                "start_frames_rule": "none",
            },
            "monologue_audio_24fps": {
                "overlap_frames": 13,
                "overlap_mode": "cut",
                "overlap_side": "source",
                "math_operation": "none",
                "safe_mode": "none",
                "start_frames_rule": "none",
            },
            "target_extension_ltx2": {
                "overlap_frames": 10,
                "overlap_mode": "linear_blend",
                "overlap_side": "source",
                "math_operation": "a-1",
                "safe_mode": "none",
                "start_frames_rule": "none",
            },
            "cut_bestofk_16": {"overlap_frames": 10, "overlap_mode": "cut", "overlap_side": "new_images"},
            "cut_bestofk_16_luma": {"overlap_frames": 10, "overlap_mode": "cut", "overlap_side": "new_images"},
            "cut_bestofk_32": {"overlap_frames": 16, "overlap_mode": "cut", "overlap_side": "new_images"},
            "micro_crossfade_3": {"overlap_frames": 3, "overlap_mode": "filmic_crossfade", "overlap_side": "source"},
        }
        cfg = preset_map.get(preset)
        overlap_frames_in = int(overlap_frames)
        if cfg is not None:
            overlap_frames_in = int(cfg.get("overlap_frames", overlap_frames_in))
            overlap_mode = str(cfg.get("overlap_mode", overlap_mode))
            overlap_side = str(cfg.get("overlap_side", overlap_side))
            math_operation = str(cfg.get("math_operation", math_operation))
            safe_mode = str(cfg.get("safe_mode", safe_mode))
            start_frames_rule = str(cfg.get("start_frames_rule", start_frames_rule))

        source_dir = _resolve_output_path(source_dir)
        output_dir = _resolve_output_path(output_dir)
        start_dir = _resolve_output_path(start_dir)
        new_dir = _resolve_output_path(new_dir) if str(new_dir or "").strip() else ""

        source_files = _list_frame_files(source_dir)
        if not source_files:
            raise ValueError(f"source_dir has no frame files: {source_dir}")
        new_files = _list_frame_files(new_dir) if new_dir else []

        source_count = len(source_files)

        temp_dirs_to_cleanup = []
        output_write_dir = output_dir
        start_write_dir = start_dir
        staged_output = False
        staged_start = False

        if output_dir in {source_dir, new_dir}:
            output_write_dir = tempfile.mkdtemp(prefix="iamccs_ext_out_", dir=os.path.dirname(output_dir) or None)
            temp_dirs_to_cleanup.append(output_write_dir)
            staged_output = True
        if start_dir in {source_dir, new_dir, output_dir}:
            start_write_dir = tempfile.mkdtemp(prefix="iamccs_ext_start_", dir=os.path.dirname(start_dir) or None)
            temp_dirs_to_cleanup.append(start_write_dir)
            staged_start = True

        _clean_directory(output_write_dir)
        _clean_directory(start_write_dir)

        if overlap_frames_in < 1:
            overlap_frames_in = 1

        written = 0
        extension_frames_count = 0

        try:
            if not new_files:
                for idx, src_path in enumerate(source_files):
                    ext = os.path.splitext(src_path)[1] or ".png"
                    _copy_frame(src_path, os.path.join(output_write_dir, f"frame_{idx:05d}{ext}"))
                    written += 1
                base_count = written
            else:
                new_count = len(new_files)
                blend_overlap = min(overlap_frames_in, source_count, new_count)
                ext = _build_ext(source_files[0], new_files[0])

                if overlap_mode == "cut":
                    if overlap_side == "new_images":
                        ordered = source_files + new_files[blend_overlap:]
                    else:
                        ordered = source_files[:-blend_overlap] + new_files
                    for idx, src_path in enumerate(ordered):
                        _copy_frame(src_path, os.path.join(output_write_dir, f"frame_{idx:05d}{ext}"))
                    written = len(ordered)
                else:
                    if overlap_side == "source":
                        prefix = source_files[:-blend_overlap]
                        src_overlap = source_files[-blend_overlap:]
                        dst_overlap = new_files[:blend_overlap]
                        suffix = new_files[blend_overlap:]
                    else:
                        prefix = source_files
                        src_overlap = new_files[:blend_overlap]
                        dst_overlap = source_files[-blend_overlap:]
                        suffix = new_files[blend_overlap:]

                    for src_path in prefix:
                        _copy_frame(src_path, os.path.join(output_write_dir, f"frame_{written:05d}{ext}"))
                        written += 1

                    for i in range(blend_overlap):
                        alpha = float(i + 1) / float(blend_overlap + 1)
                        _blend_frame_pair(src_overlap[i], dst_overlap[i], os.path.join(output_write_dir, f"frame_{written:05d}{ext}"), overlap_mode, alpha)
                        written += 1

                    for src_path in suffix:
                        _copy_frame(src_path, os.path.join(output_write_dir, f"frame_{written:05d}{ext}"))
                        written += 1

                base_count = written
                extension_frames_count = max(0, int(base_count - source_count))

            if base_count <= 0:
                raise ValueError("No output frames were written")

            if safe_mode == "native_workflow_safe":
                if base_count <= 1:
                    start_index = 0
                    calculated_frames = 1
                else:
                    start_end = base_count - 1
                    start_index = max(0, start_end - overlap_frames_in)
                    calculated_frames = max(1, start_end - start_index)
            else:
                start_index = max(0, base_count - overlap_frames_in)
                calculated_frames = overlap_frames_in
                if enable_math and math_operation != "none":
                    calculated_frames = self._execute_math(math_operation, overlap_frames_in, math_value_b)
                max_start_frames = max(1, base_count - start_index)
                calculated_frames = max(1, min(int(calculated_frames), max_start_frames))
                calculated_frames = self._apply_ltx2_frame_rule(calculated_frames, str(start_frames_rule), max_start_frames)

            output_files = _list_frame_files(output_write_dir)
            start_files = output_files[start_index:start_index + calculated_frames]
            for idx, src_path in enumerate(start_files):
                ext = os.path.splitext(src_path)[1] or ".png"
                _copy_frame(src_path, os.path.join(start_write_dir, f"start_{idx:05d}{ext}"))

            if staged_output:
                _promote_staged_directory(output_write_dir, output_dir)
                output_write_dir = output_dir
            if staged_start:
                _promote_staged_directory(start_write_dir, start_dir)
                start_write_dir = start_dir
        finally:
            for temp_dir in temp_dirs_to_cleanup:
                if os.path.isdir(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)

        report = (
            f"Source dir: {source_dir} ({source_count} frames) | "
            f"New dir: {new_dir or '[none]'} ({len(new_files)} frames) | "
            f"Output dir: {output_dir} ({base_count} frames) | "
            f"Start dir: {start_dir} ({len(start_files)} frames) | "
            f"Overlap: {overlap_frames_in} | Mode: {overlap_mode} | Side: {overlap_side} | "
            f"Math: {math_operation if enable_math else 'disabled'} | Safe: {safe_mode} | "
            f"Preset: {preset} | Extension delta: +{extension_frames_count}"
        )
        _log.info("[LTX2_ExtensionModule_Disk] %s", report)
        return (output_dir, start_dir, int(overlap_frames_in), int(len(start_files)), int(extension_frames_count), report)


class IAMCCS_LoadImagesFromDirLite:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "iamccs_extension_disk/extended", "tooltip": "Directory containing image frames."}),
                "mode": (["all", "from_start", "from_end", "range"], {"default": "all"}),
                "count": ("INT", {"default": 9, "min": 1, "max": 100000, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "end_index": ("INT", {"default": 9, "min": 0, "max": 100000, "step": 1}),
            },
            "optional": {
                "count_in": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1, "tooltip": "Optional linked override for count."}),
                "start_index_in": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1, "tooltip": "Optional linked override for range start index."}),
                "end_index_in": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1, "tooltip": "Optional linked override for range end index."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("images", "count", "report")
    FUNCTION = "load"
    CATEGORY = "IAMCCS/LTX-2"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Directory contents can change without the path string changing.
        return float("nan")

    def load(
        self,
        directory: str,
        mode: str,
        count: int,
        start_index: int,
        end_index: int,
        count_in: int | None = None,
        start_index_in: int | None = None,
        end_index_in: int | None = None,
    ):
        directory = _resolve_output_path(directory)
        files = _list_frame_files(directory)
        total = len(files)
        if total <= 0:
            raise ValueError(f"No images found in directory: {directory}")

        # Optional linked overrides can arrive as zero-valued defaults even when
        # the socket is not meaningfully used. Treat non-positive count/end
        # values as "no override" to preserve widget-configured ranges.
        if count_in is not None and int(count_in) > 0:
            count = int(count_in)
        if start_index_in is not None:
            start_index = int(start_index_in)
        if end_index_in is not None and int(end_index_in) > 0:
            end_index = int(end_index_in)

        count = max(1, int(count))
        start_index = max(0, int(start_index))
        end_index = max(0, int(end_index))

        if mode == "from_start":
            selected = files[:count]
        elif mode == "from_end":
            selected = files[-count:]
        elif mode == "range":
            if start_index >= total:
                fallback_start = max(0, total - count)
                _log.warning(
                    "[LoadImagesFromDirLite] start_index=%s out of range for total=%s in %s; falling back to tail slice [%s:%s]",
                    start_index,
                    total,
                    directory,
                    fallback_start,
                    total,
                )
                start_index = fallback_start
                end_index = total
            else:
                end_index = max(start_index, min(end_index, total))
            selected = files[start_index:end_index]
        else:
            selected = files

        if not selected and total > 0:
            raise ValueError(
                f"No files selected from {directory} (total={total}, mode={mode}, count={count}, start_index={start_index}, end_index={end_index})"
            )

        images = _load_images_from_files(selected)
        report = (
            f"Loaded {int(images.shape[0])} frames from {directory} "
            f"(total={total}, mode={mode}, count={count}, start_index={start_index}, end_index={end_index})"
        )
        return (images, int(images.shape[0]), report)


class IAMCCS_SourceFramesToDisk:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Source video frames loaded once from a single video input."}),
                "output_dir": ("STRING", {"default": "iamccs_source_frames/source_video", "tooltip": "Directory where the source frame cache will be written."}),
                "prefix": ("STRING", {"default": "source", "tooltip": "Frame filename prefix."}),
                "image_format": (["jpg", "png", "webp"], {"default": "jpg"}),
                "jpg_quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
                "clear_existing": ("BOOLEAN", {"default": True}),
                "start_number": ("INT", {"default": 0, "min": 0, "max": 100000000, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("frames_dir", "frame_count", "report")
    FUNCTION = "save"
    CATEGORY = "IAMCCS/LTX-2"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def save(self, images, output_dir: str, prefix: str, image_format: str, jpg_quality: int, clear_existing: bool, start_number: int):
        try:
            from PIL import Image  # type: ignore
        except Exception as e:
            raise RuntimeError(f"PIL (Pillow) is required for IAMCCS_SourceFramesToDisk: {e!r}")

        if not torch.is_tensor(images) or images.ndim != 4:
            raise ValueError("images must be an IMAGE tensor batch with shape [N,H,W,C]")

        out_dir = _resolve_output_path(output_dir)
        os.makedirs(out_dir, exist_ok=True)
        prefix = str(prefix or "source").strip() or "source"
        image_format = str(image_format or "jpg").lower()
        if image_format not in ("jpg", "png", "webp"):
            image_format = "jpg"
        jpg_quality = max(1, min(100, int(jpg_quality)))
        start_number = max(0, int(start_number))

        if bool(clear_existing):
            try:
                pfx = f"{prefix}_"
                for name in os.listdir(out_dir):
                    name_l = name.lower()
                    if not name.startswith(pfx):
                        continue
                    if not (name_l.endswith(".png") or name_l.endswith(".jpg") or name_l.endswith(".jpeg") or name_l.endswith(".webp")):
                        continue
                    try:
                        os.remove(os.path.join(out_dir, name))
                    except Exception:
                        pass
            except Exception as cleanup_error:
                _log.warning("[IAMCCS_SourceFramesToDisk] cleanup failed in %s: %s", out_dir, cleanup_error)

        img_cpu = torch.clamp(images.detach().to("cpu"), 0.0, 1.0)
        frame_count = int(img_cpu.shape[0])
        for idx in range(frame_count):
            filename = f"{prefix}_{start_number + idx:05d}.{image_format}"
            arr = (img_cpu[idx].numpy() * 255.0).round().astype(np.uint8)
            image = Image.fromarray(arr)
            save_path = os.path.join(out_dir, filename)
            if image_format == "jpg":
                image.save(save_path, format="JPEG", quality=jpg_quality)
            elif image_format == "webp":
                image.save(save_path, format="WEBP", quality=jpg_quality)
            else:
                image.save(save_path, format="PNG")

        report = f"Saved {frame_count} source frames to {out_dir} with prefix={prefix} format={image_format}"
        _log.info("[IAMCCS_SourceFramesToDisk] %s", report)
        return (out_dir, frame_count, report)


class IAMCCS_StartDirToVideoLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_dir": ("STRING", {"default": "iamccs_extension_disk/start", "tooltip": "Directory containing the start frames for the next segment."}),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "mode": (["all", "from_start", "from_end"], {"default": "all"}),
                "count": ("INT", {"default": 9, "min": 1, "max": 512, "step": 1}),
                "insert_at_pixel_frame": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "preprocess": ("BOOLEAN", {"default": True}),
                "preprocess_crf": ("INT", {"default": 33, "min": 0, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "STRING")
    RETURN_NAMES = ("latent", "frames_loaded", "report")
    FUNCTION = "inject"
    CATEGORY = "IAMCCS/LTX-2"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Reads start frames from disk. The folder path often stays constant while
        # the actual files change each iteration, so cache must be bypassed.
        return float("nan")

    def inject(self, start_dir: str, vae, latent, mode: str, count: int, insert_at_pixel_frame: int, strength: float, preprocess: bool, preprocess_crf: int):
        start_dir = _resolve_output_path(start_dir)
        files = _list_frame_files(start_dir)
        if not files:
            raise ValueError(f"No start frames found in directory: {start_dir}")

        count = max(1, int(count))
        if mode == "from_start":
            files = files[:count]
        elif mode == "from_end":
            files = files[-count:]

        images = _load_images_from_files(files)

        if preprocess:
            try:
                import comfy_extras.nodes_lt as nodes_lt  # type: ignore

                images = nodes_lt.LTXVPreprocess().execute(images, int(preprocess_crf))[0]
            except Exception as e:
                _log.warning("[IAMCCS_StartDirToVideoLatent] preprocess fallback: %s", e)

        samples = latent["samples"].clone()
        scale_factors = getattr(vae, "downscale_index_formula", (8, 32, 32))
        time_scale_factor, height_scale_factor, width_scale_factor = scale_factors
        batch, _, latent_frames, latent_height, latent_width = samples.shape
        width = latent_width * width_scale_factor
        height = latent_height * height_scale_factor

        if images.shape[1] != height or images.shape[2] != width:
            try:
                import comfy.utils  # type: ignore
            except Exception as e:
                raise ImportError("comfy.utils is required for IAMCCS_StartDirToVideoLatent") from e

            pixels = comfy.utils.common_upscale(images.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        else:
            pixels = images

        encoded = vae.encode(pixels[:, :, :, :3])
        if isinstance(encoded, dict):
            encoded = encoded.get("samples", encoded)
        if encoded.ndim == 4:
            encoded = encoded.unsqueeze(2)
        if encoded.ndim != 5:
            raise ValueError(f"Unexpected encoded latent shape: {tuple(encoded.shape)}")

        if encoded.shape[0] != batch:
            if encoded.shape[0] == 1 and batch == 1:
                pass
            elif batch == 1:
                encoded = encoded[:1]
            else:
                raise ValueError("Encoded batch does not match target latent batch")

        if "noise_mask" in latent:
            conditioning_latent_frames_mask = latent["noise_mask"].clone()
        else:
            conditioning_latent_frames_mask = torch.ones((batch, 1, latent_frames, 1, 1), dtype=torch.float32, device=samples.device)

        latent_idx = max(0, min(int(insert_at_pixel_frame) // max(1, int(time_scale_factor)), latent_frames - 1))
        end_index = min(latent_idx + int(encoded.shape[2]), latent_frames)
        samples[:, :, latent_idx:end_index] = encoded[:, :, :end_index - latent_idx]
        conditioning_latent_frames_mask[:, :, latent_idx:end_index] = 1.0 - float(max(0.0, min(1.0, strength)))

        report = (
            f"Loaded {int(images.shape[0])} start frames from {start_dir} | "
            f"insert_pixel={int(insert_at_pixel_frame)} -> latent_idx={latent_idx} | "
            f"encoded_t={int(encoded.shape[2])} | replaced={int(end_index - latent_idx)} latent slots"
        )
        return ({"samples": samples, "noise_mask": conditioning_latent_frames_mask}, int(images.shape[0]), report)


class IAMCCS_VideoCombineFromDir:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames_dir": ("STRING", {"default": "iamccs_extension_disk/final_extended", "tooltip": "Directory containing sequential frame files."}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}),
                "filename_prefix": ("STRING", {"default": "IAMCCS/LTX23_LOW_RAM", "tooltip": "Relative output prefix inside ComfyUI output, or absolute output path without extension."}),
                "crf": ("INT", {"default": 19, "min": 0, "max": 51, "step": 1}),
                "pix_fmt": (["yuv420p", "yuv444p"], {"default": "yuv420p"}),
                "trim_to_audio": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "audio": ("AUDIO", {}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "report")
    FUNCTION = "combine"
    CATEGORY = "IAMCCS/LTX-2"
    OUTPUT_NODE = True

    def _coerce_frames_dir(self, frames_dir: Any) -> str:
        current = frames_dir
        for _ in range(6):
            if current is None:
                break
            if isinstance(current, str):
                value = current.strip()
                if value:
                    return value
                break
            if isinstance(current, Mapping):
                if "frames_dir" in current:
                    current = current.get("frames_dir")
                    continue
                if "value1" in current:
                    current = current.get("value1")
                    continue
                if len(current) == 1:
                    current = next(iter(current.values()))
                    continue
                break
            if isinstance(current, (list, tuple)):
                if not current:
                    break
                current = current[0]
                continue
            break
        raise ValueError(f"frames_dir must resolve to a non-empty path string, got {type(frames_dir).__name__}")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Output nodes that read frame folders should not be satisfied from cache,
        # otherwise ComfyUI can mux an old on-disk sequence without rerunning the graph.
        return float("nan")

    def _find_ffmpeg(self) -> str:
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
        try:
            import imageio_ffmpeg  # type: ignore

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as e:
            raise RuntimeError("ffmpeg not found in PATH and imageio_ffmpeg unavailable") from e

    def _build_output_path(self, filename_prefix: str) -> str:
        prefix = str(filename_prefix or "IAMCCS/LTX23_LOW_RAM").strip()
        if prefix.lower().endswith(".mp4"):
            prefix = prefix[:-4]
        if os.path.isabs(prefix):
            out_path = prefix + ".mp4"
        else:
            try:
                from folder_paths import get_output_directory  # type: ignore

                out_dir = get_output_directory()
            except Exception:
                out_dir = os.getcwd()
            out_path = os.path.join(out_dir, prefix + ".mp4")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if not os.path.exists(out_path):
            return out_path

        stem, ext = os.path.splitext(out_path)
        index = 1
        while True:
            candidate = f"{stem}_{index:05d}{ext}"
            if not os.path.exists(candidate):
                return candidate
            index += 1

    def _sequence_pattern(self, files: list[str]) -> tuple[str, int] | None:
        if not files:
            return None
        m = re.match(r"^(.*?)(\d+)(\.[^.]+)$", os.path.basename(files[0]))
        if not m:
            return None
        prefix, digits, ext = m.groups()
        width = len(digits)
        for idx, path in enumerate(files):
            name = os.path.basename(path)
            m2 = re.match(r"^(.*?)(\d+)(\.[^.]+)$", name)
            if not m2:
                return None
            p2, d2, e2 = m2.groups()
            if p2 != prefix or e2 != ext or len(d2) != width:
                return None
        return (os.path.join(os.path.dirname(files[0]), f"{prefix}%0{width}d{ext}"), int(digits))

    def _unwrap_audio(self, audio: Any) -> Any:
        current = audio
        for _ in range(6):
            if current is None:
                return None
            if isinstance(current, Mapping):
                if "waveform" in current and "sample_rate" in current:
                    return {
                        "waveform": current["waveform"],
                        "sample_rate": current["sample_rate"],
                    }
                if "audio" in current:
                    current = current.get("audio")
                    continue
                if len(current) == 1:
                    current = next(iter(current.values()))
                    continue
                return current
            if isinstance(current, dict):
                if "waveform" in current and "sample_rate" in current:
                    return current
                if "audio" in current:
                    current = current.get("audio")
                    continue
                if len(current) == 1:
                    current = next(iter(current.values()))
                    continue
                return current
            if isinstance(current, (list, tuple)):
                if not current:
                    return None
                current = current[0]
                continue
            if hasattr(current, "waveform") and hasattr(current, "sample_rate"):
                return {
                    "waveform": getattr(current, "waveform"),
                    "sample_rate": getattr(current, "sample_rate"),
                }
            return current
        return current

    def _write_audio_wav(self, audio: Any, temp_dir: str) -> str | None:
        audio = self._unwrap_audio(audio)
        if audio is None:
            return None
        if not isinstance(audio, dict):
            _log.warning("[IAMCCS_VideoCombineFromDir] unsupported audio payload type: %s", type(audio).__name__)
            return None
        waveform = audio.get("waveform")
        sample_rate = int(audio.get("sample_rate", 0) or 0)
        if waveform is None or sample_rate <= 0:
            _log.warning("[IAMCCS_VideoCombineFromDir] audio payload missing waveform/sample_rate")
            return None

        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform)
        wf = waveform.detach().to("cpu")
        if wf.ndim == 3:
            wf = wf[0]
        if wf.ndim == 1:
            wf = wf.unsqueeze(0)
        if wf.ndim != 2:
            raise ValueError(f"Unsupported audio waveform shape: {tuple(wf.shape)}")

        wf = wf.clamp(-1.0, 1.0)
        pcm = (wf.numpy().T * 32767.0).astype(np.int16)
        wav_path = os.path.join(temp_dir, "audio.wav")
        with wave.open(wav_path, "wb") as wav_file:
            wav_file.setnchannels(int(pcm.shape[1]))
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm.tobytes())
        return wav_path

    def combine(self, frames_dir: Any, frame_rate: float, filename_prefix: str, crf: int, pix_fmt: str, trim_to_audio: bool, audio: Optional[Any] = None):
        frames_dir = _resolve_output_path(self._coerce_frames_dir(frames_dir))
        files = _list_frame_files(frames_dir)
        if not files:
            raise ValueError(f"No frames found in directory: {frames_dir}")

        ffmpeg = self._find_ffmpeg()
        out_path = self._build_output_path(filename_prefix)
        frame_rate = max(1.0, float(frame_rate))
        crf = max(0, min(51, int(crf)))
        pix_fmt = str(pix_fmt or "yuv420p")

        with tempfile.TemporaryDirectory(prefix="iamccs_ffmpeg_") as temp_dir:
            wav_path = self._write_audio_wav(audio, temp_dir)
            seq = self._sequence_pattern(files)
            if seq is not None:
                pattern, start_number = seq
                cmd = [ffmpeg, "-y", "-framerate", f"{frame_rate:.6f}", "-start_number", str(start_number), "-i", pattern]
            else:
                list_path = os.path.join(temp_dir, "frames.txt")
                with open(list_path, "w", encoding="utf-8") as f:
                    for path in files:
                        escaped = path.replace("'", "'\\''")
                        f.write(f"file '{escaped}'\n")
                        f.write(f"duration {1.0 / frame_rate:.12f}\n")
                    escaped = files[-1].replace("'", "'\\''")
                    f.write(f"file '{escaped}'\n")
                cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", list_path]

            if wav_path:
                cmd += ["-i", wav_path]

            cmd += ["-c:v", "libx264", "-pix_fmt", pix_fmt, "-crf", str(crf)]
            if wav_path:
                cmd += ["-c:a", "aac", "-b:a", "192k"]
                if trim_to_audio:
                    cmd += ["-shortest"]
            cmd += [out_path]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr.strip() or result.stdout.strip()}")

        mux_mode = "with_audio" if wav_path else "no_audio"
        report = f"Combined {len(files)} frames from {frames_dir} -> {out_path} @ {frame_rate:.3f}fps | {mux_mode}"
        _log.info("[IAMCCS_VideoCombineFromDir] %s", report)
        return (out_path, report)


class IAMCCS_LTX2_GetImageFromBatch:
    """
    Extracts a specific range of images from a batch.
    Useful for:
    - Getting start images for next iteration
    - Extracting specific frames from generation
    - Creating sub-batches from large batches
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input image batch"
                }),
                "mode": (["from_start", "from_end", "range", "drop_start", "drop_end"], {
                    "default": "from_end",
                    "tooltip": "Extraction mode"
                }),
                "count": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Number of frames to extract (for from_start/from_end)"
                }),

                # Upgrade options (default: none = keep current behavior)
                "auto_count_mode": (["none", "prefer_input", "use_widget"], {
                    "default": "none",
                    "tooltip": "Optional: auto-drive count from an INT input (e.g. overlap_frames)"
                }),
                "diagnostics": (["none", "basic"], {
                    "default": "none",
                    "tooltip": "Optional: expose start/end indices as extra outputs"
                }),

                "count_rule": (["none", "ltx2_round_down", "ltx2_nearest"], {
                    "default": "none",
                    "tooltip": "Optional: force count to LTX rule (1 + 8*x) for VideoVAE encode"
                }),

                "safe_mode": (["none", "native_workflow_safe"], {
                    "default": "none",
                    "tooltip": "Compatibility: mimic original GetImageRangeFromBatch behavior for from_end (images[-count:-1])"
                }),
            },
            "optional": {
                "count_in": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Optional count input (used if auto_count_mode=prefer_input)"
                }),
                "start_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Start index for range mode"
                }),
                "end_index": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "End index for range mode (exclusive)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "STRING", "INT", "INT")
    RETURN_NAMES = ("images", "count", "report", "start_index", "end_index")
    FUNCTION = "extract"
    CATEGORY = "IAMCCS/LTX-2"
    
    def extract(self, images, mode, count, auto_count_mode, diagnostics, count_rule, safe_mode, count_in=None, start_index=0, end_index=10):
        """Extract images from batch"""
        total = images.shape[0]

        def apply_ltx_rule(n: int, rule: str, max_allowed: int) -> int:
            n = int(n)
            max_allowed = max(1, int(max_allowed))
            n = max(1, min(n, max_allowed))
            if rule == "none" or rule == "":
                return n
            remainder = (n - 1) % 8
            if remainder == 0:
                return n
            down = max(1, n - remainder)
            up = n + (8 - remainder)
            if rule == "ltx2_round_down":
                return max(1, min(down, max_allowed))
            candidates = []
            if down <= max_allowed:
                candidates.append(down)
            if up <= max_allowed:
                candidates.append(up)
            if not candidates:
                return max(1, min(down, max_allowed))
            candidates.sort(key=lambda v: (abs(v - n), v > n))
            return int(candidates[0])

        # Option C: Auto-Count
        effective_count = int(count)
        if auto_count_mode != "none" and count_in is not None:
            if auto_count_mode == "prefer_input":
                effective_count = int(count_in)
            elif auto_count_mode == "use_widget":
                effective_count = int(count)

        effective_count = max(1, min(effective_count, int(total)))

        safe_mode = str(safe_mode or "none")
        if safe_mode == "native_workflow_safe" and mode == "from_end":
            # Match: start = total - count, end = total - 1 (exclusive)
            if int(total) <= 1:
                result = images[:1]
                used_start = 0
                used_end = int(result.shape[0])
                report = f"Extracted (safe) {result.shape[0]} frames from batch of {total}"
            else:
                used_start = max(0, int(total) - int(effective_count))
                used_end = max(0, int(total) - 1)
                result = images[used_start:used_end]
                report = f"Extracted (safe) frames {used_start} to {used_end} ({result.shape[0]} frames) from batch of {total}"
            return (result, result.shape[0], report, used_start, used_end)

        # Optional: enforce LTX (1+8*x) rule for VideoVAE encode
        # Only meaningful when we EXTRACT a fixed number of frames.
        if mode in ("from_start", "from_end"):
            effective_count = apply_ltx_rule(effective_count, str(count_rule), int(total))

        # Drop modes: remove frames but keep the remaining tail/head.
        # These are intentionally NOT LTX-rule adjusted: they are trimming utilities.
        if mode == "drop_start":
            drop = max(0, min(int(effective_count), max(0, int(total) - 1)))
            result = images[drop:]
            used_start = int(drop)
            used_end = int(total)
            report = f"Dropped first {drop} frames from batch of {total}; kept {result.shape[0]} frames"
            return (result, result.shape[0], report, used_start, used_end)

        if mode == "drop_end":
            drop = max(0, min(int(effective_count), max(0, int(total) - 1)))
            keep_end = int(total) - int(drop)
            result = images[:keep_end]
            used_start = 0
            used_end = int(keep_end)
            report = f"Dropped last {drop} frames from batch of {total}; kept {result.shape[0]} frames"
            return (result, result.shape[0], report, used_start, used_end)
        
        if mode == "from_start":
            result = images[:effective_count]
            used_start = 0
            used_end = effective_count
            report = f"Extracted first {effective_count} frames from batch of {total}"
        
        elif mode == "from_end":
            result = images[-effective_count:]
            used_start = int(total) - int(effective_count)
            used_end = int(total)
            report = f"Extracted last {effective_count} frames from batch of {total}"
        
        else:  # range
            start_index = max(0, min(start_index, total))
            end_index = max(start_index, min(end_index, total))
            result = images[start_index:end_index]
            actual_count = result.shape[0]
            used_start = int(start_index)
            used_end = int(end_index)
            report = f"Extracted frames {start_index} to {end_index} ({actual_count} frames) from batch of {total}"
        
        return (result, result.shape[0], report, used_start, used_end)


class IAMCCS_LTX2_ReferenceImageSwitch:
    """Selects a reference image (or keeps the default).

    Intended use: feed the output into a segment node's optional/secondary image input
    (e.g. `image_1`) to reinforce identity/style consistency WITHOUT touching the
    overlap/start-image continuity input.

    Default behavior is `none` which preserves old workflows.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "default_image": ("IMAGE", {"tooltip": "Fallback image (e.g. EmptyImage)"}),
                "mode": (["none", "use_reference", "blend"], {
                    "default": "none",
                    "tooltip": "none: pass default_image | use_reference: output reference_image | blend: mix both"
                }),
                "blend_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Only used if mode=blend (0=default, 1=reference)"
                }),
            },
            "optional": {
                "reference_image": ("IMAGE", {"tooltip": "Optional reference image (usually batch size 1)"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "select"
    CATEGORY = "IAMCCS/LTX-2"

    def _match_batch(self, base: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        """Broadcast/crop `other` batch to match `base` batch length when possible."""
        bn = int(base.shape[0])
        on = int(other.shape[0])
        if bn == on:
            return other
        if on == 1 and bn > 1:
            return other.repeat(bn, 1, 1, 1)
        if bn == 1 and on > 1:
            return other[:1]
        # Both >1 but mismatch: crop to min
        m = min(bn, on)
        return other[:m]

    def _resize_to(self, image: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        if int(image.shape[1]) == int(target_h) and int(image.shape[2]) == int(target_w):
            return image
        # IMAGE tensors in ComfyUI are [N, H, W, C]
        x = image.permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(int(target_h), int(target_w)), mode="bilinear", align_corners=False)
        x = x.permute(0, 2, 3, 1)
        return x.clamp(0, 1)

    def select(self, default_image: torch.Tensor, mode: str, blend_strength: float, reference_image: Optional[torch.Tensor] = None):
        mode = str(mode or "none")
        if mode == "none" or reference_image is None:
            return (default_image, f"Reference switch: {mode} (using default_image)")

        ref = self._match_batch(default_image, reference_image)
        base = default_image
        # If we cropped the ref, crop base too to keep alignment.
        if int(ref.shape[0]) != int(base.shape[0]):
            base = base[: int(ref.shape[0])]

        target_h, target_w = int(base.shape[1]), int(base.shape[2])
        resized = False
        if int(ref.shape[1]) != target_h or int(ref.shape[2]) != target_w:
            ref = self._resize_to(ref, target_h, target_w)
            resized = True

        if mode == "use_reference":
            return (ref, f"Reference switch: use_reference{' (resized)' if resized else ''}")

        # blend
        s = float(max(0.0, min(1.0, blend_strength)))
        out = ((1.0 - s) * base + s * ref).clamp(0, 1)
        return (out, f"Reference switch: blend (strength={s:.2f}){' (resized)' if resized else ''}")


class IAMCCS_LTX2_ReferenceStartFramesInjector:
    """Inject a reference image into the conditioning frames (start_images).

    Why: in LTX extension workflows, the model mostly follows `images` (conditioning frames).
    Feeding a reference into an auxiliary/empty-latent image slot often has little/no effect on identity.
    This node lets you (optionally) blend the reference into the last (or first) K conditioning frames.

    Default mode is `none` to preserve old workflows.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_images": ("IMAGE", {"tooltip": "Conditioning frames (e.g. start_images from ExtensionModule)"}),
                "mode": (["none", "inject", "blend"], {
                    "default": "none",
                    "tooltip": "none: passthrough | inject: replace frames with reference | blend: mix with existing"
                }),
                "blend_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Used for mode=blend (0=no change, 1=full reference). For mode=inject it's treated as 1.0"
                }),
                "frames_to_inject": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "How many conditioning frames to modify"
                }),
                "ramp": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If true, gradually increases strength across the injected frames"
                }),
                "position": (["tail", "head"], {
                    "default": "tail",
                    "tooltip": "Where to inject (tail=last K frames, head=first K frames)"
                }),
            },
            "optional": {
                "reference_image": ("IMAGE", {"tooltip": "Reference image (batch 1 is ok; will be resized to match start_images)"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("start_images", "report")
    FUNCTION = "inject"
    CATEGORY = "IAMCCS/LTX-2"

    def _resize_to(self, image: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        if int(image.shape[1]) == int(target_h) and int(image.shape[2]) == int(target_w):
            return image
        x = image.permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(int(target_h), int(target_w)), mode="bilinear", align_corners=False)
        x = x.permute(0, 2, 3, 1)
        return x.clamp(0, 1)

    def _repeat_or_crop_batch(self, desired_n: int, image: torch.Tensor) -> torch.Tensor:
        n = int(image.shape[0])
        if n == desired_n:
            return image
        if n == 1 and desired_n > 1:
            return image.repeat(desired_n, 1, 1, 1)
        return image[:desired_n]

    def inject(
        self,
        start_images: torch.Tensor,
        mode: str,
        blend_strength: float,
        frames_to_inject: int,
        ramp: bool,
        position: str,
        reference_image: Optional[torch.Tensor] = None,
    ):
        mode = str(mode or "none")
        if mode == "none" or reference_image is None:
            return (start_images, f"StartFrames injector: {mode} (passthrough)")

        base = start_images
        total = int(base.shape[0])
        k = int(max(1, min(int(frames_to_inject), total)))
        pos = str(position or "tail")

        target_h, target_w = int(base.shape[1]), int(base.shape[2])
        ref = self._resize_to(reference_image, target_h, target_w)
        ref = self._repeat_or_crop_batch(k, ref)

        out = base.clone()
        if pos == "head":
            idxs = list(range(0, k))
        else:  # tail
            idxs = list(range(total - k, total))

        # Strength handling
        if mode == "inject":
            max_s = 1.0
        else:
            max_s = float(max(0.0, min(1.0, blend_strength)))

        used = 0
        for j, i in enumerate(idxs):
            if ramp and k > 1:
                s = max_s * float(j + 1) / float(k)
            else:
                s = max_s
            out[i] = ((1.0 - s) * out[i] + s * ref[j]).clamp(0, 1)
            used += 1

        return (out, f"StartFrames injector: {mode} ({pos}, frames={used}, strength={max_s:.2f}, resized)")


class IAMCCS_LTX2_FrameCountValidator:
    """
    Validates and corrects frame counts for LTX-2 (8n+1 rule).
    Outputs: validated count, is_valid flag, nearest valid count.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_count": ("INT", {
                    "default": 81,
                    "min": 1,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Frame count to validate"
                }),
                "auto_correct": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically correct to nearest valid value"
                }),
                "correction_mode": (["nearest", "round_up", "round_down"], {
                    "default": "nearest",
                    "tooltip": "How to correct invalid values"
                }),
            }
        }
    
    RETURN_TYPES = ("INT", "BOOLEAN", "INT", "STRING")
    RETURN_NAMES = ("validated_count", "is_valid", "nearest_valid", "report")
    FUNCTION = "validate"
    CATEGORY = "IAMCCS/LTX-2"
    
    def validate(self, frame_count, auto_correct, correction_mode):
        """Validate LTX-2 frame count (8n+1 rule)"""
        
        # Check if valid
        remainder = (frame_count - 1) % 8
        is_valid = remainder == 0
        
        if is_valid:
            report = f"✅ {frame_count} is valid (8n+1 rule)"
            return (frame_count, True, frame_count, report)
        
        # Calculate corrections
        down = frame_count - remainder
        up = frame_count + (8 - remainder)
        
        if correction_mode == "round_up":
            nearest = up
        elif correction_mode == "round_down":
            nearest = max(1, down)
        else:  # nearest
            nearest = up if (up - frame_count) <= (frame_count - down) else max(1, down)
        
        # Output
        output_count = nearest if auto_correct else frame_count
        
        report = (
            f"❌ {frame_count} is NOT valid (8n+1 rule)\n"
            f"Remainder: {remainder}\n"
            f"Nearest valid: {nearest} (n={(nearest-1)//8})\n"
            f"Output: {output_count} ({'corrected' if auto_correct else 'uncorrected'})"
        )
        
        if auto_correct:
            _log.info(f"[LTX2_Validator] Corrected {frame_count} → {nearest}")
        
        return (output_count, False, nearest, report)


class IAMCCS_LTX2_ExtensionModule_simple(IAMCCS_LTX2_ExtensionModule):
    """A truly minimal Extension Module.

    Goals:
    - Keep ONLY the core widgets (overlap + blend + math)
    - No additional "quality" options (color match / seam search / metrics)
    - No user-facing safe_mode/start_frames_rule widgets
    - Always enforce LTX-2 start-frames rule (1 + 8*k) automatically
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_images": ("IMAGE", {
                    "tooltip": "The source images to extend (from previous generation)"
                }),
                "overlap_frames": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 256,
                    "step": 1,
                    "tooltip": "Number of overlapping frames between batches"
                }),
                "overlap_side": (["source", "new_images"], {
                    "default": "source",
                    "tooltip": "Which side to take overlap frames from"
                }),
                "overlap_mode": ([
                    "cut",
                    "linear_blend",
                    "ease_in_out",
                    "filmic_crossfade",
                    "perceptual_crossfade"
                ], {
                    "default": "linear_blend",
                    "tooltip": "Blending method for overlapping frames"
                }),
                "enable_math": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable math calculations for frame adjustments"
                }),
                "math_operation": (["none", "a-b", "a-1", "a+b", "a*b", "a/b", "min(a,b)", "max(a,b)"], {
                    "default": "a-b",
                    "tooltip": "Math operation to perform on overlap value"
                }),
                "math_value_b": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 256,
                    "step": 1,
                    "tooltip": "Second operand for math operations (b)"
                }),
            },
            "optional": {
                "new_images": ("IMAGE", {
                    "tooltip": "The newly generated images to extend with"
                }),
            },
        }

    RETURN_TYPES = IAMCCS_LTX2_ExtensionModule.RETURN_TYPES
    RETURN_NAMES = IAMCCS_LTX2_ExtensionModule.RETURN_NAMES
    FUNCTION = IAMCCS_LTX2_ExtensionModule.FUNCTION
    CATEGORY = "IAMCCS/LTX-2"

    def process_extension(
        self,
        source_images: torch.Tensor,
        overlap_frames: int,
        overlap_side: str,
        overlap_mode: str,
        enable_math: bool,
        math_operation: str,
        math_value_b: int,
        new_images: Optional[torch.Tensor] = None,
    ):
        # Fixed behavior knobs (not user-exposed in the simple node)
        safe_mode = "none"
        start_frames_rule = "ltx2_round_down"  # always enforce 8n+1

        color_match_mode = "none"
        color_match_strength = 0.0
        color_reference_window = 8

        seam_search_mode = "none"
        k_search = 0
        metric_weight_color = 1.0
        metric_weight_edges = 0.5

        return super().process_extension(
            source_images=source_images,
            overlap_frames=overlap_frames,
            overlap_side=overlap_side,
            overlap_mode=overlap_mode,
            enable_math=enable_math,
            math_operation=math_operation,
            safe_mode=safe_mode,
            start_frames_rule=start_frames_rule,
            color_match_mode=color_match_mode,
            color_match_strength=color_match_strength,
            color_reference_window=color_reference_window,
            seam_search_mode=seam_search_mode,
            k_search=k_search,
            metric_weight_color=metric_weight_color,
            metric_weight_edges=metric_weight_edges,
            new_images=new_images,
            math_value_b=int(math_value_b),
        )


class IAMCCS_LTX2_FirstLastFramesController:
    """
    First-Last Frame (FLF) controller for LTX-2 image conditioning.

    Injects a reference first_frame and/or last_frame directly into the
    `images` conditioning tensor used by the sampler.  Works on the
    'MISTO' pattern: the tensor already contains both external images and
    generated frames — this node simply overwrites / blends the head and/or
    tail K frames with the supplied references.

    Modes
    -----
    hard_lock   : replace the K frames completely with the reference
    linear_blend: weighted blend  (reference * strength + original * (1-strength))
    ramp        : progressive blend, strength ramps from 0 → strength over K frames
                  (for head: 0→strength left-to-right;  for tail: strength→0 left-to-right)

    Positions
    ---------
    head  : operate on first K frames only
    tail  : operate on last  K frames only
    both  : operate on both ends simultaneously
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Conditioning image batch (the 'images' input to the sampler)"
                }),
                "k_frames": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Number of frames to affect at each injection site"
                }),
                "mode": (["hard_lock", "linear_blend", "ramp"], {
                    "default": "hard_lock",
                    "tooltip": (
                        "hard_lock: full replace  |  "
                        "linear_blend: uniform blend at given strength  |  "
                        "ramp: progressive blend from 0 to strength"
                    ),
                }),
                "position": (["head", "tail", "both"], {
                    "default": "both",
                    "tooltip": "Where to inject references (head=first K, tail=last K, both=head+tail)",
                }),
                "blend_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Max blend weight (ignored for hard_lock which always uses 1.0)"
                }),
            },
            "optional": {
                "first_frame": ("IMAGE", {
                    "tooltip": "Reference image to inject at the HEAD of the batch (ignored if position=tail)"
                }),
                "last_frame": ("IMAGE", {
                    "tooltip": "Reference image to inject at the TAIL of the batch (ignored if position=head)"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "apply"
    CATEGORY = "IAMCCS/LTX-2"

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resize_to(image: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """Resize image tensor [N,H,W,C] to (target_h, target_w)."""
        if int(image.shape[1]) == target_h and int(image.shape[2]) == target_w:
            return image
        x = image.permute(0, 3, 1, 2)
        x = F.interpolate(x.float(), size=(target_h, target_w), mode="bilinear", align_corners=False)
        return x.permute(0, 2, 3, 1).clamp(0.0, 1.0).to(image.dtype)

    @staticmethod
    def _broadcast_ref(ref: torch.Tensor, k: int) -> torch.Tensor:
        """Ensure ref has exactly k frames (repeat single-frame or crop)."""
        n = int(ref.shape[0])
        if n == k:
            return ref
        if n == 1:
            return ref.repeat(k, 1, 1, 1)
        return ref[:k]

    @staticmethod
    def _blend_weights(k: int, mode: str, max_s: float, ramp_direction: str) -> list:
        """
        Returns list of k blend weights.
        ramp_direction: 'up' = 0→max_s, 'down' = max_s→0
        """
        if mode == "hard_lock":
            return [1.0] * k
        if mode == "linear_blend":
            return [max_s] * k
        # ramp
        if k == 1:
            return [max_s]
        if ramp_direction == "up":
            return [max_s * float(i + 1) / float(k) for i in range(k)]
        else:  # down
            return [max_s * float(k - i) / float(k) for i in range(k)]

    def _inject(
        self,
        out: torch.Tensor,
        ref: torch.Tensor,
        idxs: list,
        weights: list,
    ) -> torch.Tensor:
        """Blend ref frames into out at given indices with given per-frame weights."""
        h, w = int(out.shape[1]), int(out.shape[2])
        ref_r = self._resize_to(ref, h, w)
        ref_r = self._broadcast_ref(ref_r, len(idxs))
        for j, i in enumerate(idxs):
            s = float(weights[j])
            out[i] = ((1.0 - s) * out[i].float() + s * ref_r[j].float()).clamp(0.0, 1.0).to(out.dtype)
        return out

    # ------------------------------------------------------------------
    # main
    # ------------------------------------------------------------------
    def apply(
        self,
        images: torch.Tensor,
        k_frames: int,
        mode: str,
        position: str,
        blend_strength: float,
        first_frame: Optional[torch.Tensor] = None,
        last_frame: Optional[torch.Tensor] = None,
    ):
        total = int(images.shape[0])
        k = max(1, min(int(k_frames), total // 2 if total > 1 else 1))
        max_s = 1.0 if mode == "hard_lock" else float(max(0.0, min(1.0, blend_strength)))

        out = images.clone()
        ops = []

        do_head = position in ("head", "both")
        do_tail = position in ("tail", "both")

        if do_head and first_frame is not None:
            idxs = list(range(0, k))
            # ramp up: 0 → max_s  (anchor gets full weight at the end)
            weights = self._blend_weights(k, mode, max_s, "up")
            out = self._inject(out, first_frame, idxs, weights)
            ops.append(f"head(k={k},mode={mode},s={max_s:.2f})")

        if do_tail and last_frame is not None:
            idxs = list(range(total - k, total))
            # ramp down: max_s → 0  (anchor gets full weight at the start)
            weights = self._blend_weights(k, mode, max_s, "down")
            out = self._inject(out, last_frame, idxs, weights)
            ops.append(f"tail(k={k},mode={mode},s={max_s:.2f})")

        if not ops:
            report = f"FLF Controller: no-op (position={position}, first_frame={'yes' if first_frame is not None else 'no'}, last_frame={'yes' if last_frame is not None else 'no'})"
        else:
            report = "FLF Controller: " + " + ".join(ops) + f" | total_frames={total}"

        _log.debug(report)
        return (out, report)


class IAMCCS_LTX2_ContextLatent:
    """LTX-2 video continuation context injection (LATENT).

    IAMCCS-native equivalent of TTP's `LTXVContext_TTP`.

    Takes the last N frames from `previous_video`, encodes with VAE, and embeds
    them into the beginning of `latent` plus a `noise_mask` to partially lock
    those frames.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "previous_video": ("IMAGE", {
                    "tooltip": "Previous segment frames (IMAGE batch = frames)"
                }),
                "vae": ("VAE", {}),
                "latent": ("LATENT", {
                    "tooltip": "Empty latent for the next segment"
                }),
                "enable": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If false, passthrough latent (disables context injection)"
                }),
                "context_latent_frames": ("INT", {
                    "default": 6,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "How many latent frames to embed at start (LTX uses 8n+1 mapping)"
                }),
                "exclude_last_frame": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If true, excludes the very last frame of previous_video when building context (often reduces over-constraint)"
                }),
            },
            "optional": {
                "context_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "1.0=fully locked context, 0.0=no lock"
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "report")
    FUNCTION = "apply_context"
    CATEGORY = "IAMCCS/LTX-2"

    @staticmethod
    def _common_upscale_nhwc_to(images: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        # NOTE: name kept internal; use comfy.utils.common_upscale for exact ComfyUI behavior.
        try:
            import comfy.utils  # type: ignore
        except Exception as e:
            raise ImportError("comfy.utils is required for this node") from e

        if int(images.shape[1]) == int(target_h) and int(images.shape[2]) == int(target_w):
            return images
        x = comfy.utils.common_upscale(
            images.movedim(-1, 1),
            int(target_w),
            int(target_h),
            "bilinear",
            "center",
        ).movedim(1, -1)
        return x

    @staticmethod
    def _match_latent_batch(base_samples: torch.Tensor, other_samples: torch.Tensor) -> torch.Tensor:
        bb = int(base_samples.shape[0])
        ob = int(other_samples.shape[0])
        if bb == ob:
            return other_samples
        if ob == 1 and bb > 1:
            reps = [bb] + [1] * (other_samples.dim() - 1)
            return other_samples.repeat(*reps)
        return other_samples[:bb]

    def apply_context(self, previous_video, vae, latent, enable, context_latent_frames, exclude_last_frame=True, context_strength=1.0):
        if not bool(enable):
            return (latent, "Context: disabled (passthrough)")
        if previous_video is None:
            return (latent, "Context: no-op (previous_video=None)")

        samples_in = latent.get("samples")
        if samples_in is None:
            raise ValueError("LATENT input is missing 'samples'")

        samples = samples_in.clone()
        batch, channels, latent_frames, latent_height, latent_width = samples.shape

        # VAE scale factors -> target pixel dims
        _, height_scale_factor, width_scale_factor = vae.downscale_index_formula
        target_width = int(latent_width) * int(width_scale_factor)
        target_height = int(latent_height) * int(height_scale_factor)

        # LTX mapping: original_frames = (latent_frames - 1) * 8 + 1
        lf = max(1, int(context_latent_frames))
        required_frames = (lf - 1) * 8 + 1

        total_video_frames = int(previous_video.shape[0])
        if total_video_frames < 1:
            return (latent, "Context: no-op (previous_video empty)")

        end_idx = total_video_frames - 1 if bool(exclude_last_frame) else total_video_frames
        end_idx = max(0, min(end_idx, total_video_frames))
        start_idx = max(0, end_idx - required_frames)
        context_frames = previous_video[start_idx:end_idx]

        if int(context_frames.shape[0]) < 1:
            return (latent, "Context: no-op (no frames after exclude_last_frame)")

        pixels = self._common_upscale_nhwc_to(context_frames, target_height, target_width)
        encode_pixels = pixels[:, :, :, :3]
        context_latent = vae.encode(encode_pixels)

        context_latent = self._match_latent_batch(samples, context_latent)
        actual_latent_frames = int(context_latent.shape[2])
        embed_frames = min(actual_latent_frames, int(latent_frames))
        if embed_frames <= 0:
            return (latent, "Context: no-op (embed_frames=0)")

        samples[:, :, :embed_frames] = context_latent[:, :, :embed_frames]

        # Initialize / merge noise_mask (keep stronger constraints)
        if "noise_mask" in latent and latent["noise_mask"] is not None:
            noise_mask = latent["noise_mask"].clone()
        else:
            noise_mask = torch.ones((batch, 1, latent_frames, 1, 1), dtype=torch.float32, device=samples.device)

        s = float(max(0.0, min(1.0, context_strength)))
        new_mask_val = 1.0 - s
        current = noise_mask[:, :, :embed_frames]
        noise_mask[:, :, :embed_frames] = torch.minimum(current, torch.full_like(current, new_mask_val))

        out_latent = dict(latent)
        out_latent["samples"] = samples
        out_latent["noise_mask"] = noise_mask

        report = f"Context: frames[{start_idx}:{end_idx}] -> embed_latent_frames={embed_frames} (strength={s:.2f})"
        return (out_latent, report)


class IAMCCS_LTX2_MiddleFrames:
    """Accumulate middle-frame constraints for FLF (anytype).

    IAMCCS-native equivalent of TTP's `LTXVMiddleFrame_TTP`.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "position": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Relative position inside the latent timeline (0=head, 1=tail)"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
            },
            "optional": {
                "middle_frames": ("*", {
                    "tooltip": "Accumulator input (anytype)"
                }),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("middle_frames",)
    FUNCTION = "execute"
    CATEGORY = "IAMCCS/LTX-2"

    def execute(self, image, position, strength, middle_frames=None):
        if middle_frames is None:
            frames_list = []
        else:
            frames_list = list(middle_frames.get("frames", []))

        frames_list.append({
            "image": image,
            "position": float(position),
            "strength": float(strength),
        })

        return ({"frames": frames_list},)


class IAMCCS_LTX2_FirstLastLatentControl:
    """First/Last frame control for LTX-2 via LATENT + noise_mask.

    IAMCCS-native equivalent of TTP's `LTXVFirstLastFrameControl_TTP`.

    Embeds first/last (and optional middle) images into the latent samples via
    VAE encode and applies a noise_mask lock strength.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE", {}),
                "latent": ("LATENT", {}),
                "first_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
                "last_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
            },
            "optional": {
                "first_image": ("IMAGE", {"tooltip": "Optional first frame image"}),
                "last_image": ("IMAGE", {"tooltip": "Optional last frame image"}),
                "middle_frames": ("*", {"tooltip": "Optional middle-frame accumulator (anytype)"}),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "report")
    FUNCTION = "execute"
    CATEGORY = "IAMCCS/LTX-2"

    @staticmethod
    def _common_upscale_nhwc_to(images: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        try:
            import comfy.utils  # type: ignore
        except Exception as e:
            raise ImportError("comfy.utils is required for this node") from e

        if int(images.shape[1]) == int(target_h) and int(images.shape[2]) == int(target_w):
            return images
        return comfy.utils.common_upscale(
            images.movedim(-1, 1),
            int(target_w),
            int(target_h),
            "bilinear",
            "center",
        ).movedim(1, -1)

    @classmethod
    def _encode_image(cls, vae, image: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        pixels = cls._common_upscale_nhwc_to(image, int(target_h), int(target_w))
        encode_pixels = pixels[:, :, :, :3]
        return vae.encode(encode_pixels)

    @staticmethod
    def _match_latent_batch(base_samples: torch.Tensor, other_samples: torch.Tensor) -> torch.Tensor:
        bb = int(base_samples.shape[0])
        ob = int(other_samples.shape[0])
        if bb == ob:
            return other_samples
        if ob == 1 and bb > 1:
            reps = [bb] + [1] * (other_samples.dim() - 1)
            return other_samples.repeat(*reps)
        return other_samples[:bb]

    @staticmethod
    def _ensure_noise_mask(latent: Dict[str, Any], samples: torch.Tensor) -> torch.Tensor:
        batch, _, latent_frames, _, _ = samples.shape
        if "noise_mask" in latent and latent["noise_mask"] is not None:
            noise_mask = latent["noise_mask"].clone()
            if int(noise_mask.shape[0]) != int(batch):
                # best-effort crop/repeat
                if int(noise_mask.shape[0]) == 1 and int(batch) > 1:
                    noise_mask = noise_mask.repeat(int(batch), 1, 1, 1, 1)
                else:
                    noise_mask = noise_mask[: int(batch)]
            return noise_mask
        return torch.ones((batch, 1, latent_frames, 1, 1), dtype=torch.float32, device=samples.device)

    def execute(self, vae, latent, first_strength=1.0, last_strength=1.0, first_image=None, last_image=None, middle_frames=None):
        has_middle = middle_frames is not None and len(middle_frames.get("frames", [])) > 0
        if first_image is None and last_image is None and not has_middle:
            return (latent, "FLF(latent): no-op")

        # Robustness: callers sometimes pass a multi-frame IMAGE batch (e.g. 8n+1 start frames).
        # This node is meant to constrain only the first/last *frame*, so we collapse to 1 frame.
        if first_image is not None and int(first_image.shape[0]) > 1:
            first_image = first_image[:1]
        if last_image is not None and int(last_image.shape[0]) > 1:
            last_image = last_image[-1:]

        samples_in = latent.get("samples")
        if samples_in is None:
            raise ValueError("LATENT input is missing 'samples'")

        samples = samples_in.clone()
        batch, _, latent_frames, latent_height, latent_width = samples.shape

        _, height_scale_factor, width_scale_factor = vae.downscale_index_formula
        width = int(latent_width) * int(width_scale_factor)
        height = int(latent_height) * int(height_scale_factor)

        noise_mask = self._ensure_noise_mask(latent, samples)

        ops = []

        fs = float(max(0.0, min(1.0, first_strength)))
        if first_image is not None and fs > 0.0:
            first_latent = self._encode_image(vae, first_image, height, width)
            first_latent = self._match_latent_batch(samples, first_latent)
            flf = int(first_latent.shape[2])
            if flf > 0:
                flf = min(flf, int(latent_frames))
                samples[:, :, :flf] = first_latent[:, :, :flf]
                cur = noise_mask[:, :, :flf]
                noise_mask[:, :, :flf] = torch.minimum(cur, torch.full_like(cur, 1.0 - fs))
                ops.append(f"first(frames={flf},s={fs:.2f})")

        ls = float(max(0.0, min(1.0, last_strength)))
        if last_image is not None and ls > 0.0:
            last_latent = self._encode_image(vae, last_image, height, width)
            last_latent = self._match_latent_batch(samples, last_latent)
            llf = int(last_latent.shape[2])
            if llf > 0:
                if llf > int(latent_frames):
                    last_latent = last_latent[:, :, : int(latent_frames)]
                    llf = int(latent_frames)
                    last_start_idx = 0
                else:
                    last_start_idx = int(latent_frames) - llf
                samples[:, :, last_start_idx:] = last_latent[:, :, :llf]
                cur = noise_mask[:, :, last_start_idx:]
                noise_mask[:, :, last_start_idx:] = torch.minimum(cur, torch.full_like(cur, 1.0 - ls))
                ops.append(f"last(frames={llf},s={ls:.2f})")

        if has_middle:
            frames_list = [] if middle_frames is None else middle_frames.get("frames", [])
            for frame_data in frames_list:
                image = frame_data.get("image")
                position = float(frame_data.get("position", 0.5))
                strength = float(frame_data.get("strength", 1.0))
                strength = float(max(0.0, min(1.0, strength)))
                if image is None or strength <= 0.0:
                    continue

                mid_latent = self._encode_image(vae, image, height, width)
                mid_latent = self._match_latent_batch(samples, mid_latent)
                mlf = int(mid_latent.shape[2])
                if mlf <= 0:
                    continue

                middle_frame_idx = round(position * (int(latent_frames) - 1))
                middle_frame_idx = max(0, min(int(middle_frame_idx), int(latent_frames) - mlf))

                samples[:, :, middle_frame_idx:middle_frame_idx + mlf] = mid_latent[:, :, :mlf]
                cur = noise_mask[:, :, middle_frame_idx:middle_frame_idx + mlf]
                noise_mask[:, :, middle_frame_idx:middle_frame_idx + mlf] = torch.minimum(
                    cur,
                    torch.full_like(cur, 1.0 - strength),
                )

            ops.append(f"middle(count={len(frames_list)})")

        out_latent = dict(latent)
        out_latent["samples"] = samples
        out_latent["noise_mask"] = noise_mask

        report = "FLF(latent): " + (" + ".join(ops) if ops else "no-op")
        return (out_latent, report)


class IAMCCS_LTX2_FirstLastLatentControl_Pro(IAMCCS_LTX2_FirstLastLatentControl):
    """First/Last frame control for LTX-2 via LATENT + noise_mask (Pro).

    This variant mirrors the *core* stability trick used in `WanImageMotionPro`:
    cap how many temporal latent slots are locked for start/end, because some VAEs
    may encode even a single image to T>1 latent slots.

    Extras:
    - first_lock_slots / last_lock_slots: cap temporal slots to overwrite+lock.
    - end_transition_slots: optional smooth transition zone before the hard-locked end.
    """

    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        required = dict(base.get("required", {}))
        # Insert the extra widgets right after strengths (stable + discoverable).
        # NOTE: This is a NEW node type, so adding widgets won't break old workflows.
        required.update({
            "first_lock_slots": ("INT", {
                "default": 1,
                "min": 0,
                "max": 8,
                "step": 1,
                "tooltip": "Max temporal latent slots to overwrite+lock at the start (0 disables start lock).",
            }),
            "last_lock_slots": ("INT", {
                "default": 1,
                "min": 0,
                "max": 8,
                "step": 1,
                "tooltip": "Max temporal latent slots to overwrite+lock at the end (0 disables end lock).",
            }),
            "end_transition_slots": ("INT", {
                "default": 0,
                "min": 0,
                "max": 32,
                "step": 1,
                "tooltip": "Optional transition zone (in latent slots) before the hard-locked end.",
            }),
        })
        return {
            "required": required,
            "optional": base.get("optional", {}),
        }

    RETURN_TYPES = IAMCCS_LTX2_FirstLastLatentControl.RETURN_TYPES
    RETURN_NAMES = IAMCCS_LTX2_FirstLastLatentControl.RETURN_NAMES
    FUNCTION = IAMCCS_LTX2_FirstLastLatentControl.FUNCTION
    CATEGORY = IAMCCS_LTX2_FirstLastLatentControl.CATEGORY

    @staticmethod
    def _smoothstep(x: torch.Tensor) -> torch.Tensor:
        return x * x * (3.0 - 2.0 * x)

    def execute(
        self,
        vae,
        latent,
        first_strength=1.0,
        last_strength=1.0,
        first_lock_slots=1,
        last_lock_slots=1,
        end_transition_slots=0,
        first_image=None,
        last_image=None,
        middle_frames=None,
    ):
        has_middle = middle_frames is not None and len(middle_frames.get("frames", [])) > 0
        if first_image is None and last_image is None and not has_middle:
            return (latent, "FLF(latent_pro): no-op")

        # Same robustness as the base node: collapse multi-frame IMAGE batches to 1 frame.
        if first_image is not None and int(first_image.shape[0]) > 1:
            first_image = first_image[:1]
        if last_image is not None and int(last_image.shape[0]) > 1:
            last_image = last_image[-1:]

        samples_in = latent.get("samples")
        if samples_in is None:
            raise ValueError("LATENT input is missing 'samples'")

        samples = samples_in.clone()
        batch, _, latent_frames, latent_height, latent_width = samples.shape

        _, height_scale_factor, width_scale_factor = vae.downscale_index_formula
        width = int(latent_width) * int(width_scale_factor)
        height = int(latent_height) * int(height_scale_factor)

        noise_mask = self._ensure_noise_mask(latent, samples)

        ops = []

        lock_first = int(max(0, min(8, int(first_lock_slots))))
        lock_last = int(max(0, min(8, int(last_lock_slots))))
        trans_slots = int(max(0, min(32, int(end_transition_slots))))

        fs = float(max(0.0, min(1.0, first_strength)))
        flf = 0
        if first_image is not None and fs > 0.0 and lock_first > 0:
            first_latent = self._encode_image(vae, first_image, height, width)
            first_latent = self._match_latent_batch(samples, first_latent)
            flf_raw = int(first_latent.shape[2])
            flf = min(flf_raw, int(latent_frames), int(lock_first))
            if flf > 0:
                samples[:, :, :flf] = first_latent[:, :, :flf]
                cur = noise_mask[:, :, :flf]
                noise_mask[:, :, :flf] = torch.minimum(cur, torch.full_like(cur, 1.0 - fs))
                ops.append(f"first(frames={flf}/{flf_raw},s={fs:.2f})")
        elif first_image is not None and fs > 0.0 and lock_first == 0:
            ops.append("first(disabled)")

        ls = float(max(0.0, min(1.0, last_strength)))
        llf = 0
        last_start_idx = 0
        end_ref = None
        if last_image is not None and ls > 0.0 and lock_last > 0:
            last_latent = self._encode_image(vae, last_image, height, width)
            last_latent = self._match_latent_batch(samples, last_latent)
            llf_raw = int(last_latent.shape[2])

            llf = min(llf_raw, int(latent_frames), int(lock_last))
            if llf > 0:
                last_start_idx = int(latent_frames) - llf
                # Overwrite+lock ONLY the last `llf` slots.
                samples[:, :, last_start_idx:] = last_latent[:, :, -llf:]
                cur = noise_mask[:, :, last_start_idx:]
                noise_mask[:, :, last_start_idx:] = torch.minimum(cur, torch.full_like(cur, 1.0 - ls))

                # End reference slot = first slot of the locked zone in last_latent.
                end_ref_idx = max(0, int(last_latent.shape[2]) - llf)
                end_ref = last_latent[:, :, end_ref_idx : end_ref_idx + 1].to(
                    device=samples.device, dtype=samples.dtype
                )

                ops.append(f"last(frames={llf}/{llf_raw},s={ls:.2f})")
        elif last_image is not None and ls > 0.0 and lock_last == 0:
            ops.append("last(disabled)")

        # Optional end transition zone (helps avoid a visible 'stop' right before the locked end).
        if end_ref is not None and llf > 0 and trans_slots > 0 and last_start_idx > 0:
            trans_end = int(last_start_idx)  # exclusive
            # Never transition inside the start-locked zone.
            trans_start = max(int(flf), trans_end - trans_slots)
            trans_count = int(trans_end - trans_start)
            if trans_count > 0:
                x_vals = torch.linspace(
                    0.0,
                    1.0,
                    steps=trans_count + 2,
                    device=samples.device,
                    dtype=torch.float32,
                )[1:-1]
                for i in range(trans_count):
                    alpha = float(self._smoothstep(x_vals[i : i + 1]).item())
                    t = trans_start + i
                    # Blend samples toward end_ref.
                    samples[:, :, t : t + 1] = (
                        (1.0 - alpha) * samples[:, :, t : t + 1] + alpha * end_ref
                    )
                    # Gradually lock via noise_mask (stronger closer to the end).
                    target_mask = 1.0 - (ls * alpha)
                    cur = noise_mask[:, :, t : t + 1]
                    noise_mask[:, :, t : t + 1] = torch.minimum(cur, torch.full_like(cur, float(target_mask)))

                ops.append(f"end_transition(slots={trans_count})")

        # Middle frames behavior is inherited from base (same semantics).
        if has_middle:
            frames_list = [] if middle_frames is None else middle_frames.get("frames", [])
            for frame_data in frames_list:
                image = frame_data.get("image")
                position = float(frame_data.get("position", 0.5))
                strength = float(frame_data.get("strength", 1.0))
                strength = float(max(0.0, min(1.0, strength)))
                if image is None or strength <= 0.0:
                    continue

                mid_latent = self._encode_image(vae, image, height, width)
                mid_latent = self._match_latent_batch(samples, mid_latent)
                mlf = int(mid_latent.shape[2])
                if mlf <= 0:
                    continue

                middle_frame_idx = round(position * (int(latent_frames) - 1))
                middle_frame_idx = max(0, min(int(middle_frame_idx), int(latent_frames) - mlf))

                samples[:, :, middle_frame_idx:middle_frame_idx + mlf] = mid_latent[:, :, :mlf]
                cur = noise_mask[:, :, middle_frame_idx:middle_frame_idx + mlf]
                noise_mask[:, :, middle_frame_idx:middle_frame_idx + mlf] = torch.minimum(
                    cur,
                    torch.full_like(cur, 1.0 - strength),
                )

            ops.append(f"middle(count={len(frames_list)})")

        out_latent = dict(latent)
        out_latent["samples"] = samples
        out_latent["noise_mask"] = noise_mask

        report = "FLF(latent_pro): " + (" + ".join(ops) if ops else "no-op")
        return (out_latent, report)


# Node registration
NODE_CLASS_MAPPINGS = {
    "IAMCCS_LTX2_ExtensionModule": IAMCCS_LTX2_ExtensionModule,
    "IAMCCS_LTX2_ExtensionModule_simple": IAMCCS_LTX2_ExtensionModule_simple,
    "IAMCCS_LTX2_GetImageFromBatch": IAMCCS_LTX2_GetImageFromBatch,
    "IAMCCS_LTX2_ReferenceImageSwitch": IAMCCS_LTX2_ReferenceImageSwitch,
    "IAMCCS_LTX2_ReferenceStartFramesInjector": IAMCCS_LTX2_ReferenceStartFramesInjector,
    "IAMCCS_LTX2_FrameCountValidator": IAMCCS_LTX2_FrameCountValidator,
    "IAMCCS_LTX2_FirstLastFramesController": IAMCCS_LTX2_FirstLastFramesController,
    "IAMCCS_LTX2_ContextLatent": IAMCCS_LTX2_ContextLatent,
    "IAMCCS_LTX2_MiddleFrames": IAMCCS_LTX2_MiddleFrames,
    "IAMCCS_LTX2_FirstLastLatentControl": IAMCCS_LTX2_FirstLastLatentControl,
    "IAMCCS_LTX2_FirstLastLatentControl_Pro": IAMCCS_LTX2_FirstLastLatentControl_Pro,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_LTX2_ExtensionModule": "LTX-2 Extension Module 🎬",
    "IAMCCS_LTX2_ExtensionModule_simple": "LTX-2 Extension Module (simple) 🎬",
    "IAMCCS_LTX2_GetImageFromBatch": "LTX-2 Get Images From Batch 🎞️",
    "IAMCCS_LTX2_ReferenceImageSwitch": "LTX-2 Reference Image Switch 🧷",
    "IAMCCS_LTX2_ReferenceStartFramesInjector": "LTX-2 Inject Reference Into Start Frames 🧬",
    "IAMCCS_LTX2_FrameCountValidator": "LTX-2 Frame Count Validator ✅ (8n+1)",
    "IAMCCS_LTX2_FirstLastFramesController": "LTX-2 First-Last Frames Controller 🎯",
    "IAMCCS_LTX2_ContextLatent": "LTX-2 Context → Latent (continue) 🧩",
    "IAMCCS_LTX2_MiddleFrames": "LTX-2 Middle Frames (accumulator) 🧷",
    "IAMCCS_LTX2_FirstLastLatentControl": "LTX-2 First/Last → Latent (noise_mask) 🎯",
    "IAMCCS_LTX2_FirstLastLatentControl_Pro": "LTX-2 First/Last → Latent (Pro, slot caps) 🎯",
}
