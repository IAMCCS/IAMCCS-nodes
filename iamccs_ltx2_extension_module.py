# iamccs_ltx2_extension_module.py
# ===============================================================
# IAMCCS LTX-2 Extension Module
# All-in-one node for LTX-2 video extension workflows
# Combines: Image batch extension, overlap management, math operations
# ===============================================================

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


_log = logging.getLogger("IAMCCS.LTX2.ExtensionModule")


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
        new_images: Optional[torch.Tensor] = None,
        math_value_b: int = 1,
    ):
        # Initialize
        source_count = int(source_images.shape[0])
        overlap_frames_in = int(overlap_frames)
        
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
                "mode": (["from_start", "from_end", "range"], {
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
        if mode in ("from_start", "from_end"):
            effective_count = apply_ltx_rule(effective_count, str(count_rule), int(total))
        
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


# Node registration
NODE_CLASS_MAPPINGS = {
    "IAMCCS_LTX2_ExtensionModule": IAMCCS_LTX2_ExtensionModule,
    "IAMCCS_LTX2_ExtensionModule_simple": IAMCCS_LTX2_ExtensionModule_simple,
    "IAMCCS_LTX2_GetImageFromBatch": IAMCCS_LTX2_GetImageFromBatch,
    "IAMCCS_LTX2_ReferenceImageSwitch": IAMCCS_LTX2_ReferenceImageSwitch,
    "IAMCCS_LTX2_ReferenceStartFramesInjector": IAMCCS_LTX2_ReferenceStartFramesInjector,
    "IAMCCS_LTX2_FrameCountValidator": IAMCCS_LTX2_FrameCountValidator,
    "IAMCCS_LTX2_FirstLastFramesController": IAMCCS_LTX2_FirstLastFramesController,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_LTX2_ExtensionModule": "LTX-2 Extension Module 🎬",
    "IAMCCS_LTX2_ExtensionModule_simple": "LTX-2 Extension Module (simple) 🎬",
    "IAMCCS_LTX2_GetImageFromBatch": "LTX-2 Get Images From Batch 🎞️",
    "IAMCCS_LTX2_ReferenceImageSwitch": "LTX-2 Reference Image Switch 🧷",
    "IAMCCS_LTX2_ReferenceStartFramesInjector": "LTX-2 Inject Reference Into Start Frames 🧬",
    "IAMCCS_LTX2_FrameCountValidator": "LTX-2 Frame Count Validator ✅ (8n+1)",
    "IAMCCS_LTX2_FirstLastFramesController": "LTX-2 First-Last Frames Controller 🎯",
}
