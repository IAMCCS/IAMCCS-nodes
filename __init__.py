# ==========================================================
# __init__.py — Registro nodi IAMCCS
# ==========================================================

import logging
import os

from .iamccs_wan_lora_stack import (
    IAMCCS_WanLoRAStack,
    IAMCCS_ModelWithLoRA,
)
from .iamccs_wan_lora_stack_simple import (
    IAMCCS_WanLoRAStackModelIO,
)

from .iamccs_ltx2_lora_stack import (
    IAMCCS_LTX2_LoRAStack,
    IAMCCS_LTX2_LoRAStackStaged,
    IAMCCS_ModelWithLoRA_LTX2,
    IAMCCS_ModelWithLoRA_LTX2_Staged,
    IAMCCS_LTX2_LoRAStackModelIO,
)

from .iamccs_ltx2_lora_stack_segmented6 import (
    IAMCCS_LTX2_LoRAStackSegmented6,
    IAMCCS_LTX2_ModelWithLoRA_Segmented6,
)

from .iamccs_ltx2_tools import (
    IAMCCS_LTX2_FrameRateSync,
    IAMCCS_LTX2_Validator,
    IAMCCS_LTX2_TimeFrameCount,
    IAMCCS_LTX2_EnsureFrames8nPlus1,
    IAMCCS_LTX2_ControlPreprocess,
    IAMCCS_LTX2_ImageBatchPadReflect,
    IAMCCS_LTX2_ImageBatchCropByPad,
)

from .iamccs_ltx2_extension_module import (
    IAMCCS_LTX2_ExtensionModule,
    IAMCCS_LTX2_ExtensionModule_simple,
    IAMCCS_LTX2_GetImageFromBatch,
    IAMCCS_LTX2_ReferenceImageSwitch,
    IAMCCS_LTX2_ReferenceStartFramesInjector,
    IAMCCS_LTX2_FrameCountValidator,
)

from .iamccs_wan_svipro_motion import (
    IAMCCS_WanImageMotion,
)

from .iamccs_autolink import (
    IAMCCS_SetAutoLink,
    IAMCCS_GetAutoLink,
    IAMCCS_AutoLinkConverter,
    IAMCCS_AutoLinkArguments,
)

from .iamccs_gguf_accelerator import (
    IAMCCS_GGUF_accelerator,
)

# Nodi principali
NODE_CLASS_MAPPINGS = {
    "IAMCCS_WanLoRAStack": IAMCCS_WanLoRAStack,
    "IAMCCS_ModelWithLoRA": IAMCCS_ModelWithLoRA,
    "IAMCCS_WanLoRAStackModelIO": IAMCCS_WanLoRAStackModelIO,
    # Backward-compatible key (kept as-is for existing workflows)
    "iamccs_ltx2_lora_stack": IAMCCS_LTX2_LoRAStack,
    # Preferred explicit names
    "IAMCCS_LTX2_LoRAStack": IAMCCS_LTX2_LoRAStack,
    "IAMCCS_LTX2_LoRAStackStaged": IAMCCS_LTX2_LoRAStackStaged,
    "IAMCCS_ModelWithLoRA_LTX2": IAMCCS_ModelWithLoRA_LTX2,
    "IAMCCS_ModelWithLoRA_LTX2_Staged": IAMCCS_ModelWithLoRA_LTX2_Staged,
    "IAMCCS_LTX2_LoRAStackModelIO": IAMCCS_LTX2_LoRAStackModelIO,
    "IAMCCS_LTX2_LoRAStackSegmented6": IAMCCS_LTX2_LoRAStackSegmented6,
    "IAMCCS_LTX2_ModelWithLoRA_Segmented6": IAMCCS_LTX2_ModelWithLoRA_Segmented6,

    "IAMCCS_LTX2_FrameRateSync": IAMCCS_LTX2_FrameRateSync,
    "IAMCCS_LTX2_Validator": IAMCCS_LTX2_Validator,
    "IAMCCS_LTX2_TimeFrameCount": IAMCCS_LTX2_TimeFrameCount,
    "IAMCCS_LTX2_EnsureFrames8nPlus1": IAMCCS_LTX2_EnsureFrames8nPlus1,
    "IAMCCS_LTX2_ControlPreprocess": IAMCCS_LTX2_ControlPreprocess,
    "IAMCCS_LTX2_ImageBatchPadReflect": IAMCCS_LTX2_ImageBatchPadReflect,
    "IAMCCS_LTX2_ImageBatchCropByPad": IAMCCS_LTX2_ImageBatchCropByPad,
    "IAMCCS_LTX2_ExtensionModule": IAMCCS_LTX2_ExtensionModule,
    "IAMCCS_LTX2_ExtensionModule_simple": IAMCCS_LTX2_ExtensionModule_simple,
    "IAMCCS_LTX2_GetImageFromBatch": IAMCCS_LTX2_GetImageFromBatch,
    "IAMCCS_LTX2_ReferenceImageSwitch": IAMCCS_LTX2_ReferenceImageSwitch,
    "IAMCCS_LTX2_ReferenceStartFramesInjector": IAMCCS_LTX2_ReferenceStartFramesInjector,
    "IAMCCS_LTX2_FrameCountValidator": IAMCCS_LTX2_FrameCountValidator,
    "IAMCCS_WanImageMotion": IAMCCS_WanImageMotion,
    
    "IAMCCS_SetAutoLink": IAMCCS_SetAutoLink,
    "IAMCCS_GetAutoLink": IAMCCS_GetAutoLink,
    "IAMCCS_AutoLinkConverter": IAMCCS_AutoLinkConverter,
    "IAMCCS_AutoLinkArguments": IAMCCS_AutoLinkArguments,

    "IAMCCS_GGUF_accelerator": IAMCCS_GGUF_accelerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_WanLoRAStack": "LoRA Stack (WAN-style remap)",
    "IAMCCS_ModelWithLoRA": "Apply LoRA to MODEL (Native)",
    "IAMCCS_WanLoRAStackModelIO": "LoRA Stack (Model In→Out) WAN",
    "iamccs_ltx2_lora_stack": "iamccs_ltx2_lora_stack (3 slots)",
    "IAMCCS_LTX2_LoRAStack": "LoRA Stack (LTX-2, 3 slots)",
    "IAMCCS_LTX2_LoRAStackStaged": "LoRA Stack (LTX-2, staged: stage1+stage2) (BETA)",
    "IAMCCS_ModelWithLoRA_LTX2": "Apply LoRA to MODEL (LTX-2, quiet logs)",
    "IAMCCS_ModelWithLoRA_LTX2_Staged": "Apply LoRA to MODEL (LTX-2, staged) (BETA)",
    "IAMCCS_LTX2_LoRAStackModelIO": "LoRA Stack (Model In→Out) LTX-2",
    "IAMCCS_LTX2_LoRAStackSegmented6": "LoRA Stack (LTX-2, segmented: 3 seg × 2 stages)",
    "IAMCCS_LTX2_ModelWithLoRA_Segmented6": "Apply LoRA to MODEL (LTX-2, segmented: 3 seg × 2 stages)",

    "IAMCCS_LTX2_FrameRateSync": "LTX-2 FrameRate Sync (int+float)",
    "IAMCCS_LTX2_Validator": "LTX-2 Validator (16px, 8n +1)",
    "IAMCCS_LTX2_TimeFrameCount": "LTX-2 TimeFrameCount",
    "IAMCCS_LTX2_EnsureFrames8nPlus1": "LTX-2 Ensure Frames (8n + 1)",
    "IAMCCS_LTX2_ControlPreprocess": "LTX-2 Control Preprocess (aux)",
    "IAMCCS_LTX2_ImageBatchPadReflect": "LTX-2 Pad Reflect (IMAGE batch)",
    "IAMCCS_LTX2_ImageBatchCropByPad": "LTX-2 Crop By Pad (IMAGE batch)",
    "IAMCCS_LTX2_ExtensionModule": "LTX-2 Extension Module 🎬",
    "IAMCCS_LTX2_ExtensionModule_simple": "LTX-2 Extension Module (simple) 🎬",
    "IAMCCS_LTX2_GetImageFromBatch": "LTX-2 Get Images From Batch 🎞️",
    "IAMCCS_LTX2_ReferenceImageSwitch": "LTX-2 Reference Image Switch 🧷",
    "IAMCCS_LTX2_ReferenceStartFramesInjector": "LTX-2 Inject Reference Into Start Frames 🧬",
    "IAMCCS_LTX2_FrameCountValidator": "LTX-2 Frame Count Validator ✅ (8n+1)",
    "IAMCCS_WanImageMotion": "WanImageMotion",
    
    "IAMCCS_SetAutoLink": "Set AutoLink",
    "IAMCCS_GetAutoLink": "Get AutoLink",
    "IAMCCS_AutoLinkConverter": "AutoLink Converter",
    "IAMCCS_AutoLinkArguments": "AutoLink Arguments",

    "IAMCCS_GGUF_accelerator": "GGUF Accelerator (patch_on_device)",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


def _iamccs_install_ltx2_vae_encode_autofix() -> None:
    """Prevents hard-crash when LTX-2 VAE receives invalid frame counts.

    Lightricks video VAE encode requires a frame count of the form 1 + 8*x.
    Some workflows can produce off-by-a-few batches (e.g. 240 instead of 241),
    which otherwise raises ValueError and stops execution.

    This patch pads by repeating the last frame up to the next valid count.
    Opt-in via IAMCCS_LTX2_VAE_ENCODE_AUTOFIX=1.
    """

    # Default OFF: user requested workflow-level fixes without monkeypatching VAE.
    if str(os.getenv("IAMCCS_LTX2_VAE_ENCODE_AUTOFIX", "0")).strip().lower() in {"0", "false", "no", "off"}:
        return

    log = logging.getLogger("IAMCCS.LTX2.VAE")

    try:
        import torch
    except Exception:
        return

    try:
        from comfy.ldm.lightricks.vae import causal_video_autoencoder as _cvae
    except Exception:
        # ComfyUI / LTXVideo not installed or import path changed.
        return

    cls = getattr(_cvae, "CausalVideoAutoencoder", None)
    if cls is None:
        return

    orig_encode = getattr(cls, "encode", None)
    if orig_encode is None:
        return

    if getattr(orig_encode, "__iamccs_ltx2_autofix__", False):
        return

    def _round_up_8n1(frames: int) -> int:
        frames = int(frames)
        if frames <= 1:
            return 1
        rem = (frames - 1) % 8
        if rem == 0:
            return frames
        return frames + (8 - rem)

    def _is_valid_8n1(frames: int) -> bool:
        frames = int(frames)
        return frames >= 1 and (frames - 1) % 8 == 0

    def _pad_repeat_last(x: "torch.Tensor", dim: int, pad: int) -> "torch.Tensor":
        # Take last slice along `dim` (keeps dimension) and repeat it `pad` times.
        slc = [slice(None)] * x.ndim
        slc[dim] = slice(-1, None)
        last = x[tuple(slc)]
        reps = [1] * x.ndim
        reps[dim] = int(pad)
        last_rep = last.repeat(*reps)
        return torch.cat([x, last_rep], dim=dim)

    def _candidate_frame_dims(x: "torch.Tensor") -> list[int]:
        # Most common layouts:
        # - (B, C, T, H, W)  -> frames dim = 2
        # - (T, H, W, C)     -> frames dim = 0 (ComfyUI IMAGE batches)
        # We only try dims that are >1 and *not obviously channels*.
        dims: list[int] = []
        if x.ndim == 5:
            # Prefer T, then fallbacks
            dims = [2, 0, 1]
        elif x.ndim == 4:
            dims = [0]
        else:
            dims = [0]

        out: list[int] = []
        for d in dims:
            try:
                size = int(x.shape[d])
            except Exception:
                continue
            if size <= 1:
                continue
            # Heuristic: channels are usually small (1..4). Don't treat that as frames.
            if size in (1, 2, 3, 4) and x.ndim >= 4 and d in (1, 3):
                continue
            out.append(d)
        # Ensure uniqueness, preserve order
        seen = set()
        unique: list[int] = []
        for d in out:
            if d in seen:
                continue
            seen.add(d)
            unique.append(d)
        return unique

    def encode_patched(self, pixels_in: "torch.Tensor"):
        try:
            return orig_encode(self, pixels_in)
        except ValueError as e:
            msg = str(e)
            if "Invalid number of frames" not in msg:
                raise

            if not isinstance(pixels_in, torch.Tensor) or pixels_in.ndim < 4:
                raise

            # Try padding along the most likely frame dimension(s).
            last_err: Exception | None = e
            for dim in _candidate_frame_dims(pixels_in):
                frames_in = int(pixels_in.shape[dim])
                if _is_valid_8n1(frames_in):
                    continue

                frames_fixed = _round_up_8n1(frames_in)
                pad = frames_fixed - frames_in
                if pad <= 0:
                    continue

                try:
                    pixels_fixed = _pad_repeat_last(pixels_in, dim=dim, pad=pad)
                    log.warning(
                        "[LTX2 VAE encode autofix] Padded frames dim=%d %d -> %d (pad=%d) to satisfy 1+8*x rule",
                        dim,
                        frames_in,
                        frames_fixed,
                        pad,
                    )
                    return orig_encode(self, pixels_fixed)
                except Exception as ee:
                    last_err = ee
                    continue

            # If all attempts failed, re-raise the original ValueError.
            raise e

    encode_patched.__iamccs_ltx2_autofix__ = True
    setattr(cls, "encode", encode_patched)


_iamccs_install_ltx2_vae_encode_autofix()
