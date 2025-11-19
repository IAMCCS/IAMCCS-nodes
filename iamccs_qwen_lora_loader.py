"""
IAMCCS Qwen Image LoRA Loader/Stack
A Nunchaku Qwen Image LoRA loader node
compatible with current `nunchaku` where Qwen image transformer is aliased
as `NunchakuSanaTransformer2DModel`.
"""

import copy
import logging
import os
import sys

# Support both old/new nunchaku class names (alias to a common name)
try:
    from nunchaku import NunchakuQwenImageTransformer2DModel
except Exception:
    from nunchaku import NunchakuSanaTransformer2DModel as NunchakuQwenImageTransformer2DModel

import folder_paths

# Logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _get_wrappers_module():
    """Dynamically load wrappers.qwenimage from ComfyUI-QwenImageLoraLoader."""
    import importlib.util
    # Try to locate the sibling custom node folder
    # typical structure: .../custom_nodes/ComfyUI-QwenImageLoraLoader
    base_custom_nodes = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    qwen_node_dir = os.path.join(base_custom_nodes, "ComfyUI-QwenImageLoraLoader")

    # Fallback: if running outside expected layout, try current sys.path entries
    candidate_dirs = [qwen_node_dir] + [p for p in sys.path if isinstance(p, str) and p.endswith("ComfyUI-QwenImageLoraLoader")]

    wrappers_path = None
    for d in candidate_dirs:
        wp = os.path.join(d, "wrappers", "qwenimage.py")
        if os.path.exists(wp):
            wrappers_path = wp
            break

    if not wrappers_path:
        raise ImportError("Cannot locate ComfyUI-QwenImageLoraLoader/wrappers/qwenimage.py")

    spec = importlib.util.spec_from_file_location("wrappers.qwenimage", wrappers_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec for {wrappers_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


class IAMCCS_QwenImageLoraLoader:
    """
    Load and apply a single LoRA to a Nunchaku Qwen Image model.
    """
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """ComfyUI calls IS_CHANGED with widget values only; avoid positional args warnings.
        Build a hash from model reference (stringified) and LoRA parameters.
        """
        import hashlib
        m = hashlib.sha256()
        model = kwargs.get("model")
        if model is not None:
            m.update(str(model).encode())
        lora_name = kwargs.get("lora_name", "")
        m.update(lora_name.encode())
        preset = str(kwargs.get("lora_strength_preset", "1.00"))
        m.update(preset.encode())
        return m.hexdigest()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": "The diffusion model the LoRA will be applied to. Make sure the model is loaded by a Nunchaku Qwen Image loader.",
                    },
                ),
                "lora_name": (
                    folder_paths.get_filename_list("loras"),
                    {"tooltip": "The file name of the LoRA."},
                ),
                "lora_strength_preset": (
                    ["0.25", "0.50", "0.76", "1.00", "1.25", "1.50"],
                    {
                        "default": "1.00",
                        "tooltip": "Preset strength values.",
                    },
                ),
                "composition_mode": (
                    ["append", "merge_v2"],
                    {
                        "default": "merge_v2",
                        "tooltip": "How to apply LoRAs: 'append' (original behavior, may change shapes) or 'merge_v2' (in-place delta, no rank expansion).",
                    },
                ),
                "offload_policy": (
                    ["rebuild", "disable"],
                    {
                        "default": "rebuild",
                        "tooltip": "CPU offload with LoRAs: 'rebuild' re-enables offload after composing; 'disable' keeps it off while LoRAs are active.",
                    },
                ),
                "offload_num_blocks_on_gpu": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 64,
                        "step": 1,
                        "tooltip": "How many transformer blocks stay on GPU when offload is enabled (higher = more VRAM, more speed).",
                    },
                ),
                "offload_use_pin_memory": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Use pinned host memory for offload transfers (can improve bandwidth, uses more system RAM).",
                    },
                ),
                "offload_auto_tune": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Automatically choose offload settings based on free VRAM (overrides saved settings).",
                    },
                ),
                "vram_margin_gb": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 0.0,
                        "max": 16.0,
                        "step": 0.25,
                        "tooltip": "VRAM margin used when cpu_offload_setting='auto' to decide enabling offload for composition.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("The modified diffusion model.",)
    FUNCTION = "load_lora"
    TITLE = "IAMCCS QwenImgLoraLoaderFix"
    CATEGORY = "IAMCCS/Nunchaku"
    DESCRIPTION = "Apply a single LoRA to a Nunchaku Qwen Image model."

    def load_lora(self, model, lora_name: str, lora_strength_preset: str = "1.00", composition_mode: str = "merge_v2", offload_policy: str = "rebuild", offload_num_blocks_on_gpu: int = 1, offload_use_pin_memory: bool = False, offload_auto_tune: bool = True, vram_margin_gb: float = 4.0):
        # Resolve effective strength from preset or default
        import math
        strength_val: float = 1.0
        try:
            strength_val = float(lora_strength_preset)
        except Exception:
            strength_val = 1.0
        if not math.isfinite(strength_val):
            strength_val = 1.0
        if abs(strength_val) < 1e-5:
            return (model,)

        # Coerce offload_num_blocks_on_gpu to a sane integer (avoid NaN/None/inf)
        import math as _math
        try:
            _tmp_val = float(offload_num_blocks_on_gpu)
            if not _math.isfinite(_tmp_val):
                _tmp_val = 1.0
            if _tmp_val < 1:
                _tmp_val = 1.0
            if _tmp_val > 64:
                _tmp_val = 64.0
            offload_num_blocks_on_gpu = int(_tmp_val)
        except Exception:
            offload_num_blocks_on_gpu = 1

        # Advanced toggle removed: always respect user-provided widget values

        model_wrapper = model.model.diffusion_model

        wrappers_module = _get_wrappers_module()
        ComfyQwenImageWrapper = wrappers_module.ComfyQwenImageWrapper

        # Debug logging
        model_wrapper_type_name = type(model_wrapper).__name__
        model_wrapper_module = type(model_wrapper).__module__
        logger.info(f"🔍 Model wrapper type: '{model_wrapper_type_name}'")
        logger.info(f"🔍 Model wrapper module: {model_wrapper_module}")

        if hasattr(model_wrapper, 'model') and hasattr(model_wrapper, 'loras'):
            logger.info("✅ Model is already wrapped (detected via attributes)")
            transformer = model_wrapper.model
        elif (
            model_wrapper_type_name in ("NunchakuQwenImageTransformer2DModel", "NunchakuSanaTransformer2DModel")
            or model_wrapper_type_name.endswith("NunchakuQwenImageTransformer2DModel")
            or model_wrapper_type_name.endswith("NunchakuSanaTransformer2DModel")
        ):
            logger.info("🔧 Wrapping Nunchaku*Qwen/Sana* Transformer with ComfyQwenImageWrapper")
            wrapped_model = ComfyQwenImageWrapper(
                model_wrapper,
                getattr(model_wrapper, 'config', {}),
                None,
                {},
                "auto",
                vram_margin_gb,
                lora_offload_policy=offload_policy,
                offload_num_blocks_on_gpu=offload_num_blocks_on_gpu,
                offload_use_pin_memory=offload_use_pin_memory,
                offload_auto_tune=offload_auto_tune,
            )
            # Forward composition mode flag (attribute-based to avoid strict kwargs requirements)
            try:
                setattr(wrapped_model, "lora_composition_mode", composition_mode)
            except Exception:
                pass
            model.model.diffusion_model = wrapped_model
            model_wrapper = wrapped_model
            transformer = model_wrapper.model
        else:
            logger.error(f"❌ Model type mismatch! Type: {model_wrapper_type_name}, Module: {model_wrapper_module}")
            raise TypeError(
                f"This LoRA loader works with Nunchaku Qwen Image models; got {model_wrapper_type_name}."
            )

        # Remove expensive deepcopy (caused device divergence under offload); mutate in place
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        try:
            setattr(model_wrapper, "lora_composition_mode", composition_mode)
        except Exception:
            pass
        model_wrapper.loras.append((lora_path, strength_val))

        # Ensure wrapper model resides fully on the original device (avoid mixed cpu/cuda modules)
        try:
            target_device = next(transformer.parameters()).device
            if target_device.type == "cuda":
                for p in transformer.parameters():
                    if p.device != target_device:
                        p.data = p.data.to(target_device)
                for b in transformer.buffers():
                    if b.device != target_device:
                        b.data = b.data.to(target_device)
        except Exception:
            pass

        logger.info(f"LoRA added: {lora_name} (strength={strength_val})")
        return (model,)


