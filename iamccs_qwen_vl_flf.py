# ==========================================================
# iamccs_qwen_vl_flf.py — IAMCCS QwenVL First/Last Frame
# ==========================================================
# Dual-image QwenVL node: accepts a FIRST FRAME and a LAST FRAME,
# then queries QwenVL to describe the motion/action occurring
# between the two frames — the ideal prompt for FLF video generators
# (WAN SVI Pro, LTX-2 FLF, etc.).
#
# This is a 1:1 extension of AILab_QwenVL (ComfyUI-QwenVL)
# with the image input replaced by two independent IMAGE inputs.
#
# Author : IAMCCS (carminecristalloscalzi.com / patreon.com/IAMCCS)
# License: GPL-3.0
# ==========================================================

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Dynamic import of QwenVLBase from the ComfyUI-QwenVL custom node
# ---------------------------------------------------------------------------

def _import_qwen_base():
    """Locate and import QwenVLBase from ComfyUI-QwenVL, however it was loaded."""

    # 1) Already loaded by ComfyUI's module system?
    for module_name, module in sys.modules.items():
        if "AILab_QwenVL" in module_name:
            if hasattr(module, "QwenVLBase"):
                return module.QwenVLBase

    # 2) Look at sibling custom_node directories
    this_dir = Path(__file__).resolve().parent            # …/IAMCCS-nodes
    custom_nodes_dir = this_dir.parent                    # …/custom_nodes

    candidates = [
        custom_nodes_dir / "ComfyUI-QwenVL" / "AILab_QwenVL.py",
        custom_nodes_dir / "comfyui-qwenvl" / "AILab_QwenVL.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("AILab_QwenVL_ext", str(candidate))
            mod  = importlib.util.module_from_spec(spec)
            sys.modules["AILab_QwenVL_ext"] = mod
            spec.loader.exec_module(mod)
            return mod.QwenVLBase

    raise ImportError(
        "[IAMCCS_QWEN_VL_FLF] Cannot find QwenVLBase. "
        "Make sure ComfyUI-QwenVL is installed under custom_nodes/ComfyUI-QwenVL."
    )


# Lazy-load so the import error is surfaced only when the node is used
_QwenVLBase = None

def _get_base():
    global _QwenVLBase
    if _QwenVLBase is None:
        _QwenVLBase = _import_qwen_base()
    return _QwenVLBase


# ---------------------------------------------------------------------------
# FLF-specific prompt presets
# ---------------------------------------------------------------------------

FLF_PRESET_PROMPTS = [
    "🎬 Video Action Description (FLF)",
    "🎥 Cinematic Motion Prompt (FLF)",
    "🏃 Subject Movement & Camera (FLF)",
    "🌀 Scene Transition Description (FLF)",
    "📷 Static Shot Action Prompt (FLF)",
    "🌊 WAN 2.2 SVI Pro 2 — FLF Prompt",
    "⚡ LTX-2 FLF Prompt",
    "⚡ LTX-2 FLF Prompt + User Description",
]

FLF_SYSTEM_PROMPTS = {
    "🎬 Video Action Description (FLF)": (
        "You are given two images: the FIRST FRAME and the LAST FRAME of a video clip. "
        "Your task is to write a single, concise video-generation prompt (2-4 sentences) that describes "
        "the motion, action, and visual transformation occurring between these two frames. "
        "Include: subject actions, camera movement (pan, tilt, zoom, static, etc.), environmental changes, "
        "lighting shifts, and any notable visual effects. "
        "Write in present tense, imperative style, as if directing an AI video generator. "
        "Do NOT describe what is in the images statically — focus entirely on the MOTION and TRANSITION."
    ),
    "🎥 Cinematic Motion Prompt (FLF)": (
        "You are given the FIRST FRAME and the LAST FRAME of a cinematic video shot. "
        "Describe the complete camera movement and subject action as a professional cinematography prompt. "
        "Include: shot type (close-up, wide, medium), camera movement (dolly, pan, handheld shake, etc.), "
        "subject movement direction and speed, focus changes, and mood/lighting evolution. "
        "Output a single fluid paragraph suitable for an AI video generator."
    ),
    "🏃 Subject Movement & Camera (FLF)": (
        "Compare the first frame and the last frame provided. "
        "Write a detailed motion description focused on: "
        "1) How the main subject(s) move between the two frames (direction, speed, posture changes), "
        "2) Camera behavior (static, following, pulling back, zooming in/out), "
        "3) Background/environment changes. "
        "Summarize in 2-3 sentences optimized for AI video generation input."
    ),
    "🌀 Scene Transition Description (FLF)": (
        "You are shown the opening frame and the closing frame of a video sequence. "
        "Analyse the differences and infer what visual narrative connects them. "
        "Write a prompt that describes the scene transition: object positions, lighting evolution, "
        "atmospheric changes, and any implied motion. Be specific and concise (2-3 sentences). "
        "The output should work as a direct input for an AI video generator."
    ),
    "📷 Static Shot Action Prompt (FLF)": (
        "Given the first and last frame of a static-camera video clip, "
        "describe only the subject's actions and movements within the fixed frame. "
        "Mention entry/exit directions, gestures, expressions, interaction with objects, "
        "and any notable background activity. "
        "Output a crisp 1-3 sentence prompt for an AI video generator."
    ),

    "🌊 WAN 2.2 SVI Pro 2 — FLF Prompt": (
        "You are an AI video prompt expert for the WAN 2.2 SVI Pro 2 model in ComfyUI. "
        "I will give you two images: the FIRST FRAME and the LAST FRAME of a video clip. "
        "Your job is to write one single, detailed prompt in clear English "
        "that describes the motion and transformation occurring between these two frames, "
        "suitable for use directly with WAN 2.2 SVI Pro 2. "
        "Rules: "
        "Write normal sentences, not JSON, not a list. "
        "Include: subject action and movement, camera motion (pan, tilt, zoom, dolly, static), "
        "environment and background evolution, lighting and atmosphere changes, "
        "and overall motion style (slow, fast, smooth, handheld). "
        "Focus entirely on the MOTION and TRANSITION between the two frames — "
        "do NOT describe the frames as static images. "
        "Keep it under 4 sentences. "
        "Do not mention these rules in your answer."
    ),

    "⚡ LTX-2 FLF Prompt": (
        "You are an AI video prompt expert for the LTX-2 First/Last Frame (FLF) model in ComfyUI. "
        "I will give you two images: the FIRST FRAME and the LAST FRAME of a video clip. "
        "Your job is to write one single, detailed prompt in clear English "
        "that describes the visual and motion continuity connecting these two frames, "
        "optimised for LTX-2 FLF video generation. "
        "Rules: "
        "Write normal sentences, not JSON, not a list. "
        "Include: subject description and action, precise camera movement, "
        "spatial transitions (near-to-far, left-to-right, etc.), "
        "lighting and color mood evolution between frames, "
        "and motion speed/smoothness (e.g. slow drift, rapid motion, gradual zoom). "
        "LTX-2 responds best to prompts that are visually rich and temporally explicit — "
        "describe what changes and how it changes, not just what is visible. "
        "Keep it under 4 sentences. "
        "Do not mention these rules in your answer."
    ),

    "⚡ LTX-2 FLF Prompt + User Description": (
        "You are an AI video prompt expert for the LTX-2 / LTXV model in ComfyUI. "
        "I will give you two images: the FIRST FRAME and the LAST FRAME of a video clip, "
        "plus a short user description at the bottom of this message. "
        "Your job is to write one single, detailed prompt in clear English — a single flowing paragraph — "
        "that describes the motion and transformation occurring between these two frames, "
        "suitable for direct use with LTX-2 FLF video generation. "
        "Rules: "
        "- Write a single continuous paragraph, not JSON, not a list. "
        "- Start directly with the main action in the first sentence, without any preamble. "
        "- Describe things chronologically: what starts, what happens, what ends. "
        "- Include: subject action and movement, precise character or object appearances, "
        "camera angle and movement (pan, tilt, dolly, zoom, static, handheld), "
        "environment and background evolution, lighting and color changes, "
        "and overall motion style (slow, fast, smooth, handheld, etc.). "
        "- Be literal and precise, like a cinematographer describing a shot list in flowing prose. "
        "- Focus entirely on the MOTION and TRANSITION between the two frames — "
        "do NOT describe the frames as static images. "
        "- Keep it under 200 words — ideally 3 to 4 rich, dense sentences. "
        "- Do not mention these rules in your answer. "
        "\n\nUser description:"
    ),
}

FLF_TOOLTIPS = {
    "first_frame":    "The opening frame of the video clip (frame 0).",
    "last_frame":     "The closing frame of the video clip (last frame).",
    "preset_prompt":  "Built-in FLF instruction set for QwenVL. Each preset focuses on a different aspect of motion description.",
    "custom_prompt":  "Optional override — replaces the preset completely when filled in.",
    "model_name":     "Pick the Qwen-VL checkpoint. First run downloads weights into models/LLM/Qwen-VL.",
    "quantization":   "Precision vs VRAM. FP16 = best quality; 8-bit = 8-16 GB GPUs; 4-bit = 6 GB or lower.",
    "attention_mode": "auto tries SageAttention / Flash-Attn v2 and falls back to SDPA.",
    "max_tokens":     "Maximum tokens to generate. 256-512 is usually sufficient for motion prompts.",
    "keep_model_loaded": "Keep model in VRAM after generation to skip reloading on next run.",
    "seed":           "Random seed — reuse to reproduce the same description.",
    "use_torch_compile": "Enable torch.compile (reduce-overhead) on supported CUDA/Torch 2.1+ builds.",
    "device":         "Inference device: auto, cpu, mps, or cuda:N.",
    "temperature":    "Sampling randomness (when num_beams=1). 0.2-0.4 focused, 0.7+ creative.",
    "top_p":          "Nucleus sampling cutoff (when num_beams=1).",
    "num_beams":      "Beam-search width. >1 disables temperature/top_p for more stable output.",
    "repetition_penalty": "Values >1 penalise repeated phrases (1.1-1.3 recommended).",
}


# ---------------------------------------------------------------------------
# FLF mixin — overrides generate() to accept two frames
# ---------------------------------------------------------------------------

class _FLFMixin:
    """Mixin that provides dual-image (first/last frame) generation."""

    @staticmethod
    def tensor_to_pil(tensor):
        if tensor is None:
            return None
        if tensor.dim() == 4:
            tensor = tensor[0]
        array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        from PIL import Image
        return Image.fromarray(array)

    @torch.no_grad()
    def generate_flf(
        self,
        prompt_text,
        first_frame,
        last_frame,
        max_tokens,
        temperature,
        top_p,
        num_beams,
        repetition_penalty,
    ):
        """Build a two-image conversation: [first_frame, last_frame, text prompt]."""
        content = []

        img1 = self.tensor_to_pil(first_frame)
        img2 = self.tensor_to_pil(last_frame)

        if img1 is not None:
            content.append({"type": "image", "image": img1})
        if img2 is not None:
            content.append({"type": "image", "image": img2})

        content.append({"type": "text", "text": prompt_text})

        conversation = [{"role": "user", "content": content}]

        chat = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        images = [item["image"] for item in content if item["type"] == "image"]
        processed = self.processor(
            text=chat,
            images=images or None,
            videos=None,
            return_tensors="pt",
        )

        model_device = next(self.model.parameters()).device
        model_inputs = {
            k: v.to(model_device) if torch.is_tensor(v) else v
            for k, v in processed.items()
        }

        stop_tokens = [self.tokenizer.eos_token_id]
        if hasattr(self.tokenizer, "eot_id") and self.tokenizer.eot_id is not None:
            stop_tokens.append(self.tokenizer.eot_id)

        kwargs = {
            "max_new_tokens":      max_tokens,
            "repetition_penalty":  repetition_penalty,
            "num_beams":           num_beams,
            "eos_token_id":        stop_tokens,
            "pad_token_id":        self.tokenizer.pad_token_id,
        }
        if num_beams == 1:
            kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
        else:
            kwargs["do_sample"] = False

        outputs = self.model.generate(**model_inputs, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        input_len = model_inputs["input_ids"].shape[-1]
        text = self.tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)
        return text.strip()

    def run_flf(
        self,
        model_name,
        quantization,
        preset_prompt,
        custom_prompt,
        first_frame,
        last_frame,
        max_tokens,
        temperature,
        top_p,
        num_beams,
        repetition_penalty,
        seed,
        keep_model_loaded,
        attention_mode,
        use_torch_compile,
        device,
    ):
        from comfy.utils import ProgressBar
        pbar = ProgressBar(3)

        torch.manual_seed(seed)
        prompt = FLF_SYSTEM_PROMPTS.get(preset_prompt, preset_prompt)
        if custom_prompt and custom_prompt.strip():
            prompt = custom_prompt.strip()

        pbar.update_absolute(1, 3, None)

        self.load_model(
            model_name,
            quantization,
            attention_mode,
            use_torch_compile,
            device,
            keep_model_loaded,
        )

        pbar.update_absolute(2, 3, None)

        try:
            text = self.generate_flf(
                prompt,
                first_frame,
                last_frame,
                max_tokens,
                temperature,
                top_p,
                num_beams,
                repetition_penalty,
            )
            pbar.update_absolute(3, 3, None)
            return (text,)
        finally:
            if not keep_model_loaded:
                self.clear()


# ---------------------------------------------------------------------------
# Node class factory (deferred because QwenVLBase is lazy-loaded)
# ---------------------------------------------------------------------------

def _build_node_classes():
    """Return (IAMCCS_QWEN_VL_FLF, IAMCCS_QWEN_VL_FLF_Advanced) after QwenVLBase loads."""
    Base = _get_base()

    # Import Quantization enum from the same module as Base
    import sys
    qwen_mod = sys.modules.get("AILab_QwenVL") or sys.modules.get("AILab_QwenVL_ext")
    if qwen_mod is None:
        # The module might be registered under a different key
        for k, v in sys.modules.items():
            if "AILab_QwenVL" in k and hasattr(v, "Quantization"):
                qwen_mod = v
                break
    if qwen_mod is None:
        raise ImportError("[IAMCCS_QWEN_VL_FLF] Could not locate Quantization enum in QwenVL module.")

    Quantization   = qwen_mod.Quantization
    ATTENTION_MODES = qwen_mod.ATTENTION_MODES
    HF_VL_MODELS   = qwen_mod.HF_VL_MODELS

    # ------------------------------------------------------------------
    # Simple version
    # ------------------------------------------------------------------
    class IAMCCS_QWEN_VL_FLF(_FLFMixin, Base):
        """QwenVL node with FIRST FRAME + LAST FRAME inputs for FLF video generation."""

        @classmethod
        def INPUT_TYPES(cls):
            # Refresh model list at call time (models may be downloaded after startup)
            models = list(HF_VL_MODELS.keys())
            default_model = models[0] if models else "Qwen2.5-VL-3B-Instruct"
            default_prompt = (
                "🎬 Video Action Description (FLF)"
                if "🎬 Video Action Description (FLF)" in FLF_PRESET_PROMPTS
                else FLF_PRESET_PROMPTS[0]
            )
            return {
                "required": {
                    "model_name":   (models, {"default": default_model,         "tooltip": FLF_TOOLTIPS["model_name"]}),
                    "quantization": (Quantization.get_values(), {"default": Quantization.FP16.value, "tooltip": FLF_TOOLTIPS["quantization"]}),
                    "attention_mode": (ATTENTION_MODES, {"default": "auto",     "tooltip": FLF_TOOLTIPS["attention_mode"]}),
                    "preset_prompt":  (FLF_PRESET_PROMPTS, {"default": default_prompt, "tooltip": FLF_TOOLTIPS["preset_prompt"]}),
                    "custom_prompt":  ("STRING",  {"default": "", "multiline": True, "tooltip": FLF_TOOLTIPS["custom_prompt"]}),
                    "max_tokens":     ("INT",     {"default": 384, "min": 64, "max": 2048, "tooltip": FLF_TOOLTIPS["max_tokens"]}),
                    "keep_model_loaded": ("BOOLEAN", {"default": True,          "tooltip": FLF_TOOLTIPS["keep_model_loaded"]}),
                    "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1, "tooltip": FLF_TOOLTIPS["seed"]}),
                },
                "optional": {
                    "first_frame": ("IMAGE", {"tooltip": FLF_TOOLTIPS["first_frame"]}),
                    "last_frame":  ("IMAGE", {"tooltip": FLF_TOOLTIPS["last_frame"]}),
                },
            }

        RETURN_TYPES  = ("STRING",)
        RETURN_NAMES  = ("FLF_PROMPT",)
        FUNCTION      = "process"
        CATEGORY      = "IAMCCS/QwenVL"
        DESCRIPTION   = (
            "Uses QwenVL to analyse the FIRST and LAST frame of a video clip "
            "and generate a motion/action description prompt for FLF video generators "
            "(WAN SVI Pro, LTX-2 FLF, Wan2.1 i2v, etc.)."
        )

        def process(
            self,
            model_name,
            quantization,
            attention_mode,
            preset_prompt,
            custom_prompt,
            max_tokens,
            keep_model_loaded,
            seed,
            first_frame=None,
            last_frame=None,
        ):
            return self.run_flf(
                model_name, quantization, preset_prompt, custom_prompt,
                first_frame, last_frame,
                max_tokens,
                temperature=0.6, top_p=0.9, num_beams=1,
                repetition_penalty=1.2,
                seed=seed,
                keep_model_loaded=keep_model_loaded,
                attention_mode=attention_mode,
                use_torch_compile=False,
                device="auto",
            )

    # ------------------------------------------------------------------
    # Advanced version
    # ------------------------------------------------------------------
    class IAMCCS_QWEN_VL_FLF_Advanced(_FLFMixin, Base):
        """Advanced version of IAMCCS_QWEN_VL_FLF with full parameter control."""

        @classmethod
        def INPUT_TYPES(cls):
            models = list(HF_VL_MODELS.keys())
            default_model = models[0] if models else "Qwen2.5-VL-3B-Instruct"
            default_prompt = (
                "🎬 Video Action Description (FLF)"
                if "🎬 Video Action Description (FLF)" in FLF_PRESET_PROMPTS
                else FLF_PRESET_PROMPTS[0]
            )

            num_gpus = torch.cuda.device_count()
            gpu_list = [f"cuda:{i}" for i in range(num_gpus)]
            device_options = ["auto", "cpu", "mps"] + gpu_list

            return {
                "required": {
                    "model_name":       (models, {"default": default_model,         "tooltip": FLF_TOOLTIPS["model_name"]}),
                    "quantization":     (Quantization.get_values(), {"default": Quantization.FP16.value, "tooltip": FLF_TOOLTIPS["quantization"]}),
                    "attention_mode":   (ATTENTION_MODES, {"default": "auto",         "tooltip": FLF_TOOLTIPS["attention_mode"]}),
                    "use_torch_compile":("BOOLEAN", {"default": False,                "tooltip": FLF_TOOLTIPS["use_torch_compile"]}),
                    "device":           (device_options, {"default": "auto",          "tooltip": FLF_TOOLTIPS["device"]}),
                    "preset_prompt":    (FLF_PRESET_PROMPTS, {"default": default_prompt, "tooltip": FLF_TOOLTIPS["preset_prompt"]}),
                    "custom_prompt":    ("STRING",  {"default": "", "multiline": True, "tooltip": FLF_TOOLTIPS["custom_prompt"]}),
                    "max_tokens":       ("INT",     {"default": 512,  "min": 64, "max": 4096, "tooltip": FLF_TOOLTIPS["max_tokens"]}),
                    "temperature":      ("FLOAT",   {"default": 0.6,  "min": 0.1, "max": 1.0, "step": 0.05, "tooltip": FLF_TOOLTIPS["temperature"]}),
                    "top_p":            ("FLOAT",   {"default": 0.9,  "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": FLF_TOOLTIPS["top_p"]}),
                    "num_beams":        ("INT",     {"default": 1,    "min": 1, "max": 8,     "tooltip": FLF_TOOLTIPS["num_beams"]}),
                    "repetition_penalty": ("FLOAT", {"default": 1.2,  "min": 0.5, "max": 2.0, "step": 0.05, "tooltip": FLF_TOOLTIPS["repetition_penalty"]}),
                    "keep_model_loaded":("BOOLEAN", {"default": True,              "tooltip": FLF_TOOLTIPS["keep_model_loaded"]}),
                    "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1,   "tooltip": FLF_TOOLTIPS["seed"]}),
                },
                "optional": {
                    "first_frame": ("IMAGE", {"tooltip": FLF_TOOLTIPS["first_frame"]}),
                    "last_frame":  ("IMAGE", {"tooltip": FLF_TOOLTIPS["last_frame"]}),
                },
            }

        RETURN_TYPES  = ("STRING",)
        RETURN_NAMES  = ("FLF_PROMPT",)
        FUNCTION      = "process"
        CATEGORY      = "IAMCCS/QwenVL"
        DESCRIPTION   = (
            "Advanced version of IAMCCS QwenVL FLF node with full control over "
            "generation parameters. Accepts FIRST FRAME + LAST FRAME and outputs "
            "an action/motion description prompt for AI video generators."
        )

        def process(
            self,
            model_name,
            quantization,
            attention_mode,
            use_torch_compile,
            device,
            preset_prompt,
            custom_prompt,
            max_tokens,
            temperature,
            top_p,
            num_beams,
            repetition_penalty,
            keep_model_loaded,
            seed,
            first_frame=None,
            last_frame=None,
        ):
            return self.run_flf(
                model_name, quantization, preset_prompt, custom_prompt,
                first_frame, last_frame,
                max_tokens, temperature, top_p, num_beams, repetition_penalty,
                seed, keep_model_loaded, attention_mode, use_torch_compile, device,
            )

    return IAMCCS_QWEN_VL_FLF, IAMCCS_QWEN_VL_FLF_Advanced


# ---------------------------------------------------------------------------
# Module-level instantiation (deferred, with graceful fallback)
# ---------------------------------------------------------------------------

try:
    IAMCCS_QWEN_VL_FLF, IAMCCS_QWEN_VL_FLF_Advanced = _build_node_classes()

    NODE_CLASS_MAPPINGS = {
        "IAMCCS_QWEN_VL_FLF":          IAMCCS_QWEN_VL_FLF,
        "IAMCCS_QWEN_VL_FLF_Advanced": IAMCCS_QWEN_VL_FLF_Advanced,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "IAMCCS_QWEN_VL_FLF":          "QwenVL FLF — First/Last Frame Prompt 🎬",
        "IAMCCS_QWEN_VL_FLF_Advanced": "QwenVL FLF — First/Last Frame Prompt (Advanced) 🎬",
    }

    print("[IAMCCS] IAMCCS_QWEN_VL_FLF nodes loaded OK")

except Exception as _err:
    print(f"[IAMCCS] WARNING: IAMCCS_QWEN_VL_FLF could not load — {_err}")
    print("[IAMCCS]   Make sure ComfyUI-QwenVL is installed in custom_nodes/ComfyUI-QwenVL")

    NODE_CLASS_MAPPINGS        = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    IAMCCS_QWEN_VL_FLF          = None
    IAMCCS_QWEN_VL_FLF_Advanced = None
