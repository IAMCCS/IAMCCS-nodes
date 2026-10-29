# SPDX-License-Identifier: GPL-3.0-or-later
"""Settings bridge for the stock-node Open-H3 delivery branch."""

import folder_paths


PREFIX = "h3_open_refine_"
SIGMAS = {
    "gentle_3step": "0.8, 0.6857142857, 0.48, 0",
    "reference_3step": "0.909375, 0.857143, 0.666667, 0",
}


def settings_schema():
    loras = ["", *folder_paths.get_filename_list("loras")]
    upscalers = ["", *folder_paths.get_filename_list("latent_upscale_models")] if "latent_upscale_models" in folder_paths.folder_names_and_paths else [""]
    refiner = next((name for name in loras if name.lower().endswith("minimax_h3_open_regenerate_2k_refiner_rank_384_bf16.safetensors")), "")
    distill = next((name for name in loras if name.lower().endswith("minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors")), "")
    upscaler = next((name for name in upscalers if name.lower().endswith("minimax_h3_latent_upscaler_3d_fp16.safetensors")), "")
    fields = {
        "enabled": ("BOOLEAN", {"default": False, "display_name": "OPEN-H3 · ENABLE REFINEMENT", "tooltip": "One native generation followed by Open-H3. Replaces the other upscale routes; requires the Open-H3 workflow branch."}),
        "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.05, "display_name": "OPEN-H3 · SPATIAL SCALE", "tooltip": "Learned latent upscale. Output follows the native aspect ratio, rounded to the H3 grid. Full-resolution refinement can exceed low-VRAM capacity."}),
        "model_name": (upscalers, {"default": upscaler, "display_name": "OPEN-H3 · LATENT UPSCALER"}),
        "precision": (["fp16", "bf16", "fp32"], {"default": "fp16", "display_name": "OPEN-H3 · UPSCALER PRECISION"}),
        "device": (["cuda", "rocm", "cpu"], {"default": "cuda", "display_name": "OPEN-H3 · UPSCALER DEVICE"}),
        "temporal_chunking": ("BOOLEAN", {"default": True, "display_name": "OPEN-H3 · UPSCALER CHUNKING", "tooltip": "Chunks the latent upscaler only. The three-step H3 sampler still processes the complete target latent."}),
        "lora_name": (loras, {"default": refiner, "display_name": "OPEN-H3 · REFINER LORA"}),
        "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "display_name": "OPEN-H3 · REFINER STRENGTH"}),
        "distill_lora_name": (loras, {"default": distill, "display_name": "OPEN-H3 · DISTILLATION LORA", "tooltip": "Kijai's rank-384 file contains only the refiner. The supplied community recipe pairs it with LightX2V v0.1; it is not NVIDIA's complete rank-576 bundle. Leave blank when the selected refiner already includes distillation."}),
        "distill_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "display_name": "OPEN-H3 · DISTILLATION STRENGTH"}),
        "sigma_preset": (["gentle_3step", "reference_3step", "custom"], {"default": "gentle_3step", "display_name": "OPEN-H3 · NOISE SCHEDULE", "tooltip": "Gentle starts at 0.8. Reference starts at 0.909375 and can change identity or performance. Neither preset guarantees preservation."}),
        "custom_sigmas": ("STRING", {"default": SIGMAS["gentle_3step"], "multiline": False, "display_name": "OPEN-H3 · CUSTOM SIGMAS", "tooltip": "Used only with Custom. Comma-separated Euler sigmas ending in zero."}),
        "seed": ("INT", {"default": 777, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": False, "display_name": "OPEN-H3 · REFINE SEED"}),
        "prompt": ("STRING", {"default": "", "multiline": True, "display_name": "OPEN-H3 · REFINE PROMPT", "tooltip": "Blank uses the current shot prompt. Describe the existing action and appearance; new instructions can redraw the performance."}),
    }
    return {PREFIX + name: spec for name, spec in fields.items()}


def configuration(settings):
    return {name[len(PREFIX):]: settings.get(name, spec[1]["default"]) for name, spec in settings_schema().items()}


class IAMCCS_MiniMaxH3OpenRefineControl:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"cine_linx": ("IAMCCS_SUPERNODE_LINX",), "segment_index": ("INT", {"default": 0, "min": 0, "forceInput": True})}}

    # File names feed both legacy list-combos and V3 COMBO sockets.
    RETURN_TYPES = ("BOOLEAN", "STRING", "FLOAT", "*", "COMBO", "COMBO", "BOOLEAN", "*", "FLOAT", "*", "FLOAT", "STRING", "INT", "BOOLEAN")
    RETURN_NAMES = ("enabled", "prompt", "scale", "upscaler_name", "precision", "device", "temporal_chunking", "refiner_lora", "refiner_strength", "distill_lora", "distill_strength", "sigmas", "seed", "use_face_source")
    FUNCTION = "resolve"
    CATEGORY = "IAMCCS/MiniMax H3/Delivery"

    def resolve(self, cine_linx, segment_index):
        # Atomic imports Shotboard, which imports this module's settings schema.
        from .iamccs_minimax_h3_atomic_backend import _resolve_shotplan

        plan = _resolve_shotplan(cine_linx)
        cfg = plan.get("open_h3_refine") or configuration({})
        enabled = bool(cfg["enabled"])
        if enabled and bool(plan.get("upscale_enabled")) and plan.get("upscale_mode", "off") != "off":
            raise ValueError("Open-H3 replaces the other upscale routes. Select Open-H3 in Settings PRO Mode Assistant.")
        if enabled and not cfg["lora_name"]:
            raise ValueError("Select an Open-H3 refiner LoRA in Settings PRO Output.")
        if enabled and not cfg["model_name"]:
            raise ValueError("Select an H3 latent upscaler in Settings PRO Output.")
        chunks = plan.get("chunks") or []
        chunk = chunks[segment_index] if 0 <= segment_index < len(chunks) else {}
        prompt = cfg["prompt"] or chunk.get("prompt") or plan.get("global_prompt", "")
        sigmas = cfg["custom_sigmas"] if cfg["sigma_preset"] == "custom" else SIGMAS[cfg["sigma_preset"]]
        seed = (int(cfg["seed"]) + int(segment_index)) % (1 << 64)
        distill_strength = float(cfg["distill_strength"]) if cfg["distill_lora_name"] else 0.0
        use_face_source = bool(plan.get("face_detailer_settings", {}).get("enabled", False)) and plan.get("task_mode") != "longvid_keyframe_continuous"
        return (enabled, prompt, float(cfg["scale"]), cfg["model_name"], cfg["precision"], cfg["device"], bool(cfg["temporal_chunking"]), cfg["lora_name"], float(cfg["strength"]), cfg["distill_lora_name"], distill_strength, sigmas, seed, use_face_source)


NODE_CLASS_MAPPINGS = {"IAMCCS_MiniMaxH3OpenRefineControl": IAMCCS_MiniMaxH3OpenRefineControl}
NODE_DISPLAY_NAME_MAPPINGS = {"IAMCCS_MiniMaxH3OpenRefineControl": "MiniMax H3 · Open-H3 Refinement Settings"}
