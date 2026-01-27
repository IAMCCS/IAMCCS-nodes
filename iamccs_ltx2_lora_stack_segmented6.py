# iamccs_ltx2_lora_stack_segmented6.py
# ===============================================================
# Segmented LoRA stacks for workflows with 3 segments × 2 stages.
# Outputs either 6 LoRA stacks or 6 MODELs (MODEL only, no CLIP).
# ===============================================================

import logging
from typing import Any, Dict, Optional

import comfy.sd
import comfy.utils
import folder_paths

from .iamccs_ltx2_lora_stack import SuppressLTX2MissingKeysFilter, standardize_ltx2_lora_keys


def _load_lora_state_dict(name: str, cache: Dict[str, Any]) -> Optional[dict]:
    if not name or name == "no":
        return None
    if name in cache:
        return cache[name]

    path = folder_paths.get_full_path_or_raise("loras", name)
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    sd = standardize_ltx2_lora_keys(sd)
    cache[name] = sd
    return sd


def _append_lora(stack: list, name: str, strength: float, cache: Dict[str, Any]) -> None:
    if not name or name == "no":
        return
    s = float(strength)
    if s == 0.0:
        return

    sd = _load_lora_state_dict(name, cache)
    if not sd:
        return

    stack.append({"name": name, "strength": s, "state_dict": sd})


def _build_segment_stage_stack(
    *,
    fixed_lora: str,
    fixed_strength: float,
    var_lora1: str,
    var1_strength: float,
    var_lora2: str,
    var2_strength: float,
    cache: Dict[str, Any],
) -> list:
    stack: list = []
    _append_lora(stack, fixed_lora, fixed_strength, cache)
    _append_lora(stack, var_lora1, var1_strength, cache)
    _append_lora(stack, var_lora2, var2_strength, cache)
    return stack


def _apply_lora_stack_to_model(model, lora_stack: list):
    if not lora_stack:
        return model

    model_out = model
    for entry in lora_stack:
        sd = entry["state_dict"]
        strength = float(entry["strength"])
        model_out, _ = comfy.sd.load_lora_for_models(model_out, None, sd, strength, 0)
    return model_out


class IAMCCS_LTX2_LoRAStackSegmented6:
    """Builds 6 LORA stacks: 3 segments × 2 stages (MODEL-only workflows)."""

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras") + ["no"]

        required: Dict[str, Any] = {
            "fixed_lora": (lora_list, {"default": "no"}),
        }

        # 3 segments (0..2), each has 2 stages and 2 variable loras
        for seg in range(3):
            required[f"seg{seg}_var_lora1"] = (lora_list, {"default": "no"})
            required[f"seg{seg}_var_lora2"] = (lora_list, {"default": "no"})

            # fixed strength per stage
            required[f"seg{seg}_fixed_strength_stage1"] = (
                "FLOAT",
                {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01},
            )
            required[f"seg{seg}_fixed_strength_stage2"] = (
                "FLOAT",
                {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01},
            )

            # var strengths per stage
            for i in (1, 2):
                required[f"seg{seg}_var{i}_strength_stage1"] = (
                    "FLOAT",
                    {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01},
                )
                required[f"seg{seg}_var{i}_strength_stage2"] = (
                    "FLOAT",
                    {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01},
                )

        return {"required": required}

    RETURN_TYPES = ("LORA", "LORA", "LORA", "LORA", "LORA", "LORA")
    RETURN_NAMES = (
        "seg0_stage1_lora",
        "seg0_stage2_lora",
        "seg1_stage1_lora",
        "seg1_stage2_lora",
        "seg2_stage1_lora",
        "seg2_stage2_lora",
    )
    FUNCTION = "build"
    CATEGORY = "IAMCCS/LoRA"

    def build(self, fixed_lora: str, **kwargs):
        cache: Dict[str, Any] = {}

        out: list[list] = []
        for seg in range(3):
            var1 = str(kwargs.get(f"seg{seg}_var_lora1") or "no")
            var2 = str(kwargs.get(f"seg{seg}_var_lora2") or "no")

            fixed_s1 = kwargs.get(f"seg{seg}_fixed_strength_stage1", 0.0)
            fixed_s2 = kwargs.get(f"seg{seg}_fixed_strength_stage2", 0.0)

            v1s1 = kwargs.get(f"seg{seg}_var1_strength_stage1", 0.0)
            v1s2 = kwargs.get(f"seg{seg}_var1_strength_stage2", 0.0)
            v2s1 = kwargs.get(f"seg{seg}_var2_strength_stage1", 0.0)
            v2s2 = kwargs.get(f"seg{seg}_var2_strength_stage2", 0.0)

            out.append(
                _build_segment_stage_stack(
                    fixed_lora=fixed_lora,
                    fixed_strength=fixed_s1,
                    var_lora1=var1,
                    var1_strength=v1s1,
                    var_lora2=var2,
                    var2_strength=v2s1,
                    cache=cache,
                )
            )
            out.append(
                _build_segment_stage_stack(
                    fixed_lora=fixed_lora,
                    fixed_strength=fixed_s2,
                    var_lora1=var1,
                    var1_strength=v1s2,
                    var_lora2=var2,
                    var2_strength=v2s2,
                    cache=cache,
                )
            )

        # Logging summary (compact)
        total = sum(len(s) for s in out)
        if total == 0:
            logging.warning("[IAMCCS_LTX2_LoRAStackSegmented6] ⚠ No LoRA selected")
        else:
            logging.info(f"[IAMCCS_LTX2_LoRAStackSegmented6] ✅ Built 6 stacks ({total} active entries)")

        return tuple(out)


class IAMCCS_LTX2_ModelWithLoRA_Segmented6:
    """Applies 6 stacks (3 segments × 2 stages) to a base MODEL and outputs 6 MODELs."""

    @classmethod
    def INPUT_TYPES(cls):
        # Mirror config of stack node, but include base model
        lora_list = folder_paths.get_filename_list("loras") + ["no"]

        required: Dict[str, Any] = {
            "model": ("MODEL",),
            "fixed_lora": (lora_list, {"default": "no"}),
        }

        for seg in range(3):
            required[f"seg{seg}_var_lora1"] = (lora_list, {"default": "no"})
            required[f"seg{seg}_var_lora2"] = (lora_list, {"default": "no"})

            required[f"seg{seg}_fixed_strength_stage1"] = (
                "FLOAT",
                {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01},
            )
            required[f"seg{seg}_fixed_strength_stage2"] = (
                "FLOAT",
                {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01},
            )

            for i in (1, 2):
                required[f"seg{seg}_var{i}_strength_stage1"] = (
                    "FLOAT",
                    {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01},
                )
                required[f"seg{seg}_var{i}_strength_stage2"] = (
                    "FLOAT",
                    {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01},
                )

        return {"required": required}

    RETURN_TYPES = ("MODEL", "MODEL", "MODEL", "MODEL", "MODEL", "MODEL")
    RETURN_NAMES = (
        "seg0_stage1_model",
        "seg0_stage2_model",
        "seg1_stage1_model",
        "seg1_stage2_model",
        "seg2_stage1_model",
        "seg2_stage2_model",
    )
    FUNCTION = "apply_segmented"
    CATEGORY = "IAMCCS/LoRA"

    def apply_segmented(self, model, fixed_lora: str, **kwargs):
        cache: Dict[str, Any] = {}

        # Build all 6 stacks
        stacks: list[list] = []
        for seg in range(3):
            var1 = str(kwargs.get(f"seg{seg}_var_lora1") or "no")
            var2 = str(kwargs.get(f"seg{seg}_var_lora2") or "no")

            fixed_s1 = kwargs.get(f"seg{seg}_fixed_strength_stage1", 0.0)
            fixed_s2 = kwargs.get(f"seg{seg}_fixed_strength_stage2", 0.0)

            v1s1 = kwargs.get(f"seg{seg}_var1_strength_stage1", 0.0)
            v1s2 = kwargs.get(f"seg{seg}_var1_strength_stage2", 0.0)
            v2s1 = kwargs.get(f"seg{seg}_var2_strength_stage1", 0.0)
            v2s2 = kwargs.get(f"seg{seg}_var2_strength_stage2", 0.0)

            stacks.append(
                _build_segment_stage_stack(
                    fixed_lora=fixed_lora,
                    fixed_strength=fixed_s1,
                    var_lora1=var1,
                    var1_strength=v1s1,
                    var_lora2=var2,
                    var2_strength=v2s1,
                    cache=cache,
                )
            )
            stacks.append(
                _build_segment_stage_stack(
                    fixed_lora=fixed_lora,
                    fixed_strength=fixed_s2,
                    var_lora1=var1,
                    var1_strength=v1s2,
                    var_lora2=var2,
                    var2_strength=v2s2,
                    cache=cache,
                )
            )

        # Apply with log suppression
        logger = logging.getLogger()
        missing_keys_filter = SuppressLTX2MissingKeysFilter()
        logger.addFilter(missing_keys_filter)
        try:
            models = []
            for idx, stack in enumerate(stacks):
                out_model = _apply_lora_stack_to_model(model, stack)
                models.append(out_model)
                if stack:
                    names = ", ".join(f"{e['name']}({e['strength']})" for e in stack)
                    logging.info(f"[IAMCCS_LTX2_ModelWithLoRA_Segmented6] segStage[{idx}] -> {names}")
            return tuple(models)
        finally:
            logger.removeFilter(missing_keys_filter)


NODE_CLASS_MAPPINGS = {
    "IAMCCS_LTX2_LoRAStackSegmented6": IAMCCS_LTX2_LoRAStackSegmented6,
    "IAMCCS_LTX2_ModelWithLoRA_Segmented6": IAMCCS_LTX2_ModelWithLoRA_Segmented6,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_LTX2_LoRAStackSegmented6": "LoRA Stack (LTX-2, segmented: 3 seg × 2 stages)",
    "IAMCCS_LTX2_ModelWithLoRA_Segmented6": "Apply LoRA to MODEL (LTX-2, segmented: 3 seg × 2 stages)",
}
