# iamccs_wan_lora_stack_simple.py
# ===============================================================
# IAMCCS_WanLoRAStackModelIO
# Multi-LoRA loader (WAN-style remap) that takes MODEL in and outputs MODEL
# ===============================================================

import logging
import comfy.utils
import comfy.sd
import folder_paths

from .iamccs_wan_lora_stack import (
    standardize_wan_lora_keys,
    SuppressOptionalKeysFilter,
)


class IAMCCS_WanLoRAStackModelIO:
    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras") + ["no"]
        return {
            "required": {
                "model": ("MODEL",),
                "lora1": (lora_list, {"default": "no"}),
                "strength1": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.01}),
                "lora2": (lora_list, {"default": "no"}),
                "strength2": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01}),
                "lora3": (lora_list, {"default": "no"}),
                "strength3": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01}),
                "lora4": (lora_list, {"default": "no"}),
                "strength4": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01}),
                "model_type": (["wan2x", "flow", "standard"], {"default": "flow"}),
            },
            "optional": {
                # Allow chaining in externally prepared LORA stacks if provided (optional)
                "lora": ("LORA",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_stack"
    CATEGORY = "IAMCCS/LoRA"

    def _build_lora_entries(self, lora1, strength1, lora2, strength2, lora3, strength3, lora4, strength4, model_type):
        loras = []
        for name, strength in [
            (lora1, strength1),
            (lora2, strength2),
            (lora3, strength3),
            (lora4, strength4),
        ]:
            if not name or name == "no" or strength == 0.0:
                continue
            path = folder_paths.get_full_path_or_raise("loras", name)
            sd = comfy.utils.load_torch_file(path, safe_load=True)
            if model_type != "standard":
                sd = standardize_wan_lora_keys(sd)
            loras.append({"name": name, "strength": strength, "state_dict": sd})
        return loras

    def apply_stack(self, model,
                    lora1, strength1,
                    lora2, strength2,
                    lora3, strength3,
                    lora4, strength4,
                    model_type="flow",
                    lora=None):
        model_out = model

        loras = self._build_lora_entries(
            lora1, strength1,
            lora2, strength2,
            lora3, strength3,
            lora4, strength4,
            model_type,
        )

        if lora is not None and isinstance(lora, list):
            loras.extend(lora)

        if not loras:
            logging.warning("[IAMCCS_WanLoRAStackModelIO] ⚠ No LoRA selected; returning input model unchanged")
            return (model_out,)

        logger = logging.getLogger()
        optional_filter = SuppressOptionalKeysFilter()
        logger.addFilter(optional_filter)

        try:
            for entry in loras:
                sd = entry["state_dict"]
                strength = entry["strength"]
                model_out, _ = comfy.sd.load_lora_for_models(model_out, None, sd, strength, 0)
                logging.info(f"[IAMCCS_WanLoRAStackModelIO] ✅ '{entry['name']}' strength={strength}")

            if optional_filter.suppressed_count > 0:
                keys_types = ", ".join(sorted(optional_filter.suppressed_keys))
                logging.info(f"[IAMCCS_WanLoRAStackModelIO] ℹ {optional_filter.suppressed_count} optional keys not present in LORA ({keys_types})")
        finally:
            logger.removeFilter(optional_filter)

        return (model_out,)


NODE_CLASS_MAPPINGS = {
    "IAMCCS_WanLoRAStackModelIO": IAMCCS_WanLoRAStackModelIO,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_WanLoRAStackModelIO": "LoRA Stack (Model In→Out) WAN",
}
