# iamccs_ltx2_lora_stack.py
# ===============================================================
# IAMCCS_LTX2_LoRAStack / IAMCCS_ModelWithLoRA_LTX2
# 3-slot LoRA loader intended for LTX-2 style LoRAs.
# ===============================================================

import logging

import comfy.sd
import comfy.utils
import folder_paths


def standardize_ltx2_lora_keys(sd: dict) -> dict:
    """Minimal standardization for LTX-2 LoRA state_dict.

    Notes:
    - Does NOT apply WAN-style key remapping.
    - Converts old LoRA formats using lora_down/lora_up to lora_A/lora_B.

    This is intentionally conservative: LTX-2/Flux/WAN families differ in key
    naming, and aggressive remapping can cause widespread 'key not loaded' spam.
    """

    if not sd:
        return sd

    sample_keys = list(sd.keys())[:80]
    has_down_up = any(".lora_down." in k or ".lora_up." in k for k in sample_keys)
    has_a_b = any(".lora_A." in k or ".lora_B." in k for k in sample_keys)
    needs_convert = has_down_up and not has_a_b

    if not needs_convert:
        return sd

    new_sd = {}
    for k, v in sd.items():
        nk = k
        if nk.endswith(".lora_down.weight"):
            nk = nk.replace(".lora_down.weight", ".lora_A.weight")
        elif nk.endswith(".lora_up.weight"):
            nk = nk.replace(".lora_up.weight", ".lora_B.weight")
        new_sd[nk] = v

    logging.info("[IAMCCS_LTX2_LoRAStack] Detected legacy LoRA format - converted lora_down/up → lora_A/B")
    return new_sd


class SuppressLTX2MissingKeysFilter(logging.Filter):
    """Suppresses noisy 'lora key not loaded' lines for known-mismatch key prefixes.

    This does NOT change which weights are actually applied; it only reduces log spam.
    """

    def __init__(self):
        super().__init__()
        self.suppressed_count = 0
        self.suppressed_prefixes = set()

        # Common spammy prefixes seen when LTX-2 style LoRAs are applied to a model
        # with a different internal layout (e.g. Flux variants).
        self._prefixes = (
            "diffusion_model.transformer_blocks.",
            "text_embedding_projection.",
        )

    def filter(self, record):
        msg = record.getMessage()
        msg_lower = msg.lower()

        if "lora key not loaded:" not in msg_lower:
            return True

        # Message format usually: "lora key not loaded: <key>"
        key = msg.split(":", 1)[-1].strip() if ":" in msg else msg
        for p in self._prefixes:
            if key.startswith(p):
                self.suppressed_count += 1
                self.suppressed_prefixes.add(p)
                return False

        return True


# ===============================================================
# Node 1 — LTX-2 LoRA Stack (up to 3 LoRAs with strength)
# ===============================================================
class IAMCCS_LTX2_LoRAStack:
    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras") + ["no"]
        return {
            "required": {
                "lora1": (lora_list, {"default": "no"}),
                "strength1": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.01}),
                "lora2": (lora_list, {"default": "no"}),
                "strength2": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01}),
                "lora3": (lora_list, {"default": "no"}),
                "strength3": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01}),
            },
            "optional": {
                "lora": ("LORA",),
            },
        }

    RETURN_TYPES = ("LORA",)
    FUNCTION = "load_ltx2"
    CATEGORY = "IAMCCS/LoRA"

    def load_ltx2(self, lora1, strength1, lora2, strength2, lora3, strength3, lora=None):
        loras = []

        for name, strength in (
            (lora1, strength1),
            (lora2, strength2),
            (lora3, strength3),
        ):
            if not name or name == "no" or strength == 0.0:
                continue

            path = folder_paths.get_full_path_or_raise("loras", name)
            sd = comfy.utils.load_torch_file(path, safe_load=True)
            sd = standardize_ltx2_lora_keys(sd)
            loras.append({"name": name, "strength": strength, "state_dict": sd})

        if lora is not None and isinstance(lora, list):
            loras.extend(lora)

        if not loras:
            logging.warning("[IAMCCS_LTX2_LoRAStack] ⚠ No LoRA selected")
        else:
            total_keys = sum(len(l.get("state_dict", {})) for l in loras)
            logging.info("=" * 60)
            logging.info(f"[IAMCCS_LTX2_LoRAStack] ✅ SUCCESS: {len(loras)} LoRA(s) loaded")
            logging.info(f"  • {total_keys} total parameters active")
            logging.info("=" * 60)

        return (loras,)


# ===============================================================
# Node 1b — LTX-2 LoRA Stack (staged: outputs 2 stacks)
# ===============================================================
class IAMCCS_LTX2_LoRAStackStaged:
    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras") + ["no"]
        return {
            "required": {
                "lora1": (lora_list, {"default": "no"}),
                "strength1_stage1": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.01}),
                "strength1_stage2": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01}),

                "lora2": (lora_list, {"default": "no"}),
                "strength2_stage1": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01}),
                "strength2_stage2": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01}),

                "lora3": (lora_list, {"default": "no"}),
                "strength3_stage1": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01}),
                "strength3_stage2": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01}),
            },
            "optional": {
                "lora": ("LORA",),
            },
        }

    RETURN_TYPES = ("LORA", "LORA")
    RETURN_NAMES = ("lora_stage1", "lora_stage2")
    FUNCTION = "load_ltx2_staged"
    CATEGORY = "IAMCCS/LoRA"

    def load_ltx2_staged(
        self,
        lora1,
        strength1_stage1,
        strength1_stage2,
        lora2,
        strength2_stage1,
        strength2_stage2,
        lora3,
        strength3_stage1,
        strength3_stage2,
        lora=None,
    ):
        loras_stage1 = []
        loras_stage2 = []

        for name, s1, s2 in (
            (lora1, strength1_stage1, strength1_stage2),
            (lora2, strength2_stage1, strength2_stage2),
            (lora3, strength3_stage1, strength3_stage2),
        ):
            if not name or name == "no":
                continue

            if float(s1) == 0.0 and float(s2) == 0.0:
                continue

            path = folder_paths.get_full_path_or_raise("loras", name)
            sd = comfy.utils.load_torch_file(path, safe_load=True)
            sd = standardize_ltx2_lora_keys(sd)

            if float(s1) != 0.0:
                loras_stage1.append({"name": name, "strength": float(s1), "state_dict": sd})
            if float(s2) != 0.0:
                loras_stage2.append({"name": name, "strength": float(s2), "state_dict": sd})

        if lora is not None and isinstance(lora, list):
            # External stack is included identically in both stages
            loras_stage1.extend(lora)
            loras_stage2.extend(lora)

        if not loras_stage1:
            logging.warning("[IAMCCS_LTX2_LoRAStackStaged] ⚠ Stage 1: no LoRA selected")
        if not loras_stage2:
            logging.warning("[IAMCCS_LTX2_LoRAStackStaged] ⚠ Stage 2: no LoRA selected")

        return (loras_stage1, loras_stage2)


# ===============================================================
# Node 2 — Apply LoRA to MODEL (LTX-2 oriented, with log suppression)
# ===============================================================
class IAMCCS_ModelWithLoRA_LTX2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora": ("LORA",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "IAMCCS/LoRA"

    def apply(self, model, lora):
        if not lora:
            return (model,)

        model_out = model

        logger = logging.getLogger()
        missing_keys_filter = SuppressLTX2MissingKeysFilter()
        logger.addFilter(missing_keys_filter)

        try:
            for entry in lora:
                sd = entry["state_dict"]
                strength = entry["strength"]
                model_out, _ = comfy.sd.load_lora_for_models(model_out, None, sd, strength, 0)
                logging.info(f"[IAMCCS_ModelWithLoRA_LTX2] ✅ '{entry['name']}' strength={strength}")

            if missing_keys_filter.suppressed_count > 0:
                prefixes = ", ".join(sorted(missing_keys_filter.suppressed_prefixes))
                logging.info(
                    "[IAMCCS_ModelWithLoRA_LTX2] ℹ Suppressed "
                    f"{missing_keys_filter.suppressed_count} 'lora key not loaded' lines (prefixes: {prefixes})."
                )
                logging.info(
                    "[IAMCCS_ModelWithLoRA_LTX2] ℹ If the LoRA effect is weak/absent, verify the base model matches the LoRA family."
                )
        finally:
            logger.removeFilter(missing_keys_filter)

        return (model_out,)


# ===============================================================
# Node 2b — Apply staged LoRA stacks to MODEL (LTX-2)
# ===============================================================
class IAMCCS_ModelWithLoRA_LTX2_Staged:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_stage1": ("LORA",),
                "lora_stage2": ("LORA",),
            },
            "optional": {
                # If omitted, stage2 uses the same model as stage1.
                "model_stage2": ("MODEL",),
            },
        }

    RETURN_TYPES = ("MODEL", "MODEL")
    RETURN_NAMES = ("model_stage1", "model_stage2")
    FUNCTION = "apply_staged"
    CATEGORY = "IAMCCS/LoRA"

    def apply_staged(self, model, lora_stage1, lora_stage2, model_stage2=None):
        model_stage1_out = model
        model_stage2_out = model_stage2 if model_stage2 is not None else model

        lora_stage1_list = lora_stage1 if isinstance(lora_stage1, list) else []
        lora_stage2_list = lora_stage2 if isinstance(lora_stage2, list) else []

        if not lora_stage1_list and not lora_stage2_list:
            return (model_stage1_out, model_stage2_out)

        logger = logging.getLogger()
        missing_keys_filter = SuppressLTX2MissingKeysFilter()
        logger.addFilter(missing_keys_filter)

        try:
            if lora_stage1_list:
                for entry in lora_stage1_list:
                    sd = entry["state_dict"]
                    strength = entry["strength"]
                    model_stage1_out, _ = comfy.sd.load_lora_for_models(model_stage1_out, None, sd, strength, 0)
                    logging.info(f"[IAMCCS_ModelWithLoRA_LTX2_Staged] ✅ Stage1: '{entry['name']}' strength={strength}")

            if lora_stage2_list:
                for entry in lora_stage2_list:
                    sd = entry["state_dict"]
                    strength = entry["strength"]
                    model_stage2_out, _ = comfy.sd.load_lora_for_models(model_stage2_out, None, sd, strength, 0)
                    logging.info(f"[IAMCCS_ModelWithLoRA_LTX2_Staged] ✅ Stage2: '{entry['name']}' strength={strength}")

            if missing_keys_filter.suppressed_count > 0:
                prefixes = ", ".join(sorted(missing_keys_filter.suppressed_prefixes))
                logging.info(
                    "[IAMCCS_ModelWithLoRA_LTX2_Staged] ℹ Suppressed "
                    f"{missing_keys_filter.suppressed_count} 'lora key not loaded' lines (prefixes: {prefixes})."
                )
                logging.info(
                    "[IAMCCS_ModelWithLoRA_LTX2_Staged] ℹ If the LoRA effect is weak/absent, verify the base model matches the LoRA family."
                )
        finally:
            logger.removeFilter(missing_keys_filter)

        return (model_stage1_out, model_stage2_out)


# ===============================================================
# Node 3 — LTX-2 LoRA Stack (MODEL In→Out)
# ===============================================================
class IAMCCS_LTX2_LoRAStackModelIO:
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
            },
            "optional": {
                "lora": ("LORA",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_stack"
    CATEGORY = "IAMCCS/LoRA"

    def apply_stack(self, model, lora1, strength1, lora2, strength2, lora3, strength3, lora=None):
        entries = []

        for name, strength in (
            (lora1, strength1),
            (lora2, strength2),
            (lora3, strength3),
        ):
            if not name or name == "no" or float(strength) == 0.0:
                continue
            path = folder_paths.get_full_path_or_raise("loras", name)
            sd = comfy.utils.load_torch_file(path, safe_load=True)
            sd = standardize_ltx2_lora_keys(sd)
            entries.append({"name": name, "strength": float(strength), "state_dict": sd})

        if lora is not None and isinstance(lora, list):
            entries.extend(lora)

        if not entries:
            logging.warning("[IAMCCS_LTX2_LoRAStackModelIO] ⚠ No LoRA selected; returning input model unchanged")
            return (model,)

        model_out = model

        logger = logging.getLogger()
        missing_keys_filter = SuppressLTX2MissingKeysFilter()
        logger.addFilter(missing_keys_filter)

        try:
            for entry in entries:
                sd = entry["state_dict"]
                strength = entry["strength"]
                model_out, _ = comfy.sd.load_lora_for_models(model_out, None, sd, strength, 0)
                logging.info(f"[IAMCCS_LTX2_LoRAStackModelIO] ✅ '{entry['name']}' strength={strength}")

            if missing_keys_filter.suppressed_count > 0:
                prefixes = ", ".join(sorted(missing_keys_filter.suppressed_prefixes))
                logging.info(
                    "[IAMCCS_LTX2_LoRAStackModelIO] ℹ Suppressed "
                    f"{missing_keys_filter.suppressed_count} 'lora key not loaded' lines (prefixes: {prefixes})."
                )
        finally:
            logger.removeFilter(missing_keys_filter)

        return (model_out,)


NODE_CLASS_MAPPINGS = {
    "IAMCCS_LTX2_LoRAStack": IAMCCS_LTX2_LoRAStack,
    "IAMCCS_LTX2_LoRAStackStaged": IAMCCS_LTX2_LoRAStackStaged,
    "IAMCCS_ModelWithLoRA_LTX2": IAMCCS_ModelWithLoRA_LTX2,
    "IAMCCS_ModelWithLoRA_LTX2_Staged": IAMCCS_ModelWithLoRA_LTX2_Staged,
    "IAMCCS_LTX2_LoRAStackModelIO": IAMCCS_LTX2_LoRAStackModelIO,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_LTX2_LoRAStack": "LoRA Stack (LTX-2, 3 slots)",
    "IAMCCS_LTX2_LoRAStackStaged": "LoRA Stack (LTX-2, staged: stage1+stage2) (BETA)",
    "IAMCCS_ModelWithLoRA_LTX2": "Apply LoRA to MODEL (LTX-2, quiet logs)",
    "IAMCCS_ModelWithLoRA_LTX2_Staged": "Apply LoRA to MODEL (LTX-2, staged) (BETA)",
    "IAMCCS_LTX2_LoRAStackModelIO": "LoRA Stack (Model In→Out) LTX-2",
}
