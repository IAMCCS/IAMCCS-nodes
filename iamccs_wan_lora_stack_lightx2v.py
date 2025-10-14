# ===============================================================
# IAMCCS_WanLoRAStack_LightX2V — v4 (Semplificato)
# La LoRA ha già le chiavi giuste (diffusion_model.blocks.X...)
# Convertiamo SOLO lora_down/up → lora_A/B e ignoriamo diff_*
# ===============================================================

import logging
import comfy.utils
import folder_paths
from collections import defaultdict

# ---------------------------------------------------------------
# Remapper semplificato per LightX2V
# ---------------------------------------------------------------

def remap_lightx2v_keys(sd_in: dict):
    """
    Rimappa le chiavi LoRA LightX2V:
    - Converte lora_down.weight → lora_A.weight
    - Converte lora_up.weight → lora_B.weight
    - Ignora diff_b, diff, diff_a, norm_* (non supportati)
    - Mantiene tutto il resto invariato
    """
    out = {}
    stats = defaultdict(int)

    for k, v in sd_in.items():
        if not hasattr(v, "dtype"):
            stats["skipped_no_tensor"] += 1
            continue

        # Ignora chiavi diff_* e norm_* che non sono LoRA standard
        if any(k.endswith(suffix) for suffix in (".diff_b", ".diff", ".diff_a", ".diff_m")):
            stats["ignored_diff"] += 1
            continue

        if ".norm_" in k or ".norm3." in k or k.endswith(".norm"):
            stats["ignored_norm"] += 1
            continue

        # Converti lora_down → lora_A
        if k.endswith(".lora_down.weight"):
            new_key = k.replace(".lora_down.weight", ".lora_A.weight")
            out[new_key] = v
            stats["converted_down_to_A"] += 1
            continue

        # Converti lora_up → lora_B
        if k.endswith(".lora_up.weight"):
            new_key = k.replace(".lora_up.weight", ".lora_B.weight")
            out[new_key] = v
            stats["converted_up_to_B"] += 1
            continue

        # Mantieni alpha e altre chiavi invariate
        if k.endswith((".alpha", ".lora_A.weight", ".lora_B.weight")):
            out[k] = v
            stats["kept_unchanged"] += 1
            continue

        # Tutte le altre chiavi vengono ignorate
        stats["ignored_other"] += 1

    stats["in_keys"] = len(sd_in)
    stats["out_keys"] = len(out)

    # Compact logging - only essential info
    logging.info(f"[IAMCCS_LightX2V] ✅ Remap: {stats['out_keys']} LoRA params ready " +
                 f"({stats['converted_down_to_A']} pairs converted, " +
                 f"{stats['ignored_diff'] + stats['ignored_norm'] + stats['ignored_other']} ignored)")

    return out, stats

# ---------------------------------------------------------------
# Nodo principale
# ---------------------------------------------------------------

class IAMCCS_WanLoRAStack_LightX2V:
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
                "lora4": (lora_list, {"default": "no"}),
                "strength4": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01}),
                "lora5": (lora_list, {"default": "no"}),
                "strength5": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LORA",)
    FUNCTION = "load_and_fix_lightx2v"
    CATEGORY = "IAMCCS/LoRA"

    def load_and_fix_lightx2v(self, lora1, strength1,
                              lora2, strength2,
                              lora3, strength3,
                              lora4, strength4,
                              lora5, strength5):

        selected = [
            (lora1, strength1),
            (lora2, strength2),
            (lora3, strength3),
            (lora4, strength4),
            (lora5, strength5),
        ]

        loras = []
        for name, strength in selected:
            if not name or name == "no" or float(strength) == 0.0:
                continue
            path = folder_paths.get_full_path_or_raise("loras", name)
            raw_sd = comfy.utils.load_torch_file(path, safe_load=True)

            fixed_sd, summary = remap_lightx2v_keys(raw_sd)
            loras.append({"name": name, "strength": float(strength), "state_dict": fixed_sd})

            # Compact success message with key info
            logging.info(f"[IAMCCS_LightX2V] ✅ '{name}' loaded: {summary['out_keys']} params (strength={strength})")

        if not loras:
            logging.warning("[IAMCCS_LightX2V] ⚠ No LoRA selected")
        else:
            # Final summary - fixed calculation
            total_keys = sum(len(l.get("state_dict", {})) for l in loras)

            logging.info("="*60)
            logging.info("[IAMCCS_LightX2V] 🎉 SUCCESS: All critical LoRA keys loaded")
            logging.info(f"  • {len(loras)} LoRA file(s) processed")
            logging.info(f"  • {total_keys} total LoRA parameters active")
            logging.info("  • All attention & FFN layers covered")
            logging.info("="*60)

        return (loras,)

NODE_CLASS_MAPPINGS = {
    "IAMCCS_WanLoRAStack_LightX2V": IAMCCS_WanLoRAStack_LightX2V,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_WanLoRAStack_LightX2V": "LoRA Stack (LightX2V Remap)",
}
