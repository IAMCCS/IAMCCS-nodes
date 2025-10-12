# iamccs_wan_lora_stack.py
# ===============================================================
# IAMCCS_WanLoRAStack / IAMCCS_ModelWithLoRA
# Versione multi-LoRA (4 slots con strength dedicato)
# ===============================================================

import logging
import comfy.utils
import comfy.sd
import folder_paths


# --- Normalizzatore chiavi WAN (stesso del precedente) ---
def standardize_wan_lora_keys(sd: dict) -> dict:
    new_sd = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("transformer."):
            nk = nk.replace("transformer.", "diffusion_model.")
        if nk.startswith("pipe.dit."):
            nk = nk.replace("pipe.dit.", "diffusion_model.")
        if nk.startswith("blocks."):
            nk = nk.replace("blocks.", "diffusion_model.blocks.")
        if ".attn1." in nk or ".attn2." in nk:
            tgt = ".cross_attn." if ".attn2." in nk else ".self_attn."
            nk = nk.replace(".attn1.", tgt).replace(".attn2.", tgt)
            nk = nk.replace(".to_k.", ".k.").replace(".to_q.", ".q.")
            nk = nk.replace(".to_v.", ".v.").replace(".to_out.0.", ".o.")
        if nk.startswith("lora_unet__"):
            core, *rest = nk.split(".")
            core = core.replace("lora_unet__", "diffusion_model.")
            core = core.replace("_self_attn", ".self_attn").replace("_cross_attn", ".cross_attn").replace("blocks_", "blocks.")
            nk = ".".join([core] + rest)
        nk = nk.replace("img_attn.proj", "img_attn_proj")
        nk = nk.replace("img_attn.qkv", "img_attn_qkv")
        nk = nk.replace("txt_attn.proj", "txt_attn_proj")
        nk = nk.replace("txt_attn.qkv", "txt_attn_qkv")
        new_sd[nk] = v
    return new_sd


# ===============================================================
# Nodo 1 — LoRA Stack (fino a 4 LoRA con strength)
# ===============================================================
class IAMCCS_WanLoRAStack:
    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "lora1": (lora_list,),
                "strength1": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.01}),
                "lora2": (lora_list,),
                "strength2": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01}),
                "lora3": (lora_list,),
                "strength3": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01}),
                "lora4": (lora_list,),
                "strength4": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01}),
                "model_type": (["wan2x", "flow", "standard"], {"default": "flow"}),
            }
        }

    RETURN_TYPES = ("LORA",)
    FUNCTION = "load_and_standardize"
    CATEGORY = "IAMCCS/LoRA"

    def load_and_standardize(self, lora1, strength1,
                             lora2, strength2,
                             lora3, strength3,
                             lora4, strength4,
                             model_type="flow"):

        loras = []
        for name, strength in [
            (lora1, strength1),
            (lora2, strength2),
            (lora3, strength3),
            (lora4, strength4),
        ]:
            if name and strength != 0.0:
                path = folder_paths.get_full_path_or_raise("loras", name)
                sd = comfy.utils.load_torch_file(path, safe_load=True)
                if model_type != "standard":
                    sd = standardize_wan_lora_keys(sd)
                loras.append({"name": name, "strength": strength, "state_dict": sd})

        logging.info(f"[IAMCCS_WanLoRAStack] ✅ Caricate {len(loras)} LoRA")
        return (loras,)


# ===============================================================
# Nodo 2 — Apply LoRA to MODEL (ponte, senza strength)
# ===============================================================
class IAMCCS_ModelWithLoRA:
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
        for entry in lora:
            sd = entry["state_dict"]
            strength = entry["strength"]
            model_out, _ = comfy.sd.load_lora_for_models(model_out, None, sd, strength, 0)
            logging.info(f"[IAMCCS_ModelWithLoRA] ✅ '{entry['name']}' strength={strength}")
        return (model_out,)


# ===============================================================
# Registrazione
# ===============================================================
NODE_CLASS_MAPPINGS = {
    "IAMCCS_WanLoRAStack": IAMCCS_WanLoRAStack,
    "IAMCCS_ModelWithLoRA": IAMCCS_ModelWithLoRA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_WanLoRAStack": "LoRA Stack (WAN-style remap)",
    "IAMCCS_ModelWithLoRA": "Apply LoRA to MODEL (Native)",
}
