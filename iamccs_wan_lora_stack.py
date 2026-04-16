# iamccs_wan_lora_stack.py
# ===============================================================
# IAMCCS_WanLoRAStack / IAMCCS_ModelWithLoRA
# Versione multi-LoRA (4 slots con strength dedicato)
# ===============================================================

import logging
from collections import OrderedDict

import comfy.utils
import comfy.sd
import folder_paths


_STATE_DICT_CACHE: OrderedDict[tuple[str, str], dict] = OrderedDict()
_PATCHED_MODEL_CACHE: OrderedDict[tuple[int, tuple[tuple[str, float], ...]], object] = OrderedDict()
_MAX_STATE_DICT_CACHE = 16
_MAX_PATCHED_MODEL_CACHE = 8


def _lora_stack_debug_summary(lora) -> str:
    if not lora:
        return "empty"

    parts = []
    for entry in lora:
        name = str(entry.get("name") or "unnamed")
        strength = float(entry.get("strength", 0.0) or 0.0)
        origin = str(entry.get("_iamccs_lora_origin") or "manual")
        generation_index = entry.get("_iamccs_generation_index")
        slot = entry.get("_iamccs_schedule_slot")
        rule = str(entry.get("_iamccs_schedule_rule") or "")

        extras = [origin]
        if generation_index is not None:
            extras.append(f"gen={generation_index}")
        if slot is not None:
            extras.append(f"slot={int(slot):02d}")
        if rule:
            extras.append(rule)

        parts.append(f"{name}({strength}) [{' | '.join(extras)}]")

    return "; ".join(parts)


def _lora_stack_debug_context(lora) -> str:
    if not lora:
        return ""

    prompt_ids = sorted({str(entry.get("_iamccs_prompt_id") or "") for entry in lora if entry.get("_iamccs_prompt_id")})
    schedule_nodes = sorted({str(entry.get("_iamccs_schedule_node_id") or "") for entry in lora if entry.get("_iamccs_schedule_node_id")})
    schedule_names = sorted({str(entry.get("_iamccs_schedule_log_prefix") or "") for entry in lora if entry.get("_iamccs_schedule_log_prefix")})
    generation_indexes = sorted({int(entry.get("_iamccs_generation_index")) for entry in lora if entry.get("_iamccs_generation_index") is not None})

    parts = []
    if prompt_ids:
        parts.append(f"prompt={','.join(prompt_ids)}")
    if schedule_nodes:
        parts.append(f"schedule_node={','.join(schedule_nodes)}")
    if schedule_names:
        parts.append(f"schedule={','.join(schedule_names)}")
    if generation_indexes:
        parts.append(f"generation={','.join(str(value) for value in generation_indexes)}")
    return " | ".join(parts)


def _cache_put(cache: OrderedDict, key, value, max_size: int):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_size:
        cache.popitem(last=False)


def _state_dict_for_lora(name: str, model_type: str) -> dict:
    cache_key = (str(name), str(model_type or "flow"))
    cached = _STATE_DICT_CACHE.get(cache_key)
    if cached is not None:
        _STATE_DICT_CACHE.move_to_end(cache_key)
        return cached

    path = folder_paths.get_full_path_or_raise("loras", name)
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    if model_type != "standard":
        sd = standardize_wan_lora_keys(sd)
    _cache_put(_STATE_DICT_CACHE, cache_key, sd, _MAX_STATE_DICT_CACHE)
    return sd


# --- Log Filter per sopprimere spam img_* e diff_m keys ---
class SuppressOptionalKeysFilter(logging.Filter):
    """Filtra i log 'lora key not loaded' per chiavi opzionali (img_*, diff_m, ecc.)"""
    def __init__(self):
        super().__init__()
        self.suppressed_count = 0
        self.suppressed_keys = set()

    def filter(self, record):
        msg_lower = record.getMessage().lower()
        if "not loaded" in msg_lower:
            msg = record.getMessage()

            # Sopprimi TUTTE le chiavi con "img" - opzionali per I2V/T2V
            if "img" in msg_lower and "diffusion_model" in msg:
                self.suppressed_count += 1
                self.suppressed_keys.add("img_* layers (k_img, v_img, norm_*_img, etc.)")
                return False  # Sopprimi questo log

            # Sopprimi chiavi diff_m (opzionali, non tutte le LORA le hanno)
            if ".diff_m" in msg or msg.endswith("diff_m"):
                self.suppressed_count += 1
                self.suppressed_keys.add("diff_m layers")
                return False  # Sopprimi questo log

        return True  # Lascia passare tutti gli altri log


# --- Normalizzatore chiavi WAN ---
def standardize_wan_lora_keys(sd: dict) -> dict:
    """
    Standardizes WAN LORA keys and converts Wan 2.1 format to Wan 2.2
    - Remaps various naming conventions to diffusion_model.* format
    - Converts lora_down/up → lora_A/B for Wan 2.1 compatibility
    - Preserves diff_b, diff, norm_* keys (supported by ComfyUI)
    """
    new_sd = {}

    # First, check if this is Wan 2.1 format (has lora_down/up)
    sample_keys = list(sd.keys())[:50]
    has_down_up = any('.lora_down.' in k or '.lora_up.' in k for k in sample_keys)
    has_a_b = any('.lora_A.' in k or '.lora_B.' in k for k in sample_keys)
    is_wan21 = has_down_up and not has_a_b

    for k, v in sd.items():
        nk = k

        # Standard key remapping
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

        # Wan 2.1 → Wan 2.2 conversion
        if is_wan21:
            if nk.endswith(".lora_down.weight"):
                nk = nk.replace(".lora_down.weight", ".lora_A.weight")
            elif nk.endswith(".lora_up.weight"):
                nk = nk.replace(".lora_up.weight", ".lora_B.weight")

        new_sd[nk] = v

    if is_wan21:
        logging.info(f"[IAMCCS_WanLoRAStack] Detected Wan 2.1 format - converted lora_down/up → lora_A/B")

    return new_sd


# ===============================================================
# Nodo 1 — LoRA Stack (fino a 4 LoRA con strength)
# ===============================================================
class IAMCCS_WanLoRAStack:
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
                "model_type": (["wan2x", "flow", "standard"], {"default": "flow"}),
            },
            "optional": {
                "lora": ("LORA",),
            }
        }

    RETURN_TYPES = ("LORA",)
    FUNCTION = "load_and_standardize"
    CATEGORY = "IAMCCS/LoRA"

    def load_and_standardize(self, lora1, strength1,
                             lora2, strength2,
                             lora3, strength3,
                             lora4, strength4,
                             model_type="flow",
                             lora=None):

        loras = []
        for name, strength in [
            (lora1, strength1),
            (lora2, strength2),
            (lora3, strength3),
            (lora4, strength4),
        ]:
            # Skip if "no" selected or strength is 0
            if not name or name == "no" or strength == 0.0:
                continue
            sd = _state_dict_for_lora(name, model_type)
            loras.append({"name": name, "strength": strength, "state_dict": sd})

        # Concatena la stack LORA opzionale se fornita
        if lora is not None and isinstance(lora, list):
            loras.extend(lora)

        if not loras:
            logging.warning("[IAMCCS_WanLoRAStack] ⚠ No LoRA selected")
        else:
            total_keys = sum(len(l.get("state_dict", {})) for l in loras)
            logging.info("="*60)
            logging.info(f"[IAMCCS_WanLoRAStack] ✅ SUCCESS: {len(loras)} LoRA(s) loaded")
            logging.info(f"  • {total_keys} total parameters active")
            logging.info("="*60)

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

        stack_context = _lora_stack_debug_context(lora)
        stack_summary = _lora_stack_debug_summary(lora)
        signature = tuple((str(entry.get("name") or ""), float(entry.get("strength", 0.0) or 0.0)) for entry in lora)
        cache_key = (id(model), signature)
        cached_model = _PATCHED_MODEL_CACHE.get(cache_key)
        if cached_model is not None:
            _PATCHED_MODEL_CACHE.move_to_end(cache_key)
            if stack_context:
                logging.info(f"[IAMCCS_ModelWithLoRA] ♻ cache hit: {len(signature)} LoRA(s) reused | {stack_context}")
            else:
                logging.info(f"[IAMCCS_ModelWithLoRA] ♻ cache hit: {len(signature)} LoRA(s) reused")
            logging.info(f"[IAMCCS_ModelWithLoRA] active_stack={stack_summary}")
            return (cached_model,)

        model_out = model

        if stack_context:
            logging.info(f"[IAMCCS_ModelWithLoRA] apply request | {stack_context}")
        logging.info(f"[IAMCCS_ModelWithLoRA] active_stack={stack_summary}")

        # Installa filtro per sopprimere spam di chiavi opzionali (img_*, diff_m, ecc.)
        logger = logging.getLogger()
        optional_filter = SuppressOptionalKeysFilter()
        logger.addFilter(optional_filter)

        try:
            for entry in lora:
                sd = entry["state_dict"]
                strength = entry["strength"]
                model_out, _ = comfy.sd.load_lora_for_models(model_out, None, sd, strength, 0)
                logging.info(f"[IAMCCS_ModelWithLoRA] ✅ '{entry['name']}' strength={strength}")

            # Mostra riepilogo chiavi soppresse
            if optional_filter.suppressed_count > 0:
                keys_types = ", ".join(sorted(optional_filter.suppressed_keys))
                logging.info(f"[IAMCCS_ModelWithLoRA] ℹ {optional_filter.suppressed_count} optional keys not present in LORA ({keys_types})")

            _cache_put(_PATCHED_MODEL_CACHE, cache_key, model_out, _MAX_PATCHED_MODEL_CACHE)

        finally:
            # Rimuovi filtro
            logger.removeFilter(optional_filter)

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
