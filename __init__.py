# ==========================================================
# __init__.py — Registro nodi IAMCCS LoRA
# Versione pulita: mantiene solo i nodi principali
# ==========================================================

from .iamccs_wan_lora_stack import (
    IAMCCS_WanLoRAStack,
    IAMCCS_ModelWithLoRA,
)

NODE_CLASS_MAPPINGS = {
    "IAMCCS_WanLoRAStack": IAMCCS_WanLoRAStack,
    "IAMCCS_ModelWithLoRA": IAMCCS_ModelWithLoRA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_WanLoRAStack": "LoRA Stack (WAN-style remap)",
    "IAMCCS_ModelWithLoRA": "Apply LoRA to MODEL (Native)",
}
