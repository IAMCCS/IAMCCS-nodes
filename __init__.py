# ==========================================================
# __init__.py — Registro nodi IAMCCS LoRA
# Mantiene i nodi esistenti + aggiunge il nuovo LightX2V stack
# ==========================================================

from .iamccs_wan_lora_stack import (
    IAMCCS_WanLoRAStack,
    IAMCCS_ModelWithLoRA,
)

from .iamccs_wan_lora_stack_lightx2v import (
    IAMCCS_WanLoRAStack_LightX2V,
)

NODE_CLASS_MAPPINGS = {
    "IAMCCS_WanLoRAStack": IAMCCS_WanLoRAStack,
    "IAMCCS_ModelWithLoRA": IAMCCS_ModelWithLoRA,
    "IAMCCS_WanLoRAStack_LightX2V": IAMCCS_WanLoRAStack_LightX2V,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_WanLoRAStack": "LoRA Stack (WAN-style remap)",
    "IAMCCS_ModelWithLoRA": "Apply LoRA to MODEL (Native)",
    "IAMCCS_WanLoRAStack_LightX2V": "LoRA Stack (LightX2V Remap)",
}
