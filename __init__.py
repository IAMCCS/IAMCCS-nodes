# __init__.py — Registro nodi IAMCCS (solo nuovi nodi LoRA)
from .iamccs_wan_lora_stack import (
    IAMCCS_WanLoRAStack,
    IAMCCS_ModelWithLoRA
)

NODE_CLASS_MAPPINGS = {
    "IAMCCS_WanLoRAStack": IAMCCS_WanLoRAStack,
    "IAMCCS_ModelWithLoRA": IAMCCS_ModelWithLoRA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_WanLoRAStack": "LoRA Stack (WAN-style remap)",
    "IAMCCS_ModelWithLoRA": "Apply LoRA to MODEL (Native)",
}
