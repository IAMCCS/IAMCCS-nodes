# ==========================================================
# __init__.py — Registro nodi IAMCCS
# ==========================================================

from .iamccs_wan_lora_stack import (
    IAMCCS_WanLoRAStack,
    IAMCCS_ModelWithLoRA,
)
from .iamccs_wan_lora_stack_simple import (
    IAMCCS_WanLoRAStackModelIO,
)

from .iamccs_wan_svipro_motion import (
    IAMCCS_WanImageMotion,
)

# Nodi principali
NODE_CLASS_MAPPINGS = {
    "IAMCCS_WanLoRAStack": IAMCCS_WanLoRAStack,
    "IAMCCS_ModelWithLoRA": IAMCCS_ModelWithLoRA,
    "IAMCCS_WanLoRAStackModelIO": IAMCCS_WanLoRAStackModelIO,
    "IAMCCS_WanImageMotion": IAMCCS_WanImageMotion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_WanLoRAStack": "LoRA Stack (WAN-style remap)",
    "IAMCCS_ModelWithLoRA": "Apply LoRA to MODEL (Native)",
    "IAMCCS_WanLoRAStackModelIO": "LoRA Stack (Model In→Out) WAN",
    "IAMCCS_WanImageMotion": "IAMCCS WanImageMotion",
}

# Web directory for JavaScript extensions
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
