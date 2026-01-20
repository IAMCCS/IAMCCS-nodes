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

from .iamccs_ltx2_lora_stack import (
    IAMCCS_LTX2_LoRAStack,
    IAMCCS_LTX2_LoRAStackStaged,
    IAMCCS_ModelWithLoRA_LTX2,
    IAMCCS_ModelWithLoRA_LTX2_Staged,
    IAMCCS_LTX2_LoRAStackModelIO,
)

from .iamccs_ltx2_tools import (
    IAMCCS_LTX2_FrameRateSync,
    IAMCCS_LTX2_Validator,
    IAMCCS_LTX2_TimeFrameCount,
    IAMCCS_LTX2_ControlPreprocess,
)

from .iamccs_wan_svipro_motion import (
    IAMCCS_WanImageMotion,
)

# Nodi principali
NODE_CLASS_MAPPINGS = {
    "IAMCCS_WanLoRAStack": IAMCCS_WanLoRAStack,
    "IAMCCS_ModelWithLoRA": IAMCCS_ModelWithLoRA,
    "IAMCCS_WanLoRAStackModelIO": IAMCCS_WanLoRAStackModelIO,
    # Backward-compatible key (kept as-is for existing workflows)
    "iamccs_ltx2_lora_stack": IAMCCS_LTX2_LoRAStack,
    # Preferred explicit names
    "IAMCCS_LTX2_LoRAStack": IAMCCS_LTX2_LoRAStack,
    "IAMCCS_LTX2_LoRAStackStaged": IAMCCS_LTX2_LoRAStackStaged,
    "IAMCCS_ModelWithLoRA_LTX2": IAMCCS_ModelWithLoRA_LTX2,
    "IAMCCS_ModelWithLoRA_LTX2_Staged": IAMCCS_ModelWithLoRA_LTX2_Staged,
    "IAMCCS_LTX2_LoRAStackModelIO": IAMCCS_LTX2_LoRAStackModelIO,

    "IAMCCS_LTX2_FrameRateSync": IAMCCS_LTX2_FrameRateSync,
    "IAMCCS_LTX2_Validator": IAMCCS_LTX2_Validator,
    "IAMCCS_LTX2_TimeFrameCount": IAMCCS_LTX2_TimeFrameCount,
    "IAMCCS_LTX2_ControlPreprocess": IAMCCS_LTX2_ControlPreprocess,
    "IAMCCS_WanImageMotion": IAMCCS_WanImageMotion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_WanLoRAStack": "LoRA Stack (WAN-style remap)",
    "IAMCCS_ModelWithLoRA": "Apply LoRA to MODEL (Native)",
    "IAMCCS_WanLoRAStackModelIO": "LoRA Stack (Model In→Out) WAN",
    "iamccs_ltx2_lora_stack": "iamccs_ltx2_lora_stack (3 slots)",
    "IAMCCS_LTX2_LoRAStack": "LoRA Stack (LTX-2, 3 slots)",
    "IAMCCS_LTX2_LoRAStackStaged": "LoRA Stack (LTX-2, staged: stage1+stage2) (BETA)",
    "IAMCCS_ModelWithLoRA_LTX2": "Apply LoRA to MODEL (LTX-2, quiet logs)",
    "IAMCCS_ModelWithLoRA_LTX2_Staged": "Apply LoRA to MODEL (LTX-2, staged) (BETA)",
    "IAMCCS_LTX2_LoRAStackModelIO": "LoRA Stack (Model In→Out) LTX-2",

    "IAMCCS_LTX2_FrameRateSync": "LTX-2 FrameRate Sync (int+float)",
    "IAMCCS_LTX2_Validator": "LTX-2 Validator (16px, 8n +1)",
    "IAMCCS_LTX2_TimeFrameCount": "LTX-2 TimeFrameCount",
    "IAMCCS_LTX2_ControlPreprocess": "LTX-2 Control Preprocess (aux)",
    "IAMCCS_WanImageMotion": "IAMCCS WanImageMotion",
}

# Web directory for JavaScript extensions
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
