# ==========================================================
# __init__.py — Registro nodi IAMCCS
# Versione estesa: include LoRA + Qwen Bridge Conditioning
# ==========================================================

# Apply safety monkeypatches on import (no-op if target not present)
from . import iamccs_qwen_monkeypatch  # noqa: F401

from .iamccs_wan_lora_stack import (
    IAMCCS_WanLoRAStack,
    IAMCCS_ModelWithLoRA,
)
from .iamccs_wan_lora_stack_simple import (
    IAMCCS_WanLoRAStackModelIO,
)


# Qwen Image LoRA loader (fixed copy)
from .iamccs_qwen_lora_loader import (
    IAMCCS_QwenImageLoraLoader,
)

# Nodi principali
NODE_CLASS_MAPPINGS = {
    "IAMCCS_WanLoRAStack": IAMCCS_WanLoRAStack,
    "IAMCCS_ModelWithLoRA": IAMCCS_ModelWithLoRA,
    "IAMCCS_WanLoRAStackModelIO": IAMCCS_WanLoRAStackModelIO,
    "IAMCCS_qwenloraloader": IAMCCS_QwenImageLoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_WanLoRAStack": "LoRA Stack (WAN-style remap)",
    "IAMCCS_ModelWithLoRA": "Apply LoRA to MODEL (Native)",
    "IAMCCS_WanLoRAStackModelIO": "LoRA Stack (Model In→Out) WAN",
    "IAMCCS_qwenloraloader": "IAMCCS QwenImgLoraLoaderFix",
}

# Web directory for JavaScript extensions
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
