from __future__ import annotations

from .audio_control_efx import IAMCCS_ControlAudEfx


class IAMCCS_ControlAudEfxPanel(IAMCCS_ControlAudEfx):
    """Replacement UI-facing node for the legacy ControlAudEfx instance."""


NODE_CLASS_MAPPINGS = {
    "IAMCCS_ControlAudEfxPanel": IAMCCS_ControlAudEfxPanel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_ControlAudEfxPanel": "IAMCCS ControlAudEfx Panel",
}