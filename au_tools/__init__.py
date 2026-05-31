from .audio_board_arranger import IAMCCS_AudioBoardArranger
from .audio_bus_out import IAMCCS_BusOut
from .audio_board_mixer import IAMCCS_AudioBoardMixer
from .audio_control_efx import IAMCCS_ControlAudEfx
from .dialogue_script_planner import IAMCCS_DialogueScriptPlanner

NODE_CLASS_MAPPINGS = {
    "IAMCCS_AudioBoardArranger": IAMCCS_AudioBoardArranger,
    "IAMCCS_BusOut": IAMCCS_BusOut,
    "IAMCCS_AudioBoardMixer": IAMCCS_AudioBoardMixer,
    "IAMCCS_ControlAudEfx": IAMCCS_ControlAudEfx,
    "IAMCCS_DialogueScriptPlanner": IAMCCS_DialogueScriptPlanner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_AudioBoardArranger": "IAMCCS AudioBoard Arranger",
    "IAMCCS_BusOut": "IAMCCS BusOut",
    "IAMCCS_AudioBoardMixer": "IAMCCS AudioBoard Mixer",
    "IAMCCS_ControlAudEfx": "IAMCCS ControlAudEfx",
    "IAMCCS_DialogueScriptPlanner": "IAMCCS DialogueScript Planner",
}
