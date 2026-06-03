from .audio_board_arranger import IAMCCS_AudioBoardArranger
from .audio_bus_out import IAMCCS_BusOut
from .audio_board_mixer import IAMCCS_AudioBoardMixer
from .audio_control_efx import IAMCCS_ControlAudEfx
from .dialogue_tag_editor import IAMCCS_DialogueTagEditor, IAMCCS_DialogueAudioBoardBridge
from .cine_audio_info import IAMCCS_CineAudioInfo

NODE_CLASS_MAPPINGS = {
    "IAMCCS_AudioBoardArranger": IAMCCS_AudioBoardArranger,
    "IAMCCS_BusOut": IAMCCS_BusOut,
    "IAMCCS_AudioBoardMixer": IAMCCS_AudioBoardMixer,
    "IAMCCS_ControlAudEfx": IAMCCS_ControlAudEfx,
    "IAMCCS_DialogueTagEditor": IAMCCS_DialogueTagEditor,
    "IAMCCS_DialogueAudioBoardBridge": IAMCCS_DialogueAudioBoardBridge,
    "IAMCCS_CineAudioInfo": IAMCCS_CineAudioInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_AudioBoardArranger": "IAMCCS AudioBoard Arranger",
    "IAMCCS_BusOut": "IAMCCS BusOut",
    "IAMCCS_AudioBoardMixer": "IAMCCS AudioBoard Mixer",
    "IAMCCS_ControlAudEfx": "IAMCCS ControlAudEfx",
    "IAMCCS_DialogueTagEditor": "IAMCCS Dialogue Tag Editor",
    "IAMCCS_DialogueAudioBoardBridge": "IAMCCS Dialogue AudioBoard Bridge",
    "IAMCCS_CineAudioInfo": "IAMCCS CineAudioInfo",
}
