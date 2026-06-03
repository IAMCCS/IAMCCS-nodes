# Legacy compatibility shim. The active app-style nodes live in dialogue_tag_editor.py.
from .dialogue_tag_editor import IAMCCS_DialogueTagEditor, IAMCCS_DialogueAudioBoardBridge

NODE_CLASS_MAPPINGS = {
    "IAMCCS_DialogueTagEditor": IAMCCS_DialogueTagEditor,
    "IAMCCS_DialogueAudioBoardBridge": IAMCCS_DialogueAudioBoardBridge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_DialogueTagEditor": "IAMCCS Dialogue Tag Editor",
    "IAMCCS_DialogueAudioBoardBridge": "IAMCCS Dialogue AudioBoard Bridge",
}
