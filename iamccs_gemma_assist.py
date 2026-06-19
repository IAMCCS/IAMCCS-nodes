import os
import time


class IAMCCS_GemmaAssistLazyGate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": False}),
                "disabled_text": ("STRING", {"multiline": True, "default": "Gemma Assist disabled. Enable this gate to run only the assistant branch."}),
            },
            "optional": {
                "gemma_text": ("STRING", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("text", "enabled")
    FUNCTION = "run"
    CATEGORY = "IAMCCS/Cine/Ideogram"

    def check_lazy_status(self, enabled, disabled_text="", gemma_text=None, **kwargs):
        if bool(enabled) and gemma_text is None:
            return ["gemma_text"]
        return []

    def run(self, enabled=False, disabled_text="", gemma_text=None):
        if bool(enabled):
            return (str(gemma_text or ""), True)
        return (str(disabled_text or ""), False)


class IAMCCS_GemmaAssistOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "IAMCCS/gemma_assist/sheet_suggestion"}),
                "save_text": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "save"
    CATEGORY = "IAMCCS/Cine/Ideogram"
    OUTPUT_NODE = True

    def save(self, text, filename_prefix="IAMCCS/gemma_assist/sheet_suggestion", save_text=True):
        value = str(text or "")
        path = ""
        if bool(save_text):
            try:
                import folder_paths

                out_dir = folder_paths.get_output_directory()
                prefix = str(filename_prefix or "IAMCCS/gemma_assist/sheet_suggestion").replace("\\", "/").strip("/")
                directory = os.path.join(out_dir, os.path.dirname(prefix))
                os.makedirs(directory, exist_ok=True)
                stem = os.path.basename(prefix) or "sheet_suggestion"
                path = os.path.join(directory, f"{stem}_{int(time.time())}.txt")
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(value)
            except Exception as err:
                value = value + f"\n\n[IAMCCS_GemmaAssistOutput save error: {err}]"
        return {"ui": {"text": [value], "path": [path] if path else []}, "result": (value,)}


NODE_CLASS_MAPPINGS = {
    "IAMCCS_GemmaAssistLazyGate": IAMCCS_GemmaAssistLazyGate,
    "IAMCCS_GemmaAssistOutput": IAMCCS_GemmaAssistOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_GemmaAssistLazyGate": "IAMCCS Gemma Assist Lazy Gate",
    "IAMCCS_GemmaAssistOutput": "IAMCCS Gemma Assist Output",
}
