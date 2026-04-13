class IAMCCS_MultilinePromptSplitter8:
    DISPLAY_NAME = "IAMCCS Multiline Prompt Splitter 8"
    CATEGORY = "IAMCCS/Prompt"
    FUNCTION = "split_prompts"

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "INT",
    )
    RETURN_NAMES = (
        "prompt_1",
        "prompt_2",
        "prompt_3",
        "prompt_4",
        "prompt_5",
        "prompt_6",
        "prompt_7",
        "prompt_8",
        "count",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "separator": ("STRING", {"default": "\\n", "multiline": False}),
                "strip_empty": ("BOOLEAN", {"default": True}),
                "fill_mode": (("empty", "repeat_last", "wrap"), {"default": "empty"}),
            },
        }

    def split_prompts(self, text, separator="\\n", strip_empty=True, fill_mode="empty"):
        sep = (separator or "\\n").replace("\\n", "\n")
        raw_parts = (text or "").split(sep)

        if strip_empty:
            prompts = [part.strip() for part in raw_parts if part.strip()]
        else:
            prompts = [part.strip() for part in raw_parts]

        count = len(prompts)
        outputs = list(prompts[:8])

        while len(outputs) < 8:
            if fill_mode == "repeat_last" and prompts:
                outputs.append(prompts[-1])
            elif fill_mode == "wrap" and prompts:
                outputs.append(prompts[len(outputs) % len(prompts)])
            else:
                outputs.append("")

        return (*outputs, count)