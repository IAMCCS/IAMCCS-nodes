class IAMCCS_IntValueMonitor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": -2147483648, "max": 2147483647, "step": 1}),
                "label": ("STRING", {"default": "Qwen batch count"}),
            }
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("value", "report")
    FUNCTION = "inspect"
    CATEGORY = "IAMCCS/Utils"
    OUTPUT_NODE = True

    def inspect(self, value, label="Qwen batch count"):
        current_value = int(value)
        report = f"{label}: {current_value}"
        return {"ui": {"text": [report]}, "result": (current_value, report)}
