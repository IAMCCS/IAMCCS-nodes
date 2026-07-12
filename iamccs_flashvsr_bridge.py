import math

import torch


class IAMCCS_FlashVSRPanelBatchPrep:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "min_frames": ("INT", {"default": 24, "min": 21, "max": 96, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("flashvsr_frames", "original_count", "repeat_factor", "report")
    FUNCTION = "prepare"
    CATEGORY = "IAMCCS/Ideogram"

    def prepare(self, images, min_frames=24):
        if images is None or images.shape[0] <= 0:
            raise ValueError("IAMCCS_FlashVSRPanelBatchPrep received an empty IMAGE batch")

        original_count = int(images.shape[0])
        min_frames = max(21, int(min_frames))
        repeat_factor = max(1, int(math.ceil(min_frames / float(original_count))))

        if repeat_factor > 1:
            frames = torch.repeat_interleave(images, repeat_factor, dim=0).contiguous()
        else:
            frames = images.contiguous()

        report = (
            f"IAMCCS FlashVSR prep: original_count={original_count}, "
            f"repeat_factor={repeat_factor}, flashvsr_frames={int(frames.shape[0])}, "
            f"shape={tuple(frames.shape)}"
        )
        return (frames, original_count, repeat_factor, report)


class IAMCCS_FlashVSRPanelBatchRestore:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "original_count": ("INT", {"default": 6, "min": 1, "max": 512, "step": 1}),
                "repeat_factor": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "restore"
    CATEGORY = "IAMCCS/Ideogram"

    def restore(self, frames, original_count=6, repeat_factor=4):
        if frames is None or frames.shape[0] <= 0:
            raise ValueError("IAMCCS_FlashVSRPanelBatchRestore received an empty IMAGE batch")

        original_count = max(1, int(original_count))
        repeat_factor = max(1, int(repeat_factor))
        total = int(frames.shape[0])

        picks = [min(i * repeat_factor, total - 1) for i in range(original_count)]
        idx = torch.tensor(picks, dtype=torch.long, device=frames.device)
        restored = frames.index_select(0, idx).contiguous()

        report = (
            f"IAMCCS FlashVSR restore: input_frames={total}, "
            f"original_count={original_count}, repeat_factor={repeat_factor}, "
            f"picked_indexes={picks}, output_shape={tuple(restored.shape)}"
        )
        return (restored, report)


NODE_CLASS_MAPPINGS = {
    "IAMCCS_FlashVSRPanelBatchPrep": IAMCCS_FlashVSRPanelBatchPrep,
    "IAMCCS_FlashVSRPanelBatchRestore": IAMCCS_FlashVSRPanelBatchRestore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_FlashVSRPanelBatchPrep": "IAMCCS FlashVSR Panel Batch Prep",
    "IAMCCS_FlashVSRPanelBatchRestore": "IAMCCS FlashVSR Panel Batch Restore",
}
