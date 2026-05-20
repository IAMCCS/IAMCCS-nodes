from __future__ import annotations

import json
from typing import Any


class IAMCCS_CineStage2PreviewToggle:
    """Gate KJ/LTX sampling previews on the model branch that feeds Stage 2."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "stage2_preview_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "OFF removes the KJ LTX2 sampling_preview wrapper from this model branch. "
                        "Use it before Stage 2 to avoid TAELTX preview OOM while keeping the render active."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "toggle"
    CATEGORY = "IAMCCS/Cine/02 Single Generation"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    @staticmethod
    def _count_sampling_preview_wrappers(model: Any) -> int:
        try:
            import comfy.patcher_extension as patcher_extension

            wrappers = model.get_wrappers(patcher_extension.WrappersMP.OUTER_SAMPLE, "sampling_preview")
            return len(wrappers or [])
        except Exception:
            return 0

    def toggle(self, model: Any, stage2_preview_enabled: bool = False):
        before = self._count_sampling_preview_wrappers(model)
        selected_model = model
        removed = 0
        if not bool(stage2_preview_enabled):
            try:
                import comfy.patcher_extension as patcher_extension

                selected_model = model.clone()
                selected_model.remove_wrappers_with_key(
                    patcher_extension.WrappersMP.OUTER_SAMPLE,
                    "sampling_preview",
                )
                after = self._count_sampling_preview_wrappers(selected_model)
                removed = max(0, int(before) - int(after))
            except Exception as exc:
                report = {
                    "node": "IAMCCS_CineStage2PreviewToggle",
                    "stage2_preview_enabled": bool(stage2_preview_enabled),
                    "status": "failed_to_remove_wrapper",
                    "sampling_preview_wrappers_before": int(before),
                    "error": str(exc),
                }
                return model, json.dumps(report, indent=2)
        else:
            after = before

        report = {
            "node": "IAMCCS_CineStage2PreviewToggle",
            "stage2_preview_enabled": bool(stage2_preview_enabled),
            "status": "preview_passthrough" if bool(stage2_preview_enabled) else "stage2_preview_wrapper_removed",
            "sampling_preview_wrappers_before": int(before),
            "sampling_preview_wrappers_after": int(after),
            "sampling_preview_wrappers_removed": int(removed),
            "truth": (
                "Place this node only on the Stage 2 model branch. OFF disables the KJ/TAELTX sampling "
                "preview callback for Stage 2 without disabling PromptRelay, guide data, or Stage 2 sampling."
            ),
        }
        return selected_model, json.dumps(report, indent=2)

