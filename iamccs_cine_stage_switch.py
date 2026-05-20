from __future__ import annotations

import json
from typing import Any


class IAMCCS_CineStage2BypassSwitch:
    """Lazy switch for choosing Stage 1 or Stage 2 latents before decode."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_second_stage": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "True: decode Stage 2 output. False: bypass Stage 2 and decode Stage 1 output.",
                }),
            },
            "optional": {
                "stage1_audio_latent": ("LATENT", {"lazy": True}),
                "stage1_video_latent": ("LATENT", {"lazy": True}),
                "stage2_audio_latent": ("LATENT", {"lazy": True}),
                "stage2_video_latent": ("LATENT", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT", "STRING")
    RETURN_NAMES = ("audio_latent", "video_latent", "report")
    FUNCTION = "switch"
    CATEGORY = "IAMCCS/Cine/02 Single Generation"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    def check_lazy_status(
        self,
        use_second_stage: bool,
        stage1_audio_latent: Any = None,
        stage1_video_latent: Any = None,
        stage2_audio_latent: Any = None,
        stage2_video_latent: Any = None,
        **kwargs,
    ):
        if bool(use_second_stage):
            needed = []
            if stage2_audio_latent is None:
                needed.append("stage2_audio_latent")
            if stage2_video_latent is None:
                needed.append("stage2_video_latent")
            return needed
        needed = []
        if stage1_audio_latent is None:
            needed.append("stage1_audio_latent")
        if stage1_video_latent is None:
            needed.append("stage1_video_latent")
        return needed

    def switch(
        self,
        use_second_stage: bool,
        stage1_audio_latent: Any = None,
        stage1_video_latent: Any = None,
        stage2_audio_latent: Any = None,
        stage2_video_latent: Any = None,
    ):
        if bool(use_second_stage):
            if stage2_audio_latent is None or stage2_video_latent is None:
                raise ValueError("Stage 2 is enabled, but Stage 2 latents are not connected.")
            report = {
                "node": "IAMCCS_CineStage2BypassSwitch",
                "use_second_stage": True,
                "selected": "stage2",
            }
            return stage2_audio_latent, stage2_video_latent, json.dumps(report, indent=2)
        if stage1_audio_latent is None or stage1_video_latent is None:
            raise ValueError("Stage 2 bypass is enabled, but Stage 1 latents are not connected.")
        report = {
            "node": "IAMCCS_CineStage2BypassSwitch",
            "use_second_stage": False,
            "selected": "stage1_bypass",
        }
        return stage1_audio_latent, stage1_video_latent, json.dumps(report, indent=2)
