"""Lazy bridge between IAMCCS H3 universal workflows and Viggle-Animate-H3.

The bridge deliberately does not import Viggle's implementation.  It only
exposes IAMCCS media already carried by CineLinX and selects which completed
branch is allowed to reach the editor.  Consequently the established R42/R43
generation branches stay untouched and Viggle remains an optional dependency.
"""

from copy import deepcopy
from typing import Any

import torch

from .iamccs_minimax_h3_universal_delivery import (
    IAMCCS_MiniMaxH3UniversalPathRouterR42,
)


SUPERNODE_LINX_TYPE = "IAMCCS_SUPERNODE_LINX"
CATEGORY = "IAMCCS/MiniMax H3/Viggle"
VIGGLE_MODE = "viggle_animation"
STANDARD_MODE = "standard_universal"


def _resources(cine_linx: Any) -> dict[str, Any]:
    if not isinstance(cine_linx, dict):
        return {}
    resources = cine_linx.get("resources")
    return resources if isinstance(resources, dict) else {}


def _shotplan_entry(cine_linx: Any):
    resources = _resources(cine_linx)
    for key in ("iamccs_minimax_h3_shotplan", "minimax_h3_shotplan", "shotplan"):
        plan = resources.get(key)
        if isinstance(plan, dict) and str(plan.get("schema", "")).startswith("iamccs.minimax_h3.shotplan"):
            return key, plan
    return None, None


def _audio_ok(audio: Any) -> bool:
    return isinstance(audio, dict) and torch.is_tensor(audio.get("waveform"))


class IAMCCS_MiniMaxH3UniversalPathRouterEditorR42(
    IAMCCS_MiniMaxH3UniversalPathRouterR42
):
    """The established R42 path router without its terminal-output flag.

    The editor/Viggle selector becomes the sole terminal path in the augmented
    workflows, which is what makes the two large branches genuinely lazy.
    """

    OUTPUT_NODE = False


class IAMCCS_ViggleCineMedia:
    """Read Viggle driving media from the existing Cine Info H3 transport."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"cine_linx": (SUPERNODE_LINX_TYPE,)}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "AUDIO", "INT", "STRING")
    RETURN_NAMES = (
        "driving_video",
        "reference_image",
        "driving_audio",
        "fps",
        "report",
    )
    FUNCTION = "extract"
    CATEGORY = CATEGORY

    def extract(self, cine_linx):
        resources = _resources(cine_linx)
        driving = resources.get("iamccs_minimax_h3_ref_video")
        reference = resources.get("iamccs_minimax_h3_ref_image_1")
        audio = resources.get("iamccs_minimax_h3_ref_video_audio")

        if not torch.is_tensor(driving) or driving.ndim != 4 or int(driving.shape[0]) < 1:
            raise ValueError(
                "Viggle mode needs a driving video. Connect Load Video IMAGE to "
                "Cine Info H3 > reference_video."
            )
        if not torch.is_tensor(reference) or reference.ndim != 4 or int(reference.shape[0]) < 1:
            raise ValueError(
                "Viggle mode needs a character image. Connect Load Image to "
                "Cine Info H3 > reference_image_1."
            )

        fps = 24
        if not _audio_ok(audio):
            samples = max(1, int(round(int(driving.shape[0]) / fps * 32000)))
            audio = {
                "waveform": torch.zeros((1, 2, samples), dtype=torch.float32),
                "sample_rate": 32000,
            }
            audio_state = "silent fallback"
        else:
            audio_state = "source video audio"

        report = (
            f"Viggle Cine media | driving={int(driving.shape[0])} frames | "
            f"reference={int(reference.shape[0])} image(s) | fps={fps} | audio={audio_state}"
        )
        return driving, reference[:1], audio, fps, report


class IAMCCS_MiniMaxH3UniversalViggleSelectorR42:
    """Lazily select the complete universal graph or the Viggle graph."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "mode": ([STANDARD_MODE, VIGGLE_MODE], {"default": STANDARD_MODE}),
            },
            "optional": {
                "standard_path": ("STRING", {"lazy": True}),
                "standard_frames": ("IMAGE", {"lazy": True}),
                "standard_audio": ("AUDIO", {"lazy": True}),
                "standard_current_segment": ("INT", {"lazy": True}),
                "standard_total_segments": ("INT", {"lazy": True}),
                "standard_fps": ("INT", {"lazy": True}),
                "standard_render_id": ("STRING", {"lazy": True}),
                "viggle_frames": ("IMAGE", {"lazy": True}),
                "viggle_audio": ("AUDIO", {"lazy": True}),
            },
        }

    RETURN_TYPES = (
        SUPERNODE_LINX_TYPE,
        "STRING",
        "IMAGE",
        "AUDIO",
        "INT",
        "INT",
        "INT",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "editor_cine_linx",
        "video_path",
        "frames",
        "audio",
        "current_segment",
        "total_segments",
        "fps",
        "resolved_render_id",
        "report",
    )
    FUNCTION = "select"
    OUTPUT_NODE = False
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    def check_lazy_status(self, cine_linx, mode, **kwargs):
        if str(mode) == VIGGLE_MODE:
            names = ("viggle_frames", "viggle_audio")
        else:
            names = (
                "standard_path",
                "standard_frames",
                "standard_audio",
                "standard_current_segment",
                "standard_total_segments",
                "standard_fps",
                "standard_render_id",
            )
        return [name for name in names if kwargs.get(name) is None]

    @staticmethod
    def _viggle_cine_linx(cine_linx, frame_count: int, fps: int):
        key, original_plan = _shotplan_entry(cine_linx)
        if key is None or original_plan is None:
            raise ValueError("Viggle selector requires an IAMCCS H3 shotplan in CineLinX")

        adapted = dict(cine_linx)
        resources = dict(_resources(cine_linx))
        plan = deepcopy(original_plan)
        plan["task_mode"] = VIGGLE_MODE
        plan["fps"] = int(fps)
        plan["requested_duration_seconds"] = float(frame_count) / float(fps)
        plan["total_unique_frames"] = int(frame_count)
        plan["viggle_source"] = "Cine Info H3 reference_video + reference_image_1"
        plan["chunks"] = [{
            "index": 0,
            "slot_index": 0,
            "slot_label": "Viggle animation",
            "frame_count": int(frame_count),
            "requested_frame_count": int(frame_count),
            "unique_frames": int(frame_count),
            "timeline_start_frame": 0,
            "trim_head_frames": 0,
            "pre_roll_frames": 0,
            "post_roll_frames": 0,
        }]
        resources[key] = plan
        adapted["resources"] = resources
        adapted["mode"] = VIGGLE_MODE
        adapted["active_stage"] = "IAMCCS Universal Viggle Selector R42/R43"
        return adapted

    def select(self, cine_linx, mode, **kwargs):
        selected_mode = str(mode or STANDARD_MODE)
        if selected_mode != VIGGLE_MODE:
            frames = kwargs.get("standard_frames")
            audio = kwargs.get("standard_audio")
            if not torch.is_tensor(frames) or frames.ndim != 4:
                raise ValueError("Universal selector received no standard IMAGE frames")
            if not _audio_ok(audio):
                raise ValueError("Universal selector received no standard AUDIO")
            current = max(0, int(kwargs.get("standard_current_segment") or 0))
            total = max(1, int(kwargs.get("standard_total_segments") or 1))
            fps = max(1, int(kwargs.get("standard_fps") or 24))
            path = str(kwargs.get("standard_path") or "")
            render_id = str(kwargs.get("standard_render_id") or "")
            report = (
                f"Universal/Viggle selector | mode={STANDARD_MODE} | "
                f"segment={current + 1}/{total} | fps={fps}"
            )
            return cine_linx, path, frames, audio, current, total, fps, render_id, report

        frames = kwargs.get("viggle_frames")
        audio = kwargs.get("viggle_audio")
        if not torch.is_tensor(frames) or frames.ndim != 4 or int(frames.shape[0]) < 1:
            raise ValueError("Viggle selector received no generated IMAGE frames")
        if not _audio_ok(audio):
            raise ValueError("Viggle selector received no source AUDIO")
        fps = 24
        adapted = self._viggle_cine_linx(cine_linx, int(frames.shape[0]), fps)
        report = (
            f"Universal/Viggle selector | mode={VIGGLE_MODE} | "
            f"frames={int(frames.shape[0])} | fps={fps} | source audio preserved"
        )
        return adapted, "", frames, audio, 0, 1, fps, "viggle_animation", report


NODE_CLASS_MAPPINGS = {
    "IAMCCS_MiniMaxH3UniversalPathRouterEditorR42": IAMCCS_MiniMaxH3UniversalPathRouterEditorR42,
    "IAMCCS_ViggleCineMedia": IAMCCS_ViggleCineMedia,
    "IAMCCS_MiniMaxH3UniversalViggleSelectorR42": IAMCCS_MiniMaxH3UniversalViggleSelectorR42,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_MiniMaxH3UniversalPathRouterEditorR42": "R42/R43 Universal Path → Editor (lazy)",
    "IAMCCS_ViggleCineMedia": "IAMCCS Viggle · Cine Video + Image",
    "IAMCCS_MiniMaxH3UniversalViggleSelectorR42": "R42/R43 · Universal or Viggle Animation",
}
