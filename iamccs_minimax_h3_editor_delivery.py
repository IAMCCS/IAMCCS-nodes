"""Delivery-file bridge for the isolated R39/R40/R42 editor workflows.

The production R39/R40 delivery nodes intentionally write an MP4 and return its
path.  The IAMCCS editor, on the other hand, consumes decoded IMAGE/AUDIO media.
This adapter keeps those contracts separate and never changes the established
R37/R39 generation or editor nodes.
"""

from pathlib import Path
from typing import Any, Dict

import torch

from comfy_api.latest import InputImpl


SUPERNODE_LINX_TYPE = "IAMCCS_SUPERNODE_LINX"
CATEGORY = "IAMCCS/MiniMax H3/Editor"


def _shotplan(cine_linx: Any) -> Dict[str, Any]:
    if not isinstance(cine_linx, dict):
        return {}
    for container_name in ("resources", "outputs", "payload"):
        container = cine_linx.get(container_name)
        if not isinstance(container, dict):
            continue
        for key in ("iamccs_minimax_h3_shotplan", "minimax_h3_shotplan", "shotplan"):
            value = container.get(key)
            if isinstance(value, dict) and str(value.get("schema", "")).startswith("iamccs.minimax_h3.shotplan"):
                return value
    return {}


def _audio_ok(audio: Any) -> bool:
    return isinstance(audio, dict) and torch.is_tensor(audio.get("waveform"))


class IAMCCS_MiniMaxH3EditorDeliveryMedia:
    """Decode the selected per-shot or one-film delivery for the editor.

    Non-LongVid modes publish each rendered ``segment_NNNN.mp4`` to the chosen
    editor lane.  LongVid publishes only the completed ``final_film.mp4`` as one
    asset.  This mirrors IAMCCS_MiniMaxH3EditorTakeRoute semantics while ensuring
    the editor receives the upscaled delivery when delivery is enabled.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "delivery_video_path": ("STRING", {"forceInput": True}),
                "current_segment": ("INT", {"forceInput": True}),
                "total_segments": ("INT", {"forceInput": True}),
                "native_frames": ("IMAGE",),
                "native_audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("delivery_frames", "delivery_audio", "master_ready", "resolved_path", "report")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # The path can be reused by an automatic segment loop.  Always verify
        # what exists on disk rather than caching an intermediate readiness flag.
        return float("nan")

    def load(self, cine_linx, delivery_video_path, current_segment, total_segments, native_frames, native_audio):
        plan = _shotplan(cine_linx)
        if not plan:
            raise ValueError("H3 Editor Delivery requires an IAMCCS H3 shotplan in cine_linx")

        index = max(0, int(current_segment))
        total = max(1, int(total_segments))
        task_mode = str(plan.get("task_mode", "") or "").strip().lower()
        upscale_enabled = bool(plan.get("upscale_enabled", False))
        upscale_mode = str(plan.get("upscale_mode", "off") or "off").strip().lower()
        editorial_per_shot = upscale_enabled and upscale_mode == "ltx23_per_chunk"
        # A normal LongVid remains one editorial asset.  The explicit LTX
        # per-shot contract is the exception: every generated H3 chunk is a
        # separately decoded roll so pre/post-roll and cut decisions remain
        # available inside the Video Editor.
        longvid = task_mode.startswith("longvid") and not editorial_per_shot
        authored = Path(str(delivery_video_path or "").strip())

        if longvid and index + 1 < total:
            return (
                native_frames,
                native_audio,
                False,
                "",
                f"H3 editor delivery waiting for LongVid master | completed={index + 1}/{total}",
            )

        if longvid:
            resolved = authored
        else:
            # On the final pass the delivery node returns final_film.mp4.  The
            # editor still needs the current shot, not a duplicate concatenated
            # programme, so resolve the deterministic per-segment asset.
            resolved = authored.parent / f"segment_{index + 1:04d}.mp4"
            if not resolved.is_file() and authored.name.lower() != "final_film.mp4":
                resolved = authored

        if not resolved.is_file():
            raise FileNotFoundError(f"H3 editor delivery checkpoint not found: {resolved}")

        components = InputImpl.VideoFromFile(str(resolved)).get_components()
        frames = components.images
        audio = components.audio if _audio_ok(components.audio) else native_audio
        if not torch.is_tensor(frames) or frames.ndim != 4 or int(frames.shape[0]) < 1:
            raise ValueError(f"H3 editor delivery decoded no frames from: {resolved}")
        if not _audio_ok(audio):
            raise ValueError(f"H3 editor delivery decoded no audio from: {resolved}")

        master_ready = bool(longvid and index + 1 >= total)
        delivery_kind = (
            "LongVid final master"
            if longvid
            else (f"LTX editorial roll {index + 1}/{total}" if editorial_per_shot else f"slot {index + 1}/{total}")
        )
        report = (
            f"H3 editor delivery checkpoint ready | {delivery_kind} | "
            f"frames={int(frames.shape[0])} | fps={float(components.frame_rate):.3f} | path={resolved}"
        )
        return frames, audio, master_ready, str(resolved), report


NODE_CLASS_MAPPINGS = {
    "IAMCCS_MiniMaxH3EditorDeliveryMedia": IAMCCS_MiniMaxH3EditorDeliveryMedia,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_MiniMaxH3EditorDeliveryMedia": "MiniMax H3 · Delivery Checkpoint → Editor Chunk",
}
