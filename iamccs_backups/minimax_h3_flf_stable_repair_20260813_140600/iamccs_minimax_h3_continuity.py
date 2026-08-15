"""MiniMax H3 decoded-frame continuity settings for FLF chunk handoffs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

import folder_paths


MOTION_CONTEXT_TYPE = "IAMCCS_H3_MOTION_CONTEXT"
_CONTEXT_FRAMES = (22, 39, 56)
REFERENCE_MOTION_CARRY = "reference_motion_carry"
HYBRID_HARD_ANCHOR = "hybrid_hard_anchor"


class IAMCCS_MiniMaxH3MotionContext:
    """Configure stock REF2VA carry-over from the preceding FLF chunk."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": False}),
                # Keep this physical slot stable so existing workflows preserve
                # their selected temporal tail size.
                "motion_tail_frames": (["22", "39", "56"], {"default": "22"}),
                "continue_audio": ("BOOLEAN", {"default": True}),
                "unused_legacy_freeze_safe_handover": ("BOOLEAN", {"default": True}),
                "unused_legacy_freeze_safety_margin": ("INT", {"default": 3, "min": 0, "max": 24, "step": 1}),
                # Append-only: keeps historical widgets_values positions intact.
                "audio_tail_seconds": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 15.0, "step": 0.5}),
                "continuity_strategy": ([REFERENCE_MOTION_CARRY, HYBRID_HARD_ANCHOR], {"default": REFERENCE_MOTION_CARRY}),
            }
        }

    RETURN_TYPES = (MOTION_CONTEXT_TYPE, "STRING")
    RETURN_NAMES = ("motion_context", "report")
    FUNCTION = "configure"
    CATEGORY = "IAMCCS/MiniMax H3/Continuity"

    def configure(
        self,
        enabled=False,
        motion_tail_frames="22",
        continue_audio=True,
        unused_legacy_freeze_safe_handover=True,
        unused_legacy_freeze_safety_margin=3,
        audio_tail_seconds=4.0,
        continuity_strategy=REFERENCE_MOTION_CARRY,
    ):
        # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        del unused_legacy_freeze_safe_handover, unused_legacy_freeze_safety_margin
        config = {
            "schema": "iamccs.minimax_h3.motion_context",
            "schema_version": 6,
            "method": "decoded_frame_reference_motion_carry",
            "enabled": bool(enabled),
            "continue_audio": bool(continue_audio),
            "audio_tail_seconds": max(0.5, float(audio_tail_seconds)),
            "continuity_strategy": continuity_strategy if continuity_strategy in {REFERENCE_MOTION_CARRY, HYBRID_HARD_ANCHOR} else REFERENCE_MOTION_CARRY,
            "motion_tail_frames": max(5, int(motion_tail_frames)),
        }
        # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        report = (
            f"H3 {config['continuity_strategy']} | {'on' if config['enabled'] else 'off'} | "
            f"motion_tail={config['motion_tail_frames']}f | "
            f"audio={'on' if config['continue_audio'] else 'off'} "
            f"tail={config['audio_tail_seconds']:.1f}s"
        )
        return config, report


def is_active(config: Any, segment_index: Any) -> bool:
    return isinstance(config, dict) and bool(config.get("enabled")) and int(segment_index) > 0


def continuity_strategy(config: Any) -> str:
    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    if isinstance(config, dict) and config.get("continuity_strategy") == HYBRID_HARD_ANCHOR:
        return HYBRID_HARD_ANCHOR
    return REFERENCE_MOTION_CARRY


def uses_reference_motion_carry(config: Any, segment_index: Any) -> bool:
    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    return is_active(config, segment_index) and continuity_strategy(config) == REFERENCE_MOTION_CARRY


def _cache_path(render_id: str, segment_index: int) -> Path | None:
    safe = str(render_id or "").strip()
    if not safe or int(segment_index) < 0:
        return None
    return Path(folder_paths.get_output_directory()) / "minimax_h3_shotboard" / "motion_context" / f"{safe}_seg_{int(segment_index):04d}_media.pt"


def save_reference_media(
    render_id: str,
    segment_index: int,
    images: torch.Tensor,
    audio: dict[str, Any] | None,
    max_video_frames: int | None = None,
) -> bool:
    """Persist the previous chunk's decoded frames + audio for the next chunk's carry-over."""
    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    path = _cache_path(render_id, segment_index)
    if path is None or not torch.is_tensor(images) or images.ndim != 4 or int(images.shape[0]) < 1:
        return False
    cached_images = images
    if max_video_frames is not None:
        cached_images = images[-max(1, int(max_video_frames)):, ...]
    waveform = audio.get("waveform") if isinstance(audio, dict) else None
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "segment_index": int(segment_index),
        "images": cached_images.detach().to(device="cpu", dtype=torch.float16, copy=True).contiguous(),
        "audio_waveform": waveform.detach().to(device="cpu", copy=True).contiguous() if torch.is_tensor(waveform) else None,
        "audio_sample_rate": int(audio.get("sample_rate", 32000)) if isinstance(audio, dict) else 32000,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return True


def load_reference_media(render_id: str, segment_index: int) -> dict[str, Any] | None:
    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    path = _cache_path(render_id, segment_index)
    if path is None or not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return None
    if (
        not isinstance(payload, dict)
        or int(payload.get("segment_index", -1)) != int(segment_index)
        or not torch.is_tensor(payload.get("images"))
    ):
        return None
    return payload


def build_reference_carry_over(
    previous: dict[str, Any],
    *,
    continue_audio: bool,
    audio_tail_seconds: float,
    motion_tail_frames: int | None = None,
    max_video_frames: int | None = None,
) -> dict[str, Any]:
    """Turn a cached previous-chunk decode into REF2VA carry-over inputs.

    Returns ``ref_video`` (previous clip, capped to its own tail), ``ref_image``
    (its last frame, a continuity anchor) and optional ``ref_audio`` (its tail).

    Stock ``MiniMaxH3ReferenceToVideo`` truncates an over-length ``ref_video``
    to its *first* ``frame_count`` frames (see ``comfy_extras/nodes_minimax_h3.py``).
    Handing it the whole previous clip therefore references that clip's
    opening composition, not its ending. ``max_video_frames`` (the next
    segment's own frame count) keeps only the true tail so the stock
    truncation is a no-op and the real end is what gets referenced.
    """
    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    images = previous["images"].to(dtype=torch.float32)
    video = images
    cap = int(video.shape[0])
    if max_video_frames is not None:
        cap = min(cap, max(5, int(max_video_frames)))
    if motion_tail_frames is not None:
        cap = min(cap, max(5, int(motion_tail_frames)))
    if int(video.shape[0]) > cap:
        video = video[-cap:]
    result: dict[str, Any] = {
        "ref_video": video,
        "ref_image": images[-1:].clone(),
        "ref_audio": None,
    }
    waveform = previous.get("audio_waveform")
    if continue_audio and torch.is_tensor(waveform) and int(waveform.shape[-1]) > 0:
        sample_rate = max(1, int(previous.get("audio_sample_rate", 32000)))
        tail_samples = min(int(waveform.shape[-1]), max(1, int(round(float(audio_tail_seconds) * sample_rate))))
        result["ref_audio"] = {"waveform": waveform[..., waveform.shape[-1] - tail_samples:].clone(), "sample_rate": sample_rate}
    return result


NODE_CLASS_MAPPINGS = {
    "IAMCCS_MiniMaxH3MotionContext": IAMCCS_MiniMaxH3MotionContext,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_MiniMaxH3MotionContext": "MiniMax H3 FLF Motion Carry-Over",
}
