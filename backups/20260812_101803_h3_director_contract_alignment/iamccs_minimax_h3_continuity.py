"""IAMCCS-native temporal continuity contract for MiniMax H3 chunk handoffs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

import folder_paths


MOTION_CONTEXT_TYPE = "IAMCCS_H3_MOTION_CONTEXT"
_CONTEXT_FRAMES = (22, 39, 56)
_FRAME_PATTERN = (1, 4, 4, 4, 4)
_FPS = 24.0
_AUDIO_HZ = 40.0
_AUDIO_CONTEXT_FRAMES = 24
_FRAME_MARKER = "iamccs_h3_context_frame"
_AUDIO_MARKER = "iamccs_h3_context_audio_end"
_LAYOUT_MARKER = "_iamccs_h3_context_layout_patch"
_PAYLOAD_MARKER = "_iamccs_h3_context_payload_patch"
_layout_original = None
_payload_original = None


class IAMCCS_MiniMaxH3MotionContext:
    """Explicit, append-only configuration node for native H3 temporal handoff."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": False}),
                "context_frames": (["22", "39", "56"], {"default": "22"}),
                "continue_audio": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = (MOTION_CONTEXT_TYPE, "STRING")
    RETURN_NAMES = ("motion_context", "report")
    FUNCTION = "configure"
    CATEGORY = "IAMCCS/MiniMax H3/Continuity"

    def configure(self, enabled=False, context_frames="22", continue_audio=True):
        # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        frames = normalize_frames(context_frames)
        config = {
            "schema": "iamccs.minimax_h3.motion_context",
            "schema_version": 1,
            "enabled": bool(enabled),
            "frames": frames,
            "continue_audio": bool(continue_audio),
        }
        report = f"H3 motion context | {'on' if config['enabled'] else 'off'} | {frames}f | audio={'on' if config['continue_audio'] else 'off'}"
        return config, report


def normalize_frames(value: Any) -> int:
    try:
        frame_count = int(str(value).rstrip("fF"))
    except (TypeError, ValueError):
        return 22
    return frame_count if frame_count in _CONTEXT_FRAMES else 22


def is_active(config: Any, segment_index: Any) -> bool:
    return isinstance(config, dict) and bool(config.get("enabled")) and int(segment_index) > 0


def frame_budget(visible_frames: Any, context_frames: Any) -> tuple[int, int]:
    """Return legal H3 sample size and the exact context prefix to trim."""
    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    visible = max(5, int(visible_frames))
    context = normalize_frames(context_frames)
    sampled = visible + context
    while sampled % 17 != 5:
        sampled += 1
    return sampled, context


def _cache_path(render_id: str) -> Path | None:
    safe = str(render_id or "").strip()
    if not safe:
        return None
    return Path(folder_paths.get_output_directory()) / "minimax_h3_shotboard" / "motion_context" / f"{safe}.pt"


def save_sampled_av(render_id: str, sampled: dict[str, Any], export_start_frames: int, export_frames: int) -> bool:
    """Persist both AV streams on CPU; NestedTensor has no tensor-level detach()."""
    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    path = _cache_path(render_id)
    samples = sampled.get("samples") if isinstance(sampled, dict) else None
    if path is None or samples is None or not hasattr(samples, "unbind"):
        return False
    streams = list(samples.unbind())
    if len(streams) != 2 or not all(torch.is_tensor(stream) for stream in streams):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video": streams[0].detach().to(device="cpu", copy=True).contiguous(),
        "audio": streams[1].detach().to(device="cpu", copy=True).contiguous(),
        "export_start_frames": int(export_start_frames),
        "export_frames": int(export_frames),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return True


def load_sampled_av(render_id: str) -> dict[str, Any] | None:
    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    path = _cache_path(render_id)
    if path is None or not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return None
    if not isinstance(payload, dict) or not torch.is_tensor(payload.get("video")) or not torch.is_tensor(payload.get("audio")):
        return None
    return payload


def _pixel_starts(latent_steps: int) -> list[int]:
    starts: list[int] = []
    current = 0
    for index in range(int(latent_steps)):
        starts.append(current)
        current += _FRAME_PATTERN[index % len(_FRAME_PATTERN)]
    return starts


def _pixel_frames(latent_steps: int) -> int:
    return sum(_FRAME_PATTERN[index % len(_FRAME_PATTERN)] for index in range(int(latent_steps)))


def _steps_for_frames(frame_count: int) -> int:
    covered = 0
    for step in range(1, 100000):
        covered += _FRAME_PATTERN[(step - 1) % len(_FRAME_PATTERN)]
        if covered == int(frame_count):
            return step
        if covered > int(frame_count):
            break
    raise ValueError(f"IAMCCS H3 context {frame_count}f is not aligned to the VAE temporal grid")


def _phase_aligned_tail(video: torch.Tensor, export_start_frames: int, export_frames: int, context_frames: int) -> tuple[int, list[int], int, int]:
    """Choose a full-grid prior AV window ending at or before the exported tail."""
    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    total_steps = int(video.shape[2])
    run_steps = _steps_for_frames(context_frames)
    export_end = max(0, int(export_start_frames) + int(export_frames))
    best_start = None
    best_end = -1
    for start in range(0, total_steps - run_steps + 1, len(_FRAME_PATTERN)):
        pin_end = _pixel_frames(start) + _pixel_frames(run_steps)
        if pin_end <= export_end and pin_end >= best_end:
            best_start, best_end = start, pin_end
    if best_start is None:
        raise ValueError("IAMCCS H3 context cannot phase-align a prior AV tail to the exported segment")
    return best_start, _pixel_starts(run_steps), best_end, max(0, export_end - best_end)


def _ensure_runtime_patches() -> None:
    """Install guarded layout/payload patches only when IAMCCS continuity is used."""
    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    global _layout_original, _payload_original
    import comfy.ldm.minimax.model as minimax_model
    import comfy.model_base as model_base

    current_layout = minimax_model.PackedLayout.__init__
    if not getattr(current_layout, _LAYOUT_MARKER, False):
        if getattr(current_layout, "_h3_motion_context_layout_patch", False) or getattr(current_layout, "_h3_external_continuity_layout_patch", False):
            raise RuntimeError("IAMCCS H3 continuity found another MiniMax layout owner. Disable the other continuity runtime and restart ComfyUI.")
        if getattr(current_layout, "__module__", "") != minimax_model.__name__:
            raise RuntimeError("IAMCCS H3 continuity found a foreign MiniMax layout override. Restart with one continuity runtime enabled.")
        _layout_original = current_layout

        def layout_patch(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=None, refs=None, frame_count=None):
            stock_keyframes = []
            for keyframe in keyframes or []:
                item = dict(keyframe)
                if _FRAME_MARKER in item:
                    item["resolved_frame_index"] = 0
                stock_keyframes.append(item)
            _layout_original(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=stock_keyframes or None, refs=refs, frame_count=frame_count)
            spans = [(start, stop) for start, stop, kind in self.segments if kind == "cond"]
            if len(spans) != len(keyframes or []):
                raise RuntimeError("IAMCCS H3 continuity layout condition span mismatch")
            origin = float(self.position_ids[self.segments[-1][0], 0])
            for (start, stop), keyframe in zip(spans, keyframes or []):
                if _FRAME_MARKER in keyframe:
                    self.position_ids[start:stop, 0] = origin + float(minimax_model.FRAME_RESCALE) * float(keyframe[_FRAME_MARKER])
            marked_audio = [index for index, ref in enumerate(refs or []) if _AUDIO_MARKER in ref]
            if marked_audio:
                if len(marked_audio) != 1:
                    raise RuntimeError("IAMCCS H3 continuity requires exactly one marked audio reference")
                audio_refs = [ref for ref in refs or [] if ref.get("kind") == "audio" and int(ref.get("ref_audio_t", 0)) > 0]
                marked_ref = (refs or [])[marked_audio[0]]
                audio_index = next((index for index, ref in enumerate(audio_refs) if ref is marked_ref), None)
                audio_spans = [(start, stop) for start, stop, kind in self.segments if kind == "ref_audio"]
                if audio_index is None or audio_index >= len(audio_spans):
                    raise RuntimeError("IAMCCS H3 continuity audio layout span is unavailable")
                audio_start, audio_stop = audio_spans[audio_index]
                audio_steps = int(marked_ref["ref_audio_t"])
                if audio_stop - audio_start != audio_steps * 2:
                    raise RuntimeError("IAMCCS H3 continuity audio layout row count is invalid")
                desired_start = origin + float(minimax_model.FRAME_RESCALE) * float(marked_ref[_AUDIO_MARKER]) - float(audio_steps)
                slot_start = float(self.position_ids[audio_start, 0])
                self.position_ids[audio_start:audio_stop, 0] += desired_start - slot_start

        setattr(layout_patch, _LAYOUT_MARKER, True)
        minimax_model.PackedLayout.__init__ = layout_patch

    model_cls = getattr(model_base, "MiniMaxH3", None)
    current_payload = getattr(model_cls, "extra_conds", None)
    if model_cls is None or current_payload is None:
        raise RuntimeError("IAMCCS H3 continuity cannot locate the MiniMax conditioning payload hook")
    if getattr(current_payload, _PAYLOAD_MARKER, False):
        return
    if getattr(current_payload, "_h3_motion_context_payload_patch", False) or getattr(current_payload, "_h3_external_continuity_payload_patch", False):
        raise RuntimeError("IAMCCS H3 continuity found another MiniMax payload owner. Disable the other continuity runtime and restart ComfyUI.")
    if getattr(current_payload, "__module__", "") != model_base.__name__:
        raise RuntimeError("IAMCCS H3 continuity found a foreign MiniMax payload override. Restart with one continuity runtime enabled.")
    _payload_original = current_payload

    def payload_patch(self, **kwargs):
        result = _payload_original(self, **kwargs)
        keyframes = kwargs.get("minimax_keyframes") or []
        refs = kwargs.get("minimax_refs") or []
        if not any(_FRAME_MARKER in item for item in keyframes) and not any(_AUDIO_MARKER in item for item in refs):
            return result
        condition = result.get("minimax_payload")
        payload = getattr(condition, "cond", None)
        if not isinstance(payload, dict):
            raise RuntimeError("IAMCCS H3 continuity cannot merge temporal condition payload")
        payload["cond_video_latents"] = (
            [item["latent"] for item in keyframes if torch.is_tensor(item.get("latent"))]
            + [item["latent"] for item in refs if torch.is_tensor(item.get("latent"))]
        )
        payload["cond_audio_latents"] = [item["audio_latent"] for item in refs if torch.is_tensor(item.get("audio_latent"))]
        if kwargs.get("minimax_frame_count") is not None:
            payload["frame_count"] = kwargs["minimax_frame_count"]
        return result

    setattr(payload_patch, _PAYLOAD_MARKER, True)
    model_cls.extra_conds = payload_patch


def apply_context(conditioning: Any, previous: dict[str, Any], context_frames: Any, continue_audio: bool, sample_frames: int) -> tuple[Any, str]:
    """Pin the prior exported AV tail into the next H3 conditioning payload."""
    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    context = normalize_frames(context_frames)
    video = previous["video"]
    audio = previous["audio"]
    start, offsets, pin_end, trim_previous_export = _phase_aligned_tail(
        video,
        int(previous.get("export_start_frames", 0)),
        int(previous.get("export_frames", 0)),
        context,
    )
    _ensure_runtime_patches()
    keyframes = [
        {"resolved_frame_index": 0, _FRAME_MARKER: offset, "latent": video[:1, :, start + index:start + index + 1].clone()}
        for index, offset in enumerate(offsets)
    ]
    refs: list[dict[str, Any]] = []
    if continue_audio and int(audio.shape[-1]) > 0:
        audio_steps = max(1, min(int(audio.shape[-1]), int(round(_AUDIO_CONTEXT_FRAMES / _FPS * _AUDIO_HZ))))
        audio_end = max(audio_steps, min(int(audio.shape[-1]), int(round(pin_end / _FPS * _AUDIO_HZ))))
        refs.append({"kind": "audio", "ref_audio_t": audio_steps, "audio_latent": audio[:1, :, :, audio_end - audio_steps:audio_end].clone(), _AUDIO_MARKER: context})
    import node_helpers

    existing = conditioning[0][1].get("minimax_keyframes", []) if conditioning else []
    terminal = []
    for item in existing:
        if int(item.get("resolved_frame_index", -1)) > 0:
            anchor = dict(item)
            anchor[_FRAME_MARKER] = int(sample_frames) - 1
            terminal.append(anchor)
    result = node_helpers.conditioning_set_values(conditioning, {
        "minimax_keyframes": keyframes + terminal,
        "minimax_refs": refs or None,
        "minimax_frame_count": int(sample_frames),
    })
    return result, (
        f"pinned={context}f video=sampled_av steps={len(keyframes)} "
        f"audio={'40t' if refs else 'off'} pin_end={pin_end} "
        f"phase_tail_trim={trim_previous_export}f"
    )


NODE_CLASS_MAPPINGS = {
    "IAMCCS_MiniMaxH3MotionContext": IAMCCS_MiniMaxH3MotionContext,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_MiniMaxH3MotionContext": "MiniMax H3 Motion Context (Native AV Tail)",
}
