"""MiniMax H3 decoded-frame continuity settings for FLF chunk handoffs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

import folder_paths


MOTION_CONTEXT_TYPE = "IAMCCS_H3_MOTION_CONTEXT"
_CONTEXT_FRAMES = (22, 39, 56)
REFERENCE_MOTION_CARRY = "reference_motion_carry"
HYBRID_HARD_ANCHOR = "hybrid_hard_anchor"
# The opt-in native AV-tail path pins a legal multi-frame tail, samples a
# longer H3 latent, then removes that pinned prefix before delivery.  The
# regular Shotboard FL2VA path keeps authored A->B / B->C keyframes.
NATIVE_AV_CONTEXT = "native_av_context"
_FRAME_PATTERN = (1, 4, 4, 4, 4)
_FRAME_MARKER = "iamccs_native_av_tail_index"
_AUDIO_MARKER = "iamccs_native_av_tail_end"
_LAYOUT_MARKER = "_iamccs_native_av_tail_layout_patch"
_PAYLOAD_MARKER = "_iamccs_native_av_tail_payload_patch"
LOG = logging.getLogger("iamccs.minimax_h3.continuity")


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
                "continuity_strategy": ([REFERENCE_MOTION_CARRY, HYBRID_HARD_ANCHOR, NATIVE_AV_CONTEXT], {"default": REFERENCE_MOTION_CARRY}),
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
            "continuity_strategy": (
                continuity_strategy
            ) if continuity_strategy in {REFERENCE_MOTION_CARRY, HYBRID_HARD_ANCHOR, NATIVE_AV_CONTEXT} else REFERENCE_MOTION_CARRY,
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
    if isinstance(config, dict):
        value = str(config.get("continuity_strategy") or "")
        if value in {HYBRID_HARD_ANCHOR, NATIVE_AV_CONTEXT}:
            return value
    return REFERENCE_MOTION_CARRY


def uses_reference_motion_carry(config: Any, segment_index: Any) -> bool:
    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    return is_active(config, segment_index) and continuity_strategy(config) == REFERENCE_MOTION_CARRY


def uses_native_av_context(config: Any, segment_index: Any) -> bool:
    """True only for IAMCCS' opt-in multi-frame AV-tail handoff."""
    return is_active(config, segment_index) and continuity_strategy(config) == NATIVE_AV_CONTEXT


def native_av_context_from_shotplan(shotplan: Any) -> dict[str, Any] | None:
    """Read the opt-in continuity contract carried inside CineLinX.

    Keeping this small configuration in the Shotboard plan means the feature is
    owned by the IAMCCS Shotboard/backend path: a separate control node is no
    longer required for a normal FL2VA workflow.  ``None`` deliberately means
    the historical planned-keyframe path, preserving every existing board.
    """
    if not isinstance(shotplan, dict):
        return None
    raw = shotplan.get("native_av_continuity")
    if not isinstance(raw, dict):
        return None
    if str(raw.get("mode") or "") != NATIVE_AV_CONTEXT:
        return None
    if not bool(raw.get("enabled")):
        return None
    return {
        "schema": "iamccs.minimax_h3.motion_context",
        "schema_version": 7,
        "method": "shotboard_native_av_continuity",
        "enabled": True,
        "continue_audio": bool(raw.get("continue_audio", True)),
        "audio_tail_seconds": max(0.5, float(raw.get("audio_tail_seconds", 4.0))),
        "continuity_strategy": NATIVE_AV_CONTEXT,
        "motion_tail_frames": max(5, int(raw.get("tail_frames", 22))),
        "source": "shotboard_cine_linx",
    }


def _align_h3_frames(value: int) -> int:
    frames = max(5, int(value))
    while frames % 17 != 5:
        frames += 1
    return frames


def prepare_native_av_context(
    config: Any,
    *,
    render_id: str,
    segment_index: int,
    visible_frames: int,
) -> dict[str, Any] | None:
    """Load an IAMCCS checkpoint tail and calculate the legal next sample.

    The cached multi-frame AV tail is encoded and injected by the IAMCCS
    backend itself.  A missing cache is deliberately a hard error: silently
    reverting to an independent FL2VA sample would be a false continuity mode.
    """
    if not uses_native_av_context(config, segment_index):
        return None
    previous = load_reference_media(str(render_id or ""), int(segment_index) - 1)
    if previous is None:
        raise RuntimeError(
            "IAMCCS native AV continuity needs the previous native checkpoint cache. "
            "Start from chunk 1 and keep Native Checkpoint connected to Motion Context."
        )
    context_frames = max(5, int(config.get("motion_tail_frames", 22)))
    export_frames = _align_h3_frames(int(visible_frames))
    sample_frames = _align_h3_frames(export_frames + context_frames)
    if sample_frames > 362:
        raise ValueError(
            f"IAMCCS native AV continuity needs {sample_frames} frames for this chunk; "
            "reduce the chunk duration or its motion-tail window so it stays within 362 frames."
        )
    return {
        "method": NATIVE_AV_CONTEXT,
        "previous": previous,
        "context_frames": context_frames,
        "export_frames": export_frames,
        "sample_frames": sample_frames,
        "continue_audio": bool(config.get("continue_audio", True)),
    }


def _pixel_starts(latent_steps: int) -> list[int]:
    starts: list[int] = []
    cursor = 0
    for index in range(max(0, int(latent_steps))):
        starts.append(cursor)
        cursor += _FRAME_PATTERN[index % len(_FRAME_PATTERN)]
    return starts


def _frames_for_steps(latent_steps: int) -> int:
    return sum(_FRAME_PATTERN[index % len(_FRAME_PATTERN)] for index in range(max(0, int(latent_steps))))


def _streams_from_av_latent(latent: dict[str, Any]) -> list[torch.Tensor]:
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if hasattr(samples, "unbind"):
        streams = list(samples.unbind())
    elif isinstance(samples, (list, tuple)):
        streams = list(samples)
    else:
        raise ValueError("IAMCCS native AV continuity expected a MiniMax H3 AV latent.")
    if len(streams) < 2 or not all(torch.is_tensor(stream) for stream in streams[:2]):
        raise ValueError("IAMCCS native AV continuity requires video and audio latent streams.")
    return streams


def _resize_frames_for_vae(images: torch.Tensor, width: int, height: int) -> torch.Tensor:
    import comfy.utils

    rgb = images[..., :3].to(dtype=torch.float32).movedim(-1, 1)
    scaled = comfy.utils.common_upscale(rgb, int(width), int(height), "lanczos", "disabled")
    return scaled.movedim(1, -1)


def _conditioning_keyframes(conditioning: Any) -> list[dict[str, Any]]:
    try:
        meta = conditioning[0][1] if conditioning else None
        keyframes = meta.get("minimax_keyframes", []) if isinstance(meta, dict) else []
        return [dict(item) for item in keyframes if isinstance(item, dict)]
    except Exception:
        return []


def _encode_audio_tail(audio_vae: Any, previous: dict[str, Any], frames: int) -> tuple[torch.Tensor, int] | None:
    waveform = previous.get("audio_waveform")
    if not torch.is_tensor(waveform) or int(waveform.shape[-1]) < 1:
        return None
    source_rate = max(1, int(previous.get("audio_sample_rate", 32000)))
    target_rate = max(1, int(getattr(audio_vae, "audio_sample_rate", 32000)))
    audio = waveform[:1].to(dtype=torch.float32)
    if source_rate != target_rate:
        try:
            import torchaudio
        except Exception as exc:
            raise RuntimeError(
                f"IAMCCS native AV continuity needs resampling {source_rate}Hz to {target_rate}Hz, but torchaudio is unavailable."
            ) from exc
        audio = torchaudio.functional.resample(audio, source_rate, target_rate)
    # H3 audio is stereo.  Preserve a stereo source; make a mono source dual-mono
    # before encoding so there is no free/generative audio half.
    if int(audio.shape[1]) == 1:
        audio = audio.repeat(1, 2, 1)
    elif int(audio.shape[1]) > 2:
        audio = audio[:, :2, :]
    wanted = max(1, int(round(float(frames) * target_rate / 24.0)))
    if int(audio.shape[-1]) > wanted:
        audio = audio[..., -wanted:]
    elif int(audio.shape[-1]) < wanted:
        audio = torch.nn.functional.pad(audio, (wanted - int(audio.shape[-1]), 0))
    encoded = audio_vae.encode(audio.movedim(1, -1))
    if not torch.is_tensor(encoded) or encoded.ndim != 4:
        raise RuntimeError("IAMCCS native AV continuity received an unexpected H3 audio VAE latent.")
    return encoded, int(encoded.shape[-1])


def _target_origin(layout: Any) -> float:
    target = next(((start, stop) for start, stop, kind in getattr(layout, "segments", []) if kind == "video"), None)
    if target is None:
        raise RuntimeError("IAMCCS native AV continuity could not locate the H3 target video timeline.")
    return float(layout.position_ids[target[0], 0])


def _keyframe_time(minimax_model: Any, text_len: float, latent_t: int, frame_count: int | None, pixel_index: int) -> float:
    index = int(pixel_index)
    if index == 0:
        return float(text_len)
    if frame_count is not None and index == int(frame_count) - 1:
        return float(text_len) + sum(minimax_model._video_t_spans(latent_t)) - float(minimax_model.FRAME_RESCALE)
    return float(text_len) + float(minimax_model.FRAME_RESCALE) * index


def _patch_layout_and_payload() -> None:
    """Install IAMCCS-only, marker-gated MiniMax H3 continuity hooks lazily."""
    import comfy.ldm.minimax.model as minimax_model
    import comfy.model_base as model_base

    current_layout = minimax_model.PackedLayout.__init__
    if not getattr(current_layout, _LAYOUT_MARKER, False):
        module_name = str(getattr(current_layout, "__module__", ""))
        function_name = str(getattr(current_layout, "__name__", ""))
        if (
            getattr(current_layout, "_h3_motion_context_layout_patch", False)
            or getattr(current_layout, "_director_continuity_layout_patch", False)
            or "h3_motion_context" in module_name
            or "MiniMaxH3_Director" in module_name
        ):
            raise RuntimeError(
                "IAMCCS native AV continuity found another H3 temporal-layout owner. Disable that continuity pack and restart ComfyUI."
            )
        def layout_patch(
            self,
            text_len,
            latent_t,
            latent_h,
            latent_w,
            audio_t,
            keyframes=None,
            refs=None,
            _stock_layout=current_layout,
        ):
            source_keyframes = [dict(item) for item in (keyframes or [])]
            stock_keyframes = []
            for item in source_keyframes:
                copy = dict(item)
                if _FRAME_MARKER in copy:
                    copy["resolved_frame_index"] = 0
                stock_keyframes.append(copy)
            # ComfyUI 0.31 derives frame count from latent_t; its stock
            # PackedLayout API no longer accepts a frame_count keyword.
            _stock_layout(
                self, text_len, latent_t, latent_h, latent_w, audio_t,
                keyframes=stock_keyframes or None, refs=refs,
            )
            marked = any(_FRAME_MARKER in item for item in source_keyframes)
            marked_audio = any(_AUDIO_MARKER in item for item in (refs or []))
            if not marked and not marked_audio:
                return
            cond_spans = [(start, stop) for start, stop, kind in self.segments if kind == "cond"]
            if len(cond_spans) != len(source_keyframes):
                raise RuntimeError("IAMCCS native AV continuity layout condition span mismatch.")
            origin = _target_origin(self)
            offset = origin - float(text_len)
            layout_frame_count = _frames_for_steps(int(latent_t))
            for (start, stop), item in zip(cond_spans, source_keyframes):
                index = item.get(_FRAME_MARKER, item.get("resolved_frame_index", 0))
                self.position_ids[start:stop, 0] = _keyframe_time(
                    minimax_model, text_len, latent_t, layout_frame_count, int(index)
                ) + offset
            marked_refs = [item for item in (refs or []) if _AUDIO_MARKER in item]
            if marked_refs:
                if len(marked_refs) != 1 or marked_refs[0].get("kind") != "audio":
                    raise RuntimeError("IAMCCS native AV continuity supports exactly one marked audio tail.")
                audio_steps = int(marked_refs[0].get("ref_audio_t", 0) or 0)
                spans = [(start, stop) for start, stop, kind in self.segments if kind == "ref_audio"]
                if not spans or audio_steps < 1:
                    raise RuntimeError("IAMCCS native AV continuity could not locate its audio-tail rows.")
                start, stop = spans[0]
                if int(stop - start) != audio_steps * 2:
                    raise RuntimeError("IAMCCS native AV continuity audio-tail row count is invalid.")
                desired_end = origin + float(minimax_model.FRAME_RESCALE) * float(marked_refs[0][_AUDIO_MARKER])
                stock_end = float(self.position_ids[start + audio_steps * 2 - 1, 0]) + 1.0
                self.position_ids[start:stop, 0] += desired_end - stock_end

        setattr(layout_patch, _LAYOUT_MARKER, True)
        minimax_model.PackedLayout.__init__ = layout_patch
        LOG.info("IAMCCS native AV continuity installed its marker-gated H3 layout hook")

    model_cls = getattr(model_base, "MiniMaxH3", None)
    current_payload = getattr(model_cls, "extra_conds", None) if model_cls is not None else None
    if model_cls is None or current_payload is None:
        raise RuntimeError("IAMCCS native AV continuity cannot locate the MiniMax H3 payload hook.")
    if getattr(current_payload, _PAYLOAD_MARKER, False):
        return
    module_name = str(getattr(current_payload, "__module__", ""))
    if (
        getattr(current_payload, "_h3_motion_context_payload_patch", False)
        or getattr(current_payload, "_director_continuity_payload_patch", False)
        or "h3_motion_context" in module_name
        or "MiniMaxH3_Director" in module_name
    ):
        raise RuntimeError(
            "IAMCCS native AV continuity found another H3 temporal-payload owner. Disable that continuity pack and restart ComfyUI."
        )
    def payload_patch(self, _stock_payload=current_payload, **kwargs):
        result = _stock_payload(self, **kwargs)
        keyframes = kwargs.get("minimax_keyframes") or []
        refs = kwargs.get("minimax_refs") or []
        if not any(_FRAME_MARKER in item for item in keyframes) and not any(_AUDIO_MARKER in item for item in refs):
            return result
        holder = result.get("minimax_payload") if isinstance(result, dict) else None
        payload = getattr(holder, "cond", None) if holder is not None else None
        if not isinstance(payload, dict):
            raise RuntimeError("IAMCCS native AV continuity could not merge the H3 conditioning payload.")
        payload["cond_video_latents"] = (
            [item["latent"] for item in keyframes if torch.is_tensor(item.get("latent"))]
            + [item["latent"] for item in refs if torch.is_tensor(item.get("latent"))]
        )
        payload["cond_audio_latents"] = [item["audio_latent"] for item in refs if torch.is_tensor(item.get("audio_latent"))]
        return result

    setattr(payload_patch, _PAYLOAD_MARKER, True)
    model_cls.extra_conds = payload_patch
    LOG.info("IAMCCS native AV continuity installed its marker-gated H3 payload hook")


def apply_native_av_context(
    conditioning: Any,
    latent: dict[str, Any],
    *,
    video_vae: Any,
    audio_vae: Any,
    prepared: dict[str, Any],
) -> tuple[Any, int, int, str]:
    """Inject the real previous AV tail through IAMCCS' own H3 runtime path."""
    previous = prepared.get("previous") if isinstance(prepared, dict) else None
    if not isinstance(previous, dict) or not torch.is_tensor(previous.get("images")):
        raise RuntimeError("IAMCCS native AV continuity cache has no decoded previous video frames.")
    streams = _streams_from_av_latent(latent)
    video_latent = streams[0]
    if video_latent.ndim == 4:
        video_latent = video_latent.unsqueeze(0)
    if video_latent.ndim != 5:
        raise RuntimeError("IAMCCS native AV continuity received an invalid H3 video latent shape.")
    target_height = int(video_latent.shape[3]) * 16
    target_width = int(video_latent.shape[4]) * 16
    available = int(previous["images"].shape[0])
    requested = max(5, int(prepared["context_frames"]))
    context_frames = max(frame for frame in _CONTEXT_FRAMES if frame <= min(requested, available))
    tail = _resize_frames_for_vae(previous["images"][-context_frames:], target_width, target_height)
    encoded_video = video_vae.encode(tail)
    if not torch.is_tensor(encoded_video) or encoded_video.ndim != 5:
        raise RuntimeError("IAMCCS native AV continuity received an unexpected H3 video VAE latent.")
    steps = int(encoded_video.shape[2])
    covered = _frames_for_steps(steps)
    if covered != context_frames:
        raise RuntimeError(
            f"IAMCCS native AV continuity VAE grid mismatch: requested {context_frames} frames, encoded {covered}."
        )
    _patch_layout_and_payload()
    keyframes = [
        {
            "resolved_frame_index": 0,
            _FRAME_MARKER: offset,
            "latent": encoded_video[:, :, index:index + 1].clone(),
        }
        for index, offset in enumerate(_pixel_starts(steps))
    ]
    refs: list[dict[str, Any]] = []
    audio_steps = 0
    if bool(prepared.get("continue_audio", True)):
        encoded_audio = _encode_audio_tail(audio_vae, previous, context_frames)
        if encoded_audio is not None:
            audio_latent, audio_steps = encoded_audio
            refs.append({
                "kind": "audio",
                "ref_audio_t": int(audio_steps),
                "audio_latent": audio_latent,
                _AUDIO_MARKER: int(context_frames),
            })
    existing = _conditioning_keyframes(conditioning)
    terminal = [item for item in existing if int(item.get("resolved_frame_index", -1)) > 0]
    if not terminal:
        raise RuntimeError("IAMCCS native AV continuity requires the authored FL2VA final keyframe.")
    import node_helpers
    positive = node_helpers.conditioning_set_values(
        conditioning,
        {
            "minimax_keyframes": keyframes + terminal,
            "minimax_refs": refs or None,
            "minimax_frame_count": int(prepared["sample_frames"]),
        },
    )
    return (
        positive,
        int(context_frames),
        0,
        f"iamccs_native_av_context pinned={context_frames}f video_steps={steps} audio_steps={audio_steps}",
    )


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
