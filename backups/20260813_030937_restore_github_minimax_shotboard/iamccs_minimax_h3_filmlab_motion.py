"""Filmlab MiniMax H3 AV-latent motion continuity for IAMCCS chunks."""

from __future__ import annotations

import logging
import hashlib
from pathlib import Path
from typing import Any

import torch

import folder_paths


LOG = logging.getLogger("IAMCCS.MiniMaxH3.Filmlab")
SOURCE_REVISION = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]

# Reuse the established state cable so Filmlab can traverse the existing
# conditioning -> generation -> checkpoint workflow boundaries.
FILMLAB_CONTEXT_TYPE = "IAMCCS_H3_MOTION_CONTEXT"
FRAME_MARKER = "iamccs_filmlab_context_frame"
AUDIO_MARKER = "iamccs_filmlab_context_audio_end"
FRAME_PATTERN = (1, 4, 4, 4, 4)
CONTEXT_FRAME_CHOICES = (5, 22, 39, 56)
DEFAULT_CONTEXT_FRAMES = 22
DEFAULT_AUDIO_CONTEXT_FRAMES = 24
FPS = 24.0
AUDIO_HZ = 40.0
FRAME_RESCALE = 5.0 / 3.0
_LAYOUT_MARKER = "_iamccs_filmlab_h3_layout_patch"
_PAYLOAD_MARKER = "_iamccs_filmlab_h3_payload_patch"
_layout_original = None
_payload_original = None


class IAMCCS_MiniMaxH3FilmlabMotionContext:
    """Filmlab configuration for AV-latent continuity between FL2VA chunks."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "context_frames": ([str(value) for value in CONTEXT_FRAME_CHOICES], {"default": "22"}),
                "continue_audio": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = (FILMLAB_CONTEXT_TYPE, "STRING")
    RETURN_NAMES = ("filmlab_context", "report")
    FUNCTION = "configure"
    CATEGORY = "IAMCCS/MiniMax H3/Filmlab"

    def configure(self, enabled=True, context_frames="22", continue_audio=True):
        # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        frames = snap_context_frames(context_frames)
        config = {
            "schema": "iamccs.minimax_h3.filmlab_motion_context",
            "schema_version": 1,
            "enabled": bool(enabled),
            "context_frames": frames,
            "audio_context_frames": DEFAULT_AUDIO_CONTEXT_FRAMES,
            "continue_audio": bool(continue_audio),
            "method": "av_latent_full_chunk_continue",
        }
        # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
        LOG.warning("FILMLAB TRACE context | source=%s | rev=%s | enabled=%s | context=%d | audio=%s", __file__, SOURCE_REVISION, config["enabled"], frames, config["continue_audio"])
        return config, f"Filmlab H3 motion context | {'on' if config['enabled'] else 'off'} | {frames}f | audio={'on' if config['continue_audio'] else 'off'}"


def snap_context_frames(value: Any) -> int:
    """Use the MiniMax VAE-native temporal grids used by Filmlab."""
    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    try:
        requested = int(value)
    except (TypeError, ValueError):
        requested = DEFAULT_CONTEXT_FRAMES
    return min(CONTEXT_FRAME_CHOICES, key=lambda candidate: (abs(candidate - requested), -candidate))


def is_continuity_active(config: Any, segment_index: Any) -> bool:
    """Match the Filmlab gate: enabled, non-leading, multi-segment execution."""
    return isinstance(config, dict) and bool(config.get("enabled")) and int(segment_index) > 0


def _align_frames(value: int) -> int:
    frames = max(5, int(value))
    return frames + ((5 - frames) % 17)


def generation_frame_budget(visible_frames: int, context_frames: int) -> tuple[int, int]:
    """Return a hidden-context sample budget for a generated-only continuation."""
    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    visible = _align_frames(visible_frames)
    context = snap_context_frames(context_frames) if context_frames else 0
    if context <= 0:
        return visible, 0
    sample = _align_frames(visible + context)
    if context >= sample:
        raise ValueError(f"Filmlab context {context}f must be shorter than {sample}f sample")
    # Preserve the predecessor tail internally, then remove it after decode so
    # every exported frame belongs to the new continuation.
    return sample, context


def reinforce_fl2va_prompt(prompt: str, *, has_start_frame: bool, has_end_frame: bool) -> str:
    """Match the active Filmlab keyframes without describing a missing start image."""
    body = str(prompt or "").strip()
    if has_start_frame and has_end_frame:
        prefix = "完全保持首尾帧。视频第一帧必须与给定首帧画面一致，最后一帧必须与给定尾帧画面一致；"
        suffix = "首尾帧是硬锁定关键帧，不是软参考，禁止改动首尾画面、主体外观与机位。"
    elif has_end_frame:
        prefix = "完全保持尾帧。视频最后一帧必须与给定尾帧画面一致；"
        suffix = "尾帧是硬锁定关键帧，不是软参考，禁止改动尾帧画面、主体外观与机位。"
    elif has_start_frame:
        prefix = "完全保持首帧。视频第一帧必须与给定首帧画面一致；"
        suffix = "首帧是硬锁定关键帧，不是软参考，禁止改动首帧画面、主体外观与机位。"
    else:
        raise ValueError("Filmlab FL2VA prompt requires at least one real keyframe")
    return f"{prefix}{suffix}中间过程：{body}" if body else f"{prefix}{suffix}"


def continuation_prompt_body(creative_prompt: str, audio_handoff_prompt: str) -> str:
    """Remove the Shotboard-only first-frame briefing before AV context takes over."""
    parts = []
    for part in str(creative_prompt or "").split("\n\n"):
        if "[current chunk boundaries]" in part.lower():
            continue
        if part.strip():
            parts.append(part.strip())
    audio = str(audio_handoff_prompt or "").strip()
    if audio:
        parts.append(audio)
    return "\n\n".join(parts)


def _cache_path(render_id: str, segment_index: int) -> Path | None:
    safe = str(render_id or "").strip()
    if not safe or int(segment_index) < 0:
        return None
    return Path(folder_paths.get_output_directory()) / "minimax_h3_shotboard" / "filmlab_motion" / f"{safe}_seg_{int(segment_index):04d}.pt"


def _streams(sampled: dict[str, Any]) -> list[torch.Tensor]:
    value = sampled.get("samples") if isinstance(sampled, dict) else None
    if hasattr(value, "unbind"):
        return list(value.unbind())
    if isinstance(value, (list, tuple)):
        return list(value)
    raise ValueError("Filmlab continuity requires a MiniMax H3 AV latent")


def save_av_handoff(render_id: str, segment_index: int, sampled: dict[str, Any], images: torch.Tensor, audio: dict | None, export_start_frames: int, export_frames: int) -> bool:
    """Persist the sampled AV latent and exported media needed by the next chunk."""
    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    path = _cache_path(render_id, segment_index)
    if path is None or not torch.is_tensor(images) or images.ndim != 4:
        return False
    try:
        video, audio_latent = _streams(sampled)[:2]
    except (ValueError, IndexError):
        return False
    if not torch.is_tensor(video) or not torch.is_tensor(audio_latent):
        return False
    waveform = audio.get("waveform") if isinstance(audio, dict) else None
    payload = {
        "segment_index": int(segment_index),
        "video": video.detach().to(device="cpu", copy=True).contiguous(),
        "audio": audio_latent.detach().to(device="cpu", copy=True).contiguous(),
        "images": images.detach().to(device="cpu", dtype=torch.float16, copy=True).contiguous(),
        "audio_waveform": waveform.detach().to(device="cpu", copy=True).contiguous() if torch.is_tensor(waveform) else None,
        "audio_sample_rate": int(audio.get("sample_rate", 32000)) if isinstance(audio, dict) else 32000,
        "export_start_frames": int(export_start_frames),
        "export_frames": int(export_frames),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return True


def load_av_handoff(render_id: str, segment_index: int) -> dict[str, Any] | None:
    """Load exactly the prior segment handoff; stale or mismatched entries are rejected."""
    path = _cache_path(render_id, segment_index)
    if path is None or not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return None
    required = ("video", "audio", "images")
    if not isinstance(payload, dict) or int(payload.get("segment_index", -1)) != int(segment_index) or not all(torch.is_tensor(payload.get(key)) for key in required):
        return None
    return payload


def trim_cached_export(render_id: str, segment_index: int, trim_frames: int, *, fps: float = FPS) -> tuple[torch.Tensor, dict | None] | None:
    """Trim and persist a prior export when phase alignment leaves an orphan tail."""
    payload = load_av_handoff(render_id, segment_index)
    trim = max(0, int(trim_frames))
    if payload is None or trim <= 0:
        return None
    images = payload["images"]
    if int(images.shape[0]) <= trim:
        raise ValueError("Filmlab cannot remove the full previous export")
    images = images[:-trim].contiguous()
    audio = None
    waveform = payload.get("audio_waveform")
    sample_rate = int(payload.get("audio_sample_rate") or 32000)
    if torch.is_tensor(waveform):
        wanted = int(round(int(images.shape[0]) / float(fps) * sample_rate))
        waveform = waveform[..., :wanted].contiguous()
        audio = {"waveform": waveform, "sample_rate": sample_rate}
    payload["images"] = images
    payload["audio_waveform"] = waveform
    payload["export_frames"] = int(images.shape[0])
    path = _cache_path(render_id, segment_index)
    if path is None:
        return None
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return images, audio


def _pixel_frames(steps: int) -> int:
    return sum(FRAME_PATTERN[index % len(FRAME_PATTERN)] for index in range(int(steps)))


def _step_offsets(steps: int) -> list[int]:
    offsets, frame = [], 0
    for index in range(int(steps)):
        offsets.append(frame)
        frame += FRAME_PATTERN[index % len(FRAME_PATTERN)]
    return offsets


def _steps_for_frames(frames: int) -> int:
    covered = 0
    for step in range(1, 100000):
        covered += FRAME_PATTERN[(step - 1) % len(FRAME_PATTERN)]
        if covered == int(frames):
            return step
        if covered > int(frames):
            break
    raise ValueError(f"Filmlab context {frames}f is outside the MiniMax latent grid")


def _phase_tail(video: torch.Tensor, context_frames: int, export_end: int) -> tuple[int, int, int]:
    """Find the last cycle-aligned latent window ending within the previous export."""
    steps = _steps_for_frames(context_frames)
    total_steps = int(video.shape[2])
    best_start, best_end = None, -1
    for start in range(0, total_steps - steps + 1, len(FRAME_PATTERN)):
        end = _pixel_frames(start) + _pixel_frames(steps)
        if end <= int(export_end) and end >= best_end:
            best_start, best_end = start, end
    if best_start is None:
        raise ValueError("Filmlab cannot phase-align the prior AV latent to the exported tail")
    return best_start, best_end, max(0, int(export_end) - best_end)


def lock_generated_video_prefix(latent: dict, previous: dict[str, Any], *, context_frames: int) -> tuple[dict, int]:
    """Copy the sampled predecessor tail into the target and mask it from denoising."""
    import comfy.nested_tensor

    target_video, target_audio = _streams(latent)[:2]
    previous_video = previous["video"]
    if target_video.ndim != 5 or previous_video.ndim != 5:
        raise ValueError("Filmlab generated-frame lock requires [B,C,T,H,W] video latents")
    if tuple(target_video.shape[:2]) != tuple(previous_video.shape[:2]) or tuple(target_video.shape[3:]) != tuple(previous_video.shape[3:]):
        raise ValueError(
            "Filmlab generated-frame lock requires matching predecessor and target latent geometry"
        )
    context = snap_context_frames(context_frames)
    steps = _steps_for_frames(context)
    start, _, _ = _phase_tail(
        previous_video,
        context,
        int(previous["export_start_frames"]) + int(previous["export_frames"]),
    )
    if int(target_video.shape[2]) <= steps:
        raise ValueError("Filmlab generated-frame lock prefix exceeds the target latent")
    video = target_video.clone()
    video[:, :, :steps] = previous_video[:1, :, start:start + steps].to(
        device=video.device,
        dtype=video.dtype,
    )
    video_mask = torch.ones(
        (int(video.shape[0]), 1, int(video.shape[2]), 1, 1),
        device=video.device,
        dtype=torch.float32,
    )
    video_mask[:, :, :steps] = 0.0
    audio_mask = torch.ones(
        (int(target_audio.shape[0]), 1, int(target_audio.shape[2]), int(target_audio.shape[3])),
        device=target_audio.device,
        dtype=torch.float32,
    )
    output = dict(latent)
    output["samples"] = comfy.nested_tensor.NestedTensor((video, target_audio.clone()))
    output["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    LOG.info(
        "Filmlab generated-frame prefix locked | context=%df | latent_steps=%d | tail_steps=%d..%d",
        context,
        steps,
        start,
        start + steps - 1,
    )
    LOG.warning(
        "FILMLAB TRACE lock | source=%s | rev=%s | target_video=%s | source_video=%s | "
        "mask_video=%s locked_minmax=%.1f/%.1f | mask_audio=%s minmax=%.1f/%.1f",
        __file__, SOURCE_REVISION, tuple(video.shape), tuple(previous_video.shape), tuple(video_mask.shape),
        float(video_mask[:, :, :steps].min()), float(video_mask[:, :, :steps].max()), tuple(audio_mask.shape),
        float(audio_mask.min()), float(audio_mask.max()),
    )
    return output, steps


def _ensure_runtime_patches() -> None:
    """Install MiniMax layout/payload support only while Filmlab is used."""
    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    global _layout_original, _payload_original
    import comfy.ldm.minimax.model as minimax_model
    import comfy.model_base as model_base

    current_layout = minimax_model.PackedLayout.__init__
    if not getattr(current_layout, _LAYOUT_MARKER, False):
        if (
            getattr(current_layout, "_h3_motion_context_layout_patch", False)
            or getattr(current_layout, "_h3_external_continuity_layout_patch", False)
            or getattr(current_layout, "_h3_director_continuity_layout_patch", False)
        ):
            raise RuntimeError("Filmlab found another MiniMax H3 temporal layout runtime. Restart with only Filmlab continuity active.")
        _layout_original = current_layout

        def layout_patch(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=None, refs=None, frame_count=None):
            stock_keyframes = []
            for keyframe in keyframes or []:
                item = dict(keyframe)
                if FRAME_MARKER in item:
                    item["resolved_frame_index"] = 0
                stock_keyframes.append(item)
            _layout_original(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=stock_keyframes or None, refs=refs, frame_count=frame_count)
            spans = [(start, stop) for start, stop, kind in self.segments if kind == "cond"]
            if len(spans) != len(keyframes or []):
                raise RuntimeError("Filmlab keyframe layout span mismatch")
            target_start, target_stop, target_kind = self.segments[-1]
            if target_kind != "video" or target_stop <= target_start:
                raise RuntimeError("Filmlab expected the target video as the final packed segment")
            origin = float(self.position_ids[target_start, 0])
            for (start, stop), keyframe in zip(spans, keyframes or []):
                if FRAME_MARKER in keyframe:
                    self.position_ids[start:stop, 0] = origin + float(minimax_model.FRAME_RESCALE) * float(keyframe[FRAME_MARKER])
                    continue
                frame_index = int(keyframe.get("resolved_frame_index", -1))
                if frame_count is None or frame_index != int(frame_count) - 1:
                    raise RuntimeError("Filmlab retained an unsupported non-terminal stock keyframe")
                terminal_time = origin + sum(minimax_model._video_t_spans(latent_t)) - float(minimax_model.FRAME_RESCALE)
                self.position_ids[start:stop, 0] = terminal_time
            marked = [index for index, ref in enumerate(refs or []) if AUDIO_MARKER in ref]
            if marked:
                if len(marked) != 1:
                    raise RuntimeError("Filmlab requires exactly one marked audio context reference")
                requested: list[tuple[int, str]] = []
                for index, ref_item in enumerate(refs or []):
                    kind = ref_item.get("kind")
                    audio_steps = int(ref_item.get("ref_audio_t", 0) or 0)
                    if kind == "image":
                        requested.append((index, "ref_img"))
                    elif kind == "audio":
                        if audio_steps > 0:
                            requested.append((index, "ref_audio"))
                    elif kind in {"video", "video_audio"}:
                        if audio_steps > 0:
                            requested.append((index, "ref_audio"))
                        requested.append((index, "ref_img"))
                    else:
                        raise RuntimeError(f"Filmlab found an unknown reference kind {kind!r}")
                available = [(start, stop, kind) for start, stop, kind in self.segments if kind in {"ref_img", "ref_audio"}]
                if len(requested) != len(available):
                    raise RuntimeError("Filmlab reference layout span count mismatch")
                ref_spans: dict[int, dict[str, tuple[int, int]]] = {}
                for (ref_index, expected_kind), (start, stop, actual_kind) in zip(requested, available):
                    if expected_kind != actual_kind:
                        raise RuntimeError(
                            f"Filmlab reference {ref_index} expected {expected_kind}, got {actual_kind}"
                        )
                    ref_spans.setdefault(ref_index, {})[actual_kind] = (start, stop)
                ref = refs[marked[0]]
                audio_span = ref_spans.get(marked[0], {}).get("ref_audio")
                if audio_span is None:
                    raise RuntimeError("Filmlab marked audio context has no packed audio span")
                audio_start, audio_stop = audio_span
                audio_steps = int(ref["ref_audio_t"])
                if audio_stop - audio_start != audio_steps * 2:
                    raise RuntimeError("Filmlab audio layout row count is invalid")
                desired = origin + float(minimax_model.FRAME_RESCALE) * float(ref[AUDIO_MARKER]) - float(audio_steps)
                self.position_ids[audio_start:audio_stop, 0] += desired - float(self.position_ids[audio_start, 0])

        setattr(layout_patch, _LAYOUT_MARKER, True)
        minimax_model.PackedLayout.__init__ = layout_patch

    model_cls = getattr(model_base, "MiniMaxH3", None)
    current_payload = getattr(model_cls, "extra_conds", None)
    if model_cls is None or current_payload is None:
        raise RuntimeError("Filmlab cannot locate the MiniMax H3 payload hook")
    if not getattr(current_payload, _PAYLOAD_MARKER, False):
        if (
            getattr(current_payload, "_h3_motion_context_payload_patch", False)
            or getattr(current_payload, "_h3_external_continuity_payload_patch", False)
            or getattr(current_payload, "_h3_director_continuity_payload_patch", False)
        ):
            raise RuntimeError("Filmlab found another MiniMax H3 temporal payload runtime. Restart with only Filmlab continuity active.")
        _payload_original = current_payload

        def payload_patch(self, **kwargs):
            result = _payload_original(self, **kwargs)
            keyframes = kwargs.get("minimax_keyframes") or []
            refs = kwargs.get("minimax_refs") or []
            if not any(FRAME_MARKER in item for item in keyframes) and not any(AUDIO_MARKER in item for item in refs):
                return result
            payload = getattr(result.get("minimax_payload"), "cond", None)
            if not isinstance(payload, dict):
                raise RuntimeError("Filmlab cannot merge the MiniMax temporal payload")
            payload["cond_video_latents"] = [item["latent"] for item in keyframes if torch.is_tensor(item.get("latent"))]
            payload["cond_video_latents"] += [item["latent"] for item in refs if torch.is_tensor(item.get("latent"))]
            payload["cond_audio_latents"] = [item["audio_latent"] for item in refs if torch.is_tensor(item.get("audio_latent"))]
            if kwargs.get("minimax_frame_count") is not None:
                payload["frame_count"] = kwargs["minimax_frame_count"]
            LOG.info(
                "Filmlab payload pinned | video_latents=%d | audio_latents=%d | frame_count=%s",
                len(payload["cond_video_latents"]),
                len(payload["cond_audio_latents"]),
                payload.get("frame_count"),
            )
            return result

        setattr(payload_patch, _PAYLOAD_MARKER, True)
        model_cls.extra_conds = payload_patch


def apply_motion_context(positive, latent: dict, previous: dict[str, Any], *, context_frames: int, continue_audio: bool, sample_frames: int) -> tuple[Any, int, int]:
    """Make the prior AV tail the sole visual conditioning for a continued chunk."""
    # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    import node_helpers

    _ensure_runtime_patches()
    video = previous["video"]
    audio = previous["audio"]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    context = snap_context_frames(context_frames)
    start, pin_end, previous_trim = _phase_tail(video, context, int(previous["export_start_frames"]) + int(previous["export_frames"]))
    steps = _steps_for_frames(context)
    keyframes = [
        {"resolved_frame_index": 0, FRAME_MARKER: offset, "latent": video[:1, :, start + index:start + index + 1].clone()}
        for index, offset in enumerate(_step_offsets(steps))
    ]
    refs: list[dict[str, Any]] = []
    if continue_audio and int(audio.shape[-1]) > 0:
        audio_steps = min(int(audio.shape[-1]), max(1, int(round(DEFAULT_AUDIO_CONTEXT_FRAMES / FPS * AUDIO_HZ))))
        audio_end = max(audio_steps, min(int(audio.shape[-1]), int(round(pin_end / FPS * AUDIO_HZ))))
        sampled_frames = _pixel_frames(int(video.shape[2]))
        overhang = max(0.0, float(audio.shape[-1]) - FRAME_RESCALE * float(sampled_frames))
        audio_end_frame = float(context) + overhang / FRAME_RESCALE
        audio_end_frame = round(FRAME_RESCALE * audio_end_frame) / FRAME_RESCALE
        refs.append({"kind": "audio", "ref_audio_t": audio_steps, "audio_latent": audio[:1, ..., audio_end - audio_steps:audio_end].clone(), AUDIO_MARKER: audio_end_frame})
    output = node_helpers.conditioning_set_values(positive, {
        "minimax_keyframes": keyframes,
        "minimax_frame_count": int(sample_frames),
    })
    if refs:
        output = node_helpers.conditioning_set_values(
            output,
            {"minimax_refs": refs},
            append=True,
        )
    LOG.info(
        "Filmlab AV pin ready | context=%df | latent_steps=%d | tail_steps=%d..%d | "
        "terminal=none | sample_frames=%d | previous_trim=%df | audio=%s",
        context,
        len(keyframes),
        start,
        start + steps - 1,
        sample_frames,
        previous_trim,
        "on" if refs else "off",
    )
    return output, context, previous_trim


def trim_context_prefix(images: torch.Tensor, audio: dict | None, trim_frames: int, *, fps: float = FPS) -> tuple[torch.Tensor, dict | None]:
    """Trim the pinned decoded prefix and its matching audio duration."""
    trim = max(0, int(trim_frames))
    if trim > 0:
        if int(images.shape[0]) <= trim:
            raise ValueError("Filmlab cannot remove the full decoded segment")
        images = images[trim:]
    if not isinstance(audio, dict) or not torch.is_tensor(audio.get("waveform")):
        return images, audio
    sample_rate = int(audio.get("sample_rate") or 32000)
    start = int(round(trim / float(fps) * sample_rate))
    waveform = audio["waveform"][..., start:]
    wanted = int(round(int(images.shape[0]) / float(fps) * sample_rate))
    return images, {"waveform": waveform[..., :wanted], "sample_rate": sample_rate}


NODE_CLASS_MAPPINGS = {"IAMCCS_MiniMaxH3FilmlabMotionContext": IAMCCS_MiniMaxH3FilmlabMotionContext}
NODE_DISPLAY_NAME_MAPPINGS = {"IAMCCS_MiniMaxH3FilmlabMotionContext": "MiniMax H3 Filmlab Motion Context"}
