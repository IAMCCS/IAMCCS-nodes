import json
import logging
import os
from pathlib import Path

import comfy.nested_tensor
import folder_paths
from safetensors import safe_open
from safetensors.torch import load_file as st_load, save_file as st_save

from .iamccs_h3_continuous_av.latent_math import FPS, pixel_frames
from .iamccs_h3_continuous_av.nodes import H3ContinuousContinueV11, _streams_from_latent


_LOG = logging.getLogger("iamccs.h3.pskf")
PSKF_FORMAT = "iamccs_pskf_v1"
PSKF_ENGINE = "Pianosequenza-True_long"


def _normalize_handover(handover, frame_count, head_context_frames, visible_frame_count,
                        technical_padding_frames):
    if not isinstance(handover, dict) or not handover.get("available"):
        return {"available": False, "frame_count": frame_count}
    clean = dict(handover)
    analyzed_frames = int(clean.get("frame_count", visible_frame_count))
    if analyzed_frames not in {visible_frame_count, frame_count}:
        raise ValueError(
            "PSkf: handover frame count does not match the visible or technical latent "
            f"({analyzed_frames} not in {{{visible_frame_count}, {frame_count}}})"
        )
    if analyzed_frames == visible_frame_count and head_context_frames:
        coordinate_keys = {
            "freeze_start_frame", "lock_start_frame", "handover_end_frame",
            "ideal_handover_end_frame", "legacy_handover_end_frame",
            "phase_aligned_source_end_frame", "phase_aligned_source_start_frame",
            "phase_aligned_target_end_frame", "phase_aware_handover_end_frame",
            "phase_aware_target_end_frame", "no_lock_fallback_target_end_frame",
            "analysis_start_frame",
        }
        for key in coordinate_keys:
            if key not in clean:
                continue
            try:
                value = int(clean[key])
            except (TypeError, ValueError):
                continue
            if value >= 0:
                clean[key] = value + head_context_frames
    visible_start_frame = head_context_frames
    visible_end_frame = head_context_frames + visible_frame_count - 1
    clean.update({
        "available": True,
        "frame_count": frame_count,
        "visible_frame_count": visible_frame_count,
        "visible_start_frame": visible_start_frame,
        "visible_end_frame": visible_end_frame,
        "technical_padding_frames": technical_padding_frames,
        "terminal_target_end_frame": max(0, visible_end_frame),
    })
    return clean


def _make_state(latent, clip_index=1, head_context_frames=0, visible_frame_count=0,
                source_id="", handover=None, source_path=""):
    video, audio = _streams_from_latent(latent)
    frame_count = pixel_frames(video.shape[2])
    head_context_frames = max(0, int(head_context_frames or 0))
    if head_context_frames > frame_count:
        raise ValueError(
            f"PSkf: head context {head_context_frames} exceeds technical latent length {frame_count}"
        )
    visible_frame_count = max(0, int(visible_frame_count or 0))
    if visible_frame_count == 0:
        visible_frame_count = frame_count - head_context_frames
    if head_context_frames + visible_frame_count > frame_count:
        raise ValueError(
            "PSkf: head context + visible frames exceeds the technical latent "
            f"({head_context_frames} + {visible_frame_count} > {frame_count})"
        )
    technical_padding_frames = frame_count - head_context_frames - visible_frame_count
    return {
        "format": PSKF_FORMAT,
        "engine": PSKF_ENGINE,
        "latent": latent,
        "handover": _normalize_handover(
            handover, frame_count, head_context_frames, visible_frame_count,
            technical_padding_frames,
        ),
        "clip_index": int(clip_index),
        "frame_count": frame_count,
        "head_context_frames": head_context_frames,
        "visible_frame_count": visible_frame_count,
        "technical_padding_frames": technical_padding_frames,
        "source_id": str(source_id or ""),
        "source_path": str(source_path or ""),
    }


def _require_state(pskf):
    if not isinstance(pskf, dict) or pskf.get("format") != PSKF_FORMAT:
        raise ValueError("PSkf: expected an IAMCCS Pianosequenza State Keyframe")
    if "latent" not in pskf:
        raise ValueError("PSkf: state has no native H3 AV latent")
    _streams_from_latent(pskf["latent"])
    return pskf


def _state_info(state):
    return (
        f"{PSKF_ENGINE} | clip {state['clip_index']} | {state['frame_count']} technical frames | "
        f"visible {state['visible_frame_count']} | head {state['head_context_frames']} | "
        f"padding {state['technical_padding_frames']}"
    )


def _contained_output_path(value):
    root = Path(folder_paths.get_output_directory()).resolve()
    raw = Path(str(value or "").strip().strip('"').strip("'"))
    candidate = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("PSkf: load path must stay inside the ComfyUI output directory") from exc
    return candidate


def _resolve_pskf_path(value, clip_index):
    candidate = _contained_output_path(value or "iamccs_true_long")
    if candidate.is_file():
        return candidate
    if not candidate.is_dir():
        raise FileNotFoundError(f"PSkf: {candidate} is neither a file nor a folder")
    files = list(candidate.glob("*.pskf.safetensors"))
    if not files:
        raise FileNotFoundError(f"PSkf: no .pskf.safetensors files in {candidate}")
    index = int(clip_index)
    if index > 0:
        suffix = f"_{index:05d}.pskf.safetensors"
        files = [item for item in files if item.name.endswith(suffix)]
        if not files:
            raise FileNotFoundError(f"PSkf: no fixed slot {index} in {candidate}")
    return max(files, key=lambda item: item.stat().st_mtime)


class IAMCCS_H3PSkfPack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Complete accepted H3 video+audio sampler latent."}),
                "clip_index": ("INT", {"default": 1, "min": 0, "max": 99999}),
                "head_context_frames": ("INT", {"default": 0, "min": 0, "max": 3600}),
                "visible_frame_count": ("INT", {"default": 0, "min": 0, "max": 3600,
                    "tooltip": "0 derives the visible interval from the native latent."}),
                "source_id": ("STRING", {"default": "", "tooltip": "Optional project/shot lineage identifier."}),
            },
            "optional": {
                "handover": ("H3_CONTINUOUS_HANDOVER",),
            },
        }

    RETURN_TYPES = ("IAMCCS_H3_PSKF", "STRING")
    RETURN_NAMES = ("pskf", "state_info")
    FUNCTION = "pack"
    CATEGORY = "IAMCCS/MiniMax H3/Pianosequenza-True_long"
    DESCRIPTION = "Build an autonomous Pianosequenza State Keyframe from the complete native H3 video+audio latent."

    def pack(self, latent, clip_index=1, head_context_frames=0, visible_frame_count=0,
             source_id="", handover=None):
        state = _make_state(
            latent, clip_index=clip_index, head_context_frames=head_context_frames,
            visible_frame_count=visible_frame_count, source_id=source_id, handover=handover,
        )
        return (state, _state_info(state))


class IAMCCS_H3PSkfSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pskf": ("IAMCCS_H3_PSKF",),
                "filename_prefix": ("STRING", {"default": "iamccs_true_long/clip"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("pskf_path", "state_info")
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "IAMCCS/MiniMax H3/Pianosequenza-True_long"
    DESCRIPTION = "Save the complete native H3 AV state as a versioned .pskf.safetensors checkpoint."

    def save(self, pskf, filename_prefix="iamccs_true_long/clip"):
        state = _require_state(pskf)
        video, audio = _streams_from_latent(state["latent"])
        video = video.detach().cpu().contiguous()
        audio = audio.detach().cpu().contiguous()
        folder, filename, counter, _, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory()
        )
        os.makedirs(folder, exist_ok=True)
        slot = state["clip_index"] if state["clip_index"] > 0 else int(counter)
        path = os.path.join(folder, f"{filename}_{slot:05d}.pskf.safetensors")
        metadata = {
            "format": PSKF_FORMAT,
            "engine": PSKF_ENGINE,
            "schema_version": "1",
            "fps": str(FPS),
            "clip_index": str(state["clip_index"]),
            "frame_count": str(state["frame_count"]),
            "head_context_frames": str(state["head_context_frames"]),
            "visible_frame_count": str(state["visible_frame_count"]),
            "technical_padding_frames": str(state["technical_padding_frames"]),
            "video_shape": json.dumps(list(video.shape)),
            "audio_shape": json.dumps(list(audio.shape)),
            "source_id": state["source_id"],
            "handover_json": json.dumps(state["handover"], separators=(",", ":"), sort_keys=True),
        }
        st_save({"video": video, "audio": audio}, path, metadata=metadata)
        info = _state_info(state)
        _LOG.info("PSkf: saved %s (%s)", path, info)
        return (path, info)


class IAMCCS_H3PSkfLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pskf_path": ("STRING", {"default": "iamccs_true_long",
                    "tooltip": "File or folder relative to ComfyUI/output."}),
                "clip_index": ("INT", {"default": 1, "min": 0, "max": 99999,
                    "tooltip": "0 loads the newest PSkf file."}),
            }
        }

    RETURN_TYPES = ("IAMCCS_H3_PSKF", "LATENT", "H3_CONTINUOUS_HANDOVER", "STRING", "STRING")
    RETURN_NAMES = ("pskf", "latent", "handover", "resolved_path", "state_info")
    FUNCTION = "load"
    CATEGORY = "IAMCCS/MiniMax H3/Pianosequenza-True_long"
    DESCRIPTION = "Load an IAMCCS PSkf checkpoint without any external timeline node dependency."

    def load(self, pskf_path="iamccs_true_long", clip_index=1):
        path = _resolve_pskf_path(pskf_path, clip_index)
        tensors = st_load(str(path), device="cpu")
        if "video" not in tensors or "audio" not in tensors:
            raise ValueError("PSkf: checkpoint must contain both video and audio tensors")
        video, audio = tensors["video"], tensors["audio"]
        if video.ndim != 5 or audio.ndim != 4:
            raise ValueError(f"PSkf: invalid shapes video={tuple(video.shape)}, audio={tuple(audio.shape)}")
        with safe_open(str(path), framework="pt", device="cpu") as opened:
            metadata = opened.metadata() or {}
        if metadata.get("format") != PSKF_FORMAT:
            raise ValueError(f"PSkf: unsupported checkpoint format {metadata.get('format')!r}")
        try:
            handover = json.loads(metadata.get("handover_json") or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError("PSkf: invalid handover metadata") from exc
        if not isinstance(handover, dict):
            handover = {}
        handover.setdefault("available", False)
        latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
        state = _make_state(
            latent,
            clip_index=int(metadata.get("clip_index", clip_index)),
            head_context_frames=int(metadata.get("head_context_frames", 0)),
            visible_frame_count=int(metadata.get("visible_frame_count", 0)),
            source_id=metadata.get("source_id", ""),
            handover=handover,
            source_path=str(path),
        )
        info = _state_info(state)
        _LOG.info("PSkf: loaded %s (%s)", path, info)
        return (state, latent, handover, str(path), info)


class IAMCCS_H3TrueLongExtend:
    @classmethod
    def INPUT_TYPES(cls):
        base = H3ContinuousContinueV11.INPUT_TYPES()
        required = {}
        for name, spec in base["required"].items():
            required[name] = ("IAMCCS_H3_PSKF", {
                "tooltip": "Previous accepted Pianosequenza State Keyframe."
            }) if name == "previous_latent" else spec
        optional = dict(base.get("optional", {}))
        optional.pop("handover", None)
        return {"required": required, "optional": optional}

    RETURN_TYPES = H3ContinuousContinueV11.RETURN_TYPES
    RETURN_NAMES = H3ContinuousContinueV11.RETURN_NAMES
    FUNCTION = "build"
    CATEGORY = "IAMCCS/MiniMax H3/Pianosequenza-True_long"
    DESCRIPTION = "Autonomous IAMCCS Extend: phase-aligned native H3 video+audio state continuation from PSkf."

    def build(self, clip, vae, previous_latent, prompt, width, height, duration,
              context_frames="22", handover_mode="auto", alignment_mode="phase_aligned_extended",
              manual_landing_tail_frames=34, ref_image_size="match", last_frame=None,
              reference_image=None, authored_visible_frames=None, opening_image=None):
        state = _require_state(previous_latent)
        result = H3ContinuousContinueV11().build(
            clip, vae, state["latent"], prompt, width, height, duration,
            context_frames=context_frames, handover_mode=handover_mode,
            alignment_mode=alignment_mode,
            manual_landing_tail_frames=manual_landing_tail_frames,
            ref_image_size=ref_image_size, handover=state["handover"],
            last_frame=last_frame, reference_image=reference_image,
            authored_visible_frames=authored_visible_frames, opening_image=opening_image,
        )
        if state["handover"].get("boundary_revision") == 2 and authored_visible_frames is not None:
            if result[1]["iamccs_authored_interval"]["replayed_frames"] != 0:
                raise RuntimeError("True Long shared AV boundary mismatch: unexpected regenerated seam frames")
        return (*result[:-1], f"{PSKF_ENGINE} | {result[-1]}")


NODE_CLASS_MAPPINGS = {
    "IAMCCS_H3PSkfPack": IAMCCS_H3PSkfPack,
    "IAMCCS_H3PSkfSave": IAMCCS_H3PSkfSave,
    "IAMCCS_H3PSkfLoad": IAMCCS_H3PSkfLoad,
    "IAMCCS_H3TrueLongExtend": IAMCCS_H3TrueLongExtend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_H3PSkfPack": "Pianosequenza-True_long · Pack PSkf",
    "IAMCCS_H3PSkfSave": "Pianosequenza-True_long · Save PSkf",
    "IAMCCS_H3PSkfLoad": "Pianosequenza-True_long · Load PSkf",
    "IAMCCS_H3TrueLongExtend": "Pianosequenza-True_long · Extend",
}
