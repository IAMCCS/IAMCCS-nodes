# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later

"""Isolated IAMCCS MiniMax H3 Motion Context backend variant.

The proven R21/R22 atomic backend is intentionally not modified here.  This
module adapts CineLinX and the IAMCCS segment queue to the upstream standalone
Motion Context implementation, then delegates sampling to the existing atomic
generator.  A lazy result switch guarantees that only the selected generation
branch executes in workflows that expose both backends.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from typing import Any

import torch
import torch.nn.functional as F

from .iamccs_minimax_h3_atomic_backend import (
    H3_FPS,
    SUPERNODE_LINX_TYPE,
    IAMCCS_MiniMaxH3GenerationBackendV2,
    _audio_slice,
    _load_image,
    _load_timeline_audio,
    _resolve_shotplan,
)


LOG = logging.getLogger("IAMCCS.MiniMaxH3.MotionContextVariant")
CATEGORY = "IAMCCS/MiniMax H3/Motion Context Variant"
PROVIDER = "ComfyUI-H3-Motion-Context-Auto-Chain-addon"


def _provider_node(name: str):
    import nodes as comfy_nodes

    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(name)
    if cls is None:
        raise RuntimeError(
            f"IAMCCS Motion Context R37 requires {PROVIDER} V0.1.2 or newer; "
            f"provider node '{name}' is unavailable. Update/enable the addon and restart ComfyUI."
        )
    return cls


def _chunk(plan: dict[str, Any], index: int) -> dict[str, Any]:
    chunks = plan.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("IAMCCS Motion Context R37 received a shotplan without chunks")
    index = int(index)
    if index < 0 or index >= len(chunks):
        raise IndexError(f"segment_index={index} outside 0..{len(chunks) - 1}")
    return chunks[index]


def _variant_active(cine_linx: Any) -> bool:
    plan = _resolve_shotplan(cine_linx)
    return (
        str(plan.get("backend_variant", "")).lower() == "motion_context_auto_chain_v1"
        or str(plan.get("acceleration", "")).lower() == "comfy_kitchen"
    )


def _motion_context_active(cine_linx: Any) -> bool:
    plan = _resolve_shotplan(cine_linx)
    contract = plan.get("motion_context_auto_chain")
    return bool(isinstance(contract, dict) and contract.get("enabled"))


def _replace_plan(cine_linx: Any, plan: dict[str, Any]) -> dict[str, Any]:
    """Replace every known CineLinX shotplan mirror by name, never position."""
    if isinstance(cine_linx, dict) and cine_linx.get("schema") == "iamccs.minimax_h3.shotplan":
        return copy.deepcopy(plan)
    result = copy.deepcopy(cine_linx)
    resources = result.get("resources") if isinstance(result.get("resources"), dict) else {}
    outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
    payload = resources.get("cine_payload") if isinstance(resources.get("cine_payload"), dict) else {}
    for key in ("iamccs_minimax_h3_shotplan", "minimax_h3_shotplan", "shotplan"):
        if key in resources:
            resources[key] = copy.deepcopy(plan)
    if "shotplan" in outputs:
        outputs["shotplan"] = copy.deepcopy(plan)
    for key in ("minimax_h3_shotplan", "shotplan"):
        if key in payload:
            payload[key] = copy.deepcopy(plan)
    if not any(key in resources for key in ("iamccs_minimax_h3_shotplan", "minimax_h3_shotplan", "shotplan")):
        resources["iamccs_minimax_h3_shotplan"] = copy.deepcopy(plan)
    resources["cine_payload"] = payload
    result["resources"] = resources
    result["outputs"] = outputs
    return result


def _safe_run_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "iamccs_h3")).strip("._")
    return name or "iamccs_h3"


def _chain_config(render_id: str, segment_index: int) -> dict[str, Any]:
    index = int(segment_index)
    run_name = _safe_run_name(render_id)
    return {
        "chain_id": run_name,
        "run_name": run_name,
        "latent_prefix": f"h3_context/iamccs_{run_name}_clip",
        "load_clip_index": max(0, index),
        "save_clip_index": index + 1,
        "reset": index == 0,
    }


def _fit_audio(audio: Any, frames: int, fps: float) -> Any:
    if not isinstance(audio, dict) or not torch.is_tensor(audio.get("waveform")):
        return audio
    sample_rate = max(1, int(audio.get("sample_rate", 32000)))
    wanted = max(1, int(round(int(frames) / float(fps) * sample_rate)))
    waveform = audio["waveform"]
    if int(waveform.shape[-1]) > wanted:
        waveform = waveform[..., :wanted]
    elif int(waveform.shape[-1]) < wanted:
        waveform = F.pad(waveform, (0, wanted - int(waveform.shape[-1])))
    result = dict(audio)
    result["waveform"] = waveform
    result["sample_rate"] = sample_rate
    return result


def _apply_positioned_guides(
    conditioning,
    latent,
    video_vae,
    audio_vae,
    plan: dict[str, Any],
    chunk: dict[str, Any],
    context_offset_frames: int,
):
    """Materialize R37 Shotboard guides before adding the prior AV tail.

    The legacy atomic conditioner deliberately knows nothing about the
    isolated ``longvid_motion_context`` mode.  Its T2VA base conditioning is
    therefore correct, but R37 must apply its own positioned guides.  The
    offset reserves the latent head occupied by the previous chunk on every
    continuation segment.
    """
    if str(plan.get("task_mode", "") or "").lower() != "longvid_motion_context":
        return conditioning, []
    guide_events = chunk.get("guides")
    if not isinstance(guide_events, list):
        return conditioning, []

    from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide

    applied: list[str] = []
    custom_audio_drive = str(plan.get("audio_mode", "") or "").lower() == "h3_custom_audio_drive"
    for guide in guide_events:
        if not isinstance(guide, dict):
            continue
        kind = str(guide.get("kind", "") or "").strip().lower()
        guide_id = str(guide.get("id", "guide") or "guide").strip()
        source_path = str(guide.get("source_path", "") or "").strip()
        frame_index = max(0, int(guide.get("local_frame", 0) or 0)) + max(0, int(context_offset_frames))
        if kind == "image":
            image = _load_image(source_path)
            if image is None:
                raise ValueError(f"R37 LongVid image guide '{guide_id}' has no source image")
            conditioning = MiniMaxH3AddGuide.execute(
                positive=conditioning,
                latent=latent,
                frame_idx=frame_index,
                vae=video_vae,
                image=image,
            )[0]
            applied.append(f"image:{guide_id}@{frame_index}")
        elif kind == "audio":
            if custom_audio_drive:
                # The same rebased waveform is already encoded as a locked AV
                # latent by AtomicAudioDrive. Adding it again would duplicate
                # the conditioning and can degrade lip synchronization.
                applied.append(f"audio-lock-only:{guide_id}@{frame_index}")
                continue
            audio = _load_timeline_audio(source_path)
            if audio is None:
                raise ValueError(f"R37 LongVid audio guide '{guide_id}' has no source audio")
            source_offset_seconds = max(0.0, float(guide.get("source_offset_frames", 0) or 0) / H3_FPS)
            duration_seconds = max(1.0 / H3_FPS, float(guide.get("duration_frames", 1) or 1) / H3_FPS)
            audio = _audio_slice(audio, source_offset_seconds, duration_seconds)
            conditioning = MiniMaxH3AddGuide.execute(
                positive=conditioning,
                latent=latent,
                frame_idx=frame_index,
                audio_vae=audio_vae,
                audio=audio,
            )[0]
            applied.append(f"audio:{guide_id}@{frame_index}")
        else:
            raise ValueError(f"Unsupported R37 LongVid guide kind '{kind}' for '{guide_id}'")
    return conditioning, applied


class IAMCCS_MiniMaxH3MotionContextConditionR37:
    """Load the prior IAMCCS AV latent and apply the upstream V0.1.2 core."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "video_vae": ("VAE",),
                "latent": ("LATENT",),
                "audio_vae": ("VAE",),
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "segment_index": ("INT", {"forceInput": True}),
                "render_id": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "INT", "STRING")
    RETURN_NAMES = ("conditioning", "trim_frames", "report")
    FUNCTION = "apply"
    CATEGORY = CATEGORY

    def apply(self, conditioning, video_vae, latent, audio_vae, cine_linx, segment_index, render_id=""):
        plan = _resolve_shotplan(cine_linx)
        chunk = _chunk(plan, segment_index)
        if not _motion_context_active(cine_linx):
            return conditioning, 0, "R37 Motion Context bypass: selected branch uses ComfyKitchen only"

        contract = plan.get("motion_context_auto_chain") or {}
        context_frames = int(contract.get("context_frames", 22) or 22)
        audio_context_frames = int(contract.get("audio_context_frames", 24) or 0)
        expected_trim = int(chunk.get("motion_context_trim_frames", 0) or 0)
        conditioning, applied_guides = _apply_positioned_guides(
            conditioning,
            latent,
            video_vae,
            audio_vae,
            plan,
            chunk,
            expected_trim,
        )
        guide_report = ",".join(applied_guides) if applied_guides else "none"
        if applied_guides:
            LOG.info(
                "MiniMax H3 R37 positioned guides materialized | chunk=%d/%d | %s",
                int(segment_index) + 1,
                len(plan.get("chunks", [])),
                ", ".join(applied_guides),
            )
        if int(segment_index) <= 0 or expected_trim <= 0:
            return conditioning, 0, (
                "R37 Motion Context clip 1: upstream pass-through (no previous latent) | "
                f"positioned_guides={guide_report}"
            )

        config = _chain_config(render_id, segment_index)
        context_latent = _provider_node("MiniMaxH3AutoChainLoadLatent")().load(
            latent_path=config["latent_prefix"],
            clip_index=config["load_clip_index"],
            reset=False,
            chain_config=config,
        )[0]
        if context_latent is None:
            raise RuntimeError(
                "IAMCCS Motion Context R37 could not load the previous native AV latent. "
                f"render_id={render_id!r}, previous_segment={int(segment_index)}. "
                "Do not purge h3_context while a chain is running."
            )
        conditioned, actual_trim = _provider_node("MiniMaxH3AutoChainMotionContext")().apply(
            conditioning=conditioning,
            vae=video_vae,
            latent=latent,
            context_length=str(context_frames),
            audio_context_length=audio_context_frames,
            context_latent=context_latent,
            audio_vae=audio_vae,
        )
        if int(actual_trim) != expected_trim:
            raise RuntimeError(
                "IAMCCS Motion Context trim contract changed underneath the planner: "
                f"provider={actual_trim}, planned={expected_trim}. Refusing a shifted join."
            )
        report = (
            f"R37 upstream Motion Context applied | segment={int(segment_index) + 1}/{len(plan['chunks'])} | "
            f"video_tail={context_frames}f | audio_tail={audio_context_frames}f | trim={actual_trim}f | "
            f"positioned_guides={guide_report}"
        )
        LOG.info(report)
        return conditioned, int(actual_trim), report


class IAMCCS_MiniMaxH3MotionContextGenerationR37:
    """R37 generator: optional CK attention, legacy sampler delegation, exact trim."""

    @classmethod
    def INPUT_TYPES(cls):
        spec = copy.deepcopy(IAMCCS_MiniMaxH3GenerationBackendV2.INPUT_TYPES())
        spec["required"]["trim_frames"] = ("INT", {"forceInput": True})
        return spec

    RETURN_TYPES = IAMCCS_MiniMaxH3GenerationBackendV2.RETURN_TYPES
    RETURN_NAMES = IAMCCS_MiniMaxH3GenerationBackendV2.RETURN_NAMES
    FUNCTION = "render"
    CATEGORY = CATEGORY

    def render(self, model, positive, latent, video_vae, audio_vae, cine_linx, chunk_index,
               seed, seed_stride, steps, sampler_name, scheduler, denoise,
               shift_video, shift_audio, trim_frames, motion_state=None):
        plan = copy.deepcopy(_resolve_shotplan(cine_linx))
        chunk = _chunk(plan, chunk_index)
        acceleration = str(plan.get("acceleration", "native") or "native").lower()
        attention_report = "authored acceleration delegated"
        if acceleration == "comfy_kitchen":
            from comfy_extras.nodes_model_advanced import ModelAttentionBackend

            model = ModelAttentionBackend().patch(model, "comfy kitchen attention")[0]
            # The old backend remains immutable and therefore receives native;
            # CK has already been applied per-model above.
            plan["acceleration"] = "native"
            attention_report = "ComfyKitchen INT8 per-model attention"
        adapted_linx = _replace_plan(cine_linx, plan)
        result = IAMCCS_MiniMaxH3GenerationBackendV2().render(
            model=model,
            positive=positive,
            latent=latent,
            video_vae=video_vae,
            audio_vae=audio_vae,
            cine_linx=adapted_linx,
            chunk_index=chunk_index,
            seed=seed,
            seed_stride=seed_stride,
            steps=steps,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
            shift_video=shift_video,
            shift_audio=shift_audio,
            motion_state=motion_state,
        )
        frames, audio, _, sampled_latent, fps, base_report = result
        trim_frames = max(0, int(trim_frames))
        if trim_frames:
            frames, audio = _provider_node("MiniMaxH3AutoChainMotionContextTrim")().trim(
                images=frames,
                trim_frames=trim_frames,
                audio=audio,
                fps=float(fps),
                match_tail=True,
            )
        visible_frames = int(chunk.get("visible_frame_count", int(frames.shape[0])) or int(frames.shape[0]))
        if int(frames.shape[0]) < visible_frames:
            raise RuntimeError(
                f"R37 decoded {int(frames.shape[0])} visible frames, planner requires {visible_frames}."
            )
        frames = frames[:visible_frames]
        audio = _fit_audio(audio, visible_frames, float(fps))
        bridge_last = frames[-1:].detach().clone()
        report = (
            f"{base_report} | R37={attention_report} | upstream_motion_context_trim={trim_frames}f | "
            f"delivered={visible_frames}f"
        )
        return frames, audio, bridge_last, sampled_latent, fps, report


class IAMCCS_MiniMaxH3MotionContextStateCommitR37:
    """Persist the sampler AV latent after IAMCCS resolves the numbered render ID."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "audio": ("AUDIO",),
                "sampled_latent": ("LATENT",),
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "segment_index": ("INT", {"forceInput": True}),
                "resolved_render_id": ("STRING", {"forceInput": True}),
                "checkpoint_report": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("frames", "audio", "resolved_render_id", "report")
    FUNCTION = "commit"
    CATEGORY = CATEGORY

    def commit(self, frames, audio, sampled_latent, cine_linx, segment_index,
               resolved_render_id, checkpoint_report):
        if not _motion_context_active(cine_linx):
            return frames, audio, resolved_render_id, checkpoint_report
        config = _chain_config(resolved_render_id, segment_index)
        path = _provider_node("MiniMaxH3AutoChainSaveLatent")().save(
            latent=sampled_latent,
            filename_prefix=config["latent_prefix"],
            clip_index=config["save_clip_index"],
            chain_config=config,
        )[0]
        report = f"{checkpoint_report} | R37 Motion Context AV latent committed: {path}"
        LOG.info("IAMCCS Motion Context state committed | segment=%d | %s", int(segment_index) + 1, path)
        return frames, audio, resolved_render_id, report


class IAMCCS_MiniMaxH3BackendLazySwitchR37:
    """Select one six-output generator branch without evaluating the other."""

    @classmethod
    def INPUT_TYPES(cls):
        lazy_image = ("IMAGE", {"lazy": True})
        lazy_audio = ("AUDIO", {"lazy": True})
        lazy_latent = ("LATENT", {"lazy": True})
        lazy_int = ("INT", {"lazy": True})
        lazy_string = ("STRING", {"lazy": True})
        return {
            "required": {"cine_linx": (SUPERNODE_LINX_TYPE,)},
            "optional": {
                "legacy_frames": lazy_image,
                "legacy_audio": lazy_audio,
                "legacy_bridge": lazy_image,
                "legacy_latent": lazy_latent,
                "legacy_fps": lazy_int,
                "legacy_report": lazy_string,
                "variant_frames": lazy_image,
                "variant_audio": lazy_audio,
                "variant_bridge": lazy_image,
                "variant_latent": lazy_latent,
                "variant_fps": lazy_int,
                "variant_report": lazy_string,
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "IMAGE", "LATENT", "INT", "STRING")
    RETURN_NAMES = ("frames", "audio", "bridge", "sampled_latent", "fps", "report")
    FUNCTION = "select"
    CATEGORY = CATEGORY

    @staticmethod
    def _names(cine_linx):
        prefix = "variant" if _variant_active(cine_linx) else "legacy"
        return [f"{prefix}_{suffix}" for suffix in ("frames", "audio", "bridge", "latent", "fps", "report")]

    def check_lazy_status(self, cine_linx, **kwargs):
        return [name for name in self._names(cine_linx) if kwargs.get(name) is None]

    def select(self, cine_linx, **kwargs):
        names = self._names(cine_linx)
        missing = [name for name in names if kwargs.get(name) is None]
        if missing:
            raise ValueError("R37 lazy backend branch is incomplete: " + ", ".join(missing))
        branch = "motion-context/CK variant" if names[0].startswith("variant") else "legacy atomic"
        LOG.info("IAMCCS MiniMax H3 lazy backend selected: %s", branch)
        return tuple(kwargs[name] for name in names)


NODE_CLASS_MAPPINGS = {
    "IAMCCS_MiniMaxH3MotionContextConditionR37": IAMCCS_MiniMaxH3MotionContextConditionR37,
    "IAMCCS_MiniMaxH3MotionContextGenerationR37": IAMCCS_MiniMaxH3MotionContextGenerationR37,
    "IAMCCS_MiniMaxH3MotionContextStateCommitR37": IAMCCS_MiniMaxH3MotionContextStateCommitR37,
    "IAMCCS_MiniMaxH3BackendLazySwitchR37": IAMCCS_MiniMaxH3BackendLazySwitchR37,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_MiniMaxH3MotionContextConditionR37": "MiniMax H3 · Multi-Shot Upstream Motion Context",
    "IAMCCS_MiniMaxH3MotionContextGenerationR37": "MiniMax H3 · Multi-Shot Motion Context + ComfyKitchen",
    "IAMCCS_MiniMaxH3MotionContextStateCommitR37": "MiniMax H3 · Native AV Context Commit",
    "IAMCCS_MiniMaxH3BackendLazySwitchR37": "MiniMax H3 · Standard / Motion Context Lazy Switch",
}
