# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later

"""IAMCCS-native LongerVid FL2VA Continuous AV and Guided AV Loop.

The internal continuation engine is a GPLv3 source port of Herrgotts H3
Infinite Continuation Suite v1.2.1. See iamccs_h3_continuous_av/ATTRIBUTION.md.
No separately installed provider custom node is required at runtime.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any

import folder_paths
import torch
import comfy.nested_tensor

from .iamccs_minimax_h3_atomic_backend import (
    H3_FPS,
    SUPERNODE_LINX_TYPE,
    IAMCCS_MiniMaxH3GenerationBackendV2,
    _load_image,
    _resolve_shotplan,
)
from .iamccs_minimax_h3_motion_context_variant import _replace_plan
from .iamccs_minimax_h3_shotboard import _encode_images
from .iamccs_minimax_h3_audio_drive import (
    BEHAVIOR_LOCKED,
    _fit_audio_to_duration,
    _lock_audio_stream,
    _resource_audio,
    _slice_audio_for_chunk,
)
from .iamccs_h3_continuous_av.nodes import (
    H3ContinuousAnalyzeHandoverV11,
    H3ContinuousContinueV11,
    H3ContinuousSeamlessJoinV11,
    H3ContinuousStartV11,
    H3ContinuousStitchOutputV11,
)
from .iamccs_h3_continuous_av.runtime_patches import ensure_h3_runtime_patches
from .iamccs_h3_pskf import IAMCCS_H3PSkfPack, IAMCCS_H3PSkfSave, IAMCCS_H3TrueLongExtend
from .iamccs_h3_true_long import terminal_boundary
from .iamccs_h3_continuous_av.nodes import _streams_from_latent
from .iamccs_minimax_h3_extended_av import core_masks_available


LOG = logging.getLogger("IAMCCS.MiniMaxH3.ContinuousAV")
CATEGORY = "IAMCCS/MiniMax H3/FL2VA Continuous AV"
ENGINE = "IAMCCS FL2VA Continuous AV R45.2"

_INTERNAL_NODES = {
    "H3ContinuousStartV11": H3ContinuousStartV11,
    "H3ContinuousContinueV11": H3ContinuousContinueV11,
    "H3ContinuousAnalyzeHandoverV11": H3ContinuousAnalyzeHandoverV11,
    "H3ContinuousStitchOutputV11": H3ContinuousStitchOutputV11,
    "H3ContinuousSeamlessJoinV11": H3ContinuousSeamlessJoinV11,
}


def _provider_node(name: str):
    cls = _INTERNAL_NODES.get(name)
    if cls is None:
        raise RuntimeError(f"{ENGINE} internal class '{name}' is unavailable")
    return cls


def _ensure_provider_runtime(_provider_cls=None):
    """Preflight the internal marker-gated H3 hooks before sampling."""
    try:
        ensure_h3_runtime_patches()
    except RuntimeError as exc:
        raise RuntimeError(
            "FL2VA Continuous AV runtime preflight failed before sampling. "
            "Restart ComfyUI once so the unified IAMCCS H3 continuation hooks are loaded. "
            f"Runtime detail: {exc}"
        ) from exc


def _chunks(plan: dict[str, Any]) -> list[dict[str, Any]]:
    chunks = plan.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("FL2VA Continuous AV received an empty Shotboard interval plan")
    return [item for item in chunks if isinstance(item, dict)]


def _contract(plan: dict[str, Any]) -> dict[str, Any]:
    contract = plan.get("iamccs_continuous_av")
    if not isinstance(contract, dict):
        contract = plan.get("herrgotts_direct_av_chain")
    if not isinstance(contract, dict):
        contract = plan.get("iamccs_guided_av_loop")
    if not isinstance(contract, dict) or not bool(contract.get("enabled")):
        raise ValueError(
            "Select FL2VA CONTINUOUS AV or GUIDED AV LOOP · EXPERIMENTAL in IAMCCS Settings/Shotboard."
        )
    return contract




def _final_visible_handover(handover: dict[str, Any] | None, total_frames: int) -> dict[str, Any] | None:
    """Keep the first landing frame; remove only subsequent detected hold frames."""
    if not isinstance(handover, dict) or not handover.get("available"):
        return handover
    total = max(1, int(total_frames))
    result = copy.deepcopy(handover)
    freeze_detected = bool(result.get("freeze_detected", False))
    freeze_start = int(result.get("freeze_start_frame", -1) or -1)
    if freeze_detected and 0 < freeze_start < total:
        end_frame = freeze_start
        policy = "trim_locked_hold_keep_first_landing"
    else:
        end_frame = total - 1
        policy = "keep_complete_terminal_motion_no_lock"
    result["handover_end_frame"] = int(end_frame)
    result["landing_tail_frames"] = int(total - (end_frame + 1))
    result["final_visible_policy"] = policy
    return result


def _checkpoint_run_dir(unique_id: Any) -> tuple[Path, str]:
    stamp = int(time.time() * 1000)
    node = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(unique_id or "continuous"))
    run = f"continuous_{node}_{stamp}"
    root = Path(folder_paths.get_output_directory()) / "IAMCCS" / "MiniMaxH3" / "ContinuousAV" / run
    root.mkdir(parents=True, exist_ok=True)
    return root, run


def _publish_checkpoint(unique_id: Any, run: str, path: Path, label: str, index: int, total: int) -> None:
    if unique_id is None:
        return
    try:
        from server import PromptServer
        subfolder = path.parent.relative_to(Path(folder_paths.get_output_directory())).as_posix()
        PromptServer.instance.send_sync("iamccs-h3-continuous-checkpoint", {
            "node": str(unique_id),
            "run": str(run),
            "label": str(label),
            "segment": int(index),
            "total": int(total),
            "filename": path.name,
            "subfolder": subfolder,
            "type": "output",
        })
    except Exception as exc:
        LOG.warning("Continuous AV live checkpoint publish failed: %s", exc)


def _reset_checkpoint_run(run: str) -> int:
    run = str(run or "").strip()
    if not run.startswith("continuous_") or Path(run).name != run:
        raise ValueError("Invalid Continuous AV checkpoint run")
    root = (Path(folder_paths.get_output_directory()) / "IAMCCS" / "MiniMaxH3" / "ContinuousAV").resolve()
    target = (root / run).resolve()
    if target.parent != root:
        raise ValueError("Continuous AV checkpoint run must stay inside the output folder")
    if not target.exists():
        return 0
    removed = sum(1 for path in target.rglob("*") if path.is_file())
    shutil.rmtree(target)
    return removed


def _register_reset_route() -> None:
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return
    prompt_server = getattr(PromptServer, "instance", None)
    if prompt_server is None or getattr(prompt_server, "_iamccs_continuous_reset_route", False):
        return

    @prompt_server.routes.post("/api/iamccs/h3/continuous/reset_segments")
    async def reset_segments(request):
        running, pending = prompt_server.prompt_queue.get_current_queue()
        if running or pending:
            return web.json_response(
                {"ok": False, "error": "Wait for the ComfyUI queue to finish before resetting segments."},
                status=409,
            )
        payload = await request.json()
        try:
            removed = _reset_checkpoint_run(payload.get("run", ""))
        except ValueError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, "removed_files": removed})

    prompt_server._iamccs_continuous_reset_route = True


_register_reset_route()


def _sampling(plan: dict[str, Any]) -> dict[str, Any]:
    authored = plan.get("sampling") if isinstance(plan.get("sampling"), dict) else {}
    return {
        "seed": int(authored.get("seed", 42)),
        "seed_stride": int(authored.get("seed_stride", 1)),
        "steps": int(authored.get("steps", 20)),
        "sampler_name": str(authored.get("sampler_name", "res_multistep")),
        "scheduler": str(authored.get("scheduler", "simple")),
        "denoise": float(authored.get("denoise", 1.0)),
        "shift_video": float(authored.get("shift_video", 12.0)),
        "shift_audio": float(authored.get("shift_audio", 3.0)),
    }


def _provider_plan(cine_linx: Any, plan: dict[str, Any]) -> Any:
    """Make the existing sampler see FL2VA without changing the source plan."""
    adapted = copy.deepcopy(plan)
    adapted["task_mode"] = "fl2va"
    adapted["generation_mode"] = "fl2va"
    adapted.pop("masked_loop_guided", None)
    if str(plan.get("task_mode", "")) == "longvid_keyframe_continuous":
        adapted["terminal_endpoint_mode"] = "hard_image"
        adapted["pianosequenza_2stage_enabled"] = False
    for chunk in adapted.get("chunks", []):
        if isinstance(chunk, dict):
            chunk["task_mode"] = "fl2va"
    return _replace_plan(cine_linx, adapted)


class IAMCCS_MiniMaxH3ContinuousAVSegment:
    """Inspect one dynamic Shotboard A→B interval used by Continuous AV."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "INT", "INT", "FLOAT", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "first_frame", "last_frame", "prompt", "width", "height", "duration_seconds",
        "is_first", "total_segments", "report",
    )
    FUNCTION = "inspect"
    CATEGORY = CATEGORY

    def inspect(self, cine_linx, segment_index):
        plan = _resolve_shotplan(cine_linx)
        _contract(plan)
        chunks = _chunks(plan)
        index = int(segment_index)
        if index < 0 or index >= len(chunks):
            raise IndexError(f"Continuous AV segment_index={index} outside 0..{len(chunks) - 1}")
        chunk = chunks[index]
        first_path = str(chunk.get("first_image", "") or "")
        last_path = str(chunk.get("last_image", "") or "")
        first = _load_image(first_path) if first_path else None
        last = _load_image(last_path) if last_path else None
        if index == 0 and not torch.is_tensor(first):
            raise ValueError("Continuous AV interval 1 requires the first chronological Shotboard image")
        if not torch.is_tensor(last):
            raise ValueError(f"Continuous AV interval {index + 1} requires its destination Shotboard image")
        # A valid IMAGE is returned for the inspection socket on continuation
        # intervals, although the provider correctly ignores it and consumes
        # the prior full AV latent as its opening authority.
        first_out = first if torch.is_tensor(first) else last[:1]
        requested_frames = max(5, int(chunk.get("requested_frame_count", chunk.get("frame_count", 5))))
        duration = requested_frames / float(H3_FPS)
        report = (
            f"IAMCCS Continuous AV interval {index + 1}/{len(chunks)} | "
            f"{'native FL2VA start' if index == 0 else 'direct previous AV latent'} -> authored destination | "
            f"requested={requested_frames}f/{duration:.3f}s | {int(plan['width'])}x{int(plan['height'])}"
        )
        return (
            first_out[:1], last[:1], str(chunk.get("prompt", "")), int(plan["width"]),
            int(plan["height"]), float(duration), int(index == 0), len(chunks), report,
        )


class IAMCCS_MiniMaxH3FL2VAContinuousAV:
    """Dynamic N-guide FL2VA chain using IAMCCS' internal GPLv3 engine."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "video_vae": ("VAE",),
                "audio_vae": ("VAE",),
                "cine_linx": (SUPERNODE_LINX_TYPE,),
            },
            "optional": {
                "reference_image": ("IMAGE", {
                    "tooltip": "Optional Qwen-only <Picture 1> identity/style reference."
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (
        "IMAGE", "AUDIO", "IMAGE", "LATENT", "INT", "STRING",
        "IMAGE", "AUDIO", "LATENT", "INT", "INT", "INT",
    )
    RETURN_NAMES = (
        "frames", "audio", "bridge_last_frame", "sampled_latent", "fps", "report",
        "segment_frames", "segment_audio", "segment_latents",
        "segment_index", "segment_total", "segment_head_trim",
    )
    OUTPUT_IS_LIST = (
        False, False, False, False, False, False,
        True, True, True, True, True, True,
    )
    FUNCTION = "render"
    CATEGORY = CATEGORY

    def render(self, model, clip, video_vae, audio_vae, cine_linx, reference_image=None, unique_id=None):
        plan = _resolve_shotplan(cine_linx)
        contract = _contract(plan)
        chunks = _chunks(plan)
        audio_mode = str(plan.get("audio_mode", "h3_native_generated") or "h3_native_generated").strip().lower()
        if audio_mode not in {"h3_native_generated", "external_audio_post", "h3_custom_audio_drive"}:
            raise ValueError(
                "FL2VA Continuous AV supports H3 Generated Audio, Custom Audio Drive or Audio in Post. "
                f"The active audio mode is '{audio_mode}'."
            )
        custom_audio_master = None
        custom_audio_source = "none"
        if audio_mode == "h3_custom_audio_drive":
            custom_audio_master, custom_audio_source = _resource_audio(cine_linx, BEHAVIOR_LOCKED)
            if custom_audio_master is None:
                raise ValueError(
                    "LONG KEYFRAME CONTINUOUS + Custom Audio requires the AudioBoard/CineInfo custom audio resource. "
                    "The exact source audio is locked into every technical FL2VA interval before sampling."
                )

        context_frames = str(contract.get("context_frames", "22"))
        if context_frames not in {"5", "22", "39"}:
            raise ValueError("Continuous AV context must be one of 5, 22 or 39 frames")
        LOG.info(
            "IAMCCS FL2VA Continuous AV contract | context=%sf | alignment=%s | video_crossfade=%df | authored_keyframes=%s",
            context_frames,
            str(contract.get("alignment_mode", "phase_aligned_extended")),
            int(contract.get("video_crossfade_frames", 0) or 0),
            "preserved" if bool(contract.get("preserve_authored_intermediate_keyframes", True)) else "adaptive",
        )
        width, height = int(plan["width"]), int(plan["height"])
        authored = _sampling(plan)
        provider_linx = _provider_plan(cine_linx, plan)
        Start = _provider_node("H3ContinuousStartV11")
        Continue = _provider_node("H3ContinuousContinueV11")
        Analyze = _provider_node("H3ContinuousAnalyzeHandoverV11")
        StitchOutput = _provider_node("H3ContinuousStitchOutputV11")
        Join = _provider_node("H3ContinuousSeamlessJoinV11")
        _ensure_provider_runtime(Continue)

        true_long = plan.get("longvid_engine") == "pianosequenza_true_long"
        previous_pskf = None
        previous_latent = None
        previous_handover = None
        previous_full_images = None
        combined_images = None
        combined_audio = None
        segment_lengths: list[int] = []
        segment_audio_outputs: list[dict[str, Any]] = []
        segment_latent_outputs: list[dict[str, Any]] = []
        segment_head_trims: list[int] = []
        reports: list[str] = []
        # The public LONG KEYFRAME CONTINUOUS mode must *reach* authored guides
        # without showing H3's terminal still-frame lock.  Older serialized
        # continuous aliases keep their historical behavior for compatibility.
        preserve_authored_keyframes = bool(
            contract.get("preserve_authored_intermediate_keyframes", True)
        )
        public_motion_safe = str(plan.get("task_mode", "")).strip().lower() == "longvid_keyframe_continuous"
        if public_motion_safe and not core_masks_available():
            raise RuntimeError("LONG KEYFRAME CONTINUOUS requires native H3 per-stream denoise masks, as used by Extended AV.")
        segment_checkpoints = bool(contract.get("segment_checkpoints", public_motion_safe))
        checkpoint_root = checkpoint_run = None
        if segment_checkpoints:
            checkpoint_root, checkpoint_run = _checkpoint_run_dir(unique_id)
        if public_motion_safe:
            LOG.info(
                "IAMCCS Continuous AV authored-keyframe motion-safe policy | "
                "guide=destination | visible locked tail=trimmed | latent handover=analyzer cutoff | checkpoints=%s",
                "on" if segment_checkpoints else "off",
            )
        elif preserve_authored_keyframes:
            LOG.info(
                "IAMCCS Continuous AV legacy authored-keyframe compatibility | intermediate tails=preserved | "
                "continuation landing=final authored frame | analyzer=diagnostics only"
            )

        for index, chunk in enumerate(chunks):
            first_path = str(chunk.get("first_image", "") or "")
            last_path = str(chunk.get("last_image", "") or "")
            first = _load_image(first_path) if first_path else None
            last = _load_image(last_path) if last_path else None
            if index == 0 and not torch.is_tensor(first):
                raise ValueError("Continuous AV clip 1 is missing the opening Shotboard guide")
            if not torch.is_tensor(last):
                raise ValueError(f"Continuous AV clip {index + 1} is missing its final Shotboard guide")
            requested_frames = max(5, int(chunk.get("requested_frame_count", chunk.get("frame_count", 5))))
            duration = requested_frames / float(H3_FPS)
            prompt = str(chunk.get("prompt", "") or "")
            if public_motion_safe:
                # Provider resolves the endpoint clock after selecting history.
                # Do not reuse the planner's old technical-duration alignment.
                prompt = str(chunk.get("creative_prompt", prompt))
            authored_args = {"authored_visible_frames": requested_frames} if public_motion_safe else {}

            if index == 0:
                positive, latent = Start().build(
                    clip=clip, vae=video_vae, prompt=prompt, width=width, height=height,
                    duration=duration, first_frame=first[:1], last_frame=last[:1],
                    ref_image_size=str(plan.get("ref_image_size", "match")),
                    reference_image=reference_image,
                    **authored_args,
                )
                actual_head = 0
                handover_info = "native FL2VA start"
            else:
                continuation_handover_mode = (
                    "auto" if public_motion_safe
                    else ("manual" if preserve_authored_keyframes else str(contract.get("handover_mode", "auto")))
                )
                continuation_landing_tail = (
                    0 if preserve_authored_keyframes else int(contract.get("manual_landing_tail_frames", 34))
                )
                positive, latent, actual_head, _ignored_tail, handover_info = (IAMCCS_H3TrueLongExtend() if true_long else Continue()).build(
                    clip=clip, vae=video_vae, previous_latent=previous_pskf if true_long else previous_latent,
                    prompt=prompt, width=width, height=height, duration=duration,
                    context_frames=context_frames,
                    alignment_mode=str(contract.get("alignment_mode", "phase_aligned_extended")),
                    handover_mode=continuation_handover_mode,
                    manual_landing_tail_frames=continuation_landing_tail,
                    ref_image_size=str(plan.get("ref_image_size", "match")),
                    **({} if true_long else {"handover": previous_handover if (public_motion_safe or not preserve_authored_keyframes) else None}),
                    last_frame=last[:1], reference_image=reference_image,
                    **authored_args,
                    **({"opening_image": combined_images[-1:]} if public_motion_safe else {}),
                )

            timing = latent.get("iamccs_authored_interval", {})
            technical_duration = int(timing.get("technical_frames", round(duration * H3_FPS))) / float(H3_FPS)

            custom_drive_report = ""
            custom_chunk_audio = None
            if custom_audio_master is not None:
                # The continued technical clip contains a hidden phase-aligned
                # head. Feed that head the exact earlier portion of the same
                # master audio so mouth/performance timing is continuous across
                # the latent handover instead of restarting at the visible cut.
                visible_start_seconds = float(
                    chunk.get("timeline_start_seconds", float(chunk.get("timeline_start_frame", 0) or 0) / float(H3_FPS))
                    or 0.0
                )
                drive_start_seconds = max(0.0, visible_start_seconds - (int(actual_head) / float(H3_FPS)))
                custom_chunk_audio, slice_report = _slice_audio_for_chunk(
                    custom_audio_master, drive_start_seconds, technical_duration, pad_to_duration=True
                )
                latent, lock_report = _lock_audio_stream(
                    latent, custom_chunk_audio, audio_vae, preserve_prefix_frames=0
                )
                custom_drive_report = (
                    f"custom_audio={custom_audio_source} start={drive_start_seconds:.3f}s "
                    f"head={int(actual_head)}f samples={slice_report.get('output_samples')} "
                    f"lock={lock_report.get('fit_rule', 'exact')}"
                )
                LOG.info(
                    "IAMCCS Continuous AV custom audio drive | clip=%d/%d | start=%.3fs | hidden_head=%df",
                    index + 1, len(chunks), drive_start_seconds, int(actual_head),
                )

            full_images, full_audio, _bridge, sampled_latent, fps, sample_report = (
                IAMCCS_MiniMaxH3GenerationBackendV2().render(
                    model=model, positive=positive, latent=latent, video_vae=video_vae,
                    audio_vae=audio_vae, cine_linx=provider_linx, chunk_index=index,
                    seed=authored["seed"], seed_stride=authored["seed_stride"],
                    steps=authored["steps"], sampler_name=authored["sampler_name"],
                    scheduler=authored["scheduler"], denoise=authored["denoise"],
                    shift_video=authored["shift_video"], shift_audio=authored["shift_audio"],
                )
            )
            if custom_chunk_audio is not None:
                # Delivery audio for the technical clip is the pristine source,
                # not the AudioVAE decode used to drive generation. This keeps
                # the subsequent seam assembly sample-accurate.
                full_audio, _custom_fit = _fit_audio_to_duration(
                    custom_chunk_audio, int(full_images.shape[0]) / float(max(1, int(fps)))
                )

            analysis_frames = int(timing.get("endpoint_frame", int(full_images.shape[0]) - 1)) + 1
            analyzed = Analyze().analyze(
                images=full_images[:analysis_frames],
                preset=str(contract.get("handover_preset", "Balanced")),
                context_frames=context_frames,
            )
            handover = analyzed[0]
            handover_status = str(analyzed[1])
            is_final = index == len(chunks) - 1
            visible_handover = (
                _final_visible_handover(handover, analysis_frames)
                if public_motion_safe else handover
            )
            if true_long:
                # The authored endpoint is authoritative. H3 renders a hidden
                # 34-frame suffix after it, matching Extended/OBVPM-style
                # continuation geometry, so endpoint settling happens outside
                # delivery. Never shorten an authored interval because a
                # detector found the destination early and never inherit the
                # hidden runway into the next take.
                visible_end = int(timing["endpoint_frame"])
                total_frames = int(full_images.shape[0])
                if visible_end < int(actual_head) or visible_end >= total_frames:
                    raise RuntimeError(
                        f"True Long authored boundary {visible_end} is outside the rendered interval "
                        f"head={int(actual_head)} total={total_frames}."
                    )
                visible_handover = dict(
                    handover,
                    available=True,
                    frame_count=total_frames,
                    handover_end_frame=visible_end,
                    visible_end_frame=visible_end,
                    landing_tail_frames=total_frames - visible_end - 1,
                    phase_aligned_target_end_frame=visible_end,
                    phase_aware_target_end_frame=visible_end,
                    detector_mode="iamccs_authored_editorial_boundary",
                    no_lock_fallback_applied=False,
                    boundary_policy="hidden_runway_trimmed_never_inherited",
                    boundary_revision=4,
                    alignment_trim_frames=0,
                    short_hold_trim_frames=0,
                    boundary_runway_frames=int(timing.get("boundary_runway_frames", 0)),
                    grid_padding_frames=int(timing.get("grid_padding_frames", 0)),
                )
                LOG.info(
                    "True Long authored boundary | clip=%d | end=%d | visible=%df | hidden runway=%df | grid padding=%df",
                    index + 1, visible_end, requested_frames,
                    visible_handover["boundary_runway_frames"], visible_handover["grid_padding_frames"],
                )
            if public_motion_safe:
                visible_end = int(visible_handover["handover_end_frame"])
                total_frames = int(full_images.shape[0])
                visible_handover["frame_count"] = total_frames
                visible_handover["landing_tail_frames"] = total_frames - visible_end - 1
                # Reuse history ending at the visible landing, snapping only the
                # latent boundary. The child hides any quantization replay.
                handover = dict(visible_handover, frame_count=total_frames, visible_end_frame=visible_end,
                                detector_mode="iamccs_visible_terminal", no_lock_fallback_applied=False,
                                phase_aligned_target_end_frame=visible_end,
                                phase_aware_target_end_frame=visible_end)

            # Prepare the exact visible interval used by delivery.  In the
            # public mode no-lock fallback is never a visible tail trim.
            if public_motion_safe:
                segment_images, segment_audio, segment_trim_info, _seg_head, _seg_tail = StitchOutput().prepare(
                    images=full_images, output_mode="Stitch Ready", handover=visible_handover,
                    audio=full_audio, head_context_frames=int(actual_head),
                )
            else:
                legacy_segment_mode = (
                    "Final Clip" if is_final
                    else ("Full" if preserve_authored_keyframes else "Stitch Ready")
                )
                segment_images, segment_audio, segment_trim_info, _seg_head, _seg_tail = StitchOutput().prepare(
                    images=full_images, output_mode=legacy_segment_mode, handover=handover,
                    audio=full_audio, head_context_frames=int(actual_head),
                )

            if segment_checkpoints and checkpoint_root is not None and checkpoint_run is not None:
                checkpoint_path = checkpoint_root / f"segment_{index + 1:04d}_of_{len(chunks):04d}.mp4"
                _encode_images(segment_images, segment_audio, int(fps), checkpoint_path)
                diagnostics = {
                    "segment": index + 1, "total_segments": len(chunks),
                    "expected_visible_frames": requested_frames,
                    "actual_visible_frames": int(segment_images.shape[0]),
                    "technical_frames": int(full_images.shape[0]),
                    "hidden_head_frames": int(actual_head),
                    "requested_context_frames": int(context_frames),
                    "effective_context_frames": timing.get("context_frames", int(actual_head)),
                    "replayed_frames": timing.get("replayed_frames", 0),
                    "grid_padding_frames": timing.get("grid_padding_frames", 0),
                    "tail_removed_frames": int(_seg_tail),
                    "freeze_detected": bool(handover.get("freeze_detected")),
                    "guide_arrival_verified": False,
                    "duration_shortfall_frames": max(0, requested_frames - int(segment_images.shape[0])),
                }
                checkpoint_path.with_suffix(".json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
                if diagnostics["actual_visible_frames"] != requested_frames:
                    LOG.warning("Continuous AV interval %d: expected %df, delivered %df after terminal/boundary trim; review checkpoint %s",
                                index + 1, requested_frames, int(segment_images.shape[0]), checkpoint_path)
                _publish_checkpoint(
                    unique_id, checkpoint_run, checkpoint_path,
                    f"SEGMENT {index + 1}/{len(chunks)} · {int(segment_images.shape[0])}/{requested_frames}f"
                    + (" · CHECK DURATION" if int(segment_images.shape[0]) != requested_frames else ""),
                    index + 1, len(chunks),
                )
                LOG.info(
                    "IAMCCS Continuous AV checkpoint saved | clip=%d/%d | frames=%d | %s",
                    index + 1, len(chunks), int(segment_images.shape[0]), checkpoint_path,
                )

            if public_motion_safe and custom_audio_master is not None and int(segment_images.shape[0]) != requested_frames:
                raise RuntimeError(
                    f"Continuous AV interval {index + 1} delivered {int(segment_images.shape[0])}/{requested_frames} frames "
                    "after hold cleanup. Review the segment checkpoint and regenerate this take: "
                    "joining it would desynchronize the authored Custom Audio master."
                )

            segment_lengths.append(int(segment_images.shape[0]))
            segment_audio_outputs.append({
                **dict(segment_audio),
                "waveform": segment_audio["waveform"].detach().cpu().contiguous(),
            })
            segment_video_latent, segment_audio_latent = _streams_from_latent(sampled_latent)
            segment_latent_payload = {
                "samples": comfy.nested_tensor.NestedTensor((
                    segment_video_latent.detach().cpu().contiguous(),
                    segment_audio_latent.detach().cpu().contiguous(),
                ))
            }
            # FAST LATENT 2-PASS consumes the per-segment latent list rather
            # than the final combined latent. Preserve the exact authored
            # True Long clock on each CPU checkpoint: the technical length,
            # hidden head, editorial endpoint and hidden runway are the only
            # authoritative geometry for rebuilding Stage-2 conditioning.
            authored_interval = sampled_latent.get("iamccs_authored_interval")
            if isinstance(authored_interval, dict):
                segment_latent_payload["iamccs_authored_interval"] = copy.deepcopy(authored_interval)
            segment_latent_outputs.append(segment_latent_payload)
            segment_head_trims.append(int(actual_head))

            if index == 0:
                combined_images, combined_audio = segment_images, segment_audio
                trim_info = segment_trim_info
            else:
                next_output_mode = (
                    "Stitch Ready" if public_motion_safe
                    else ("Final Clip" if is_final or preserve_authored_keyframes else "Stitch Ready")
                )
                combined_images, combined_audio, trim_info = Join().join(
                    previous_images=combined_images, next_images=full_images,
                    next_output_mode=next_output_mode,
                    next_head_context_frames=int(actual_head),
                    # OBVPM's invisible extend seam hands over across the full
                    # time-registered pinned head: those child frames represent
                    # the same moments as the parent's tail. Replacing that
                    # tail with a cosine handover changes no duration and makes
                    # the first new body frame continue from the child's own
                    # rendering. Keep the legacy hard join for non-PSkf modes.
                    video_crossfade_frames=(
                        int(actual_head)
                        if true_long
                        else (0 if public_motion_safe else int(contract.get("video_crossfade_frames", 0)))
                    ),
                    audio_crossfade_ms=(
                        0.0 if custom_audio_master is not None
                        # True Long keeps exact timeline duration across its
                        # registered video handover, while native generated
                        # audio benefits from a slightly
                        # softer equal-gain boundary.  This remains short
                        # enough to avoid doubled words/transients and never
                        # touches an authored Custom Audio master.
                        else (35.0 if true_long else float(contract.get("audio_crossfade_ms", 15.0)))
                    ),
                    luminance_match=False,
                    luminance_fade_frames=16,
                    max_luminance_correction_percent=10.0,
                    max_safe_tail_bridge_frames=(
                        0
                        if public_motion_safe
                        else (0 if preserve_authored_keyframes else int(contract.get("max_safe_tail_bridge_frames", 2)))
                    ),
                    previous_audio=combined_audio, next_audio=full_audio,
                    next_handover=visible_handover if public_motion_safe else (None if is_final or preserve_authored_keyframes else handover),
                    previous_full_images=previous_full_images,
                    previous_handover=previous_handover if public_motion_safe else (None if preserve_authored_keyframes else previous_handover),
                )

            reports.append(
                f"clip {index + 1}/{len(chunks)} | authored={requested_frames}f visible={int(segment_images.shape[0])}f | head={int(actual_head)}f | "
                f"{handover_info} | {handover_status} | visible={segment_trim_info} | join={trim_info} | {sample_report}"
                + (f" | {custom_drive_report}" if custom_drive_report else "")
            )
            if true_long:
                previous_pskf, state_info = IAMCCS_H3PSkfPack().pack(
                    sampled_latent, clip_index=index + 1, head_context_frames=int(actual_head),
                    visible_frame_count=int(segment_images.shape[0]), source_id=str(checkpoint_run),
                    handover=handover,
                )
                prefix = str((checkpoint_root / "state").relative_to(Path(folder_paths.get_output_directory())))
                state_path, _ = IAMCCS_H3PSkfSave().save(previous_pskf, prefix)
                reports.append(f"{state_info} | checkpoint={state_path}")
            previous_latent = sampled_latent
            previous_handover = handover
            previous_full_images = None if public_motion_safe else full_images.detach().to("cpu")
            LOG.info(
                "IAMCCS FL2VA Continuous AV | clip=%d/%d | head=%df | output=%df",
                index + 1, len(chunks), int(actual_head), int(combined_images.shape[0]),
            )

        if not torch.is_tensor(combined_images) or int(combined_images.shape[0]) < 1:
            raise RuntimeError("FL2VA Continuous AV produced no final frames")
        if custom_audio_master is not None:
            # The global AudioBoard file is final soundtrack authority. The
            # per-interval locks above drive lip/performance generation; the
            # final mux receives one pristine, seam-free master slice.
            combined_audio, final_audio_fit = _fit_audio_to_duration(
                custom_audio_master, int(combined_images.shape[0]) / float(H3_FPS)
            )
            reports.append(
                "custom audio final master | "
                f"source={custom_audio_source} | action={final_audio_fit.get('action')} | "
                f"samples={final_audio_fit.get('after_samples')}"
            )

        bridge = combined_images[-1:].detach().clone()
        segment_frame_outputs = []
        segment_cursor = 0
        for segment_length in segment_lengths:
            segment_frame_outputs.append(
                combined_images[segment_cursor:segment_cursor + segment_length]
            )
            segment_cursor += segment_length
        if segment_cursor != int(combined_images.shape[0]):
            raise RuntimeError(
                "True Long segment checkpoints do not cover the joined master: "
                f"segments={segment_cursor}f master={int(combined_images.shape[0])}f"
            )
        report = (
            f"IAMCCS FL2VA Continuous AV complete | engine={ENGINE} | "
            f"guides={len(chunks) + 1} intervals={len(chunks)} frames={int(combined_images.shape[0])} | "
            f"latent_handover=full_native_AV | audio_authority={'custom_locked_master' if custom_audio_master is not None else audio_mode} | freeze_analysis=automatic | "
            f"intermediate_keyframes={'preserved' if preserve_authored_keyframes else 'freeze-trimmed'} | "
            f"phase={contract.get('alignment_mode')} | "
            f"safe_tail_bridge<={0 if preserve_authored_keyframes else contract.get('max_safe_tail_bridge_frames')}f\n"
            + "\n".join(reports)
        )
        segment_total = len(segment_frame_outputs)
        return (
            combined_images, combined_audio, bridge, previous_latent, int(fps), report,
            segment_frame_outputs,
            segment_audio_outputs,
            segment_latent_outputs,
            list(range(segment_total)),
            [segment_total] * segment_total,
            segment_head_trims,
        )


NODE_CLASS_MAPPINGS = {
    "IAMCCS_MiniMaxH3ContinuousAVSegment": IAMCCS_MiniMaxH3ContinuousAVSegment,
    "IAMCCS_MiniMaxH3FL2VAContinuousAV": IAMCCS_MiniMaxH3FL2VAContinuousAV,
    # Backward-compatible workflow aliases; no external provider is used.
    "IAMCCS_MiniMaxH3HerrgottsSegment": IAMCCS_MiniMaxH3ContinuousAVSegment,
    "IAMCCS_MiniMaxH3HerrgottsDirectAVChain": IAMCCS_MiniMaxH3FL2VAContinuousAV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_MiniMaxH3ContinuousAVSegment": "FL2VA CONTINUOUS AV · Shotboard Interval",
    "IAMCCS_MiniMaxH3FL2VAContinuousAV": "FL2VA CONTINUOUS AV · Phase-Aligned Latent Handover",
    "IAMCCS_MiniMaxH3HerrgottsSegment": "FL2VA CONTINUOUS AV · Shotboard Interval (Legacy ID)",
    "IAMCCS_MiniMaxH3HerrgottsDirectAVChain": "FL2VA CONTINUOUS AV · Phase-Aligned Latent Handover",
}
