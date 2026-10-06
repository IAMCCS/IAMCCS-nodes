# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fast direct-latent MiniMax H3 two-pass delivery.

This is an isolated delivery variant.  It does not replace the native R39/R40
generation path or the conservative R38B pixel/windowed route.
"""

from __future__ import annotations

import copy
import gc
import importlib
import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import folder_paths
import torch
import torch.nn.functional as F

from .iamccs_minimax_h3_atomic_backend import (
    SUPERNODE_LINX_TYPE,
    IAMCCS_MiniMaxH3AtomicConditioningBackend,
    _resolve_shotplan,
)
from .iamccs_minimax_h3_latent_upres_variant import (
    _delivery_size,
    _grid_up,
    _replace_plan,
    _safe_render_id,
    _upres_settings,
    IAMCCS_MiniMaxH3LatentUpresSamplingR38,
)
from .iamccs_minimax_h3_pixel_refine_variant import (
    _finish_segment,
    _next_prompt,
    _rtx_finish_segment,
)


LOG = logging.getLogger("IAMCCS.MiniMaxH3.FastLatent2Pass")
CATEGORY = "IAMCCS/MiniMax H3/Quality Latent 2-Pass"
FAST_ROUTE = "h3_fast_latent_2pass"


def _result_item(value: Any, index: int = 0):
    """Read both legacy tuples and Comfy's current NodeOutput objects."""
    try:
        return value[index]
    except (IndexError, KeyError, TypeError):
        result = getattr(value, "result", None)
        if result is None:
            raise
        return result[index]


def _fast_stage_size(plan: dict[str, Any]) -> tuple[int, int]:
    """R41 reuses R38 tile width/height as the direct Stage-2 canvas.

    Those fields are not used for tiling on this route.  Reusing them avoids a
    positional widget/schema change in the proven Shotboard and Settings nodes.
    """
    upres = _upres_settings(plan)
    native_w = max(256, int(plan.get("width", 960) or 960))
    native_h = max(256, int(plan.get("height", 544) or 544))
    stage_w = _grid_up(int(upres.get("tile_width", 1504) or 1504))
    stage_h = _grid_up(int(upres.get("tile_height", 832) or 832))
    if stage_w < native_w or stage_h < native_h:
        raise ValueError(
            f"FAST LATENT 2-PASS Stage 2 must not downscale: native={native_w}x{native_h}, "
            f"stage2={stage_w}x{stage_h}. Apply a Fast Latent preset or increase the Stage-2 canvas."
        )
    return stage_w, stage_h


def _load_upscaler_class():
    import nodes

    klass = nodes.NODE_CLASS_MAPPINGS.get("MinimaxH3LatentUpscaler3D")
    if klass is None:
        raise RuntimeError(
            "FAST LATENT 2-PASS requires Comfyui_Minimax_h3_latent_Upscaler and its "
            "MinimaxH3LatentUpscaler3D node."
        )
    return klass


def _split_av(av_latent):
    from comfy_extras.nodes_lt import LTXVSeparateAVLatent

    result = LTXVSeparateAVLatent.execute(av_latent=av_latent)
    return _result_item(result, 0), _result_item(result, 1)


def _concat_av(video_latent, audio_latent):
    from comfy_extras.nodes_lt import LTXVConcatAVLatent

    result = LTXVConcatAVLatent.execute(video_latent=video_latent, audio_latent=audio_latent)
    return _result_item(result, 0)


_UPSCALER_PATCH_LOCK = threading.RLock()


def _adaptive_temporal_chunks(width: int, height: int, device: str, requested: str = "AUTO") -> list[int]:
    """Resolve the FAST LATENT 3D chunk policy.

    The upstream 3D latent upscaler currently uses a hard-coded temporal chunk
    of 32.  At 1.3MP+ this can push 12-16GB cards over the edge because the
    effective segment also contains temporal overlap on both sides.  IAMCCS
    treats this as a delivery policy: the filmmaker selects HD 2-PASS and the
    backend chooses a safe chunk.
    """
    requested = str(requested or "AUTO").strip().upper()
    if requested not in {"AUTO", "32", "24", "16"}:
        requested = "AUTO"

    chain = {32: [32, 24, 16], 24: [24, 16], 16: [16]}
    if requested != "AUTO":
        primary = int(requested)
        LOG.info(
            "FAST LATENT 2-PASS temporal policy | setting=%s | attempts=%s",
            requested, chain[primary],
        )
        return chain[primary]

    if str(device).lower() != "cuda" or not torch.cuda.is_available():
        return [32]
    target_mp = max(1, int(width)) * max(1, int(height)) / 1_000_000.0
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        free_gb = free_bytes / (1024 ** 3)
    except Exception:
        free_gb = 12.0

    if free_gb <= 9.5 or (target_mp >= 2.0 and free_gb < 18.0):
        primary = 16
    elif free_gb <= 14.5 or (target_mp >= 1.25 and free_gb < 20.0):
        primary = 24
    else:
        primary = 32

    LOG.info(
        "FAST LATENT 2-PASS temporal policy | setting=AUTO | target=%dx%d %.3fMP | free_vram=%.2fGB | attempts=%s",
        int(width), int(height), target_mp, free_gb, chain[primary],
    )
    return chain[primary]


@contextmanager
def _temporary_upscaler_temporal_chunk(klass, chunk_size: int):
    """Temporarily override only the upstream model forward temporal stride.

    This keeps IAMCCS self-contained: users do not need to patch the external
    Comfyui_Minimax_h3_latent_Upscaler checkout.  The original method is always
    restored after the call, even when CUDA raises OOM.
    """
    module = importlib.import_module(klass.__module__)
    model_cls = getattr(module, "LatentResizer3D", None)
    if model_cls is None or not hasattr(model_cls, "_forward_seg"):
        yield False
        return

    original_forward = model_cls.forward
    chunk_size = max(8, int(chunk_size))

    def iamccs_forward(self, x, scale=None, target_size=None, enable_chunking=True):
        if target_size is not None:
            size = target_size
        elif scale is not None:
            size = tuple(int(round(s * scale)) for s in x.shape[-3:])
        else:
            return x
        if size == x.shape[-3:]:
            return x

        B, C, T, H, W = x.shape
        overlap = 0
        for block in self.in_blocks:
            dwconv = getattr(block, "dwconv", None)
            weight = getattr(dwconv, "weight", None)
            if weight is not None and weight.ndim == 5 and int(weight.shape[2]) > 1:
                overlap = int(weight.shape[2])
                break

        chunk = chunk_size
        if not enable_chunking or T <= chunk:
            return self._forward_seg(x, scale, size)

        print(
            f"[IAMCCS MinimaxH3-3D] adaptive temporal chunking: "
            f"T={T} chunk={chunk} chunks={(T + chunk - 1) // chunk} overlap={overlap}"
        )
        x_padded = F.pad(x, (0, 0, 0, 0, overlap, overlap), mode="replicate")
        out_full = torch.zeros(B, C, T, size[-2], size[-1], device=x.device, dtype=x.dtype)
        weight_full = torch.zeros(1, 1, T, 1, 1, device=x.device, dtype=x.dtype)

        start = 0
        while start < T:
            seg_start = start
            seg_end = min(T, start + chunk)
            out_start = max(0, seg_start - overlap)
            out_end = min(T, seg_end + overlap)
            lo = max(0, out_start - overlap)
            hi = min(T + 2 * overlap, out_end + overlap)

            seg = x_padded[:, :, lo:hi].contiguous()
            seg_size = (hi - lo, size[-2], size[-1])
            seg_out = self._forward_seg(seg, scale, seg_size)

            s0 = (out_start + overlap) - lo
            s1 = s0 + (out_end - out_start)
            valid_out = seg_out[:, :, s0:s1]
            n_valid = out_end - out_start

            weight = torch.ones(n_valid, device=x.device, dtype=x.dtype)
            if seg_start > out_start:
                blend_len = seg_start - out_start
                weight[:blend_len] = torch.arange(1, blend_len + 1, device=x.device, dtype=x.dtype) / (blend_len + 1)
            if out_end > seg_end:
                blend_len = out_end - seg_end
                weight[-blend_len:] = torch.arange(blend_len, 0, -1, device=x.device, dtype=x.dtype) / (blend_len + 1)

            out_full[:, :, out_start:out_end] += valid_out * weight.view(1, 1, n_valid, 1, 1)
            weight_full[:, :, out_start:out_end] += weight.view(1, 1, n_valid, 1, 1)
            start += chunk
            del seg, seg_out, valid_out, weight

        return out_full / weight_full.clamp(min=1e-8)

    with _UPSCALER_PATCH_LOCK:
        model_cls.forward = iamccs_forward
        try:
            yield True
        finally:
            model_cls.forward = original_forward


def _release_upscaler_after_oom(klass) -> None:
    """Move cached upscaler weights off CUDA before an adaptive retry."""
    try:
        module = importlib.import_module(klass.__module__)
        cache = getattr(module, "MODEL_CACHE", {})
        if isinstance(cache, dict):
            for model in cache.values():
                try:
                    model.to("cpu")
                except Exception:
                    pass
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            import comfy.model_management as mm
            mm.soft_empty_cache()
        except Exception:
            torch.cuda.empty_cache()


def _upscale_video_latent(video_latent, plan, width, height):
    upres = _upres_settings(plan)
    model_name = str(upres.get("model_name", "") or "").strip()
    if not model_name or not folder_paths.get_full_path("latent_upscale_models", model_name):
        raise ValueError(
            "FAST LATENT 2-PASS needs an installed minimax_h3_latent_upscaler_3d checkpoint "
            "selected in IAMCCS H3 Settings."
        )
    precision = str(upres.get("precision", "bf16") or "bf16").lower()
    device = str(upres.get("device", "cuda") or "cuda").lower()
    chunk_setting = str(upres.get("fast_latent_temporal_chunk", "AUTO") or "AUTO").strip().upper()
    klass = _load_upscaler_class()
    dynamic_mode = {"mode": "target dimensions", "width": int(width), "height": int(height)}
    attempts = _adaptive_temporal_chunks(int(width), int(height), device, chunk_setting)
    last_oom = None

    for attempt_index, temporal_chunk in enumerate(attempts, start=1):
        LOG.info(
            "FAST LATENT 2-PASS learned upres start | model=%s | target=%dx%d | %s/%s | chunk_setting=%s | temporal_chunk=%d | attempt=%d/%d",
            model_name, width, height, device, precision, chunk_setting, temporal_chunk, attempt_index, len(attempts),
        )
        try:
            with _temporary_upscaler_temporal_chunk(klass, temporal_chunk) as patched:
                if not patched and temporal_chunk != 32:
                    LOG.warning(
                        "FAST LATENT 2-PASS temporal override unavailable; falling back to the upstream temporal policy"
                    )
                result = klass.execute(
                    latent=video_latent,
                    model_name=model_name,
                    mode=dynamic_mode,
                    align=32,
                    enable_temporal_chunking=True,
                    force_unload=True,
                    device=device,
                    precision=precision,
                )
            return _result_item(result, 0)
        except torch.OutOfMemoryError as exc:
            last_oom = exc
            _release_upscaler_after_oom(klass)
            if attempt_index >= len(attempts) or temporal_chunk <= 16:
                LOG.error(
                    "FAST LATENT 2-PASS learned upres exhausted temporal chunks at %d for %dx%d",
                    temporal_chunk, width, height,
                )
                raise
            LOG.warning(
                "FAST LATENT 2-PASS CUDA OOM at temporal_chunk=%d; retrying only learned upres with chunk=%d. Native H3 is already checkpointed.",
                temporal_chunk, attempts[attempt_index],
            )

    if last_oom is not None:
        raise last_oom
    raise RuntimeError("FAST LATENT 2-PASS learned upres did not execute")


def _locked_av_latent(upscaled_video, original_audio):
    video = dict(upscaled_video)
    audio = dict(original_audio)
    video["noise_mask"] = torch.ones_like(video["samples"])
    # Stage 2 is a visual refinement.  The native H3 audio latent remains the
    # exact phonetic/timing authority and is never re-denoised.
    audio["noise_mask"] = torch.zeros_like(audio["samples"])
    return _concat_av(video, audio)


def _check_segment_parity(plan, index, total, native_frames, native_audio):
    """Fail before HD sampling if the visible/audio source is not chunk i."""
    chunks = plan.get("chunks") or []
    if len(chunks) != total or not 0 <= index < len(chunks):
        raise ValueError(
            f"FAST LATENT 2-PASS segment count disagrees with Shotboard: "
            f"queue={total}, plan={len(chunks)}, index={index}"
        )
    chunk = chunks[index]
    if int(chunk.get("index", index)) != index:
        raise ValueError("FAST LATENT 2-PASS chunk index is not the Shotboard index")
    visible = int(native_frames.shape[0])
    if str(plan.get("task_mode", "")).lower() == "longvid_motion_context":
        expected = int(chunk.get("visible_frame_count", visible))
        if visible != expected:
            raise ValueError(
                f"FAST LATENT 2-PASS chunk {index + 1}: native {visible}f "
                f"does not match Shotboard visible {expected}f"
            )
    waveform = native_audio.get("waveform") if isinstance(native_audio, dict) else None
    rate = int(native_audio.get("sample_rate", 0) or 0) if isinstance(native_audio, dict) else 0
    if not torch.is_tensor(waveform) or rate <= 0:
        raise ValueError("FAST LATENT 2-PASS needs the matching native AUDIO for this chunk")
    audio_frames = float(waveform.shape[-1]) / rate * 24.0
    if abs(audio_frames - visible) > 1.0:
        raise ValueError(
            f"FAST LATENT 2-PASS chunk {index + 1}: audio spans {audio_frames:.2f}f "
            f"but native video spans {visible}f"
        )
    return chunk


def _target_conditioning(model, clip, video_vae, audio_vae, cine_linx, segment_index,
                         stage_width, stage_height, stage1_conditioning,
                         bridge_frame=None, first_frame_override=None, last_frame_override=None,
                         ref_image_1=None, ref_image_2=None, ref_image_3=None, ref_image_4=None,
                         ref_video=None, ref_video_audio=None, ref_audio=None,
                         motion_state=None, render_id=""):
    plan = _resolve_shotplan(cine_linx)
    # Conditioning tensors that contain image/guide latents are spatially
    # authored. Re-materialize them at the Stage-2 grid. Atomic deliberately
    # leaves R37 Motion Context positioned guides to its separate condition
    # node, so that mode needs the same guide pass again at target resolution.
    target_plan = dict(plan)
    target_plan["width"] = int(stage_width)
    target_plan["height"] = int(stage_height)
    if (
        isinstance(motion_state, dict)
        and str(motion_state.get("method", "")) == "external_saved_av_continuation"
    ):
        # The source checkpoint has the Stage-1 latent grid and cannot be
        # reloaded as direct context on a larger Stage-2 grid. The upscaled
        # sampled latent already carries that AV history; rebuild only the
        # new Shotboard mode/endpoint conditioning at the target canvas.
        target_plan["continuation_settings"] = {
            **dict(target_plan.get("continuation_settings") or {}),
            "enabled": False,
        }
    target_plan["upscale_enabled"] = False
    target_plan["upscale_mode"] = "off"
    target_linx = _replace_plan(cine_linx, target_plan)
    result = IAMCCS_MiniMaxH3AtomicConditioningBackend().prepare(
        model=model,
        clip=clip,
        video_vae=video_vae,
        audio_vae=audio_vae,
        cine_linx=target_linx,
        segment_index=int(segment_index),
        bridge_frame=bridge_frame,
        render_id=render_id,
        first_frame_override=first_frame_override,
        last_frame_override=last_frame_override,
        ref_image_1=ref_image_1,
        ref_image_2=ref_image_2,
        ref_image_3=ref_image_3,
        ref_image_4=ref_image_4,
        ref_video=ref_video,
        ref_video_audio=ref_video_audio,
        ref_audio=ref_audio,
    )
    target_conditioning = result[1]
    guide_report = "none"
    task_mode = str(plan.get("task_mode", "") or "").lower()
    if task_mode in {"longvid_motion_context", "longvid_continuous_guided"}:
        from .iamccs_minimax_h3_motion_context_variant import _apply_positioned_guides

        chunks = plan.get("chunks") or []
        if not 0 <= int(segment_index) < len(chunks):
            raise IndexError("FAST LATENT 2-PASS Motion Context segment is outside the Shotboard plan")
        chunk = chunks[int(segment_index)]
        context_offset = max(0, int(chunk.get("motion_context_trim_frames", 0) or 0))
        target_conditioning, applied = _apply_positioned_guides(
            target_conditioning, result[2], video_vae, audio_vae,
            target_plan, chunk, context_offset,
        )
        guide_report = ",".join(applied) if applied else "none"
    return (
        result[0], target_conditioning,
        f"mode-matched target conditioning {stage_width}x{stage_height}; "
        f"Stage-1 context retained in latent; R37 target guides={guide_report}",
    )


def _stage2_window_policy(plan: dict[str, Any]) -> tuple[int | None, int, str]:
    """Resolve the H3 Stage-2 temporal sampling budget.

    This is deliberately independent from ``fast_latent_temporal_chunk``:
    the latter only chunks the learned 3D latent upscaler. Stage-2 runs the
    full MiniMax H3 transformer and needs its own temporal window on 8-16GB
    GPUs, especially once the spatial latent has been doubled.
    """
    upres = _upres_settings(plan)
    requested = str(upres.get("fast_latent_stage2_window", "AUTO") or "AUTO").strip().upper()
    overlap_requested = str(upres.get("fast_latent_stage2_overlap", "AUTO") or "AUTO").strip().upper()
    width, height = _delivery_size(plan)
    target_mp = max(1, int(width)) * max(1, int(height)) / 1_000_000.0

    if requested == "FULL":
        return None, 0, "FULL"
    if requested in {"39", "56", "73", "107"}:
        window = int(requested)
        source = requested
    else:
        try:
            total_gb = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 ** 3)
        except Exception:
            total_gb = 12.0
        # Conservative defaults. 1536x896 (1.376MP) on a 12GB card lands on
        # 56f, reducing the QKV projection working set to roughly one sixth of
        # the former 362f all-at-once refine.
        if total_gb <= 10.0:
            window = 39
        elif total_gb <= 13.0:
            window = 56 if target_mp >= 1.20 else 73
        elif total_gb <= 17.0:
            window = 73 if target_mp >= 1.20 else 107
        else:
            window = 107
        source = f"AUTO:{total_gb:.1f}GB/{target_mp:.3f}MP"

    if overlap_requested in {"5", "22"}:
        overlap = int(overlap_requested)
    else:
        overlap = 22 if window >= 56 else 5
    overlap = max(0, min(overlap, window - 1))
    return window, overlap, source


def _sample_stage2(model, conditioning, latent, cine_linx, segment_index):
    from comfy_extras.nodes_custom_sampler import BasicGuider, SamplerCustomAdvanced

    active_model, noise, sampler, sigmas, sampling_report = (
        IAMCCS_MiniMaxH3LatentUpresSamplingR38().prepare(model, cine_linx, int(segment_index))[:5]
    )
    guider = _result_item(BasicGuider.execute(model=active_model, conditioning=conditioning), 0)
    plan = _resolve_shotplan(cine_linx)
    window_frames, overlap_frames, policy_source = _stage2_window_policy(plan)
    from .iamccs_h3_stage2_guides import conditioning_guides, pin_still_guides
    from .iamccs_minimax_h3_pixel_refine_variant import _provider
    guides = conditioning_guides(conditioning)
    guide_mode = str(_upres_settings(plan).get("fast_latent_stage2_guides", "soft")).lower()
    if guide_mode == "masked":
        from comfy.nested_tensor import NestedTensor
        common = _provider("common")
        loop = _provider("nodes_looping_sampler")
        if not loop.per_row_mask_is_continuous():
            raise RuntimeError("Stage-2 Masked guides require H3 per-row mask support")
        video, audio = common.unpack_av(latent, "Stage-2 guides")
        vm, am = loop._split_mask(latent)
        video, vm, count = pin_still_guides(video, vm, guides)
        am = torch.ones_like(audio) if am is None else am
        latent = common.pack_av(latent, video, audio, noise_mask=NestedTensor([vm, am]))
        LOG.info("FAST LATENT Stage-2 Masked: pinned %d image guides; audio mask preserved", count)
    LOG.info("FAST LATENT Stage-2 guides=%d mode=%s", len(guides), guide_mode)

    if window_frames is None:
        LOG.warning(
            "FAST LATENT 2-PASS Stage-2 policy=FULL; all frames will be refined in one H3 sample. "
            "This can OOM on 12GB GPUs at HD delivery resolutions."
        )
        sampled = SamplerCustomAdvanced.execute(
            noise=noise, guider=guider, sampler=sampler, sigmas=sigmas, latent_image=latent,
        )
        return _result_item(sampled, 1), sampling_report + "; stage2=FULL"

    from .iamccs_minimax_h3_pixel_refine_variant import _provider
    loop = _provider("nodes_looping_sampler")
    if not loop.per_row_mask_is_continuous():
        raise RuntimeError(
            "FAST LATENT Stage-2 windowed refine requires the current H3 continuous per-row mask contract. "
            "Update ComfyUI rather than falling back to the OOM-prone full-window sampler."
        )
    common = _provider("common")
    original_video, original_audio = common.unpack_av(latent, "FAST LATENT Stage-2 source")
    total_frames = common.latents_to_frames(int(original_video.shape[2]))
    length, overlap, _, _, windows = _provider("nodes_windows")._plan(
        total_frames, int(window_frames), int(overlap_frames), "standard_static"
    )
    LOG.info(
        "FAST LATENT 2-PASS Stage-2 windowed refine | policy=%s | passes=%d | window=%df | overlap=%df | total=%df | %s",
        policy_source, len(windows), common.latents_to_frames(length),
        common.frame_at_latent(overlap) if overlap else 0, total_frames, sampling_report,
    )
    sampled = loop.MMH3LoopingSampler.execute(
        noise=noise, guider=guider, sampler=sampler, sigmas=sigmas,
        cond_set={"conds": [conditioning], "encoded_keyframes": guides}, latent=latent,
        chunk_frames=int(window_frames), overlap_frames=int(overlap_frames),
        carry="mask", overlap_strength_video=1.0, overlap_strength_audio=1.0,
        audio_denoise_mask=torch.zeros((1, 1, 1)),
    )[0]
    video, audio = common.unpack_av(sampled, "FAST LATENT Stage-2 refined")
    if tuple(video.shape) != tuple(original_video.shape):
        raise RuntimeError("FAST LATENT Stage-2 windowing changed the video latent extent")
    if original_audio is not None and not torch.equal(original_audio.cpu(), audio.cpu()):
        raise RuntimeError("FAST LATENT Stage-2 windowing changed the locked H3 audio latent")
    return sampled, sampling_report + f"; stage2_window={common.latents_to_frames(length)}f/{len(windows)}pass"


class IAMCCS_MiniMaxH3FastLatent2PassR41:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "video_vae": ("VAE",),
                "audio_vae": ("VAE",),
                "sampled_latent": ("LATENT",),
                "stage1_conditioning": ("CONDITIONING",),
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "native_frames": ("IMAGE",),
                "native_audio": ("AUDIO",),
                "resolved_render_id": ("STRING", {"forceInput": True}),
                "native_saved_report": ("STRING", {"forceInput": True}),
                "current_segment": ("INT", {"forceInput": True}),
                "total_segments": ("INT", {"forceInput": True}),
                "context_trim_frames": ("INT", {"forceInput": True}),
                "join_trim_frames": ("INT", {"forceInput": True}),
                "queue_next_segment": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "motion_state": ("IAMCCS_H3_MOTION_CONTEXT",),
                "bridge_frame": ("IMAGE",),
                "first_frame_override": ("IMAGE",),
                "last_frame_override": ("IMAGE",),
                "ref_image_1": ("IMAGE",),
                "ref_image_2": ("IMAGE",),
                "ref_image_3": ("IMAGE",),
                "ref_image_4": ("IMAGE",),
                "ref_video": ("IMAGE",),
                "ref_video_audio": ("AUDIO",),
                "ref_audio": ("AUDIO",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "report")
    FUNCTION = "finish"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def finish(self, model, clip, video_vae, audio_vae, sampled_latent, stage1_conditioning,
               cine_linx, native_frames, native_audio, resolved_render_id, native_saved_report,
               current_segment, total_segments, context_trim_frames, join_trim_frames,
               queue_next_segment=True, motion_state=None, bridge_frame=None,
               first_frame_override=None, last_frame_override=None,
               ref_image_1=None, ref_image_2=None, ref_image_3=None, ref_image_4=None,
               ref_video=None, ref_video_audio=None, ref_audio=None,
               prompt=None, extra_pnginfo=None):
        import comfy.model_management as mm
        from .iamccs_minimax_h3_shotboard import (
            _concat_videos,
            _concat_videos_overlap,
            _current_prompt,
            _encode_images,
            _enqueue,
            _joined_frame_count,
            _trim_audio_frames,
            _write_segment_metadata,
        )

        plan = _resolve_shotplan(cine_linx)
        enabled = bool(plan.get("upscale_enabled")) and str(plan.get("upscale_mode", "off")) == FAST_ROUTE
        if not enabled:
            raise ValueError("This workflow is FAST LATENT 2-PASS. Enable that exact upscale route in IAMCCS H3 Settings.")
        if not native_saved_report:
            raise ValueError("FAST LATENT 2-PASS must run after the native H3 checkpoint.")

        index, total = int(current_segment), max(1, int(total_segments))
        if index < 0 or index >= total:
            raise ValueError("FAST LATENT 2-PASS received an invalid segment index.")
        chunk = _check_segment_parity(plan, index, total, native_frames, native_audio)
        if (
            isinstance(motion_state, dict)
            and str(motion_state.get("method", "")) == "external_saved_av_continuation"
        ):
            raise ValueError(
                "FAST LATENT 2-PASS does not yet upscale an external saved-AV continuation: "
                "the source checkpoint lives on the native latent grid. Use native or Pixel Safe delivery "
                "for this continuation render; existing Shotboard/LongVid modes remain supported by Fast Latent."
            )
        run = _safe_render_id(resolved_render_id)
        root = Path(folder_paths.get_output_directory()) / "IAMCCS" / "MiniMaxH3" / "FAST_LATENT_2PASS" / run
        root.mkdir(parents=True, exist_ok=True)
        output = root / f"segment_{index + 1:04d}.mp4"
        if output.exists():
            raise FileExistsError(f"FAST LATENT 2-PASS refuses to overwrite {output}.")

        visible = int(native_frames.shape[0])
        from .iamccs_minimax_h3_shotboard import _delivery_join_frames
        join = _delivery_join_frames(cine_linx, index, join_trim_frames)
        frames = visible - (1 if join == 1 else 0)
        audio = _trim_audio_frames(native_audio, 1, 24) if join == 1 else native_audio
        trim = max(0, int(context_trim_frames))
        if isinstance(motion_state, dict) and motion_state.get("active"):
            trim = max(trim, int(motion_state.get("trim_frames", 0) or 0))

        delivery_width, delivery_height = _delivery_size(plan)
        upres = _upres_settings(plan)
        final_rtx = bool(upres.get("rtx_requested", False))

        # 2026-09-25 R41 canvas-truth fix:
        # h3_upres_tile_width/height are legacy R38 spatial tile controls. They
        # are NOT the Fast Latent Stage-2 output canvas. Without RTX the learned
        # latent upres + H3 refinement must run directly on the authored delivery
        # canvas (linked HIGH or explicit upscale_width/upscale_height). Reusing
        # the tile defaults here was the source of the false 864x480 ceiling.
        # With RTX enabled we preserve the intentional intermediate Stage-2
        # canvas, because RTX owns the separate final delivery resize.
        if final_rtx:
            stage_width, stage_height = _fast_stage_size(plan)
            stage_canvas_source = "legacy_intermediate_for_rtx"
        else:
            stage_width, stage_height = int(delivery_width), int(delivery_height)
            stage_canvas_source = "delivery_canvas_truth"

        native_width = max(256, int(plan.get("width", 960) or 960))
        native_height = max(256, int(plan.get("height", 544) or 544))
        if stage_width < native_width or stage_height < native_height:
            raise ValueError(
                "FAST LATENT 2-PASS Stage 2 must not downscale: "
                f"native={native_width}x{native_height}, stage2={stage_width}x{stage_height}."
            )

        LOG.info(
            "FAST LATENT 2-PASS canvas truth | native=%dx%d | stage2=%dx%d (%s) | delivery=%dx%d | rtx=%s",
            native_width, native_height, stage_width, stage_height, stage_canvas_source,
            delivery_width, delivery_height, final_rtx,
        )

        LOG.info(
            "FAST LATENT 2-PASS start | segment=%d/%d | shotboard_start=%sf | native=%dx%d | stage2=%dx%d | delivery=%dx%d | audio=locked",
            index + 1, total, int(chunk.get("timeline_start_frame", 0)),
            int(plan.get("width", 0)), int(plan.get("height", 0)),
            stage_width, stage_height, delivery_width, delivery_height,
        )

        intermediate = None
        try:
            mm.unload_all_models()
            mm.soft_empty_cache()
            stage1_video, stage1_audio = _split_av(sampled_latent)
            upscaled_video = _upscale_video_latent(stage1_video, plan, stage_width, stage_height)
            del stage1_video
            locked_latent = _locked_av_latent(upscaled_video, stage1_audio)
            del upscaled_video
            mm.unload_all_models()
            mm.soft_empty_cache()

            target_model, target_conditioning, conditioning_report = _target_conditioning(
                model, clip, video_vae, audio_vae, cine_linx, index,
                stage_width, stage_height, stage1_conditioning,
                bridge_frame=bridge_frame,
                first_frame_override=first_frame_override,
                last_frame_override=last_frame_override,
                ref_image_1=ref_image_1, ref_image_2=ref_image_2,
                ref_image_3=ref_image_3, ref_image_4=ref_image_4,
                ref_video=ref_video, ref_video_audio=ref_video_audio, ref_audio=ref_audio,
                motion_state=motion_state, render_id=run,
            )
            mm.unload_all_models()
            mm.soft_empty_cache()
            refined, sampling_report = _sample_stage2(
                target_model, target_conditioning, locked_latent, cine_linx, index,
            )
            del locked_latent
            mm.unload_all_models()
            mm.soft_empty_cache()

            intermediate = _provider_stream_save(
                refined, video_vae, run, index, int(upres.get("pixel_groups", 1) or 1),
            )
            del refined
            if final_rtx:
                mm.unload_all_models()
                mm.soft_empty_cache()
                _rtx_finish_segment(
                    intermediate, output, audio, trim + (1 if join == 1 else 0), frames,
                    stage_width, stage_height, delivery_width, delivery_height, 24,
                    str(upres.get("rtx_quality", "ULTRA") or "ULTRA").upper(),
                )
            else:
                _finish_segment(
                    intermediate, output, audio, trim + (1 if join == 1 else 0), frames,
                    delivery_width, delivery_height, 24,
                )
        finally:
            mm.unload_all_models()
            gc.collect()
            mm.soft_empty_cache()

        provenance = {
            "render_id": run, "stage": "fast_latent_2pass", "segment_index": index,
            "total_segments": total, "native_visible_frames": visible,
            "delivery_frames": frames, "context_trim_frames": trim,
            "join_trim_frames": join, "native_audio_locked": True,
            "stage2_canvas": [stage_width, stage_height],
            "delivery_canvas": [delivery_width, delivery_height],
            "chunk": chunk, "shotplan": plan,
        }
        source_workflow = extra_pnginfo.get("workflow") if isinstance(extra_pnginfo, dict) else None
        _write_segment_metadata(output, frames, 24, "fast_latent_2pass_audio_locked",
                                provenance=provenance, source_workflow=source_workflow)
        preview = output
        if index == total - 1:
            paths = [root / f"segment_{part + 1:04d}.mp4" for part in range(total)]
            preview = root / "final_film.mp4"
            if preview.exists():
                raise FileExistsError(f"FAST LATENT 2-PASS final film already exists: {preview}")
            if join > 1:
                _concat_videos_overlap(paths, preview, join, 24)
            else:
                _concat_videos(paths, preview)
            master_frames = _joined_frame_count(paths, join if join > 1 else 0)
            _write_segment_metadata(
                preview, master_frames,
                24, "fast_latent_2pass_audio_locked",
                provenance={"render_id": run, "stage": "fast_latent_2pass_master", "total_segments": total,
                            "segment_files": [path.name for path in paths], "shotplan": plan},
                source_workflow=source_workflow,
            )

        queued = False
        if bool(queue_next_segment) and index + 1 < total:
            live, extra, outputs, sensitive = _current_prompt()
            next_prompt = _next_prompt(live if live is not None else prompt, run, index + 1)
            _enqueue(next_prompt, extra_data=extra, outputs=outputs, sensitive=sensitive)
            queued = True
        report = (
            f"FAST LATENT 2-PASS saved {index + 1}/{total} | native={plan.get('width')}x{plan.get('height')} "
            f"-> stage2={stage_width}x{stage_height} -> delivery={delivery_width}x{delivery_height} | "
            f"audio=exact native lock | next queued={queued} | {preview}"
        )
        LOG.info("%s | %s | %s", report, conditioning_report, sampling_report)
        return {
            "ui": {
                "text": [report],
                "images": [{
                    "filename": preview.name,
                    "subfolder": preview.parent.relative_to(Path(folder_paths.get_output_directory())).as_posix(),
                    "type": "output",
                }],
                "animated": (True,),
            },
            "result": (str(preview), report),
        }


def _provider_stream_save(latent, video_vae, run, index, groups, route_folder="FAST_LATENT_2PASS"):
    # Reuse the audited streamed VAE/ffmpeg implementation already vendored by
    # the conservative R38B route; this avoids a full high-resolution IMAGE batch.
    from .iamccs_minimax_h3_pixel_refine_variant import _provider

    return _provider("nodes_save").MMH3StreamingSave.execute(
        latent=latent,
        vae=video_vae,
        groups_per_chunk=max(1, int(groups)),
        fps=24.0,
        filename_prefix=f"IAMCCS/MiniMaxH3/{route_folder}/{run}/stage2_{index + 1:04d}",
        crf=16,
        save_metadata=False,
    )[0]


NODE_CLASS_MAPPINGS = {
    "IAMCCS_MiniMaxH3FastLatent2PassR41": IAMCCS_MiniMaxH3FastLatent2PassR41,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_MiniMaxH3FastLatent2PassR41": "MiniMax H3 · QUALITY LATENT 2-PASS · Streamed Delivery",
}
