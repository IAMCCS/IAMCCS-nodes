# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later

"""Isolated R38 MiniMax H3 learned latent-upres delivery variant."""

from __future__ import annotations

import gc
import logging
import math
import re
from pathlib import Path
from typing import Any

import folder_paths
import torch

from .iamccs_minimax_h3_atomic_backend import (
    H3_FPS,
    SUPERNODE_LINX_TYPE,
    IAMCCS_MiniMaxH3AtomicConditioningBackend,
    IAMCCS_MiniMaxH3DeliveryRouterV2,
    _accelerate,
    _apply_secondary_lora,
    _apply_turbo_lora,
    _resize_frames_exact_cpu,
    _resolve_shotplan,
)
from .iamccs_minimax_h3_motion_context_variant import _replace_plan


LOG = logging.getLogger("IAMCCS.MiniMaxH3.LatentUpresR38")
CATEGORY = "IAMCCS/MiniMax H3/Latent Upres R38"
H3_UPSCALE_PARAM = "H3_UPSCALE_PARAM"
H3_TEMPORAL_PARAM = "H3_TEMPORAL_PARAM"
H3_SPATIAL_PARAM = "H3_SPATIAL_PARAM"


def _upres_settings(plan: dict[str, Any]) -> dict[str, Any]:
    settings = plan.get("upscale_settings") if isinstance(plan.get("upscale_settings"), dict) else {}
    upres = settings.get("h3_latent_upres") if isinstance(settings.get("h3_latent_upres"), dict) else {}
    return upres


def _delivery_size(plan: dict[str, Any]) -> tuple[int, int]:
    settings = plan.get("upscale_settings") if isinstance(plan.get("upscale_settings"), dict) else {}
    native_width = max(256, int(plan.get("width", 960) or 960))
    native_height = max(256, int(plan.get("height", 544) or 544))
    return (
        max(256, int(settings.get("target_width", native_width * 2) or native_width * 2)),
        max(256, int(settings.get("target_height", native_height * 2) or native_height * 2)),
    )


def _grid_up(value: int, multiple: int = 32) -> int:
    return max(multiple, int(math.ceil(int(value) / multiple) * multiple))


def _safe_render_id(value: Any) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "iamccs_h3_r38")).strip("._")
    return name or "iamccs_h3_r38"


def _exact_delivery(frames: torch.Tensor, width: int, height: int) -> tuple[torch.Tensor, str]:
    source_height = int(frames.shape[1])
    source_width = int(frames.shape[2])
    if source_width == int(width) and source_height == int(height):
        return frames, "exact"
    if source_width >= int(width) and source_height >= int(height):
        left = max(0, (source_width - int(width)) // 2)
        top = max(0, (source_height - int(height)) // 2)
        cropped = frames[:, top:top + int(height), left:left + int(width), :]
        if int(cropped.shape[1]) == int(height) and int(cropped.shape[2]) == int(width):
            return cropped.contiguous(), f"center_crop {source_width}x{source_height}->{width}x{height}"
    return _resize_frames_exact_cpu(frames, int(width), int(height))


class IAMCCS_MiniMaxH3LatentUpresControlR38:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"cine_linx": (SUPERNODE_LINX_TYPE,)}}

    RETURN_TYPES = (
        H3_UPSCALE_PARAM, H3_TEMPORAL_PARAM, H3_SPATIAL_PARAM,
        "INT", "INT", "INT", "INT", "BOOLEAN", "STRING", "STRING",
    )
    RETURN_NAMES = (
        "latent_upscale_param", "temporal_split_param", "spatial_split_param",
        "stage_width", "stage_height", "delivery_width", "delivery_height",
        "rtx_enabled", "rtx_quality", "report",
    )
    FUNCTION = "prepare"
    CATEGORY = CATEGORY

    def prepare(self, cine_linx):
        plan = _resolve_shotplan(cine_linx)
        enabled = bool(plan.get("upscale_enabled", False)) and str(plan.get("upscale_mode", "off")).lower() == "h3_latent_upres"
        upres = _upres_settings(plan)
        model_name = str(upres.get("model_name", "") or "").strip()
        if enabled and (not model_name or not folder_paths.get_full_path("latent_upscale_models", model_name)):
            raise ValueError(
                "H3 Latent Upres requires an installed minimax_h3_latent_upscaler_3d checkpoint. "
                "Select an existing file in IAMCCS H3 Settings; FP16, BF16 and FP32 are supported."
            )
        delivery_width, delivery_height = _delivery_size(plan)
        rtx_requested = bool(upres.get("rtx_requested", False))
        rtx_enabled = bool(enabled and rtx_requested)
        if rtx_enabled:
            stage_width = _grid_up(math.ceil(delivery_width / 2))
            stage_height = _grid_up(math.ceil(delivery_height / 2))
        else:
            stage_width = _grid_up(delivery_width)
            stage_height = _grid_up(delivery_height)

        tile_width = min(stage_width, _grid_up(int(upres.get("tile_width", 864) or 864)))
        tile_height = min(stage_height, _grid_up(int(upres.get("tile_height", 480) or 480)))
        overlap_width = max(0, _grid_up(int(upres.get("overlap_width", 128) or 0)) if int(upres.get("overlap_width", 128) or 0) else 0)
        overlap_height = max(0, _grid_up(int(upres.get("overlap_height", 128) or 0)) if int(upres.get("overlap_height", 128) or 0) else 0)
        overlap_width = min(overlap_width, max(0, tile_width - 32))
        overlap_height = min(overlap_height, max(0, tile_height - 32))
        fade_width = min(overlap_width, max(0, _grid_up(int(upres.get("fade_width", 32) or 0)) if int(upres.get("fade_width", 32) or 0) else 0))
        fade_height = min(overlap_height, max(0, _grid_up(int(upres.get("fade_height", 32) or 0)) if int(upres.get("fade_height", 32) or 0) else 0))
        min_tile_size = min(tile_width, tile_height, max(0, _grid_up(int(upres.get("min_tile_size", 256) or 0)) if int(upres.get("min_tile_size", 256) or 0) else 0))

        chunk_length = max(17, int(upres.get("temporal_chunk", 85) or 85))
        chunk_length = max(17, round(chunk_length / 17) * 17)
        temporal_overlap = max(0, int(upres.get("temporal_overlap", 17) or 0))
        temporal_overlap = min(round(temporal_overlap / 17) * 17, chunk_length - 17)
        latent_param = {
            "model_name": model_name,
            "width": stage_width,
            "height": stage_height,
            "device": str(upres.get("device", "cuda") or "cuda"),
            "precision": str(upres.get("precision", "fp16") or "fp16"),
            "keep_resident": bool(upres.get("keep_models_resident", False)),
        }
        temporal_param = {
            "chunk_length": chunk_length,
            "temporal_overlap": temporal_overlap,
            "anchor_strength": max(0.0, min(1.0, float(upres.get("anchor_strength", 0.999) or 0.0))),
        }
        spatial_param = {
            "tile_width": tile_width,
            "tile_height": tile_height,
            "spatial_w_overlap": overlap_width,
            "spatial_h_overlap": overlap_height,
            "fade_width": fade_width,
            "fade_height": fade_height,
            "min_tile_size": min_tile_size,
            "overlap_mode": str(upres.get("overlap_mode", "earlier") or "earlier"),
            "overlap_blend": str(upres.get("overlap_blend", "linear") or "linear"),
        }
        quality = str(upres.get("rtx_quality", "ULTRA") or "ULTRA").upper()
        report = (
            f"R38 H3 Latent Upres | model={model_name} | stage={stage_width}x{stage_height} "
            f"-> delivery={delivery_width}x{delivery_height} | temporal={chunk_length}/{temporal_overlap} | "
            f"tile={tile_width}x{tile_height}+{overlap_width}x{overlap_height} | "
            f"rtx={'on' if rtx_enabled else 'off'}:{quality} | values=CineLinX"
        )
        if enabled:
            LOG.info(report)
        return latent_param, temporal_param, spatial_param, stage_width, stage_height, delivery_width, delivery_height, rtx_enabled, quality, report


class IAMCCS_MiniMaxH3LatentUpresSamplingR38:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "segment_index": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "NOISE", "SAMPLER", "SIGMAS", "STRING")
    RETURN_NAMES = ("model", "noise", "sampler", "sigmas", "report")
    FUNCTION = "prepare"
    CATEGORY = CATEGORY

    def prepare(self, model, cine_linx, segment_index):
        from comfy_extras.nodes_custom_sampler import BasicScheduler, KSamplerSelect, RandomNoise
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift

        plan = _resolve_shotplan(cine_linx)
        upres = _upres_settings(plan)
        sampling = plan.get("sampling") if isinstance(plan.get("sampling"), dict) else {}
        seed = int(sampling.get("seed", 0) or 0)
        stride = int(sampling.get("seed_stride", 1) or 1)
        settings = plan.get("upscale_settings") if isinstance(plan.get("upscale_settings"), dict) else {}
        seed += int(segment_index) * stride + int(settings.get("seed_offset", 10000) or 0)
        seed &= 0xFFFFFFFFFFFFFFFF
        steps = max(1, int(upres.get("steps", 1) or 1))
        denoise = max(0.0, min(1.0, float(upres.get("denoise", 0.2) or 0.0)))
        sampler_name = str(upres.get("sampler", "sa_solver") or "sa_solver")
        scheduler = str(upres.get("scheduler", "simple") or "simple")
        shift_video = float(sampling.get("shift_video", 12.0))
        shift_audio = float(sampling.get("shift_audio", 3.0))
        turbo_model, turbo_report = _apply_turbo_lora(model, plan)
        lora_model, secondary_lora_report = _apply_secondary_lora(turbo_model, plan)
        shifted = MiniMaxH3SigmaShift.execute(model=lora_model, shift_video=shift_video, shift_audio=shift_audio)[0]
        if str(plan.get("acceleration", "native")).lower() == "comfy_kitchen":
            from comfy_extras.nodes_model_advanced import ModelAttentionBackend

            active_model = ModelAttentionBackend().patch(shifted, "comfy kitchen attention")[0]
            acceleration_report = "ComfyKitchen per-model attention"
        else:
            active_model, acceleration_report = _accelerate(shifted, plan)
        noise = RandomNoise.execute(noise_seed=seed)[0]
        sampler = KSamplerSelect.execute(sampler_name=sampler_name)[0]
        sigmas = BasicScheduler.execute(model=active_model, scheduler=scheduler, steps=steps, denoise=denoise)[0]
        report = (
            f"R38 refine sampling | seed={seed} | {steps} steps | {sampler_name}+{scheduler} | "
            f"denoise={denoise:.3f} | turbo={turbo_report} | secondary_lora={secondary_lora_report} | "
            f"acceleration={acceleration_report}"
        )
        LOG.info(report)
        return active_model, noise, sampler, sigmas, report


class IAMCCS_MiniMaxH3LatentUpresConditioningR38:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",), "clip": ("CLIP",), "video_vae": ("VAE",), "audio_vae": ("VAE",),
                "cine_linx": (SUPERNODE_LINX_TYPE,), "segment_index": ("INT", {"forceInput": True}),
                "target_width": ("INT", {"forceInput": True}), "target_height": ("INT", {"forceInput": True}),
            },
            "optional": {
                "stage1_conditioning": ("CONDITIONING",), "bridge_frame": ("IMAGE",), "render_id": ("STRING", {"default": ""}),
                "first_frame_override": ("IMAGE",), "last_frame_override": ("IMAGE",),
                "ref_image_1": ("IMAGE",), "ref_image_2": ("IMAGE",), "ref_image_3": ("IMAGE",), "ref_image_4": ("IMAGE",),
                "ref_video": ("IMAGE",), "ref_video_audio": ("AUDIO",), "ref_audio": ("AUDIO",),
                "stage1_latent": ("LATENT",),
                "stage1_motion_state": ("IAMCCS_H3_MOTION_CONTEXT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("conditioning", "target_latent", "report")
    FUNCTION = "prepare"
    CATEGORY = CATEGORY

    def prepare(self, model, clip, video_vae, audio_vae, cine_linx, segment_index, target_width, target_height,
                stage1_conditioning=None, bridge_frame=None, render_id="", first_frame_override=None,
                last_frame_override=None, ref_image_1=None, ref_image_2=None, ref_image_3=None, ref_image_4=None,
                ref_video=None, ref_video_audio=None, ref_audio=None, stage1_latent=None, stage1_motion_state=None):
        plan = _resolve_shotplan(cine_linx)
        shotboard_task = str(plan.get("task_mode", "") or "").lower()
        native_context = isinstance(stage1_motion_state, dict) and stage1_motion_state.get("active")
        if (shotboard_task == "longvid_motion_context" or native_context) and stage1_conditioning is not None and stage1_latent is not None:
            return (
                stage1_conditioning,
                stage1_latent,
                "R38 conditioning | reused Stage-1 Motion Context conditioning/latent | "
                "MMH3 performs internal target resize | no second Qwen encode",
            )
        target_plan = dict(plan)
        target_plan["width"] = _grid_up(target_width)
        target_plan["height"] = _grid_up(target_height)
        target_plan["upscale_enabled"] = False
        target_plan["upscale_mode"] = "off"
        target_linx = _replace_plan(cine_linx, target_plan)
        result = IAMCCS_MiniMaxH3AtomicConditioningBackend().prepare(
            model=model, clip=clip, video_vae=video_vae, audio_vae=audio_vae, cine_linx=target_linx,
            segment_index=segment_index, bridge_frame=bridge_frame, render_id=render_id,
            first_frame_override=first_frame_override, last_frame_override=last_frame_override,
            ref_image_1=ref_image_1, ref_image_2=ref_image_2, ref_image_3=ref_image_3, ref_image_4=ref_image_4,
            ref_video=ref_video, ref_video_audio=ref_video_audio, ref_audio=ref_audio,
        )
        positive, target_latent = result[1], result[2]
        if shotboard_task == "longvid_motion_context" and stage1_conditioning is not None:
            positive = stage1_conditioning
            source = "stage1_motion_context_conditioning"
        else:
            source = "mode_matched_target_conditioning"
        report = f"R38 conditioning | {source} | target={target_plan['width']}x{target_plan['height']} | {result[10]}"
        return positive, target_latent, report


class IAMCCS_MiniMaxH3LatentUpresChunkCheckpointR38:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,), "native_visible_frames": ("IMAGE",),
                "current_segment": ("INT", {"forceInput": True}), "total_segments": ("INT", {"forceInput": True}),
                "trim_head_frames": ("INT", {"forceInput": True}), "resolved_render_id": ("STRING", {"forceInput": True}),
                "delivery_width": ("INT", {"forceInput": True}), "delivery_height": ("INT", {"forceInput": True}),
            },
            "optional": {
                "upscaled_frames": ("IMAGE", {"lazy": True}),
                "join_trim_frames": ("INT", {"default": 0, "forceInput": True}),
                "motion_state": ("IAMCCS_H3_MOTION_CONTEXT",),
            },
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN", "STRING")
    RETURN_NAMES = ("frames", "master_ready", "report")
    FUNCTION = "collect"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def check_lazy_status(self, cine_linx, native_visible_frames, current_segment, total_segments, trim_head_frames,
                          resolved_render_id, delivery_width, delivery_height, upscaled_frames=None, **kwargs):
        plan = _resolve_shotplan(cine_linx)
        enabled = bool(plan.get("upscale_enabled", False)) and str(plan.get("upscale_mode", "off")).lower() == "h3_latent_upres"
        return ["upscaled_frames"] if enabled and upscaled_frames is None else []

    def collect(self, cine_linx, native_visible_frames, current_segment, total_segments, trim_head_frames,
                resolved_render_id, delivery_width, delivery_height, upscaled_frames=None, join_trim_frames=0, motion_state=None):
        plan = _resolve_shotplan(cine_linx)
        enabled = bool(plan.get("upscale_enabled", False)) and str(plan.get("upscale_mode", "off")).lower() == "h3_latent_upres"
        if not enabled:
            return native_visible_frames, False, "R38 chunk accumulator bypassed: H3 Latent Upres is off"
        if not torch.is_tensor(upscaled_frames):
            raise ValueError("R38 H3 Latent Upres is enabled, but the decoded Stage-2 IMAGE branch is not connected")
        index = int(current_segment)
        total = max(1, int(total_segments))
        trim = max(0, int(trim_head_frames))
        if isinstance(motion_state, dict) and motion_state.get("active") and motion_state.get("method") == "native_av_context":
            trim = max(0, int(motion_state.get("trim_frames", 0)))
        visible = int(native_visible_frames.shape[0])
        if int(upscaled_frames.shape[0]) < trim + visible:
            raise RuntimeError(
                f"R38 decoded {int(upscaled_frames.shape[0])} frames, but needs {trim}+{visible} for the visible chunk"
            )
        chunk = upscaled_frames[trim:trim + visible].detach().to(device="cpu", dtype=torch.float16).contiguous()
        join_trim = max(0, int(join_trim_frames))
        if join_trim == 1:
            chunk = chunk[1:].contiguous()
        root = Path(folder_paths.get_output_directory()) / "minimax_h3_shotboard" / "r38_latent_upres" / _safe_render_id(resolved_render_id)
        root.mkdir(parents=True, exist_ok=True)
        chunk_path = root / f"segment_{index:05d}.pt"
        torch.save({"frames": chunk, "overlap": join_trim if join_trim > 1 else 0}, chunk_path)
        LOG.info("R38 upscaled chunk parked | render=%s | segment=%d/%d | frames=%d", resolved_render_id, index + 1, total, int(chunk.shape[0]))
        ready = index >= total - 1
        if not ready:
            return chunk, False, f"R38 parked upscaled segment {index + 1}/{total} at {chunk_path}"
        missing = [part for part in range(total) if not (root / f"segment_{part:05d}.pt").is_file()]
        if missing:
            raise RuntimeError(f"R38 cannot assemble upscaled master; missing segments: {missing}")
        parts = []
        for part in range(total):
            saved = torch.load(root / f"segment_{part:05d}.pt", map_location="cpu", weights_only=True)
            frames = saved["frames"]
            overlap = int(saved["overlap"]) if parts else 0
            if overlap:
                overlap = min(overlap, int(parts[-1].shape[0]), int(frames.shape[0]))
                alpha = torch.linspace(0, 1, overlap, dtype=frames.dtype).reshape(-1, 1, 1, 1)
                seam = torch.lerp(parts[-1][-overlap:], frames[:overlap], alpha)
                parts[-1] = parts[-1][:-overlap]
                parts.append(seam)
                frames = frames[overlap:]
            parts.append(frames)
        master = torch.cat(parts, dim=0)
        exact_report = "H3 intermediate retained for RTX"
        if not bool(_upres_settings(plan).get("rtx_requested", False)):
            master, exact_report = _exact_delivery(master, int(delivery_width), int(delivery_height))
        LOG.info("R38 upscaled master ready | segments=%d | frames=%d | %s", total, int(master.shape[0]), exact_report)
        return master, True, f"R38 upscaled master ready | segments={total} | frames={int(master.shape[0])} | {exact_report}"


class IAMCCS_MiniMaxH3LatentUpresRTXR38:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"cine_linx": (SUPERNODE_LINX_TYPE,), "images": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "upscale"
    CATEGORY = CATEGORY

    def upscale(self, cine_linx, images):
        from .iamccs_rtx_vfx import apply_rtx_vfx

        plan = _resolve_shotplan(cine_linx)
        width, height = _delivery_size(plan)
        upres = _upres_settings(plan)
        quality = str(upres.get("rtx_quality", "ULTRA") or "ULTRA").upper()
        mode = {
            "ULTRA": "VSR Ultra", "HIGH": "VSR High", "MEDIUM": "VSR Medium", "LOW": "VSR Low",
        }.get(quality, "VSR Ultra")
        result = apply_rtx_vfx(
            images,
            mode=mode,
            resize_type="Manual",
            width=int(width),
            height=int(height),
            divisible_by="8",
            resize_method="Center Crop (Fill)",
        )
        return result, f"R38 NVIDIA RTX VSR | {mode} | target={width}x{height}"


class IAMCCS_MiniMaxH3DeliveryRouterR38:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,), "native_frames": ("IMAGE",), "native_audio": ("AUDIO",),
                "bridge_last_frame": ("IMAGE",), "master_ready": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "h3_upres_frames": ("IMAGE", {"lazy": True}),
                "h3_upres_rtx_frames": ("IMAGE", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "IMAGE", "INT", "STRING", "STRING")
    RETURN_NAMES = ("delivery_frames", "audio", "bridge_last_frame", "delivery_fps", "upscale_applied", "report")
    FUNCTION = "route"
    CATEGORY = CATEGORY

    @staticmethod
    def _selected(cine_linx, master_ready):
        if not bool(master_ready):
            return None
        plan = _resolve_shotplan(cine_linx)
        if not bool(plan.get("upscale_enabled", False)) or str(plan.get("upscale_mode", "off")).lower() != "h3_latent_upres":
            return None
        upres = _upres_settings(plan)
        return "h3_upres_rtx_frames" if bool(upres.get("rtx_requested", False)) else "h3_upres_frames"

    def check_lazy_status(self, cine_linx, native_frames, native_audio, bridge_last_frame, master_ready=True,
                          h3_upres_frames=None, h3_upres_rtx_frames=None, **kwargs):
        selected = self._selected(cine_linx, master_ready)
        if selected == "h3_upres_frames" and h3_upres_frames is None:
            return [selected]
        if selected == "h3_upres_rtx_frames" and h3_upres_rtx_frames is None:
            return [selected]
        return []

    def route(self, cine_linx, native_frames, native_audio, bridge_last_frame, master_ready=True,
              h3_upres_frames=None, h3_upres_rtx_frames=None):
        plan = _resolve_shotplan(cine_linx)
        selected = self._selected(cine_linx, master_ready)
        if bool(master_ready) and bool(plan.get("upscale_enabled", False)) and str(plan.get("upscale_mode", "off")) not in {"off", "h3_latent_upres"}:
            raise ValueError("This R38 workflow wires H3 Latent Upres / RTX only. Use the existing LTX/Wan workflow for that delivery route, or select H3 Latent Upres / Off.")
        width, height = _delivery_size(plan)
        if selected == "h3_upres_frames":
            delivery = h3_upres_frames
            applied = "H3 Latent Upres R38"
        elif selected == "h3_upres_rtx_frames":
            delivery = h3_upres_rtx_frames
            applied = "H3 Latent Upres R38 + NVIDIA RTX VSR"
        else:
            delivery = native_frames
            applied = "off/native"
        if not torch.is_tensor(delivery):
            raise ValueError(f"R38 delivery selected {selected}, but that lazy IMAGE branch is not connected")
        exact_report = "native"
        if selected:
            delivery, exact_report = _exact_delivery(delivery, width, height)
            import comfy.model_management as model_management

            model_management.unload_all_models()
            gc.collect()
            model_management.soft_empty_cache()
        rife_mode = str(plan.get("rife_mode", "off") or "off").lower()
        delivery, fps, rife_report = IAMCCS_MiniMaxH3DeliveryRouterV2._rife(delivery, rife_mode)
        report = f"R38 delivery | {applied} | {exact_report} | {rife_report} | native bridge preserved"
        return delivery, native_audio, bridge_last_frame, int(fps), applied, report


NODE_CLASS_MAPPINGS = {
    "IAMCCS_MiniMaxH3LatentUpresControlR38": IAMCCS_MiniMaxH3LatentUpresControlR38,
    "IAMCCS_MiniMaxH3LatentUpresSamplingR38": IAMCCS_MiniMaxH3LatentUpresSamplingR38,
    "IAMCCS_MiniMaxH3LatentUpresConditioningR38": IAMCCS_MiniMaxH3LatentUpresConditioningR38,
    "IAMCCS_MiniMaxH3LatentUpresChunkCheckpointR38": IAMCCS_MiniMaxH3LatentUpresChunkCheckpointR38,
    "IAMCCS_MiniMaxH3LatentUpresRTXR38": IAMCCS_MiniMaxH3LatentUpresRTXR38,
    "IAMCCS_MiniMaxH3DeliveryRouterR38": IAMCCS_MiniMaxH3DeliveryRouterR38,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_MiniMaxH3LatentUpresControlR38": "MiniMax H3 R38 Latent Upres Control",
    "IAMCCS_MiniMaxH3LatentUpresSamplingR38": "MiniMax H3 R38 Refine Sampling",
    "IAMCCS_MiniMaxH3LatentUpresConditioningR38": "MiniMax H3 R38 Target Conditioning",
    "IAMCCS_MiniMaxH3LatentUpresChunkCheckpointR38": "MiniMax H3 R38 Upscaled Chunk Accumulator",
    "IAMCCS_MiniMaxH3LatentUpresRTXR38": "MiniMax H3 R38 NVIDIA RTX Delivery",
    "IAMCCS_MiniMaxH3DeliveryRouterR38": "MiniMax H3 R38 Lazy Delivery",
}
