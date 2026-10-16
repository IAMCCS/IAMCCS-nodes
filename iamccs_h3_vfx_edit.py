"""Ref2VA VFX editing with aligned native AddGuide conditioning."""
from __future__ import annotations

import json
import os

import torch

from .iamccs_h3_edit_media import load_edit_video, select_video

MODE = "ref2va_vfx_edit"
LORA = "minimax_h3_vfx_edit_v1.0_r128.safetensors"
RESOURCE = "iamccs_h3_vfx_source"


def settings_schema():
    import folder_paths
    names = ["", *folder_paths.get_filename_list("loras")]
    return {"h3_vfx_lora_name": (names, {"default": LORA if LORA in names else "", "display_name": "VFX EDIT · LORA"}),
            "h3_vfx_lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            "h3_edit_video_source": (["auto", "prompter", "backend"], {"default": "auto", "display_name": "EDIT VIDEO · SOURCE"})}


class IAMCCS_H3VFXInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"cine_linx": ("IAMCCS_SUPERNODE_LINX",),
                             "video_path": ("STRING", {"default": "", "vhs_path_extensions": ["mp4", "mov", "mkv", "webm"]}),
                             "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0}),
                             "duration_seconds": ("FLOAT", {"default": 0.0, "min": 0.0})},
                "optional": {"source_video": ("IMAGE", {"lazy": True}),
                             "source_audio": ("AUDIO", {"lazy": True}),
                             "source_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0})}}

    RETURN_TYPES = ("IAMCCS_SUPERNODE_LINX",)
    RETURN_NAMES = ("cine_linx",)
    FUNCTION = "attach"
    CATEGORY = "IAMCCS/MiniMax H3/VFX Edit"

    @classmethod
    def IS_CHANGED(cls, video_path="", **kwargs):
        return os.path.getmtime(video_path) if os.path.isfile(video_path) else video_path

    def check_lazy_status(self, cine_linx, **kwargs):
        from .iamccs_minimax_h3_atomic_backend import _resolve_shotplan
        plan = _resolve_shotplan(cine_linx)
        video = cine_linx.get("resources", {}).get("iamccs_h3_edit_video")
        policy = plan.get("vfx_edit", {}).get("video_source", "auto")
        if plan.get("task_mode") != MODE or policy == "prompter" or (policy == "auto" and video):
            return []
        return [name for name in ("source_video", "source_audio") if name in kwargs and kwargs[name] is None]

    def attach(self, cine_linx, video_path, start_seconds=0.0, duration_seconds=0.0,
               source_video=None, source_audio=None, source_fps=24.0):
        from .iamccs_minimax_h3_atomic_backend import _resolve_shotplan
        plan = _resolve_shotplan(cine_linx)
        if plan.get("task_mode") != MODE:
            return (cine_linx,)
        config = plan.get("vfx_edit", {})
        video = cine_linx.get("resources", {}).get("iamccs_h3_edit_video")
        policy = config.get("video_source", "auto")
        if torch.is_tensor(source_video) and policy != "prompter" and not (policy == "auto" and video):
            from .iamccs_minimax_h3_v2v_backend import _frame_indices, _select_frames, _slice_audio, _materialize_optional_audio
            available = len(source_video) / float(source_fps) - float(start_seconds)
            duration = float(duration_seconds or plan.get("effective_duration_seconds", available))
            count = max(1, round(min(duration, available) * 24))
            if available <= 0:
                raise ValueError("VFX uploaded video trim is empty.")
            indices, _ = _frame_indices(source_frames=len(source_video), source_fps=float(source_fps),
                start_seconds=float(start_seconds), requested_frames=count, aligned_frames=count, end_policy="error")
            source = {"video": _select_frames(source_video, indices, "vfx_uploaded_source"),
                      "audio": _slice_audio(_materialize_optional_audio(source_audio), start_seconds=float(start_seconds), requested_frames=count, aligned_frames=count),
                      "fps": 24.0, "descriptor": {"path": "uploaded_video_frames", "start_seconds": float(start_seconds)}}
        else:
            selected = select_video(video, {"path": video_path, "start_seconds": start_seconds,
                                          "duration_seconds": duration_seconds}, policy)
            source = load_edit_video(selected, plan["width"], plan["height"], plan.get("duration_seconds", plan.get("effective_duration_seconds", 0)))
        out = dict(cine_linx)
        out["resources"] = {**cine_linx.get("resources", {}), RESOURCE: source}
        return (out,)


def apply_lora(model, plan):
    if plan.get("task_mode") != MODE:
        return model
    import folder_paths
    import nodes
    config = plan.get("vfx_edit", {})
    name = config.get("lora_name", LORA)
    if not name or not folder_paths.get_full_path("loras", name):
        raise ValueError("VFX Edit requires its selected LoRA: " + str(name))
    return nodes.LoraLoaderModelOnly().load_lora_model_only(model, name, float(config.get("strength", 1)))[0]


def prepare(model, clip, video_vae, audio_vae, cine_linx, segment_index, prompt_override=""):
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo, MiniMaxH3AddGuide
    from .iamccs_minimax_h3_atomic_backend import _resolve_shotplan, _chunk, _run_h3_conditioning_with_cpu_fallback
    from .iamccs_minimax_h3_v2v_backend import _frame_indices, _select_frames, _fit_frames, _slice_audio
    plan = _resolve_shotplan(cine_linx)
    chunk = _chunk(plan, segment_index)
    source = cine_linx.get("resources", {}).get(RESOURCE)
    if not source:
        raise ValueError("VFX Edit needs the dedicated H3 VFX Input branch connected.")
    prompt = str(prompt_override or chunk.get("prompt", "")).strip()
    if not prompt.lower().startswith("vfx_edit:"):
        raise ValueError("VFX Edit Global must start with 'vfx_edit:' and describe the requested edit.")
    frames = int(chunk["frame_count"])
    requested = int(chunk.get("requested_frame_count", frames))
    start = float(chunk.get("timeline_start_frame", 0)) / 24
    indices, _ = _frame_indices(source_frames=len(source["video"]), source_fps=24,
        start_seconds=start, requested_frames=requested, aligned_frames=frames, end_policy="hold_last_for_grid")
    guide = _fit_frames(_select_frames(source["video"], indices, "vfx_source"), plan["width"], plan["height"], "canvas_pad")
    result, encoder_report = _run_h3_conditioning_with_cpu_fallback(clip, plan,
        lambda active_clip: MiniMaxH3ReferenceToVideo.execute(clip=active_clip, vae=video_vae,
            audio_vae=audio_vae, prompt=prompt, width=plan["width"], height=plan["height"], length=frames))
    positive, latent = result[0], result[1]
    positive = MiniMaxH3AddGuide.execute(positive=positive, latent=latent, vae=video_vae, image=guide, frame_idx=0)[0]
    report = f"VFX Edit | Ref2VA | aligned AddGuide={frames} frames | source={source['descriptor']['path']} | {encoder_report}"
    return (model, positive, latent, guide[:1], guide[-1:], json.dumps({"task": "ref2va", "guide_frames": frames}),
            prompt, int(segment_index), len(plan["chunks"]), 0, report, None)


def restore_audio(images, audio, cine_linx, chunk):
    from .iamccs_minimax_h3_v2v_backend import _slice_audio
    source = cine_linx.get("resources", {}).get(RESOURCE, {})
    requested = int(chunk.get("requested_frame_count", len(images)))
    original = _slice_audio(source.get("audio"), start_seconds=float(chunk.get("timeline_start_frame", 0)) / 24,
                            requested_frames=requested, aligned_frames=requested)
    return images[:requested], original if original is not None else audio


NODE_CLASS_MAPPINGS = {"IAMCCS_H3VFXInput": IAMCCS_H3VFXInput}
NODE_DISPLAY_NAME_MAPPINGS = {"IAMCCS_H3VFXInput": "IAMCCS H3 VFX Edit · Video Source"}
