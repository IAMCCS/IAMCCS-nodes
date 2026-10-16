"""Full-frame Ref2VA character replacement, isolated from the SAM3 crop route."""
from __future__ import annotations

MODE = "v2va_face_swap_lora"
SWAP_LORA = "h3_character_swap_pro4500_1000.safetensors"
TURBO_LORA = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
REALISM_LORA = "h3-realism-people-t2v-i2v-r2v.safetensors"


def settings_schema():
    import folder_paths

    names = ["", *folder_paths.get_filename_list("loras")]
    return {
        "h3_swap_lora_name": (names, {"default": SWAP_LORA if SWAP_LORA in names else "", "display_name": "SWAP -LorA · CHARACTER LORA"}),
        "h3_swap_lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
        "h3_swap_realism_lora_name": (names, {"default": REALISM_LORA if REALISM_LORA in names else "", "display_name": "SWAP -LorA · REALISM LORA"}),
        "h3_swap_realism_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
        "h3_swap_refine_seed": ("INT", {"default": 777, "min": 0, "max": 0xffffffffffffffff}),
    }


def swap_plan(settings, task_mode):
    return {
        "enabled": task_mode == MODE,
        "lora_name": settings.get("h3_swap_lora_name", SWAP_LORA),
        "strength": float(settings.get("h3_swap_lora_strength", 1.0)),
        "realism_lora_name": settings.get("h3_swap_realism_lora_name", REALISM_LORA),
        "realism_strength": float(settings.get("h3_swap_realism_strength", 1.0)),
        "refine_seed": int(settings.get("h3_swap_refine_seed", 777)),
    }


def apply_loras(model, plan):
    config = plan.get("character_swap_lora", {})
    if not config.get("enabled"):
        return model
    import folder_paths
    import nodes

    for name, strength in (
        (config.get("lora_name", SWAP_LORA), float(config.get("strength", 1.0))),
        (config.get("realism_lora_name", REALISM_LORA), float(config.get("realism_strength", 1.0))),
    ):
        if not strength:
            continue
        if not name or not folder_paths.get_full_path("loras", name):
            raise ValueError(f"SUBJECT SWAP -LorA requires the selected LoRA: {name or 'not selected'}")
        model = nodes.LoraLoaderModelOnly().load_lora_model_only(model, name, strength)[0]
    return model


def reference_media(cine_linx, plan, chunk, segment_index, frames):
    from .iamccs_h3_face_swap import FACE_SWAP_RESOURCE
    from .iamccs_minimax_h3_v2v_backend import (
        _segment_source_start, _frame_indices, _select_frames, _fit_frames, _slice_audio,
    )

    source = cine_linx.get("resources", {}).get(FACE_SWAP_RESOURCE)
    if not source:
        raise ValueError("SUBJECT SWAP -LorA needs source_video and reference_face on H3 Swap Input.")
    config = plan.get("v2v_settings", {})
    start = _segment_source_start(plan, chunk, segment_index,
        config.get("source_range_policy", "timeline_segment"), float(config.get("source_offset_seconds", 0)))
    requested = int(chunk.get("requested_frame_count", frames))
    indices, _ = _frame_indices(source_frames=len(source["video"]), source_fps=source["fps"],
        start_seconds=start, requested_frames=requested, aligned_frames=frames,
        end_policy=config.get("source_end_policy", "hold_last_for_grid"))
    video = _fit_frames(_select_frames(source["video"], indices, "character_swap_source"),
        int(plan["width"]), int(plan["height"]), config.get("source_fit", "canvas_pad"))
    audio = _slice_audio(source.get("audio"), start_seconds=start, requested_frames=requested, aligned_frames=frames)
    return [source["reference"][:1], None, None, None], video, audio


def restore_source_audio(images, audio, cine_linx, plan, chunk, segment_index):
    from .iamccs_h3_face_swap import FACE_SWAP_RESOURCE
    from .iamccs_minimax_h3_v2v_backend import _slice_audio, _segment_source_start

    requested = int(chunk.get("requested_frame_count", len(images)))
    source = cine_linx.get("resources", {}).get(FACE_SWAP_RESOURCE, {})
    config = plan.get("v2v_settings", {})
    start = _segment_source_start(plan, chunk, segment_index,
        config.get("source_range_policy", "timeline_segment"), float(config.get("source_offset_seconds", 0)))
    original = _slice_audio(source.get("audio"), start_seconds=start, requested_frames=requested, aligned_frames=requested)
    return images[:requested], original if original is not None else audio
