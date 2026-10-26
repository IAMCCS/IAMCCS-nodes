"""Native ComfyUI FastH3 Omni profile; no external inference runtime required."""

import copy

from .iamccs_minimax_h3_shotboard_core import H3_FPS, H3_MAX_TRAINED_FRAMES, H3_MIN_TRAINED_FRAMES, align_h3_frames


def enabled(plan):
    return bool(plan.get("fasth3_omni", {}).get("enabled"))


def configure(plan, settings):
    active = bool(settings.get("h3_omni_enabled", False)) and plan.get("task_mode") == "ref2va"
    plan["fasth3_omni"] = {"enabled": active, "sampling": "native_comfy_pdd32_8_forwards",
                           "reference_order": "images_videos_audio", "joint_target_latent": True}
    if not active:
        return
    sampling = plan["sampling"]
    expected = dict(steps=8, sampler_name="euler", scheduler="simple", denoise=1.0,
                    shift_video=12.0, shift_audio=3.0)
    for key, value in expected.items():
        if sampling.get(key) != value:
            raise ValueError(f"FastH3 Omni requires {key}={value}; select FASTH3 OMNI · 8 STEP in Settings PRO.")
    for key in ("turbo", "pdd", "fasth3", "fused_turbo_preview"):
        if plan.get(key, {}).get("enabled"):
            raise ValueError("FastH3 Omni already contains distillation; disable additional speed adapters.")
    if plan.get("veda", {}).get("enabled"):
        raise ValueError("FastH3 Omni uses its own trained attention gates; disable VEDA for the native baseline.")
    if settings.get("h3_attention_backend") in {"sol_low_vram", "sage_sol"}:
        raise ValueError("The native FastH3 Omni profile supports dense/Sage attention; select its 8-step preset to reset attention.")
    total = max(1, round(float(plan["requested_duration_seconds"]) * H3_FPS))
    if total > H3_MAX_TRAINED_FRAMES:
        configure_longvid(plan, settings, total)
        return
    chunks = plan["chunks"]
    if len(chunks) > 1 or int(chunks[0]["unique_frames"]) != total:
        # Camera slots become prompt intervals in a single target, not separate
        # generation jobs. Reference images remain on the CineInfo media path.
        local = [f"[Shot {i + 1}] At {slot['start_seconds']:.6f}s: {slot['prompt']}"
                 for i, slot in enumerate(plan["slots"]) if slot.get("prompt")]
        prompt = plan.get("global_prompt", "") if len(chunks) > 1 else chunks[0]["prompt"]
        if local:
            instructions = "\n\n" + "\n\n".join(local)
            if "\noverall_soundscape:" in prompt:
                prompt = prompt.replace("\noverall_soundscape:", instructions + "\n\noverall_soundscape:", 1)
            else:
                prompt += instructions
        chunk = copy.deepcopy(chunks[0])
        frames = max(H3_MIN_TRAINED_FRAMES, align_h3_frames(total))
        chunk.update(index=0, frame_count=frames, requested_frame_count=total, unique_frames=total,
                     visible_frame_count=total, duration_seconds=frames / H3_FPS,
                     visible_duration_seconds=total / H3_FPS, timeline_start_frame=0, timeline_start_seconds=0,
                     first_image="", last_image="", prompt=prompt, creative_prompt=prompt,
                     local_prompt="", join_mode="none", trim_head_frames=0, overlap_frames=0)
        plan.update(chunks=[chunk], total_segments=1, total_unique_frames=total,
                    effective_duration_seconds=total / H3_FPS,
                    prompt_map=[{"chunk_index": 0, "prompt": prompt, "duration_seconds": frames / H3_FPS}])
    plan["fasth3_omni"]["authored_camera_slots"] = len(chunks)
    plan["mode_contract"] = {"mode": "fasth3_omni", "joint_av_target": True,
                             "camera_cuts": "model_generated_prompt_intervals", "exact_cut_guarantee": False}


def validate_model(model):
    diffusion = model.model.diffusion_model
    heads = diffusion.final_layer.video_out.weight.shape[0] // diffusion.final_layer.video_dim
    if heads != 32:
        raise ValueError("The Omni loader requires a native FastH3 Omni checkpoint with 32 PDD heads.")


def configure_longvid(plan, settings, total):
    """Reuse native LongVid AV-tail continuation, one joint target per window."""
    tail = 22
    requested_window = int(settings.get("motion_context_window_frames", H3_MAX_TRAINED_FRAMES))
    window = min(H3_MAX_TRAINED_FRAMES, max(H3_MIN_TRAINED_FRAMES, requested_window))
    while window % 17 != 5:
        window -= 1
    template = plan["chunks"][0]
    chunks = []
    start = 0
    while start < total:
        prefix = tail if chunks else 0
        visible = min(total - start, window - prefix)
        sample = max(H3_MIN_TRAINED_FRAMES, align_h3_frames(prefix + visible))
        end = start + visible
        prompt = plan.get("global_prompt", "")
        local = []
        for i, slot in enumerate(plan["slots"]):
            slot_start = float(slot["start_seconds"]) * H3_FPS
            slot_end = slot_start + float(slot.get("duration_seconds", total / H3_FPS)) * H3_FPS
            if slot_start < end and slot_end > start and slot.get("prompt"):
                local.append(f"[Shot {i + 1}] At {max(0, slot_start - start + prefix) / H3_FPS:.6f}s: {slot['prompt']}")
        prompt += (f"\n\nThis target covers global timeline {start / H3_FPS:.6f}–{end / H3_FPS:.6f}s. "
                   f"Its first {prefix / H3_FPS:.6f}s are inherited AV context; continue the same physical action and ambience. "
                   "Use only events belonging to this interval, with local times offset by the inherited context.\n" + "\n".join(local))
        chunk = copy.deepcopy(template)
        chunk.update(index=len(chunks), frame_count=sample, requested_frame_count=visible,
                     unique_frames=visible, visible_frame_count=visible, duration_seconds=sample / H3_FPS,
                     visible_duration_seconds=visible / H3_FPS, timeline_start_frame=start,
                     timeline_start_seconds=start / H3_FPS, first_image="", last_image="",
                     prompt=prompt, creative_prompt=prompt, local_prompt="", join_mode="omni_av_tail",
                     trim_head_frames=prefix, overlap_frames=0, latent_tail_adaptive_window=True,
                     latent_tail_context_prefix_frames=prefix)
        chunks.append(chunk)
        start = end
    plan.update(chunks=chunks, total_segments=len(chunks), total_unique_frames=total,
                effective_duration_seconds=total / H3_FPS,
                longvid_latent_tail={"enabled": True, "tail_frames": tail},
                longvid_latent_tail_junction={"enabled": True},
                prompt_map=[{"chunk_index": c["index"], "prompt": c["prompt"], "duration_seconds": c["duration_seconds"]} for c in chunks])
    plan["fasth3_omni"].update(longvid=True, joint_target_latent=False,
                              joint_latent_per_window=True, window_frames=window, tail_frames=tail,
                              soft_video_frames=4, soft_audio_ms=15.0)
    plan["mode_contract"] = {"mode": "fasth3_omni_longvid", "joint_av_target_per_window": True,
                             "continuity": "sampled_av_tail", "experimental": True,
                             "exact_cut_guarantee": False}


def reference_manifest(images, videos, video_audios, audios):
    """Mirror ReferenceToVideo's audio-before-video, standalone-audio-last order."""
    items = [{"label": f"<Picture {i}>", "role": "ordered_image_reference"}
             for i in range(1, len(images) + 1)]
    audio_index = 0
    for i, name in enumerate(videos or {}, 1):
        soundtrack = (video_audios or {}).get("ref_video_audio_" + name.rsplit("_", 1)[-1])
        if soundtrack is not None:
            audio_index += 1
            items.append({"label": f"<Audio {audio_index}>", "role": f"soundtrack_of_Video_{i}"})
        items.append({"label": f"<Video {i}>", "role": "ordered_video_reference"})
    for _ in (audios or {}).values():
        audio_index += 1
        items.append({"label": f"<Audio {audio_index}>", "role": "standalone_audio_reference"})
    return items
