# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""R46 restored Long Keyframe Continuous: sampled native AV latent timeline.

This is NOT FL2VA pairwise destination generation. It is a strict adapter
around the historical positioned-guide planner and native R38B compositor.
A take advances on one editorial clock, carrying the predecessor AV tail.
"""
from __future__ import annotations

ENGINES = {
    "extended_av_continuous": {
        "engine": "extended_av_continuous", "backend_family": "extended_av",
        "terminal_endpoint_mode": "latent_free_closure",
        "description": "Real masked/pinned AV continuation, own 39f native grid; trimmed technical prefix",
        "experimental": True,
    },
    "pianosequenza_evolving": {
        "engine": "pianosequenza_evolving", "backend_family": "native_tail",
        "terminal_endpoint_mode": "latent_free_closure",
        "description": "Native TimelineDirector with timed Evolving prompts/AudioCon",
        "experimental": True,
    },
    "pianosequenza_1_native_tail": {
        "engine": "pianosequenza_1_native_tail", "backend_family": "native_tail",
        "terminal_endpoint_mode": "latent_free_closure",
        "description": "Native AV sampled tail / TimelineDirector / incoming owns overlap / LATENT FREE",
        "experimental": False,
    },
    "pianosequenza_2_drift": {
        "engine": "pianosequenza_2_drift", "backend_family": "native_tail",
        "terminal_endpoint_mode": "pianosequenza_drift",
        "description": "Same TimelineDirector AV junction / sigma-matched DRIFT terminal experiment",
        "experimental": True,
    },
    "pianosequenza_3_phase": {
        "engine": "pianosequenza_3_phase", "backend_family": "native_tail",
        "terminal_endpoint_mode": "pianosequenza_phase",
        "description": "Same TimelineDirector AV junction / phase-matched terminal experiment",
        "experimental": True,
    },
}


def resolve_continuous_engine(name: str) -> dict:
    key = str(name or "pianosequenza_1_native_tail").strip().lower()
    if key not in ENGINES:
        raise ValueError(f"Unknown Long Keyframe engine {key!r}; available: {', '.join(ENGINES)}")
    return dict(ENGINES[key])


def incoming_overlap_geometry(unique_frames: int, index: int, overlap: int = 22) -> dict:
    """Make the editorial contract testable independently of a GPU."""
    unique = max(1, int(unique_frames))
    prefix = int(overlap) if int(index) > 0 else 0
    bridge = 1 if int(index) > 0 else 0
    visible = prefix + unique
    required_decoded = visible + bridge
    technical = max(5, required_decoded)
    while technical % 17 != 5:
        technical += 1
    return {
        "unique_frames": unique, "native_incoming_overlap": prefix,
        "rgb_bridge_frames": bridge, "required_decoded_frames": required_decoded,
        "delivered_segment_frames": visible, "technical_sample_frames": technical,
        "editorial_endpoint_frames": visible,
    }


def master_geometry(saved_segments: list[int], overlap: int = 22) -> dict:
    if not saved_segments or any(int(n) <= 0 for n in saved_segments):
        raise ValueError("Master needs positive saved segment frame counts")
    counts = [int(n) for n in saved_segments]
    tail = max(0, int(overlap))
    if any(n <= tail for n in counts[:-1]):
        raise ValueError("A non-final segment is shorter than the incoming-owned overlap")
    kept = [n - tail if i < len(counts) - 1 else n for i, n in enumerate(counts)]
    return {"saved_frames": counts, "kept_frames": kept, "total_frames": sum(kept),
            "fps": 24, "duration_seconds": sum(kept) / 24.0}


def annotate_restored_plan(plan: dict, engine: dict, overlap_frames: int = 22) -> dict:
    if not isinstance(plan, dict) or str(plan.get("task_mode", "")).lower() != "longvid_guides":
        raise ValueError("TimelineDirector may only wrap the native longvid_guides planner")
    chunks = plan.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("TimelineDirector planner returned no native chunks")
    if engine.get("backend_family") != "native_tail":
        raise ValueError("Extended AV must use its own planner and cannot be annotated as native TimelineDirector")
    tail = int(overlap_frames)
    if tail not in {22, 39, 56}:
        raise ValueError("TimelineDirector native overlap must be 22, 39, or 56 H3-compatible frames")
    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or not chunk.get("timeline_director_native"):
            raise RuntimeError(f"TimelineDirector chunk {i+1} did not inherit native planning")
        geom = incoming_overlap_geometry(chunk["unique_frames"], i, tail)
        if int(chunk.get("frame_count", -1)) != geom["technical_sample_frames"]:
            raise RuntimeError(f"TimelineDirector chunk {i+1} sampled-frame geometry mismatch: {chunk.get('frame_count')} vs {geom['technical_sample_frames']}")
        if i > 0 and not bool(chunk.get("uses_bridge_first_frame")):
            raise RuntimeError(f"TimelineDirector chunk {i+1} lost its single technical RGB bridge")
        chunk["timeline_director_contract"] = geom
    plan["timeline_director_native"] = True
    plan["longvid_engine"] = str(engine["engine"])
    plan["requested_task_mode"] = "longvid_keyframe_continuous"
    plan["backend_revision"] = "R46-TimelineDirector-Phase2-v4-AudioCon"
    plan["backend_variant"] = "iamccs_native_sampled_latent_tail_timeline_director"
    plan["mode_contract"] = {
        "mode": "longvid_keyframe_continuous",
        "public_name": "LONG KEYFRAME CONTINUOUS",
        "backend": "iamccs_native_sampled_latent_tail_timeline_director",
        "selected_engine": engine["engine"],
        "experimental": bool(engine["experimental"]),
        "sampled_native_video_audio_tail": True,
        "editorial_endpoint_handoff": True,
        "incoming_owns_overlap": True,
        "overlap_frames": tail,
        "rgb_bridge_remove_frames": 1,
        "terminal_endpoint_mode": engine["terminal_endpoint_mode"],
        "fl2va_pairwise_destination_backend": False,
        "delivery": "native_only_before_whole_sequence_upscale",
        "requires": "ComfyUI H3 native latent + compatible per-stream mask runtime",
    }
    plan["continuation_mode"] = "native_av_latent_tail_editorial_handoff"
    return plan


def apply_evolving_native_conditioning(plan: dict, timeline: dict, global_prompt: str) -> dict:
    """Use the real timed-conditioning resolver on native TimelineDirector chunks.

    Evolving is opt-in and MUST NOT silently run as plain native-tail. The
    authored Extended Prompt blocks own timeline beats; an existing H3 schedule
    is supported if the board contains no blocks. Video/Audio tail path remains
    the native TimelineDirector route, NOT masked Extended AV.
    """
    import importlib.util
    from pathlib import Path
    try:
        from . import iamccs_minimax_h3_conditioning_schedule as cond
    except ImportError:
        path = Path(__file__).with_name("iamccs_minimax_h3_conditioning_schedule.py")
        spec = importlib.util.spec_from_file_location("iamccs_native_evolving_cond", path)
        cond = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cond)
    blocks = timeline.get("extended_prompt_blocks", timeline.get("extendedPromptBlocks", []))
    schedule = cond.validate_schedule(timeline.get("pan_h3_conditioning_v1"))
    if isinstance(blocks, list) and blocks:
        beats = []
        for i, block in enumerate(blocks):
            if not isinstance(block, dict) or block.get("enabled", True) is False:
                continue
            start = int(block.get("start_frame", block.get("start", 0)) or 0)
            end = int(block.get("end_frame", start + int(block.get("length", 1) or 1)) or start + 1)
            prompt = str(block.get("prompt", block.get("action", "")) or "").strip()
            if prompt and end > start:
                beats.append({"id": str(block.get("id") or f"beat_{i+1}"), "start_frame": start,
                              "end_frame": end, "action": prompt})
        schedule = cond.validate_schedule({"schema": cond.SCHEMA, "policy": "evolving",
                    "global_context": str(global_prompt or ""), "beats": beats,
                    "reference_calls": (schedule or {}).get("reference_calls", []),
                    "source": "shotboard_extended_prompt_blocks"})
    if not schedule or schedule.get("policy") != "evolving" or not schedule.get("beats"):
        raise ValueError("Pianosequenza EVOLVING requires timed Extended Prompt blocks or a valid pan_h3_conditioning_v1 evolving schedule with beats. It will not silently run as plain Native Tail.")
    schedule["global_context"] = str(global_prompt or "")
    for chunk in plan["chunks"]:
        start = int(chunk["timeline_start_frame"])
        visible = int(chunk["unique_frames"])
        prefix = int(chunk.get("latent_tail_context_prefix_frames", 0) or 0)
        resolved = cond.resolve_chunk_conditioning(schedule, start_frame=start,
                    visible_frames=visible, context_prefix_frames=prefix, fps=24)
        if resolved:
            # Preserve independently authored guide prompts as well.
            former = str(chunk.get("prompt", "") or "").strip()
            timed = str(resolved["local_prompt"] or "").strip()
            chunk["prompt"] = (former + "\n\n[EVOLVING TIMED BEATS]\n" + timed) if timed else former
            chunk["creative_prompt"] = chunk["prompt"]
            chunk["conditioning_beat_bindings"] = resolved["beat_bindings"]
            chunk["conditioning_reference_calls"] = resolved["reference_calls"]
            chunk["conditioning_revision"] = resolved["conditioning_revision"]
    plan["conditioning_schedule"] = {"schema": cond.SCHEMA, "policy": "evolving",
                    "beat_count": len(schedule["beats"]), "backend": "native_timeline_director"}
    plan["mode_contract"]["evolving_conditioning"] = True
    for prompt in plan.get("prompt_map", []):
        index = int(prompt.get("chunk_index", -1))
        if 0 <= index < len(plan["chunks"]):
            prompt["prompt"] = plan["chunks"][index]["prompt"]
    return plan
