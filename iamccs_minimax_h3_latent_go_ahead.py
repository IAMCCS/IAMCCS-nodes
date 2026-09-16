# SPDX-License-Identifier: GPL-3.0-or-later
"""IAMCCS LatentGoAhead: past-positioned inference AV history.

Independent ComfyUI implementation. No Wan2GP source or runtime is imported.
Conceptual reference: wan2gp-h3-latent-continue 0.2.2 (saved latent history,
negative temporal coordinates, no re-encoding of generated video).  A
continuation window is phase-aligned exactly like the reference: the visible
window budget is split between past context and newly sampled future frames.
Core H3 PackedLayout supplies the coordinate convention.
"""
from __future__ import annotations
import copy
import hashlib
import json
import logging
import math
import uuid
from pathlib import Path

LOG = logging.getLogger("IAMCCS.H3.LatentGoAhead")


def delivery_audio_settings(audio_join, audio_smoothing_ms):
    """Normalise appended controls from current and legacy workflow schemas."""
    policy = str(audio_join or "click_safe").strip().lower()
    if policy not in ("click_safe", "hard_cut"):
        policy = "click_safe"
    try:
        milliseconds = int(float(str(audio_smoothing_ms).strip()))
    except (TypeError, ValueError):
        milliseconds = 20
    return policy, max(0, min(250, milliseconds))


def latent_fingerprint(latent):
    """Stable audit ID proving which sampled video latent feeds the next pass."""
    import torch
    tensor = latent["samples"].tensors[0].detach().cpu().contiguous()
    raw = tensor.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def frame_fingerprint(frame):
    """Audit ID for the exact decoded frame used as the next Qwen boundary."""
    import torch
    if not torch.is_tensor(frame) or frame.ndim != 4 or int(frame.shape[0]) != 1:
        raise ValueError("LatentGoAhead: invalid decoded boundary frame")
    raw = frame.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def continuation_chunk(
    chunk,
    target_frame_count=None,
    *,
    is_first_chunk=False,
    is_final_chunk=False,
):
    """Use the source tail as Picture 1 and authored destination as Picture 2."""
    from .iamccs_minimax_h3_shotboard_core import (
        _keyframe_alignment_prompt,
    )
    result=copy.deepcopy(chunk)
    creative=result.get('creative_prompt')
    if creative is None:
        raise ValueError('LatentGoAhead requires separately authored creative_prompt; refusing ambiguous legacy prompt')
    alignment=_keyframe_alignment_prompt(
        'fl2va',int(target_frame_count or result['frame_count']),True,True)
    effective_frames = int(target_frame_count or result['frame_count'])
    audio_handoff = (
        '[AUDIO CONTINUITY] Treat the boundary as continuous time. Do not introduce '
        'an unauthored restart, click, or silence gap. Continue or change sound and '
        'dialogue exactly as specified by the current Shotboard.'
        if not is_first_chunk else ''
    )
    result.update(
        first_image='',
        uses_explicit_first_keyframe=False,
        alignment_prompt=alignment,
        audio_handoff_prompt=audio_handoff,
    )
    result['prompt']='\n\n'.join(
        p for p in (alignment,str(creative),audio_handoff) if p
    )
    return result


def history_positions(token_count, delivered_frames, context_frames, frame_pattern):
    if delivered_frames < 1 or context_frames < 1 or not frame_pattern:
        raise ValueError("LatentGoAhead: invalid history clock")
    spans = []
    cursor = 0
    for index in range(token_count):
        end = cursor + frame_pattern[index % len(frame_pattern)]
        if end > max(0, delivered_frames-context_frames) and cursor < delivered_frames:
            spans.append((index, cursor-(delivered_frames-1)))
        cursor = end
    if cursor < delivered_frames or not spans:
        raise ValueError("LatentGoAhead: latent does not cover delivered video")
    return spans


def continuation_window(total_window_frames, context_frames):
    """Return independent history and future budgets.

    Direct latent history is conditioning at negative temporal coordinates; it
    does *not* occupy rows in the newly sampled target.  This is the contract
    used by the reference latent-continuation implementation: a 141-frame
    authored interval remains a 141-frame future while (for example) 35 frames
    of prior motion are supplied separately.  Subtracting history from the
    target shortened every successor and forced its destination keyframe early.
    """
    total = int(total_window_frames)
    context = min(int(context_frames), total)
    if total < 5 or context < 1:
        raise ValueError("LatentGoAhead: invalid continuation window")
    if total % 17 != 5:
        raise ValueError(f"LatentGoAhead: target future {total}f is outside H3's 17k+5 grid")
    history = context
    future = total
    return history, future


def frozen_tail_length(frames, max_frames=6, threshold=0.003):
    """Return only a terminal run of perceptually static decoded frames.

    This is deliberately bounded and operates on the generated future before
    it becomes the next checkpoint boundary.  It never blends frames and it
    never removes an isolated low-motion frame inside the shot.
    """
    import torch
    if not torch.is_tensor(frames) or frames.ndim < 4 or len(frames) < 3:
        return 0
    limit = min(max(0, int(max_frames)), len(frames) - 2)
    if not limit:
        return 0
    # Bound analysis memory even on large canvases: retain a uniform RGB sample.
    stride_h = max(1, (frames.shape[1] + 159) // 160)
    stride_w = max(1, (frames.shape[2] + 159) // 160)
    value = frames[-(limit + 1):, ::stride_h, ::stride_w].detach().float().cpu()
    scale = 255.0 if float(value.max()) > 2.0 else 1.0
    cutoff = max(0.0, float(threshold)) * scale
    count = 0
    for index in range(len(value) - 1, max(0, len(value) - limit - 1), -1):
        delta = float((value[index] - value[index - 1]).abs().mean())
        if delta > cutoff:
            break
        count += 1
    # One quiet frame is ordinary motion cadence, not a frozen tail.
    return count if count >= 2 else 0


def append_history(positive, previous, delivered_frames, context_frames, audio_seconds, target_shape):
    import torch
    import node_helpers
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN
    samples = previous.get("samples")
    if samples is None or not getattr(samples, "is_nested", False) or len(samples.tensors) != 2:
        raise ValueError("LatentGoAhead requires original nested H3 AV samples")
    video, audio = samples.tensors
    if video.ndim != 5 or tuple(video.shape[:2]) != (1,24) or tuple(video.shape[-2:]) != tuple(target_shape):
        raise ValueError("LatentGoAhead: source/target video geometry mismatch")
    if audio.ndim != 4 or audio.shape[0] != 1 or not torch.isfinite(video).all() or not torch.isfinite(audio).all():
        raise ValueError("LatentGoAhead: invalid AV history")
    existing = list(positive[0][1].get("minimax_keyframes", []))
    records = []
    positions = history_positions(video.shape[2], delivered_frames, context_frames, FRAME_PER_TOKEN)
    for index, frame in positions:
        records.append({"resolved_frame_index": frame, "latent": video[:,:,index:index+1].clone()})
    if audio_seconds > 0:
        boundary = (delivered_frames-1)/24
        start = max(0, math.floor((boundary-audio_seconds)*40))
        stop = min(audio.shape[-1], math.ceil(delivered_frames/24*40))
        if stop > start:
            records.append({"resolved_frame_index": (start/40-boundary)*24,
                            "audio_latent": audio[...,start:stop].clone()})
    # Match the reference pipeline: historical blocks are packed first, then
    # the authored destination guide.  Reversing this order changes both the
    # packed presentation and attention row order.
    records.extend(existing)
    LOG.info("LatentGoAhead history | video_blocks=%d | positions=%s | audio=%.2fs | reencode=no | crossfade=0", len(positions), positions, audio_seconds)
    return node_helpers.conditioning_set_values(positive, {"minimax_keyframes": records})


def prepare_continuation(active_clip, video_vae, prompt, width, height, length,
                         source_last_frame, destination):
    """Build reference-equivalent continuation conditioning.

    The Qwen presentation sees the exact decoded source boundary followed by
    the authored destination.  Only the destination becomes a new target
    keyframe; source motion is supplied separately by saved latent history.
    """
    import node_helpers
    from comfy_extras.nodes_minimax_h3 import _empty_av_latent, _resize
    latent, frame_count = _empty_av_latent(width, height, length)
    opener = _resize(source_last_frame[:1], width, height, "disabled")
    final = _resize(destination[:1], width, height, "center")
    tokens = active_clip.tokenize(prompt, images=[opener, final])
    positive = active_clip.encode_from_tokens_scheduled(tokens)
    final_latent = video_vae.encode(final)
    positive = node_helpers.conditioning_set_values(positive, {
        "minimax_keyframes": [{
            "resolved_frame_index": frame_count - 1,
            "latent": final_latent,
        }]
    })
    return positive, latent


class IAMCCS_MiniMaxH3LatentGoAhead:
    @classmethod
    def INPUT_TYPES(cls):
        from .iamccs_minimax_h3_atomic_backend import SUPERNODE_LINX_TYPE
        return {"required": {"model":("MODEL",), "clip":("CLIP",),
            "video_vae":("VAE",), "audio_vae":("VAE",), "cine_linx":(SUPERNODE_LINX_TYPE,),
            "video_context_frames":(["18","35","52"], {"default":"35"}),
            "audio_context_seconds":("FLOAT", {"default":1.0,"min":0.0,"max":2.0,"step":0.5}),
            "freeze_tail_max_frames":("INT", {"default":6,"min":0,"max":72,"step":1}),
            "freeze_tail_threshold":("FLOAT", {"default":0.003,"min":0.0,"max":0.02,"step":0.0005})},
            "optional":{"cut_plan":("STRING", {"default":"{}", "multiline":True}),
                "join_blend":(["none","linear","smoothstep"], {"default":"smoothstep", "tooltip":"Delivery-only AV overlap. Original checkpoints remain untouched. Each join shortens delivery by blend_frames / 24 seconds."}),
                "blend_frames":("INT", {"default":9,"min":1,"max":24}),
                # Kept as tolerant strings because older serialized graphs can
                # place a legacy empty DOM-widget value in these appended slots.
                # The Control Room supplies the constrained choices and render()
                # normalises them before use, avoiding pre-execution INT/combo
                # validation failures during workflow migration.
                "audio_join":("STRING", {"default":"click_safe", "tooltip":"Managed by AHEAD Control Room: click_safe or hard_cut."}),
                "audio_smoothing_ms":("STRING", {"default":"20", "tooltip":"Managed by AHEAD Control Room: 0-250 milliseconds per side."}),
                "flash_guard":("STRING", {"default":"off", "tooltip":"Optional delivery interpolation: off, auto or all_periodic. Can introduce ghosting; does not fix sampling."}),
                "flash_guard_sensitivity":("STRING", {"default":"2.0", "tooltip":"AUTO detection ratio, 1.1-5.0."}),
                "flash_guard_radius":("STRING", {"default":"2", "tooltip":"Frames on each side of a detected periodic flash, 1-6."})},
            "hidden":{"unique_id":"UNIQUE_ID", "prompt":"PROMPT", "extra_pnginfo":"EXTRA_PNGINFO"}}
    RETURN_TYPES=("IMAGE","AUDIO","IMAGE","LATENT","INT","STRING")
    RETURN_NAMES=("frames","audio","bridge","sampled_latent","fps","report")
    # LatentGoAhead owns its interval checkpoints and the assembled master.
    # Its dedicated workflow mutes the legacy universal checkpoint, so this is
    # the sole terminal for the selected branch (and cannot cause a duplicate
    # legacy render).
    OUTPUT_NODE=True
    FUNCTION="render"
    CATEGORY="IAMCCS/MiniMax H3/Experimental Continuity"

    def render(self, model, clip, video_vae, audio_vae, cine_linx, video_context_frames="35", audio_context_seconds=1.0,
               freeze_tail_max_frames=6, freeze_tail_threshold=0.003, unique_id=None, prompt=None, extra_pnginfo=None,
               cut_plan="{}", join_blend="smoothstep", blend_frames=9, audio_join="click_safe", audio_smoothing_ms=20,
               flash_guard="off", flash_guard_sensitivity="2.0", flash_guard_radius="2"):
        import torch
        import folder_paths
        from safetensors.torch import save_file
        requested_audio_join, requested_audio_ms = audio_join, audio_smoothing_ms
        audio_join, audio_smoothing_ms = delivery_audio_settings(audio_join, audio_smoothing_ms)
        flash_guard = str(flash_guard or 'off').strip().lower()
        if flash_guard not in ('off','auto','all_periodic'): flash_guard='off'
        try: flash_guard_sensitivity=max(1.1,min(5.0,float(flash_guard_sensitivity or 2.0)))
        except (TypeError,ValueError): flash_guard_sensitivity=2.0
        try: flash_guard_radius=max(1,min(6,int(float(flash_guard_radius or 2))))
        except (TypeError,ValueError): flash_guard_radius=2
        if (str(requested_audio_join or "").strip().lower() != audio_join
                or str(requested_audio_ms or "").strip() != str(audio_smoothing_ms)):
            LOG.warning(
                "LatentGoAhead migrated delivery audio controls | audio_join=%r -> %s | "
                "audio_smoothing_ms=%r -> %d",
                requested_audio_join, audio_join, requested_audio_ms, audio_smoothing_ms)
        from .iamccs_minimax_h3_atomic_backend import (
            _resolve_shotplan, IAMCCS_MiniMaxH3AtomicConditioningBackend,
            IAMCCS_MiniMaxH3GenerationBackendV2)
        from .iamccs_minimax_h3_motion_context_variant import _replace_plan
        original = _resolve_shotplan(cine_linx)
        if original.get("task_mode") != "latent_go_ahead":
            raise ValueError("Select LatentGoAhead in Shotboard/Settings")
        acceleration = str(original.get("acceleration", "native"))
        if acceleration == "matlowai_fused_turbo_manual_sigma":
            raise ValueError("LatentGoAhead requires the FL2VA model; Fused Fast is T2VA only.")
        if acceleration.startswith("iamccs_progressive_") and float(audio_context_seconds) > 0:
            raise ValueError("Progressive sampling cannot preserve the audio history mask. Choose Native, PDD, FastH3, SLA or a compatible Turbo LoRA; or explicitly set audio context to zero.")
        if original.get("audio_mode") != "h3_native_generated":
            raise ValueError("LatentGoAhead currently continues generated H3 audio only")
        from .iamccs_ahead_cuts import validate
        exact_cuts = validate(cut_plan, original)
        sampling = original.get("sampling", {})
        plan=copy.deepcopy(original)
        plan["task_mode"]="fl2va"
        plan["native_av_continuity"]={"enabled":False}
        chunks=plan["chunks"]
        if not chunks: raise ValueError("LatentGoAhead has no image intervals")
        width,height=int(plan['width']),int(plan['height'])
        if width % 32 >= 16 or height % 32 >= 16:
            raise ValueError(
                f"Unsafe MiniMax H3 keyframe canvas {width}x{height}: local patchify_video can corrupt "
                "condition frames when width or height modulo 32 is at least 16. Choose the adjacent "
                "multiple-of-32 canvas in Shotboard; the workflow will not silently resize authored media.")
        for i, chunk in enumerate(chunks):
            frame_count=int(chunk.get('frame_count',0))
            if frame_count < 5 or (frame_count-5) % 17:
                raise ValueError(
                    f"Chunk {i+1} has {frame_count} frames; MiniMax H3 requires 17*k+5. "
                    "Adjust duration/FPS in Shotboard so the compiler emits a legal H3 window.")
            if not chunk.get("last_image") or (i==0 and not chunk.get("first_image")):
                raise ValueError("LatentGoAhead needs opening image and every destination image")
            if i:
                _history_frames, target_frames = continuation_window(
                    int(chunk['frame_count']), int(video_context_frames))
                chunk=continuation_chunk(
                    chunk,
                    target_frames,
                    is_first_chunk=False,
                    is_final_chunk=i + 1 >= len(chunks),
                )
                chunks[i]=chunk
            chunk.update(uses_bridge_first_frame=False, trim_head_frames=0, overlap_frames=0)
        linx=_replace_plan(cine_linx,plan)
        frames_out=[]; audio_out=[]; previous=None; previous_frames=0; rate=None
        chain_model=None
        previous_last_frame=None
        previous_audio_origin=0.0
        expected_parent_sha=None
        run=Path(folder_paths.get_output_directory())/"IAMCCS"/"LatentGoAhead"/uuid.uuid4().hex
        run.mkdir(parents=True,exist_ok=False)
        # Snapshot queue-time data before any interval can mutate runtime state.
        from .iamccs_ahead_provenance import snapshot, write_sidecar
        provenance = snapshot(prompt, extra_pnginfo, original, unique_id, {
            'video_context_frames':video_context_frames, 'audio_context_seconds':audio_context_seconds,
            'freeze_tail_max_frames':freeze_tail_max_frames, 'freeze_tail_threshold':freeze_tail_threshold,
            'cut_plan':cut_plan, 'join_blend':join_blend, 'blend_frames':blend_frames,
            'audio_join':audio_join, 'audio_smoothing_ms':audio_smoothing_ms,
            'flash_guard':flash_guard, 'flash_guard_sensitivity':flash_guard_sensitivity,
            'flash_guard_radius':flash_guard_radius,
            'effective_cuts':exact_cuts, 'cut_recipe_ignored':bool(cut_plan not in ('{}','') and not exact_cuts)})
        provenance['metadata']['interval_audits'] = []
        provenance['metadata']['flash_guard_audit'] = []
        # Keep inference/history frames untouched, but prepare an independent
        # delivery copy after every interval so the live preview represents the
        # same flash correction that will be used by the final export.
        from .iamccs_ahead_blend import periodic_flash_guard
        delivery_clips=[]; flash_audit=[]
        from server import PromptServer
        def publish(path, label):
            write_sidecar(path, provenance, label, run.name)
            if unique_id is not None:
                PromptServer.instance.send_sync('iamccs-latentgoahead-video',{
                    'node':str(unique_id),'run':run.name,'label':label,
                    'filename':path.name,'subfolder':path.parent.relative_to(Path(folder_paths.get_output_directory())).as_posix(),
                    'type':'output'})
        for i,chunk in enumerate(chunks):
            history_pixel_frames = 0
            future_frame_count = int(chunk['frame_count'])
            if previous is None:
                prepared=IAMCCS_MiniMaxH3AtomicConditioningBackend().prepare(
                    model=model,clip=clip,video_vae=video_vae,audio_vae=audio_vae,cine_linx=linx,segment_index=i)
                active_model, positive, latent=prepared[:3]
                # Freeze one acceleration/model-patch contract for the complete
                # chain.  Falling back to the raw input model on intervals 2+
                # made later slots use different attention/cache/LoRA settings.
                chain_model=active_model
            else:
                from .iamccs_minimax_h3_atomic_backend import (
                    _load_image, _resize_reference_image,
                    _run_h3_conditioning_with_cpu_fallback)
                if not torch.is_tensor(previous_last_frame):
                    raise RuntimeError("LatentGoAhead lost the decoded source boundary")
                destination=_load_image(str(chunk['last_image']))
                if not torch.is_tensor(destination):
                    raise ValueError(f"LatentGoAhead missing destination for interval {i+1}")
                destination,_=_resize_reference_image(destination[:1],plan,'last')
                history_pixel_frames,future_frame_count=continuation_window(
                    int(chunk['frame_count']),int(video_context_frames))
                result,_=_run_h3_conditioning_with_cpu_fallback(clip,plan,
                    lambda active_clip: prepare_continuation(
                        active_clip=active_clip, video_vae=video_vae,
                        prompt=str(chunk.get('prompt','')),
                        width=int(plan['width']), height=int(plan['height']),
                        length=future_frame_count,
                        source_last_frame=previous_last_frame,
                        destination=destination))
                positive,latent=result[0],result[1]
                if chain_model is None:
                    raise RuntimeError('LatentGoAhead model contract was not initialised')
                active_model=chain_model.clone()
            if previous is not None:
                parent_sha = latent_fingerprint(previous)
                if expected_parent_sha is None or parent_sha != expected_parent_sha:
                    raise RuntimeError(
                        "LatentGoAhead chain integrity failure: the next interval did not receive "
                        "the immediately preceding sampled latent"
                    )
                if (
                    str(previous.get("iamccs_latentgoahead_sha256", "")) != parent_sha
                    or int(previous.get("iamccs_latentgoahead_source_interval", -1)) != i - 1
                ):
                    raise RuntimeError(
                        "LatentGoAhead chain provenance failure: cached or non-adjacent history was supplied"
                    )
                positive=append_history(positive,previous,previous_frames,int(video_context_frames),0.0,latent["samples"].tensors[0].shape[-2:])
                LOG.info(
                    "LatentGoAhead parent chain verified | interval=%d/%d | parent_video_sha256=%s | source_interval=%d",
                    i + 1, len(chunks), parent_sha, i,
                )
            else:
                parent_sha = "initial_shotboard_keyframe"
            prefix=None
            audio_origin=0.0
            if previous is not None and float(audio_context_seconds)>0:
                from .iamccs_latentgoahead_audio import prefix_window,make_prefix_target,model_with_audio_origin
                source_audio=previous['samples'].tensors[1]
                start,stop,audio_origin=prefix_window(delivered_frames=previous_frames,fps=24,
                    source_tokens=source_audio.shape[-1],source_origin_seconds=previous_audio_origin,
                    context_seconds=float(audio_context_seconds))
                target,mask,prefix=make_prefix_target(source_audio,first=start,last=stop,
                    origin_seconds=audio_origin,output_seconds=future_frame_count/24)
                video=latent['samples'].tensors[0]
                latent=dict(latent)
                nested=type(latent['samples'])
                latent['samples']=nested((video,target))
                latent['noise_mask']=nested((torch.ones_like(video),mask))
                active_model=model_with_audio_origin(active_model,audio_origin)
            LOG.info(
                "LatentGoAhead CONDITIONING TRUTH | interval=%d/%d | source=%s | destination=%s | prompt=%s",
                i + 1,
                len(chunks),
                parent_sha,
                json.dumps(str(chunk.get("last_image", "")), ensure_ascii=False),
                json.dumps(str(chunk.get("prompt", "")), ensure_ascii=False),
            )
            params={k:sampling.get(k,v) for k,v in dict(seed=42,seed_stride=1,steps=20,
                sampler_name="res_multistep",scheduler="simple",denoise=1.0,shift_video=12.0,shift_audio=3.0).items()}
            frames,audio,bridge,sampled,fps,report=IAMCCS_MiniMaxH3GenerationBackendV2().render(
                model=active_model,positive=positive,latent=latent,video_vae=video_vae,audio_vae=audio_vae,
                cine_linx=linx,chunk_index=i,**params)
            if fps!=24: raise ValueError("LatentGoAhead requires H3 24fps")
            # Preserve inspectable output even if the AV invariant fails.
            from .iamccs_minimax_h3_shotboard import _encode_images
            _encode_images(frames, None, fps, run/f"interval_{i+1:04d}_video_review.mp4")
            publish(run/f"interval_{i+1:04d}_video_review.mp4",f"Interval {i+1} · visual review (silent)")
            LOG.info("LatentGoAhead video review saved | interval=%d | %s",i+1,run/f"interval_{i+1:04d}_video_review.mp4")
            if prefix is not None:
                from .iamccs_latentgoahead_audio import assert_prefix_unchanged,trim_decoded_audio
                actual=sampled['samples'].tensors[1][...,:prefix.shape[-1]].detach().cpu()
                expected=prefix.detach().cpu()
                delta=(actual.float()-expected.float()).abs()
                LOG.info("LatentGoAhead prefix audit | max_abs=%g | mean_abs=%g | expected_dtype=%s | actual_dtype=%s",
                    delta.max().item(),delta.mean().item(),expected.dtype,actual.dtype)
                try:
                    assert_prefix_unchanged(sampled['samples'].tensors[1],prefix)
                except Exception:
                    save_file({'expected_prefix':expected.contiguous(),'actual_prefix':actual.contiguous()},
                        str(run/f"interval_{i+1:04d}_prefix_failure.safetensors"))
                    raise
                # Restore exact source values after the validated scaling
                # round trip, then decode past+future together again.
                from comfy_extras.nodes_audio import VAEDecodeAudio
                restored=sampled['samples'].tensors[1].clone()
                restored[...,:prefix.shape[-1]]=prefix.to(restored)
                sampled=dict(sampled)
                sampled['samples']=type(sampled['samples'])((sampled['samples'].tensors[0],restored))
                audio=VAEDecodeAudio.execute(vae=audio_vae,samples=sampled)[0]
                audio=dict(audio)
                audio['waveform']=trim_decoded_audio(audio['waveform'],origin_seconds=audio_origin,
                    output_seconds=len(frames)/24,sample_rate=audio['sample_rate'])
            # The editorial Cut Finder needs the complete generated ending,
            # including any terminal freeze, with its matching audio.  The
            # delivered interval below remains the trimmed programme master.
            raw_review=run/f"interval_{i+1:04d}_raw_av_review.mp4"
            _encode_images(frames,{'waveform':audio['waveform'],'sample_rate':audio['sample_rate']},fps,raw_review)
            publish(raw_review,f"Interval {i+1} · raw AV ending for Cut Finder")
            tail_trim=frozen_tail_length(
                frames,max_frames=int(freeze_tail_max_frames),threshold=float(freeze_tail_threshold))
            tail_trim_source='manual' if i in exact_cuts else 'freeze_aware'
            if i in exact_cuts:
                if exact_cuts[i] > len(frames):
                    raise ValueError("Exact cut exceeds decoded frame count")
                tail_trim = len(frames) - exact_cuts[i]
            if tail_trim:
                LOG.info(
                    "LatentGoAhead freeze-tail trim | interval=%d/%d | removed=%df | threshold=%g | blend=off",
                    i+1,len(chunks),tail_trim,float(freeze_tail_threshold))
                frames=frames[:-tail_trim]
                keep_samples=round(len(frames)*audio['sample_rate']/24)
                audio=dict(audio)
                audio['waveform']=audio['waveform'][...,:keep_samples]
            provenance['metadata']['interval_audits'].append({'interval':i+1,'sampled_future_frames':future_frame_count,'delivered_frames':len(frames),'trim_frames':tail_trim,'trim_source':tail_trim_source,'effective_prompt':chunk.get('prompt'),'creative_prompt':chunk.get('creative_prompt')})
            previous_audio_origin=audio_origin
            previous_frames=len(frames)
            previous_last_frame=frames[-1:].detach().cpu().contiguous().clone()
            boundary_sha=frame_fingerprint(previous_last_frame)
            # Keep exact inference tensors. These files are diagnostic checkpoints,
            # not cross-session resume contracts; no automatic disk reload occurs.
            v,a=(t.detach().cpu().contiguous().clone() for t in sampled["samples"].tensors)
            previous={
                "samples":type(sampled["samples"])((v,a)),
                "iamccs_latentgoahead_source_interval":i,
            }
            current_sha=latent_fingerprint(previous)
            previous["iamccs_latentgoahead_sha256"]=current_sha
            expected_parent_sha=current_sha
            save_file({"video":v,"audio":a,"last_frame":previous_last_frame},str(run/f"interval_{i+1:04d}.safetensors"),
                metadata={"format":"iamccs.latentgoahead.v3","audio_origin_seconds":str(audio_origin),
                    "source_sha256":original.get("authored_source",{}).get("sha256",""),
                    "parent_video_sha256":parent_sha,"video_sha256":current_sha,
                    "source_interval":str(i),"boundary_frame_sha256":boundary_sha,
                    "freeze_tail_trim_frames":str(tail_trim),
                    "freeze_tail_trim_source":tail_trim_source,
                    "frames":str(previous_frames)})
            # Reference-equivalent assembly: historical pixels occupy the
            # negative-time context window and are not part of ``frames``.
            # Every decoded target frame is therefore new future material.
            # Dropping frame zero here caused a one-frame motion discontinuity
            # even though the latent parent SHA was correct.
            trim=0
            frames_out.append(frames[trim:].detach().cpu())
            if rate is not None and rate!=audio["sample_rate"]: raise ValueError("Audio sample rate changed")
            rate=audio["sample_rate"]
            audio_out.append(audio["waveform"][...,round(trim*rate/24):].detach().cpu())
            _encode_images(frames_out[-1],{'waveform':audio_out[-1],'sample_rate':rate},fps,
                run/f"interval_{i+1:04d}_delivered.mp4")
            publish(run/f"interval_{i+1:04d}_delivered.mp4",f"Interval {i+1} · validated AV")
            guarded, interval_flash_audit = periodic_flash_guard(
                [frames_out[-1]], flash_guard, flash_guard_sensitivity, flash_guard_radius)
            for item in interval_flash_audit:
                item['interval']=i+1
            delivery_clips.append(guarded[0])
            flash_audit.extend(interval_flash_audit)
            provenance['metadata']['flash_guard_audit'] = list(flash_audit)
            if flash_guard != 'off':
                guarded_path=run/f"interval_{i+1:04d}_flash_guarded_preview.mp4"
                _encode_images(guarded[0],{'waveform':audio_out[-1],'sample_rate':rate},fps,guarded_path)
                publish(guarded_path,
                    f"Interval {i+1} · FLASH GUARDED preview · {len(interval_flash_audit)} repair(s)")
            LOG.info(
                "LatentGoAhead interval=%d/%d | window=%df | history=%df | sampled_future=%df | "
                "freeze_tail_trim=%df | duplicate trim=%df | video_sha256=%s | boundary_sha256=%s | checkpoint=%s",
                i+1,len(chunks),int(chunk['frame_count']),history_pixel_frames,
                future_frame_count,tail_trim,trim,current_sha,boundary_sha,run)
        final_frames=torch.cat(frames_out)
        final_audio={"waveform":torch.cat(audio_out,dim=-1),"sample_rate":rate}
        _encode_images(final_frames,final_audio,24,run/'final_film.mp4')
        provenance['metadata']['flash_guard_audit'] = list(flash_audit)
        publish(run/'final_film.mp4','RAW MASTER · original AV joins')
        delivery = run/'final_film.mp4'
        if flash_audit or join_blend != 'none' or (audio_join == 'click_safe' and int(audio_smoothing_ms) > 0):
            from .iamccs_ahead_blend import assemble
            final_frames, waveform = assemble(
                delivery_clips, audio_out, rate, 24, join_blend, blend_frames,
                audio_join=audio_join, audio_smoothing_ms=audio_smoothing_ms)
            final_audio = {"waveform":waveform,"sample_rate":rate}
            delivery = run/('final_blended.mp4' if join_blend != 'none' else ('final_flash_guarded.mp4' if flash_audit else 'final_audio_click_safe.mp4'))
            _encode_images(final_frames,final_audio,24,delivery)
        publish(delivery,'FINAL · '+str(join_blend)+' video · '+str(audio_join)+' audio · flash guard '+str(flash_guard))
        return final_frames,final_audio,previous_last_frame,sampled,24,f"LatentGoAhead | {len(chunks)} intervals | checkpoints={run} | final={delivery} | original latent history; delivery blend={join_blend}, overlap={blend_frames if join_blend != 'none' else 0}f; audio={audio_join}/{audio_smoothing_ms}ms; flash_guard={flash_guard}/{len(flash_audit)} repairs; experimental"

class IAMCCS_MiniMaxH3LatentGoAheadBranch(IAMCCS_MiniMaxH3LatentGoAhead):
    """Universal graphs execute this branch only through the lazy selector."""
    OUTPUT_NODE = False

NODE_CLASS_MAPPINGS={"IAMCCS_MiniMaxH3LatentGoAhead":IAMCCS_MiniMaxH3LatentGoAhead,
    "IAMCCS_MiniMaxH3LatentGoAheadBranch":IAMCCS_MiniMaxH3LatentGoAheadBranch}
NODE_DISPLAY_NAME_MAPPINGS={"IAMCCS_MiniMaxH3LatentGoAhead":"IAMCCS H3 · LatentGoAhead · Native AV History", "IAMCCS_MiniMaxH3LatentGoAheadBranch":"IAMCCS H3 · LatentGoAhead · Universal Branch"}
