import json
import logging
import math
import os

import torch
from safetensors import safe_open
from safetensors.torch import load_file as st_load, save_file as st_save

import comfy.model_management
import comfy.nested_tensor
import comfy.utils
import folder_paths
import node_helpers
import nodes

from .latent_math import (
    FPS, AUDIO_HZ, FRAME_RESCALE, CONTEXT_TO_STEPS,
    temporal_shape, pixel_frames, context_slice, phase_aware_context_slice, phase_aligned_extended_context_slice, audio_slice_for_pixel_window,
)
from .patch_layout import HC_INDEX, HC_AUDIO_END_FRAME
from .runtime_patches import ensure_h3_runtime_patches
from .motion_analysis import analyze_freeze_tail, phase_aware_safety_from_confidence
from .release_utils import (
    duration_to_requested_frames, normalize_alignment_mode, normalize_safety_mode,
    resolve_freeze_settings, stitch_trim_plan, apply_no_lock_fallback,
)
from .seamless_stitch import (
    LUMINANCE_ANALYSIS_FRAMES, context_aligned_video_join, context_aligned_audio_join,
    frame_trimmed_audio, resolve_saved_head_context, fit_audio_length,
    blend_video_overlap, blend_audio_overlap, estimate_luminance_gain,
    apply_rgb_gain, apply_luminance_gain_fade, safe_tail_bridge_plan,
    extract_safe_tail_bridge_images,
)

_LOG = logging.getLogger("h3_continuous")
CANVAS_MULTIPLE = 32
REF_IMAGE_SHORT_EDGE = 2048


def _resize(image, width, height, crop):
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _empty_av_latent(width, height, length, batch_size=1):
    frame_count, vt, at = temporal_shape(length)
    dev = comfy.model_management.intermediate_device()
    video = torch.zeros([batch_size, 24, vt, height // 16, width // 16], device=dev)
    audio = torch.zeros([batch_size, 32, 2, at], device=dev)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


def _streams_from_latent(latent):
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("h3_continuous: expected a LATENT dict with 'samples'")
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        raise ValueError(f"h3_continuous: expected H3 AV nested latent, got {type(samples)!r}")
    if len(parts) < 2:
        raise ValueError("h3_continuous: H3 latent must contain both video and audio streams")
    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if video.ndim != 5 or audio.ndim != 4:
        raise ValueError(
            f"h3_continuous: unexpected H3 shapes video={tuple(video.shape)}, audio={tuple(audio.shape)}"
        )
    return video, audio


def _prepare_qwen_reference_image(image, width, height, mode):
    """Resize an optional <Picture 1> for Qwen only.

    v0.4 deliberately does NOT VAE-encode this image and does NOT put it in
    ``minimax_refs``. This preserves the previously working production
    FFLF+Qwen-reference behavior instead of turning Picture 1 into a persistent
    DiT reference latent.
    """
    h, w = int(image.shape[1]), int(image.shape[2])
    if mode == "match":
        scale = min(1.0, math.sqrt((width * height) / float(w * h)))
    else:
        scale = min(1.0, REF_IMAGE_SHORT_EDGE / float(min(w, h)))
    tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return _resize(image[:1], tw, th, "disabled")


def _require_patches():
    # v1.1.4: importing/installing the node pack must not alter ComfyUI's H3
    # runtime. Install the two narrowly marker-gated hooks only when a direct
    # latent continuation is actually requested.
    ensure_h3_runtime_patches()


class H3ContinuousStart:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "width": ("INT", {"default": 1344, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "length": ("INT", {"default": 243, "min": 5, "max": 3600, "step": 17,
                                   "tooltip": "Frames at 24 fps; internally snapped upward to H3's 17k+5 grid. 243 ~= 10.1s."}),
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
                "ref_image_size": (["match", "max"], {"default": "match"}),
            },
            "optional": {
                "reference_image": ("IMAGE", {"tooltip": "Optional Qwen-only identity/style reference. Address it as <Picture 1>. It is NOT added to minimax_refs."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "build"
    CATEGORY = "H3 Continuous"
    DESCRIPTION = "Clip 1: native FL2VA first/last anchors. Optional <Picture 1> is Qwen-only (no persistent ref latent), matching the working production behavior."

    def build(self, clip, vae, prompt, width, height, length, first_frame, last_frame,
              ref_image_size="match", reference_image=None):
        # IMPORTANT: Clip 1 intentionally stays on the native FL2VA endpoint
        # path. No continuation/runtime-patch metadata is attached here.
        latent, frame_count = _empty_av_latent(width, height, length)

        first = _resize(first_frame[:1], width, height, "disabled")
        last = _resize(last_frame[:1], width, height, "center")
        keyframes = [
            {"resolved_frame_index": 0, "latent": vae.encode(first)},
            {"resolved_frame_index": frame_count - 1, "latent": vae.encode(last)},
        ]

        # Match the previously working production behavior:
        # <Picture 1> is Qwen-only. It is NOT inserted into minimax_refs and
        # therefore cannot act as a persistent DiT reference latent mid-clip.
        if reference_image is not None:
            ref = _prepare_qwen_reference_image(reference_image, width, height, ref_image_size)
            tokens = clip.tokenize(prompt, minimax_ref_items=[{"type": "image", "data": ref}])
        else:
            # Native FL2VA presentation when no extra reference is supplied.
            tokens = clip.tokenize(prompt, images=[first, last])

        cond = clip.encode_from_tokens_scheduled(tokens)
        cond = node_helpers.conditioning_set_values(cond, {
            "minimax_keyframes": keyframes,
            "minimax_frame_count": frame_count,
        })
        return (cond, latent)


class H3ContinuousContinue:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "previous_latent": ("LATENT", {"tooltip": "Loaded sampler output from the previous accepted clip."}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "width": ("INT", {"default": 1344, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "length": ("INT", {"default": 243, "min": 5, "max": 3600, "step": 17}),
                "context_frames": (["5", "22", "39"], {"default": "22",
                    "tooltip": "Minimum requested direct-latent motion/audio history. phase_aligned_extended may extend backward to the nearest phase-0 start so the head stays on H3's canonical timeline."}),
                "handover_mode": (["auto", "manual"], {"default": "auto",
                    "tooltip": "AUTO uses freeze-analysis metadata saved with the previous latent. MANUAL uses manual_landing_tail_frames."}),
                "alignment_mode": (["phase_aligned_extended", "phase_aware", "legacy_17"], {"default": "phase_aligned_extended",
                    "tooltip": "phase_aligned_extended (v0.4.6 recommended; handover geometry unchanged from v0.4.3): keep the late cutoff but extend context backward to a phase-0 source start, matching the target head timeline. phase_aware is the v0.4.1 experimental non-zero-phase mode; legacy_17 is the conservative baseline."}),
                "manual_landing_tail_frames": ("INT", {"default": 34, "min": 0, "max": 3400, "step": 1,
                    "tooltip": "Manual/fallback desired pixel tail. phase_aligned_extended/phase_aware snap the END only to an actual latent boundary; legacy_17 requires a multiple of 17. Never trims rendered video."}),
                "ref_image_size": (["match", "max"], {"default": "match"}),
            },
            "optional": {
                "handover": ("H3_CONTINUOUS_HANDOVER", {"tooltip": "Auto-handover metadata from Load AV Latent."}),
                "last_frame": ("IMAGE", {"tooltip": "Recommended: next pre-generated keyframe / target endpoint."}),
                "reference_image": ("IMAGE", {"tooltip": "Optional Qwen-only identity/style reference. Address it as <Picture 1>."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "INT", "INT", "STRING")
    RETURN_NAMES = ("positive", "latent", "actual_head_context_frames", "ignored_tail_frames", "handover_info")
    FUNCTION = "build"
    CATEGORY = "H3 Continuous"
    DESCRIPTION = "Clip 2+: direct AV-latent handover. v0.4.6 keeps the proven phase_aligned_extended handover; calibrated Stable-Tail Consensus defaults plus fixed/adaptive safety selection."

    def build(self, clip, vae, previous_latent, prompt, width, height, length,
              context_frames="22", handover_mode="auto", alignment_mode="phase_aligned_extended",
              manual_landing_tail_frames=34, ref_image_size="match", handover=None,
              last_frame=None, reference_image=None):
        _require_patches()
        context_frames = int(context_frames)
        manual_landing_tail_frames = int(manual_landing_tail_frames)
        alignment_mode = str(alignment_mode).lower()
        if alignment_mode not in ("phase_aligned_extended", "phase_aware", "legacy_17"):
            raise ValueError(f"h3_continuous: unknown alignment_mode {alignment_mode!r}")

        prev_video, prev_audio = _streams_from_latent(previous_latent)
        previous_frame_count = pixel_frames(prev_video.shape[2])

        # Resolve the desired cutoff. In phase-aware AUTO mode we deliberately
        # use the analyzer's IDEAL pixel cutoff, not its already-snapped v0.3
        # value. New v0.4.2 metadata comes from final-frame-lock detection; older metadata remains usable but preserves its older cutoff.
        ideal_last_frame = None
        landing_tail_frames = manual_landing_tail_frames
        handover_source = "manual"
        if str(handover_mode).lower() == "auto":
            if isinstance(handover, dict) and handover.get("available"):
                try:
                    meta_frames = int(handover.get("frame_count", -1))
                    if meta_frames != previous_frame_count:
                        raise ValueError(
                            f"metadata frame_count {meta_frames} != latent frame_count {previous_frame_count}"
                        )
                    if handover.get("detector_mode") not in ("final_frame_lock", "final_frame_lock_robust", "stable_tail_consensus"):
                        _LOG.warning(
                            "h3_continuous: loaded handover metadata predates final-frame-lock detection; "
                            "the stored cutoff will still work, but re-analyze/re-save the source clip to get the new later lock point"
                        )
                    if alignment_mode in ("phase_aligned_extended", "phase_aware"):
                        # v0.4.2+ metadata stores a target derived from the stricter
                        # final-frame-lock detector. Older metadata can still be used,
                        # but its freeze point came from the previous low-motion detector
                        # and therefore cannot gain the v0.4.2 detection improvement
                        # unless the source clip is re-analyzed and re-saved.
                        if alignment_mode == "phase_aligned_extended" and "phase_aligned_target_end_frame" in handover:
                            ideal_last_frame = int(handover["phase_aligned_target_end_frame"])
                            handover_source = "auto handover metadata / phase-aligned-extended"
                        elif "phase_aware_target_end_frame" in handover:
                            ideal_last_frame = int(handover["phase_aware_target_end_frame"])
                            handover_source = f"auto handover metadata / {alignment_mode}"
                        elif handover.get("freeze_detected") and int(handover.get("freeze_start_frame", -1)) >= 0:
                            freeze_start = int(handover["freeze_start_frame"])
                            confidence = float(handover.get("confidence", 0.0))
                            configured_safety = int(handover.get("safety_margin", 1))
                            legacy_safety_mode = str(handover.get("safety_mode", "adaptive"))
                            effective_safety = phase_aware_safety_from_confidence(
                                confidence, configured_safety, safety_mode=legacy_safety_mode
                            )
                            ideal_last_frame = freeze_start - 1 - effective_safety
                            ideal_last_frame = max(context_frames - 1, ideal_last_frame)
                            handover_source = (
                                f"legacy auto metadata / {alignment_mode} "
                                f"(derived from old freeze point; {legacy_safety_mode} safety {effective_safety})"
                            )
                        elif "ideal_handover_end_frame" in handover:
                            ideal_last_frame = int(handover["ideal_handover_end_frame"])
                            handover_source = f"auto metadata / {alignment_mode} conservative fallback"
                        else:
                            raise ValueError("metadata has no usable phase-aware cutoff")
                    else:
                        legacy_tail = handover.get("legacy_landing_tail_frames", handover.get("landing_tail_frames", -1))
                        legacy_tail = int(legacy_tail)
                        if legacy_tail < 0 or legacy_tail % 17 != 0:
                            raise ValueError(f"invalid legacy auto landing tail {legacy_tail}")
                        landing_tail_frames = legacy_tail
                        handover_source = "auto freeze analysis / legacy-17"
                except Exception as e:
                    _LOG.warning(
                        "h3_continuous: auto handover metadata rejected (%s); using manual fallback %s",
                        e, manual_landing_tail_frames
                    )
                    handover_source = "manual fallback (auto metadata invalid)"
            else:
                handover_source = "manual fallback (no auto metadata)"

        # Build the direct source-latent slice.
        if alignment_mode == "phase_aligned_extended":
            if ideal_last_frame is not None:
                sl = phase_aligned_extended_context_slice(
                    prev_video.shape[2], context_frames, ideal_last_frame=ideal_last_frame
                )
            else:
                sl = phase_aligned_extended_context_slice(
                    prev_video.shape[2], context_frames,
                    desired_tail_frames=max(0, manual_landing_tail_frames)
                )
        elif alignment_mode == "phase_aware":
            if ideal_last_frame is not None:
                sl = phase_aware_context_slice(
                    prev_video.shape[2], context_frames, ideal_last_frame=ideal_last_frame
                )
            else:
                sl = phase_aware_context_slice(
                    prev_video.shape[2], context_frames,
                    desired_tail_frames=max(0, manual_landing_tail_frames)
                )
        else:
            if landing_tail_frames < 0 or landing_tail_frames % 17 != 0:
                raise ValueError(
                    "h3_continuous: legacy_17 mode requires landing tail 0 or a multiple of 17"
                )
            sl = context_slice(prev_video.shape[2], context_frames, landing_tail_frames)

        target_latent, frame_count = _empty_av_latent(width, height, length)
        target_video, _ = _streams_from_latent(target_latent)
        if tuple(prev_video.shape[-2:]) != tuple(target_video.shape[-2:]):
            raise ValueError(
                "h3_continuous: direct latent continuation requires identical resolution. "
                f"Previous latent grid {tuple(prev_video.shape[-2:])}, target grid {tuple(target_video.shape[-2:])}."
            )

        source = prev_video[:1, :, sl["start_t"]:sl["end_t"]].clone()
        if int(source.shape[2]) != sl["context_steps"]:
            raise RuntimeError("h3_continuous: internal video context slice length mismatch")

        # phase_aligned_extended deliberately starts the source run on H3 phase 0,
        # so these offsets exactly match the target clip's canonical head grid.
        # The retained phase_aware fallback can still use non-canonical source-relative
        # offsets for A/B comparison with v0.4.1.
        keyframes = []
        for k, pixel_offset in enumerate(sl["offsets"]):
            keyframes.append({
                "resolved_frame_index": 0,
                HC_INDEX: int(pixel_offset),
                "latent": source[:, :, k:k + 1],
            })

        keyframe_images = []
        if last_frame is not None:
            last = _resize(last_frame[:1], width, height, "center")
            keyframe_images.append(last)
            keyframes.append({
                # Mark the continuation endpoint too. This lets the v1.1.4
                # layout wrapper touch only this suite's own keyframes and
                # leave unrelated stock FL2VA/Ref2VA graphs unchanged.
                "resolved_frame_index": 0,
                HC_INDEX: frame_count - 1,
                "latent": vae.encode(last),
            })

        a0, a1, end_error_steps = audio_slice_for_pixel_window(
            prev_audio.shape[-1], sl["source_start_frame"], sl["source_end_frame"]
        )
        audio_context = prev_audio[:1, ..., a0:a1].clone()
        ref_audio_t = int(audio_context.shape[-1])
        actual_context_frames = int(sl.get("actual_context_frames", context_frames))
        audio_end_frame = float(actual_context_frames) + float(end_error_steps) / FRAME_RESCALE
        refs = [{
            "kind": "audio",
            "ref_audio_t": ref_audio_t,
            "audio_latent": audio_context,
            HC_AUDIO_END_FRAME: audio_end_frame,
        }]

        ref_items = []
        if reference_image is not None:
            resized = _prepare_qwen_reference_image(reference_image, width, height, ref_image_size)
            ref_items.append({"type": "image", "data": resized})

        if ref_items:
            tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        elif keyframe_images:
            tokens = clip.tokenize(prompt, images=keyframe_images)
        else:
            tokens = clip.tokenize(prompt)
        cond = clip.encode_from_tokens_scheduled(tokens)
        cond = node_helpers.conditioning_set_values(cond, {
            "minimax_keyframes": keyframes,
            "minimax_frame_count": frame_count,
            "minimax_refs": refs,
        })

        ignored_tail = int(sl.get("ignored_tail_frames", previous_frame_count - sl["source_end_frame"]))
        freeze_note = ""
        if isinstance(handover, dict) and handover.get("available"):
            if handover.get("freeze_detected"):
                freeze_note = (
                    f" | detected freeze start {handover.get('freeze_start_frame')} "
                    f"ideal end {handover.get('ideal_handover_end_frame')}"
                )
            elif handover.get("no_lock_fallback_applied"):
                freeze_note = (
                    f" | no trailing freeze detected | NO-LOCK FALLBACK "
                    f"requested exclude {handover.get('no_lock_fallback_requested_excluded_frames')} frame(s) "
                    f"| effective end {handover.get('handover_end_frame')} "
                    f"| effective tail {handover.get('landing_tail_frames')}"
                )
            else:
                freeze_note = " | no trailing freeze detected"

        phase_note = ""
        if alignment_mode == "phase_aligned_extended":
            phase_note = (
                f" | source phase {sl['source_start_phase']}->{sl['source_end_phase']} "
                f"| canonical-head offsets=yes "
                f"| context extension +{sl.get('context_extension_frames', 0)} frame(s) "
                f"| cutoff quantization loss {sl['cutoff_loss_frames']} frame(s)"
            )
        elif alignment_mode == "phase_aware":
            phase_note = (
                f" | source phase {sl['source_start_phase']}->{sl['source_end_phase']} "
                f"| canonical-head offsets=no "
                f"| cutoff quantization loss {sl['cutoff_loss_frames']} frame(s)"
            )

        info = (
            f"handover={handover_source} | alignment={alignment_mode} | "
            f"source frames {sl['source_start_frame']}..{sl['source_end_frame'] - 1} "
            f"of {sl['previous_frame_count']} | video latent {sl['start_t']}:{sl['end_t']} "
            f"({sl['context_steps']} steps / {actual_context_frames} actual frames; "
            f"requested {context_frames}) | offsets {sl['offsets']} | "
            f"audio latent {a0}:{a1} ({ref_audio_t} steps) | "
            f"ignored previous tail {ignored_tail} frames (latent handover only)" +
            phase_note + freeze_note
        )
        _LOG.info("h3_continuous: %s", info)
        return (cond, target_latent, actual_context_frames, ignored_tail, info)


class H3ContinuousSaveLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Sampler output AV latent for the accepted clip."}),
                "filename_prefix": ("STRING", {"default": "h3_continuous/clip"}),
                "clip_index": ("INT", {"default": 1, "min": 0, "max": 99999,
                    "tooltip": "Fixed chain slot. Clip 1 -> 1, clip 2 -> 2. Re-rendering overwrites that slot. 0 = auto-numbered attempt."}),
            },
            "optional": {
                "handover": ("H3_CONTINUOUS_HANDOVER", {"tooltip": "Optional freeze-analysis metadata. The full latent is still saved unchanged."}),
                "head_context_frames": ("INT", {
                    "forceInput": True,
                    "tooltip": "Clip 1: leave unconnected (0). Clip 2+: connect actual_head_context_frames from Continue from Latent so the saved file is self-describing for later seamless stitching.",
                }),
            },
        }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("latent_path", "latent_info")
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "H3 Continuous"
    DESCRIPTION = "Save the COMPLETE H3 AV latent plus non-destructive handover/head-context metadata for later continuation or seamless saved-chain stitching."

    def save(self, latent, filename_prefix, clip_index=1, handover=None, head_context_frames=0):
        video, audio = _streams_from_latent(latent)
        video_cpu = video.detach().cpu().contiguous()
        audio_cpu = audio.detach().cpu().contiguous()
        frame_count = pixel_frames(video_cpu.shape[2])

        folder, filename, counter, _, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory()
        )
        os.makedirs(folder, exist_ok=True)
        if int(clip_index) > 0:
            path = os.path.join(folder, f"{filename}_{int(clip_index):05d}.safetensors")
        else:
            path = os.path.join(folder, f"{filename}_{int(counter):05d}_.safetensors")

        head_context_frames = max(0, int(head_context_frames or 0))
        metadata = {
            "format": "h3_continuous_av_v8",
            "release_version": "1.2.1",
            "fps": str(FPS),
            "frame_count": str(frame_count),
            "clip_index": str(int(clip_index)),
            "head_context_frames": str(head_context_frames),
            "video_shape": json.dumps(list(video_cpu.shape)),
            "audio_shape": json.dumps(list(audio_cpu.shape)),
        }
        handover_summary = "no handover metadata"
        if isinstance(handover, dict) and handover.get("available"):
            analyzed_frames = int(handover.get("frame_count", frame_count))
            if analyzed_frames != frame_count:
                _LOG.warning(
                    "h3_continuous: analyzer frame_count %s != saved latent frame_count %s; "
                    "handover metadata will not be saved", analyzed_frames, frame_count
                )
                handover_summary = "handover metadata rejected (frame-count mismatch)"
            else:
                clean = dict(handover)
                clean["frame_count"] = frame_count
                metadata["handover_json"] = json.dumps(clean, separators=(",", ":"), sort_keys=True)
                handover_summary = (
                    f"phase-aligned tail {clean.get('landing_tail_frames', '?')} | "
                    f"legacy tail {clean.get('legacy_landing_tail_frames', '?')} | "
                    f"freeze={'yes' if clean.get('freeze_detected') else 'no'}"
                )
                if clean.get("no_lock_fallback_applied"):
                    handover_summary += (
                        f" | no-lock-fallback=yes "
                        f"(requested {clean.get('no_lock_fallback_requested_excluded_frames')} -> "
                        f"effective tail {clean.get('landing_tail_frames')})"
                    )

        st_save({"video": video_cpu, "audio": audio_cpu}, path, metadata=metadata)
        info = (
            f"{frame_count} frames | video {tuple(video_cpu.shape)} | audio {tuple(audio_cpu.shape)} | "
            f"head context {head_context_frames} | {handover_summary}"
        )
        _LOG.info("h3_continuous: saved %s (%s)", path, info)
        return (path, info)


def _resolve_latent_path(path, clip_index):
    p = (path or "").strip().strip('"').strip("'")
    if not p:
        p = "h3_continuous"
    candidates = [p, os.path.join(folder_paths.get_output_directory(), p)]
    for c in candidates:
        if os.path.isfile(c):
            return c
        if os.path.isdir(c):
            files = [os.path.join(c, f) for f in os.listdir(c) if f.endswith(".safetensors")]
            if not files:
                raise FileNotFoundError(f"h3_continuous: no .safetensors files in {c}")
            idx = int(clip_index)
            if idx > 0:
                suffix = f"_{idx:05d}.safetensors"
                fixed = [f for f in files if f.endswith(suffix)]
                if not fixed:
                    raise FileNotFoundError(
                        f"h3_continuous: no fixed slot {idx} in {c} (expected *{suffix})"
                    )
                return max(fixed, key=os.path.getmtime)
            return max(files, key=os.path.getmtime)
    raise FileNotFoundError(
        f"h3_continuous: {p!r} is neither a file nor a folder (also tried ComfyUI output directory)"
    )


class H3ContinuousLoadLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_path": ("STRING", {"default": "h3_continuous",
                    "tooltip": "Specific safetensors file or folder, absolute or relative to ComfyUI/output."}),
                "clip_index": ("INT", {"default": 1, "min": 0, "max": 99999,
                    "tooltip": "Clip to continue FROM. Generating clip 2: load 1. 0 = newest file (not retry-safe)."}),
            }
        }
    RETURN_TYPES = ("LATENT", "STRING", "STRING", "H3_CONTINUOUS_HANDOVER")
    RETURN_NAMES = ("latent", "resolved_path", "latent_info", "handover")
    FUNCTION = "load"
    CATEGORY = "H3 Continuous"
    DESCRIPTION = "Load a complete H3 AV latent and its optional saved automatic handover metadata."

    def load(self, latent_path, clip_index=1):
        path = _resolve_latent_path(latent_path, clip_index)
        tensors = st_load(path, device="cpu")
        if "video" not in tensors or "audio" not in tensors:
            raise ValueError("h3_continuous: file does not contain both 'video' and 'audio' tensors")
        video, audio = tensors["video"], tensors["audio"]
        if video.ndim != 5 or audio.ndim != 4:
            raise ValueError(f"h3_continuous: invalid shapes video={tuple(video.shape)}, audio={tuple(audio.shape)}")
        latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
        frame_count = pixel_frames(video.shape[2])

        metadata = {}
        try:
            with safe_open(path, framework="pt", device="cpu") as f:
                metadata = f.metadata() or {}
        except Exception as e:
            _LOG.warning("h3_continuous: could not read safetensors metadata from %s: %s", path, e)

        handover = {"available": False, "frame_count": frame_count}
        raw = metadata.get("handover_json")
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    parsed["available"] = True
                    parsed["frame_count"] = frame_count
                    handover = parsed
            except Exception as e:
                _LOG.warning("h3_continuous: invalid handover metadata in %s: %s", path, e)

        if handover.get("available"):
            hinfo = (
                f"phase-aligned tail {handover.get('landing_tail_frames')} | "
                f"legacy tail {handover.get('legacy_landing_tail_frames', '?')} | "
                f"freeze={'yes' if handover.get('freeze_detected') else 'no'}"
            )
            if handover.get("no_lock_fallback_applied"):
                hinfo += (
                    f" | no-lock-fallback=yes "
                    f"(requested {handover.get('no_lock_fallback_requested_excluded_frames')} -> "
                    f"effective tail {handover.get('landing_tail_frames')})"
                )
        else:
            hinfo = "no auto handover metadata"
        saved_head = metadata.get("head_context_frames")
        head_info = f"saved head context {saved_head} | " if saved_head is not None else ""
        info = f"{frame_count} frames | video {tuple(video.shape)} | audio {tuple(audio.shape)} | {head_info}{hinfo}"
        _LOG.info("h3_continuous: loaded %s (%s)", path, info)
        return (latent, path, info, handover)


class H3ContinuousAnalyzeHandover:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Decoded FULL rendered frames from the accepted H3 clip."}),
                "analysis_window": ("INT", {"default": 72, "min": 12, "max": 480, "step": 1,
                    "tooltip": "Inspect this many final frames first. If the whole window is already locked, analysis automatically expands backward."}),
                "freeze_hold": ("INT", {"default": 12, "min": 2, "max": 60, "step": 1,
                    "tooltip": "Minimum trailing lock length. Stable-tail consensus must persist for at least this many ending frames."}),
                "safety_margin": ("INT", {"default": 3, "min": 0, "max": 12, "step": 1,
                    "tooltip": "Pixel-frame safety before the detected lock. With safety_mode=fixed (recommended), this configured margin is always respected; adaptive preserves the older confidence-based behavior."}),
                "context_frames": (["5", "22", "39"], {"default": "22",
                    "tooltip": "Minimum motion/audio history. phase_aligned_extended may extend backward beyond this value to restore canonical phase-0 alignment."}),
                "analysis_size": ("INT", {"default": 192, "min": 64, "max": 512, "step": 16,
                    "tooltip": "Max edge used only for analysis. The rendered video is never resized or modified."}),
                "final_mean_diff_threshold": ("FLOAT", {"default": 0.0120, "min": 0.0001, "max": 0.03, "step": 0.0001,
                    "tooltip": "PRIMARY lock test: maximum mean blurred RGB difference between a candidate frame and the median stable-tail reference. Lower = stricter/later lock."}),
                "final_active_pixel_threshold": ("FLOAT", {"default": 0.025, "min": 0.001, "max": 0.2, "step": 0.001,
                    "tooltip": "PRIMARY lock test: per-pixel RGB-difference level used to decide which image areas differ from the median stable-tail reference."}),
                "max_final_active_area_percent": ("FLOAT", {"default": 3.0, "min": 0.05, "max": 20.0, "step": 0.05,
                    "tooltip": "PRIMARY lock test: at most this percentage of the image may differ materially from the median stable-tail reference."}),
                "transition_mean_diff_threshold": ("FLOAT", {"default": 0.0020, "min": 0.0001, "max": 0.03, "step": 0.0001,
                    "tooltip": "SECONDARY safety test: maximum mean luminance change between consecutive frames inside the detected final-frame lock."}),
                "transition_active_pixel_threshold": ("FLOAT", {"default": 0.010, "min": 0.001, "max": 0.2, "step": 0.001,
                    "tooltip": "SECONDARY safety test: per-pixel luminance change that counts as residual motion."}),
                "max_transition_active_area_percent": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 20.0, "step": 0.05,
                    "tooltip": "SECONDARY safety test: maximum visibly changing area allowed for an individual transition."}),
                "min_static_transition_percent": ("FLOAT", {"default": 70.0, "min": 50.0, "max": 100.0, "step": 1.0,
                    "tooltip": "ROBUST secondary gate: minimum percentage of transitions inside the final-frame-matching suffix that must be near-static. Isolated shimmer/outliers are allowed."}),
                "max_consecutive_motion_outliers": ("INT", {"default": 2, "min": 0, "max": 12, "step": 1,
                    "tooltip": "ROBUST secondary gate: maximum consecutive non-static transitions allowed inside an otherwise locked suffix. Prevents sustained real motion from being accepted."}),
                "final_reference_frames": ("INT", {"default": 15, "min": 3, "max": 31, "step": 2,
                    "tooltip": "STABLE-TAIL reference: build the final-state image from the pixel-wise median of this many ending frames instead of trusting one possibly shimmering last frame."}),
                "min_final_match_percent": ("FLOAT", {"default": 75.0, "min": 50.0, "max": 100.0, "step": 1.0,
                    "tooltip": "ROBUST primary gate: minimum percentage of frames in the candidate locked suffix that must match the median final-state reference."}),
                "max_consecutive_final_outliers": ("INT", {"default": 3, "min": 0, "max": 12, "step": 1,
                    "tooltip": "ROBUST primary gate: maximum consecutive final-state mismatches allowed inside the locked suffix. Candidate start itself must always match."}),
                "safety_mode": (["fixed", "adaptive"], {"default": "fixed",
                    "tooltip": "fixed (recommended): always keep safety_margin frames before the detected lock. adaptive: legacy v0.4.1-v0.4.5 behavior that can reduce the effective margin at high confidence."}),
            }
        }
    RETURN_TYPES = ("H3_CONTINUOUS_HANDOVER", "STRING", "BOOLEAN", "INT", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = (
        "handover", "analysis_info", "freeze_detected", "freeze_start_frame",
        "ideal_handover_end_frame", "phase_aligned_handover_end_frame", "ignored_tail_frames", "confidence"
    )
    FUNCTION = "analyze"
    CATEGORY = "H3 Continuous"
    DESCRIPTION = "v0.4.6 calibrated Stable-Tail Consensus detector + phase-aligned-extended handover. Defaults are tuned for safe-early lock detection; fixed safety is recommended."

    def analyze(self, images, analysis_window=72, freeze_hold=12, safety_margin=3,
                context_frames="22", analysis_size=192,
                final_mean_diff_threshold=0.0120,
                final_active_pixel_threshold=0.025,
                max_final_active_area_percent=3.0,
                transition_mean_diff_threshold=0.0020,
                transition_active_pixel_threshold=0.010,
                max_transition_active_area_percent=1.0,
                min_static_transition_percent=70.0,
                max_consecutive_motion_outliers=2,
                final_reference_frames=15,
                min_final_match_percent=75.0,
                max_consecutive_final_outliers=3,
                safety_mode="fixed"):
        context_frames = int(context_frames)
        result = analyze_freeze_tail(
            images,
            analysis_window=analysis_window,
            freeze_hold=freeze_hold,
            safety_margin=safety_margin,
            context_frames=context_frames,
            analysis_size=analysis_size,
            final_mean_diff_threshold=final_mean_diff_threshold,
            final_active_pixel_threshold=final_active_pixel_threshold,
            max_final_active_area_percent=max_final_active_area_percent,
            transition_mean_diff_threshold=transition_mean_diff_threshold,
            transition_active_pixel_threshold=transition_active_pixel_threshold,
            max_transition_active_area_percent=max_transition_active_area_percent,
            min_static_transition_percent=min_static_transition_percent,
            max_consecutive_motion_outliers=max_consecutive_motion_outliers,
            final_reference_frames=final_reference_frames,
            min_final_match_percent=min_final_match_percent,
            max_consecutive_final_outliers=max_consecutive_final_outliers,
            safety_mode=safety_mode,
        )
        if result["freeze_detected"]:
            status = (
                f"FINAL-FRAME LOCK detected | starts frame {result['freeze_start_frame']} | "
                f"locked frames {result['trailing_locked_frames']} | "
                f"conservative ideal end {result['ideal_handover_end_frame']} | "
                f"phase-aligned target end {result['phase_aligned_target_end_frame']} "
                f"({result['safety_mode']} safety {result['phase_aware_effective_safety_margin']}) | "
                f"phase-aligned latent end {result['handover_end_frame']} | "
                f"phase-aligned ignored tail {result['landing_tail_frames']} | "
                f"phase-aligned context {result['phase_aligned_context_frames']} "
                f"(+{result['phase_aligned_context_extension_frames']} extension) | "
                f"v0.4.1 phase-aware end {result['phase_aware_handover_end_frame']} | "
                f"legacy-17 end {result['legacy_handover_end_frame']} | "
                f"legacy-17 tail {result['legacy_landing_tail_frames']} | "
                f"cutoff loss {result['phase_aligned_cutoff_loss_frames']} frame(s) | "
                f"final-match mean diff {result['lock_final_mean_diff']:.6f} | "
                f"final-match active area {result['lock_final_active_area_percent']:.3f}% | "
                f"stable-ref {result['final_reference_frames']} frames | "
                f"primary consensus {result['primary_final_match_ratio_percent']:.1f}% "
                f"({result['primary_final_match_count']}/{result['primary_final_match_frames']}; "
                f"outliers {result['primary_final_match_outliers']}, max streak {result['primary_final_max_consecutive_outliers']}) | "
                f"residual motion mean {result['lock_transition_mean_diff']:.6f} | "
                f"residual active area {result['lock_transition_active_area_percent']:.3f}% | "
                f"residual static {result['residual_static_ratio_percent']:.1f}% "
                f"({result['residual_static_transitions']}/{result['residual_total_transitions']}; "
                f"outliers {result['residual_motion_outliers']}, max streak {result['residual_max_consecutive_outliers']}) | "
                f"confidence {result['confidence']:.3f}"
            )
        else:
            status = (
                f"NO final-frame lock detected | reason {result['no_lock_reason']} | "
                f"primary final-match frames {result['primary_final_match_frames']} | "
                f"phase-aligned tail 0 | legacy tail 0 | "
                f"trailing final-match mean diff {result['lock_final_mean_diff']:.6f} | "
                f"trailing final-match active area {result['lock_final_active_area_percent']:.3f}% | "
                f"stable-ref {result['final_reference_frames']} frames | "
                f"primary consensus {result['primary_final_match_ratio_percent']:.1f}% "
                f"({result['primary_final_match_count']}/{result['primary_final_match_frames']}; "
                f"outliers {result['primary_final_match_outliers']}, max streak {result['primary_final_max_consecutive_outliers']}; "
                f"required >= {result['min_final_match_percent']:.1f}%, streak <= {result['max_consecutive_final_outliers']}) | "
                f"residual motion mean {result['lock_transition_mean_diff']:.6f} | "
                f"residual active area {result['lock_transition_active_area_percent']:.3f}% | "
                f"residual static {result['residual_static_ratio_percent']:.1f}% "
                f"({result['residual_static_transitions']}/{result['residual_total_transitions']}; "
                f"outliers {result['residual_motion_outliers']}, max streak {result['residual_max_consecutive_outliers']}; "
                f"required >= {result['min_static_transition_percent']:.1f}%, streak <= {result['max_consecutive_motion_outliers']})"
            )
        _LOG.info("h3_continuous: %s", status)
        return (
            result, status, bool(result["freeze_detected"]), int(result["freeze_start_frame"]),
            int(result["ideal_handover_end_frame"]), int(result["handover_end_frame"]),
            int(result["landing_tail_frames"]), float(result["confidence"])
        )


class H3ContinuousTrim:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "head_trim_frames": ("INT", {"default": 0, "min": 0, "max": 4096,
                    "tooltip": "OPTIONAL RENDERED-OUTPUT trim only. 0 keeps the complete generated head."}),
                "tail_trim_frames": ("INT", {"default": 0, "min": 0, "max": 4096,
                    "tooltip": "OPTIONAL RENDERED-OUTPUT trim only. 0 keeps the complete generated tail. This is independent of landing_tail_frames in the continuation node."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001}),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
        }
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "trim"
    CATEGORY = "H3 Continuous"
    DESCRIPTION = "Optional rendered-output trim only. It never changes the saved AV latent or the continuation node's latent handover window."

    def trim(self, images, head_trim_frames, tail_trim_frames, fps=24.0, audio=None):
        head = max(0, int(head_trim_frames))
        tail = max(0, int(tail_trim_frames))
        # True bypass: 0/0 must not touch either picture OR audio. In v0.1 the
        # audio branch still length-normalized even at 0/0, which made the node
        # technically non-transparent.
        if head == 0 and tail == 0:
            return (images, audio)
        total = int(images.shape[0])
        if head + tail >= total:
            raise ValueError(f"h3_continuous: head({head}) + tail({tail}) >= clip frames({total})")
        end = total - tail if tail else total
        out_images = images[head:end]

        out_audio = audio
        if audio is not None:
            waveform = audio["waveform"]
            sr = int(audio["sample_rate"])
            head_samples = int(round(head / float(fps) * sr))
            kept_frames = total - head - tail
            want_samples = int(round(kept_frames / float(fps) * sr))
            if head_samples >= waveform.shape[-1]:
                raise ValueError("h3_continuous: audio is shorter than requested head trim")
            waveform = waveform[..., head_samples:]
            # Exact duration match removes H3's small per-clip audio-grid rounding drift.
            waveform = waveform[..., :min(want_samples, waveform.shape[-1])]
            out_audio = {"waveform": waveform, "sample_rate": sr}
        return (out_images, out_audio)


class H3ContinuousLatentInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",)}}
    RETURN_TYPES = ("STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("info", "frame_count", "video_steps", "audio_steps")
    FUNCTION = "inspect"
    CATEGORY = "H3 Continuous"

    def inspect(self, latent):
        video, audio = _streams_from_latent(latent)
        frames = pixel_frames(video.shape[2])
        info = (
            f"H3 AV latent | {frames} frames @ 24fps | "
            f"video {tuple(video.shape)} | audio {tuple(audio.shape)}"
        )
        return (info, frames, int(video.shape[2]), int(audio.shape[-1]))



# ---------------------------------------------------------------------------
# v1.0 release-facing nodes
# ---------------------------------------------------------------------------

class H3ContinuousStartV1(H3ContinuousStart):
    @classmethod
    def INPUT_TYPES(cls):
        base = H3ContinuousStart.INPUT_TYPES()
        required = dict(base["required"])
        required.pop("length", None)
        ordered = {}
        for name, spec in base["required"].items():
            if name == "length":
                ordered["duration"] = ("FLOAT", {
                    "default": 10.0, "min": 0.25, "max": 150.0, "step": 0.1,
                    "tooltip": "Requested duration in seconds at H3's native 24 fps. The actual clip snaps upward to H3's 17k+5 frame grid (10.0 s -> 243 frames ~= 10.125 s).",
                })
            else:
                ordered[name] = required[name]
        return {"required": ordered, "optional": dict(base.get("optional", {}))}

    CATEGORY = "H3 Continuous"
    DESCRIPTION = "v1.0 Clip 1: native FL2VA first/last anchors with user-facing duration in seconds. Optional <Picture 1> remains Qwen-only."

    def build(self, clip, vae, prompt, width, height, duration, first_frame, last_frame,
              ref_image_size="match", reference_image=None):
        requested_frames = duration_to_requested_frames(duration)
        frame_count, _, _ = temporal_shape(requested_frames)
        _LOG.info(
            "h3_continuous: duration %.3fs -> %s requested frames -> %s H3 frames (%.3fs)",
            float(duration), requested_frames, frame_count, frame_count / FPS,
        )
        return super().build(
            clip, vae, prompt, width, height, requested_frames, first_frame, last_frame,
            ref_image_size=ref_image_size, reference_image=reference_image,
        )


class H3ContinuousContinueV1(H3ContinuousContinue):
    @classmethod
    def INPUT_TYPES(cls):
        base = H3ContinuousContinue.INPUT_TYPES()
        required = dict(base["required"])
        ordered = {}
        for name, spec in base["required"].items():
            if name == "length":
                ordered["duration"] = ("FLOAT", {
                    "default": 10.0, "min": 0.25, "max": 150.0, "step": 0.1,
                    "tooltip": "Requested duration in seconds at H3's native 24 fps. The actual clip snaps upward to H3's 17k+5 frame grid (10.0 s -> 243 frames ~= 10.125 s).",
                })
            elif name == "alignment_mode":
                ordered[name] = ([
                    "phase_aligned_extended",
                    "phase_aware (Legacy)",
                    "legacy_17 (Legacy)",
                ], {
                    "default": "phase_aligned_extended",
                    "tooltip": "phase_aligned_extended is the v1.0 recommended direct-latent handover. phase_aware and legacy_17 remain only for reproducing older workflows / A-B diagnostics.",
                })
            else:
                ordered[name] = required[name]
        return {"required": ordered, "optional": dict(base.get("optional", {}))}

    CATEGORY = "H3 Continuous"
    DESCRIPTION = "v1.0 Clip 2+: phase-aligned direct video+audio latent continuation with duration in seconds. Legacy alignment modes remain available for reproducibility."

    def build(self, clip, vae, previous_latent, prompt, width, height, duration,
              context_frames="22", handover_mode="auto", alignment_mode="phase_aligned_extended",
              manual_landing_tail_frames=34, ref_image_size="match", handover=None,
              last_frame=None, reference_image=None):
        requested_frames = duration_to_requested_frames(duration)
        frame_count, _, _ = temporal_shape(requested_frames)
        internal_alignment = normalize_alignment_mode(alignment_mode)
        _LOG.info(
            "h3_continuous: duration %.3fs -> %s requested frames -> %s H3 frames (%.3fs)",
            float(duration), requested_frames, frame_count, frame_count / FPS,
        )
        return super().build(
            clip, vae, previous_latent, prompt, width, height, requested_frames,
            context_frames=context_frames, handover_mode=handover_mode,
            alignment_mode=internal_alignment,
            manual_landing_tail_frames=manual_landing_tail_frames,
            ref_image_size=ref_image_size, handover=handover,
            last_frame=last_frame, reference_image=reference_image,
        )


class H3ContinuousAnalyzeHandoverV1(H3ContinuousAnalyzeHandover):
    @classmethod
    def INPUT_TYPES(cls):
        base = H3ContinuousAnalyzeHandover.INPUT_TYPES()
        old = dict(base["required"])
        ordered = {
            "images": old.pop("images"),
            "preset": (["Balanced", "Motion Safe", "Custom"], {
                "default": "Balanced",
                "tooltip": "Balanced = validated v1.0 detector settings. Motion Safe = same validated detector with a larger fixed pre-lock safety margin. Custom = use the advanced values below.",
            }),
        }
        for name, spec in old.items():
            if name == "safety_mode":
                ordered[name] = (["fixed", "adaptive (Legacy)"], {
                    "default": "fixed",
                    "tooltip": "Used only by Custom. fixed is recommended. adaptive (Legacy) can reduce safety at high confidence and is kept only for old workflow reproduction.",
                })
            else:
                kind, opts = spec
                opts = dict(opts)
                tip = opts.get("tooltip", "")
                opts["tooltip"] = ("Advanced: used only when preset = Custom. " + tip).strip()
                ordered[name] = (kind, opts)
        return {"required": ordered}

    CATEGORY = "H3 Continuous"
    DESCRIPTION = "v1.0 Stable-Tail Consensus analyzer. Balanced is the validated default; Motion Safe keeps the same detector and increases the fixed pre-lock margin."

    def analyze(self, images, preset="Balanced", analysis_window=72, freeze_hold=12, safety_margin=3,
                context_frames="22", analysis_size=192,
                final_mean_diff_threshold=0.0120,
                final_active_pixel_threshold=0.025,
                max_final_active_area_percent=3.0,
                transition_mean_diff_threshold=0.0020,
                transition_active_pixel_threshold=0.010,
                max_transition_active_area_percent=1.0,
                min_static_transition_percent=70.0,
                max_consecutive_motion_outliers=2,
                final_reference_frames=15,
                min_final_match_percent=75.0,
                max_consecutive_final_outliers=3,
                safety_mode="fixed"):
        custom = {
            "analysis_window": analysis_window,
            "freeze_hold": freeze_hold,
            "safety_margin": safety_margin,
            "analysis_size": analysis_size,
            "final_mean_diff_threshold": final_mean_diff_threshold,
            "final_active_pixel_threshold": final_active_pixel_threshold,
            "max_final_active_area_percent": max_final_active_area_percent,
            "transition_mean_diff_threshold": transition_mean_diff_threshold,
            "transition_active_pixel_threshold": transition_active_pixel_threshold,
            "max_transition_active_area_percent": max_transition_active_area_percent,
            "min_static_transition_percent": min_static_transition_percent,
            "max_consecutive_motion_outliers": max_consecutive_motion_outliers,
            "final_reference_frames": final_reference_frames,
            "min_final_match_percent": min_final_match_percent,
            "max_consecutive_final_outliers": max_consecutive_final_outliers,
            "safety_mode": normalize_safety_mode(safety_mode),
        }
        preset_id, effective = resolve_freeze_settings(preset, custom)
        _LOG.info(
            "h3_continuous: freeze preset=%s | fixed safety margin=%s | detector thresholds=%s",
            preset_id, effective["safety_margin"],
            "validated-balanced" if preset_id != "custom" else "custom",
        )
        out = list(super().analyze(
            images,
            analysis_window=effective["analysis_window"],
            freeze_hold=effective["freeze_hold"],
            safety_margin=effective["safety_margin"],
            context_frames=context_frames,
            analysis_size=effective["analysis_size"],
            final_mean_diff_threshold=effective["final_mean_diff_threshold"],
            final_active_pixel_threshold=effective["final_active_pixel_threshold"],
            max_final_active_area_percent=effective["max_final_active_area_percent"],
            transition_mean_diff_threshold=effective["transition_mean_diff_threshold"],
            transition_active_pixel_threshold=effective["transition_active_pixel_threshold"],
            max_transition_active_area_percent=effective["max_transition_active_area_percent"],
            min_static_transition_percent=effective["min_static_transition_percent"],
            max_consecutive_motion_outliers=effective["max_consecutive_motion_outliers"],
            final_reference_frames=effective["final_reference_frames"],
            min_final_match_percent=effective["min_final_match_percent"],
            max_consecutive_final_outliers=effective["max_consecutive_final_outliers"],
            safety_mode=effective["safety_mode"],
        ))
        result = out[0]
        result["release_preset"] = preset_id
        result["release_version"] = "1.0.0"
        label = {"balanced": "Balanced", "motion_safe": "Motion Safe", "custom": "Custom"}[preset_id]
        out[1] = f"preset {label} | {out[1]}"
        return tuple(out)


class H3ContinuousStitchOutputV1:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Decoded FULL rendered frames for the current H3 clip."}),
                "output_mode": (["Full", "Stitch Ready"], {
                    "default": "Full",
                    "tooltip": "Full is a true bypass. Stitch Ready removes the reused head context (clip 2+) and the tail after this clip's exact phase-aligned latent handover boundary.",
                }),
            },
            "optional": {
                "handover": ("H3_CONTINUOUS_HANDOVER", {
                    "tooltip": "Connect the current clip's Auto Handover Analyzer output. Required for Stitch Ready so the tail matches the exact latent cutoff.",
                }),
                "audio": ("AUDIO",),
                "head_context_frames": ("INT", {
                    "forceInput": True,
                    "tooltip": "Clip 1: leave unconnected (0). Clip 2+: connect actual_head_context_frames from H3 Continuous - Continue from Latent v1.0.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "INT", "INT")
    RETURN_NAMES = ("images", "audio", "trim_info", "head_trim_frames", "tail_trim_frames")
    FUNCTION = "prepare"
    CATEGORY = "H3 Continuous"
    DESCRIPTION = "v1.0 rendered AV output helper. Full preserves the complete render; Stitch Ready removes dynamic continuation overlap and the exact phase-aligned freeze tail for direct concatenation."

    def prepare(self, images, output_mode="Full", head_context_frames=0, handover=None, audio=None):
        total = int(images.shape[0])
        plan = stitch_trim_plan(total, output_mode, head_context_frames, handover)
        head = int(plan["head_trim_frames"])
        tail = int(plan["tail_trim_frames"])
        if head == 0 and tail == 0:
            out_images, out_audio = images, audio
        else:
            out_images, out_audio = H3ContinuousTrim().trim(
                images, head, tail, fps=FPS, audio=audio
            )
        info = (
            f"{plan['mode']} | source {total} frames | head trim {head} | tail trim {tail} | "
            f"kept {int(out_images.shape[0])} frames @ {FPS:g}fps"
        )
        if "handover_end_frame" in plan:
            info += f" | phase-aligned source end frame {plan['handover_end_frame']}"
        if plan["mode"] == "stitch_ready" and isinstance(handover, dict) and handover.get("no_lock_fallback_applied"):
            info += (
                f" | no-lock-fallback=yes "
                f"(requested exclude {handover.get('no_lock_fallback_requested_excluded_frames')}, "
                f"effective tail {handover.get('landing_tail_frames')})"
            )
        _LOG.info("h3_continuous: output %s", info)
        return (out_images, out_audio, info, head, tail)


# ---------------------------------------------------------------------------
# v1.2 release-facing nodes
# ---------------------------------------------------------------------------

class H3ContinuousStartV11(H3ContinuousStartV1):
    CATEGORY = "Herrgotts H3 Infinite Continuation Suite"
    DESCRIPTION = "v1.2 Clip 1: native FL2VA first/last anchors with Duration (Seconds). Optional <Picture 1> remains Qwen-only."


class H3ContinuousContinueV11(H3ContinuousContinueV1):
    CATEGORY = "Herrgotts H3 Infinite Continuation Suite"
    DESCRIPTION = "v1.2 Clip 2+: phase-aligned direct video+audio latent continuation. Auto handover consumes lock or no-lock-fallback metadata from the v1.2 analyzer."


def _format_handover_status_v11(result):
    if result["freeze_detected"]:
        return (
            f"FINAL-FRAME LOCK detected | starts frame {result['freeze_start_frame']} | "
            f"locked frames {result['trailing_locked_frames']} | "
            f"conservative ideal end {result['ideal_handover_end_frame']} | "
            f"phase-aligned target end {result['phase_aligned_target_end_frame']} "
            f"({result['safety_mode']} safety {result['phase_aware_effective_safety_margin']}) | "
            f"phase-aligned latent end {result['handover_end_frame']} | "
            f"phase-aligned ignored tail {result['landing_tail_frames']} | "
            f"phase-aligned context {result['phase_aligned_context_frames']} "
            f"(+{result['phase_aligned_context_extension_frames']} extension) | "
            f"cutoff loss {result['phase_aligned_cutoff_loss_frames']} frame(s) | "
            f"final-match mean diff {result['lock_final_mean_diff']:.6f} | "
            f"final-match active area {result['lock_final_active_area_percent']:.3f}% | "
            f"stable-ref {result['final_reference_frames']} frames | "
            f"primary consensus {result['primary_final_match_ratio_percent']:.1f}% "
            f"({result['primary_final_match_count']}/{result['primary_final_match_frames']}; "
            f"outliers {result['primary_final_match_outliers']}, max streak {result['primary_final_max_consecutive_outliers']}) | "
            f"residual motion mean {result['lock_transition_mean_diff']:.6f} | "
            f"residual active area {result['lock_transition_active_area_percent']:.3f}% | "
            f"residual static {result['residual_static_ratio_percent']:.1f}% "
            f"({result['residual_static_transitions']}/{result['residual_total_transitions']}; "
            f"outliers {result['residual_motion_outliers']}, max streak {result['residual_max_consecutive_outliers']}) | "
            f"confidence {result['confidence']:.3f}"
        )

    fallback = ""
    if result.get("no_lock_fallback_applied"):
        fallback = (
            f" | NO-LOCK FALLBACK applied: exclude final "
            f"{result['no_lock_fallback_requested_excluded_frames']} frame(s) before phase alignment | "
            f"fallback target end {result['no_lock_fallback_target_end_frame']} | "
            f"phase-aligned latent end {result['handover_end_frame']} | "
            f"effective ignored tail {result['landing_tail_frames']} | "
            f"phase-aligned context {result['phase_aligned_context_frames']} "
            f"(+{result['phase_aligned_context_extension_frames']} extension) | "
            f"cutoff loss {result['phase_aligned_cutoff_loss_frames']} frame(s)"
        )
    return (
        f"NO final-frame lock detected | reason {result['no_lock_reason']} | "
        f"primary final-match frames {result['primary_final_match_frames']} | "
        f"trailing final-match mean diff {result['lock_final_mean_diff']:.6f} | "
        f"trailing final-match active area {result['lock_final_active_area_percent']:.3f}% | "
        f"stable-ref {result['final_reference_frames']} frames | "
        f"primary consensus {result['primary_final_match_ratio_percent']:.1f}% "
        f"({result['primary_final_match_count']}/{result['primary_final_match_frames']}; "
        f"outliers {result['primary_final_match_outliers']}, max streak {result['primary_final_max_consecutive_outliers']}; "
        f"required >= {result['min_final_match_percent']:.1f}%, streak <= {result['max_consecutive_final_outliers']}) | "
        f"residual motion mean {result['lock_transition_mean_diff']:.6f} | "
        f"residual active area {result['lock_transition_active_area_percent']:.3f}% | "
        f"residual static {result['residual_static_ratio_percent']:.1f}% "
        f"({result['residual_static_transitions']}/{result['residual_total_transitions']}; "
        f"outliers {result['residual_motion_outliers']}, max streak {result['residual_max_consecutive_outliers']}; "
        f"required >= {result['min_static_transition_percent']:.1f}%, streak <= {result['max_consecutive_motion_outliers']})"
        + fallback
    )


class H3ContinuousAnalyzeHandoverV11(H3ContinuousAnalyzeHandoverV1):
    @classmethod
    def INPUT_TYPES(cls):
        base = H3ContinuousAnalyzeHandover.INPUT_TYPES()
        old = dict(base["required"])
        images = old.pop("images")
        # Custom starts from the same calibrated release baseline as Balanced.
        kind, opts = old["freeze_hold"]
        opts = dict(opts)
        opts["default"] = 8
        old["freeze_hold"] = (kind, opts)
        ordered = {
            "images": images,
            "preset": (["Balanced", "Motion Safe", "Custom"], {
                "default": "Balanced",
                "tooltip": "Balanced = validated detector with freeze_hold 8 and fixed 3-frame safety. Motion Safe = same detector with 6-frame safety. Custom reveals the advanced controls.",
            }),
        }
        for name, spec in old.items():
            if name == "safety_mode":
                ordered[name] = (["fixed", "adaptive (Legacy)"], {
                    "default": "fixed",
                    "tooltip": "Custom only. fixed is recommended; adaptive (Legacy) can reduce safety at high confidence.",
                    "advanced": True,
                })
            else:
                kind, opts = spec
                opts = dict(opts)
                tip = opts.get("tooltip", "")
                opts["tooltip"] = ("Custom only. " + tip).strip()
                # Native ComfyUI advanced-widget metadata. The frontend
                # extension additionally ties visibility to preset=Custom.
                opts["advanced"] = True
                ordered[name] = (kind, opts)
        return {"required": ordered}

    CATEGORY = "Herrgotts H3 Infinite Continuation Suite"
    DESCRIPTION = "v1.2 Auto Handover. Balanced/Motion Safe use freeze_hold=8. If no lock is found, freeze_hold-1 ending frames are excluded before phase-aligned latent cutoff selection."

    def analyze(self, images, preset="Balanced", analysis_window=72, freeze_hold=8, safety_margin=3,
                context_frames="22", analysis_size=192,
                final_mean_diff_threshold=0.0120,
                final_active_pixel_threshold=0.025,
                max_final_active_area_percent=3.0,
                transition_mean_diff_threshold=0.0020,
                transition_active_pixel_threshold=0.010,
                max_transition_active_area_percent=1.0,
                min_static_transition_percent=70.0,
                max_consecutive_motion_outliers=2,
                final_reference_frames=15,
                min_final_match_percent=75.0,
                max_consecutive_final_outliers=3,
                safety_mode="fixed"):
        context_frames = int(context_frames)
        custom = {
            "analysis_window": analysis_window,
            "freeze_hold": freeze_hold,
            "safety_margin": safety_margin,
            "analysis_size": analysis_size,
            "final_mean_diff_threshold": final_mean_diff_threshold,
            "final_active_pixel_threshold": final_active_pixel_threshold,
            "max_final_active_area_percent": max_final_active_area_percent,
            "transition_mean_diff_threshold": transition_mean_diff_threshold,
            "transition_active_pixel_threshold": transition_active_pixel_threshold,
            "max_transition_active_area_percent": max_transition_active_area_percent,
            "min_static_transition_percent": min_static_transition_percent,
            "max_consecutive_motion_outliers": max_consecutive_motion_outliers,
            "final_reference_frames": final_reference_frames,
            "min_final_match_percent": min_final_match_percent,
            "max_consecutive_final_outliers": max_consecutive_final_outliers,
            "safety_mode": normalize_safety_mode(safety_mode),
        }
        preset_id, effective = resolve_freeze_settings(preset, custom)
        _LOG.info(
            "h3_continuous: v1.2 freeze preset=%s | freeze_hold=%s | fixed safety margin=%s | detector thresholds=%s",
            preset_id, effective["freeze_hold"], effective["safety_margin"],
            "validated-balanced" if preset_id != "custom" else "custom",
        )
        result = analyze_freeze_tail(
            images,
            analysis_window=effective["analysis_window"],
            freeze_hold=effective["freeze_hold"],
            safety_margin=effective["safety_margin"],
            context_frames=context_frames,
            analysis_size=effective["analysis_size"],
            final_mean_diff_threshold=effective["final_mean_diff_threshold"],
            final_active_pixel_threshold=effective["final_active_pixel_threshold"],
            max_final_active_area_percent=effective["max_final_active_area_percent"],
            transition_mean_diff_threshold=effective["transition_mean_diff_threshold"],
            transition_active_pixel_threshold=effective["transition_active_pixel_threshold"],
            max_transition_active_area_percent=effective["max_transition_active_area_percent"],
            min_static_transition_percent=effective["min_static_transition_percent"],
            max_consecutive_motion_outliers=effective["max_consecutive_motion_outliers"],
            final_reference_frames=effective["final_reference_frames"],
            min_final_match_percent=effective["min_final_match_percent"],
            max_consecutive_final_outliers=effective["max_consecutive_final_outliers"],
            safety_mode=effective["safety_mode"],
        )
        result = apply_no_lock_fallback(
            result, freeze_hold=effective["freeze_hold"], context_frames=context_frames
        )
        result["release_preset"] = preset_id
        result["release_version"] = "1.2.1"
        result["version"] = max(int(result.get("version", 0)), 10)
        status = _format_handover_status_v11(result)
        label = {"balanced": "Balanced", "motion_safe": "Motion Safe", "custom": "Custom"}[preset_id]
        status = f"preset {label} | {status}"
        _LOG.info("h3_continuous: %s", status)
        return (
            result, status, bool(result["freeze_detected"]), int(result["freeze_start_frame"]),
            int(result["ideal_handover_end_frame"]), int(result["handover_end_frame"]),
            int(result["landing_tail_frames"]), float(result["confidence"])
        )


class H3ContinuousStitchOutputV11(H3ContinuousStitchOutputV1):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Decoded FULL rendered frames for the current H3 clip."}),
                "output_mode": (["Full", "Stitch Ready", "Final Clip"], {
                    "default": "Full",
                    "tooltip": "Full keeps the complete render. Stitch Ready removes reused head context and the tail after the effective handover boundary. Final Clip removes only reused head context so the last segment keeps its complete final-keyframe landing.",
                }),
            },
            "optional": {
                "handover": ("H3_CONTINUOUS_HANDOVER", {
                    "tooltip": "Required for Stitch Ready tail trimming. Final Clip ignores the tail cutoff but the analyzer can remain connected for saved metadata or later extension.",
                }),
                "audio": ("AUDIO",),
                "head_context_frames": ("INT", {
                    "forceInput": True,
                    "tooltip": "Clip 1: leave unconnected (0). Clip 2+: connect actual_head_context_frames from Continue from Latent. Final Clip uses this head trim but keeps the full tail.",
                }),
            },
        }

    CATEGORY = "Herrgotts H3 Infinite Continuation Suite"
    DESCRIPTION = "v1.2 rendered AV output helper. Full keeps everything; Stitch Ready removes continuation overlap plus the effective freeze-safe tail; Final Clip removes only the reused head so the last segment can reach its complete Last Frame landing."



class H3ContinuousSeamlessJoinV11:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "previous_images": ("IMAGE", {"tooltip": "Already prepared/combined previous timeline. For the first join, Clip 1 Stitch Ready is fine."}),
                "next_images": ("IMAGE", {"tooltip": "FULL decoded render of the next clip. Its reused context head is needed for the context-aligned seam."}),
                "next_output_mode": (["Stitch Ready", "Final Clip"], {
                    "default": "Stitch Ready",
                    "tooltip": "Intermediate next clip = Stitch Ready. Current last clip = Final Clip so its complete ending is preserved.",
                }),
                "next_head_context_frames": ("INT", {"forceInput": True,
                    "tooltip": "Connect actual_head_context_frames from the next Continue from Latent node."}),
                "video_crossfade_frames": ("INT", {"default": 4, "min": 0, "max": 16, "step": 1,
                    "tooltip": "Short context-aligned video blend after Safe Tail Bridge. 4 frames is the recommended default; 0 disables the blend."}),
                "audio_crossfade_ms": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Short audio de-click crossfade. 15 ms is the tested default; keep it short to reduce phasing/doubled transients."}),
                "luminance_match": ("BOOLEAN", {"default": False,
                    "tooltip": "Experimental fallback only. Safe Tail Bridge is the release default; enable luminance matching only if a persistent brightness seam remains."}),
                "luminance_fade_frames": ("INT", {"default": 16, "min": 0, "max": 96, "step": 1,
                    "tooltip": "Experimental luminance-match fade length. Ignored when luminance_match is off."}),
                "max_luminance_correction_percent": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 25.0, "step": 0.5,
                    "tooltip": "Experimental luminance-match safety clamp. Ignored when luminance_match is off."}),
                "max_safe_tail_bridge_frames": ("INT", {"default": 2, "min": 0, "max": 4, "step": 1,
                    "tooltip": "Recommended: 2. Reuses only detector-approved rendered frames lost to phase alignment; never borrows from the freeze safety margin."}),
            },
            "optional": {
                "previous_audio": ("AUDIO",),
                "next_audio": ("AUDIO",),
                "next_handover": ("H3_CONTINUOUS_HANDOVER", {
                    "tooltip": "Required when next_output_mode is Stitch Ready so the next clip's freeze-safe tail is removed."}),
                "previous_full_images": ("IMAGE", {
                    "tooltip": "Optional FULL decoded render of the previous individual clip. Connect this together with previous_handover to enable Safe Tail Bridge."}),
                "previous_handover": ("H3_CONTINUOUS_HANDOVER", {
                    "tooltip": "Previous clip handover metadata. Together with previous_full_images it exposes up to 1-2 safe rendered frames that latent phase alignment had to discard."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "join_info")
    FUNCTION = "join"
    CATEGORY = "Herrgotts H3 Infinite Continuation Suite"
    DESCRIPTION = "v1.2 context-aligned rendered AV join. Safe Tail Bridge replaces the first few potentially unstable continuation frames with detector-approved pixels from the previous clip; video gets a short corresponding-context blend and audio keeps the tested 15 ms de-click crossfade."

    def join(self, previous_images, next_images, next_output_mode="Stitch Ready",
             next_head_context_frames=22, video_crossfade_frames=4, audio_crossfade_ms=15.0,
             luminance_match=False, luminance_fade_frames=16, max_luminance_correction_percent=10.0,
             max_safe_tail_bridge_frames=2, previous_audio=None, next_audio=None, next_handover=None,
             previous_full_images=None, previous_handover=None):
        previous_frames = int(previous_images.shape[0])
        next_total = int(next_images.shape[0])
        plan = stitch_trim_plan(
            next_total, next_output_mode, int(next_head_context_frames), next_handover
        )
        audio_head = int(plan["head_trim_frames"])
        tail = int(plan["tail_trim_frames"])

        bridge_images = previous_images[:0]
        bridge_stats = safe_tail_bridge_plan(previous_handover, int(max_safe_tail_bridge_frames))
        bridge = int(bridge_stats["safe_tail_bridge_frames"])
        if bridge > 0:
            if previous_full_images is None:
                # Missing pixel source: fail safe to the old seam rather than invent frames.
                bridge = 0
                bridge_stats = safe_tail_bridge_plan(None, 0)
            else:
                bridge_images, bridge_stats = extract_safe_tail_bridge_images(
                    previous_full_images, previous_handover, int(max_safe_tail_bridge_frames)
                )
                bridge = int(bridge_images.shape[0])

        # The bridge takes video time positions that would otherwise be the first
        # generated body frames of the next clip. Keep the audio boundary unchanged.
        max_bridge_for_next = max(0, next_total - tail - audio_head - 1)
        if bridge > max_bridge_for_next:
            bridge = max_bridge_for_next
            bridge_images = bridge_images[:bridge]
        previous_video = previous_images
        if bridge > 0:
            bridge_images = bridge_images.to(previous_images.device, previous_images.dtype)
            previous_video = torch.cat((previous_images, bridge_images), dim=0)
        video_head = audio_head + bridge

        out_images, vstats = context_aligned_video_join(
            previous_video, next_images, video_head, tail, int(video_crossfade_frames),
            luminance_match=bool(luminance_match),
            luminance_fade_frames=int(luminance_fade_frames),
            max_luminance_correction_percent=float(max_luminance_correction_percent),
        )
        out_audio, astats = context_aligned_audio_join(
            previous_audio, next_audio,
            previous_output_frames=previous_frames,
            next_total_frames=next_total,
            next_head_context_frames=audio_head,
            next_tail_trim_frames=tail,
            crossfade_ms=float(audio_crossfade_ms),
            fps=FPS,
        )
        # Bridge adds N previous pixels and removes N next pixels, so the video
        # timeline must still equal the audio/hard-stitch timeline exactly.
        expected_frames = previous_frames + next_total - audio_head - tail
        if int(out_images.shape[0]) != expected_frames:
            raise ValueError(
                f"safe-tail bridge changed timeline length: output {int(out_images.shape[0])}, expected {expected_frames}"
            )

        luma_text = "off"
        if vstats.get("luminance_match_enabled"):
            luma_text = (
                f"gain {vstats.get('luminance_applied_gain', 1.0):.4f} "
                f"(measured {vstats.get('luminance_measured_gain', 1.0):.4f}, "
                f"fade {vstats.get('luminance_fade_frames', 0)}f, "
                f"clamped={'yes' if vstats.get('luminance_clamped') else 'no'})"
            )
        bridge_text = f"{bridge}f"
        available = int(bridge_stats.get("safe_tail_bridge_available_frames", 0))
        if available > bridge:
            bridge_text += f"/{available}f available"
        info = (
            f"context-aligned join | previous {previous_frames} frames | next source {next_total} frames | "
            f"next audio head {audio_head} | video head {video_head} | safe tail bridge {bridge_text} | "
            f"next tail {tail} | video crossfade {vstats['video_crossfade_frames']} frame(s) | "
            f"boundary luminance {luma_text} | audio crossfade {astats.get('audio_crossfade_samples', 0)} samples "
            f"(~{astats.get('audio_crossfade_ms_effective', 0)} ms) | output {int(out_images.shape[0])} frames"
        )
        _LOG.info("h3_continuous: %s", info)
        return (out_images, out_audio, info)

def _read_safetensors_metadata(path):
    metadata = {}
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
    except Exception as e:
        _LOG.warning("h3_continuous: could not read safetensors metadata from %s: %s", path, e)
    handover = None
    raw = metadata.get("handover_json")
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                parsed["available"] = True
                handover = parsed
        except Exception as e:
            _LOG.warning("h3_continuous: invalid handover metadata in %s: %s", path, e)
    return metadata, handover


def _saved_chain_base(prefix):
    p = (prefix or "").strip().strip('"').strip("'")
    if not p:
        p = "h3_continuous/clip"
    if p.lower().endswith(".safetensors"):
        p = p[:-len(".safetensors")]
    if os.path.isabs(p):
        return p
    return os.path.join(folder_paths.get_output_directory(), p)


def _saved_chain_file(prefix, clip_index):
    base = _saved_chain_base(prefix)
    exact = f"{base}_{int(clip_index):05d}.safetensors"
    if os.path.isfile(exact):
        return exact
    # Folder fallback for users who pass h3_continuous rather than h3_continuous/clip.
    if os.path.isdir(base):
        suffix = f"_{int(clip_index):05d}.safetensors"
        matches = [os.path.join(base, f) for f in os.listdir(base) if f.endswith(suffix)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"h3_continuous: multiple saved chains contain clip {clip_index} in {base}; "
                "use the exact latent prefix, e.g. h3_continuous/clip"
            )
    raise FileNotFoundError(f"h3_continuous: saved chain clip {clip_index} not found at {exact}")


def _discover_saved_last_clip(prefix, first_clip):
    import re
    base = _saved_chain_base(prefix)
    directory = os.path.dirname(base)
    stem = os.path.basename(base)
    if os.path.isdir(base):
        directory = base
        pattern = re.compile(r".*_(\d{5})\.safetensors$")
    else:
        pattern = re.compile(re.escape(stem) + r"_(\d{5})\.safetensors$")
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"h3_continuous: saved chain directory does not exist: {directory}")
    indices = []
    for name in os.listdir(directory):
        m = pattern.fullmatch(name)
        if m:
            idx = int(m.group(1))
            if idx >= int(first_clip):
                indices.append(idx)
    if not indices:
        raise FileNotFoundError(f"h3_continuous: no saved chain clips found for prefix {prefix!r}")
    return max(indices)


def _saved_tail_trim(frame_count, handover, is_final):
    if is_final:
        return 0
    if not isinstance(handover, dict) or not handover.get("available"):
        raise ValueError("intermediate saved clips require handover metadata for freeze-safe stitching")
    meta_frames = int(handover.get("frame_count", frame_count))
    if meta_frames != int(frame_count):
        raise ValueError(f"saved handover frame_count {meta_frames} != latent frame_count {frame_count}")
    end_frame = int(handover.get("handover_end_frame", frame_count - 1))
    tail = int(frame_count) - (end_frame + 1)
    stored = handover.get("landing_tail_frames")
    if stored is not None and int(stored) != tail:
        raise ValueError("saved handover metadata has inconsistent landing_tail_frames")
    return max(0, tail)


def _decode_saved_av(video_vae, audio_vae, video_latent, audio_latent):
    images = video_vae.decode(video_latent)
    if len(images.shape) == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    if images.ndim != 4:
        raise ValueError(f"unexpected decoded video shape {tuple(images.shape)}")

    waveform = audio_vae.decode(audio_latent).movedim(-1, 1)
    std = torch.std(waveform, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    waveform = waveform / std
    sr = int(getattr(audio_vae, "audio_sample_rate_output", getattr(audio_vae, "audio_sample_rate", 44100)))
    return images, {"waveform": waveform, "sample_rate": sr}


class H3ContinuousStitchSavedChainV11:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_vae": ("VAE", {"tooltip": "MiniMax H3 Video VAE used to decode the saved full video latents."}),
                "audio_vae": ("VAE", {"tooltip": "MiniMax H3 Audio VAE used to decode the saved full audio latents."}),
                "latent_prefix": ("STRING", {"default": "h3_continuous/clip",
                    "tooltip": "Saved latent prefix relative to ComfyUI/output, e.g. h3_continuous/clip for clip_00001.safetensors, clip_00002.safetensors, ..."}),
                "first_clip": ("INT", {"default": 1, "min": 1, "max": 99999}),
                "last_clip": ("INT", {"default": 0, "min": 0, "max": 99999,
                    "tooltip": "0 = automatically use the highest numbered clip for this prefix."}),
                "filename_prefix": ("STRING", {"default": "video/Herrgotts_H3_Infinite_Stitched"}),
                "video_crossfade_frames": ("INT", {"default": 4, "min": 0, "max": 16, "step": 1,
                    "tooltip": "Context-aligned video crossfade. Recommended: 4 frames."}),
                "audio_crossfade_ms": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Short audio de-click crossfade. Recommended: 15 ms."}),
                "luminance_match": ("BOOLEAN", {"default": False,
                    "tooltip": "Experimental fallback only. Safe Tail Bridge is the release default; enable only if a persistent brightness seam remains."}),
                "luminance_fade_frames": ("INT", {"default": 16, "min": 0, "max": 96, "step": 1,
                    "tooltip": "Frames over which the temporary brightness correction returns to native luminance. Recommended: 16."}),
                "max_luminance_correction_percent": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 25.0, "step": 0.5,
                    "tooltip": "Safety clamp for automatic brightness correction. Recommended: 10%."}),
                "crf": ("INT", {"default": 18, "min": 0, "max": 51, "step": 1,
                    "tooltip": "H.264 quality. Lower = larger/higher quality. 18 is a high-quality default."}),
                "max_safe_tail_bridge_frames": ("INT", {"default": 2, "min": 0, "max": 4, "step": 1,
                    "tooltip": "Recommended: 2. Keeps only detector-approved rendered frames lost to phase alignment, then skips the same number of early video frames in the next clip."}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "stitch_info")
    FUNCTION = "stitch"
    OUTPUT_NODE = True
    CATEGORY = "Herrgotts H3 Infinite Continuation Suite"
    DESCRIPTION = "v1.2 memory-bounded saved-chain stitcher. Uses Safe Tail Bridge plus short context-aligned video/audio seam smoothing and encodes directly to MP4 so peak memory does not grow with chain length."

    def stitch(self, video_vae, audio_vae, latent_prefix="h3_continuous/clip",
               first_clip=1, last_clip=0, filename_prefix="video/Herrgotts_H3_Infinite_Stitched",
               video_crossfade_frames=4, audio_crossfade_ms=15.0, luminance_match=False,
               luminance_fade_frames=16, max_luminance_correction_percent=10.0, crf=18,
               max_safe_tail_bridge_frames=2):
        # PyAV is part of current ComfyUI's video/audio stack. Import lazily so
        # installing the pack does not add import-time dependencies or side effects.
        try:
            import av
            import numpy as np
            from fractions import Fraction
        except Exception as e:
            raise RuntimeError("h3_continuous: PyAV is required for Saved Chain Stitching (normally provided by ComfyUI)") from e

        first = int(first_clip)
        last = int(last_clip)
        if last == 0:
            last = _discover_saved_last_clip(latent_prefix, first)
        if last < first:
            raise ValueError(f"last_clip {last} must be >= first_clip {first}")

        paths = [_saved_chain_file(latent_prefix, i) for i in range(first, last + 1)]
        if len(paths) < 1:
            raise ValueError("no clips selected")

        folder, filename, counter, _, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory()
        )
        os.makedirs(folder, exist_ok=True)
        out_path = os.path.join(folder, f"{filename}_{int(counter):05d}_.mp4")
        while os.path.exists(out_path):
            counter += 1
            out_path = os.path.join(folder, f"{filename}_{int(counter):05d}_.mp4")

        output = None
        vstream = None
        astream = None
        video_written = 0
        audio_written = 0
        sr = None
        channels = None
        layout = None
        pending_video = None
        pending_bridge_video = None
        pending_audio = None
        previous_handover = None
        bridge_from_previous = 0
        logical_frames = 0
        logical_audio_samples = 0
        requested_vfade = max(0, int(video_crossfade_frames))
        requested_afade_ms = max(0.0, float(audio_crossfade_ms))
        requested_luma_match = bool(luminance_match)
        requested_luma_fade = max(0, int(luminance_fade_frames))
        requested_luma_max_percent = max(0.0, float(max_luminance_correction_percent))
        requested_bridge_max = max(0, int(max_safe_tail_bridge_frames))
        # Hold a few extra video frames only for the robust boundary luminance
        # estimator. This buffer is tiny and remains independent of chain length.
        video_boundary_hold = max(requested_vfade, LUMINANCE_ANALYSIS_FRAMES if requested_luma_match else 0)
        clip_summaries = []

        def write_video(frames):
            nonlocal video_written
            if frames is None or int(frames.shape[0]) == 0:
                return
            for frame_tensor in frames:
                img = (frame_tensor * 255).clamp(0, 255).byte().detach().cpu().numpy()
                frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")
                frame = frame.reformat(format="yuv420p")
                frame.pts = video_written
                frame.time_base = Fraction(1, int(FPS))
                for packet in vstream.encode(frame):
                    output.mux(packet)
                video_written += 1

        def write_audio(wave):
            nonlocal audio_written
            if wave is None or int(wave.shape[-1]) == 0:
                return
            arr = wave[0].float().detach().cpu().contiguous().numpy()
            frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(arr), format="fltp", layout=layout)
            frame.sample_rate = sr
            frame.pts = audio_written
            frame.time_base = Fraction(1, sr)
            for packet in astream.encode(frame):
                output.mux(packet)
            audio_written += int(wave.shape[-1])

        try:
            for offset, (clip_index, path) in enumerate(zip(range(first, last + 1), paths)):
                tensors = st_load(path, device="cpu")
                if "video" not in tensors or "audio" not in tensors:
                    raise ValueError(f"saved clip {clip_index} lacks video/audio tensors: {path}")
                video_latent, audio_latent = tensors["video"], tensors["audio"]
                if video_latent.ndim != 5 or audio_latent.ndim != 4:
                    raise ValueError(
                        f"invalid saved shapes for clip {clip_index}: video={tuple(video_latent.shape)}, audio={tuple(audio_latent.shape)}"
                    )
                frame_count = pixel_frames(video_latent.shape[2])
                metadata, handover = _read_safetensors_metadata(path)
                if handover is not None:
                    handover["frame_count"] = frame_count
                head, head_source = resolve_saved_head_context(metadata, clip_index, previous_handover)
                is_final = clip_index == last
                tail = _saved_tail_trim(frame_count, handover, is_final)

                images, audio = _decode_saved_av(video_vae, audio_vae, video_latent, audio_latent)
                if int(images.shape[0]) != frame_count:
                    raise ValueError(
                        f"decoded clip {clip_index} has {int(images.shape[0])} frames, expected {frame_count}"
                    )
                wave = audio["waveform"]
                clip_sr = int(audio["sample_rate"])
                clip_channels = int(wave.shape[1])

                if output is None:
                    h, w = int(images.shape[1]), int(images.shape[2])
                    if w % 2 or h % 2:
                        raise ValueError(f"H.264 output requires even dimensions, got {w}x{h}")
                    sr = clip_sr
                    channels = clip_channels
                    if channels not in (1, 2):
                        raise ValueError(f"Saved Chain Stitch currently supports mono/stereo audio, got {channels} channels")
                    layout = "mono" if channels == 1 else "stereo"
                    output = av.open(out_path, mode="w", options={"movflags": "use_metadata_tags+faststart"})
                    output.metadata["herrgotts_h3_infinite_version"] = "1.2.1"
                    output.metadata["clip_range"] = f"{first}-{last}"
                    output.metadata["video_crossfade_frames"] = str(requested_vfade)
                    output.metadata["audio_crossfade_ms"] = str(requested_afade_ms)
                    output.metadata["boundary_luminance_match"] = str(requested_luma_match).lower()
                    output.metadata["luminance_fade_frames"] = str(requested_luma_fade)
                    output.metadata["max_luminance_correction_percent"] = str(requested_luma_max_percent)
                    output.metadata["max_safe_tail_bridge_frames"] = str(requested_bridge_max)
                    vstream = output.add_stream("h264", rate=Fraction(int(FPS), 1))
                    vstream.codec_context.max_b_frames = 0
                    vstream.codec_context.time_base = Fraction(1, int(FPS))
                    vstream.width = w
                    vstream.height = h
                    vstream.pix_fmt = "yuv420p"
                    vstream.options = {"crf": str(int(crf))}
                    astream = output.add_stream("aac", rate=sr, layout=layout)
                else:
                    if (int(images.shape[2]), int(images.shape[1])) != (vstream.width, vstream.height):
                        raise ValueError(f"clip {clip_index} resolution differs from the first clip")
                    if clip_sr != sr:
                        raise ValueError(f"clip {clip_index} audio sample rate {clip_sr} != {sr}")
                    if clip_channels != channels:
                        raise ValueError(f"clip {clip_index} audio channels {clip_channels} != {channels}")

                end = frame_count - tail if tail else frame_count
                incoming_bridge = max(0, int(bridge_from_previous))
                # Safe Tail Bridge is video-only: keep N detector-approved pixels
                # from the previous clip and skip the same N early video frames
                # here. Audio keeps its already-tested boundary unchanged.
                incoming_bridge = min(incoming_bridge, max(0, end - head - 1))
                video_head = head + incoming_bridge
                body_images = images[video_head:end]
                body_audio = frame_trimmed_audio(audio, frame_count, head, tail, FPS)["waveform"]

                # The hard-stitch timeline contribution stays unchanged. The
                # previous bridge adds N video frames while this body loses N.
                base_body_frames = int(frame_count - head - tail)
                target_total_frames = logical_frames + base_body_frames
                target_total_samples = int(round(target_total_frames / float(FPS) * sr))
                want_body_samples = max(0, target_total_samples - logical_audio_samples)
                body_audio = fit_audio_length(body_audio, want_body_samples)
                logical_frames = target_total_frames
                logical_audio_samples = target_total_samples

                # Prepare up to 1-2 safe rendered pixels that latent phase
                # alignment had to discard. They are kept only until the next
                # video seam and never affect audio timing.
                future_bridge_video = images[:0].detach().cpu()
                future_bridge_stats = safe_tail_bridge_plan(handover, requested_bridge_max)
                future_bridge_count = 0
                if not is_final and int(future_bridge_stats.get("safe_tail_bridge_frames", 0)) > 0:
                    future_bridge_video, future_bridge_stats = extract_safe_tail_bridge_images(
                        images, handover, requested_bridge_max
                    )
                    future_bridge_video = future_bridge_video.detach().cpu()
                    future_bridge_count = int(future_bridge_video.shape[0])

                if offset == 0:
                    # First selected clip has no incoming join. Hold only the
                    # tiny boundary buffers required for the next seam.
                    if not is_final:
                        vn = min(video_boundary_hold, int(body_images.shape[0]))
                        an_req = int(round(requested_afade_ms / 1000.0 * sr))
                        an = min(an_req, int(body_audio.shape[-1]))
                        write_video(body_images[:-vn] if vn else body_images)
                        pending_video = body_images[-vn:].detach().cpu() if vn else body_images[:0].detach().cpu()
                        pending_bridge_video = future_bridge_video
                        write_audio(body_audio[..., :-an] if an else body_audio)
                        pending_audio = body_audio[..., -an:].detach().cpu() if an else body_audio[..., :0].detach().cpu()
                    else:
                        write_video(body_images)
                        write_audio(body_audio)
                        pending_bridge_video = None
                    effective_vfade = 0
                    effective_afade = 0
                    luma_gain = 1.0
                    luma_measured = 1.0
                    luma_clamped = False
                    luma_faded = 0
                    luma_analysis = 0
                else:
                    # The visible previous boundary consists of the normal
                    # latent-cutoff tail plus any safe rendered bridge frames.
                    previous_boundary = pending_video
                    if pending_bridge_video is not None and int(pending_bridge_video.shape[0]) > 0:
                        if previous_boundary is None:
                            previous_boundary = pending_bridge_video
                        else:
                            previous_boundary = torch.cat((previous_boundary, pending_bridge_video), dim=0)

                    luma_gain = 1.0
                    luma_measured = 1.0
                    luma_clamped = False
                    luma_faded = 0
                    luma_analysis = 0
                    if requested_luma_match and previous_boundary is not None and video_head > 0:
                        luma_analysis = min(LUMINANCE_ANALYSIS_FRAMES, video_head, int(previous_boundary.shape[0]))
                        if luma_analysis > 0:
                            lstats = estimate_luminance_gain(
                                previous_boundary[-luma_analysis:],
                                images[video_head - luma_analysis:video_head].detach().cpu().to(previous_boundary.dtype),
                                max_correction_percent=requested_luma_max_percent,
                            )
                            luma_gain = float(lstats["luminance_applied_gain"])
                            luma_measured = float(lstats["luminance_measured_gain"])
                            luma_clamped = bool(lstats["luminance_clamped"])
                            body_images, luma_faded = apply_luminance_gain_fade(
                                body_images, luma_gain, requested_luma_fade, inplace=True
                            )

                    vn = min(requested_vfade, video_head, int(previous_boundary.shape[0]) if previous_boundary is not None else 0)
                    if previous_boundary is not None:
                        if int(previous_boundary.shape[0]) > vn:
                            write_video(previous_boundary[:-vn] if vn else previous_boundary)
                        if vn:
                            prev_tail = previous_boundary[-vn:]
                            next_overlap = images[video_head - vn:video_head].detach().cpu().to(prev_tail.dtype)
                            if requested_luma_match:
                                next_overlap = apply_rgb_gain(next_overlap, luma_gain)
                            write_video(blend_video_overlap(prev_tail, next_overlap))

                    head_samples = int(round(head / float(FPS) * sr))
                    an_req = int(round(requested_afade_ms / 1000.0 * sr))
                    an = min(an_req, head_samples, int(pending_audio.shape[-1]) if pending_audio is not None else 0)
                    if pending_audio is not None:
                        if int(pending_audio.shape[-1]) > an:
                            write_audio(pending_audio[..., :-an] if an else pending_audio)
                        if an:
                            prev_tail_a = pending_audio[..., -an:]
                            next_overlap_a = wave[..., head_samples - an:head_samples].detach().cpu().to(prev_tail_a.dtype)
                            write_audio(blend_audio_overlap(prev_tail_a, next_overlap_a))

                    if not is_final:
                        hold_v = min(video_boundary_hold, int(body_images.shape[0]))
                        hold_a_req = int(round(requested_afade_ms / 1000.0 * sr))
                        hold_a = min(hold_a_req, int(body_audio.shape[-1]))
                        write_video(body_images[:-hold_v] if hold_v else body_images)
                        pending_video = body_images[-hold_v:].detach().cpu() if hold_v else body_images[:0].detach().cpu()
                        pending_bridge_video = future_bridge_video
                        write_audio(body_audio[..., :-hold_a] if hold_a else body_audio)
                        pending_audio = body_audio[..., -hold_a:].detach().cpu() if hold_a else body_audio[..., :0].detach().cpu()
                    else:
                        write_video(body_images)
                        write_audio(body_audio)
                        pending_video = None
                        pending_bridge_video = None
                        pending_audio = None
                    effective_vfade = vn
                    effective_afade = an

                luma_summary = "off"
                if requested_luma_match and offset > 0:
                    luma_summary = (
                        f"gain {luma_gain:.4f} (measured {luma_measured:.4f}, "
                        f"analysis {luma_analysis}f, fade {luma_faded}f, "
                        f"clamped={'yes' if luma_clamped else 'no'})"
                    )
                clip_summaries.append(
                    f"clip {clip_index}: audio-head {head} ({head_source}), video-head {video_head}, tail {tail}, "
                    f"bridge-in {incoming_bridge}f, bridge-out {future_bridge_count}f, "
                    f"join {effective_vfade}f/{round(effective_afade / sr * 1000.0, 1) if sr else 0}ms, "
                    f"luma {luma_summary}"
                )
                bridge_from_previous = future_bridge_count
                previous_handover = handover

                del tensors, video_latent, audio_latent, images, audio, wave, body_images, body_audio
                comfy.model_management.soft_empty_cache()

            if pending_video is not None:
                write_video(pending_video)
            if pending_audio is not None:
                write_audio(pending_audio)

            for packet in vstream.encode(None):
                output.mux(packet)
            for packet in astream.encode(None):
                output.mux(packet)
            output.close()
            output = None
        except BaseException:
            if output is not None:
                try:
                    output.close()
                except Exception:
                    pass
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except Exception:
                    pass
            raise

        if video_written != logical_frames:
            raise ValueError(
                f"saved-chain video timeline mismatch after Safe Tail Bridge: wrote {video_written} frames, expected {logical_frames}"
            )
        expected_audio_samples = int(round(video_written / float(FPS) * sr)) if sr else 0
        drift = audio_written - expected_audio_samples
        info = (
            f"saved-chain seamless stitch complete | clips {first}-{last} | {video_written} frames @ {FPS:g}fps | "
            f"audio {audio_written} samples @ {sr}Hz | A/V sample rounding delta {drift} | "
            f"safe tail bridge <= {requested_bridge_max}f | video crossfade <= {requested_vfade}f | "
            f"audio crossfade <= {requested_afade_ms:g}ms | "
            f"boundary luminance {'on' if requested_luma_match else 'off'} "
            f"(experimental; fade {requested_luma_fade}f, max ±{requested_luma_max_percent:g}%) | "
            + " ; ".join(clip_summaries)
        )
        _LOG.info("h3_continuous: %s", info)
        return (out_path, info)

# Shared release nodes live with the v1.2 suite in the Add Node menu.
H3ContinuousSaveLatent.CATEGORY = "Herrgotts H3 Infinite Continuation Suite"
H3ContinuousLoadLatent.CATEGORY = "Herrgotts H3 Infinite Continuation Suite"
H3ContinuousLatentInfo.CATEGORY = "Herrgotts H3 Infinite Continuation Suite"

NODE_CLASS_MAPPINGS = {
    # v1.2 release-facing nodes
    "H3ContinuousStartV11": H3ContinuousStartV11,
    "H3ContinuousContinueV11": H3ContinuousContinueV11,
    "H3ContinuousAnalyzeHandoverV11": H3ContinuousAnalyzeHandoverV11,
    "H3ContinuousStitchOutputV11": H3ContinuousStitchOutputV11,
    "H3ContinuousSeamlessJoinV11": H3ContinuousSeamlessJoinV11,
    "H3ContinuousStitchSavedChainV11": H3ContinuousStitchSavedChainV11,
    # v1.0 compatibility nodes
    "H3ContinuousStartV1": H3ContinuousStartV1,
    "H3ContinuousContinueV1": H3ContinuousContinueV1,
    "H3ContinuousAnalyzeHandoverV1": H3ContinuousAnalyzeHandoverV1,
    "H3ContinuousStitchOutputV1": H3ContinuousStitchOutputV1,
    # Shared persistence / inspection nodes
    "H3ContinuousSaveLatent": H3ContinuousSaveLatent,
    "H3ContinuousLoadLatent": H3ContinuousLoadLatent,
    "H3ContinuousLatentInfo": H3ContinuousLatentInfo,
    # v0.x compatibility
    "H3ContinuousStart": H3ContinuousStart,
    "H3ContinuousContinue": H3ContinuousContinue,
    "H3ContinuousAnalyzeHandover": H3ContinuousAnalyzeHandover,
    "H3ContinuousTrim": H3ContinuousTrim,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ContinuousStartV11": "H3 Infinite - Start FFLF v1.2",
    "H3ContinuousContinueV11": "H3 Infinite - Continue from Latent v1.2",
    "H3ContinuousAnalyzeHandoverV11": "H3 Infinite - Auto Handover v1.2",
    "H3ContinuousStitchOutputV11": "H3 Infinite - Output / Stitch v1.2",
    "H3ContinuousSeamlessJoinV11": "H3 Infinite - Seamless AV Join v1.2",
    "H3ContinuousStitchSavedChainV11": "H3 Infinite - Stitch Saved Chain v1.2",
    "H3ContinuousSaveLatent": "H3 Infinite - Save AV Latent",
    "H3ContinuousLoadLatent": "H3 Infinite - Load AV Latent",
    "H3ContinuousLatentInfo": "H3 Infinite - Latent Info",
    "H3ContinuousStartV1": "H3 Continuous - Start FFLF (Legacy v1.0)",
    "H3ContinuousContinueV1": "H3 Continuous - Continue from Latent (Legacy v1.0)",
    "H3ContinuousAnalyzeHandoverV1": "H3 Continuous - Auto Handover Analyzer (Legacy v1.0)",
    "H3ContinuousStitchOutputV1": "H3 Continuous - Output / Stitch (Legacy v1.0)",
    "H3ContinuousStart": "H3 Continuous - Start FFLF (Legacy v0.x)",
    "H3ContinuousContinue": "H3 Continuous - Continue from Latent (Legacy v0.x)",
    "H3ContinuousAnalyzeHandover": "H3 Continuous - Auto Handover Analyzer (Legacy v0.x)",
    "H3ContinuousTrim": "H3 Continuous - Trim Rendered Output (Legacy)",
}
