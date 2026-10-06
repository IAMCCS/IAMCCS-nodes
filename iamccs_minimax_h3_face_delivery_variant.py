# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional delivery-only FaceRefine. Native motion state is committed upstream."""
import copy
import logging
import math
import re

import torch

from .iamccs_minimax_h3_atomic_backend import SUPERNODE_LINX_TYPE, _resolve_shotplan
from .iamccs_minimax_h3_face_detailer import _face_node_class

LOG = logging.getLogger("IAMCCS.MiniMaxH3.FaceDeliveryR38B")


def _defaults(node_class):
    values = {}
    for name, spec in node_class.INPUT_TYPES()["required"].items():
        kind, options = spec[0], spec[1] if len(spec) > 1 else {}
        if "default" in options:
            values[name] = options["default"]
        elif isinstance(kind, list):
            values[name] = kind[0]
    return values


def _unpack_track_result(track_class, track_result):
    """Read FaceRefine tracker outputs across the historical 6/7-output contracts."""
    if not isinstance(track_result, (tuple, list)):
        raise RuntimeError(
            "Unsupported H3FaceTrackCrop result: expected tuple/list, "
            f"got {type(track_result).__name__}."
        )
    names = tuple(getattr(track_class, "RETURN_NAMES", ()) or ())
    if names and len(names) == len(track_result):
        outputs = dict(zip(names, track_result))
        required = ("crops", "transform", "canvas_w", "canvas_h")
        missing = [name for name in required if name not in outputs]
        if missing:
            raise RuntimeError(
                "Unsupported H3FaceTrackCrop output contract; missing: "
                + ", ".join(missing)
            )
        return (
            outputs["crops"], outputs["transform"],
            int(outputs["canvas_w"]), int(outputs["canvas_h"]),
            str(outputs.get("report", "") or ""),
        )
    if len(track_result) < 6:
        raise RuntimeError(
            "Unsupported H3FaceTrackCrop result: expected at least 6 outputs, "
            f"got {len(track_result)}."
        )
    return (
        track_result[0], track_result[1],
        int(track_result[4]), int(track_result[5]),
        str(track_result[3] if len(track_result) > 3 else ""),
    )




def _face_setting(face, key, default, cast=None, minimum=None, maximum=None):
    value = face.get(key, default)
    try:
        value = cast(value) if cast is not None else value
    except Exception:
        value = default
    if minimum is not None and value < minimum:
        value = minimum
    if maximum is not None and value > maximum:
        value = maximum
    return value


def _run_per_frame_denoise(node_class, model, av_latent, transform, small_strength, large_strength):
    """Bridge H3PerFrameDenoise across FaceRefine 1.0.x and 1.1+ contracts.

    1.1+ renamed the strength inputs to denoise_multiplier_* and moved the node into
    the MODEL path. Older releases accepted only the latent/transform path. Return a
    stable (latent, model, report) tuple so the IAMCCS sampling path can route the
    patched model to both BasicScheduler and BasicGuider when the provider exposes it.
    """
    contract = node_class.INPUT_TYPES()
    supported = set(contract.get("required", {})) | set(contract.get("optional", {}))
    kwargs = _defaults(node_class)
    kwargs["av_latent"] = av_latent
    kwargs["transform"] = transform
    if "model" in supported:
        kwargs["model"] = model

    if "denoise_multiplier_small_face" in supported:
        kwargs["denoise_multiplier_small_face"] = float(small_strength)
    elif "strength_small_face" in supported:
        kwargs["strength_small_face"] = float(small_strength)

    if "denoise_multiplier_large_face" in supported:
        kwargs["denoise_multiplier_large_face"] = float(large_strength)
    elif "strength_large_face" in supported:
        kwargs["strength_large_face"] = float(large_strength)

    result = node_class().run(**kwargs)
    if not isinstance(result, (tuple, list)):
        raise RuntimeError(
            "Unsupported H3PerFrameDenoise result: expected tuple/list, "
            f"got {type(result).__name__}."
        )

    names = tuple(getattr(node_class, "RETURN_NAMES", ()) or ())
    if names and len(names) == len(result):
        outputs = dict(zip(names, result))
        latent_out = outputs.get("av_latent", result[0] if result else av_latent)
        model_out = outputs.get("model", model)
        report = str(outputs.get("report", "") or "")
        return latent_out, model_out, report

    # Historical provider: first output is the latent and no model patch was returned.
    latent_out = result[0] if result else av_latent
    report = str(result[1] if len(result) > 1 and isinstance(result[1], str) else "")
    model_out = result[2] if len(result) > 2 else model
    return latent_out, model_out, report


def _reported_max_faces(report):
    match = re.search(r"faces:\s*max\s+(\d+)\s+in one frame", str(report or ""), re.I)
    return max(1, int(match.group(1))) if match else 1


def _track_coverage(transform):
    """Fraction of source frames with a real detector hit for this tracked subject."""
    if not isinstance(transform, dict):
        return 1.0
    detected = transform.get("detected") or []
    source_frames = max(1, int(transform.get("source_frames", len(detected) or 1)))
    if not detected:
        return 1.0
    return min(1.0, sum(bool(v) for v in detected) / float(source_frames))


def _median(values):
    if not values:
        return float("inf")
    values = sorted(float(v) for v in values)
    n = len(values)
    m = n // 2
    return values[m] if n % 2 else 0.5 * (values[m - 1] + values[m])


def _same_subject_track(a, b):
    """Conservative duplicate guard for multi-face passes.

    select_index is anchored on the first frame that contains that rank. In shots where
    people enter progressively, two indices can occasionally converge on the same person.
    Compare source-space crop trajectories and skip only near-identical tracks.
    """
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False

    def points(t):
        boxes = list(t.get("boxes") or [])
        frames = list(t.get("source") or range(len(boxes)))
        weights = list(t.get("weights") or [1.0] * len(boxes))
        out = {}
        for i, box in enumerate(boxes):
            if i >= len(frames) or box is None or len(box) < 4:
                continue
            if i < len(weights) and float(weights[i]) <= 0.05:
                continue
            x, y, w, h = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            if w <= 0.0 or h <= 0.0:
                continue
            out[int(frames[i])] = (x + 0.5 * w, y + 0.5 * h, w, h)
        return out

    pa, pb = points(a), points(b)
    common = sorted(set(pa).intersection(pb))
    if len(common) < 5:
        return False
    distances, size_ratios = [], []
    for frame in common:
        ax, ay, aw, ah = pa[frame]
        bx, by, bw, bh = pb[frame]
        scale = max(1.0, 0.5 * (math.sqrt(aw * ah) + math.sqrt(bw * bh)))
        distances.append(math.hypot(ax - bx, ay - by) / scale)
        area_a, area_b = aw * ah, bw * bh
        size_ratios.append(max(area_a, area_b) / max(1e-6, min(area_a, area_b)))
    # Deliberately strict: neighbouring faces often have overlapping crop boxes, but the
    # same tracked face has almost the same centre trajectory and crop scale.
    return _median(distances) < 0.16 and _median(size_ratios) < 1.18


class IAMCCS_MiniMaxH3FaceDeliveryR38B:
    @classmethod
    def INPUT_TYPES(cls):
        track = _face_node_class("H3FaceTrackCrop").INPUT_TYPES()["required"]
        stitch = _face_node_class("H3FaceStitch").INPUT_TYPES()["required"]
        return {"required": {
            "model": ("MODEL",), "video_vae": ("VAE",), "sampled_latent": ("LATENT",),
            "conditioning": ("CONDITIONING",), "cine_linx": (SUPERNODE_LINX_TYPE,),
            "native_frames": ("IMAGE",), "native_saved_report": ("STRING", {"forceInput": True}),
            "context_trim_frames": ("INT", {"forceInput": True}), "segment_index": ("INT", {"forceInput": True}),
            "detector": track["detector"], "canvas_width": track["canvas_width"], "canvas_height": track["canvas_height"],
            "crop_factor": track["crop_factor"], "confidence": track["confidence"],
            "steps": ("INT", {"default":4,"min":1,"max":100}),
            "denoise": ("FLOAT", {"default":0.2,"min":0.0,"max":1.0,"step":0.01}),
            "window_frames": ("INT", {"default":73,"min":34,"max":510}),
            "window_overlap": ("INT", {"default":22,"min":0,"max":170}),
            "strength_small_face": ("FLOAT", {"default":1.0,"min":0.0,"max":1.0,"step":0.01}),
            "strength_large_face": ("FLOAT", {"default":0.35,"min":0.0,"max":1.0,"step":0.01}),
            "blend": ("FLOAT", {"default":0.7,"min":0.0,"max":1.0,"step":0.01}),
            "paste_region": stitch["paste_region"], "feather": stitch["feather"],
        }, "optional": {
            "sam_model": ("SAM_MODEL", {"lazy":True}),
            "face_mode": (["from_shotboard", "single", "multi_face"], {"default":"from_shotboard"}),
            "max_faces": ("INT", {"default":4,"min":1,"max":8,"step":1}),
            "min_face_coverage": ("FLOAT", {"default":0.10,"min":0.0,"max":1.0,"step":0.01}),
        }}

    RETURN_TYPES = ("LATENT", "IMAGE", "STRING")
    RETURN_NAMES = ("delivery_latent", "delivery_frames", "report")
    FUNCTION = "refine"
    CATEGORY = "IAMCCS/MiniMax H3/Pixel Refine R38B"

    def check_lazy_status(self, cine_linx, sam_model=None, **kwargs):
        face = _resolve_shotplan(cine_linx).get("face_detailer_settings", {})
        if face.get("enabled") and face.get("use_sam_mask") and sam_model is None:
            return ["sam_model"]
        return []

    def refine(self, model, video_vae, sampled_latent, conditioning, cine_linx, native_frames,
               native_saved_report, context_trim_frames, segment_index, detector, canvas_width,
               canvas_height, crop_factor, confidence, steps, denoise, window_frames, window_overlap,
               strength_small_face, strength_large_face, blend, paste_region, feather, sam_model=None,
               face_mode="from_shotboard", max_faces=4, min_face_coverage=0.10):
        face = _resolve_shotplan(cine_linx).get("face_detailer_settings", {})
        if not face.get("enabled", False):
            return sampled_latent, native_frames, "Face delivery OFF: native inputs untouched; no models loaded."
        if not native_saved_report:
            raise ValueError("Face delivery must follow native save and motion-state commit.")
        if face.get("use_sam_mask") and sam_model is None:
            raise ValueError("Face SAM is enabled; connect the optional SAMLoader or choose FACE ON without SAM.")
        profile = str(face.get("profile", "balanced") or "balanced").strip().lower()
        requested_mode = str(face_mode or "from_shotboard").strip().lower()
        if requested_mode not in {"from_shotboard", "single", "multi_face"}:
            requested_mode = "from_shotboard"

        # R44 Settings/Settings PRO is authoritative when the delivery node is left
        # on FROM SHOTBOARD. ``auto_from_profile`` keeps old workflows byte-compatible:
        # the existing MULTI FACE profile enables the sequential path, other profiles
        # remain single-subject. Explicit node overrides still win when selected.
        if requested_mode == "from_shotboard":
            settings_mode = str(face.get("mode", "auto_from_profile") or "auto_from_profile").strip().lower()
            if settings_mode not in {"auto_from_profile", "single", "multi_face"}:
                settings_mode = "auto_from_profile"
            resolved_mode = (
                "multi_face" if profile == "multi_face" else "single"
            ) if settings_mode == "auto_from_profile" else settings_mode
            configured_faces = face.get("max_faces", max_faces)
            configured_coverage = face.get("min_face_coverage", min_face_coverage)
        else:
            resolved_mode = requested_mode
            configured_faces = max_faces
            configured_coverage = min_face_coverage
        requested_faces = max(1, min(8, int(configured_faces))) if resolved_mode == "multi_face" else 1
        min_face_coverage = max(0.0, min(1.0, float(configured_coverage)))

        detector = str(face.get("detector", detector) or detector)
        canvas_width = _face_setting(face, "canvas_width", canvas_width, int, 256, 1536)
        canvas_height = _face_setting(face, "canvas_height", canvas_height, int, 256, 1536)
        crop_factor = _face_setting(face, "crop_factor", crop_factor, float, 1.0, 6.0)
        confidence = _face_setting(face, "confidence", confidence, float, 0.01, 1.0)
        steps = _face_setting(face, "steps", steps, int, 1, 100)
        denoise = _face_setting(face, "denoise", denoise, float, 0.0, 1.0)
        window_frames = _face_setting(face, "window_frames", window_frames, int, 34, 510)
        window_overlap = _face_setting(face, "window_overlap", window_overlap, int, 0, 170)
        strength_small_face = _face_setting(face, "strength_small_face", strength_small_face, float, 0.0, 1.0)
        strength_large_face = _face_setting(face, "strength_large_face", strength_large_face, float, 0.0, 1.0)
        blend = _face_setting(face, "blend", blend, float, 0.0, 1.0)
        paste_region = str(face.get("paste_region", paste_region) or paste_region)
        feather = _face_setting(face, "feather", feather, int, 0, 128)
        if window_overlap >= window_frames or canvas_width % 32 or canvas_height % 32:
            raise ValueError("Face canvas must be a multiple of 32; overlap must be smaller than the window.")

        import comfy.model_management as mm
        import nodes
        from comfy_extras.nodes_minimax_h3 import EmptyMiniMaxH3LatentAV
        from .iamccs_minimax_h3_pixel_refine_variant import _provider, _refine

        common = _provider("common")
        source_v, source_a = common.unpack_av(sampled_latent, "face source")
        source_count = common.latents_to_frames(source_v.shape[2])
        trim, visible = int(context_trim_frames), len(native_frames)
        if trim < 0 or trim + visible > source_count:
            raise ValueError("Face delivery received an inconsistent native context trim.")
        LOG.info(
            "R38B Face delivery | native context retained | %df | canvas=%dx%d | steps=%d | "
            "denoise=%.3f | mode=%s | max_faces=%d | min_coverage=%.2f",
            source_count, canvas_width, canvas_height, steps, denoise, resolved_mode, requested_faces, min_face_coverage,
        )
        mm.unload_all_models()
        mm.soft_empty_cache()
        try:
            # R43 face isolation: decode native latent but never feed LongVid/Motion-Context
            # technical head/padding into the second H3 FaceRefine pass.
            source_full = nodes.VAEDecode().decode(vae=video_vae, samples=sampled_latent)[0]
            source = source_full[trim:trim + visible].clone()
            if len(source) != visible:
                raise ValueError("Face delivery could not isolate the native visible frame range.")

            # MiniMax H3 AV latents must live on the 17k+5 temporal grid. Pad ONLY with
            # copies of the final visible frame; synthetic padding is discarded afterwards.
            refine_count = max(5, int(visible))
            while refine_count % 17 != 5:
                refine_count += 1
            face_pad = refine_count - int(visible)
            if face_pad:
                source_refine = torch.cat(
                    (source, source[-1:].expand(face_pad, -1, -1, -1).clone()), dim=0,
                )
            else:
                source_refine = source

            LOG.info(
                "R43 Face visible-only | visible=%df | refine_grid=%df | synthetic_tail_pad=%df",
                visible, refine_count, face_pad,
            )

            track_class = _face_node_class("H3FaceTrackCrop")
            stitch_class = _face_node_class("H3FaceStitch")
            inject_class = _face_node_class("H3InjectVideoLatent")
            per_frame_class = _face_node_class("H3PerFrameDenoise")
            mask_class = _face_node_class("H3FaceMaskSAM") if face.get("use_sam_mask") else None

            face_plan = copy.deepcopy(_resolve_shotplan(cine_linx))
            face_plan.setdefault("upscale_settings", {}).setdefault("h3_latent_upres", {}).update(
                steps=steps, denoise=denoise, window_frames=window_frames, window_overlap=window_overlap
            )

            # Tracking always sees the untouched source. Stitching accumulates onto the
            # returned canvas, but before the first accepted pass we alias source_refine
            # instead of cloning the full frame batch (important on 12 GB systems).
            composite_refine = source_refine
            accepted_transforms = []
            accepted_indices = []
            skipped = []
            cached_first = None
            detector_ceiling = requested_faces

            candidate_index = 0
            while candidate_index < detector_ceiling and len(accepted_indices) < requested_faces:
                if candidate_index == 0 and cached_first is not None:
                    track_result = cached_first
                else:
                    track_contract = track_class.INPUT_TYPES()
                    supported_track_inputs = set(track_contract.get("required", {})) | set(track_contract.get("optional", {}))
                    track_kwargs = {
                        **_defaults(track_class),
                        "images": source_refine,
                        "detector": detector,
                        "canvas_width": canvas_width,
                        "canvas_height": canvas_height,
                        "crop_factor": crop_factor,
                        "confidence": confidence,
                    }
                    if "identity_track" in supported_track_inputs:
                        # deterministic/offline: do not trigger a hidden InsightFace download
                        track_kwargs["identity_track"] = False
                    if "select" in supported_track_inputs:
                        track_kwargs["select"] = "largest_face"
                    if "select_index" in supported_track_inputs:
                        track_kwargs["select_index"] = candidate_index
                    elif candidate_index > 0:
                        raise RuntimeError(
                            "MULTI FACE requires an H3FaceTrackCrop version exposing select_index. "
                            "Update ComfyUI-H3-FaceRefine or switch face_mode to single."
                        )
                    track_result = track_class().run(**track_kwargs)
                    if candidate_index == 0:
                        cached_first = track_result

                crops, transform, cw, ch, track_report = _unpack_track_result(track_class, track_result)
                if candidate_index == 0:
                    detected_max = _reported_max_faces(track_report)
                    detector_ceiling = 1 if resolved_mode == "single" else min(8, detected_max)
                    LOG.info(
                        "R44 Face tracker ceiling | detected_max=%d | requested=%d | candidate_ceiling=%d",
                        detected_max, requested_faces, detector_ceiling,
                    )

                coverage = _track_coverage(transform)
                duplicate_of = next(
                    (accepted_indices[i] for i, previous in enumerate(accepted_transforms)
                     if _same_subject_track(transform, previous)),
                    None,
                )
                if resolved_mode == "multi_face" and coverage < min_face_coverage:
                    skipped.append(f"index {candidate_index}: coverage {coverage:.0%} < {min_face_coverage:.0%}")
                    LOG.info("R44 MultiFace skip index=%d | coverage=%.3f", candidate_index, coverage)
                    del crops, transform
                    candidate_index += 1
                    continue
                if duplicate_of is not None:
                    skipped.append(f"index {candidate_index}: duplicate of index {duplicate_of}")
                    LOG.info("R44 MultiFace skip index=%d | duplicate_of=%d", candidate_index, duplicate_of)
                    del crops, transform
                    candidate_index += 1
                    continue

                mask = None
                if mask_class is not None:
                    mask = mask_class().run(**{
                        **_defaults(mask_class), "crops": crops,
                        "transform": transform, "sam_model": sam_model,
                    })[0]

                # Build one isolated AV latent for this subject. Audio is temporary and
                # discarded; the native audio tensor is restored untouched at delivery.
                template = EmptyMiniMaxH3LatentAV.execute(width=cw, height=ch, length=refine_count)[0]
                template_v, template_a = common.unpack_av(template, "face visible canvas")
                template = common.pack_av(
                    {}, template_v.to(source_v),
                    template_a.to(source_a) if source_a is not None else template_a,
                )
                cropped = inject_class().run(av_latent=template, images=crops, vae=video_vae)[0]
                del template, template_v, template_a
                def _per_frame_adapter(active_model, latent_for_refine):
                    return _run_per_frame_denoise(
                        per_frame_class, active_model, latent_for_refine, transform,
                        strength_small_face, strength_large_face,
                    )

                LOG.info(
                    "R44 Face pass %d/%d | select_index=%d | coverage=%.1f%%",
                    len(accepted_indices) + 1, detector_ceiling, candidate_index, coverage * 100.0,
                )
                mm.unload_all_models()
                mm.soft_empty_cache()
                refined = _refine(
                    model, conditioning, cropped, face_plan, segment_index,
                    pre_scheduler_transform=_per_frame_adapter,
                )
                del cropped
                mm.unload_all_models()
                mm.soft_empty_cache()
                pixels = nodes.VAEDecode().decode(vae=video_vae, samples=refined)[0]
                del refined
                if len(pixels) != refine_count:
                    raise ValueError(
                        f"FaceRefine changed the H3-aligned frame count: expected {refine_count}, "
                        f"got {len(pixels)}."
                    )

                next_composite = stitch_class().run(**{
                    **_defaults(stitch_class),
                    "base_images": composite_refine,
                    "refined_crops": pixels,
                    "transform": transform,
                    "masks": mask,
                    "blend": blend,
                    "paste_region": paste_region,
                    "feather": feather,
                })[0]
                del pixels, crops, mask
                if len(next_composite) != refine_count:
                    raise ValueError(
                        f"Face stitch changed the H3-aligned frame count: expected {refine_count}, "
                        f"got {len(next_composite)}."
                    )
                del composite_refine
                composite_refine = next_composite
                accepted_transforms.append(transform)
                accepted_indices.append(candidate_index)
                candidate_index += 1

            if not accepted_indices:
                # This can only happen in multi-face mode with a deliberately high minimum
                # coverage. Preserve the native master instead of failing a long render.
                stitched = source.clone()
            else:
                stitched = composite_refine[:visible].clone()
            delivery_frames = stitched.clone()

            # The upscaler must receive the stitched video, not the untouched native latent.
            # Restore only the refined VISIBLE pixels into the original full native extent.
            upscale = bool(face_plan.get("upscale_enabled")) and face_plan.get("upscale_mode", "off") != "off"
            if upscale and accepted_indices:
                stitched_full = source_full.clone()
                stitched_full[trim:trim + visible] = stitched
                delivery = inject_class().run(
                    av_latent=sampled_latent, images=stitched_full, vae=video_vae,
                )[0]
                del stitched_full
            else:
                delivery = dict(sampled_latent)

            del stitched, composite_refine, source_refine, source, source_full
            out_v, out_a = common.unpack_av(delivery, "face delivery")
            if out_v.shape != source_v.shape or (
                source_a is not None and not torch.equal(out_a, source_a)
            ):
                raise ValueError("Face delivery changed native AV extent or audio.")

            delivery["iamccs_r38b_face_applied"] = True
            delivery["iamccs_face_mode"] = resolved_mode
            delivery["iamccs_face_subjects"] = len(accepted_indices)
            accepted_desc = ",".join(str(v) for v in accepted_indices) if accepted_indices else "none"
            skipped_desc = "; ".join(skipped) if skipped else "none"
            return delivery, delivery_frames, (
                f"Face delivery complete: mode={resolved_mode}; {visible} visible frames; "
                f"subjects={len(accepted_indices)} (select_index={accepted_desc}); "
                f"tracker ceiling={detector_ceiling}; "
                f"{'single-face fallback; ' if resolved_mode == 'multi_face' and detector_ceiling == 1 else ''}"
                f"skipped={skipped_desc}; "
                f"H3 synthetic tail pad={face_pad}f; native LongVid context/padding excluded; "
                "audio/context timing unchanged."
            )
        finally:
            mm.unload_all_models()
            mm.soft_empty_cache()



class IAMCCS_MiniMaxH3WideCharacterDetailer12GB(IAMCCS_MiniMaxH3FaceDeliveryR38B):
    """Named 12 GB preset for distant visible faces in medium/wide shots.

    This deliberately reuses the audited track/refine/stitch path.  It raises
    crop resolution and lowers detection confidence, but does not pretend to
    be a full-frame or multi-person body restorer.
    """

    CATEGORY = "IAMCCS/MiniMax H3/Detailer"

    @classmethod
    def INPUT_TYPES(cls):
        result = copy.deepcopy(super().INPUT_TYPES())
        required = result["required"]
        required["canvas_width"] = ("INT", {"default": 640, "min": 256, "max": 1536, "step": 32})
        required["canvas_height"] = ("INT", {"default": 640, "min": 256, "max": 1536, "step": 32})
        required["crop_factor"] = ("FLOAT", {"default": 2.2, "min": 1.0, "max": 6.0, "step": 0.05})
        required["confidence"] = ("FLOAT", {"default": 0.25, "min": 0.01, "max": 1.0, "step": 0.01})
        required["steps"] = ("INT", {"default": 4, "min": 1, "max": 100})
        required["denoise"] = ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01})
        required["window_frames"] = ("INT", {"default": 73, "min": 34, "max": 510})
        required["window_overlap"] = ("INT", {"default": 22, "min": 0, "max": 170})
        required["strength_small_face"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})
        required["strength_large_face"] = ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01})
        required["blend"] = ("FLOAT", {"default": 0.72, "min": 0.0, "max": 1.0, "step": 0.01})
        return result
