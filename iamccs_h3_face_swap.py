"""SAM3 tracked subject swap: lazy crop/inpaint/uncrop branch for universal H3."""
from __future__ import annotations

import json

import folder_paths
import torch

FACE_SWAP_MODE = "v2va_face_swap"
FACE_SWAP_RESOURCE = "iamccs_h3_face_swap_source"
FACE_SWAP_LATENT = "iamccs_h3_face_swap_crop"


def settings_schema():
    checkpoints = [name for name in folder_paths.get_filename_list("checkpoints") if "sam3" in name.lower()]
    sam3 = next(
        (name for name in checkpoints if "multiplex" in name.lower()),
        checkpoints[0] if checkpoints else "",
    )
    background_models = [name for name in folder_paths.get_filename_list("background_removal") if name.lower().endswith(".safetensors")]
    birefnet = next((name for name in background_models if "birefnet" in name.lower()), background_models[0] if background_models else "")
    return {
        "h3_faceswap_sam_model": (["", *checkpoints], {"default": sam3, "tooltip": "Installed SAM3 multiplex checkpoint. Required only when no source mask is connected."}),
        "h3_faceswap_birefnet_model": (["", *background_models], {"default": birefnet, "tooltip": "BiRefNet model used by FACE SWAP v1 to place both identity views on white before stitching them."}),
        "h3_faceswap_mask_prompt": ("STRING", {"default": "head", "tooltip": "Tracked identity region. Keep head for hair, ears, jaw and profile stability."}),
        "h3_faceswap_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Foxydits reference threshold for stable single-subject SAM3 tracking."}),
        "h3_faceswap_objects": ("STRING", {"default": "", "tooltip": "Leave empty for one subject. For multi-person footage, enter the intended tracked-object index explicitly."}),
        "h3_faceswap_cleanup_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
        "h3_faceswap_cleanup_shrink": ("INT", {"default": 12, "min": 1, "max": 128, "step": 1}),
        "h3_faceswap_cleanup_min_frames": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
        "h3_faceswap_cleanup_edge_grow": ("INT", {"default": 16, "min": 0, "max": 128, "step": 1}),
        "h3_faceswap_crop_scale": ("FLOAT", {"default": 1.75, "min": 1.0, "max": 4.0, "step": 0.05}),
        "h3_faceswap_crop_megapixels": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 4.0, "step": 0.05, "tooltip": "Minimum crop sampling area; 0 preserves the tracked crop size. Editable workload, not a GPU guarantee."}),
        "h3_faceswap_grow_spatial": ("INT", {"default": 36, "min": 0, "max": 256, "step": 1}),
        "h3_faceswap_grow_temporal": ("INT", {"default": 1, "min": 0, "max": 64, "step": 1}),
        "h3_faceswap_feather": ("INT", {"default": 16, "min": 0, "max": 256, "step": 1}),
    }


def face_swap_settings(named):
    return {name.removeprefix("h3_faceswap_"): named.get(name, spec[1]["default"]) for name, spec in settings_schema().items()}


def validate_plan(plan):
    if plan.get("task_mode") != FACE_SWAP_MODE:
        return
    if plan.get("upscale_enabled") and plan.get("upscale_mode", "off") not in {"off", "rtx_final"}:
        raise ValueError("Face Swap restores a cropped latent into full frames. Select native output or RTX final delivery; H3/LTX latent refinement requires a matching full-frame latent.")
    if plan.get("face_detailer_enabled"):
        raise ValueError("Face Swap already owns the masked identity edit. Disable the separate face detailer for this mode.")


def _mvex(name):
    import nodes
    cls = nodes.NODE_CLASS_MAPPINGS.get(name)
    if cls is None:
        raise ValueError(f"Face Swap requires MaskVidExperiments ({name}). Install the provided dependency and restart ComfyUI.")
    return cls


class IAMCCS_H3FaceSwapInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "cine_linx": ("IAMCCS_SUPERNODE_LINX",),
            "source_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0}),
        }, "optional": {"source_video": ("IMAGE", {"lazy": True}), "reference_face": ("IMAGE", {"lazy": True}),
                         "source_audio": ("AUDIO", {"lazy": True}), "source_mask": ("MASK", {"lazy": True}),
                         "reference_face_2": ("IMAGE", {"lazy": True})}}

    RETURN_TYPES = ("IAMCCS_SUPERNODE_LINX",)
    RETURN_NAMES = ("cine_linx",)
    FUNCTION = "attach"
    CATEGORY = "IAMCCS/MiniMax H3/Face Swap"

    def check_lazy_status(self, cine_linx, source_fps=24.0, **kwargs):
        from .iamccs_minimax_h3_atomic_backend import _resolve_shotplan
        if _resolve_shotplan(cine_linx).get("task_mode") != FACE_SWAP_MODE:
            return []
        return [name for name in ("source_video", "reference_face") if kwargs.get(name) is None]

    def attach(self, cine_linx, source_fps, source_video=None, reference_face=None, source_audio=None, source_mask=None, reference_face_2=None):
        from .iamccs_minimax_h3_atomic_backend import _resolve_shotplan
        from .iamccs_supernodes_linx import build_stage_linx_payload
        plan = _resolve_shotplan(cine_linx)
        if plan.get("task_mode") != FACE_SWAP_MODE:
            return (cine_linx,)
        validate_plan(plan)
        if not torch.is_tensor(source_video) or not torch.is_tensor(reference_face) or source_video.ndim != 4 or not len(source_video) or reference_face.ndim != 4 or not len(reference_face):
            raise ValueError("SAM3 Subject Swap requires a non-empty source video and one reference identity image.")
        if source_mask is not None and (source_mask.ndim != 3 or len(source_mask) != len(source_video)):
            raise ValueError("Face Swap source masks must contain one mask for every source video frame.")
        for name in ("MVEx_SubjectCrop", "MVEx_SubjectUncrop", "MVEx_MaskCleanup", "MVEx_MaskToLatentSpace"):
            _mvex(name)
        config = plan.get("face_swap", {})
        reference_mode = "two_view_birefnet_legacy" if (
            torch.is_tensor(reference_face_2) and reference_face_2.ndim == 4 and len(reference_face_2)
        ) else "single_reference_sam3"
        if source_mask is None and not folder_paths.get_full_path("checkpoints", config.get("sam_model", "")):
            raise ValueError("Select an installed SAM3 checkpoint in Face Swap settings, or connect source_mask.")
        if reference_mode == "two_view_birefnet_legacy" and not folder_paths.get_full_path("background_removal", config.get("birefnet_model", "")):
            raise ValueError("Two-view BiRefNet legacy mode requires its model in models/background_removal.")
        data = {"video": source_video, "fps": float(source_fps), "reference": reference_face,
                "reference_2": reference_face_2, "audio": source_audio, "mask": source_mask}
        return (build_stage_linx_payload(cine_linx, stage_name="H3 Face Swap input", stage_kind="minimax_h3_face_swap",
                payload={"source_frames": len(source_video), "source_fps": source_fps},
                report="SAM3 Subject Swap source · lazy single-reference tracked branch", resources={FACE_SWAP_RESOURCE: data}),)


def _build_white_multiview_reference(reference_a, reference_b, model_name):
    """Reproduce the source workflow: BiRefNet cut-outs on white, stitched right."""
    import comfy.model_management as mm
    from comfy_extras.nodes_bg_removal import LoadBackgroundRemovalModel, RemoveBackground
    from comfy_extras.nodes_images import ImageStitch

    bg_model = LoadBackgroundRemovalModel.execute(model_name)[0]
    # BiRefNet's native 1024px pass can require several GB of temporary
    # attention memory.  A Ref2VA/H3 model may already occupy most of a
    # 12--16 GB card, so keep this identity-card step deterministic instead
    # of letting it trigger an avoidable CUDA OOM.  On a roomy card the
    # native ComfyUI device is retained for speed; otherwise the two still
    # images are processed on CPU and the result is returned to the normal
    # intermediate device.
    try:
        free_vram = mm.get_free_memory()
    except Exception:
        free_vram = 0
    try:
        total_vram = torch.cuda.get_device_properties(mm.get_torch_device()).total_memory
    except Exception:
        total_vram = 0
    # The 1024px BiRefNet backbone can peak near 11 GB on its own.  Reserve
    # GPU execution for cards with a genuine 24 GB class budget; this keeps
    # the public 12/16 GB presets reliable while preserving the fast path on
    # workstation cards.
    if total_vram < 24 * 1024 ** 3 or free_vram < 8 * 1024 ** 3:
        cpu = torch.device("cpu")
        bg_model.load_device = cpu
        if hasattr(bg_model, "offload_device"):
            bg_model.offload_device = cpu
        if hasattr(bg_model, "patcher"):
            bg_model.patcher.load_device = cpu
            bg_model.patcher.offload_device = cpu
        if hasattr(bg_model, "model"):
            bg_model.model.to(cpu)

    def on_white(image):
        image = image[:1]
        foreground = RemoveBackground.execute(bg_model, image)[0].to(device=image.device, dtype=image.dtype)
        foreground = foreground.clamp(0, 1)[..., None]
        return image * foreground + torch.ones_like(image) * (1.0 - foreground)

    card_a = on_white(reference_a)
    card_b = on_white(reference_b)
    stitched = ImageStitch.execute(card_a, "right", True, 0, "white", card_b)[0]
    del bg_model, card_a, card_b
    return stitched


def prepare_face_swap(model, clip, video_vae, audio_vae, cine_linx, segment_index, prompt_override=""):
    import comfy.model_management as mm
    import nodes
    from comfy_extras.nodes_sam3 import SAM3_VideoTrack, SAM3_TrackToMask
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
    from .iamccs_minimax_h3_atomic_backend import _resolve_shotplan, _run_h3_conditioning_with_cpu_fallback
    from .iamccs_minimax_h3_v2v_backend import _shotplan_chunk, _requested_frames, _segment_source_start, _frame_indices, _select_frames, _fit_frames, _slice_audio
    from .iamccs_minimax_h3_shotboard_core import align_h3_frames
    from .iamccs_minimax_h3_audio_drive import _lock_audio_stream

    plan = _resolve_shotplan(cine_linx)
    validate_plan(plan)
    config = plan.get("face_swap", {})
    source = cine_linx.get("resources", {}).get(FACE_SWAP_RESOURCE)
    if not source:
        raise ValueError("FACE SWAP mode requires IAMCCS H3 Face Swap Input between Shotboard and the atomic backend.")
    chunk = _shotplan_chunk(plan, segment_index)
    requested = _requested_frames(chunk)
    aligned = align_h3_frames(requested)
    v2v = plan.get("v2v", {})
    start = _segment_source_start(plan, chunk, segment_index, v2v.get("source_range_policy", "timeline_segment"), float(v2v.get("source_offset_seconds", 0)))
    indices, index_report = _frame_indices(source_frames=len(source["video"]), source_fps=source["fps"], start_seconds=start,
            requested_frames=requested, aligned_frames=aligned, end_policy=v2v.get("source_end_policy", "hold_last_for_grid"))
    raw = _fit_frames(_select_frames(source["video"], indices, "face_swap_source"), int(plan["width"]), int(plan["height"]), "canvas_pad").cpu()
    if source["mask"] is not None:
        mask_images = _select_frames(source["mask"][..., None].repeat(1, 1, 1, 3), indices, "face_swap_mask")
        masks = _fit_frames(mask_images, int(plan["width"]), int(plan["height"]), "canvas_pad")[..., 0].cpu()
    else:
        checkpoint = str(config.get("sam_model", ""))
        sam_model, sam_clip, _ = nodes.CheckpointLoaderSimple().load_checkpoint(checkpoint)
        conditioning = nodes.CLIPTextEncode().encode(sam_clip, str(config.get("mask_prompt", "head")))[0]
        tracks = SAM3_VideoTrack.execute(images=raw, model=sam_model, conditioning=conditioning,
                detection_threshold=float(config.get("threshold", 0.5)),
                max_objects=1, detect_interval=1)[0]
        masks = SAM3_TrackToMask.execute(track_data=tracks, object_indices=str(config.get("objects", "")))[0].cpu()
        del sam_model, sam_clip, conditioning, tracks
        mm.unload_all_models()
        mm.soft_empty_cache()
    masks = _mvex("MVEx_MaskCleanup").execute(masks=masks, threshold=float(config.get("cleanup_threshold", 0.3)),
            method={"method": "shrink_grow", "shrink": int(config.get("cleanup_shrink", 12)),
                    "min_frames": int(config.get("cleanup_min_frames", 4))},
            edge_grow=int(config.get("cleanup_edge_grow", 16)))[0]
    if not torch.any(masks > 0):
        raise ValueError("Face Swap mask is empty. Adjust the SAM3 prompt, object indices or threshold; no render was started.")
    crops, crop_masks, boxes, *_ = _mvex("MVEx_SubjectCrop").execute(original_images=raw, masks=masks,
            mode={"mode": "tracked", "crop_scale": float(config.get("crop_scale", 1.75)), "padding": "firm", "prefer": "stillness", "aspect_ratio": 0.0, "seamless_loop": False},
            divisible_by=32, upscale_megapixels=float(config.get("crop_megapixels", 0.5)))
    width, height = int(crops.shape[2]), int(crops.shape[1])
    reference_mode = "two_view_birefnet_legacy" if source.get("reference_2") is not None else "single_reference_sam3"
    if reference_mode == "two_view_birefnet_legacy":
        # Compatibility path only. The default SAM3 workflow uses Picture 1
        # directly and therefore has no BiRefNet dependency or second image.
        identity_card = source.get("identity_card")
        if identity_card is None:
            identity_card = _build_white_multiview_reference(source["reference"], source["reference_2"],
                    str(config.get("birefnet_model", "")))
            source["identity_card"] = identity_card
        reference = identity_card
    else:
        reference = source["reference"][:1]
    ref_images = {"ref_image_1": reference}
    directed_prompt = str(prompt_override or chunk.get("prompt", "")).strip()
    prompt = "<Subject 1> is the character represented in <Picture 1>. The video is a close up face of <Subject 1>."
    if directed_prompt:
        prompt += " " + directed_prompt
    ref_audios = None
    sliced_source_audio = None
    if plan.get("audio_mode") == "h3_custom_audio_drive" and isinstance(source.get("audio"), dict):
        # Ref2VA must hear the exact same timeline slice that is later locked
        # into this chunk. Passing the full programme here makes chunk 2+ hear
        # the opening phonemes again even though the output audio is sliced.
        sliced_source_audio = _slice_audio(
            source["audio"],
            start_seconds=start,
            requested_frames=requested,
            aligned_frames=aligned,
        )
        sliced_source_audio = {
            **sliced_source_audio,
            "iamccs_pre_sliced": True,
            "iamccs_source_start_seconds": start,
        }
        ref_audios = {"ref_audio_1": sliced_source_audio}
        prompt += " Use <Audio 1> as the timing authority for mouth, jaw and facial articulation; preserve the source performance and do not invent dialogue."
    result, encoder_report = _run_h3_conditioning_with_cpu_fallback(clip, plan,
        lambda active_clip: MiniMaxH3ReferenceToVideo.execute(clip=active_clip, vae=video_vae, audio_vae=audio_vae,
            prompt=prompt, width=width, height=height, length=aligned, ref_image_size="match", ref_images=ref_images,
            ref_audios=ref_audios))
    positive, empty_av = result[0], result[1]
    encoded = nodes.VAEEncode().encode(video_vae, crops)[0]
    mask = _mvex("MVEx_MaskToLatentSpace").execute(masks=crop_masks, compression={"compression": "auto"},
            spatial_method="max", temporal_method="max", grow_spatial=int(config.get("grow_spatial", 36)),
            grow_temporal=int(config.get("grow_temporal", 1)), vae=video_vae)[0]
    from comfy_extras.nodes_lt import LTXVConcatAVLatent
    # The existing H3 audio lock preserves the video stream's supplied noise mask.
    video = nodes.SetLatentNoiseMask().set_mask(encoded, mask)[0]
    audio_stream = {"samples": empty_av["samples"].unbind()[1]}
    latent = LTXVConcatAVLatent.execute(video_latent=video, audio_latent=audio_stream)[0]
    original_audio = None
    if plan.get("audio_mode") == "h3_custom_audio_drive":
        if sliced_source_audio is not None:
            original_audio = sliced_source_audio
        elif source.get("audio") is not None:
            original_audio = _slice_audio(
                source["audio"],
                start_seconds=start,
                requested_frames=requested,
                aligned_frames=aligned,
            )
            original_audio = {
                **original_audio,
                "iamccs_pre_sliced": True,
                "iamccs_source_start_seconds": start,
            }
        else:
            original_audio = {"waveform": torch.zeros(1, 2, round(aligned / 24 * 32000)), "sample_rate": 32000}
            original_audio = {
                **original_audio,
                "iamccs_pre_sliced": True,
                "iamccs_source_start_seconds": start,
            }
        latent, _ = _lock_audio_stream(latent, original_audio, audio_vae)
    latent[FACE_SWAP_LATENT] = {"original": raw, "masks": crop_masks.cpu(), "boxes": boxes,
            "requested": requested, "audio": original_audio, "feather": int(config.get("feather", 16))}
    identity_report = "BiRefNet 2-view legacy card" if reference_mode == "two_view_birefnet_legacy" else "single Picture 1 reference"
    report = f"SAM3 SUBJECT SWAP · tracked crop/inpaint/uncrop | SAM3 {config.get('mask_prompt', 'head')}@{float(config.get('threshold', 0.5)):.2f} max=1 interval=1 | {identity_report} | source={plan['width']}x{plan['height']} | crop={width}x{height} | frames={requested}/{aligned} | encoder={encoder_report}"
    return (model, positive, latent, raw[:1], raw[-1:], json.dumps({"task": FACE_SWAP_MODE, "source": index_report}),
            prompt, int(segment_index), len(plan["chunks"]), 0, report, {"active": False})


def restore_face_swap(images, audio, latent):
    data = latent.get(FACE_SWAP_LATENT)
    if not data:
        raise ValueError("Face Swap sampling lost its crop metadata; refusing to export an unplaced face crop.")
    if len(images) != len(data["original"]):
        raise ValueError("Face Swap decode frame count differs from the source crop contract.")
    # FACE SWAP v1 leaves cropped_masks disconnected here and
    # feathers the tracked crop rectangle. Keep that contract for 1:1 parity.
    output = _mvex("MVEx_SubjectUncrop").execute(cropped_images=images, original_images=data["original"],
                bboxes=data["boxes"], cropped_masks=None, feather=data["feather"])[0]
    count = int(data["requested"])
    original_audio = data.get("audio") or audio
    if original_audio is not None:
        original_audio = {**original_audio, "waveform": original_audio["waveform"][..., :round(count / 24 * original_audio["sample_rate"])]}
    return output[:count], original_audio


NODE_CLASS_MAPPINGS = {"IAMCCS_H3FaceSwapInput": IAMCCS_H3FaceSwapInput}
NODE_DISPLAY_NAME_MAPPINGS = {"IAMCCS_H3FaceSwapInput": "SAM3 SUBJECT SWAP · tracked crop + H3 inpaint"}
