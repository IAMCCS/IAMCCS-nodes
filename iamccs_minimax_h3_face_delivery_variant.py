# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional delivery-only FaceRefine. Native motion state is committed upstream."""
import copy
import logging

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
        }, "optional": {"sam_model": ("SAM_MODEL", {"lazy":True})}}

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
               strength_small_face, strength_large_face, blend, paste_region, feather, sam_model=None):
        face = _resolve_shotplan(cine_linx).get("face_detailer_settings", {})
        if not face.get("enabled", False):
            return sampled_latent, native_frames, "Face delivery OFF: native inputs untouched; no models loaded."
        if not native_saved_report:
            raise ValueError("Face delivery must follow native save and motion-state commit.")
        if face.get("use_sam_mask") and sam_model is None:
            raise ValueError("Face SAM is enabled; connect the optional SAMLoader or choose FACE ON without SAM.")
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
        LOG.info("R38B Face delivery | native context retained | %df | canvas=%dx%d | steps=%d | denoise=%.3f",
                 source_count, canvas_width, canvas_height, steps, denoise)
        mm.unload_all_models()
        mm.soft_empty_cache()
        try:
            # R43 face isolation:
            # decode the native latent, but NEVER feed LongVid/Motion-Context
            # technical head/padding into the second H3 FaceRefine pass.
            # Native generation remains the sole authority for temporal continuity.
            source_full = nodes.VAEDecode().decode(
                vae=video_vae, samples=sampled_latent
            )[0]
            source = source_full[trim:trim + visible].clone()
            if len(source) != visible:
                raise ValueError(
                    "Face delivery could not isolate the native visible frame range."
                )

            # MiniMax H3 AV latents must live on the 17k+5 temporal grid.
            # Pad ONLY with copies of the final visible frame: never borrow
            # LongVid context/padding from sampled_latent for the FaceRefine pass.
            refine_count = max(5, int(visible))
            while refine_count % 17 != 5:
                refine_count += 1
            face_pad = refine_count - int(visible)
            if face_pad:
                source_refine = torch.cat(
                    (source, source[-1:].expand(face_pad, -1, -1, -1).clone()),
                    dim=0,
                )
            else:
                source_refine = source

            LOG.info(
                "R43 Face visible-only | visible=%df | refine_grid=%df | synthetic_tail_pad=%df",
                visible, refine_count, face_pad,
            )
            track = _face_node_class("H3FaceTrackCrop")
            crops, transform, _, _, cw, ch = track().run(**{
                **_defaults(track), "images":source_refine, "detector":detector,
                "canvas_width":canvas_width, "canvas_height":canvas_height,
                "crop_factor":crop_factor,"confidence":confidence,
                "identity_track":False,  # no hidden InsightFace download
            })
            mask = None
            if face.get("use_sam_mask"):
                mask_class = _face_node_class("H3FaceMaskSAM")
                mask = mask_class().run(**{**_defaults(mask_class),"crops":crops,
                    "transform":transform,"sam_model":sam_model})[0]
            # Build an isolated AV latent whose temporal extent is ONLY the
            # visible delivery range. Its temporary audio is discarded later.
            template = EmptyMiniMaxH3LatentAV.execute(
                width=cw, height=ch, length=refine_count
            )[0]
            template_v, template_a = common.unpack_av(template, "face visible canvas")
            template = common.pack_av(
                {}, template_v.to(source_v),
                template_a.to(source_a) if source_a is not None else template_a
            )
            inject = _face_node_class("H3InjectVideoLatent")
            cropped = inject().run(av_latent=template,images=crops,vae=video_vae)[0]
            del template, template_v, template_a
            per_frame = _face_node_class("H3PerFrameDenoise")
            cropped = per_frame().run(**{**_defaults(per_frame),"av_latent":cropped,
                "transform":transform,"strength_small_face":strength_small_face,
                "strength_large_face":strength_large_face})[0]
            face_plan = copy.deepcopy(_resolve_shotplan(cine_linx))
            face_plan.setdefault("upscale_settings", {}).setdefault("h3_latent_upres", {}).update(
                steps=steps,denoise=denoise,window_frames=window_frames,window_overlap=window_overlap)
            mm.unload_all_models()
            mm.soft_empty_cache()
            refined = _refine(model,conditioning,cropped,face_plan,segment_index)
            del cropped
            mm.unload_all_models()
            mm.soft_empty_cache()
            pixels = nodes.VAEDecode().decode(vae=video_vae,samples=refined)[0]
            del refined
            if len(pixels) != refine_count:
                raise ValueError(
                    f"FaceRefine changed the H3-aligned frame count: "
                    f"expected {refine_count}, got {len(pixels)}."
                )

            # Stitch on the same H3-legal padded range so tracker transforms,
            # optional SAM masks and refined crops remain frame-aligned.
            stitch = _face_node_class("H3FaceStitch")
            stitched_refine = stitch().run(**{
                **_defaults(stitch),
                "base_images": source_refine,
                "refined_crops": pixels,
                "transform": transform,
                "masks": mask,
                "blend": blend,
                "paste_region": paste_region,
                "feather": feather,
            })[0]
            del pixels, crops, mask

            if len(stitched_refine) != refine_count:
                raise ValueError(
                    f"Face stitch changed the H3-aligned frame count: "
                    f"expected {refine_count}, got {len(stitched_refine)}."
                )

            # Synthetic tail exists only to satisfy H3's 17k+5 grid.
            # It is never delivered and never enters LongVid continuity.
            stitched = stitched_refine[:visible].clone()
            del stitched_refine, source_refine, source
            delivery_frames = stitched.clone()

            # The upscaler must receive the stitched video, not the untouched native
            # latent. If upscale is enabled, restore only the refined VISIBLE pixels
            # into the original full native extent AFTER FaceRefine has finished.
            upscale = bool(face_plan.get("upscale_enabled")) and face_plan.get("upscale_mode", "off") != "off"
            if upscale:
                stitched_full = source_full.clone()
                stitched_full[trim:trim + visible] = stitched
                delivery = inject().run(
                    av_latent=sampled_latent,
                    images=stitched_full,
                    vae=video_vae,
                )[0]
                del stitched_full
            else:
                delivery = dict(sampled_latent)
            del stitched, source_full
            out_v, out_a = common.unpack_av(delivery, "face delivery")
            if out_v.shape != source_v.shape or (
                source_a is not None and not torch.equal(out_a, source_a)
            ):
                raise ValueError("Face delivery changed native AV extent or audio.")
            delivery["iamccs_r38b_face_applied"] = True
            return delivery, delivery_frames, (
                f"Face delivery complete: {visible} visible frames refined in isolation; "
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
