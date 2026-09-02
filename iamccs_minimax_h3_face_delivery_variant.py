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
            # Refine the complete source extent, including context and grid padding;
            # trim only delivery pixels, exactly once, with the native trim value.
            source = nodes.VAEDecode().decode(vae=video_vae, samples=sampled_latent)[0]
            track = _face_node_class("H3FaceTrackCrop")
            crops, transform, _, _, cw, ch = track().run(**{
                **_defaults(track), "images":source, "detector":detector,
                "canvas_width":canvas_width, "canvas_height":canvas_height,
                "crop_factor":crop_factor,"confidence":confidence,
                "identity_track":False,  # no hidden InsightFace download
            })
            mask = None
            if face.get("use_sam_mask"):
                mask_class = _face_node_class("H3FaceMaskSAM")
                mask = mask_class().run(**{**_defaults(mask_class),"crops":crops,
                    "transform":transform,"sam_model":sam_model})[0]
            template = EmptyMiniMaxH3LatentAV.execute(width=cw,height=ch,length=source_count)[0]
            template_v, _ = common.unpack_av(template, "face canvas")
            template = common.pack_av({}, template_v.to(source_v), source_a)
            inject = _face_node_class("H3InjectVideoLatent")
            cropped = inject().run(av_latent=template,images=crops,vae=video_vae)[0]
            del template, template_v
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
            if len(pixels) != len(source):
                raise ValueError("FaceRefine changed frame count; refusing an audio offset.")
            stitch = _face_node_class("H3FaceStitch")
            stitched = stitch().run(**{**_defaults(stitch),"base_images":source,"refined_crops":pixels,
                "transform":transform,"masks":mask,"blend":blend,"paste_region":paste_region,"feather":feather})[0]
            del source, pixels, crops, mask
            delivery_frames = stitched[trim:trim+visible].clone()
            # The upscaler must receive the stitched video, not the untouched native
            # latent. Use FaceRefine's own injector; audio remains byte-identical.
            upscale = bool(face_plan.get("upscale_enabled")) and face_plan.get("upscale_mode", "off") != "off"
            delivery = inject().run(av_latent=sampled_latent,images=stitched,vae=video_vae)[0] if upscale else dict(sampled_latent)
            del stitched
            out_v, out_a = common.unpack_av(delivery, "face delivery")
            if out_v.shape != source_v.shape or not torch.equal(out_a, source_a):
                raise ValueError("Face delivery changed native AV extent or audio.")
            delivery["iamccs_r38b_face_applied"] = True
            return delivery, delivery_frames, f"Face delivery complete: {visible}f, audio/context timing unchanged; optional upscale next."
        finally:
            mm.unload_all_models()
            mm.soft_empty_cache()
