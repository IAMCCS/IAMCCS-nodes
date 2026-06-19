"""IAMCCS SCAIL-2 identity helper nodes.

These nodes are adapted from Brobert-in-aus/scail-auto-extend for the D: IAMCCS
runtime. They intentionally keep the SAM3 tracking data contract used by
ComfyUI's official SCAIL2ColoredMask node.
"""

import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import comfy.model_management
import comfy.utils
import folder_paths


class SCAIL2IdentitySeeder:
    """Produce one binary mask per person from explicit point or box prompts, so
    each subject becomes a distinct tracked object (and thus a distinct colour) in
    SCAIL-2 multi-person workflows.

    Why this exists: SAM3_VideoTrack's auto-detection runs mask NMS using an
    IoU+IoM overlap test at a fixed 0.5 threshold, which collapses close or
    overlapping people into a single object — and its object roster is seeded from
    the first frame, so late-appearing people fail the same overlap gate. Neither
    constant is exposed, so detection_threshold can't fix it. Feeding explicit
    per-object masks into SAM3_VideoTrack's `initial_mask` (with conditioning left
    disconnected) bypasses detection entirely: the tracker propagates exactly the
    masks you seed, one colour each.

    Output: MASK of shape [N_people, H, W]. Wire into SAM3_VideoTrack.initial_mask.
    Feed `image` at the SAME resolution the tracker will run on (the resized
    reference image, or the resized pose video's first frame).
    """

    DESCRIPTION = (
        "One mask per person from point or box prompts -> SAM3_VideoTrack.initial_mask. "
        "Guarantees one tracked object (one colour) per subject, bypassing the "
        "auto-detector's overlap-merging. Leave SAM3_VideoTrack conditioning disconnected."
    )
    CATEGORY = "conditioning/video_models/scail"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("initial_masks",)
    FUNCTION = "seed"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "SAM3 model (same one feeding SAM3_VideoTrack)."}),
                "image": ("IMAGE", {"tooltip": "Frame to segment, at the resolution the tracker runs on (resized reference image, or pose video's first frame)."}),
                "mode": (["points", "boxes"], {"default": "points",
                          "tooltip": "points: one positive click per person (from a PointsEditor). boxes: one bounding box per person."}),
                "refine_iterations": ("INT", {"default": 2, "min": 0, "max": 5,
                                      "tooltip": "SAM decoder refinement passes per object (0 = raw prompt mask)."}),
            },
            "optional": {
                "points": ("STRING", {"default": "", "multiline": True,
                            "tooltip": "points mode: JSON list, one positive point per person, e.g. [{\"x\":120,\"y\":210},{\"x\":480,\"y\":205}]. Order sets identity order."}),
                "bboxes": ("BOUNDING_BOX", {"tooltip": "boxes mode: one bounding box per person. Order sets identity order."}),
            },
        }

    def seed(self, model, image, mode, refine_iterations, points="", bboxes=None):
        B, H, W, C = image.shape
        image_in = comfy.utils.common_upscale(
            image[:1, ..., :3].movedim(-1, 1), 1008, 1008, "bilinear", crop="disabled")
        comfy.model_management.load_model_gpu(model)
        device = comfy.model_management.get_torch_device()
        dtype = model.model.get_dtype()
        sam3_model = model.model.diffusion_model
        frame = image_in.to(device=device, dtype=dtype)

        def _refine(mask_logit):
            for _ in range(max(0, refine_iterations - 1)):
                mask_logit = sam3_model.forward_segment(frame, mask_inputs=mask_logit)
            mask = F.interpolate(mask_logit, size=(H, W), mode="bilinear", align_corners=False)
            return (mask[0] > 0).float()  # [1, H, W]

        masks = []
        if mode == "points":
            pts = json.loads(points) if points.strip() else []
            if not pts:
                raise ValueError("SCAIL-2 Identity Seeder (points mode): provide at least one point "
                                 "(one per person) in `points`.")
            for p in pts:
                coords = torch.tensor([[[p["x"] / W * 1008, p["y"] / H * 1008]]], dtype=dtype, device=device)
                labels = torch.ones((1, 1), dtype=torch.int32, device=device)
                mask_logit = sam3_model.forward_segment(
                    frame, point_inputs={"point_coords": coords, "point_labels": labels})
                masks.append(_refine(mask_logit))
        else:  # boxes
            box_list = bboxes if isinstance(bboxes, list) else ([bboxes] if bboxes else [])
            if not box_list:
                raise ValueError("SCAIL-2 Identity Seeder (boxes mode): provide one bounding box "
                                 "per person in `bboxes`.")
            for d in box_list:
                x1 = d["x"] / W * 1008
                y1 = d["y"] / H * 1008
                x2 = (d["x"] + d["width"]) / W * 1008
                y2 = (d["y"] + d["height"]) / H * 1008
                sam_box = torch.tensor([[[x1, y1], [x2, y2]]], device=device, dtype=dtype)
                mask_logit = sam3_model.forward_segment(frame, box_inputs=sam_box)
                masks.append(_refine(mask_logit))

        out = torch.cat(masks, dim=0).to(comfy.model_management.intermediate_device())  # [N_people, H, W]
        return (out,)


def _sam3_segment(sam3, image, markers, refine_iterations, device, dtype):
    """Run SAM3 segment per marker on the first frame of `image` (B,H,W,C).
    markers: list of {"type":"point","x","y"} or {"type":"box","x","y","w","h"}
    in `image` pixel coords. Returns [N, H, W] float masks at native H,W, or None."""
    if not markers:
        return None
    B, H, W, C = image.shape
    frame = comfy.utils.common_upscale(
        image[:1, ..., :3].movedim(-1, 1), 1008, 1008, "bilinear", crop="disabled").to(device=device, dtype=dtype)

    def _refine(mask_logit):
        for _ in range(max(0, refine_iterations - 1)):
            mask_logit = sam3.forward_segment(frame, mask_inputs=mask_logit)
        mask = F.interpolate(mask_logit, size=(H, W), mode="bilinear", align_corners=False)
        return (mask[0] > 0).float()  # [1, H, W]

    masks = []
    for m in markers:
        if m.get("type") == "box":
            x1 = m["x"] / W * 1008
            y1 = m["y"] / H * 1008
            x2 = (m["x"] + m["w"]) / W * 1008
            y2 = (m["y"] + m["h"]) / H * 1008
            sam_box = torch.tensor([[[x1, y1], [x2, y2]]], device=device, dtype=dtype)
            mask_logit = sam3.forward_segment(frame, box_inputs=sam_box)
        else:  # one or more points per identity, each positive (label 1) or negative (0)
            pts = m.get("points") or [[m.get("x", 0), m.get("y", 0), 1]]
            coords = torch.tensor([[[p[0] / W * 1008, p[1] / H * 1008] for p in pts]], dtype=dtype, device=device)
            labels = torch.tensor([[int(p[2]) if len(p) > 2 else 1 for p in pts]], dtype=torch.int32, device=device)
            mask_logit = sam3.forward_segment(frame, point_inputs={"point_coords": coords, "point_labels": labels})
        masks.append(_refine(mask_logit))
    return torch.cat(masks, dim=0)  # [N, H, W]


class SCAIL2IdentityTracker:
    """Canvas-driven multi-person seeding + dual SAM3 tracking for SCAIL-2.

    Turns a *processed* reference image and a driving video into the two
    SAM3_TRACK_DATA bundles SCAIL2ColoredMask consumes, with identities seeded by
    points/boxes you draw on an in-node canvas (placement order = colour order).
    Optional auto-detection appends late arrivals on the driving side.

    Because it's an OUTPUT_NODE, ComfyUI shows a play button on it: press it with no
    markers drawn and it only renders the two frames to the canvas (no tracking, no
    sampler) thanks to partial execution. Draw your points/boxes, then run normally.

    Feed the reference image AFTER background-removal/padding so masks match the
    pixels the model sees. Keep SCAIL2ColoredMask sort_by = "none" so the order you
    draw is preserved.
    """

    DESCRIPTION = (
        "Draw ordered points/boxes per person on the reference image and driving "
        "video, get ref_track_data + driving_track_data out. Play button previews "
        "the frames without tracking (partial execution). Auto-detect adds latecomers."
    )
    CATEGORY = "conditioning/video_models/scail"
    RETURN_TYPES = ("SAM3_TRACK_DATA", "SAM3_TRACK_DATA", "IMAGE", "IMAGE")
    RETURN_NAMES = ("ref_track_data", "driving_track_data", "reference_image", "pose_video")
    FUNCTION = "track"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_model": ("MODEL", {"tooltip": "SAM3 model (e.g. from CheckpointLoaderSimple)."}),
                "reference_image": ("IMAGE", {"tooltip": "Processed reference (post background-removal + padding), at model resolution."}),
                "pose_video": ("IMAGE", {"tooltip": "Driving/pose video frames, at the resolution fed to the sampler."}),
                "refine_iterations": ("INT", {"default": 2, "min": 0, "max": 5,
                                      "tooltip": "SAM decoder refinement passes per seed."}),
                "auto_detect": ("BOOLEAN", {"default": True,
                                "tooltip": "Master switch for text detection. When on, reference_conditioning / driving_conditioning (if connected) drive SAM3 text detection on that side, alongside any drawn boxes."}),
                "detection_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "Score threshold for auto-detected latecomers."}),
                "detect_interval": ("INT", {"default": 1, "min": 1, "max": 64,
                                    "tooltip": "Run auto-detection every N frames."}),
                "markers": ("STRING", {"default": "{}", "multiline": True,
                            "tooltip": "Canvas markers (JSON). Managed by the node's canvas widget."}),
            },
            "optional": {
                "reference_conditioning": ("CONDITIONING", {"tooltip": "Optional CLIPTextEncode (e.g. 'person') to auto-detect identities on the reference image instead of / alongside drawn boxes. Needs auto_detect on. Text-only loses explicit colour ordering."}),
                "driving_conditioning": ("CONDITIONING", {"tooltip": "Optional CLIPTextEncode to auto-detect / append people on the driving video. Needs auto_detect on."}),
            },
        }

    def _empty_track(self, N, H, W):
        return {"packed_masks": None, "orig_size": (H, W), "n_frames": int(N), "scores": []}

    def _track_side(self, sam3, images, seed_masks, text_prompts, device, dtype,
                    detection_threshold, max_objects, detect_interval):
        N, H, W, C = images.shape
        if seed_masks is None and text_prompts is None:
            return self._empty_track(N, H, W)
        frames_in = images[..., :3].movedim(-1, 1)
        init_masks = seed_masks.unsqueeze(1).to(device=device, dtype=dtype) if seed_masks is not None else None
        pbar = comfy.utils.ProgressBar(N)
        result = sam3.forward_video(
            images=frames_in, initial_masks=init_masks, pbar=pbar, text_prompts=text_prompts,
            new_det_thresh=detection_threshold, max_objects=max_objects,
            detect_interval=detect_interval, target_device=device, target_dtype=dtype)
        result["orig_size"] = (H, W)
        return result

    def _save_preview(self, img_hwc, prefix):
        arr = (img_hwc.detach().clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        pil = Image.fromarray(arr, "RGB")
        fname = f"{prefix}_{random.randint(0, 0xffffffff):08x}.png"
        tdir = folder_paths.get_temp_directory()
        os.makedirs(tdir, exist_ok=True)
        pil.save(os.path.join(tdir, fname), compress_level=4)
        return {"filename": fname, "subfolder": "", "type": "temp"}

    def track(self, sam3_model, reference_image, pose_video, refine_iterations, auto_detect,
              detection_threshold, detect_interval, markers,
              reference_conditioning=None, driving_conditioning=None):
        if not isinstance(markers, str):
            markers = ""
        try:
            data = json.loads(markers) if markers.strip() else {}
        except (json.JSONDecodeError, ValueError):
            data = {}
        if not isinstance(data, dict):  # stale/old widget value (e.g. a bare int) -> treat as empty
            data = {}
        ref_markers = data.get("reference", []) or []
        drv_markers = data.get("driving", []) or []

        previews = {
            "reference_preview": [self._save_preview(reference_image[0], "scail_ref")],
            "driving_preview": [self._save_preview(pose_video[0], "scail_drv")],
        }

        has_text = auto_detect and (
            (reference_conditioning is not None and len(reference_conditioning) > 0)
            or (driving_conditioning is not None and len(driving_conditioning) > 0))
        # No markers and no text prompts -> play-button preview pass. Emit frames, skip tracking.
        if not ref_markers and not drv_markers and not has_text:
            ref_td = self._empty_track(reference_image.shape[0], reference_image.shape[1], reference_image.shape[2])
            drv_td = self._empty_track(pose_video.shape[0], pose_video.shape[1], pose_video.shape[2])
            return {"ui": previews, "result": (ref_td, drv_td, reference_image, pose_video)}

        comfy.model_management.load_model_gpu(sam3_model)
        device = comfy.model_management.get_torch_device()
        dtype = sam3_model.model.get_dtype()
        sam3 = sam3_model.model.diffusion_model

        ref_seed = _sam3_segment(sam3, reference_image, ref_markers, refine_iterations, device, dtype)
        drv_seed = _sam3_segment(sam3, pose_video, drv_markers, refine_iterations, device, dtype)

        def _text(cond):
            if not (auto_detect and cond is not None and len(cond) > 0):
                return None
            from comfy_extras.nodes_sam3 import _extract_text_prompts
            return [(emb, m) for emb, m, _ in _extract_text_prompts(cond, device, dtype)]
        ref_text = _text(reference_conditioning)
        drv_text = _text(driving_conditioning)

        ref_count = int(ref_seed.shape[0]) if ref_seed is not None else 0
        drv_count = int(drv_seed.shape[0]) if drv_seed is not None else 0

        # With auto_detect on, text can add identities beyond the drawn boxes: up to 6 on the
        # reference, and up to the reference box count on the driving side (ref_text/drv_text
        # are already None when auto_detect is off). With auto_detect off there is no text to
        # fill gaps, so warn when there are fewer reference identities than driving subjects.
        if not auto_detect and ref_count < drv_count:
            print(f"[SCAIL-2 Identity Tracker] WARNING: {ref_count} reference identit(y/ies) but "
                  f"{drv_count} driving subject(s) seeded with auto_detect off; "
                  f"{drv_count - ref_count} driving subject(s) have no reference to map to.")

        # Reference: drawn seeds plus, with auto_detect on, text detection up to 6.
        ref_td = self._track_side(sam3, reference_image, ref_seed, ref_text, device, dtype,
                                  detection_threshold, 6 if ref_text is not None else 0, detect_interval)
        # Driving cap = reference box count (max 6); driving boxes seed identities, text fills
        # the remaining headroom with non-overlapping detections. No reference boxes -> cap 6.
        drv_max = min(ref_count, 6) if ref_count > 0 else 6
        drv_td = self._track_side(sam3, pose_video, drv_seed, drv_text, device, dtype,
                                  detection_threshold, drv_max, detect_interval)

        return {"ui": previews, "result": (ref_td, drv_td, reference_image, pose_video)}


class SCAIL2MultiReference:
    """EXPERIMENTAL concept-validation node for true per-identity references.

    Instead of compositing every character into one reference frame (which makes
    the model bind identities by spatial position), this stacks up to six
    single-character images as separate reference frames, each tagged with one
    palette colour by input order (image_1 -> colour 0/blue, image_2 -> red, ...).
    Each identity isolated in its own frame removes the within-frame x-position
    shortcut, so the colour mask should become the binding signal.

    Wiring (single pass, <=81 frames; not the auto-extend sampler yet):
      WanSCAILToVideo (leave reference_image / reference_image_mask EMPTY; it still
      builds pose, driving mask, latent) -> this node (adds the multi-frame
      reference) -> SamplerCustom -> VAEDecode.

    `length` and `replacement_mode` must match the WanSCAILToVideo upstream. Colour
    order must match the driving side (image_i <-> driving colour i). Per-character
    `mask_i` (silhouette, e.g. from RMBG) is recommended; without it the whole
    frame is treated as the character.
    """

    DESCRIPTION = (
        "EXPERIMENTAL: stack up to 6 single-character images as separate reference "
        "frames (image_1->blue, image_2->red, ...) to test colour-based identity "
        "binding. Wire after WanSCAILToVideo (reference left empty), before SamplerCustom."
    )
    CATEGORY = "conditioning/video_models/scail"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, 7):
            optional[f"image_{i}"] = ("IMAGE", {"tooltip": f"Single character -> palette colour {i - 1}. Order must match the driving colours."})
            optional[f"mask_{i}"] = ("MASK", {"tooltip": f"Silhouette for image_{i} (e.g. RMBG mask). Optional; defaults to the whole frame."})
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 512, "min": 32, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 896, "min": 32, "max": 8192, "step": 32}),
                "length": ("INT", {"default": 81, "min": 1, "max": 1024, "step": 4,
                           "tooltip": "Generation length in frames. Must match the WanSCAILToVideo length."}),
                "replacement_mode": ("BOOLEAN", {"default": True,
                                     "tooltip": "Must match WanSCAILToVideo. Replacement composites each ref on black; animation uses the full frame."}),
            },
            "optional": optional,
        }

    def apply(self, positive, negative, vae, width, height, length, replacement_mode, **kwargs):
        import node_helpers
        from comfy_extras.nodes_scail import _extract_mask_to_28ch, DEFAULT_PALETTE

        refs = []
        for i in range(1, 7):
            img = kwargs.get(f"image_{i}")
            if img is not None:
                refs.append((img, kwargs.get(f"mask_{i}")))
        if not refs:
            return (positive, negative)

        ref_latents = []
        colour_masks = []
        for idx, (img, msk) in enumerate(refs):
            image = comfy.utils.common_upscale(
                img[:1, ..., :3].movedim(-1, 1), width, height, "bicubic", "center").movedim(1, -1)  # [1,H,W,3]
            if msk is not None:
                m = F.interpolate(msk[:1].unsqueeze(1), size=(height, width), mode="nearest")[:, 0]  # [1,H,W]
            else:
                m = torch.ones((1, height, width), device=image.device, dtype=image.dtype)
            m3 = m.unsqueeze(-1).to(image.dtype)  # [1,H,W,1]

            if replacement_mode:
                ref_img = image * (m3 > 0.1).to(image.dtype)
            else:
                ref_img = image
            ref_latents.append(vae.encode(ref_img[:, :, :, :3]))  # [1,16,1,h,w]

            colour = torch.tensor(DEFAULT_PALETTE[idx % len(DEFAULT_PALETTE)], device=image.device, dtype=image.dtype).view(1, 1, 1, 3)
            bg = 0.0 if replacement_mode else 1.0
            char_sel = (m3 > 0.5).to(image.dtype)
            colour_img = char_sel * colour + (1.0 - char_sel) * bg  # [1,H,W,3]
            colour_masks.append(_extract_mask_to_28ch(colour_img))  # [1,1,28,h',w']

        reference_latent = torch.cat(ref_latents, dim=2)        # [1,16,N,h,w]
        ref_colour = torch.cat(colour_masks, dim=1)             # [1,N,28,h',w']
        lat_t = ((length - 1) // 4) + 1
        zeros = torch.zeros((1, lat_t, 28, ref_colour.shape[-2], ref_colour.shape[-1]),
                            device=ref_colour.device, dtype=ref_colour.dtype)
        ref_mask_28ch = torch.cat([ref_colour, zeros], dim=1)   # [1, N+lat_t, 28, h',w']

        positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [reference_latent]}, append=True)
        negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [reference_latent]}, append=True)
        positive = node_helpers.conditioning_set_values(positive, {"ref_mask_28ch": ref_mask_28ch})
        negative = node_helpers.conditioning_set_values(negative, {"ref_mask_28ch": ref_mask_28ch})
        return (positive, negative)





class IAMCCS_ScailIdentitySeeder(SCAIL2IdentitySeeder):
    CATEGORY = "IAMCCS/video/SCAIL-2"


class IAMCCS_ScailIdentityTracker(SCAIL2IdentityTracker):
    CATEGORY = "IAMCCS/video/SCAIL-2"
    DESCRIPTION = (
        SCAIL2IdentityTracker.DESCRIPTION
        + " IAMCCS variant: use it as a preflight canvas before running the full sampler."
    )


class IAMCCS_ScailMultiReference(SCAIL2MultiReference):
    CATEGORY = "IAMCCS/video/SCAIL-2"
    DESCRIPTION = (
        SCAIL2MultiReference.DESCRIPTION
        + " IAMCCS note: experimental; composited single-reference layouts are usually higher fidelity."
    )


NODE_CLASS_MAPPINGS = {
    "IAMCCS_ScailIdentitySeeder": IAMCCS_ScailIdentitySeeder,
    "IAMCCS_ScailIdentityTracker": IAMCCS_ScailIdentityTracker,
    "IAMCCS_ScailMultiReference": IAMCCS_ScailMultiReference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_ScailIdentitySeeder": "IAMCCS SCAIL Identity Seeder",
    "IAMCCS_ScailIdentityTracker": "IAMCCS SCAIL Identity Tracker",
    "IAMCCS_ScailMultiReference": "IAMCCS SCAIL Multi-Reference (experimental)",
}
