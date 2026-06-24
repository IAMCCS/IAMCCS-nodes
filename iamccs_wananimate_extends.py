import math

import torch
from PIL import Image
import numpy as np

import comfy.model_management
import comfy.utils


def _to_4n1(value):
    return ((max(1, int(value)) - 1) // 4) * 4 + 1


def _image_frames(x):
    if x is None:
        return 0
    try:
        return int(x.shape[0])
    except Exception:
        return 0




def _preview_image_tuple(images, mode):
    if images is None or int(images.shape[0]) <= 0:
        return None
    mode = str(mode or "middle_frame")
    if mode == "off":
        return None
    if mode == "first_frame":
        idx = 0
    elif mode == "last_frame":
        idx = int(images.shape[0]) - 1
    else:
        idx = int(images.shape[0]) // 2
    frame = images[idx].detach().clamp(0.0, 1.0).cpu().numpy()
    img = Image.fromarray(np.clip(frame * 255.0, 0, 255).astype(np.uint8))
    return ("PNG", img, None)


def _estimate_wananimate_plan(source_frames, first_chunk_length, next_chunk_length, continue_motion_max_frames):
    plan = []
    total = 0
    idx = 0
    while total < int(source_frames):
        length = first_chunk_length if idx == 0 else next_chunk_length
        trim_hint = 0 if idx == 0 else continue_motion_max_frames
        add_hint = length if idx == 0 else max(1, length - continue_motion_max_frames)
        add_hint = min(add_hint, int(source_frames) - total)
        plan.append(f"#{idx + 1}:len{length}/trim~{trim_hint}/add~{add_hint}")
        total += add_hint
        idx += 1
        if idx > 10000:
            break
    return plan

def _clone_trim_latent(latent, trim_amount):
    out = latent.copy()
    samples = latent["samples"]
    trim_amount = max(0, int(trim_amount))
    if trim_amount > 0:
        out["samples"] = samples[:, :, trim_amount:]
    else:
        out["samples"] = samples
    return out


class IAMCCS_WanAnimateExtends:
    DESCRIPTION = (
        "IAMCCS WanAnimate long-video wrapper. It runs WanAnimateToVideo + KSampler + VAE decode "
        "in chunks, uses the previous visible tail as continue_motion, trims native latent/image "
        "anchors, and can grade only the first boundary frames of each joined chunk."
    )
    CATEGORY = "IAMCCS/video/WanAnimate"
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "images",
        "frame_count",
        "source_frames",
        "trimmed_frames",
        "chunk_plan",
    )
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers

        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 16}),
                "target_frames": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100000,
                        "step": 1,
                        "tooltip": "0 = infer from pose/background/face/mask frame count.",
                    },
                ),
                "first_chunk_length": ("INT", {"default": 81, "min": 5, "max": 1024, "step": 4}),
                "next_chunk_length": (
                    "INT",
                    {
                        "default": 77,
                        "min": 5,
                        "max": 1024,
                        "step": 4,
                        "tooltip": "Matches the common WanAnimate loop math: 77 length with 5 continue frames contributes 72 visible frames.",
                    },
                ),
                "continue_motion_max_frames": ("INT", {"default": 5, "min": 1, "max": 129, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xffffffffffffffff,
                        "control_after_generate": True,
                    },
                ),
                "seed_mode": (["fixed", "increment"], {"default": "fixed"}),
                "steps": ("INT", {"default": 6, "min": 1, "max": 10000, "step": 1}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "continuity_profile": (
                    ["off", "iamccs_boundary_ramp", "external_1to1"],
                    {
                        "default": "iamccs_boundary_ramp",
                        "tooltip": "boundary_ramp only color matches the first frames after a chunk seam.",
                    },
                ),
                "color_method": (["reinhard_lab", "mkl_lab", "histogram"], {"default": "reinhard_lab"}),
                "boundary_frames": ("INT", {"default": 8, "min": 1, "max": 96, "step": 1}),
                "boundary_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "live_chunk_preview": (
                    "STRING",
                    {
                        "default": "middle_frame",
                        "tooltip": "off, first_frame, middle_frame, last_frame. Stale numeric values are treated as middle_frame.",
                    },
                ),
                "empty_cache_each_chunk": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "reference_image": ("IMAGE",),
                "face_video": ("IMAGE",),
                "pose_video": ("IMAGE",),
                "background_video": ("IMAGE",),
                "character_mask": ("MASK",),
            },
        }

    def _match_external(self, color_node, contrib, ref_frame, method):
        return color_node.execute(
            image_target=contrib,
            image_ref=ref_frame,
            method=method,
            source_stats={"source_stats": "per_frame"},
            strength=1.0,
        ).args[0]

    def _match_boundary(self, color_node, contrib, ref_frame, method, boundary_frames, strength):
        if strength <= 0.0 or contrib.shape[0] == 0:
            return contrib
        n = min(int(boundary_frames), int(contrib.shape[0]))
        head = contrib[:n]
        matched = color_node.execute(
            image_target=head,
            image_ref=ref_frame,
            method=method,
            source_stats={"source_stats": "per_frame"},
            strength=1.0,
        ).args[0]
        if n == 1:
            weights = torch.ones((1, 1, 1, 1), device=head.device, dtype=head.dtype) * strength
        else:
            weights = torch.linspace(strength, 0.0, n, device=head.device, dtype=head.dtype).view(n, 1, 1, 1)
        corrected = torch.lerp(head, matched.to(device=head.device, dtype=head.dtype), weights.clamp(0.0, 1.0))
        if n == contrib.shape[0]:
            return corrected
        return torch.cat([corrected, contrib[n:]], dim=0)

    def generate(
        self,
        model,
        positive,
        negative,
        vae,
        width,
        height,
        target_frames,
        first_chunk_length,
        next_chunk_length,
        continue_motion_max_frames,
        batch_size,
        seed,
        seed_mode,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        continuity_profile,
        color_method,
        boundary_frames,
        boundary_strength,
        live_chunk_preview,
        empty_cache_each_chunk,
        clip_vision_output=None,
        reference_image=None,
        face_video=None,
        pose_video=None,
        background_video=None,
        character_mask=None,
    ):
        import nodes
        from comfy_extras.nodes_post_processing import ColorTransfer
        from comfy_extras.nodes_wan import WanAnimateToVideo

        first_chunk_length = _to_4n1(first_chunk_length)
        next_chunk_length = _to_4n1(next_chunk_length)
        continue_motion_max_frames = _to_4n1(continue_motion_max_frames)
        inferred = max(
            _image_frames(pose_video),
            _image_frames(background_video),
            _image_frames(face_video),
            _image_frames(character_mask),
        )
        source_frames = int(target_frames) if int(target_frames) > 0 else inferred
        if source_frames <= 0:
            source_frames = first_chunk_length

        if width % 16 != 0 or height % 16 != 0:
            raise ValueError(f"IAMCCS_WanAnimateExtends: width/height must be divisible by 16, got {width}x{height}.")

        sampler = nodes.KSampler()
        decoder = nodes.VAEDecode()

        chunks = []
        prev_motion = None
        prev_color_ref = None
        offset = 0
        chunk_index = 0
        plan = []

        estimated_plan = _estimate_wananimate_plan(
            source_frames,
            first_chunk_length,
            next_chunk_length,
            continue_motion_max_frames,
        )
        estimated_chunks = len(estimated_plan)
        print(
            f"[IAMCCS_WanAnimateExtends] PLAN source_frames={source_frames} "
            f"first_chunk_length={first_chunk_length} next_chunk_length={next_chunk_length} "
            f"continue_motion_max_frames={continue_motion_max_frames} estimated_chunks={estimated_chunks} "
            f"plan={' | '.join(estimated_plan)}",
            flush=True,
        )

        max_chunks = max(estimated_chunks + 4, math.ceil(source_frames / max(1, next_chunk_length - continue_motion_max_frames)) + 4)
        pbar = comfy.utils.ProgressBar(max_chunks)

        while sum(int(c.shape[0]) for c in chunks) < source_frames:
            comfy.model_management.throw_exception_if_processing_interrupted()
            length = first_chunk_length if chunk_index == 0 else next_chunk_length
            this_seed = int(seed) + chunk_index if seed_mode == "increment" else int(seed)

            cond = WanAnimateToVideo.execute(
                positive=positive,
                negative=negative,
                vae=vae,
                width=int(width),
                height=int(height),
                length=int(length),
                batch_size=int(batch_size),
                continue_motion_max_frames=int(continue_motion_max_frames),
                video_frame_offset=int(offset),
                reference_image=reference_image,
                clip_vision_output=clip_vision_output,
                face_video=face_video,
                pose_video=pose_video,
                continue_motion=prev_motion,
                background_video=background_video,
                character_mask=character_mask,
            )
            pos_c, neg_c, latent, trim_latent, trim_image, offset = cond.args

            sampled = sampler.sample(
                model=model,
                seed=this_seed,
                steps=int(steps),
                cfg=float(cfg),
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=pos_c,
                negative=neg_c,
                latent_image=latent,
                denoise=float(denoise),
            )[0]
            sampled = _clone_trim_latent(sampled, int(trim_latent))
            images = decoder.decode(vae=vae, samples=sampled)[0]
            if images.ndim == 5:
                images = images.reshape(-1, *images.shape[-3:])

            trim_image = max(0, int(trim_image))
            contrib = images[trim_image:] if trim_image > 0 else images
            if contrib.shape[0] == 0:
                raise RuntimeError(
                    f"IAMCCS_WanAnimateExtends: chunk {chunk_index + 1} produced no visible frames "
                    f"(length={length}, trim_image={trim_image})."
                )

            if chunk_index > 0 and continuity_profile != "off" and prev_color_ref is not None:
                if continuity_profile == "external_1to1":
                    contrib = self._match_external(ColorTransfer, contrib, prev_color_ref, color_method)
                else:
                    contrib = self._match_boundary(
                        ColorTransfer,
                        contrib,
                        prev_color_ref,
                        color_method,
                        boundary_frames,
                        boundary_strength,
                    )

            chunks.append(contrib)
            visible_so_far = sum(int(c.shape[0]) for c in chunks)
            prev_motion = torch.cat(chunks, dim=0)[-continue_motion_max_frames:].detach()
            prev_color_ref = contrib[-1:].detach()

            plan.append(f"{length}/trimL{int(trim_latent)}/trimI{trim_image}/add{int(contrib.shape[0])}/off{int(offset)}")
            print(
                f"[IAMCCS_WanAnimateExtends] chunk {chunk_index + 1}: "
                f"length={length} seed={this_seed} trim_latent={int(trim_latent)} "
                f"trim_image={trim_image} contributed={int(contrib.shape[0])} "
                f"visible={visible_so_far}/{source_frames} offset={int(offset)}"
            )
            pbar.update_absolute(chunk_index + 1, max_chunks, _preview_image_tuple(contrib, live_chunk_preview))
            chunk_index += 1

            if empty_cache_each_chunk:
                comfy.model_management.soft_empty_cache()
            if chunk_index > max_chunks:
                raise RuntimeError("IAMCCS_WanAnimateExtends: safety stop hit while chunking.")

        images = torch.cat([c.to(device=chunks[0].device, dtype=chunks[0].dtype) for c in chunks], dim=0)
        if images.shape[0] > source_frames:
            images = images[:source_frames]
        trimmed_frames = max(0, sum(int(c.shape[0]) for c in chunks) - int(images.shape[0]))

        return (
            images,
            int(images.shape[0]),
            int(source_frames),
            int(trimmed_frames),
            " | ".join(plan),
        )


class IAMCCS_WanAnimateExtendPlan:
    DESCRIPTION = "Plan IAMCCS WanAnimate chunk contribution without sampling."
    CATEGORY = "IAMCCS/video/WanAnimate"
    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("source_frames", "estimated_chunks", "chunk_plan_hint")
    FUNCTION = "plan"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_frames": ("INT", {"default": 161, "min": 1, "max": 100000, "step": 1}),
                "first_chunk_length": ("INT", {"default": 81, "min": 5, "max": 1024, "step": 4}),
                "next_chunk_length": ("INT", {"default": 77, "min": 5, "max": 1024, "step": 4}),
                "continue_motion_max_frames": ("INT", {"default": 5, "min": 1, "max": 129, "step": 4}),
            }
        }

    def plan(self, source_frames, first_chunk_length, next_chunk_length, continue_motion_max_frames):
        first_chunk_length = _to_4n1(first_chunk_length)
        next_chunk_length = _to_4n1(next_chunk_length)
        continue_motion_max_frames = _to_4n1(continue_motion_max_frames)
        contributed = min(source_frames, first_chunk_length)
        chunks = [f"{first_chunk_length}->+{contributed}"]
        while contributed < source_frames:
            add = max(1, next_chunk_length - continue_motion_max_frames)
            contributed += add
            chunks.append(f"{next_chunk_length}-trim{continue_motion_max_frames}->+{add}")
        return (int(source_frames), len(chunks), " | ".join(chunks))


NODE_CLASS_MAPPINGS = {
    "IAMCCS_WanAnimateExtends": IAMCCS_WanAnimateExtends,
    "IAMCCS_WanAnimateExtendPlan": IAMCCS_WanAnimateExtendPlan,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_WanAnimateExtends": "IAMCCS WanAnimate Extends",
    "IAMCCS_WanAnimateExtendPlan": "IAMCCS WanAnimate Extend Plan",
}
