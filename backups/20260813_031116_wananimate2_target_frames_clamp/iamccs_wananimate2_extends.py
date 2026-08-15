import math

import numpy as np
import torch
from PIL import Image

import comfy.model_management
import comfy.utils


def _to_4n1(value):
    return ((max(1, int(value)) - 1) // 4) * 4 + 1


def _image_frames(value):
    if value is None:
        return 0
    try:
        return int(value.shape[0])
    except Exception:
        return 0


def _preview_image_tuple(images, mode):
    if images is None or int(images.shape[0]) <= 0 or mode == "off":
        return None
    if mode == "first_frame":
        index = 0
    elif mode == "last_frame":
        index = int(images.shape[0]) - 1
    else:
        index = int(images.shape[0]) // 2
    frame = images[index].detach().clamp(0.0, 1.0).cpu().numpy()
    image = Image.fromarray(np.clip(frame * 255.0, 0, 255).astype(np.uint8))
    return ("PNG", image, None)


def _trim_video_latent(latent, trim_amount):
    output = latent.copy()
    output["samples"] = latent["samples"][:, :, max(0, int(trim_amount)):]
    return output


def _repeat_or_hold(frames, count):
    if frames.shape[0] == count:
        return frames
    if frames.shape[0] == 1:
        return frames.repeat((count,) + (1,) * (frames.ndim - 1))
    if frames.shape[0] < count:
        return torch.cat((frames, frames[-1:].repeat((count - frames.shape[0],) + (1,) * (frames.ndim - 1))), dim=0)
    return frames[:count]


def _resize_images(images, count, width, height):
    if images is None or _image_frames(images) <= 0:
        return None
    images = _repeat_or_hold(images, count)
    return comfy.utils.common_upscale(
        images.movedim(-1, 1), int(width), int(height), "area", "center"
    ).movedim(1, -1)


def _resize_mask(mask, count, width, height):
    if mask is None:
        return None
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    elif mask.ndim == 4:
        if mask.shape[-1] in (1, 3, 4):
            mask = mask[..., :3].amax(dim=-1)
        elif mask.shape[1] == 1:
            mask = mask[:, 0]
        else:
            raise ValueError("Composite mask must be [T,H,W], [T,H,W,C], or [T,1,H,W].")
    if mask.ndim != 3:
        raise ValueError("Composite mask must resolve to [T,H,W].")
    mask = _repeat_or_hold(mask, count).unsqueeze(1)
    mask = comfy.utils.common_upscale(mask, int(width), int(height), "nearest-exact", "center")
    return mask[:, 0].clamp(0.0, 1.0)


def _mask_reference(reference_image, reference_character_mask, width, height):
    if reference_character_mask is None:
        raise ValueError(
            "reference_background_mode=isolate_character requires reference_character_mask "
            "(white character, black background)."
        )
    reference = _resize_images(reference_image, 1, width, height)
    mask = _resize_mask(reference_character_mask, 1, width, height).unsqueeze(-1)
    return reference * mask.to(device=reference.device, dtype=reference.dtype)


def _apply_output_background(
    native_images,
    output_background_mode,
    composite_mask,
    reference_image,
    pose_video,
    background_image,
    background_video,
    width,
    height,
):
    if output_background_mode == "native_generated":
        return native_images
    if composite_mask is None:
        raise ValueError(
            f"output_background_mode={output_background_mode} requires composite_mask "
            "(white selects the generated character)."
        )

    if output_background_mode == "source_video_composite":
        plate = pose_video
    elif output_background_mode == "reference_image_composite":
        plate = background_image if background_image is not None else reference_image
    elif output_background_mode == "custom_background_composite":
        plate = background_video if background_video is not None else background_image
    else:
        raise ValueError(f"Unknown output_background_mode: {output_background_mode}")

    if plate is None or _image_frames(plate) <= 0:
        raise ValueError(f"No background plate is available for {output_background_mode}.")

    count = int(native_images.shape[0])
    plate = _resize_images(plate, count, width, height).to(
        device=native_images.device, dtype=native_images.dtype
    )
    alpha = _resize_mask(composite_mask, count, width, height).to(
        device=native_images.device, dtype=native_images.dtype
    ).unsqueeze(-1)
    return native_images * alpha + plate * (1.0 - alpha)


def _estimate_plan(source_frames, chunk_length):
    source_frames = max(1, int(source_frames))
    chunk_length = _to_4n1(chunk_length)
    plan = []
    visible = 0
    index = 0
    while visible < source_frames:
        trim = 0 if index == 0 else 1
        contribution = min(chunk_length - trim, source_frames - visible)
        plan.append(f"#{index + 1}:len{chunk_length}/trim{trim}/add{contribution}")
        visible += contribution
        index += 1
        if index > 10000:
            raise RuntimeError("Wan-Animate-2 chunk planning safety limit reached.")
    return plan


class IAMCCS_WanAnimate2Extends:
    DESCRIPTION = (
        "Long-video wrapper for native WanAnimate2ToVideo. It samples 4n+1 chunks, "
        "passes the previous generated frame through continue_motion, advances the raw "
        "driving-video offset, trims native latent/image anchors, and optionally composites "
        "the generated character over a masked background plate."
    )
    CATEGORY = "IAMCCS/video/WanAnimate-2"
    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "images",
        "native_images",
        "frame_count",
        "source_frames",
        "trimmed_frames",
        "chunk_plan",
    )
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "reference_image": ("IMAGE",),
                "pose_video": (
                    "IMAGE",
                    {
                        "tooltip": "Raw driving-video frames. Wan-Animate-2 does not require a DWPose render here.",
                    },
                ),
                "width": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 16}),
                "target_frames": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100000,
                        "step": 1,
                        "tooltip": "0 uses the complete driving-video frame count.",
                    },
                ),
                "chunk_length": (
                    "INT",
                    {
                        "default": 81,
                        "min": 5,
                        "max": 1025,
                        "step": 4,
                        "tooltip": "Wan-Animate-2 official workflow default is 81 frames.",
                    },
                ),
                "noise_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "seed_mode": (["fixed", "increment"], {"default": "fixed"}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "reference_image_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
                "pose_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
                "pose_start_percent": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "pose_end_percent": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "add_noise": ("BOOLEAN", {"default": True}),
                "enable_context_windows": ("BOOLEAN", {"default": True}),
                "context_length_latents": ("INT", {"default": 21, "min": 1, "max": 1024, "step": 1}),
                "context_overlap_latents": ("INT", {"default": 8, "min": 0, "max": 1023, "step": 1}),
                "context_schedule": (
                    ["standard_static", "standard_uniform", "looped_uniform", "batched"],
                    {"default": "standard_static"},
                ),
                "context_fuse_method": (
                    ["pyramid", "relative", "flat", "overlap-linear"],
                    {"default": "pyramid"},
                ),
                "enable_pose_cache": ("BOOLEAN", {"default": True}),
                "cache_device": (["cpu", "gpu"], {"default": "cpu"}),
                "cache_dtype": (["int8", "int4", "default"], {"default": "int8"}),
                "reference_background_mode": (
                    ["keep_reference_background", "isolate_character"],
                    {"default": "keep_reference_background"},
                ),
                "output_background_mode": (
                    [
                        "native_generated",
                        "source_video_composite",
                        "reference_image_composite",
                        "custom_background_composite",
                    ],
                    {"default": "native_generated"},
                ),
                "live_chunk_preview": (
                    ["off", "first_frame", "middle_frame", "last_frame"],
                    {"default": "middle_frame"},
                ),
                "empty_cache_each_chunk": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "positive_pose": ("CONDITIONING",),
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "clip_vision_output_pose": ("CLIP_VISION_OUTPUT",),
                "reference_character_mask": (
                    "MASK",
                    {"tooltip": "White character mask used only by isolate_character."},
                ),
                "composite_mask": (
                    "MASK",
                    {"tooltip": "White selects generated character; black selects the background plate."},
                ),
                "background_image": ("IMAGE",),
                "background_video": ("IMAGE",),
            },
        }

    def _prepare_model(
        self,
        model,
        enable_context_windows,
        context_length_latents,
        context_overlap_latents,
        context_schedule,
        context_fuse_method,
    ):
        if not enable_context_windows:
            return model
        if int(context_overlap_latents) >= int(context_length_latents):
            raise ValueError("context_overlap_latents must be smaller than context_length_latents.")
        from comfy_extras.nodes_context_windows import ContextWindowsManualNode

        return ContextWindowsManualNode.execute(
            model=model,
            context_length=int(context_length_latents),
            context_overlap=int(context_overlap_latents),
            context_schedule=context_schedule,
            context_stride=1,
            closed_loop=False,
            fuse_method=context_fuse_method,
            dim=2,
            freenoise=True,
            cond_retain_index_list="0",
            split_conds_to_windows=False,
            latent_retain_index_list="",
            causal_window_fix=True,
        ).args[0]

    def generate(
        self,
        model,
        positive,
        negative,
        vae,
        sampler,
        sigmas,
        reference_image,
        pose_video,
        width,
        height,
        target_frames,
        chunk_length,
        noise_seed,
        seed_mode,
        cfg,
        reference_image_strength,
        pose_strength,
        pose_start_percent,
        pose_end_percent,
        add_noise,
        enable_context_windows,
        context_length_latents,
        context_overlap_latents,
        context_schedule,
        context_fuse_method,
        enable_pose_cache,
        cache_device,
        cache_dtype,
        reference_background_mode,
        output_background_mode,
        live_chunk_preview,
        empty_cache_each_chunk,
        positive_pose=None,
        clip_vision_output=None,
        clip_vision_output_pose=None,
        reference_character_mask=None,
        composite_mask=None,
        background_image=None,
        background_video=None,
    ):
        from comfy_extras.nodes_custom_sampler import SamplerCustom
        from comfy_extras.nodes_wan import WanAnimate2Cache, WanAnimate2ToVideo

        width = int(width)
        height = int(height)
        if width % 16 != 0 or height % 16 != 0:
            raise ValueError(f"Width and height must be divisible by 16, got {width}x{height}.")
        if float(pose_start_percent) > float(pose_end_percent):
            raise ValueError("pose_start_percent must not exceed pose_end_percent.")
        if _image_frames(reference_image) <= 0 or _image_frames(pose_video) <= 0:
            raise ValueError("Wan-Animate-2 Extends requires a reference image and raw driving video frames.")

        chunk_length = _to_4n1(chunk_length)
        available_frames = _image_frames(pose_video)
        source_frames = int(target_frames) if int(target_frames) > 0 else available_frames
        if source_frames > available_frames:
            raise ValueError(
                f"target_frames ({source_frames}) exceeds driving-video frames ({available_frames}). "
                "Trim target_frames or extend the driving video first."
            )
        source_frames = max(1, source_frames)

        # Match the official Wan-Animate-2 graph: both the reference and driving
        # frames enter the native conditioning node at the requested generation size.
        conditioned_reference = _resize_images(reference_image, 1, width, height)
        conditioned_pose_video = _resize_images(pose_video, available_frames, width, height)
        if reference_background_mode == "isolate_character":
            conditioned_reference = _mask_reference(
                reference_image, reference_character_mask, width, height
            )

        prepared_model = self._prepare_model(
            model,
            enable_context_windows,
            context_length_latents,
            context_overlap_latents,
            context_schedule,
            context_fuse_method,
        )

        estimated_plan = _estimate_plan(source_frames, chunk_length)
        max_chunks = len(estimated_plan)
        print(
            f"[IAMCCS_WanAnimate2Extends] PLAN source_frames={source_frames} "
            f"chunk_length={chunk_length} chunks={max_chunks} "
            f"background={output_background_mode} plan={' | '.join(estimated_plan)}",
            flush=True,
        )

        progress = comfy.utils.ProgressBar(max_chunks)
        chunks = []
        previous_motion = None
        video_frame_offset = 0
        visible_frames = 0
        chunk_index = 0
        plan = []

        while visible_frames < source_frames:
            comfy.model_management.throw_exception_if_processing_interrupted()
            seed = int(noise_seed) + chunk_index if seed_mode == "increment" else int(noise_seed)
            sample_model = prepared_model
            pose_cache = None
            if enable_pose_cache:
                sample_model = WanAnimate2Cache.execute(
                    model=prepared_model,
                    device=cache_device,
                    dtype=cache_dtype,
                ).args[0]
                pose_cache = sample_model.model_options.get("transformer_options", {}).get("animate2_cache")

            try:
                conditioned = WanAnimate2ToVideo.execute(
                    positive=positive,
                    negative=negative,
                    vae=vae,
                    width=width,
                    height=height,
                    length=chunk_length,
                    batch_size=1,
                    reference_image=conditioned_reference,
                    pose_video=conditioned_pose_video,
                    clip_vision_output=clip_vision_output,
                    positive_pose=positive_pose,
                    clip_vision_output_pose=clip_vision_output_pose,
                    continue_motion=previous_motion,
                    video_frame_offset=int(video_frame_offset),
                    pose_strength=float(pose_strength),
                    pose_start_percent=float(pose_start_percent),
                    pose_end_percent=float(pose_end_percent),
                    reference_image_strength=float(reference_image_strength),
                )
                pos_chunk, neg_chunk, latent, trim_latent, trim_image, video_frame_offset = conditioned.args

                sampled = SamplerCustom.execute(
                    model=sample_model,
                    add_noise=bool(add_noise),
                    noise_seed=seed,
                    cfg=float(cfg),
                    positive=pos_chunk,
                    negative=neg_chunk,
                    sampler=sampler,
                    sigmas=sigmas,
                    latent_image=latent,
                )
                denoised = _trim_video_latent(sampled.args[1], int(trim_latent))
                images = vae.decode(denoised["samples"])
            finally:
                if pose_cache is not None:
                    pose_cache.free()

            if images.ndim == 5:
                images = images.reshape(-1, *images.shape[-3:])
            trim_image = max(0, int(trim_image))
            contribution = images[trim_image:] if trim_image else images
            if int(contribution.shape[0]) <= 0:
                raise RuntimeError(
                    f"Chunk {chunk_index + 1} produced no visible frames "
                    f"(trim_latent={int(trim_latent)}, trim_image={trim_image})."
                )

            remaining = source_frames - visible_frames
            contribution = contribution[:remaining]
            chunks.append(contribution)
            previous_motion = contribution[-1:].detach()
            visible_frames += int(contribution.shape[0])
            plan.append(
                f"#{chunk_index + 1}:len{chunk_length}/trimL{int(trim_latent)}/"
                f"trimI{trim_image}/add{int(contribution.shape[0])}/off{int(video_frame_offset)}"
            )
            print(
                f"[IAMCCS_WanAnimate2Extends] chunk {chunk_index + 1}/{max_chunks} "
                f"seed={seed} trim_latent={int(trim_latent)} trim_image={trim_image} "
                f"visible={visible_frames}/{source_frames} offset={int(video_frame_offset)}",
                flush=True,
            )
            progress.update_absolute(
                chunk_index + 1,
                max_chunks,
                _preview_image_tuple(contribution, live_chunk_preview),
            )
            chunk_index += 1

            if empty_cache_each_chunk:
                comfy.model_management.soft_empty_cache()
            if chunk_index > max_chunks:
                raise RuntimeError("Wan-Animate-2 extender exceeded its planned chunk count.")

        native_images = torch.cat(
            [chunk.to(device=chunks[0].device, dtype=chunks[0].dtype) for chunk in chunks], dim=0
        )[:source_frames]
        final_images = _apply_output_background(
            native_images,
            output_background_mode,
            composite_mask,
            reference_image,
            conditioned_pose_video[:source_frames],
            background_image,
            background_video,
            width,
            height,
        )
        generated_before_trim = chunk_length + max(0, chunk_index - 1) * (chunk_length - 1)
        trimmed_frames = max(0, generated_before_trim - int(native_images.shape[0]))

        return (
            final_images,
            native_images,
            int(final_images.shape[0]),
            int(source_frames),
            int(trimmed_frames),
            " | ".join(plan),
        )


class IAMCCS_WanAnimate2ExtendPlan:
    DESCRIPTION = "Plan Wan-Animate-2 long-video chunks without loading or sampling the model."
    CATEGORY = "IAMCCS/video/WanAnimate-2"
    RETURN_TYPES = ("INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("source_frames", "chunk_count", "generated_frames", "chunk_plan")
    FUNCTION = "plan"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_frames": ("INT", {"default": 161, "min": 1, "max": 100000, "step": 1}),
                "chunk_length": ("INT", {"default": 81, "min": 5, "max": 1025, "step": 4}),
            }
        }

    def plan(self, source_frames, chunk_length):
        chunk_length = _to_4n1(chunk_length)
        plan = _estimate_plan(source_frames, chunk_length)
        generated_frames = chunk_length + max(0, len(plan) - 1) * (chunk_length - 1)
        return int(source_frames), len(plan), int(generated_frames), " | ".join(plan)


NODE_CLASS_MAPPINGS = {
    "IAMCCS_WanAnimate2Extends": IAMCCS_WanAnimate2Extends,
    "IAMCCS_WanAnimate2ExtendPlan": IAMCCS_WanAnimate2ExtendPlan,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_WanAnimate2Extends": "IAMCCS Wan-Animate-2 Extends",
    "IAMCCS_WanAnimate2ExtendPlan": "IAMCCS Wan-Animate-2 Extend Plan",
}
