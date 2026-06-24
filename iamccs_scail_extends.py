import math

import torch

import comfy.model_management
import comfy.utils


def _to_4n1(value):
    return ((max(1, int(value)) - 1) // 4) * 4 + 1


def _plan_chunks(n_frames, chunk_len, overlap):
    n_eff = _to_4n1(n_frames)
    if n_eff <= chunk_len:
        return n_eff, [n_eff]
    step = chunk_len - overlap
    k = math.ceil((n_eff - chunk_len) / step)
    final_len = n_eff - step * k
    return n_eff, [chunk_len] * k + [final_len]


class IAMCCS_ScailExtends:
    DESCRIPTION = (
        "IAMCCS SCAIL-2 long-video sampler. It wraps the official WanSCAILToVideo, "
        "SamplerCustom and ColorTransfer nodes, splitting the source into 81-frame "
        "chunks with a 5-frame SCAIL anchor by default."
    )
    CATEGORY = "IAMCCS/video/SCAIL-2"
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("images", "frame_count", "source_frames", "trimmed_frames", "chunk_plan")
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
                "pose_video": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 32, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 896, "min": 32, "max": 8192, "step": 32}),
                "noise_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xffffffffffffffff,
                        "control_after_generate": True,
                    },
                ),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "chunk_length": (
                    "INT",
                    {
                        "default": 81,
                        "min": 9,
                        "max": 1024,
                        "step": 4,
                        "tooltip": "Max frames per SCAIL chunk. SCAIL-2 was designed around 81.",
                    },
                ),
                "overlap": (
                    "INT",
                    {
                        "default": 5,
                        "min": 1,
                        "max": 81,
                        "step": 4,
                        "tooltip": "Anchor frames from the previous chunk. SCAIL-2 default is 5.",
                    },
                ),
                "seed_mode": (["increment", "fixed"], {"default": "increment"}),
                "continuity_profile": (
                    ["external_1to1", "iamccs_boundary_ramp", "off"],
                    {
                        "default": "off",
                        "tooltip": "external_1to1 matches the source plugin behavior. boundary_ramp only grades the first seam frames.",
                    },
                ),
                "color_method": (
                    ["reinhard_lab", "mkl_lab", "histogram"],
                    {"default": "reinhard_lab"},
                ),
                "boundary_frames": (
                    "INT",
                    {
                        "default": 12,
                        "min": 1,
                        "max": 76,
                        "step": 1,
                        "tooltip": "Only used by iamccs_boundary_ramp.",
                    },
                ),
                "boundary_strength": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Initial seam color correction strength for iamccs_boundary_ramp.",
                    },
                ),
                "dimension_policy": (
                    ["warn", "error"],
                    {
                        "default": "warn",
                        "tooltip": "SCAIL-2 pose/mask conditioning is safest when width and height are divisible by 32.",
                    },
                ),
            },
            "optional": {
                "pose_video_mask": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "reference_image_mask": ("IMAGE",),
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "replacement_mode": ("BOOLEAN", {"default": True}),
                "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "pose_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pose_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "add_noise": ("BOOLEAN", {"default": True}),
                "empty_cache_each_chunk": ("BOOLEAN", {"default": False}),
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
        sampler,
        sigmas,
        pose_video,
        width,
        height,
        noise_seed,
        cfg,
        chunk_length,
        overlap,
        seed_mode,
        continuity_profile,
        color_method,
        boundary_frames,
        boundary_strength,
        dimension_policy,
        pose_video_mask=None,
        reference_image=None,
        reference_image_mask=None,
        clip_vision_output=None,
        replacement_mode=True,
        pose_strength=1.0,
        pose_start=0.0,
        pose_end=1.0,
        add_noise=True,
        empty_cache_each_chunk=False,
    ):
        from comfy_extras.nodes_custom_sampler import SamplerCustom
        from comfy_extras.nodes_post_processing import ColorTransfer
        from comfy_extras.nodes_scail import WanSCAILToVideo

        chunk_length = _to_4n1(chunk_length)
        overlap = _to_4n1(overlap)
        if chunk_length - overlap < 4:
            raise ValueError(
                f"IAMCCS_ScailExtends: chunk_length ({chunk_length}) must exceed overlap ({overlap}) by at least 4."
            )
        if width % 32 != 0 or height % 32 != 0:
            msg = (
                f"IAMCCS_ScailExtends: width/height {width}x{height} are not both divisible by 32. "
                "SCAIL-2 pose/mask conditioning can wrap/pad at unsafe sizes."
            )
            if dimension_policy == "error":
                raise ValueError(msg)
            print("[IAMCCS_ScailExtends] WARNING:", msg)

        source_frames = int(pose_video.shape[0])
        n_eff, lengths = _plan_chunks(source_frames, chunk_length, overlap)
        trimmed_frames = source_frames - n_eff
        chunk_plan = ",".join(str(x) for x in lengths)
        print(
            f"[IAMCCS_ScailExtends] source={source_frames} effective={n_eff} "
            f"trimmed={trimmed_frames} chunks=[{chunk_plan}]"
        )

        pbar = comfy.utils.ProgressBar(len(lengths))
        chunks = []
        prev_anchor_frames = None
        prev_color_ref = None
        offset = 0

        for i, length in enumerate(lengths):
            comfy.model_management.throw_exception_if_processing_interrupted()
            seed = int(noise_seed) + i if seed_mode == "increment" else int(noise_seed)

            cond = WanSCAILToVideo.execute(
                positive=positive,
                negative=negative,
                vae=vae,
                width=width,
                height=height,
                length=length,
                batch_size=1,
                pose_strength=pose_strength,
                pose_start=pose_start,
                pose_end=pose_end,
                video_frame_offset=offset,
                previous_frame_count=overlap,
                replacement_mode=replacement_mode,
                reference_image=reference_image,
                clip_vision_output=clip_vision_output,
                pose_video=pose_video,
                pose_video_mask=pose_video_mask,
                reference_image_mask=reference_image_mask,
                previous_frames=prev_anchor_frames,
            )
            pos_c, neg_c, latent, offset = cond.args

            sampled = SamplerCustom.execute(
                model=model,
                add_noise=add_noise,
                noise_seed=seed,
                cfg=cfg,
                positive=pos_c,
                negative=neg_c,
                sampler=sampler,
                sigmas=sigmas,
                latent_image=latent,
            )
            denoised = sampled.args[1]
            images = vae.decode(denoised["samples"])
            if images.ndim == 5:
                images = images.reshape(-1, *images.shape[-3:])

            if i == 0:
                contrib = images
            else:
                contrib = images[overlap:]
                if continuity_profile != "off" and prev_color_ref is not None:
                    color_node = ColorTransfer
                    if continuity_profile == "external_1to1":
                        contrib = self._match_external(color_node, contrib, prev_color_ref, color_method)
                    else:
                        contrib = self._match_boundary(
                            color_node,
                            contrib,
                            prev_color_ref,
                            color_method,
                            boundary_frames,
                            boundary_strength,
                        )

            chunks.append(contrib)
            prev_anchor_frames = contrib[-overlap:].detach()
            prev_color_ref = contrib[-1:].detach()

            pbar.update(1)
            print(
                f"[IAMCCS_ScailExtends] chunk {i + 1}/{len(lengths)} "
                f"length={length} contributed={contrib.shape[0]} offset={offset}"
            )

            if empty_cache_each_chunk:
                comfy.model_management.soft_empty_cache()

        out = torch.cat([c.to(device=chunks[0].device, dtype=chunks[0].dtype) for c in chunks], dim=0)
        if out.shape[0] > n_eff:
            out = out[:n_eff]
        return (out, int(out.shape[0]), source_frames, trimmed_frames, chunk_plan)


class IAMCCS_ScailExtendPlan:
    DESCRIPTION = "Plan SCAIL-2 long-video chunks without sampling."
    CATEGORY = "IAMCCS/video/SCAIL-2"
    RETURN_TYPES = ("INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("effective_frames", "trimmed_frames", "chunk_count", "chunk_plan")
    FUNCTION = "plan"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_frames": ("INT", {"default": 81, "min": 1, "max": 100000, "step": 1}),
                "chunk_length": ("INT", {"default": 81, "min": 9, "max": 1024, "step": 4}),
                "overlap": ("INT", {"default": 5, "min": 1, "max": 81, "step": 4}),
            }
        }

    def plan(self, source_frames, chunk_length, overlap):
        chunk_length = _to_4n1(chunk_length)
        overlap = _to_4n1(overlap)
        if chunk_length - overlap < 4:
            raise ValueError("chunk_length must exceed overlap by at least 4.")
        n_eff, lengths = _plan_chunks(source_frames, chunk_length, overlap)
        return (n_eff, int(source_frames - n_eff), len(lengths), ",".join(str(x) for x in lengths))


NODE_CLASS_MAPPINGS = {
    "IAMCCS_ScailExtends": IAMCCS_ScailExtends,
    "IAMCCS_ScailExtendPlan": IAMCCS_ScailExtendPlan,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_ScailExtends": "IAMCCS SCAIL Extends",
    "IAMCCS_ScailExtendPlan": "IAMCCS SCAIL Extend Plan",
}
