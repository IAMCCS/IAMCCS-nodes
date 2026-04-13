"""
IAMCCS Qwen Multi-Gen Node
Generates up to 8 images in a single workflow execution by looping over the
multiline prompt variants produced by IAMCCS QE PromptEnhancer.

This implementation is hosted in IAMCCS-nodes so existing workflows can keep
using the same node type name while the runtime ownership moves out of the QE
repository.
"""

import math
import os
import re

import numpy as np
import torch
from PIL import Image as _PILImage

import comfy.sample
import comfy.samplers
import comfy.utils
import folder_paths
import latent_preview
import node_helpers


def _encode_qwen(clip, vae, image, prompt: str):
    llama_template = (
        "<|im_start|>system\n"
        "Describe the key features of the input image (color, shape, size, "
        "texture, objects, background), then explain how the user's text "
        "instruction should alter or modify the image. Generate a new image "
        "that meets the user's requirements while maintaining consistency with "
        "the original input where appropriate.<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    ref_latents = []
    images_vl = []
    image_prompt = ""

    if image is not None:
        samples = image.movedim(-1, 1)

        total_vl = int(384 * 384)
        scale_vl = math.sqrt(total_vl / (samples.shape[3] * samples.shape[2]))
        w_vl = round(samples.shape[3] * scale_vl)
        h_vl = round(samples.shape[2] * scale_vl)
        s_vl = comfy.utils.common_upscale(samples, w_vl, h_vl, "area", "disabled")
        images_vl.append(s_vl.movedim(1, -1))

        if vae is not None:
            total_lat = int(1024 * 1024)
            scale_lat = math.sqrt(total_lat / (samples.shape[3] * samples.shape[2]))
            w_lat = round(samples.shape[3] * scale_lat / 8.0) * 8
            h_lat = round(samples.shape[2] * scale_lat / 8.0) * 8
            s_lat = comfy.utils.common_upscale(samples, w_lat, h_lat, "area", "disabled")
            ref_latents.append(vae.encode(s_lat.movedim(1, -1)[:, :, :, :3]))

        image_prompt = "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"

    tokens = clip.tokenize(
        image_prompt + prompt,
        images=images_vl,
        llama_template=llama_template,
    )
    cond = clip.encode_from_tokens_scheduled(tokens)

    if ref_latents:
        cond = node_helpers.conditioning_set_values(
            cond, {"reference_latents": ref_latents}, append=True
        )
    return cond


def _apply_ref_method(cond, method: str):
    if "uxo" in method or "uso" in method:
        method = "uxo"
    return node_helpers.conditioning_set_values(
        cond, {"reference_latents_method": method}
    )


def _prompt_to_slug(prompt: str, max_words: int = 5) -> str:
    text = re.sub(r"<[^>]+>", "", prompt).strip()
    words = [re.sub(r"[^a-zA-Z0-9]", "", w) for w in text.split()[:max_words] if w]
    filtered = [w for w in words if w]
    return "_".join(filtered) if filtered else "prompt"


def _unique_png_path(output_dir: str, prefix: str) -> str:
    counter = 1
    while True:
        path = os.path.join(output_dir, f"{prefix}_{counter:05d}.png")
        if not os.path.exists(path):
            return path
        counter += 1


class IAMCCS_QwenMultiGen:
    DISPLAY_NAME = "IAMCCS Qwen Multi-Gen"
    CATEGORY = "IAMCCS/Qwen"
    FUNCTION = "generate"

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "count")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "multi_prompt": ("STRING", {"forceInput": True}),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                }),
                "steps": ("INT", {"default": 4, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01,
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
            },
            "optional": {
                "separator": ("STRING", {
                    "default": "\\n",
                    "multiline": False,
                }),
                "reference_latents_method": (
                    ["index_timestep_zero", "offset", "index", "uxo/uno"],
                    {"default": "index_timestep_zero"},
                ),
                "output_prefix": ("STRING", {"default": "qwen_multi"}),
                "save_images": ("BOOLEAN", {"default": True}),
            },
        }

    def generate(
        self,
        model, clip, vae, image, multi_prompt,
        seed, steps, cfg, sampler_name, scheduler, denoise=1.0,
        separator="\\n",
        reference_latents_method="index_timestep_zero",
        output_prefix="qwen_multi",
        save_images=True,
    ):
        sep = separator.replace("\\n", "\n")
        prompts = [p.strip() for p in multi_prompt.split(sep) if p.strip()]

        if not prompts:
            blank = torch.zeros((1, image.shape[1], image.shape[2], 3))
            return (blank, 0)

        neg_cond = _encode_qwen(clip, vae, image, "")
        neg_cond = _apply_ref_method(neg_cond, reference_latents_method)
        latent_base = vae.encode(image[:, :, :, :3])

        output_dir = folder_paths.get_output_directory() if save_images else None
        results = []

        for index, prompt in enumerate(prompts):
            pos_cond = _encode_qwen(clip, vae, image, prompt)
            pos_cond = _apply_ref_method(pos_cond, reference_latents_method)

            noise = comfy.sample.prepare_noise(latent_base, seed + index, None)
            callback = latent_preview.prepare_callback(model, steps)
            samples = comfy.sample.sample(
                model, noise, steps, cfg,
                sampler_name, scheduler,
                pos_cond, neg_cond, latent_base,
                denoise=denoise,
                callback=callback,
                disable_pbar=False,
                seed=seed + index,
            )

            decoded = vae.decode(samples)
            if decoded.ndim == 5:
                decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
            decoded = decoded.cpu()
            results.append(decoded)

            if save_images and output_dir is not None:
                slug = _prompt_to_slug(prompt)
                prefix = f"{output_prefix}_{slug}"
                path = _unique_png_path(output_dir, prefix)
                arr = (decoded[0].numpy() * 255).clip(0, 255).astype(np.uint8)
                _PILImage.fromarray(arr).save(path)

        images_out = torch.cat(results, dim=0)
        return (images_out, len(prompts))