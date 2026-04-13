import math
import os
import re

import numpy as np
import torch
from PIL import Image as _PILImage

import comfy.utils
import folder_paths
import node_helpers


RESOLUTION_PRESETS = [
    "512x512",
    "720x1280",
    "768x768",
    "768x1024",
    "832x1216",
    "864x1536",
    "1024x768",
    "1024x1024",
    "1024x1536",
    "1280x720",
    "1360x768",
    "1216x832",
    "1536x1024",
    "1536x864",
    "1920x1080",
]


def _safe_unpack(value):
    current = value
    while isinstance(current, tuple) and len(current) == 1:
        current = current[0]
    if hasattr(current, "args"):
        args = getattr(current, "args", None)
        if isinstance(args, (tuple, list)) and len(args) > 0:
            return _safe_unpack(args[0])
    return current


def _encode_text(clip, prompt):
    tokens = clip.tokenize(prompt or "")
    return clip.encode_from_tokens_scheduled(tokens)


def _apply_reference_latent(conditioning, latent, method):
    conditioning = node_helpers.conditioning_set_values(
        conditioning,
        {"reference_latents": [latent]},
        append=True,
    )
    if "uxo" in method or "uso" in method:
        method = "uxo"
    return node_helpers.conditioning_set_values(
        conditioning,
        {"reference_latents_method": method},
    )


def _prompt_to_slug(prompt, max_words=5):
    text = re.sub(r"<[^>]+>", "", prompt or "").strip()
    words = [re.sub(r"[^a-zA-Z0-9]", "", word) for word in text.split()[:max_words] if word]
    filtered = [word for word in words if word]
    return "_".join(filtered) if filtered else "prompt"


def _unique_png_path(output_dir, prefix):
    counter = 1
    while True:
        path = os.path.join(output_dir, f"{prefix}_{counter:05d}.png")
        if not os.path.exists(path):
            return path
        counter += 1


def _round_to_multiple(value, multiple=16):
    return max(multiple, int(round(float(value) / float(multiple))) * multiple)


def _parse_resolution_preset(preset):
    try:
        width_text, height_text = str(preset).lower().split("x", 1)
        return int(width_text.strip()), int(height_text.strip())
    except Exception as exc:
        raise ValueError(f"Invalid resolution preset: {preset}") from exc


def _resize_image_to_dimensions(image, width, height, upscale_method):
    target_width = _round_to_multiple(width)
    target_height = _round_to_multiple(height)
    samples = image.movedim(-1, 1)
    scaled = comfy.utils.common_upscale(samples, target_width, target_height, upscale_method, "disabled")
    return scaled.movedim(1, -1)


def _scale_image_to_megapixels(image, target_megapixels, upscale_method):
    if target_megapixels <= 0:
        return image

    height = int(image.shape[1])
    width = int(image.shape[2])
    total_pixels = max(1.0, float(height * width))
    target_pixels = max(256.0, float(target_megapixels) * 1000000.0)
    scale = math.sqrt(target_pixels / total_pixels)
    target_width = _round_to_multiple(width * scale)
    target_height = _round_to_multiple(height * scale)

    samples = image.movedim(-1, 1)
    scaled = comfy.utils.common_upscale(samples, target_width, target_height, upscale_method, "disabled")
    return scaled.movedim(1, -1)


def _prepare_flux_image(image, resolution_mode, resolution_preset, custom_width, custom_height, target_megapixels, upscale_method):
    mode = (resolution_mode or "auto").strip().lower()

    if mode == "auto":
        return _scale_image_to_megapixels(image, float(target_megapixels), upscale_method)

    if mode == "match_input":
        return _resize_image_to_dimensions(image, int(image.shape[2]), int(image.shape[1]), upscale_method)

    if mode == "preset":
        width, height = _parse_resolution_preset(resolution_preset)
        return _resize_image_to_dimensions(image, width, height, upscale_method)

    if mode == "custom":
        return _resize_image_to_dimensions(image, int(custom_width), int(custom_height), upscale_method)

    return _scale_image_to_megapixels(image, float(target_megapixels), upscale_method)


def _build_flux2_latent(width, height, batch_size):
    try:
        import nodes

        node_cls = getattr(nodes, "EmptyFlux2LatentImage", None)
        if node_cls is not None:
            instance = node_cls()
            if hasattr(instance, "generate"):
                return _safe_unpack(instance.generate(width, height, batch_size))
            if hasattr(instance, "execute"):
                return _safe_unpack(instance.execute(width, height, batch_size))
            if hasattr(node_cls, "execute"):
                return _safe_unpack(node_cls.execute(width, height, batch_size))
    except Exception:
        pass

    try:
        from comfy_extras.nodes_flux import EmptyFlux2LatentImage

        return _safe_unpack(EmptyFlux2LatentImage.execute(width, height, batch_size))
    except Exception:
        return None


def _sample_flux2(model, positive, negative, width, height, steps, cfg, sampler_name, seed):
    latent_image = _build_flux2_latent(width, height, 1)
    if latent_image is None:
        raise RuntimeError("EmptyFlux2LatentImage is not available")

    try:
        from comfy_extras.nodes_custom_sampler import CFGGuider, KSamplerSelect, RandomNoise, SamplerCustomAdvanced
        from comfy_extras.nodes_flux import Flux2Scheduler

        noise = _safe_unpack(RandomNoise.execute(int(seed)))
        guider = _safe_unpack(CFGGuider.execute(model, positive, negative, float(cfg)))
        sampler = _safe_unpack(KSamplerSelect.execute(str(sampler_name)))
        sigmas = _safe_unpack(Flux2Scheduler.execute(int(steps), int(width), int(height)))
        return _safe_unpack(SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent_image))
    except Exception as exc:
        raise RuntimeError(f"Flux2 advanced sampler path failed: {exc}") from exc


def _decode_latent(vae, latent):
    latent_samples = latent.get("samples") if isinstance(latent, dict) else latent
    decoded = vae.decode(latent_samples)
    if decoded.ndim == 5:
        decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
    return decoded.cpu()


class IAMCCS_FluxKleinMultiGen:
    DISPLAY_NAME = "IAMCCS Flux Klein Multi-Gen"
    CATEGORY = "IAMCCS/Flux"
    FUNCTION = "generate"

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "count")

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import comfy.samplers

            samplers = comfy.samplers.KSampler.SAMPLERS
        except Exception:
            samplers = ["euler"]

        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "multi_prompt": ("STRING", {"forceInput": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "sampler_name": (samplers,),
            },
            "optional": {
                "separator": ("STRING", {"default": "\\n", "multiline": False}),
                "reference_latents_method": (
                    ["index_timestep_zero", "offset", "index", "uxo/uno"],
                    {"default": "index_timestep_zero"},
                ),
                "target_megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 16.0, "step": 0.1}),
                "upscale_method": (["lanczos", "nearest-exact", "bilinear", "area", "bicubic"], {"default": "lanczos"}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "output_prefix": ("STRING", {"default": "flux_klein_multi"}),
                "save_images": ("BOOLEAN", {"default": False}),
                "resolution_mode": (["auto", "match_input", "preset", "custom"], {"default": "auto"}),
                "resolution_preset": (RESOLUTION_PRESETS, {"default": "1024x1024"}),
                "custom_width": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
                "custom_height": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
            },
        }

    def generate(
        self,
        model,
        clip,
        vae,
        image,
        multi_prompt,
        seed,
        steps,
        cfg,
        sampler_name,
        separator="\\n",
        reference_latents_method="index_timestep_zero",
        target_megapixels=1.0,
        upscale_method="lanczos",
        negative_prompt="",
        output_prefix="flux_klein_multi",
        save_images=False,
        resolution_mode="auto",
        resolution_preset="1024x1024",
        custom_width=1024,
        custom_height=1024,
    ):
        sep = separator.replace("\\n", "\n")
        prompts = [prompt.strip() for prompt in (multi_prompt or "").split(sep) if prompt.strip()]

        if not prompts:
            blank = torch.zeros((1, image.shape[1], image.shape[2], 3), dtype=torch.float32)
            return (blank, 0)

        prepared_image = _prepare_flux_image(
            image,
            resolution_mode,
            resolution_preset,
            custom_width,
            custom_height,
            target_megapixels,
            upscale_method,
        )
        reference_latent = vae.encode(prepared_image[:, :, :, :3])
        output_dir = folder_paths.get_output_directory() if save_images else None
        results = []

        for index, prompt in enumerate(prompts):
            positive = _encode_text(clip, prompt)
            negative = _encode_text(clip, negative_prompt)
            positive = _apply_reference_latent(positive, reference_latent, reference_latents_method)
            negative = _apply_reference_latent(negative, reference_latent, reference_latents_method)

            width = int(prepared_image.shape[2])
            height = int(prepared_image.shape[1])
            latent = _sample_flux2(
                model,
                positive,
                negative,
                width,
                height,
                int(steps),
                float(cfg),
                sampler_name,
                int(seed) + index,
            )
            decoded = _decode_latent(vae, latent)
            results.append(decoded)

            if save_images and output_dir is not None:
                slug = _prompt_to_slug(prompt)
                prefix = f"{output_prefix}_{slug}"
                path = _unique_png_path(output_dir, prefix)
                arr = (decoded[0].numpy() * 255).clip(0, 255).astype(np.uint8)
                _PILImage.fromarray(arr).save(path)

        return (torch.cat(results, dim=0), len(prompts))