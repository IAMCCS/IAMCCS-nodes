from __future__ import annotations

"""IAMCCS sampling bridge for the goyAI Influencer applications.

This module is an independent implementation built on ComfyUI's public sampling
API.  It does not import, vendor, or execute RES4LYF code.  The public contract is
deliberately limited to the profiles used by the IAMCCS image applications:

* ``linear/euler`` for the first creative pass;
* ``exponential/ddim`` and ``exponential/res_2s`` for the existing edit/reference
  workflows;
* ``exponential/res_4s_munthe-kaas`` as the application's high-order refinement
  profile, implemented with ComfyUI's maintained residual multistep sampler.

The latter name is retained at the application boundary so existing saved settings
remain readable.  Internally the clean-room implementation uses ComfyUI's native
``res_multistep_ancestral`` (or the closest maintained fallback) and therefore does
not claim bit-identical numerical output to any third-party implementation.
"""

import logging
from typing import Any

import torch

import comfy.model_management as model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview


LOGGER = logging.getLogger("IAMCCS.InfluencerSampler")

MAX_SEED = 0xFFFFFFFFFFFFFFFF
SAMPLER_PROFILES = [
    "linear/euler",
    "exponential/ddim",
    "exponential/res_2s",
    "exponential/res_4s_munthe-kaas",
]
SCHEDULER_PROFILES = [
    "simple",
    "sgm_uniform",
    "karras",
    "exponential",
    "ddim_uniform",
    "beta",
    "normal",
    "linear_quadratic",
    "kl_optimal",
    "bong_tangent",
    "beta57",
]


def _available_samplers() -> list[str]:
    return list(getattr(comfy.samplers.KSampler, "SAMPLERS", ()))


def _available_schedulers() -> list[str]:
    return list(getattr(comfy.samplers.KSampler, "SCHEDULERS", ()))


def _resolve_scheduler(name: str) -> str:
    available = _available_schedulers()
    if name in available:
        return name
    aliases = {
        "beta57": "beta",
        "bong_tangent": "normal",
    }
    candidate = aliases.get(name, "normal")
    if candidate in available:
        LOGGER.warning("Scheduler %s is unavailable; using %s.", name, candidate)
        return candidate
    if not available:
        raise RuntimeError("This ComfyUI build exposes no sampling schedulers.")
    LOGGER.warning("Scheduler %s is unavailable; using %s.", name, available[0])
    return available[0]


def _resolve_sampler(profile: str) -> tuple[str, bool]:
    """Return (ComfyUI sampler name, accepts eta)."""

    available = _available_samplers()
    if profile == "linear/euler":
        candidates = (("euler", False),)
    elif profile == "exponential/ddim":
        candidates = (("ddim", False), ("euler", False))
    elif profile in {
        "exponential/res_2s",
        "exponential/res_4s_munthe-kaas",
    }:
        candidates = (
            ("res_multistep_ancestral", True),
            ("res_multistep", False),
            ("dpmpp_2m", False),
            ("heun", False),
        )
    else:
        raise ValueError(f"Unsupported IAMCCS sampler profile: {profile}")

    for sampler_name, accepts_eta in candidates:
        if sampler_name in available:
            if sampler_name != candidates[0][0]:
                LOGGER.warning(
                    "Preferred sampler for %s is unavailable; using %s.",
                    profile,
                    sampler_name,
                )
            return sampler_name, accepts_eta
    raise RuntimeError(
        f"No compatible ComfyUI sampler is available for IAMCCS profile {profile}."
    )


def _copy_latent(latent: dict[str, Any], samples: torch.Tensor) -> dict[str, Any]:
    result = latent.copy()
    result.pop("downscale_ratio_spacial", None)
    result.pop("downscale_ratio_temporal", None)
    result["samples"] = samples
    return result


def _native_sampler(name: str, eta: float, accepts_eta: bool):
    factory = getattr(comfy.samplers, "ksampler", None)
    if callable(factory):
        options = {"eta": float(eta), "s_noise": 1.0} if accepts_eta else {}
        return factory(name, extra_options=options)
    factory = getattr(comfy.samplers, "sampler_object", None)
    if callable(factory):
        if accepts_eta:
            LOGGER.warning(
                "This ComfyUI version cannot forward eta to %s; using its native default.",
                name,
            )
        return factory(name)
    raise RuntimeError("This ComfyUI build exposes no native sampler factory.")


class IAMCCSInfluencerSampler:
    """Two-profile sampler used by goyAI Influencer Studio and Identity."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "eta": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "Ancestral noise strength for the refinement profile.",
                    },
                ),
                "sampler_name": (SAMPLER_PROFILES, {"default": "linear/euler"}),
                "scheduler": (SCHEDULER_PROFILES, {"default": "beta"}),
                "steps": ("INT", {"default": 15, "min": 1, "max": 10000}),
                "steps_to_run": ("INT", {"default": -1, "min": -1, "max": 10000}),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "cfg": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": MAX_SEED,
                        "control_after_generate": True,
                    },
                ),
                "sampler_mode": (["standard"], {"default": "standard"}),
                "bongmath": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Compatibility field retained for saved IAMCCS settings.",
                    },
                ),
            },
            "optional": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "sigmas": ("SIGMAS",),
                "guides": ("GUIDES",),
                "options": ("OPTIONS",),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT", "OPTIONS")
    RETURN_NAMES = ("output", "denoised", "options")
    FUNCTION = "sample"
    CATEGORY = "IAMCCS/AI Influencer"
    DESCRIPTION = (
        "Independent ComfyUI-native dual-stage sampler for the IAMCCS AI Influencer apps."
    )

    def sample(
        self,
        eta: float,
        sampler_name: str,
        scheduler: str,
        steps: int,
        steps_to_run: int,
        denoise: float,
        cfg: float,
        seed: int,
        sampler_mode: str,
        bongmath: bool,
        model=None,
        positive=None,
        negative=None,
        latent_image=None,
        sigmas=None,
        guides=None,
        options=None,
    ):
        del guides, bongmath
        if sampler_mode != "standard":
            raise ValueError("IAMCCS Influencer Sampler supports standard mode only.")
        if model is None or positive is None or negative is None or latent_image is None:
            raise ValueError("Model, positive, negative and latent_image are required.")
        if not isinstance(latent_image, dict) or "samples" not in latent_image:
            raise TypeError("latent_image must be a ComfyUI LATENT dictionary.")

        profile = str(sampler_name)
        native_name, accepts_eta = _resolve_sampler(profile)
        native_scheduler = _resolve_scheduler(str(scheduler))
        steps = max(1, int(steps))
        run_steps = steps if int(steps_to_run) < 0 else max(1, min(steps, int(steps_to_run)))
        denoise = min(1.0, max(0.0, float(denoise)))

        latent_samples = comfy.sample.fix_empty_latent_channels(
            model,
            latent_image["samples"],
            latent_image.get("downscale_ratio_spacial"),
            latent_image.get("downscale_ratio_temporal"),
        )
        batch_inds = latent_image.get("batch_index")
        noise = comfy.sample.prepare_noise(latent_samples, int(seed), batch_inds)
        noise_mask = latent_image.get("noise_mask")

        schedule = comfy.samplers.KSampler(
            model,
            steps=steps,
            device=model.load_device,
            sampler=native_name,
            scheduler=native_scheduler,
            denoise=denoise,
            model_options=model.model_options,
        )
        sigma_values = sigmas if sigmas is not None else schedule.sigmas
        if not isinstance(sigma_values, torch.Tensor):
            raise TypeError("sigmas must be a torch.Tensor when supplied.")
        sigma_values = sigma_values[: run_steps + 1]
        if sigma_values.numel() < 2:
            raise ValueError("The selected step range contains no sampling interval.")

        preview_callback = latent_preview.prepare_callback(model, run_steps)
        last_denoised: list[torch.Tensor | None] = [None]

        def callback(step, x0, x, total_steps):
            last_denoised[0] = x0.detach()
            if preview_callback is not None:
                preview_callback(step, x0, x, total_steps)

        # Preserve UI progress and ComfyUI's interruption support through its own
        # maintained sample_custom execution path.
        native_sampler = _native_sampler(native_name, float(eta), accepts_eta)
        sampled = comfy.sample.sample_custom(
            model,
            noise,
            float(cfg),
            native_sampler,
            sigma_values,
            positive,
            negative,
            latent_samples,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
            seed=int(seed),
        )

        intermediate_device = model_management.intermediate_device()
        intermediate_dtype = model_management.intermediate_dtype()
        sampled = sampled.to(device=intermediate_device, dtype=intermediate_dtype)
        clean = last_denoised[0] if last_denoised[0] is not None else sampled
        clean = clean.to(device=intermediate_device, dtype=intermediate_dtype)

        option_out = dict(options) if isinstance(options, dict) else {}
        option_out["iamccs_influencer_sampler"] = {
            "profile": profile,
            "native_sampler": native_name,
            "scheduler": native_scheduler,
            "steps": steps,
            "steps_to_run": run_steps,
            "eta": float(eta),
        }
        LOGGER.info(
            "Completed profile=%s native=%s scheduler=%s steps=%d/%d seed=%d",
            profile,
            native_name,
            native_scheduler,
            run_steps,
            steps,
            int(seed),
        )
        return (
            _copy_latent(latent_image, sampled),
            _copy_latent(latent_image, clean),
            option_out,
        )


class IAMCCSInfluencerLatent:
    """V3-compatible latent/encode bridge used by the Krea application graph.

    Studio and Identity consume ``empty_latent`` (output index 3).  The image,
    mask, and latent inputs are implemented as useful compatibility paths for
    imported IAMCCS workflows without depending on third-party VAE utilities.
    """

    INTERPOLATIONS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    METHODS = ["stretch", "fill / crop"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resize_to_input": (
                    ["false", "image_1", "image_2", "mask", "latent"],
                    {"default": "false"},
                ),
                "width": ("INT", {"default": 1024, "min": 8, "max": 16384, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 8, "max": 16384, "step": 8}),
                "mask_channel": (["red", "green", "blue", "alpha"], {"default": "red"}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "latent_type": (["4_channels", "16_channels"], {"default": "16_channels"}),
                "interpolation": (cls.INTERPOLATIONS, {"default": "lanczos"}),
                "method": (cls.METHODS, {"default": "fill / crop"}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "mask": ("IMAGE",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT", "MASK", "LATENT", "INT", "INT")
    RETURN_NAMES = ("latent_1", "latent_2", "mask", "empty_latent", "width", "height")
    FUNCTION = "encode"
    CATEGORY = "IAMCCS/AI Influencer"
    DESCRIPTION = "ComfyUI-native Krea latent and optional VAE encode bridge."

    @staticmethod
    def _latent_size(latent: Any) -> tuple[int, int] | None:
        try:
            samples = latent["samples"]
            return int(samples.shape[-1] * 8), int(samples.shape[-2] * 8)
        except (KeyError, TypeError, AttributeError, IndexError):
            return None

    @staticmethod
    def _image_size(image: Any) -> tuple[int, int] | None:
        if isinstance(image, torch.Tensor) and image.ndim >= 3:
            return int(image.shape[-2]), int(image.shape[-3])
        return None

    @staticmethod
    def _resize(image: torch.Tensor, width: int, height: int, interpolation: str, method: str):
        samples = image.movedim(-1, 1)
        crop = "center" if method == "fill / crop" else "disabled"
        resized = comfy.utils.common_upscale(samples, width, height, interpolation, crop)
        return resized.movedim(1, -1)

    @staticmethod
    def _encode_image(image: torch.Tensor | None, vae: Any, fallback: dict[str, Any]):
        if image is None:
            return fallback.copy()
        if vae is None:
            raise ValueError("A VAE is required when an image input is connected.")
        pixels = image[..., :3]
        return {"samples": vae.encode(pixels)}

    def encode(
        self,
        resize_to_input: str,
        width: int,
        height: int,
        mask_channel: str,
        invert_mask: bool,
        latent_type: str,
        interpolation: str = "lanczos",
        method: str = "fill / crop",
        image_1=None,
        image_2=None,
        mask=None,
        latent=None,
        vae=None,
    ):
        selected_size = None
        if resize_to_input == "image_1":
            selected_size = self._image_size(image_1)
        elif resize_to_input == "image_2":
            selected_size = self._image_size(image_2)
        elif resize_to_input == "mask":
            selected_size = self._image_size(mask)
        elif resize_to_input == "latent":
            selected_size = self._latent_size(latent)
        if selected_size:
            width, height = selected_size

        width = max(8, int(width) // 8 * 8)
        height = max(8, int(height) // 8 * 8)
        channels = 16 if latent_type == "16_channels" else 4
        empty_samples = torch.zeros(
            (1, channels, height // 8, width // 8),
            device=model_management.intermediate_device(),
            dtype=model_management.intermediate_dtype(),
        )
        empty = {"samples": empty_samples}

        if isinstance(image_1, torch.Tensor):
            image_1 = self._resize(image_1, width, height, interpolation, method)
        if isinstance(image_2, torch.Tensor):
            image_2 = self._resize(image_2, width, height, interpolation, method)

        if isinstance(mask, torch.Tensor):
            mask = self._resize(mask, width, height, interpolation, method)
            channel_index = {"red": 0, "green": 1, "blue": 2, "alpha": 3}[mask_channel]
            channel_index = min(channel_index, mask.shape[-1] - 1)
            mask_out = mask[..., channel_index]
        else:
            mask_out = torch.zeros(
                (1, height, width),
                device=model_management.intermediate_device(),
                dtype=torch.float32,
            )
        if invert_mask:
            mask_out = 1.0 - mask_out

        latent_1 = self._encode_image(image_1, vae, empty)
        latent_2 = self._encode_image(image_2, vae, empty)
        return latent_1, latent_2, mask_out, empty, width, height


NODE_CLASS_MAPPINGS = {
    "IAMCCS_InfluencerSampler": IAMCCSInfluencerSampler,
    "IAMCCS_InfluencerLatent": IAMCCSInfluencerLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_InfluencerSampler": "IAMCCS Influencer Sampler",
    "IAMCCS_InfluencerLatent": "IAMCCS Influencer Latent / VAE",
}
