"""IAMCCS Krea 2 Identity Multi-Gen.

Runs one Krea 2 image-edit generation for every non-empty prompt in
``multi_prompt``.  It intentionally composes the installed Krea 2 Edit,
Conditioning Rebalance and RES4LYF nodes instead of duplicating their logic.
"""

import json
import secrets

import torch

import nodes
import comfy.model_management


def _runtime_node(name):
    cls = nodes.NODE_CLASS_MAPPINGS.get(name)
    if cls is None:
        raise RuntimeError(
            f"Required node '{name}' is not installed. Install/update the Krea 2 "
            "Edit, Conditioning Rebalance and RES4LYF custom nodes."
        )
    return cls()


def _split_prompts(text, separator):
    sep = str(separator or "\\n").replace("\\n", "\n")
    if not sep:
        sep = "\n"
    return [part.strip() for part in str(text or "").split(sep) if part.strip()]


class IAMCCS_Krea2MultiGen:
    DISPLAY_NAME = "IAMCCS Krea 2 Identity Multi-Gen"
    CATEGORY = "IAMCCS/Identity"
    FUNCTION = "generate"

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("images", "count", "report")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "multi_prompt": ("STRING", {"forceInput": True, "multiline": True}),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                }),
                "seed_mode": (
                    ["fixed", "increment", "random"],
                    {"default": "fixed"},
                ),
                "width": ("INT", {
                    "default": 1024, "min": 64, "max": 8192, "step": 16,
                }),
                "height": ("INT", {
                    "default": 1024, "min": 64, "max": 8192, "step": 16,
                }),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01,
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "separator": ("STRING", {"default": "\\n"}),
                "negative_prompt": ("STRING", {
                    "default": "", "multiline": True,
                }),
                "ref_boost": ("FLOAT", {
                    "default": 1.15, "min": 0.0, "max": 1000.0, "step": 0.01,
                }),
                "ref_boost_a": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01,
                }),
                "grounding_px": ("INT", {
                    "default": 768, "min": 0, "max": 4096, "step": 64,
                }),
                "system_prompt": ("STRING", {
                    "default": (
                        "Prioritize facial geometry, apparent adult age, eye color, "
                        "hair characteristics, skin tone, distinctive facial marks "
                        "and body proportions. Preserve identity while following "
                        "the requested pose, framing and environment."
                    ),
                    "multiline": True,
                }),
                "rebalance_multiplier": ("FLOAT", {
                    "default": 4.0, "min": -1000000000.0,
                    "max": 1000000000.0, "step": 0.01,
                }),
                "per_layer_weights": ("STRING", {
                    "default": "1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0",
                }),
                "eta": ("FLOAT", {
                    "default": 0.5, "min": -100.0, "max": 100.0, "step": 0.01,
                }),
                "sampler_name": ("STRING", {"default": "exponential/ddim"}),
                "scheduler": ("STRING", {"default": "beta57"}),
            },
            "optional": {
                "image_b": ("IMAGE",),
            },
        }

    def generate(
        self, model, clip, vae, image, multi_prompt, seed, seed_mode,
        width, height, steps, cfg, denoise, separator, negative_prompt,
        ref_boost, ref_boost_a, grounding_px, system_prompt,
        rebalance_multiplier, per_layer_weights, eta, sampler_name, scheduler,
        image_b=None,
    ):
        prompts = _split_prompts(multi_prompt, separator)
        if not prompts:
            blank = torch.zeros((1, int(height), int(width), 3), dtype=torch.float32)
            return (blank, 0, json.dumps({"status": "empty", "prompts": 0}))

        vae_encoder = nodes.VAEEncode()
        source_latent = vae_encoder.encode(vae, image)[0]
        source_latent_b = (
            vae_encoder.encode(vae, image_b)[0] if image_b is not None else None
        )

        patcher = _runtime_node("Krea2EditModelPatch")
        patched_model = patcher.patch(
            model=model,
            source_latent=source_latent,
            source_latent_b=source_latent_b,
            ref_boost=ref_boost,
            ref_boost_a=ref_boost_a,
            vae=vae,
            source_image=image,
            source_image_b=image_b,
            fit_mode="fit",
        )[0]

        encoder = _runtime_node("Krea2EditGroundedEncode")
        rebalance = _runtime_node("ConditioningKrea2Rebalance")
        sampler = _runtime_node("ClownsharKSampler_Beta")
        decoder = nodes.VAEDecode()

        negative = encoder.encode(
            clip=clip, prompt=negative_prompt, image=image, image_b=image_b,
            grounding_px=grounding_px, system_prompt=system_prompt,
        )[0]

        results = []
        used_seeds = []
        for index, prompt in enumerate(prompts):
            if seed_mode == "random":
                current_seed = secrets.randbelow(0xFFFFFFFFFFFFFFFF)
            elif seed_mode == "increment":
                current_seed = (int(seed) + index) & 0xFFFFFFFFFFFFFFFF
            else:
                current_seed = int(seed)
            used_seeds.append(current_seed)

            positive = encoder.encode(
                clip=clip, prompt=prompt, image=image, image_b=image_b,
                grounding_px=grounding_px, system_prompt=system_prompt,
            )[0]
            positive = rebalance.main(
                positive, rebalance_multiplier, per_layer_weights
            )[0]
            latent = {
                "samples": torch.zeros(
                    [1, 16, int(height) // 8, int(width) // 8],
                    device=comfy.model_management.intermediate_device(),
                    dtype=comfy.model_management.intermediate_dtype(),
                ),
                "downscale_ratio_spacial": 8,
            }
            sampled = sampler.main(
                model=patched_model,
                positive=positive,
                negative=negative,
                latent_image=latent,
                eta=eta,
                sampler_name=sampler_name,
                scheduler=scheduler,
                steps=steps,
                steps_to_run=-1,
                denoise=denoise,
                cfg=cfg,
                seed=current_seed,
                sampler_mode="standard",
                bongmath=True,
            )[0]
            decoded = decoder.decode(vae, sampled)[0]
            if decoded.ndim == 5:
                decoded = decoded.flatten(0, 1)
            results.append(decoded.cpu())

        report = json.dumps({
            "status": "ok",
            "prompts": len(prompts),
            "seeds": used_seeds,
            "references": 2 if image_b is not None else 1,
        })
        return (torch.cat(results, dim=0), len(prompts), report)
