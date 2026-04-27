"""IAMCCS_FluxKleinMultiGen - Complete pipeline reconstruction.

Pipeline graph (mirrors working reference workflow 1:1):
  image
    -> ImageScaleToTotalPixels(upscale_method, megapixels, resolution_steps)
    -> scaled_image

  scaled_image + vae
    -> VAEEncode
    -> reference_latent          <- shared between positive AND negative

  clip + positive_prompt  -> CLIPTextEncode -> pos_cond
  pos_cond + reference_latent  -> ReferenceLatent -> positive_with_ref

  clip + negative_prompt  -> CLIPTextEncode -> neg_cond
  neg_cond + reference_latent  -> ReferenceLatent -> negative_with_ref

  output_width, output_height -> Flux2Scheduler(steps, width, height)    -> sigmas
  output_width, output_height -> EmptyFlux2LatentImage(width, height, 1) -> latent_image
  seed         -> RandomNoise     -> noise
  sampler_name -> KSamplerSelect  -> sampler
  model + positive_with_ref + negative_with_ref + cfg -> CFGGuider -> guider
  noise + guider + sampler + sigmas + latent_image -> SamplerCustomAdvanced -> output_latent
  output_latent + vae -> VAEDecode -> image_out
"""

import json
import os
import re

import numpy as np
import torch
from PIL import Image as _PILImage

import folder_paths
import node_helpers


# ---------------------------------------------------------------------------
# Unpack helpers
# io.NodeOutput stores outputs in .args; old-style nodes return plain tuples.
# ---------------------------------------------------------------------------

def _unpack(value):
    """Unwrap io.NodeOutput or single-element tuple to the actual value."""
    if hasattr(value, "args"):          # io.NodeOutput (new ComfyUI API)
        if value.args:
            return value.args[0]
        return value
    if isinstance(value, (tuple, list)) and len(value) == 1:
        return value[0]
    return value


def _unpack_sampler(value):
    """SamplerCustomAdvanced returns (output, denoised_output); take output (index 0)."""
    if hasattr(value, "args") and value.args:
        return value.args[0]
    if isinstance(value, (tuple, list)):
        return value[0]
    return value


# ---------------------------------------------------------------------------
# One function per node in the working workflow (exact mirrors)
# ---------------------------------------------------------------------------

def _node_ImageScaleToTotalPixels(image, upscale_method, megapixels, resolution_steps):
    """Node: ImageScaleToTotalPixels"""
    from comfy_extras.nodes_post_processing import ImageScaleToTotalPixels
    return _unpack(
        ImageScaleToTotalPixels.execute(
            image,
            str(upscale_method),
            float(megapixels),
            int(resolution_steps),
        )
    )


def _node_VAEEncode(vae, pixels):
    """Node: VAEEncode"""
    import nodes
    return _unpack(nodes.VAEEncode().encode(vae, pixels))


def _node_CLIPTextEncode(clip, text):
    """Node: CLIPTextEncode"""
    import nodes
    return _unpack(nodes.CLIPTextEncode().encode(clip, text or ""))


def _node_ReferenceLatent(conditioning, latent):
    """Node: ReferenceLatent  -  latent must be a {'samples': tensor} dict"""
    from comfy_extras.nodes_edit_model import ReferenceLatent
    return _unpack(ReferenceLatent.execute(conditioning, latent))


def _node_Flux2Scheduler(steps, width, height):
    """Node: Flux2Scheduler"""
    from comfy_extras.nodes_flux import Flux2Scheduler
    return _unpack(Flux2Scheduler.execute(int(steps), int(width), int(height)))


def _node_EmptyFlux2LatentImage(width, height):
    """Node: EmptyFlux2LatentImage  (batch_size always 1 per iteration)"""
    from comfy_extras.nodes_flux import EmptyFlux2LatentImage
    return _unpack(EmptyFlux2LatentImage.execute(int(width), int(height), 1))


def _node_RandomNoise(seed):
    """Node: RandomNoise"""
    from comfy_extras.nodes_custom_sampler import RandomNoise
    return _unpack(RandomNoise.execute(int(seed)))


def _node_KSamplerSelect(sampler_name):
    """Node: KSamplerSelect"""
    from comfy_extras.nodes_custom_sampler import KSamplerSelect
    return _unpack(KSamplerSelect.execute(str(sampler_name)))


def _node_CFGGuider(model, positive, negative, cfg):
    """Node: CFGGuider"""
    from comfy_extras.nodes_custom_sampler import CFGGuider
    return _unpack(CFGGuider.execute(model, positive, negative, float(cfg)))


def _node_SamplerCustomAdvanced(noise, guider, sampler, sigmas, latent_image):
    """Node: SamplerCustomAdvanced  -  returns first output slot (output latent)"""
    from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
    return _unpack_sampler(
        SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent_image)
    )


def _node_VAEDecode(vae, latent):
    """Node: VAEDecode"""
    import nodes
    return _unpack(nodes.VAEDecode().decode(vae, latent))


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def _shape(t):
    return list(t.shape) if hasattr(t, "shape") else None


def _write_debug(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def _unique_path(directory, prefix, ext):
    i = 1
    while True:
        p = os.path.join(directory, f"{prefix}_{i:05d}.{ext}")
        if not os.path.exists(p):
            return p
        i += 1


def _prompt_slug(text, max_words=5):
    words = [re.sub(r"[^a-zA-Z0-9]", "", w) for w in re.sub(r"<[^>]+>", "", text or "").split()]
    words = [w for w in words if w][:max_words]
    return "_".join(words) or "prompt"


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

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
                "model":        ("MODEL",),
                "clip":         ("CLIP",),
                "vae":          ("VAE",),
                "image":        ("IMAGE",),
                "multi_prompt": ("STRING", {"forceInput": True}),
                "seed":         ("INT",   {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "steps":        ("INT",   {"default": 8,   "min": 1,   "max": 100}),
                "cfg":          ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "sampler_name": (samplers,),
            },
            "optional": {
                "separator":                ("STRING",  {"default": "\\n", "multiline": False}),
                "reference_latents_method": (
                    ["workflow_default", "index_timestep_zero", "offset", "index", "uxo/uno"],
                    {"default": "workflow_default"},
                ),
                "target_megapixels":        ("FLOAT",   {"default": 1.0, "min": 0.1, "max": 16.0, "step": 0.1}),
                "upscale_method":           (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {"default": "nearest-exact"}),
                "negative_prompt":          ("STRING",  {"default": "", "multiline": True}),
                "output_prefix":            ("STRING",  {"default": "flux_klein_multi"}),
                "save_images":              ("BOOLEAN", {"default": False}),
                "resolution_steps":         ("INT",     {"default": 1, "min": 1, "max": 256, "step": 1}),
                "output_width":             ("INT",     {"default": 720,  "min": 16, "max": 8192, "step": 16}),
                "output_height":            ("INT",     {"default": 1024, "min": 16, "max": 8192, "step": 16}),
                "debug_enabled":            ("BOOLEAN", {"default": False}),
                "debug_prefix":             ("STRING",  {"default": "flux_klein_debug"}),
                "defer_cpu_transfer":       ("BOOLEAN", {"default": False}),
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
        reference_latents_method="workflow_default",
        target_megapixels=1.0,
        upscale_method="nearest-exact",
        negative_prompt="",
        output_prefix="flux_klein_multi",
        save_images=False,
        resolution_steps=1,
        output_width=720,
        output_height=1024,
        debug_enabled=False,
        debug_prefix="flux_klein_debug",
        defer_cpu_transfer=False,
    ):
        sep = separator.replace("\\n", "\n")
        prompts = [p.strip() for p in (multi_prompt or "").split(sep) if p.strip()]

        if not prompts:
            blank = torch.zeros((1, image.shape[1], image.shape[2], 3), dtype=torch.float32)
            return (blank, 0)

        # -- debug setup --
        dbg = None
        if debug_enabled:
            out_dir = folder_paths.get_output_directory()
            dbg = _unique_path(out_dir, debug_prefix, "jsonl")
            _write_debug(dbg, {
                "event": "run_start",
                "prompt_count": len(prompts),
                "seed": int(seed), "steps": int(steps), "cfg": float(cfg),
                "sampler_name": str(sampler_name),
                "reference_latents_method": str(reference_latents_method),
                "target_megapixels": float(target_megapixels),
                "upscale_method": str(upscale_method),
                "resolution_steps": int(resolution_steps),
                "output_width": int(output_width),
                "output_height": int(output_height),
                "input_image_shape": _shape(image),
            })
            print(f"[IAMCCS_FluxKleinMultiGen] debug -> {dbg}")

        # ── STEP 1  ImageScaleToTotalPixels ──────────────────────────────────
        # Mirrors node: image -> ImageScaleToTotalPixels -> scaled
        scaled = _node_ImageScaleToTotalPixels(
            image, upscale_method, target_megapixels, resolution_steps
        )
        if dbg:
            _write_debug(dbg, {"event": "scaled_image", "shape": _shape(scaled)})

        # ── STEP 2  VAEEncode ─────────────────────────────────────────────────
        # Mirrors node: scaled + vae -> VAEEncode -> reference_latent
        # ONE encode shared between BOTH positive and negative ReferenceLatent nodes
        reference_latent = _node_VAEEncode(vae, scaled)
        if dbg:
            s = reference_latent.get("samples") if isinstance(reference_latent, dict) else reference_latent
            _write_debug(dbg, {"event": "reference_latent", "samples_shape": _shape(s)})

        # ── STEP 3  Negative conditioning ─────────────────────────────────────
        # Mirrors: clip + neg_text -> CLIPTextEncode -> neg_cond
        #          neg_cond + reference_latent -> ReferenceLatent -> neg_with_ref
        neg_cond     = _node_CLIPTextEncode(clip, negative_prompt)
        neg_with_ref = _node_ReferenceLatent(neg_cond, reference_latent)

        # Optional method override (workflow_default = no override = matches reference pipeline)
        _method = None
        if reference_latents_method and reference_latents_method != "workflow_default":
            _method = "uxo" if ("uxo" in reference_latents_method or "uso" in reference_latents_method) else reference_latents_method
            neg_with_ref = node_helpers.conditioning_set_values(
                neg_with_ref, {"reference_latents_method": _method}
            )

        # ── STEP 4  Sampler components (constant across all prompts) ──────────
        # Mirrors: sampler_name -> KSamplerSelect -> sampler
        #          steps + output_width + output_height -> Flux2Scheduler -> sigmas
        #          output_width + output_height -> EmptyFlux2LatentImage -> latent_image
        sampler_obj = _node_KSamplerSelect(sampler_name)
        sigmas      = _node_Flux2Scheduler(steps, output_width, output_height)
        latent_img  = _node_EmptyFlux2LatentImage(output_width, output_height)

        # ── GENERATION LOOP ────────────────────────────────────────────────────
        results  = []
        save_dir = folder_paths.get_output_directory() if save_images else None

        for idx, prompt in enumerate(prompts):
            current_seed = int(seed) + idx

            # Mirrors: clip + prompt -> CLIPTextEncode -> pos_cond
            #          pos_cond + reference_latent -> ReferenceLatent -> pos_with_ref
            pos_cond     = _node_CLIPTextEncode(clip, prompt)
            pos_with_ref = _node_ReferenceLatent(pos_cond, reference_latent)

            if _method is not None:
                pos_with_ref = node_helpers.conditioning_set_values(
                    pos_with_ref, {"reference_latents_method": _method}
                )

            # Mirrors: seed -> RandomNoise -> noise
            #          model + pos_with_ref + neg_with_ref + cfg -> CFGGuider -> guider
            #          SamplerCustomAdvanced(noise, guider, sampler, sigmas, latent_image)
            noise         = _node_RandomNoise(current_seed)
            guider        = _node_CFGGuider(model, pos_with_ref, neg_with_ref, cfg)
            output_latent = _node_SamplerCustomAdvanced(
                noise, guider, sampler_obj, sigmas, latent_img
            )

            # Mirrors: output_latent + vae -> VAEDecode -> image_out
            decoded = _node_VAEDecode(vae, output_latent)
            if isinstance(decoded, (tuple, list)):
                decoded = decoded[0]
            if decoded.ndim == 3:
                decoded = decoded.unsqueeze(0)

            # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
            # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
            # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
            # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
            # By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
            if defer_cpu_transfer:
                stored_decoded = decoded
            else:
                stored_decoded = decoded.cpu()

            if dbg:
                _write_debug(dbg, {
                    "event": "iteration_result",
                    "idx": idx, "seed": current_seed,
                    "prompt": prompt,
                    "decoded_shape": _shape(stored_decoded),
                })

            results.append(stored_decoded)

            if save_images and save_dir:
                slug = _prompt_slug(prompt)
                path = _unique_path(save_dir, f"{output_prefix}_{slug}", "png")
                save_decoded = stored_decoded if not defer_cpu_transfer else decoded.detach().cpu()
                arr  = (save_decoded[0].numpy() * 255).clip(0, 255).astype(np.uint8)
                _PILImage.fromarray(arr).save(path)

        if dbg:
            _write_debug(dbg, {"event": "run_end", "count": len(results)})

        batched = torch.cat(results, dim=0)
        if defer_cpu_transfer:
            batched = batched.cpu()
        return (batched, len(prompts))
