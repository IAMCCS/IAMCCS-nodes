from __future__ import annotations

import logging
from typing import Any, Tuple

import torch

import comfy.model_management as mm


_log = logging.getLogger("IAMCCS.SamplerAdvancedV1")


def _call_sampler_custom_advanced(
    noise: Any,
    guider: Any,
    sampler: Any,
    sigmas: Any,
    latent_image: Any,
    disable_progress: bool,
) -> Tuple[Any, Any]:
    """Call ComfyUI's SamplerCustomAdvanced in a version-tolerant way."""

    def _as_output_tuple(value: Any) -> Tuple[Any, ...]:
        v = value

        # Unwrap common wrappers like (x,) or ((x,),)
        while isinstance(v, tuple) and len(v) == 1:
            v = v[0]

        if isinstance(v, tuple):
            return v
        if isinstance(v, list):
            return tuple(v)

        # Newer ComfyUI API can wrap node returns as comfy_api.latest._io.NodeOutput
        # (can't reliably import the class here), so we use duck-typing.
        args = getattr(v, "args", None)
        if isinstance(args, (tuple, list)):
            return tuple(args)

        # Fallbacks seen in some wrappers
        value_attr = getattr(v, "value", None)
        if value_attr is not None and value_attr is not v:
            return _as_output_tuple(value_attr)

        raise TypeError(f"Unexpected SamplerCustomAdvanced output: {type(value)}")

    # ComfyUI exposes this in comfy_extras.
    from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced  # type: ignore

    node = SamplerCustomAdvanced()

    # Newer/older builds differ slightly in naming/signature.
    # We try the most informative signature first, then progressively fall back.
    for method_name in ("sample", "execute"):
        fn = getattr(node, method_name, None)
        if fn is None:
            continue

        try:
            out = fn(
                noise=noise,
                guider=guider,
                sampler=sampler,
                sigmas=sigmas,
                latent_image=latent_image,
                disable=bool(disable_progress),
            )
            break
        except TypeError:
            out = fn(
                noise=noise,
                guider=guider,
                sampler=sampler,
                sigmas=sigmas,
                latent_image=latent_image,
            )
            break
    else:
        raise RuntimeError("SamplerCustomAdvanced not available in this ComfyUI build")

    out_tuple = _as_output_tuple(out)
    if len(out_tuple) < 2:
        raise RuntimeError(f"SamplerCustomAdvanced returned too few outputs: {len(out_tuple)}")

    return out_tuple[0], out_tuple[1]


class IAMCCS_SamplerAdvancedVersion1:
    """Non-windowed sampler wrapper for video workflows.

    This node intentionally avoids any latent slicing/windowing. It delegates the actual
    diffusion loop to ComfyUI's SamplerCustomAdvanced (noise+guider+sampler+sigmas+latent).

    Added knobs:
    - disable_progress: reduce UI overhead for long video sampling.
    - cleanup: optional post-run VRAM cleanup for low-VRAM setups.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
                "disable_progress": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Disable the progress bar/step UI updates (can be slightly faster for long video denoises).",
                    },
                ),
                "cleanup": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "After sampling, run ComfyUI soft_empty_cache + torch.cuda.empty_cache (helps low VRAM at the cost of a bit of time).",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "IAMCCS/Sampling"

    def sample(
        self,
        noise,
        guider,
        sampler,
        sigmas,
        latent_image,
        disable_progress: bool,
        cleanup: bool,
    ):
        # Avoid surprises: do not mutate upstream graph objects.
        try:
            latent_in = latent_image.copy() if isinstance(latent_image, dict) else latent_image
        except Exception:
            latent_in = latent_image

        with torch.inference_mode():
            out_latent, denoised_latent = _call_sampler_custom_advanced(
                noise=noise,
                guider=guider,
                sampler=sampler,
                sigmas=sigmas,
                latent_image=latent_in,
                disable_progress=bool(disable_progress),
            )

        if cleanup:
            try:
                mm.soft_empty_cache()
            except Exception:
                pass
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        # Log minimal info (useful when comparing performance).
        try:
            s = out_latent.get("samples") if isinstance(out_latent, dict) else None
            if isinstance(s, torch.Tensor):
                _log.info(
                    "SamplerAdvancedV1 done: shape=%s dtype=%s device=%s",
                    tuple(s.shape),
                    str(s.dtype),
                    str(s.device),
                )
        except Exception:
            pass

        return (out_latent, denoised_latent)
