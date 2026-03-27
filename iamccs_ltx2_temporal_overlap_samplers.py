import copy
import importlib

import torch

SamplerCustomAdvanced = None
try:
    SamplerCustomAdvanced = importlib.import_module(
        "comfy_extras.nodes_custom_sampler"
    ).SamplerCustomAdvanced
except Exception:
    SamplerCustomAdvanced = None

# ---------------------------------------------------------------------------
# NestedTensor helpers — ComfyUI uses NestedTensor for AV (video+audio) latents.
# The outer latent dict's "samples" can be a NestedTensor where tensors[0] is
# the video latent and tensors[1] is the audio latent.
# Individual inner tensors may be 4-D (B,C,F,spatial_flat) rather than 5-D.
# We must extract the video tensor before doing temporal slicing.
# ---------------------------------------------------------------------------
_NestedTensor = None
try:
    from comfy.nested_tensor import NestedTensor as _NestedTensor  # noqa: F401
except Exception:
    pass


def _is_nested(t) -> bool:
    return _NestedTensor is not None and isinstance(t, _NestedTensor)


def _extract_video(samples):
    """Return (video_tensor, is_av).  For non-AV just returns as-is."""
    if _is_nested(samples):
        return samples.tensors[0], True
    return samples, False


def _rebuild_av(video_tensor, original_samples):
    """Put the modified video tensor back into the NestedTensor (if AV)."""
    if _is_nested(original_samples):
        new_tensors = list(original_samples.tensors)
        new_tensors[0] = video_tensor
        return _NestedTensor(new_tensors)
    return video_tensor


def _pixel_frames_to_latent_frames(pixel_frames: int, time_scale_factor: int) -> int:
    # LTX-style indexing: 1 + floor((N-1)/scale)
    if pixel_frames is None:
        return 0
    pixel_frames = int(pixel_frames)
    if pixel_frames <= 0:
        return 0
    time_scale_factor = max(int(time_scale_factor), 1)
    return 1 + max(0, (pixel_frames - 1) // time_scale_factor)


def _get_time_scale_factor_from_vae(vae) -> int:
    # ComfyUI LTX VAE exposes downscale_index_formula = (time, height, width)
    ts = getattr(vae, "downscale_index_formula", None)
    if ts and isinstance(ts, (tuple, list)) and len(ts) >= 1:
        try:
            return int(ts[0])
        except Exception:
            pass
    return 8


def _clone_latent_dict(latent: dict) -> dict:
    out = {}
    for k, v in latent.items():
        if torch.is_tensor(v) or _is_nested(v):
            out[k] = v
        else:
            out[k] = copy.deepcopy(v)
    return out


def _temporal_slice(t, start_f: int, end_f_exclusive: int):
    """Slice a tensor (4D or 5D) along the *time* axis (dim 2)."""
    if t is None:
        return None
    if _is_nested(t):
        vid, _ = _extract_video(t)
        sliced_vid = vid[:, :, start_f:end_f_exclusive]
        return _rebuild_av(sliced_vid, t)
    # Regular tensor — 4D or 5D, time is always dim 2
    return t[:, :, start_f:end_f_exclusive]


def _slice_latent(latent: dict, start_f: int, end_f_exclusive: int) -> dict:
    s = _clone_latent_dict(latent)
    s["samples"] = _temporal_slice(s["samples"], start_f, end_f_exclusive)
    if "noise_mask" in s and s["noise_mask"] is not None:
        s["noise_mask"] = _temporal_slice(s["noise_mask"], start_f, end_f_exclusive)
    return s


def _mask_shape_for(video_t) -> tuple:
    """Return the 1-valued shape for a noise-mask that broadcasts over video_t."""
    # video_t can be 4D (B,C,F,S) or 5D (B,C,F,H,W)
    b = video_t.shape[0]
    f = video_t.shape[2]
    if video_t.ndim == 5:
        return (b, 1, f, 1, 1)
    # 4D: (B,C,F,S) → mask shape (B,1,F,1)
    return (b, 1, f, 1)


def _ensure_noise_mask(latent: dict, default_value: float = 1.0) -> dict:
    """Add a noise_mask to the latent dict if absent. Works for 4D/5D/NestedTensor.
    Always guarantees noise_mask is a plain assignable tensor (never a NestedTensor)."""
    s = _clone_latent_dict(latent)
    # Use the video tensor for shape inspection
    vid, _ = _extract_video(s["samples"])
    if "noise_mask" not in s or s["noise_mask"] is None or _is_nested(s["noise_mask"]):
        # NestedTensor or absent: create a fresh plain float mask
        shape = _mask_shape_for(vid)
        s["noise_mask"] = torch.full(
            shape,
            float(default_value),
            device=vid.device,
            dtype=torch.float32,
        )
    return s


def _apply_adain(samples: torch.Tensor, reference: torch.Tensor, factor: float) -> torch.Tensor:
    # Simple AdaIN-like stat alignment per (batch, channel) across all dims.
    # samples/reference: B x C x F x H x W
    factor = float(factor)
    if factor <= 0.0:
        return samples

    eps = 1e-8
    # Compute mean/std across (F,H,W)
    s_mean = samples.mean(dim=(2, 3, 4), keepdim=True)
    s_std = samples.std(dim=(2, 3, 4), keepdim=True).clamp_min(eps)
    r_mean = reference.mean(dim=(2, 3, 4), keepdim=True)
    r_std = reference.std(dim=(2, 3, 4), keepdim=True).clamp_min(eps)

    normalized = (samples - s_mean) / s_std
    adapted = normalized * r_std + r_mean
    return torch.lerp(samples, adapted, factor)


def _blend_overlap(prev: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
    # Linear crossfade across overlap frames, keeping continuity.
    # prev/cur: B x C x F x H x W where F == overlap
    f = prev.shape[2]
    if f <= 1:
        return cur
    alpha = torch.linspace(0.0, 1.0, steps=f, device=prev.device, dtype=prev.dtype)
    alpha = alpha.view(1, 1, f, 1, 1)
    return prev * (1.0 - alpha) + cur * alpha


class IAMCCS_LTX2_LoopingSampler:
    """Temporal-tiling sampler with latent-overlap conditioning.

    Clean-room implementation: no code copied from other custom nodes.

    Goal: reduce motion discontinuities at segment boundaries by:
    - Sampling the video in temporal tiles
    - Conditioning the start of each tile on the previous tile's tail latents
    - Blending overlap region to avoid visible seams

    Notes:
    - `temporal_tile_size` and `temporal_overlap` are expressed in *pixel frames* (LTX convention).
    - Internally we convert them to latent-frame counts using `vae.downscale_index_formula[0]`.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "noise": ("NOISE",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "guider": ("GUIDER",),
                "latents": ("LATENT",),
                "temporal_tile_size": (
                    "INT",
                    {"default": 80, "min": 24, "max": 10000, "step": 8},
                ),
                "temporal_overlap": (
                    "INT",
                    {"default": 24, "min": 8, "max": 2000, "step": 8},
                ),
                "temporal_overlap_cond_strength": (
                    "FLOAT",
                    {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
            "optional": {
                "adain_factor": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("denoised_output",)
    FUNCTION = "sample"
    CATEGORY = "IAMCCS/LTX-2/sampling"

    def sample(
        self,
        model,
        vae,
        noise,
        sampler,
        sigmas,
        guider,
        latents,
        temporal_tile_size,
        temporal_overlap,
        temporal_overlap_cond_strength,
        adain_factor=0.0,
    ):
        if SamplerCustomAdvanced is None:
            raise RuntimeError(
                "SamplerCustomAdvanced not available. Update ComfyUI / comfy_extras."
            )

        time_scale = _get_time_scale_factor_from_vae(vae)
        tile_f = _pixel_frames_to_latent_frames(int(temporal_tile_size), time_scale)
        overlap_f = _pixel_frames_to_latent_frames(int(temporal_overlap), time_scale)

        samples = latents["samples"]
        total_f = int(samples.shape[2])

        if tile_f <= 0:
            tile_f = max(total_f, 1)
        overlap_f = max(min(overlap_f, tile_f), 1)

        chunk_f = min(tile_f + overlap_f, total_f)

        # First chunk
        chunk0 = _slice_latent(latents, 0, chunk_f)
        out0 = SamplerCustomAdvanced().sample(
            noise=noise,
            guider=guider,
            sampler=sampler,
            sigmas=sigmas,
            latent_image=chunk0,
        )[1]

        out_samples = out0["samples"]

        # Subsequent chunks
        start = tile_f
        while start < total_f:
            end = min(start + tile_f + overlap_f, total_f)
            chunk = _slice_latent(latents, start, end)

            # Condition overlap with previous output tail
            prev_overlap = out_samples[:, :, -overlap_f:, :, :]
            chunk = _ensure_noise_mask(chunk, default_value=1.0)
            chunk["samples"][:, :, :overlap_f, :, :] = prev_overlap
            chunk["noise_mask"][:, :, :overlap_f, :, :] = 1.0 - float(
                temporal_overlap_cond_strength
            )

            denoised = SamplerCustomAdvanced().sample(
                noise=noise,
                guider=guider,
                sampler=sampler,
                sigmas=sigmas,
                latent_image=chunk,
            )[1]

            chunk_out = denoised["samples"]

            # Optional per-tile statistic alignment for stability
            if float(adain_factor) > 0.0 and chunk_out.shape[2] > overlap_f:
                new_part = chunk_out[:, :, overlap_f:, :, :]
                ref = prev_overlap
                # Use the reference overlap replicated to match frames for stats stability
                ref = ref.expand(-1, -1, new_part.shape[2], -1, -1)
                chunk_out[:, :, overlap_f:, :, :] = _apply_adain(
                    new_part, ref, float(adain_factor)
                )

            # Blend overlap into existing output tail
            blended = _blend_overlap(prev_overlap, chunk_out[:, :, :overlap_f, :, :])
            out_samples[:, :, -overlap_f:, :, :] = blended

            # Append non-overlap part
            out_samples = torch.cat(
                [out_samples, chunk_out[:, :, overlap_f:, :, :]], dim=2
            )

            start += tile_f

        return ({"samples": out_samples, "noise_mask": None},)


class IAMCCS_LTX2_ExtendSampler:
    """Extend an existing latent video with overlap conditioning.

    Clean-room implementation: no code copied from other custom nodes.

    This is the simpler building block for multi-segment workflows:
    - Take last `temporal_overlap` frames from the existing latent
    - Sample a new chunk that begins with those frames partially locked
    - Blend overlap and append the new frames
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "noise": ("NOISE",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "guider": ("GUIDER",),
                "latents": ("LATENT",),
                "num_new_frames": (
                    "INT",
                    {"default": 80, "min": 1, "max": 10000, "step": 1},
                ),
                "temporal_overlap": (
                    "INT",
                    {"default": 24, "min": 8, "max": 2000, "step": 8},
                ),
                "temporal_overlap_cond_strength": (
                    "FLOAT",
                    {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
            "optional": {
                "adain_factor": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("extended_latents",)
    FUNCTION = "extend"
    CATEGORY = "IAMCCS/LTX-2/sampling"

    def extend(
        self,
        model,
        vae,
        noise,
        sampler,
        sigmas,
        guider,
        latents,
        num_new_frames,
        temporal_overlap,
        temporal_overlap_cond_strength,
        adain_factor=0.0,
    ):
        if SamplerCustomAdvanced is None:
            raise RuntimeError(
                "SamplerCustomAdvanced not available. Update ComfyUI / comfy_extras."
            )

        time_scale = _get_time_scale_factor_from_vae(vae)
        overlap_f = _pixel_frames_to_latent_frames(int(temporal_overlap), time_scale)
        overlap_f = max(overlap_f, 1)

        existing = latents["samples"]
        b, c, f, h, w = existing.shape

        prev_overlap = existing[:, :, -overlap_f:, :, :]

        # Convert requested *pixel* frames to latent frames for the new chunk.
        new_f = _pixel_frames_to_latent_frames(int(num_new_frames), time_scale)
        new_f = max(new_f, 1)

        chunk_samples = torch.zeros(
            (b, c, overlap_f + new_f, h, w), device=existing.device, dtype=existing.dtype
        )
        chunk_samples[:, :, :overlap_f, :, :] = prev_overlap
        chunk = {"samples": chunk_samples}
        chunk = _ensure_noise_mask(chunk, default_value=1.0)
        chunk["noise_mask"][:, :, :overlap_f, :, :] = 1.0 - float(
            temporal_overlap_cond_strength
        )

        denoised = SamplerCustomAdvanced().sample(
            noise=noise,
            guider=guider,
            sampler=sampler,
            sigmas=sigmas,
            latent_image=chunk,
        )[1]

        chunk_out = denoised["samples"]

        if float(adain_factor) > 0.0 and chunk_out.shape[2] > overlap_f:
            new_part = chunk_out[:, :, overlap_f:, :, :]
            ref = prev_overlap.expand(-1, -1, new_part.shape[2], -1, -1)
            chunk_out[:, :, overlap_f:, :, :] = _apply_adain(
                new_part, ref, float(adain_factor)
            )

        blended = _blend_overlap(prev_overlap, chunk_out[:, :, :overlap_f, :, :])
        extended = torch.cat(
            [existing[:, :, : f - overlap_f, :, :], blended, chunk_out[:, :, overlap_f:, :, :]],
            dim=2,
        )

        return ({"samples": extended, "noise_mask": None},)


class IAMCCS_LTX2_ConditionNextLatentWithPrevOverlap:
    """Condition the *start* of a "next" latent with the *end* of a previous latent.

    This is designed for multi-segment graphs that already build a per-segment init latent
    (e.g. via First/Last frame locking and/or context encoding) and then run a sampler.

    We don't run any sampling here — we only:
    - Copy the last `temporal_overlap` frames from `prev_latents` into the first frames of `next_latents`
    - Ensure `noise_mask` exists on `next_latents`
    - Partially lock those overlap frames via `noise_mask = 1 - temporal_overlap_cond_strength`

    Notes:
    - `temporal_overlap` is expressed in *pixel frames* (LTX convention). We convert using the VAE
      time scale when provided; otherwise we assume 8.
    - Works with both video-only and AV latents as long as they use the standard
      `{"samples": BxCxFxHxW}` structure.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prev_latents": ("LATENT",),
                "next_latents": ("LATENT",),
                "temporal_overlap": (
                    "INT",
                    {"default": 24, "min": 1, "max": 2000, "step": 1},
                ),
                "temporal_overlap_cond_strength": (
                    "FLOAT",
                    {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
            "optional": {
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("conditioned_latents",)
    FUNCTION = "condition"
    CATEGORY = "IAMCCS/LTX-2/latents"

    def condition(
        self,
        prev_latents,
        next_latents,
        temporal_overlap,
        temporal_overlap_cond_strength,
        vae=None,
    ):
        if not isinstance(prev_latents, dict) or "samples" not in prev_latents:
            raise ValueError("prev_latents must be a LATENT dict with a 'samples' tensor")
        if not isinstance(next_latents, dict) or "samples" not in next_latents:
            raise ValueError("next_latents must be a LATENT dict with a 'samples' tensor")

        time_scale = _get_time_scale_factor_from_vae(vae) if vae is not None else 8
        overlap_f = _pixel_frames_to_latent_frames(int(temporal_overlap), time_scale)
        overlap_f = max(int(overlap_f), 1)

        # --- Extract the video tensor from AV NestedTensors if needed ---
        prev_vid, prev_is_av = _extract_video(prev_latents["samples"])
        next_vid, next_is_av = _extract_video(next_latents["samples"])

        # Validate basic shape compatibility on the VIDEO tensors
        if prev_vid.ndim < 3 or next_vid.ndim < 3:
            raise ValueError("Expected video tensors to have at least 3 dimensions (B,C,T,...)")
        if prev_vid.shape[0] != next_vid.shape[0] or prev_vid.shape[1] != next_vid.shape[1]:
            raise ValueError("prev/next latents must have matching batch and channel dimensions")
        if prev_vid.shape[3:] != next_vid.shape[3:]:
            raise ValueError("prev/next latents must have matching spatial latent dimensions")

        # Clamp overlap to available frames
        overlap_f = min(overlap_f, int(prev_vid.shape[2]), int(next_vid.shape[2]))

        # Prepare output latent dict
        out = _clone_latent_dict(next_latents)
        out = _ensure_noise_mask(out, default_value=1.0)

        # --- Copy tail of prev video into head of next video ---
        # Work on a plain copy of next_vid so we can modify it
        new_next_vid = next_vid.clone()
        prev_tail = prev_vid[:, :, -overlap_f:]   # works for both 4D and 5D
        new_next_vid[:, :, :overlap_f] = prev_tail

        # Rebuild samples (AV or plain)
        if next_is_av:
            out["samples"] = _rebuild_av(new_next_vid, next_latents["samples"])
        else:
            out["samples"] = new_next_vid

        # --- Set noise_mask for the overlap region ---
        # noise_mask is guaranteed plain tensor by _ensure_noise_mask.
        mask_val = 1.0 - float(temporal_overlap_cond_strength)
        nm = out["noise_mask"].clone()
        nm[:, :, :overlap_f] = mask_val
        out["noise_mask"] = nm

        return (out,)
