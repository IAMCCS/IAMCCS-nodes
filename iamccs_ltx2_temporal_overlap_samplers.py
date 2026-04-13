import copy
import importlib
import inspect
import logging
import sys

import torch

try:
    import comfy.model_management as _mm
except Exception:
    _mm = None

log = logging.getLogger("IAMCCS")

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


def _sanitize_aux_latent(latent: dict | None) -> dict | None:
    if not isinstance(latent, dict):
        return None
    samples = latent.get("samples")
    if not torch.is_tensor(samples) or samples.ndim != 5:
        return None
    out = _clone_latent_dict(latent)
    out.pop("iamccs_prev_latents", None)
    out.pop("iamccs_bridge_transition_applied", None)
    return out


def _strip_iamccs_keys(latent: dict) -> dict:
    latent.pop("iamccs_prev_latents", None)
    latent.pop("iamccs_reference_latents", None)
    latent.pop("iamccs_seed_offset", None)
    return latent


def _prepare_noise_for_segment(noise, seed_offset: int):
    if noise is None or not hasattr(noise, "seed"):
        return noise
    try:
        new_noise = copy.copy(noise)
        new_noise.seed = int(getattr(noise, "seed")) + int(seed_offset)
        return new_noise
    except Exception:
        return noise


def _cleanup_after_chunk(unload_all: bool = True, soft_empty_cache: bool = True):
    if _mm is None:
        return
    try:
        if unload_all:
            _mm.unload_all_models()
    except Exception:
        pass


def _coerce_int(value, default: int, label: str) -> int:
    try:
        return int(value)
    except Exception:
        log.warning(
            "[IAMCCS LTX2] Invalid integer for %s: %r. Falling back to %s.",
            label,
            value,
            default,
        )
        return int(default)
    try:
        if soft_empty_cache:
            _mm.soft_empty_cache()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _parse_cond_indices(optional_cond_indices, total_pixel_frames: int, cond_image_count: int | None = None) -> list[int] | None:
    if optional_cond_indices is None:
        return None
    if isinstance(optional_cond_indices, str):
        text = optional_cond_indices.strip()
        if not text or text == "0":
            return []
        if text.lower() in {"all", "*", "auto"}:
            frame_count = int(total_pixel_frames)
            if cond_image_count is not None:
                frame_count = min(frame_count, _coerce_int(cond_image_count, frame_count, "cond_image_count"))
            return list(range(max(0, frame_count)))
        raw = [part.strip() for part in text.split(",") if part.strip()]
        out = []
        for item in raw:
            value = int(item)
            if value < 0:
                value += total_pixel_frames
            out.append(value)
        return out
    try:
        return [int(x) for x in optional_cond_indices]
    except Exception:
        return None


def _build_auto_cond_indices(
    total_pixel_frames: int,
    cond_image_count: int | None,
    time_scale: int,
    guiding_latents_active: bool,
) -> tuple[list[int] | None, bool]:
    if cond_image_count is None or int(cond_image_count) != int(total_pixel_frames):
        return None, False

    total_pixel_frames = int(total_pixel_frames)
    if total_pixel_frames <= 0:
        return [], True

    if not guiding_latents_active:
        return list(range(total_pixel_frames)), True

    stride = max(8, int(time_scale))
    indices = list(range(0, total_pixel_frames, stride))
    last_index = total_pixel_frames - 1
    if indices[-1] != last_index and last_index % 8 != 1:
        indices.append(last_index)
    return indices, True


def _select_chunk_cond_images(
    optional_cond_images,
    absolute_indices,
    chunk_start_px: int,
    chunk_end_px_exclusive: int,
    guiding_latents_active: bool = False,
    direct_frame_lookup: bool = False,
):
    if optional_cond_images is None or absolute_indices is None:
        return None, None, []
    selected_images = []
    relative_indices = []
    skipped_indices = []
    if direct_frame_lookup:
        index_iter = ((int(abs_idx), optional_cond_images[int(abs_idx)]) for abs_idx in absolute_indices if 0 <= int(abs_idx) < int(optional_cond_images.shape[0]))
    else:
        index_iter = ((int(abs_idx), image) for image, abs_idx in zip(optional_cond_images, absolute_indices))

    for abs_idx, image in index_iter:
        abs_idx = int(abs_idx)
        if chunk_start_px <= abs_idx < chunk_end_px_exclusive:
            relative_idx = abs_idx - chunk_start_px
            if guiding_latents_active and relative_idx % 8 == 1:
                skipped_indices.append(relative_idx)
                continue
            selected_images.append(image.unsqueeze(0))
            relative_indices.append(relative_idx)
    if not selected_images:
        return None, None, skipped_indices
    return torch.cat(selected_images, dim=0), ",".join(str(idx) for idx in relative_indices), skipped_indices


def _latent_frame_to_pixel_index(latent_index: int, time_scale_factor: int) -> int:
    latent_index = max(0, int(latent_index))
    time_scale_factor = max(1, int(time_scale_factor))
    return latent_index * time_scale_factor


def _slice_spatial_latent(latent: dict | None, v_start: int, v_end: int, h_start: int, h_end: int) -> dict | None:
    if latent is None:
        return None
    out = _clone_latent_dict(latent)
    out["samples"] = out["samples"][:, :, :, v_start:v_end, h_start:h_end]
    if "noise_mask" in out and out["noise_mask"] is not None and torch.is_tensor(out["noise_mask"]):
        out["noise_mask"] = out["noise_mask"][:, :, :, v_start:v_end, h_start:h_end]
    return out


def _slice_spatial_images(images, v_start_px: int, v_end_px: int, h_start_px: int, h_end_px: int):
    if images is None:
        return None
    return images[:, v_start_px:v_end_px, h_start_px:h_end_px, :]


def _compute_spatial_bounds(total: int, tiles: int, overlap: int) -> list[tuple[int, int]]:
    total = int(total)
    tiles = max(1, int(tiles))
    overlap = max(0, int(overlap))
    base = (total + (tiles - 1) * overlap) // tiles
    bounds = []
    for index in range(tiles):
        start = index * max(1, (base - overlap))
        end = total if index == tiles - 1 else min(total, start + base)
        bounds.append((start, end))
    return bounds


def _make_spatial_tile_weight(tile_shape, v_idx: int, h_idx: int, vertical_tiles: int, horizontal_tiles: int, overlap: int, device, dtype):
    _, _, _, tile_h, tile_w = tile_shape
    weight = torch.ones((1, 1, 1, tile_h, tile_w), device=device, dtype=dtype)
    ramp = max(0, int(overlap))
    if ramp > 0:
        ramp_h = min(ramp, tile_h)
        ramp_w = min(ramp, tile_w)
        if h_idx > 0 and ramp_w > 0:
            alpha = torch.linspace(0.0, 1.0, steps=ramp_w, device=device, dtype=dtype)
            weight[:, :, :, :, :ramp_w] *= alpha.view(1, 1, 1, 1, -1)
        if h_idx < horizontal_tiles - 1 and ramp_w > 0:
            alpha = torch.linspace(1.0, 0.0, steps=ramp_w, device=device, dtype=dtype)
            weight[:, :, :, :, -ramp_w:] *= alpha.view(1, 1, 1, 1, -1)
        if v_idx > 0 and ramp_h > 0:
            alpha = torch.linspace(0.0, 1.0, steps=ramp_h, device=device, dtype=dtype)
            weight[:, :, :, :ramp_h, :] *= alpha.view(1, 1, 1, -1, 1)
        if v_idx < vertical_tiles - 1 and ramp_h > 0:
            alpha = torch.linspace(1.0, 0.0, steps=ramp_h, device=device, dtype=dtype)
            weight[:, :, :, -ramp_h:, :] *= alpha.view(1, 1, 1, -1, 1)
    return weight


def _stitch_spatial_latents(tile_entries, full_h: int, full_w: int, vertical_tiles: int, horizontal_tiles: int, overlap: int) -> dict:
    first_samples = tile_entries[0][0]["samples"]
    batch, channels, frames, _, _ = first_samples.shape
    out = torch.zeros((batch, channels, frames, full_h, full_w), device=first_samples.device, dtype=first_samples.dtype)
    weights = torch.zeros_like(out)
    for tile_latent, v_idx, h_idx, v_start, v_end, h_start, h_end in tile_entries:
        tile_samples = tile_latent["samples"]
        tile_weight = _make_spatial_tile_weight(
            tile_samples.shape,
            v_idx,
            h_idx,
            vertical_tiles,
            horizontal_tiles,
            overlap,
            tile_samples.device,
            tile_samples.dtype,
        )
        out[:, :, :, v_start:v_end, h_start:h_end] += tile_samples * tile_weight
        weights[:, :, :, v_start:v_end, h_start:h_end] += tile_weight
    out = out / weights.clamp_min(1e-8)
    return {"samples": out, "noise_mask": None}


def _apply_source_lock(latent: dict | None, strength: float) -> dict | None:
    if latent is None:
        return None
    strength = float(max(0.0, min(1.0, strength)))
    if strength <= 0.0:
        return latent
    locked = _ensure_noise_mask(latent, default_value=1.0)
    mask_val = 1.0 - strength
    locked["noise_mask"] = torch.minimum(
        locked["noise_mask"],
        torch.full_like(locked["noise_mask"], mask_val),
    )
    return locked


class IAMCCS_LTX2_OneShotLowRAMLooper:
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
                    {"default": 80, "min": 24, "max": 1000, "step": 8},
                ),
                "temporal_overlap": (
                    "INT",
                    {"default": 24, "min": 16, "max": 256, "step": 8},
                ),
                "guiding_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "temporal_overlap_cond_strength": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
            "optional": {
                "optional_guiding_latents": ("LATENT",),
                "optional_cond_images": ("IMAGE",),
                "optional_cond_image_indices": ("STRING", {"default": "0"}),
                "optional_positive_conditionings": ("CONDITIONING",),
                "optional_negative_index_latents": ("LATENT",),
                "optional_normalizing_latents": ("LATENT",),
                "cond_image_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "adain_factor": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "source_lock_strength": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "guiding_start_step": (
                    "INT",
                    {"default": 0, "min": 0, "max": 1000},
                ),
                "guiding_end_step": (
                    "INT",
                    {"default": 1000, "min": 0, "max": 1000},
                ),
                "unload_all_between_chunks": ("BOOLEAN", {"default": True}),
                "soft_empty_cache_between_chunks": ("BOOLEAN", {"default": True}),
                "cleanup_every_n_chunks": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "horizontal_tiles": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "vertical_tiles": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "spatial_overlap": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("denoised_output", "report")
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
        guiding_strength,
        temporal_overlap_cond_strength,
        optional_guiding_latents=None,
        optional_cond_images=None,
        optional_cond_image_indices="0",
        optional_positive_conditionings=None,
        optional_negative_index_latents=None,
        optional_normalizing_latents=None,
        cond_image_strength=1.0,
        adain_factor=1.0,
        source_lock_strength=0.0,
        guiding_start_step=0,
        guiding_end_step=1000,
        unload_all_between_chunks=True,
        soft_empty_cache_between_chunks=True,
        cleanup_every_n_chunks=1,
        horizontal_tiles=1,
        vertical_tiles=1,
        spatial_overlap=0,
    ):
        in_context_sampler_cls = _resolve_external_class("LTXVInContextSampler")
        extend_sampler_cls = _resolve_external_class("LTXVExtendSampler")
        if in_context_sampler_cls is None or extend_sampler_cls is None:
            raise RuntimeError("Required LTX samplers were not found. Ensure ComfyUI-LTXVideo is loaded.")

        init_latents = _clone_latent_dict(latents)
        total_f = int(init_latents["samples"].shape[2])
        time_scale = _get_time_scale_factor_from_vae(vae)
        temporal_tile_size = _coerce_int(temporal_tile_size, 80, "temporal_tile_size")
        temporal_overlap = _coerce_int(temporal_overlap, 24, "temporal_overlap")
        cleanup_every_n_chunks = _coerce_int(cleanup_every_n_chunks, 1, "cleanup_every_n_chunks")
        horizontal_tiles = _coerce_int(horizontal_tiles, 1, "horizontal_tiles")
        vertical_tiles = _coerce_int(vertical_tiles, 1, "vertical_tiles")
        spatial_overlap = _coerce_int(spatial_overlap, 0, "spatial_overlap")

        tile_f = max(1, _pixel_frames_to_latent_frames(temporal_tile_size, time_scale))
        overlap_f = max(1, _pixel_frames_to_latent_frames(temporal_overlap, time_scale))
        overlap_f = min(overlap_f, max(1, tile_f - 1), max(1, total_f - 1)) if total_f > 1 else 1
        step_f = max(1, tile_f - overlap_f)
        total_pixel_frames = max(1, (total_f - 1) * time_scale + 1)
        cond_image_count = None
        if optional_cond_images is not None and hasattr(optional_cond_images, "shape"):
            try:
                cond_image_count = int(optional_cond_images.shape[0])
            except Exception:
                cond_image_count = None
        direct_frame_lookup = False
        absolute_cond_indices = _parse_cond_indices(
            optional_cond_image_indices,
            total_pixel_frames,
            cond_image_count=cond_image_count,
        )
        if absolute_cond_indices == []:
            auto_indices, auto_direct_lookup = _build_auto_cond_indices(
                total_pixel_frames,
                cond_image_count,
                time_scale,
                guiding_latents_active=optional_guiding_latents is not None,
            )
            if auto_indices is not None:
                absolute_cond_indices = auto_indices
                direct_frame_lookup = auto_direct_lookup
        elif cond_image_count is not None and int(cond_image_count) == int(total_pixel_frames):
            direct_frame_lookup = True
        scale_tuple = getattr(vae, "downscale_index_formula", (8, 32, 32))
        width_scale = int(scale_tuple[1]) if len(scale_tuple) > 1 else 32
        height_scale = int(scale_tuple[2]) if len(scale_tuple) > 2 else 32
        horizontal_tiles = max(1, horizontal_tiles)
        vertical_tiles = max(1, vertical_tiles)
        spatial_overlap = max(0, spatial_overlap)
        cleanup_every_n_chunks = max(1, cleanup_every_n_chunks)
        spatial_tiling_requested = horizontal_tiles > 1 or vertical_tiles > 1

        if spatial_tiling_requested:
            log.warning(
                "[IAMCCS LTX2] Spatial sampling tiling requested (%sx%s @ %s), "
                "but tiled denoising changes scene semantics across tiles. Falling back to 1x1.",
                vertical_tiles,
                horizontal_tiles,
                spatial_overlap,
            )
            horizontal_tiles = 1
            vertical_tiles = 1
            spatial_overlap = 0

        output_latents = None
        normalization_reference = _sanitize_aux_latent(optional_normalizing_latents)
        chunk_reports = []
        chunk_index = 0
        start_f = 0

        while start_f < total_f:
            end_f = min(start_f + tile_f, total_f)
            latent_chunk = _slice_latent(init_latents, start_f, end_f)
            latent_chunk = _apply_source_lock(latent_chunk, source_lock_strength)
            guiding_chunk = (
                _slice_latent(optional_guiding_latents, start_f, end_f)
                if optional_guiding_latents is not None
                else None
            )

            chunk_start_px = _latent_frame_to_pixel_index(start_f, time_scale)
            chunk_end_px = _latent_frame_to_pixel_index(end_f - 1, time_scale) + 1
            cond_images_chunk, cond_indices_chunk, skipped_cond_indices = _select_chunk_cond_images(
                optional_cond_images,
                absolute_cond_indices,
                chunk_start_px,
                chunk_end_px,
                guiding_latents_active=guiding_chunk is not None,
                direct_frame_lookup=direct_frame_lookup,
            )

            guider_for_chunk = _prepare_chunk_guider(guider, optional_positive_conditionings)
            if horizontal_tiles == 1 and vertical_tiles == 1:
                chunk_noise = _prepare_noise_for_segment(noise, start_f)
                if output_latents is None:
                    if guiding_chunk is not None:
                        output_latents = in_context_sampler_cls().sample(
                            vae=vae,
                            guider=guider_for_chunk,
                            sampler=sampler,
                            sigmas=sigmas,
                            noise=chunk_noise,
                            guiding_latents=guiding_chunk,
                            optional_cond_images=cond_images_chunk,
                            optional_cond_indices=cond_indices_chunk,
                            num_frames=-1,
                            optional_initialization_latents=latent_chunk,
                            optional_negative_index_latents=optional_negative_index_latents,
                            cond_image_strength=cond_image_strength,
                            guiding_strength=guiding_strength,
                            guiding_start_step=guiding_start_step,
                            guiding_end_step=guiding_end_step,
                        )[0]
                    else:
                        base_sampler_cls = _resolve_external_class("LTXVBaseSampler")
                        if base_sampler_cls is None:
                            raise RuntimeError("LTXVBaseSampler not found.")
                        output_latents = base_sampler_cls().sample(
                            model=model,
                            vae=vae,
                            width=int(latent_chunk["samples"].shape[-1]) * width_scale,
                            height=int(latent_chunk["samples"].shape[-2]) * height_scale,
                            num_frames=(int(latent_chunk["samples"].shape[2]) - 1) * time_scale + 1,
                            guider=guider_for_chunk,
                            sampler=sampler,
                            sigmas=sigmas,
                            noise=chunk_noise,
                            optional_cond_images=cond_images_chunk,
                            optional_cond_indices=cond_indices_chunk,
                            optional_initialization_latents=latent_chunk,
                            optional_negative_index_latents=optional_negative_index_latents,
                            guiding_start_step=guiding_start_step,
                            guiding_end_step=guiding_end_step,
                        )[0]
                    skip_report = ""
                    if skipped_cond_indices:
                        skip_report = f" skip_cond={skipped_cond_indices}"
                    chunk_reports.append(f"chunk_{chunk_index+1}=in_context frames[{start_f}:{end_f}){skip_report}")
                else:
                    output_latents = extend_sampler_cls().sample(
                        model=model,
                        vae=vae,
                        latents=output_latents,
                        num_new_frames=max(1, ((int(latent_chunk["samples"].shape[2]) - overlap_f) * time_scale)),
                        frame_overlap=max(1, overlap_f * time_scale),
                        guider=guider_for_chunk,
                        sampler=sampler,
                        sigmas=sigmas,
                        noise=chunk_noise,
                        strength=temporal_overlap_cond_strength,
                        guiding_strength=guiding_strength,
                        cond_image_strength=cond_image_strength,
                        optional_guiding_latents=guiding_chunk,
                        optional_cond_images=cond_images_chunk,
                        optional_cond_indices=cond_indices_chunk,
                        optional_reference_latents=normalization_reference,
                        optional_initialization_latents=latent_chunk,
                        adain_factor=adain_factor,
                        optional_negative_index_latents=optional_negative_index_latents,
                        guiding_start_step=guiding_start_step,
                        guiding_end_step=guiding_end_step,
                        normalize_per_frame=normalization_reference is not None,
                    )[0]
                    skip_report = ""
                    if skipped_cond_indices:
                        skip_report = f" skip_cond={skipped_cond_indices}"
                    chunk_reports.append(f"chunk_{chunk_index+1}=extend frames[{start_f}:{end_f}){skip_report}")
            else:
                v_bounds = _compute_spatial_bounds(int(latent_chunk["samples"].shape[3]), vertical_tiles, spatial_overlap)
                h_bounds = _compute_spatial_bounds(int(latent_chunk["samples"].shape[4]), horizontal_tiles, spatial_overlap)
                tile_entries = []
                base_sampler_cls = _resolve_external_class("LTXVBaseSampler")
                if base_sampler_cls is None:
                    raise RuntimeError("LTXVBaseSampler not found.")
                for v_idx, (v_start, v_end) in enumerate(v_bounds):
                    for h_idx, (h_start, h_end) in enumerate(h_bounds):
                        tile_latent = _slice_spatial_latent(latent_chunk, v_start, v_end, h_start, h_end)
                        tile_guiding = _slice_spatial_latent(guiding_chunk, v_start, v_end, h_start, h_end)
                        tile_negative = _slice_spatial_latent(optional_negative_index_latents, v_start, v_end, h_start, h_end)
                        tile_prev_output = _slice_spatial_latent(output_latents, v_start, v_end, h_start, h_end) if output_latents is not None else None
                        tile_normalizing = _slice_spatial_latent(normalizing_chunk, v_start, v_end, h_start, h_end) if normalizing_chunk is not None else None
                        if tile_normalizing is None and first_chunk_output is not None:
                            tile_normalizing = _slice_spatial_latent(first_chunk_output, v_start, v_end, h_start, h_end)
                        tile_cond_images = None
                        if cond_images_chunk is not None:
                            tile_cond_images = _slice_spatial_images(
                                cond_images_chunk,
                                v_start * height_scale,
                                v_end * height_scale,
                                h_start * width_scale,
                                h_end * width_scale,
                            )
                        tile_noise = _prepare_noise_for_segment(noise, start_f + (v_idx * horizontal_tiles + h_idx))
                        if tile_prev_output is None:
                            if tile_guiding is not None:
                                tile_output = in_context_sampler_cls().sample(
                                    vae=vae,
                                    guider=guider_for_chunk,
                                    sampler=sampler,
                                    sigmas=sigmas,
                                    noise=tile_noise,
                                    guiding_latents=tile_guiding,
                                    optional_cond_images=tile_cond_images,
                                    optional_cond_indices=cond_indices_chunk,
                                    num_frames=-1,
                                    optional_initialization_latents=tile_latent,
                                    optional_negative_index_latents=tile_negative,
                                    cond_image_strength=cond_image_strength,
                                    guiding_strength=guiding_strength,
                                    guiding_start_step=guiding_start_step,
                                    guiding_end_step=guiding_end_step,
                                )[0]
                            else:
                                tile_output = base_sampler_cls().sample(
                                    model=model,
                                    vae=vae,
                                    width=int(tile_latent["samples"].shape[-1]) * width_scale,
                                    height=int(tile_latent["samples"].shape[-2]) * height_scale,
                                    num_frames=(int(tile_latent["samples"].shape[2]) - 1) * time_scale + 1,
                                    guider=guider_for_chunk,
                                    sampler=sampler,
                                    sigmas=sigmas,
                                    noise=tile_noise,
                                    optional_cond_images=tile_cond_images,
                                    optional_cond_indices=cond_indices_chunk,
                                    optional_initialization_latents=tile_latent,
                                    optional_negative_index_latents=tile_negative,
                                    guiding_start_step=guiding_start_step,
                                    guiding_end_step=guiding_end_step,
                                )[0]
                        else:
                            tile_output = extend_sampler_cls().sample(
                                model=model,
                                vae=vae,
                                latents=tile_prev_output,
                                num_new_frames=max(1, ((int(tile_latent["samples"].shape[2]) - overlap_f) * time_scale)),
                                frame_overlap=max(1, overlap_f * time_scale),
                                guider=guider_for_chunk,
                                sampler=sampler,
                                sigmas=sigmas,
                                noise=tile_noise,
                                strength=temporal_overlap_cond_strength,
                                guiding_strength=guiding_strength,
                                cond_image_strength=cond_image_strength,
                                optional_guiding_latents=tile_guiding,
                                optional_cond_images=tile_cond_images,
                                optional_cond_indices=cond_indices_chunk,
                                optional_reference_latents=tile_normalizing,
                                optional_initialization_latents=tile_latent,
                                adain_factor=adain_factor,
                                optional_negative_index_latents=tile_negative,
                                guiding_start_step=guiding_start_step,
                                guiding_end_step=guiding_end_step,
                                normalize_per_frame=optional_normalizing_latents is not None,
                            )[0]
                        tile_entries.append((tile_output, v_idx, h_idx, v_start, v_end, h_start, h_end))
                output_latents = _stitch_spatial_latents(
                    tile_entries,
                    int(latent_chunk["samples"].shape[3]),
                    int(latent_chunk["samples"].shape[4]),
                    vertical_tiles,
                    horizontal_tiles,
                    spatial_overlap,
                )
                if first_chunk_output is None:
                    first_chunk_output = _clone_latent_dict(output_latents)
                    chunk_reports.append(f"chunk_{chunk_index+1}=in_context_tiled frames[{start_f}:{end_f}) tiles={vertical_tiles}x{horizontal_tiles}")
                else:
                    chunk_reports.append(f"chunk_{chunk_index+1}=extend_tiled frames[{start_f}:{end_f}) tiles={vertical_tiles}x{horizontal_tiles}")

            chunk_index += 1
            if end_f >= total_f:
                break

            if chunk_index % cleanup_every_n_chunks == 0:
                _cleanup_after_chunk(
                    unload_all=bool(unload_all_between_chunks),
                    soft_empty_cache=bool(soft_empty_cache_between_chunks),
                )
            start_f += step_f

        report = (
            f"one_shot_low_ram chunks={chunk_index} latent_frames={total_f} "
            f"tile_lat={tile_f} overlap_lat={overlap_f} step_lat={step_f} spatial={vertical_tiles}x{horizontal_tiles}@{spatial_overlap} cleanup_n={cleanup_every_n_chunks}"
            f" spatial_tiling_requested={int(spatial_tiling_requested)} source_lock={float(source_lock_strength):.2f} | "
            + " ; ".join(chunk_reports)
        )
        log.info("[IAMCCS LTX2] %s", report)
        return (_strip_iamccs_keys(output_latents), report)


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


def _resolve_external_class(class_name: str):
    for module in list(sys.modules.values()):
        if module is None:
            continue
        mappings = getattr(module, "NODE_CLASS_MAPPINGS", None)
        if isinstance(mappings, dict):
            mapped = mappings.get(class_name)
            if inspect.isclass(mapped) and hasattr(mapped, "sample"):
                return mapped
        candidate = getattr(module, class_name, None)
        if inspect.isclass(candidate) and hasattr(candidate, "sample"):
            return candidate
    return None


def _prepare_chunk_guider(guider, optional_positive_conditionings):
    if optional_positive_conditionings is None:
        return guider

    try:
        positive = optional_positive_conditionings[0]
        negative = None
        if hasattr(guider, "raw_conds") and guider.raw_conds is not None:
            _, negative = guider.raw_conds
        elif hasattr(guider, "original_conds") and guider.original_conds is not None:
            _, negative = guider.original_conds
        if negative is None:
            return guider

        new_guider = copy.copy(guider)
        new_guider.set_conds(positive, negative)
        if hasattr(new_guider, "raw_conds"):
            new_guider.raw_conds = (positive, negative)
        return new_guider
    except Exception:
        return guider


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


def _infer_leading_overlap_frames(latent: dict) -> int:
    noise_mask = latent.get("noise_mask")
    if noise_mask is None:
        return 0

    if _is_nested(noise_mask):
        noise_mask, _ = _extract_video(noise_mask)

    if not torch.is_tensor(noise_mask) or noise_mask.ndim < 3:
        return 0

    leading = 0
    total_f = int(noise_mask.shape[2])
    for idx in range(total_f):
        frame_mask = noise_mask[:, :, idx]
        if torch.any(frame_mask < 0.999):
            leading += 1
            continue
        break
    return leading


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


class IAMCCS_LTX2_InitLatentSampler:
    """Single-pass latent sampler for segment detailer/upscale workflows.

    This node is intentionally simple: it denoises the provided initialization
    latents once using the upstream guider/sampler/sigmas, without temporal
    tiling or overlap extension. It exists to replace external looping samplers
    when IAMCCS queue segmentation is already handling the long-video split.
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
            },
            "optional": {
                "optional_cond_images": ("IMAGE",),
                "optional_guiding_latents": ("LATENT",),
                "optional_positive_conditionings": ("CONDITIONING",),
                "optional_negative_index_latents": ("LATENT",),
                "optional_normalizing_latents": ("LATENT",),
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
        optional_cond_images=None,
        optional_guiding_latents=None,
        optional_positive_conditionings=None,
        optional_negative_index_latents=None,
        optional_normalizing_latents=None,
    ):
        if SamplerCustomAdvanced is None:
            raise RuntimeError(
                "SamplerCustomAdvanced not available. Update ComfyUI / comfy_extras."
            )

        init_latents = _clone_latent_dict(latents)
        prev_segment_latents = _sanitize_aux_latent(
            init_latents.pop("iamccs_prev_latents", None)
        )
        reference_latents = _sanitize_aux_latent(optional_normalizing_latents)
        if reference_latents is None:
            reference_latents = _sanitize_aux_latent(
                init_latents.pop("iamccs_reference_latents", None)
            )
        seed_offset = int(init_latents.pop("iamccs_seed_offset", 0) or 0)
        segment_noise = _prepare_noise_for_segment(noise, seed_offset)

        overlap_f = _infer_leading_overlap_frames(init_latents)
        extend_sampler_cls = _resolve_external_class("LTXVExtendSampler")

        # For V2V/detailer use-cases, optional_guiding_latents usually carry the
        # source-video latents. Ignoring them turns the pass into a fresh
        # generation anchored only by noisy initialization, which is why faces and
        # body structure drift badly even though spatial resolution increases.
        if (
            optional_guiding_latents is not None
            and overlap_f > 0
            and extend_sampler_cls is not None
        ):
            time_scale = _get_time_scale_factor_from_vae(vae)
            extend_source_latents = prev_segment_latents
            if extend_source_latents is None:
                extend_source_latents = _slice_latent(init_latents, 0, overlap_f)
            if reference_latents is None:
                reference_latents = prev_segment_latents
            guider_for_sample = _prepare_chunk_guider(
                guider, optional_positive_conditionings
            )
            log.info(
                "[IAMCCS LTX2] Using original-style extend continuation overlap_lat=%s full_prev=%s seed_offset=%s ref=%s",
                overlap_f,
                prev_segment_latents is not None,
                seed_offset,
                reference_latents is not None,
            )
            denoised = extend_sampler_cls().sample(
                model=model,
                vae=vae,
                latents=extend_source_latents,
                num_new_frames=-1,
                frame_overlap=max(int(overlap_f * time_scale), 1),
                guider=guider_for_sample,
                sampler=sampler,
                sigmas=sigmas,
                noise=segment_noise,
                optional_guiding_latents=optional_guiding_latents,
                optional_cond_images=optional_cond_images,
                optional_initialization_latents=init_latents,
                optional_reference_latents=reference_latents,
                adain_factor=1.0 if reference_latents is not None else 0.0,
                optional_negative_index_latents=optional_negative_index_latents,
            )[0]
            target_frames = int(init_latents["samples"].shape[2])
            output_frames = int(denoised["samples"].shape[2])
            if output_frames > target_frames:
                denoised = _slice_latent(
                    denoised,
                    max(0, output_frames - target_frames),
                    output_frames,
                )
            denoised = _strip_iamccs_keys(denoised)
            denoised["iamccs_bridge_transition_applied"] = True
            return (denoised,)

        in_context_sampler_cls = _resolve_external_class("LTXVInContextSampler")
        if optional_guiding_latents is not None and in_context_sampler_cls is not None:
            guider_for_sample = _prepare_chunk_guider(
                guider, optional_positive_conditionings
            )
            denoised = in_context_sampler_cls().sample(
                vae=vae,
                guider=guider_for_sample,
                sampler=sampler,
                sigmas=sigmas,
                noise=segment_noise,
                guiding_latents=optional_guiding_latents,
                optional_cond_images=optional_cond_images,
                num_frames=-1,
                optional_initialization_latents=init_latents,
                optional_negative_index_latents=optional_negative_index_latents,
            )[0]
            denoised = _strip_iamccs_keys(denoised)
            return (denoised,)

        denoised = SamplerCustomAdvanced().sample(
            noise=segment_noise,
            guider=guider,
            sampler=sampler,
            sigmas=sigmas,
            latent_image=init_latents,
        )[1]
        denoised = _strip_iamccs_keys(denoised)
        return (denoised,)


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
