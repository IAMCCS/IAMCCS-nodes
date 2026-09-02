"""Shared helpers for MMH3Tools.

Grid constants and payload shapes mirror ComfyUI v0.30.0:
  comfy_extras/nodes_minimax_h3.py   (node-side conditioning construction)
  comfy/ldm/minimax/model.py         (PackedLayout, which consumes it)

H3 AV latents are a NestedTensor pair with DIFFERENT temporal axes:
    video  [B, 24, T,  h, w]   -> temporal dim 2
    audio  [B, 32, 2,  T40]    -> temporal dim 3  (dim 2 is the stereo axis)

Anything that slices or concatenates these must use per-sub-tensor dims. Code
that assumes a single temporal dim for both will silently stack audio on the
stereo axis instead of extending its duration.
"""

import math

import torch
import torch.nn.functional as F

from comfy.nested_tensor import NestedTensor

FPS = 24
AUDIO_LATENT_FPS = 40
FRAMES_PER_GROUP = 17
FRAME_BASE = 5
LATENTS_PER_GROUP = 5
LATENT_BASE = 2

VIDEO_T_DIM = 2
AUDIO_T_DIM = 3


# --------------------------------------------------------------------------
# grid math
# --------------------------------------------------------------------------

def snap_latents(n):
    """Snap DOWN onto the model's 5j+2 video-latent grid (minimum 2)."""
    if n < LATENT_BASE:
        return LATENT_BASE
    return LATENTS_PER_GROUP * ((n - LATENT_BASE) // LATENTS_PER_GROUP) + LATENT_BASE


def snap_frames(n):
    """Snap UP onto the model's 17j+5 frame grid, matching align_frame_count()."""
    n = max(FRAME_BASE, int(n))
    while n % FRAMES_PER_GROUP != FRAME_BASE:
        n += 1
    return n


def latents_to_frames(latent_t):
    return FRAMES_PER_GROUP * ((latent_t - LATENT_BASE) // LATENTS_PER_GROUP) + FRAME_BASE


def frames_to_latents(frame_count):
    """Mirror of video_latent_t() in the stock nodes."""
    if frame_count <= FRAME_BASE:
        return LATENT_BASE
    return ((frame_count - FRAME_BASE) // FRAMES_PER_GROUP) * LATENTS_PER_GROUP + LATENT_BASE


def frames_to_audio_t(frame_count):
    return int(round(frame_count / FPS * AUDIO_LATENT_FPS))


def on_grid(latent_t):
    return snap_latents(latent_t) == latent_t


# Frames covered by each latent step, indexed by step % 5. Mirrors FRAME_PER_TOKEN in
# comfy/ldm/minimax/model.py: every fifth step spans ONE frame and the rest span four.
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)


def frame_at_latent(k):
    """First pixel frame of latent step `k`, for ANY k.

    latents_to_frames() is the inverse of the 5j+2 grid and is only meaningful ON that
    grid; asking it about an arbitrary index gives nonsense (index 1 comes back as -12).
    Window bounds are arbitrary indices, so reporting the frames a window covers needs
    this instead. The two agree wherever both are valid: frame_at_latent(37) == 124 ==
    latents_to_frames(37).
    """
    k = int(k)
    if k <= 0:
        return 0
    full, rem = divmod(k, len(FRAME_PER_TOKEN))
    return full * sum(FRAME_PER_TOKEN) + sum(FRAME_PER_TOKEN[:rem])



# --------------------------------------------------------------------------
# latent plumbing
# --------------------------------------------------------------------------

def unpack_av(latent, name="latent", allow_video_only=False):
    """Return (video, audio) from an H3 AV latent dict.

    With allow_video_only, a PLAIN 5D video latent [B,24,T,h,w] is accepted and
    audio comes back None. That is what VAEEncode produces when you encode real
    footage with the H3 video VAE -- it has no audio component.
    """
    samples = latent["samples"]
    if isinstance(samples, NestedTensor):
        parts = samples.unbind()
        if len(parts) != 2:
            raise ValueError("'%s': expected 2 nested tensors (video, audio), got %d"
                             % (name, len(parts)))
        return parts[0], parts[1]

    if allow_video_only and hasattr(samples, "ndim") and samples.ndim == 5:
        return samples, None

    raise ValueError(
        "'%s' is a plain latent, not a MiniMax H3 AV latent (NestedTensor video+audio "
        "pair). Wire it to an H3 sampler output or Empty MiniMax H3 AV Latent. Note that "
        "VAEEncode with the H3 video VAE also produces a plain, audio-less latent."
        % name
    )


def pack_av(latent, video, audio, noise_mask=None):
    """Rebuild a latent dict around new video/audio tensors."""
    out = dict(latent)
    out["samples"] = NestedTensor([video, audio])
    if noise_mask is not None:
        out["noise_mask"] = noise_mask
    return out


def slice_av_tail(video, audio, latent_t):
    """Take the last `latent_t` video latents plus the matching audio span."""
    latent_t = min(latent_t, video.shape[VIDEO_T_DIM])
    v = video[:, :, -latent_t:, :, :]
    frames = latents_to_frames(latent_t)
    audio_t = min(frames_to_audio_t(frames), audio.shape[AUDIO_T_DIM])
    a = audio[:, :, :, -audio_t:] if audio_t > 0 else None
    return v.contiguous(), (a.contiguous() if a is not None else None), frames, audio_t


CANVAS_MULTIPLE = 32
VAE_SPATIAL = 16
PATCH = 2
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344


def supported_downscale_factors(latent_h, latent_w):
    """Factors that keep BOTH latent dims integral AND even (required by the 2x2 patch).

    latent/f must be an even integer, i.e. f must divide latent//2 on both axes,
    so the valid set is exactly the divisors of gcd(latent_h//2, latent_w//2).
    For the 1344x768 canvas (latent 84x48) that is [1, 2, 3, 6] -- note 4 is NOT
    valid, because 84/4 = 21 is odd.
    """
    g = math.gcd(max(1, latent_h // PATCH), max(1, latent_w // PATCH))
    return [f for f in range(1, g + 1) if g % f == 0]


def snap_downscale(requested, latent_h, latent_w):
    """Snap a requested factor to the nearest supported one, tie-breaking gentler."""
    valid = supported_downscale_factors(latent_h, latent_w)
    return min(valid, key=lambda f: (abs(f - requested), f))


def downscale_video_latent(v, factor):
    """Bilinear spatial downscale, snapped to a factor that keeps the patch grid valid.

    Returns (tensor, latent_h, latent_w, factor_used).
    """
    h, w = int(v.shape[3]), int(v.shape[4])
    factor = snap_downscale(max(1, int(factor)), h, w)
    if factor <= 1:
        return v.contiguous(), h, w, 1
    b, c, t = v.shape[0], v.shape[1], v.shape[2]
    nh, nw = h // factor, w // factor
    x = v.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).float()
    x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
    x = x.reshape(b, t, c, nh, nw).permute(0, 2, 1, 3, 4).contiguous().to(v.dtype)
    return x, nh, nw, factor


# --------------------------------------------------------------------------
# conditioning
# --------------------------------------------------------------------------

def append_cond_list(conditioning, key, items):
    """Append to a list-valued conditioning key without clobbering existing entries."""
    out = []
    for entry in conditioning:
        d = entry[1].copy()
        d[key] = list(d.get(key) or []) + list(items)
        out.append([entry[0], d] + list(entry[2:]))
    return out


def evict_text_encoder(clip, tag):
    """Drop a CLIP's weights from VRAM once the last prompt has been encoded.

    H3's text encoder is large enough that leaving it resident denies the diffusion
    model the room it needs, at which point sampling falls back to system RAM and
    effectively hangs. A node that encodes every prompt a run will use is the last
    thing that needs it, so it can hand the VRAM back.

    `unload_model_and_clones`, NOT `unload_all_models`: this evicts the text encoder
    alone and leaves the VAEs where they are.
    """
    import logging

    import comfy.model_management as _mm

    patcher = getattr(clip, "patcher", None)
    if patcher is None:
        logging.warning("[%s] unload_text_encoder is on but this CLIP exposes no "
                        ".patcher; nothing evicted", tag)
        return False
    _mm.unload_model_and_clones(patcher)
    _mm.soft_empty_cache()
    logging.info("[%s] text encoder evicted from VRAM", tag)
    return True


def set_cond_values(conditioning, values):
    out = []
    for entry in conditioning:
        d = entry[1].copy()
        d.update(values)
        out.append([entry[0], d] + list(entry[2:]))
    return out


def empty_av_latent(width, height, length, batch_size=1, device=None):
    """Mirror of _empty_av_latent() in the stock nodes."""
    import comfy.model_management

    if device is None:
        device = comfy.model_management.intermediate_device()
    frame_count = snap_frames(max(FRAME_BASE, length))
    latent_t = frames_to_latents(frame_count)
    audio_t = frames_to_audio_t(frame_count)
    video = torch.zeros([batch_size, 24, latent_t, height // 16, width // 16], device=device)
    audio = torch.zeros([batch_size, 32, 2, audio_t], device=device)
    return {"samples": NestedTensor([video, audio])}, frame_count


def frames_to_qwen_items(frames):
    """Subsample decoded frames to 2fps with truthful timestamps, matching the stock node.

    The stock node derives timestamps from the sample INDEX assuming 2fps. Here they
    are derived from real frame positions, so the pair stays correct if the step ever
    changes.
    """
    n = frames.shape[0]
    step = max(1, FPS // 2)
    idx = list(range(0, n, step))
    return frames[idx], [(i * step) / FPS for i in range(len(idx))]


def make_ref_block(video, audio, latent_h, latent_w, audio_t):
    """Build one minimax_refs entry.

    video=None yields an AUDIO-ONLY block. That block carries no "latent" key, which
    model_base filters on, so it contributes audio conditioning without adding any
    video rows for the model to render back into the output.
    """
    has_audio = audio is not None and audio_t > 0
    if video is None:
        if not has_audio:
            raise ValueError("make_ref_block needs video, audio, or both")
        return {"kind": "audio", "ref_audio_t": int(audio_t), "audio_latent": audio}
    return {
        "kind": "video_audio" if has_audio else "video",
        "latent_t": int(video.shape[VIDEO_T_DIM]),
        "latent_h": int(latent_h),
        "latent_w": int(latent_w),
        "ref_audio_t": int(audio_t) if has_audio else 0,
        "latent": video,
        "audio_latent": audio if has_audio else None,
    }
