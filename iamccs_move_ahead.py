"""
IAMCCS MoveAhead – FreeLong spectral blending + RoPE motion scaling for Wan 2.2.

Based on FreeLong (NeurIPS 2024) — SpectralBlend Temporal Attention.
Original LongLook implementation by https://github.com/StevenGuo30/comfyUI-LongLook.
Adapted and improved for IAMCCS-nodes:
  - Extended latent-frame auto-detection (covers 5–161 video frames, step 4)
  - Auto window mode: local_window_frames=0 → auto-computes 40% of detected frames
  - blend_curve dropdown: flat (constant) or sigmoid (ramp across blocks)
  - Dynamic total_blocks: end block respects actual model depth via transformer_options
  - Consistent IAMCCS logging prefix and CATEGORY
  - Per-chunk normalisation fix (weight clamp before division)
  - IAMCCS node naming conventions

Nodes:
  IAMCCS_MoveAhead          – replaced WanFreeLong
  IAMCCS_MoveAheadEnforcer  – replaces WanFreeLongEnforcer (3-tier spectral lock)
  IAMCCS_MotionScale        – replaces WanMotionScale (temporal RoPE scale)
  IAMCCS_MotionScaleAdvanced – replaces WanMotionScaleAdvanced (+ theta)
"""

import logging
import math
import torch
from typing import Dict, Any, Optional

logger = logging.getLogger("IAMCCS.MoveAhead")


def _parse_video_frames(text: str) -> Optional[int]:
    if text is None:
        return None
    s = str(text).strip().lower()
    if not s or s in {"auto", "a", "0", "none"}:
        return None
    try:
        v = int(float(s))
    except Exception:
        return None
    if v <= 0:
        return None
    return v


def _vram_preset_settings(preset: str) -> dict:
    p = (preset or "normal").strip().lower()
    if p == "low":
        return {"chunk_cols": 65_536, "cpu_fallback": True}
    if p == "high":
        return {"chunk_cols": 524_288, "cpu_fallback": True}
    return {"chunk_cols": 262_144, "cpu_fallback": True}


# ---------------------------------------------------------------------------
# Candidate latent-frame detection table.
# Wan 2.2 latent-frames = ((video_frames - 1) // 4) + 1.
# 5 video frames  → 2 latent frames
# 9               → 3
# 13              → 4
# 17              → 5
# 21              → 6
# 25              → 7
# ...up to 161    → 41
# We test every (4k+1) step so we cover all common Wan video lengths.
# ---------------------------------------------------------------------------
_CANDIDATE_LATENT_FRAMES = [(((vf - 1) // 4) + 1) for vf in range(5, 165, 4)]
# Deduplicate while preserving order, largest first (prefer more windows)
_seen = set()
_CANDIDATES_SORTED_DESC: list[int] = []
for _c in sorted(_CANDIDATE_LATENT_FRAMES, reverse=True):
    if _c not in _seen and _c > 1:
        _seen.add(_c)
        _CANDIDATES_SORTED_DESC.append(_c)
del _seen, _CANDIDATE_LATENT_FRAMES, _c


# ============================================================================
# Helper – spectral blend (standard 2-tier: global low / local high)
# ============================================================================

def _spectral_blend(
    global_features: torch.Tensor,
    local_features: torch.Tensor,
    low_freq_ratio: float = 0.25,
    blend_strength: float = 1.0,
) -> torch.Tensor:
    """
    Blend global and local features using spectral decomposition.

    Low frequencies (global) → motion direction consistency.
    High frequencies (local) → spatial detail sharpness.

    Uses a cosine transition at the cutoff to avoid Gibbs ringing.
    """
    if blend_strength == 0.0:
        return local_features

    orig_dtype = global_features.dtype
    g_f = global_features.float()
    l_f = local_features.float()

    g_freq = torch.fft.rfft(g_f, dim=1)
    l_freq = torch.fft.rfft(l_f, dim=1)
    del g_f, l_f

    num_freq = g_freq.shape[1]
    cutoff = int(num_freq * low_freq_ratio)
    tw = max(1, min(num_freq // 4, cutoff // 2))  # transition width

    mask = torch.ones(num_freq, device=global_features.device, dtype=torch.float32)
    t_start = max(0, cutoff - tw // 2)
    t_end = min(num_freq, cutoff + tw // 2)
    if t_end > t_start:
        t_idx = torch.arange(t_start, t_end, device=global_features.device, dtype=torch.float32)
        mask[t_start:t_end] = 0.5 * (
            1 + torch.cos(torch.pi * (t_idx - t_start) / (t_end - t_start))
        )
        del t_idx
    mask[t_end:] = 0.0
    mask = mask.view(1, -1, *([1] * (len(g_freq.shape) - 2)))

    blended = g_freq * mask + l_freq * (1.0 - mask)
    del g_freq, l_freq, mask

    result = torch.fft.irfft(blended, n=global_features.shape[1], dim=1)
    del blended

    if blend_strength < 1.0:
        result = local_features.float() * (1.0 - blend_strength) + result * blend_strength

    return result.to(orig_dtype)


def _is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if "out of memory" in msg or "allocation" in msg and "device" in msg:
        return True
    try:
        return isinstance(exc, torch.OutOfMemoryError)
    except Exception:
        return False


def _make_cosine_lowpass_mask(
    num_freq: int,
    low_freq_ratio: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    cutoff = int(num_freq * low_freq_ratio)
    tw = max(1, min(max(1, num_freq // 4), max(1, cutoff // 2)))

    mask = torch.ones(num_freq, device=device, dtype=dtype)
    t_start = max(0, cutoff - tw // 2)
    t_end = min(num_freq, cutoff + tw // 2)
    if t_end > t_start:
        t_idx = torch.arange(t_start, t_end, device=device, dtype=dtype)
        mask[t_start:t_end] = 0.5 * (
            1 + torch.cos(torch.pi * (t_idx - t_start) / (t_end - t_start))
        )
        del t_idx
    mask[t_end:] = 0.0
    return mask


def _spectral_blend_temporal_chunked_inplace(
    global_features: torch.Tensor,
    local_features: torch.Tensor,
    *,
    num_frames: int,
    spatial_size: int,
    low_freq_ratio: float,
    blend_strength: float,
    chunk_cols: int = 262_144,
    cpu_fallback: bool = True,
) -> torch.Tensor:
    """Chunked temporal FFT blend to reduce peak VRAM.

    Operates over the latent-frame axis (length=num_frames) and processes
    (spatial_size * hidden_dim) columns in chunks.

    Writes the blended result back into `local_features` (in-place) to avoid
    allocating an extra full-sized output tensor.
    """
    if blend_strength == 0.0:
        return local_features

    batch, seq_len, hidden_dim = global_features.shape
    if seq_len != num_frames * spatial_size:
        # Safety fallback (should not happen if temporal structure was detected)
        return _spectral_blend(global_features, local_features, low_freq_ratio, blend_strength)

    device = global_features.device
    orig_dtype = global_features.dtype

    def _is_pow2(n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0

    # Using float16 on CUDA keeps FFT outputs in complex32 (lower memory),
    # but cuFFT only supports fp16 RFFT for power-of-two signal lengths.
    use_fp16_fft = (device.type == "cuda") and _is_pow2(int(num_frames))
    fft_in_dtype = torch.float16 if use_fp16_fft else torch.float32
    cols = spatial_size * hidden_dim

    g_view = global_features.view(batch, num_frames, cols)
    out_view = local_features.view(batch, num_frames, cols)

    num_freq = num_frames // 2 + 1
    mask_real = _make_cosine_lowpass_mask(num_freq, low_freq_ratio, device, torch.float32)
    mask = mask_real.view(1, -1, 1)

    did_cpu_fallback = False

    for start in range(0, cols, chunk_cols):
        end = min(start + chunk_cols, cols)
        # Grab local slice before overwriting it
        l_src = out_view[:, :, start:end]
        g_src = g_view[:, :, start:end]

        try:
            g_in = g_src.to(dtype=fft_in_dtype)
            l_in = l_src.to(dtype=fft_in_dtype)

            g_freq = torch.fft.rfft(g_in, dim=1)
            l_freq = torch.fft.rfft(l_in, dim=1)

            blended_freq = g_freq * mask + l_freq * (1.0 - mask)
            del g_freq, l_freq, g_in, l_in

            blended = torch.fft.irfft(blended_freq, n=num_frames, dim=1)
            del blended_freq

            if blend_strength < 1.0:
                # Mix in float32 for numeric stability, then cast back
                blended = l_src.float() * (1.0 - blend_strength) + blended.float() * blend_strength

            out_view[:, :, start:end] = blended.to(dtype=orig_dtype)
            del blended

        except Exception as exc:
            if (not cpu_fallback) or (not _is_oom_error(exc)):
                raise
            if device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            did_cpu_fallback = True

            # CPU fallback for this chunk
            g_cpu = g_src.detach().float().cpu()
            l_cpu = l_src.detach().float().cpu()
            g_freq = torch.fft.rfft(g_cpu, dim=1)
            l_freq = torch.fft.rfft(l_cpu, dim=1)

            mask_cpu = mask_real.cpu().view(1, -1, 1)
            blended_freq = g_freq * mask_cpu + l_freq * (1.0 - mask_cpu)
            blended = torch.fft.irfft(blended_freq, n=num_frames, dim=1)

            if blend_strength < 1.0:
                blended = l_cpu * (1.0 - blend_strength) + blended * blend_strength

            out_view[:, :, start:end] = blended.to(device=device, dtype=orig_dtype)
            del g_cpu, l_cpu, g_freq, l_freq, blended_freq, blended, mask_cpu

    if did_cpu_fallback:
        logger.warning("[IAMCCS.MoveAhead] FFT blend hit OOM; chunk(s) retried on CPU (slower but safe).")

    return local_features


# ============================================================================
# Helper – enforced 3-tier spectral blend
# ============================================================================

def _enforced_spectral_blend(
    global_features: torch.Tensor,
    local_features: torch.Tensor,
    motion_lock_ratio: float = 0.15,
    low_freq_ratio: float = 0.5,
    blend_strength: float = 1.0,
) -> torch.Tensor:
    """
    3-tier spectral blend for stronger motion locking:
      - [0 .. motion_lock_ratio)     → 100 % global  (motion skeleton locked)
      - [motion_lock_ratio .. low_freq_ratio) → cosine blend global→local
      - [low_freq_ratio .. 1]        → 100 % local   (fine detail)
    """
    if blend_strength == 0.0:
        return local_features

    orig_dtype = global_features.dtype
    g_f = global_features.float()
    l_f = local_features.float()

    g_freq = torch.fft.rfft(g_f, dim=1)
    l_freq = torch.fft.rfft(l_f, dim=1)
    del g_f, l_f

    num_freq = g_freq.shape[1]
    lock_cut = int(num_freq * motion_lock_ratio)
    blend_cut = int(num_freq * low_freq_ratio)

    # global_mask: 1 = take global, 0 = take local
    global_mask = torch.zeros(num_freq, device=global_features.device, dtype=torch.float32)
    global_mask[:lock_cut] = 1.0  # Tier 1 – fully locked

    if blend_cut > lock_cut:  # Tier 2 – transition
        mid_len = blend_cut - lock_cut
        trans = torch.linspace(1.0, 0.0, mid_len + 2,
                               device=global_features.device, dtype=torch.float32)[1:-1]
        global_mask[lock_cut:blend_cut] = trans
        del trans
    # Tier 3 – already 0.0

    global_mask = global_mask.view(1, -1, *([1] * (len(g_freq.shape) - 2)))

    blended = g_freq * global_mask + l_freq * (1.0 - global_mask)
    del g_freq, l_freq, global_mask

    result = torch.fft.irfft(blended, n=global_features.shape[1], dim=1)
    del blended

    if blend_strength < 1.0:
        result = local_features.float() * (1.0 - blend_strength) + result * blend_strength

    return result.to(orig_dtype)


def _enforced_spectral_blend_temporal_chunked_inplace(
    global_features: torch.Tensor,
    local_features: torch.Tensor,
    *,
    num_frames: int,
    spatial_size: int,
    motion_lock_ratio: float,
    low_freq_ratio: float,
    blend_strength: float,
    chunk_cols: int = 262_144,
    cpu_fallback: bool = True,
) -> torch.Tensor:
    if blend_strength == 0.0:
        return local_features

    batch, seq_len, hidden_dim = global_features.shape
    if seq_len != num_frames * spatial_size:
        return _enforced_spectral_blend(
            global_features,
            local_features,
            motion_lock_ratio=motion_lock_ratio,
            low_freq_ratio=low_freq_ratio,
            blend_strength=blend_strength,
        )

    device = global_features.device
    orig_dtype = global_features.dtype

    def _is_pow2(n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0

    use_fp16_fft = (device.type == "cuda") and _is_pow2(int(num_frames))
    fft_in_dtype = torch.float16 if use_fp16_fft else torch.float32
    cols = spatial_size * hidden_dim

    g_view = global_features.view(batch, num_frames, cols)
    out_view = local_features.view(batch, num_frames, cols)

    num_freq = num_frames // 2 + 1
    lock_cut = int(num_freq * motion_lock_ratio)
    blend_cut = int(num_freq * low_freq_ratio)

    global_mask = torch.zeros(num_freq, device=device, dtype=torch.float32)
    global_mask[:lock_cut] = 1.0
    if blend_cut > lock_cut:
        mid_len = blend_cut - lock_cut
        trans = torch.linspace(1.0, 0.0, mid_len + 2, device=device, dtype=torch.float32)[1:-1]
        global_mask[lock_cut:blend_cut] = trans
        del trans
    global_mask = global_mask.view(1, -1, 1)

    did_cpu_fallback = False

    for start in range(0, cols, chunk_cols):
        end = min(start + chunk_cols, cols)
        l_src = out_view[:, :, start:end]
        g_src = g_view[:, :, start:end]

        try:
            g_in = g_src.to(dtype=fft_in_dtype)
            l_in = l_src.to(dtype=fft_in_dtype)

            g_freq = torch.fft.rfft(g_in, dim=1)
            l_freq = torch.fft.rfft(l_in, dim=1)

            blended_freq = g_freq * global_mask + l_freq * (1.0 - global_mask)
            del g_freq, l_freq, g_in, l_in

            blended = torch.fft.irfft(blended_freq, n=num_frames, dim=1)
            del blended_freq

            if blend_strength < 1.0:
                blended = l_src.float() * (1.0 - blend_strength) + blended.float() * blend_strength

            out_view[:, :, start:end] = blended.to(dtype=orig_dtype)
            del blended

        except Exception as exc:
            if (not cpu_fallback) or (not _is_oom_error(exc)):
                raise
            if device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            did_cpu_fallback = True

            g_cpu = g_src.detach().float().cpu()
            l_cpu = l_src.detach().float().cpu()
            g_freq = torch.fft.rfft(g_cpu, dim=1)
            l_freq = torch.fft.rfft(l_cpu, dim=1)

            gm_cpu = global_mask.cpu()
            blended_freq = g_freq * gm_cpu + l_freq * (1.0 - gm_cpu)
            blended = torch.fft.irfft(blended_freq, n=num_frames, dim=1)

            if blend_strength < 1.0:
                blended = l_cpu * (1.0 - blend_strength) + blended * blend_strength

            out_view[:, :, start:end] = blended.to(device=device, dtype=orig_dtype)
            del g_cpu, l_cpu, g_freq, l_freq, blended_freq, blended, gm_cpu

    if did_cpu_fallback:
        logger.warning("[IAMCCS.MoveAhead] Enforcer FFT blend hit OOM; chunk(s) retried on CPU (slower but safe).")

    return local_features


# ============================================================================
# Per-block effective strength (flat vs sigmoid curve)
# ============================================================================

def _compute_effective_strength(
    blend_strength: float,
    blend_curve: str,
    block_index: int,
    total_blocks: int,
) -> float:
    """
    Compute the effective blend strength for a given transformer block.

    flat    – constant blend_strength for every block (default, original behaviour).
    sigmoid – logistic ramp: strength starts near 0 for the first blocks and
              asymptotically reaches blend_strength for later blocks.  Early blocks
              establish the global motion skeleton; later blocks progressively add
              spectral detail-blending.

              The sigmoid is centred at ~30 %% through the stack so the transition
              is rapid and most of the stack runs at full blend_strength.
    """
    if blend_curve != "sigmoid" or blend_strength == 0.0:
        return blend_strength
    t = block_index / max(total_blocks - 1, 1)  # normalised 0‥1
    x = (t - 0.3) * 10.0                        # centred at 30 %%, sharp slope
    s = 1.0 / (1.0 + math.exp(-x))              # logistic 0 → 1
    return blend_strength * s


# ============================================================================
# Shared windowed-attention processor
# ============================================================================

def _run_windowed_attention(
    block_args: dict,
    original_block,
    x: torch.Tensor,
    settings: dict,
) -> Optional[torch.Tensor]:
    """
    Run the transformer block on overlapping temporal windows and accumulate.

    Returns:
        local_x  – flattened [batch, seq_len, dim] weighted-average output.
        None     – if temporal structure cannot be detected (caller must fallback).
    """
    batch_size, seq_len, hidden_dim = x.shape

    # ---- cache detection ----
    if settings.get("cached_seq_len") == seq_len:
        num_frames  = settings["cached_num_frames"]
        spatial_size = settings["cached_spatial_size"]
        freqs_seq_dim = settings.get("cached_freqs_seq_dim")
        overlap     = settings["cached_overlap"]
        stride      = settings["cached_stride"]
        window_size = settings["local_window_size"]  # already resolved (incl. auto)
    else:
        # --- 1. Detect latent frame count (largest divisor wins) ---
        num_frames = None
        for candidate in _CANDIDATES_SORTED_DESC:
            if seq_len % candidate == 0:
                num_frames = candidate
                break

        if num_frames is None:
            if settings.get("_log_fallback", True):
                settings["_log_fallback"] = False
                logger.warning("[IAMCCS.MoveAhead] Cannot detect temporal structure "
                               "(seq_len=%d). Falling back to global-only.", seq_len)
            return None

        # --- 2. Resolve auto window (0 = 40 %% of detected frames, min 3) ---
        if settings.get("local_window_size", 0) == 0:
            auto_win = max(3, round(num_frames * 0.4))
            settings["local_window_size"] = auto_win
            logger.info("[IAMCCS.MoveAhead] Auto window: %d latent-frames (40%% of %d detected)",
                        auto_win, num_frames)

        window_size = settings["local_window_size"]

        if num_frames <= window_size:
            if settings.get("_log_fallback", True):
                settings["_log_fallback"] = False
                logger.warning("[IAMCCS.MoveAhead] Window (%d) >= num_frames (%d). "
                               "Falling back to global-only.", window_size, num_frames)
            return None

        spatial_size = seq_len // num_frames
        overlap = window_size // 2
        stride  = window_size - overlap

        freqs = block_args.get("pe")
        freqs_seq_dim = None
        if freqs is not None:
            for dim_idx, dim_size in enumerate(freqs.shape):
                if dim_size == seq_len:
                    freqs_seq_dim = dim_idx
                    break

        settings["cached_seq_len"]      = seq_len
        settings["cached_num_frames"]   = num_frames
        settings["cached_spatial_size"] = spatial_size
        settings["cached_freqs_seq_dim"] = freqs_seq_dim
        settings["cached_overlap"]      = overlap
        settings["cached_stride"]       = stride

        if overlap > 0:
            ramp_up = torch.linspace(0, 1, overlap + 1, device=x.device, dtype=x.dtype)[1:]
            ramp_dn = torch.linspace(1, 0, overlap + 1, device=x.device, dtype=x.dtype)[:-1]
            settings["cached_ramp_up"] = ramp_up
            settings["cached_ramp_dn"] = ramp_dn

        if settings.get("_log_structure", True):
            settings["_log_structure"] = False
            logger.info("[IAMCCS.MoveAhead] Temporal structure: %d latent-frames × %d spatial tokens "
                        "(window=%d, overlap=%d, stride=%d)",
                        num_frames, spatial_size, window_size, overlap, stride)

    freqs = block_args.get("pe")
    cached_ramp_up = settings.get("cached_ramp_up")
    cached_ramp_dn = settings.get("cached_ramp_dn")

    x_temporal = x.view(batch_size, num_frames, spatial_size, hidden_dim)

    local_acc = torch.zeros(batch_size, num_frames, spatial_size, hidden_dim,
                            device=x.device, dtype=x.dtype)
    weight_acc = torch.zeros(batch_size, num_frames, 1, 1,
                             device=x.device, dtype=x.dtype)

    orig_img = block_args.get("img")
    orig_pe = block_args.get("pe")

    start_frame = 0
    while start_frame < num_frames:
        end_frame = min(start_frame + window_size, num_frames)
        wf = end_frame - start_frame

        win_x = x_temporal[:, start_frame:end_frame, :, :]
        win_flat = win_x.reshape(batch_size, wf * spatial_size, hidden_dim)
        block_args["img"] = win_flat

        if freqs is not None and freqs_seq_dim is not None:
            st = start_frame * spatial_size
            et = end_frame * spatial_size
            sl = [slice(None)] * len(freqs.shape)
            sl[freqs_seq_dim] = slice(st, et)
            block_args["pe"] = freqs[tuple(sl)]

        win_out = original_block(block_args)
        win_res = win_out["img"].view(batch_size, wf, spatial_size, hidden_dim)

        w = torch.ones(wf, device=x.device, dtype=x.dtype)
        if start_frame > 0 and overlap > 0 and cached_ramp_up is not None:
            rl = min(overlap, wf)
            w[:rl] = cached_ramp_up[:rl]
        if end_frame < num_frames and overlap > 0 and cached_ramp_dn is not None:
            rl = min(overlap, wf)
            w[-rl:] = w[-rl:] * cached_ramp_dn[:rl]

        w = w.view(1, wf, 1, 1)
        local_acc[:, start_frame:end_frame].add_(win_res * w)
        weight_acc[:, start_frame:end_frame].add_(w)

        del win_x, win_flat, win_out, win_res, w

        start_frame += stride
        if start_frame >= num_frames:
            break

    # Restore block_args
    block_args["img"] = orig_img
    if orig_pe is not None:
        block_args["pe"] = orig_pe

    weight_acc.clamp_(min=1e-8)
    local_acc.div_(weight_acc)
    del weight_acc

    local_x = local_acc.reshape(batch_size, seq_len, hidden_dim)
    del local_acc, x_temporal

    return local_x


# ============================================================================
# IAMCCS_MoveAhead  –  replaces WanFreeLong
# ============================================================================

class IAMCCS_MoveAhead:
    """
    IAMCCS MoveAhead – FreeLong spectral blending for Wan 2.2 video.

    Hooks into Wan's transformer blocks and runs dual-stream attention:
      1. Global stream  – full-sequence attention (preserves motion direction)
      2. Local stream   – windowed attention     (preserves spatial detail)
      3. Spectral blend – low-freq from global, high-freq from local

    Based on FreeLong (NeurIPS 2024). Improved frame auto-detection covers
    all common Wan video lengths (5–161 frames) without manual tuning.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable / disable MoveAhead. Leave connected and toggle for quick A/B.",
                }),
                "blend_strength": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": (
                        "How strongly the spectral blend affects the output. "
                        "0 = passthrough, 1 = full blend. 0.8 is a safe default."
                    ),
                }),
                "low_freq_ratio": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "Fraction of the frequency spectrum taken from the global (full-sequence) "
                        "stream. Higher → smoother motion, less detail. Lower → sharper but may drift. "
                        "0.8 is a good balance for most scenarios."
                    ),
                }),
                "auto_window": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "AUTO window mode. When enabled, MoveAhead auto-detects latent frames and "
                        "uses a window of ~40% of the detected length."
                    ),
                }),
                "manual_window_frames": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Manual window size in VIDEO frames (example: 33 for 81-frame videos). "
                        "Used only if AUTO window is OFF."
                    ),
                }),
                "blend_curve": (["flat", "sigmoid"], {
                    "default": "flat",
                    "tooltip": (
                        "flat = constant blend_strength across all blocks (original behaviour). "
                        "sigmoid = strength ramps from near-zero at the first blocks to full "
                        "blend_strength at later blocks. Early blocks keep global motion anchoring, "
                        "later blocks progressively add detail blending."
                    ),
                }),
                "blend_start_block": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 40,
                    "step": 1,
                    "tooltip": "Transformer block at which blending starts (0 = first block).",
                }),
                "blend_end_block": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 40,
                    "step": 1,
                    "tooltip": "Transformer block at which blending ends (-1 = all blocks, respects actual model depth).",
                }),
                "vram_preset": (["low", "normal", "high"], {
                    "default": "normal",
                    "tooltip": (
                        "VRAM preset for the FFT blending implementation. "
                        "low = smallest chunks (slowest, safest), normal = balanced, high = larger chunks (faster)."
                    ),
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch_model"
    CATEGORY = "IAMCCS/Wan"
    DESCRIPTION = (
        "Apply FreeLong spectral blending to Wan 2.2 for better motion consistency "
        "in long videos. Connect between model loader and KSampler."
    )

    def patch_model(
        self,
        model,
        enabled: bool,
        blend_strength: float,
        low_freq_ratio: float,
        auto_window: bool,
        manual_window_frames: str,
        blend_curve: str,
        blend_start_block: int,
        blend_end_block: int,
        vram_preset: str,
    ):
        # 0 = auto (resolved later per-inference once frame count is known)
        if auto_window:
            local_window_size = 0
            win_str = "AUTO"
        else:
            vframes = _parse_video_frames(manual_window_frames)
            if vframes is None:
                local_window_size = 0
                win_str = "AUTO"
            else:
                local_window_size = ((vframes - 1) // 4) + 1
                win_str = f"{vframes}-frames({local_window_size}-latent)"

        if not enabled:
            logger.info("[IAMCCS.MoveAhead] DISABLED – returning model unchanged.")
            return (model,)

        vram_cfg = _vram_preset_settings(vram_preset)
        logger.info("[IAMCCS.MoveAhead] Applying spectral-blend patches: "
                "strength=%.2f curve=%s low_freq=%.2f window=%s blocks=[%d,%s] vram=%s",
                blend_strength, blend_curve, low_freq_ratio, win_str,
                    blend_start_block, "END" if blend_end_block == -1 else str(blend_end_block),
                    vram_preset)

        model = model.clone()

        settings: dict = {
            "blend_strength": blend_strength,
            "blend_curve": blend_curve,
            "low_freq_ratio": low_freq_ratio,
            "local_window_size": local_window_size,
            "chunk_cols": vram_cfg["chunk_cols"],
            "cpu_fallback": vram_cfg["cpu_fallback"],
            "blend_start_block": blend_start_block,
            "blend_end_block": blend_end_block,
            "_first_run": True,
            "_log_fallback": True,
            "_log_structure": True,
        }

        def _block_patch(block_args, extra_args):
            original_block = extra_args["original_block"]
            t_opts = block_args.get("transformer_options", {})
            block_index = t_opts.get("block_index", 0)
            total_blocks = t_opts.get("total_blocks", 40)

            b_start = settings["blend_start_block"]
            b_end = settings["blend_end_block"] if settings["blend_end_block"] != -1 else total_blocks

            if block_index < b_start or block_index >= b_end:
                return original_block(block_args)

            strength = _compute_effective_strength(
                settings["blend_strength"], settings["blend_curve"], block_index, total_blocks
            )
            if strength == 0.0:
                return original_block(block_args)

            try:
                x = block_args["img"]  # [B, seq, dim]

                if settings["_first_run"]:
                    settings["_first_run"] = False
                    logger.info("[IAMCCS.MoveAhead] First denoising step: "
                                "batch=%d seq_len=%d dim=%d", *x.shape)

                # --- Global stream ---
                global_out = original_block(block_args)
                global_x = global_out["img"]

                # --- Local (windowed) stream ---
                local_x = _run_windowed_attention(block_args, original_block, x, settings)

                if local_x is None:
                    # Fallback: spectral-smooth global only
                    return {"img": _spectral_blend(global_x, global_x,
                                                   settings["low_freq_ratio"], strength)}

                # --- Spectral blend ---
                num_frames = settings.get("cached_num_frames")
                spatial_size = settings.get("cached_spatial_size")
                if num_frames is None or spatial_size is None:
                    blended = _spectral_blend(global_x, local_x,
                                              settings["low_freq_ratio"], strength)
                    del global_x, local_x
                    return {"img": blended}

                blended = _spectral_blend_temporal_chunked_inplace(
                    global_x,
                    local_x,
                    num_frames=int(num_frames),
                    spatial_size=int(spatial_size),
                    low_freq_ratio=settings["low_freq_ratio"],
                    blend_strength=strength,
                    chunk_cols=int(settings.get("chunk_cols", 262_144)),
                    cpu_fallback=bool(settings.get("cpu_fallback", True)),
                )
                del global_x
                return {"img": blended}

            except Exception as exc:
                logger.error("[IAMCCS.MoveAhead] Error in block %d: %s",
                             block_index, exc, exc_info=True)
                return original_block(block_args)

        num_blocks = 40
        for i in range(num_blocks):
            model.set_model_patch_replace(_block_patch, "dit", "double_block", i)

        logger.info("[IAMCCS.MoveAhead] Registered patches for %d transformer blocks.", num_blocks)
        return (model,)


# ============================================================================
# IAMCCS_MoveAheadEnforcer  –  replaces WanFreeLongEnforcer
# ============================================================================

class IAMCCS_MoveAheadEnforcer:
    """
    IAMCCS MoveAhead Enforcer – stronger motion locking variant of FreeLong.

    Uses a 3-tier spectral blend:
      • Ultra-low frequencies  → 100 % global (motion skeleton locked)
      • Mid frequencies        → cosine blend global→local
      • High frequencies       → 100 % local  (fine detail)

    Additionally, the first N transformer blocks use global-only attention
    to lock the motion trajectory before detail processing begins.

    Use this when standard MoveAhead still shows motion drift or direction reversals.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable / disable the Enforcer.",
                }),
                "motion_lock_ratio": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.05,
                    "tooltip": (
                        "Fraction of the spectrum that is 100 %% locked to the global stream. "
                        "Higher = stronger motion lock but may reduce dynamism. "
                        "0.15 is a good starting point."
                    ),
                }),
                "blend_strength": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Overall blend strength. 0 = passthrough, 1 = full enforced blend.",
                }),
                "low_freq_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "Upper bound of the blended mid-frequency tier. "
                        "Frequencies above this are fully local (detail). "
                        "Must be > motion_lock_ratio."
                    ),
                }),
                "auto_window": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "AUTO window mode (recommended).",
                }),
                "manual_window_frames": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Manual window size in VIDEO frames (example: 33 for 81-frame videos). "
                        "Used only if AUTO window is OFF."
                    ),
                }),
                "blend_curve": (["flat", "sigmoid"], {
                    "default": "flat",
                    "tooltip": (
                        "flat = constant blend_strength. "
                        "sigmoid = ramps from near-zero in early blocks to full strength later. "
                        "Combines well with motion_lock_blocks."
                    ),
                }),
                "motion_lock_blocks": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 20,
                    "step": 1,
                    "tooltip": (
                        "First N transformer blocks run global-only attention (no local stream). "
                        "Establishes motion direction early. 5 is a good default. "
                        "0 = disable this feature."
                    ),
                }),
                "vram_preset": (["low", "normal", "high"], {
                    "default": "normal",
                    "tooltip": (
                        "VRAM preset for the FFT blending implementation. "
                        "low = smallest chunks (slowest, safest), normal = balanced, high = larger chunks (faster)."
                    ),
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch_model"
    CATEGORY = "IAMCCS/Wan"
    DESCRIPTION = (
        "Stricter FreeLong variant with locked motion skeleton. "
        "Use when standard MoveAhead still shows direction drift."
    )

    def patch_model(
        self,
        model,
        enabled: bool,
        motion_lock_ratio: float,
        blend_strength: float,
        low_freq_ratio: float,
        auto_window: bool,
        manual_window_frames: str,
        blend_curve: str,
        motion_lock_blocks: int,
        vram_preset: str,
    ):
        if auto_window:
            local_window_size = 0
            win_str = "AUTO"
        else:
            vframes = _parse_video_frames(manual_window_frames)
            if vframes is None:
                local_window_size = 0
                win_str = "AUTO"
            else:
                local_window_size = ((vframes - 1) // 4) + 1
                win_str = f"{vframes}-frames({local_window_size}-latent)"

        if not enabled:
            logger.info("[IAMCCS.MoveAhead.Enforcer] DISABLED – returning model unchanged.")
            return (model,)

        vram_cfg = _vram_preset_settings(vram_preset)
        logger.info("[IAMCCS.MoveAhead.Enforcer] Applying enforced patches: "
                    "lock=%.0f%% blend=[%.0f%%..%.0f%%] strength=%.2f curve=%s "
                "window=%s lock_blocks=%d vram=%s",
                    motion_lock_ratio * 100, motion_lock_ratio * 100, low_freq_ratio * 100,
                    blend_strength, blend_curve, win_str, motion_lock_blocks,
                    vram_preset)

        model = model.clone()

        settings: dict = {
            "motion_lock_ratio": motion_lock_ratio,
            "low_freq_ratio": low_freq_ratio,
            "blend_strength": blend_strength,
            "blend_curve": blend_curve,
            "local_window_size": local_window_size,
            "motion_lock_blocks": motion_lock_blocks,
            "chunk_cols": vram_cfg["chunk_cols"],
            "cpu_fallback": vram_cfg["cpu_fallback"],
            "_first_run": True,
            "_log_fallback": True,
            "_log_structure": True,
        }

        def _block_patch(block_args, extra_args):
            original_block = extra_args["original_block"]
            t_opts = block_args.get("transformer_options", {})
            block_index = t_opts.get("block_index", 0)
            total_blocks = t_opts.get("total_blocks", 40)

            strength = _compute_effective_strength(
                settings["blend_strength"], settings["blend_curve"], block_index, total_blocks
            )
            if strength == 0.0:
                return original_block(block_args)

            try:
                x = block_args["img"]

                if settings["_first_run"]:
                    settings["_first_run"] = False
                    logger.info("[IAMCCS.MoveAhead.Enforcer] First denoising step: "
                                "batch=%d seq_len=%d dim=%d", *x.shape)

                # --- Global stream (always) ---
                global_out = original_block(block_args)
                global_x = global_out["img"]

                # Early blocks: global only (motion skeleton phase)
                if block_index < settings["motion_lock_blocks"]:
                    return {"img": global_x}

                # --- Local stream ---
                local_x = _run_windowed_attention(block_args, original_block, x, settings)

                if local_x is None:
                    return {"img": global_x}

                # --- 3-tier enforced blend ---
                num_frames = settings.get("cached_num_frames")
                spatial_size = settings.get("cached_spatial_size")
                if num_frames is None or spatial_size is None:
                    blended = _enforced_spectral_blend(
                        global_x, local_x,
                        motion_lock_ratio=settings["motion_lock_ratio"],
                        low_freq_ratio=settings["low_freq_ratio"],
                        blend_strength=strength,
                    )
                    del global_x, local_x
                    return {"img": blended}

                blended = _enforced_spectral_blend_temporal_chunked_inplace(
                    global_x,
                    local_x,
                    num_frames=int(num_frames),
                    spatial_size=int(spatial_size),
                    motion_lock_ratio=settings["motion_lock_ratio"],
                    low_freq_ratio=settings["low_freq_ratio"],
                    blend_strength=strength,
                    chunk_cols=int(settings.get("chunk_cols", 262_144)),
                    cpu_fallback=bool(settings.get("cpu_fallback", True)),
                )
                del global_x
                return {"img": blended}

            except Exception as exc:
                logger.error("[IAMCCS.MoveAhead.Enforcer] Error in block %d: %s",
                             block_index, exc, exc_info=True)
                return original_block(block_args)

        num_blocks = 40
        for i in range(num_blocks):
            model.set_model_patch_replace(_block_patch, "dit", "double_block", i)

        logger.info("[IAMCCS.MoveAhead.Enforcer] Registered patches for %d transformer blocks.",
                    num_blocks)
        return (model,)


# ============================================================================
# IAMCCS_MotionScale  –  replaces WanMotionScale
# ============================================================================

class IAMCCS_MotionScale:
    """
    IAMCCS MotionScale – temporal RoPE position scaling for Wan 2.2.

    Sets ``rope_options`` in ``model_options["transformer_options"]`` so that
    ComfyUI's Wan diffusion model scales temporal (and optionally spatial)
    position embeddings before attention is computed.

    scale_t > 1 → frames appear further apart in time → faster apparent motion.
    scale_t < 1 → frames appear closer together       → slower apparent motion.

    Typical usage: generate with scale_t = 1.5, then RIFE-interpolate the
    output to fill the gaps → longer video with greater motion coverage.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable / disable motion scaling.",
                }),
                "scale_t": ("FLOAT", {
                    "default": 1.5,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.05,
                    "tooltip": (
                        "Temporal (time) position scale. "
                        ">1 = faster motion,  <1 = slower motion,  <0 = reversed positions. "
                        "1.5 is a good starting point for long-video generation."
                    ),
                }),
            },
            "optional": {
                "scale_y": ("FLOAT", {
                    "default": 1.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.05,
                    "tooltip": "Height (Y) spatial position scale. 1.0 = unchanged.",
                }),
                "scale_x": ("FLOAT", {
                    "default": 1.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.05,
                    "tooltip": "Width (X) spatial position scale. 1.0 = unchanged.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch_model"
    CATEGORY = "IAMCCS/Wan"
    DESCRIPTION = (
        "Scale temporal RoPE positions to control apparent motion speed. "
        "Use with RIFE interpolation for longer, more dynamic videos."
    )

    def patch_model(
        self,
        model,
        enabled: bool,
        scale_t: float,
        scale_y: float = 1.0,
        scale_x: float = 1.0,
        shift_t: float = 0.0,
        shift_y: float = 0.0,
        shift_x: float = 0.0,
    ):
        if not enabled:
            logger.info("[IAMCCS.MotionScale] DISABLED – returning model unchanged.")
            return (model,)

        no_scale = (scale_t == 1.0 and scale_y == 1.0 and scale_x == 1.0)
        no_shift = (shift_t == 0.0 and shift_y == 0.0 and shift_x == 0.0)
        if no_scale and no_shift:
            logger.info("[IAMCCS.MotionScale] All values at default – returning model unchanged.")
            return (model,)

        model = model.clone()

        rope_options = {
            "scale_t": scale_t,
            "scale_y": scale_y,
            "scale_x": scale_x,
            "shift_t": shift_t,
            "shift_y": shift_y,
            "shift_x": shift_x,
        }
        model.model_options.setdefault("transformer_options", {})["rope_options"] = rope_options

        logger.info("[IAMCCS.MotionScale] rope_options set: "
                    "scale_t=%.3f scale_y=%.3f scale_x=%.3f shift_t=%.2f shift_y=%.2f shift_x=%.2f",
                    scale_t, scale_y, scale_x, shift_t, shift_y, shift_x)
        return (model,)


# ============================================================================
# IAMCCS_MotionScaleAdvanced  –  replaces WanMotionScaleAdvanced
# ============================================================================

class IAMCCS_MotionScaleAdvanced:
    """
    IAMCCS MotionScale Advanced – RoPE scaling + theta modification.

    Extends MotionScale with direct access to ``rope_embedder.theta``.
    Theta controls the base frequency of the positional encoding:

    • Higher theta (e.g. 20 000–50 000) → longer effective context window,
      frames appear "closer together" in the embedding space – may help
      videos longer than 81 frames stay more coherent.
    • Lower theta (e.g. 2 000–5 000)    → frames appear "further apart",
      may increase motion intensity (or cause instability).

    ⚠ EXPERIMENTAL: theta modification changes fundamental model behaviour.
    Start with small deviations from the default (10 000) and compare outputs.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable / disable all modifications.",
                }),
            },
            "optional": {
                "scale_t": ("FLOAT", {
                    "default": 1.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.05,
                    "tooltip": "Temporal scale (same as basic MotionScale).",
                }),
                "scale_y": ("FLOAT", {
                    "default": 1.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.05,
                    "tooltip": "Height (Y) position scale.",
                }),
                "scale_x": ("FLOAT", {
                    "default": 1.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.05,
                    "tooltip": "Width (X) position scale.",
                }),
                "theta": ("FLOAT", {
                    "default": 10000.0,
                    "min": 100.0,
                    "max": 1000000.0,
                    "step": 100.0,
                    "tooltip": (
                        "RoPE base frequency. Wan 2.2 default = 10 000. "
                        "Higher (20 000–50 000) = longer effective context, may help >81-frame coherence. "
                        "Lower (2 000–5 000) = higher motion intensity / potential instability. "
                        "⚠ EXPERIMENTAL – change in small increments."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch_model"
    CATEGORY = "IAMCCS/Wan"
    DESCRIPTION = (
        "Advanced RoPE control: motion scaling + theta modification. "
        "Experimental — use small deviations from theta=10000."
    )

    def patch_model(
        self,
        model,
        enabled: bool,
        scale_t: float = 1.0,
        scale_y: float = 1.0,
        scale_x: float = 1.0,
        theta: float = 10000.0,
    ):
        if not enabled:
            logger.info("[IAMCCS.MotionScaleAdv] DISABLED – returning model unchanged.")
            return (model,)

        no_scale = (scale_t == 1.0 and scale_y == 1.0 and scale_x == 1.0)
        default_theta = (theta == 10000.0)

        if no_scale and default_theta:
            logger.info("[IAMCCS.MotionScaleAdv] All values at default – returning model unchanged.")
            return (model,)

        model = model.clone()

        if not no_scale:
            rope_options = {"scale_t": scale_t, "scale_y": scale_y, "scale_x": scale_x}
            model.model_options.setdefault("transformer_options", {})["rope_options"] = rope_options
            logger.info("[IAMCCS.MotionScaleAdv] rope_options set: "
                        "scale_t=%.3f scale_y=%.3f scale_x=%.3f", scale_t, scale_y, scale_x)

        if not default_theta:
            try:
                diffusion_model = model.model.diffusion_model
                if hasattr(diffusion_model, "rope_embedder"):
                    original_theta = diffusion_model.rope_embedder.theta
                    diffusion_model.rope_embedder.theta = theta
                    logger.info("[IAMCCS.MotionScaleAdv] rope_embedder.theta: %.0f → %.0f",
                                original_theta, theta)
                else:
                    logger.warning("[IAMCCS.MotionScaleAdv] Model has no rope_embedder – "
                                   "theta not applied.")
            except Exception as exc:
                logger.error("[IAMCCS.MotionScaleAdv] Failed to set theta: %s", exc)

        return (model,)
