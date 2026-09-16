"""Final-frame lock detection for H3 continuous handover.

v0.4.6 builds on the Stable-Tail Consensus detector introduced in v0.4.5.
The current defaults are calibrated for safe-early handover: tolerate decoded
brightness/texture shimmer in the final-state gate, while the residual-motion
gate and consecutive-outlier limit continue to reject sustained motion.

v0.4.2 originally stopped treating generic low motion as the primary freeze
signal. MiniMax H3 FL2VA often decelerates while converging toward the supplied
last frame, then locks onto (nearly) the final image and repeats it. A low-motion
detector can therefore cut too early.

The v0.4.2 detector instead asks two stricter questions for the trailing run:

  1. Does each frame already match the *actual final generated frame* extremely
     closely in both global RGB error and changed-area ratio?
  2. Are the frame-to-frame transitions inside that run also near-static?

Only a run satisfying BOTH conditions all the way to the clip end is called a
freeze/lock. The full rendered video and full AV latent remain untouched; this
module only recommends where the next clip should take its history from.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from .latent_math import video_latent_t, phase_aware_context_slice, phase_aligned_extended_context_slice, snap_landing_tail
except ImportError:  # direct test import from the package directory
    from latent_math import video_latent_t, phase_aware_context_slice, phase_aligned_extended_context_slice, snap_landing_tail


def _target_hw(height: int, width: int, max_edge: int):
    max_edge = max(32, int(max_edge))
    scale = min(1.0, max_edge / float(max(height, width)))
    th = max(16, int(round(height * scale)))
    tw = max(16, int(round(width * scale)))
    return th, tw


def _prepare_rgb(images: torch.Tensor, max_edge: int) -> torch.Tensor:
    """Downsample RGB frames and lightly blur them. Returns [T,3,H,W]."""
    if images.ndim != 4 or images.shape[-1] < 3:
        raise ValueError(f"Expected IMAGE tensor [T,H,W,C>=3], got {tuple(images.shape)}")
    x = images[..., :3].detach().float().movedim(-1, 1)
    th, tw = _target_hw(int(x.shape[-2]), int(x.shape[-1]), max_edge)
    if (th, tw) != (int(x.shape[-2]), int(x.shape[-1])):
        x = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False, antialias=True)
    # A tiny blur suppresses single-pixel H3 shimmer without erasing visible
    # subject/pose motion or convergence toward the final frame.
    x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    return x


def _to_gray(rgb: torch.Tensor) -> torch.Tensor:
    return 0.2126 * rgb[:, 0:1] + 0.7152 * rgb[:, 1:2] + 0.0722 * rgb[:, 2:3]


def _trailing_run(flags) -> int:
    run = 0
    for flag in reversed(flags):
        if flag:
            run += 1
        else:
            break
    return run


def _handover_snaps(
    frame_count: int,
    phase_aware_target_end_frame: int,
    legacy_ideal_last_frame: int,
    context_frames: int,
):
    """Return phase-aligned, phase-aware, and conservative legacy cutoffs."""
    frame_count = int(frame_count)
    vt = video_latent_t(frame_count)
    if frame_count < context_frames:
        raise ValueError("Clip is shorter than the requested handover context")

    pa = phase_aware_context_slice(
        vt, context_frames, ideal_last_frame=int(phase_aware_target_end_frame)
    )
    aligned = phase_aligned_extended_context_slice(
        vt, context_frames, ideal_last_frame=int(phase_aware_target_end_frame)
    )
    legacy_tail, legacy_end = snap_landing_tail(
        frame_count, int(legacy_ideal_last_frame), context_frames
    )
    return {
        # v0.4.3 default recommendation: same late end boundary as phase-aware,
        # but start from phase 0 and extend the source history backward as needed.
        "handover_end_frame": int(aligned["source_end_frame"] - 1),
        "landing_tail_frames": int(aligned["ignored_tail_frames"]),
        "phase_aligned_end_t": int(aligned["end_t"]),
        "phase_aligned_start_t": int(aligned["start_t"]),
        "phase_aligned_context_frames": int(aligned["actual_context_frames"]),
        "phase_aligned_context_extension_frames": int(aligned["context_extension_frames"]),
        "phase_aligned_cutoff_loss_frames": int(aligned["cutoff_loss_frames"]),
        "phase_aligned_source_start_frame": int(aligned["source_start_frame"]),
        "phase_aligned_source_end_frame": int(aligned["source_end_frame"]),
        # Keep v0.4.x phase-aware diagnostics for A/B comparison.
        "phase_aware_handover_end_frame": int(pa["source_end_frame"] - 1),
        "phase_aware_landing_tail_frames": int(pa["ignored_tail_frames"]),
        "phase_aware_end_t": int(pa["end_t"]),
        "phase_aware_start_t": int(pa["start_t"]),
        "phase_aware_context_frames": int(pa["actual_context_frames"]),
        "phase_aware_cutoff_loss_frames": int(pa["cutoff_loss_frames"]),
        "legacy_handover_end_frame": int(legacy_end),
        "legacy_landing_tail_frames": int(legacy_tail),
    }


def phase_aware_safety_from_confidence(
    confidence: float, configured_safety_margin: int, safety_mode: str = "adaptive"
) -> int:
    """Return the pixel safety margin used for the phase-aware target.

    ``fixed`` always respects the configured margin.  This is the v0.4.6
    recommended behavior because a slightly early motion handover is safer than
    allowing already-locked frames into the continuation context.

    ``adaptive`` preserves the historical v0.4.1-v0.4.5 behavior for A/B and
    legacy-metadata compatibility: high-confidence locks can reduce safety to 0.
    """
    c = float(confidence)
    configured = max(0, int(configured_safety_margin))
    mode = str(safety_mode).strip().lower()
    if mode == "fixed":
        return configured
    if mode != "adaptive":
        raise ValueError(f"Unknown safety_mode {safety_mode!r}; expected 'fixed' or 'adaptive'")
    if c >= 0.80:
        return 0
    if c >= 0.55:
        return max(1, configured)
    return max(2, configured)


def analyze_freeze_tail(
    images: torch.Tensor,
    analysis_window: int = 72,
    freeze_hold: int = 12,
    safety_margin: int = 3,
    context_frames: int = 22,
    analysis_size: int = 192,
    final_mean_diff_threshold: float = 0.0120,
    final_active_pixel_threshold: float = 0.025,
    max_final_active_area_percent: float = 3.0,
    transition_mean_diff_threshold: float = 0.0020,
    transition_active_pixel_threshold: float = 0.010,
    max_transition_active_area_percent: float = 1.0,
    min_static_transition_percent: float = 70.0,
    max_consecutive_motion_outliers: int = 2,
    final_reference_frames: int = 15,
    min_final_match_percent: float = 75.0,
    max_consecutive_final_outliers: int = 3,
    safety_mode: str = "fixed",
) -> dict:
    """Detect H3's stable trailing final-state lock.

    v0.4.5 no longer uses one potentially shimmering last decoded frame as the
    sole visual reference.  Instead it builds a robust pixel-wise MEDIAN from
    the final ``final_reference_frames`` frames and asks whether a trailing
    suffix belongs to that stable final-state consensus.

    Primary stable-tail condition:
      * each frame is compared to the median final-state reference,
      * at least ``min_final_match_percent`` percent of frames in the suffix
        satisfy the strict final-state similarity thresholds,
      * no more than ``max_consecutive_final_outliers`` primary mismatches occur
        consecutively, and the candidate start frame itself must match.

    Secondary residual-motion condition is unchanged from v0.4.4:
      * at least ``min_static_transition_percent`` percent of transitions are
        near-static,
      * no more than ``max_consecutive_motion_outliers`` failing transitions
        occur consecutively.

    This makes the detector robust to tiny final-frame shimmer/brightness/VAE
    outliers without turning slow, sustained convergence into a lock.  The
    rendered video and saved AV latent remain untouched; only handover metadata
    is produced.
    """
    if images.ndim != 4:
        raise ValueError(f"Expected IMAGE tensor [T,H,W,C], got {tuple(images.shape)}")
    n = int(images.shape[0])
    if n < 2:
        raise ValueError("Need at least two frames for handover analysis")

    analysis_window = max(2, min(n, int(analysis_window)))
    freeze_hold = max(2, int(freeze_hold))
    safety_margin = max(0, int(safety_margin))
    context_frames = int(context_frames)
    final_max_area_ratio = max(0.0, float(max_final_active_area_percent)) / 100.0
    transition_max_area_ratio = max(0.0, float(max_transition_active_area_percent)) / 100.0
    min_static_ratio = max(0.0, min(1.0, float(min_static_transition_percent) / 100.0))
    max_consecutive_motion_outliers = max(0, int(max_consecutive_motion_outliers))
    min_final_match_ratio = max(0.0, min(1.0, float(min_final_match_percent) / 100.0))
    max_consecutive_final_outliers = max(0, int(max_consecutive_final_outliers))
    final_reference_frames = max(1, int(final_reference_frames))
    safety_mode = str(safety_mode).strip().lower()
    if safety_mode not in ("fixed", "adaptive"):
        raise ValueError(f"Unknown safety_mode {safety_mode!r}; expected 'fixed' or 'adaptive'")
    requested_window = analysis_window

    # Build the stable reference ONCE from the true clip tail, so expanding the
    # analysis window cannot change what "final state" means.
    full_tail_rgb = _prepare_rgb(images[max(0, n - final_reference_frames):], analysis_size)
    # Pixel-wise median is intentionally robust to an odd final decoded frame.
    stable_final_ref = full_tail_rgb.median(dim=0, keepdim=True).values
    actual_final_reference_frames = int(full_tail_rgb.shape[0])

    def compute_stats(start_frame: int):
        rgb = _prepare_rgb(images[start_frame:], analysis_size)
        gray = _to_gray(rgb)

        # PRIMARY: similarity to the stable MEDIAN final-state reference.
        final_diff = (rgb - stable_final_ref).abs()
        final_mean = final_diff.mean(dim=(1, 2, 3))
        final_pixel_mag = final_diff.mean(dim=1)  # [T,H,W]
        final_active = (
            final_pixel_mag >= float(final_active_pixel_threshold)
        ).float().mean(dim=(1, 2))
        final_match = (
            (final_mean <= float(final_mean_diff_threshold))
            & (final_active <= final_max_area_ratio)
        )

        # SECONDARY: residual frame-to-frame motion, robust to isolated outliers.
        trans_diff = (gray[1:] - gray[:-1]).abs()
        trans_mean = trans_diff.mean(dim=(1, 2, 3))
        trans_active = (
            trans_diff[:, 0] >= float(transition_active_pixel_threshold)
        ).float().mean(dim=(1, 2))
        trans_static = (
            (trans_mean <= float(transition_mean_diff_threshold))
            & (trans_active <= transition_max_area_ratio)
        )

        return {
            "final_match": final_match.detach().cpu().tolist(),
            "final_mean": final_mean.detach().cpu().tolist(),
            "final_active": final_active.detach().cpu().tolist(),
            "trans_static": trans_static.detach().cpu().tolist(),
            "trans_mean": trans_mean.detach().cpu().tolist(),
            "trans_active": trans_active.detach().cpu().tolist(),
        }

    def max_false_streak(flags) -> int:
        best = cur = 0
        for flag in flags:
            if flag:
                cur = 0
            else:
                cur += 1
                best = max(best, cur)
        return best

    def primary_gate(stats, frame_start_idx: int):
        """Stable-final-state consensus for suffix [frame_start_idx .. final]."""
        flags = stats["final_match"][frame_start_idx:]
        total = len(flags)
        if total <= 0:
            return {
                "passed": False, "match_count": 0, "outlier_count": 0,
                "match_ratio": 0.0, "max_outlier_streak": 0,
            }
        match_count = sum(1 for x in flags if x)
        outlier_count = total - match_count
        match_ratio = match_count / float(total)
        max_streak = max_false_streak(flags)
        # Never declare the lock before a frame that itself already belongs to
        # the stable final state. Outliers are tolerated only *inside* the tail.
        start_matches = bool(flags[0])
        return {
            "passed": bool(
                start_matches
                and match_ratio >= min_final_match_ratio
                and max_streak <= max_consecutive_final_outliers
            ),
            "match_count": int(match_count),
            "outlier_count": int(outlier_count),
            "match_ratio": float(match_ratio),
            "max_outlier_streak": int(max_streak),
        }

    def residual_gate(stats, frame_start_idx: int):
        """Evaluate residual motion for suffix [frame_start_idx .. final]."""
        trans_flags = stats["trans_static"][frame_start_idx:]
        total = len(trans_flags)
        if total <= 0:
            return {
                "passed": True,
                "static_count": 0,
                "outlier_count": 0,
                "static_ratio": 1.0,
                "max_outlier_streak": 0,
            }
        static_count = sum(1 for x in trans_flags if x)
        outlier_count = total - static_count
        static_ratio = static_count / float(total)
        max_streak = max_false_streak(trans_flags)
        return {
            "passed": bool(
                static_ratio >= min_static_ratio
                and max_streak <= max_consecutive_motion_outliers
            ),
            "static_count": int(static_count),
            "outlier_count": int(outlier_count),
            "static_ratio": float(static_ratio),
            "max_outlier_streak": int(max_streak),
        }

    def find_lock(stats):
        """Return earliest suffix passing stable-final AND residual consensus."""
        total_frames = len(stats["final_match"])
        latest_allowed_start = total_frames - freeze_hold
        first_primary_idx = None
        first_primary_gate = None
        last_residual_gate = residual_gate(stats, max(0, latest_allowed_start))

        if latest_allowed_start < 0:
            return None, None, None, last_residual_gate

        for candidate_start in range(0, latest_allowed_start + 1):
            pg = primary_gate(stats, candidate_start)
            if not pg["passed"]:
                continue
            if first_primary_idx is None:
                first_primary_idx = candidate_start
                first_primary_gate = pg
            rg = residual_gate(stats, candidate_start)
            last_residual_gate = rg
            if rg["passed"]:
                return candidate_start, pg, first_primary_idx, rg

        return None, first_primary_gate, first_primary_idx, last_residual_gate

    start = n - analysis_window
    stats = compute_stats(start)
    lock_start_idx, primary_gate_info, primary_candidate_idx, gate = find_lock(stats)

    # If the requested analysis window already begins inside a stable final-state
    # consensus, expand backward so the true lock start is not clipped by window.
    if start > 0 and primary_gate(stats, 0)["passed"]:
        start = 0
        stats = compute_stats(start)
        lock_start_idx, primary_gate_info, primary_candidate_idx, gate = find_lock(stats)

    freeze_detected = lock_start_idx is not None
    trailing_frames = (len(stats["final_match"]) - lock_start_idx) if freeze_detected else 0
    freeze_start_frame = -1
    ideal_handover_end_frame = n - 1
    phase_aware_safety_margin = 0
    phase_aware_target_end_frame = n - 1

    if freeze_detected:
        diagnostic_start_idx = lock_start_idx
        no_lock_reason = ""
        primary_gate_info = primary_gate(stats, lock_start_idx)
        gate = residual_gate(stats, lock_start_idx)
    else:
        if primary_candidate_idx is not None:
            diagnostic_start_idx = int(primary_candidate_idx)
            primary_gate_info = primary_gate(stats, diagnostic_start_idx)
            gate = residual_gate(stats, diagnostic_start_idx)
            no_lock_reason = "residual_motion_gate_failed"
        else:
            diagnostic_start_idx = max(0, len(stats["final_mean"]) - min(freeze_hold, len(stats["final_mean"])))
            primary_gate_info = primary_gate(stats, diagnostic_start_idx)
            gate = residual_gate(stats, diagnostic_start_idx)
            no_lock_reason = "stable_final_consensus_failed"

    lock_final_means = stats["final_mean"][diagnostic_start_idx:]
    lock_final_areas = stats["final_active"][diagnostic_start_idx:]
    lock_trans_means = stats["trans_mean"][diagnostic_start_idx:]
    lock_trans_areas = stats["trans_active"][diagnostic_start_idx:]

    avg_final_mean = sum(lock_final_means) / max(1, len(lock_final_means))
    avg_final_area = sum(lock_final_areas) / max(1, len(lock_final_areas))
    avg_trans_mean = sum(lock_trans_means) / max(1, len(lock_trans_means)) if lock_trans_means else 0.0
    avg_trans_area = sum(lock_trans_areas) / max(1, len(lock_trans_areas)) if lock_trans_areas else 0.0

    if freeze_detected:
        freeze_start_frame = start + lock_start_idx

        final_mean_score = 1.0 - min(
            1.0, avg_final_mean / max(float(final_mean_diff_threshold), 1e-9)
        )
        final_area_score = 1.0 - min(
            1.0, avg_final_area / max(final_max_area_ratio, 1e-9)
        ) if final_max_area_ratio > 0 else 1.0
        trans_mean_score = 1.0 - min(
            1.0, avg_trans_mean / max(float(transition_mean_diff_threshold), 1e-9)
        )
        trans_area_score = 1.0 - min(
            1.0, avg_trans_area / max(transition_max_area_ratio, 1e-9)
        ) if transition_max_area_ratio > 0 else 1.0
        run_strength = min(1.0, trailing_frames / float(max(freeze_hold * 2, 1)))
        primary_consensus_score = float(primary_gate_info["match_ratio"])
        confidence = max(0.0, min(
            1.0,
            0.35 * final_mean_score
            + 0.20 * final_area_score
            + 0.10 * trans_mean_score
            + 0.10 * trans_area_score
            + 0.10 * run_strength
            + 0.15 * primary_consensus_score
        ))

        ideal_handover_end_frame = freeze_start_frame - 1 - safety_margin
        ideal_handover_end_frame = max(context_frames - 1, ideal_handover_end_frame)

        phase_aware_safety_margin = phase_aware_safety_from_confidence(
            confidence, safety_margin, safety_mode=safety_mode
        )
        phase_aware_target_end_frame = freeze_start_frame - 1 - phase_aware_safety_margin
        phase_aware_target_end_frame = max(context_frames - 1, phase_aware_target_end_frame)
    else:
        confidence = 0.0

    snapped = _handover_snaps(
        n, phase_aware_target_end_frame, ideal_handover_end_frame, context_frames
    )

    result = {
        "available": True,
        "version": 9,
        "detector_mode": "stable_tail_consensus",
        "frame_count": n,
        "analysis_window": requested_window,
        "actual_analyzed_frames": n - start,
        "analysis_start_frame": start,
        "freeze_detected": bool(freeze_detected),
        "freeze_start_frame": int(freeze_start_frame),
        "lock_start_frame": int(freeze_start_frame),
        "trailing_locked_frames": int(trailing_frames),
        # Backward-compatible name; in v0.4.5 this is the chosen/diagnostic
        # stable-tail consensus length rather than a strict contiguous run.
        "primary_final_match_frames": int(len(stats["final_match"]) - diagnostic_start_idx),
        "trailing_static_transitions": int(gate["static_count"]),
        "freeze_hold": freeze_hold,
        "safety_margin": safety_margin,
        "safety_mode": safety_mode,
        "ideal_handover_end_frame": int(ideal_handover_end_frame),
        "phase_aware_target_end_frame": int(phase_aware_target_end_frame),
        "phase_aware_effective_safety_margin": int(phase_aware_safety_margin),
        "handover_end_frame": int(snapped["handover_end_frame"]),
        "landing_tail_frames": int(snapped["landing_tail_frames"]),
        "phase_aligned_target_end_frame": int(phase_aware_target_end_frame),
        "phase_aligned_end_t": int(snapped["phase_aligned_end_t"]),
        "phase_aligned_start_t": int(snapped["phase_aligned_start_t"]),
        "phase_aligned_context_frames": int(snapped["phase_aligned_context_frames"]),
        "phase_aligned_context_extension_frames": int(snapped["phase_aligned_context_extension_frames"]),
        "phase_aligned_cutoff_loss_frames": int(snapped["phase_aligned_cutoff_loss_frames"]),
        "phase_aligned_source_start_frame": int(snapped["phase_aligned_source_start_frame"]),
        "phase_aligned_source_end_frame": int(snapped["phase_aligned_source_end_frame"]),
        "phase_aware_handover_end_frame": int(snapped["phase_aware_handover_end_frame"]),
        "phase_aware_landing_tail_frames": int(snapped["phase_aware_landing_tail_frames"]),
        "phase_aware_end_t": int(snapped["phase_aware_end_t"]),
        "phase_aware_start_t": int(snapped["phase_aware_start_t"]),
        "phase_aware_context_frames": int(snapped["phase_aware_context_frames"]),
        "phase_aware_cutoff_loss_frames": int(snapped["phase_aware_cutoff_loss_frames"]),
        "legacy_handover_end_frame": int(snapped["legacy_handover_end_frame"]),
        "legacy_landing_tail_frames": int(snapped["legacy_landing_tail_frames"]),
        "context_frames": context_frames,
        # Stable-tail median-reference configuration and diagnostics.
        "final_reference_frames": int(actual_final_reference_frames),
        "min_final_match_percent": float(min_final_match_percent),
        "max_consecutive_final_outliers": int(max_consecutive_final_outliers),
        "primary_final_match_count": int(primary_gate_info["match_count"]),
        "primary_final_match_outliers": int(primary_gate_info["outlier_count"]),
        "primary_final_match_ratio_percent": float(primary_gate_info["match_ratio"] * 100.0),
        "primary_final_max_consecutive_outliers": int(primary_gate_info["max_outlier_streak"]),
        "primary_gate_passed": bool(primary_gate_info["passed"]),
        # Final-state similarity thresholds + diagnostics.
        "final_mean_diff_threshold": float(final_mean_diff_threshold),
        "final_active_pixel_threshold": float(final_active_pixel_threshold),
        "max_final_active_area_percent": float(max_final_active_area_percent),
        "lock_final_mean_diff": float(avg_final_mean),
        "lock_final_active_area_percent": float(avg_final_area * 100.0),
        # Secondary robust residual-motion thresholds + diagnostics.
        "transition_mean_diff_threshold": float(transition_mean_diff_threshold),
        "transition_active_pixel_threshold": float(transition_active_pixel_threshold),
        "max_transition_active_area_percent": float(max_transition_active_area_percent),
        "min_static_transition_percent": float(min_static_transition_percent),
        "max_consecutive_motion_outliers": int(max_consecutive_motion_outliers),
        "lock_transition_mean_diff": float(avg_trans_mean),
        "lock_transition_active_area_percent": float(avg_trans_area * 100.0),
        "residual_static_transitions": int(gate["static_count"]),
        "residual_total_transitions": int(gate["static_count"] + gate["outlier_count"]),
        "residual_motion_outliers": int(gate["outlier_count"]),
        "residual_static_ratio_percent": float(gate["static_ratio"] * 100.0),
        "residual_max_consecutive_outliers": int(gate["max_outlier_streak"]),
        "residual_gate_passed": bool(gate["passed"]),
        "no_lock_reason": no_lock_reason,
        "trailing_mean_diff": float(avg_trans_mean),
        "trailing_active_area_percent": float(avg_trans_area * 100.0),
        "confidence": float(confidence),
    }
    return result
