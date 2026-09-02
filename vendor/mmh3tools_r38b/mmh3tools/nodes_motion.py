"""Where a clip asks a latent token to hold more motion than it can.

H3's temporal grid is `(1, 4, 4, 4, 4)`: four of every five latent time tokens
carry FOUR pixel frames. When motion is fast enough that those four frames need
four distinct poses, one token cannot represent them, and the decode smears. The
defect is structural, so it does not respond to steps or resolution -- the poses
were never generated, and re-denoising cannot recover what was never there.

This node measures where that happens. It reads a finished latent, so it says
nothing about a clip you have not rendered; it exists to answer "does this
footage have the problem at all" before anything is built on the answer.

METHOD. Third difference of the latent values along the token axis, |d3|,
averaged over channels and space, then phase-normalised. Prior art is the jerk
oracle in matlowai/ComfyUI-MAINodes (MIT), which compiles the same profile into
a frame-hold map; this is the measurement half only, reimplemented against
`common.py`'s grid rather than copied.

WHAT IT CANNOT DO. A quantile threshold marks a fixed share of tokens hot no
matter what it is given, so the profile RANKS and does not DETECT -- MAINodes
records the same limit ("the oracle can rank but cannot abstain"). The contrast
ratios in the report exist for that reason: on a clip with no burst the flagged
tokens are no hotter than the rest and the ratio sits near 1.0, which is the
only thing here that can say "nothing to see".
"""

import json
import logging

from comfy_api.latest import io

from .common import FPS, FRAME_PER_TOKEN, frame_at_latent, unpack_av

__all__ = ["MMH3MotionOverload", "contrast", "hot_runs", "is_flat", "jerk_profile",
           "phase_normalise", "ratio_str", "span_frames"]

DIFF_ORDER = 3


def phase_normalise(prof):
    """Divide out the (1, 4, 4, 4, 4) bias so tokens are comparable.

    A phase-0 token spans one pixel frame and phases 1-4 span four, so a raw
    per-token difference is measuring the grid as much as the motion. Every
    per-token statistic on H3 needs this before tokens are compared to
    each other.
    """
    import numpy as np

    prof = np.asarray(prof, dtype="float64").copy()
    period = len(FRAME_PER_TOKEN)
    for phase in range(period):
        mean = prof[phase::period].mean() if len(prof[phase::period]) else 0.0
        if mean > 0:
            prof[phase::period] /= mean
    return prof


def jerk_profile(video, phase_normalize=True):
    """Per-token motion-overload score from a video latent [B, C, T, H, W]."""
    import numpy as np

    v = video.detach().float().cpu().numpy()
    t_lat = v.shape[2]
    if t_lat <= DIFF_ORDER:
        return np.zeros(t_lat, dtype="float64")
    j = np.abs(np.diff(v, n=DIFF_ORDER, axis=2)).mean(axis=(0, 1, 3, 4))
    # centre the difference on the token it describes; edge-pad rather than
    # zero-pad so the ends do not read as artificially calm
    lead = DIFF_ORDER // 2
    prof = np.pad(j, (lead, t_lat - len(j) - lead), mode="edge")
    return phase_normalise(prof) if phase_normalize else prof


def hot_runs(hot):
    """Contiguous [start, stop) token runs where `hot` is True."""
    runs, start = [], None
    for i, flag in enumerate(list(hot) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i))
            start = None
    return runs


def span_frames(t0, t1, length):
    """Pixel-frame span [first, last] covered by token run [t0, t1)."""
    first = frame_at_latent(t0)
    last = min(length, frame_at_latent(t1)) - 1
    return first, max(first, last)


def is_flat(prof):
    """Whether the profile carries no variation at all, to its own scale."""
    import numpy as np

    scale = float(np.abs(prof).mean())
    return scale <= 0.0 or float(prof.max() - prof.min()) <= scale * 1e-9


def contrast(prof, hot):
    """(peak/typical, hot mean / cold mean). 1.0 means the cut separates nothing.

    A FLAT profile returns (1.0, 1.0), which is the whole point of having these
    numbers: identical tokens are the strongest possible "nothing here", and a
    ratio against a zero baseline would report it as infinite separation --
    exactly backwards. Genuine unbounded separation (a hot span against tokens
    that are exactly still) stays inf and the report renders it as a word.
    """
    import numpy as np

    if is_flat(prof):
        return 1.0, 1.0
    # median is the right baseline -- one spike barely moves it -- but a profile
    # more than half of whose tokens are exactly zero has none, so fall back to
    # the mean, which a non-negative profile with variation always has
    baseline = float(np.median(prof)) or float(prof.mean())
    peak = float(prof.max()) / baseline if baseline > 0 else float("inf")
    cold, hot_vals = prof[~hot], prof[hot]
    if not len(cold) or not len(hot_vals) or float(cold.mean()) <= 0:
        return peak, float("inf")
    return peak, float(hot_vals.mean()) / float(cold.mean())


def ratio_str(x):
    """Render a ratio, naming the unbounded case rather than printing 'inf'."""
    return "unbounded" if not (x == x) or x in (float("inf"),) else "%.2f" % x


class MMH3MotionOverload(io.ComfyNode):
    """Rank a rendered clip's latent tokens by how much motion they carry."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3MotionOverload",
            display_name="MMH3 Motion Overload",
            category="MMH3Tools/utils",
            description=(
                "Measures which latent time tokens of a RENDERED clip carry more "
                "motion than one token can represent -- the cause of H3's fast-motion "
                "smear, since four of every five tokens span four pixel frames. Third "
                "difference of the latent along the token axis, phase-normalised for "
                "the (1,4,4,4,4) grid. Reports contrast ratios and the hot spans in "
                "tokens, frames and seconds. Reads a finished latent, so it diagnoses "
                "footage you already have."
            ),
            inputs=[
                io.Latent.Input(
                    "latent",
                    tooltip="A rendered H3 latent. The video half is measured; audio "
                            "is ignored."),
                io.Float.Input(
                    "quantile", default=0.75, min=0.0, max=0.99, step=0.01,
                    optional=True,
                    tooltip="Tokens at or above this quantile of the profile are "
                            "marked hot. It selects a fixed share of tokens whatever "
                            "the clip contains; the contrast ratios say whether that "
                            "share is meaningfully hotter than the rest."),
                io.Boolean.Input(
                    "phase_normalize", default=True, optional=True,
                    tooltip="Divide each token's score by the mean of its grid phase. "
                            "Off leaves the (1,4,4,4,4) span difference in the "
                            "profile, so phase-0 tokens score differently for "
                            "covering one frame rather than four."),
            ],
            outputs=[
                io.String.Output(display_name="profile_json"),
                io.Float.Output(display_name="hot_over_cold"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, latent, quantile=0.75, phase_normalize=True) -> io.NodeOutput:
        import numpy as np

        notes = []
        video, audio = unpack_av(latent, name="latent", allow_video_only=True)
        if audio is None:
            notes.append("video-only latent; nothing here reads audio, so this is "
                         "not a limitation, only a note on what was handed over")

        t_lat = int(video.shape[2])
        length = frame_at_latent(t_lat)
        prof = jerk_profile(video, phase_normalize=phase_normalize)

        if t_lat <= DIFF_ORDER:
            notes.append("%d tokens is too few for a %d-order difference; the profile "
                         "is all zeros and every number below is meaningless"
                         % (t_lat, DIFF_ORDER))
            hot = np.zeros(t_lat, dtype=bool)
            thr = 0.0
        else:
            thr = float(np.quantile(prof, quantile))
            hot = prof >= thr

        peak_ratio, hot_cold = contrast(prof, hot) if t_lat > DIFF_ORDER else (1.0, 1.0)
        if t_lat > DIFF_ORDER and is_flat(prof):
            notes.append("the profile has NO variation -- every token scores the same, "
                         "so there is nothing here to rank and the spans below are an "
                         "artefact of the quantile, not a finding")
        runs = hot_runs(hot)
        spans = []
        for t0, t1 in runs:
            f0, f1 = span_frames(t0, t1, length)
            spans.append({"token_start": t0, "token_stop": t1,
                          "frame_start": f0, "frame_end": f1,
                          "t_start_s": round(f0 / float(FPS), 3),
                          "t_end_s": round(f1 / float(FPS), 3),
                          "peak": round(float(prof[t0:t1].max()), 4)})

        data = {
            "tokens": t_lat,
            "frames": length,
            "duration_s": round(length / float(FPS), 3),
            "quantile": float(quantile),
            "threshold": round(thr, 6),
            "phase_normalized": bool(phase_normalize),
            # null rather than Infinity: json.dumps would emit a bare `Infinity`,
            # which Python reads back but strict JSON parsers reject
            "peak_over_median": (round(float(peak_ratio), 4)
                                 if peak_ratio == peak_ratio
                                 and peak_ratio != float("inf") else None),
            "hot_over_cold": (round(float(hot_cold), 4)
                              if hot_cold == hot_cold
                              and hot_cold != float("inf") else None),
            "hot_tokens": int(hot.sum()),
            "spans": spans,
            "profile": [round(float(x), 5) for x in prof],
        }

        lines = ["MMH3 Motion Overload -- %d tokens, %d frames, %.2fs at %d fps"
                 % (t_lat, length, length / float(FPS), FPS), ""]
        lines.append("contrast")
        lines.append("  peak / typical     %s" % ratio_str(peak_ratio))
        lines.append("  hot / cold mean    %-9s (1.00 = the cut is separating "
                     "nothing)" % ratio_str(hot_cold))
        lines.append("  quantile %.2f marks %d of %d tokens hot BY CONSTRUCTION. This "
                     "ranks, it does not detect -- read the ratios before believing "
                     "the spans." % (quantile, int(hot.sum()), t_lat))
        lines.append("")
        if spans:
            lines.append("hot spans")
            for s in spans:
                lines.append("  tokens %3d-%-3d  frames %4d-%-4d  %6.2fs-%-6s  "
                             "peak %.2f"
                             % (s["token_start"], s["token_stop"] - 1,
                                s["frame_start"], s["frame_end"],
                                s["t_start_s"], "%.2fs" % s["t_end_s"], s["peak"]))
        else:
            lines.append("hot spans: none")
        if not phase_normalize:
            lines.append("")
            lines.append("phase normalisation OFF -- phase-0 tokens span one pixel "
                         "frame and phases 1-4 span four, so this profile is partly "
                         "measuring the grid.")
        if notes:
            lines.append("")
            lines.extend("  ! " + n for n in notes)

        report = "\n".join(lines)
        logging.info("[MMH3MotionOverload] %d tokens, peak/median %.2f, hot/cold %.2f",
                     t_lat, peak_ratio, hot_cold)
        return io.NodeOutput(json.dumps(data), float(hot_cold), report)
