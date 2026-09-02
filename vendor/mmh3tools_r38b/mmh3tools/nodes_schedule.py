"""Ask for a length; get a schedule that actually tiles.

The frame calculator answers "what does 22.2 seconds round to". Useful, and not
the question. Three of those answers, chosen independently, still produce a
schedule whose last window is clamped -- because the thing that has to be true is
a relationship BETWEEN the three, and no single conversion can see it.

    60s / 20s window / 3s overlap  ->  4 chunks, the last striding 7.08s
                                       instead of 17.00s, re-rendering 12.2
                                       seconds a previous chunk already made

So this node solves instead of converting. You say roughly what you want; it
returns the nearest combination that tiles evenly, in FRAMES, ready to wire.

THE ARITHMETIC. Everything lives on two grids: video latents are `5j+2` and
frames are `17j+5`. Write the group counts as

    total  t = 5c+2      window  L = 5a+2      overlap  O = 5b+2

then stride = L-O = 5(a-b), which is a multiple of 5 for ANY a and b -- so every
window keeps grid phase 0 automatically, and the five-window pulse cannot happen.
What is NOT automatic is the tiling:

    total_t - L  ==  5(c-a)     must divide by     stride == 5(a-b)

    i.e.   (c - a) % (a - b) == 0        and then   chunks = (c-a)//(a-b) + 1

That reduction to small integers is why the search below is a plain sweep rather
than anything clever: c is around 85 for a minute of video, so the whole space is
a few thousand pairs.
"""

import json
import logging

from comfy_api.latest import io

from .common import (FPS, FRAME_BASE, FRAMES_PER_GROUP, LATENT_BASE,
                     LATENTS_PER_GROUP, frames_to_latents, latents_to_frames,
                     snap_frames)

MIN_STRIDE_GROUPS = 2     # stride >= 10 latents; below this the chunk count explodes
MAX_CHUNKS = 40


AUDIO_LATENT_HZ = 40


def is_av_exact(frames):
    """Whether a 24 fps frame boundary is whole on H3's 40 Hz audio grid."""
    return (int(frames) * AUDIO_LATENT_HZ) % FPS == 0


def schedule_av_exact(a, b):
    """Both boundaries that carry audio: the carry edge AND the chunk stride.

    The overlap decides where the preserved audio ends inside a chunk; the stride
    decides where chunk i starts on the global clock. Either landing between 40 Hz
    ticks rounds, and video and audio then pin to instants up to a third of a tick
    (8.3 ms) apart. Only every third H3 run is exact -- 39, 90, 141, 192, step 51
    frames -- so most schedules miss on at least one of the two.
    """
    wf = latents_to_frames(from_groups(a))
    of = latents_to_frames(from_groups(b)) if b > 0 else FRAME_BASE
    return is_av_exact(of) and is_av_exact(wf - of)


def groups(latent_t):
    """`j` from a 5j+2 latent count."""
    return (int(latent_t) - LATENT_BASE) // LATENTS_PER_GROUP


def from_groups(j):
    """5j+2 latents back from `j`."""
    return LATENTS_PER_GROUP * int(j) + LATENT_BASE


def seconds_to_groups(seconds):
    """Nearest `j` to a duration, via the frame grid the model actually has."""
    target = float(seconds) * FPS
    j = round((target - FRAME_BASE) / float(FRAMES_PER_GROUP))
    return max(0, int(j))


def frames_to_groups(frames):
    """Nearest `j` to a frame count. The frames variant's ONLY added conversion.

    `seconds_to_groups` multiplies by FPS and then lands here; asking in frames just
    skips that multiply. Everything downstream -- the tiling search, av_align, the
    overlap ladder -- is unit-free, which is why the two nodes share all of it.
    """
    j = round((float(frames) - FRAME_BASE) / float(FRAMES_PER_GROUP))
    return max(0, int(j))


def chunk_count(c, a, b):
    """Chunks for a regular schedule, or None when it does not tile."""
    stride = a - b
    if stride < MIN_STRIDE_GROUPS or a <= b or a > c:
        return None
    if (c - a) % stride:
        return None
    return (c - a) // stride + 1


def solve(c_req, a_req, b_req, prefer, total_leeway=4, want_chunks=0,
          av_align="ignore"):
    """Nearest (c, a, b) that tiles evenly. Returns None when nothing qualifies.

    `prefer` decides what is allowed to move, which is the only real choice here:
    a schedule that tiles always exists if you may shrink the stride far enough,
    so the question is never "is there one" but "which compromise do you want".

    `want_chunks` pins the chunk count, which is usually the number you actually
    hold an opinion about -- it is how many prompts you write and how many joins
    you get -- and it makes the window a RESULT rather than another guess. Fixing
    both c and n leaves a one-parameter family: stride = (c-b)/n must come out
    whole, so only the overlap is still free.
    """
    c_options = [c_req] if prefer == "keep total" else [
        c for c in range(max(1, c_req - total_leeway), c_req + total_leeway + 1)]

    best = None
    for c in c_options:
        for a in range(1, c + 1):
            for b in range(0, a):
                n = chunk_count(c, a, b)
                if n is None or n < 2 or n > MAX_CHUNKS:
                    continue
                if want_chunks and n != want_chunks:
                    continue
                av = schedule_av_exact(a, b)
                if av_align == "require" and not av:
                    continue
                d_total = abs(c - c_req)
                d_window = abs(a - a_req)
                d_overlap = abs(b - b_req)
                # 'prefer' ranks an aligned schedule first but never refuses one
                av_penalty = 0 if (av or av_align != "prefer") else 100
                if prefer == "fewer chunks":
                    # buy a shorter chunk list with window drift, but never by
                    # moving the deliverable length
                    score = (av_penalty, n, d_window + d_overlap, d_total)
                else:
                    score = (av_penalty + 2 * d_total + d_window + d_overlap,
                             n, d_window)
                if best is None or score < best[0]:
                    best = (score, c, a, b, n)
    return None if best is None else best[1:]


def reachable_overlaps(c, n):
    """Every overlap `b` that tiles at this total and chunk count.

    With c and n both fixed the stride is (c-b)/n and must come out whole, so b
    keeps c's residue mod n -- the reachable overlaps sit exactly n GROUPS apart.
    The chunk count IS the overlap's step size, which is why asking for more
    chunks makes the overlap coarser rather than finer."""
    out = []
    for b in range(0, c + 1):
        if (c - b) % n:
            continue
        stride = (c - b) // n
        if stride < MIN_STRIDE_GROUPS or b + stride > c:
            continue
        out.append(b)
    return out


class MMH3ChunkSchedule(io.ComfyNode):
    """Roughly what you want in, a schedule that tiles out."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3ChunkSchedule",
            display_name="MMH3 Chunk Schedule",
            category="MMH3Tools/calculators",
            description=(
                "Solves total / window / overlap together instead of rounding each one "
                "on its own, and emits FRAMES ready for MMH3 Window Plan and the "
                "looping sampler. Independently-rounded values leave the last window "
                "clamped -- striding short and re-rendering seconds a previous chunk "
                "already made -- because what has to hold is a relationship between "
                "the three. Ask for roughly what you want; the report says how far it "
                "moved and why."
            ),
            inputs=[
                io.Float.Input(
                    "total_seconds", default=60.0, min=1.0, max=600.0, step=0.1,
                    tooltip="Roughly how long the finished piece should be. Rounded to "
                            "the 17j+5 frame grid; `prefer` decides whether it may move "
                            "beyond that."),
                io.Float.Input(
                    "window_seconds", default=20.0, min=1.0, max=150.0, step=0.1,
                    tooltip="Roughly how much each chunk covers. Moves to whatever makes "
                            "the schedule tile."),
                io.Float.Input(
                    "overlap_seconds", default=3.0, min=0.0, max=60.0, step=0.1,
                    tooltip="Roughly how much of the previous chunk each chunk carries. "
                            "Moves with the window."),
                io.Combo.Input(
                    "prefer", options=["keep total", "nearest", "fewer chunks"],
                    default="keep total", optional=True,
                    tooltip="What may move. `keep total` holds the length and shifts the "
                            "window and overlap. `nearest` lets the length move a few "
                            "groups to find a closer fit. `fewer chunks` takes the "
                            "shortest chunk list it can, holding the length."),
                io.Int.Input(
                    "chunks", default=0, min=0, max=40, step=1, optional=True,
                    tooltip="How many chunks you want. 0 lets the solver choose from "
                            "the window instead. Setting this makes the WINDOW a result "
                            "rather than a second guess, and it is usually the number "
                            "you have an opinion about: it is how many prompts you "
                            "write and how many joins the piece has."),
                io.Combo.Input(
                    "av_align", options=["ignore", "prefer", "require"],
                    default="ignore", optional=True,
                    tooltip="Whether the overlap and the stride must land whole on H3's "
                            "40 Hz audio grid. Off the grid they round, and the "
                            "preserved audio pins to an instant up to a third of a tick "
                            "(8.3 ms) from the preserved video. `prefer` ranks aligned "
                            "schedules first, `require` returns only aligned ones and "
                            "will move the total to find one. Matters when each chunk "
                            "GENERATES its audio; a supplied master track pinned at "
                            "mask 0 has no per-chunk audio seam to misalign."),
            ],
            outputs=[
                io.Int.Output(display_name="total_frames"),
                io.Int.Output(display_name="window_frames"),
                io.Int.Output(display_name="overlap_frames"),
                io.Int.Output(display_name="chunk_count"),
                # the WINDOW's duration, not the clip's: this is what
                # `seconds_per_chunk` on either scene-plan node consumes, and it
                # pairs with chunk_count above so the writer and the schedule
                # cannot disagree about how long a chunk is
                io.Float.Output(display_name="seconds_per_chunk"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, total_seconds, window_seconds, overlap_seconds,
                prefer="keep total", chunks=0, av_align="ignore") -> io.NodeOutput:
        return solve_and_report(
            seconds_to_groups(total_seconds), seconds_to_groups(window_seconds),
            seconds_to_groups(overlap_seconds), prefer, chunks, av_align,
            "  asked for %.2fs / %.2fs / %.2fs" % (
                total_seconds, window_seconds, overlap_seconds))


class MMH3ChunkScheduleFrames(io.ComfyNode):
    """The same solver, asked in frames."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3ChunkScheduleFrames",
            display_name="MMH3 Chunk Schedule (Frames)",
            category="MMH3Tools/calculators",
            description=(
                "MMH3 Chunk Schedule asked in FRAMES instead of seconds. Identical "
                "solver, identical grid rules, identical outputs -- it just skips the "
                "seconds-to-frames conversion, for when you already hold frame counts "
                "and do not want a duration rounded on the way in.\n\n"
                "Values still SNAP to the 17j+5 grid and are still solved together so "
                "the schedule tiles. Asking in frames does not mean asking for "
                "arbitrary frames."
            ),
            inputs=[
                io.Int.Input(
                    "total_frames", default=1433, min=5, max=14405, step=1,
                    tooltip="Roughly how many frames the finished piece should be. "
                            "Snapped to the nearest 17j+5; `prefer` decides whether it "
                            "may move further than that. Off-grid values are rounded, "
                            "not refused."),
                io.Int.Input(
                    "window_frames", default=481, min=5, max=3605, step=1,
                    tooltip="Roughly how many frames each chunk covers. Moves to "
                            "whatever makes the schedule tile."),
                io.Int.Input(
                    "overlap_frames", default=73, min=0, max=1445, step=1,
                    tooltip="Roughly how many frames of the previous chunk each chunk "
                            "carries. Moves with the window."),
                io.Combo.Input(
                    "prefer", options=["keep total", "nearest", "fewer chunks"],
                    default="keep total", optional=True,
                    tooltip="What may move. `keep total` holds the length and shifts the "
                            "window and overlap. `nearest` lets the length move a few "
                            "groups to find a closer fit. `fewer chunks` takes the "
                            "shortest chunk list it can, holding the length."),
                io.Int.Input(
                    "chunks", default=0, min=0, max=40, step=1, optional=True,
                    tooltip="How many chunks you want. 0 lets the solver choose from "
                            "the window instead. Setting this makes the WINDOW a result "
                            "rather than a second guess, and it is usually the number "
                            "you have an opinion about: it is how many prompts you "
                            "write and how many joins the piece has."),
                io.Combo.Input(
                    "av_align", options=["ignore", "prefer", "require"],
                    default="ignore", optional=True,
                    tooltip="Whether the overlap and the stride must land whole on H3's "
                            "40 Hz audio grid. Off the grid they round, and the "
                            "preserved audio pins to an instant up to a third of a tick "
                            "(8.3 ms) from the preserved video. `prefer` ranks aligned "
                            "schedules first, `require` returns only aligned ones and "
                            "will move the total to find one. Matters when each chunk "
                            "GENERATES its audio; a supplied master track pinned at "
                            "mask 0 has no per-chunk audio seam to misalign."),
            ],
            outputs=[
                io.Int.Output(display_name="total_frames"),
                io.Int.Output(display_name="window_frames"),
                io.Int.Output(display_name="overlap_frames"),
                io.Int.Output(display_name="chunk_count"),
                io.Float.Output(display_name="seconds_per_chunk"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, total_frames, window_frames, overlap_frames,
                prefer="keep total", chunks=0, av_align="ignore") -> io.NodeOutput:
        return solve_and_report(
            frames_to_groups(total_frames), frames_to_groups(window_frames),
            frames_to_groups(overlap_frames), prefer, chunks, av_align,
            "  asked for %d / %d / %d frames" % (
                int(total_frames), int(window_frames), int(overlap_frames)))


def solve_and_report(c_req, a_req, b_req, prefer, chunks, av_align, asked_line):
    """Everything both nodes do once their request is expressed in GROUPS.

    Shared rather than duplicated. `asked_line` is the one genuinely unit-dependent
    thing in the report, so each node supplies its own and the rest is common.
    """
    want = int(chunks)

    notes = []
    # Relax in the order that costs least. Moving the deliverable LENGTH by a
    # group is cheaper than silently changing how many chunks and prompts the
    # piece has, so the chunk count is the LAST thing given up -- an earlier
    # ordering released the count first and then honoured it anyway, so the
    # report claimed a release that had not happened.
    found = solve(c_req, a_req, b_req, prefer, want_chunks=want,
                  av_align=av_align)
    if found is None and av_align == "require":
        # 60s is the worked example: its total is 84 groups, 0 mod 3, and an
        # aligned schedule needs 2 mod 3 -- so NOTHING at that exact length
        # qualifies however the window and overlap move.
        found = solve(c_req, a_req, b_req, "nearest", want_chunks=want,
                      av_align="require")
        if found is not None:
            notes.append("no aligned schedule exists at this exact length, so the "
                         "total was allowed to move -- alignment needs the total's "
                         "group count to be 2 mod 3")
    if found is None and want:
        found = solve(c_req, a_req, b_req, prefer, av_align=av_align)
        if found is not None:
            notes.append("no schedule with exactly %d chunks fits this length and "
                         "overlap, so the count was released and solved from the "
                         "window instead" % want)
    if found is None:
        # never raise: emit the request, snapped, and say it does not tile
        c, a, b = c_req, a_req, b_req
        n = None
        notes.append("no schedule near this request tiles evenly, so the numbers "
                     "below are your request snapped to the grid and NOT solved -- "
                     "expect a clamped final window")
    else:
        c, a, b, n = found

    total_f = snap_frames(latents_to_frames(from_groups(c)))
    window_f = latents_to_frames(from_groups(a))
    overlap_f = latents_to_frames(from_groups(b)) if b > 0 else FRAME_BASE
    stride_f = window_f - overlap_f

    def moved(label, req, got, unit_s=FRAMES_PER_GROUP / float(FPS)):
        if req == got:
            return None
        return "%s %+.2fs (%d group%s)" % (label, (got - req) * unit_s,
                                           got - req, "" if abs(got - req) == 1 else "s")

    moves = [m for m in (moved("total", c_req, c), moved("window", a_req, a),
                         moved("overlap", b_req, b)) if m]

    if b == 0:
        notes.append("overlap solved to 0 -- chunks share nothing, so every join is "
                     "a hard cut")

    lines = ["MMH3 Chunk Schedule -- %s" % ("%d chunks, all regular" % n if n
                                            else "NOT SOLVED"), ""]
    lines.append("  total    %5d frames  %7.2fs   %4d latents"
                 % (total_f, total_f / float(FPS), from_groups(c)))
    lines.append("  window   %5d frames  %7.2fs   %4d latents"
                 % (window_f, window_f / float(FPS), from_groups(a)))
    lines.append("  overlap  %5d frames  %7.2fs   %4d latents"
                 % (overlap_f, overlap_f / float(FPS),
                    from_groups(b) if b > 0 else LATENT_BASE))
    lines.append("  stride   %5d frames  %7.2fs" % (stride_f, stride_f / float(FPS)))
    lines.append("")
    lines.append("  40 Hz audio grid   overlap %-9s stride %s"
                 % ("EXACT" if is_av_exact(overlap_f) else "off by 1/3 tick,",
                    "EXACT" if is_av_exact(stride_f) else "off by 1/3 tick (8.3 ms)"))
    lines.append("")
    lines.append(asked_line + ("  ->  moved " + ", ".join(moves) if moves
                               else "  ->  landed on the grid unchanged"))
    if n:
        lines.append("  tiling   (%d - %d) %% %d = 0"
                     % (from_groups(c), from_groups(a), from_groups(a) - from_groups(b)))
        lines.append("")
        lines.append("  chunks")
        for i in range(n):
            f0 = i * stride_f
            lines.append("    %d: %7.2fs - %7.2fs" % (i, f0 / float(FPS),
                                                      (f0 + window_f) / float(FPS)))
    if n:
        options = reachable_overlaps(c, n)
        if len(options) > 1:
            # a rung is only REACHABLE if its whole schedule qualifies. Marking
            # rungs by the overlap alone offered options the solver would never
            # take: under `require` the STRIDE must land on the grid too, which
            # removes two of every three.
            ok = [o for o in options if schedule_av_exact(o + (c - o) // n, o)]
            here = options.index(b) if b in options else -1
            lo = max(0, here - 2) if here >= 0 else 0
            shown = options[lo:lo + 6]
            lines.append("")
            lines.append("  reachable overlaps at %d chunks -- the COUNT is the "
                         "step (%d group%s = %d latents)"
                         % (n, n, "" if n == 1 else "s", n * LATENTS_PER_GROUP))
            for opt in shown:
                of_f = latents_to_frames(from_groups(opt)) if opt > 0 else FRAME_BASE
                st_f = latents_to_frames(from_groups(opt + (c - opt) // n)) - of_f
                lines.append("    %s %4d latents  %5d frames  %6.2fs   stride %6.2fs %s"
                             % ("->" if opt == b else "  ", from_groups(opt),
                                of_f, of_f / float(FPS), st_f / float(FPS),
                                "  AUDIO-GRID" if opt in ok else ""))
            lines.append("    nothing sits between these; a finer overlap needs a "
                         "different chunk count, or `prefer` off `keep total`.")
            if av_align != "ignore":
                step = (ok[1] - ok[0]) * LATENTS_PER_GROUP if len(ok) > 1 else 0
                lines.append("    av_align=%s, so ONLY the AUDIO-GRID rows are "
                             "reachable -- the stride has to land on the grid too, "
                             "which steps them %d latents apart, not %d."
                             % (av_align, step, n * LATENTS_PER_GROUP))
            elif not ok:
                lines.append("    none of them is on the 40 Hz audio grid at %d "
                             "chunks -- try `av_align` if this clip generates its "
                             "own audio per chunk" % n)

    if notes:
        lines.append("")
        lines.extend("  ! " + x for x in notes)

    report = "\n".join(lines)
    logging.info("[MMH3ChunkSchedule] %s frames=%d/%d/%d",
                 "%d chunks" % n if n else "unsolved", total_f, window_f, overlap_f)
    return io.NodeOutput(int(total_f), int(window_f), int(overlap_f),
                         int(n or 0), float(window_f) / FPS, report)