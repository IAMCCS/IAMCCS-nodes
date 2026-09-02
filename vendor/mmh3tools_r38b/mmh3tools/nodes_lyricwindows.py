"""Slice an aligned lyric by render window, so each chunk knows what is sung in it.

THE JOIN between a song's timeline and a sampler's. `MMH3WindowPlan` decides the
chunks; `MMH3ForcedAlign` decides when each word happens; this puts the two together
so chunk *i* can be prompted with the words actually sung during chunk *i*.

Three things it exists to get right, each of which is silently wrong if hand-rolled:

  * **Window-relative timestamps.** H3 shot times are measured from the start of the
    CHUNK, not the song. A window opening at 70.15s holding a word at 72.40s must
    emit 00:02.250. Absolute time produces prompts H3 cannot act on and nothing
    errors.
  * **`has_lyrics`.** Intros, instrumental breaks and outros are real windows with
    nothing sung in them. They need their own prompt branch -- told that no words
    are sung, rather than left to invent some.
  * **Section context.** A window can straddle a boundary: part chorus, part bridge.
    Uniform windows and musical sections do not divide, so the boundary is passed
    as text rather than pretended away.

Inputs mirror `MMH3SplitAudioToWindows` exactly so both wire from the same
`MMH3WindowPlan` and cannot disagree about which frames window *i* covers.
"""

import json
import logging

from comfy_api.latest import io

from .common import FPS


def _stamp(seconds):
    """MM:SS.mmm, the format H3 shot timestamps use."""
    seconds = max(0.0, float(seconds))
    return "%02d:%06.3f" % (int(seconds // 60), seconds % 60)


def load_alignment(alignment_json):
    """Accept the node's JSON, or a bare list of words, and normalise it."""
    if not (alignment_json or "").strip():
        raise ValueError(
            "MMH3LyricsToWindows: alignment is empty. Wire MMH3 Forced Align's "
            "`alignment_json`, or paste a saved one.")
    try:
        data = json.loads(alignment_json)
    except Exception as exc:
        raise ValueError("MMH3LyricsToWindows: alignment is not valid JSON (%s). "
                         "It wants `alignment_json`, not a report." % exc)
    if isinstance(data, list):                       # a bare whisper_alignment
        return {"words": data, "lines": [], "sections": []}
    for key in ("words", "lines", "sections"):
        data.setdefault(key, [])
    return data


def overlapping(spans, t0, t1):
    """Spans that intersect [t0, t1). Touching at an endpoint does not count."""
    return [s for s in spans
            if float(s["end"]) > t0 and float(s["start"]) < t1]


def sections_for(section_spans, t0, t1):
    """Which sections the window covers, and where a boundary falls inside it.

    Reported in WINDOW-RELATIVE seconds, because that is the only frame of
    reference the prompt for this chunk can act on.
    """
    hits = overlapping(section_spans, t0, t1)
    if not hits:
        return "", []
    if len(hits) == 1:
        return hits[0]["value"], []
    cuts = []
    for s in hits[1:]:
        at = float(s["start"]) - t0
        if 0.0 < at < (t1 - t0):
            cuts.append((s["value"], at))
    return " -> ".join(h["value"] for h in hits), cuts


class MMH3LyricsToWindows(io.ComfyNode):
    """What is sung during render window `index`, verbatim, on the chunk's clock."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3LyricsToWindows",
            display_name="MMH3 Lyrics to Windows",
            category="MMH3Tools/audio",
            description=(
                "Slice an aligned lyric by render window. Emits the lines sung in "
                "window `index` with timestamps rebased to the chunk's own clock, the "
                "neighbouring windows as context, whether anything is sung at all, and "
                "which section(s) the window covers. Inputs mirror MMH3 Split Audio to "
                "Windows so both read the same schedule."
            ),
            inputs=[
                io.String.Input(
                    "alignment_json", multiline=True, default="", force_input=True,
                    tooltip="MMH3 Forced Align's `alignment_json`, or a saved copy. "
                            "Carries words, lines and sections together, which is why "
                            "it is the JSON rather than one `whisper_alignment`."),
                io.Int.Input(
                    "total_frames", default=960, min=1, max=1000000,
                    tooltip="Wire MMH3 Window Plan's `total_frames`."),
                io.Int.Input(
                    "window_frames", default=124, min=1, max=100000,
                    tooltip="Wire MMH3 Window Plan's `window_frames`."),
                io.Int.Input(
                    "overlap_frames", default=22, min=0, max=100000,
                    tooltip="Wire MMH3 Window Plan's `overlap_frames`."),
                io.Combo.Input(
                    "context_schedule", options=["standard_static", "standard_uniform"],
                    default="standard_static",
                    tooltip="Must match the plan, or this slices frames the sampler "
                            "never renders."),
                io.Int.Input(
                    "index", default=0, min=0, max=100000,
                    tooltip="Which window, 0-based. Drive from the loop index."),
                io.Int.Input(
                    "lookaround", default=1, min=0, max=8, optional=True,
                    tooltip="How many windows either side to include as context. 1 is "
                            "the ±1 the beats stage was designed around."),
                io.Boolean.Input(
                    "include_times", default=True, optional=True,
                    tooltip="Prefix each line with its window-relative start, so the "
                            "writer can hang shot timestamps on the words rather than "
                            "inventing them."),
                io.String.Input(
                    "music_json", multiline=True, default="", optional=True,
                    tooltip="MMH3 Music Analysis' `analysis_json`. Adds this window's "
                            "energy and its bar lines. Optional, but an instrumental "
                            "window has no words and no section text, so without it "
                            "there is nothing to tell the prompt whether the music "
                            "here is a soft fall or a drop."),
            ],
            outputs=[
                io.String.Output(display_name="lyrics"),
                io.String.Output(display_name="prev_lyrics"),
                io.String.Output(display_name="next_lyrics"),
                io.Boolean.Output(display_name="has_lyrics"),
                io.String.Output(display_name="section"),
                io.String.Output(display_name="shot_times"),
                io.Int.Output(display_name="window_count"),
                io.Int.Output(display_name="first_frame"),
                io.Int.Output(display_name="last_frame"),
                io.String.Output(display_name="report"),
                io.String.Output(display_name="energy"),
                io.String.Output(display_name="bar_times"),
            ],
        )

    @classmethod
    def execute(cls, alignment_json, total_frames, window_frames, overlap_frames,
                context_schedule, index, lookaround=1, include_times=True,
                music_json="") -> io.NodeOutput:
        from .nodes_windows import _plan, _window_frame_spans

        data = load_alignment(alignment_json)
        words = data["words"]
        lines = data["lines"] or []
        section_spans = data["sections"] or []

        _length, _overlap, total_f, _total_t, windows = _plan(
            total_frames, window_frames, overlap_frames, context_schedule)
        spans = _window_frame_spans(windows, total_f)
        n = len(spans)

        i = max(0, min(int(index), n - 1))
        notes = []
        if int(index) != i:
            notes.append("index %d is outside 0..%d; clamped to %d"
                         % (int(index), n - 1, i))

        def window_text(k, rebase_to=None):
            """Lines sung in window k, timestamped against `rebase_to`'s start."""
            if k < 0 or k >= n:
                return ""
            a, b = spans[k]
            t0, t1 = a / float(FPS), (b + 1) / float(FPS)
            base = (spans[rebase_to][0] / float(FPS)) if rebase_to is not None else t0
            out = []
            for ls in overlapping(lines, t0, t1):
                text = ls["value"]
                if include_times:
                    text = "[%s] %s" % (_stamp(float(ls["start"]) - base), text)
                out.append(text)
            return "\n".join(out)

        a, b = spans[i]
        t0, t1 = a / float(FPS), (b + 1) / float(FPS)
        here = window_text(i)
        # Context is timestamped on THIS window's clock too, so a neighbour's line
        # reads as -00:03.200 rather than as a second, contradictory timeline.
        before = "\n".join(x for x in (window_text(k, rebase_to=i)
                                       for k in range(max(0, i - lookaround), i)) if x)
        after = "\n".join(x for x in (window_text(k, rebase_to=i)
                                      for k in range(i + 1,
                                                     min(n, i + 1 + lookaround))) if x)

        in_window = overlapping(words, t0, t1)
        shot_times = ", ".join("%s %s" % (_stamp(float(w["start"]) - t0), w["value"])
                               for w in in_window)
        section, cuts = sections_for(section_spans, t0, t1)
        if cuts:
            section += " (" + "; ".join("%s begins at %s" % (name, _stamp(at))
                                        for name, at in cuts) + ")"

        has = bool(in_window)
        if not has:
            notes.append("nothing is sung in this window -- it is an intro, an "
                         "instrumental break or an outro, and the prompt for it needs "
                         "the no-lyrics branch rather than invented words")
        if not lines and words:
            notes.append("the alignment has words but no lines, so `lyrics` is empty; "
                         "wire alignment_json rather than a bare words list")
        if not section_spans:
            notes.append("no sections in the alignment; add [Verse 1]/[Chorus] tags to "
                         "the lyrics if you want section context")

        # measured, not guessed: the prompt is TOLD what the music does here
        energy_txt, bar_txt = "", ""
        if (music_json or "").strip():
            try:
                from .nodes_music_analysis import describe_energy, energy_in

                music = json.loads(music_json)
                level, trend = energy_in(music.get("energy") or [], t0, t1)
                energy_txt = describe_energy(level, trend)
                bars = [b for b in (music.get("bars") or [])
                        if t0 <= float(b["beat_s"]) < t1 and b.get("beat_in_bar") == 1]
                bar_txt = ", ".join("%s (bar %d)" % (_stamp(float(b["beat_s"]) - t0),
                                                     b["bar"]) for b in bars)
                if not bars:
                    notes.append("no bar lines fall in this window; check the bpm in "
                                 "the analysis report before cutting to them")
            except Exception as exc:
                notes.append("music_json could not be read (%s); energy and bars are "
                             "empty" % exc)
        elif not has:
            notes.append("no music_json, so this silent window has NOTHING describing "
                         "it -- wire MMH3 Music Analysis or the prompt writes blind")

        report = ("window %d of %d | frames %d-%d | %.2f-%.2fs (%.2fs)\n"
                  "  section : %s\n  lines   : %d here, %d before, %d after\n"
                  "  words   : %d | energy: %s | bar lines: %d\n%s"
                  % (i, n, a, b, t0, t1, t1 - t0, section or "(none)",
                     len(here.splitlines()) if here else 0,
                     len(before.splitlines()) if before else 0,
                     len(after.splitlines()) if after else 0, len(in_window),
                     energy_txt or "(unknown)",
                     len(bar_txt.split(",")) if bar_txt else 0,
                     "\n".join("  ! " + x for x in notes) if notes
                     else "  no warnings"))
        logging.info("[MMH3LyricsToWindows] %s", report.splitlines()[0])
        return io.NodeOutput(here, before, after, has, section, shot_times,
                             n, a, b, report, energy_txt, bar_txt)
