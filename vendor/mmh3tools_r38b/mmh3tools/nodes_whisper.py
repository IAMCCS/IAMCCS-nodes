"""Whisper alignment -> text a prompt writer can read.

ComfyUI-Whisper emits `whisper_alignment`: a list of `{"value", "start", "end"}`,
one entry per word. That is the right shape for burning captions and the wrong shape
for handing lyrics to an LLM, which wants prose with enough timing to know where it
is. This flattens the one into the other.

ADOPTED INTO THIS PACK 2026-08-24. It previously lived in a loose
`ComfyUI-WhisperAlignmentToText` folder that was never published, so the MusicVideo
workflow listed a dependency nobody could install -- asked about in
`minimax_h3_resources` the same day, unanswered. The node id is unchanged so existing
graphs keep working.

Its sibling `WhisperAlignmentToSegments` was NOT adopted: it cuts on 25 fps and a
4n+1 frame grid, which is LTX's, not H3's 24 fps / 17j+5. MMH3 Window Plan and Split
Audio to Windows already do that job on the right grid.
"""

import json

from comfy_api.latest import io

WhisperAlignment = io.Custom("whisper_alignment")

TS_FORMATS = ["[M:SS]", "[SS.s]", "[total_seconds]"]
OUT_FORMATS = ["continuous", "per_line", "per_word"]


def format_timestamp(seconds, fmt):
    if fmt == "[M:SS]":
        return "[%d:%02d]" % (int(seconds // 60), int(seconds % 60))
    if fmt == "[SS.s]":
        return "[%.1fs]" % seconds
    if fmt == "[total_seconds]":
        return "[%d]" % int(seconds)
    return "[%.1f]" % seconds


def build_text(alignment, interval, fmt, output_format):
    """Words -> text, with timestamp markers every `interval` seconds."""
    if output_format == "per_word":
        return "\n".join("%s %s" % (format_timestamp(seg["start"], fmt), seg["value"])
                         for seg in alignment)

    parts = []
    # markers land on interval BOUNDARIES, not on whichever word happened to cross
    # one, so the same song always marks at the same times
    last = -interval if interval > 0 else float("inf")
    for seg in alignment:
        if interval > 0 and seg["start"] >= last + interval:
            marker = (seg["start"] // interval) * interval
            parts.append("\n" + format_timestamp(marker, fmt))
            last = marker
        parts.append(("\n" if output_format == "per_line" else "") + seg["value"])

    if output_format == "continuous":
        text = " ".join(parts).replace(" \n", "\n").replace("\n ", "\n")
    else:
        text = "".join(parts)
    return text.strip()


class WhisperAlignmentToText(io.ComfyNode):
    """Whisper's per-word alignment as prose a prompt writer can use."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WhisperAlignmentToText",
            display_name="Whisper to Text (LLM Ready)",
            category="MMH3Tools/audio",
            description=(
                "Flattens ComfyUI-Whisper's per-word `whisper_alignment` into text a "
                "prompt writer can read, with timestamp markers so it knows where in "
                "the song each line falls. Feed the result to the scene-plan nodes as "
                "lyrics.\n\n"
                "Timestamp markers land on interval BOUNDARIES rather than on whichever "
                "word crossed one, so the same song always marks at the same times."
            ),
            inputs=[
                WhisperAlignment.Input(
                    "alignment",
                    tooltip="Per-word alignment from ComfyUI-Whisper: one entry per "
                            "word with `value`, `start` and `end`."),
                io.Float.Input(
                    "timestamp_interval", default=5.0, min=0.0, max=60.0, step=1.0,
                    optional=True,
                    tooltip="Insert a timestamp marker every N seconds. 0 disables "
                            "them, which is what you want when the text is going "
                            "somewhere that has its own timing."),
                io.Combo.Input(
                    "timestamp_format", options=TS_FORMATS, default="[M:SS]",
                    optional=True,
                    tooltip="[M:SS] reads naturally to an LLM. [SS.s] and "
                            "[total_seconds] are easier to parse mechanically."),
                io.Combo.Input(
                    "output_format", options=OUT_FORMATS, default="continuous",
                    optional=True,
                    tooltip="continuous: flowing prose. per_line: a newline per word "
                            "group, which keeps a lyric sheet looking like one. "
                            "per_word: one word per line with its own timestamp, for "
                            "when you need the timing more than the reading."),
                io.Boolean.Input(
                    "include_timing_data", default=False, optional=True,
                    tooltip="Also emit the raw spans as JSON on the second output. Off "
                            "returns '[]' rather than nothing, so a wired downstream "
                            "node still parses."),
            ],
            outputs=[
                io.String.Output(display_name="text"),
                io.String.Output(display_name="timing_data"),
                io.Float.Output(display_name="duration"),
            ],
        )

    @classmethod
    def execute(cls, alignment, timestamp_interval=5.0, timestamp_format="[M:SS]",
                output_format="continuous", include_timing_data=False) -> io.NodeOutput:
        if not alignment:
            # an empty alignment is a real outcome (silence, a failed transcribe), so
            # it returns empty rather than raising -- but 0.0 duration downstream is
            # the tell, since a wired graph would otherwise plan a zero-length song
            return io.NodeOutput("", "[]", 0.0)

        duration = max(float(seg["end"]) for seg in alignment)
        text = build_text(alignment, float(timestamp_interval), timestamp_format,
                          output_format)
        timing = "[]"
        if include_timing_data:
            timing = json.dumps(
                [{"start": s["start"], "end": s["end"], "text": s["value"]}
                 for s in alignment], indent=2)
        return io.NodeOutput(text, timing, float(duration))
