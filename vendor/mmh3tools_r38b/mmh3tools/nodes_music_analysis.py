"""Deterministic facts about a song, for windows whose boundaries are already fixed.

Ported from music-director's `music.py`, minus the parts that only make sense if you
get to CHOOSE where scenes begin. Its cut-salience blend and agglomerative
segmentation both exist to place boundaries; the looping sampler renders uniform,
grid-locked windows, so there is nothing here for a boundary-placing signal to do.

What remains is what still helps INSIDE a window someone else decided:

  * **The bar grid.** Landing a shot change on a bar is what makes a cut feel
    musical, and bars fall where they fall no matter how the window was sized.
    These are cut candidates alongside the word onsets from forced alignment.
  * **RMS energy.** Level and direction. An instrumental window has no words and no
    section text -- energy is the only thing that distinguishes a soft fall from a
    drop, and without it the prompt for that window is written blind.
  * **BPM, key and mode.** One line of film-wide context.

Nothing here is a judgement. Every value is measured, which is why it belongs in a
node rather than in a prompt: the model is told what the music does, not asked to
guess it.
"""

import json
import logging

from comfy_api.latest import io

ANALYSIS_SR = 22050          # what music-director analyses at
ENERGY_HZ = 10               # hop = sr // 10, so frames land exactly 0.1s apart

_KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Krumhansl-Schmuckler profiles, the standard pair
_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def estimate_key(chroma_mean):
    """Best (key, mode) by correlating the chroma profile against both templates."""
    import numpy as np

    best, best_r = ("C", "major"), -2.0
    for i in range(12):
        for name, profile in (("major", _MAJOR), ("minor", _MINOR)):
            rolled = np.roll(chroma_mean, -i)
            r = float(np.corrcoef(rolled, profile)[0, 1])
            if r > best_r:
                best_r, best = r, (_KEYS[i], name)
    return best


def analyse(samples, sr):
    """BPM, key/mode, a bar grid and a 10 Hz energy curve. All measured."""
    import librosa
    import numpy as np

    duration = float(len(samples)) / sr

    tempo, beat_frames = librosa.beat.beat_track(y=samples, sr=sr, units="frames")
    bpm = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    # 4/4 assumed, as music-director does: bar number and position come from the
    # beat index, which is right for the overwhelming majority of songs and wrong
    # in a way that is obvious rather than subtle.
    bars = [{"beat_s": round(float(t), 3),
             "bar": i // 4 + 1, "beat_in_bar": i % 4 + 1}
            for i, t in enumerate(beat_times)]

    hop = max(1, sr // ENERGY_HZ)
    rms = librosa.feature.rms(y=samples, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    energy = [{"t_s": round(float(t), 2), "rms": round(float(r), 6)}
              for t, r in zip(times, rms)]

    chroma = librosa.feature.chroma_cqt(y=samples, sr=sr)
    key, mode = estimate_key(chroma.mean(axis=1))

    return {"bpm": round(bpm, 1), "key": key, "mode": mode,
            "duration_s": round(duration, 2), "bars": bars, "energy": energy}


def energy_in(energy, t0, t1):
    """(mean level 0-1 against the track, trend) over a span, or (None, "")."""
    import numpy as np

    if not energy:
        return None, ""
    vals = np.array([e["rms"] for e in energy], dtype="float64")
    times = np.array([e["t_s"] for e in energy], dtype="float64")
    sel = (times >= t0) & (times < t1)
    if not sel.any():
        return None, ""
    span = vals[sel]
    lo, hi = float(vals.min()), float(vals.max())
    level = (float(span.mean()) - lo) / max(hi - lo, 1e-9)
    # first half against second half: enough to say rising or falling without
    # pretending to a precision the 10 Hz curve does not have
    half = max(1, len(span) // 2)
    a, b = float(span[:half].mean()), float(span[half:].mean() if len(span) > half
                                            else span[half - 1])
    ratio = b / max(a, 1e-9)
    trend = "rising" if ratio > 1.25 else "falling" if ratio < 0.8 else "steady"
    return level, trend


def describe_energy(level, trend):
    """Plain words, because the prompt reads this rather than plotting it."""
    if level is None:
        return ""
    band = ("near-silent" if level < 0.12 else "quiet" if level < 0.3 else
            "moderate" if level < 0.55 else "loud" if level < 0.8 else "peak")
    return "%s and %s" % (band, trend)


class MMH3MusicAnalysis(io.ComfyNode):
    """Measured facts about the track: bpm, key, bars, energy."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3MusicAnalysis",
            display_name="MMH3 Music Analysis",
            category="MMH3Tools/audio",
            description=(
                "Librosa analysis of a song: BPM, key and mode, a 4/4 bar grid, and a "
                "10 Hz RMS energy curve. Feeds MMH3 Lyrics to Windows so each window "
                "knows how loud it is and where its bars fall. Deliberately omits cut "
                "detection -- that exists to place boundaries, and the looping "
                "sampler's windows are uniform and already fixed."
            ),
            inputs=[
                io.Audio.Input(
                    "audio",
                    tooltip="The FULL MIX, not the vocal stem. Bars and energy come "
                            "from the whole arrangement; a stem has neither drums nor "
                            "the loudness the picture should react to."),
                io.Int.Input(
                    "beats_per_bar", default=4, min=1, max=16, optional=True,
                    tooltip="Assumed time signature. 4 is right for nearly everything "
                            "and wrong in a way you will notice immediately."),
            ],
            outputs=[
                io.String.Output(display_name="analysis_json"),
                io.Float.Output(display_name="bpm"),
                io.String.Output(display_name="key"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, audio, beats_per_bar=4) -> io.NodeOutput:
        import torch
        import torchaudio

        wav = audio["waveform"]
        sr = int(audio["sample_rate"])
        if wav.ndim == 3:
            wav = wav[0]
        if wav.ndim == 2 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.reshape(-1).to(torch.float32)
        if sr != ANALYSIS_SR:
            wav = torchaudio.functional.resample(wav, sr, ANALYSIS_SR)
        samples = wav.cpu().numpy()

        data = analyse(samples, ANALYSIS_SR)
        if beats_per_bar != 4:
            for i, b in enumerate(data["bars"]):
                b["bar"] = i // beats_per_bar + 1
                b["beat_in_bar"] = i % beats_per_bar + 1
        data["beats_per_bar"] = int(beats_per_bar)

        notes = []
        if data["bpm"] < 40 or data["bpm"] > 200:
            notes.append("bpm %.1f is outside the range beat tracking is reliable in; "
                         "the bar grid is probably wrong" % data["bpm"])
        if len(data["bars"]) < 8:
            notes.append("only %d beats found -- too sparse or too short to track"
                         % len(data["bars"]))

        report = ("%.1f BPM, %s %s, %d beats over %.2fs (%d bars of %d)\n%s"
                  % (data["bpm"], data["key"], data["mode"], len(data["bars"]),
                     data["duration_s"],
                     (len(data["bars"]) + beats_per_bar - 1) // beats_per_bar,
                     beats_per_bar,
                     "\n".join("  ! " + x for x in notes) if notes
                     else "  no warnings"))
        logging.info("[MMH3MusicAnalysis] %s", report.splitlines()[0])
        return io.NodeOutput(json.dumps(data, ensure_ascii=False),
                             data["bpm"], "%s %s" % (data["key"], data["mode"]),
                             report)
