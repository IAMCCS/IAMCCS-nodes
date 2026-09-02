"""Forced alignment of KNOWN lyrics against isolated vocals.

FORCED ALIGNMENT IS NOT TRANSCRIPTION, and the difference is the whole point.

A transcriber listens and guesses what was sung. On singing it guesses badly, and
everything built downstream inherits the mistake -- prompts describing words nobody
sang, typography quoting a mishearing. An aligner is handed the true words and solves
only for WHEN each one occurs. It cannot mishear, because it is not listening for
what.

So the lyrics input is authoritative. This node never invents, corrects or reorders a
word; it raises if the returned word sequence differs from what went in.

FEED IT ISOLATED VOCALS. Alignment against a full mix degrades badly -- the model
tries to find words under drums. Any separator works (MelBandRoFormer, a stem from
your DAW); nothing here depends on a particular one.

Outputs the `whisper_alignment` type that ComfyUI-Whisper emits, so the existing
`Whisper -> Text` / `Whisper -> Segments` nodes consume it unchanged, plus a JSON
string so a song can be aligned once and reloaded instead of paying for a 3 GB model
on every run.
"""

import json
import logging
import math
import re

from comfy_api.latest import io

WhisperAlignment = io.Custom("whisper_alignment")

MODEL_SIZES = ["large-v3", "large-v2", "large-v3-turbo", "medium", "small", "base"]
SAMPLE_RATE = 16000          # what Whisper's frontend expects
_MODEL_CACHE = {}

# Square AND curly. Suno emits both -- `{bridge}` reached a real run as a sung word
# and took its whole section with it, because the section tag was never recognised
# and its lines were absorbed by the chorus above. The word-sequence contract cannot
# catch that: the tag is a "word" on both sides of alignment, so both sides agree.
_BRACKET = re.compile(r"[\[\{]([^\]\}]*)[\]\}]")
_OPENERS = "[{"
_PAREN = re.compile(r"\(([^)]*)\)")

# Suno emits TWO kinds of bracketed tag and they must not be treated alike:
#   structural  [Verse 1] [Chorus] [Guitar Solo]   -> a section boundary
#   direction   [soft piano builds] [whispered]    -> a production note, no section
# Matched on whole words, so "Guitar Solo" is structural via `solo` while
# "soft piano builds" is not ("builds" is not "build", and neither is listed).
_STRUCTURAL = {
    "intro", "verse", "chorus", "pre", "post", "hook", "refrain", "bridge",
    "breakdown", "break", "interlude", "instrumental", "solo", "outro", "coda",
    "vamp", "drop", "adlib", "adlibs",
}


def _is_structural(tag):
    return any(w in _STRUCTURAL for w in re.findall(r"[a-z]+", tag.lower()))


def parse_lyrics(text, strip_parentheticals=False):
    """Split lyrics into alignable lines, section boundaries, and what was removed.

    Returns ``(lines, sections, notes)``. `lines` are the sung lines with every
    bracketed tag stripped; each section is ``{"name", "first_line", "last_line"}``
    indexing into `lines`.

    Tags never reach the aligner -- nobody sings "Verse 1" -- but their POSITION is
    what makes section awareness possible later, so it is kept.

    Pasting straight from Suno breaks two naive assumptions, so both are handled:
    a tag can share a line with sung words (``[Chorus] Nobody came``), and most
    bracketed text is direction rather than structure.

    Parentheticals are KEPT by default: in Suno output ``(ooh)`` is usually a
    backing vocal that is genuinely in the audio, and removing it would leave the
    aligner with fewer words than it can hear.
    """
    lines, sections, directions, parens = [], [], [], []

    def open_section(name):
        if sections:
            sections[-1]["last_line"] = len(lines) - 1
        sections.append({"name": name, "first_line": len(lines), "last_line": None})

    for raw in (text or "").splitlines():
        s = raw.strip()
        if not s:
            continue
        tags = _BRACKET.findall(s)
        rest = _BRACKET.sub(" ", s)
        # An UNCLOSED bracket survives the regex, and a truncated paste produces
        # them ("[Chorus" with no closer). Left alone it reaches the aligner as a
        # lyric and gets timed as a sung word, silently. Brackets are tags here, so
        # take the rest of the line with it.
        cut = min((rest.index(c) for c in _OPENERS if c in rest), default=-1)
        if cut >= 0:
            head, tail = rest[:cut], rest[cut + 1:]
            if tail.strip():
                tags.append(tail.strip())
            rest = head
        if strip_parentheticals:
            parens.extend(p.strip() for p in _PAREN.findall(rest) if p.strip())
            rest = _PAREN.sub(" ", rest)
        rest = " ".join(rest.split())
        # tags first: a leading [Chorus] opens the section that THIS line starts
        for t in tags:
            t = t.strip()
            if not t:
                continue
            if _is_structural(t):
                open_section(t)
            else:
                directions.append(t)
        if rest:
            lines.append(rest)

    if sections:
        sections[-1]["last_line"] = len(lines) - 1

    # A section with no sung line under it -- [Guitar Solo], [Instrumental] -- cannot
    # be timed from words, so it cannot become a span. Report it rather than let it
    # disappear: those are exactly the windows that need the no-lyrics prompt branch.
    # `is not None`, NOT truthiness: last_line 0 is a real section ending on the
    # first line, and `or -1` silently deletes it.
    empty = [s["name"] for s in sections
             if s["last_line"] is None or s["first_line"] > s["last_line"]]
    sections = [s for s in sections
                if s["last_line"] is not None and s["first_line"] <= s["last_line"]]
    return lines, sections, {"directions": directions, "empty_sections": empty,
                             "parentheticals": parens}


def _words_of(text):
    """Comparable word sequence: what must survive alignment unchanged."""
    return re.findall(r"[^\W_]+", (text or "").lower(), re.UNICODE)


def _envelope(samples, sr, hop=0.05):
    """Coarse RMS envelope, plus the levels that count as silent and as voiced.

    Adaptive rather than absolute: a quiet stem and a hot one both have a floor
    and a singing level, and what matters is which of the two a span sits nearer.
    """
    import numpy as np

    n = max(1, int(sr * hop))
    trimmed = samples[:len(samples) - len(samples) % n]
    if trimmed.size < n:
        return None
    env = np.sqrt((trimmed.reshape(-1, n).astype("float64") ** 2).mean(axis=1))
    # Threshold as a FRACTION OF THE SINGING LEVEL, not as a measured noise floor.
    # A percentile floor only works if enough of the track is silent: on a stem that
    # is 86% vocal, the 20th percentile IS a voiced level, and every span looks
    # quiet by comparison. A fraction of the loud end holds either way.
    voiced = float(np.percentile(env, 90))
    return env, hop, max(0.10 * voiced, 1e-6)


def _level(env_pack, t0, t1):
    """Mean envelope level over [t0, t1), or None if the span is off the end."""
    import numpy as np

    env, hop, _ = env_pack
    a, b = int(max(0.0, t0) / hop), int(max(0.0, t1) / hop) + 1
    a, b = min(a, len(env)), min(max(b, a + 1), len(env))
    if a >= len(env):
        return None
    return float(np.mean(env[a:b]))


def group_into_lines(raw_words, lines):
    """Assign aligned words back to the lyric's own lines, by consuming them in order.

    Removes any dependence on stable-ts's segment grouping. `original_split=True`
    asks the aligner to honour the line breaks, but it also changes the token
    windows it aligns over -- and music-director, which aligns the same songs
    correctly, does not pass it. The lyric already tells us how many words each
    line holds, and the word sequence is guaranteed to match, so the grouping is
    ours to compute rather than a parameter to request.
    """
    groups, i = [], 0
    for line in lines:
        want = len(_words_of(line))
        got, taken = 0, []
        while i < len(raw_words) and got < want:
            taken.append(raw_words[i])
            got += len(_words_of(raw_words[i].word))
            i += 1
        groups.append(taken)
    # anything left over belongs to the last line rather than being dropped
    if i < len(raw_words) and groups:
        groups[-1].extend(raw_words[i:])
    return groups


def snap_onsets(raw_words, env_pack, min_gap=5.0, hold=3):
    """Pull a late word back to where audio RESUMES after the word before it.

    For a glitched or stuttered refrain the audio holds one word many times while
    the lyric holds it once. Forced alignment must choose a single instant and
    chooses the last clean utterance, so the word is timed at the END of a passage
    it occupies entirely. For typography and lipsync you want the onset.

    This overrides the model with energy, so it is opt-in and only touches words
    the diagnostic already flagged. It only ever moves a word EARLIER, never past
    the previous word, and never changes which word is which.
    """
    env, hop, thr = env_pack
    moved = []
    for i in range(1, len(raw_words)):
        prev_end, start = float(raw_words[i - 1].end), float(raw_words[i].start)
        if start - prev_end <= min_gap:
            continue
        a, b = int(prev_end / hop) + 1, min(int(start / hop), len(env))
        onset = None
        for k in range(a, b):
            # `hold` consecutive loud frames, not one: a single spike is a click or
            # a stray transient, and snapping to it would be worse than not snapping
            if all(env[j] >= thr for j in range(k, min(k + hold, b))):
                onset = k * hop
                break
        # A move of a few frames is the aligner already being right to within the
        # envelope's resolution. Only report a snap that actually relocates a word.
        if onset is None or start - onset < 0.25:
            continue
        moved.append((raw_words[i].word.strip(), start, onset))
        raw_words[i].start = onset
    return moved


def diagnose(words, line_spans, section_spans, duration, samples=None,
             sample_rate=SAMPLE_RATE, snapped=()):
    """Name the ways forced alignment goes wrong WITHOUT dropping a word.

    Every failure seen on real songs has the same shape: a silence the aligner
    cannot skip, so it stretches the words around it instead. That produces three
    signatures, and only the first is visible as a plain gap.

    A gap ON a section boundary is an instrumental break and is correct. A gap
    INSIDE a line is a misalignment -- that line's timings are wrong.
    """
    notes = []
    if not words:
        return notes

    starts = {round(s["start"], 2) for s in section_spans}

    def on_boundary(t):
        return any(abs(t - s) < 0.05 for s in starts)

    env_pack = None
    if samples is not None and len(samples):
        env_pack = _envelope(samples, sample_rate)

    def quiet(t0, t1):
        """True/False if the audio can answer, None if it cannot."""
        if env_pack is None:
            return None
        lvl = _level(env_pack, t0, t1)
        # bool(), not the bare comparison: numpy returns np.bool_, and
        # `np.bool_(True) is True` is False -- every verdict would fall through the
        # identity checks below and silently report the un-evidenced fallback.
        return None if lvl is None else bool(lvl < env_pack[2])

    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        if gap <= 5.0:
            continue
        w = words[i]["value"]
        silent = quiet(words[i - 1]["end"], words[i]["start"])
        # The audio settles it. A gap over SILENCE is the aligner correctly skipping
        # a passage with no vocal in it; a gap over AUDIBLE vocal means it skipped
        # singing, which is the actual error.
        if silent is True:
            notes.append("%.1fs before %r is SILENT in the stem -- correctly skipped, "
                         "not an error" % (gap, w))
        elif silent is False:
            notes.append("%.1fs before %r HAS AUDIO in it -- the aligner skipped a "
                         "stretch that is not silent. If that is singing, these "
                         "timings are wrong: raise nonspeech_skip above %.0fs so it "
                         "stops jumping the gap. (On a separated stem the sound could "
                         "also be bleed or a wordless ad-lib, which is harmless.)"
                         % (gap, w, math.ceil(gap)))
        elif on_boundary(words[i]["start"]):
            notes.append("%.1fs instrumental before %r -- lands on a section start, "
                         "so probably a break" % (gap, w))
        else:
            notes.append("%.1fs gap before %r MID-LINE -- probably a misalignment"
                         % (gap, w))

    # Words placed where the stem is silent cannot have been sung there. This is the
    # signature that separates a real misplacement from a musical pause: the pause
    # leaves words on audio, the misplacement strands them on nothing.
    if env_pack is not None:
        stranded = [w["value"] for w in words
                    if quiet(w["start"], max(w["end"], w["start"] + 0.05)) is True]
        if stranded:
            notes.append("%d word%s placed on SILENCE (%s) -- misaligned; nothing was "
                         "sung where they sit"
                         % (len(stranded), "" if len(stranded) == 1 else "s",
                            ", ".join(repr(x) for x in stranded[:6])
                            + (" ..." if len(stranded) > 6 else "")))

    # A line whose duration is wildly out of step with an identical line elsewhere is
    # the clearest smear signal a song gives you, because choruses repeat.
    by_text = {}
    for ls in line_spans:
        by_text.setdefault(ls["value"], []).append(ls["end"] - ls["start"])
    # A snapped word legitimately spans its whole delivery, so the line holding it
    # is SUPPOSED to be long. Comparing it against a clean instance would report the
    # fix as the fault.
    snap_times = [t for _, _, t in snapped]
    explained = {ls["value"] for ls in line_spans
                 for t in snap_times if ls["start"] <= t <= ls["end"]}
    for text, durs in by_text.items():
        if len(durs) < 2 or text in explained:
            continue
        lo, hi = min(durs), max(durs)
        if lo > 0 and hi > lo * 2.5:
            notes.append("%r takes %.1fs in one place and %.1fs in another -- the "
                         "long one is stretched across a gap" % (text, lo, hi))

    zero = [w["value"] for w in words if w["end"] <= w["start"]]
    if zero:
        notes.append("%d zero-length word%s (%s) -- they carry no timing and usually "
                     "sit beside a stretched one"
                     % (len(zero), "" if len(zero) == 1 else "s",
                        ", ".join(repr(z) for z in zero[:4])))

    if words[-1]["end"] > duration + 0.5:
        notes.append("last word ends %.2fs past the audio (%.2fs)"
                     % (words[-1]["end"], duration))

    # One value clears every skipped stretch at once. Chasing them one at a time just
    # moves the threshold past the smallest gap and re-reports the rest next run.
    skipped = [words[i]["start"] - words[i - 1]["end"] for i in range(1, len(words))
               if words[i]["start"] - words[i - 1]["end"] > 5.0]
    if skipped:
        notes.append("BEFORE tuning anything: check whether the audio REPEATS a line "
                     "that the lyrics hold only once. Suno stutters refrains and "
                     "doubles hooks, and one copy of a line against three utterances "
                     "produces exactly these gaps. Write the line as many times as it "
                     "is sung -- no setting here can substitute for that.")
    if len(skipped) > 1:
        notes.append("=> %d stretches skipped in total; nonspeech_skip %d clears them "
                     "ALL in one run (0 never skips)"
                     % (len(skipped), math.ceil(max(skipped)) + 1))
    return notes


def sections_to_spans(sections, line_spans):
    """Turn section tag positions into (start, end) using the aligned lines."""
    out = []
    for s in sections:
        lo, hi = s["first_line"], s["last_line"]
        if lo >= len(line_spans) or hi < lo:
            continue
        hi = min(hi, len(line_spans) - 1)
        out.append({"value": s["name"],
                    "start": line_spans[lo]["start"],
                    "end": line_spans[hi]["end"]})
    return out


def _resolve_download_root():
    """Where openai-whisper should look for / place its .pt checkpoints.

    Mirrors ComfyUI-Whisper's key order so both nodes agree on one location, rather
    than each downloading its own copy of a 3 GB checkpoint. None falls back to
    openai-whisper's own cache, which is where any non-ComfyUI install of it (a
    command-line run, another project) will already have put the weights.
    """
    try:
        import folder_paths
    except Exception:
        return None
    for key in ("stt_whisper", "whisper"):
        try:
            paths = folder_paths.get_folder_paths(key)
            if paths:
                return paths[0]
        except Exception:
            pass
    return None


def write_temp_wav(audio):
    """Write the AUDIO input to a temp wav at its NATIVE rate, and return the path.

    stable-ts is handed a path, not an array, so that IT does the decode and the
    resample to Whisper's 16 kHz -- exactly as music-director does, which aligns
    these songs correctly. Doing the conversion here instead means a different
    resampler and a different mono fold on the way in, and this node is not the
    place to be second-guessing the aligner's front end.
    """
    import os
    import tempfile

    import torch

    wav = audio["waveform"]
    if wav.ndim == 3:                       # [B, C, T] -- align one take, not a batch
        wav = wav[0]
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    try:
        import folder_paths
        root = folder_paths.get_temp_directory()
        os.makedirs(root, exist_ok=True)
    except Exception:
        root = tempfile.gettempdir()
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="mmh3_align_", dir=root)
    os.close(fd)
    # soundfile, NOT torchaudio.save: torchaudio routes through torchcodec, which
    # cannot load its ffmpeg DLLs in this environment. The ffmpeg BINARY is present
    # and is what stable-ts uses to read the file back, so only the writer needed
    # replacing. Channels-last is soundfile's layout.
    import soundfile as sf

    data = wav.to(torch.float32).cpu().numpy().T
    sf.write(path, data, int(audio["sample_rate"]), subtype="FLOAT")
    return path


def _to_mono_16k(audio):
    """ComfyUI AUDIO -> float32 mono numpy at 16 kHz."""
    import torch
    import torchaudio

    wav = audio["waveform"]
    sr = int(audio["sample_rate"])
    if wav.ndim == 3:                      # [B, C, T] -- align one take, not a batch
        wav = wav[0]
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.reshape(-1).to(torch.float32)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav.cpu().numpy(), float(wav.shape[-1]) / SAMPLE_RATE


def _vram_used_mb():
    """Allocated CUDA memory in MB, or None when there is no CUDA to measure."""
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.memory_allocated() / (1024.0 ** 2)


def release_model(key):
    """Actually get large-v3 off the GPU. Returns MB freed, or None.

    `del` alone is not enough, and this was wrong for several runs. Three things
    have to happen, in order:

      * ``.cpu()`` FIRST -- moves the weights off the device. Without it, anything
        still holding a reference keeps ~6 GB of fp32 resident no matter what is
        deleted afterwards.
      * ``gc.collect()`` -- torch modules hold parent/child reference cycles, so
        refcounting does not free them at the point of the last ``del``.
      * ``empty_cache()`` -- returns the freed blocks to the driver. Do it last;
        before the collect it has nothing to return.

    music-director's `release_model()` does the first and third; the collect is the
    part a long-lived ComfyUI process needs and a short-lived script does not.
    """
    import gc

    import torch

    before = _vram_used_mb()
    model = _MODEL_CACHE.pop(key, None)
    if model is not None:
        try:
            model.cpu()
        except Exception as exc:                      # never fail a good alignment
            logging.warning("[MMH3ForcedAlign] could not move model to CPU: %s", exc)
        del model
    gc.collect()
    try:
        import comfy.model_management as mm
        mm.soft_empty_cache()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    after = _vram_used_mb()
    return None if before is None or after is None else max(0.0, before - after)


class MMH3ForcedAlign(io.ComfyNode):
    """Place known lyrics on the timeline. Never guesses at the words."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3ForcedAlign",
            display_name="MMH3 Forced Align (Lyrics)",
            category="MMH3Tools/audio",
            description=(
                "Forced-align KNOWN lyrics against isolated vocals: the words are "
                "given, only their timing is solved. Not transcription -- it cannot "
                "mishear. Emits the same `whisper_alignment` type ComfyUI-Whisper "
                "does, plus JSON so a song is aligned once and reloaded."
            ),
            inputs=[
                io.Audio.Input(
                    "audio",
                    tooltip="ISOLATED VOCALS. Aligning against a full mix degrades "
                            "badly -- the model hunts for words under drums. Any "
                            "separator will do; nothing here depends on one."),
                io.String.Input(
                    "lyrics", multiline=True, default="",
                    tooltip="The lyrics AS PERFORMED, not as prompted. Lines that are "
                            "only a bracketed tag ([Verse 1], {bridge}) are kept as "
                            "section boundaries and never sent to the aligner.\n\n"
                            "THE ONE THING THAT MATTERS: forced alignment assumes the "
                            "text and the audio hold the same words the same number of "
                            "times. Suno takes liberties -- it repeats lines, stutters "
                            "a refrain, adds ad-libs. If a line is sung three times, "
                            "WRITE IT THREE TIMES. One copy against three utterances "
                            "leaves the aligner to pick one and strand the rest, which "
                            "shows up as large gaps, stretched words and whole sections "
                            "landing early. No parameter fixes that; correcting the "
                            "lyrics does, and it makes the alignment easy."),
                io.Combo.Input(
                    "model_size", options=MODEL_SIZES, default="large-v3",
                    tooltip="large-v3 is what music-director validated on singing. "
                            "Smaller models align faster and drift more on melisma."),
                io.String.Input(
                    "language", default="en",
                    tooltip="ISO 639-1 code. Alignment still needs to know the "
                            "language even though it is not choosing the words."),
                io.Combo.Input(
                    "device", options=["auto", "cuda", "cpu"], default="auto",
                    optional=True,
                    tooltip="cpu is slow but leaves VRAM alone if something else "
                            "is resident."),
                io.Float.Input(
                    "nonspeech_skip", default=5.0, min=0.0, max=120.0, step=0.5,
                    optional=True,
                    tooltip="Skip stretches of non-speech at least this long instead "
                            "of aligning through them. THE lever when the report says "
                            "a gap HAS AUDIO: the VAD has marked real singing as "
                            "non-speech, jumped it, and pushed the words that belonged "
                            "inside out to the far side. Raise it above the reported "
                            "gap, or set 0 to never skip. Lower only if genuine "
                            "instrumental passages are being aligned through."),
                io.Float.Input(
                    "max_word_dur", default=3.0, min=0.0, max=30.0, step=0.1,
                    optional=True,
                    tooltip="Re-align any word that ends up longer than this. THE "
                            "lever for smearing: when a silence sits mid-line the "
                            "aligner stretches the words around it, and stable-ts's "
                            "3.0s default is too loose to catch a 2.6s word. Try "
                            "1.5 when the report flags a mid-line gap. 0 disables."),
                io.Boolean.Input(
                    "snap_to_onset", default=False, optional=True,
                    tooltip="Pull a late word back to where audio RESUMES after the "
                            "word before it. For a glitched or stuttered refrain the "
                            "audio holds one word many times while the lyric holds it "
                            "once; alignment must pick one instant and picks the LAST "
                            "utterance, so the word is timed at the end of a passage "
                            "it occupies entirely. Typography and lipsync want the "
                            "onset. Overrides the model with energy, so it is off by "
                            "default and only touches words the report already "
                            "flagged. Never moves a word later, or past its "
                            "predecessor."),
                io.Boolean.Input(
                    "strip_parentheticals", default=False, optional=True,
                    tooltip="Remove (parenthesised) text before aligning. OFF by "
                            "default because Suno's `(ooh)` is usually a backing "
                            "vocal that IS in the audio -- stripping it leaves the "
                            "aligner with fewer words than it can hear. Turn it on "
                            "when the parentheses are stage directions."),
                io.Boolean.Input(
                    "vad", default=False, optional=True,
                    tooltip="Use Silero VAD to mark where SPEECH actually is, instead "
                            "of letting the aligner place words anywhere. This is the "
                            "fix when whole lines land on an instrumental passage: on "
                            "a separated stem, bleed is loud but is not a voice, and "
                            "energy alone cannot tell them apart. Downloads a small "
                            "model via torch.hub on first use.\n\n"
                            "OBSERVED 2026-08-14: on a heavily produced SUNG vocal "
                            "this made alignment far worse -- 131 of 190 words came "
                            "back zero-length, because Silero is trained on speech "
                            "and does not fire on singing, so nearly the whole track "
                            "was masked as non-speech. Useful for spoken word; try "
                            "a much lower vad_threshold before trusting it on song."),
                io.Float.Input(
                    "vad_threshold", default=0.35, min=0.05, max=0.95, step=0.05,
                    optional=True,
                    tooltip="Speech confidence Silero must reach. LOWER is stricter "
                            "about calling something silence, so raise it if bleed is "
                            "still being treated as voice. Ignored unless vad is on."),
                io.Boolean.Input(
                    "unload_after", default=True, optional=True,
                    tooltip="Free the model when alignment finishes. large-v3 is "
                            "~3 GB, and this runs before sampling wants the card. "
                            "Off keeps it cached for repeated runs on one song."),
            ],
            outputs=[
                WhisperAlignment.Output(display_name="words_alignment"),
                WhisperAlignment.Output(display_name="lines_alignment"),
                WhisperAlignment.Output(display_name="sections_alignment"),
                io.String.Output(display_name="alignment_json"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, audio, lyrics, model_size, language, device="auto",
                nonspeech_skip=5.0, max_word_dur=3.0, snap_to_onset=False, vad=False, vad_threshold=0.35, strip_parentheticals=False,
                unload_after=True) -> io.NodeOutput:
        import torch

        lines, sections, stripped = parse_lyrics(lyrics, strip_parentheticals)
        if not lines:
            raise ValueError(
                "MMH3ForcedAlign: no lyric text. Every line was empty or a bracketed "
                "tag, and there is nothing to place on the timeline.")
        clean = "\n".join(lines)
        expected = _words_of(clean)

        # 16 kHz mono is built ONLY for the energy diagnostic; the aligner gets the
        # file and does its own front end.
        samples, duration = _to_mono_16k(audio)
        audio_path = write_temp_wav(audio)

        dev = device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"

        key = (model_size, dev)
        if key not in _MODEL_CACHE:
            import stable_whisper
            root = _resolve_download_root()
            logging.info("[MMH3ForcedAlign] loading %s on %s (download_root=%s)",
                         model_size, dev, root or "whisper default cache")
            _MODEL_CACHE[key] = stable_whisper.load_model(
                model_size, device=dev, download_root=root)
        model = _MODEL_CACHE[key]

        # Deliberately the same call music-director makes, which aligns these songs
        # correctly: NO original_split. The other two arguments are stable-ts's own
        # defaults unless the user moves them, so at rest this is align(audio, text,
        # language) and nothing more. Line grouping is reconstructed afterwards from
        # the lyric instead of being requested from the aligner.
        result = model.align(audio_path, clean, language=language or None,
                             max_word_dur=(max_word_dur or None),
                             nonspeech_skip=(nonspeech_skip or None),
                             **({"vad": True, "vad_threshold": float(vad_threshold)}
                                if vad else {}))

        # Grouped by line, and kept as the raw objects so a snap can move them and
        # have the line and section spans follow. Line bounds come from the WORDS
        # rather than seg.start/seg.end, which do not track a moved word.
        raw_words = [w for w in result.all_words() if w.word.strip()]
        grouped = group_into_lines(raw_words, lines)

        env_pack = _envelope(samples, SAMPLE_RATE) if len(samples) else None
        snapped = []
        if snap_to_onset and env_pack is not None:
            snapped = snap_onsets(raw_words, env_pack)

        words = [{"value": w.word.strip(), "start": round(float(w.start), 3),
                  "end": round(float(w.end), 3)} for w in raw_words]
        line_spans = [{"value": " ".join(x.word.strip() for x in g).strip() or text,
                       "start": round(min(float(x.start) for x in g), 3) if g else 0.0,
                       "end": round(max(float(x.end) for x in g), 3) if g else 0.0}
                      for g, text in zip(grouped, lines)]
        section_spans = sections_to_spans(sections, line_spans)

        # The contract: alignment places words, it does not change them. A mismatch
        # means the result is not the lyric that went in, and everything downstream
        # -- typography, lipsync, per-window slicing -- would be quoting fiction.
        got = _words_of(" ".join(w["value"] for w in words))
        if got != expected:
            raise ValueError(
                "MMH3ForcedAlign: the aligned word sequence does not match the lyrics "
                "(%d in, %d out). Forced alignment must never alter the words, so "
                "this is a failure rather than a partial result. Check that the audio "
                "is the vocal for THESE lyrics and that `language` is right."
                % (len(expected), len(got)))

        notes = diagnose(words, line_spans, section_spans, duration,
                         samples=samples, sample_rate=SAMPLE_RATE,
                         snapped=snapped)
        for word, was, now in snapped:
            dur = next((w["end"] - w["start"] for w in words
                        if abs(w["start"] - now) < 0.01), None)
            notes.append("snapped %r from %.2fs back to %.2fs (%.1fs earlier) -- its "
                         "audio starts there%s"
                         % (word, was, now, was - now,
                            "; it now runs %.1fs, so CHECK BY EAR that it is sung "
                            "throughout and not parked on bleed" % dur
                            if dur and dur > 8.0 else ""))
        if not sections:
            notes.append("no structural [Section] tags found, so sections_alignment "
                         "is empty; add them if you want structure awareness")
        if stripped["directions"]:
            d = stripped["directions"]
            notes.append("%d direction tag%s stripped, NOT treated as sections: %s"
                         % (len(d), "" if len(d) == 1 else "s",
                            ", ".join(repr(x) for x in d[:5])
                            + (" ..." if len(d) > 5 else "")))
        if stripped["empty_sections"]:
            notes.append("section%s with no sung line, so untimeable here: %s -- these "
                         "are the instrumental windows"
                         % ("" if len(stripped["empty_sections"]) == 1 else "s",
                            ", ".join(repr(x) for x in stripped["empty_sections"])))
        if stripped["parentheticals"]:
            notes.append("%d parenthetical%s removed; if those were backing vocals the "
                         "timings will drift -- turn strip_parentheticals off"
                         % (len(stripped["parentheticals"]),
                            "" if len(stripped["parentheticals"]) == 1 else "s"))

        payload = {"words": words, "lines": line_spans, "sections": section_spans,
                   "meta": {"model": model_size, "language": language,
                            "duration_s": round(duration, 3), "stripped": stripped, "snapped": [
                                [w, round(a, 3), round(b, 3)]
                                for w, a, b in snapped],
                            "word_count": len(words)}}

        try:
            import os
            os.remove(audio_path)
        except Exception:
            pass

        freed_mb = None
        if unload_after:
            freed_mb = release_model(key)
            logging.info("[MMH3ForcedAlign] model unloaded (%s)",
                         "%.0f MB freed" % freed_mb if freed_mb is not None
                         else "VRAM not measurable")

        # The section spans, printed. Everything above is inference; this is the one
        # line that can be checked against your own ears in a single glance, and a
        # whole section landing in the wrong place is the failure those inferences
        # kept circling without ever naming.
        spans = "".join("\n    %-16s %7.2f - %7.2f" % (x["value"], x["start"], x["end"])
                        for x in section_spans)
        report = ("%d words, %d lines, %d sections over %.2fs (%s on %s)%s\n%s"
                  % (len(words), len(line_spans), len(section_spans), duration,
                     model_size, dev, spans,
                     "\n".join("  ! " + n for n in notes) if notes
                     else "  no warnings"))
        logging.info("[MMH3ForcedAlign] %s", report.splitlines()[0])
        return io.NodeOutput(words, line_spans, section_spans,
                             json.dumps(payload, ensure_ascii=False, indent=1),
                             report)
