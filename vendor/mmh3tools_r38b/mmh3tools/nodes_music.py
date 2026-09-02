"""System prompt for MiniMax Music 3's Structured Caption.

Same problem as H3, same shape of answer. Music 3 wants a STRUCTURED caption, not a
comma-separated tag list, and MiniMax ships a hosted `music-caption-rewriter` skill to
produce one from a rough idea. Running locally there is no rewriter, so the sections
have to be authored -- or a local LLM has to be told how. This emits that instruction.

⚠ DO NOT prompt Music 3 from the older MiniMax music guide. The one in MiniMax's
skills repo (`skills/frontend-dev/references/minimax-music-guide.md`) targets the
HOSTED API of the previous generation: comma-separated descriptors, an
`--instrumental` flag, a bitrate setting, 24-hour result URLs and "~25-30 seconds per
generation". None of that applies to the open Music 3 weights, whose ComfyUI node
takes a `caption` and a `lyrics` field and runs to six minutes. Its LYRICS tags do
carry over; its caption advice does not.

Numbers below are read off the installed model, not the model card -- the card says
"~5 minutes" while `MAX_AUDIO_FRAMES / AUDIO_FRAMES_PER_SECOND` is 9000/25 = 360.0s.
"""

import logging
import re

from comfy_api.latest import io

# Read from core so these cannot drift from the model actually installed. The values
# live in comfy/ldm/minimax_music/ar.py; the fallbacks are what they were at v0.33.0.
try:
    from comfy.ldm.minimax_music.ar import (
        AUDIO_FRAMES_PER_SECOND as _FPS,
        MAX_AUDIO_FRAMES as _MAX_FRAMES,
    )
except Exception:  # pragma: no cover - core without Music 3
    _FPS, _MAX_FRAMES = 25, 9000

MUSIC_FPS = int(_FPS)
MUSIC_MAX_FRAMES = int(_MAX_FRAMES)
MUSIC_MAX_SECONDS = MUSIC_MAX_FRAMES / float(MUSIC_FPS)

# Valid on their own line, per the Music 3 model card.
SECTION_TAGS = ["[Intro]", "[Verse]", "[Pre-Chorus]", "[Chorus]", "[Post-Chorus]",
                "[Bridge]", "[Instrumental]", "[Solo]", "[Outro]"]

LYRICS_MODES = ["write lyrics", "lyrics supplied", "instrumental"]
DELIVERY = ["sung", "spoken word"]

# Tags that ASK for a sung hook. Harmless in a song, actively harmful in a monologue:
# a chorus is the one section the model has every reason to sing.
SUNG_TAGS = ["[Chorus]", "[Pre-Chorus]", "[Post-Chorus]"]

_BASE = """You turn a rough music idea into the exact inputs MiniMax Music 3 expects.
Output ONLY the fields asked for. No preamble, no commentary, no code fences."""

_CAPTION = """## caption - a STRUCTURED CAPTION, not a tag list

Three sections, in this order, each a run of prose rather than comma-separated tags.
Write them as arrangement notes a session player could act on.

    Global Metadata: genre, subgenre, BPM, key, scale, emotional progression,
      listening scenario, production profile.
    Vocal Details: vocal gender, timbre, performance style, harmony, backing vocals,
      vocal effects.
    Arrangement: primary and secondary instruments, SECTION-LEVEL INSTRUMENT
      EVOLUTION, groove, bass, percussion, textures, spatial effects.

- Give BPM as a number and key as a named key (e.g. 96 BPM, F# minor). "Mid-tempo" is
  not a BPM and "sad key" is not a key.
- Emotional PROGRESSION, not an emotional label: say where it starts and where it
  ends. "Melancholy" is a label; "restrained and close, opening out to release at the
  final chorus" is a progression.
- Section-level instrument evolution is the part that distinguishes this from a tag
  list. Say what enters, leaves or changes at each section, and name the sections you
  are talking about so they line up with the lyrics' tags.
- Production profile means the finish: analogue or digital, dry or reverberant, wide
  or narrow, compressed or open, tape/lo-fi/hi-fi.
- Describe only what can be HEARD. No cover art, no video, no backstory."""

_LYRICS_TAGS = """## lyrics - words plus structure tags

- Section tags go on a LINE OF THEIR OWN, in square brackets:
  %s
- Numbered variants are fine where a section recurs with new words: [Verse 1], [Verse 2].
- Backing vocals and performance directions go in PARENTHESES on the lyric line:
  (Ooh, yeah)   (Harmonize)   (Whispered)   (Fade out...)
- Even a bare [Verse] / [Chorus] skeleton improves the arrangement, so always tag.
- Write only words meant to be SUNG. No section commentary, no timestamps, no chords.""" % (
    "  ".join(SECTION_TAGS))

_MATCH = """## the two fields must agree

The caption and the lyrics are read together. A caption whose mood, energy or era
contradicts the lyrics produces an inconsistent take, and the disagreement is usually
invisible until you hear it. Before finishing, check that:

- the section tags named in the Arrangement exist in the lyrics, and vice versa
- the emotional progression in Global Metadata matches where the lyrics actually go
- the vocal gender and performance style in Vocal Details suit the words as written"""

_INSTRUMENTAL = """## instrumental

There are no lyrics. Emit the caption only.

- Say so explicitly in Vocal Details: no vocals, an instrumental.
- Structure still matters. Use the Arrangement section to carry it -- name the
  sections and what changes at each, since there are no lyric tags to imply them."""

_SUPPLIED = """## the lyrics are FIXED - do not reproduce them

The text below is the finished lyric. It is given to you as CONTEXT so the caption
fits it. It is wired straight to the model from elsewhere and never passes through
you, so:

- **Do NOT output the lyrics.** Do not copy them, quote them, echo them back, or
  include a `lyrics:` field at all. Emit the caption ONLY.
- Do not rewrite, translate, paraphrase, tidy, extend or "improve" them, even in your
  own head. They are not a draft.
- READ them, and write the caption to fit: derive the emotional progression, the vocal
  details and the section-level arrangement from what is actually here. Name the
  sections these lyrics actually contain so the Arrangement lines up with them.

LYRICS (context only, do not output):
%s"""


_SPOKEN = """## SPOKEN WORD - the delivery is narration, never singing

This is a dramatic spoken monologue over music. Music 3 is a MUSIC model and its
default behaviour is to sing; asking once at the top is not enough, because the
delivery drifts back into song as the piece goes on. Every rule below exists to hold
it in speech for the whole duration.

- Vocal Details must say it plainly and NEGATIVELY, not just positively: spoken word,
  narrated, monologue - and explicitly no melody, no pitched singing, no vibrato, no
  sustained notes, no harmony, no backing vocals. State the speaking register and
  cadence instead: measured, conversational, urgent, hushed, declamatory.
- REPEAT IT PER SECTION. In the Arrangement's section-level notes, restate that the
  voice stays spoken at each section, including the last. A single statement at the
  top decays; that is exactly how a take starts spoken and ends sung.
- GIVE THE MELODY SOMEWHERE ELSE TO LIVE. Name an instrumental melodic lead and let it
  carry the tune. With no melodic instrument the model has nowhere to put a melody
  except the voice, and it will.
- Do NOT write a chorus. %s ask for a sung hook by name. Use [Verse] for spoken
  passages and [Instrumental], [Interlude], [Solo], [Intro] and [Outro] for the music
  between them.
- Repetition invites melody. A refrain that recurs word-for-word will tend to acquire
  a tune, so vary the wording when an idea returns.
- Write the words as PROSE SENTENCES with real punctuation, not as metred lyric lines.
  Short end-rhymed lines of even length read as a lyric and get sung.
- State the arc in the emotional progression - a monologue earns its drama from
  delivery and arrangement, since there is no chorus to build to.""" % (
    ", ".join(SUNG_TAGS))


def plan_sections(seconds, spoken=False):
    """A section skeleton that fits `seconds`, plus a note. -> (list, note)

    Structure is the thing most easily got wrong at a given length: ask for a full
    verse/chorus/bridge song in 30 seconds and it either rushes or truncates. These
    are conventional pop-song shapes, not model constraints, so the node offers rather
    than enforces -- the LLM is told it may deviate with reason.
    """
    s = float(seconds)
    if spoken:
        # No chorus of any kind: those tags name a sung hook. Instrumental sections
        # carry the structure a chorus would otherwise provide.
        if s < 20:
            return ["[Intro]", "[Verse]"], "a single spoken passage"
        if s < 45:
            return ["[Intro]", "[Verse]", "[Outro]"], "one passage, framed by music"
        if s < 90:
            return ["[Intro]", "[Verse]", "[Instrumental]", "[Verse]", "[Outro]"], \
                "two passages with a musical breath between them"
        if s < 180:
            return ["[Intro]", "[Verse]", "[Instrumental]", "[Verse]", "[Bridge]",
                    "[Verse]", "[Outro]"], \
                "three passages, the bridge turning the argument"
        return ["[Intro]", "[Verse]", "[Instrumental]", "[Verse]", "[Bridge]",
                "[Verse]", "[Solo]", "[Verse]", "[Outro]"], \
            "long form - lean on the instrumental sections so the voice is not constant"
    if s < 20:
        return ["[Intro]", "[Verse]"], ("under ~20s fits little more than a fragment; "
                                        "one idea, no chorus return")
    if s < 45:
        return ["[Intro]", "[Verse]", "[Chorus]"], "a single statement of the hook"
    if s < 90:
        return ["[Intro]", "[Verse]", "[Chorus]", "[Verse]", "[Chorus]", "[Outro]"], \
            "two verses around a repeated chorus"
    if s < 180:
        return ["[Intro]", "[Verse 1]", "[Pre-Chorus]", "[Chorus]", "[Verse 2]",
                "[Pre-Chorus]", "[Chorus]", "[Bridge]", "[Chorus]", "[Outro]"], \
            "a full song form"
    return ["[Intro]", "[Verse 1]", "[Pre-Chorus]", "[Chorus]", "[Verse 2]",
            "[Pre-Chorus]", "[Chorus]", "[Bridge]", "[Solo]", "[Chorus]",
            "[Post-Chorus]", "[Outro]"], \
        "long form -- consider an instrumental section so it is not all vocal"


class MMH3MusicCaptionSystemPrompt(io.ComfyNode):
    """System prompt turning a rough idea into a Music 3 Structured Caption."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3MusicCaptionSystemPrompt",
            display_name="MMH3 Music Caption System Prompt",
            category="MMH3Tools/prompt",
            description=(
                "System prompt for your own LLM node, teaching it MiniMax Music 3's "
                "three-section Structured Caption and the lyrics tag format. The local "
                "stand-in for MiniMax's hosted music-caption-rewriter."
            ),
            inputs=[
                io.Float.Input(
                    "seconds", default=120.0, min=0.04, max=MUSIC_MAX_SECONDS, step=0.04,
                    tooltip="Target duration, used to suggest a section skeleton that "
                            "fits. The model may end EARLIER than this -- the node's "
                            "max_duration is a ceiling, not a length. Hard ceiling is "
                            "%.1fs (%d frames at %d fps), read from the installed model."
                            % (MUSIC_MAX_SECONDS, MUSIC_MAX_FRAMES, MUSIC_FPS),
                ),
                io.Combo.Input(
                    "lyrics_mode", options=LYRICS_MODES, default="write lyrics",
                    tooltip="'write lyrics' asks for caption AND lyrics. 'lyrics "
                            "supplied' fixes your words and writes the caption around "
                            "them. 'instrumental' emits the caption only.",
                ),
                io.Combo.Input(
                    "delivery", options=DELIVERY, default="sung",
                    tooltip="'spoken word' for dramatic monologue over music. Music 3 "
                            "is a MUSIC model and defaults to singing, so this adds a "
                            "block of countermeasures rather than one request: negative "
                            "vocal rules, per-section reinforcement, an instrumental "
                            "melodic lead to give the melody somewhere else to live, "
                            "prose-not-verse lyric shape, and a chorus-free structure "
                            "(%s name a sung hook). Ignored when lyrics_mode is "
                            "'instrumental'." % ", ".join(SUNG_TAGS),
                ),
                io.String.Input(
                    "supplied_lyrics", multiline=True, default="", optional=True,
                    tooltip="Used only by 'lyrics supplied'. Existing [Section] tags are "
                            "kept verbatim; if there are none the LLM may add them, "
                            "which is the only edit it is permitted.",
                ),
                io.Boolean.Input(
                    "suggest_structure", default=True, optional=True,
                    tooltip="Include a section skeleton sized to `seconds`. Conventional "
                            "song shapes, not model constraints -- the LLM is told it "
                            "may deviate with reason.",
                ),
                io.String.Input(
                    "extra_rules", multiline=True, default="", optional=True,
                    tooltip="Appended verbatim as a final block.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="system_prompt"),
                io.String.Output(display_name="report"),
                io.String.Output(
                    display_name="lyrics",
                    tooltip="`supplied_lyrics` passed through VERBATIM, for wiring "
                            "straight to MiniMax Music3 Text Encode's `lyrics` input. "
                            "This is the point of 'lyrics supplied' mode: the words "
                            "never touch the LLM, so they cannot be rewritten. Empty "
                            "in the other two modes, where the lyrics come from the "
                            "LLM via MMH3 Music Caption Split instead.",
                ),
            ],
        )

    @classmethod
    def execute(cls, seconds, lyrics_mode, delivery="sung", supplied_lyrics="",
                suggest_structure=True, extra_rules="") -> io.NodeOutput:
        secs = max(0.04, min(float(seconds), MUSIC_MAX_SECONDS))
        notes = []
        if float(seconds) > MUSIC_MAX_SECONDS:
            notes.append("seconds %.2f is past the %.1fs ceiling (%d frames at %d fps); "
                         "clamped." % (float(seconds), MUSIC_MAX_SECONDS,
                                       MUSIC_MAX_FRAMES, MUSIC_FPS))

        supplied = (supplied_lyrics or "").strip()
        if lyrics_mode == "lyrics supplied":
            notes.append("the LLM writes the CAPTION ONLY. Wire this node's `lyrics` "
                         "output straight to MiniMax Music3 Text Encode -- do not route "
                         "the words through the LLM or the split node, or they will "
                         "come back rewritten.")
        if lyrics_mode == "lyrics supplied" and not supplied:
            raise ValueError(
                "MMH3MusicCaptionSystemPrompt: lyrics_mode is 'lyrics supplied' but "
                "supplied_lyrics is empty. Paste the words, or switch to 'write "
                "lyrics'/'instrumental'.")
        if lyrics_mode != "lyrics supplied" and supplied:
            notes.append("supplied_lyrics is filled but lyrics_mode is %r, so it is "
                         "ignored." % lyrics_mode)

        spoken = delivery == "spoken word" and lyrics_mode != "instrumental"
        if delivery == "spoken word" and lyrics_mode == "instrumental":
            notes.append("delivery 'spoken word' is meaningless with lyrics_mode "
                         "'instrumental' (there is no voice); ignored.")

        parts = [_BASE]
        if lyrics_mode == "instrumental":
            parts += ["## Output\n\nEmit ONE field:\n\n    caption: <the structured "
                      "caption>", _CAPTION, _INSTRUMENTAL]
        elif lyrics_mode == "lyrics supplied":
            # The lyrics are wired straight to the encoder; the LLM writes the caption
            # only. Asking a language model to reproduce a text verbatim is asking it
            # not to be a language model -- it drifts, and the drift is the lyric.
            parts += ["## Output\n\nEmit ONE field:\n\n    caption: <the structured "
                      "caption>", _CAPTION, _LYRICS_TAGS]
            if spoken:
                parts.append(_SPOKEN)
            parts += [_SUPPLIED % supplied, _MATCH]
        else:
            parts += ["## Output\n\nEmit TWO fields, in this order, blank line between:"
                      "\n\n    caption: <the structured caption>\n\n    lyrics: <the "
                      "%s, with section tags on their own lines>"
                      % ("spoken text" if spoken else "words"),
                      _CAPTION, _LYRICS_TAGS]
            if spoken:
                parts.append(_SPOKEN)
            parts.append(_MATCH)

        skeleton, why = plan_sections(secs, spoken=spoken)
        if spoken and lyrics_mode == "lyrics supplied":
            bad = [t for t in SUNG_TAGS if t.lower() in supplied.lower()]
            if bad:
                notes.append("supplied lyrics contain %s, which name a sung hook and "
                             "work against spoken delivery. Consider [Verse] or an "
                             "instrumental tag instead." % ", ".join(bad))
        if suggest_structure and lyrics_mode != "lyrics supplied":
            parts.append(
                "## Length\n\nTarget about %.1f seconds. A shape that fits, which you "
                "may deviate from with reason - %s:\n\n    %s\n\nThe duration is a "
                "CEILING, not a target: the model may end the song earlier, so do not "
                "pad to fill it."
                % (secs, why, "  ".join(skeleton)))
        elif suggest_structure:
            parts.append(
                "## Length\n\nTarget about %.1f seconds, as a CEILING rather than a "
                "target - the model may end earlier. Do not alter the supplied lyrics "
                "to fit it; write the arrangement to suit their natural length."
                % secs)

        if (extra_rules or "").strip():
            parts.append(extra_rules.strip())

        system = "\n\n".join(parts)
        report = ("%s, %s | %.2fs of %.1fs max (%d frames at %d fps)\n"
                  "  structure: %s\n%s"
                  % (lyrics_mode, "SPOKEN WORD" if spoken else delivery,
                     secs, MUSIC_MAX_SECONDS,
                     round(secs * MUSIC_FPS), MUSIC_FPS,
                     " ".join(skeleton) if suggest_structure else "(not suggested)",
                     "\n".join("  ! " + n for n in notes) if notes
                     else "  no warnings"))
        logging.info("[MMH3MusicCaptionSystemPrompt] %s", report.splitlines()[0])
        # verbatim, deliberately: the whole point of 'lyrics supplied' is that these
        # words bypass the LLM entirely on their way to the encoder
        return io.NodeOutput(system, report,
                             supplied if lyrics_mode == "lyrics supplied" else "")


# ---------------------------------------------------------------------------
# The LLM answers with BOTH fields in one string; MiniMaxMusic3TextEncode wants
# them on two sockets. Splitting is the join between the two halves of the graph.

_FENCE = re.compile(r"^\s*```[^\n]*\n(.*?)\n\s*```\s*$", re.S)
# label at the start of a line, case-insensitive, colon optional-ish, tolerant of
# an LLM bolding it or bulleting it
_LABEL = r"^[ \t]*(?:[-*>]\s*)?\**\s*%s\s*\**\s*:\s*\**[ \t]*"
# a lyrics field of nothing but [tags] yields a wordless track
_LYRIC_TAG_STRIPPED = re.compile(r"\[[^\]]*\]")


def _strip_fence(text):
    """Unwrap a single ``` fenced block, if the whole reply is one."""
    m = _FENCE.match(text or "")
    return m.group(1) if m else (text or "")


def split_caption_lyrics(text):
    """LLM reply -> (caption, lyrics, notes).

    Deliberately NOT a str.split(). Real replies arrive fenced, prefaced with
    "Here's the caption:", with the labels bolded, or with the lyrics label absent
    entirely because the piece is instrumental. Every one of those should yield a
    usable caption rather than an exception or, worse, a silently empty one.

    Markdown is left ALONE. `clean_caption()` in comfy/ldm/minimax_music/prompt.py
    already strips it downstream, and stripping twice risks eating an asterisk that
    was part of the text.
    """
    notes = []
    body = _strip_fence(text or "").strip()
    if not body:
        return "", "", ["input was empty"]

    cap_m = re.search(_LABEL % "caption", body, re.I | re.M)
    lyr_m = re.search(_LABEL % "lyrics", body, re.I | re.M)

    if cap_m is None and lyr_m is None:
        # No labels at all. The whole reply is most likely the caption -- an LLM that
        # ignored the output format still usually wrote the thing that was asked for.
        notes.append("no 'caption:' or 'lyrics:' label found; treating the whole reply "
                     "as the caption. Check the LLM is following the system prompt.")
        return body, "", notes

    if cap_m is None:
        notes.append("no 'caption:' label found, only 'lyrics:'. The caption is what "
                     "carries style -- without it the model invents one.")
        caption = body[:lyr_m.start()].strip()
        lyrics = body[lyr_m.end():].strip()
        if caption:
            notes.append("text before 'lyrics:' was used as the caption.")
        return caption, lyrics, notes

    if lyr_m is None:
        notes.append("no 'lyrics:' label found; instrumental, or the LLM omitted it.")
        return body[cap_m.end():].strip(), "", notes

    if lyr_m.start() < cap_m.start():
        notes.append("'lyrics:' appears BEFORE 'caption:'; read in the order found.")
        return (body[cap_m.end():].strip(),
                body[lyr_m.end():cap_m.start()].strip(), notes)

    return (body[cap_m.end():lyr_m.start()].strip(),
            body[lyr_m.end():].strip(), notes)


class MMH3MusicCaptionSplit(io.ComfyNode):
    """Split an LLM's reply into MiniMaxMusic3TextEncode's caption and lyrics."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3MusicCaptionSplit",
            display_name="MMH3 Music Caption Split",
            category="MMH3Tools/prompt",
            description=(
                "Split one LLM reply into the `caption` and `lyrics` strings MiniMax "
                "Music3 Text Encode takes. Tolerates code fences, preamble, bolded or "
                "bulleted labels, and a missing lyrics field."
            ),
            inputs=[
                io.String.Input(
                    "text", multiline=True, default="",
                    tooltip="The LLM's reply to MMH3 Music Caption System Prompt.",
                ),
                io.Boolean.Input(
                    "strict", default=False, optional=True,
                    tooltip="Raise instead of warning when the caption comes back "
                            "empty, or when no labels are found at all. Off reports "
                            "and carries on, which is usually what you want while "
                            "tuning a local model.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="caption"),
                io.String.Output(display_name="lyrics"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, text, strict=False) -> io.NodeOutput:
        caption, lyrics, notes = split_caption_lyrics(text)

        if not caption.strip():
            msg = ("no caption recovered. Music 3 reads style ONLY from the caption, "
                   "so an empty one means the model invents the entire arrangement.")
            if strict:
                raise ValueError("MMH3MusicCaptionSplit: " + msg)
            notes.append(msg)
        if strict and any(n.startswith("no 'caption:' or 'lyrics:'") for n in notes):
            raise ValueError(
                "MMH3MusicCaptionSplit: the reply carried no 'caption:' or 'lyrics:' "
                "label. The LLM is not following the system prompt's output format.")

        # A lyrics field that is only section tags produces a track with no words --
        # silent-looking output that is easy to misread as a model failure.
        if lyrics.strip() and not _LYRIC_TAG_STRIPPED.sub("", lyrics).strip():
            notes.append("lyrics contain section tags but no words; the result will be "
                         "effectively instrumental.")

        report = ("caption %d chars, lyrics %d chars\n%s"
                  % (len(caption), len(lyrics),
                     "\n".join("  ! " + n for n in notes) if notes
                     else "  clean split"))
        logging.info("[MMH3MusicCaptionSplit] %s", report.splitlines()[0])
        return io.NodeOutput(caption, lyrics, report)



# ---------------------------------------------------------------------------
# Sectionising supplied lyrics, WITHOUT touching the words.
#
# Community finding (banodoco #minimax_music3, 2026-08): the model fits song length
# to the LYRICS, allocating time per SECTION -- so a long block of text inside one
# [Verse] is compressed into that section's slot and the delivery rushes. RuneX, who
# reported the rushing, diagnosed it himself: the verses were too long, and breaking
# them up with something between should make the song run longer instead.
#
# That fix is a pure text operation, so it must NOT go through an LLM -- handing a
# fixed lyric to a language model is what rewrote it in the first place. Splitting
# here only ever INSERTS tag lines between existing blocks; the word sequence is
# checked identical before and after.

_WORDS = re.compile(r"\S+")
_TAG_LINE = re.compile(r"^\s*\[[^\]]+\]\s*$")
# sentence end followed by whitespace -- the preferred split point inside a paragraph
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _words_only(text):
    """The word sequence with tag lines removed -- the thing that must not change."""
    kept = [ln for ln in (text or "").splitlines() if not _TAG_LINE.match(ln)]
    return _WORDS.findall(" ".join(kept))


def _blocks(text):
    """Split into the smallest units we are willing to keep together.

    Preference order, strongest authorial signal first: blank-line paragraphs, then
    sentence ends. Never mid-sentence -- a boundary inside a sentence is worse than
    an over-long section.
    """
    paras = [p for p in re.split(r"\n\s*\n", (text or "").strip()) if p.strip()]
    out = []
    for p in paras:
        pieces = [s for s in _SENTENCE_END.split(p.strip()) if s.strip()]
        out.extend(pieces if pieces else [p.strip()])
    return out


def sectionize_lyrics(text, seconds, words_per_section=35, interlude=True,
                      tag="[Verse]", interlude_tag="[Instrumental]"):
    """Insert section tags into fixed lyrics. -> (tagged_text, notes)

    The words are never altered, reflowed or re-punctuated; only tag lines are
    inserted between blocks. Verified by comparing the word sequence.
    """
    notes = []
    src = (text or "").strip()
    if not src:
        return "", ["no lyrics to sectionise"]

    if any(_TAG_LINE.match(ln) for ln in src.splitlines()):
        notes.append("lyrics already carry section tags; left exactly as they are.")
        return src, notes

    blocks = _blocks(src)
    total_words = sum(len(_WORDS.findall(b)) for b in blocks)
    wps = max(4, int(words_per_section))
    want = max(1, int(round(total_words / float(wps))))
    want = min(want, len(blocks))          # cannot have more sections than blocks

    # greedy fill: start a new section once the current one is at its word budget, so
    # a section boundary always lands on a block boundary
    per = max(1, int(round(total_words / float(want))))
    sections, cur, cur_words = [], [], 0
    for b in blocks:
        n = len(_WORDS.findall(b))
        if cur and cur_words + n > per and len(sections) < want - 1:
            sections.append(cur)
            cur, cur_words = [], 0
        cur.append(b)
        cur_words += n
    if cur:
        sections.append(cur)

    # NUMBER the sections when there is more than one. The caption's section-level
    # instrument evolution refers to sections by name -- "strings enter at Verse 2" --
    # and identical bare [Verse] tags give it nothing to bind to. Numbered variants are
    # sanctioned by the model card for exactly this.
    def _numbered(t, i, total):
        return t if total < 2 else "%s %d]" % (t[:-1], i + 1)

    lines = []
    for i, sec in enumerate(sections):
        if interlude and i:
            lines.append(_numbered(interlude_tag, i - 1, len(sections) - 1))
        lines.append(_numbered(tag, i, len(sections)))
        lines.append("\n".join(sec))
    out = "\n".join(lines)

    # the guarantee, checked rather than promised in prose
    if _words_only(out) != _WORDS.findall(src):
        raise RuntimeError(
            "sectionize_lyrics changed the words, which must never happen. Report "
            "this with the input text.")

    rate = total_words / max(1e-6, float(seconds))
    notes.append("%d words over %d section%s (~%d words each)"
                 % (total_words, len(sections),
                    "" if len(sections) == 1 else "s", per))
    if rate > 3.0:
        notes.append("%.1f words/second at %.0fs is faster than dramatic speech "
                     "(~2-2.5 w/s). Sectionising spreads the words out but cannot "
                     "create time -- raise the duration or cut words." % (rate, seconds))
    elif rate > 2.5:
        notes.append("%.1f words/second is brisk for spoken delivery; consider a "
                     "longer duration." % rate)
    return out, notes


class MMH3LyricsSectionize(io.ComfyNode):
    """Insert section tags into fixed lyrics, deterministically and verbatim."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3LyricsSectionize",
            display_name="MMH3 Lyrics Sectionize",
            category="MMH3Tools/prompt",
            description=(
                "Split fixed lyrics across sections WITHOUT changing a word. Music 3 "
                "allocates time per section, so one long block gets compressed and the "
                "delivery rushes. Deterministic on purpose: an LLM asked to re-emit a "
                "fixed lyric rewrites it."
            ),
            inputs=[
                io.String.Input(
                    "lyrics", multiline=True, default="",
                    tooltip="The finished words. Only tag lines are inserted -- no "
                            "reflow, no re-punctuation. Lyrics that ALREADY carry "
                            "section tags are passed through untouched.",
                ),
                io.Float.Input(
                    "seconds", default=120.0, min=0.04, max=MUSIC_MAX_SECONDS, step=0.04,
                    tooltip="Target duration, used ONLY to warn when the words cannot "
                            "fit at a speakable rate. Sectionising spreads words out; "
                            "it cannot create time.\n\n"
                            "Deliberately EXCLUDED from this node's cache key, because "
                            "it changes no text: were it included, nudging it would "
                            "re-run the whole autoregressive generation downstream for "
                            "an identical string. The side effect is that the report's "
                            "rate warning can be stale until something else changes.",
                ),
                io.Int.Input(
                    "words_per_section", default=35, min=4, max=400,
                    tooltip="Roughly how many words per section. ~35 is about 12-15s of "
                            "dramatic speech. Lower = more sections = longer and less "
                            "hurried; too low chops the sense.",
                ),
                io.Boolean.Input(
                    "interlude", default=True,
                    tooltip="Insert an instrumental tag between sections so the music "
                            "carries the gaps instead of the words running together. "
                            "The spoken-word equivalent of the community's 'verses "
                            "with a chorus in between' fix.",
                ),
                io.Combo.Input(
                    "section_tag", options=["[Verse]", "[Intro]", "[Bridge]", "[Outro]"],
                    default="[Verse]", optional=True,
                    tooltip="Tag for the spoken passages. The chorus tags are "
                            "deliberately absent -- they ask for a sung hook.",
                ),
                io.Combo.Input(
                    "interlude_tag",
                    options=["[Instrumental]", "[Interlude]", "[Solo]"],
                    default="[Instrumental]", optional=True,
                ),
            ],
            outputs=[
                io.String.Output(display_name="lyrics"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, lyrics="", seconds=0.0, words_per_section=35,
                           interlude=True, section_tag="[Verse]",
                           interlude_tag="[Instrumental]", **_):
        """Cache key over the inputs that change the TEXT -- `seconds` excluded.

        `seconds` only drives the rate warning in the report; it does not move a
        single word. Without this, nudging it would invalidate this node, and a
        link's cache key is the upstream node's key rather than its output value --
        so the downstream MiniMaxMusic3TextEncode would re-run the ENTIRE
        autoregressive generation to arrive at a byte-identical string.

        The cost is that the report's warning goes stale until something else
        changes. That is the right trade: the report is diagnostic, the lyrics are
        the product, and the AR pass is minutes.
        """
        return (lyrics, int(words_per_section), bool(interlude),
                section_tag, interlude_tag)

    @classmethod
    def execute(cls, lyrics, seconds, words_per_section, interlude,
                section_tag="[Verse]", interlude_tag="[Instrumental]") -> io.NodeOutput:
        out, notes = sectionize_lyrics(lyrics, seconds, words_per_section, interlude,
                                       section_tag, interlude_tag)
        report = ("%d chars in, %d out\n%s"
                  % (len((lyrics or "").strip()), len(out),
                     "\n".join("  " + n for n in notes) if notes else "  unchanged"))
        logging.info("[MMH3LyricsSectionize] %s", notes[0] if notes else "unchanged")
        return io.NodeOutput(out, report)
