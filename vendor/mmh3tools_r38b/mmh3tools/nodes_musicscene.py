"""Section-by-section prompt building for a MUSIC VIDEO.

Same three-stage shape as `MMH3ScenePlanPrompt` and a different set of rules,
because the cinematic version would fight a song.

WHAT CHANGES, AND WHY

  * **The arc is the song's.** The cinematic beats stage exists to invent an
    escalation and forbid early resolution. A song already has its structure, and
    its choruses are MEANT to land the same way twice. Told to escalate, the model
    pushes through a repeat that should feel like a return.
  * **The words are given.** Nothing is invented about what is sung: the aligned
    lyric supplies the text, and the word onsets supply the shot timestamps. The
    two things the cinematic loop got wrong by inventing -- the words and the
    timings -- are supplied here.
  * **Typography is assigned once, across the whole song.** Decided per chunk with
    lyrics in hand, every chunk reaches for it and you get it on all of them.
  * **Some windows have no words at all.** An intro or instrumental break needs its
    own branch, told that nothing is sung, or it will invent singing to fill it.

The shots stage is the only one that runs per window, and it is handed that
window's verbatim lines on the CHUNK's clock -- so a timestamp it writes is one it
was given, not one it guessed.
"""

import logging

from comfy_api.latest import io

STAGES = ["definitions", "beats", "shots"]
TYPOGRAPHY = ["off", "exact lyrics", "text bursts"]
MUSIC_SOURCE = ["supplied", "generated"]
TREATMENTS = ["music video", "restrained"]

_BRIEF = """=== THE VIDEO'S IDEA ===

This is what the video is ABOUT, so you can build it. It is not narration and no
character speaks it.

%s"""

_DEFINITIONS = """You are writing the FILM-WIDE sections of a MiniMax H3 prompt for a
MUSIC VIDEO, once. They are reused BYTE-IDENTICALLY in every chunk, so they must be
complete and final now.

Your reply is these six headers, in exactly this order:

    subject_definitions:
    summary:
    retention_analysis:
    detailed_description:
    overall_soundscape:
    non_diegetic_music:

FOUR of them YOU WRITE, in full: subject_definitions, retention_analysis,
overall_soundscape and non_diegetic_music. Writing those IS the job; a reply that
returns them blank is a failed reply.

TWO are per-chunk and get filled in later: summary and detailed_description. Emit
ONLY the bare header for those two, nothing after it, in place and in order.

Emit each header EXACTLY ONCE. No preamble, no commentary, no code fences.

## subject_definitions - one line per label

    <Subject N>  reusable visible content: the performer, a costume, a location, a
                 recurring object or motif.
    <Picture N>  standalone only when the image is a concrete frame anchor; if it
                 merely defines a look, cite it INSIDE a <Subject N> line.
    <Video N>    whole-video relationships only: editing, continuation, borrowed
                 camera movement.
    <Audio N>    an audio asset, bound to a speaker where it maps to one.

DEFINE EVERY LABEL YOU WILL USE, AND USE EVERY LABEL YOU DEFINE. Describe what is
VISIBLE and permanent: build, hair, clothing, markings, the space. Not mood, not
backstory.

**NO TYPOGRAPHY INSTRUCTIONS IN HERE.** A <Subject> line says what a reference asset
IS. Animation or text directives in it are a category error -- they are not
properties of the asset, and they leak into every chunk that reuses the line. Where
on-screen text goes and how it behaves is decided per chunk, not here.

A music video returns to the same performer and the same places repeatedly, so
these carry more weight here than in a scene: they are what makes chunk 8 look like
chunk 1.

## retention_analysis - ONE LINE PER LABEL, no exceptions

    <Subject 1>: fully_preserved - her build, hair and coat are identical throughout.

Visible markers: fully_preserved, partially_preserved, attribute_transfer,
weak_reference. Audio markers: fully_copy, partially_copy, reference, weak_reference.
Those are values to CHOOSE BETWEEN, never to list.

@@AUDIO_RULES@@"""


_AUDIO_SUPPLIED = """## overall_soundscape

THE SONG IS THE AUDIO. Do not invent room tone, weather or footsteps competing with
it. Describe the acoustic world only where the picture implies one, and say plainly
that the track carries the sound.

## non_diegetic_music - the actual track, not an invented score

Describe THIS song: instrumentation, tempo, texture, how the vocal sits. You are
describing something that already exists and will be supplied as audio, so do not
compose an alternative."""


_AUDIO_GENERATED = """## overall_soundscape

THERE IS NO TRACK YET. The model writes the audio in the same pass as the picture,
out of these words. Never say the audio "is provided", "is supplied", or "carries
the sound" -- there is nothing there to carry it.

This section is AMBIENCE AND ACTION SOUND ONLY: room tone, rain, a door, a footstep
the picture shows. Never the vocal and never the backing -- those belong to
non_diegetic_music and to the sung lines. If the picture implies no diegetic sound,
write N/A.

## non_diegetic_music - the SPEC the model performs

You are not describing a song that exists. You are specifying the one to generate,
and this section is what the model reads to make it.

- Name genre and tempo: an approximate BPM, or a plain descriptor.
- Name instruments SPECIFICALLY and always with a playing style -- "brushed snare",
  "fingerpicked steel-string", "plucky analog synth bass" -- never "drums, bass".
- Restate the same tempo, key and core rhythm section in EVERY chunk, in the same
  terms. That repetition is what stops the song drifting between chunks. It is not
  padding to trim.
- Then say what this chunk does differently from the one before, and end with ONE
  clause describing the shape WITHIN it: "lifting into the second half", "swelling
  and receding", "building to a peak before falling away".
- Do NOT ask for a structural change inside a chunk. The model holds a groove and
  executes dynamics; a section changes because a new chunk begins, not because you
  asked."""


_SUNG_GENERATED = """
=== THE WORDS ARE SUNG, AND ONLY YOU CAN ASK FOR THEM ===

The model generates the vocal. If the words are not written into
detailed_description they are NOT SUNG, and the result is an instrumental with a
mouth moving to nothing.

- Write the line as <d>[English] the words</d>, VERBATIM from the lyrics you were
  given for this chunk. Do not paraphrase, reorder or improve them.
- Attribute it: the subject SINGS, never says. Use the same (Sx) id as speech --
  Subject 1 (S1) sings: <d>[English] ...</d>
- Hang the action off the line, OPENING the passage with it rather than appending it
  after a run of description.
- Describe the singing physically: sustained open vowels, a held note, breath taken
  before a phrase, jaw and throat. Sung mouth shapes are not spoken ones and the
  model needs telling which it is.
- Put the words ONLY here. Never repeat them in overall_soundscape or
  non_diegetic_music.
- TIME THE CUTS TO THE VOICE: cut on a breath, a phrase end, or a rest. Never cut
  mid-word -- the mouth is mid-vowel on both sides of the join."""


_TREAT_MUSIC_VIDEO = """- **SPLIT FRAMES ARE A TOOL HERE, so reach for them on purpose.** Two places, two
  times or two framings named inside ONE shot makes H3 divide the frame -- split
  screen, inset, banded overlay. That is a music-video technique, not an artifact.
  Say which halves hold what, and whether they move together or against each other.
- The performer can be present, absent, or multiplied. A music video is not obliged
  to be literal about who is singing, and several of her in one frame -- tiled,
  mirrored, out of step -- is a legitimate image rather than a mistake.
- Vary it. A whole video of split frames is as flat as none; the technique lands
  where a single-image shot came before it."""


_TREAT_RESTRAINED = """- **ONE IMAGE PER SHOT. Do not divide the frame.** No split screen, no inset, no
  banded overlay, no tiled or mirrored copies of the performer. Naming two places or
  two framings inside one shot is what makes H3 split the frame, so name ONE.
- The performer stays whole and present. If she is singing, the frame holds her
  close enough to read her face and mouth. That is the subject, not a surface to
  decorate.
- Carry the interest in camera, light and staging instead: a move that means
  something, a practical that pulses with the track, a change of distance."""


_TREAT_MENU_MV = """Known to render well: **split frames**, **RGB channel split**, **slow motion**.
Those are starting points, not the whole vocabulary -- name the treatment
explicitly, say how strong it is, and say WHEN it hits. An effect that runs
continuously stops reading as an effect."""


_TREAT_MENU_RESTRAINED = """Keep it optical and physical: bloom, halation, smeared highlights, shallow focus,
slow motion. Name the treatment explicitly, say how strong it is, and say WHEN it
hits. An effect that runs continuously stops reading as an effect. No frame
division, no channel splitting, no datamosh."""


_BEATS = """You are writing the SUMMARY of every chunk of a MUSIC VIDEO, all at once,
as a beat sheet. One summary per chunk, %d of them, separated by a single |
character.

Output ONLY the summaries and the separators. No numbering, no labels, no headings.

**THE `|` IS MANDATORY AND IS THE ONLY SEPARATOR.** Put a single `|` between every
pair of summaries -- %d summaries means exactly %d pipe characters. The bracketed
prefix is NOT a separator: `...narrowing back.[reference generation] Before the...`
run together with no pipe is ONE summary as far as everything downstream is
concerned, and the whole sheet then gets spliced into every chunk instead of one
beat each. Do not end the last summary with a pipe.

Each is one paragraph opening with a bracketed TASK-TYPE prefix, e.g.
[reference generation], reusing only the labels you were given. Introduce no new
labels.

**The prefix is a TASK TYPE. It is NEVER a section name.** `[subject_definitions]`,
`[summary]`, `[retention_analysis]`, `[detailed_description]`, `[overall_soundscape]`
and `[non_diegetic_music]` are the sections of the prompt being assembled -- they are
not tasks and must never appear in a bracket here. A summary prefixed with a section
name gets copied wholesale into that section further down the chain, and the chunk
loses its shot.

## THE ARC IS THE SONG'S, NOT YOURS

You are given the whole lyric with its sections and timings. That structure is the
plan; your job is to give it pictures, not to impose a second story on top.

- **Do NOT invent an escalation.** A verse into a chorus is a return, not a rise.
- **A repeated chorus should FEEL like the same chorus.** Come back to the same
  place, the same framing, the same motif. Vary the treatment -- closer, wider,
  more damaged, more crowded -- but do not restage it as somewhere new. Repetition
  is what a chorus is for.
- **Let the sections differ from each other**, though. Verses and choruses should
  not look alike; that contrast is the song's own shape and the video should show it.
- A bridge is usually the one place something genuinely changes. Treat it as the
  exception rather than the rule.

## Per chunk

Each chunk is about %.1f seconds -- long enough for several shots, so a summary can
carry more than one image. Say WHERE it is, WHO is in frame, and what the picture is
doing while those words are sung.

Chunks often straddle a section boundary. When one does, say what the picture does
at the turn.
%s
## Length and shape

Write for the words that are actually sung in each chunk. You have them below with
timings, so a summary that describes a moment nothing is sung in is describing
nothing."""

_TYPO_BEATS = """
## TYPOGRAPHY IS ASSIGNED HERE, ONCE, ACROSS THE WHOLE SONG

You can see all %d chunks. Nothing downstream can, so this is the only place it can
be rationed.

- Name the chunks that carry on-screen text and the chunks that do not. MOST SHOULD
  NOT. Text everywhere reads as a lyric video, and the hook stops landing.
- Prefer the hook, the title line, or a line that repeats. %s
- Say it plainly in the summary, e.g. "on-screen text on the hook" -- the chunk's own
  writer will render it and will not add any of its own.

**Decide the TYPOGRAPHIC IDENTITY once, here, and state it in the first summary that
carries text.** One material, one behaviour, for the whole video -- otherwise each
chunk invents its own and the video has no design.

Build it from THIS video's world, not from a font menu. The type should look like it
belongs to the thing the song is about: circuit traces and etched copper for a
machine song, vapour and sugar-floss for a candy one, wet concrete, torn paper,
CRT phosphor, embroidery. Name the material, the case and the weight, and the chunks
will render that identity rather than defaulting to plain white capitals.
"""

_TYPO_EXACT = ("The text must be the sung line VERBATIM, so pick lines worth "
               "reading whole.")
_TYPO_BURST = ("The text is a short burst drawn from the sung line -- a fragment, a "
               "word, a re-spelling. It does not have to be the whole line or be "
               "literal, but it must MEAN something standing alone, so choose lines "
               "that contain a word worth putting on a screen. A line whose only "
               "short words are 'much' or 'just' is a bad candidate.")

_TYPO_TREATMENT = """
## MAKE IT DESIGN, NOT A SUBTITLE

A word at readable size in the middle of frame IS a subtitle and reads as one. On
screen text in a music video is a graphic decision, and the default choice is always
the flat one.

- **SCALE FIRST, and the safe answer is wrong.** Fill the frame edge to edge, crop
  the letters at the sides, or bury it small in a corner. Mid-sized and centred is
  the single option that looks like captioning.
- **DECIDE WHERE IT LIVES.** In the world -- painted on the wall behind her, lit on a
  monitor in shot, printed across her coat, caught in the mirror -- or an overlay
  that commits to being one: hard cut, no fade, sitting flat on top of the picture.
  Pick one. The undecided version is the subtitle.
- **GIVE IT A MOMENT.** It arrives ON a word onset or a bar line from the timings
  above, not whenever. It can stamp in, flicker, hold a single beat and go, or stay
  while the shot cuts underneath it.
- **USE THE IDENTITY THE BEAT SHEET SET.** If it named a material -- circuit traces,
  vapour, torn paper -- render the letters IN that material, and do not invent a
  different one. If it named none, choose one from this video's world and describe
  it as a material rather than as a font: what the letters are MADE of, their case
  and weight, their colour against the frame. "Text appears" is not a description.
- **One treatment per chunk.** Text that behaves differently each time it appears
  reads as an accident rather than a decision.
"""


_SHOTS = """You are writing ONE section of ONE chunk of a MUSIC VIDEO:
detailed_description.

Return ONLY that section's text. No section label, no other sections, no preamble,
no code fences, no markdown.

**WRITE THE SHOT. DO NOT RESTATE THE SUMMARY.** You are given this chunk's summary as
your instruction, not as your answer. Never copy it back, never open with a bracketed
label taken from it, and never describe the chunk in the third person -- "for the line
X, she walks past Y" is a plan, not a shot. Your output MUST contain `[Shot 1]`. If it
does not, the chunk renders from a paragraph about a video instead of a description of
one.

You are writing **chunk %d of %d**, which covers **%s** and runs %.1f seconds.

## Structure

- One or two style sentences FIRST, before [Shot 1]. Look only, no shot content.
- [Shot 1] carries NO timestamp. Every later shot does:
  "[Shot 2] At 00:03.500, the camera cuts to ..." Times strictly increase and stay
  inside %.1f seconds.
- Use ONLY the labels you were given.

## CUT ON THE WORDS

The timings below are measured from the audio. They are the truth about when things
are sung, and they are given in THIS CHUNK's time, starting at 00:00.000.

- Hang your shot changes on them. A cut that lands on a word onset feels like the
  video is listening; one that lands anywhere else feels like a slideshow.
- DO NOT invent a timestamp. Every time you write should be one you were given, or
  sit deliberately between two of them.
- You do not need a shot per line. Two or three strong shots beat six weak ones.

## Write it as a music video

- Give the camera and the light INTENT. Motion, texture, a practical that pulses
  with the track.
@@TREATMENTS@@

## FRAME TREATMENTS COME FROM THE VIDEO'S WORLD

The same rule as the typography: an effect belongs if the song earns it. A machine
song earns RGB channel split, scanlines, datamosh, dropped frames, interlace tear.
A soft one earns bloom, halation, print-through, smeared highlights. Take the
treatment from what the video is ABOUT rather than from a menu of filters.

@@TREATMENT_MENU@@

## MOTIVATE THE CUT

A cut lands when something carries across it. Let motion, a moving edge, a light, or
matched geometry drive the change rather than ending one shot and starting another.

- A hand sweeping left hands off to a curtain sweeping left. A circle becomes a
  circle. A highlight travelling down a rail becomes a highlight travelling down her
  arm. Say what carries.
- The cut is still placed on a word onset or a bar line -- this decides what makes it
  feel inevitable once it is there.
- Not every cut needs it. Two or three motivated cuts in a chunk read as design; all
  of them read as a showreel.

## CHOOSE A TEMPO FOR THE CHUNK, AND SAY IT

Speed is a decision, and leaving it unmade gets you the model's default rather than
yours.

- **SLOW MOTION** renders well and holds when you ask for it. Say so explicitly --
  it is a choice you make, not one the model makes for you.
- **REALTIME** is the other choice. If a chunk ever comes back slower than intended,
  opening the style sentences with "live-action video" is the documented counter.
- **RAPID MICRO-SHOTS** are the other pole: many short shots instead of one take,
  aggressive reframing, snap-zooms into eyes and hands, orbits. Dense and edit-led,
  the opposite of a held frame.
- Match it to the music. The energy reading for this chunk is above -- a peak
  chunk and a near-silent one should not move at the same speed.
- Add visual incident that changes no story: weather, crowd, a surface reacting,
  something breaking at the edge of frame.
- Prefer one exact, strange image over four general ones.
- Write only what a camera can record. No interior states, no symbolism, no
  "representing" or "conveying".
%s%s
=== WHAT IS SUNG IN THIS CHUNK (chunk-relative times) ===

%s
=== WORD ONSETS, for cutting ===

%s
"""

_SHOTS_CONTEXT = """
=== THE CHUNK BEFORE / AFTER (context only -- do not write these) ===

%s
"""

_INSTRUMENTAL = """
## NOTHING IS SUNG IN THIS CHUNK

This window is an intro, an instrumental passage or an outro. There are no words.

- Do NOT write singing, lip movement or dialogue. Nobody is singing here.
- This is where VISUAL EVENT carries the chunk instead: something arrives, breaks,
  turns, floods, empties. Give it the thing the words were doing elsewhere.
- It is also the natural place for a change of location or a reveal, since no lyric
  is anchoring the picture.
- NO on-screen text, whatever the beat sheet assigned elsewhere -- there is no line
  to quote.
"""

_TYPO_SHOTS_EXACT = """
## ON-SCREEN TEXT: VERBATIM

The beat sheet assigned on-screen text to this chunk.

- Put the line in DOUBLE QUOTES, exactly as it appears above, spelling and
  punctuation unchanged. Double quotes are what makes H3 render text ON SCREEN.
- Say where it sits, how it behaves and when it appears, using the times above.
- Quote a line that is actually sung in this chunk, and nothing else.
"""

_TYPO_SHOTS_BURST = """
## ON-SCREEN TEXT: BURSTS

The beat sheet assigned on-screen text to this chunk.

**Do it in TWO STEPS, and never merge them.**

1. **CHOOSE** the phrase. Paraphrase freely -- semantic intent beats word accuracy.
   Aim for THREE WORDS, ALL CAPS. It does not have to be the whole line, or literal:
   a fragment, one word, a re-spelling, a fracture of it. You have the real words in
   front of you, so what you make of them is grounded rather than guessed.
2. **RENDER** it as a literal string in DOUBLE QUOTES. Double quotes are what make
   H3 draw text ON SCREEN.

**VALIDITY RULE: a prompt that describes text moving without quoting the text is
invalid.** Writing "words of devotion cascade in neon" names no string, so the model
draws invented letterforms -- observed, and it is gibberish on screen. It can only
draw a string it was given. If you describe type doing something, the thing it is
doing must appear in quotes in the same sentence.
- **IT MUST MEAN SOMETHING ALONE.** A noun, a verb, a name, an order. The word or
  phrase a stranger would remember from the line, not the shortest one in it.
- NEVER a function word. Not "much", "very", "just", "so", "the", "and", "that",
  "this", "some", "then", "when", "with". Those are the shortest words in any lyric
  and they say nothing on a screen; picking by length lands on them every time.
- If no SINGLE word in the line carries, take the shortest PHRASE that does -- two
  to four words. Short matters less than legible.
- The test: printed alone on a poster, would it read as a statement, or as a
  fragment someone forgot to finish? If the second, it is the wrong choice.
- Short, but not at the cost of the above. Text on screen is read in a second or
  not at all -- and an empty word read in a second is still empty.
- Say where it sits, how it behaves and when it appears.
"""


_SHOWN_THE_CHARACTER = """
## YOU ARE BEING SHOWN THE SUBJECT

Reference image(s) are attached to this message. They are not mood or inspiration:
they ARE the subject, and the same images are handed to the video model as
references, so what you write here has to match what it will be rendering.

- Describe WHAT YOU SEE. Build, face, hair, clothing, markings, palette. Do not
  invent an appearance and do not improve on the one in front of you.
- Every detail you write is rendered in every chunk. A feature you add that is not
  in the image gets generated as though it were real, in all of them.
- If several images are attached they are the SAME subject, from different angles or
  moments. Describe the person, not one of the photographs, and do not define a
  second <Subject> for what is only a different shot of the first.
- Where the image and the brief disagree about how someone looks, THE IMAGE WINS.
  The brief says what happens; the image says who it happens to.
"""


class MMH3MusicScenePlanPrompt(io.ComfyNode):
    """System prompt for one stage of music-video prompt building."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3MusicScenePlanPrompt",
            display_name="MMH3 Music Scene Plan Prompt",
            category="MMH3Tools/prompt",
            description=(
                "Build N chunk prompts for a MUSIC VIDEO section by section. Same "
                "three stages as MMH3 Scene Plan Prompt, different rules: the arc is "
                "the song's rather than invented, the sung words and their timings "
                "are supplied rather than guessed, typography is rationed once across "
                "the whole song, and a window with nothing sung in it gets its own "
                "branch."
            ),
            inputs=[
                io.Combo.Input(
                    "stage", options=STAGES, default="definitions",
                    tooltip="'definitions' writes the film-wide sections ONCE. 'beats' "
                            "writes all N summaries together and assigns typography. "
                            "'shots' expands ONE chunk using its own verbatim lyrics "
                            "and word onsets."),
                io.String.Input(
                    "brief", multiline=True, default="",
                    tooltip="What the video is about. Dramatised, never narrated."),
                io.Int.Input(
                    "chunk_count", default=8, min=1, max=64,
                    tooltip="Wire MMH3 Window Plan's window_count."),
                io.Float.Input(
                    "seconds_per_chunk", default=19.3, min=0.2, max=150.0, step=0.1,
                    tooltip="Bounds the shot timestamps and tells the writer how much "
                            "fits in a chunk."),
                io.Combo.Input(
                    "typography", options=TYPOGRAPHY, default="off",
                    tooltip="'off' never puts words on screen. 'exact lyrics' quotes "
                            "the sung line verbatim. 'text bursts' allows fragments "
                            "and re-spellings drawn from it -- invention grounded in "
                            "the real words rather than replacing them."),
                io.Int.Input(
                    "beat_index", default=0, min=0, max=63, optional=True,
                    tooltip="'shots' only: which chunk this call writes, 0-based."),
                io.String.Input(
                    "beat_sheet", multiline=True, default="", optional=True,
                    tooltip="'shots' only: the full pipe-separated beat sheet."),
                io.String.Input(
                    "definitions", multiline=True, default="", optional=True,
                    tooltip="The definitions text, so labels exist and none are "
                            "invented."),
                io.String.Input(
                    "lyrics", multiline=True, default="", optional=True,
                    tooltip="'beats': the whole sectioned lyric. 'shots': MMH3 Lyrics "
                            "to Windows' `lyrics` for THIS chunk, already on the "
                            "chunk's clock."),
                io.String.Input(
                    "context_lyrics", multiline=True, default="", optional=True,
                    tooltip="'shots' only: the neighbouring windows' lines, for "
                            "continuity. Wire prev_lyrics and next_lyrics through a "
                            "concat, or either alone."),
                io.String.Input(
                    "section", default="", optional=True,
                    tooltip="'shots' only: MMH3 Lyrics to Windows' `section`, which "
                            "names a boundary falling inside the chunk."),
                io.String.Input(
                    "shot_times", multiline=True, default="", optional=True,
                    tooltip="'shots' only: MMH3 Lyrics to Windows' `shot_times` -- the "
                            "word onsets the writer should cut on instead of "
                            "inventing timestamps."),
                io.Boolean.Input(
                    "has_lyrics", default=True, optional=True,
                    tooltip="'shots' only: wire MMH3 Lyrics to Windows' `has_lyrics`. "
                            "False switches to the instrumental branch, which forbids "
                            "singing and on-screen text and asks for visual event "
                            "instead."),
                io.String.Input(
                    "extra_rules", multiline=True, default="", optional=True,
                    tooltip="Appended verbatim as a final block."),
                io.Boolean.Input(
                    "reference_images", default=False, optional=True,
                    tooltip="Turn on when reference image(s) are wired to the "
                            "LlamaGenerate running the DEFINITIONS stage. Without it "
                            "the model is handed pictures with no instruction about "
                            "them and describes an invented character anyway. Tells it "
                            "the images ARE the subject, that the same images go to "
                            "the video model, that several images are one person from "
                            "different angles, and that the image beats the brief on "
                            "appearance. Needs a vision-capable model on that call."),
                io.Combo.Input(
                    "music_source", options=MUSIC_SOURCE, default="supplied",
                    optional=True,
                    tooltip="'supplied' (default): the track already exists and is "
                            "handed to the sampler, so non_diegetic_music DESCRIBES "
                            "it and overall_soundscape says the track carries the "
                            "sound.\n\n"
                            "'generated': H3 writes the audio in the same pass. "
                            "non_diegetic_music becomes the SPEC the model performs, "
                            "overall_soundscape stops claiming a track was provided, "
                            "and the sung words must be quoted as "
                            "<d>[English] ...</d> in detailed_description. Without "
                            "that the model sings nothing and the mouth moves to "
                            "silence."),
                io.Combo.Input(
                    "treatments", options=TREATMENTS, default="music video",
                    optional=True,
                    tooltip="'music video' (default): split frames, RGB channel "
                            "split, the full vocabulary.\n\n"
                            "'restrained': one image per shot, no frame division, no "
                            "multiplied performer. For a piece whose subject is the "
                            "performance itself -- a split frame halves the singer "
                            "exactly when the mouth is the point."),
            ],
            outputs=[
                io.String.Output(display_name="system_prompt"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, stage, brief, chunk_count, seconds_per_chunk, typography,
                beat_index=0, beat_sheet="", definitions="", lyrics="",
                context_lyrics="", section="", shot_times="", has_lyrics=True,
                extra_rules="", reference_images=False, music_source="supplied",
                treatments="music video") -> io.NodeOutput:
        n = max(1, int(chunk_count))
        secs = float(seconds_per_chunk)
        generated = music_source == "generated"
        notes, parts = [], []

        # Only definitions is shown the image, deliberately: the description is
        # written once and reused verbatim, and re-deriving it per call is how the
        # subject drifts between chunks.
        if reference_images and stage != "definitions":
            notes.append("reference_images is on for the %s stage, where it does "
                         "nothing -- only definitions is shown the image, so the "
                         "description is written once instead of re-derived" % stage)

        if stage == "definitions":
            parts.append(_DEFINITIONS)
            if reference_images:
                parts.append(_SHOWN_THE_CHARACTER)
            else:
                notes.append("no reference_images, so subject_definitions describes an "
                             "INVENTED character; wire the image to this stage's "
                             "LlamaGenerate and turn this on")

        elif stage == "beats":
            typo = ""
            if typography != "off":
                typo = _TYPO_BEATS % (n, _TYPO_EXACT if typography == "exact lyrics"
                                      else _TYPO_BURST)
            parts.append(_BEATS % (n, n, n - 1, secs, typo))
            if not (lyrics or "").strip():
                notes.append("no lyrics given, so the beat sheet cannot follow the "
                             "song; wire the sectioned lyric or the alignment's lines")

        else:
            i = max(0, min(int(beat_index), n - 1))
            if int(beat_index) != i:
                notes.append("beat_index %d outside 0..%d; clamped to %d"
                             % (int(beat_index), n - 1, i))
            sung = (lyrics or "").strip()
            if has_lyrics and not sung:
                raise ValueError(
                    "MMH3MusicScenePlanPrompt: has_lyrics is true but `lyrics` is "
                    "empty. The shots stage writes against the words sung in THIS "
                    "chunk -- without them it invents them, which is the failure this "
                    "node exists to remove. Wire MMH3 Lyrics to Windows, or set "
                    "has_lyrics false for an instrumental window.")

            branch = _INSTRUMENTAL if not has_lyrics else ""
            # Without this the words are never asked for: the writer treats the
            # lyrics as timing for the picture and H3 sings nothing.
            if generated and has_lyrics:
                branch += _SUNG_GENERATED
            typo_block = ""
            if has_lyrics and typography == "exact lyrics":
                typo_block = _TYPO_SHOTS_EXACT + _TYPO_TREATMENT
            elif has_lyrics and typography == "text bursts":
                typo_block = _TYPO_SHOTS_BURST + _TYPO_TREATMENT

            parts.append(_SHOTS % (
                i + 1, n, section.strip() or "an unnamed section", secs, secs,
                branch, typo_block,
                sung or "(nothing is sung in this chunk)",
                (shot_times or "").strip() or "(none)"))
            if (context_lyrics or "").strip():
                parts.append(_SHOTS_CONTEXT % context_lyrics.strip())
            if (beat_sheet or "").strip():
                parts.append("=== THE BEAT SHEET ===\n\n%s" % beat_sheet.strip())
            else:
                notes.append("no beat_sheet, so this chunk cannot see the song's shape "
                             "or what typography it was assigned")
            if has_lyrics and not (shot_times or "").strip():
                notes.append("no shot_times, so the writer has no onsets to cut on and "
                             "will invent timestamps")

        if (brief or "").strip():
            parts.append(_BRIEF % brief.strip())

        if (definitions or "").strip() and stage != "definitions":
            parts.append("=== LABELS ALREADY DEFINED (use these, invent none) ===\n\n%s"
                         % definitions.strip())
        elif stage != "definitions":
            notes.append("no `definitions`, so the writer may invent labels the "
                         "assembled prompt does not define")

        if stage == "beats" and (lyrics or "").strip():
            parts.append("=== THE WHOLE LYRIC, WITH SECTIONS AND TIMES ===\n\n%s"
                         % lyrics.strip())

        if (extra_rules or "").strip():
            parts.append(extra_rules.strip())

        system = "\n\n".join(parts)
        system = system.replace(
            "@@AUDIO_RULES@@", _AUDIO_GENERATED if generated else _AUDIO_SUPPLIED)
        system = system.replace(
            "@@TREATMENTS@@",
            _TREAT_RESTRAINED if treatments == "restrained" else _TREAT_MUSIC_VIDEO)
        system = system.replace(
            "@@TREATMENT_MENU@@",
            _TREAT_MENU_RESTRAINED if treatments == "restrained" else _TREAT_MENU_MV)
        report = ("stage: %s | %d chunk%s of %.1fs | typography: %s | "
                  "audio: %s | treatments: %s%s\n%s"
                  % (stage, n, "" if n == 1 else "s", secs, typography,
                     music_source, treatments,
                     (" | chunk %d of %d%s"
                      % (min(int(beat_index), n - 1) + 1, n,
                         "" if has_lyrics else " (INSTRUMENTAL)"))
                     if stage == "shots" else "",
                     "\n".join("  ! " + x for x in notes) if notes
                     else "  no warnings"))
        logging.info("[MMH3MusicScenePlanPrompt] %s", report.splitlines()[0])
        return io.NodeOutput(system, report)
