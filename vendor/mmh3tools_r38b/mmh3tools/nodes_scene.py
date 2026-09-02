"""Build N chunk prompts SECTION BY SECTION instead of chunk by chunk.

WHY TRANSPOSE THE LOOP

Writing chunk i in isolation asks the model for a complete arc in every chunk. It
has no way to know it is the middle, so every chunk sets up, escalates and resolves
-- observed 2026-08-13 as five variants of the same scene, each ending on its own
climax. That is not a prompting failure; it is what the loop shape asks for.

Three things fall out of transposing it:

  * `subject_definitions` and `retention_analysis` MUST be byte-identical across
    chunks -- wording drift there IS character drift. Rewriting them N times
    guarantees drift. Written once and reused verbatim, drift becomes impossible.
  * ESCALATION lives across chunks, so it can only be planned where all N are
    visible at once. That is the beat sheet.
  * DIALOGUE planned per chunk repeats itself. Observed: the same closing line in
    three of five chunks. Planned across the set, it cannot.

And it costs FEWER calls, not more: 1 + 1 + N, against 2N for the chunk-by-chunk
shape. Eight chunks goes from 16 calls to 10.

THE THREE STAGES

  definitions  once   -> every FILM-WIDE section, reused verbatim: definitions,
                         retention, soundscape, score -- plus bare `summary:` and
                         `detailed_description:` headers for the loop to fill
  beats        once   -> all N summaries, the escalation ladder, pipe separated
  shots        N      -> one chunk's detailed_description, with the WHOLE beat sheet
                         in context and its own beat named

Soundscape and score are film-wide for the same reason the definitions are: a sound
world that drifts between chunks is audible drift. So the definitions call emits a
COMPLETE six-section skeleton, and assembly needs no new machinery -- per chunk,
`MMH3ReplaceSection` splices in that beat's summary and detailed_description, then
`MMH3PromptAccumulate` gathers the N results into the pipe-separated string
`MMH3ReferenceMultiPrompt` already consumes. (The skeleton must be complete:
`MMH3ReplaceSection` refuses to splice into a prompt missing sections, which is why
the bare headers are non-negotiable.)
"""

import logging

from comfy_api.latest import io

STAGES = ["definitions", "beats", "shots"]

_BRIEF = """=== HOW TO READ THE BRIEF ===

The BRIEF below tells you what the film is about so you can build it. It is not
dialogue and not narration.

- No character may speak it, restate it, or refer to it.
- If a line of dialogue could double as a summary of the brief, it is wrong.
- Dramatise it. Do not announce it.

=== BRIEF ===

%s"""

_DEFINITIONS = """You are writing the FILM-WIDE sections of a MiniMax H3 prompt, once,
for an entire film. They will be reused BYTE-IDENTICALLY in every chunk of it, so they
must be complete and final now.

Your reply is these six headers, in exactly this order:

    subject_definitions:
    summary:
    retention_analysis:
    detailed_description:
    overall_soundscape:
    non_diegetic_music:

FOUR of them YOU WRITE, in full, right now: subject_definitions, retention_analysis,
overall_soundscape and non_diegetic_music. Writing those IS the job. A reply that
returns them blank is a failed reply, and everything downstream is built on top of it.

TWO of them are per-chunk and get filled in later: summary and detailed_description.
Emit ONLY the bare header for those two, nothing after it, in place and in order. A
header that is missing cannot be filled in later.

Emit each header EXACTLY ONCE. No preamble, no commentary, no code fences, and no
headers besides these six.

## subject_definitions - one line per label

    <Subject N>  reusable visible content: a person, animal, object, scene, costume,
                 style, action or pose. Use this for anything that APPEARS.
    <Picture N>  standalone ONLY when the image is itself a concrete frame anchor.
                 If it merely defines a character or a style, cite it INSIDE the
                 <Subject N> line and give it no line of its own.
    <Video N>    ONLY whole-video relationships: editing, continuing from, or
                 borrowing camera movement / cuts / rhythm.
    <Audio N>    an audio asset. If it maps to a speaker, bind it with the speaker
                 id: <Subject N> (S1).

Labels are 1-based per type and numbered independently.

- DEFINE EVERY LABEL YOU WILL USE, AND USE EVERY LABEL YOU DEFINE. A label defined
  and never referenced is dead weight; a label referenced and never defined is a
  dangling tag the model cannot resolve.
- Describe what is VISIBLE and permanent about each: build, hair, markings. Not mood,
  not backstory, not what they want.
- STATE WHAT EVERY PERSON IS WEARING, head to foot, in their <Subject N> line --
  garment, cut, colour, fabric, and anything worn on top of it. This is not optional
  and not a detail to leave to the shot. subject_definitions is the ONLY section
  repeated byte-identically in every chunk; wardrobe established anywhere else is
  invisible to every chunk but the one that wrote it, and the subject changes clothes
  partway through the film.
- These must cover every chunk of the film, not just its opening. Anything that
  appears later still gets defined here.

## retention_analysis - ONE LINE PER LABEL, no exceptions

Each line names the label, picks ONE marker, and says in a clause what it means here.

    <Subject 1>: fully_preserved - her build, hair and suit are identical throughout.
    <Audio 1>: reference - only the voice timbre is borrowed, not the recording.

Visible markers: fully_preserved, partially_preserved, attribute_transfer,
weak_reference. Audio markers: fully_copy, partially_copy, reference, weak_reference.

- Those are values to CHOOSE BETWEEN. Never list them.
- A label with no line makes the section useless. Count them before you finish.

## overall_soundscape - the world's sound, for the whole film

Diegetic sound only: room tone, weather, machinery, footsteps, the acoustic of the
space. This is also reused in every chunk, so describe what is CONSTANT about the
sound world, not a moment of it. No music here.

## non_diegetic_music - score, or the absence of it

One or two sentences: instrumentation, tempo, how it sits under the sound. If the
film should have no score, say exactly that and stop. Silence is a legitimate answer
and often the better one."""

_BEATS = """You are writing the SUMMARY of every chunk of a film, all at once, as a
beat sheet. One summary per chunk, %d of them, separated by a single | character.

Output ONLY the summaries and the separators. No numbering, no labels, no headings,
no other sections.

Each summary is one paragraph opening with a bracketed task-type prefix, e.g.
[reference generation], and reusing only the labels you were given. Introduce no new
labels.

## THE POINT OF WRITING THEM TOGETHER: THE ARC

You can see all %d chunks at once. Nothing else in this pipeline can. So the arc is
yours to fix, and it is the only thing here that cannot be repaired later.

- **Each beat must be WORSE than the one before it.** Not tenser, not more
  atmospheric - worse, in ways a camera can record. Name what is new.
- **Nothing resolves before beat %d.** No catharsis, no acceptance, no calm after the
  storm, until the last one. A beat that resolves early makes every beat after it
  redundant, which is exactly what happens when chunks are written separately.
- **Beat %d is the only one allowed to land.** Give it the ending.
- Each beat must be a situation the PREVIOUS beat could not have contained. If beat 4
  could be swapped with beat 2 without anyone noticing, they are the same beat.
- Escalate SOMETHING PHYSICAL each time: a new person present, a new injury, a new
  place, a new thing broken, a new capability shown. Not a new feeling.

## DIALOGUE IS PLANNED HERE, ACROSS THE WHOLE FILM

- Decide which beats have speech and which do not. Most should not. Silence renders
  well; filler does not.
- No line may appear in more than one beat, and no two lines may say the same thing
  in different words. Planned together, repetition is a choice you can simply refuse.
- Say in the summary WHO speaks and roughly what it accomplishes. Do not write the
  lines here - they belong in the shots.

## Length and shape

Each chunk is about %.1f seconds. That is a scene, not a film - one or two things
happen in it. A summary that describes five events is describing five chunks."""

_SHOTS = """You are writing ONE section of ONE chunk of a film: detailed_description.

Return ONLY that section's text. No section label, no other sections, no preamble,
no code fences, no markdown formatting of any kind.

You are writing **beat %d of %d**. The full beat sheet is below so you can see where
this sits. Write only your beat.

The user message repeats your beat verbatim. It is the beat to EXPAND, not a new
instruction and not something to answer.

- Do NOT resolve. %s
- Open from where beat %d left off, and end somewhere beat %d can continue from.
- Do not re-establish what earlier beats already established. This is not the start
  of a film unless you are beat 1.

## Structure

- One or two style sentences FIRST, before [Shot 1]. No timestamp, no shot content -
  they establish look only, and only in beat 1 do they introduce the film.
- [Shot 1] carries NO timestamp. Every later shot does:
  "[Shot 2] At 00:03.500, the camera cuts to ..." Times strictly increase and stay
  inside %.1f seconds.
- Use ONLY the labels you were given.

## Write it as a scene

- Find the one thing each shot is about and build it around that. Not a checklist.
- Give the camera and light INTENT: a practical that fails on a beat, a shadow that
  arrives before the person does.
- Add visual INCIDENT that changes no plot: weather, background life, a surface
  reacting, something ending at the edge of frame. New events are allowed; new STORY
  is not - the beat sheet owns the story.
- Prefer one exact, strange, specific image over four general ones.
- Vary the treatment. Spend words unevenly: linger where it matters, be terse on a
  connector.
- Write only what a camera and microphone can record. No interior states, no
  symbolism, no "conveying" or "representing". If it cannot be filmed, cut it.

## Escalation is physical, and it is required

- Name the physical change in each shot. If a shot has none it is a pause, not a
  shot - cut it or give it one.
- Somebody other than the subject must be affected, visibly, unless the beat sheet
  says this beat is solitary.
- BANNED: a beat that resolves into conversation. A beat whose only change is
  lighting or mood.

## Dialogue

The BANALITY RULE APPLIES TO SPEECH ONLY. What people SAY is ordinary. What HAPPENS
is not. Never write a banal scene - write banal speech over an escalating one.

- Write only the lines the beat sheet assigns to this beat. Add none.
- These people are at work. Let them talk like it: procedural, administrative,
  unremarkable. The most frightening line available is usually the most banal one
  that is still true.
- No character says what is visible, what they feel, or what is about to happen.
- No line answers the line before it directly.
- Keep lines SHORT. Past about twelve words a line is explaining something.
- Identity, action and delivery go OUTSIDE <d>; only the language tag and the spoken
  words go inside:  The woman (S1) says: <d>[English] I'll send it tonight.</d>
- NEVER put spoken lines in double quotes - quotes mean text shown ON SCREEN.
- Target %.0f-%.0f words.

=== THE BEAT SHEET ===

%s"""


_CONTINUITY = """=== CONTINUITY: CONTINUE FROM THE PREVIOUS CHUNK ===

Below is the PREVIOUS chunk's detailed_description. Your [Shot 1] opens on its
FINAL frame -- the same positions, poses, point of contact, injuries, props,
wardrobe state and camera it ended on -- and ADVANCES from there. Do not
re-describe it, and do not re-open the scene with fresh establishing atmosphere;
only beat 1 introduces the film, every later beat is a continuation of this.

=== PREVIOUS CHUNK ===

%s"""


_SHOTS_TH = """You are writing ONE chunk of a single, ABSOLUTELY LOCKED, CONTINUOUS
talking-head shot: detailed_description. Return ONLY that section's text -- no section
label, no other sections, no preamble, no code fences, no markdown.

This is chunk %d of %d of ONE unbroken take of a person speaking straight to camera.
There is no scene, no arc, no escalation, and NO EDIT. The only things that change chunk
to chunk are what is said and the ordinary life of a person talking. %s

## THE CAMERA DOES NOT MOVE

- A fixed tripod. Same framing for the entire film: same shot size, same angle, same
  lens, the subject in the same place in frame, same background, same lighting, from
  first frame to last. NO cut, NO push-in or pull-out, NO pan or tilt, NO reframe, NO
  handheld drift. If an instruction implies the camera moving, it is wrong.
- BEGIN the section with [Shot 1], carrying NO timestamp. There is exactly ONE shot
  in this chunk and in the whole film. Never write a [Shot 2]: a second shot is an
  edit, and there are no edits.
- Nobody enters, nothing is introduced or knocked over, nothing happens in the room. A
  talking head is still on purpose; the words carry it. There is no story to advance.
- APPEARANCE COMES FROM THE DEFINITION. Do not introduce, change or re-describe
  anything permanent -- wardrobe, hair, build, markings. You may let the action touch
  it (a sleeve pushed back, hair moved off the face), but a detail stated here and
  nowhere else exists for this chunk only, and the next chunk will contradict it.

## THE PERSON IS ALIVE, NOT A STATUE

- Natural micro-life only, and only what a camera records: blinks, breath, small head
  tilts, a brow raised, a glance away and back, hand gestures a person makes while
  speaking. Keep every one consistent with the subject's definition.
- No interior states, no "conveys" or "represents". If it cannot be filmed, cut it.

## DIALOGUE IS THE ENTIRE CONTENT

- Write the lines this chunk speaks and CONTINUE the monologue -- pick up mid-thought
  from where the previous chunk ended (shown below) and never repeat a line or a point
  already made.
- The subject speaks ABOUT the topic. Dramatise a real train of thought aloud; do not
  announce or read out a summary of it.
- Identity and delivery go OUTSIDE <d>; only the language tag and spoken words go
  inside:  The woman (S1) says: <d>[English] ...</d>
- NEVER put spoken lines in double quotes -- quotes mean text shown ON SCREEN.
- Natural speech: contractions, mid-sentence turns, the rhythm of thinking aloud. Fill
  roughly %.0f-%.0f words of spoken content for this chunk's length -- a talking head
  that goes quiet is dead air."""


_BRIEF_TH = """=== WHAT THE SUBJECT IS TALKING ABOUT ===

The monologue is about the following. The subject speaks their own thoughts about it, as
a real person thinking aloud -- not reading it out and not announcing a summary of it,
but their speech IS about this.

%s"""


_CONTINUITY_TH = """=== CONTINUITY: THIS IS THE SAME UNBROKEN SHOT ===

Below is the PREVIOUS chunk's detailed_description. This chunk is the SAME continuous
locked take: identical framing, subject position, lighting and background, and the
subject carries on from the exact pose and mid-gesture the previous chunk ended on. Do
NOT cut, do not re-establish, do not re-block, do not move the camera. Continue the
monologue from where it stopped and do not repeat what was already said.

=== PREVIOUS CHUNK ===

%s"""


def _prev_body(text):
    """The `detailed_description` out of whatever `prev_detailed` was handed.

    The continuity block promises the model "below is the PREVIOUS chunk's
    detailed_description", but the thing the pack tells you to wire there --
    MMH3PromptAccumulate's `prior_context` -- emits a whole SIX-SECTION prompt,
    because it returns the last piece split on the `|` separator. Pasting that under
    a label saying it is one section shows the writer a complete prompt and asks it
    for a section: observed 2026-08-17, the model matched the format it was shown and
    returned a full prompt, which MMH3ReplaceSection then spliced INTO
    detailed_description, nesting a prompt inside its own body. Chunk 0 was clean
    because it has no prior context to imitate.

    Pulling the one section also stops the payload growing: re-sending every earlier
    prompt in full is the bloat `prior_context` exists to avoid.

    A bare body (no headers) is passed through -- that is already the right thing.
    """
    text = (text or "").strip()
    if not text:
        return "", None
    from .nodes_lint import _SECTIONS_A, _SECTIONS_B, _section
    following = list(dict.fromkeys(_SECTIONS_B + _SECTIONS_A))
    body = _section(text, "detailed_description", following)
    if not (body or "").strip():
        return text, None
    return body.strip(), ("prev_detailed held a full prompt (%d chars); used its "
                          "detailed_description (%d chars) as the continuity example, "
                          "since showing the writer a whole prompt is what makes it "
                          "return one" % (len(text), len(body.strip())))


class MMH3ScenePlanPrompt(io.ComfyNode):
    """System prompt for one stage of section-by-section prompt building."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3ScenePlanPrompt",
            display_name="MMH3 Scene Plan Prompt",
            category="MMH3Tools/prompt",
            description=(
                "System prompt for building N chunk prompts SECTION BY SECTION rather "
                "than chunk by chunk: shared definitions once, the whole beat sheet "
                "once, then each chunk's shots with the arc in view. Fixes chunks that "
                "each resolve on their own, definitions that drift, and dialogue that "
                "repeats."
            ),
            inputs=[
                io.Combo.Input(
                    "stage", options=STAGES, default="definitions",
                    tooltip="'definitions' writes subject_definitions + "
                            "retention_analysis ONCE for the whole film. 'beats' writes "
                            "all N summaries together -- this is where escalation is "
                            "decided. 'shots' expands ONE beat into "
                            "detailed_description with the whole beat sheet in view.",
                ),
                io.String.Input(
                    "brief", multiline=True, default="",
                    tooltip="The premise. Framed for the model as a brief: something to "
                            "dramatise, never to be spoken aloud by a character.",
                ),
                io.Int.Input(
                    "chunk_count", default=4, min=1, max=64,
                    tooltip="How many chunks the film is. Wire MMH3 Window Plan's "
                            "window_count so the beat count matches the render.",
                ),
                io.Float.Input(
                    "seconds_per_chunk", default=8.0, min=0.2, max=150.0, step=0.1,
                    tooltip="Used to tell the writer how much fits in one beat, and to "
                            "bound the shot timestamps.",
                ),
                io.Int.Input(
                    "beat_index", default=0, min=0, max=63, optional=True,
                    tooltip="'shots' stage only: which beat this call writes, 0-based.",
                ),
                io.String.Input(
                    "beat_sheet", multiline=True, default="", optional=True,
                    tooltip="'shots' stage only: the FULL pipe-separated beat sheet from "
                            "the 'beats' stage. Every shots call gets all of it -- that "
                            "is what stops a chunk resolving on its own.",
                ),
                io.String.Input(
                    "definitions", multiline=True, default="", optional=True,
                    tooltip="'beats' and 'shots' stages: the definitions text, so the "
                            "writer uses labels that exist and invents none.",
                ),
                io.String.Input(
                    "extra_rules", multiline=True, default="", optional=True,
                    tooltip="Appended verbatim as a final block.",
                ),
                io.String.Input(
                    "prev_detailed", multiline=True, default="", optional=True,
                    tooltip="'shots' stage: the PREVIOUS chunk's detailed_description, "
                            "wired from MMH3 Prompt Accumulate's prior_context (mode "
                            "'last'). Appended as a continuation block so the writer "
                            "opens this chunk on that chunk's final frame and advances, "
                            "rather than opening a fresh scene. Empty on beat 0.",
                ),
                io.Combo.Input(
                    "mode", options=["cinematic", "talking_head"], default="cinematic",
                    optional=True,
                    tooltip="'cinematic' (default): an escalating scene, each beat worse "
                            "than the last. 'talking_head': ONE absolutely-locked "
                            "continuous shot of a person speaking -- the shots stage "
                            "holds the frame and continues the monologue instead of "
                            "escalating, and does not require a beat_sheet.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="system_prompt"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, stage, brief, chunk_count, seconds_per_chunk, beat_index=0,
                beat_sheet="", definitions="", extra_rules="", prev_detailed="",
                mode="cinematic") -> io.NodeOutput:
        n = max(1, int(chunk_count))
        secs = float(seconds_per_chunk)
        notes = []
        parts = []

        if stage == "definitions":
            parts.append(_DEFINITIONS)
        elif stage == "beats":
            parts.append(_BEATS % (n, n, n, n, secs))
            if n == 1:
                notes.append("chunk_count is 1, so there is no arc to plan; the beats "
                             "stage degenerates to a single summary.")
        else:
            i = max(0, min(int(beat_index), n - 1))
            if int(beat_index) != i:
                notes.append("beat_index %d is outside 0..%d; clamped to %d."
                             % (int(beat_index), n - 1, i))
            if mode == "talking_head":
                establish = ("This is the FIRST chunk: establish the subject speaking to "
                             "camera and begin the monologue."
                             if i == 0 else
                             "This continues the unbroken take -- do NOT re-establish or "
                             "re-introduce anything; pick up mid-thought from the "
                             "previous chunk shown below. The take is ALREADY RUNNING: "
                             "this chunk does not open on anything. No fade in, no cut "
                             "from black, no light coming up, no camera settling, no "
                             "reveal. Its first frame continues the previous frame.")
                # ~2-3.5 spoken words/sec; a locked talking head is carried by speech
                lo, hi = int(secs * 2), int(secs * 3.5)
                parts.append(_SHOTS_TH % (i + 1, n, establish, lo, hi))
            else:
                sheet = (beat_sheet or "").strip()
                if not sheet:
                    raise ValueError(
                        "MMH3ScenePlanPrompt: the 'shots' stage needs `beat_sheet`. "
                        "Without it the writer cannot see where this chunk sits, and "
                        "every chunk writes its own complete arc -- the exact failure "
                        "this node exists to remove. (talking_head mode does not need "
                        "one.)")
                last = i == n - 1
                resolve = ("This is the LAST beat. It is the only one allowed to land, "
                           "so give it the ending." if last else
                           "Beats after this one still have to happen. End it worse than "
                           "you found it, not settled.")
                # a dialogue-heavy beat needs room; the format guide says fitting the
                # spoken timeline beats hitting a word count
                lo, hi = (250, 400) if secs <= 10 else (350, 500)
                parts.append(_SHOTS % (i + 1, n, resolve, i, i + 2, secs, lo, hi, sheet))

        if (brief or "").strip():
            if mode == "talking_head" and stage == "shots":
                parts.append(_BRIEF_TH % brief.strip())
            else:
                parts.append(_BRIEF % brief.strip())
        elif stage == "shots" and mode == "talking_head":
            notes.append("talking_head shots with no brief -- the subject has no topic "
                         "to talk about; wire the premise into `brief`.")
        elif stage != "shots":
            notes.append("no brief given; the writer has only the labels to work from.")

        if (definitions or "").strip() and stage != "definitions":
            parts.append("=== LABELS ALREADY DEFINED (use these, invent none) ===\n\n%s"
                         % definitions.strip())
        elif stage != "definitions":
            notes.append("no `definitions` wired, so the writer may invent labels that "
                         "the assembled prompt does not define.")

        if (extra_rules or "").strip():
            parts.append(extra_rules.strip())

        if stage == "shots":
            prev, prev_note = _prev_body(prev_detailed)
            if prev_note:
                notes.append(prev_note)
            if prev:
                parts.append((_CONTINUITY_TH if mode == "talking_head"
                              else _CONTINUITY) % prev)
            elif i > 0:
                notes.append("no `prev_detailed` wired for beat %d, so it opens cold "
                             "instead of continuing the previous chunk's final frame -- "
                             "wire MMH3 Prompt Accumulate's prior_context (mode 'last')."
                             % i)

        system = "\n\n".join(parts)
        report = ("stage: %s | %d chunk%s of %.1fs%s\n%s"
                  % (stage, n, "" if n == 1 else "s", secs,
                     (" | beat %d of %d" % (min(int(beat_index), n - 1) + 1, n))
                     if stage == "shots" else "",
                     "\n".join("  ! " + x for x in notes) if notes else "  no warnings"))
        logging.info("[MMH3ScenePlanPrompt] %s", report.splitlines()[0])
        return io.NodeOutput(system, report)


class MMH3PromptPart(io.ComfyNode):
    """Take the i-th piece of a separated string -- one beat out of a beat sheet."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3PromptPart",
            display_name="MMH3 Prompt Part",
            category="MMH3Tools/prompt",
            description=(
                "Split a separated string and emit one piece by index, plus the count. "
                "The join between a beat sheet written all at once and a loop that "
                "renders one beat per iteration."
            ),
            inputs=[
                io.String.Input(
                    "text", multiline=True, default="",
                    tooltip="The separated string, e.g. a beat sheet from MMH3 Scene "
                            "Plan Prompt's 'beats' stage.",
                ),
                io.Int.Input(
                    "index", default=0, min=0, max=63,
                    tooltip="Which piece, 0-based. Drive from a for-loop index.",
                ),
                io.String.Input(
                    "separator", default="|", optional=True,
                    tooltip="Same separator MMH3 Prompt Accumulate and MMH3 Reference "
                            "(Multi-Prompt) use, so the three agree by default.",
                ),
                io.Boolean.Input(
                    "clamp", default=True, optional=True,
                    tooltip="Past the end, repeat the LAST piece -- matching how the "
                            "looping sampler reuses the last cond for extra chunks. "
                            "Off raises instead, which is what you want if a count "
                            "mismatch should stop the run.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="part"),
                io.Int.Output(display_name="count"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, text, index, separator="|", clamp=True) -> io.NodeOutput:
        sep = separator or "|"
        body = (text or "").strip()
        # an LLM that ignored "no code fences" should not break the split
        if body.startswith("```"):
            lines = body.splitlines()
            body = "\n".join(lines[1:-1] if lines[-1].strip().startswith("```")
                             else lines[1:]).strip()
        pieces = [p.strip() for p in body.split(sep) if p.strip()]
        if not pieces:
            raise ValueError(
                "MMH3PromptPart: nothing to split -- the text is empty, or the writer "
                "did not use %r as a separator. Check the beats stage's output." % sep)

        i, notes = int(index), []
        if len(pieces) == 1:
            # One piece means the separator never appeared. With clamp on, every
            # index then returns the WHOLE text and each chunk is spliced with the
            # entire sheet instead of its own beat -- silently, and it looks like a
            # model failure downstream rather than a split failure here.
            logging.warning(
                "[MMH3PromptPart] the text did not split: no %r found, so all %d "
                "chars are being returned as a single piece. If this is a beat sheet "
                "or a lyric block, the writer dropped the separator.", sep, len(body))
            notes.append("NO SPLIT: %r never appeared -- this is the whole text, not "
                         "one piece of it" % sep)
        if i >= len(pieces):
            if not clamp:
                raise ValueError(
                    "MMH3PromptPart: index %d but only %d piece%s. The beat sheet has "
                    "fewer beats than the run has chunks."
                    % (i, len(pieces), "" if len(pieces) == 1 else "s"))
            notes.append("index %d past the end; repeating the last of %d."
                         % (i, len(pieces)))
            i = len(pieces) - 1

        part = pieces[i]
        report = ("piece %d of %d, %d chars\n%s"
                  % (i, len(pieces), len(part),
                     "\n".join("  ! " + x for x in notes) if notes
                     else "  clean split"))
        logging.info("[MMH3PromptPart] %s", report.splitlines()[0])
        return io.NodeOutput(part, len(pieces), report)
