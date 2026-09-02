"""Assemble a system prompt for your own LLM node, scoped to the selected task types.

H3 expects a structured prompt that MiniMax's hosted H3-Context-IR normally produces.
Running locally you write it yourself, and a single monolithic system prompt carries
rules that contradict each other across tasks - audio reuse wants the source's words
transcribed verbatim, audio reference wants them left out entirely. This emits only
the blocks that apply to the combination you pick.

Rules are drawn from docs/upstream/VIDEO_PROMPT_WRITING_GUIDE_{base,ref}_en.md.
"""

import logging
import re

from comfy_api.latest import io

from .common import FPS, FRAMES_PER_GROUP, FRAME_BASE

MODES = ["Ref2VA", "T2VA", "I2VA", "L2VA", "FL2VA"]
MASKED_AUDIO_KINDS = ["none", "background music", "speech", "sung lyrics"]
FORMAT_A = {"T2VA", "I2VA", "L2VA", "FL2VA"}

# every section name across both formats, for MMH3ReplaceSection's picker
_ALL_SECTIONS = ["detailed_description", "integrated_multimodal_description",
                 "subject_definitions", "summary", "retention_analysis",
                 "overall_soundscape", "non_diegetic_music"]

TASKS = [
    ("keyframe_completion", "keyframe completion"),
    ("reference_generation", "reference generation"),
    ("video_editing", "video editing"),
    ("video_continuation", "video continuation"),
    ("audio_reuse", "audio reuse"),
    ("audio_reference", "audio reference"),
]

_BASE = """You convert a rough video idea into the exact structured prompt format the MiniMax H3
video model expects. Output ONLY the prompt. No preamble, no commentary, no code fences.

Write everything in English EXCEPT dialogue and lyrics inside <d>, and text visibly shown
on screen, which keep their original language verbatim."""

_FMT_A = """## Format

{instruction}Then exactly three fields, blank line between each:

    integrated_multimodal_description: [Shot 1] <style>, <shot 1> [Shot 2] At 00:SS.mmm, the camera cuts to <shot 2> [Shot 3] At 00:SS.mmm, the camera cuts to <shot 3>

    overall_soundscape: <1-4 sentences covering the WHOLE clip>

    non_diegetic_music: <1-3 sentences covering the WHOLE clip, or N/A>

THREE field labels appear in your entire output, once each. EVERY shot lives inside
the single integrated_multimodal_description, one after another in that one field.
Never repeat a field label. Never emit a set of fields per shot -- there is no
per-shot audio field, and a repeated label makes the prompt unparseable.
overall_soundscape and non_diegetic_music describe the whole clip, so cover what is
heard across all of it rather than the current shot.

State the style ONCE, at the START of [Shot 1], and not again per shot: Live-action,
cinematic, 2D-animated, 3D CG, claymation, watercolour, vintage film. Pick one; a
list is not a style.

Length: roughly 100-150 words for a single-shot 5-8s clip; with several shots, about
40-70 words per shot.

Shots run in playback order, but that governs the ORDER OF SHOTS only. It does NOT
mean a shot is narrated strictly chronologically: inside a shot that has speech, the
<d> line comes FIRST and the action is hung off it. See "ORDER WITHIN A SHOT" in
Shared syntax.

There is NO summary section and NO task-type prefix in this format."""

_INSTR = {
    "I2VA": ("Begin with this line, then ONE blank line:\n\n"
             "    For the target video, at 0.00 seconds into the target video, <Picture 1> "
             "(from [Shot 1]) is fully referenced.\n\n"),
    "L2VA": ("Begin with this line, then ONE blank line:\n\n"
             "    How the reference pictures align with the target video - <Picture 1> "
             "(from [Shot N]) aligns with the S.SS-second mark of the target video.\n\n"
             "S.SS is the effective duration to exactly two decimals. Structure the body as: "
             "plausible preceding state -> action and transition path -> gradual convergence "
             "-> landing on the image at the end.\n\n"),
    "FL2VA": ("Begin with this line, then ONE blank line:\n\n"
              "    How the reference pictures align with the target video - Picture 1 (from "
              "Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from "
              "Shot N) aligns with the S.SS-second mark of the target video.\n\n"
              "S.SS is the effective duration to exactly two decimals. Prefer a SINGLE shot so "
              "the model can interpolate continuously. Describe the motion path between the two "
              "frames, not two static descriptions.\n\n"),
    "T2VA": "",
}

_FMT_B = """## Format

Six sections, this order, lowercase keys with colons, blank line between each:

    subject_definitions:
    summary:
    retention_analysis:
    detailed_description:
    overall_soundscape:
    non_diegetic_music:

subject_definitions - one line per referenced item tracked later.
  <Subject N>  reusable visible content: person, animal, object, scene, costume, style,
               action, pose. Use this for anything that APPEARS in the target.
  <Picture N>  standalone ONLY when the image is itself a concrete frame anchor or a
               storyboard. If it merely defines a character or style, cite it INSIDE the
               <Subject N> definition and give it no line of its own.
  <Video N>    ONLY whole-video relationships: editing, continuing from, or referencing
               its camera movement / cuts / rhythm.
  <Audio N>    an audio asset. When it maps to a speaker write <Subject N> (Sx), or a
               stable voice description followed by (Sx), reusing the target's speaker ID.

Labels are 1-BASED per type, numbered independently, in the order assets are supplied.
The same file can be <Video 1> and <Audio 2>.

summary - one paragraph opening with the bracketed task-type prefix below. Reuse existing
labels only; introduce none here.

retention_analysis - ONE LINE PER LABEL. Each line names the label, picks ONE marker, and
says in a clause what that means for this clip. Write lines like these:

    <Subject 1>: fully_preserved - her face and build are identical in every shot.
    <Audio 1>: fully_copy - the source track is the finished audio.

Pick the visible marker from fully_preserved, partially_preserved, attribute_transfer,
weak_reference. Pick the audio marker from fully_copy, partially_copy, reference,
weak_reference. Those are values to CHOOSE BETWEEN - never write a list of them into the
section, and never write "visible:" or "audio:" as lines. A label with no line, or a line
with no marker, makes the section useless.

detailed_description - the body. One or two style sentences BEFORE [Shot 1]. Then shot by
shot in playback order. Roughly 350-500 words; dialogue-heavy content prioritises fitting
the spoken timeline over word count.
  Playback order governs the ORDER OF SHOTS. It does NOT mean each shot is narrated
  strictly chronologically: inside a shot that has speech, the <d> line comes FIRST and
  the action is hung off it. See "ORDER WITHIN A SHOT" in Shared syntax."""

_SYNTAX = """## Shared syntax

- [Shot 1] has NO timestamp. Later shots: [Shot 2] At 00:03.500, the camera cuts to ...
  Cut times strictly increase and stay inside the duration.
- Cuts: "the camera cuts to", "the shot cuts to", "the shot transitions to". A cut must
  introduce new information; for a small change of distance or angle use camera motion.
- EVERY cut carries its own [Shot N] marker and timestamp. A cut phrase buried in a
  shot's prose - "then transitions to a close-up, then to the doll on the left" - is a
  HIDDEN CUT: the model has no time to place it and will not perform it. Either give it
  a numbered, timestamped shot of its own, or express it as camera motion within the
  current shot.
- Camera motion as natural English inside the shot: type + amplitude + speed.
  Zoom In/Out, Push In, Pull Out, Pan Left/Right, Truck Left/Right, Tilt Up/Down,
  Pedestal Up/Down, Arc Shot, Tracking Shot, Static Shot, Shake Slightly/Strongly, POV,
  Roll Clockwise/Counterclockwise; "with small/large amplitude", "at slow/fast speed".
  Omit amplitude and speed when unremarkable.
- Speakers get stable (S1), (S2) by order of vocal events in the TARGET. Joint speech
  (S1,S2). Characters who never vocalise get no ID. Establish identity on first
  appearance: type, age, on/off screen, pitch, timbre, pace, accent.
- Dialogue: identity, action and delivery go OUTSIDE <d>; only the language tag and the
  spoken words go inside.  The woman (S1) says: <d>[English] I almost didn't come.</d>
- ORDER WITHIN A SHOT: when a shot contains BOTH action and speech, lead with the
  speech and attach the action to it. Do not describe the action first and append the
  line afterwards.
    good:  The woman (S1) says: <d>[English] I almost didn't come.</d> as she crosses
           the room and sets her bag on the table.
    bad:   The woman crosses the room and sets her bag on the table. She (S1) says:
           <d>[English] I almost didn't come.</d>
  Both read the same to a person; they do not sound the same. A line trailing a run of
  action prose lands late and the audio around it degrades. Put the <d> block as early
  in the shot as the sense allows, then hang the action off it.
- NEVER wrap dialogue in double quotes. Double quotes mean text visible ON SCREEN, so a
  quoted spoken line asks for a sign instead of speech.
- Voiceover: use the exact phrase "says in an off-screen voiceover", then state that the
  on-screen character's lips remain closed.
- <scenetrans> at both connecting points when a line crosses a cut; <cutoff> when speech
  is truncated by the end of the video.
- On-screen text in double quotes, verbatim, untranslated: a neon sign reading "OPEN".
- overall_soundscape: 1-4 sentences of ambience, physical action sound and non-verbal
  human sound. No dialogue, no diegetic music. N/A only if silence is requested.
- non_diegetic_music: 1-3 sentences on instrumentation, tempo, rhythm, dynamics. Never
  mood words, never emotional function. Music characters can hear is diegetic and belongs
  in the body. N/A if none."""

_SUPPLIED_DIALOGUE = """## Supplied dialogue

The lines under DIALOGUE: are FIXED. Reproduce each one exactly once, in the order
given. Lines keep their chronological order relative to each other, but each one OPENS
the passage it belongs to - write the line first and hang its action off it, never the
action first with the line appended.

- Keep the wording verbatim. Do not translate, paraphrase, shorten, expand or "fix" it.
- Write NO dialogue that is not listed. If the duration leaves room, fill it with
  action, camera and ambience - never with extra lines.
- If the supplied lines will not fit, keep all of them and reduce the surrounding
  action instead. Say so in one sentence before the prompt.
- One <d> block per line, in order:  <d>[English] The words go here.</d>
  Replace [English] with that line's actual language.
- Inside the tag put ONLY the language tag and the spoken words. Speaker name, (Sx),
  delivery and stage direction all go OUTSIDE, before the tag.
- Place each line EARLY in its shot and hang the action off it, rather than describing
  the action and appending the line. "She says: <d>...</d> as she crosses the room"
  rather than "She crosses the room. She says: <d>...</d>" A line that trails a run of
  action prose lands late and the audio around it degrades.
- Never wrap dialogue in double quotes - double quotes mean text visible ON SCREEN, so
  a quoted spoken line asks for a sign instead of speech.
- Standardise punctuation to , . ? ! only. Strip emoji, tildes, long ellipses and
  decorative marks. End each line with . ? or ! before the closing </d>.

DIALOGUE:
%s"""

_MASKED_AUDIO_COMMON = """## Supplied audio track

AN AUDIO CLIP IS ATTACHED TO THIS REQUEST. Listen to it before you write anything.
Everything below depends on what is actually in it, and you cannot write the picture
correctly from the text idea alone.

The track is masked, so it arrives in the finished clip VERBATIM. It is not generated,
cannot be altered, and nothing written here changes it.

This inverts what the audio fields are for. overall_soundscape and non_diegetic_music
normally REQUEST sound; here they DESCRIBE sound that already exists. Their only job is
to tell the model what it is about to hear so the picture matches it.

- Describe only what is actually audible in the track. Asking for a sound that is not
  there does not add it - it just makes the video expect something that never arrives.
- Never describe the track as something to be created, added, or that "begins".
- The VIDEO is what is being generated. Every visible event must be consistent with the
  audio's own timing: action lands on what is audible, not on an invented rhythm."""

_MASKED_AUDIO = {
    "background music": """### The track is background music

- Put it in non_diegetic_music: instrumentation, tempo, rhythm, dynamics. Never mood
  words, never emotional function.
- If something VISIBLE produces the music - a radio, a busker, a live band - it is
  diegetic instead: describe the source and its placement in the body, and set
  non_diegetic_music: N/A.
- NOBODY SPEAKS. Emit no <d> block at all. Do not assign any (Sx). Keep mouths closed or
  occupied; a character shown mid-speech with no voice in the track reads as broken.
- Cut and move in sympathy with the music, but do not invent hits, stops or drops. If
  the track does not have them, a beat-matched edit will simply land on nothing.
- overall_soundscape covers ambience and physical action sound only, and only what is
  genuinely audible under the music.""",

    "speech": """### The track is speech

- FIRST, before writing anything else, listen to the attached audio and transcribe the
  spoken words. Do this as a step of its own - do not start composing the prompt and
  fit the words in as you go.
- Then write each line as <d>[Language] the words</d>, in the order heard, so the lips
  match what is audible. Each line OPENS its passage - write the line, then hang the
  action off it. Never describe the action first and append the line.
- Transcribe what is actually said. If a passage is unclear, transcribe what you can and
  leave the rest out rather than guessing - invented words are worse than missing ones,
  because the lips get animated to them.
- Assign (Sx) by order of vocal events and establish the speaker on first appearance:
  type, age, on/off screen, pitch, timbre, pace, accent - matching the voice in the
  track, not an invented one.
- State visible mouth movement explicitly, and keep the speaker's lips moving for the
  whole line. Silence in the picture over audible speech is the failure to avoid.
- Off-screen speech: use the exact phrase "says in an off-screen voiceover" and state
  that the on-screen character's lips remain closed.
- overall_soundscape describes the NON-speech content only - it must never contain
  dialogue - and must not contradict what is audible under the voice.
- non_diegetic_music: N/A unless music is genuinely present in the track.""",

    "sung lyrics": """### The track is sung

- FIRST, before writing anything else, listen to the attached audio and transcribe the
  SUNG WORDS. Do this as a step of its own and finish it before composing the prompt.
  Sung words are harder to make out than speech, especially over backing instruments -
  spend the effort here, not on the prose.
- If you cannot make out a passage, transcribe what you can and leave the rest out.
  Never substitute plausible lyrics: invented words get animated onto the mouth.
- Then write each line as <d>[Language] the words</d>, in the order heard, each one
  OPENING its passage with the action hung off it rather than appended after. Lyrics
  keep their ORIGINAL language verbatim, exactly like dialogue.
- Write that the character SINGS, never says. Assign (Sx) as for speech.
- Describe singing physically: sustained open vowels, held notes, breath before a
  phrase, jaw and throat movement. Sung mouth shapes differ from spoken ones and the
  model needs telling which it is.
- Instrumental backing that is audible in the track goes in non_diegetic_music, or is
  described in the body as diegetic if the players or the source are visible.
- overall_soundscape covers ambience and action sound only - never the vocal, never the
  backing.
- TIME THE CUTS TO THE VOICE. Put each cut on a breath, a phrase end or a rest that you
  can actually hear. Never cut in the middle of a sung word: the mouth is mid-vowel on
  both sides of the join and the jump is obvious however good the lipsync is. If a
  phrase runs long, hold the shot or move the camera within it rather than cutting.""",
}

_TASK_RULES = {
    "keyframe completion": """### keyframe completion
The image IS a concrete frame of the target, not guidance. Give it its OWN line in
subject_definitions as <Picture N> - do not fold it into a <Subject N>. In
retention_analysis note the frame role: "<Picture 1> ([Shot 1] first frame):
fully_preserved - ...". In the body use natural anchoring language: "the shot begins from
<Picture 1>", "the shot ends on <Picture 2>". Only first-frame and last-frame anchors
exist; there is no mid-clip anchor.""",

    "reference generation": """### reference generation
The asset GUIDES appearance, style, action, camera or storyboard - it is not a concrete
frame and not a source being edited or continued. Cite the image INSIDE the corresponding
<Subject N> definition; it gets NO standalone <Picture N> line. If an image acts as a
storyboard, state which shots it maps to and what planning information it provides.""",

    "video editing": """### video editing
The source video is DIRECTLY MODIFIED. Begin the summary, immediately after the task-type
prefix, with exactly:

    The target video is an edited version of <Video 1>.

Describe what changes and what is left alone. If the original audio remains audible, the
prefix must also include audio reuse.""",

    "video continuation": """### video continuation
New content CONTINUES, extends or resumes from the source video - none of the source's
timeline is reproduced in the target. State the relationship flatly in the summary, e.g.
"The target video continues from the final frame of <Video 1>."

Do NOT re-describe the source's scene in the body: a re-described setting is an
instruction to draw it again. One clause acknowledging the resume point, then all new
action. Carry the END STATE forward explicitly - shot size, camera motion still in
progress and its direction, subject pose - or the model will reset the framing.

Cite <Video N> in the body where the continuation relationship applies.""",

    "audio reuse": """### audio reuse
The audio SIGNAL becomes the target's audio. Transcribe the spoken words EXACTLY into <d>,
preserving wording and original language; write [unclear] for unintelligible spans rather
than guessing; standardise punctuation to , . ? ! and drop decorative marks. This
transcription drives lipsync timing and is not optional.
    fully_copy      the complete source audio is the complete final track
    partially_copy  only part of the timeline or some layers are copied, or sounds are
                    added / removed / replaced afterwards""",

    "audio reference": """### audio reference
Only timbre, delivery, rhythm, music style or texture is borrowed - the signal is NOT
copied. Marker is reference. Do NOT carry the source's dialogue into the target; write the
target's own lines.

The text encoder never receives the audio, only an <Audio N>: label placeholder, so
everything the model knows comes from your description. Define it concretely in
subject_definitions: what is said or played, voice type and pitch, pace, recording
character, roughly how long it runs.

An audio reference tends to SUPPRESS generated ambience. If you want room tone, rain,
traffic, state it explicitly and continuously in overall_soundscape.""",
}

_SCOPING = """## Choosing task types

The mere presence of an asset does NOT create a task type. A video supplying only camera
movement, cuts or rhythm is reference generation, not continuation. Use video editing or
video continuation only when that video is directly edited or continued.

Markers must agree with the prefix. Declaring audio reference while writing fully_copy is
self-contradictory."""


# role -> (task type, retention marker, standalone label or folded into a Subject)
_IMAGE_ROLES = {
    "character appearance": ("reference generation", "fully_preserved", False),
    "scene or style":       ("reference generation", "weak_reference", False),
    "first frame anchor":   ("keyframe completion", "fully_preserved", True),
    "last frame anchor":    ("keyframe completion", "fully_preserved", True),
    "storyboard":           ("reference generation", "weak_reference", True),
}
_VIDEO_ROLES = {
    "continuation source":      ("video continuation", "weak_reference", True),
    "editing source":           ("video editing", "fully_preserved", True),
    "motion or camera reference": ("reference generation", "weak_reference", True),
    "motion transfer":          ("reference generation", "attribute_transfer", True),
}
_AUDIO_ROLES = {
    "voice timbre":  ("audio reference", "reference", True),
    "music style":   ("audio reference", "reference", True),
    "sound texture": ("audio reference", "reference", True),
    "reuse signal":  ("audio reuse", "fully_copy", True),
}
_KINDS = [
    ("images", "Picture", _IMAGE_ROLES, 9),
    ("videos", "Video", _VIDEO_ROLES, 3),
    ("audios", "Audio", _AUDIO_ROLES, 3),
]


def _parse_assets(text, roles, kind_name):
    """'role: description' per line, in wiring order. Returns [(role, desc, warn)]."""
    out = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        role, _, desc = line.partition(":")
        key = role.strip().lower()
        if key not in roles:
            out.append((None, line, "unknown %s role %r - expected one of: %s"
                        % (kind_name, role.strip(), ", ".join(sorted(roles)))))
        else:
            out.append((key, desc.strip(), None))
    return out


def _achievable(seconds):
    """Nearest achievable duration on the 17j+5 frame grid."""
    target = seconds * FPS
    j = max(0, int((target - FRAME_BASE) // FRAMES_PER_GROUP))
    lo = FRAMES_PER_GROUP * j + FRAME_BASE
    hi = lo + FRAMES_PER_GROUP
    f = lo if (target - lo) <= (hi - target) else hi
    return f, f / FPS


class MMH3AssetPlan(io.ComfyNode):
    """Declare what each reference asset IS, so labels, task types and markers agree.

    Labels are positional - <Picture 1> means "whatever is wired first" - so the prompt
    and the wiring drift apart easily. One line per asset here, in the same order you
    wire them, and the role determines the rest deterministically:

        role -> task type      (which behaviour the model selects)
             -> marker         (what survives, in retention_analysis)
             -> standalone?    (its own <Picture N> line, or folded into a <Subject N>)

    That last one is the structural tell between keyframe completion and reference
    generation, and it is the easiest thing to get backwards by hand.

    Emits INSTRUCTIONS, not finished prose: subject_definitions stays in the model's
    voice, it just knows exactly what it is describing.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3AssetPlan",
            display_name="MMH3 Asset Plan",
            category="MMH3Tools/prompt",
            description=(
                "Declare reference assets as 'role: description', one per line in wiring "
                "order. Derives the task-type prefix, retention markers and label scheme."
            ),
            inputs=[
                io.String.Input("images", multiline=True, default="", optional=True,
                                tooltip="One per line, in the order wired to ref_image_*.\n"
                                        "role: description\n"
                                        "roles: character appearance | scene or style | "
                                        "first frame anchor | last frame anchor | storyboard"),
                io.String.Input("videos", multiline=True, default="", optional=True,
                                tooltip="One per line, in the order wired to ref_video_*.\n"
                                        "roles: continuation source | editing source | "
                                        "motion or camera reference | motion transfer"),
                io.String.Input("audios", multiline=True, default="", optional=True,
                                tooltip="One per line. A reference video's own soundtrack "
                                        "counts here too - it gets its own <Audio N>.\n"
                                        "roles: voice timbre | music style | sound texture | "
                                        "reuse signal"),
            ],
            outputs=[
                io.String.Output(display_name="inventory"),
                io.String.Output(display_name="task_prefix"),
                io.String.Output(display_name="retention_skeleton"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, images="", videos="", audios="") -> io.NodeOutput:
        parsed = {}
        warnings = []
        for key, label, roles, limit in _KINDS:
            text = {"images": images, "videos": videos, "audios": audios}[key]
            items = _parse_assets(text, roles, label)
            for _, _, w in items:
                if w:
                    warnings.append(w)
            if len(items) > limit:
                warnings.append("%d %s declared but the model accepts at most %d"
                                % (len(items), key, limit))
            parsed[key] = items

        total = sum(len(v) for v in parsed.values())
        if total > 12:
            warnings.append("%d assets total; the model accepts at most 12 files" % total)
        if total == 0:
            warnings.append("no assets declared - this is a text-only task, so Ref2VA and "
                            "task types do not apply")

        lines, retention, tasks = [], [], []
        for key, label, roles, _ in _KINDS:
            for i, (role, desc, _) in enumerate(parsed[key], start=1):   # labels are 1-BASED
                tag = "<%s %d>" % (label, i)
                if role is None:
                    lines.append("%s  role UNKNOWN - %s" % (tag, desc))
                    continue
                task, marker, standalone = roles[role]
                if task not in tasks:
                    tasks.append(task)
                placement = ("its own line in subject_definitions"
                             if standalone else
                             "cited INSIDE the relevant <Subject N> definition, no line of its own")
                lines.append("%s  role: %s\n    %s\n    marker: %s\n    placement: %s"
                             % (tag, role, desc or "(no description given)", marker, placement))
                retention.append("%s: %s - " % (tag, marker))

        order = [name for _, name in TASKS]
        tasks.sort(key=lambda t: order.index(t))
        prefix = "[%s]" % " + ".join(tasks) if tasks else ""

        if "audio reuse" in tasks and "audio reference" in tasks:
            warnings.append("both audio reuse and audio reference are implied; valid only if "
                            "they apply to DIFFERENT <Audio N>")
        if "video editing" in tasks and "video continuation" in tasks:
            warnings.append("both video editing and video continuation are implied; check "
                            "that is really what you mean")

        inventory = ("## Assets\n\nThese labels already exist. Use them exactly; do not "
                     "renumber or invent others. Numbering is 1-based and independent per "
                     "type, so one source file can be both <Video 1> and <Audio 2>.\n\n"
                     + ("\n".join(lines) if lines else "(none)"))
        skeleton = "\n".join(retention) if retention else ""
        report = ("assets: %d image(s), %d video(s), %d audio\nprefix: %s\n%s"
                  % (len(parsed["images"]), len(parsed["videos"]), len(parsed["audios"]),
                     prefix or "(none)",
                     "\n".join("  ! " + w for w in warnings) if warnings else "  no warnings"))
        print("[MMH3AssetPlan] " + report)
        return io.NodeOutput(inventory, prefix, skeleton, report)


class MMH3TaskSystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3TaskSystemPrompt",
            display_name="MMH3 Task System Prompt",
            category="MMH3Tools/prompt",
            description=(
                "Emit a system prompt for your own LLM node, containing only the rules that "
                "apply to the selected mode and task-type combination."
            ),
            inputs=[
                io.Combo.Input("mode", options=MODES, default="Ref2VA",
                               tooltip="Ref2VA uses the six-section format and task types. "
                                       "T2VA/I2VA/L2VA/FL2VA use the three-field format and "
                                       "have NO task types at all."),
                io.Boolean.Input("keyframe_completion", default=False),
                io.Boolean.Input("reference_generation", default=False),
                io.Boolean.Input("video_editing", default=False),
                io.Boolean.Input("video_continuation", default=False),
                io.Boolean.Input("audio_reuse", default=False),
                io.Boolean.Input("audio_reference", default=False),
                io.Float.Input("seconds", default=5.167, min=0.2, max=150.0, step=0.001,
                               tooltip="Target duration. Wire from MMH3 Frame Calculator's "
                                       "actual_seconds so the model plans cuts against the "
                                       "duration it will really get."),
                io.Boolean.Input("include_chained_defaults", default=False,
                                 tooltip="Add the long-form chunking defaults: no score per "
                                         "chunk, ambience restated explicitly, complete a "
                                         "mid-cut utterance rather than starting a new line."),
                io.String.Input("extra_rules", multiline=True, default="", optional=True,
                                tooltip="Appended verbatim at the end."),
                io.String.Input("task_prefix_override", default="", optional=True,
                                tooltip="Wire MMH3 Asset Plan's task_prefix here. When set it "
                                        "REPLACES the booleans above, so roles decide the task "
                                        "types and the two cannot disagree."),
                io.String.Input("asset_inventory", multiline=True, default="", optional=True,
                                tooltip="Wire MMH3 Asset Plan's inventory here so the model "
                                        "knows which labels exist and what each one is before "
                                        "it writes subject_definitions."),
                io.String.Input(
                    "dialogue", multiline=True, default="", optional=True,
                    tooltip="Spoken lines to use VERBATIM, one per line. When set, the system "
                            "prompt fixes them: reproduce exactly, invent none, and treat the "
                            "word budget as a ceiling rather than a target. Leave empty to let "
                            "the model write its own dialogue."),
                io.Combo.Input(
                    "masked_audio", options=MASKED_AUDIO_KINDS, default="none", optional=True,
                    tooltip="Set when you mask a supplied audio latent so the track survives "
                            "verbatim into the output. Tells the model the audio fields "
                            "DESCRIBE existing sound rather than requesting it, and what is in "
                            "the track, so the picture is generated to match. Base modes only "
                            "- on Ref2VA use the audio reuse task type instead."),
            ],
            outputs=[
                io.String.Output(display_name="system"),
                io.String.Output(display_name="task_prefix"),
                io.String.Output(display_name="report"),
                # Wire into MMH3PromptLint's mode_override. The mode decides which
                # section set the linter expects, and setting it in two places is a
                # silent failure: linting a three-field prompt as Ref2VA reports four
                # missing sections that are not missing, and linting a six-section
                # prompt as a base mode reports three. Both look like the LLM erred.
                io.String.Output(display_name="mode"),
            ],
        )

    @classmethod
    def execute(cls, mode, keyframe_completion, reference_generation, video_editing,
                video_continuation, audio_reuse, audio_reference, seconds,
                include_chained_defaults, extra_rules="", task_prefix_override="",
                asset_inventory="", dialogue="", masked_audio="none") -> io.NodeOutput:
        override = (task_prefix_override or "").strip().strip("[]").strip()
        if override:
            known = {name for _, name in TASKS}
            wanted = [p.strip() for p in override.split("+")]
            chosen = [name for _, name in TASKS if name in wanted]
            unknown = [w for w in wanted if w not in known]
        else:
            flags = {
                "keyframe completion": keyframe_completion,
                "reference generation": reference_generation,
                "video editing": video_editing,
                "video continuation": video_continuation,
                "audio reuse": audio_reuse,
                "audio reference": audio_reference,
            }
            chosen = [name for _, name in TASKS if flags[name]]
            unknown = []
        prefix = "[%s]" % " + ".join(chosen) if chosen else ""
        is_a = mode in FORMAT_A

        frames, actual = _achievable(seconds)
        notes = []
        if unknown:
            notes.append("task_prefix_override contains unrecognised types, ignored: %s"
                         % ", ".join(unknown))
        if override:
            notes.append("task types came from task_prefix_override; the booleans are ignored.")
        if is_a and chosen:
            notes.append("mode %s uses the three-field format, which has NO task types - "
                         "the selected types are ignored and no prefix is emitted." % mode)
        if not is_a and not chosen:
            notes.append("Ref2VA with no task type selected: the summary prefix will be "
                         "missing, and the model falls back to reference generation.")
        if audio_reuse and audio_reference:
            notes.append("audio reuse + audio reference are contradictory for the SAME asset "
                         "(one copies the signal, one does not). Valid only across different "
                         "<Audio N>.")
        if video_editing and video_continuation:
            notes.append("video editing + video continuation both claim the source video; "
                         "check that is really what you mean.")
        if video_editing and not audio_reuse:
            notes.append("editing a source whose audio stays audible should also take "
                         "audio reuse.")
        if video_continuation and not (audio_reuse or audio_reference):
            notes.append("continuing a source whose audio character carries should also take "
                         "audio reference.")
        if abs(actual - seconds) > 0.001:
            notes.append("%.3fs is not achievable; nearest is %.3fs (%d frames)."
                         % (seconds, actual, frames))
        if masked_audio != "none" and not is_a:
            notes.append("masked_audio is for the BASE modes. On Ref2VA the trained path is "
                         "the audio reuse task type with a fully_copy marker - using both "
                         "describes the same track two different ways.")
        if masked_audio in ("speech", "sung lyrics") and not dialogue.strip():
            notes.append("masked_audio is '%s' with no dialogue supplied, so the words rely "
                         "entirely on the writing model's own transcription. Asking one call "
                         "to transcribe AND compose is where this usually fails%s - a "
                         "dedicated ASR pass into the dialogue input is far more reliable."
                         % (masked_audio,
                            ", and sung words over backing are much harder than speech"
                            if masked_audio == "sung lyrics" else ""))
        if masked_audio == "background music" and dialogue.strip():
            notes.append("masked_audio is 'background music' but dialogue was supplied. The "
                         "track has no voice to carry it, so any <d> line will be mouthed "
                         "over silence.")

        parts = [_BASE]
        parts.append(_FMT_A.format(instruction=_INSTR[mode]) if is_a else _FMT_B)
        parts.append(_SYNTAX)

        if asset_inventory.strip():
            parts.append(asset_inventory.strip())

        if not is_a and chosen:
            parts.append("## Task type\n\nBegin the summary with exactly:  %s" % prefix)
            parts.append("\n\n".join(_TASK_RULES[c] for c in chosen))
            parts.append(_SCOPING)

        d_lines = [ln.strip() for ln in dialogue.splitlines() if ln.strip()]
        d_words = sum(len(ln.split()) for ln in d_lines)
        budget = max(0, round((actual - 1) * 2.5))

        # A bare word target invites a small model to pad up to it. Harmless when the
        # model is writing its own lines; destructive when the lines are the user's,
        # because the invented ones arrive correctly formatted and are easy to miss.
        if d_lines:
            budget_rule = (
                "- At conversational pace about %d words fit in this duration. That is a\n"
                "  CEILING, not a target. The supplied dialogue is %d word%s across %d line%s;\n"
                "  do NOT add lines to reach the ceiling."
                % (budget, d_words, "" if d_words == 1 else "s",
                   len(d_lines), "" if len(d_lines) == 1 else "s"))
            if masked_audio in ("speech", "sung lyrics"):
                # the lines are a TRANSCRIPT of a fixed track, so its real timing governs;
                # a word estimate that disagrees would only invite trimming the transcript
                budget_rule = (
                    "- The dialogue is a transcript of the supplied audio, so the track's own\n"
                    "  timing governs. Place each line where it is actually heard. Do not add,\n"
                    "  cut or re-time lines to fit a word estimate.")
            elif d_words > budget:
                notes.append("supplied dialogue is %d words but only ~%d fit in %.3fs - keep "
                             "every line and cut surrounding action, or lengthen the chunk"
                             % (d_words, budget, actual))
        else:
            budget_rule = ("- At conversational pace budget about 2.5 words per second and leave\n"
                           "  ~1s at the end, so roughly %d words of dialogue TOTAL." % budget)

        parts.append(
            "## Constraints\n\n"
            "- Target duration is %.3f seconds (%d frames at 24fps). Cut times must fall\n"
            "  inside it. Frame counts are 17j+5, so achievable durations are discrete.\n"
            "- Ref2VA accepts at most 9 images, 3 videos, 3 audio clips, 12 files total.\n"
            "  Each reference video or audio clip is 2-15s; each media type totals 15s max.\n"
            "%s" % (actual, frames, budget_rule)
        )

        if include_chained_defaults:
            parts.append(
                "## Chained / long-form defaults\n\n"
                "- Set non_diegetic_music: N/A. Score is added over the finished timeline;\n"
                "  independently generated chunks share no key, tempo or bar position.\n"
                "- Restate the ambience explicitly and continuously - it is the signal that\n"
                "  hides a join.\n"
                "- If the source ends MID-UTTERANCE, open by completing that sentence rather\n"
                "  than starting a new one, and mark the carry-over with <scenetrans>."
            )

        # before the dialogue block: it establishes that the audio is fixed, which is the
        # premise the supplied lines are a transcript OF rather than a request for
        if masked_audio != "none":
            parts.append(_MASKED_AUDIO_COMMON + "\n\n" + _MASKED_AUDIO[masked_audio])

        if d_lines:
            parts.append(_SUPPLIED_DIALOGUE % "\n".join(d_lines))

        # "invent concrete detail" has to be narrowed when the dialogue is fixed, or it
        # licenses exactly the padding the block above forbids.
        parts.append("## Output\n\nEmit only the finished prompt. If the idea is too thin for "
                     "the duration, invent concrete %s consistent with the intent rather "
                     "than padding with adjectives.%s"
                     % ("action, camera and ambience detail" if d_lines else "detail",
                        " Never invent dialogue." if d_lines else ""))

        if extra_rules.strip():
            parts.append(extra_rules.strip())

        system = "\n\n".join(parts)
        report = "mode: %s | format %s | prefix: %s | %.3fs (%d frames)\n%s" % (
            mode, "A" if is_a else "B", prefix or "(none)", actual, frames,
            "\n".join("  ! " + n for n in notes) if notes else "  no warnings")
        print("[MMH3TaskSystemPrompt] " + report)
        return io.NodeOutput(system, prefix, report, mode)


class MMH3ReplaceSection(io.ComfyNode):
    """Splice a rewritten section back into a prompt, so a refiner cannot drop the rest.

    Asking one instruct model to reproduce five sections verbatim AND rewrite the sixth
    is the fragile half of a refinement pass. It reliably does the rewrite and then
    returns the body alone, or wraps every label in markdown -- both of which read
    downstream as "the model forgot the format".

    Give the refiner ONE job instead: return the new body, no labels. This node holds
    the structure. Dropping a section becomes impossible rather than unlikely.

    It also normalises what comes back: code fences, a repeated label, and markdown
    decoration are all stripped, because the text encoder receives those characters
    literally and H3 was trained on plain labels.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3ReplaceSection",
            display_name="MMH3 Replace Section",
            category="MMH3Tools/prompt",
            description=(
                "Replace one section of a prompt with new text and re-emit all sections "
                "in canonical order with plain labels. Lets a refiner return only the "
                "rewritten body instead of the whole prompt."
            ),
            inputs=[
                io.String.Input("prompt", multiline=True, force_input=True,
                                tooltip="The ORIGINAL, complete prompt. Its other sections "
                                        "are carried through untouched."),
                io.String.Input("replacement", multiline=True, force_input=True,
                                tooltip="The refiner's output: just the new section body. "
                                        "A repeated label, code fences and markdown "
                                        "decoration are stripped."),
                io.Combo.Input("section", options=_ALL_SECTIONS,
                               default="detailed_description",
                               tooltip="Which section the replacement is."),
                io.Combo.Input("mode", options=MODES, default="Ref2VA",
                               tooltip="Selects the canonical section set and order. Wire "
                                       "MMH3 Task System Prompt's mode."),
            ],
            outputs=[
                io.String.Output(display_name="prompt"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, prompt, replacement, section, mode) -> io.NodeOutput:
        from .nodes_lint import _SECTIONS_A, _SECTIONS_B, _section
        sections = _SECTIONS_A if mode in FORMAT_A else _SECTIONS_B

        if section not in sections:
            raise ValueError(
                "section %r is not part of the %s format, which has: %s. Wire the mode "
                "from MMH3 Task System Prompt so the two cannot disagree."
                % (section, mode, ", ".join(sections)))

        notes = []

        # A REPLACEMENT carrying the format's own headers is a whole prompt in the
        # wrong socket, not a section body. Splicing it in nests a prompt inside its
        # own section and the result renders without ever erroring -- observed
        # 2026-08-17, three chunks deep, from a writer that had been shown a complete
        # prompt as its continuity example and copied the format. Two headers, not
        # one: a body may legitimately mention a field name in prose.
        rep_headers = [name for name in sections if _section(replacement, name, sections)
                       is not None]
        if len(rep_headers) >= 2:
            raise ValueError(
                "the replacement for %r is itself a prompt -- it carries %d of the %d "
                "%s section headers (%s). A replacement is ONE section's body, so "
                "splicing this in would nest a whole prompt inside %r. The writer was "
                "almost certainly shown a complete prompt as an example and copied it: "
                "check what reaches its system prompt, and wire continuity through "
                "MMH3 Scene Plan Prompt's `prev_detailed` (which extracts the one "
                "section) rather than concatenating a prior prompt by hand."
                % (section, len(rep_headers), len(sections), mode,
                   ", ".join(rep_headers), section))

        bodies, missing = {}, []
        for name in sections:
            b = _section(prompt, name, sections)
            if b is None:
                missing.append(name)
            bodies[name] = b or ""
        if missing:
            # A section the writer left out is RECOVERABLE, and refusing here was
            # costing whole runs. The format fixes the order, so an absent header has
            # exactly one correct position -- inserting it empty loses nothing and is
            # not a guess. What is NOT recoverable is a prompt with no sections at
            # all: that is the refiner's output, or a report, wired in by mistake.
            # HALF, not all: a report or a stray paragraph can contain one header
            # by accident, and "recovering" that into a five-section skeleton would
            # manufacture a prompt out of nothing. A real skeleton always carries the
            # sections the writer actually filled in.
            if len(sections) - len(missing) < len(sections) / 2.0:
                raise ValueError(
                    "the prompt is missing %d of the %d %s sections (%s), which is "
                    "too many to be a prompt with headers left out -- it is a report, "
                    "a bare paragraph, or the refiner's output wired in by mistake. "
                    "This input wants the complete prompt from the first LLM call."
                    % (len(missing), len(sections), mode, ", ".join(missing)))
            notes.append("inserted %d missing section%s: %s. The writer omitted "
                         "%s -- models resist emitting a header with nothing under "
                         "it, and the format decides where it belongs."
                         % (len(missing), "" if len(missing) == 1 else "s",
                            ", ".join(missing),
                            "it" if len(missing) == 1 else "them"))

        new = (replacement or "").strip()
        # code fences
        if new.startswith("```"):
            new = re.sub(r"^```[a-zA-Z]*\n?|```$", "", new).strip()
            notes.append("stripped code fences")
        # the label repeated back at us, decorated or not
        m = re.match(r"^[^\w\n]{0,8}%s[^\w\n]{0,8}:?[^\S\n]*\n?" % re.escape(section), new)
        if m:
            new = new[m.end():].lstrip()
            notes.append("stripped a repeated %r label" % section)
        if not new:
            raise ValueError("replacement is empty after cleaning; nothing to splice in.")

        bodies[section] = new
        out = "\n\n".join("%s:\n%s" % (name, bodies[name]) for name in sections)

        report = "%s: %d -> %d chars | %d sections rebuilt in canonical order" % (
            section, len(_section(prompt, section, sections) or ""), len(new), len(sections))
        for n in notes:
            report += "\n  " + n
        logging.info("[MMH3ReplaceSection] " + report.splitlines()[0])
        return io.NodeOutput(out, report)


_STABLE_SECTIONS = ("subject_definitions", "retention_analysis")

_CTX_HEADERS = {
    "all": (
        "Prompts already written for earlier windows of this clip. Keep "
        "subject_definitions and retention_analysis byte-identical to these; only "
        "detailed_description should differ.\n\n"),
    "last": (
        "The prompt written for the PREVIOUS window of this clip. Keep "
        "subject_definitions and retention_analysis byte-identical to it. Its "
        "detailed_description describes a DIFFERENT stretch of the song -- do not "
        "reuse its shots, its cut times or its lyrics.\n\n"),
    "last_definitions": (
        "The sections that must not change between windows, taken from the previous "
        "window. Reproduce them byte-identical. Everything else -- summary, "
        "detailed_description, the audio fields -- describes THIS window and must be "
        "written fresh from the audio you were given.\n\n"),
}


def _stable_sections(piece):
    """subject_definitions + retention_analysis, or None if neither parses.

    Deferred import: nodes_lint imports FROM this module, so a top-level import
    is a cycle. Same reason _achievable's consumers do it this way.
    """
    from .nodes_lint import _SECTIONS_A, _SECTIONS_B, _section
    following = list(dict.fromkeys(_SECTIONS_B + _SECTIONS_A))
    out = []
    for name in _STABLE_SECTIONS:
        body = _section(piece, name, following)
        if body:
            out.append("%s:\n%s" % (name, body))
    return "\n\n".join(out) if out else None


def _prior_context(prior_pieces, mode):
    """What the writing model is shown of the work already done.

    WHY THIS IS A CHOICE. 'all' re-sends every earlier prompt in full, which grows
    linearly: on a 20s-window clip that is ~7,900 tokens by window 7, against a few
    hundred tokens for the window's own audio. The model is then reading twenty
    times more "here is what you already wrote, stay consistent" than "here is the
    new material", and it does the obvious thing.

    Worse, 'all' carries the previous detailed_descriptions -- and the header asks
    for exactly that section to differ. The instruction and the payload point in
    opposite directions, so 'last_definitions' withholds the section that must be
    fresh and sends only the ones that must not change.
    """
    if not prior_pieces:
        return ""
    if mode == "all":
        body = "\n\n".join("--- window %d ---\n%s" % (i + 1, p)
                           for i, p in enumerate(prior_pieces))
        return _CTX_HEADERS["all"] + body

    last = prior_pieces[-1]
    if mode == "last_definitions":
        stable = _stable_sections(last)
        if stable is not None:
            return _CTX_HEADERS["last_definitions"] + stable
        # A prompt we cannot parse is not a reason to send nothing -- consistency
        # is the whole job of this output. Fall back and say so.
        logging.warning("[MMH3PromptAccumulate] prior_context_mode=last_definitions "
                        "but the previous prompt has no parseable %s; sending it "
                        "whole instead", " or ".join(_STABLE_SECTIONS))
    return _CTX_HEADERS["last"] + last


class MMH3PromptAccumulate(io.ComfyNode):
    """Build one pipe-separated string across a loop, one prompt per iteration.

    A for loop cannot hand a growing list between iterations -- only values it
    carries back through its END node -- so per-window prompts have to be
    accumulated as text and split apart later. MMH3ReferenceMultiPrompt takes
    exactly that: one string, pipe separated, in chunk order.

    FIRST ITERATION. The loop's carried slot is unwired on the first pass, so
    `accumulated` arrives as None. That is treated as "nothing yet" and the
    separator is NOT emitted -- otherwise every run would open with an empty
    leading piece. Blank strings count as nothing too, since a loop wired with
    an empty-string initial value is the same situation.

    `prior_context` exists for feeding earlier prompts back to the writing model.
    Do NOT use an LLM node's `history` input for this if the node attaches audio
    or images: a chat history keeps the base64 of every prior turn, so a handful
    of windows becomes megabytes re-sent every iteration, and the model ends up
    looking at all the previous windows' audio while writing this one.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3PromptAccumulate",
            display_name="MiniMax H3 Prompt Accumulate",
            category="MMH3Tools/prompt",
            description=(
                "Append one prompt to a running pipe-separated string, for a "
                "for-loop that writes one prompt per window. Wire the loop's "
                "carried value into `accumulated` and this node's `text` output "
                "back into the loop's END node. The finished string goes to "
                "MiniMax H3 Reference (Multi-Prompt)."
            ),
            inputs=[
                io.String.Input(
                    "prompt", multiline=True, force_input=True, optional=True,
                    tooltip="This iteration's prompt. OPTIONAL, because a SECOND "
                            "copy of this node at the TOP of the loop body -- fed "
                            "only `accumulated` -- is how you read `prior_context` "
                            "without a cycle. The accumulating copy sits after the "
                            "writing model, so its outputs cannot reach anything "
                            "upstream of it."),
                io.String.Input(
                    "accumulated", multiline=True, force_input=True, optional=True,
                    tooltip="Everything so far -- the loop's carried value. Leave "
                            "the loop's INITIAL value unwired: arriving as None or "
                            "empty is how this node knows it is the first pass."),
                io.String.Input(
                    "separator", multiline=False, default=" | ",
                    tooltip="Must contain the pipe that MMH3ReferenceMultiPrompt "
                            "splits on. Spaces around it are cosmetic; the split "
                            "strips them."),
                io.Boolean.Input(
                    "strip_fences", default=True,
                    tooltip="Remove ``` code fences a writing model wrapped its "
                            "answer in. They are never part of an H3 prompt and "
                            "would ride into the encode."),
                io.Combo.Input(
                    "prior_context_mode",
                    options=["all", "last", "last_definitions"], default="all",
                    tooltip="How much of the earlier work `prior_context` hands back "
                            "to the writing model.\n\n"
                            "'all' sends every earlier prompt in full. That is ~7,900 "
                            "tokens by window 7 of a 20s-window clip, against a few "
                            "hundred for the new audio -- roughly 20:1 in favour of "
                            "copying, which is what makes late windows re-describe "
                            "earlier ones.\n\n"
                            "'last' sends only the previous window's prompt.\n\n"
                            "'last_definitions' sends only the previous window's "
                            "subject_definitions and retention_analysis -- the parts "
                            "that must stay byte-identical. It withholds "
                            "detailed_description on purpose: that is the section "
                            "that must DIFFER, so supplying it as an example is what "
                            "makes it not differ."),
            ],
            outputs=[
                io.String.Output(display_name="text"),
                io.Int.Output(display_name="count"),
                io.String.Output(display_name="prior_context"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, prompt=None, accumulated=None, separator=" | ",
                strip_fences=True, prior_context_mode="all") -> io.NodeOutput:
        sep = separator if separator else " | "
        if "|" not in sep:
            raise ValueError(
                "MMH3PromptAccumulate: separator %r has no pipe in it. "
                "MMH3ReferenceMultiPrompt splits on '|', so anything else "
                "produces one enormous prompt instead of N." % separator)

        new = (prompt or "").strip()
        if strip_fences and new.startswith("```"):
            new = re.sub(r"^```[a-zA-Z]*\n?|```$", "", new).strip()

        prior = (accumulated or "").strip()
        # None AND empty both mean "first pass" -- a loop whose initial value is
        # unwired gives None, one wired to an empty primitive gives "".
        if not prior:
            text = new
        elif not new:
            text = prior
        else:
            text = prior + sep + new

        pieces = [p.strip() for p in text.split("|") if p.strip()]
        n = len(pieces)

        # Derived from what CAME IN, never from the result. Two reasons: with a
        # prompt supplied it is the same set either way, and with none supplied --
        # the copy at the top of the loop body, which is the one that can actually
        # reach the writing model -- taking it from the result would drop the most
        # recent window, the very one the model most needs to stay consistent with.
        prior_pieces = [p.strip() for p in prior.split("|") if p.strip()]
        ctx = _prior_context(prior_pieces, prior_context_mode)

        note = ""
        if not new:
            note = "  ! this iteration's prompt was empty; nothing appended"
        elif prior and new in pieces[:-1]:
            note = ("  ! this prompt is identical to an earlier one -- the loop may "
                    "be feeding the same window, or the carried value is not "
                    "advancing")
        report = ("%d prompt%s, %d chars\n  latest: %s"
                  % (n, "" if n == 1 else "s", len(text),
                     (pieces[-1][:70] + "...") if pieces else "(none)"))
        # What the writing model is actually handed back. Invisible otherwise, and
        # its growth under 'all' is the thing that makes late windows repeat.
        if prior_pieces:
            report += ("\n  prior_context: %s, %d of %d earlier prompt%s, %d chars "
                       "(~%d tokens)"
                       % (prior_context_mode,
                          len(prior_pieces) if prior_context_mode == "all" else 1,
                          len(prior_pieces), "" if len(prior_pieces) == 1 else "s",
                          len(ctx), len(ctx) // 4))
            if prior_context_mode == "all" and len(prior_pieces) >= 3:
                report += ("\n  ! %d full prompts against one window of audio -- if "
                           "late windows repeat earlier ones, try "
                           "'last_definitions'" % len(prior_pieces))
        if note:
            report += "\n" + note
        logging.info("[MMH3PromptAccumulate] %d prompt(s), %d chars", n, len(text))
        return io.NodeOutput(text, n, ctx, report)
