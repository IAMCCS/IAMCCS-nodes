"""Check a finished H3 prompt against the format rules, before anything is sampled.

MMH3TaskSystemPrompt validates the SETTINGS you gave the node. This validates the
TEXT an LLM wrote from them, which is where the interesting failures are: a local
model follows most of a long rule list and quietly drops the rest.

The economics are the whole argument. A chunk is minutes of sampling, and most
format errors do not crash - they render something subtly wrong. A cut timed past
the end of the clip simply never happens; a quoted line of dialogue asks for a sign
instead of speech and you get a caption; a voiceover missing its lips-closed clause
gets mouthed. Every one of those costs a full generation to discover by watching,
and a second to discover here.
"""

import logging
import re

from comfy_api.latest import io

from .nodes_prompt import FORMAT_A, MODES, _achievable

# non_diegetic_music wants instrumentation, tempo, rhythm, dynamics -- never the
# emotional function, which the model cannot render and which crowds out what it can
_MOOD = re.compile(
    r"\b(sad|epic|uplifting|triumphant|menacing|cheerful|tense|heartwarming|nostalgic|"
    r"melancholy|joyful|ominous|hopeful|romantic|eerie|whimsical|dramatic|emotional|"
    r"haunting|playful|somber|sombre)\b", re.I)

# retention_analysis markers, visible then audio. Mirrors _FMT_B in nodes_prompt.
_MARKERS = ["fully_preserved", "partially_preserved", "attribute_transfer",
            "weak_reference", "fully_copy", "partially_copy", "reference"]

_SECTIONS_B = ["subject_definitions", "summary", "retention_analysis",
               "detailed_description", "overall_soundscape", "non_diegetic_music"]
_SECTIONS_A = ["integrated_multimodal_description", "overall_soundscape",
               "non_diegetic_music"]


def _section(prompt, name, following):
    """Body of a `name:` section, up to whichever of `following` comes first.

    The boundary tolerates leading whitespace. Without `\\s*` an indented prompt --
    and LLMs indent these constantly -- never matches the stop, so EVERY section
    silently runs to the end of the document and every downstream check reads the
    wrong text.

    `following` should be EVERY section label, its own included, so a repeat of the
    same field also terminates it. The last field otherwise runs to \\Z and swallows
    every repeated block after it -- which is how a phantom mood word turned up in
    non_diegetic_music, read out of a later shot's description.
    """
    stop = "|".join(r"\n\s*%s\s*:" % re.escape(f) for f in following) or r"\Z"
    m = re.search(r"%s\s*:\s*\n?(.*?)(?=%s|\Z)" % (re.escape(name), stop), prompt, re.S)
    return m.group(1).strip() if m else None


def _section_count(prompt, name):
    return len(re.findall(r"(?m)^\s*%s\s*:" % re.escape(name), prompt))


def lint_prompt(prompt, mode="Ref2VA", seconds=0.0):
    """Return a list of problem strings. Empty means clean."""
    p = prompt or ""
    out = []
    is_a = mode in FORMAT_A
    sections = _SECTIONS_A if is_a else _SECTIONS_B
    # the shot body is the FIRST field in the three-field format but the FOURTH in the
    # six-section one -- taking sections[0] silently lints subject_definitions instead
    body_field = _SECTIONS_A[0] if is_a else "detailed_description"

    if not p.strip():
        return ["prompt is empty"]

    # Count, don't just test for presence. A prompt with the fields repeated PER SHOT
    # is the most destructive malformation there is -- the format has exactly one of
    # each, with every [Shot N] inside the single description -- and it used to lint
    # clean, because re.search finds the first and stops.
    for i, name in enumerate(sections):
        n = _section_count(p, name)
        if n == 0:
            # A label the model DECORATED is the common cause, and "missing" reads as
            # "the model forgot it" when really it wrote `**subject_definitions:**` or
            # `### subject_definitions`. The decoration is a real defect -- the text
            # encoder sees the literal characters and H3 was trained on plain labels --
            # but naming it turns six baffling absences into one obvious fix.
            # with a colon (`**name:**`, `- name:`) or without (`### name`) -- markdown
            # headings drop the colon entirely, which is the form that used to slip past
            m = (re.search(r"(?mi)^(.{0,8}%s.{0,8}:)" % re.escape(name), p)
                 or re.search(r"(?mi)^([^\w\n]{0,8}%s[^\w\n]{0,8})$" % re.escape(name), p))
            if m:
                out.append("section %s is DECORATED, not plain: %r - the label must be "
                           "exactly '%s:' at the start of its own line, because the text "
                           "encoder receives those characters literally"
                           % (name, m.group(1).strip(), name))
            else:
                out.append("missing section: %s" % name)
        elif n > 1:
            out.append("%s appears %d times; there is exactly ONE of each field for the "
                       "whole clip, with every [Shot N] inside the single %s"
                       % (name, n, body_field))

    body = _section(p, body_field, sections) or ""

    # --- shot structure -------------------------------------------------------
    # The two formats put the style in DIFFERENT places, and requiring format A's
    # shape everywhere flagged correct Ref2VA prompts:
    #   A: "[Shot 1] <style>, <shot 1>"        style INSIDE shot 1
    #   B: "One or two style sentences BEFORE [Shot 1]."
    if body:
        lead = body.split("[Shot 1]")[0].strip() if "[Shot 1]" in body else body.strip()
        if "[Shot 1]" not in body:
            out.append("%s has no [Shot 1]" % body_field)
        elif is_a:
            if lead:
                out.append("%s does not open with [Shot 1]; in this format the style goes "
                           "INSIDE it: '[Shot 1] <style>, <shot 1>'" % body_field)
        elif lead:
            # a lead-in is correct here, but it is style only -- no timing, no action
            if re.search(r"\d{2}:\d{2}", lead):
                out.append("the style sentences before [Shot 1] carry a timestamp; they "
                           "establish look only, and timed content belongs in a shot")
            if len(lead.split()) > 80:
                out.append("the lead-in before [Shot 1] is %d words; it should be one or "
                           "two style sentences, with shot content inside a numbered shot"
                           % len(lead.split()))
    if re.search(r"\[Shot 1\]\s+At\b", body):
        out.append("[Shot 1] carries a timestamp; only later shots are timed")

    ts = [float(a) * 60 + float(b) for a, b in
          re.findall(r"\[Shot \d+\] At (\d{2}):(\d{2}(?:\.\d+)?)", body)]
    if ts != sorted(ts):
        out.append("shot timestamps are not increasing: %s" % ts)
    if len(ts) != len(set(ts)):
        out.append("duplicate shot timestamps: %s" % ts)
    if seconds > 0 and ts and max(ts) >= seconds:
        _, actual = _achievable(seconds)
        out.append("a cut at %.3fs falls outside the %.3fs clip, so it never happens"
                   % (max(ts), actual))

    nums = [int(n) for n in re.findall(r"\[Shot (\d+)\]", body)]
    if nums and nums != list(range(1, len(nums) + 1)):
        out.append("shot numbers are not 1..N in order: %s" % nums)

    # --- dialogue -------------------------------------------------------------
    if p.count("<d>") != p.count("</d>"):
        out.append("unbalanced <d> tags: %d open, %d close" % (p.count("<d>"), p.count("</d>")))
    for d in re.findall(r"<d>(.*?)</d>", p, re.S):
        if not re.match(r"\s*\[[^\]]+\]", d):
            out.append("<d> block has no [Language] tag: %r" % d.strip()[:60])
        if re.search(r"\(S\d+", d):
            out.append("speaker ID inside <d>; it belongs outside: %r" % d.strip()[:60])
        if re.search(r"\b(says|said|whispers|shouts|sings)\b", d, re.I):
            out.append("delivery verb inside <d>; only the words belong there: %r"
                       % d.strip()[:60])
    for q in re.findall(r'"[^"\n]{0,120}"\s*(?=</d>)|<d>[^<]{0,20}"', p):
        out.append("dialogue in double quotes; quotes mean text shown ON SCREEN, so this "
                   "asks for a sign instead of speech")
        break

    # --- voiceover ------------------------------------------------------------
    # Two things this pattern has to get right, both learned from false reports:
    #
    #   .{0,40}?<d>  -- the dialogue must follow the phrase CLOSELY. With an
    #     unbounded .*? under re.S the match leaps across the whole document to
    #     whatever </d> comes next and then judges the text after THAT, so the
    #     phrase appearing in prose -- or in the format rules, which contain it
    #     verbatim as an instruction -- reported a failure with the lips-closed
    #     statement sitting untouched right beside it.
    #
    #   (?=(...))    -- the trailing window is a LOOKAHEAD so the match does not
    #     consume it. Consuming 120 characters swallowed the start of the next
    #     voiceover, and finditer skipped it; a prompt whose SECOND voiceover was
    #     the broken one linted clean.
    for vo in re.finditer(
            r"says in an off-screen voiceover.{0,40}?<d>(?:(?!</d>).)*</d>(?=(.{0,120}))",
            p, re.S):
        if "lips remain" not in vo.group(1):
            out.append("off-screen voiceover is not followed by the lips-closed statement, "
                       "so the character will be animated speaking it: %r"
                       % p[vo.start():vo.end() + 60].strip()[:110])

    # --- audio fields ---------------------------------------------------------
    sound = _section(p, "overall_soundscape", sections) or ""
    if "<d>" in sound:
        out.append("overall_soundscape contains dialogue; it covers ambience, action sound "
                   "and non-verbal human sound only")

    music = _section(p, "non_diegetic_music", sections) or ""
    for w in sorted(set(m.lower() for m in _MOOD.findall(music))):
        out.append("mood word in non_diegetic_music: %r - describe instrumentation, tempo, "
                   "rhythm and dynamics instead" % w)

    # --- labels ---------------------------------------------------------------
    if not is_a:
        defined = set(re.findall(r"<(Picture|Video|Audio|Subject) (\d+)>",
                                 _section(p, "subject_definitions", sections) or ""))
        used = set(re.findall(r"<(Picture|Video|Audio|Subject) (\d+)>", body))
        for kind, n in sorted(used - defined):
            out.append("<%s %s> is used in the body but never defined in "
                       "subject_definitions" % (kind, n))
        # "summary - Reuse existing labels only; introduce none here." An undefined
        # label in the summary was invisible, because only the body was checked.
        summary_used = set(re.findall(r"<(Picture|Video|Audio|Subject) (\d+)>",
                                      _section(p, "summary", sections) or ""))
        for kind, n in sorted(summary_used - defined):
            out.append("<%s %s> appears in the summary but is never defined in "
                       "subject_definitions" % (kind, n))
        # A retention marker written into subject_definitions instead of its own section.
        # Anchored to a marker POSITION -- after a comma or colon, at end of line -- because
        # 'reference' is also an ordinary word: "the voice-timbre reference for <Subject 1>"
        # is correct prose and matched a bare \b'reference'\b.
        for m in sorted(set(re.findall(
                r"(?m)[,:]\s*(%s)\s*$" % "|".join(_MARKERS),
                _section(p, "subject_definitions", sections) or ""))):
            out.append("retention marker %r in subject_definitions; markers belong only "
                       "in retention_analysis" % m)
        retention = _section(p, "retention_analysis", sections) or ""
        if re.search(r"\(S\d+", retention):
            out.append("speaker ID (Sx) in retention_analysis; it belongs only in "
                       "subject_definitions and the body")

        # The model echoing the marker MENU instead of choosing from it. The section
        # then looks populated and every other check passes, so this linted clean while
        # saying nothing about what survives -- which is the section's entire job.
        if re.search(r"\|\s*(%s)\b" % "|".join(_MARKERS), retention):
            out.append("retention_analysis repeats the marker MENU instead of choosing "
                       "from it; write one line per label, e.g. '<Subject 1>: "
                       "attribute_transfer - traits carry, the rendering is new'")
        else:
            for line in retention.splitlines():
                line = line.strip()
                if not line or not re.match(r"<(Picture|Video|Audio|Subject) \d+>", line):
                    continue
                if not any(re.search(r"\b%s\b" % m, line) for m in _MARKERS):
                    out.append("retention line has no marker: %r - one of %s"
                               % (line[:60], ", ".join(_MARKERS)))
            # a Subject always appears in the target, so it always needs a marker
            retained = set(re.findall(r"<(Subject) (\d+)>", retention))
            for kind, n in sorted({(k, v) for k, v in defined if k == "Subject"} - retained):
                out.append("<%s %s> is defined but has no retention_analysis line, so "
                           "nothing states how much of it survives" % (kind, n))

        summary = _section(p, "summary", sections) or ""
        if summary and not summary.lstrip().startswith("["):
            out.append("summary does not begin with a [task type] prefix")

    return out


class MMH3PromptLint(io.ComfyNode):
    """Validate a finished prompt before it costs a generation."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3PromptLint",
            display_name="MMH3 Prompt Lint",
            category="MMH3Tools/prompt",
            description=(
                "Check an LLM-written H3 prompt against the format rules. Passes the "
                "prompt through unchanged so it can sit inline between the LLM and the "
                "conditioning node."
            ),
            inputs=[
                io.String.Input("prompt", multiline=True, force_input=True),
                io.Combo.Input("mode", options=MODES, default="Ref2VA",
                               tooltip="Selects which section set is expected: the "
                                       "three-field format for the base modes, the "
                                       "six-section format for Ref2VA."),
                io.Float.Input("seconds", default=0.0, min=0.0, max=150.0, step=0.001,
                               tooltip="Clip duration, so a cut timed past the end is "
                                       "caught. Wire MMH3 Frame Calculator's "
                                       "actual_seconds. 0 skips that check."),
                io.Combo.Input(
                    "on_problem", options=["warn", "error"], default="warn",
                    tooltip="'warn' logs the problems and passes the prompt through "
                            "unchanged. 'error' raises, stopping the queue before "
                            "sampling starts.",
                ),
                io.String.Input(
                    "mode_override", default="", optional=True, force_input=True,
                    tooltip="Wire MMH3 Task System Prompt's 'mode' output here and the two "
                            "can never disagree. Setting the mode in two places is a silent "
                            "failure: the wrong one reports every section of the OTHER "
                            "format as missing, which reads like the LLM ignored the rules. "
                            "Takes precedence over the mode widget when connected.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="prompt"),
                io.String.Output(display_name="report"),
                io.Int.Output(display_name="problems"),
            ],
        )

    @classmethod
    def execute(cls, prompt, mode, seconds, on_problem, mode_override="") -> io.NodeOutput:
        wired = (mode_override or "").strip()
        if wired:
            if wired not in MODES:
                raise ValueError(
                    "[MMH3PromptLint] mode_override is %r, which is not one of %s. Wire "
                    "MMH3 Task System Prompt's 'mode' output, not its 'report'."
                    % (wired, ", ".join(MODES)))
            if wired != mode:
                logging.info("[MMH3PromptLint] mode %s (wired), overriding the widget's %s",
                             wired, mode)
            mode = wired

        problems = lint_prompt(prompt, mode, seconds)
        # State the mode on every report. A finding list that does not say which format
        # it was checking against is unreadable when the answer is that the mode is wrong.
        head = "mode %s (%s)" % (mode, "wired" if wired else "widget")
        if problems:
            report = "%s -- %d problem%s:\n%s" % (
                head, len(problems), "" if len(problems) == 1 else "s",
                "\n".join("  ! " + x for x in problems))
            logging.warning("[MMH3PromptLint] " + report)
            if on_problem == "error":
                raise ValueError("[MMH3PromptLint] " + report)
        else:
            report = "%s -- clean" % head
            logging.info("[MMH3PromptLint] " + report)
        return io.NodeOutput(prompt, report, len(problems))
