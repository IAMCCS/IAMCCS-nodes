"""Load one skill file. Chain the nodes to stack them.

Deliberately one file per node. Selecting several in one node means deciding up
front which kinds exist and how many of each you may have; a chain decides nothing
-- add another node, get another skill, in whatever order you wire them. The type of
a skill lives in its filename (`look-…`, `experiment-…`, `typography-…`), which is
enough to find it in a dropdown and enough for this node to notice an experiment.

Output goes to `extra_rules` on `MMH3ScenePlanPrompt` or `MMH3MusicScenePlanPrompt`,
which append it verbatim.

WHY BLOCKS AND NOT VENDOR SKILLS. MiniMax publish nine H3 skills; all nine are agent
procedures for their own hub, and two say so outright -- "Requires the MiniMax Hub
agent (canvas workspace and MiniMax H3 generation); not portable to generic agent
harnesses." Around their style guidance sits numbered steps, confirmation gates,
prescribed shot counts (15s -> 4 shots, 30s -> 5-6, 60s -> 7-9), fixed time segments
written for 15-second clips, and voiceover the model is expected to generate and
then measure against the video. Pasted whole into a prompt, that fights a
grid-locked window and a pinned master audio. The files in `styles/` are the visual
core lifted out with the procedure left behind.
"""

import logging
import os

from comfy_api.latest import io


def styles_dir():
    """Where skill files live. Beside the pack, so it survives a git pull."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "styles")


def list_skills():
    """Every skill file, by name. `_` prefixed files are notes, not skills."""
    try:
        return ["none"] + sorted(
            os.path.splitext(f)[0] for f in os.listdir(styles_dir())
            if f.lower().endswith((".md", ".txt")) and not f.startswith("_"))
    except Exception:
        return ["none"]


def strip_frontmatter(text):
    """Drop a leading `---` block if there is one, so it cannot reach the prompt."""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4:].strip()
    return text.strip()


def read_skill(name):
    """Body of a skill file, or '' if it is missing."""
    if not name or name == "none":
        return ""
    root = styles_dir()
    for ext in (".md", ".txt"):
        path = os.path.join(root, name + ext)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                return strip_frontmatter(fh.read())
    return ""


def is_experiment(name):
    """Filename carries the type, so an experiment announces itself."""
    return bool(name) and name.lower().startswith("experiment")


class MMH3LoadSkill(io.ComfyNode):
    """One skill file in, text out. Chain for more."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3LoadSkill",
            display_name="MMH3 Load Skill",
            category="MMH3Tools/prompt",
            description=(
                "Load one skill file from the pack's `styles/` folder and emit it for "
                "`extra_rules` on either scene-plan node. Chain these to stack "
                "skills: wire one node's output into the next node's `previous`. The "
                "type is in the filename, and an `experiment-` file is flagged in the "
                "report as untested."
            ),
            inputs=[
                io.Combo.Input(
                    "skill", options=list_skills(), default="none",
                    tooltip="A file from `styles/`. Its name carries its type -- "
                            "look-, typography-, experiment- -- and anything you drop "
                            "in there appears here."),
                io.String.Input(
                    "previous", multiline=True, default="", optional=True,
                    force_input=True,
                    tooltip="Another MMH3 Load Skill's output. Chain nodes to stack "
                            "skills; this one appends after whatever arrives here, so "
                            "wiring order is stacking order."),
                io.Boolean.Input(
                    "enabled", default=True, optional=True,
                    tooltip="Off passes `previous` through untouched, so a node can "
                            "stay in the chain while its skill is switched out."),
            ],
            outputs=[
                io.String.Output(display_name="extra_rules"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        # editing a skill file is the whole point of keeping them in files, so the
        # node has to re-run when one changes rather than serving a cached prompt
        try:
            root = styles_dir()
            return str(sorted((f, os.path.getmtime(os.path.join(root, f)))
                              for f in os.listdir(root)))
        except Exception:
            return ""

    @classmethod
    def execute(cls, skill, previous="", enabled=True) -> io.NodeOutput:
        chain = (previous or "").strip()
        notes = []

        if not enabled:
            return io.NodeOutput(chain, "disabled -- passing %d chars through"
                                 % len(chain))

        body = read_skill(skill)
        if skill and skill != "none" and not body:
            notes.append("%r is selected but its file is missing or empty" % skill)
        if is_experiment(skill) and body:
            notes.append("EXPERIMENTAL: %r is untested. It says what we want to find "
                         "out H3 can do, not what has been observed working -- judge "
                         "the result on its own, not as a known recipe." % skill)

        text = "\n\n".join(x for x in (chain, body) if x)
        report = ("%s | %d chars in, %d out\n%s"
                  % (skill, len(chain), len(text),
                     "\n".join("  ! " + n for n in notes) if notes
                     else "  no warnings"))
        logging.info("[MMH3LoadSkill] %s", report.splitlines()[0])
        return io.NodeOutput(text, report)
