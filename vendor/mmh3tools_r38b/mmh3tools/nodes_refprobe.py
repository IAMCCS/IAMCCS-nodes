"""Which reference is each part of the clip actually attending to?

H3 has **no cross-attention** (`grep -c cross_attn comfy/ldm/minimax/model.py` -> 0):
references are packed into the same sequence and everything is self-attention. So
"which reference is being used here" is a measurable quantity -- attention mass from
the target rows onto each reference's key rows -- rather than an inference from the
output.

The spans are not guessed. `comfy/ldm/minimax/model.py` builds the sequence as a list
of `(kind, length)` segments (`segments.append(("ref_audio", rt * 2))`, one per
reference block, in reference order) and `PackedLayout` exposes them as
`(start, stop, kind)`. This module captures that list the same way Sol-Attn does --
patching `PackedLayout.__init__` and keying on `id(position_ids)` -- so every
reference's exact key rows are known.

WHAT IT APPROXIMATES. The full attention matrix never exists: at 113k tokens one
head is 25 GB. Queries are pooled to one centroid per 64 rows -- the same pooling
sol_attn documents at ~5e-4 cosine for its routing. The DENOMINATOR is exact, streamed
over key chunks inside a memory budget. Pooling the keys as well was tried first and
is wrong here: logsumexp over a key block is near its MAX while a centroid is its
MEAN, so a row attending one sharp key had its tail understated and both references
came back at ~0.50 when the truth was ~0.005.

WHAT IT DOES NOT MEAN. Attention mass is where the model LOOKED, not what it took. A
row can attend a reference heavily and still not adopt its timbre. Treat a flat 50/50
split as evidence that no binding formed; do not read a 70/30 split as "the voice is
70% correct".
"""

import logging
import sys

import torch

from comfy_api.latest import io

BLOCK = 64                      # query pooling; matches sol_attn's routing granularity
AUDIO_LATENT_HZ = 40
FPS = 24

# id(position_ids) -> segment list. The layout object is kept alive on purpose so the
# id cannot be recycled underneath us.
_SEGMENTS = {}
_PATCHED = set()

# Accumulates across every recorded call. Written during sampling, read afterwards by
# MMH3RefAttentionMap, because the probe has nowhere to return an image from.
_RECORD = {"mass": None, "hits": 0, "labels": [], "query": "", "calls": 0,
           "layers": set(), "note": ""}


def reset_record():
    _RECORD.update({"mass": None, "hits": 0, "labels": [], "query": "", "calls": 0,
                    "layers": set(), "note": ""})


def _patch_packed_layout():
    """Capture every layout's segment list without mutating the layout."""
    mod = sys.modules.get("comfy.ldm.minimax.model")
    if mod is None:
        import comfy.ldm.minimax.model as mod           # noqa: F401
    layout_cls = getattr(mod, "PackedLayout", None)
    if layout_cls is None:
        raise RuntimeError("comfy.ldm.minimax.model has no PackedLayout; this build of "
                           "ComfyUI lays the H3 sequence out differently and the "
                           "reference spans cannot be located.")
    if id(layout_cls) in _PATCHED:
        return
    original = layout_cls.__init__

    def __init__(self, *a, **kw):
        original(self, *a, **kw)
        try:
            segs = getattr(self, "segments", None)
            pid = getattr(self, "position_ids", None)
            if segs and torch.is_tensor(pid):
                _SEGMENTS[id(pid)] = (self, list(segs))
        except Exception as exc:                        # never break model construction
            logging.info("[MMH3RefAttentionProbe] layout capture failed: %s", exc)

    layout_cls.__init__ = __init__
    _PATCHED.add(id(layout_cls))


def _spans_for(transformer_options):
    """(ref spans, target span, labels) for the sequence this call belongs to."""
    pid = (transformer_options or {}).get("position_ids")
    entry = None
    if torch.is_tensor(pid):
        entry = _SEGMENTS.get(id(pid))
    if entry is None and len(_SEGMENTS) == 1:
        entry = next(iter(_SEGMENTS.values()))          # only one layout in flight
    if entry is None:
        return None, None, []
    _layout, segs = entry
    refs, labels = [], []
    n_audio = n_img = n_vid = 0
    for a, b, kind in segs:
        if kind == "ref_audio":
            n_audio += 1
            labels.append("Audio %d" % n_audio)
            refs.append((int(a), int(b)))
        elif kind == "ref_img":
            n_img += 1
            labels.append("Picture %d" % n_img)
            refs.append((int(a), int(b)))
        elif kind == "ref_video":
            n_vid += 1
            labels.append("Video %d" % n_vid)
            refs.append((int(a), int(b)))
    return refs, {k: (int(a), int(b)) for a, b, k in segs}, labels


def _as_bhtd(x, heads, skip_reshape):
    """-> [B, H, T, D] whichever way the attention backend handed it over."""
    if skip_reshape:
        return x
    b, t, c = x.shape
    return x.view(b, t, heads, c // heads).transpose(1, 2)


@torch.no_grad()
def measure(q, k, refs, q_lo, q_hi, scale, budget_bytes=256 << 20):
    """Attention mass from pooled query blocks onto each reference span.

    Returns [n_blocks, n_refs] in 0..1, averaged over heads and batch.

    The denominator is computed EXACTLY, streamed over key chunks. An earlier cut
    pooled the non-reference keys into block centroids the way sol_attn pools its
    routed-out tail, and that is wrong for this measurement: logsumexp over a block
    is near its MAX, while a centroid is its MEAN, so a row attending one sharp key
    somewhere in the clip had its tail understated and the references came back at
    ~0.50 each when the true answer was ~0.005. A probe that reports "both references
    are being used" when neither is would be worse than no probe.

    Only the QUERY side is approximated -- one centroid per 64 rows, which is the
    pooling sol_attn documents at ~5e-4 cosine for its routing decisions.
    """
    q = q[:, :, q_lo:q_hi]                              # only the rows we asked about
    b, h, t, d = q.shape
    if t == 0:
        return None
    nb = (t + BLOCK - 1) // BLOCK
    pad = nb * BLOCK - t
    if pad:
        q = torch.cat([q, q[:, :, -1:].expand(b, h, pad, d)], dim=2)
    qb = q.view(b, h, nb, BLOCK, d).mean(dim=3).float()     # centroid per block
    kf = k.float()
    tk = kf.shape[2]

    # exact logsumexp over EVERY key, in chunks sized to a memory budget
    per_key = b * h * nb * 4
    chunk = max(256, min(tk, int(budget_bytes // max(1, per_key))))
    running = None
    for c0 in range(0, tk, chunk):
        s_c = torch.matmul(qb, kf[:, :, c0:c0 + chunk].transpose(-1, -2)) * scale
        lse = torch.logsumexp(s_c, dim=-1)                                  # [B,H,nb]
        running = lse if running is None else torch.logaddexp(running, lse)
        del s_c

    parts = []
    for (a, z) in refs:
        s = torch.matmul(qb, kf[:, :, a:z].transpose(-1, -2)) * scale
        parts.append(torch.logsumexp(s, dim=-1))
        del s

    mass = torch.exp(torch.stack(parts, dim=-1) - running.unsqueeze(-1))
    return mass.mean(dim=(0, 1))                                            # [nb, R]


class MMH3RefAttentionProbe(io.ComfyNode):
    """Record which reference each part of the clip attends to, during sampling."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3RefAttentionProbe",
            display_name="MMH3 Reference Attention Probe",
            category="MMH3Tools/utils",
            description=(
                "Records how much attention each REFERENCE receives from the target "
                "rows, over the length of the clip. H3 has no cross-attention -- "
                "references sit in the same sequence -- so this is a real measurement "
                "rather than an inference from the output. Wire it anywhere in the "
                "model chain and read the result with MMH3 Reference Attention Map.\n\n"
                "It never forms the attention matrix (one head is 25 GB at 113k "
                "tokens): queries are pooled to their block centroid and everything "
                "outside the references is folded into one pooled tail, the same "
                "approximation sol_attn makes for its non-routed blocks.\n\n"
                "Attention mass is where the model LOOKED, not what it took."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "query_rows", options=["audio", "video", "both"], default="audio",
                    tooltip="Which target rows to measure FROM. 'audio' answers 'which "
                            "voice reference is this speech using'."),
                io.String.Input(
                    "layers", default="", optional=True,
                    tooltip="DiT blocks to record, e.g. '20-30' or '24,28'. Empty "
                            "records every block, which is slower and blurs early "
                            "layers into late ones. Middle blocks are the usual "
                            "choice."),
                io.Boolean.Input(
                    "enabled", default=True, optional=True,
                    tooltip="Off is a clean pass-through, so this can stay wired for "
                            "an A/B."),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, query_rows, layers="", enabled=True) -> io.NodeOutput:
        if not enabled:
            return io.NodeOutput(model)

        want = set()
        for piece in (layers or "").replace(",", " ").split():
            if "-" in piece:
                a, b = piece.split("-", 1)
                want.update(range(int(a), int(b) + 1))
            elif piece.strip():
                want.add(int(piece))

        _patch_packed_layout()
        reset_record()
        _RECORD["query"] = query_rows

        m = model.clone()
        opts = m.model_options.setdefault("transformer_options", {})
        previous = opts.get("optimized_attention_override")

        def override(func, q, k, v, heads, mask=None, attn_precision=None,
                     skip_reshape=False, skip_output_reshape=False, **kwargs):
            def passthrough():
                target = func if previous is None else (
                    lambda *a, **kw: previous(func, *a, **kw))
                return target(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                              skip_reshape=skip_reshape,
                              skip_output_reshape=skip_output_reshape, **kwargs)

            topts = kwargs.get("transformer_options") or {}
            block = topts.get("sol_block")
            if want and block is not None and block not in want:
                return passthrough()
            try:
                _record(q, k, heads, skip_reshape, topts, kwargs.get("scale"), block)
            except Exception as exc:                    # a probe must never break a run
                if not _RECORD["note"]:
                    _RECORD["note"] = "recording stopped: %s" % exc
                    logging.warning("[MMH3RefAttentionProbe] %s", exc)
            return passthrough()

        opts["optimized_attention_override"] = override
        return io.NodeOutput(m)


@torch.no_grad()
def _record(q, k, heads, skip_reshape, topts, scale, block):
    refs, spans, labels = _spans_for(topts)
    if not refs or not spans:
        return
    qb = _as_bhtd(q, heads, skip_reshape)
    kb = _as_bhtd(k, heads, skip_reshape)
    d = qb.shape[-1]
    scale = float(scale) if scale else d ** -0.5

    which = _RECORD["query"]
    target = []
    if which in ("audio", "both") and "audio" in spans:
        target.append(spans["audio"])
    if which in ("video", "both") and "video" in spans:
        target.append(spans["video"])
    if not target:
        return
    lo = min(a for a, _ in target)
    hi = max(b for _, b in target)

    mass = measure(qb, kb, refs, lo, hi, scale)
    if mass is None:
        return
    prev = _RECORD["mass"]
    if prev is None or prev.shape != mass.shape:
        _RECORD["mass"] = mass.cpu()
        _RECORD["hits"] = 1
    else:
        _RECORD["mass"] = prev + mass.cpu()
        _RECORD["hits"] += 1
    _RECORD["labels"] = labels
    _RECORD["calls"] += 1
    if block is not None:
        _RECORD["layers"].add(int(block))


def _ramp(v):
    """0..1 -> RGB. Dark blue (ignored) through green to yellow (dominant)."""
    v = v.clamp(0, 1)
    r = (v * 2 - 0.4).clamp(0, 1)
    g = (v * 1.8 - 0.1).clamp(0, 1)
    b = (0.45 - v * 1.2).clamp(0, 1) + (v < 0.02).float() * 0.10
    return torch.stack([r, g, b], dim=-1)


class MMH3RefAttentionMap(io.ComfyNode):
    """Read what the probe recorded: a [reference x time] heatmap."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3RefAttentionMap",
            display_name="MMH3 Reference Attention Map",
            category="MMH3Tools/utils",
            description=(
                "Turns what MMH3 Reference Attention Probe recorded into a heatmap: "
                "one row per reference, time along x. Bright is attended, dark is "
                "ignored.\n\n"
                "Read a flat split across two voice references as evidence that no "
                "binding formed -- the model is not choosing per speaker. Do NOT read "
                "70/30 as 'the voice is 70% right': this is where attention went, not "
                "what the output took from it."
            ),
            inputs=[
                io.Int.Input("height", default=64, min=8, max=512, step=8,
                             tooltip="Pixel height of each reference's band."),
                io.Int.Input("width", default=1024, min=64, max=4096, step=64,
                             tooltip="Pixels across. The recording is one column per "
                                     "64-row query block; this only resamples it."),
                io.Boolean.Input(
                    "normalize_columns", default=False, optional=True,
                    tooltip="Scale each time column so the references sum to 1, "
                            "showing WHICH reference won rather than how much total "
                            "attention the references got. Hides the case where all "
                            "of them were ignored, so leave OFF for the first look."),
            ],
            outputs=[
                io.Image.Output(display_name="heatmap"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, height, width, normalize_columns=False) -> io.NodeOutput:
        mass, hits = _RECORD["mass"], _RECORD["hits"]
        if mass is None or not hits:
            blank = torch.zeros(1, height, width, 3)
            return io.NodeOutput(blank, (
                "MMH3 Reference Attention Map -- nothing recorded.\n\n"
                "  The probe records during SAMPLING, so run a generation with it "
                "wired into the model chain first.\n"
                "  If you did: the run had no reference blocks, or `layers` excluded "
                "every block that ran."
                + ("\n  ! " + _RECORD["note"] if _RECORD["note"] else "")))

        m = (mass / float(hits)).transpose(0, 1)             # [refs, blocks]
        labels = _RECORD["labels"] or ["ref %d" % i for i in range(m.shape[0])]

        shown = m
        if normalize_columns:
            shown = m / m.sum(dim=0, keepdim=True).clamp_min(1e-6)

        img = torch.nn.functional.interpolate(
            shown[None, None], size=(m.shape[0], width), mode="nearest")[0, 0]
        rgb = _ramp(img).repeat_interleave(height, dim=0)     # [refs*height, width, 3]
        # a dark rule between bands so adjacent references are distinguishable
        for i in range(1, m.shape[0]):
            rgb[i * height - 1] = 0.12
        out = rgb[None]

        lines = ["MMH3 Reference Attention Map -- %d attention calls over %d block(s)"
                 % (_RECORD["calls"], len(_RECORD["layers"]) or 0), ""]
        lines.append("  measured FROM the %s rows, %d query blocks of %d rows"
                     % (_RECORD["query"], m.shape[1], BLOCK))
        if _RECORD["layers"]:
            ls = sorted(_RECORD["layers"])
            lines.append("  DiT blocks recorded: %s" % (
                "%d-%d" % (ls[0], ls[-1]) if len(ls) > 3 else ls))
        lines.append("")
        lines.append("  mean attention mass per reference, over the whole clip")
        order = torch.argsort(m.mean(dim=1), descending=True)
        for i in order.tolist():
            row = m[i]
            lines.append("    %-12s mean %.4f   min %.4f   max %.4f"
                         % (labels[i], row.mean(), row.min(), row.max()))
        total = float(m.sum(dim=0).mean())
        lines.append("")
        lines.append("  references together take %.1f%% of the row on average; the "
                     "rest went to the clip itself." % (total * 100.0))
        if m.shape[0] >= 2:
            # Judge PER COLUMN, never on the time average. Under perfect alternation
            # -- reference A holding the first half, B the second, which is exactly
            # what a working binding looks like -- the two averages come out equal,
            # and an average-based test calls the best possible result "no binding".
            top2 = torch.topk(m, 2, dim=0).values                  # [2, blocks]
            margin = (top2[0] - top2[1])
            winner = torch.argmax(m, dim=0)
            switches = int((winner[1:] != winner[:-1]).sum())
            decisive = float(margin.mean())
            share = [float((winner == i).float().mean()) for i in range(m.shape[0])]

            lines.append("")
            lines.append("  per-moment margin %.4f | lead changes %d time(s)"
                         % (decisive, switches))
            lines.append("  share of the clip each reference leads: %s"
                         % ", ".join("%s %.0f%%" % (labels[i], share[i] * 100)
                                     for i in range(m.shape[0])))
            lines.append("")
            if decisive < 0.01:
                lines.append("  ! NO BINDING: no reference leads at any moment. The "
                             "model is not selecting per speaker, which is what a "
                             "reference-swap test would also show.")
            elif switches == 0:
                lines.append("  ! ONE REFERENCE LEADS THE WHOLE CLIP. If both speakers "
                             "talk in this clip, that is the failure people describe as "
                             "the model deciding a voice belongs to a face -- it never "
                             "hands over.")
            else:
                lines.append("  the lead changes %d time(s), so a per-speaker selection "
                             "IS forming. Compare the switch points against who is "
                             "actually speaking -- binding to the WRONG reference looks "
                             "identical here to binding to the right one." % switches)
        lines.append("")
        lines.append("  Attention mass is where the model LOOKED. It is not proof the "
                     "output took the voice.")
        if _RECORD["note"]:
            lines.append("  ! " + _RECORD["note"])
        return io.NodeOutput(out, "\n".join(lines))
