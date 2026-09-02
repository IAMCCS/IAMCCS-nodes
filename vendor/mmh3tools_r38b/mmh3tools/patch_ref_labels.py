"""Let a reference item choose its own text label, e.g. `<base_video>`.

WHY
---
`MiniMaxH3Tokenizer.tokenize_with_weights` emits a reference's label as ORDINARY
TEXT immediately before its vision block:

    add_text("<Video %d>: " % counters["video"])
    add_vision(frames[i:i + 2], video_block=True)

`_text_ids()` tokenizes that string like any other. There is no special token and no
vocabulary entry for `<Video 1>` -- the tag is a convention written in plain text,
and the format is simply hardcoded.

That matters because the hosted regeneration endpoint does not describe the 768p as
a reference video. It sends it with **`role=base_video`**, a role distinct from
`reference_video`. Whether H3 was trained on a matching TEXT tag is not something
the open layout can answer -- the layout has kinds, not roles -- but it is not a
question that has to be settled by argument either. The tag is a string; the model
can be handed it and asked.

This wrap adds one optional key. An item carrying `"label"` gets that text instead
of the counter-generated one:

    {"type": "video", "label": "<base_video>", "data": ..., "timestamps": ...}
    ->  "<base_video>: " <vision blocks>

WHAT IT DOES NOT CHANGE
-----------------------
Only the emitted string. The vision blocks, the per-2-frame timestamp tags, the
counters, and the DiT-side `minimax_refs` layout are all untouched -- a labelled
item still advances `counters` exactly as before, so every OTHER item's number is
unaffected and no prompt tag shifts underneath it.

INERT unless an item carries "label", so a graph that never sets one produces
byte-identical tokens to stock. Self-tested at import against the live class; it
declines to install rather than corrupt conditioning.
"""

import logging

_APPLIED = False
_ORIG = None


def is_applied():
    return _APPLIED


def _wrap(orig):
    def tokenize_with_weights(self, text, return_word_ids=False, images=[],
                              minimax_ref_items=None, **kwargs):
        items = minimax_ref_items
        if not items or not any(isinstance(i, dict) and i.get("label") for i in items):
            return orig(self, text, return_word_ids=return_word_ids, images=images,
                        minimax_ref_items=items, **kwargs)

        # Rewrite in two passes rather than reimplementing the emitter: run stock on
        # each item ALONE to get its exact token sequence, then swap the leading label
        # tokens. Cheaper to keep correct than forking the whole method, and it tracks
        # any change core makes to the vision/timestamp emission for free.
        #
        # The return is {encoder_key: [entries]}, so unwrap per item and rewrap once.
        # The key is read from stock's own output rather than hardcoded.
        key, out = None, []
        counters = {"image": 0, "audio": 0, "video": 0}
        default = {"image": "<Picture %d>: ", "audio": "<Audio %d>: ",
                   "video": "<Video %d>: "}
        for item in items:
            kind = item["type"]
            counters[kind] += 1
            packed = orig(self, "", return_word_ids=False, images=[],
                          minimax_ref_items=[item], **kwargs)
            if key is None:
                key = next(iter(packed))
            one = list(packed[key][0])
            # stock numbered this lone item as 1; drop that label and splice in the
            # real number, or the caller's override
            lead_n = len(_text_entries(self, default[kind] % 1))
            label = item.get("label") or (default[kind] % counters[kind])
            if not label.endswith(": "):
                label += ": "
            out.extend(_text_entries(self, label) + one[lead_n:])

        out.extend(_text_entries(self, text))
        if not out:
            out.append((151643, 1.0))
        if return_word_ids:
            out = [t + (0,) for t in out]
        return {key or "qwen3vl_32b": [out]}

    return tokenize_with_weights


def apply():
    """Install the wrap. Safe to call repeatedly."""
    global _APPLIED, _ORIG
    if _APPLIED:
        return True
    try:
        from comfy.text_encoders.minimax import MiniMaxH3Tokenizer as T
    except Exception as e:
        logging.info("[MMH3Tools] ref-label wrap: no MiniMax tokenizer (%s); skipped", e)
        return False
    if not hasattr(T, "tokenize_with_weights"):
        logging.warning("[MMH3Tools] ref-label wrap: tokenize_with_weights is gone; "
                        "core changed, NOT installing")
        return False
    orig = T.tokenize_with_weights
    wrapped = _wrap(orig)
    if not _self_test(T, orig, wrapped):
        logging.warning("[MMH3Tools] ref-label wrap: self-test FAILED, NOT installing. "
                        "`label` on a reference item will be ignored.")
        return False
    T.tokenize_with_weights = wrapped
    _ORIG, _APPLIED = orig, True
    logging.info("[MMH3Tools] ref-label wrap installed (reference items may set `label`)")
    return True


def _text_entries(tok, s):
    """A text run as (token, weight) entries, using whatever core currently does.

    Core moved this twice: a private `_text_ids()` returning bare ids, then (#15808)
    a pass through the inner tokenizer with `disable_weights=True`, which is what
    also makes `embedding:` resolve. Calling core's own path rather than
    reimplementing it means this wrap tracks future moves for free -- and when it
    cannot, the self-test refuses to install rather than dropping labels quietly.
    """
    if not s:
        return []
    legacy = getattr(tok, "_text_ids", None)
    if legacy is not None:                       # cores older than #15808
        return [(t, 1.0) for t in legacy(s)]
    inner = getattr(tok, "qwen3vl_32b", None)
    if inner is None:
        raise AttributeError("MiniMaxH3Tokenizer has neither _text_ids nor qwen3vl_32b")
    batches = inner.tokenize_with_weights(s, return_word_ids=False,
                                          disable_weights=True)
    if len(batches) != 1:
        raise ValueError("label text spilled into %d batches" % len(batches))
    return list(batches[0])


def _self_test(T, orig, wrapped):
    """Unlabelled items must tokenize IDENTICALLY to stock, labelled ones must differ."""
    try:
        import torch
        tok = T.__new__(T)
        try:
            tok.__init__()
        except Exception:
            pass
        # neither helper present means core moved again; refuse rather than guess
        if not hasattr(tok, "_text_ids") and not hasattr(tok, "qwen3vl_32b"):
            return False
        def flat(packed):
            return list(packed[next(iter(packed))][0])

        img = torch.zeros([1, 32, 32, 3])
        # two items, so the counter path is exercised rather than just the n=1 case
        plain = [{"type": "image", "data": img}, {"type": "image", "data": img}]
        a = flat(orig(tok, "hello", minimax_ref_items=plain))
        b = flat(wrapped(tok, "hello", minimax_ref_items=plain))
        if a != b:
            logging.warning("[MMH3Tools] ref-label wrap: unlabelled path diverged from "
                            "stock (%d vs %d entries)", len(a), len(b))
            return False
        labelled = [{"type": "image", "data": img},
                    {"type": "image", "label": "<base_video>", "data": img}]
        c = flat(wrapped(tok, "hello", minimax_ref_items=labelled))
        if c == a:
            logging.warning("[MMH3Tools] ref-label wrap: a label changed nothing")
            return False
        # the FIRST item must be untouched: labelling one item must not renumber
        # or otherwise disturb any other
        if c[:len(a) // 2] != a[:len(a) // 2]:
            logging.warning("[MMH3Tools] ref-label wrap: labelling one item disturbed "
                            "another")
            return False
        return True
    except Exception as e:
        logging.warning("[MMH3Tools] ref-label wrap self-test errored: %r", e)
        return False
