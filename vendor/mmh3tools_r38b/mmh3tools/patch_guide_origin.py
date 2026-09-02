"""Anchor keyframe guides on the TARGET origin when references are present.

WHAT IS WRONG
-------------
#15439 computes a guide's time coordinate as

    cond_t = float(text_len) + FRAME_RESCALE * kf["resolved_frame_index"]

but the target does not begin at `text_len`. References are laid out from a cursor
that starts there and each block advances it, and the target uses the cursor's final
value as its origin. So every guide lands `ref_advance` units BEFORE the clip it is
supposed to anchor. Measured on the real PackedLayout, guide origin against target
origin:

    no refs             0
    one image ref      -1
    audio / voice ref  -320
    video_audio ref    -37
    image + audio      -321

Nothing errors. The guide simply anchors into the reference region, and `cond_audio`
goes with it, so a carried tail's AUDIO lands early too. It matters MORE under
#15439, not less, because the same PR fixes the `cond_video_latents` clobber
specifically so guides and references can coexist -- it makes the broken
configuration reachable.

WHY A WRAP RATHER THAN A CORE EDIT
----------------------------------
No PR carries this fix yet, so a core edit is a diff to re-apply after every `git
pull` and to remember when reading a bug report from someone who does not have it.
This lives in the pack instead, and comes out the moment upstream lands its own.

⚠ OBSOLETE ON CURRENT CORE (2026-08-13). **#15439 merged**, and the merged version
anchors the guide on the target origin by itself -- measured on the live class,
guide 11.000 against target 11.000 with one image reference, where the draft gave
-1. The wrap would now OVER-correct by exactly the reference advance.

It does not, because the self-test catches it: the shift is applied, the result is
compared against the target origin, and a mismatch rolls the wrap back and leaves
stock alone. `is_applied()` returns False and the log says so. That is the designed
behaviour and the reason the self-test exists -- an obsolete patch that silently
double-corrects is worse than no patch.

Kept rather than deleted because it is inert and self-disabling, and because anyone
on a core predating the merge still needs it. Delete it once the required ComfyUI
version is unambiguously past #15439.

Also why a WRAP rather than the source-rewriting some packs use for the same file:
matching against core's source text breaks when ComfyUI reformats that block, and
embedding core's lines in this file would put GPL-3.0 source in an MIT pack. This
touches neither -- `PackedLayout.__init__` is a plain method with no closure cells,
so it wraps at a callable boundary, and the correction is a pure position shift
applied after stock has built the layout.

THREE PROPERTIES, matching the pack's other patches
---------------------------------------------------
  * INERT UNLESS BOTH ARE PRESENT. With only guides, or only references, the cursor
    never leaves `text_len` and stock is already correct. Untouched.
  * ABSOLUTE, not incremental. It shifts by the advance computed from the refs it
    was given, so applying it twice to the same layout is impossible -- each
    __init__ builds its own.
  * SELF-TESTED at import against the live class, and refuses to install rather
    than silently misplacing a guide.

Translation, not per-row assignment: `+= advance` preserves whatever intra-block
structure stock built -- row order, the rows-per-step factor, fractional offsets --
so nothing about the block's internals is assumed.
"""

import logging

_MARK = "_mmh3_guide_origin_patched"
_state = {"done": False, "ok": False, "msg": ""}


def _ref_cursor_advance(mm, refs):
    """How far the reference blocks push the target origin past `text_len`.

    Mirrors the cursor arithmetic in `PackedLayout.__init__`'s `if refs:` block.
    Keep the two in step; a drift here is a silently misplaced guide, which is the
    exact failure this exists to remove.
    """
    advance = 0.0
    for blk in (refs or []):
        kind = blk.get("kind")
        if kind == "image":
            advance += 1.0
        elif kind == "audio":
            advance += float(blk.get("ref_audio_t", 0))
        elif kind in ("video", "video_audio"):
            advance += max(float(blk.get("ref_audio_t", 0)),
                           sum(mm._video_t_spans(blk["latent_t"])))
    return advance


def _selftest(mm, cls):
    """A guide alongside one image reference must land ON the target origin.

    Also checks the guide-only case is untouched, since an over-eager patch that
    shifted every layout would pass a drift check and break everything else.
    """
    import torch
    kf = [{"resolved_frame_index": 0, "latent": torch.zeros([1, 24, 1, 4, 4])}]
    ref = [{"kind": "image", "latent_h": 4, "latent_w": 4,
            "latent": torch.zeros([1, 24, 1, 4, 4])}]

    def origins(refs):
        lay = cls(8, 7, 4, 4, 8, keyframes=kf, refs=refs)
        seg = {k: a for a, _b, k in lay.segments}
        if "cond" not in seg or "video" not in seg:
            return None
        return (float(lay.position_ids[seg["cond"], 0]),
                float(lay.position_ids[seg["video"], 0]))

    with_ref = origins(ref)
    if with_ref is None:
        return False, "no cond/video segments in the probe layout"
    if abs(with_ref[0] - with_ref[1]) > 1e-6:
        return False, ("guide %.3f still does not match target %.3f with one reference"
                       % with_ref)

    alone = origins(None)
    if alone is None or abs(alone[0] - float(8)) > 1e-6:
        return False, "the guide-only case moved; it should be untouched"
    return True, "guide origin follows the target when references are present"


def apply(verbose=True):
    """Idempotent. Returns (ok, message)."""
    if _state["done"]:
        return _state["ok"], _state["msg"]
    _state["done"] = True

    try:
        import comfy.ldm.minimax.model as mm
    except Exception as e:
        _state["msg"] = "MiniMax H3 not present in this ComfyUI (%s)" % e
        return False, _state["msg"]

    cls = getattr(mm, "PackedLayout", None)
    if cls is None:
        _state["msg"] = "comfy.ldm.minimax.model.PackedLayout not found"
        return False, _state["msg"]
    if getattr(cls, _MARK, False):
        _state["ok"] = True
        _state["msg"] = "already applied"
        return True, _state["msg"]

    # Nothing to correct on a core that cannot express interior guides at all.
    import inspect
    try:
        src = inspect.getsource(cls.__init__)
    except Exception:
        src = ""
    if "only first/last keyframe anchors" in src:
        _state["msg"] = ("this ComfyUI has no interior guides (#15439 not applied), "
                         "so there is no guide origin to correct")
        return False, _state["msg"]
    if "guide_origin" in src:
        cls._mmh3_guide_origin_patched = True
        _state["ok"] = True
        _state["msg"] = "core already anchors guides on the target origin"
        return True, _state["msg"]

    original = cls.__init__

    def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t,
                 keyframes=None, refs=None, *args, **kwargs):
        original(self, text_len, latent_t, latent_h, latent_w, audio_t,
                 keyframes=keyframes, refs=refs, *args, **kwargs)
        if not (keyframes and refs):
            return
        advance = _ref_cursor_advance(mm, refs)
        if advance == 0.0:
            return
        for a, b, kind in self.segments:
            if kind in ("cond", "cond_audio"):
                self.position_ids[a:b, 0] += advance

    cls.__init__ = __init__
    ok, msg = _selftest(mm, cls)
    if not ok:
        cls.__init__ = original
        _state["msg"] = msg + " - rolled back, stock behaviour kept"
        logging.warning("[MMH3Tools] guide-origin patch: " + _state["msg"])
        return False, _state["msg"]

    cls._mmh3_guide_origin_patched = True
    _state["ok"] = True
    _state["msg"] = "guide origin corrected (%s)" % msg
    if verbose:
        logging.info("[MMH3Tools] " + _state["msg"])
    return True, _state["msg"]


def is_applied():
    return bool(_state["ok"])
