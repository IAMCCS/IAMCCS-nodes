"""Lazy runtime patch for MiniMax H3 interior latent anchors and timeline audio.

Nothing is patched when ComfyUI imports this node pack. The hook is installed
only when a continuation node is actually executed. Once installed, the
wrapper is gated on this suite's own marker keys, so unrelated H3 graphs stay
on the stock path.
"""

import inspect
import logging

from .patch_utils import classify_callable

HC_INDEX = "h3_continuous_index"
HC_AUDIO_END_FRAME = "h3_continuous_audio_end_frame"
LAYOUT_PATCH_MARKER = "_herrgotts_h3_infinite_layout_patch"

_LOG = logging.getLogger("h3_continuous")
_ORIGINAL_INIT = None
_ORIGINAL_ACCEPTS_FRAME_COUNT = None
_APPLIED = False
_MM = None

_KNOWN_EXTERNAL_MARKERS = (
    ("_h3_motion_context_layout_patch", "ComfyUI-H3-Motion-Context"),
)


def is_compatible_foreign_owner():
    """Whether LongerVid may safely compose over the live layout wrapper."""
    try:
        mm = _import_mm()
        fn = getattr(getattr(mm, "PackedLayout", None), "__init__", None)
    except Exception:
        return False
    return bool(fn is not None and getattr(fn, "_h3_motion_context_layout_patch", False))


def _import_mm():
    import comfy.ldm.minimax.model as mm
    return mm


def _accepts_frame_count(fn):
    """Probe the live PackedLayout signature instead of a ComfyUI version."""
    try:
        return "frame_count" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return "frame_count" in getattr(getattr(fn, "__code__", None), "co_varnames", ())


def _call_init(fn, layout, text_len, latent_t, latent_h, latent_w, audio_t,
               keyframes=None, refs=None, frame_count=None):
    if _accepts_frame_count(fn):
        return fn(layout, text_len, latent_t, latent_h, latent_w, audio_t,
                  keyframes=keyframes, refs=refs, frame_count=frame_count)
    return fn(layout, text_len, latent_t, latent_h, latent_w, audio_t,
              keyframes=keyframes, refs=refs)


def get_layout_patch_status():
    try:
        mm = _import_mm()
    except Exception as exc:
        return None, f"cannot import comfy.ldm.minimax.model: {exc}"
    cls = getattr(mm, "PackedLayout", None)
    fn = getattr(cls, "__init__", None) if cls is not None else None
    if cls is None or fn is None:
        return None, "ComfyUI PackedLayout.__init__ is unavailable"
    status = classify_callable(cls, fn, LAYOUT_PATCH_MARKER, _KNOWN_EXTERNAL_MARKERS)
    return status, None


def _ref_cursor_advance(mm, refs):
    if not refs:
        return 0.0
    cursor = 0.0
    for blk in refs:
        kind = blk.get("kind")
        if kind == "image":
            cursor += 1.0
        elif kind == "audio":
            cursor += float(blk.get("ref_audio_t", 0))
        elif kind in ("video", "video_audio"):
            rt = float(blk.get("ref_audio_t", 0))
            vt = int(blk.get("latent_t", 0))
            cursor += max(rt, sum(mm._video_t_spans(vt)))
    return cursor


def _cond_t(mm, text_len, latent_t, frame_count, p):
    p = float(p)
    if p == 0.0:
        return float(text_len)
    if frame_count is not None and p == float(frame_count - 1):
        return float(text_len) + sum(mm._video_t_spans(latent_t)) - mm.FRAME_RESCALE
    return float(text_len) + mm.FRAME_RESCALE * p


def _fix_keyframes(mm, layout, text_len, latent_t, frame_count, keyframes, refs):
    """Move only LongerVid-marked keyframes onto the target timeline."""
    offset = _ref_cursor_advance(mm, refs)
    cond_spans = [(a, b) for a, b, kind in layout.segments if kind == "cond"]
    if len(cond_spans) != len(keyframes):
        raise RuntimeError(
            "h3_continuous: PackedLayout cond segment count changed; refusing unsafe continuation"
        )
    for (a, b), kf in zip(cond_spans, keyframes):
        if HC_INDEX not in kf:
            continue
        p = kf[HC_INDEX]
        layout.position_ids[a:b, 0] = _cond_t(mm, text_len, latent_t, frame_count, p) + offset


def _fix_audio(mm, layout, text_len, refs):
    marked = [(i, r) for i, r in enumerate(refs or []) if HC_AUDIO_END_FRAME in r]
    if not marked:
        return
    if len(marked) != 1:
        raise RuntimeError("h3_continuous: exactly one timeline audio context block is supported")
    ref_index, blk = marked[0]
    if ref_index != 0 or blk.get("kind") != "audio":
        raise RuntimeError(
            "h3_continuous: timeline audio context must be the first H3 ref block"
        )
    rt = int(blk.get("ref_audio_t", 0))
    if rt <= 0:
        return

    ref_audio_segments = [(a, b) for a, b, kind in layout.segments if kind == "ref_audio"]
    if not ref_audio_segments:
        raise RuntimeError("h3_continuous: PackedLayout produced no ref_audio segment")
    a, b = ref_audio_segments[0]
    if (b - a) != rt * 2:
        raise RuntimeError(
            f"h3_continuous: unexpected audio row count {b-a} for {rt} latent steps"
        )

    target_origin = float(text_len) + _ref_cursor_advance(mm, refs)
    desired_end = target_origin + mm.FRAME_RESCALE * float(blk[HC_AUDIO_END_FRAME])
    stock_end = float(text_len) + float(rt)
    layout.position_ids[a:b, 0] += desired_end - stock_end


def _run_self_test(mm, original_init, patched_init):
    """Validate the live ComfyUI layout before committing the wrapper."""
    import torch

    text_len, latent_t, lh, lw, audio_t = 7, 7, 22, 38, 16
    frame_count = sum(mm.FRAME_PER_TOKEN[k % 5] for k in range(latent_t))

    class _StubLatent:
        def __init__(self, shape):
            self.shape = shape

    # Current ComfyUI emits a cond segment only when a keyframe carries a
    # video latent.  The layout reads shape only, so this allocation-free
    # stand-in keeps the same self-test valid across old and current cores.
    stub = _StubLatent((1, 1, 1, lh, lw))

    # LongerVid endpoint coordinates must reproduce stock H3 exactly.
    stock_kf = [
        {"resolved_frame_index": 0, "latent": stub},
        {"resolved_frame_index": frame_count - 1, "latent": stub},
    ]
    custom_kf = [
        {"resolved_frame_index": 0, HC_INDEX: 0, "latent": stub},
        {"resolved_frame_index": 0, HC_INDEX: frame_count - 1, "latent": stub},
    ]
    s = mm.PackedLayout.__new__(mm.PackedLayout)
    stock_has_frame_count = _accepts_frame_count(original_init)
    compat_frame_count = frame_count if stock_has_frame_count else None
    _call_init(original_init, s, text_len, latent_t, lh, lw, audio_t,
               keyframes=stock_kf, frame_count=frame_count)
    c = mm.PackedLayout.__new__(mm.PackedLayout)
    _call_init(original_init, c, text_len, latent_t, lh, lw, audio_t,
               keyframes=custom_kf, frame_count=frame_count)
    _fix_keyframes(mm, c, text_len, latent_t, compat_frame_count, custom_kf, None)
    if not torch.equal(s.position_ids, c.position_ids):
        raise RuntimeError("custom endpoint positions differ from stock H3")

    # The continuation Last Frame is also marked in v1.1.4, allowing the
    # wrapper to leave every unmarked stock graph untouched.
    refs = [{"kind": "audio", "ref_audio_t": 11, HC_AUDIO_END_FRAME: 5.0}]
    marked_last = [{"resolved_frame_index": 0, HC_INDEX: frame_count - 1, "latent": stub}]
    n = mm.PackedLayout.__new__(mm.PackedLayout)
    _call_init(original_init, n, text_len, latent_t, lh, lw, audio_t,
               keyframes=marked_last, refs=refs, frame_count=frame_count)
    _fix_keyframes(mm, n, text_len, latent_t, compat_frame_count, marked_last, refs)
    cond_a = next(a for a, _, kind in n.segments if kind == "cond")
    video_a = next(a for a, _, kind in n.segments if kind == "video")
    expected_last = (
        float(n.position_ids[video_a, 0])
        + sum(mm._video_t_spans(latent_t))
        - mm.FRAME_RESCALE
    )
    if abs(float(n.position_ids[cond_a, 0]) - expected_last) > 1e-9:
        raise RuntimeError("marked Last Frame was not shifted onto the target timeline")

    canonical = [
        {"resolved_frame_index": 0, HC_INDEX: p, "latent": stub}
        for p in (0, 1, 5, 9, 13, 17, 18, 22)
    ]
    cr = mm.PackedLayout.__new__(mm.PackedLayout)
    _call_init(original_init, cr, text_len, latent_t, lh, lw, audio_t,
               keyframes=canonical, frame_count=frame_count)
    _fix_keyframes(mm, cr, text_len, latent_t, compat_frame_count, canonical, None)
    cts = [float(cr.position_ids[a, 0]) for a, _, kind in cr.segments if kind == "cond"]
    expected_cts = [float(text_len) + mm.FRAME_RESCALE * p for p in (0, 1, 5, 9, 13, 17, 18, 22)]
    if any(abs(a - b) > 1e-9 for a, b in zip(cts, expected_cts)):
        raise RuntimeError("canonical phase-aligned keyframe times changed")

    run = [
        {"resolved_frame_index": 0, HC_INDEX: p, "latent": stub}
        for p in (0, 4, 8, 9, 13, 17)
    ]
    r = mm.PackedLayout.__new__(mm.PackedLayout)
    _call_init(original_init, r, text_len, latent_t, lh, lw, audio_t,
               keyframes=run, frame_count=frame_count)
    _fix_keyframes(mm, r, text_len, latent_t, compat_frame_count, run, None)
    ts = [float(r.position_ids[a, 0]) for a, _, kind in r.segments if kind == "cond"]
    if any(ts[i] >= ts[i + 1] for i in range(len(ts) - 1)):
        raise RuntimeError("interior keyframe times are not strictly increasing")

    # Crucial v1.1.4 isolation test: unmarked keyframes+refs must be exactly the
    # stock layout even after the wrapper exists.
    unrelated_kf = [{"resolved_frame_index": frame_count - 1, "latent": stub}]
    unrelated_refs = [{"kind": "audio", "ref_audio_t": 5}]
    a = mm.PackedLayout.__new__(mm.PackedLayout)
    _call_init(original_init, a, text_len, latent_t, lh, lw, audio_t,
               keyframes=unrelated_kf, refs=unrelated_refs, frame_count=frame_count)
    b = mm.PackedLayout.__new__(mm.PackedLayout)
    patched_init(b, text_len, latent_t, lh, lw, audio_t,
                 keyframes=unrelated_kf, refs=unrelated_refs, frame_count=frame_count)
    if not torch.equal(a.position_ids, b.position_ids) or a.segments != b.segments:
        raise RuntimeError("unmarked H3 graph changed under the continuation layout wrapper")


def install_layout_patch():
    global _ORIGINAL_INIT, _ORIGINAL_ACCEPTS_FRAME_COUNT, _APPLIED, _MM
    if _APPLIED:
        return True

    status, err = get_layout_patch_status()
    if status is None:
        _LOG.error("h3_continuous: layout patch unavailable: %s", err)
        return False
    if status.state == "ours":
        _APPLIED = True
        _LOG.info("IAMCCS H3 LongerVid: compatible layout patch is already active")
        return True
    if status.state == "foreign" and not is_compatible_foreign_owner():
        _LOG.error(
            "h3_continuous: H3 runtime-patch conflict: %s already owns "
            "PackedLayout.__init__ (%s). Disable one H3 chaining pack and restart ComfyUI.",
            status.owner, status.module,
        )
        return False
    if status.state == "foreign":
        _LOG.info(
            "IAMCCS H3 LongerVid: composing its marker-gated layout hook over "
            "the compatible %s owner (%s)", status.owner, status.module,
        )

    mm = _import_mm()
    _MM = mm
    _ORIGINAL_INIT = mm.PackedLayout.__init__
    _ORIGINAL_ACCEPTS_FRAME_COUNT = _accepts_frame_count(_ORIGINAL_INIT)

    def patched_init(self, text_len, latent_t, latent_h, latent_w, audio_t,
                     keyframes=None, refs=None, frame_count=None):
        _call_init(_ORIGINAL_INIT, self, text_len, latent_t, latent_h, latent_w,
                   audio_t, keyframes=keyframes, refs=refs, frame_count=frame_count)
        compat_frame_count = frame_count if _ORIGINAL_ACCEPTS_FRAME_COUNT else None
        has_ours_kf = bool(keyframes) and any(HC_INDEX in k for k in keyframes)
        has_ours_audio = bool(refs) and any(HC_AUDIO_END_FRAME in r for r in refs)
        if has_ours_kf:
            _fix_keyframes(mm, self, text_len, latent_t, compat_frame_count, keyframes, refs)
        if has_ours_audio:
            _fix_audio(mm, self, text_len, refs)
        # No LongerVid marker -> the wrapped graph is returned unchanged.

    setattr(patched_init, LAYOUT_PATCH_MARKER, True)
    # Keep the shared marker visible on the outer wrapper so a subsequently
    # loaded Motion Context copy recognises the composite as compatible.
    if getattr(_ORIGINAL_INIT, "_h3_motion_context_layout_patch", False):
        setattr(patched_init, "_h3_motion_context_layout_patch", True)

    try:
        _run_self_test(mm, _ORIGINAL_INIT, patched_init)
    except Exception as exc:
        _LOG.error(
            "h3_continuous: live ComfyUI layout self-test FAILED (%s). "
            "No layout patch was installed.", exc,
        )
        _ORIGINAL_INIT = None
        _ORIGINAL_ACCEPTS_FRAME_COUNT = None
        _MM = None
        return False

    mm.PackedLayout.__init__ = patched_init
    _APPLIED = True
    _LOG.info(
        "IAMCCS H3 LongerVid: lazy, marker-gated layout hook installed on first continuation use"
    )
    return True


def uninstall_layout_patch_if_owned():
    """Best-effort rollback used only if paired patch installation fails."""
    global _ORIGINAL_INIT, _ORIGINAL_ACCEPTS_FRAME_COUNT, _APPLIED, _MM
    if _MM is None or _ORIGINAL_INIT is None:
        return False
    current = getattr(getattr(_MM, "PackedLayout", None), "__init__", None)
    if current is None or not getattr(current, LAYOUT_PATCH_MARKER, False):
        return False
    _MM.PackedLayout.__init__ = _ORIGINAL_INIT
    _ORIGINAL_INIT = None
    _ORIGINAL_ACCEPTS_FRAME_COUNT = None
    _MM = None
    _APPLIED = False
    _LOG.info("IAMCCS H3 LongerVid: rolled back layout hook")
    return True


def is_applied():
    return _APPLIED
