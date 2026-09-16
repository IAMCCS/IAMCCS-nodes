"""Lazy, marker-gated MiniMax H3 keyframe/reference payload coexistence patch."""

import inspect
import logging

from .patch_layout import HC_INDEX, HC_AUDIO_END_FRAME
from .patch_utils import classify_callable

PAYLOAD_PATCH_MARKER = "_herrgotts_h3_infinite_payload_patch"
_LOG = logging.getLogger("h3_continuous")
_ORIGINAL_EXTRA_CONDS = None
_APPLIED = False
_MODEL_BASE = None

_KNOWN_EXTERNAL_MARKERS = (
    ("_h3_motion_context_payload_patch", "ComfyUI-H3-Motion-Context"),
)


def _import_model_base():
    import comfy.model_base as model_base
    return model_base


def get_payload_patch_status():
    try:
        model_base = _import_model_base()
    except Exception as exc:
        return None, f"cannot import comfy.model_base: {exc}"
    cls = getattr(model_base, "MiniMaxH3", None)
    fn = getattr(cls, "extra_conds", None) if cls is not None else None
    if cls is None or fn is None:
        return None, "ComfyUI MiniMaxH3.extra_conds is unavailable"
    status = classify_callable(cls, fn, PAYLOAD_PATCH_MARKER, _KNOWN_EXTERNAL_MARKERS)
    return status, None


def _graph_has_our_markers(keyframes, refs):
    return (
        bool(keyframes) and any(HC_INDEX in kf for kf in keyframes)
    ) or (
        bool(refs) and any(HC_AUDIO_END_FRAME in r for r in refs)
    )


def _rewrite_marked_payload(out, keyframes, refs, frame_count=None):
    holder = out.get("minimax_payload") if isinstance(out, dict) else None
    payload = getattr(holder, "cond", None) if holder is not None else None
    if not isinstance(payload, dict):
        raise RuntimeError(
            "h3_continuous: could not access the MiniMax H3 conditioning payload; "
            "refusing an unsafe continuation"
        )
    payload["cond_video_latents"] = (
        [k["latent"] for k in keyframes if "latent" in k]
        + [r["latent"] for r in refs if "latent" in r]
    )
    payload["cond_audio_latents"] = [
        r["audio_latent"] for r in refs if r.get("audio_latent") is not None
    ]
    if frame_count is not None:
        payload["frame_count"] = frame_count
    return out


def _validate_live_extra_conds(fn):
    if not callable(fn):
        raise RuntimeError("MiniMaxH3.extra_conds is not callable")
    sig = inspect.signature(fn)
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        raise RuntimeError("MiniMaxH3.extra_conds no longer accepts **kwargs")


def install_payload_patch():
    global _ORIGINAL_EXTRA_CONDS, _APPLIED, _MODEL_BASE
    if _APPLIED:
        return True

    status, err = get_payload_patch_status()
    if status is None:
        _LOG.error("h3_continuous: payload patch unavailable: %s", err)
        return False
    if status.state == "ours":
        _APPLIED = True
        _LOG.info("h3_continuous: compatible Herrgotts H3 payload patch is already active")
        return True
    if status.state == "foreign":
        _LOG.error(
            "h3_continuous: H3 runtime-patch conflict: %s already owns "
            "MiniMaxH3.extra_conds (%s). Disable one H3 chaining pack and restart ComfyUI.",
            status.owner, status.module,
        )
        return False

    model_base = _import_model_base()
    cls = model_base.MiniMaxH3
    _ORIGINAL_EXTRA_CONDS = cls.extra_conds
    try:
        _validate_live_extra_conds(_ORIGINAL_EXTRA_CONDS)
    except Exception as exc:
        _LOG.error(
            "h3_continuous: live ComfyUI payload compatibility check FAILED (%s). "
            "No payload patch was installed.", exc,
        )
        _ORIGINAL_EXTRA_CONDS = None
        return False

    def patched_extra_conds(self, **kwargs):
        out = _ORIGINAL_EXTRA_CONDS(self, **kwargs)
        keyframes = kwargs.get("minimax_keyframes")
        refs = kwargs.get("minimax_refs")
        if not keyframes or not refs:
            return out
        if not _graph_has_our_markers(keyframes, refs):
            return out  # unrelated H3 graph: exact stock result
        return _rewrite_marked_payload(
            out, keyframes, refs, frame_count=kwargs.get("minimax_frame_count")
        )

    setattr(patched_extra_conds, PAYLOAD_PATCH_MARKER, True)
    cls.extra_conds = patched_extra_conds
    _MODEL_BASE = model_base
    _APPLIED = True
    _LOG.info(
        "h3_continuous v1.2.1: lazy, marker-gated H3 payload patch installed on first continuation use"
    )
    return True


def uninstall_payload_patch_if_owned():
    """Best-effort rollback used only if paired patch installation fails."""
    global _ORIGINAL_EXTRA_CONDS, _APPLIED, _MODEL_BASE
    if _MODEL_BASE is None or _ORIGINAL_EXTRA_CONDS is None:
        return False
    cls = getattr(_MODEL_BASE, "MiniMaxH3", None)
    current = getattr(cls, "extra_conds", None) if cls is not None else None
    if current is None or not getattr(current, PAYLOAD_PATCH_MARKER, False):
        return False
    cls.extra_conds = _ORIGINAL_EXTRA_CONDS
    _ORIGINAL_EXTRA_CONDS = None
    _MODEL_BASE = None
    _APPLIED = False
    _LOG.info("h3_continuous: rolled back Herrgotts H3 payload patch")
    return True


def is_applied():
    return _APPLIED
