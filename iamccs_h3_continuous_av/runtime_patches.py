"""Install the two H3 runtime hooks atomically on first continuation use."""

import logging
import threading

from . import patch_layout, patch_payload

_LOG = logging.getLogger("h3_continuous")
_LOCK = threading.RLock()


def _conflict_message(kind, status, err):
    if status is None:
        return f"{kind}: {err}"
    if status.state == "foreign":
        return f"{kind}: {status.owner} already owns {status.module}"
    return None


def ensure_h3_runtime_patches():
    """Ensure both hooks are active, but never wrap a foreign H3 patch."""
    with _LOCK:
        if patch_layout.is_applied() and patch_payload.is_applied():
            return True

        layout_status, layout_err = patch_layout.get_layout_patch_status()
        payload_status, payload_err = patch_payload.get_payload_patch_status()
        problems = [
            p for p in (
                _conflict_message("PackedLayout", layout_status, layout_err),
                _conflict_message("MiniMaxH3.extra_conds", payload_status, payload_err),
            ) if p
        ]
        if problems:
            detail = "; ".join(problems)
            raise RuntimeError(
                "Herrgotts H3 Infinite Continuation Suite cannot install its H3 runtime hooks: "
                f"{detail}. Disable/remove the other H3 chaining pack (or update ComfyUI if an "
                "expected H3 API is missing), restart ComfyUI, then run the continuation again."
            )

        payload_was_ours = payload_status is not None and payload_status.state == "ours"
        layout_was_ours = layout_status is not None and layout_status.state == "ours"

        if not patch_payload.install_payload_patch():
            raise RuntimeError(
                "Herrgotts H3 Infinite Continuation Suite could not install the MiniMax H3 "
                "payload hook. See the ComfyUI console for the compatibility-check error."
            )
        if not patch_layout.install_layout_patch():
            if not payload_was_ours:
                patch_payload.uninstall_payload_patch_if_owned()
            raise RuntimeError(
                "Herrgotts H3 Infinite Continuation Suite could not install the MiniMax H3 "
                "layout hook. The temporary payload hook was rolled back. See the ComfyUI "
                "console for the live self-test error."
            )

        if not (patch_layout.is_applied() and patch_payload.is_applied()):
            if not layout_was_ours:
                patch_layout.uninstall_layout_patch_if_owned()
            if not payload_was_ours:
                patch_payload.uninstall_payload_patch_if_owned()
            raise RuntimeError("h3_continuous: runtime hooks did not reach a consistent active state")

        _LOG.info(
            "h3_continuous v1.2.1: H3 runtime hooks ready; unrelated H3 graphs remain on stock behavior"
        )
        return True
