# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later

"""Append-only lazy router for the optional R42 continuity backends.

The established R42 standard/Motion Context selector remains the authority for
every existing mode.  This outer router asks ComfyUI to evaluate one of the two
new continuity branches only when its explicit Shotboard task mode is active.
"""

from __future__ import annotations

import logging
from typing import Any

from .iamccs_minimax_h3_atomic_backend import SUPERNODE_LINX_TYPE, _resolve_shotplan


LOG = logging.getLogger("IAMCCS.MiniMaxH3.ContinuousRouter")
CATEGORY = "IAMCCS/MiniMax H3/Continuity"

_CONTINUOUS_AV_MODES = {
    "longvid_masked_loop_guided",
    "masked_loop_guided",
    "long_masked_loop_guided",
}

_GUIDED_AV_LOOP_MODES = {
    "guided_av_loop_experimental",
    "longvid_guided_av_loop_experimental",
}


class IAMCCS_MiniMaxH3ContinuousBackendLazyRouterR42:
    """Select established R42, Continuous AV, or Guided AV Loop lazily."""

    @classmethod
    def INPUT_TYPES(cls):
        lazy_image = ("IMAGE", {"lazy": True})
        lazy_audio = ("AUDIO", {"lazy": True})
        lazy_latent = ("LATENT", {"lazy": True})
        lazy_int = ("INT", {"lazy": True})
        lazy_string = ("STRING", {"lazy": True})
        optional: dict[str, Any] = {}
        for prefix in ("r42", "continuous", "guided", "latentgoahead"):
            optional.update({
                f"{prefix}_frames": lazy_image,
                f"{prefix}_audio": lazy_audio,
                f"{prefix}_bridge": lazy_image,
                f"{prefix}_latent": lazy_latent,
                f"{prefix}_fps": lazy_int,
                f"{prefix}_report": lazy_string,
            })
        optional.update({
            "r42_current_segment": lazy_int,
            "r42_total_segments": lazy_int,
            "r42_trim_head_frames": lazy_int,
        })
        return {
            "required": {"cine_linx": (SUPERNODE_LINX_TYPE,)},
            "optional": optional,
        }

    RETURN_TYPES = (
        "IMAGE", "AUDIO", "IMAGE", "LATENT", "INT", "STRING", "INT", "INT", "INT",
    )
    RETURN_NAMES = (
        "frames", "audio", "bridge", "sampled_latent", "fps", "report",
        "current_segment", "total_segments", "trim_head_frames",
    )
    FUNCTION = "select"
    CATEGORY = CATEGORY

    @staticmethod
    def _branch(cine_linx):
        plan = _resolve_shotplan(cine_linx)
        task_mode = str(plan.get("task_mode", "") or "").strip().lower()
        requested = str(plan.get("requested_task_mode", "") or "").strip().lower()
        modes = {task_mode, requested}
        if "latent_go_ahead" in modes:
            return "latentgoahead"
        if modes & _GUIDED_AV_LOOP_MODES:
            return "guided"
        if modes & _CONTINUOUS_AV_MODES:
            return "continuous"
        return "r42"

    @classmethod
    def _names(cls, cine_linx):
        prefix = cls._branch(cine_linx)
        return [
            f"{prefix}_{suffix}"
            for suffix in ("frames", "audio", "bridge", "latent", "fps", "report")
        ]

    def check_lazy_status(self, cine_linx, **kwargs):
        names = self._names(cine_linx)
        if self._branch(cine_linx) == "r42":
            names += ["r42_current_segment", "r42_total_segments", "r42_trim_head_frames"]
        return [name for name in names if kwargs.get(name) is None]

    def select(self, cine_linx, **kwargs):
        names = self._names(cine_linx)
        required = list(names)
        if self._branch(cine_linx) == "r42":
            required += ["r42_current_segment", "r42_total_segments", "r42_trim_head_frames"]
        missing = [name for name in required if kwargs.get(name) is None]
        if missing:
            raise ValueError(
                "R42 continuity lazy branch is incomplete: " + ", ".join(missing)
            )
        branch = self._branch(cine_linx)
        label = {
            "latentgoahead": "LatentGoAhead · past-positioned native AV history",
            "r42": "established R42 standard / Motion Context",
            "continuous": "FL2VA Continuous AV · phase-aligned latent handover",
            "guided": "Guided AV Loop · one-master experimental",
        }[branch]
        LOG.info("IAMCCS MiniMax H3 continuity backend selected: %s", label)
        values = tuple(kwargs[name] for name in names)
        if branch == "r42":
            metadata = (
                int(kwargs["r42_current_segment"]),
                max(1, int(kwargs["r42_total_segments"])),
                max(0, int(kwargs["r42_trim_head_frames"])),
            )
        else:
            # Both new backends render their whole authored programme in one
            # outer execution. Their technical windows/intervals are internal
            # and must never arm R42's queue-next segment machinery.
            metadata = (0, 1, 0)
        return values + metadata


NODE_CLASS_MAPPINGS = {
    "IAMCCS_MiniMaxH3ContinuousBackendLazyRouterR42":
        IAMCCS_MiniMaxH3ContinuousBackendLazyRouterR42,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_MiniMaxH3ContinuousBackendLazyRouterR42":
        "MiniMax H3 · R42 / Continuous AV / Guided Loop Lazy Router",
}
