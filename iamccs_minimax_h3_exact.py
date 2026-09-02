# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later

"""Opt-in MiniMax H3 exact/ClipProj components.

This module is intentionally isolated from the historical IAMCCS loaders.  A
workflow must connect this node explicitly; otherwise the existing CLIP route
is byte-for-byte unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

import folder_paths

from .iamccs_minimax_h3_atomic_backend import SUPERNODE_LINX_TYPE, _resolve_shotplan


CATEGORY = "IAMCCS/MiniMax H3/Exact Memory"
LOG = logging.getLogger("IAMCCS.MiniMaxH3.Exact")


def _exact_settings(cine_linx: Any) -> dict[str, Any]:
    plan = _resolve_shotplan(cine_linx)
    settings = plan.get("h3_exact_optimization")
    return settings if isinstance(settings, dict) else {}


def _encoder_candidates(size: str) -> list[str]:
    wanted = str(size).lower()
    candidates = []
    for name in folder_paths.get_filename_list("text_encoders"):
        lower = str(name).lower().replace("-", "_")
        if "qwen3" not in lower or "vl" not in lower or wanted not in lower:
            continue
        if not lower.endswith((".safetensors", ".gguf")):
            continue
        candidates.append(str(name))

    def score(name: str):
        lower = name.lower()
        return (
            0 if "fp8_scaled" in lower else 1,
            0 if lower.endswith(".safetensors") else 1,
            0 if "int8" in lower else 1,
            len(name),
            lower,
        )

    return sorted(candidates, key=score)


class IAMCCS_MiniMaxH3ExactClipProjLoader:
    """Lazily choose 4B/8B ClipProj or the workflow's historical CLIP.

    Only the selected encoder is materialized.  The fallback socket is lazy, so
    connecting the old 32B/GGUF loader does not cause it to load when 4B/8B is
    active.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
            },
            "optional": {
                "fallback_clip": ("CLIP", {"lazy": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("clip", "report")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    @staticmethod
    def _profile(cine_linx: Any) -> str:
        profile = str(_exact_settings(cine_linx).get("clipproj_profile", "off") or "off").lower()
        return profile if profile in {"off", "4b_v3.1", "8b_v3.1"} else "off"

    def check_lazy_status(self, cine_linx, fallback_clip=None, **kwargs):
        if self._profile(cine_linx) == "off" and fallback_clip is None:
            return ["fallback_clip"]
        return []

    def load(self, cine_linx, fallback_clip=None, unique_id=None):
        settings = _exact_settings(cine_linx)
        profile = self._profile(cine_linx)
        if profile == "off":
            if fallback_clip is None:
                raise ValueError(
                    "H3 ClipProj is Off and fallback_clip is not connected. "
                    "Connect the historical IAMCCS CLIP loader or select 4b_v3.1/8b_v3.1 in IAMCCS H3 Settings."
                )
            return fallback_clip, "H3 Exact Clip route | workflow fallback CLIP (lazy)"

        size = "4b" if profile.startswith("4b") else "8b"
        candidates = _encoder_candidates(size)
        if not candidates:
            raise FileNotFoundError(
                f"No ComfyUI-format Qwen3-VL-{size.upper()} text encoder is visible in text_encoders/clip paths."
            )
        clip_name = candidates[0]
        projection = f"mmh3-{size}-ClipProj-v3.1.safetensors"
        projection_path = folder_paths.get_full_path("clip_projections", projection)
        if not projection_path:
            raise FileNotFoundError(
                f"Missing ClipProj matrix: models/clip_projections/{projection}"
            )

        import nodes as comfy_nodes

        loader_cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("ClipProjLoader")
        if loader_cls is None:
            raise RuntimeError(
                "H3 ClipProj requires custom_nodes/ComfyUI-ClipProj 0.1.13 or newer."
            )
        clip_type = "krea2" if size == "4b" else "boogu"
        load_mode = str(settings.get("clipproj_load_mode", "dynamic") or "dynamic").lower()
        if load_mode not in {"dynamic", "streaming"}:
            load_mode = "dynamic"
        clip = loader_cls().load(
            clip_name=clip_name,
            type=clip_type,
            projection=projection,
            device="cuda:0",
            mode=load_mode,
            unique_id=unique_id,
        )[0]
        report = (
            f"H3 Exact ClipProj | profile={profile} | encoder={clip_name} | "
            f"projection={projection} | type={clip_type} | mode={load_mode} | only_selected_encoder_loaded=yes"
        )
        LOG.info(report)
        return clip, report


NODE_CLASS_MAPPINGS = {
    "IAMCCS_MiniMaxH3ExactClipProjLoader": IAMCCS_MiniMaxH3ExactClipProjLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_MiniMaxH3ExactClipProjLoader": "MiniMax H3 Exact ClipProj 4B / 8B (Lazy)",
}
