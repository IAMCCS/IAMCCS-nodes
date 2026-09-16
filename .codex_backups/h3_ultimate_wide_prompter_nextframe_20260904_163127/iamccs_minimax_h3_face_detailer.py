# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""IAMCCS routing and 1:1 public aliases for ComfyUI-H3-FaceRefine.

The actual face tracker, latent injector, per-frame denoise, SAM mask and
stitch math remains owned by the installed H3 FaceRefine custom-node package.
These aliases deliberately expose its exact sockets and widgets under the
IAMCCS/MiniMax H3 namespace, while the Router makes its stitched result an
optional post-native stage before the independent LTX/Wan delivery branch.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch

from .iamccs_prompter import SUPERNODE_LINX_TYPE


CATEGORY = "IAMCCS/MiniMax H3/Face Detailer"
_EXTERNAL_FACE_NODE_PATH = Path(__file__).resolve().parents[1] / "ComfyUI-H3-FaceRefine" / "nodes.py"


def _face_node_class(node_name: str):
    """Resolve the installed FaceRefine node without duplicating its algorithms."""
    try:
        import nodes as comfy_nodes

        node_class = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}).get(node_name)
        if node_class is not None:
            return node_class
    except Exception:
        pass

    module_name = "iamccs_external_h3_face_refine"
    module = sys.modules.get(module_name)
    if module is None:
        if not _EXTERNAL_FACE_NODE_PATH.is_file():
            raise RuntimeError(
                "ComfyUI-H3-FaceRefine is not installed. Install the FaceRefine custom node "
                "before using IAMCCS H3 Face Detailer aliases."
            )
        spec = importlib.util.spec_from_file_location(module_name, _EXTERNAL_FACE_NODE_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError("Unable to load ComfyUI-H3-FaceRefine nodes.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    node_class = getattr(module, "NODE_CLASS_MAPPINGS", {}).get(node_name)
    if node_class is None:
        raise RuntimeError(f"ComfyUI-H3-FaceRefine does not provide '{node_name}'")
    return node_class


class _FaceRefineAlias:
    """Base proxy that retains every original widget and input type verbatim."""

    EXTERNAL_NODE = ""
    FUNCTION = "run"
    CATEGORY = CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return copy.deepcopy(_face_node_class(cls.EXTERNAL_NODE).INPUT_TYPES())

    def run(self, **kwargs):
        return _face_node_class(self.EXTERNAL_NODE)().run(**kwargs)


class IAMCCS_MiniMaxH3FaceTrackCrop(_FaceRefineAlias):
    EXTERNAL_NODE = "H3FaceTrackCrop"
    RETURN_TYPES = ("IMAGE", "H3FACEXFORM", "IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("crops", "transform", "preview", "report", "canvas_w", "canvas_h")
    DESCRIPTION = "IAMCCS alias of H3 Face Track + Crop. Exact tracker/crop controls from ComfyUI-H3-FaceRefine."


class IAMCCS_MiniMaxH3FaceStitch(_FaceRefineAlias):
    EXTERNAL_NODE = "H3FaceStitch"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    DESCRIPTION = "IAMCCS alias of H3 Face Stitch. Exact mask, feather, colour-match and blend controls."


class IAMCCS_MiniMaxH3InjectVideoLatent(_FaceRefineAlias):
    EXTERNAL_NODE = "H3InjectVideoLatent"
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("av_latent", "report")
    DESCRIPTION = "IAMCCS alias of H3 Inject Video Latent for the face img2img branch."


class IAMCCS_MiniMaxH3PerFrameDenoise(_FaceRefineAlias):
    EXTERNAL_NODE = "H3PerFrameDenoise"
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("av_latent", "report")
    DESCRIPTION = "IAMCCS alias of H3 Per-Frame Denoise. Exact small/large-face strength controls."


class IAMCCS_MiniMaxH3FaceMaskSAM(_FaceRefineAlias):
    EXTERNAL_NODE = "H3FaceMaskSAM"
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("masks", "report")
    DESCRIPTION = "IAMCCS alias of H3 Face Mask (SAM), including temporal smoothing."


class IAMCCS_MiniMaxH3FaceTransformInfo(_FaceRefineAlias):
    EXTERNAL_NODE = "H3FaceTransformInfo"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    OUTPUT_NODE = True
    DESCRIPTION = "IAMCCS alias of H3 Face Transform Info for pre-render tracker verification."


def _face_settings_from_linx(cine_linx: Any) -> dict[str, Any]:
    if not isinstance(cine_linx, dict):
        return {}
    if isinstance(cine_linx.get("face_detailer_settings"), dict):
        return copy.deepcopy(cine_linx["face_detailer_settings"])
    resources = cine_linx.get("resources") if isinstance(cine_linx.get("resources"), dict) else {}
    outputs = cine_linx.get("outputs") if isinstance(cine_linx.get("outputs"), dict) else {}
    payload = resources.get("cine_payload") if isinstance(resources.get("cine_payload"), dict) else {}
    candidates = (
        resources.get("iamccs_minimax_h3_shotplan"),
        resources.get("minimax_h3_shotplan"),
        outputs.get("shotplan"),
        payload.get("minimax_h3_shotplan"),
        payload.get("shotplan"),
    )
    for plan in candidates:
        if isinstance(plan, dict) and isinstance(plan.get("face_detailer_settings"), dict):
            return copy.deepcopy(plan["face_detailer_settings"])
    return {}


class IAMCCS_MiniMaxH3FaceDetailerRouter:
    """Select native H3 frames or an optional stitched-face result before upscale.

    Wire native master frames/audio here. When the Settings node or Shotboard
    enables Face Detailer, connect `face_detail_frames` to the output of IAMCCS
    H3 Face Stitch. The router's single IMAGE/AUDIO output then goes to the
    existing optional Upscale Router. Thus the four valid routes are native,
    native+face, native+upscale and native+face+upscale without changing the
    native generator or the upscaler implementation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "native_frames": ("IMAGE",),
                "native_audio": ("AUDIO",),
            },
            "optional": {
                # These inputs must remain lazy.  A wired FaceRefine graph can
                # contain detection/SAM and a second H3 pass; evaluating it
                # merely because it is present in the workflow defeats the
                # optional-stage contract and consumes VRAM while disabled.
                "face_detail_frames": ("IMAGE", {"lazy": True}),
                "face_detail_audio": ("AUDIO", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("frames", "audio", "report")
    FUNCTION = "route"
    CATEGORY = CATEGORY

    def check_lazy_status(
        self,
        cine_linx,
        native_frames,
        native_audio,
        face_detail_frames=None,
        face_detail_audio=None,
        **kwargs,
    ):
        """Request the expensive face branch only when Settings enables it."""
        settings = _face_settings_from_linx(cine_linx)
        if bool(settings.get("enabled", False)) and face_detail_frames is None:
            return ["face_detail_frames"]
        return []

    def route(self, cine_linx, native_frames, native_audio, face_detail_frames=None, face_detail_audio=None):
        settings = _face_settings_from_linx(cine_linx)
        enabled = bool(settings.get("enabled", False))
        profile = str(settings.get("profile", "balanced") or "balanced")
        sam = bool(settings.get("use_sam_mask", False))
        if not enabled:
            return native_frames, native_audio, "Face Detailer router: native H3 (optional face branch disabled)"
        if not torch.is_tensor(face_detail_frames):
            raise ValueError(
                "Face Detailer is enabled in MiniMax H3 Settings, but face_detail_frames is not connected. "
                "Connect IAMCCS H3 Face Stitch here, or disable Face Detailer."
            )
        audio = face_detail_audio if isinstance(face_detail_audio, dict) else native_audio
        return (
            face_detail_frames,
            audio,
            f"Face Detailer router: stitched H3 faces | profile={profile} | sam_mask={'on' if sam else 'off'} | next=optional upscale",
        )


NODE_CLASS_MAPPINGS = {
    "IAMCCS_MiniMaxH3FaceTrackCrop": IAMCCS_MiniMaxH3FaceTrackCrop,
    "IAMCCS_MiniMaxH3FaceStitch": IAMCCS_MiniMaxH3FaceStitch,
    "IAMCCS_MiniMaxH3InjectVideoLatent": IAMCCS_MiniMaxH3InjectVideoLatent,
    "IAMCCS_MiniMaxH3PerFrameDenoise": IAMCCS_MiniMaxH3PerFrameDenoise,
    "IAMCCS_MiniMaxH3FaceMaskSAM": IAMCCS_MiniMaxH3FaceMaskSAM,
    "IAMCCS_MiniMaxH3FaceTransformInfo": IAMCCS_MiniMaxH3FaceTransformInfo,
    "IAMCCS_MiniMaxH3FaceDetailerRouter": IAMCCS_MiniMaxH3FaceDetailerRouter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_MiniMaxH3FaceTrackCrop": "IAMCCS H3 Face Track + Crop",
    "IAMCCS_MiniMaxH3FaceStitch": "IAMCCS H3 Face Stitch",
    "IAMCCS_MiniMaxH3InjectVideoLatent": "IAMCCS H3 Inject Video Latent",
    "IAMCCS_MiniMaxH3PerFrameDenoise": "IAMCCS H3 Per-Frame Denoise",
    "IAMCCS_MiniMaxH3FaceMaskSAM": "IAMCCS H3 Face Mask (SAM)",
    "IAMCCS_MiniMaxH3FaceTransformInfo": "IAMCCS H3 Face Transform Info",
    "IAMCCS_MiniMaxH3FaceDetailerRouter": "IAMCCS H3 Optional Face Detailer Router",
}
