# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""Small IAMCCS preset adapter for the installed OBVPM H3 MCtx nodes."""
from __future__ import annotations

import json


def _pin_requests(operation: str, previous_mctx, next_mctx) -> list[tuple[object, str, str]]:
    mode = str(operation or "extend").strip().lower()
    if mode == "extend":
        if previous_mctx is None:
            raise ValueError("MCtx Extend needs the previous take connected to PREVIOUS MCTX")
        return [(previous_mctx, "tail", "before")]
    if mode == "prepend":
        if next_mctx is None:
            raise ValueError("MCtx Prepend needs the following take connected to NEXT MCTX")
        return [(next_mctx, "head", "after")]
    if mode == "bridge":
        if previous_mctx is None or next_mctx is None:
            raise ValueError("MCtx Bridge needs both PREVIOUS MCTX and NEXT MCTX")
        return [(previous_mctx, "tail", "before"), (next_mctx, "head", "after")]
    raise ValueError(f"Unknown MCtx operation: {operation}")


def _obvpm_pin_class():
    from nodes import NODE_CLASS_MAPPINGS

    node = NODE_CLASS_MAPPINGS.get("H3MCtxPinSpec")
    if node is None:
        raise RuntimeError(
            "OBVPM MCtx nodes are not available. Install/enable comfyui-obvpm-timeline and restart ComfyUI."
        )
    return node


class IAMCCS_H3MCtxRoute:
    CATEGORY = "IAMCCS/H3/LongVid"
    FUNCTION = "route"
    RETURN_TYPES = ("OBVPM_H3_PINSPECS", "STRING")
    RETURN_NAMES = ("pin_specs", "route_report")
    DESCRIPTION = (
        "User-facing Extend / Prepend / Bridge preset for the installed OBVPM MCtx engine. "
        "The adapter builds pin specifications only; OBVPM remains the owner of latent slicing, masks and trimming."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "operation": (["extend", "prepend", "bridge"], {"default": "extend"}),
                "window": (["39", "90", "141", "22", "56", "5"], {"default": "39"}),
                "pin_mode": (["masked", "both", "guide"], {"default": "masked"}),
                "mask_ramp_frames": ("INT", {"default": 0, "min": 0, "max": 256}),
                "mask_ramp_edge": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "mask_hold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "previous_mctx": ("OBVPM_H3_MCTX",),
                "next_mctx": ("OBVPM_H3_MCTX",),
            },
        }

    def route(
        self,
        operation,
        window,
        pin_mode,
        mask_ramp_frames,
        mask_ramp_edge,
        mask_hold,
        previous_mctx=None,
        next_mctx=None,
    ):
        requests = _pin_requests(operation, previous_mctx, next_mctx)
        pin_class = _obvpm_pin_class()
        stack = None
        resolved = []
        for mctx, take_from, place in requests:
            stack = pin_class().build(
                mctx=mctx,
                window=str(window),
                take_from=take_from,
                take_from_frame=0,
                place=place,
                place_at_frame=0,
                audio_window=0,
                mode=str(pin_mode),
                mask_ramp_frames=int(mask_ramp_frames),
                mask_ramp_edge=float(mask_ramp_edge),
                mask_hold=float(mask_hold),
                pin_specs=stack,
            )[0]
            resolved.append({"take_from": take_from, "place": place})
        report = {
            "engine": "comfyui-obvpm-timeline",
            "operation": str(operation),
            "window": int(window),
            "pin_mode": str(pin_mode),
            "pins": resolved,
            "next": "Connect pin_specs to H3 MCtx Apply Pins, then trim/save through the OBVPM pins output.",
        }
        return stack, json.dumps(report, ensure_ascii=False)


NODE_CLASS_MAPPINGS = {"IAMCCS_H3MCtxRoute": IAMCCS_H3MCtxRoute}
NODE_DISPLAY_NAME_MAPPINGS = {"IAMCCS_H3MCtxRoute": "IAMCCS H3 LongVid · MCtx Route"}
