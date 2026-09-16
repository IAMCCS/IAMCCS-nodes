# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later

"""MiniMax H3 Fun Union bridge for the current ComfyUI model-patch API.

ComfyUI commits d3eaf6ad/02aa7078 replaced the draft CONTROL_NET contract
with MODEL_PATCH. A Fun Union patch belongs on the MODEL once; all prompts in
an MMH3 condition set then share that patched model. IAMCCS Shotboard performs
its own strict per-chunk media windowing before this official apply operation.
"""

import logging

from comfy_api.latest import io


def _first(result):
    return result.result[0] if hasattr(result, "result") else result[0]


class MMH3CondSetApplyControl(io.ComfyNode):
    """Apply the official H3 Fun Union MODEL_PATCH to the sampler model."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3CondSetApplyControl",
            display_name="MMH3 Apply H3 Fun Union Model Patch",
            category="MMH3Tools/model",
            description=(
                "Current ComfyUI bridge for MiniMax H3 Fun Union. Connect a MODEL "
                "and the output of Load Model Patch. This replaces the obsolete "
                "ControlNetLoader/CONDITIONING route used by the draft API."
            ),
            inputs=[
                io.Model.Input("model"),
                io.ModelPatch.Input("model_patch"),
                io.Vae.Input("vae"),
                io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0,
                               step=0.001, optional=True),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0,
                               step=0.001, optional=True),
                io.Image.Input("control_video", optional=True),
                io.Mask.Input("mask", optional=True,
                              tooltip="1 marks the regions to regenerate."),
                io.Image.Input("source_video", optional=True,
                               tooltip="Video behind the mask; read only with a mask."),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, model, model_patch, vae, strength, start_percent=0.0,
                end_percent=1.0, control_video=None, mask=None,
                source_video=None) -> io.NodeOutput:
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3FunControlNetApply

        result = MiniMaxH3FunControlNetApply.execute(
            model=model,
            model_patch=model_patch,
            vae=vae,
            strength=strength,
            start_percent=start_percent,
            end_percent=end_percent,
            control_video=control_video,
            mask=mask,
            source_video=source_video,
        )
        patched = _first(result)
        frames = None if control_video is None else int(control_video.shape[0])
        report = (
            "MMH3 H3 Fun Union | official MODEL_PATCH API | "
            f"control={frames if frames is not None else 'inpaint/none'} frames | "
            f"strength={float(strength):.3f} | "
            f"range={float(start_percent):.3f}-{float(end_percent):.3f}"
        )
        logging.info("[%s] %s", cls.__name__, report)
        return io.NodeOutput(patched, report)
