"""Fun ControlNet across a chunked render.

Core's `MiniMaxH3ControlNet.get_control` picks its hint frames with
`torch.arange(pixel_t)` -- from ZERO, three times over (the control video, the inpaint
mask, the source video) -- and caches the encoded result keyed on
`cond_hint.shape[2:]`. Both are right for a whole-clip pass and wrong for a chunked
one: every chunk has the same shape, so chunk 0's encode is reused for all of them,
and every chunk is driven by the control video's OPENING frames. No error, plausible
output, wrong content.

The fix does not reimplement `get_control` or override `_fit_frames` -- the latter
would catch the control video and the source but miss the mask, whose `arange` is
inline. Instead the wrapper hands core a WINDOW: `cond_hint_original`, `inpaint_video`
and `inpaint_mask` are sliced to this chunk's span before delegating, so core's
arange-from-zero is correct because zero is now the chunk's first frame.

The offset arrives as `transformer_options["mmh3_control_frame0"]`, published per
chunk by MMH3LoopingSampler. Unset means 0, which is exactly right for a single pass
through a stock sampler -- this is inert until something chunks.

BUILT AGAINST A DRAFT: comfy-kitchen PR #15860, unmerged. The attributes windowed here
are core's internals, so `_supported()` checks for each one and the node refuses with a
message rather than silently mis-windowing if kijai renames them.
"""

import logging

import torch

from comfy_api.latest import io

from .nodes_multiprompt import MMH3CondSet

OFFSET_KEY = "mmh3_control_frame0"
# the attributes we take over; if core renames one, refuse rather than guess
REQUIRED = ("cond_hint_original", "inpaint_video", "inpaint_mask", "cond_hint",
            "get_control")


def _core():
    import comfy.controlnet as cn
    return getattr(cn, "MiniMaxH3ControlNet", None)


def _supported(control_net):
    base = _core()
    if base is None:
        return False, ("this ComfyUI has no MiniMaxH3ControlNet. Fun ControlNet support "
                       "is PR #15860 and is not merged; apply it or wait.")
    if not isinstance(control_net, base):
        return False, "this is not a MiniMax H3 Fun ControlNet."
    missing = [a for a in REQUIRED if not hasattr(control_net, a)]
    if missing:
        return False, ("core's MiniMaxH3ControlNet no longer has %s, so the per-chunk "
                       "window cannot be applied. Without it every chunk would be "
                       "driven by the control video's opening frames."
                       % ", ".join(missing))
    return True, ""


def _window(frames, start, count):
    """`count` frames from `start`, clamped -- never wraps, never pads short."""
    if frames is None:
        return None
    n = int(frames.shape[0])
    if n == 0 or start <= 0:
        return frames
    lo = min(int(start), n - 1)
    hi = min(lo + int(count), n) if count else n
    return frames[lo:hi]


def make_chunk_aware(control_net, frames_per_latent_group=17, latents_per_group=5):
    """A copy of `control_net` that windows its hints to the chunk being sampled."""
    base = type(control_net)

    class ChunkAware(base):
        _mmh3_offset = -1

        def get_control(self, x_noisy, t, cond, batched_number, transformer_options):
            off = int((transformer_options or {}).get(OFFSET_KEY, 0) or 0)

            # The cache invalidates on SHAPE, and every chunk shares a shape, so
            # without this the second chunk silently reuses the first one's encode.
            if off != self._mmh3_offset:
                self.cond_hint = None
                self._mmh3_offset = off

            if off <= 0:
                return super().get_control(x_noisy, t, cond, batched_number,
                                           transformer_options)

            # how many pixel frames this chunk covers, by core's own conversion
            shapes = cond.get("latent_shapes", None)
            vs = tuple(shapes[0]) if shapes is not None else tuple(x_noisy.shape)
            latent_t = vs[2]
            count = max((latent_t - 2) // latents_per_group, 0) * frames_per_latent_group + 5

            saved = (self.cond_hint_original, self.inpaint_video, self.inpaint_mask)
            try:
                self.cond_hint_original = _window(saved[0], off, count)
                self.inpaint_video = _window(saved[1], off, count)
                self.inpaint_mask = _window(saved[2], off, count)
                return super().get_control(x_noisy, t, cond, batched_number,
                                           transformer_options)
            finally:
                (self.cond_hint_original, self.inpaint_video,
                 self.inpaint_mask) = saved

        def copy(self):
            out = super().copy()
            out.__class__ = ChunkAware
            out._mmh3_offset = -1
            return out

    out = control_net.copy()
    out.__class__ = ChunkAware
    out._mmh3_offset = -1
    return out


class MMH3CondSetApplyControl(io.ComfyNode):
    """Apply a Fun ControlNet to every prompt in a cond set."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3CondSetApplyControl",
            display_name="MMH3 Cond Set Apply ControlNet",
            category="MMH3Tools/conditioning",
            description=(
                "Applies a MiniMax H3 Fun ControlNet to EVERY prompt in a cond set, so "
                "a chunked render can use one. Core's apply node takes a single "
                "CONDITIONING and this pack's sampler takes a cond set, so the two do "
                "not meet without this.\n\n"
                "It also makes the control CHUNK-AWARE. Core picks its hint frames "
                "from index 0 and caches the encode by shape; every chunk shares a "
                "shape, so unwrapped, all of them would be driven by the control "
                "video's opening frames with no error anywhere. The wrapper hands core "
                "a window of this chunk's span instead.\n\n"
                "Built against PR #15860, which is a DRAFT. If core renames the "
                "internals this windows, the node refuses rather than mis-windowing."
            ),
            inputs=[
                MMH3CondSet.Input("cond_set"),
                io.ControlNet.Input("control_net"),
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
                MMH3CondSet.Output(display_name="cond_set"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, cond_set, control_net, vae, strength, start_percent=0.0,
                end_percent=1.0, control_video=None, mask=None,
                source_video=None) -> io.NodeOutput:
        conds = list(cond_set.get("conds", []))
        if not conds:
            raise ValueError("MMH3CondSetApplyControl: the cond set is empty.")
        if strength == 0 or (control_video is None and mask is None):
            return io.NodeOutput(cond_set, "strength 0 or no hint wired -- passed "
                                           "through untouched")

        ok, why = _supported(control_net)
        if not ok:
            raise ValueError("MMH3CondSetApplyControl: " + why)

        from comfy_extras.nodes_minimax_h3 import MiniMaxH3FunControlNetApply as Apply

        wrapped = make_chunk_aware(control_net)
        out = []
        for cond in conds:
            res = Apply.execute(cond, wrapped, vae, strength, start_percent,
                                end_percent, control_video, mask, source_video)
            out.append(res.result[0] if hasattr(res, "result") else res[0])

        frames = None if control_video is None else int(control_video.shape[0])
        lines = ["MMH3 Cond Set Apply ControlNet -- %d prompt(s)" % len(out), ""]
        lines.append("  control video : %s" % ("%d frames" % frames if frames
                                               else "none"))
        lines.append("  inpaint mask  : %s" % ("%d frames" % int(mask.shape[0])
                                               if mask is not None else "none"))
        lines.append("  strength %.2f over %.3f-%.3f" % (strength, start_percent,
                                                         end_percent))
        lines.append("")
        lines.append("  chunk-aware: each chunk is windowed to its own span via "
                     "transformer_options['%s']." % OFFSET_KEY)
        lines.append("  Wired into a stock sampler the offset is 0, which is correct "
                     "for a single whole-clip pass.")
        if frames:
            lines.append("")
            lines.append("  the control video must cover the WHOLE clip, not one "
                         "chunk -- windows are cut from it by frame index, and a short "
                         "one clamps to its last frame rather than erroring.")
        logging.info("[MMH3CondSetApplyControl] wrapped %d cond(s), chunk-aware", len(out))
        return io.NodeOutput({"conds": out,
                              "prompts": cond_set.get("prompts", []),
                              "fingerprint": cond_set.get("fingerprint")},
                             "\n".join(lines))
