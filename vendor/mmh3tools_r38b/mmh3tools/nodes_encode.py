"""Chunked VAE encode, so long clips at high resolution can be encoded at all.

`F.pad(..., mode="reflect")` -- used by H3's CausalConv3d for its spatial padding --
requires the tensor to fit 32-bit indexing, i.e. under 2**31 = 2,147,483,648
elements. A pixel batch is [1, 3, T, H, W], so the ceiling is a joint limit on
length AND resolution:

    1024x768    906 frames  (37.7s)
    1536x1152   396 frames  (16.5s)
    2048x1536   226 frames  ( 9.4s)   <- 481 frames is 2.11x over

Past that, `VAEEncode` dies with "input tensor must fit into 32-bit index math".
That is NOT an out-of-memory error, so `model_management.raise_non_oom()` re-raises
it and ComfyUI's automatic retry-with-tiled-encoding never fires -- you get a hard
stop rather than a slow fallback. The ceiling shrinks as an upscale ladder climbs,
so a length that sails through stage 1 can fail at stage 3.

WHY CHUNKING IS EXACT HERE. `encode_temporal` slices into NON-OVERLAPPING 17-frame
clips and encodes each with no carried state:

    for i in range(num_chunks):
        clip_x = x[:, :, i*17:(i+1)*17, :, :]
        z_list.append(self._adaptive_encode(clip_x))
    z = torch.cat(z_list, dim=2)
    if self.token_drop > 0:
        z = z[:, :, :-self.token_drop]

So clip boundaries are free -- unlike LTX, whose encoder has a causal receptive
field across boundaries and needs left context re-encoded and trimmed per chunk.

THE TRAP. The tail padding and `token_drop` are applied once PER CALL, so looping
`vae.encode()` over chunks silently loses 3 latents per chunk: 39 frames encode to
12 latents whole, but 2+2+2 = 6 as three calls. Not an error, just a shorter latent
that decodes to a shorter, wrong video. This node therefore drives `_adaptive_encode`
directly and applies the pad and the drop exactly once, then reproduces `encode()`'s
moments-to-latent step.

THE SECOND TRAP, which cost a run: **a view inherits its parent's stride extent.**
Building the full [1, 3, T, H, W] tensor and then slicing 17-frame views out of it
fails exactly as VAEEncode does -- 481 frames at 2048x1536 is 4.54e9 elements, so
nothing carved from it is 32-bit addressable no matter how small the slice. Chunks
are therefore built from the IMAGE batch: each chunk's 5D tensor is materialised
separately and made contiguous, so no tensor is ever full length. `frames_per_chunk`
is also clamped to what the resolution allows, so the node cannot be configured into
the failure it exists to prevent.

Verified bit-identical to a whole-tensor encode -- max|diff| 0.00e+00 at 39 and 124
frames across chunk sizes 17, 34, 85 and 1700. (Measured before the view fix; the
arithmetic is unchanged -- same clips in the same order, one pad and one token_drop,
and normalisation is elementwise so per-chunk equals whole-batch.)

SCOPE. This bounds the tensors the VAE sees; it does not give constant RAM, because
the incoming IMAGE batch already exists in full before the node runs. Constant RAM
needs reading frames from disk per chunk, the way LTXAVTools' streaming encode does.
"""

import logging

import torch

import comfy.model_management
from comfy_api.latest import io

# reflect-pad (and most CUDA kernels) need the tensor -- and anything it is a VIEW
# of -- to be addressable in 32-bit
INDEX_LIMIT = 2 ** 31


def _h3_video_vae(vae):
    """The inner H3 video VAE, or None if this isn't one."""
    m = getattr(vae, "first_stage_model", None)
    need = ("clip_length", "token_drop", "_adaptive_encode", "pixel_mean", "pixel_std",
            "latents_mean", "latents_std")
    if m is not None and all(hasattr(m, a) for a in need):
        return m
    return None


class MMH3StreamingEncode(io.ComfyNode):
    """VAE encode in chunks, bypassing the 32-bit index ceiling."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3StreamingEncode",
            display_name="MiniMax H3 Streaming Encode",
            category="MMH3Tools/utils",
            description=(
                "Encode a long video in chunks. Drop-in for VAEEncode when the clip is "
                "too long for its resolution -- past ~226 frames at 2048x1536, VAEEncode "
                "fails with 'input tensor must fit into 32-bit index math'. Output is "
                "bit-identical to a whole-tensor encode."
            ),
            inputs=[
                io.Image.Input("images"),
                io.Vae.Input("vae", tooltip="The H3 VIDEO vae."),
                io.Int.Input(
                    "frames_per_chunk", default=85, min=17, max=1700, step=17,
                    tooltip="Frames encoded per pass, snapped to a multiple of 17 (the "
                            "VAE's clip length). Smaller means lower peak memory and more "
                            "passes. The result does not change: clips are encoded "
                            "independently, so any chunk size gives an identical latent. "
                            "Automatically CAPPED to what the resolution allows: a chunk's "
                            "tensor is 3*frames*H*W and must stay under 2^31. At 2048x1536 "
                            "that caps it at 221 frames; at 1024x768, 901.",
                ),
                io.Boolean.Input(
                    "offload_latents", default=True, optional=True,
                    tooltip="Move each chunk's latents to CPU as they are produced. Keeps "
                            "accumulated latents off the GPU on very long clips; costs a "
                            "transfer per chunk.",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.String.Output(display_name="label"),
            ],
        )

    @classmethod
    def execute(cls, images, vae, frames_per_chunk, offload_latents=True) -> io.NodeOutput:
        m = _h3_video_vae(vae)
        if m is None:
            raise ValueError(
                "MMH3StreamingEncode needs the MiniMax H3 VIDEO vae. The vae given is a "
                "%s, which has no 17-frame clip structure to chunk on. (The H3 AUDIO vae "
                "is not encodable this way either -- use VAEEncodeAudio.)"
                % type(getattr(vae, "first_stage_model", vae)).__name__)

        clip = int(m.clip_length)
        fpc = max(clip, (int(frames_per_chunk) // clip) * clip)
        n_frames = int(images.shape[0])

        # VAE.encode() loads the model itself; going around it means doing that here
        pixels = vae.process_input(images)
        try:
            # budget for ONE chunk, not the whole clip -- that is the point of this node.
            # The shape must be the 5D [B, C, T, H, W] the ENCODER receives, not the
            # IMAGE batch: core reads frames/height/width off shape[2:5], and
            # process_input is elementwise (`image * 2.0 - 1.0`) so pixels is still the
            # 4D [N, H, W, C] that came in. Passing that raised IndexError on shape[4]
            # EVERY call, so the hint was never once delivered and every encode loaded
            # with memory_required=0 -- no reservation at all, on the node whose whole
            # job is bounded-memory encoding.
            mem = int(vae.memory_used_encode(
                (1, 3, min(n_frames, fpc), int(images.shape[1]), int(images.shape[2])),
                vae.vae_dtype))
        except Exception as e:
            logging.warning("[MMH3StreamingEncode] memory_used_encode unavailable (%s); "
                            "loading with NO vram reservation", type(e).__name__)
            mem = 0  # load_models_gpu adds this to a reserve and cannot take None
        comfy.model_management.load_models_gpu(
            [vae.patcher], memory_required=mem, force_full_load=vae.disable_offload)

        # The 32-bit ceiling applies to the tensor a chunk is SLICED FROM, not just to
        # the slice: a view inherits its parent's stride extent, so carving 17-frame
        # views out of a full-length tensor fails exactly as VAEEncode does. Chunks are
        # therefore built from the IMAGE batch, and no 5D tensor is ever full length.
        h, w = int(images.shape[1]), int(images.shape[2])
        max_fpc = (INDEX_LIMIT - 1) // max(1, 3 * h * w)
        max_fpc = (max_fpc // clip) * clip
        if max_fpc < clip:
            raise ValueError(
                "%dx%d is too large to encode even one %d-frame clip: %d elements "
                "against the %d 32-bit limit. Reduce the resolution."
                % (w, h, clip, 3 * clip * h * w, INDEX_LIMIT))
        clamped = fpc > max_fpc
        fpc = min(fpc, max_fpc)

        pad = (-n_frames) % clip
        total = n_frames + pad
        n_clips = total // clip

        with torch.inference_mode():
            zs = []
            for start in range(0, total, fpc):
                end = min(start + fpc, total)
                sub = pixels[start:min(end, n_frames)]
                if end > n_frames:                       # tail pad, last chunk only
                    sub = torch.cat([sub, pixels[-1:].repeat(end - n_frames, 1, 1, 1)], dim=0)

                x = sub.to(vae.vae_dtype).to(vae.device)
                x = x.movedim(-1, 1).movedim(1, 0).unsqueeze(0).contiguous()
                x = x.add(1.0).mul_(0.5).sub_(m.pixel_mean.to(x)).div_(m.pixel_std.to(x))

                for j in range(x.shape[2] // clip):
                    z = m._adaptive_encode(x[:, :, j * clip:(j + 1) * clip])
                    zs.append(z.cpu() if offload_latents else z)
                del x, sub
                comfy.model_management.throw_exception_if_processing_interrupted()

            moments = torch.cat(zs, dim=2)
            del zs
            # token_drop is what turns 5j clips into the 5j+2 grid -- ONCE, at the end
            if m.token_drop > 0:
                moments = moments[:, :, :-int(m.token_drop)]

            mean = torch.chunk(moments.float(), 2, dim=1)[0]
            lm = m.latents_mean.view(1, -1, 1, 1, 1).to(mean)
            ls = m.latents_std.view(1, -1, 1, 1, 1).to(mean)
            out = ((mean - lm) / ls).to(vae.output_device).to(vae.vae_output_dtype())

        n_passes = (n_clips + (fpc // clip) - 1) // max(1, fpc // clip)
        label = ("%d frames -> %d latents, %d clips in %d pass%s of %d frames"
                 % (n_frames, int(out.shape[2]), n_clips, n_passes,
                    "" if n_passes == 1 else "es", fpc))
        if fpc != int(frames_per_chunk):
            label += "\n  ! frames_per_chunk %d -> %d (multiple of %d)" % (
                int(frames_per_chunk), fpc, clip)
        if pad:
            label += "\n  padded %d frame%s to fill the last clip" % (pad, "" if pad == 1 else "s")
        logging.info("[MMH3StreamingEncode] " + label.splitlines()[0])
        return io.NodeOutput({"samples": out}, label)
