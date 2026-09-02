"""Chunked latent -> pixels -> upscale -> latent, for the refine leg of a 2K pass.

WHY THIS IS A LATENT->LATENT NODE AND NOT AN IMAGE SLICER
---------------------------------------------------------
The obvious shape -- slice an IMAGE batch, upscale each slice, return one IMAGE --
saves nothing. The wall is the RETURNED tensor: at 2x from 2K the output is
108 MB/frame against the input's 27, and a node that hands back an IMAGE must
materialise all of it however it was filled. Chunking only buys anything when the
result LEAVES the graph or goes into a consumer that is also chunked.

So the chunking runs the whole way across: decode a slice of the stage-1 latent,
upscale those frames, encode them at 2K, keep the LATENT and drop the pixels.
Latents are ~1/100 the size of the frames they came from, so the accumulated output
is small at any length and the pixel footprint is one chunk. What crosses the node
boundary is a 2K latent ready to refine.

WHY LATENT UPSCALING IS NOT AN OPTION HERE
------------------------------------------
A 24-channel latent at /16 is not a spatially smooth signal. Interpolating between
two latent positions produces codes the decoder was never trained on, and it renders
them as blocking -- the "chunky latent upscale" problem. `downscale_video_latent` in
common.py is bilinear, but it is only ever applied to REFERENCE slices, which are
never denoised; approximate context is fine, approximate content is not. This node
therefore goes through pixels, where a real upscaler can be used.

THE GRIDS LINE UP EXACTLY, which is what makes this cheap:

    decode emits 17 frames per latent GROUP of 5     (the 17j+5 <-> 5j+2 grid)
    encode consumes NON-OVERLAPPING 17-frame CLIPS   (encode_temporal)

so every decode batch hands the encoder a whole number of clips. Round trip:

    T = 5j+2  ->  17j+5 frames  ->  pad to 17(j+1)  ->  5(j+1) latents
              ->  token_drop 3  ->  5j+2            (the length is preserved)

TWO TRAPS INHERITED FROM THE NODES THIS IS BUILT ON
---------------------------------------------------
1. Decode chunks need LEFT CONTEXT and LOOKAHEAD, and the trailing 5 frames of a
   partial decode are written raw where a full decode would blend them. Same scheme
   as MMH3StreamingSave -- see its module docstring for why.
2. `token_drop` and the tail pad are applied ONCE PER CALL by the VAE, so looping
   `vae.encode()` silently loses 3 latents per chunk. This drives `_adaptive_encode`
   directly and drops once, at the end. Same as MMH3StreamingEncode.

Audio is carried through untouched: this is a resolution pass and audio has no
resolution.
"""

import logging

import torch
import torch.nn.functional as F

import comfy.model_management
from comfy_api.latest import io

from .common import latents_to_frames, unpack_av
from .nodes_encode import INDEX_LIMIT, _h3_video_vae
from .nodes_save import TAIL_FRAMES, vae_grid

# torch modes that make sense going UP. "area" is a downsampler and "nearest" is
# nearest-exact's off-by-half predecessor; neither belongs on this list.
_TORCH_MODES = ["lanczos-ish bicubic", "bilinear", "nearest-exact"]
_MODE_MAP = {"lanczos-ish bicubic": "bicubic", "bilinear": "bilinear",
             "nearest-exact": "nearest-exact"}
_RTX = "rtx_vsr"


def upscale_frames(px, width, height, method, rtx_quality="ULTRA", device=None,
                   sub_batch=17):
    """[F,H,W,C] in 0..1 -> [F,height,width,C], back on px's device and dtype.

    Both paths work a SUB-BATCH at a time. A whole chunk resized in one call has
    the full upscaled tensor live alongside the source -- 3.4 GB for 68 frames at
    2688x1536 -- which defeats the point of chunking at the level above. 17 frames
    is one encode clip, so the bound is one clip's worth either way.

    RTX VSR is imported lazily and per call: the pack must not hard-depend on
    nvvfx, which ships with a separate node pack and is not present on every
    install. A missing binding is reported as a wiring problem, not a stack trace.
    """
    f, h, w, c = px.shape
    width, height = int(width), int(height)
    if (h, w) == (height, width):
        return px

    out = torch.empty((f, height, width, c), device=px.device, dtype=px.dtype)
    dev = device if device is not None else px.device

    if method == _RTX:
        try:
            import nvvfx
        except Exception as e:
            raise RuntimeError(
                "MMH3ChunkedPixelUpscale: method 'rtx_vsr' needs the nvvfx binding "
                "from the NVIDIA RTX node pack (comfyui_nvidia_rtx_nodes), which is "
                "not importable here (%s). Install it, or pick a torch method."
                % type(e).__name__)
        level = getattr(nvvfx.effects.QualityLevel, str(rtx_quality).upper(),
                        nvvfx.effects.QualityLevel.ULTRA)
        # One effect instance for the whole call: load() is the expensive part, and
        # VSR is stateless across frames (verified), so reusing it is free AND the
        # result does not depend on how the batch was split.
        with nvvfx.VideoSuperRes(level) as sr:
            sr.output_width, sr.output_height = width, height
            sr.load()
            for i in range(f):
                frame = px[i].movedim(-1, 0).float().cuda().contiguous()
                up = torch.from_dlpack(sr.run(frame).image)
                out[i] = up.movedim(0, -1).to(dtype=px.dtype, device=px.device)
                del frame, up
    else:
        mode = _MODE_MAP[method]
        kw = {} if mode == "nearest-exact" else {"align_corners": False}
        for i in range(0, f, sub_batch):
            j = min(i + sub_batch, f)
            x = px[i:j].movedim(-1, 1).to(device=dev, dtype=torch.float32)
            x = F.interpolate(x, size=(height, width), mode=mode, **kw)
            out[i:j] = x.movedim(1, -1).to(dtype=px.dtype, device=px.device)
            del x

    return out.clamp_(0.0, 1.0)


class MMH3ChunkedPixelUpscale(io.ComfyNode):
    """Stage-1 latent -> 2K latent, through pixels, a chunk at a time."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3ChunkedPixelUpscale",
            display_name="MMH3 Chunked Pixel Upscale",
            category="MMH3Tools/latent",
            description=(
                "Decode, upscale and re-encode a stage-1 latent one chunk at a time, "
                "so a 2K refine pass costs one chunk of pixels instead of the whole "
                "clip. Goes through pixels because latent upscaling decodes as "
                "blocking. Audio rides through untouched. Wire width/height from "
                "MMH3 Regenerate-2K Dimensions."
            ),
            inputs=[
                io.Latent.Input(
                    "latent",
                    tooltip="The stage-1 AV latent to upscale. A plain video-only "
                            "latent is accepted; audio simply stays absent."),
                io.Vae.Input("vae", tooltip="The H3 VIDEO vae, used for both legs."),
                io.Int.Input(
                    "width", default=2688, min=32, max=16384, step=32,
                    tooltip="Target width. Wire MMH3 Regenerate-2K Dimensions' "
                            "width_2k. Must be a multiple of 32 so the latent dim "
                            "stays even for the 2x2 patch."),
                io.Int.Input(
                    "height", default=1536, min=32, max=16384, step=32,
                    tooltip="Target height. Wire its height_2k."),
                io.Combo.Input(
                    "method", options=[_RTX] + _TORCH_MODES, default=_RTX,
                    tooltip="rtx_vsr is NVIDIA Video Super Resolution, a learned "
                            "upscaler, and needs the RTX node pack installed. The "
                            "others are torch resamplers and need nothing."),
                io.Int.Input(
                    "groups_per_chunk", default=4, min=1, max=64, step=1,
                    tooltip="Latent GROUPS per pass. One group is 5 latents = 17 "
                            "frames = one encode clip. Each pass also decodes one "
                            "extra group of context that is discarded. Capped to what "
                            "the target resolution allows under the 32-bit limit."),
                io.Combo.Input(
                    "rtx_quality", options=["LOW", "MEDIUM", "HIGH", "ULTRA"],
                    default="ULTRA", optional=True,
                    tooltip="VSR quality level. Ignored by the torch methods."),
                io.Boolean.Input(
                    "offload_latents", default=True, optional=True,
                    tooltip="Move each chunk's latents to CPU as they are produced."),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, latent, vae, width, height, method, groups_per_chunk,
                rtx_quality="ULTRA", offload_latents=True) -> io.NodeOutput:
        m = _h3_video_vae(vae)
        if m is None:
            raise ValueError(
                "MMH3ChunkedPixelUpscale needs the MiniMax H3 VIDEO vae; the vae given "
                "is a %s, which has neither the 17-frame clip structure to encode on "
                "nor the group structure to decode on."
                % type(getattr(vae, "first_stage_model", vae)).__name__)

        width, height = int(width), int(height)
        if width % 32 or height % 32:
            raise ValueError(
                "MMH3ChunkedPixelUpscale: %dx%d is not on the 32px canvas grid. Latent "
                "dims are px/16 and must stay EVEN for the 2x2 patch, so both axes must "
                "be multiples of 32. MMH3 Regenerate-2K Dimensions emits valid pairs."
                % (width, height))

        video, audio = unpack_av(latent, "latent", allow_video_only=True)
        # One clip at a time, as everywhere else in the pack. BOTH halves have to be
        # reduced together -- taking video[:1] and leaving audio at batch N builds a
        # NestedTensor whose two tensors disagree on batch, which is not an error at
        # construction and goes wrong much later.
        if int(video.shape[0]) > 1:
            logging.warning("[MMH3ChunkedPixelUpscale] batch of %d; upscaling the first "
                            "only", int(video.shape[0]))
        video = video[:1]
        if audio is not None:
            audio = audio[:1]
        T = int(video.shape[2])

        group, _lookahead, fpg = vae_grid(vae)
        clip = int(m.clip_length)
        if fpg != clip:
            raise ValueError(
                "MMH3ChunkedPixelUpscale: this VAE decodes %d frames per group but "
                "encodes %d-frame clips. The two grids must agree or a decode batch "
                "cannot be handed to the encoder whole." % (fpg, clip))

        n_groups = max(0, (T - 2) // group)
        if n_groups < 1:
            raise ValueError(
                "MMH3ChunkedPixelUpscale: %d latents is under one group (%d + 2). "
                "There is nothing to chunk -- decode, upscale and encode normally."
                % (T, group))
        tail = T - (n_groups * group + 2)
        if tail:
            logging.warning("[MMH3ChunkedPixelUpscale] %d latents is off the %dj+2 grid "
                            "by %d; the remainder is not carried", T, group, tail)

        src_h, src_w = int(video.shape[3]) * 16, int(video.shape[4]) * 16
        expected = fpg * n_groups + TAIL_FRAMES

        # A chunk's pixel tensor is 3*F*H*W and must be 32-bit addressable -- the same
        # ceiling MMH3StreamingEncode exists for, and it BITES HARDER here because the
        # frames are at the upscaled size. Cap rather than let it fail mid-pass.
        max_frames = (INDEX_LIMIT - 1) // max(1, 3 * height * width)
        max_groups = max(1, max_frames // fpg)
        gpc = max(1, int(groups_per_chunk))
        clamped = gpc > max_groups
        gpc = min(gpc, max_groups)
        if max_frames < fpg:
            raise ValueError(
                "MMH3ChunkedPixelUpscale: %dx%d cannot hold even one %d-frame group: "
                "%d elements against the %d 32-bit limit. Lower the target resolution."
                % (width, height, fpg, 3 * fpg * height * width, INDEX_LIMIT))

        logging.info("[MMH3ChunkedPixelUpscale] %dx%d -> %dx%d (%.2fx), %d frames, "
                     "%d groups, %d per pass, method %s",
                     src_w, src_h, width, height, width / float(src_w),
                     expected, n_groups, gpc, method)

        zs, written, g0 = [], 0, 0
        with torch.inference_mode():
            while g0 < n_groups:
                comfy.model_management.throw_exception_if_processing_interrupted()
                g1 = min(g0 + gpc, n_groups)
                last = g1 >= n_groups

                # --- decode, with one group of left context and 2 latents of lookahead
                lo = max(0, group * g0 - group)
                hi = min(T, group * g1 + 2)
                px = vae.decode(video[:, :, lo:hi])
                if isinstance(px, tuple):
                    px = px[0]
                if px.ndim == 5:
                    px = px.reshape(-1, *px.shape[-3:])
                head = fpg if g0 > 0 else 0
                keep = fpg * (g1 - g0)
                # the trailing 5 are written raw by a partial decode but BLENDED in a
                # full one -- keep them only where there is no next chunk to blend into
                px = px[head:head + keep + (TAIL_FRAMES if last else 0)]

                # --- upscale
                n_frames = int(px.shape[0])
                px = upscale_frames(px, width, height, method, rtx_quality,
                                    device=vae.device)
                written += n_frames

                # --- pad the final chunk out to a whole clip, then encode
                if last:
                    pad = (-int(px.shape[0])) % clip
                    if pad:
                        px = torch.cat([px, px[-1:].repeat(pad, 1, 1, 1)], dim=0)

                x = vae.process_input(px).to(vae.vae_dtype).to(vae.device)
                del px
                x = x.movedim(-1, 1).movedim(1, 0).unsqueeze(0).contiguous()
                x = x.add(1.0).mul_(0.5).sub_(m.pixel_mean.to(x)).div_(m.pixel_std.to(x))
                for j in range(x.shape[2] // clip):
                    z = m._adaptive_encode(x[:, :, j * clip:(j + 1) * clip])
                    zs.append(z.cpu() if offload_latents else z)
                del x

                logging.info("[MMH3ChunkedPixelUpscale] groups [%d,%d) of %d -> "
                             "%d frames upscaled (total %d)",
                             g0, g1, n_groups, n_frames, written)
                g0 = g1

            moments = torch.cat(zs, dim=2)
            del zs
            # ONCE, over the whole sequence -- this is what turns 5j clips into 5j+2
            if m.token_drop > 0:
                moments = moments[:, :, :-int(m.token_drop)]
            mean = torch.chunk(moments.float(), 2, dim=1)[0]
            del moments
            lm = m.latents_mean.view(1, -1, 1, 1, 1).to(mean)
            ls = m.latents_std.view(1, -1, 1, 1, 1).to(mean)
            out_v = ((mean - lm) / ls).to(vae.output_device).to(vae.vae_output_dtype())

        if written != expected:
            logging.warning("[MMH3ChunkedPixelUpscale] upscaled %d frames, expected %d. "
                            "The VAE's chunking no longer matches the %dj+2 grid this "
                            "slices on.", written, expected, group)
        out_t = int(out_v.shape[2])
        if out_t != T - tail:
            logging.warning("[MMH3ChunkedPixelUpscale] round trip gave %d latents from "
                            "%d. The length should be preserved.", out_t, T - tail)

        # A stale noise_mask would be the wrong shape now, so the dict is rebuilt
        # rather than copied. Audio is untouched: this pass has no opinion about it.
        if audio is not None:
            from .common import pack_av
            # dtype as well as device: the video half comes back at the VAE's output
            # dtype, and a NestedTensor whose halves disagree is a problem deferred,
            # not avoided. Same reconciliation MMH3Regenerate2KReference does.
            out = pack_av({}, out_v, audio.to(dtype=out_v.dtype, device=out_v.device))
        else:
            out = {"samples": out_v}

        report = ("%dx%d -> %dx%d (%.3fx), %s\n"
                  "%d frames, %d groups in %d pass%s of %d\n"
                  "%d latents in, %d out (%.2fs)"
                  % (src_w, src_h, width, height, width / float(src_w), method,
                     written, n_groups, (n_groups + gpc - 1) // gpc,
                     "" if (n_groups + gpc - 1) // gpc == 1 else "es", gpc,
                     T, out_t, latents_to_frames(out_t) / 24.0))
        if clamped:
            report += ("\n  ! groups_per_chunk %d -> %d, the most %dx%d can hold under "
                       "the 32-bit limit" % (int(groups_per_chunk), gpc, width, height))
        if audio is None:
            report += "\n  video-only latent: no audio to carry"
        logging.info("[MMH3ChunkedPixelUpscale] %s", report.splitlines()[0])
        return io.NodeOutput(out, report)
