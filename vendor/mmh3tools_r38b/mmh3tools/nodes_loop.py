"""Assembling chunks: measuring the join, pairing modalities, splicing time.

NOTE ON noise_mask: masks DO reach the model. `comfy/samplers.py` packs the AV pair
into a flat tensor before sampling (L1280) and explicitly unbinds a nested mask
(L1296), so the sampler never sees a NestedTensor and the inpaint arithmetic is fine.
An earlier version of this file claimed otherwise; that was wrong.

What stock ComfyUI lacks is per-row TIMESTEP handling: preserved rows still run at
the generation timestep, so the model gets clean content labelled as noisy and the
mask accomplishes nothing. drozbay's per-row masking fixes it, open upstream as
**#15375**. MMH3SeedOverlap requires that PR and REFUSES to run without it, rather
than appearing to work.

Applying an upstream PR is not monkeypatching, which is why this lives here rather
than on `keyframe-anchors`: the PR is somebody else's pending change that will merge,
not a wrap this pack maintains forever.

Joins are still trimmed AFTER decode - latent trims sit on the 5j+2 grid, i.e.
17-frame steps, and latent concatenation is unsound regardless (see MMH3JoinAV).
"""

import logging

import torch

import comfy.utils
from comfy.nested_tensor import NestedTensor
from comfy_api.latest import io

from .common import (
    AUDIO_T_DIM,
    FPS,
    LATENTS_PER_GROUP,
    LATENT_BASE,
    VIDEO_T_DIM,
    frame_at_latent,
    frames_to_audio_t,
    latents_to_frames,
    on_grid,
    pack_av,
    slice_av_tail,
    unpack_av,
)


def _mask_side(latent, which):
    """Pull the video (0) or audio (1) half out of a latent's noise_mask, or None.

    Masks arrive nested to match an AV latent, but a hand-built graph can hand over
    a plain one, so fall back to inferring the modality from rank: video masks are
    5D [B,1,T,h,w], audio masks 4D [B,1,2,T40].
    """
    m = latent.get("noise_mask")
    if m is None:
        return None
    if isinstance(m, NestedTensor):
        parts = m.unbind()
        return parts[0] if which == 0 else parts[-1]
    if which == 0:
        return m if getattr(m, "ndim", 0) == 5 else None
    return m if getattr(m, "ndim", 0) == 4 else None


def _per_row_masking_available():
    """Whether #15375's per-row masking is present in this ComfyUI.

    Detected rather than assumed, because without it a noise mask has NO effect at all:
    preserved rows still run at the generation timestep, so the model gets clean content
    labelled as noisy. A node that quietly returns a longer clip with a regenerated head
    reads as a model failure rather than a missing PR.
    """
    try:
        import comfy.ldm.minimax.model as mm
    except Exception:
        return False
    # `mask_row_targets` was RENAMED to `mask_row_values` when #15375 was rebased onto
    # the merged #15439 -- and the rename is the point: it returns per-row FLOATS now
    # rather than bools, so a partial mask is genuinely partial. Accept either name;
    # checking only the old one silently disabled every masking node.
    has_mask_fn = hasattr(mm, "mask_row_values") or hasattr(mm, "mask_row_targets")
    return has_mask_fn and hasattr(mm, "_mod_row")


def per_row_mask_is_continuous():
    """Whether #15375 blends the TIMESTEP continuously, or thresholds at 0.5.

    The original PR reduced a mask to one bool per 2x2 patch row, so partial
    `overlap_strength` blended the latent continuously while the timestep
    conditioning stayed all-or-nothing. The rebased version returns floats. Anything
    documenting "binarises at 0.5" is describing the old behaviour.
    """
    try:
        import comfy.ldm.minimax.model as mm
    except Exception:
        return False
    return hasattr(mm, "mask_row_values")


def _ones_mask_for(t):
    """A 1-channel all-denoise mask matching t's batch, temporal and spatial extent.

    One channel rather than ones_like(t): samplers.py unbinds the nested mask and
    runs prepare_mask per sub-latent, which broadcasts channels, so carrying 24 or
    32 identical copies buys nothing.
    """
    return torch.ones([t.shape[0], 1] + list(t.shape[2:]), dtype=torch.float32,
                      device=t.device)


class MMH3FindDivergence(io.ComfyNode):
    """Find where a continuation stops reproducing its source, in FRAMES.

    H3 tends to re-render the carried reference at the head of a continuation
    before generating new content. That span is not frame-aligned with the source
    (the model regenerates rather than copies) and will not land on the 5j+2 latent
    grid, whose cut points are 17 frames apart -- so the trim has to happen after
    decode, where granularity is one frame.

    Method: assume the reproduced span ENDS at the source's final frame, so a run of
    length K means continuation[i] ~ source[-K+i]. For each candidate K the mean error
    over that exact alignment is scored, and the best K wins.

    Per-frame nearest-match does NOT work here: in visually repetitive footage (a
    talking head on a static background) every new frame also finds a close match
    somewhere in the source, so divergence is never detected. Requiring the whole run
    to align contiguously fixes that -- a wrong K misaligns every frame at once, which
    produces an order-of-magnitude error separation.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3FindDivergence",
            display_name="MiniMax H3 Find Divergence",
            category="MMH3Tools/utils",
            description=(
                "Compare the head of a continuation against the tail of its source and "
                "report how many frames are reproduced, so you can trim the join."
            ),
            inputs=[
                io.Image.Input("source", tooltip="Decoded frames of the previous chunk"),
                io.Image.Input("continuation", tooltip="Decoded frames of the new chunk"),
                io.Int.Input("search_frames", default=96, min=1, max=2048, step=1,
                             tooltip="How far into the continuation to look."),
                io.Int.Input("source_tail_frames", default=96, min=1, max=2048, step=1,
                             tooltip="How much of the source tail to match against."),
                io.Float.Input("threshold", default=0.05, min=0.0, max=1.0, step=0.001,
                               tooltip="Reject the alignment if its mean absolute error exceeds "
                                       "this, and report 0 reproduced. Check the reported best/"
                                       "median separation to calibrate."),
                io.Int.Input("downsample", default=48, min=8, max=256, step=8,
                             tooltip="Frames are greyscaled and resized to this before "
                                     "comparison. Smaller is faster and more tolerant of noise."),
                io.Combo.Input("compare", options=["structure", "raw"], default="structure",
                               tooltip="'structure' removes each frame's mean and contrast before "
                                       "comparing, so an exposure or colour shift between source "
                                       "and generation cannot mask a real match. 'raw' is plain "
                                       "MAE. Error magnitudes differ between the two, so "
                                       "recalibrate threshold when switching."),
            ],
            outputs=[
                io.Int.Output(display_name="trim_frames"),
                io.Float.Output(display_name="mean_error"),
                io.String.Output(display_name="report"),
            ],
        )

    @staticmethod
    def _prep(img, size, structure):
        x = img[..., :3].mean(dim=-1, keepdim=True).movedim(-1, 1).float()
        x = torch.nn.functional.interpolate(x, size=(size, size), mode="area")
        if structure:
            # zero-mean, unit-contrast per frame: an exposure or colour shift between
            # the source and the generated chunk otherwise puts a floor under every
            # comparison and flattens the curve, hiding a genuine match.
            m = x.mean(dim=(1, 2, 3), keepdim=True)
            s = x.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
            x = (x - m) / s
        return x

    @classmethod
    def execute(cls, source, continuation, search_frames, source_tail_frames,
                threshold, downsample, compare="structure") -> io.NodeOutput:
        n_src, n_con = source.shape[0], continuation.shape[0]
        tail = min(int(source_tail_frames), n_src)
        search = min(int(search_frames), n_con)
        structure = (compare == "structure")

        a = cls._prep(source[-tail:], int(downsample), structure)
        b = cls._prep(continuation[:search], int(downsample), structure)

        # pairwise mean absolute error, [search, tail]
        d = (b.unsqueeze(1) - a.unsqueeze(0)).abs().mean(dim=(2, 3, 4))

        # score each candidate run length K as the diagonal ending at the source's
        # last frame: continuation[i] vs source[-K+i]
        limit = min(search, tail)
        errs = []
        for k in range(1, limit + 1):
            i = torch.arange(k, device=d.device)
            errs.append(float(d[i, tail - k + i].mean()))
        err_t = torch.tensor(errs)

        best_k = int(err_t.argmin().item()) + 1
        best_err = float(err_t.min())
        median_err = float(err_t.median())
        # a real reproduction shows a sharp minimum, not a flat curve
        separation = median_err / best_err if best_err > 1e-8 else float("inf")

        trim = best_k if best_err <= float(threshold) else 0

        lo, hi = max(1, best_k - 2), min(limit, best_k + 2)
        around = ", ".join("%d:%.4f" % (k, errs[k - 1]) for k in range(lo, hi + 1))
        lines = [
            "reproduced : %d frames (%.3fs @24fps)%s"
            % (best_k, best_k / 24.0, "" if trim else "   REJECTED (error > threshold)"),
            "best error : %.5f   median %.5f   separation %.1fx (threshold %.4f)"
            % (best_err, median_err, separation, threshold),
            "curve near best: %s" % around,
        ]
        if separation < 3.0:
            lines.append("WARNING weak minimum -- the curve is flat, so this alignment is "
                         "not trustworthy. Check the clips actually overlap.")
        if best_k == limit:
            lines.append("NOTE best K is at the search limit; raise search_frames / "
                         "source_tail_frames.")
        mean_err = best_err

        report = "\n".join(lines)
        print("[MMH3FindDivergence]\n" + report)
        return io.NodeOutput(int(trim), mean_err, report)


class MMH3JoinAV(io.ComfyNode):
    """Join two decoded chunks in PIXEL and WAVEFORM space.

    Latent concatenation is unsound for H3's video VAE. Two on-grid chunks sum to
    5(j+k)+4 latents, which is never on the 5j+2 grid, so the decoder's 17-frame
    causal chunking misaligns from the join onward and the second half pulses. Even
    with an on-grid trim, chunk B's latent 0 is a causal anchor spanning one frame
    and ends up mid-group where the decoder expects four.

    Decoding each chunk separately avoids all of that, and gives frame granularity
    instead of the latent grid's 17-frame steps.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3JoinAV",
            display_name="MiniMax H3 Join AV",
            category="MMH3Tools/latent",
            description=(
                "Trim and crossfade two decoded chunks. Video joins per frame, audio "
                "crossfades in the waveform domain (the DAC/BigVGAN latents do not blend)."
            ),
            inputs=[
                io.Image.Input("images_a"),
                io.Image.Input("images_b"),
                io.Int.Input("trim_b_frames", default=0, min=0, max=4096, step=1,
                             tooltip="Frames to drop from the head of B, e.g. a reproduced span "
                                     "measured by MMH3FindDivergence."),
                io.Int.Input("crossfade_frames", default=0, min=0, max=240, step=1,
                             tooltip="Linear crossfade across the seam, taken from A's tail and "
                                     "B's head. 0 is a hard cut."),
                io.Audio.Input("audio_a", optional=True),
                io.Audio.Input("audio_b", optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Audio.Output(display_name="audio"),
                io.String.Output(display_name="label"),
            ],
        )

    @classmethod
    def execute(cls, images_a, images_b, trim_b_frames, crossfade_frames,
                audio_a=None, audio_b=None) -> io.NodeOutput:
        b = images_b[int(trim_b_frames):] if trim_b_frames > 0 else images_b
        if b.shape[0] == 0:
            raise ValueError("trim_b_frames removed the whole of images_b")
        if images_a.shape[1:] != b.shape[1:]:
            raise ValueError("Frame size mismatch: A is %s, B is %s"
                             % (tuple(images_a.shape[1:3]), tuple(b.shape[1:3])))

        n = max(0, min(int(crossfade_frames), images_a.shape[0], b.shape[0]))
        if n > 0:
            w = torch.linspace(0, 1, n + 2, device=images_a.device)[1:-1].view(-1, 1, 1, 1)
            blend = images_a[-n:] * (1 - w) + b[:n].to(images_a.dtype) * w
            video = torch.cat([images_a[:-n], blend, b[n:].to(images_a.dtype)], dim=0)
        else:
            video = torch.cat([images_a, b.to(images_a.dtype)], dim=0)

        audio = None
        if audio_a is not None and audio_b is not None:
            sr = int(audio_a["sample_rate"])
            if int(audio_b["sample_rate"]) != sr:
                raise ValueError("Sample rate mismatch: %d vs %d" % (sr, audio_b["sample_rate"]))
            wa, wb = audio_a["waveform"], audio_b["waveform"].to(audio_a["waveform"].dtype)
            cut = int(round(int(trim_b_frames) / FPS * sr))
            wb = wb[:, :, cut:] if cut > 0 else wb
            m = max(0, min(int(round(n / FPS * sr)), wa.shape[-1], wb.shape[-1]))
            if m > 0:
                w = torch.linspace(0, 1, m + 2, device=wa.device)[1:-1].view(1, 1, -1)
                mid = wa[:, :, -m:] * (1 - w) + wb[:, :, :m] * w
                wav = torch.cat([wa[:, :, :-m], mid, wb[:, :, m:]], dim=-1)
            else:
                wav = torch.cat([wa, wb], dim=-1)
            audio = {"waveform": wav, "sample_rate": sr}
        elif audio_a is not None or audio_b is not None:
            logging.warning("[MMH3JoinAV] only one audio input connected; audio not joined")
            audio = audio_a if audio_a is not None else audio_b

        label = "%d + %d frames (trimmed %d, crossfade %d) -> %d frames, %.3fs" % (
            images_a.shape[0], images_b.shape[0], trim_b_frames, n,
            video.shape[0], video.shape[0] / FPS)
        print("[MMH3JoinAV] " + label)
        return io.NodeOutput(video, audio, label)


class MMH3PackAV(io.ComfyNode):
    """Zip a video latent and an audio latent into one H3 AV latent.

    Encoding real footage gives two SEPARATE plain latents -- VAEEncode with the
    H3 video VAE, and VAEEncodeAudio with the H3 audio VAE -- and nothing pairs
    them. This is that pairing. It is not a concatenation: ConcatAV joins two AV
    latents along TIME, this joins video and audio along MODALITY.

    Audio length is reconciled to what the video length implies
    (round(frames / 24 * 40)), padding with silence or trimming as needed, since
    the two streams run on independent clocks and encoders will not agree exactly.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3PackAV",
            display_name="MiniMax H3 Pack AV",
            category="MMH3Tools/latent",
            description=(
                "Combine a video latent (VAEEncode, H3 video VAE) and an audio latent "
                "(VAEEncodeAudio, H3 audio VAE) into a single H3 AV latent. Omit the "
                "audio to pair with silence."
            ),
            inputs=[
                io.Latent.Input("video_latent", tooltip="Plain video latent [B,24,T,h,w]"),
                io.Latent.Input(
                    "audio_latent", optional=True,
                    tooltip="Plain audio latent [B,32,2,T40]. If omitted, silence of the "
                            "correct length is generated.",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.String.Output(display_name="label"),
            ],
        )

    @classmethod
    def execute(cls, video_latent, audio_latent=None) -> io.NodeOutput:
        v, _ = unpack_av(video_latent, "video_latent", allow_video_only=True)
        vt = int(v.shape[VIDEO_T_DIM])
        frames = latents_to_frames(vt)
        want_at = frames_to_audio_t(frames)

        note = ""
        if audio_latent is None:
            a = torch.zeros([v.shape[0], 32, 2, want_at], dtype=v.dtype, device=v.device)
            note = "silent audio generated"
        else:
            a = audio_latent["samples"]
            if isinstance(a, NestedTensor):
                a = a.unbind()[1]
            if a.ndim != 4 or a.shape[1] != 32:
                raise ValueError(
                    "'audio_latent' is not an H3 audio latent; expected [B,32,2,T40], got %s. "
                    "Encode with VAEEncodeAudio using the H3 audio VAE." % (tuple(a.shape),)
                )
            have = int(a.shape[AUDIO_T_DIM])
            if have > want_at:
                a = a[:, :, :, :want_at]
                note = "audio trimmed %d -> %d" % (have, want_at)
            elif have < want_at:
                pad = torch.zeros([a.shape[0], a.shape[1], a.shape[2], want_at - have],
                                  dtype=a.dtype, device=a.device)
                a = torch.cat([a, pad], dim=AUDIO_T_DIM)
                note = "audio padded %d -> %d" % (have, want_at)
            a = a.to(v.dtype)

        if not on_grid(vt):
            note = (note + "; " if note else "") + "WARNING video T=%d is off the 5j+2 grid" % vt

        out = dict(video_latent)
        out["samples"] = NestedTensor([v.contiguous(), a.contiguous()])

        # Carry any input mask into a nested pair rather than dropping it. Filling the
        # missing side with ones means "denoise everything there", so pairing a masked
        # video latent with unmasked audio does what you would expect.
        vm = video_latent.get("noise_mask")
        am = audio_latent.get("noise_mask") if audio_latent is not None else None
        if isinstance(vm, NestedTensor):
            vm = vm.unbind()[0]
        if isinstance(am, NestedTensor):
            am = am.unbind()[-1]
        if vm is not None or am is not None:
            if vm is None:
                vm = torch.ones([v.shape[0], 1, v.shape[2], v.shape[3], v.shape[4]],
                                dtype=torch.float32, device=v.device)
            if am is None:
                am = torch.ones([a.shape[0], 1, a.shape[2], a.shape[3]],
                                dtype=torch.float32, device=a.device)

            # Normalize each half onto ITS latent's shape, with the SAME
            # interpolation core's prepare_mask applies at sampling time. Core
            # accepts a mask of ANY size (a 32x32 pin image is legal) and
            # interpolates it; the looping sampler SLICES masks by time, which
            # assumes the time axis is real. Whole-clip and chunked runs then
            # see identical mask semantics. Identity for masks already shaped
            # to their latent; [:, :1] keeps the pack's one-channel convention
            # (prepare_mask re-expands channels at sampling either way).
            in_shapes = (tuple(vm.shape), tuple(am.shape))
            vm = comfy.utils.reshape_mask(vm.to(torch.float32), v.shape)[:, :1]
            am = comfy.utils.reshape_mask(am.to(torch.float32), a.shape)[:, :1]
            changed = in_shapes != (tuple(vm.shape), tuple(am.shape))
            out["noise_mask"] = NestedTensor([vm.contiguous(), am.contiguous()])
            logging.info("[MMH3PackAV] carried an input noise_mask into the AV pair%s",
                         " (normalized %s / %s -> %s / %s)" % (
                             in_shapes[0], in_shapes[1],
                             tuple(vm.shape), tuple(am.shape)) if changed else "")
        else:
            out.pop("noise_mask", None)

        label = "%d video latents (%d frames, %.3fs) + %d audio latents%s" % (
            vt, frames, frames / 24.0, int(a.shape[AUDIO_T_DIM]), ("  [" + note + "]") if note else "")
        print("[MMH3PackAV] " + label)
        return io.NodeOutput(out, label)


class MMH3SeedOverlap(io.ComfyNode):
    """LTXAV-style mask-and-extend: seed the target head, mask it, denoise the rest.

    REQUIRES per-row masking, open upstream as #15375. Stock ComfyUI accepts a nested
    mask and packs it correctly, but preserved rows still run at the GENERATION
    timestep, so the model receives clean content labelled as noisy and the mask
    achieves nothing at all. The PR pins masked rows to the COND timestep -- the same
    treatment reference rows get. This node refuses to run without it rather than
    appearing to work.

        overlap_strength 1.0 -> mask 0.0 -> fully preserved (pinned)
        overlap_strength 0.0 -> mask 1.0 -> fully regenerated

    PARTIAL STRENGTH IS GENUINELY PARTIAL -- as of the rebased #15375 (2026-08-13).

    It was not always. The original PR reduced the mask to one BOOL per 2x2 patch row
    (`mask_row_targets`, `>= 0.5`), so a strength of 0.3 and one of 0.4 both landed as
    "preserved" for TIMESTEP purposes and only the sampler's own latent blend varied.
    That docstring warned to re-check if the PR changed before merge. It changed:

        old:  target = m.reshape(-1) >= 0.5   # bool, all-or-nothing
        new:  values = m.reshape(-1)          # float in [0, 1]   (mask_row_values)

    So the AdaLN lerp now receives a continuous weight, and partial strength grades
    the TIMESTEP conditioning as well as the latent. A feathered spatial mask no
    longer hardens at the 0.5 contour. `per_row_mask_is_continuous()` reports which
    behaviour the installed core has; both are still max-pooled per 2x2 patch, which
    is a property of the patch grid rather than of the threshold.

    Video and audio are masked independently on their own temporal axes (video dim
    2, audio dim 3) and reach the model as separate denoise_mask / audio_denoise_mask
    conditions, so lipsync can carry audio harder than video.

    Video and audio are masked independently on their own temporal axes (video dim
    2, audio dim 3) and reach the model as separate denoise_mask / audio_denoise_mask
    conditions, so lipsync can carry audio harder than video.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3SeedOverlap",
            display_name="MiniMax H3 Seed Overlap",
            category="MMH3Tools/latent",
            description=(
                "Seed the head of a target AV latent with the tail of a previous chunk and "
                "emit a matching nested noise_mask. Requires the per-row masking patch."
            ),
            inputs=[
                io.Latent.Input("latent", tooltip="Target AV latent (from Empty MiniMax H3 AV Latent)"),
                io.Latent.Input("source", tooltip="Previous chunk's AV latent"),
                io.Int.Input(
                    "overlap_latents", default=5, min=5, max=512, step=5,
                    tooltip="Video latents PREPENDED as overlap. Must be a multiple of 5: the "
                            "target is 5a+2 and the total must stay 5c+2, so only multiples of "
                            "5 keep the result decodable. 5 latents = 17 frames = 0.708s.",
                ),
                io.Float.Input(
                    "overlap_strength_video", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="1.0 preserves the overlap (noise_mask 0), 0.0 regenerates it. "
                            "Intermediate values are real partial pins, not thresholds.",
                ),
                io.Float.Input(
                    "overlap_strength_audio", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="Same scale as video, applied to the AUDIO latents and set "
                            "independently of it.",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="overlap_frames"),
                io.Int.Output(
                    display_name="overlap_latents",
                    tooltip="Wire into ConcatAV's trim_b_latents so the overlap is not "
                            "duplicated at the join.",
                ),
            ],
        )

    @classmethod
    def execute(cls, latent, source, overlap_latents, overlap_strength_video,
                overlap_strength_audio) -> io.NodeOutput:
        # Refuse rather than run. Without per-row masking the mask has NO effect at
        # all -- preserved rows still run at the generation timestep -- so this would
        # return a longer clip with a regenerated head and read as a model failure
        # rather than a missing PR.
        if not _per_row_masking_available():
            raise RuntimeError(
                "MMH3SeedOverlap needs per-row masking (upstream PR #15375), which is "
                "not applied. See docs/core-changes.md. Stock ComfyUI has no "
                "per-row TIMESTEP handling, so preserved rows run at the generation "
                "timestep and the mask accomplishes nothing -- the node would appear to "
                "work and quietly do nothing. Apply the diff, or wait for #15375.")

        tgt_v, tgt_a = unpack_av(latent, "latent")
        src_v, src_a = unpack_av(source, "source", allow_video_only=True)

        if src_v.shape[3:] != tgt_v.shape[3:]:
            raise ValueError(
                "Spatial mismatch: source latent is %dx%d, target is %dx%d. "
                "Overlap seeding requires identical dimensions."
                % (src_v.shape[4] * 16, src_v.shape[3] * 16,
                   tgt_v.shape[4] * 16, tgt_v.shape[3] * 16)
            )

        # PREPEND the overlap so the target keeps its full requested duration. The total
        # must stay on the 5j+2 grid, and (5a+2)+(5b+2) never is -- so the overlap has to
        # be a multiple of 5, which adds exactly 17 frames each.
        k = max(5, (int(overlap_latents) // 5) * 5)
        k = min(k, int(src_v.shape[VIDEO_T_DIM]))
        k = (k // 5) * 5
        if k < 5:
            raise ValueError("source has fewer than 5 video latents; nothing to overlap")

        n_tgt = int(tgt_v.shape[VIDEO_T_DIM])
        total = n_tgt + k
        tgt_frames = latents_to_frames(n_tgt)
        total_frames = latents_to_frames(total)
        overlap_frames = total_frames - tgt_frames          # == 17 * (k // 5)
        overlap_audio = frames_to_audio_t(total_frames) - frames_to_audio_t(tgt_frames)

        v = torch.cat([src_v[:, :, -k:, :, :].to(tgt_v.dtype), tgt_v], dim=VIDEO_T_DIM)

        if src_a is not None and overlap_audio > 0:
            take = min(overlap_audio, int(src_a.shape[AUDIO_T_DIM]))
            head = src_a[:, :, :, -take:].to(tgt_a.dtype)
            if take < overlap_audio:                        # source shorter than needed
                pad = torch.zeros([head.shape[0], head.shape[1], head.shape[2],
                                   overlap_audio - take], dtype=tgt_a.dtype, device=tgt_a.device)
                head = torch.cat([pad, head], dim=AUDIO_T_DIM)
        else:
            if src_a is None:
                logging.info("[MMH3SeedOverlap] source has no audio; overlap audio is silent")
            head = torch.zeros([tgt_a.shape[0], tgt_a.shape[1], tgt_a.shape[2], overlap_audio],
                               dtype=tgt_a.dtype, device=tgt_a.device)
        a = torch.cat([head, tgt_a], dim=AUDIO_T_DIM)

        # noise_mask: 1.0 = denoise, 0.0 = preserve.
        #
        # The TARGET may already carry a mask -- a supplied audio track pinned by
        # MMH3ReferenceMultiPrompt's use_input_audio, say. Building these from ones
        # threw it away, so the track was regenerated everywhere except the overlap.
        # The prepended carry has no incoming mask of its own (it is new rows), so
        # it is simply set; everything after it starts from what the target asked
        # for, defaulting to full denoise.
        in_vm, in_am = _mask_side(latent, 0), _mask_side(latent, 1)

        vm = torch.ones([v.shape[0], 1, v.shape[2], v.shape[3], v.shape[4]],
                        dtype=torch.float32, device=v.device)
        if in_vm is not None and in_vm.shape[VIDEO_T_DIM] == tgt_v.shape[VIDEO_T_DIM]:
            vm[:, :, k:] = in_vm.to(dtype=vm.dtype, device=vm.device)
        vm[:, :, :k] = 1.0 - float(overlap_strength_video)


        am = torch.ones([a.shape[0], 1, a.shape[2], a.shape[3]],
                        dtype=torch.float32, device=a.device)
        if in_am is not None and in_am.shape[AUDIO_T_DIM] == tgt_a.shape[AUDIO_T_DIM]:
            am[:, :, :, overlap_audio:] = in_am.to(dtype=am.dtype, device=am.device)
        if overlap_audio > 0:
            am[:, :, :, :overlap_audio] = 1.0 - float(overlap_strength_audio)

        out = pack_av(latent, v, a, noise_mask=NestedTensor([vm, am]))
        # States the fact, not a method. Joining in PIXEL space cuts these frames
        # after decode; MMH3LoopingSampler joins in latent space and cuts k+2
        # LATENTS at the join instead, which is a few frames more. Saying "trim N
        # frames after decode" read as an instruction in both cases.
        logging.info("[MMH3SeedOverlap] %d + %d = %d latents (%d frames); the first "
                     "%d frames reproduce the source and come off at the join",
                     k, n_tgt, total, total_frames, overlap_frames)
        return io.NodeOutput(out, int(overlap_frames), int(k))


class MMH3ConcatAV(io.ComfyNode):
    """Concatenate two AV latents on their correct, DIFFERENT temporal axes.

    Video is dim 2, audio is dim 3. Generic nested-tensor concat helpers that
    assume one shared temporal dim will stack audio on its stereo axis instead,
    producing 4 channels at unchanged duration rather than a longer clip.

    Masks live on those same axes, so joining them is the same cat with the same
    dims -- an earlier comment here claimed a per-frame mask "cannot span the
    join", which was never true. The reason masks are still dropped by DEFAULT is
    semantic, not structural: an inherited mask described a generation that has
    already happened, so it is spent. See `carry_masks`.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3ConcatAV",
            display_name="MiniMax H3 Concat AV",
            category="MMH3Tools/latent",
            description="Join two H3 AV latents end to end (video dim 2, audio dim 3).",
            inputs=[
                io.Latent.Input("latent_a"),
                io.Latent.Input("latent_b"),
                io.Int.Input(
                    "trim_b_latents", default=0, min=0, max=512, step=1,
                    tooltip="Drop this many video latents from the head of B before joining -- "
                            "use it to discard a seeded overlap region that A already contains. "
                            "The value is honoured as given (only clamped so B keeps 2 latents). "
                            "5m removes a SeedOverlap exactly but leaves the total OFF the 5j+2 "
                            "grid; 5m+2 lands on grid but leaves ~7 frames of overlap duplicated. "
                            "You cannot have both -- if you need both, decode the chunks and use "
                            "MMH3JoinAV, which cuts per frame.",
                ),
                io.Boolean.Input(
                    "carry_masks", default=False, optional=True,
                    tooltip="Concatenate the inputs' noise_masks alongside the latents, filling "
                            "an absent side with ones (denoise everything there). OFF by default "
                            "because an inherited mask is usually SPENT: it described a "
                            "generation that has already happened, so re-sampling the join would "
                            "pin two finished seams and regenerate everything between them. Turn "
                            "it on when the join is deliberately the INPUT to a bridging pass.",
                ),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, latent_a, latent_b, trim_b_latents, carry_masks=False) -> io.NodeOutput:
        va, aa = unpack_av(latent_a, "latent_a")
        vb, ab = unpack_av(latent_b, "latent_b")

        if va.shape[3:] != vb.shape[3:]:
            raise ValueError("Spatial mismatch: cannot concatenate latents of different sizes.")

        vma, vmb = _mask_side(latent_a, 0), _mask_side(latent_b, 0)
        ama, amb = _mask_side(latent_a, 1), _mask_side(latent_b, 1)

        k = 0
        drop_audio = 0
        if trim_b_latents > 0:
            # There is NO snap that is right for every use, so this one does not snap.
            # Two incompatible goals want different k, given A = 5a+2 and B = 5b+2:
            #
            #   k = 5m    removes a SeedOverlap exactly and leaves B's remainder on
            #             grid (5(b-m)+2), but the TOTAL is 5(a+b)+4-k, off grid.
            #   k = 5m+2  puts the total back on grid, but leaves 2 latents (~7 frames)
            #             of the overlap duplicated at the join.
            #
            # k cannot be 0 and 2 mod 5 at once. Before 0.9.0 this silently snapped to
            # the second family, so wiring SeedOverlap's overlap_latents=5 in trimmed 2
            # and the overlap was mostly still there. Now the value is honoured and the
            # trade-off is logged, because which one you want depends on whether you
            # decode this latent (grid) or feed it onward (dedup).
            n_b = int(vb.shape[VIDEO_T_DIM])
            k = max(0, min(int(trim_b_latents), n_b - LATENT_BASE))
            if k != int(trim_b_latents):
                logging.info("[MMH3ConcatAV] trim_b_latents %d clamped to %d (B must keep at "
                             "least %d latents)", int(trim_b_latents), k, LATENT_BASE)

        if k > 0:
            # audio_t is round(frames / 24 * 40), which is NOT additive, so every
            # route here works in DIFFERENCES of totals rather than converting a
            # dropped frame count directly. Which totals depends on k.
            n_a_v = int(va.shape[VIDEO_T_DIM])
            n_b_v = int(vb.shape[VIDEO_T_DIM])
            total_v = n_a_v + n_b_v - k
            if on_grid(total_v) and aa is not None and ab is not None:
                # k = 5m+2. B's REMAINDER is off grid, so latents_to_frames(n_b - k)
                # floors to the group below and the else-branch drop comes out ~20
                # latents too large -- half a second of audio lost per seam, and it
                # COMPOUNDS: four chained chunks measured 1.48s short. The master is
                # on grid though, so ask what IT needs and drop exactly the excess.
                # This is the family a chained loop must use, since only an on-grid
                # total can be decoded at all.
                want = frames_to_audio_t(latents_to_frames(total_v))
                drop_audio = (int(aa.shape[AUDIO_T_DIM])
                              + int(ab.shape[AUDIO_T_DIM])) - want
            else:
                # k = 5m. B's remainder IS on grid and the total is not, so there is
                # no master length to ask about; B's own difference is exact. Right
                # for feeding the result onward rather than decoding it.
                frames_b = latents_to_frames(n_b_v)
                frames_keep = latents_to_frames(n_b_v - k)
                drop_audio = frames_to_audio_t(frames_b) - frames_to_audio_t(frames_keep)
            drop_audio = max(0, min(drop_audio, int(ab.shape[AUDIO_T_DIM]) - 1))
            vb = vb[:, :, k:, :, :]
            ab = ab[:, :, :, drop_audio:]
            # B's masks describe the UNTRIMMED B, so they take the same computed cuts --
            # reuse k and drop_audio, never the raw widget value, or the mask silently
            # stops lining up with the content it describes.
            if vmb is not None:
                vmb = vmb[:, :, k:, :, :]
            if amb is not None:
                amb = amb[:, :, :, drop_audio:]
            rem = k % LATENTS_PER_GROUP
            note = ("removes a %d-latent overlap exactly; the total will be OFF grid"
                    % k if rem == 0 else
                    "keeps the total ON grid" if rem == LATENT_BASE else
                    "is neither a multiple of 5 nor 5m+2, so it neither removes a whole "
                    "overlap nor lands on grid")
            # frame_at_latent is the GENERAL form, valid off grid too -- the dropped
            # span is B's first k steps, and latents_to_frames would floor here.
            logging.info("[MMH3ConcatAV] trimmed %d latents (%d frames, %d audio) off B's head "
                         "-- %s", k, frame_at_latent(k), drop_audio, note)

        v = torch.cat([va, vb.to(va.dtype)], dim=VIDEO_T_DIM)
        a = torch.cat([aa, ab.to(aa.dtype)], dim=AUDIO_T_DIM)

        total = int(v.shape[VIDEO_T_DIM])
        if not on_grid(total):
            logging.warning(
                "[MMH3ConcatAV] result is %d latents, OFF the 5j+2 grid. Two on-grid chunks "
                "sum to 5(j+k)+4, which is never on-grid, so the causal VAE will misalign "
                "from the join onward and the second half will pulse. Trim 5m+2 from B "
                "(minimum 2), or better: decode the chunks separately and use MMH3JoinAV.",
                total)

        out = dict(latent_a)
        out["samples"] = NestedTensor([v, a])

        have_masks = any(m is not None for m in (vma, vmb, ama, amb))
        if carry_masks and have_masks:
            if vma is None:
                vma = _ones_mask_for(va)
            if vmb is None:
                vmb = _ones_mask_for(vb)
            if ama is None:
                ama = _ones_mask_for(aa)
            if amb is None:
                amb = _ones_mask_for(ab)
            vm = torch.cat([vma, vmb.to(vma.dtype)], dim=VIDEO_T_DIM)
            am = torch.cat([ama, amb.to(ama.dtype)], dim=AUDIO_T_DIM)
            if vm.shape[VIDEO_T_DIM] != v.shape[VIDEO_T_DIM]:
                logging.warning(
                    "[MMH3ConcatAV] carried video mask is %d latents but the latent is %d. "
                    "prepare_mask will silently RESIZE it, so the preserved region will land "
                    "somewhere other than intended.",
                    int(vm.shape[VIDEO_T_DIM]), int(v.shape[VIDEO_T_DIM]))
            if am.shape[AUDIO_T_DIM] != a.shape[AUDIO_T_DIM]:
                logging.warning(
                    "[MMH3ConcatAV] carried audio mask is %d frames but the latent is %d.",
                    int(am.shape[AUDIO_T_DIM]), int(a.shape[AUDIO_T_DIM]))
            out["noise_mask"] = NestedTensor([vm, am])
            logging.info("[MMH3ConcatAV] carried noise_masks across the join "
                         "(video %d, audio %d)",
                         int(vm.shape[VIDEO_T_DIM]), int(am.shape[AUDIO_T_DIM]))
        else:
            if have_masks:
                logging.info("[MMH3ConcatAV] dropped an input noise_mask; enable carry_masks "
                             "to concatenate it instead")
            out.pop("noise_mask", None)
        return io.NodeOutput(out)
