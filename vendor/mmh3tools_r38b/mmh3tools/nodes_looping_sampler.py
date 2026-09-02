"""Chained AV generation in one node: fill a clip chunk by chunk, one graph cost.

WHY ONE NODE. Driving N chunks from the graph costs a copy of every downstream
node per chunk -- sampler, decode, save -- and a graph that size is what starts
breaking ComfyUI. A Python loop inside one node costs the same whether it runs
4 chunks or 40. The trade is that nothing inside the loop can be a graph node,
so every prompt has to exist BEFORE sampling starts. That is what the cond_set
is: MMH3ReferenceMultiPrompt encodes all N up front, in one text-encoder load.

THE LATENT IS THE WHOLE CLIP. You hold a song of known length; you do not know
how many chunks that is. So the total is given and the chunk count is DERIVED,
the way LTXAVTools' looping sampler does it -- an earlier version of this node
took ONE CHUNK's latent and a chunk count, which made the total emergent and
every question about it ("how long will this be", "which chunk owns frame N",
"which slice of audio does chunk 3 get") an exercise in reconstruction.

The schedule comes from `_plan`, the SAME function MMH3WindowPlan and
MMH3SplitAudioToWindows use. So chunks and windows are the same thing by
construction: the prompt written against window 3's audio is the prompt chunk 3
renders, and `window_count` really is the chunk count rather than merely looking
like it.

AUDIO COMES FREE. The master latent holds the whole track, so each chunk slices
its own span out of it. No per-chunk audio input, no risk of every chunk getting
the first chunk's music.

TWO CARRY ROUTES.

`mask` masks the chunk's first `overlap` latents -- they already hold the
previous chunk's output, since chunks are slices of one master -- so the model
conditions on them without denoising them. Needs per-row masking (#15375).

`keyframe` passes those same latents as a GUIDE anchored at frame 0: re-injected
every step, never denoised, carrying a multi-step clip plus its audio at the same
cond_t. Needs #15439, and the guide-origin wrap when references are also present.

Neither needs a join. Chunks are written back into the master in place, so there
is no trim, no grid-safe cut, and no frames lost per seam -- all of which existed
only because the old shape concatenated separately-allocated chunks.

GUIDES ARE BUILT HERE, per chunk, and stale ones are stripped off incoming
conditioning first -- an upstream guide node or a cond cached from a previous
run would anchor this chunk to somebody else's frames. Straight from
LTXAVTools, where the same leak had the same cause.

GUIDER COPYING is the part that is easy to get wrong. copy.copy is shallow, so
new_g.original_conds is the SAME dict object as the source guider's, and
set_conds assigns into it -- chunk 0 would overwrite the BASE conditioning and
every later chunk would read chunk 0's conds back as "base". Rebinding the dict
per chunk is the fix. Learned in LTXAVTools, where the symptom was every chunk
getting chunk 0's speaker.
"""

import copy
import logging

import node_helpers
import torch
import comfy.utils
from comfy_api.latest import io
from comfy.nested_tensor import NestedTensor

from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced, SplitSigmas

from .common import (AUDIO_LATENT_FPS, AUDIO_T_DIM, FPS, LATENTS_PER_GROUP,
                     LATENT_BASE, VAE_SPATIAL, VIDEO_T_DIM,
                     frame_at_latent, frames_to_audio_t, latents_to_frames,
                     pack_av, unpack_av)
from .nodes_loop import per_row_mask_is_continuous
from .nodes_multiprompt import MMH3CondSet
from .nodes_windows import _audio_index_at, _plan, _window_frame_spans

_GUIDE_KEYS = ("minimax_keyframes", "minimax_frame_count")
MASK_MODES = ["max", "min", "mean", "last"]


def _strip_guide_keys(cond, label):
    """Remove keyframe bookkeeping from conditioning.

    This node registers ALL of its own guides, per chunk. Anything arriving
    pre-registered is stale -- an upstream guide node, or a cond cached from a
    previous run -- and would anchor this chunk to the wrong frames. Straight
    from LTXAVTools, where the same leak had the same cause.
    """
    out, stripped = [], False
    for t, d in cond:
        if any(k in d for k in _GUIDE_KEYS):
            d = {k: v for k, v in d.items() if k not in _GUIDE_KEYS}
            stripped = True
        out.append([t, d])
    if stripped:
        logging.info("[MMH3LoopingSampler] stripped stale keyframes from %s; this node "
                     "builds its own per chunk", label)
    return out


def _has_refs(cond):
    return any("minimax_refs" in d and d["minimax_refs"] for _t, d in cond)


def _guides_available():
    """Whether #15439's any-index guides are present.

    Detected, not assumed: stock raises on any anchor that is not first/last,
    and a guide carrying a multi-step clip is not expressible at all.
    """
    try:
        import inspect
        import comfy.ldm.minimax.model as mm
        return "only first/last keyframe anchors" not in inspect.getsource(
            mm.PackedLayout.__init__)
    except Exception:
        return False


def _guide_origin_correct():
    """Whether a guide anchors on the TARGET origin when references are present.

    #15439 anchors to text_len, but the target begins after the refs. See
    patch_guide_origin; this asks the patch first and probes only as a fallback.
    """
    from . import patch_guide_origin
    if patch_guide_origin.is_applied():
        return True
    try:
        import comfy.ldm.minimax.model as mm
        lay = mm.PackedLayout(
            8, 7, 4, 4, 8,
            keyframes=[{"resolved_frame_index": 0,
                        "latent": torch.zeros([1, 24, 1, 4, 4])}],
            refs=[{"kind": "image", "latent_h": 4, "latent_w": 4,
                   "latent": torch.zeros([1, 24, 1, 4, 4])}])
        seg = {k: a for a, _b, k in lay.segments}
        return abs(float(lay.position_ids[seg["cond"], 0])
                   - float(lay.position_ids[seg["video"], 0])) < 1e-6
    except Exception:
        return False


def _raw_conds(guider):
    """This guider's (positive, negative) in the form set_conds accepts.

    negative is None for a BasicGuider, which has no such key at all --
    Guider_Basic.set_conds takes ONE argument. Indexing for it raises KeyError,
    and calling set_conds with two raises TypeError.
    """
    if hasattr(guider, "raw_conds"):
        return guider.raw_conds
    conds = getattr(guider, "original_conds", {}) or {}
    return (conds.get("positive"), conds.get("negative"))


def _chunk_guider(guider, positive, frame0=None):
    """This chunk's guider: the wired one, with its POSITIVE replaced.

    `frame0` is published to the model as `mmh3_control_frame0` so anything reading
    the sequence per chunk -- the Fun ControlNet wrapper, for one -- knows WHERE in
    the clip this chunk starts. Core's controlnet picks its hint frames from index 0
    and caches by shape, and every chunk shares a shape, so without this every chunk
    is driven by the control video's opening frames with no error raised.
    """
    new_g = copy.copy(guider)
    # SHALLOW copy shares original_conds; set_conds assigns into it. Rebind
    # before touching it or chunk 0 clobbers the base conditioning.
    new_g.original_conds = dict(guider.original_conds)
    if frame0 is not None:
        # model_options is shared by the same shallow copy, so rebind BOTH levels
        # rather than writing the offset into the caller's guider.
        opts = dict(getattr(guider, "model_options", {}) or {})
        opts["transformer_options"] = dict(opts.get("transformer_options", {}))
        opts["transformer_options"]["mmh3_control_frame0"] = int(frame0)
        new_g.model_options = opts
    _, negative = _raw_conds(guider)
    if negative is None:
        new_g.set_conds(positive)              # Guider_Basic: no CFG, no negative
    else:
        new_g.set_conds(positive, negative)
    new_g.raw_conds = (positive, negative)
    return new_g


def _run_sampling(noise, guider, sampler, sigmas, av_init,
                  sampling_start_step, sampling_end_step,
                  phase2_sampler=None, phase2_guider=None, phase2_start_step=0):
    """Run one chunk's schedule as sequential resample-continuation segments.

    NOTHING HERE WINDOWS A GUIDE. A keyframe guide is not a per-step influence:
    it is registered on the conditioning and re-injected every step, so it is
    structural for the whole chunk. Releasing it mid-schedule would mean changing
    the packed layout between steps, which is not expressible -- and the phase-2
    guider is deliberately given this chunk's FULL conditioning, so it cannot
    release it either. To drop a guide you need a separate pass whose
    conditioning never had it.

    sampling_start_step / sampling_end_step WINDOW THE SCHEDULE, using core
    SplitSigmas' own slicing (first output sigmas[:step+1], second output
    sigmas[step:], sharing the boundary sigma). Start skips the steps before it,
    re-noising the incoming latent to that sigma; end discards the ones after,
    leaving a partially denoised latent. Both are ABSOLUTE indices into the
    incoming schedule, so pass 1 `end N` hands off to pass 2 `start N` with no
    arithmetic on the user's part.

    The remaining split point is phase2_start_step, where the sampler/guider pair
    switches for dual-solver schedules. It is rebased onto the sliced schedule.

    Straight from LTXAVTools' looping sampler, which solved this first.
    """
    start = max(0, int(sampling_start_step))
    if start > 0:
        if sampling_end_step <= start:
            raise ValueError(
                "MMH3LoopingSampler: sampling_start_step (%d) must be below "
                "sampling_end_step (%d) -- that window contains no steps. These are "
                "absolute indices into the sigma schedule: start skips the steps "
                "before it, end drops the ones after."
                % (start, int(sampling_end_step)))
        _, sigmas = SplitSigmas().get_sigmas(sigmas, start)
        if len(sigmas) <= 1:
            raise ValueError(
                "MMH3LoopingSampler: sampling_start_step %d is at or past the end of "
                "the sigma schedule -- nothing left to sample. Lower it, or give the "
                "scheduler more steps." % start)

    use_phase2 = phase2_sampler is not None and int(phase2_start_step) > 0
    phase2_local = int(phase2_start_step) - start
    end_step = int(sampling_end_step) - start

    cut_points = sorted(
        p for p in ({phase2_local} if use_phase2 else set()) if 0 < p < end_step)

    segments = []
    remaining = sigmas
    prev = 0
    for p in cut_points:
        seg, remaining = SplitSigmas().get_sigmas(remaining, p - prev)
        segments.append((prev, seg))
        prev = p
    tail, _ = SplitSigmas().get_sigmas(remaining, end_step - prev)
    segments.append((prev, tail))

    current = av_init
    for seg_start, seg_sigmas in segments:
        if len(seg_sigmas) <= 1:
            continue
        seg_sampler, seg_guider = sampler, guider
        if use_phase2 and seg_start >= phase2_local:
            seg_sampler = phase2_sampler
            seg_guider = phase2_guider if phase2_guider is not None else guider
        _, current = SamplerCustomAdvanced().sample(
            noise, seg_guider, seg_sampler, seg_sigmas, current)
    return current


def _chunk_noise(noise, index):
    """A distinct noise per chunk. Reusing one object gives every chunk the
    same noise, which reads as the model refusing to advance."""
    if index == 0 or not hasattr(noise, "seed"):
        return noise
    n = copy.copy(noise)
    n.seed = int(noise.seed) + index
    return n


def _parse_indices(text, total_frames):
    """Comma-separated GLOBAL pixel-frame indices; negatives count from the end.

    Resolved here rather than passed through: PackedLayout takes a negative
    literally, so cond_t would fall below text_len and collide with the text
    token positions. Out of range is an error -- silently dropping a keyframe
    the user asked for is worse than stopping.
    """
    out = []
    for piece in (text or "").split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            v = int(piece)
        except ValueError:
            raise ValueError(
                "MMH3LoopingSampler: keyframe_indices has %r in it, which is not a "
                "whole number. Expected something like '0, 60, -1'." % piece)
        if v < 0:
            v += total_frames
        if not 0 <= v < total_frames:
            raise ValueError(
                "MMH3LoopingSampler: keyframe index %s is outside the clip, which is "
                "%d frames (0-%d)." % (piece, total_frames, total_frames - 1))
        out.append(v)
    return out


def _fit_keyframe(px, tgt_w, tgt_h, is_opener):
    """Resize one keyframe still to the TARGET grid. -> (pixels, note or None).

    Keyframe rows share the target's spatial grid -- PackedLayout reads only the
    latent's TIME dim and sizes the segment from the target, so a still at any
    other resolution reserves the wrong number of rows and dies deep in the model
    with a broadcast error that names nothing. The other two keyframe paths already
    handle this (MMH3ImageKeyframe resizes internally, MMH3LatentKeyframe has an
    opt-in guard); this is the third, and it used to just encode whatever arrived.

    Resizing here rather than refusing is deliberate: a ladder runs the same still
    against 2-3 different target resolutions, and making the graph carry a resize
    per stage is busywork the node has the numbers to do itself.

    Aspect policy mirrors MMH3ImageKeyframe's 'auto', which is the stock node's:
    the frame-0 opener STRETCHES because it sets the clip's geometry, every later
    anchor CENTRE CROPS because it follows one already established. Identical
    results when the aspect already matches, which is the normal case.
    """
    src_h, src_w = int(px.shape[1]), int(px.shape[2])
    if (src_h, src_w) == (int(tgt_h), int(tgt_w)):
        return px, None
    crop = "disabled" if is_opener else "center"
    out = comfy.utils.common_upscale(
        px[..., :3].movedim(-1, 1), int(tgt_w), int(tgt_h), "lanczos", crop
    ).movedim(1, -1)
    return out, ("%dx%d -> %dx%d (%s)"
                 % (src_w, src_h, int(tgt_w), int(tgt_h),
                    "stretch" if is_opener else "centre crop"))


def _owner(spans, overlap_frames, g):
    """Which chunk RENDERS global frame g, and its local frame there.

    Chunks overlap, and a frame inside a chunk's overlap is carried from the
    previous chunk rather than generated, so anchoring a keyframe there paints
    a frame that is conditioned on, not drawn. Assign it to the LAST chunk whose
    NEW content covers it; chunk 0 has no carry so it owns its whole span.
    """
    owner = None
    for i, (a, b) in enumerate(spans):
        if not a <= g <= b:
            continue
        if i == 0 or (g - a) >= overlap_frames:
            owner = (i, g - a)
    if owner is None:
        for i, (a, b) in enumerate(spans):
            if a <= g <= b:
                return (i, g - a)
    return owner


class MMH3LoopingSampler(io.ComfyNode):
    """Fill one clip chunk by chunk, carrying each chunk's tail into the next."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3LoopingSampler",
            display_name="MiniMax H3 Looping Sampler",
            category="MMH3Tools/sampling",
            description=(
                "Fill a whole clip chunk by chunk in ONE node execution. The latent "
                "is the finished length; the chunk count is derived from it, so a "
                "127s track needs no arithmetic on your part.\n\n"
                "The schedule is the SAME one MMH3 Window Plan and MMH3 Split Audio "
                "to Windows compute, so chunk N renders the audio that window N's "
                "prompt was written against. Prompts must all exist before sampling "
                "starts -- wire a cond_set from MiniMax H3 Reference (Multi-Prompt)."
            ),
            inputs=[
                io.Noise.Input("noise"),
                io.Guider.Input(
                    "guider",
                    tooltip="Supplies the MODEL, the cfg, and the negative if it is a "
                            "CFG guider. Its POSITIVE is replaced every chunk from the "
                            "cond_set, so whatever is wired there is ignored -- wire "
                            "MMH3 Cond Select at index 0 so the graph is valid and says "
                            "what it means. A Basic Guider works too; it has no negative."),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                MMH3CondSet.Input("cond_set"),
                io.Latent.Input(
                    "latent",
                    tooltip="The WHOLE clip's AV latent -- the finished length, not one "
                            "chunk. Chunks are slices of it and are written back in "
                            "place, so the output is exactly this length.\n\n"
                            "If its audio half holds a real track (MMH3 Reference "
                            "(Multi-Prompt)'s use_input_audio), each chunk gets ITS OWN "
                            "span of it automatically."),
                io.Int.Input(
                    "chunk_frames", default=192, min=0, max=3600, step=17,
                    tooltip="New content per chunk, in frames at 24 fps. Snapped to the "
                            "5j+2 latent grid. The chunk COUNT falls out of this and the "
                            "clip length -- wire the same value MMH3 Window Plan gets "
                            "and the two schedules are identical.\n\n"
                            "0 = ONE CHUNK covering everything to be generated, sized "
                            "here: the region plus the carry plus any grid padding. One "
                            "chunk means one prompt, whatever the length works out to."),
                io.Int.Input(
                    "overlap_frames", default=22, min=0, max=3600, step=17,
                    tooltip="Frames each chunk carries from the previous one. Snapped so "
                            "the stride keeps every chunk on latent phase 0; an overlap "
                            "that is a multiple of 5 rather than 5m+2 walks the phase "
                            "0,2,4,1,3, which is a five-chunk beat."),
                io.Combo.Input(
                    "carry", options=["mask", "keyframe"], default="mask",
                    tooltip="HOW the previous chunk reaches the next.\n\n"
                            "'mask' masks the overlap latents, which already hold the "
                            "previous chunk's output, so the model conditions on them "
                            "without denoising them. Needs #15375.\n\n"
                            "'keyframe' passes them as a GUIDE anchored at frame 0 -- "
                            "re-injected every step, never denoised, carrying the audio "
                            "at the same coordinate. Needs #15439."),
                io.Float.Input(
                    "overlap_strength_video", default=1.0, min=0.0, max=1.0, step=0.05,
                    tooltip="1.0 preserves the carried region outright, 0.0 regenerates "
                            "it. `mask` carry only."),
                io.Float.Input(
                    "overlap_strength_audio", default=0.9, min=0.0, max=1.0, step=0.05,
                    tooltip="Noise mask for the carried AUDIO latents: mask = 1 - "
                            "strength. 1.0 pins them to the previous chunk's audio, 0.0 "
                            "regenerates them. Independent of the video strength. `mask` "
                            "carry only."),
                io.Int.Input(
                    "sampling_start_step", default=0, min=0, max=1000,
                    tooltip="Begin at this step, skipping the ones before it -- the "
                            "incoming latent is re-noised to that sigma and finished from "
                            "there. Same slice as core SplitSigmas' second output "
                            "(sigmas[step:]). Use it to continue a partial pass: set "
                            "sampling_end_step N on pass 1 and sampling_start_step N on "
                            "pass 2. 0 = start at the beginning. Step numbers are "
                            "ABSOLUTE indices into the incoming schedule.\n\n"
                            "Applies WITHIN every chunk, not across chunks."),
                io.Int.Input(
                    "sampling_end_step", default=1000, min=0, max=1000,
                    tooltip="Stop after this step; later ones are DISCARDED, leaving a "
                            "partially denoised latent. Same slice as core SplitSigmas' "
                            "first output (sigmas[:step+1]). Pairs with "
                            "sampling_start_step to window the schedule without wiring "
                            "SplitSigmas. Not a guide control -- a keyframe guide is "
                            "structural for the whole chunk and cannot be windowed. Step "
                            "numbers are ABSOLUTE."),
                io.Int.Input(
                    "phase2_start_step", default=0, min=0, max=1000,
                    tooltip="Schedule step where phase 2 takes over (0 = disabled). E.g. "
                            "4 on a 12-step schedule: steps 0-3 use the main "
                            "sampler/guider, steps 4+ use the phase-2 pair. Rebased onto "
                            "the window sampling_start_step leaves, so it stays an "
                            "ABSOLUTE index like the other two."),
                io.Sampler.Input(
                    "phase2_sampler", optional=True,
                    tooltip="Second-phase sampler for dual-solver schedules (e.g. a heavy "
                            "solver for the first steps, euler for the rest). Takes over "
                            "at phase2_start_step within EVERY chunk's schedule, "
                            "resample-style continuation."),
                io.Guider.Input(
                    "phase2_guider", optional=True,
                    tooltip="Guider for the second phase (e.g. cfg 1.0 while phase 1 runs "
                            "cfg 2.0). Like the main guider its POSITIVE is replaced every "
                            "chunk from the cond_set -- only its guidance settings apply. "
                            "Falls back to the main guider if unconnected."),
                io.Image.Input(
                    "keyframes", optional=True,
                    tooltip="A BATCH of stills to pin, one per index in keyframe_indices. "
                            "Encoded once here. Independent of `carry`. Resized to the "
                            "generation's resolution first -- keyframe rows share the "
                            "target grid, so a still at another size cannot be used as-is. "
                            "The frame-0 opener is stretched, later anchors are centre "
                            "cropped; every resize is named in the log and the report."),
                io.String.Input(
                    "keyframe_indices", multiline=False, default="",
                    tooltip="Comma-separated frame indices into the WHOLE clip -- place a "
                            "shot where it belongs and this works out which chunk renders "
                            "it. Negatives count from the end. An index inside a chunk's "
                            "carried overlap goes to the chunk that actually draws it. "
                            "Ignored entirely when no `keyframes` are attached, so one "
                            "graph can be reused across passes that do and do not anchor."),
                io.Vae.Input(
                    "vae", optional=True,
                    tooltip="The H3 VIDEO vae, needed only to encode `keyframes`."),
                io.Mask.Input(
                    "denoise_mask", optional=True,
                    tooltip="Denoise mask for the VIDEO half over the WHOLE clip: WHITE "
                            "regenerates, BLACK keeps the input latent's content. The "
                            "AUDIO half is NOT affected -- mask it with "
                            "`audio_denoise_mask`.\n\n"
                            "Reduced onto the latent grid with the VAE's real geometry "
                            "and snapped to the DiT's 2x2 patch, so 32 pixels is the "
                            "finest feature it can express.\n\n"
                            "Merged keep-wins (elementwise min) with the mask the "
                            "latent already carried and with the overlap carry, then "
                            "sliced per chunk. Kept regions reproduce the input latent, "
                            "so an empty latent pins BLACK -- this is a v2v tool."),
                io.Combo.Input(
                    "denoise_mask_mode", options=MASK_MODES, default="max",
                    optional=True,
                    tooltip="How BOTH masks are reduced onto the latent grid, "
                            "spatially and across each latent's frame group. 'max' is "
                            "the union and the safe default; 'min' is the intersection; "
                            "'mean' gives fractional edges; 'last' takes each group's "
                            "final frame. `audio_denoise_mask` is pooled to 1x1, so "
                            "only the temporal grouping reaches it."),
                io.Mask.Input(
                    "audio_denoise_mask", optional=True,
                    tooltip="Denoise mask for the AUDIO half: WHITE regenerates, BLACK "
                            "keeps the input latent's audio. Only its TIME axis is "
                            "read -- each frame reduces to one value, mapped onto the "
                            "audio grid through the same boundary conversion the chunk "
                            "loop uses, so a span frozen here lines up with the "
                            "picture.\n\n"
                            "INDEPENDENT of `denoise_mask`: freeing video does not free "
                            "audio, or the reverse. Left unconnected, audio is masked "
                            "only by whatever the input latent already carried -- see "
                            "`preserve_masks` on MiniMax H3 Split AV, which is what "
                            "lets a `use_input_audio` pin survive a split and repack."),
                # HIDDEN 2026-08-21. The prior-continuation path never behaved
                # correctly in practice, so the socket is withheld from the schema
                # rather than deleted: `execute` still accepts `prior_av_latent=None`
                # and the prepend/pad/phase implementation below is untouched.
                # Restore by uncommenting -- it was the LAST input, so putting it
                # back appends and cannot disturb widget order in saved graphs.
                # io.Latent.Input(
                # "prior_av_latent", optional=True,
                # tooltip="An already-rendered AV latent to continue from. It is copied "
                # "to the output verbatim and never sampled; `latent` then "
                # "describes only the NEW region, and the output is prior + new.\n\n"
                # "The schedule is planned over the COMBINED length, and every "
                # "window lying inside the prior is skipped. The first generated "
                # "chunk therefore overlaps the prior's tail and carries it like "
                # "any earlier chunk's tail, so the prior's length does not have "
                # "to line up with anything.\n\n"
                # "Prompts map to the GENERATED chunks: cond 0 is the first chunk "
                # "actually sampled, not the first window of the combined clip."),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="chunks_rendered"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, noise, guider, sampler, sigmas, cond_set, latent, chunk_frames,
                overlap_frames, carry, overlap_strength_video, overlap_strength_audio,
                sampling_start_step=0, sampling_end_step=1000,
                phase2_start_step=0, phase2_sampler=None, phase2_guider=None,
                keyframes=None, keyframe_indices="", vae=None,
                prior_av_latent=None, denoise_mask=None, denoise_mask_mode="max",
                audio_denoise_mask=None) -> io.NodeOutput:
        conds = (cond_set or {}).get("conds") or []
        if not conds:
            raise ValueError("MMH3LoopingSampler: cond_set holds no conditioning.")

        if carry == "keyframe" and not _guides_available():
            raise RuntimeError(
                "MMH3LoopingSampler: carry='keyframe' needs any-index guides (upstream "
                "PR #15439), which is not applied. See docs/core-changes.md. Use "
                "carry='mask', which needs only #15375.")

        master_v, master_a = unpack_av(latent, "latent")

        # A prior is PREPENDED and the schedule planned over the combined clip, so the
        # prior needs no relationship to the window size: whichever window first
        # reaches past it becomes chunk 0, already overlapping it, and the ordinary
        # carry rule takes the prior's tail from there.
        prior_t = prior_at = 0
        if prior_av_latent is not None:
            prior_v, prior_a = unpack_av(prior_av_latent, "prior_av_latent")
            if prior_v.shape[1] != master_v.shape[1] or \
                    tuple(prior_v.shape[3:]) != tuple(master_v.shape[3:]):
                raise ValueError(
                    "MMH3LoopingSampler: prior_av_latent is %s but the target latent is "
                    "%s. They are concatenated on the time axis, so channels and frame "
                    "size must match."
                    % (tuple(prior_v.shape), tuple(master_v.shape)))
            if (prior_a is None) != (master_a is None):
                raise ValueError(
                    "MMH3LoopingSampler: one of prior_av_latent / latent carries audio "
                    "and the other does not. Both must be AV, or both video-only.")
            prior_t = int(prior_v.shape[VIDEO_T_DIM])
            keep_v = prior_v.to(master_v.dtype).clone()
            keep_a = None

            # GRID. A standalone clip is 5j+2 latents, so prior + new is 5k+4 -- not a
            # valid clip at all, and latents_to_frames() floors it, leaving the tail of
            # the new region outside every window and never sampled. Pad the NEW region
            # up instead: the prior is real footage that must not be invented or
            # discarded, and (prior_t + new_t) must be 2 mod 5, so the new side has to
            # be 0 mod 5. Costs at most 4 latents of extra generation.
            #
            # It also settles the phase: offset = prior_t - overlap = (5a+2) - (5m+2),
            # a multiple of 5, so every window still starts on phase 0 -- which H3's
            # FRAME_PER_TOKEN (1,4,4,4,4) indexing requires.
            pad_t = (LATENT_BASE - (prior_t + int(master_v.shape[VIDEO_T_DIM]))) \
                % LATENTS_PER_GROUP
            if pad_t:
                tail = master_v[:, :, -1:].repeat_interleave(pad_t, dim=VIDEO_T_DIM)
                master_v = torch.cat([master_v, torch.zeros_like(tail)], VIDEO_T_DIM)

            master_v = torch.cat([keep_v, master_v], VIDEO_T_DIM)
            if master_a is not None:
                prior_at = int(prior_a.shape[AUDIO_T_DIM])
                # The two axes are concatenated independently, so a prior whose audio
                # does not correspond to its own video silently shifts EVERYTHING after
                # it. This is easy to hit: encoding a loaded .mp4 counts audio from the
                # track's duration, and encoders routinely pad it past the last frame.
                want_at = frames_to_audio_t(latents_to_frames(prior_t))
                if prior_at != want_at:
                    drift = (prior_at - want_at) / float(AUDIO_LATENT_FPS)
                    raise ValueError(
                        "MMH3LoopingSampler: prior_av_latent has %d audio latents but "
                        "its %d video latents (%d frames) need %d -- %+.3fs of drift. "
                        "Everything after the prior would shift by that much. Trim the "
                        "prior's audio to its video (MMH3 Trim AV), or re-encode the "
                        "source so the two match."
                        % (prior_at, prior_t, latents_to_frames(prior_t), want_at, drift))
                keep_a = prior_a.to(master_a.dtype).clone()
                master_a = torch.cat([keep_a, master_a], AUDIO_T_DIM)
                # Audio is 40Hz against 24fps video, so it does not pad by the same
                # count. Size it from the combined VIDEO length, which is now grid
                # valid, rather than adding pad_t audio latents and hoping.
                want = frames_to_audio_t(
                    latents_to_frames(int(master_v.shape[VIDEO_T_DIM])))
                have = int(master_a.shape[AUDIO_T_DIM])
                if have < want:
                    master_a = torch.cat(
                        [master_a, torch.zeros_like(master_a[:, :, :, :1])
                         .repeat_interleave(want - have, dim=AUDIO_T_DIM)], AUDIO_T_DIM)
                elif have > want:
                    master_a = master_a[:, :, :, :want]

        total_t = int(master_v.shape[VIDEO_T_DIM])
        total_a = 0 if master_a is None else int(master_a.shape[AUDIO_T_DIM])
        total_f = latents_to_frames(total_t)

        # THE SAME schedule MMH3WindowPlan and MMH3SplitAudioToWindows compute, so
        # chunk N renders the audio window N's prompt was written against.
        #
        # With a prior, the schedule covers only the GENERATED span -- one carry plus
        # the new region -- and is then offset onto the combined clip. Planning over
        # the combined clip instead made the prior's length shift every window
        # boundary, so whether the new region fell in one chunk or two depended on how
        # the total happened to divide: chunk_frames 124 gave one chunk after a 10s
        # prior and two after a 20s one. Planning the generated span makes the prior's
        # length genuinely irrelevant, which is the whole point of it being arbitrary.
        # chunk_frames 0 asks for a single chunk over everything being generated. The
        # size is not something to work out by hand: it is the region PLUS the carry
        # PLUS whatever the grid padding came to, and getting it wrong by one grid step
        # silently costs a second chunk and therefore a second prompt.
        cf = int(chunk_frames)
        one_chunk = cf <= 0
        probe_cf = 3600 if one_chunk else cf

        plan_over_f = total_f
        if prior_t:
            _l, ov_probe, _pf, _pt, _w = _plan(
                latents_to_frames(total_t - prior_t), probe_cf,
                int(overlap_frames), "standard_static")
            plan_over_f = latents_to_frames(total_t - prior_t + ov_probe)
        if one_chunk:
            cf = plan_over_f

        length, overlap, plan_f, _plan_t, windows = _plan(
            plan_over_f, cf, int(overlap_frames), "standard_static")
        ov_frames = frame_at_latent(overlap)

        # Where the generated span begins in the combined clip: one carry before the
        # new region, so chunk 0 opens on the prior's tail.
        offset = max(0, prior_t - overlap) if prior_t else 0
        spans = [(frame_at_latent(min(w.index_list[0] + offset, total_t - 1)),
                  min(frame_at_latent(w.index_list[-1] + 1 + offset) - 1, total_f - 1))
                 for w in windows]
        n = len(windows)

        # the carry settings belong in the SUMMARY, not just the widgets: without
        # them two runs are indistinguishable in the log, and comparing a render
        # against an earlier one is exactly what diagnosing carry drift needs
        lines = ["%d chunk%s of %d latents (%d frames) over %d frames (%.2fs), "
                 "overlap %d latents (%d frames)"
                 % (n, "" if n == 1 else "s", length, latents_to_frames(length),
                    total_f, total_f / float(FPS), overlap, ov_frames),
                 "carry %s, strength video %.2f / audio %.2f, noise seed %s"
                 % (carry, float(overlap_strength_video), float(overlap_strength_audio),
                    getattr(noise, "seed", "?"))]
        if prior_t:
            lines.append("prior: %d latents (%d frames, %.2fs) kept verbatim; generating "
                         "from frame %d, carrying %d frames of it"
                         % (prior_t, frame_at_latent(prior_t),
                            frame_at_latent(prior_t) / float(FPS), spans[0][0],
                            frame_at_latent(prior_t - offset)))
        if len(conds) != n:
            lines.append("! %d prompt%s for %d chunks -- %s"
                         % (len(conds), "" if len(conds) == 1 else "s", n,
                            "the last repeats" if len(conds) < n
                            else "the extras are unused"))
            logging.warning("[MMH3LoopingSampler] %s", lines[-1][2:])

        # Schedule windowing is per chunk and invisible in the output otherwise --
        # a partially denoised master looks like a bad seed, not a setting.
        n_sig = max(0, len(sigmas) - 1)
        if int(sampling_start_step) > 0 or int(sampling_end_step) < n_sig:
            lines.append("schedule window: steps %d-%d of %d, per chunk"
                         % (int(sampling_start_step),
                            min(int(sampling_end_step), n_sig), n_sig))
            if int(sampling_end_step) < n_sig:
                lines[-1] += " -- output is PARTIALLY denoised"
        if phase2_sampler is not None and int(phase2_start_step) > 0:
            lines.append("phase 2 from step %d%s"
                         % (int(phase2_start_step),
                            "" if phase2_guider is None else ", with its own guider"))

        # ---- keyframes, encoded ONCE, placed on the real timeline -----------
        guides_by_chunk = {}
        if keyframes is None:
            # Indices with no images attached are INERT, not fatal. A ladder reuses
            # one graph across passes and usually only the first pass carries
            # anchors, so a live keyframe_indices string with the image input
            # unplugged is the normal state of a refine pass. Parsing is skipped
            # entirely rather than parsed-then-discarded: with nothing to place,
            # an out-of-range index is not a mistake worth stopping for either.
            wanted = []
            if (keyframe_indices or "").strip():
                logging.info("[MMH3LoopingSampler] keyframe_indices is set (%r) but no "
                             "keyframes are attached; ignoring it",
                             (keyframe_indices or "").strip())
                lines.append("  keyframe_indices ignored: no keyframes attached")
        else:
            wanted = _parse_indices(keyframe_indices, total_f)
        if keyframes is not None and wanted:
            if vae is None:
                raise ValueError("MMH3LoopingSampler: keyframes need the H3 video vae.")
            if int(keyframes.shape[0]) != len(wanted):
                raise ValueError(
                    "MMH3LoopingSampler: %d keyframe image(s) against %d index/indices. "
                    "They are zipped, so the counts must match."
                    % (int(keyframes.shape[0]), len(wanted)))
            if not _guides_available():
                raise RuntimeError(
                    "MMH3LoopingSampler: keyframes need any-index guides (PR #15439).")
            # keyframe rows share the TARGET grid, so a still at any other size has
            # to be fitted here or it fails deep in the model -- see _fit_keyframe
            tgt_h = int(master_v.shape[3]) * VAE_SPATIAL
            tgt_w = int(master_v.shape[4]) * VAE_SPATIAL
            for img_i, g in enumerate(wanted):
                ci, local = _owner(spans, ov_frames, g)
                px, note = _fit_keyframe(keyframes[img_i:img_i + 1], tgt_w, tgt_h,
                                         is_opener=(g == 0))
                if note:
                    logging.info("[MMH3LoopingSampler] keyframe frame %d resized %s",
                                 g, note)
                z = vae.encode(px)
                guides_by_chunk.setdefault(ci, []).append(
                    {"resolved_frame_index": int(local), "latent": z})
                lines.append("  keyframe frame %d -> chunk %d local frame %d%s"
                             % (g, ci, local, "" if not note else ", resized " + note))

        # ---- fill the master, chunk by chunk --------------------------------
        out_v = master_v.clone()
        out_a = None if master_a is None else master_a.clone()
        in_mask_v, in_mask_a = _split_mask(latent)

        if in_mask_v is not None or in_mask_a is not None:
            lines.append("  input latent carried a noise_mask (video %s, audio %s)"
                         % ("yes" if in_mask_v is not None else "no",
                            "yes" if in_mask_a is not None else "no"))

        if denoise_mask is not None or audio_denoise_mask is not None:
            if not per_row_mask_is_continuous():
                raise RuntimeError(
                    "MMH3LoopingSampler: the mask inputs need per-row masking (#15375). "
                    "On this core a mask is accepted and IGNORED -- you would get a full "
                    "regeneration with nothing raised. Update ComfyUI, or unwire the "
                    "mask.")

        # The two halves are INDEPENDENT. `denoise_mask` reduces onto the video grid and
        # stops there; audio is masked only by `audio_denoise_mask`, or by whatever the
        # latent already carried.
        #
        # Deriving one from the other reads as a safety feature -- the two can then never
        # disagree about a frozen span -- and is wrong, because a mask only carries
        # temporal intent when it IS temporal. A subject matted out of every frame is
        # white somewhere at every timestep, so the spatial reduction returned "free"
        # everywhere and regenerated a track that `use_input_audio` had pinned.
        if denoise_mask is not None:
            mv = _mask_to_video_latent(
                denoise_mask.to(master_v.device), total_t,
                int(master_v.shape[3]), int(master_v.shape[4]), denoise_mask_mode)
            kept = float((mv < 0.5).float().mean()) * 100.0
            lines.append("  denoise_mask: %.1f%% of the video grid kept (pinned to the "
                         "input latent); the audio half is untouched" % kept)
            if float(master_v.abs().max()) < 1e-6:
                lines.append("  ! the input video latent is all zeros, so kept regions "
                             "pin BLACK -- encode the source into `latent`")
            in_mask_v = mv if in_mask_v is None else torch.minimum(
                in_mask_v.to(mv.dtype), mv)

        if audio_denoise_mask is not None:
            if master_a is None:
                lines.append("  ! audio_denoise_mask is wired but this latent has no "
                             "audio half -- ignored")
            else:
                am = _mask_to_video_latent(
                    audio_denoise_mask.to(master_v.device), total_t, 1, 1,
                    denoise_mask_mode)
                prof = _video_mask_to_audio(am, total_t, total_a, denoise_mask_mode)
                free = float((prof > 0.5).float().mean()) * 100.0
                lines.append("  audio_denoise_mask: %.1f%% of the track free to "
                             "regenerate" % free)
                ma = _audio_profile_to_mask(prof, master_a)
                in_mask_a = ma if in_mask_a is None else torch.minimum(
                    in_mask_a.to(ma.dtype), ma)

        for i, w in enumerate(windows):
            idx = w.index_list
            v0, v1 = idx[0] + offset, min(idx[-1] + 1 + offset, total_t)
            a0 = _audio_index_at(v0, total_t, total_a)
            a1 = _audio_index_at(v1, total_t, total_a)

            # Carry whatever the PREVIOUS window actually reached, not the nominal
            # overlap. Core clamps the LAST window back so it ends on the clip end,
            # which makes it physically overlap its predecessor by far more than
            # `overlap` -- 62 latents against a nominal 7 on a 127s clip in 20s
            # chunks. Preserving only the nominal 7 left the other 55 to be
            # regenerated under THIS chunk's prompt and written over content the
            # previous chunk had already drawn: up to 12s of the second-to-last
            # section silently taking the last section's conditioning. Middle
            # windows are unaffected -- there actual == nominal.
            #
            # With a prior, chunk 0 is an EXTEND chunk: the prior ends at prior_t, so
            # that is its predecessor's end and the same rule takes the prior's tail.
            prev_end = (windows[i - 1].index_list[-1] + 1 + offset) if i else prior_t
            carried = min(max(0, prev_end - v0), v1 - v0) if (i or prior_t) else 0
            if carried > overlap:
                lines.append("  chunk %d: clamped tail -- carries %d latents (%d "
                             "frames) instead of %d, so the previous chunk's content "
                             "survives" % (i, carried, frame_at_latent(carried),
                                           overlap))
                logging.info("[MMH3LoopingSampler] %s", lines[-1].strip())

            sub_v = out_v[:, :, v0:v1].clone()
            sub_a = None if out_a is None else out_a[:, :, :, a0:a1].clone()
            chunk = pack_av(latent, sub_v, sub_a, noise_mask=None)

            chunk_cond = _strip_guide_keys(conds[min(i, len(conds) - 1)],
                                           "chunk %d prompt" % i)
            chunk_guides = list(guides_by_chunk.get(i, []))

            if carried and carry == "keyframe":
                kf = {"resolved_frame_index": 0,
                      "latent": sub_v[:, :, :carried].contiguous()}
                if sub_a is not None:
                    ca = _audio_index_at(v0 + carried, total_t, total_a) - a0
                    if ca > 0:
                        kf["audio_latent"] = sub_a[:, :, :, :ca].contiguous()
                chunk_guides.insert(0, kf)
            elif carried:
                chunk["noise_mask"] = _carry_mask(
                    sub_v, sub_a, carried,
                    _audio_index_at(v0 + carried, total_t, total_a) - a0,
                    float(overlap_strength_video), float(overlap_strength_audio),
                    in_mask_v, in_mask_a, v0, v1, a0, a1)
            elif in_mask_v is not None or in_mask_a is not None:
                chunk["noise_mask"] = _sliced_mask(sub_v, sub_a, in_mask_v, in_mask_a,
                                                   v0, v1, a0, a1)

            if chunk_guides:
                if _has_refs(chunk_cond) and not _guide_origin_correct():
                    raise RuntimeError(
                        "MMH3LoopingSampler: chunk %d carries a reference AND a "
                        "keyframe, but this ComfyUI anchors guides on text_len instead "
                        "of the target origin. See docs/core-changes.md." % i)
                chunk_cond = node_helpers.conditioning_set_values(
                    chunk_cond, {"minimax_keyframes": chunk_guides})

            # frame_at_latent, NOT latents_to_frames: v0 is an arbitrary window
            # bound, and latents_to_frames is only meaningful on the 5j+2 grid --
            # it answers -12 for index 1.
            g = _chunk_guider(guider, chunk_cond, frame0=frame_at_latent(v0))
            # The phase-2 guider gets THIS chunk's conditioning too. Only its
            # guidance settings are wanted; its own positive would hand the tail
            # of every chunk whatever prompt happens to be wired to it.
            g2 = None if phase2_guider is None else _chunk_guider(
                phase2_guider, chunk_cond, frame0=frame_at_latent(v0))
            done = _run_sampling(
                _chunk_noise(noise, i), g, sampler, sigmas, chunk,
                sampling_start_step, sampling_end_step,
                phase2_sampler, g2, phase2_start_step)

            dv, da = unpack_av(done, "chunk %d output" % i)
            out_v[:, :, v0:v1] = dv.to(out_v.dtype)
            if out_a is not None and da is not None:
                out_a[:, :, :, a0:a1] = da.to(out_a.dtype)

            lines.append("  chunk %d: prompt %d, frames %d-%d, %d carried%s"
                         % (i, min(i, len(conds) - 1), spans[i][0], spans[i][1],
                            frame_at_latent(carried) if carried else 0,
                            ", %d keyframe(s)" % (len(chunk_guides) - (1 if carried
                                                  and carry == "keyframe" else 0))
                            if guides_by_chunk.get(i) else ""))
            logging.info("[MMH3LoopingSampler] chunk %d/%d done", i + 1, n)

        # The first generated chunk OVERLAPS the prior -- that is what the carry is --
        # so its slice covers the prior's tail and the write-back lands on it. The mask
        # protects that tail only at strength 1.0, and overlap_strength_audio defaults
        # to 0.9, so the source's last fraction of a second would come back altered.
        # The carry has already done its job as context by now; restore the prior so
        # the output is the input plus the addition, byte for byte.
        if prior_t:
            out_v[:, :, :prior_t] = keep_v
            if out_a is not None and keep_a is not None:
                out_a[:, :, :, :prior_at] = keep_a

        out = pack_av(latent, out_v, out_a, noise_mask=None)
        lines.append("master: %d latents (%d frames, %.2fs) -- the input length, exactly"
                     % (total_t, total_f, total_f / float(FPS)))
        report = "\n".join(lines)
        logging.info("[MMH3LoopingSampler] " + lines[0])
        return io.NodeOutput(out, n, report)


def _split_mask(latent):
    """The latent's own noise mask per modality, or (None, None).

    A supplied audio track pinned upstream arrives this way, and slicing it per
    chunk is the whole reason chunks read their mask rather than building one.
    """
    m = latent.get("noise_mask")
    if m is None:
        return None, None
    if isinstance(m, NestedTensor):
        parts = m.unbind()
        return parts[0], parts[-1]
    if getattr(m, "ndim", 0) == 5:
        return m, None
    return None, m


def _ones_like_mask(v, a):
    vm = torch.ones([v.shape[0], 1] + list(v.shape[2:]), dtype=torch.float32,
                    device=v.device)
    am = None if a is None else torch.ones(
        [a.shape[0], 1, a.shape[2], a.shape[3]], dtype=torch.float32, device=a.device)
    return vm, am


# The DiT reads the mask per 2x2 LATENT patch, so 2 latent cells -- 32 pixels at the
# VAE's /16 -- is the finest feature a mask can express. Snapping each patch to one
# value keeps an edge from landing mid-patch, where the token would otherwise carry a
# partial pooled strength and get its own timestep (rows_t = 1 - m*sigma).
MASK_TOKEN_PATCH = 2

_MASK_REDUCE = {
    "max": lambda g, d: g.amax(dim=d),
    "min": lambda g, d: g.amin(dim=d),
    "mean": lambda g, d: g.mean(dim=d),
    "last": lambda g, d: g.select(d, g.shape[d] - 1),
}


def _token_snap(x, method):
    """One value per 2x2 latent patch, replicate-padded on odd edges."""
    t, h, w = x.shape
    p = MASK_TOKEN_PATCH
    x = torch.nn.functional.pad(x[:, None], (0, -w % p, 0, -h % p), mode="replicate")
    if method == "min":
        x = -torch.nn.functional.max_pool2d(-x, p)
    elif method == "mean":
        x = torch.nn.functional.avg_pool2d(x, p)
    else:
        x = torch.nn.functional.max_pool2d(x, p)
    x = x.repeat_interleave(p, dim=-2).repeat_interleave(p, dim=-1)
    return x[:, 0, :h, :w]


def _mask_to_video_latent(mask, total_t, lat_h, lat_w, mode):
    """MASK batch -> [1,1,total_t,lat_h,lat_w] on the latent grid.

    Spatially pooled rather than interpolated: bilinear averages, which manufactures
    a fractional value along every edge, and on H3 each such cell then denoises at
    its own timestep. Temporally grouped on the VAE's real cycle -- latent k covers
    frames [frame_at_latent(k), frame_at_latent(k+1)) -- so an edge lands where the
    encoder actually put it. A uniform split misplaces it: the first latent of every
    17-frame group covers ONE frame and the other four cover four each.
    """
    m = mask
    if m.ndim == 4:
        m = m[..., 0] if m.shape[-1] == 1 else m.mean(dim=-1)
    if m.ndim == 2:
        m = m[None]
    m = m.to(torch.float32)

    reduce = _MASK_REDUCE.get(mode, _MASK_REDUCE["max"])
    x = m[:, None]
    if mode == "min":
        x = -torch.nn.functional.adaptive_max_pool2d(-x, (lat_h, lat_w))
    elif mode == "mean":
        x = torch.nn.functional.adaptive_avg_pool2d(x, (lat_h, lat_w))
    else:
        x = torch.nn.functional.adaptive_max_pool2d(x, (lat_h, lat_w))
    x = _token_snap(x[:, 0], mode)

    n = int(x.shape[0])
    if n == 1:
        out = x.expand(total_t, -1, -1)
    elif n == total_t:
        out = x
    else:
        px = frame_at_latent(total_t)
        if n != px:
            idx = torch.linspace(0, n - 1, px).round().long().clamp(0, n - 1)
            x = x[idx]
        rows = []
        for k in range(total_t):
            a, b = frame_at_latent(k), frame_at_latent(k + 1)
            a, b = min(a, x.shape[0] - 1), min(max(b, a + 1), x.shape[0])
            rows.append(reduce(x[a:b], 0))
        out = torch.stack(rows)
    return out[None, None].contiguous()


def _video_mask_to_audio(mv, total_t, total_a, mode):
    """Put a mask already on the video latent grid onto the audio grid.

    Video latent k owns audio [_audio_index_at(k), _audio_index_at(k+1)); each span
    takes the spatial reduction of that latent's mask. Boundaries are converted
    independently because audio_t is not additive.

    Called ONLY on `audio_denoise_mask`, which arrives pooled to 1x1, so the spatial
    reduction is a no-op and this is purely the time mapping. Deliberately NOT applied
    to `denoise_mask`: a spatial mask carries no temporal intent, and reducing a
    subject matte this way returns "free" at every timestep.
    """
    reduce = _MASK_REDUCE.get(mode, _MASK_REDUCE["max"])
    per_t = reduce(reduce(mv[0, 0].reshape(total_t, -1), 1)[:, None], 1)
    prof = torch.ones(total_a, dtype=torch.float32, device=mv.device)
    for k in range(total_t):
        a = _audio_index_at(k, total_t, total_a)
        b = _audio_index_at(k + 1, total_t, total_a) if k + 1 < total_t else total_a
        if b > a:
            prof[a:b] = per_t[k]
    return prof


def _audio_profile_to_mask(prof, sub_a_like):
    """[T40] profile -> [B,1,2,T40] on the audio latent's own axes (temporal = dim 3)."""
    return prof.view(1, 1, 1, -1).expand(
        sub_a_like.shape[0], 1, sub_a_like.shape[2], -1).contiguous()


def _sliced_mask(sub_v, sub_a, in_v, in_a, v0, v1, a0, a1):
    """This chunk's slice of whatever mask the master carried."""
    vm, am = _ones_like_mask(sub_v, sub_a)
    if in_v is not None:
        vm = in_v[:, :, v0:v1].to(dtype=vm.dtype, device=vm.device).clone()
    if am is not None and in_a is not None:
        am = in_a[:, :, :, a0:a1].to(dtype=am.dtype, device=am.device).clone()
    return NestedTensor([vm, am]) if am is not None else vm


def _carry_mask(sub_v, sub_a, carried, carried_a, strength_v, strength_a,
                in_v, in_a, v0, v1, a0, a1):
    """Preserve the carried head, honouring any mask the master already had.

    Starts from the master's own mask for this span -- a pinned audio track has
    to survive -- then pins the carry on top.

    There is deliberately NO feather. A ramp of intermediate mask values makes the
    seam NOISY on a core with the rebased #15375: each ramped cell gets its own
    timestep (rows_t = 1 - m*sigma) while the sampler blends its content as
    the two are now reconciled by core: scale_latent_inpaint pre-compensates so
    every pixel lands at its token's pooled strength. The feather was removed in
    0.73.0 on the evidence, and that stands -- but the mechanism recorded for it
    did not survive re-reading (see CHANGELOG 0.76.0). Observed 2026-08-13; the
    `feather_latents` input was removed in 0.73.0 rather than left as a trap.
    """
    packed = _sliced_mask(sub_v, sub_a, in_v, in_a, v0, v1, a0, a1)
    if isinstance(packed, NestedTensor):
        vm, am = packed.unbind()
    else:
        vm, am = packed, None

    vm[:, :, :carried] = 1.0 - strength_v
    if am is not None and carried_a > 0:
        am[:, :, :, :carried_a] = 1.0 - strength_a
    return NestedTensor([vm, am]) if am is not None else vm


class MMH3KeyframePlanner(io.ComfyNode):
    """End-anchored keyframe indices for a chained run.

    TRAVEL SEMANTICS, from LTXAVTools' planner. The first keyframe (optional)
    opens the clip at frame 0; every later one sits at the END of its chunk, so
    each chunk generates TOWARD its destination image and the next continues
    from the arrived state through the ordinary carry. Start-anchoring instead
    would put the image in the NEXT chunk and invite a snap at every seam. The
    final chunk's end is the clip's end, emitted as -1.

    Takes the same three numbers the sampler does and derives the same schedule
    from `_plan`, so the two cannot disagree about where a chunk ends.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3KeyframePlanner",
            display_name="MiniMax H3 Keyframe Planner",
            category="MMH3Tools/calculators",
            description=(
                "End-anchored keyframe indices: frame 0 opens, each chunk travels to a "
                "keyframe at its end, the last ends on -1. Wire `indices` to the Looping "
                "Sampler's keyframe_indices and `count` tells you how many images its "
                "`keyframes` batch needs, in that order."
            ),
            inputs=[
                io.Int.Input("total_frames", default=192, min=5, max=3600, step=17,
                             tooltip="The whole clip, same as the sampler's latent."),
                io.Int.Input("chunk_frames", default=192, min=5, max=3600, step=17,
                             tooltip="Same value the sampler gets."),
                io.Int.Input("overlap_frames", default=22, min=0, max=3600, step=17,
                             tooltip="Same value the sampler gets. It changes where each "
                                     "chunk ends, so it changes every index."),
                io.Boolean.Input("include_start", default=True,
                                 tooltip="A keyframe at frame 0 -- the opening image."),
                io.Boolean.Input("include_end", default=True,
                                 tooltip="A keyframe at -1 -- the closing image."),
            ],
            outputs=[
                io.String.Output(display_name="indices"),
                io.Int.Output(display_name="count"),
                io.Int.Output(display_name="chunk_count"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, total_frames, chunk_frames, overlap_frames,
                include_start, include_end) -> io.NodeOutput:
        _length, _ov, total_f, _t, windows = _plan(
            int(total_frames), int(chunk_frames), int(overlap_frames),
            "standard_static")
        spans = _window_frame_spans(windows, total_f)

        # A chunk's travel destination is the last frame IT renders, not the last
        # frame it spans. The final chunk is clamped to end on the clip, so it
        # overlaps its predecessor by far more than the nominal overlap -- and a
        # frame at that predecessor's span end is then inside the last chunk's NEW
        # content, which claims it. Using span ends left the second-to-last chunk
        # with no destination and gave the last one two.
        ov_f = frame_at_latent(_ov)
        dests = []
        for i in range(len(spans) - 1):
            d = min(spans[i + 1][0] + ov_f - 1, spans[i][1])
            dests.append(max(d, spans[i][0]))

        entries = ([0] if include_start else []) + dests
        if include_end:
            entries.append(-1)

        idx = ", ".join(str(e) for e in entries)
        lines = ["%d chunk%s over %d frames (%.2fs), %d keyframe%s"
                 % (len(spans), "" if len(spans) == 1 else "s", total_f,
                    total_f / float(FPS), len(entries),
                    "" if len(entries) == 1 else "s")]
        for e in entries:
            g = total_f - 1 if e < 0 else e
            lines.append("  frame %-6s %6.2fs%s"
                         % (e, g / float(FPS), "   (end of clip)" if e < 0 else ""))
        if not entries:
            lines.append("  ! nothing to place -- one chunk with both ends off")
        report = "\n".join(lines)
        logging.info("[MMH3KeyframePlanner] " + lines[0])
        return io.NodeOutput(idx, len(entries), len(spans), report)
