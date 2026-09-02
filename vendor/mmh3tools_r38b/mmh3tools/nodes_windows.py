"""Context windows for MiniMax H3, without patching core.

ComfyUI's context windowing already has a complete multimodal design and H3 opts
into it for free -- `is_multimodal = len(latent_shapes) > 1` is true for any packed
AV latent. Two things stop it working out of the box:

  1. `map_context_window_to_modalities` has ZERO implementations tree-wide. The
     multimodal path raises NotImplementedError for every model.
  2. `WindowingState` uses ONE `dim` for every modality. H3's video is dim 2 and
     audio is dim 3, so the stock path would window audio [B,32,2,T40] on its
     STEREO axis -- size 2, not T40. It would not crash; it would produce a ratio
     of 2/T and nonsense indices.

Neither needs a core edit. The handler is just an object in a dict
(`model.model_options["context_handler"]`, read back in samplers.py), so a subclass
works. Overriding `prepare_window()` means the unimplemented model hook is never
called at all. That matters: this survives `git pull`, and when upstream refactors
it fails loudly with an AttributeError rather than silently doing the wrong thing,
which is exactly what a stale diff does.

INTENDED USE: low-denoise upscale passes. At low denoise every window starts from the
same upscaled base, so coherence comes from the input rather than from attention
spanning the clip. Attach this on stages 2 and 3 of an upscale ladder.

FULL DENOISE needs `freenoise`. Without it each window is an independent generation
from independent noise, and the overlap blend averages two different images -- hard
jitter at every seam. FreeNoise copies each window's noise forward into the next
window's region, permuted, so overlapping windows start from RELATED noise. That is
how Wan windows at full denoise with drift rather than seams, and Wan's own node
defaults it on. It is off here because the upscale passes this was built for do not
want it.

Separately, H3 needs its window STARTS on phase 0 (index % 5 == 0), because its
temporal grid is FRAME_PER_TOKEN = (1,4,4,4,4) indexed by k % 5 -- every fifth latent
spans ONE frame and the rest span four. A window is read as starting at k=0, so an
off-phase start assigns every latent the wrong span, differently per phase. Wan's grid
is uniform after latent 0, so a shifted window is a translation and the error is
constant, which shows as drift; H3's is a re-phasing, which shows as jitter. The 5j+2
length and 5m+2 overlap rules keep every start on phase 0.
"""

import dataclasses
import logging

import torch

import comfy.patcher_extension
import comfy.utils
from comfy.context_windows import (
    ContextFuseMethods,
    ContextSchedules,
    IndexListCallbacks,
    IndexListContextHandler,
    IndexListContextWindow,
    WindowingState,
    apply_freenoise,
    create_prepare_sampling_wrapper,
    create_sampler_sample_wrapper,
    get_context_weights,
    get_matching_context_schedule,
    get_matching_fuse_method,
    get_shape_for_dim,
    match_weights_to_dim,
)
from comfy_api.latest import io

from .common import (
    AUDIO_LATENT_FPS,
    AUDIO_T_DIM,
    FPS,
    FRAMES_PER_GROUP,
    FRAME_BASE,
    LATENTS_PER_GROUP,
    LATENT_BASE,
    VIDEO_T_DIM,
    frame_at_latent,
    frames_to_latents,
    latents_to_frames,
    snap_frames,
)


def _audio_index_at(n, total_v, total_a):
    """Audio-latent index at video-latent boundary `n`.

    Uses `frame_at_latent`, which walks the VAE's real FRAME_PER_TOKEN cycle
    (1,4,4,4,4) and is therefore EXACT at every index, not only on the 5j+2 grid.
    Then audio_t = frames/24*40.

    It previously inverted the grid as frames(n) = 5 + 17*(n-2)/5 -- a LINEAR
    interpolation of a NON-UNIFORM mapping, since the first latent of each 17-frame
    cycle covers ONE frame and the other four cover four each. That agreed only on
    the grid and drifted up to 1.8 frames between grid points, which is 3 audio
    latents, 75 ms.

    Callers happened to be safe: window and overlap latents are both snapped to
    5j+2, so the stride is a multiple of 5 and `v0 + carried` always landed back on
    the grid. The audio carry boundary was correct by arithmetic coincidence rather
    than by construction, and any off-grid caller -- a context_schedule that does not
    snap, a hand-set overlap -- would have shifted it 75 ms with nothing raised.

    Boundaries are converted independently and subtracted rather than converting a
    window LENGTH, because audio_t = round(frames/24*40) is not additive -- the same
    correction MMH3ConcatAV needed.
    """
    if total_v <= 0 or total_a <= 0:
        return 0
    if n <= 0:
        return 0
    if n >= total_v:
        return total_a
    idx = int(round(frame_at_latent(n) / FPS * AUDIO_LATENT_FPS))
    return max(0, min(total_a, idx))


class MMH3WindowingState(WindowingState):
    """WindowingState that gives each modality its OWN temporal dim."""

    def prepare_window(self, window, model):
        if not self.is_multimodal or len(self.latents) < 2:
            return window

        video, audio = self.latents[0], self.latents[1]
        total_v = int(video.shape[VIDEO_T_DIM])
        total_a = int(audio.shape[AUDIO_T_DIM])
        idx = list(window.index_list)
        if not idx or total_v <= 0 or total_a <= 0:
            return window

        # contiguous span; looped/wrapping schedules are rejected by the node
        a0 = _audio_index_at(idx[0], total_v, total_a)
        a1 = _audio_index_at(idx[-1] + 1, total_v, total_a)
        if a1 <= a0:
            a1 = min(total_a, a0 + 1)
        audio_indices = list(range(a0, a1))

        ratio = total_a / float(total_v)
        audio_window = IndexListContextWindow(
            audio_indices, dim=AUDIO_T_DIM, total_frames=total_a,
            context_overlap=max(0, int(round(window.context_overlap * ratio))))

        return IndexListContextWindow(
            idx, dim=VIDEO_T_DIM, total_frames=total_v,
            modality_windows={1: audio_window},
            context_overlap=window.context_overlap)


def _mod_dim(mod_idx):
    """Temporal dim per modality: video is 2, audio is 3."""
    return VIDEO_T_DIM if mod_idx == 0 else AUDIO_T_DIM


def _resize_denoise_mask(cond_key, cond_value, window, x_in, device, cond_item):
    """Slice a denoise mask to the window, per modality.

    Core resizes `model_conds` entries only when they are raw tensors, plus hand-written
    cases for `audio_embed` and `vace_context`. A denoise mask is a CONDRegular, so it
    falls through UNWINDOWED: the model then receives a full-length mask against a
    windowed latent and dies in `_mod_scale_shift` with

        The size of tensor a (640) must match the size of tensor b (866)

    -- the mod-row weight vector sized for the whole clip, applied to one window's rows.

    LTXAV solves this by overriding `resize_cond_for_context_window` on the model class.
    MiniMaxH3 has no such override, and adding one would mean patching core. This does it
    from the handler's own RESIZE_COND_ITEM callback instead, which needs no core change
    and dies loudly if the hook is ever removed.

    Each modality slices on ITS OWN axis: video masks are [B,1,T,h,w] on dim 2, audio
    masks [B,1,2,T40] on dim 3. Using one dim for both would slice audio on its stereo
    axis, which is the same trap MMH3WindowingState exists to avoid.
    """
    if cond_key not in ("denoise_mask", "audio_denoise_mask"):
        return None
    tensor = getattr(cond_value, "cond", None)
    if not isinstance(tensor, torch.Tensor):
        return None

    if cond_key == "denoise_mask":
        target = window
    else:
        target = window.get_window_for_modality(1) if getattr(
            window, "modality_windows", None) else None
        if target is None:
            return None

    # already the right length -- a mask built for this window, or a single frame
    if tensor.shape[target.dim] == len(target.index_list):
        return None
    if tensor.shape[target.dim] != target.total_frames:
        # not the shape we know how to cut; leave it alone rather than guess
        return None
    return cond_value._copy_with(target.get_tensor(tensor, device))


class MMH3ContextHandler(IndexListContextHandler):
    def __init__(self, *args, **kwargs):
        # not an upstream parameter -- pop before super() sees it
        self.accumulator_device = kwargs.pop("accumulator_device", "gpu")
        super().__init__(*args, **kwargs)
        # windowing a MASKED latent is otherwise a hard crash; see _resize_denoise_mask
        comfy.patcher_extension.add_callback_with_key(
            IndexListCallbacks.RESIZE_COND_ITEM, "mmh3_denoise_mask",
            _resize_denoise_mask, self.callbacks)

    def combine_context_window_results(self, x_in, sub_conds_out, sub_conds, window,
                                       window_idx, total_windows, timestep,
                                       conds_final, counts_final, biases_final):
        """Upstream's method with `self.dim` replaced by the WINDOW's dim.

        Upstream builds the fuse weights on the handler's dim, so for audio it sizes a
        93-long weight vector onto dim 2 -- the stereo axis -- and dies with
        "size of tensor a (2) must match the size of tensor b (93)". `window` here is
        already the per-modality window, so its own dim is the right one.
        """
        dim = getattr(window, "dim", self.dim)
        if self.fuse_method.name == ContextFuseMethods.RELATIVE:
            for pos, idx in enumerate(window.index_list):
                bias = 1 - abs(idx - (window.index_list[0] + window.index_list[-1]) / 2) / (
                    (window.index_list[-1] - window.index_list[0] + 1e-2) / 2)
                bias = max(1e-2, bias)
                for i in range(len(sub_conds_out)):
                    if conds_final[i] is None:  # cfg 1.0: uncond has no accumulator
                        continue
                    bias_total = biases_final[i][idx]
                    prev_weight = bias_total / (bias_total + bias)
                    new_weight = bias / (bias_total + bias)
                    idx_window = tuple([slice(None)] * dim + [idx])
                    pos_window = tuple([slice(None)] * dim + [pos])
                    # .to() is a no-op on the gpu path; on the cpu path it moves one
                    # window-slice, which is the whole point of hosting accum off-GPU
                    conds_final[i][idx_window] = (
                        conds_final[i][idx_window] * prev_weight
                        + sub_conds_out[i][pos_window].to(conds_final[i].device) * new_weight)
                    biases_final[i][idx] = bias_total + bias
        else:
            weights = get_context_weights(window.context_length, x_in.shape[dim],
                                          window.index_list, self, sigma=timestep,
                                          context_overlap=window.context_overlap)
            weights_tensor = match_weights_to_dim(weights, x_in, dim, device=x_in.device)
            for i in range(len(sub_conds_out)):
                if conds_final[i] is None:  # cfg 1.0: uncond has no accumulator
                    continue
                dev = conds_final[i].device
                # weighting happens on the GPU where the window output already is;
                # only the finished window-sized product crosses to the accumulator
                window.add_window(conds_final[i], (sub_conds_out[i] * weights_tensor).to(dev))
                window.add_window(counts_final[i], weights_tensor.to(dev))

        for callback in comfy.patcher_extension.get_all_callbacks(
                IndexListCallbacks.COMBINE_CONTEXT_WINDOW_RESULTS, self.callbacks):
            callback(self, x_in, sub_conds_out, sub_conds, window, window_idx,
                     total_windows, timestep, conds_final, counts_final, biases_final)

    def _alloc_accumulators(self, latents, conds):
        """counts/biases sized on each modality's OWN temporal dim.

        Upstream allocates both with `self.dim`, so audio [B,32,2,T40] gets a counts
        tensor of extent 2 (stereo) instead of T40, and a biases list of length 2.

        Two further allocation changes against upstream, both for VRAM held DURING
        the window loop -- the phase where activation peaks live:

        - A None cond gets NO accumulator. cfg 1.0 skips the uncond but still passes
          it as a list entry, and upstream holds a full-length zeros tensor through
          the whole loop for a cond no window ever writes. The zeros are materialized
          at fuse time instead (see execute), after the loop's activations are freed.
          Same tensor returned, allocated later.

        - accumulator_device="cpu" hosts the real accumulators in system RAM. Every
          write to them is a window-sized slice, so the loop pays one window-sized
          transfer per window per cond; the fused result moves back to the GPU once
          per step, after the loop. Values are identical to the gpu path -- the same
          ops run, on the same numbers, on a different device.
        """
        if isinstance(conds, int):  # older callers passed a count
            conds = [object()] * conds
        dev = torch.device("cpu") if getattr(self, "accumulator_device", "gpu") == "cpu" \
            else None
        fill = torch.ones if self.fuse_method.name == ContextFuseMethods.RELATIVE else torch.zeros
        accum, counts, biases = [], [], []
        for i, m in enumerate(latents):
            d = _mod_dim(i)
            accum.append([None if c is None else
                          torch.zeros_like(m, device=dev or m.device) for c in conds])
            counts.append([None if c is None else
                           fill(get_shape_for_dim(m, d), device=dev or m.device)
                           for c in conds])
            biases.append([[0.0] * m.shape[d] for _ in conds])
        return accum, counts, biases

    def _build_window_state(self, x_in, conds, model):
        st = super()._build_window_state(x_in, conds, model)
        return MMH3WindowingState(
            **{f.name: getattr(st, f.name) for f in dataclasses.fields(st)})

    def execute(self, calc_cond_batch, model, conds, x_in, timestep, model_options):
        """Upstream's execute, with the accumulators allocated per modality.

        Copied rather than wrapped because the allocation is inline. The two changed
        lines are marked; everything else is upstream's. If upstream refactors this,
        it breaks loudly here rather than quietly windowing the wrong axis.
        """
        self._model = model
        self.set_step(timestep, model_options)

        window_state = self._build_window_state(x_in, conds, model)
        num_modalities = len(window_state.latents)

        context_windows = self.get_context_windows(model, window_state.latents[0], model_options)
        enumerated_context_windows = list(enumerate(context_windows))
        total_windows = len(enumerated_context_windows)

        # CHANGED: per-modality dims instead of self.dim for counts/biases; conds
        # passed whole so None entries (cfg 1.0 uncond) allocate nothing
        accum, counts, biases = self._alloc_accumulators(window_state.latents, conds)

        for callback in comfy.patcher_extension.get_all_callbacks(
                IndexListCallbacks.EXECUTE_START, self.callbacks):
            callback(self, model, x_in, conds, timestep, model_options)

        for enum_window in enumerated_context_windows:
            results = self.evaluate_context_windows(
                calc_cond_batch, model, x_in, conds, timestep, [enum_window],
                model_options, window_state=window_state, total_windows=total_windows)
            for result in results:
                for mod_idx in range(num_modalities):
                    mod_out = [result.sub_conds_out[ci][mod_idx] for ci in range(len(conds))]
                    modality_window = result.window.get_window_for_modality(mod_idx)
                    self.combine_context_window_results(
                        window_state.latents[mod_idx], mod_out, result.sub_conds, modality_window,
                        result.window_idx, total_windows, timestep,
                        accum[mod_idx], counts[mod_idx], biases[mod_idx])

        try:
            result_out = []
            for ci in range(len(conds)):
                finalized = []
                for mod_idx in range(num_modalities):
                    dev = window_state.latents[mod_idx].device
                    a = accum[mod_idx][ci]
                    if a is None:
                        # cfg 1.0: the uncond was never evaluated. Materialize its
                        # zeros only now, after the window loop has freed its
                        # activations -- the caller gets the same tensor upstream
                        # would have returned, allocated a loop later.
                        f = torch.zeros_like(window_state.latents[mod_idx])
                    else:
                        if self.fuse_method.name != ContextFuseMethods.RELATIVE:
                            a /= counts[mod_idx][ci]
                        # no-op on the gpu path; brings a cpu-hosted accumulator
                        # back to the GPU once per step, after the loop
                        f = a.to(dev)
                    if window_state.guide_latents[mod_idx] is not None:
                        # CHANGED: guide frames concat on the modality's own dim
                        f = torch.cat([f, window_state.guide_latents[mod_idx]],
                                      dim=_mod_dim(mod_idx))
                    finalized.append(f)

                if window_state.is_multimodal and len(finalized) > 1:
                    packed, _ = comfy.utils.pack_latents(finalized)
                else:
                    packed = finalized[0]
                result_out.append(packed)
            return result_out
        finally:
            for callback in comfy.patcher_extension.get_all_callbacks(
                    IndexListCallbacks.EXECUTE_CLEANUP, self.callbacks):
                callback(self, model, x_in, conds, timestep, model_options)

    def _apply_freenoise(self, noise, conds, seed):
        """Shuffle the VIDEO noise only, on its own dim.

        FreeNoise copies each window's noise forward into the next window's region,
        permuted, so overlapping windows start from related noise instead of
        independent noise. That is what lets Wan window at FULL denoise with drift
        rather than hard seams -- without it, each window is an independent
        generation and the overlap blend averages two different images.

        Stock's multimodal path shuffles every modality on the PRIMARY dim. For audio
        [B,32,2,T40] that is the stereo axis: with ratio 2/57 it computes a context
        length of 1 and permutes the left channel into the right. Audio is left alone
        here instead -- on an upscale pass the sampler's audio is discarded anyway,
        and shuffling a 40/sec stream against a ~7/sec video window is meaningless.
        """
        shapes = self._get_latent_shapes(conds)
        if shapes is None or len(shapes) < 2:
            return super()._apply_freenoise(noise, conds, seed)

        mods = list(comfy.utils.unpack_latents(noise, shapes))
        apply_freenoise(mods[0], VIDEO_T_DIM, self.context_length,
                        self.context_overlap, seed)
        logging.info("[MMH3ContextWindows] freenoise applied to video on dim %d "
                     "(%d latents, window %d, overlap %d); audio untouched",
                     VIDEO_T_DIM, int(mods[0].shape[VIDEO_T_DIM]),
                     self.context_length, self.context_overlap)
        out, _ = comfy.utils.pack_latents(mods)
        return out


def _snap_grid(n):
    """Snap DOWN to the 5j+2 latent grid, minimum 2."""
    n = int(n)
    if n < LATENT_BASE:
        return LATENT_BASE
    return LATENTS_PER_GROUP * ((n - LATENT_BASE) // LATENTS_PER_GROUP) + LATENT_BASE


class MMH3ContextWindows(io.ComfyNode):
    """Sliding-window sampling over the video latent, with audio windowed correctly."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3ContextWindows",
            display_name="MiniMax H3 Context Windows",
            category="MMH3Tools/sampling",
            description=(
                "Sample a long AV latent in overlapping windows. FOR LOW-DENOISE PASSES "
                "ONLY -- at full denoise each window invents its own content and they "
                "disagree. Windows snap to the 5j+2 grid; audio is windowed on its own "
                "temporal axis."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Int.Input(
                    "context_length", default=17, min=7, max=512, step=5,
                    tooltip="Window size in VIDEO LATENTS, snapped down to 5j+2 (7, 12, 17, "
                            "22...). 17 latents is 58 frames, ~2.4s. The model only ever saw "
                            "5j+2 clip lengths, so an off-grid window is off-distribution.\n\n"
                            "THIS is the VRAM lever: peak activation cost scales with the "
                            "window, not the clip. At 2048x1152 a 17-latent window is ~39k "
                            "tokens against ~131k for the whole 192-frame clip, and attention "
                            "is quadratic. Drop to 12 or 7 if you are tight -- but for MEMORY only: "
                            "smaller windows are not faster, since more of them exactly cancels "
                            "the smaller square while the linear cost keeps rising. It does not "
                            "reduce WEIGHT memory, so it does not make H3 loadable on a card "
                            "that could not already load it.",
                ),
                io.Int.Input(
                    "context_overlap", default=7, min=0, max=256, step=5,
                    tooltip="Overlap in video latents, snapped to 5m+2 (2, 7, 12, 17...). "
                            "NOT a multiple of 5: stride is length-overlap, and only 5m+2 "
                            "keeps every window at the same phase against H3's 2+5k latent "
                            "groups. A multiple of 5 makes the phase cycle every five "
                            "windows, which shows up as pulsing.\n\n"
                            "This changes how MANY windows run, not the cost of each, so it "
                            "trades time and seam quality -- not VRAM. For memory, use "
                            "context_length.",
                ),
                io.Combo.Input(
                    "fuse_method", options=ContextFuseMethods.LIST_STATIC,
                    default=ContextFuseMethods.PYRAMID,
                    tooltip="How overlapping windows are blended. Pyramid weights each "
                            "window's centre most and tapers to its edges; flat weights "
                            "every position equally.",
                ),
                io.Combo.Input(
                    "context_schedule",
                    options=[ContextSchedules.STATIC_STANDARD, ContextSchedules.UNIFORM_STANDARD],
                    default=ContextSchedules.STATIC_STANDARD,
                    tooltip="Looped and batched schedules are not offered: they can emit "
                            "non-contiguous or wrapping windows, which the audio mapping "
                            "cannot express as a time span.",
                ),
                io.Int.Input("context_stride", default=1, min=1, max=32, step=1,
                             tooltip="Uniform schedules only."),
                io.Boolean.Input(
                    "freenoise", default=False, optional=True,
                    tooltip="Copy each window's noise forward into the next window's "
                            "region, permuted, so overlapping windows start from RELATED "
                            "noise instead of independent noise. "
                            "Leave OFF for low-denoise upscale passes -- there is little "
                            "noise left to shuffle and the input already supplies "
                            "coherence. Turn ON for FULL denoise: without it each window "
                            "is an independent generation and the overlap blend averages "
                            "two different images, which is the hard jitter at seams. "
                            "Wan's own context-window node defaults this ON. "
                            "Video only; audio noise is left alone.",
                ),
                io.Boolean.Input(
                    "split_conds_to_windows", default=False, optional=True,
                    tooltip="Give each window the prompt for ITS region of the timeline "
                            "instead of the whole script. Core picks by the window's "
                            "midpoint: region = int(center_ratio * number_of_prompts), so "
                            "prompt 0 covers the start and the last covers the end. "
                            "Needs a conditioning holding MORE THAN ONE entry -- wire "
                            "MMH3 Cond Set Spread; with a single prompt this does nothing. "
                            "Without it every window renders the same instructions, which "
                            "is why a windowed pass can look like it is attempting the "
                            "entire script over and over.",
                ),
                io.Combo.Input(
                    "accumulator_device", options=["gpu", "cpu"], default="gpu",
                    optional=True,
                    tooltip="Where the per-step fuse accumulators live. gpu is stock "
                            "behaviour. cpu keeps the full-length accumulation buffers "
                            "in system RAM: each window writes its slice across PCIe "
                            "during the loop, and the fused result moves back to the "
                            "GPU once per step after the loop. Frees one full-length "
                            "fp32 latent of VRAM per evaluated cond for the duration "
                            "of the window loop. Values are identical either way.",
                ),
            ],
            outputs=[io.Model.Output(display_name="model"),
                     io.String.Output(display_name="label")],
        )

    @classmethod
    def execute(cls, model, context_length, context_overlap, fuse_method,
                context_schedule, context_stride, freenoise=False,
                split_conds_to_windows=False, accumulator_device="gpu") -> io.NodeOutput:
        length = _snap_grid(context_length)
        # Overlap must be 5m+2, NOT a multiple of 5. Stride is length - overlap, and
        # H3's latent groups start at 2+5k, so the window phase is what matters:
        #   overlap 5m   -> stride 5(j-m)+2 -> phase advances 2 every window,
        #                   cycling 0,2,4,1,3 -- a five-window beat, seen as pulsing
        #   overlap 5m+2 -> stride 5(j-m)   -> every window at the same phase
        overlap = _snap_grid(context_overlap) if context_overlap >= LATENT_BASE else 0
        overlap = min(overlap, max(0, length - LATENTS_PER_GROUP))

        notes = []
        if length != int(context_length):
            notes.append("context_length %d -> %d (5j+2 grid)" % (int(context_length), length))
        if overlap != int(context_overlap):
            notes.append("context_overlap %d -> %d" % (int(context_overlap), overlap))

        m = model.clone()
        m.model_options["context_handler"] = MMH3ContextHandler(
            context_schedule=get_matching_context_schedule(context_schedule),
            fuse_method=get_matching_fuse_method(fuse_method),
            context_length=length,
            context_overlap=overlap,
            context_stride=context_stride,
            closed_loop=False,
            dim=VIDEO_T_DIM,
            freenoise=bool(freenoise),
            # prepends an anchor frame to every non-zero window, which would push
            # each one to 5j+3 latents -- off the only grid the model has seen
            causal_window_fix=False,
            split_conds_to_windows=bool(split_conds_to_windows),
            accumulator_device=accumulator_device,
        )
        create_prepare_sampling_wrapper(m)
        if freenoise:
            # stock only installs this wrapper when freenoise is on
            create_sampler_sample_wrapper(m)

        frames = FRAMES_PER_GROUP * ((length - LATENT_BASE) // LATENTS_PER_GROUP) + FRAME_BASE
        ov_frames = FRAMES_PER_GROUP * (overlap // LATENTS_PER_GROUP)
        label = ("window %d latents (%d frames, %.2fs), overlap %d (%d frames), "
                 "freenoise %s, split conds %s%s"
                 % (length, frames, frames / float(FPS), overlap, ov_frames,
                    "ON" if freenoise else "off",
                    "ON" if split_conds_to_windows else "off",
                    ", accumulators on CPU" if accumulator_device == "cpu" else ""))
        for n in notes:
            label += "\n  ! " + n
            logging.info("[MMH3ContextWindows] %s", n)
        logging.info("[MMH3ContextWindows] " + label.splitlines()[0])
        return io.NodeOutput(m, label)


def _plan(total_frames, window_frames, overlap_frames, context_schedule):
    """Resolve a windowing request to the schedule sampling will actually run.

    Shared by MMH3WindowPlan and MMH3SplitAudioToWindows so they cannot disagree.
    If the splitter's spans drifted from the planner's, the LLM would be writing each
    prompt against audio that window never renders -- a failure that would look like
    the model ignoring the prompt.

    The window list comes from core's own scheduler rather than reimplemented stride
    arithmetic, for the same reason.
    """
    total_f = snap_frames(int(total_frames))
    total_t = frames_to_latents(total_f)

    length = _snap_grid(frames_to_latents(int(window_frames)))
    length = min(length, total_t)
    ov_req = frames_to_latents(int(overlap_frames)) if int(overlap_frames) > 0 else 0
    overlap = _snap_grid(ov_req) if ov_req >= LATENT_BASE else 0
    overlap = min(overlap, max(0, length - LATENTS_PER_GROUP))

    handler = MMH3ContextHandler(
        context_schedule=get_matching_context_schedule(context_schedule),
        fuse_method=get_matching_fuse_method("pyramid"),
        context_length=length, context_overlap=overlap, context_stride=1,
        closed_loop=False, dim=VIDEO_T_DIM, freenoise=False, causal_window_fix=False)
    windows = handler.get_context_windows(None, torch.zeros([1, 1, total_t, 2, 2]), {})
    return length, overlap, total_f, total_t, windows


def _window_frame_spans(windows, total_f):
    """(first_frame, last_frame) per window, inclusive, clamped to the clip."""
    out = []
    for w in windows:
        a, b = w.index_list[0], w.index_list[-1]
        out.append((min(frame_at_latent(a), total_f - 1),
                    min(frame_at_latent(b + 1) - 1, total_f - 1)))
    return out


class MMH3WindowPlan(io.ComfyNode):
    """Work out the whole windowing schedule up front, in frames, and emit it.

    Three things were previously only knowable by running a generation:

      * whether your window and overlap survive snapping. Both must land on the 5j+2
        latent grid, and an overlap that is a multiple of 5 rather than 5m+2 walks the
        window phase 0,2,4,1,3 -- a five-window beat, which is the pulsing.

      * HOW MANY WINDOWS you get. That is the number of prompts to write for
        split_conds_to_windows, because regions are cut per window midpoint. Guess low
        and several windows share a prompt; guess high and the last prompts are never
        reached.

      * which frames each window actually covers, so a prompt can describe the right
        part of the clip.

    The count comes from asking core's own scheduler rather than reimplementing it, so
    it cannot drift from what sampling will really do.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3WindowPlan",
            display_name="MMH3 Window Plan",
            category="MMH3Tools/calculators",
            description=(
                "Plan a windowed pass in frames and emit the latent values the chain "
                "needs: context_length, context_overlap, the snapped frame count, and "
                "how many windows -- which is how many prompts to write."
            ),
            inputs=[
                io.Int.Input(
                    "total_frames", default=192, min=5, max=3600, step=17,
                    tooltip="Length of the whole clip. Wire MMH3 Frame Calculator's "
                            "frame_count. Snapped UP to the 17j+5 grid.",
                ),
                io.Int.Input(
                    "window_frames", default=124, min=5, max=3600, step=17,
                    tooltip="How much of the clip each window sees. Snapped to what the "
                            "5j+2 latent grid can express. Bigger windows cost quadratically "
                            "more attention; smaller ones drift further apart.",
                ),
                io.Int.Input(
                    "overlap_frames", default=22, min=0, max=3600, step=17,
                    tooltip="Shared frames between neighbouring windows. Snapped so the "
                            "LATENT overlap is 5m+2, which keeps every window at the same "
                            "grid phase -- any other value pulses.",
                ),
                io.Combo.Input(
                    "context_schedule", options=["standard_static", "standard_uniform"],
                    default="standard_static",
                    tooltip="Must match the MMH3 Context Windows node, or the count is wrong.",
                ),
                io.Int.Input(
                    "prompt_count", default=0, min=0, max=32, step=1, optional=True,
                    tooltip="If set, the report shows which prompt each window would use "
                            "under split_conds_to_windows, so you can see whether any "
                            "prompt is unreachable or doubled up. 0 skips it.",
                ),
            ],
            outputs=[
                # units in the name: these two are LATENTS and their frame-domain
                # twins sit five sockets below, which is how they get crossed
                io.Int.Output(display_name="context_length (latents)"),
                io.Int.Output(display_name="context_overlap (latents)"),
                io.Int.Output(display_name="window_count"),
                io.Int.Output(display_name="total_frames (frames)"),
                io.Int.Output(display_name="total_latents (latents)"),
                io.String.Output(display_name="report"),
                # APPENDED LAST -- outputs serialise positionally too.
                # Under windowing the layout is rebuilt per window from the WINDOW's
                # latent_t, so anything that takes a target_frame_count (the keyframe
                # nodes) needs this, not total_frames. minimax_frame_count is NOT
                # patched per window, so passing the clip length puts a last-frame
                # anchor at the end of every window instead of the end of the clip.
                io.Int.Output(display_name="window_frames (frames)"),
                # frame_at_latent, NOT latents_to_frames: an overlap of 0 is legal and
                # 0 is off the 5j+2 grid, so the latter floors to the group below and
                # emits -12 frames. frame_at_latent is the general form and gives 0.
                # context_length / context_overlap are LATENTS, for MMH3ContextWindows.
                # MMH3SplitAudioToWindows takes FRAMES. Wiring context_overlap into it
                # silently re-snaps a latent count as a frame count and the splitter's
                # schedule stops matching this plan -- which reads as the model ignoring
                # the prompt, since each window then describes audio it never renders.
                io.Int.Output(display_name="overlap_frames (frames)"),
            ],
        )

    @classmethod
    def execute(cls, total_frames, window_frames, overlap_frames, context_schedule,
                prompt_count=0) -> io.NodeOutput:
        length, overlap, total_f, total_t, windows = _plan(
            total_frames, window_frames, overlap_frames, context_schedule)

        notes = []
        if total_f != int(total_frames):
            notes.append("total %d -> %d frames (17j+5)" % (int(total_frames), total_f))
        if latents_to_frames(length) != int(window_frames):
            notes.append("window %d -> %d frames (%d latents)"
                         % (int(window_frames), latents_to_frames(length), length))
        if latents_to_frames(overlap) != int(overlap_frames) and overlap > 0:
            notes.append("overlap %d -> %d frames (%d latents)"
                         % (int(overlap_frames), latents_to_frames(overlap), overlap))
        if overlap == 0 and int(overlap_frames) > 0:
            notes.append("overlap collapsed to 0 -- the window is too short to keep any")
        if length >= total_t:
            notes.append("the window covers the whole clip, so there is only one window "
                         "and windowing does nothing")

        lines = []
        spans = _window_frame_spans(windows, total_f)
        for i, (w, (fa, fb)) in enumerate(zip(windows, spans)):
            a, b = w.index_list[0], w.index_list[-1]
            row = "  %2d  latents %3d-%-3d  frames %4d-%-4d  %5.2fs-%.2fs" % (
                i, a, b, fa, fb, fa / float(FPS), fb / float(FPS))
            if prompt_count > 0:
                row += "  -> prompt %d" % w.get_region_index(int(prompt_count))
            lines.append(row)

        if prompt_count > 0:
            used = {w.get_region_index(int(prompt_count)) for w in windows}
            missing = sorted(set(range(int(prompt_count))) - used)
            if missing:
                notes.append("prompt%s %s never used -- fewer windows than prompts"
                             % ("" if len(missing) == 1 else "s",
                                ", ".join(str(x) for x in missing)))
            if len(windows) > int(prompt_count):
                notes.append("%d windows share %d prompts, so some span two regions"
                             % (len(windows), int(prompt_count)))

        report = ("%d window%s over %d frames (%.2fs), window %d latents / %d frames, "
                  "overlap %d latents / %d frames\n%s"
                  % (len(windows), "" if len(windows) == 1 else "s", total_f,
                     total_f / float(FPS), length, latents_to_frames(length),
                     overlap, latents_to_frames(overlap), "\n".join(lines)))
        for n in notes:
            report += "\n  ! " + n
        logging.info("[MMH3WindowPlan] " + report.splitlines()[0])
        return io.NodeOutput(length, overlap, len(windows), total_f, total_t, report,
                             latents_to_frames(length), frame_at_latent(overlap))


def _timecode(frames):
    """mm:ss.d for a frame index at FPS."""
    s = frames / float(FPS)
    return "%02d:%04.1f" % (int(s // 60), s % 60)


class MMH3WindowContext(io.ComfyNode):
    """Tell the writing model WHERE in the clip this window sits.

    Without this the per-window loop hands the model the same text every
    iteration and only the audio changes -- so on a repetitive track, where the
    windows sound alike, nothing distinguishes window 5 from window 2. Add
    MMH3PromptAccumulate's prior_context, which says to keep the earlier
    sections' definitions byte-identical, and the model has one strong
    instruction pulling toward sameness and none pushing the other way. It
    converges: the late windows re-describe the same shots, often the same
    ending, and the same lyric lands in three or four of them.

    The span comes from `_plan`, the SAME function MMH3WindowPlan,
    MMH3SplitAudioToWindows and MMH3LoopingSampler use, so the timecode names
    the audio the window actually renders. Taking first_frame/last_frame from
    the splitter would work today and drift the first time anything about the
    schedule changes.

    SHOT TIMES STAY WINDOW-LOCAL. The line says so explicitly, because the
    obvious failure of handing a model a song timecode is that it starts writing
    [Shot 2] At 01:41.300 -- H3 reads shot times relative to the clip it is
    generating, which is this window.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3WindowContext",
            display_name="MMH3 Window Context",
            category="MMH3Tools/prompt",
            description=(
                "One line of text saying which span of the song this window covers, "
                "for the per-window prompt loop. Wire the same three numbers "
                "MMH3 Window Plan emits and the loop's index; the span is computed "
                "from the same schedule the sampler runs, so it cannot disagree.\n\n"
                "Concatenate it onto the END of the writing model's prompt -- it has "
                "to outweigh prior_context's 'keep these byte-identical', which sits "
                "at the end of the system prompt."
            ),
            inputs=[
                io.Int.Input(
                    "index", default=0, min=0, max=1023,
                    tooltip="Which window. Wire the for-loop's index -- the same one "
                            "MMH3 Split Audio to Windows gets, so the text and the "
                            "audio describe the same span."),
                io.Int.Input(
                    "total_frames", default=192, min=5, max=3600, step=17,
                    tooltip="Length of the whole clip. Wire MMH3 Window Plan's "
                            "total_frames output."),
                io.Int.Input(
                    "window_frames", default=124, min=5, max=3600, step=17,
                    tooltip="Wire MMH3 Window Plan's window_frames output, not the "
                            "value you typed into it -- the plan snaps it."),
                io.Int.Input(
                    "overlap_frames", default=22, min=0, max=3600, step=17,
                    tooltip="Wire MMH3 Window Plan's overlap_frames output."),
                io.Combo.Input(
                    "context_schedule", options=["standard_static", "standard_uniform"],
                    default="standard_static",
                    tooltip="Must match the rest of the chain, or the span is wrong."),
                io.Boolean.Input(
                    "state_local_times", default=True,
                    tooltip="Append the reminder that [Shot N] timestamps are relative "
                            "to THIS window and start at 00:00. Leave on unless your "
                            "system prompt already says it -- a model handed a song "
                            "timecode will otherwise start writing shot times in song "
                            "time, which H3 reads as the window's own clock."),
            ],
            outputs=[
                io.String.Output(display_name="context"),
                io.Int.Output(display_name="first_frame"),
                io.Int.Output(display_name="last_frame"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, index, total_frames, window_frames, overlap_frames,
                context_schedule="standard_static",
                state_local_times=True) -> io.NodeOutput:
        _length, _overlap, total_f, _total_t, windows = _plan(
            total_frames, window_frames, overlap_frames, context_schedule)
        spans = _window_frame_spans(windows, total_f)
        n = len(windows)

        idx = int(index)
        if idx >= n:
            # Same refusal as MMH3SplitAudioToWindows: a loop running one iteration
            # too many should stop, not quietly relabel window 0.
            raise ValueError(
                "MMH3WindowContext: index %d but only %d window%s (0-%d). Drive the "
                "loop's `total` from MMH3 Window Plan's window_count."
                % (idx, n, "" if n == 1 else "s", n - 1))

        fa, fb = spans[idx]
        context = ("This window covers %s to %s of a %s clip (window %d of %d)."
                   % (_timecode(fa), _timecode(fb + 1), _timecode(total_f),
                      idx + 1, n))
        if state_local_times:
            context += (" Write ONLY what is heard here. [Shot N] timestamps are "
                        "relative to THIS window and start at 00:00.")

        report = "window %d of %d: frames %d-%d, %s-%s" % (
            idx + 1, n, fa, fb, _timecode(fa), _timecode(fb + 1))
        logging.info("[MMH3WindowContext] " + report)
        return io.NodeOutput(context, fa, fb, report)


MAX_WINDOW_AUDIO = 8


class MMH3SplitAudioToWindows(io.ComfyNode):
    """Cut a track into one clip per context window, for writing per-window prompts.

    The point is to let an omni LLM hear exactly what each window will render, so the
    prompt it writes for that region describes the right music. Windows OVERLAP and the
    last one is CLAMPED to the clip end, so a uniform sequential split cannot express
    the schedule -- at 362 frames with a 124/22 window the real spans are

        0-123, 102-225, 204-327, 238-361

    and a uniform stride of 102 would put the fourth at 306-429: past the end, over
    audio the model never sees there. The prompt written from it would describe music
    that is not in that window.

    Takes ONE window length rather than a per-segment frame count, because the
    schedule is uniform by construction and the clamping is derived, not chosen. Spans
    come from the same `_plan()` the calculator uses, so the two cannot drift.

    Feeds MMH3ReferenceMultiPrompt: N prompts written against N segments, then
    MMH3CondSetSpread, then split_conds_to_windows. Set prompt_count equal to
    window_count and the mapping is one prompt per window.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3SplitAudioToWindows",
            display_name="MMH3 Split Audio to Windows",
            category="MMH3Tools/audio",
            description=(
                "Cut a track into one clip per context window, matching the real "
                "schedule including the overlap and the clamped final window. Feed each "
                "to an LLM that can hear, to write a prompt per window.\n\n"
                "Two ways out. The numbered sockets fan every window across the graph at "
                "once, which costs a copy of every downstream node per window. The "
                "`audio` output emits ONE window, chosen by `index` -- drive that from a "
                "for-loop and the graph is the same size for 4 windows or 40."
            ),
            inputs=[
                io.Audio.Input("audio"),
                io.Int.Input(
                    "total_frames", default=192, min=5, max=3600, step=17,
                    tooltip="Length of the whole clip. Same value the plan gets.",
                ),
                io.Int.Input(
                    "window_frames", default=124, min=5, max=3600, step=17,
                    tooltip="One window length -- the schedule is uniform, and the last "
                            "window's clamping is derived rather than chosen.",
                ),
                io.Int.Input(
                    "overlap_frames", default=22, min=0, max=3600, step=17,
                    tooltip="Shared frames between neighbouring windows.",
                ),
                io.Combo.Input(
                    "context_schedule", options=["standard_static", "standard_uniform"],
                    default="standard_static",
                    tooltip="Must match MMH3 Context Windows, or the segments describe a "
                            "schedule that is not the one being sampled.",
                ),
                io.Int.Input(
                    "index", default=0, min=0, max=4095,
                    tooltip="0-based, and drives the `audio` output only -- the numbered "
                            "sockets are unaffected. Wire a for-loop's index here and the "
                            "graph stays one node per stage no matter how many windows "
                            "there are. Out of range is an error rather than a wrap.",
                ),
            ],
            outputs=[
                io.Int.Output(display_name="window_count"),
                io.String.Output(display_name="report"),
            ] + [io.Audio.Output(display_name="audio_%d" % i)
                 for i in range(1, MAX_WINDOW_AUDIO + 1)] + [
                io.Audio.Output(display_name="audio"),
                io.Int.Output(display_name="first_frame"),
                io.Int.Output(display_name="last_frame"),
            ],
        )

    @classmethod
    def execute(cls, audio, total_frames, window_frames, overlap_frames,
                context_schedule, index=0) -> io.NodeOutput:
        _, _, total_f, _, windows = _plan(
            total_frames, window_frames, overlap_frames, context_schedule)
        spans = _window_frame_spans(windows, total_f)

        wav = audio["waveform"]
        sr = int(audio.get("sample_rate", 44100))
        if wav.ndim == 2:
            wav = wav.unsqueeze(0)
        have = int(wav.shape[-1])

        notes = []
        if len(spans) > MAX_WINDOW_AUDIO:
            notes.append("%d windows but only %d numbered sockets; audio_1..%d stop at "
                         "window %d. The `audio` output has no such ceiling -- drive it "
                         "with `index` instead."
                         % (len(spans), MAX_WINDOW_AUDIO, MAX_WINDOW_AUDIO,
                            MAX_WINDOW_AUDIO - 1))

        clip_seconds = total_f / float(FPS)
        if have < int(clip_seconds * sr) - sr // 10:      # more than 0.1s short
            notes.append("track is %.2fs but the clip is %.2fs; short windows are padded "
                         "with silence" % (have / float(sr), clip_seconds))

        # Every window is cut, not just the ones with a numbered socket -- `index` must be
        # able to reach past MAX_WINDOW_AUDIO, which is what makes the loop form worth
        # having. Slicing a waveform is cheap; this is not the expensive part.
        segments, lines = [], []
        for i, (fa, fb) in enumerate(spans):
            # fb is inclusive, so the span ends at the START of the frame after it
            s0 = int(round(fa / float(FPS) * sr))
            s1 = int(round((fb + 1) / float(FPS) * sr))
            seg = wav[..., max(0, s0):min(have, s1)]
            want = s1 - s0
            if int(seg.shape[-1]) < want:
                pad = torch.zeros(list(seg.shape[:-1]) + [want - int(seg.shape[-1])],
                                  dtype=wav.dtype, device=wav.device)
                seg = torch.cat([seg, pad], dim=-1)
            if seg.shape[1] == 1:                          # mono -> stereo, as H3 wants
                seg = seg.repeat(1, 2, 1)
            segments.append({"waveform": seg.contiguous(), "sample_rate": sr})
            lines.append("  %d  frames %4d-%-4d  %6.2fs-%6.2fs  (%.2fs)"
                         % (i, fa, fb, s0 / float(sr), s1 / float(sr),
                            (s1 - s0) / float(sr)))

        emitted = len(segments)

        # The indexed output. Out of range is an error rather than a wrap, matching
        # MMH3CondSelect -- a loop running one iteration too many should stop, not
        # quietly render window 0 a second time.
        idx = int(index)
        if idx >= emitted:
            raise ValueError(
                "MMH3SplitAudioToWindows: index %d but only %d window%s (0-%d). Drive "
                "`total` on the loop from this node's window_count, or from "
                "MMH3WindowPlan -- the two agree by construction."
                % (idx, emitted, "" if emitted == 1 else "s", emitted - 1))
        sel_audio = segments[idx]
        sel_fa, sel_fb = spans[idx]

        numbered = segments[:MAX_WINDOW_AUDIO]
        while len(numbered) < MAX_WINDOW_AUDIO:
            numbered.append(None)

        report = ("%d window%s of %.2fs audio  (index %d -> frames %d-%d)\n%s"
                  % (emitted, "" if emitted == 1 else "s", have / float(sr),
                     idx, sel_fa, sel_fb, "\n".join(lines)))
        for n in notes:
            report += "\n  ! " + n
        logging.info("[MMH3SplitAudioToWindows] " + report.splitlines()[0])
        return io.NodeOutput(emitted, report, *numbered,
                             sel_audio, int(sel_fa), int(sel_fb))
