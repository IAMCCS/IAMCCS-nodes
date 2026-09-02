"""Calculators and inspection.

Follows the LTXAVTools convention: concise typed outputs plus a short `label`
string, rather than a verbose info block.

Difference from LTXDimensionCalculator: LTX emitted a fixed `width_half` /
`height_half` pair for its two-stage pipeline. H3 has no second stage -- the
secondary pair here is the REFERENCE size, driven by a downscale factor and
snapped to what the patch grid supports.
"""

import logging
import math

import folder_paths
from comfy.nested_tensor import NestedTensor
from comfy_api.latest import io

from .common import (
    AUDIO_T_DIM,
    BASE_SHORT_EDGE,
    CANVAS_MULTIPLE,
    FPS,
    FRAMES_PER_GROUP,
    FRAME_BASE,
    MAX_PIXELS,
    PATCH,
    VAE_SPATIAL,
    VIDEO_T_DIM,
    frames_to_audio_t,
    frames_to_latents,
    latents_to_frames,
    on_grid,
    snap_downscale,
    supported_downscale_factors,
    unpack_av,
)


class MMH3FrameCalculator(io.ComfyNode):
    """Seconds -> frame count on the model's 17j+5 grid.

    At 24fps achievable durations are discrete. Solving 24s = 5 (mod 17) gives
    s = 8 (mod 17), so 8.000s (192 frames) is the only whole-second duration in
    the 4-15s supported range.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3FrameCalculator",
            display_name="MMH3 Frame Calculator",
            category="MMH3Tools",
            description="Duration in seconds -> frame count on the 17j+5 grid, with the "
                        "video and audio latent counts it implies.",
            inputs=[
                # min and step ARE the grid: achievable durations are (17j+5)/24, which
                # is 5/24 with a spacing of exactly 17/24. At the old step of 0.01 every
                # arrow-click landed between two valid durations and the node silently
                # snapped, so the frame count moved a whole group without the widget
                # showing anything -- and downstream that is a window appearing.
                io.Float.Input("seconds",
                               default=FRAME_BASE / FPS + 7 * FRAMES_PER_GROUP / FPS,
                               min=FRAME_BASE / FPS, max=150.0,
                               step=FRAMES_PER_GROUP / FPS,
                               tooltip="Achievable durations are discrete: (17j+5)/24, so "
                                       "0.208, 0.917, 1.625, 2.333 ... spaced 0.708s apart. "
                                       "The arrows step exactly one duration; a typed value "
                                       "is snapped per `rounding` and the label says where "
                                       "it landed. 8.000s is the only whole second in the "
                                       "4-15s trained range."),
                io.Combo.Input("rounding", options=["nearest", "up", "down"], default="nearest"),
            ],
            outputs=[
                io.Int.Output(display_name="frame_count"),
                io.Int.Output(display_name="latent_frames"),
                io.Int.Output(display_name="audio_latent_frames"),
                io.Float.Output(display_name="actual_seconds"),
                io.String.Output(display_name="label"),
            ],
        )

    @classmethod
    def execute(cls, seconds, rounding) -> io.NodeOutput:
        target = seconds * FPS
        j = int(math.floor((target - FRAME_BASE) / FRAMES_PER_GROUP))
        lo = FRAMES_PER_GROUP * max(0, j) + FRAME_BASE
        if lo > target:
            lo = FRAME_BASE
        hi = lo if lo >= target else lo + FRAMES_PER_GROUP

        if rounding == "up":
            f = hi
        elif rounding == "down":
            f = lo
        else:
            f = lo if (target - lo) <= (hi - target) else hi

        actual = f / FPS
        drift = actual - seconds
        label = "%.3fs = %d frames, %d latents" % (actual, f, frames_to_latents(f))
        if abs(drift) >= 1e-4:
            # Say it out loud. This is the whole failure mode: the widget keeps showing
            # what was typed, the pipeline uses something else, and the difference only
            # surfaces as a window count that is not what was expected.
            label += "  (%s from %.3fs, %+.3fs)" % (
                "snapped " + ("up" if drift > 0 else "down"), seconds, drift)
            if abs(drift) > FRAMES_PER_GROUP / FPS / 2:
                label += "\n  ! more than half a step -- the nearest duration below is " \
                         "%.3fs" % ((f - FRAMES_PER_GROUP) / FPS)
        return io.NodeOutput(f, frames_to_latents(f), frames_to_audio_t(f), actual, label)


# ---------------------------------------------------------------------------
# Resolution presets  (mirrors LTXAVTools' calculator, on H3's 32px grid)
# ---------------------------------------------------------------------------
RATIOS = [
    (21, 9, "21:9 - ultrawide, cinematic", "9:21 - ultrawide portrait"),
    (16, 9, "16:9 - YouTube, HD, TV", "9:16 - TikTok, Reels, Shorts"),
    (3, 2, "3:2 - photography, DSLR", "2:3 - portrait photo"),
    (4, 3, "4:3 - classic TV, monitor", "3:4 - tablet portrait"),
    (1, 1, "1:1 - square, Instagram", "1:1 - square, Instagram"),
]
MEGAPIXELS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.2, 1.5, 2.0]


def _snap32(x):
    return max(CANVAS_MULTIPLE, int(round(x / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE)


def native_canvas(rl, rs):
    """The model's own adapt_canvas(): 768 short edge, capped at 768*1344, rounded to 32."""
    r = rl / rs
    nom_w, nom_h = (BASE_SHORT_EDGE * r, float(BASE_SHORT_EDGE))
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return _snap32(nom_w), _snap32(nom_h)


def build_options(ratio_long, ratio_short, landscape):
    """Resolutions for one ratio, smallest first. '[native]' marks the trained canvas."""
    r = ratio_long / max(ratio_short, 1)
    nat = native_canvas(ratio_long, ratio_short)
    seen, out = set(), []
    entries = []
    for mp in MEGAPIXELS:
        h = _snap32(math.sqrt(mp * 1e6 / r))
        entries.append((_snap32(h * r), h))
    entries.append(nat)
    for lw, lh in sorted(set(entries), key=lambda t: t[0] * t[1]):
        w, h = (lw, lh) if landscape else (lh, lw)
        tag = "%dx%d  %.2fMP" % (w, h, w * h / 1e6)
        if (lw, lh) == nat:
            tag += "  [native]"
        if tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


# Declared options must be the FULL union across every ratio and orientation, because
# the JS narrows options.values client-side but ComfyUI validates the submitted value
# against what Python declared. Declaring only the 16:9 landscape list makes every
# other choice fail with "Some input values are not available for this node" - a
# validation error that never reaches the console.
_DEFAULT_RATIOS = ([lab for _, _, lab, _ in RATIOS] +
                   [p for _, _, lab, p in RATIOS if p not in [l for _, _, l, _ in RATIOS]])
_ALL_OPTS = []
for _rl, _rs, _, _ in RATIOS:
    for _land in (True, False):
        for _o in build_options(_rl, _rs, _land):
            if _o not in _ALL_OPTS:
                _ALL_OPTS.append(_o)
_DEFAULT_OPTS = build_options(16, 9, landscape=True)
_DEFAULT_OPT = next((o for o in _DEFAULT_OPTS if "[native]" in o), _DEFAULT_OPTS[0])
_DEFAULT_RATIO = _DEFAULT_RATIOS[1]


class MMH3DimensionCalculator(io.ComfyNode):
    """Generation dimensions plus a reference pair sized by a snapped downscale factor.

    Pixel dims snap to 32 (16x VAE spatial then a 2x2 patch). Latent dims are px/16
    and must stay EVEN, so a downscale factor is valid only when latent/f is an even
    integer on both axes -- the divisors of gcd(latent_h//2, latent_w//2). For
    1344x768 that is [1, 2, 3, 6]; 4 is NOT valid.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3DimensionCalculator",
            display_name="MMH3 Dimension Calculator",
            category="MMH3Tools",
            description="Snap generation dimensions to the 32px grid and derive a reference "
                        "size from a downscale factor, snapped to what the patch grid allows.",
            inputs=[
                io.Combo.Input("ratio", options=_DEFAULT_RATIOS, default=_DEFAULT_RATIO,
                               tooltip="Common aspect ratios and their typical uses."),
                io.Combo.Input("orientation", options=["Landscape", "Portrait"],
                               default="Landscape"),
                io.Combo.Input("resolution", options=_ALL_OPTS, default=_DEFAULT_OPT,
                               tooltip="Resolutions for the selected ratio, all multiples of 32. "
                                       "[native] marks the 768-short-edge canvas the model was "
                                       "trained on; larger costs tokens quadratically at "
                                       "every sampling step."),
                io.Int.Input("downscale_factor", default=2, min=1, max=32, step=1,
                             tooltip="Reference downscale. Snapped to the nearest factor that "
                                     "keeps both latent dims even; ties resolve gentler."),
                io.Boolean.Input("use_custom", default=False, optional=True,
                                 tooltip="Override the preset with custom_width/custom_height. "
                                         "Toggle-controlled so a bypassed upstream node cannot "
                                         "silently switch modes."),
                io.Int.Input("custom_width", default=0, min=0, max=16384, step=8, optional=True),
                io.Int.Input("custom_height", default=0, min=0, max=16384, step=8, optional=True),
            ],
            outputs=[
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.Int.Output(display_name="width_ref"),
                io.Int.Output(display_name="height_ref"),
                io.String.Output(display_name="label"),
            ],
        )

    @classmethod
    def validate_inputs(cls, **kwargs):
        # The JS narrows the ratio/resolution lists per selection, so a submitted value
        # can legitimately sit outside what any single declared list contains. execute()
        # parses "WxH" out of the string and tolerates anything well-formed.
        return True

    @classmethod
    def execute(cls, ratio, orientation, resolution, downscale_factor, use_custom=False,
                custom_width=0, custom_height=0) -> io.NodeOutput:
        if use_custom and custom_width > 0 and custom_height > 0:
            w, h = _snap32(custom_width), _snap32(custom_height)
        else:
            if use_custom:
                logging.info("[MMH3DimensionCalculator] use_custom on but dims <= 0 "
                             "(upstream bypassed?); falling back to the dropdown")
            # the option list already encodes orientation, so just parse it
            w, h = (int(x) for x in resolution.split()[0].split("x"))
        lw, lh = w // VAE_SPATIAL, h // VAE_SPATIAL

        f = snap_downscale(downscale_factor, lh, lw)
        rw, rh = (lw // f) * VAE_SPATIAL, (lh // f) * VAE_SPATIAL

        tok = (lw // PATCH) * (lh // PATCH)
        rtok = ((lw // f) // PATCH) * ((lh // f) // PATCH)
        label = "%dx%d -> ref %dx%d (%dx, %d%% tokens)" % (w, h, rw, rh, f, round(100.0 * rtok / max(1, tok)))
        if w * h > MAX_PIXELS:
            label += "  OVER CANVAS CAP"

        return io.NodeOutput(w, h, rw, rh, label)


class MMH3LatentInfo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3LatentInfo",
            display_name="MMH3 Latent Info",
            category="MMH3Tools/utils",
            description="Report AV latent shapes, implied frame count, and grid alignment.",
            inputs=[io.Latent.Input("latent")],
            outputs=[io.String.Output(display_name="info")],
        )

    @classmethod
    def execute(cls, latent) -> io.NodeOutput:
        video, audio = unpack_av(latent)
        t = int(video.shape[VIDEO_T_DIM])
        frames = latents_to_frames(t)
        expected_audio = frames_to_audio_t(frames)
        actual_audio = int(audio.shape[AUDIO_T_DIM])

        lines = [
            "video latent : %s   (B,C,T,h,w)" % (tuple(video.shape),),
            "audio latent : %s   (B,C,stereo,T40)" % (tuple(audio.shape),),
            "pixel size   : %d x %d" % (video.shape[4] * VAE_SPATIAL, video.shape[3] * VAE_SPATIAL),
            "video T      : %d latents -> %d frames (%.3fs @ %dfps)" % (t, frames, frames / FPS, FPS),
            "audio T40    : %d (expected %d)%s" % (
                actual_audio, expected_audio,
                "" if actual_audio == expected_audio else "  <-- MISMATCH"),
            "5j+2 grid    : %s" % ("yes" if on_grid(t) else "NO -- off grid"),
            "downscale    : valid factors %s" % (
                ", ".join(str(x) for x in supported_downscale_factors(
                    int(video.shape[3]), int(video.shape[4])))),
        ]
        lines += cls._mask_lines(latent.get("noise_mask"), t, actual_audio)
        info = "\n".join(lines)
        print("[MMH3LatentInfo]\n" + info)
        return io.NodeOutput(info)

    @staticmethod
    def _mask_lines(mask, video_t, audio_t):
        """Describe the noise mask against THIS latent's dimensions.

        Chunked sampling slices the mask by time (video dim 2, audio dim 3), so a
        mask whose time extent does not match its half's latent is sliced out of
        range mid-run -- chunk 0 clamps silently, a later chunk gets zero elements
        and dies deep in core's reshape. The mismatch is only visible here, before
        sampling, which is why this reports it.
        """
        if mask is None:
            return ["noise_mask   : none"]
        if isinstance(mask, NestedTensor):
            parts = mask.unbind()
            vm = parts[0]
            am = parts[1] if len(parts) > 1 else None
        else:
            vm, am = mask, None

        out = ["noise_mask   : %s" % ("NestedTensor(video, audio)"
                                      if am is not None else type(mask).__name__)]
        vt = int(vm.shape[2]) if vm.ndim >= 3 else -1
        out.append("  video mask : %s   time dim %s (video T %d)%s" % (
            tuple(vm.shape), vt, video_t,
            "" if vt == video_t else "  <-- MISMATCH: sliced by video time in chunked runs"))
        if am is not None:
            at = int(am.shape[-1])
            hint = ""
            if at < audio_t:
                hint = "  <-- MISMATCH: chunked runs slice this by audio time; a " \
                       "chunk starting past %d slices ZERO elements and crashes" % at
            elif at > audio_t:
                hint = "  <-- longer than the audio: slices stay in range, but the " \
                       "mask was built for a different timeline than these samples"
            out.append("  audio mask : %s   time dim %s (audio T40 %d)%s" % (
                tuple(am.shape), at, audio_t, hint))
        elif audio_t:
            out.append("  audio mask : none -- audio present but unmasked")
        return out



# ---------------------------------------------------------------------------
# Three-stage upscale ladder
# ---------------------------------------------------------------------------
# A ratio lands EXACTLY on the 32px grid only at integer multiples of its minimal
# on-grid unit: 16:9 needs w/h = 16/9 with both /32, which is w=512k, h=288k. Every
# k is exact 16:9, exactly on grid. Working in k instead of pixels means no stage is
# ever snapped, so the aspect cannot drift between stages -- which matters here,
# because a low-denoise pass onto a slightly different aspect resamples the whole
# frame instead of just adding detail.
LADDER_RATIOS = [
    (16, 9, "16:9 - YouTube, HD, TV"),
    (4, 3, "4:3 - classic TV, monitor"),
    (3, 2, "3:2 - photography, DSLR"),
    (1, 1, "1:1 - square"),
    (21, 9, "21:9 - ultrawide, cinematic"),
]
LADDER_RATIO_LABELS = [lab for _, _, lab in LADDER_RATIOS]


def ladder_stages(a, b, landscape, target_long, min_megapixels):
    """Three (w, h) stages, coarsest first, plus a list of notes.

    Constraints, all of which come from measurement rather than taste:
      * every stage exact-aspect and on the 32 grid  -> integer multiples of the unit
      * no step above 2x  -> a low-denoise pass cannot invent more than that
      * stage 1 at or above min_megapixels  -> below it the first pass is not
        upscalable, and stage 2 sharpens mush rather than repairing it
    """
    notes = []
    g = math.gcd(a, b)
    uw, uh = CANVAS_MULTIPLE * (a // g), CANVAS_MULTIPLE * (b // g)
    if not landscape:
        uw, uh = uh, uw

    k3 = max(1, int(target_long) // max(uw, uh))
    if k3 * max(uw, uh) != int(target_long):
        notes.append("%d is not a multiple of the %d px unit long edge; using %d"
                     % (int(target_long), max(uw, uh), k3 * max(uw, uh)))

    k1 = max(1, math.ceil(math.sqrt(max(0.0, min_megapixels) * 1e6 / float(uw * uh))))
    if k1 >= k3:
        notes.append("min_megapixels %.2f already needs the full target; the ladder "
                     "collapses to a single stage" % min_megapixels)
        k1 = k2 = k3
    else:
        # step1 = k2/k1 <= 2 and step2 = k3/k2 <= 2, so k2 in [k3/2, 2*k1].
        # Feasible only when k3 <= 4*k1 -- three stages cannot exceed 4x in total.
        lo, hi = max(k1 + 1, math.ceil(k3 / 2)), min(2 * k1, k3 - 1)
        if hi < lo:
            # two distinct reasons, and blaming the wrong one sends you the wrong way
            if k3 > 4 * k1:
                notes.append("%.1fx total upscale needs more than 3 stages at 2x per step; "
                             "raise min_megapixels or lower target_long_edge"
                             % (k3 / float(k1)))
            else:
                notes.append("only %.2fx total upscale, and no on-grid stage fits between "
                             "%dx%d and the target; this is really a 2-stage ladder"
                             % (k3 / float(k1), uw * k1, uh * k1))
            k2 = max(k1 + 1, math.ceil(k3 / 2))
        else:
            # geometric mean spreads the work evenly across the two steps
            k2 = min(hi, max(lo, int(round(math.sqrt(k1 * k3)))))

    return [(uw * k, uh * k) for k in (k1, k2, k3)], notes


def _snap32(v):
    """Round to the 32px canvas unit, never below one unit.

    32, not 16: latent dims are px/16 and must stay EVEN for the 2x2 patch.
    """
    return max(CANVAS_MULTIPLE, int(round(v / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE)


def base_canvas(a, b, landscape):
    """What H3-Base actually generates for this aspect: adapt_canvas, reproduced.

    768 short edge, area capped at 768*1344, each axis rounded to 32. Copied from
    core's `adapt_canvas` rather than invented, because stage 1 has to be the size
    the model really emits -- a stage-1 number that merely looks reasonable makes
    the 2K stage a resize of something that was never rendered.
    """
    w_r, h_r = (a, b) if landscape else (b, a)
    ratio = w_r / float(h_r)
    if ratio >= 1.0:
        nw, nh = BASE_SHORT_EDGE * ratio, float(BASE_SHORT_EDGE)
    else:
        nw, nh = float(BASE_SHORT_EDGE), BASE_SHORT_EDGE / ratio
    if nw * nh > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nw * nh))
        nw, nh = nw * s, nh * s
    return _snap32(nw), _snap32(nh)


class MMH3Regenerate2KDims(io.ComfyNode):
    """Stage-1 (H3-Base) and stage-2 (2K) dimensions for a Regenerate-2K pass.

    H3-Regenerate-2K, per the model card, "feeds the 768p result together with the
    original context back into H3 to regenerate the output at 2K". It is not open
    sourced, so this sizes the same two stages for a local equivalent: generate at
    the resolution H3-Base really uses, then re-run at 2K.

    STAGE 1 IS NOT A CHOICE. It comes from core's `adapt_canvas` -- 768 short edge,
    area capped at 768*1344, axes rounded to 32 -- because that is what the model
    emits whatever you ask for. Picking a nicer-looking stage-1 number just means
    stage 2 upscales something that was never rendered at that size.

    BOTH STAGES SHARE ONE ASPECT. The 2K stage is derived from the stage-1 canvas,
    not from the nominal ratio, so the rounding at stage 1 cannot leave the two
    stages fractionally different -- which is what puts a squeeze or a crop in an
    otherwise clean upscale.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3Regenerate2KDims",
            display_name="MMH3 Regenerate-2K Dimensions",
            category="MMH3Tools/calculators",
            description=(
                "Matched stage-1 and stage-2 dimensions for a two-pass 768p -> 2K run. "
                "Stage 1 is what H3-Base actually generates for the chosen aspect; "
                "stage 2 is the same aspect at 2K, on the 32px grid."
            ),
            inputs=[
                io.Combo.Input("ratio", options=LADDER_RATIO_LABELS,
                               default=LADDER_RATIO_LABELS[0]),
                io.Combo.Input("orientation", options=["Landscape", "Portrait"],
                               default="Landscape"),
                io.Int.Input(
                    "target_long_edge", default=2048, min=768, max=8192, step=32,
                    tooltip="Long edge of the 2K stage. Snapped to 32. The short edge "
                            "follows from the stage-1 canvas, so the aspect matches "
                            "stage 1 exactly rather than the nominal ratio."),
            ],
            outputs=[
                io.Int.Output(display_name="width_1"),
                io.Int.Output(display_name="height_1"),
                io.Int.Output(display_name="width_2k"),
                io.Int.Output(display_name="height_2k"),
                io.Float.Output(display_name="scale"),
                io.String.Output(display_name="label"),
            ],
        )

    @classmethod
    def execute(cls, ratio, orientation, target_long_edge) -> io.NodeOutput:
        a, b = next(((x, y) for x, y, lab in LADDER_RATIOS if lab == ratio), (16, 9))
        landscape = orientation == "Landscape"
        w1, h1 = base_canvas(a, b, landscape)

        # Stage 2 is an INTEGER multiple of stage 1's on-grid unit, not the requested
        # long edge rounded to 32. Rounding each axis independently drifts the aspect
        # -- 16:9 at a 2048 long edge lands on 2048x1184, which is 1.7297 -- and the
        # squeeze shows up in every frame. Reducing w1:h1 to its smallest 32px-aligned
        # pair gives a unit that can only ever scale exactly.
        pw, ph = w1 // CANVAS_MULTIPLE, h1 // CANVAS_MULTIPLE
        g = math.gcd(pw, ph)
        unit_w, unit_h = CANVAS_MULTIPLE * pw // g, CANVAS_MULTIPLE * ph // g
        unit_long = max(unit_w, unit_h)

        j = max(1, int(round(_snap32(target_long_edge) / float(unit_long))))
        w2, h2 = unit_w * j, unit_h * j

        long1 = max(w1, h1)
        long2 = max(w2, h2)
        scale = long2 / float(long1)
        notes = []
        if long2 != _snap32(target_long_edge):
            notes.append("long edge %d, not the %d asked for: %dx%d is the nearest "
                         "multiple of this aspect's %dx%d unit that keeps the ratio exact"
                         % (long2, _snap32(target_long_edge), w2, h2, unit_w, unit_h))
        if scale < 1.0:
            notes.append("target_long_edge %d is BELOW the stage-1 long edge %d -- this "
                         "is a downscale, not a 2K pass" % (long2, long1))
        elif scale > 4.0:
            notes.append("%.2fx in one step; H3 was trained at 768 short edge and a jump "
                         "this large is outside anything measured" % scale)
        drift = abs((w2 / float(h2)) - (w1 / float(h1)))
        if drift > 0.01:
            notes.append("aspect drifts %.3f between stages after 32px rounding; the "
                         "upscale will squeeze slightly" % drift)

        label = "stage 1  %dx%d (%.2f MP)  ->  2K  %dx%d (%.2f MP)   %.2fx" % (
            w1, h1, w1 * h1 / 1e6, w2, h2, w2 * h2 / 1e6, scale)
        for n in notes:
            label += "\n  ! " + n
            logging.warning("[MMH3Regenerate2KDims] %s", n)
        return io.NodeOutput(w1, h1, w2, h2, round(scale, 4), label)


class MMH3UpscaleLadder(io.ComfyNode):
    """Dimensions for a three-stage generate-small-then-denoise-up pipeline."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3UpscaleLadder",
            display_name="MMH3 Upscale Ladder",
            category="MMH3Tools/calculators",
            description=(
                "Three exact-aspect, on-grid stages for progressive generation: generate "
                "at stage 1, decode/upscale/re-encode and low-denoise at stage 2, again "
                "at stage 3. No step exceeds 2x."
            ),
            inputs=[
                io.Combo.Input("ratio", options=LADDER_RATIO_LABELS,
                               default=LADDER_RATIO_LABELS[0]),
                io.Combo.Input("orientation", options=["Landscape", "Portrait"],
                               default="Landscape"),
                io.Int.Input(
                    "target_long_edge", default=2048, min=256, max=8192, step=32,
                    tooltip="Long edge of the FINAL stage. 2048 is 2K. Rounded down to a "
                            "multiple of the ratio's on-grid unit so the aspect stays exact.",
                ),
                io.Float.Input(
                    "min_megapixels", default=0.4, min=0.05, max=8.0, step=0.05,
                    tooltip="Floor for stage 1. Measured: below about 0.4 MP the first "
                            "pass stops being upscalable, and a light denoise at stage 2 "
                            "sharpens mush instead of repairing structure.",
                ),
            ],
            outputs=[
                io.Int.Output(display_name="width_1"),
                io.Int.Output(display_name="height_1"),
                io.Int.Output(display_name="width_2"),
                io.Int.Output(display_name="height_2"),
                io.Int.Output(display_name="width_3"),
                io.Int.Output(display_name="height_3"),
                io.String.Output(display_name="label"),
            ],
        )

    @classmethod
    def execute(cls, ratio, orientation, target_long_edge, min_megapixels) -> io.NodeOutput:
        a, b = next(((x, y) for x, y, lab in LADDER_RATIOS if lab == ratio), (16, 9))
        stages, notes = ladder_stages(a, b, orientation == "Landscape",
                                      target_long_edge, min_megapixels)
        (w1, h1), (w2, h2), (w3, h3) = stages
        steps = " ".join("%.2fx" % (stages[i + 1][0] / float(stages[i][0]))
                         for i in range(2))
        label = "%dx%d (%.2f MP) -> %dx%d -> %dx%d  [%s]" % (
            w1, h1, w1 * h1 / 1e6, w2, h2, w3, h3, steps)
        for n in notes:
            label += "\n  ! " + n
            logging.warning("[MMH3UpscaleLadder] %s", n)
        return io.NodeOutput(w1, h1, w2, h2, w3, h3, label)


# Target aspects worth reframing TO: the ladder's set plus the vertical and square
# shapes MiniMax's own MV skill lists per platform, since a social crop is the main
# reason to reframe at all.
REFRAME_RATIOS = [
    (9, 16, "9:16 - TikTok, Reels, Shorts"),
    (4, 5, "4:5 - Instagram feed"),
    (1, 1, "1:1 - square"),
    (16, 9, "16:9 - YouTube, HD"),
    (4, 3, "4:3 - classic"),
    (21, 9, "21:9 - ultrawide"),
]
REFRAME_LABELS = [lab for _, _, lab in REFRAME_RATIOS]

REFRAME_MODES = ["extend", "crop", "balanced"]
ANCHORS = ["center", "top", "bottom", "left", "right"]


def _split(delta, anchor, lo_name, hi_name):
    """Divide `delta` between two sides, keeping both on the canvas multiple."""
    if anchor == lo_name:
        return 0, delta
    if anchor == hi_name:
        return delta, 0
    lo = (delta // 2 // CANVAS_MULTIPLE) * CANVAS_MULTIPLE
    return lo, delta - lo


def reframe_plan(src_w, src_h, ratio_w, ratio_h, mode, anchor):
    """Target size and per-side crop/pad to reach ratio_w:ratio_h.

    Three strategies, because reframing an orientation is a genuine trade:

      extend    grow the short axis. Keeps every pixel of the source, but a 16:9 -> 9:16
                flip roughly TRIPLES the frame, and attention is quadratic.
      crop      shrink the long axis. Costs no compute and loses content at the edges.
      balanced  do both, landing near the SOURCE pixel count -- crop the long axis part
                of the way and extend the short axis the rest. Usually the right answer
                for an orientation flip, where pure extension is unaffordable and pure
                cropping throws away most of the frame.

    Returns (moves, out_w, out_h, notes) where `moves` is a dict of four SIGNED sides:
    positive moves that edge outward (pad, generated), negative inward (crop, discarded).
    An edge can only ever go one way, so one number per side says everything.
    """
    notes = []
    want = ratio_w / float(ratio_h)

    if mode == "extend":
        out_w, out_h = max(src_w, src_h * want), max(src_h, src_w / want)
    elif mode == "crop":
        out_w, out_h = min(src_w, src_h * want), min(src_h, src_w / want)
    else:
        # same area, target ratio: w = sqrt(A*r), h = sqrt(A/r)
        area = float(src_w * src_h)
        out_w, out_h = math.sqrt(area * want), math.sqrt(area / want)

    # both axes must land on the canvas multiple, so an exact ratio is not always
    # reachable -- round to nearest rather than up, or every call grows by a step
    out_w = max(CANVAS_MULTIPLE, int(round(out_w / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE)
    out_h = max(CANVAS_MULTIPLE, int(round(out_h / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE)

    dw, dh = out_w - src_w, out_h - src_h
    moves = {"left": 0, "right": 0, "top": 0, "bottom": 0}
    if dw:
        lo, hi = _split(abs(dw), anchor, "left", "right")
        sign = 1 if dw > 0 else -1
        moves["left"], moves["right"] = sign * lo, sign * hi
    if dh:
        lo, hi = _split(abs(dh), anchor, "top", "bottom")
        sign = 1 if dh > 0 else -1
        moves["top"], moves["bottom"] = sign * lo, sign * hi

    got = out_w / float(out_h)
    if abs(got - want) > 0.02:
        notes.append("landed on %.3f rather than %.3f -- both axes must sit on %dpx"
                     % (got, want, CANVAS_MULTIPLE))
    return moves, out_w, out_h, notes


class MMH3ReframePads(io.ComfyNode):
    """Work out the crop and pad amounts that take a clip to a different aspect ratio.

    Reframing is the main reason to outpaint -- a 16:9 generation wanted vertically for
    Reels -- and doing it by hand means getting up to eight numbers right, all multiples
    of 32, that sum to a size both axes can actually land on.

    The mode is the real decision, and it is a trade rather than a preference:

        extend    keeps every pixel, but 16:9 -> 9:16 is ~3.1x the frame, and attention
                  is quadratic in sequence length, so ~10x the cost per step. Outpainting
                  converges in about half the steps of a normal generation, though, so
                  the real bill is nearer 5x.
        crop      free, and throws away most of the width.
        balanced  crops the long axis part of the way and extends the short axis the
                  rest, landing near the SOURCE pixel count. Usually what you want for
                  an orientation flip.

    The four outputs are SIGNED and go straight to MMH3 Outpaint Latent's four sides:
    positive pads outward, negative crops inward. Balanced mode emits both signs at once
    -- crop the long axis, extend the short one.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3ReframePads",
            display_name="MMH3 Reframe Pads",
            category="MMH3Tools/utils",
            description=(
                "Given a source size and a target aspect, emit the crop and pad amounts "
                "for MMH3 Outpaint Latent, snapped to 32 and anchored where you choose."
            ),
            inputs=[
                io.Int.Input("source_width", default=1344, min=32, max=16384, step=32,
                             tooltip="Current width in PIXELS."),
                io.Int.Input("source_height", default=768, min=32, max=16384, step=32),
                io.Combo.Input("target_ratio", options=REFRAME_LABELS,
                               default=REFRAME_LABELS[0]),
                io.Combo.Input(
                    "mode", options=REFRAME_MODES, default="balanced",
                    tooltip="extend: grow only, keeps every pixel, costs the most. "
                            "crop: shrink only, free, loses the edges. "
                            "balanced: both, landing near the source pixel count.",
                ),
                io.Combo.Input(
                    "anchor", options=ANCHORS, default="center",
                    tooltip="Where the existing frame sits in the new one. 'bottom' "
                            "keeps the subject low and grows headroom, and so on. "
                            "Applies to crops the same way.",
                ),
            ],
            outputs=[
                io.Int.Output(display_name="left"),
                io.Int.Output(display_name="right"),
                io.Int.Output(display_name="top"),
                io.Int.Output(display_name="bottom"),
                io.Int.Output(display_name="target_width"),
                io.Int.Output(display_name="target_height"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, source_width, source_height, target_ratio, mode,
                anchor) -> io.NodeOutput:
        rw, rh, _ = next(r for r in REFRAME_RATIOS if r[2] == target_ratio)
        sw = (int(source_width) // CANVAS_MULTIPLE) * CANVAS_MULTIPLE
        sh = (int(source_height) // CANVAS_MULTIPLE) * CANVAS_MULTIPLE

        moves, ow, oh, notes = reframe_plan(sw, sh, rw, rh, mode, anchor)
        src_px, out_px = sw * sh, ow * oh
        grow = out_px / float(src_px)

        report = ("%dx%d -> %dx%d (%s, %s, %s)\n"
                  "  L%+d R%+d T%+d B%+d   (+ pads outward, - crops inward)\n"
                  "  %.2f MP -> %.2f MP, %.2fx the pixels, roughly %.1fx the attention "
                  "cost per step (outpainting needs about half the steps, so ~%.1fx the "
                  "generation)"
                  % (sw, sh, ow, oh, target_ratio.split(" - ")[0], mode, anchor,
                     moves["left"], moves["right"], moves["top"], moves["bottom"],
                     src_px / 1e6, out_px / 1e6, grow, grow ** 2, (grow ** 2) / 2.0))

        if (sw, sh) != (int(source_width), int(source_height)):
            notes.append("source snapped to %dx%d" % (sw, sh))
        if not any(moves.values()):
            # the "landed on x rather than y" note is noise when nothing moved: the
            # source is simply already within a canvas step of the target
            notes = [n for n in notes if not n.startswith("landed on")]
            notes.append("already at this ratio, within one %dpx step -- nothing to do"
                         % CANVAS_MULTIPLE)
        if out_px > MAX_PIXELS:
            notes.append("%.2f MP is past H3's %.2f MP canvas -- beyond what the open "
                         "weights were trained at. 'balanced' or a downscale first would "
                         "stay inside it" % (out_px / 1e6, MAX_PIXELS / 1e6))
        if mode == "crop" and any(v < 0 for v in moves.values()):
            lost = 100.0 * (1.0 - out_px / float(src_px))
            notes.append("crop discards %.0f%% of the frame; nothing regenerates it"
                         % lost)
        for n in notes:
            report += "\n  ! " + n
        logging.info("[MMH3ReframePads] " + report.splitlines()[0])
        return io.NodeOutput(moves["left"], moves["right"], moves["top"],
                             moves["bottom"], ow, oh, report)


# ---------------------------------------------------------------------------
# AdaLN reference patch
# ---------------------------------------------------------------------------

_ADALN_SUFFIXES = ("adaln_proj.linear.weight", "adaln_proj.linear.bias")


def _parse_block_set(spec, n_blocks):
    """'30-49', '0-2,20,40-49', '' -> a set of block indices. Negatives from the end."""
    out = set()
    for piece in (spec or "").split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece[1:]:
            lo, _, hi = piece[1:].rpartition("-")
            lo = piece[0] + lo
        else:
            lo = hi = piece
        try:
            a, b = int(lo), int(hi)
        except ValueError:
            raise ValueError(
                "MMH3AdaLNRefPatch: could not read %r in `blocks`. Expected things like "
                "'30-49', '0-2,40', '-1'." % piece)
        if a < 0:
            a += n_blocks
        if b < 0:
            b += n_blocks
        if a > b:
            a, b = b, a
        if not (0 <= a < n_blocks and 0 <= b < n_blocks):
            raise ValueError(
                "MMH3AdaLNRefPatch: block range %d-%d is outside this model's %d blocks."
                % (a, b, n_blocks))
        out.update(range(a, b + 1))
    return out


class MMH3AdaLNRefPatch(io.ComfyNode):
    """Take AdaLN modulation from another H3 checkpoint, per block.

    fl2va and ref2va are the SAME model except for AdaLN. Measured on the int8
    checkpoints: attention, MLP, condition_proj, the patch projections and the
    output heads all sit at cosine 0.999+, while every adaln_proj lands between
    -0.42 and -0.91. AdaLN is where reference conditioning is routed into the
    residual stream, so that one component is the whole difference between "can
    condition on a reference" and "cannot".

    WHY THERE IS NO STRENGTH SLIDER. The two are ANTI-correlated at near-equal
    norms (272.8 vs 272.1 on block 25), so a linear blend cancels rather than
    mixes: at 0.5 the modulation collapses to ~32% of either endpoint and the
    model runs with most of its conditioning routing switched off. It would look
    like a broken merge, not a dial set halfway. Each block therefore takes one
    side or the other, which is what the published hybrid checkpoints do -- not
    out of caution, but because it is the only sound operation.

    Per-row and per-term controls are absent for a different reason: the
    difference is UNIFORM. Every modality row (video/cond/ref, text, audio) and
    every term (shift/scale/gate, msa and mlp) sits in the same -0.4 to -0.9
    band, so there is no sub-structure to isolate.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3AdaLNRefPatch",
            display_name="MMH3 AdaLN Reference Patch",
            category="MMH3Tools/model",
            description=(
                "Replace AdaLN modulation in chosen transformer blocks with another H3 "
                "checkpoint's -- the one component that differs between fl2va and "
                "ref2va, and the one that routes reference conditioning. Reads only the "
                "adaln_proj tensors from the source, not the whole model."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "source", options=folder_paths.get_filename_list("diffusion_models"),
                    tooltip="Checkpoint to take AdaLN from -- ref2va, to give an fl2va "
                            "base its reference routing. Only its adaln_proj tensors are "
                            "read (~100MB of a 20GB file), never the rest."),
                io.String.Input(
                    "blocks", multiline=False, default="25-49",
                    tooltip="Which blocks take the source's AdaLN. Ranges and lists: "
                            "'25-49', '0-2,40-49', '-1' for the last. The published "
                            "hybrids are 30-49 / 25-49 / 20-49 / 15-49, so those values "
                            "reproduce them without a download. Empty patches nothing."),
                io.Boolean.Input(
                    "final_layer", default=True,
                    tooltip="Also take final_layer.adaln_proj, the last modulation before "
                            "the output heads. Measured at cosine -0.830 between fl2va "
                            "and ref2va, and left untouched by the published hybrid "
                            "checkpoints."),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, model, source, blocks, final_layer=True) -> io.NodeOutput:
        import re
        import comfy.utils
        from safetensors import safe_open

        path = folder_paths.get_full_path_or_raise("diffusion_models", source)
        sd = model.model.state_dict()

        n_blocks = 0
        for k in sd:
            m = re.match(r"(?:diffusion_model\.)?blocks\.(\d+)\.", k)
            if m:
                n_blocks = max(n_blocks, int(m.group(1)) + 1)
        if not n_blocks:
            raise ValueError(
                "MMH3AdaLNRefPatch: this model has no `blocks.N.` keys, so it is not a "
                "MiniMax H3 transformer.")

        want = _parse_block_set(blocks, n_blocks)
        prefix = "diffusion_model." if any(
            k.startswith("diffusion_model.") for k in sd) else ""

        # Only the adaln tensors are read. They are unquantized -- no weight_scale
        # sibling -- so the delta is exact rather than something reconstructed
        # through int8 and then re-quantized.
        patches, missing, lines = {}, [], []
        with safe_open(path, framework="pt") as f:
            src_keys = set(f.keys())
            targets = []
            for b in sorted(want):
                targets += ["blocks.%d.%s" % (b, s) for s in _ADALN_SUFFIXES]
            if final_layer:
                targets += ["final_layer.%s" % s for s in _ADALN_SUFFIXES]

            for k in targets:
                if k not in src_keys:
                    missing.append(k)
                    continue
                mk = prefix + k
                cur = sd.get(mk)
                if cur is None:
                    missing.append(k)
                    continue
                new = f.get_tensor(k)
                if tuple(new.shape) != tuple(cur.shape):
                    raise ValueError(
                        "MMH3AdaLNRefPatch: %s is %s in the source but %s in the model. "
                        "These are different architectures, not two H3 checkpoints."
                        % (k, tuple(new.shape), tuple(cur.shape)))
                # ("diff", (delta,)) is core's raw-weight-delta patch; add_patches
                # applies it as weight + strength * delta, and strength 1.0 makes the
                # result exactly the source's tensor.
                delta = new.to(dtype=cur.dtype) - cur.cpu().to(dtype=cur.dtype)
                patches[mk] = ("diff", (delta,))

        m = model.clone()
        applied = m.add_patches(patches, 1.0) if patches else set()

        lines.append("AdaLN from %s" % source)
        lines.append("blocks %s of %d%s"
                     % (blocks or "(none)", n_blocks,
                        ", plus final_layer" if final_layer else ""))
        lines.append("%d tensors patched (%d requested)" % (len(applied), len(patches)))
        if missing:
            lines.append("! %d not found and skipped: %s"
                         % (len(missing), ", ".join(missing[:3])))
            logging.warning("[MMH3AdaLNRefPatch] %d target tensors missing", len(missing))
        if not patches:
            lines.append("! nothing patched -- `blocks` is empty and final_layer is off")
        report = "\n".join(lines)
        logging.info("[MMH3AdaLNRefPatch] %s", lines[0] + " | " + lines[2])
        return io.NodeOutput(m, report)
