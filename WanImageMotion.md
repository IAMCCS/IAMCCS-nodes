# IAMCCS WanImageMotion

`IAMCCS_WanImageMotion` is a **drop-in replacement** for the SVIPro latent-conditioning node used in WAN image-to-video workflows. Its purpose is to build the *conditioning* fields required by the WAN I2V pipeline while optionally boosting perceived motion via a controllable `motion` parameter.

This node **does not perform sampling**. It only:
- prepares an “empty” latent sequence to be denoised by the sampler, and
- injects `concat_latent_image` and `concat_mask` into both positive/negative conditioning.

---

## Inputs

Required:
- `positive` / `negative` (`CONDITIONING`): conditioning streams to be augmented.
- `length` (`INT`): number of frames in the video. Internally converted to latent-frame count:  
  $T = \left\lfloor\frac{length-1}{4}\right\rfloor + 1$.
- `anchor_samples` (`LATENT`): the “anchor” latent(s), typically representing the initial visual content.
- `motion_latent_count` (`INT`): how many latent frames to take from `prev_samples` (if present) to seed motion.
- `motion` (`FLOAT`): motion amplification factor. `1.0` means “no change”. Values > `1.0` increase motion.
- `motion_mode` (dropdown): chooses *where* the motion boost is applied.
- `latent_precision` (dropdown): controls the dtype used for the **empty latent** allocation (quality vs VRAM).
  - `auto`: matches anchor samples dtype
  - `fp16`: half precision (lower VRAM, slight quality loss)
  - `fp32`: full precision (higher VRAM, maximum quality)
- `vram_profile` (dropdown): chooses *how* the motion boost is computed to reduce peak VRAM.
  - `normal`: process all frames at once (fastest, highest VRAM)
  - `chunked_blocks_2` / `chunked_blocks_4`: process in chunks (balanced)
  - `loop_per_frame (lowest_vram)`: process one frame at a time
  - `cpu_offload (slowest)`: offload computation to CPU (extreme low VRAM)
- `include_padding_in_motion` (`BOOLEAN`): if enabled, the motion boost may also affect padded latent frames.
  - **Critical for single-frame anchors**: when `anchor_samples` has only `T=1` and there are no `prev_samples`, this must be `True` to apply any motion boost.
  - The node will log a warning if motion_range is empty and suggest enabling this option.

Optional:
- `prev_samples` (`LATENT`): previous latent sequence; when provided, the last `motion_latent_count` latent frames are appended after the anchor to seed motion.

---

## Outputs

- `positive` / `negative` (`CONDITIONING`): same as input, but with added conditioning keys:
  - `concat_latent_image`
  - `concat_mask`
- `latent` (`LATENT`): an **empty latent sequence** shaped like the target video latents. This is what the sampler will denoise.

---

## Core Logic

### 1) Create the empty latent sequence
The node allocates an empty latent tensor with shape:
- `[B, 16, T, H, W]` where `T` is derived from `length`.

This tensor is intentionally initialized to zeros.

`latent_precision` affects only this allocation:
- `auto`: matches the dtype of `anchor_samples` (recommended).
- `fp16`: forces FP16 (lower VRAM, can be slightly less stable).
- `fp32`: forces FP32 (higher VRAM, can be slightly more stable).

### 2) Build `concat_latent_image`
The node builds a latent conditioning sequence (`image_cond_latent`) by concatenating:
1. `anchor_samples["samples"]` (anchor latents)
2. the last `motion_latent_count` frames from `prev_samples["samples"]` (only if provided)
3. zero padding to reach exactly `T` latent frames

Padding is processed with `Wan21().process_out(...)` to match expected latent formatting.

### 3) Build `concat_mask`
A mask is created with shape `[1, 1, T, H, W]`.
- The first latent frame is unmasked: `mask[:, :, :1] = 0.0`
- All subsequent latent frames are masked: `1.0`

### 4) Inject into conditioning
The node injects:
- `concat_latent_image = image_cond_latent`
- `concat_mask = mask`

into **both** `positive` and `negative` conditioning.

---

## Motion Boost (`motion`)

When `motion > 1.0`, the node amplifies motion by modifying selected latent frames while preserving the per-frame mean offset to reduce brightness/shift artifacts.

Let:
- `base` be the first latent frame `image_cond_latent[:, :, 0:1]`
- `x` be the target latent frames to be modified

The transformation is:
1. `diff = x - base`
2. `mean = mean(diff over C,H,W)` (per-batch/per-time)
3. `diff_centered = diff - mean`
4. `scaled = base + diff_centered * motion + mean`
5. clamp to a safe range: `[-6, 6]`

By default, the node **does not modify padding frames**.

If `include_padding_in_motion = true`, the node may treat padded frames as motion targets. This can help when `anchor_samples` provides only a single latent frame (e.g. `T=1`) and there are no motion latents from `prev_samples`.

---

## Motion Mode (two modes)

### `motion_only (prev_samples)`
- Applies the motion boost **only** to the latent frames coming from `prev_samples`.
- Conservative: changes less of the anchor content.
- Recommended when you want motion injection without destabilizing the initial anchor.

### `all_nonfirst (anchor+motion)`
- Applies the motion boost to **all real latent frames except the first** (anchor + motion latents).
- More aggressive: stronger motion effect, but can change the look more.

---

## VRAM Profile

These profiles only change *how the motion boost is computed* (peak memory vs speed). They do not change the rest of the pipeline.

- `normal`: processes the selected time range in one tensor block (fastest, highest peak VRAM).
- `chunked_blocks_2`: processes 2 latent frames at a time (lower peak VRAM).
- `chunked_blocks_4`: processes 4 latent frames at a time (middle ground).
- `loop_per_frame (lowest_vram)`: processes 1 latent frame at a time (lowest peak VRAM, slower).
- `cpu_offload (slowest)`: moves the targeted slice to CPU for the computation, then copies back (lowest GPU peak, highest runtime cost).

---

## Notes / Troubleshooting

- If you are hitting CUDA OOM at high resolutions, try:
  1) `vram_profile = chunked_blocks_2`
  2) then `loop_per_frame (lowest_vram)`
  3) then (only if necessary) `cpu_offload (slowest)`

- If you want to isolate whether OOM is caused by motion scaling vs sampling, set `motion = 1.0` temporarily.
