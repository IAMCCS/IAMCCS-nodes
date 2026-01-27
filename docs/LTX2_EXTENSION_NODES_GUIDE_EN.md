# IAMCCS LTX-2 Extension Nodes — Final Guide (EN)

This document explains how to use the IAMCCS LTX-2 nodes for **long-length / multi-segment video generation and extension** in ComfyUI, including the purpose of each widget and recommended usage patterns.

## What problems these nodes solve

1. **Seam artifacts between segments** (visible cut, flicker, exposure shift)
2. **Bad seam position** (the extension starts at an awkward frame)
3. **LTX VideoVAE frame-count constraint**: some encode paths require the number of guide frames to be of the form:

$$N = 1 + 8k$$

4. **Workflow simplification**: reduce reliance on multiple helper nodes for overlap math, ranges, etc.

---

## Quick decision guide (what to touch first)

### If you get the LTX VideoVAE error: “Encode input must have 1 + 8 * x frames”

This is the **`8n+1`** rule: the number of frames going into certain LTX/LTXV encode paths must be:

$$N = 1 + 8k$$

In iterative extension workflows, this usually affects **the guide/start frames** you feed into the next segment.

Use these fixes in this order:

1) **Set `safe_mode = native_workflow_safe`** in `IAMCCS_LTX2_ExtensionModule`
   - This extracts the start frames exactly like the original stable workflow:
     - `start_images = extended_images[-overlap_frames:-1]`

2) If you still need a strict `8n+1` count, set **`start_frames_rule`**:
   - `ltx2_round_down`: most predictable and “never increases” the frame count.
   - `ltx2_nearest`: useful if you want the closest valid count (may go up or down).

3) If the frame rule is needed elsewhere (not on the extension module), use `IAMCCS_LTX2_FrameCountValidator` on the integer driving that node.

### If the seam is visible (hard cut / flicker)

- Start with:
  - `overlap_mode = ease_in_out` (or `linear_blend` if you want the simplest behavior)
  - keep overlap modest (common range: ~8–24 frames; larger overlap can help but costs compute/time)

### If exposure/white balance shifts at the seam

- Enable color matching:
  - `color_match_mode = luma_only` (usually the safest)
  - `color_match_strength = 0.3..0.7`
  - `color_reference_window = 6..12`

### If the seam “restarts weirdly” (bad timing / rewind)

- Use seam search (try only after overlap/blend):
  - `seam_search_mode = best_of_k`
  - `k_search = 8..24`

### If you are using AutoLink overlap loops

- Prefer wiring `autolink_overlap_in` / `autolink_overlap_out` so each iteration can override overlap cleanly.

### About `IAMCCS_LTX2_ExtensionModule_simple`

- `IAMCCS_LTX2_ExtensionModule_simple` is the **minimal** variant of the Extension Module.
- It exposes only the core overlap/blend/math widgets (no color match, seam search, metrics).
- It does **not** expose `safe_mode` or `start_frames_rule` as widgets.
- It **always enforces** the LTX-2 start-frame rule $N = 1 + 8k$ automatically (round-down), to avoid VideoVAE encode frame-count errors.

## Nodes overview

- `IAMCCS_LTX2_ExtensionModule`
  - Merges the previous segment (`source_images`) with the new segment (`new_images`) using overlap/blend.
  - Outputs `extended_images` (merged batch) and `start_images` (frames used to guide the next segment).
  - Optional seam improvements: exposure/color matching and best-of-k seam selection.
  - Optional “native safe” extraction that matches the original stable workflow behavior.
  - Optional AutoLink overlap loop I/O:
    - `autolink_overlap_in` (override overlap when > 0)
    - `autolink_overlap_out` (feed the next iteration)

- `IAMCCS_LTX2_GetImageFromBatch`
  - Extracts frames from the start/end of an image batch, or by an explicit range.
  - Adds optional auto-count and diagnostics outputs.
  - Optional “native safe” mode matching `images[-count:-1]` in from-end mode.

- `IAMCCS_LTX2_ReferenceImageSwitch`
  - Safe way to inject a **reference image** to improve identity/style consistency **without breaking overlap continuity**.
  - Default is `none`, so existing workflows are unchanged.

- `IAMCCS_LTX2_ReferenceStartFramesInjector`
  - (New) Injects/blends the reference directly into the **guide/conditioning frames** (`start_images` / segment `images`).
  - Useful when feeding the reference into `image_1` (empty latent image) has **weak or no identity effect**.
  - Can be applied to **only one segment** (e.g. segment 3 only).

- `IAMCCS_LTX2_FrameCountValidator`
  - Helper to validate/correct an integer frame count to the `1 + 8*k` rule.

---

## 1) IAMCCS_LTX2_ExtensionModule

### Inputs

**Required**

- `source_images` (IMAGE)
  - The current accumulated batch (previous segment output).
- `overlap_frames` (INT)
  - How many frames overlap between segments.
- `overlap_side` (dropdown)
  - `source`: overlap uses the tail of `source_images` against the head of `new_images`.
  - `new_images`: swaps which side is treated as source/destination for blending.
- `overlap_mode` (dropdown)
  - `cut`: hard cut (fastest, most visible seam).
  - `linear_blend`: linear crossfade.
  - `ease_in_out`: smoother crossfade.
  - `filmic_crossfade`: gamma-aware blend (often smoother in highlights).
  - `perceptual_crossfade`: LAB blend via Kornia (falls back if Kornia not installed).
- `enable_math` (BOOLEAN)
  - Enables the built-in “how many start frames to output” calculation.
- `math_operation` (dropdown)
  - Applies to `overlap_frames` (as `a`) and `math_value_b` (as `b`) when computing how many frames to output as `start_images`.
  - Typical: `a-b` or `a-1`.

**Safety / LTX rule**

- `safe_mode` (dropdown)
  - `none`: uses the node’s normal start-images logic.
  - `native_workflow_safe`: extracts start images exactly like the proven stable graph:
    - `start_images = extended_images[-overlap_frames:-1]`
  - Use this if you are hitting the LTX VideoVAE error “Encode input must have 1 + 8 * x frames”.

- `start_frames_rule` (dropdown)
  - `none`: do not modify the calculated number of start frames.
  - `ltx2_round_down`: force the count down to the nearest valid `1 + 8*k`.
  - `ltx2_nearest`: choose the nearest valid `1 + 8*k` within bounds.
  - Use this when a downstream node (VideoVAE encode/guide) requires `1 + 8*k` frame counts.

**Quality upgrades (defaults are safe/off)**

- `color_match_mode` (dropdown)
  - `none`: no change (original behavior).
  - `luma_only`: match exposure/contrast on luma.
  - `per_channel`: match mean/std per RGB channel.
- `color_match_strength` (FLOAT 0..1)
  - Blend between original and matched.
- `color_reference_window` (INT)
  - Number of frames used from tail/head for statistics.

- `seam_search_mode` (dropdown)
  - `none`: no seam search.
  - `best_of_k`: search for a better seam by testing candidate offsets.
- `k_search` (INT)
  - How many candidate offsets to test (0 disables).
- `metric_weight_color` (FLOAT)
  - Weight of luma continuity in the seam score.
- `metric_weight_edges` (FLOAT)
  - Weight of edge continuity in the seam score.

**Optional**

- `new_images` (IMAGE)
  - The newly generated segment.
  - If omitted, the node can be used as a “prep” node (it will still output `start_images` from the current batch).
- `math_value_b` (INT)
  - Used by `math_operation`.

### Outputs

- `source_images` (IMAGE) — passthrough
- `start_images` (IMAGE) — frames to feed as guide for the next segment
- `extended_images` (IMAGE) — merged batch
- `overlap_frames` (INT)
- `calculated_frames` (INT) — actual number of frames output in `start_images`
- `extension_frames` (INT) — how many frames were added
- `report` (STRING)

### Recommended settings

- Most stable: `safe_mode = native_workflow_safe`, `overlap_mode = ease_in_out` (or `linear_blend`)
- If you see exposure shift: `color_match_mode = luma_only`, `strength = 0.3..0.7`
- If you see weird seam timing: `seam_search_mode = best_of_k`, `k_search = 8..24`

---

## 2) IAMCCS_LTX2_GetImageFromBatch

### Purpose
A small helper to extract frames for the next segment or for debugging.

### Inputs

- `images` (IMAGE)
- `mode` (dropdown)
  - `from_start`: take the first `count` frames
  - `from_end`: take the last `count` frames
  - `range`: take `[start_index:end_index)`
- `count` (INT)

**Upgrades**

- `auto_count_mode` (dropdown)
  - `none`: use `count` widget.
  - `prefer_input`: use `count_in` if connected.
  - `use_widget`: explicitly use the widget value.
- `diagnostics` (dropdown)
  - `none`: normal behavior.
  - `basic`: exposes `start_index` and `end_index` outputs.

**Safety / LTX rule**

- `count_rule` (dropdown)
  - `none` / `ltx2_round_down` / `ltx2_nearest` for `1 + 8*k`.
- `safe_mode` (dropdown)
  - `none`: normal extraction.
  - `native_workflow_safe`: for `from_end` uses `images[-count:-1]`.

**Optional**

- `count_in` (INT)
- `start_index` / `end_index` (INT) for `range` mode.

### Outputs

- `images` (IMAGE)
- `count` (INT)
- `report` (STRING)
- `start_index`, `end_index` (INT)

---

## 3) IAMCCS_LTX2_ReferenceImageSwitch

### Why this node exists
In long-length generation, you typically want:
- **Continuity** driven by overlap/start frames
- **Identity/style consistency** reinforced by a stable reference image

This node lets you add a reference image **without replacing** the overlap continuity input.

### Inputs

- `default_image` (IMAGE)
  - What the workflow already used before (pass-through by default).
- `mode` (dropdown)
  - `none`: output `default_image` (fully backward-compatible).
  - `use_reference`: output `reference_image`.
  - `blend`: output mix of `default_image` and `reference_image`.
- `blend_strength` (FLOAT)
  - Only for `blend` mode.
- `reference_image` (optional IMAGE)
  - If not connected, the node behaves like `none`.

### Output

- `image` (IMAGE)
- `report` (STRING)

### Practical usage

- Insert it on the **auxiliary** image input of your segment sampler (often called `image_1`).
- Keep overlap/start frames connected exactly as before.
- If you enable `use_reference`/`blend`, the reference is **automatically resized** to match `default_image` (more stable for downstream nodes).

Note: in many LTX/LTXV workflows, feeding the reference into `image_1` (empty latent image) may not be enough to “lock” identity when a face is revealed later in the segment. In that case, use the node below.

---

## 3b) IAMCCS_LTX2_ReferenceStartFramesInjector

### Why it exists
If identity drifts even with a reference, it often means the reference is connected to an input that the model barely uses. This node modifies the actual guide/conditioning frames.

### Inputs

- `start_images` (IMAGE)
  - The guide frames that feed the segment (typically `start_images` from the extension module, or the sampler’s `images` input).
- `mode`
  - `none`: passthrough.
  - `inject`: replaces the selected frames with the reference.
  - `blend`: mixes reference and original frames.
- `blend_strength` (0..1)
  - Only used for `blend` (0 = no effect, 1 = full reference). In `inject` it behaves like 1.
- `frames_to_inject` (INT)
  - How many guide frames to modify.
- `ramp` (BOOLEAN)
  - If `true`, applies a gradual ramp across the injected frames.
- `position`
  - `tail`: last K frames (usually best, closest to the seam).
  - `head`: first K frames.
- `reference_image` (optional IMAGE)
  - Usually the output of `IAMCCS_LTX2_ReferenceImageSwitch`.

### Outputs

- `start_images` (IMAGE)
- `report` (STRING)

### Recommended starter settings

- If identity is not sticking but you want to preserve continuity:
  - `mode = blend`
  - `frames_to_inject = 3..6`
  - `blend_strength = 0.5..0.85`
  - `ramp = true`
  - `position = tail`

If you see seam discontinuity, lower `blend_strength` and/or reduce `frames_to_inject`.

---

## How to decide when/where to use a reference

Quick checklist:

1. **Is the face/identity visible in the first frames of the segment?**
   - Yes → a reference can work well.
   - No (reveal happens mid/late segment) → the reference may have little leverage: consider cutting segments so the reveal starts at the segment boundary, or use `ReferenceStartFramesInjector` (and/or dedicated tools like FaceID/IPAdapter if compatible).

2. **What are you stabilizing?**
   - Style / global look → `ReferenceImageSwitch` (or `color_match_mode` in ExtensionModule) is often enough.
   - Identity (specific face) → `ReferenceStartFramesInjector` is more likely required.

3. **Where to wire it?**
   - `image_1` / empty latent image: can be a hint, not guaranteed.
   - `images` / start frames (conditioning): highest impact.

4. **How to limit it to one segment (e.g. segment 3 only)**
   - Place `ReferenceStartFramesInjector` only in the path feeding that segment’s `images` / `start_images`.
   - Leave other segments untouched (no injector).

## 4) IAMCCS_LTX2_FrameCountValidator

### Inputs

- `frame_count` (INT)
- `auto_correct` (BOOLEAN)
- `correction_mode` (`nearest` / `round_up` / `round_down`)

### Outputs

- `validated_count` (INT)
- `is_valid` (BOOLEAN)
- `nearest_valid` (INT)
- `report` (STRING)

---

## Common workflows / use cases

### A) Long-length extension (multi segment)
1. Generate segment 1.
2. Use `IAMCCS_LTX2_ExtensionModule` to compute `start_images` and merge segments.
3. Feed `start_images` into the next segment guide/conditioning.
4. Repeat.

Recommended: enable `safe_mode = native_workflow_safe` if you see LTX frame-count errors.

### B) Reduce seams
- Prefer `ease_in_out` or `filmic_crossfade`.
- Use `color_match_mode` if you see exposure shifts.
- Use `best_of_k` seam search if the seam starts at a bad moment.

### C) Improve identity consistency
- Add `IAMCCS_LTX2_ReferenceImageSwitch` to `image_1`.
- Connect a single reference image and set mode to `blend` (start at 0.2..0.4).

---

## Troubleshooting

- **“IAMCCS_LTX2_ReferenceImageSwitch not found”**
  - Ensure you updated the IAMCCS nodes and restart ComfyUI.
  - The node must be exported in the package registry (`__init__.py`).

- **“Encode input must have 1 + 8 * x frames”**
  - Use `safe_mode = native_workflow_safe` or set `start_frames_rule/count_rule` to enforce `1 + 8*k`.

- **Border motion artifacts (edge warping / flicker)**
  - Note: `metric_weight_edges` and `best_of_k` improve seam selection *inside the overlap* between segments; they do not automatically “fix” frame borders.
  - Common improvements:
    - Avoid changing resize/crop between segments; keep one resolution end-to-end.
    - Prefer “clean” resolutions (multiples of 64 where possible) to reduce VAE boundary artifacts.
    - Quick workaround: apply a small crop (e.g., 8–16 px per side) then resize back.
  - Helpful nodes (IAMCCS):
    - `IAMCCS_LTX2_ImageBatchPadReflect`: adds a reflect border (increases resolution).
    - `IAMCCS_LTX2_ImageBatchCropByPad`: removes that border (back to target resolution).
  - Recommended usage (when you want the model to have more border context):
    - Pick `pad_x/pad_y` (e.g., 16).
    - Generate at a higher resolution: `W_pad = W + 2*pad_x`, `H_pad = H + 2*pad_y` (including `EmptyImage`).
    - If you have “initial”/reference images at the old resolution, run them through `PadReflect` to reach `W_pad x H_pad`.
    - At the end (before `CreateVideo`), run `CropByPad` with the same `pad_x/pad_y` to return to `W x H`.

- **Reference image causes a resolution error**
  - Resize/crop the reference to match your workflow resolution before feeding it.
