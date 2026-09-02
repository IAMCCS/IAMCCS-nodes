# MMH3Tools

MiniMax H3 latent tooling for ComfyUI — latent-domain conditioning and correct AV
splicing for **chained long-form generation**.

## What has actually been run

**`carry="mask"` is the tested path.** Every example workflow ships with it, and every
observation in the docs was measured on it.

**`carry="keyframe"` has never generated a clip.** The plumbing is unit-tested — the
guide is built and anchored at frame 0, it carries a multi-step clip *plus its audio*
rather than a still, no mask is set, and the node refuses up front on a core without
#15439. But that is structure checked against a fake sampler, not output anyone has
looked at.

It stays because it is written, gated and harmless where it sits — not because it is
recommended. If you switch `carry` to `keyframe`, you are the first, and the seam is
the thing to watch.

## Requirements

**Stock ComfyUI, `v0.33.0-20-gff6c8a8a` or newer.** For everything except the
ControlNet workflow: no patches, no carried diffs.

Note that current ComfyUI also raised its own floor to **`av>=17.0.0`**. If you update
core and ComfyUI then fails to start with `cannot import name 'ColorPrimaries' from
'av.video.reformatter'`, that is why — `pip install --upgrade "av>=17.0.0"`.

Everything this pack needs from upstream has merged:

| PR | merged | needed by |
|---|---|---|
| [#15375](https://github.com/Comfy-Org/ComfyUI/pull/15375) per-token masking | 2026-08-18 | `MMH3SeedOverlap`, latent outpaint, and **`MMH3LoopingSampler` with `carry="mask"`** — the default |
| [#15439](https://github.com/Comfy-Org/ComfyUI/pull/15439) guides at any frame | 2026-08-13 | `MMH3LoopingSampler` with `carry="keyframe"`, and any use of `keyframes` |
| [#15808](https://github.com/Comfy-Org/ComfyUI/pull/15808) H3's seven special tokens | 2026-08-22 | nothing in the pack — it makes `MMH3OfficialTokens` redundant, see below |

On an older ComfyUI the pack does not pretend. `MMH3SeedOverlap` and the keyframe path
**refuse to run**, and the looping sampler checks before it starts.

> ⚠️ **One exception, and it is the default path.** `carry="mask"` is *not* gated on
> older cores: the mask is accepted and ignored, preserved rows still run at the
> generation timestep, and you get seams with no error anywhere. If chunks are not
> carrying, check your ComfyUI version first. `per_row_mask_is_continuous()` reports
> what the installed core actually does.

### One thing is NOT merged: the Fun ControlNet

**`MMH3CondSetApplyControl` and `MMH3_Looping_I2V_ControlNet.json` need
[PR #15860](https://github.com/Comfy-Org/ComfyUI/pull/15860) applied to your ComfyUI.**
It is kijai's, and it is a **draft** — the only carried diff this pack asks for, and
the only one that can move under you. Everything else in the pack runs on stock.

```
git -C /path/to/ComfyUI apply <(curl -sL https://github.com/Comfy-Org/ComfyUI/pull/15860.diff)
```

Without it the node **refuses with a message** rather than half-working: it checks for
`MiniMaxH3ControlNet` and for each internal it windows, and says which is missing.

It also needs weights — the union checkpoint, ~2.1 GB, into `models/controlnet/`:

| file | from |
|---|---|
| `minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors` | [Kijai/MiniMax-H3-experimental](https://huggingface.co/Kijai/MiniMax-H3-experimental/tree/main/controlnet) |

Prefer `int8_convrot` on torch cu130; there is a `bf16` (3.9 GB) in the same folder.
One checkpoint covers **Canny, Depth, HED, MLSD, Pose** and video inpainting — the
control branch is `control_proj_in` plus 5 `control_blocks`, attaching to layers 0, 10,
20, 30 and 40 of the 50 through zero-gated projections.

Three things about it that are easy to get wrong:

- The checkpoint takes an **already-detected** control video — canny/depth/HED/MLSD/pose
  passes, not raw footage. The example workflow does that inline with
  `AIO_Preprocessor` from
  [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux), so you
  feed it ordinary video; wire your own detector if you prefer, or set the
  preprocessor to `none` when the pass is already rendered.
- The control video must cover the **whole clip**, not one chunk. Windows are cut from
  it by frame index, and a short one clamps to its last frame rather than erroring.
- It is **guidance-distilled**: run at guidance 1.0 through a `BasicGuider`, not CFG.

What #15375 gives the pack is three things, not one: the mask reaches the model as a
cond, preserved rows run at the cond timestep, and `MiniMaxH3` gets a
`scale_latent_inpaint` override that stock `BaseModel` has no equivalent for. The
third only matters for *intermediate* mask values. Full account under
[Latent joins happen in pixel space](#latent-joins-happen-in-pixel-space).

### One runtime patch, inert on current core

`mmh3tools/patch_guide_origin.py` wraps `PackedLayout` at import — no core edit, and it
survives `git pull`. **On any ComfyUI new enough to run this pack it does nothing:**
merged #15439 anchors the guide correctly by itself, so the wrap's self-test finds
nothing to fix, rolls back, and logs that it stood down. `is_applied()` returning
False is the success case.

It remains because the failure it prevents is silent. The **draft** #15439 anchored a
guide at `text_len`, but the target does not begin there — references advance a cursor
first, so a guide landed *before* the clip it was meant to anchor, measured at −1 for
one image ref and −321 for image+audio. Nothing errored; the guide just landed in the
reference region and a carried tail's audio went early with it. The looping sampler
asks `is_applied()` and refuses when a chunk carries both a reference and a keyframe on
a build that needs the fix and lacks it.

## Why this exists

Three facts about H3 shape everything here:

1. **References are latents that are never denoised.** `PackedLayout` packs them
   with `update=False`, so they are re-injected at every sampling step as pure
   context. There is no shared region between chunks to blend.
2. **The stock reference node takes pixels** and calls `vae.encode()`. In a chain
   the previous chunk is already latent, so that roundtrip is generation loss
   compounding once per hop.
3. **Video and audio latents have different temporal axes.**

   | tensor | shape | temporal dim |
   |---|---|---|
   | video | `[B, 24, T, h, w]` | **2** |
   | audio | `[B, 32, 2, T40]` | **3** (dim 2 is stereo) |

   Generic nested-tensor helpers that assume one shared temporal dim will stack
   audio on its stereo axis — producing 4 channels at unchanged duration instead
   of a longer clip. It fails silently.

## Example workflows

[`workflows/MMH3_Scene_Prompt_Builder.json`](workflows/) — the prompt half on its
own: N chunk prompts written **section by section**, ending at a pipe-separated
string ready for **MMH3 Reference (Multi-Prompt)**. No sampler, no VAE, no weights —
it runs against an LLM server alone, so you can iterate on a film's prompts without
paying for a generation to find out they were wrong.

Three stages, `1 + 1 + N` LLM calls: definitions once, the whole beat sheet once,
then one call per chunk for its shots. See **MMH3 Scene Plan Prompt** below for why
that shape rather than a prompt per chunk.

Every `MMH3WindowPlan` input is derived from a duration rather than typed in —
60s total, 10.7s window, 2s overlap — and the window's `actual_seconds` drives
`seconds_per_chunk` on all three stages **and** the lint, so the writer and the
checker cannot disagree about how long a chunk is.

Needs [ComfyUI-LlamaOmni](https://github.com/ckinpdx/ComfyUI-LlamaOmni) and
ComfyUI-Easy-Use (the for-loop). The model names on the `Llama Connectivity` nodes
are local llama-swap ids — swap them for yours. Two of them matter:
`unload_after` is ON for the one-shot definitions call so its model frees VRAM for
the next, and OFF for beats and shots, which share a model that should stay resident
across every iteration of the loop.

[`workflows/MMH3_Looping_Cinematic.json`](workflows/) — the looping T2V film
pipeline, and the successor to the old `MMH3_Looping_T2V`. Definitions, beat sheet
and shots are written **section by section** (three **MMH3 Scene Plan Prompt**
stages), then the loop re-details each chunk against the **previous chunk's rendered
output** — the second **MMH3 Prompt Accumulate** carries that continuity forward — so
each clip continues the last instead of restarting. **MMH3 Prompt Lint** guards chunk
length against the window.

[`workflows/MMH3_Looping_Monologue.json`](workflows/) — the same looping backbone in
talking-head mode: one continuous monologue, absolutely-locked camera, no story arc.
A stress test of whether the looping sampler joins chunks seamlessly when there is no
cut to hide the seam. **MMH3 Chunk Schedule** paces the spoken-word budget per window.

[`workflows/MMH3_Looping_I2V_PromptBuilding.json`](workflows/) — image-to-video that
builds its own prompts (**MMH3 Load Skill** + an LLM) ahead of a three-window looping
sampler, finishing on a chunked pixel-upscale ladder.

[`workflows/MMH3_Looping_I2V_ManualPrompt.json`](workflows/) — the same
image-to-video start with the prompt **typed, not generated**: no LLM anywhere in the
graph, just a `PrimitiveStringMultiline` feeding **MMH3 Reference MultiPrompt**. Use it
when you already know the shot and want the prompt-building half out of the way.

It is also the fullest example of the **three-stage ladder**. One looping sampler
generates at 192-frame chunks, then two more refine passes each sit behind their own
**MMH3 Chunked Pixel Upscale** (`rtx_vsr`, 2688x1536) at the size **MMH3 Upscale
Ladder** picks. The last pass splits the pair and pins the audio under a **zero**
`SolidMask` before re-packing, so only the picture is resampled and the track that
came out of stage one survives both passes untouched. Finishes on a streaming save
plus a size-capped copy.

[`workflows/MMH3_Looping_RefVideo_Chunked.json`](workflows/) — ManualPrompt driven by a
**windowed video reference**. `window_ref_video` is on, and `chunk_frames` /
`overlap_frames` come from the **same MMH3 Chunk Schedule the sampler reads**, so
reference window *i* is by construction the span chunk *i* renders. The loader's audio
goes to `ref_video_audios` alongside, cut on the same clock.

Do not type those two numbers here. If they drift from the sampler's, the windows stop
matching the chunks and every chunk conditions on somebody else's footage — no error,
plausible output.

The still on `ref_images` is left wired and stays whole: `<Picture 1>` for identity,
`<Video 1>` for the windowed motion. Unwire it for a pure video reference.

[`workflows/MMH3_Looping_I2V_ControlNet.json`](workflows/) — the ManualPrompt graph
with a **Fun ControlNet** driving the generate pass. A `ControlNetLoader` and a control
video feed **MMH3 Cond Set Apply ControlNet**, which sits between the multiprompt node
and the sampler's `cond_set`, so every chunk gets the control windowed to its own span.

The two refine passes are deliberately left alone: they run at low denoise off zeroed
conditioning, where a control video would be fighting a picture that already exists.

Needs [PR #15860](https://github.com/Comfy-Org/ComfyUI/pull/15860) applied (a draft)
and the union checkpoint from
[Kijai/MiniMax-H3-experimental](https://huggingface.co/Kijai/MiniMax-H3-experimental/tree/main/controlnet).
Raw video goes in: an `AIO_Preprocessor` sits between the loader and the apply node,
defaulting to **DepthAnythingV2** at 768, and its dropdown covers all five conditions
the checkpoint accepts. Set it to `none` if your pass is already rendered. The video
must cover the WHOLE clip either way, since windows are cut from it by frame index. One checkpoint covers all five conditions plus inpainting. It is
**guidance-distilled**, so run it at guidance 1.0 through a BasicGuider rather than CFG.

[`workflows/MMH3_LoopingSampler_MusicVideo.json`](workflows/) — the music-video
variant: windows are locked to musical beats and lyrics mapped per window, feeding the
looping sampler; full-res and size-capped saves. Its three character references are
carried by a chained **`ImageTensorList`** rather than a batch, so each keeps its own
size — the loaders are `resize: False` and two of the three are crops, so batching was
conforming references 2 and 3 to reference 1's frame.

Its prompts are written by **one** local model across three staged calls — definitions,
then beats, then shots — each with its own `Llama Connectivity` and `Llama Options`.
The graph ships pointing at `qwen3.6-fable-27b-uncensored-vision`, which is a
llama-swap id: swap it for yours. The load is split across three calls deliberately
rather than asking one call to produce every prompt at once, which is also what makes a
smaller model viable — a community report has **Qwen3-VL-4B-Instruct-Q8_0** working
through the same three stages. Contrast **Scene Prompt Builder**, which uses two
different models on purpose.

[`workflows/MMH3_LoopingSampler_Regenerate2K.json`](workflows/) — the Regenerate-2K
path: a looping 2K pass over an existing render, driven by the dedicated **MMH3
Regenerate 2K Dims** / **Reference** nodes.

### What the example workflows need

Beyond this pack and comfy-core:

| | needed by |
|---|---|
| [RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF) — `ClownSampler_Beta`, sigma nodes | all but Scene Prompt Builder |
| [KJNodes](https://github.com/kijai/ComfyUI-KJNodes) — `VAELoaderKJ`, `LoadAndResizeImage`, `VRAM_Debug` | all but Scene Prompt Builder |
| [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) — `VHS_LoadVideo`, `VHS_LoadAudioUpload` | all but Scene Prompt Builder |
| [ComfyUI-LlamaOmni](https://github.com/ckinpdx/ComfyUI-LlamaOmni) — the prompt-writing nodes | every workflow that builds its own prompts |
| [ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use) — `easy forLoopStart` / `forLoopEnd` | the prompt-loop workflows |
| [rgthree-comfy](https://github.com/rgthree/rgthree-comfy) — Fast Groups Bypasser | MusicVideo, I2V Prompt Building, I2V ControlNet |
| [ComfyUI-MelBandRoFormer](https://github.com/kijai/ComfyUI-MelBandRoFormer) — vocal separation | MusicVideo only |
| [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) — `AIO_Preprocessor` | I2V ControlNet only |

**I2V ControlNet needs more than node packs**: [PR #15860](https://github.com/Comfy-Org/ComfyUI/pull/15860)
applied to core (a draft) and the union checkpoint in `models/controlnet/`. See
[Requirements](#requirements). The detection itself is handled in the workflow by
`AIO_Preprocessor`, whose dropdown covers all five conditions the checkpoint accepts —
`CannyEdgePreprocessor`, `DepthAnythingV2Preprocessor`, `HEDPreprocessor`,
`M-LSDPreprocessor`, `OpenposePreprocessor`/`DWPreprocessor` — so raw video in is fine.

**`SolAttnMiniMax` is Kijai's single-file Sol-Attn node**
([arXiv 2607.24027](https://arxiv.org/abs/2607.24027)), which reaches H3's attention
through comfy-kitchen's CUDA kernels — it wants `comfy_kitchen` built with `sol_attn`
(bf16, head_dim 128, sm_80+) and otherwise falls back to the existing backend. Eleven
of the twelve workflows carry it because it is a speed override, not pipeline logic.

The workflows here are on **v3**, the file linked at the bottom of
[comfy-kitchen PR #117](https://github.com/Comfy-Org/comfy-kitchen/pull/117). It keeps
the same `node_id` as v2 but the input list changed, so the two files cannot coexist
and v2's widget values do not carry over positionally — Kijai's own instruction was to
remake the node. `tau` now lives under a **`selection`** dynamic combo offering
`adaptive tau` (tau + `tau_profile`) or `top-k (SLA)` (`keep_percent`, the selection
lightx2v's SLA turbo LoRAs were distilled against). `routed_cap_percent` is gone.
These graphs ship on `adaptive tau` at the same 1.3 they used before.

**If you do not have it, delete the node** and wire `ModelAttentionBackend` straight
through to the LoRA loader. Nothing else in the graph depends on it.

The prompt nodes are easy to swap for your own — see the Note on each canvas.

[`workflows/MMH3_Looping_Upscale.json`](workflows/) — a refine pass over an **existing
render** rather than a generation. The clip is encoded, run through **MMH3 Chunked
Pixel Upscale** (`rtx_vsr`) at the target **MMH3 Upscale Ladder** picks from an aspect
preset, then **MMH3 Split AV** takes the pair apart so the audio half can be re-packed
under a **zero** `SolidMask` — held, never sampled — while the looping sampler
re-samples only the video at the higher resolution. Finishes on a streaming save plus
a size-capped copy. Distinct from Regenerate-2K, which drives the dedicated
Regenerate-2K nodes instead of the ladder.

[`workflows/MMH3_LoopingSampler_Masking.json`](workflows/) — masked v2v over an
existing clip. **SAM3 Detect** (core) mattes a subject, KJNodes' **GrowMaskWithBlur**
softens it, and the result drives the looping sampler's `denoise_mask`, so the matted
region re-renders while the rest of the frame is pinned to the encoded source.

It is also the reference for how the two halves are masked **independently**. The
supplied track is protected by `use_input_audio`, whose pin rides through **MMH3 Split
AV** (`preserve_masks`) and back through **MMH3 Pack AV** without the sampler being
told anything about it. Nothing is wired into `audio_denoise_mask`: a subject matte is
white somewhere in every frame and carries no temporal intent, so it must not be
allowed to reach the audio half. Wire a mask there only to freeze or free a **span**.

[`workflows/MMH3_Outpaint.json`](workflows/) — reframing, not generation: a landscape
clip is encoded with **MMH3 Streaming Encode**, extended in the latent by **MMH3
Outpaint Latent**, and **MMH3 Reframe Pads** solves the pad geometry for a 9:16 target.
**MMH3 Context Windows** carries one sampler pass across the whole clip instead of
chunking it through the looping sampler.

## Nodes

In the Add Node menu these are filed under `MMH3Tools/…`, following the same layout
as LTXAVTools: `sampling`, `calculators`, `prompt`, `conditioning`, `reference`,
`latent`, `audio`, `utils`, with the two plain calculators at the root. The headings
below group by what a node is *for*, which is close but not identical — the menu path
for any node is in its tooltip.

### Conditioning
- **MiniMax H3 Latent to Reference** — carry a chunk's tail forward as a
  `minimax_refs` block, no VAE roundtrip. `ref_downscale` is the cost lever:
  reference tokens are attended at *every* step, so 2× cuts their cost ~4×.
- **MMH3 Regenerate-2K Reference** — the second pass of a 768p → 2K run, with the
  reference **sliced per window**. A cond_set is already per chunk and the sampler
  passes `minimax_refs` straight through, so a reference attached to cond *i* reaches
  chunk *i* and nothing else — the slicing is a build-time concern and the sampler
  needs no changes. That matters because reference tokens ride every sampling step:
  handing the whole clip to every chunk multiplies that by the chunk count, and on a
  12-window clip slicing measured ~9.9× less reference attention per chunk.

  Feed it stage 1's own `cond_set` and each window keeps **its own** prompt while
  gaining its own reference; a single `conditioning` replicates one to all of them.
  Latent-only, like Latent to Reference — the reference never reaches the text
  encoder, so nothing is decoded and the CLIP is never touched in the 2K pass. That
  also happens to be the right semantics: MiniMax's `base_video` role carries no
  prompt label either, because the prompt is the *original* one and never mentions
  the 768p.

  It **conditions** the pass without seeding it — see
  [Refine vs regenerate](#refine-vs-regenerate). Mechanics, dimensions, audio pinning
  and the open divergence past one chunk are in
  [`docs/regenerate-2k.md`](docs/regenerate-2k.md).
- **MiniMax H3 Image to Reference** — append a still to `minimax_refs`. Fills the
  last hole in the matrix: latents could become refs or keyframes and images could
  become keyframes, but nothing put an image into refs *by appending*. Stock
  `MiniMaxH3ReferenceToVideo` takes `ref_images` but BUILDS conditioning from
  clip+prompt, so it can't add a still alongside carried latent refs.

  Unlike keyframes, reference blocks carry their own `latent_h`/`latent_w`, so this
  is free to resize. `match` scales to the generation's pixel area; `max` uses a
  2048px short edge for best identity — on a 3000×4000 source that's 5440 tokens per
  step against 999, paid at every step of every window.

- **MiniMax H3 Latent Keyframe** — first/last frame anchor from a latent frame.
  Shares the *target* spatial grid, so the source must match generation
  dimensions exactly.
- **MiniMax H3 Image Keyframe** — the same anchor from a **still image**.
  Resizes and encodes internally, precisely because keyframe rows cannot be
  downscaled; a still encoded at the wrong size fails deep in the model with an
  unhelpful broadcast error. Both keyframe nodes *append*, filling a gap in
  `MiniMaxH3ReferenceToVideo`, which has no keyframe inputs of its own.

  **Fixed in core as of the #15439 merge (2026-08-13).** `extra_conds` used to
  assign `cond_video_latents` from keyframes and then assign it *again* from
  references, so the references won and every keyframe was silently dropped. Core
  concatenates now. On a ComfyUI predating the merge, keyframes and references still
  cannot coexist.

  `frame_index` accepts `0` or `-1` only, because stock `PackedLayout` raises
  *"only first/last keyframe anchors are supported"* and the node refuses rather
  than failing deeper in. MiniMax's guide lists interior anchors as valid and they
  do work; the merged **#15439** removes the restriction in core, and the **Looping
  Sampler** exposes it as `keyframe_indices`.

### Sequences
- **MiniMax H3 Reference (Multi-Prompt)** + **MMH3 Cond Select** — the stock
  reference node with N prompts. For a text-driven sequence with locked identity,
  every chunk shares one reference set and differs only in its prompt.

  The win is the **model swap**, not the encode. Qwen3-VL-32B and a 33B DiT can't
  be resident together in 32GB, and ComfyUI resolves outputs depth-first, so N
  chunks in a naive graph run `load TE → cond → evict → load DiT → sample → evict
  → …` N times. One node execution collapses that to a single swap for the whole
  sequence, and the references are resized and encoded once instead of N times.

  Per-prompt memoization means editing one prompt re-encodes only that prompt.
  Swapping a reference invalidates all of them.

  **`ref_images` takes a batch OR a list, and the difference is not cosmetic.** A batch
  is one tensor and a tensor cannot be ragged, so every batching node — core's
  `ImageBatch`, KJNodes' `ImageBatchMulti` — resizes and **centre-crops** every image to
  the first one's frame before this node runs. References of different shapes are
  already cropped by then and nothing here can undo it. Wire a **list** instead — use
  **MMH3 Image List**, an Autogrow of image sockets (up to 50) that emits one — and each
  reference keeps its native size and gets its own aspect-correct target, which is what
  core's per-socket node does. KJNodes' `ImageTensorList` also emits a list but takes
  exactly two inputs, so N references need N-1 chained nodes. Either way the node logs each reference's incoming and resolved size, and
  says so when a multi-image batch arrives.

  **`window_ref_video` cuts the reference video to each chunk's own span.** Off by
  default and byte-identical when off. On, the node encodes one reference set per
  chunk instead of one for the sequence, so chunk *i* is conditioned on the footage it
  is actually rendering. The spans come from the **same `_plan`** the sampler and
  Window Plan run, so reference window *i* is by construction the span chunk *i*
  renders — wire the sampler's own `chunk_frames` and `overlap_frames`, or they stop
  matching and every chunk conditions on somebody else's footage.

  The soundtrack is windowed with it, cut on the **same clock** (seconds) rather than
  by latent arithmetic — 24 fps against 40 Hz is not additive, so cutting each from
  seconds is exact where deriving one from the other accumulates drift. Measured 0.00
  frames of drift across four windows.

  It costs N text-encodes rather than one, but **inside a single text-encoder load** —
  N forward passes, not N model swaps, which is the thing this node exists to protect.
  The vision work is partitioned, not duplicated. And at sampling time it is *cheaper*:
  reference tokens are attended at every step, and a window is a fraction of the whole
  reference — measured at **~32%** for a 600-frame reference in four windows.
  Timestamps restart at 0 within each window, which is what the model sees at
  generation time.

  **`embeddings` prepends H3 text embeddings to every chunk's prompt.** One filename
  per line from `models/embeddings/`; a bare name goes on every chunk, `name: N` or
  `name: A-B` (1-based) schedules it, which works because each chunk has its own prompt
  anyway. Several lines stack, and their cost is exactly additive — measured, a plain
  prompt splices 0 vectors, `bullet_time` splices 94 (its exact row count), and
  `bullet_time` + `storm_magic` splices 231 = 94 + 137.

  These are DiffSynth-Studio's *Diffusion Templates* — textual inversion for H3, a
  lightweight alternative to LoRA on a 33B DiT. They are **not free**: 50–142 token
  slots each, attended at **every** sampling step of every chunk, and the node's log
  prints the per-chunk total so the bill is visible before you pay it.

  It **refuses** on a core that does not splice `embedding:` in H3 prompts. Before
  [#15808](https://github.com/Comfy-Org/ComfyUI/pull/15808) (merged 2026-08-22) the H3
  tokenizer never looked for the marker and it went through as ordinary words — no
  error, no embedding. Probed for real behaviour rather than a version number, and a
  name with no matching file stops the run rather than being dropped with a log line.

  **All audio is normalised to STEREO before it reaches the VAE.** H3's audio VAE
  expects two channels and core's `_encode_ref_audio` hands the waveform straight to
  it, so a mono track encodes without complaint and is quietly wrong — sglang refuses
  the same input outright. Mono is duplicated, anything above stereo is summed into
  both sides with a warning that the image is gone. Traced by **fredbliss** in
  `minimax_h3_chatter`, 2026-08-22: *"you just do NOT want to pass mono audio encoded
  into h3."*

  `use_input_audio` also **trims the waveform to the clip before encoding** rather
  than encoding the whole track and discarding latents past the end — a five-minute
  track for a sixty-second render was five minutes of VAE encode to keep one. A small
  margin is left on, so the latent-side cut still decides the final length.
  Reference audio is *not* trimmed, since a reference is chosen rather than derived,
  but anything over 30s is reported: references are attended at every step.

- **MMH3 Image List** — collect many reference images into a LIST rather than a batch,
  one Autogrow socket each, up to 50. Feed it to **MMH3 Reference (Multi-Prompt)**'s
  `ref_images`.

  Batching cannot preserve differing shapes — a tensor cannot be ragged, so core's
  `BatchImagesNode` and KJNodes' `ImageBatchMulti` both centre-crop everything to the
  first image, and by the time it reaches the reference node the crop is undetectable.
  Socket order is `<Picture i>` order; empty sockets are skipped, so gaps are fine; a
  socket holding a multi-frame batch expands to one `<Picture>` per frame.

  The report names every reference's dimensions, says how many distinct shapes survived
  and what batching would have cropped them to, and warns past 9 references — they are
  attended at **every** sampling step, so the cost lands on every chunk of every pass.
  (9 is the hosted API's cap, not the model's.)

- **MMH3 Cond Set Apply ControlNet** — applies a MiniMax H3 **Fun ControlNet** to
  every prompt in a cond set, so a chunked render can use one. Core's apply node takes
  a single CONDITIONING and this pack's sampler takes a cond set, so the two do not
  meet without it.

  It also makes the control **chunk-aware**, which is the part that matters. Core's
  `get_control` picks hint frames with `torch.arange(pixel_t)` — from zero, three
  times over (control video, inpaint mask, source video) — and caches the encode keyed
  on `cond_hint.shape[2:]`. Every chunk shares a shape, so unwrapped, **all of them are
  driven by the control video's opening frames** and chunk 0's encode is reused
  throughout, with no error anywhere. The wrapper slices those three inputs to the
  chunk's own span before delegating, so core's arange-from-zero is right because zero
  is now the chunk's first frame. The offset comes from the sampler as
  `transformer_options['mmh3_control_frame0']`, computed with `frame_at_latent` — not
  `latents_to_frames`, which is only meaningful on the 5j+2 grid and answers -12 for
  index 1.

  Unset or 0 is a clean pass-through, which is exactly right for a whole-clip pass
  through a stock sampler. The control video must cover the **whole clip**, not one
  chunk: windows are cut from it by frame index, and a short one clamps to its last
  frame rather than erroring.

  > ⚠️ Built against [**PR #15860**](https://github.com/Comfy-Org/ComfyUI/pull/15860),
  > which is a **draft** — you need it applied to your ComfyUI. The node windows core's
  > internals (`cond_hint_original`, `inpaint_video`, `inpaint_mask`, `cond_hint`) and
  > **refuses with a message** if any of them is renamed, rather than mis-windowing
  > quietly.

- **MMH3 Cond To Set** — the inverse of Cond Select: wrap an already-encoded
  CONDITIONING as a one-entry cond_set, no text encoder involved. The looping
  sampler requires a cond_set and ignores the guider's conditioning, and every
  other producer of one goes through the CLIP — so a refine pass conditioned by
  a zero-out (no prompt, no encoder anywhere in the graph) had no way to reach
  the sampler without loading 20 GB to tokenize an empty string. `count`
  replicates the same conditioning N times; 1 already covers any chunk count,
  since the sampler reuses the last entry.

- **MMH3 Cond Set Strip Text** — drop the prompt from every entry of a cond_set
  while the reference media rides through untouched. For a refine pass whose
  windows are **smaller than the chunk the prompt was written for**: core picks a
  window's prompt region from the window's midpoint, so a window covering a
  fraction of the timeline gets text describing all of it and is asked to render
  the whole script into its slice. At low denoise nothing is invented anyway — the
  content is already in the latent, and identity is the only thing worth
  conditioning on.

  It works because the two are in different halves of a conditioning entry: the
  prompt is the tensor, the references are keys in the dict. `zero` blanks the
  text values and keeps the span's length; `vision only` keeps just the image
  tokens and drops the prose, shortening `text_len` — but references appended
  after encoding never registered with the tokenizer, so for those it leaves the
  text span empty. The node reports that rather than preventing it.

  This path needs nothing beyond stock ComfyUI. A few nodes on `main` ask for an
  upstream PR and say so; only MONKEYPATCHES live on the **`keyframe-anchors`**
  branch. See [`docs/core-changes.md`](docs/core-changes.md).

### Prompting
- **MMH3 Asset Plan** / **MMH3 Task System Prompt** — build a Context-IR system
  prompt for your own LLM node from the task type (or combination) and the
  assets in play, emitting only the relevant rule blocks. See
  `docs/context-ir-system-prompt.md` for the full spec these are derived from.
- **MMH3 Music Caption System Prompt** — the same idea for **MiniMax Music 3**, whose
  `caption` field wants a three-section Structured Caption (Global Metadata / Vocal
  Details / Arrangement) rather than a tag list. MiniMax ships a hosted
  `music-caption-rewriter` to produce one; locally there is none, so this emits the
  rules for your own LLM. Three `lyrics_mode`s — write, supplied (words fixed), or
  instrumental — and an optional section skeleton sized to the duration.

  Duration constants are read from the **installed** model
  (`comfy.ldm.minimax_music.ar`), so the ceiling is the real 360.0s rather than the
  model card's "~5 minutes". Needs ComfyUI v0.33.0+ for Music 3 itself.

  Note that MiniMax's *older* music guide targets the previous generation's hosted
  API — comma-separated descriptors, `--instrumental`, bitrates. Its lyrics tags carry
  over to Music 3; its caption advice does not.
- **MMH3 Lyrics Sectionize** — split fixed lyrics across numbered `[Verse 1]` /
  `[Verse 2]` sections **without changing a word**, with `[Instrumental]` between them.
  Music 3 allocates time **per section**, so one long block is compressed into one slot
  and the delivery rushes — diagnosed in the community's own testing, where the fix was
  breaking long verses up rather than slowing anything down.

  Deterministic on purpose: an LLM asked to re-emit a fixed lyric rewrites it, which is
  the whole reason this is not part of the caption prompt. Boundaries land on paragraph
  breaks then sentence ends, never mid-sentence, and the word sequence is **compared
  before and after** — it raises rather than drifting. Numbering matters because the
  caption's section-level instrument evolution refers to sections by name.

  Wire its one output twice: to the encoder's `lyrics`, and to the caption node's
  `supplied_lyrics` so the caption is written against the same sectioned text.

- **MMH3 Music Caption Split** — the join to `MiniMaxMusic3TextEncode`: one LLM reply
  in, `caption` and `lyrics` out. Tolerates code fences, preamble, bolded or bulleted
  labels, and a missing lyrics field. Names an empty caption and a tags-but-no-words
  lyrics block rather than passing either on silently, since both look like model
  failures downstream.

  Full path: idea -> LLM (system prompt) -> Split -> caption/lyrics -> Text Encode.
- **MMH3 Prompt Lint** — check a written prompt against the format its `mode`
  implies: missing sections, a `retention_analysis` line with no marker, a hidden
  cut, timestamps out of order, `[Shot 1]` carrying one. Reports rather than
  rewrites.
- **MMH3 Replace Section** — splice one refined section back into a complete prompt.
  The two-model route: the technical model writes the whole prompt, a second expands
  `detailed_description`, this puts it back. Both formats' section sets are known,
  so it refuses a section the selected mode does not have.
- **MiniMax H3 Prompt Accumulate** — append one prompt to a running pipe-separated
  string, for a graph loop writing one prompt per window. Exists because a loop
  carries values, not lists. The first pass is the case that goes wrong: the carried
  slot is unwired on iteration 0, and a naive accumulator emits a leading separator
  or the literal text `None`. `prior_context` formats the earlier prompts for
  feeding back to the writing model — put a second copy at the *top* of the loop
  body to read it, since this node sits after the model and its own output cannot
  reach upstream.

  **`prior_context_mode` is the lever on repetitive output.** `all` (the default)
  re-sends every earlier prompt in full — ~7,900 tokens by window 7 of a 20s-window
  clip, against a few hundred for the new audio, which is roughly 20:1 in favour of
  copying. It also re-sends every earlier `detailed_description`, the one section
  the header asks to *differ*. `last_definitions` sends only the previous window's
  `subject_definitions` and `retention_analysis` — what must stay identical, and
  nothing to imitate for what should not. If late windows are re-describing earlier
  ones, start here.

- **MMH3 Scene Plan Prompt** / **MMH3 Prompt Part** — build N chunk prompts **section
  by section** instead of chunk by chunk.

  Writing chunk *i* in isolation asks the model for a complete arc in every chunk. It
  cannot know it is the middle, so every chunk sets up, escalates and resolves — in
  testing, five variants of one scene, each landing its own climax. That is the loop's
  shape, not the wording, so no amount of rule-tightening fixes it.

  Transposing the loop fixes three things at once. `subject_definitions` and
  `retention_analysis` are written **once** and reused verbatim, so the drift that
  produces a stray `<Subject 2>` with no retention line becomes impossible. Escalation
  is decided where all N chunks are visible — the beat sheet — with an explicit floor:
  nothing resolves before beat N. And dialogue planned across the whole set cannot
  repeat a line in three chunks, which per-chunk planning reliably does.

  It also costs **fewer** LLM calls, not more: `1 + 1 + N` against `2N`. Eight chunks
  goes from 16 calls to 10.

  | stage | calls | writes |
  |---|---|---|
  | `definitions` | 1 | every film-wide section — definitions, retention, soundscape, score — plus bare `summary:` / `detailed_description:` headers |
  | `beats` | 1 | all N summaries, pipe-separated: the escalation ladder |
  | `shots` | N | one chunk's `detailed_description`, given the **whole** beat sheet and told which beat it is |

  Soundscape and score are film-wide for the same reason the definitions are — a sound
  world that drifts between chunks is audible drift — so the `definitions` call emits a
  complete six-section skeleton. The bare headers are not optional: **MMH3 Replace
  Section** refuses to splice into a prompt with sections missing.

  Wiring, using nodes you already have:

  ```
  Scene Plan (definitions) -> LLM ------------------------> skeleton
  Scene Plan (beats)       -> LLM ------------------------> beat sheet
    for i in 0..N-1:
      Prompt Part(beat sheet, i) ------------------------->  beat i
      Prompt Accumulate(carried).prior_context (mode last) -> prev chunk's detailed
      Scene Plan (shots, beat_index=i, beat_sheet=.., prev_detailed=..) -> LLM -> shots i
      Replace Section(skeleton, beat i,  "summary")
      Replace Section(     ^  , shots i, "detailed_description")
      Prompt Accumulate -> pipe-separated string -> MMH3 Reference (Multi-Prompt)
  ```

  **MMH3 Prompt Part** is the join between a sheet written all at once and a loop
  rendering one beat per pass: it splits on the same `|` the accumulator and
  multi-prompt use, tolerates the code fences an LLM adds anyway, and past the end
  either repeats the last beat (matching how the looping sampler reuses the last cond)
  or raises, your choice.

  **Chunk-to-chunk continuity** (`prev_detailed`, optional, append-only input). By
  default the `shots` writer sees only the beat *sheet* — the summaries — never the
  previous chunk's realised output, so it continues the story but re-invents the
  staging, and a cut-to-cut render resets the camera and poses every chunk. Wire the
  loop's carried prompts back through **MMH3 Prompt Accumulate**'s `prior_context`
  output in mode **`last`** into `prev_detailed`, and each chunk is told to open
  [Shot 1] on the previous chunk's FINAL frame and advance — so chunk *i+1* picks up
  where chunk *i* ended (same positions, injuries, camera) instead of opening cold.
  Empty on beat 0; unwired past beat 0, the node warns. This is the cinematic opposite
  of the music-video path's `last_definitions`, which withholds `detailed_description`
  on purpose because there the windows must *differ*.

  **`mode`: `cinematic` (default) / `talking_head`.** `talking_head` repurposes the same
  machinery for one ABSOLUTELY-LOCKED continuous take — a person speaking to camera. The
  `shots` stage holds a fixed-tripod frame (no cut, no camera move, no new action),
  writes a continuing spoken monologue instead of escalating, needs no `beat_sheet`, and
  reads `brief` as the topic the subject talks *about*. Definitions and the continuity
  feed carry over unchanged. Because there is no cut at the chunk boundary to hide a
  seam, it is also the strictest test of whether the looping sampler joins chunks
  seamlessly.

  The `shots` stage refuses to run without a `beat_sheet` rather than quietly writing
  a self-contained chunk — that failure is the one this exists to remove. Its banality
  rule is scoped to **speech only**: banal lines over an escalating scene, never a
  banal scene.

### Music video

A separate three-stage chain from the cinematic one, because a song already has an
arc and the cinematic rules fight it. Full pipeline: separate the vocal, align the
lyrics against it, slice the alignment by render window, then write prompts.

- **MMH3 Forced Align (Lyrics)** — place KNOWN lyrics on the timeline. Forced
  alignment, not transcription: the words are given and only their timing is solved,
  so it cannot mishear. That matters because a transcriber guesses badly at singing,
  and everything downstream inherits the mistake — prompts describing words nobody
  sang, typography quoting a mishearing.

  ⚠ **Feed it the lyrics AS PERFORMED, not as prompted.** Alignment assumes the text
  and the audio hold the same words the same number of times. Suno repeats hooks and
  stutters refrains; one copy of a line against three utterances leaves the aligner
  to pick one and strand the rest, which surfaces as large gaps, stretched words and
  whole sections landing early. **No parameter fixes that** — `nonspeech_skip`,
  `max_word_dur`, VAD and `snap_to_onset` all decide *where a word may land*, and
  none of them can conjure two missing repeats. Writing the line three times does.

  It refuses to return a word sequence that differs from its input, since a
  misaligned lyric is fiction every consumer would quote. Emits the same
  `whisper_alignment` type ComfyUI-Whisper does — so `Whisper → Text` and
  `Whisper → Segments` consume it unchanged — plus JSON, so a song is aligned once
  and reloaded instead of paying for a 3 GB model every run.

  Needs isolated vocals; any separator will do. Needs `stable-ts`, which is one pure-
  Python package on top of `openai-whisper`. `large-v3.pt` is shared with any
  non-ComfyUI install through `folder_paths`, so there is no second copy.

  **The report is the interface.** It prints the section map — the one line you can
  check against your own ears — and classifies every anomaly using the audio itself
  rather than guessing: a gap over silence is a correct skip, a gap over audio is a
  skipped passage, and words sitting on silence are misplaced. That distinction is
  what separates a musical pause from a misalignment, and no amount of timing
  arithmetic can make it.

  ⚠ `vad` (Silero) is available and was **far worse** on a produced vocal —
  131 of 190 words came back zero-length, because Silero is trained on speech and
  does not fire on singing. Useful for spoken word; not for song.

- **MMH3 Music Analysis** — librosa: BPM, key and mode, a 4/4 bar grid, and a 10 Hz
  RMS energy curve, from the FULL MIX rather than the stem. Ported from
  music-director's `music.py` minus its cut-salience blend and agglomerative
  segmentation — both exist to *choose* scene boundaries, and the looping sampler's
  windows are uniform and already fixed.

  What survives is what still helps inside a window someone else decided: bar lines
  are cut candidates alongside word onsets, and energy is the only thing that tells
  an instrumental window whether the music there is a soft fall or a drop.

- **MMH3 Lyrics to Windows** — the join between a song's timeline and a sampler's
  schedule. Inputs mirror **MMH3 Split Audio to Windows** exactly so both read the
  same plan and cannot disagree about which frames window *i* covers.

  Three things it exists to get right, each silently wrong if hand-rolled:

  **Window-relative timestamps.** H3 shot times are measured from the start of the
  CHUNK. A window opening at 70.15s holding a word at 72.40s must emit `00:02.250`;
  absolute time produces prompts H3 cannot act on and nothing errors. Context
  windows are rebased onto *this* window's clock too, so a neighbouring line does
  not read as a second, contradictory timeline.

  **`has_lyrics`.** Intros, instrumental breaks and outros are real windows with
  nothing sung in them, and they need a prompt branch that says so rather than one
  left to invent singing.

  **Section context.** Uniform windows and musical sections do not divide, so a
  straddling window reports `chorus -> bridge (bridge begins at 00:07.000)` rather
  than pretending it sits in one.

- **MMH3 Music Scene Plan Prompt** — the same three stages as **MMH3 Scene Plan
  Prompt**, with the rules inverted where a song demands it:

  | | cinematic | music video |
  |---|---|---|
  | the arc | invented, nothing resolves before beat N | **the song's** — "do NOT invent an escalation" |
  | a repeat | would be redundant | **should feel like the same chorus** |
  | the words | invented | supplied, verbatim, per window |
  | shot times | invented | supplied as word onsets |

  `non_diegetic_music` describes **this** track rather than composing an
  alternative, and `overall_soundscape` opens with "the song is the audio" so it
  does not invent room tone competing with it.

  **Typography is rationed in `beats`, once, across the whole song** — "most should
  not." Decided per chunk with lyrics in hand, every chunk reaches for it and the
  result reads as a lyric video. Two modes: `exact lyrics` quotes the sung line
  verbatim; `text bursts` allows fragments and re-spellings, which is invention
  grounded in the real words rather than replacing them.

  A window with `has_lyrics: false` switches to an instrumental branch that forbids
  singing, asks for visual event instead, and **suppresses typography even when the
  beat sheet assigned it** — there is no line to quote.

  **`music_source`** — `supplied` (default) or `generated`. The rules assume the
  track exists and will be handed to the sampler; `generated` is for a graph where H3
  writes the audio in the same pass. It turns `non_diegetic_music` from a description
  into the **spec the model performs**, stops `overall_soundscape` claiming a track
  was provided, and — the part nothing else supplies — makes the shots stage quote
  the window's lyrics as `<d>[English] ...</d>` with the subject **singing** them.
  Without it the words reach the writer, get spent deciding what the picture does,
  and none are sung.

  **`treatments`** — `music video` (default) or `restrained`. The default reaches for
  split frames on purpose, which is a music-video technique rather than an artifact.
  `restrained` forbids frame division, inset, banded overlay and multiplied
  performers, and keeps the effect vocabulary optical. For a piece whose subject is
  the performance, a split frame halves the singer exactly when the mouth is the
  point.

  `reference_images` tells the definitions stage that attached images ARE the
  subject, that several images are one person from different angles, and that the
  image beats the brief on appearance. Definitions only: the description is written
  once and reused, and re-deriving it per call is how a subject drifts. Without the
  flag a vision model handed pictures and no instruction describes an invented
  character anyway.

  The whole chain — grid, alignment, the three stages, typography, symptom table —
  is written up in [`docs/music-video.md`](docs/music-video.md).

- **MMH3 Official H3 Tokens** — adds H3's seven special tokens to a CLIP's
  tokenizer. **Superseded by [#15808](https://github.com/Comfy-Org/ComfyUI/pull/15808)
  (merged 2026-08-22): core now adds all seven at tokenizer init, on their documented
  ids.** On a current ComfyUI this node detects that and passes the CLIP through
  untouched — *"already patched, nothing to do"*. It is kept for older cores and does
  no harm wired in. Verified on `v0.33.0-49`: all seven land correctly and
  `<d>[English] hello.</d>` is **7 ids** where the unpatched tokenizer produced 15
  with the marker split across `'<d'`, `'>['` and `'.</'`, `'d'`, `'>'`. ComfyUI routes H3 text through the shared `qwen25_tokenizer/`, whose
  `added_tokens_decoder` stops at **151668**; H3 adds seven ids on top of stock
  Qwen3-VL, and the model card says its own tokenizer config is required. Without
  them `<d>` is not a reserved id — it is ordinary subwords that merge with
  neighbouring whitespace, language tags and punctuation:

  | | `The woman says, <d>[English] We need to leave now.</d>` |
  |---|---|
  | stock | `… 'Ġ<' 'd' '>[' 'English' … 'Ġnow' '.</' 'd' '>'` — 17 ids |
  | patched | `… 'Ġ' '<d>' '[' 'English' … 'Ġnow' '.' '</d>'` — 16 ids |

  Note `'>['` swallowing the language bracket and `'.</'` pulling the sentence's
  final stop inside the marker.

  It patches a **copy**: `CLIP.clone()` shares its tokenizer by reference, so an
  in-place edit would follow the loaded model around and survive bypassing the node.
  The incoming CLIP is untouched, `enabled` off is a pass-through, re-running is a
  no-op, and it **refuses a non-H3 CLIP** rather than shifting ids some other model
  does have. The ids are verified against the H3 config after the fact, not assumed
  from the vocabulary length.

  Wire it between `CLIPLoader` and whatever encodes prompts. The embedding rows
  exist (`[151936, 5120]`), so the ids are in range — but see
  [`docs/music-video.md`](docs/music-video.md) for what is and is not known about
  whether this improves output.

- **MMH3 Load Skill** — loads one file from the pack's `styles/` folder and emits it
  for `extra_rules` on either scene-plan node. **Chain the nodes to stack skills**:
  wire one node's output into the next node's `previous`, so wiring order is stacking
  order. `enabled` off passes `previous` through untouched.

  One file per node on purpose. Selecting several in one node means deciding up front
  which kinds of skill exist and how many of each you may have; a chain decides
  nothing. The type lives in the filename — `look-`, `typography-`, `experiment-` —
  which is enough to find it in the dropdown, and anything you drop in the folder
  appears there with no registration step.

  An `experiment-` file is flagged in the report as untested: those say what we want
  to find out H3 can do, not what has been observed working, so judge the result on
  its own rather than as a known recipe.

  **Why blocks and not the vendor skills.** MiniMax publish nine H3 skills; all nine
  are agent procedures for their own hub, and two say so outright. Around their visual
  guidance sits numbered steps, confirmation gates, prescribed shot counts and time
  segments written for 15-second clips. Pasted whole into a prompt, that fights a
  grid-locked window and a pinned master audio. `styles/` is the visual core lifted
  out with the procedure left behind.

#### Observed — 2026-08-15

What a first full music-video run taught — the typography corrections, thematic type,
split frames as a technique rather than an artifact, and why slow motion is treated as
a choice here — now lives in
[`docs/music-video.md`](docs/music-video.md) §9, alongside the chain it applies to.

### Sampling
- **MiniMax H3 Looping Sampler** — fill a whole clip chunk by chunk in one node
  execution. The graph is the same size for 4 chunks or 40, which is the point.

  **The latent is the finished clip**, and the chunk count is derived from it — you
  hold a song of known length and do not know how many chunks that is. Chunks are
  slices written back in place, so there is no join, no trim, and the output is
  exactly the length you passed in. Each chunk also slices its own span of audio, so
  a track pinned by `use_input_audio` reaches every chunk.

  The schedule comes from the same `_plan` as **Window Plan** and **Split Audio to
  Windows**, so chunk N renders the audio window N's prompt was written against.
  Two carry routes (masked overlap, or a guide), keyframe indices in clip frames,
  and a per-chunk guider swap.

  Its optional **`denoise_mask`** takes a MASK over the whole clip — white
  regenerates, black keeps the input latent's content — so a region or a span can be
  held while the rest re-renders. It masks the **video half only**; the audio half is
  reached solely through `audio_denoise_mask`. The mask is reduced ONCE onto the
  master grid and
  merged keep-wins (elementwise min) with whatever mask the latent carried and with
  the overlap carry, then sliced per chunk, so `_carry_mask` composes with it and
  nothing in the loop changed.

  Geometry follows the VAE rather than a resize. Spatially it is **pooled, not
  interpolated** (bilinear averages, and every fractional cell then denoises at its
  own timestep), then snapped to the **2×2 patch the DiT reads the mask through** —
  so 32 pixels is the finest feature a mask can express and no token straddles an
  edge. Temporally it groups on the real `FRAME_PER_TOKEN` cycle `(1,4,4,4,4)`, where
  the first latent of every 17-frame group covers a **single** frame; a uniform 17/5
  split puts a temporal edge in the wrong place.

  **`audio_denoise_mask`** masks the audio half, and is the only input that does.
  Only its TIME axis is read: each frame reduces to one value, then maps onto the
  audio grid through `_audio_index_at`, the same boundary conversion the chunk loop
  uses, so a span frozen here lines up with the picture. Left unconnected, audio is
  masked only by whatever the input latent already carried.

  The two are **independent by design**, which is a correction (2026-08-22). Audio
  used to be *derived* from the video mask whenever no explicit audio mask was wired,
  on the reasoning that the two modalities could then never disagree about a frozen
  span. That is only true of a mask carrying temporal intent. A **spatial** mask — a
  SAM3 subject matte, say — is white somewhere in every single frame, so the spatial
  reduction returned "free" at every timestep and regenerated the whole track while
  the video mask did exactly what was asked. Measured on a subject matte: 75% of the
  video grid held, 100% of the audio freed. Deriving one modality from the other is
  not a safety net, so it is gone.

  It **refuses** on a core without per-row masking (#15375), where a mask is accepted
  and silently ignored, and warns when the input latent is all zeros, since kept
  regions would pin black. This is a v2v tool.

  The sigma schedule can be **windowed per chunk** with `sampling_start_step` /
  `sampling_end_step` — absolute indices, sliced exactly as core `SplitSigmas` does,
  so a two-pass run is `end N` then `start N` with no arithmetic. `phase2_start_step`
  plus an optional `phase2_sampler` / `phase2_guider` switches solver mid-schedule
  for dual-solver setups. All three carry LTXAVTools' semantics unchanged. See
  [`docs/looping-sampler.md`](docs/looping-sampler.md) — including what is still
  unmeasured.

- **MiniMax H3 Keyframe Planner** — end-anchored keyframe indices for a chained run,
  ported from LTXAVTools' planner. Frame 0 opens, each chunk travels to a keyframe at
  the last frame **it renders**, the final one ends on `-1`. Start-anchoring instead
  would put each image in the NEXT chunk and invite a snap at every seam. Emits
  `indices` for the sampler's `keyframe_indices`, `count` for how many images the
  batch needs, and `chunk_count`.

  Same three numbers as the sampler, same `_plan`, so the two cannot disagree about
  where a chunk ends.
- **MiniMax H3 Context Windows** — windowed sampling over one long latent, per
  modality: video on dim 2, audio on dim 3, each with its own window. Snaps length
  and overlap to the grid, since an overlap that is a multiple of 5 rather than
  `5m+2` walks the window phase `0,2,4,1,3` — a five-window beat, which is the
  pulsing. See [`docs/context-windows.md`](docs/context-windows.md).

  Windows are **not** a way to grow a clip: every window is a slice of one
  preallocated latent, and all of them sit at the same noise level at every step.
  Chaining is what grows.

  Windows bound the model's *compute*, not the sampler's *storage* — the full
  latent, its noise, and the fuse accumulators stay resident at full length, so a
  longer clip still costs VRAM at a fixed window size. Two things trim that:
  a cond skipped by cfg 1.0 no longer allocates its accumulator at all (its zeros
  are materialized after the window loop instead — automatic, saves one full-length
  fp32 latent), and `accumulator_device: cpu` hosts the remaining accumulators in
  system RAM, writing window-sized slices across PCIe during the loop and moving
  the fused result back once per step. Values are identical either way.
- **MMH3 Chunk Schedule** — say roughly what you want; get a schedule that actually
  tiles. Solves total, window and overlap **together** and emits frames.

  The frame calculator answers "what does 22.2 seconds round to". Useful, and not the
  question. Three of those answers chosen independently still leave the last window
  clamped, because what has to hold is a relationship *between* the three: 60s with a
  20s window and 3s overlap gives four chunks whose last one strides 7.08s instead of
  17.00s, re-rendering 12.2 seconds a previous chunk already made under a different
  prompt. No single conversion can see that, so no amount of widget precision fixes it.

  Write the group counts as `t = 5c+2`, `L = 5a+2`, `O = 5b+2`. Then stride is
  `5(a-b)`, a multiple of 5 for **any** a and b — so grid phase is safe automatically
  and the five-window pulse cannot happen. What is not automatic is
  `(c - a) % (a - b) == 0`, and that is the whole search.

  **`chunks` is usually the input you actually have an opinion about** — it is how many
  prompts you write and how many joins the piece has. Set it and the window becomes a
  *result* rather than a second guess. An unreachable count is released and reported,
  never raised.

  **`chunk_count` and `seconds_per_chunk` wire straight into either scene-plan
  node's inputs of the same names**, so the writer and the schedule cannot disagree
  about how many chunks there are or how long one is. `seconds_per_chunk` is the
  WINDOW's duration, not the clip's.

  `prefer` decides what may move: `keep total` holds the deliverable length and shifts
  the window and overlap, `nearest` lets the length drift a few groups for a closer
  fit, `fewer chunks` takes the shortest chunk list it can. The report names every
  move and prints the divisibility as proof.

  It also lists the **reachable overlaps** for the chunk count you are on, because
  the count is the overlap's step size: with the total and the count both fixed the
  stride is `(c-b)/n` and must come out whole, so valid overlaps sit exactly `n`
  groups apart. At 3 chunks that is 2, 17, 32, 47 latents and nothing between — which
  reads as the widget refusing to move until you know why. Asking for MORE chunks
  makes the overlap COARSER, not finer.
- **MMH3 Chunk Schedule (Frames)** — the same node asked in frames. Identical solver,
  identical grid rules, identical outputs; it only skips the seconds-to-frames
  conversion, for when you already hold frame counts and do not want a duration
  rounded on the way in. The two share one implementation rather than being copies,
  so they cannot drift.

  It still **snaps**: values land on the nearest `17j+5` and are still solved
  together. Asking in frames does not mean asking for arbitrary frames — 1445 and
  1446 both resolve to 1450. Its defaults mirror the seconds node's 60.0s / 20.0s /
  3.0s exactly (1433 / 481 / 73), so the two agree out of the box.
- **MMH3 Window Plan** — resolve the whole schedule up front, in frames. How many
  windows you get is how many prompts to write; whether your window and overlap
  survive snapping is otherwise only knowable by running a generation.

  **Every output carries its unit in its name**, because crossing them is the one
  mistake this node invites: `context_length (latents)` / `context_overlap
  (latents)` go to Context Windows, while `window_frames (frames)` /
  `overlap_frames (frames)` go to the looping sampler and Split Audio to Windows.
  The frame pair sits five sockets below the latent pair, which is exactly how they
  get swapped. Crossing them does not error — a latent count is a valid frame
  count, just a much smaller one — it re-snaps and the schedules quietly diverge.
  Observed 2026-08-17: `context_length` 117 wired into the sampler's `chunk_frames`
  re-snapped to 32 latents, so the sampler ran **11 chunks of 4.5s** while both
  planners reported 3 chunks of 16.5s.
- **MMH3 Split Audio to Windows** — cut a track into one clip per window, matching
  the real schedule including the overlap and the clamped final window. The numbered
  sockets fan every window across the graph at once; the `audio` output emits ONE,
  chosen by `index`, so a for loop keeps the graph constant-size. `index` also
  reaches past the numbered ceiling.
- **MMH3 Window Context** — one line saying which span of the song a window covers,
  for the per-window prompt loop. Without it the loop hands the writing model the
  same text every iteration and only the audio changes, so on a repetitive track
  nothing distinguishes window 5 from window 2 — and `prior_context`'s "keep these
  byte-identical" then pulls the late windows onto the same shots. Same `_plan` as
  everything else, so the timecode names the audio the window really renders.
  Concatenate onto the **END** of the model's prompt, after `prior_context`.

### Latent
- **MiniMax H3 Seed Overlap** — **prepends** overlap latents to the target and masks
  them, giving frame-level seam continuity. Prepending rather than overwriting means
  the chunk keeps its full requested duration and the overlap is cut off afterwards.
  Needs **#15375**; refuses without it.
- **MiniMax H3 Outpaint Latent** — grow or crop a latent's canvas, masking the new
  region so the model fills it. Edges are **signed**: positive pads, negative crops,
  and each snaps toward zero so a value between steps never crops more than asked.
  An inward `feather` ramps into the source region. H3 has no cross-attention, so
  margin rows attend directly to real rows at every layer, and scene fill converges
  in very few steps.
- **MiniMax H3 Join AV** — join two clips in **pixel** space, at frame granularity.
  Latent joins land on 17-frame boundaries; this is what Find Divergence's answer
  feeds.
- **MiniMax H3 Reference from Latent** — build a `minimax_refs` block from a latent
  directly.
- **MMH3 Chunked Pixel Upscale** — stage-1 latent → 2K latent, through pixels, a
  chunk at a time. For the **refine** leg of a 2K pass. See
  [Refine vs regenerate](#refine-vs-regenerate).
- **MiniMax H3 Streaming Encode** / **MMH3 Streaming Save** — encode and export
  in bounded RAM. Save decodes group by group and writes as it goes rather than
  holding the whole clip, which is the difference between exporting a long master and
  running out of memory. Slower per frame; for long videos only. `save_metadata` (default on) embeds
  the workflow and prompt so the file drags back into ComfyUI; it goes in an
  ffmetadata file rather than a command-line argument because a workflow is 45–95 KB
  and Windows caps a command line near 32,767 characters. `faststart` is deliberately
  not applied — relocating the moov atom rewrites the whole file and would undo the
  constant-cost decode this node exists for.
- **Whisper to Text (LLM Ready)** — flattens ComfyUI-Whisper's per-word
  `whisper_alignment` into text a prompt writer can read, with timestamp markers so it
  knows where in the song each line falls. Feed it to the scene-plan nodes as lyrics.
  Markers land on interval **boundaries** rather than on whichever word crossed one,
  so the same song always marks at the same times.

  **Adopted into this pack 2026-08-24.** It used to live in a loose
  `ComfyUI-WhisperAlignmentToText` folder that was never published, so the MusicVideo
  workflow listed a dependency nobody could install. The node id is unchanged, so
  existing graphs keep working — if you have that folder, delete it, or two packs will
  claim the same node. Its sibling `WhisperAlignmentToSegments` was deliberately NOT
  adopted: it cuts on 25 fps and a 4n+1 frame grid, which is LTX's, not H3's 24 fps /
  17j+5, and **MMH3 Window Plan** and **Split Audio to Windows** already do that job on
  the right grid.
- **MMH3 Reference Attention Probe** / **MMH3 Reference Attention Map** — which
  reference is each part of the clip actually attending to. H3 has **no
  cross-attention** (`grep -c cross_attn comfy/ldm/minimax/model.py` → 0): references
  are packed into the same sequence, so this is a measurement rather than an inference
  from the output. The spans are not guessed either — `model.py` records the layout as
  `("ref_audio", rt * 2)` segments, one per reference in order, and the probe captures
  that list the way Sol-Attn does.

  Wire the probe anywhere in the model chain, render, then read the Map: one row per
  reference, time along x. It chains any existing attention override, so it coexists
  with Sol-Attn.

  **It judges per moment, never on the time average.** Under a working binding — one
  reference leading while speaker A talks, the other while B talks — the two averages
  come out equal, so an average-based test would call the best possible result "no
  binding". The report gives the per-moment margin, how many times the lead changes,
  and what share of the clip each reference leads, which separates the three outcomes:
  no reference ever leads, one reference leads the entire clip and never hands over,
  or the lead alternates.

  Queries are pooled to one centroid per 64 rows (sol_attn's own routing granularity,
  ~5e-4 cosine). The **denominator is exact**, streamed over key chunks inside a
  memory budget — pooling the keys as well was tried first and is wrong: logsumexp
  over a key block is near its MAX while a centroid is its MEAN, so a row attending
  one sharp key elsewhere had its tail understated and both references came back at
  ~0.50 when the truth was ~0.005.

  **Attention mass is where the model LOOKED, not what it took.** A row can attend a
  reference heavily and not adopt its timbre, and binding to the WRONG reference looks
  identical here to binding to the right one.
- **MMH3 Size Capped Copy** — a second copy of a finished file under a hard size
  ceiling, for upload limits. Chains off Streaming Save's `file_path`; takes any
  video, not just H3 output. `target_mb` is a **ceiling, never a target**: a file
  already under it is not re-encoded, and the node returns the source path
  unchanged rather than writing a copy. See [Delivery copies](#delivery-copies).
- **MiniMax H3 Trim AV** — drop latents from the head and/or tail, cutting audio and
  masks to match. Note the grid rule **inverts** relative to Concat AV: trimming one
  latent, `5m` keeps the result on grid and `5m+2` takes it off, because there the
  constraint is on the joined *total* rather than the piece being cut.
- **MiniMax H3 Split AV** — pull an AV latent into plain video and audio latents. The
  exact inverse of Pack AV, so carrying stage 1's audio through an upscale ladder is
  something the graph states rather than a discipline you have to remember.
  **`preserve_masks`** (default on) hands each half its own `noise_mask`. Pack AV has
  always had a branch for re-pairing masks, but until 2026-08-22 Split emitted bare
  `{"samples": ...}` dicts, so there was never anything left to re-pair and a
  split/repack silently discarded the pin `use_input_audio` installs to protect a
  supplied track. Turn it off to discard the mask deliberately.
- **MiniMax H3 Pack AV** — pair a video latent with an audio latent. Encoding real
  footage gives two *separate* plain latents (`VAEEncode` + `VAEEncodeAudio`) and
  nothing joins them. Audio is reconciled to `round(frames / 24 * 40)`. This is a
  **modality** join; Concat AV is a **time** join.
- **MiniMax H3 Find Divergence** — measures how many frames a continuation
  reproduces from its source, so the join can be trimmed at frame granularity.
- **MiniMax H3 Concat AV** — join two AV latents on the correct axes (video dim 2,
  audio dim 3), with optional `trim_b_latents` and `carry_masks`.

  `trim_b_latents` is honoured as given, because **no single snap is correct**.
  With `A = 5a+2` and `B = 5b+2`:

  | trim | effect |
  |---|---|
  | `5m` | removes a Seed Overlap **exactly**; the total is `5(a+b)+4−k`, **off grid** |
  | `5m+2` | total lands **on grid**; ~7 frames of overlap stay duplicated |

  `k` cannot be `0` and `2 (mod 5)` at once. If you need both, that is what
  **Join AV** is for — it cuts per frame in pixel space. The node logs which
  property the value you gave it actually gets.

### Latent joins happen in pixel space

Latent concatenation is unsound here. Two on-grid chunks sum to `5(j+k)+4`
latents, never back on the `5j+2` grid, so the VAE's 17-frame causal chunking
misaligns from the join onward and the second half pulses. **Join AV** and
**Find Divergence** therefore work on decoded frames, where granularity is one
frame rather than 17, and audio crossfades in the **waveform** domain — the
DAC/BigVGAN latents do not blend.

> **On `noise_mask`:** masks do reach the model — `samplers.py` packs latents
> before sampling and explicitly handles `denoise_mask.is_nested`. Stock is missing
> **three** things, and it is worth separating them, because only the first is
> usually quoted:
>
> 1. **Per-row timesteps.** Preserved rows still run at the generation timestep, so
>    the model gets clean content labelled as noisy and the mask accomplishes nothing.
> 2. **The mask never reaches the model as a cond.** #15375 unpacks it and passes
>    `denoise_mask` / `audio_denoise_mask` through, which is what makes (1) possible.
> 3. **No `scale_latent_inpaint` override on `MiniMaxH3`.** Stock falls back to
>    `BaseModel`'s noise blend; #15375 injects preserved regions at H3's cond timestep
>    (`VISUAL_COND_TIMESTEP`, 0.999) and rescales the audio half for `audio_scale`.
>    Verified against the class directly — stock `MiniMaxH3` has no such method.
>
> (3) is the one that shows up as artifacting, and it is confined to **intermediate**
> mask values. A hard 0/1 mask is unaffected either way; only intermediate ones
> takes the same path either way. That is why the seam noise in 0.72.x tracked
> `feather_latents` and vanished when the feather was removed in 0.73.0 — a feather
> was the only thing in the pack producing intermediate values at
> `overlap_strength=1.0`.
>
> **drozbay's per-row masking fixes all three — upstream PR
> [#15375](https://github.com/Comfy-Org/ComfyUI/pull/15375).** `MMH3SeedOverlap`
> and the outpaint node need it, and refuse to run without it rather than
> appearing to work. Applying an upstream PR is not monkeypatching, which is why
> they live here rather than on `keyframe-anchors` — see
> [`docs/core-changes.md`](docs/core-changes.md).

For **audio-driven video**, use an audio reference with the `[audio reuse]` task
type and the `fully_copy` marker, not a mask. That is a trained capability.

### Refine vs regenerate

Two ways to get from a 768p stage 1 to 2K, and **the upscale question only exists in
one of them**:

| | refine | regenerate |
|---|---|---|
| node | **Chunked Pixel Upscale** → sampler | **Regenerate-2K Reference** |
| stage 2 starts from | the upscaled stage-1 latent | an **empty** 2K latent |
| stage 1 arrives as | the thing being denoised | `minimax_refs`, never denoised |
| cost | partial denoise | full sampling at 2K |
| drift from stage 1 | low | possible |
| distribution | off — H3 wasn't trained for this | the trained shape |

Regenerate needs no upscale at all: H3 has no cross-attention, so the reference rows
are attended directly at every layer and the 2K target is generated fresh against
them. Refine is cheaper and holds tighter to stage 1, and that is where an upscale
has to happen.

**Do not upscale in latent space for it.** A 24-channel latent at /16 is not a
spatially smooth signal — interpolating between latent positions gives the decoder
codes it never saw, which is the blocking people mean by "chunky latent upscale".
`downscale_video_latent` is bilinear, but it only ever touches *reference* slices,
which are never denoised; approximate context is fine, approximate content is not.

Chunked Pixel Upscale therefore goes through pixels, and chunks the whole way across
so length is not a constraint. If you are decoding stage 1 anyway for a preview, the
expensive half of the round trip is already paid — only the re-encode is new.

**Stage scales are not integers.** `Regenerate-2K Dimensions` guarantees an exact
*aspect*, not an integer factor. At 16:9 a `target_long_edge` of 2048 is **1.5x**;
**2688** is exactly 2x. Stage 1 is 6 of that aspect's 224x128 units, so integer
scales land on multiples of 6.

### Delivery copies

**Streaming Save's `crf` cannot hit a file size.** CRF targets *quality* — it
spends whatever bitrate the picture needs and the file lands where it lands. That
is the right setting for a master and the wrong one for an upload limit, so a copy
under a fixed ceiling is a **second encode**, not a knob on the first. Wire
`file_path` into **Size Capped Copy**; the master is read, never modified.

It measures the duration, solves the video bitrate for the budget, and two-passes
libx264 at it — landing within a percent or two, biased under. The budget is in
**MiB**, because upload limits are quoted in binary megabytes and at a 100 "MB"
ceiling the two differ by 5 MB.

**`max_height` is not optional past a few minutes.** The budget is duration-driven,
and long videos run out of bitrate before they run out of pixels:

| Length | Video budget at 95 MiB | Sensible height |
|---|---|---|
| 2 min | ~6,300 kbps | native |
| 5 min | ~2,450 kbps | 1080 |
| 20 min | ~520 kbps | 720 |
| 1 hr | ~90 kbps | split the file |

(at the default `audio_kbps` of 128, which comes off the top before video is solved)

At 2K, 520 kbps is mush; at 720p it is watchable. A source already shorter than the
cap is never upscaled into it. Under 150 kbps the node warns rather than pretending
the result is usable.

### Model
- **MMH3 AdaLN Reference Patch** — take AdaLN modulation from another H3 checkpoint,
  per block. `fl2va` and `ref2va` are the same model *except* for AdaLN: attention,
  MLP, `condition_proj`, the patch projections and the output heads all measure at
  cosine 0.999+, while every `adaln_proj` lands between −0.42 and −0.91. AdaLN is
  where reference conditioning enters the residual stream, so that one component is
  the whole difference between a checkpoint that can condition on a reference and one
  that cannot. Reads only the `adaln_proj` tensors from the source — ~100 MB of a
  20 GB file.

  `blocks` takes ranges and lists (`25-49`, `0-2,40-49`, `-1`), so the published
  hybrid checkpoints are widget values rather than downloads, and non-contiguous sets
  are possible. `final_layer` covers the last modulation before the output heads
  (cosine −0.830), which those hybrids leave alone.

  **There is no strength slider on purpose.** The two AdaLNs are anti-correlated at
  near-equal norms, so a blend cancels instead of mixing — at 0.5 the modulation drops
  to 32% of either endpoint and most of the conditioning routing switches off. Per
  block it is one side or the other. Per-row and per-term controls are absent too: the
  difference is uniform across all three modality rows and all six terms, so there is
  nothing to isolate.

### Util
- **MMH3 Latent Info** — shapes, frame count, audio-length mismatch, grid
  alignment, mask presence.
- **MMH3 Motion Overload** — which latent time tokens of a **rendered** clip carry
  more motion than one token can represent. Four of every five tokens span four
  pixel frames, so when fast motion needs four distinct poses in those frames, one
  token cannot hold them and the decode smears; the poses were never generated, which
  is why the artefact does not answer to steps or resolution. Third difference along
  the token axis, phase-normalised for the `(1,4,4,4,4)` grid, reported as hot spans
  in tokens, frames and seconds.

  **Read the contrast ratios, not the spans.** A quantile threshold marks a fixed
  share of tokens whatever it is handed, so the profile *ranks* and does not *detect*.
  `hot / cold mean` near 1.00 means the cut separated nothing, and a flat profile is
  reported as having no variation rather than as infinite separation. This is a
  measurement, not a fix — it tells you whether the footage has the problem before
  anything gets built on the answer.
- **MMH3 Cond Set Spread** — spread a cond_set's N prompts across a windowed
  generation, so each window gets the one written for it. Regions are cut per window
  midpoint; guess the prompt count low and windows share a prompt, guess high and the
  last prompts are never reached. **MMH3 Window Plan** tells you the number.
- **MMH3 Reframe Pads** — pick a target aspect and get the four **signed** edges for
  Outpaint Latent. `extend` grows to reach it, `crop` cuts, `balanced` does both.
  Snapped to the canvas multiple, so what it emits is what outpaint will honour.
- **MMH3 Upscale Ladder** — an aspect and a target long edge in, a ladder of
  `width_N`/`height_N` out, every rung on the canvas grid. For staged upscales,
  so the stage sizes agree by construction rather than by arithmetic you redo.
- **MMH3 Regenerate-2K Dimensions** — the two stages of a 768p → 2K pass.
  **Stage 1 is not a choice**: it reproduces core's `adapt_canvas`, because that is
  what H3-Base emits whatever you ask for, and sizing it any other way makes stage 2
  an upscale of something never rendered. Stage 2 is an integer multiple of stage 1's
  on-grid unit, so the aspect is exact — rounding each axis to 32 instead puts 16:9 at
  2048x1184 (1.7297), and that squeeze is in every frame. The label says when the
  requested long edge could not be honoured. Every ratio is tabulated in
  [`docs/regenerate-2k.md`](docs/regenerate-2k.md).

Calculators follow the LTXAVTools convention — concise typed outputs plus a short
`label`, flat category.

- **MMH3 Frame Calculator** — seconds in. → `frame_count`, `latent_frames`,
  `audio_latent_frames`, `actual_seconds`. `rounding` is nearest / up / down.
- **MMH3 Dimension Calculator** — → `width`, `height`, `width_ref`, `height_ref`,
  `label`. Where `LTXDimensionCalculator` emitted a fixed `width_half`/`height_half`
  pair for its two-stage pipeline, H3 has no second stage — the secondary pair is
  the **reference** size, set by `downscale_factor` and snapped to a factor the
  patch grid supports.

#### Achievable durations

Frames must be `17j+5` at 24fps, so durations are discrete. Solving
`24s ≡ 5 (mod 17)` gives `s ≡ 8 (mod 17)` — **8.000s is the only whole-second
duration in the 4–15s range**:

| asked | frames | actual | drift |
|---|---|---|---|
| 4s | 90 | 3.750s | −0.250 |
| 5s | 124 | 5.167s | +0.167 |
| 6s | 141 | 5.875s | −0.125 |
| **8s** | **192** | **8.000s** | **0** |
| 10s | 243 | 10.125s | +0.125 |
| 12s | 294 | 12.250s | +0.250 |
| 15s | 362 | 15.083s | +0.083 |

This matters when chaining: per-chunk drift accumulates against wall-clock, so
plan chunk lengths in frames, not seconds — or use 192-frame chunks, which stay
on whole seconds indefinitely.
- **MMH3 Dimension Calculator** — snaps width/height to
  the 32px grid, reports latent dims and **tokens per latent frame**, and snaps a
  requested reference downscale to a factor the patch grid supports.

#### Valid reference downscale factors

Latent dims are `px/16` and must stay **even** for the 2×2 patch, so a downscale
factor `f` is valid only when `latent/f` is an even integer on both axes — the
divisors of `gcd(latent_h//2, latent_w//2)`:

| canvas | latent | tokens/frame | valid factors |
|---|---|---|---|
| 1344×768 | 84×48 | 1008 | 1, 2, 3, **6** |
| 1024×1024 | 64×64 | 1024 | 1, 2, 4, 8, 16, 32 |
| 1280×704 | 80×44 | 880 | 1, 2 |
| 1152×640 | 72×40 | 720 | 1, 2, 4 |

Note **4× is invalid on the native 1344×768 canvas** (84/4 = 21, odd) and snaps
to 3×. The factor set depends entirely on the aspect ratio.

## Carrying content between chunks

On stock ComfyUI there is one channel, and it does not do what its name suggests:

| channel | mechanism | carries | position |
|---|---|---|---|
| `MMH3LatentToRef` | `minimax_refs`, never denoised | identity, voice, motion style | before the clip, contiguously |

Two more need a patched core. `MMH3SeedOverlap` (target latent + `noise_mask`)
needs per-row timestep handling to mean anything -- **#15375**. Positioned anchors on
the clip's own timeline need interior indices and the accumulate fix -- **#15439**,
which the **Looping Sampler** uses as `carry="keyframe"`. Both are upstream PRs
applied to core, not monkeypatches; see [`docs/core-changes.md`](docs/core-changes.md).

**References are positioned.** The layout lays them out from a cursor starting at
`text_len`, a `video`/`video_audio` block advances that cursor by its own temporal
span, and the target uses the cursor's final value as its origin — so a carried tail
sits contiguously immediately before the clip, not floating outside time. What it
costs is *distance*: a 39-frame carry moves target frame 0 from 320 to 385 at
`text_len` 320. Audio is free, though — `FRAME_RESCALE` is 5/3 and `40/24` is 5/3, so
a matched audio tail spans exactly what the video spans and the layout's `max()` is a
no-op.

**On stock, a noise mask pins at the sampler, not the model.** Each step the model
predicts the whole clip and the mask overwrites the pinned region afterwards, so it is
corrected rather than conditioned — it never knows the region is fixed when predicting
the rest. **#15375 changes this**: the mask is passed through as a cond and preserved
rows run at the cond timestep, so the model does know. The distance argument above is
unaffected either way — that is about layout, not masking.

A third channel, **positioned keyframe anchors**, pins a run of consecutive tail
frames on the clip's own timeline at **no distance cost** -- measured, target origin
`text_len + 0` against `text_len + 65` for the same carry as a `video_audio` ref. It
needs **#15439**, and the Looping Sampler's `carry="keyframe"` is it. The
`keyframe-anchors` branch reached the same place with monkeypatches and is superseded
now that core carries the PR. Not yet run against real weights.

## Grid reference

| | relation |
|---|---|
| frames | `17j + 5` |
| video latents | `5j + 2` |
| audio latents | `round(frames / 24 * 40)` |
| trained range | 124–362 frames (~5.2–15.1s) |
| node ceiling | 3600 frames (150s) |

Keep core's **`ModelSamplingMiniMaxH3`** (node id `MiniMaxH3SigmaShift` — it is
searchable under the display name, not the id, and it is a stock ComfyUI node rather
than one of these) at video `12.0` / audio `3.0`, and constant across chunks: the DiT
derives the audio schedule from the video one, so varying it per chunk desynchronises
them.

## Known limitations

- Carried references are **not** registered with the tokenizer, so Qwen3-VL never
  sees them. Don't use `<Video k>` tags for a carried chunk. The DiT still gets the
  latents, so pixel/motion/identity continuity works; only the semantic path is
  skipped. For continuation that's arguably correct — you rarely want the encoder
  re-describing the previous chunk.
- `ref2va` **does** respond to keyframe (`cond`) rows. Two bugs used to sit in the
  way; **both are fixed in core** as of the #15439 merge (2026-08-13). It stops
  `model_base.py` overwriting `cond_video_latents` — it concatenates keyframes-then-refs,
  so refs no longer erase keyframes — and the merged version also anchors the guide on
  the **target origin** rather than on `text_len`, which the draft did not.

  On a core predating the merge the position bug is live: a guide lands `ref_advance`
  units before the clip whenever refs are present — measured at **−1** for one image
  reference and **−320** for a chunk's worth of voice audio. Nothing errors; it just
  anchors into the reference region. `patch_guide_origin.py` corrects that, and
  **stands down by self-test** on a core that no longer needs it. Drift table in
  [`docs/core-changes.md`](docs/core-changes.md).
- Latent-space downscaling is bilinear and approximate.
- Audio seams: the audio VAE is DAC encoder + BigVGAN decoder. Crossfade in the
  **waveform** domain after decode, never in latent space.

## Tests

```bash
cd C:/ComfyUI/custom_nodes/ComfyUI-MMH3Tools
for t in tests/test_*.py; do C:/ComfyUI/venv/Scripts/python.exe "$t" || echo "FAIL $t"; done
```

Plain scripts, no pytest — each prints PASS/FAIL per assertion and exits non-zero on
any failure. They import from `mmh3tools`, so they need ComfyUI's interpreter and
ComfyUI on the path; they never touch weights, a GPU, or the network.

They are here because most of what this pack asserts is **arithmetic that is wrong
silently** — a frame count off the `17j+5` grid, an audio window that drifts a latent
per chunk, a section spliced into the wrong format. None of that raises; it renders,
and looks slightly bad. The tests are the record of which of those have been pinned
down, and several encode a bug that actually shipped. Read them as the honest version
of the claims above.
