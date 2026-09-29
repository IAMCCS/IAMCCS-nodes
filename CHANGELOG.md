# Phase B1 — Audio Boundary Polish + ENGINES + Automatic Same-Slot EXTEND (2026-09-27)

- Added an audio-only EXTEND boundary polish after the proven hidden-child-context Soft-AV handover. The first visible child sample is matched to the completed parent/handover endpoint and the correction decays over a short configurable cosine-squared window; no samples are inserted/removed and video is untouched.
- Added Settings PRO AUDIO controls: Soft-AV Handover (default 15 ms), Boundary Polish (default 3 ms) and Polish Strength (default 1.0).
- Renamed the mode-specific rail folder to ENGINES, moved it after the numbered production panels, and grouped AHEAD CONTROL ROOM, LONG TAKE · EXTEND, SAVED TAKE · CONTINUATION, REFMOD, CONTROLNET, SAM3 SWAP, FACE REFINE and SCOUT under the purple engine rail.
- Phase B automatic duration resolver: one authored image/prompt slot longer than 15 s automatically compiles to EXTEND-style MASKED+EXACT. T2V/I2V/FL2V roots use the proven T2VA+guide root; REF2VA keeps REF2VA on the root and uses T2VA for continuation children. Multi-shot timelines are not collapsed into one long take.
- Pianosequenza Shotboard no longer treats 15 s as an editorial drag limit; slots can extend freely and show a 15 s boundary plus an EXTENDED · AUTO / MASKED + EXACT region.
- Verified the existing long-form continuous-audio mode and promoted its user-facing name to LONGVID · MULTI-ANGLE LIPSYNC: multiple positioned image guides share one rebased locked AudioBoard performance across technical H3 windows.
- LongVid EXTEND-style junction migration is intentionally staged for Phase B2: locked AudioBoard lip-sync uses a zero audio mask and must receive an exact-video-only junction, while native-audio LongVid can use full AV EXTEND. GUIDED AV LOOP already owns one persistent AV latent and must not receive a second junction authority.

# Phase 9.3 — EXTEND Soft-AV Rate Parity (2026-09-27)

- Fixed R38B EXTEND segments silently reverting to legacy 48 kHz async-resampled audio after the native checkpoint.
- R38B now preserves EXTEND native float PCM/sample-rate in native, upscale and RTX finish paths.
- Added PyAV hidden-context rate conformance as compatibility fallback for older/mixed segments.
- Video MASKED+EXACT EXTEND junction remains unchanged.
- Expected smoke invariant: `soft_av=2 | fallback_declick=0` for a 3-take / 2-seam render.

# IAMCCS Nodes - Changelog

## 2026-09-27 - Phase 9.2 - EXTEND hidden-context Soft AV + Settings PRO mode-control rail

- EXTEND video junction remains MASKED + EXACT with hidden pinned-head trim and direct butt join.
- Added audio-only hidden-context Soft AV handover: each continuation child preserves the final milliseconds of its hidden audio head before trim; the outgoing parent tail converges into that time-corresponding child context with qsin/equal-power gains.
- Removed the need to dip EXTEND seams to digital silence when hidden context is available; Phase 9.1 5 ms cos^2 de-click remains a per-seam fallback only.
- Mirrored EXTEND hidden audio context next to R38B delivery segments so R38B final masters consume the same Soft AV seam source.
- Added explicit EXTEND contract metadata `soft_audio_handover_ms=15.0` and `audio_master_policy=hidden_context_soft_av`.
- Refactored Settings PRO left rail: LONG TAKE · EXTEND, SAVED TAKE · CONTINUATION and REFMOD now live under a collapsible, visually distinct MODE CONTROLS folder.

## 2026-09-19 - version 1.5.5 - IAMCCS MiniMax H3 Engine

- Added MiniMax H3 Engine from the IAMCCS architecture.
- Documented GPL-3.0-or-later project licensing and third-party attribution.

## 2026-08-24 - version 1.5.2 - MiniMax H3 utilities and workflow polish

- Added MiniMax H3 audio-drive, audio-timeline and Shotboard utilities, including improved multi-chunk lane handling and long-video lip-sync support.
- Improved the MiniMax H3 atomic backend, device handling and compiler fallbacks for more resilient generation workflows.
- Refined Shotboard, dialogue-tag editor, prompter and exporter interfaces, with supporting RTX/VFX workflow improvements.
- Removed in-repository backup archives from distribution; backups are now kept outside the node package.

## 2026-07-21 - version 1.5.1 - Multigen Roll, Master Audio EDL Export and AudioBoard UX

- Added non-destructive pre-roll/post-roll support to the multigen editorial path.
- Added exporter roll de-duplication so revealed pre/post-roll windows retain correct audio/video lip sync without extending the final export with repeated material.
- Added IAMCCS Shotboarder Aud+Vid Exporter PRO direct-master workflow and encoded output preview.
- Improved AudioBoard control grouping and readability.

## 2026-07-12 - version 1.5.0 - Shotboard multigen pipeline, Shotboard Editor and AudioBoard

- Added the integrated Shotboard multi-timeline generation pipeline and take/audio routing.
- Added the Shotboard Video Editor workflow with video, audio and master-audio lanes.
- Added AudioBoard editing, real waveform handling, master-audio support and package export.
- Added timeline identity and audio-lane metadata alignment across generation and editing stages.

## 2026-06-19 - version 1.4.9 - Frame Designer V2 added

## 🆕 2026-06-04 - version 1.4.8 —Add audioboard arranger + utilities and new functions

## 🆕 2026-05-31 - version 1.4.7 —Add utilities and new functions and bugs fixed

## 🆕 2026-05-20 - version 1.4.6 — Shotboard planner v2 and v3 added

## 🆕 2026-05-12 - version 1.4.5 — Cine nodes added

## 🆕 2026-05-04 - version 1.4.4 — Supernodes v.2 and bug fixed pplus utilities added

## 🆕 2026-04-24 - version 1.4.3 — Supwenodes, more LTX and WAN 2.2 nodes and utilities added

## 🆕 2026-04-13 - version 1.4.1 — Multigen wrappers added

## 🆕 2026-03-27 - version 1.4.0 — Audio extension ltx-2.3

Changes:
- Added new nodes audio extension for LTX-2.3
- ADDED NEW NODES FOR LOW RAM USERS

Changes:
- Added new video utility node: `IAMCCS_WanJunctionBridge`
  - Adapts latent junctions between `SVI -> FLF` and `FLF -> SVI`
  - Cleans `prev_samples` tails via trim + tail extraction
  - Builds `anchor_samples` from decoded last frame, target anchor image, or latent fallback
  - Applies optional DC drift correction to preserve motion continuity across generations
  - Outputs both adapted `prev_samples` and `anchor_samples` for downstream Wan nodes

## 🆕 2026-02-24 — 🆕 Version 1.3.6  WanImageMotionPro + Motion Safety Preset + Bug Fixed

Changes:
- Added new video node: `WanImageMotionPro` (Motion + FLF End Lock)
  - Optional `end_samples` to lock the ending latent slots (FLF-style end control)
- Added `safety_preset` to motion nodes (`IAMCCS_WanImageMotion` and `WanImageMotionPro`)
  - `safe` (default): enables stabilizations only when `motion > 1.15`
  - `safer`: stronger stabilization for higher motion values
  - `legacy`: keeps the older behavior

Docs:
- Added `docs/wanimagemotion_instructions.md` (Simple + Pro guide + example recipes)
- Updated `docs/WanImageMotion.md`

## 🆕 Version 1.3.4 — Video Performance + Low-RAM Tools

Date: 2026-02-01

### Sampler wrapper (video workflows)
- Added `IAMCCS_SamplerAdvancedVersion1` ("Sampler Advanced v1"):
  - Delegates to ComfyUI `SamplerCustomAdvanced` (no algorithm swap), but adds:
    - `disable_progress` to reduce progress/UI overhead on long video queues
    - optional VRAM cleanup after sampling
  - Improved compatibility with newer ComfyUI return types (`NodeOutput`).

### VAE decoding (low RAM)
- Added `IAMCCS_VAEDecodeToDisk` ("VAE Decode → Disk"):
  - Frame-by-frame decode with on-disk output to minimize peak system RAM for long clips.

### Hardware probe
- Added `IAMCCS_HWProbeRecommendations`:
  - Exposes the HW probe recommendations as JSON + extracted fields usable in workflows.

### GGUF Accelerator
- Extended `IAMCCS_GGUF_accelerator`:
  - New patch move strategies (`move_policy`) and VRAM reserve (`leave_free_vram_mb`).
  - Input ordering kept backward-compatible with existing workflows.

### Frontend UX
- Bus Group: “Hide options” state now persists across sessions.
- HW probe apply is user-controlled:
  - apply mode: `overwrite` vs `fill_missing`
  - preset sync can be disabled to keep manual tuning.

---

## 🆕 Version 1.3.3 — AutoLink + LTX-2 Extension Module (Stability Update)

Date: 2026-01-26

### AutoLink (frontend)
- AutoLink Set/Get + Converter for compact “wireless” graphs
- Convert/Restore tools:
  - `Convert All Links`
  - `Restore Direct Links`
- Group-aware filters: `GroupExclude`, `GroupInOutExclude`
- Layout controls: multiple align modes (including `Proportional`) + packing/anti-overlap
- Styling controls: color presets, optional separate Set/Get colors, title text color
- Blacklist improvements: per-node (directional) and per-type entries

Stability fixes:
- AutoLink links are now materialized automatically during queue/prompt serialization (then restored), preventing “missing required input” prompt errors
- Works with nested graphs/subgraphs
- Long AutoLink titles are truncated with an ellipsis (`…`) to prevent overflow

### LTX-2 Extension (backend nodes)
- Added **LTX-2 Extension Module** (`IAMCCS_LTX2_ExtensionModule`):
  - Extends/merges image batches with overlap management
  - Built-in math operations for overlap/start-frames logic
  - AutoLink integration for overlap sharing between iterations (`autolink_overlap_in/out`)
  - Multiple blending modes: cut, linear_blend, ease_in_out, filmic_crossfade, perceptual_crossfade
  - Automatic `start_images` extraction for the next pass
  - `total_frames` / `validate_ltx2` moved out to dedicated validation utilities

- Added **LTX-2 Get Images From Batch** (`IAMCCS_LTX2_GetImageFromBatch`):
  - Extract frames from start/end or by explicit range

- Added **LTX-2 Frame Count Validator** (`IAMCCS_LTX2_FrameCountValidator`):
  - Validates/corrects counts to the LTX-2 `8n+1` rule
  - Intended to be placed before the LTX Sampler

### LTX-2 frame-count robustness
- `IAMCCS_LTX2_TimeFrameCount` snaps computed `length` to the next valid `8n+1`
- UI seconds↔length sync snaps to valid `8n+1` lengths
- Optional VAE encode auto-padding to valid `8n+1` (defensive safeguard)

---

## 🆕 Version 1.3.2 — LTX-2 Nodes Pack

Date: 2026-01-15

Changes:
- Added new LTX-2 LoRA nodes:
  - `IAMCCS_LTX2_LoRAStackModelIO` ("LoRA Stack (Model In→Out) LTX-2")
  - `IAMCCS_LTX2_LoRAStackStaged` ("LoRA Stack (LTX-2, staged: stage1+stage2)")
  - (Existing apply node) `IAMCCS_ModelWithLoRA_LTX2` used to apply staged stacks per stage

- Added new LTX-2 utility nodes:
  - `IAMCCS_LTX2_FrameRateSync` ("LTX-2 FrameRate Sync (int+float)") — includes `fixed` mode
  - `IAMCCS_LTX2_Validator` ("LTX-2 Validator (16px, 8n +1)") — replaces the removed ShapeValidator; seconds/length are UI-synced; fps handled by FrameRateSync
  - `IAMCCS_LTX2_ControlPreprocess` ("LTX-2 Control Preprocess (aux)")

- Added documentation: `LTX2iamccsnodes.md`
- Bumped versions (`version.json`, `pyproject.toml`) to 1.3.2.

---

## 🆕 Version 1.3.1 — WAN SVI Pro Motion Control Node

Date: 2026-01-07

Changes:
- Added new node `IAMCCS_WanImageMotion` ("IAMCCS WanImageMotion"):
  - Drop-in replacement for KJNodes `WanImageToVideoSVIPro` with motion amplitude control
  - Multiple motion modes: apply to `prev_samples` only or all non-first latents
  - VRAM profiles: normal / chunked (2/4 blocks) / per-frame loop / CPU offload
  - Latent precision control: auto / fp16 / fp32 for quality vs VRAM tradeoff
  - Optional `include_padding_in_motion` toggle: allows motion boost on padded frames when anchor has single frame
  - Optional `add_reference_latents` for additional conditioning stability
  - Comprehensive logging with motion_range diagnostics and warnings when no frames are modified

- When anchor_samples has only 1 frame and no prev_samples: enable `include_padding_in_motion=True` to apply motion boost
- Full documentation in `docs/WanImageMotion.md`

## 🆕 Version 1.3.0 — MODEL In→Out LoRA Stack

Date: 2025-11-19

Changes:
- Added new node `IAMCCS_WanLoRAStackModelIO` ("LoRA Stack (Model In→Out) WAN") for direct multi-LoRA application to an incoming MODEL (WAN 2.2 / Flow / Standard).
- Preserves WAN key remap + optional chaining via existing `IAMCCS_WanLoRAStack` (use optional `lora` input to extend beyond 4 slots).
- Bumped versions (`version.json`, `pyproject.toml`) to 1.3.0.
- Neutralized deprecated Save&Load DragCrop code (frontend/backend) — removed from active registration.

Notes:
- Existing workflows using the older two-node stack + apply pattern continue to work unchanged.
- Use `IAMCCS_WanLoRAStackModelIO` to simplify WAN 2.2 graphs or reduce node count before samplers.

---


---
## 🆕 Version 1.2.3 — Stackable LoRA Input

- Added optional `lora` input to IAMCCS_WanLoRAStack node
- Now supports up to 8 LoRA models in total (4 direct + 4 from chained stack)
- Enable daisy-chaining multiple IAMCCS_WanLoRAStack nodes for extended LoRA capabilities
- Maintains backward compatibility with existing workflows

## 🆕 Version 1.2.1 — Extended Wan 2.1 Compatibility

 Removed the deprecated LightX2V node (now merged into WAN-style remap)
- Updated `__init__.py` and mappings
- Improved overall LoRA compatibility with Wan style remap node
- Added new `CHANGELOG.md` file

## 🆕 Version 1.2.0 — LightX2V Update
Extended support for WAN 2.2 LoRA models (LightX2V).

This release introduces a new node — **LightX2V (Remap)** — which supports both **WAN 2.2 high and low LoRA models** as well as **character LoRAs**.  
For detailed compatibility, check the following list of LoRAs supported by each node:
