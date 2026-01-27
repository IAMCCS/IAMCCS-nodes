# IAMCCS Nodes - Changelog

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
