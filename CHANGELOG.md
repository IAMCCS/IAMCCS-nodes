# 1.5.8 - H3 IAMCCS utilities (2026-10-16)

- Add LoRA-based Character Swap and reference-image environment editing, plus a dedicated video VFX edit mode and Prompter media routing.
- Make native single-pass delivery the SwapChar default; expose the optional H3 Detail second pass through synchronized Native and mode-engine controls.
- Separate sampling recipes from attention/memory selection in Settings PRO; add compact Turbo groups and VEDA compatibility handling while preserving legacy widget order.
- Update LongVid continuity engines, shared AV boundary handling, stage-two guide rebuilding, tiled LTX sampling, and the local licensed H3 latent upscaler.
- Preserve Shotboard audio authority, dialogue stem timing, optional source audio and editor/export handoff.
- Keep the package ID unchanged and align pyproject.toml and version.json at 1.5.8. The local version.json previously declared 1.5.7 while the repository declared 1.5.5.

## R51 - 2026-10-09 - IAMCCS-internal PSkf Stage-2 upscale

- Fast Latent 2-Pass no longer resolves OBVPM context-window nodes or the external MiniMax H3 3D upscaler through the global ComfyUI node registry.
- The per-sigma PSkf window planner and pyramid fusion execute from the IAMCCS-local provider. The MIT 3D latent upscaler implementation and license are vendored under vendor/.
- Model-folder registration now happens from the local module before checkpoint lookup, so the route boots after the external custom-node folders are removed.
- Restored masked still-guide pinning and exact audio-mask preservation for both full and windowed Stage-2 sampling. The validated first pass is unchanged.
- Added regression coverage for external-node independence, local imports, guide pins, audio lock and current upscaler call contract.

## R49 — 2026-10-08 — True Long shared AV boundary

- Public `longvid_keyframe_continuous` and the explicit True Long alias now enter the PSkf planner before legacy-engine resolution. Saved `longvid_continuous_engine` values are ignored for this mode; historical engines remain installed and available to their existing workflows.
- Fixed the exclusive PSkf endpoint being consumed as inclusive (one hidden frame per continuation).
- True Long intermediate delivery now ends at the exact inherited AV boundary, with a zero-replay invariant. Added short static-tail detection and an isolated Settings Pro panel.
- Native True Long joins now use a 35 ms equal-gain audio boundary to suppress small generated-audio clicks. Authored Custom Audio remains untouched and is still muxed as the pristine master.
- Final delivery trims a single strictly detected duplicate terminal frame as well as longer short holds; intermediate PSkf endpoints keep the stricter threshold and are unchanged.
- Universal R49 already exposes the safe complete-film upscale route as `LTX 2.3 MASTER`: True Long is assembled first and its audio is preserved. The H3 Fast Latent 2-Pass branch remains per-segment and is not advertised as a True Long master route.
- CPU boundary and real continuation-construction tests pass. GPU seam/guide validation remains required; internal model-generated cuts are not certified fixed. Legacy workflows remain available.

## R48 — 2026-10-08 — Pianosequenza-True_long PSkf core

- Integrated the explicit True_long task in Universal/Settings Pro, replacing the visible positioned-guided choice while retaining its legacy backend identifier. Continuous AV rendering now packs and saves PSkf checkpoints and uses True_long Extend between intervals. CPU routing smoke passes; GPU visual validation and PAN exposure remain pending.

- Added the autonomous `Pianosequenza-True_long` core and versioned `PSkf` (Pianosequenza State Keyframe) checkpoints. PSkf persists the complete native H3 video+audio latent, delivery ranges, lineage and handover metadata in `.pskf.safetensors` files under ComfyUI output.
- Added Pack, Save, Load and Extend nodes. Extend delegates to the validated IAMCCS phase-aligned native AV continuation primitive and does not import or call OBVPM.
- Removed the optional MCtx adapter from default node registration. Its source remains only as an audit/A-B reference; product workflows no longer expose or require external OBVPM nodes.
- Prepend and Bridge are intentionally not advertised as complete until their native AV masking and trim contracts pass GPU render validation.

## R47 — 2026-10-08 — optional OBVPM MCtx route

- Added one append-only `IAMCCS H3 LongVid · MCtx Route` adapter. It exposes user-facing Extend, Prepend and Bridge presets but delegates latent slicing, native guide/masked/both pins and trim ownership to the installed GPL-3.0 `comfyui-obvpm-timeline` nodes.
- The adapter does not replace Pianosequenza 1/R46 and does not copy OBVPM tensor code. It fails closed when OBVPM is unavailable and keeps the experimental MCtx route isolated for A/B smoke tests.

## R46 — 2026-10-08 — LONG KEYFRAME · native TimelineDirector restore

- Public `longvid_keyframe_continuous` now routes through the historical sampled native VIDEO+AUDIO latent-tail/AudioCon engine instead of R45 pairwise FL2VA; technical windows keep incoming-owned 22f overlap and remove just a one-frame RGB bridge.
- Three Pianosequenza engine options in Settings PRO → ENGINES → LONG KEYFRAME. PIANOSEQUENZA 1 default; 2 DRIFT / 3 PHASE experimental. Appended optional widget for saved-workflow position stability.
- Editorial-endpoint tail checkpoints, LATENT FREE default, CPU regression tests for 30s/720f, video/audio mask transport and compatibility. Native Delivery required until full GPU validation.
- See `docs/R46_LONGVID_NATIVE_TIMELINEDIRECTOR_RESTORE_IT.md`.

# R45.2 candidate — authored interval budget and native masked AV history (2026-10-07)

- LONG KEYFRAME CONTINUOUS adds history to the technical duration, positions the destination at the authored endpoint and strips only technical padding when no freeze is detected.
- Carries raw AV history in the target using native per-stream masks. Quantization replay stays hidden; public video joins no longer replace frames with Safe Tail Bridge pixels.
- Separates visible landing from latent cutoff, keeps the first detected landing frame, removes artificial editorial audio silence and adds checkpoint duration diagnostics.
- Custom Audio covers the full technical interval. A shortened interval cannot silently produce a desynchronized custom-audio master.
- Ordinary FLF, Extended AV and legacy continuation aliases keep their own contracts. See `docs/R45_2_CONTINUOUS_AUDIT_2026-10-07.md` for tests and the limits of visual validation; these corrections do not guarantee a continuous model-generated trajectory.

# R45.1 — Long Keyframe motion-safe landings + live segment checkpoints (2026-10-07)

- Fixed LONG KEYFRAME CONTINUOUS preserving H3 terminal still-frame locks as visible programme footage. The public mode now uses the freeze analyzer as a real motion-safe handover authority: intermediate visible tails are trimmed at the same phase-aligned cutoff used by the next latent continuation.
- Continuation intervals no longer inherit the last frozen latent frames merely to preserve an authored guide. The next FL2VA interval still targets the next authored Shotboard image, but its AV history ends before the detected lock. Legacy serialized continuous aliases retain their historical behavior.
- Final interval now removes only a detected terminal lock and keeps the last genuinely moving frame immediately before it; when no lock is detected the complete ending is preserved. This removes the final snap/hold on the last authored image.
- Re-enabled the detector-approved Safe Tail Bridge (up to 2 frames) for the public continuous mode so phase-alignment quantization can retain the latest safe rendered approach without borrowing from the freeze safety margin.
- Added live per-segment checkpoints for LONG KEYFRAME CONTINUOUS. Every cleaned interval is encoded before final assembly and published immediately to a video player on the FL2VA CONTINUOUS AV node, so a bad interval can be reviewed and the run interrupted before all later intervals finish.
- Segment checkpoint videos show the exact delivery interval after reused-head and freeze-tail removal, not the raw technical H3 sample. Final master assembly remains mandatory and independent of the optional Video Editor.

# R45.0 — Universal LongVid / Integrated Scout / Optional Editor (2026-10-07)

- Unified the R43 Universal Scout/Editor topology with the R44.8.1 LongVid fixes: Standard/Motion Context, Long Multi-Shot Hard Cut, Long Keyframe Continuous, Guided AV Loop Experimental and Latent New now coexist in one Universal graph while legacy task IDs remain readable.
- Added `longvid_keyframe_continuous` to the Scout history-preservation bypass: Scout can never replace a full-AV continuous handoff with a standalone candidate. Long Multi-Shot Hard Cut remains Scout-compatible because its authored shots are intentionally independent.
- Settings PRO > ENGINES > SCOUT is now the operational Scout surface. It mirrors the cached R40 candidate bank, replays the existing candidate preview-frame batches as inline animated previews, shows candidate seeds, selects the winning take, and switches SCOUT / DELIVER phases without moving cache truth into the frontend.
- Fixed the legacy R40 phase controller so Scout OFF means normal production; it no longer mutes production/editor nodes merely because the saved phase property says `scout`.
- Added append-only `h3_editor_enabled` (default OFF). The new R45 final lazy gate runs after normal delivery/master assembly: OFF returns the direct joined master; ON requests the optional Video Editor branch. Chunk/master joining is never bypassed.
- Added a non-terminal R45 exporter subclass used only behind the optional Editor gate. The historical exporter remains an OUTPUT_NODE, so old workflows retain their previous terminal behaviour.
- New Universal workflow keeps the R40 Scout backend/cache nodes for compatibility but treats them as internal plumbing; the Settings PRO Scout panel is the intended user-facing control room.

# R44.8 — LongVid cleanup / dedicated Engines / continuous custom-audio authority (2026-10-07)

- Added public `LONG MULTI-SHOT · HARD CUT`: every authored Shotboard slot is an independent H3 shot with a real hard-cut boundary, while one Custom Audio master can remain the performance/lip-sync timing authority across the full edit.
- Added public `LONG KEYFRAME CONTINUOUS`: N authored keyframes compile to N−1 FL2VA intervals using `phase_aligned_extended` direct full native AV latent handover; decoded video crossfade defaults to 0 so technical boundaries are not hidden by dissolves.
- Preserved historical LongVid IDs as load-compatible aliases. Legacy `longvid_guides` remains readable but is hidden from the primary Mode Assistant for new work.
- Added a defensive Pianosequenza endpoint fallback: historical workflows requesting a latent-tail endpoint without `iamccs_longvid_latent_tail_ab` no longer crash at the next/final chunk.
- Settings PRO now exposes a dedicated Engine panel for each public Mode Assistant mode. Every mode card has an `ENGINE ▸` control that selects the mode and opens only its relevant engine variables; `DIRECT` remains visible as `DIRECT · LEGACY` for compatibility.
- Added mode-specific Engine presets for Long Multi-Shot, Long Keyframe Continuous and Keyframe Joint Native/Latent-New experimental operation.
- `h3_custom_audio_drive` is now supported by Long Keyframe Continuous: exact timeline slices are VAE-locked into every technical interval for performance/lip-sync authority, including the hidden phase-aligned continuation head, while the final mux restores one pristine global Custom Audio master.
- Long Multi-Shot keeps the standard per-shot locked Custom Audio drive while final assembly preserves one continuous authored audio timeline across hard video cuts.
- Added regression coverage for both new public LongVid contracts, legacy alias loading, custom-audio preservation and Settings PRO public-mode state.
- Face Delivery now recognizes the whole-programme output of Long Keyframe Continuous: it tracks/refines the joined native frame master once instead of trying to decode the final technical interval latent as though it represented the complete programme. Ordinary I2V/FLF face delivery remains on the existing latent-based path.

# R44.7 — Mixer realtime meters / smooth faders + Face preset stack + RTX UI polish (2026-10-06)

- Video Editor audio slave now exposes realtime A1–A5 + master meter snapshots to IAMCCS AudioBoard Mixer.
- Mixer fader/gain changes no longer force-seek preview audio; preview volume is smoothed to eliminate click/crackle while dragging controls.
- Face Refinement Quality / Face Size / Definition preset groups now keep independent simultaneous active selections.
- Face preset families are orthogonal: Quality no longer overwrites Definition tuning, and Definition no longer overwrites Face Size scaling.
- Added `RESET PRESETS · CURRENT DEFAULTS` without changing Face Refine enable/mode/max-face policy.
- Fixed Face preset contextual help leaking the same preset description into all three groups.
- Compact RTX runtime action buttons prevent text overlap while retaining full English tooltips.

# R44.6 — AudioBoard Mixer → Video Editor Audio Slave (2026-10-06)

- Fixed `mixer_overrides_cine_linx`: authored Mixer track volume/gain/pan/mute/solo now wins over upstream generation defaults.
- Added live Video Editor audio-slave API: A1–A5, master gain and master bus follow IAMCCS AudioBoard Mixer even without AudioBoard Arranger.
- Mixer transport can audition the downstream Video Editor when no Arranger is connected.
- Video Editor preview now includes track gain dB and Mixer master gain.
- Final Video Editor render now bakes imported Mixer `masterAudioGain` in addition to track settings and master bus FX.
- Deno remains untouched: IAMCCS controls its own editorial audio state only.

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
