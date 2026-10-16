# IAMCCS H3 R45 Universal — LongVid + Integrated Scout + Optional Editor

## Production topology

The R45 Universal workflow merges the former R43 Universal Scout/Editor graph with the R44.8.1 LongVid corrections. The generation router contains the standard R42 path plus FL2VA Continuous AV, Guided AV Loop and Latent New branches. Public `LONG MULTI-SHOT · HARD CUT` remains on the independent-shot standard path; public `LONG KEYFRAME CONTINUOUS` uses the FL2VA Continuous AV branch and `phase_aligned_extended` full-AV latent handover.

Legacy task IDs remain load-compatible. `longvid_guides` is still readable but is not the preferred continuous keyframe mode.

## Scout

`Settings PRO > ENGINES > SCOUT` is the R45 user-facing Scout control room. The existing R40 candidate nodes remain the backend/cache truth. The panel exposes the Scout settings, SCOUT/DELIVER phases, four cached candidate previews, candidate seeds and winning-take selection. Preview playback is reconstructed from the existing R40 PreviewImage frame cache; changing the selected take does not resample the candidate bank.

Scout is bypassed for continuous/native-history engines, including `longvid_keyframe_continuous`, because replacing one interval with a standalone candidate would destroy the full-AV latent handoff. `LONG MULTI-SHOT · HARD CUT` remains Scout-compatible because its shots are intentionally independent.

When Scout is OFF, the old R40 phase property no longer mutes production nodes. Normal production is active.

## Optional Video Editor

`h3_editor_enabled` is append-only and defaults to OFF in R45.

Editor OFF:

1. H3 generates all technical chunks/shots required by the selected mode.
2. The normal Universal delivery/checkpoint chain assembles the final programme/master exactly as before.
3. The R45 final lazy gate requests the direct final master path.
4. The Video Editor branch is not executed.

Editor ON:

1. The same normal master/roll assembly still occurs first.
2. The R45 final lazy gate requests the optional Editor branch.
3. The new lazy exporter is evaluated only in this route.

The historical Shotboarder Exporter PRO remains a terminal `OUTPUT_NODE`; only the new R45 Universal workflow uses the non-terminal lazy subclass. This prevents regression in older workflows.

## R44.8.1 fixes retained

- `LONG KEYFRAME CONTINUOUS` public mode and R43+ Continuous router.
- 39-frame context preservation; 22 remains only the default.
- Pianosequenza endpoint compatibility guard.
- `phase_aligned_extended` continuous full-AV handover with zero decoded video crossfade.
- Custom Audio authority for continuous/multishot LongVid.
- whole-programme Face Refinement source for joined Continuous masters.
