# H3 IAMCCS utilities - 1.5.8

By IAMCCS AI Research

This update publishes the H3 backend and Settings PRO changes used by the R49 workflows. Existing node IDs and the package registry ID are retained.

## Included behavior

- Character Swap LoRA: source video, replacement identity, optional second reference for the environment, and original-audio handling. Native single pass is the default.
- Optional H3 Detail second pass: the Native delivery toggle and SwapChar engine variables share the same settings. This is a generative delivery/upscale pass; SAM face refinement remains a separate option.
- Prompter: identity swap, identity plus environment, and video VFX edit authoring. Reference/video selection can use Prompter media or backend inputs; sampled video frames provide vision evidence while generation uses the source sequence.
- Settings PRO: compact family-specific Turbo recipes, independent attention/memory presets, and VEDA controls with incompatible combinations rejected. Legacy acceleration values and saved widget order remain supported.
- LongVid: restored continuity routing, native-tail/Extended engine selection, PSkf shared AV boundaries and intermediate checkpoint preview.
- Delivery: local licensed H3 latent upscaler, stage-two guide reconstruction, LTX tiled sampling, dialogue timing and Shotboard/editor audio handoff fixes.

## Installation and models

Update this node pack, restart ComfyUI when the queue is idle, and refresh the browser frontend. H3 model implementations and optional node packs remain separate installations. This commit contains code and notices only: no model weights, LoRAs, media, private workflow JSONs or API credentials.

Character Swap and VFX require their matching H3 LoRAs. Turbo weights must match the selected task family and recipe. VEDA requires its predictor. Attention and delivery backends require their corresponding optional dependencies. Existing WAN, Krea and other node families remain available.

## Validation limits

Automated checks exercise Python syntax, JavaScript syntax, planner/adapter contracts, settings serialization, delivery routing and mocked sampling/audio handoffs. CPU contract checks do not establish GPU quality or performance for every optional engine, duration and model combination.

Release checks: 41 relevant test files passed. The unchanged Bus Group widget-identity test also fails on the original repository HEAD in this environment; its implementation is outside this H3 update. Optional external-media fixtures are opt-in and skipped when absent. No full GPU render was run for this packaging commit.
