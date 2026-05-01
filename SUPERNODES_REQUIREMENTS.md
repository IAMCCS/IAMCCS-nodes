# IAMCCS SuperNodes Requirements

The IAMCCS SuperNodes are workflow wrappers. They do not replace the underlying
ComfyUI, LTXV, audio, VAE, stitching, and helper nodes; they orchestrate them.
If one dependency is missing or outdated, the SuperNode may load in the graph but
fail at prompt validation, skip a backend path, or produce unstable results.

## Required Base

- A recent ComfyUI build with native LTXV/LTX-Video audio-video nodes available.
- A working LTXV audio-video model setup, including model, CLIP/text encoder,
  Video VAE, and Audio VAE.
- `IAMCCS-nodes` installed as the active package, with no duplicated copies in
  `custom_nodes`.

## Required ComfyUI/LTXV Nodes

These are used directly by the AU+IMG2VID SuperNodes backend:

- `EmptyLTXVLatentVideo`
- `LTXVConditioning`
- `LTXVPreprocess`
- `LTXVImgToVideoInplace`
- `LTXVConcatAVLatent`
- `LTXVSeparateAVLatent`
- `LTXVCropGuides`
- `LTXVAudioVAEEncode`
- `LTXVEmptyLatentAudio`
- `LTXVAudioVAEDecode`
- `BasicScheduler`
- `CFGGuider`
- `KSamplerSelect`
- `RandomNoise`
- `SamplerCustomAdvanced`

## Required IAMCCS Helper Nodes

These ship with `IAMCCS-nodes` and are expected by the wrapper:

- `IAMCCS_SegmentPlanner`
- `IAMCCS_SamplerAdvancedVersion1`
- `IAMCCS_AudioExtensionMath`
- `IAMCCS_AudioExtender`
- `IAMCCS_AudioTimelineGate`
- `IAMCCS_LTX2_ExtensionModule`
- `IAMCCS_LTX2_ExtensionModule_Disk`
- `IAMCCS_StartImagesToVideoLatent`
- `IAMCCS_StartDirToVideoLatent`
- `IAMCCS_VAEDecodeToDisk`
- `IAMCCS_VAEDecodeTiledSafe`
- `IAMCCS_VideoCombineFromDir`
- `IAMCCS_VRAMFlushLatent`

## Required External Custom Nodes

- VideoHelperSuite / VHS, for final image/audio video combine paths.
- MTB nodes, for `Audio Duration (mtb)` used by planner duration math.
- MelBand RoFormer nodes, for vocal-focused conditioning:
  - `MelBandRoFormerModelLoader`
  - `MelBandRoFormerSampler`

If MelBand is not installed, use the planner audio mode `raw_audio_only`. For
spoken/lipsync workflows, MelBand is recommended because the model receives a
cleaner vocal conditioning signal.

## Required Models And Assets

- LTXV audio-video model compatible with the workflow.
- Video VAE compatible with the LTXV model.
- Audio VAE compatible with the LTXV audio latent path.
- Optional MelBand model, for example `MelBandRoformer_fp32.safetensors`.
- Optional latent upscaler model only if the beta second-stage upscaler is used.

## Recommended Stable Settings

For first validation, keep the SuperNode close to the reference simple workflow:

- `generation_type`: `aud+img2video_simple`
- `ui_preset`: `custom`
- `backend_mode`: `single_best` or `auto`
- `vae_mode`: `normal_tiled_vhs`
- `planner audio mode`: `melband_vocals_duration_math`
- `second_stage_mode`: `off`
- `vram_flush`: `off`, unless VRAM pressure requires it

After the simple route is confirmed, test segmented and loop routes separately.

## Common Failure Symptoms

- Missing dependency: the graph may validate with missing-node errors, or a
  SuperNode may report that a helper node is unavailable.
- Duplicated `IAMCCS-nodes` folder: ComfyUI may fail at startup, register routes
  twice, or load an old node definition.
- Old saved workflow with changed widget order: refresh/recreate the affected
  SuperNode if dropdown values appear in the wrong fields.
- No lipsync: verify that the log reports `uses_input_audio=True`,
  `generates_audio=False`, `conditioning_audio=linx_audio_conditioning_single`,
  and `melband=True` when MelBand is expected.
- Distortion over time: see the next section.

## Why Frames Can Distort As Generation Continues

Long video generation is not a single static image process. Each segment depends
on an initial image/latent, audio conditioning, frame count, VAE decode, and
stitch/continuity settings. Distortion over time is usually caused by one or more
of these factors:

- The segment is too long for the model to keep identity and geometry stable.
- The prompt asks for more motion than the source image can support.
- `image_strength` is too low, so the source image stops anchoring the face/body.
- `motion_intensity` is too high, causing exaggeration and temporal drift.
- Anchor refresh is off or too weak in multi-segment/loop routes.
- The previous segment tail is used as the next anchor after it has already
  degraded, so each continuation compounds the error.
- Overlap/stitch settings blend incompatible frames or reuse the wrong side.
- Width/height or VAE mode differs from the validated reference workflow.
- Audio conditioning is weak, noisy, or not vocal-focused.
- VRAM pressure/offload changes timing or forces lower-memory decode paths.

For troubleshooting, validate a single segment first. Then add segmentation with
`cut` stitching, 9 overlap frames, MelBand vocal conditioning, and conservative
motion. Only after continuity is stable should second-stage refine, upscaling,
or stronger anti-drift be enabled.

