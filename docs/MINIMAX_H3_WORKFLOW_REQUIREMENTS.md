# IAMCCS MiniMax H3 Shotboard Workflow Requirements

This document covers the IAMCCS MiniMax H3 Shotboard production workflows,
including the native H3 route, Turbo sampling, live preview, LTX/Wan finishing,
RIFE interpolation and optional RTX Video Super Resolution delivery.

## Base runtime

- A current ComfyUI build with native MiniMax H3 AV conditioning/sampling and
  current LTX audio-video nodes.
- A recent NVIDIA driver and a PyTorch/CUDA build compatible with the selected
  attention extensions. Reinstall compiled attention wheels after changing the
  PyTorch/CUDA build.
- `IAMCCS-nodes` installed once in `custom_nodes`. Remove or move duplicate and
  backup copies outside `custom_nodes`, otherwise ComfyUI can register stale
  classes or report import failures.
- NVIDIA CUDA GPU for the supplied accelerated graphs. Twelve GB VRAM is
  supported through dynamic weight offload; more VRAM reduces offload and wait
  time. At least 32 GB system RAM is practical, while 64 GB is recommended for
  H3 plus an LTX/Wan finishing pass.

## Required node packs for the supplied H3 graphs

### IAMCCS-nodes

Provides the Shotboard and its workflow-facing wrappers:

- `IAMCCS_MiniMaxH3ShotPlanner`
- `IAMCCS_MiniMaxH3AtomicModelRouter`
- `IAMCCS_MiniMaxH3AtomicConditioningBackend`
- `IAMCCS_MiniMaxH3GenerationBackendV2`
- `IAMCCS_MiniMaxH3PostUpscaleControlV2`
- `IAMCCS_MiniMaxH3DeliveryRouterV2`
- `IAMCCS_MiniMaxH3SegmentQueueLoop`
- `IAMCCS_MiniMaxH3SequentialLTXLoaderV2`
- `IAMCCS_MiniMaxH3OptionalLTXDetailerLoRA`
- `IAMCCS_MiniMaxH3RTX4KPost`
- `IAMCCS_Prompter` in the Prompter editions

Repository: https://github.com/IAMCCS/IAMCCS-nodes

### ComfyUI-GGUF

Required when either the H3 diffusion model or Qwen3-VL text encoder is loaded
from GGUF. The supplied graphs use `UnetLoaderGGUFAdvanced` and
`CLIPLoaderGGUF`.

Repository: https://github.com/city96/ComfyUI-GGUF

### ComfyUI-KJNodes

Used by the supplied graphs for MiniMax H3 Sage/low-VRAM patches, image resize,
TAEH3 preview override, KJ VAE loading and the LTX spatiotemporal tiled decode.

Repository: https://github.com/kijai/ComfyUI-KJNodes

### MiniMax H3 Turbo

Required only when the Shotboard Turbo route is enabled. It supplies the Turbo
LoRA loader and `MiniMaxH3TurboSampler` used by the reference Turbo graphs.

Repository: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo

## Optional finishing and acceleration packs

- LTX Video nodes: recent ComfyUI contains the native LTX path used by the
  current graph. `ComfyUI-LTXVideo` remains useful for compatible extended LTX
  workflows: https://github.com/Lightricks/ComfyUI-LTXVideo
- Wan finishing: install the node pack required by the selected Wan branch;
  the IAMCCS reference environment uses
  https://github.com/kijai/ComfyUI-WanVideoWrapper
- RIFE frame interpolation:
  https://github.com/Fannovel16/ComfyUI-Frame-Interpolation
- Sol Attention: https://github.com/kijai/ComfyUI-SolAttn_triton
- Spectrum for MiniMax H3:
  https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3
- MiniMax H3 Adaptive Cache:
  https://github.com/FFFFFFpy/ComfyUI-MiniMaxH3-AdaptiveCache

These accelerators are alternatives or composable options only where the
Shotboard/backend explicitly reports them as active. A node merely present in
the graph does not accelerate an execution path that is bypassed.

## Optional RTX 4K delivery

The RTX final pass uses `RTXVideoSuperResolution` from:

https://github.com/BetaDoggo/comfyui-rtx-simple

It additionally requires NVIDIA VFX. Install it in the exact Python environment
that launches ComfyUI:

```text
python -m pip install nvidia-vfx --extra-index-url https://pypi.nvidia.com/
```

RTX VSR is an optional final delivery stage. It does not replace the LTX
generative finishing pass, and it must stay bypassed when `RTX final 4K` is off
in the Shotboard settings.

## Model families expected by the workflow

- One MiniMax H3 T2VA/I2VA/FL2VA model and, for REF2VA, the compatible REF2VA
  model. Full, pruned INT8 and GGUF variants can be routed when their loader is
  compatible with ComfyUI's native H3 model type.
- The matching Qwen3-VL MiniMax H3 text/vision encoder.
- MiniMax H3 video VAE and audio VAE.
- TAEH3 decoder for the live denoise preview.
- Turbo LoRA matching the chosen base model when Turbo is enabled.
- For LTX finishing: the selected LTX diffusion model, Gemma text encoder,
  text projection, video VAE and audio VAE. A finishing LoRA is optional; the
  Shotboard selector intentionally exposes all compatible installed LTX LoRAs.
- The selected latent/upscale model when the LTX graph uses a latent upres stage.

## Low-VRAM execution contract

The H3 backend uses a phased memory contract:

1. Qwen3-VL conditioning runs GPU-first.
2. On GPUs up to 17 GB, IAMCCS supplies a temporary activation reserve so
   ComfyUI dynamically offloads some Qwen weights instead of filling VRAM with
   the entire encoder.
3. FL2VA keyframes or REF2VA media are encoded by their VAE.
4. `unload_all_models()`, model cleanup and CUDA cache cleanup run before the H3
   sampler requests the diffusion model.
5. A full CPU conditioning retry is used only after a genuine CUDA OOM.

On a cold run, look for a log line containing `dynamic_reserve`, followed by
`conditioning complete`, `pre-sampler barrier`, and only then
`Requested to load MiniMaxH3`. If a warm Queue reuses cached conditioning, the
text-encoder load lines may legitimately be absent.

## Installation validation

After restarting ComfyUI:

1. Open the workflow and confirm there are no red or `UNKNOWN` nodes.
2. Queue a short native H3 take with upscale, RIFE and RTX disabled.
3. Confirm the preview tap updates and the sampler advances.
4. Test LTX/Wan, RIFE and RTX as separate finishing checks before combining
   them in a long multi-segment render.
5. Confirm the final saver uses numbered filenames so no previous take is
   overwritten.
