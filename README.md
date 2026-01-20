# 🌀 IAMCCS-nodes

<img src="icon.png" width="150" height="150">

## Author: IAMCCS (Carmine Cristallo Scalzi)

### Category: ComfyUI Custom Nodes
### Main Feature: Fix for LoRA loading in native WANAnimate workflows

Version: 1.3.2

# UPDATE VERSION 1-3-2

## 🆕 Version 1.3.2 — LTX-2 Nodes Pack

![[Node piece](assets/cover_sq.png)](https://github.com/IAMCCS/IAMCCS-nodes/blob/main/assets/cover_sq.png)

Highlights:
- Added/updated **LTX-2 LoRA nodes** (category `IAMCCS/LoRA`):
  - `LoRA Stack (LTX-2, 3 slots)` (`IAMCCS_LTX2_LoRAStack`)
  - `LoRA Stack (LTX-2, staged: stage1+stage2) (BETA)` (`IAMCCS_LTX2_LoRAStackStaged`)

![[Node piece](assets/stage.png)](https://github.com/IAMCCS/IAMCCS-nodes/blob/main/assets/stage.png)

  - `Apply LoRA to MODEL (LTX-2, quiet logs)` (`IAMCCS_ModelWithLoRA_LTX2`)
  - `Apply LoRA to MODEL (LTX-2, staged) (BETA)` (`IAMCCS_ModelWithLoRA_LTX2_Staged`)
  - `LoRA Stack (Model In→Out) LTX-2` (`IAMCCS_LTX2_LoRAStackModelIO`)

![[Node piece](assets/validator.png)](https://github.com/IAMCCS/IAMCCS-nodes/blob/main/assets/validator.png)

- Added/updated **LTX-2 workflow utilities** (category `IAMCCS/LTX-2`):
  - `LTX-2 FrameRate Sync (int+float)` (`IAMCCS_LTX2_FrameRateSync`) — keeps FPS INT/FLOAT consistent.
  - `LTX-2 Validator (16px, 8n +1)` (`IAMCCS_LTX2_Validator`) — EmptyImage-like IMAGE + validated `length` output; enforces `8n+1` and a permissive spatial multiple (16px).
    - `fps` is handled by `LTX-2 FrameRate Sync` (no fps input on the Validator).
    - `seconds` + `length` are both visible; the UI auto-syncs them.
  - `LTX-2 TimeFrameCount` (`IAMCCS_LTX2_TimeFrameCount`) — duration-only helper for I2V workflows: `seconds` ↔ `length` kept in sync in the UI (uses nearest FrameRateSync, fallback 24fps).
  - `LTX-2 Control Preprocess (aux)` (`IAMCCS_LTX2_ControlPreprocess`) — lightweight grayscale/threshold/edges helper for control-style workflows.

![[Node piece](assets/frame.png)](https://github.com/IAMCCS/IAMCCS-nodes/blob/main/assets/frame.png)

---

# UPDATE VERSION 1-3-1

## 🆕 Version 1.3.1 — WAN SVI Pro Motion Control


![[Node piece](assets/wanmotion.png)](https://github.com/IAMCCS/IAMCCS-nodes/blob/main/assets/wanmotion.png)

Highlights:
- Added `IAMCCS WanImageMotion` node: drop-in replacement for KJNodes `WanImageToVideoSVIPro` with motion amplitude control to fix slow-motion issues in WAN SVI Pro workflows.
- Motion modes: apply boost to `prev_samples` only or all non-first latents.
- VRAM profiles: normal / chunked / per-frame loop / CPU offload for memory-constrained systems.
- `include_padding_in_motion` toggle: enables motion boost on padded frames when anchor has single frame (T=1).
- Comprehensive logging with warnings when motion_range is empty.
- Full documentation: `WanImageMotion.md`
- Removed the previously included external-model LoRA loader node and related documentation.

### New Node: IAMCCS WanImageMotion

Use this node in WAN SVI Pro workflows to control motion intensity and prevent slow-motion artifacts.

Inputs:
- `positive` / `negative`: conditioning
- `length`: video length
- `anchor_samples`: base latent samples
- `motion`: motion amplitude (1.0-2.0, default 1.15)
- `motion_mode`: choose where to apply boost
- `motion_latent_count`: frames from prev_samples to use as motion reference
- `include_padding_in_motion`: enable to apply motion on padded frames
- `vram_profile`: memory optimization strategy
- `latent_precision`: dtype control (auto/fp16/fp32)
- `add_reference_latents`: optional conditioning stabilization
- Optional `prev_samples`: previous latents for motion continuity

Outputs:
- Updated `positive` / `negative` conditioning with motion-boosted latents
- `latent`: empty latent for sampling

---

# UPDATE VERSION 1-3-0

## 🆕 Version 1.3.0 — New MODEL IO LoRA Stack

Highlights:
- Added `LoRA Stack (Model In→Out) WAN` node: directly applies up to 4 WAN / Flow / Standard LoRAs to an incoming MODEL and outputs a patched MODEL (ideal for WAN 2.2 workflows where a single node step is preferred).
- Extended internal WAN key remapping for seamless WAN 2.2 (Flow) + WAN 2.1 cross-compatibility.
- Version bump across project files.

### New Node: LoRA Stack (Model In→Out) WAN

![Node piece no_7](assets/lora_stack_model_I_O.png)lora_stack_model_I_O.png

Use this node when you already have a base MODEL loaded (WAN 2.2, Flow, SDXL, etc.) and want a single pass application of multiple LoRAs without an intermediate stack/output hand-off. It mirrors the behavior of the classic stack + apply pair but merges them for simpler graphs (especially animation or chained sampler pipelines).

Inputs:
- `model`: base diffusion MODEL.
- `lora1..lora4` + `strength1..strength4` (skips if "no" or strength == 0.0)
- `model_type`: choose `flow`, `wan2x`, or `standard` to control remapping logic.
- Optional `lora` (LORA) input: allows concatenating a previously built stack from `IAMCCS_WanLoRAStack` for more than 4 total LoRAs.

Output:
- Patched `MODEL` ready for samplers / video pipelines.

Recommended Use (WAN 2.2 workflows):
1. Load base WAN 2.2 / LightX2V model.
2. Add `LoRA Stack (Model In→Out) WAN` and select up to 4 LoRAs.
3. (Optional) Chain a classic `IAMCCS_WanLoRAStack` into the optional `lora` input if you need >4.
4. Connect output to KSampler / Animate nodes.

Why this node: Eliminates one extra node hop, reduces graph complexity and clarifies model lineage in large animation workflows.

## Previous Versions

### Version 1.2.3 — New input lora - add another StackLoraModel (concatenate) + Extended Wan 2.1 Compatibility

### Version 1.2.1

# UPDATE VERSION 1-2-1

## 🆕 Version 1.2.1 — Extended Wan 2.1 Compatibility

The **WAN-style remap** node now supports **LightX2V 2.1 LoRA models**.  
This version extends overall compatibility to all LoRA types — even those without dedicated weight tensors (these will simply display a non-critical “missing optional weights” message).  

This ensures smoother cross-compatibility between LightX2V 2.1 / 2.2 and any WAN-based or character LoRA setup.

See full changelog → [CHANGELOG.md](./CHANGELOG.md)

# Overview

The IAMCCS-nodes package introduces a fix for a key limitation in native WANAnimate workflows:
when users run animation pipelines without the WanVideoWrapper, LoRA models fail to load correctly — most weights are ignored, and the visual consistency breaks.

This package contains two complementary nodes that work together to fix this problem and restore full LoRA functionality while keeping the workflow lightweight and modular.

The IAMCCS Native LoRA System introduces an optimized way to handle multiple LoRAs inside native ComfyUI workflows.
It is composed of two interconnected nodes designed to work seamlessly together.

## 1. LoRA Stack (WAN-style remap)

This node lets you combine several LoRA models—especially those made for WAN2.x / Animate / Flow architectures—into one unified output.

![[Node piece](assets/lora stack.png)](https://github.com/IAMCCS/IAMCCS-nodes/blob/main/assets/lora%20stack.png)

Each LoRA slot includes:

an independent strength control,

automatic WAN-style key remapping for full compatibility,

and support for .safetensors files across any model type (flow, wan, sdxl, etc).

It produces a stacked LoRA bundle that merges all the active LoRAs and prepares them for efficient native injection.

## 2. Apply LoRA to MODEL (Native)

This node applies the generated LoRA stack directly to a loaded diffusion model at the Torch level, without relying on older ComfyUI Apply LoRA wrappers.

![Node piece no_2](assets/lora%20to%20model.png)

Works natively with FP16 accumulation (recommended when paired with Model Patch Torch Settings)

Maintains precision and speed

Fully compatible with WAN2.x and other Flow-type diffusion models

Output: a ready-to-run patched model for image or video generation.

This structure replaces multiple chained LoRA nodes with a single modular system, improving both stability and performance.
Ideal for WANAnimate, WANVideo, or any Flow-based cinematic model.

![Node piece no_3](assets/ensemble.png)

# LoRA Concatenation (1.2.3)

![Node piece no_4](assets/lora_concatenatel.png)lora_concatenatel.png

### LoRA Stack (WAN-style remap)
Supports the following LoRAs:  
- New Moe distill WAN 2.2 LightX2V High Model  
- New Moe distill WAN 2.2 LightX2V Low Model 
- WAN 2.2 LightX2V High Model  
- WAN 2.2 LightX2V Low Model 
- WAN Boost Realism  
- WAN 2.2 LightX2V 4-Step High  
- WAN 2.2 LightX2V 4-Step Low
- Character LoRAs   
- WAN 2.1 LightX2V Model  

# Installation

The node has now been officially accepted on ComfyUI Manager, You can install it directly from there (just search for IAMCCS).

or

You can grab it manually:

digit in your terminal:

 cd ComfyUI/custom_nodes
 git clone https://github.com/IAMCCS/IAMCCS-nodes.git

Compatibility

ComfyUI ≥ 0.3.0

Python ≥ 3.12

Torch ≥ 2.8 (CUDA 12.6 or 12.8)

Compatible with:
WAN2.1, WAN2.2, WANAnimate, WANAnimate_relight, Pulid, Flux, and multi-LoRA setups.


# Technical Insight

LoRA weights fail to load in native WANAnimate pipelines because the model initialization bypasses the internal LoRA merge functions used in the wrapper.
By separating the process into two modular nodes, IAMCCS-nodes restores full LoRA compatibility without depending on WanVideoWrapper, keeping performance high and structure clean.

Node 1 replaces the missing LoRA-loading phase.

Node 2 reintroduces dynamic LoRA reapplication and blending inside the animation graph.

This modular architecture makes LoRA management in WANAnimate flexible, transparent, and fully native.

### If my work helped you, and you’d like to say thanks — grab me a coffee ☕

<a href="https://www.buymeacoffee.com/iamccs" target="_blank">
  <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" width="200" />
</a>

