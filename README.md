# 🌀 IAMCCS-nodes

<img src="icon.png" width="150" height="150">

## Author: IAMCCS (Carmine Cristallo Scalzi)

### Category: ComfyUI Custom Nodes
### Main Feature: Fix for LoRA loading in native WANAnimate workflows
Version: 1.0.0

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

# UPDATE VERSION 1-2-0

## 🆕 Version 1.2.0 — LightX2V Update
Extended support for WAN 2.2 LoRA models (LightX2V).

This release introduces a new node — **LightX2V (Remap)** — which supports both **WAN 2.2 high and low LoRA models** as well as **character LoRAs**.  
For detailed compatibility, check the following list of LoRAs supported by each node:

### LoRA Stack (WAN-style remap)
Supports the following LoRAs:  
- WAN 2.2 LightX2V High Model  
- WAN Boost Realism  
- WAN 2.2 LightX2V 4-Step High 
- Character LoRAs  
- WAN 2.2 LightX2V 4-Step Low  
- WAN 2.2 LightX2V Low Model  

### LoRA Stack (LightX2V Remap)
Supports the following LoRAs:  
- New MoE Distill LightX2V HIGH  
- WAN 2.2 LightX2V 4-Step  
- Realism Booster  
- WAN 2.2 LightX2V High Model  
- WAN 2.2 LightX2V 4-Step Low  
- LightX2V Seko Low  
- Character LoRAs  

### Notes
Depending on your setup, you can **experiment by pairing** the *LightX2V Remap* node with **High LoRAs**, and the *WAN-style Remap* node with **Low LoRAs** — or vice versa.  
**Experiment freely!**

Finally, a note on the new *LightX2V MoE Distill* LoRA (supported by the new **LightX2V Remap** node): according to the LightX2V authors,  
> *“This model focuses on training for high noise.”*  
So it works perfectly when combined with a different low model accepted by both modules.

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
