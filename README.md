🌀 IAMCCS-nodes

Author: IAMCCS (Carmine Cristallo Scalzi)

Category: ComfyUI Custom Nodes
Main Feature: Fix for LoRA loading in native WANAnimate workflows
Version: 1.0.0

💡 Overview

The IAMCCS-nodes package introduces a fix for a key limitation in native WANAnimate workflows:
when users run animation pipelines without the WanVideoWrapper, LoRA models fail to load correctly — most weights are ignored, and the visual consistency breaks.

This package contains two complementary nodes that work together to fix this problem and restore full LoRA functionality while keeping the workflow lightweight and modular.

🧩 Node Structure
1. IAMCCS_WanModelWithLora

This node replaces the standard model loader for WANAnimate.

Function:

Loads the selected WAN model (e.g. WAN2.1, WAN2.2, WANAnimate_relight).

Checks if one or more LoRA files are attached.

Merges LoRA weights directly into the model layers before the animation process starts.
It explicitly maps keys like blocks.*.lora_A.weight and blocks.*.lora_B.weight, which are normally ignored in the native WAN loader.

Result:
The base WAN model becomes LoRA-aware even when used in pure animation workflows, without any wrapper.

2. IAMCCS_WanApplyLora

This secondary node provides manual control over LoRA injection.

Function:

Reads the WAN model from the first node and applies additional LoRA weights or strength adjustments at runtime.

Supports stacking multiple LoRA models with custom strength and priority.

Can be chained between different WANAnimate processing stages, enabling fine-tuned control.

Result:
The user can dynamically mix and reapply LoRA sets, allowing for artistic control over motion, lighting, or realism during animation generation.

⚙️ How It Works (Internally)

Both nodes share a modified implementation of the load_lora_weights() and merge_lora_keys() logic used inside WanVideoWrapper.

Instead of requiring the wrapper’s internal bridge class, these nodes operate directly on the base model’s state dict.

Missing keys or mismatched LoRA layers are gracefully skipped, ensuring full compatibility and stability.

The resulting model can run natively in WANAnimate pipelines, with all LoRAs correctly injected and active.

🧱 Installation

Install from ComfyUI Manager:

IAMCCS-nodes


or manually:

cd ComfyUI/custom_nodes
git clone https://github.com/IAMCCS/IAMCCS-nodes.git

🧪 Compatibility

ComfyUI ≥ 0.3.0

Python ≥ 3.12

Torch ≥ 2.8 (CUDA 12.6 or 12.8)

Compatible with:
WAN2.1, WAN2.2, WANAnimate, WANAnimate_relight, Pulid, Flux, and multi-LoRA setups.

🧠 Technical Insight

LoRA weights fail to load in native WANAnimate pipelines because the model initialization bypasses the internal LoRA merge functions used in the wrapper.
By separating the process into two modular nodes, IAMCCS-nodes restores full LoRA compatibility without depending on WanVideoWrapper, keeping performance high and structure clean.

Node 1 replaces the missing LoRA-loading phase.

Node 2 reintroduces dynamic LoRA reapplication and blending inside the animation graph.

This modular architecture makes LoRA management in WANAnimate flexible, transparent, and fully native.

If my work helped you, and you’d like to say thanks — grab me a coffee ☕

<a href="https://www.buymeacoffee.com/iamccs" target="_blank">
  <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" width="200" />
</a>
