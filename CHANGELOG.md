## 🆕 Version 1.3.0 — MODEL In→Out LoRA Stack & Qwen Loader Docs

Date: 2025-11-19

Changes:
- Added new node `IAMCCS_WanLoRAStackModelIO` ("LoRA Stack (Model In→Out) WAN") for direct multi-LoRA application to an incoming MODEL (WAN 2.2 / Flow / Standard).
- Preserves WAN key remap + optional chaining via existing `IAMCCS_WanLoRAStack` (use optional `lora` input to extend beyond 4 slots).
- Updated `README.md` with explicit low VRAM instructions for Qwen Image LoRA loader and dependency checklist (`ComfyUI-nunchaku`, `ComfyUI-QwenImageLoraLoader`).
- Bumped versions (`version.json`, `pyproject.toml`) to 1.3.0.
- Neutralized deprecated Save&Load DragCrop code (frontend/backend) — removed from active registration.

Notes:
- Existing workflows using the older two-node stack + apply pattern continue to work unchanged.
- Use `IAMCCS_WanLoRAStackModelIO` to simplify WAN 2.2 graphs or reduce node count before samplers.

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
