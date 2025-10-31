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
