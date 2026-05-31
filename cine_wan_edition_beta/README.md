# IAMCCS Shotboard V3 WAN Edition BETA

This folder is intentionally isolated from the stable LTX Shotboard V3 code.

Commit safety:
- The module is optional.
- The root `__init__.py` loader is guarded with `try/except`.
- If this folder is excluded from a commit, IAMCCS-nodes still starts.
- If the folder is included without workflow usage, existing LTX workflows are unchanged.

Nodes:
- `IAMCCS_CineShotboardPlannerV3WANEdition_BETA`
- `IAMCCS_WanPromptRelayBridge_BETA`

Purpose:
- Compile Shotboard-style WAN FLF/SVI metadata.
- Export true PromptRelay inputs for WAN.
- Patch the connected WAN model only inside `IAMCCS_WanPromptRelayBridge_BETA`.

