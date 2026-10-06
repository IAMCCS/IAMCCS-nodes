# IAMCCS R44.4 — Video Editor duration authority + RTX Exporter preflight

## Video Editor

- Audio hard cuts are now lane-local. A clip on A1 is not shortened because a clip starts later on A2/A3/A4/A5.
- Manifests previously shortened by the old global hard-cut rule are healed when the cut carries the IAMCCS system marker.
- Manual video/audio linking no longer pairs unrelated imports merely because they share a T01/T02 lane index.
- Generated H3 video/audio pairing uses take + clip index when available.
- The legacy `zz_iamccs_shotboard_video_editor_media_fix.js` no longer replaces the newer integrated server-media importer (2026-09-18+). This restores authoritative server probing for manual video duration and embedded audio.

## IAMCCS Exporter PRO / NVIDIA RTX VFX

- Added non-throwing RTX runtime preflight.
- Exporter UI reports whether the NVIDIA RTX VFX runtime is actually available instead of claiming RTX is active as soon as the checkbox is enabled.
- If RTX is requested but `nvvfx` is missing, the exporter logs an explicit warning and completes the export with original frames instead of aborting the entire delivery.
- Metadata/report distinguish `rtx_requested` from `rtx_effective`.
- `apply_rtx_vfx()` now imports/preflights the optional runtime before unloading Comfy models.

Actual RTX VFX processing still requires the NVIDIA/Deno runtime. The fallback is intentionally not labelled as RTX output.
