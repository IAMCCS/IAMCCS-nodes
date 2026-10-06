# IAMCCS R44.5 — RTX preflight UI + Face Refine preset UX

## RTX Exporter PRO
- IAMCCS never modifies the Deno repository.
- Runtime preflight now reports READY / NOT INSTALLED, runtime source, Python executable, nvvfx module path and VideoSuperRes availability.
- Added CHECK RTX INSTALLATION and OPEN OFFICIAL INSTALL GUIDE controls.
- Deno prepared runtime markers remain discovery-only; missing RTX falls back to native export.

## Settings PRO / Face Refinement
- Face preset groups use a three-column aligned layout with responsive single-column fallback.
- Every preset has English production guidance and a hover tooltip.
- Face Size presets no longer change Face Detailer Profile, so explicit Single / Multi Face strategy is preserved.
- Face canvas/window validation now runs after Settings PRO overrides are resolved.

## Video Editor / AudioBoard Mixer
Recommended editorial control route:
EditorTakeRoute cine_linx -> IAMCCS AudioBoard Mixer -> IAMCCS BusOut -> Video Editor cine_editor_linx.
The main EditorTakeRoute cine_linx connection to the Video Editor remains in place for take/shot truth.
