# IAMCCS R44.7

## Mixer / Video Editor

The Shotboard Video Editor remains an audio slave when driven by IAMCCS AudioBoard Mixer. R44.7 adds per-track A1–A5 and master meter data to the slave snapshot. Meter values are sampled from the editor audio peak data at the live playhead and include track fader/gain and master gain.

Continuous volume/gain/master-gain edits no longer call a forced audio seek. HTML audio preview gain is ramped over successive animation frames to avoid zipper/click artefacts while dragging faders. Timeline scrubbing and explicit seeks keep the existing seek behaviour.

## Face Refinement presets

Quality, Face Size and Definition are independent contextual preset families. One preset can be active in each family at the same time. Their value domains are intentionally separated so changing one family does not silently invalidate another active family. A reset action clears all three highlights and restores the current IAMCCS Face Refinement technical defaults while preserving enable/mode/max-face policy.

## RTX Exporter

The RTX runtime panel keeps the same preflight/probe behaviour but uses compact action labels (`CHECK RTX`, `INSTALL GUIDE`) with full English tooltips so the controls remain legible at the normal node width.
