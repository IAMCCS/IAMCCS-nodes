# IAMCCS Nodes – Low VRAM Video Tips

This doc describes the low-VRAM features added to IAMCCS nodes for LTX video workflows.

## 1) Hardware Probe + One-Click Apply

IAMCCS exposes a small backend endpoint:

- `GET /api/iamccs/hw_probe`
- Optional query params: `width`, `height`, `frames`, `fps`

The IAMCCS UI extension adds buttons to several nodes:

- **Probe HW & Apply** – reads your current GPU/RAM and (best-effort) reads the workflow context (width/height/frames/fps). It then applies recommended widget values immediately.
- **Copy HW report** – copies the full JSON report to clipboard.

Notes:
- Recommendations are heuristics. Final best values depend on the model, resolution, and clip length.

Frontend control (not rigid):
- **HW probe apply mode**
  - `overwrite`: always overwrite widgets with recommended values
  - `fill_missing`: only fills empty fields (does not clobber manual tuning)
- **Preset sync (profile → widgets)** (on `IAMCCS_HwSupporter` / `IAMCCS_HwSupporterAny`)
  - When ON: changing `profile` updates the other widgets to match the preset.
  - When OFF: you keep full manual control; profile changes won’t overwrite your values.

## 2) VAE Decode Tiled Safe (Video)

Node:
- `VAE Decode Tiled (safe, optional cleanup)` (`IAMCCS_VAEDecodeTiledSafe`)

Tips:
- For long videos, the most important VRAM control is **temporal chunking** (`temporal_size`).
- If you see CUDA OOM during decode, reduce:
  - `tile_size`
  - `temporal_size`
  - keep `overlap` and `temporal_overlap` small but non-zero

The HW probe can also recommend values for VAE decode based on:
- GPU VRAM
- width/height
- frames/fps (if detected)

## 3) Debug / Verification

Where to look:
- **ComfyUI server console**:
  - `/api/iamccs/hw_probe` logs a short line whenever the button is used.
- **Browser devtools console**:
  - the UI prints the full hw probe JSON under `[IAMCCS HW Probe]`.

If the button updates widgets but values get overwritten:
- ensure you clicked the button last (after changing profile/preset),
- or disable any profile auto-sync if you prefer manual tuning.

## 4) Recommended Workflow Pattern (Low VRAM)

Typical ordering:
- GGUF model loader
- `IAMCCS_GGUF_accelerator`
- `IAMCCS_HwSupporter` (or `IAMCCS_HwSupporterAny`)
- sampler
- VAE decode tiled safe

## 5) VAE Decode → Disk (True Low-RAM Mode)

New node:
- `VAE Decode → Disk (frames, low RAM)` (`IAMCCS_VAEDecodeToDisk`)

What it does:
- Decodes **one frame at a time** and writes frames to disk, instead of keeping the full `IMAGE` batch in RAM.
- This is the most reliable way to avoid CPU OOM on long clips when you still want full-resolution outputs.

When to use it:
- Very long videos (hundreds of frames)
- Low system RAM (or heavy multitasking)
- When `VAEDecodeTiled` still spikes CPU allocator memory

Tip:
- Keep `cleanup_between_frames=true` if you’re tight on VRAM.
- Use PNG for best quality; use JPG if disk size is a problem.

## 6) GGUF Accelerator – Safer “move_patches_now”

`IAMCCS_GGUF_accelerator` now supports:
- `move_policy`: `all_or_nothing` / `partial_small_first` / `partial_large_first`
- `leave_free_vram_mb`: how much VRAM to keep free during eager patch moves

Practical guidance:
- **8GB VRAM**: `move_policy=partial_small_first`, `leave_free_vram_mb=1500` (best chance to avoid OOM)
- **12–16GB VRAM**: `all_or_nothing`, `leave_free_vram_mb=1200`
- **24GB+ VRAM**: `all_or_nothing`, `leave_free_vram_mb=1024` (fastest)

## 7) Presets (Low / Normal / High)

These are sane starting points for LTX-style video workflows (no windowing):

### Low (8GB VRAM or low RAM)
- Sampler: `IAMCCS_SamplerAdvancedVersion1` with `disable_progress=true`, `cleanup=true`
- GGUF: `mode=auto_oom_safe`, `patch_on_device=true`, `move_patches_now=true`, `move_policy=partial_small_first`, `leave_free_vram_mb=1500`
- VAE: prefer `IAMCCS_VAEDecodeTiledSafe` with smaller `tile_size` and `temporal_size=64`
- If CPU RAM is the limiter: use `IAMCCS_VAEDecodeToDisk`

### Normal (12–16GB VRAM, 32GB RAM)
- Sampler: `disable_progress=true`, `cleanup=false`
- GGUF: `move_policy=all_or_nothing`, `leave_free_vram_mb=1200`
- VAE: `IAMCCS_VAEDecodeTiledSafe` with `tiling_mode=auto` (or manual: `tile_size≈384–512`, `temporal_size=64–96`)

### High (24GB+ VRAM, 64GB+ RAM)
- Sampler: `disable_progress=true`, `cleanup=false`
- GGUF: `all_or_nothing`, `leave_free_vram_mb=1024`
- VAE: you can often increase `tile_size` and `temporal_size=128` for faster decode

