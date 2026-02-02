# IAMCCS_HwSupporter

Node pack per ComfyUI che applica in modo “auto / preset / manual” alcune impostazioni anti-OOM e speed knobs, con un report JSON in output.

## Nodi

### 1) HW Supporter (auto VRAM/attention/torch knobs)
- File: `iamccs_hw_supporter.py` (`IAMCCS_HwSupporter`)
- Input principale: `model` (MODEL)
- Output: `model`, `clip` (passthrough), `vae` (passthrough), `report_json`

Posizionamento consigliato:
- Mettilo subito dopo il nodo che crea/carica il `MODEL` (e prima di LoRA/sampling).
- Se vuoi anche `vae_tiling_suggestion` nel report, collega anche `vae` in input (opzionale).

Cosa fa:
- VRAM reserve: imposta `comfy.model_management.EXTRA_RESERVED_VRAM` (simile al nodo reservedvram).
- SageAttention: se installato, patcha l’attenzione del modello via `model.model_options["transformer_options"]["optimized_attention_override"]`.
- PyTorch knobs: `torch.backends.cuda.matmul.allow_fp16_accumulation`, TF32.
- (Opzionale) `torch.compile`: prova a compilare `model.model.diffusion_model` (attenzione: può aumentare picco VRAM al primo run).
- Nel `report_json` include anche `vae_tiling_suggestion` (tile_size/overlap consigliati) basati su VRAM rilevata.
- Se `console_log=true` stampa una riga riassuntiva nel terminale (e i warning).

### 2) VRAM Cleanup (unload + empty cache)
- File: `iamccs_hw_supporter.py` (`IAMCCS_VRAMCleanup`)
- Utility node per forzare `unload_all_models()` + `soft_empty_cache()` (più `gc.collect()` e `torch.cuda.empty_cache()`).

### 3) VAE Decode Tiled (safe, optional cleanup)
- File: `iamccs_hw_supporter.py` (`IAMCCS_VAEDecodeTiledSafe`)
- Wrapper di `vae.decode_tiled(...)` con tile/overlap e supporto chunk temporale (video VAE).
- Opzione `cleanup_before_decode` per ridurre i picchi VRAM quando il decode arriva dopo il sampling.
- Nuova opzione `tiling_mode`:
	- `auto`: sceglie automaticamente `tile_size` e `overlap` in base alla VRAM rilevata (conservativo, anti-OOM)
	- `manual`: usa i valori inseriti a mano

## Preset consigliati (12GB VRAM / 32GB RAM)
Impostazione pratica (conservativa):
- `profile`: `12GB_VRAM_32GB_RAM`
- `reserved_vram_gb`: 1.25 (oppure 1.5 se spesso in OOM)
- `sage_attention`: `auto` (se disponibile)
- `torch_compile_mode`: `off` (in genere più stabile su low-vram/offload)
- `fp16_accumulation`: `auto`
- `tf32`: `auto`

## Note importanti
- `PYTORCH_CUDA_ALLOC_CONF`: in genere va impostato **prima** di avviare ComfyUI per influenzare l’allocator. Il nodo riporta un warning/nota, ma non “garantisce” di cambiare l’allocator a runtime.
- `torch.compile`: in molti setup low-vram/offload può dare instabilità o aumentare il picco VRAM (soprattutto al primo run). Usalo solo se hai margine.

## Suggerimento pratico (pipeline 12GB)
- Sampling → (opzionale) `VRAM Cleanup` → `VAE Decode Tiled (safe)` con `tiling_mode=auto` e `cleanup_before_decode=true` se sei al limite.

## Debug
Se qualcosa non funziona:
- guarda `report_json` (warnings + applied).
- prova a disabilitare SageAttention o `torch.compile`.
- inserisci `VRAM Cleanup` tra fasi pesanti (es. prima del VAE decode).

## Crash Triton su Windows (libtriton.pyd / 0x80000003)
Se vedi un hard-crash tipo `libtriton.pyd` + `Exception Code: 0x80000003`, non è un OOM: di solito è un crash interno Triton/MLIR.

Mitigazioni consigliate:
- In `IAMCCS_HwSupporter`: `torch_compile_mode = off`.
- In `IAMCCS_HwSupporter`: evita modalità SageAttention basate su Triton.
	- usa `sageattn_qk_int8_pv_fp16_cuda` (consigliato) oppure `disabled`.
- Riavvia ComfyUI dopo i cambi (i crash Triton non sono “recoverable”).

---

# HW Probe & Apply (English)

IAMCCS provides a **Hardware Probe** endpoint and UI buttons to automatically recommend and apply settings.

## What you get

- Backend endpoint: `GET /api/iamccs/hw_probe`
- Optional query params (best-effort context): `width`, `height`, `frames`, `fps`
- Frontend buttons (added to several IAMCCS nodes):
	- **Probe HW & Apply**: updates widgets immediately (visible in real-time)
	- **Copy HW report**: copies the full JSON report

## Nodes supported by the button

- `IAMCCS_HwSupporter`
- `IAMCCS_HwSupporterAny`
- `IAMCCS_SamplerCustomAdvancedWindowed`
- `IAMCCS_VAEDecodeTiledSafe`

## Tips

- The hw probe uses heuristics; best values still depend on your resolution and clip length.
- For long videos, the most important VRAM lever is **temporal chunking** (`temporal_size`).

### torch.compile on Windows

- Default is `torch_compile_mode=off` (safest).
- If you set `torch_compile_mode=auto`, the node will attempt compilation (internally uses a conservative mode, typically `reduce-overhead`).
- On Windows, torch.compile may still hard-crash depending on Torch/Inductor/driver; if you get hard crashes, switch back to `off`.

See `LOW_VRAM_VIDEO_TIPS.md` for practical guidance.
