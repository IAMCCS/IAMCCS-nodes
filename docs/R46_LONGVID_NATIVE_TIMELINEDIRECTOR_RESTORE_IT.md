# IAMCCS-nodes R46 · Long Keyframe Continuous / Pianosequenza native restore
Data: 2026-10-08. Source: `IAMCCS-nodes_latest.zip` and September historical `xip new(1).rar`.

## Problema che questa build corregge
Il mode pubblico `longvid_keyframe_continuous` veniva compilato sul backend R45
`iamccs_fl2va_continuous_av_v1` come successione di clip A→B/B→C. Questo
reintroduce l'approccio a destinazione FLF e può bloccare il movimento prima
dell'endpoint editoriale. La restaurazione usa il percorso legacy Pianosequenza /
TimelineDirector nativo, non modifica il motore FL2VA Extended AV separato.

## Come utilizzare il nuovo pannello
1. Apri il workflow Universal R45 integrato che usavi (R45 è conservato nello ZIP).
2. Apri `IAMCCS Shotboard H3 Settings` e, nel pannello Settings PRO,
   vai a **ENGINES → LONG KEYFRAME**.
3. Seleziona `LONG KEYFRAME · CONTINUITY ENGINE`:
   - **pianosequenza_1_native_tail** (predefinito, raccomandato per il primo smoke):
     native sampled VIDEO+AUDIO latent tail, 22f overlap, AudioCon, endpoint
     `LATENT FREE` (`latent_free_closure`), senza reanchor terminale rigido.
   - **pianosequenza_2_drift**: stesso trasporto nativo, terminal policy DRIFT.
     **SPERIMENTALE**.
   - **pianosequenza_3_phase**: stesso trasporto nativo, terminal policy PHASE.
     **SPERIMENTALE**.
4. Seleziona H3 Native Generated Audio, Native Delivery e, per il primo test,
   **960×544**, **24 fps**, **30 secondi**, guides su Shotboard.
5. Non attivare inline upscale per questo mode. Il modello di upscale è
   **whole-sequence dopo la consegna del master**, come fase distinta.
6. Avvia dal primo chunk: la cache latente e i segmenti successivi richiedono
   lo stesso render ID e la coda sequenziale. Non eseguire chunk 2 isolato.

## Contratto della timeline
- root **362f**.
- technical continuation 2 **226f** = 22 overlap + 1 RGB bridge + 187 new + 16 padding.
- technical continuation 3 **209f** = 22 overlap + 1 RGB bridge + 171 new + 15 padding.
- R38B elimina soltanto il bridge RGB e il padding: segmenti **362 / 209 / 193**.
- Assembly `incoming owns overlap`: 340 + 187 + 193 = **720f a 24 fps = 30.00s**.
- Checkpoint v4: cache native AV tail estratta all'**editorial endpoint** del segmento
  (chunk 2 = 209f) e non alla fine tecnica (226f).
- AudioCon salva e trasporta un vero audio native latent tail, con video taper e
  audio protected prefix + half-cosine release. Il master PCM viene muxato
  tramite ffmpeg senza dissolve video; il trasporto AudioCon **non equivale**
  a fare crossfade/concatenazione audio al posto dei latenti.
- Le guide autoriali rimangono eventi temporizzati su un orologio globale, non
  costituiscono clip FLF di destinazione autonome.

## Compatibilità
- Il mode pubblico è compilato nel planner nativo `longvid_guides` e nel
  router `r42` con `requested_task_mode=longvid_keyframe_continuous` conservato.
- Lo Universal integrato R45, i checkpoint, Scout bypass, Video Editor opzionale,
  gli altri engine FL2VA e le altre modalità restano nei file originali.
- Il nuovo widget di Settings PRO è **append-only nella sezione optional**:
  i widget salvati storici non vengono spostati.
- La configurazione salvata contiene solo la scelta del motore; le altre
  opzioni H3 esistenti non vengono sovrascritte fuori da questo mode.
- I preset 2 DRIFT / 3 PHASE **non** sono dichiarati visivamente convalidati.

## Test eseguiti su questa build (senza GPU)
- 25 test legacy LongVid passati;
- 8 test R46 inclusi CPU (planner 720f, 3 motori, endpoint, compatibilità,
  guida sul boundary tecnico, AudioCon con video/audio tail e half-cosine);
- compilazione Python (`compileall`) e parsing di entrambe le UI JS (`node --check`).
- NON è stato eseguito un render MiniMax H3 completo con checkpoint reali,
  ComfyUI, CUDA, VAE e FFmpeg. **Il successo visivo e l'assenza di freeze
  richiedono lo smoke sulla macchina GPU dell'utente**.

## Smoke GPU raccomandato
1. 30s / 24fps / 960×544 / Pianosequenza 1 / H3 Native Audio / Native Delivery.
2. Nel log: `task_mode=longvid_guides`, `requested_task_mode=longvid_keyframe_continuous`,
   `backend_variant=iamccs_native_sampled_latent_tail_timeline_director`.
3. Geometria H3 raw: 362/226/209. Segmenti film: 362/209/193, master 720f.
4. Cache tail: `audiocon=True`, `requested_endpoint_frames` 362 e 209 per
   i primi due chunk, e `audio_release=half_cosine` nelle continuazioni.
5. Verifica movimento sui raccordi ~15.0s e ~22.8s, audio, identità del
   soggetto, assenza di un freeze prima della guida finale.
6. Solo dopo validazione, esporta/upscala il master nativo intero.

## Installazione
1. Backup di `ComfyUI/custom_nodes/IAMCCS-nodes`.
2. Sostituisci **l'intera** cartella con `IAMCCS-nodes` dello ZIP;
   evita merge parziali che lasciano moduli incompatibili.
3. Riavvia ComfyUI. Aggiorna forzatamente la pagina del browser (Ctrl+F5)
   per ricaricare i pannelli JS aggiornati.
4. Apri il workflow Universal integrato già noto: non serve sostituire i
   workflow salvati per usare il nuovo dropdown in Settings PRO.

Non eseguire gli script storici `apply_*.py` sul latest: questa release incorpora
l'innesto controllato preservando il runtime moderno AudioCon.
