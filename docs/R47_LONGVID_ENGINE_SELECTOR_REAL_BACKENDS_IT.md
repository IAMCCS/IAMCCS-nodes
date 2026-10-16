# IAMCCS H3 R47 — LONG KEYFRAME: SELETTORE MOTORI EFFETTIVI

**Base:** R46 TimelineDirector Restore, senza rimuovere i mode Universal precedenti.
**Pannello:** IAMCCS Settings PRO → ENGINES → LONG KEYFRAME. **Task pubblico:** `longvid_keyframe_continuous`.

## Motori realmente distinti

| Dropdown | Runtime effettivo | Contesto e join | Default/stato |
| --- | --- | --- | --- |
| `pianosequenza_1_native_tail` | `longvid_guides` + TimelineDirector + AudioCon | Overlap consegnato dal segmento entrante, bridge RGB 1f rimosso, LATENT FREE; selezione 22/39/56f | **Default**, 22f smoke storico |
| `extended_av_continuous` | **`fl2va_extended_av`** v4 masked EXTEND | Pinned RAW AV context, **trim tecnico**: NON incoming-owned overlap; SAFE 192f/39f | Profilo SAFE 8–12GB predefinito, sperimentale |
| `pianosequenza_evolving` | `longvid_guides` + TimelineDirector/AudioCon + H3 Evolving timed-conditioning | Overlap native 22/39/56f; local beat prompts con tempi di Shotboard e state carry | Sperimentale; richiede Extended Prompt blocks con tempo definito |
| `pianosequenza_2_drift` | `longvid_guides` + TimelineDirector/AudioCon, terminal policy `pianosequenza_drift` | Overlap native 22/39/56f | Sperimentale |
| `pianosequenza_3_phase` | `longvid_guides` + TimelineDirector/AudioCon, terminal policy `pianosequenza_phase` | Overlap native 22/39/56f | Sperimentale |

**IMPORTANTE:** DRIFT e PHASE condividono la geometria e il runtime di continuità con Pianosequenza 1: non sono un terzo/quarto sistema di trasporto AV. Cambiano la strategia di trattamento terminale implementata in `iamccs_minimax_h3_terminal_strategies.py`.

### Semantica dei frame

- **Native Tail**: 22/39/56f sono i contesti AV H3 compatibili. Il planner non forza più sempre 22f; mostra il valore scelto dal menu. Per la prova storica a 30s il master resta **720f**. I campioni tecnici possono contenere padding e il bridge RGB; questi NON aggiungono tempo editoriale.
- **Extended AV**: usa una griglia AV diversa (39+51k), non il layout Native Tail. I preset disponibili sono SAFE (192f/39f), BALANCED (243f/39f), QUALITY (294f/90f), MAX (345f/141f), o CUSTOM; la parte pinned viene tagliata dalla consegna. Non passare un overlap Native Tail al join Extended.
- **Evolving Native**: la direzione temporizzata viene iniettata per chunk dal vero resolver `iamccs_minimax_h3_conditioning_schedule.py`. Se manca un programma valido di beat temporizzati, **blocca Queue con un errore leggibile** invece di generare indistinguibilmente da Pianosequenza 1.

## Trasporto dalla UI al sampler

1. Il campo `longvid_continuous_engine` è già **append-only** nella Settings PRO R46: R47 estende le opzioni, NON inserisce widget in mezzo a quelli dei workflow salvati.
2. La scelta viene serializzata nei Settings CineLinX via `IAMCCS_ShotboardH3SettingsPro.export` e letta con `_h3_settings_from_cine_linx` nel vero `IAMCCS_MiniMaxH3ShotPlanner.plan`.
3. La scelta Native attiva `timeline_director_native=True`, AudioCon e `annotate_restored_plan`; la scelta Extended compila il **vero** task backend `fl2va_extended_av` e il suo contratto di prefix trim. Il resto dell'Universal continua a leggere `plan.task_mode` e usa i rispettivi runtime.
4. Evolving chiama esplicitamente il resolver dei beat Shotboard. DRIFT e PHASE inviano il rispettivo `terminal_endpoint_mode` al sampler H3 nativo.
5. Il campo `flf_continuity_tail_frames` si applica solo alle varianti Native; gli `extended_av_*` compaiono solo selezionando Extended nel mode LONG KEYFRAME. I pulsanti di preset resettano la modalità di audio su H3 generato nativo e disattivano l'upscale inline per il primo smoke.

## Come eseguire gli smoke

Prima di usare R47, fare backup della propria cartella IAMCCS-nodes. Dopo aver sostituito la cartella, riavviare ComfyUI e aggiornare il frontend con Ctrl+F5. Caricare l'Universal R45 già incluso; nella Settings PRO selezionare LONG KEYFRAME CONTINUOUS.

Per il **primo render CUDA** selezionare Pianosequenza 1 + **22f** + Native Audio + 960×544 + 24 fps + 30s + **Native Delivery**. Verificare frame finali, due junction e audio nativo. Poi ripetere con 39f, e infine Extended SAFE e Evolving con almeno due Extended Prompt blocks. Non usare l'upscale generativo per concludere che la continuità funziona: valutare l'upscale sul master intero come fase separata.

### Validazione automatica e limiti

- `python -m unittest discover -s tests -p 'test_h3_r47*.py' -q`: test selettore/semantica CPU.
- `python tests/manual/r47_engine_route_cinelinx_cpu_smoke.py`: vero planner con cinque Settings CineLinX diversi e backend selezionato, ma stubs per ComfyUI e nessuna inferenza GPU.
- `python -m unittest discover -s tests -p 'test_h3_longvid*.py' -q`: regressioni LongVid precedenti.
- `python -m unittest discover -s tests -p 'test_h3_r46*.py' -q`: regressioni TimelineDirector R46.
- `python -m unittest discover -s tests -p 'test_h3_extended_av.py' -q`: test dell'engine AV esteso, maschere e sidecar (CPU).

**Nessuno smoke CUDA H3 reale è stato eseguito qui.** R47 è una build di prova: test CPU e route sono necessari ma insufficienti per certificare qualità movimento, waveform continuity, VRAM o upscale.
