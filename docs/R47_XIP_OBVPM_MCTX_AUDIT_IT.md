# IAMCCS R47 · audit xip new / OBVPM MCtx

Data: 2026-10-08. Verifica eseguita sul codice installato, senza dichiarare un render GPU riuscito.

## Esito

Il report fornito dall'utente è sostanzialmente corretto. `comfyui-obvpm-timeline` ottiene la continuità soprattutto conservando i latenti AV originali in `.mctx.safetensors`, reinserendo finestre precise nella generazione successiva e proteggendole con denoise mask native. Non è emersa una LoRA fotografica autonoma responsabile dei raccordi.

Le finestre guide seguono la scala 5/22/39/56 frame. Le finestre `masked` devono invece rispettare la griglia AV condivisa, a partire da 39 frame e poi 90/141. `take_from=at_frame` consente di tagliare una sorgente su un punto valido; `place=at_frame`, cioè un pin interno arbitrario nel nuovo clip, è ancora esplicitamente non implementato nel pacchetto verificato.

## Confronto con xip new e R46

`xip new(1).rar` contiene una fotografia completa più vecchia di IAMCCS-nodes, non un singolo nodo. Il suo percorso Pianosequenza dinamico è stato superato dall'attuale R46: R46 conserva il sampled VIDEO+AUDIO latent tail, AudioCon, endpoint editoriale, maschere per stream e test di geometria 30 s. Sostituire la cartella corrente con xip reintrodurrebbe codice e UI precedenti.

La differenza principale rimane questa:

- Pianosequenza 1/R46: trasporta una coda AV nativa su una timeline autoriale con guide temporizzate.
- Extended AV: continuità masked/pinned a finestra tecnica, ma senza sidecar editoriale OBVPM.
- OBVPM MCtx: conserva ogni take e la sua genealogia come asset persistente, permette Extend/Prepend/Bridge, trim e sostituzione take, quindi privilegia la continuità da movimento già riuscito.

Questo spiega perché i migliori LongVid OBVPM possono risultare più fluidi: non obbligano il modello a “raggiungere” molte fotografie forti nella stessa generazione; continuano moto già riuscito e consentono una selezione editoriale iterativa dei take.

## Integrazione scelta

È stato aggiunto un solo nodo append-only: `IAMCCS H3 LongVid · MCtx Route`.

Il nodo espone tre operazioni leggibili:

- `extend`: previous tail → before, normalmente `masked`;
- `prepend`: next head → after, consigliato `both`;
- `bridge`: previous tail → before + next head → after, consigliato `both`.

L'adapter produce soltanto `OBVPM_H3_PINSPECS` chiamando il nodo pubblico `H3MCtxPinSpec` installato. Non copia slicing, maschere, sampler o trim: questi restano di proprietà OBVPM. Collegare l'output a `H3 MCtx Apply Pins`, quindi usare il suo output `pins` nel trim/save OBVPM.

## Limite di compatibilità importante

OBVPM richiede il `PackedLayout` e `MiniMaxH3.extra_conds` nativi aggiornati. IAMCCS LongerVid storico può installare hook globali su quegli stessi punti quando viene eseguito. I due proprietari non devono essere attivi insieme nella stessa sessione: per lo smoke MCtx riavviare ComfyUI e provare il ramo MCtx in una sessione pulita prima di eseguire un ramo IAMCCS LongerVid che installa gli hook. L'adapter non nasconde questa regola e non tenta di sovrascrivere patch di terzi.

## Smoke A/B richiesto

Usare la stessa clip sorgente, risoluzione, prompt, seed e durata:

1. Pianosequenza 1 · Native Tail, 22f.
2. Extended AV · masked, 39f.
3. OBVPM MCtx · Extend masked, 39f.
4. OBVPM MCtx · Extend both, 39f.

Confrontare: identità, variazione luce/colore, continuità movimento, audio al junction, freeze vicino alla fine e VRAM. Salvare ogni take; non valutare soltanto il master finale. Solo dopo un A/B GPU riuscito il ramo MCtx deve entrare nel workflow Universal o nella UI PAN.

## Conclusione

OBVPM è un riferimento valido e implementabile, ma non sostituisce automaticamente la regia di sette fotografie intermedie. Per quello servirà un motore successivo che combini MCtx ai bordi con conditioning fotografico graduato e non conflittuale all'interno dell'intervallo. Prima si valida la continuità 1:1; poi si estende il controllo autoriale.
