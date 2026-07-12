# IAMCCS Navigator - guida pratica

IAMCCS Navigator non e' un nodo di generazione. Non produce immagini, non modifica prompt e non va collegato ad altri nodi.

E' un sistema di navigazione: ogni nodo `IAMCCS Navigator` e' un segnaposto sul canvas. L'indice legge tutti questi segnaposto e ti permette di saltare da una zona importante del workflow a un'altra.

## Idea base

Invece di cercare a mano tra centinaia di nodi, metti piccoli bookmark nelle zone principali:

- Storyboard
- Prompt Builder
- Character Setup
- Sampler
- Preview
- Upscale
- Export

Poi apri l'indice e clicchi il nome della zona.

## Primo setup in 5 minuti

1. Apri ComfyUI e carica un workflow grande.
2. Vai in una zona importante del workflow, per esempio la parte Prompt.
3. Aggiungi un nodo `IAMCCS Navigator`.
4. Nel campo `bookmark_name` scrivi `Prompt Builder`.
5. Nel campo `category` scrivi `PREPRODUCTION` oppure `PROMPT`.
6. Scegli un colore.
7. Lascia il nodo vicino alla zona che deve rappresentare.
8. Ripeti la stessa cosa per altre zone: `Sampler`, `Preview`, `Upscale`, `Export`.
9. Premi `Ctrl+Alt+N` per aprire l'indice persistente.
10. Clicca un nome nell'indice: la vista del canvas salta a quel bookmark.

## Regola importante

Serve un nodo Navigator per ogni destinazione.

Se nel workflow hai un solo nodo `IAMCCS Navigator`, l'indice avra' un solo bookmark. In quel caso sembra inutile per forza: il valore nasce quando hai 5, 10, 20 segnaposti distribuiti nel workflow.

## Campi del nodo

`bookmark_name`
Nome visibile nell'indice. Esempi: `Storyboard`, `WAN Sampler`, `Final Preview`.

`category`
Gruppo mostrato nell'indice. Esempi: `PREPRODUCTION`, `GENERATION`, `POST`, `DEBUG`.

`color`
Colore del bookmark. Serve per leggere l'indice piu' velocemente.

`custom_color`
Usato solo se `color` e' impostato su `custom`.

`icon`
Testo breve opzionale davanti al nome. Puo' essere vuoto.

`note`
Promemoria di una riga. Esempi: `CFG 1`, `First frame starts here`, `Use low VRAM`.

`zoom_mode`
Scegli se il salto mantiene lo zoom attuale oppure ripristina lo zoom salvato.

`saved_zoom`
Zoom da ripristinare quando `zoom_mode` e' `restore saved zoom`.

`order`
Ordine manuale nell'indice. Se lasci `0`, l'ordine segue posizione verticale/orizzontale sul canvas.

`show_in_index`
Se spento, il bookmark resta sul canvas ma non compare nell'indice.

## Pulsanti sul nodo

`Open Index`
Apre l'indice persistente.

`Help`
Apre la guida rapida in overlay.

`Capture Zoom`
Salva lo zoom corrente nel campo `saved_zoom`.

`Test Jump Here`
Testa il salto verso quel bookmark.

## Hotkey

`Ctrl+Alt+N`
Apre o chiude l'indice persistente.

`Ctrl+Space`
Apre la ricerca rapida tipo command palette.

`Ctrl+Shift+M`
Salva un punto di ritorno temporaneo.

`Alt+Backspace`
Torna all'ultimo salto Navigator, oppure al punto temporaneo salvato.

## Indice persistente

L'indice e' il telecomando del workflow.

Pulsanti:

- `Find`: apre la ricerca rapida.
- `Back`: torna alla posizione precedente.
- `Mode`: cambia forma dell'indice.
- `Pin`: rende l'indice persistente dopo il reload del browser.
- `?`: apre la guida rapida.
- `-`: collassa il pannello.
- `x`: chiude il pannello.

## Modalita' dell'indice

`float`
Finestra flottante trascinabile, stile piccola utility.

`bottom`
Striscia alla base del workspace. E' la modalita' piu' vicina all'idea di barra-app modulare.

`rail`
Barra compatta alla base, molto leggera.

Premi `Mode` per ciclare tra queste modalita'.

## Workflow consigliato

Per un workflow grande, usa 6-12 bookmark:

1. `Start / Inputs`
2. `Prompt Builder`
3. `Character Setup`
4. `Reference Images`
5. `Sampler`
6. `Preview`
7. `Upscale`
8. `Export`
9. `Debug`

Per workflow enormi, usa categorie:

- `PRE`
- `GEN`
- `POST`
- `DEBUG`

## Cosa NON fa in questa versione

- Non crea thumbnail.
- Non apre ancora UI complesse di altri nodi.
- Non crea una minimappa.
- Non riordina bookmark con drag and drop.
- Non salva impostazioni nel cloud.

La V1 serve a rendere navigabile il workflow. La direzione successiva e' far diventare la striscia persistente una barra modulare che puo' richiamare UI importanti dei nodi.

## Troubleshooting

Se il nodo non compare:

1. Riavvia ComfyUI.
2. Fai hard refresh del browser.
3. Cerca `IAMCCS Navigator`.
4. Verifica che il file esista in `ComfyUI/custom_nodes/IAMCCS-nodes/iamccs_navigator.py`.

Se l'indice e' vuoto:

1. Assicurati di avere aggiunto almeno un nodo `IAMCCS Navigator`.
2. Controlla che `show_in_index` sia attivo.
3. Chiudi e riapri l'indice con `Ctrl+Alt+N`.

Se il salto va nel posto sbagliato:

1. Sposta fisicamente il nodo Navigator nel punto giusto del canvas.
2. Il bookmark punta alla posizione del nodo stesso.
3. Usa `Test Jump Here` per controllare.

Se lo zoom non e' quello desiderato:

1. Imposta `zoom_mode` su `restore saved zoom`.
2. Posiziona il canvas allo zoom desiderato.
3. Premi `Capture Zoom` sul nodo Navigator.

## Sintesi

Navigator funziona bene quando lo tratti come un indice del workflow:

un nodo = una destinazione.

molti nodi = workflow navigabile come un'app.
