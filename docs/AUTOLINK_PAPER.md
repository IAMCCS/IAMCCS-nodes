# IAMCCS AutoLink — Paper & Usage Instructions (EN/IT)

## English

### 1) What is AutoLink?
AutoLink is a **Set/Get** workflow tool designed to keep ComfyUI graphs clean and maintainable.

Instead of long cables across the canvas, AutoLink lets you:
- Convert direct connections into **Set** (source) + **Get** (destination) pairs
- Restore the original direct connections when needed
- Apply repeatable filters (groups/blacklist), layout rules, and colors

Everything is controlled by a dedicated “tool” node that operates on the canvas.

### 2) Components
AutoLink is made of four logical elements:

1. **AutoLink Converter**
  - Buttons to convert/restore links.
2. **AutoLink Arguments**
  - Central configuration: group filters, alignment/layout, packing/anti-overlap, colors, blacklist.
3. **AutoLink Set**
  - Created near the source node: captures an output and exposes it under a key.
4. **AutoLink Get**
  - Created near the destination node: retrieves the key and feeds the target input.

### 3) Quickstart
1. Add to the canvas:
  - **AutoLink Arguments**
  - **AutoLink Converter**
2. Connect **AutoLink Arguments** output to the Converter `arg` input.
3. Adjust options (or keep defaults).
4. Click **Convert All Links**.

To revert:
- Click **Restore Direct Links**.

### 4) v1.3.3 reliability updates (important)
AutoLink Set/Get nodes are **UI tools** and are treated as **virtual** nodes. To prevent “missing required input” prompt errors, the extension automatically:
- Materializes direct links **only during prompt serialization/queue**, then restores the AutoLink wiring
- Supports nested graphs/subgraphs
- Truncates long AutoLink titles with an ellipsis (`…`) so they stay inside the node header

---

## Italiano

### 1) Cos’è AutoLink
AutoLink è un sistema **Set/Get** pensato per rendere i workflow ComfyUI più ordinati, leggibili e facili da mantenere.

Invece di avere cavi lunghi che attraversano la canvas, AutoLink permette di:
- Convertire automaticamente collegamenti diretti in coppie **Set** (sorgente) + **Get** (destinazione)
- Ripristinare i collegamenti originali quando serve
- Gestire filtri, gruppi, layout e colori in modo ripetibile

Il tutto è controllato da un nodo “tool” che opera sulla canvas.

### 2) I nodi coinvolti
AutoLink è composto da quattro elementi logici:

1. **AutoLink Converter**
  - Contiene i pulsanti per convertire/ripristinare i collegamenti.
2. **AutoLink Arguments**
  - Contiene tutte le opzioni: filtri per gruppi, layout, packing/anti-overlap, colori, blacklist.
3. **AutoLink Set**
  - Viene creato vicino al nodo sorgente: cattura un output e lo espone con una chiave.
4. **AutoLink Get**
  - Viene creato vicino al nodo destinazione: recupera la chiave del Set e alimenta l’input.

### 3) Quickstart (workflow consigliato)
1. Aggiungi in canvas:
  - **AutoLink Arguments**
  - **AutoLink Converter**
2. Collega l’output di **AutoLink Arguments** all’input `arg` di **AutoLink Converter**.
3. Imposta le opzioni nel nodo **AutoLink Arguments** (anche lasciando i default).
4. Premi **Convert All Links** nel nodo **AutoLink Converter**.

Per tornare indietro:
- Premi **Restore Direct Links** nel Converter.

### 4) Aggiornamenti affidabilità v1.3.3 (importante)
I nodi Set/Get di AutoLink sono strumenti **lato UI** e vengono trattati come nodi **virtuali**. Per evitare errori di prompt del tipo “required input missing”, l’estensione:
- Materializza i link diretti **solo durante la queue/serializzazione del prompt**, poi ripristina il wiring AutoLink
- Supporta grafi annidati/subgraph
- Tronca i titoli AutoLink troppo lunghi con ellissi (`…`) per non farli uscire dal nodo

---

## 5) Opzioni principali (Arguments)

### 4.1 GroupExclude
- Se abilitato, **non converte** i collegamenti tra due nodi che stanno **dentro lo stesso group**.
- I collegamenti che **entrano** o **escono** dal group possono comunque essere convertiti (dipende anche da GroupInOutExclude).

Quando usarlo:
- Se un group rappresenta un “blocco logico” che vuoi tenere cablato internamente.

### 4.2 GroupInOutExclude
Gestisce i link che attraversano un confine di group:
- `None`: nessuna esclusione.
- `ExcludeEnter`: non converte i link che **entrano** in un group.
- `ExcludeExit`: non converte i link che **escono** da un group.
- `ExcludeBoth`: combina entrambe.

### 4.3 Align mode
Determina come vengono posizionati Set/Get dopo la conversione e quando fai relayout.

Opzioni principali:
- `TopToDown`, `BottomToTop`, `CenterUpDown`, `CenterDownUp`
- `AlignX_Right`, `AlignX_Left`
- `Columns_Down`, `Columns_Up`
- `Rake_Down`, `Rake_Up`
- **`Proportional`** (consigliato per layout “come i cavi”)

#### Align = Proportional (come nell’immagine)
Con `Proportional`, Set e Get vengono agganciati alla **stessa altezza (Y)** del relativo connettore (slot) del nodo:
- Set: si allinea alla Y dello **slot di output** sorgente
- Get: si allinea alla Y dello **slot di input** destinazione

In caso di collisioni, mantiene la Y e cerca spazio spostandosi orizzontalmente.

### 4.4 Packing mode
Controlla l’anti-overlap durante posizionamento e relayout:
- `AvoidAll`: evita sovrapposizioni con tutti i nodi.
- `AvoidNonAutoLink`: evita solo i nodi non-AutoLink (Set/Get possono compattarsi fra loro).

### 4.5 SeparateCol + colori
- `SeparateCol`: se attivo, permette di usare colori diversi per Set e Get.
- `AutoLinkColor`: colore base (Set).
- `AutoLinkColorGet`: colore dei Get (solo se SeparateCol è attivo).

### 4.6 ColorTitles
Cambia il colore del testo del titolo dei nodi AutoLink:
- `White`
- `Black`
- `Auto`

### 4.7 Blacklist (ID e Types)
AutoLink permette di escludere nodi dalla conversione:

- `all_nodes_sel`:
  - OFF: la blacklist lavora per **tipo** (`[TYPE] ...`)
  - ON: la blacklist lavora per **ID singolo nodo**

- `add_to_blacklist`:
  - Scegli un nodo (ID) o un tipo.

- `blacklist_mode` (solo per nodi singoli):
  - `both`: esclude link dove il nodo è sorgente o destinazione
  - `only_output`: esclude solo quando il nodo è sorgente (output)
  - `only_input`: esclude solo quando il nodo è destinazione (input)

- `EXECUTE`:
  - Applica davvero l’inserimento (o l’update della modalità) e poi pulisce i widget.

- `blacklist_view`:
  - Elenco leggibile: `id - nome nodo - (modalità)` e `[TYPE] ...`.
  - Selezionare una voce **non rimuove nulla**.

- `remove_blacklist`:
  - Rimuove la voce attualmente selezionata in `blacklist_view`.

---

## 6) Best practices
- Prima di convertire “tutto”, imposta la blacklist per escludere nodi che vuoi lasciare cablati.
- Usa `GroupExclude` per mantenere “blocchi” interni puliti.
- Usa `Proportional` quando vuoi un layout che segua visivamente l’ordine degli slot (come routing naturale dei cavi).
- Se la canvas è molto piena, prova `PackingMode = AvoidAll`.

---

## 7) Troubleshooting
- **Convert All Links non sembra fare nulla**:
  - Verifica che `AutoLink Arguments` sia collegato all’input `arg` del Converter.
  - Controlla blacklist e filtri group.
- **Nodi sovrapposti**:
  - Prova `PackingMode = AvoidAll`.
  - Cambia align mode o usa relayout cambiando `align_mode`.

---

## 8) Documentazione correlata
- AUTOLINK_README.md
- AUTOLINK_TECHNICAL_PAPER.md
