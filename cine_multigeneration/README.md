# IAMCCS Cine Multigeneration

Optional, removable module for Shotboard Plus + AudioBoard multigeneration.

This module is separate from the WAN edition work. It adds a small sequence layer that reads
`IAMCCS_BusOut` audio metadata, splits it into take windows, and writes a
generation index plus concat plan back into `cine_linx`.

Current nodes:

- `IAMCCS_MultiTimelineBridge`
  - Reads BusOut master/stem JSON or BusOut resources inside `cine_linx`.
  - Builds chunked takes from 10s, 15s, 20s, 25s, or custom durations.
  - Outputs an active take with local audio start at frame 0.
  - Keeps the original BusOut master/stems available for final concat/mix.
- `IAMCCS_MultiTimelineTakePicker`
  - Selects a take from the generated index and exposes that take as active
    `cine_linx` audio metadata.
- `IAMCCS_VideoHardConcat`
  - Hard-concatenates up to five generated take videos into one final Comfy
    `VIDEO` object.
  - Can concatenate clip audio, use a supplied master audio input, keep only
    the first video's audio, or output silent video.

Commit safety:

- The root loader imports this folder inside a guarded `try`.
- If this folder is excluded from a commit, IAMCCS-nodes still loads normally.
- The module does not patch stable Shotboard or AudioBoard backend files.

Design rule:

External/generated audio remains custom-audio metadata from AudioBoard/BusOut.
The multigeneration layer only creates local take windows for sequential
video-driven generation.
