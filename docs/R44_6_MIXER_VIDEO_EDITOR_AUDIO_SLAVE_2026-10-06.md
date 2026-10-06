# R44.6 — IAMCCS AudioBoard Mixer → Shotboard Video Editor Audio Slave

The Mixer is the editorial audio master when `sync_policy = mixer_overrides_cine_linx`.

Supported live slave state in Shotboard Video Editor:
- A1–A5 volume
- gain dB
- pan
- mute / solo
- normalize / bypass / reverb metadata
- effect chains
- master bus
- master gain

Recommended graph:

`EditorTakeRoute cine_linx -> AudioBoard Mixer -> BusOut -> Video Editor cine_editor_linx`

Keep the original `EditorTakeRoute cine_linx -> Video Editor cine_linx` connection as generation/take truth. The editorial audio branch must not replace it.

The Video Editor frontend exposes `_iamccsVideoEditorAudioSlave`; the Mixer discovers the downstream editor through graph links and mirrors its state live. Backend queue/render repeats the same authority so preview and final export agree.
