# Dialogue audio Render

Restart ComfyUI and reload the page after updating the Python worker and UI.

1. Edit dialogue in Tag Editor, choose the TTS engine and create its rig with **Add Rig**.
2. **Render** queues the connected audio rig up to AudioBoard only. It saves the generated WAVs, updates AudioBoard and publishes to the connected Shotboard. It does not start video generation.
3. After completion, ordinary **Queue** reads the saved AudioBoard clips and the current Shotboard. The upstream TTS rig remains connected visually but is not executed again.
4. Use **Render** again for new dialogue audio. Explicit Prompter actions remain available.

If Shotboard is edited while TTS is running, the new audio is retained in AudioBoard without replacing Shotboard. Use **Publish** to import it when ready.

The current Shotboard audio, including intentional deletion, takes precedence over upstream audio at Queue. The active take's audio and duration maps are saved with its current edits instead of being restored from older T01 data.

The Long Multi-Shot preset's B4 line is: `Because even the universe doesn't know how to say goodbye.` Existing authored dialogue is retained until the preset is explicitly loaded.

OmniVoice rig creation separates spoken text from supported voice-design instructions. Existing rig engines whose instructions contain the dialogue are repaired. The isolated worker uses UTF-8 for its JSON protocol on Windows.

Validation: audio-only queue and handoff scenarios, concurrent Shotboard edits, queue failures, refresh with stale T01 audio, backend audio authority, existing LongVid contracts, and native OmniVoice instruction validation. Worker Unicode-error transport was checked without loading model weights; a complete GPU speech/video render was not run.
