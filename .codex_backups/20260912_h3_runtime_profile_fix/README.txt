IAMCCS H3 runtime-profile fix — 2026-09-12

Purpose
-------
Herrgotts FL2VA Continuous AV and Motion Context Auto-Chain originally used
separate process-global MiniMax H3 hooks. The final fix replaces that split
with one marker-gated superset hook compatible with both ABIs.

Files changed
-------------
D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-H3-Motion-Context-Auto-Chain-addon\__init__.py
D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-H3-Motion-Context-Auto-Chain-addon\motion_context_core.py
D:\ComfyUI\ComfyUI\custom_nodes\comfyui-h3-multishot\h3_avbank_probe.py
D:\ComfyUI\ComfyUI\custom_nodes\IAMCCS-nodes\iamccs_minimax_h3_herrgotts.py

Behavior
--------
Use the normal ComfyUI launcher. Motion Context and Continuous AV markers are
handled by the same wrapper; unrelated H3 graphs remain on stock behavior.
H3AVBank recognises the shared Motion Context marker and stands down safely.
The IAMCCS adapter preflights compatibility before the first expensive sample.
Herrgotts' standalone self-test was also updated for ComfyUI 0.34, whose
PackedLayout constructor no longer accepts frame_count.

No pack was deleted, moved, disabled globally, or wrapped twice.
