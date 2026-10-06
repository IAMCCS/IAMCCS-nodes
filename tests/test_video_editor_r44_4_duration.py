from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "cine_multigeneration" / "__init__.py"


def _load_functions(*names):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in set(names)
    ]
    module = ast.Module(body=nodes, type_ignores=[])
    ns = {"Any": Any, "Dict": Dict, "re": __import__("re")}
    exec(compile(module, str(SOURCE), "exec"), ns)
    return ns


def test_audio_hard_cuts_are_lane_local_and_heal_old_global_cut():
    ns = _load_functions("_safe_float", "_safe_int", "_editor_audio_lane_key", "_enforce_editor_audio_hard_cuts")
    enforce = ns["_enforce_editor_audio_hard_cuts"]
    manifest = {
        "fps": 24,
        "clips": [
            {
                "id": "a1-long", "type": "audio", "trackId": "A1", "audioLane": "A1",
                "startTime": 0.0, "duration": 3.0, "trimStart": 0.0, "trimEnd": 3.0,
                "sourceDuration": 30.0, "sourceDurationLimit": 30.0,
                "audioHardCutAt": 3.0, "audioCollisionPolicy": "stop_at_next_audio_edit",
            },
            {
                "id": "a2-other-lane", "type": "audio", "trackId": "A2", "audioLane": "A2",
                "startTime": 3.0, "duration": 5.0, "trimStart": 0.0, "trimEnd": 5.0,
                "sourceDuration": 5.0, "sourceDurationLimit": 5.0,
            },
        ],
    }
    enforce(manifest)
    first = manifest["clips"][0]
    assert first["duration"] == 30.0
    assert first["trimEnd"] == 30.0
    assert "audioHardCutAt" not in first


def test_audio_hard_cut_still_applies_to_next_clip_on_same_lane():
    ns = _load_functions("_safe_float", "_safe_int", "_editor_audio_lane_key", "_enforce_editor_audio_hard_cuts")
    enforce = ns["_enforce_editor_audio_hard_cuts"]
    manifest = {
        "fps": 24,
        "clips": [
            {
                "id": "a1-first", "type": "audio", "trackId": "A1", "audioLane": "A1",
                "startTime": 0.0, "duration": 20.0, "trimStart": 0.0, "trimEnd": 20.0,
                "sourceDuration": 20.0, "sourceDurationLimit": 20.0,
            },
            {
                "id": "a1-second", "type": "audio", "trackId": "A1", "audioLane": "A1",
                "startTime": 7.0, "duration": 4.0, "trimStart": 0.0, "trimEnd": 4.0,
                "sourceDuration": 4.0, "sourceDurationLimit": 4.0,
            },
        ],
    }
    enforce(manifest)
    first = manifest["clips"][0]
    assert first["duration"] == 7.0
    assert first["trimEnd"] == 7.0
    assert first["audioHardCutAt"] == 7.0


def test_manual_pair_key_does_not_pair_unrelated_manual_imports():
    ns = _load_functions("_manual_editor_pair_key")
    pair = ns["_manual_editor_pair_key"]
    video = {"id": "clip_manual_video_1000_4", "manual": True}
    embedded = {"id": "clip_manual_audio_1000_4", "manual": True}
    separate = {"id": "clip_manual_audio_2000_5", "manual": True}
    assert pair(video) == pair(embedded) == "1000"
    assert pair(video) != pair(separate)
