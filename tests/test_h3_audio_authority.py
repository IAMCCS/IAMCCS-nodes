import ast
import copy
import json
from pathlib import Path
from typing import Any
import unittest


ROOT = Path(__file__).parents[1]
TREE = ast.parse((ROOT / "iamccs_minimax_h3_shotboard.py").read_text(encoding="utf-8"))
NAMES = {"H3_AUDIO_MODE_ALIASES", "H3_AUDIO_MODES", "_timeline_dict", "_normalise_h3_audio_mode", "_apply_audioboard_to_minimax"}
PARTS = [item for item in TREE.body if
         isinstance(item, ast.FunctionDef) and item.name in NAMES or
         isinstance(item, ast.Assign) and any(isinstance(target, ast.Name) and target.id in NAMES for target in item.targets)]
SCOPE = {"copy": copy, "json": json, "Any": Any}
exec(compile(ast.Module(body=PARTS, type_ignores=[]), str(ROOT / "iamccs_minimax_h3_shotboard.py"), "exec"), SCOPE)


class ShotboardAudioAuthorityTests(unittest.TestCase):
    def apply(self, timeline, mode="h3_custom_audio_drive"):
        old_audio = {"audioSegments": [{"id": "old", "audioFile": "old.wav", "start": 0, "length": 360}]}
        linx = {"resources": {"cine_audio_timeline_json": json.dumps(old_audio)}}
        result, report = SCOPE["_apply_audioboard_to_minimax"](linx, json.dumps(timeline), mode)
        return json.loads(result), report

    def test_current_trim_and_duration_win_over_upstream_audio(self):
        timeline = {"truth_revision": 42, "duration_seconds": 9,
                    "audioSegments": [{"id": "current", "audioFile": "new.wav", "start": 8, "trimStart": 20, "length": 130}]}
        result, report = self.apply(timeline)
        self.assertEqual(result, timeline)
        self.assertFalse(report["applied"])

    def test_intentional_deletion_does_not_resurrect_audio(self):
        timeline = {"truth_revision": 43, "duration_seconds": 7, "audioSegments": []}
        result, _ = self.apply(timeline)
        self.assertEqual(result, timeline)

    def test_legacy_board_can_still_import_published_lanes(self):
        result, report = self.apply({"duration_seconds": 9})
        self.assertTrue(report["applied"])
        self.assertEqual(result["audioSegments"][0]["audioFile"], "old.wav")
        self.assertEqual(result["duration_seconds"], 9)

    def test_native_h3_audio_keeps_its_existing_policy(self):
        result, report = self.apply({"truth_revision": 43, "audioSegments": [{"audioFile": "new.wav"}]}, "h3_native_generated")
        self.assertEqual(result["audioSegments"], [])
        self.assertEqual(report["reason"], "native_h3_audio_selected")


if __name__ == "__main__":
    unittest.main()
