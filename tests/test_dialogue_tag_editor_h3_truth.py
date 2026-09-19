import importlib.util
import json
from pathlib import Path
import unittest


PATH = Path(__file__).parents[1] / "audio" / "dialogue_tag_editor.py"
SPEC = importlib.util.spec_from_file_location("dialogue_tag_editor_under_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class DialogueTagEditorH3TruthTests(unittest.TestCase):
    def _contract_linx(self):
        speakers = [
            {"id": "A", "name": "Actor A", "language": "it", "subject_tag": "<Subject 1>", "speaker_tag": "(S1)"},
            {"id": "B", "name": "Actor B", "language": "en", "subject_tag": "<Subject 2>", "speaker_tag": "(S2)"},
        ]
        lines = [
            {"id": "one", "speaker": "A", "text": "Resta qui.", "local_prompt": "Tight close-up, stable eyeline."},
            {"id": "two", "speaker": "B", "text": "I cannot.", "local_prompt": "Reverse close-up.", "minimax_boundary_mode": "cutoff_at_end"},
        ]
        contract = MODULE._build_minimax_dialogue_contract("Night interior.", speakers, lines)
        return {"resources": {"iamccs_minimax_h3_dialogue_contract": contract}}

    def test_h3_prompt_grammar_and_cutoff(self):
        contract = self._contract_linx()["resources"]["iamccs_minimax_h3_dialogue_contract"]
        self.assertIn("<Subject 1> (S1): <d>[Italian] Resta qui.</d>", contract["global_truth"])
        self.assertIn("<Subject 2> (S2): <d>[English] I cannot. <cutoff></d>", contract["global_truth"])
        self.assertIn("integrated_multimodal_description:", contract["local_truth"][0])

    def test_explicit_injection_changes_prompts_but_preserves_shotboard_truth(self):
        timeline = {
            "rows": [
                {"id": "guide_5", "type": "image", "prompt": "Old five", "transition": "continuous", "camera": "locked", "start": 120, "length": 48, "imageFile": "g5.png"},
                {"id": "guide_6", "type": "image", "prompt": "Old six", "transition": "hard_cut", "camera": "pan", "start": 168, "length": 48, "imageFile": "g6.png"},
            ],
            "segments": [
                {"id": "guide_5", "type": "image", "prompt": "Old five", "transition": "continuous", "camera": "locked", "start": 120, "length": 48, "imageFile": "g5.png"},
                {"id": "guide_6", "type": "image", "prompt": "Old six", "transition": "hard_cut", "camera": "pan", "start": 168, "length": 48, "imageFile": "g6.png"},
            ],
        }
        final_global, encoded, report = MODULE.apply_dialogue_to_minimax(
            self._contract_linx(), "Existing Shotboard global", json.dumps(timeline)
        )
        updated = json.loads(encoded)
        self.assertTrue(report["applied"])
        self.assertIn("Existing Shotboard global", final_global)
        for collection_name in ("rows", "segments"):
            original = timeline[collection_name]
            result = updated[collection_name]
            for before, after in zip(original, result):
                for key in ("id", "transition", "camera", "start", "length", "imageFile"):
                    self.assertEqual(after[key], before[key])
                self.assertIn("integrated_multimodal_description:", after["prompt"])


if __name__ == "__main__":
    unittest.main()
