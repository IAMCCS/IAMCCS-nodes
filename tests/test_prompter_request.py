import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_prompter_native_qwen import prompter


class RequestCompilationTests(unittest.TestCase):
    def test_request_is_authority_and_selected_drafts_are_not_context(self):
        sections = {"scene": "Previous railway scene", "camera": "Old camera", "acting": "Keep her tired performance"}
        system, user = prompter._assistant_instruction(
            "t2va", sections, "Develop the requested shot", ["scene", "camera"],
            request_text="Una donna posa una tazza sul tavolo.",
        )
        message = json.loads(user)
        self.assertEqual(message["request"], "Una donna posa una tazza sul tavolo.")
        self.assertEqual(message["existing_context"], {"acting": "Keep her tired performance"})
        self.assertNotIn("Previous railway scene", user)
        self.assertIn("in English", system)
        self.assertIn("Never emit timing placeholders", system)
        self.assertEqual(sections["scene"], "Previous railway scene")

    def test_request_returns_generated_fields(self):
        generated = {"scene": "A woman approaches a table holding a cup.", "camera": "A fixed eye-level composition holds the table in view."}
        with patch.object(prompter, "native_json_chat", return_value=generated) as writer:
            fields, report = prompter.rewrite_sections_with_ai(
                "qwen_native", "", "__workflow__", "", "t2va", {"scene": "", "camera": ""},
                target_keys=["scene", "camera"], request_text="Una donna posa una tazza sul tavolo.",
            )
        self.assertEqual(fields, generated)
        self.assertEqual(report["rewritten_sections"], ["camera", "scene"])
        self.assertIn('"request":', writer.call_args.args[2])

    def test_request_echo_is_not_reported_as_success(self):
        brief = "Una donna posa una tazza sul tavolo."
        with patch.object(prompter, "native_json_chat", return_value={"scene": brief, "camera": brief}):
            with self.assertRaisesRegex(RuntimeError, "echoed REQUEST"):
                prompter.rewrite_sections_with_ai(
                    "qwen_native", "", "__workflow__", "", "t2va", {"scene": "", "camera": ""},
                    target_keys=["scene", "camera"], request_text=brief,
                )

    def test_single_field_rewrite_keeps_its_original_contract(self):
        _, user = prompter._assistant_instruction("t2va", {"scene": "A quiet room"}, target_keys=["scene"])
        self.assertEqual(json.loads(user)["rough_sections"], {"scene": "A quiet room"})


if __name__ == "__main__":
    unittest.main()
