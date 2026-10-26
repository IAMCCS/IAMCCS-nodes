import base64
from contextvars import Context
import importlib
import io
import json
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "native_prompter_tests"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
# Route registration is irrelevant to these tests and must not start a server.
with patch.dict(sys.modules, {"server": SimpleNamespace(PromptServer=SimpleNamespace(instance=None))}):
    native = importlib.import_module(PACKAGE + ".iamccs_h3_native_prompt")
    prompter = importlib.import_module(PACKAGE + ".iamccs_prompter")


class NativeQwenTests(unittest.TestCase):
    def setUp(self):
        self.progress = SimpleNamespace(PROGRESS_BAR_HOOK=Mock())
        self.progress.set_progress_bar_global_hook = lambda hook: setattr(self.progress, "PROGRESS_BAR_HOOK", hook)
        self.comfy = ModuleType("comfy")
        self.comfy.utils = self.progress
        self.modules = patch.dict(sys.modules, {"comfy": self.comfy, "comfy.utils": self.progress})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_every_prompter_mode_routes_to_native_without_http(self):
        frontend = (ROOT / "web" / "iamccs_prompter_ui.js").read_text(encoding="utf-8").split("const MODE_META =", 1)[1].split("const MODE_ALIASES", 1)[0]
        self.assertEqual(set(re.findall(r"^    (\w+): \{", frontend, re.MULTILINE)), set(prompter.MODE_SECTIONS))
        for mode, fields in prompter.MODE_SECTIONS.items():
            key = fields[0][0]
            text = "r34l1sm Replace the main person in <Video 1> with <Picture 1>." if mode == "v2va_face_swap_lora" else "A person walks slowly through a forest."
            with self.subTest(mode=mode), patch.object(prompter, "native_json_chat", return_value={key: text}) as chat, patch.object(prompter, "_http_json", side_effect=AssertionError("HTTP was called")):
                sections, report = prompter.rewrite_sections_with_ai("qwen_native", "", "__workflow__", "", mode,
                    {key: "rough idea"}, target_keys=[key], user_direction="A person walks slowly through a forest.")
                self.assertTrue(sections[key])
                self.assertEqual(report["provider"], "qwen_native")
                self.assertEqual(chat.call_count, 1)

    def test_long_multishot_composition_preserves_continuous_dialogue_and_cut_plan(self):
        project = {"sections": {"multishot_global_direction": "Two speakers in one room.",
            "multishot_shot_plan": "[Shot 1] Wide view. [Shot 2] Hard cut to the listener.",
            "multishot_dialogue_map": "<Subject 1> (S1): <d>[Italian] Resta qui con me.</d>",
            "multishot_environment": "Quiet room tone."}}
        text, report = prompter._compose_prompt(project, "multi_shot_lipsync", "manual", "")
        self.assertIn("Hard cut to the listener", text)
        self.assertIn("<d>[Italian] Resta qui con me.</d>", text)
        self.assertIn("Quiet room tone", text)
        self.assertEqual(report["task_mode"], "multi_shot_lipsync")

    def test_reference_panels_retain_order_and_input_is_unchanged(self):
        items = []
        for slot, color in (("1", "red"), ("2", "blue")):
            buffer = io.BytesIO()
            Image.new("RGB", (40, 40), color).save(buffer, format="PNG")
            items.append({"slot": slot, "data": base64.b64encode(buffer.getvalue()).decode()})
        original = json.dumps(items)
        tensor = native._image_tensor(items)
        self.assertEqual(tuple(tensor.shape), (1, 536, 1024, 3))
        self.assertGreater(float(tensor[0, 30, 250, 0]), .9)
        self.assertGreater(float(tensor[0, 30, 762, 2]), .9)
        self.assertEqual(json.dumps(items), original)

    def test_partial_json_and_wrong_field_types_are_rejected(self):
        schema = {"type": "object", "required": ["prompt"], "properties": {"prompt": {"type": "string"}}}
        for text in ('{"prompt":"unfinished', '{"other":"text"}', '{"prompt":[]}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                native._parse(text, schema)
        self.assertEqual(native._parse('```json\n{"prompt":"ok"}\n```', schema), {"prompt": "ok"})

    def test_one_encoder_per_request_and_bounded_json_repair(self):
        clip = object()
        loader = Mock()
        loader.load.return_value = (clip,)
        generator = Mock()
        generator.generate.side_effect = [('broken draft',), ('{"prompt":"first"}',), ('{"prompt":"second"}',)]
        fake_nodes = SimpleNamespace(NODE_CLASS_MAPPINGS={"ClipProjDeviceLoader": lambda: loader, "ClipProjGenerate": lambda: generator})
        fake_mm = SimpleNamespace(get_torch_device=lambda: "cpu")
        fake_comfy = self.comfy
        fake_comfy.model_management = fake_mm
        server = SimpleNamespace(PromptServer=SimpleNamespace(instance=SimpleNamespace(prompt_queue=SimpleNamespace(get_current_queue=lambda: ([], [])))))
        with patch.dict(sys.modules, {"nodes": fake_nodes, "comfy": fake_comfy, "comfy.model_management": fake_mm, "server": server}), patch.object(native, "native_models", return_value=["qwen3vl_4b_fp8_scaled.safetensors"]):
            def request():
                first = native.native_json_chat("__workflow__", "system", "first")
                second = native.native_json_chat("__workflow__", "system", "second")
                return first, second
            result = native.run_ai_request({"provider": "qwen_native"}, request)
        self.assertEqual(result, ({"prompt": "first"}, {"prompt": "second"}))
        self.assertEqual(loader.load.call_count, 1)
        self.assertEqual(generator.generate.call_count, 3)
        self.assertIsNone(native._session.get())

    def test_busy_render_and_overlapping_requests_do_not_load_models(self):
        server = SimpleNamespace(PromptServer=SimpleNamespace(instance=SimpleNamespace(prompt_queue=SimpleNamespace(get_current_queue=lambda: (["render"], [])))))
        fake_comfy = self.comfy
        fake_comfy.model_management = SimpleNamespace()
        with native.native_request():
            with self.assertRaisesRegex(RuntimeError, "already"):
                with native.native_request():
                    pass
            with patch.dict(sys.modules, {"nodes": SimpleNamespace(), "comfy": fake_comfy, "comfy.model_management": fake_comfy.model_management, "server": server}), self.assertRaisesRegex(RuntimeError, "rendering"):
                native.native_json_chat("__workflow__", "system", "user")
        with native.native_request():
            self.assertIsNotNone(native._session.get())

    def test_request_progress_does_not_update_a_render_and_hook_is_restored_on_error(self):
        original = self.progress.PROGRESS_BAR_HOOK
        with self.assertRaises(ValueError):
            with native.native_request():
                self.progress.PROGRESS_BAR_HOOK(1, 5, None)
                original.assert_not_called()
                # A render in a separate context still receives its normal progress.
                Context().run(self.progress.PROGRESS_BAR_HOOK, 2, 5, None)
                original.assert_called_once_with(2, 5, None)
                raise ValueError("request failed")
        self.assertIs(self.progress.PROGRESS_BAR_HOOK, original)


if __name__ == "__main__":
    unittest.main()
