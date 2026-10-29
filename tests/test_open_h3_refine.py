import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
PACKAGE = "_open_h3_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(PACKAGE + ".iamccs_minimax_h3_open_refine", ROOT / "iamccs_minimax_h3_open_refine.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
atomic = types.ModuleType(PACKAGE + ".iamccs_minimax_h3_atomic_backend")
atomic._resolve_shotplan = lambda value: value


class OpenH3SettingsTests(unittest.TestCase):
    def resolve(self, settings=None, **plan):
        payload = {"open_h3_refine": module.configuration(settings or {}), **plan}
        with patch.dict(sys.modules, {atomic.__name__: atomic}):
            return module.IAMCCS_MiniMaxH3OpenRefineControl().resolve(payload, 1)

    def test_defaults_keep_existing_workflows_disabled(self):
        self.assertFalse(self.resolve()[0])

    def test_open_refine_does_not_stack_another_upscale(self):
        with self.assertRaisesRegex(ValueError, "replaces the other upscale"):
            self.resolve({"h3_open_refine_enabled": True}, upscale_enabled=True, upscale_mode="h3_ultimate_tiled")

    def test_missing_assets_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "refiner LoRA"):
            self.resolve({"h3_open_refine_enabled": True, "h3_open_refine_lora_name": ""})

    def test_custom_schedule_prompt_and_segment_seed(self):
        result = self.resolve({"h3_open_refine_sigma_preset": "custom", "h3_open_refine_custom_sigmas": "0.5,0.2,0", "h3_open_refine_seed": 100, "h3_open_refine_prompt": "Preserve the yawn."}, global_prompt="Other prompt")
        self.assertEqual((result[1], result[11], result[12]), ("Preserve the yawn.", "0.5,0.2,0", 101))

    def test_blank_distill_is_disabled_and_shot_prompt_wins(self):
        result = self.resolve({"h3_open_refine_distill_lora_name": ""}, global_prompt="Global", chunks=[{}, {"prompt": "Local shot"}])
        self.assertEqual(result[1], "Local shot")
        self.assertEqual(result[10], 0.0)

    def test_reference_schedule_and_output_contract(self):
        result = self.resolve({"h3_open_refine_sigma_preset": "reference_3step"})
        self.assertEqual(result[11], module.SIGMAS["reference_3step"])
        self.assertEqual(len(result), len(module.IAMCCS_MiniMaxH3OpenRefineControl.RETURN_TYPES))
        self.assertEqual(len(result), len(module.IAMCCS_MiniMaxH3OpenRefineControl.RETURN_NAMES))

    def test_optional_face_source_is_retained_for_native_shots(self):
        self.assertTrue(self.resolve(face_detailer_settings={"enabled": True}, task_mode="ref2va")[13])
        self.assertFalse(self.resolve(face_detailer_settings={"enabled": True}, task_mode="longvid_keyframe_continuous")[13])


if __name__ == "__main__":
    unittest.main()
