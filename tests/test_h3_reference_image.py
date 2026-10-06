import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = load('iamccs_h3_reference_image')
core = load('iamccs_minimax_h3_shotboard_core')
advisor = load('iamccs_h3_advisor')


class ReferenceTests(unittest.TestCase):
    def test_text_only_real_planner_and_profile_retained(self):
        plan = core.build_shotplan(timeline_data='{}', global_prompt='A bronze vase',
                                  duration_seconds=5/24, task_mode='ref2va', width=640, height=384)
        plan['sampling'] = {'steps':16, 'seed':123}
        plan['performance_profile'] = 'rtx_xx60_safe'
        result = reference.reference_image_plan(plan, 'A bronze vase')
        self.assertEqual(result['total_segments'], 1)
        self.assertEqual(result['chunks'][0]['frame_count'], 5)
        self.assertEqual(result['chunks'][0]['task_mode'], 'ref2va')
        self.assertEqual(result['sampling'], plan['sampling'])
        self.assertEqual(result['performance_profile'], 'rtx_xx60_safe')
        self.assertFalse(result['upscale_enabled'])
        self.assertEqual(result['reference_image']['source_policy'], 'prompter_ref2v_then_references')
        self.assertEqual(result['reference_image']['prompter_paths'], [])
        self.assertEqual(result['reference_image']['fallback_paths'], [])
        self.assertNotIn('reference_image', plan)

    def test_empty_description_rejected(self):
        with self.assertRaises(ValueError):
            reference.reference_image_plan({}, '  ')

    def test_family_and_serialization(self):
        self.assertEqual(advisor.task_family('reference_image'), 'ref2va')
        self.assertEqual(advisor.task_family('i2va'), 'fl2va')
        import json
        path = Path('X:/1_UNIVERSAL_42_43/R42_REFERENCE_IMAGE_GENERATOR_API.json')
        prompt = json.loads(path.read_text(encoding='utf-8'))['prompt']
        self.assertEqual(prompt['811']['inputs']['task_mode'], 'reference_image')
        self.assertEqual(prompt['9']['inputs']['model'], ['8',0])
        self.assertEqual(prompt['9']['inputs']['clip'], ['625',0])
        self.assertFalse(any('Editor' in n['class_type'] for n in prompt.values()))
        for node in prompt.values():
            for value in node['inputs'].values():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
                    self.assertIn(value[0], prompt)

    def test_prompter_saved_images_supply_backend_without_direct_socket(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, color in (("character.png", (40, 80, 120)), ("prop.png", (120, 80, 40))):
                Image.new("RGB", (32, 24), color).save(root / name)
            project = {"ai_visual_files": [{"path": "character.png"}, {"path": "prop.png"}]}
            linx = {"iamccs_minimax_h3_settings": {"settings": {
                "task_mode": "reference_image", "h3_reference_image_source": "auto_prompter_then_direct",
                "width": 3840, "height": 2160,
            }}}
            bridge = reference.IAMCCS_R42ReferenceImagePromptBridge()
            with patch.object(reference.folder_paths, "get_input_directory", return_value=directory), \
                 patch.object(reference.folder_paths, "get_annotated_filepath", side_effect=lambda name: str(root / name)):
                self.assertEqual(bridge.check_lazy_status("prompt", json.dumps(project), linx), [])
                result = bridge.adapt("prompt", json.dumps(project), linx)
            self.assertEqual(tuple(result[3].shape), (1, 24, 32, 3))
            self.assertEqual(tuple(result[8].shape), (1, 24, 32, 3))
            self.assertIsNone(result[9])
            self.assertIn("prompter_ref2v+inject", result[7])


if __name__ == '__main__':
    unittest.main()
