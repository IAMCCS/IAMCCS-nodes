import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
package = types.ModuleType('omni_prompter_tests')
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location(package.__name__ + '.iamccs_prompter', ROOT / 'iamccs_prompter.py')
PROMPTER = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PROMPTER)


class OmniPrompterTests(unittest.TestCase):
    def test_nine_vision_references_survive_normalization(self):
        images = [{'slot': str(i), 'data': 'YWJj', 'name': str(i)} for i in range(1, 10)]
        self.assertEqual(len(PROMPTER._normalise_ai_images(images)), 9)

    def test_voice_mapping_is_deterministic_and_media_survives_transport(self):
        project = {'task_mode': 'ref2va', 'omni_preset': True,
                   'sections': {'summary': 'Two people talk.', 'detailed_description': '[Shot 1] A close-up.'},
                   'ai_visual_files': [{'path': f'picture{i}.png'} for i in range(1, 10)],
                   'omni_audio_files': [{'path': 'voice.wav'}, None],
                   'omni_voice_bindings': [{'enabled': True, 'shot': 1, 'picture': 2, 'audio': 1, 'character': 'Knight'},
                                           {'enabled': False, 'shot': 2, 'picture': 1, 'audio': 2}]}
        prompt, _ = PROMPTER._compose_prompt(project, 'ref2va', 'manual', '')
        self.assertIn('<Picture 2> speaks with the voice identity of <Audio 1>', prompt)
        self.assertNotIn('<Audio 2>', prompt)
        bus = PROMPTER._append_prompter_stage({}, {}, [], prompt, json.dumps(project), '', 'ref2va', 'global', len(prompt))
        self.assertEqual(len(bus['resources']['iamccs_h3_omni_image_files']), 9)
        self.assertEqual(bus['resources']['iamccs_h3_omni_audio_files'][0]['path'], 'voice.wav')
        self.assertEqual(len(bus['resources']['iamccs_h3_omni_voice_bindings']), 1)
        project['omni_preset'] = False
        bus = PROMPTER._append_prompter_stage({}, {}, [], prompt, json.dumps(project), '', 'ref2va', 'global', len(prompt))
        self.assertEqual(bus['resources']['iamccs_h3_omni_audio_files'], [])


if __name__ == '__main__':
    unittest.main()
