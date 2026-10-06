import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('prompter', Path(__file__).resolve().parents[1] / 'iamccs_prompter.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class ImageVisionTests(unittest.TestCase):
    def rewrite(self, **kwargs):
        defaults = dict(provider='ollama', base_url='http://localhost:11434', model='writer', api_key='',
                        task_mode='reference_image', sections={}, target_keys=['detailed_description'],
                        user_direction='REFERENCE SHEET: character and book together')
        defaults.update(kwargs)
        return p.rewrite_sections_with_ai(**defaults)

    def test_two_models_keep_every_picture_and_writer_gets_only_observations(self):
        images = [dict(slot=str(i), data='YWJj', name=f'pic{i}', role='reference') for i in (1, 2)]
        observations = {'pictures': [{'slot': '1', 'visible_attributes': 'blue jacket'},
                                     {'slot': '2', 'visible_attributes': 'red book'}]}
        calls = []
        def transport(url, payload, headers, timeout):
            calls.append(payload)
            output = observations if payload['model'] == 'vision' else {'detailed_description': '[Shot 1] One sheet: blue jacket character and red book.'}
            return {'message': {'content': json.dumps(output)}}
        with patch.object(p, '_http_json', side_effect=transport):
            result, report = self.rewrite(images=images, vision_model='vision')
        self.assertEqual([item['model'] for item in calls], ['vision', 'writer'])
        self.assertEqual(len(calls[0]['messages'][1]['images']), 2)
        self.assertNotIn('images', calls[1]['messages'][1])
        self.assertIn('red book', calls[1]['messages'][1]['content'])
        self.assertIn('ONE combined canvas', calls[1]['messages'][0]['content'])
        self.assertEqual(report['vision_model'], 'vision')
        self.assertIn('red book', result['detailed_description'])

    def test_empty_request_derives_subject_and_preserves_reference_style(self):
        images = [dict(slot='1', data='YWJj', name='reference', role='reference')]
        calls = []
        def transport(url, payload, headers, timeout):
            calls.append(payload)
            if payload['model'] == 'vision':
                output = {'pictures': [{
                    'slot': '1',
                    'subjects': 'ceramic bird sculpture',
                    'visible_attributes': 'blue glaze and chipped beak',
                    'medium_style': 'studio product photograph',
                    'lighting': 'soft side light',
                    'uncertainties': '',
                    'role': 'reference',
                }]}
            else:
                output = {'detailed_description': '[Shot 1] A studio product-photography reference sheet of the blue ceramic bird sculpture.'}
            return {'message': {'content': json.dumps(output)}}
        with patch.object(p, '_http_json', side_effect=transport):
            result, report = self.rewrite(
                user_direction='', images=images, vision_model='vision',
                target_keys=['detailed_description'], sections={},
            )
        writer_system = calls[1]['messages'][0]['content']
        writer_user = calls[1]['messages'][1]['content']
        self.assertIn('Preserve the source medium', writer_system)
        self.assertIn('ceramic bird sculpture', writer_user)
        self.assertIn('studio product photograph', writer_user)
        self.assertNotIn('girl', writer_system.lower())
        self.assertIn('ceramic bird sculpture', result['detailed_description'])
        self.assertEqual(report['vision_model'], 'vision')

    def test_incomplete_vision_stops_before_writer(self):
        with patch.object(p, '_http_json', return_value={'message': {'content': '{"pictures": []}'}}) as call:
            with self.assertRaisesRegex(RuntimeError, 'Vision model'):
                self.rewrite(images=[dict(slot='1', data='YWJj')], vision_model='vision')
            self.assertEqual(call.call_count, 1)

    def test_text_only_does_not_call_vision(self):
        with patch.object(p, '_http_json', return_value={'message': {'content': '{"detailed_description":"[Shot 1] Bronze vase."}'}}) as call:
            self.rewrite(vision_model='vision')
        self.assertEqual(call.call_count, 1)
        self.assertEqual(call.call_args.args[1]['model'], 'writer')

    def test_rejects_invented_picture_or_temporal_shots(self):
        for text in ('<Picture 1> shows a vase.', '[Shot 2] A second view.'):
            with patch.object(p, '_http_json', return_value={'message': {'content': json.dumps({'detailed_description': text})}}):
                with self.assertRaises(RuntimeError):
                    self.rewrite()

    def test_existing_modes_and_single_model_images(self):
        for mode in ('t2va', 'i2va', 'fl2va', 'ref2va', 'v2va_object_swap', 'audio_driven'):
            key = p.MODE_SECTIONS[mode][0][0]
            with patch.object(p, '_http_json', return_value={'message': {'content': json.dumps({key: 'User scene'})}}) as call:
                result, _ = self.rewrite(task_mode=mode, target_keys=[key], images=[dict(data='YWJj')])
            self.assertEqual(result[key], 'User scene')
            self.assertIn('images', call.call_args.args[1]['messages'][1])

    def test_composer_uses_full_reference_grammar(self):
        result, _ = p._compose_prompt({'sections': {'subject_definitions': '<Subject 1>: bronze vase',
                                                   'summary': 'One still', 'detailed_description': '[Shot 1] Static tabletop.'}},
                                     'reference_image', 'manual', '')
        self.assertIn('subject_definitions:', result)
        self.assertIn('summary:\n[reference generation]', result)
        self.assertNotIn('integrated_multimodal_description', result)
        self.assertNotIn('<Picture', result)


if __name__ == '__main__':
    unittest.main()
