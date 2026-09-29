import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import sys
import types

path = Path(__file__).parents[1] / 'iamccs_prompter.py'
spec = importlib.util.spec_from_file_location('prompter_lm_test', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class LMStudioTests(unittest.TestCase):
    def test_prompter_is_partial_execution_output(self):
        self.assertIs(module.IAMCCS_Prompter.OUTPUT_NODE, True)

    def test_local_url_and_no_cloud_key_inheritance(self):
        answer = {'choices':[{'message':{'content':'{"scene":"Test rewritten scene"}'}}]}
        with patch.object(module, '_http_json', return_value=answer) as http, patch.dict(os.environ, {'OPENAI_API_KEY':'cloud-key-must-not-leak'}):
            value, report = module.rewrite_sections_with_ai('lm_studio', '', 'local-model', '', 't2va', {'scene':'test'})
        self.assertEqual(http.call_args.args[0], 'http://localhost:1234/v1/chat/completions')
        self.assertEqual(http.call_args.args[2], {})
        self.assertEqual(value['scene'], 'Test rewritten scene')
        self.assertEqual(report['provider'], 'lm_studio')

    def test_host_only_url_gets_v1_once(self):
        answer = {'choices':[{'message':{'content':'{"scene":"Test"}'}}]}
        with patch.object(module, '_http_json', return_value=answer) as http:
            module.rewrite_sections_with_ai('lm_studio', 'http://localhost:1234/', 'local-model', '', 't2va', {'scene':'test'})
        self.assertEqual(http.call_args.args[0], 'http://localhost:1234/v1/chat/completions')

    def test_ollama_contract_unchanged(self):
        with patch.object(module, '_http_json', return_value={'message':{'content':'{"scene":"Test"}'}}) as http:
            module.rewrite_sections_with_ai('ollama', '', 'local-model', '', 't2va', {'scene':'test'})
        self.assertEqual(http.call_args.args[0], 'http://127.0.0.1:11434/api/chat')

    def test_narrative_request_can_fill_blank_extended_evolving_targets(self):
        answer = {'choices':[{'message':{'content':'{"action":"0-4 seconds: breathe steadily.\\nAt 4 seconds start walking.","shot_list":"0-4 seconds: breathe steadily.\\nAt 4 seconds start walking."}'}}]}
        with patch.object(module, '_http_json', return_value=answer):
            value, report = module.rewrite_sections_with_ai(
                'lm_studio', '', 'local-model', '', 'fl2va',
                {'action':'', 'shot_list':''},
                user_direction='Create timed phases from this request.',
                target_keys=['action', 'shot_list'],
            )
        self.assertIn('At 4 seconds', value['action'])
        self.assertEqual(report['selected_sections'], ['action', 'shot_list'])

    def test_continuous_request_can_fill_one_blank_action_target(self):
        answer = {'message': {'content': '{"action":"A goblin swims steadily toward the camera for the complete uninterrupted take."}'}}
        with patch.object(module, '_http_json', return_value=answer):
            value, report = module.rewrite_sections_with_ai(
                'ollama', '', 'gemma4:e4b', '', 'fl2va',
                {'action': '', 'shot_list': ''},
                user_direction='Keep the requested action continuous without times or phases.',
                target_keys=['action'],
            )
        self.assertIn('complete uninterrupted take', value['action'])
        self.assertEqual(report['selected_sections'], ['action'])

    def test_ollama_retries_compact_prompt_after_repeat_limit(self):
        answers = [
            RuntimeError('AI provider HTTP 500: prediction aborted, token repeat limit reached'),
            {'message': {'content': '{"scene":"An empty station.","shot_list":[{"description":"A woman walks while the camera tracks backward."}]}'}}
        ]
        with patch.object(module, '_http_json', side_effect=answers) as http:
            value, report = module.rewrite_sections_with_ai(
                'ollama', '', 'qwen2.5vl:3b', '', 't2va',
                {'scene': '', 'shot_list': ''},
                user_direction='A woman walks through an empty station.',
                target_keys=['scene', 'shot_list'],
            )
        self.assertEqual(http.call_count, 2)
        self.assertEqual(value['scene'], 'An empty station.')
        self.assertIn('description:', value['shot_list'])
        self.assertEqual(report['transport_retry'], 'ollama_compact_after_repeat_limit')

    def test_evolving_recovers_user_timestamps_when_model_misformats_timeline(self):
        answer = {'message': {'content': '{"action":"[ONSET ONCE] She walks. [THEN SUSTAIN] She keeps walking.","shot_list":"malformed"}'}}
        with patch.object(module, '_http_json', return_value=answer):
            value, _ = module.rewrite_sections_with_ai(
                'ollama', '', 'gemma4:e4b', '', 'fl2va',
                {'action': '', 'shot_list': ''},
                user_direction='Extended Evolving. REQUEST: A woman walks toward camera; at 4 seconds she starts running.',
                target_keys=['action', 'shot_list'],
            )
        self.assertEqual(value['action'], value['shot_list'])
        self.assertIn('0-4 seconds:', value['action'])
        self.assertIn('At 4 seconds:', value['action'])

    def test_prompter_returns_plain_final_prompt_without_audio(self):
        project = module.default_project()
        project['sections']['scene'] = 'A controlled one-shot performance.'
        result = module.IAMCCS_Prompter().compose(
            project_data=module.json.dumps(project),
            task_mode='t2va',
            injection_target='global',
            writing_mode='manual',
            merge_policy='replace',
            character_budget=6800,
        )
        self.assertEqual(len(result['result']), 6)
        self.assertIn('A controlled one-shot performance.', result['result'][1])
        self.assertEqual(result['result'][4], '')
        self.assertEqual(result['result'][5], '')

    def test_audio_transcript_is_formatted_as_h3_dialogue_tag(self):
        project = module.default_project()
        project['sections']['action'] = 'A close dialogue performance.'
        project['_transcribe_once'] = True

        class FakeCompiler:
            _clean = staticmethod(lambda value: str(value).strip())
            _transcribe = staticmethod(lambda *args, **kwargs: 'Resta qui con me.')

        fake_audio_module = types.ModuleType('iamccs_cine_audio_dialogue')
        fake_audio_module.IAMCCS_CineAudioTranscriptPromptCompiler = FakeCompiler
        with patch.dict(sys.modules, {'iamccs_cine_audio_dialogue': fake_audio_module}):
            result = module.IAMCCS_Prompter().compose(
                project_data=module.json.dumps(project),
                task_mode='fl2va',
                injection_target='global',
                writing_mode='manual',
                merge_policy='replace',
                character_budget=6800,
                audio_transcription_model='tiny',
                audio_dialogue_language='Italian',
                audio_dialogue_subject='2',
                audio={'waveform': object(), 'sample_rate': 16000},
            )
        self.assertEqual(result['result'][4], 'Resta qui con me.')
        self.assertEqual(
            result['result'][5],
            '<Subject 2> (S2): <d>[Italian] Resta qui con me.</d>',
        )

    def test_audio_only_partial_execution_can_transcribe_without_prompt_boxes(self):
        project = module.default_project()
        project['_transcribe_once'] = True

        class FakeCompiler:
            _clean = staticmethod(lambda value: str(value).strip())
            _transcribe = staticmethod(lambda *args, **kwargs: 'Audio only works.')

        fake_audio_module = types.ModuleType('iamccs_cine_audio_dialogue')
        fake_audio_module.IAMCCS_CineAudioTranscriptPromptCompiler = FakeCompiler
        with patch.dict(sys.modules, {'iamccs_cine_audio_dialogue': fake_audio_module}):
            result = module.IAMCCS_Prompter().compose(
                project_data=module.json.dumps(project),
                task_mode='t2va',
                injection_target='global',
                writing_mode='manual',
                merge_policy='replace',
                character_budget=6800,
                audio={'waveform': object(), 'sample_rate': 16000},
            )
        self.assertEqual(result['result'][1], '')
        self.assertEqual(result['result'][4], 'Audio only works.')
        self.assertEqual(result['ui']['iamccs_audio_transcription_error'], [''])

    def test_connected_audio_is_lazy_and_inert_during_global_queue(self):
        project = module.default_project()
        project['sections']['scene'] = 'One shot.'
        node = module.IAMCCS_Prompter()
        self.assertEqual(node.check_lazy_status(module.json.dumps(project), audio=None), [])
        result = node.compose(module.json.dumps(project), 't2va', 'global', 'manual', 'replace', 6800,
                              audio={'waveform': object(), 'sample_rate': 16000})
        self.assertEqual(result['result'][4], '')
        self.assertEqual(result['result'][5], '')
        project['_transcribe_once'] = True
        self.assertEqual(node.check_lazy_status(module.json.dumps(project), audio=None), ['audio'])

    def test_global_queue_keeps_existing_transcript_without_retranscribing(self):
        project = module.default_project()
        project['sections']['scene'] = 'One shot.'
        project['audio_transcript'] = 'Keep these exact words.'
        project['audio_dialogue_tag'] = '<Subject 1> (S1): <d>[English] Keep these exact words.</d>'
        result = module.IAMCCS_Prompter().compose(
            module.json.dumps(project), 't2va', 'global', 'manual', 'replace', 6800,
            audio={'waveform': object(), 'sample_rate': 16000})
        self.assertEqual(result['result'][4], project['audio_transcript'])
        self.assertEqual(result['result'][5], project['audio_dialogue_tag'])

    def test_ai_rewrite_preserves_exact_transcribed_dialogue_line(self):
        dialogue = '<Subject 1> (S1): <d>[English] Do not turn around.</d>'
        answer = {'choices':[{'message':{'content':'{"audio_dialogue_map":"S1 says turn around."}'}}]}
        with patch.object(module, '_http_json', return_value=answer):
            value, _ = module.rewrite_sections_with_ai(
                'lm_studio', '', 'local-model', '', 'audio_driven',
                {'audio_dialogue_map': dialogue}, target_keys=['audio_dialogue_map'])
        self.assertIn(dialogue, value['audio_dialogue_map'])

    def test_request_global_and_locals_preserves_exact_transcript(self):
        dialogue = '<Subject 2> (S2): <d>[Italian] Aspettami qui.</d>'
        answer = {'global_prompt': 'A quiet train platform.', 'shots': [{'slot': 1, 'local_prompt': 'A speaker waits.'}]}
        with patch.object(module, '_multimodal_json_chat', return_value=answer):
            plan, _ = module.build_visual_story_plan_with_ai(
                'lm_studio', '', 'local-model', '', f'One scene. {dialogue}', 'audio_driven', [])
        self.assertIn(dialogue, plan['global_prompt'])

if __name__ == '__main__': unittest.main()
