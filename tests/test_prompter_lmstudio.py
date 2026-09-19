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

if __name__ == '__main__': unittest.main()
