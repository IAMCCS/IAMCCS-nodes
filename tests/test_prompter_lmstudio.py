import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

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

if __name__ == '__main__': unittest.main()
