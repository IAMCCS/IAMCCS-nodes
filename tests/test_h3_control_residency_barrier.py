import ast
import gc
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch


class ControlResidencyBarrierTest(unittest.TestCase):
    def test_lazy_control_preserves_patchers_in_both_model_families(self):
        source = Path(__file__).resolve().parents[1] / 'iamccs_minimax_h3_atomic_backend.py'
        tree = ast.parse(source.read_text(encoding='utf-8-sig'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_release_conditioning_models')
        mm = types.ModuleType('comfy.model_management')
        mm.unload_all_models = Mock()
        mm.cleanup_models = Mock()
        mm.soft_empty_cache = Mock()
        comfy = types.ModuleType('comfy')
        comfy.model_management = mm
        scope = {'Any': object, 'torch': types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)), 'gc': gc, 'os': types.SimpleNamespace(name='posix'), 'LOG': Mock()}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), scope)
        with patch.dict(sys.modules, {'comfy': comfy, 'comfy.model_management': mm}):
            for task in ['t2va', 'i2va', 'fl2va', 'ref2va']:
                report = scope['_release_conditioning_models']({'fun_controlnet': {'enabled': True}}, task)
                self.assertIn('patchers preserved', report)
            mm.unload_all_models.assert_not_called()
            scope['_release_conditioning_models']({'fun_controlnet': {'enabled': False}}, 'ref2va')
            mm.unload_all_models.assert_called_once()


if __name__ == '__main__':
    unittest.main()
