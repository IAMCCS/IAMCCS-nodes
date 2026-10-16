import ast
import copy
import importlib.util
import json
import math
from collections.abc import Mapping
import sys
import types
import unittest
from pathlib import Path
from typing import Any, Iterable

import torch

ROOT = Path(__file__).parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


CORE = module('iamccs_minimax_h3_shotboard_core')


def v2v_helpers():
    tree = ast.parse((ROOT / 'iamccs_minimax_h3_v2v_backend.py').read_text(encoding='utf-8'))
    boundary = next(n.lineno for n in tree.body if isinstance(n, ast.ClassDef))
    body = [n for n in tree.body if n.lineno < boundary and isinstance(n, (ast.Assign, ast.FunctionDef))]
    scope = {'Any': Any, 'Iterable': Iterable, 'Mapping': Mapping, 'torch': torch,
             'math': math, 'json': json, 'H3_FPS': 24, 'align_h3_frames': CORE.align_h3_frames}
    exec(compile(ast.Module(body=body, type_ignores=[]), '<v2v-helpers>', 'exec'), scope)
    return scope


def node_scope(file, class_name, **extra):
    tree = ast.parse((ROOT / file).read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    scope = {'Any': Any, 'torch': torch, 'json': json, 'copy': copy,
             '__package__': 'r49_mode_test', 'SUPERNODE_LINX_TYPE': 'IAMCCS_SUPERNODE_LINX',
             'CATEGORY': 'test', 'H3_INPUT_CATEGORY': 'test', 'STAGE_NAME': 'test',
             'RESOURCE_PREFIX': 'iamccs_minimax_h3_', 'MODE': 'ref2va_vfx_edit',
             '_resolve_shotplan': lambda linx: linx['plan'],
             '_as_dict': lambda value: value if isinstance(value, dict) else {},
             '_h3_plan': lambda linx: linx['plan'], '_has_h3_plan': lambda _: True,
             '_recompile_routed_h3_plan': lambda linx: (linx, 'shotboard'),
             '_normalise_audio_mode': lambda value: value, '_clean_previous_h3_info': lambda linx: linx,
             '_shape': lambda image: list(image.shape) if torch.is_tensor(image) else [],
             '_audio_meta': lambda _: {}, 'build_stage_linx_payload': lambda base, **kwargs: {'resources': kwargs['resources']},
             'IMAGE_ROLES': [], 'VIDEO_ROLES': [], 'AUDIO_ROLES': [], 'TASK_OVERRIDES': [], **extra}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), file, 'exec'), scope)
    return scope[class_name]()


class PreparedModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_modules = {name: sys.modules.get(name) for name in (
            'r49_mode_test', 'r49_mode_test.iamccs_minimax_h3_atomic_backend',
            'r49_mode_test.iamccs_minimax_h3_v2v_backend')}
        package = types.ModuleType('r49_mode_test')
        package.__path__ = []
        backend = types.ModuleType('r49_mode_test.iamccs_minimax_h3_atomic_backend')
        backend._resolve_shotplan = lambda linx: linx['plan']
        sys.modules[package.__name__] = package
        sys.modules[backend.__name__] = backend
        helpers = types.ModuleType('r49_mode_test.iamccs_minimax_h3_v2v_backend')
        helpers.__dict__.update(v2v_helpers())
        sys.modules[helpers.__name__] = helpers

    @classmethod
    def tearDownClass(cls):
        for name, value in cls.old_modules.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value

    def linx(self, mode):
        return {'plan': {'task_mode': mode, 'audio_mode': 'h3_native_generated',
                         'reference_roles': ['subject_identity']*3 + ['disabled'],
                         'reference_video_role': 'off', 'reference_audio_role': 'off'},
                'resources': {'iamccs_minimax_h3_settings': {'h3_controlnet_enabled': True}}}

    def test_only_requested_reference_loaders_are_evaluated(self):
        node = node_scope('iamccs_minimax_h3_cine_info.py', 'IAMCCS_CineInfoH3')
        kwargs = {f'reference_image_{i}': None for i in range(1, 5)}
        kwargs.update(reference_video=None, reference_video_audio=None)
        for mode in ('i2va', 'fl2va', 'longvid_multishot_hardcut', 'v2va_controlnet'):
            self.assertEqual(node.check_lazy_status(self.linx(mode), active_mode_only=True, **kwargs), [])
        for mode in ('ref2va', 'ref2vid_lipsync'):
            self.assertEqual(node.check_lazy_status(self.linx(mode), active_mode_only=True, **kwargs),
                             ['reference_image_1', 'reference_image_2', 'reference_image_3'])
            linx = self.linx(mode)
            linx['plan']['reference_video_role'] = 'motion_camera'
            self.assertIn('reference_video', node.check_lazy_status(linx, active_mode_only=True, **kwargs))

    def test_settings_roles_replace_stale_cineinfo_widget_roles(self):
        node = node_scope('iamccs_minimax_h3_cine_info.py', 'IAMCCS_CineInfoH3')
        picture = torch.zeros(1, 2, 2, 3)
        result = node.attach(self.linx('ref2va'), 'from_shotboard',
                            'subject_identity', 'disabled', 'composition', 'style', 'off', 'off',
                            'match', 'from_shotboard', 0.5, 'area',
                            reference_image_1=picture, reference_image_2=picture,
                            reference_image_3=picture, reference_image_4=picture, active_mode_only=True)[0]['resources']
        self.assertEqual(result['iamccs_minimax_h3_cine_info']['reference_roles'], ['subject_identity']*3 + ['disabled'])
        self.assertIsNone(result['iamccs_minimax_h3_ref_image_4'])
        self.assertIs(result['iamccs_minimax_h3_ref_image_3'], picture)

    def test_control_video_is_not_loaded_by_other_modes_even_with_stale_enable_flag(self):
        node = node_scope('iamccs_cine_h3_bus.py', 'IAMCCS_CineH3FunControlInput')
        for mode in ('i2va', 'ref2va', 'ref2vid_lipsync', 'longvid_multishot_hardcut', 'v2va_face_swap'):
            linx = self.linx(mode)
            self.assertEqual(node.check_lazy_status(linx, active_mode_only=True, source_video=None), [])
            self.assertIs(node.inject(linx, 24, active_mode_only=True)[0], linx)
        self.assertEqual(node.check_lazy_status(self.linx('v2va_controlnet'), active_mode_only=True, source_video=None), ['source_video'])
        self.assertTrue(node._enabled(self.linx('i2va')), 'Legacy control transport remains available')

    def test_vfx_upload_is_lazy_and_respects_prompter_source_choice(self):
        node = node_scope('iamccs_h3_vfx_edit.py', 'IAMCCS_H3VFXInput')
        for mode in ('i2va', 'ref2va', 'v2va_face_swap', 'longvid_multishot_hardcut'):
            linx = self.linx(mode)
            self.assertEqual(node.check_lazy_status(linx, source_video=None), [])
            self.assertIs(node.attach(linx, '')[0], linx)
        linx = self.linx('ref2va_vfx_edit')
        self.assertEqual(node.check_lazy_status(linx, source_video=None, source_audio=None), ['source_video', 'source_audio'])
        linx['resources']['iamccs_h3_edit_video'] = {'path': 'selected_in_prompter.mp4'}
        self.assertEqual(node.check_lazy_status(linx, source_video=None), [])

    def test_vfx_uploaded_video_and_audio_share_the_trimmed_clock(self):
        node = node_scope('iamccs_h3_vfx_edit.py', 'IAMCCS_H3VFXInput', RESOURCE='vfx_source')
        linx = self.linx('ref2va_vfx_edit')
        linx['plan']['effective_duration_seconds'] = 1
        linx['plan']['vfx_edit'] = {'video_source': 'backend'}
        frames = torch.arange(90).float().reshape(90, 1, 1, 1).expand(-1, 2, 2, 3)
        audio = {'sample_rate': 24000, 'waveform': torch.arange(72000).reshape(1, 1, -1).float()}
        source = node.attach(linx, '', start_seconds=0.5, source_video=frames,
                             source_audio=audio, source_fps=30)[0]['resources']['vfx_source']
        self.assertEqual(len(source['video']), 24)
        self.assertEqual(source['video'][0, 0, 0, 0].item(), 15)
        self.assertEqual(source['audio']['waveform'].shape[-1], 24000)
        self.assertEqual(source['audio']['waveform'][0, 0, 0].item(), 12000)
        self.assertEqual(source['fps'], 24)
        class SilentVideoAudio(Mapping):
            def __getitem__(self, key):
                raise ValueError('Uploaded video has no audio stream')
            def __iter__(self):
                return iter(('waveform', 'sample_rate'))
            def __len__(self):
                return 2
        silent = node.attach(linx, '', source_video=frames, source_audio=SilentVideoAudio(),
                             source_fps=30)[0]['resources']['vfx_source']
        self.assertIsNone(silent['audio'])

    def test_object_swap_loads_only_the_chosen_raw_guides(self):
        helpers = v2v_helpers()
        node = node_scope('iamccs_minimax_h3_v2v_backend.py', 'IAMCCS_CineInfoH3V2V',
                          _resolve_v2v_config=helpers['_resolve_v2v_config'],
                          AUDIO_BEHAVIOR_NATIVE='h3_native_generated')
        media = {name: None for name in ('source_video', 'source_audio', 'depth_guide', 'pose_guide',
                                         *(f'reference_image_{i}' for i in range(1, 5)))}
        for mode in ('i2va', 'longvid_multishot_hardcut', 'v2va_controlnet', 'ref2va'):
            self.assertEqual(node.check_lazy_status(self.linx(mode), active_mode_only=True, **media), [])
        linx = self.linx('v2va_object_swap')
        for guide, wanted in [('raw_only', []), ('raw_depth', ['depth_guide']),
                              ('raw_pose', ['pose_guide']), ('raw_depth_pose', ['depth_guide', 'pose_guide'])]:
            linx['plan']['v2v_settings'] = {'guide_mode': guide}
            requested = node.check_lazy_status(linx, active_mode_only=True, **media)
            self.assertEqual(requested, ['source_video', 'reference_image_1', 'reference_image_2', 'reference_image_3', *wanted])

    def test_multishot_balances_windows_without_moving_cuts_or_audio_clock(self):
        for seconds in (15, 15.1, 15.5, 16, 20, 29, 30, 35, 900):
            total = round(seconds*24)
            rows = [dict(id=str(i), type='image', start=start, length=min(83, total-start),
                         imageFile=f'picture_{i}.png', prompt=f'Performance {i}')
                    for i, start in enumerate(range(0, total, 83))]
            plan = CORE.build_shotplan(timeline_data={'rows': rows, 'fps': 24}, global_prompt='Film',
                                      duration_seconds=seconds, task_mode='longvid_multishot_hardcut', width=640, height=384)
            self.assertEqual(sum(c['visible_frame_count'] for c in plan['chunks']), total)
            cursor = 0
            for chunk in plan['chunks']:
                self.assertEqual(chunk['timeline_start_frame'], cursor)
                self.assertLessEqual(chunk['frame_count'], 362)
                self.assertGreaterEqual(chunk['frame_count'], 124)
                self.assertEqual(chunk['frame_count'] % 17, 5)
                cursor += chunk['visible_frame_count']
            actual_guides = [g['global_frame'] for c in plan['chunks'] for g in c['guides']]
            self.assertEqual(actual_guides, [r['start'] for r in rows])
            if seconds == 16:
                self.assertEqual([c['visible_frame_count'] for c in plan['chunks']], [192, 192])
                self.assertTrue(all(c['multishot_generated_runway_frames'] >= 17 for c in plan['chunks']))


if __name__ == '__main__':
    unittest.main()
