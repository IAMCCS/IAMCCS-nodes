import ast
import copy
import importlib.util
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).parents[1]
PACKAGE = 'omni_test_package'
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package


def load(name):
    spec = importlib.util.spec_from_file_location(PACKAGE + '.' + name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OMNI = load('iamccs_h3_omni')
CORE = load('iamccs_minimax_h3_shotboard_core')


def plan(mode='ref2va'):
    result = CORE.build_shotplan(timeline_data={'fps': 24, 'rows': [
        dict(id='a', type='text', start=0, length=124, prompt='The knight looks left.'),
        dict(id='b', type='text', start=124, length=119, prompt='A reverse camera angle.')
    ]}, global_prompt='A two-camera scene.', duration_seconds=243/24, task_mode=mode)
    result['sampling'] = dict(steps=8, sampler_name='euler', scheduler='simple', denoise=1., shift_video=12., shift_audio=3.)
    return result


class OmniTests(unittest.TestCase):
    def test_native_reference_limit_is_deferred_only_for_window_planner(self):
        kwargs = dict(timeline_data={'rows': []}, global_prompt='A continuous forest scene.',
                      duration_seconds=16, task_mode='ref2va')
        with self.assertRaises(ValueError):
            CORE.build_shotplan(**kwargs)
        p = CORE.build_shotplan(**kwargs, defer_reference_windows=True)
        p['sampling'] = plan()['sampling']
        OMNI.configure(p, {'h3_omni_enabled': True, 'motion_context_window_frames': 209})
        self.assertEqual([c['unique_frames'] for c in p['chunks']], [209, 175])

    def test_longvid_covers_exact_timeline_with_hidden_av_context(self):
        for total in (363, 480, 1201):
            p = plan()
            p['requested_duration_seconds'] = total / 24
            OMNI.configure(p, {'h3_omni_enabled': True, 'motion_context_window_frames': 209})
            self.assertEqual(sum(c['unique_frames'] for c in p['chunks']), total)
            cursor = 0
            for i, c in enumerate(p['chunks']):
                prefix = 22 if i else 0
                self.assertEqual(c['timeline_start_frame'], cursor)
                self.assertEqual(c['latent_tail_context_prefix_frames'], prefix)
                self.assertGreaterEqual(c['frame_count'], prefix + c['unique_frames'])
                self.assertLessEqual(c['frame_count'], 209)
                self.assertEqual(c['frame_count'] % 17, 5)
                cursor += c['unique_frames']
            self.assertTrue(p['longvid_latent_tail']['enabled'])
            self.assertFalse(p['fasth3_omni']['joint_target_latent'])
            self.assertTrue(p['fasth3_omni']['joint_latent_per_window'])

    def test_multiple_camera_slots_keep_one_target_latent(self):
        p = plan()
        self.assertGreater(len(p['chunks']), 1)
        OMNI.configure(p, {'h3_omni_enabled': True})
        self.assertEqual(len(p['chunks']), 1)
        self.assertEqual(p['chunks'][0]['frame_count'], 243)
        self.assertIn('knight looks left', p['chunks'][0]['prompt'])
        self.assertIn('reverse camera angle', p['chunks'][0]['prompt'])
        self.assertTrue(OMNI.enabled(p))

    def test_disabled_profile_preserves_legacy_chunks(self):
        p = plan()
        old = copy.deepcopy(p)
        OMNI.configure(p, {})
        p.pop('fasth3_omni')
        self.assertEqual(p, old)

    def test_remembered_profile_never_hijacks_longvid_joint(self):
        p = CORE.build_shotplan(timeline_data={'fps': 24, 'rows': [dict(id='a', type='image', start=0, length=124, imageFile='camera.png')]},
                                global_prompt='A scene.', duration_seconds=124/24, task_mode='longvid_multishot_hardcut')
        old = copy.deepcopy(p['chunks'])
        OMNI.configure(p, {'h3_omni_enabled': True})
        self.assertFalse(OMNI.enabled(p))
        self.assertEqual(p['chunks'], old)

    def test_wrong_schedule_or_stacked_speed_adapter_is_rejected(self):
        for key, value in [('steps', 4), ('sampler_name', 'res_multistep'), ('shift_video', 10)]:
            p = plan()
            p['sampling'][key] = value
            with self.assertRaises(ValueError):
                OMNI.configure(p, {'h3_omni_enabled': True})
        for name in ['turbo', 'pdd', 'fasth3', 'fused_turbo_preview', 'veda']:
            p = plan()
            p[name] = {'enabled': True}
            with self.assertRaises(ValueError):
                OMNI.configure(p, {'h3_omni_enabled': True})

    def test_router_loads_only_the_selected_model(self):
        tree = ast.parse((ROOT / 'iamccs_minimax_h3_atomic_backend.py').read_text('utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'IAMCCS_MiniMaxH3AtomicModelRouter')
        scope = dict(omni=OMNI, _resolve_shotplan=lambda x: x, _external_continuation_active=lambda *x: False,
                     _is_fused_turbo_preview=lambda p: False, _chunk=lambda *x: {}, _effective_task=lambda *x: 'ref2va',
                     _task_family=lambda t: t, CATEGORY='test')
        exec(compile(ast.Module(body=[cls], type_ignores=[]), '<real-model-router>', 'exec'), scope)
        router = scope[cls.name]()
        p = {'fasth3_omni': {'enabled': True}}
        self.assertEqual(router.check_lazy_status(p, 0), ['omni_model'])
        self.assertEqual(router.check_lazy_status(p, 0, omni_model=object()), [])
        p['fasth3_omni']['enabled'] = False
        self.assertEqual(router.check_lazy_status(p, 0), ['ref2va_model'])

    def test_audio_tags_match_native_video_soundtrack_order(self):
        manifest = OMNI.reference_manifest({'image_1': object()}, {'ref_video_1': object(), 'ref_video_2': object()},
                                           {'ref_video_audio_2': object()}, {'ref_audio_1': object(), 'ref_audio_2': object()})
        self.assertEqual([v['label'] for v in manifest], ['<Picture 1>', '<Video 1>', '<Audio 1>', '<Video 2>', '<Audio 2>', '<Audio 3>'])


if __name__ == '__main__':
    unittest.main()
