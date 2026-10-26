import ast
import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORE = load('iamccs_minimax_h3_shotboard_core')
VEDA = load('iamccs_h3_veda')


class MultishotResolutionTests(unittest.TestCase):
    def test_compiler_owns_shot_numbers_and_preserves_spoken_words(self):
        speech = '<d>[English] Did you mean [Shot 1], or another shot?</d>'
        rows = [dict(id=str(i), type='image', start=i*80, length=80,
                     imageFile=f'picture_{i}.png', prompt='integrated_multimodal_description:\n[Shot 1] Static camera. ' + speech)
                for i in range(4)]
        for width, height in [(640, 384), (864, 480), (1280, 768)]:
            plan = CORE.build_shotplan(timeline_data={'rows': rows, 'fps': 24},
                                      global_prompt='[Shot 1] Shared scene direction.',
                                      duration_seconds=320/24, task_mode='longvid_multishot_hardcut',
                                      width=width, height=height)
            prompt = plan['chunks'][0]['prompt']
            self.assertEqual(prompt.count(speech), 4)
            without_dialogue = prompt.replace(speech, '')
            for index in range(1, 5):
                self.assertEqual(without_dialogue.count(f'[Shot {index}]'), 1)
            self.assertEqual([g['global_frame'] for g in plan['chunks'][0]['guides']], [0, 80, 160, 240])

    def test_visible_speaker_binding_follows_authored_slots_and_subject_ids(self):
        subjects = [7, 2, 7, 11, 2]
        rows = [dict(id=str(i), type='image', start=i*64, length=64,
                     imageFile=f'picture_{i}.png',
                     prompt=f'<Subject {subject}>: <d>[English] Line {i}.</d>\nOnly <Subject {subject}> (S{subject}) speaks and moves their lips.')
                for i, subject in enumerate(subjects)]
        plan = CORE.build_shotplan(timeline_data={'rows': rows, 'fps': 24}, global_prompt='', duration_seconds=320/24,
                                  task_mode='longvid_multishot_hardcut', width=864, height=480)
        prompt = plan['chunks'][0]['prompt']
        for i, subject in enumerate(subjects, 1):
            self.assertIn(f'<Subject {subject}> is the visible speaking performer in <Picture {i}>.', prompt)
        self.assertEqual([g['global_frame'] for g in plan['chunks'][0]['guides']], [0, 64, 128, 192, 256])

    def test_no_inferred_binding_for_offscreen_or_ambiguous_speakers(self):
        visible = 'Only <Subject 7> (S3) speaks and moves their lips.'
        prompts = [visible + ' Off-screen narration.', 'A silent reaction shot.',
                   visible + ' Only <Subject 9> (S4) speaks and moves their lips.',
                   visible + ' Voiceover performance.']
        rows = [dict(id=str(i), type='image', start=i*80, length=80,
                     imageFile=f'picture_{i}.png', prompt=prompt)
                for i, prompt in enumerate(prompts)]
        plan = CORE.build_shotplan(timeline_data={'rows': rows, 'fps': 24}, global_prompt='', duration_seconds=320/24,
                                  task_mode='longvid_multishot_hardcut', width=864, height=480)
        self.assertNotIn('is the visible speaking performer', plan['chunks'][0]['prompt'])

    def test_shot_normalization_preserves_inline_references(self):
        text = '[Shot 1] Static camera. Compare the composition with [Shot 2].'
        self.assertEqual(CORE._multishot_prompt_sections(text)[0],
                         'Static camera. Compare the composition with [Shot 2].')

    def test_h3_rope_guide_times_do_not_depend_on_spatial_resolution(self):
        model_path = ROOT.parents[1] / 'comfy/ldm/minimax/model.py'
        tree = ast.parse(model_path.read_text(encoding='utf-8'))
        names = {'_axis_from_sqrt_area', '_frame_grid', '_video_t_spans', '_video_t_grid',
                 '_ref_t_span', '_audio_grid', '_video_grid', 'PackedLayout'}
        scope = {'torch': torch, 'math': math, 'FRAME_RESCALE': 5/3, 'FRAME_PER_TOKEN': (1, 4, 4, 4, 4)}
        exec(compile(ast.Module(body=[n for n in tree.body if getattr(n, 'name', '') in names],
                                type_ignores=[]), str(model_path), 'exec'), scope)
        for width, height in [(640, 384), (864, 480), (1280, 768)]:
            guides = [dict(resolved_frame_index=f, latent=torch.empty(1, 24, 1, height//16, width//16))
                      for f in (0, 106, 192, 259)]
            layout = scope['PackedLayout'](1000, 107, height//16, width//16, 603, keyframes=guides)
            origin = next(float(layout.position_ids[a, 0]) for a, b, kind in layout.segments if kind == 'video')
            actual = [round(float(layout.position_ids[a, 0]-origin)/(5/3), 6)
                      for a, b, kind in layout.segments if kind == 'cond']
            self.assertEqual(actual, [0, 106, 192, 259])

    def veda_call(self, mode):
        calls = []
        class FakeVeda:
            @staticmethod
            def execute(**kwargs):
                calls.append(kwargs)
                return (kwargs['model'],)
        fake_nodes = types.SimpleNamespace(NODE_CLASS_MAPPINGS={'VedaSparseAttention': FakeVeda})
        model = object()
        with patch.dict(sys.modules, nodes=fake_nodes):
            result = VEDA.apply_veda(model, {'task_mode': mode, 'veda': {'enabled': True}}, 't2va')
        self.assertIs(result, model)
        return calls[0]

    def test_joint_multishot_keeps_full_guide_attention_and_dense_first_step(self):
        call = self.veda_call('longvid_multishot_hardcut')
        self.assertEqual(call['reference_sparsity'], '0%')
        self.assertEqual(call['full_attention_steps'], '0')
        self.assertNotIn('generated_sparsity', call)

    def test_other_modes_keep_their_existing_veda_options(self):
        for mode in ('t2va', 'i2va', 'fl2va', 'longvid_keyframe_continuous', 'ref2va'):
            call = self.veda_call(mode)
            self.assertNotIn('reference_sparsity', call)
            self.assertNotIn('full_attention_steps', call)

    def test_disabled_veda_remains_disabled(self):
        model = object()
        self.assertIs(VEDA.apply_veda(model, {'task_mode': 'longvid_multishot_hardcut', 'veda': {'enabled': False}}, 't2va'), model)


if __name__ == '__main__':
    unittest.main()
