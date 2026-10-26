import ast
import importlib.util
import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('shotboard_core', ROOT / 'iamccs_minimax_h3_shotboard_core.py')
CORE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CORE)


def arguments(lengths):
    rows, start = [], 0
    for index, length in enumerate(lengths):
        rows.append(dict(id=str(index), type='image', start=start, length=length,
                         imageFile=f'camera_{index}.png', prompt=f'Camera {index} performance.'))
        start += length
    return dict(timeline_data={'fps': 24, 'rows': rows}, global_prompt='Global camera montage.',
                duration_seconds=start / 24, task_mode='longvid_multishot_hardcut')


class MultiShotBoundaryTests(unittest.TestCase):
    def test_legacy_joint_plan_is_unchanged(self):
        args = arguments([106, 86, 67, 97])
        self.assertEqual(CORE.build_shotplan(**args), CORE.build_shotplan(**args, multishot_boundary_mode='joint'))

    def test_authored_cuts_duration_and_own_image_only(self):
        plan = CORE.build_shotplan(**arguments([106, 86, 67, 97]), multishot_boundary_mode='isolated_shots')
        self.assertEqual([c['timeline_start_frame'] for c in plan['chunks']], [0, 106, 192, 259])
        self.assertEqual([c['unique_frames'] for c in plan['chunks']], [106, 86, 67, 97])
        self.assertEqual(sum(c['unique_frames'] for c in plan['chunks']), plan['total_unique_frames'])
        for index, chunk in enumerate(plan['chunks']):
            self.assertEqual([v['source_path'] for v in chunk['multishot_visual_slots']], [f'camera_{index}.png'])
            self.assertIn(f'Camera {index} performance.', chunk['prompt'])
            self.assertNotIn('Global camera montage', chunk['prompt'])
            self.assertEqual(chunk['latent_tail_context_prefix_frames'], 0)
            self.assertTrue(chunk['isolated_shot_start'])
            self.assertGreaterEqual(chunk['frame_count'] - chunk['unique_frames'], 17)
            self.assertEqual(chunk['guides'][0]['local_frame'], 0)
            self.assertEqual(chunk['guides'][0]['global_frame'], chunk['timeline_start_frame'])
            self.assertEqual(chunk['multishot_visual_slots'][0]['global_frame'], chunk['timeline_start_frame'])

    def test_long_shots_continue_only_inside_the_same_camera(self):
        plan = CORE.build_shotplan(**arguments([800, 86, 400]), multishot_boundary_mode='isolated_shots')
        cursor = 0
        for index, chunk in enumerate(plan['chunks']):
            self.assertEqual(chunk['timeline_start_frame'], cursor)
            cursor += chunk['unique_frames']
            self.assertEqual(chunk['frame_count'] % 17, 5)
            self.assertLessEqual(chunk['frame_count'], 362)
            self.assertEqual(chunk['latent_tail_context_prefix_frames'], 0 if chunk['isolated_shot_start'] else 22)
            self.assertEqual(len(chunk['multishot_visual_slots']), 1)
            self.assertEqual(chunk['multishot_visual_slots'][0]['global_frame'], [0, 800, 886][chunk['isolated_shot_index']])
            if index == len(plan['chunks']) - 1 or plan['chunks'][index + 1]['isolated_shot_start']:
                self.assertGreaterEqual(chunk['frame_count'] - chunk['unique_frames'] - chunk['latent_tail_context_prefix_frames'], 17)
        self.assertEqual(cursor, 1286)
        self.assertEqual([c['timeline_start_frame'] for c in plan['chunks'] if c['isolated_shot_start']], [0, 800, 886])

    def test_backend_never_reads_previous_camera_tail(self):
        tree = ast.parse((ROOT / 'iamccs_minimax_h3_atomic_backend.py').read_text(encoding='utf-8'))
        expression = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'latent_tail_active' for t in n.targets))
        code = compile(ast.Expression(expression), '<backend-tail-condition>', 'eval')
        for index, start, enabled, expected in [(0, True, True, False), (3, True, True, False),
                                               (4, False, True, True), (4, False, False, False)]:
            self.assertEqual(eval(code, dict(chunk={'isolated_shot_start': start}, segment_index=index,
                                              latent_tail_requested=enabled)), expected)
        self.assertTrue(eval(code, dict(chunk={}, segment_index=1, latent_tail_requested=True)))

    def test_native_audio_and_locked_audio_contracts_are_explicit(self):
        for mode, locked in [('h3_native_generated', False), ('h3_custom_audio_drive', True)]:
            plan = CORE.build_shotplan(**arguments([106, 86]), audio_mode=mode, multishot_boundary_mode='isolated_shots')
            self.assertEqual(plan['mode_contract']['audio_timeline_continuous'], locked)
            self.assertEqual(plan['mode_contract']['decoded_frame_replacement'], False)
            self.assertEqual('supplied soundtrack' in plan['chunks'][0]['prompt'], locked)

    def test_delivery_discards_runway_without_freezing_or_dissolving_frames(self):
        tree = ast.parse((ROOT / 'iamccs_minimax_h3_atomic_backend.py').read_text(encoding='utf-8'))
        block = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                     and ast.unparse(n.test).startswith("str(shotplan.get('task_mode', '') or '').strip().lower() == 'longvid_multishot_hardcut'"))
        code = compile(ast.Module(body=[block], type_ignores=[]), '<actual-multishot-delivery>', 'exec')
        chunks = CORE.build_shotplan(**arguments([106, 86, 67, 97]), multishot_boundary_mode='isolated_shots')['chunks']
        for index, chunk in enumerate(chunks):
            frames = torch.arange(chunk['frame_count']).reshape(-1, 1, 1, 1)
            wave = torch.arange(chunk['frame_count'] * 2000).reshape(1, 1, -1)
            scope = dict(torch=torch, LOG=logging.getLogger(__name__), H3_FPS=24, chunk=chunk, chunk_index=index,
                         shotplan={'task_mode': 'longvid_multishot_hardcut', 'chunks': chunks},
                         native_frames=frames, native_audio={'sample_rate': 48000, 'waveform': wave})
            exec(code, scope)
            torch.testing.assert_close(scope['native_frames'], frames[:chunk['unique_frames']])
            torch.testing.assert_close(scope['native_audio']['waveform'], wave[..., :chunk['unique_frames'] * 2000])

    def test_custom_audio_master_keeps_every_sample_across_camera_cuts(self):
        tree = ast.parse((ROOT / 'iamccs_minimax_h3_shotboard.py').read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_build_locked_audio_wave_master')
        scope = dict(np=np, Path=Path, H3_FPS=24)
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<locked-audio-master>', 'exec'), scope)
        counts = [106, 86, 67, 97]
        master = np.arange(sum(counts) * 2000, dtype=np.float32)[None, :]
        with tempfile.TemporaryDirectory() as temp:
            paths, offset = [], 0
            for index, count in enumerate(counts):
                path = Path(temp) / f'{index}.mp4'
                samples = count * 2000
                np.savez(path.with_suffix('.mp4.locked_audio.npz'), waveform=master[:, offset:offset + samples], sample_rate=48000)
                paths.append(path)
                offset += samples
            result, rate, layout = scope['_build_locked_audio_wave_master'](paths, counts)
        np.testing.assert_array_equal(result, master)
        self.assertEqual((rate, layout), (48000, 'mono'))


if __name__ == '__main__':
    unittest.main()
