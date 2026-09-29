import ast
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORE = load('iamccs_minimax_h3_shotboard_core')
DRIVE = load('iamccs_minimax_h3_audio_drive')
TIMELINE = load('iamccs_minimax_h3_audio_timeline')


def plan(mode='t2va_continuous', duration=30, rows=None, **kwargs):
    return CORE.build_shotplan(
        timeline_data={'rows': [r for r in (rows or []) if r.get('type') != 'audio'],
                       'audioSegments': [r for r in (rows or []) if r.get('type') == 'audio'], 'fps': 24},
        global_prompt='Keep walking through the corridor.', duration_seconds=duration,
        task_mode=mode, width=640, height=384, **kwargs)


class B1bisModes(unittest.TestCase):
    def test_short_text_continuous_is_one_native_t2va_sample(self):
        result = plan(duration=10)
        self.assertEqual(len(result['chunks']), 1)
        self.assertEqual(result['chunks'][0]['task_mode'], 't2va')
        self.assertNotIn('extended_av', result)

    def test_long_text_continuous_reuses_b1_geometry_and_local_direction(self):
        rows = [{'type': 'text', 'start': 0, 'length': 720, 'prompt': 'Wind moves the coat.'}]
        result = plan(rows=rows)
        baseline = plan(mode='fl2va_extended_av')
        for chunk, original in zip(result['chunks'], baseline['chunks']):
            for key in ('frame_count', 'visible_frame_count', 'raw_timeline_start_frame', 'extended_av_context_prefix_frames'):
                self.assertEqual(chunk[key], original[key])
            self.assertIn('Wind moves the coat.', chunk['prompt'])
        self.assertEqual(sum(c['visible_frame_count'] for c in result['chunks']), 720)

    def test_text_mode_does_not_inherit_image_guides(self):
        rows = [{'type': 'image', 'imageFile': 'old.png', 'start': 0, 'length': 720}]
        result = plan(rows=rows)
        self.assertFalse(any(g['kind'] == 'image' for c in result['chunks'] for g in c['guides']))

    def test_multiple_text_slots_remain_hard_cuts(self):
        rows = [{'id': str(i), 'type': 'text', 'start': i * 120, 'length': 120,
                 'prompt': f'Independent shot {i}', 'transition': 'continuous'} for i in range(2)]
        for mode in ('t2va', 't2va_continuous'):
            result = plan(mode=mode, duration=10, rows=rows)
            self.assertEqual(len(result['chunks']), 2)
            self.assertEqual(result['chunks'][1]['transition'], 'hard_cut')
            self.assertEqual(result['chunks'][1]['overlap_frames'], 0)

    def test_latent_new_uses_b1_not_go_ahead(self):
        rows = [{'id': str(i), 'type': 'image', 'imageFile': f'{i}.png',
                 'start': i * 240, 'length': 240, 'prompt': f'Guide action {i}'} for i in range(3)]
        result = plan(mode='keyframe_joint_native', rows=rows, keyframe_joint_latent_new=True)
        self.assertEqual(result['task_mode'], 'fl2va_extended_av')
        self.assertTrue(result['mode_contract']['experimental'])
        self.assertIn('Guide action 2', result['chunks'][-1]['prompt'])
        self.assertEqual(sum(c['visible_frame_count'] for c in result['chunks']), 720)


class B1bisAudio(unittest.TestCase):
    def test_locked_flf_does_not_insert_silence_handles(self):
        rows = [{'id': str(i), 'type': 'image', 'imageFile': f'{i}.png',
                 'start': i * 120, 'length': 120} for i in range(3)]
        result = plan(mode='fl2va', duration=15, rows=rows, audio_mode='h3_custom_audio_drive')
        self.assertEqual(result['audio_handoff_policy']['speech_free_head_seconds_after_first'], 0)
        self.assertEqual(result['audio_handoff_policy']['speech_free_tail_seconds'], 0)
        for chunk in result['chunks']:
            self.assertEqual(chunk['audio_handoff_prompt'], '')
            self.assertEqual(chunk['audio_handoff_silence_head_seconds'], 0)
            self.assertEqual(chunk['audio_handoff_silence_tail_seconds'], 0)

    def locked_plan(self):
        return plan(mode='fl2va_extended_av', audio_mode='h3_custom_audio_drive', rows=[
            {'type': 'audio', 'audioFile': 'master.wav', 'start': 0, 'length': 720}])

    def test_locked_plan_and_both_audio_slicers_use_raw_start(self):
        result = self.locked_plan()
        self.assertTrue(result['extended_av']['video_only'])
        for chunk in result['chunks']:
            expected = chunk['raw_timeline_start_frame'] / 24
            self.assertEqual(DRIVE._chunk_timing(result, chunk)[0], expected)
            self.assertEqual(TIMELINE._chunk_interval(result, chunk)[0], expected)

    def test_delivery_removes_hidden_audio_exactly_once(self):
        result = self.locked_plan()
        chunk = result['chunks'][1]
        rate = 24000
        start = chunk['raw_timeline_start_frame'] * 1000
        raw = torch.arange(start, start + chunk['frame_count'] * 1000, dtype=torch.float32).view(1, 1, -1)
        delivered, mode, report = DRIVE.IAMCCS_MiniMaxH3AudioOutputPolicyR21().select(
            h3_audio=None, video_frames=torch.zeros(chunk['visible_frame_count'], 1, 1, 3),
            cine_linx=result, segment_index=1, locked_audio_slice={'waveform': raw, 'sample_rate': rate})
        expected_start = chunk['timeline_start_frame'] * 1000
        self.assertEqual(delivered['waveform'][0, 0, 0].item(), expected_start)
        self.assertEqual(delivered['waveform'].shape[-1], chunk['visible_frame_count'] * 1000)

    def test_master_does_not_double_hidden_context_or_deliver_padding(self):
        result = self.locked_plan()
        audio = {'sample_rate': 24000, 'waveform': torch.linspace(-0.25, 0.25, 720000).view(1, 1, -1)}
        linx = {'resources': {'iamccs_minimax_h3_shotplan': result, 'cine_audio_timeline_json': {
            'fps': 24, 'audioSegments': [{'id': 'master', 'audio_input': 1, 'start_seconds': 0,
                                        'duration_seconds': 30, 'gain': 1.0}]}}}
        _, master, selected, _ = TIMELINE.mix_audio_timeline(
            linx, 1, [audio], target_sample_rate=24000, headroom='none')
        self.assertEqual(master['waveform'].shape, audio['waveform'].shape)
        torch.testing.assert_close(master['waveform'], audio['waveform'], rtol=0, atol=0)
        start = result['chunks'][1]['raw_timeline_start_frame'] * 1000
        self.assertEqual(selected['waveform'][0, 0, 0], audio['waveform'][0, 0, start])

    def test_final_pcm_assembly_is_sample_exact(self):
        tree = ast.parse((ROOT / 'iamccs_minimax_h3_shotboard.py').read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_build_locked_audio_wave_master')
        ns = {'Path': Path, 'np': np, 'H3_FPS': 24}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<locked-master>', 'exec'), ns)
        with tempfile.TemporaryDirectory() as td:
            paths = [Path(td) / f'{i}.mp4' for i in range(2)]
            parts = [np.full((2, 24000), v, dtype=np.float32) for v in (0.1, -0.2)]
            for path, wave in zip(paths, parts):
                np.savez(path.with_suffix('.mp4.locked_audio.npz'), waveform=wave, sample_rate=24000)
            actual, rate, layout = ns['_build_locked_audio_wave_master'](paths, [24, 24])
            np.testing.assert_array_equal(actual, np.concatenate(parts, axis=1))
            self.assertEqual((rate, layout), (24000, 'stereo'))


if __name__ == '__main__':
    unittest.main()
