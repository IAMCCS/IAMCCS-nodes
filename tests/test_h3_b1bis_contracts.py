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
    def multishot_plan(self, audio_segments):
        lengths = [81, 92, 106, 81]
        starts = [0, 81, 173, 279]
        rows = [
            {'id': f'shot_{index}', 'type': 'image', 'imageFile': f'{index}.png',
             'start': start, 'length': length, 'duration_frames': length,
             'transition': 'start' if index == 0 else 'hard_cut'}
            for index, (start, length) in enumerate(zip(starts, lengths))
        ]
        return CORE.build_shotplan(
            timeline_data={'rows': rows, 'audioSegments': audio_segments, 'fps': 24, 'duration_seconds': 15},
            global_prompt='Four independent camera angles.', duration_seconds=15,
            task_mode='longvid_multishot_hardcut', audio_mode='h3_custom_audio_drive',
            width=640, height=384,
        )

    def test_multishot_unlinked_master_follows_every_camera_angle(self):
        segments = [
            {'id': 'master', 'audio_input': 1, 'start_seconds': 0, 'duration_seconds': 15},
        ]
        result = self.multishot_plan(segments)
        rate = 24000
        audio = {'sample_rate': rate, 'waveform': torch.linspace(-0.5, 0.5, 15 * rate).view(1, 1, -1)}
        linx = {'resources': {'iamccs_minimax_h3_shotplan': result,
                              'cine_audio_timeline_json': {'fps': 24, 'audioSegments': segments}}}
        _, master, selected, report = TIMELINE.mix_audio_timeline(
            linx, 0, [audio], target_sample_rate=rate, headroom='none')
        self.assertEqual(report['mapping_policy'], 'absolute_timeline')
        self.assertEqual([c['timeline_start_frame'] for c in result['chunks']], [0])
        self.assertEqual(master['waveform'].shape[-1], 15 * rate)
        torch.testing.assert_close(master['waveform'], audio['waveform'], rtol=0, atol=0)
        self.assertEqual(selected['waveform'][0, 0, 0], audio['waveform'][0, 0, 0])

    def test_multishot_single_audio_socket_is_a_global_master(self):
        result = self.multishot_plan([])
        audio = {'sample_rate': 24000, 'waveform': torch.full((1, 1, 360000), 0.2)}
        _, master, _, report = TIMELINE.mix_audio_timeline(
            result, 0, [audio], target_sample_rate=24000, headroom='none')
        torch.testing.assert_close(master['waveform'], audio['waveform'], rtol=0, atol=0)
        self.assertTrue(report['fallback_one_input_per_chunk'])

    def test_multishot_linked_dialogue_does_not_bleed_into_other_angles(self):
        lengths = [81, 92, 106, 81]
        starts = [0, 81, 173, 279]
        segments = [
            {'id': f'line_{index}', 'audio_input': 1 if index % 2 == 0 else 2,
             'linkedVisualId': f'shot_{index}', 'start': start, 'length': length}
            for index, (start, length) in enumerate(zip(starts, lengths))
        ]
        result = self.multishot_plan(segments)
        rate = 24000
        a = {'sample_rate': rate, 'waveform': torch.full((1, 1, 15 * rate), 0.25)}
        b = {'sample_rate': rate, 'waveform': torch.full((1, 1, 15 * rate), -0.5)}
        linx = {'resources': {'iamccs_minimax_h3_shotplan': result,
                              'cine_audio_timeline_json': {'fps': 24, 'audioSegments': segments}}}
        _, _, _, report = TIMELINE.mix_audio_timeline(
            linx, 0, [a, b], target_sample_rate=rate, headroom='none')
        by_chunk = {}
        for assignment in report['mapped_assignments']:
            by_chunk.setdefault(assignment['chunk_index'], []).append(assignment['segment_id'])
        self.assertEqual(by_chunk, {0: ['line_0', 'line_1', 'line_2', 'line_3']})

    def test_long_multishot_master_removes_hidden_audio_once(self):
        rate = 24000
        rows = [{'id': f'shot_{i}', 'type': 'image', 'imageFile': f'{i}.png',
                 'start': i * 120, 'length': 120} for i in range(7)]
        result = CORE.build_shotplan(
            timeline_data={'rows': rows, 'fps': 24, 'duration_seconds': 35},
            global_prompt='Dialogue across views.', duration_seconds=35,
            task_mode='longvid_multishot_hardcut', audio_mode='h3_custom_audio_drive',
            width=640, height=384,
        )
        source = {'sample_rate': rate, 'waveform': torch.linspace(-0.4, 0.4, 35 * rate).view(1, 1, -1)}
        segments = [{'id': 'master', 'audio_input': 1, 'start_seconds': 0, 'duration_seconds': 35}]
        linx = {'resources': {'iamccs_minimax_h3_shotplan': result,
                              'cine_audio_timeline_json': {'fps': 24, 'audioSegments': segments}}}
        _, master, selected, _ = TIMELINE.mix_audio_timeline(
            linx, 1, [source], target_sample_rate=rate, headroom='none')
        torch.testing.assert_close(master['waveform'], source['waveform'], rtol=0, atol=0)
        chunk = result['chunks'][1]
        delivered, _, _ = DRIVE.IAMCCS_MiniMaxH3AudioOutputPolicyR21().select(
            h3_audio=None, video_frames=torch.zeros(chunk['visible_frame_count'], 1, 1, 3),
            cine_linx=result, segment_index=1, locked_audio_slice=selected)
        start = chunk['timeline_start_frame'] * 1000
        end = start + chunk['visible_frame_count'] * 1000
        torch.testing.assert_close(delivered['waveform'], source['waveform'][..., start:end], rtol=0, atol=0)

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
