import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location('h3_image_truth_core', ROOT / 'iamccs_minimax_h3_shotboard_core.py')
CORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CORE)


class ShotboardImageTruthTests(unittest.TestCase):
    def test_edited_image_prompt_and_deleted_slot_win_over_imported_mirrors(self):
        rows = [dict(id='current', type='image', ref=1, frame=0, length=120,
                     imageFile='current.png', imageTruthPath='imported.png', prompt='Current action',
                     local_prompt='Old action', relay_prompt='Old injection'),
                dict(id='deleted', type='image', ref=2, frame=120, length=120,
                     imageFile='', imageTruthPath='deleted.png', prompt='', local_prompt='Deleted action')]
        for mode in ('i2va', 'fl2va', 'longvid_multishot_hardcut', 'longvid_guides'):
            with self.subTest(mode=mode):
                result = CORE.build_shotplan(
                    timeline_data={'truth_revision': 12, 'rows': rows,
                                   'segments': [dict(rows[0], imageFile='imported.png', prompt='Old action')],
                                   'image_paths': ['imported.png', 'deleted.png'], 'fps': 24},
                    global_prompt='Current global', duration_seconds=10,
                    task_mode=mode, width=640, height=384)
                chunks = result['chunks']
                images = [g['source_path'] for c in chunks for g in c.get('guides', []) if g['kind'] == 'image']
                if mode in ('i2va', 'fl2va'):
                    images = [c['first_image'] for c in chunks if c.get('first_image')]
                self.assertIn('current.png', images)
                self.assertNotIn('imported.png', images)
                self.assertNotIn('deleted.png', images)
                prompt = '\n'.join(c['prompt'] for c in chunks)
                self.assertIn('Current action', prompt)
                self.assertNotIn('Old action', prompt)
                self.assertNotIn('Old injection', prompt)
                self.assertNotIn('Deleted action', prompt)

    def test_multishot_over_fifteen_minutes_preserves_authored_clock(self):
        for seconds in (900, 901, 1200):
            for audio in ('h3_custom_audio_drive', 'h3_native_generated'):
                with self.subTest(seconds=seconds, audio=audio):
                    frames = seconds * 24
                    rows = [dict(id=str(i), type='image', frame=start,
                                 length=min(120, frames-start), imageFile=f'user_{i}.png', prompt=f'Action {i}')
                            for i, start in enumerate(range(0, frames, 120))]
                    result = CORE.build_shotplan(
                        timeline_data={'rows': rows, 'fps': 24}, global_prompt='Film',
                        duration_seconds=seconds, task_mode='longvid_multishot_hardcut',
                        audio_mode=audio, width=640, height=384)
                    cursor = 0
                    for index, chunk in enumerate(result['chunks']):
                        self.assertEqual(chunk['timeline_start_frame'], cursor)
                        self.assertLessEqual(chunk['frame_count'], 362)
                        self.assertEqual(chunk.get('latent_tail_context_prefix_frames', 0), 22 if index else 0)
                        cursor += chunk['visible_frame_count']
                    self.assertEqual(cursor, frames)
                    self.assertEqual(sum(len(c['guides']) for c in result['chunks']), len(rows))


if __name__ == '__main__':
    unittest.main()
