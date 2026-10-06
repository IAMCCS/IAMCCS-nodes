import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('cleanup', Path(__file__).parents[1] / 'iamccs_editor_parking_cleanup.py')
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)


class CleanupTests(unittest.TestCase):
    def test_preserves_projects_recent_and_foreign_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'session'; root.mkdir()
            projects = Path(tmp) / 'projects'; projects.mkdir()
            for name in ['T01_old.pt', 'T02_live.pt', 'T03_saved.pt', 'T04_recent.pt', 'original.mp4']:
                path = root / name; path.write_bytes(b'123')
                if 'recent' not in name:
                    os.utime(path, (1, 1))
            (projects / 'workflow.json').write_text(json.dumps({'widgets_values': [json.dumps({'parking_tensor_path': str(root / 'T03_saved.pt')})]}))
            manifest = {'schema':'iamccs.shotboard_video_editor.v1', 'assets':{'x':{'path':str(root / 'T02_live.pt')}}}
            result = cleanup.purge_unused(root, [manifest], [projects])
            self.assertEqual(result['deleted_files'], 1)
            self.assertEqual(result['deleted_bytes'], 3)
            self.assertFalse((root / 'T01_old.pt').exists())
            self.assertEqual(len(list(root.iterdir())), 4)

    def test_missing_manifest_and_unreadable_project_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / 'T01_old.pt'; file.write_bytes(b'keep'); os.utime(file, (1,1))
            with self.assertRaises(ValueError):
                cleanup.purge_unused(root, [], [])
            (root / 'broken.json').write_text('{"parking_tensor_path":')
            with self.assertRaises(ValueError):
                cleanup.purge_unused(root, [{'schema':'iamccs.shotboard_video_editor.v1'}], [root])
            self.assertTrue(file.exists())


if __name__ == '__main__':
    unittest.main()
