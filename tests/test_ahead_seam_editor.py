import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "iamccs_ahead_seam_editor.py"
SPEC = importlib.util.spec_from_file_location("iamccs_ahead_seam_editor_tested", MODULE_PATH)
SEAMS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SEAMS)


class AheadSeamEditorTests(unittest.TestCase):
    def test_curve_is_saved_in_window_and_frame_count_is_preserved(self):
        info = {"boundaries": [5], "frames": 10}
        windows = SEAMS.windows_for(info, [{"seam": 0, "radius": 1, "offset": 0, "method": "cosine"}])
        frames = [np.full((1, 1, 3), value, np.uint8) for value in range(10)]

        result = list(SEAMS.smooth_frames(iter(frames), windows))

        self.assertEqual(len(result), len(frames))
        self.assertEqual(windows[0]["method"], "cosine")

    def test_invalid_curve_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "curve"):
            SEAMS.windows_for({"boundaries": [5], "frames": 10},
                              [{"seam": 0, "radius": 1, "method": "unknown"}])


if __name__ == "__main__":
    unittest.main()
