import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch


ROOT = Path(__file__).parents[1]


def _load_face_swap():
    if "folder_paths" not in sys.modules:
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.get_filename_list = lambda _kind: []
        folder_paths.get_full_path = lambda *_args: None
        sys.modules["folder_paths"] = folder_paths
    spec = importlib.util.spec_from_file_location("iamccs_h3_face_swap_under_test", ROOT / "iamccs_h3_face_swap.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


FACE_SWAP = _load_face_swap()


class FaceSwapMaskRepairTests(unittest.TestCase):
    def test_new_audio_is_explicit_opt_in(self):
        self.assertFalse(FACE_SWAP.face_swap_settings({})["generate_new_audio"])
        self.assertTrue(FACE_SWAP.face_swap_settings({"h3_faceswap_generate_new_audio": True})["generate_new_audio"])

    def test_leading_empty_tracking_frames_are_filled_from_first_detection(self):
        masks = torch.zeros((6, 4, 4), dtype=torch.float32)
        masks[2:, 1:3, 1:3] = 1.0
        repaired, report = FACE_SWAP._fill_empty_mask_frames(masks)
        self.assertTrue(torch.equal(repaired[0], masks[2]))
        self.assertTrue(torch.equal(repaired[1], masks[2]))
        self.assertEqual(report["first_active_before"], 2)
        self.assertEqual(report["first_active_after"], 0)
        self.assertEqual(report["filled_frames"], 2)

    def test_valid_masks_are_not_rewritten(self):
        masks = torch.ones((3, 2, 2), dtype=torch.float32)
        repaired, report = FACE_SWAP._fill_empty_mask_frames(masks)
        self.assertTrue(torch.equal(repaired, masks))
        self.assertEqual(report["filled_frames"], 0)


if __name__ == "__main__":
    unittest.main()
