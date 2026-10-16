from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FaceProgrammeSourceContractTests(unittest.TestCase):
    def test_continuous_programme_uses_joined_frames_and_preserves_i2v_path(self):
        source = (ROOT / "iamccs_minimax_h3_face_delivery_variant.py").read_text(encoding="utf-8")
        self.assertIn('"longvid_keyframe_continuous"', source)
        self.assertIn("programme_frame_source", source)
        self.assertIn("source = native_frames.clone()", source)
        self.assertIn("source = source_full[trim:trim + visible].clone()", source)
        self.assertIn('delivery["iamccs_face_programme_frame_source"]', source)


if __name__ == "__main__":
    unittest.main()
