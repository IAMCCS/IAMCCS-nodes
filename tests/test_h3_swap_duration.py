import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("swap_duration_core", ROOT / "iamccs_minimax_h3_shotboard_core.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


class SwapDurationTests(unittest.TestCase):
    def test_source_duration_clips_old_shotboard_slot_and_keeps_only_grid_padding(self):
        plan = core.build_shotplan(
            timeline_data={"duration_seconds": 200 / 24, "fps": 24,
                           "rows": [{"id": "old", "type": "image", "start": 0, "length": 356}]},
            global_prompt="Replace the character in Video 1 with Picture 1.",
            duration_seconds=200 / 24, task_mode="v2va_face_swap_lora", width=640, height=384,
        )
        self.assertAlmostEqual(plan["effective_duration_seconds"], 200 / 24)
        self.assertEqual(sum(chunk["requested_frame_count"] for chunk in plan["chunks"]), 200)

    def probe(self, fps, frames):
        capture = SimpleNamespace(get=lambda key: {1: fps, 2: frames}[key], release=lambda: setattr(self, "released", True))
        self.released = False
        tree = ast.parse((ROOT / "iamccs_h3_edit_media.py").read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "video_metadata")
        namespace = {"cv2": SimpleNamespace(VideoCapture=lambda _: capture, CAP_PROP_FPS=1, CAP_PROP_FRAME_COUNT=2),
                     "local_video_path": lambda path: path}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "video_metadata", "exec"), namespace)
        return namespace["video_metadata"]("source.mp4")

    def test_metadata_uses_frame_count_and_fps_without_decoding_or_sampling(self):
        result = self.probe(24, 200)
        self.assertEqual(result["duration_seconds"], 200 / 24)
        self.assertTrue(self.released)

    def test_invalid_metadata_releases_capture(self):
        with self.assertRaisesRegex(ValueError, "duration/FPS"):
            self.probe(0, 200)
        self.assertTrue(self.released)


if __name__ == "__main__":
    unittest.main()
