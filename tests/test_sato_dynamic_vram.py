import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2] / "ComfyUI-H3-IMG-Gen-SatoDive" / "__init__.py"
spec = importlib.util.spec_from_file_location("sato_h3_test", ROOT)
sato = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sato)


class DummyStage:
    pass


class DummyClip:
    def __init__(self):
        self.cond_stage_model = DummyStage()


class DynamicVRAMTests(unittest.TestCase):
    def test_12gb_auto_caps_native_but_keeps_requested_aspect(self):
        original = sato._cuda_total_gib
        sato._cuda_total_gib = lambda _clip: 12.5
        try:
            width, height, report = sato._safe_native_canvas(3840, 2176, object(), "auto_dynamic", 1.05)
        finally:
            sato._cuda_total_gib = original
        self.assertLessEqual(width * height, 1_100_000)
        self.assertEqual(width % 32, 0)
        self.assertEqual(height % 32, 0)
        self.assertAlmostEqual(width / height, 3840 / 2176, delta=0.08)
        self.assertIn("12.5GB", report)

    def test_large_gpu_keeps_native_4k(self):
        original = sato._cuda_total_gib
        sato._cuda_total_gib = lambda _clip: 24.0
        try:
            result = sato._safe_native_canvas(3840, 2176, object(), "auto_dynamic", 1.05)
        finally:
            sato._cuda_total_gib = original
        self.assertEqual(result[:2], (3840, 2176))

    def test_dynamic_reserve_reaches_clip_loader(self):
        clip = DummyClip()
        original = sato._cuda_total_gib
        sato._cuda_total_gib = lambda _clip: 12.5
        try:
            value, report = sato._run_conditioning_low_vram(
                clip, "auto_dynamic",
                lambda active: active.cond_stage_model.memory_estimation_function([], None),
            )
        finally:
            sato._cuda_total_gib = original
        self.assertGreaterEqual(value, 5120 * 1024 * 1024)
        self.assertEqual(report, "dynamic_reserve=5120MB")
        self.assertFalse(hasattr(clip.cond_stage_model, "memory_estimation_function"))


if __name__ == "__main__":
    unittest.main()
