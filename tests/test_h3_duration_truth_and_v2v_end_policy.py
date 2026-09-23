import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch


ROOT = Path(__file__).parents[1]


def _load_core():
    spec = importlib.util.spec_from_file_location("iamccs_h3_duration_core_under_test", ROOT / "iamccs_minimax_h3_shotboard_core.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _load_v2v(core):
    package_name = "iamccs_h3_duration_test_package"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT)]
    sys.modules[package_name] = package
    sys.modules[f"{package_name}.iamccs_minimax_h3_shotboard_core"] = core

    atomic = types.ModuleType(f"{package_name}.iamccs_minimax_h3_atomic_backend")
    atomic._resolve_shotplan = lambda value: value
    atomic._run_h3_conditioning_with_cpu_fallback = lambda *args, **kwargs: None
    sys.modules[atomic.__name__] = atomic

    linx = types.ModuleType(f"{package_name}.iamccs_supernodes_linx")
    linx.build_stage_linx_payload = lambda *args, **kwargs: {}
    sys.modules[linx.__name__] = linx

    spec = importlib.util.spec_from_file_location(
        f"{package_name}.iamccs_minimax_h3_v2v_backend",
        ROOT / "iamccs_minimax_h3_v2v_backend.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CORE = _load_core()
V2V = _load_v2v(CORE)


class DurationTruthRegressionTests(unittest.TestCase):
    def test_live_frame_length_wins_over_stale_serialized_seconds(self):
        result = CORE.build_shotplan(
            timeline_data={
                "schema": "iamccs.cine.filmmaker_timeline",
                "frame_rate": 24,
                "duration_seconds": 6,
                "rows": [{
                    "id": "edited_range",
                    "type": "text",
                    "start": 0,
                    "length": 144,
                    "duration_seconds": 124 / 24,
                    "use_guide": False,
                }],
            },
            global_prompt="test",
            duration_seconds=6,
            task_mode="v2va_face_swap",
            width=864,
            height=480,
        )
        self.assertEqual(result["slots"][0]["requested_frame_count"], 144)

    def test_master_duration_clips_stale_row_before_h3_alignment(self):
        result = CORE.build_shotplan(
            timeline_data={
                "fps": 24,
                "duration_seconds": 6,
                "rows": [{
                    "id": "stale_six_second_row",
                    "type": "image",
                    "start": 0,
                    "length": 144,
                    "imageFile": "guide.png",
                    "use_guide": True,
                }],
            },
            global_prompt="test",
            duration_seconds=5,
            task_mode="i2va",
            width=960,
            height=544,
        )
        self.assertEqual(result["slots"][0]["requested_frame_count"], 120)
        self.assertEqual(result["slots"][0]["frame_count"], 124)

    def test_master_duration_keeps_flf_terminal_image_anchor(self):
        result = CORE.build_shotplan(
            timeline_data={
                "fps": 24,
                "duration_seconds": 6,
                "rows": [
                    {"id": "a", "type": "image", "start": 0, "length": 12, "imageFile": "a.png", "use_guide": True},
                    {"id": "b", "type": "image", "start": 120, "length": 12, "imageFile": "b.png", "use_guide": True},
                ],
            },
            global_prompt="test",
            duration_seconds=5,
            task_mode="fl2va",
            width=960,
            height=544,
        )
        self.assertEqual(len(result["chunks"]), 1)
        self.assertEqual(result["chunks"][0]["requested_frame_count"], 120)
        self.assertEqual(result["chunks"][0]["last_image"], "b.png")

    def test_recommended_policy_holds_only_the_alignment_tail(self):
        indices, report = V2V._frame_indices(
            source_frames=124,
            source_fps=24,
            start_seconds=0,
            requested_frames=120,
            aligned_frames=124,
            end_policy="hold_last_for_grid",
        )
        self.assertEqual(int(indices[-1]), 119)
        self.assertEqual(report["grid_tail_hold_frames"], 4)
        self.assertEqual(report["visible_last_frame_hold_frames"], 0)

    def test_recommended_policy_rejects_a_genuinely_short_visible_source(self):
        with self.assertRaisesRegex(ValueError, "requested_frames=120"):
            V2V._frame_indices(
                source_frames=100,
                source_fps=24,
                start_seconds=0,
                requested_frames=120,
                aligned_frames=124,
                end_policy="hold_last_for_grid",
            )

    def test_tolerant_policy_declares_video_hold_and_audio_silence(self):
        indices, report = V2V._frame_indices(
            source_frames=100,
            source_fps=24,
            start_seconds=0,
            requested_frames=120,
            aligned_frames=124,
            end_policy="hold_last_visible",
        )
        self.assertEqual(int(indices[-1]), 99)
        self.assertEqual(report["visible_last_frame_hold_frames"], 20)

        audio = {"waveform": torch.ones((1, 2, 1000)), "sample_rate": 240}
        padded = V2V._slice_audio(
            audio,
            start_seconds=0,
            requested_frames=120,
            aligned_frames=124,
            end_policy="hold_last_visible",
        )
        self.assertEqual(int(padded["waveform"].shape[-1]), 1240)
        self.assertEqual(padded["iamccs_source_visible_pad_samples"], 200)


if __name__ == "__main__":
    unittest.main()
