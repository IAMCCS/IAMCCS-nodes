"""CPU-only R46 TimelineDirector public mode regression contracts.

The native model, ComfyUI, ffmpeg and GPU are not executed here.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).parents[1]

def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module

CORE = load("iamccs_r46_test_core", "iamccs_minimax_h3_shotboard_core.py")
RESTORE = load("iamccs_r46_test_restore", "iamccs_minimax_h3_timeline_director_restore.py")


def rows():
    return [
        {"id": f"image_{i}", "type": "image", "start": (i-1)*180,
         "length": 180, "duration_frames": 180, "imageFile": f"image_{i}.png",
         "prompt": f"Direction of scene {i}.", "transition": "start" if i == 1 else "continuous",
         "use_guide": True} for i in range(1, 5)
    ]


def build(restored=True, engine="pianosequenza_1_native_tail"):
    return CORE._longvid_guide_plan(
        timeline={"rows": rows(), "fps": 24, "duration_seconds": 30.0},
        global_prompt="One continuous authored motion timeline.",
        duration_seconds=30.0, prompt_mapping="global_plus_local",
        resolved_width=960, resolved_height=544,
        audio_mode="h3_native_generated", acceleration="pdd_native_8step",
        ref_image_size="match", text_encoder_device="gpu_auto",
        roles=["composition"]*4, reference_video_role="off", reference_audio_role="off",
        sol_conditioning="exact_kv", spectrum_profile="low_vram",
        vram_clean_before_decode=True, rife_mode="off",
        active_upscale_mode="off", upscale_enabled=False,
        voice_reference_picture_index=0, motion_context_tail_frames=0,
        latent_tail_context_frames=22 if restored else 0,
        timeline_director_native=restored,
        motion_context_audio=True, motion_context_window_frames=362,
        terminal_endpoint_mode=RESTORE.ENGINES[engine]["terminal_endpoint_mode"] if restored else "hard_image",
    )


class TimelineDirectorR46Tests(unittest.TestCase):
    def test_30s_sample_and_delivered_geometries(self):
        plan = build()
        RESTORE.annotate_restored_plan(plan, RESTORE.resolve_continuous_engine("pianosequenza_1_native_tail"))
        chunks = plan["chunks"]
        self.assertEqual([c["frame_count"] for c in chunks], [362, 226, 209])
        self.assertEqual([c["unique_frames"] for c in chunks], [362, 187, 171])
        self.assertEqual([c["uses_bridge_first_frame"] for c in chunks], [False, True, True])
        self.assertEqual([c["trim_head_frames"] for c in chunks], [0, 1, 1])
        self.assertEqual([c["latent_tail_adaptive_window"] for c in chunks], [False, False, False])
        delivered = [c["timeline_director_contract"]["delivered_segment_frames"] for c in chunks]
        self.assertEqual(delivered, [362, 209, 193])
        self.assertEqual(RESTORE.master_geometry(delivered)["total_frames"], 720)
        self.assertEqual(RESTORE.master_geometry(delivered)["duration_seconds"], 30.0)
        self.assertEqual(plan["requested_task_mode"], "longvid_keyframe_continuous")
        self.assertFalse(plan["mode_contract"]["fl2va_pairwise_destination_backend"])

    def test_editorial_endpoints_not_technical_padding(self):
        plan = RESTORE.annotate_restored_plan(build(), RESTORE.resolve_continuous_engine("pianosequenza_1_native_tail"))
        chunks = plan["chunks"]
        self.assertEqual([c["timeline_director_contract"]["editorial_endpoint_frames"] for c in chunks], [362, 209, 193])
        self.assertEqual([c["timeline_director_contract"]["rgb_bridge_frames"] for c in chunks], [0, 1, 1])
        self.assertGreater(chunks[1]["frame_count"], 209)
        self.assertGreater(chunks[2]["frame_count"], 193)

    def test_three_engines_share_native_transport(self):
        for engine in RESTORE.ENGINES:
            if RESTORE.ENGINES[engine]["backend_family"] != "native_tail":
                continue
            with self.subTest(engine=engine):
                info = RESTORE.resolve_continuous_engine(engine)
                p = RESTORE.annotate_restored_plan(build(engine=engine), info)
                self.assertEqual(p["longvid_engine"], engine)
                self.assertTrue(p["mode_contract"]["sampled_native_video_audio_tail"])
                self.assertTrue(p["mode_contract"]["incoming_owns_overlap"])
                self.assertEqual(p["mode_contract"]["overlap_frames"], 22)
                self.assertNotEqual(p["backend_variant"], "iamccs_fl2va_continuous_av_v1")

    def test_authored_image_on_technical_boundary_never_replaces_native_bridge(self):
        original = rows
        boundary_rows = [
            {"id": "opening", "type": "image", "start": 0, "length": 362,
             "duration_frames": 362, "imageFile": "start.png", "prompt": "Stay moving.", "transition": "start", "use_guide": True},
            {"id": "at_362", "type": "image", "start": 362, "length": 358,
             "duration_frames": 358, "imageFile": "middle.png", "prompt": "Keep moving.", "transition": "continuous", "use_guide": True},
        ]
        try:
            globals()["rows"] = lambda: boundary_rows
            plan = build()
        finally:
            globals()["rows"] = original
        RESTORE.annotate_restored_plan(plan, RESTORE.resolve_continuous_engine("pianosequenza_1_native_tail"))
        child = plan["chunks"][1]
        self.assertTrue(child["uses_bridge_first_frame"])
        self.assertEqual(child["trim_head_frames"], 1)
        self.assertTrue(any(g["id"] == "at_362" and g["local_frame"] == 0 for g in child["guides"]))

    def test_legacy_mode_keeps_original_planner(self):
        legacy = build(restored=False)
        self.assertNotIn("timeline_director_native", legacy)
        self.assertEqual(legacy["task_mode"], "longvid_guides")
        self.assertNotEqual([c["unique_frames"] for c in legacy["chunks"]], [362, 187, 171])

    def test_unknown_engine_fails_closed(self):
        with self.assertRaises(ValueError):
            RESTORE.resolve_continuous_engine("fl2va_pairwise")

    def test_motion_context_conflict_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "no competing Motion Context"):
            CORE._longvid_guide_plan(
                timeline={"rows": rows(), "fps": 24}, global_prompt="test",
                duration_seconds=30, prompt_mapping="global_plus_local",
                resolved_width=960, resolved_height=544, audio_mode="h3_native_generated",
                acceleration="native", ref_image_size="match", text_encoder_device="gpu_auto",
                roles=["composition"]*4, reference_video_role="off", reference_audio_role="off",
                sol_conditioning="exact_kv", spectrum_profile="low_vram",
                vram_clean_before_decode=True, rife_mode="off", active_upscale_mode="off",
                upscale_enabled=False, voice_reference_picture_index=0,
                motion_context_tail_frames=22, latent_tail_context_frames=22,
                timeline_director_native=True,
            )

if __name__ == "__main__":
    unittest.main()
