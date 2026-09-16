import importlib.util
from pathlib import Path
import unittest


CORE_PATH = Path(__file__).parents[1] / "iamccs_minimax_h3_shotboard_core.py"
SPEC = importlib.util.spec_from_file_location("iamccs_h3_core_under_test", CORE_PATH)
CORE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(CORE)


def image_row(index, start, length, transition="continuous"):
    return {
        "id": f"pose_{index}",
        "type": "image",
        "start": start,
        "length": length,
        "duration_frames": length,
        "imageFile": f"pose_{index}.png",
        "prompt": f"Continuous action pose {index}.",
        "transition": transition,
        "use_guide": True,
    }


def plan(rows, *, duration_frames, tail=22, window=362):
    return CORE._longvid_guide_plan(
        timeline={"rows": rows, "fps": 24, "duration_seconds": duration_frames / 24},
        global_prompt="One uninterrupted continuous action.",
        duration_seconds=duration_frames / 24,
        prompt_mapping="global_plus_local",
        resolved_width=960,
        resolved_height=544,
        audio_mode="h3_native_generated",
        acceleration="pdd_native_8step",
        ref_image_size="match",
        text_encoder_device="gpu_auto",
        roles=["composition"] * 4,
        reference_video_role="off",
        reference_audio_role="off",
        sol_conditioning="exact_kv",
        spectrum_profile="low_vram",
        vram_clean_before_decode=True,
        rife_mode="off",
        active_upscale_mode="off",
        upscale_enabled=False,
        voice_reference_picture_index=0,
        motion_context_tail_frames=tail,
        motion_context_audio=True,
        motion_context_window_frames=window,
    )


class LongVidMotionContextRegressionTests(unittest.TestCase):
    def test_legacy_workflow_defaults_to_proven_r37_full_window(self):
        result = CORE.build_shotplan(
            timeline_data={"rows": [image_row(1, 0, 378)], "duration_seconds": 15.75},
            global_prompt="One uninterrupted take.",
            duration_seconds=15.75,
            task_mode="longvid_motion_context",
            width=960,
            height=544,
        )
        self.assertEqual(result["chunk_max_frames"], 362)
        self.assertEqual(result["motion_context_auto_chain"]["context_frames"], 22)

    def test_continuous_pose_sequence_keeps_active_slot_across_native_boundary(self):
        rows = [image_row(i + 1, i * 102, 102, "start" if i == 0 else "continuous") for i in range(4)]
        result = plan(rows, duration_frames=408)

        self.assertEqual(result["task_mode"], "longvid_motion_context")
        self.assertEqual([chunk["timeline_start_frame"] for chunk in result["chunks"]], [0, 340])
        self.assertEqual([chunk["motion_context_trim_frames"] for chunk in result["chunks"]], [0, 22])
        image_ids = [
            [guide["id"] for guide in chunk["guides"] if guide["kind"] == "image"]
            for chunk in result["chunks"]
        ]
        self.assertEqual(image_ids, [["pose_1", "pose_2", "pose_3", "pose_4"], ["pose_4"]])

    def test_active_slot_is_rebased_only_when_explicit_small_window_requires_it(self):
        rows = [image_row(1, 0, 204, "start"), image_row(2, 204, 102, "continuous")]
        result = plan(rows, duration_frames=306)
        image_ids = [
            [guide["id"] for guide in chunk["guides"] if guide["kind"] == "image"]
            for chunk in result["chunks"]
        ]
        self.assertEqual(image_ids, [["pose_1", "pose_2"]])

    def test_slot_transition_does_not_disable_native_tail(self):
        rows = [image_row(1, 0, 400, "start"), image_row(2, 400, 100, "hard_cut")]
        result = plan(rows, duration_frames=500)

        self.assertEqual(result["chunks"][1]["transition"], "motion_context_continuation")
        self.assertEqual(result["chunks"][1]["motion_context_trim_frames"], 22)
        self.assertNotIn("motion_context_reset_at_authored_cut", result["chunks"][1])


class LongContinuousGuidedContractTests(unittest.TestCase):
    def compile(self, count, *, duration_frames=408, tail=22):
        spacing = duration_frames // max(1, count)
        rows = [
            image_row(index + 1, index * spacing, spacing, "start" if index == 0 else "continuous")
            for index in range(count)
        ]
        return CORE.build_shotplan(
            timeline_data={"rows": rows, "fps": 24, "duration_seconds": duration_frames / 24},
            global_prompt="One uninterrupted evolving take.",
            duration_seconds=duration_frames / 24,
            task_mode="longvid_continuous_guided",
            width=736,
            height=416,
            motion_context_tail_frames=tail,
            motion_context_audio=True,
        )

    def test_two_guides_compile_one_native_flf_interval(self):
        result = self.compile(2, duration_frames=240)
        self.assertEqual(result["total_segments"], 1)
        self.assertEqual(result["chunks"][0]["task_mode"], "fl2va")
        self.assertEqual(result["chunks"][0]["motion_context_trim_frames"], 0)
        self.assertEqual(result["chunks"][0]["first_image"], "pose_1.png")
        self.assertEqual(result["chunks"][0]["last_image"], "pose_2.png")

    def test_four_guides_compile_three_continuous_destination_intervals(self):
        result = self.compile(4)
        self.assertEqual(result["total_segments"], 3)
        self.assertEqual(result["backend_variant"], "motion_context_auto_chain_v1")
        self.assertEqual(result["continuation_mode"], "direct_native_av_latent_plus_next_shotboard_destination")
        self.assertEqual([chunk["task_mode"] for chunk in result["chunks"]], ["fl2va"] * 3)
        self.assertEqual([chunk["motion_context_trim_frames"] for chunk in result["chunks"]], [0, 22, 22])
        self.assertEqual([chunk["last_image"] for chunk in result["chunks"]], ["pose_2.png", "pose_3.png", "pose_4.png"])
        self.assertTrue(all("guides" not in chunk for chunk in result["chunks"]))
        self.assertEqual(
            result["total_unique_frames"],
            sum(chunk["frame_count"] - chunk["motion_context_trim_frames"] for chunk in result["chunks"]),
        )

    def test_continuation_sample_never_exceeds_h3_model_limit(self):
        rows = [image_row(1, 0, 5, "start"), image_row(2, 20, 5), image_row(3, 375, 5)]
        with self.assertRaisesRegex(ValueError, "LONG CONTINUOUS GUIDED interval"):
            CORE.build_shotplan(
                timeline_data={"rows": rows, "fps": 24, "duration_seconds": 380 / 24},
                global_prompt="One take.", duration_seconds=380 / 24,
                task_mode="longvid_continuous_guided", width=736, height=416,
                motion_context_tail_frames=22,
            )


if __name__ == "__main__":
    unittest.main()
