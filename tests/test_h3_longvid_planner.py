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


class LongVidPositionedGuidesV2RegressionTests(unittest.TestCase):
    def test_cross_window_image_slot_is_not_reinjected_and_generated_bridge_opens_next_window(self):
        rows = [image_row(i + 1, i * 102, 102, "start" if i == 0 else "continuous") for i in range(4)]
        result = plan(rows, duration_frames=408, tail=0)

        self.assertEqual(result["task_mode"], "longvid_guides")
        self.assertEqual(result["backend_revision"], "r42-positioned-guides-v2")
        self.assertEqual(result["continuation_mode"], "longvid_positioned_guides_v2_generated_frame_bridge")
        self.assertEqual([chunk["timeline_start_frame"] for chunk in result["chunks"]], [0, 362])
        image_ids = [
            [guide["id"] for guide in chunk["guides"] if guide["kind"] == "image"]
            for chunk in result["chunks"]
        ]
        self.assertEqual(image_ids, [["pose_1", "pose_2", "pose_3", "pose_4"], []])
        self.assertEqual([chunk["task_mode"] for chunk in result["chunks"]], ["t2va", "i2va"])
        self.assertEqual([chunk["uses_bridge_first_frame"] for chunk in result["chunks"]], [False, True])
        self.assertEqual([chunk["trim_head_frames"] for chunk in result["chunks"]], [0, 1])
        self.assertIn("[LONGVID POSITIONED GUIDES V2]", result["chunks"][0]["prompt"])
        self.assertIn("begin adapting motion early enough", result["chunks"][0]["prompt"].lower())
        self.assertIn("immediately preceding generated frame", result["chunks"][1]["prompt"].lower())

    def test_authored_guide_exactly_on_chunk_boundary_remains_opening_authority(self):
        rows = [
            image_row(1, 0, 100, "start"),
            image_row(2, 362, 46, "continuous"),
        ]
        result = plan(rows, duration_frames=408, tail=0)
        second = result["chunks"][1]
        images = [guide for guide in second["guides"] if guide["kind"] == "image"]

        self.assertEqual(len(images), 1)
        self.assertEqual(images[0]["id"], "pose_2")
        self.assertEqual(images[0]["local_frame"], 0)
        self.assertEqual(second["task_mode"], "t2va")
        self.assertFalse(second["uses_bridge_first_frame"])
        self.assertEqual(second["trim_head_frames"], 0)
        self.assertEqual(second["positioned_guides_v2"]["opening_authority"], "authored_guide")


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
        self.assertEqual([chunk["timeline_start_frame"] for chunk in result["chunks"]], [0, 102, 204, 306])
        self.assertEqual([chunk["motion_context_trim_frames"] for chunk in result["chunks"]], [0, 22, 22, 22])
        image_ids = [
            [guide["id"] for guide in chunk["guides"] if guide["kind"] == "image"]
            for chunk in result["chunks"]
        ]
        self.assertEqual(image_ids, [["pose_1"], ["pose_2"], ["pose_3"], ["pose_4"]])
        self.assertTrue(all(chunk["motion_context_guide_boundary"] == "safe_handoff" for chunk in result["chunks"]))

    def test_short_final_technical_window_is_rebalanced_without_changing_duration(self):
        result = plan([image_row(i + 1, i * 102, 102) for i in range(4)],
                      duration_frames=408, window=209)
        chunks = result["chunks"]
        self.assertEqual([c["visible_frame_count"] for c in chunks], [102, 102, 102, 102])
        self.assertEqual(sum(c["visible_frame_count"] for c in chunks), 408)
        self.assertTrue(all(124 <= c["frame_count"] <= 209 for c in chunks))
        self.assertTrue(all((c["frame_count"] - 5) % 17 == 0 for c in chunks))

    def test_continuation_prompt_clock_matches_conditioning_clock(self):
        result = plan([image_row(1, 0, 204), image_row(2, 204, 204)],
                      duration_frames=408, window=209)
        for chunk in result["chunks"]:
            offset = chunk["motion_context_trim_frames"]
            for guide in chunk["guides"]:
                if guide["kind"] == "image":
                    self.assertIn(f"at {(guide['local_frame'] + offset) / 24:.2f}s: {guide['prompt']}",
                                  chunk["prompt"])

    def test_active_slot_is_rebased_only_when_explicit_small_window_requires_it(self):
        rows = [image_row(1, 0, 204, "start"), image_row(2, 204, 102, "continuous")]
        result = plan(rows, duration_frames=306)
        image_ids = [
            [guide["id"] for guide in chunk["guides"] if guide["kind"] == "image"]
            for chunk in result["chunks"]
        ]
        self.assertEqual(image_ids, [["pose_1"], ["pose_2"]])

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
        self.assertEqual([chunk["first_image"] for chunk in result["chunks"]], ["pose_1.png", "", ""])
        self.assertEqual([chunk["last_image"] for chunk in result["chunks"]], ["pose_2.png", "pose_3.png", "pose_4.png"])
        self.assertEqual(
            [chunk["flf_anchor_contract"] for chunk in result["chunks"]],
            [
                "authored_opening_and_destination",
                "previous_native_av_head_plus_authored_destination",
                "previous_native_av_head_plus_authored_destination",
            ],
        )
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

    def test_uneven_authored_positions_drive_interval_lengths_without_presets(self):
        rows = [
            image_row(1, 0, 48, "start"),
            image_row(2, 60, 96, "continuous"),
            image_row(3, 220, 48, "continuous"),
        ]
        result = CORE.build_shotplan(
            timeline_data={"rows": rows, "fps": 24, "duration_seconds": 300 / 24},
            global_prompt="One take.", duration_seconds=300 / 24,
            task_mode="longvid_continuous_guided", width=736, height=416,
            motion_context_tail_frames=22,
        )

        # Slot centres are 24, 108 and 244, so the authored gaps are 84:136.
        # The planner preserves that ratio across the requested 300-frame take;
        # no fixed 102/124-frame recipe is allowed to override the Shotboard.
        self.assertEqual([chunk["requested_frame_count"] for chunk in result["chunks"]], [115, 185])
        self.assertEqual([chunk["first_image"] for chunk in result["chunks"]], ["pose_1.png", ""])
        self.assertEqual([chunk["last_image"] for chunk in result["chunks"]], ["pose_2.png", "pose_3.png"])


class HerrgottsDirectAVContractTests(unittest.TestCase):
    def compile(self, rows, duration_frames=360, tail=22, window=192):
        return CORE.build_shotplan(
            timeline_data={"rows": rows, "fps": 24, "duration_seconds": duration_frames / 24},
            global_prompt="One continuous guided take.",
            duration_seconds=duration_frames / 24,
            task_mode="longvid_masked_loop_guided",
            width=736,
            height=416,
            motion_context_tail_frames=tail,
            motion_context_window_frames=window,
        )

    def test_three_guides_compile_two_direct_av_intervals(self):
        rows = [
            image_row(1, 0, 120, "start"),
            image_row(2, 120, 120, "continuous"),
            image_row(3, 240, 120, "continuous"),
        ]
        result = self.compile(rows)
        self.assertEqual(result["task_mode"], "longvid_masked_loop_guided")
        self.assertEqual(result["total_segments"], 2)
        self.assertEqual([chunk["task_mode"] for chunk in result["chunks"]], ["fl2va", "fl2va"])
        self.assertEqual([chunk["requested_frame_count"] for chunk in result["chunks"]], [180, 180])
        self.assertEqual([chunk["first_image"] for chunk in result["chunks"]], ["pose_1.png", ""])
        self.assertEqual([chunk["last_image"] for chunk in result["chunks"]], ["pose_2.png", "pose_3.png"])
        self.assertEqual(result["chunks"][1]["join_mode"], "phase_aligned_full_av")
        self.assertEqual(result["backend_variant"], "iamccs_fl2va_continuous_av_v1")
        self.assertTrue(result["herrgotts_direct_av_chain"]["enabled"])
        self.assertNotIn("masked_loop_guided", result)

    def test_authored_positions_are_not_replaced_by_equal_spacing(self):
        rows = [
            image_row(1, 0, 48, "start"),
            image_row(2, 77, 80, "continuous"),
            image_row(3, 251, 40, "continuous"),
        ]
        result = self.compile(rows, duration_frames=320, tail=39, window=209)
        self.assertEqual([chunk["requested_frame_count"] for chunk in result["chunks"]], [120, 200])
        self.assertEqual(result["herrgotts_direct_av_chain"]["context_frames"], "39")
        self.assertEqual(result["herrgotts_direct_av_chain"]["alignment_mode"], "phase_aligned_extended")

    def test_requires_two_distinct_images(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            self.compile([image_row(1, 0, 120, "start")], duration_frames=120)


if __name__ == "__main__":
    unittest.main()
