import importlib.util
from pathlib import Path
import unittest


CORE_PATH = Path(__file__).parents[1] / "iamccs_minimax_h3_shotboard_core.py"
SPEC = importlib.util.spec_from_file_location("iamccs_h3_hd_core_under_test", CORE_PATH)
CORE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(CORE)


def image_row(index, start, length):
    return {
        "id": f"guide_{index}",
        "type": "image",
        "start": start,
        "length": length,
        "duration_frames": length,
        "imageFile": f"guide_{index}.png",
        "prompt": f"Guide {index}",
        "transition": "start" if index == 1 else "continuous",
        "use_guide": True,
    }


def plan(*, endpoint="pianosequenza_hd", adaptive=False, window=362, duration_frames=850,
         rows=None, mapping="global_plus_local", width=1280, height=768, timed=True):
    if rows is None:
        rows = [image_row(index + 1, index * 170, 170) for index in range(5)]
    return CORE._longvid_guide_plan(
        timeline={"rows": rows, "fps": 24, "duration_seconds": duration_frames / 24,
                  "longvid_prompt_timing": timed},
        global_prompt="One uninterrupted take.",
        duration_seconds=duration_frames / 24,
        prompt_mapping=mapping,
        resolved_width=width,
        resolved_height=height,
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
        motion_context_tail_frames=0,
        latent_tail_context_frames=22 if adaptive else 0,
        motion_context_audio=True,
        motion_context_window_frames=window,
        terminal_endpoint_mode=endpoint,
        pianosequenza_hd_context_frames=22,
    )


class PianosequenzaHDPlannerTests(unittest.TestCase):
    def test_subwater_final_window_uses_only_its_final_guide_prompt(self):
        starts = [0, 102, 204, 306, 408, 512, 616, 720]
        rows = [image_row(i + 1, start, starts[i + 1] - start)
                for i, start in enumerate(starts[:-1])]
        chunks = plan(endpoint="pianosequenza_drift", rows=rows, duration_frames=720)["chunks"]
        first, last = chunks
        self.assertIn("Timeline 4.25s to 8.50s: Guide 2", first["prompt"])
        self.assertEqual(last["prompt"], "Guide 7")
        self.assertNotIn("One uninterrupted take.", last["prompt"])
        active = [item["guide_id"] for item in last["prompt_guide_bindings"] if item["conditioning_active"]]
        self.assertEqual(active, ["guide_7"])
        self.assertEqual([item["guide_id"] for item in last["prompt_guide_bindings"]],
                         ["guide_4", "guide_5", "guide_6", "guide_7"])
        self.assertNotIn("guide_4", [g["id"] for g in last["guides"]])
        self.assertTrue(last["positioned_guides_v2"]["terminal_reanchor"])

    def test_terminal_reanchor_final_prompt_overrides_global_only_mapping(self):
        chunks = plan(mapping="global_only")["chunks"]
        for chunk in chunks[:-1]:
            self.assertEqual(chunk["prompt"], "One uninterrupted take.")
        self.assertEqual(chunks[-1]["prompt"], "Guide 5")
        self.assertFalse(chunks[-1]["positioned_guides_v2"]["labels_in_conditioning"])

    def test_local_only_does_not_inject_global(self):
        chunks = plan(mapping="local_only")["chunks"]
        for chunk in chunks:
            self.assertNotIn("One uninterrupted take.", chunk["prompt"])
            self.assertIn("Guide", chunk["prompt"])
        self.assertEqual(chunks[-1]["prompt"], "Guide 5")

    def test_pure_text_opt_out_keeps_fixed_prompt_selection(self):
        last = plan(timed=False)["chunks"][-1]
        self.assertEqual(last["prompt"], "Guide 5")
        self.assertNotIn("Timeline", last["prompt"])

    def test_text_clock_includes_hd_and_adaptive_hidden_prefix(self):
        for adaptive in (False, True):
            chunk = plan(adaptive=adaptive)["chunks"][1]
            self.assertIn("Timeline 0.92s", chunk["prompt"])

    def test_resolution_does_not_change_guides_or_prompt_clock(self):
        low = plan(width=640, height=384)["chunks"]
        for width, height in ((960, 544), (1280, 768), (1920, 1088)):
            high = plan(width=width, height=height)["chunks"]
            for a, b in zip(low, high):
                self.assertEqual(a["prompt"], b["prompt"])
                self.assertEqual(a["guides"], b["guides"])

    def test_standard_362_reserves_hidden_context_without_changing_editorial_duration(self):
        result = plan()
        chunks = result["chunks"]
        self.assertEqual([c["timeline_start_frame"] for c in chunks], [0, 362, 702])
        self.assertEqual([c["frame_count"] for c in chunks], [362, 362, 175])
        self.assertEqual([c["unique_frames"] for c in chunks], [362, 340, 148])
        self.assertEqual(
            [c.get("pianosequenza_hd_context_prefix_frames", 0) for c in chunks],
            [0, 22, 22],
        )
        self.assertEqual(sum(c["unique_frames"] for c in chunks), 850)
        self.assertTrue(all(c["frame_count"] <= 362 for c in chunks))
        self.assertTrue(all((c["frame_count"] - 5) % 17 == 0 for c in chunks))
        self.assertTrue(all(c["task_mode"] == "t2va" for c in chunks))
        self.assertTrue(all(not c["uses_bridge_first_frame"] for c in chunks))
        self.assertTrue(all(c["trim_head_frames"] == 0 for c in chunks))

    def test_standard_hd_keeps_guides_on_global_clock_before_backend_context_offset(self):
        result = plan()
        second = result["chunks"][1]
        guides = {g["id"]: g for g in second["guides"] if g["kind"] == "image"}
        self.assertEqual(second["timeline_start_frame"], 362)
        self.assertEqual(guides["guide_4"]["global_frame"], 510)
        self.assertEqual(guides["guide_4"]["local_frame"], 148)
        self.assertEqual(second["pianosequenza_hd_context_prefix_frames"], 22)
        # Atomic adds the hidden context exactly once: 148 + 22 = sample frame 170.
        self.assertEqual(guides["guide_4"]["local_frame"] + 22, 170)

    def test_adaptive_window_remains_independent_and_does_not_enable_hd_standard_contract(self):
        result = plan(adaptive=True, window=209)
        chunks = result["chunks"]
        self.assertTrue(any(c.get("latent_tail_adaptive_window") for c in chunks))
        self.assertTrue(all("pianosequenza_hd_hidden_context" not in c for c in chunks))
        self.assertEqual(sum(c["unique_frames"] for c in chunks), 850)
        self.assertTrue(all(c["frame_count"] <= 209 for c in chunks))

    def test_non_hd_standard_planner_is_unchanged(self):
        result = plan(endpoint="pianosequenza_drift", adaptive=False, duration_frames=850)
        chunks = result["chunks"]
        self.assertEqual([c["timeline_start_frame"] for c in chunks], [0, 362, 724])
        self.assertEqual([c["unique_frames"] for c in chunks], [362, 362, 126])
        self.assertEqual(sum(c["unique_frames"] for c in chunks), 850)
        self.assertTrue(all("pianosequenza_hd_context_prefix_frames" not in c for c in chunks))


if __name__ == "__main__":
    unittest.main()
