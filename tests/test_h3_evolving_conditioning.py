import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


CORE = load_module("iamccs_h3_evolving_core_under_test", "iamccs_minimax_h3_shotboard_core.py")


def schedule(policy="evolving"):
    return {
        "schema": "iamccs.h3.evolving.v1",
        "policy": policy,
        "global_context": "Same traveler, blue coat, uninterrupted city walk.",
        "beats": [
            {"id": "walk", "start_frame": 0, "end_frame": 240, "action": "walk steadily through the street"},
            {"id": "turn", "start_frame": 240, "end_frame": 480, "action": "stop and turn toward the fountain"},
            {"id": "cross", "start_frame": 480, "end_frame": 720, "action": "cross the square and accelerate"},
        ],
        "reference_calls": [{"id": "picture_1_return", "at_frame": 600, "reference_id": "picture-1", "role": "composition"}],
    }


def compile_plan(policy="evolving", root="i2va"):
    return CORE.build_shotplan(
        timeline_data={
            "rows": [{"id": "root", "type": "image", "start": 0, "length": 720, "imageFile": "root.png", "prompt": "root guide", "use_guide": True}],
            "fps": 24,
            "duration_seconds": 30,
            "pan_h3_conditioning_v1": schedule(policy),
        },
        global_prompt="Legacy global prompt.",
        duration_seconds=30,
        task_mode="fl2va_extended_av",
        extended_av_root_task=root,
        width=960,
        height=544,
    )


class EvolvingConditioningTests(unittest.TestCase):
    def test_extended_chunks_receive_only_their_active_action_beats(self):
        result = compile_plan()
        prompts = [chunk["prompt"] for chunk in result["chunks"]]
        self.assertEqual(result["chunks"][0]["task_mode"], "i2va")
        self.assertIn("walk steadily", prompts[0])
        self.assertNotIn("cross the square", prompts[0])
        self.assertTrue(any("stop and turn" in prompt for prompt in prompts[1:]))
        self.assertTrue(any("cross the square" in prompt for prompt in prompts[1:]))
        self.assertTrue(all(chunk.get("conditioning_revision") for chunk in result["chunks"]))
        self.assertEqual(len({chunk["conditioning_revision"] for chunk in result["chunks"]}), len(result["chunks"]))

    def test_reference_call_is_bound_only_to_its_visible_chunk(self):
        result = compile_plan()
        calls = [(chunk["timeline_start_frame"], chunk.get("conditioning_reference_calls", [])) for chunk in result["chunks"]]
        owners = [(start, values) for start, values in calls if values]
        self.assertEqual(len(owners), 1)
        self.assertEqual(owners[0][1][0]["id"], "picture_1_return")
        self.assertGreaterEqual(owners[0][1][0]["sample_local_frame"], 0)

    def test_continuous_policy_preserves_legacy_global_prompt(self):
        result = compile_plan("continuous")
        self.assertTrue(all(chunk["prompt"] == "Legacy global prompt." for chunk in result["chunks"]))
        self.assertTrue(all("conditioning_revision" not in chunk for chunk in result["chunks"]))

    def test_visible_shotboard_prompt_replaces_stale_schedule_context(self):
        result = compile_plan()
        self.assertTrue(all("Legacy global prompt." in chunk["prompt"] for chunk in result["chunks"]))
        self.assertTrue(all("Same traveler, blue coat" not in chunk["prompt"] for chunk in result["chunks"]))

    def test_extended_prompt_blocks_are_the_editable_queue_truth(self):
        timeline = {
            "rows": [{"id": "root", "type": "image", "start": 0, "length": 720, "imageFile": "root.png", "use_guide": True}],
            "fps": 24,
            "duration_seconds": 30,
            # Deliberately stale cache: visible blocks below must replace it.
            "pan_h3_conditioning_v1": schedule("continuous"),
            "extended_prompt_blocks": [
                {"id": "hold", "start_frame": 0, "end_frame": 192, "prompt": "hold still and breathe"},
                {"id": "hand", "start_frame": 192, "end_frame": 384, "prompt": "raise the right hand"},
                {"id": "march", "start_frame": 384, "end_frame": 576, "prompt": "begin the forward march"},
                {"id": "sword", "start_frame": 576, "end_frame": 720, "prompt": "draw the sword while marching"},
            ],
        }
        result = CORE.build_shotplan(
            timeline_data=timeline,
            global_prompt="Same commander and uninterrupted battlefield continuity.",
            duration_seconds=30,
            task_mode="fl2va_extended_av",
            extended_av_root_task="i2va",
            width=960,
            height=544,
        )
        prompts = [chunk["prompt"] for chunk in result["chunks"]]
        self.assertIn("hold still", prompts[0])
        self.assertNotIn("raise the right hand", prompts[0])
        self.assertNotIn("draw the sword", prompts[0])
        self.assertIn("raise the right hand", prompts[1])
        self.assertTrue(any("begin the forward march" in prompt for prompt in prompts[2:]))
        self.assertTrue(any("draw the sword while marching" in prompt for prompt in prompts[3:]))

    def test_action_crossing_a_chunk_boundary_is_marked_as_state_not_restarted(self):
        result = compile_plan()
        continued = [
            chunk for chunk in result["chunks"]
            if any(binding.get("continued_from_previous_chunk") for binding in chunk.get("conditioning_beat_bindings", []))
        ]
        self.assertTrue(continued)
        for chunk in continued:
            self.assertIn("CARRIED STATE FROM THE PREVIOUS CHUNK", chunk["local_prompt"])



    def test_continued_chunk_uses_positive_carried_state_without_repeating_event_words(self):
        timeline = {
            "rows": [{"id": "root", "type": "image", "start": 0, "length": 720, "imageFile": "root.png", "prompt": "root guide", "use_guide": True}],
            "fps": 24,
            "duration_seconds": 30,
            "pan_h3_conditioning_v1": {
                "schema": "iamccs.h3.evolving.v1",
                "policy": "evolving",
                "global_context": "Same soldier, battlefield continuity and stable camera axis.",
                "beats": [
                    {"id": "walk", "start_frame": 0, "end_frame": 288, "action": "[ONSET ONCE] the soldier woman takes one deep breath; [THEN SUSTAIN] she walks toward the camera while the army follows in formation"},
                    {"id": "scream", "start_frame": 288, "end_frame": 528, "action": "[ONSET ONCE] the soldier woman gives one brief rage scream and the soldiers answer once; [RESOLVED STATE] their mouths close and their faces return to a focused marching expression; [THEN SUSTAIN] after both screams have finished, they continue advancing toward the camera in the same formation without repeating either scream before 22 seconds"},
                ],
                "reference_calls": [],
            },
        }
        result = CORE.build_shotplan(
            timeline_data=timeline,
            global_prompt="Same soldier, battlefield continuity and stable camera axis.",
            duration_seconds=30,
            task_mode="fl2va_extended_av",
            extended_av_root_task="i2va",
            width=960,
            height=544,
        )
        carried = next(
            chunk for chunk in result["chunks"]
            if 288 < int(chunk["timeline_start_frame"]) < 528
            and "CARRIED STATE FROM THE PREVIOUS CHUNK" in chunk.get("local_prompt", "")
            and "scream" not in chunk.get("local_prompt", "").lower()
        )
        self.assertIn("CARRIED STATE FROM THE PREVIOUS CHUNK", carried["local_prompt"])
        self.assertNotIn("scream", carried["local_prompt"].lower())
        self.assertIn("focused marching expression", carried["local_prompt"].lower())
        self.assertIn("continue advancing toward the camera", carried["local_prompt"].lower())


    def test_legacy_onset_once_and_sustain_aliases_strip_completed_scream_from_chunk3(self):
        timeline = {
            "rows": [{"id": "root", "type": "image", "start": 0, "length": 720, "imageFile": "root.png", "prompt": "root guide", "use_guide": True}],
            "fps": 24,
            "duration_seconds": 30,
            "pan_h3_conditioning_v1": {
                "schema": "iamccs.h3.evolving.v1",
                "policy": "evolving",
                "global_context": "Same soldier woman, army, battlefield, lighting and backward tracking camera.",
                "beats": [
                    {"id": "walk", "start_frame": 0, "end_frame": 288, "action": "[ONSET_once] the soldier woman takes one deep breath [SUSTAIN] she walks toward the camera while the army follows behind her in formation"},
                    {"id": "scream", "start_frame": 288, "end_frame": 528, "action": "[ONSET_once] the soldier woman gives one brief rage scream toward the camera and the soldiers immediately answer with one brief collective scream [SUSTAIN] after both screams have finished, they continue advancing toward the camera in the same formation without repeating either scream before 22 seconds"},
                    {"id": "laugh", "start_frame": 528, "end_frame": 720, "action": "the soldier woman stops and starts to laugh while the army behind her starts to sing epic battle songs"},
                ],
                "reference_calls": [],
            },
        }
        result = CORE.build_shotplan(
            timeline_data=timeline,
            global_prompt="Same soldier woman, army, battlefield, lighting and backward tracking camera.",
            duration_seconds=30,
            task_mode="fl2va_extended_av",
            extended_av_root_task="i2va",
            width=960,
            height=544,
        )
        onset_chunk = next(
            chunk for chunk in result["chunks"]
            if int(chunk["timeline_start_frame"]) <= 288
            < int(chunk["timeline_start_frame"]) + int(chunk["unique_frames"])
        )["local_prompt"].lower()
        carried_chunk = next(
            chunk for chunk in result["chunks"]
            if 288 < int(chunk["timeline_start_frame"]) < 528
            and "scream" not in chunk.get("local_prompt", "").lower()
        )["local_prompt"].lower()
        self.assertIn("scream", onset_chunk)
        self.assertIn("then sustain", onset_chunk)
        self.assertNotIn("after both screams", onset_chunk)
        self.assertNotIn("without repeating", onset_chunk)
        self.assertNotIn("scream", carried_chunk)
        self.assertIn("continue advancing toward the camera", carried_chunk)

if __name__ == "__main__":
    unittest.main()
