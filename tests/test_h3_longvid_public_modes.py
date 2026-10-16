import importlib.util
import ast
import logging
import torch
from pathlib import Path
import unittest

CORE_PATH = Path(__file__).parents[1] / "iamccs_minimax_h3_shotboard_core.py"
SPEC = importlib.util.spec_from_file_location("iamccs_h3_core_public_modes", CORE_PATH)
CORE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(CORE)


def row(index: int, start: int, length: int = 96):
    return {
        "id": f"kf_{index}",
        "type": "image",
        "start": start,
        "length": length,
        "duration_frames": length,
        "imageFile": f"kf_{index}.png",
        "prompt": f"Shot {index}.",
        "transition": "start" if index == 1 else "continuous",
        "use_guide": True,
    }


def build(mode: str, audio_mode: str = "h3_custom_audio_drive", context_frames: int = 22):
    rows = [row(1, 0), row(2, 96), row(3, 192)]
    return CORE.build_shotplan(
        timeline_data={"rows": rows, "fps": 24, "duration_seconds": 12},
        global_prompt="One authored film timeline.",
        duration_seconds=12,
        task_mode=mode,
        audio_mode=audio_mode,
        width=640,
        height=384,
        motion_context_tail_frames=context_frames,
    )


class PublicLongVidModesTests(unittest.TestCase):
    def test_multishot_static_camera_is_interval_scoped_and_dialogue_is_verbatim(self):
        rows = [row(1, 0, 120), row(2, 120, 120)]
        line = '<d>[Italian] Allora... perché sei qui?</d>'
        rows[0]['prompt'] = 'Locked-off tripod. ' + line + '\n\noverall_soundscape: Room tone.\n\nnon_diegetic_music: N/A'
        rows[1]['prompt'] = 'The camera slowly pushes toward the door.\n\noverall_soundscape: Room tone.\n\nnon_diegetic_music: N/A'
        result = CORE.build_shotplan(
            timeline_data={'rows': rows, 'fps': 24}, global_prompt='Static camera by default.',
            duration_seconds=10, task_mode='longvid_multishot_hardcut', width=640, height=384,
        )
        prompt = result['chunks'][0]['prompt']
        self.assertEqual(prompt.count(line), 1)
        for key in ('integrated_multimodal_description:', 'overall_soundscape:', 'non_diegetic_music:'):
            self.assertEqual(prompt.count(key), 1)
        self.assertIn('[Shot 2] At 00:05.000, the camera cuts directly to', prompt)
        self.assertIn('camera holds a Static Shot for this entire interval through 5.000000s', prompt)
        self.assertEqual(prompt.count('camera holds a Static Shot'), 1)
        self.assertIn('The camera slowly pushes toward the door.', prompt)
        self.assertNotIn('Shot kf_', prompt)
        self.assertFalse(any(g.get('terminal_reanchor') for g in result['chunks'][0]['guides']))

    def test_multishot_missing_checkpoint_latent_fails_before_conditioning(self):
        tree = ast.parse((CORE_PATH.parent / "iamccs_minimax_h3_atomic_backend.py").read_text(encoding="utf-8"))
        guard = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.If) and 'execution_prompt' in ast.unparse(node.test)
            and 'longvid_multishot_hardcut' in ast.unparse(node.test)
        )
        code = compile(ast.Module(body=[guard], type_ignores=[]), '<checkpoint-preflight>', 'exec')
        namespace = {
            'shotplan': {'task_mode': 'longvid_multishot_hardcut', 'chunks': [{}, {}]},
            'execution_prompt': {'582': {'class_type': 'IAMCCS_MiniMaxH3NativeCheckpointSave', 'inputs': {}}},
        }
        with self.assertRaisesRegex(ValueError, 'sampled_latent'):
            exec(code, namespace)
        namespace['execution_prompt']['582']['inputs']['sampled_latent'] = ['10', 3]
        exec(code, namespace)
        namespace['shotplan']['task_mode'] = 'i2va'
        namespace['execution_prompt']['582']['inputs'].pop('sampled_latent')
        exec(code, namespace)

    def test_multishot_encoder_receives_all_authored_images(self):
        tree = ast.parse((CORE_PATH.parent / "iamccs_minimax_h3_atomic_backend.py").read_text(encoding="utf-8"))
        branch = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.If) and 'task ==' in ast.unparse(node.test)
            and 'multishot_joint_window' in ast.unparse(node.test)
        )
        captured = {}
        class Clip:
            def tokenize(self, prompt, images):
                captured['images'] = images
                captured['prompt'] = prompt
                return prompt
            def encode_from_tokens_scheduled(self, tokens):
                return [[tokens, {}]]
        slot_paths = [f'user_{i}.png' for i in range(5)]
        loaded = {path: torch.full((1, 2, 2, 3), float(i)) for i, path in enumerate(slot_paths)}
        namespace = {
            'torch': torch, '_load_image': loaded.get,
            '_resize': lambda image, width, height, crop: image,
            '_empty_av_latent': lambda width, height, frames: ({'samples': 'joint'}, frames),
            '_run_h3_conditioning_with_cpu_fallback': lambda clip, plan, build: (build(clip), 'test'),
            'clip': Clip(), 'shotplan': {'chunks': [{}]}, 'segment_index': 0,
            'chunk': {'multishot_visual_slots': [{'source_path': path} for path in slot_paths]},
            'width': 640, 'height': 384, 'frames': 362, 'prompt': 'User camera slots',
            'manifest': [], 'LOG': logging.getLogger('multishot_test'),
        }
        body = [node for node in branch.body if not isinstance(node, ast.ImportFrom)]
        exec(compile(ast.Module(body=body, type_ignores=[]), '<multishot-conditioning>', 'exec'), namespace)
        self.assertEqual(len(captured['images']), 5)
        self.assertEqual([image.mean().item() for image in captured['images']], list(range(5)))
        self.assertEqual([item['label'] for item in namespace['manifest']], [f'<Picture {i + 1}>' for i in range(5)])
        self.assertNotIn('minimax_keyframes', namespace['result'][0][0][1])

    def test_multishot_uses_one_native_window_despite_saved_smaller_window(self):
        result = CORE.build_shotplan(
            timeline_data={'rows': [row(i + 1, i * 90, 90) for i in range(4)], 'fps': 24},
            global_prompt='Dialogue', duration_seconds=15, task_mode='longvid_multishot_hardcut',
            motion_context_window_frames=209, width=640, height=384,
        )
        self.assertEqual(len(result['chunks']), 1)
        self.assertEqual(result['chunks'][0]['frame_count'], 362)
        self.assertEqual(len(result['chunks'][0]['multishot_visual_slots']), 4)
        self.assertEqual([g['global_frame'] for g in result['chunks'][0]['guides']], [0, 90, 180, 270])

    def test_only_joint_multishot_owns_the_model_family_override(self):
        tree = ast.parse((CORE_PATH.parent / "iamccs_minimax_h3_atomic_backend.py").read_text(encoding="utf-8"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_effective_task")
        namespace = {"Any": object, "_cine_info_h3": lambda linx: (linx, {})}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<task-routing>", "exec"), namespace)
        resolve = namespace["_effective_task"]
        self.assertEqual(resolve({}, {"task_mode": "i2va"}), "i2va")
        self.assertEqual(resolve({"task_override": "ref2va"}, {"task_mode": "i2va"}), "ref2va")
        self.assertEqual(resolve({"task_override": "ref2va"}, {"task_mode": "t2va", "multishot_joint_window": True}), "t2va")

    def test_multishot_slots_share_one_fl2va_window(self):
        result = build("longvid_multishot_hardcut")
        self.assertEqual(result["task_mode"], "longvid_multishot_hardcut")
        self.assertEqual(result["backend_revision"], "iamccs-fl2va-multishot-joint-v2")
        self.assertEqual(result["audio_mode"], "h3_custom_audio_drive")
        self.assertTrue(result["mode_contract"]["single_latent_per_window"])
        self.assertTrue(result["mode_contract"]["audio_timeline_continuous"])
        self.assertTrue(result["mode_contract"]["custom_audio_performance_drive"])
        self.assertEqual(len(result["chunks"]), 1)
        chunk = result["chunks"][0]
        self.assertEqual(chunk["task_mode"], "t2va")
        self.assertEqual(chunk["first_image"], "")
        self.assertEqual(chunk["last_image"], "")
        self.assertEqual([g["global_frame"] for g in chunk["guides"]], [0, 96, 192])
        self.assertFalse(any(g.get("terminal_reanchor") for g in chunk["guides"]))
        self.assertEqual(chunk["visible_frame_count"], 288)
        self.assertIn("At 4.000000s", chunk["prompt"])
        self.assertIn("At 8.000000s", chunk["prompt"])

    def test_long_multishot_uses_technical_windows_not_one_sample_per_slot(self):
        rows = [row(i + 1, i * 120, 120) for i in range(7)]
        result = CORE.build_shotplan(
            timeline_data={"rows": rows, "fps": 24, "duration_seconds": 35},
            global_prompt="Dialogue across camera angles.", duration_seconds=35,
            task_mode="longvid_multishot_hardcut", audio_mode="h3_native_generated",
            width=640, height=384,
        )
        self.assertLess(len(result["chunks"]), len(rows))
        self.assertTrue(result["longvid_latent_tail"]["enabled"])
        self.assertEqual(sum(c["unique_frames"] for c in result["chunks"]), 840)
        self.assertTrue(all(c["frame_count"] <= 362 for c in result["chunks"]))
        guides = [g for c in result["chunks"] for g in c["guides"]]
        self.assertEqual([g["global_frame"] for g in guides], [i * 120 for i in range(7)])
        self.assertTrue(all(c["latent_tail_context_prefix_frames"] == 22 for c in result["chunks"][1:]))

    def test_multishot_guides_follow_moved_user_slots(self):
        starts = [0, 49, 155, 240, 309]
        ends = starts[1:] + [360]
        rows = [row(i + 1, start, end - start) for i, (start, end) in enumerate(zip(starts, ends))]
        rows[2]["imageFile"] = "user_replaced.png"
        result = CORE.build_shotplan(
            timeline_data={"rows": rows, "fps": 24, "duration_seconds": 15},
            global_prompt="User camera edits.", duration_seconds=15,
            task_mode="longvid_multishot_hardcut", width=640, height=384,
        )
        chunk = result["chunks"][0]
        self.assertEqual([g["global_frame"] for g in chunk["guides"]], starts)
        self.assertEqual([g["end_frame"] for g in chunk["guides"]], ends)
        self.assertEqual(chunk["guides"][2]["source_path"], "user_replaced.png")
        self.assertEqual(len(chunk["guides"]), 5)
        self.assertNotIn("81", [str(g["global_frame"]) for g in chunk["guides"]])
        self.assertIn("At 2.041667s", chunk["prompt"])

    def test_multishot_native_audio_has_no_synthetic_silence_handles(self):
        result = build("longvid_multishot_hardcut", audio_mode="h3_native_generated")
        for chunk in result["chunks"]:
            self.assertEqual(chunk["audio_handoff_prompt"], "")
            self.assertEqual(chunk["audio_handoff_silence_head_seconds"], 0)
            self.assertEqual(chunk["audio_handoff_silence_tail_seconds"], 0)

    def test_keyframe_continuous_is_n_minus_one_phase_aligned_flf(self):
        result = build("longvid_keyframe_continuous")
        self.assertEqual(result["task_mode"], "longvid_keyframe_continuous")
        self.assertEqual(result["backend_revision"], "iamccs-fl2va-continuous-av-v1")
        self.assertEqual(result["audio_mode"], "h3_custom_audio_drive")
        contract = result["iamccs_continuous_av"]
        self.assertEqual(contract["alignment_mode"], "phase_aligned_extended")
        self.assertEqual(contract["latent_transport"], "direct_full_native_av")
        self.assertTrue(contract["preserve_authored_intermediate_keyframes"])
        self.assertEqual(contract["freeze_tail_analysis"], "automatic_motion_safe")
        self.assertTrue(contract["segment_checkpoints"])
        self.assertEqual(contract["visible_landing_policy"], "trim_generated_lock_keep_motion")
        self.assertEqual(contract["video_crossfade_frames"], 0)
        self.assertEqual(len(result["chunks"]), 2)
        first, second = result["chunks"]
        self.assertEqual((first["task_mode"], first["first_image"], first["last_image"]), ("fl2va", "kf_1.png", "kf_2.png"))
        self.assertEqual((second["task_mode"], second["first_image"], second["last_image"]), ("fl2va", "", "kf_3.png"))
        self.assertEqual(second["join_mode"], "phase_aligned_full_av")
        self.assertEqual(second["overlap_frames"], 0)

    def test_keyframe_continuous_preserves_39_frame_context(self):
        result = build("longvid_keyframe_continuous", context_frames=39)
        self.assertEqual(result["iamccs_continuous_av"]["context_frames"], "39")
        self.assertEqual(result["herrgotts_direct_av_chain"]["context_frames"], "39")

    def test_public_continuous_clock_excludes_grid_and_has_no_editorial_silence(self):
        result = build("longvid_keyframe_continuous", audio_mode="h3_native_generated")
        cursor = 0
        for chunk in result["chunks"]:
            self.assertEqual(chunk["timeline_start_frame"], cursor)
            self.assertEqual(chunk["unique_frames"], chunk["requested_frame_count"])
            self.assertEqual(chunk["audio_handoff_prompt"], "")
            self.assertEqual(chunk["audio_handoff_silence_head_seconds"], 0)
            self.assertEqual(chunk["audio_handoff_silence_tail_seconds"], 0)
            cursor += chunk["requested_frame_count"]
        self.assertEqual(cursor, 12 * 24)

    def test_legacy_continuous_alias_remains_load_compatible(self):
        result = build("longvid_masked_loop_guided")
        self.assertEqual(result["task_mode"], "longvid_masked_loop_guided")
        self.assertEqual(result["backend_revision"], "iamccs-fl2va-continuous-av-v1")
        self.assertEqual(result["mode_contract"]["public_name"], "FL2VA CONTINUOUS AV")
        self.assertEqual(result["audio_mode"], "h3_custom_audio_drive")
        self.assertEqual(result["iamccs_continuous_av"]["alignment_mode"], "phase_aligned_extended")


if __name__ == "__main__":
    unittest.main()
