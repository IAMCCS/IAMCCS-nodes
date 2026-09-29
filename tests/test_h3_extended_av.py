import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest

import torch


ROOT = Path(__file__).parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


EXT = load_module("iamccs_h3_extended_av_under_test", "iamccs_minimax_h3_extended_av.py")
CORE = load_module("iamccs_h3_extended_av_core_under_test", "iamccs_minimax_h3_shotboard_core.py")


def image_row(index, start, length, *, prompt=None):
    return {
        "id": f"extended_{index}",
        "type": "image",
        "start": start,
        "length": length,
        "duration_frames": length,
        "imageFile": f"extended_{index}.png",
        "prompt": prompt or f"Continue motion state {index}.",
        "transition": "continuous",
        "use_guide": True,
    }


def compile_extended(rows, frames, **kwargs):
    return CORE.build_shotplan(
        timeline_data={"rows": rows, "fps": 24, "duration_seconds": frames / 24},
        global_prompt="One uninterrupted continuous audiovisual take.",
        duration_seconds=frames / 24,
        task_mode="fl2va_extended_av",
        width=960,
        height=544,
        **kwargs,
    )


class ExtendedAVGridTests(unittest.TestCase):
    def test_shared_av_grid_matches_upstream_masked_grid(self):
        legal = [39, 90, 141, 192, 243, 294, 345]
        illegal = [5, 22, 56, 73, 124, 209, 362]
        self.assertTrue(all(EXT.shared_av_window_ok(value) for value in legal))
        self.assertTrue(all(not EXT.shared_av_window_ok(value) for value in illegal))
        self.assertEqual(EXT.shared_av_snap_up(40), 90)
        self.assertEqual(EXT.shared_av_snap_down(191), 141)

    def test_profiles_keep_nonfinal_runs_on_shared_av_grid(self):
        expected = {
            "safe_8_12gb": (192, 39),
            "balanced_12_16gb": (243, 39),
            "quality_20_24gb": (294, 90),
            "max_32gb_plus": (345, 141),
        }
        for profile, pair in expected.items():
            contract = EXT.resolve_contract(profile=profile)
            self.assertEqual((contract["sample_window_frames"], contract["overlap_frames"]), pair)
            self.assertTrue(EXT.shared_av_window_ok(pair[0]))
            self.assertTrue(EXT.shared_av_window_ok(pair[1]))
            self.assertLess(pair[1], pair[0])
            self.assertEqual(contract["boundary_runway_frames"], 34)
            self.assertEqual(contract["boundary_runway_mode"], "auto_universal")
            self.assertEqual(
                contract["unique_capacity_frames"],
                pair[0] - pair[1] - 34,
            )

    def test_universal_runway_also_fits_smallest_legal_custom_window(self):
        contract = EXT.resolve_contract(
            profile="custom", custom_window_frames=90, custom_overlap_frames=39
        )
        self.assertEqual(contract["boundary_runway_frames"], 34)
        self.assertEqual(contract["unique_capacity_frames"], 17)
        self.assertEqual(contract["root_visible_capacity_frames"], 56)

    def test_extend_role_is_masked_and_audio_release_matches_before_pin(self):
        exact = EXT.resolve_contract(profile="safe_8_12gb")
        runway = EXT.resolve_contract(profile="safe_8_12gb", mask_profile="runway")
        self.assertEqual(exact["role"], "extend")
        self.assertEqual(exact["pin_mode"], "masked")
        self.assertEqual(exact["delivery_policy"], "trim_pinned_head_then_butt_join")
        self.assertEqual(exact["audio_mask"], "pinned_exact_with_internal_release")
        self.assertEqual(exact["masked_audio_release_ticks"], 8)
        self.assertEqual(exact["seam_repair_default"], "off_for_masked_exact")
        self.assertEqual(runway["video_ramp_frames"], 17)
        self.assertEqual(runway["video_edge"], 0.60)

    def test_hard_before_pin_handover_is_full_window_trim(self):
        exact = EXT.resolve_contract(mask_profile="exact")
        self.assertEqual(EXT.handover_frames(39, exact), 39)
        self.assertEqual(EXT.resolved_handover_mode(exact), "masked_trim")

    def test_phase7_does_not_claim_unwired_global_joint_refine(self):
        with self.assertRaises(ValueError):
            EXT.resolve_contract(level_lock_mode="global")
        with self.assertRaises(ValueError):
            EXT.resolve_contract(joint_refine_mode="full")


class ExtendedAVSidecarTests(unittest.TestCase):
    def setUp(self):
        self.contract = EXT.resolve_contract(profile="balanced_12_16gb")
        self.raw_frames = 243
        self.video = torch.arange(
            EXT.video_tokens_for_frames(self.raw_frames), dtype=torch.float32
        ).view(1, 1, -1, 1, 1).expand(1, 24, -1, 2, 2).clone()
        self.audio = torch.arange(
            EXT.audio_ticks_for_frames(self.raw_frames, require_exact=True), dtype=torch.float32
        ).view(1, 1, 1, -1).expand(1, 32, 2, -1).clone()
        self.latent = {"samples": (self.video, self.audio)}

    def test_sidecar_records_extend_style_lineage_and_delivered_geometry(self):
        original_folder = EXT._sidecar_folder
        try:
            with tempfile.TemporaryDirectory() as td:
                EXT._sidecar_folder = lambda: Path(td)
                EXT.save_sidecar(
                    "roundtrip", 0, self.latent, contract=self.contract,
                    delivered_frames=243, pinned_head_frames=0, padding_tail_frames=0,
                )
                parent = EXT.load_sidecar("roundtrip", 0)
                self.assertIsNotNone(parent)
                self.assertEqual(parent["meta"]["relation"], "root")
                self.assertEqual(int(parent["meta"]["delivered_frames"]), 243)
                self.assertEqual(int(parent["meta"]["pinned_head_frames"]), 0)

                EXT.save_sidecar(
                    "roundtrip", 1, self.latent, contract=self.contract,
                    delivered_frames=204, pinned_head_frames=39, padding_tail_frames=0,
                    parent_segment_index=0,
                )
                child = EXT.load_sidecar("roundtrip", 1)
                self.assertEqual(child["meta"]["parent_id"], parent["meta"]["self_id"])
                self.assertEqual(child["meta"]["relation"], "extends")
                self.assertEqual(int(child["meta"]["parent_join_frame"]), 243)
                self.assertEqual(int(child["meta"]["pinned_head_frames"]), 39)
                self.assertEqual(int(child["meta"]["delivered_frames"]), 204)
                self.assertIn('"place":"before"', child["meta"]["pins"])
                self.assertIn('"mode":"masked"', child["meta"]["pins"])
        finally:
            EXT._sidecar_folder = original_folder

    def test_hidden_boundary_runway_is_not_inherited_by_next_chunk(self):
        original_folder = EXT._sidecar_folder
        try:
            with tempfile.TemporaryDirectory() as td:
                EXT._sidecar_folder = lambda: Path(td)
                EXT.save_sidecar(
                    "runway", 0, self.latent, contract=self.contract,
                    delivered_frames=209, pinned_head_frames=0, padding_tail_frames=34,
                    boundary_runway_frames=34, grid_padding_frames=0,
                )
                parent = EXT.load_sidecar("runway", 0)
                self.assertEqual(int(parent["meta"]["delivered_frames"]), 209)
                self.assertEqual(int(parent["meta"]["boundary_runway_frames"]), 34)
                self.assertEqual(int(parent["meta"]["editorial_endpoint_raw_frame"]), 209)
                vtail, atail = EXT._tail_from_sidecar(parent, 39)
                # The inherited source ends at frame 209. Its 39f tail starts
                # at frame 170, NOT at the physical RAW end 243-39=204.
                self.assertEqual(float(vtail[0, 0, 0, 0, 0]), 50.0)
                self.assertEqual(float(atail[0, 0, 0, 0]), float(EXT.audio_total(170)))
        finally:
            EXT._sidecar_folder = original_folder

    def test_chained_tail_uses_delivered_to_raw_mapping_not_raw_tensor_tail(self):
        bundle = {
            "meta": {
                "pinned_head_frames": "39",
                "delivered_frames": "204",
                "raw_frames": "243",
            },
            "video": self.video,
            "audio": self.audio,
        }
        vtail, atail = EXT._tail_from_sidecar(bundle, 39)
        # raw_start = 39 + 204 - 39 = 204, which is latent step 60.
        self.assertEqual(float(vtail[0, 0, 0, 0, 0]), 60.0)
        self.assertEqual(vtail.shape[2], EXT.video_tokens_for_frames(39))
        self.assertEqual(atail.shape[-1], 65)
        self.assertEqual(float(atail[0, 0, 0, 0]), float(EXT.audio_total(204)))

    def test_apply_prefix_writes_parent_tail_into_fresh_target_and_full_masks(self):
        class FakeNestedTensor:
            def __init__(self, tensors):
                self.tensors = tuple(tensors)

        comfy = types.ModuleType("comfy")
        nested = types.ModuleType("comfy.nested_tensor")
        nested.NestedTensor = FakeNestedTensor
        comfy.nested_tensor = nested
        previous_comfy = sys.modules.get("comfy")
        previous_nested = sys.modules.get("comfy.nested_tensor")
        old_masks = EXT.core_masks_available
        sys.modules["comfy"] = comfy
        sys.modules["comfy.nested_tensor"] = nested
        EXT.core_masks_available = lambda: True
        try:
            target_video = torch.full_like(self.video, -99.0)
            target_audio = torch.full_like(self.audio, -99.0)
            target = {"samples": FakeNestedTensor((target_video, target_audio))}
            parent = {
                "meta": {"pinned_head_frames": "0", "delivered_frames": "243", "raw_frames": "243"},
                "video": self.video,
                "audio": self.audio,
            }
            out, conditioning, details = EXT.apply_prefix(
                target, [[torch.zeros(1), {}]], parent, contract=self.contract,
            )
            video, audio = out["samples"].tensors
            vmask, amask = out["noise_mask"].tensors
            steps = EXT.video_tokens_for_frames(39)
            ticks = EXT.audio_ticks_for_frames(39, require_exact=True)
            self.assertTrue(torch.equal(video[:, :, :steps], self.video[:, :, -steps:]))
            self.assertTrue(torch.equal(audio[..., :ticks], self.audio[..., -ticks:]))
            self.assertEqual(tuple(vmask.shape), (1, 1) + tuple(video.shape[2:]))
            self.assertEqual(tuple(amask.shape), (1, 1) + tuple(audio.shape[2:]))
            self.assertTrue(torch.all(vmask[:, :, :steps] == 0))
            self.assertTrue(torch.all(vmask[:, :, steps:] == 1))
            self.assertTrue(torch.all(amask[..., :ticks - 8] == 0))
            self.assertGreater(float(amask[..., ticks - 1].max()), 0.9)
            self.assertTrue(torch.all(amask[..., ticks:] == 1))
            self.assertEqual(details["trim_head_frames"], 39)
            self.assertEqual(details["role"], "extend")
            self.assertIs(conditioning[0][1].__class__, dict)
            video_only, _, video_details = EXT.apply_prefix(
                target, conditioning, parent, contract={**self.contract, "video_only": True},
            )
            torch.testing.assert_close(video_only["samples"].tensors[0], video, rtol=0, atol=0)
            torch.testing.assert_close(video_only["samples"].tensors[1], target_audio, rtol=0, atol=0)
            self.assertTrue(torch.all(video_only["noise_mask"].tensors[1] == 1))
            self.assertEqual(video_details["audio_release_ticks"], 0)
        finally:
            EXT.core_masks_available = old_masks
            if previous_comfy is None:
                sys.modules.pop("comfy", None)
            else:
                sys.modules["comfy"] = previous_comfy
            if previous_nested is None:
                sys.modules.pop("comfy.nested_tensor", None)
            else:
                sys.modules["comfy.nested_tensor"] = previous_nested


class ExtendedAVPlannerTests(unittest.TestCase):
    def test_long_i2v_is_dedicated_masked_extend_not_longvid_overlap(self):
        frames = 500
        result = compile_extended([image_row(1, 0, frames)], frames)
        self.assertEqual(result["task_mode"], "fl2va_extended_av")
        self.assertEqual(result["backend_variant"], "iamccs_fl2va_extend_style_v4_boundary_runway")
        self.assertEqual(result["extended_av"]["overlap_frames"], 39)
        self.assertGreater(len(result["chunks"]), 2)
        self.assertEqual(sum(chunk["unique_frames"] for chunk in result["chunks"]), frames)

        for index, chunk in enumerate(result["chunks"]):
            self.assertFalse(chunk.get("uses_bridge_first_frame", False))
            self.assertFalse(chunk.get("extended_av_retain_overlap", False))
            if index == 0:
                self.assertEqual(chunk["extended_av_role"], "root")
                self.assertEqual(chunk["extended_av_context_prefix_frames"], 0)
                self.assertEqual(chunk["extended_av_trim_head_frames"], 0)
                self.assertEqual(len(chunk.get("guides", [])), 1)
            else:
                self.assertEqual(chunk["extended_av_role"], "extend")
                self.assertEqual(chunk["extended_av_place"], "before")
                self.assertEqual(chunk["extended_av_context_prefix_frames"], 39)
                self.assertEqual(chunk["extended_av_trim_head_frames"], 39)
                self.assertEqual(chunk["join_mode"], "extended_av_masked_extend")
                # A single Picture 1 must not be re-injected into children.
                self.assertEqual(chunk.get("guides", []), [])

    def test_nonfinal_raw_runs_land_on_shared_grid_final_only_needs_h3_grid(self):
        result = compile_extended([image_row(1, 0, 480)], 480, extended_av_profile="balanced_12_16gb")
        chunks = result["chunks"]
        self.assertEqual(
            [
                (
                    c["frame_count"],
                    c["extended_av_context_prefix_frames"],
                    c["unique_frames"],
                    c["extended_av_boundary_runway_frames"],
                    c["extended_av_grid_padding_frames"],
                    c["extended_av_padding_frames"],
                )
                for c in chunks
            ],
            [
                (243, 0, 209, 34, 0, 34),
                (243, 39, 170, 34, 0, 34),
                (175, 39, 101, 34, 1, 35),
            ],
        )
        self.assertTrue(all(EXT.shared_av_window_ok(c["frame_count"]) for c in chunks[:-1]))
        self.assertTrue(EXT.h3_run_ok(chunks[-1]["frame_count"]))
        self.assertFalse(EXT.shared_av_window_ok(chunks[-1]["frame_count"]))
        self.assertEqual(sum(c["unique_frames"] for c in chunks), 480)

    def test_boundary_runway_is_hidden_from_delivery_and_next_lineage(self):
        result = compile_extended([image_row(1, 0, 458)], 458, extended_av_profile="safe_8_12gb")
        chunks = result["chunks"]
        self.assertEqual(result["extended_av"]["boundary_runway_frames"], 34)
        self.assertEqual(sum(c["unique_frames"] for c in chunks), 458)
        self.assertTrue(all(c["extended_av_boundary_runway_frames"] == 34 for c in chunks))
        self.assertTrue(all(c["extended_av_padding_frames"] >= 34 for c in chunks))
        # Every non-final parent ends its editorial range BEFORE the physical
        # end of the raw sample, and that endpoint remains on the 17f phase so
        # _tail_from_sidecar can inherit the delivered tail, not the runway.
        for chunk in chunks[:-1]:
            endpoint = chunk["extended_av_context_prefix_frames"] + chunk["unique_frames"]
            overlap = result["extended_av"]["overlap_frames"]
            source_start = endpoint - overlap
            self.assertEqual(source_start % 17, 0)
            self.assertEqual(
                chunk["frame_count"] - endpoint,
                chunk["extended_av_padding_frames"],
            )
            self.assertGreaterEqual(chunk["frame_count"] - endpoint, 34)

    def test_boundary_runway_is_vram_profile_independent(self):
        for profile in ("safe_8_12gb", "balanced_12_16gb", "quality_20_24gb", "max_32gb_plus"):
            result = compile_extended([image_row(1, 0, 720)], 720, extended_av_profile=profile)
            self.assertEqual(result["extended_av"]["boundary_runway_frames"], 34)
            self.assertEqual(sum(c["unique_frames"] for c in result["chunks"]), 720)
            self.assertTrue(all(c["extended_av_boundary_runway_frames"] == 34 for c in result["chunks"]))

    def test_later_authored_image_remains_later_guide_not_root_reinjection(self):
        result = compile_extended(
            [image_row(1, 0, 250), image_row(2, 250, 250)],
            500,
        )
        hits = [
            (i, guide)
            for i, chunk in enumerate(result["chunks"])
            for guide in chunk.get("guides", [])
            if str(guide.get("id")) == "extended_2"
        ]
        self.assertEqual(len(hits), 1)
        index, guide = hits[0]
        self.assertGreater(index, 0)
        self.assertEqual(int(guide["global_frame"]), 250)
        self.assertGreaterEqual(int(guide["local_frame"]), 39)

    def test_custom_profile_rejects_non_shared_nonfinal_window(self):
        with self.assertRaises(ValueError):
            compile_extended(
                [image_row(1, 0, 400)], 400,
                extended_av_profile="custom",
                extended_av_custom_window_frames=209,
                extended_av_custom_overlap_frames=39,
            )

    def test_extended_av_root_owns_delivery_trim_for_hidden_runway(self):
        source = (ROOT / "iamccs_minimax_h3_atomic_backend.py").read_text(encoding="utf-8")
        # R47.1 regression guard: the root has no incoming masked prefix, but
        # still owns a hidden terminal runway and therefore must pass through
        # the same Extended-AV delivery crop as continuation chunks.
        self.assertIn('"fl2va_extended_av"\n                    if extended_requested', source)
        self.assertIn('elif extended_requested:', source)
        self.assertIn('"extended_av_role": "extend" if extended_active else "root"', source)
        self.assertIn('native_frames = native_frames[trim_head:trim_head + export_frames, ...]', source)

    def test_r38b_master_butt_joins_already_trimmed_takes(self):
        source = (ROOT / "iamccs_minimax_h3_pixel_refine_variant.py").read_text(encoding="utf-8")
        self.assertIn("R38B Extended AV EXTEND-style master complete", source)
        self.assertIn("mode=masked_trim_butt_join", source)
        self.assertIn('audio_join_policy="extend_soft_av"', source)
        self.assertIn("extend_boundary_polish_ms", source)
        self.assertIn("extend_boundary_polish_strength", source)
        self.assertIn('"extended_av_masked_trim"', source)
        self.assertIn("decoded_crossfade=off", source.lower())
        self.assertNotIn("R38B Extended AV master complete | chunks=%d | mode=direct", source)


    def test_extend_audio_master_is_sample_exact_pcm_then_single_aac(self):
        source = (ROOT / "iamccs_minimax_h3_shotboard.py").read_text(encoding="utf-8")
        # Take saver must match the external reference path: native float PCM
        # straight to AAC, with no intermediate int16 WAV/async resample.
        self.assertIn("def _write_extend_audio_raw_float", source)
        self.assertIn('audio_policy="extend_exact_float"', source)
        self.assertIn("native_float_pcm_direct_aac", source)
        # Master decode/assembly keeps PyAV/native PCM and now prefers the
        # child hidden-head Soft-AV audio as a time-corresponding seam source.
        self.assertIn("def _decode_extend_audio_part_obvpm", source)
        self.assertIn("av.audio.resampler.AudioResampler", source)
        self.assertIn("def _build_extend_audio_wave_master", source)
        self.assertIn("np.cos(np.linspace(0.0, np.pi / 2.0, n", source)
        self.assertIn("def _mux_extend_video_audio_obvpm", source)
        self.assertIn("av.AudioFrame.from_ndarray", source)
        self.assertIn('audio_join_policy: str = "legacy_filter"', source)
        self.assertIn("extend_soft_av", source)
        self.assertIn("hidden_child_context_qsin", source)
        self.assertIn("_extend_soft_audio_context_path", source)
        self.assertIn("_save_extend_soft_audio_context", source)
        self.assertIn("_load_extend_soft_audio_context", source)
        self.assertIn("def _resample_extend_soft_audio_context", source)
        # EXTEND checkpoint tail-matches decoded H3 audio to delivered frames.
        self.assertIn("Match OBVPM H3TrimPinned even for the root", source)


    def test_r38b_preserves_extend_native_audio_rate_and_float_path(self):
        source = (ROOT / "iamccs_minimax_h3_pixel_refine_variant.py").read_text(encoding="utf-8")
        # EXTEND must not fall back through the generic 48 kHz async-resample
        # segment saver, otherwise the hidden 32 kHz Soft-AV context cannot be
        # used at the master seam. Other modes keep the legacy path.
        self.assertIn('audio_policy="extend_exact_float" if extended_mode else "legacy"', source)
        self.assertIn('exact_audio=extended_mode', source)
        self.assertIn('def _finish_segment(intermediate, output, audio, trim, frames, width, height, fps, *, exact_audio=False)', source)
        self.assertIn('def _rtx_finish_segment(intermediate, output, audio, trim, frames, crop_width, crop_height,', source)
        self.assertIn('_write_extend_audio_raw_float', source)

    def test_checkpoint_contract_is_masked_trim_not_same_time_overlap(self):
        source = (ROOT / "iamccs_minimax_h3_shotboard.py").read_text(encoding="utf-8")
        self.assertIn('"mode": "masked_extend"', source)
        self.assertIn('"overlap_frames": 0', source)
        self.assertIn("must not retain decoded overlap", source)
        self.assertIn("bridge cache suppressed", source)
        self.assertIn('_join_label = "masked_extend"', source)

    def test_atomic_runtime_has_no_motion_reference_second_authority(self):
        source = (ROOT / "iamccs_minimax_h3_atomic_backend.py").read_text(encoding="utf-8")
        self.assertIn('"config": None', source)
        self.assertIn("EXTEND-style masked EXTEND continuation", source)
        self.assertIn("pinned_head_trimmed", source)
        self.assertIn("decoded_overlap_for_master=0f", source)


    def test_extend_phase92_captures_hidden_child_audio_before_trim(self):
        atomic = (ROOT / "iamccs_minimax_h3_atomic_backend.py").read_text(encoding="utf-8")
        contract = (ROOT / "iamccs_minimax_h3_extended_av.py").read_text(encoding="utf-8")
        self.assertIn('_iamccs_extended_soft_audio', atomic)
        self.assertIn('hidden Soft-AV audio context captured', atomic)
        self.assertIn('soft_audio_handover_ms', contract)
        self.assertIn('audio_master_policy', contract)
        # Video delivery contract stays MASKED + EXACT / trimmed-head.
        self.assertIn('decoded_overlap_for_master=0f', atomic)

    def test_extended_editor_delivery_is_final_master_only_without_changing_other_modes(self):
        source = (ROOT / "iamccs_minimax_h3_shotboard.py").read_text(encoding="utf-8")
        self.assertIn('editor_longvid = plan_mode.startswith("longvid") or plan_mode == "fl2va_extended_av"', source)
        self.assertIn('Editor delivery: EXTENDED AV single final master', source)
        self.assertIn('audio_join_policy="extend_soft_av"', source)
        # Independent modes must retain the generic per-shot path.
        self.assertIn('Editor delivery: independent take T{current_segment + 1:02d}', source)



class ExtendedAVJointGeometryTests(unittest.TestCase):
    def test_joint_plan_is_reserved_and_bounded(self):
        windows = EXT.joint_window_plan(500, 192, 39)
        self.assertGreater(len(windows), 1)
        self.assertEqual(windows[0]["start_frame"], 0)
        self.assertEqual(windows[-1]["end_frame"], 500)
        self.assertEqual(windows[1]["start_frame"], 153)
        self.assertTrue(all(w["end_frame"] > w["start_frame"] for w in windows))


class ExtendedAVSettingsSchemaTests(unittest.TestCase):
    def test_settings_extended_fields_are_true_append_only(self):
        source = (ROOT / "iamccs_minimax_h3_shotboard.py").read_text(encoding="utf-8")
        self.assertIn("and name not in _H3_EXTENDED_AV_SETTINGS_NODE_FIELDS", source)
        self.assertIn("**extended_av_optional", source)
        self.assertGreater(source.rfind("**extended_av_optional"), source.rfind('"h3_fast_latent_stage2_overlap"'))


if __name__ == "__main__":
    unittest.main()

class ExtendedAVPhaseBAudioAndAutoTests(unittest.TestCase):
    def test_phase_b_audio_polish_contract_is_audio_only_and_clamped(self):
        contract = EXT.resolve_contract(
            soft_audio_handover_ms=22.0,
            boundary_polish_ms=4.5,
            boundary_polish_strength=0.75,
        )
        self.assertEqual(contract["soft_audio_handover_ms"], 22.0)
        self.assertEqual(contract["boundary_polish_ms"], 4.5)
        self.assertEqual(contract["boundary_polish_strength"], 0.75)
        self.assertEqual(contract["pin_mode"], "masked")
        self.assertEqual(contract["mask_profile"], "exact")
        self.assertEqual(contract["delivery_policy"], "trim_pinned_head_then_butt_join")
        self.assertFalse(contract["decoded_overlap"])

    def test_phase_b_ref2va_can_own_root_then_children_return_to_t2va(self):
        result = CORE.build_shotplan(
            timeline_data={"rows": [], "fps": 24, "duration_seconds": 20.0},
            global_prompt="A subject performs continuously while references preserve identity.",
            duration_seconds=20.0,
            task_mode="fl2va_extended_av",
            width=960,
            height=544,
            extended_av_root_task="ref2va",
            extended_av_profile="balanced_12_16gb",
        )
        self.assertGreaterEqual(len(result["chunks"]), 2)
        self.assertEqual(result["chunks"][0]["task_mode"], "ref2va")
        self.assertTrue(all(chunk["task_mode"] == "t2va" for chunk in result["chunks"][1:]))
        self.assertEqual(result["extended_av_root_task"], "ref2va")
