"""R47 Long Keyframe dropdown: real backend families and mode isolation.

CPU-only planner and configuration checks; no model sampling is claimed.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORE = load('iamccs_r47_engine_core', 'iamccs_minimax_h3_shotboard_core.py')
ENGINE = load('iamccs_r47_engine_selector', 'iamccs_minimax_h3_timeline_director_restore.py')


def board(evolving=False):
    result = {
        'rows': [
            {'id': 'first', 'type': 'image', 'start': 0, 'length': 300, 'duration_frames': 300,
             'imageFile': 'first.png', 'prompt': 'Continue moving', 'transition': 'start', 'use_guide': True},
            {'id': 'second', 'type': 'image', 'start': 300, 'length': 420, 'duration_frames': 420,
             'imageFile': 'second.png', 'prompt': 'Movement evolves', 'transition': 'continuous', 'use_guide': True},
        ], 'fps': 24, 'duration_seconds': 30.0}
    if evolving:
        result['extended_prompt_blocks'] = [
            {'id': 'walk', 'start_frame': 0, 'end_frame': 280, 'prompt': '[ONSET ONCE] Begin walking. [THEN SUSTAIN] Keep walking.'},
            {'id': 'turn', 'start_frame': 280, 'end_frame': 720, 'prompt': '[RESOLVED STATE] Turn toward the house.'},
        ]
    return result


def native(overlap=22, engine='pianosequenza_1_native_tail', timeline=None):
    plan = CORE._longvid_guide_plan(
        timeline=timeline or board(), global_prompt='One continuous film.', duration_seconds=30,
        prompt_mapping='global_plus_local', resolved_width=960, resolved_height=544,
        audio_mode='h3_native_generated', acceleration='native', ref_image_size='match',
        text_encoder_device='gpu_auto', roles=['composition']*4,
        reference_video_role='off', reference_audio_role='off', sol_conditioning='exact_kv',
        spectrum_profile='low_vram', vram_clean_before_decode=True, rife_mode='off',
        active_upscale_mode='off', upscale_enabled=False, voice_reference_picture_index=0,
        motion_context_tail_frames=0, latent_tail_context_frames=overlap,
        timeline_director_native=True, motion_context_window_frames=362,
        terminal_endpoint_mode=ENGINE.resolve_continuous_engine(engine)['terminal_endpoint_mode'],
    )
    return ENGINE.annotate_restored_plan(plan, ENGINE.resolve_continuous_engine(engine), overlap_frames=overlap)


class ActualEngineSelectionTests(unittest.TestCase):
    def test_engines_are_five_real_named_choices_with_distinct_families(self):
        self.assertEqual(set(ENGINE.ENGINES), {
            'pianosequenza_1_native_tail', 'extended_av_continuous',
            'pianosequenza_evolving', 'pianosequenza_2_drift', 'pianosequenza_3_phase',
        })
        self.assertEqual(ENGINE.resolve_continuous_engine('extended_av_continuous')['backend_family'], 'extended_av')
        self.assertEqual(ENGINE.resolve_continuous_engine('pianosequenza_evolving')['backend_family'], 'native_tail')

    def test_native_overlap_is_not_hardcoded_and_30s_master_remains_720f(self):
        for overlap, expected in ((22, [362, 209, 193]), (39, [362, 226, 210]), (56, [362, 243, 227])):
            with self.subTest(overlap=overlap):
                plan = native(overlap=overlap)
                delivered = [c['timeline_director_contract']['delivered_segment_frames'] for c in plan['chunks']]
                self.assertEqual(delivered, expected)
                self.assertEqual(plan['mode_contract']['overlap_frames'], overlap)
                self.assertEqual(ENGINE.master_geometry(delivered, overlap)['total_frames'], 720)
                self.assertTrue(plan['mode_contract']['sampled_native_video_audio_tail'])

    def test_extended_is_real_masked_planner_not_native_tail_alias(self):
        plan = CORE.build_shotplan(
            timeline_data=board(evolving=True), global_prompt='One continuous film.',
            duration_seconds=30.0, task_mode='fl2va_extended_av', audio_mode='h3_native_generated',
            width=960, height=544, motion_context_tail_frames=0,
            extended_av_profile='safe_8_12gb', extended_av_root_task='i2va',
            extended_av_pin_mode='masked', extended_av_mask_profile='exact',
        )
        self.assertEqual(plan['task_mode'], 'fl2va_extended_av')
        self.assertIn('extend_style', plan['backend_variant'])
        self.assertEqual(plan['chunks'][0]['task_mode'], 'i2va')
        self.assertEqual(plan['chunks'][1]['task_mode'], 't2va')
        self.assertEqual(plan['extended_av']['overlap_frames'], 39)
        self.assertEqual(plan['chunks'][1]['extended_av_context_prefix_frames'], 39)
        self.assertEqual(plan['chunks'][1]['extended_av_trim_head_frames'], 39)
        self.assertFalse(plan['chunks'][1]['extended_av_retain_overlap'])
        self.assertNotIn('timeline_director_native', plan['mode_contract'])
        self.assertTrue(plan['chunks'][1]['conditioning_beat_bindings'])

    def test_extended_cannot_be_annotated_as_native(self):
        with self.assertRaisesRegex(ValueError, 'own planner'):
            ENGINE.annotate_restored_plan(native(), ENGINE.resolve_continuous_engine('extended_av_continuous'))

    def test_evolving_has_real_timed_conditioning_and_fails_closed(self):
        plan = native(engine='pianosequenza_evolving', timeline=board(evolving=True))
        plan = ENGINE.apply_evolving_native_conditioning(plan, board(evolving=True), 'One continuous film.')
        self.assertTrue(plan['mode_contract']['evolving_conditioning'])
        self.assertEqual(plan['conditioning_schedule']['beat_count'], 2)
        self.assertIn('[EVOLVING TIMED BEATS]', plan['chunks'][0]['prompt'])
        self.assertTrue(any(binding['continued_from_previous_chunk'] for c in plan['chunks'][1:] for binding in c['conditioning_beat_bindings']))
        with self.assertRaisesRegex(ValueError, 'requires timed Extended Prompt'):
            ENGINE.apply_evolving_native_conditioning(native(engine='pianosequenza_evolving'), board(), 'One continuous film.')

    def test_drift_and_phase_select_different_terminal_policies(self):
        for engine, terminal in (
            ('pianosequenza_1_native_tail', 'latent_free_closure'),
            ('pianosequenza_2_drift', 'pianosequenza_drift'),
            ('pianosequenza_3_phase', 'pianosequenza_phase'),
        ):
            self.assertEqual(native(engine=engine)['mode_contract']['terminal_endpoint_mode'], terminal)

    def test_settings_is_append_only_and_choice_reaches_backend(self):
        python = (ROOT / 'iamccs_minimax_h3_shotboard.py').read_text()
        javascript = (ROOT / 'web' / 'iamccs_h3_settings_pro_ui.js').read_text()
        marker = python.rindex('"longvid_continuous_engine": (')
        self.assertLess(python.index('"face_detailer_min_coverage":'), marker)
        self.assertIn('"longvid_continuous_engine", longvid_continuous_engine', python)
        self.assertIn('compiled_task_mode = "fl2va_extended_av"', python)
        self.assertIn('compiled_task_mode = "longvid_guides"', python)
        self.assertIn('apply_evolving_native_conditioning(plan', python)
        self.assertIn('plan["mode_contract"]["selected_engine"] = "extended_av_continuous"', python)
        for name in ENGINE.ENGINES:
            self.assertIn(f'"{name}"', javascript)
        self.assertIn('fieldRelevant(name, mode)', javascript)
        self.assertIn('name.startsWith("extended_av_")', javascript)


if __name__ == '__main__':
    unittest.main()
