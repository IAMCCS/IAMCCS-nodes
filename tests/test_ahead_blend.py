import importlib.util
from pathlib import Path
import unittest

import torch


MODULE_PATH = Path(__file__).parents[1] / "iamccs_ahead_blend.py"
SPEC = importlib.util.spec_from_file_location("iamccs_ahead_blend_tested", MODULE_PATH)
BLEND = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BLEND)

LGA_PATH = Path(__file__).parents[1] / "iamccs_minimax_h3_latent_go_ahead.py"
LGA_SPEC = importlib.util.spec_from_file_location("iamccs_lga_settings_tested", LGA_PATH)
LGA = importlib.util.module_from_spec(LGA_SPEC)
LGA_SPEC.loader.exec_module(LGA)


class AheadBlendTests(unittest.TestCase):
    def test_legacy_empty_audio_widgets_migrate_before_render(self):
        self.assertEqual(LGA.delivery_audio_settings("", ""), ("click_safe", 20))
        self.assertEqual(LGA.delivery_audio_settings("hard_cut", "999"), ("hard_cut", 250))

    def test_video_overlap_exposes_successor_frame_zero_immediately(self):
        first = torch.arange(6, dtype=torch.float32).reshape(6, 1, 1, 1)
        second = torch.arange(10, 16, dtype=torch.float32).reshape(6, 1, 1, 1)
        audio = [torch.arange(6, dtype=torch.float32).reshape(1, 1, 6),
                 torch.arange(10, 16, dtype=torch.float32).reshape(1, 1, 6)]

        video, _ = BLEND.assemble([first, second], audio, 24, fps=24, mode="linear", overlap=2)

        # Four untouched frames, then two interior blends. The first blend already
        # contains successor frame 0; the old 0..1 ramp returned exactly frame 4.
        self.assertEqual(video.shape[0], 10)
        self.assertTrue(4 < video[4].item() < 10)
        self.assertTrue(5 < video[5].item() < 11)


    def test_click_safe_audio_preserves_duration_and_removes_sample_edge_jump(self):
        first = torch.ones((1, 1, 100), dtype=torch.float32)
        second = -torch.ones((1, 1, 100), dtype=torch.float32)

        audio = BLEND.click_safe_audio([first, second], sample_rate=1000, milliseconds=10)

        self.assertEqual(audio.shape[-1], 200)
        self.assertLess(audio[..., 99].abs().item(), 1e-6)
        self.assertLess(audio[..., 100].abs().item(), 1e-6)
        self.assertTrue(first.eq(1).all() and second.eq(-1).all())


    def test_cut_with_click_safe_audio_keeps_every_video_frame(self):
        clips = [torch.zeros((8, 1, 1, 1)), torch.ones((8, 1, 1, 1))]
        waves = [torch.ones((1, 1, 80)), -torch.ones((1, 1, 80))]

        video, audio = BLEND.assemble(
            clips, waves, sample_rate=240, fps=24, mode="none", overlap=9,
            audio_join="click_safe", audio_smoothing_ms=20)

        self.assertEqual(video.shape[0], 16)
        self.assertEqual(audio.shape[-1], 160)

    def test_periodic_flash_guard_repairs_spike_and_preserves_source_and_length(self):
        clip = torch.linspace(0, 0.2, 30).reshape(30, 1, 1, 1).expand(-1, 8, 8, 3).clone()
        clip[6] = 1.0  # transition 5 -> 6, the first H3 cadence candidate
        source = clip.clone()

        treated, audit = BLEND.periodic_flash_guard([clip], mode="auto", sensitivity=1.4, radius=2)

        self.assertEqual(treated[0].shape, source.shape)
        self.assertTrue(torch.equal(clip, source))
        self.assertEqual(len(audit), 1)
        self.assertLess(treated[0][6].mean().item(), 0.5)

    def test_periodic_flash_guard_does_not_flatten_regular_motion(self):
        clip = torch.linspace(0, 1, 40).reshape(40, 1, 1, 1).expand(-1, 4, 4, 3).clone()

        treated, audit = BLEND.periodic_flash_guard([clip], mode="auto", sensitivity=1.8, radius=3)

        self.assertEqual(audit, [])
        self.assertTrue(torch.equal(treated[0], clip))


if __name__ == "__main__":
    unittest.main()
