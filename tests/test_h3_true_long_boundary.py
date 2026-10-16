"""Regression for the inclusive PSkf endpoint and the shared native AV cut."""
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_h3_authored_continuous as core

package = types.ModuleType("true_long_boundary_test")
package.__path__ = [str(core.ROOT)]
sys.modules[package.__name__] = package
sys.modules[package.__name__ + ".iamccs_h3_continuous_av"] = sys.modules[core.package.__name__]
sys.modules[package.__name__ + ".iamccs_h3_continuous_av.latent_math"] = core.M
sys.modules[package.__name__ + ".iamccs_h3_continuous_av.nodes"] = core.N
B = importlib.import_module(package.__name__ + ".iamccs_h3_true_long")
P = importlib.import_module(package.__name__ + ".iamccs_h3_pskf")


class BoundaryTests(unittest.TestCase):
    def test_every_endpoint_has_zero_replayed_frames(self):
        for context in (5, 22, 39):
            for end in range(60, 174):
                images = torch.arange(end + 1).view(-1, 1, 1, 1).expand(-1, 4, 4, 3).float()
                h = B.terminal_boundary(images, {}, core.M.video_latent_t(175), context, 0)
                sl = core.M.phase_aligned_extended_context_slice(
                    core.M.video_latent_t(175), context, ideal_last_frame=h["handover_end_frame"])
                timing = core.R.authored_interval_timing(120, sl, h["visible_end_frame"])
                self.assertEqual(timing["replayed_frames"], 0)
                self.assertEqual(h["handover_end_frame"] + 1, sl["source_end_frame"])

    def test_short_terminal_hold_removed_but_moving_end_preserved(self):
        moving = torch.arange(60).view(-1, 1, 1, 1).expand(-1, 8, 8, 3).float() / 60
        held = torch.cat((moving, moving[-1:].repeat(3, 1, 1, 1)))
        result = B.terminal_boundary(held, {}, core.M.video_latent_t(73), 22, 0, final=True)
        self.assertEqual(result["handover_end_frame"], 59)
        one_duplicate = torch.cat((moving, moving[-1:]))
        result = B.terminal_boundary(one_duplicate, {}, core.M.video_latent_t(73), 22, 0, final=True)
        self.assertEqual(result["handover_end_frame"], 59)
        self.assertEqual(result["short_hold_trim_frames"], 1)
        result = B.terminal_boundary(moving, {}, core.M.video_latent_t(73), 22, 0, final=True)
        self.assertEqual(result["handover_end_frame"], 59)
        self.assertEqual(result["short_hold_trim_frames"], 0)

    def test_saved_state_continuation_hides_phase_replay_before_exact_visible_interval(self):
        clip, vae = core.Clip(), core.VAE()
        image = torch.zeros(1, 32, 32, 3)
        with patch.object(core.N, "_require_patches"):
            _, previous = core.N.H3ContinuousStartV11().build(
                clip, vae, "move", 32, 32, 5, image, image, authored_visible_frames=120)
            video, _ = previous["samples"].unbind()
            technical_frames = core.M.pixel_frames(video.shape[2])
            end = 119
            handover = {
                "available": True,
                "frame_count": technical_frames,
                "handover_end_frame": end,
                "visible_end_frame": end,
                "landing_tail_frames": technical_frames - end - 1,
                "phase_aligned_target_end_frame": end,
                "phase_aware_target_end_frame": end,
                "detector_mode": "iamccs_authored_editorial_boundary",
            }
            state, _ = P.IAMCCS_H3PSkfPack().pack(
                previous, visible_frame_count=120, handover=handover)
            self.assertEqual(state["handover"]["visible_end_frame"], end)
            _, latent, head, _, _ = P.IAMCCS_H3TrueLongExtend().build(
                clip, vae, state, "continue", 32, 32, 5, context_frames="39",
                last_frame=image, opening_image=image, authored_visible_frames=120)
            timing = latent["iamccs_authored_interval"]
            self.assertEqual(timing["endpoint_frame"] + 1 - head, 120)
            self.assertEqual(head, timing["context_frames"] + timing["replayed_frames"])
            self.assertLess(timing["replayed_frames"], 17)
            self.assertEqual(timing["boundary_runway_frames"], 34)
            self.assertGreaterEqual(timing["technical_frames"] - timing["endpoint_frame"] - 1, 34)


if __name__ == "__main__":
    unittest.main()
