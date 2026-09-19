import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).parents[1]
COMFY = ROOT.parents[1]
sys.path.insert(0, str(COMFY))
package = types.ModuleType("iamccs_scout_testpkg")
package.__path__ = [str(ROOT)]
sys.modules["iamccs_scout_testpkg"] = package
MODULE = importlib.import_module("iamccs_scout_testpkg.iamccs_minimax_h3_seed_scout_variant")


class ScoutDeliveryRouterTests(unittest.TestCase):
    def selection(self, task_mode, enabled=True):
        plan = {"task_mode": task_mode, "seed_scout_settings": {"enabled": enabled}}
        with patch.object(MODULE, "_resolve_shotplan", return_value=plan):
            return MODULE.IAMCCS_MiniMaxH3ScoutDeliveryLazyRouterR43._selection({})

    def test_ref2vid_remains_scout_eligible(self):
        prefix, reason = self.selection("ref2vid_lipsync")
        self.assertEqual(prefix, "scout")
        self.assertIn("ref2vid_lipsync", reason)

    def test_joint_and_continuous_engines_preserve_original_branch(self):
        for mode in ("keyframe_joint_native", "latent_go_ahead", "longvid_continuous_guided", "viggle_animation"):
            prefix, reason = self.selection(mode)
            self.assertEqual(prefix, "original")
            self.assertIn("latent history", reason)

    def test_disabled_scout_is_lazy_original(self):
        prefix, reason = self.selection("i2va", enabled=False)
        self.assertEqual((prefix, reason), ("original", "Scout disabled"))


if __name__ == "__main__":
    unittest.main()
