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
        for mode in ("longvid_multishot_hardcut", "keyframe_joint_native", "latent_go_ahead", "longvid_continuous_guided", "longvid_keyframe_continuous", "viggle_animation"):
            prefix, reason = self.selection(mode)
            self.assertEqual(prefix, "original")
            self.assertIn("latent history", reason)

    def test_disabled_scout_is_lazy_original(self):
        prefix, reason = self.selection("i2va", enabled=False)
        self.assertEqual((prefix, reason), ("original", "Scout disabled"))


class ScoutBypassTests(unittest.TestCase):
    def plan(self, enabled=False, count=4, mode="t2va"):
        return {"task_mode":mode,"seed_scout_settings":{"enabled":enabled,"candidate_count":count}}

    def test_disabled_preview_does_not_request_any_candidate_dependencies(self):
        candidate=MODULE.IAMCCS_MiniMaxH3SeedCandidateR40()
        with patch.object(MODULE,"_resolve_shotplan",return_value=self.plan()), patch.object(MODULE,"_release_conditioning_models",side_effect=AssertionError("Unexpected model cleanup")), patch.object(MODULE,"_active_sampling_model",side_effect=AssertionError("Unexpected sampling")):
            for index in range(4):
                self.assertEqual(candidate.check_lazy_status({},index),[])
                result=candidate.sample(cine_linx={},candidate_index=index)
                self.assertIsInstance(result[0],MODULE.ExecutionBlocker)
                self.assertIsInstance(result[1],MODULE.ExecutionBlocker)
                self.assertFalse(result[3])

    def test_enabled_candidate_requests_its_heavy_inputs(self):
        candidate=MODULE.IAMCCS_MiniMaxH3SeedCandidateR40()
        with patch.object(MODULE,"_resolve_shotplan",return_value=self.plan(True,2)):
            self.assertEqual(candidate.check_lazy_status({},0),["model","positive","latent","chunk_index"])
            self.assertEqual(candidate.check_lazy_status({},1,model=object(),chunk_index=0),["positive","latent"])
            self.assertEqual(candidate.check_lazy_status({},2),[])
            self.assertIsInstance(candidate.sample(cine_linx={},candidate_index=3)[0],MODULE.ExecutionBlocker)

    def test_temporal_route_bypasses_even_when_scout_requested(self):
        candidate=MODULE.IAMCCS_MiniMaxH3SeedCandidateR40()
        with patch.object(MODULE,"_resolve_shotplan",return_value=self.plan(True,4,"latent_go_ahead")):
            self.assertEqual(candidate.check_lazy_status({},0),[])
            self.assertFalse(candidate.sample(cine_linx={})[3])

    def test_inactive_preview_never_requests_vae_or_allocates_an_image(self):
        preview=MODULE.IAMCCS_MiniMaxH3SeedPreviewR40()
        latent={"iamccs_r40_candidate_active":False}
        self.assertEqual(preview.check_lazy_status(latent),[])
        self.assertIsInstance(preview.decode(latent)[0],MODULE.ExecutionBlocker)
        self.assertEqual(preview.check_lazy_status({"iamccs_r40_candidate_active":True}),["video_vae","trim_frames"])

    def test_existing_socket_names_are_preserved_and_heavy_inputs_are_lazy(self):
        candidate=MODULE.IAMCCS_MiniMaxH3SeedCandidateR40.INPUT_TYPES()["required"]
        for name in ("model","positive","latent","chunk_index"):
            self.assertTrue(candidate[name][1]["lazy"])
        preview=MODULE.IAMCCS_MiniMaxH3SeedPreviewR40.INPUT_TYPES()["required"]
        self.assertTrue(preview["video_vae"][1]["lazy"])
        self.assertTrue(preview["trim_frames"][1]["lazy"])


if __name__ == "__main__":
    unittest.main()
