import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).parents[1]
PACKAGE = "iamccs_motion_context_test"

package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package

atomic = types.ModuleType(f"{PACKAGE}.iamccs_minimax_h3_atomic_backend")
atomic.H3_FPS = 24
atomic.SUPERNODE_LINX_TYPE = "CINE_LINX"


class AtomicBackendStub:
    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "CINE_LINX")
    RETURN_NAMES = ("frames", "audio", "report", "cine_linx")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}


atomic.IAMCCS_MiniMaxH3GenerationBackendV2 = AtomicBackendStub
atomic._audio_slice = lambda *args, **kwargs: None
atomic._load_image = lambda *args, **kwargs: None
atomic._load_timeline_audio = lambda *args, **kwargs: None
atomic._resolve_shotplan = lambda value: value
sys.modules[atomic.__name__] = atomic

path = ROOT / "iamccs_minimax_h3_motion_context_variant.py"
spec = importlib.util.spec_from_file_location(f"{PACKAGE}.variant", path)
variant = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(variant)


class MotionContextRoutingTests(unittest.TestCase):
    def test_chain_config_includes_new_provider_trim_contract(self):
        config = variant._chain_config("render_1", 1, effective_trim_frames=22, fps=24)
        self.assertEqual(config["effective_trim_frames"], 22)
        self.assertEqual(config["clip_index"], 2)
        self.assertEqual(config["fps"], 24.0)

    def test_provider_call_does_not_let_provider_mutate_original_chain_config(self):
        config = variant._chain_config(
            "render_1", 1, effective_trim_frames=22, fps=24
        )

        class MutatingProvider:
            def load(self, chain_config):
                chain_config.pop("effective_trim_frames", None)
                chain_config["fps"] = 999
                return (None,)

        with patch.object(variant, "_provider_node", return_value=MutatingProvider):
            variant._provider_call(
                "MiniMaxH3AutoChainLoadLatent",
                "load",
                chain_config=config,
            )

        self.assertEqual(config["effective_trim_frames"], 22)
        self.assertEqual(config["fps"], 24.0)

    def test_provider_call_repairs_missing_effective_trim_frames_for_new_provider(self):
        legacy_config = {
            "chain_id": "render_1",
            "latent_prefix": "h3_context/render_1_clip",
            "load_clip_index": 1,
            "save_clip_index": 2,
            "reset": False,
        }

        class NewProvider:
            def apply(self, chain_config):
                self.seen = chain_config
                return (chain_config["effective_trim_frames"],)

        provider_class = NewProvider
        with patch.object(variant, "_provider_node", return_value=provider_class):
            result = variant._provider_call(
                "MiniMaxH3AutoChainMotionContext",
                "apply",
                chain_config=legacy_config,
                context_length="22",
                fps=24.0,
            )

        self.assertEqual(result[0], 22)
        self.assertNotIn("effective_trim_frames", legacy_config)

    def test_enabled_contract_selects_motion_context_even_without_variant_mirror(self):
        plan = {
            "task_mode": "longvid_motion_context",
            "motion_context_auto_chain": {"enabled": True},
        }
        self.assertTrue(variant._variant_active(plan))

    def test_longvid_never_silently_falls_back_to_t2v(self):
        with self.assertRaisesRegex(RuntimeError, "refusing to fall back to T2V"):
            variant._variant_active({"task_mode": "longvid_motion_context"})

    def test_continuous_guided_selects_isolated_motion_context_branch(self):
        plan = {
            "task_mode": "longvid_continuous_guided",
            "backend_variant": "motion_context_auto_chain_v1",
            "motion_context_auto_chain": {"enabled": True},
        }
        self.assertTrue(variant._variant_active(plan))

    def test_continuous_guided_never_silently_falls_back_to_t2v(self):
        with self.assertRaisesRegex(RuntimeError, "refusing to fall back to T2V"):
            variant._variant_active({"task_mode": "longvid_continuous_guided"})

    def test_declared_variant_requires_enabled_contract(self):
        with self.assertRaisesRegex(RuntimeError, "without an enabled"):
            variant._variant_active({"backend_variant": "motion_context_auto_chain_v1"})

    def test_shotboard_sampling_is_truth_over_stale_r37_node_widgets(self):
        resolved = variant._sampling_from_plan(
            {"sampling": {
                "seed": 101,
                "seed_stride": 3,
                "steps": 12,
                "sampler_name": "res_multistep",
                "scheduler": "simple",
                "denoise": 0.9,
                "shift_video": 12,
                "shift_audio": 3,
            }},
            seed=1,
            seed_stride=1,
            steps=8,
            sampler_name="euler",
            scheduler="normal",
            denoise=1.0,
            shift_video=1,
            shift_audio=1,
        )
        self.assertEqual(resolved["seed"], 101)
        self.assertEqual(resolved["seed_stride"], 3)
        self.assertEqual(resolved["steps"], 12)
        self.assertEqual(resolved["sampler_name"], "res_multistep")
        self.assertEqual(resolved["scheduler"], "simple")
        self.assertAlmostEqual(resolved["denoise"], 0.9)
        self.assertEqual(resolved["shift_video"], 12.0)
        self.assertEqual(resolved["shift_audio"], 3.0)

    def test_old_direct_nodes_keep_their_sampler_values_without_plan_sampling(self):
        resolved = variant._sampling_from_plan(
            {}, seed=7, seed_stride=1, steps=8, sampler_name="euler",
            scheduler="simple", denoise=1.0, shift_video=12, shift_audio=3,
        )
        self.assertEqual(resolved["seed"], 7)
        self.assertEqual(resolved["steps"], 8)
        self.assertEqual(resolved["sampler_name"], "euler")

    def test_continuation_does_not_reanchor_the_active_image_after_native_tail(self):
        guide = {
            "kind": "image",
            "continued_from_previous_chunk": True,
            "local_frame": 0,
        }
        self.assertTrue(variant._guide_uses_native_tail(guide, 22))
        self.assertFalse(variant._guide_uses_native_tail(guide, 0))

    def test_future_image_guide_is_not_suppressed(self):
        guide = {
            "kind": "image",
            "continued_from_previous_chunk": False,
            "local_frame": 140,
        }
        self.assertFalse(variant._guide_uses_native_tail(guide, 22))


if __name__ == "__main__":
    unittest.main()
