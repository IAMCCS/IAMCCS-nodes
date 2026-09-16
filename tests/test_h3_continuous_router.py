import importlib.util
from pathlib import Path
import sys
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "iamccs_continuous_router_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package

atomic = types.ModuleType(f"{PACKAGE}.iamccs_minimax_h3_atomic_backend")
atomic.SUPERNODE_LINX_TYPE = "CINE_LINX"
atomic._resolve_shotplan = lambda value: value
sys.modules[atomic.__name__] = atomic

spec = importlib.util.spec_from_file_location(
    f"{PACKAGE}.router", ROOT / "iamccs_minimax_h3_continuous_router.py"
)
router_module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(router_module)


def payload(prefix):
    return {
        f"{prefix}_frames": f"{prefix}-frames",
        f"{prefix}_audio": f"{prefix}-audio",
        f"{prefix}_bridge": f"{prefix}-bridge",
        f"{prefix}_latent": f"{prefix}-latent",
        f"{prefix}_fps": 24,
        f"{prefix}_report": f"{prefix}-report",
    }


class ContinuousRouterTests(unittest.TestCase):
    def setUp(self):
        self.router = router_module.IAMCCS_MiniMaxH3ContinuousBackendLazyRouterR42()

    def test_existing_r42_keeps_segment_metadata(self):
        kwargs = payload("r42")
        kwargs.update(r42_current_segment=2, r42_total_segments=4, r42_trim_head_frames=1)
        result = self.router.select({"task_mode": "i2va"}, **kwargs)
        self.assertEqual(result[:6], tuple(kwargs[f"r42_{name}"] for name in ("frames", "audio", "bridge", "latent", "fps", "report")))
        self.assertEqual(result[6:], (2, 4, 1))

    def test_continuous_av_is_one_outer_master(self):
        kwargs = payload("continuous")
        result = self.router.select({"task_mode": "longvid_masked_loop_guided"}, **kwargs)
        self.assertEqual(result[0], "continuous-frames")
        self.assertEqual(result[6:], (0, 1, 0))

    def test_guided_loop_is_one_outer_master(self):
        kwargs = payload("guided")
        result = self.router.select({"task_mode": "guided_av_loop_experimental"}, **kwargs)
        self.assertEqual(result[0], "guided-frames")
        self.assertEqual(result[6:], (0, 1, 0))

    def test_lazy_status_requests_only_selected_branch(self):
        missing = self.router.check_lazy_status({"task_mode": "guided_av_loop_experimental"})
        self.assertEqual(missing, [
            "guided_frames", "guided_audio", "guided_bridge", "guided_latent",
            "guided_fps", "guided_report",
        ])


if __name__ == "__main__":
    unittest.main()
