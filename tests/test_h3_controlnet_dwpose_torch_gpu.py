import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).parents[1]
PACKAGE_NAME = "iamccs_h3_controlnet_dwpose_test_package"
PACKAGE = types.ModuleType(PACKAGE_NAME)
PACKAGE.__path__ = [str(ROOT)]
PREVIS = types.ModuleType(f"{PACKAGE_NAME}.iamccs_h3_previs")
PREVIS.IAMCCS_H3PrevisControl = type("IAMCCS_H3PrevisControl", (), {})
sys.modules[PACKAGE_NAME] = PACKAGE
sys.modules[PREVIS.__name__] = PREVIS
SPEC = importlib.util.spec_from_file_location(
    f"{PACKAGE_NAME}.iamccs_cine_h3_bus", ROOT / "iamccs_cine_h3_bus.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
with mock.patch.dict(sys.modules, {"folder_paths": types.ModuleType("folder_paths")}):
    SPEC.loader.exec_module(MODULE)


class DWPoseTorchGpuTests(unittest.TestCase):
    def test_settings_toggle_is_read_from_cinelinx(self):
        cine_linx = {"resources": {"iamccs_minimax_h3_settings": {
            "settings": {"h3_controlnet_dwpose_torch_gpu": True}
        }}}
        self.assertTrue(MODULE.IAMCCS_CineH3FunControlInput._settings_dwpose_torch_gpu(cine_linx))
        self.assertFalse(MODULE.IAMCCS_CineH3FunControlInput._settings_dwpose_torch_gpu({}))

    def test_preprocess_resolution_uses_settings_or_preserves_node_default(self):
        selector = MODULE.IAMCCS_CineH3FunControlInput._settings_preprocess_resolution
        self.assertEqual(selector({}, 512), 512)
        for choice in ("node_default", "412", "768", "1024"):
            cine_linx = {"resources": {"iamccs_minimax_h3_settings": {
                "settings": {"h3_controlnet_preprocess_resolution": choice}
            }}}
            self.assertEqual(selector(cine_linx, 512), 512 if choice == "node_default" else int(choice))

    def test_inject_passes_settings_resolution_to_preprocessor(self):
        frames = torch.zeros((1, 64, 64, 3))
        cine_linx = {"resources": {"iamccs_minimax_h3_settings": {
            "settings": {"h3_controlnet_kind": "pose_dwpose", "h3_controlnet_preprocess_resolution": "768"}
        }}}
        with mock.patch.object(MODULE.IAMCCS_CineH3FunControlInput, "_preprocess", return_value=frames) as preprocess:
            result = MODULE.IAMCCS_CineH3FunControlInput().inject(
                cine_linx, 24.0, "from_iamccs_settings", 512, source_video=frames
            )
        _output = result["result"][0]
        self.assertEqual(preprocess.call_args.args[2], 768)
        self.assertEqual(_output["resources"]["iamccs_minimax_h3_control_video_meta"]["preprocess_resolution"], 768)

    def test_torch_route_selects_both_torchscript_models(self):
        calls = []
        frames = torch.zeros((1, 64, 64, 3))

        class FakeDW:
            def estimate_pose(self, **kwargs):
                calls.append(kwargs)
                return frames, None

        fake_nodes = types.SimpleNamespace(NODE_CLASS_MAPPINGS={"DWPreprocessor": FakeDW})
        with mock.patch.dict(sys.modules, {"nodes": fake_nodes}), mock.patch.object(torch.cuda, "is_available", return_value=True):
            result = MODULE.IAMCCS_CineH3FunControlInput._preprocess(frames, "dwpose", 512, True)
        self.assertIs(result, frames)
        self.assertEqual(calls[0]["bbox_detector"], "yolox_l.torchscript.pt")
        self.assertEqual(calls[0]["pose_estimator"], "dw-ll_ucoco_384_bs5.torchscript.pt")

    def test_torch_route_fails_clearly_without_cuda(self):
        frames = torch.zeros((1, 64, 64, 3))
        with mock.patch.dict(sys.modules, {"nodes": types.SimpleNamespace(NODE_CLASS_MAPPINGS={})}), mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                MODULE.IAMCCS_CineH3FunControlInput._preprocess(frames, "dwpose", 512, True)

    def test_off_preserves_existing_aio_route(self):
        calls = []
        frames = torch.zeros((1, 64, 64, 3))

        class FakeAIO:
            def execute(self, **kwargs):
                calls.append(kwargs)
                return (frames,)

        fake_nodes = types.SimpleNamespace(NODE_CLASS_MAPPINGS={"AIO_Preprocessor": FakeAIO})
        with mock.patch.dict(sys.modules, {"nodes": fake_nodes}):
            result = MODULE.IAMCCS_CineH3FunControlInput._preprocess(frames, "dwpose", 512, False)
        self.assertIs(result, frames)
        self.assertEqual(calls[0]["preprocessor"], "DWPreprocessor")


if __name__ == "__main__":
    unittest.main()
