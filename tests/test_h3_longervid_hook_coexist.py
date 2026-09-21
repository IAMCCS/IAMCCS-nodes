"""LongerVid must compose only with the published Motion Context hook ABI."""

import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "iamccs_longervid_hook_test"


def _modules():
    package = sys.modules.get(PACKAGE)
    if package is None:
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(ROOT / "iamccs_h3_continuous_av")]
        sys.modules[PACKAGE] = package
    payload = importlib.import_module(PACKAGE + ".patch_payload")
    layout = importlib.import_module(PACKAGE + ".patch_layout")
    runtime = importlib.import_module(PACKAGE + ".runtime_patches")
    return payload, layout, runtime


def _reset_payload(module):
    module._ORIGINAL_EXTRA_CONDS = None
    module._APPLIED = False
    module._MODEL_BASE = None


class LongerVidHookCoexistTests(unittest.TestCase):
    def tearDown(self):
        payload, _, _ = _modules()
        _reset_payload(payload)

    def test_longervid_composes_over_avbank_motion_context_payload(self):
        payload, _, runtime = _modules()
        _reset_payload(payload)

        class Holder:
            def __init__(self):
                self.cond = {"cond_video_latents": ["wrong"]}

        def avbank_owner(self, **kwargs):
            return {"minimax_payload": Holder()}

        avbank_owner._h3_motion_context_payload_patch = True
        avbank_owner._h3_avbank_merge = True

        class MiniMaxH3:
            extra_conds = avbank_owner

        fake_model_base = types.SimpleNamespace(MiniMaxH3=MiniMaxH3)
        with patch.object(payload, "_import_model_base", return_value=fake_model_base):
            status, error = payload.get_payload_patch_status()
            self.assertIsNone(error)
            self.assertEqual(status.state, "foreign")
            self.assertTrue(payload.is_compatible_foreign_owner())
            self.assertIsNone(runtime._conflict_message("payload", status, error, True))
            self.assertTrue(payload.install_payload_patch())

            live = MiniMaxH3.extra_conds
            self.assertTrue(getattr(live, payload.PAYLOAD_PATCH_MARKER, False))
            self.assertTrue(getattr(live, "_h3_motion_context_payload_patch", False))
            self.assertTrue(getattr(live, "_h3_avbank_merge", False))

            video_a = torch.ones(1)
            video_b = torch.ones(1) * 2
            audio = torch.ones(1) * 3
            result = live(
                object(),
                minimax_keyframes=[{payload.HC_INDEX: 0, "latent": video_a}],
                minimax_refs=[{
                    payload.HC_AUDIO_END_FRAME: 5,
                    "latent": video_b,
                    "audio_latent": audio,
                }],
                minimax_frame_count=73,
            )
            cond = result["minimax_payload"].cond
            self.assertIs(cond["cond_video_latents"][0], video_a)
            self.assertIs(cond["cond_video_latents"][1], video_b)
            self.assertIs(cond["cond_audio_latents"][0], audio)
            self.assertEqual(cond["frame_count"], 73)


    def test_longervid_still_rejects_unknown_payload_owner(self):
        payload, _, _ = _modules()
        _reset_payload(payload)

        def unknown_owner(self, **kwargs):
            return {}

        unknown_owner.__module__ = "unknown_h3_patch"

        class MiniMaxH3:
            extra_conds = unknown_owner

        fake_model_base = types.SimpleNamespace(MiniMaxH3=MiniMaxH3)
        with patch.object(payload, "_import_model_base", return_value=fake_model_base):
            self.assertFalse(payload.is_compatible_foreign_owner())
            self.assertFalse(payload.install_payload_patch())
            self.assertIs(MiniMaxH3.extra_conds, unknown_owner)


    def test_longervid_recognises_motion_context_layout_abi(self):
        _, layout, runtime = _modules()

        def motion_context_layout(self, *args, **kwargs):
            return None

        motion_context_layout._h3_motion_context_layout_patch = True

        class PackedLayout:
            __init__ = motion_context_layout

        fake_mm = types.SimpleNamespace(PackedLayout=PackedLayout)
        with patch.object(layout, "_import_mm", return_value=fake_mm):
            status, error = layout.get_layout_patch_status()
            self.assertIsNone(error)
            self.assertEqual(status.state, "foreign")
            self.assertTrue(layout.is_compatible_foreign_owner())
            self.assertIsNone(runtime._conflict_message("layout", status, error, True))


if __name__ == "__main__":
    unittest.main()
