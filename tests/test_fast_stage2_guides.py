import ast
import importlib.util
from pathlib import Path
import unittest
import sys
import types
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stage2_guides", ROOT / "iamccs_h3_stage2_guides.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def frame_at_latent(k):
    return (k // 5) * 17 + sum((1, 4, 4, 4, 4)[:k % 5])


tree = ast.parse((ROOT / "vendor/mmh3tools_r38b/mmh3tools/nodes_looping_sampler.py").read_text())
namespace = {"frame_at_latent": frame_at_latent}
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_window_encoded_guides")
exec(compile(ast.Module(body=[node], type_ignores=[]), "window_guides", "exec"), namespace)
window_guides = namespace["_window_encoded_guides"]


class Stage2GuidesTest(unittest.TestCase):
    def setUp(self):
        self.video = torch.zeros(1, 2, 32, 2, 2)
        self.guide = {"resolved_frame_index": 68, "latent": torch.ones(1, 2, 1, 2, 2)}

    def test_soft_second_window_rebased(self):
        self.assertEqual(window_guides([self.guide], 34, 90)[0]["resolved_frame_index"], 34)
        self.assertEqual(window_guides([self.guide], 0, 56), [])

    def test_window_boundary_exclusive(self):
        self.assertEqual(window_guides([self.guide], 0, 68), [])
        self.assertEqual(window_guides([self.guide], 68, 100)[0]["resolved_frame_index"], 0)

    def test_overlap_receives_guide(self):
        self.assertEqual(len(window_guides([self.guide], 34, 85)), 1)
        self.assertEqual(len(window_guides([self.guide], 51, 102)), 1)

    def test_masked_video_only_and_no_mutation(self):
        original = torch.ones_like(self.video)
        video, mask, count = helpers.pin_still_guides(self.video, original, [self.guide])
        self.assertEqual(count, 1)
        self.assertTrue(torch.all(video[:, :, 20] == 1))
        self.assertTrue(torch.all(mask[:, :, 20] == 0))
        self.assertEqual(torch.count_nonzero(self.video), 0)
        self.assertTrue(torch.all(original == 1))
        self.assertTrue(torch.all(mask[:, :, :20] == 1))

    def test_non_grid_frame_containing_token(self):
        guide = dict(self.guide, resolved_frame_index=72)
        _, mask, _ = helpers.pin_still_guides(self.video, None, [guide])
        self.assertTrue(torch.all(mask[:, :, 21] == 0))

    def test_duplicate_token_rejected(self):
        with self.assertRaisesRegex(ValueError, "collide"):
            helpers.pin_still_guides(self.video, None, [self.guide, self.guide])

    def test_existing_protection_rejected(self):
        with self.assertRaisesRegex(ValueError, "protected"):
            helpers.pin_still_guides(self.video, torch.zeros_like(self.video), [self.guide])

    def test_repeated_metadata_deduplicated(self):
        cond = [[None, {"minimax_keyframes": [self.guide]}]] * 2
        self.assertEqual(len(helpers.conditioning_guides(cond)), 1)

    def test_stage2_dispatch_preserves_guides_and_audio(self):
        # Execute the real dispatcher with sampler/model boundaries stubbed on CPU.
        sys.path.insert(0, str(ROOT.parents[1]))
        from comfy.nested_tensor import NestedTensor
        src = ast.parse((ROOT / "iamccs_minimax_h3_fast_latent_2pass.py").read_text())
        fn = next(n for n in src.body if isinstance(n, ast.FunctionDef) and n.name == "_sample_stage2")
        audio = torch.randn(1, 2, 2, 40)
        common = types.SimpleNamespace(
            unpack_av=lambda x, label: x["samples"].unbind(),
            pack_av=lambda x, v, a, noise_mask: dict(x, samples=NestedTensor([v, a]), noise_mask=noise_mask),
            latents_to_frames=lambda t: frame_at_latent(t), frame_at_latent=frame_at_latent)
        for mode in ("soft", "masked"):
            for full in (False, True):
                calls = []
                def sample(**kwargs):
                    calls.append(kwargs)
                    latent = kwargs.get("latent", kwargs.get("latent_image"))
                    return latent, latent
                loop = types.SimpleNamespace(
                    per_row_mask_is_continuous=lambda: True,
                    _split_mask=lambda x: x["noise_mask"].unbind(),
                    MMH3LoopingSampler=types.SimpleNamespace(execute=sample))
                windows = types.SimpleNamespace(_plan=lambda *a: (17, 2, None, None, [1, 2]))
                modules = {
                    "stage2_test.iamccs_h3_stage2_guides": helpers,
                    "stage2_test.iamccs_minimax_h3_pixel_refine_variant": types.SimpleNamespace(
                        _provider=lambda name: {"common": common, "nodes_looping_sampler": loop, "nodes_windows": windows}[name]),
                    "comfy_extras.nodes_custom_sampler": types.SimpleNamespace(
                        BasicGuider=types.SimpleNamespace(execute=lambda **kw: (object(),)),
                        SamplerCustomAdvanced=types.SimpleNamespace(execute=sample)),
                }
                ns = {"__package__": "stage2_test", "torch": torch,
                      "LOG": types.SimpleNamespace(info=lambda *a: None, warning=lambda *a: None),
                      "IAMCCS_MiniMaxH3LatentUpresSamplingR38": lambda: types.SimpleNamespace(prepare=lambda *a: (None, None, None, None, "test")),
                      "_result_item": lambda x, i: x[i], "_resolve_shotplan": lambda x: {},
                      "_upres_settings": lambda p: {"fast_latent_stage2_guides": mode},
                      "_stage2_window_policy": lambda p: (None if full else 56, 5, "test")}
                exec(compile(ast.Module(body=[fn], type_ignores=[]), "dispatcher", "exec"), ns)
                latent = {"samples": NestedTensor([self.video, audio]),
                          "noise_mask": NestedTensor([torch.ones_like(self.video), torch.zeros_like(audio)])}
                with patch.dict(sys.modules, modules):
                    result, _ = ns["_sample_stage2"](None, [[None, {"minimax_keyframes": [self.guide]}]], latent, None, 0)
                self.assertTrue(torch.equal(result["samples"].unbind()[1], audio))
                self.assertTrue(torch.all(result["noise_mask"].unbind()[1] == 0))
                if not full:
                    self.assertEqual(calls[0]["cond_set"]["encoded_keyframes"], [self.guide])
                if mode == "masked":
                    self.assertTrue(torch.all(result["samples"].unbind()[0][:, :, 20] == 1))


if __name__ == "__main__":
    unittest.main()
