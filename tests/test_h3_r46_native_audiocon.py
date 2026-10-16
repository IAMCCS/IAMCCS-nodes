"""CPU verification for historical VIDEO+AUDIO latent tail handoff and masks."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
ROOT = Path(__file__).parents[1]

class NativeAVAudioConR46Tests(unittest.TestCase):
    def test_tail_endpoint_and_protected_audio_release(self):
        spec = importlib.util.spec_from_file_location("iamccs_r46_audiocon", ROOT / "iamccs_minimax_h3_continuity.py")
        obj = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        with tempfile.TemporaryDirectory() as folder:
            fake_paths = types.ModuleType("folder_paths")
            fake_paths.get_output_directory = lambda: folder
            with patch.dict(sys.modules, {"folder_paths": fake_paths}):
                spec.loader.exec_module(obj)
                class NestedTensor:
                    def __init__(self, tensors): self.tensors = tensors
                    def unbind(self): return self.tensors
                package = types.ModuleType("comfy")
                package.__path__ = []
                nested = types.ModuleType("comfy.nested_tensor")
                nested.NestedTensor = NestedTensor
                with patch.dict(sys.modules, {"comfy": package, "comfy.nested_tensor": nested}):
                    # H3 grid: 226 technical frames, 67 native video tokens, 377 audio ticks.
                    video = torch.arange(67, dtype=torch.float32).reshape(1,1,67,1,1)
                    audio = torch.arange(377, dtype=torch.float32).reshape(1,1,1,377)
                    sample = {"samples": NestedTensor((video, audio))}
                    self.assertTrue(obj.save_longvid_sampled_latent_tail("r46test", 1, sample, tail_frames=22, endpoint_frames=209))
                    saved = obj.load_longvid_sampled_latent_tail("r46test", 1)
                    self.assertIsNotNone(saved)
                    self.assertEqual(saved["requested_endpoint_frames"], 209)
                    self.assertEqual(saved["endpoint_tokens"], 62)
                    self.assertEqual(saved["video_tokens"], 7)
                    self.assertEqual(saved["audio_ticks"], 37)
                    self.assertEqual(saved["video_tail"].flatten().tolist(), list(range(55, 62)))
                    self.assertEqual(saved["audio_tail"].flatten().tolist(), list(range(311, 348)))
                    # Target H3 child latent (209 frames = 62 video tokens).
                    next_video = torch.full((1,1,62,1,1), -1.0)
                    next_audio = torch.full((1,1,1,348), -1.0)
                    result, details = obj.apply_longvid_sampled_latent_tail(
                        {"samples": NestedTensor((next_video, next_audio))}, saved, tail_frames=22)
                    target_video, target_audio = result["samples"].unbind()
                    mv, ma = result["noise_mask"].unbind()
                    self.assertEqual(target_video.flatten()[:7].tolist(), list(range(55, 62)))
                    self.assertEqual(target_audio.flatten()[:37].tolist(), list(range(311, 348)))
                    self.assertTrue(torch.all(ma[..., :29] == 0))
                    self.assertAlmostEqual(float(ma[..., 36].item()), 1.0, places=6)
                    self.assertAlmostEqual(float(mv[..., 0, :, :].item()), 0.0, places=6)
                    self.assertAlmostEqual(float(mv[..., 6, :, :].item()), 1.0, places=6)
                    self.assertEqual(details["audio_release"], "half_cosine")
                    self.assertTrue(result["iamccs_audiocon"])

if __name__ == "__main__": unittest.main()
