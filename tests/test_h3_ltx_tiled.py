import ast
from copy import copy, deepcopy
import importlib.util
import json
import logging
import math
from pathlib import Path
import time
import types
import unittest
from unittest.mock import patch
import sys
import tempfile

import numpy as np
import torch


ROOT = Path(__file__).parents[1]
COMFY = ROOT.parents[1]
spec = importlib.util.spec_from_file_location("test_nested", COMFY / "comfy/nested_tensor.py")
nested = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nested)


class Progress:
    def __init__(self, count):
        self.count = count

    def update_absolute(self, *args):
        pass


def reshape_mask(mask, shape):
    return torch.nn.functional.interpolate(mask, size=shape[2:], mode="trilinear").expand(shape)


comfy = types.SimpleNamespace(nested_tensor=nested,
                              utils=types.SimpleNamespace(ProgressBar=Progress, reshape_mask=reshape_mask))
namespace = dict(torch=torch, comfy=comfy, copy=copy, logging=logging, math=math, time=time)
tree = ast.parse((ROOT / "iamccs_minimax_h3_ltx_tiled.py").read_text(encoding="utf-8"))
exec(compile(ast.Module(body=[n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))],
                        type_ignores=[]), str(ROOT / "iamccs_minimax_h3_ltx_tiled.py"), "exec"), namespace)


class Noise:
    seed = 123

    def generate_noise(self, latent):
        v, a = latent["samples"].unbind()
        return nested.NestedTensor((torch.arange(v.numel()).reshape(v.shape).float() / 100,
                                    torch.ones_like(a)))


class LTXTiledTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.guider = types.SimpleNamespace(original_conds={"positive": [{"frame_rate": 25, "cross_attn": object()}],
                                                            "negative": [{"frame_rate": 25}]})

        def sample(**kwargs):
            latent = kwargs["latent_image"]
            v, a = latent["samples"].unbind()
            vm, am = latent["noise_mask"].unbind()
            vn, an = kwargs["noise"].generate_noise(latent).unbind()
            self.assertTrue(torch.all(am == 0))
            self.assertTrue(torch.all(an == 0))
            self.assertEqual(kwargs["noise"].seed, 123)
            self.assertLessEqual(math.prod(v.shape[2:]), namespace["MAX_TILE_TOKENS"])
            self.assertEqual(kwargs["guider"].original_conds["positive"][0]["frame_rate"], 24)
            self.calls.append(tuple(v.shape))
            return ({"samples": nested.NestedTensor((v + vn * vm, a + 99))},
                    {"samples": nested.NestedTensor((v + vn * vm + vm, a + 100))})

        namespace["SamplerCustomAdvanced"] = types.SimpleNamespace(execute=sample)

    def run_sample(self, shape, settings):
        video = torch.arange(math.prod(shape)).reshape(shape).float() / 100
        audio = torch.randn(1, 2, 67, 4)
        mask = torch.ones(1, 1, shape[2], 1, 1)
        mask[:, :, 0] = 0
        latent = {"samples": nested.NestedTensor((video, audio)),
                  "noise_mask": nested.NestedTensor((mask, torch.ones_like(audio))), "tag": "keep"}
        original = video.clone()
        result = namespace["_sample_tiled"](Noise(), self.guider, object(), torch.tensor([.5, .1, 0]),
                                              latent, settings, 24)
        vn = Noise().generate_noise(latent).unbind()[0]
        full_mask = mask.expand_as(video)
        torch.testing.assert_close(result[0]["samples"].unbind()[0], video + vn * full_mask)
        torch.testing.assert_close(result[1]["samples"].unbind()[0], video + (vn + 1) * full_mask)
        torch.testing.assert_close(video, original)
        for item in result:
            self.assertEqual(item["tag"], "keep")
            self.assertIs(item["samples"].unbind()[1], audio)
            self.assertTrue(torch.isfinite(item["samples"].unbind()[0]).all())
        self.assertEqual(self.guider.original_conds["positive"][0]["frame_rate"], 25)

    def test_actual_failed_geometry_is_bounded_and_reassembled(self):
        self.run_sample((1, 2, 45, 36, 60), {"temporal_tile_size": 24, "temporal_overlap": 24})
        self.assertGreater(len(self.calls), 1)
        self.assertLess(max(t * h * w for _, _, t, h, w in self.calls), 45 * 36 * 60)

    def test_odd_grid_user_tiles_masks_and_denoised_output(self):
        self.run_sample((1, 3, 13, 7, 11), {"temporal_tile_size": 80, "temporal_overlap": 80,
                                                "horizontal_tiles": 3, "vertical_tiles": 2,
                                                "spatial_overlap": 3})

    def test_single_frame_and_small_edges(self):
        self.run_sample((1, 2, 1, 1, 1), {"horizontal_tiles": 6, "vertical_tiles": 6, "spatial_overlap": 8})

    def test_ranges_cover_every_index_with_positive_blend_weights(self):
        for length in (1, 2, 3, 7, 45, 129):
            for size in (1, 3, 9, 80):
                ranges = namespace["_ranges"](length, size, 24)
                weights = torch.zeros(length)
                for i, (start, end) in enumerate(ranges):
                    weights[start:end] += namespace["_axis_weight"](ranges, i, torch.float32)
                self.assertTrue(torch.all(weights > 0))
                self.assertEqual(ranges[-1][1], length)


class PromptUpgradeTests(unittest.TestCase):
    def graph(self):
        return {"prompt": {
            "different:loader": {"class_type": "IAMCCS_MiniMaxH3LTXDenoiseStackLoaderV3", "inputs": {}},
            "linx": {"class_type": "Planner", "inputs": {}},
            "control": {"class_type": "IAMCCS_MiniMaxH3PostUpscaleControlV2", "inputs": {"cine_linx": ["linx", 0]}},
            "up": {"class_type": "LTXVLatentUpsampler", "inputs": {"model": ["different:loader", 3], "image": ["control", 0]}},
            "av": {"class_type": "LTXVConcatAVLatent", "inputs": {"video": ["up", 0]}},
            "render:419": {"class_type": "SamplerCustomAdvanced", "inputs": {"latent_image": ["av", 0]}},
            "h3": {"class_type": "SamplerCustomAdvanced", "inputs": {"latent_image": ["other", 0]}},
        }}

    def test_only_iamccs_ltx_sampler_upgraded_and_idempotent(self):
        graph = self.graph()
        other = deepcopy(graph["prompt"]["h3"])
        upgrade = namespace["upgrade_ltx_prompt"]
        upgrade(graph)
        self.assertEqual(graph["prompt"]["render:419"]["class_type"], namespace["SAMPLER"])
        self.assertEqual(graph["prompt"]["render:419"]["inputs"]["cine_linx"], ["linx", 0])
        self.assertEqual(graph["prompt"]["h3"], other)
        once = deepcopy(graph)
        upgrade(graph)
        self.assertEqual(graph, once)

    def test_unrelated_ltx_and_missing_control_unchanged(self):
        graph = self.graph()
        del graph["prompt"]["control"]
        before = deepcopy(graph)
        namespace["upgrade_ltx_prompt"](graph)
        self.assertEqual(graph, before)

    def test_empty_payload_unchanged(self):
        self.assertEqual(namespace["upgrade_ltx_prompt"]({}), {})


class ResumeTests(unittest.TestCase):
    def test_saved_plan_and_pristine_pcm_loaded_without_h3(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            video = root / "native.mp4"
            segment = root / "seg.mp4"
            video.write_bytes(b"video")
            segment.write_bytes(b"segment")
            segment.with_suffix(".mp4.iamccs.json").write_text(json.dumps({"frame_count": 2}))
            segment.with_suffix(".mp4.locked_audio.npz").write_bytes(b"pcm")
            metadata = {"frame_count": 2, "fps": 24, "generation": {
                "render_id": "source", "segment_files": [segment.name],
                "shotplan": {"schema": "iamccs.minimax_h3.shotplan", "upscale_settings": {"prompt": "old"}}}}
            video.with_suffix(".mp4.iamccs.json").write_text(json.dumps(metadata))
            frames = torch.zeros(2, 2, 2, 3)
            pcm = np.ones((1, 4000), dtype=np.float32)
            namespace.update(Path=Path, json=json, __package__="ltx_resume_tests",
                             folder_paths=types.SimpleNamespace(get_output_directory=lambda: temp))
            sibling = types.ModuleType("ltx_resume_tests.iamccs_minimax_h3_shotboard")
            sibling._decode_video_master = lambda path, fps: (frames, {"waveform": torch.zeros(1, 1, 4000)})
            sibling._build_locked_audio_wave_master = lambda paths, counts: (pcm, 48000, "mono")
            with patch.dict(sys.modules, {sibling.__name__: sibling}):
                loaded = namespace["IAMCCS_MiniMaxH3LTXResumeSource"]().load(str(video), "current dialogue")
            self.assertIs(loaded[0], frames)
            self.assertTrue(torch.all(loaded[1]["waveform"] == 1))
            self.assertEqual(loaded[1]["sample_rate"], 48000)
            self.assertEqual(loaded[2]["upscale_mode"], "ltx23")
            self.assertEqual(loaded[2]["upscale_settings"]["prompt"], "current dialogue")
            self.assertEqual(metadata["generation"]["shotplan"]["upscale_settings"]["prompt"], "old")
            self.assertEqual(loaded[3], True)
            self.assertEqual(loaded[5], 24)

    def test_resume_cannot_read_outside_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            (root / "output").mkdir()
            outside = root / "outside.mp4"
            outside.write_bytes(b"video")
            namespace.update(Path=Path, folder_paths=types.SimpleNamespace(get_output_directory=lambda: str(root / "output")))
            with self.assertRaisesRegex(ValueError, "under ComfyUI/output"):
                namespace["IAMCCS_MiniMaxH3LTXResumeSource"]._path(str(outside))


if __name__ == "__main__":
    unittest.main()
