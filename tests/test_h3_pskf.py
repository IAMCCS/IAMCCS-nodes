from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest

import torch


ROOT = Path(__file__).parents[1]
PACKAGE = "iamccs_pskf_testpkg"


class FakeNestedTensor(tuple):
    def __new__(cls, values):
        return tuple.__new__(cls, values)

    def unbind(self):
        return tuple(self)


class FakeContinue:
    RETURN_TYPES = ("CONDITIONING", "LATENT", "INT", "INT", "STRING")
    RETURN_NAMES = ("positive", "latent", "head", "tail", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",), "vae": ("VAE",), "previous_latent": ("LATENT",),
                "prompt": ("STRING",), "width": ("INT",), "height": ("INT",),
                "duration": ("FLOAT",), "context_frames": (["5", "22", "39"],),
                "handover_mode": (["auto", "manual"],),
                "alignment_mode": (["phase_aligned_extended"],),
                "manual_landing_tail_frames": ("INT",), "ref_image_size": (["match"],),
            },
            "optional": {"handover": ("H3_CONTINUOUS_HANDOVER",), "last_frame": ("IMAGE",)},
        }

    def build(self, *args, **kwargs):
        return ("conditioning", {"samples": "next"}, 22, 0, "delegated")


def streams_from_latent(latent):
    samples = latent["samples"]
    video, audio = samples.unbind() if hasattr(samples, "unbind") else samples
    return video, audio


def load_module(output_dir):
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(ROOT)]
    sys.modules[PACKAGE] = package

    comfy = types.ModuleType("comfy")
    nested = types.ModuleType("comfy.nested_tensor")
    nested.NestedTensor = FakeNestedTensor
    comfy.nested_tensor = nested
    sys.modules["comfy"] = comfy
    sys.modules["comfy.nested_tensor"] = nested

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_output_directory = lambda: str(output_dir)

    def get_save_image_path(prefix, root):
        raw = Path(prefix)
        folder = Path(root) / raw.parent
        return str(folder), raw.name, 1, "", prefix

    folder_paths.get_save_image_path = get_save_image_path
    sys.modules["folder_paths"] = folder_paths

    av_package_name = f"{PACKAGE}.iamccs_h3_continuous_av"
    av_package = types.ModuleType(av_package_name)
    av_package.__path__ = []
    sys.modules[av_package_name] = av_package

    latent_math = types.ModuleType(f"{av_package_name}.latent_math")
    latent_math.FPS = 24
    latent_math.pixel_frames = lambda temporal_steps: int(temporal_steps)
    sys.modules[latent_math.__name__] = latent_math

    nodes = types.ModuleType(f"{av_package_name}.nodes")
    nodes.H3ContinuousContinueV11 = FakeContinue
    nodes._streams_from_latent = streams_from_latent
    sys.modules[nodes.__name__] = nodes

    name = f"{PACKAGE}.iamccs_h3_pskf"
    spec = importlib.util.spec_from_file_location(name, ROOT / "iamccs_h3_pskf.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class PSkfTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name)
        self.module = load_module(self.output)
        video = torch.zeros((1, 24, 5, 2, 2))
        audio = torch.zeros((1, 32, 2, 10))
        self.latent = {"samples": FakeNestedTensor((video, audio))}

    def tearDown(self):
        self.temp.cleanup()

    def test_pack_owns_native_video_and_audio_state(self):
        state, info = self.module.IAMCCS_H3PSkfPack().pack(
            self.latent, clip_index=3, head_context_frames=1, visible_frame_count=4,
            source_id="shot-03", handover={
                "available": True, "frame_count": 4, "landing_tail_frames": 2,
                "handover_end_frame": 3,
            },
        )
        self.assertEqual(state["format"], "iamccs_pskf_v1")
        self.assertEqual(state["engine"], "Pianosequenza-True_long")
        self.assertEqual(state["source_id"], "shot-03")
        self.assertEqual(state["handover"]["frame_count"], 5)
        self.assertEqual(state["handover"]["handover_end_frame"], 4)
        self.assertEqual(state["handover"]["terminal_target_end_frame"], 4)
        self.assertEqual(state["handover"]["visible_end_frame"], 4)
        self.assertIn("clip 3", info)

    def test_save_and_load_round_trip(self):
        state, _ = self.module.IAMCCS_H3PSkfPack().pack(self.latent, clip_index=1)
        path, _ = self.module.IAMCCS_H3PSkfSave().save(state, "true_long/clip")
        loaded, latent, handover, resolved, info = self.module.IAMCCS_H3PSkfLoad().load(path, 1)
        self.assertEqual(Path(resolved), Path(path))
        self.assertEqual(loaded["format"], "iamccs_pskf_v1")
        video, audio = streams_from_latent(latent)
        self.assertEqual(tuple(video.shape), (1, 24, 5, 2, 2))
        self.assertEqual(tuple(audio.shape), (1, 32, 2, 10))
        self.assertFalse(handover["available"])
        self.assertIn("Pianosequenza-True_long", info)

    def test_loader_rejects_paths_outside_comfy_output(self):
        outside = self.output.parent / "outside.pskf.safetensors"
        with self.assertRaisesRegex(ValueError, "inside the ComfyUI output"):
            self.module._contained_output_path(outside)

    def test_extend_delegates_to_internal_iamccs_engine(self):
        state, _ = self.module.IAMCCS_H3PSkfPack().pack(self.latent)
        result = self.module.IAMCCS_H3TrueLongExtend().build(
            "clip", "vae", state, "prompt", 1344, 768, 10.0,
        )
        self.assertEqual(result[:4], ("conditioning", {"samples": "next"}, 22, 0))
        self.assertIn("Pianosequenza-True_long", result[-1])
        self.assertIn("IAMCCS_H3TrueLongExtend", self.module.NODE_CLASS_MAPPINGS)


if __name__ == "__main__":
    unittest.main()
