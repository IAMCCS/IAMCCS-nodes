import ast
from collections.abc import Mapping
from copy import deepcopy
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import imageio_ffmpeg
import torch

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("h3_swap_video_test", ROOT / "iamccs_h3_swap_video.py")
swap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(swap)

# Exercise the actual installed VHS lazy-audio implementation without importing
# its unrelated model and UI modules or loading video frames onto the GPU.
vhs_path = ROOT.parent / "comfyui-videohelpersuite/videohelpersuite/utils.py"
tree = ast.parse(vhs_path.read_text(encoding="utf-8"))
audio_nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in {"get_audio", "LazyAudioMap"}]
scope = dict(Mapping=Mapping, subprocess=subprocess, torch=torch, re=re,
             ffmpeg_path=imageio_ffmpeg.get_ffmpeg_exe(), ENCODE_ARGS=("utf-8", "replace"))
exec(compile(ast.Module(body=audio_nodes, type_ignores=[]), str(vhs_path), "exec"), scope)


class OptionalSourceAudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.silent = Path(cls.temp.name) / "silent.mp4"
        cls.sound = Path(cls.temp.name) / "sound.mp4"
        base = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=32x32:r=24:d=0.25"]
        subprocess.run(base + ["-c:v", "libx264", "-an", str(cls.silent)], capture_output=True, check=True)
        subprocess.run(base + ["-f", "lavfi", "-i", "sine=frequency=440:duration=0.25", "-c:v", "libx264", "-c:a", "aac", "-shortest", str(cls.sound)], capture_output=True, check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def lazy(self, path, start=0, duration=0):
        return scope["LazyAudioMap"](str(path), start, duration)

    def test_silent_clip_returns_none_without_materializing_lazy_audio(self):
        audio = self.lazy(self.silent)
        self.assertIsNone(swap.optional_video_audio(audio))
        self.assertIsNone(audio._dict)
        with self.assertRaisesRegex(Exception, "failed to extract audio"):
            dict(audio)  # Reproduces the original failure in the cache traversal.

    def test_sound_clip_preserves_real_vhs_audio_and_trim(self):
        audio = swap.optional_video_audio(self.lazy(self.sound, duration=0.125))
        expected = scope["get_audio"](str(self.sound), duration=0.125)
        self.assertIsInstance(audio, dict)
        self.assertEqual(audio["sample_rate"], expected["sample_rate"])
        self.assertTrue(torch.equal(audio["waveform"], expected["waveform"]))
        self.assertGreater(float(audio["waveform"].abs().max()), 0)

    def test_absent_and_materialized_audio_pass_through(self):
        self.assertIsNone(swap.optional_video_audio(None))
        audio = {"waveform": torch.ones(1, 1, 32), "sample_rate": 32000}
        self.assertIs(swap.optional_video_audio(audio), audio)

    def test_real_decode_errors_are_not_hidden_as_silence(self):
        with self.assertRaises(Exception):
            swap.optional_video_audio(self.lazy(Path(self.temp.name) / "missing.mp4"))

    def test_loader_normalizes_output_before_comfy_cache_walk(self):
        frames = object()
        original = SimpleNamespace(
            RETURN_TYPES=("IMAGE", "INT", "AUDIO", "VHS_VIDEOINFO"),
            INPUT_TYPES=lambda: {"required": {"video": ("STRING", {})}},
            IS_CHANGED=lambda **kwargs: kwargs["video"], VALIDATE_INPUTS=lambda video: True,
        )
        class Delegate:
            RETURN_TYPES = original.RETURN_TYPES
            INPUT_TYPES = staticmethod(original.INPUT_TYPES)
            IS_CHANGED = staticmethod(original.IS_CHANGED)
            VALIDATE_INPUTS = staticmethod(original.VALIDATE_INPUTS)
            def load_video(inner, video, **kwargs):
                return frames, 6, self.lazy(video), {"loaded_fps": 24}
        with patch.dict(sys.modules, {"nodes": SimpleNamespace(NODE_CLASS_MAPPINGS={"VHS_LoadVideo": Delegate, "VHS_LoadVideoPath": Delegate})}):
            for cls in (swap.IAMCCS_H3SwapVideoLoader, swap.IAMCCS_H3SwapVideoPathLoader):
                self.assertEqual(cls.INPUT_TYPES(), original.INPUT_TYPES())
                self.assertEqual(cls.IS_CHANGED(video="file.mp4"), "file.mp4")
                self.assertTrue(cls.VALIDATE_INPUTS("file.mp4"))
                silent = cls().load_video(video=str(self.silent))
                self.assertIs(silent[0], frames)
                self.assertIsNone(silent[2])
                sound = cls().load_video(video=str(self.sound))
                self.assertIsInstance(sound[2], dict)
                # ModelPatcher's cache walker may now safely inspect the output.
                self.assertEqual(set(sound[2]), {"waveform", "sample_rate"})


class QueueAdapterTests(unittest.TestCase):
    def prompt(self):
        return {"prompt": {
            "1": {"class_type": "IAMCCS_ShotboardH3SettingsPro", "inputs": {"task_mode": "v2va_face_swap_lora"}},
            "2": {"class_type": "VHS_LoadVideo", "inputs": {"video": "source.mp4"}},
            "3:9": {"class_type": "IAMCCS_H3FaceSwapInput", "inputs": {"source_video": ["2", 0]}},
            "4": {"class_type": "VHS_LoadVideoPath", "inputs": {"video": "unrelated.mp4"}},
        }}

    def test_only_connected_swap_loader_is_replaced_and_adapter_is_idempotent(self):
        data = self.prompt()
        swap.upgrade_swap_video_prompt(data)
        self.assertEqual(data["prompt"]["2"]["class_type"], "IAMCCS_H3SwapVideoLoader")
        self.assertEqual(data["prompt"]["4"]["class_type"], "VHS_LoadVideoPath")
        previous = deepcopy(data)
        swap.upgrade_swap_video_prompt(data)
        self.assertEqual(data, previous)

    def test_other_modes_are_unchanged(self):
        data = self.prompt()
        data["prompt"]["1"]["inputs"]["task_mode"] = "i2va"
        previous = deepcopy(data)
        swap.upgrade_swap_video_prompt(data)
        self.assertEqual(data, previous)

    def test_connected_path_loader_through_video_transform(self):
        data = self.prompt()
        data["prompt"]["3:9"]["inputs"]["source_video"] = ["5", 0]
        data["prompt"]["5"] = {"class_type": "ImageResize", "inputs": {"image": ["4", 0]}}
        swap.upgrade_swap_video_prompt(data)
        self.assertEqual(data["prompt"]["4"]["class_type"], "IAMCCS_H3SwapVideoPathLoader")
        self.assertEqual(data["prompt"]["2"]["class_type"], "VHS_LoadVideo")


if __name__ == "__main__":
    unittest.main()
