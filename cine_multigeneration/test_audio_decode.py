"""Standalone decoder regression: python -m unittest discover -s <this directory> -p test_audio_decode.py."""
import importlib.util
import pathlib
import subprocess
import ast
import os
import sys
import typing
import unittest
from unittest.mock import patch
import torch  # Import before patching subprocess.run, also used by Windows DLL discovery.

spec = importlib.util.spec_from_file_location("manual_audio_decode", pathlib.Path(__file__).with_name("audio_decode.py"))
decoder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decoder)


class AudioDecodeTests(unittest.TestCase):
    def test_editor_loader_falls_back_for_reported_mp4(self):
        fixture = os.environ.get("IAMCCS_TEST_AUDIO_VIDEO", "")
        path = pathlib.Path(fixture)
        if not fixture or not path.is_file():
            self.skipTest("Set IAMCCS_TEST_AUDIO_VIDEO to a stereo MP4 regression fixture")
        import torchaudio
        tree = ast.parse(pathlib.Path(__file__).with_name("__init__.py").read_text(encoding="utf-8-sig"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_load_audio_waveform_file")
        namespace = {"__package__": "_pan_decoder_regression", "os": os, "torch": torch, "torchaudio": torchaudio, "Tuple": typing.Tuple}
        with patch.dict(sys.modules, {"_pan_decoder_regression.audio_decode": decoder}):
            exec(compile(ast.Module(body=[function], type_ignores=[]), "editor_loader_regression", "exec"), namespace)
            waveform, sample_rate = namespace["_load_audio_waveform_file"](str(path))
        self.assertEqual(sample_rate, 48000)
        self.assertEqual(tuple(waveform.shape[:2]), (1, 2))
        self.assertGreater(float(waveform.abs().max()), 0)

    def test_stereo_shape_sample_rate_and_ffmpeg_audio_mapping(self):
        import numpy as np
        samples = np.array([[.1, -.2], [.3, -.4]], dtype="<f4")
        with patch.object(decoder.shutil, "which", return_value="ffmpeg"), patch.object(
            decoder.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, samples.tobytes(), b"")
        ) as run:
            waveform, sample_rate = decoder.decode_audio_with_ffmpeg("source.mp4")
        self.assertEqual(tuple(waveform.shape), (1, 2, 2))
        self.assertEqual(sample_rate, 48000)
        np.testing.assert_allclose(waveform.numpy()[0], samples.T)
        self.assertIn("0:a:0", run.call_args.args[0])
        self.assertIn("-vn", run.call_args.args[0])

    def test_missing_audio_is_reported_not_silently_discarded(self):
        with patch.object(decoder.shutil, "which", return_value="ffmpeg"), patch.object(
            decoder.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, b"", b"Stream map matches no streams")
        ):
            with self.assertRaisesRegex(ValueError, "matches no streams"):
                decoder.decode_audio_with_ffmpeg("silent.mp4")

    def test_invalid_pcm_output_is_rejected(self):
        with patch.object(decoder.shutil, "which", return_value="ffmpeg"), patch.object(
            decoder.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, b"abc", b"")
        ):
            with self.assertRaisesRegex(ValueError, "could not decode"):
                decoder.decode_audio_with_ffmpeg("source.mp4")


if __name__ == "__main__":
    unittest.main()
