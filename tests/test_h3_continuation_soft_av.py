"""Real FFmpeg smoke for duration-preserving external Continuation Soft AV."""

import importlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import av
import torch


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
PACKAGE = types.ModuleType("iamccs_h3_continuation_soft_testpkg")
PACKAGE.__path__ = [str(ROOT)]
sys.modules[PACKAGE.__name__] = PACKAGE
SHOTBOARD = importlib.import_module(f"{PACKAGE.__name__}.iamccs_minimax_h3_shotboard")


def _ffmpeg():
    candidates = [
        Path("D:/ComfyUI/python_embeded/ffmpeg-bin/ffmpeg.exe"),
        Path("D:/ComfyUI/python_embeded/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe"),
    ]
    return next((str(path) for path in candidates if path.is_file()), None)


def _decoded_frames(path: Path) -> int:
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


def _decoded_audio_seconds(path: Path) -> float:
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        samples = sum(frame.samples for frame in container.decode(stream))
        return samples / float(stream.rate)


class ContinuationSoftAVTests(unittest.TestCase):
    def setUp(self):
        self.ffmpeg = _ffmpeg()
        if not self.ffmpeg:
            self.skipTest("FFmpeg unavailable")
        self.temporary = tempfile.TemporaryDirectory(prefix="iamccs-h3-soft-av-")
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)

    @staticmethod
    def _audio(frames: int):
        samples = int(round(frames / 24.0 * 48000))
        waveform = torch.linspace(-0.2, 0.2, samples).repeat(1, 2, 1)
        return {"waveform": waveform, "sample_rate": 48000}

    def _write(self, name: str, images: torch.Tensor) -> Path:
        path = self.output / name
        SHOTBOARD._encode_images(images, self._audio(int(images.shape[0])), 24, path)
        SHOTBOARD._write_segment_metadata(path, int(images.shape[0]), 24, "test")
        return path

    def test_soft_boundary_and_master_keep_all_visible_frames(self):
        previous_images = torch.zeros(12, 64, 64, 3)
        previous_images[..., 0] = 0.8
        context_images = torch.zeros(4, 64, 64, 3)
        context_images[..., 0] = 0.55
        context_images[..., 1] = 0.25
        incoming_images = torch.zeros(12, 64, 64, 3)
        incoming_images[..., 1] = 0.8

        with patch.dict(os.environ, {"VHS_FORCE_FFMPEG_PATH": self.ffmpeg}):
            previous = self._write("previous.mp4", previous_images)
            context = self._write("context.mp4", context_images)
            incoming = self._write("incoming.mp4", incoming_images)
            softened = self.output / "previous_soft.mp4"
            stats = SHOTBOARD._soften_outgoing_segment_with_context(
                previous, context, softened,
                video_frames=4, video_curve="smoothstep", audio_ms=15.0, fps=24,
            )
            master = self.output / "master.mp4"
            SHOTBOARD._concat_videos([softened, incoming], master, audio_edge_fade_ms=0.0)

        self.assertEqual(stats["video_frames"], 4)
        self.assertEqual(stats["audio_ms"], 15.0)
        self.assertEqual(_decoded_frames(softened), 12)
        self.assertEqual(_decoded_frames(master), 24)
        self.assertAlmostEqual(_decoded_audio_seconds(master), 1.0, delta=0.03)
        self.assertEqual(SHOTBOARD._read_segment_frame_count(softened), 12)


if __name__ == "__main__":
    unittest.main()
