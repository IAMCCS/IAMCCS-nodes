"""Real FFmpeg smoke tests for disk-upscale segment assembly."""

import importlib
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
PACKAGE = types.ModuleType("iamccs_h3_disk_join_testpkg")
PACKAGE.__path__ = [str(ROOT)]
sys.modules[PACKAGE.__name__] = PACKAGE
SHOTBOARD = importlib.import_module(f"{PACKAGE.__name__}.iamccs_minimax_h3_shotboard")
DISK = importlib.import_module(f"{PACKAGE.__name__}.iamccs_h3_disk_upscale")


def _ffmpeg():
    candidates = [
        Path("D:/ComfyUI/python_embeded/ffmpeg-bin/ffmpeg.exe"),
        Path("D:/ComfyUI/python_embeded/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe"),
    ]
    return next((str(path) for path in candidates if path.is_file()), None)


class DiskUpscaleJoinTests(unittest.TestCase):
    def setUp(self):
        self.ffmpeg = _ffmpeg()
        if not self.ffmpeg:
            self.skipTest("FFmpeg unavailable")
        self.temporary = tempfile.TemporaryDirectory(prefix="iamccs-h3-join-test-")
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)

    def _segment(self, run, index, colour, mode, overlap):
        directory = self.output / DISK.ROOT_NAME / run / "segments"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"segment_{index:05d}.mp4"
        subprocess.run([
            self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-f", "lavfi", "-i", f"color=c={colour}:size=64x64:rate=24",
            "-f", "lavfi", "-i", f"sine=frequency={440 + index * 110}:sample_rate=48000",
            "-frames:v", "12", "-t", "0.5", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "96k", str(path),
        ], check=True, capture_output=True)
        SHOTBOARD._write_segment_metadata(path, 12, 24, "native_audio_locked")
        manifest = {
            "schema": DISK.SCHEMA, "render_id": run, "segment_index": index,
            "segment_path": str(path), "segment_sha256": DISK._sha256(path),
            "frame_count": 12, "width": 64, "height": 64, "fps": 24,
            "has_audio": True, "join_mode": mode,
            "join_overlap_frames": overlap if mode == "crossfade" else 0,
        }
        path.with_suffix(".json").write_text(json.dumps(manifest), encoding="utf-8")

    def _assemble(self, run, mode, overlap):
        self._segment(run, 0, "red", mode, 0)
        self._segment(run, 1, "blue", mode, overlap)
        with patch.object(DISK.folder_paths, "get_output_directory", return_value=str(self.output)), \
                patch.dict(os.environ, {"VHS_FORCE_FFMPEG_PATH": self.ffmpeg}):
            path, _ = DISK.IAMCCS_H3DiskUpscaleAssemble().assemble(run, 2, "master")
        return Path(path)

    def test_hard_cut_has_sum_of_segment_frames(self):
        output = self._assemble("cut_test", "cut", 0)
        self.assertEqual(DISK._video_frames(output), (24, 64, 64, True))

    def test_crossfade_subtracts_one_overlap(self):
        output = self._assemble("crossfade_test", "crossfade", 3)
        self.assertEqual(DISK._video_frames(output), (21, 64, 64, True))


if __name__ == "__main__":
    unittest.main()
