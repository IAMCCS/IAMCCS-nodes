"""Real FFmpeg smoke for duration-preserving external Continuation Soft AV."""

import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np
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

    def test_locked_master_mux_receives_pristine_pcm_without_seam_processing(self):
        parts = []
        waves = []
        with patch.dict(os.environ, {"VHS_FORCE_FFMPEG_PATH": self.ffmpeg}):
            for index in range(2):
                path = self._write(f"locked_{index}.mp4", torch.zeros(12, 64, 64, 3))
                wave = np.full((2, 24000), 0.1 + index * 0.1, dtype=np.float32)
                np.savez(path.with_suffix('.mp4.locked_audio.npz'), waveform=wave, sample_rate=48000)
                parts.append(path)
                waves.append(wave)
            master = self.output / 'locked_master.mp4'
            with patch.object(SHOTBOARD, '_mux_extend_video_audio_obvpm', wraps=SHOTBOARD._mux_extend_video_audio_obvpm) as mux:
                SHOTBOARD._concat_videos(parts, master, audio_join_policy='locked_master')
            np.testing.assert_array_equal(mux.call_args.args[2], np.concatenate(waves, axis=1))
        self.assertEqual(_decoded_frames(master), 24)
        self.assertAlmostEqual(_decoded_audio_seconds(master), 1.0, delta=0.03)

    def test_public_planner_routes_text_continuous_and_joint_to_b1(self):
        for mode, rows, saved, audio_mode in (
            ('t2va_continuous', [], {}, 'h3_native_generated'),
            ('t2va_continuous', [], {}, 'h3_custom_audio_drive'),
            ('keyframe_joint_native', [
                {'type': 'image', 'imageFile': f'{i}.png', 'start': i * 240, 'length': 240}
                for i in range(3)], {'keyframe_joint_latent_new': True}, 'h3_native_generated'),
        ):
            result = SHOTBOARD.IAMCCS_MiniMaxH3ShotPlanner().plan(
                global_prompt='A continuous camera move.',
                timeline_data=json.dumps({'rows': rows, 'fps': 24, 'h3_saved_settings': saved,
                                          'audioSegments': [{'audioFile': 'master.wav', 'start': 0, 'length': 720}]}),
                duration_seconds=30, task_mode=mode, audio_mode=audio_mode,
                prompt_mapping='global_plus_local', upscale_mode='off', width=640, height=384,
                acceleration='native')
            plan = SHOTBOARD._resolve_shotplan(result[0])
            self.assertEqual(plan['task_mode'], 'fl2va_extended_av')
            self.assertEqual(plan['extended_av']['video_only'], audio_mode == 'h3_custom_audio_drive')
            self.assertFalse(plan['native_av_continuity']['enabled'])
            self.assertFalse(plan['longvid_latent_tail']['enabled'])
            self.assertEqual(sum(c['unique_frames'] for c in plan['chunks']), 720)


if __name__ == "__main__":
    unittest.main()
