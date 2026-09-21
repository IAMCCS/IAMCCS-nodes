"""CPU-only contract tests; no H3 model or GPU render is required."""

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "iamccs_h3_disk_upscale.py"


class DiskUpscaleContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="iamccs-h3-disk-test-")
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        fake_paths = types.ModuleType("folder_paths")
        fake_paths.get_output_directory = lambda: str(self.output)
        fake_paths.get_full_path = lambda category, name: (
            str(self.output / name) if name == "h3.safetensors" else None
        )
        fake_paths.get_filename_list = lambda category: ["h3.safetensors"]
        original = sys.modules.get("folder_paths")
        sys.modules["folder_paths"] = fake_paths
        self.addCleanup(self._restore, "folder_paths", original)
        spec = importlib.util.spec_from_file_location("iamccs_h3_disk_upscale_contract", MODULE_PATH)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    @staticmethod
    def _restore(name, old):
        if old is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old

    @staticmethod
    def _inputs():
        video = torch.arange(24 * 7 * 4 * 4, dtype=torch.float16).reshape(1, 24, 7, 4, 4)
        audio_latent = torch.zeros(1, 32, 2, 20, dtype=torch.float16)
        waveform = torch.zeros(1, 2, 24000, dtype=torch.float32)
        return {"samples": (video, audio_latent)}, {"waveform": waveform, "sample_rate": 24000}

    def _checkpoint(self):
        latent, audio = self._inputs()
        return self.module.IAMCCS_H3DiskUpscaleCheckpoint().save(
            latent, audio, "cpu_contract", 0, 22, 24, 0, 0, "cut"
        )[0]

    def test_checkpoint_round_trip_and_no_overwrite(self):
        checkpoint = self._checkpoint()
        path, manifest, video, audio_latent, wave, rate = self.module._read_checkpoint(checkpoint)
        self.assertEqual(path, Path(checkpoint))
        self.assertEqual(manifest["source_frames"], 22)
        self.assertEqual(video.shape, (1, 24, 7, 4, 4))
        self.assertEqual(audio_latent.shape, (1, 32, 2, 20))
        self.assertEqual(wave.shape, (1, 2, 24000))
        self.assertEqual(rate, 24000)
        with self.assertRaises(FileExistsError):
            self._checkpoint()

    def test_zero_source_frames_uses_exact_h3_grid(self):
        latent, audio = self._inputs()
        checkpoint, _ = self.module.IAMCCS_H3DiskUpscaleCheckpoint().save(
            latent, audio, "auto_frames", 0, 0, 24, 0, 0, "cut"
        )
        _, manifest, *_ = self.module._read_checkpoint(checkpoint)
        self.assertEqual(manifest["source_frames"], 22)

    def test_tampered_checkpoint_is_rejected(self):
        checkpoint = self._checkpoint()
        with open(checkpoint, "ab") as stream:
            stream.write(b"damaged")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.module._read_checkpoint(checkpoint)

    def test_manifest_failure_rolls_back_orphan_checkpoint(self):
        latent, audio = self._inputs()
        with patch.object(self.module, "_atomic_new_bytes", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.module.IAMCCS_H3DiskUpscaleCheckpoint().save(
                    latent, audio, "rollback", 0, 22, 24, 0, 0, "cut"
                )
        checkpoint = self.output / self.module.ROOT_NAME / "rollback" / "checkpoints" / "segment_00000.safetensors"
        self.assertFalse(checkpoint.exists())

    def test_invalid_frame_grid_and_fps_are_rejected(self):
        latent, audio = self._inputs()
        saver = self.module.IAMCCS_H3DiskUpscaleCheckpoint()
        with self.assertRaisesRegex(ValueError, "24 fps"):
            saver.save(latent, audio, "bad_fps", 0, 22, 30, 0, 0, "cut")
        with self.assertRaisesRegex(ValueError, "cannot decode"):
            saver.save(latent, audio, "bad_frames", 0, 23, 24, 0, 0, "cut")

    def test_tiling_contract(self):
        latent, temporal, spatial = self.module._tiled_params(
            "h3.safetensors", 1920, 1088, 512, 384, 96, 68, 17, "cuda", "fp16"
        )
        self.assertEqual(latent["height"], 1088)
        self.assertEqual(temporal["chunk_length"], 68)
        self.assertEqual(spatial["tile_width"], 512)
        with self.assertRaisesRegex(ValueError, "fp32"):
            self.module._tiled_params("h3.safetensors", 1920, 1088, 512, 384, 96, 68, 17, "cpu", "fp16")

    def test_grid_free_cover_preserves_native_aspect_before_delivery_crop(self):
        self.assertEqual(
            self.module._delivery_cover_size(640, 384, 1920, 1080),
            (1920, 1152),
        )
        self.assertEqual(
            self.module._delivery_cover_size(1280, 720, 1920, 1080),
            (1920, 1088),
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            self.module._delivery_cover_size(640, 0, 1920, 1080)

    def test_delivery_preset_is_resolved_from_real_source_dimensions(self):
        self.assertEqual(
            self.module._delivery_dimensions(1280, 720, "full_hd_1920x1080", 800, 600),
            (1920, 1080),
        )
        self.assertEqual(
            self.module._delivery_dimensions(1280, 768, "source_1_5x", 800, 600),
            (1920, 1152),
        )
        self.assertEqual(
            self.module._delivery_dimensions(640, 384, "custom", 2048, 1152),
            (2048, 1152),
        )

    def test_grid_free_node_does_not_require_h3_diffusion_inputs(self):
        inputs = self.module.IAMCCS_H3DiskUpscaleLearned3D.INPUT_TYPES()
        self.assertEqual(
            set(inputs["required"]),
            {
                "checkpoint_path", "output_render_id", "video_vae", "upscaler_model",
                "target_preset", "target_width",
                "target_height", "upscaler_device", "upscaler_precision",
                "temporal_core_tokens", "temporal_halo_tokens",
                "decode_groups_per_chunk",
            },
        )
        self.assertNotIn("model", inputs["required"])
        self.assertNotIn("conditioning", inputs["required"])
        self.assertNotIn("sampler", inputs["required"])

    def test_learned_lift_microchunks_full_spatial_windows_and_stitches_time(self):
        calls = []

        class FakeUpscaler:
            @classmethod
            def execute(cls, **kwargs):
                window = kwargs["latent"]["samples"]
                calls.append(int(window.shape[2]))
                out = window[:, :, :, :1, :1].expand(-1, -1, -1, 2, 2).clone()
                return ({"samples": out},)

        video = torch.arange(1 * 24 * 7 * 4 * 4, dtype=torch.float16).reshape(1, 24, 7, 4, 4)
        result = self.module._learned_3d_temporal_lift(
            FakeUpscaler, video, "h3.safetensors", 32, 32, "cpu", "fp32", 2, 1
        )
        self.assertEqual(result.shape, (1, 24, 7, 2, 2))
        self.assertEqual(calls, [3, 4, 4, 2])
        torch.testing.assert_close(result[:, :, :, 0, 0], video[:, :, :, 0, 0])

    def test_path_outside_output_root_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "inside ComfyUI"):
            self.module._read_checkpoint(str(Path(self.temporary.name).parent / "elsewhere.safetensors"))

    def test_validated_segment_is_hashed_before_atomic_publish(self):
        segment_dir = self.output / self.module.ROOT_NAME / "publish" / "segments"
        segment_dir.mkdir(parents=True)
        temporary = segment_dir / "segment_00000.unique.tmp.mp4"
        final = segment_dir / "segment_00000.mp4"
        payload = b"validated encoded segment bytes"
        temporary.write_bytes(payload)
        manifest_path, published = self.module._publish_validated_segment(
            temporary,
            final,
            {
                "schema": self.module.SCHEMA,
                "render_id": "publish",
                "segment_index": 0,
                "frame_count": 17,
                "width": 1920,
                "height": 1080,
                "fps": 24,
                "has_audio": True,
                "join_mode": "cut",
                "join_overlap_frames": 0,
            },
        )
        self.assertFalse(temporary.exists())
        self.assertEqual(final.read_bytes(), payload)
        self.assertEqual(published["segment_path"], str(final))
        self.assertEqual(published["segment_sha256"], self.module._sha256(final))
        self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8")), published)
        self.assertTrue(final.with_suffix(".mp4.iamccs.json").is_file())

    def test_ffmpeg_trim_keeps_exact_frames_canvas_and_audio(self):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            try:
                import imageio_ffmpeg
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                local = Path("D:/ComfyUI/python_embeded/ffmpeg-bin/ffmpeg.exe")
                ffmpeg = str(local) if local.is_file() else None
        if not ffmpeg:
            self.skipTest("FFmpeg unavailable")
        source = self.output / "source.mp4"
        final = self.output / "trimmed.mp4"
        subprocess.run([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=24",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
            "-frames:v", "22", "-t", "0.916666667", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-c:a", "aac", str(source),
        ], check=True, capture_output=True)
        package = types.ModuleType("iamccs_h3_disk_tests")
        package.__path__ = []
        shotboard = types.ModuleType("iamccs_h3_disk_tests.iamccs_minimax_h3_shotboard")
        shotboard._find_ffmpeg = lambda: ffmpeg
        for name, fake in ((package.__name__, package), (shotboard.__name__, shotboard)):
            original = sys.modules.get(name)
            sys.modules[name] = fake
            self.addCleanup(self._restore, name, original)
        self.module.__package__ = package.__name__
        self.module._trim_encoded(source, final, 5, 22, 64, 64, 24)
        self.assertEqual(self.module._video_frames(final), (17, 64, 64, True))


if __name__ == "__main__":
    unittest.main()
