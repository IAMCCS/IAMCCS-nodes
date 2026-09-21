import importlib
import gc
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from imageio_ffmpeg import read_frames
import torch
from safetensors.torch import save_file


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
package = types.ModuleType("iamccs_pixel_safe_testpkg")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
importlib.import_module(f"{package.__name__}.iamccs_minimax_h3_shotboard")
DELIVERY = importlib.import_module(f"{package.__name__}.iamccs_minimax_h3_universal_delivery")
LIBRARY = importlib.import_module(f"{package.__name__}.iamccs_minimax_h3_asset_library")
CORE = importlib.import_module(f"{package.__name__}.iamccs_minimax_h3_shotboard_core")
ATOMIC = importlib.import_module(f"{package.__name__}.iamccs_minimax_h3_atomic_backend")


class PixelSafeRouteTests(unittest.TestCase):
    def test_planner_accepts_pixel_safe_route(self):
        plan = CORE.build_shotplan(timeline_data="", global_prompt="A moving subject.",
                                   duration_seconds=5, task_mode="t2va", upscale_enabled=True,
                                   upscale_mode="pixel_tiled_low_vram", width=960, height=544)
        self.assertEqual(plan["upscale_mode"], "pixel_tiled_low_vram")

    def test_route_uses_existing_lazy_rtx_socket(self):
        plan = {"upscale_enabled": True, "upscale_mode": "pixel_tiled_low_vram"}
        with patch.object(DELIVERY, "_resolve_shotplan", return_value=plan):
            control = DELIVERY.IAMCCS_MiniMaxH3UniversalRouteControlR42().resolve({})
            self.assertEqual(control[:4], (False, True, False, "pixel_tiled_low_vram"))
            self.assertEqual(DELIVERY.IAMCCS_MiniMaxH3UniversalPathRouterR42._selected_input({}), "rtx_final_path")

    def test_cpu_lanczos_streams_native_audio_without_upscale_model(self):
        frames = torch.zeros(2, 16, 16, 3)
        frames[1, :, :, 0] = 1
        audio = {"waveform": torch.zeros(1, 2, 8000), "sample_rate": 32000}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pixel_safe.mp4"
            DELIVERY._stream_pixel_safe_frames(frames, output, audio, 32, 32, 8, {"method": "cpu_lanczos"})
            self.assertGreater(output.stat().st_size, 100)
            reader = read_frames(str(output))
            self.assertEqual(next(reader)["size"], (32, 32))
            reader.close()


    def test_local_model_branch_tiles_one_frame_and_streams(self):
        from comfy_extras.nodes_upscale_model import UpscaleModelLoader
        import comfy.model_management as mm

        class FakeModel:
            scale = 2
            patcher = types.SimpleNamespace(load_device="cpu")

            def __call__(self, patch):
                return torch.nn.functional.interpolate(patch, scale_factor=2, mode="nearest")

        frames = torch.rand(2, 16, 16, 3)
        audio = {"waveform": torch.zeros(1, 2, 8000), "sample_rate": 32000}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model_tiled.mp4"
            with patch.object(UpscaleModelLoader, "execute", return_value=(FakeModel(),)), patch.object(mm, "load_models_gpu"):
                DELIVERY._stream_pixel_safe_frames(frames, output, audio, 32, 32, 8, {
                    "method": "model_tiled", "model_name": "fake.safetensors", "tile_size": 128, "overlap": 16,
                })
            reader = read_frames(str(output))
            self.assertEqual(next(reader)["size"], (32, 32))
            reader.close()


class RefModAutomaticBranchTests(unittest.TestCase):
    def test_off_is_conditioning_identity(self):
        conditioning = [[object(), {"minimax_refs": []}]]
        output, report = ATOMIC._apply_h3_refmod(conditioning, {})
        self.assertIs(output, conditioning)
        self.assertEqual(report, "off")

    def test_enabled_invokes_provider_loader_and_apply_once(self):
        calls = []

        class Loader:
            def load(self, **kwargs):
                calls.append(kwargs)
                return ([(object(), 0.8)], "hint")

        class Apply:
            @staticmethod
            def execute(**kwargs):
                calls.append(kwargs)
                return types.SimpleNamespace(result=("provider-conditioning", None))

        import nodes as comfy_nodes
        with patch.dict(comfy_nodes.NODE_CLASS_MAPPINGS, {
            "MiniMaxH3RefModsLoader": Loader, "MiniMaxH3RefModApply": Apply,
        }):
            output, report = ATOMIC._apply_h3_refmod([], {"refmod_settings": {
                "enabled": True, "name": "characters/hero.safetensors", "strength": 0.8,
                "retention": 0.7, "max_tokens": 4096,
            }})
        self.assertEqual(output, "provider-conditioning")
        self.assertIn("characters/hero", report)
        self.assertEqual(calls[0]["mod_1"], "characters/hero")
        self.assertEqual(calls[1]["retention"], 0.7)

    def test_enabled_without_name_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "valid saved name"):
            ATOMIC._apply_h3_refmod([], {"refmod_settings": {"enabled": True, "name": ""}})


class H3AssetLibraryTests(unittest.TestCase):
    def test_selective_asset_purge_preserves_preview_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "hero.safetensors"
            preview = path.with_suffix(".png")
            sidecar = path.with_suffix(".json")
            path.write_bytes(b"latent")
            preview.write_bytes(b"preview")
            sidecar.write_text("{}", encoding="utf-8")
            with patch.object(LIBRARY, "_roots", return_value=[root.resolve()]):
                with self.assertRaisesRegex(ValueError, "protected"):
                    LIBRARY.purge_asset("refmod", 0, path.name, "preview")
                result = LIBRARY.purge_asset("refmod", 0, path.name, "asset")
                self.assertEqual(result["removed"], ["hero.safetensors"])
                self.assertFalse(path.exists())
                self.assertTrue(preview.exists())
                self.assertTrue(sidecar.exists())

    def test_cache_purge_is_single_file_and_confined_to_allowlisted_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = (Path(directory) / "latent_tail").resolve()
            nested = root / "render_a"
            nested.mkdir(parents=True)
            latent = nested / "segment_0002.pt"
            latent.write_bytes(b"latent-cache")
            protected = nested / "final_video.mp4"
            protected.write_bytes(b"do-not-delete")
            specs = [{
                "id": "h3_latent_tail",
                "label": "LongVid latent tail",
                "root": root,
                "extensions": {".pt", ".tmp"},
            }]
            with patch.object(LIBRARY, "_cache_roots", return_value=specs):
                items = LIBRARY.list_cache_files()
                self.assertEqual([item["path"] for item in items], ["render_a/segment_0002.pt"])
                result = LIBRARY.purge_cache_file("h3_latent_tail", "render_a/segment_0002.pt")
                self.assertEqual(result["removed"], ["segment_0002.pt"])
                self.assertFalse(latent.exists())
                self.assertTrue(protected.is_file())
                with self.assertRaises(ValueError):
                    LIBRARY.purge_cache_file("h3_latent_tail", "render_a/final_video.mp4")
                with self.assertRaises(ValueError):
                    LIBRARY.purge_cache_file("h3_latent_tail", "../outside.pt")

    def test_refmod_metadata_preview_cache_and_filename_rename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "hero.safetensors"
            save_file({"latent": torch.zeros(1, 24, 1, 2, 2)}, str(path), metadata={
                "refmod_meta": json.dumps({"name": "hero", "kind": "image"}),
            })
            mod = types.SimpleNamespace(name="hero", kind="image", path=str(path.with_suffix("")),
                                        latent=torch.zeros(1, 24, 1, 2, 2))
            vae = types.SimpleNamespace(decode=lambda latent: torch.ones(1, 16, 16, 3))
            with patch.object(LIBRARY, "_roots", return_value=[root]):
                self.assertEqual(LIBRARY.list_assets("refmod")[0]["name"], "hero")
                preview = LIBRARY.IAMCCS_MiniMaxH3RefModPreviewCache().cache([(mod, 1.0)], vae)[0]
                self.assertTrue(Path(preview).is_file())
                self.assertEqual(LIBRARY.rename_asset("refmod", 0, path.name, "hero_v2"), "hero_v2.safetensors")
                self.assertTrue((root / "hero_v2.png").is_file())

    def test_list_and_rename_preserve_preview_and_reject_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "clip_00001.safetensors"
            save_file({"video": torch.zeros(1)}, str(checkpoint), metadata={
                "format": "h3_continuous_av_v8", "frame_count": "243", "head_context_frames": "22",
            })
            checkpoint.with_suffix(".png").write_bytes(b"preview")
            with patch.object(LIBRARY, "_roots", return_value=[root]):
                assets = LIBRARY.list_assets("continuation")
                self.assertEqual(len(assets), 1)
                self.assertTrue(assets[0]["preview"])
                self.assertEqual(LIBRARY.rename_asset("continuation", 0, checkpoint.name, "accepted_shot"), "accepted_shot.safetensors")
                self.assertTrue((root / "accepted_shot.png").exists())
                with self.assertRaises(ValueError):
                    LIBRARY._asset("continuation", 0, "../outside.safetensors")

    def test_iamccs_checkpoint_save_load_and_preview(self):
        latent = {"samples": (torch.zeros(1, 24, 3, 2, 2), torch.zeros(1, 32, 2, 12))}
        preview = torch.ones(1, 16, 16, 3)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(LIBRARY.folder_paths, "get_output_directory", return_value=directory):
                saver = LIBRARY.IAMCCS_MiniMaxH3ContinuationSave()
                first, _ = saver.save(latent, preview_image=preview)
                second, _ = saver.save(latent, preview_image=preview)
                self.assertNotEqual(first, second)
                self.assertTrue(Path(first).with_suffix(".png").is_file())
                loaded = LIBRARY.IAMCCS_MiniMaxH3ContinuationLoad().load(first)
                self.assertEqual(loaded[1], first)
                del loaded
                gc.collect()
                plan = {"continuation_settings": {"checkpoint": first, "context_frames": "39",
                                                  "handover_mode": "manual", "manual_tail_frames": 51}}
                with patch.object(LIBRARY, "_resolve_shotplan", return_value=plan):
                    selected = LIBRARY.IAMCCS_MiniMaxH3ContinuationLoad().load("", cine_linx={})
                    self.assertEqual(selected[1], first)
                    del selected
                    with patch.object(LIBRARY.H3ContinuousContinueV11, "build", return_value="built") as build:
                        result = LIBRARY.IAMCCS_MiniMaxH3ContinuationContinue().build(
                            None, None, None, "next shot", 960, 544, 5, cine_linx={})
                        self.assertEqual(result, "built")
                        self.assertEqual(build.call_args.kwargs["context_frames"], "39")
                        self.assertEqual(build.call_args.kwargs["manual_landing_tail_frames"], 51)
                gc.collect()

    def test_continued_checkpoint_maps_visible_handover_into_technical_timeline(self):
        latent = {"samples": (torch.zeros(1, 24, 3, 2, 2), torch.zeros(1, 32, 2, 12))}
        handover = {
            "available": True,
            "frame_count": 5,
            "detector_mode": "stable_tail_consensus",
            "phase_aligned_target_end_frame": 4,
            "phase_aware_target_end_frame": 4,
            "freeze_start_frame": 4,
            "landing_tail_frames": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(LIBRARY.folder_paths, "get_output_directory", return_value=directory):
                path, info = LIBRARY.IAMCCS_MiniMaxH3ContinuationSave().save(
                    latent, handover=handover, head_context_frames=2,
                    visible_frame_count=5,
                )
                loaded = LIBRARY.IAMCCS_MiniMaxH3ContinuationLoad().load(path)
                mapped = loaded[3]
                self.assertEqual(mapped["frame_count"], 9)
                self.assertEqual(mapped["visible_start_frame"], 2)
                self.assertEqual(mapped["visible_end_frame"], 7)
                self.assertEqual(mapped["technical_padding_frames"], 2)
                self.assertEqual(mapped["phase_aligned_target_end_frame"], 6)
                self.assertEqual(mapped["terminal_target_end_frame"], 6)
                self.assertIn("technical padding 2", info)
                with patch.object(LIBRARY.H3ContinuousContinueV11, "build", return_value="terminal") as build:
                    result = LIBRARY.IAMCCS_MiniMaxH3ContinuationContinue().build(
                        None, None, loaded[0], "next", 64, 64, 5,
                        handover=mapped, handover_mode="terminal",
                    )
                self.assertEqual(result, "terminal")
                self.assertEqual(build.call_args.kwargs["handover_mode"], "auto")
                self.assertEqual(build.call_args.kwargs["handover"]["phase_aligned_target_end_frame"], 6)
                del mapped, loaded, build
                gc.collect()

    def test_legacy_chain_inference_reads_generation_key(self):
        shotboard = sys.modules[f"{package.__name__}.iamccs_minimax_h3_shotboard"]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            checkpoint = output / "IAMCCS" / "MiniMaxH3" / "CONTINUATION" / "demo_00001.safetensors"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            media = output / "IAMCCS" / "MiniMaxH3" / "demo_seg_0001.mp4"
            media.parent.mkdir(parents=True, exist_ok=True)
            media.write_bytes(b"video")
            Path(str(media) + ".iamccs.json").write_text(json.dumps({
                "generation": {"render_id": "demo", "stage": "native", "segment_index": 0},
            }), encoding="utf-8")
            with patch.object(shotboard.folder_paths, "get_output_directory", return_value=directory):
                chain = shotboard._infer_continuation_chain_from_output(checkpoint)
            self.assertEqual(chain["resolved_segments"], [media.resolve()])
            self.assertTrue(checkpoint.with_suffix(".chain.json").is_file())


if __name__ == "__main__":
    unittest.main()
