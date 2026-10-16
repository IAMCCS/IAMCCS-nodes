"""CPU contract tests; run this file directly in the ComfyUI environment."""
import ast
import copy
import importlib
import importlib.util
import logging
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
original_argv = sys.argv[:]
sys.argv = [sys.argv[0], "--cpu"]
package = types.ModuleType("authored_continuous_test")
package.__path__ = [str(ROOT / "iamccs_h3_continuous_av")]
sys.modules[package.__name__] = package
N = importlib.import_module(package.__name__ + ".nodes")
R = importlib.import_module(package.__name__ + ".release_utils")
M = importlib.import_module(package.__name__ + ".latent_math")
sys.argv = original_argv


class Clip:
    def tokenize(self, prompt, **kwargs):
        self.prompt, self.images = prompt, kwargs
        return prompt

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1), {}]]


class VAE:
    def __init__(self):
        self.calls = 0

    def encode(self, image):
        self.calls += 1
        return torch.full((1, 24, 1, 2, 2), float(image.mean()))


def load_render_namespace():
    # Exercise the real orchestrator without importing unrelated UI/model loaders.
    source = ast.parse((ROOT / "iamccs_minimax_h3_longervid.py").read_text(encoding="utf-8"))
    keep = {"_final_visible_handover", "IAMCCS_MiniMaxH3FL2VAContinuousAV"}
    source.body = [node for node in source.body if getattr(node, "name", "") in keep]
    ns = {"Any": object, "copy": copy, "torch": torch, "comfy": N.comfy,
          "_streams_from_latent": N._streams_from_latent,
          "CATEGORY": "test", "H3_FPS": 24,
          "LOG": logging.getLogger("authored_continuous_test"), "ENGINE": "test"}
    exec(compile(source, "iamccs_minimax_h3_longervid.py", "exec"), ns)
    return ns


def load_reset_namespace(output):
    source = ast.parse((ROOT / "iamccs_minimax_h3_longervid.py").read_text(encoding="utf-8"))
    source.body = [
        node for node in source.body
        if getattr(node, "name", "") == "_reset_checkpoint_run"
    ]
    ns = {
        "Path": Path,
        "shutil": __import__("shutil"),
        "folder_paths": types.SimpleNamespace(get_output_directory=lambda: str(output)),
    }
    exec(compile(source, "iamccs_minimax_h3_longervid.py", "exec"), ns)
    return ns


class AuthoredContinuousTests(unittest.TestCase):
    def test_reset_segments_removes_only_selected_run(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            selected = output / "IAMCCS" / "MiniMaxH3" / "ContinuousAV" / "continuous_813_123"
            sibling = selected.parent / "continuous_813_456"
            selected.mkdir(parents=True)
            sibling.mkdir()
            (selected / "segment_0001.mp4").write_bytes(b"video")
            (selected / "state_00001.pskf.safetensors").write_bytes(b"state")
            (sibling / "segment_0001.mp4").write_bytes(b"keep")
            reset = load_reset_namespace(output)["_reset_checkpoint_run"]
            self.assertEqual(reset(selected.name), 2)
            self.assertFalse(selected.exists())
            self.assertTrue(sibling.exists())
            with self.assertRaisesRegex(ValueError, "Invalid"):
                reset("../outside")

    def test_live_core_supports_masks_and_interior_endpoint_layout(self):
        N._require_patches()
        import comfy.model_base
        import comfy.ldm.minimax.model as mm
        self.assertTrue(hasattr(comfy.model_base.MiniMaxH3, "scale_latent_inpaint"))
        kf = {"resolved_frame_index": 123, N.HC_INDEX: 119, "latent": torch.zeros(1, 24, 1, 2, 2)}
        layout = mm.PackedLayout(7, M.video_latent_t(124), 2, 2, M.temporal_shape(124)[2],
                                 keyframes=[kf], refs=[], frame_count=124)
        a, b = [(a, b) for a, b, kind in layout.segments if kind == "cond"][0]
        torch.testing.assert_close(layout.position_ids[a:b, 0], torch.full_like(layout.position_ids[a:b, 0], 7 + 119 * mm.FRAME_RESCALE))

    def test_all_contexts_and_visible_boundaries_preserve_new_frame_budget(self):
        for context in (5, 22, 39):
            for end in range(60, 124):
                sl = M.phase_aligned_extended_context_slice(M.video_latent_t(124), context, ideal_last_frame=end)
                timing = R.authored_interval_timing(120, sl, end)
                self.assertEqual(timing["endpoint_frame"] + 1 - timing["head_trim_frames"], 120)
                self.assertEqual(timing["technical_frames"] % 17, 5)
                self.assertLess(timing["grid_padding_frames"], 17)
                self.assertEqual(timing["boundary_runway_frames"], 34)
                self.assertGreaterEqual(timing["technical_padding_frames"], 34)
                self.assertEqual(sl["source_start_frame"] + timing["head_trim_frames"], end + 1)

    def test_real_provider_pins_raw_av_and_places_destination_after_hidden_runway(self):
        clip, vae = Clip(), VAE()
        first, last = torch.zeros(1, 32, 32, 3), torch.ones(1, 32, 32, 3)
        with patch.object(N, "_require_patches"):
            p, previous = N.H3ContinuousStartV11().build(clip, vae, "move", 32, 32, 5, first, last,
                                                       authored_visible_frames=120)
            start_meta = p[0][1]
            self.assertEqual(
                start_meta["minimax_keyframes"][-1]["resolved_frame_index"],
                start_meta["minimax_frame_count"] - 1,
            )
            self.assertNotIn(N.HC_INDEX, start_meta["minimax_keyframes"][-1])
            pv, pa = previous["samples"].unbind()
            pv.copy_(torch.arange(pv.shape[2]).view(1, 1, -1, 1, 1))
            pa.copy_(torch.arange(pa.shape[-1]).view(1, 1, 1, -1))
            handover = {"available": True, "frame_count": M.pixel_frames(pv.shape[2]), "visible_end_frame": 119,
                        "phase_aligned_target_end_frame": 119, "detector_mode": "iamccs_visible_terminal"}
            for context in (5, 22, 39):
                calls_before = vae.calls
                positive, latent, head, _, _ = N.H3ContinuousContinueV11().build(
                    clip, vae, previous, "continue motion", 32, 32, 5,
                    context_frames=str(context), handover=handover, last_frame=last,
                    authored_visible_frames=120, opening_image=torch.full_like(first, 0.25))
                sl = M.phase_aligned_extended_context_slice(pv.shape[2], context, ideal_last_frame=119)
                v, a = latent["samples"].unbind()
                vm, am = latent["noise_mask"].unbind()
                torch.testing.assert_close(v[:, :, :sl["context_steps"]], pv[:, :, sl["start_t"]:sl["end_t"]])
                a0, a1, _ = M.audio_slice_for_pixel_window(pa.shape[-1], sl["source_start_frame"], sl["source_end_frame"])
                torch.testing.assert_close(a[..., :a1-a0], pa[..., a0:a1])
                self.assertEqual(vm[:, :, :sl["context_steps"]].count_nonzero(), 0)
                self.assertTrue(bool(vm[:, :, sl["context_steps"]:].all()))
                self.assertEqual(am[..., :a1-a0].count_nonzero(), 0)
                keyframes = positive[0][1]["minimax_keyframes"]
                self.assertEqual(
                    keyframes[-1]["resolved_frame_index"],
                    positive[0][1]["minimax_frame_count"] - 1,
                )
                self.assertNotIn(N.HC_INDEX, keyframes[-1])
                self.assertEqual(len(keyframes), 1)
                self.assertEqual(positive[0][1]["minimax_refs"], [])
                self.assertIn(f"{(head + 119) / 24:.3f} seconds", clip.prompt)
                self.assertIn(
                    f"{(positive[0][1]['minimax_frame_count'] - 1) / 24:.3f} seconds",
                    clip.prompt,
                )
                self.assertIn("Picture 2 is the terminal destination", clip.prompt)
                self.assertEqual(len(clip.images["images"]), 2)
                self.assertAlmostEqual(float(clip.images["images"][0].mean()), 0.25, delta=1 / 255)
                self.assertEqual(vae.calls - calls_before, 1)

    def test_no_lock_keeps_landing_and_lock_keeps_first_landing_frame(self):
        f = load_render_namespace()["_final_visible_handover"]
        base = {"available": True, "frame_count": 124, "freeze_detected": False,
                "handover_end_frame": 114, "landing_tail_frames": 9}
        self.assertEqual(f(base, 120)["handover_end_frame"], 119)
        locked = dict(base, freeze_detected=True, freeze_start_frame=93)
        self.assertEqual(f(locked, 120)["handover_end_frame"], 93)
        self.assertEqual(base["landing_tail_frames"], 9)

    def test_custom_audio_lock_preserves_pinned_video_prefix_and_clock(self):
        spec = importlib.util.spec_from_file_location("authored_audio_test", ROOT / "iamccs_minimax_h3_audio_drive.py")
        audio_drive = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audio_drive)
        class AudioVAE:
            audio_sample_rate = 32000
            def encode(self, waveform):
                return torch.ones(1, 32, 2, round(waveform.shape[1] / 32000 * 40))
        target, _ = N._empty_av_latent(32, 32, 175)
        v, a = target["samples"].unbind()
        v[:, :, :12] = 7
        vm = torch.ones((1, 1) + tuple(v.shape[2:]))
        vm[:, :, :12] = 0
        am = torch.ones((1, 1) + tuple(a.shape[2:]))
        target["noise_mask"] = N.comfy.nested_tensor.NestedTensor((vm, am))
        source = {"waveform": torch.ones(1, 2, 32000 * 20), "sample_rate": 32000}
        timing = R.authored_interval_timing(120, {"actual_context_frames": 47, "source_end_frame": 115}, 119)
        target["iamccs_authored_interval"] = timing
        start = 5 - timing["head_trim_frames"] / 24
        duration = timing["technical_frames"] / 24
        sliced, _ = audio_drive._slice_audio_for_chunk(source, start, duration, pad_to_duration=True)
        self.assertEqual(sliced["waveform"].shape[-1], round(duration * 32000))
        locked, _ = audio_drive._lock_audio_stream(target, sliced, AudioVAE())
        torch.testing.assert_close(locked["samples"].unbind()[0], v)
        torch.testing.assert_close(locked["noise_mask"].unbind()[0], vm)
        self.assertEqual(locked["noise_mask"].unbind()[1].count_nonzero(), 0)
        self.assertEqual(locked["iamccs_authored_interval"], timing)

    def test_seven_interval_orchestrator_outputs_840_frames_without_replay_or_bridge(self):
        ns = load_render_namespace()
        calls = []
        freeze = False
        class Sampler:
            def render(self, **kw):
                timing = kw["latent"]["iamccs_authored_interval"]
                calls.append(timing)
                n = timing["technical_frames"]
                head = timing["head_trim_frames"]
                images = (torch.arange(n) - head + kw["chunk_index"] * 120).view(-1, 1, 1, 1).float()
                audio = {"waveform": torch.ones(1, 2, round(n * 32000 / 24)), "sample_rate": 32000}
                return images, audio, images[-1:], kw["latent"], 24, "mock sampling"
        class Analyzer:
            def analyze(self, images, **kw):
                return ({"available": True, "frame_count": len(images), "freeze_detected": freeze,
                         "freeze_start_frame": len(images) - 20 if freeze else -1,
                         "handover_end_frame": len(images)-10, "landing_tail_frames": 9}, "no lock fallback")
        plan = {"task_mode": "longvid_keyframe_continuous", "width": 32, "height": 32}
        chunks = [{"first_image": "a" if i == 0 else "", "last_image": "b", "requested_frame_count": 120,
                   "creative_prompt": "move", "timeline_start_seconds": i * 5} for i in range(7)]
        providers = {"H3ContinuousStartV11": N.H3ContinuousStartV11,
                     "H3ContinuousContinueV11": N.H3ContinuousContinueV11,
                     "H3ContinuousAnalyzeHandoverV11": Analyzer,
                     "H3ContinuousStitchOutputV11": N.H3ContinuousStitchOutputV11,
                     "H3ContinuousSeamlessJoinV11": N.H3ContinuousSeamlessJoinV11}
        ns.update(_resolve_shotplan=lambda _: plan, _contract=lambda _: {"context_frames": "39", "segment_checkpoints": False},
                  _chunks=lambda _: chunks, _provider_plan=lambda *args: None, _provider_node=providers.__getitem__,
                  _ensure_provider_runtime=lambda _: None, _load_image=lambda _: torch.zeros(1, 32, 32, 3),
                  core_masks_available=lambda: True,
                  _sampling=lambda _: dict(seed=1, seed_stride=1, steps=20, sampler_name="euler", scheduler="simple",
                                          denoise=1, shift_video=12, shift_audio=3),
                  IAMCCS_MiniMaxH3GenerationBackendV2=Sampler)
        with patch.object(N, "_require_patches"):
            result = ns["IAMCCS_MiniMaxH3FL2VAContinuousAV"]().render(None, Clip(), VAE(), None, None)
        images, audio, *_ = result
        torch.testing.assert_close(images.flatten(), torch.arange(840).float())
        self.assertEqual(audio["waveform"].shape[-1], 35 * 32000)
        self.assertTrue(all(c["technical_frames"] > 120 for c in calls[1:]))
        self.assertEqual(len(result), 12)
        self.assertEqual([int(item.shape[0]) for item in result[6]], [120] * 7)
        self.assertEqual(len(result[7]), 7)
        self.assertEqual(len(result[8]), 7)
        self.assertEqual(result[9], list(range(7)))
        self.assertEqual(result[10], [7] * 7)
        self.assertEqual(result[11], [0] + [call["head_trim_frames"] for call in calls[1:]])
        spec = importlib.util.spec_from_file_location("authored_audio_loop_test", ROOT / "iamccs_minimax_h3_audio_drive.py")
        drive = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(drive)
        master = {"waveform": torch.ones(1, 2, 35 * 32000), "sample_rate": 32000}
        class AudioVAE:
            audio_sample_rate = 32000
            def encode(self, waveform):
                return torch.ones(1, 32, 2, round(waveform.shape[1] / 32000 * 40))
        plan["audio_mode"] = "h3_custom_audio_drive"
        slices = []
        def audio_slice(source, start, duration, **kw):
            slices.append((start, duration))
            return drive._slice_audio_for_chunk(source, start, duration, **kw)
        ns.update(BEHAVIOR_LOCKED="h3_custom_audio_drive", _resource_audio=lambda *args: (master, "test"),
                  _slice_audio_for_chunk=audio_slice, _lock_audio_stream=drive._lock_audio_stream,
                  _fit_audio_to_duration=drive._fit_audio_to_duration)
        calls.clear()
        with patch.object(N, "_require_patches"):
            images, audio, *_ = ns["IAMCCS_MiniMaxH3FL2VAContinuousAV"]().render(None, Clip(), VAE(), AudioVAE(), None)
        torch.testing.assert_close(images.flatten(), torch.arange(840).float())
        torch.testing.assert_close(audio["waveform"], master["waveform"])
        for i, (start, duration) in enumerate(slices):
            self.assertAlmostEqual(start + calls[i]["head_trim_frames"] / 24, i * 5)
            self.assertAlmostEqual(duration, calls[i]["technical_frames"] / 24)
        freeze = True
        with patch.object(N, "_require_patches"), self.assertRaisesRegex(RuntimeError, "desynchronize"):
            ns["IAMCCS_MiniMaxH3FL2VAContinuousAV"]().render(None, Clip(), VAE(), AudioVAE(), None)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
