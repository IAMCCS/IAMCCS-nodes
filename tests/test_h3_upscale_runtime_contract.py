"""Smoke the IAMCCS call contract against the current H3 3D upscaler API."""

import ast
import logging
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]


def test_fast_latent_calls_current_3d_api_and_releases_upscaler():
    source = ast.parse((ROOT / "iamccs_minimax_h3_fast_latent_2pass.py").read_text(encoding="utf-8"))
    functions = [
        node for node in source.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_adaptive_temporal_chunks", "_upscale_video_latent"}
    ]
    calls = []

    class CurrentUpscaler:
        @classmethod
        def execute(cls, *, latent, model_name, mode, align,
                    enable_temporal_chunking, force_unload, device, precision):
            calls.append((mode, align, enable_temporal_chunking, force_unload, device, precision))
            return SimpleNamespace(result=({"samples": "upscaled"},))

    @contextmanager
    def passthrough_temporal_chunk(_klass, _chunk_size):
        yield False

    namespace = {
        "_upres_settings": lambda _plan: {
            "model_name": "minimax_h3_latent_upscaler_3d_fp16.safetensors",
            "precision": "fp16", "device": "cuda",
        },
        "folder_paths": SimpleNamespace(get_full_path=lambda *_args: "installed"),
        "_load_upscaler_class": lambda: CurrentUpscaler,
        "_result_item": lambda value, index: value.result[index],
        "_temporary_upscaler_temporal_chunk": passthrough_temporal_chunk,
        "LOG": logging.getLogger(__name__),
        "torch": torch,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), "<fast-upscale>", "exec"), namespace)
    result = namespace["_upscale_video_latent"]({"samples": "native"}, {}, 1280, 768)
    assert result == {"samples": "upscaled"}
    assert calls == [(
        {"mode": "target dimensions", "width": 1280, "height": 768},
        32, True, True, "cuda", "fp16",
    )]


def test_universal_workflows_keep_fast_route_connected():
    import json

    directory = Path(r"X:\1_UNIVERSAL_42_43")
    for name in (
        "A_IAMCCS_H3_MINIMAX_R42_UNIVERSAL_180926.json",
        "B_IAMCCS_H3_R43_KEYFRAME_JOINT_LATENT_NEW_UNIVERSAL_FIXED_MOTION_STATE.json",
        "D_IAMCCS_H3_R42_UNIVERSAL_VIGGLE_VIDEO_EDITOR.json",
        "E_IAMCCS_H3_R43_UNIVERSAL_VIGGLE_VIDEO_EDITOR.json",
        "G_IAMCCS_H3_R42_UNIVERSAL_SCOUT_VIGGLE_VIDEO_EDITOR.json",
        "H_IAMCCS_H3_R43_UNIVERSAL_SCOUT_VIGGLE_VIDEO_EDITOR.json",
    ):
        workflow = json.loads((directory / name).read_text(encoding="utf-8"))
        nodes = {node["id"]: node for node in workflow["nodes"]}
        assert nodes[800]["type"] == "IAMCCS_MiniMaxH3UniversalFastR42"
        assert nodes[806]["type"] in {
            "IAMCCS_MiniMaxH3UniversalPathRouterR42",
            "IAMCCS_MiniMaxH3UniversalPathRouterEditorR42",
        }
        fast_path = next(item for item in nodes[806]["inputs"] if item["name"] == "fast_path")
        assert fast_path["link"] is not None
        link = next(item for item in workflow["links"] if item[0] == fast_path["link"])
        assert link[1] == 800
        links = {item[0]: item for item in workflow["links"]}
        sources = {
            item["name"]: nodes[links[item["link"]][1]]["type"]
            for item in nodes[800]["inputs"] if item.get("link") is not None
        }
        assert sources["cine_linx"] == "IAMCCS_MiniMaxH3AudioTimelineMixR21"
        assert sources["sampled_latent"] == sources["native_frames"] == "IAMCCS_MiniMaxH3FaceDeliveryR38B"
        for name in ("native_audio", "resolved_render_id", "native_saved_report"):
            assert sources[name] == "IAMCCS_MiniMaxH3MotionContextStateCommitR37"
        assert sources["motion_state"] == "IAMCCS_MiniMaxH3AtomicConditioningBackend"
        control_links = {item["name"]: links[item["link"]] for item in nodes[800]["inputs"] if item.get("link") is not None}
        assert control_links["current_segment"][1] == control_links["total_segments"][1]
        assert sources["current_segment"] in {
            "IAMCCS_MiniMaxH3AtomicConditioningBackend",
            "IAMCCS_MiniMaxH3ContinuousBackendLazyRouterR42",
            "IAMCCS_MiniMaxH3ScoutDeliveryLazyRouterR43",
        }


def test_fast_latent_rebuilds_r37_guides_at_stage2_grid_without_fixed_chunk_count():
    source = ast.parse((ROOT / "iamccs_minimax_h3_fast_latent_2pass.py").read_text(encoding="utf-8"))
    function = next(
        node for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == "_target_conditioning"
    )
    package_name = "iamccs_fast_latent_contract_test"
    package = types.ModuleType(package_name)
    package.__path__ = []
    variant = types.ModuleType(f"{package_name}.iamccs_minimax_h3_motion_context_variant")
    calls = []

    def apply_guides(conditioning, latent, video_vae, audio_vae, plan, chunk, offset):
        calls.append((plan["width"], plan["height"], chunk["index"], offset, latent))
        return "stage2-guided", [f"image:guide-{chunk['index']}"]

    variant._apply_positioned_guides = apply_guides
    sys.modules[package_name] = package
    sys.modules[variant.__name__] = variant

    class Atomic:
        def prepare(self, **kwargs):
            return "model", "plain-conditioning", "target-grid-latent"

    plan = {
        "task_mode": "longvid_motion_context",
        "chunks": [
            {"index": index, "motion_context_trim_frames": 0 if index == 0 else 22}
            for index in range(7)
        ],
    }
    namespace = {
        "__package__": package_name,
        "_resolve_shotplan": lambda _linx: plan,
        "_replace_plan": lambda _linx, updated: updated,
        "IAMCCS_MiniMaxH3AtomicConditioningBackend": Atomic,
    }
    try:
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<fast-target>", "exec"), namespace)
        result = namespace["_target_conditioning"](
            None, None, None, None, {}, 6, 1504, 832, "native-conditioning"
        )
        assert result[1] == "stage2-guided"
        assert calls == [(1504, 832, 6, 22, "target-grid-latent")]
        assert "R37 target guides=image:guide-6" in result[2]

        plan["task_mode"] = "i2va"
        calls.clear()
        result = namespace["_target_conditioning"](
            None, None, None, None, {}, 6, 1504, 832, "native-conditioning"
        )
        assert result[1] == "plain-conditioning"
        assert calls == []
    finally:
        sys.modules.pop(variant.__name__, None)
        sys.modules.pop(package_name, None)


def test_fast_latent_keeps_arbitrary_chunk_audio_video_parity():
    source = ast.parse((ROOT / "iamccs_minimax_h3_fast_latent_2pass.py").read_text(encoding="utf-8"))
    function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "_check_segment_parity")
    namespace = {"torch": SimpleNamespace(is_tensor=lambda value: hasattr(value, "shape"))}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<fast-parity>", "exec"), namespace)
    check = namespace["_check_segment_parity"]
    plan = {
        "task_mode": "longvid_motion_context",
        "chunks": [{"index": i, "visible_frame_count": count} for i, count in enumerate([102, 102, 36, 102, 53, 70, 19])],
    }
    for index, chunk in enumerate(plan["chunks"]):
        count = chunk["visible_frame_count"]
        frames = SimpleNamespace(shape=(count, 416, 736, 3))
        waveform = SimpleNamespace(shape=(1, 1, round(count / 24 * 32000)))
        assert check(plan, index, 7, frames, {"waveform": waveform, "sample_rate": 32000}) is chunk
    for mode in ("i2va", "fl2va", "ref2va", "longvid_guides", "longvid_guided_lipsync",
                 "ref2vid_lipsync", "keyframe_joint_native"):
        plan["task_mode"] = mode
        for index, chunk in enumerate(plan["chunks"]):
            count = chunk["visible_frame_count"]
            frames = SimpleNamespace(shape=(count, 416, 736, 3))
            waveform = SimpleNamespace(shape=(1, 1, round(count / 24 * 32000)))
            assert check(plan, index, len(plan["chunks"]), frames,
                         {"waveform": waveform, "sample_rate": 32000}) is chunk
    plan["task_mode"] = "longvid_motion_context"
    try:
        check(plan, 2, 7, SimpleNamespace(shape=(36,)), {"waveform": SimpleNamespace(shape=(1, 1, 32000)), "sample_rate": 32000})
    except ValueError as exc:
        assert "audio spans" in str(exc)
    else:
        raise AssertionError("An audio/video timing mismatch must fail before Stage 2")


def test_fast_latent_uses_each_native_segment_endpoints_for_stage2_fl2va():
    source = ast.parse((ROOT / "iamccs_minimax_h3_fast_latent_2pass.py").read_text(encoding="utf-8"))
    function = next(
        node for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == "_stage2_endpoint_images"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<fast-endpoints>", "exec"), namespace)
    endpoints = namespace["_stage2_endpoint_images"]

    segment = torch.arange(5).view(5, 1, 1, 1).float()
    first, last = endpoints(segment)
    assert first.item() == 0
    assert last.item() == 4

    override_first = torch.full((1, 1, 1, 1), 10.0)
    override_last = torch.full((1, 1, 1, 1), 20.0)
    first, last = endpoints(segment, override_first, override_last)
    assert first.item() == 10
    assert last.item() == 20


def test_true_long_stage2_keeps_technical_runway_and_native_terminal_destination():
    source = ast.parse((ROOT / "iamccs_minimax_h3_fast_latent_2pass.py").read_text(encoding="utf-8"))
    names = {
        "_conditioning_geometry",
        "_prepare_true_long_stage2_plan",
        "_remap_true_long_stage2_conditioning",
    }
    functions = [
        node for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "copy": __import__("copy"),
        "H3_CONTINUOUS_INDEX": "h3_continuous_index",
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), "<fast-true-long>", "exec"), namespace)

    stage1 = [[None, {
        "minimax_frame_count": 175,
        "minimax_keyframes": [{
            "resolved_frame_index": 174,
            "latent": "native-destination",
        }],
    }]]
    plan = {
        "task_mode": "longvid_keyframe_continuous",
        "chunks": [{"index": 0}, {"index": 1, "frame_count": 124}],
    }
    technical, source_guides = namespace["_prepare_true_long_stage2_plan"](plan, 1, stage1)
    assert technical == 175
    assert plan["chunks"][1]["frame_count"] == 175
    assert plan["chunks"][1]["requested_frame_count"] == 175

    target = [[None, {
        "minimax_frame_count": 175,
        "minimax_keyframes": [
            {"resolved_frame_index": 0, "latent": "target-opening"},
            {"resolved_frame_index": 174, "latent": "target-destination"},
        ],
    }]]
    remapped = namespace["_remap_true_long_stage2_conditioning"](
        target, technical, source_guides,
    )
    guides = remapped[0][1]["minimax_keyframes"]
    assert len(guides) == 1
    assert guides[0]["latent"] == "target-destination"
    assert guides[0]["resolved_frame_index"] == 174
    assert "h3_continuous_index" not in guides[0]
    assert remapped[0][1]["minimax_frame_count"] == 175


def test_fast_latent_runtime_is_fully_iamccs_internal():
    source = (ROOT / "iamccs_minimax_h3_fast_latent_2pass.py").read_text(encoding="utf-8")
    assert 'from .vendor.minimax_h3_latent_upscaler_3d import MinimaxH3LatentUpscaler3D' in source
    assert '_provider("nodes_windows")' in source
    assert "provider=IAMCCS_INTERNAL" in source
    assert 'NODE_CLASS_MAPPINGS.get("MinimaxH3LatentUpscaler3D")' not in source
    assert 'NODE_CLASS_MAPPINGS.get("H3ContextWindows")' not in source
    assert "provider=OBVPM" not in source
    assert "comfyui-obvpm" not in source.lower()
    assert (ROOT / "vendor" / "minimax_h3_latent_upscaler_3d.py").is_file()
    assert (ROOT / "vendor" / "MINIMAX_H3_LATENT_UPSCALER_LICENSE.txt").is_file()
