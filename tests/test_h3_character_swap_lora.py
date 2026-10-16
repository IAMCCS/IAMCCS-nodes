import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_file(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_character_swap_uses_ref2va_and_preserves_editorial_duration():
    core = load_file("iamccs_minimax_h3_shotboard_core")
    plan = core.build_shotplan(timeline_data={}, global_prompt="swap", duration_seconds=200 / 24,
        task_mode="v2va_face_swap_lora", audio_mode="h3_custom_audio_drive", width=384, height=672)
    assert plan["task_mode"] == "v2va_face_swap_lora"
    assert plan["total_unique_frames"] == 200
    assert [(c["task_mode"], c["requested_frame_count"], c["frame_count"], c["unique_frames"])
            for c in plan["chunks"]] == [("ref2va", 200, 209, 200)]


def test_swap_adapters_only_load_in_selected_mode():
    swap = load_file("iamccs_h3_character_swap_lora")
    calls = []
    modules = {"folder_paths": SimpleNamespace(get_full_path=lambda *_: "installed"),
        "nodes": SimpleNamespace(LoraLoaderModelOnly=lambda: SimpleNamespace(
            load_lora_model_only=lambda model, name, strength: (calls.append((name, strength)) or model,)))}
    with patch.dict(sys.modules, modules):
        model = object()
        assert swap.apply_loras(model, {"character_swap_lora": swap.swap_plan({}, "t2va")}) is model
        assert calls == []
        assert swap.apply_loras(model, {"character_swap_lora": swap.swap_plan({}, swap.MODE)}) is model
        assert calls == [(swap.SWAP_LORA, 1.0), (swap.REALISM_LORA, 1.0)]


def test_delivery_router_never_requests_inactive_generation_branch():
    source = ast.parse((ROOT / "iamccs_minimax_h3_universal_delivery.py").read_text(encoding="utf-8"))
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef)
               and n.name == "IAMCCS_MiniMaxH3UniversalDeliverySourceR49")
    namespace = {"SUPERNODE_LINX_TYPE": "LINX", "CATEGORY": "test", "_resolve_shotplan": lambda x: x}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "<delivery-source>", "exec"), namespace)
    router = namespace[cls.name]()
    selected = {f"selected_{name}": object() for name in router.FIELDS}
    for mode in ("t2va", "ref2va", "v2va_face_swap_lora", "v2va_face_swap", "fl2va_extended_av"):
        assert router.check_lazy_status({"task_mode": mode}, **selected) == []
        assert router.select({"task_mode": mode}, **selected) == tuple(selected.values())
    requested = router.check_lazy_status({"task_mode": "longvid_keyframe_continuous"}, **selected)
    assert requested and all(name.startswith("continuous_") for name in requested)


def test_singularity_model_is_only_requested_by_lora_swap():
    source = ast.parse((ROOT / "iamccs_h3_face_swap.py").read_text(encoding="utf-8"))
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef)
               and n.name == "IAMCCS_H3CharacterSwapModelR49")
    namespace = {"__package__": "r49_test", "LORA_SWAP_MODE": "v2va_face_swap_lora"}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "<swap-model>", "exec"), namespace)
    backend = SimpleNamespace(_resolve_shotplan=lambda x: x)
    with patch.dict(sys.modules, {"r49_test.iamccs_minimax_h3_atomic_backend": backend}):
        router = namespace[cls.name]()
        base, swap = object(), object()
        for mode in ("ref2va", "v2va_face_swap", "reference_image", "t2va"):
            assert router.check_lazy_status({"task_mode": mode}, base_model=base) == []
            assert router.select({"task_mode": mode}, base_model=base) == (base,)
        assert router.check_lazy_status({"task_mode": "v2va_face_swap_lora"}, base_model=base) == ["swap_model"]
        assert router.select({"task_mode": "v2va_face_swap_lora"}, swap_model=swap) == (swap,)


def test_image_generator_does_not_request_video_delivery():
    source = ast.parse((ROOT / "iamccs_minimax_h3_editor_delivery.py").read_text(encoding="utf-8"))
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef)
               and n.name == "IAMCCS_MiniMaxH3EditorOptionalFinalR45")
    namespace = {"Any": object, "CATEGORY": "test", "SUPERNODE_LINX_TYPE": "LINX", "_shotplan": lambda x: x,
                 "_find_named_setting": lambda x, key: x.get(key)}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "<final-gate>", "exec"), namespace)
    gate = namespace[cls.name]()
    plan = {"task_mode": "reference_image"}
    assert gate.check_lazy_status(plan) == []
    assert gate.select(plan)["result"][:2] == ("", False)
    assert gate.check_lazy_status({"task_mode": "ref2va"}) == ["direct_video_path"]


def test_official_ref2v_turbo4_stale_metadata_does_not_replace_selected_adapter():
    advisor = load_file("iamccs_h3_advisor")
    name = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
    header = dict(metadata={"base_model": "Comfy-Org/MiniMax-H3 minimax_h3_fl2va_bf16.safetensors",
        "target_format": "ComfyUI generic LoRA"}, native=True, lora=True, pdd=False,
        header_sha256="e21c56a66485e59fa7ae57ae287a14ad0612d973f73f8a947cc741aa6d0256ba", bytes=1956193000)
    with patch.object(advisor, "inspect_header", return_value=header):
        asset = advisor.describe_asset(name, __file__)
        assert asset["family"] == "ref2va" and not asset["conflict"]
        advisor.validate_speed_asset(asset, "v2va_face_swap_lora")
        assert asset["recipe"] == "lightx_ref2_544_4"
        # The exception must not relax metadata validation for any other file.
        other = advisor.describe_asset("other_ref2v_turbo_4step.safetensors", __file__)
        assert other["conflict"] and other["error"]


def test_gpu_auto_keeps_dynamic_clip_and_legacy_cpu_mode_stays_supported():
    tree = ast.parse((ROOT / "iamccs_minimax_h3_atomic_backend.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_place_text_encoder")
    calls = []
    cpu_clip = object()
    namespace = {"Any": object, "LOG": SimpleNamespace(info=lambda *_: None),
        "_select_cpu_text_encoder": lambda clip: calls.append(clip) or cpu_clip}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<text-placement>", "exec"), namespace)
    clip = object()
    assert namespace[fn.name](clip, {"text_encoder_device":"gpu_auto"})[0] is clip
    assert calls == []
    assert namespace[fn.name](clip, {"text_encoder_device":"cpu_direct"})[0] is cpu_clip
    assert calls == [clip]


def test_post_decode_swap_uses_generation_chunk_index_and_restores_exact_source_audio():
    import math
    from collections.abc import Mapping
    import torch
    import torch.nn.functional as F

    tree = ast.parse((ROOT / "iamccs_minimax_h3_v2v_backend.py").read_text(encoding="utf-8"))
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
        {"_finite_float", "_requested_frames", "_segment_source_start", "_slice_audio"}]
    scope = {"Any": object, "Mapping": Mapping, "torch": torch, "F": F, "math": math, "H3_FPS": 24}
    exec(compile(ast.Module(body=helpers, type_ignores=[]), "<source-audio>", "exec"), scope)
    swap = load_file("iamccs_h3_character_swap_lora")
    swap.__package__ = "r49_decode_test"
    backend = ast.parse((ROOT / "iamccs_minimax_h3_atomic_backend.py").read_text(encoding="utf-8"))
    cls = next(n for n in backend.body if isinstance(n, ast.ClassDef)
        and n.name == "IAMCCS_MiniMaxH3GenerationBackendV2")
    render = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "render")
    branch = next(n for n in render.body if isinstance(n, ast.If)
        and ast.unparse(n.test) == "shotplan.get('task_mode') == 'v2va_face_swap_lora'")
    code = compile(ast.Module(body=[branch], type_ignores=[]), "<post-decode-swap>", "exec")
    waveform = torch.arange(400, dtype=torch.float32).reshape(1, 1, -1)
    chunks = [{"requested_frame_count":200}, {"requested_frame_count":50}]
    modules = {
        "r49_decode_test.iamccs_h3_character_swap_lora": swap,
        "r49_decode_test.iamccs_h3_face_swap": SimpleNamespace(FACE_SWAP_RESOURCE="swap_source"),
        "r49_decode_test.iamccs_minimax_h3_v2v_backend": SimpleNamespace(
            _slice_audio=scope["_slice_audio"], _segment_source_start=scope["_segment_source_start"]),
    }
    with patch.dict(sys.modules, modules):
        for index, aligned, requested, start in [(0,209,200,0), (1,56,50,200)]:
            frames = torch.zeros(aligned,2,2,3)
            generated_audio = {"waveform":torch.ones(1,1,aligned), "sample_rate":24}
            values = {"__package__":"r49_decode_test", "native_frames":frames,
                "native_audio":generated_audio, "chunk_index":index, "chunk":chunks[index],
                "shotplan":{"task_mode":swap.MODE,"chunks":chunks,
                    "v2v_settings":{"source_range_policy":"sequential_requested"}},
                "cine_linx":{"resources":{"swap_source":{"audio":{"waveform":waveform,"sample_rate":24}}}}}
            exec(code, values)
            assert values["native_frames"].shape[0] == requested
            assert torch.equal(values["native_audio"]["waveform"], waveform[...,start:start+requested])
            values["shotplan"]["task_mode"] = "t2va"
            values["native_frames"], values["native_audio"] = frames, generated_audio
            exec(code, values)
            assert values["native_frames"] is frames and values["native_audio"] is generated_audio
