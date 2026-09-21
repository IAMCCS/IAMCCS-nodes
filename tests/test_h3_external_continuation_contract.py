"""Static/small-unit contracts for opt-in external H3 AV continuation."""

import ast
import json
from pathlib import Path
import sys
import types

import torch


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = Path(r"X:\1_UNIVERSAL_42_43")


def test_external_checkpoint_is_first_chunk_only_and_explicit():
    tree = ast.parse((ROOT / "iamccs_minimax_h3_atomic_backend.py").read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_external_continuation_active"
    )
    namespace = {"Any": object}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<continuation>", "exec"), namespace)
    active = namespace["_external_continuation_active"]
    plan = {"continuation_settings": {"enabled": True, "checkpoint": "IAMCCS/MiniMaxH3/CONTINUATION/a.safetensors"}}
    assert active(plan, 0)
    assert not active(plan, 1)
    assert not active({"continuation_settings": {"enabled": False, "checkpoint": "a"}}, 0)
    assert not active({"continuation_settings": {"enabled": True, "checkpoint": ""}}, 0)


def test_settings_toggles_are_appended_after_historical_asset_fields():
    source = (ROOT / "iamccs_minimax_h3_shotboard.py").read_text(encoding="utf-8")
    assert source.index('required["h3_refmod_max_tokens"]') < source.index('required["h3_continuation_enabled"]')
    assert source.index('required["h3_continuation_enabled"]') < source.index('required["h3_continuation_save_enabled"]')
    assert source.index('"h3_continuation_run_and_gun_enabled"') < source.index('"h3_continuation_run_and_gun_join"')
    assert '"default": "soft_av"' in source
    assert '"default": 4, "min": 0, "max": 16' in source
    assert '"default": 15.0, "min": 0.0, "max": 100.0' in source


def test_soft_av_is_delivery_only_and_captures_hidden_context_before_trim():
    atomic = (ROOT / "iamccs_minimax_h3_atomic_backend.py").read_text(encoding="utf-8")
    capture = atomic.index('sampled["_iamccs_continuation_soft_av"]')
    delivery_trim = atomic.index('native_frames = native_frames[head:stop, ...]', capture)
    assert capture < delivery_trim
    assert '"run_and_gun_join": str(' in atomic


def test_all_universal_fast_branches_receive_atomic_motion_state():
    names = (
        "A_IAMCCS_H3_MINIMAX_R42_UNIVERSAL_180926.json",
        "B_IAMCCS_H3_R43_KEYFRAME_JOINT_LATENT_NEW_UNIVERSAL_FIXED_MOTION_STATE.json",
        "D_IAMCCS_H3_R42_UNIVERSAL_VIGGLE_VIDEO_EDITOR.json",
        "E_IAMCCS_H3_R43_UNIVERSAL_VIGGLE_VIDEO_EDITOR.json",
        "G_IAMCCS_H3_R42_UNIVERSAL_SCOUT_VIGGLE_VIDEO_EDITOR.json",
        "H_IAMCCS_H3_R43_UNIVERSAL_SCOUT_VIGGLE_VIDEO_EDITOR.json",
    )
    for name in names:
        workflow = json.loads((WORKFLOWS / name).read_text(encoding="utf-8"))
        nodes = {node["id"]: node for node in workflow["nodes"]}
        links = {link[0]: link for link in workflow["links"]}
        target = next(item for item in nodes[800]["inputs"] if item["name"] == "motion_state")
        assert target["link"] is not None
        link = links[target["link"]]
        assert link[1] == 9
        assert nodes[link[1]]["outputs"][link[2]]["name"] == "motion_state"
        settings = next(node for node in workflow["nodes"] if node["type"] == "IAMCCS_ShotboardH3SettingsPro")
        assert settings["widgets_values_named"]["h3_continuation_enabled"] is False
        assert settings["widgets_values_named"]["h3_continuation_save_enabled"] is False


def test_locked_audioboard_preserves_external_continuation_prefix():
    tree = ast.parse((ROOT / "iamccs_minimax_h3_audio_drive.py").read_text(encoding="utf-8"))
    wanted = {"_is_audio", "_normalize_audio_channels", "_validate_joint_av_latent", "_first", "_lock_audio_stream"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]

    class Nested:
        is_nested = True

        def __init__(self, *streams):
            self.streams = streams

        def unbind(self):
            return self.streams

    class Encode:
        @staticmethod
        def execute(vae, audio):
            return ({"samples": torch.full((1, 2, 2, 4), 2.0)},)

    class Separate:
        @staticmethod
        def execute(av_latent):
            video, audio = av_latent["samples"].unbind()
            return ({"samples": video}, {"samples": audio})

    class Concat:
        @staticmethod
        def fit_audio(target, source, mask):
            fitted = torch.zeros_like(target)
            fitted[..., :min(target.shape[-1], source.shape[-1])] = source[..., :target.shape[-1]]
            return fitted, torch.zeros_like(fitted)

        @staticmethod
        def execute(video_latent, audio_latent):
            video = video_latent["samples"]
            audio = audio_latent["samples"]
            return ({"samples": Nested(video, audio),
                     "noise_mask": Nested(torch.ones_like(video), audio_latent["noise_mask"])},)

    audio_module = types.ModuleType("comfy_extras.nodes_audio")
    audio_module.VAEEncodeAudio = Encode
    lt_module = types.ModuleType("comfy_extras.nodes_lt")
    lt_module.LTXVSeparateAVLatent = Separate
    lt_module.LTXVConcatAVLatent = Concat
    sys.modules[audio_module.__name__] = audio_module
    sys.modules[lt_module.__name__] = lt_module
    try:
        namespace = {"Any": object, "Mapping": __import__("collections.abc").abc.Mapping,
                     "torch": torch, "math": __import__("math")}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "<audio-prefix>", "exec"), namespace)
        video = torch.zeros((1, 4, 2, 3, 3))
        native_audio = torch.full((1, 2, 2, 12), 7.0)
        latent = {"samples": Nested(video, native_audio)}
        audio = {"waveform": torch.zeros((1, 2, 100)), "sample_rate": 32000}
        locked, report = namespace["_lock_audio_stream"](
            latent, audio, object(), preserve_prefix_frames=3,
        )
        result_audio = locked["samples"].unbind()[1]
        assert torch.all(result_audio[..., :5] == 7.0)
        assert torch.all(result_audio[..., 5:9] == 2.0)
        assert report["preserved_native_prefix_audio_steps"] == 5
    finally:
        sys.modules.pop(audio_module.__name__, None)
        sys.modules.pop(lt_module.__name__, None)
