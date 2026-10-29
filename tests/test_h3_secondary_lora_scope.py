"""Exercise the real adapter loader's segment gate without GPU inference."""
import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]


def backend(monkeypatch):
    package = ModuleType("secondary_scope_test")
    package.__path__ = []
    swap = ModuleType(package.__name__ + ".iamccs_h3_character_swap_lora")
    swap.apply_loras = lambda model, plan: model
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, swap.__name__, swap)
    loader = Mock(side_effect=lambda **kw: (object(),))
    monkeypatch.setitem(sys.modules, "nodes", SimpleNamespace(
        LoraLoaderModelOnly=lambda: SimpleNamespace(load_lora_model_only=loader)))
    names = {"_external_continuation_active", "_secondary_lora_settings",
             "_secondary_lora_applies_to_segment", "_apply_secondary_lora"}
    tree = ast.parse((ROOT / "iamccs_minimax_h3_atomic_backend.py").read_text(encoding="utf-8"))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {"__package__": package.__name__, "Any": Any,
          "folder_paths": SimpleNamespace(get_full_path=lambda *args: "registered-adapter"),
          "_h3_lora_compatibility": lambda name: (True, "compatible")}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "<secondary-scope>", "exec"), ns)
    return ns["_apply_secondary_lora"], loader


@pytest.mark.parametrize("segment", [0, 1, 3])
def test_legacy_all_segments_stays_enabled(monkeypatch, segment):
    apply, loader = backend(monkeypatch)
    original = object()
    plan = {"secondary_lora": {"enabled": True, "lora_name": "test.safetensors", "strength": .5}}
    result, report = apply(original, plan, chunk_index=segment)
    assert result is not original
    assert loader.call_count == 1
    assert "all-segments" in report


def test_first_segment_skips_then_continuations_apply_without_mutating_plan(monkeypatch):
    apply, loader = backend(monkeypatch)
    original = object()
    settings = {"enabled": True, "continuation_only": True, "lora_name": "test.safetensors", "strength": .5}
    plan = {"secondary_lora": settings.copy()}
    result, report = apply(original, plan, chunk_index=0)
    assert result is original and loader.call_count == 0
    assert "starts at segment 2" in report
    for segment in (1, 2):
        result, report = apply(original, plan, chunk_index=segment)
        assert result is not original
        assert "continuation-only" in report
    assert loader.call_count == 2
    assert plan["secondary_lora"] == settings
    # Returning to the first segment must not inherit the patched model.
    assert apply(original, plan, chunk_index=0)[0] is original
    assert loader.call_count == 2


@pytest.mark.parametrize("checkpoint,enabled,expected", [("native.pt", True, 1), ("", True, 0), ("native.pt", False, 0)])
def test_checkpoint_resume_counts_as_continuation(monkeypatch, checkpoint, enabled, expected):
    apply, loader = backend(monkeypatch)
    plan = {"secondary_lora": {"enabled": True, "continuation_only": True,
                               "lora_name": "test.safetensors", "strength": .5},
            "continuation_settings": {"enabled": enabled, "checkpoint": checkpoint}}
    apply(object(), plan, chunk_index=0)
    assert loader.call_count == expected


def test_disabled_slot_never_loads_on_continuation(monkeypatch):
    apply, loader = backend(monkeypatch)
    apply(object(), {"secondary_lora": {"enabled": False, "continuation_only": True}}, chunk_index=2)
    assert loader.call_count == 0


def test_settings_schema_is_append_only():
    tree = ast.parse((ROOT / "iamccs_minimax_h3_shotboard.py").read_text(encoding="utf-8"))
    settings = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "IAMCCS_ShotboardH3Settings")
    method = next(n for n in settings.body if isinstance(n, ast.FunctionDef) and n.name == "INPUT_TYPES")
    keys = [n.value for n in ast.walk(method) if isinstance(n, ast.Constant) and n.value == "secondary_lora_continuation_only"]
    assert len(keys) == 1
    # New widget remains after the final historical extension.
    source = ast.get_source_segment((ROOT / "iamccs_minimax_h3_shotboard.py").read_text(encoding="utf-8"), method)
    assert source.index('"secondary_lora_continuation_only"') > source.index("**continuity_guard_schema()")
