"""RefMod may affect a universal render only when Settings PRO enables it."""

import ast
import logging
from pathlib import Path
import sys
import types
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def test_refmod_is_strictly_opt_in():
    source = ast.parse((ROOT / "iamccs_minimax_h3_atomic_backend.py").read_text(encoding="utf-8"))
    function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "_apply_h3_refmod")
    namespace = {"LOG": logging.getLogger(__name__), "Any": object}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<refmod>", "exec"), namespace)
    apply = namespace["_apply_h3_refmod"]
    original = object()
    assert apply(original, {"refmod_settings": {"enabled": False, "name": "some-mod"}}) == (original, "off")

    class Loader:
        def load(self, **kwargs):
            assert kwargs["mod_1"] == "identity"
            return ["mod"], "loaded"

    class Apply:
        @staticmethod
        def execute(**kwargs):
            assert kwargs["conditioning"] is original
            return types.SimpleNamespace(result=("conditioned",))

    nodes = types.ModuleType("nodes")
    nodes.NODE_CLASS_MAPPINGS = {"MiniMaxH3RefModsLoader": Loader, "MiniMaxH3RefModApply": Apply}
    with patch.dict(sys.modules, {"nodes": nodes}):
        result, report = apply(original, {"refmod_settings": {"enabled": True, "name": "identity.safetensors"}})
    assert result == "conditioned"
    assert "identity" in report
