import ast
from pathlib import Path
import re
from types import SimpleNamespace
from unittest.mock import patch
from unittest import TestCase

ROOT = Path(__file__).resolve().parents[1]


def contract_functions():
    source = ast.parse((ROOT / "iamccs_h3_edit_prompt.py").read_text(encoding="utf-8"))
    functions = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in {"character_swap_prompt", "character_swap_contract"}]
    namespace = {"re": re}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "<swap-contract>", "exec"), namespace)
    return namespace


def test_native_swap_contract_has_single_identity_sheet_and_environment_roles():
    fn = contract_functions()["character_swap_contract"]
    assert "ONE identity" in fn({})["instruction"]
    result = {"source_target": "the main person", "direction": "", "identity": "an astronaut in an ivory suit", "environment": "a volcanic valley"}
    prompt = fn({"variant": "environment", "result": result})["sections"]["swap_prompt"]
    assert prompt.startswith("r34l1sm\n")
    assert "<Video 1>" in prompt and "<Picture 1>" in prompt and "<Picture 2>" in prompt
    assert "Preserve the source environment" not in prompt
    assert "original audio" in prompt and "single replacement character" in prompt
    identity_prompt = fn({"result": result})["sections"]["swap_prompt"]
    assert "<Picture 2>" not in identity_prompt
    assert "Preserve the source environment" in identity_prompt


def test_native_swap_contract_rejects_missing_observations():
    fn = contract_functions()["character_swap_contract"]
    with TestCase().assertRaises(ValueError):
        fn({"result": {"identity": "astronaut"}})
    with TestCase().assertRaises(ValueError):
        fn({"variant": "environment", "result": {"source_target": "person", "direction": "", "identity": "astronaut", "environment": ""}})


def test_lora_reference_media_keeps_sheet_as_picture_one_and_environment_as_picture_two():
    source = ast.parse((ROOT / "iamccs_h3_character_swap_lora.py").read_text(encoding="utf-8"))
    fn = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "reference_media")
    namespace = {"__package__": "swap_test"}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<swap-media>", "exec"), namespace)
    import sys
    source = {"video": [1, 2], "fps": 24, "reference": ["sheet"], "environment": ["planet"]}
    modules = {"swap_test.iamccs_h3_face_swap": SimpleNamespace(FACE_SWAP_RESOURCE="source"),
               "swap_test.iamccs_minimax_h3_v2v_backend": SimpleNamespace(_segment_source_start=lambda *args: 0, _frame_indices=lambda **kw: ([0], None), _select_frames=lambda *args: [1], _fit_frames=lambda *args: [1], _slice_audio=lambda *args, **kw: None)}
    with patch.dict(sys.modules, modules):
        refs, _, _ = namespace["reference_media"]({"resources": {"source": source}}, {"width": 640, "height": 384}, {}, 0, 1)
        assert refs == [["sheet"], ["planet"], None, None]
        source["environment"] = None
        assert namespace["reference_media"]({"resources": {"source": source}}, {"width": 640, "height": 384}, {}, 0, 1)[0] == [["sheet"], None, None, None]


def test_atomic_conditioning_uses_the_environment_returned_by_swap_input():
    tree = ast.parse((ROOT / "iamccs_minimax_h3_atomic_backend.py").read_text(encoding="utf-8"))
    branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and isinstance(node.test, ast.Compare)
                  and ast.unparse(node.test) == "shotplan.get('task_mode') == 'v2va_face_swap_lora'"
                  and any(isinstance(item, ast.ImportFrom) and any(alias.name == "reference_media" for alias in item.names)
                          for item in node.body))
    statements = [node for node in branch.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id in {"external_images", "shotplan"} for target in node.targets)]
    for environment in (None, object()):
        refs = [object(), environment, None, None]
        ns = {"swap_images": refs, "shotplan": {}, "torch": SimpleNamespace(is_tensor=lambda value: value is not None)}
        exec(compile(ast.Module(body=statements, type_ignores=[]), "<swap-conditioning>", "exec"), ns)
        assert ns["external_images"] is refs
        assert ns["shotplan"]["reference_roles"] == ["subject_identity", "composition" if environment is not None else "disabled", "disabled", "disabled"]
