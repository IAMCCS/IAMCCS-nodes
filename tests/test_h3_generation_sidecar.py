"""Exercise the sidecar writer without importing the GPU-dependent package."""

import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def test_native_sidecar_has_plan_and_importable_workflow_source():
    source = ast.parse((ROOT / "iamccs_minimax_h3_shotboard.py").read_text(encoding="utf-8"))
    functions = [
        node for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name in {
            "_segment_meta_path", "_sidecar_json_safe", "_write_segment_metadata",
        }
    ]
    namespace = {"Any": object, "Path": Path, "torch": SimpleNamespace(is_tensor=lambda _value: False), "json": json}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "<sidecar>", "exec"), namespace)
    with TemporaryDirectory() as directory:
        video = Path(directory) / "segment_0001.mp4"
        workflow = {"nodes": [{"id": 1, "type": "IAMCCS"}], "links": []}
        namespace["_write_segment_metadata"](
            video, 102, 24, "native_av_direct_join",
            provenance={"render_id": "run_1", "shotplan": {"task_mode": "longvid_motion_context", "chunks": [{"index": 0}]}},
            source_workflow=workflow,
        )
        sidecar = json.loads(video.with_suffix(".mp4.iamccs.json").read_text(encoding="utf-8"))
        assert sidecar["frame_count"] == 102
        assert sidecar["generation"]["shotplan"]["task_mode"] == "longvid_motion_context"
        restored = json.loads((Path(directory) / sidecar["source_workflow_file"]).read_text(encoding="utf-8"))
        assert restored == workflow


def test_master_frame_count_uses_capped_overlap_for_short_segments():
    source = ast.parse((ROOT / "iamccs_minimax_h3_shotboard.py").read_text(encoding="utf-8"))
    function = next(
        node for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == "_joined_frame_count"
    )
    counts = {"one": 102, "two": 3, "three": 36}
    namespace = {"Path": Path, "_read_segment_frame_count": lambda path: counts[str(path)]}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<join-count>", "exec"), namespace)
    paths = [Path(name) for name in counts]
    assert namespace["_joined_frame_count"](paths, 8) == 102 + 3 + 36 - 2 - 2
    assert namespace["_joined_frame_count"](paths, 0) == 102 + 3 + 36
