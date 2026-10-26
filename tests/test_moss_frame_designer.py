import importlib.util
import base64
import io
import json
from pathlib import Path
import unittest
import tempfile
import sys
import types
from unittest.mock import patch
from PIL import Image


spec = importlib.util.spec_from_file_location("moss_frame_designer", Path(__file__).resolve().parents[1] / "iamccs_moss_frame_designer.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MOSSLayoutTests(unittest.TestCase):
    def test_region_style_is_explicit_and_overlay_is_excluded_from_diffusion(self):
        design = {"canvas": {"width": 1280, "height": 720}, "items": [
            {"kind": "obj", "desc": "A cat", "style": "2D cel shading"},
            {"kind": "text", "text": "Exact title", "text_mode": "overlay"},
            {"kind": "text", "text": "Native title"},
        ]}
        raw, _, _, _, _, report = module.compile_layout(design)
        items = json.loads(raw)["items"]
        self.assertEqual(len(items), 2)
        self.assertIn("2D cel shading", items[0]["caption"])
        self.assertEqual(items[1]["text"], "Native title")
        self.assertIn("MOSS Result", report["warnings"][0])

    def test_result_notifications_are_strict_json_with_cached_inputs(self):
        package = types.ModuleType("moss_test_package")
        package.__path__ = [str(Path(__file__).resolve().parents[1])]
        routes = types.SimpleNamespace(get=lambda path: lambda fn: fn, post=lambda path: lambda fn: fn)
        server = types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(
            routes=routes, add_on_prompt_handler=lambda fn: None)))
        with patch.dict(sys.modules, {"moss_test_package": package, "moss_test_package.iamccs_moss_frame_designer": module,
                                      "server": server, "nodes": types.SimpleNamespace()}):
            source = Path(__file__).resolve().parents[1] / "iamccs_moss_frame_output.py"
            output_spec = importlib.util.spec_from_file_location("moss_test_package.output", source)
            output = importlib.util.module_from_spec(output_spec)
            output_spec.loader.exec_module(output)
            keys = [output.IAMCCS_MOSSFrameResult.IS_CHANGED() for _ in range(2)]
            self.assertNotEqual(*keys)
            json.dumps(keys, allow_nan=False)
            dirty = {"1": {"class_type": "IAMCCS_MOSSFrameDesigner", "is_changed": [float("nan")],
                           "inputs": {"design_data": json.dumps({**module.DEFAULT_DESIGN, "generated": {"data_url": "previous PNG"}})}}}
            clean, extra = output.render_metadata(dirty, {})
            json.dumps(clean, allow_nan=False)
            self.assertNotIn("generated", json.loads(clean["1"]["inputs"]["design_data"]))
            self.assertIn("is_changed", dirty["1"])

    def test_embedded_image_roundtrip_and_h3_reference_contract(self):
        buffer = io.BytesIO()
        Image.new("RGB", (32, 16), (80, 120, 160)).save(buffer, format="PNG")
        design = {"canvas": {"width": 32, "height": 16}, "h3": {"slot": 2, "role": "composition"},
                  "generated": {"data_url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()}}
        image = module.image_from_project(design)
        self.assertEqual(list(image.shape), [1, 16, 32, 3])
        linx = module.image_linx(design, image)
        self.assertIs(linx["resources"]["iamccs_minimax_h3_ref_image_2"], image)
        self.assertEqual(linx["resources"]["iamccs_minimax_h3_cine_info"]["reference_roles"],
                         ["disabled", "composition", "disabled", "disabled"])
        self.assertEqual(module.IAMCCS_MOSSFrameDesigner().compile(json.dumps(design))["result"][6]["type"], "IAMCCS_SUPERNODE_LINX")
        for target, key in (("swap_subject", "iamccs_h3_swap_reference"), ("swap_environment", "iamccs_h3_edit_environment")):
            design["h3"]["target"] = target
            self.assertEqual(module.image_linx(design, image)["resources"][key]["data_url"], design["generated"]["data_url"])

    def test_empty_project_does_not_fabricate_an_image(self):
        linx = module.image_linx(module.DEFAULT_DESIGN)
        self.assertFalse(linx["outputs"]["moss_image_ready"])
        self.assertNotIn("iamccs_minimax_h3_ref_image_1", linx["resources"])

    def test_embedded_image_validation(self):
        with self.assertRaises(ValueError):
            module.image_from_project({"generated": {"data_url": "file:///not-an-embedded-image.png"}})

    def test_local_asset_reload_and_path_containment(self):
        with tempfile.TemporaryDirectory() as folder:
            Image.new("RGB", (16, 32), "red").save(Path(folder) / "frame.png")
            with patch.object(module.folder_paths, "get_directory_by_type", return_value=folder):
                design = {"generated": {"files": [{"filename": "frame.png", "type": "output"}]}}
                self.assertEqual(list(module.image_from_project(design).shape), [1, 32, 16, 3])
                design["generated"]["files"][0]["filename"] = "../outside.png"
                with self.assertRaises(ValueError):
                    module.image_from_project(design)

    def test_non_square_coordinates_roles_and_literal_text(self):
        design = {"canvas": {"width": 1280, "height": 720}, "scene": {"high_level_description": "Film poster"}, "items": [
            {"kind": "obj", "role": "character", "x": 200, "y": 100, "w": 200, "h": 600, "desc": "Actor"},
            {"kind": "text", "x": 0, "y": 0, "w": 1000, "h": 100, "text": "IAMCCS \"MOSS\"", "desc": "Title"},
        ]}
        raw, width, height, elements, boxes, report = module.compile_layout(design)
        layout = json.loads(raw)
        self.assertEqual([i["bbox"] for i in layout["items"]], [[-288, -384, 144, -128], [-360, -640, -288, 640]])
        self.assertEqual(layout["items"][0]["label"], "character")
        self.assertEqual(layout["items"][1]["text"], 'IAMCCS "MOSS"')
        self.assertEqual(elements[0]["bbox"], [100, 200, 700, 400])
        self.assertEqual(boxes[0][0]["x"], 256)
        self.assertEqual(report["characters"], 1)

    def test_v2_import_keeps_description_palette_and_scene(self):
        source = {"schema": "iamccs.ideogram_storyboard_frame_designer", "canvas": {"width": 1024, "height": 1792},
                  "scene": {"high_level_description": "Casting portrait", "lighting": "Window light"},
                  "items": [{"kind": "obj", "x": 0, "y": 0, "w": 1000, "h": 1000, "desc": "One actor", "color_palette": ["#123456"]}]}
        original = json.dumps(source)
        raw = module.compile_layout(source)[0]
        layout = json.loads(raw)
        self.assertEqual(layout["items"][0]["bbox"], [-896, -512, 896, 512])
        self.assertEqual(layout["items"][0]["palette"], ["#123456"])
        self.assertIn("Window light", layout["summary"]["caption"])
        self.assertEqual(json.dumps(source), original)

    def test_reference_layers_do_not_become_generated_objects(self):
        design = {"canvas": {"width": 512, "height": 512}, "items": [
            {"kind": "image", "image_path": "reference.png"}, {"kind": "mask"}]}
        result = module.compile_layout(design)
        self.assertEqual(json.loads(result[0])["items"], [])
        self.assertEqual(len(result[5]["warnings"]), 2)

    def test_node_returns_authoritative_ui_prompt(self):
        result = module.IAMCCS_MOSSFrameDesigner().compile(json.dumps(module.DEFAULT_DESIGN))
        self.assertEqual(result["ui"]["moss_prompt"][0], result["result"][0])
        layout = json.loads(result["result"][0])
        self.assertEqual(len(layout["items"]), 2)
        self.assertLess(layout["items"][0]["bbox"][3], layout["items"][1]["bbox"][1])

    def test_invalid_geometry_is_reported(self):
        with self.assertRaises(ValueError):
            module.compile_layout({"canvas": {"width": 0, "height": 512}})
        with self.assertRaises(ValueError):
            module.compile_layout({"items": [{"w": -1, "h": 100}]})


if __name__ == "__main__":
    unittest.main()
