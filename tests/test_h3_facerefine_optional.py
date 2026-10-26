"""Exercise optional FaceRefine registration without loading models or ComfyUI."""
import ast
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
ALIAS_NAMES = (
    "H3FaceTrackCrop", "H3FaceStitch", "H3InjectVideoLatent",
    "H3PerFrameDenoise", "H3FaceMaskSAM", "H3FaceTransformInfo",
)


class FaceRefineOptionalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.modules = patch.dict(sys.modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.package = "_iamccs_optional_face_test"
        package = types.ModuleType(self.package)
        package.__path__ = [str(ROOT)]
        sys.modules[self.package] = package
        for name in ("iamccs_prompter", "iamccs_minimax_h3_atomic_backend"):
            module = types.ModuleType(f"{self.package}.{name}")
            module.SUPERNODE_LINX_TYPE = "SUPERNODE_LINX"
            module._resolve_shotplan = lambda linx: linx
            sys.modules[module.__name__] = module
        comfy_nodes = types.ModuleType("nodes")
        comfy_nodes.NODE_CLASS_MAPPINGS = {}
        sys.modules["nodes"] = comfy_nodes
        sys.modules.pop("iamccs_external_h3_face_refine", None)

    def load_detailer(self):
        name = f"{self.package}.iamccs_minimax_h3_face_detailer"
        module = types.ModuleType(name)
        module.__file__ = str(Path(self.temp.name) / "IAMCCS-nodes" / "iamccs_minimax_h3_face_detailer.py")
        module.__package__ = self.package
        sys.modules[name] = module
        source = ROOT / "iamccs_minimax_h3_face_detailer.py"
        exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), module.__dict__)
        return module

    def delivery_registration(self):
        # Execute the real registration tail, excluding unrelated pixel inference code.
        source = ROOT / "iamccs_minimax_h3_pixel_refine_variant.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        start = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.ImportFrom)
                     and node.module == "iamccs_minimax_h3_face_delivery_variant")
        tail = ast.Module(body=tree.body[start:], type_ignores=[])
        namespace = {"__package__": self.package, "NODE_CLASS_MAPPINGS": {}, "NODE_DISPLAY_NAME_MAPPINGS": {}}
        exec(compile(tail, str(source), "exec"), namespace)
        return namespace

    def test_absent_dependency_registers_only_independent_router(self):
        detailer = self.load_detailer()
        self.assertEqual(set(detailer.NODE_CLASS_MAPPINGS), {"IAMCCS_MiniMaxH3FaceDetailerRouter"})
        for node in detailer.NODE_CLASS_MAPPINGS.values():
            self.assertIn("required", node.INPUT_TYPES())
        delivery = self.delivery_registration()
        self.assertEqual(delivery["NODE_CLASS_MAPPINGS"], {})
        self.assertEqual(delivery["NODE_DISPLAY_NAME_MAPPINGS"], {})

    def test_disabled_router_passes_native_objects_without_dependency(self):
        detailer = self.load_detailer()
        router = detailer.IAMCCS_MiniMaxH3FaceDetailerRouter()
        frames, audio = object(), {"waveform": object()}
        result = router.route({}, frames, audio)
        self.assertIs(result[0], frames)
        self.assertIs(result[1], audio)
        self.assertEqual(router.check_lazy_status({}, frames, audio), [])

    def test_explicit_use_still_reports_missing_dependency(self):
        detailer = self.load_detailer()
        with self.assertRaisesRegex(RuntimeError, "before using IAMCCS H3 Face Detailer"):
            detailer.IAMCCS_MiniMaxH3FaceTrackCrop().run(images=object())

    def test_registered_provider_preserves_alias_schema_and_execution(self):
        schema = {"required": {"images": ("IMAGE",), "detector": (["installed-detector"],)}}
        class Provider:
            @classmethod
            def INPUT_TYPES(cls):
                return schema

            def run(self, **kwargs):
                return (kwargs["images"],)

        sys.modules["nodes"].NODE_CLASS_MAPPINGS = dict.fromkeys(ALIAS_NAMES, Provider)
        detailer = self.load_detailer()
        self.assertEqual(len(detailer.NODE_CLASS_MAPPINGS), 7)
        for name in ALIAS_NAMES:
            alias = detailer.NODE_CLASS_MAPPINGS[f"IAMCCS_MiniMax{name}"]
            result = alias.INPUT_TYPES()
            self.assertEqual(result, schema)
            self.assertIsNot(result, schema)
            marker = object()
            self.assertIs(alias().run(images=marker)[0], marker)
        self.assertEqual(len(self.delivery_registration()["NODE_CLASS_MAPPINGS"]), 2)

    def test_installed_package_registers_aliases_without_eager_import(self):
        provider = Path(self.temp.name) / "ComfyUI-H3-FaceRefine" / "nodes.py"
        provider.parent.mkdir()
        provider.write_text("raise AssertionError('Provider imported during registration')", encoding="utf-8")
        detailer = self.load_detailer()
        self.assertEqual(len(detailer.NODE_CLASS_MAPPINGS), 7)
        self.assertEqual(len(self.delivery_registration()["NODE_CLASS_MAPPINGS"]), 2)
        self.assertNotIn("iamccs_external_h3_face_refine", sys.modules)


if __name__ == "__main__":
    unittest.main()
