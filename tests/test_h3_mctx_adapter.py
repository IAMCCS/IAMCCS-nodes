from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest


ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("iamccs_h3_mctx_adapter_test", ROOT / "iamccs_h3_mctx_adapter.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class FakePinSpec:
    def build(self, **kwargs):
        item = {"source": kwargs["mctx"], "take_from": kwargs["take_from"], "place": kwargs["place"],
                "window": kwargs["window"], "mode": kwargs["mode"]}
        return (list(kwargs.get("pin_specs") or []) + [item],)


class MCtxAdapterTests(unittest.TestCase):
    def setUp(self):
        self.previous_nodes = sys.modules.get("nodes")
        sys.modules["nodes"] = types.SimpleNamespace(NODE_CLASS_MAPPINGS={"H3MCtxPinSpec": FakePinSpec})

    def tearDown(self):
        if self.previous_nodes is None:
            sys.modules.pop("nodes", None)
        else:
            sys.modules["nodes"] = self.previous_nodes

    def test_extend_uses_previous_tail_before(self):
        stack, report = adapter.IAMCCS_H3MCtxRoute().route("extend", "39", "masked", 0, 0.0, 0.0, previous_mctx={"id": "a"})
        self.assertEqual((stack[0]["take_from"], stack[0]["place"]), ("tail", "before"))
        self.assertIn('"operation": "extend"', report)

    def test_prepend_uses_next_head_after_and_defaults_can_be_both(self):
        stack, _ = adapter.IAMCCS_H3MCtxRoute().route("prepend", "39", "both", 0, 0.0, 0.0, next_mctx={"id": "b"})
        self.assertEqual((stack[0]["take_from"], stack[0]["place"], stack[0]["mode"]), ("head", "after", "both"))

    def test_bridge_builds_one_pin_per_side(self):
        stack, _ = adapter.IAMCCS_H3MCtxRoute().route("bridge", "39", "both", 0, 0.0, 0.0,
                                                        previous_mctx={"id": "a"}, next_mctx={"id": "b"})
        self.assertEqual([(item["place"], item["take_from"]) for item in stack],
                         [("before", "tail"), ("after", "head")])

    def test_missing_required_take_fails_with_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "PREVIOUS MCTX"):
            adapter.IAMCCS_H3MCtxRoute().route("extend", "39", "masked", 0, 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
