import importlib.util
from pathlib import Path
import unittest

import torch

spec = importlib.util.spec_from_file_location("moss_text_layers", Path(__file__).resolve().parents[1] / "moss_text_layers.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TextLayerTests(unittest.TestCase):
    def test_native_text_does_not_modify_image(self):
        images = torch.rand(1, 64, 128, 3)
        result, metadata = module.compose_text_layers(images, {"items": [{"kind": "text", "text": "MOSS"}]})
        self.assertIs(result, images)
        self.assertEqual(metadata, [])

    def test_overlay_changes_only_its_box_and_keeps_input_batch(self):
        images = torch.zeros(2, 128, 256, 3)
        design = {"items": [{"id": "title", "kind": "text", "text_mode": "overlay", "text": "IAMCCS\nMOSS",
                             "x": 250, "y": 250, "w": 500, "h": 500, "font_size": 80}]}
        result, metadata = module.compose_text_layers(images, design)
        self.assertEqual(result.shape, images.shape)
        self.assertEqual(images.sum().item(), 0)
        self.assertGreater(result[:, 32:96, 64:192].sum().item(), 0)
        self.assertEqual(result[:, :32].sum().item(), 0)
        self.assertEqual(result[:, 96:].sum().item(), 0)
        self.assertEqual(result[:, :, :64].sum().item(), 0)
        self.assertEqual(result[:, :, 192:].sum().item(), 0)
        self.assertTrue(torch.equal(result[0], result[1]))
        self.assertEqual(metadata[0]["text"], "IAMCCS\nMOSS")

    def test_invalid_box_is_reported(self):
        with self.assertRaises(ValueError):
            module.compose_text_layers(torch.zeros(1, 32, 32, 3), {"items": [
                {"kind": "text", "text_mode": "overlay", "text": "Title", "x": 2000}]})


if __name__ == "__main__":
    unittest.main()
