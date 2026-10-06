import json
from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]
WORKFLOW_DIR = ROOT / "workflows" / "qwen_image_2_1_prompt_enhancer"
WORKFLOWS = {
    "t2i": WORKFLOW_DIR / "01_q21_prompt_enh_t2i_smoke.json",
    "image_edit": WORKFLOW_DIR / "02_q21_prompt_enh_image_edit_smoke.json",
    "outpaint": WORKFLOW_DIR / "03_q21_prompt_enh_outpaint_smoke.json",
    "two_image_edit": WORKFLOW_DIR / "04_q21_prompt_enh_two_image_edit_smoke.json",
}


class PromptQ21WorkflowTests(unittest.TestCase):
    def load(self, mode):
        with WORKFLOWS[mode].open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def node_types(self, workflow):
        return [node["type"] for node in workflow["nodes"]]

    def only_node(self, workflow, node_type):
        matches = [node for node in workflow["nodes"] if node["type"] == node_type]
        self.assertEqual(len(matches), 1, f"expected one {node_type}, found {len(matches)}")
        return matches[0]

    def source_for_input(self, workflow, target, input_name):
        input_slot = next(i for i, item in enumerate(target["inputs"]) if item["name"] == input_name)
        link_id = target["inputs"][input_slot]["link"]
        link = next(item for item in workflow["links"] if item[0] == link_id)
        source = next(node for node in workflow["nodes"] if node["id"] == link[1])
        return source, source["outputs"][link[2]]["name"]

    def test_workflow_files_are_structurally_consistent(self):
        for mode in WORKFLOWS:
            with self.subTest(mode=mode):
                workflow = self.load(mode)
                self.assertEqual(workflow["version"], 0.4)
                node_by_id = {node["id"]: node for node in workflow["nodes"]}
                self.assertEqual(len(node_by_id), len(workflow["nodes"]))
                link_ids = set()
                for link in workflow["links"]:
                    link_id, source_id, output_slot, target_id, input_slot, data_type = link
                    self.assertNotIn(link_id, link_ids)
                    link_ids.add(link_id)
                    source = node_by_id[source_id]
                    target = node_by_id[target_id]
                    self.assertLess(output_slot, len(source["outputs"]))
                    self.assertLess(input_slot, len(target["inputs"]))
                    self.assertIn(link_id, source["outputs"][output_slot]["links"])
                    self.assertEqual(target["inputs"][input_slot]["link"], link_id)
                    self.assertEqual(source["outputs"][output_slot]["type"], data_type)
                    self.assertEqual(target["inputs"][input_slot]["type"], data_type)

    def test_enhancer_is_connected_to_qwen_encoder(self):
        for mode in WORKFLOWS:
            with self.subTest(mode=mode):
                workflow = self.load(mode)
                enhancer = self.only_node(workflow, "IAMCCS_PromptQ21Enh")
                encoder = self.only_node(workflow, "TextEncodeQwenImage21")
                self.assertEqual(len(enhancer["outputs"]), 10)
                self.assertEqual(enhancer["widgets_values"][-1], "official")
                prompt_source, prompt_output = self.source_for_input(workflow, encoder, "prompt")
                negative_source, negative_output = self.source_for_input(workflow, encoder, "negative_prompt")
                self.assertEqual(prompt_source["id"], enhancer["id"])
                self.assertEqual(prompt_output, "positive_prompt")
                self.assertEqual(negative_source["id"], enhancer["id"])
                self.assertEqual(negative_output, "negative_prompt")

    def test_mode_specific_reference_and_latent_routing(self):
        t2i = self.load("t2i")
        self.assertNotIn("LoadImage", self.node_types(t2i))
        t2i_enhancer = self.only_node(t2i, "IAMCCS_PromptQ21Enh")
        self.assertEqual(t2i_enhancer["widgets_values"][0], "t2i")
        t2i_sampler = self.only_node(t2i, "KSampler")
        latent_source, latent_output = self.source_for_input(t2i, t2i_sampler, "latent_image")
        self.assertEqual((latent_source["type"], latent_output), ("EmptyLatentImage", "LATENT"))

        edit = self.load("image_edit")
        self.assertEqual(self.node_types(edit).count("LoadImage"), 1)
        edit_encoder = self.only_node(edit, "TextEncodeQwenImage21")
        image_source, image_output = self.source_for_input(edit, edit_encoder, "images.image_1")
        self.assertEqual((image_source["type"], image_output), ("IAMCCS_PromptQ21Enh", "generator_image_1"))

        outpaint = self.load("outpaint")
        self.assertIn("ImagePadForOutpaint", self.node_types(outpaint))
        outpaint_enhancer = self.only_node(outpaint, "IAMCCS_PromptQ21Enh")
        padded_source, padded_output = self.source_for_input(outpaint, outpaint_enhancer, "image_1")
        self.assertEqual((padded_source["type"], padded_output), ("ImagePadForOutpaint", "IMAGE"))

        two_image = self.load("two_image_edit")
        self.assertEqual(self.node_types(two_image).count("LoadImage"), 2)
        two_enhancer = self.only_node(two_image, "IAMCCS_PromptQ21Enh")
        self.assertEqual(two_enhancer["widgets_values"][4], "subject1_on_canvas2")
        two_encoder = self.only_node(two_image, "TextEncodeQwenImage21")
        first_source, first_output = self.source_for_input(two_image, two_encoder, "images.image_1")
        second_source, second_output = self.source_for_input(two_image, two_encoder, "images.image_2")
        self.assertEqual(first_source["id"], two_enhancer["id"])
        self.assertEqual(second_source["id"], two_enhancer["id"])
        self.assertEqual((first_output, second_output), ("generator_image_1", "generator_image_2"))


if __name__ == "__main__":
    unittest.main()
