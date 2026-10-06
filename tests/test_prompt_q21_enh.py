import importlib.util
from pathlib import Path
import unittest
from unittest import mock


PATH = Path(__file__).parents[1] / "iamccs_prompt_q21_enh.py"
SPEC = importlib.util.spec_from_file_location("iamccs_prompt_q21_enh_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeImage:
    ndim = 4
    shape = (1, 256, 256, 3)

    def __len__(self):
        return 1

    def __getitem__(self, key):
        return self

    def movedim(self, source, destination):
        return self


class FakeClip:
    def __init__(self, answer):
        self.answer = answer
        self.token_call = None
        self.generate_call = None

    def tokenize(self, prompt, **kwargs):
        self.token_call = (prompt, kwargs)
        return "tokens"

    def generate(self, tokens, **kwargs):
        self.generate_call = (tokens, kwargs)
        return "ids"

    def decode(self, ids):
        return self.answer


class PromptQ21EnhTests(unittest.TestCase):
    def test_t2i_native_path_and_json(self):
        clip = FakeClip('<think>compose</think>{"rewritten_prompt":"A detailed frame","wh_ratio":"16:9"}')
        output = MODULE.IAMCCS_PromptQ21Enh().enhance(clip, "t2i", "a village", seed=7)
        self.assertEqual(output[:6], ("A detailed frame", "", "16:9", "", "compose", True))
        self.assertEqual(output[6:], (None, None, "general", None))
        self.assertEqual(clip.generate_call[1]["presence_penalty"], 1.5)
        self.assertEqual(clip.generate_call[1]["seed"], 7)
        self.assertEqual(clip.generate_call[1]["max_length"], 16256)
        self.assertTrue(clip.token_call[1]["thinking"])
        self.assertNotIn("images", clip.token_call[1])
        self.assertIn("Return one strictly valid JSON object", clip.token_call[0])

    def test_edit_images_and_ratio_exclusivity(self):
        clip = FakeClip('{"rewritten_prompt":"Replace the sky","wh_ratio":"4:3","ratio_follow":"<image1>"}')
        image = FakeImage()
        output = MODULE.IAMCCS_PromptQ21Enh().enhance(
            clip, "edit", "replace the sky", image_1=image, top_k=31
        )
        self.assertEqual(output[:6], ("Replace the sky", "", "4:3", "", "", True))
        self.assertIs(output[6], image)
        self.assertIsNone(output[7])
        self.assertEqual(output[8], "general")
        self.assertIsNone(output[9])
        self.assertEqual(len(clip.token_call[1]["images"]), 1)
        self.assertIn("<image1>", clip.token_call[0])
        self.assertEqual(clip.generate_call[1]["presence_penalty"], 0.0)
        self.assertEqual(clip.generate_call[1]["top_k"], 31)
        self.assertEqual(clip.generate_call[1]["max_length"], 24000)
        self.assertIn("Image Reference Rules", clip.token_call[0])

    def test_prompt_enhancer_caps_large_image_without_changing_generator_image(self):
        class LargeImage(FakeImage):
            shape = (1, 2048, 2048, 3)

        clip = FakeClip('{"rewritten_prompt":"Edit the frame"}')
        source = LargeImage()
        pe_image = FakeImage()
        with mock.patch.object(MODULE.comfy.utils, "common_upscale", return_value=pe_image) as upscale:
            output = MODULE.IAMCCS_PromptQ21Enh().enhance(
                clip, "edit", "edit the frame", image_1=source
            )
        self.assertIs(output[6], source)
        self.assertIs(clip.token_call[1]["images"][0], pe_image)
        target_width, target_height = upscale.call_args.args[1:3]
        self.assertLessEqual(target_width * target_height, 1024 * 1024)
        self.assertEqual(target_width % 32, 0)
        self.assertEqual(target_height % 32, 0)

    def test_profiles_and_explicit_token_limit(self):
        clip = FakeClip('{"rewritten_prompt":"Keep it concise"}')
        MODULE.IAMCCS_PromptQ21Enh().enhance(
            clip, "t2i", "idea", generation_profile="fast"
        )
        self.assertEqual(clip.generate_call[1]["max_length"], 4096)
        MODULE.IAMCCS_PromptQ21Enh().enhance(
            clip, "t2i", "idea", max_new_tokens=50000
        )
        self.assertEqual(clip.generate_call[1]["max_length"], 32768)

    def test_missing_image_is_explicit(self):
        clip = FakeClip("")
        with self.assertRaisesRegex(ValueError, "requires at least one"):
            MODULE.IAMCCS_PromptQ21Enh().enhance(clip, "edit", "change color")
        self.assertEqual(
            MODULE.IAMCCS_PromptQ21Enh().enhance(clip, "edit", "change color", missing_image_policy="pass_through")[0],
            "change color",
        )

    def test_sparse_image_slots_keep_prompt_references_aligned(self):
        clip = FakeClip('{"rewritten_prompt":"Use <image1>"}')
        MODULE.IAMCCS_PromptQ21Enh().enhance(
            clip, "edit", "Use the subject from <image3>", image_3=FakeImage()
        )
        self.assertIn("Use the subject from <image1>", clip.token_call[0])
        with self.assertRaisesRegex(ValueError, "not connected"):
            MODULE.IAMCCS_PromptQ21Enh().enhance(
                clip, "edit", "Use <image2>", image_3=FakeImage()
            )

    def test_three_generator_images_keep_routed_order(self):
        clip = FakeClip('{"rewritten_prompt":"Combine all three","ratio_follow":"<image1>"}')
        image_1, image_2, image_3 = FakeImage(), FakeImage(), FakeImage()
        output = MODULE.IAMCCS_PromptQ21Enh().enhance(
            clip, "edit", "Combine image 1, image 2 and image 3",
            image_1=image_1, image_2=image_2, image_3=image_3,
        )
        self.assertIs(output[6], image_1)
        self.assertIs(output[7], image_2)
        self.assertIs(output[9], image_3)
        self.assertEqual(len(clip.token_call[1]["images"]), 3)

    def test_outpaint_keeps_explicit_ratio(self):
        clip = FakeClip('{"rewritten_prompt":"Outpaint the image on both sides","wh_ratio":"16:9","ratio_follow":"<image1>"}')
        output = MODULE.IAMCCS_PromptQ21Enh().enhance(
            clip, "edit", "Outpaint image 1 to 16:9", image_1=FakeImage()
        )
        self.assertEqual(output[0], "Outpaint the image on both sides")
        self.assertEqual(output[2], "16:9")
        self.assertEqual(output[3], "")

    def test_fallback_and_empty(self):
        clip = FakeClip("plain response")
        self.assertEqual(MODULE.IAMCCS_PromptQ21Enh().enhance(clip, "t2i", "idea")[0], "plain response")
        self.assertFalse(MODULE.IAMCCS_PromptQ21Enh().enhance(clip, "t2i", "idea")[5])
        self.assertEqual(
            MODULE.IAMCCS_PromptQ21Enh().enhance(clip, "t2i", "  "),
            ("", "", "", "", "", True, None, None, "empty prompt", None),
        )


if __name__ == "__main__":
    unittest.main()
