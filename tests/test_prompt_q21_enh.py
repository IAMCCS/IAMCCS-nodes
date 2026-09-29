import importlib.util
from pathlib import Path
import unittest


PATH = Path(__file__).parents[1] / "iamccs_prompt_q21_enh.py"
SPEC = importlib.util.spec_from_file_location("iamccs_prompt_q21_enh_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeImage:
    ndim = 4

    def __len__(self):
        return 1

    def __getitem__(self, key):
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
        self.assertEqual(output, ("A detailed frame", "", "16:9", "", "compose", True))
        self.assertEqual(clip.generate_call[1]["presence_penalty"], 1.5)
        self.assertEqual(clip.generate_call[1]["seed"], 7)
        self.assertEqual(clip.generate_call[1]["max_length"], 768)
        self.assertFalse(clip.token_call[1]["thinking"])
        self.assertNotIn("images", clip.token_call[1])

    def test_edit_images_and_ratio_exclusivity(self):
        clip = FakeClip('{"rewritten_prompt":"Replace the sky","wh_ratio":"4:3","ratio_follow":"<image1>"}')
        output = MODULE.IAMCCS_PromptQ21Enh().enhance(
            clip, "edit", "replace the sky", image_1=FakeImage(), top_k=31
        )
        self.assertEqual(output, ("Replace the sky", "", "4:3", "", "", True))
        self.assertEqual(len(clip.token_call[1]["images"]), 1)
        self.assertIn("<image1>", clip.token_call[0])
        self.assertEqual(clip.generate_call[1]["presence_penalty"], 0.0)
        self.assertEqual(clip.generate_call[1]["top_k"], 31)
        self.assertEqual(clip.generate_call[1]["max_length"], 1024)

    def test_explicit_token_limit_is_clamped(self):
        clip = FakeClip('{"rewritten_prompt":"Keep it concise"}')
        MODULE.IAMCCS_PromptQ21Enh().enhance(
            clip, "t2i", "idea", max_new_tokens=24000
        )
        self.assertEqual(clip.generate_call[1]["max_length"], 4096)

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

    def test_fallback_and_empty(self):
        clip = FakeClip("plain response")
        self.assertEqual(MODULE.IAMCCS_PromptQ21Enh().enhance(clip, "t2i", "idea")[0], "plain response")
        self.assertFalse(MODULE.IAMCCS_PromptQ21Enh().enhance(clip, "t2i", "idea")[-1])
        self.assertEqual(MODULE.IAMCCS_PromptQ21Enh().enhance(clip, "t2i", "  "), ("", "", "", "", "", True))


if __name__ == "__main__":
    unittest.main()
