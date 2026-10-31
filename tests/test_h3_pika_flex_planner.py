"""Exercise the real core planner before configuring the Pika sampling recipe."""
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORE = load("iamccs_minimax_h3_shotboard_core")
PIKA = load("iamccs_h3_pika_flex")


def compile_plan(mode, acceleration):
    return CORE.build_shotplan(
        timeline_data={"fps": 24, "rows": []},
        global_prompt="A knight walks through the forest.",
        duration_seconds=5,
        task_mode=mode,
        width=864,
        height=480,
        acceleration=acceleration,
    )


class PikaPlannerTests(unittest.TestCase):
    def test_pika_reaches_configuration_through_real_planner(self):
        for mode in ("t2va", "i2va", "fl2va", "ref2va"):
            with self.subTest(mode=mode):
                plan = compile_plan(mode, PIKA.PROFILE)
                plan["sampling"] = {"seed": 2026080901, "steps": 30}
                PIKA.configure(plan, {
                    "turbo_mode": "off",
                    "pika_flex_fl2va_lora": "fl.safetensors",
                    "pika_flex_ref2va_lora": "ref.safetensors",
                })
                self.assertEqual(plan["acceleration"], PIKA.PROFILE)
                self.assertTrue(PIKA.active(plan))
                self.assertEqual(plan["sampling"], {"seed": 2026080901, **PIKA.VALUES})
                self.assertTrue(plan["chunks"])
                self.assertEqual(plan["pika_flex"]["sigmas"], PIKA.SIGMAS)

    def test_existing_acceleration_plans_are_unchanged(self):
        native = compile_plan("t2va", "native")
        for profile in ("native", "fasth3_dense_6step", "pdd_native_8step", "h3_sage"):
            with self.subTest(profile=profile):
                plan = compile_plan("t2va", profile)
                self.assertEqual(plan["acceleration"], profile)
                self.assertEqual(plan["chunks"], native["chunks"])
                plan["sampling"] = {"steps": 20}
                PIKA.configure(plan, {})
                self.assertFalse(PIKA.active(plan))
                self.assertEqual(plan["sampling"], {"steps": 20})

    def test_unknown_acceleration_still_fails(self):
        with self.assertRaisesRegex(ValueError, "pika_h3_flex_typo"):
            compile_plan("t2va", "pika_h3_flex_typo")


if __name__ == "__main__":
    unittest.main()
