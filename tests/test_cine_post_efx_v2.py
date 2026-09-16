from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest

import torch


MODULE_PATH = Path(__file__).parents[1] / "cine_nodes" / "temporal_film_grain.py"
SPEC = importlib.util.spec_from_file_location("iamccs_temporal_film_grain_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _image() -> torch.Tensor:
    y = torch.linspace(0.0, 1.0, 48).view(48, 1)
    x = torch.linspace(0.0, 1.0, 64).view(1, 64)
    red = x.expand(48, 64)
    green = y.expand(48, 64)
    blue = ((x + y) * 0.5).clamp(0.0, 1.0)
    image = torch.stack((red, green, blue), dim=-1)
    image[12:28, 20:44] = torch.tensor((0.95, 0.72, 0.18))
    return image.unsqueeze(0)


def _settings(**overrides):
    values = {
        "engine": "film_emulsion",
        "preset": "custom_box_values",
        "input_transfer": "srgb",
        "blend_method": "linear_additive",
        "strength": 0.0,
        "grain_size_4k_px": 1.0,
        "softness": 0.18,
        "roughness": 0.24,
        "complexity": 3,
        "temporal_correlation": 0.0,
        "chroma_amount": 0.05,
        "shadow_response": 0.62,
        "midtone_response": 1.0,
        "highlight_response": 0.38,
        "red_response": 1.0,
        "green_response": 1.0,
        "blue_response": 1.0,
        "seed": 12345,
        "frame_start": 7,
        "horizontal_instability": 0.0,
        "chroma_spill": 0.0,
        "luma_noise": 0.0,
        "chroma_noise": 0.0,
        "color_drift": 0.0,
        "edge_echo": 0.0,
        "line_dropout": 0.0,
        "scanline_strength": 0.0,
        "field_interlace": 0.0,
        "vertical_roll": 0.0,
        "highlight_glow": 0.0,
        "head_switch_distortion": 0.0,
        "luma_trail": 0.0,
        "chroma_delay": 0.0,
        "analog_mix": 1.0,
        "tracking_error": 0.0,
        "tape_warp": 0.0,
        "ghost_echo": 0.0,
        "signal_saturation": 1.0,
        "black_lift": 0.0,
        "chroma_phase_noise": 0.0,
        "chroma_loss": 0.0,
        "grain_profile": "negative_stock",
        "texture_microcontrast": 0.18,
        "analog_color_look": "neutral",
        "disturbance_amount": 1.0,
        "lens_preset": "lens_none",
        "lens_master": 0.0,
        "lens_distortion": 0.0,
        "lens_edge_stretch": 0.0,
        "lens_anamorphic_width": 0.0,
        "lens_zoom": 1.0,
        "lens_keystone_x": 0.0,
        "lens_keystone_y": 0.0,
        "lens_tilt_angle": 0.0,
        "lens_focus_position": 0.0,
        "lens_tilt_blur": 0.0,
        "lens_chromatic_aberration": 0.0,
        "lens_vignette": 0.0,
        "settings_json": "",
    }
    values.update(overrides)
    return values


class CinePostEfxV2Tests(unittest.TestCase):
    def test_each_analogue_effect_changes_the_frame(self):
        image = _image()[0]
        for effect in MODULE.ANALOG_FIELDS:
            with self.subTest(effect=effect):
                settings = _settings(**{effect: 1.0})
                generator = torch.Generator(device=image.device).manual_seed(12345)
                result = MODULE._apply_cine_post_effects(image, settings, 7, generator)
                self.assertEqual(result.shape, image.shape)
                self.assertTrue(torch.isfinite(result).all())
                self.assertFalse(torch.equal(result, image), effect)

    def test_backend_presets_are_render_truth(self):
        image = _image()
        presets = ("digital_cinema_sensor", "soft_cassette_memory", "worn_video_copy", "midnight_airwave", "projector_to_tape")
        for preset in presets:
            with self.subTest(preset=preset):
                result, grain_map, report = MODULE.IAMCCS_CinePostEfxV2().apply(image, **_settings(preset=preset))
                self.assertEqual(result.shape, image.shape)
                self.assertEqual(grain_map.shape, image.shape[:3])
                self.assertFalse(torch.equal(result, image), preset)
                self.assertIn(f"preset={preset}", report)

    def test_film_grain_is_visibly_above_rounding_noise(self):
        image = _image()
        clean, _map, _report = MODULE.IAMCCS_CinePostEfxV2().apply(
            image, **_settings(preset="35mm_fine_negative")
        )
        documentary, _map, _report = MODULE.IAMCCS_CinePostEfxV2().apply(
            image, **_settings(preset="16mm_documentary")
        )
        clean_delta = float((clean - image).abs().mean())
        documentary_delta = float((documentary - image).abs().mean())
        self.assertGreater(clean_delta, 0.004)
        self.assertGreater(documentary_delta, clean_delta * 2.0)

    def test_tape_and_broadcast_presets_have_distinct_signatures(self):
        image = _image()
        names = (
            "family_camcorder_1988",
            "overplayed_rental_tape",
            "late_night_relay",
            "damaged_airwave",
        )
        rendered = {
            name: MODULE.IAMCCS_CinePostEfxV2().apply(image, **_settings(preset=name))[0]
            for name in names
        }
        pairs = (
            ("family_camcorder_1988", "overplayed_rental_tape"),
            ("late_night_relay", "damaged_airwave"),
            ("family_camcorder_1988", "late_night_relay"),
        )
        for first, second in pairs:
            with self.subTest(first=first, second=second):
                difference = float((rendered[first] - rendered[second]).abs().mean())
                self.assertGreater(difference, 0.025)

    def test_surveillance_presets_are_registered_and_monochrome_is_real(self):
        required = {
            "cctv_monochrome_1997",
            "parking_garage_cctv",
            "camcorder_night_recording",
            "industrial_monitor_feed",
        }
        self.assertTrue(required.issubset(MODULE.PRO_PRESET_VALUES))
        image = _image()
        result, _map, report = MODULE.IAMCCS_CinePostEfxV2().apply(
            image, **_settings(preset="cctv_monochrome_1997")
        )
        self.assertLess(float(result.std(dim=-1).mean()), 1e-6)
        self.assertIn("preset=cctv_monochrome_1997", report)

    def test_vectorized_emulsion_layers_are_normalized_and_independent(self):
        generator = torch.Generator().manual_seed(19)
        layers = MODULE._pro_emulsion_layers(
            48, 64, 1.4, 0.20, 0.45, 4, 3, "negative_stock", generator, torch.device("cpu")
        )
        self.assertEqual(layers.shape, (1, 3, 48, 64))
        self.assertTrue(torch.isfinite(layers).all())
        self.assertLess(float(layers.mean(dim=(-2, -1)).abs().max()), 1e-5)
        self.assertGreater(float((layers[:, 0] - layers[:, 1]).abs().mean()), 0.25)

    def test_grain_profiles_and_analogue_color_looks_are_distinct(self):
        fields = []
        for profile in MODULE.GRAIN_PROFILES:
            generator = torch.Generator().manual_seed(23)
            fields.append(MODULE._pro_emulsion_layers(
                48, 64, 1.3, 0.18, 0.46, 4, 1, profile, generator, torch.device("cpu")
            ))
        self.assertGreater(float((fields[0] - fields[2]).abs().mean()), 0.02)

        image = _image()
        names = (
            "vhs_consumer_color_pop", "vhs_warm_family_tape", "vhs_cool_camcorder",
            "vhs_magenta_generation_loss", "crt_saturated_broadcast", "night_vision_cctv_green",
        )
        outputs = {
            name: MODULE.IAMCCS_CinePostEfxV2().apply(image, **_settings(preset=name))[0]
            for name in names
        }
        warm = outputs["vhs_warm_family_tape"]
        cool = outputs["vhs_cool_camcorder"]
        self.assertGreater(float((warm - cool).abs().mean()), 0.025)
        night = outputs["night_vision_cctv_green"]
        channel_means = night[..., :3].mean(dim=(0, 1, 2))
        self.assertGreater(float(channel_means[1]), float(channel_means[0]) * 1.5)
        self.assertGreater(float(channel_means[1]), float(channel_means[2]) * 1.5)

    def test_analogue_presets_have_deliberate_colour_signatures(self):
        neutral_exceptions = {"cctv_monochrome_1997"}
        analogue = {
            name: values for name, values in MODULE.PRO_PRESET_VALUES.items()
            if values["engine"] in ("analog_tape", "broadcast_signal")
        }
        for name, values in analogue.items():
            with self.subTest(preset=name):
                if name in neutral_exceptions:
                    self.assertEqual(values["analog_color_look"], "neutral")
                else:
                    self.assertNotEqual(values["analog_color_look"], "neutral")

        image = _image()[0]
        looks = (
            "tungsten_home_video", "fluorescent_cctv", "sodium_vapor_cctv",
            "late_night_blue", "sun_bleached_tape", "rf_cyan_fade", "archival_amber",
        )
        outputs = [MODULE._apply_analog_color_look(image, look) for look in looks]
        for first, second in zip(outputs, outputs[1:]):
            self.assertGreater(float((first - second).abs().mean()), 0.035)

    def test_master_disturbance_scales_all_signal_defects_but_keeps_color_look(self):
        image = _image()[0]
        settings = _settings(
            engine="analog_tape",
            analog_color_look="warm_camcorder",
            horizontal_instability=0.18,
            tape_warp=0.22,
            luma_noise=0.24,
            chroma_noise=0.18,
            scanline_strength=0.20,
            ghost_echo=0.16,
        )
        color_only = MODULE._apply_analog_color_look(image, "warm_camcorder")
        outputs = {}
        for amount in (0.0, 1.0, 2.0):
            generator = torch.Generator().manual_seed(12345)
            outputs[amount] = MODULE._apply_cine_post_effects(
                image, {**settings, "disturbance_amount": amount}, 7, generator
            )
        self.assertTrue(torch.allclose(outputs[0.0], color_only, atol=1e-6))
        normal_delta = float((outputs[1.0] - color_only).abs().mean())
        boosted_delta = float((outputs[2.0] - color_only).abs().mean())
        self.assertGreater(normal_delta, 0.01)
        self.assertGreater(boosted_delta, normal_delta * 1.20)

    def test_tv_scanlines_render_at_the_preview_signal_scale(self):
        height, width = 324, 576
        image = torch.full((height, width, 3), 0.65)
        settings = _settings(
            engine="broadcast_signal",
            scanline_strength=0.70,
            disturbance_amount=1.0,
        )
        result = MODULE._apply_cine_post_effects(
            image, settings, 7, torch.Generator().manual_seed(12345)
        )
        row_signal = result.mean(dim=(1, 2))
        spectrum = torch.fft.rfft(row_signal - row_signal.mean()).abs()
        peak = int(torch.argmax(spectrum[2:]).item() + 2)
        self.assertGreaterEqual(peak, 78)
        self.assertLessEqual(peak, 84)

        bypass = MODULE._apply_cine_post_effects(
            image,
            {**settings, "analog_mix": 0.0},
            7,
            torch.Generator().manual_seed(12345),
        )
        self.assertTrue(torch.equal(bypass, image))

    def test_ui_snapshot_is_the_render_truth_for_preview_parity(self):
        image = _image()
        preview_values = _settings(
            engine="broadcast_signal",
            preset="custom_box_values",
            input_transfer="linear",
            strength=0.0,
            scanline_strength=0.73,
            field_interlace=0.41,
            luma_noise=0.04,
            chroma_noise=0.02,
            disturbance_amount=1.35,
            analog_color_look="crt_broadcast",
            seed=98765,
            frame_start=13,
        )
        snapshot = {
            name: value for name, value in preview_values.items()
            if name not in {"preset", "settings_json"}
        }
        direct, _map, _report = MODULE.IAMCCS_CinePostEfxV2().apply(
            image, **preview_values
        )
        queued, _map, report = MODULE.IAMCCS_CinePostEfxV2().apply(
            image,
            **{
                **_settings(preset="damaged_airwave"),
                "settings_json": json.dumps(snapshot),
            },
        )
        self.assertTrue(torch.equal(queued, direct))
        self.assertIn("ui_snapshot=yes", report)

    def test_temporal_randomness_is_independent_deterministic_and_backward_compatible(self):
        frame = _image()[0]
        base = _settings(
            engine="analog_tape",
            strength=0.0,
            horizontal_instability=0.62,
            tracking_error=0.58,
            tape_warp=0.54,
            color_drift=0.48,
            scanline_strength=0.52,
            field_interlace=0.46,
            vertical_roll=0.70,
            head_switch_distortion=0.64,
        )
        zero_randomness = {
            **base,
            **{name: 0.0 for name in MODULE.TEMPORAL_RANDOMNESS_FIELDS},
        }
        full_randomness = {
            **base,
            **{name: 1.0 for name in MODULE.TEMPORAL_RANDOMNESS_FIELDS},
        }

        def render(settings, frame_index):
            generator = torch.Generator().manual_seed(7419)
            return MODULE._apply_cine_post_effects(frame, settings, frame_index, generator)

        legacy = render(base, 17)
        explicit_zero = render(zero_randomness, 17)
        random_a = render(full_randomness, 17)
        random_b = render(full_randomness, 17)
        random_next = render(full_randomness, 18)

        self.assertTrue(torch.equal(legacy, explicit_zero))
        self.assertTrue(torch.equal(random_a, random_b))
        self.assertGreater(float((random_a - explicit_zero).abs().mean()), 0.001)
        self.assertGreater(float((random_next - random_a).abs().mean()), 0.001)

    def test_analogue_presets_include_per_effect_temporal_randomness(self):
        preset = MODULE.PRO_PRESET_VALUES["overplayed_rental_tape"]
        for randomness_name in MODULE.TEMPORAL_RANDOMNESS_FIELDS:
            self.assertIn(randomness_name, preset)
            self.assertGreaterEqual(float(preset[randomness_name]), 0.0)
            self.assertLessEqual(float(preset[randomness_name]), 1.0)
        self.assertGreater(float(preset["vertical_roll_randomness"]), 0.0)
        self.assertGreater(float(preset["tracking_randomness"]), 0.0)

    def test_ui_snapshot_allows_exact_zero_disturbance_without_legacy_fallback(self):
        image = _image()
        preview_values = _settings(
            engine="analog_tape",
            preset="custom_box_values",
            strength=0.0,
            disturbance_amount=0.0,
        )
        snapshot = {
            name: value for name, value in preview_values.items()
            if name not in {"preset", "settings_json"}
        }
        rendered, _map, report = MODULE.IAMCCS_CinePostEfxV2().apply(
            image,
            **{
                **_settings(engine="analog_tape", preset="soft_cassette_memory"),
                "settings_json": json.dumps(snapshot),
            },
        )
        self.assertTrue(torch.equal(rendered, image))
        self.assertIn("ui_snapshot=yes", report)

    def test_tilt_focus_uses_resolution_aware_smooth_defocus(self):
        height, width = 384, 512
        y = torch.arange(height).view(height, 1)
        x = torch.arange(width).view(1, width)
        checker = (((x // 8 + y // 8) % 2).to(torch.float32) * 0.90 + 0.05)
        image = torch.stack((checker, checker, checker), dim=-1)
        settings = _settings(
            lens_master=1.0,
            lens_tilt_blur=1.0,
            lens_tilt_angle=0.0,
            lens_focus_position=0.0,
        )
        result = MODULE._apply_lens_effects(image, settings)
        source_detail = float((image[:, 1:] - image[:, :-1]).abs().mean())
        focus_detail = float((result[height // 2 - 12 : height // 2 + 12, 1:] - result[height // 2 - 12 : height // 2 + 12, :-1]).abs().mean())
        defocused_detail = float((result[:24, 1:] - result[:24, :-1]).abs().mean())
        self.assertGreater(focus_detail, source_detail * 0.90)
        self.assertLess(defocused_detail, focus_detail * 0.08)
        self.assertTrue(torch.isfinite(result).all())

    def test_professional_lens_presets_are_finite_and_geometrically_distinct(self):
        image = _image()
        names = (
            "wide_angle_14mm", "fisheye_8mm", "anamorphic_2x_cinema",
            "tilt_shift_architecture", "tilt_shift_miniature", "telephoto_pincushion",
        )
        outputs = {}
        for name in names:
            with self.subTest(preset=name):
                result, _grain_map, report = MODULE.IAMCCS_CinePostEfxV2().apply(
                    image, **_settings(lens_preset=name)
                )
                self.assertTrue(torch.isfinite(result).all())
                self.assertFalse(torch.equal(result, image))
                self.assertIn(f"lens={name}", report)
                outputs[name] = result
        self.assertGreater(float((outputs["fisheye_8mm"] - outputs["telephoto_pincushion"]).abs().mean()), 0.035)
        self.assertGreater(float((outputs["anamorphic_2x_cinema"] - outputs["tilt_shift_architecture"]).abs().mean()), 0.025)

    def test_lens_master_zero_is_exact_bypass_and_alpha_stays_untouched(self):
        image = _image()
        alpha = torch.linspace(0.0, 1.0, image.shape[1] * image.shape[2]).view(1, image.shape[1], image.shape[2], 1)
        rgba = torch.cat((image, alpha), dim=-1)
        bypass, _map, _report = MODULE.IAMCCS_CinePostEfxV2().apply(
            rgba, **_settings(lens_preset="custom_lens", lens_master=0.0, lens_distortion=1.0)
        )
        self.assertTrue(torch.equal(bypass, rgba))
        warped, _map, _report = MODULE.IAMCCS_CinePostEfxV2().apply(
            rgba, **_settings(lens_preset="fisheye_8mm")
        )
        self.assertFalse(torch.equal(warped[..., :3], rgba[..., :3]))
        self.assertTrue(torch.equal(warped[..., 3:4], alpha))

    def test_non_film_engines_are_active(self):
        image = _image()
        for engine in ("digital_sensor", "analog_tape", "broadcast_signal"):
            with self.subTest(engine=engine):
                result, _grain_map, report = MODULE.IAMCCS_CinePostEfxV2().apply(
                    image,
                    **_settings(engine=engine, strength=0.18),
                )
                self.assertFalse(torch.equal(result, image), engine)
                self.assertIn(f"engine={engine}", report)

    def test_plate_engine_uses_connected_plate(self):
        image = _image()
        plate = torch.rand((1, 24, 32, 3), generator=torch.Generator().manual_seed(9))
        result, _grain_map, report = MODULE.IAMCCS_CinePostEfxV2().apply(
            image,
            grain_plate=plate,
            **_settings(engine="scanned_grain_plate", strength=0.18),
        )
        self.assertFalse(torch.equal(result, image))
        self.assertIn("engine=scanned_grain_plate", report)

    def test_alpha_channel_is_preserved(self):
        image = _image()
        alpha = torch.linspace(0.0, 1.0, image.shape[1] * image.shape[2]).view(1, image.shape[1], image.shape[2], 1)
        rgba = torch.cat((image, alpha), dim=-1)
        result, _grain_map, _report = MODULE.IAMCCS_CinePostEfxV2().apply(
            rgba,
            **_settings(preset="worn_video_copy"),
        )
        self.assertTrue(torch.equal(result[..., 3:4], alpha))

    def test_film_preset_clears_stale_analogue_controls_in_backend(self):
        image = _image()
        _result, _grain_map, report = MODULE.IAMCCS_CinePostEfxV2().apply(
            image,
            **_settings(preset="65mm_clean_scan", horizontal_instability=1.0, scanline_strength=1.0),
        )
        self.assertIn("analog_effects=no", report)

    def test_new_name_and_legacy_workflow_alias_resolve_to_same_node(self):
        self.assertIs(MODULE.NODE_CLASS_MAPPINGS["IAMCCS-CinePostEfx-v2"], MODULE.IAMCCS_CinePostEfxV2)
        underscore_legacy = MODULE.NODE_CLASS_MAPPINGS["IAMCCS_CinePostEfxV2"]
        self.assertTrue(issubclass(underscore_legacy, MODULE.IAMCCS_CinePostEfxV2))
        self.assertTrue(underscore_legacy.DEPRECATED)
        legacy = MODULE.NODE_CLASS_MAPPINGS["IAMCCS_CineFilmGrainPro"]
        self.assertTrue(issubclass(legacy, MODULE.IAMCCS_CinePostEfxV2))
        self.assertTrue(legacy.DEPRECATED)
        self.assertEqual(MODULE.NODE_DISPLAY_NAME_MAPPINGS["IAMCCS-CinePostEfx-v2"], "IAMCCS-CinePostEfx-v2")


if __name__ == "__main__":
    unittest.main()
