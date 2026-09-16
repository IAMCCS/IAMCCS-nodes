"""Resolution-aware, temporally evolving film-grain finishing for IMAGE batches.

The implementation follows published film-grain rendering principles rather than
overlaying a repeated texture: grain is stochastic per frame, processed in linear
light, tone dependent, resolution aware, and gently correlated over time.
"""

from __future__ import annotations

import json
import math

import torch
import torch.nn.functional as F


PRESETS = (
    "65mm_4k_scan_subtle",
    "35mm_vision3_fine",
    "35mm_500t_texture",
    "16mm_fine_documentary",
    "custom_box_values",
)
BLEND_METHODS = ("density_exposure", "linear_additive", "soft_light_luma")

PRO_ENGINES = ("film_emulsion", "digital_sensor", "scanned_grain_plate", "analog_tape", "broadcast_signal")
GRAIN_PROFILES = ("modern_fine", "negative_stock", "high_speed_negative", "reversal_stock", "print_stock")
ANALOG_COLOR_LOOKS = (
    "neutral", "consumer_color", "warm_camcorder", "cool_camcorder",
    "tungsten_home_video", "fluorescent_cctv", "sodium_vapor_cctv",
    "late_night_blue", "sun_bleached_tape", "rf_cyan_fade", "archival_amber",
    "faded_magenta", "crt_broadcast", "night_vision_green",
)
LENS_PRESETS = (
    "lens_none", "wide_angle_14mm", "ultra_wide_10mm", "fisheye_8mm",
    "anamorphic_1_33x", "anamorphic_2x_cinema", "tilt_shift_architecture",
    "tilt_shift_miniature", "vintage_28mm_barrel", "telephoto_pincushion",
    "custom_lens",
)
LENS_PRESET_VALUES = {
    "lens_none": {
        "lens_master": 0.0, "lens_distortion": 0.0, "lens_edge_stretch": 0.0,
        "lens_anamorphic_width": 0.0, "lens_zoom": 1.0, "lens_keystone_x": 0.0,
        "lens_keystone_y": 0.0, "lens_tilt_angle": 0.0, "lens_focus_position": 0.0,
        "lens_tilt_blur": 0.0, "lens_chromatic_aberration": 0.0, "lens_vignette": 0.0,
    },
    "wide_angle_14mm": {
        "lens_master": 1.0, "lens_distortion": 0.30, "lens_edge_stretch": 0.20,
        "lens_zoom": 1.24, "lens_chromatic_aberration": 0.07, "lens_vignette": 0.13,
    },
    "ultra_wide_10mm": {
        "lens_master": 1.0, "lens_distortion": 0.52, "lens_edge_stretch": 0.34,
        "lens_zoom": 1.42, "lens_chromatic_aberration": 0.11, "lens_vignette": 0.20,
    },
    "fisheye_8mm": {
        "lens_master": 1.0, "lens_distortion": 0.92, "lens_edge_stretch": 0.55,
        "lens_zoom": 1.60, "lens_chromatic_aberration": 0.16, "lens_vignette": 0.30,
    },
    "anamorphic_1_33x": {
        "lens_master": 1.0, "lens_distortion": 0.10, "lens_anamorphic_width": 0.10,
        "lens_edge_stretch": 0.22, "lens_zoom": 1.12,
        "lens_chromatic_aberration": 0.08, "lens_vignette": 0.14,
    },
    "anamorphic_2x_cinema": {
        "lens_master": 1.0, "lens_distortion": 0.17, "lens_anamorphic_width": 0.22,
        "lens_edge_stretch": 0.38, "lens_zoom": 1.22,
        "lens_chromatic_aberration": 0.14, "lens_vignette": 0.23,
    },
    "tilt_shift_architecture": {
        "lens_master": 1.0, "lens_zoom": 1.26, "lens_keystone_x": 0.0,
        "lens_keystone_y": -0.24, "lens_tilt_angle": 0.0, "lens_tilt_blur": 0.0,
    },
    "tilt_shift_miniature": {
        "lens_master": 1.0, "lens_zoom": 1.14, "lens_keystone_y": -0.06,
        "lens_tilt_angle": -8.0, "lens_focus_position": -0.05, "lens_tilt_blur": 0.72,
        "lens_vignette": 0.14,
    },
    "vintage_28mm_barrel": {
        "lens_master": 1.0, "lens_distortion": 0.25, "lens_edge_stretch": 0.14,
        "lens_zoom": 1.20, "lens_chromatic_aberration": 0.24, "lens_vignette": 0.38,
    },
    "telephoto_pincushion": {
        "lens_master": 1.0, "lens_distortion": -0.20, "lens_edge_stretch": -0.08,
        "lens_zoom": 1.03, "lens_chromatic_aberration": 0.04, "lens_vignette": 0.08,
    },
}
GRAIN_GAIN = 0.48
GRAIN_MAP_DIVISOR = 4.0
PRO_PRESET_VALUES = {
    "65mm_clean_scan": {
        "engine": "film_emulsion", "blend_method": "log_density", "strength": 0.070,
        "grain_size_4k_px": 0.70, "softness": 0.30, "roughness": 0.12, "complexity": 2,
        "temporal_correlation": 0.0, "chroma_amount": 0.025,
        "shadow_response": 0.50, "midtone_response": 0.78, "highlight_response": 0.28,
        "red_response": 1.00, "green_response": 0.98, "blue_response": 1.03,
    },
    "35mm_fine_negative": {
        "engine": "film_emulsion", "blend_method": "log_density", "strength": 0.115,
        "grain_size_4k_px": 1.00, "softness": 0.18, "roughness": 0.24, "complexity": 3,
        "temporal_correlation": 0.0, "chroma_amount": 0.055,
        "shadow_response": 0.62, "midtone_response": 1.00, "highlight_response": 0.38,
        "red_response": 1.00, "green_response": 0.96, "blue_response": 1.06,
    },
    "35mm_high_speed": {
        "engine": "film_emulsion", "blend_method": "log_density", "strength": 0.185,
        "grain_size_4k_px": 1.35, "softness": 0.12, "roughness": 0.46, "complexity": 4,
        "temporal_correlation": 0.0, "chroma_amount": 0.085,
        "shadow_response": 0.90, "midtone_response": 1.20, "highlight_response": 0.42,
        "red_response": 1.02, "green_response": 0.95, "blue_response": 1.10,
    },
    "16mm_documentary": {
        "engine": "film_emulsion", "blend_method": "log_density", "strength": 0.255,
        "grain_size_4k_px": 1.90, "softness": 0.08, "roughness": 0.62, "complexity": 4,
        "temporal_correlation": 0.0, "chroma_amount": 0.11,
        "shadow_response": 1.02, "midtone_response": 1.30, "highlight_response": 0.48,
        "red_response": 1.04, "green_response": 0.94, "blue_response": 1.13,
    },
    "8mm_expression": {
        "engine": "film_emulsion", "blend_method": "log_density", "strength": 0.36,
        "grain_size_4k_px": 3.10, "softness": 0.04, "roughness": 0.82, "complexity": 4,
        "temporal_correlation": 0.0, "chroma_amount": 0.16,
        "shadow_response": 1.12, "midtone_response": 1.42, "highlight_response": 0.58,
        "red_response": 1.06, "green_response": 0.92, "blue_response": 1.18,
    },
    "digital_cinema_sensor": {
        "engine": "digital_sensor", "blend_method": "linear_additive", "strength": 0.12,
        "grain_size_4k_px": 0.62, "softness": 0.05, "roughness": 0.18, "complexity": 2,
        "temporal_correlation": 0.18, "chroma_amount": 0.12,
        "shadow_response": 1.22, "midtone_response": 0.55, "highlight_response": 0.18,
        "red_response": 1.00, "green_response": 0.92, "blue_response": 1.16,
    },
    "soft_cassette_memory": {
        "engine": "analog_tape", "blend_method": "linear_additive", "strength": 0.035,
        "grain_size_4k_px": 1.25, "softness": 0.42, "roughness": 0.18, "complexity": 2,
        "temporal_correlation": 0.28, "chroma_amount": 0.10,
        "horizontal_instability": 0.12, "chroma_spill": 0.28, "luma_noise": 0.10,
        "chroma_noise": 0.08, "color_drift": 0.10, "edge_echo": 0.08,
        "line_dropout": 0.02, "scanline_strength": 0.08, "field_interlace": 0.04,
        "vertical_roll": 0.0, "highlight_glow": 0.16, "head_switch_distortion": 0.03,
        "luma_trail": 0.10, "chroma_delay": 0.18, "analog_mix": 0.88,
        "tracking_error": 0.03, "tape_warp": 0.12, "ghost_echo": 0.02,
        "signal_saturation": 0.96, "black_lift": 0.07, "chroma_phase_noise": 0.02,
        "chroma_loss": 0.0,
    },
    "worn_video_copy": {
        "engine": "analog_tape", "blend_method": "linear_additive", "strength": 0.08,
        "grain_size_4k_px": 1.65, "softness": 0.28, "roughness": 0.46, "complexity": 3,
        "temporal_correlation": 0.42, "chroma_amount": 0.18,
        "horizontal_instability": 0.52, "chroma_spill": 0.58, "luma_noise": 0.42,
        "chroma_noise": 0.38, "color_drift": 0.34, "edge_echo": 0.36,
        "line_dropout": 0.34, "scanline_strength": 0.28, "field_interlace": 0.30,
        "vertical_roll": 0.14, "highlight_glow": 0.22, "head_switch_distortion": 0.52,
        "luma_trail": 0.36, "chroma_delay": 0.52, "analog_mix": 0.92,
        "tracking_error": 0.58, "tape_warp": 0.62, "ghost_echo": 0.28,
        "signal_saturation": 0.74, "black_lift": 0.34, "chroma_phase_noise": 0.42,
        "chroma_loss": 0.28,
    },
    "midnight_airwave": {
        "engine": "broadcast_signal", "blend_method": "linear_additive", "strength": 0.055,
        "grain_size_4k_px": 0.80, "softness": 0.14, "roughness": 0.28, "complexity": 2,
        "temporal_correlation": 0.12, "chroma_amount": 0.20,
        "horizontal_instability": 0.22, "chroma_spill": 0.46, "luma_noise": 0.32,
        "chroma_noise": 0.48, "color_drift": 0.24, "edge_echo": 0.62,
        "line_dropout": 0.12, "scanline_strength": 0.40, "field_interlace": 0.54,
        "vertical_roll": 0.06, "highlight_glow": 0.12, "head_switch_distortion": 0.0,
        "luma_trail": 0.18, "chroma_delay": 0.40, "analog_mix": 0.88,
        "tracking_error": 0.18, "tape_warp": 0.04, "ghost_echo": 0.62,
        "signal_saturation": 0.82, "black_lift": 0.22, "chroma_phase_noise": 0.52,
        "chroma_loss": 0.18,
    },
    "projector_to_tape": {
        "engine": "broadcast_signal", "blend_method": "log_density", "strength": 0.12,
        "grain_size_4k_px": 1.70, "softness": 0.16, "roughness": 0.48, "complexity": 4,
        "temporal_correlation": 0.08, "chroma_amount": 0.12,
        "horizontal_instability": 0.16, "chroma_spill": 0.24, "luma_noise": 0.16,
        "chroma_noise": 0.10, "color_drift": 0.18, "edge_echo": 0.30,
        "line_dropout": 0.08, "scanline_strength": 0.18, "field_interlace": 0.12,
        "vertical_roll": 0.02, "highlight_glow": 0.52, "head_switch_distortion": 0.06,
        "luma_trail": 0.22, "chroma_delay": 0.24, "analog_mix": 0.76,
        "tracking_error": 0.08, "tape_warp": 0.08, "ghost_echo": 0.20,
        "signal_saturation": 0.84, "black_lift": 0.18, "chroma_phase_noise": 0.10,
        "chroma_loss": 0.04,
    },
    "pristine_tape_master": {
        "engine": "analog_tape", "blend_method": "linear_additive", "strength": 0.025,
        "grain_size_4k_px": 1.10, "softness": 0.46, "roughness": 0.10, "complexity": 2,
        "temporal_correlation": 0.20, "chroma_amount": 0.06,
        "horizontal_instability": 0.02, "chroma_spill": 0.08, "luma_noise": 0.025,
        "chroma_noise": 0.02, "color_drift": 0.0, "edge_echo": 0.08,
        "line_dropout": 0.0, "scanline_strength": 0.025, "field_interlace": 0.02,
        "vertical_roll": 0.0, "highlight_glow": 0.025, "head_switch_distortion": 0.0,
        "luma_trail": 0.03, "chroma_delay": 0.03, "analog_mix": 1.0,
        "tracking_error": 0.0, "tape_warp": 0.025, "ghost_echo": 0.015,
        "signal_saturation": 0.96, "black_lift": 0.06, "chroma_phase_noise": 0.01,
        "chroma_loss": 0.0,
    },
    "family_camcorder_1988": {
        "engine": "analog_tape", "blend_method": "linear_additive", "strength": 0.075,
        "grain_size_4k_px": 1.45, "softness": 0.30, "roughness": 0.42, "complexity": 3,
        "temporal_correlation": 0.36, "chroma_amount": 0.16,
        "horizontal_instability": 0.34, "chroma_spill": 0.52, "luma_noise": 0.30,
        "chroma_noise": 0.28, "color_drift": 0.25, "edge_echo": 0.30,
        "line_dropout": 0.16, "scanline_strength": 0.15, "field_interlace": 0.12,
        "vertical_roll": 0.05, "highlight_glow": 0.42, "head_switch_distortion": 0.38,
        "luma_trail": 0.50, "chroma_delay": 0.44, "analog_mix": 1.0,
        "tracking_error": 0.32, "tape_warp": 0.38, "ghost_echo": 0.10,
        "signal_saturation": 1.15, "black_lift": 0.18, "chroma_phase_noise": 0.22,
        "chroma_loss": 0.10,
    },
    "overplayed_rental_tape": {
        "engine": "analog_tape", "blend_method": "linear_additive", "strength": 0.11,
        "grain_size_4k_px": 1.80, "softness": 0.18, "roughness": 0.68, "complexity": 4,
        "temporal_correlation": 0.48, "chroma_amount": 0.24,
        "horizontal_instability": 0.72, "chroma_spill": 0.78, "luma_noise": 0.68,
        "chroma_noise": 0.60, "color_drift": 0.52, "edge_echo": 0.54,
        "line_dropout": 0.58, "scanline_strength": 0.36, "field_interlace": 0.48,
        "vertical_roll": 0.20, "highlight_glow": 0.18, "head_switch_distortion": 0.82,
        "luma_trail": 0.68, "chroma_delay": 0.74, "analog_mix": 1.0,
        "tracking_error": 0.82, "tape_warp": 0.86, "ghost_echo": 0.44,
        "signal_saturation": 0.70, "black_lift": 0.40, "chroma_phase_noise": 0.68,
        "chroma_loss": 0.58,
    },
    "sun_faded_cassette": {
        "engine": "analog_tape", "blend_method": "linear_additive", "strength": 0.06,
        "grain_size_4k_px": 1.50, "softness": 0.34, "roughness": 0.38, "complexity": 3,
        "temporal_correlation": 0.30, "chroma_amount": 0.12,
        "horizontal_instability": 0.22, "chroma_spill": 0.62, "luma_noise": 0.22,
        "chroma_noise": 0.32, "color_drift": 0.32, "edge_echo": 0.24,
        "line_dropout": 0.08, "scanline_strength": 0.22, "field_interlace": 0.14,
        "vertical_roll": 0.04, "highlight_glow": 0.56, "head_switch_distortion": 0.12,
        "luma_trail": 0.42, "chroma_delay": 0.56, "analog_mix": 1.0,
        "tracking_error": 0.16, "tape_warp": 0.24, "ghost_echo": 0.14,
        "signal_saturation": 0.42, "black_lift": 0.54, "chroma_phase_noise": 0.30,
        "chroma_loss": 0.22,
    },
    "late_night_relay": {
        "engine": "broadcast_signal", "blend_method": "linear_additive", "strength": 0.045,
        "grain_size_4k_px": 0.85, "softness": 0.12, "roughness": 0.28, "complexity": 2,
        "temporal_correlation": 0.10, "chroma_amount": 0.22,
        "horizontal_instability": 0.12, "chroma_spill": 0.56, "luma_noise": 0.42,
        "chroma_noise": 0.62, "color_drift": 0.28, "edge_echo": 0.82,
        "line_dropout": 0.10, "scanline_strength": 0.46, "field_interlace": 0.42,
        "vertical_roll": 0.04, "highlight_glow": 0.34, "head_switch_distortion": 0.0,
        "luma_trail": 0.20, "chroma_delay": 0.58, "analog_mix": 1.0,
        "tracking_error": 0.12, "tape_warp": 0.02, "ghost_echo": 0.72,
        "signal_saturation": 0.78, "black_lift": 0.24, "chroma_phase_noise": 0.72,
        "chroma_loss": 0.30,
    },
    "damaged_airwave": {
        "engine": "broadcast_signal", "blend_method": "linear_additive", "strength": 0.075,
        "grain_size_4k_px": 0.75, "softness": 0.06, "roughness": 0.52, "complexity": 3,
        "temporal_correlation": 0.08, "chroma_amount": 0.30,
        "horizontal_instability": 0.38, "chroma_spill": 0.70, "luma_noise": 0.78,
        "chroma_noise": 0.86, "color_drift": 0.54, "edge_echo": 0.96,
        "line_dropout": 0.42, "scanline_strength": 0.62, "field_interlace": 0.72,
        "vertical_roll": 0.18, "highlight_glow": 0.26, "head_switch_distortion": 0.0,
        "luma_trail": 0.34, "chroma_delay": 0.82, "analog_mix": 1.0,
        "tracking_error": 0.46, "tape_warp": 0.04, "ghost_echo": 0.88,
        "signal_saturation": 0.66, "black_lift": 0.32, "chroma_phase_noise": 0.94,
        "chroma_loss": 0.72,
    },
    "archival_film_to_video": {
        "engine": "broadcast_signal", "blend_method": "log_density", "strength": 0.15,
        "grain_size_4k_px": 1.85, "softness": 0.16, "roughness": 0.52, "complexity": 4,
        "temporal_correlation": 0.02, "chroma_amount": 0.10,
        "horizontal_instability": 0.06, "chroma_spill": 0.28, "luma_noise": 0.18,
        "chroma_noise": 0.12, "color_drift": 0.16, "edge_echo": 0.56,
        "line_dropout": 0.04, "scanline_strength": 0.14, "field_interlace": 0.08,
        "vertical_roll": 0.01, "highlight_glow": 0.62, "head_switch_distortion": 0.0,
        "luma_trail": 0.18, "chroma_delay": 0.26, "analog_mix": 0.86,
        "tracking_error": 0.04, "tape_warp": 0.02, "ghost_echo": 0.24,
        "signal_saturation": 0.82, "black_lift": 0.20, "chroma_phase_noise": 0.14,
        "chroma_loss": 0.06,
    },
}

_ANALOGUE_PRESET_BASE = {
    "engine": "broadcast_signal", "blend_method": "linear_additive", "strength": 0.05,
    "grain_size_4k_px": 0.90, "softness": 0.16, "roughness": 0.36, "complexity": 3,
    "temporal_correlation": 0.12, "chroma_amount": 0.16,
    "shadow_response": 0.90, "midtone_response": 0.90, "highlight_response": 0.32,
    "red_response": 1.0, "green_response": 1.0, "blue_response": 1.0,
    "horizontal_instability": 0.10, "chroma_spill": 0.30, "luma_noise": 0.20,
    "chroma_noise": 0.18, "color_drift": 0.10, "edge_echo": 0.30,
    "line_dropout": 0.05, "scanline_strength": 0.22, "field_interlace": 0.25,
    "vertical_roll": 0.02, "highlight_glow": 0.10, "head_switch_distortion": 0.0,
    "luma_trail": 0.12, "chroma_delay": 0.22, "analog_mix": 1.0,
    "tracking_error": 0.08, "tape_warp": 0.02, "ghost_echo": 0.16,
    "signal_saturation": 0.90, "black_lift": 0.12, "chroma_phase_noise": 0.18,
    "chroma_loss": 0.08,
}


def _analogue_preset(**changes):
    return {**_ANALOGUE_PRESET_BASE, **changes}


PRO_PRESET_VALUES.update({
    "cctv_monochrome_1997": _analogue_preset(
        strength=0.035, chroma_amount=0.01, horizontal_instability=0.05, chroma_spill=0.08,
        luma_noise=0.52, chroma_noise=0.0, color_drift=0.0, edge_echo=0.48,
        line_dropout=0.12, scanline_strength=0.58, field_interlace=0.70,
        luma_trail=0.18, chroma_delay=0.0, tracking_error=0.12, ghost_echo=0.08,
        signal_saturation=0.0, black_lift=0.30, chroma_phase_noise=0.0, chroma_loss=0.0,
    ),
    "parking_garage_cctv": _analogue_preset(
        strength=0.055, roughness=0.58, chroma_amount=0.03, horizontal_instability=0.12,
        chroma_spill=0.16, luma_noise=0.76, chroma_noise=0.05, edge_echo=0.56,
        line_dropout=0.28, scanline_strength=0.64, field_interlace=0.78,
        vertical_roll=0.05, luma_trail=0.28, tracking_error=0.24, ghost_echo=0.12,
        signal_saturation=0.08, black_lift=0.46, chroma_phase_noise=0.04, chroma_loss=0.06,
    ),
    "camcorder_night_recording": _analogue_preset(
        engine="analog_tape", strength=0.09, grain_size_4k_px=1.55, softness=0.24,
        roughness=0.66, temporal_correlation=0.42, chroma_amount=0.14,
        horizontal_instability=0.34, chroma_spill=0.58, luma_noise=0.82, chroma_noise=0.46,
        color_drift=0.28, edge_echo=0.28, line_dropout=0.18, scanline_strength=0.22,
        field_interlace=0.20, highlight_glow=0.58, head_switch_distortion=0.32,
        luma_trail=0.58, chroma_delay=0.52, tracking_error=0.36, tape_warp=0.44,
        ghost_echo=0.14, signal_saturation=0.38, black_lift=0.48,
        chroma_phase_noise=0.38, chroma_loss=0.26,
    ),
    "vhs_pause_damage": _analogue_preset(
        engine="analog_tape", strength=0.07, grain_size_4k_px=1.45, roughness=0.54,
        temporal_correlation=0.70, chroma_amount=0.22, horizontal_instability=0.62,
        chroma_spill=0.72, luma_noise=0.54, chroma_noise=0.58, color_drift=0.34,
        edge_echo=0.48, line_dropout=0.76, scanline_strength=0.46, field_interlace=0.64,
        vertical_roll=0.08, head_switch_distortion=1.0, luma_trail=0.46,
        chroma_delay=0.78, tracking_error=1.0, tape_warp=0.58, ghost_echo=0.30,
        signal_saturation=0.62, black_lift=0.34, chroma_phase_noise=0.72, chroma_loss=0.66,
    ),
    "school_av_vhs": _analogue_preset(
        engine="analog_tape", strength=0.045, grain_size_4k_px=1.30, softness=0.38,
        roughness=0.30, temporal_correlation=0.32, chroma_amount=0.10,
        horizontal_instability=0.18, chroma_spill=0.46, luma_noise=0.22, chroma_noise=0.18,
        color_drift=0.16, edge_echo=0.22, line_dropout=0.05, scanline_strength=0.14,
        field_interlace=0.12, highlight_glow=0.28, head_switch_distortion=0.08,
        luma_trail=0.34, chroma_delay=0.38, tracking_error=0.14, tape_warp=0.20,
        ghost_echo=0.10, signal_saturation=0.68, black_lift=0.28,
        chroma_phase_noise=0.16, chroma_loss=0.10,
    ),
    "public_access_studio": _analogue_preset(
        strength=0.035, softness=0.22, roughness=0.24, horizontal_instability=0.08,
        chroma_spill=0.40, luma_noise=0.22, chroma_noise=0.32, color_drift=0.20,
        edge_echo=0.68, line_dropout=0.04, scanline_strength=0.34, field_interlace=0.36,
        highlight_glow=0.30, luma_trail=0.16, chroma_delay=0.46, tracking_error=0.08,
        ghost_echo=0.52, signal_saturation=1.12, black_lift=0.18,
        chroma_phase_noise=0.38, chroma_loss=0.12,
    ),
    "cable_channel_1993": _analogue_preset(
        strength=0.055, roughness=0.42, chroma_amount=0.24, horizontal_instability=0.20,
        chroma_spill=0.62, luma_noise=0.48, chroma_noise=0.68, color_drift=0.42,
        edge_echo=0.76, line_dropout=0.18, scanline_strength=0.50, field_interlace=0.56,
        vertical_roll=0.10, highlight_glow=0.20, luma_trail=0.24, chroma_delay=0.70,
        tracking_error=0.26, ghost_echo=0.66, signal_saturation=0.84, black_lift=0.26,
        chroma_phase_noise=0.78, chroma_loss=0.42,
    ),
    "weak_rooftop_antenna": _analogue_preset(
        strength=0.065, softness=0.05, roughness=0.62, chroma_amount=0.32,
        horizontal_instability=0.48, chroma_spill=0.68, luma_noise=0.92, chroma_noise=1.0,
        color_drift=0.64, edge_echo=0.88, line_dropout=0.62, scanline_strength=0.58,
        field_interlace=0.66, vertical_roll=0.24, luma_trail=0.30, chroma_delay=0.88,
        tracking_error=0.58, ghost_echo=0.78, signal_saturation=0.58, black_lift=0.36,
        chroma_phase_noise=1.0, chroma_loss=0.86,
    ),
    "clean_local_news_archive": _analogue_preset(
        strength=0.025, softness=0.32, roughness=0.16, chroma_amount=0.08,
        horizontal_instability=0.03, chroma_spill=0.22, luma_noise=0.10, chroma_noise=0.10,
        color_drift=0.06, edge_echo=0.44, line_dropout=0.01, scanline_strength=0.16,
        field_interlace=0.22, highlight_glow=0.18, luma_trail=0.08, chroma_delay=0.20,
        tracking_error=0.02, ghost_echo=0.22, signal_saturation=0.94, black_lift=0.12,
        chroma_phase_noise=0.12, chroma_loss=0.02,
    ),
    "industrial_monitor_feed": _analogue_preset(
        strength=0.045, chroma_amount=0.05, horizontal_instability=0.16,
        chroma_spill=0.18, luma_noise=0.58, chroma_noise=0.08, color_drift=0.08,
        edge_echo=0.62, line_dropout=0.22, scanline_strength=0.72, field_interlace=0.82,
        vertical_roll=0.12, highlight_glow=0.06, luma_trail=0.22, chroma_delay=0.08,
        tracking_error=0.32, ghost_echo=0.18, signal_saturation=0.18, black_lift=0.38,
        chroma_phase_noise=0.10, chroma_loss=0.16,
    ),
})

PRO_PRESET_VALUES.update({
    "vhs_consumer_color_pop": _analogue_preset(
        engine="analog_tape", strength=0.055, grain_size_4k_px=1.30, softness=0.34,
        roughness=0.34, temporal_correlation=0.34, chroma_amount=0.12,
        horizontal_instability=0.20, chroma_spill=0.42, luma_noise=0.22, chroma_noise=0.20,
        color_drift=0.14, edge_echo=0.24, line_dropout=0.06, scanline_strength=0.13,
        field_interlace=0.12, highlight_glow=0.34, head_switch_distortion=0.10,
        luma_trail=0.32, chroma_delay=0.40, tracking_error=0.16, tape_warp=0.22,
        ghost_echo=0.10, signal_saturation=1.18, black_lift=0.18,
        chroma_phase_noise=0.18, chroma_loss=0.08, analog_color_look="consumer_color",
    ),
    "vhs_warm_family_tape": _analogue_preset(
        engine="analog_tape", strength=0.065, grain_size_4k_px=1.45, softness=0.30,
        roughness=0.42, temporal_correlation=0.38, chroma_amount=0.14,
        horizontal_instability=0.30, chroma_spill=0.54, luma_noise=0.30, chroma_noise=0.26,
        color_drift=0.24, edge_echo=0.28, line_dropout=0.12, scanline_strength=0.16,
        field_interlace=0.15, highlight_glow=0.48, head_switch_distortion=0.30,
        luma_trail=0.48, chroma_delay=0.50, tracking_error=0.28, tape_warp=0.34,
        ghost_echo=0.12, signal_saturation=1.08, black_lift=0.22,
        chroma_phase_noise=0.28, chroma_loss=0.14, analog_color_look="warm_camcorder",
    ),
    "vhs_cool_camcorder": _analogue_preset(
        engine="analog_tape", strength=0.06, grain_size_4k_px=1.35, softness=0.28,
        roughness=0.38, temporal_correlation=0.30, chroma_amount=0.16,
        horizontal_instability=0.26, chroma_spill=0.48, luma_noise=0.34, chroma_noise=0.34,
        color_drift=0.20, edge_echo=0.30, line_dropout=0.10, scanline_strength=0.18,
        field_interlace=0.20, highlight_glow=0.30, head_switch_distortion=0.22,
        luma_trail=0.38, chroma_delay=0.46, tracking_error=0.24, tape_warp=0.28,
        ghost_echo=0.15, signal_saturation=1.02, black_lift=0.20,
        chroma_phase_noise=0.34, chroma_loss=0.16, analog_color_look="cool_camcorder",
    ),
    "vhs_magenta_generation_loss": _analogue_preset(
        engine="analog_tape", strength=0.09, grain_size_4k_px=1.65, softness=0.20,
        roughness=0.58, temporal_correlation=0.48, chroma_amount=0.24,
        horizontal_instability=0.58, chroma_spill=0.76, luma_noise=0.58, chroma_noise=0.66,
        color_drift=0.52, edge_echo=0.44, line_dropout=0.42, scanline_strength=0.32,
        field_interlace=0.40, vertical_roll=0.12, highlight_glow=0.22,
        head_switch_distortion=0.68, luma_trail=0.64, chroma_delay=0.78,
        tracking_error=0.66, tape_warp=0.70, ghost_echo=0.38, signal_saturation=0.76,
        black_lift=0.38, chroma_phase_noise=0.72, chroma_loss=0.52,
        analog_color_look="faded_magenta",
    ),
    "crt_saturated_broadcast": _analogue_preset(
        strength=0.04, softness=0.20, roughness=0.28, chroma_amount=0.20,
        horizontal_instability=0.10, chroma_spill=0.48, luma_noise=0.26, chroma_noise=0.36,
        color_drift=0.18, edge_echo=0.72, line_dropout=0.05, scanline_strength=0.44,
        field_interlace=0.46, highlight_glow=0.38, luma_trail=0.14, chroma_delay=0.50,
        tracking_error=0.08, ghost_echo=0.48, signal_saturation=1.14, black_lift=0.18,
        chroma_phase_noise=0.42, chroma_loss=0.12, analog_color_look="crt_broadcast",
    ),
    "night_vision_cctv_green": _analogue_preset(
        strength=0.055, chroma_amount=0.0, horizontal_instability=0.10, chroma_spill=0.0,
        luma_noise=0.78, chroma_noise=0.0, color_drift=0.0, edge_echo=0.54,
        line_dropout=0.24, scanline_strength=0.66, field_interlace=0.74,
        vertical_roll=0.06, highlight_glow=0.16, luma_trail=0.26, chroma_delay=0.0,
        tracking_error=0.24, ghost_echo=0.12, signal_saturation=1.0, black_lift=0.32,
        chroma_phase_noise=0.0, chroma_loss=0.0, analog_color_look="night_vision_green",
    ),
})

# Every analogue preset carries a colour signature as well as a signal/noise
# signature.  Older presets pre-dated the colour engine and otherwise silently
# fell back to neutral, which made cameras and tape generations look alike.
_ANALOG_PRESET_COLOR_DEFAULTS = {
    "soft_cassette_memory": "warm_camcorder",
    "worn_video_copy": "faded_magenta",
    "midnight_airwave": "late_night_blue",
    "projector_to_tape": "archival_amber",
    "pristine_tape_master": "consumer_color",
    "family_camcorder_1988": "tungsten_home_video",
    "overplayed_rental_tape": "faded_magenta",
    "sun_faded_cassette": "sun_bleached_tape",
    "late_night_relay": "late_night_blue",
    "damaged_airwave": "rf_cyan_fade",
    "archival_film_to_video": "archival_amber",
    "cctv_monochrome_1997": "neutral",
    "parking_garage_cctv": "sodium_vapor_cctv",
    "camcorder_night_recording": "cool_camcorder",
    "vhs_pause_damage": "faded_magenta",
    "school_av_vhs": "sun_bleached_tape",
    "public_access_studio": "crt_broadcast",
    "cable_channel_1993": "consumer_color",
    "weak_rooftop_antenna": "rf_cyan_fade",
    "clean_local_news_archive": "crt_broadcast",
    "industrial_monitor_feed": "fluorescent_cctv",
}
for _preset_name, _color_look in _ANALOG_PRESET_COLOR_DEFAULTS.items():
    PRO_PRESET_VALUES[_preset_name]["analog_color_look"] = _color_look

_FILM_PROFILE_DEFAULTS = {
    "65mm_clean_scan": ("modern_fine", 0.10),
    "35mm_fine_negative": ("negative_stock", 0.18),
    "35mm_high_speed": ("high_speed_negative", 0.24),
    "16mm_documentary": ("high_speed_negative", 0.30),
    "8mm_expression": ("reversal_stock", 0.34),
    "digital_cinema_sensor": ("modern_fine", 0.20),
    "archival_film_to_video": ("print_stock", 0.22),
}
for _preset_name, _preset_values in PRO_PRESET_VALUES.items():
    _profile, _microcontrast = _FILM_PROFILE_DEFAULTS.get(_preset_name, ("negative_stock", 0.12))
    _preset_values.setdefault("grain_profile", _profile)
    _preset_values.setdefault("texture_microcontrast", _microcontrast)
    _preset_values.setdefault("analog_color_look", "neutral")
    for _randomness_name, _source_name in (
        ("horizontal_instability_randomness", "horizontal_instability"),
        ("tracking_randomness", "tracking_error"),
        ("tape_warp_randomness", "tape_warp"),
        ("color_drift_randomness", "color_drift"),
        ("scanline_randomness", "scanline_strength"),
        ("interlace_randomness", "field_interlace"),
        ("vertical_roll_randomness", "vertical_roll"),
        ("head_switch_randomness", "head_switch_distortion"),
    ):
        _source_amount = float(_preset_values.get(_source_name, 0.0))
        _preset_values.setdefault(
            _randomness_name,
            min(0.85, 0.10 + 0.70 * _source_amount) if _source_amount > 0.0 else 0.0,
        )

del _ANALOGUE_PRESET_BASE
del _ANALOG_PRESET_COLOR_DEFAULTS
del _FILM_PROFILE_DEFAULTS
PRO_PRESETS = (*PRO_PRESET_VALUES, "custom_box_values")
PRO_BLEND_METHODS = ("log_density", "density_exposure", "linear_additive", "soft_light_luma")
INPUT_TRANSFERS = ("srgb", "rec709", "linear")

ANALOG_FIELDS = (
    "horizontal_instability", "chroma_spill", "luma_noise", "chroma_noise", "color_drift",
    "edge_echo", "line_dropout", "scanline_strength", "field_interlace", "vertical_roll",
    "highlight_glow", "head_switch_distortion", "luma_trail", "chroma_delay",
    "tracking_error", "tape_warp", "ghost_echo", "chroma_phase_noise", "chroma_loss",
)
TEMPORAL_RANDOMNESS_FIELDS = (
    "horizontal_instability_randomness", "tracking_randomness", "tape_warp_randomness",
    "color_drift_randomness", "scanline_randomness", "interlace_randomness",
    "vertical_roll_randomness", "head_switch_randomness",
)


def _temporal_random_value(frame_index: int, seed: int, salt: int, rate: float = 0.37) -> float:
    """Smooth deterministic temporal noise, independent from render order/device."""
    position = max(0.0, float(frame_index)) * rate
    index = int(math.floor(position))
    fraction = position - index
    fraction = fraction * fraction * (3.0 - 2.0 * fraction)

    def hashed(sample: int) -> float:
        value = (sample * 0x9E3779B1 + int(seed) * 0x85EBCA77 + salt * 0xC2B2AE3D) & 0xFFFFFFFF
        value ^= value >> 16
        value = (value * 0x7FEB352D) & 0xFFFFFFFF
        value ^= value >> 15
        value = (value * 0x846CA68B) & 0xFFFFFFFF
        value ^= value >> 16
        return (value / 0xFFFFFFFF) * 2.0 - 1.0

    return hashed(index) * (1.0 - fraction) + hashed(index + 1) * fraction


def _srgb_to_linear(value: torch.Tensor) -> torch.Tensor:
    return torch.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055).pow(2.4))


def _linear_to_srgb(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp_min(0.0)
    return torch.where(value <= 0.0031308, value * 12.92, 1.055 * value.pow(1.0 / 2.4) - 0.055)


def _rec709_to_linear(value: torch.Tensor) -> torch.Tensor:
    return torch.where(value < 0.081, value / 4.5, ((value + 0.099) / 1.099).pow(1.0 / 0.45))


def _linear_to_rec709(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp_min(0.0)
    return torch.where(value < 0.018, value * 4.5, 1.099 * value.pow(0.45) - 0.099)


def _decode_transfer(value: torch.Tensor, transfer: str) -> torch.Tensor:
    if transfer == "linear":
        return value
    if transfer == "rec709":
        return _rec709_to_linear(value)
    return _srgb_to_linear(value)


def _encode_transfer(value: torch.Tensor, transfer: str) -> torch.Tensor:
    if transfer == "linear":
        return value
    if transfer == "rec709":
        return _linear_to_rec709(value)
    return _linear_to_srgb(value)


def _normalized_field(
    height: int,
    width: int,
    pixel_size: float,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    scale = max(1.0, float(pixel_size))
    source_h = max(2, int(math.ceil(height / scale)))
    source_w = max(2, int(math.ceil(width / scale)))
    field = torch.randn((1, 1, source_h, source_w), generator=generator, device=device, dtype=torch.float32)
    if source_h != height or source_w != width:
        field = F.interpolate(field, size=(height, width), mode="bicubic", align_corners=False)
    # A weak second octave avoids electronic white-noise texture while keeping
    # the result fine enough for a 4K finishing pass.
    clump_h = max(2, int(math.ceil(height / max(1.0, scale * 2.75))))
    clump_w = max(2, int(math.ceil(width / max(1.0, scale * 2.75))))
    clump = torch.randn((1, 1, clump_h, clump_w), generator=generator, device=device, dtype=torch.float32)
    clump = F.interpolate(clump, size=(height, width), mode="bicubic", align_corners=False)
    field = field * 0.86 + clump * 0.14
    return (field - field.mean()) / field.std(unbiased=False).clamp_min(1e-6)


def _soft_light(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    low = base - (1.0 - 2.0 * blend) * base * (1.0 - base)
    d = torch.where(base <= 0.25, ((16.0 * base - 12.0) * base + 4.0) * base, base.sqrt())
    high = base + (2.0 * blend - 1.0) * (d - base)
    return torch.where(blend <= 0.5, low, high)


def _single_scale_field(
    height: int,
    width: int,
    pixel_size: float,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Unit-variance random field at a physical output-pixel scale."""
    return _single_scale_layers(height, width, pixel_size, 1, generator, device)


def _single_scale_layers(
    height: int,
    width: int,
    pixel_size: float,
    channels: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Generate several emulsion layers in one interpolation pass."""
    scale = max(1.0, float(pixel_size))
    source_h = max(2, int(math.ceil(height / scale)))
    source_w = max(2, int(math.ceil(width / scale)))
    field = torch.randn(
        (1, max(1, int(channels)), source_h, source_w),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    if source_h != height or source_w != width:
        field = F.interpolate(field, size=(height, width), mode="bicubic", align_corners=False)
    field = field - field.mean(dim=(-2, -1), keepdim=True)
    return field / field.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-6)


_GRAIN_PROFILE_SHAPE = {
    # tail, dye-cloud mixing, local AR correlation, output gain
    "modern_fine": (0.82, 0.045, 0.055, 0.88),
    "negative_stock": (1.00, 0.10, 0.085, 1.00),
    "high_speed_negative": (1.22, 0.18, 0.13, 1.08),
    "reversal_stock": (1.32, 0.15, 0.10, 1.04),
    "print_stock": (0.92, 0.08, 0.15, 0.86),
}


def _pro_emulsion_layers(
    height: int,
    width: int,
    pixel_size: float,
    softness: float,
    roughness: float,
    complexity: int,
    channels: int,
    profile: str,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Vectorized density-domain emulsion layers with local spatial correlation."""
    octaves = max(1, min(4, int(complexity)))
    rough = max(0.0, min(1.0, float(roughness)))
    tail, cloud_mix, ar_mix, output_gain = _GRAIN_PROFILE_SHAPE.get(
        str(profile), _GRAIN_PROFILE_SHAPE["negative_stock"]
    )
    weights = [1.0] + [(0.14 + rough * 0.30) / octave for octave in range(1, octaves)]
    field = None
    for octave, weight in enumerate(weights):
        component = _single_scale_layers(
            height,
            width,
            max(1.0, float(pixel_size) * (2.12 ** octave)),
            channels,
            generator,
            device,
        )
        field = component * weight if field is None else field.add_(component, alpha=weight)

    # Local AR-like correlation avoids the electronic appearance of independent
    # pixels. A second, nonlinear cloud component creates occasional dye clumps.
    local = F.avg_pool2d(F.pad(field, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)
    field = torch.lerp(field, local, ar_mix + rough * 0.08)
    cloud = F.avg_pool2d(F.pad(field, (2, 2, 2, 2), mode="replicate"), kernel_size=5, stride=1)
    cloud = torch.sign(cloud) * cloud.abs().pow(1.35 + rough * 0.45)
    field = field + cloud * (cloud_mix + rough * 0.08)

    exponent = 1.0 + rough * 0.30 * tail
    field = torch.sign(field) * field.abs().pow(exponent)
    soft = max(0.0, min(1.0, float(softness)))
    if soft > 0.01:
        kernel = 3 if soft < 0.65 else 5
        radius = kernel // 2
        blurred = F.avg_pool2d(F.pad(field, (radius, radius, radius, radius), mode="replicate"), kernel_size=kernel, stride=1)
        field = torch.lerp(field, blurred, soft * 0.62)
    field = field - field.mean(dim=(-2, -1), keepdim=True)
    field = field / field.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-6)
    return field * output_gain


def _pro_emulsion_field(
    height: int,
    width: int,
    pixel_size: float,
    softness: float,
    roughness: float,
    complexity: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Multiscale stochastic emulsion texture, with no repeated texture tile."""
    return _pro_emulsion_layers(
        height, width, pixel_size, softness, roughness, complexity, 1,
        "negative_stock", generator, device,
    )


def _plate_field(plate_frame: torch.Tensor, height: int, width: int) -> torch.Tensor:
    rgb = plate_frame[..., :3].to(torch.float32).permute(2, 0, 1)[None]
    if rgb.shape[-2:] != (height, width):
        rgb = F.interpolate(rgb, size=(height, width), mode="bicubic", align_corners=False)
    # Remove the plate's photographed exposure/colour and retain its texture.
    low = F.avg_pool2d(rgb, kernel_size=9, stride=1, padding=4)
    field = rgb - low
    field = field - field.mean(dim=(-2, -1), keepdim=True)
    field = field / field.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-6)
    return field


def _shift_hwc(frame: torch.Tensor, shift_x: int = 0, shift_y: int = 0) -> torch.Tensor:
    """Translate an HWC frame with edge replication instead of wraparound."""
    height, width = frame.shape[:2]
    x = torch.arange(width, device=frame.device).sub(int(shift_x)).clamp_(0, width - 1)
    y = torch.arange(height, device=frame.device).sub(int(shift_y)).clamp_(0, height - 1)
    return frame.index_select(0, y).index_select(1, x)


def _blur_hwc(frame: torch.Tensor, kernel_y: int, kernel_x: int) -> torch.Tensor:
    kernel_y = max(1, int(kernel_y) | 1)
    kernel_x = max(1, int(kernel_x) | 1)
    nchw = frame.permute(2, 0, 1).unsqueeze(0)
    padded = F.pad(nchw, (kernel_x // 2, kernel_x // 2, kernel_y // 2, kernel_y // 2), mode="replicate")
    return F.avg_pool2d(padded, kernel_size=(kernel_y, kernel_x), stride=1).squeeze(0).permute(1, 2, 0)


def _gaussian_blur_hwc(frame: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable, resolution-aware Gaussian blur with bounded kernel cost."""
    sigma = max(0.01, float(sigma))
    source = frame.permute(2, 0, 1).unsqueeze(0)
    height, width = frame.shape[:2]
    reduction = max(1, int(math.ceil(sigma / 8.0)))
    if reduction > 1:
        reduced_h = max(2, int(round(height / reduction)))
        reduced_w = max(2, int(round(width / reduction)))
        source = F.interpolate(source, size=(reduced_h, reduced_w), mode="bicubic", align_corners=False, antialias=True)
        sigma /= reduction

    radius = max(1, min(32, int(math.ceil(sigma * 3.0))))
    coordinates = torch.arange(-radius, radius + 1, device=frame.device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (coordinates / sigma).square())
    kernel = (kernel / kernel.sum()).to(source.dtype)
    channels = source.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    source = F.conv2d(F.pad(source, (radius, radius, 0, 0), mode="replicate"), horizontal, groups=channels)
    source = F.conv2d(F.pad(source, (0, 0, radius, radius), mode="replicate"), vertical, groups=channels)
    if reduction > 1:
        source = F.interpolate(source, size=(height, width), mode="bicubic", align_corners=False)
    return source[0].permute(1, 2, 0)


def _analogue_signal_size(height: int, width: int) -> tuple[int, int]:
    """Fit the frame into the same 288x162 signal raster used by the UI monitor."""
    scale = min(1.0, 288.0 / max(1, width), 162.0 / max(1, height))
    return max(2, int(round(height * scale))), max(2, int(round(width * scale)))


def _warp_rows(frame: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    height, width, channels = frame.shape
    columns = torch.arange(width, device=frame.device).view(1, width) - offsets.round().to(torch.long).view(height, 1)
    columns = columns.clamp_(0, width - 1).unsqueeze(-1).expand(height, width, channels)
    return frame.gather(1, columns)


def _lens_sampling_grid(
    height: int,
    width: int,
    settings: dict[str, float | int | str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build an inverse optical warp in normalized image coordinates."""
    amount = max(0.0, min(2.0, float(settings.get("lens_master", 0.0))))
    zoom = max(0.70, min(1.60, float(settings.get("lens_zoom", 1.0))))
    distortion = max(-1.0, min(1.0, float(settings.get("lens_distortion", 0.0)))) * amount
    edge_stretch = max(-1.0, min(1.0, float(settings.get("lens_edge_stretch", 0.0)))) * amount
    anamorphic = max(-0.5, min(0.5, float(settings.get("lens_anamorphic_width", 0.0)))) * amount
    keystone_x = max(-0.75, min(0.75, float(settings.get("lens_keystone_x", 0.0)))) * amount
    keystone_y = max(-0.75, min(0.75, float(settings.get("lens_keystone_y", 0.0)))) * amount

    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    aspect = float(width) / max(1.0, float(height))
    optical_y = yy / aspect
    radius2 = xx.square() + optical_y.square()
    # Brown-Conrady-style radial terms plus an independent peripheral stretch.
    radial = 1.0 + distortion * (0.34 * radius2 + 0.16 * radius2.square())
    radial = radial + edge_stretch * 0.16 * radius2.pow(1.5)
    source_x = xx * radial
    source_y = optical_y * radial
    # Anamorphic glass bends the horizontal field more strongly and changes the
    # apparent width without treating the image as a simple aspect resize.
    source_x = source_x * (1.0 - anamorphic * 0.48)
    source_x = source_x * (1.0 + anamorphic * 0.22 * source_y.square())
    denominator = (1.0 + keystone_x * source_x * 0.55 + keystone_y * source_y * 0.72).clamp_min(0.42)
    source_x = source_x / denominator
    source_y = source_y / denominator
    source_x = source_x / zoom
    source_y = source_y * aspect / zoom
    return torch.stack((source_x, source_y), dim=-1).unsqueeze(0), xx, yy


def _apply_lens_effects(frame: torch.Tensor, settings: dict[str, float | int | str]) -> torch.Tensor:
    """GPU-friendly optical geometry, tilt plane, CA and lens falloff."""
    amount = max(0.0, min(2.0, float(settings.get("lens_master", 0.0))))
    if amount <= 0.0:
        return frame
    height, width = frame.shape[:2]
    grid, xx, yy = _lens_sampling_grid(height, width, settings, frame.device)
    source = frame.permute(2, 0, 1).unsqueeze(0)
    aberration = max(0.0, min(1.0, float(settings.get("lens_chromatic_aberration", 0.0)))) * amount
    if aberration > 0.0:
        radius2 = xx.square() + (yy / (float(width) / max(1.0, float(height)))).square()
        shift = aberration * 0.0065 * radius2
        red_grid = grid.clone()
        blue_grid = grid.clone()
        red_grid[..., 0] = red_grid[..., 0] * (1.0 + shift)
        blue_grid[..., 0] = blue_grid[..., 0] * (1.0 - shift)
        channels = (
            F.grid_sample(source[:, 0:1], red_grid, mode="bicubic", padding_mode="border", align_corners=True),
            F.grid_sample(source[:, 1:2], grid, mode="bicubic", padding_mode="border", align_corners=True),
            F.grid_sample(source[:, 2:3], blue_grid, mode="bicubic", padding_mode="border", align_corners=True),
        )
        warped = torch.cat(channels, dim=1)
    else:
        warped = F.grid_sample(source, grid, mode="bicubic", padding_mode="border", align_corners=True)
    result = warped[0].permute(1, 2, 0)

    tilt_amount = max(0.0, min(2.0, float(settings.get("lens_tilt_blur", 0.0)) * amount))
    if tilt_amount > 0.0:
        angle = math.radians(max(-90.0, min(90.0, float(settings.get("lens_tilt_angle", 0.0)))))
        focus_position = max(-1.0, min(1.0, float(settings.get("lens_focus_position", 0.0))))
        distance = (yy * math.cos(angle) + xx * math.sin(angle) - focus_position).abs()
        focus_strength = min(1.0, tilt_amount)
        focus_band = 0.16 + 0.20 * (1.0 - focus_strength)
        coc = ((distance - focus_band) / max(0.08, 1.0 - focus_band)).clamp(0.0, 1.0)
        coc = coc.square() * (3.0 - 2.0 * coc)
        short_edge = float(min(height, width))
        maximum_sigma = max(0.85, short_edge * (0.004 + 0.020 * focus_strength) * tilt_amount)
        medium = _gaussian_blur_hwc(result, maximum_sigma * 0.38)
        broad = _gaussian_blur_hwc(result, maximum_sigma)
        broad_weight = ((coc - 0.32) / 0.68).clamp(0.0, 1.0)
        broad_weight = broad_weight.square() * (3.0 - 2.0 * broad_weight)
        defocused = torch.lerp(medium, broad, broad_weight[..., None])
        result = torch.lerp(result, defocused, (coc * focus_strength)[..., None])

    vignette = max(0.0, min(1.0, float(settings.get("lens_vignette", 0.0)))) * amount
    if vignette > 0.0:
        aspect = float(width) / max(1.0, float(height))
        radius = torch.sqrt(xx.square() + (yy / aspect).square()).clamp(0.0, 1.25)
        falloff = 1.0 - vignette * 0.62 * radius.pow(2.35)
        result = result * falloff[..., None]
    return result.clamp(0.0, 1.0)


def _smooth_row_profile(values: torch.Tensor, radius: int) -> torch.Tensor:
    """Low-pass a row displacement/noise profile without introducing wraparound."""
    radius = max(1, int(radius))
    kernel = radius * 2 + 1
    signal = values.view(1, 1, -1)
    padded = F.pad(signal, (radius, radius), mode="replicate")
    return F.avg_pool1d(padded, kernel_size=kernel, stride=1).view(-1)


def _rgb_to_signal(frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Separate display RGB into luminance and two signed analogue colour axes."""
    red, green, blue = frame.unbind(dim=-1)
    luma = red * 0.299 + green * 0.587 + blue * 0.114
    axis_a = red * 0.596 - green * 0.274 - blue * 0.322
    axis_b = red * 0.211 - green * 0.523 + blue * 0.312
    return luma[..., None], axis_a[..., None], axis_b[..., None]


def _signal_to_rgb(luma: torch.Tensor, axis_a: torch.Tensor, axis_b: torch.Tensor) -> torch.Tensor:
    red = luma + axis_a * 0.956 + axis_b * 0.621
    green = luma - axis_a * 0.272 - axis_b * 0.647
    blue = luma - axis_a * 1.106 + axis_b * 1.703
    return torch.cat((red, green, blue), dim=-1)


_ANALOG_LOOK_PARAMS = {
    # saturation, contrast, lift, RGB gain, RGB offset
    "consumer_color": (1.38, 1.08, 0.018, (1.075, 1.025, 0.90), (0.014, 0.002, -0.010)),
    "warm_camcorder": (1.18, 1.03, 0.040, (1.105, 1.025, 0.84), (0.018, 0.004, -0.014)),
    "cool_camcorder": (1.12, 1.06, 0.030, (0.89, 1.020, 1.125), (-0.012, 0.002, 0.018)),
    "tungsten_home_video": (1.26, 1.04, 0.045, (1.145, 1.035, 0.76), (0.025, 0.006, -0.020)),
    "fluorescent_cctv": (0.58, 1.12, 0.055, (0.78, 1.135, 1.00), (-0.012, 0.018, 0.006)),
    "sodium_vapor_cctv": (0.72, 1.10, 0.065, (1.19, 1.075, 0.58), (0.025, 0.014, -0.026)),
    "late_night_blue": (0.86, 1.09, 0.050, (0.76, 0.96, 1.18), (-0.020, -0.002, 0.025)),
    "sun_bleached_tape": (0.62, 0.91, 0.125, (1.13, 1.035, 0.77), (0.026, 0.010, -0.018)),
    "rf_cyan_fade": (0.54, 0.96, 0.105, (0.75, 1.055, 1.12), (-0.018, 0.008, 0.020)),
    "archival_amber": (0.78, 0.94, 0.095, (1.16, 1.045, 0.72), (0.022, 0.008, -0.022)),
    "faded_magenta": (0.70, 0.88, 0.135, (1.13, 0.82, 1.105), (0.026, -0.012, 0.022)),
    "crt_broadcast": (1.32, 1.12, 0.012, (1.045, 1.015, 1.075), (0.006, 0.000, 0.010)),
}

_ANALOG_LOOK_MATRICES = {
    "consumer_color": ((1.090, 0.022, -0.050), (0.008, 1.040, -0.018), (-0.035, 0.020, 1.055)),
    "warm_camcorder": ((1.095, 0.025, -0.055), (0.018, 1.025, -0.018), (-0.045, 0.030, 0.985)),
    "cool_camcorder": ((0.970, 0.005, -0.018), (-0.016, 1.025, 0.018), (-0.050, 0.025, 1.095)),
    "tungsten_home_video": ((1.120, 0.030, -0.065), (0.020, 1.035, -0.020), (-0.055, 0.035, 0.965)),
    "fluorescent_cctv": ((0.940, 0.020, -0.020), (-0.030, 1.095, 0.020), (-0.045, 0.055, 1.030)),
    "sodium_vapor_cctv": ((1.100, 0.035, -0.070), (0.025, 1.070, -0.030), (-0.060, 0.035, 0.900)),
    "late_night_blue": ((0.950, 0.010, -0.025), (-0.020, 1.015, 0.025), (-0.060, 0.035, 1.110)),
    "sun_bleached_tape": ((1.080, 0.030, -0.040), (0.018, 1.025, -0.010), (-0.030, 0.040, 0.940)),
    "rf_cyan_fade": ((0.950, 0.020, -0.020), (-0.025, 1.050, 0.025), (-0.045, 0.050, 1.075)),
    "archival_amber": ((1.095, 0.035, -0.055), (0.020, 1.035, -0.015), (-0.040, 0.035, 0.915)),
    "faded_magenta": ((1.075, -0.025, 0.045), (0.015, 0.900, 0.008), (0.050, -0.020, 1.065)),
    "crt_broadcast": ((1.070, 0.008, -0.020), (-0.010, 1.040, 0.004), (-0.020, 0.016, 1.080)),
}

_ANALOG_LOOK_SPLIT_TONES = {
    # shadow tint, highlight tint; offsets are deliberately asymmetric so the
    # look remains apparent even when the composite carrier loses saturation.
    "consumer_color": ((0.000, -0.006, 0.010), (0.018, 0.006, -0.018)),
    "warm_camcorder": ((0.010, 0.000, -0.012), (0.035, 0.015, -0.035)),
    "cool_camcorder": ((-0.018, 0.002, 0.025), (-0.010, 0.008, 0.022)),
    "tungsten_home_video": ((0.018, 0.004, -0.025), (0.052, 0.025, -0.055)),
    "fluorescent_cctv": ((-0.018, 0.032, 0.010), (-0.020, 0.026, 0.006)),
    "sodium_vapor_cctv": ((0.030, 0.020, -0.040), (0.055, 0.030, -0.065)),
    "late_night_blue": ((-0.035, -0.006, 0.055), (-0.018, 0.004, 0.035)),
    "sun_bleached_tape": ((0.020, 0.010, -0.025), (0.045, 0.026, -0.050)),
    "rf_cyan_fade": ((-0.028, 0.018, 0.035), (-0.020, 0.014, 0.025)),
    "archival_amber": ((0.022, 0.010, -0.032), (0.048, 0.026, -0.055)),
    "faded_magenta": ((0.025, -0.025, 0.032), (0.040, -0.020, 0.035)),
    "crt_broadcast": ((-0.006, -0.004, 0.015), (0.012, 0.004, 0.016)),
}


def _apply_analog_color_look(frame: torch.Tensor, look: str) -> torch.Tensor:
    """Small analytic 3-channel look transforms used by analogue presets."""
    look = str(look)
    luma = frame.mul(frame.new_tensor((0.299, 0.587, 0.114))).sum(dim=-1, keepdim=True)
    if look == "night_vision_green":
        shaped = ((luma - 0.035).clamp_min(0.0) * 1.20).pow(0.88)
        return torch.cat((shaped * 0.32, shaped * 1.04, shaped * 0.25), dim=-1).clamp(0.0, 1.0)
    params = _ANALOG_LOOK_PARAMS.get(look)
    if params is None:
        return frame
    saturation, contrast, lift, gains, offsets = params
    result = luma + (frame - luma) * saturation
    result = (result - 0.5) * contrast + 0.5
    result = result * (1.0 - lift) + lift * 0.5
    result = result * frame.new_tensor(gains) + frame.new_tensor(offsets)
    matrix = frame.new_tensor(_ANALOG_LOOK_MATRICES[look])
    result = torch.matmul(result, matrix.transpose(0, 1))
    shadow_tint, highlight_tint = _ANALOG_LOOK_SPLIT_TONES[look]
    look_luma = result.mul(frame.new_tensor((0.299, 0.587, 0.114))).sum(dim=-1, keepdim=True)
    shadow_position = (look_luma / 0.58).clamp(0.0, 1.0)
    shadow_weight = 1.0 - shadow_position.square() * (3.0 - 2.0 * shadow_position)
    highlight_position = ((look_luma - 0.42) / 0.58).clamp(0.0, 1.0)
    highlight_weight = highlight_position.square() * (3.0 - 2.0 * highlight_position)
    result = (
        result
        + shadow_weight * frame.new_tensor(shadow_tint)
        + highlight_weight * frame.new_tensor(highlight_tint)
    )
    # Gentle channel shoulder emulates the compressed colour peaks of a
    # consumer analogue chain without clipping them like a simple RGB gain.
    result = result / (1.0 + result.clamp_min(0.0) * 0.035)
    return result.clamp(0.0, 1.0)


def _apply_cine_post_effects(
    frame: torch.Tensor,
    settings: dict[str, float | int | str],
    frame_index: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """IAMCCS signal-space tape/broadcast finishing, authored independently."""
    color_look = str(settings.get("analog_color_look", "neutral"))
    disturbance_amount = max(0.0, min(2.0, float(settings.get("disturbance_amount", 1.0))))

    def disturbance(name: str) -> float:
        """Scale a preset's coupled signal defects with one master control."""
        return max(0.0, min(1.0, float(settings.get(name, 0.0)) * disturbance_amount))

    temporal_seed = int(settings.get("seed", 1))

    def randomness(name: str) -> float:
        return max(0.0, min(1.0, float(settings.get(name, 0.0))))

    def temporal_noise(salt: int, rate: float = 0.37) -> float:
        return _temporal_random_value(frame_index, temporal_seed, salt, rate)

    if not any(float(settings.get(name, 0.0)) > 0.0 for name in ANALOG_FIELDS) and color_look == "neutral":
        return frame

    full_resolution_source = frame
    engine = str(settings.get("engine", "analog_tape"))
    use_signal_raster = disturbance_amount > 0.0 and engine in ("analog_tape", "broadcast_signal")
    if use_signal_raster:
        signal_height, signal_width = _analogue_signal_size(*frame.shape[:2])
        if (signal_height, signal_width) != frame.shape[:2]:
            signal = F.interpolate(
                frame.permute(2, 0, 1).unsqueeze(0),
                size=(signal_height, signal_width),
                mode="area",
            )[0].permute(1, 2, 0)
        else:
            signal = frame
    else:
        signal = frame
    source = signal
    result = _apply_analog_color_look(signal, color_look)
    height, width = result.shape[:2]
    is_tape = engine == "analog_tape"
    if disturbance_amount <= 0.0:
        # Keep the selected camera/tape colour treatment while bypassing every
        # time-varying signal defect. Saturation and black lift are grading
        # controls, so they intentionally remain active at zero disturbance.
        saturation = max(0.0, min(2.0, float(settings.get("signal_saturation", 1.0))))
        black_lift = max(0.0, min(1.0, float(settings.get("black_lift", 0.0))))
        if saturation != 1.0 or black_lift > 0.0:
            luma, axis_a, axis_b = _rgb_to_signal(result)
            axis_a = axis_a * saturation
            axis_b = axis_b * saturation
            if black_lift > 0.0:
                luma = luma * (1.0 - 0.34 * black_lift) + black_lift * 0.075
            result = _signal_to_rgb(luma, axis_a, axis_b)
        mix = max(0.0, min(1.0, float(settings.get("analog_mix", 1.0))))
        return torch.lerp(full_resolution_source, result.clamp(0.0, 1.0), mix)
    phase = frame_index * 0.37 + int(settings.get("seed", 1)) * 0.00031

    roll = disturbance("vertical_roll")
    if roll > 0.0:
        random_amount = randomness("vertical_roll_randomness")
        regular_wave = math.sin(phase * 0.43)
        irregular_wave = temporal_noise(101, 0.21)
        roll_wave = regular_wave * (1.0 - random_amount) + irregular_wave * random_amount
        # At high randomness, occasional deterministic gate slips create film-like
        # vertical jumps without introducing one-frame white-noise flicker.
        jump_gate = temporal_noise(103, 0.91)
        if random_amount > 0.0 and jump_gate > 0.72 - 0.34 * random_amount:
            roll_wave += math.copysign((jump_gate - 0.35) * random_amount, temporal_noise(107, 0.53))
        shift_y = int(round(roll_wave * height * (0.10 if is_tape else 0.16) * roll))
        if shift_y:
            result = torch.roll(result, shifts=shift_y, dims=0)

    instability = disturbance("horizontal_instability")
    tape_warp = disturbance("tape_warp")
    tracking = disturbance("tracking_error")
    if instability > 0.0 or tape_warp > 0.0 or tracking > 0.0:
        instability_random = randomness("horizontal_instability_randomness")
        warp_random = randomness("tape_warp_randomness")
        tracking_random = randomness("tracking_randomness")
        instability_gain = 1.0 + temporal_noise(211, 0.43) * 0.70 * instability_random
        warp_gain = 1.0 + temporal_noise(223, 0.19) * 0.65 * warp_random
        warped_phase = phase + temporal_noise(227, 0.16) * math.pi * warp_random
        rows = torch.arange(height, device=result.device, dtype=torch.float32)
        slow = torch.sin(rows * (0.014 + tape_warp * 0.018) + warped_phase) * width * 0.018 * tape_warp * warp_gain
        flutter = torch.sin(rows * 0.17 + phase * 2.7 + temporal_noise(229, 0.48) * instability_random) * width * 0.0045 * instability * instability_gain
        random_profile = torch.randn((height,), generator=generator, device=result.device)
        random_profile = _smooth_row_profile(random_profile, max(1, height // 42))
        offsets = slow + flutter + random_profile * width * 0.012 * instability * (1.0 + 0.75 * instability_random)
        if tracking > 0.0:
            # Tracking faults are short groups of displaced lines, not another sine warp.
            fault_count = max(1, int(round(1.0 + tracking * 5.0 * (1.0 + 0.55 * tracking_random))))
            for fault_index in range(fault_count):
                band_h = max(1, int(torch.randint(1, max(2, height // 16), (), generator=generator, device=result.device)))
                start = int(torch.randint(0, max(1, height - band_h + 1), (), generator=generator, device=result.device))
                direction = -1.0 if int(torch.randint(0, 2, (), generator=generator, device=result.device)) == 0 else 1.0
                displacement = direction * width * (0.008 + 0.055 * tracking) * (
                    1.0 + temporal_noise(241 + fault_index, 0.67) * 0.65 * tracking_random
                )
                envelope = torch.sin(torch.linspace(0.0, math.pi, band_h, device=result.device))
                offsets[start : start + band_h] += envelope * displacement
        result = _warp_rows(result, offsets)

    luma, axis_a, axis_b = _rgb_to_signal(result)
    saturation = max(0.0, min(2.0, float(settings.get("signal_saturation", 1.0))))
    black_lift = max(0.0, min(1.0, float(settings.get("black_lift", 0.0))))
    axis_a = axis_a * saturation
    axis_b = axis_b * saturation
    if black_lift > 0.0:
        luma = luma * (1.0 - 0.34 * black_lift) + black_lift * 0.075

    spill = disturbance("chroma_spill")
    delay = disturbance("chroma_delay")
    if spill > 0.0:
        kernel_x = min(41, 1 + 2 * max(1, int(round(spill * max(2.0, width / 110.0)))))
        kernel_y = 3 if is_tape and spill > 0.18 else 1
        axis_a = _blur_hwc(axis_a, kernel_y, kernel_x)
        axis_b = _blur_hwc(axis_b, kernel_y, kernel_x)
    if delay > 0.0:
        pixels = max(1, int(round(delay * max(3.0, width / 78.0))))
        axis_a = _shift_hwc(axis_a, pixels)
        axis_b = _shift_hwc(axis_b, pixels + (1 if is_tape else -1))

    phase_noise = disturbance("chroma_phase_noise")
    if phase_noise > 0.0:
        row_noise = torch.randn((height,), generator=generator, device=result.device)
        row_noise = _smooth_row_profile(row_noise, 2 if is_tape else 1)
        angle = row_noise.view(height, 1, 1) * phase_noise * (0.34 if is_tape else 0.58)
        cosine, sine = torch.cos(angle), torch.sin(angle)
        original_a = axis_a
        axis_a = original_a * cosine - axis_b * sine
        axis_b = original_a * sine + axis_b * cosine

    chroma_loss = disturbance("chroma_loss")
    if chroma_loss > 0.0:
        row_gate = torch.rand((height,), generator=generator, device=result.device)
        row_gate = (row_gate < chroma_loss * (0.13 if is_tape else 0.24)).view(height, 1, 1)
        if row_gate.any():
            axis_a = torch.where(row_gate, axis_a * (0.08 if is_tape else 0.0), axis_a)
            axis_b = torch.where(row_gate, axis_b * (0.08 if is_tape else 0.0), axis_b)

    result = _signal_to_rgb(luma, axis_a, axis_b)

    drift = disturbance("color_drift")
    if drift > 0.0:
        drift_random = randomness("color_drift_randomness")
        drift_phase = phase + temporal_noise(307, 0.23) * math.pi * drift_random
        drift_gain = 1.0 + temporal_noise(311, 0.41) * 0.55 * drift_random
        gains = result.new_tensor(
            (
                1.0 + math.sin(drift_phase) * 0.075 * drift * drift_gain,
                1.0 + math.sin(drift_phase + 2.1) * 0.045 * drift * drift_gain,
                1.0 + math.sin(drift_phase + 4.2) * 0.085 * drift * drift_gain,
            )
        )
        result = result * gains

    resonance = disturbance("edge_echo")
    if resonance > 0.0:
        current_luma, current_a, current_b = _rgb_to_signal(result)
        edge = current_luma - _blur_hwc(current_luma, 1, 5 if width >= 320 else 3)
        distance = max(1, int(round((1.0 + resonance * 6.0) * max(1.0, width / 960.0))))
        ringing = _shift_hwc(edge, distance) - _shift_hwc(edge, distance * 2) * 0.62
        current_luma = current_luma + ringing * resonance * (0.95 if engine == "broadcast_signal" else 0.58)
        result = _signal_to_rgb(current_luma, current_a, current_b)

    ghost = disturbance("ghost_echo")
    if ghost > 0.0:
        distance = max(2, int(round((8.0 + ghost * 34.0) * max(0.5, width / 720.0))))
        first = _shift_hwc(result, distance)
        second = _shift_hwc(result, distance * 2)
        result = result + first * ghost * 0.22 + second * ghost * 0.08
        result = result / (1.0 + ghost * 0.18)

    trail = disturbance("luma_trail")
    if trail > 0.0:
        current_luma = result.mul(result.new_tensor((0.2126, 0.7152, 0.0722))).sum(dim=-1, keepdim=True)
        distance = max(1, int(round((2.0 + trail * 18.0) * max(1.0, width / 1920.0))))
        smeared = (
            current_luma
            + _shift_hwc(current_luma, distance)
            + _shift_hwc(current_luma, distance * 2) * 0.6
        ) / 2.6
        result = result + (smeared - current_luma) * trail

    luma_noise = disturbance("luma_noise")
    if luma_noise > 0.0:
        noise = torch.randn((height, width, 1), generator=generator, device=result.device)
        if is_tape:
            noise = _blur_hwc(noise, 1, 3)
        noise = noise * ((0.095 if is_tape else 0.13) * luma_noise)
        result = result + noise
    chroma_noise = disturbance("chroma_noise")
    if chroma_noise > 0.0:
        colour_noise = torch.randn((height, width, 3), generator=generator, device=result.device)
        colour_noise = colour_noise - colour_noise.mean(dim=-1, keepdim=True)
        if is_tape:
            colour_noise = _blur_hwc(colour_noise, 3, 5)
        result = result + colour_noise * ((0.10 if is_tape else 0.16) * chroma_noise)

    interlace = disturbance("field_interlace")
    if interlace > 0.0 and height > 1:
        interlace_random = randomness("interlace_randomness")
        rows = torch.arange(height, device=result.device).remainder(2).bool().view(height, 1, 1)
        regular_direction = 1 if frame_index % 2 == 0 else -1
        random_direction = 1 if temporal_noise(401, 1.0) >= 0.0 else -1
        direction = (
            random_direction
            if interlace_random > 0.0 and temporal_noise(409, 0.83) > 1.0 - 2.0 * interlace_random
            else regular_direction
        )
        field_gain = max(0.25, 1.0 + temporal_noise(419, 0.52) * 0.70 * interlace_random)
        shifted = _shift_hwc(result, direction)
        field = result * (1.0 - 0.20 * interlace * field_gain) + shifted * (0.08 * interlace * field_gain)
        result = torch.where(rows, field, result)

    scanlines = disturbance("scanline_strength")
    if scanlines > 0.0:
        scanline_random = randomness("scanline_randomness")
        rows = torch.arange(height, device=result.device, dtype=torch.float32)
        scan_phase = phase * 0.13 + temporal_noise(503, 0.61) * math.pi * scanline_random
        pattern = 0.5 + 0.5 * torch.cos(rows * math.pi + scan_phase)
        depth = (0.22 if is_tape else 0.38) * max(
            0.35, 1.0 + temporal_noise(509, 0.34) * 0.60 * scanline_random
        )
        result = result * (1.0 - pattern.view(height, 1, 1) * scanlines * depth)

    dropouts = disturbance("line_dropout")
    if dropouts > 0.0:
        band_count = max(1, int(round(dropouts * (9.0 if is_tape else 14.0))))
        for _ in range(band_count):
            band_h = max(1, int(torch.randint(1, max(2, int(height * 0.025) + 1), (), generator=generator, device=result.device)))
            y0 = int(torch.randint(0, max(1, height - band_h + 1), (), generator=generator, device=result.device))
            x0 = int(torch.randint(0, max(1, width), (), generator=generator, device=result.device))
            length = max(1, int(width * (0.08 + 0.55 * dropouts)))
            x1 = min(width, x0 + length)
            if x1 > x0:
                stripe = torch.rand((band_h, x1 - x0, 1), generator=generator, device=result.device)
                alpha = 0.28 + 0.55 * dropouts
                if not is_tape:
                    stripe = (stripe > 0.58).to(result.dtype) * 0.82
                result[y0 : y0 + band_h, x0:x1] = torch.lerp(
                    result[y0 : y0 + band_h, x0:x1], stripe.expand(-1, -1, 3), alpha
                )

    head_switch = disturbance("head_switch_distortion")
    if head_switch > 0.0 and is_tape:
        head_random = randomness("head_switch_randomness")
        head_gain = max(0.25, 1.0 + temporal_noise(601, 0.73) * 0.70 * head_random)
        band_h = max(1, int(round(height * (0.015 + 0.075 * head_switch * head_gain))))
        start = height - band_h
        offsets = torch.linspace(width * 0.065 * head_switch * head_gain, 0.0, band_h, device=result.device)
        damaged = _warp_rows(result[start:], offsets)
        damaged = damaged * (1.0 - 0.18 * head_switch)
        damaged = damaged + torch.randn(damaged.shape, generator=generator, device=result.device) * (0.055 * head_switch)
        result = torch.cat((result[:start], damaged), dim=0)

    glow = disturbance("highlight_glow")
    if glow > 0.0:
        highlights = (result - 0.55).clamp_min(0.0)
        kernel = 3 + 2 * max(1, int(round(glow * 4.0)))
        result = result + _blur_hwc(highlights, kernel, kernel) * (0.42 * glow)

    mix = max(0.0, min(1.0, float(settings.get("analog_mix", 1.0))))
    result = result.clamp(0.0, 1.0)
    if result.shape[:2] != full_resolution_source.shape[:2]:
        result = F.interpolate(
            result.permute(2, 0, 1).unsqueeze(0),
            size=full_resolution_source.shape[:2],
            mode="bilinear",
            align_corners=False,
        )[0].permute(1, 2, 0)
    return torch.lerp(full_resolution_source, result, mix)


class IAMCCS_CineTemporalFilmGrain4K:
    """Fine animated grain intended after detail/upscale and before encoding."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "preset": (PRESETS, {"default": "65mm_4k_scan_subtle"}),
                "blend_method": (BLEND_METHODS, {"default": "density_exposure"}),
                "strength": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "grain_size_4k_px": ("FLOAT", {"default": 0.58, "min": 0.35, "max": 4.0, "step": 0.05, "display": "slider"}),
                "temporal_persistence": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 0.85, "step": 0.01, "display": "slider"}),
                "chroma_amount": ("FLOAT", {"default": 0.025, "min": 0.0, "max": 0.5, "step": 0.01, "display": "slider"}),
                "shadow_response": ("FLOAT", {"default": 0.48, "min": 0.0, "max": 2.0, "step": 0.02, "display": "slider"}),
                "midtone_response": ("FLOAT", {"default": 0.82, "min": 0.0, "max": 2.0, "step": 0.02, "display": "slider"}),
                "highlight_response": ("FLOAT", {"default": 0.24, "min": 0.0, "max": 2.0, "step": 0.02, "display": "slider"}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0x7FFFFFFFFFFFFFFF}),
                "frame_start": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("images", "grain_map", "report")
    FUNCTION = "apply"
    CATEGORY = "IAMCCS/Cine Nodes/Film Delivery"
    DESCRIPTION = (
        "Animated, resolution-aware film grain in linear light. Place after the final detail/upscale stage "
        "and before the encoder. Box values are render truth; preset selection only fills editable defaults in the UI."
    )

    def apply(
        self,
        images,
        preset,
        blend_method,
        strength,
        grain_size_4k_px,
        temporal_persistence,
        chroma_amount,
        shadow_response,
        midtone_response,
        highlight_response,
        seed,
        frame_start,
    ):
        if not torch.is_tensor(images) or images.ndim != 4 or images.shape[-1] not in (3, 4):
            raise ValueError("IAMCCS Cine Temporal Film Grain expects an IMAGE batch in BHWC RGB/RGBA format")
        if blend_method not in BLEND_METHODS:
            raise ValueError(f"Unknown grain blend method: {blend_method}")
        batch, height, width, channels = images.shape
        if batch == 0:
            return images, images.new_zeros((0, height, width)), "No frames received."
        if float(strength) <= 0.0:
            return (
                images.clone(),
                images.new_zeros((batch, height, width)),
                f"IAMCCS Cine Temporal Film Grain 4K | preset={preset} | bypassed=strength_zero | frames={batch}",
            )

        original_dtype = images.dtype
        # Keep only one output-sized allocation. Converting a complete long
        # fp16 batch to fp32 here would defeat the node's frame-wise memory
        # behavior; only the active frame is promoted for linear-light maths.
        output = images.clone()
        grain_map = torch.empty((batch, height, width), device=images.device, dtype=torch.float32)
        long_edge = max(height, width)
        resolved_size = max(0.35, float(grain_size_4k_px) * long_edge / 4096.0)
        persistence = min(0.85, max(0.0, float(temporal_persistence)))
        fresh_weight = math.sqrt(max(0.0, 1.0 - persistence * persistence))
        previous = None

        for frame_index in range(batch):
            generator = torch.Generator(device=images.device)
            generator.manual_seed((int(seed) + int(frame_start) + frame_index) & 0x7FFFFFFFFFFFFFFF)
            common = _normalized_field(height, width, resolved_size, generator, images.device)
            if previous is not None and persistence > 0.0:
                common = common * fresh_weight + previous * persistence
                common = (common - common.mean()) / common.std(unbiased=False).clamp_min(1e-6)
            previous = common
            common_hwc = common[0, 0, :, :, None]

            if float(chroma_amount) > 0.0:
                chroma = torch.cat([
                    _normalized_field(height, width, resolved_size * 1.08, generator, images.device)
                    for _ in range(3)
                ], dim=1)[0].permute(1, 2, 0)
                # Blue-biased chroma sensitivity is subtle and never replaces
                # the shared luminance grain structure.
                chroma[..., 2] *= 1.08
                noise = common_hwc * (1.0 - float(chroma_amount)) + chroma * float(chroma_amount)
            else:
                noise = common_hwc.expand(height, width, 3)

            frame_srgb = images[frame_index, ..., :3].to(torch.float32).clamp(0.0, 1.0)
            frame_linear = _srgb_to_linear(frame_srgb)
            luma = (frame_linear[..., 0] * 0.2126 + frame_linear[..., 1] * 0.7152 + frame_linear[..., 2] * 0.0722)
            shadow_w = ((0.50 - luma) / 0.50).clamp(0.0, 1.0)
            highlight_w = ((luma - 0.50) / 0.50).clamp(0.0, 1.0)
            mid_w = (1.0 - shadow_w - highlight_w).clamp(0.0, 1.0)
            tone = (
                shadow_w * float(shadow_response)
                + mid_w * float(midtone_response)
                + highlight_w * float(highlight_response)
            ).clamp(0.0, 2.0)
            sigma = float(strength) * 0.19 * tone[..., None]

            if blend_method == "density_exposure":
                processed_linear = frame_linear * torch.exp(noise * sigma - 0.5 * sigma.square())
                processed = _linear_to_srgb(processed_linear)
            elif blend_method == "linear_additive":
                processed = _linear_to_srgb(frame_linear + noise * sigma * 0.32)
            else:
                blend = (0.5 + noise * sigma * 1.9).clamp(0.0, 1.0)
                processed = _soft_light(frame_srgb, blend)

            output[frame_index, ..., :3] = processed.clamp(0.0, 1.0).to(original_dtype)
            grain_map[frame_index] = (common[0, 0].abs() / 3.0).clamp(0.0, 1.0)

        report = (
            f"IAMCCS Cine Temporal Film Grain 4K | preset={preset} | blend={blend_method} | "
            f"frames={batch} | {width}x{height} | strength={float(strength):.3f} | "
            f"grain_size_4k={float(grain_size_4k_px):.2f}px | resolved_size={resolved_size:.2f}px | "
            f"temporal_persistence={persistence:.2f} | chroma={float(chroma_amount):.2f} | "
            "linear_light=yes | repeated_texture=no"
        )
        return output, grain_map.to(original_dtype), report


class IAMCCS_CinePostEfxV2:
    """Colour-managed grain plus original IAMCCS analogue post-production effects."""

    SEARCH_ALIASES = [
        "IAMCCS CinePostEfx v2", "IAMCCS Grain Pro", "cine post effects",
        "film grain", "VHS", "analogue tape", "broadcast signal",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "engine": (PRO_ENGINES, {"default": "film_emulsion"}),
                "preset": (PRO_PRESETS, {"default": "35mm_fine_negative"}),
                "input_transfer": (INPUT_TRANSFERS, {"default": "srgb"}),
                "blend_method": (PRO_BLEND_METHODS, {"default": "log_density"}),
                "strength": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.005, "display": "slider"}),
                "grain_size_4k_px": ("FLOAT", {"default": 1.00, "min": 0.35, "max": 6.0, "step": 0.05, "display": "slider"}),
                "softness": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "roughness": ("FLOAT", {"default": 0.24, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "complexity": ("INT", {"default": 3, "min": 1, "max": 4, "step": 1, "display": "slider"}),
                "temporal_correlation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.95, "step": 0.01, "display": "slider"}),
                "chroma_amount": ("FLOAT", {"default": 0.055, "min": 0.0, "max": 0.5, "step": 0.005, "display": "slider"}),
                "shadow_response": ("FLOAT", {"default": 0.62, "min": 0.0, "max": 2.0, "step": 0.02, "display": "slider"}),
                "midtone_response": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.02, "display": "slider"}),
                "highlight_response": ("FLOAT", {"default": 0.38, "min": 0.0, "max": 2.0, "step": 0.02, "display": "slider"}),
                "red_response": ("FLOAT", {"default": 1.00, "min": 0.5, "max": 1.5, "step": 0.01, "display": "slider"}),
                "green_response": ("FLOAT", {"default": 0.96, "min": 0.5, "max": 1.5, "step": 0.01, "display": "slider"}),
                "blue_response": ("FLOAT", {"default": 1.06, "min": 0.5, "max": 1.5, "step": 0.01, "display": "slider"}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0x7FFFFFFFFFFFFFFF}),
                "frame_start": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
                "horizontal_instability": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "chroma_spill": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "luma_noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "chroma_noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "color_drift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "edge_echo": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "line_dropout": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "scanline_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "field_interlace": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "vertical_roll": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "highlight_glow": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "head_switch_distortion": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "luma_trail": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "chroma_delay": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "analog_mix": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "tracking_error": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "tape_warp": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "ghost_echo": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "signal_saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "display": "slider"}),
                "black_lift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "chroma_phase_noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "chroma_loss": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "grain_profile": (GRAIN_PROFILES, {"default": "negative_stock"}),
                "texture_microcontrast": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "analog_color_look": (ANALOG_COLOR_LOOKS, {"default": "neutral"}),
                "disturbance_amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "display": "slider"}),
                "lens_preset": (LENS_PRESETS, {"default": "lens_none"}),
                "lens_master": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01, "display": "slider"}),
                "lens_distortion": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "lens_edge_stretch": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "lens_anamorphic_width": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01, "display": "slider"}),
                "lens_zoom": ("FLOAT", {"default": 1.0, "min": 0.70, "max": 1.60, "step": 0.01, "display": "slider"}),
                "lens_keystone_x": ("FLOAT", {"default": 0.0, "min": -0.75, "max": 0.75, "step": 0.01, "display": "slider"}),
                "lens_keystone_y": ("FLOAT", {"default": 0.0, "min": -0.75, "max": 0.75, "step": 0.01, "display": "slider"}),
                "lens_tilt_angle": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 1.0, "display": "slider"}),
                "lens_focus_position": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "lens_tilt_blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "lens_chromatic_aberration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "lens_vignette": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "settings_json": ("STRING", {"default": "", "multiline": False}),
                "horizontal_instability_randomness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "tracking_randomness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "tape_warp_randomness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "color_drift_randomness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "scanline_randomness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "interlace_randomness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "vertical_roll_randomness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "head_switch_randomness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
            },
            "optional": {"grain_plate": ("IMAGE",)},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("images", "grain_map", "report")
    FUNCTION = "apply"
    CATEGORY = "IAMCCS/Cine Nodes/Film Delivery"
    DESCRIPTION = (
        "IAMCCS vectorized film/sensor/plate grain plus analogue tape and broadcast finishing. "
        "Includes stock texture profiles, density-aware grain, optical lens geometry and analogue colour looks. "
        "Includes temporal warp, colour separation, signal noise, echoes, drop-outs, scanlines, "
        "interlacing, roll, glow, head-switch distortion and luma trails, with independent smooth "
        "randomness controls for every cyclic signal-motion family."
    )

    @torch.inference_mode()
    def apply(
        self,
        images,
        engine,
        preset,
        input_transfer,
        blend_method,
        strength,
        grain_size_4k_px,
        softness,
        roughness,
        complexity,
        temporal_correlation,
        chroma_amount,
        shadow_response,
        midtone_response,
        highlight_response,
        red_response,
        green_response,
        blue_response,
        seed,
        frame_start,
        horizontal_instability=0.0,
        chroma_spill=0.0,
        luma_noise=0.0,
        chroma_noise=0.0,
        color_drift=0.0,
        edge_echo=0.0,
        line_dropout=0.0,
        scanline_strength=0.0,
        field_interlace=0.0,
        vertical_roll=0.0,
        highlight_glow=0.0,
        head_switch_distortion=0.0,
        luma_trail=0.0,
        chroma_delay=0.0,
        analog_mix=1.0,
        tracking_error=0.0,
        tape_warp=0.0,
        ghost_echo=0.0,
        signal_saturation=1.0,
        black_lift=0.0,
        chroma_phase_noise=0.0,
        chroma_loss=0.0,
        grain_profile="negative_stock",
        texture_microcontrast=0.18,
        analog_color_look="neutral",
        disturbance_amount=1.0,
        lens_preset="lens_none",
        lens_master=0.0,
        lens_distortion=0.0,
        lens_edge_stretch=0.0,
        lens_anamorphic_width=0.0,
        lens_zoom=1.0,
        lens_keystone_x=0.0,
        lens_keystone_y=0.0,
        lens_tilt_angle=0.0,
        lens_focus_position=0.0,
        lens_tilt_blur=0.0,
        lens_chromatic_aberration=0.0,
        lens_vignette=0.0,
        settings_json="",
        horizontal_instability_randomness=0.0,
        tracking_randomness=0.0,
        tape_warp_randomness=0.0,
        color_drift_randomness=0.0,
        scanline_randomness=0.0,
        interlace_randomness=0.0,
        vertical_roll_randomness=0.0,
        head_switch_randomness=0.0,
        grain_plate=None,
    ):
        if not torch.is_tensor(images) or images.ndim != 4 or images.shape[-1] not in (3, 4):
            raise ValueError("IAMCCS-CinePostEfx-v2 expects an IMAGE batch in BHWC RGB/RGBA format")

        settings = {
            "engine": engine, "input_transfer": input_transfer,
            "blend_method": blend_method, "strength": strength,
            "grain_size_4k_px": grain_size_4k_px, "softness": softness, "roughness": roughness,
            "complexity": complexity, "temporal_correlation": temporal_correlation,
            "chroma_amount": chroma_amount, "shadow_response": shadow_response,
            "midtone_response": midtone_response, "highlight_response": highlight_response,
            "red_response": red_response, "green_response": green_response, "blue_response": blue_response,
            "seed": seed, "frame_start": frame_start,
            "horizontal_instability": horizontal_instability, "chroma_spill": chroma_spill,
            "luma_noise": luma_noise, "chroma_noise": chroma_noise, "color_drift": color_drift,
            "edge_echo": edge_echo, "line_dropout": line_dropout,
            "scanline_strength": scanline_strength, "field_interlace": field_interlace,
            "vertical_roll": vertical_roll, "highlight_glow": highlight_glow,
            "head_switch_distortion": head_switch_distortion, "luma_trail": luma_trail,
            "chroma_delay": chroma_delay, "analog_mix": analog_mix,
            "tracking_error": tracking_error, "tape_warp": tape_warp, "ghost_echo": ghost_echo,
            "signal_saturation": signal_saturation, "black_lift": black_lift,
            "chroma_phase_noise": chroma_phase_noise, "chroma_loss": chroma_loss,
            "grain_profile": grain_profile, "texture_microcontrast": texture_microcontrast,
            "analog_color_look": analog_color_look, "disturbance_amount": disturbance_amount,
            "lens_preset": lens_preset, "lens_master": lens_master,
            "lens_distortion": lens_distortion, "lens_edge_stretch": lens_edge_stretch,
            "lens_anamorphic_width": lens_anamorphic_width, "lens_zoom": lens_zoom,
            "lens_keystone_x": lens_keystone_x, "lens_keystone_y": lens_keystone_y,
            "lens_tilt_angle": lens_tilt_angle, "lens_focus_position": lens_focus_position,
            "lens_tilt_blur": lens_tilt_blur,
            "lens_chromatic_aberration": lens_chromatic_aberration,
            "lens_vignette": lens_vignette,
            "horizontal_instability_randomness": horizontal_instability_randomness,
            "tracking_randomness": tracking_randomness,
            "tape_warp_randomness": tape_warp_randomness,
            "color_drift_randomness": color_drift_randomness,
            "scanline_randomness": scanline_randomness,
            "interlace_randomness": interlace_randomness,
            "vertical_roll_randomness": vertical_roll_randomness,
            "head_switch_randomness": head_switch_randomness,
        }
        snapshot_active = False
        if settings_json:
            try:
                snapshot = json.loads(str(settings_json))
            except json.JSONDecodeError as error:
                raise ValueError("IAMCCS-CinePostEfx-v2 received an invalid UI settings snapshot") from error
            if isinstance(snapshot, dict):
                settings.update({name: snapshot[name] for name in settings.keys() & snapshot.keys()})
                snapshot_active = True

        if not snapshot_active:
            preset_values = PRO_PRESET_VALUES.get(str(preset))
            if preset_values:
                settings.update({name: 0.0 for name in (*ANALOG_FIELDS, *TEMPORAL_RANDOMNESS_FIELDS)})
                settings["analog_mix"] = 1.0
                settings.update(preset_values)
            lens_preset_values = LENS_PRESET_VALUES.get(str(settings.get("lens_preset", "lens_none")))
            if lens_preset_values is not None:
                settings.update(lens_preset_values)

        engine = str(settings["engine"])
        input_transfer = str(settings["input_transfer"])
        blend_method = str(settings["blend_method"])
        strength = float(settings["strength"])
        grain_size_4k_px = float(settings["grain_size_4k_px"])
        softness = float(settings["softness"])
        roughness = float(settings["roughness"])
        complexity = int(settings["complexity"])
        temporal_correlation = float(settings["temporal_correlation"])
        chroma_amount = float(settings["chroma_amount"])
        shadow_response = float(settings["shadow_response"])
        midtone_response = float(settings["midtone_response"])
        highlight_response = float(settings["highlight_response"])
        red_response = float(settings["red_response"])
        green_response = float(settings["green_response"])
        blue_response = float(settings["blue_response"])
        seed = int(settings["seed"])
        frame_start = int(settings["frame_start"])
        grain_profile = str(settings.get("grain_profile", "negative_stock"))
        texture_microcontrast = max(0.0, min(1.0, float(settings.get("texture_microcontrast", 0.18))))
        analog_color_look = str(settings.get("analog_color_look", "neutral"))
        disturbance_amount = max(0.0, min(2.0, float(settings.get("disturbance_amount", 1.0))))
        lens_preset = str(settings.get("lens_preset", "lens_none"))
        lens_master = max(0.0, min(2.0, float(settings.get("lens_master", 0.0))))

        if (
            not snapshot_active
            and engine in ("analog_tape", "broadcast_signal")
            and not any(float(settings.get(name, 0.0)) > 0.0 for name in ANALOG_FIELDS)
        ):
            baseline = "soft_cassette_memory" if engine == "analog_tape" else "midnight_airwave"
            settings.update({
                name: value
                for name, value in PRO_PRESET_VALUES[baseline].items()
                if name in ANALOG_FIELDS or name in TEMPORAL_RANDOMNESS_FIELDS or name == "analog_mix"
            })

        if engine not in PRO_ENGINES:
            raise ValueError(f"Unknown CinePost engine: {engine}")
        if input_transfer not in INPUT_TRANSFERS:
            raise ValueError(f"Unknown input transfer: {input_transfer}")
        if blend_method not in PRO_BLEND_METHODS:
            raise ValueError(f"Unknown CinePost grain blend method: {blend_method}")
        if grain_profile not in GRAIN_PROFILES:
            raise ValueError(f"Unknown CinePost grain profile: {grain_profile}")
        if analog_color_look not in ANALOG_COLOR_LOOKS:
            raise ValueError(f"Unknown CinePost analogue colour look: {analog_color_look}")
        if lens_preset not in LENS_PRESETS:
            raise ValueError(f"Unknown CinePost lens preset: {lens_preset}")
        if engine == "scanned_grain_plate" and (
            not torch.is_tensor(grain_plate) or grain_plate.ndim != 4 or grain_plate.shape[0] == 0
        ):
            raise ValueError("scanned_grain_plate requires an IMAGE batch connected to grain_plate")

        batch, height, width, _channels = images.shape
        if batch == 0:
            return images, images.new_zeros((0, height, width)), "No frames received."
        analog_enabled = (
            any(float(settings.get(name, 0.0)) > 0.0 for name in ANALOG_FIELDS)
            or analog_color_look != "neutral"
        )
        lens_enabled = lens_master > 0.0
        if strength <= 0.0 and not analog_enabled and not lens_enabled:
            return images.clone(), images.new_zeros((batch, height, width)), (
                f"IAMCCS-CinePostEfx-v2 | engine={engine} | preset={preset} | "
                f"ui_snapshot={'yes' if snapshot_active else 'legacy'} | bypassed=no_effects_enabled"
            )

        original_dtype = images.dtype
        device = images.device
        output = images.clone()
        grain_map = torch.empty((batch, height, width), device=device, dtype=torch.float32)
        # The sub-pixel base keeps fine stocks fine while still allowing the
        # stock-size control to remain visibly different on common 1K/2K jobs.
        resolved_size = max(1.0, 0.72 + float(grain_size_4k_px) * max(height, width) / 4096.0)
        correlation = max(0.0, min(0.95, float(temporal_correlation)))
        fresh_weight = math.sqrt(max(0.0, 1.0 - correlation * correlation))
        previous_common = None
        fixed_pattern = None
        channel_response = torch.tensor(
            [float(red_response), float(green_response), float(blue_response)],
            device=device,
            dtype=torch.float32,
        ).view(1, 1, 3)
        profile_gain = _GRAIN_PROFILE_SHAPE[grain_profile][3]

        for frame_index in range(batch):
            generator = torch.Generator(device=device)
            generator.manual_seed((int(seed) + int(frame_start) + frame_index) & 0x7FFFFFFFFFFFFFFF)

            if engine == "scanned_grain_plate":
                plate_source = grain_plate[frame_index % grain_plate.shape[0]].to(device)
                # A still scan is translated per frame so it behaves as moving
                # emulsion rather than as a static screen-door overlay.
                if grain_plate.shape[0] == 1:
                    shift_x = int((int(seed) * 17 + (int(frame_start) + frame_index) * 37) % max(1, plate_source.shape[1]))
                    shift_y = int((int(seed) * 29 + (int(frame_start) + frame_index) * 53) % max(1, plate_source.shape[0]))
                    plate_source = torch.roll(plate_source, shifts=(shift_y, shift_x), dims=(0, 1))
                plate = _plate_field(plate_source, height, width)
                common = plate.mean(dim=1, keepdim=True)
                colour_field = plate[0].permute(1, 2, 0)
            else:
                layer_count = 3 if float(chroma_amount) > 0.0 else 1
                emulsion_layers = _pro_emulsion_layers(
                    height,
                    width,
                    resolved_size,
                    softness,
                    roughness,
                    complexity,
                    layer_count,
                    grain_profile,
                    generator,
                    device,
                )
                common = emulsion_layers.mean(dim=1, keepdim=True)
                common = common - common.mean(dim=(-2, -1), keepdim=True)
                common = common / common.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-6)
                common = common * profile_gain
                colour_field = emulsion_layers[0].permute(1, 2, 0) if layer_count == 3 else None

            if engine == "digital_sensor" and correlation > 0.0:
                if fixed_pattern is None:
                    fixed_generator = torch.Generator(device=device)
                    fixed_generator.manual_seed((int(seed) ^ 0x4F1BBCDC) & 0x7FFFFFFFFFFFFFFF)
                    fixed_pattern = _single_scale_field(height, width, 1.0, fixed_generator, device)
                common = common * fresh_weight + fixed_pattern * correlation
                common = (common - common.mean()) / common.std(unbiased=False).clamp_min(1e-6)
                common = common * profile_gain
            elif previous_common is not None and correlation > 0.0:
                common = common * fresh_weight + previous_common * correlation
                common = (common - common.mean()) / common.std(unbiased=False).clamp_min(1e-6)
                common = common * profile_gain
            previous_common = common
            common_hwc = common[0, 0, :, :, None]

            if colour_field is None:
                colour_field = common_hwc.expand(height, width, 3)

            chroma = max(0.0, min(0.5, float(chroma_amount)))
            noise = (common_hwc * (1.0 - chroma) + colour_field * chroma) * channel_response
            frame_display = images[frame_index, ..., :3].to(torch.float32).clamp(0.0, 1.0)
            if lens_enabled:
                frame_display = _apply_lens_effects(frame_display, settings)
            frame_linear = _decode_transfer(frame_display, input_transfer).clamp(0.0, 1.0)
            luma = frame_linear[..., 0] * 0.2126 + frame_linear[..., 1] * 0.7152 + frame_linear[..., 2] * 0.0722
            shadow_position = (luma / 0.58).clamp(0.0, 1.0)
            highlight_position = ((luma - 0.42) / 0.58).clamp(0.0, 1.0)
            shadow_w = 1.0 - shadow_position.square() * (3.0 - 2.0 * shadow_position)
            highlight_w = highlight_position.square() * (3.0 - 2.0 * highlight_position)
            mid_w = (1.0 - shadow_w - highlight_w).clamp(0.0, 1.0)
            tone = (
                shadow_w * float(shadow_response)
                + mid_w * float(midtone_response)
                + highlight_w * float(highlight_response)
            ).clamp(0.0, 2.0)
            # Fine scene detail masks grain perceptually; broad flat areas retain
            # the stock response. This avoids noisy edges and synthetic halos.
            luma_nchw = luma[None, None]
            local_luma = F.avg_pool2d(F.pad(luma_nchw, (1, 1, 1, 1), mode="replicate"), 3, stride=1)[0, 0]
            detail_mask = (1.0 - (luma - local_luma).abs() * 4.5).clamp(0.70, 1.0)
            tone = tone * detail_mask
            sigma = float(strength) * GRAIN_GAIN * tone[..., None]

            grain_base = frame_linear
            if texture_microcontrast > 0.0:
                base_blur = _blur_hwc(frame_linear, 3, 3)
                fine_detail = frame_linear - base_blur
                grain_base = frame_linear + fine_detail * texture_microcontrast * (0.16 + float(strength) * 0.42)

            if engine == "digital_sensor":
                # Sensor mode separates exposure-dependent photon noise from a
                # small read-noise floor; correlation becomes fixed-pattern noise.
                shot_scale = torch.sqrt(grain_base.clamp_min(1e-4))
                read_floor = 0.035 + 0.12 * float(roughness)
                processed_linear = grain_base + noise * sigma * (shot_scale + read_floor) * 0.92
                processed = _encode_transfer(processed_linear, input_transfer)
            elif blend_method == "log_density":
                density = -torch.log2(grain_base.clamp_min(1e-5))
                processed_linear = torch.pow(2.0, -(density - noise * sigma * 0.95))
                processed = _encode_transfer(processed_linear, input_transfer)
            elif blend_method == "density_exposure":
                processed_linear = grain_base * torch.exp(noise * sigma - 0.5 * sigma.square())
                processed = _encode_transfer(processed_linear, input_transfer)
            elif blend_method == "linear_additive":
                processed = _encode_transfer(grain_base + noise * sigma * 0.62, input_transfer)
            else:
                blend = (0.5 + noise * sigma * 1.9).clamp(0.0, 1.0)
                processed = _soft_light(_encode_transfer(grain_base, input_transfer).clamp(0.0, 1.0), blend)

            processed = _apply_cine_post_effects(
                processed.clamp(0.0, 1.0),
                settings,
                int(frame_start) + frame_index,
                generator,
            )
            output[frame_index, ..., :3] = processed.clamp(0.0, 1.0).to(original_dtype)
            # Neutral grey is zero grain; unlike an absolute map, this reveals
            # both positive and negative full-frame structure.
            grain_map[frame_index] = (0.5 + common[0, 0] / GRAIN_MAP_DIVISOR).clamp(0.0, 1.0)

        signal_height, signal_width = (
            _analogue_signal_size(height, width)
            if engine in ("analog_tape", "broadcast_signal") and disturbance_amount > 0.0
            else (height, width)
        )
        active_randomness = [
            float(settings.get(name, 0.0))
            for name in TEMPORAL_RANDOMNESS_FIELDS
            if float(settings.get(name, 0.0)) > 0.0
        ]
        report = (
            f"IAMCCS-CinePostEfx-v2 | engine={engine} | preset={preset} | blend={blend_method} | "
            f"transfer={input_transfer} | frames={batch} | {width}x{height} | strength={float(strength):.3f} | "
            f"grain_size_4k={float(grain_size_4k_px):.2f}px | resolved={resolved_size:.2f}px | "
            f"profile={grain_profile} | softness={float(softness):.2f} | roughness={float(roughness):.2f} | "
            f"complexity={int(complexity)} | microcontrast={texture_microcontrast:.2f} | "
            f"temporal_correlation={correlation:.2f} | analog_effects={'yes' if analog_enabled else 'no'} | "
            f"analog_look={analog_color_look} | analog_mix={float(settings['analog_mix']):.2f} | "
            f"disturbance={disturbance_amount:.2f}x | "
            f"signal_raster={signal_width}x{signal_height} | "
            f"temporal_randomness={len(active_randomness)}/{len(TEMPORAL_RANDOMNESS_FIELDS)} "
            f"(max={max(active_randomness, default=0.0):.2f}) | "
            f"ui_snapshot={'yes' if snapshot_active else 'legacy'} | "
            f"lens={lens_preset} | lens_master={lens_master:.2f}x | "
            "full_frame=yes | repeated_texture=no | vectorized_layers=yes"
        )
        return output, grain_map.to(original_dtype), report


class IAMCCS_CineFilmGrainProLegacy(IAMCCS_CinePostEfxV2):
    """Load old workflows without keeping the former node in the add-node menu."""

    DEPRECATED = True


class IAMCCS_CinePostEfxV2UnderscoreLegacy(IAMCCS_CinePostEfxV2):
    """Compatibility for the short-lived underscored v2 identifier."""

    DEPRECATED = True


NODE_CLASS_MAPPINGS = {
    "IAMCCS_CineTemporalFilmGrain4K": IAMCCS_CineTemporalFilmGrain4K,
    "IAMCCS-CinePostEfx-v2": IAMCCS_CinePostEfxV2,
    "IAMCCS_CinePostEfxV2": IAMCCS_CinePostEfxV2UnderscoreLegacy,
    "IAMCCS_CineFilmGrainPro": IAMCCS_CineFilmGrainProLegacy,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_CineTemporalFilmGrain4K": "IAMCCS Cine Temporal Film Grain · 4K Scan",
    "IAMCCS-CinePostEfx-v2": "IAMCCS-CinePostEfx-v2",
    "IAMCCS_CinePostEfxV2": "IAMCCS-CinePostEfx-v2",
    "IAMCCS_CineFilmGrainPro": "IAMCCS-CinePostEfx-v2 · Legacy workflow alias",
}
