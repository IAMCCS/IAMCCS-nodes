// SPDX-License-Identifier: GPL-3.0-or-later

import { app } from "/scripts/app.js";

const NODE_TYPES = new Set(["IAMCCS-CinePostEfx-v2", "IAMCCS_CinePostEfxV2", "IAMCCS_CineFilmGrainPro"]);

// Presets fill editable controls. The widget values remain the render truth.
const PRESETS = {
    "65mm_clean_scan": {
        engine: "film_emulsion", blend_method: "log_density", strength: 0.055,
        grain_size_4k_px: 0.70, softness: 0.30, roughness: 0.12, complexity: 2,
        temporal_correlation: 0.0, chroma_amount: 0.025,
        shadow_response: 0.50, midtone_response: 0.78, highlight_response: 0.28,
        red_response: 1.00, green_response: 0.98, blue_response: 1.03,
    },
    "35mm_fine_negative": {
        engine: "film_emulsion", blend_method: "log_density", strength: 0.10,
        grain_size_4k_px: 1.00, softness: 0.18, roughness: 0.24, complexity: 3,
        temporal_correlation: 0.0, chroma_amount: 0.055,
        shadow_response: 0.62, midtone_response: 1.00, highlight_response: 0.38,
        red_response: 1.00, green_response: 0.96, blue_response: 1.06,
    },
    "35mm_high_speed": {
        engine: "film_emulsion", blend_method: "log_density", strength: 0.16,
        grain_size_4k_px: 1.35, softness: 0.12, roughness: 0.46, complexity: 4,
        temporal_correlation: 0.0, chroma_amount: 0.085,
        shadow_response: 0.90, midtone_response: 1.20, highlight_response: 0.42,
        red_response: 1.02, green_response: 0.95, blue_response: 1.10,
    },
    "16mm_documentary": {
        engine: "film_emulsion", blend_method: "log_density", strength: 0.22,
        grain_size_4k_px: 1.90, softness: 0.08, roughness: 0.62, complexity: 4,
        temporal_correlation: 0.0, chroma_amount: 0.11,
        shadow_response: 1.02, midtone_response: 1.30, highlight_response: 0.48,
        red_response: 1.04, green_response: 0.94, blue_response: 1.13,
    },
    "8mm_expression": {
        engine: "film_emulsion", blend_method: "log_density", strength: 0.31,
        grain_size_4k_px: 3.10, softness: 0.04, roughness: 0.82, complexity: 4,
        temporal_correlation: 0.0, chroma_amount: 0.16,
        shadow_response: 1.12, midtone_response: 1.42, highlight_response: 0.58,
        red_response: 1.06, green_response: 0.92, blue_response: 1.18,
    },
    "digital_cinema_sensor": {
        engine: "digital_sensor", blend_method: "linear_additive", strength: 0.12,
        grain_size_4k_px: 0.62, softness: 0.05, roughness: 0.18, complexity: 2,
        temporal_correlation: 0.18, chroma_amount: 0.12,
        shadow_response: 1.22, midtone_response: 0.55, highlight_response: 0.18,
        red_response: 1.00, green_response: 0.92, blue_response: 1.16,
    },
    "soft_cassette_memory": {
        engine: "analog_tape", blend_method: "linear_additive", strength: 0.035,
        grain_size_4k_px: 1.25, softness: 0.42, roughness: 0.18, complexity: 2,
        temporal_correlation: 0.28, chroma_amount: 0.10,
        horizontal_instability: 0.12, chroma_spill: 0.28, luma_noise: 0.10,
        chroma_noise: 0.08, color_drift: 0.10, edge_echo: 0.08,
        line_dropout: 0.02, scanline_strength: 0.08, field_interlace: 0.04,
        vertical_roll: 0.0, highlight_glow: 0.16, head_switch_distortion: 0.03,
        luma_trail: 0.10, chroma_delay: 0.18, analog_mix: 0.82,
    },
    "worn_video_copy": {
        engine: "analog_tape", blend_method: "linear_additive", strength: 0.08,
        grain_size_4k_px: 1.65, softness: 0.28, roughness: 0.46, complexity: 3,
        temporal_correlation: 0.42, chroma_amount: 0.18,
        horizontal_instability: 0.52, chroma_spill: 0.58, luma_noise: 0.42,
        chroma_noise: 0.38, color_drift: 0.34, edge_echo: 0.36,
        line_dropout: 0.34, scanline_strength: 0.28, field_interlace: 0.30,
        vertical_roll: 0.14, highlight_glow: 0.22, head_switch_distortion: 0.52,
        luma_trail: 0.36, chroma_delay: 0.52, analog_mix: 0.92,
    },
    "midnight_airwave": {
        engine: "broadcast_signal", blend_method: "linear_additive", strength: 0.055,
        grain_size_4k_px: 0.80, softness: 0.14, roughness: 0.28, complexity: 2,
        temporal_correlation: 0.12, chroma_amount: 0.20,
        horizontal_instability: 0.22, chroma_spill: 0.46, luma_noise: 0.32,
        chroma_noise: 0.48, color_drift: 0.24, edge_echo: 0.62,
        line_dropout: 0.12, scanline_strength: 0.40, field_interlace: 0.54,
        vertical_roll: 0.06, highlight_glow: 0.12, head_switch_distortion: 0.0,
        luma_trail: 0.18, chroma_delay: 0.40, analog_mix: 0.88,
    },
    "projector_to_tape": {
        engine: "broadcast_signal", blend_method: "log_density", strength: 0.12,
        grain_size_4k_px: 1.70, softness: 0.16, roughness: 0.48, complexity: 4,
        temporal_correlation: 0.08, chroma_amount: 0.12,
        horizontal_instability: 0.16, chroma_spill: 0.24, luma_noise: 0.16,
        chroma_noise: 0.10, color_drift: 0.18, edge_echo: 0.30,
        line_dropout: 0.08, scanline_strength: 0.18, field_interlace: 0.12,
        vertical_roll: 0.02, highlight_glow: 0.52, head_switch_distortion: 0.06,
        luma_trail: 0.22, chroma_delay: 0.24, analog_mix: 0.76,
    },
};

// V2.1 analogue characters. These are IAMCCS-authored settings that expose
// separate tape, tracking, colour-carrier and broadcast behaviours.
Object.assign(PRESETS["65mm_clean_scan"], {strength: .070});
Object.assign(PRESETS["35mm_fine_negative"], {strength: .115});
Object.assign(PRESETS["35mm_high_speed"], {strength: .185});
Object.assign(PRESETS["16mm_documentary"], {strength: .255});
Object.assign(PRESETS["8mm_expression"], {strength: .36});
Object.assign(PRESETS["soft_cassette_memory"], {analog_mix:.88,tracking_error:.03,tape_warp:.12,ghost_echo:.02,signal_saturation:.96,black_lift:.07,chroma_phase_noise:.02,chroma_loss:0});
Object.assign(PRESETS["worn_video_copy"], {tracking_error:.58,tape_warp:.62,ghost_echo:.28,signal_saturation:.74,black_lift:.34,chroma_phase_noise:.42,chroma_loss:.28});
Object.assign(PRESETS["midnight_airwave"], {tracking_error:.18,tape_warp:.04,ghost_echo:.62,signal_saturation:.82,black_lift:.22,chroma_phase_noise:.52,chroma_loss:.18});
Object.assign(PRESETS["projector_to_tape"], {tracking_error:.08,tape_warp:.08,ghost_echo:.20,signal_saturation:.84,black_lift:.18,chroma_phase_noise:.10,chroma_loss:.04});
Object.assign(PRESETS, {
    pristine_tape_master: {engine:"analog_tape",blend_method:"linear_additive",strength:.025,grain_size_4k_px:1.10,softness:.46,roughness:.10,complexity:2,temporal_correlation:.20,chroma_amount:.06,horizontal_instability:.02,chroma_spill:.08,luma_noise:.025,chroma_noise:.02,color_drift:0,edge_echo:.08,line_dropout:0,scanline_strength:.025,field_interlace:.02,vertical_roll:0,highlight_glow:.025,head_switch_distortion:0,luma_trail:.03,chroma_delay:.03,analog_mix:1,tracking_error:0,tape_warp:.025,ghost_echo:.015,signal_saturation:.96,black_lift:.06,chroma_phase_noise:.01,chroma_loss:0},
    family_camcorder_1988: {engine:"analog_tape",blend_method:"linear_additive",strength:.075,grain_size_4k_px:1.45,softness:.30,roughness:.42,complexity:3,temporal_correlation:.36,chroma_amount:.16,horizontal_instability:.34,chroma_spill:.52,luma_noise:.30,chroma_noise:.28,color_drift:.25,edge_echo:.30,line_dropout:.16,scanline_strength:.15,field_interlace:.12,vertical_roll:.05,highlight_glow:.42,head_switch_distortion:.38,luma_trail:.50,chroma_delay:.44,analog_mix:1,tracking_error:.32,tape_warp:.38,ghost_echo:.10,signal_saturation:1.15,black_lift:.18,chroma_phase_noise:.22,chroma_loss:.10},
    overplayed_rental_tape: {engine:"analog_tape",blend_method:"linear_additive",strength:.11,grain_size_4k_px:1.80,softness:.18,roughness:.68,complexity:4,temporal_correlation:.48,chroma_amount:.24,horizontal_instability:.72,chroma_spill:.78,luma_noise:.68,chroma_noise:.60,color_drift:.52,edge_echo:.54,line_dropout:.58,scanline_strength:.36,field_interlace:.48,vertical_roll:.20,highlight_glow:.18,head_switch_distortion:.82,luma_trail:.68,chroma_delay:.74,analog_mix:1,tracking_error:.82,tape_warp:.86,ghost_echo:.44,signal_saturation:.70,black_lift:.40,chroma_phase_noise:.68,chroma_loss:.58},
    sun_faded_cassette: {engine:"analog_tape",blend_method:"linear_additive",strength:.06,grain_size_4k_px:1.50,softness:.34,roughness:.38,complexity:3,temporal_correlation:.30,chroma_amount:.12,horizontal_instability:.22,chroma_spill:.62,luma_noise:.22,chroma_noise:.32,color_drift:.32,edge_echo:.24,line_dropout:.08,scanline_strength:.22,field_interlace:.14,vertical_roll:.04,highlight_glow:.56,head_switch_distortion:.12,luma_trail:.42,chroma_delay:.56,analog_mix:1,tracking_error:.16,tape_warp:.24,ghost_echo:.14,signal_saturation:.42,black_lift:.54,chroma_phase_noise:.30,chroma_loss:.22},
    late_night_relay: {engine:"broadcast_signal",blend_method:"linear_additive",strength:.045,grain_size_4k_px:.85,softness:.12,roughness:.28,complexity:2,temporal_correlation:.10,chroma_amount:.22,horizontal_instability:.12,chroma_spill:.56,luma_noise:.42,chroma_noise:.62,color_drift:.28,edge_echo:.82,line_dropout:.10,scanline_strength:.46,field_interlace:.42,vertical_roll:.04,highlight_glow:.34,head_switch_distortion:0,luma_trail:.20,chroma_delay:.58,analog_mix:1,tracking_error:.12,tape_warp:.02,ghost_echo:.72,signal_saturation:.78,black_lift:.24,chroma_phase_noise:.72,chroma_loss:.30},
    damaged_airwave: {engine:"broadcast_signal",blend_method:"linear_additive",strength:.075,grain_size_4k_px:.75,softness:.06,roughness:.52,complexity:3,temporal_correlation:.08,chroma_amount:.30,horizontal_instability:.38,chroma_spill:.70,luma_noise:.78,chroma_noise:.86,color_drift:.54,edge_echo:.96,line_dropout:.42,scanline_strength:.62,field_interlace:.72,vertical_roll:.18,highlight_glow:.26,head_switch_distortion:0,luma_trail:.34,chroma_delay:.82,analog_mix:1,tracking_error:.46,tape_warp:.04,ghost_echo:.88,signal_saturation:.66,black_lift:.32,chroma_phase_noise:.94,chroma_loss:.72},
    archival_film_to_video: {engine:"broadcast_signal",blend_method:"log_density",strength:.15,grain_size_4k_px:1.85,softness:.16,roughness:.52,complexity:4,temporal_correlation:.02,chroma_amount:.10,horizontal_instability:.06,chroma_spill:.28,luma_noise:.18,chroma_noise:.12,color_drift:.16,edge_echo:.56,line_dropout:.04,scanline_strength:.14,field_interlace:.08,vertical_roll:.01,highlight_glow:.62,head_switch_distortion:0,luma_trail:.18,chroma_delay:.26,analog_mix:.86,tracking_error:.04,tape_warp:.02,ghost_echo:.24,signal_saturation:.82,black_lift:.20,chroma_phase_noise:.14,chroma_loss:.06},
});

const ANALOG_PRESET_BASE={engine:"broadcast_signal",blend_method:"linear_additive",strength:.05,grain_size_4k_px:.90,softness:.16,roughness:.36,complexity:3,temporal_correlation:.12,chroma_amount:.16,shadow_response:.90,midtone_response:.90,highlight_response:.32,red_response:1,green_response:1,blue_response:1,horizontal_instability:.10,chroma_spill:.30,luma_noise:.20,chroma_noise:.18,color_drift:.10,edge_echo:.30,line_dropout:.05,scanline_strength:.22,field_interlace:.25,vertical_roll:.02,highlight_glow:.10,head_switch_distortion:0,luma_trail:.12,chroma_delay:.22,analog_mix:1,tracking_error:.08,tape_warp:.02,ghost_echo:.16,signal_saturation:.90,black_lift:.12,chroma_phase_noise:.18,chroma_loss:.08};
const analogPreset=(values)=>({...ANALOG_PRESET_BASE,...values});
Object.assign(PRESETS, {
    cctv_monochrome_1997: analogPreset({strength:.035,chroma_amount:.01,horizontal_instability:.05,chroma_spill:.08,luma_noise:.52,chroma_noise:0,color_drift:0,edge_echo:.48,line_dropout:.12,scanline_strength:.58,field_interlace:.70,luma_trail:.18,chroma_delay:0,tracking_error:.12,ghost_echo:.08,signal_saturation:0,black_lift:.30,chroma_phase_noise:0,chroma_loss:0}),
    parking_garage_cctv: analogPreset({strength:.055,roughness:.58,chroma_amount:.03,horizontal_instability:.12,chroma_spill:.16,luma_noise:.76,chroma_noise:.05,edge_echo:.56,line_dropout:.28,scanline_strength:.64,field_interlace:.78,vertical_roll:.05,luma_trail:.28,tracking_error:.24,ghost_echo:.12,signal_saturation:.08,black_lift:.46,chroma_phase_noise:.04,chroma_loss:.06}),
    camcorder_night_recording: analogPreset({engine:"analog_tape",strength:.09,grain_size_4k_px:1.55,softness:.24,roughness:.66,temporal_correlation:.42,chroma_amount:.14,horizontal_instability:.34,chroma_spill:.58,luma_noise:.82,chroma_noise:.46,color_drift:.28,edge_echo:.28,line_dropout:.18,scanline_strength:.22,field_interlace:.20,highlight_glow:.58,head_switch_distortion:.32,luma_trail:.58,chroma_delay:.52,tracking_error:.36,tape_warp:.44,ghost_echo:.14,signal_saturation:.38,black_lift:.48,chroma_phase_noise:.38,chroma_loss:.26}),
    vhs_pause_damage: analogPreset({engine:"analog_tape",strength:.07,grain_size_4k_px:1.45,roughness:.54,temporal_correlation:.70,chroma_amount:.22,horizontal_instability:.62,chroma_spill:.72,luma_noise:.54,chroma_noise:.58,color_drift:.34,edge_echo:.48,line_dropout:.76,scanline_strength:.46,field_interlace:.64,vertical_roll:.08,head_switch_distortion:1,luma_trail:.46,chroma_delay:.78,tracking_error:1,tape_warp:.58,ghost_echo:.30,signal_saturation:.62,black_lift:.34,chroma_phase_noise:.72,chroma_loss:.66}),
    school_av_vhs: analogPreset({engine:"analog_tape",strength:.045,grain_size_4k_px:1.30,softness:.38,roughness:.30,temporal_correlation:.32,chroma_amount:.10,horizontal_instability:.18,chroma_spill:.46,luma_noise:.22,chroma_noise:.18,color_drift:.16,edge_echo:.22,line_dropout:.05,scanline_strength:.14,field_interlace:.12,highlight_glow:.28,head_switch_distortion:.08,luma_trail:.34,chroma_delay:.38,tracking_error:.14,tape_warp:.20,ghost_echo:.10,signal_saturation:.68,black_lift:.28,chroma_phase_noise:.16,chroma_loss:.10}),
    public_access_studio: analogPreset({strength:.035,softness:.22,roughness:.24,horizontal_instability:.08,chroma_spill:.40,luma_noise:.22,chroma_noise:.32,color_drift:.20,edge_echo:.68,line_dropout:.04,scanline_strength:.34,field_interlace:.36,highlight_glow:.30,luma_trail:.16,chroma_delay:.46,tracking_error:.08,ghost_echo:.52,signal_saturation:1.12,black_lift:.18,chroma_phase_noise:.38,chroma_loss:.12}),
    cable_channel_1993: analogPreset({strength:.055,roughness:.42,chroma_amount:.24,horizontal_instability:.20,chroma_spill:.62,luma_noise:.48,chroma_noise:.68,color_drift:.42,edge_echo:.76,line_dropout:.18,scanline_strength:.50,field_interlace:.56,vertical_roll:.10,highlight_glow:.20,luma_trail:.24,chroma_delay:.70,tracking_error:.26,ghost_echo:.66,signal_saturation:.84,black_lift:.26,chroma_phase_noise:.78,chroma_loss:.42}),
    weak_rooftop_antenna: analogPreset({strength:.065,softness:.05,roughness:.62,chroma_amount:.32,horizontal_instability:.48,chroma_spill:.68,luma_noise:.92,chroma_noise:1,color_drift:.64,edge_echo:.88,line_dropout:.62,scanline_strength:.58,field_interlace:.66,vertical_roll:.24,luma_trail:.30,chroma_delay:.88,tracking_error:.58,ghost_echo:.78,signal_saturation:.58,black_lift:.36,chroma_phase_noise:1,chroma_loss:.86}),
    clean_local_news_archive: analogPreset({strength:.025,softness:.32,roughness:.16,chroma_amount:.08,horizontal_instability:.03,chroma_spill:.22,luma_noise:.10,chroma_noise:.10,color_drift:.06,edge_echo:.44,line_dropout:.01,scanline_strength:.16,field_interlace:.22,highlight_glow:.18,luma_trail:.08,chroma_delay:.20,tracking_error:.02,ghost_echo:.22,signal_saturation:.94,black_lift:.12,chroma_phase_noise:.12,chroma_loss:.02}),
    industrial_monitor_feed: analogPreset({strength:.045,chroma_amount:.05,horizontal_instability:.16,chroma_spill:.18,luma_noise:.58,chroma_noise:.08,color_drift:.08,edge_echo:.62,line_dropout:.22,scanline_strength:.72,field_interlace:.82,vertical_roll:.12,highlight_glow:.06,luma_trail:.22,chroma_delay:.08,tracking_error:.32,ghost_echo:.18,signal_saturation:.18,black_lift:.38,chroma_phase_noise:.10,chroma_loss:.16}),
});

Object.assign(PRESETS, {
    vhs_consumer_color_pop: analogPreset({engine:"analog_tape",strength:.055,grain_size_4k_px:1.30,softness:.34,roughness:.34,temporal_correlation:.34,chroma_amount:.12,horizontal_instability:.20,chroma_spill:.42,luma_noise:.22,chroma_noise:.20,color_drift:.14,edge_echo:.24,line_dropout:.06,scanline_strength:.13,field_interlace:.12,highlight_glow:.34,head_switch_distortion:.10,luma_trail:.32,chroma_delay:.40,tracking_error:.16,tape_warp:.22,ghost_echo:.10,signal_saturation:1.18,black_lift:.18,chroma_phase_noise:.18,chroma_loss:.08,analog_color_look:"consumer_color"}),
    vhs_warm_family_tape: analogPreset({engine:"analog_tape",strength:.065,grain_size_4k_px:1.45,softness:.30,roughness:.42,temporal_correlation:.38,chroma_amount:.14,horizontal_instability:.30,chroma_spill:.54,luma_noise:.30,chroma_noise:.26,color_drift:.24,edge_echo:.28,line_dropout:.12,scanline_strength:.16,field_interlace:.15,highlight_glow:.48,head_switch_distortion:.30,luma_trail:.48,chroma_delay:.50,tracking_error:.28,tape_warp:.34,ghost_echo:.12,signal_saturation:1.08,black_lift:.22,chroma_phase_noise:.28,chroma_loss:.14,analog_color_look:"warm_camcorder"}),
    vhs_cool_camcorder: analogPreset({engine:"analog_tape",strength:.06,grain_size_4k_px:1.35,softness:.28,roughness:.38,temporal_correlation:.30,chroma_amount:.16,horizontal_instability:.26,chroma_spill:.48,luma_noise:.34,chroma_noise:.34,color_drift:.20,edge_echo:.30,line_dropout:.10,scanline_strength:.18,field_interlace:.20,highlight_glow:.30,head_switch_distortion:.22,luma_trail:.38,chroma_delay:.46,tracking_error:.24,tape_warp:.28,ghost_echo:.15,signal_saturation:1.02,black_lift:.20,chroma_phase_noise:.34,chroma_loss:.16,analog_color_look:"cool_camcorder"}),
    vhs_magenta_generation_loss: analogPreset({engine:"analog_tape",strength:.09,grain_size_4k_px:1.65,softness:.20,roughness:.58,temporal_correlation:.48,chroma_amount:.24,horizontal_instability:.58,chroma_spill:.76,luma_noise:.58,chroma_noise:.66,color_drift:.52,edge_echo:.44,line_dropout:.42,scanline_strength:.32,field_interlace:.40,vertical_roll:.12,highlight_glow:.22,head_switch_distortion:.68,luma_trail:.64,chroma_delay:.78,tracking_error:.66,tape_warp:.70,ghost_echo:.38,signal_saturation:.76,black_lift:.38,chroma_phase_noise:.72,chroma_loss:.52,analog_color_look:"faded_magenta"}),
    crt_saturated_broadcast: analogPreset({strength:.04,softness:.20,roughness:.28,chroma_amount:.20,horizontal_instability:.10,chroma_spill:.48,luma_noise:.26,chroma_noise:.36,color_drift:.18,edge_echo:.72,line_dropout:.05,scanline_strength:.44,field_interlace:.46,highlight_glow:.38,luma_trail:.14,chroma_delay:.50,tracking_error:.08,ghost_echo:.48,signal_saturation:1.14,black_lift:.18,chroma_phase_noise:.42,chroma_loss:.12,analog_color_look:"crt_broadcast"}),
    night_vision_cctv_green: analogPreset({strength:.055,chroma_amount:0,horizontal_instability:.10,chroma_spill:0,luma_noise:.78,chroma_noise:0,color_drift:0,edge_echo:.54,line_dropout:.24,scanline_strength:.66,field_interlace:.74,vertical_roll:.06,highlight_glow:.16,luma_trail:.26,chroma_delay:0,tracking_error:.24,ghost_echo:.12,signal_saturation:1,black_lift:.32,chroma_phase_noise:0,chroma_loss:0,analog_color_look:"night_vision_green"}),
});

const FILM_PROFILE_DEFAULTS={"65mm_clean_scan":["modern_fine",.10],"35mm_fine_negative":["negative_stock",.18],"35mm_high_speed":["high_speed_negative",.24],"16mm_documentary":["high_speed_negative",.30],"8mm_expression":["reversal_stock",.34],"digital_cinema_sensor":["modern_fine",.20],"archival_film_to_video":["print_stock",.22]};
const ANALOG_PRESET_COLOR_DEFAULTS={soft_cassette_memory:"warm_camcorder",worn_video_copy:"faded_magenta",midnight_airwave:"late_night_blue",projector_to_tape:"archival_amber",pristine_tape_master:"consumer_color",family_camcorder_1988:"tungsten_home_video",overplayed_rental_tape:"faded_magenta",sun_faded_cassette:"sun_bleached_tape",late_night_relay:"late_night_blue",damaged_airwave:"rf_cyan_fade",archival_film_to_video:"archival_amber",cctv_monochrome_1997:"neutral",parking_garage_cctv:"sodium_vapor_cctv",camcorder_night_recording:"cool_camcorder",vhs_pause_damage:"faded_magenta",school_av_vhs:"sun_bleached_tape",public_access_studio:"crt_broadcast",cable_channel_1993:"consumer_color",weak_rooftop_antenna:"rf_cyan_fade",clean_local_news_archive:"crt_broadcast",industrial_monitor_feed:"fluorescent_cctv"};
const TEMPORAL_RANDOMNESS_PAIRS=[["horizontal_instability_randomness","horizontal_instability"],["tracking_randomness","tracking_error"],["tape_warp_randomness","tape_warp"],["color_drift_randomness","color_drift"],["scanline_randomness","scanline_strength"],["interlace_randomness","field_interlace"],["vertical_roll_randomness","vertical_roll"],["head_switch_randomness","head_switch_distortion"]];
Object.entries(ANALOG_PRESET_COLOR_DEFAULTS).forEach(([name,look])=>{if(PRESETS[name])PRESETS[name].analog_color_look=look;});
Object.entries(PRESETS).forEach(([name,values])=>{const [profile,microcontrast]=FILM_PROFILE_DEFAULTS[name]||["negative_stock",.12];values.grain_profile??=profile;values.texture_microcontrast??=microcontrast;values.analog_color_look??="neutral";TEMPORAL_RANDOMNESS_PAIRS.forEach(([randomName,sourceName])=>{const amount=Number(values[sourceName]||0);values[randomName]??=amount>0?Math.min(.85,.10+.70*amount):0;});});

const LENS_PRESET_VALUES={
    lens_none:{lens_master:0,lens_distortion:0,lens_edge_stretch:0,lens_anamorphic_width:0,lens_zoom:1,lens_keystone_x:0,lens_keystone_y:0,lens_tilt_angle:0,lens_focus_position:0,lens_tilt_blur:0,lens_chromatic_aberration:0,lens_vignette:0},
    wide_angle_14mm:{lens_master:1,lens_distortion:.30,lens_edge_stretch:.20,lens_anamorphic_width:0,lens_zoom:1.08,lens_keystone_x:0,lens_keystone_y:0,lens_tilt_angle:0,lens_focus_position:0,lens_tilt_blur:0,lens_chromatic_aberration:.07,lens_vignette:.13},
    ultra_wide_10mm:{lens_master:1,lens_distortion:.52,lens_edge_stretch:.34,lens_anamorphic_width:0,lens_zoom:1.16,lens_keystone_x:0,lens_keystone_y:0,lens_tilt_angle:0,lens_focus_position:0,lens_tilt_blur:0,lens_chromatic_aberration:.11,lens_vignette:.20},
    fisheye_8mm:{lens_master:1,lens_distortion:.92,lens_edge_stretch:.55,lens_anamorphic_width:0,lens_zoom:1.28,lens_keystone_x:0,lens_keystone_y:0,lens_tilt_angle:0,lens_focus_position:0,lens_tilt_blur:0,lens_chromatic_aberration:.16,lens_vignette:.30},
    anamorphic_1_33x:{lens_master:1,lens_distortion:.10,lens_edge_stretch:.22,lens_anamorphic_width:.10,lens_zoom:1.06,lens_keystone_x:0,lens_keystone_y:0,lens_tilt_angle:0,lens_focus_position:0,lens_tilt_blur:0,lens_chromatic_aberration:.08,lens_vignette:.14},
    anamorphic_2x_cinema:{lens_master:1,lens_distortion:.17,lens_edge_stretch:.38,lens_anamorphic_width:.22,lens_zoom:1.10,lens_keystone_x:0,lens_keystone_y:0,lens_tilt_angle:0,lens_focus_position:0,lens_tilt_blur:0,lens_chromatic_aberration:.14,lens_vignette:.23},
    tilt_shift_architecture:{lens_master:1,lens_distortion:0,lens_edge_stretch:0,lens_anamorphic_width:0,lens_zoom:1.08,lens_keystone_x:0,lens_keystone_y:-.24,lens_tilt_angle:0,lens_focus_position:0,lens_tilt_blur:0,lens_chromatic_aberration:0,lens_vignette:0},
    tilt_shift_miniature:{lens_master:1,lens_distortion:0,lens_edge_stretch:0,lens_anamorphic_width:0,lens_zoom:1.06,lens_keystone_x:0,lens_keystone_y:-.06,lens_tilt_angle:-8,lens_focus_position:-.05,lens_tilt_blur:.72,lens_chromatic_aberration:0,lens_vignette:.14},
    vintage_28mm_barrel:{lens_master:1,lens_distortion:.25,lens_edge_stretch:.14,lens_anamorphic_width:0,lens_zoom:1.08,lens_keystone_x:0,lens_keystone_y:0,lens_tilt_angle:0,lens_focus_position:0,lens_tilt_blur:0,lens_chromatic_aberration:.24,lens_vignette:.38},
    telephoto_pincushion:{lens_master:1,lens_distortion:-.20,lens_edge_stretch:-.08,lens_anamorphic_width:0,lens_zoom:1.03,lens_keystone_x:0,lens_keystone_y:0,lens_tilt_angle:0,lens_focus_position:0,lens_tilt_blur:0,lens_chromatic_aberration:.04,lens_vignette:.08},
};
Object.assign(LENS_PRESET_VALUES.wide_angle_14mm,{lens_zoom:1.24});
Object.assign(LENS_PRESET_VALUES.ultra_wide_10mm,{lens_zoom:1.42});
Object.assign(LENS_PRESET_VALUES.fisheye_8mm,{lens_zoom:1.60});
Object.assign(LENS_PRESET_VALUES.anamorphic_1_33x,{lens_zoom:1.12});
Object.assign(LENS_PRESET_VALUES.anamorphic_2x_cinema,{lens_zoom:1.22});
Object.assign(LENS_PRESET_VALUES.tilt_shift_architecture,{lens_zoom:1.26});
Object.assign(LENS_PRESET_VALUES.tilt_shift_miniature,{lens_zoom:1.14});
Object.assign(LENS_PRESET_VALUES.vintage_28mm_barrel,{lens_zoom:1.20});

function findWidget(node, name) {
    return (node.widgets || []).find((candidate) => candidate.name === name);
}

function setWidget(node, name, value) {
    const target = findWidget(node, name);
    if (!target) return;
    target.value = value;
    target.callback?.(value);
}

function applyPreset(node, presetName) {
    const values = PRESETS[presetName];
    if (!values) return;
    node._iamccsApplyingProPreset = true;
    try {
        ["horizontal_instability","chroma_spill","luma_noise","chroma_noise","color_drift","edge_echo","line_dropout","scanline_strength","field_interlace","vertical_roll","highlight_glow","head_switch_distortion","luma_trail","chroma_delay","tracking_error","tape_warp","ghost_echo","chroma_phase_noise","chroma_loss",...TEMPORAL_RANDOMNESS_PAIRS.map(([name])=>name)].forEach((name) => setWidget(node, name, 0));
        setWidget(node, "signal_saturation", 1);
        setWidget(node, "black_lift", 0);
        setWidget(node, "analog_color_look", "neutral");
        setWidget(node, "analog_mix", 1);
        Object.entries(values).forEach(([name, value]) => setWidget(node, name, value));
    } finally {
        node._iamccsApplyingProPreset = false;
    }
    node._iamccsWriteProSnapshot?.();
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function applyLensPreset(node,presetName){
    const values=LENS_PRESET_VALUES[presetName];if(!values)return;
    node._iamccsApplyingLensPreset=true;
    try{Object.entries(values).forEach(([name,value])=>setWidget(node,name,value));}
    finally{node._iamccsApplyingLensPreset=false;}
    node._iamccsWriteProSnapshot?.();
    node.setDirtyCanvas?.(true,true);app.graph?.setDirtyCanvas?.(true,true);
}

const PRO_FIELDS = ["engine", "preset", "input_transfer", "blend_method", "strength", "grain_size_4k_px", "softness", "roughness", "complexity", "temporal_correlation", "chroma_amount", "shadow_response", "midtone_response", "highlight_response", "red_response", "green_response", "blue_response", "seed", "frame_start", "horizontal_instability", "chroma_spill", "luma_noise", "chroma_noise", "color_drift", "edge_echo", "line_dropout", "scanline_strength", "field_interlace", "vertical_roll", "highlight_glow", "head_switch_distortion", "luma_trail", "chroma_delay", "analog_mix", "tracking_error", "tape_warp", "ghost_echo", "signal_saturation", "black_lift", "chroma_phase_noise", "chroma_loss", "grain_profile", "texture_microcontrast", "analog_color_look", "disturbance_amount", "lens_preset", "lens_master", "lens_distortion", "lens_edge_stretch", "lens_anamorphic_width", "lens_zoom", "lens_keystone_x", "lens_keystone_y", "lens_tilt_angle", "lens_focus_position", "lens_tilt_blur", "lens_chromatic_aberration", "lens_vignette", "horizontal_instability_randomness", "tracking_randomness", "tape_warp_randomness", "color_drift_randomness", "scanline_randomness", "interlace_randomness", "vertical_roll_randomness", "head_switch_randomness", "settings_json"];
const SNAPSHOT_FIELDS = PRO_FIELDS.filter((name)=>name!=="settings_json"&&name!=="preset");
function readWidget(node, name, fallback) { return findWidget(node, name)?.value ?? fallback; }
function hideWidget(target) {
    if (!target || target._iamccsProHidden) return;
    target.serializeValue ||= (() => target.value);
    target.type = "hidden"; target.hidden = true; target.computeSize = () => [0, 0]; target.draw = () => {};
    target._iamccsProHidden = true;
}
function proNoise(x, y, frame, seed) {
    let value = (x * 374761393 + y * 668265263 + frame * 1442695041 + seed * 69069) | 0;
    value = Math.imul(value ^ (value >>> 13), 1274126177); value ^= value >>> 16;
    return ((value >>> 0) / 4294967295) * 2 - 1;
}
function proTemporalNoise(frame,seed,salt,rate=.37){const position=Math.max(0,Number(frame)||0)*rate,index=Math.floor(position),raw=position-index,fraction=raw*raw*(3-2*raw),first=proNoise(salt,index,0,seed),second=proNoise(salt,index+1,0,seed);return first+(second-first)*fraction;}
const clamp01 = (value) => Math.max(0, Math.min(1, value));
function decodeTransfer(value, transfer) {
    const c=clamp01(value);
    if(transfer==="linear")return c;
    if(transfer==="rec709")return c<.081?c/4.5:Math.pow((c+.099)/1.099,1/.45);
    return c<=.04045?c/12.92:Math.pow((c+.055)/1.055,2.4);
}
function encodeTransfer(value, transfer) {
    const c=Math.max(0,value);
    if(transfer==="linear")return c;
    if(transfer==="rec709")return c<.018?c*4.5:1.099*Math.pow(c,.45)-.099;
    return c<=.0031308?c*12.92:1.055*Math.pow(c,1/2.4)-.055;
}
function softLight(base, blend) {
    if(blend<=.5)return base-(1-2*blend)*base*(1-base);
    const d=base<=.25?((16*base-12)*base+4)*base:Math.sqrt(base);
    return base+(2*blend-1)*(d-base);
}
const ANALOG_LOOK_PARAMS={consumer_color:[1.38,1.08,.018,[1.075,1.025,.90],[.014,.002,-.010]],warm_camcorder:[1.18,1.03,.040,[1.105,1.025,.84],[.018,.004,-.014]],cool_camcorder:[1.12,1.06,.030,[.89,1.020,1.125],[-.012,.002,.018]],tungsten_home_video:[1.26,1.04,.045,[1.145,1.035,.76],[.025,.006,-.020]],fluorescent_cctv:[.58,1.12,.055,[.78,1.135,1],[-.012,.018,.006]],sodium_vapor_cctv:[.72,1.10,.065,[1.19,1.075,.58],[.025,.014,-.026]],late_night_blue:[.86,1.09,.050,[.76,.96,1.18],[-.020,-.002,.025]],sun_bleached_tape:[.62,.91,.125,[1.13,1.035,.77],[.026,.010,-.018]],rf_cyan_fade:[.54,.96,.105,[.75,1.055,1.12],[-.018,.008,.020]],archival_amber:[.78,.94,.095,[1.16,1.045,.72],[.022,.008,-.022]],faded_magenta:[.70,.88,.135,[1.13,.82,1.105],[.026,-.012,.022]],crt_broadcast:[1.32,1.12,.012,[1.045,1.015,1.075],[.006,0,.010]]};
const ANALOG_LOOK_MATRICES={consumer_color:[[1.090,.022,-.050],[.008,1.040,-.018],[-.035,.020,1.055]],warm_camcorder:[[1.095,.025,-.055],[.018,1.025,-.018],[-.045,.030,.985]],cool_camcorder:[[.970,.005,-.018],[-.016,1.025,.018],[-.050,.025,1.095]],tungsten_home_video:[[1.120,.030,-.065],[.020,1.035,-.020],[-.055,.035,.965]],fluorescent_cctv:[[.940,.020,-.020],[-.030,1.095,.020],[-.045,.055,1.030]],sodium_vapor_cctv:[[1.100,.035,-.070],[.025,1.070,-.030],[-.060,.035,.900]],late_night_blue:[[.950,.010,-.025],[-.020,1.015,.025],[-.060,.035,1.110]],sun_bleached_tape:[[1.080,.030,-.040],[.018,1.025,-.010],[-.030,.040,.940]],rf_cyan_fade:[[.950,.020,-.020],[-.025,1.050,.025],[-.045,.050,1.075]],archival_amber:[[1.095,.035,-.055],[.020,1.035,-.015],[-.040,.035,.915]],faded_magenta:[[1.075,-.025,.045],[.015,.900,.008],[.050,-.020,1.065]],crt_broadcast:[[1.070,.008,-.020],[-.010,1.040,.004],[-.020,.016,1.080]]};
const ANALOG_LOOK_SPLIT_TONES={consumer_color:[[0,-.006,.010],[.018,.006,-.018]],warm_camcorder:[[.010,0,-.012],[.035,.015,-.035]],cool_camcorder:[[-.018,.002,.025],[-.010,.008,.022]],tungsten_home_video:[[.018,.004,-.025],[.052,.025,-.055]],fluorescent_cctv:[[-.018,.032,.010],[-.020,.026,.006]],sodium_vapor_cctv:[[.030,.020,-.040],[.055,.030,-.065]],late_night_blue:[[-.035,-.006,.055],[-.018,.004,.035]],sun_bleached_tape:[[.020,.010,-.025],[.045,.026,-.050]],rf_cyan_fade:[[-.028,.018,.035],[-.020,.014,.025]],archival_amber:[[.022,.010,-.032],[.048,.026,-.055]],faded_magenta:[[.025,-.025,.032],[.040,-.020,.035]],crt_broadcast:[[-.006,-.004,.015],[.012,.004,.016]]};
function applyAnalogColorLook(data,look){
    if(!look||look==="neutral")return;
    for(let i=0;i<data.length;i+=4){
        const r=data[i]/255,g=data[i+1]/255,b=data[i+2]/255,luma=r*.299+g*.587+b*.114;
        if(look==="night_vision_green"){
            const shaped=Math.pow(Math.max(0,(luma-.035)*1.20),.88);data[i]=clamp01(shaped*.32)*255;data[i+1]=clamp01(shaped*1.04)*255;data[i+2]=clamp01(shaped*.25)*255;continue;
        }
        const params=ANALOG_LOOK_PARAMS[look];if(!params)continue;
        const [saturation,contrast,lift,gains,offsets]=params,channels=[r,g,b],shaped=[];
        for(let c=0;c<3;c++){let value=luma+(channels[c]-luma)*saturation;value=(value-.5)*contrast+.5;shaped[c]=(value*(1-lift)+lift*.5)*gains[c]+offsets[c];}
        const matrix=ANALOG_LOOK_MATRICES[look],graded=[0,0,0];for(let c=0;c<3;c++)graded[c]=shaped[0]*matrix[c][0]+shaped[1]*matrix[c][1]+shaped[2]*matrix[c][2];
        const gradedLuma=graded[0]*.299+graded[1]*.587+graded[2]*.114,shadowPos=clamp01(gradedLuma/.58),shadowWeight=1-shadowPos*shadowPos*(3-2*shadowPos),highlightPos=clamp01((gradedLuma-.42)/.58),highlightWeight=highlightPos*highlightPos*(3-2*highlightPos),tones=ANALOG_LOOK_SPLIT_TONES[look];
        for(let c=0;c<3;c++){let value=graded[c]+shadowWeight*tones[0][c]+highlightWeight*tones[1][c];value=value/(1+Math.max(0,value)*.035);data[i+c]=clamp01(value)*255;}
    }
}
function applyAnalogPreview(data, width, height, values, frame) {
    const fields=["horizontal_instability","chroma_spill","luma_noise","chroma_noise","color_drift","edge_echo","line_dropout","scanline_strength","field_interlace","vertical_roll","highlight_glow","head_switch_distortion","luma_trail","chroma_delay","tracking_error","tape_warp","ghost_echo","chroma_phase_noise","chroma_loss"];
    const colorLook=String(values.analog_color_look||"neutral");
    if(!fields.some((name)=>Number(values[name]||0)>0)&&colorLook==="neutral")return;
    const originalSource=new Uint8ClampedArray(data),source=new Uint8ClampedArray(data),mix=clamp01(Number(values.analog_mix??1)),seed=Number(values.seed||1),phase=frame*.37+seed*.00031,isTape=values.engine==="analog_tape",disturbanceAmount=Math.max(0,Math.min(2,Number(values.disturbance_amount??1))),disturbance=(name)=>clamp01(Number(values[name]||0)*disturbanceAmount),randomness=(name)=>clamp01(Number(values[name]||0)),temporal=(salt,rate=.37)=>proTemporalNoise(frame,seed,salt,rate);
    applyAnalogColorLook(source,colorLook);
    if(disturbanceAmount<=0){
        const saturation=Number(values.signal_saturation??1),blackLift=Number(values.black_lift||0);
        for(let i=0;i<data.length;i+=4){const grey=source[i]*.2126+source[i+1]*.7152+source[i+2]*.0722;for(let c=0;c<3;c++){let value=grey+(source[i+c]-grey)*saturation;value=value*(1-.34*blackLift)+blackLift*19.125;data[i+c]=originalSource[i+c]*(1-mix)+Math.max(0,Math.min(255,value))*mix;}}
        return;
    }
    const sample=(x,y,c)=>source[(Math.max(0,Math.min(height-1,y))*width+Math.max(0,Math.min(width-1,x)))*4+c];
    const instability=disturbance("horizontal_instability"),tapeWarp=disturbance("tape_warp"),tracking=disturbance("tracking_error"),spill=disturbance("chroma_spill"),delay=disturbance("chroma_delay"),resonance=disturbance("edge_echo"),ghost=disturbance("ghost_echo"),trail=disturbance("luma_trail"),roll=disturbance("vertical_roll"),scan=disturbance("scanline_strength"),interlace=disturbance("field_interlace"),drift=disturbance("color_drift"),lumaNoise=disturbance("luma_noise"),chromaNoise=disturbance("chroma_noise"),phaseNoise=disturbance("chroma_phase_noise"),chromaLoss=disturbance("chroma_loss"),drop=disturbance("line_dropout"),head=isTape?disturbance("head_switch_distortion"):0,glow=disturbance("highlight_glow"),saturation=Number(values.signal_saturation??1),blackLift=Number(values.black_lift||0);
    const rollRandom=randomness("vertical_roll_randomness"),rollGate=temporal(103,.91),rollWave=Math.sin(phase*.43)*(1-rollRandom)+temporal(101,.21)*rollRandom+(rollRandom>0&&rollGate>.72-.34*rollRandom?Math.sign(temporal(107,.53)||1)*(rollGate-.35)*rollRandom:0),rollY=Math.round(rollWave*height*(isTape?.10:.16)*roll),instabilityRandom=randomness("horizontal_instability_randomness"),trackingRandom=randomness("tracking_randomness"),warpRandom=randomness("tape_warp_randomness"),instabilityGain=1+temporal(211,.43)*.70*instabilityRandom,warpGain=1+temporal(223,.19)*.65*warpRandom,warpedPhase=phase+temporal(227,.16)*Math.PI*warpRandom,driftRandom=randomness("color_drift_randomness"),driftPhase=phase+temporal(307,.23)*Math.PI*driftRandom,driftStrength=1+temporal(311,.41)*.55*driftRandom,interlaceRandom=randomness("interlace_randomness"),interlaceGain=Math.max(.25,1+temporal(419,.52)*.70*interlaceRandom),interlaceDirection=interlaceRandom>0&&temporal(409,.83)>1-2*interlaceRandom?(temporal(401,1)>=0?1:-1):(frame%2?-1:1),scanRandom=randomness("scanline_randomness"),scanPhase=phase*.13+temporal(503,.61)*Math.PI*scanRandom,scanDepth=(isTape?.22:.38)*Math.max(.35,1+temporal(509,.34)*.60*scanRandom),headRandom=randomness("head_switch_randomness"),headGain=Math.max(.25,1+temporal(601,.73)*.70*headRandom),delayPx=Math.max(1,Math.round(delay*Math.max(3,width/78))),ringPx=Math.max(1,Math.round((1+resonance*6)*Math.max(1,width/960))),ghostPx=Math.max(2,Math.round((8+ghost*34)*Math.max(.5,width/720))),trailPx=Math.max(1,Math.round((2+trail*18)*Math.max(1,width/1920))),headStart=height-Math.max(1,Math.round(height*(.015+.075*head*headGain)));
    for(let y=0;y<height;y++){
        const fault=tracking>0&&proNoise(5,Math.floor(y/Math.max(1,height/18)),Math.floor(frame/3),seed+733)>(1-tracking*.22*(1+.55*trackingRandom))?Math.sin(y*.31+phase)*width*.055*tracking*(1+temporal(241,.67)*.65*trackingRandom):0;
        const rowWarp=Math.round(Math.sin(y*(.014+tapeWarp*.018)+warpedPhase)*width*.018*tapeWarp*warpGain+Math.sin(y*.17+phase*2.7+temporal(229,.48)*instabilityRandom)*width*.0045*instability*instabilityGain+proNoise(0,Math.floor(y/3),frame,seed+911)*width*.012*instability*(1+.75*instabilityRandom)+fault);
        const phaseShift=Math.round(proNoise(7,Math.floor(y/2),frame,seed+1211)*phaseNoise*(isTape?3:6));
        const loseChroma=chromaLoss>0&&proNoise(11,y,Math.floor(frame/2),seed+1337)>(1-chromaLoss*(isTape?.13:.24));
        const isDrop=drop>0&&proNoise(3,y,Math.floor(frame/2),seed+1771)>(1-drop*(isTape?.16:.24)),dropStart=Math.floor((proNoise(9,y,frame,seed+1913)*.5+.5)*width),dropLength=Math.floor(width*(.08+.55*drop));
        for(let x=0;x<width;x++){
            const out=(y*width+x)*4,srcY=(y-rollY+height)%height,headShift=y>=headStart?Math.round((height-y)/Math.max(1,height-headStart)*width*.065*head):0,srcX=x-rowWarp-headShift;
            const base=[sample(srcX,srcY,0),sample(srcX,srcY,1),sample(srcX,srcY,2)],grey=base[0]*.2126+base[1]*.7152+base[2]*.0722;
            const chromaShift=[sample(srcX-delayPx-phaseShift,srcY,0),sample(srcX-delayPx,srcY,1),sample(srcX-delayPx+phaseShift,srcY,2)];
            const spread=Math.max(2,Math.round(2+spill*8)),blur=[0,1,2].map((c)=>(sample(srcX-spread,srcY,c)+sample(srcX,srcY,c)+sample(srcX+spread,srcY,c))/3);
            const driftGain=[1+Math.sin(driftPhase)*.075*drift*driftStrength,1+Math.sin(driftPhase+2.1)*.045*drift*driftStrength,1+Math.sin(driftPhase+4.2)*.085*drift*driftStrength];
            for(let c=0;c<3;c++){
                let value=grey+(base[c]-grey)*saturation;
                value=value*(1-.34*blackLift)+blackLift*19.125;
                value+=((blur[c]-grey)-(base[c]-grey))*spill;
                value+=(chromaShift[c]-base[c])*delay;
                if(loseChroma)value=grey+(value-grey)*(isTape?.08:0);
                const edge=sample(srcX-ringPx,srcY,c)-sample(srcX-ringPx*2,srcY,c);
                value+=edge*resonance*(isTape?.12:.22);
                value+=sample(srcX-ghostPx,srcY,c)*ghost*.22+sample(srcX-ghostPx*2,srcY,c)*ghost*.08-value*ghost*.18;
                value+=(sample(srcX-trailPx,srcY,c)+sample(srcX-trailPx*2,srcY,c)*.6-value*1.6)/2.6*trail;
                value*=driftGain[c];
                value+=proNoise(Math.floor(x/(isTape?3:1)),y,frame,seed+23)*(isTape?24:33)*lumaNoise;
                value+=proNoise(Math.floor(x/(isTape?4:1))+c*139,Math.floor(y/(isTape?2:1)),frame,seed+47+c*13)*(isTape?25:41)*chromaNoise*(c===1?-.5:1);
                if(interlace>0&&(y&1))value=value*(1-.20*interlace*interlaceGain)+sample(srcX+interlaceDirection,srcY,c)*.08*interlace*interlaceGain;
                value*=1-(.5+.5*Math.cos(y*Math.PI+scanPhase))*scan*scanDepth;
                if(glow>0){const bright=Math.max(0,(sample(srcX-2,srcY,c)+sample(srcX+2,srcY,c)+sample(srcX,srcY-2,c)+sample(srcX,srcY+2,c))/4-140);value+=bright*.42*glow;}
                if(isDrop&&x>=dropStart&&x<dropStart+dropLength)value=value*(1-(.28+.55*drop))+((proNoise(x,y,frame,seed+2903)*.5+.5)*255)*(.28+.55*drop);
                if(y>=headStart)value+=proNoise(x,y,frame,seed+3301)*14*head*headGain;
                data[out+c]=originalSource[out+c]*(1-mix)+Math.max(0,Math.min(255,value))*mix;
            }
        }
    }
}

function gaussianBlurPreview(data,width,height,sigma){
    sigma=Math.max(.01,Number(sigma));const radius=Math.max(1,Math.min(18,Math.ceil(sigma*3))),kernel=new Float32Array(radius*2+1);let total=0;
    for(let offset=-radius;offset<=radius;offset++){const weight=Math.exp(-.5*(offset/sigma)*(offset/sigma));kernel[offset+radius]=weight;total+=weight;}
    for(let i=0;i<kernel.length;i++)kernel[i]/=total;
    const horizontal=new Float32Array(data.length),result=new Float32Array(data.length);
    for(let y=0;y<height;y++)for(let x=0;x<width;x++){const out=(y*width+x)*4;for(let c=0;c<3;c++){let value=0;for(let offset=-radius;offset<=radius;offset++){const sx=Math.max(0,Math.min(width-1,x+offset));value+=data[(y*width+sx)*4+c]*kernel[offset+radius];}horizontal[out+c]=value;}horizontal[out+3]=255;}
    for(let y=0;y<height;y++)for(let x=0;x<width;x++){const out=(y*width+x)*4;for(let c=0;c<3;c++){let value=0;for(let offset=-radius;offset<=radius;offset++){const sy=Math.max(0,Math.min(height-1,y+offset));value+=horizontal[(sy*width+x)*4+c]*kernel[offset+radius];}result[out+c]=value;}result[out+3]=255;}
    return result;
}
function applyLensPreview(data,width,height,values){
    const amount=Math.max(0,Math.min(2,Number(values.lens_master||0)));if(amount<=0)return;
    const source=new Uint8ClampedArray(data),aspect=width/Math.max(1,height),zoom=Math.max(.70,Math.min(1.60,Number(values.lens_zoom||1))),distortion=Math.max(-1,Math.min(1,Number(values.lens_distortion||0)))*amount,edge=Math.max(-1,Math.min(1,Number(values.lens_edge_stretch||0)))*amount,anamorphic=Math.max(-.5,Math.min(.5,Number(values.lens_anamorphic_width||0)))*amount,kx=Math.max(-.75,Math.min(.75,Number(values.lens_keystone_x||0)))*amount,ky=Math.max(-.75,Math.min(.75,Number(values.lens_keystone_y||0)))*amount,aberration=Math.max(0,Math.min(1,Number(values.lens_chromatic_aberration||0)))*amount,vignette=Math.max(0,Math.min(1,Number(values.lens_vignette||0)))*amount,tiltAmount=Math.max(0,Math.min(2,Number(values.lens_tilt_blur||0)*amount)),focusStrength=Math.min(1,tiltAmount),tiltAngle=Number(values.lens_tilt_angle||0)*Math.PI/180,focusPosition=Math.max(-1,Math.min(1,Number(values.lens_focus_position||0))),focusBand=.16+.20*(1-focusStrength),cocMap=tiltAmount>0?new Float32Array(width*height):null;
    const sample=(px,py,c)=>{px=Math.max(0,Math.min(width-1,px));py=Math.max(0,Math.min(height-1,py));const x0=Math.floor(px),y0=Math.floor(py),x1=Math.min(width-1,x0+1),y1=Math.min(height-1,y0+1),fx=px-x0,fy=py-y0,a=source[(y0*width+x0)*4+c],b=source[(y0*width+x1)*4+c],d=source[(y1*width+x0)*4+c],e=source[(y1*width+x1)*4+c];return (a+(b-a)*fx)*(1-fy)+(d+(e-d)*fx)*fy;};
    for(let y=0;y<height;y++)for(let x=0;x<width;x++){
        const nx=width>1?x/(width-1)*2-1:0,ny=height>1?y/(height-1)*2-1:0,oy=ny/aspect,r2=nx*nx+oy*oy,radial=1+distortion*(.34*r2+.16*r2*r2)+edge*.16*Math.pow(r2,1.5);
        let sx=nx*radial,sy=oy*radial;sx*=1-anamorphic*.48;sx*=1+anamorphic*.22*sy*sy;const denominator=Math.max(.42,1+kx*sx*.55+ky*sy*.72);sx=sx/denominator/zoom;sy=sy/denominator*aspect/zoom;
        const baseX=(sx+1)*.5*(width-1),baseY=(sy+1)*.5*(height-1),ca=aberration*.0065*r2*(width-1)*.5,falloff=1-vignette*.62*Math.pow(Math.min(1.25,Math.sqrt(r2)),2.35),out=(y*width+x)*4;
        for(let c=0;c<3;c++){const channelX=baseX+(c===0?ca:c===2?-ca:0);data[out+c]=Math.max(0,Math.min(255,sample(channelX,baseY,c)*falloff));}
        if(cocMap){let coc=Math.max(0,Math.min(1,(Math.abs(ny*Math.cos(tiltAngle)+nx*Math.sin(tiltAngle)-focusPosition)-focusBand)/Math.max(.08,1-focusBand)));coc=coc*coc*(3-2*coc);cocMap[y*width+x]=coc;}
    }
    if(cocMap){const maximumSigma=Math.max(.85,Math.min(width,height)*(.004+.020*focusStrength)*tiltAmount),medium=gaussianBlurPreview(data,width,height,maximumSigma*.38),broad=gaussianBlurPreview(data,width,height,maximumSigma);for(let pixel=0;pixel<width*height;pixel++){const coc=cocMap[pixel],blend=Math.max(0,Math.min(1,(coc-.32)/.68)),broadWeight=blend*blend*(3-2*blend),amount=coc*focusStrength,index=pixel*4;for(let c=0;c<3;c++){const defocused=medium[index+c]+(broad[index+c]-medium[index+c])*broadWeight;data[index+c]+= (defocused-data[index+c])*amount;}}}
}
function mountPremiumUI(node) {
    if (node._iamccsProPremium) return;
    node._iamccsProPremium = true;
    PRO_FIELDS.forEach((name) => hideWidget(findWidget(node, name)));
    const root = document.createElement("div");
    root.className = "iamccs-pro-root";
    root.innerHTML = [
      "<style>.iamccs-pro-root{width:100%;height:100%;padding:10px;box-sizing:border-box;border:1px solid #655237;border-radius:12px;background:linear-gradient(140deg,#0b1015,#172029);color:#e8edf1;font:10px Inter,Segoe UI,sans-serif;overflow:hidden}.iamccs-pro-root *{box-sizing:border-box}.pro-head{height:34px;display:flex;align-items:center;gap:9px;border-bottom:1px solid #33404a}.pro-chip{padding:4px 8px;border:1px solid #c3984e;border-radius:99px;background:#312718;color:#f4d18c;font-size:8px;font-weight:900}.pro-title{font:700 14px Georgia,serif}.pro-live{margin-left:auto;color:#7bdba6;font-size:8px;font-weight:900}.pro-layout{display:grid;grid-template-columns:320px minmax(430px,1fr);gap:10px;height:calc(100% - 42px);padding-top:8px}.pro-controls{overflow:auto;padding-right:4px}.pro-selects{display:grid;grid-template-columns:1fr 1fr;gap:6px}.pro-field{display:grid;gap:3px;color:#94a1ac;font-size:7px;font-weight:900}.pro-wide{grid-column:1/-1}.pro-field select,.pro-field input{width:100%;height:28px;border:1px solid #45525d;border-radius:6px;background:#101820;color:#e8edf1;padding:0 7px;font-size:9px}.pro-sliders{display:grid;grid-template-columns:1fr 1fr;gap:9px 11px;margin-top:8px;padding:10px;border:1px solid #34424c;border-radius:11px;background:linear-gradient(145deg,#111a22,#0c1218);box-shadow:inset 0 1px 0 rgba(255,255,255,.025),0 8px 22px rgba(0,0,0,.18)}.pro-slider{position:relative;display:grid;grid-template-columns:1fr auto;align-items:center;gap:7px;padding:8px 9px 9px;border:1px solid #293640;border-radius:8px;background:linear-gradient(145deg,rgba(31,42,51,.88),rgba(15,22,28,.94));color:#aeb9c1;font-size:7px;font-weight:900;letter-spacing:.35px;transition:border-color .16s,background .16s,box-shadow .16s}.pro-slider:hover{border-color:#6e5b3c;background:linear-gradient(145deg,#26323b,#141d24);box-shadow:0 4px 13px rgba(0,0,0,.25)}.pro-slider:focus-within{border-color:#d1a451;box-shadow:0 0 0 2px rgba(209,164,81,.14),0 5px 16px rgba(0,0,0,.3)}.pro-slider output{min-width:44px;padding:3px 7px;border:1px solid #725a31;border-radius:999px;background:linear-gradient(180deg,#3b2d17,#241b10);color:#ffdc92;font:800 9px Consolas,monospace;text-align:center;box-shadow:inset 0 1px 0 rgba(255,232,180,.12)}.pro-slider input[type=range]{--fill:50%;grid-column:1/-1;width:100%;height:18px;margin:0;padding:0;appearance:none;-webkit-appearance:none;background:transparent;cursor:pointer;outline:none}.pro-slider input[type=range]::-webkit-slider-runnable-track{height:6px;border:1px solid #34414a;border-radius:999px;background:linear-gradient(90deg,#856026 0%,#e1b65d var(--fill),#27323a var(--fill),#1a2229 100%);box-shadow:inset 0 2px 3px rgba(0,0,0,.58),0 1px 0 rgba(255,255,255,.04)}.pro-slider input[type=range]::-webkit-slider-thumb{width:18px;height:18px;margin-top:-7px;border:2px solid #f1ce81;border-radius:50%;appearance:none;-webkit-appearance:none;background:radial-gradient(circle at 35% 30%,#fff0bd 0 14%,#d6a84e 28%,#76501f 72%,#2d2112 100%);box-shadow:0 0 0 3px rgba(209,164,81,.13),0 3px 8px rgba(0,0,0,.75);transition:transform .12s,box-shadow .12s}.pro-slider input[type=range]:hover::-webkit-slider-thumb{transform:scale(1.1);box-shadow:0 0 0 5px rgba(209,164,81,.16),0 4px 10px rgba(0,0,0,.8)}.pro-slider input[type=range]:active::-webkit-slider-thumb{transform:scale(.96);box-shadow:0 0 0 7px rgba(209,164,81,.2),0 2px 7px rgba(0,0,0,.7)}.pro-slider input[type=range]::-moz-range-track{height:6px;border:1px solid #34414a;border-radius:999px;background:#27323a}.pro-slider input[type=range]::-moz-range-progress{height:6px;border-radius:999px;background:linear-gradient(90deg,#856026,#e1b65d)}.pro-slider input[type=range]::-moz-range-thumb{width:15px;height:15px;border:2px solid #f1ce81;border-radius:50%;background:#c9963d;box-shadow:0 0 0 3px rgba(209,164,81,.13),0 3px 8px rgba(0,0,0,.75)}.pro-monitor{display:grid;grid-template-rows:minmax(250px,1fr) auto auto;gap:6px}.pro-screen{position:relative;min-height:0;border:1px solid #59656f;border-radius:9px;background:#050709;overflow:hidden}.pro-screen canvas{width:100%;height:100%;object-fit:contain;image-rendering:auto}.pro-hint{position:absolute;inset:0;display:grid;place-items:center;text-align:center;color:#aab4bd;pointer-events:none}.pro-tools{display:grid;grid-template-columns:repeat(5,auto) minmax(90px,1fr);gap:5px}.pro-btn,.pro-view{height:29px;border:1px solid #485560;border-radius:6px;background:#202a33;color:#e8edf1;padding:0 9px;font-size:8px;font-weight:900;cursor:pointer}.pro-gold{background:#8c682a;border-color:#d9ab57;color:#fff1ca}.pro-file{position:absolute;width:1px;height:1px;opacity:0}.pro-status{padding:6px 8px;border-left:2px solid #c99a4c;background:#111920;color:#aab6bf}.pro-hd{position:fixed;inset:0;z-index:100000;display:none;padding:20px;background:rgba(2,5,8,.97)}.pro-hd.open{display:grid;grid-template-columns:1fr 240px;gap:14px}.pro-hd canvas{width:100%;height:100%;object-fit:contain;background:#000;border:1px solid #59656f;border-radius:10px}.pro-side{display:flex;flex-direction:column;gap:9px}.pro-side h2{color:#f0d39b;font:700 18px Georgia,serif}.pro-side p{color:#95a3ae;line-height:1.5}.pro-presets{position:fixed;inset:0;z-index:100001;display:none;padding:18px;background:rgba(2,5,8,.97);overflow:auto}.pro-presets.open{display:block}.pro-preset-head{position:sticky;top:-18px;z-index:2;display:flex;align-items:center;gap:14px;padding:12px 0;background:#05090d}.pro-preset-head h2{margin:0;color:#f0d39b;font:700 22px Georgia,serif}.pro-preset-head p{margin:0;color:#9ba8b2}.pro-preset-head button{margin-left:auto}.pro-preset-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:9px}.pro-preset-card{padding:6px;border:1px solid #45525d;border-radius:8px;background:#111920;color:#e8edf1;text-align:left;cursor:pointer}.pro-preset-card:hover,.pro-preset-card.active{border-color:#d9ab57;box-shadow:0 0 0 1px #8c682a}.pro-preset-card canvas{display:block;width:100%;aspect-ratio:16/9;border-radius:5px;background:#000}.pro-preset-card strong{display:block;padding:6px 2px 1px;color:#f0cc83;font-size:9px}.pro-preset-card span{display:block;padding:0 2px 2px;color:#8fa0ac;font-size:8px}</style>",
      "<style>.pro-slider.pro-master{grid-column:1/-1;border-color:#b7883e;background:linear-gradient(145deg,#3c2d17,#171b1e);color:#ffe0a0;font-size:9px;box-shadow:inset 0 1px 0 rgba(255,225,165,.12),0 6px 18px rgba(0,0,0,.28)}.pro-slider.pro-master output{border-color:#e0b45c;background:linear-gradient(180deg,#60471f,#34250f);color:#fff0c4}.pro-slider.pro-lens-master{grid-column:1/-1;border-color:#547a91;background:linear-gradient(145deg,#193545,#131d23);color:#bce9ff;font-size:9px}.pro-slider.pro-lens-master output{border-color:#6295ad;background:linear-gradient(180deg,#244c61,#152d39);color:#d7f4ff}.pro-slider.pro-randomness{border-color:#3f5d73;background:linear-gradient(145deg,#172a37,#101a22);color:#a9dcf4}.pro-slider.pro-randomness output{border-color:#507890;background:linear-gradient(180deg,#24475a,#142c39);color:#ccefff}</style><div class='pro-head'><span class='pro-chip'>CINE POST V2</span><span class='pro-title'>IAMCCS-CinePostEfx-v2 · Film / Sensor / Lens / Tape / Broadcast</span><span class='pro-live'>● QUEUE TRUTH</span></div>",
      "<div class='pro-layout'><section class='pro-controls'><div class='pro-selects'><label class='pro-field pro-wide'>PRESET<select data-select='preset'></select></label><label class='pro-field'>ENGINE<select data-select='engine'></select></label><label class='pro-field'>TRANSFER<select data-select='input_transfer'></select></label><label class='pro-field'>GRAIN PROFILE<select data-select='grain_profile'></select></label><label class='pro-field'>ANALOG LUT / LOOK<select data-select='analog_color_look'></select></label><label class='pro-field pro-wide'>LENS PROFILE<select data-select='lens_preset'></select></label><label class='pro-field pro-wide'>BLEND<select data-select='blend_method'></select></label></div><div class='pro-sliders'></div><div class='pro-selects' style='margin-top:7px'><label class='pro-field'>SEED<input data-number='seed' type='number'></label><label class='pro-field'>FRAME START<input data-number='frame_start' type='number'></label></div></section>",
      "<section class='pro-monitor'><div class='pro-screen'><canvas width='288' height='162'></canvas><div class='pro-hint'>ANALOG TV TEST CARD<br>Open or drop local media for a private preview.</div></div><div class='pro-tools'><button class='pro-btn pro-gold' data-action='open'>OPEN MEDIA</button><button class='pro-btn' data-action='plate'>OPEN PLATE</button><button class='pro-btn pro-gold' data-action='presets'>PRESET PREVIEWS</button><button class='pro-btn' data-action='hd'>OPEN HD</button><button class='pro-btn' data-action='clear'>CLEAR</button><select class='pro-view'><option value='grain'>CINE POST</option><option value='split'>SPLIT</option><option value='original'>ORIGINAL</option><option value='map'>GRAIN MAP</option></select><input class='pro-file pro-media-file' type='file' accept='image/*,video/*'><input class='pro-file pro-plate-file' type='file' accept='image/*'></div><div class='pro-status'>Local preview only. Connected IMAGE and grain_plate remain Queue truth.</div></section></div>",
      "<style>.pro-hd.open{grid-template-columns:minmax(0,1fr) 340px}.pro-side{min-height:0}.pro-hd-settings{min-height:0;overflow:auto;padding-right:4px}.pro-hd-settings .pro-sliders{grid-template-columns:1fr}.pro-save{margin-top:auto}</style><div class='pro-hd'><canvas width='384' height='216'></canvas><aside class='pro-side'><h2>IAMCCS-CinePostEfx-v2 · HD</h2><p>All controls below stay synchronized with Queue truth.</p><div class='pro-hd-settings'></div><button class='pro-btn pro-gold pro-save' data-action='save'>SAVE SETTINGS IN NODE</button><button class='pro-btn pro-gold' data-action='hd-open'>OPEN MEDIA</button><button class='pro-btn' data-action='hd-plate'>OPEN PLATE</button><button class='pro-btn' data-action='hd-close'>CLOSE HD</button></aside></div><div class='pro-presets'><div class='pro-preset-head'><div><h2>Preset previews</h2><p>Post-process and optical lens previews on an analogue test card. Click to apply.</p></div><button class='pro-btn pro-gold' data-action='presets-close'>CLOSE</button></div><div class='pro-preset-grid'></div></div>"
    ].join("");
    const q = (selector) => root.querySelector(selector);
    const controls = new Map();
    const markCustom = () => setWidget(node, "preset", "custom_box_values");
    const markControlCustom=(name)=>name.startsWith("lens_")?setWidget(node,"lens_preset","custom_lens"):markCustom();
    const addSelect = (name, values) => {
        const select = q("[data-select='" + name + "']");
        values.forEach((value) => select.add(new Option(value.replaceAll("_", " ").toUpperCase(), value)));
        select.value = String(readWidget(node, name, values[0]));
        select.onchange = () => { if(name==="lens_preset"){setWidget(node,name,select.value);applyLensPreset(node,select.value);}else{if(name!=="preset")markCustom();setWidget(node,name,select.value);}sync(); };
        controls.set(name, select);
    };
    addSelect("preset", [...Object.keys(PRESETS), "custom_box_values"]);
    addSelect("engine", ["film_emulsion", "digital_sensor", "scanned_grain_plate", "analog_tape", "broadcast_signal"]);
    addSelect("input_transfer", ["srgb", "rec709", "linear"]);
    addSelect("blend_method", ["log_density", "density_exposure", "linear_additive", "soft_light_luma"]);
    addSelect("grain_profile", ["modern_fine", "negative_stock", "high_speed_negative", "reversal_stock", "print_stock"]);
    addSelect("analog_color_look", ["neutral", "consumer_color", "warm_camcorder", "cool_camcorder", "tungsten_home_video", "fluorescent_cctv", "sodium_vapor_cctv", "late_night_blue", "sun_bleached_tape", "rf_cyan_fade", "archival_amber", "faded_magenta", "crt_broadcast", "night_vision_green"]);
    addSelect("lens_preset", [...Object.keys(LENS_PRESET_VALUES), "custom_lens"]);
    const paintSlider = (input) => {
        if (!input) return;
        const minimum=Number(input.min||0),maximum=Number(input.max||1),current=Number(input.value||0);
        const fill=maximum>minimum?Math.max(0,Math.min(100,(current-minimum)/(maximum-minimum)*100)):0;
        input.style.setProperty("--fill",fill.toFixed(2)+"%");
    };
    let sliderInteracting=false;
    const defs = [["disturbance_amount","MASTER DISTURBANCE",0,2,.01,2],["strength","GRAIN STRENGTH",0,1,.005,3],["grain_size_4k_px","4K SIZE PX",.35,6,.05,2],["softness","SOFTNESS",0,1,.01,2],["roughness","ROUGHNESS",0,1,.01,2],["complexity","OCTAVES",1,4,1,0],["texture_microcontrast","TEXTURE MICRO-CONTRAST",0,1,.01,2],["temporal_correlation","TEMPORAL",0,.95,.01,2],["chroma_amount","GRAIN CHROMA",0,.5,.005,3],["shadow_response","SHADOWS",0,2,.02,2],["midtone_response","MIDTONES",0,2,.02,2],["highlight_response","HIGHLIGHTS",0,2,.02,2],["red_response","RED",.5,1.5,.01,2],["green_response","GREEN",.5,1.5,.01,2],["blue_response","BLUE",.5,1.5,.01,2],["horizontal_instability","HORIZONTAL INSTABILITY",0,1,.01,2],["horizontal_instability_randomness","HORIZONTAL RANDOMNESS",0,1,.01,2],["tracking_error","TRACKING ERROR",0,1,.01,2],["tracking_randomness","TRACKING RANDOMNESS",0,1,.01,2],["tape_warp","TAPE WARP",0,1,.01,2],["tape_warp_randomness","TAPE WARP RANDOMNESS",0,1,.01,2],["chroma_spill","CHROMA SPILL",0,1,.01,2],["chroma_delay","CHROMA DELAY",0,1,.01,2],["chroma_phase_noise","CHROMA PHASE NOISE",0,1,.01,2],["chroma_loss","CHROMA LOSS",0,1,.01,2],["signal_saturation","SIGNAL SATURATION",0,2,.01,2],["black_lift","BLACK LIFT",0,1,.01,2],["luma_noise","LUMA SIGNAL NOISE",0,1,.01,2],["chroma_noise","CHROMA SIGNAL NOISE",0,1,.01,2],["color_drift","COLOR DRIFT",0,1,.01,2],["color_drift_randomness","COLOR DRIFT RANDOMNESS",0,1,.01,2],["edge_echo","EDGE RESONANCE",0,1,.01,2],["ghost_echo","DELAYED GHOST",0,1,.01,2],["line_dropout","LINE DROPOUT",0,1,.01,2],["scanline_strength","SCANLINES",0,1,.01,2],["scanline_randomness","SCANLINE RANDOMNESS",0,1,.01,2],["field_interlace","FIELD INTERLACE",0,1,.01,2],["interlace_randomness","INTERLACE RANDOMNESS",0,1,.01,2],["vertical_roll","VERTICAL ROLL",0,1,.01,2],["vertical_roll_randomness","VERTICAL ROLL RANDOMNESS",0,1,.01,2],["highlight_glow","HIGHLIGHT GLOW",0,1,.01,2],["head_switch_distortion","HEAD-SWITCH DISTORTION",0,1,.01,2],["head_switch_randomness","HEAD-SWITCH RANDOMNESS",0,1,.01,2],["luma_trail","LUMA TRAIL",0,1,.01,2],["analog_mix","ANALOG MIX",0,1,.01,2]];
    defs.unshift(
        ["lens_master","MASTER LENS GEOMETRY",0,2,.01,2],
        ["lens_distortion","LENS BARREL / PINCUSHION",-1,1,.01,2],
        ["lens_edge_stretch","PERIPHERAL STRETCH",-1,1,.01,2],
        ["lens_anamorphic_width","ANAMORPHIC WIDTH",-.5,.5,.01,2],
        ["lens_zoom","LENS OVERSCAN / ZOOM",.70,1.60,.01,2],
        ["lens_keystone_x","SHIFT / KEYSTONE X",-.75,.75,.01,2],
        ["lens_keystone_y","SHIFT / KEYSTONE Y",-.75,.75,.01,2],
        ["lens_tilt_angle","TILT PLANE ANGLE",-90,90,1,0],
        ["lens_focus_position","TILT FOCUS POSITION",-1,1,.01,2],
        ["lens_tilt_blur","TILT DEFOCUS",0,1,.01,2],
        ["lens_chromatic_aberration","LENS CHROMATIC ABERRATION",0,1,.01,2],
        ["lens_vignette","OPTICAL VIGNETTE",0,1,.01,2],
    );
    defs.forEach(([name, label, min, max, step, digits]) => {
        const row = document.createElement("label"), output = document.createElement("output"), input = document.createElement("input");
        row.className = "pro-slider"+(name==="disturbance_amount"?" pro-master":name==="lens_master"?" pro-lens-master":name.endsWith("_randomness")?" pro-randomness":""); row.append(document.createTextNode(label), output); input.type = "range";
        Object.assign(input, { min, max, step, value: readWidget(node, name, min) }); input.dataset.name = name; input.dataset.digits = String(digits); input.title=name.endsWith("_randomness")?label+" — 0 REGULAR / 1 IRREGULAR":label;
        output.value = Number(input.value).toFixed(digits);
        paintSlider(input);
        input.oninput = () => { const value = digits ? Number(input.value) : Math.round(Number(input.value)); output.value = value.toFixed(digits); paintSlider(input); markControlCustom(name); setWidget(node, name, value); };
        input.onpointerdown=()=>{sliderInteracting=true;};
        input.onpointerup=input.onpointercancel=()=>{sliderInteracting=false;scheduleRender(true);};
        row.append(input); q(".pro-sliders").append(row); controls.set(name, { input, output, digits });
    });
    ["seed", "frame_start"].forEach((name) => { const input=q("[data-number='"+name+"']"); input.value=readWidget(node,name,0); input.onchange=()=>setWidget(node,name,Math.max(0,Math.trunc(Number(input.value)||0))); controls.set(name,input); });
    function sync() {
        controls.forEach((control,name) => { const value=readWidget(node,name,control.value); if(control.input){control.input.value=value;control.output.value=Number(value).toFixed(control.digits);paintSlider(control.input);}else control.value=value; });
        root.querySelectorAll(".pro-hd-settings [data-select]").forEach((el) => { el.value=String(readWidget(node,el.dataset.select,el.value)); });
        root.querySelectorAll(".pro-hd-settings input[data-name]").forEach((el) => { el.value=String(readWidget(node,el.dataset.name,el.value));paintSlider(el); const out=el.parentElement?.querySelector("output"); if(out)out.value=Number(el.value).toFixed(Number(el.dataset.digits||2)); });
        root.querySelectorAll(".pro-hd-settings [data-number]").forEach((el) => { el.value=String(readWidget(node,el.dataset.number,el.value)); });
    }
    function writeSettingsSnapshot(){
        const target=findWidget(node,"settings_json");if(!target)return;
        const snapshot={};SNAPSHOT_FIELDS.forEach((name)=>snapshot[name]=readWidget(node,name,null));
        const encoded=JSON.stringify(snapshot);if(target.value!==encoded)target.value=encoded;
    }
    node._iamccsWriteProSnapshot=writeSettingsSnapshot;
    const settingsSnapshotWidget=findWidget(node,"settings_json");
    if(settingsSnapshotWidget)settingsSnapshotWidget.serializeValue=()=>{writeSettingsSnapshot();return settingsSnapshotWidget.value;};
    q("[data-select='preset']").onchange = () => { const name=q("[data-select='preset']").value; setWidget(node,"preset",name); applyPreset(node,name); sync(); };
    const video=document.createElement("video"), image=new Image(), plateImage=new Image(); video.muted=true; video.loop=true; video.playsInline=true;
    const tonal=document.createElement("canvas"), tonalContext=tonal.getContext("2d"); tonal.width=640; tonal.height=360;
    function drawAnalogTestCard(context,width,height){
        context.fillStyle="#16191c";context.fillRect(0,0,width,height);
        context.strokeStyle="#3a4147";context.lineWidth=1;
        for(let x=0;x<=width;x+=32){context.beginPath();context.moveTo(x,0);context.lineTo(x,height);context.stroke();}
        for(let y=0;y<=height;y+=30){context.beginPath();context.moveTo(0,y);context.lineTo(width,y);context.stroke();}
        const bars=["#eeeae0","#e8dc19","#14d8dc","#19cf35","#d91bd5","#dc2920","#2635da","#090b0e"];
        bars.forEach((color,index)=>{context.fillStyle=color;context.fillRect(42+index*69,45,70,128);});
        const greys=["#050505","#242424","#484848","#707070","#989898","#c4c4c4","#ededed"];
        greys.forEach((color,index)=>{context.fillStyle=color;context.fillRect(76+index*70,190,71,42);});
        context.fillStyle="#ece7db";context.fillRect(76,250,488,58);
        for(let x=76;x<564;x++){const band=Math.floor((x-76)/61);const period=Math.max(2,18-band*2);context.fillStyle=(Math.floor((x-76)/period)&1)?"#171717":"#e7e3da";context.fillRect(x,250,1,58);}
        context.strokeStyle="rgba(250,245,225,.82)";context.lineWidth=3;context.strokeRect(18,18,width-36,height-36);
        context.beginPath();context.arc(width/2,height/2,137,0,Math.PI*2);context.stroke();
        context.lineWidth=1;context.beginPath();context.moveTo(width/2,18);context.lineTo(width/2,height-18);context.moveTo(18,height/2);context.lineTo(width-18,height/2);context.stroke();
        context.fillStyle="#101317";context.fillRect(width/2-103,10,206,26);context.fillStyle="#f2ddb0";context.font="700 14px Segoe UI, sans-serif";context.textAlign="center";context.fillText("IAMCCS · ANALOG TEST",width/2,28);
    }
    drawAnalogTestCard(tonalContext,tonal.width,tonal.height);
    let url="", plateUrl="", source=null, frame=0;
    const values=()=>Object.fromEntries(PRO_FIELDS.map((name)=>[name,readWidget(node,name,null)]));
    const mediaSource=()=>source==="video"&&video.readyState>=2?video:source==="image"&&image.complete?image:tonal;
    function draw(canvas, overrideValues=null, forcedView=null) {
        const ctx=canvas.getContext("2d",{alpha:false,willReadFrequently:true}),W=canvas.width,H=canvas.height,media=mediaSource();
        ctx.fillStyle="#050709";ctx.fillRect(0,0,W,H);
        const sw=media.videoWidth||media.naturalWidth||media.width,sh=media.videoHeight||media.naturalHeight||media.height,scale=Math.min(W/sw,H/sh),dw=sw*scale,dh=sh*scale;
        ctx.drawImage(media,(W-dw)/2,(H-dh)/2,dw,dh);
        const original=ctx.getImageData(0,0,W,H),output=new ImageData(new Uint8ClampedArray(original.data),W,H),data=output.data,v=overrideValues||values(),view=forcedView||q(".pro-view").value;
        if(view!=="map")applyLensPreview(data,W,H,v);
        const sourcePixels=new Uint8ClampedArray(data),sourceSample=(x,y,c)=>sourcePixels[(Math.max(0,Math.min(H-1,y))*W+Math.max(0,Math.min(W-1,x)))*4+c]/255;
        let plateData=null;
        if(v.engine==="scanned_grain_plate"&&plateImage.complete&&plateImage.naturalWidth){const pc=document.createElement("canvas"),px=pc.getContext("2d");pc.width=W;pc.height=H;px.drawImage(plateImage,0,0,W,H);plateData=px.getImageData(0,0,W,H).data;}
        const profileShapes={modern_fine:[.82,.88],negative_stock:[1,1],high_speed_negative:[1.22,1.08],reversal_stock:[1.32,1.04],print_stock:[.92,.86]},profileShape=profileShapes[v.grain_profile]||profileShapes.negative_stock,profileTail=profileShape[0],profileGain=profileShape[1];
        const grainSize=Math.max(.6,Number(v.grain_size_4k_px||1)*Math.max(W,H)/1024),octaves=Math.max(1,Math.min(4,Math.round(Number(v.complexity||3)))),rough=Number(v.roughness||0),soft=Number(v.softness||0),temporal=Number(v.temporal_correlation||0),fresh=Math.sqrt(Math.max(0,1-temporal*temporal)),chroma=Number(v.chroma_amount||0),responses=[Number(v.red_response||1),Number(v.green_response||1),Number(v.blue_response||1)],sqrt3=Math.sqrt(3);
        const field=(x,y,f,salt=0)=>{let sum=0,norm=0;for(let o=0;o<octaves;o++){const cell=grainSize*Math.pow(2.12,o),nx=Math.floor(x/cell),ny=Math.floor(y/cell),weight=o===0?1:(.14+rough*.30)/o;let n=proNoise(nx,ny,f,Number(v.seed||1)+salt);if(soft>0){const around=(proNoise(nx+1,ny,f,Number(v.seed||1)+salt)+proNoise(nx,ny+1,f,Number(v.seed||1)+salt)+proNoise(nx+1,ny+1,f,Number(v.seed||1)+salt))/3;n=n*(1-soft*.62)+around*soft*.62;}sum+=n*weight;norm+=weight;}let n=sum/Math.max(.001,norm)*sqrt3;return Math.sign(n)*Math.pow(Math.abs(n),1+rough*.30*profileTail)*profileGain;};
        for(let y=0;y<H;y++)for(let x=0;x<W;x++){
            const i=(y*W+x)*4,currentFrame=frame+Number(v.frame_start||0);
            let common=plateData?((plateData[i]+plateData[i+1]+plateData[i+2])/765-.5)*3:field(x,y,currentFrame)*fresh+field(x,y,Math.max(0,currentFrame-1))*temporal;
            if(view==="map"){const mapped=clamp01(.5+common/4)*255;data[i]=data[i+1]=data[i+2]=mapped;continue;}
            const linear=[decodeTransfer(data[i]/255,v.input_transfer),decodeTransfer(data[i+1]/255,v.input_transfer),decodeTransfer(data[i+2]/255,v.input_transfer)],micro=Number(v.texture_microcontrast||0)*(0.16+Number(v.strength||0)*.42);
            if(micro>0){for(let c=0;c<3;c++){const local=(sourceSample(x-1,y,c)+sourceSample(x+1,y,c)+sourceSample(x,y-1,c)+sourceSample(x,y+1,c))/4,detail=data[i+c]/255-local;linear[c]=Math.max(0,linear[c]+detail*micro*.42);}}
            const luma=linear[0]*.2126+linear[1]*.7152+linear[2]*.0722,sp=clamp01(luma/.58),hp=clamp01((luma-.42)/.58),swgt=1-sp*sp*(3-2*sp),hwgt=hp*hp*(3-2*hp),mwgt=clamp01(1-swgt-hwgt),tone=Math.max(0,Math.min(2,swgt*Number(v.shadow_response||0)+mwgt*Number(v.midtone_response||0)+hwgt*Number(v.highlight_response||0))),sigma=Number(v.strength||0)*.48*tone;
            for(let c=0;c<3;c++){const independent=proNoise(Math.floor(x/grainSize)+c*139,Math.floor(y/grainSize),currentFrame,Number(v.seed||1)+17+c*13)*sqrt3,noise=(common*(1-chroma)+independent*chroma)*responses[c],base=data[i+c]/255;let result;if(v.engine==="digital_sensor")result=encodeTransfer(linear[c]+noise*sigma*(Math.sqrt(Math.max(linear[c],1e-4))+.035+.12*rough)*.92,v.input_transfer);else if(v.blend_method==="log_density")result=encodeTransfer(Math.pow(2,-(-Math.log2(Math.max(linear[c],1e-5))-noise*sigma*.95)),v.input_transfer);else if(v.blend_method==="density_exposure")result=encodeTransfer(linear[c]*Math.exp(noise*sigma-.5*sigma*sigma),v.input_transfer);else if(v.blend_method==="linear_additive")result=encodeTransfer(linear[c]+noise*sigma*.62,v.input_transfer);else result=softLight(base,clamp01(.5+noise*sigma*1.9));data[i+c]=clamp01(result)*255;}
        }
        if(view!=="map")applyAnalogPreview(data,W,H,v,frame+Number(v.frame_start||0));
        ctx.putImageData(output,0,0);
        if(view==="split"){ctx.putImageData(original,0,0,0,0,Math.floor(W/2),H);ctx.fillStyle="#efc879";ctx.fillRect(Math.floor(W/2)-1,0,2,H);}else if(view==="original")ctx.putImageData(original,0,0);
    }
    function renderAll() { if(q(".pro-hd").classList.contains("open"))draw(q(".pro-hd canvas")); else draw(q(".pro-screen canvas")); }
    let renderQueued=false,lastRenderAt=0,renderTimer=0;
    function scheduleRender(force=false){
        if(renderQueued&&!force)return;
        if(force&&renderTimer){clearTimeout(renderTimer);renderTimer=0;renderQueued=false;}
        const now=performance.now(),interval=sliderInteracting?52:(source==="video"?33:16),delay=force?0:Math.max(0,interval-(now-lastRenderAt));
        renderQueued=true;
        const render=()=>requestAnimationFrame(()=>{renderQueued=false;renderTimer=0;lastRenderAt=performance.now();renderAll();});
        if(delay>4)renderTimer=setTimeout(render,delay);else render();
    }
    let previewRestoreVersion=0;
    function restorePreview(){
        const version=++previewRestoreVersion;
        [0,64,180].forEach((delay)=>{
            const redraw=()=>requestAnimationFrame(()=>{
                if(version!==previewRestoreVersion||!root.isConnected)return;
                const surface=q(".pro-hd").classList.contains("open")?q(".pro-hd canvas"):q(".pro-screen canvas"),rect=surface?.getBoundingClientRect?.();
                if(!rect||rect.width<2||rect.height<2)return;
                renderAll();
                node.graph?.setDirtyCanvas?.(true,true);
                app.canvas?.setDirty?.(true,true);
            });
            if(delay)setTimeout(redraw,delay);else redraw();
        });
    }
    const presetOverlay=q(".pro-presets"),presetGrid=q(".pro-preset-grid");
    let presetGalleryToken=0;
    const analogueReset={horizontal_instability:0,chroma_spill:0,luma_noise:0,chroma_noise:0,color_drift:0,edge_echo:0,line_dropout:0,scanline_strength:0,field_interlace:0,vertical_roll:0,highlight_glow:0,head_switch_distortion:0,luma_trail:0,chroma_delay:0,tracking_error:0,tape_warp:0,ghost_echo:0,chroma_phase_noise:0,chroma_loss:0,signal_saturation:1,black_lift:0,analog_mix:1,disturbance_amount:1};
    function closePresetGallery(){presetGalleryToken++;presetOverlay.classList.remove("open");}
    function openPresetGallery(){
        const token=++presetGalleryToken,snapshot=values(),entries=Object.entries(PRESETS),selected=String(snapshot.preset||"");
        presetGrid.replaceChildren();presetOverlay.classList.add("open");
        const cards=entries.map(([name,presetValues])=>{
            const card=document.createElement("button"),canvas=document.createElement("canvas"),strong=document.createElement("strong"),meta=document.createElement("span");
            card.type="button";card.className="pro-preset-card"+(name===selected?" active":"");card.style.contain="content";canvas.width=176;canvas.height=99;
            strong.textContent=name.replaceAll("_"," ").toUpperCase();
            const character=presetValues.analog_color_look&&presetValues.analog_color_look!=="neutral"?presetValues.analog_color_look:presetValues.grain_profile;
            meta.textContent=(String(presetValues.engine||"film_emulsion")+(character?" · "+character:"")).replaceAll("_"," ").toUpperCase();
            card.append(canvas,strong,meta);card.onclick=()=>{setWidget(node,"preset",name);applyPreset(node,name);sync();writeSettingsSnapshot();scheduleRender(true);closePresetGallery();};presetGrid.append(card);
            return {canvas,name,presetValues,kind:"post"};
        });
        Object.entries(LENS_PRESET_VALUES).filter(([name])=>name!=="lens_none").forEach(([name,lensValues])=>{
            const card=document.createElement("button"),canvas=document.createElement("canvas"),strong=document.createElement("strong"),meta=document.createElement("span");
            card.type="button";card.className="pro-preset-card"+(name===String(snapshot.lens_preset||"")?" active":"");card.style.contain="content";canvas.width=176;canvas.height=99;strong.textContent=name.replaceAll("_"," ").toUpperCase();meta.textContent="OPTICAL LENS · GEOMETRY";card.append(canvas,strong,meta);card.onclick=()=>{setWidget(node,"lens_preset",name);applyLensPreset(node,name);sync();writeSettingsSnapshot();scheduleRender(true);closePresetGallery();};presetGrid.append(card);cards.push({canvas,name,presetValues:lensValues,kind:"lens"});
        });
        const queueIdle=(callback)=>typeof requestIdleCallback==="function"?requestIdleCallback(callback,{timeout:90}):setTimeout(callback,18);
        const renderCard=(index)=>{
            if(token!==presetGalleryToken||!presetOverlay.classList.contains("open")||index>=cards.length)return;
            const {canvas,name,presetValues,kind}=cards[index];
            const previewValues=kind==="lens"?{...snapshot,...analogueReset,strength:0,analog_color_look:"neutral",...presetValues,lens_preset:name}:{...snapshot,...analogueReset,...presetValues,preset:name};
            draw(canvas,previewValues,"grain");
            queueIdle(()=>renderCard(index+1));
        };
        queueIdle(()=>renderCard(0));
    }
    q("[data-action='presets']").onclick=openPresetGallery;
    q("[data-action='presets-close']").onclick=closePresetGallery;
    let dragging = false, videoFrameHandle = 0;
    function videoFrameLoop() {
        if (!root.isConnected || source !== "video") return;
        if (!dragging && !document.hidden) { frame++; scheduleRender(); }
        if (typeof video.requestVideoFrameCallback === "function") videoFrameHandle = video.requestVideoFrameCallback(videoFrameLoop);
    }
    function startVideoPreview() {
        if (typeof video.requestVideoFrameCallback === "function") {
            if (videoFrameHandle) video.cancelVideoFrameCallback?.(videoFrameHandle);
            videoFrameHandle = video.requestVideoFrameCallback(videoFrameLoop);
        } else {
            video.ontimeupdate = () => { if (!dragging && !document.hidden) { frame++; scheduleRender(); } };
        }
    }
    const file=q(".pro-media-file"), plateFile=q(".pro-plate-file"); function open(){file.click();} q("[data-action='open']").onclick=q("[data-action='hd-open']").onclick=open;
    q("[data-action='plate']").onclick=q("[data-action='hd-plate']").onclick=()=>plateFile.click();
    file.onchange=()=>{const selected=file.files?.[0];if(!selected)return;if(url)URL.revokeObjectURL(url);url=URL.createObjectURL(selected);if(selected.type.startsWith("video/")){source="video";video.src=url;video.onloadeddata=()=>{startVideoPreview();video.play().catch(()=>{});};}else{source="image";image.src=url;image.onload=renderAll;}q(".pro-hint").style.display="none";q(".pro-status").textContent="Private preview: "+selected.name+". Not saved or queued.";file.value="";};
    plateFile.onchange=()=>{const selected=plateFile.files?.[0];if(!selected)return;if(plateUrl)URL.revokeObjectURL(plateUrl);plateUrl=URL.createObjectURL(selected);plateImage.src=plateUrl;plateImage.onload=renderAll;q(".pro-status").textContent="Preview plate: "+selected.name+". Connect grain_plate separately for Queue.";plateFile.value="";};
    const hdControls=q(".pro-controls").cloneNode(true); hdControls.className="pro-hd-controls"; q(".pro-hd-settings").append(hdControls);
    hdControls.querySelectorAll("[data-select]").forEach((el)=>el.onchange=()=>{const name=el.dataset.select;if(name==="lens_preset"){setWidget(node,name,el.value);applyLensPreset(node,el.value);}else{if(name!=="preset")markCustom();setWidget(node,name,el.value);if(name==="preset")applyPreset(node,el.value);}sync();scheduleRender();});
    hdControls.querySelectorAll("input[data-name]").forEach((el)=>el.oninput=()=>{const value=Number(el.dataset.digits)?Number(el.value):Math.round(Number(el.value));markControlCustom(el.dataset.name);setWidget(node,el.dataset.name,value);sync();scheduleRender();});
    hdControls.querySelectorAll("[data-number]").forEach((el)=>el.onchange=()=>{setWidget(node,el.dataset.number,Math.max(0,Math.trunc(Number(el.value)||0)));sync();scheduleRender();});
    q("[data-action='save']").onclick=()=>{node.properties ||= {}; node.properties.iamccsCinePostEfxV2Saved={}; PRO_FIELDS.forEach((name)=>node.properties.iamccsCinePostEfxV2Saved[name]=readWidget(node,name,null)); app.graph?.setDirtyCanvas?.(true,true); q(".pro-status").textContent="Settings committed to this node. Save the workflow to persist them on disk.";};
    const hdOverlay=q(".pro-hd"), hdClose=q("[data-action='hd-close']");
    const closeHD=()=>{hdOverlay.classList.remove("open");if(document.fullscreenElement===hdOverlay)document.exitFullscreen?.().catch?.(()=>{});requestAnimationFrame(renderAll);};
    q("[data-action='hd']").onclick=async()=>{hdOverlay.classList.add("open");sync();renderAll();try{await hdOverlay.requestFullscreen();}catch{}};
    hdClose.addEventListener("pointerdown",(event)=>{event.preventDefault();event.stopPropagation();closeHD();},{capture:true});
    hdClose.onclick=(event)=>{event.preventDefault();event.stopPropagation();closeHD();};
    document.addEventListener("fullscreenchange",()=>{if(document.fullscreenElement!==hdOverlay&&hdOverlay.classList.contains("open"))hdOverlay.classList.remove("open");scheduleRender();});
    document.addEventListener("keydown",(event)=>{if(event.key!=="Escape")return;if(presetOverlay.classList.contains("open"))closePresetGallery();else if(hdOverlay.classList.contains("open"))closeHD();});
    q("[data-action='clear']").onclick=()=>{video.pause();source=null;q(".pro-hint").style.display="grid";};
    q(".pro-screen").ondragover=(event)=>event.preventDefault();q(".pro-screen").ondrop=(event)=>{event.preventDefault();const dt=new DataTransfer();if(event.dataTransfer?.files?.[0])dt.items.add(event.dataTransfer.files[0]);file.files=dt.files;file.onchange();};
    root.addEventListener("input",(event)=>{writeSettingsSnapshot();if(!event.target.closest(".pro-hd-settings"))scheduleRender();});
    root.addEventListener("change",(event)=>{writeSettingsSnapshot();if(!event.target.closest(".pro-hd-settings"))scheduleRender();});
    root.addEventListener("pointerdown",(event)=>{if(event.target.matches?.("input[type='range']"))sliderInteracting=true;},{passive:true});
    const handleGlobalPointerDown=(event)=>{dragging=!root.contains(event.target);};
    const handleGlobalPointerEnd=()=>{const wasSlider=sliderInteracting;dragging=false;sliderInteracting=false;if(wasSlider)scheduleRender(true);restorePreview();};
    const handleVisibilityChange=()=>{if(!document.hidden)restorePreview();};
    window.addEventListener("pointerdown",handleGlobalPointerDown,{capture:true,passive:true});
    window.addEventListener("pointerup",handleGlobalPointerEnd,{capture:true,passive:true});
    window.addEventListener("pointercancel",handleGlobalPointerEnd,{capture:true,passive:true});
    window.addEventListener("blur",handleGlobalPointerEnd,{passive:true});
    document.addEventListener("visibilitychange",handleVisibilityChange,{passive:true});
    const dom=node.addDOMWidget("CinePostEfx v2 Lab","iamccs_cine_post_efx_v2_lab",root,{serialize:false});dom.computeSize=()=>[Math.max(850,(node.size?.[0]||920)-14),500];node.setSize?.([920,570]);
    const resizeObserver=typeof ResizeObserver==="function"?new ResizeObserver(()=>restorePreview()):null;
    resizeObserver?.observe(q(".pro-screen"));
    const intersectionObserver=typeof IntersectionObserver==="function"?new IntersectionObserver((entries)=>{if(entries.some((entry)=>entry.isIntersecting))restorePreview();}):null;
    intersectionObserver?.observe(root);
    const chainPreviewRestore=(name)=>{const previous=node[name];node[name]=function(){const result=previous?.apply?.(this,arguments);restorePreview();return result;};};
    chainPreviewRestore("onResize");
    chainPreviewRestore("onMouseUp");
    chainPreviewRestore("onMoved");
    const previousRemoved=node.onRemoved;
    node.onRemoved=function(){
        previewRestoreVersion++;
        delete node._iamccsWriteProSnapshot;
        resizeObserver?.disconnect();intersectionObserver?.disconnect();
        window.removeEventListener("pointerdown",handleGlobalPointerDown,true);
        window.removeEventListener("pointerup",handleGlobalPointerEnd,true);
        window.removeEventListener("pointercancel",handleGlobalPointerEnd,true);
        window.removeEventListener("blur",handleGlobalPointerEnd);
        document.removeEventListener("visibilitychange",handleVisibilityChange);
        if(videoFrameHandle&&typeof video.cancelVideoFrameCallback==="function")video.cancelVideoFrameCallback(videoFrameHandle);
        video.pause();if(url)URL.revokeObjectURL(url);if(plateUrl)URL.revokeObjectURL(plateUrl);
        return previousRemoved?.apply?.(this,arguments);
    };
    sync();writeSettingsSnapshot();renderAll();restorePreview();
}

app.registerExtension({
    name: "IAMCCS.CinePostEfxV2.UI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_TYPES.has(nodeData?.name)) return;
        const previousCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = previousCreated?.apply?.(this, arguments);
            this.color = "#2a241b";
            this.bgcolor = "#10161b";

            const presetWidget = findWidget(this, "preset");
            if (presetWidget) {
                const previousCallback = presetWidget.callback;
                presetWidget.callback = (value) => {
                    previousCallback?.call(presetWidget, value);
                    if (this._iamccsProReady && !this._iamccsProConfiguring && !this._iamccsApplyingProPreset) applyPreset(this, String(value));
                };
            }

            const previousConfigure = this.onConfigure;
            this.onConfigure = function () {
                this._iamccsProConfiguring = true;
                try { return previousConfigure?.apply?.(this, arguments); }
                finally { this._iamccsProConfiguring = false; }
            };

            queueMicrotask(() => { this._iamccsProReady = true; mountPremiumUI(this); });
            return result;
        };
    },
});
