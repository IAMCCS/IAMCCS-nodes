import { app } from "../../../scripts/app.js";

const IAMCCS_SUPERNODES_EXEC_UI_VERSION = "2026-05-28-taeltx-preview-v2";

const PRESET_CONFIGS = {
    "IAMCCS-SuperNodes AU+IMG2VID Exec Render": {
        presetWidget: "ui_preset",
        defaultPreset: "custom",
        values: {
            low_ram_safe: { vae_mode: "very_low_ram_disk", steps: 16, image_compression: 40, stitch_preset: "custom", overlap_side: "source", overlap_mode: "cut", start_frames_rule: "none", continuity_anchor_mode: "off", anchor_refresh_interval: 2, anchor_image_strength: 0.0, anti_drift_mode: "off", anti_drift_strength: 0.0, identity_persistence_strength: 0.0, vram_flush: true, motion_intensity: 1.0 },
            balanced: { vae_mode: "normal_tiled_vhs", steps: 20, image_compression: 28, stitch_preset: "custom", overlap_side: "source", overlap_mode: "cut", start_frames_rule: "none", continuity_anchor_mode: "off", anchor_refresh_interval: 2, anchor_image_strength: 0.0, anti_drift_mode: "off", anti_drift_strength: 0.0, identity_persistence_strength: 0.0, vram_flush: false, motion_intensity: 1.0 },
            high_quality: { vae_mode: "high_vram", steps: 24, image_compression: 20, stitch_preset: "custom", overlap_side: "source", overlap_mode: "cut", start_frames_rule: "none", continuity_anchor_mode: "off", anchor_refresh_interval: 1, anchor_image_strength: 0.0, anti_drift_mode: "off", anti_drift_strength: 0.0, identity_persistence_strength: 0.0, vram_flush: false, motion_intensity: 1.0 },
            fast_preview: { vae_mode: "low_ram_disk", steps: 12, image_compression: 45, stitch_preset: "custom", overlap_side: "source", overlap_mode: "cut", start_frames_rule: "none", continuity_anchor_mode: "off", anchor_refresh_interval: 2, anti_drift_mode: "off", anti_drift_strength: 0.0, identity_persistence_strength: 0.0, vram_flush: true, motion_intensity: 1.0 },
            motion_controlled: { stitch_preset: "custom", overlap_side: "source", overlap_mode: "cut", start_frames_rule: "none", continuity_anchor_mode: "off", anchor_image_strength: 0.0, anti_drift_mode: "off", anti_drift_strength: 0.0, identity_persistence_strength: 0.0, motion_intensity: 1.35, image_strength: 0.78, image_compression: 32 },
        },
        visibility: {
            low_ram_safe: { anchor_image_strength: false, anti_drift_mode: true, anti_drift_strength: false, identity_persistence_strength: false, output_root: true },
            balanced: { anchor_image_strength: false, anti_drift_mode: true, anti_drift_strength: false, identity_persistence_strength: false, output_root: true },
            high_quality: { anchor_image_strength: false, anti_drift_mode: true, anti_drift_strength: false, identity_persistence_strength: false, output_root: true },
            fast_preview: { anchor_image_strength: false, anti_drift_mode: false, anti_drift_strength: false, identity_persistence_strength: false, output_root: false },
            motion_controlled: { anchor_image_strength: false, anti_drift_mode: true, anti_drift_strength: false, identity_persistence_strength: false, output_root: true },
        },
    },
    "IAMCCS-SuperNodes AU+IMG2VID Exec VAE": {
        presetWidget: "ui_preset",
        defaultPreset: "custom",
        values: {
            balanced: { decode_mode: "normal_tiled_vhs", vram_flush: false, jpg_quality: 95, crf: 19, tiled_tile_size: 512, tiled_overlap: 64, tiled_temporal_size: 256, tiled_temporal_overlap: 32, cleanup_before_decode: false },
            high_quality: { decode_mode: "high_vram", vram_flush: false, jpg_quality: 100, crf: 16, tiled_tile_size: 768, tiled_overlap: 96, tiled_temporal_size: 256, tiled_temporal_overlap: 32, cleanup_before_decode: false },
            fast_preview: { decode_mode: "low_ram_disk", vram_flush: true, jpg_quality: 88, crf: 24, tiled_tile_size: 384, tiled_overlap: 48, tiled_temporal_size: 128, tiled_temporal_overlap: 16, cleanup_before_decode: true },
            low_ram_safe: { decode_mode: "low_ram_disk", vram_flush: true, jpg_quality: 92, crf: 21, tiled_tile_size: 384, tiled_overlap: 48, tiled_temporal_size: 128, tiled_temporal_overlap: 16, cleanup_before_decode: true },
            very_low_ram_decode: { decode_mode: "very_low_ram_disk", vram_flush: true, jpg_quality: 90, crf: 22, tiled_tile_size: 256, tiled_overlap: 32, tiled_temporal_size: 128, tiled_temporal_overlap: 16, cleanup_before_decode: true },
        },
        visibility: {
            balanced: {},
            high_quality: {},
            fast_preview: {},
            low_ram_safe: {},
            very_low_ram_decode: {},
        },
    },
};

const NODE_GROUPS = {
    "IAMCCS-SuperNodes AU+IMG2VID Exec Planner": [
        { key: "route", label: "A+I2V Route", color: "#2f8f80", widgets: ["audio_img2vid_backend", "audio_img2vid_mode", "single_duration_seconds", "segment_count", "segment_seconds", "overlap_frames", "ltx_round_mode", "fps", "debug_verbose"] },
        { key: "audio", label: "Audio Source", color: "#3d8a5c", widgets: ["audio_preprocess_mode", "melband_model_name"] },
    ],
    "IAMCCS-SuperNodes AU+IMG2VID Exec Render": [
        { key: "generation", label: "Generation", color: "#8a5ca0", widgets: ["generation_type", "generated_media_duration_seconds", "generated_media_fps", "debug_verbose"] },
        { key: "prompts", label: "Prompting", color: "#9a6b2f", widgets: ["positive_text", "negative_text"] },
        { key: "video", label: "Video", color: "#3f6fb0", widgets: ["width", "height"] },
        { key: "sampling", label: "Sampling", color: "#8352a6", widgets: ["steps", "cfg", "sampler_name", "seed", "max_shift", "base_shift", "sigma_terminal", "show_manual_sigmas", "manual_sigmas", "image_strength", "image_compression"] },
        { key: "taeltx_preview", label: "TAELTX Preview", color: "#4e7c9b", widgets: ["taeltx_preview", "taeltx_preview_max_frames", "taeltx_preview_fps"] },
        { key: "transition", label: "Transition / Stitch", color: "#7a7040", widgets: ["stitch_preset", "overlap_side", "overlap_mode", "start_frames_rule", "color_match_mode", "color_match_strength"] },
        { key: "audio_context", label: "Audio Context", color: "#477c7a", widgets: ["audio_context_mode", "audio_left_context_s", "audio_right_context_s"] },
        { key: "latent_refresh", label: "Latent Refresh (beta)", color: "#b35c5c", widgets: ["continuity_anchor_mode", "anchor_refresh_interval", "anchor_image_strength", "anti_drift_mode", "anti_drift_strength", "identity_persistence_strength"] },
        { key: "stage2", label: "Second Stage", color: "#8a5ca0", widgets: ["second_stage_mode", "stage2_model_policy", "second_stage_upscale_model", "second_stage_reinject_strength", "second_stage_cfg", "second_stage_manual_sigmas"] },
    ],
    "IAMCCS-SuperNodes Second Stage": [
        { key: "stage2", label: "Second Stage", color: "#8a5ca0", widgets: ["second_stage_mode", "stage2_model_policy", "second_stage_upscale_model", "second_stage_reinject_strength", "second_stage_cfg", "second_stage_manual_sigmas"] },
    ],
    IAMCCS_GC_AudioConcatSupernode: [
        { key: "concat", label: "Audio Concat", color: "#4e8f6b", widgets: ["concat_mode", "clip_durations_seconds", "gap_seconds", "intro_seconds", "outro_seconds"] },
    ],
    "IAMCCS-SuperNodes AU+IMG2VID Exec VAE": [
        { key: "mode", label: "VAE Mode", color: "#4b79a6", widgets: ["decode_mode", "frame_rate", "tiled_tile_size", "tiled_overlap", "tiled_temporal_size", "tiled_temporal_overlap", "cleanup_before_decode", "frames_subdir", "image_format", "jpg_quality", "output_root", "vram_flush"] },
        { key: "output", label: "Output", color: "#9b7441", widgets: ["filename_prefix", "crf", "trim_to_audio", "debug_verbose"] },
    ],
    "IAMCCS-SuperNodes AU+IMG2VID Exec Finalize": [
        { key: "output", label: "Output", color: "#6a85a0", widgets: ["frame_rate", "filename_prefix", "crf", "pix_fmt", "trim_to_audio", "debug_verbose"] },
    ],
};

const SECTION_BUTTON_COLLAPSED_BG = "#1f2935";
const SECTION_BUTTON_COLLAPSED_BORDER = "#6d7785";
const SECTION_BUTTON_RADIUS = 6;
const SECTION_BUTTON_TEXT = "#f8fbff";
const SECTION_BUTTON_HEIGHT = 30;
const DEFAULT_COLLAPSED_SECTION_KEYS = new Set(["taeltx_preview", "latent_refresh", "stage2", "output"]);
const RENDER_INTERNAL_WIDGETS = new Set([
    "ui_preset",
    "backend_mode",
    "generation_mode",
    "media_mode",
    "vae_mode",
    "motion_intensity",
    "vram_flush",
    "downstream_stage_mode",
    "output_root",
    "segment_overlay_mode",
    "segment_overlay_text",
]);
const GENERATED_DURATION_TYPES = new Set(["img2video", "text2video"]);
const REF_MAIN_SIGMAS = "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0";
const REF_STAGE2_SIGMAS = "0.909375, 0.725, 0.421875, 0.0";
const CARTOON_STAGE2_SIGMAS = "0.8025, 0.6332, 0.3425, 0.0";
const REF_LTX23_UPSCALE_MODEL = "ltx-2.3-spatial-upscaler-x2-1.0.safetensors";
const DEFAULT_VIDEO_WIDTH = 1280;
const DEFAULT_VIDEO_HEIGHT = 736;
const DEFAULT_GENERATED_DURATION_SECONDS = 10.0;
const DEFAULT_GENERATED_FPS = 25.0;
const DEFAULT_VAE_FRAME_RATE = DEFAULT_GENERATED_FPS;
const REFERENCE_AUDIO_IMG2VID_TYPE = "audio+image2video";
const AUDIO_IMAGE_LEGACY_TYPES = new Set(["aud+img2video_simple", "aud+img2video_2_segments", "aud+img2video_infinite"]);
const REFERENCE_AUDIO_IMG2VID_FPS = 25.0;
const REFERENCE_AUDIO_IMG2VID_WIDTH = DEFAULT_VIDEO_WIDTH;
const REFERENCE_AUDIO_IMG2VID_HEIGHT = 720;
const STALE_AUDIO_IMG2VID_WIDTH = 768;
const STALE_AUDIO_IMG2VID_HEIGHT = 512;
const REFERENCE_AUDIO_IMG2VID_NEGATIVE = "closed mouth, smiling without speaking, singing concert, dancing, exaggerated head movement, blurry mouth";
const COMMON_GENERATION_DEFAULTS = {
    ui_preset: "custom",
    stitch_preset: "custom",
    overlap_side: "source",
    overlap_mode: "cut",
    start_frames_rule: "none",
    color_match_mode: "none",
    color_match_strength: 0.25,
    vram_flush: false,
    motion_intensity: 1.0,
    stage2_model_policy: "stage2_model_if_connected",
    continuity_anchor_mode: "off",
    anchor_image_strength: 0.0,
    anti_drift_mode: "off",
    anti_drift_strength: 0.0,
    identity_persistence_strength: 0.0,
    second_stage_mode: "off",
    second_stage_reinject_strength: 0.0,
    show_manual_sigmas: false,
    taeltx_preview: false,
    taeltx_preview_max_frames: 17,
    taeltx_preview_fps: 8,
};
// Regression guard: these sampling defaults are the contract for the
// generation_type switch. They are applied only when the type changes; after
// that, the user-selected frontend values must remain the backend inputs.
const GENERATION_SAMPLING_DEFAULTS = {
    "audio+image2video": {
        steps: 8,
        cfg: 1.0,
        sampler_name: "lcm",
        manual_sigmas: "",
        image_strength: 0.8,
        image_compression: 33,
    },
    "aud+img2video_simple": {
        steps: 8,
        cfg: 1.0,
        sampler_name: "lcm",
        manual_sigmas: "",
        image_strength: 0.8,
        image_compression: 33,
    },
    "aud+img2video_2_segments": {
        steps: 8,
        cfg: 1.0,
        sampler_name: "euler",
        show_manual_sigmas: true,
        manual_sigmas: REF_MAIN_SIGMAS,
        image_strength: 0.9,
        image_compression: 35,
    },
    "aud+img2video_infinite": {
        steps: 8,
        cfg: 1.0,
        sampler_name: "euler",
        show_manual_sigmas: true,
        manual_sigmas: REF_MAIN_SIGMAS,
        image_strength: 0.9,
        image_compression: 35,
    },
    "text+audio2video": {
        steps: 8,
        cfg: 1.0,
        sampler_name: "euler",
        show_manual_sigmas: true,
        manual_sigmas: REF_MAIN_SIGMAS,
        image_strength: 0.0,
        image_compression: 35,
    },
    img2video: {
        steps: 8,
        cfg: 1.0,
        sampler_name: "lcm",
        max_shift: 2.05,
        base_shift: 0.95,
        sigma_terminal: 0.1,
        manual_sigmas: "",
        image_strength: 1.0,
        image_compression: 33,
    },
    text2video: {
        steps: 8,
        cfg: 1.0,
        sampler_name: "euler",
        max_shift: 2.05,
        base_shift: 0.95,
        sigma_terminal: 0.1,
        manual_sigmas: "",
        image_strength: 0.0,
        image_compression: 33,
    },
};
const ANCHOR_MODE_DEFAULTS = {
    off: { anchor_refresh_interval: 2, anchor_image_strength: 0.0, anti_drift_mode: "off", anti_drift_strength: 0.0, identity_persistence_strength: 0.0 },
    tail_only: { anchor_refresh_interval: 2, anchor_image_strength: 0.25, anti_drift_mode: "rolling_adain", anti_drift_strength: 0.12, identity_persistence_strength: 0.0 },
    periodic_tail_only: { anchor_refresh_interval: 2, anchor_image_strength: 0.25, anti_drift_mode: "rolling_adain", anti_drift_strength: 0.12, identity_persistence_strength: 0.0 },
    tail_then_source_refresh: { anchor_refresh_interval: 2, anchor_image_strength: 0.45, anti_drift_mode: "dual_reference_adain", anti_drift_strength: 0.16, identity_persistence_strength: 0.06 },
    periodic_tail_then_source_refresh: { anchor_refresh_interval: 2, anchor_image_strength: 0.45, anti_drift_mode: "dual_reference_adain", anti_drift_strength: 0.16, identity_persistence_strength: 0.06 },
    periodic_source_refresh: { anchor_refresh_interval: 2, anchor_image_strength: 0.45, anti_drift_mode: "dual_reference_adain", anti_drift_strength: 0.16, identity_persistence_strength: 0.06 },
    always_source_refresh: { anchor_refresh_interval: 1, anchor_image_strength: 0.45, anti_drift_mode: "dual_reference_adain", anti_drift_strength: 0.16, identity_persistence_strength: 0.06 },
};
const GENERATION_TYPE_CONFIGS = {
    "audio+image2video": { ...COMMON_GENERATION_DEFAULTS, ...GENERATION_SAMPLING_DEFAULTS["audio+image2video"], generation_mode: "img2vid", backend_mode: "auto", media_mode: "input_audio_img2vid", width: REFERENCE_AUDIO_IMG2VID_WIDTH, height: REFERENCE_AUDIO_IMG2VID_HEIGHT, vae_mode: "normal_tiled_iamccs", second_stage_cfg: 1.0, second_stage_manual_sigmas: REF_STAGE2_SIGMAS },
    "aud+img2video_simple": { ...COMMON_GENERATION_DEFAULTS, ...GENERATION_SAMPLING_DEFAULTS["audio+image2video"], generation_mode: "img2vid", backend_mode: "single_best", media_mode: "input_audio_img2vid", width: REFERENCE_AUDIO_IMG2VID_WIDTH, height: REFERENCE_AUDIO_IMG2VID_HEIGHT, vae_mode: "normal_tiled_iamccs", second_stage_cfg: 1.0, second_stage_manual_sigmas: REF_STAGE2_SIGMAS },
    "aud+img2video_2_segments": { ...COMMON_GENERATION_DEFAULTS, ...GENERATION_SAMPLING_DEFAULTS["audio+image2video"], generation_mode: "img2vid", backend_mode: "two_segments_normal_vram", media_mode: "input_audio_img2vid", width: REFERENCE_AUDIO_IMG2VID_WIDTH, height: REFERENCE_AUDIO_IMG2VID_HEIGHT, vae_mode: "normal_tiled_iamccs", second_stage_cfg: 1.0, second_stage_manual_sigmas: REF_STAGE2_SIGMAS },
    "aud+img2video_infinite": { ...COMMON_GENERATION_DEFAULTS, ...GENERATION_SAMPLING_DEFAULTS["audio+image2video"], generation_mode: "img2vid", backend_mode: "auto", media_mode: "input_audio_img2vid", width: REFERENCE_AUDIO_IMG2VID_WIDTH, height: REFERENCE_AUDIO_IMG2VID_HEIGHT, vae_mode: "normal_tiled_iamccs", second_stage_cfg: 1.0, second_stage_manual_sigmas: REF_STAGE2_SIGMAS },
    "text+audio2video": { ...COMMON_GENERATION_DEFAULTS, ...GENERATION_SAMPLING_DEFAULTS["text+audio2video"], generation_mode: "t2v", backend_mode: "single_best", media_mode: "input_audio_t2v", width: DEFAULT_VIDEO_WIDTH, height: DEFAULT_VIDEO_HEIGHT, vae_mode: "normal_tiled_iamccs", second_stage_cfg: 1.0, second_stage_manual_sigmas: CARTOON_STAGE2_SIGMAS },
    img2video: { ...COMMON_GENERATION_DEFAULTS, ...GENERATION_SAMPLING_DEFAULTS.img2video, generation_mode: "img2vid", backend_mode: "single_best", media_mode: "img2vid_pure", width: DEFAULT_VIDEO_WIDTH, height: DEFAULT_VIDEO_HEIGHT, generated_media_duration_seconds: DEFAULT_GENERATED_DURATION_SECONDS, generated_media_fps: DEFAULT_GENERATED_FPS, vae_mode: "normal_tiled_vhs", second_stage_upscale_model: REF_LTX23_UPSCALE_MODEL, second_stage_cfg: 1.0, second_stage_manual_sigmas: REF_STAGE2_SIGMAS },
    text2video: { ...COMMON_GENERATION_DEFAULTS, ...GENERATION_SAMPLING_DEFAULTS.text2video, generation_mode: "t2v", backend_mode: "single_best", media_mode: "t2v_pure", width: DEFAULT_VIDEO_WIDTH, height: DEFAULT_VIDEO_HEIGHT, generated_media_duration_seconds: DEFAULT_GENERATED_DURATION_SECONDS, generated_media_fps: DEFAULT_GENERATED_FPS, vae_mode: "normal_tiled_vhs", second_stage_upscale_model: REF_LTX23_UPSCALE_MODEL, second_stage_cfg: 1.0, second_stage_manual_sigmas: REF_STAGE2_SIGMAS },
};
const RENDER_UI_PRESET_LEGACY_VALUES = [
    "loop_lipsync_safe",
    "img2vid_generated_audio",
    "t2v_generated_audio",
    "img2vid_pure",
    "t2v_pure",
    "loop_img2vid_pure_normal_vram",
    "loop_t2v_pure_normal_vram",
    "loop_img2vid_pure_low_ram",
];
const RENDER_UI_PRESET_VALUES = new Set(["custom", ...Object.keys(PRESET_CONFIGS["IAMCCS-SuperNodes AU+IMG2VID Exec Render"]?.values || {})]);
const RENDER_UI_PRESET_VISIBLE_VALUES = ["custom", "low_ram_safe", "balanced", "high_quality", "fast_preview", "motion_controlled"];
const RENDER_BACKEND_MODE_VISIBLE_VALUES = [
    "auto",
    "single_best",
    "ti2v_incremental_advanced",
    "legacy_single",
    "legacy_two_segments",
    "legacy_loop",
    "two_segments_normal_vram",
    "three_segments_normal_vram",
    "loop_normal_vram",
    "loop_low_ram_disk",
];
const RENDER_BACKEND_MODE_VALUES = new Set([...RENDER_BACKEND_MODE_VISIBLE_VALUES, "legacy backend"]);
const TI2V_RENDER_BACKEND_VISIBLE_VALUES = ["single_best", "ti2v_incremental_advanced"];
const RENDER_STITCH_PRESET_VALUES = new Set([
    "custom",
    "lossless_refresh_24fps",
    "lossless_refresh_strong_24fps",
    "videoclip_audio_24fps",
    "monologue_audio_24fps",
    "target_extension_ltx2",
    "cut_bestofk_16",
    "cut_bestofk_16_luma",
    "cut_bestofk_32",
    "micro_crossfade_3",
]);
const RENDER_OVERLAP_SIDE_VALUES = new Set(["source", "new_images"]);
const RENDER_OVERLAP_MODE_VALUES = new Set(["cut", "linear_blend", "ease_in_out", "filmic_crossfade"]);
const RENDER_START_FRAMES_RULE_VALUES = new Set(["none", "ltx2_round_down", "ltx2_nearest"]);
const RENDER_COLOR_MATCH_MODE_VALUES = new Set(["none", "luma_only", "per_channel"]);
const RENDER_AUDIO_CONTEXT_MODE_VALUES = new Set(["left_context_only", "right_context_only", "symmetric_context", "no_overlap"]);
const LEGACY_RENDER_BACKEND_DEFAULTS = {
    legacy_single: {
        width: REFERENCE_AUDIO_IMG2VID_WIDTH,
        height: REFERENCE_AUDIO_IMG2VID_HEIGHT,
        steps: 8,
        cfg: 1.0,
        sampler_name: "euler",
        manual_sigmas: REF_MAIN_SIGMAS,
        image_strength: 1.0,
        image_compression: 33,
        max_shift: 2.05,
        base_shift: 0.95,
        sigma_terminal: 0.1,
        audio_context_mode: "left_context_only",
        audio_left_context_s: 0.5,
        audio_right_context_s: 0.0,
        stitch_preset: "custom",
        overlap_side: "source",
        overlap_mode: "cut",
        start_frames_rule: "none",
        color_match_mode: "none",
        color_match_strength: 0.25,
        continuity_anchor_mode: "off",
        anchor_refresh_interval: 2,
        anchor_image_strength: 0.0,
        anti_drift_mode: "off",
        anti_drift_strength: 0.0,
        identity_persistence_strength: 0.0,
        second_stage_mode: "off",
        second_stage_cfg: 1.0,
        second_stage_manual_sigmas: "0.909375, 0.725, 0.421875, 0.0",
    },
    legacy_two_segments: {
        width: REFERENCE_AUDIO_IMG2VID_WIDTH,
        height: REFERENCE_AUDIO_IMG2VID_HEIGHT,
        steps: 8,
        cfg: 1.0,
        sampler_name: "euler",
        manual_sigmas: REF_MAIN_SIGMAS,
        image_strength: 1.0,
        image_compression: 33,
        max_shift: 2.05,
        base_shift: 0.95,
        sigma_terminal: 0.1,
        audio_context_mode: "left_context_only",
        audio_left_context_s: 0.5,
        audio_right_context_s: 0.0,
        stitch_preset: "custom",
        overlap_side: "source",
        overlap_mode: "cut",
        start_frames_rule: "none",
        color_match_mode: "none",
        color_match_strength: 0.25,
        continuity_anchor_mode: "off",
        anchor_refresh_interval: 2,
        anchor_image_strength: 0.0,
        anti_drift_mode: "off",
        anti_drift_strength: 0.0,
        identity_persistence_strength: 0.0,
        second_stage_mode: "off",
        second_stage_cfg: 1.0,
        second_stage_manual_sigmas: "0.909375, 0.725, 0.421875, 0.0",
    },
    legacy_loop: {
        width: REFERENCE_AUDIO_IMG2VID_WIDTH,
        height: REFERENCE_AUDIO_IMG2VID_HEIGHT,
        steps: 8,
        cfg: 1.0,
        sampler_name: "euler",
        manual_sigmas: REF_MAIN_SIGMAS,
        image_strength: 1.0,
        image_compression: 33,
        max_shift: 2.05,
        base_shift: 0.95,
        sigma_terminal: 0.1,
        audio_context_mode: "left_context_only",
        audio_left_context_s: 0.5,
        audio_right_context_s: 0.0,
        stitch_preset: "custom",
        overlap_side: "source",
        overlap_mode: "cut",
        start_frames_rule: "none",
        color_match_mode: "none",
        color_match_strength: 0.25,
        continuity_anchor_mode: "off",
        anchor_refresh_interval: 2,
        anchor_image_strength: 0.0,
        anti_drift_mode: "off",
        anti_drift_strength: 0.0,
        identity_persistence_strength: 0.0,
        second_stage_mode: "off",
        second_stage_cfg: 1.0,
        second_stage_manual_sigmas: "0.909375, 0.725, 0.421875, 0.0",
    },
};
const VAE_TILED_DECODE_MODES = new Set(["normal_tiled_iamccs", "normal_tiled_vhs", "custom_mode", "inherit_render_backend"]);
const VAE_DISK_DECODE_MODES = new Set(["low_ram", "low_ram_disk", "very_low_ram", "very_low_ram_disk"]);
const VAE_DECODE_MODE_VALUES = new Set(["normal_tiled_iamccs", "low_ram", "very_low_ram", "high_vram", "inherit_render_backend", "normal_tiled_vhs", "low_ram_disk", "very_low_ram_disk", "custom_mode"]);
const VAE_UI_PRESET_VALUES = new Set(["custom", ...Object.keys(PRESET_CONFIGS["IAMCCS-SuperNodes AU+IMG2VID Exec VAE"]?.values || {})]);
const VAE_DECODE_MODE_DEFAULTS = {
    normal_tiled_iamccs: { tiled_tile_size: 512, tiled_overlap: 64, tiled_temporal_size: 256, tiled_temporal_overlap: 32, cleanup_before_decode: false },
    inherit_render_backend: { tiled_tile_size: 512, tiled_overlap: 64, tiled_temporal_size: 256, tiled_temporal_overlap: 32, cleanup_before_decode: false },
    normal_tiled_vhs: { tiled_tile_size: 512, tiled_overlap: 64, tiled_temporal_size: 256, tiled_temporal_overlap: 32, cleanup_before_decode: false },
    custom_mode: { tiled_tile_size: 512, tiled_overlap: 64, tiled_temporal_size: 256, tiled_temporal_overlap: 32, cleanup_before_decode: false },
    low_ram: { tiled_tile_size: 384, tiled_overlap: 48, tiled_temporal_size: 128, tiled_temporal_overlap: 16, cleanup_before_decode: true },
    low_ram_disk: { tiled_tile_size: 384, tiled_overlap: 48, tiled_temporal_size: 128, tiled_temporal_overlap: 16, cleanup_before_decode: true },
    very_low_ram: { tiled_tile_size: 256, tiled_overlap: 32, tiled_temporal_size: 128, tiled_temporal_overlap: 16, cleanup_before_decode: true },
    very_low_ram_disk: { tiled_tile_size: 256, tiled_overlap: 32, tiled_temporal_size: 128, tiled_temporal_overlap: 16, cleanup_before_decode: true },
    high_vram: { tiled_tile_size: 768, tiled_overlap: 96, tiled_temporal_size: 256, tiled_temporal_overlap: 32, cleanup_before_decode: false },
};
const RENDER_GENERATION_TYPE_VALUES = new Set(Object.keys(GENERATION_TYPE_CONFIGS));

function normalizeReferenceAudioImg2VidValues(valuesByName) {
    const generationType = String(valuesByName?.generation_type || "");
    if (!valuesByName || (generationType !== REFERENCE_AUDIO_IMG2VID_TYPE && !AUDIO_IMAGE_LEGACY_TYPES.has(generationType))) {
        return valuesByName;
    }
    const config = GENERATION_TYPE_CONFIGS[REFERENCE_AUDIO_IMG2VID_TYPE];
    valuesByName.generation_type = REFERENCE_AUDIO_IMG2VID_TYPE;
    valuesByName.generation_mode = config.generation_mode;
    valuesByName.backend_mode = config.backend_mode;
    valuesByName.media_mode = config.media_mode;

    const ensureNumber = (widgetName, fallback) => {
        if (!Number.isFinite(Number(valuesByName[widgetName])) || isSectionButtonCaption(valuesByName[widgetName])) {
            valuesByName[widgetName] = fallback;
        }
    };
    const ensureText = (widgetName, fallback) => {
        const text = String(valuesByName[widgetName] ?? "");
        if (!text || isSectionButtonCaption(text)) {
            valuesByName[widgetName] = fallback;
        }
    };

    ensureText("ui_preset", "custom");
    ensureNumber("generated_media_fps", REFERENCE_AUDIO_IMG2VID_FPS);
    if (isSectionButtonCaption(String(valuesByName.negative_text ?? ""))) {
        valuesByName.negative_text = "";
    }
    ensureNumber("width", config.width);
    ensureNumber("height", config.height);
    ensureNumber("steps", config.steps);
    ensureNumber("cfg", config.cfg);
    ensureText("sampler_name", config.sampler_name);
    ensureText("manual_sigmas", config.manual_sigmas);
    ensureNumber("image_strength", config.image_strength);
    ensureNumber("image_compression", config.image_compression);
    ensureText("vae_mode", config.vae_mode);
    ensureText("second_stage_mode", "off");
    ensureNumber("second_stage_reinject_strength", 0.0);
    ensureNumber("second_stage_cfg", config.second_stage_cfg);
    ensureText("second_stage_manual_sigmas", config.second_stage_manual_sigmas);
    ensureText("color_match_mode", "none");
    ensureNumber("color_match_strength", 0.25);
    ensureText("continuity_anchor_mode", "off");
    ensureNumber("anchor_image_strength", 0.0);
    ensureText("anti_drift_mode", "off");
    ensureNumber("anti_drift_strength", 0.0);
    ensureNumber("identity_persistence_strength", 0.0);
    return valuesByName;
}
const RENDER_LEGACY_WIDGET_ORDER_PRE_ADVANCED_MANUAL_SIGMAS = [
    "generation_mode", "backend_mode", "positive_text", "negative_text", "width", "height", "steps", "cfg", "sampler_name", "seed", "control_after_generate",
    "max_shift", "base_shift", "sigma_terminal", "manual_sigmas", "image_strength", "image_compression",
    "audio_context_mode", "audio_left_context_s", "audio_right_context_s",
    "stitch_preset", "overlap_side", "overlap_mode", "start_frames_rule", "color_match_mode", "color_match_strength",
    "continuity_anchor_mode", "anchor_refresh_interval", "anchor_image_strength", "anti_drift_mode", "anti_drift_strength", "identity_persistence_strength",
    "vae_mode", "downstream_stage_mode", "output_root", "segment_overlay_mode", "segment_overlay_text",
    "second_stage_mode", "stage2_model_policy", "second_stage_upscale_model", "second_stage_reinject_strength", "second_stage_cfg", "second_stage_manual_sigmas",
    "media_mode", "vram_flush", "motion_intensity", "debug_verbose", "ui_preset", "generated_media_duration_seconds", "generation_type", "taeltx_preview", "taeltx_preview_max_frames", "taeltx_preview_fps",
];
const RENDER_LEGACY_WIDGET_ORDER = [
    "generation_mode", "backend_mode", "positive_text", "negative_text", "width", "height", "steps", "cfg", "sampler_name", "seed", "control_after_generate",
    "max_shift", "base_shift", "sigma_terminal", "show_manual_sigmas", "manual_sigmas", "image_strength", "image_compression",
    "audio_context_mode", "audio_left_context_s", "audio_right_context_s",
    "stitch_preset", "overlap_side", "overlap_mode", "start_frames_rule", "color_match_mode", "color_match_strength",
    "continuity_anchor_mode", "anchor_refresh_interval", "anchor_image_strength", "anti_drift_mode", "anti_drift_strength", "identity_persistence_strength",
    "vae_mode", "downstream_stage_mode", "output_root", "segment_overlay_mode", "segment_overlay_text",
    "second_stage_mode", "stage2_model_policy", "second_stage_upscale_model", "second_stage_reinject_strength", "second_stage_cfg", "second_stage_manual_sigmas",
    "media_mode", "vram_flush", "motion_intensity", "debug_verbose", "ui_preset", "generated_media_duration_seconds", "generation_type", "taeltx_preview", "taeltx_preview_max_frames", "taeltx_preview_fps",
];
const RENDER_CURRENT_WIDGET_ORDER_PRE_GENERATED_FPS_PRE_ADVANCED_MANUAL_SIGMAS = [
    "generation_type", "ui_preset", "generated_media_duration_seconds", "generation_mode", "backend_mode", "positive_text", "negative_text",
    "width", "height", "steps", "cfg", "sampler_name", "seed", "control_after_generate",
    "max_shift", "base_shift", "sigma_terminal", "manual_sigmas", "image_strength", "image_compression",
    "audio_context_mode", "audio_left_context_s", "audio_right_context_s",
    "stitch_preset", "overlap_side", "overlap_mode", "start_frames_rule", "color_match_mode", "color_match_strength",
    "continuity_anchor_mode", "anchor_refresh_interval", "anchor_image_strength", "anti_drift_mode", "anti_drift_strength", "identity_persistence_strength",
    "vae_mode", "downstream_stage_mode", "output_root", "segment_overlay_mode", "segment_overlay_text",
    "second_stage_mode", "stage2_model_policy", "second_stage_upscale_model", "second_stage_reinject_strength", "second_stage_cfg", "second_stage_manual_sigmas",
    "media_mode", "vram_flush", "motion_intensity", "debug_verbose", "taeltx_preview", "taeltx_preview_max_frames", "taeltx_preview_fps",
];
const RENDER_CURRENT_WIDGET_ORDER_PRE_GENERATED_FPS = [
    "generation_type", "ui_preset", "generated_media_duration_seconds", "generation_mode", "backend_mode", "positive_text", "negative_text",
    "width", "height", "steps", "cfg", "sampler_name", "seed", "control_after_generate",
    "max_shift", "base_shift", "sigma_terminal", "show_manual_sigmas", "manual_sigmas", "image_strength", "image_compression",
    "audio_context_mode", "audio_left_context_s", "audio_right_context_s",
    "stitch_preset", "overlap_side", "overlap_mode", "start_frames_rule", "color_match_mode", "color_match_strength",
    "continuity_anchor_mode", "anchor_refresh_interval", "anchor_image_strength", "anti_drift_mode", "anti_drift_strength", "identity_persistence_strength",
    "vae_mode", "downstream_stage_mode", "output_root", "segment_overlay_mode", "segment_overlay_text",
    "second_stage_mode", "stage2_model_policy", "second_stage_upscale_model", "second_stage_reinject_strength", "second_stage_cfg", "second_stage_manual_sigmas",
    "media_mode", "vram_flush", "motion_intensity", "debug_verbose", "taeltx_preview", "taeltx_preview_max_frames", "taeltx_preview_fps",
];
const RENDER_CURRENT_WIDGET_ORDER_PRE_ADVANCED_MANUAL_SIGMAS = [
    "generation_type", "ui_preset", "generated_media_duration_seconds", "generated_media_fps", "generation_mode", "backend_mode", "positive_text", "negative_text",
    "width", "height", "steps", "cfg", "sampler_name", "seed", "control_after_generate",
    "max_shift", "base_shift", "sigma_terminal", "manual_sigmas", "image_strength", "image_compression",
    "audio_context_mode", "audio_left_context_s", "audio_right_context_s",
    "stitch_preset", "overlap_side", "overlap_mode", "start_frames_rule", "color_match_mode", "color_match_strength",
    "continuity_anchor_mode", "anchor_refresh_interval", "anchor_image_strength", "anti_drift_mode", "anti_drift_strength", "identity_persistence_strength",
    "vae_mode", "downstream_stage_mode", "output_root", "segment_overlay_mode", "segment_overlay_text",
    "second_stage_mode", "stage2_model_policy", "second_stage_upscale_model", "second_stage_reinject_strength", "second_stage_cfg", "second_stage_manual_sigmas",
    "media_mode", "vram_flush", "motion_intensity", "debug_verbose", "taeltx_preview", "taeltx_preview_max_frames", "taeltx_preview_fps",
];
const RENDER_CURRENT_WIDGET_ORDER = [
    "generation_type", "ui_preset", "generated_media_duration_seconds", "generated_media_fps", "generation_mode", "backend_mode", "positive_text", "negative_text",
    "width", "height", "steps", "cfg", "sampler_name", "seed", "control_after_generate",
    "max_shift", "base_shift", "sigma_terminal", "show_manual_sigmas", "manual_sigmas", "image_strength", "image_compression",
    "audio_context_mode", "audio_left_context_s", "audio_right_context_s",
    "stitch_preset", "overlap_side", "overlap_mode", "start_frames_rule", "color_match_mode", "color_match_strength",
    "continuity_anchor_mode", "anchor_refresh_interval", "anchor_image_strength", "anti_drift_mode", "anti_drift_strength", "identity_persistence_strength",
    "vae_mode", "downstream_stage_mode", "output_root", "segment_overlay_mode", "segment_overlay_text",
    "second_stage_mode", "stage2_model_policy", "second_stage_upscale_model", "second_stage_reinject_strength", "second_stage_cfg", "second_stage_manual_sigmas",
    "media_mode", "vram_flush", "motion_intensity", "debug_verbose", "taeltx_preview", "taeltx_preview_max_frames", "taeltx_preview_fps",
];
const VAE_LEGACY_WIDGET_ORDER = [
    "frame_rate", "decode_mode", "tiled_tile_size", "tiled_overlap", "frames_subdir", "image_format", "jpg_quality",
    "output_root", "filename_prefix", "crf", "pix_fmt", "trim_to_audio", "save_metadata", "vram_flush", "ui_preset", "debug_verbose",
];
const VAE_LEGACY_PY_WIDGET_ORDER = [
    "frame_rate", "decode_mode", "filename_prefix", "output_root", "frames_subdir", "image_format", "jpg_quality",
    "tiled_tile_size", "tiled_overlap", "crf", "pix_fmt", "trim_to_audio", "save_metadata", "vram_flush", "ui_preset", "debug_verbose",
];
const VAE_CURRENT_WIDGET_ORDER = [
    "frame_rate", "decode_mode", "tiled_tile_size", "tiled_overlap", "tiled_temporal_size", "tiled_temporal_overlap",
    "cleanup_before_decode", "frames_subdir", "image_format", "jpg_quality", "output_root", "filename_prefix",
    "crf", "pix_fmt", "trim_to_audio", "save_metadata", "vram_flush", "ui_preset", "debug_verbose",
];
const VAE_CURRENT_PY_WIDGET_ORDER = [
    "frame_rate", "decode_mode", "filename_prefix", "output_root", "frames_subdir", "image_format", "jpg_quality",
    "tiled_tile_size", "tiled_overlap", "tiled_temporal_size", "tiled_temporal_overlap", "cleanup_before_decode",
    "crf", "pix_fmt", "trim_to_audio", "save_metadata", "vram_flush", "ui_preset", "debug_verbose",
];
const PLANNER_ROUTE_MODE_VALUES = new Set([
    "single generation (duration only)",
    "choose segment count (audio / segments)",
    "choose seconds per segment (auto count)",
]);
const PLANNER_A2I_BACKEND_MODERN = "modern pipeline";
const PLANNER_A2I_BACKEND_LEGACY = "legacy exact pipeline";
const PLANNER_A2I_MODE_SINGLE = "single generation";
const PLANNER_A2I_MODE_2_SEGMENTS = "2 segments";
const PLANNER_A2I_MODE_3_SEGMENTS = "3 segments";
const PLANNER_A2I_MODE_LOOP = "loop / 4+ segments";
const PLANNER_AUDIO_IMG2VID_BACKEND_VALUES = new Set([PLANNER_A2I_BACKEND_MODERN, PLANNER_A2I_BACKEND_LEGACY, "modern", "legacy"]);
const PLANNER_AUDIO_IMG2VID_MODE_VALUES = new Set([PLANNER_A2I_MODE_SINGLE, PLANNER_A2I_MODE_2_SEGMENTS, PLANNER_A2I_MODE_3_SEGMENTS, PLANNER_A2I_MODE_LOOP, "single", "2_segments", "3_segments", "loop"]);
const PLANNER_MODE_VALUES = new Set(["manual_segment_seconds", "explicit_preset_seconds"]);
const PLANNER_SEGMENT_PRESET_VALUES = new Set(["5sec", "10sec", "15sec", "20sec"]);
const PLANNER_ROUND_MODE_VALUES = new Set(["up", "nearest", "down"]);
const PLANNER_AUDIO_MODE_VALUES = new Set(["melband_vocals_duration_math", "raw_audio_only"]);
const PLANNER_VISUAL_WIDGET_ORDER = [
    "audio_img2vid_backend", "audio_img2vid_mode", "single_duration_seconds", "segment_count", "segment_seconds", "fps",
    "audio_preprocess_mode", "melband_model_name", "planning_mode", "segment_preset",
    "overlap_frames", "ltx_round_mode", "route_mode", "audio_concat_payload", "debug_verbose",
];
const PLANNER_LEGACY_VISUAL_WIDGET_ORDER = [
    "route_mode", "single_duration_seconds", "segment_count", "segment_seconds", "fps",
    "audio_preprocess_mode", "melband_model_name", "planning_mode", "segment_preset",
    "overlap_frames", "ltx_round_mode", "audio_concat_payload", "debug_verbose",
];
const PLANNER_PY_WIDGET_ORDER = [
    "fps", "segment_seconds", "planning_mode", "segment_preset", "overlap_frames", "ltx_round_mode",
    "audio_preprocess_mode", "melband_model_name", "audio_img2vid_backend", "audio_img2vid_mode", "route_mode", "segment_count",
    "single_duration_seconds", "audio_concat_payload", "debug_verbose",
];
const PLANNER_LEGACY_PY_WIDGET_ORDER = [
    "fps", "segment_seconds", "planning_mode", "segment_preset", "overlap_frames", "ltx_round_mode",
    "audio_preprocess_mode", "melband_model_name", "route_mode", "segment_count",
    "single_duration_seconds", "audio_concat_payload", "debug_verbose",
];

function hasWidgetList(node) {
    return !!node && Array.isArray(node.widgets);
}

function findWidget(node, name) {
    if (!hasWidgetList(node)) {
        return undefined;
    }
    return node.widgets.find((widget) => widget?.name === name);
}

function setWidgetVisibility(widget, visible) {
    if (!widget || widget.type === "converted-widget") {
        return;
    }

    widget.hidden = !visible;
    widget.disabled = !visible;

    if (widget.element) {
        widget.element.style.display = visible ? "" : "none";
    }
    if (widget.inputEl) {
        widget.inputEl.style.display = visible ? "" : "none";
    }

    if (visible) {
        if (Object.prototype.hasOwnProperty.call(widget, "__iamccsOrigComputeSize")) {
            widget.computeSize = widget.__iamccsOrigComputeSize;
        } else {
            delete widget.computeSize;
        }
    } else {
        if (!Object.prototype.hasOwnProperty.call(widget, "__iamccsOrigComputeSize")) {
            widget.__iamccsOrigComputeSize = widget.computeSize;
        }
        widget.computeSize = () => [0, -4];
        widget.y = undefined;
        widget.last_y = undefined;
    }
}

function fitNodeToWidgets(node) {
    if (!node) {
        return;
    }
    const size = node.computeSize?.() || node.size;
    if (!Array.isArray(size) || typeof node.setSize !== "function") {
        return;
    }
    node.setSize([
        Math.max(node.size?.[0] || 0, size[0]),
        Math.max(size[1], size[1] + 8),
    ]);
}

function setWidgetLabel(node, widgetName, label) {
    const widget = findWidget(node, widgetName);
    if (!widget) {
        return;
    }
    widget.label = label;
}

function markCanvasDirty() {
    try {
        app.graph?.setDirtyCanvas?.(true, true);
    } catch (error) {
        warnUiError("markCanvasDirty", error);
    }
}

function warnUiError(scope, error) {
    console.warn(`[IAMCCS SuperNodes] UI ${scope} skipped`, error);
}

function safeCall(scope, fn, fallback = undefined) {
    try {
        return fn();
    } catch (error) {
        warnUiError(scope, error);
        return fallback;
    }
}

function areNodeLinksResolved(node) {
    if (!node || !app?.graph || !Array.isArray(node.inputs) || !Array.isArray(node.outputs)) {
        return false;
    }
    const checkLinkId = (linkId) => {
        if (linkId === undefined || linkId === null) {
            return true;
        }
        return !!getGraphLink(linkId);
    };
    for (const input of node.inputs) {
        if (!input) {
            continue;
        }
        if (!checkLinkId(input.link)) {
            return false;
        }
        if (Array.isArray(input.links) && input.links.some((linkId) => !checkLinkId(linkId))) {
            return false;
        }
    }
    for (const output of node.outputs) {
        if (!output || !Array.isArray(output.links)) {
            continue;
        }
        if (output.links.some((linkId) => !checkLinkId(linkId))) {
            return false;
        }
    }
    return true;
}

function isGraphReadyForLinxSync(node) {
    if (!node || !app?.graph) {
        return false;
    }
    if (node.__iamccsConfiguring) {
        return false;
    }
    return Array.isArray(node.inputs) && Array.isArray(node.outputs) && areNodeLinksResolved(node);
}

function runLinxSyncWhenReady(node, scope, fn, options = {}) {
    const maxAttempts = Number.isFinite(options.maxAttempts) ? Math.max(1, Number(options.maxAttempts)) : 12;
    const delayMs = Number.isFinite(options.delayMs) ? Math.max(0, Number(options.delayMs)) : 25;
    const run = (attempt = 0) => safeCall(scope, () => {
        if (!isGraphReadyForLinxSync(node)) {
            if (attempt + 1 < maxAttempts) {
                setTimeout(() => run(attempt + 1), delayMs);
            }
            return;
        }
        fn();
    });
    // Memory fix: never traverse LINX synchronously while LiteGraph is
    // rebuilding a workflow. Widget/index normalization stays synchronous;
    // only link graph sync is deferred to avoid undefined output on reload.
    if (options.defer || !isGraphReadyForLinxSync(node)) {
        setTimeout(() => run(0), delayMs);
        return;
    }
    run(0);
}

function syncReferenceAudioImg2VidTimingWhenReady(renderNode, timingOptions = {}, reason = "reference_timing", options = {}) {
    runLinxSyncWhenReady(renderNode, `syncReferenceAudioImg2VidTiming.${reason}`, () => {
        syncReferenceAudioImg2VidTiming(renderNode, timingOptions);
    }, options);
}

function syncTextAudio2VideoPlannerModeWhenReady(renderNode, reason = "text_audio_planner", options = {}) {
    runLinxSyncWhenReady(renderNode, `syncTextAudio2VideoPlannerMode.${reason}`, () => {
        syncTextAudio2VideoPlannerMode(renderNode);
    }, options);
}

function syncLegacyBackendPlannerModeWhenReady(renderNode, reason = "legacy_backend_planner", options = {}) {
    runLinxSyncWhenReady(renderNode, `syncLegacyBackendPlannerMode.${reason}`, () => {
        syncLegacyBackendPlannerMode(renderNode, options);
    }, options);
}

function syncRenderLinxWhenReady(renderNode, reason = "render", options = {}) {
    runLinxSyncWhenReady(renderNode, `syncRenderLinx.${reason}`, () => {
        syncDownstreamVaeDecodeModes(renderNode);
        syncReferenceAudioImg2VidTiming(renderNode, { applyFpsDefault: false });
        syncTextAudio2VideoPlannerMode(renderNode);
    }, options);
}

function syncPlannerFpsDownstreamWhenReady(plannerNode, reason = "planner", options = {}) {
    runLinxSyncWhenReady(plannerNode, `syncPlannerFpsDownstream.${reason}`, () => {
        syncPlannerFpsDownstream(plannerNode, options);
    }, options);
}

function syncVaeUpstreamWhenReady(vaeNode, reason = "vae", options = {}) {
    runLinxSyncWhenReady(vaeNode, `syncVaeUpstream.${reason}`, () => {
        syncUpstreamFpsFromVae(vaeNode);
        syncUpstreamDecodeFromVae(vaeNode);
    }, options);
}

function getGraphLink(linkId) {
    const links = app.graph?.links;
    if (!links || linkId === undefined || linkId === null) {
        return null;
    }
    const wanted = Number(linkId);
    try {
        if (typeof links.get === "function") {
            return links.get(linkId) || links.get(String(linkId)) || links.get(wanted) || null;
        }
        const direct = links[linkId] || links[String(linkId)];
        if (direct) {
            const directId = Array.isArray(direct) ? direct[0] : direct.id;
            if (directId === undefined || Number(directId) === wanted) {
                return direct;
            }
        }
        if (Array.isArray(links)) {
            return links.find((link) => {
                if (!link) {
                    return false;
                }
                const id = Array.isArray(link) ? link[0] : link.id;
                return Number(id) === wanted;
            }) || null;
        }
    } catch (error) {
        warnUiError("getGraphLink", error);
    }
    return null;
}

function getLinkTargetId(link) {
    if (Array.isArray(link)) {
        return link[3];
    }
    return link?.target_id ?? link?.targetId ?? link?.target?.id ?? link?.target;
}

function getLinkSourceId(link) {
    if (Array.isArray(link)) {
        return link[1];
    }
    return link?.origin_id ?? link?.originId ?? link?.source_id ?? link?.sourceId ?? link?.origin?.id ?? link?.source?.id ?? link?.source;
}

function sectionButtonCaption(group, isExpanded) {
    return `${isExpanded ? "[-]" : "[+]"} ${group.label}`;
}

function sectionButtonStyle(group, isExpanded) {
    const expandedBg = group.color || "#4f6f8f";
    return {
        background: isExpanded ? expandedBg : SECTION_BUTTON_COLLAPSED_BG,
        border: isExpanded ? expandedBg : SECTION_BUTTON_COLLAPSED_BORDER,
        rail: expandedBg,
        text: SECTION_BUTTON_TEXT,
    };
}

function roundedRectPath(ctx, x, y, width, height, radius) {
    const r = Math.max(0, Math.min(radius, height * 0.5, width * 0.5));
    if (typeof ctx.roundRect === "function") {
        ctx.roundRect(x, y, width, height, r);
        return;
    }
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + width - r, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + r);
    ctx.lineTo(x + width, y + height - r);
    ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
    ctx.lineTo(x + r, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
}

function installSectionButtonDraw(button) {
    if (!button || button._iamccsCustomSectionDraw) {
        return;
    }
    button._iamccsCustomSectionDraw = true;
    button.draw = function(ctx, node, widgetWidth, y, widgetHeight) {
        const style = this._iamccsButtonStyle || sectionButtonStyle({ color: "#4f6f8f" }, true);
        const x = 8;
        const height = Math.max(SECTION_BUTTON_HEIGHT - 4, (widgetHeight || SECTION_BUTTON_HEIGHT) - 4);
        const width = Math.max(40, (widgetWidth || node?.size?.[0] || 280) - 16);
        const top = y + 2;
        const label = String(this.label || this.name || "");

        ctx.save();
        ctx.beginPath();
        roundedRectPath(ctx, x, top, width, height, SECTION_BUTTON_RADIUS);
        ctx.fillStyle = style.background;
        ctx.fill();
        ctx.lineWidth = 1;
        ctx.strokeStyle = style.border;
        ctx.stroke();

        ctx.beginPath();
        roundedRectPath(ctx, x, top, 6, height, SECTION_BUTTON_RADIUS);
        ctx.fillStyle = style.rail;
        ctx.fill();

        ctx.fillStyle = style.text;
        ctx.font = "bold 12px sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(label, x + 14, top + height * 0.5);
        ctx.restore();
        return true;
    };
}

function insertWidget(node, widget, index) {
    if (!hasWidgetList(node) || !widget) {
        return;
    }
    const widgets = node.widgets || [];
    const currentIndex = widgets.indexOf(widget);
    if (currentIndex >= 0) {
        widgets.splice(currentIndex, 1);
    }
    widgets.splice(Math.max(0, Math.min(index, widgets.length)), 0, widget);
}

function moveWidgetsAfter(node, anchorName, widgetNames) {
    if (!hasWidgetList(node) || !node.widgets.length) {
        return;
    }
    const moving = [];
    for (const widgetName of widgetNames) {
        const index = node.widgets.findIndex((widget) => widget?.name === widgetName);
        if (index >= 0) {
            moving.push(node.widgets.splice(index, 1)[0]);
        }
    }
    if (!moving.length) {
        return;
    }
    const anchorIndex = node.widgets.findIndex((widget) => widget?.name === anchorName);
    const insertAt = anchorIndex >= 0 ? anchorIndex + 1 : node.widgets.length;
    node.widgets.splice(insertAt, 0, ...moving);
}

function reorderWidgetsByName(node, orderedNames) {
    if (!hasWidgetList(node) || !node.widgets.length) {
        return;
    }
    const ordered = [];
    const used = new Set();
    for (const name of orderedNames) {
        const index = node.widgets.findIndex((widget) => widget?.name === name && !used.has(widget));
        if (index >= 0) {
            ordered.push(node.widgets[index]);
            used.add(node.widgets[index]);
        }
    }
    const rest = node.widgets.filter((widget) => !used.has(widget));
    node.widgets.splice(0, node.widgets.length, ...ordered, ...rest);
}

function normalizePlannerWidgetOrder(node) {
    if (node?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Planner") {
        return;
    }
    reorderWidgetsByName(node, [
        "audio_img2vid_backend",
        "audio_img2vid_mode",
        "single_duration_seconds",
        "segment_count",
        "segment_seconds",
        "fps",
        "audio_preprocess_mode",
        "melband_model_name",
        "planning_mode",
        "segment_preset",
        "overlap_frames",
        "ltx_round_mode",
        "route_mode",
        "audio_concat_payload",
    ]);
}

function normalizeRenderWidgetOrder(node) {
    if (node?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        return;
    }
    moveWidgetsAfter(node, "sigma_terminal", ["show_manual_sigmas", "manual_sigmas"]);
    moveWidgetsAfter(node, "start_frames_rule", ["color_match_mode", "color_match_strength"]);
    moveWidgetsAfter(node, "second_stage_mode", ["stage2_model_policy", "second_stage_upscale_model", "second_stage_reinject_strength", "second_stage_cfg", "second_stage_manual_sigmas"]);
    moveWidgetsAfter(node, "debug_verbose", ["taeltx_preview", "taeltx_preview_max_frames", "taeltx_preview_fps"]);
}

function normalizeVaeWidgetOrder(node) {
    if (node?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec VAE") {
        return;
    }
    moveWidgetsAfter(node, "decode_mode", ["tiled_tile_size", "tiled_overlap", "tiled_temporal_size", "tiled_temporal_overlap", "cleanup_before_decode"]);
    moveWidgetsAfter(node, "cleanup_before_decode", ["frame_rate", "frames_subdir", "image_format", "jpg_quality", "output_root", "filename_prefix"]);
    moveWidgetsAfter(node, "filename_prefix", ["crf", "pix_fmt", "trim_to_audio", "save_metadata", "vram_flush", "debug_verbose"]);
}

function setWidgetValue(node, widgetName, value, options = {}) {
    const widget = findWidget(node, widgetName);
    if (!widget) {
        return;
    }
    let nextValue = value;
    const values = Array.isArray(widget.options?.values) ? widget.options.values : [];
    if (typeof value === "string" && values.length > 0 && !values.includes(value)) {
        const wanted = value.toLowerCase();
        const wantedStem = wanted.replace(/\.[^.]+$/, "");
        const matched = values.find((option) => String(option).toLowerCase() === wanted)
            || values.find((option) => String(option).toLowerCase().includes(wantedStem));
        if (matched !== undefined) {
            nextValue = matched;
        }
    }
    if (widget.value === nextValue) {
        return;
    }
    widget.value = nextValue;
    if (options.notify !== false) {
        widget.callback?.(nextValue);
    }
}

function coerceWidgetCallbackValue(widget, args) {
    if (!widget || !args?.length) {
        return;
    }
    const value = args[0];
    if (value === undefined || value === widget || value?.name) {
        return;
    }
    widget.value = value;
}

function readFiniteWidgetNumber(node, widgetName, fallback = null) {
    const value = Number(findWidget(node, widgetName)?.value);
    return Number.isFinite(value) ? value : fallback;
}

const EXEC_PLANNER_PRESETS = {
    "5sec": { seconds: 5.0, overlap: 9 },
    "10sec": { seconds: 10.0, overlap: 9 },
    "15sec": { seconds: 15.0, overlap: 9 },
    "20sec": { seconds: 20.0, overlap: 9 },
    videoclip: { seconds: 10.0, overlap: 9 },
    monologue: { seconds: 15.0, overlap: 9 },
};

function getExecPlannerPreset(value) {
    return EXEC_PLANNER_PRESETS[String(value || "15sec")] || EXEC_PLANNER_PRESETS["15sec"];
}

function getExecPlannerPresetForSeconds(seconds) {
    const value = Number(seconds);
    if (!Number.isFinite(value)) {
        return null;
    }
    for (const presetName of ["5sec", "10sec", "15sec", "20sec"]) {
        const rec = EXEC_PLANNER_PRESETS[presetName];
        if (Math.abs(value - rec.seconds) <= 0.001) {
            return rec;
        }
    }
    return null;
}

function getExecPlannerPresetNameForSeconds(seconds) {
    const value = Number(seconds);
    if (!Number.isFinite(value)) {
        return "10sec";
    }
    for (const presetName of ["5sec", "10sec", "15sec", "20sec"]) {
        const rec = EXEC_PLANNER_PRESETS[presetName];
        if (Math.abs(value - rec.seconds) <= 0.001) {
            return presetName;
        }
    }
    return "10sec";
}

function ltxSafeFramesAtLeast(frameCount) {
    const frames = Math.max(1, Math.round(Number(frameCount) || 1));
    const remainder = (frames - 1) % 8;
    return remainder === 0 ? frames : frames + (8 - remainder);
}

function readPlannerNumber(node, widgetName, fallback) {
    const value = Number(findWidget(node, widgetName)?.value);
    return Number.isFinite(value) ? value : fallback;
}

function getExecPlannerLiveConfig(node) {
    const fps = Math.max(0.001, readPlannerNumber(node, "fps", 24.0));
    const routeMode = String(findWidget(node, "route_mode")?.value || "choose segment count (audio / segments)");
    const planningMode = String(findWidget(node, "planning_mode")?.value || "manual_segment_seconds");
    const segmentPreset = String(findWidget(node, "segment_preset")?.value || "10sec");
    const presetRec = getExecPlannerPreset(segmentPreset);
    const singleDuration = Math.max(0.1, readPlannerNumber(node, "single_duration_seconds", 10.0));
    const segmentCount = Math.max(1, Math.round(readPlannerNumber(node, "segment_count", 2)));
    let seconds = routeMode.includes("single") ? singleDuration : Math.max(0.01, readPlannerNumber(node, "segment_seconds", presetRec.seconds));
    const overlap = Math.max(0, Math.round(readPlannerNumber(node, "overlap_frames", 9)));
    let overlapSource = routeMode.includes("segment count") ? `${segmentCount} segment request` : "manual/default";

    if (routeMode.includes("seconds per segment")) {
        const secondsRec = getExecPlannerPresetForSeconds(seconds);
        if (secondsRec) {
            overlapSource = `${secondsRec.seconds}s duration`;
        }
    }

    return { fps, planningMode, segmentPreset, seconds, overlap, overlapSource, routeMode, segmentCount, singleDuration };
}

function updateExecPlannerLivePreview(node) {
    if (node?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Planner") {
        return;
    }
    node.properties = node.properties || {};
    const config = getExecPlannerLiveConfig(node);
    const uniqueFrames = Math.max(1, Math.round(config.seconds * config.fps));
    const firstRaw = ltxSafeFramesAtLeast(uniqueFrames);
    const continuationTarget = uniqueFrames + config.overlap;
    const continuationRaw = ltxSafeFramesAtLeast(continuationTarget);
    const lines = [
        `live ${config.seconds.toFixed(2)}s @ ${config.fps.toFixed(2)}fps -> ${uniqueFrames}f unique`,
        `overlap ${config.overlap}f (${config.overlapSource}) | first raw ${firstRaw}f | cont raw ${continuationRaw}f`,
    ];

    const storedDuration = Number(node.properties.iamccsPlannerDuration);
    const storedTotalFrames = Number(node.properties.iamccsPlannerTotalFrames);
    const hasDuration = Number.isFinite(storedDuration) && storedDuration > 0;
    const totalFrames = Number.isFinite(storedTotalFrames) && storedTotalFrames > 0
        ? Math.round(storedTotalFrames)
        : hasDuration
            ? Math.round(storedDuration * config.fps)
            : 0;

    if (totalFrames > 0) {
        const segmentCount = Math.max(1, Math.ceil(totalFrames / uniqueFrames));
        lines.push(`audio ${hasDuration ? storedDuration.toFixed(2) : "cached"}s -> ${totalFrames}f | segments ${segmentCount}`);
        const parts = [];
        for (let index = 0; index < Math.min(segmentCount, 4); index += 1) {
            const remaining = Math.max(0, totalFrames - index * uniqueFrames);
            const effective = Math.max(1, Math.min(uniqueFrames, remaining));
            const target = index === 0 ? effective : effective + config.overlap;
            parts.push(`s${index + 1}:${effective}${index === 0 ? "" : `+${config.overlap}`}->${ltxSafeFramesAtLeast(target)}`);
        }
        lines.push(`${parts.join(" ")}${segmentCount > 4 ? " ..." : ""}`);
    } else {
        lines.push("audio duration unknown until planner runs once");
    }

    node.properties.iamccsPlannerLivePreview = lines.join("\n");
}

function syncExecPlannerExplicitPreset(node, sourceWidgetName = "") {
    if (node?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Planner") {
        return;
    }
    const planningMode = String(findWidget(node, "planning_mode")?.value || "manual_segment_seconds");
    if (planningMode === "explicit_preset_seconds") {
        const rec = getExecPlannerPreset(findWidget(node, "segment_preset")?.value);
        setWidgetValue(node, "segment_seconds", rec.seconds);
        if (sourceWidgetName === "planning_mode" || sourceWidgetName === "segment_preset") {
            setWidgetValue(node, "overlap_frames", rec.overlap);
        }
    }
    updateExecPlannerLivePreview(node);
}

function installExecPlannerExplicitPresetSync(node) {
    if (!node) {
        return;
    }
    if (node._iamccsExecPlannerExplicitPresetSyncInstalled) {
        syncExecPlannerExplicitPreset(node);
        updateExecPlannerLivePreview(node);
        return;
    }
    node._iamccsExecPlannerExplicitPresetSyncInstalled = true;
    for (const widgetName of ["fps", "audio_img2vid_backend", "audio_img2vid_mode", "route_mode", "single_duration_seconds", "segment_count", "segment_seconds", "planning_mode", "segment_preset", "overlap_frames", "ltx_round_mode", "audio_preprocess_mode"]) {
        const widget = findWidget(node, widgetName);
        if (!widget) {
            continue;
        }
        const originalCallback = widget.callback;
        widget.callback = (...args) => {
            originalCallback?.apply(widget, args);
            if (widgetName === "planning_mode" || widgetName === "route_mode" || widgetName === "audio_preprocess_mode" || widgetName === "audio_img2vid_backend" || widgetName === "audio_img2vid_mode") {
                if (widgetName === "audio_img2vid_backend" || widgetName === "audio_img2vid_mode") {
                    setWidgetValue(node, "audio_img2vid_backend", plannerA2IBackendDisplay(findWidget(node, "audio_img2vid_backend")?.value), { notify: false });
                    setWidgetValue(node, "audio_img2vid_mode", plannerA2IModeDisplay(findWidget(node, "audio_img2vid_mode")?.value), { notify: false });
                    syncPlannerA2IModeToInternalWidgets(node);
                    if (normalizePlannerA2IBackend(findWidget(node, "audio_img2vid_backend")?.value) === "legacy") {
                        setWidgetValue(node, "fps", 24, { notify: false });
                        setWidgetValue(node, "audio_preprocess_mode", "raw_audio_only", { notify: false });
                        setWidgetValue(node, "planning_mode", "manual_segment_seconds", { notify: false });
                        setWidgetValue(node, "overlap_frames", 9, { notify: false });
                        setWidgetValue(node, "ltx_round_mode", "up", { notify: false });
                    }
                }
                applyPlannerModeVisibility(node);
            } else {
                syncExecPlannerExplicitPreset(node, widgetName);
            }
            updateExecPlannerLivePreview(node);
            const syncOptions = (widgetName === "audio_img2vid_backend" || widgetName === "audio_img2vid_mode")
                ? { defer: true, applyRenderDefaults: true }
                : {};
            syncPlannerFpsDownstreamWhenReady(node, "planner_widget", syncOptions);
            markCanvasDirty();
        };
    }
    syncExecPlannerExplicitPreset(node);
    updateExecPlannerLivePreview(node);
    syncPlannerFpsDownstreamWhenReady(node, "planner_install", { defer: true });
}

function getLinxTargets(node) {
    const results = [];
    try {
        for (const output of Array.isArray(node?.outputs) ? node.outputs : []) {
            if (!output || output.type !== "IAMCCS_SUPERNODE_LINX") {
                continue;
            }
            const linkIds = Array.isArray(output.links)
                ? output.links.filter((linkId) => linkId !== undefined && linkId !== null)
                : [];
            for (const linkId of linkIds) {
                const link = getGraphLink(linkId);
                if (!link) {
                    continue;
                }
                const targetId = getLinkTargetId(link);
                if (targetId === undefined || targetId === null) {
                    continue;
                }
                const targetNode = app.graph?.getNodeById?.(targetId);
                if (targetNode) {
                    results.push(targetNode);
                }
            }
        }
    } catch (error) {
        console.warn("[IAMCCS SuperNodes] Ignored invalid LINX link while syncing UI", error);
    }
    return results;
}

function getLinxSources(node) {
    const results = [];
    try {
        for (const input of Array.isArray(node?.inputs) ? node.inputs : []) {
            if (!input || (input.type && input.type !== "IAMCCS_SUPERNODE_LINX")) {
                continue;
            }
            const linkIds = [];
            if (input.link !== undefined && input.link !== null) {
                linkIds.push(input.link);
            }
            if (Array.isArray(input.links)) {
                linkIds.push(...input.links.filter((linkId) => linkId !== undefined && linkId !== null));
            }
            for (const linkId of linkIds) {
                const link = getGraphLink(linkId);
                if (!link) {
                    continue;
                }
                const sourceId = getLinkSourceId(link);
                if (sourceId === undefined || sourceId === null) {
                    continue;
                }
                const sourceNode = app.graph?.getNodeById?.(sourceId);
                if (sourceNode) {
                    results.push(sourceNode);
                }
            }
        }
    } catch (error) {
        console.warn("[IAMCCS SuperNodes] Ignored invalid LINX source while syncing UI", error);
    }
    return results;
}

function getUpstreamLinxNodes(startNode) {
    const results = [];
    const visited = new Set();
    const queue = [...getLinxSources(startNode)];
    while (queue.length > 0) {
        const node = queue.shift();
        if (!node || visited.has(node.id)) {
            continue;
        }
        visited.add(node.id);
        results.push(node);
        queue.push(...getLinxSources(node));
    }
    return results;
}

function getRenderResolvedFps(renderNode) {
    if (!renderNode || renderNode?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        return null;
    }
    if (renderUsesGeneratedDuration(renderNode)) {
        return readFiniteWidgetNumber(renderNode, "generated_media_fps", DEFAULT_GENERATED_FPS);
    }
    const plannerNode = getUpstreamLinxNodes(renderNode).find((node) => node?.comfyClass === "IAMCCS-SuperNodes AU+IMG2VID Exec Planner");
    const plannerFps = readFiniteWidgetNumber(plannerNode, "fps", null);
    return Number.isFinite(plannerFps)
        ? plannerFps
        : (renderUsesReferenceAudioImg2Vid(renderNode) ? REFERENCE_AUDIO_IMG2VID_FPS : null);
}

function renderUsesReferenceAudioImg2Vid(renderNode) {
    if (!renderNode || renderNode?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        return false;
    }
    return String(findWidget(renderNode, "generation_type")?.value || "") === REFERENCE_AUDIO_IMG2VID_TYPE;
}

function renderUsesTextAudio2Video(renderNode) {
    if (!renderNode || renderNode?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        return false;
    }
    return String(findWidget(renderNode, "generation_type")?.value || "") === "text+audio2video";
}

function syncTextAudio2VideoPlannerMode(renderNode) {
    if (!renderUsesTextAudio2Video(renderNode)) {
        return;
    }
    for (const node of getUpstreamLinxNodes(renderNode)) {
        if (node?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Planner") {
            continue;
        }
        const segmentSeconds = readPlannerNumber(node, "segment_seconds", 10.0);
        const currentSingleDuration = readPlannerNumber(node, "single_duration_seconds", NaN);
        setWidgetValue(node, "audio_img2vid_backend", PLANNER_A2I_BACKEND_MODERN, { notify: false });
        setWidgetValue(node, "audio_img2vid_mode", PLANNER_A2I_MODE_SINGLE, { notify: false });
        setWidgetValue(node, "route_mode", "single generation (duration only)", { notify: false });
        setWidgetValue(node, "segment_count", 1, { notify: false });
        if (!Number.isFinite(currentSingleDuration) || currentSingleDuration <= 0) {
            setWidgetValue(node, "single_duration_seconds", segmentSeconds, { notify: false });
        }
        applyPlannerModeVisibility(node);
        updateExecPlannerLivePreview(node);
        syncPlannerFpsDownstreamWhenReady(node, "text_audio_generation_type", { defer: true });
    }
}

function syncLegacyBackendPlannerMode(renderNode, options = {}) {
    if (!renderUsesReferenceAudioImg2Vid(renderNode)) {
        return;
    }
    const applyDefaults = options.applyDefaults === true;
    const backendMode = normalizeRenderBackendValue(findWidget(renderNode, "backend_mode")?.value);
    if (!LEGACY_RENDER_BACKEND_DEFAULTS[backendMode]) {
        return;
    }
    for (const node of getUpstreamLinxNodes(renderNode)) {
        if (node?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Planner") {
            continue;
        }
        const currentSegmentSeconds = Math.max(0.01, readPlannerNumber(node, "segment_seconds", 10.0));
        const currentSingleDuration = Math.max(0.1, readPlannerNumber(node, "single_duration_seconds", currentSegmentSeconds));
        const currentSegmentCount = Math.max(1, Math.round(readPlannerNumber(node, "segment_count", 2)));
        const segmentSeconds = Number.isFinite(currentSegmentSeconds) ? currentSegmentSeconds : 10.0;
        const singleDuration = Number.isFinite(currentSingleDuration) ? currentSingleDuration : segmentSeconds;
        let targetSegmentCount = currentSegmentCount;
        setWidgetValue(node, "audio_img2vid_backend", PLANNER_A2I_BACKEND_LEGACY, { notify: false });
        if (backendMode === "legacy_single") {
            setWidgetValue(node, "audio_img2vid_mode", PLANNER_A2I_MODE_SINGLE, { notify: false });
            setWidgetValue(node, "route_mode", "single generation (duration only)", { notify: false });
            setWidgetValue(node, "segment_count", 1, { notify: false });
            setWidgetValue(node, "single_duration_seconds", singleDuration, { notify: false });
        } else {
            if (backendMode === "legacy_two_segments") {
                targetSegmentCount = 2;
                setWidgetValue(node, "audio_img2vid_mode", PLANNER_A2I_MODE_2_SEGMENTS, { notify: false });
            } else {
                targetSegmentCount = Math.max(2, currentSegmentCount || 2);
                setWidgetValue(node, "audio_img2vid_mode", PLANNER_A2I_MODE_LOOP, { notify: false });
            }
            setWidgetValue(node, "route_mode", "choose segment count (audio / segments)", { notify: false });
            setWidgetValue(node, "segment_count", targetSegmentCount, { notify: false });
            setWidgetValue(node, "segment_seconds", segmentSeconds, { notify: false });
        }
        if (applyDefaults) {
            setWidgetValue(node, "fps", 24, { notify: false });
            setWidgetValue(node, "planning_mode", "manual_segment_seconds", { notify: false });
            setWidgetValue(node, "segment_preset", getExecPlannerPresetNameForSeconds(segmentSeconds), { notify: false });
            setWidgetValue(node, "overlap_frames", 9, { notify: false });
            setWidgetValue(node, "ltx_round_mode", "up", { notify: false });
            setWidgetValue(node, "audio_preprocess_mode", "raw_audio_only", { notify: false });
        }
        applyPlannerModeVisibility(node, { skipPresetSync: true });
        updateExecPlannerLivePreview(node);
        syncPlannerFpsDownstreamWhenReady(node, `legacy_backend_${backendMode}`, { defer: true });
    }
}

function syncReferenceAudioImg2VidTiming(renderNode, options = {}) {
    if (!renderUsesReferenceAudioImg2Vid(renderNode)) {
        return;
    }
    const applyFpsDefault = options.applyFpsDefault === true;
    const renderBackendMode = normalizeRenderBackendValue(findWidget(renderNode, "backend_mode")?.value);
    const usesLegacyBackend = Boolean(LEGACY_RENDER_BACKEND_DEFAULTS[renderBackendMode]);
    const defaultFps = usesLegacyBackend ? 24.0 : REFERENCE_AUDIO_IMG2VID_FPS;
    let resolvedFps = null;
    for (const node of getUpstreamLinxNodes(renderNode)) {
        if (node?.comfyClass === "IAMCCS-SuperNodes AU+IMG2VID Exec Planner") {
            if (applyFpsDefault) {
                setWidgetValue(node, "fps", defaultFps, { notify: false });
            }
            resolvedFps = readFiniteWidgetNumber(node, "fps", resolvedFps);
            applyPlannerModeVisibility(node, { skipPresetSync: true });
            updateExecPlannerLivePreview(node);
        }
    }
    if (!Number.isFinite(resolvedFps) && applyFpsDefault) {
        resolvedFps = defaultFps;
    }
    const visited = new Set();
    const queue = [...getLinxTargets(renderNode)];
    while (queue.length > 0) {
        const node = queue.shift();
        if (!node || visited.has(node.id)) {
            continue;
        }
        visited.add(node.id);
        if (node?.comfyClass === "IAMCCS-SuperNodes AU+IMG2VID Exec VAE") {
            if (Number.isFinite(resolvedFps)) {
                setWidgetValue(node, "frame_rate", resolvedFps, { notify: false });
            }
            if (applyFpsDefault) {
                setWidgetValue(node, "decode_mode", "normal_tiled_iamccs", { notify: false });
                setWidgetValue(node, "trim_to_audio", usesLegacyBackend, { notify: false });
                applyVaeDecodeModeDefaults(node, "normal_tiled_iamccs");
                applyVaeDecodeModeVisibility(node);
            }
            continue;
        }
        if (node?.comfyClass === "IAMCCS-SuperNodes Second Stage") {
            queue.push(...getLinxTargets(node));
        }
    }
}

function syncUpstreamFpsFromVae(vaeNode) {
    if (vaeNode?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec VAE") {
        return;
    }
    const frameRate = readFiniteWidgetNumber(vaeNode, "frame_rate", null);
    if (!Number.isFinite(frameRate)) {
        return;
    }
    for (const node of getUpstreamLinxNodes(vaeNode)) {
        if (node?.comfyClass === "IAMCCS-SuperNodes AU+IMG2VID Exec Render" && (renderUsesGeneratedDuration(node) || renderUsesReferenceAudioImg2Vid(node))) {
            setWidgetValue(node, "generated_media_fps", frameRate, { notify: false });
            sanitizeRenderWidgetValues(node);
            applyRenderGeneratedDurationVisibility(node);
            applyRenderInternalWidgetVisibility(node);
        }
        if (node?.comfyClass === "IAMCCS-SuperNodes AU+IMG2VID Exec Planner") {
            setWidgetValue(node, "fps", frameRate, { notify: false });
            updateExecPlannerLivePreview(node);
        }
    }
}

function syncPlannerFpsDownstream(plannerNode, options = {}) {
    if (plannerNode?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Planner") {
        return;
    }
    const visited = new Set();
    const queue = [...getLinxTargets(plannerNode)];
    while (queue.length > 0) {
        const node = queue.shift();
        if (!node || visited.has(node.id)) {
            continue;
        }
        visited.add(node.id);
        if (node?.comfyClass === "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
            if (renderUsesReferenceAudioImg2Vid(node)) {
                const backendMode = resolvePlannerA2IRenderBackend(plannerNode);
                const backendWidget = findWidget(node, "backend_mode");
                const previousBackend = normalizeRenderBackendValue(backendWidget?.value);
                if (backendWidget && (previousBackend !== backendMode || options.applyRenderDefaults === true)) {
                    setWidgetValue(node, "backend_mode", backendMode, { notify: false });
                    applyRenderBackendModeDefaults(node, backendMode);
                }
                sanitizeRenderWidgetValues(node);
                applyRenderInternalWidgetVisibility(node);
            }
            syncRenderLinxWhenReady(node, "planner_downstream_render", { defer: true });
            continue;
        }
        queue.push(...getLinxTargets(node));
    }
}

function syncUpstreamDecodeFromVae(vaeNode) {
    if (vaeNode?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec VAE") {
        return;
    }
    const decodeMode = String(findWidget(vaeNode, "decode_mode")?.value || "");
    if (!VAE_DECODE_MODE_VALUES.has(decodeMode) || decodeMode === "inherit_render_backend") {
        return;
    }
    for (const node of getUpstreamLinxNodes(vaeNode)) {
        if (node?.comfyClass === "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
            setWidgetValue(node, "vae_mode", decodeMode, { notify: false });
            sanitizeRenderWidgetValues(node);
            applyRenderInternalWidgetVisibility(node);
        }
    }
}

function applyVaeDecodeModeVisibility(node) {
    if (node?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec VAE") {
        return;
    }
    normalizeVaeWidgetOrder(node);
    const decodeMode = String(findWidget(node, "decode_mode")?.value || "normal_tiled_iamccs");
    const normalizedMode = decodeMode === "normal_tiled_vhs" ? "normal_tiled_iamccs" : decodeMode;
    const isLowRam = normalizedMode === "low_ram" || normalizedMode === "low_ram_disk";
    const isVeryLowRam = normalizedMode === "very_low_ram" || normalizedMode === "very_low_ram_disk";
    const isHighVram = normalizedMode === "high_vram";
    const isTiled = normalizedMode === "normal_tiled_iamccs" || normalizedMode === "custom_mode" || normalizedMode === "inherit_render_backend";
    const supportsDecodeSettings = isTiled || isLowRam || isVeryLowRam;
    const modeExpanded = node.properties?.iamccs_section_mode !== false;
    const outputExpanded = !!node.properties?.iamccs_section_output;

    setWidgetVisibility(findWidget(node, "ui_preset"), false);
    setWidgetVisibility(findWidget(node, "pix_fmt"), false);
    setWidgetVisibility(findWidget(node, "save_metadata"), false);

    setWidgetLabel(node, "decode_mode", "VAE Mode");
    setWidgetLabel(node, "frame_rate", "FPS");
    setWidgetLabel(node, "tiled_tile_size", isVeryLowRam ? "Very Low Tile Size" : isLowRam ? "Low RAM Tile Size" : "Tile Size");
    setWidgetLabel(node, "tiled_overlap", isVeryLowRam ? "Very Low Overlap" : isLowRam ? "Low RAM Overlap" : "Tile Overlap");
    setWidgetLabel(node, "tiled_temporal_size", isVeryLowRam ? "Very Low Temporal Size" : isLowRam ? "Low RAM Temporal Size" : "Temporal Size");
    setWidgetLabel(node, "tiled_temporal_overlap", isVeryLowRam ? "Very Low Temporal Overlap" : isLowRam ? "Low RAM Temporal Overlap" : "Temporal Overlap");
    setWidgetLabel(node, "cleanup_before_decode", "Cleanup Before Decode");
    setWidgetLabel(node, "frames_subdir", isVeryLowRam ? "Very Low Frames Folder" : "Low RAM Frames Folder");
    setWidgetLabel(node, "image_format", "Disk Frame Format");
    setWidgetLabel(node, "jpg_quality", "Disk JPG Quality");
    setWidgetLabel(node, "output_root", isVeryLowRam ? "Very Low Output Root" : "Low RAM Output Root");
    setWidgetLabel(node, "vram_flush", "Flush VRAM");
    setWidgetLabel(node, "debug_verbose", "Verbose Debug");

    setWidgetVisibility(findWidget(node, "decode_mode"), modeExpanded);
    setWidgetVisibility(findWidget(node, "frame_rate"), modeExpanded);
    setWidgetVisibility(findWidget(node, "tiled_tile_size"), modeExpanded && supportsDecodeSettings);
    setWidgetVisibility(findWidget(node, "tiled_overlap"), modeExpanded && supportsDecodeSettings);
    setWidgetVisibility(findWidget(node, "tiled_temporal_size"), modeExpanded && supportsDecodeSettings);
    setWidgetVisibility(findWidget(node, "tiled_temporal_overlap"), modeExpanded && supportsDecodeSettings);
    setWidgetVisibility(findWidget(node, "cleanup_before_decode"), modeExpanded && supportsDecodeSettings && !isHighVram);
    setWidgetVisibility(findWidget(node, "frames_subdir"), modeExpanded && (isLowRam || isVeryLowRam));
    setWidgetVisibility(findWidget(node, "image_format"), modeExpanded && (isLowRam || isVeryLowRam));
    setWidgetVisibility(findWidget(node, "jpg_quality"), modeExpanded && (isLowRam || isVeryLowRam));
    setWidgetVisibility(findWidget(node, "output_root"), modeExpanded && (isLowRam || isVeryLowRam));
    setWidgetVisibility(findWidget(node, "vram_flush"), modeExpanded && (isLowRam || isVeryLowRam));
    setWidgetVisibility(findWidget(node, "filename_prefix"), outputExpanded);
    setWidgetVisibility(findWidget(node, "crf"), outputExpanded);
    setWidgetVisibility(findWidget(node, "trim_to_audio"), outputExpanded);
    setWidgetVisibility(findWidget(node, "debug_verbose"), outputExpanded);

    for (const stale of ["decode_settings"]) {
        const staleButton = node.widgets?.find?.((widget) => widget?._iamccsSectionKey === stale);
        if (staleButton) {
            setWidgetVisibility(staleButton, false);
        }
    }
    fitNodeToWidgets(node);
}

function applyVaeDecodeModeDefaults(node, modeOverride = null) {
    if (node?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec VAE") {
        return;
    }
    const decodeMode = String(modeOverride || findWidget(node, "decode_mode")?.value || "normal_tiled_iamccs");
    if (modeOverride !== null) {
        const widget = findWidget(node, "decode_mode");
        if (widget) {
            widget.value = decodeMode;
        }
    }
    const defaults = VAE_DECODE_MODE_DEFAULTS[decodeMode];
    if (!defaults) {
        return;
    }
    Object.entries(defaults).forEach(([widgetName, widgetValue]) => setWidgetValue(node, widgetName, widgetValue));
}

function applyRenderAnchorModeDefaults(node) {
    if (node?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        return;
    }
    const anchorMode = String(findWidget(node, "continuity_anchor_mode")?.value || "off");
    const defaults = ANCHOR_MODE_DEFAULTS[anchorMode];
    if (!defaults) {
        return;
    }
    Object.entries(defaults).forEach(([widgetName, widgetValue]) => setWidgetValue(node, widgetName, widgetValue));
}

function normalizeRenderBackendValue(value) {
    const text = String(value || "auto");
    if (text === "legacy backend") {
        return "legacy_single";
    }
    return text;
}

function applyRenderBackendModeDefaults(node, modeOverride = null) {
    if (node?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        return;
    }
    const generationType = String(findWidget(node, "generation_type")?.value || "");
    const backendMode = normalizeRenderBackendValue(modeOverride ?? findWidget(node, "backend_mode")?.value);
    if (GENERATED_DURATION_TYPES.has(generationType)) {
        if (backendMode === "ti2v_incremental_advanced") {
            setWidgetValue(node, "second_stage_mode", "latent_upscale_refine_x2_beta", { notify: false });
            setWidgetValue(node, "stage2_model_policy", "keep_stage1_model", { notify: false });
            setWidgetValue(node, "second_stage_upscale_model", REF_LTX23_UPSCALE_MODEL, { notify: false });
            setWidgetValue(node, "second_stage_reinject_strength", generationType === "text2video" ? 0.0 : 1.0, { notify: false });
            setWidgetValue(node, "second_stage_cfg", 1.0, { notify: false });
            setWidgetValue(node, "second_stage_manual_sigmas", REF_STAGE2_SIGMAS, { notify: false });
        } else if (backendMode === "single_best") {
            setWidgetValue(node, "second_stage_mode", "off", { notify: false });
            setWidgetValue(node, "second_stage_reinject_strength", 0.0, { notify: false });
        }
        return;
    }
    if (generationType !== REFERENCE_AUDIO_IMG2VID_TYPE) {
        return;
    }
    const defaults = LEGACY_RENDER_BACKEND_DEFAULTS[backendMode];
    if (!defaults) {
        return;
    }
    const backendWidget = findWidget(node, "backend_mode");
    if (backendWidget && backendWidget.value !== backendMode) {
        backendWidget.value = backendMode;
    }
    Object.entries(defaults).forEach(([widgetName, widgetValue]) => setWidgetValue(node, widgetName, widgetValue, { notify: false }));
}

function applyRenderAnchorVisibility(node) {
    if (node?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        return;
    }
    const anchorMode = String(findWidget(node, "continuity_anchor_mode")?.value || "off");
    const antiDriftMode = String(findWidget(node, "anti_drift_mode")?.value || "off");
    const sectionExpanded = !!node.properties?.iamccs_section_latent_refresh;
    const anchorsEnabled = sectionExpanded && anchorMode !== "off";
    const periodicAnchor = anchorMode.startsWith("periodic_");
    setWidgetVisibility(findWidget(node, "continuity_anchor_mode"), sectionExpanded);
    setWidgetVisibility(findWidget(node, "anchor_refresh_interval"), anchorsEnabled && periodicAnchor);
    setWidgetVisibility(findWidget(node, "anchor_image_strength"), anchorsEnabled);
    setWidgetVisibility(findWidget(node, "anti_drift_mode"), anchorsEnabled);
    setWidgetVisibility(findWidget(node, "anti_drift_strength"), anchorsEnabled && antiDriftMode !== "off");
    setWidgetVisibility(findWidget(node, "identity_persistence_strength"), anchorsEnabled && antiDriftMode === "dual_reference_adain");
    fitNodeToWidgets(node);
}

function applyRenderAnchorLabels(node) {
    if (node?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        return;
    }
    setWidgetLabel(node, "backend_mode", "Render Backend");
    setWidgetLabel(node, "generation_mode", "Generation Mode");
    setWidgetLabel(node, "media_mode", "Media Mode");
    setWidgetLabel(node, "generation_type", "Generation Type");
    setWidgetLabel(node, "generated_media_duration_seconds", "Generated Duration");
    setWidgetLabel(node, "generated_media_fps", "Generated FPS");
    setWidgetLabel(node, "stitch_preset", "Stitch Preset");
    setWidgetLabel(node, "overlap_side", "Overlap Side");
    setWidgetLabel(node, "overlap_mode", "Blend Mode");
    setWidgetLabel(node, "start_frames_rule", "Start Frames Rule");
    setWidgetLabel(node, "color_match_mode", "Color Match");
    setWidgetLabel(node, "color_match_strength", "Color Strength");
    setWidgetLabel(node, "audio_context_mode", "Audio Context");
    setWidgetLabel(node, "audio_left_context_s", "Left Context (s)");
    setWidgetLabel(node, "audio_right_context_s", "Right Context (s)");
    setWidgetLabel(node, "continuity_anchor_mode", "Latent Refresh Mode");
    setWidgetLabel(node, "anchor_refresh_interval", "Refresh Every N Segments");
    setWidgetLabel(node, "anchor_image_strength", "Reference Strength");
    setWidgetLabel(node, "motion_intensity", "Motion Intensity");
    setWidgetLabel(node, "vram_flush", "VRAM Flush");
    setWidgetLabel(node, "second_stage_mode", "Second Stage");
    setWidgetLabel(node, "stage2_model_policy", "Stage2 Model Policy");
    setWidgetLabel(node, "second_stage_upscale_model", "2x Upscale Model");
    setWidgetLabel(node, "second_stage_reinject_strength", "Anchor Reinject Strength");
    setWidgetLabel(node, "debug_verbose", "Verbose Debug");
}

function markRenderGenerationTypeDefaultsApplied(node) {
    if (node?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        return;
    }
    node.properties = node.properties || {};
    const generationType = String(findWidget(node, "generation_type")?.value || "audio+image2video");
    node.properties.iamccs_last_generation_type = generationType;
    node.properties.iamccs_generation_defaults_applied = true;
}

function syncRenderGenerationType(node, options = {}) {
    if (node?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        return;
    }
    node.properties = node.properties || {};
    const generationType = String(findWidget(node, "generation_type")?.value || "audio+image2video");
    const alreadyApplied = node.properties.iamccs_generation_defaults_applied === true;
    const lastGenerationType = String(node.properties.iamccs_last_generation_type || "");
    if (!options.force && alreadyApplied && lastGenerationType === generationType) {
        syncReferenceAudioImg2VidTimingWhenReady(node, { applyFpsDefault: false }, "generation_type_unchanged", { defer: true });
        return;
    }
    const config = GENERATION_TYPE_CONFIGS[generationType] || GENERATION_TYPE_CONFIGS["audio+image2video"];
    Object.entries(config).forEach(([widgetName, widgetValue]) => setWidgetValue(node, widgetName, widgetValue));
    if (generationType === REFERENCE_AUDIO_IMG2VID_TYPE) {
        setWidgetValue(node, "generated_media_fps", REFERENCE_AUDIO_IMG2VID_FPS, { notify: false });
    }
    applyRenderAnchorModeDefaults(node);
    markRenderGenerationTypeDefaultsApplied(node);
    syncReferenceAudioImg2VidTimingWhenReady(node, { applyFpsDefault: generationType === REFERENCE_AUDIO_IMG2VID_TYPE }, "generation_type_defaults", { defer: true });
    syncTextAudio2VideoPlannerModeWhenReady(node, "generation_type_defaults", { defer: true });
}

function applyRenderGenerationTypeChange(node) {
    syncRenderGenerationType(node, { force: true });
    sanitizeRenderWidgetValues(node);
    applyRenderAnchorVisibility(node);
    applyRenderSecondStageVisibility(node);
    applyRenderInternalWidgetVisibility(node);
    syncTextAudio2VideoPlannerModeWhenReady(node, "generation_type_change", { defer: true });
    syncRenderLinxWhenReady(node, "generation_type_change");
    markCanvasDirty();
}

function renderUsesGeneratedDuration(node) {
    const generationType = String(findWidget(node, "generation_type")?.value || "");
    return GENERATED_DURATION_TYPES.has(generationType);
}


function applyRenderGeneratedDurationVisibility(node) {
    if (node?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        return;
    }
    const generationExpanded = node.properties?.iamccs_section_generation !== false;
    setWidgetVisibility(findWidget(node, "generated_media_duration_seconds"), generationExpanded && renderUsesGeneratedDuration(node));
    setWidgetVisibility(findWidget(node, "generated_media_fps"), generationExpanded && renderUsesGeneratedDuration(node));
    fitNodeToWidgets(node);
}

function applyRenderManualSigmasVisibility(node) {
    if (node?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        return;
    }
    const samplingExpanded = node.properties?.iamccs_section_sampling !== false;
    const showWidget = findWidget(node, "show_manual_sigmas");
    const manualWidget = findWidget(node, "manual_sigmas");
    const backendMode = normalizeRenderBackendValue(findWidget(node, "backend_mode")?.value);
    const isLegacyBackend = Boolean(LEGACY_RENDER_BACKEND_DEFAULTS[backendMode]);
    setWidgetLabel(node, "show_manual_sigmas", "Advanced Manual Sigmas");
    setWidgetLabel(node, "manual_sigmas", "Manual Sigmas");
    if (isLegacyBackend && manualWidget && !String(manualWidget.value || "").trim()) {
        manualWidget.value = REF_MAIN_SIGMAS;
    }
    setWidgetVisibility(showWidget, samplingExpanded);
    setWidgetVisibility(manualWidget, samplingExpanded && showWidget?.value === true);
    fitNodeToWidgets(node);
}

function applyRenderTaeltxPreviewVisibility(node) {
    if (node?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        return;
    }
    const previewExpanded = !!node.properties?.iamccs_section_taeltx_preview;
    const enabledWidget = findWidget(node, "taeltx_preview");
    const maxFramesWidget = findWidget(node, "taeltx_preview_max_frames");
    const fpsWidget = findWidget(node, "taeltx_preview_fps");
    setWidgetLabel(node, "taeltx_preview", "TAELTX Preview");
    setWidgetLabel(node, "taeltx_preview_max_frames", "Preview Frames");
    setWidgetLabel(node, "taeltx_preview_fps", "Preview FPS");
    setWidgetVisibility(enabledWidget, previewExpanded);
    setWidgetVisibility(maxFramesWidget, previewExpanded && enabledWidget?.value === true);
    setWidgetVisibility(fpsWidget, previewExpanded && enabledWidget?.value === true);
    fitNodeToWidgets(node);
}

function inferRenderGenerationType(valuesByName) {
    const generationMode = String(valuesByName.generation_mode || "img2vid");
    const backendMode = String(valuesByName.backend_mode || "auto");
    const mediaMode = String(valuesByName.media_mode || "");
    if (mediaMode === "t2v_pure") {
        return "text2video";
    }
    if (mediaMode === "img2vid_pure" || mediaMode === "generated_audio_img2vid") {
        return "img2video";
    }
    if (generationMode === "t2v" || mediaMode === "input_audio_t2v" || mediaMode === "generated_audio_t2v") {
        return "text+audio2video";
    }
    if (backendMode === "ti2v_incremental_advanced") {
        return generationMode === "t2v" ? "text2video" : "img2video";
    }
    if (
        backendMode === "two_segments_normal_vram"
        || backendMode === "three_segments_normal_vram"
        || backendMode === "loop_normal_vram"
        || backendMode === "loop_low_ram_disk"
        || backendMode === "legacy backend"
        || backendMode === "legacy_single"
        || backendMode === "legacy_two_segments"
        || backendMode === "legacy_loop"
        || backendMode === "auto"
    ) {
        return "audio+image2video";
    }
    return "audio+image2video";
}

function applyRenderPresetDropdownOptions(node) {
    if (node?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        return;
    }
    const generationType = String(findWidget(node, "generation_type")?.value || "audio+image2video");
    const widget = findWidget(node, "ui_preset");
    if (widget?.options) {
        widget.options.values = RENDER_UI_PRESET_VISIBLE_VALUES;
    }
    const backendWidget = findWidget(node, "backend_mode");
    if (backendWidget?.options) {
        backendWidget.options.values = GENERATED_DURATION_TYPES.has(generationType)
            ? TI2V_RENDER_BACKEND_VISIBLE_VALUES
            : RENDER_BACKEND_MODE_VISIBLE_VALUES;
    }
}

function applyRenderInternalWidgetVisibility(node) {
    if (node?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        return;
    }
    for (const widgetName of RENDER_INTERNAL_WIDGETS) {
        setWidgetVisibility(findWidget(node, widgetName), false);
    }
    const generationType = String(findWidget(node, "generation_type")?.value || "audio+image2video");
    const generationExpanded = node.properties?.iamccs_section_generation !== false;
    setWidgetVisibility(findWidget(node, "backend_mode"), generationExpanded && GENERATED_DURATION_TYPES.has(generationType));
    applyRenderManualSigmasVisibility(node);
    applyRenderTaeltxPreviewVisibility(node);
}

function applyRenderSecondStageVisibility(node) {
    if (node?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        return;
    }

    const mode = String(findWidget(node, "second_stage_mode")?.value || "off");
    const stageExpanded = node.properties?.iamccs_section_stage2 !== false;
    const enabled = stageExpanded && mode !== "off";
    setWidgetVisibility(findWidget(node, "stage2_model_policy"), enabled);
    setWidgetVisibility(findWidget(node, "second_stage_reinject_strength"), enabled);
    setWidgetVisibility(findWidget(node, "second_stage_cfg"), enabled);
    setWidgetVisibility(findWidget(node, "second_stage_manual_sigmas"), enabled);
    setWidgetVisibility(findWidget(node, "second_stage_upscale_model"), enabled && mode === "latent_upscale_refine_x2_beta");
    fitNodeToWidgets(node);
}

function syncDownstreamVaeDecodeModes(renderNode) {
    const renderDecode = String(findWidget(renderNode, "vae_mode")?.value || "normal_tiled_iamccs");
    const renderFps = getRenderResolvedFps(renderNode);
    const renderBackendMode = normalizeRenderBackendValue(findWidget(renderNode, "backend_mode")?.value);
    const usesLegacyBackend = Boolean(LEGACY_RENDER_BACKEND_DEFAULTS[renderBackendMode]);
    const shouldSyncFps = Number.isFinite(renderFps);
    const visited = new Set();
    const queue = [...getLinxTargets(renderNode)];
    while (queue.length > 0) {
        const node = queue.shift();
        if (!node || visited.has(node.id)) {
            continue;
        }
        visited.add(node.id);
        if (node?.comfyClass === "IAMCCS-SuperNodes AU+IMG2VID Exec VAE") {
            const decodeWidget = findWidget(node, "decode_mode");
            if (decodeWidget && String(decodeWidget.value || "") === "inherit_render_backend") {
                setWidgetValue(node, "decode_mode", renderDecode);
            }
            if (shouldSyncFps) {
                setWidgetValue(node, "frame_rate", renderFps);
            }
            node.properties = node.properties || {};
            if (usesLegacyBackend && node.properties.iamccs_legacy_trim_default_applied_for_backend !== renderBackendMode) {
                setWidgetValue(node, "trim_to_audio", true, { notify: false });
                node.properties.iamccs_legacy_trim_default_applied_for_backend = renderBackendMode;
            }
            setWidgetValue(node, "ui_preset", "custom");
            applyVaeDecodeModeVisibility(node);
            markCanvasDirty();
            continue;
        }
        if (node?.comfyClass === "IAMCCS-SuperNodes Second Stage") {
            queue.push(...getLinxTargets(node));
        }
    }
}

function applyPresetConfig(node, nodeName) {
    const config = PRESET_CONFIGS[nodeName];
    if (!config) {
        return;
    }
    const presetWidget = findWidget(node, config.presetWidget);
    if (!presetWidget) {
        return;
    }
    const presetName = String(presetWidget.value || config.defaultPreset || "custom");
    if (presetName !== "custom") {
        const values = config.values?.[presetName] || {};
        Object.entries(values).forEach(([widgetName, widgetValue]) => setWidgetValue(node, widgetName, widgetValue));
    }
    const visibilityMap = config.visibility?.[presetName] || {};
    Object.entries(visibilityMap).forEach(([widgetName, isVisible]) => {
        setWidgetVisibility(findWidget(node, widgetName), !!isVisible);
    });
    if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec VAE") {
        sanitizeVaeWidgetValues(node);
        applyVaeDecodeModeVisibility(node);
    }
    if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        sanitizeRenderWidgetValues(node);
        applyRenderAnchorLabels(node);
        applyRenderAnchorVisibility(node);
        applyRenderSecondStageVisibility(node);
        applyRenderGeneratedDurationVisibility(node);
        applyRenderTaeltxPreviewVisibility(node);
        applyRenderPresetDropdownOptions(node);
    }
    fitNodeToWidgets(node);
}

function getSerializableWidgets(node) {
    if (!hasWidgetList(node)) {
        return [];
    }
    return node.widgets.filter((widget) => widget && widget.serialize !== false && !widget._iamccsSectionKey);
}

function isSectionButtonCaption(value) {
    const text = String(value ?? "");
    return /^\[[+-]\]\s/.test(text) || text.startsWith("â–¶ ") || text.startsWith("â–¼ ");
}

function sanitizeSerializedValues(values) {
    return (values || []).filter((value) => value !== null && value !== undefined && !isSectionButtonCaption(value));
}

function applyMissingGenerationDefaults(valuesByName, defaults) {
    if (!valuesByName || !defaults) {
        return valuesByName;
    }
    Object.entries(defaults).forEach(([widgetName, widgetValue]) => {
        const current = valuesByName[widgetName];
        const invalidNumber = typeof widgetValue === "number" && !Number.isFinite(Number(current));
        if (current === undefined || current === null || isSectionButtonCaption(current) || invalidNumber) {
            valuesByName[widgetName] = widgetValue;
        }
    });
    return valuesByName;
}

function isSerializedBoolean(value) {
    if (typeof value === "boolean") {
        return true;
    }
    const text = String(value ?? "").toLowerCase();
    return text === "true" || text === "false";
}

function chooseRenderSerializedOrder(cleanValues, currentOrder, previousOrder) {
    const advancedIndex = currentOrder.indexOf("show_manual_sigmas");
    const colorAwareOrder = (order) => {
        const colorIndex = order.indexOf("color_match_mode");
        if (colorIndex < 0) {
            return order;
        }
        if (colorIndex >= cleanValues.length) {
            return order.filter((name) => name !== "color_match_mode" && name !== "color_match_strength");
        }
        const colorValue = String(cleanValues[colorIndex] ?? "");
        if (!RENDER_COLOR_MATCH_MODE_VALUES.has(colorValue)) {
            return order.filter((name) => name !== "color_match_mode" && name !== "color_match_strength");
        }
        return order;
    };
    if (advancedIndex < 0 || advancedIndex >= cleanValues.length) {
        return colorAwareOrder(previousOrder || currentOrder);
    }
    return colorAwareOrder(isSerializedBoolean(cleanValues[advancedIndex]) ? currentOrder : (previousOrder || currentOrder));
}

function readPlannerSerializedValues(cleanValues) {
    const valuesByName = {
        audio_img2vid_backend: PLANNER_A2I_BACKEND_MODERN,
        audio_img2vid_mode: PLANNER_A2I_MODE_SINGLE,
        route_mode: "choose segment count (audio / segments)",
        single_duration_seconds: 10.0,
        segment_count: 2,
        segment_seconds: 10.0,
        fps: REFERENCE_AUDIO_IMG2VID_FPS,
        audio_preprocess_mode: "melband_vocals_duration_math",
        melband_model_name: "MelBandRoformer_fp32.safetensors",
        planning_mode: "manual_segment_seconds",
        segment_preset: "10sec",
        overlap_frames: 9,
        ltx_round_mode: "up",
        audio_concat_payload: "",
        debug_verbose: false,
    };
    const first = String(cleanValues[0] ?? "");
    let order = PLANNER_PY_WIDGET_ORDER;
    if (PLANNER_AUDIO_IMG2VID_BACKEND_VALUES.has(first)) {
        order = PLANNER_VISUAL_WIDGET_ORDER;
    } else if (PLANNER_ROUTE_MODE_VALUES.has(first)) {
        order = PLANNER_LEGACY_VISUAL_WIDGET_ORDER;
    } else if (PLANNER_ROUTE_MODE_VALUES.has(String(cleanValues[8] ?? ""))) {
        order = PLANNER_LEGACY_PY_WIDGET_ORDER;
    }
    order.forEach((widgetName, index) => {
        if (index < cleanValues.length) {
            valuesByName[widgetName] = cleanValues[index];
        }
    });
    if (!PLANNER_AUDIO_IMG2VID_BACKEND_VALUES.has(String(valuesByName.audio_img2vid_backend || ""))) {
        valuesByName.audio_img2vid_backend = PLANNER_A2I_BACKEND_MODERN;
    }
    if (!PLANNER_AUDIO_IMG2VID_MODE_VALUES.has(String(valuesByName.audio_img2vid_mode || ""))) {
        const routeMode = String(valuesByName.route_mode || "");
        const count = Number(valuesByName.segment_count || 2);
        valuesByName.audio_img2vid_mode = routeMode.includes("single")
            ? PLANNER_A2I_MODE_SINGLE
            : count === 2
                ? PLANNER_A2I_MODE_2_SEGMENTS
                : count === 3
                    ? PLANNER_A2I_MODE_3_SEGMENTS
                    : PLANNER_A2I_MODE_LOOP;
    }
    valuesByName.audio_img2vid_backend = plannerA2IBackendDisplay(valuesByName.audio_img2vid_backend);
    valuesByName.audio_img2vid_mode = plannerA2IModeDisplay(valuesByName.audio_img2vid_mode);
    if (!PLANNER_ROUTE_MODE_VALUES.has(String(valuesByName.route_mode || ""))) {
        valuesByName.route_mode = "choose segment count (audio / segments)";
    }
    if (!PLANNER_MODE_VALUES.has(String(valuesByName.planning_mode || ""))) {
        valuesByName.planning_mode = "manual_segment_seconds";
    }
    if (!PLANNER_SEGMENT_PRESET_VALUES.has(String(valuesByName.segment_preset || ""))) {
        valuesByName.segment_preset = "10sec";
    }
    if (!PLANNER_ROUND_MODE_VALUES.has(String(valuesByName.ltx_round_mode || ""))) {
        valuesByName.ltx_round_mode = "up";
    }
    if (!PLANNER_AUDIO_MODE_VALUES.has(String(valuesByName.audio_preprocess_mode || ""))) {
        valuesByName.audio_preprocess_mode = "melband_vocals_duration_math";
    }
    for (const [widgetName, fallback] of Object.entries({
        fps: REFERENCE_AUDIO_IMG2VID_FPS,
        segment_seconds: 10.0,
        overlap_frames: 9,
        segment_count: 2,
        single_duration_seconds: 10.0,
    })) {
        if (!Number.isFinite(Number(valuesByName[widgetName])) || isSectionButtonCaption(valuesByName[widgetName])) {
            valuesByName[widgetName] = fallback;
        }
    }
    if (typeof valuesByName.debug_verbose !== "boolean") {
        valuesByName.debug_verbose = false;
    }
    return valuesByName;
}

function readRenderSerializedValues(cleanValues) {
    const valuesByName = {};
    if (RENDER_GENERATION_TYPE_VALUES.has(String(cleanValues[0] || ""))) {
        const hasGeneratedFps = !["img2vid", "t2v"].includes(String(cleanValues[3] || ""));
        const renderOrder = hasGeneratedFps
            ? chooseRenderSerializedOrder(cleanValues, RENDER_CURRENT_WIDGET_ORDER, RENDER_CURRENT_WIDGET_ORDER_PRE_ADVANCED_MANUAL_SIGMAS)
            : chooseRenderSerializedOrder(cleanValues, RENDER_CURRENT_WIDGET_ORDER_PRE_GENERATED_FPS, RENDER_CURRENT_WIDGET_ORDER_PRE_GENERATED_FPS_PRE_ADVANCED_MANUAL_SIGMAS);
        renderOrder.forEach((widgetName, index) => {
            if (index < cleanValues.length) {
                valuesByName[widgetName] = cleanValues[index];
            }
        });
        valuesByName.show_manual_sigmas = isSerializedBoolean(valuesByName.show_manual_sigmas)
            ? String(valuesByName.show_manual_sigmas).toLowerCase() === "true"
            : false;
        const generationConfig = GENERATION_TYPE_CONFIGS[String(valuesByName.generation_type)] || GENERATION_TYPE_CONFIGS["audio+image2video"];
        valuesByName.ui_preset = RENDER_UI_PRESET_VALUES.has(String(valuesByName.ui_preset || "")) ? valuesByName.ui_preset : "custom";
        valuesByName.generated_media_duration_seconds = Number.isFinite(Number(valuesByName.generated_media_duration_seconds))
            ? Number(valuesByName.generated_media_duration_seconds)
            : 10.0;
        valuesByName.generated_media_fps = Number.isFinite(Number(valuesByName.generated_media_fps))
            ? Number(valuesByName.generated_media_fps)
            : Number(generationConfig.generated_media_fps || DEFAULT_GENERATED_FPS);
        valuesByName.taeltx_preview = isSerializedBoolean(valuesByName.taeltx_preview)
            ? String(valuesByName.taeltx_preview).toLowerCase() === "true"
            : false;
        valuesByName.taeltx_preview_max_frames = Number.isFinite(Number(valuesByName.taeltx_preview_max_frames))
            ? Number(valuesByName.taeltx_preview_max_frames)
            : 17;
        valuesByName.taeltx_preview_fps = Number.isFinite(Number(valuesByName.taeltx_preview_fps))
            ? Number(valuesByName.taeltx_preview_fps)
            : 8;
        return normalizeReferenceAudioImg2VidValues(valuesByName);
    }
    if (["img2vid", "t2v"].includes(String(cleanValues[0] || ""))) {
        const renderOrder = chooseRenderSerializedOrder(cleanValues, RENDER_LEGACY_WIDGET_ORDER, RENDER_LEGACY_WIDGET_ORDER_PRE_ADVANCED_MANUAL_SIGMAS);
        renderOrder.forEach((widgetName, index) => {
            if (index < cleanValues.length) {
                valuesByName[widgetName] = cleanValues[index];
            }
        });
        valuesByName.show_manual_sigmas = isSerializedBoolean(valuesByName.show_manual_sigmas)
            ? String(valuesByName.show_manual_sigmas).toLowerCase() === "true"
            : false;
        valuesByName.generation_type = RENDER_GENERATION_TYPE_VALUES.has(String(valuesByName.generation_type || ""))
            ? valuesByName.generation_type
            : inferRenderGenerationType(valuesByName);
        valuesByName.ui_preset = RENDER_UI_PRESET_VALUES.has(String(valuesByName.ui_preset || "")) ? valuesByName.ui_preset : "custom";
        valuesByName.generated_media_duration_seconds = Number.isFinite(Number(valuesByName.generated_media_duration_seconds))
            ? Number(valuesByName.generated_media_duration_seconds)
            : 10.0;
        const generationConfig = GENERATION_TYPE_CONFIGS[String(valuesByName.generation_type)] || GENERATION_TYPE_CONFIGS["audio+image2video"];
        applyMissingGenerationDefaults(valuesByName, generationConfig);
        valuesByName.generated_media_fps = Number.isFinite(Number(valuesByName.generated_media_fps))
            ? Number(valuesByName.generated_media_fps)
            : Number(generationConfig.generated_media_fps || DEFAULT_GENERATED_FPS);
        valuesByName.taeltx_preview = isSerializedBoolean(valuesByName.taeltx_preview)
            ? String(valuesByName.taeltx_preview).toLowerCase() === "true"
            : false;
        valuesByName.taeltx_preview_max_frames = Number.isFinite(Number(valuesByName.taeltx_preview_max_frames))
            ? Number(valuesByName.taeltx_preview_max_frames)
            : 17;
        valuesByName.taeltx_preview_fps = Number.isFinite(Number(valuesByName.taeltx_preview_fps))
            ? Number(valuesByName.taeltx_preview_fps)
            : 8;
        return normalizeReferenceAudioImg2VidValues(valuesByName);
    }
    return null;
}

function readVaeSerializedValues(cleanValues) {
    const valuesByName = {
        tiled_temporal_size: 256,
        tiled_temporal_overlap: 32,
        cleanup_before_decode: false,
    };
    const decodeFirstOrder = [
        "decode_mode", "tiled_tile_size", "tiled_overlap", "tiled_temporal_size", "tiled_temporal_overlap",
        "cleanup_before_decode", "frame_rate", "frames_subdir", "image_format", "jpg_quality", "output_root",
        "filename_prefix", "crf", "pix_fmt", "trim_to_audio", "save_metadata", "vram_flush", "ui_preset", "debug_verbose",
    ];
    let order = VAE_CURRENT_WIDGET_ORDER;
    if (VAE_DECODE_MODE_VALUES.has(String(cleanValues[0] || ""))) {
        order = decodeFirstOrder;
    } else if (typeof cleanValues[2] === "string") {
        order = cleanValues.length >= VAE_CURRENT_PY_WIDGET_ORDER.length
            ? VAE_CURRENT_PY_WIDGET_ORDER
            : VAE_LEGACY_PY_WIDGET_ORDER;
    } else if (typeof cleanValues[4] === "string") {
        order = VAE_LEGACY_WIDGET_ORDER;
    }
    order.forEach((widgetName, index) => {
        if (index < cleanValues.length) {
            valuesByName[widgetName] = cleanValues[index];
        }
    });
    valuesByName.ui_preset = VAE_UI_PRESET_VALUES.has(String(valuesByName.ui_preset || "")) ? valuesByName.ui_preset : "custom";
    valuesByName.decode_mode = VAE_DECODE_MODE_VALUES.has(String(valuesByName.decode_mode || "")) ? valuesByName.decode_mode : "normal_tiled_iamccs";
    return valuesByName;
}

function applyValuesByNameToWidgets(node, valuesByName) {
    if (!valuesByName) {
        return;
    }
    for (const widget of getSerializableWidgets(node)) {
        if (Object.prototype.hasOwnProperty.call(valuesByName, widget.name)) {
            widget.value = valuesByName[widget.name];
        }
    }
}

function normalizeConfigureWidgetValues(node, nodeName, info) {
    if (!info || !Array.isArray(info.widgets_values) || !hasWidgetList(node) || !node.widgets.length) {
        return;
    }
    const cleanValues = sanitizeSerializedValues(info.widgets_values);
    let valuesByName = null;
    if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        valuesByName = readRenderSerializedValues(cleanValues);
    } else if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec VAE") {
        valuesByName = readVaeSerializedValues(cleanValues);
    } else if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Planner") {
        valuesByName = readPlannerSerializedValues(cleanValues);
    }
    if (!valuesByName) {
        return;
    }
    const normalized = [];
    for (const widget of node.widgets) {
        if (!widget || widget.serialize === false || widget._iamccsSectionKey) {
            normalized.push(null);
        } else if (Object.prototype.hasOwnProperty.call(valuesByName, widget.name)) {
            normalized.push(valuesByName[widget.name]);
        } else {
            normalized.push(widget.value ?? null);
        }
    }
    info.widgets_values = normalized;
}

function sanitizeRenderCombo(node, widgetName, fallback, validValues) {
    const widget = findWidget(node, widgetName);
    if (!widget) {
        return;
    }
    const text = String(widget.value ?? "");
    if (isSectionButtonCaption(text) || (validValues?.size && !validValues.has(text))) {
        widget.value = fallback;
    }
}

function sanitizeRenderNumber(node, widgetName, fallback) {
    const widget = findWidget(node, widgetName);
    if (!widget) {
        return;
    }
    const value = Number(widget.value);
    if (isSectionButtonCaption(widget.value) || !Number.isFinite(value)) {
        widget.value = fallback;
    }
}

function sanitizeBooleanWidget(node, widgetName, fallback) {
    const widget = findWidget(node, widgetName);
    if (!widget) {
        return;
    }
    if (isSectionButtonCaption(widget.value) || typeof widget.value !== "boolean") {
        widget.value = !!fallback;
    }
}

function sanitizeRenderWidgetValues(node) {
    if (node?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        return;
    }
    applyRenderPresetDropdownOptions(node);
    sanitizeRenderCombo(node, "ui_preset", "custom", RENDER_UI_PRESET_VALUES);
    sanitizeRenderCombo(node, "generation_type", "audio+image2video", RENDER_GENERATION_TYPE_VALUES);

    const durationWidget = findWidget(node, "generated_media_duration_seconds");
    if (durationWidget && (isSectionButtonCaption(durationWidget.value) || !Number.isFinite(Number(durationWidget.value)))) {
        durationWidget.value = 10.0;
    }
    const generationType = String(findWidget(node, "generation_type")?.value || "audio+image2video");
    const generationConfig = GENERATION_TYPE_CONFIGS[generationType] || GENERATION_TYPE_CONFIGS["audio+image2video"];
    const backendWidget = findWidget(node, "backend_mode");
    if (String(backendWidget?.value || "") === "legacy backend") {
        backendWidget.value = "legacy_single";
    }
    sanitizeRenderCombo(node, "backend_mode", generationConfig.backend_mode || "auto", RENDER_BACKEND_MODE_VALUES);
    sanitizeRenderCombo(node, "stitch_preset", "custom", RENDER_STITCH_PRESET_VALUES);
    sanitizeRenderCombo(node, "overlap_side", "source", RENDER_OVERLAP_SIDE_VALUES);
    sanitizeRenderCombo(node, "overlap_mode", "cut", RENDER_OVERLAP_MODE_VALUES);
    sanitizeRenderCombo(node, "start_frames_rule", "none", RENDER_START_FRAMES_RULE_VALUES);
    sanitizeRenderCombo(node, "color_match_mode", "none", RENDER_COLOR_MATCH_MODE_VALUES);
    sanitizeRenderCombo(node, "audio_context_mode", "left_context_only", RENDER_AUDIO_CONTEXT_MODE_VALUES);
    sanitizeRenderNumber(node, "generated_media_fps", DEFAULT_GENERATED_FPS);
    sanitizeRenderNumber(node, "width", DEFAULT_VIDEO_WIDTH);
    sanitizeRenderNumber(node, "height", generationConfig.height || DEFAULT_VIDEO_HEIGHT);
    if (generationType === REFERENCE_AUDIO_IMG2VID_TYPE) {
        const widthWidget = findWidget(node, "width");
        const heightWidget = findWidget(node, "height");
        // Code by Carmine Cristallo Scalzi AI research (IAMCCS)
        // patreon.com/IAMCCS
        // Old SuperNode saves carried the generic 1280x736 LTX value. The
        // working A+I2V reference workflow feeds ImageResizeKJv2 with 1280x720.
        if (Number(widthWidget?.value) === REFERENCE_AUDIO_IMG2VID_WIDTH && Number(heightWidget?.value) === DEFAULT_VIDEO_HEIGHT) {
            heightWidget.value = REFERENCE_AUDIO_IMG2VID_HEIGHT;
        }
        if (Number(widthWidget?.value) === STALE_AUDIO_IMG2VID_WIDTH && Number(heightWidget?.value) === STALE_AUDIO_IMG2VID_HEIGHT) {
            widthWidget.value = REFERENCE_AUDIO_IMG2VID_WIDTH;
            heightWidget.value = REFERENCE_AUDIO_IMG2VID_HEIGHT;
        }
    }
    sanitizeRenderNumber(node, "steps", 8);
    sanitizeRenderNumber(node, "cfg", 1.0);
    sanitizeRenderNumber(node, "seed", 0);
    sanitizeRenderNumber(node, "max_shift", 2.05);
    sanitizeRenderNumber(node, "base_shift", 0.95);
    sanitizeRenderNumber(node, "sigma_terminal", 0.1);
    sanitizeRenderNumber(node, "image_strength", 0.9);
    sanitizeRenderNumber(node, "image_compression", 35);
    sanitizeRenderNumber(node, "color_match_strength", 0.25);
    sanitizeRenderNumber(node, "audio_left_context_s", 0.5);
    sanitizeRenderNumber(node, "audio_right_context_s", 0.0);
    sanitizeRenderNumber(node, "anchor_refresh_interval", 2);
    sanitizeRenderNumber(node, "anchor_image_strength", 0.0);
    sanitizeRenderNumber(node, "anti_drift_strength", 0.0);
    sanitizeRenderNumber(node, "identity_persistence_strength", 0.0);
    sanitizeRenderNumber(node, "second_stage_reinject_strength", 0.0);
    sanitizeRenderNumber(node, "second_stage_cfg", 1.0);
    sanitizeRenderNumber(node, "motion_intensity", 1.0);
    sanitizeRenderNumber(node, "taeltx_preview_max_frames", 17);
    sanitizeRenderNumber(node, "taeltx_preview_fps", 8);
    sanitizeBooleanWidget(node, "show_manual_sigmas", false);
    sanitizeBooleanWidget(node, "taeltx_preview", false);
    sanitizeBooleanWidget(node, "debug_verbose", false);
    const normalizedBackendMode = normalizeRenderBackendValue(findWidget(node, "backend_mode")?.value);
    if (LEGACY_RENDER_BACKEND_DEFAULTS[normalizedBackendMode]) {
        const manualWidget = findWidget(node, "manual_sigmas");
        if (manualWidget && !String(manualWidget.value || "").trim()) {
            manualWidget.value = REF_MAIN_SIGMAS;
        }
    }

    applyRenderGeneratedDurationVisibility(node);
    applyRenderManualSigmasVisibility(node);
    applyRenderTaeltxPreviewVisibility(node);
}

function sanitizeVaeWidgetValues(node) {
    if (node?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec VAE") {
        return;
    }
    sanitizeRenderCombo(node, "ui_preset", "custom", VAE_UI_PRESET_VALUES);
    sanitizeRenderCombo(node, "decode_mode", "normal_tiled_iamccs", VAE_DECODE_MODE_VALUES);
    sanitizeRenderNumber(node, "frame_rate", DEFAULT_VAE_FRAME_RATE);
    sanitizeRenderNumber(node, "jpg_quality", 95);
    sanitizeRenderNumber(node, "tiled_tile_size", 512);
    sanitizeRenderNumber(node, "tiled_overlap", 64);
    sanitizeRenderNumber(node, "tiled_temporal_size", 256);
    sanitizeRenderNumber(node, "tiled_temporal_overlap", 32);
    sanitizeBooleanWidget(node, "cleanup_before_decode", false);
    sanitizeRenderNumber(node, "crf", 19);
    sanitizeBooleanWidget(node, "trim_to_audio", true);
    sanitizeBooleanWidget(node, "save_metadata", false);
    sanitizeBooleanWidget(node, "vram_flush", false);
    sanitizeBooleanWidget(node, "debug_verbose", false);
}

function sanitizePlannerWidgetValues(node) {
    if (node?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Planner") {
        return;
    }
    sanitizeRenderCombo(node, "audio_img2vid_backend", PLANNER_A2I_BACKEND_MODERN, PLANNER_AUDIO_IMG2VID_BACKEND_VALUES);
    sanitizeRenderCombo(node, "audio_img2vid_mode", PLANNER_A2I_MODE_SINGLE, PLANNER_AUDIO_IMG2VID_MODE_VALUES);
    setWidgetValue(node, "audio_img2vid_backend", plannerA2IBackendDisplay(findWidget(node, "audio_img2vid_backend")?.value), { notify: false });
    setWidgetValue(node, "audio_img2vid_mode", plannerA2IModeDisplay(findWidget(node, "audio_img2vid_mode")?.value), { notify: false });
    sanitizeRenderCombo(node, "route_mode", "choose segment count (audio / segments)", PLANNER_ROUTE_MODE_VALUES);
    sanitizeRenderCombo(node, "planning_mode", "manual_segment_seconds", PLANNER_MODE_VALUES);
    sanitizeRenderCombo(node, "segment_preset", "10sec", PLANNER_SEGMENT_PRESET_VALUES);
    sanitizeRenderCombo(node, "ltx_round_mode", "up", PLANNER_ROUND_MODE_VALUES);
    sanitizeRenderCombo(node, "audio_preprocess_mode", "melband_vocals_duration_math", PLANNER_AUDIO_MODE_VALUES);
    sanitizeRenderNumber(node, "fps", REFERENCE_AUDIO_IMG2VID_FPS);
    sanitizeRenderNumber(node, "segment_seconds", 10.0);
    sanitizeRenderNumber(node, "overlap_frames", 9);
    sanitizeRenderNumber(node, "segment_count", 2);
    sanitizeRenderNumber(node, "single_duration_seconds", 10.0);
    sanitizeBooleanWidget(node, "debug_verbose", false);
}

function normalizePlannerA2IBackend(value) {
    const text = String(value || "").trim().toLowerCase();
    return text.includes("legacy") ? "legacy" : "modern";
}

function plannerA2IBackendDisplay(value) {
    return normalizePlannerA2IBackend(value) === "legacy" ? PLANNER_A2I_BACKEND_LEGACY : PLANNER_A2I_BACKEND_MODERN;
}

function normalizePlannerA2IMode(value) {
    const text = String(value || "").trim().toLowerCase();
    if (text.includes("single") || text === "1" || text === "one") {
        return "single";
    }
    if (text === "2" || text === "2_segments" || text.includes("2 segment")) {
        return "2_segments";
    }
    if (text === "3" || text === "3_segments" || text.includes("3 segment")) {
        return "3_segments";
    }
    if (text.includes("loop") || text.includes("4+")) {
        return "loop";
    }
    return "single";
}

function plannerA2IModeDisplay(value) {
    const mode = normalizePlannerA2IMode(value);
    if (mode === "2_segments") {
        return PLANNER_A2I_MODE_2_SEGMENTS;
    }
    if (mode === "3_segments") {
        return PLANNER_A2I_MODE_3_SEGMENTS;
    }
    if (mode === "loop") {
        return PLANNER_A2I_MODE_LOOP;
    }
    return PLANNER_A2I_MODE_SINGLE;
}

function resolvePlannerA2IRenderBackend(plannerNode) {
    const family = normalizePlannerA2IBackend(findWidget(plannerNode, "audio_img2vid_backend")?.value);
    const mode = normalizePlannerA2IMode(findWidget(plannerNode, "audio_img2vid_mode")?.value);
    if (family === "legacy") {
        if (mode === "single") {
            return "legacy_single";
        }
        if (mode === "2_segments") {
            return "legacy_two_segments";
        }
        return "legacy_loop";
    }
    if (mode === "single") {
        return "single_best";
    }
    if (mode === "2_segments") {
        return "two_segments_normal_vram";
    }
    if (mode === "3_segments") {
        return "three_segments_normal_vram";
    }
    return "loop_normal_vram";
}

function syncPlannerA2IModeToInternalWidgets(node) {
    if (node?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Planner") {
        return;
    }
    const mode = normalizePlannerA2IMode(findWidget(node, "audio_img2vid_mode")?.value);
    if (mode === "single") {
        setWidgetValue(node, "route_mode", "single generation (duration only)", { notify: false });
        setWidgetValue(node, "segment_count", 1, { notify: false });
    } else {
        setWidgetValue(node, "route_mode", "choose segment count (audio / segments)", { notify: false });
        if (mode === "2_segments") {
            setWidgetValue(node, "segment_count", 2, { notify: false });
        } else if (mode === "3_segments") {
            setWidgetValue(node, "segment_count", 3, { notify: false });
        } else {
            const count = Math.max(4, Math.round(readPlannerNumber(node, "segment_count", 4)));
            setWidgetValue(node, "segment_count", count, { notify: false });
        }
    }
}

function plannerTargetsReferenceAudioImg2Vid(plannerNode) {
    if (plannerNode?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Planner") {
        return false;
    }
    const visited = new Set();
    const queue = [...getLinxTargets(plannerNode)];
    let sawRender = false;
    while (queue.length > 0) {
        const node = queue.shift();
        if (!node || visited.has(node.id)) {
            continue;
        }
        visited.add(node.id);
        if (node?.comfyClass === "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
            sawRender = true;
            if (renderUsesReferenceAudioImg2Vid(node)) {
                return true;
            }
            continue;
        }
        queue.push(...getLinxTargets(node));
    }
    return !sawRender;
}

function rehydrateSerializedWidgets(node, serializedValues) {
    if (!Array.isArray(serializedValues) || !hasWidgetList(node) || !node.widgets.length) {
        return;
    }

    const cleanValues = sanitizeSerializedValues(serializedValues);
    if (node?.comfyClass === "IAMCCS-SuperNodes AU+IMG2VID Exec Render" && RENDER_GENERATION_TYPE_VALUES.has(String(cleanValues[0] || ""))) {
        applyValuesByNameToWidgets(node, readRenderSerializedValues(cleanValues));
        syncReferenceAudioImg2VidTimingWhenReady(node, {}, "rehydrate_generation_type", { defer: true });
        return;
    }
    if (node?.comfyClass === "IAMCCS-SuperNodes AU+IMG2VID Exec Render" && ["img2vid", "t2v"].includes(String(cleanValues[0] || ""))) {
        applyValuesByNameToWidgets(node, readRenderSerializedValues(cleanValues));
        syncReferenceAudioImg2VidTimingWhenReady(node, {}, "rehydrate_legacy_mode", { defer: true });
        return;
    }
    if (node?.comfyClass === "IAMCCS-SuperNodes AU+IMG2VID Exec VAE") {
        applyValuesByNameToWidgets(node, readVaeSerializedValues(cleanValues));
        sanitizeVaeWidgetValues(node);
        return;
    }
    if (node?.comfyClass === "IAMCCS-SuperNodes AU+IMG2VID Exec Planner") {
        applyValuesByNameToWidgets(node, readPlannerSerializedValues(cleanValues));
        sanitizePlannerWidgetValues(node);
        return;
    }

    const widgets = getSerializableWidgets(node);
    const count = Math.min(widgets.length, cleanValues.length);
    for (let index = 0; index < count; index += 1) {
        const widget = widgets[index];
        if (!widget) {
            continue;
        }
        widget.value = cleanValues[index];
    }
}

function addSectionButton(node, group, index) {
    if (!hasWidgetList(node) || !group) {
        return null;
    }
    // Duplicate guard: if a button for this section key is already in the widget list,
    // return it immediately.  This prevents double-insertion if onNodeCreated or
    // enhanceNodeLayout is called more than once on the same node instance.
    const existing = node.widgets.find((w) => w._iamccsSectionKey === group.key);
    if (existing) {
        installSectionButtonDraw(existing);
        insertWidget(node, existing, index);
        return existing;
    }

    node.properties = node.properties || {};
    const propKey = `iamccs_section_${group.key}`;
    if (node.properties[propKey] === undefined) {
        node.properties[propKey] = !DEFAULT_COLLAPSED_SECTION_KEYS.has(group.key);
    }

    const sectionButton = node.addWidget("button", "", "", () => {
        node.properties[propKey] = !node.properties[propKey];
        applyGroupVisibility(node, group, propKey, sectionButton);
    });
    // Visual-only section buttons must never enter widgets_values/prompt payloads.
    sectionButton.serialize = false;
    sectionButton._iamccsSectionKey = group.key;
    sectionButton.computeSize = (width) => [width || 280, SECTION_BUTTON_HEIGHT];
    sectionButton.label = group.label;
    installSectionButtonDraw(sectionButton);
    sectionButton.options = {
        bgcolor: group.color || "#4f6f8f",
        background_color: group.color || "#4f6f8f",
        color: SECTION_BUTTON_TEXT,
    };
    sectionButton.value = "";

    applyGroupVisibility(node, group, propKey, sectionButton);
    insertWidget(node, sectionButton, index);
    return sectionButton;
}

function applyGroupVisibility(node, group, propKey, button) {
    if (!node || !group || !button) {
        return;
    }
    node.properties = node.properties || {};
    const isExpanded = !!node.properties[propKey];
    const caption = sectionButtonCaption(group, isExpanded);
    button.name = caption;
    button.label = caption;
    button.value = "";
    const style = sectionButtonStyle(group, isExpanded);
    button._iamccsButtonStyle = style;
    button.options = {
        bgcolor: style.background,
        background_color: style.background,
        border_color: style.border,
        color: SECTION_BUTTON_TEXT,
    };
    if (node?.comfyClass === "IAMCCS-SuperNodes AU+IMG2VID Exec Planner" && ["route", "audio"].includes(group.key)) {
        applyPlannerModeVisibility(node, { skipPresetSync: true });
        markCanvasDirty();
        return;
    }
    for (const widgetName of group.widgets) {
        setWidgetVisibility(findWidget(node, widgetName), isExpanded);
    }
    if (node?.comfyClass === "IAMCCS-SuperNodes AU+IMG2VID Exec Render" && group.key === "generation") {
        applyRenderGeneratedDurationVisibility(node);
        applyRenderInternalWidgetVisibility(node);
    }
    if (node?.comfyClass === "IAMCCS-SuperNodes AU+IMG2VID Exec Render" && group.key === "sampling") {
        applyRenderManualSigmasVisibility(node);
    }
    if (node?.comfyClass === "IAMCCS-SuperNodes AU+IMG2VID Exec Render" && group.key === "stage2") {
        applyRenderSecondStageVisibility(node);
    }
    if (node?.comfyClass === "IAMCCS-SuperNodes AU+IMG2VID Exec Render" && (group.key === "anchor" || group.key === "latent_refresh")) {
        applyRenderAnchorVisibility(node);
    }
    if (node?.comfyClass === "IAMCCS-SuperNodes AU+IMG2VID Exec VAE" && ["mode", "output", "decode"].includes(group.key)) {
        applyVaeDecodeModeVisibility(node);
    }
    fitNodeToWidgets(node);
    markCanvasDirty();
}

function applyPlannerModeVisibility(node, options = {}) {
    if (node?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Planner") {
        return;
    }
    sanitizePlannerWidgetValues(node);
    const showA2IControls = plannerTargetsReferenceAudioImg2Vid(node);
    if (showA2IControls) {
        syncPlannerA2IModeToInternalWidgets(node);
    }
    const routeMode = String(findWidget(node, "route_mode")?.value || "choose segment count (audio / segments)");
    const mode = showA2IControls
        ? normalizePlannerA2IMode(findWidget(node, "audio_img2vid_mode")?.value)
        : (routeMode.includes("single") ? "single" : "loop");
    const routeExpanded = node.properties?.iamccs_section_route !== false;
    const audioExpanded = node.properties?.iamccs_section_audio !== false;
    const isSingle = mode === "single";
    const isLoop = mode === "loop";
    const audioMode = String(findWidget(node, "audio_preprocess_mode")?.value || "melband_vocals_duration_math");

    setWidgetLabel(node, "audio_img2vid_backend", "Pipeline Family");
    setWidgetLabel(node, "audio_img2vid_mode", "Pipeline Shape");
    setWidgetLabel(node, "route_mode", "Internal Length Rule");
    setWidgetLabel(node, "single_duration_seconds", "Duration (s)");
    setWidgetLabel(node, "segment_count", "Segments");
    setWidgetLabel(node, "segment_seconds", "Seconds / Segment");
    setWidgetLabel(node, "fps", "FPS");
    setWidgetLabel(node, "audio_preprocess_mode", "Audio Mode");
    setWidgetLabel(node, "melband_model_name", "MelBand Model");
    setWidgetLabel(node, "overlap_frames", "Overlap Frames");
    setWidgetLabel(node, "ltx_round_mode", "Frame Rounding");
    setWidgetLabel(node, "debug_verbose", "Verbose Debug");

    setWidgetVisibility(findWidget(node, "audio_img2vid_backend"), routeExpanded && showA2IControls);
    setWidgetVisibility(findWidget(node, "audio_img2vid_mode"), routeExpanded && showA2IControls);
    setWidgetVisibility(findWidget(node, "route_mode"), false);
    setWidgetVisibility(findWidget(node, "fps"), routeExpanded);
    setWidgetVisibility(findWidget(node, "single_duration_seconds"), routeExpanded && isSingle);
    setWidgetVisibility(findWidget(node, "segment_count"), routeExpanded && isLoop);
    setWidgetVisibility(findWidget(node, "segment_seconds"), routeExpanded && !isSingle);
    setWidgetVisibility(findWidget(node, "planning_mode"), false);
    setWidgetVisibility(findWidget(node, "segment_preset"), false);
    setWidgetVisibility(findWidget(node, "overlap_frames"), routeExpanded && !isSingle);
    setWidgetVisibility(findWidget(node, "ltx_round_mode"), routeExpanded);
    setWidgetVisibility(findWidget(node, "audio_concat_payload"), false);
    setWidgetVisibility(findWidget(node, "debug_verbose"), routeExpanded);
    sanitizeBooleanWidget(node, "debug_verbose", false);
    setWidgetVisibility(findWidget(node, "audio_preprocess_mode"), audioExpanded);
    setWidgetVisibility(findWidget(node, "melband_model_name"), audioExpanded && audioMode !== "raw_audio_only");
    if (!options.skipPresetSync) {
        syncExecPlannerExplicitPreset(node);
    }
    fitNodeToWidgets(node);
}

function removeStaleSectionButtons(node, nodeName) {
    if (!hasWidgetList(node)) {
        return;
    }
    const validKeys = new Set((NODE_GROUPS[nodeName] || []).map((group) => group.key));
    node.widgets = node.widgets.filter((widget) => !widget?._iamccsSectionKey || validKeys.has(widget._iamccsSectionKey));
}

function refreshNodeLayoutState(node, nodeName) {
    if (!hasWidgetList(node)) {
        return;
    }
    removeStaleSectionButtons(node, nodeName);
    const groups = NODE_GROUPS[nodeName] || [];
    for (const group of groups) {
        const button = node.widgets.find((widget) => widget?._iamccsSectionKey === group.key);
        if (!button) {
            continue;
        }
        applyGroupVisibility(node, group, `iamccs_section_${group.key}`, button);
    }

    const config = PRESET_CONFIGS[nodeName];
    if (config) {
        const presetWidget = findWidget(node, config.presetWidget);
        const presetName = String(presetWidget?.value || config.defaultPreset || "custom");
        const visibilityMap = config.visibility?.[presetName] || {};
        Object.entries(visibilityMap).forEach(([widgetName, isVisible]) => {
            setWidgetVisibility(findWidget(node, widgetName), !!isVisible);
        });
    }

    if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec VAE") {
        sanitizeVaeWidgetValues(node);
        applyVaeDecodeModeVisibility(node);
    } else if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Planner") {
        sanitizePlannerWidgetValues(node);
        applyPlannerModeVisibility(node);
    } else if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        sanitizeRenderWidgetValues(node);
        applyRenderAnchorVisibility(node);
        applyRenderSecondStageVisibility(node);
        applyRenderInternalWidgetVisibility(node);
    } else {
        fitNodeToWidgets(node);
    }
}

function addPlannerChip(node) {
    if (!hasWidgetList(node) || node.widgets.some((widget) => widget?.name === "planner_chip_preview")) {
        return;
    }
    const chipWidget = {
        type: "custom",
        name: "planner_chip_preview",
        serialize: false,
        computeSize(width) {
            return [Math.max(220, (width || 260) - 20), 30];
        },
        draw(ctx, widget, nodeRef, widgetWidth, y) {
            const text = nodeRef.properties?.iamccsPlannerChip || "";
            if (!text) {
                return;
            }
            const x = 12;
            const w = Math.max(180, widgetWidth - 24);
            const h = 22;
            const top = y + 4;

            ctx.save();
            ctx.fillStyle = "#18312b";
            ctx.strokeStyle = "#6ea88d";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.roundRect(x, top, w, h, 9);
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = "#d8efe2";
            ctx.font = "12px Segoe UI";
            ctx.textBaseline = "middle";
            ctx.fillText(text, x + 10, top + h / 2 + 0.5);
            ctx.restore();
        },
    };
    node.widgets.push(chipWidget);
}

function addStatusBox(node, propertyName, widgetName, fill, stroke, textColor) {
    if (!hasWidgetList(node) || node.widgets.some((widget) => widget?.name === widgetName)) {
        return;
    }
    const statusWidget = {
        type: "custom",
        name: widgetName,
        serialize: false,
        computeSize(width) {
            return [Math.max(220, (width || 260) - 20), 42];
        },
        draw(ctx, widget, nodeRef, widgetWidth, y) {
            const text = nodeRef.properties?.[propertyName] || "";
            if (!text) {
                return;
            }
            const x = 12;
            const w = Math.max(180, widgetWidth - 24);
            const h = 34;
            const top = y + 4;

            ctx.save();
            ctx.fillStyle = fill;
            ctx.strokeStyle = stroke;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.roundRect(x, top, w, h, 10);
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = textColor;
            ctx.font = "12px Segoe UI";
            ctx.textBaseline = "middle";
            ctx.fillText(text, x + 10, top + h / 2 + 0.5);
            ctx.restore();
        },
    };
    node.widgets.push(statusWidget);
}

function applyPlannerOutputLabels(node) {
    if (node?.comfyClass !== "IAMCCS-SuperNodes AU+IMG2VID Exec Planner" || !Array.isArray(node.outputs)) {
        return;
    }
    const labels = ["plan (direct)", "duration", "frames", "segments", "first frames", "left ctx", "summary", "linx", "report"];
    node.outputs.forEach((output, index) => {
        if (output && labels[index]) {
            output.name = labels[index];
        }
    });
}

function addMultiLineStatusBox(node, propertyName, widgetName, fill, stroke, textColor) {
    if (!hasWidgetList(node) || node.widgets.some((widget) => widget?.name === widgetName)) {
        return;
    }
    const statusWidget = {
        type: "custom",
        name: widgetName,
        serialize: false,
        computeSize(width) {
            return [Math.max(220, (width || 260) - 20), 78];
        },
        draw(ctx, widget, nodeRef, widgetWidth, y) {
            const text = nodeRef.properties?.[propertyName] || "";
            if (!text) {
                return;
            }
            const lines = String(text).split("\n").filter(Boolean).slice(0, 4);
            const x = 12;
            const w = Math.max(180, widgetWidth - 24);
            const h = 70;
            const top = y + 4;

            ctx.save();
            ctx.fillStyle = fill;
            ctx.strokeStyle = stroke;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.roundRect(x, top, w, h, 10);
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = textColor;
            ctx.font = "12px Segoe UI";
            ctx.textBaseline = "top";
            lines.forEach((line, index) => {
                ctx.fillText(line, x + 10, top + 8 + index * 15);
            });
            ctx.restore();
        },
    };
    node.widgets.push(statusWidget);
}

function enhanceNodeLayout(node, nodeName) {
    const groups = NODE_GROUPS[nodeName] || [];
    if (!groups.length || !hasWidgetList(node) || !node.widgets.length) {
        return;
    }

    removeStaleSectionButtons(node, nodeName);

    if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Planner") {
        normalizePlannerWidgetOrder(node);
    }
    if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        normalizeRenderWidgetOrder(node);
    }
    if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec VAE") {
        normalizeVaeWidgetOrder(node);
    }

    if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Planner") {
        applyPlannerOutputLabels(node);
        node.widgets = node.widgets.filter((widget) => widget?.name !== "planner_details_preview");
        addPlannerChip(node);
        addMultiLineStatusBox(node, "iamccsPlannerLivePreview", "planner_live_preview", "#1d2f38", "#67a7bb", "#e5f3f7");
    }
    if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
        addStatusBox(node, "iamccsRenderStatus", "render_status_preview", "#1e2f3f", "#6f93ba", "#e4edf7");
    }
    if (nodeName === "IAMCCS-SuperNodes Second Stage") {
        addStatusBox(node, "iamccsSecondStageStatus", "second_stage_status_preview", "#30243d", "#9f77bf", "#f1e7f8");
    }
    if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec VAE") {
        sanitizeVaeWidgetValues(node);
        applyVaeDecodeModeVisibility(node);
    }

    for (let index = groups.length - 1; index >= 0; index -= 1) {
        const group = groups[index];
        const firstWidgetIndex = node.widgets.findIndex((widget) => group.widgets.includes(widget?.name));
        if (firstWidgetIndex >= 0) {
            addSectionButton(node, group, firstWidgetIndex);
        }
    }

    applyPresetConfig(node, nodeName);
    fitNodeToWidgets(node);
}

app.registerExtension({
    name: `IAMCCS.SuperNodesExecUI.${IAMCCS_SUPERNODES_EXEC_UI_VERSION}`,

    async beforeRegisterNodeDef(nodeType, nodeData) {
        const nodeName = nodeData?.name;
        if (!NODE_GROUPS[nodeName]) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = safeCall(`${nodeName}.onNodeCreated.original`, () => onNodeCreated?.apply(this, arguments));
            safeCall(`${nodeName}.onNodeCreated.iamccs`, () => {
                this.properties = this.properties || {};
                // MUST run synchronously, BEFORE configure() assigns widgets_values by index.
                // Section buttons inserted here occupy their index slots so that the null
                // placeholders in widgets_values land on buttons, not on real widgets.
                // (Using setTimeout here was the root cause of NaN on load / after undo.)
                enhanceNodeLayout(this, nodeName);
                if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
                    // Do not apply generation defaults here: workflow reload has
                    // not restored widgets_values yet. Defaults are applied only
                    // from the generation_type callback/user change path.
                    sanitizeRenderWidgetValues(this);
                    applyRenderInternalWidgetVisibility(this);
                    syncRenderLinxWhenReady(this, "node_created_render", { defer: true });
                }
                const config = PRESET_CONFIGS[nodeName];
                const presetWidget = config ? findWidget(this, config.presetWidget) : null;
                if (presetWidget) {
                    const originalCallback = presetWidget.callback;
                    presetWidget.callback = (...args) => safeCall(`${nodeName}.${presetWidget.name}.callback`, () => {
                        coerceWidgetCallbackValue(presetWidget, args);
                        originalCallback?.apply(presetWidget, args);
                        applyPresetConfig(this, nodeName);
                        if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
                            syncRenderLinxWhenReady(this, "preset_widget");
                        }
                        if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec VAE") {
                            applyVaeDecodeModeVisibility(this);
                            syncVaeUpstreamWhenReady(this, "preset_widget");
                        }
                        markCanvasDirty();
                    });
                }
                if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
                    const vaeModeWidget = findWidget(this, "vae_mode");
                    if (vaeModeWidget) {
                        const originalCallback = vaeModeWidget.callback;
                        vaeModeWidget.callback = (...args) => safeCall(`${nodeName}.vae_mode.callback`, () => {
                            coerceWidgetCallbackValue(vaeModeWidget, args);
                            originalCallback?.apply(vaeModeWidget, args);
                            syncRenderLinxWhenReady(this, "vae_mode_widget");
                            markCanvasDirty();
                        });
                        syncRenderLinxWhenReady(this, "vae_mode_install", { defer: true });
                    }
                    const generationTypeWidget = findWidget(this, "generation_type");
                    if (generationTypeWidget) {
                        const originalCallback = generationTypeWidget.callback;
                        generationTypeWidget.callback = (...args) => safeCall(`${nodeName}.generation_type.callback`, () => {
                            coerceWidgetCallbackValue(generationTypeWidget, args);
                            originalCallback?.apply(generationTypeWidget, args);
                            applyRenderGenerationTypeChange(this);
                        });
                    }
                    for (const widgetName of ["backend_mode", "generated_media_duration_seconds", "generated_media_fps", "show_manual_sigmas", "taeltx_preview", "second_stage_mode", "continuity_anchor_mode", "anti_drift_mode"]) {
                        const widget = findWidget(this, widgetName);
                        if (!widget) {
                            continue;
                        }
                        const originalCallback = widget.callback;
                        widget.callback = (...args) => safeCall(`${nodeName}.${widgetName}.callback`, () => {
                            coerceWidgetCallbackValue(widget, args);
                            originalCallback?.apply(widget, args);
                            if (widgetName === "backend_mode") {
                                applyRenderBackendModeDefaults(this, widget.value);
                            }
                            if (widgetName === "continuity_anchor_mode") {
                                applyRenderAnchorModeDefaults(this);
                            }
                            sanitizeRenderWidgetValues(this);
                            applyRenderAnchorVisibility(this);
                            applyRenderSecondStageVisibility(this);
                            applyRenderInternalWidgetVisibility(this);
                            syncRenderLinxWhenReady(this, `${widgetName}_widget`);
                            markCanvasDirty();
                        });
                    }
                    sanitizeRenderWidgetValues(this);
                    applyRenderSecondStageVisibility(this);
                    applyRenderInternalWidgetVisibility(this);
                    syncRenderLinxWhenReady(this, "node_created_render_final", { defer: true });
                }
                if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Planner") {
                    installExecPlannerExplicitPresetSync(this);
                    applyPlannerModeVisibility(this);
                }
                if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec VAE") {
                    sanitizeVaeWidgetValues(this);
                    const decodeModeWidget = findWidget(this, "decode_mode");
                    if (decodeModeWidget) {
                        const originalCallback = decodeModeWidget.callback;
                        decodeModeWidget.callback = (...args) => safeCall(`${nodeName}.decode_mode.callback`, () => {
                            coerceWidgetCallbackValue(decodeModeWidget, args);
                            originalCallback?.apply(decodeModeWidget, args);
                            applyVaeDecodeModeDefaults(this, decodeModeWidget.value);
                            sanitizeVaeWidgetValues(this);
                            applyVaeDecodeModeVisibility(this);
                            syncVaeUpstreamWhenReady(this, "decode_mode_widget");
                            markCanvasDirty();
                        });
                    }
                    const frameRateWidget = findWidget(this, "frame_rate");
                    if (frameRateWidget) {
                        const originalCallback = frameRateWidget.callback;
                        frameRateWidget.callback = (...args) => safeCall(`${nodeName}.frame_rate.callback`, () => {
                            coerceWidgetCallbackValue(frameRateWidget, args);
                            originalCallback?.apply(frameRateWidget, args);
                            sanitizeVaeWidgetValues(this);
                            syncVaeUpstreamWhenReady(this, "frame_rate_widget");
                            markCanvasDirty();
                        });
                    }
                    applyVaeDecodeModeVisibility(this);
                    syncVaeUpstreamWhenReady(this, "node_created_vae", { defer: true });
                }
            });
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            let result;
            this.__iamccsConfiguring = true;
            try {
                safeCall(`${nodeName}.onConfigure.preNormalize`, () => {
                    this.properties = this.properties || {};
                    enhanceNodeLayout(this, nodeName);
                    normalizeConfigureWidgetValues(this, nodeName, info);
                });
                result = safeCall(`${nodeName}.onConfigure.original`, () => onConfigure?.apply(this, arguments));
                safeCall(`${nodeName}.onConfigure.iamccs`, () => {
                    this.properties = this.properties || {};
                    enhanceNodeLayout(this, nodeName);
                    if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
                        sanitizeRenderWidgetValues(this);
                        applyRenderInternalWidgetVisibility(this);
                    }
                    rehydrateSerializedWidgets(this, info?.widgets_values);
                    if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
                        sanitizeRenderWidgetValues(this);
                        markRenderGenerationTypeDefaultsApplied(this);
                        applyRenderInternalWidgetVisibility(this);
                    }
                    if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec VAE") {
                        sanitizeVaeWidgetValues(this);
                        applyVaeDecodeModeVisibility(this);
                    }
                    if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Planner") {
                        installExecPlannerExplicitPresetSync(this);
                    }
                    refreshNodeLayoutState(this, nodeName);
                });
            } finally {
                this.__iamccsConfiguring = false;
            }
                if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
                    syncRenderLinxWhenReady(this, "on_configure_render", { defer: true });
                }
            if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec VAE") {
                syncVaeUpstreamWhenReady(this, "on_configure_vae", { defer: true });
            }
            if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Planner") {
                syncPlannerFpsDownstreamWhenReady(this, "on_configure_planner", { defer: true });
            }
            return result;
        };

        const onWidgetChanged = nodeType.prototype.onWidgetChanged;
        nodeType.prototype.onWidgetChanged = function (name) {
            const result = safeCall(`${nodeName}.onWidgetChanged.original`, () => onWidgetChanged?.apply(this, arguments));
            safeCall(`${nodeName}.onWidgetChanged.iamccs`, () => {
                const changedName = typeof name === "string" ? name : name?.name;
                if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Render" && changedName === "generation_type") {
                    applyRenderGenerationTypeChange(this);
                }
                if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Render" && changedName === "backend_mode") {
                    applyRenderBackendModeDefaults(this);
                    sanitizeRenderWidgetValues(this);
                    applyRenderInternalWidgetVisibility(this);
                    syncRenderLinxWhenReady(this, "widget_changed_backend_mode");
                    markCanvasDirty();
                }
                if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec VAE" && changedName === "decode_mode") {
                    applyVaeDecodeModeDefaults(this);
                    sanitizeVaeWidgetValues(this);
                    applyVaeDecodeModeVisibility(this);
                    syncVaeUpstreamWhenReady(this, "widget_changed_decode_mode");
                }
                if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec VAE" && changedName === "frame_rate") {
                    sanitizeVaeWidgetValues(this);
                    syncVaeUpstreamWhenReady(this, "widget_changed_frame_rate");
                }
                if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Render" && changedName === "generated_media_fps") {
                    sanitizeRenderWidgetValues(this);
                    syncRenderLinxWhenReady(this, "widget_changed_generated_media_fps");
                }
                if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Render" && changedName === "show_manual_sigmas") {
                    sanitizeRenderWidgetValues(this);
                    applyRenderManualSigmasVisibility(this);
                    markCanvasDirty();
                }
                if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Render" && changedName === "taeltx_preview") {
                    sanitizeRenderWidgetValues(this);
                    applyRenderTaeltxPreviewVisibility(this);
                    markCanvasDirty();
                }
            });
            return result;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = safeCall(`${nodeName}.onConnectionsChange.original`, () => onConnectionsChange?.apply(this, arguments));
            safeCall(`${nodeName}.onConnectionsChange.iamccs`, () => {
                if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
                    sanitizeRenderWidgetValues(this);
                    syncRenderLinxWhenReady(this, "connections_change_render", { defer: true });
                }
                if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec VAE") {
                    sanitizeVaeWidgetValues(this);
                    applyVaeDecodeModeVisibility(this);
                    syncVaeUpstreamWhenReady(this, "connections_change_vae", { defer: true });
                }
            });
            return result;
        };

        if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Planner") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                safeCall(`${nodeName}.onExecuted.original`, () => onExecuted?.apply(this, arguments));
                safeCall(`${nodeName}.onExecuted.iamccs`, () => {
                    this.properties = this.properties || {};
                    if (Array.isArray(message?.planner_chip) && message.planner_chip.length > 0) {
                        this.properties.iamccsPlannerChip = String(message.planner_chip[0] || "");
                    }
                    const details = [];
                    const duration = Array.isArray(message?.planned_duration_seconds) && message.planned_duration_seconds.length > 0
                        ? message.planned_duration_seconds[0]
                        : Array.isArray(message?.duration_seconds) && message.duration_seconds.length > 0
                            ? message.duration_seconds[0]
                            : null;
                    const totalFrames = Array.isArray(message?.planned_total_frames) && message.planned_total_frames.length > 0
                        ? message.planned_total_frames[0]
                        : Array.isArray(message?.total_frames) && message.total_frames.length > 0
                            ? message.total_frames[0]
                            : null;
                    const segments = Array.isArray(message?.planned_segment_count) && message.planned_segment_count.length > 0
                        ? message.planned_segment_count[0]
                        : Array.isArray(message?.segment_count) && message.segment_count.length > 0
                            ? message.segment_count[0]
                            : null;
                    if (duration !== null) {
                        this.properties.iamccsPlannerDuration = Number(duration || 0);
                    }
                    if (totalFrames !== null) {
                        this.properties.iamccsPlannerTotalFrames = Number(totalFrames || 0);
                    }
                    if (segments !== null) {
                        this.properties.iamccsPlannerSegmentCount = Number(segments || 0);
                    }
                    if (duration !== null) {
                        details.push(`duration ${Number(duration || 0).toFixed(2)}s`);
                    }
                    if (totalFrames !== null) {
                        details.push(`total ${totalFrames}f`);
                    }
                    if (segments !== null) {
                        details.push(`segments ${segments}`);
                    }
                    if (Array.isArray(message?.recommended_overlap_frames) && message.recommended_overlap_frames.length > 0) {
                        details.push(`overlap ${message.recommended_overlap_frames[0]}f`);
                    }
                    if (Array.isArray(message?.recommended_audio_left_context_s) && message.recommended_audio_left_context_s.length > 0) {
                        details.push(`left ctx ${Number(message.recommended_audio_left_context_s[0] || 0).toFixed(2)}s`);
                    }
                    const plannerReport = Array.isArray(message?.planning_report) && message.planning_report.length > 0
                        ? message.planning_report[0]
                        : Array.isArray(message?.report) && message.report.length > 0
                            ? message.report[0]
                            : null;
                    if (plannerReport !== null) {
                        details.push(String(plannerReport || ""));
                    }
                    updateExecPlannerLivePreview(this);
                    if (details.length > 0) {
                        this.properties.iamccsPlannerDetails = details.join("\n");
                        markCanvasDirty();
                    }
                });
            };
        }

        if (nodeName === "IAMCCS-SuperNodes AU+IMG2VID Exec Render") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                safeCall(`${nodeName}.onExecuted.original`, () => onExecuted?.apply(this, arguments));
                safeCall(`${nodeName}.onExecuted.iamccs`, () => {
                    if (Array.isArray(message?.report) && message.report.length > 0) {
                        this.properties = this.properties || {};
                        this.properties.iamccsRenderStatus = String(message.report[0] || "").split("\n")[0];
                        markCanvasDirty();
                    }
                });
            };
        }

        if (nodeName === "IAMCCS-SuperNodes Second Stage") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                safeCall(`${nodeName}.onExecuted.original`, () => onExecuted?.apply(this, arguments));
                safeCall(`${nodeName}.onExecuted.iamccs`, () => {
                    if (Array.isArray(message?.report) && message.report.length > 0) {
                        this.properties = this.properties || {};
                        this.properties.iamccsSecondStageStatus = String(message.report[0] || "").split("\n")[0];
                        markCanvasDirty();
                    }
                });
            };
        }
    },
});





