// SPDX-License-Identifier: GPL-3.0-or-later
import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { mountAheadRoom } from './iamccs_ahead_control_room.js';
import { rigMedia } from './iamccs_h3_rig.js';
import { createSwapDuration, installSwapDurationQueueGuard } from './iamccs_h3_swap_duration.js';

const NODE_TYPE = "IAMCCS_ShotboardH3SettingsPro";
const SHOTBOARD_TYPE = "IAMCCS_MiniMaxH3ShotPlanner";
const BRIDGE_TYPES = new Set(["IAMCCS_CineH3Input", "IAMCCS_CineH3FunControlInput", "IAMCCS_MiniMaxH3FunControlInput"]);
const SHOTBOARD_OWNED = new Set(["task_mode"]);
const OPEN_H3_REFINE_FIELDS = ["h3_open_refine_enabled", "h3_open_refine_scale", "h3_open_refine_model_name", "h3_open_refine_precision", "h3_open_refine_device", "h3_open_refine_temporal_chunking", "h3_open_refine_lora_name", "h3_open_refine_strength", "h3_open_refine_distill_lora_name", "h3_open_refine_distill_strength", "h3_open_refine_sigma_preset", "h3_open_refine_custom_sigmas", "h3_open_refine_seed", "h3_open_refine_prompt"];
const AUTO_IMPORT_BLOCKED = new Set([
  "global_prompt", "timeline_data", "image_paths", "frame_rate", "audio_mode",
  "guide_policy", "min_guide_gap_seconds", "max_guides", "default_force", "promptrelay_epsilon",
  "ltx_round_mode", "image_width", "image_height", "image_resize_method", "image_multiple_of", "img_compression",
  "seed_control_after_generate_compat", "h3_advisor_state",
]);
const INTERNAL = new Set([
  "seed_control_after_generate_compat", "h3_advisor_state",
  // Legacy/detailed multi-stage knobs remain serialized for workflow compatibility,
  // but Settings PRO exposes only the explicit toggle + simple profile dropdown.
  "longvid_pianosequenza_stage_count", "longvid_pianosequenza_split_mode",
  "longvid_pianosequenza_auto_profile",
  "longvid_pianosequenza_manual_stage1_steps",
  "longvid_pianosequenza_manual_stage2_steps",
  "longvid_pianosequenza_manual_stage3_steps",
  // Extended AV future/deferred runtime knobs remain serialized for forward
  // compatibility, but are intentionally hidden until their execution path is wired.
  "extended_av_level_lock_mode", "extended_av_joint_refine_mode",
  "extended_av_joint_window_frames", "extended_av_joint_window_overlap",
  // Phase B: auto extension is compiler-owned once an authored slot exceeds
  // 15 s. Keep the former manual switch load-compatible but off the PRO UI.
  "h3_auto_extend_mode", "h3_auto_extend_continue_audio", "keyframe_joint_latent_new",
  "extended_av_soft_audio_handover_ms", "extended_av_boundary_polish_ms", "extended_av_boundary_polish_strength",
]);
// These widgets remain in the exact standard Settings schema for saved-workflow
// compatibility.  They are mirrored/editorial aliases which the standard
// compiler intentionally does not publish to CineLinX, so the PRO surface does
// not present them as a second source of truth.
const COMPATIBILITY_ONLY = new Set([
  "h3_guard_reference_memory", "h3_guard_repair_denoise", "h3_guard_repair_steps", "h3_guard_motion_tolerance",
  "frame_rate", "guide_policy", "min_guide_gap_seconds", "max_guides", "default_force",
  "promptrelay_epsilon", "ltx_round_mode", "image_width", "image_height",
  "image_resize_method", "image_multiple_of", "img_compression",
]);
const GROUPS = [
  { id: "continuity_guard", label: "CONTINUITY MONITOR", title: "Read-only Extended AV diagnostics. Repair has been retired; this monitor does not prevent degradation.", modeEngine: true, fields: ["h3_guard_mode", "h3_guard_detail_ratio"] },
  { id: "ahead", label: "AHEAD CONTROL ROOM", title: "LatentGoAhead seam controls", fields: [] },
  { id: "extension", label: "LONG TAKE · EXTEND", title: "Automatic native AV extension for one authored shot beyond 15 seconds", modeEngine: true, fields: ["extended_av_profile", "extended_av_custom_window_frames", "extended_av_custom_overlap_frames"] },
  { id: "continuation", label: "SAVED TAKE · CONTINUATION", title: "Start a new generation from a previously saved AV checkpoint", fields: ["h3_continuation_enabled", "h3_continuation_save_enabled", "h3_continuation_checkpoint", "h3_continuation_context_frames", "h3_continuation_handover_mode", "h3_continuation_manual_tail_frames", "h3_continuation_visual_handover", "h3_continuation_run_and_gun_enabled", "h3_continuation_run_and_gun_join", "h3_continuation_soft_video_frames", "h3_continuation_soft_video_curve", "h3_continuation_soft_audio_ms"] },
  { id: "refmod", label: "REFMOD", title: "Reference latent library", fields: ["h3_refmod_enabled", "h3_refmod_name", "h3_refmod_strength", "h3_refmod_retention", "h3_refmod_max_tokens"] },
  { id: "assistant", label: "MODE ASSISTANT", title: "Guided setup", assistant: true, fields: [] },
  { id: "engine_reference_image", label: "IMAGE GENERATOR", title: "Image Generator · Reference engine", modeEngine: true, fields: ["h3_reference_image_source", "h3_reference_image_preset", "h3_reference_image_vram_profile", "h3_reference_image_detail_enabled", "h3_reference_image_detail_model", "h3_reference_image_detail_strength", "reference_role_1", "reference_role_2", "reference_role_3", "reference_role_4", "reference_resize_policy", "reference_resize_megapixels", "reference_resize_filter", "ref_image_size"] },
  { id: "engine_t2va", label: "T2VA", title: "T2VA · Text / hard-cut shot engine", modeEngine: true, fields: ["audio_mode", "prompt_mapping"] },
  { id: "engine_i2va", label: "I2VA", title: "I2VA · Opening-image shot engine", modeEngine: true, fields: ["audio_mode", "reference_resize_policy", "reference_resize_megapixels", "reference_resize_filter", "ref_image_size"] },
  { id: "engine_flf", label: "FLF KEYFRAMES", title: "FLF · Authored keyframe-to-keyframe engine", modeEngine: true, fields: ["audio_mode", "flf_join_mode", "flf_overlap_frames", "flf_continuity_mode", "flf_continuity_tail_frames", "flf_continuity_audio"] },
  { id: "engine_flf_continuous", label: "FL2VA NATIVE AV", title: "FL2VA · Native AV keyframe-chain engine", modeEngine: true, fields: ["audio_mode", "flf_continuity_tail_frames", "flf_continuity_audio", "flf_join_mode", "flf_overlap_frames"] },
  { id: "engine_ref2va", label: "REF2VA", title: "REF2VA · Identity / object / style reference engine", modeEngine: true, fields: ["h3_omni_enabled", "audio_mode", "reference_role_1", "reference_role_2", "reference_role_3", "reference_role_4", "reference_video_role", "reference_audio_role", "reference_resize_policy", "reference_resize_megapixels", "reference_resize_filter", "ref_image_size"] },
  { id: "engine_ref2vid", label: "REF2VID AUDIO", title: "REF2VID · Locked audio performance engine", modeEngine: true, fields: ["audio_mode", "voice_reference_picture_index", "reference_role_1", "reference_role_2", "reference_role_3", "reference_role_4", "reference_video_role", "reference_audio_role", "reference_resize_policy", "reference_resize_megapixels", "reference_resize_filter", "ref_image_size"] },
  { id: "engine_multishot", label: "LONG MULTI-SHOT", title: "Long Multi-Shot · Hard Cut engine", modeEngine: true, fields: ["h3_multishot_boundary_mode", "audio_mode", "reference_resize_policy", "reference_resize_megapixels", "reference_resize_filter", "prompt_mapping"] },
  { id: "engine_keyframe_continuous", label: "LONG KEYFRAME", title: "Long Keyframe · selectable native or Extended AV backend", modeEngine: true, fields: ["longvid_continuous_engine", "flf_continuity_tail_frames", "flf_continuity_audio", "extended_av_profile", "extended_av_custom_window_frames", "extended_av_custom_overlap_frames", "extended_av_pin_mode", "extended_av_mask_profile", "audio_mode", "reference_resize_policy", "reference_resize_megapixels", "reference_resize_filter", "ref_image_size"] },
  { id: "engine_true_long", label: "TRUE LONG", title: "Pianosequenza True Long · shared AV boundary", modeEngine: true, fields: ["flf_continuity_tail_frames", "flf_continuity_audio", "audio_mode", "ref_image_size"] },
  { id: "engine_keyframe_joint", label: "KEYFRAME JOINT", title: "Keyframe Joint · Native / latent-new engine", modeEngine: true, fields: ["audio_mode", "keyframe_joint_latent_new", "reference_resize_policy", "reference_resize_megapixels", "reference_resize_filter", "ref_image_size"] },
  { id: "engine_object_swap", label: "OBJECT SWAP", title: "Object Swap · V2VA engine", modeEngine: true, fields: ["audio_mode", "v2v_guide_mode", "v2v_source_range_policy", "v2v_source_offset_seconds", "v2v_source_fit", "v2v_source_end_policy", "v2v_audio_pairing", "reference_role_1", "reference_role_2"] },
  { id: "overview", label: "1 · NATIVE", title: "Native H3 canvas and programme duration", fields: ["duration_seconds", "width", "height", "upscale_link_to_native", "upscale_link_factor", "reference_resize_policy", "reference_resize_megapixels", "reference_resize_filter", "prompt_mapping"] },
  { id: "audio", label: "2 · AUDIO", title: "Audio authority and EXTEND seam polish", fields: ["audio_mode", "reference_audio_role", "voice_reference_picture_index", "extended_av_soft_audio_handover_ms", "extended_av_boundary_polish_ms", "extended_av_boundary_polish_strength"] },
  { id: "memory", label: "3 · MEMORY", title: "VRAM preset", fields: ["performance_profile", "motion_context_window_frames", "text_encoder_device", "h3_exact_profile", "h3_exact_chunk_rows", "h3_exact_precision_mode", "h3_exact_qkv_streaming", "h3_exact_attention_memory", "h3_clipproj_profile", "h3_clipproj_load_mode", "vram_clean_before_decode"] },
  { id: "sampling", label: "4 · SAMPLE", title: "Native H3 sampling", fields: ["seed", "seed_policy", "seed_stride", "steps", "sampler_name", "scheduler", "denoise", "shift_video", "shift_audio"] },
{ id: "speed", label: "5 · SPEED", title: "Acceleration recipe", fields: ["h3_attention_backend", "h3_veda_enabled", "h3_veda_predictor", "h3_veda_verbose", "acceleration", "turbo_mode", "turbo_lora_name", "turbo_strength", "turbo_sampler_mode", "fused_turbo_model_name", "fused_turbo_sigma_preset", "pdd_lora_name", "pdd_strength", "secondary_lora_enabled", "secondary_lora_name", "secondary_lora_strength", "secondary_lora_continuation_only", "ref_image_size", "sol_conditioning", "spectrum_profile", "h3_sla_sparsity", "h3_sla_dense_last_steps"] },
  { id: "direction", label: "6 · DIRECT · LEGACY", title: "Legacy Direct compatibility", fields: ["reference_role_1", "reference_role_2", "reference_role_3", "reference_role_4", "reference_video_role", "v2v_guide_mode", "v2v_source_range_policy", "v2v_source_offset_seconds", "v2v_source_fit", "v2v_source_end_policy", "v2v_audio_pairing", "flf_join_mode", "flf_overlap_frames", "flf_continuity_mode", "flf_continuity_tail_frames", "flf_continuity_audio", "longvid_guide_window_policy", "keyframe_joint_latent_new", "longvid_terminal_endpoint_mode", "longvid_pianosequenza_2stage_enabled"] }, // IAMCCS_LONGVID_ENDPOINT_STRATEGIES_V1
  { id: "control", label: "CONTROLNET", title: "H3 Fun ControlNet", contextual: "control", modeEngine: true, fields: ["h3_controlnet_enabled", "h3_controlnet_name", "h3_controlnet_kind", "h3_controlnet_preprocess_resolution", "h3_controlnet_strength", "h3_controlnet_start_percent", "h3_controlnet_end_percent", "h3_controlnet_frame_scope", "h3_controlnet_end_policy", "h3_controlnet_use_video_audio", "h3_controlnet_dwpose_torch_gpu"] },
  { id: "vfx_edit", label: "VFX EDIT", title: "REF2VA · VFX EDIT · aligned video guide", modeEngine: true, fields: ["h3_vfx_lora_name", "h3_vfx_lora_strength", "h3_edit_video_source", "upscale_enabled", "upscale_mode", "upscale_width", "upscale_height"] },
  { id: "swap_lora", label: "SWAP -LorA", title: "SUBJECT SWAP -LorA · Full-frame Ref2VA", modeEngine: true, fields: ["h3_edit_video_source", "h3_swap_lora_name", "h3_swap_lora_strength", "h3_swap_realism_lora_name", "h3_swap_realism_strength", "h3_upres_seed", "h3_upres_model_name", "h3_upres_steps", "h3_upres_denoise", "h3_upres_sampler", "h3_upres_scheduler", "upscale_width", "upscale_height", "v2v_source_range_policy", "v2v_source_offset_seconds", "v2v_source_fit", "v2v_source_end_policy", "audio_mode"] },
  { id: "face", label: "SAM3 SWAP", title: "SAM3 Subject Swap", contextual: "face", modeEngine: true, fields: ["h3_faceswap_generate_new_audio", "h3_faceswap_sam_model", "h3_faceswap_birefnet_model", "h3_faceswap_mask_prompt", "h3_faceswap_threshold", "h3_faceswap_objects", "h3_faceswap_cleanup_threshold", "h3_faceswap_cleanup_shrink", "h3_faceswap_cleanup_min_frames", "h3_faceswap_cleanup_edge_grow", "h3_faceswap_crop_scale", "h3_faceswap_crop_megapixels", "h3_faceswap_grow_spatial", "h3_faceswap_grow_temporal", "h3_faceswap_feather"] },
  { id: "face_refine", label: "FACE REFINE", title: "Face Refinement", fields: ["face_detailer_enabled", "face_detailer_profile", "face_detailer_mode", "face_detailer_max_faces", "face_detailer_min_coverage", "face_detailer_use_sam_mask", "face_detailer_canvas_width", "face_detailer_canvas_height", "face_detailer_crop_factor", "face_detailer_confidence", "face_detailer_steps", "face_detailer_denoise", "face_detailer_window_frames", "face_detailer_window_overlap", "face_detailer_strength_small_face", "face_detailer_strength_large_face", "face_detailer_blend", "face_detailer_paste_region", "face_detailer_feather"] },
  { id: "scout", label: "SCOUT", title: "Candidate seed scout", fields: ["h3_r40_seed_scout_enabled", "h3_r40_candidate_count", "h3_r40_seed_stride", "h3_r40_preview_max_frames", "h3_r40_sparse_enabled", "h3_r40_sparse_video_budget", "h3_r40_sparse_denser_edges"] },
  { id: "finish", label: "7 · OUTPUT", title: "Delivery", fields: ["h3_editor_enabled", "upscale_mode", "upscale_enabled", "upscale_width", "upscale_height", "h3_pixel_tiled_method", "h3_pixel_tiled_model_name", "h3_pixel_tiled_tile_size", "h3_pixel_tiled_overlap", "upscale_prompt", "upscale_sage", "upscale_seed_offset", "wan_upscale_denoise", "ltx_seam_safe", "ltx_detailer_enabled", "ltx_detailer_lora_name", "ltx_detailer_strength", "ltx_4k_enabled", "ltx_4k_quality", "ltx_looper_temporal_tile_size", "ltx_looper_temporal_overlap", "ltx_looper_guiding_strength", "ltx_looper_overlap_strength", "ltx_looper_cond_image_strength", "ltx_looper_horizontal_tiles", "ltx_looper_vertical_tiles", "ltx_looper_spatial_overlap", "h3_upres_model_name", "h3_upres_precision", "h3_upres_device", "h3_upres_keep_models_resident", "h3_fast_latent_temporal_chunk", "h3_fast_latent_stage2_window", "h3_fast_latent_stage2_overlap", "h3_fast_latent_stage2_guides", "h3_upres_seed", "h3_upres_steps", "h3_upres_denoise", "h3_upres_sampler", "h3_upres_scheduler", "h3_upres_temporal_chunk", "h3_upres_temporal_overlap", "h3_upres_anchor_strength", "h3_upres_tile_width", "h3_upres_tile_height", "h3_upres_overlap_width", "h3_upres_overlap_height", "h3_upres_fade_width", "h3_upres_fade_height", "h3_upres_min_tile_size", "h3_upres_overlap_mode", "h3_upres_overlap_blend", "h3_upres_rtx_enabled", "h3_upres_rtx_quality", "h3_upres_pixel_groups", "h3_upres_window_frames", "h3_upres_window_overlap", "h3_upres_pixel_method"] },
  { id: "advanced", label: "TECHNICAL", title: "Technical controls", dynamic: true, fields: [] },
];
GROUPS.find(group => group.id === "finish").fields.push(...OPEN_H3_REFINE_FIELDS);
const ENGINE_IDS = Object.freeze(GROUPS.filter((group) => group.modeEngine || ["ahead", "continuation", "refmod", "face_refine", "scout"].includes(group.id)).map((group) => group.id));
const ENGINE_ID_SET = new Set(ENGINE_IDS);
const ASSIGNED = new Set(GROUPS.flatMap((group) => group.fields));
const FUNCTIONAL_LAYOUT = {
  extension: [
    ["EXTENDED AV · MASKED + EXACT", ["extended_av_profile", "extended_av_custom_window_frames", "extended_av_custom_overlap_frames"]],
  ],
  audio: [
    ["AUDIO AUTHORITY", ["audio_mode", "reference_audio_role", "voice_reference_picture_index"]],
    ["EXTEND · SOFT AV SEAM", ["extended_av_soft_audio_handover_ms", "extended_av_boundary_polish_ms", "extended_av_boundary_polish_strength"]],
  ],
  memory: [
    ["VRAM PRESET", ["performance_profile", "motion_context_window_frames", "text_encoder_device", "vram_clean_before_decode"]],
    ["ATTENTION MEMORY", ["h3_exact_profile", "h3_exact_chunk_rows", "h3_exact_precision_mode", "h3_exact_qkv_streaming", "h3_exact_attention_memory"]],
    ["CLIPPROJ", ["h3_clipproj_profile", "h3_clipproj_load_mode"]],
  ],
  direction: [
    ["MOTION CONTEXT", ["flf_continuity_tail_frames", "flf_continuity_audio"]],
    ["REFERENCE ROLES", ["reference_role_1", "reference_role_2", "reference_role_3", "reference_role_4", "reference_video_role"]],
    ["V2VA SOURCE", ["v2v_guide_mode", "v2v_source_range_policy", "v2v_source_offset_seconds", "v2v_source_fit", "v2v_source_end_policy", "v2v_audio_pairing"]],
    ["FLF CONTINUITY", ["flf_join_mode", "flf_overlap_frames", "flf_continuity_mode", "flf_continuity_tail_frames", "flf_continuity_audio"]],
    ["LONGVID GUIDE WINDOWS", ["longvid_guide_window_policy", "motion_context_window_frames"]],
    ["JOINT / LATENT NEW", ["keyframe_joint_latent_new"]],
    ["LONGVID ENDPOINT", ["longvid_terminal_endpoint_mode", "longvid_pianosequenza_2stage_enabled", "longvid_pianosequenza_stage_profile"]],
  ],
  speed: [
    ["ATTENTION & MEMORY", ["h3_attention_backend", "sol_conditioning"]],
    ["VEDA · EXPERIMENTAL SPARSE ATTENTION", ["h3_veda_enabled", "h3_veda_predictor", "h3_veda_verbose"]],
    ["ENGINE", ["acceleration", "h3_sla_sparsity", "h3_sla_dense_last_steps"]],
    ["TURBO LORA", ["turbo_mode", "turbo_lora_name", "turbo_strength", "turbo_sampler_mode"]],
    ["PDD 8-STEP", ["pdd_lora_name", "pdd_strength"]],
    ["FUSED MODEL", ["fused_turbo_model_name", "fused_turbo_sigma_preset"]],
    ["EXACT ATTENTION & CLIPPROJ", ["h3_exact_profile", "h3_exact_chunk_rows", "h3_exact_precision_mode", "h3_exact_qkv_streaming", "h3_exact_attention_memory", "h3_clipproj_profile", "h3_clipproj_load_mode"]],
  ],
  face_refine: [
    ["FACE REFINEMENT", ["face_detailer_enabled", "face_detailer_profile", "face_detailer_mode", "face_detailer_max_faces", "face_detailer_min_coverage", "face_detailer_use_sam_mask"]],
    ["FACE TRACKING", ["face_detailer_crop_factor", "face_detailer_confidence", "face_detailer_window_frames", "face_detailer_window_overlap", "face_detailer_paste_region", "face_detailer_feather"]],
    ["FACE DETAIL", ["face_detailer_canvas_width", "face_detailer_canvas_height", "face_detailer_steps", "face_detailer_denoise", "face_detailer_strength_small_face", "face_detailer_strength_large_face", "face_detailer_blend"]],
  ],
  finish: [
    ["OUTPUT", ["rife_mode", "upscale_enabled", "upscale_mode", "upscale_width", "upscale_height", "upscale_prompt", "upscale_sage", "upscale_seed_offset", "wan_upscale_denoise"]],
    ["OPEN-H3 REFINEMENT · AFTER ONE NATIVE GENERATION", OPEN_H3_REFINE_FIELDS],
    ["PIXEL SAFE · LOW VRAM", ["h3_pixel_tiled_method", "h3_pixel_tiled_model_name", "h3_pixel_tiled_tile_size", "h3_pixel_tiled_overlap"]],
    ["LTX DELIVERY", ["ltx_seam_safe", "ltx_detailer_enabled", "ltx_detailer_lora_name", "ltx_detailer_strength", "ltx_4k_enabled", "ltx_4k_quality", "ltx_looper_temporal_tile_size", "ltx_looper_temporal_overlap", "ltx_looper_guiding_strength", "ltx_looper_overlap_strength", "ltx_looper_cond_image_strength", "ltx_looper_horizontal_tiles", "ltx_looper_vertical_tiles", "ltx_looper_spatial_overlap"]],
    ["H3 2-PASS MODEL", ["h3_upres_model_name", "h3_upres_precision", "h3_upres_device", "h3_upres_keep_models_resident"]],
    ["FAST LATENT GPU BUDGET", ["h3_fast_latent_temporal_chunk", "h3_fast_latent_stage2_window", "h3_fast_latent_stage2_overlap", "h3_fast_latent_stage2_guides"]],
    ["H3 DETAIL / SECOND-PASS SAMPLING", ["h3_upres_seed", "h3_upres_steps", "h3_upres_denoise", "h3_upres_sampler", "h3_upres_scheduler", "h3_upres_anchor_strength"]],
    ["TEMPORAL WINDOWS", ["h3_upres_temporal_chunk", "h3_upres_temporal_overlap"]],
    ["SPATIAL TILES", ["h3_upres_tile_width", "h3_upres_tile_height", "h3_upres_overlap_width", "h3_upres_overlap_height", "h3_upres_fade_width", "h3_upres_fade_height", "h3_upres_min_tile_size", "h3_upres_overlap_mode", "h3_upres_overlap_blend"]],
    ["PIXEL REFINE", ["h3_upres_pixel_groups", "h3_upres_window_frames", "h3_upres_window_overlap", "h3_upres_pixel_method"]],
    ["RTX DELIVERY", ["h3_upres_rtx_enabled", "h3_upres_rtx_quality"]],
  ],
};
const MODE_CHOICES = [
  ["IMAGE GENERATOR · REFERENCES", "reference_image", "Create one exact 4K still through the R42 SatoDive branch. The reference can come from a Prompter REF2V visual followed by Inject, or from the direct image socket. Optional same-size Skin Detail is controlled by Settings PRO and is off by default."],
  ["9 · AUTO · READ CURRENT SHOTBOARD", "auto_from_shotboard", "Import the board's current generation settings once; Settings PRO then becomes the technical master, including duration, and syncs that duration back to Shotboard. Prompts, guides, media and audio remain Shotboard truth."],
  ["1 · T2VA · TEXT / HARD-CUT SHOTS", "t2va", "Text-only shots. Separate authored slots remain hard cuts."],
  ["2 · I2VA · OPENING IMAGE / HARD-CUT SHOTS", "i2va", "One image per shot. Multiple image boxes are independent shots separated by hard cuts."],
  ["5 · LONG TAKE CONTINUOUS · IMAGE", "fl2va_extended_av", "One authored shot beyond the native H3 window. Picture 1 is the opener; RAW AV overlap carries motion and audio across internal chunks. This is EXTENSION, not resume-from-checkpoint continuation."],
  ["6 · T2VA CONTINUOUS", "t2va_continuous", "One text-only take; beyond 15 seconds the B1 MASKED + EXACT engine extends it. Separate authored slots remain independent shots."],
  ["3 · FLF · KEYFRAME-TO-KEYFRAME", "fl2va_stable", "A→B, B→C with authored destination keyframes. This is not long-take extension."],
  ["4 · FL2VA · KEYFRAME CHAIN + NATIVE AV", "fl2va_continuous", "Carry a native AV context between authored FL2VA destination chunks; distinct from automatic long-take extension."],
  ["REF2VA · VFX EDIT", "ref2va_vfx_edit", "Edit a source video with the VFX LoRA and aligned AddGuide; native or shared second-pass delivery."],
  ["FASTH3 OMNI · 8 STEP", "ref2va_omni", "Full Omni checkpoint: one joint AV target up to 362 frames. Longer timelines use experimental LongVid AV-tail windows with Soft AV handover. Ordered image, video and voice references; dedicated loader."],
  ["7 · REF2VA · REFERENCES", "ref2va", "Reference blocks for identity, object or style; no temporal carry."],
  ["8 · REF2VID · AUDIO PERFORMANCE", "ref2vid_lipsync", "Reference image plus one locked AudioBoard performance per hard-cut shot."],
  ["10 · LONG MULTI-SHOT · FL2VA", "longvid_multishot_hardcut", "Choose joint native multi-shot or isolated camera shots for exact editorial cuts. Isolated shots discard a generated tail handle and use each local shot prompt; Custom Audio keeps the full performance on one timeline."],
  ["11 · LONGVID · TRUE LONG PSkf", "longvid_keyframe_continuous", "One continuous authored timeline powered exclusively by Pianosequenza True Long with native video/audio PSkf checkpoints."],
  ["12 · KEYFRAME JOINT · NATIVE", "keyframe_joint_native", "Multiple positioned keyframes inside one native H3 sample window. Latent New remains an explicit experimental option inside this mode's Engine panel."],
  ["LATENTGOAHEAD · EXPERIMENTAL", "latent_go_ahead", "Continue with original AV latent history in past time. Compatible acceleration and H3 generated audio; masked audio excludes progressive sampling. Requires the LatentGoAhead branch; no pixel crossfade. Context controls are on that node."],
  ["LONG MULTI-SHOT", "longvid_motion_context", "Timed image anchors inside H3 windows, with native AV carry. Several slots can share one sample; an anchor does not lock the entire slot or guarantee a smooth transition. For separate shots use I2VA hard cuts; for connected destinations use Long Continuous Guided."],
  ["LONG GUIDED DESTINATION CHAIN", "longvid_continuous_guided", "One evolving guided sequence where each image is an authored destination; distinct from single-opener Extended AV."],
  ["FL2VA DESTINATION CHAIN · CONTINUOUS AV", "longvid_masked_loop_guided", "Stable phase-aligned full AV-latent handover between authored FL2VA destinations."],
  ["GUIDED AV LOOP · EXPERIMENTAL", "guided_av_loop_experimental", "One full AV master latent with internal masked windows and positioned guides. No outer decoded clip concatenation; experimental."],
  ["CONTROL VIDEO", "v2va_controlnet", "Drive pose, depth or edges from video."],
  ["OBJECT SWAP", "v2va_object_swap", "Replace a tracked object in source video."],
  ["SAM3 SUBJECT SWAP", "v2va_face_swap", "Track a source-video subject and regenerate the identity from Picture 1."],
  ["SUBJECT SWAP -LorA", "v2va_face_swap_lora", "Replace the character through full-frame Ref2VA and character/realism LoRAs. Source video + Picture 1; no SAM tracking or crop."],
];
const UNLISTED_MODES = new Set(["reference_image"]);
const HIDDEN_MODES = new Set(["latent_go_ahead", "longvid_guides", "longvid_motion_context", "longvid_continuous_guided", "longvid_masked_loop_guided", "guided_av_loop_experimental"]);
const MODE_ENGINE_GROUP = Object.freeze({
  reference_image: "engine_reference_image", t2va: "engine_t2va", i2va: "engine_i2va",
  fl2va_stable: "engine_flf", fl2va_continuous: "engine_flf_continuous",
  fl2va_extended_av: "extension", t2va_continuous: "extension",
  ref2va_vfx_edit: "vfx_edit", ref2va_omni: "engine_ref2va", ref2va: "engine_ref2va", ref2vid_lipsync: "engine_ref2vid",
  longvid_multishot_hardcut: "engine_multishot", longvid_keyframe_continuous: "engine_true_long",
  pianosequenza_true_long: "engine_true_long",
  keyframe_joint_native: "engine_keyframe_joint", v2va_controlnet: "control",
  v2va_object_swap: "engine_object_swap", v2va_face_swap: "face", v2va_face_swap_lora: "swap_lora",
});
function modeEngineGroup(node, mode = assistantModeKey(node)) {
  const key = String(mode || "");
  return MODE_ENGINE_GROUP[key] || null;
}
const FRIENDLY_VALUES = { // IAMCCS_LONGVID_PIANOSEQUENZA_V2_UPSTREAM_PARITY
  h3_ultimate_tiled: "H3 DETAIL · 1-STEP REFINE",
  safe_8_12gb: "SAFE · 8–12 GB · 192F / 39F overlap",
  balanced_12_16gb: "BALANCED · 12–16 GB · 243F / 39F overlap",
  quality_20_24gb: "QUALITY · 20–24 GB · 294F / 90F overlap",
  max_32gb_plus: "MAX · 32 GB+ · 345F / 141F overlap",
  masked: "MASKED · preserve RAW AV",
  both: "BOTH · masked + native guide rows",
  exact: "EXACT · hard preserved overlap",
  runway: "RUNWAY · soft video release near join",
  soft: "SOFT · partially regenerated video overlap",
  same_time_equal_power: "SAME-TIME · equal-power handover",
  motion_context_auto_chain: "LEGACY · motion-context auto chain",
  hard_image: "HARD IMAGE · exact terminal keyframe (legacy)",
  latent_free_closure: "LATENT FREE CLOSURE · soft semantic terminal refinement",
  pianosequenza_linear: "PIANOSEQUENZA LINEAR · static direct-latent release",
  pianosequenza_drift: "PIANOSEQUENZA DRIFT · sigma-matched dynamic release",
  pianosequenza_native: "PIANOSEQUENZA NATIVE · native latent-tail guide",
  pianosequenza_phase: "PIANOSEQUENZA PHASE · canonical phase-aligned latent guides",
  pianosequenza_frozen: "PIANOSEQUENZA FROZEN · exact frozen latent prefix",
  pianosequenza_hd: "PIANOSEQUENZA HD · progressive native-tail continuity",
  latent_free: "LEGACY LATENT FREE → Closure",
  rtx_xx60_safe: "8–12 GB VRAM · Safe", rtx_xx70_balanced: "12–16 GB VRAM · Balanced",
  rtx_xx80_quality: "16–24 GB VRAM · Quality", rtx_xx90_max: "24 GB+ VRAM · Maximum",
  rtx3060_draft: "8–12 GB VRAM · Draft (legacy)", rtx3060_balanced: "8–12 GB VRAM · Balanced (legacy)",
  rtx3060_turbo: "8–12 GB VRAM · Fast (legacy)", auto_3060: "Automatic · 8–12 GB VRAM (legacy)",
  fixed_per_generation: "Fixed · one seed for the whole generation", random_per_generation: "Random · once per generation", fixed_per_chunk: "Fixed sequence · base + chunk × stride", random_per_chunk: "Random · each chunk (replayable)",
  rtx_xx60_8_12gb_124: "8–12 GB VRAM · 124 frames", rtx_xx70_12_16gb_209: "12–16 GB VRAM · 209 frames",
  rtx_xx80_16_24gb_294: "16–24 GB VRAM · 294 frames", rtx_xx90_24gb_362: "24 GB+ VRAM · 362 frames",
  rtx3060_12gb_124: "8–12 GB VRAM · 124 frames (legacy)", rtx3060_12gb_209: "12 GB VRAM · 209 frames (legacy)",
  native: "Native exact attention", low_vram_auto: "Exact attention · automatic low-VRAM",
  h3_exact: "H3 exact attention", h3_sage: "Sage attention", sage: "Sage attention · legacy",
  sol_low_vram: "SOL sparse attention · low-VRAM", sol_adaptive_safe: "SOL adaptive · safe",
  sol_adaptive_balanced: "SOL adaptive · balanced", sage_sol: "SOL low-VRAM · legacy alias",
  adaptive_safe: "Adaptive attention · safe", spectrum: "Spectrum attention", sage_spectrum: "Sage + Spectrum",
  comfy_kitchen: "Comfy Kitchen optimized attention", auto_3060: "Automatic · 8–12 GB VRAM (legacy)",
  pdd_native_8step: "PDD · complete 8-step recipe", fasth3_dense_6step: "FastH3 · complete 6-step recipe",
  matlowai_fused_turbo_manual_sigma: "Fused Fast · checkpoint-specific recipe", h3_sla: "SLA · complete 4-step recipe",
  "2_stage_full": "2 STAGE FULL · LOW H3 → HIGH H3",
  "2_stage_safe_delivery": "2 STAGE SAFE · LOW H3 → learned lift → HIGH output",
  "3_stage_safe_mid": "3 STAGE SAFE · LOW H3 → MID H3 → learned lift → HIGH output",
  "3_stage_full": "3 STAGE FULL · LOW H3 → MID H3 → HIGH H3",
};

const H3_NATIVE_RESOLUTION_PRESETS = Object.freeze([
  ["H3 IMAGE · 4K UHD · 3840 × 2160", "3840x2160"],
  ["TEST · 416 × 288 · LOW-RES", "416x288"],
  ["H · 5:3 · 640 × 384 · LIGHT", "640x384"],
  ["H · 5:3 · 960 × 576", "960x576"],
  ["H · 5:3 · 1120 × 672", "1120x672"],
  ["H · 5:3 · 1280 × 768", "1280x768"],
  ["H · 5:3 · 1600 × 960", "1600x960"],
  ["H · 5:3 · 1920 × 1152", "1920x1152"],
  ["H · ≈16:9 · 736 × 416 · FHD LIGHT", "736x416"],
  ["H · ≈16:9 · 768 × 448 · Draft", "768x448"],
  ["H · 1.80 · 864 × 480 · DETAIL", "864x480"],
  ["H · ≈16:9 · 960 × 544 · Balanced", "960x544"],
  ["H · 16:9 · 1024 × 576", "1024x576"],
  ["H · 720-source legal · 1280 × 736", "1280x736"],
  ["H · ≈16:9 · 1344 × 768 · H3 quality", "1344x768"],
  ["H · 16:9 · 1536 × 864 · H3 native high", "1536x864"],
  ["H · ≈16:9 · 1664 × 928 · H3 native high", "1664x928"],
  ["H · 1.80 · 1728 × 960 · H3 native high", "1728x960"],
  ["H · ≈16:9 · 1920 × 1088 · H3 legal FHD-class", "1920x1088"],
  ["H · 16:9 · 2048 × 1152 · H3 legal 2K-class", "2048x1152"],
  ["H · 4:3 · 1024 × 768", "1024x768"],
  ["H · 3:2 · 1152 × 768", "1152x768"],
  ["H · DCI ≈1.90 · 1216 × 640", "1216x640"],
  ["SCOPE · 2.00 · 1024 × 512", "1024x512"],
  ["SCOPE · ≈2.20 · 1120 × 512", "1120x512"],
  ["SCOPE · 2.40 · 1152 × 480", "1152x480"],
  ["SCOPE · 2.40 · 1536 × 640 · Quality", "1536x640"],
  ["SCOPE · 2.40 · 1920 × 800 · H3 native high", "1920x800"],
  ["SCOPE · ≈2.37 · 2048 × 864 · H3 legal 2K-class", "2048x864"],
  ["V · ≈9:16 · 448 × 768 · Draft", "448x768"],
  ["V · 3:5 · 384 × 640 · LIGHT", "384x640"],
  ["V · 3:5 · 576 × 960", "576x960"],
  ["V · 3:5 · 672 × 1120", "672x1120"],
  ["V · 3:5 · 768 × 1280", "768x1280"],
  ["V · 3:5 · 960 × 1600", "960x1600"],
  ["V · 3:5 · 1152 × 1920", "1152x1920"],
  ["V · ≈9:16 · 544 × 960 · Balanced", "544x960"],
  ["V · 9:16 · 576 × 1024", "576x1024"],
  ["V · ≈9:16 · 768 × 1344 · H3 quality", "768x1344"],
  ["V · 9:16 · 864 × 1536 · H3 native high", "864x1536"],
  ["V · ≈9:16 · 928 × 1664 · H3 native high", "928x1664"],
  ["V · ≈9:16 · 1088 × 1920 · H3 legal FHD-class", "1088x1920"],
  ["V · 9:16 · 1152 × 2048 · H3 legal 2K-class", "1152x2048"],
  ["V · 4:5 · 768 × 960", "768x960"],
  ["V · 2:3 · 640 × 960", "640x960"],
  ["H/V · 1:1 · 768 × 768", "768x768"],
]);
// FAST LATENT 2-PASS uses one full Stage-2 canvas, not spatial tiles. Both
// dimensions must stay on the H3/upscaler 32-pixel grid; 1088 is the legal
// FHD-class height before an optional final 1080 crop elsewhere.
const H3_FAST_2PASS_PAIRS = Object.freeze([
  [640,384,1280,768], [768,448,1536,896], [896,512,1792,1024],
  [1024,576,2048,1152], [1280,736,2560,1472],
  [736,416,1504,832], [864,480,1728,960], [960,544,1920,1088],
  [1280,736,1920,1088], [1024,576,1536,864],
  [640,384,960,576], [960,576,1280,768], [960,576,1600,960],
  [960,576,1920,1152], [1120,672,1920,1152], [1280,768,1920,1152],
  [768,384,1536,768], [1024,512,2048,1024],
  [768,320,1536,640], [1152,480,1536,640], [1152,480,1920,800],
  [1536,640,1920,800], [1024,768,1536,1152], [1152,768,1728,1152],
].flatMap(([lw,lh,hw,hh]) => [[lw,lh,hw,hh],[lh,lw,hh,hw]])
 .concat([[768,768,1536,1536]])
 .map(([lw,lh,hw,hh]) => {
   const ratio = lw / lh;
   const aspect = Math.abs(ratio - 5/3) < .001 ? "5:3" : Math.abs(ratio - 3/5) < .001 ? "3:5" : ratio.toFixed(2);
   const orientation = ratio >= 2 ? "SCOPE H" : ratio <= .5 ? "SCOPE V" : ratio > 1 ? "H" : ratio < 1 ? "V" : "SQUARE";
   return [`${orientation} · ${aspect} · ${lw} × ${lh} → ${hw} × ${hh}`,lw,lh,hw,hh];
 }));

const H3_DELIVERY_MODES = Object.freeze([
  ["off","NATIVE · no upscale","native-delivery"],
  ["h3_fast_latent_2pass","H3 FAST · latent 2-pass","upres"],
  ["h3_ultimate_tiled","H3 DETAIL · tiled latent refine","ultimate-tiled"],
  ["h3_pixel_refine","H3 PIXEL REFINE","h3-pixel"],
  ["pixel_tiled_low_vram","PIXEL SAFE · tiled / Lanczos","pixel-safe"],
  ["ltx23","LTX 2.3 · master","ltx-master"],
  ["ltx23_per_chunk","LTX 2.3 · per shot","ltx-per-shot"],
  ["rtx_final","RTX · final upscale","rtx-final"],
  ["h3_latent_upres","H3 LATENT UPRES · legacy branch",null],
  ["wan22_5b","WAN 2.2 · 5B",null],
]);
function deliveryResolutionGrid(mode) {
  return ["pixel_tiled_low_vram","rtx_final"].includes(mode) ? 8 : 32;
}
function normalizeDeliveryResolution(node, mode = String(widget(node,"upscale_mode")?.value || "off")) {
  if (mode === "off") return;
  const grid = deliveryResolutionGrid(mode);
  for (const [axis,max] of [["width",7680],["height",4320]]) {
    const raw = Number(widget(node,`upscale_${axis}`)?.value || Number(widget(node,axis)?.value || 512)*2);
    const next = Math.max(256,Math.min(Math.floor(max/grid)*grid,Math.ceil(raw/grid)*grid));
    if (Number(widget(node,`upscale_${axis}`)?.value) !== next) setValue(node,`upscale_${axis}`,next,false);
  }
  if (mode === "h3_fast_latent_2pass") {
    for (const axis of ["width","height"]) setValue(node,`h3_upres_tile_${axis}`,widget(node,`upscale_${axis}`)?.value,false);
  }
}
function selectDeliveryMode(node, mode) {
  if (!deliveryModeAvailable(node,mode)) return false;
  const option = H3_DELIVERY_MODES.find(([key]) => key === mode);
  if (!option) return false;
  setValue(node,"h3_open_refine_enabled",false,false);
  if (["h3_fast_latent_2pass","h3_ultimate_tiled"].includes(mode)) {
    const selected = String(widget(node,"h3_upres_model_name")?.value || "");
    const installed = selected && !/^none$/i.test(selected) ? selected : firstChoice(node,"h3_upres_model_name",name => /minimax_h3.*upscaler/i.test(name));
    if (!installed) { alert("Select an installed H3 3D latent upscaler in Output first."); return false; }
    setValue(node,"h3_upres_model_name",installed,false);
  }
  node.properties ||= {};
  if (mode !== "off") node.properties.iamccs_delivery_route = mode;
  if (option[2]) applyRecipe(node,option[2]);
  else { setValue(node,"upscale_mode",mode,false); setValue(node,"upscale_enabled",true,false); }
  normalizeDeliveryResolution(node,mode);
  document.dispatchEvent(new CustomEvent("iamccs:h3-settings-changed",{detail:{source_node_id:node.id,recipe:"delivery_route"}}));
  return true;
}

const widget = (node, name) => (node?.widgets || []).find((item) => item?.name === name);

function openH3ResolutionEstimate(node) {
  const width = Number(widget(node, "width")?.value), height = Number(widget(node, "height")?.value);
  const scale = Number(widget(node, "h3_open_refine_scale")?.value);
  if (![width, height, scale].every(value => Number.isFinite(value) && value > 0)) return null;
  // Match the latent upscaler's Python round-to-even on its 32-pixel grid.
  const align = value => {
    const cells = value / 32, lower = Math.floor(cells);
    return Math.max(16, (cells - lower === 0.5 ? lower + lower % 2 : Math.round(cells)) * 32);
  };
  const targetWidth = align(width * scale), targetHeight = align(height * scale);
  return {width, height, scale, targetWidth, targetHeight,
    widthGain: targetWidth / width, heightGain: targetHeight / height,
    pixelGain: targetWidth * targetHeight / (width * height),
    enabled: Boolean(widget(node, "h3_open_refine_enabled")?.value)};
}

function openH3ResolutionSummary(node) {
  const box = document.createElement("div");
  box.className = "h3p-context";
  box.dataset.openH3Resolution = "true";
  box.style.cssText = "margin:0 0 10px;line-height:1.6";
  box.setAttribute("role", "status");
  box.setAttribute("aria-label", "Open-H3 resolution and pixel gain");
  const estimate = openH3ResolutionEstimate(node);
  if (!estimate) { box.textContent = "Set the Native width, height and Open-H3 scale to preview the output size."; return box; }
  const {width, height, targetWidth, targetHeight, widthGain, heightGain, pixelGain, enabled} = estimate;
  const format = value => value.toLocaleString("en-US", {maximumFractionDigits: 2});
  const size = document.createElement("b");
  size.textContent = `${width} × ${height} → ${targetWidth} × ${targetHeight}`;
  const gain = document.createElement("div");
  gain.textContent = `${format(widthGain)}× width · ${format(heightGain)}× height · ${format(pixelGain)}× pixels`;
  const note = document.createElement("div");
  note.style.cssText = "color:#a9b7c5;font-size:9px";
  note.textContent = `${enabled ? "Estimated Open-H3 output" : "Open-H3 OFF: configured target only; refinement disabled"}. Based on the Native canvas, rounded to 32 px. Pixel gain does not measure quality gain.`;
  box.append(size, gain, note);
  return box;
}
const nodeClass = (node) => String(node?.comfyClass || node?.type || "");
function hideWidget(item) {
  if (!item || item._iamccsProHidden) return;
  item.serializeValue ||= (() => item.value);
  item.type = "hidden"; item.hidden = true; item.computeSize = () => [0, 0]; item.draw = () => {};
  item._iamccsProHidden = true;
}
function setValue(node, name, value, notify = true) {
  const item = widget(node, name); if (!item) return false;
  item.value = value; try { item.callback?.(value); } catch {}
  if (notify) document.dispatchEvent(new CustomEvent("iamccs:h3-settings-changed", { detail: { source_node_id: node.id, field: name } }));
  node.setDirtyCanvas?.(true, true); app.graph?.change?.(); return true;
}
function setSubjectReferenceCount(node, count) {
  for (let index = 1; index <= 4; index++) {
    setValue(node, `reference_role_${index}`, index <= count ? "subject_identity" : "disabled", false);
  }
  document.dispatchEvent(new CustomEvent("iamccs:h3-settings-changed", { detail: { source_node_id: node.id, field: "reference_roles" } }));
}
function choices(item) {
  let values = item?.options?.values;
  try { if (typeof values === "function") values = values(); } catch { values = []; }
  if (!Array.isArray(values)) values = item?.options?.options;
  if (!Array.isArray(values) && Array.isArray(item?.combo_values)) values = item.combo_values;
  return Array.isArray(values) ? values : [];
}
function human(name) {
  const labels = {
    ltx_looper_temporal_tile_size: "LTX window frames · automatic subdivision",
    ltx_looper_temporal_overlap: "LTX overlap frames",
    ltx_looper_horizontal_tiles: "LTX horizontal tiles · minimum",
    ltx_looper_vertical_tiles: "LTX vertical tiles · minimum",
    ltx_looper_spatial_overlap: "LTX spatial overlap",
    extended_av_profile: "EXTENSION PROFILE · VRAM / window",
    extended_av_pin_mode: "EXTENSION · ROLE (legacy/locked)",
    extended_av_mask_profile: "VIDEO HOLD · mask profile",
    extended_av_custom_window_frames: "CUSTOM · technical window frames",
    extended_av_custom_overlap_frames: "CUSTOM · preserved overlap frames",
    extended_av_handover_mode: "EXTENSION · ASSEMBLY (legacy/locked)",
    h3_auto_extend_mode: "LEGACY AUTO EXTEND · compatibility engine",
    h3_auto_extend_continue_audio: "LEGACY AUTO EXTEND · continue audio",
    h3_continuation_enabled: "START FROM SAVED AV · enable",
    h3_continuation_save_enabled: "SAVE TERMINAL AV · enable",
    h3_continuation_checkpoint: "SAVED AV · checkpoint",
    h3_continuation_context_frames: "SAVED AV · context frames",
    h3_continuation_handover_mode: "SAVED AV · start boundary",
  };
  if (labels[name]) return labels[name];
  if (name === "h3_controlnet_preprocess_resolution") return "PREPROCESS RESOLUTION · source video";
  if (name === "h3_controlnet_use_video_audio") return "USE VIDEO AUDIO · override Shotboard AudioBoard";
  if (name === "h3_controlnet_dwpose_torch_gpu") return "DWPOSE · USE TORCH GPU";
  return String(name).replace(/^h3_/, "").replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
}
function friendly(value) { return FRIENDLY_VALUES[String(value)] || String(value || "AUTO / NONE").replaceAll("_", " "); }
function restoreNamedValues(node, info, nodeData) {
  const specs = {...nodeData?.input?.required, ...nodeData?.input?.optional};
  const names = Object.keys(specs);
  let named = info?.widgets_values_named;
  if (!named || typeof named !== "object" || Array.isArray(named) || !Object.keys(named).length) {
    const values = info?.widgets_values;
    const savedNames = info?.properties?.iamccs_h3_settings_schema;
    const schema = Array.isArray(savedNames) && savedNames.length === values?.length ? savedNames
      : Array.isArray(values) && values.length === names.length ? names : null;
    if (!schema) return;
    named = Object.fromEntries(schema.map((name,index) => [name,values[index]]));
  }
  for (const [name, spec] of Object.entries(specs)) {
    const item = widget(node, name);
    if (!item) continue;
    const kind = spec?.[0], options = spec?.[1] || {};
    let value = Object.prototype.hasOwnProperty.call(named, name) ? named[name] : options.default;
    if (Array.isArray(kind) && !kind.includes(value)) value = options.default ?? kind[0];
    if ((kind === "INT" || kind === "FLOAT") && (value === "" || value == null || !Number.isFinite(Number(value)) ||
      (options.min !== undefined && Number(value) < Number(options.min)) ||
      (options.max !== undefined && Number(value) > Number(options.max)))) value = options.default ?? 0;
    if (kind === "BOOLEAN" && typeof value !== "boolean") value = options.default ?? false;
    if (kind === "STRING" && typeof value !== "string") value = options.default ?? "";
    item.value = value;
  }
  const values = names.map((name) => widget(node, name)?.value);
  node.widgets_values = values.slice();
  info.widgets_values = values.slice();
  info.widgets_values_named = Object.fromEntries(names.map((name) => [name,widget(node,name)?.value]));
}
function installAssetOptions(node, nodeData) {
  node._iamccsH3SettingsNodeData = nodeData;
  const specs = {...nodeData?.input?.required, ...nodeData?.input?.optional};
  for (const [name, spec] of Object.entries(specs)) {
    const item = widget(node, name), declared = spec?.[1] || {};
    if (!item || !Array.isArray(spec?.[0])) continue;
    item.options ||= {};
    // ComfyUI checks the backing widget's choices, even when PRO renders its
    // own controls. Restore the server list as well as the selected value.
    item.options.values = spec[0].slice();
    for (const key of ["iamccs_h3_assets", "iamccs_h3_recipes", "iamccs_sla_available"]) if (key in declared) item.options[key] = declared[key];
  }
}
function serializeNamedValues(node, info, nodeData) {
  if (!info || typeof info !== "object") return;
  restoreNamedValues(node, {widgets_values_named: Object.fromEntries((node.widgets || []).map(item => [item.name, item.value]))}, nodeData);
  const names = Object.keys({...nodeData?.input?.required, ...nodeData?.input?.optional});
  info.properties ||= {};
  info.properties.iamccs_h3_settings_schema = names;
  info.widgets_values_named = Object.fromEntries(names.map((name) => [name, widget(node, name)?.value]));
  info.widgets_values = names.map((name) => widget(node, name)?.value);
}

function normalizePianosequenzaStageControls(node) {
  const stageProfile = widget(node, "longvid_pianosequenza_stage_profile");
  const stageCount = widget(node, "longvid_pianosequenza_stage_count");
  const splitMode = widget(node, "longvid_pianosequenza_split_mode");
  const profile = widget(node, "longvid_pianosequenza_auto_profile");
  const s1 = widget(node, "longvid_pianosequenza_manual_stage1_steps");
  const s2 = widget(node, "longvid_pianosequenza_manual_stage2_steps");
  const s3 = widget(node, "longvid_pianosequenza_manual_stage3_steps");
  if (stageProfile && !["2_stage_full", "2_stage_safe_delivery", "3_stage_safe_mid", "3_stage_full"].includes(String(stageProfile.value ?? ""))) stageProfile.value = "2_stage_full";
  if (stageCount && !["2_stage", "3_stage"].includes(String(stageCount.value ?? ""))) stageCount.value = "2_stage";
  if (splitMode && !["auto", "ratio_75_25", "ratio_80_20", "ratio_67_33", "manual"].includes(String(splitMode.value ?? ""))) splitMode.value = "auto";
  if (profile && !["balanced", "low_authority", "high_refine", "fast_safe", "pdd_safe", "toetao_safe"].includes(String(profile.value ?? ""))) profile.value = "balanced";

  // Hidden compatibility INT widgets still participate in ComfyUI prompt
  // validation.  Number("") === 0 in JavaScript, so the old finite-number
  // check accidentally allowed an empty string to survive until Python tried
  // int("") and rejected the whole prompt.  Canonicalize them to real INTs
  // using the same defaults declared by INPUT_TYPES.
  for (const [item, fallback] of [[s1, 4], [s2, 2], [s3, 1]]) {
    if (!item) continue;
    const raw = item.value;
    const blank = raw == null || (typeof raw === "string" && raw.trim() === "");
    const parsed = blank ? Number.NaN : Number(raw);
    item.value = Number.isFinite(parsed)
      ? Math.max(0, Math.min(64, Math.trunc(parsed)))
      : fallback;
  }
}

function installPianosequenzaStageQueueGuards(node) {
  const names = [
    "longvid_pianosequenza_manual_stage1_steps",
    "longvid_pianosequenza_manual_stage2_steps",
    "longvid_pianosequenza_manual_stage3_steps",
  ];
  for (const name of names) {
    const item = widget(node, name);
    if (!item || item._iamccsPianosequenzaIntGuard) continue;
    item._iamccsPianosequenzaIntGuard = true;
    item.serializeValue = () => {
      normalizePianosequenzaStageControls(node);
      return item.value;
    };
  }
}

const IAMCCS_2STAGE_RESOLUTION_PAIRS = Object.freeze(H3_FAST_2PASS_PAIRS.map(([,lw,lh,hw,hh]) => `${lw}x${lh} -> ${hw}x${hh}`));
function normalize2StageResolutionLink(node, activeMode = shotboardMode(node)) {
  const toggle = widget(node, "upscale_link_to_native");
  const pair = widget(node, "upscale_link_factor");
  if (toggle && typeof toggle.value !== "boolean") toggle.value = false;
  if (pair) {
    const value = String(pair.value ?? "");
    if (!IAMCCS_2STAGE_RESOLUTION_PAIRS.includes(value)) pair.value = IAMCCS_2STAGE_RESOLUTION_PAIRS[0];
  }
  // Settings PRO retains LongVid choices while another mode is active, but
  // those dormant choices must never rewrite the active mode's native canvas.
  if (String(activeMode || "").toLowerCase() !== "longvid_guides") return;
  if (toggle?.value && pair) {
    const match = String(pair.value).match(/^(\d+)x(\d+) -> (\d+)x(\d+)$/);
    if (match) {
      const highW = Number(match[3]), highH = Number(match[4]);
      const w = widget(node, "width"), h = widget(node, "height");
      if (w) w.value = highW;
      if (h) h.value = highH;
    }
  }
}

function downstreamNodes(node) {
  const graph = node.graph || app.graph; if (!graph) return [];
  const found = [], queue = [node], seen = new Set([String(node.id)]);
  while (queue.length) {
    const current = queue.shift();
    for (const output of current?.outputs || []) for (const linkId of output?.links || []) {
      const link = graph.links?.[linkId];
      const target = graph.getNodeById?.(link?.target_id) || (graph._nodes || []).find((entry) => String(entry.id) === String(link?.target_id));
      if (!target || seen.has(String(target.id))) continue;
      seen.add(String(target.id)); found.push(target); if (BRIDGE_TYPES.has(nodeClass(target))) queue.push(target);
    }
  }
  return found;
}
function connectedNodes(node) {
  const graph = node.graph || app.graph; if (!graph) return [];
  const found = [], queue = [node], seen = new Set([String(node.id)]);
  while (queue.length) {
    const current = queue.shift();
    const links = [];
    for (const input of current?.inputs || []) if (input?.link != null) links.push(input.link);
    for (const output of current?.outputs || []) for (const linkId of output?.links || []) links.push(linkId);
    for (const linkId of links) {
      const link = graph.links?.[linkId]; if (!link) continue;
      for (const id of [link.origin_id, link.target_id]) {
        const candidate = graph.getNodeById?.(id) || (graph._nodes || []).find((entry) => String(entry.id) === String(id));
        if (!candidate || seen.has(String(candidate.id))) continue;
        seen.add(String(candidate.id)); found.push(candidate); queue.push(candidate);
        // Branch capabilities remain discoverable inside connected native subgraphs.
        const collect=graph=>{for(const child of graph?._nodes||[]){found.push(child);if(child.subgraph)collect(child.subgraph);}};
        if(candidate.subgraph)collect(candidate.subgraph);
      }
    }
  }
  return found;
}
function branchGate(node, name) {
  const classes = node._iamccsSettingsProConnectedClasses || connectedNodes(node).map(nodeClass);
  const has = (pattern) => classes.some((value) => pattern.test(value));
  if (name.startsWith("h3_open_refine_")) return {
    available: has(/^IAMCCS_MiniMaxH3OpenRefineControl$/),
    reason: "Load the R49 Open-H3 workflow with its connected refinement settings branch.",
  };
  if (name === "keyframe_joint_latent_new") return {
    available: Boolean(widget(node, "extended_av_profile")),
    reason: "LATENT NEW uses the B1 EXTEND backend.",
  };
  if (name.startsWith("h3_controlnet_")) return {
    available: has(/CineH3FunControlInput|MiniMaxH3FunControlInput/i),
    reason: "Connect the H3 Fun Control input branch to enable this control.",
  };
  if (name.startsWith("h3_faceswap_")) return {
    available: has(/H3FaceSwapInput/i),
    reason: "Connect the H3 SAM3 Subject Swap input branch to enable this control.",
  };
  if (name.startsWith("face_detailer_")) return {
    available: has(/MiniMaxH3Face(Delivery|Detailer|Track|Stitch|Mask)/i),
    reason: "Connect the optional H3 Face Detailer branch to enable this control.",
  };
  if (name.startsWith("h3_r40_")) return {
    available: has(/SeedScout|ShotLabControl|TakeSelectControl|PixelRefineR40/i),
    reason: "Connect the optional candidate-scout branch to enable this control.",
  };
  if (name.startsWith("ltx_") || name === "wan_upscale_denoise") return {
    available: name.startsWith("ltx_") ? has(/LTX/i) : has(/Wan/i),
    reason: `Connect the optional ${name.startsWith("ltx_") ? "LTX" : "Wan"} delivery branch to enable this control.`,
  };
  if (name.startsWith("h3_upres_") || name.startsWith("h3_fast_latent_")) return {
    available: has(/LatentUpres|FastLatent2Pass|UltimateTiled|ProgressiveSpatial|UniversalFast/i),
    reason: "Connect an H3 latent-upres delivery branch to enable this control.",
  };
  return { available: true, reason: "" };
}
function modeAvailability(node, mode) {
  const classes = node._iamccsSettingsProConnectedClasses || connectedNodes(node).map(nodeClass);
  const has = (pattern) => classes.some((value) => pattern.test(value));
  const value = String(mode || "").toLowerCase();
  if (value === "reference_image") return {
    available: has(/IAMCCS_R42ReferenceImagePromptBridge/) && has(/SatoDiveH3Image/),
    reason: "Use the R42 Reference Image workflow with both the IAMCCS prompt/preset bridge and SatoDive exact-still branch.",
  };
  if (["fl2va_extended_av", "t2va_continuous"].includes(value)) {
    const taskChoices = choices(widget(node, "task_mode")).map(String);
    const hasContract = Boolean(widget(node, "extended_av_profile"));
    return {
      available: hasContract && (!taskChoices.length || taskChoices.includes(value)),
      reason: hasContract ? "" : "Update/restart IAMCCS-nodes: Extended AV backend controls were not found.",
    };
  }
  if (value === "latent_go_ahead") return {
    available: has(/^IAMCCS_MiniMaxH3LatentGoAhead/i),
    reason: "Connect the IAMCCS LatentGoAhead generation branch to enable this mode.",
  };
  if (["longvid_motion_context", "longvid_continuous_guided"].includes(value)) return {
    available: has(/IAMCCS_MiniMaxH3MotionContextGenerationR37/i),
    reason: "Connect the IAMCCS Motion Context generation branch to enable this LongVid mode.",
  };
  if (["longvid_masked_loop_guided", "guided_av_loop_experimental"].includes(value)) return {
    available: has(/IAMCCS_MiniMaxH3MaskedLoopGuidedSampler/i) && has(/IAMCCS_MiniMaxH3MaskedLoopConditioning/i),
    reason: "Connect both IAMCCS Guided AV Loop conditioning and sampler branches to enable this mode.",
  };
  if (["ref2vid_lipsync", "longvid_guided_lipsync"].includes(value)) return {
    available: has(/IAMCCS_MiniMaxH3AtomicAudioDrive/i),
    reason: "Connect the IAMCCS Atomic Audio Drive branch to enable this AudioBoard-driven mode.",
  };
  if (value === "v2va_controlnet") return {
    available: has(/CineH3FunControlInput|MiniMaxH3FunControlInput/i),
    reason: "Connect the H3 Fun Control input branch to enable Control Video.",
  };
  if (value === "ref2va_vfx_edit") return {available:has(/IAMCCS_H3VFXInput/),reason:"Connect the dedicated H3 VFX Input branch."};
  if (value === "v2va_face_swap_lora") return {
    available: has(/H3FaceSwapInput/),
    reason: "Connect H3 Swap Input with source_video and reference_face to enable Character Swap -LorA.",
  };
  if (value === "v2va_face_swap") return {
    available: has(/H3FaceSwapInput|MiniMaxH3FaceDelivery|MiniMaxH3FaceDetailer/i),
    reason: "Connect the H3 SAM3 branch to enable Subject Swap.",
  };
  // Core modes and experimental modes implemented inside the universal backend
  // remain selectable. Do not infer missing support from a filename alone.
  return { available: true, reason: "" };
}
function latentGoAheadBranch(node) {
  return connectedNodes(node).find((candidate) => /^IAMCCS_MiniMaxH3LatentGoAhead/i.test(nodeClass(candidate))) || null;
}
function selectedGuideWindowStrategy(node) {
  const policy = String(widget(node, "longvid_guide_window_policy")?.value || "legacy_auto");
  const continuity = String(widget(node, "flf_continuity_mode")?.value || "stable_keyframes");
  const tail = Number(widget(node, "flf_continuity_tail_frames")?.value || 0);
  const windowFrames = Number(widget(node, "motion_context_window_frames")?.value || 362);
  const adaptive = policy === "adaptive_latent_tail" || (policy === "legacy_auto" && continuity === "longvid_latent_tail_experimental");
  if (!adaptive) return "guide_macro_362";
  if (tail === 22 && windowFrames <= 124) return "guide_safe_124";
  if (tail === 22 && windowFrames <= 209) return "guide_safe_209";
  return "guide_auto";
}
function applyGuideWindowStrategy(node, strategy) {
  const set = (name, value) => setValue(node, name, value, false);
  if (strategy === "guide_auto") {
    // Use the known-good 640x384 x 362f native workload as a conservative
    // pixel-frame reference, then select the largest supported technical
    // window that does not exceed it. This does not claim a universal VRAM
    // law; it is an explicit IAMCCS safety heuristic and remains overrideable.
    const width = Math.max(1, Number(widget(node, "width")?.value || 640));
    const height = Math.max(1, Number(widget(node, "height")?.value || 384));
    const budget = 640 * 384 * 362;
    const candidates = [362, 294, 209, 124];
    const chosen = candidates.find((frames) => width * height * frames <= budget) || 124;
    if (chosen >= 362) {
      set("longvid_guide_window_policy", "standard_positioned");
      set("flf_continuity_mode", "stable_keyframes");
      set("motion_context_window_frames", 362);
    } else {
      set("longvid_guide_window_policy", "adaptive_latent_tail");
      set("flf_continuity_mode", "longvid_latent_tail_experimental");
      set("flf_continuity_tail_frames", "22");
      set("flf_continuity_audio", true);
      set("motion_context_window_frames", chosen);
      set("flf_join_mode", "h3_keyframe_cut"); set("flf_overlap_frames", 0);
    }
  } else if (strategy === "guide_safe_124") {
    set("longvid_guide_window_policy", "adaptive_latent_tail");
    set("flf_continuity_mode", "longvid_latent_tail_experimental");
    set("flf_continuity_tail_frames", "22");
    set("flf_continuity_audio", true);
    set("motion_context_window_frames", 124);
    set("flf_join_mode", "h3_keyframe_cut"); set("flf_overlap_frames", 0);
  } else if (strategy === "guide_safe_209") {
    set("longvid_guide_window_policy", "adaptive_latent_tail");
    set("flf_continuity_mode", "longvid_latent_tail_experimental");
    set("flf_continuity_tail_frames", "22");
    set("flf_continuity_audio", true);
    set("motion_context_window_frames", 209);
    set("flf_join_mode", "h3_keyframe_cut"); set("flf_overlap_frames", 0);
  } else {
    set("longvid_guide_window_policy", "standard_positioned");
    set("flf_continuity_mode", "stable_keyframes");
    set("motion_context_window_frames", 362);
  }
  document.dispatchEvent(new CustomEvent("iamccs:h3-settings-changed", { detail: { source_node_id: node.id, field: "longvid_guide_window_policy", strategy } }));
  node._iamccsSettingsProRefresh?.();
}
function appendLatentGoAheadProxyControls(node, grid) {
  const branch = latentGoAheadBranch(node);
  const section = document.createElement("section");
  section.className = "h3p-functional";
  section.innerHTML = '<div class="h3p-functional-title">LATENT NEW / GO AHEAD BRANCH</div>';
  const inner = document.createElement("div"); inner.className = "h3p-functional-grid";
  if (!branch) {
    const note = document.createElement("div"); note.className = "h3p-field muted";
    note.innerHTML = '<label>BRANCH NOT CONNECTED</label><small>LATENT NEW remains disabled. Add/connect IAMCCS_MiniMaxH3LatentGoAhead to expose native history, audio history and delivery overlap controls here.</small>';
    inner.append(note); section.append(inner); grid.append(section); return;
  }
  const wanted = [
    "video_context_frames", "audio_context_seconds", "freeze_tail_max_frames", "freeze_tail_threshold",
    "join_blend", "blend_frames", "audio_join", "audio_smoothing_ms", "flash_guard",
    "flash_guard_sensitivity", "flash_guard_radius",
  ];
  for (const name of wanted) {
    const item = (branch.widgets || []).find((entry) => entry?.name === name); if (!item) continue;
    const box = document.createElement("div"); box.className = "h3p-field";
    const label = document.createElement("label"); label.textContent = human(name); box.append(label);
    const opts = choices(item); let control;
    if (opts.length) { control = document.createElement("select"); opts.forEach((value) => control.add(new Option(friendly(value), String(value)))); control.value = String(item.value ?? ""); }
    else { control = document.createElement("input"); control.type = typeof item.value === "number" ? "number" : "text"; control.value = item.value ?? ""; if (control.type === "number") { if (Number.isFinite(item.options?.min)) control.min = item.options.min; if (Number.isFinite(item.options?.max)) control.max = item.options.max; control.step = item.options?.step ?? "any"; } }
    control.onchange = () => { item.value = control.type === "number" ? Number(control.value) : control.value; try { item.callback?.(item.value); } catch {} app.graph?.change?.(); branch.setDirtyCanvas?.(true, true); };
    box.append(control); const hint=document.createElement("small"); hint.textContent=String(item.options?.tooltip || "Direct control of the connected LatentGoAhead branch."); box.append(hint); inner.append(box);
  }
  section.append(inner); grid.append(section);
}
function linkedShotboard(node) { return downstreamNodes(node).find((candidate) => nodeClass(candidate) === SHOTBOARD_TYPE) || null; }
function shotboardMode(node) {
  if (node.properties?.iamccs_settings_master) {
    const owned = String(widget(node, "task_mode")?.value || "");
    if (owned && owned !== "auto_from_timeline" && owned !== "auto_from_shotboard") return owned;
  }
  const board = linkedShotboard(node);
  if (!board) return String(widget(node, "task_mode")?.value || "auto_from_timeline");
  const direct = String(widget(board, "task_mode")?.value || "auto_from_timeline"); if (direct !== "auto_from_timeline") return direct;
  try { const data = JSON.parse(String(widget(board, "timeline_data")?.value || "{}")); return String(data.task_mode || data.mode || direct); } catch { return direct; }
}
function setShotboardMode(node, mode) {
  const board = linkedShotboard(node);
  setValue(node, "task_mode", mode, false);
  if (!board) {
    document.dispatchEvent(new CustomEvent("iamccs:h3-settings-changed", { detail: { source_node_id: node.id, task_mode: mode } }));
    node._iamccsSettingsProRefresh?.();
    return true;
  }
  setValue(board, "task_mode", mode, false);
  const timelineWidget = widget(board, "timeline_data");
  if (timelineWidget) {
    let data = {}; try { data = JSON.parse(String(timelineWidget.value || "{}")); } catch {}
    data.task_mode = mode; data.mode = mode;
    if (data.h3_saved_settings && typeof data.h3_saved_settings === "object") data.h3_saved_settings.task_mode = mode;
    setValue(board, "timeline_data", JSON.stringify(data), false);
  }
  document.dispatchEvent(new CustomEvent("iamccs:h3-settings-changed", { detail: { source_node_id: node.id, task_mode: mode } }));
  node._iamccsSettingsProRefresh?.(); return true;
}
function setShotboardAudio(node, audioMode) {
  const board = linkedShotboard(node);
  setValue(node, "audio_mode", audioMode, false);
  if (board) {
    setValue(board, "audio_mode", audioMode, false);
    const timelineWidget = widget(board, "timeline_data");
    if (timelineWidget) {
      let data = {}; try { data = JSON.parse(String(timelineWidget.value || "{}")); } catch {}
      data.audio_mode = audioMode;
      if (data.h3_saved_settings && typeof data.h3_saved_settings === "object") data.h3_saved_settings.audio_mode = audioMode;
      setValue(board, "timeline_data", JSON.stringify(data), false);
    }
  }
  document.dispatchEvent(new CustomEvent("iamccs:h3-settings-changed", { detail: { source_node_id: node.id, audio_mode: audioMode } }));
  node._iamccsSettingsProRefresh?.(); return true;
}
function setShotboardDuration(node, durationSeconds, reason = "custom") {
  const duration = Math.max(0.01, Number(durationSeconds) || 0.01);
  const board = linkedShotboard(node);
  node.properties ||= {};
  if (shotboardMode(node) === "v2va_face_swap_lora") {
    node.properties.iamccs_swap_duration ||= {};
    node.properties.iamccs_swap_duration.mode = reason === "swap_video" ? "source" : "custom";
  }
  node.properties.iamccs_settings_master = true;
  node.properties.iamccs_auto_from_shotboard_active = false;
  setValue(node, "duration_seconds", duration, false);
  if (board) {
    setValue(board, "duration_seconds", duration, false);
    const timelineWidget = widget(board, "timeline_data");
    if (timelineWidget) {
      let data = {}; try { data = JSON.parse(String(timelineWidget.value || "{}")); } catch {}
      data.duration_seconds = duration;
      data.duration_authority = "settings_pro";
      if (data.timeline && typeof data.timeline === "object") data.timeline.duration_seconds = duration;
      if (data.h3_saved_settings && typeof data.h3_saved_settings === "object") data.h3_saved_settings.duration_seconds = duration;
      setValue(board, "timeline_data", JSON.stringify(data), false);
    }
  }
  document.dispatchEvent(new CustomEvent("iamccs:h3-settings-changed", {
    detail: { source_node_id: node.id, field: "duration_seconds", duration_seconds: duration, authority: "settings_pro" },
  }));
  node._iamccsSettingsProRefresh?.();
  return true;
}
function mirrorShotboardAuthority(node) {
  const board = linkedShotboard(node); if (!board) return;
  const names = node.properties?.iamccs_settings_master
    ? ["frame_rate", "audio_mode"]
    : ["task_mode", "duration_seconds", "frame_rate", "audio_mode"];
  for (const name of names) {
    const source = widget(board, name), target = widget(node, name);
    if (source && target) target.value = source.value;
  }
}
function importSettingsFromShotboard(node) {
  const board = linkedShotboard(node);
  if (!board) {
    alert("AUTO FROM SHOTBOARD needs Settings PRO connected through Cine H3 Input to a MiniMax H3 Shotboard.");
    return false;
  }
  let timeline = {};
  try { timeline = JSON.parse(String(widget(board, "timeline_data")?.value || "{}")); } catch {}
  const saved = timeline?.h3_saved_settings && typeof timeline.h3_saved_settings === "object"
    ? timeline.h3_saved_settings : {};
  const imported = [];
  for (const target of node.widgets || []) {
    const name = String(target?.name || "");
    if (!name || AUTO_IMPORT_BLOCKED.has(name) || INTERNAL.has(name) || COMPATIBILITY_ONLY.has(name)) continue;
    const source = widget(board, name);
    const hasVisible = source && source.value !== undefined && source.value !== null;
    const hasSaved = Object.prototype.hasOwnProperty.call(saved, name);
    if (!hasVisible && !hasSaved) continue;
    setValue(node, name, hasVisible ? source.value : saved[name], false);
    imported.push(name);
  }
  restoreNamedValues(node, {widgets_values_named: Object.fromEntries((node.widgets || []).map(item => [item.name, item.value]))}, node._iamccsH3SettingsNodeData);
  const importedMode = shotboardMode(node);
  const effectiveMode = HIDDEN_MODES.has(importedMode) ? "auto_from_timeline" : importedMode;
  if (effectiveMode !== importedMode) setShotboardMode(node, effectiveMode);
  if (effectiveMode && effectiveMode !== "auto_from_timeline") {
    setValue(node, "task_mode", effectiveMode, false);
    if (!imported.includes("task_mode")) imported.push("task_mode");
  }
  node.properties ||= {};
  node.properties.iamccs_settings_master = true;
  node.properties.iamccs_auto_from_shotboard_active = true;
  node.properties.iamccs_auto_import_snapshot = {
    imported_count: imported.length,
    imported_fields: imported,
    source_node_id: board.id,
    source_truth_revision: timeline?.truth_revision ?? timeline?._iamccs_v3_truth_revision ?? null,
    effective_mode: effectiveMode,
    imported_at: new Date().toISOString(),
  };
  document.dispatchEvent(new CustomEvent("iamccs:h3-settings-changed", {
    detail: { source_node_id: node.id, policy: "auto_from_shotboard_once", effective_mode: effectiveMode },
  }));
  app.graph?.change?.();
  node.setDirtyCanvas?.(true, true);
  node._iamccsSettingsProRefresh?.();
  return true;
}
function assistantModeKey(node, mode = shotboardMode(node)) {
  if (node.properties?.iamccs_auto_from_shotboard_active) return "auto_from_shotboard";
  if (String(mode) === "ref2va" && widget(node, "h3_omni_enabled")?.value) return "ref2va_omni";
  if (String(mode) !== "fl2va") return String(mode);
  const board = linkedShotboard(node);
  return String(widget(board || node, "flf_continuity_mode")?.value || "stable_keyframes") === "native_av_context"
    ? "fl2va_continuous" : "fl2va_stable";
}
function normalizeExtendedAvProductionContract(node) {
  if (!["fl2va_extended_av", "t2va_continuous"].includes(String(shotboardMode(node)))) return false;
  let changed = false;
  const locked = {
    extended_av_pin_mode: "masked",
    extended_av_mask_profile: "exact",
    extended_av_handover_mode: "auto",
    extended_av_level_lock_mode: "off",
    extended_av_joint_refine_mode: "off",
    h3_auto_extend_mode: "off",
    flf_overlap_frames: 0,
    flf_continuity_mode: "stable_keyframes",
    keyframe_joint_latent_new: false,
  };
  Object.entries(locked).forEach(([name, value]) => {
    const w = widget(node, name);
    if (w && w.value !== value) { setValue(node, name, value, false); changed = true; }
  });
  return changed;
}

function setAssistantMode(node, key) {
  if (String(key) === "auto_from_shotboard") return importSettingsFromShotboard(node);
  const board = linkedShotboard(node);
  node.properties ||= {};
  node.properties.iamccs_settings_master = true;
  node.properties.iamccs_auto_from_shotboard_active = false;
  const swapFields = {steps:4,sampler_name:"euler",scheduler:"beta",denoise:1,shift_video:12,shift_audio:3,
    acceleration:"h3_sage",turbo_mode:"early_8_10",turbo_lora_name:"minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
    turbo_strength:1,turbo_sampler_mode:"res_multistep_stock",secondary_lora_enabled:false,
    h3_swap_lora_name:"h3_character_swap_pro4500_1000.safetensors",h3_swap_lora_strength:1,
    h3_swap_realism_lora_name:"h3-realism-people-t2v-i2v-r2v.safetensors",h3_swap_realism_strength:1,h3_swap_refine_seed:777,
    h3_veda_enabled:true,h3_veda_predictor:"auto",upscale_enabled:false,upscale_mode:"h3_ultimate_tiled",
    h3_upres_steps:1,h3_upres_denoise:0.2,h3_upres_sampler:"sa_solver",h3_upres_scheduler:"simple",
    audio_mode:"h3_custom_audio_drive",face_detailer_enabled:false,
    text_encoder_device:"gpu_auto",h3_clipproj_profile:"4b_v3.1",h3_clipproj_load_mode:"dynamic"};
  if (shotboardMode(node) === "v2va_face_swap_lora" && String(key) !== "v2va_face_swap_lora") {
    for (const [name,value] of Object.entries(node.properties.iamccs_before_swap_lora || {})) setValue(node,name,value,false);
    delete node.properties.iamccs_before_swap_lora;
  }
  if (shotboardMode(node)==="ref2va_vfx_edit" && String(key)!=="ref2va_vfx_edit") {
    for (const [name,value] of Object.entries(node.properties.iamccs_before_vfx_edit||{})) setValue(node,name,value,false);
    delete node.properties.iamccs_before_vfx_edit;
  }
  if (String(key) === "v2va_face_swap_lora" && shotboardMode(node) !== "v2va_face_swap_lora") {
    node.properties.iamccs_before_swap_lora = Object.fromEntries(Object.keys(swapFields).filter(name=>widget(node,name)).map(name=>[name,widget(node,name).value]));
    for (const [name,value] of Object.entries(swapFields)) setValue(node,name,value,false);
  }
  const vfxFields = {steps:20,sampler_name:"euler",scheduler:"simple",denoise:1,shift_video:12,shift_audio:3,
    turbo_mode:"off",turbo_lora_name:"",pdd_lora_name:"",secondary_lora_enabled:false,
    h3_vfx_lora_name:"minimax_h3_vfx_edit_v1.0_r128.safetensors",h3_vfx_lora_strength:1,
    audio_mode:"h3_native_generated",upscale_enabled:false,face_detailer_enabled:false};
  if (String(key)==="ref2va_vfx_edit" && shotboardMode(node)!=="ref2va_vfx_edit") {
    node.properties.iamccs_before_vfx_edit=Object.fromEntries(Object.keys(vfxFields).filter(name=>widget(node,name)).map(name=>[name,widget(node,name).value]));
    for (const [name,value] of Object.entries(vfxFields)) setValue(node,name,value,false);
  }
  // Only the two legacy FL2VA assistant aliases map back to the historical
  // `fl2va` task. Extended AV is a real independent backend mode.
  setValue(node, "h3_omni_enabled", key === "ref2va_omni", false);
  const actualMode = key === "ref2va_omni" ? "ref2va" : ["fl2va_stable", "fl2va_continuous"].includes(String(key)) ? "fl2va" : key;
  setShotboardMode(node, actualMode);
  if (node.properties.iamccs_r49_all_modes) {
    for (const target of connectedNodes(node)) {
      const modes = target.properties?.iamccs_r49_preview_modes;
      if (Array.isArray(modes)) target.mode = modes.includes(actualMode) ? 0 : 2;
    }
  }
  if (key === "ref2va_omni") applyRecipe(node, "omni");
  if (key === "v2va_controlnet") applyRecipe(node, "control");
  if (key === "v2va_face_swap") applyRecipe(node, "face");
  if (String(key) === "reference_image") {
    // The still branch owns a genuine 4K canvas.  Apply it when the user
    // selects the mode; later manual resolution choices remain authoritative.
    for (const [name, value] of Object.entries({width:3840,height:2160,image_width:3840,image_height:2160})) {
      setValue(node, name, value, false);
      if (board) setValue(board, name, value, false);
    }
  }
  if (key === "keyframe_joint_native") {
    // Public KEYFRAME JOINT is genuinely native. Historical workflows that
    // already serialized latent-new=true still load unchanged; selecting the
    // mode again intentionally returns to the stable native contract.
    setValue(node, "keyframe_joint_latent_new", false, false);
    if (board) setValue(board, "keyframe_joint_latent_new", false, false);
  }
  if (["fl2va_extended_av", "t2va_continuous"].includes(String(key))) {
    normalizeExtendedAvProductionContract(node);
    if (!String(widget(node, "extended_av_profile")?.value || "")) setValue(node, "extended_av_profile", "safe_8_12gb", false);
  }
  // Custom Audio Drive remains the performance authority across mode changes.
  if (["fl2va_stable", "fl2va_continuous"].includes(key)) {
    const continuity = key === "fl2va_continuous" ? "native_av_context" : "stable_keyframes";
    setValue(node, "flf_continuity_mode", continuity, false);
    if (board) {
      setValue(board, "flf_continuity_mode", continuity, false);
      const timeline = widget(board, "timeline_data");
      if (timeline) {
        let data = {}; try { data = JSON.parse(String(timeline.value || "{}")); } catch {}
        data.flf_continuity_mode = continuity;
        if (data.h3_saved_settings) data.h3_saved_settings.flf_continuity_mode = continuity;
        setValue(board, "timeline_data", JSON.stringify(data), false);
      }
    }
  }
  if (["ref2vid_lipsync", "longvid_guided_lipsync"].includes(key)) setShotboardAudio(node, "h3_custom_audio_drive");
  if (key === "longvid_multishot_hardcut") {
    setValue(node, "flf_join_mode", "h3_keyframe_cut", false);
    setValue(node, "flf_overlap_frames", 0, false);
    setValue(node, "keyframe_joint_latent_new", false, false);
  }
  if (["longvid_keyframe_continuous", "pianosequenza_true_long"].includes(key)) {
    // Best proven visual/audio seam: direct full AV latent history with the
    // phase_aligned_extended provider. Pixel crossfade remains zero. These
    // visible values configure context only; the alignment mode itself is a
    // backend invariant for this public engine.
    setValue(node, "flf_join_mode", "h3_keyframe_cut", false);
    setValue(node, "flf_overlap_frames", 0, false);
    const currentContext = String(widget(node, "flf_continuity_tail_frames")?.value || "22");
    setValue(node, "flf_continuity_tail_frames", ["5", "22", "39"].includes(currentContext) ? currentContext : "22", false);
    setValue(node, "flf_continuity_audio", true, false);
    setValue(node, "keyframe_joint_latent_new", false, false);
  }
  if (key === "longvid_guides") {
    setValue(node, "longvid_pianosequenza_2stage_enabled", true, false);
    if (board) setValue(board, "longvid_pianosequenza_2stage_enabled", true, false);
    normalize2StageResolutionLink(node);
  }
  // Continuous AV transports H3's complete generated AV latent. Switching
  // from LipSync must clear that mode's locked AudioBoard route so the active
  // mode and Queue Truth cannot disagree.
  if (["guided_av_loop_experimental", "latent_go_ahead"].includes(key)) setShotboardAudio(node, "h3_native_generated");
  document.dispatchEvent(new CustomEvent("iamccs:h3-settings-changed", { detail: { source_node_id: node.id, assistant_mode: key } }));
  node._iamccsSettingsProRefresh?.(); return true;
}
function modeContext(mode) { const value = String(mode).toLowerCase(); return value === "v2va_face_swap_lora" ? "swap_lora" : value.includes("controlnet") ? "control" : value.includes("face_swap") ? "face" : "core"; }
function modeFamily(mode) {
  const value = String(mode).toLowerCase();
  return value.includes("ref2") || ["reference_image", "v2va_object_swap", "v2va_face_swap", "v2va_face_swap_lora"].includes(value) ? "ref2" : "fl2";
}
function h3AssetDescriptor(node, field, selectedName) {
  const name = String(selectedName || "");
  const assets = widget(node, field)?.options?.iamccs_h3_assets;
  return Array.isArray(assets) ? assets.find((item) => String(item?.name || "") === name) || null : null;
}
function inferredH3AssetFamily(name, descriptor = null) {
  const declared = String(descriptor?.family || "").toLowerCase();
  if (declared.includes("ref2")) return "ref2";
  if (declared.includes("fl2")) return "fl2";
  const value = String(name || "").toLowerCase();
  const ref = /(^|[^a-z0-9])ref2v?a?([^a-z0-9]|$)/.test(value);
  const fl = /(^|[^a-z0-9])fl2v?a?([^a-z0-9]|$)/.test(value);
  return ref && !fl ? "ref2" : fl && !ref ? "fl2" : "";
}
function selectedH3AssetCompatibility(node, mode) {
  const issues = [];
  const expected = modeFamily(mode);
  const acceleration = String(widget(node, "acceleration")?.value || "native").toLowerCase();
  const turboMode = String(widget(node, "turbo_mode")?.value || "off").toLowerCase();
  const entries = [];
  const turboName = String(widget(node, "turbo_lora_name")?.value || "");
  const turboActive = turboMode !== "off" || ["fasth3_dense_6step", "h3_sla"].includes(acceleration);
  if (turboActive) entries.push(["Acceleration LoRA", "turbo_lora_name", turboName, "speed"]);
  if (["pdd_native_8step", "iamccs_progressive_pdd_2stage"].includes(acceleration)) {
    entries.push(["PDD LoRA", "pdd_lora_name", String(widget(node, "pdd_lora_name")?.value || ""), "pdd"]);
  }
  if (Boolean(widget(node, "secondary_lora_enabled")?.value)) {
    entries.push(["Secondary creative LoRA", "secondary_lora_name", String(widget(node, "secondary_lora_name")?.value || ""), "creative"]);
  }
  for (const [label, field, name, slot] of entries) {
    if (!name) { issues.push(["error", `${label} is active but no file is selected.`]); continue; }
    const descriptor = h3AssetDescriptor(node, field, name) || h3AssetDescriptor(node, "turbo_lora_name", name);
    const family = inferredH3AssetFamily(name, descriptor);
    const role = String(descriptor?.role || "unknown").toLowerCase();
    if (descriptor?.error || descriptor?.conflict) {
      issues.push(["error", `${label} · ${name}: ${descriptor.error || "filename and Safetensors base-model metadata disagree"}.`]);
      continue;
    }
    if (family && family !== expected) {
      issues.push(["error", `${label} · ${name} is ${family.toUpperCase()} but mode ${String(mode).toUpperCase()} requires ${expected.toUpperCase()}. Select the matching model-family LoRA.`]);
    } else if (!family) {
      issues.push(["warn", `${label} · ${name} does not declare FL2V/REF2V family in filename or Safetensors metadata. Compatibility cannot be proven automatically.`]);
    }
    if (slot === "pdd" && role && !["pdd", "unknown"].includes(role)) {
      issues.push(["error", `${label} · ${name} is classified as ${role.toUpperCase()}, not as a native PDD adapter.`]);
    }
    if (slot === "speed" && role === "style") {
      issues.push(["error", `${label} · ${name} is a style LoRA. Move it to Secondary creative LoRA instead of the acceleration slot.`]);
    }
    if (slot === "creative" && ["pdd", "fasth3"].includes(role)) {
      issues.push(["warn", `${label} · ${name} is an acceleration adapter (${role.toUpperCase()}); use its dedicated speed recipe instead of the creative slot.`]);
    }
  }
  if (acceleration === "matlowai_fused_turbo_manual_sigma") {
    const fused = String(widget(node, "fused_turbo_model_name")?.value || "");
    const convrot = fused.toLowerCase().includes("convrot");
    const supported = convrot ? ["t2va", "i2va", "fl2va", "ref2va"] : ["t2va"];
    if (!fused) issues.push(["error", "Fused Fast is active but no diffusion checkpoint is selected."]);
    else if (!supported.includes(String(mode).toLowerCase())) issues.push(["error", `${fused} is not compatible with ${String(mode).toUpperCase()}. This fused profile supports ${supported.join(", ").toUpperCase()} only.`]);
    if (Boolean(widget(node, "secondary_lora_enabled")?.value)) issues.push(["error", "Fused Fast is a complete checkpoint profile and cannot be combined with Secondary creative LoRA."]);
  }
  return issues;
}
function firstChoice(node, name, predicate) { return choices(widget(node, name)).map(String).find((value) => value && predicate(value.toLowerCase())) || ""; }
function isFastH3LoRA(name) { const value = String(name).toLowerCase(); return (value.includes("fasth3") || value.includes("fast_h3") || value.includes("fast-h3")) && !value.includes("fused") && !value.includes("adapter_model"); }
function fastH3Asset(node, mode = shotboardMode(node)) {
  return choices(widget(node, "turbo_lora_name")).map(String).find(name => {
    const asset = h3AssetDescriptor(node, "turbo_lora_name", name);
    return isFastH3LoRA(name) && asset?.role === "fasth3" && !asset.error && !asset.conflict
      && inferredH3AssetFamily(name, asset) === modeFamily(mode);
  }) || "";
}
function turboModeGroup(mode) {
  if (modeFamily(mode) === "ref2") return "ref2";
  return String(mode).startsWith("t2") || mode === "v2va_controlnet" ? "t2" : String(mode).startsWith("i2") ? "i2" : "fl2";
}
function isPddLoRA(name, family) {
  const value = String(name).toLowerCase();
  return value.includes(family) && (value.includes("pdd") || value.includes("acc-8step") || value.includes("acc_8step"));
}
function declaredTurboSteps(node, name) {
  const selected = String(name || ""); if (!selected) return 0;
  const assets = widget(node, "turbo_lora_name")?.options?.iamccs_h3_assets;
  const descriptor = Array.isArray(assets) ? assets.find((item) => String(item?.name || "") === selected) : null;
  const metadataStep = Number(descriptor?.declared_steps || 0);
  if (Number.isInteger(metadataStep) && metadataStep > 0) return metadataStep;
  const match = selected.match(/(?:^|[^0-9])(\d{1,2})[\s_.-]*steps?(?:[^0-9]|$)/i);
  return match ? Number(match[1]) : 0;
}
function assistantTurboAssets(node, family) {
  const options = widget(node, "turbo_lora_name")?.options || {};
  const installed = new Set(choices(widget(node, "turbo_lora_name")).map(String));
  return (options.iamccs_h3_assets || []).filter(asset => installed.has(asset.name)
    && asset.role === "turbo" && !asset.error && !asset.conflict
    && inferredH3AssetFamily(asset.name, asset) === family
    && options.iamccs_h3_recipes?.[asset.recipe]?.values
    && options.iamccs_h3_recipes[asset.recipe].values.acceleration !== "h3_sla");
}
function applyAssistantTurbo(node, name) {
  const asset = assistantTurboAssets(node, modeFamily(shotboardMode(node))).find(item => item.name === name);
  if (!asset) return false;
  setValue(node, "h3_omni_enabled", false, false);
  const values = widget(node, "turbo_lora_name").options.iamccs_h3_recipes[asset.recipe].values;
  const set = (key, value) => setValue(node, key, value, false);
  rememberAttentionChoice(node);
  if (["pdd_native_8step", "iamccs_progressive_pdd_2stage", "fasth3_dense_6step", "h3_sla", "matlowai_fused_turbo_manual_sigma"].includes(String(widget(node, "acceleration")?.value))) set("acceleration", "native");
  resetAcceleration(set);
  Object.entries(values).forEach(([key, value]) => set(key, value));
  set("turbo_lora_name", name);
  // Preserve the successful R49 four-step swap recipe.
  if (shotboardMode(node) === "v2va_face_swap_lora" && asset.recipe === "lightx_ref2_544_4") set("scheduler", "beta");
  document.dispatchEvent(new CustomEvent("iamccs:h3-settings-changed", {detail:{source_node_id:node.id, recipe:"turbo", asset:name}}));
  node._iamccsSettingsProRefresh?.();
  return true;
}
const ATTENTION_PRESETS = [
  ["AUTO LOW VRAM", "auto_3060"], ["SAGE + LOW VRAM · RECOMMENDED", "h3_sage"],
  ["EXACT", "h3_exact"], ["SOL + LOW VRAM", "sol_low_vram"], ["NATIVE", "native"],
];
function integratedAttention(node) {
  const mode = String(widget(node,"acceleration")?.value || "native");
  const labels = {fasth3_dense_6step:"FASTH3 · SAGE + LOW VRAM", h3_sla:"SLA · SPARSE ATTENTION", matlowai_fused_turbo_manual_sigma:"FUSED · LOADER MANAGED", iamccs_progressive_2stage:"PROGRESSIVE · LOW VRAM", iamccs_progressive_3stage:"PROGRESSIVE · LOW VRAM", iamccs_progressive_pdd_2stage:"PROGRESSIVE PDD · LOW VRAM"};
  return labels[mode] || (mode === "pdd_native_8step" && widget(node,"longvid_pianosequenza_2stage_enabled")?.value ? "PDD MULTISTAGE · LOW VRAM" : "");
}
function selectedAttention(node) {
  const value = String(widget(node,"h3_attention_backend")?.value || "from_acceleration");
  const mode = value === "from_acceleration" ? String(widget(node,"acceleration")?.value || "native") : value;
  return ({sage:"h3_sage",sage_sol:"sol_low_vram",low_vram_auto:"auto_3060"})[mode] || mode;
}
function attentionLabel(node) {
  return integratedAttention(node) || ATTENTION_PRESETS.find(([,id]) => id === selectedAttention(node))?.[0] || friendly(selectedAttention(node));
}
function omniActive(node) {
  return Boolean(widget(node, "h3_omni_enabled")?.value) && shotboardMode(node) === "ref2va";
}
function omniLockedField(node, name) {
  return omniActive(node) && (["steps", "sampler_name", "scheduler", "denoise", "shift_video", "shift_audio", "acceleration"].includes(name)
    || /^(turbo_|pdd_|fused_turbo_|h3_veda_|h3_sla_)/.test(name));
}
function omniLockedRecipe(node, id) {
  return omniActive(node) && (["native", "pdd", "fasth3", "sla", "fused", "attention:sol_low_vram", "attention:sage_sol"].includes(id) || String(id).startsWith("turbo:") || String(id).startsWith("veda"));
}
function vedaAvailable(node) {
  if (omniActive(node)) return false;
  const mode = integratedAttention(node) ? String(widget(node,"acceleration")?.value) : selectedAttention(node);
  return !["sol_low_vram", "sage_sol", "sol_adaptive_safe", "sol_adaptive_balanced", "h3_sla", "matlowai_fused_turbo_manual_sigma"].includes(mode);
}
function rememberAttentionChoice(node) {
  const field = widget(node,"h3_attention_backend");
  if (!field || field.value !== "from_acceleration") return;
  const current = selectedAttention(node);
  const value = ATTENTION_PRESETS.some(([,id]) => id === current) ? current : integratedAttention(node) ? "auto_3060" : "from_acceleration";
  if (value !== "from_acceleration") setValue(node,"h3_attention_backend",value,false);
}
function setAttentionChoice(node, value) {
  if (!widget(node,"h3_attention_backend") || integratedAttention(node)) return false;
  if (value !== "from_acceleration" && !ATTENTION_PRESETS.some(([,id]) => id === value)) return false;
  setValue(node,"h3_attention_backend",value,false);
  if (!vedaAvailable(node)) setValue(node,"h3_veda_enabled",false,false);
  document.dispatchEvent(new CustomEvent("iamccs:h3-settings-changed", {detail:{source_node_id:node.id, field:"h3_attention_backend"}}));
  node._iamccsSettingsProRefresh?.();
  return true;
}
function setVedaEnabled(node, enabled) {
  if (enabled && !vedaAvailable(node)) { alert("VEDA is unavailable with Sol, SLA or Fused attention. Choose Sage / Auto Low VRAM or disable VEDA."); return false; }
  setValue(node, "h3_veda_enabled", enabled, false);
  if (enabled) setValue(node, "h3_veda_predictor", "auto", false);
  document.dispatchEvent(new CustomEvent("iamccs:h3-settings-changed", {detail:{source_node_id:node.id, field:"h3_veda_enabled"}}));
  node._iamccsSettingsProRefresh?.();
}
function applyDeclaredTurboContract(node) {
  if (shotboardMode(node) === "v2va_face_swap_lora") return false;
  if (String(widget(node, "turbo_mode")?.value || "off") === "off") return false;
  if (["pdd_native_8step", "iamccs_progressive_pdd_2stage", "fasth3_dense_6step", "matlowai_fused_turbo_manual_sigma"].includes(String(widget(node, "acceleration")?.value || "native"))) return false;
  const options = widget(node, "turbo_lora_name")?.options || {};
  const asset = (options.iamccs_h3_assets || []).find(asset => asset.name === widget(node, "turbo_lora_name")?.value);
  const contract = options.iamccs_h3_recipes?.[asset?.recipe]?.values;
  if (contract) {
    Object.entries(contract).forEach(([name, value]) => setValue(node, name, value, false));
    return true;
  }
  const steps = declaredTurboSteps(node, widget(node, "turbo_lora_name")?.value);
  if (![3, 4, 6, 8].includes(steps)) return false;
  setValue(node, "steps", steps, false); return true;
}

function speedPresetCanonicalValues(node) {
  const acceleration = String(widget(node, "acceleration")?.value || "native");
  if (["pdd_native_8step", "iamccs_progressive_pdd_2stage"].includes(acceleration)) {
    return {steps:8, sampler_name:"euler", scheduler:"simple", denoise:1, shift_video:12, shift_audio:3, pdd_strength:1};
  }
  if (acceleration === "fasth3_dense_6step") {
    return {steps:6, sampler_name:"euler", scheduler:"simple", denoise:1, shift_video:12, shift_audio:3, turbo_strength:1, turbo_mode:"off"};
  }
  if (acceleration === "h3_sla") {
    return {steps:4, sampler_name:"euler", scheduler:"simple", denoise:1, shift_video:6, shift_audio:3, turbo_strength:1, h3_sla_sparsity:0.85, h3_sla_dense_last_steps:0, turbo_sampler_mode:"res_multistep_stock"};
  }
  if (acceleration === "matlowai_fused_turbo_manual_sigma") {
    const count = Number(String(widget(node, "fused_turbo_sigma_preset")?.value || "4_step").split("_")[0]);
    return {steps:[4,6,8].includes(count)?count:4, sampler_name:"res_multistep", scheduler:"simple", denoise:1, shift_video:12, shift_audio:3};
  }
  return null;
}

function normalizeSpeedPresetContract(node) {
  const values = speedPresetCanonicalValues(node);
  if (!values) return false;
  Object.entries(values).forEach(([name, value]) => { if (widget(node, name)) setValue(node, name, value, false); });
  return true;
}
function resetAcceleration(set) { set("turbo_mode", "off"); set("turbo_lora_name", ""); set("pdd_lora_name", ""); set("fused_turbo_model_name", ""); }
function selectedAcceleration(node) {
  const value = String(widget(node, "acceleration")?.value || "native");
  const turbo = String(widget(node, "turbo_mode")?.value || "off") !== "off";
  if (widget(node, "h3_omni_enabled")?.value && shotboardMode(node) === "ref2va") return "omni";
  if (value === "h3_sla") return "sla";
  if (turbo && ["pdd_native_8step", "iamccs_progressive_pdd_2stage", "fasth3_dense_6step", "matlowai_fused_turbo_manual_sigma"].includes(value)) return "conflict";
  if (turbo) return "turbo";
  return ["pdd_native_8step", "iamccs_progressive_pdd_2stage"].includes(value) ? "pdd" : value === "fasth3_dense_6step" ? "fasth3" : value === "matlowai_fused_turbo_manual_sigma" ? "fused" : ["native", "h3_sage", "sage", "auto_3060", "low_vram_auto", "h3_exact", "sol_low_vram", "sage_sol", "comfy_kitchen"].includes(value) ? "native" : "custom";
}
function samplingRecipeLabel(node) {
  const selected = selectedAcceleration(node), steps = Number(widget(node,"steps")?.value || 0);
  return ({omni:"FASTH3 OMNI",native:"NATIVE",custom:"NATIVE / MANUAL",turbo:"TURBO",pdd:"PDD",fasth3:"FASTH3",sla:"SLA",fused:"FUSED",conflict:"CONFLICTING SPEED SETTINGS"}[selected] || selected) + ` · ${steps} STEPS`;
}
function selectedMemory(node) {
  const value = String(widget(node, "performance_profile")?.value || "");
  if (value.includes("xx90")) return "vram24"; if (value.includes("xx80")) return "vram16"; if (value.includes("xx70")) return "vram12"; return "vram8";
}
// IAMCCS FACE/UPSCALE MODE ASSISTANT V2
function selectedDelivery(node) {
  if (widget(node, "h3_open_refine_enabled")?.value) return "open-h3-refine";
  if (!widget(node, "upscale_enabled")?.value) return "native-delivery";
  const mode = String(widget(node, "upscale_mode")?.value || "off");
  if (!deliveryBranchGate(node, mode).available) return "native-delivery";
  return ({
    h3_ultimate_tiled: "ultimate-tiled",
    h3_fast_latent_2pass: "upres",
    h3_pixel_refine: "h3-pixel",
    pixel_tiled_low_vram: "pixel-safe",
    ltx23: "ltx-master",
    ltx23_per_chunk: "ltx-per-shot",
    rtx_final: "rtx-final",
  })[mode] || "custom-delivery";
}
function selectedFaceRefine(node) { return !!widget(node, "face_detailer_enabled")?.value; }
function selectedFaceMode(node) {
  const explicit = String(widget(node, "face_detailer_mode")?.value || "auto_from_profile").trim().toLowerCase();
  if (explicit === "single" || explicit === "multi_face") return explicit;
  return String(widget(node, "face_detailer_profile")?.value || "balanced").trim().toLowerCase() === "multi_face" ? "multi_face" : "single";
}
function deliveryBranchGate(node, mode) {
  const value = String(mode || "off").toLowerCase();
  if (["", "off", "native"].includes(value)) return {available:true, reason:""};
  if (["h3_fast_latent_2pass","h3_ultimate_tiled"].includes(value)) return branchGate(node, "h3_upres_model_name");
  if (["h3_latent_upres","wan22_5b"].includes(value)) {
    const inputName = value === "wan22_5b" ? "wan_frames" : "h3_latent_frames";
    const wired = connectedNodes(node).some(candidate => nodeClass(candidate) === "IAMCCS_MiniMaxH3UniversalMasterSaveR42"
      && candidate.inputs?.some(input => input.name === inputName && input.link != null));
    return {available:wired,reason:`Connect ${inputName} to Universal Master Save to enable this delivery branch.`};
  }
  if (value.startsWith("ltx23")) return branchGate(node, "ltx_seam_safe");
  if (value.includes("wan")) return branchGate(node, "wan_upscale_denoise");
  const classes = node._iamccsSettingsProConnectedClasses || connectedNodes(node).map(nodeClass);
  const has = (pattern) => classes.some((item) => pattern.test(item));
  if (value === "rtx_final") return {
    available: has(/MiniMaxH3(?:RTX4KPost|UniversalRTXFinal)/i),
    reason: "Connect the optional RTX final-delivery branch to enable this control.",
  };
  if (["pixel_tiled_low_vram", "h3_pixel_refine"].includes(value)) return {
    available: has(/MiniMaxH3(?:UniversalNativeWindowed|UniversalFast|PixelRefine)/i),
    reason: "Connect the optional H3 pixel-delivery branch to enable this control.",
  };
  return {available:true, reason:""};
}
function deliveryModeAvailable(node, mode) {
  if (shotboardMode(node) === "v2va_face_swap" && !["off", "rtx_final"].includes(mode)) return false;
  const values = widget(node, "upscale_mode")?.options?.values;
  const declared = !Array.isArray(values) || values.includes(mode);
  return declared && deliveryBranchGate(node, mode).available;
}

function applyRecipe(node, recipe) {
  const mode = shotboardMode(node), family = modeFamily(mode), set = (name, value) => setValue(node, name, value, false);
  if (["native-delivery", "ultimate-tiled", "upres", "h3-pixel", "pixel-safe", "ltx-master", "ltx-per-shot", "rtx-final"].includes(recipe)) set("h3_open_refine_enabled", false);
  if (["native", "pdd", "fasth3", "sla", "fused"].includes(recipe)) set("h3_omni_enabled", false);
  if (recipe === "open-h3-refine") {
    if (!branchGate(node, "h3_open_refine_enabled").available) return false;
    set("upscale_enabled", false); set("upscale_mode", "off");
    set("h3_open_refine_enabled", true); set("rife_mode", "off");
    set("h3_open_refine_sigma_preset", "gentle_3step");
  } else if (recipe === "omni") {
    resetAcceleration(set); set("h3_omni_enabled", true); set("acceleration", "h3_sage"); set("h3_attention_backend", "h3_sage");
    set("steps", 8); set("sampler_name", "euler"); set("scheduler", "simple");
    set("denoise", 1); set("shift_video", 12); set("shift_audio", 3);
    set("h3_veda_enabled", false); set("secondary_lora_enabled", false);
  } else if (recipe === "native") {
    rememberAttentionChoice(node);
    resetAcceleration(set); set("acceleration", "native"); set("steps", 20); set("sampler_name", "res_multistep");
    set("scheduler", "simple"); set("denoise", 1); set("shift_video", 12); set("shift_audio", 3);
  } else if (recipe === "vram8") {
    set("performance_profile", "rtx_xx60_safe"); set("motion_context_window_frames", 124);
    set("h3_exact_profile", "rtx_xx60_8_12gb_124"); set("h3_exact_chunk_rows", 2048);
    set("h3_clipproj_profile", "4b_v3.1"); set("h3_clipproj_load_mode", "dynamic"); set("vram_clean_before_decode", true);
  } else if (recipe === "vram12") {
    set("performance_profile", "rtx_xx70_balanced"); set("motion_context_window_frames", 209);
    set("h3_exact_profile", "rtx_xx70_12_16gb_209"); set("h3_exact_chunk_rows", 4096);
    set("h3_clipproj_profile", "4b_v3.1"); set("h3_clipproj_load_mode", "dynamic"); set("vram_clean_before_decode", true);
  } else if (recipe === "vram16") {
    set("performance_profile", "rtx_xx80_quality"); set("motion_context_window_frames", 294);
    set("h3_exact_profile", "rtx_xx80_16_24gb_294"); set("h3_exact_chunk_rows", 8192);
    set("h3_clipproj_profile", "4b_v3.1"); set("h3_clipproj_load_mode", "dynamic"); set("vram_clean_before_decode", true);
  } else if (recipe === "vram24") {
    set("performance_profile", "rtx_xx90_max"); set("motion_context_window_frames", 362);
    set("h3_exact_profile", "rtx_xx90_24gb_362"); set("h3_exact_chunk_rows", 16384);
    set("h3_clipproj_profile", "4b_v3.1"); set("h3_clipproj_load_mode", "dynamic"); set("vram_clean_before_decode", true);
  } else if (recipe === "pdd") {
    const asset = firstChoice(node, "pdd_lora_name", (name) => isPddLoRA(name, family));
    if (!asset) { alert(`PDD is unavailable: install a ${family.toUpperCase()} PDD LoRA before selecting this recipe.`); return false; }
    rememberAttentionChoice(node);
    resetAcceleration(set); set("pdd_lora_name", asset); set("pdd_strength", 1); set("acceleration", "pdd_native_8step");
    set("steps", 8); set("sampler_name", "euler"); set("scheduler", "simple"); set("denoise", 1); set("shift_video", 12); set("shift_audio", 3);
  } else if (recipe === "fasth3") {
    const asset = fastH3Asset(node, mode); if (!asset) { alert("FastH3 is unavailable: no compatible FastH3 LoRA is installed."); return false; }
    rememberAttentionChoice(node);
    resetAcceleration(set); set("turbo_mode", "off"); set("turbo_lora_name", asset); set("turbo_strength", 1);
    set("acceleration", "fasth3_dense_6step"); set("steps", 6); set("sampler_name", "euler"); set("scheduler", "simple"); set("denoise", 1); set("shift_video", 12); set("shift_audio", 3);
  } else if (recipe === "sla") {
    const asset = firstChoice(node, "turbo_lora_name", (name) => name.includes("sla") && name.includes(family)); if (!asset) { alert(`SLA is unavailable: no compatible ${family.toUpperCase()} SLA LoRA is installed.`); return false; }
    rememberAttentionChoice(node); set("h3_veda_enabled", false);
    resetAcceleration(set); set("turbo_mode", "early_8_10"); set("turbo_lora_name", asset); set("turbo_strength", 1);
    set("acceleration", "h3_sla"); set("steps", 4); set("h3_sla_sparsity", 0.85); set("h3_sla_dense_last_steps", 0); set("sampler_name", "euler"); set("scheduler", "simple"); set("denoise", 1); set("shift_video", 6); set("shift_audio", 3); set("turbo_sampler_mode", "res_multistep_stock");
  } else if (recipe === "fused") {
    const current = String(widget(node, "fused_turbo_model_name")?.value || "");
    const asset = current || firstChoice(node, "fused_turbo_model_name", (name) => (name.includes("fused") || name.includes("turbo")) && !name.includes("ref2"));
    if (!asset) { alert("Fused Fast is unavailable: no compatible fused diffusion model is installed."); return false; }
    const convrot = asset.toLowerCase().includes("convrot");
    if (!(convrot ? ["t2va", "i2va", "fl2va", "ref2va"] : ["t2va"]).includes(shotboardMode(node))) { alert("This fused checkpoint does not support the selected mode."); return false; }
    rememberAttentionChoice(node); set("h3_veda_enabled", false);
    resetAcceleration(set); set("fused_turbo_model_name", asset); set("acceleration", "matlowai_fused_turbo_manual_sigma");
    set("fused_turbo_sigma_preset", "4_step"); set("steps", 4); set("sampler_name", "res_multistep"); set("scheduler", "simple");
    set("denoise", 1); set("shift_video", 12); set("shift_audio", 3);
  } else if (recipe === "control") {
    set("h3_controlnet_enabled", true); const asset = firstChoice(node, "h3_controlnet_name", () => true); if (asset) set("h3_controlnet_name", asset);
    if (!(Number(widget(node, "h3_controlnet_strength")?.value) > 0)) set("h3_controlnet_strength", 1);
  } else if (recipe === "face") {
    set("face_detailer_enabled", false); set("upscale_enabled", false);
    if (!String(widget(node, "h3_faceswap_sam_model")?.value || "").trim()) {
      const asset = firstChoice(node, "h3_faceswap_sam_model", () => true);
      if (asset) set("h3_faceswap_sam_model", asset);
    }
    set("h3_faceswap_generate_new_audio", false); set("audio_mode", "h3_custom_audio_drive");
  } else if (recipe === "face-refine") {
    const enabled = !Boolean(widget(node, "face_detailer_enabled")?.value);
    set("face_detailer_enabled", enabled);
    if (enabled && !String(widget(node, "face_detailer_profile")?.value || "")) set("face_detailer_profile", "balanced");
    if (enabled && !String(widget(node, "face_detailer_mode")?.value || "")) set("face_detailer_mode", "auto_from_profile");
    if (enabled && !(Number(widget(node, "face_detailer_max_faces")?.value) >= 1)) set("face_detailer_max_faces", 4);
    if (enabled && !Number.isFinite(Number(widget(node, "face_detailer_min_coverage")?.value))) set("face_detailer_min_coverage", 0.10);
    if (enabled && !(Number(widget(node, "face_detailer_canvas_width")?.value) >= 256)) set("face_detailer_canvas_width", 512);
    if (enabled && !(Number(widget(node, "face_detailer_canvas_height")?.value) >= 256)) set("face_detailer_canvas_height", 512);
    if (enabled && !Number.isFinite(Number(widget(node, "face_detailer_crop_factor")?.value))) set("face_detailer_crop_factor", 2.5);
    if (enabled && !Number.isFinite(Number(widget(node, "face_detailer_confidence")?.value))) set("face_detailer_confidence", 0.35);
    if (enabled && !(Number(widget(node, "face_detailer_steps")?.value) >= 1)) set("face_detailer_steps", 4);
    if (enabled && !Number.isFinite(Number(widget(node, "face_detailer_denoise")?.value))) set("face_detailer_denoise", 0.20);
    if (enabled && !(Number(widget(node, "face_detailer_window_frames")?.value) >= 34)) set("face_detailer_window_frames", 73);
    if (enabled && !Number.isFinite(Number(widget(node, "face_detailer_window_overlap")?.value))) set("face_detailer_window_overlap", 22);
    if (enabled && !Number.isFinite(Number(widget(node, "face_detailer_strength_small_face")?.value))) set("face_detailer_strength_small_face", 1.0);
    if (enabled && !Number.isFinite(Number(widget(node, "face_detailer_strength_large_face")?.value))) set("face_detailer_strength_large_face", 0.35);
    if (enabled && !Number.isFinite(Number(widget(node, "face_detailer_blend")?.value))) set("face_detailer_blend", 0.70);
    if (enabled && !String(widget(node, "face_detailer_paste_region")?.value || "")) set("face_detailer_paste_region", "face_only");
    if (enabled && !Number.isFinite(Number(widget(node, "face_detailer_feather")?.value))) set("face_detailer_feather", 6);
  } else if (recipe === "native-delivery") {
    set("upscale_enabled", false); set("upscale_mode", "off"); set("rife_mode", "off");
  } else if (recipe === "ultimate-tiled") {
    if (mode === "v2va_face_swap") return false;
    node.properties ||= {}; node.properties.iamccs_native_two_pass_mode = "h3_ultimate_tiled";
    set("upscale_enabled", true); set("upscale_mode", "h3_ultimate_tiled");
    set("h3_upres_steps", 1); set("h3_upres_denoise", 0.2);
    set("h3_upres_sampler", "sa_solver"); set("h3_upres_scheduler", "simple");
    set("h3_upres_seed", 777);
    set("h3_upres_precision", "fp16"); set("h3_upres_device", "cuda"); set("h3_upres_keep_models_resident", false);
    const installed = firstChoice(node, "h3_upres_model_name", name => name.endsWith("minimax_h3_latent_upscaler_3d_fp16.safetensors"));
    if (!String(widget(node,"h3_upres_model_name")?.value || "").trim() && installed) set("h3_upres_model_name", installed);
    const portrait = Number(widget(node,"height")?.value) > Number(widget(node,"width")?.value);
    for (const [key, value] of Object.entries({h3_upres_temporal_chunk:85,h3_upres_temporal_overlap:17,h3_upres_anchor_strength:0.999,
      h3_upres_tile_width:portrait?480:864,h3_upres_tile_height:portrait?864:480,h3_upres_overlap_width:128,h3_upres_overlap_height:128,
      h3_upres_fade_width:32,h3_upres_fade_height:32,h3_upres_min_tile_size:256,h3_upres_overlap_mode:"earlier",h3_upres_overlap_blend:"linear",h3_upres_pixel_groups:2})) set(key,value);
    for (const axis of ["width","height"]) if (!(Number(widget(node,`upscale_${axis}`)?.value) > Number(widget(node,axis)?.value))) set(`upscale_${axis}`, Math.ceil(Number(widget(node,axis)?.value || 512)*2/32)*32);
  } else if (recipe === "upres") {
    node.properties ||= {}; node.properties.iamccs_native_two_pass_mode = "h3_fast_latent_2pass";
    set("upscale_enabled", true); set("upscale_mode", "h3_fast_latent_2pass");
    set("h3_upres_seed", -1);
    // Studio-style visual refinement: AV split -> learned 3D lift -> short
    // target-resolution H3 pass. Stage-1 model/LoRA stays mode-matched to Shotboard.
    set("h3_upres_precision", "fp16"); set("h3_upres_keep_models_resident", false);
    // Match the working R41 Fast Latent baseline; this is a starting recipe,
    // not a guarantee of visual lip sync at every resolution.
    set("h3_upres_steps", 8); set("h3_upres_denoise", 0.3);
    set("h3_upres_sampler", "euler"); set("h3_upres_scheduler", "simple");
  } else if (recipe === "h3-pixel") {
    set("upscale_enabled", true); set("upscale_mode", "h3_pixel_refine");
  } else if (recipe === "pixel-safe") {
    set("upscale_enabled", true); set("upscale_mode", "pixel_tiled_low_vram"); set("h3_pixel_tiled_method", "cpu_lanczos");
  } else if (recipe === "ltx-master") {
    set("upscale_enabled", true); set("upscale_mode", "ltx23");
  } else if (recipe === "ltx-per-shot") {
    set("upscale_enabled", true); set("upscale_mode", "ltx23_per_chunk");
  } else if (recipe === "rtx-final") {
    set("upscale_enabled", true); set("upscale_mode", "rtx_final");
  }
  document.dispatchEvent(new CustomEvent("iamccs:h3-settings-changed", { detail: { source_node_id: node.id, recipe } }));
  node._iamccsSettingsProRefresh?.();
  return true;
}


const FACE_REFINE_PRESET_HELP = Object.freeze({
  faceq_safe: { label: "SAFE · 12 GB", info: "512×512 face canvas, 4 refine steps and a conservative temporal window. Lowest VRAM/time cost. Definition strength is controlled separately." },
  faceq_balanced: { label: "BALANCED", info: "640×640 face canvas and 5 refine steps. Recommended production starting point for 12 GB GPUs. Definition strength is controlled separately." },
  faceq_high: { label: "HIGH", info: "640×640 face canvas, 6 refine steps and a longer temporal window. More reconstruction work without changing the Definition preset." },
  faceq_max: { label: "MAX", info: "768×768 face canvas, 7 refine steps and the heaviest temporal processing. Highest processing cost; Definition remains independently selectable." },
  facesize_wide: { label: "WIDE CHARACTER", info: "For large/close faces. Uses a wider crop context and gentler large-face denoise. Does not change Single/Multi Face mode." },
  facesize_medium: { label: "MEDIUM", info: "General-purpose face scale. Balanced crop, detector confidence and denoise scaling. Does not change Single/Multi Face mode." },
  facesize_small: { label: "SMALL FACES", info: "For faces occupying a small part of the frame. Tighter crop and stronger detail recovery. Does not change Single/Multi Face mode." },
  facesize_tiny: { label: "TINY FACES", info: "For very small/distant faces. Lowest detector threshold and strongest small-face recovery. Watch for false detections." },
  facedef_soft: { label: "SOFT / PRESERVE", info: "Minimal regeneration. Best when identity preservation is more important than added sharpness." },
  facedef_balanced: { label: "BALANCED", info: "Moderate denoise and blend. Natural default for most faces without over-sharpening." },
  facedef_defined: { label: "DEFINED", info: "Stronger reconstruction and blend for visibly soft faces. Good detail gain with moderate identity risk." },
  facedef_strong: { label: "STRONG", info: "Most aggressive definition preset. Use for badly degraded faces; can alter identity or skin texture if pushed too far." },
});

function applyFaceRefineTuning(node, recipe) {
  const set = (name, value) => setValue(node, name, value, false);
  const presets = {
    // The three preset families are intentionally orthogonal. Selecting a
    // Quality preset must not invalidate the selected Definition preset, etc.
    "faceq_safe": { face_detailer_canvas_width: 512, face_detailer_canvas_height: 512, face_detailer_steps: 4, face_detailer_window_frames: 73, face_detailer_window_overlap: 22 },
    "faceq_balanced": { face_detailer_canvas_width: 640, face_detailer_canvas_height: 640, face_detailer_steps: 5, face_detailer_window_frames: 73, face_detailer_window_overlap: 22 },
    "faceq_high": { face_detailer_canvas_width: 640, face_detailer_canvas_height: 640, face_detailer_steps: 6, face_detailer_window_frames: 90, face_detailer_window_overlap: 22 },
    "faceq_max": { face_detailer_canvas_width: 768, face_detailer_canvas_height: 768, face_detailer_steps: 7, face_detailer_window_frames: 90, face_detailer_window_overlap: 39 },
    "facesize_wide": { face_detailer_crop_factor: 2.60, face_detailer_confidence: 0.25, face_detailer_strength_small_face: 0.95, face_detailer_strength_large_face: 0.20 },
    "facesize_medium": { face_detailer_crop_factor: 2.40, face_detailer_confidence: 0.30, face_detailer_strength_small_face: 1.00, face_detailer_strength_large_face: 0.30 },
    "facesize_small": { face_detailer_crop_factor: 2.20, face_detailer_confidence: 0.28, face_detailer_strength_small_face: 1.00, face_detailer_strength_large_face: 0.40 },
    "facesize_tiny": { face_detailer_crop_factor: 2.00, face_detailer_confidence: 0.22, face_detailer_strength_small_face: 1.00, face_detailer_strength_large_face: 0.50 },
    "facedef_soft": { face_detailer_denoise: 0.16, face_detailer_blend: 0.65, face_detailer_feather: 8 },
    "facedef_balanced": { face_detailer_denoise: 0.22, face_detailer_blend: 0.72, face_detailer_feather: 6 },
    "facedef_defined": { face_detailer_denoise: 0.26, face_detailer_blend: 0.80, face_detailer_feather: 6 },
    "facedef_strong": { face_detailer_denoise: 0.30, face_detailer_blend: 0.86, face_detailer_feather: 5 },
  };
  const values = presets[String(recipe)] || null;
  if (!values) return false;
  set("face_detailer_enabled", true);
  if (!String(widget(node, "face_detailer_mode")?.value || "")) set("face_detailer_mode", "auto_from_profile");
  Object.entries(values).forEach(([name, value]) => set(name, value));
  document.dispatchEvent(new CustomEvent("iamccs:h3-settings-changed", { detail: { source_node_id: node.id, recipe } }));
  node._iamccsSettingsProRefresh?.();
  return true;
}

function resetFaceRefinePresets(node) {
  const set = (name, value) => setValue(node, name, value, false);
  const defaults = {
    face_detailer_canvas_width: 512,
    face_detailer_canvas_height: 512,
    face_detailer_crop_factor: 2.5,
    face_detailer_confidence: 0.35,
    face_detailer_steps: 4,
    face_detailer_denoise: 0.20,
    face_detailer_window_frames: 73,
    face_detailer_window_overlap: 22,
    face_detailer_strength_small_face: 1.00,
    face_detailer_strength_large_face: 0.35,
    face_detailer_blend: 0.70,
    face_detailer_paste_region: "face_only",
    face_detailer_feather: 6,
  };
  Object.entries(defaults).forEach(([name, value]) => set(name, value));
  node.properties ||= {};
  node.properties.iamccs_recipe_highlight ||= {};
  delete node.properties.iamccs_recipe_highlight.face_refine_quality;
  delete node.properties.iamccs_recipe_highlight.face_refine_size;
  delete node.properties.iamccs_recipe_highlight.face_refine_definition;
  // Clean the stale key written by R44.5/R44.6 when the face bars were
  // accidentally classified as the generic memory recipe group.
  if (/^(faceq_|facesize_|facedef_)/.test(String(node.properties.iamccs_recipe_highlight.memory || ""))) {
    delete node.properties.iamccs_recipe_highlight.memory;
  }
  app.graph?.change?.();
  document.dispatchEvent(new CustomEvent("iamccs:h3-settings-changed", { detail: { source_node_id: node.id, recipe: "face_refine_reset" } }));
  node._iamccsSettingsProRefresh?.();
  return true;
}

function applyAccelerationChoice(node, value) {
  const recipes = {
    native: "native",
    pdd_native_8step: "pdd",
    fasth3_dense_6step: "fasth3",
    h3_sla: "sla",
    matlowai_fused_turbo_manual_sigma: "fused",
  };
  const recipe = recipes[String(value || "native")];
  if (recipe) {
    const previous = String(widget(node, "acceleration")?.value || "native");
    if (!applyRecipe(node, recipe)) setValue(node, "acceleration", previous, false);
    return;
  }
  setValue(node, "acceleration", value, false);
  if (widget(node,"h3_attention_backend")) setValue(node,"h3_attention_backend","from_acceleration",false);
  if (!vedaAvailable(node)) setValue(node,"h3_veda_enabled",false,false);
  document.dispatchEvent(new CustomEvent("iamccs:h3-settings-changed", {detail:{source_node_id:node.id, field:"acceleration"}}));
}

function applyMemoryProfileChoice(node, value) {
  const profile = String(value || "low_vram_balanced");
  const set = (name, next) => setValue(node, name, next, false);
  const contracts = {
    low_vram_draft:       { window:124, exact:"rtx_xx60_8_12gb_124", rows:1024, text:"cpu_safe_12gb", clip:"streaming" },
    low_vram_balanced:    { window:124, exact:"rtx_xx60_8_12gb_124", rows:2048, text:"gpu_auto",      clip:"dynamic" },
    low_vram_turbo:       { window:209, exact:"rtx_xx70_12_16gb_209", rows:2048, text:"gpu_auto",      clip:"dynamic" },
    h3_turbo_quality:     { window:209, exact:"rtx_xx70_12_16gb_209", rows:4096, text:"gpu_auto",      clip:"dynamic" },
    h3_native_quality:    { window:294, exact:"rtx_xx80_16_24gb_294", rows:8192, text:"gpu_auto",      clip:"dynamic" },
    rtx3060_draft:        { window:124, exact:"rtx3060_12gb_124",      rows:1024, text:"cpu_safe_12gb", clip:"streaming" },
    rtx3060_balanced:     { window:124, exact:"rtx3060_12gb_124",      rows:2048, text:"gpu_auto",      clip:"dynamic" },
    rtx3060_turbo:        { window:209, exact:"rtx3060_12gb_209",      rows:2048, text:"gpu_auto",      clip:"dynamic" },
    wide_character_12gb:  { window:124, exact:"rtx3060_12gb_124",      rows:3072, text:"gpu_auto",      clip:"dynamic" },
  };
  const recipe = profile.includes("xx90") ? "vram24" : profile.includes("xx80") ? "vram16" : profile.includes("xx70") ? "vram12" : profile.includes("xx60") ? "vram8" : "";
  if (recipe) { applyRecipe(node, recipe); return; }
  const contract = contracts[profile];
  if (!contract) { node._iamccsSettingsProRefresh?.(); return; }
  set("performance_profile", profile);
  set("motion_context_window_frames", contract.window);
  set("h3_exact_profile", contract.exact);
  set("h3_exact_chunk_rows", contract.rows);
  set("text_encoder_device", contract.text);
  set("h3_clipproj_profile", "4b_v3.1");
  set("h3_clipproj_load_mode", contract.clip);
  set("vram_clean_before_decode", true);
  document.dispatchEvent(new CustomEvent("iamccs:h3-settings-changed", { detail: { source_node_id: node.id, recipe: profile } }));
  node._iamccsSettingsProRefresh?.();
}

function accelerationAvailable(node, recipe, mode = shotboardMode(node)) {
  const family = modeFamily(mode);
  if (recipe === "pdd") return Boolean(firstChoice(node, "pdd_lora_name", (name) => isPddLoRA(name, family)));
  if (recipe === "fasth3") return Boolean(fastH3Asset(node, mode));
  if (recipe === "sla") return widget(node, "turbo_lora_name")?.options?.iamccs_sla_available !== false && Boolean(firstChoice(node, "turbo_lora_name", (name) => name.includes("sla") && name.includes(family)));
  if (recipe === "fused") {
    const name = String(widget(node, "fused_turbo_model_name")?.value || "") || firstChoice(node, "fused_turbo_model_name", (name) => name.includes("fused") || name.includes("turbo"));
    return Boolean(name) && (String(name).toLowerCase().includes("convrot") ? ["t2va", "i2va", "fl2va", "ref2va"] : ["t2va"]).includes(String(mode));
  }
  return true;
}

function normalizeSeedPolicy(node) {
  const item=widget(node,'seed_policy');
  if(item && (item.value == null || String(item.value).trim() === '')) item.value='fixed_per_generation';
}

function normalizeInstalledSpeedAsset(node) {
  const acceleration = String(widget(node, "acceleration")?.value || "native");
  const family = modeFamily(shotboardMode(node));
  if (acceleration === "fasth3_dense_6step") {
    const current = String(widget(node, "turbo_lora_name")?.value || "");
    if (!isFastH3LoRA(current)) {
      const compatible = fastH3Asset(node);
      if (compatible) { setValue(node, "turbo_lora_name", compatible, false); setValue(node, "steps", 6, false); }
    }
  }
  if (acceleration === "pdd_native_8step") {
    const current = String(widget(node, "pdd_lora_name")?.value || "");
    if (!isPddLoRA(current, family)) {
      const compatible = firstChoice(node, "pdd_lora_name", (name) => isPddLoRA(name, family));
      if (compatible) setValue(node, "pdd_lora_name", compatible, false);
    }
  }
  normalizeSpeedPresetContract(node);
}

function mount(node) {
  normalizeSeedPolicy(node);
  normalizeInstalledSpeedAsset(node);
  normalizeSpeedPresetContract(node);
  normalizePianosequenzaStageControls(node);
  installPianosequenzaStageQueueGuards(node);
  const seedPolicy = widget(node, "seed_policy");
  if (seedPolicy && !seedPolicy._iamccsSeedHook) {
    seedPolicy._iamccsSeedHook = true;
    seedPolicy.serializeValue = () => {normalizeSeedPolicy(node);return seedPolicy.value;};
    seedPolicy.beforeQueued = ({isPartialExecution = false} = {}) => {
      normalizeSeedPolicy(node);
      if (isPartialExecution || !String(seedPolicy.value).startsWith("random_")) return;
      const words = crypto.getRandomValues(new Uint32Array(2));
      setValue(node, "seed", (words[0] & 0x1fffff) * 4294967296 + words[1], false);
      document.dispatchEvent(new CustomEvent("iamccs:h3-settings-changed", {detail:{source_node_id:node.id}}));
    };
  }
  const deliveryToggle = widget(node, "upscale_enabled");
  if (deliveryToggle && !deliveryToggle._iamccsOptionalBranchGuard) {
    deliveryToggle._iamccsOptionalBranchGuard = true;
    deliveryToggle.serializeValue = () => {
      const selectedMode = String(widget(node, "upscale_mode")?.value || "off");
      const gate = deliveryBranchGate(node, selectedMode);
      if (Boolean(deliveryToggle.value) && !gate.available) {
        deliveryToggle.value = false;
        node.properties ||= {};
        node.properties.iamccs_optional_delivery_muted = {
          mode: selectedMode,
          reason: gate.reason,
          at: new Date().toISOString(),
        };
      }
      return Boolean(deliveryToggle.value);
    };
  }
  if (node._iamccsSettingsProMounted) return; node._iamccsSettingsProMounted = true; (node.widgets || []).forEach(hideWidget);
  const swapDuration = createSwapDuration(node, {
    api, mode:() => shotboardMode(node), connected:() => connectedNodes(node),
    setDuration:(seconds, reason) => setShotboardDuration(node, seconds, reason),
    refresh:() => node._iamccsSettingsProRefresh?.(),
  });
  installSwapDurationQueueGuard(app);
  const root = document.createElement("div"); root.className = "iamccs-h3pro";
  root.innerHTML = `
  <style>
  .iamccs-h3pro .h3p-rail{background:linear-gradient(180deg,#17232f,#0b111a);gap:7px;overflow-y:auto}.iamccs-h3pro .h3p-tab{min-height:40px;flex-shrink:0;border:1px solid #334454;background:linear-gradient(135deg,#223140,#141d28);box-shadow:inset 0 1px #ffffff0b,0 2px 5px #0003;transition:border-color .15s,background .15s;padding:9px 12px;white-space:normal;line-height:1.25}.iamccs-h3pro .h3p-tab.active{border-color:#dec087;border-left:3px solid #f5cf8d;background:linear-gradient(100deg,#493922,#233444);box-shadow:0 0 12px #d8b26c18,inset 0 1px #fff1}.iamccs-h3pro .h3p-tab:hover{border-color:#88b6c6;color:#fff}.iamccs-h3pro .h3p-choice.active{border-left:3px solid #eac383;box-shadow:0 3px 12px #0004}.iamccs-h3pro .h3p-choice{min-height:68px;padding:12px}
  .iamccs-h3pro{height:100%;padding:12px;box-sizing:border-box;background:radial-gradient(circle at 85% 0,#273346 0,transparent 34%),linear-gradient(145deg,#090d13,#111923 62%,#0a0e14);border:1px solid #8b7046;border-radius:14px;color:#eaf0f5;font:11px Inter,Segoe UI,sans-serif;overflow:hidden}.iamccs-h3pro *{box-sizing:border-box}.h3p-head{height:48px;display:flex;align-items:center;gap:12px;border-bottom:1px solid #344253}.h3p-mark{padding:6px 10px;border:1px solid #d2a65c;border-radius:999px;background:#32281a;color:#f8d89c;font-size:9px;font-weight:900;letter-spacing:.08em}.h3p-title{font:700 17px Georgia,serif}.h3p-sub{color:#8291a0;font-size:9px}.h3p-mode{margin-left:auto;text-align:right}.h3p-mode b{display:block;color:#7ee2ad;font-size:10px}.h3p-layout{display:grid;grid-template-columns:155px minmax(500px,1fr) 260px;gap:10px;height:calc(100% - 58px);padding-top:10px}.h3p-rail,.h3p-main,.h3p-truth{min-height:0;border:1px solid #2e3a47;border-radius:10px;background:rgba(12,18,25,.88)}.h3p-rail{padding:7px;display:flex;flex-direction:column;gap:5px}.h3p-tab{height:38px;padding:0 10px;border:1px solid transparent;border-radius:7px;background:transparent;color:#94a2b0;text-align:left;font-size:9px;font-weight:850;letter-spacing:.05em;cursor:pointer}.h3p-tab:hover{background:#182330;color:#fff}.h3p-tab.active{border-color:#a98650;background:linear-gradient(90deg,#3c3020,#1c2530);color:#f3d69c}.h3p-mode-folder{height:38px;border-color:#8b5bc8;background:linear-gradient(90deg,#251638,#171d2a);color:#d9b7ff;font-weight:950;letter-spacing:.11em}.h3p-mode-folder.open,.h3p-mode-folder.context-active{border-color:#c48cff;color:#f0dcff;box-shadow:inset 4px 0 0 #bd7fff,0 0 12px #8e5ac52a}.h3p-mode-folder .h3p-folder-arrow{float:right;font-size:10px;color:#d6a6ff}.h3p-subtab{height:32px;margin-left:9px;width:calc(100% - 9px);border-left:2px solid #69438e;background:#121723;color:#b8a6c9;font-size:8px}.h3p-subtab.active{border-left-color:#c58aff;background:linear-gradient(90deg,#2d1c3f,#18222d);color:#efd7ff}.h3p-owner{margin-top:auto;padding:10px;border-radius:8px;background:#111b24;color:#8493a2;font-size:8px;line-height:1.45}.h3p-owner strong{display:block;color:#f0c97d;margin-bottom:4px}.h3p-main{padding:12px;overflow:auto}.h3p-section-title{font:700 16px Georgia,serif;color:#f0d39e}.h3p-section-note{margin:4px 0 12px;color:#8493a2;font-size:9px}.h3p-recipes{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px}.h3p-recipe,.h3p-choice{padding:8px 10px;border:1px solid #4c5c6d;border-radius:7px;background:#1a2530;color:#d9e2e9;font-size:8px;font-weight:850;cursor:pointer}.h3p-recipe:hover,.h3p-recipe.active,.h3p-choice:hover,.h3p-choice.active{border-color:#d0a45c;color:#f6d99d;background:#2b261d;box-shadow:0 0 0 1px rgba(240,190,99,.2),0 0 12px rgba(240,190,99,.16)}.h3p-recipe[disabled]{opacity:.35;cursor:not-allowed}.h3p-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.h3p-functional{grid-column:1/-1;padding:9px;border:1px solid #344353;border-radius:10px;background:linear-gradient(145deg,#111b25,#0c141c)}.h3p-functional-title{margin:0 0 8px;color:#e3bd78;font-size:8px;font-weight:900;letter-spacing:.09em}.h3p-functional-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.h3p-resolution{height:31px;min-width:220px;border:1px solid #9b7944;border-radius:7px;background:#171f27;color:#f1d49b;padding:0 8px;font-size:8px;font-weight:850}.h3p-field{min-height:58px;padding:7px;border:1px solid #2d3945;border-radius:8px;background:#101821}.h3p-field.muted{opacity:.48;border-style:dashed;background:#0b1218}.h3p-field.muted label::after{content:" · BRANCH OFF";color:#d0a45c;font-size:7px}.h3p-field.muted input,.h3p-field.muted select{cursor:not-allowed}.h3p-field label{display:block;margin-bottom:5px;color:#9ba8b5;font-size:8px;font-weight:800}.h3p-field input,.h3p-field select{width:100%;height:29px;border:1px solid #43515e;border-radius:6px;background:#0a1118;color:#edf2f6;padding:0 7px;font-size:9px}.h3p-field input[type=checkbox]{width:18px;height:18px;accent-color:#d3a758}.h3p-field small{display:block;margin-top:4px;color:#667786;font-size:7px}.h3p-truth{padding:12px;overflow:auto}.h3p-truth h3{margin:0 0 10px;color:#f0d39e;font:700 14px Georgia,serif}.h3p-truth-row{padding:8px 0;border-bottom:1px solid #26323d}.h3p-truth-row span{display:block;color:#718190;font-size:7px;font-weight:900}.h3p-truth-row b{display:block;margin-top:3px;color:#dce5eb;font:600 9px Consolas,monospace;overflow-wrap:anywhere}.h3p-health{margin-top:10px;padding:9px;border-left:3px solid #68d69a;border-radius:6px;background:#11231c;color:#a9e6c5;font-size:8px;line-height:1.45}.h3p-health.warn{border-color:#e7a14e;background:#2a2014;color:#ffd59a}.h3p-health.error{border-color:#e86767;background:#2d1619;color:#ffb0b0}.h3p-context{margin-bottom:10px;padding:9px;border:1px solid #405064;border-radius:8px;background:#14202c;color:#a9b7c5;font-size:9px}.h3p-context b{color:#f2cf8b}.h3p-question{margin:12px 0 6px;color:#f0cf91;font-size:9px;font-weight:900;letter-spacing:.05em}.h3p-choice-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px}.h3p-choice{text-align:left}.h3p-choice span{display:block;margin-top:3px;color:#8493a2;font-weight:500;line-height:1.3}.h3p-choice[disabled]{opacity:.34;filter:saturate(.45);cursor:not-allowed;border-style:dashed}.h3p-choice[disabled] span::after{content:" · BRANCH MISSING";color:#e7a14e}.h3p-recipe.active{outline:1px solid #f2c97e;outline-offset:1px}.h3p-flow{padding:10px;border:1px solid #354455;border-radius:9px;background:#0d151e}
  .h3p-recipe.active{border-color:#d0a45c;color:#f6d99d;background:#2b261d;box-shadow:inset 0 0 0 1px #6b5330}.h3p-recipe[disabled]::after{content:" · UNAVAILABLE";color:#e7a14e}.h3p-main[data-section="memory"] .h3p-recipes,.h3p-memory-recipes{padding:9px;border:1px solid #4d79a0;border-radius:9px;background:linear-gradient(135deg,#122838,#101c2a)}.h3p-main[data-section="memory"] .h3p-recipe,.h3p-memory-recipes .h3p-recipe{border-color:#4d83ad;color:#bfe4ff}.h3p-main[data-section="speed"] .h3p-recipes,.h3p-speed-recipes{padding:9px;border:1px solid #9a7140;border-radius:9px;background:linear-gradient(135deg,#302313,#201a14)}.h3p-main[data-section="speed"] .h3p-recipe,.h3p-speed-recipes .h3p-recipe{border-color:#b0844b;color:#ffe0a9}.h3p-delivery-recipes{padding:9px;border:1px solid #5b8a69;border-radius:9px;background:#12251a}.h3p-face-refine-presets-grid{width:100%;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px;align-items:stretch}.h3p-face-refine-preset-group{min-width:0;display:flex;flex-direction:column;gap:7px;padding:10px;border:1px solid #4d79a0;border-radius:9px;background:linear-gradient(135deg,#122838,#101c2a)}.h3p-face-refine-preset-group .h3p-functional-title{margin:0;color:#e3bd78}.h3p-face-refine-preset-group .h3p-face-refine-group-note{min-height:32px;color:#8198a8;font-size:7px;line-height:1.35}.h3p-face-refine-preset-group .h3p-memory-recipes{padding:0;border:0;background:transparent;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;margin:0}.h3p-face-refine-preset-group .h3p-recipe{min-height:38px;padding:6px 7px;white-space:normal;line-height:1.2}.h3p-face-refine-preset-group .h3p-face-refine-help{min-height:44px;padding:7px 8px;border-left:2px solid #d0a45c;background:#0c1720;color:#9eb0bc;font-size:7px;line-height:1.4}.h3p-face-refine-preset-group .h3p-face-refine-help b{color:#f0d39e}.h3p-main[data-section="face_refine"]>.h3p-recipes{display:flex;align-items:stretch}.h3p-main[data-section="face_refine"]>.h3p-recipes>.h3p-recipe{min-height:42px}.h3p-face-refine-stack{width:100%;display:flex;align-items:center;justify-content:space-between;gap:10px;padding:8px 10px;border:1px solid #354b5f;background:#0d1821;color:#9fb0bc;font-size:7px;line-height:1.35}.h3p-face-refine-stack b{color:#f0d39e}.h3p-face-refine-reset{flex:0 0 auto;padding:7px 10px;border:1px solid #7d6850;background:#1d2020;color:#f0d39e;font:900 7px Consolas,monospace;cursor:pointer}.h3p-face-refine-reset:hover{border-color:#d0a45c;background:#2b261d}@media(max-width:1100px){.h3p-face-refine-presets-grid{grid-template-columns:1fr}.h3p-face-refine-preset-group .h3p-face-refine-group-note,.h3p-face-refine-preset-group .h3p-face-refine-help{min-height:0}}
  .h3p-assets{grid-column:1/-1;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.h3p-library-title{grid-column:1/-1;margin-top:8px;padding:9px;border:1px solid #3d5062;border-radius:8px;background:#101a24;color:#e3bd78;font-size:8px;font-weight:900;letter-spacing:.09em}.h3p-library-title small{display:block;margin-top:4px;color:#8499a8;font-weight:500;letter-spacing:0}.h3p-asset{border:1px solid #43576a;border-radius:10px;background:#111d29;padding:8px;min-width:0}.h3p-asset img,.h3p-asset-placeholder{display:block;width:100%;height:145px;object-fit:contain;background:#080e15;border-radius:6px;color:#7e90a0;text-align:center;align-content:center}.h3p-cache-card .h3p-asset-placeholder{height:62px;color:#d7b879;font-weight:900;letter-spacing:.08em}.h3p-asset strong,.h3p-asset small{display:block;overflow-wrap:anywhere;margin:7px 0}.h3p-asset small{color:#8499a8}.h3p-asset-actions{display:flex;flex-wrap:wrap;gap:6px}.h3p-asset-actions button{padding:6px 9px;border:1px solid #a98754;border-radius:6px;background:#222b34;color:#f0d5a5;cursor:pointer}.h3p-asset-actions button.h3p-danger{border-color:#a85656;background:#34191d;color:#ffb5b5}.h3p-asset-actions button:disabled{opacity:.45;cursor:wait}.h3p-asset-status{grid-column:1/-1;color:#a5bac9}
  .h3p-open-editor{height:30px;padding:0 12px;border:1px solid #d0a45c;border-radius:7px;background:linear-gradient(135deg,#4a371d,#263543);color:#ffe2aa;font-size:8px;font-weight:900;letter-spacing:.06em;cursor:pointer;white-space:nowrap}.h3p-open-editor:hover{border-color:#f2ce88;background:linear-gradient(135deg,#6a4b21,#30485a);color:#fff3d5}.iamccs-h3pro:fullscreen,.iamccs-h3pro.h3p-monitor-open{width:100vw!important;height:100vh!important;max-width:none!important;max-height:none!important;padding:16px!important;border:0!important;border-radius:0!important}.iamccs-h3pro:fullscreen .h3p-layout,.iamccs-h3pro.h3p-monitor-open .h3p-layout{height:calc(100vh - 66px)!important;grid-template-columns:190px minmax(0,1fr) 330px}.iamccs-h3pro:fullscreen .h3p-head,.iamccs-h3pro.h3p-monitor-open .h3p-head{height:50px}.iamccs-h3pro:fullscreen .h3p-main,.iamccs-h3pro.h3p-monitor-open .h3p-main{padding:16px}.iamccs-h3pro:fullscreen .h3p-grid,.iamccs-h3pro.h3p-monitor-open .h3p-grid{grid-template-columns:repeat(3,minmax(0,1fr))}.iamccs-h3pro:fullscreen .h3p-functional-grid,.iamccs-h3pro.h3p-monitor-open .h3p-functional-grid{grid-template-columns:repeat(3,minmax(0,1fr))}.h3p-monitor-host{position:fixed;inset:0;z-index:2147483640;background:#05080c;display:block;overflow:hidden}.h3p-monitor-host>.iamccs-h3pro{position:absolute;inset:0}.h3p-monitor-host .h3p-open-editor{border-color:#efbf6d;background:#6b4518}@media(max-width:1050px){.iamccs-h3pro:fullscreen .h3p-layout,.iamccs-h3pro.h3p-monitor-open .h3p-layout{grid-template-columns:145px minmax(0,1fr) 260px}.iamccs-h3pro:fullscreen .h3p-grid,.iamccs-h3pro.h3p-monitor-open .h3p-grid,.iamccs-h3pro:fullscreen .h3p-functional-grid,.iamccs-h3pro.h3p-monitor-open .h3p-functional-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
  /* IAMCCS original Settings palette: presentation only. */
  .iamccs-h3pro{background:linear-gradient(145deg,#192124,#101719 62%,#211A11);border-color:#A97C38;border-radius:3px;color:#F4F0E8;font-family:Arial,sans-serif;box-shadow:inset 0 1px 0 #ffffff12}
  .iamccs-h3pro .h3p-head{border-bottom-color:#6D5630}
  .iamccs-h3pro .h3p-title{font-family:Arial,sans-serif;color:#F5D18A;font-weight:800;letter-spacing:.04em}
  .iamccs-h3pro .h3p-mark{border-radius:3px;background:#5D3C14;border-color:#D7A74F;color:#FFF0C5}
  .iamccs-h3pro .h3p-sub{color:#AEBBBC}
  .iamccs-h3pro .h3p-rail{background:linear-gradient(160deg,#192B2B,#11191B);border-color:#526469;border-radius:3px}
  .iamccs-h3pro .h3p-main{background:linear-gradient(145deg,#172326,#11191B);border-color:#526469;border-radius:3px}
  .iamccs-h3pro .h3p-truth{background:linear-gradient(145deg,#132522,#11191B);border-color:#4E8C70;border-radius:3px}
  .iamccs-h3pro .h3p-tab{border-radius:3px;background:#202B2E;border-color:#587177;color:#D8E4E1;box-shadow:none}
  .iamccs-h3pro .h3p-tab:hover{background:#2B3C3D;border-color:#AEBBBC;color:#F4F0E8}
  .iamccs-h3pro .h3p-tab.active{background:linear-gradient(100deg,#5D3C14,#302A1D);border-color:#D7A74F;border-left-color:#F2C671;color:#FFF0C5;box-shadow:inset 2px 0 #F2C671}
  .iamccs-h3pro .h3p-mode-folder{background:#241E30;border-color:#795B9C;color:#DBBEEE}
  .iamccs-h3pro .h3p-mode-folder.open,.iamccs-h3pro .h3p-mode-folder.context-active{background:#352747;border-color:#B79ADB;color:#F3E8FF;box-shadow:inset 3px 0 #B79ADB}
  .iamccs-h3pro .h3p-subtab{background:#201E29;border-color:#604B78;color:#C9BBD7}
  .iamccs-h3pro .h3p-subtab.active{background:#45325F;border-color:#B79ADB;color:#F3E8FF;box-shadow:none}
  .iamccs-h3pro .h3p-section-title,.iamccs-h3pro .h3p-truth h3{font-family:Arial,sans-serif;color:#F5D18A}
  .iamccs-h3pro .h3p-section-note{color:#AEBBBC}
  .iamccs-h3pro .h3p-functional{border-radius:3px;background:#152C38;border-color:#477C9C}
  .iamccs-h3pro .h3p-functional-title{color:#EAC77F}
  .iamccs-h3pro .h3p-field{border-radius:3px;background:#172326;border-color:#46595D}
  .iamccs-h3pro .h3p-field label{color:#C5D3CF}
  .iamccs-h3pro .h3p-field small{color:#9BAEAE}
  .iamccs-h3pro .h3p-field input,.iamccs-h3pro .h3p-field select,.iamccs-h3pro .h3p-resolution{border-radius:2px;background:#0F1719;border-color:#64767A;color:#F1F4EE}
  .iamccs-h3pro .h3p-field input:focus-visible,.iamccs-h3pro .h3p-field select:focus-visible{outline:2px solid #EAB86E;outline-offset:1px}
  .iamccs-h3pro .h3p-field input[type=checkbox]{accent-color:#D7A74F}
  .iamccs-h3pro .h3p-field.muted{background:#11191B;border-color:#46595D}
  .iamccs-h3pro .h3p-recipe,.iamccs-h3pro .h3p-choice{border-radius:3px;background:#202B2E;border-color:#587177;color:#D8E4E1;box-shadow:none}
  .iamccs-h3pro .h3p-recipe:not(:disabled):hover,.iamccs-h3pro .h3p-choice:not(:disabled):hover{background:#364137;border-color:#EAB86E;color:#FFF0C5;box-shadow:inset 0 1px #ffffff15}
  .iamccs-h3pro .h3p-recipe.active,.iamccs-h3pro .h3p-choice.active{background:#5D3C14;border-color:#D7A74F;color:#FFF0C5;box-shadow:inset 0 1px #ffffff22}
  .iamccs-h3pro .h3p-recipe:disabled{background:#172124;border-color:#46595D;color:#94A3A0;box-shadow:none}
  .iamccs-h3pro .h3p-open-editor{border-radius:3px;border-color:#D7A74F;background:#5D3C14;color:#FFF0C5}
  .iamccs-h3pro .h3p-open-editor:hover{background:#825A24;border-color:#F2C671;color:#FFF0C5}
  .iamccs-h3pro .h3p-owner{background:#132522;border-radius:3px;color:#AFC1C1}
  .iamccs-h3pro .h3p-owner strong{color:#EAC77F}
  .iamccs-h3pro .h3p-truth-row{border-bottom-color:#354B43}
  .iamccs-h3pro .h3p-truth-row span{color:#AFC1C1}
  .iamccs-h3pro .h3p-truth-row b{color:#D6F6E4}
  .iamccs-h3pro .h3p-memory-recipes,.iamccs-h3pro .h3p-guide-window-recipes,.iamccs-h3pro .h3p-attention-recipes{border-radius:3px;border-color:#477C9C;background:#152C38}
  .iamccs-h3pro .h3p-speed-recipes,.iamccs-h3pro .h3p-turbo-recipes{border-radius:3px;border-color:#795B9C;background:#241E30}
  .iamccs-h3pro .h3p-delivery-recipes,.iamccs-h3pro .h3p-face-refine-preset-group{border-radius:3px;border-color:#4E8C70;background:#17352D}
  .iamccs-h3pro .h3p-face-refine-group-note,.iamccs-h3pro .h3p-face-refine-help{color:#AFC1C1}
  .iamccs-h3pro .h3p-asset,.iamccs-h3pro .h3p-asset-actions button{border-radius:3px}
  /* End original Settings palette. */

  </style>
  <div class="h3p-head"><span class="h3p-mark">IAMCCS PRO</span><div><div class="h3p-title">H3 Settings PRO</div><div class="h3p-sub">Render compiler · one queue truth · no editorial duplication</div></div><div class="h3p-mode"><span class="h3p-sub">SHOTBOARD MODE</span><b data-mode>NOT CONNECTED</b></div><button type="button" class="h3p-open-editor" data-open-editor>OPEN EDITOR</button></div>
  <div class="h3p-layout"><nav class="h3p-rail"></nav><main class="h3p-main"><div class="h3p-section-title"></div><div class="h3p-section-note"></div><div class="h3p-context"></div><div class="h3p-recipes"></div><div class="h3p-grid" data-grid></div></main><aside class="h3p-truth"><h3>Queue Truth</h3><div class="h3p-truth-list"></div><div class="h3p-health"></div></aside></div>`;
  const q = (selector) => root.querySelector(selector); let active = String(node.properties?.iamccs_h3_settings_pro_active_section || "assistant"), lastMode = "";
  const openEditor = q("[data-open-editor]");
  let monitorHost = null, monitorPlaceholder = null;
  let editorRequestedOpen = false;
  const monitorOpen = () => document.fullscreenElement === root || root.classList.contains("h3p-monitor-open");
  const syncOpenEditor = () => {
    const open = editorRequestedOpen || monitorOpen();
    openEditor.textContent = open ? "CLOSE EDITOR" : "OPEN EDITOR";
    openEditor.title = open ? "Return IAMCCS Settings PRO to the node" : "Open IAMCCS Settings PRO across the complete monitor";
    openEditor.setAttribute("aria-pressed", String(open));
  };
  const closeMonitorFallback = () => {
    if (!root.classList.contains("h3p-monitor-open")) return;
    root.classList.remove("h3p-monitor-open");
    if (monitorPlaceholder?.parentNode) monitorPlaceholder.parentNode.insertBefore(root, monitorPlaceholder);
    monitorPlaceholder?.remove(); monitorHost?.remove();
    monitorPlaceholder = null; monitorHost = null;
    syncOpenEditor();
  };
  const openMonitorFallback = () => {
    if (root.classList.contains("h3p-monitor-open")) return;
    monitorPlaceholder = document.createComment("IAMCCS Settings PRO monitor placeholder");
    root.parentNode?.insertBefore(monitorPlaceholder, root);
    monitorHost = document.createElement("div"); monitorHost.className = "h3p-monitor-host";
    document.body.appendChild(monitorHost); monitorHost.appendChild(root);
    root.classList.add("h3p-monitor-open"); syncOpenEditor();
  };
  const ensureEditorOpen = () => {
    if (!editorRequestedOpen || monitorOpen()) { syncOpenEditor(); return; }
    // Browser confirmation dialogs may dismiss native fullscreen. Preserve the
    // user's editor intent with the full-monitor overlay until CLOSE EDITOR.
    openMonitorFallback();
  };
  const preserveEditorAfterRefresh = () => {
    if (!editorRequestedOpen) { syncOpenEditor(); return; }
    window.setTimeout(ensureEditorOpen, 0);
  };
  const closeSettingsEditor = async () => {
    editorRequestedOpen = false;
    if (document.fullscreenElement === root) {
      try { await document.exitFullscreen(); } catch {}
    }
    closeMonitorFallback(); syncOpenEditor();
  };
  const toggleSettingsEditor = async () => {
    if (editorRequestedOpen || monitorOpen()) { await closeSettingsEditor(); return; }
    editorRequestedOpen = true;
    try {
      await root.requestFullscreen({ navigationUI: "hide" });
    } catch {
      openMonitorFallback();
    }
    ensureEditorOpen();
  };
  const onSettingsFullscreenChange = () => {
    if (editorRequestedOpen && !monitorOpen()) window.setTimeout(ensureEditorOpen, 0);
    else syncOpenEditor();
  };
  openEditor.onclick = (event) => { event.preventDefault(); event.stopPropagation(); void toggleSettingsEditor(); };
  document.addEventListener("fullscreenchange", onSettingsFullscreenChange);
  syncOpenEditor();
  // Canvas wheel handlers must not consume the panel's native scrolling.
  root.addEventListener("wheel", (event) => event.stopPropagation(), { passive: true });
  const scrollStyle = document.createElement("style");
  scrollStyle.textContent = ".iamccs-h3pro{height:680px!important}.iamccs-h3pro .h3p-layout{min-height:0;overflow:hidden}.iamccs-h3pro .h3p-main,.iamccs-h3pro .h3p-truth{overflow-y:auto;min-height:0;overscroll-behavior:contain;pointer-events:auto}.iamccs-h3pro .h3p-truth-row b{overflow-wrap:anywhere}";
  root.append(scrollStyle);
  // IAMCCS MODE ASSISTANT HARD SCROLL FIX V2.2
  const assistantHardScrollStyle = document.createElement("style");
  assistantHardScrollStyle.textContent = `
    .iamccs-h3pro .h3p-layout{
      min-height:0!important;
      overflow:hidden!important;
      grid-template-rows:minmax(0,1fr)!important;
    }
    .iamccs-h3pro .h3p-main{
      min-height:0!important;
      height:100%!important;
      max-height:100%!important;
    }
    .iamccs-h3pro .h3p-main[data-section="assistant"]{
      overflow-y:auto!important;
      overflow-x:hidden!important;
      overscroll-behavior:contain!important;
      scrollbar-gutter:stable both-edges;
      touch-action:pan-y!important;
    }
    .iamccs-h3pro .h3p-main[data-section="assistant"] .h3p-flow,
    .iamccs-h3pro .h3p-main[data-section="assistant"] [data-grid]{
      overflow:visible!important;
      min-height:max-content!important;
    }
  `;
  root.append(assistantHardScrollStyle);

  // ComfyUI/LiteGraph can steal wheel events from DOM widgets.  In Assistant
  // mode we own the wheel explicitly and scroll the central pane ourselves.
  root.addEventListener("wheel", (event) => {
    const main = q(".h3p-main");
    if (!main || main.dataset.section !== "assistant") return;
    const target = event.target;
    if (!(target instanceof Element) || !main.contains(target)) return;

    const maxScroll = Math.max(0, main.scrollHeight - main.clientHeight);
    if (maxScroll <= 0) return;

    const before = main.scrollTop;
    let delta = Number(event.deltaY || 0);
    if (event.deltaMode === 1) delta *= 34;
    else if (event.deltaMode === 2) delta *= Math.max(120, main.clientHeight * 0.9);
    if (!Number.isFinite(delta) || delta === 0) return;

    main.scrollTop = Math.max(0, Math.min(maxScroll, before + delta));
    if (main.scrollTop !== before || (delta < 0 && before > 0) || (delta > 0 && before < maxScroll)) {
      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation?.();
    }
  }, { capture:true, passive:false });
  // IAMCCS MODE ASSISTANT SCROLL FIX V2.1
  const assistantScrollStyle = document.createElement("style");
  assistantScrollStyle.textContent = `
    .iamccs-h3pro .h3p-layout{min-height:0!important;overflow:hidden!important;}
    .iamccs-h3pro .h3p-main[data-section="assistant"]{
      min-height:0!important;
      height:100%!important;
      max-height:100%!important;
      overflow-y:scroll!important;
      overflow-x:hidden!important;
      overscroll-behavior:contain!important;
      scrollbar-gutter:stable;
      touch-action:pan-y;
    }
    .iamccs-h3pro .h3p-main[data-section="assistant"] > .h3p-grid[data-grid]{
      min-height:max-content;
      overflow:visible;
    }
  `;
  root.append(assistantScrollStyle);
  const assistantMain = q(".h3p-main");
  assistantMain?.addEventListener("wheel", (event) => event.stopPropagation(), { passive: true });
  function fieldRelevant(name, mode) {
    if (["ltx_looper_guiding_strength", "ltx_looper_overlap_strength", "ltx_looper_cond_image_strength"].includes(name)) return false;
    const value = String(mode).toLowerCase();
    const endpoint = String(widget(node, "longvid_terminal_endpoint_mode")?.value || "hard_image").toLowerCase();
    const hd = endpoint === "pianosequenza_hd";
    if (name.startsWith("h3_pixel_tiled_")) {
      if (String(widget(node, "upscale_mode")?.value || "off") !== "pixel_tiled_low_vram") return false;
      if (name !== "h3_pixel_tiled_method" && String(widget(node, "h3_pixel_tiled_method")?.value || "cpu_lanczos") !== "model_tiled") return false;
    }
    if (["longvid_terminal_endpoint_mode", "longvid_pianosequenza_2stage_enabled", "longvid_pianosequenza_stage_profile"].includes(name) && value !== "longvid_guides") return false;
    if (name === "longvid_pianosequenza_stage_profile" && (hd || !Boolean(widget(node, "longvid_pianosequenza_2stage_enabled")?.value))) return false;
    if (name === "longvid_guide_window_policy" && value !== "longvid_guides") return false;
    if (name === "keyframe_joint_latent_new" && value !== "keyframe_joint_native") return false;
    if (["upscale_link_to_native", "upscale_link_factor"].includes(name) && (value !== "longvid_guides" || hd || !Boolean(widget(node, "longvid_pianosequenza_2stage_enabled")?.value))) return false;
    if (name.startsWith("v2v_") && !value.startsWith("v2va_")) return false;
    if (name.startsWith("extended_av_") && value !== "fl2va_extended_av" && !(value === "longvid_keyframe_continuous" && String(widget(node,"longvid_continuous_engine")?.value || "") === "extended_av_continuous")) return false;
    if (name === "flf_continuity_tail_frames" && value === "longvid_keyframe_continuous" && String(widget(node,"longvid_continuous_engine")?.value || "") === "extended_av_continuous") return false;
    if (["extended_av_custom_window_frames", "extended_av_custom_overlap_frames"].includes(name) && String(widget(node, "extended_av_profile")?.value || "safe_8_12gb") !== "custom") return false;
    if (name.startsWith("h3_auto_extend_") && value === "fl2va_extended_av") return false;
    const continuityFamily = value.includes("fl2") || value.includes("longvid") || value.includes("guided_av_loop") || value === "keyframe_joint_native" || value === "latent_go_ahead" || value === "pianosequenza_true_long";
    if (name.startsWith("flf_") && !continuityFamily) return false;
    if (name.startsWith("h3_r40_") && !value.includes("scout")) return false; return true;
  }
  function visibleGroups(mode) {
    // MODE ASSISTANT exposes one dedicated mode engine at a time. Universal
    // utility engines (Continuation, RefMod, Face Refine, Scout) remain
    // discoverable. DIRECT stays visible solely as a compatibility surface.
    const targetEngine = modeEngineGroup(node, assistantModeKey(node, mode));
    return GROUPS.filter((group) => {
      if (group.id === "engine_reference_image") return false;
      if (group.id === "continuity_guard") return true;
      if (group.id === "ahead" && mode !== "latent_go_ahead") return false;
      if (group.modeEngine && group.id !== targetEngine) return false;
      return ["continuation", "refmod"].includes(group.id) || group.assistant || group.dynamic || group.fields.some((name) => widget(node, name));
    });
  }
  function fieldNames(group, mode) {
    if (!group.dynamic) return group.fields.filter((name) => widget(node, name) && fieldRelevant(name, mode));
    return (node.widgets || []).map((item) => item?.name).filter((name) => name && !ASSIGNED.has(name) && !SHOTBOARD_OWNED.has(name) && !INTERNAL.has(name) && !COMPATIBILITY_ONLY.has(name) && fieldRelevant(name, mode));
  }
  function makeControl(name) {
    const item = widget(node, name); if (!item || (name.startsWith("h3_guard_") && COMPATIBILITY_ONLY.has(name))) return null;
    const box = document.createElement("div"); box.className = "h3p-field";
    if (name === "secondary_lora_continuation_only") box.style.gridColumn = "1 / -1";
    const label = document.createElement("label"); label.textContent = name === "secondary_lora_continuation_only" ? "Second LoRA: from segment 2" : human(name); box.append(label);
    let control; const options = name === "h3_guard_mode" ? ["off", "monitor"] : choices(item);
    if (options.length) {
      control = document.createElement("select"); options.forEach((value) => control.add(new Option(
        name === "h3_fast_latent_stage2_guides"
          ? (String(value) === "masked" ? "MASKED · pin image guide tokens" : "SOFT · guide conditioning")
          : friendly(value), String(value)))); control.value = String(item.value ?? "");
    } else if (typeof item.value === "boolean") {
      control = document.createElement("input"); control.type = "checkbox"; control.checked = Boolean(item.value);
    } else {
      control = document.createElement("input"); control.type = typeof item.value === "number" ? "number" : "text"; control.value = item.value ?? "";
      if (control.type === "number") { if (Number.isFinite(item.options?.min)) control.min = item.options.min; if (Number.isFinite(item.options?.max)) control.max = item.options.max; control.step = item.options?.step ?? "any"; }
    }
    if (name.startsWith("h3_guard_")) {
      const supported = assistantModeKey(node) === "fl2va_extended_av";
      const mode = String(widget(node, "h3_guard_mode")?.value || "off");
      if (name === "h3_guard_mode" && mode === "repair") {
        control.value = "off";
        control.title = "Saved Repair mode is retired and runs as Off. Select Monitor for diagnostics.";
        const notice = document.createElement("small");
        notice.textContent = "Saved Repair mode retired - effective mode: Off.";
        box.append(notice);
      }
      control.disabled = !supported || (name !== "h3_guard_mode" && mode !== "monitor");
      if (!supported) control.title = "Continuity Guard currently supports Extended AV only.";
    }
    const gate = branchGate(node, name);
    if (!gate.available) {
      control.disabled = true;
      box.classList.add("muted");
      box.title = gate.reason;
    }
    if ((name === "h3_attention_backend" && integratedAttention(node)) || (name.startsWith("h3_veda_") && !vedaAvailable(node))) {
      control.disabled = true; box.classList.add("muted");
      box.title = name === "h3_attention_backend" ? `Managed by ${integratedAttention(node)}` : "VEDA is unavailable with Sol, SLA or Fused attention.";
    }
    if (omniLockedField(node, name)) {
      control.disabled = true; box.classList.add("muted");
      box.title = "FastH3 Omni owns its 8-step recipe. Select another generation mode to use other speed adapters.";
    }
    if (name === "h3_attention_backend" && omniActive(node)) {
      for (const option of control.options) option.disabled = ["sol_low_vram", "sage_sol"].includes(option.value);
    }
    control.dataset.field = name; control.onchange = () => {
      const value = control.type === "checkbox" ? control.checked : control.type === "number" ? Number(control.value) : control.value;
      if (name === "duration_seconds") setShotboardDuration(node, value);
      else if (name === "audio_mode") setShotboardAudio(node, value);
      else if (name === "h3_open_refine_enabled") {
        if (value) applyRecipe(node, "open-h3-refine");
        else setValue(node, name, false, false);
      }
      else if (name === "upscale_enabled") selectDeliveryMode(node,value ? (String(widget(node,"upscale_mode")?.value || "off") !== "off" ? widget(node,"upscale_mode").value : node.properties?.iamccs_delivery_route || "h3_ultimate_tiled") : "off");
      else if (name === "upscale_mode") selectDeliveryMode(node,String(value));
      else if (["upscale_width","upscale_height"].includes(name)) {
        setValue(node,name,value,false); normalizeDeliveryResolution(node);
        document.dispatchEvent(new CustomEvent("iamccs:h3-settings-changed",{detail:{source_node_id:node.id,field:name}}));
      }
      else if (name === "h3_faceswap_generate_new_audio") {
        setValue(node, name, value, false);
        // Keep the generic AUDIO panel legible, while the Face Swap backend
        // still enforces this explicit switch as its final route authority.
        setShotboardAudio(node, value ? "h3_native_generated" : "h3_custom_audio_drive");
      }
      else if (name === "performance_profile") applyMemoryProfileChoice(node, value);
      else if (name === "acceleration") applyAccelerationChoice(node, value);
      else if (name === "h3_attention_backend") setAttentionChoice(node, value);
      else if (name === "h3_veda_enabled") setVedaEnabled(node, value);
      else setValue(node, name, value);
      if (name === "fused_turbo_model_name" && selectedAcceleration(node) === "fused") applyRecipe(node, "fused");
      if (name === "fused_turbo_sigma_preset" && selectedAcceleration(node) === "fused") normalizeSpeedPresetContract(node);
      if (name === "turbo_mode" || name === "turbo_lora_name") applyDeclaredTurboContract(node);
      refresh();
    }; box.append(control);
    const guidance = name === "h3_controlnet_preprocess_resolution"
      ? "Choose 768 to match the working DWPose example. NODE DEFAULT keeps the Cine H3 Fun Control Input resolution (512 in existing smoke workflows). This setting applies only when preprocessing source_video; an already-preprocessed control_video bypasses it."
      : name === "h3_controlnet_dwpose_torch_gpu"
      ? "ON: use installed TorchScript YOLOX + DWPose on CUDA when this workflow preprocesses source_video through Cine H3 Fun Control Input. OFF: preserve the existing automatic preprocessor route. Preprocessed control_video bypasses this toggle."
      : name === "h3_controlnet_use_video_audio"
      ? "OFF (default): use Shotboard AudioBoard. ON: use the connected control video's audio when Custom Audio Drive is selected; if the video has no audio, AudioBoard remains the fallback. Native Generated audio is unchanged."
      : name === "performance_profile"
      ? "PRESET · changes the H3 window, exact-attention rows, ClipProj loading and pre-decode cleanup together. It never selects an accelerator."
      : name === "acceleration"
        ? "ENGINE · changes the attention/sampling implementation. PDD, FastH3, SLA and Fused also apply their required steps, sampler and installed asset as one atomic recipe."
        : item.options?.tooltip;
    const hint = document.createElement("small"); hint.textContent = gate.available
      ? String(guidance || "Saved here and compiled at Queue.").replace(/RTX\s*30\d0/gi, "the selected VRAM tier")
      : `MUTED · ${gate.reason}`;
    box.append(hint); return box;
  }
  function warnings(mode) {
    const issues = [], acceleration = String(widget(node, "acceleration")?.value || "native"), turbo = String(widget(node, "turbo_mode")?.value || "off");
    if (HIDDEN_MODES.has(mode)) issues.push(["warn", "A legacy mode is loaded. Select a public mode for new work."]);
    const turboName = String(widget(node, "turbo_lora_name")?.value || ""), fused = String(widget(node, "fused_turbo_model_name")?.value || ""), controlModel = String(widget(node, "h3_controlnet_name")?.value || "");
    if (!linkedShotboard(node)) issues.push(["warn", "Connect Settings PRO to Cine H3 Input, then Cine H3 Input to MiniMax H3 Shotboard."]);
    if (node.properties?.iamccs_auto_from_shotboard_active) {
      const snapshot = node.properties?.iamccs_auto_import_snapshot || {};
      issues.push(["ok", `AUTO imported ${Number(snapshot.imported_count || 0)} technical settings from Shotboard. Settings PRO is now the settings master; this is a one-time snapshot, not live two-way sync.`]);
    }
    if (acceleration === "matlowai_fused_turbo_manual_sigma" && !fused) issues.push(["error", "Fused Fast is enabled but no fused diffusion model is selected."]);
    if (acceleration === "matlowai_fused_turbo_manual_sigma" && turbo !== "off") issues.push(["error", "Fused Fast and Turbo LoRA cannot be active together."]);
    if (modeContext(mode) === "control" && !controlModel) issues.push(["error", "ControlNet mode requires an installed H3 Fun ControlNet model."]);
    if (Boolean(widget(node, "h3_controlnet_enabled")?.value) && !branchGate(node, "h3_controlnet_enabled").available) issues.push(["error", "H3 Fun ControlNet is enabled, but its input branch is not connected."]);
    if (Boolean(widget(node, "face_detailer_enabled")?.value) && !branchGate(node, "face_detailer_enabled").available) issues.push(["error", "Face Detailer is enabled, but its optional branch is not connected."]);
    const upscaleMode = String(widget(node, "upscale_mode")?.value || "off").toLowerCase();
    const externalContinuation = Boolean(widget(node,"h3_continuation_enabled")?.value);
    if (externalContinuation && !String(widget(node,"h3_continuation_checkpoint")?.value || "").trim()) issues.push(["error", "START FROM SAVED AV is enabled, but no continuation checkpoint is selected."]);
    if (externalContinuation && /ref2v|lipsync_ref2vid/.test(String(mode || "").toLowerCase())) issues.push(["error", "External saved-AV continuation is FL2VA-family only; REF2VA/Ref2Vid LipSync cannot share that denoiser branch."]);
    if (externalContinuation && upscaleMode === "h3_fast_latent_2pass" && Boolean(widget(node,"upscale_enabled")?.value)) issues.push(["error", "External saved-AV continuation cannot use Fast Latent 2-Pass yet because the saved checkpoint has the native latent grid. Use Native or Pixel Safe delivery."]);
    if (Boolean(widget(node, "upscale_enabled")?.value)) {
      const field = upscaleMode.includes("ltx") ? "ltx_seam_safe" : upscaleMode.includes("wan") ? "wan_upscale_denoise" : upscaleMode.startsWith("h3_") ? "h3_upres_model_name" : "";
      if (field && !branchGate(node, field).available) issues.push(["error", `${friendly(upscaleMode)} is enabled, but its delivery branch is not connected.`]);
      if (upscaleMode === "h3_fast_latent_2pass") issues.push(["warn", "Learned 3D lift saves Stage-1 render time, not Stage-2 VRAM. The target-resolution H3 refine still runs; on 12 GB start with a short 124F clip and a modest target canvas. H3 2-stage SAFE DELIVERY skips the HIGH H3 forward."]);
    }
    if (modeContext(mode) === "face" && !String(widget(node, "h3_faceswap_sam_model")?.value || "")) issues.push(["error", "SAM3 Subject Swap requires an installed SAM3 checkpoint, unless a complete source mask is connected to the swap input."]);
    if (modeContext(mode) === "face") {
      const generatesAudio = Boolean(widget(node, "h3_faceswap_generate_new_audio")?.value);
      const swapInput = connectedNodes(node).find((candidate) => /H3FaceSwapInput/i.test(nodeClass(candidate)));
      const sourceAudio = (swapInput?.inputs || []).find((input) => String(input?.name || "") === "source_audio");
      if (!generatesAudio && sourceAudio?.link == null) issues.push(["error", "SOURCE VIDEO AUDIO is the Face Swap default, but source_audio is not connected. Connect the Load Video audio output, or explicitly enable FACE SWAP · GENERATE NEW AUDIO."]);
      if (generatesAudio) issues.push(["warn", "FACE SWAP · GENERATE NEW AUDIO is ON: source-video audio is intentionally ignored and H3 creates a new audio stream."]);
    }
    issues.push(...selectedH3AssetCompatibility(node, mode));
    const longvidGuidesMode = String(mode || "").toLowerCase() === "longvid_guides";
    const endpoint = String(widget(node, "longvid_terminal_endpoint_mode")?.value || "hard_image").toLowerCase();
    if (longvidGuidesMode && endpoint === "pianosequenza_hd" && Boolean(widget(node, "longvid_pianosequenza_2stage_enabled")?.value)) {
      issues.push(["warn", "PIANOSEQUENZA HD owns its fixed two-stage contract; the separate Multi-Stage Spatial toggle is preserved but ignored while HD is selected."]);
    }
    if (longvidGuidesMode
        && Boolean(widget(node, "longvid_pianosequenza_2stage_enabled")?.value)
        && String(widget(node, "longvid_pianosequenza_stage_profile")?.value || "2_stage_full") === "2_stage_full"
        && ["vram8", "vram12"].includes(selectedMemory(node))
        && Number(widget(node, "width")?.value || 0) * Number(widget(node, "height")?.value || 0) >= 1280 * 768) {
      issues.push(["warn", "12 GB OOM risk: the learned LOW→HIGH lift is not the expensive part; the HIGH H3 forward at 1280×768+ is. Reduce the HIGH canvas/window or choose 2 STAGE SAFE DELIVERY (lift only, no HIGH H3 refinement)."]);
    }
    if (String(mode) === "longvid_guides") {
      const pixels = Number(widget(node,"width")?.value || 0) * Number(widget(node,"height")?.value || 0);
      if (pixels > 640 * 384 && selectedGuideWindowStrategy(node) === "guide_macro_362") issues.push(["warn", "High-resolution LongVid is still using the 362F macro window. Use 209F+22F or 124F+22F Adaptive Guide Windows for continuity-first testing."]);
    }
    if (String(widget(node, "v2v_source_end_policy")?.value || "") === "hold_last_visible") {
      issues.push(["warn", "V2V short-source tolerance is active: missing visible video is held on the final source frame and missing source audio is padded with silence. Lip-sync cannot continue beyond the real source end."]);
    }
    const memoryContracts = {
      vram8: [124,2048], vram12: [209,4096], vram16: [294,8192], vram24: [362,16384],
    };
    const expected = memoryContracts[selectedMemory(node)];
    if (expected && (Number(widget(node,"motion_context_window_frames")?.value)!==expected[0] || Number(widget(node,"h3_exact_chunk_rows")?.value)!==expected[1])) issues.push(["warn", `VRAM preset is partially overridden. Reapply it to restore ${expected[0]} frames / ${expected[1]} attention rows.`]);
    if (!issues.length) issues.push(["ok", "Configuration is coherent. Final model-family and asset checks run before sampling."]); return issues;
  }
  function renderRail(mode) {
    const rail = q(".h3p-rail");
    rail.replaceChildren();
    const groups = visibleGroups(mode);
    if (!groups.some((group) => group.id === active)) active = "assistant";
    node.properties ||= {};
    const engineChildren = groups.filter((group) => ENGINE_ID_SET.has(group.id));
    const regular = groups.filter((group) => !ENGINE_ID_SET.has(group.id));
    const engineContextActive = ENGINE_ID_SET.has(active)
      || String(mode || "").toLowerCase() === "fl2va_extended_av"
      || Boolean(widget(node, "h3_continuation_enabled")?.value)
      || Boolean(widget(node, "h3_refmod_enabled")?.value);
    const enginesStateKnown = Object.prototype.hasOwnProperty.call(node.properties, "iamccs_h3_settings_pro_engines_open");
    const legacyFolderStateKnown = Object.prototype.hasOwnProperty.call(node.properties, "iamccs_h3_settings_pro_mode_controls_open");
    let folderOpen = enginesStateKnown
      ? Boolean(node.properties.iamccs_h3_settings_pro_engines_open)
      : legacyFolderStateKnown
        ? Boolean(node.properties.iamccs_h3_settings_pro_mode_controls_open)
        : ENGINE_ID_SET.has(active);

    const appendGroupButton = (group, extraClass = "") => {
      const button = document.createElement("button");
      button.className = `h3p-tab${extraClass ? ` ${extraClass}` : ""}${active === group.id ? " active" : ""}`;
      button.textContent = group.label;
      button.onclick = () => {
        active = group.id;
        node.properties.iamccs_h3_settings_pro_active_section = active;
        if (ENGINE_ID_SET.has(group.id)) {
          node.properties.iamccs_h3_settings_pro_engines_open = true;
          node.properties.iamccs_h3_settings_pro_mode_controls_open = true; // legacy state mirror
        }
        app.graph?.change?.();
        refresh();
      };
      rail.append(button);
    };

    for (const group of regular) {
      appendGroupButton(group);
      // ENGINES belongs after the numbered production stack. It collects
      // every mode-specific/non-numbered subsystem instead of presenting
      // them as if they were mandatory stages of the base H3 pipeline.
      if (group.id !== "finish" || !engineChildren.length) continue;
      const folder = document.createElement("button");
      folder.className = `h3p-tab h3p-mode-folder h3p-engines-folder${folderOpen ? " open" : ""}${engineContextActive ? " context-active" : ""}`;
      folder.innerHTML = `ENGINES <span class="h3p-folder-arrow">${folderOpen ? "▾" : "▸"}</span>`;
      folder.title = "Mode engines: the selected MODE ASSISTANT engine plus Continuation / RefMod / Face Refine / Scout. DIRECT remains legacy compatibility.";
      folder.onclick = () => {
        const next = !folderOpen;
        node.properties.iamccs_h3_settings_pro_engines_open = next;
        node.properties.iamccs_h3_settings_pro_mode_controls_open = next; // compatibility
        app.graph?.change?.();
        refresh();
      };
      rail.append(folder);
      if (folderOpen) engineChildren.forEach((child) => appendGroupButton(child, "h3p-subtab h3p-engine-subtab"));
    }

    const owner = document.createElement("div");
    owner.className = "h3p-owner";
    owner.innerHTML = "<strong>OWNERSHIP LOCK</strong>Shotboard: prompts, guides, images/video, FPS and audio lanes. Duration is Shotboard truth alone; when Settings PRO is connected as master, its duration is synced into Shotboard and becomes generation truth.<br><br>Settings PRO: generation mode, duration when master, canvas, sampling, memory, acceleration, continuity and delivery. AUTO imports once; it is not a live two-way sync.";
    rail.append(owner);
  }
  function recipeGroup(parent) {
    // Face Refinement preset bars also use h3p-memory-recipes for styling.
    // Resolve the specific face group FIRST so Quality / Face Size / Definition
    // keep independent active states instead of all collapsing into "memory".
    if (parent?.classList?.contains("h3p-face-refine-quality-recipes")) return "face_refine_quality";
    if (parent?.classList?.contains("h3p-face-refine-size-recipes")) return "face_refine_size";
    if (parent?.classList?.contains("h3p-face-refine-definition-recipes")) return "face_refine_definition";
    if (parent?.classList?.contains("h3p-memory-recipes")) return "memory";
    if (parent?.classList?.contains("h3p-attention-recipes")) return "attention";
    if (parent?.classList?.contains("h3p-speed-recipes")) return "speed";
    if (parent?.classList?.contains("h3p-turbo-recipes")) return "speed";
    if (parent?.classList?.contains("h3p-delivery-recipes")) return "delivery";
    if (parent?.classList?.contains("h3p-guide-window-recipes")) return "guide_window";
    return `section:${active}`;
  }
  function rememberedRecipe(group) { return String(node.properties?.iamccs_recipe_highlight?.[group] || ""); }
  function rememberRecipe(group, id) {
    node.properties ||= {}; node.properties.iamccs_recipe_highlight ||= {};
    node.properties.iamccs_recipe_highlight[group] = String(id);
    app.graph?.change?.();
  }
  function addRecipe(parent, label, id, enabled = true, selected = false, handler = null) {
    if (omniLockedRecipe(node, id)) enabled = false;
    const group = recipeGroup(parent);
    const button = document.createElement("button");
    const backendState = ["speed", "memory", "delivery", "turbo", "attention"].includes(group) || ["speed", "memory", "finish"].includes(active);
    const remembered = !backendState && rememberedRecipe(group) === String(id);
    if (group === "speed" && !String(id).startsWith("turbo:")) selected = selectedAcceleration(node) === id;
    button.className = `h3p-recipe${(selected || remembered) ? " active" : ""}`;
    button.textContent = id === "fused" ? "FUSED · 4 STEP · CHECKPOINT COMPATIBILITY" : label;
    button.disabled = !enabled;
    if (omniLockedRecipe(node, id)) button.title = "Unavailable in FastH3 Omni. Select a standard H3 mode to use this recipe.";
    button.onclick = () => {
      if (!enabled) return;
      rememberRecipe(group, id);
      const applied = (handler || (() => applyRecipe(node, id)))();
      if (applied === false) return;
      if (!backendState) parent.querySelectorAll?.(".h3p-recipe").forEach((entry) => entry.classList.toggle("active", entry === button));
      node._iamccsSettingsProRefresh?.();
    };
    parent.append(button);
  }
  function renderAssistant(mode) {
    const grid = q("[data-grid]"); grid.className = "h3p-flow"; grid.replaceChildren();
    const ask = (text) => { const el = document.createElement("div"); el.className = "h3p-question"; el.textContent = text; grid.append(el); };
    ask("1 · WHAT DO YOU WANT TO CREATE?"); const modes = document.createElement("div"); modes.className = "h3p-choice-grid";
    const selectedMode = assistantModeKey(node, mode);
    const family=value=>value==='reference_image'?'REFERENCES / IDENTITY':['fl2va_extended_av','t2va_continuous'].includes(value)?'EXTENSION / LONG TAKE':value.startsWith('t2')?'TEXT / CREATE':value.startsWith('ref2')?'REFERENCES / IDENTITY':value.startsWith('v2')?'VIDEO / TRANSFORM':['i2va','fl2va_stable','fl2va_continuous'].includes(value)?'IMAGES / SHOTS':'TIMELINE / LONG FORM';
    for(const category of ['TEXT / CREATE','IMAGES / SHOTS','EXTENSION / LONG TAKE','REFERENCES / IDENTITY','VIDEO / TRANSFORM','TIMELINE / LONG FORM']){
      const box=document.createElement('section');box.style.cssText='grid-column:1/-1;border:1px solid #426475;border-radius:10px;padding:12px;background:linear-gradient(135deg,#142632,#18202d)';
      const heading=document.createElement('h3');heading.textContent=category;heading.style.cssText='margin:0 0 10px;color:#9bdccc;font-size:12px;letter-spacing:1px';box.append(heading);
      const cards=document.createElement('div');cards.className='h3p-choice-grid';
      MODE_CHOICES.filter(([,value])=>!HIDDEN_MODES.has(value)&&!UNLISTED_MODES.has(value)&&family(value)===category).forEach(([label,value,note])=>{
        const row=document.createElement('div'); row.className='h3p-mode-engine-row'; row.style.cssText='display:grid;grid-template-columns:minmax(0,1fr) auto;gap:7px;align-items:stretch;min-width:0';
        const button=document.createElement('button'); const availability=modeAvailability(node,value); button.className=`h3p-choice${selectedMode===value?' active':''}`; button.innerHTML=`${label}<span>${note}${availability.available?'':` · ${availability.reason}`}</span>`; button.disabled=!availability.available; button.title=availability.reason||note; button.onclick=()=>{if(availability.available)setAssistantMode(node,value);}; row.append(button);
        const targetEngine=MODE_ENGINE_GROUP[value];
        if(targetEngine){
          const engine=document.createElement('button'); engine.type='button'; engine.className='h3p-recipe h3p-mode-engine-toggle'; engine.textContent='ENGINE ▸'; engine.title='Select this mode and open its dedicated Settings PRO engine panel'; engine.disabled=!availability.available; engine.style.cssText='min-width:76px;padding:8px 9px;font-size:9px;letter-spacing:.7px';
          engine.onclick=()=>{if(!availability.available)return;setAssistantMode(node,value);node.properties ||= {};node.properties.iamccs_h3_settings_pro_engines_open=true;node.properties.iamccs_h3_settings_pro_mode_controls_open=true;active=targetEngine;node.properties.iamccs_h3_settings_pro_active_section=active;app.graph?.change?.();refresh();}; row.append(engine);
        }
        cards.append(row);
      });box.append(cards);modes.append(box);
    }grid.append(modes);
    if (selectedMode === "longvid_guides") {
      ask("1B · LONGVID GUIDE WINDOW STRATEGY");
      const guideWindows = document.createElement("div"); guideWindows.className = "h3p-recipes h3p-guide-window-recipes";
      const selectedGuideWindow = selectedGuideWindowStrategy(node);
      addRecipe(guideWindows, "AUTO BY CANVAS · IAMCCS PIXEL-FRAME BUDGET", "guide_auto", true, false, () => applyGuideWindowStrategy(node, "guide_auto"));
      addRecipe(guideWindows, "NATIVE MACRO · 362F", "guide_macro_362", true, selectedGuideWindow === "guide_macro_362", () => applyGuideWindowStrategy(node, "guide_macro_362"));
      addRecipe(guideWindows, "HIGH-RES SAFE · 209F + 22F LATENT TAIL", "guide_safe_209", true, selectedGuideWindow === "guide_safe_209", () => applyGuideWindowStrategy(node, "guide_safe_209"));
      addRecipe(guideWindows, "HIGH-RES ULTRA · 124F + 22F LATENT TAIL", "guide_safe_124", true, selectedGuideWindow === "guide_safe_124", () => applyGuideWindowStrategy(node, "guide_safe_124"));
      const guideNote = document.createElement("p"); guideNote.textContent = "IAMCCS Adaptive Guide Windows: the editorial guide clock is unchanged. High-resolution presets reduce only the internal H3 sampling window and carry a native 22-frame latent tail; guides are rebased into the window that owns their global frame.";
      grid.append(guideWindows, guideNote);
    }
    const rig=document.createElement('button');rig.className='h3p-recipe';rig.style.cssText='grid-column:1/-1;margin-top:18px;padding:12px 16px;justify-self:start';rig.textContent='RIG · ADD MISSING MEDIA INPUTS';rig.onclick=()=>alert(rigMedia(node,mode,connectedNodes(node)));grid.append(rig);
    if(active==='producer'){
      ask('PRODUCER / '+(MODE_CHOICES.find(([,value])=>value===selectedMode)?.[0]||mode));
      const notes=document.createElement('p');notes.textContent=(MODE_CHOICES.find(([,value])=>value===selectedMode)?.[2]||'Use the selected mode contract.')+' Presets below change only their stated section. RIG never creates a missing generation backend. Timeline modes keep their media in Shotboard.';grid.append(notes);
      const recipes=document.createElement('div');recipes.className='h3p-recipes';
      addRecipe(recipes,'NATIVE DELIVERY · no upscale / no interpolation','native-delivery');
      if(mode!=='latent_go_ahead'){addRecipe(recipes,'12–16 GB · 209-frame window','vram12');addRecipe(recipes,'16–24 GB · 294-frame window','vram16');addRecipe(recipes,'24 GB+ · 362-frame window','vram24');}
      const caution=document.createElement('p');caution.textContent='Memory presets are starting points, not VRAM guarantees. They change window and memory controls, not resolution. LatentGoAhead supports compatible accelerators; Fused is T2VA only and progressive sampling cannot preserve its audio mask. SAM3 Subject Swap uses one Picture 1 by default; a second identity view enables the optional BiRefNet legacy card. ControlNet requires a compatible control model and real preprocessed frames.';grid.append(recipes,caution);
    }
    ask("2 · WHAT MEDIA DO YOU HAVE?"); const media = document.createElement("div"); media.className = "h3p-recipes";
    const mediaText = String(mode) === "fl2va_extended_av" ? "PICTURE 1 OPENER · DESTINATION IMAGE OPTIONAL" : String(mode).startsWith("t2") ? "PROMPT ONLY" : String(mode).includes("controlnet") ? "PREPROCESSED CONTROL VIDEO" : String(mode).includes("face_swap") ? "SOURCE VIDEO + IDENTITY REFERENCES" : String(mode).includes("ref2") ? "ONE OR MORE REFERENCE IMAGES" : String(mode).includes("fl2") ? "FIRST + LAST IMAGE" : String(mode).includes("i2") ? "OPENING IMAGE" : "SHOTBOARD MEDIA";
    const mediaTag = document.createElement("button"); mediaTag.className = "h3p-choice active"; mediaTag.textContent = mediaText; media.append(mediaTag); grid.append(media);
    ask("3 · CHOOSE AUDIO AUTHORITY"); const audio = document.createElement("div"); audio.className = "h3p-recipes";
    const boardAudio = String(widget(linkedShotboard(node), "audio_mode")?.value || "h3_native_generated");
    [["GENERATED AUDIO", "h3_native_generated"], ["REFERENCE AUDIO", "h3_ref2va_audio"], ["CUSTOM AUDIO DRIVE", "h3_custom_audio_drive"], ["AUDIO IN POST", "external_audio_post"]].forEach(([label, value]) => { const button = document.createElement("button"); button.className = `h3p-recipe${boardAudio === value ? " active" : ""}`; button.textContent = label; button.onclick = () => setShotboardAudio(node, value); audio.append(button); }); grid.append(audio);
    ask("4 · SAMPLING RECIPE · CHOOSE ONE"); const speed = document.createElement("div"); speed.className = "h3p-recipes";
    const selectedSpeed = selectedAcceleration(node);
    speed.classList.add("h3p-speed-recipes");
    addRecipe(speed, "NATIVE QUALITY", "native", true, selectedSpeed === "native"); addRecipe(speed, "PDD · 8 STEP", "pdd", accelerationAvailable(node,"pdd",mode), selectedSpeed === "pdd");
    addRecipe(speed, "FASTH3 OMNI · 8 STEP", "omni", mode === "ref2va", selectedSpeed === "omni");
    addRecipe(speed, "FASTH3 · 6 STEP", "fasth3", accelerationAvailable(node,"fasth3",mode), selectedSpeed === "fasth3"); addRecipe(speed, "SLA · 4 STEP", "sla", accelerationAvailable(node,"sla",mode), selectedSpeed === "sla");
    addRecipe(speed, "FUSED FAST · T2VA ONLY", "fused", accelerationAvailable(node,"fused",mode), selectedSpeed === "fused"); grid.append(speed);
    const recipeInfo = document.createElement("p");
    recipeInfo.textContent = `ACTIVE: ${samplingRecipeLabel(node)} · ${widget(node,"sampler_name")?.value} / ${widget(node,"scheduler")?.value}. Choose one sampling recipe: selecting another replaces its adapter and settings. Step counts are not added. Omni locks its trained recipe and disables VEDA. Other modes can use compatible adapters. Output delivery remains independent.`;
    recipeInfo.className = "h3p-sampling-summary"; grid.append(recipeInfo);
    ask("TURBO RECIPES · ALTERNATIVES IN THE SAME SAMPLING CHOICE");
    const turboGroups = document.createElement("div"); turboGroups.style.cssText = "grid-column:1/-1;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px";
    for (const [label, family, category] of [["T2V / T2VA", "fl2", "t2"], ["I2V / I2VA", "fl2", "i2"], ["FLF2V / FL2VA / EXTENDED", "fl2", "fl2"], ["REF2V / REF2VA / CHARACTER SWAP", "ref2", "ref2"]]) {
      const section = document.createElement("section"); section.className = "h3p-functional"; section.style.cssText = "grid-column:auto;min-width:0;margin:0;padding:8px";
      const title = document.createElement("div"); title.className = "h3p-functional-title"; title.textContent = `${label} · ${family === "ref2" ? "REF2VA" : "FL2VA"} WEIGHTS`; section.append(title);
      const row = document.createElement("div"); row.className = "h3p-recipes h3p-turbo-recipes";
      const assets = assistantTurboAssets(node, family), compatible = modeFamily(mode) === family;
      for (const asset of assets) {
        const selected = compatible && category === turboModeGroup(mode) && selectedSpeed === "turbo" && widget(node,"turbo_lora_name")?.value === asset.name;
        const version = asset.name.match(/_v(\d+(?:\.\d+)?)/i)?.[1];
        const shortLabel = `${declaredTurboSteps(node,asset.name)} STEP${/768/.test(asset.name + asset.recipe) ? " · 768P" : ""}${version ? ` · v${version}` : ""}`;
        addRecipe(row, shortLabel, `turbo:${asset.name}`, compatible, selected, () => applyAssistantTurbo(node,asset.name));
        row.lastElementChild.title = `${asset.name}\n${widget(node,"turbo_lora_name").options.iamccs_h3_recipes[asset.recipe].note || ""}`;
      }
      const note = document.createElement("p"); note.textContent = !assets.length ? "No installed Turbo adapter with a verified family and sampling recipe." : compatible ? "Replaces the speed recipe with this adapter, steps, sampler and shifts. Shared family weights do not change the generation mode; selection is marked in the current mode group. Keeps character/style LoRAs, VEDA and delivery." : "Select a generation mode in this family to enable these adapters.";
      row.style.cssText = "display:flex;flex-wrap:wrap;gap:6px;min-width:0";
      row.querySelectorAll("button").forEach(button => button.style.cssText = "flex:0 1 auto;min-height:28px;padding:6px 8px;white-space:normal;text-align:center");
      note.style.cssText = "font-size:9px;line-height:1.3;margin:6px 0 0";
      section.append(row, note); turboGroups.append(section);
    }
    grid.append(turboGroups);
    ask("ATTENTION & MEMORY · INDEPENDENT OF SAMPLING STEPS");
    const attention = document.createElement("div"); attention.className = "h3p-recipes h3p-attention-recipes";
    const integrated = integratedAttention(node), hasAttention = Boolean(widget(node,"h3_attention_backend"));
    for (const [label,id] of ATTENTION_PRESETS) addRecipe(attention,label,`attention:${id}`,hasAttention && !integrated,!integrated && selectedAttention(node) === id,() => setAttentionChoice(node,id));
    const attentionNote = document.createElement("p");
    attentionNote.textContent = !hasAttention ? "Restart ComfyUI and reload the page to enable independent attention presets." : integrated ? `ACTIVE: ${integrated}. This sampling recipe manages attention internally. Your separate attention choice is retained for Native / Turbo.` : `ACTIVE: ${attentionLabel(node)}. Changes attention and memory handling only; keeps steps, sampler, speed LoRA and delivery. Sage + Low VRAM is the recommended starting point on RTX 3060. Sol selects its sparse low-VRAM path and turns VEDA off.`;
    grid.append(attention,attentionNote);
    const veda = document.createElement("section"); veda.className = "h3p-functional"; veda.style.gridColumn = "1/-1";
    const vedaTitle = document.createElement("div"); vedaTitle.className = "h3p-functional-title"; vedaTitle.textContent = "VEDA · OPTIONAL SPARSE ATTENTION"; veda.append(vedaTitle);
    const vedaFields = document.createElement("div"); vedaFields.className = "h3p-functional-grid";
    for (const field of ["h3_veda_enabled", "h3_veda_predictor"]) { const control = makeControl(field); if (control) vedaFields.append(control); }
    const vedaNote = document.createElement("p"); vedaNote.textContent = omniActive(node) ? "Unavailable in the native FastH3 Omni profile. Select a standard H3 mode to use VEDA." : vedaAvailable(node) ? "Optional sparse attention; keeps the sampling step count. AUTO selects the matching T2VA/R2VA predictor. Requires Veda-on-ComfyUI and its predictor." : "Unavailable with Sol, SLA or Fused attention. Select Sage / Auto Low VRAM to use VEDA.";
    veda.append(vedaFields, vedaNote); grid.append(veda);

    ask("5 · CHOOSE A VRAM PRESET"); const memory = document.createElement("div"); memory.className = "h3p-recipes h3p-memory-recipes";
    const selectedVram = selectedMemory(node), selectedFinish = selectedDelivery(node), selectedFace = selectedFaceRefine(node);
    addRecipe(memory, "≤ 8–12 GB · 124F / 2048 ROWS", "vram8", true, selectedVram === "vram8"); addRecipe(memory, "12–16 GB · 209F / 4096 ROWS", "vram12", true, selectedVram === "vram12"); addRecipe(memory, "16–24 GB · 294F / 8192 ROWS", "vram16", true, selectedVram === "vram16"); addRecipe(memory, "24 GB+ · 362F / 16384 ROWS", "vram24", true, selectedVram === "vram24"); grid.append(memory);
    ask("6 · CHOOSE DELIVERY");
    const delivery = document.createElement("div"); delivery.className = "h3p-recipes h3p-delivery-recipes";
    addRecipe(delivery, "FACE REFINE · TOGGLE", "face-refine", true, selectedFace);
    addRecipe(delivery, "NATIVE DELIVERY", "native-delivery", true, selectedFinish === "native-delivery");
    addRecipe(delivery, "OPEN-H3 REFINEMENT · 3 STEP", "open-h3-refine", branchGate(node, "h3_open_refine_enabled").available, selectedFinish === "open-h3-refine");
    addRecipe(delivery, "H3 DETAIL · 1-STEP REFINE", "ultimate-tiled", deliveryModeAvailable(node, "h3_ultimate_tiled"), selectedFinish === "ultimate-tiled");
    addRecipe(delivery, "H3 2-PASS UPRES", "upres", branchGate(node, "h3_upres_model_name").available && deliveryModeAvailable(node, "h3_fast_latent_2pass"), selectedFinish === "upres");
    addRecipe(delivery, "PIXEL SAFE · LOW VRAM", "pixel-safe", deliveryModeAvailable(node, "pixel_tiled_low_vram"), selectedFinish === "pixel-safe");
    addRecipe(delivery, "H3 PIXEL REFINE", "h3-pixel", deliveryModeAvailable(node, "h3_pixel_refine"), selectedFinish === "h3-pixel");
    addRecipe(delivery, "LTX 2.3 MASTER", "ltx-master", deliveryModeAvailable(node, "ltx23"), selectedFinish === "ltx-master");
    addRecipe(delivery, "LTX 2.3 PER-SHOT", "ltx-per-shot", deliveryModeAvailable(node, "ltx23_per_chunk"), selectedFinish === "ltx-per-shot");
    addRecipe(delivery, "RTX FINAL", "rtx-final", deliveryModeAvailable(node, "rtx_final"), selectedFinish === "rtx-final");
    grid.append(delivery);
    const detailNote = document.createElement("p");
    detailNote.textContent = mode === "v2va_face_swap" ? "SAM3 crop swap supports Native / RTX delivery. H3 Detail needs a full-frame latent and is unavailable for this legacy crop mode." : "H3 DETAIL uses the proven SwapChar second-pass recipe for full-frame video modes: learned latent upscale, 1 step, SA Solver / simple, denoise 0.2, seed 777. Output exposes the shared seed, dimensions, model, sampling and tile controls. The native pass keeps its own settings.";
    if (selectedFinish === "open-h3-refine") detailNote.textContent = "One native generation followed by Open-H3 refinement. Dedicated controls are in Output / Open-H3 Refinement. The source soundtrack is retained. Gentle sigma 0.8 still requires visual review of identity and acting; the high-resolution transformer pass is not tiled.";
    grid.append(detailNote);
  }
  async function renderAssetLibrary(kind, grid) {
    grid.replaceChildren();
    const status = document.createElement("div"); status.className = "h3p-asset-status"; status.textContent = "Reading saved latent metadata…"; grid.append(status);
    try {
      const response = await fetch(`/api/iamccs/h3/assets?kind=${kind}`, {cache:"no-store"});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      if (active !== kind) return;
      grid.replaceChildren();
      if (!data.items?.length) { status.textContent = kind === "continuation" ? "No saved AV checkpoint. Connect IAMCCS Continuation Save after the accepted native sampler latent." : "No saved RefMod. Save a RefMod in models/refmods, then use IAMCCS RefMod Preview Cache with its matching H3 VAE."; grid.append(status); return; }
      for (const item of data.items) {
        const card = document.createElement("article"); card.className = "h3p-asset";
        if (item.preview) {
          const image = document.createElement("img");
          image.src = `/api/iamccs/h3/assets/preview?${new URLSearchParams({kind,root:String(item.root),path:item.path})}`;
          image.alt = `${item.name} latent preview`; card.append(image);
        } else { const placeholder = document.createElement("div"); placeholder.className = "h3p-asset-placeholder"; placeholder.textContent = "PREVIEW NOT CACHED"; card.append(placeholder); }
        const name = document.createElement("strong"); name.textContent = item.name;
        const detail = document.createElement("small"); detail.textContent = `${item.detail} · ${item.size_mb} MB · ${item.path}`;
        const facts = document.createElement("small"); facts.textContent = Array.isArray(item.facts) ? item.facts.filter(Boolean).join(" · ") : "";
        facts.style.color = "#d7b879";
        const prompt = document.createElement("small"); prompt.textContent = item.prompt ? `PROMPT · ${item.prompt}` : "";
        prompt.style.cssText = "white-space:pre-wrap;max-height:72px;overflow:auto";
        const actions = document.createElement("div"); actions.className = "h3p-asset-actions";
        if (kind === "continuation") {
          const select = document.createElement("button"); select.textContent = String(widget(node,"h3_continuation_checkpoint")?.value || "").endsWith(item.path) ? "SELECTED" : "USE IN PRO";
          select.onclick = () => { setValue(node,"h3_continuation_checkpoint",`IAMCCS/MiniMaxH3/CONTINUATION/${item.path}`); setValue(node,"h3_continuation_enabled",true); node._iamccsSettingsProRefresh?.(); };
          actions.append(select);
        } else {
          const modName = item.path.replace(/\.safetensors$/i, "");
          const select = document.createElement("button");
          select.textContent = String(widget(node,"h3_refmod_name")?.value || "") === modName ? "SELECTED" : "USE IN PRO";
          select.onclick = () => {
            setValue(node,"h3_refmod_name",modName);
            setValue(node,"h3_refmod_enabled",true);
            node._iamccsSettingsProRefresh?.();
          };
          actions.append(select);
        }
        const copy = document.createElement("button"); copy.textContent = "COPY PATH"; copy.onclick = async () => {
          const path = kind === "continuation" ? `IAMCCS/MiniMaxH3/CONTINUATION/${item.path}` : item.path.replace(/\.safetensors$/i, "");
          try { await navigator.clipboard.writeText(path); copy.textContent = "COPIED"; } catch { window.prompt("Copy asset path", path); }
        };
        const rename = document.createElement("button"); rename.textContent = "RENAME"; rename.onclick = async () => {
          const next = window.prompt(kind === "refmod" ? "New RefMod filename (update any saved workflow that refers to the old name):" : "New checkpoint filename:", item.name);
          if (!next || next === item.name) return;
          try {
            const result = await fetch("/api/iamccs/h3/assets/rename", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({kind,root:item.root,path:item.path,name:next})});
            const payload = await result.json(); if (!result.ok) throw new Error(payload.error || `HTTP ${result.status}`);
            node._iamccsSettingsProRefresh?.();
          } catch (error) { window.alert(`Rename failed: ${error.message}`); }
        };
        const purgeAsset = document.createElement("button"); purgeAsset.textContent = "PURGE LATENT"; purgeAsset.className = "h3p-danger"; purgeAsset.onclick = async () => {
          if (!window.confirm(`Permanently delete only the latent ${item.name}? Its preview and IAMCCS lineage manifest are preserved.`)) return;
          purgeAsset.disabled = true; purgeAsset.textContent = "PURGING…";
          try {
            const result = await fetch("/api/iamccs/h3/assets/purge", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({kind,root:item.root,path:item.path,scope:"asset"})});
            const payload = await result.json(); if (!result.ok) throw new Error(payload.error || `HTTP ${result.status}`);
            if (kind === "continuation" && String(widget(node,"h3_continuation_checkpoint")?.value || "").endsWith(item.path)) {
              setValue(node,"h3_continuation_checkpoint",""); setValue(node,"h3_continuation_enabled",false);
            }
            const modName = item.path.replace(/\.safetensors$/i, "");
            if (kind === "refmod" && String(widget(node,"h3_refmod_name")?.value || "") === modName) {
              setValue(node,"h3_refmod_name",""); setValue(node,"h3_refmod_enabled",false);
            }
            node._iamccsSettingsProRefresh?.();
          } catch (error) { purgeAsset.disabled = false; purgeAsset.textContent = "PURGE ASSET"; window.alert(`Asset purge failed: ${error.message}`); }
        };
        actions.append(copy, rename, purgeAsset); card.append(name,detail,facts,prompt,actions); grid.append(card);
      }
    } catch (error) { status.textContent = `Library unavailable: ${error.message}`; }
    finally { preserveEditorAfterRefresh(); }
  }
  async function renderCacheLibrary(grid) {
    grid.replaceChildren();
    const status = document.createElement("div"); status.className = "h3p-asset-status"; status.textContent = "Reading IAMCCS disk caches…"; grid.append(status);
    try {
      const response = await fetch("/api/iamccs/h3/caches", {cache:"no-store"});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      if (active !== "continuation" && active !== "refmod") return;
      grid.replaceChildren();
      const items = Array.isArray(data.items) ? data.items : [];
      if (!items.length) { status.textContent = "No purgeable IAMCCS LongVid / latent-upres / disk-bridge cache files found."; grid.append(status); return; }

      const totalMb = items.reduce((sum,item)=>sum + Number(item.size_mb || 0), 0);
      const toolbar = document.createElement("div"); toolbar.className = "h3p-asset-actions"; toolbar.style.cssText = "display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:6px 0 10px";
      const selectAllLabel = document.createElement("label"); selectAllLabel.style.cssText = "display:flex;align-items:center;gap:6px;font-size:11px";
      const selectAll = document.createElement("input"); selectAll.type = "checkbox";
      selectAllLabel.append(selectAll, document.createTextNode(`SELECT ALL · ${items.length} files · ${totalMb.toFixed(1)} MB`));
      const purgeSelected = document.createElement("button"); purgeSelected.className = "h3p-danger"; purgeSelected.textContent = "PURGE SELECTED"; purgeSelected.disabled = true;
      toolbar.append(selectAllLabel, purgeSelected); grid.append(toolbar);

      const selected = new Map();
      const refreshSelection = () => {
        const checks = [...selected.values()];
        const count = checks.filter(c=>c.checked).length;
        purgeSelected.disabled = count === 0;
        purgeSelected.textContent = count ? `PURGE SELECTED (${count})` : "PURGE SELECTED";
        selectAll.checked = checks.length > 0 && count === checks.length;
        selectAll.indeterminate = count > 0 && count < checks.length;
      };
      const purgeOne = async (item) => {
        const result = await fetch("/api/iamccs/h3/caches/purge", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({root:item.root,path:item.path})});
        const payload = await result.json(); if (!result.ok) throw new Error(payload.error || `HTTP ${result.status}`);
      };

      const groups = new Map();
      for (const item of items) {
        const key = String(item.category || "OTHER CACHE");
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(item);
      }
      for (const [category, groupItems] of groups) {
        const details = document.createElement("details"); details.className = "h3p-cache-group"; details.style.cssText = "border:1px solid #36424b;border-radius:8px;margin:5px 0;background:#11191f";
        const summary = document.createElement("summary"); summary.style.cssText = "cursor:pointer;padding:9px 11px;font-weight:600;display:flex;align-items:center;gap:8px";
        const groupMb = groupItems.reduce((sum,item)=>sum + Number(item.size_mb || 0), 0);
        const groupToggle = document.createElement("input"); groupToggle.type = "checkbox"; groupToggle.title = `Select all ${category} cache files`;
        groupToggle.onclick = (event) => event.stopPropagation();
        const groupText = document.createElement("span"); groupText.textContent = `${category} · ${groupItems.length} file(s) · ${groupMb.toFixed(1)} MB`;
        summary.append(groupToggle,groupText);
        details.append(summary);
        const rows = document.createElement("div"); rows.style.cssText = "display:flex;flex-direction:column;gap:4px;padding:6px 10px 10px";
        for (const item of groupItems) {
          const row = document.createElement("label"); row.style.cssText = "display:grid;grid-template-columns:18px minmax(120px,1fr) auto;gap:8px;align-items:center;padding:5px 6px;border-top:1px solid #26323a;font-size:10px";
          const check = document.createElement("input"); check.type = "checkbox"; selected.set(`${item.root}:${item.path}`, check); check.onchange = refreshSelection;
          const text = document.createElement("span"); const stamp = item.modified_at ? new Date(item.modified_at * 1000).toLocaleString() : "date unavailable"; text.textContent = `${item.name} · ${item.size_mb} MB · ${stamp}`; text.title = item.path;
          const purge = document.createElement("button"); purge.type = "button"; purge.className = "h3p-danger"; purge.textContent = "PURGE"; purge.onclick = async (event) => { event.preventDefault(); event.stopPropagation(); if (!window.confirm(`Delete this IAMCCS cache file?\n\n${item.path}`)) return; purge.disabled = true; try { await purgeOne(item); renderCacheLibrary(grid); } catch (error) { purge.disabled = false; window.alert(`Cache purge failed: ${error.message}`); } };
          row.append(check,text,purge); rows.append(row);
        }
        groupToggle.onchange = () => {
          for (const item of groupItems) {
            const check = selected.get(`${item.root}:${item.path}`);
            if (check) check.checked = groupToggle.checked;
          }
          refreshSelection();
        };
        details.append(rows); grid.append(details);
      }
      selectAll.onchange = () => { for (const check of selected.values()) check.checked = selectAll.checked; refreshSelection(); };
      purgeSelected.onclick = async () => {
        const chosen = items.filter(item => selected.get(`${item.root}:${item.path}`)?.checked);
        if (!chosen.length) return;
        if (!window.confirm(`Permanently delete ${chosen.length} selected IAMCCS cache file(s)? Finished videos, continuation latents and RefMods are not part of this cache list.`)) return;
        purgeSelected.disabled = true; purgeSelected.textContent = "PURGING…";
        try { for (const item of chosen) await purgeOne(item); renderCacheLibrary(grid); }
        catch (error) { purgeSelected.disabled = false; window.alert(`Cache purge failed: ${error.message}`); }
      };
      refreshSelection();
    } catch (error) { grid.replaceChildren(status); status.textContent = `Cache library unavailable: ${error.message}`; }
    finally { preserveEditorAfterRefresh(); }
  }
  function r45ScoutController() {
    return connectedNodes(node).find((candidate) => nodeClass(candidate) === "IAMCCS_MiniMaxH3TakeSelectControlR40") || null;
  }
  function r45ApplyScoutPhase(phaseValue) {
    const controller = r45ScoutController();
    if (!controller) return;
    const phase = String(phaseValue || "scout") === "deliver" ? "deliver" : "scout";
    const scoutEnabled = Boolean(widget(node, "h3_r40_seed_scout_enabled")?.value);
    controller.properties ||= {};
    controller.properties.iamccs_r40_phase = phase;
    const never = window?.LiteGraph?.NEVER ?? 2;
    for (const target of connectedNodes(node)) {
      const role = String(target?.properties?.iamccs_r40_phase_role || "");
      if (!role) continue;
      const activeRole = scoutEnabled
        ? (role === "preview" ? phase === "scout" : phase === "deliver")
        : role !== "preview";
      target.mode = activeRole ? 0 : never;
    }
    controller.graph?.setDirtyCanvas?.(true, true);
    app.graph?.change?.();
  }
  function r45NodeOutput(graphNode) {
    if (!graphNode) return {};
    const outputs = app.nodeOutputs || {};
    if (outputs[graphNode.id]) return outputs[graphNode.id];
    const matches = Object.keys(outputs).filter((id) => id.endsWith(`:${graphNode.id}`));
    return matches.length === 1 ? outputs[matches[0]] : {};
  }
  function r45ViewUrl(item) {
    if (!item || !item.filename) return "";
    const params = new URLSearchParams({ filename: String(item.filename), type: String(item.type || "temp") });
    if (item.subfolder) params.set("subfolder", String(item.subfolder));
    return api.apiURL(`/view?${params.toString()}`);
  }
  function r45CandidateNodes(index) {
    const graphNodes = connectedNodes(node);
    const preview = graphNodes.find((candidate) => nodeClass(candidate) === "PreviewImage" && String(candidate.title || "").includes(`TAKE ${index}`));
    const seed = graphNodes.find((candidate) => {
      if (nodeClass(candidate) !== "IAMCCS_MiniMaxH3SeedCandidateR40") return false;
      const candidateWidget = widget(candidate, "candidate_index");
      return Number(candidateWidget?.value ?? -1) === index - 1;
    });
    return { preview, seed };
  }
  function r45ClearScoutPreviewTimers() {
    for (const timer of (node._iamccsScoutPreviewTimers || [])) window.clearInterval(timer);
    node._iamccsScoutPreviewTimers = [];
  }
  function renderIntegratedScout(mode) {
    const grid = q("[data-grid]"); grid.className = "h3p-flow"; grid.replaceChildren();
    const recipes = q(".h3p-recipes"); recipes.replaceChildren();
    r45ClearScoutPreviewTimers();
    const controller = r45ScoutController();
    const phase = String(controller?.properties?.iamccs_r40_phase || "scout") === "deliver" ? "deliver" : "scout";
    const enabled = Boolean(widget(node, "h3_r40_seed_scout_enabled")?.value);
    const count = Math.max(1, Math.min(4, Number(widget(node, "h3_r40_candidate_count")?.value || 1)));

    const phaseBar = document.createElement("div"); phaseBar.className = "h3p-recipes"; phaseBar.style.cssText = "grid-column:1/-1";
    const scoutButton = document.createElement("button"); scoutButton.className = `h3p-recipe${phase === "scout" ? " active" : ""}`; scoutButton.textContent = "1 · SCOUT CANDIDATES";
    const deliverButton = document.createElement("button"); deliverButton.className = `h3p-recipe${phase === "deliver" ? " active" : ""}`; deliverButton.textContent = "2 · DELIVER SELECTED TAKE";
    scoutButton.disabled = !enabled; deliverButton.disabled = !enabled;
    scoutButton.onclick = () => { if (!enabled) return; r45ApplyScoutPhase("scout"); renderIntegratedScout(mode); };
    deliverButton.onclick = () => { if (!enabled) return; r45ApplyScoutPhase("deliver"); renderIntegratedScout(mode); };
    phaseBar.append(scoutButton, deliverButton); grid.append(phaseBar);

    const controls = document.createElement("section"); controls.className = "h3p-functional";
    controls.innerHTML = '<div class="h3p-functional-title">SCOUT ENGINE · QUEUE VALUES</div>';
    const fields = document.createElement("div"); fields.className = "h3p-functional-grid";
    for (const name of ["h3_r40_seed_scout_enabled","h3_r40_candidate_count","h3_r40_seed_stride","h3_r40_preview_max_frames","h3_r40_sparse_enabled","h3_r40_sparse_video_budget","h3_r40_sparse_denser_edges"]) {
      const control = makeControl(name); if (control) fields.append(control);
    }
    controls.append(fields); grid.append(controls);

    const status = document.createElement("div");
    status.style.cssText = "grid-column:1/-1;padding:8px 10px;border-left:3px solid #d4ad62;background:#111a1e;color:#cdd8d8;font-size:10px;line-height:1.45";
    status.textContent = !controller ? "Scout backend not wired in this workflow." : !enabled ? "SCOUT is OFF. Enable it above; cached preview nodes remain untouched." : phase === "scout" ? "SCOUT PHASE · Queue renders the candidate bank/previews only. Select a winning TAKE below, then switch to DELIVER." : "DELIVER PHASE · Queue reuses the cached winning take; candidates are not resampled.";
    grid.append(status);

    const candidateGrid = document.createElement("div"); candidateGrid.style.cssText = "grid-column:1/-1;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px";
    const selectedWidget = controller ? widget(controller, "selected_candidate") : null;
    const selected = Math.max(1, Math.min(4, Number(selectedWidget?.value || 1)));
    for (let index = 1; index <= 4; index++) {
      const { preview, seed } = r45CandidateNodes(index);
      const output = r45NodeOutput(preview);
      const images = Array.isArray(output.images) ? output.images : [];
      const urls = images.map(r45ViewUrl).filter(Boolean);
      const seedOutput = r45NodeOutput(seed);
      const seedRaw = Array.isArray(seedOutput.candidate_seed) ? seedOutput.candidate_seed[0] : seedOutput.candidate_seed;
      const card = document.createElement("section");
      const activeCandidate = index <= count;
      card.style.cssText = `border:1px solid ${selected === index ? "#e2bf72" : "#3c5058"};border-radius:8px;padding:8px;background:#10181d;opacity:${activeCandidate ? "1" : ".42"};box-shadow:${selected === index ? "0 0 0 1px rgba(226,191,114,.35)" : "none"}`;
      const head = document.createElement("div"); head.style.cssText = "display:flex;justify-content:space-between;gap:8px;align-items:center;margin-bottom:7px";
      const label = document.createElement("b"); label.textContent = `TAKE ${index}${selected === index ? " · SELECTED" : ""}`; label.style.color = selected === index ? "#f2cf7d" : "#d9e3e3";
      const seedText = document.createElement("span"); seedText.textContent = seedRaw != null ? `seed ${seedRaw}` : "seed —"; seedText.style.cssText = "font-size:9px;color:#94a6a8"; head.append(label,seedText);
      const stage = document.createElement("div"); stage.style.cssText = "height:180px;background:#050809;border:1px solid #26363d;border-radius:5px;display:flex;align-items:center;justify-content:center;overflow:hidden";
      const img = document.createElement("img"); img.alt = `Scout Take ${index}`; img.style.cssText = "width:100%;height:100%;object-fit:contain;display:block";
      if (urls.length) {
        let frame = 0; img.src = urls[0];
        const timer = window.setInterval(() => { if (!img.isConnected || !urls.length) return; frame = (frame + 1) % urls.length; img.src = urls[frame]; }, 125);
        node._iamccsScoutPreviewTimers ||= []; node._iamccsScoutPreviewTimers.push(timer);
      } else {
        img.style.display = "none"; const empty = document.createElement("span"); empty.textContent = activeCandidate ? "Queue SCOUT to build video preview" : "Candidate disabled by count"; empty.style.cssText = "font-size:10px;color:#718589"; stage.append(empty);
      }
      stage.append(img);
      const choose = document.createElement("button"); choose.className = `h3p-recipe${selected === index ? " active" : ""}`; choose.textContent = selected === index ? "WINNING TAKE" : `SELECT TAKE ${index}`; choose.disabled = !controller || !activeCandidate;
      choose.style.cssText += ";width:100%;margin-top:7px";
      choose.onclick = () => { if (!selectedWidget) return; selectedWidget.value = String(index); try { selectedWidget.callback?.(String(index)); } catch {} app.graph?.change?.(); renderIntegratedScout(mode); };
      card.append(head,stage,choose); candidateGrid.append(card);
    }
    grid.append(candidateGrid);
    const footer = document.createElement("small"); footer.style.cssText = "grid-column:1/-1;color:#8ea0a2;line-height:1.45"; footer.textContent = "Preview playback is reconstructed from the existing cached R40 preview frames inside Settings PRO. The R40 candidate nodes and cache contract remain unchanged for legacy workflows."; grid.append(footer);
  }
  function renderMain(mode) {
    q(".h3p-main").dataset.section = active;
    if (active === "continuation" || active === "refmod") {
      const kind = active;
      q(".h3p-section-title").textContent = kind === "continuation" ? "CONTINUATION · SAVED AV CHECKPOINT" : "REFMOD · REFERENCE LATENTS";
      q(".h3p-section-note").textContent = kind === "continuation" ? "CONTINUATION means RESUME: begin a new generation from a previously saved terminal AV latent. It does not make the current shot longer automatically. Use LONG TAKE · EXTEND for >15s single-shot generation." : "Select a saved RefMod and enable automatic injection into native H3 conditioning. The installed provider loads and applies it once per generated chunk; leave OFF to preserve the old route.";
      q(".h3p-context").textContent = kind === "continuation" ? "Use this panel only when you already have a saved take/checkpoint to continue from. For one Picture 1 and a 20s/30s continuous take, select EXTENDED AV instead." : "RefMod is appearance/voice reference context, not temporal continuation. Do not also connect an external Apply H3 RefMod to the same branch.";
      const grid = q("[data-grid]"); grid.className = "h3p-grid"; grid.replaceChildren();
      if (kind === "continuation") {
        const controls = document.createElement("section"); controls.className = "h3p-functional";
        const heading = document.createElement("div"); heading.className = "h3p-functional-title"; heading.textContent = "QUEUE VALUES · CONNECT SHOTBOARD CINE LINX TO IAMCCS LOAD + CONTINUE";
        const fields = document.createElement("div"); fields.className = "h3p-functional-grid";
        for (const name of ["h3_continuation_enabled", "h3_continuation_save_enabled", "h3_continuation_checkpoint", "h3_continuation_context_frames", "h3_continuation_handover_mode", "h3_continuation_manual_tail_frames", "h3_continuation_visual_handover", "h3_continuation_run_and_gun_enabled", "h3_continuation_run_and_gun_join", "h3_continuation_soft_video_frames", "h3_continuation_soft_video_curve", "h3_continuation_soft_audio_ms"]) {
          if (name === "h3_continuation_manual_tail_frames" && String(widget(node,"h3_continuation_handover_mode")?.value || "auto") !== "manual") continue;
          if (name === "h3_continuation_run_and_gun_join" && !Boolean(widget(node,"h3_continuation_run_and_gun_enabled")?.value)) continue;
          if (["h3_continuation_soft_video_frames", "h3_continuation_soft_video_curve", "h3_continuation_soft_audio_ms"].includes(name) && (!Boolean(widget(node,"h3_continuation_run_and_gun_enabled")?.value) || String(widget(node,"h3_continuation_run_and_gun_join")?.value || "soft_av") !== "soft_av")) continue;
          const control = makeControl(name); if (control) fields.append(control);
        }
        controls.append(heading,fields); grid.append(controls);
      } else {
        const controls = document.createElement("section"); controls.className = "h3p-functional";
        const heading = document.createElement("div"); heading.className = "h3p-functional-title"; heading.textContent = "AUTOMATIC REFMOD · QUEUE VALUES";
        const fields = document.createElement("div"); fields.className = "h3p-functional-grid";
        for (const name of ["h3_refmod_enabled", "h3_refmod_name", "h3_refmod_strength", "h3_refmod_retention", "h3_refmod_max_tokens"]) {
          const control = makeControl(name); if (control) fields.append(control);
        }
        controls.append(heading,fields); grid.append(controls);
      }
      const libraryTitle = document.createElement("div"); libraryTitle.className = "h3p-library-title"; libraryTitle.textContent = kind === "continuation" ? "SAVED CONTINUATION LATENTS" : "SAVED REFMOD LATENTS";
      const libraryNote = document.createElement("small"); libraryNote.textContent = "PURGE LATENT removes only the selected safetensors asset. Preview PNG and IAMCCS lineage/metadata manifests remain protected."; libraryTitle.append(libraryNote);
      const library = document.createElement("div"); library.className = "h3p-assets";
      const cacheDetails = document.createElement("details"); cacheDetails.className = "h3p-cache-maintenance"; cacheDetails.style.cssText = "grid-column:1/-1;margin-top:10px";
      const cacheSummary = document.createElement("summary"); cacheSummary.className = "h3p-library-title"; cacheSummary.style.cursor = "pointer"; cacheSummary.textContent = "CACHE MAINTENANCE · OPEN / PURGE";
      const cacheNote = document.createElement("small"); cacheNote.style.cssText = "display:block;padding:4px 0 8px"; cacheNote.textContent = "Collapsed by default. Cache files are grouped in dropdowns with selection toggles. Continuation latent cards and RefMod previews remain visible above.";
      const cacheLibrary = document.createElement("div"); cacheLibrary.className = "h3p-assets";
      cacheDetails.append(cacheSummary,cacheNote,cacheLibrary);
      grid.append(libraryTitle,library,cacheDetails);
      const recipes = q(".h3p-recipes"); recipes.replaceChildren();
      const reload = document.createElement("button"); reload.className = "h3p-recipe"; reload.textContent = "REFRESH LIBRARY"; reload.onclick = () => renderAssetLibrary(kind,library); recipes.append(reload);
      const reloadCaches = document.createElement("button"); reloadCaches.className = "h3p-recipe"; reloadCaches.textContent = "OPEN / REFRESH CACHES"; reloadCaches.onclick = () => { cacheDetails.open = true; renderCacheLibrary(cacheLibrary); }; recipes.append(reloadCaches);
      cacheDetails.addEventListener("toggle", () => { if (cacheDetails.open && !cacheLibrary.dataset.loaded) { cacheLibrary.dataset.loaded = "1"; renderCacheLibrary(cacheLibrary); } });
      renderAssetLibrary(kind,library); return;
    }
    if(active === 'ahead') {
      q('.h3p-section-title').textContent='AHEAD CONTROL ROOM';
      q('.h3p-section-note').textContent='Controls for the connected LatentGoAhead branch and completed checkpoints.';
      q('.h3p-recipes').replaceChildren();
      const grid=q('[data-grid]');grid.className='h3p-flow';
      const candidates=connectedNodes(node).filter(n=>nodeClass(n).startsWith('IAMCCS_MiniMaxH3LatentGoAhead'));
      mountAheadRoom(grid,candidates.length===1?candidates[0]:null);return;
    }
    const group = visibleGroups(mode).find((item) => item.id === active) || visibleGroups(mode).find((item) => item.id !== 'ahead'); q(".h3p-section-title").textContent = group.title;
    const contextualGate = group.contextual === "control" ? branchGate(node, "h3_controlnet_enabled")
      : group.contextual === "face" ? (() => {
          const swap = branchGate(node, "h3_faceswap_sam_model");
          const detailer = branchGate(node, "face_detailer_enabled");
          return { available: swap.available || detailer.available, reason: "Connect the optional SAM3 Subject Swap or Face Detailer branch to activate these controls." };
        })() : null;
    q(".h3p-section-note").textContent = group.assistant ? "Answer in order; every accepted tag updates Queue Truth immediately."
      : active === "engine_multishot" ? "JOINT retains the original native multi-shot sample. ISOLATED SHOTS keeps each camera in separate sampling/decoding windows and discards a 17-frame generated tail before exact cuts. Local shot prompts take priority; Custom Audio preserves the master timing. Native audio is independent per shot, and sampling requires more executions."
      : active === "engine_keyframe_continuous" ? "Five actual options: native TimelineDirector/AudioCon (22, 39 or 56 frames incoming-owned overlap), distinct Extended AV masked-pin/trim backend (39f SAFE preset), native Evolving with timed prompt beats, and experimental native DRIFT/PHASE. Native Delivery for first tests."
      : active === "engine_keyframe_joint" ? "Native is the stable default inside one H3 sample window. LATENT NEW is retained here only as an explicit experimental compatibility option."
      : active === "extension" ? (String(mode) === "fl2va_extended_av" ? "EXTENDED AV is active: MASKED + EXACT is locked for the production path. Chunk 1 uses Picture 1; every later take starts from a hidden RAW AV head copied from the previous delivered tail. The pinned head is trimmed before delivery, so the master is a direct butt join with no decoded crossfade." : "Choose EXTENDED AV · LONG I2V / LONG TAKE in MODE ASSISTANT to use the new RAW AV extension engine. Legacy Auto Extend remains below only for compatibility.")
      : contextualGate && !contextualGate.available ? `Optional branch is not connected. Values are preserved but muted. ${contextualGate.reason}`
      : group.contextual ? `Branch connected; controls compile for Shotboard mode ${mode}.`
      : active === "direction" ? "Legacy compatibility surface. New workflows should use the specific panel under ENGINES; saved Direct values remain serialized and loadable."
      : "Only controls relevant to this render layer are shown.";
    q(".h3p-context").innerHTML = active === "extension" ? `<b>EXTENSION ≠ CONTINUATION.</b> EXTENDED AV uses an EXTEND-style MASKED EXTEND: previous delivered tail → RAW AV pin at child head → native H3 mask → pinned head trimmed → direct master join. SAVED TAKE · CONTINUATION is a separate resume workflow.` : active === "engine_keyframe_continuous" ? `<b>ENGINE CONTRACT:</b> Native TimelineDirector keeps incoming-owned AV overlap; Extended AV trims its masked pinned head; Evolving uses timed Shotboard prompt beats. These are different planner and runtime routes. Native Delivery recommended for initial validation.` : active === "engine_multishot" ? `<b>CUT CONTRACT:</b> JOINT cuts are generated by the model. ISOLATED SHOTS cuts are placed exactly at slot boundaries, with no cross-camera latent handoff. Custom Audio preserves the original master waveform.` : `<b>Current pipeline:</b> ${mode}. Media, prompts and FPS remain stored in Shotboard. Duration is Shotboard truth alone, or Settings PRO truth synchronized into Shotboard while PRO is master.`;
    const recipes = q(".h3p-recipes"); recipes.replaceChildren();
    if (active === "scout") { renderIntegratedScout(mode); return; }
    if (group.assistant) { renderAssistant(mode); return; }
    if (active === "speed") {
      const help = document.createElement("p");
      help.textContent = "Choose ONE speed recipe. Native: quality baseline; PDD: 8-step distillation; FastH3: 6 steps; SLA: 4-step sparse-attention recipe; Fused ConvRot: baked 4-step model, no extra Turbo/PDD LoRA. Memory presets only change memory, not speed. Manual overrides are optional below.";
      recipes.append(help);
    }
    if (active === "extension" && String(mode) === "fl2va_extended_av") {
      const current = String(widget(node, "extended_av_profile")?.value || "safe_8_12gb");
      const profileBar = document.createElement("div"); profileBar.className = "h3p-recipes h3p-memory-recipes";
      [["SAFE · 8–12 GB", "safe_8_12gb"], ["BALANCED · 12–16 GB", "balanced_12_16gb"], ["QUALITY · 20–24 GB", "quality_20_24gb"], ["MAX · 32 GB+", "max_32gb_plus"], ["CUSTOM", "custom"]].forEach(([label,id]) => addRecipe(profileBar, label, `ext:${id}`, true, current === id, () => { setValue(node, "extended_av_profile", id); refresh(); }));
      recipes.append(profileBar);
    }
    if (active === "engine_multishot") {
      const isolatedCuts = String(widget(node, "h3_multishot_boundary_mode")?.value || "joint") === "isolated_shots";
      addRecipe(recipes, "ISOLATED SHOTS · CLEAN CUTS", "multishot_clean", true, isolatedCuts, () => {
        setValue(node, "h3_multishot_boundary_mode", "isolated_shots", false);
        setValue(node, "flf_join_mode", "h3_keyframe_cut", false); setValue(node, "flf_overlap_frames", 0, false); refresh();
      });
      addRecipe(recipes, "JOINT · NATIVE MULTI-SHOT", "multishot_joint", true, !isolatedCuts, () => {
        setValue(node, "h3_multishot_boundary_mode", "joint", false); refresh();
      });
      addRecipe(recipes, "CUSTOM AUDIO · PERFORMANCE / LIP-SYNC", "multishot_custom_audio", true, false, () => { setShotboardAudio(node, "h3_custom_audio_drive"); refresh(); });
      addRecipe(recipes, "H3 NATIVE AUDIO", "multishot_native_audio", true, false, () => { setShotboardAudio(node, "h3_native_generated"); refresh(); });
    }
    if (active === "engine_ref2va") {
      addRecipe(recipes, "FASTH3 OMNI · 8 STEP", "omni", true, Boolean(widget(node,"h3_omni_enabled")?.value), () => { setAssistantMode(node,"ref2va_omni"); refresh(); });
    }
    if (["engine_ref2va", "engine_ref2vid"].includes(active)) {
      const note = document.createElement("p");
      note.textContent = "Use PICTURE 1–4 loaders; FastH3 Omni also exposes Pictures 5–9, Videos 2–3 and Audio 2–3 on CineInfoH3. These roles are read from Settings PRO at execution; disabled references are not loaded. For three people choose 3 SUBJECTS. A composition or style reference keeps its own selectable role below. Video requires an active video role.";
      recipes.append(note);
      for (const count of [1, 2, 3, 4]) {
        addRecipe(recipes, `${count} SUBJECT${count === 1 ? "" : "S"}`, `ref_subjects_${count}`, true, false, () => {
          setSubjectReferenceCount(node, count); refresh();
        });
      }
    }
    if (active === "engine_true_long") {
      const note = document.createElement("p");
      note.textContent = "PIANOSEQUENZA TRUE LONG · Shared native video/audio boundary and saved PSkf checkpoints. This workflow uses only True Long. Native Delivery recommended for the visual smoke.";
      recipes.append(note);
      const context = String(widget(node,"flf_continuity_tail_frames")?.value || "39");
      [["CONTEXT 22f","22"],["CONTEXT 39f","39"]].forEach(([label,frames]) => {
        addRecipe(recipes,label,`truelong:context:${frames}`,true,context===frames,()=>{
          setValue(node,"flf_continuity_tail_frames",frames,false); refresh();
        });
      });
    }
    if (active === "engine_keyframe_continuous") {
      const checkpointNote = document.createElement("p");
      const choice = String(widget(node,"longvid_continuous_engine")?.value || "pianosequenza_1_native_tail");
      const isExtended = choice === "extended_av_continuous";
      checkpointNote.textContent = isExtended
        ? "EXTENDED AV · separate masked/pinned native AV path. Default SAFE preset: 192-frame sample, 39-frame AV context. Technical overlap is trimmed before delivery, NOT incoming-owned. Evolving timed beats are independently available under Native Evolving."
        : "TIMELINEDIRECTOR · native sampled VIDEO + AUDIO tails, editorial endpoint handoff, incoming owns 22/39/56f overlap; 1f RGB bridge only removed. Native Evolving reads actual timeline prompt beats. Native delivery recommended for initial testing.";
      recipes.append(checkpointNote);
      [["PIANOSEQUENZA 1 · NATIVE AV TAIL · RECOMMENDED","pianosequenza_1_native_tail"],
       ["EXTENDED AV · MASKED CONTINUOUS · 39f DEFAULT","extended_av_continuous"],
       ["PIANOSEQUENZA EVOLVING · TIMED BEATS · EXPERIMENTAL","pianosequenza_evolving"],
       ["PIANOSEQUENZA 2 · DRIFT · EXPERIMENTAL","pianosequenza_2_drift"],
       ["PIANOSEQUENZA 3 · PHASE · EXPERIMENTAL","pianosequenza_3_phase"]].forEach(([title,engine]) => {
        addRecipe(recipes, title, `keycont:${engine}`, true, choice === engine, () => {
          setValue(node,"longvid_continuous_engine",engine,false);
          setValue(node,"flf_continuity_audio",true,false);
          setValue(node,"flf_overlap_frames",0,false);
          setValue(node,"upscale_mode","off",false);
          setValue(node,"upscale_enabled",false,false);
          if (engine === "extended_av_continuous") {
            setValue(node,"extended_av_pin_mode","masked",false);
            setValue(node,"extended_av_mask_profile","exact",false);
            setValue(node,"extended_av_profile","safe_8_12gb",false);
          }
          // All new engine selections start with actual native AV rather than
          // inheriting an unrelated Custom Audio recipe from a previous mode.
          setShotboardAudio(node,"h3_native_generated");
          refresh();
        });
      });
      if (!isExtended) {
        const overlap = String(widget(node,"flf_continuity_tail_frames")?.value || "22");
        [["NATIVE OVERLAP 22f · HISTORICAL","22"],["NATIVE OVERLAP 39f · LONGER","39"],["NATIVE OVERLAP 56f · EXPERIMENTAL","56"]].forEach(([label,frames]) => {
          addRecipe(recipes,label,`keycont:tail:${frames}`,true,overlap===frames,()=>{setValue(node,"flf_continuity_tail_frames",frames,false);refresh();});
        });
        addRecipe(recipes, "H3 NATIVE AUDIO · AUDIOCON", "keycont_native_audio", true, false, () => { setShotboardAudio(node,"h3_native_generated"); refresh(); });
        if (choice === "pianosequenza_evolving") {
          const evolvingNote = document.createElement("p");
          evolvingNote.textContent = "EVOLVING requires Extended Prompt blocks with authored start/end times on Shotboard (or a valid H3 evolving schedule). It refuses a missing schedule instead of rendering plain Native Tail.";
          recipes.append(evolvingNote);
        }
      } else {
        const profile = String(widget(node,"extended_av_profile")?.value || "safe_8_12gb");
        [["EXTENDED SAFE · 8–12GB · 192f / 39f","safe_8_12gb"], ["EXTENDED BALANCED · 12–16GB","balanced_12_16gb"], ["EXTENDED QUALITY · 20–24GB","quality_20_24gb"], ["EXTENDED MAX · 32GB+","max_32gb_plus"], ["EXTENDED CUSTOM · USER AV GRID","custom"]].forEach(([label,id]) => {
          addRecipe(recipes,label,`keycont:ext:${id}`,true,profile===id,()=>{setValue(node,"extended_av_profile",id,false);refresh();});
        });
      }
    }
    if (active === "engine_keyframe_joint") {
      addRecipe(recipes, "NATIVE SINGLE WINDOW · STABLE", "keyjoint_native", true, !Boolean(widget(node,"keyframe_joint_latent_new")?.value), () => { setValue(node,"keyframe_joint_latent_new",false,false); refresh(); });
      addRecipe(recipes, "LATENT NEW · EXPERIMENTAL", "keyjoint_latent_new", true, Boolean(widget(node,"keyframe_joint_latent_new")?.value), () => { setValue(node,"keyframe_joint_latent_new",true,false); refresh(); });
    }
    if (active === "face_refine") {
      node.properties ||= {}; node.properties.iamccs_recipe_highlight ||= {};
      if (/^(faceq_|facesize_|facedef_)/.test(String(node.properties.iamccs_recipe_highlight.memory || ""))) {
        delete node.properties.iamccs_recipe_highlight.memory;
      }
      const presetGrid = document.createElement("div"); presetGrid.className = "h3p-face-refine-presets-grid";
      const makeGroup = (label, cls, note, entries) => {
        const groupCard = document.createElement("section"); groupCard.className = "h3p-face-refine-preset-group";
        const cap = document.createElement("div"); cap.className = "h3p-functional-title"; cap.textContent = label;
        const groupNote = document.createElement("div"); groupNote.className = "h3p-face-refine-group-note"; groupNote.textContent = note;
        const wrap = document.createElement("div"); wrap.className = `h3p-memory-recipes ${cls}`;
        const help = document.createElement("div"); help.className = "h3p-face-refine-help"; help.innerHTML = "<b>INFO</b> Choose a preset to see its production intent.";
        entries.forEach(([text, id]) => {
          const meta = FACE_REFINE_PRESET_HELP[id] || { label: text, info: "" };
          addRecipe(wrap, text, id, true, false, () => { applyFaceRefineTuning(node, id); refresh(); });
          const button = wrap.lastElementChild;
          if (button) { button.title = meta.info; button.setAttribute("aria-label", `${meta.label}. ${meta.info}`); }
        });
        const rememberedId = rememberedRecipe(recipeGroup(wrap));
        const rememberedMeta = FACE_REFINE_PRESET_HELP[rememberedId];
        if (rememberedMeta) help.innerHTML = `<b>${rememberedMeta.label}</b> — ${rememberedMeta.info}`;
        groupCard.append(cap, groupNote, wrap, help); presetGrid.append(groupCard);
      };
      makeGroup("QUALITY PRESETS", "h3p-face-refine-quality-recipes", "Controls face-canvas resolution, refine steps and temporal processing cost.", [["SAFE · 12 GB", "faceq_safe"], ["BALANCED", "faceq_balanced"], ["HIGH", "faceq_high"], ["MAX", "faceq_max"]]);
      makeGroup("FACE SIZE PRESETS", "h3p-face-refine-size-recipes", "Adapts crop and detector/detail scaling to the apparent face size in the shot. Multi-face mode is preserved.", [["WIDE CHARACTER", "facesize_wide"], ["MEDIUM", "facesize_medium"], ["SMALL FACES", "facesize_small"], ["TINY FACES", "facesize_tiny"]]);
      makeGroup("DEFINITION PRESETS", "h3p-face-refine-definition-recipes", "Controls how strongly detail is regenerated and blended back into the original face.", [["SOFT / PRESERVE", "facedef_soft"], ["BALANCED", "facedef_balanced"], ["DEFINED", "facedef_defined"], ["STRONG", "facedef_strong"]]);
      const stack = document.createElement("div"); stack.className = "h3p-face-refine-stack";
      const stackText = document.createElement("span");
      const labelFor = (group, fallback = "NONE") => {
        const id = rememberedRecipe(group);
        return FACE_REFINE_PRESET_HELP[id]?.label || fallback;
      };
      stackText.innerHTML = `<b>ACTIVE PRESET STACK</b> · Quality: ${labelFor("face_refine_quality")} · Face size: ${labelFor("face_refine_size")} · Definition: ${labelFor("face_refine_definition")}`;
      const reset = document.createElement("button"); reset.type = "button"; reset.className = "h3p-face-refine-reset"; reset.textContent = "RESET PRESETS · CURRENT DEFAULTS";
      reset.title = "Clear all three Face Refinement preset selections and restore the current IAMCCS default tuning. Face Refinement enable/mode/max-faces settings are preserved.";
      reset.onclick = () => resetFaceRefinePresets(node);
      stack.append(stackText, reset);
      recipes.append(presetGrid, stack);
    }
    const recipeSet = active === "memory" ? [["≤ 8–12 GB · 124F / 2048 ROWS", "vram8"], ["12–16 GB · 209F / 4096 ROWS", "vram12"], ["16–24 GB · 294F / 8192 ROWS", "vram16"], ["24 GB+ · 362F / 16384 ROWS", "vram24"]] : active === "speed" ? [["NATIVE QUALITY · 20 STEP", "native"], ["PDD · 8 STEP", "pdd"], ["FASTH3 · 6 STEP", "fasth3"], ["SLA · 4 STEP", "sla"], ["FUSED FAST · T2VA ONLY", "fused"]] : active === "control" ? [["ENABLE + SELECT INSTALLED MODEL", "control"]] : active === "face" ? [["SAM3 SUBJECT SWAP · SAFE DEFAULTS", "face"]] : active === "face_refine" ? [["FACE REFINE · TOGGLE", "face-refine"]] : active === "finish" ? [["OPEN-H3 REFINEMENT · 3 STEP", "open-h3-refine"], ["NATIVE DELIVERY", "native-delivery"], ["H3 DETAIL · 1-STEP REFINE", "ultimate-tiled"], ["PIXEL SAFE · LOW VRAM", "pixel-safe"], ["H3 2-PASS UPRES", "upres"], ["H3 PIXEL REFINE", "h3-pixel"], ["LTX 2.3 MASTER", "ltx-master"], ["LTX 2.3 PER-SHOT", "ltx-per-shot"], ["RTX FINAL", "rtx-final"]] : [];
    recipeSet.forEach(([label, id]) => {
      const deliveryRecipe = ["open-h3-refine", "native-delivery", "ultimate-tiled", "pixel-safe", "upres", "h3-pixel", "ltx-master", "ltx-per-shot", "rtx-final", "face-refine"].includes(id);
      const deliveryAvailable = id === "open-h3-refine" ? branchGate(node, "h3_open_refine_enabled").available : id === "upres"
        ? branchGate(node, "h3_upres_model_name").available && deliveryModeAvailable(node, "h3_fast_latent_2pass")
        : id === "ultimate-tiled" ? deliveryModeAvailable(node, "h3_ultimate_tiled")
        : id === "h3-pixel" ? deliveryModeAvailable(node, "h3_pixel_refine")
        : id === "pixel-safe" ? deliveryModeAvailable(node, "pixel_tiled_low_vram")
        : id === "ltx-master" ? deliveryModeAvailable(node, "ltx23")
        : id === "ltx-per-shot" ? deliveryModeAvailable(node, "ltx23_per_chunk")
        : id === "rtx-final" ? deliveryModeAvailable(node, "rtx_final")
        : true;
      addRecipe(recipes, label, id, deliveryRecipe ? deliveryAvailable : accelerationAvailable(node,id,mode));
    });
    if (active === "overview" || active === "swap_lora") {
      normalize2StageResolutionLink(node, mode);
  normalizePianosequenzaStageControls(node);
      const format = document.createElement("select"); format.className = "h3p-resolution";
      const nativeResolution = `${Number(widget(node,"width")?.value || 0)}x${Number(widget(node,"height")?.value || 0)}`;
      format.add(new Option("CUSTOM · keep current", "custom"));
      H3_NATIVE_RESOLUTION_PRESETS.forEach(([label,value]) => format.add(new Option(label,value)));
      format.value = [...format.options].some((option) => option.value === nativeResolution) ? nativeResolution : "custom";
      format.onchange = () => { if (format.value === "custom") return; const [width,height] = format.value.split("x").map(Number); setValue(node,"width",width,false); setValue(node,"height",height,false); setValue(node,"image_width",width,false); setValue(node,"image_height",height,false); document.dispatchEvent(new CustomEvent("iamccs:h3-settings-changed", {detail:{source_node_id:node.id,recipe:"resolution"}})); refresh(true); };
      recipes.prepend(format);
      node.properties ||= {};
      const selectedRoute = Boolean(widget(node,"upscale_enabled")?.value) ? String(widget(node,"upscale_mode")?.value || "off") : "off";
      const stage = document.createElement("div"); stage.className = "h3p-functional";
      stage.style.cssText = "display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin:0 0 10px;width:100%";
      const stageLabel = document.createElement("label"); stageLabel.style.cssText = "display:flex;align-items:center;gap:6px;color:#f0d39e;font-weight:800";
      const stageToggle = document.createElement("input"); stageToggle.type = "checkbox"; stageToggle.checked = selectedRoute !== "off";
      stageToggle.setAttribute("aria-label","Enable upscale"); stageLabel.append(stageToggle,"UPSCALE");
      const route = document.createElement("select"); route.className = "h3p-resolution";
      route.setAttribute("aria-label","Upscale engine");
      for (const [key,label] of H3_DELIVERY_MODES) {
        const option = new Option(label,key); option.disabled = !deliveryModeAvailable(node,key);
        option.title = option.disabled ? deliveryBranchGate(node,key).reason || "Unavailable in this mode or backend" : label;
        route.add(option);
      }
      route.value = selectedRoute;
      route.onchange = () => { if (!selectDeliveryMode(node,route.value)) route.value = selectedRoute; refresh(true); };
      stageToggle.onchange = () => {
        const remembered = node.properties.iamccs_delivery_route || (String(widget(node,"upscale_mode")?.value) !== "off" ? widget(node,"upscale_mode").value : "h3_ultimate_tiled");
        if (!selectDeliveryMode(node,stageToggle.checked ? remembered : "off")) stageToggle.checked = false;
        refresh(true);
      };
      const linkedH3 = ["h3_fast_latent_2pass","h3_ultimate_tiled","h3_latent_upres","h3_pixel_refine"].includes(selectedRoute);
      const stagePair = document.createElement("select"); stagePair.className = "h3p-resolution";
      stagePair.setAttribute("aria-label",linkedH3 ? "Linked native and H3 upscale resolutions" : "Upscale output resolution");
      const nativeW = Number(widget(node,"width")?.value || 960), nativeH = Number(widget(node,"height")?.value || 576);
      const targetW = Number(widget(node,"upscale_width")?.value || nativeW*2), targetH = Number(widget(node,"upscale_height")?.value || nativeH*2);
      let current = `${targetW}x${targetH}`;
      if (linkedH3) {
        H3_FAST_2PASS_PAIRS.forEach(([label,lw,lh,hw,hh]) => stagePair.add(new Option(label,`${lw}x${lh}->${hw}x${hh}`)));
        current = `${nativeW}x${nativeH}->${targetW}x${targetH}`;
      } else {
        const grid = deliveryResolutionGrid(selectedRoute);
        const targets = [...H3_NATIVE_RESOLUTION_PRESETS,["H · 4K UHD · 3840 × 2160","3840x2160"],["V · 4K UHD · 2160 × 3840","2160x3840"],
          ["H · SCOPE · 3840 × 1600","3840x1600"],["V · SCOPE · 1600 × 3840","1600x3840"],
          ["H · FHD · 1920 × 1080","1920x1080"],["V · FHD · 1080 × 1920","1080x1920"],
          ["H · 4K grid · 3840 × 2176","3840x2176"],["V · 4K grid · 2176 × 3840","2176x3840"]];
        const seen = new Set();
        for (const [label,key] of targets) {
          const [w,h] = key.split("x").map(Number);
          if (seen.has(key) || w%grid || h%grid || w>7680 || h>4320 || w<nativeW || h<nativeH || (w>=h)!==(nativeW>=nativeH)) continue;
          seen.add(key); stagePair.add(new Option(label,key));
        }
      }
      if (![...stagePair.options].some(option => option.value === current)) stagePair.add(new Option(`CUSTOM · ${current}`,current));
      stagePair.value = current; stagePair.disabled = selectedRoute === "off";
      stagePair.onchange = () => {
        const values = stagePair.value.split(/->|x/).map(Number);
        if (linkedH3 && values.length === 4) {
          const [lw,lh,hw,hh] = values;
          for (const [name,value] of Object.entries({width:lw,height:lh,image_width:lw,image_height:lh,upscale_width:hw,upscale_height:hh})) setValue(node,name,value,false);
          node.properties.iamccs_fast_2pass_pair = stagePair.value;
        } else if (!linkedH3 && values.length === 2) {
          setValue(node,"upscale_width",values[0],false); setValue(node,"upscale_height",values[1],false);
        }
        normalizeDeliveryResolution(node,selectedRoute);
        document.dispatchEvent(new CustomEvent("iamccs:h3-settings-changed",{detail:{source_node_id:node.id,recipe:"delivery_resolution"}}));
        refresh(true);
      };
      const stageNote = document.createElement("small"); stageNote.style.cssText = "flex-basis:100%;color:#91a0ad;font-size:9px";
      stageNote.textContent = selectedRoute === "off" ? "Choose an upscale engine. Grey entries need their connected delivery branch."
        : `${deliveryResolutionGrid(selectedRoute)} px output grid. ${linkedH3 ? "Linked LOW → HIGH updates the native H3 canvas and upscale target." : "Output size changes only delivery; native generation stays at its selected canvas."} Grid compatibility does not guarantee VRAM capacity.`;
      stage.append(stageLabel,route,stagePair,stageNote); format.after(stage);
    }
    const grid = q("[data-grid]"); grid.className = "h3p-grid"; grid.replaceChildren();
    if (mode === "v2va_face_swap_lora" && ["swap_lora", "overview"].includes(active)) swapDuration.render(grid);
    let names = fieldNames(group, mode);
    if (active === "finish") names = [...new Set([...names, ...OPEN_H3_REFINE_FIELDS.filter(name => widget(node, name))])];
    if (active === "swap_lora" && !names.includes("duration_seconds")) names.unshift("duration_seconds");
    if (active === "overview" && (mode !== "longvid_guides" || String(widget(node,"upscale_mode")?.value) === "h3_fast_latent_2pass")) {
      names = names.filter((name) => !["upscale_link_to_native", "upscale_link_factor"].includes(name));
    }
    if (active === "direction") {
      const hd = String(widget(node, "longvid_terminal_endpoint_mode")?.value || "").toLowerCase() === "pianosequenza_hd";
      const multiStageOn = Boolean(widget(node, "longvid_pianosequenza_2stage_enabled")?.value);
      if (!multiStageOn || hd) names = names.filter((name) => name !== "longvid_pianosequenza_stage_profile");
    }
    if (active === "speed") {
      const toggle = document.createElement("label"), check = document.createElement("input");
      check.type = "checkbox"; check.checked = Boolean(node.properties?.iamccs_speed_manual);
      check.onchange = () => { node.properties ||= {}; node.properties.iamccs_speed_manual = check.checked; refresh(); };
      toggle.append(check, " Show advanced/manual speed controls"); recipes.append(toggle);
      if (!check.checked) {
        const selected = selectedAcceleration(node);
        const fields = {omni: ["h3_omni_enabled", "h3_clipproj_profile", "h3_clipproj_load_mode"], native: [], pdd: ["pdd_lora_name"], fasth3: ["turbo_lora_name"], sla: ["turbo_lora_name", "h3_sla_sparsity"], fused: ["fused_turbo_model_name"]}[selected] || ["turbo_mode", "turbo_lora_name"];
        names = names.filter(name => ["acceleration", "h3_attention_backend", "h3_veda_enabled", "h3_veda_predictor", "h3_veda_verbose", ...fields, "secondary_lora_enabled", "secondary_lora_name", "secondary_lora_strength", "secondary_lora_continuation_only"].includes(name));
      }
    }
    let layout = FUNCTIONAL_LAYOUT[active];
    if (active === "advanced") {
      const bucket = (name) => name.startsWith("h3_r40_") ? "SEED SCOUT" : name.startsWith("ltx_") ? "LTX DELIVERY" : name.startsWith("v2v_") ? "SOURCE VIDEO" : name.startsWith("flf_") || name.startsWith("motion_context_") ? "CONTINUITY" : name.includes("memory") || name.includes("device") || name.includes("vram") ? "MEMORY & RUNTIME" : "OTHER TECHNICAL";
      layout = [...new Set(names.map(bucket))].map((title) => [title, names.filter((name) => bucket(name) === title)]);
    }
    if (layout) {
      const used = new Set();
      layout.forEach(([title, fields]) => { const relevant = fields.filter((name) => names.includes(name)); if (!relevant.length) return; const section = document.createElement("section"); section.className = "h3p-functional"; section.innerHTML = `<div class="h3p-functional-title">${title}</div>`; if (relevant.includes("h3_open_refine_scale")) section.append(openH3ResolutionSummary(node)); const inner = document.createElement("div"); inner.className = "h3p-functional-grid"; relevant.forEach((name) => { used.add(name); const control = makeControl(name); if (control) inner.append(control); }); section.append(inner); grid.append(section); });
      const remaining = names.filter((name) => !used.has(name)); if (remaining.length) { const section = document.createElement("section"); section.className = "h3p-functional"; section.innerHTML = '<div class="h3p-functional-title">OTHER</div>'; const inner = document.createElement("div"); inner.className = "h3p-functional-grid"; remaining.forEach((name) => { const control = makeControl(name); if (control) inner.append(control); }); section.append(inner); grid.append(section); }
    } else names.forEach((name) => { const control = makeControl(name); if (control) grid.append(control); });
    if (active === "direction" && ["keyframe_joint_native", "latent_go_ahead"].includes(String(mode))) {
      appendLatentGoAheadProxyControls(node, grid);
    }
    if (active === "direction" && String(mode) === "longvid_guides") {
      const strategy = document.createElement("section"); strategy.className = "h3p-functional";
      strategy.innerHTML = '<div class="h3p-functional-title">IAMCCS ADAPTIVE GUIDE WINDOWS</div>';
      const inner = document.createElement("div"); inner.className = "h3p-recipes h3p-guide-window-recipes";
      const selectedGuideWindow = selectedGuideWindowStrategy(node);
      addRecipe(inner, "362F · MACRO", "guide_macro_362", true, selectedGuideWindow === "guide_macro_362", () => applyGuideWindowStrategy(node, "guide_macro_362"));
      addRecipe(inner, "209F + 22F TAIL · HIGH-RES SAFE", "guide_safe_209", true, selectedGuideWindow === "guide_safe_209", () => applyGuideWindowStrategy(node, "guide_safe_209"));
      addRecipe(inner, "124F + 22F TAIL · ULTRA SAFE", "guide_safe_124", true, selectedGuideWindow === "guide_safe_124", () => applyGuideWindowStrategy(node, "guide_safe_124"));
      const note = document.createElement("small"); note.textContent = "Keeps the global guide clock. SAFE profiles reduce only the internal H3 window and use the existing native latent-tail A/B carry; no decoded crossfade is enabled.";
      strategy.append(inner, note); grid.append(strategy);
    }
  }
  function renderTruth(mode) {
    const assetIssues = selectedH3AssetCompatibility(node, mode);
    const values = [["Sampling recipe", samplingRecipeLabel(node)], ["Generation authority", node.properties?.iamccs_settings_master ? "SETTINGS PRO" : "SHOTBOARD / INITIAL"], ["Duration authority", node.properties?.iamccs_settings_master ? `SETTINGS PRO · ${widget(node,"duration_seconds")?.value ?? "—"} s · synced to Shotboard` : `SHOTBOARD · ${widget(node,"duration_seconds")?.value ?? "—"} s`], ["Mode", `${mode} · Settings PRO when connected`], ["Editorial authority", "SHOTBOARD · prompts, guides, media, timeline and audio"], ["Asset compatibility", assetIssues.some(([kind]) => kind === "error") ? "INCOMPATIBLE" : assetIssues.length ? "REVIEW" : "COMPATIBLE"], ["VRAM preset", friendly(widget(node, "performance_profile")?.value)], ["Memory contract", `${widget(node,"motion_context_window_frames")?.value ?? "—"} frames · ${widget(node,"h3_exact_chunk_rows")?.value ?? "—"} rows`], ["Guide window", String(mode) === "longvid_guides" ? selectedGuideWindowStrategy(node) : "N/A"], ["Extension engine", String(mode) === "fl2va_extended_av" ? `${friendly(widget(node,"extended_av_profile")?.value)} · MASKED + EXACT · TRIM PINNED HEAD` : "OFF"], ["Sampling / legacy profile", friendly(widget(node, "acceleration")?.value)], ["Speed adapter (Turbo / FastH3 / SLA)", String(widget(node, "turbo_mode")?.value || "off") === "off" && !["fasth3_dense_6step", "h3_sla"].includes(String(widget(node, "acceleration")?.value || "")) ? "OFF" : (widget(node, "turbo_lora_name")?.value || "MISSING")], ["PDD LoRA", widget(node,"pdd_lora_name")?.value || "OFF"], ["Fused model", widget(node, "fused_turbo_model_name")?.value || "OFF"], ["Effective sampling", `${widget(node, "steps")?.value ?? "—"} steps · ${widget(node, "sampler_name")?.value ?? "—"} · ${widget(node, "scheduler")?.value ?? "—"}`], ["Canvas", `${widget(node, "width")?.value ?? "—"} × ${widget(node, "height")?.value ?? "—"}`], ["ControlNet", widget(node, "h3_controlnet_name")?.value || "OFF"], ["Delivery", widget(node, "upscale_enabled")?.value ? widget(node, "upscale_mode")?.value : "NATIVE"], ["Video Editor", Boolean(widget(node,"h3_editor_enabled")?.value) ? "ON · post-master" : "OFF · direct joined master"]];
    values.splice(1,0,["Attention & memory",attentionLabel(node)]);
    if (widget(node,"h3_open_refine_enabled")?.value) values.find(([label]) => label === "Delivery")[1] = `OPEN-H3 REFINEMENT · ${widget(node,"h3_open_refine_scale")?.value ?? 2}× · ${widget(node,"h3_open_refine_sigma_preset")?.value || "gentle_3step"}`;
    values.push(["VEDA", widget(node,"h3_veda_enabled")?.value ? `ON · ${widget(node,"h3_veda_predictor")?.value || "auto"}` : "OFF"]);
    const list = q(".h3p-truth-list"); list.replaceChildren(); values.forEach(([label, value]) => { const row = document.createElement("div"); row.className = "h3p-truth-row"; const caption = document.createElement("span"), content = document.createElement("b"); caption.textContent = label; content.textContent = String(value ?? "—"); row.append(caption, content); list.append(row); });
    const issueList = warnings(mode), health = q(".h3p-health"); health.className = `h3p-health ${issueList.some(([kind]) => kind === "error") ? "error" : issueList.some(([kind]) => kind === "warn") ? "warn" : ""}`; health.innerHTML = issueList.map(([, message]) => `• ${message}`).join("<br>");
  }
  let dropdownRefreshPending = false;
  for (const name of ["pointerdown", "mousedown"]) {
    root.addEventListener(name, (event) => {
      if (event.target?.closest?.("select")) event.stopPropagation();
    });
  }
  root.addEventListener("change", (event) => {
    if (!event.target?.matches?.("select")) return;
    // Wait for the control's handler to save its value before rebuilding it.
    window.setTimeout(() => {
      if (root.isConnected && dropdownRefreshPending) refresh(true);
    }, 0);
  }, true);
  root.addEventListener("focusout", () => {
    if (dropdownRefreshPending) window.setTimeout(() => {
      if (root.isConnected && dropdownRefreshPending) refresh();
    }, 0);
  });
  function refresh(afterDropdownSelection = false) {
    // Keep every native picker mounted during synchronization, including
    // model, sampler and engine selectors outside the resolution controls.
    if (afterDropdownSelection !== true && root.contains(document.activeElement)
        && document.activeElement?.matches?.("select")) {
      dropdownRefreshPending = true;
      return;
    }
    dropdownRefreshPending = false;
    const scrollTop = q(".h3p-main").scrollTop;
    let mode = shotboardMode(node);
    if (String(mode) === "fl2va_extended_av") normalizeExtendedAvProductionContract(node);
    mirrorShotboardAuthority(node);
    mode = shotboardMode(node);
    node._iamccsSettingsProConnectedClasses = connectedNodes(node).map(nodeClass);
    node._iamccsSettingsProBranchSignature = [...node._iamccsSettingsProConnectedClasses].sort().join("|");
    lastMode = mode; q("[data-mode]").textContent = mode.toUpperCase(); renderRail(mode); renderMain(mode); renderTruth(mode);
    q(".h3p-main").scrollTop = scrollTop;
  }
  node._iamccsSettingsProRefresh = refresh;
  const onScoutExecuted = (event) => {
    if (active !== "scout") return;
    const rawId = event?.detail?.display_node ?? event?.detail?.node ?? event?.detail?.node_id;
    const executed = rawId == null ? null : (node.graph?.getNodeById?.(Number(rawId)) || node.graph?._nodes_by_id?.[rawId]);
    const klass = nodeClass(executed);
    if (klass === "PreviewImage" || klass === "IAMCCS_MiniMaxH3SeedCandidateR40" || klass === "IAMCCS_MiniMaxH3SeedPreviewR40") {
      window.setTimeout(() => node._iamccsSettingsProRefresh?.(), 60);
    }
  };
  api.addEventListener("executed", onScoutExecuted);
  node._iamccsScoutExecutedListener = onScoutExecuted;
  const dom = node.addDOMWidget("H3 Settings PRO", "iamccs_h3_settings_pro_panel", root, { serialize: false }); dom.computeSize = (width) => [Math.max(1080, Number(width || 1080)), 680];
  node.setSize?.([1140, 750]); node.color = "#33291d"; node.bgcolor = "#0d131a"; refresh(); node._iamccsSettingsProTimer = window.setInterval(() => {
    swapDuration.tick();
    const mode = shotboardMode(node);
    const branchSignature = connectedNodes(node).map(nodeClass).sort().join("|");
    if (mode !== lastMode || branchSignature !== node._iamccsSettingsProBranchSignature) refresh();
  }, 800);
  const removed = node.onRemoved; node.onRemoved = function () {
    swapDuration.dispose();
    window.clearInterval(this._iamccsSettingsProTimer);
    r45ClearScoutPreviewTimers();
    if (this._iamccsScoutExecutedListener) api.removeEventListener("executed", this._iamccsScoutExecutedListener);
    document.removeEventListener("fullscreenchange", onSettingsFullscreenChange);
    editorRequestedOpen = false;
    if (document.fullscreenElement === root) document.exitFullscreen?.().catch?.(() => {});
    closeMonitorFallback();
    return removed?.apply(this, arguments);
  };
}

app.registerExtension({
  name: "IAMCCS.H3SettingsPro.UI",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (String(nodeData?.name || nodeData?.class_type || "") !== NODE_TYPE) return;
    // ComfyUI implicitly adds a seed-control widget when the flag is absent.
    // PRO already persists its compatibility slot and owns seed_policy.
    const seedOptions = nodeData.input?.required?.seed?.[1];
    if (seedOptions) seedOptions.control_after_generate = false;
    const created = nodeType.prototype.onNodeCreated; nodeType.prototype.onNodeCreated = function () { const result = created?.apply(this, arguments); installAssetOptions(this, nodeData); window.setTimeout(() => { mount(this); }, 0); return result; };
    const configured = nodeType.prototype.onConfigure; nodeType.prototype.onConfigure = function (info) {
      const result = configured?.apply(this, arguments);
      installAssetOptions(this, nodeData);
      restoreNamedValues(this, info, nodeData);
      normalizeSeedPolicy(this);
      normalizeInstalledSpeedAsset(this);
      normalizeSpeedPresetContract(this);
      normalize2StageResolutionLink(this);
      normalizePianosequenzaStageControls(this);
      installPianosequenzaStageQueueGuards(this);
      window.setTimeout(() => { mount(this); this._iamccsSettingsProRefresh?.(); }, 0);
      return result;
    };
    const serialized = nodeType.prototype.onSerialize; nodeType.prototype.onSerialize = function (info) {
      const result = serialized?.apply(this, arguments);
      normalizeSeedPolicy(this);
      normalizeInstalledSpeedAsset(this);
      normalizeSpeedPresetContract(this);
      normalize2StageResolutionLink(this);
      normalizePianosequenzaStageControls(this);
      installPianosequenzaStageQueueGuards(this);
      serializeNamedValues(this, info, nodeData);
      return result;
    };
  },
});


// IAMCCS LongVid Latent Tail A/B v1.4 queue-truth sync.
// This is deliberately additive: normal Settings PRO behavior is untouched.
if (!globalThis.__iamccsLongVidLatentTailABV14) {
  globalThis.__iamccsLongVidLatentTailABV14 = true;
  document.addEventListener("iamccs:h3-settings-changed", (event) => {
    try {
      const detail = event?.detail || {};
      if (detail.field !== "flf_continuity_mode") return;
      const source = app.graph?.getNodeById?.(detail.source_node_id);
      if (!source || (source.comfyClass !== NODE_TYPE && source.type !== NODE_TYPE)) return;
      const mode = String(widget(source, "flf_continuity_mode")?.value || "");
      if (mode !== "longvid_latent_tail_experimental") return;

      const board = linkedShotboard(source);
      if (!board) return;

      const boardMode = widget(board, "flf_continuity_mode");
      if (boardMode) {
        boardMode.value = mode;
        try { boardMode.callback?.(mode); } catch {}
      }

      const timeline = widget(board, "timeline_data");
      if (timeline) {
        let data = {};
        try { data = JSON.parse(String(timeline.value || "{}")); } catch {}
        data.flf_continuity_mode = mode;
        if (!data.h3_saved_settings || typeof data.h3_saved_settings !== "object") {
          data.h3_saved_settings = {};
        }
        data.h3_saved_settings.flf_continuity_mode = mode;
        timeline.value = JSON.stringify(data);
        try { timeline.callback?.(timeline.value); } catch {}
      }
      board.setDirtyCanvas?.(true, true);
      app.graph?.change?.();
      console.info("[IAMCCS] LongVid Latent Tail A/B Queue Truth synced.");
    } catch (error) {
      console.warn("[IAMCCS] LongVid Latent Tail A/B Queue Truth sync skipped:", error);
    }
  });
}
