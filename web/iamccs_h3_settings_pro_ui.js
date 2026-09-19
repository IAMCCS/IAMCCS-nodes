// SPDX-License-Identifier: GPL-3.0-or-later
import { app } from "/scripts/app.js";
import { mountAheadRoom } from './iamccs_ahead_control_room.js';
import { rigMedia } from './iamccs_h3_rig.js';

const NODE_TYPE = "IAMCCS_ShotboardH3SettingsPro";
const SHOTBOARD_TYPE = "IAMCCS_MiniMaxH3ShotPlanner";
const BRIDGE_TYPES = new Set(["IAMCCS_CineH3Input", "IAMCCS_CineH3FunControlInput", "IAMCCS_MiniMaxH3FunControlInput"]);
const SHOTBOARD_OWNED = new Set(["duration_seconds", "task_mode"]);
const AUTO_IMPORT_BLOCKED = new Set([
  "global_prompt", "timeline_data", "image_paths", "duration_seconds", "frame_rate", "audio_mode",
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
]);
// These widgets remain in the exact standard Settings schema for saved-workflow
// compatibility.  They are mirrored/editorial aliases which the standard
// compiler intentionally does not publish to CineLinX, so the PRO surface does
// not present them as a second source of truth.
const COMPATIBILITY_ONLY = new Set([
  "frame_rate", "guide_policy", "min_guide_gap_seconds", "max_guides", "default_force",
  "promptrelay_epsilon", "ltx_round_mode", "image_width", "image_height",
  "image_resize_method", "image_multiple_of", "img_compression",
]);
const GROUPS = [
  { id: "ahead", label: "AHEAD CONTROL ROOM", title: "LatentGoAhead seam controls", fields: [] },
  { id: "assistant", label: "MODE ASSISTANT", title: "Guided setup", assistant: true, fields: [] },
  { id: "overview", label: "1 · NATIVE", title: "Native H3 canvas", fields: ["width", "height", "upscale_link_to_native", "upscale_link_factor", "reference_resize_policy", "reference_resize_megapixels", "reference_resize_filter", "prompt_mapping"] },
  { id: "audio", label: "2 · AUDIO", title: "Audio authority", fields: ["audio_mode", "reference_audio_role", "voice_reference_picture_index"] },
  { id: "memory", label: "3 · MEMORY", title: "VRAM preset", fields: ["performance_profile", "motion_context_window_frames", "text_encoder_device", "h3_exact_profile", "h3_exact_chunk_rows", "h3_exact_precision_mode", "h3_exact_qkv_streaming", "h3_exact_attention_memory", "h3_clipproj_profile", "h3_clipproj_load_mode", "vram_clean_before_decode"] },
  { id: "sampling", label: "4 · SAMPLE", title: "Native H3 sampling", fields: ["seed", "seed_policy", "seed_stride", "steps", "sampler_name", "scheduler", "denoise", "shift_video", "shift_audio"] },
  { id: "speed", label: "5 · SPEED", title: "Acceleration recipe", fields: ["acceleration", "turbo_mode", "turbo_lora_name", "turbo_strength", "turbo_sampler_mode", "fused_turbo_model_name", "fused_turbo_sigma_preset", "pdd_lora_name", "pdd_strength", "secondary_lora_enabled", "secondary_lora_name", "secondary_lora_strength", "ref_image_size", "sol_conditioning", "spectrum_profile", "h3_sla_sparsity", "h3_sla_dense_last_steps"] },
  { id: "direction", label: "6 · DIRECT", title: "Mode-specific contract", fields: ["reference_role_1", "reference_role_2", "reference_role_3", "reference_role_4", "reference_video_role", "v2v_guide_mode", "v2v_source_range_policy", "v2v_source_offset_seconds", "v2v_source_fit", "v2v_source_end_policy", "v2v_audio_pairing", "flf_join_mode", "flf_overlap_frames", "flf_continuity_mode", "flf_continuity_tail_frames", "flf_continuity_audio", "longvid_guide_window_policy", "keyframe_joint_latent_new", "longvid_terminal_endpoint_mode", "longvid_pianosequenza_2stage_enabled"] }, // IAMCCS_LONGVID_ENDPOINT_STRATEGIES_V1
  { id: "control", label: "CONTROLNET", title: "H3 Fun ControlNet", contextual: "control", fields: ["h3_controlnet_enabled", "h3_controlnet_name", "h3_controlnet_kind", "h3_controlnet_strength", "h3_controlnet_start_percent", "h3_controlnet_end_percent", "h3_controlnet_frame_scope", "h3_controlnet_end_policy"] },
  { id: "face", label: "FACE SWAP", title: "Face Swap", contextual: "face", fields: ["h3_faceswap_sam_model", "h3_faceswap_birefnet_model", "h3_faceswap_mask_prompt", "h3_faceswap_threshold", "h3_faceswap_objects", "h3_faceswap_cleanup_threshold", "h3_faceswap_cleanup_shrink", "h3_faceswap_cleanup_min_frames", "h3_faceswap_cleanup_edge_grow", "h3_faceswap_crop_scale", "h3_faceswap_crop_megapixels", "h3_faceswap_grow_spatial", "h3_faceswap_grow_temporal", "h3_faceswap_feather"] },
  { id: "face_refine", label: "FACE REFINE", title: "Face Refinement", fields: ["face_detailer_enabled", "face_detailer_profile", "face_detailer_use_sam_mask"] },
  { id: "scout", label: "7 · SCOUT", title: "Candidate seed scout", fields: ["h3_r40_seed_scout_enabled", "h3_r40_candidate_count", "h3_r40_seed_stride", "h3_r40_preview_max_frames", "h3_r40_sparse_enabled", "h3_r40_sparse_video_budget", "h3_r40_sparse_denser_edges"] },
  { id: "finish", label: "8 · OUTPUT", title: "Delivery", fields: ["upscale_mode", "upscale_enabled", "upscale_width", "upscale_height", "upscale_prompt", "upscale_sage", "upscale_seed_offset", "wan_upscale_denoise", "ltx_seam_safe", "ltx_detailer_enabled", "ltx_detailer_lora_name", "ltx_detailer_strength", "ltx_4k_enabled", "ltx_4k_quality", "ltx_looper_temporal_tile_size", "ltx_looper_temporal_overlap", "ltx_looper_guiding_strength", "ltx_looper_overlap_strength", "ltx_looper_cond_image_strength", "ltx_looper_horizontal_tiles", "ltx_looper_vertical_tiles", "ltx_looper_spatial_overlap", "h3_upres_model_name", "h3_upres_precision", "h3_upres_device", "h3_upres_keep_models_resident", "h3_upres_steps", "h3_upres_denoise", "h3_upres_sampler", "h3_upres_scheduler", "h3_upres_temporal_chunk", "h3_upres_temporal_overlap", "h3_upres_anchor_strength", "h3_upres_tile_width", "h3_upres_tile_height", "h3_upres_overlap_width", "h3_upres_overlap_height", "h3_upres_fade_width", "h3_upres_fade_height", "h3_upres_min_tile_size", "h3_upres_overlap_mode", "h3_upres_overlap_blend", "h3_upres_rtx_enabled", "h3_upres_rtx_quality", "h3_upres_pixel_groups", "h3_upres_window_frames", "h3_upres_window_overlap", "h3_upres_pixel_method"] },
  { id: "advanced", label: "TECHNICAL", title: "Technical controls", dynamic: true, fields: [] },
];
const ASSIGNED = new Set(GROUPS.flatMap((group) => group.fields));
const FUNCTIONAL_LAYOUT = {
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
    ["ENGINE", ["acceleration", "h3_sla_sparsity", "h3_sla_dense_last_steps"]],
    ["TURBO LORA", ["turbo_mode", "turbo_lora_name", "turbo_strength", "turbo_sampler_mode"]],
    ["PDD 8-STEP", ["pdd_lora_name", "pdd_strength"]],
    ["FUSED MODEL", ["fused_turbo_model_name", "fused_turbo_sigma_preset"]],
    ["EXACT ATTENTION & CLIPPROJ", ["h3_exact_profile", "h3_exact_chunk_rows", "h3_exact_precision_mode", "h3_exact_qkv_streaming", "h3_exact_attention_memory", "h3_clipproj_profile", "h3_clipproj_load_mode"]],
  ],
  face_refine: [
    ["FACE REFINEMENT", ["face_detailer_enabled", "face_detailer_profile", "face_detailer_use_sam_mask"]],
  ],
  finish: [
    ["OUTPUT", ["rife_mode", "upscale_enabled", "upscale_mode", "upscale_width", "upscale_height", "upscale_prompt", "upscale_sage", "upscale_seed_offset", "wan_upscale_denoise"]],
    ["LTX DELIVERY", ["ltx_seam_safe", "ltx_detailer_enabled", "ltx_detailer_lora_name", "ltx_detailer_strength", "ltx_4k_enabled", "ltx_4k_quality", "ltx_looper_temporal_tile_size", "ltx_looper_temporal_overlap", "ltx_looper_guiding_strength", "ltx_looper_overlap_strength", "ltx_looper_cond_image_strength", "ltx_looper_horizontal_tiles", "ltx_looper_vertical_tiles", "ltx_looper_spatial_overlap"]],
    ["H3 2-PASS MODEL", ["h3_upres_model_name", "h3_upres_precision", "h3_upres_device", "h3_upres_keep_models_resident"]],
    ["H3 2-PASS SAMPLING", ["h3_upres_steps", "h3_upres_denoise", "h3_upres_sampler", "h3_upres_scheduler", "h3_upres_anchor_strength"]],
    ["TEMPORAL WINDOWS", ["h3_upres_temporal_chunk", "h3_upres_temporal_overlap"]],
    ["SPATIAL TILES", ["h3_upres_tile_width", "h3_upres_tile_height", "h3_upres_overlap_width", "h3_upres_overlap_height", "h3_upres_fade_width", "h3_upres_fade_height", "h3_upres_min_tile_size", "h3_upres_overlap_mode", "h3_upres_overlap_blend"]],
    ["PIXEL REFINE", ["h3_upres_pixel_groups", "h3_upres_window_frames", "h3_upres_window_overlap", "h3_upres_pixel_method"]],
    ["RTX DELIVERY", ["h3_upres_rtx_enabled", "h3_upres_rtx_quality"]],
  ],
};
const MODE_CHOICES = [
  ["AUTO · READ CURRENT SHOTBOARD", "auto_from_shotboard", "Import the board's current generation settings once; Settings PRO then becomes the technical master while Shotboard keeps prompts, guides, images, timeline, duration and audio truth."],
  ["T2VA · TEXT ONLY", "t2va", "One native H3 shot from prompt only."],
  ["I2VA · OPENING IMAGE", "i2va", "One image per shot; multiple boxes are independent hard cuts."],
  ["FL2VA · STABLE KEYFRAMES", "fl2va_stable", "A→B, B→C with authored shared keyframes."],
  ["FL2VA · NATIVE AV CONTINUITY", "fl2va_continuous", "Carry 22/39/56 native AV frames between FL2VA chunks."],
  ["REF2VA · REFERENCES", "ref2va", "Reference blocks for identity, object or style; no temporal carry."],
  ["REF2VID · AUDIO PERFORMANCE", "ref2vid_lipsync", "Reference image plus one locked AudioBoard performance per hard-cut shot."],
  ["LONGVID · POSITIONED GUIDES", "longvid_guides", "Global timeline guides across independent legal H3 windows."],
  ["KEYFRAME JOINT · EXPERIMENTAL", "keyframe_joint_native", "All enabled images in one native H3 sample. No FLF joins or dissolves. Limited to one trained H3 window; start at low resolution. Motion remains model-dependent."],
  ["LATENTGOAHEAD · EXPERIMENTAL", "latent_go_ahead", "Continue with original AV latent history in past time. Compatible acceleration and H3 generated audio; masked audio excludes progressive sampling. Requires the LatentGoAhead branch; no pixel crossfade. Context controls are on that node."],
  ["LONG MULTI-SHOT", "longvid_motion_context", "Timed image anchors inside H3 windows, with native AV carry. Several slots can share one sample; an anchor does not lock the entire slot or guarantee a smooth transition. For separate shots use I2VA hard cuts; for connected destinations use Long Continuous Guided."],
  ["LONG CONTINUOUS GUIDED", "longvid_continuous_guided", "One evolving take: each image is the next destination and the previous native AV latent owns every opening."],
  ["FL2VA CONTINUOUS AV", "longvid_masked_loop_guided", "Stable phase-aligned full AV-latent handover between authored FL2VA destinations."],
  ["GUIDED AV LOOP · EXPERIMENTAL", "guided_av_loop_experimental", "One full AV master latent with internal masked windows and positioned guides. No outer decoded clip concatenation; experimental."],
  ["LONGVID · MULTISHOT AUDIO DRIVE", "longvid_guided_lipsync", "Per-shot guides and rebased locked AudioBoard clips."],
  ["CONTROL VIDEO", "v2va_controlnet", "Drive pose, depth or edges from video."],
  ["OBJECT SWAP", "v2va_object_swap", "Replace a tracked object in source video."],
  ["FACE SWAP", "v2va_face_swap", "Replace a tracked identity in source video."],
];
const FRIENDLY_VALUES = { // IAMCCS_LONGVID_PIANOSEQUENZA_V2_UPSTREAM_PARITY
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
  sol_adaptive_balanced: "SOL adaptive · balanced", sage_sol: "Sage + SOL hybrid",
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
  ["TEST · 416 × 288 · LOW-RES", "416x288"],
  ["H3 UP source · 640 × 384 · LIGHT", "640x384"],
  ["H3 UP source · 736 × 416 · FHD LIGHT", "736x416"],
  ["H · ≈16:9 · 768 × 448 · Draft", "768x448"],
  ["H3 UP source · 864 × 480 · DETAIL", "864x480"],
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
  ["SCOPE · ≈2.39 · 1152 × 480", "1152x480"],
  ["SCOPE · ≈2.39 · 1536 × 640 · Quality", "1536x640"],
  ["SCOPE · 2.40 · 1920 × 800 · H3 native high", "1920x800"],
  ["SCOPE · ≈2.37 · 2048 × 864 · H3 legal 2K-class", "2048x864"],
  ["V · ≈9:16 · 448 × 768 · Draft", "448x768"],
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

const widget = (node, name) => (node.widgets || []).find((item) => item?.name === name);
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
function choices(item) {
  let values = item?.options?.values;
  try { if (typeof values === "function") values = values(); } catch { values = []; }
  if (!Array.isArray(values)) values = item?.options?.options;
  if (!Array.isArray(values) && Array.isArray(item?.combo_values)) values = item.combo_values;
  return Array.isArray(values) ? values : [];
}
function human(name) { return String(name).replace(/^h3_/, "").replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase()); }
function friendly(value) { return FRIENDLY_VALUES[String(value)] || String(value || "AUTO / NONE").replaceAll("_", " "); }
function restoreNamedValues(node, info) {
  const named = info?.widgets_values_named;
  if (!named || typeof named !== "object" || Array.isArray(named)) return;
  for (const [name, value] of Object.entries(named)) {
    const item = widget(node, name);
    if (item) item.value = value;
  }
}
function serializeNamedValues(node, info) {
  if (!info || typeof info !== "object") return;
  info.widgets_values_named = Object.fromEntries(
    (node.widgets || [])
      .filter((item) => item?.name && !String(item.name).startsWith("H3 Settings PRO"))
      .map((item) => [item.name, item.value]),
  );
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

const IAMCCS_2STAGE_RESOLUTION_PAIRS = Object.freeze([
  "640x384 -> 1280x768",
  "768x448 -> 1536x896",
  "896x512 -> 1792x1024",
  "1024x576 -> 2048x1152",
  "1280x736 -> 2560x1472",
]);
function normalize2StageResolutionLink(node) {
  const toggle = widget(node, "upscale_link_to_native");
  const pair = widget(node, "upscale_link_factor");
  if (toggle && typeof toggle.value !== "boolean") toggle.value = false;
  if (pair) {
    const value = String(pair.value ?? "");
    if (!IAMCCS_2STAGE_RESOLUTION_PAIRS.includes(value)) pair.value = IAMCCS_2STAGE_RESOLUTION_PAIRS[0];
  }
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
  if (name === "keyframe_joint_latent_new") return {
    available: connectedNodes(node).some((candidate) => /^IAMCCS_MiniMaxH3LatentGoAhead/i.test(nodeClass(candidate))),
    reason: "Connect the IAMCCS LatentGoAhead branch before enabling LATENT NEW.",
  };
  if (name.startsWith("h3_controlnet_")) return {
    available: has(/CineH3FunControlInput|MiniMaxH3FunControlInput/i),
    reason: "Connect the H3 Fun Control input branch to enable this control.",
  };
  if (name.startsWith("h3_faceswap_")) return {
    available: has(/H3FaceSwapInput/i),
    reason: "Connect the H3 Face Swap input branch to enable this control.",
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
  if (name.startsWith("h3_upres_")) return {
    available: has(/LatentUpres|FastLatent2Pass|UltimateTiled|ProgressiveSpatial|UniversalFast/i),
    reason: "Connect an H3 latent-upres delivery branch to enable this control.",
  };
  return { available: true, reason: "" };
}
function modeAvailability(node, mode) {
  const classes = node._iamccsSettingsProConnectedClasses || connectedNodes(node).map(nodeClass);
  const has = (pattern) => classes.some((value) => pattern.test(value));
  const value = String(mode || "").toLowerCase();
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
  if (value === "v2va_face_swap") return {
    available: has(/H3FaceSwapInput|MiniMaxH3FaceDelivery|MiniMaxH3FaceDetailer/i),
    reason: "Connect the H3 Face branch to enable Face Swap.",
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
function mirrorShotboardAuthority(node) {
  const board = linkedShotboard(node); if (!board) return;
  const names = node.properties?.iamccs_settings_master
    ? ["duration_seconds", "frame_rate", "audio_mode"]
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
  const effectiveMode = shotboardMode(node);
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
  if (String(mode) !== "fl2va") return String(mode);
  const board = linkedShotboard(node);
  return String(widget(board || node, "flf_continuity_mode")?.value || "stable_keyframes") === "native_av_context"
    ? "fl2va_continuous" : "fl2va_stable";
}
function setAssistantMode(node, key) {
  if (String(key) === "auto_from_shotboard") return importSettingsFromShotboard(node);
  const board = linkedShotboard(node);
  node.properties ||= {};
  node.properties.iamccs_settings_master = true;
  node.properties.iamccs_auto_from_shotboard_active = false;
  const previousMode = shotboardMode(node);
  const actualMode = String(key).startsWith("fl2va_") ? "fl2va" : key;
  setShotboardMode(node, actualMode);
  if (["ref2vid_lipsync", "longvid_guided_lipsync"].includes(previousMode) && !["ref2vid_lipsync", "longvid_guided_lipsync"].includes(key)) {
    setShotboardAudio(node, "h3_native_generated");
  }
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
  // Continuous AV transports H3's complete generated AV latent. Switching
  // from LipSync must clear that mode's locked AudioBoard route so the active
  // mode and Queue Truth cannot disagree.
  if (["longvid_masked_loop_guided", "guided_av_loop_experimental", "latent_go_ahead", "keyframe_joint_native", "fl2va_stable", "fl2va_continuous"].includes(key)) setShotboardAudio(node, "h3_native_generated");
  document.dispatchEvent(new CustomEvent("iamccs:h3-settings-changed", { detail: { source_node_id: node.id, assistant_mode: key } }));
  node._iamccsSettingsProRefresh?.(); return true;
}
function modeContext(mode) { const value = String(mode).toLowerCase(); return value.includes("controlnet") ? "control" : value.includes("face_swap") ? "face" : "core"; }
function modeFamily(mode) {
  const value = String(mode).toLowerCase();
  return value.includes("ref2") || ["v2va_object_swap", "v2va_face_swap"].includes(value) ? "ref2" : "fl2";
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
function applyDeclaredTurboContract(node) {
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
  return value === "pdd_native_8step" ? "pdd" : value === "fasth3_dense_6step" ? "fasth3" : value === "h3_sla" ? "sla" : value === "matlowai_fused_turbo_manual_sigma" ? "fused" : value === "native" ? "native" : "custom";
}
function selectedMemory(node) {
  const value = String(widget(node, "performance_profile")?.value || "");
  if (value.includes("xx90")) return "vram24"; if (value.includes("xx80")) return "vram16"; if (value.includes("xx70")) return "vram12"; return "vram8";
}
// IAMCCS FACE/UPSCALE MODE ASSISTANT V2
function selectedDelivery(node) {
  if (!widget(node, "upscale_enabled")?.value) return "native-delivery";
  const mode = String(widget(node, "upscale_mode")?.value || "off");
  return ({
    h3_fast_latent_2pass: "upres",
    h3_pixel_refine: "h3-pixel",
    ltx23: "ltx-master",
    ltx23_per_chunk: "ltx-per-shot",
    rtx_final: "rtx-final",
  })[mode] || "custom-delivery";
}
function selectedFaceRefine(node) { return !!widget(node, "face_detailer_enabled")?.value; }
function deliveryModeAvailable(node, mode) {
  const values = widget(node, "upscale_mode")?.options?.values;
  return !Array.isArray(values) || values.includes(mode);
}

function applyRecipe(node, recipe) {
  const mode = shotboardMode(node), family = modeFamily(mode), set = (name, value) => setValue(node, name, value, false);
  if (recipe === "native") {
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
    resetAcceleration(set); set("pdd_lora_name", asset); set("pdd_strength", 1); set("acceleration", "pdd_native_8step");
    set("steps", 8); set("sampler_name", "euler"); set("scheduler", "simple"); set("denoise", 1); set("shift_video", 12); set("shift_audio", 3);
  } else if (recipe === "fasth3") {
    const asset = firstChoice(node, "turbo_lora_name", isFastH3LoRA); if (!asset) { alert("FastH3 is unavailable: no compatible FastH3 LoRA is installed."); return false; }
    resetAcceleration(set); set("turbo_mode", "off"); set("turbo_lora_name", asset); set("turbo_strength", 1);
    set("acceleration", "fasth3_dense_6step"); set("steps", 6); set("sampler_name", "euler"); set("scheduler", "simple"); set("denoise", 1); set("shift_video", 12); set("shift_audio", 3);
  } else if (recipe === "sla") {
    const asset = firstChoice(node, "turbo_lora_name", (name) => name.includes("sla") && name.includes(family)); if (!asset) { alert(`SLA is unavailable: no compatible ${family.toUpperCase()} SLA LoRA is installed.`); return false; }
    resetAcceleration(set); set("turbo_mode", "early_8_10"); set("turbo_lora_name", asset); set("turbo_strength", 1);
    set("acceleration", "h3_sla"); set("steps", 4); set("h3_sla_sparsity", 0.85); set("h3_sla_dense_last_steps", 0); set("sampler_name", "euler"); set("scheduler", "simple"); set("denoise", 1); set("shift_video", 6); set("shift_audio", 3); set("turbo_sampler_mode", "res_multistep_stock");
  } else if (recipe === "fused") {
    const current = String(widget(node, "fused_turbo_model_name")?.value || "");
    const asset = current || firstChoice(node, "fused_turbo_model_name", (name) => (name.includes("fused") || name.includes("turbo")) && !name.includes("ref2"));
    if (!asset) { alert("Fused Fast is unavailable: no compatible fused diffusion model is installed."); return false; }
    const convrot = asset.toLowerCase().includes("convrot");
    if (!(convrot ? ["t2va", "i2va", "fl2va", "ref2va"] : ["t2va"]).includes(shotboardMode(node))) { alert("This fused checkpoint does not support the selected mode."); return false; }
    resetAcceleration(set); set("fused_turbo_model_name", asset); set("acceleration", "matlowai_fused_turbo_manual_sigma");
    set("fused_turbo_sigma_preset", "4_step"); set("steps", 4); set("sampler_name", "res_multistep"); set("scheduler", "simple");
    set("denoise", 1); set("shift_video", 12); set("shift_audio", 3);
  } else if (recipe === "control") {
    set("h3_controlnet_enabled", true); const asset = firstChoice(node, "h3_controlnet_name", () => true); if (asset) set("h3_controlnet_name", asset);
  } else if (recipe === "face") {
    set("face_detailer_enabled", false); set("upscale_enabled", false);
  } else if (recipe === "face-refine") {
    const enabled = !Boolean(widget(node, "face_detailer_enabled")?.value);
    set("face_detailer_enabled", enabled);
    if (enabled && !String(widget(node, "face_detailer_profile")?.value || "")) set("face_detailer_profile", "balanced");
  } else if (recipe === "native-delivery") {
    set("upscale_enabled", false); set("upscale_mode", "off"); set("rife_mode", "off");
  } else if (recipe === "upres") {
    set("upscale_enabled", true); set("upscale_mode", "h3_fast_latent_2pass");
  } else if (recipe === "h3-pixel") {
    set("upscale_enabled", true); set("upscale_mode", "h3_pixel_refine");
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
  setValue(node, "acceleration", value);
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
  if (recipe === "fasth3") return Boolean(firstChoice(node, "turbo_lora_name", isFastH3LoRA));
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
      const compatible = firstChoice(node, "turbo_lora_name", isFastH3LoRA);
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
  if (node._iamccsSettingsProMounted) return; node._iamccsSettingsProMounted = true; (node.widgets || []).forEach(hideWidget);
  const root = document.createElement("div"); root.className = "iamccs-h3pro";
  root.innerHTML = `
  <style>
  .iamccs-h3pro .h3p-rail{background:linear-gradient(180deg,#17232f,#0b111a);gap:7px;overflow-y:auto}.iamccs-h3pro .h3p-tab{min-height:40px;flex-shrink:0;border:1px solid #334454;background:linear-gradient(135deg,#223140,#141d28);box-shadow:inset 0 1px #ffffff0b,0 2px 5px #0003;transition:border-color .15s,background .15s;padding:9px 12px;white-space:normal;line-height:1.25}.iamccs-h3pro .h3p-tab.active{border-color:#dec087;border-left:3px solid #f5cf8d;background:linear-gradient(100deg,#493922,#233444);box-shadow:0 0 12px #d8b26c18,inset 0 1px #fff1}.iamccs-h3pro .h3p-tab:hover{border-color:#88b6c6;color:#fff}.iamccs-h3pro .h3p-choice.active{border-left:3px solid #eac383;box-shadow:0 3px 12px #0004}.iamccs-h3pro .h3p-choice{min-height:68px;padding:12px}
  .iamccs-h3pro{height:100%;padding:12px;box-sizing:border-box;background:radial-gradient(circle at 85% 0,#273346 0,transparent 34%),linear-gradient(145deg,#090d13,#111923 62%,#0a0e14);border:1px solid #8b7046;border-radius:14px;color:#eaf0f5;font:11px Inter,Segoe UI,sans-serif;overflow:hidden}.iamccs-h3pro *{box-sizing:border-box}.h3p-head{height:48px;display:flex;align-items:center;gap:12px;border-bottom:1px solid #344253}.h3p-mark{padding:6px 10px;border:1px solid #d2a65c;border-radius:999px;background:#32281a;color:#f8d89c;font-size:9px;font-weight:900;letter-spacing:.08em}.h3p-title{font:700 17px Georgia,serif}.h3p-sub{color:#8291a0;font-size:9px}.h3p-mode{margin-left:auto;text-align:right}.h3p-mode b{display:block;color:#7ee2ad;font-size:10px}.h3p-layout{display:grid;grid-template-columns:155px minmax(500px,1fr) 260px;gap:10px;height:calc(100% - 58px);padding-top:10px}.h3p-rail,.h3p-main,.h3p-truth{min-height:0;border:1px solid #2e3a47;border-radius:10px;background:rgba(12,18,25,.88)}.h3p-rail{padding:7px;display:flex;flex-direction:column;gap:5px}.h3p-tab{height:38px;padding:0 10px;border:1px solid transparent;border-radius:7px;background:transparent;color:#94a2b0;text-align:left;font-size:9px;font-weight:850;letter-spacing:.05em;cursor:pointer}.h3p-tab:hover{background:#182330;color:#fff}.h3p-tab.active{border-color:#a98650;background:linear-gradient(90deg,#3c3020,#1c2530);color:#f3d69c}.h3p-owner{margin-top:auto;padding:10px;border-radius:8px;background:#111b24;color:#8493a2;font-size:8px;line-height:1.45}.h3p-owner strong{display:block;color:#f0c97d;margin-bottom:4px}.h3p-main{padding:12px;overflow:auto}.h3p-section-title{font:700 16px Georgia,serif;color:#f0d39e}.h3p-section-note{margin:4px 0 12px;color:#8493a2;font-size:9px}.h3p-recipes{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px}.h3p-recipe,.h3p-choice{padding:8px 10px;border:1px solid #4c5c6d;border-radius:7px;background:#1a2530;color:#d9e2e9;font-size:8px;font-weight:850;cursor:pointer}.h3p-recipe:hover,.h3p-recipe.active,.h3p-choice:hover,.h3p-choice.active{border-color:#d0a45c;color:#f6d99d;background:#2b261d;box-shadow:0 0 0 1px rgba(240,190,99,.2),0 0 12px rgba(240,190,99,.16)}.h3p-recipe[disabled]{opacity:.35;cursor:not-allowed}.h3p-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.h3p-functional{grid-column:1/-1;padding:9px;border:1px solid #344353;border-radius:10px;background:linear-gradient(145deg,#111b25,#0c141c)}.h3p-functional-title{margin:0 0 8px;color:#e3bd78;font-size:8px;font-weight:900;letter-spacing:.09em}.h3p-functional-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.h3p-resolution{height:31px;min-width:220px;border:1px solid #9b7944;border-radius:7px;background:#171f27;color:#f1d49b;padding:0 8px;font-size:8px;font-weight:850}.h3p-field{min-height:58px;padding:7px;border:1px solid #2d3945;border-radius:8px;background:#101821}.h3p-field.muted{opacity:.48;border-style:dashed;background:#0b1218}.h3p-field.muted label::after{content:" · BRANCH OFF";color:#d0a45c;font-size:7px}.h3p-field.muted input,.h3p-field.muted select{cursor:not-allowed}.h3p-field label{display:block;margin-bottom:5px;color:#9ba8b5;font-size:8px;font-weight:800}.h3p-field input,.h3p-field select{width:100%;height:29px;border:1px solid #43515e;border-radius:6px;background:#0a1118;color:#edf2f6;padding:0 7px;font-size:9px}.h3p-field input[type=checkbox]{width:18px;height:18px;accent-color:#d3a758}.h3p-field small{display:block;margin-top:4px;color:#667786;font-size:7px}.h3p-truth{padding:12px;overflow:auto}.h3p-truth h3{margin:0 0 10px;color:#f0d39e;font:700 14px Georgia,serif}.h3p-truth-row{padding:8px 0;border-bottom:1px solid #26323d}.h3p-truth-row span{display:block;color:#718190;font-size:7px;font-weight:900}.h3p-truth-row b{display:block;margin-top:3px;color:#dce5eb;font:600 9px Consolas,monospace;overflow-wrap:anywhere}.h3p-health{margin-top:10px;padding:9px;border-left:3px solid #68d69a;border-radius:6px;background:#11231c;color:#a9e6c5;font-size:8px;line-height:1.45}.h3p-health.warn{border-color:#e7a14e;background:#2a2014;color:#ffd59a}.h3p-health.error{border-color:#e86767;background:#2d1619;color:#ffb0b0}.h3p-context{margin-bottom:10px;padding:9px;border:1px solid #405064;border-radius:8px;background:#14202c;color:#a9b7c5;font-size:9px}.h3p-context b{color:#f2cf8b}.h3p-question{margin:12px 0 6px;color:#f0cf91;font-size:9px;font-weight:900;letter-spacing:.05em}.h3p-choice-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px}.h3p-choice{text-align:left}.h3p-choice span{display:block;margin-top:3px;color:#8493a2;font-weight:500;line-height:1.3}.h3p-choice[disabled]{opacity:.34;filter:saturate(.45);cursor:not-allowed;border-style:dashed}.h3p-choice[disabled] span::after{content:" · BRANCH MISSING";color:#e7a14e}.h3p-recipe.active{outline:1px solid #f2c97e;outline-offset:1px}.h3p-flow{padding:10px;border:1px solid #354455;border-radius:9px;background:#0d151e}
  .h3p-recipe.active{border-color:#d0a45c;color:#f6d99d;background:#2b261d;box-shadow:inset 0 0 0 1px #6b5330}.h3p-recipe[disabled]::after{content:" · UNAVAILABLE";color:#e7a14e}.h3p-main[data-section="memory"] .h3p-recipes,.h3p-memory-recipes{padding:9px;border:1px solid #4d79a0;border-radius:9px;background:linear-gradient(135deg,#122838,#101c2a)}.h3p-main[data-section="memory"] .h3p-recipe,.h3p-memory-recipes .h3p-recipe{border-color:#4d83ad;color:#bfe4ff}.h3p-main[data-section="speed"] .h3p-recipes,.h3p-speed-recipes{padding:9px;border:1px solid #9a7140;border-radius:9px;background:linear-gradient(135deg,#302313,#201a14)}.h3p-main[data-section="speed"] .h3p-recipe,.h3p-speed-recipes .h3p-recipe{border-color:#b0844b;color:#ffe0a9}.h3p-delivery-recipes{padding:9px;border:1px solid #5b8a69;border-radius:9px;background:#12251a}
  </style>
  <div class="h3p-head"><span class="h3p-mark">IAMCCS PRO</span><div><div class="h3p-title">H3 Settings PRO</div><div class="h3p-sub">Render compiler · one queue truth · no editorial duplication</div></div><div class="h3p-mode"><span class="h3p-sub">SHOTBOARD MODE</span><b data-mode>NOT CONNECTED</b></div></div>
  <div class="h3p-layout"><nav class="h3p-rail"></nav><main class="h3p-main"><div class="h3p-section-title"></div><div class="h3p-section-note"></div><div class="h3p-context"></div><div class="h3p-recipes"></div><div class="h3p-grid" data-grid></div></main><aside class="h3p-truth"><h3>Queue Truth</h3><div class="h3p-truth-list"></div><div class="h3p-health"></div></aside></div>`;
  const q = (selector) => root.querySelector(selector); let active = String(node.properties?.iamccs_h3_settings_pro_active_section || "assistant"), lastMode = "";
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
    const value = String(mode).toLowerCase();
    const endpoint = String(widget(node, "longvid_terminal_endpoint_mode")?.value || "hard_image").toLowerCase();
    const hd = endpoint === "pianosequenza_hd";
    if (["longvid_terminal_endpoint_mode", "longvid_pianosequenza_2stage_enabled", "longvid_pianosequenza_stage_profile"].includes(name) && value !== "longvid_guides") return false;
    if (name === "longvid_pianosequenza_stage_profile" && (hd || !Boolean(widget(node, "longvid_pianosequenza_2stage_enabled")?.value))) return false;
    if (name === "longvid_guide_window_policy" && value !== "longvid_guides") return false;
    if (name === "keyframe_joint_latent_new" && value !== "keyframe_joint_native") return false;
    if (["upscale_link_to_native", "upscale_link_factor"].includes(name) && (value !== "longvid_guides" || hd || !Boolean(widget(node, "longvid_pianosequenza_2stage_enabled")?.value))) return false;
    if (name.startsWith("v2v_") && !value.startsWith("v2va_")) return false;
    const continuityFamily = value.includes("fl2") || value.includes("longvid") || value.includes("guided_av_loop") || value === "keyframe_joint_native" || value === "latent_go_ahead";
    if (name.startsWith("flf_") && !continuityFamily) return false;
    if (name.startsWith("h3_r40_") && !value.includes("scout")) return false; return true;
  }
  function visibleGroups(mode) {
    // Keep the entry discoverable even when automatic timeline mode resolution
    // is pending. The panel itself checks for a connected continuation branch.
    return GROUPS.filter((group) => group.id === 'ahead' || group.assistant || group.dynamic || group.fields.some((name) => widget(node, name)));
  }
  function fieldNames(group, mode) {
    if (!group.dynamic) return group.fields.filter((name) => widget(node, name) && fieldRelevant(name, mode));
    return (node.widgets || []).map((item) => item?.name).filter((name) => name && !ASSIGNED.has(name) && !SHOTBOARD_OWNED.has(name) && !INTERNAL.has(name) && !COMPATIBILITY_ONLY.has(name) && fieldRelevant(name, mode));
  }
  function makeControl(name) {
    const item = widget(node, name); if (!item) return null;
    const box = document.createElement("div"); box.className = "h3p-field";
    const label = document.createElement("label"); label.textContent = human(name); box.append(label);
    let control; const options = choices(item);
    if (options.length) {
      control = document.createElement("select"); options.forEach((value) => control.add(new Option(friendly(value), String(value)))); control.value = String(item.value ?? "");
    } else if (typeof item.value === "boolean") {
      control = document.createElement("input"); control.type = "checkbox"; control.checked = Boolean(item.value);
    } else {
      control = document.createElement("input"); control.type = typeof item.value === "number" ? "number" : "text"; control.value = item.value ?? "";
      if (control.type === "number") { if (Number.isFinite(item.options?.min)) control.min = item.options.min; if (Number.isFinite(item.options?.max)) control.max = item.options.max; control.step = item.options?.step ?? "any"; }
    }
    const gate = branchGate(node, name);
    if (!gate.available) {
      control.disabled = true;
      box.classList.add("muted");
      box.title = gate.reason;
    }
    control.dataset.field = name; control.onchange = () => {
      const value = control.type === "checkbox" ? control.checked : control.type === "number" ? Number(control.value) : control.value;
      if (name === "audio_mode") setShotboardAudio(node, value);
      else if (name === "performance_profile") applyMemoryProfileChoice(node, value);
      else if (name === "acceleration") applyAccelerationChoice(node, value);
      else setValue(node, name, value);
      if (name === "fused_turbo_model_name" && selectedAcceleration(node) === "fused") applyRecipe(node, "fused");
      if (name === "fused_turbo_sigma_preset" && selectedAcceleration(node) === "fused") normalizeSpeedPresetContract(node);
      if (name === "turbo_mode" || name === "turbo_lora_name") applyDeclaredTurboContract(node);
      refresh();
    }; box.append(control);
    const guidance = name === "performance_profile"
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
    if (Boolean(widget(node, "upscale_enabled")?.value)) {
      const field = upscaleMode.includes("ltx") ? "ltx_seam_safe" : upscaleMode.includes("wan") ? "wan_upscale_denoise" : upscaleMode.startsWith("h3_") ? "h3_upres_model_name" : "";
      if (field && !branchGate(node, field).available) issues.push(["error", `${friendly(upscaleMode)} is enabled, but its delivery branch is not connected.`]);
    }
    if (modeContext(mode) === "face" && !String(widget(node, "h3_faceswap_birefnet_model")?.value || "")) issues.push(["error", "Face Swap requires BiRefNet in background_removal."]);
    issues.push(...selectedH3AssetCompatibility(node, mode));
    const endpoint = String(widget(node, "longvid_terminal_endpoint_mode")?.value || "hard_image").toLowerCase();
    if (endpoint === "pianosequenza_hd" && Boolean(widget(node, "longvid_pianosequenza_2stage_enabled")?.value)) {
      issues.push(["warn", "PIANOSEQUENZA HD owns its fixed two-stage contract; the separate Multi-Stage Spatial toggle is preserved but ignored while HD is selected."]);
    }
    if (String(mode) === "longvid_guides") {
      const pixels = Number(widget(node,"width")?.value || 0) * Number(widget(node,"height")?.value || 0);
      if (pixels > 640 * 384 && selectedGuideWindowStrategy(node) === "guide_macro_362") issues.push(["warn", "High-resolution LongVid is still using the 362F macro window. Use 209F+22F or 124F+22F Adaptive Guide Windows for continuity-first testing."]);
    }
    const memoryContracts = {
      vram8: [124,2048], vram12: [209,4096], vram16: [294,8192], vram24: [362,16384],
    };
    const expected = memoryContracts[selectedMemory(node)];
    if (expected && (Number(widget(node,"motion_context_window_frames")?.value)!==expected[0] || Number(widget(node,"h3_exact_chunk_rows")?.value)!==expected[1])) issues.push(["warn", `VRAM preset is partially overridden. Reapply it to restore ${expected[0]} frames / ${expected[1]} attention rows.`]);
    if (!issues.length) issues.push(["ok", "Configuration is coherent. Final model-family and asset checks run before sampling."]); return issues;
  }
  function renderRail(mode) {
    const rail = q(".h3p-rail"); rail.replaceChildren(); const groups = visibleGroups(mode); if (!groups.some((group) => group.id === active)) active = "assistant";
    groups.forEach((group) => { const button = document.createElement("button"); button.className = `h3p-tab${active === group.id ? " active" : ""}`; button.textContent = group.label; button.onclick = () => { active = group.id; node.properties ||= {}; node.properties.iamccs_h3_settings_pro_active_section = active; app.graph?.change?.(); refresh(); }; rail.append(button); });
    const owner = document.createElement("div"); owner.className = "h3p-owner"; owner.innerHTML = "<strong>OWNERSHIP LOCK</strong>Shotboard: timeline, prompts, guides, images/video, duration, FPS and audio lanes.<br><br>Settings PRO: generation mode, canvas, sampling, memory, acceleration, continuity and delivery. AUTO imports once; it is not a live two-way sync."; rail.append(owner);
  }
  function recipeGroup(parent) {
    if (parent?.classList?.contains("h3p-memory-recipes")) return "memory";
    if (parent?.classList?.contains("h3p-speed-recipes")) return "speed";
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
    const group = recipeGroup(parent);
    const button = document.createElement("button");
    const remembered = rememberedRecipe(group) === String(id);
    button.className = `h3p-recipe${(selected || remembered) ? " active" : ""}`;
    button.textContent = id === "fused" ? "FUSED · 4 STEP · CHECKPOINT COMPATIBILITY" : label;
    button.disabled = !enabled;
    button.onclick = () => {
      if (!enabled) return;
      rememberRecipe(group, id);
      (handler || (() => applyRecipe(node, id)))();
      parent.querySelectorAll?.(".h3p-recipe").forEach((entry) => entry.classList.toggle("active", entry === button));
      node._iamccsSettingsProRefresh?.();
    };
    parent.append(button);
  }
  function renderAssistant(mode) {
    const grid = q("[data-grid]"); grid.className = "h3p-flow"; grid.replaceChildren();
    const ask = (text) => { const el = document.createElement("div"); el.className = "h3p-question"; el.textContent = text; grid.append(el); };
    ask("1 · WHAT DO YOU WANT TO CREATE?"); const modes = document.createElement("div"); modes.className = "h3p-choice-grid";
    const selectedMode = assistantModeKey(node, mode);
    const family=value=>value.startsWith('t2')?'TEXT / CREATE':value.startsWith('ref2')?'REFERENCES / IDENTITY':value.startsWith('v2')?'VIDEO / TRANSFORM':['i2va','fl2va_stable','fl2va_continuous'].includes(value)?'IMAGES / SHOTS':'TIMELINE / LONG FORM';
    for(const category of ['TEXT / CREATE','IMAGES / SHOTS','REFERENCES / IDENTITY','VIDEO / TRANSFORM','TIMELINE / LONG FORM']){
      const box=document.createElement('section');box.style.cssText='grid-column:1/-1;border:1px solid #426475;border-radius:10px;padding:12px;background:linear-gradient(135deg,#142632,#18202d)';
      const heading=document.createElement('h3');heading.textContent=category;heading.style.cssText='margin:0 0 10px;color:#9bdccc;font-size:12px;letter-spacing:1px';box.append(heading);
      const cards=document.createElement('div');cards.className='h3p-choice-grid';
      MODE_CHOICES.filter(([,value])=>family(value)===category).forEach(([label,value,note])=>{const button=document.createElement('button');const availability=modeAvailability(node,value);button.className=`h3p-choice${selectedMode===value?' active':''}`;button.innerHTML=`${label}<span>${note}${availability.available?'':` · ${availability.reason}`}</span>`;button.disabled=!availability.available;button.title=availability.reason||note;button.onclick=()=>{if(availability.available)setAssistantMode(node,value);};
        if(value==='longvid_guides') {
          const row=document.createElement('div');row.className='h3p-longvid-row';row.style.cssText='display:flex;align-items:stretch;gap:8px;min-width:0';
          button.style.cssText='flex:1;min-width:0';
          const stageLabel=document.createElement('label');stageLabel.className='h3p-longvid-2stage-toggle';
          stageLabel.style.cssText='display:flex;position:static;align-items:center;gap:6px;flex:0 0 auto;max-width:120px;height:auto;margin:0;padding:10px;border:1px solid #d0a45c;border-radius:8px;font-size:10px;line-height:1.3;white-space:normal';
          stageLabel.title='ON: enable LongVid progressive spatial generation. DIRECT controls the endpoint/profile; NATIVE controls the linked LOW/HIGH resolution pair.';
          const stageToggle=widget(node,'longvid_pianosequenza_2stage_enabled');
          const stageInput=document.createElement('input');stageInput.type='checkbox';stageInput.checked=Boolean(stageToggle?.value);stageInput.disabled=!stageToggle;
          stageInput.style.cssText='position:static;flex:0 0 16px;width:16px;height:16px;margin:0';
          stageInput.onchange=()=>{setValue(node,'longvid_pianosequenza_2stage_enabled',stageInput.checked);if(mode!=='longvid_guides')setAssistantMode(node,'longvid_guides');normalize2StageResolutionLink(node);refresh();};
          stageLabel.append(stageInput,document.createTextNode('2 STAGE'));row.append(button,stageLabel);cards.append(row);
        } else if(value==='keyframe_joint_native') {
          const row=document.createElement('div');row.className='h3p-joint-row';row.style.cssText='display:flex;align-items:stretch;gap:8px;min-width:0';
          button.style.cssText='flex:1;min-width:0';
          const label=document.createElement('label');label.className='h3p-latent-toggle';
          label.style.cssText='display:flex;position:static;align-items:center;gap:6px;flex:0 0 auto;max-width:120px;height:auto;margin:0;padding:10px;border:1px solid #74cbbb;border-radius:8px;font-size:10px;line-height:1.3;white-space:normal';
          label.title='ON: continue each interval from its generated AV latent tail. OFF: one joint sample. Requires the LatentGoAhead branch.';
          const toggle=widget(node,'keyframe_joint_latent_new');
          const input=document.createElement('input');input.type='checkbox';input.checked=Boolean(toggle?.value);input.disabled=!toggle || !connectedNodes(node).some(n=>/MiniMaxH3LatentGoAhead/.test(nodeClass(n)));
          input.style.cssText='position:static;flex:0 0 16px;width:16px;height:16px;margin:0';
          input.onchange=()=>{setValue(node,'keyframe_joint_latent_new',input.checked);if(mode!=='keyframe_joint_native')setAssistantMode(node,'keyframe_joint_native');refresh();};
          label.append(input,document.createTextNode('LATENT NEW'));row.append(button,label);
          const joint=document.createElement('div');joint.style.cssText='display:flex;flex-direction:column;gap:8px;min-width:0';joint.append(row);
          if(selectedMode==='keyframe_joint_native' && input.checked){
            const branch=connectedNodes(node).find(n=>String(n.comfyClass||n.type).startsWith('IAMCCS_MiniMaxH3LatentGoAhead'));
            const blend=branch?.widgets?.find(w=>w.name==='join_blend'),frames=branch?.widgets?.find(w=>w.name==='blend_frames');
            if(blend){const controls=document.createElement('div');controls.style.cssText='display:flex;gap:6px;align-items:center;flex-wrap:wrap;padding:8px;background:#162a32';
              const caption=document.createElement('span');caption.textContent='AV JOIN';const select=document.createElement('select');
              for(const [value,text] of [['none','OFF · original'],['linear','Linear blend'],['smoothstep','Smoothstep blend']])select.add(new Option(text,value));select.value=blend.value||'none';select.onchange=()=>{blend.value=select.value;blend.callback?.(blend.value);app.graph?.change?.();};controls.append(caption,select);
              if(frames){const count=document.createElement('select');for(const n of [3,6,9,12,18,24])count.add(new Option(`${n} frames`,n));count.value=frames.value||9;count.onchange=()=>{frames.value=Number(count.value);frames.callback?.(frames.value);app.graph?.change?.();};controls.append(count);}controls.title='Delivery-only AV overlap. 9 frames = 0.375 s shorter per join at 24 fps. Raw master is retained.';joint.append(controls);}
          }
          cards.append(joint);
        } else cards.append(button);});box.append(cards);modes.append(box);
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
      const caution=document.createElement('p');caution.textContent='Memory presets are starting points, not VRAM guarantees. They change window and memory controls, not resolution. LatentGoAhead supports compatible accelerators; Fused is T2VA only and progressive sampling cannot preserve its audio mask. Face Swap requires two identity views; ControlNet requires a compatible control model and real preprocessed frames.';grid.append(recipes,caution);
    }
    ask("2 · WHAT MEDIA DO YOU HAVE?"); const media = document.createElement("div"); media.className = "h3p-recipes";
    const mediaText = String(mode).startsWith("t2") ? "PROMPT ONLY" : String(mode).includes("controlnet") ? "PREPROCESSED CONTROL VIDEO" : String(mode).includes("face_swap") ? "SOURCE VIDEO + IDENTITY REFERENCES" : String(mode).includes("ref2") ? "ONE OR MORE REFERENCE IMAGES" : String(mode).includes("fl2") ? "FIRST + LAST IMAGE" : String(mode).includes("i2") ? "OPENING IMAGE" : "SHOTBOARD MEDIA";
    const mediaTag = document.createElement("button"); mediaTag.className = "h3p-choice active"; mediaTag.textContent = mediaText; media.append(mediaTag); grid.append(media);
    ask("3 · DO YOU NEED AUDIO OR CONTINUITY?"); const audio = document.createElement("div"); audio.className = "h3p-recipes";
    const boardAudio = String(widget(linkedShotboard(node), "audio_mode")?.value || "h3_native_generated");
    [["GENERATED AUDIO", "h3_native_generated"], ["REFERENCE AUDIO", "h3_ref2va_audio"], ["CUSTOM AUDIO DRIVE", "h3_custom_audio_drive"], ["AUDIO IN POST", "external_audio_post"]].forEach(([label, value]) => { const button = document.createElement("button"); button.className = `h3p-recipe${boardAudio === value ? " active" : ""}`; button.textContent = label; button.onclick = () => setShotboardAudio(node, value); audio.append(button); }); grid.append(audio);
    ask("4 · CHOOSE ACCELERATION (ONLY INSTALLED, MODE-COMPATIBLE ASSETS ARE ENABLED)"); const speed = document.createElement("div"); speed.className = "h3p-recipes";
    const selectedSpeed = selectedAcceleration(node);
    speed.classList.add("h3p-speed-recipes");
    addRecipe(speed, "NATIVE QUALITY", "native", true, selectedSpeed === "native"); addRecipe(speed, "PDD · 8 STEP", "pdd", accelerationAvailable(node,"pdd",mode), selectedSpeed === "pdd");
    addRecipe(speed, "FASTH3 · 6 STEP", "fasth3", accelerationAvailable(node,"fasth3",mode), selectedSpeed === "fasth3"); addRecipe(speed, "SLA · 4 STEP", "sla", accelerationAvailable(node,"sla",mode), selectedSpeed === "sla");
    addRecipe(speed, "FUSED FAST · T2VA ONLY", "fused", accelerationAvailable(node,"fused",mode), selectedSpeed === "fused"); grid.append(speed);
    ask("5 · CHOOSE A VRAM PRESET"); const memory = document.createElement("div"); memory.className = "h3p-recipes h3p-memory-recipes";
    const selectedVram = selectedMemory(node), selectedFinish = selectedDelivery(node), selectedFace = selectedFaceRefine(node);
    addRecipe(memory, "≤ 8–12 GB · 124F / 2048 ROWS", "vram8", true, selectedVram === "vram8"); addRecipe(memory, "12–16 GB · 209F / 4096 ROWS", "vram12", true, selectedVram === "vram12"); addRecipe(memory, "16–24 GB · 294F / 8192 ROWS", "vram16", true, selectedVram === "vram16"); addRecipe(memory, "24 GB+ · 362F / 16384 ROWS", "vram24", true, selectedVram === "vram24"); grid.append(memory);
    ask("6 · CHOOSE DELIVERY");
    const delivery = document.createElement("div"); delivery.className = "h3p-recipes h3p-delivery-recipes";
    addRecipe(delivery, "FACE REFINE · TOGGLE", "face-refine", true, selectedFace);
    addRecipe(delivery, "NATIVE DELIVERY", "native-delivery", true, selectedFinish === "native-delivery");
    addRecipe(delivery, "H3 2-PASS UPRES", "upres", branchGate(node, "h3_upres_model_name").available && deliveryModeAvailable(node, "h3_fast_latent_2pass"), selectedFinish === "upres");
    addRecipe(delivery, "H3 PIXEL REFINE", "h3-pixel", deliveryModeAvailable(node, "h3_pixel_refine"), selectedFinish === "h3-pixel");
    addRecipe(delivery, "LTX 2.3 MASTER", "ltx-master", deliveryModeAvailable(node, "ltx23"), selectedFinish === "ltx-master");
    addRecipe(delivery, "LTX 2.3 PER-SHOT", "ltx-per-shot", deliveryModeAvailable(node, "ltx23_per_chunk"), selectedFinish === "ltx-per-shot");
    addRecipe(delivery, "RTX FINAL", "rtx-final", deliveryModeAvailable(node, "rtx_final"), selectedFinish === "rtx-final");
    grid.append(delivery);
  }
  function renderMain(mode) {
    q(".h3p-main").dataset.section = active;
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
          return { available: swap.available || detailer.available, reason: "Connect the optional Face Swap or Face Detailer branch to activate these controls." };
        })() : null;
    q(".h3p-section-note").textContent = group.assistant ? "Answer in order; every accepted tag updates Queue Truth immediately."
      : contextualGate && !contextualGate.available ? `Optional branch is not connected. Values are preserved but muted. ${contextualGate.reason}`
      : group.contextual ? `Branch connected; controls compile for Shotboard mode ${mode}.`
      : "Only controls relevant to this render layer are shown.";
    q(".h3p-context").innerHTML = `<b>Current Shotboard authority:</b> ${mode}. Mode, media, prompts, duration and FPS remain stored in Shotboard.`;
    const recipes = q(".h3p-recipes"); recipes.replaceChildren(); if (group.assistant) { renderAssistant(mode); return; }
    if (active === "speed") {
      const help = document.createElement("p");
      help.textContent = "Choose ONE speed recipe. Native: quality baseline; PDD: 8-step distillation; FastH3: 6 steps; SLA: 4-step sparse-attention recipe; Fused ConvRot: baked 4-step model, no extra Turbo/PDD LoRA. Memory presets only change memory, not speed. Manual overrides are optional below.";
      recipes.append(help);
    }
    const recipeSet = active === "memory" ? [["≤ 8–12 GB · 124F / 2048 ROWS", "vram8"], ["12–16 GB · 209F / 4096 ROWS", "vram12"], ["16–24 GB · 294F / 8192 ROWS", "vram16"], ["24 GB+ · 362F / 16384 ROWS", "vram24"]] : active === "speed" ? [["NATIVE QUALITY · 20 STEP", "native"], ["PDD · 8 STEP", "pdd"], ["FASTH3 · 6 STEP", "fasth3"], ["SLA · 4 STEP", "sla"], ["FUSED FAST · T2VA ONLY", "fused"]] : active === "control" ? [["ENABLE + SELECT INSTALLED MODEL", "control"]] : active === "face" ? [["SAFE FACE SWAP", "face"]] : active === "face_refine" ? [["FACE REFINE · TOGGLE", "face-refine"]] : active === "finish" ? [["NATIVE DELIVERY", "native-delivery"], ["H3 2-PASS UPRES", "upres"], ["H3 PIXEL REFINE", "h3-pixel"], ["LTX 2.3 MASTER", "ltx-master"], ["LTX 2.3 PER-SHOT", "ltx-per-shot"], ["RTX FINAL", "rtx-final"]] : [];
    recipeSet.forEach(([label, id]) => {
      const deliveryRecipe = ["native-delivery", "upres", "h3-pixel", "ltx-master", "ltx-per-shot", "rtx-final", "face-refine"].includes(id);
      const deliveryAvailable = id === "upres"
        ? branchGate(node, "h3_upres_model_name").available && deliveryModeAvailable(node, "h3_fast_latent_2pass")
        : id === "h3-pixel" ? deliveryModeAvailable(node, "h3_pixel_refine")
        : id === "ltx-master" ? deliveryModeAvailable(node, "ltx23")
        : id === "ltx-per-shot" ? deliveryModeAvailable(node, "ltx23_per_chunk")
        : id === "rtx-final" ? deliveryModeAvailable(node, "rtx_final")
        : true;
      addRecipe(recipes, label, id, deliveryRecipe ? deliveryAvailable : accelerationAvailable(node,id,mode));
    });
    if (active === "overview") {
      normalize2StageResolutionLink(node);
  normalizePianosequenzaStageControls(node);
      const linkedNote = document.createElement("p");
      linkedNote.textContent = "Pianosequenza Multi-Stage: enable the LOW/HIGH link below to use a legal pair such as 640×384 → 1280×768. The HIGH size becomes the native canvas; LOW is used only by stage 1.";
      recipes.append(linkedNote);
      const format = document.createElement("select"); format.className = "h3p-resolution";
      const current = `${Number(widget(node,"width")?.value || 0)}x${Number(widget(node,"height")?.value || 0)}`;
      format.add(new Option("CUSTOM · keep current", "custom"));
      H3_NATIVE_RESOLUTION_PRESETS.forEach(([label,value]) => format.add(new Option(label,value)));
      format.value = [...format.options].some((option) => option.value === current) ? current : "custom";
      format.onchange = () => { if (format.value === "custom") return; const [width,height] = format.value.split("x").map(Number); setValue(node,"width",width,false); setValue(node,"height",height,false); setValue(node,"image_width",width,false); setValue(node,"image_height",height,false); document.dispatchEvent(new CustomEvent("iamccs:h3-settings-changed", {detail:{source_node_id:node.id,recipe:"resolution"}})); refresh(); };
      recipes.prepend(format);
    }
    const grid = q("[data-grid]"); grid.className = "h3p-grid"; grid.replaceChildren();
    let names = fieldNames(group, mode);
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
        const fields = {native: [], pdd: ["pdd_lora_name"], fasth3: ["turbo_lora_name"], sla: ["turbo_lora_name", "h3_sla_sparsity"], fused: ["fused_turbo_model_name"]}[selected] || ["turbo_mode", "turbo_lora_name"];
        names = names.filter(name => ["acceleration", ...fields].includes(name));
      }
    }
    let layout = FUNCTIONAL_LAYOUT[active];
    if (active === "advanced") {
      const bucket = (name) => name.startsWith("h3_r40_") ? "SEED SCOUT" : name.startsWith("ltx_") ? "LTX DELIVERY" : name.startsWith("v2v_") ? "SOURCE VIDEO" : name.startsWith("flf_") || name.startsWith("motion_context_") ? "CONTINUITY" : name.includes("memory") || name.includes("device") || name.includes("vram") ? "MEMORY & RUNTIME" : "OTHER TECHNICAL";
      layout = [...new Set(names.map(bucket))].map((title) => [title, names.filter((name) => bucket(name) === title)]);
    }
    if (layout) {
      const used = new Set();
      layout.forEach(([title, fields]) => { const relevant = fields.filter((name) => names.includes(name)); if (!relevant.length) return; const section = document.createElement("section"); section.className = "h3p-functional"; section.innerHTML = `<div class="h3p-functional-title">${title}</div>`; const inner = document.createElement("div"); inner.className = "h3p-functional-grid"; relevant.forEach((name) => { used.add(name); const control = makeControl(name); if (control) inner.append(control); }); section.append(inner); grid.append(section); });
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
    const values = [["Generation authority", node.properties?.iamccs_settings_master ? "SETTINGS PRO" : "SHOTBOARD / INITIAL"], ["Mode", `${mode} · Settings PRO when connected`], ["Editorial authority", "SHOTBOARD · prompts, guides, media, timeline and audio"], ["Asset compatibility", assetIssues.some(([kind]) => kind === "error") ? "INCOMPATIBLE" : assetIssues.length ? "REVIEW" : "COMPATIBLE"], ["VRAM preset", friendly(widget(node, "performance_profile")?.value)], ["Memory contract", `${widget(node,"motion_context_window_frames")?.value ?? "—"} frames · ${widget(node,"h3_exact_chunk_rows")?.value ?? "—"} rows`], ["Guide window", String(mode) === "longvid_guides" ? selectedGuideWindowStrategy(node) : "N/A"], ["Acceleration engine", friendly(widget(node, "acceleration")?.value)], ["Turbo LoRA", String(widget(node, "turbo_mode")?.value || "off") === "off" && !["fasth3_dense_6step", "h3_sla"].includes(String(widget(node, "acceleration")?.value || "")) ? "OFF" : (widget(node, "turbo_lora_name")?.value || "MISSING")], ["PDD LoRA", widget(node,"pdd_lora_name")?.value || "OFF"], ["Fused model", widget(node, "fused_turbo_model_name")?.value || "OFF"], ["Effective sampling", `${widget(node, "steps")?.value ?? "—"} steps · ${widget(node, "sampler_name")?.value ?? "—"} · ${widget(node, "scheduler")?.value ?? "—"}`], ["Canvas", `${widget(node, "width")?.value ?? "—"} × ${widget(node, "height")?.value ?? "—"}`], ["ControlNet", widget(node, "h3_controlnet_name")?.value || "OFF"], ["Delivery", widget(node, "upscale_enabled")?.value ? widget(node, "upscale_mode")?.value : "NATIVE"]];
    const list = q(".h3p-truth-list"); list.replaceChildren(); values.forEach(([label, value]) => { const row = document.createElement("div"); row.className = "h3p-truth-row"; const caption = document.createElement("span"), content = document.createElement("b"); caption.textContent = label; content.textContent = String(value ?? "—"); row.append(caption, content); list.append(row); });
    const issueList = warnings(mode), health = q(".h3p-health"); health.className = `h3p-health ${issueList.some(([kind]) => kind === "error") ? "error" : issueList.some(([kind]) => kind === "warn") ? "warn" : ""}`; health.innerHTML = issueList.map(([, message]) => `• ${message}`).join("<br>");
  }
  function refresh() {
    const scrollTop = q(".h3p-main").scrollTop;
    const mode = shotboardMode(node);
    mirrorShotboardAuthority(node);
    node._iamccsSettingsProConnectedClasses = connectedNodes(node).map(nodeClass);
    node._iamccsSettingsProBranchSignature = [...node._iamccsSettingsProConnectedClasses].sort().join("|");
    lastMode = mode; q("[data-mode]").textContent = mode.toUpperCase(); renderRail(mode); renderMain(mode); renderTruth(mode);
    q(".h3p-main").scrollTop = scrollTop;
  }
  node._iamccsSettingsProRefresh = refresh; const dom = node.addDOMWidget("H3 Settings PRO", "iamccs_h3_settings_pro_panel", root, { serialize: false }); dom.computeSize = (width) => [Math.max(1080, Number(width || 1080)), 680];
  node.setSize?.([1140, 750]); node.color = "#33291d"; node.bgcolor = "#0d131a"; refresh(); node._iamccsSettingsProTimer = window.setInterval(() => {
    const mode = shotboardMode(node);
    const branchSignature = connectedNodes(node).map(nodeClass).sort().join("|");
    if (mode !== lastMode || branchSignature !== node._iamccsSettingsProBranchSignature) refresh();
  }, 800);
  const removed = node.onRemoved; node.onRemoved = function () { window.clearInterval(this._iamccsSettingsProTimer); return removed?.apply(this, arguments); };
}

app.registerExtension({
  name: "IAMCCS.H3SettingsPro.UI",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (String(nodeData?.name || nodeData?.class_type || "") !== NODE_TYPE) return;
    const created = nodeType.prototype.onNodeCreated; nodeType.prototype.onNodeCreated = function () { const result = created?.apply(this, arguments); window.setTimeout(() => mount(this), 0); return result; };
    const configured = nodeType.prototype.onConfigure; nodeType.prototype.onConfigure = function (info) {
      const result = configured?.apply(this, arguments);
      restoreNamedValues(this, info);
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
      serializeNamedValues(this, info);
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
