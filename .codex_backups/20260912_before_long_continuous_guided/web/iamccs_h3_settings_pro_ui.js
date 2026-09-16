// SPDX-License-Identifier: GPL-3.0-or-later
import { app } from "/scripts/app.js";

const NODE_TYPE = "IAMCCS_ShotboardH3SettingsPro";
const SHOTBOARD_TYPE = "IAMCCS_MiniMaxH3ShotPlanner";
const BRIDGE_TYPES = new Set(["IAMCCS_CineH3Input", "IAMCCS_CineH3FunControlInput", "IAMCCS_MiniMaxH3FunControlInput"]);
const SHOTBOARD_OWNED = new Set(["duration_seconds", "task_mode"]);
const INTERNAL = new Set(["seed_control_after_generate_compat", "h3_advisor_state"]);
const GROUPS = [
  { id: "assistant", label: "MODE ASSISTANT", title: "Guided setup", assistant: true, fields: [] },
  { id: "overview", label: "1 · NATIVE", title: "Native H3 canvas", fields: ["width", "height", "reference_resize_policy", "reference_resize_megapixels", "reference_resize_filter", "prompt_mapping"] },
  { id: "audio", label: "2 · AUDIO", title: "Audio authority", fields: ["audio_mode", "reference_audio_role", "voice_reference_picture_index"] },
  { id: "sampling", label: "3 · SAMPLE", title: "Native H3 sampling", fields: ["performance_profile", "text_encoder_device", "seed", "seed_stride", "steps", "sampler_name", "scheduler", "denoise", "shift_video", "shift_audio"] },
  { id: "speed", label: "4 · SPEED", title: "Native H3 acceleration", fields: ["acceleration", "turbo_mode", "turbo_lora_name", "turbo_strength", "turbo_sampler_mode", "fused_turbo_model_name", "fused_turbo_sigma_preset", "pdd_lora_name", "pdd_strength", "secondary_lora_enabled", "secondary_lora_name", "secondary_lora_strength", "ref_image_size", "sol_conditioning", "spectrum_profile", "vram_clean_before_decode", "rife_mode", "h3_sla_sparsity", "h3_sla_dense_last_steps", "h3_exact_profile", "h3_exact_chunk_rows", "h3_exact_precision_mode", "h3_exact_qkv_streaming", "h3_exact_attention_memory", "h3_clipproj_profile", "h3_clipproj_load_mode"] },
  { id: "direction", label: "5 · DIRECT", title: "Mode-specific contract", fields: ["motion_context_window_frames", "reference_role_1", "reference_role_2", "reference_role_3", "reference_role_4", "reference_video_role", "v2v_guide_mode", "v2v_source_range_policy", "v2v_source_offset_seconds", "v2v_source_fit", "v2v_source_end_policy", "v2v_audio_pairing", "flf_join_mode", "flf_overlap_frames", "flf_continuity_mode", "flf_continuity_tail_frames", "flf_continuity_audio"] },
  { id: "control", label: "CONTROLNET", title: "H3 Fun ControlNet", contextual: "control", fields: ["h3_controlnet_enabled", "h3_controlnet_name", "h3_controlnet_kind", "h3_controlnet_strength", "h3_controlnet_start_percent", "h3_controlnet_end_percent", "h3_controlnet_frame_scope", "h3_controlnet_end_policy"] },
  { id: "face", label: "FACE SWAP", title: "Face Swap", contextual: "face", fields: ["h3_faceswap_sam_model", "h3_faceswap_birefnet_model", "h3_faceswap_mask_prompt", "h3_faceswap_threshold", "h3_faceswap_objects", "h3_faceswap_cleanup_threshold", "h3_faceswap_cleanup_shrink", "h3_faceswap_cleanup_min_frames", "h3_faceswap_cleanup_edge_grow", "h3_faceswap_crop_scale", "h3_faceswap_crop_megapixels", "h3_faceswap_grow_spatial", "h3_faceswap_grow_temporal", "h3_faceswap_feather", "face_detailer_enabled", "face_detailer_profile", "face_detailer_use_sam_mask"] },
  { id: "scout", label: "7 · SCOUT", title: "Candidate seed scout", fields: ["h3_r40_seed_scout_enabled", "h3_r40_candidate_count", "h3_r40_seed_stride", "h3_r40_preview_max_frames", "h3_r40_sparse_enabled", "h3_r40_sparse_video_budget", "h3_r40_sparse_denser_edges"] },
  { id: "finish", label: "8 · OUTPUT", title: "Delivery", fields: ["upscale_mode", "upscale_enabled", "upscale_width", "upscale_height", "upscale_prompt", "upscale_sage", "upscale_seed_offset", "wan_upscale_denoise", "ltx_seam_safe", "ltx_detailer_enabled", "ltx_detailer_lora_name", "ltx_detailer_strength", "ltx_4k_enabled", "ltx_4k_quality", "ltx_looper_temporal_tile_size", "ltx_looper_temporal_overlap", "ltx_looper_guiding_strength", "ltx_looper_overlap_strength", "ltx_looper_cond_image_strength", "ltx_looper_horizontal_tiles", "ltx_looper_vertical_tiles", "ltx_looper_spatial_overlap", "h3_upres_model_name", "h3_upres_precision", "h3_upres_device", "h3_upres_keep_models_resident", "h3_upres_steps", "h3_upres_denoise", "h3_upres_sampler", "h3_upres_scheduler", "h3_upres_temporal_chunk", "h3_upres_temporal_overlap", "h3_upres_anchor_strength", "h3_upres_tile_width", "h3_upres_tile_height", "h3_upres_overlap_width", "h3_upres_overlap_height", "h3_upres_fade_width", "h3_upres_fade_height", "h3_upres_min_tile_size", "h3_upres_overlap_mode", "h3_upres_overlap_blend", "h3_upres_rtx_enabled", "h3_upres_rtx_quality", "h3_upres_pixel_groups", "h3_upres_window_frames", "h3_upres_window_overlap", "h3_upres_pixel_method"] },
  { id: "advanced", label: "TECHNICAL", title: "Technical controls", dynamic: true, fields: [] },
];
const ASSIGNED = new Set(GROUPS.flatMap((group) => group.fields));
const FUNCTIONAL_LAYOUT = {
  direction: [
    ["MOTION CONTEXT", ["motion_context_window_frames", "flf_continuity_tail_frames", "flf_continuity_audio"]],
    ["REFERENCE ROLES", ["reference_role_1", "reference_role_2", "reference_role_3", "reference_role_4", "reference_video_role"]],
    ["V2VA SOURCE", ["v2v_guide_mode", "v2v_source_range_policy", "v2v_source_offset_seconds", "v2v_source_fit", "v2v_source_end_policy", "v2v_audio_pairing"]],
    ["FLF CONTINUITY", ["flf_join_mode", "flf_overlap_frames", "flf_continuity_mode", "flf_continuity_tail_frames", "flf_continuity_audio"]],
  ],
  speed: [
    ["ENGINE", ["acceleration", "h3_sla_sparsity", "h3_sla_dense_last_steps"]],
    ["TURBO LORA", ["turbo_mode", "turbo_lora_name", "turbo_strength", "turbo_sampler_mode"]],
    ["PDD 8-STEP", ["pdd_lora_name", "pdd_strength"]],
    ["FUSED MODEL", ["fused_turbo_model_name", "fused_turbo_sigma_preset"]],
    ["EXACT ATTENTION & CLIPPROJ", ["h3_exact_profile", "h3_exact_chunk_rows", "h3_exact_precision_mode", "h3_exact_qkv_streaming", "h3_exact_attention_memory", "h3_clipproj_profile", "h3_clipproj_load_mode"]],
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
  ["T2VA · TEXT ONLY", "t2va", "One native H3 shot from prompt only."],
  ["I2VA · OPENING IMAGE", "i2va", "One image per shot; multiple boxes are independent hard cuts."],
  ["FL2VA · STABLE KEYFRAMES", "fl2va_stable", "A→B, B→C with authored shared keyframes."],
  ["FL2VA · NATIVE AV CONTINUITY", "fl2va_continuous", "Carry 22/39/56 native AV frames between FL2VA chunks."],
  ["REF2VA · REFERENCES", "ref2va", "Reference blocks for identity, object or style; no temporal carry."],
  ["REF2VID · AUDIO PERFORMANCE", "ref2vid_lipsync", "Reference image plus one locked AudioBoard performance per hard-cut shot."],
  ["LONGVID · POSITIONED GUIDES", "longvid_guides", "Global timeline guides across independent legal H3 windows."],
  ["LONG MULTI-SHOT", "longvid_motion_context", "Positioned shot guides plus a native AV tail across technical H3 chunk boundaries."],
  ["LONGVID · MULTISHOT AUDIO DRIVE", "longvid_guided_lipsync", "Per-shot guides and rebased locked AudioBoard clips."],
  ["CONTROL VIDEO", "v2va_controlnet", "Drive pose, depth or edges from video."],
  ["OBJECT SWAP", "v2va_object_swap", "Replace a tracked object in source video."],
  ["FACE SWAP", "v2va_face_swap", "Replace a tracked identity in source video."],
];
const FRIENDLY_VALUES = {
  rtx_xx60_safe: "8–12 GB VRAM · Safe", rtx_xx70_balanced: "12–16 GB VRAM · Balanced",
  rtx_xx80_quality: "16–24 GB VRAM · Quality", rtx_xx90_max: "24 GB+ VRAM · Maximum",
  rtx3060_draft: "8–12 GB VRAM · Draft (legacy)", rtx3060_balanced: "8–12 GB VRAM · Balanced (legacy)",
  rtx3060_turbo: "8–12 GB VRAM · Fast (legacy)", auto_3060: "Automatic · 8–12 GB VRAM (legacy)",
  rtx_xx60_8_12gb_124: "8–12 GB VRAM · 124 frames", rtx_xx70_12_16gb_209: "12–16 GB VRAM · 209 frames",
  rtx_xx80_16_24gb_294: "16–24 GB VRAM · 294 frames", rtx_xx90_24gb_362: "24 GB+ VRAM · 362 frames",
  rtx3060_12gb_124: "8–12 GB VRAM · 124 frames (legacy)", rtx3060_12gb_209: "12 GB VRAM · 209 frames (legacy)",
  low_vram_auto: "Automatic low-VRAM", pdd_native_8step: "PDD · 8 steps", fasth3_dense_6step: "FastH3 · 6 steps",
  matlowai_fused_turbo_manual_sigma: "Fused Fast · manual sigma", h3_sla: "SLA · 4-step capable",
};

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
function linkedShotboard(node) { return downstreamNodes(node).find((candidate) => nodeClass(candidate) === SHOTBOARD_TYPE) || null; }
function shotboardMode(node) {
  const board = linkedShotboard(node); if (!board) return "NOT CONNECTED";
  const direct = String(widget(board, "task_mode")?.value || "auto_from_timeline"); if (direct !== "auto_from_timeline") return direct;
  try { const data = JSON.parse(String(widget(board, "timeline_data")?.value || "{}")); return String(data.task_mode || data.mode || direct); } catch { return direct; }
}
function setShotboardMode(node, mode) {
  const board = linkedShotboard(node); if (!board) return false;
  setValue(board, "task_mode", mode, false);
  const timelineWidget = widget(board, "timeline_data");
  if (timelineWidget) {
    let data = {}; try { data = JSON.parse(String(timelineWidget.value || "{}")); } catch {}
    data.task_mode = mode; data.mode = mode; setValue(board, "timeline_data", JSON.stringify(data), false);
  }
  document.dispatchEvent(new CustomEvent("iamccs:h3-settings-changed", { detail: { source_node_id: node.id, task_mode: mode } }));
  node._iamccsSettingsProRefresh?.(); return true;
}
function setShotboardAudio(node, audioMode) {
  const board = linkedShotboard(node); if (!board) return false;
  setValue(board, "audio_mode", audioMode, false);
  document.dispatchEvent(new CustomEvent("iamccs:h3-settings-changed", { detail: { source_node_id: node.id, audio_mode: audioMode } }));
  node._iamccsSettingsProRefresh?.(); return true;
}
function assistantModeKey(node, mode = shotboardMode(node)) {
  if (String(mode) !== "fl2va") return String(mode);
  const board = linkedShotboard(node);
  return String(widget(board, "flf_continuity_mode")?.value || "stable_keyframes") === "native_av_context"
    ? "fl2va_continuous" : "fl2va_stable";
}
function setAssistantMode(node, key) {
  const board = linkedShotboard(node); if (!board) return false;
  const actualMode = String(key).startsWith("fl2va_") ? "fl2va" : key;
  setShotboardMode(node, actualMode);
  if (key === "fl2va_stable") setValue(board, "flf_continuity_mode", "stable_keyframes", false);
  if (key === "fl2va_continuous") setValue(board, "flf_continuity_mode", "native_av_context", false);
  if (["ref2vid_lipsync", "longvid_guided_lipsync"].includes(key)) setValue(board, "audio_mode", "h3_custom_audio_drive", false);
  document.dispatchEvent(new CustomEvent("iamccs:h3-settings-changed", { detail: { source_node_id: node.id, assistant_mode: key } }));
  node._iamccsSettingsProRefresh?.(); return true;
}
function modeContext(mode) { const value = String(mode).toLowerCase(); return value.includes("controlnet") ? "control" : value.includes("face_swap") ? "face" : "core"; }
function modeFamily(mode) { return String(mode).toLowerCase().includes("ref2") ? "ref2" : "fl2"; }
function firstChoice(node, name, predicate) { return choices(widget(node, name)).map(String).find((value) => value && predicate(value.toLowerCase())) || ""; }
function resetAcceleration(set) { set("turbo_mode", "off"); set("turbo_lora_name", ""); set("pdd_lora_name", ""); set("fused_turbo_model_name", ""); }
function selectedAcceleration(node) {
  const value = String(widget(node, "acceleration")?.value || "native");
  return value === "pdd_native_8step" ? "pdd" : value === "fasth3_dense_6step" ? "fasth3" : value === "h3_sla" ? "sla" : value === "matlowai_fused_turbo_manual_sigma" ? "fused" : "native";
}
function selectedMemory(node) {
  const value = String(widget(node, "performance_profile")?.value || "");
  if (value.includes("xx90")) return "vram24"; if (value.includes("xx80")) return "vram16"; if (value.includes("xx70")) return "vram12"; return "vram8";
}
function selectedDelivery(node) { return widget(node, "upscale_enabled")?.value ? "upres" : "native-delivery"; }

function applyRecipe(node, recipe) {
  const mode = shotboardMode(node), family = modeFamily(mode), set = (name, value) => setValue(node, name, value, false);
  if (recipe === "native") {
    resetAcceleration(set); set("acceleration", "native"); set("steps", 20); set("sampler_name", "res_multistep");
    set("scheduler", "simple"); set("denoise", 1); set("shift_video", 6); set("shift_audio", 3);
  } else if (recipe === "vram8") {
    resetAcceleration(set); set("performance_profile", "rtx_xx60_safe"); set("motion_context_window_frames", 124);
    set("h3_exact_profile", "rtx_xx60_8_12gb_124"); set("h3_exact_chunk_rows", 2048);
    set("h3_clipproj_profile", "4b_v3.1"); set("h3_clipproj_load_mode", "dynamic"); set("vram_clean_before_decode", true);
    set("acceleration", "low_vram_auto"); set("steps", 20);
  } else if (recipe === "vram12") {
    set("performance_profile", "rtx_xx70_balanced"); set("motion_context_window_frames", 209);
    set("h3_exact_profile", "rtx_xx70_12_16gb_209"); set("h3_exact_chunk_rows", 4096);
  } else if (recipe === "vram16") {
    set("performance_profile", "rtx_xx80_quality"); set("motion_context_window_frames", 294);
    set("h3_exact_profile", "rtx_xx80_16_24gb_294"); set("h3_exact_chunk_rows", 8192);
  } else if (recipe === "vram24") {
    set("performance_profile", "rtx_xx90_max"); set("motion_context_window_frames", 362);
    set("h3_exact_profile", "rtx_xx90_24gb_362"); set("h3_exact_chunk_rows", 16384);
  } else if (recipe === "pdd") {
    const asset = firstChoice(node, "pdd_lora_name", (name) => name.includes(family) && name.includes("comfy")); if (!asset) return;
    resetAcceleration(set); set("pdd_lora_name", asset); set("pdd_strength", 1); set("acceleration", "pdd_native_8step");
    set("steps", 8); set("sampler_name", "euler"); set("scheduler", "simple");
  } else if (recipe === "fasth3") {
    const asset = firstChoice(node, "turbo_lora_name", (name) => name.includes("fasth3_dense")); if (!asset) return;
    resetAcceleration(set); set("turbo_mode", "off"); set("turbo_lora_name", asset); set("turbo_strength", 1);
    set("acceleration", "fasth3_dense_6step"); set("steps", 6);
  } else if (recipe === "sla") {
    const asset = firstChoice(node, "turbo_lora_name", (name) => name.includes("sla") && name.includes(family)); if (!asset) return;
    resetAcceleration(set); set("turbo_mode", "early_8_10"); set("turbo_lora_name", asset); set("turbo_strength", 1);
    set("acceleration", "h3_sla"); set("steps", 4); set("h3_sla_sparsity", 0.85); set("h3_sla_dense_last_steps", 0);
  } else if (recipe === "fused") {
    const current = String(widget(node, "fused_turbo_model_name")?.value || "");
    const asset = current || firstChoice(node, "fused_turbo_model_name", (name) => (name.includes("fused") || name.includes("turbo")) && !name.includes("ref2"));
    if (!asset || !["t2va", "i2va", "fl2va"].includes(String(mode))) return;
    resetAcceleration(set); set("fused_turbo_model_name", asset); set("acceleration", "matlowai_fused_turbo_manual_sigma");
    set("fused_turbo_sigma_preset", "4_step"); set("steps", 4); set("sampler_name", "euler"); set("scheduler", "simple");
    set("denoise", 1); set("shift_video", 12); set("shift_audio", 3);
  } else if (recipe === "control") {
    set("h3_controlnet_enabled", true); const asset = firstChoice(node, "h3_controlnet_name", () => true); if (asset) set("h3_controlnet_name", asset);
  } else if (recipe === "face") {
    set("face_detailer_enabled", false); set("upscale_enabled", false);
  } else if (recipe === "native-delivery") {
    set("upscale_enabled", false); set("upscale_mode", "off"); set("rife_mode", "off");
  } else if (recipe === "upres") {
    set("upscale_enabled", true); set("upscale_mode", "h3_fast_latent_2pass");
  }
  document.dispatchEvent(new CustomEvent("iamccs:h3-settings-changed", { detail: { source_node_id: node.id, recipe } }));
  node._iamccsSettingsProRefresh?.();
}

function mount(node) {
  if (node._iamccsSettingsProMounted) return; node._iamccsSettingsProMounted = true; (node.widgets || []).forEach(hideWidget);
  const root = document.createElement("div"); root.className = "iamccs-h3pro";
  root.innerHTML = `
  <style>
  .iamccs-h3pro{height:100%;padding:12px;box-sizing:border-box;background:radial-gradient(circle at 85% 0,#273346 0,transparent 34%),linear-gradient(145deg,#090d13,#111923 62%,#0a0e14);border:1px solid #8b7046;border-radius:14px;color:#eaf0f5;font:11px Inter,Segoe UI,sans-serif;overflow:hidden}.iamccs-h3pro *{box-sizing:border-box}.h3p-head{height:48px;display:flex;align-items:center;gap:12px;border-bottom:1px solid #344253}.h3p-mark{padding:6px 10px;border:1px solid #d2a65c;border-radius:999px;background:#32281a;color:#f8d89c;font-size:9px;font-weight:900;letter-spacing:.08em}.h3p-title{font:700 17px Georgia,serif}.h3p-sub{color:#8291a0;font-size:9px}.h3p-mode{margin-left:auto;text-align:right}.h3p-mode b{display:block;color:#7ee2ad;font-size:10px}.h3p-layout{display:grid;grid-template-columns:155px minmax(500px,1fr) 260px;gap:10px;height:calc(100% - 58px);padding-top:10px}.h3p-rail,.h3p-main,.h3p-truth{min-height:0;border:1px solid #2e3a47;border-radius:10px;background:rgba(12,18,25,.88)}.h3p-rail{padding:7px;display:flex;flex-direction:column;gap:5px}.h3p-tab{height:38px;padding:0 10px;border:1px solid transparent;border-radius:7px;background:transparent;color:#94a2b0;text-align:left;font-size:9px;font-weight:850;letter-spacing:.05em;cursor:pointer}.h3p-tab:hover{background:#182330;color:#fff}.h3p-tab.active{border-color:#a98650;background:linear-gradient(90deg,#3c3020,#1c2530);color:#f3d69c}.h3p-owner{margin-top:auto;padding:10px;border-radius:8px;background:#111b24;color:#8493a2;font-size:8px;line-height:1.45}.h3p-owner strong{display:block;color:#f0c97d;margin-bottom:4px}.h3p-main{padding:12px;overflow:auto}.h3p-section-title{font:700 16px Georgia,serif;color:#f0d39e}.h3p-section-note{margin:4px 0 12px;color:#8493a2;font-size:9px}.h3p-recipes{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px}.h3p-recipe,.h3p-choice{padding:8px 10px;border:1px solid #4c5c6d;border-radius:7px;background:#1a2530;color:#d9e2e9;font-size:8px;font-weight:850;cursor:pointer}.h3p-recipe:hover,.h3p-choice:hover,.h3p-choice.active{border-color:#d0a45c;color:#f6d99d;background:#2b261d}.h3p-recipe[disabled]{opacity:.35;cursor:not-allowed}.h3p-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.h3p-functional{grid-column:1/-1;padding:9px;border:1px solid #344353;border-radius:10px;background:linear-gradient(145deg,#111b25,#0c141c)}.h3p-functional-title{margin:0 0 8px;color:#e3bd78;font-size:8px;font-weight:900;letter-spacing:.09em}.h3p-functional-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.h3p-resolution{height:31px;min-width:220px;border:1px solid #9b7944;border-radius:7px;background:#171f27;color:#f1d49b;padding:0 8px;font-size:8px;font-weight:850}.h3p-field{min-height:58px;padding:7px;border:1px solid #2d3945;border-radius:8px;background:#101821}.h3p-field label{display:block;margin-bottom:5px;color:#9ba8b5;font-size:8px;font-weight:800}.h3p-field input,.h3p-field select{width:100%;height:29px;border:1px solid #43515e;border-radius:6px;background:#0a1118;color:#edf2f6;padding:0 7px;font-size:9px}.h3p-field input[type=checkbox]{width:18px;height:18px;accent-color:#d3a758}.h3p-field small{display:block;margin-top:4px;color:#667786;font-size:7px}.h3p-truth{padding:12px;overflow:auto}.h3p-truth h3{margin:0 0 10px;color:#f0d39e;font:700 14px Georgia,serif}.h3p-truth-row{padding:8px 0;border-bottom:1px solid #26323d}.h3p-truth-row span{display:block;color:#718190;font-size:7px;font-weight:900}.h3p-truth-row b{display:block;margin-top:3px;color:#dce5eb;font:600 9px Consolas,monospace;overflow-wrap:anywhere}.h3p-health{margin-top:10px;padding:9px;border-left:3px solid #68d69a;border-radius:6px;background:#11231c;color:#a9e6c5;font-size:8px;line-height:1.45}.h3p-health.warn{border-color:#e7a14e;background:#2a2014;color:#ffd59a}.h3p-health.error{border-color:#e86767;background:#2d1619;color:#ffb0b0}.h3p-context{margin-bottom:10px;padding:9px;border:1px solid #405064;border-radius:8px;background:#14202c;color:#a9b7c5;font-size:9px}.h3p-context b{color:#f2cf8b}.h3p-question{margin:12px 0 6px;color:#f0cf91;font-size:9px;font-weight:900;letter-spacing:.05em}.h3p-choice-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px}.h3p-choice{text-align:left}.h3p-choice span{display:block;margin-top:3px;color:#8493a2;font-weight:500;line-height:1.3}.h3p-flow{padding:10px;border:1px solid #354455;border-radius:9px;background:#0d151e}
  .h3p-recipe.active{border-color:#d0a45c;color:#f6d99d;background:#2b261d;box-shadow:inset 0 0 0 1px #6b5330}.h3p-recipe[disabled]::after{content:" · UNAVAILABLE";color:#e7a14e}
  </style>
  <div class="h3p-head"><span class="h3p-mark">IAMCCS PRO</span><div><div class="h3p-title">H3 Settings PRO</div><div class="h3p-sub">Render compiler · one queue truth · no editorial duplication</div></div><div class="h3p-mode"><span class="h3p-sub">SHOTBOARD MODE</span><b data-mode>NOT CONNECTED</b></div></div>
  <div class="h3p-layout"><nav class="h3p-rail"></nav><main class="h3p-main"><div class="h3p-section-title"></div><div class="h3p-section-note"></div><div class="h3p-context"></div><div class="h3p-recipes"></div><div class="h3p-grid" data-grid></div></main><aside class="h3p-truth"><h3>Queue Truth</h3><div class="h3p-truth-list"></div><div class="h3p-health"></div></aside></div>`;
  const q = (selector) => root.querySelector(selector); let active = String(node.properties?.iamccs_h3_settings_pro_active_section || "assistant"), lastMode = "";
  function fieldRelevant(name, mode) {
    const value = String(mode).toLowerCase();
    if (name.startsWith("v2v_") && !value.startsWith("v2va_")) return false;
    if (name.startsWith("flf_") && !value.includes("fl2") && !value.includes("longvid")) return false;
    if (name.startsWith("h3_r40_") && !value.includes("scout")) return false; return true;
  }
  function visibleGroups(mode) {
    const context = modeContext(mode);
    return GROUPS.filter((group) => (!group.contextual || group.contextual === context) && (group.assistant || group.dynamic || group.fields.some((name) => widget(node, name))));
  }
  function fieldNames(group, mode) {
    if (!group.dynamic) return group.fields.filter((name) => widget(node, name) && fieldRelevant(name, mode));
    return (node.widgets || []).map((item) => item?.name).filter((name) => name && !ASSIGNED.has(name) && !SHOTBOARD_OWNED.has(name) && !INTERNAL.has(name) && fieldRelevant(name, mode));
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
    control.dataset.field = name; control.onchange = () => {
      const value = control.type === "checkbox" ? control.checked : control.type === "number" ? Number(control.value) : control.value;
      setValue(node, name, value); refresh();
    }; box.append(control);
    const hint = document.createElement("small"); hint.textContent = String(item.options?.tooltip || "Saved here and compiled at Queue.").replace(/RTX\s*30\d0/gi, "the selected VRAM tier"); box.append(hint); return box;
  }
  function warnings(mode) {
    const issues = [], acceleration = String(widget(node, "acceleration")?.value || "native"), turbo = String(widget(node, "turbo_mode")?.value || "off");
    const turboName = String(widget(node, "turbo_lora_name")?.value || ""), fused = String(widget(node, "fused_turbo_model_name")?.value || ""), controlModel = String(widget(node, "h3_controlnet_name")?.value || "");
    if (!linkedShotboard(node)) issues.push(["warn", "Connect Settings PRO to Cine H3 Input, then Cine H3 Input to MiniMax H3 Shotboard."]);
    if (acceleration === "matlowai_fused_turbo_manual_sigma" && !fused) issues.push(["error", "Fused Fast is enabled but no fused diffusion model is selected."]);
    if (acceleration === "matlowai_fused_turbo_manual_sigma" && turbo !== "off") issues.push(["error", "Fused Fast and Turbo LoRA cannot be active together."]);
    if (modeContext(mode) === "control" && !controlModel) issues.push(["error", "ControlNet mode requires an installed H3 Fun ControlNet model."]);
    if (modeContext(mode) === "face" && !String(widget(node, "h3_faceswap_birefnet_model")?.value || "")) issues.push(["error", "Face Swap requires BiRefNet in background_removal."]);
    if (String(mode).includes("ref2") && /(^|[\\/])fl2v/i.test(turboName)) issues.push(["error", "The selected FL2V acceleration LoRA is incompatible with Ref2VA."]);
    if (!issues.length) issues.push(["ok", "Configuration is coherent. Final model-family and asset checks run before sampling."]); return issues;
  }
  function renderRail(mode) {
    const rail = q(".h3p-rail"); rail.replaceChildren(); const groups = visibleGroups(mode); if (!groups.some((group) => group.id === active)) active = "assistant";
    groups.forEach((group) => { const button = document.createElement("button"); button.className = `h3p-tab${active === group.id ? " active" : ""}`; button.textContent = group.label; button.onclick = () => { active = group.id; node.properties ||= {}; node.properties.iamccs_h3_settings_pro_active_section = active; app.graph?.change?.(); refresh(); }; rail.append(button); });
    const owner = document.createElement("div"); owner.className = "h3p-owner"; owner.innerHTML = "<strong>OWNERSHIP LOCK</strong>Shotboard: mode, timeline, media, prompts, duration, FPS and audio.<br><br>Settings PRO: render, memory, acceleration and delivery."; rail.append(owner);
  }
  function addRecipe(parent, label, id, enabled = true, selected = false) { const button = document.createElement("button"); button.className = `h3p-recipe${selected ? " active" : ""}`; button.textContent = label; button.disabled = !enabled; button.onclick = () => applyRecipe(node, id); parent.append(button); }
  function renderAssistant(mode) {
    const grid = q("[data-grid]"); grid.className = "h3p-flow"; grid.replaceChildren();
    const ask = (text) => { const el = document.createElement("div"); el.className = "h3p-question"; el.textContent = text; grid.append(el); };
    ask("1 · WHAT DO YOU WANT TO CREATE?"); const modes = document.createElement("div"); modes.className = "h3p-choice-grid";
    const selectedMode = assistantModeKey(node, mode);
    MODE_CHOICES.forEach(([label, value, note]) => { const button = document.createElement("button"); button.className = `h3p-choice${selectedMode === value ? " active" : ""}`; button.innerHTML = `${label}<span>${note}</span>`; button.onclick = () => setAssistantMode(node, value); modes.append(button); }); grid.append(modes);
    ask("2 · WHAT MEDIA DO YOU HAVE?"); const media = document.createElement("div"); media.className = "h3p-recipes";
    const mediaText = String(mode).startsWith("t2") ? "PROMPT ONLY" : String(mode).includes("controlnet") ? "PREPROCESSED CONTROL VIDEO" : String(mode).includes("face_swap") ? "SOURCE VIDEO + IDENTITY REFERENCES" : String(mode).includes("ref2") ? "ONE OR MORE REFERENCE IMAGES" : String(mode).includes("fl2") ? "FIRST + LAST IMAGE" : String(mode).includes("i2") ? "OPENING IMAGE" : "SHOTBOARD MEDIA";
    const mediaTag = document.createElement("button"); mediaTag.className = "h3p-choice active"; mediaTag.textContent = mediaText; media.append(mediaTag); grid.append(media);
    ask("3 · DO YOU NEED AUDIO OR CONTINUITY?"); const audio = document.createElement("div"); audio.className = "h3p-recipes";
    const boardAudio = String(widget(linkedShotboard(node), "audio_mode")?.value || "h3_native_generated");
    [["GENERATED AUDIO", "h3_native_generated"], ["REFERENCE AUDIO", "h3_ref2va_audio"], ["CUSTOM AUDIO DRIVE", "h3_custom_audio_drive"], ["AUDIO IN POST", "external_audio_post"]].forEach(([label, value]) => { const button = document.createElement("button"); button.className = `h3p-recipe${boardAudio === value ? " active" : ""}`; button.textContent = label; button.onclick = () => setShotboardAudio(node, value); audio.append(button); }); grid.append(audio);
    ask("4 · CHOOSE ACCELERATION (ONLY INSTALLED, MODE-COMPATIBLE ASSETS ARE ENABLED)"); const speed = document.createElement("div"); speed.className = "h3p-recipes";
    const selectedSpeed = selectedAcceleration(node);
    addRecipe(speed, "NATIVE QUALITY", "native", true, selectedSpeed === "native"); addRecipe(speed, "PDD · 8 STEP", "pdd", Boolean(firstChoice(node, "pdd_lora_name", (name) => name.includes(modeFamily(mode)) && name.includes("comfy"))), selectedSpeed === "pdd");
    addRecipe(speed, "FASTH3 · 6 STEP", "fasth3", Boolean(firstChoice(node, "turbo_lora_name", (name) => name.includes("fasth3_dense"))), selectedSpeed === "fasth3"); addRecipe(speed, "SLA · 4 STEP", "sla", Boolean(firstChoice(node, "turbo_lora_name", (name) => name.includes("sla") && name.includes(modeFamily(mode)))), selectedSpeed === "sla");
    addRecipe(speed, "FUSED FAST", "fused", ["t2va", "i2va", "fl2va"].includes(String(mode)) && Boolean(String(widget(node, "fused_turbo_model_name")?.value || "") || firstChoice(node, "fused_turbo_model_name", (name) => name.includes("fused") || name.includes("turbo"))), selectedSpeed === "fused"); grid.append(speed);
    ask("5 · CHOOSE MEMORY AND DELIVERY"); const memory = document.createElement("div"); memory.className = "h3p-recipes";
    const selectedVram = selectedMemory(node), selectedFinish = selectedDelivery(node);
    addRecipe(memory, "≤ 8–12 GB SAFE", "vram8", true, selectedVram === "vram8"); addRecipe(memory, "12–16 GB", "vram12", true, selectedVram === "vram12"); addRecipe(memory, "16–24 GB", "vram16", true, selectedVram === "vram16"); addRecipe(memory, "24 GB+", "vram24", true, selectedVram === "vram24"); addRecipe(memory, "NATIVE DELIVERY", "native-delivery", true, selectedFinish === "native-delivery"); addRecipe(memory, "H3 2-PASS UPRES", "upres", true, selectedFinish === "upres"); grid.append(memory);
  }
  function renderMain(mode) {
    const group = visibleGroups(mode).find((item) => item.id === active) || GROUPS[0]; q(".h3p-section-title").textContent = group.title;
    q(".h3p-section-note").textContent = group.assistant ? "Answer in order; every accepted tag updates Queue Truth immediately." : group.contextual ? `Visible because Shotboard mode resolves to ${mode}.` : "Only controls relevant to this render layer are shown.";
    q(".h3p-context").innerHTML = `<b>Current Shotboard authority:</b> ${mode}. Mode, media, prompts, duration and FPS remain stored in Shotboard.`;
    const recipes = q(".h3p-recipes"); recipes.replaceChildren(); if (group.assistant) { renderAssistant(mode); return; }
    const recipeSet = active === "overview" ? [["≤ 8–12 GB SAFE", "vram8"], ["12–16 GB", "vram12"], ["16–24 GB", "vram16"], ["24 GB+", "vram24"]] : active === "speed" ? [["NATIVE QUALITY", "native"], ["PDD 8 STEP", "pdd"], ["FASTH3 6 STEP", "fasth3"], ["SLA 4 STEP", "sla"], ["FUSED FAST", "fused"]] : active === "control" ? [["ENABLE + SELECT INSTALLED MODEL", "control"]] : active === "face" ? [["SAFE FACE SWAP", "face"]] : active === "finish" ? [["NATIVE DELIVERY", "native-delivery"], ["H3 2-PASS UPRES", "upres"]] : [];
    recipeSet.forEach(([label, id]) => addRecipe(recipes, label, id));
    if (active === "overview") {
      const format = document.createElement("select"); format.className = "h3p-resolution";
      const current = `${Number(widget(node,"width")?.value || 0)}x${Number(widget(node,"height")?.value || 0)}`;
      [["CUSTOM · keep current", "custom"], ["H3 LANDSCAPE · 768 × 448", "768x448"], ["H3 LANDSCAPE · 960 × 544", "960x544"], ["H3 LANDSCAPE · 1280 × 736", "1280x736"], ["H3 PORTRAIT · 448 × 768", "448x768"], ["H3 PORTRAIT · 544 × 960", "544x960"]].forEach(([label,value]) => format.add(new Option(label,value)));
      [["H3 LANDSCAPE · 1344 × 768", "1344x768"], ["H3 LANDSCAPE · 1536 × 864", "1536x864"], ["H3 LANDSCAPE · 1664 × 928", "1664x928"], ["H3 LANDSCAPE · 1920 × 1088", "1920x1088"], ["H3 LANDSCAPE · 2048 × 1152", "2048x1152"], ["H3 SCOPE · 1536 × 640", "1536x640"], ["H3 SCOPE · 1920 × 800", "1920x800"], ["H3 SCOPE · 2048 × 864", "2048x864"], ["H3 PORTRAIT · 768 × 1344", "768x1344"], ["H3 PORTRAIT · 1088 × 1920", "1088x1920"], ["H3 PORTRAIT · 1152 × 2048", "1152x2048"]].forEach(([label,value]) => format.add(new Option(label,value)));
      format.value = [...format.options].some((option) => option.value === current) ? current : "custom";
      format.onchange = () => { if (format.value === "custom") return; const [width,height] = format.value.split("x").map(Number); setValue(node,"width",width,false); setValue(node,"height",height,false); setValue(node,"image_width",width,false); setValue(node,"image_height",height,false); document.dispatchEvent(new CustomEvent("iamccs:h3-settings-changed", {detail:{source_node_id:node.id,recipe:"resolution"}})); refresh(); };
      recipes.prepend(format);
    }
    const grid = q("[data-grid]"); grid.className = "h3p-grid"; grid.replaceChildren();
    const names = fieldNames(group, mode);
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
  }
  function renderTruth(mode) {
    const values = [["Mode / media authority", `${mode} · Shotboard`], ["Memory tier", friendly(widget(node, "performance_profile")?.value)], ["Acceleration", friendly(widget(node, "acceleration")?.value)], ["Turbo LoRA", String(widget(node, "turbo_mode")?.value || "off") === "off" ? "OFF" : (widget(node, "turbo_lora_name")?.value || "MISSING")], ["Fused model", widget(node, "fused_turbo_model_name")?.value || "OFF"], ["Sampling", `${widget(node, "steps")?.value ?? "—"} steps · ${widget(node, "sampler_name")?.value ?? "—"} · ${widget(node, "scheduler")?.value ?? "—"}`], ["Canvas", `${widget(node, "width")?.value ?? "—"} × ${widget(node, "height")?.value ?? "—"}`], ["Motion window", `${widget(node, "motion_context_window_frames")?.value ?? "—"} frames`], ["ControlNet", widget(node, "h3_controlnet_name")?.value || "OFF"], ["Delivery", widget(node, "upscale_enabled")?.value ? widget(node, "upscale_mode")?.value : "NATIVE"]];
    const list = q(".h3p-truth-list"); list.replaceChildren(); values.forEach(([label, value]) => { const row = document.createElement("div"); row.className = "h3p-truth-row"; row.innerHTML = `<span>${label}</span><b>${String(value ?? "—")}</b>`; list.append(row); });
    const issueList = warnings(mode), health = q(".h3p-health"); health.className = `h3p-health ${issueList.some(([kind]) => kind === "error") ? "error" : issueList.some(([kind]) => kind === "warn") ? "warn" : ""}`; health.innerHTML = issueList.map(([, message]) => `• ${message}`).join("<br>");
  }
  function refresh() { const mode = shotboardMode(node); lastMode = mode; q("[data-mode]").textContent = mode.toUpperCase(); renderRail(mode); renderMain(mode); renderTruth(mode); }
  node._iamccsSettingsProRefresh = refresh; const dom = node.addDOMWidget("H3 Settings PRO", "iamccs_h3_settings_pro_panel", root, { serialize: false }); dom.computeSize = (width) => [Math.max(1080, Number(width || 1080)), 680];
  node.setSize?.([1140, 750]); node.color = "#33291d"; node.bgcolor = "#0d131a"; refresh(); node._iamccsSettingsProTimer = window.setInterval(() => { const mode = shotboardMode(node); if (mode !== lastMode) refresh(); }, 800);
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
      window.setTimeout(() => { mount(this); this._iamccsSettingsProRefresh?.(); }, 0);
      return result;
    };
    const serialized = nodeType.prototype.onSerialize; nodeType.prototype.onSerialize = function (info) {
      const result = serialized?.apply(this, arguments);
      serializeNamedValues(this, info);
      return result;
    };
  },
});
