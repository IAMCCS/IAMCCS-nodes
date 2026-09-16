import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TYPE = "IAMCCS_NextFrameBuilder";
const STYLE_ID = "iamccs-next-frame-builder-style-v10";
const UI_VERSION = "1.6.1";
const FIXED_UI_HEIGHT = 1160;
const PROJECT_SCHEMA = "iamccs.next_frame_builder.project.v1";
const DIRECTOR_TEMPLATE = "Next Scene: [camera movement] to a [shot size / angle] as the same subject [one clear action beat]. Preserve the exact identity, face, hairstyle, wardrobe, body proportions, location geometry, color palette, lighting direction and cinematic style of Image 1. Maintain spatial continuity and realistic atmospheric depth.";

const INJECT_TARGETS = {
  minimax: { label: "MiniMax H3", type: "IAMCCS_MiniMaxH3ShotPlanner" },
  v3: { label: "Shotboard V3", type: "IAMCCS_CineShotboardPlannerV3" },
  bridge: { label: "H3 Bridge", type: "IAMCCS_MiniMaxH3BridgeLoad" },
};

const MODEL_FIELDS = {
  gguf_model: ["UnetLoaderGGUF", "unet_name"],
  native_model: ["UNETLoader", "unet_name"],
  clip_model: ["CLIPLoader", "clip_name"],
  vae_model: ["VAELoader", "vae_name"],
  lightning_lora: ["LoraLoaderModelOnly", "lora_name"],
  next_scene_lora: ["LoraLoaderModelOnly", "lora_name"],
  light_lora: ["LoraLoaderModelOnly", "lora_name"],
};

const SETTINGS_SECTIONS = [
  {
    title: "Model stack",
    fields: [
      ["model_loader", "Loader", "select"],
      ["gguf_model", "GGUF diffusion model", "model"],
      ["native_model", "Native diffusion model", "model"],
      ["clip_model", "Qwen text encoder", "model"],
      ["vae_model", "VAE", "model"],
    ],
  },
  {
    title: "LoRA stack",
    fields: [
      ["lightning_lora", "Lightning 4-step", "model"],
      ["lightning_strength", "Lightning strength", "number"],
      ["next_scene_lora", "Scene / Multi-Angle LoRA", "model"],
      ["next_scene_strength", "Scene LoRA strength", "number"],
      ["light_lora", "Light & Shadow LoRA", "model"],
      ["light_strength", "Light strength", "number"],
    ],
  },
  {
    title: "Generation",
    fields: [
      ["width", "Width", "number"], ["height", "Height", "number"],
      ["seed", "Seed", "number"], ["seed_mode", "Seed mode", "select"],
      ["steps", "Steps", "number"], ["cfg", "CFG", "number"],
      ["sampler_name", "Sampler", "select"], ["scheduler", "Scheduler", "select"],
      ["denoise", "Denoise", "number"], ["shift", "AuraFlow shift", "number"],
      ["cfg_norm_strength", "CFG Norm", "number"],
      ["reference_method", "Reference method", "select"],
      ["conditioning_megapixels", "Reference MP", "number"],
      ["decode_tile_size", "Decode tile", "number"],
      ["inject_slot_seconds", "Injected slot duration (s)", "number"],
    ],
  },
];

const PROJECT_WIDGET_FIELDS = [
  "source_image", "prompt_text", "negative_prompt", "session_id",
  ...SETTINGS_SECTIONS.flatMap((section) => section.fields.map(([name]) => name)),
  "reference_image_2", "reference_image_3",
];

const SAFE_GENERATION_DEFAULTS = {
  lightning_strength: 1, next_scene_lora: "next-scene_lora-v2-3000.safetensors",
  next_scene_strength: 0.8, light_lora: "qwen2511_Add_light_and_shadow.safetensors",
  light_strength: 0.8, width: 1920, height: 1088, seed: 473146755093516,
  seed_mode: "randomize", steps: 4, cfg: 1, sampler_name: "euler", scheduler: "simple",
  denoise: 1, shift: 3.1, cfg_norm_strength: 1, reference_method: "index_timestep_zero",
  conditioning_megapixels: 3, decode_tile_size: 512, inject_slot_seconds: 5,
};
const NUMERIC_WIDGETS = new Set([
  "lightning_strength", "next_scene_strength", "light_strength", "width", "height", "seed",
  "steps", "cfg", "denoise", "shift", "cfg_norm_strength", "conditioning_megapixels",
  "decode_tile_size", "inject_slot_seconds",
]);
const REFERENCE_DEFAULTS = {
  reference_image_2: { enabled: true, role: "second_character" },
  reference_image_3: { enabled: true, role: "environment" },
};
const REFERENCE_ROLE_PROMPTS = {
  second_character: "the additional character whose identity, face, hair, body proportions and wardrobe must be preserved",
  object: "the object or prop to integrate without importing its original background",
  environment: "the environment or location reference for architecture, geography and spatial atmosphere",
  style: "the visual style or wardrobe reference, without replacing the identities from the other images",
  lighting: "the lighting reference for direction, contrast, color temperature and atmosphere",
};

function nodeType(node) {
  return String(node?.comfyClass || node?.type || node?.constructor?.type || "");
}

function widget(node, name) {
  return (node.widgets || []).find((item) => item?.name === name);
}

function read(node, name, fallback = "") {
  const item = widget(node, name);
  return item ? item.value : fallback;
}

function write(node, name, value) {
  const item = widget(node, name);
  if (!item) return;
  item.value = value;
  try { item.callback?.(value); } catch {}
  try { node.setDirtyCanvas?.(true, true); } catch {}
  try { app.graph?.setDirtyCanvas?.(true, true); } catch {}
}

function hideWidget(item) {
  if (!item) return;
  item._iamccsNextFrameOriginalType ||= item.type;
  item.hidden = true;
  item.type = item._iamccsNextFrameOriginalType;
  item.computeSize = () => [0, 0];
  item.draw = () => {};
  item.serialize = true;
  item.serializeValue = () => item.value;
  // ComfyUI 1.51+ uses options.serialize for API-prompt inclusion, while
  // widget.serialize controls workflow persistence. Keep both enabled: the
  // native controls are visually hidden, but their named values must remain
  // in the generated prompt.
  item.options = { ...(item.options || {}), hidden: true, serialize: true };
  if (item.inputEl) {
    item.inputEl.hidden = true;
    item.inputEl.style.display = "none";
  }
}

function parseBoard(value) {
  try {
    const parsed = JSON.parse(String(value || "{}"));
    return parsed && typeof parsed === "object" ? parsed : { frames: [] };
  } catch {
    return { schema: "iamccs.next_frame_builder.storyboard.v1", frames: [] };
  }
}

function framesOf(board) {
  return Array.isArray(board?.frames) ? board.frames : [];
}

function injectTargetsOf(board) {
  const saved = board?.inject_targets;
  return {
    minimax: saved?.minimax !== false,
    v3: saved?.v3 !== false,
    bridge: saved?.bridge !== false,
  };
}

function injectionSelection(board) {
  const frames = framesOf(board);
  let anchorIndex = frames.findIndex((frame) => String(frame?.id || "") === String(board?.inject_anchor_id || ""));
  if (anchorIndex < 0) anchorIndex = Math.max(0, frames.findIndex((frame) => frame?.selected !== false));
  if (!frames.length) anchorIndex = 0;
  return { anchorIndex, frames: frames.slice(anchorIndex) };
}

function buildInjectionTimeline(frames, slotSeconds, prompt, negativePrompt, startSlot = 0) {
  const fps = 24;
  const seconds = Math.max(0.25, Number(slotSeconds) || 5);
  const slotFrames = Math.max(1, Math.round(seconds * fps));
  startSlot = Math.max(0, Number(startSlot) || 0);
  const segments = frames.map((frame, index) => {
    const localPrompt = frame.role === "source" && frame.prompt === "Source frame" ? "" : String(frame.prompt || "");
    const absoluteIndex = startSlot + index;
    const start = absoluteIndex * slotFrames;
    return {
      id: String(frame.id || `nextframe_${absoluteIndex + 1}`), type: "image", label: `NextFrame ${absoluteIndex + 1}`,
      start, frame: start, second: start / fps, length: slotFrames, ref: absoluteIndex + 1,
      imageFile: String(frame.filename || ""), image_file: String(frame.filename || ""),
      imageTruthPath: String(frame.filename || ""), fileName: fileParts(frame.filename).filename,
      prompt: localPrompt, local_prompt: localPrompt, relay_prompt: localPrompt, note: localPrompt,
      use_guide: true, use_prompt: Boolean(localPrompt), guideStrength: 1, guide_strength: 1,
      force: 1, source: TYPE, nextframe_selected: true,
    };
  });
  return {
    schema: "iamccs.next_frame_builder.injection_timeline.v1", source: TYPE,
    frame_rate: fps, fps, start_slot: startSlot, duration_seconds: Math.max(0.1, (startSlot + segments.length) * seconds),
    global_prompt: String(prompt || ""), prompt: String(prompt || ""),
    negative_prompt: String(negativePrompt || ""), segments,
    rows: segments.map((segment) => ({ ...segment })), audioSegments: [],
  };
}

function parseImagePaths(value) {
  const text = String(value || "").trim();
  if (!text) return [];
  try {
    const parsed = JSON.parse(text);
    if (Array.isArray(parsed)) return parsed.map(String).filter(Boolean);
  } catch {}
  return text.split(/[\r\n,;]+/).map((item) => item.trim()).filter(Boolean);
}

function mergeInjection(target, timeline, newPaths, startSlot) {
  let existing = {};
  try { existing = JSON.parse(String(read(target, "timeline_data", "{}") || "{}")); } catch {}
  if (!existing || typeof existing !== "object" || Array.isArray(existing)) existing = {};
  const oldSegments = Array.isArray(existing.segments) ? existing.segments : [];
  const oldRows = Array.isArray(existing.rows) ? existing.rows : [];
  const prefixSegments = oldSegments.slice(0, startSlot);
  const prefixRows = oldRows.slice(0, startSlot);
  const merged = {
    ...existing, ...timeline,
    segments: [...prefixSegments, ...timeline.segments],
    rows: [...prefixRows, ...timeline.rows],
    duration_seconds: Math.max(Number(existing.duration_seconds) || 0, Number(timeline.duration_seconds) || 0),
  };
  const paths = [...parseImagePaths(read(target, "image_paths", "")).slice(0, startSlot), ...newPaths];
  return { timeline: merged, paths };
}

function firstValue(value, fallback = "") {
  return Array.isArray(value) ? (value[0] ?? fallback) : (value ?? fallback);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}

function fileParts(path) {
  const normalized = String(path || "").replaceAll("\\", "/");
  const pieces = normalized.split("/").filter(Boolean);
  return { filename: pieces.pop() || "", subfolder: pieces.join("/") };
}

function imageUrl(path, type = "input") {
  const { filename, subfolder } = fileParts(path);
  if (!filename) return "";
  const params = new URLSearchParams({ filename, type, subfolder, t: String(Date.now()) });
  return api.apiURL(`/view?${params.toString()}`);
}

function installStyle() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
    .iamccs-nfb{--bg:#080b11;--panel:#10151e;--panel2:#151c27;--line:#273244;--text:#f4f7fb;--muted:#8f9bad;--cyan:#4ee1d2;--cyan2:#1db5ab;--amber:#ffbd69;--danger:#ff6b76;position:relative;display:flex;flex-direction:column;height:1160px;color:var(--text);font:13px/1.35 Inter,Segoe UI,sans-serif;background:radial-gradient(circle at 50% -20%,#173142 0,transparent 38%),linear-gradient(160deg,#0c1119,#07090e);border:1px solid #263246;border-radius:15px;overflow:hidden;box-shadow:0 18px 46px #0008;min-width:940px;user-select:none}
    .iamccs-nfb:fullscreen{display:block;width:100vw;height:100vh;min-width:0;border:0;border-radius:0;overflow:auto;background:radial-gradient(circle at 50% -10%,#173b50 0,transparent 35%),#070a0f}.iamccs-nfb:fullscreen .iamccs-nfb-main,.iamccs-nfb:fullscreen .iamccs-nfb-board{max-width:1600px;margin-left:auto;margin-right:auto}.iamccs-nfb:fullscreen .iamccs-nfb-preview{min-height:36vh}.iamccs-nfb:fullscreen .iamccs-nfb-grid{grid-template-columns:repeat(6,minmax(0,1fr));max-height:42vh}
    .iamccs-nfb *{box-sizing:border-box}.iamccs-nfb button,.iamccs-nfb input,.iamccs-nfb textarea,.iamccs-nfb select{font:inherit}.iamccs-nfb button{cursor:pointer}
    .iamccs-nfb-head{flex:0 0 62px;height:62px;display:flex;align-items:center;justify-content:space-between;padding:0 18px;border-bottom:1px solid var(--line);background:#0d121bcc;backdrop-filter:blur(12px)}
    .iamccs-nfb-brand{display:flex;align-items:center;gap:11px}.iamccs-nfb-mark{width:34px;height:34px;border-radius:10px;background:linear-gradient(135deg,#55f0df,#166b8e);box-shadow:0 0 26px #35d4c64d;display:grid;place-items:center;color:#031110;font-weight:900}.iamccs-nfb-title{font-size:16px;font-weight:760;letter-spacing:.01em}.iamccs-nfb-sub{font-size:11px;color:var(--muted);margin-top:2px}.iamccs-nfb-head-actions{display:flex;align-items:center;gap:8px}
    .iamccs-nfb-status{display:flex;align-items:center;gap:7px;color:var(--muted);font-size:11px;margin-right:5px}.iamccs-nfb-status i{width:7px;height:7px;border-radius:50%;background:#5ee0a1;box-shadow:0 0 10px #5ee0a1}.iamccs-nfb-status.busy i{background:var(--amber);animation:nfbPulse 1s infinite}.iamccs-nfb-status.error i{background:var(--danger)}@keyframes nfbPulse{50%{opacity:.25}}
    .iamccs-nfb-icon,.iamccs-nfb-ghost{border:1px solid var(--line);background:#151c27;color:#dce4ee;border-radius:9px;min-height:34px;padding:7px 11px}.iamccs-nfb-icon:hover,.iamccs-nfb-ghost:hover{border-color:#3f526e;background:#1a2432}.iamccs-nfb-icon{width:36px;padding:0;font-size:16px}
    .iamccs-nfb-main{flex:0 0 auto;padding:16px}.iamccs-nfb-compare{display:grid;grid-template-columns:minmax(0,1fr) 62px minmax(0,1fr);align-items:stretch;gap:10px}.iamccs-nfb-preview{position:relative;min-height:260px;border:1px solid var(--line);border-radius:13px;overflow:hidden;background:linear-gradient(145deg,#121923,#090d13);box-shadow:inset 0 0 0 1px #ffffff05}.iamccs-nfb-preview.drag{border-color:var(--cyan);box-shadow:inset 0 0 0 2px #4ee1d244,0 0 28px #4ee1d21f}.iamccs-nfb-preview img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#070a0f}.iamccs-nfb-empty{position:absolute;inset:0;display:grid;place-items:center;text-align:center;color:var(--muted);padding:28px}.iamccs-nfb-empty b{display:block;color:#dce4ee;font-size:14px;margin:8px 0 3px}.iamccs-nfb-empty span{font-size:11px}.iamccs-nfb-preview.has-image .iamccs-nfb-empty{display:none}
    .iamccs-nfb-preview-top{position:absolute;z-index:2;left:10px;right:10px;top:10px;display:flex;justify-content:space-between;align-items:center;pointer-events:none}.iamccs-nfb-chip{background:#05080cc9;border:1px solid #ffffff1c;color:#e6edf6;border-radius:999px;padding:5px 9px;font-size:10px;letter-spacing:.08em;text-transform:uppercase;backdrop-filter:blur(7px)}.iamccs-nfb-preview-actions{position:absolute;z-index:2;left:10px;right:10px;bottom:10px;display:flex;justify-content:space-between;gap:8px}.iamccs-nfb-preview-actions button{border:1px solid #ffffff24;background:#070b10d9;color:white;border-radius:8px;padding:7px 10px;backdrop-filter:blur(8px)}.iamccs-nfb-preview-actions button:hover{background:#172331}.iamccs-nfb-file{max-width:62%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#b9c5d3;font-size:10px;align-self:center;background:#05080cbf;padding:5px 8px;border-radius:7px}
    .iamccs-nfb-arrow{display:grid;place-items:center;color:#64738a}.iamccs-nfb-arrow div{width:44px;height:44px;border-radius:50%;border:1px solid #304057;background:#121a25;display:grid;place-items:center;font-size:23px;box-shadow:0 7px 18px #0007}.iamccs-nfb.busy .iamccs-nfb-arrow div{color:var(--cyan);border-color:#4ee1d277;animation:nfbPulse 1s infinite}
    .iamccs-nfb-reference-deck{margin-top:10px;border:1px solid #29374b;border-radius:12px;background:#0d131c;overflow:hidden}.iamccs-nfb-reference-head{display:flex;justify-content:space-between;align-items:center;padding:7px 10px;border-bottom:1px solid #243043}.iamccs-nfb-reference-head b{font-size:10px;letter-spacing:.09em;text-transform:uppercase;color:#b9c7d8}.iamccs-nfb-reference-head span{font-size:9px;color:#728096}.iamccs-nfb-reference-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:8px}.iamccs-nfb-reference-card{display:grid;grid-template-columns:112px minmax(0,1fr);height:88px;border:1px solid #28374a;border-radius:9px;background:#101823;overflow:hidden;transition:opacity .15s,border-color .15s}.iamccs-nfb-reference-card.disabled{opacity:.48}.iamccs-nfb-reference-card.drag{border-color:var(--cyan);box-shadow:inset 0 0 0 1px #4ee1d255}.iamccs-nfb-reference-thumb{position:relative;background:#070a0f;border-right:1px solid #28374a;overflow:hidden}.iamccs-nfb-reference-thumb img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}.iamccs-nfb-reference-thumb .empty{position:absolute;inset:0;display:grid;place-items:center;text-align:center;color:#627188;font-size:9px;padding:8px}.iamccs-nfb-reference-card.has-image .empty{display:none}.iamccs-nfb-reference-badge{position:absolute;left:5px;top:5px;border:1px solid #ffffff26;border-radius:5px;background:#05080cce;color:#dce7f4;padding:3px 5px;font-size:8px;font-weight:800}.iamccs-nfb-reference-info{min-width:0;display:grid;grid-template-rows:auto auto 1fr;padding:7px 8px;gap:5px}.iamccs-nfb-reference-top{display:flex;align-items:center;justify-content:space-between;gap:7px}.iamccs-nfb-reference-toggle{display:flex;align-items:center;gap:4px;color:#8ff0e7;font-size:9px;white-space:nowrap}.iamccs-nfb-reference-toggle input{accent-color:#31cdbf}.iamccs-nfb-reference-role{width:100%;height:25px;border:1px solid #304158;border-radius:6px;background:#0a1018;color:#dbe5ef;padding:2px 6px;font-size:9px}.iamccs-nfb-reference-bottom{display:flex;align-items:end;gap:5px;min-width:0}.iamccs-nfb-reference-file{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#758398;font-size:8px}.iamccs-nfb-reference-bottom button{border:1px solid #34465e;border-radius:6px;background:#182332;color:#d7e1eb;padding:3px 7px;font-size:8px}.iamccs-nfb-reference-bottom button.remove{flex:0 0 auto;color:#f29aa2;border-color:#4a3039}
    .iamccs-nfb-prompts{display:grid;grid-template-columns:1.45fr 1fr;gap:10px;margin-top:13px}.iamccs-nfb-prompt{border:1px solid var(--line);background:#0d131c;border-radius:12px;overflow:hidden}.iamccs-nfb-prompt.negative{border-color:#3a2a34}.iamccs-nfb-prompt-head{display:flex;align-items:center;justify-content:space-between;padding:9px 12px;border-bottom:1px solid #202a38;color:#bcc7d5;font-size:11px;text-transform:uppercase;letter-spacing:.08em}.iamccs-nfb-prompt.negative .iamccs-nfb-prompt-head{color:#cfadb7;border-bottom-color:#35252e}.iamccs-nfb-prompt textarea{display:block;width:100%;height:92px;resize:vertical;border:0;outline:0;background:#0b1018;color:#eef3f8;padding:12px 13px;line-height:1.45;user-select:text}.iamccs-nfb-prompt.negative textarea{background:#120d13;color:#eadde1}.iamccs-nfb-prompt textarea::placeholder{color:#647084}.iamccs-nfb-actions{display:flex;justify-content:space-between;align-items:center;padding-top:12px;gap:10px}.iamccs-nfb-primary{border:0;border-radius:10px;padding:11px 19px;background:linear-gradient(135deg,#52eadb,#23b9b0);color:#041312;font-weight:800;box-shadow:0 8px 24px #27cfc139}.iamccs-nfb-primary:hover{filter:brightness(1.08)}.iamccs-nfb-primary:disabled{opacity:.45;cursor:wait}.iamccs-nfb-use{border:1px solid #4ee1d276;border-radius:10px;padding:10px 15px;background:#0d2828;color:#8ef6eb;font-weight:700}.iamccs-nfb-use:disabled{opacity:.35;cursor:not-allowed}
    .iamccs-nfb-prompt-head button{border:1px solid #335166;border-radius:7px;background:#122431;color:#75e6dc;padding:4px 7px;font-size:9px;text-transform:uppercase;letter-spacing:.05em}.iamccs-nfb-prompt-head button:hover{border-color:#53bcb3;background:#17343d}
    .iamccs-nfb-prompt-tools{display:flex;gap:6px}.iamccs-nfb-prompt-head button.ai{border-color:#6c4db4;background:linear-gradient(135deg,#241d40,#182d3d);color:#cbb7ff}.iamccs-nfb-ai{display:none;margin-top:10px;border:1px solid #3a315d;border-radius:12px;background:linear-gradient(145deg,#151225,#0d151f);padding:11px;box-shadow:inset 0 1px #ffffff0a}.iamccs-nfb.ai-open .iamccs-nfb-ai{display:block}.iamccs-nfb-ai-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:9px}.iamccs-nfb-ai-head b{font-size:12px;color:#ddd2ff}.iamccs-nfb-ai-head span{font-size:10px;color:#81759f}.iamccs-nfb-ai-fields{display:grid;grid-template-columns:150px 1fr 1fr 38px;gap:7px}.iamccs-nfb-ai input,.iamccs-nfb-ai select{width:100%;height:34px;border:1px solid #393552;border-radius:7px;background:#090d15;color:#e8e4f3;padding:6px 8px;outline:none;user-select:text}.iamccs-nfb-ai input:focus,.iamccs-nfb-ai select:focus{border-color:#7f6cc0}.iamccs-nfb-ai .refresh{border:1px solid #423a61;border-radius:7px;background:#19152a;color:#c8baf4}.iamccs-nfb-ai-foot{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-top:8px}.iamccs-nfb-ai-note{font-size:10px;color:#887f9e}.iamccs-nfb-ai-run{border:1px solid #816aca;border-radius:8px;background:linear-gradient(135deg,#533f93,#296372);color:white;padding:7px 12px;font-weight:750}.iamccs-nfb-ai-run:disabled{opacity:.45;cursor:wait}.iamccs-nfb-ai-progress{display:none;align-items:center;gap:7px;margin-top:8px;padding:7px 9px;border:1px solid #514579;border-radius:8px;background:#0b0d19;color:#bdb1df;font-size:10px}.iamccs-nfb-ai-progress.active{display:flex}.iamccs-nfb-ai-progress i{width:14px;height:14px;border:2px solid #6f638e;border-top-color:#6bf0e2;border-radius:50%;animation:nfbSpin .75s linear infinite}.iamccs-nfb-ai-progress b{margin-left:auto;color:#7ff0e5;font-variant-numeric:tabular-nums}@keyframes nfbSpin{to{transform:rotate(360deg)}}
    .iamccs-nfb-idea-overlay{position:absolute;z-index:35;inset:0;display:none;align-items:center;justify-content:center;padding:28px;background:#03060bd9;backdrop-filter:blur(12px)}.iamccs-nfb.idea-open .iamccs-nfb-idea-overlay{display:flex}.iamccs-nfb-idea-dialog{width:min(920px,96%);max-height:calc(100% - 28px);display:flex;flex-direction:column;border:1px solid #4b4270;border-radius:16px;overflow:hidden;background:radial-gradient(circle at 75% -15%,#29315a 0,transparent 35%),linear-gradient(155deg,#171329,#0b121c 58%,#090d14);box-shadow:0 28px 90px #000c}.iamccs-nfb-idea-head{flex:0 0 auto;display:flex;align-items:center;justify-content:space-between;padding:15px 17px;border-bottom:1px solid #393453;background:#111321bd}.iamccs-nfb-idea-head h3{margin:0;color:#eee9ff;font-size:16px}.iamccs-nfb-idea-head p{margin:3px 0 0;color:#938aa9;font-size:10px}.iamccs-nfb-idea-body{flex:1 1 auto;min-height:0;display:grid;grid-template-columns:310px minmax(0,1fr)}.iamccs-nfb-idea-form{padding:15px;border-right:1px solid #343149;background:#0c1019b8;overflow:auto}.iamccs-nfb-idea-form label{display:block;margin-bottom:5px;color:#aca2c3;font-size:10px;text-transform:uppercase;letter-spacing:.07em}.iamccs-nfb-idea-form textarea{width:100%;height:150px;resize:vertical;border:1px solid #3e3958;border-radius:9px;outline:0;background:#080c14;color:#f0edf8;padding:11px;line-height:1.45;user-select:text}.iamccs-nfb-idea-form textarea:focus{border-color:#8673cc}.iamccs-nfb-idea-controls{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px}.iamccs-nfb-idea-controls select{width:100%;height:34px;border:1px solid #3b3853;border-radius:7px;background:#0a0e17;color:#e5e0ef;padding:5px 8px}.iamccs-nfb-idea-provider{margin-top:11px;padding:9px;border:1px solid #302d45;border-radius:8px;background:#11131d;color:#8f87a4;font-size:9px;line-height:1.45}.iamccs-nfb-idea-provider b{color:#c8bdea}.iamccs-nfb-idea-generate{width:100%;margin-top:11px;border:1px solid #8b72db;border-radius:9px;background:linear-gradient(135deg,#654bb4,#257681);color:#fff;padding:10px;font-weight:800}.iamccs-nfb-idea-generate:disabled{opacity:.45;cursor:wait}.iamccs-nfb-idea-warning{margin-top:8px;color:#7e7691;font-size:9px;line-height:1.4}.iamccs-nfb-idea-progress{display:none;align-items:center;gap:8px;margin-top:10px;padding:8px 9px;border:1px solid #4c426c;border-radius:8px;color:#beb2dc;font-size:10px}.iamccs-nfb-idea-progress.active{display:flex}.iamccs-nfb-idea-progress i{width:15px;height:15px;border:2px solid #625879;border-top-color:#65f0e2;border-radius:50%;animation:nfbSpin .75s linear infinite}.iamccs-nfb-idea-progress b{margin-left:auto;color:#71eadf;font-variant-numeric:tabular-nums}.iamccs-nfb-idea-results{min-height:0;overflow:auto;padding:14px;display:grid;grid-template-columns:1fr 1fr;align-content:start;gap:10px}.iamccs-nfb-idea-empty{grid-column:1/-1;min-height:210px;display:grid;place-items:center;text-align:center;border:1px dashed #3b3752;border-radius:11px;color:#777087;padding:25px}.iamccs-nfb-idea-card{border:1px solid #38354e;border-radius:10px;overflow:hidden;background:#10141e;box-shadow:0 8px 20px #0005}.iamccs-nfb-idea-card-head{display:flex;align-items:center;gap:8px;padding:9px 10px;border-bottom:1px solid #302e43;background:#171827}.iamccs-nfb-idea-card-num{flex:0 0 23px;width:23px;height:23px;border-radius:6px;display:grid;place-items:center;background:#58439a;color:#fff;font-size:9px;font-weight:850}.iamccs-nfb-idea-card-head b{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#e6e0f2;font-size:11px}.iamccs-nfb-idea-card-body{padding:10px}.iamccs-nfb-idea-beat{min-height:29px;color:#a9a1b8;font-size:9px;line-height:1.45}.iamccs-nfb-idea-prompt{max-height:94px;overflow:auto;margin-top:8px;padding:8px;border-radius:7px;background:#080d14;color:#cbd4df;font-size:9px;line-height:1.45;user-select:text}.iamccs-nfb-idea-card-actions{display:flex;gap:6px;margin-top:9px}.iamccs-nfb-idea-card-actions button{flex:1;border:1px solid #3d3b55;border-radius:7px;background:#1a1d2a;color:#c8c1d5;padding:6px;font-size:9px}.iamccs-nfb-idea-card-actions button.use{border-color:#397c78;background:#12302f;color:#88eee4;font-weight:750}.iamccs-nfb-idea-close{border:1px solid #403b59;border-radius:8px;background:#1b1b2a;color:#d8d2e3;padding:7px 10px}
    .iamccs-nfb-board{flex:1 1 auto;min-height:0;display:flex;flex-direction:column;overflow:hidden;border-top:1px solid var(--line);padding:15px 16px 17px;background:#090d13}.iamccs-nfb-board-head{flex:0 0 auto;display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}.iamccs-nfb-board-title{font-size:12px;font-weight:750;letter-spacing:.08em;text-transform:uppercase}.iamccs-nfb-count{color:var(--muted);font-weight:500;margin-left:7px}.iamccs-nfb-board-tools{display:flex;gap:7px}.iamccs-nfb-board-tools button,.iamccs-nfb-injectbar button{border:1px solid var(--line);background:#111823;color:#aeb9c8;border-radius:8px;padding:6px 9px;font-size:11px}.iamccs-nfb-injectbar{flex:0 0 auto;display:flex;align-items:center;gap:7px;margin-bottom:11px;padding:8px 9px;border:1px solid #223247;border-radius:10px;background:#0d141e}.iamccs-nfb-injectbar-label{margin-right:3px;color:#7f8da1;font-size:10px;text-transform:uppercase;letter-spacing:.08em}.iamccs-nfb-injectbar button.target.on{border-color:#3e948d;background:#12302f;color:#8cf2e8}.iamccs-nfb-injectbar button.inject{margin-left:auto;border-color:#55ded0;background:linear-gradient(135deg,#183f3d,#123032);color:#8ff5eb;font-weight:750}
    .iamccs-nfb-grid{flex:1 1 auto;min-height:92px;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));align-content:start;gap:9px;max-height:none;overflow:auto;padding:2px 6px 8px 2px;scrollbar-gutter:stable}.iamccs-nfb-card{position:relative;border:1px solid #263246;border-radius:10px;background:#111823;overflow:hidden;min-width:0;opacity:.42}.iamccs-nfb-card:hover{border-color:#48617f}.iamccs-nfb-card.chosen{opacity:1;border-color:#3f8f89;box-shadow:inset 0 0 0 1px #4ee1d21f}.iamccs-nfb-card.anchor{border-color:var(--cyan);box-shadow:0 0 0 1px #4ee1d258,0 0 20px #4ee1d214}.iamccs-nfb-card.active{border-color:var(--cyan)}.iamccs-nfb-thumb{aspect-ratio:16/9;background:#070a0f;position:relative}.iamccs-nfb-thumb img{width:100%;height:100%;object-fit:cover}.iamccs-nfb-num{position:absolute;left:6px;top:6px;width:23px;height:23px;border-radius:6px;display:grid;place-items:center;background:#05070bdc;border:1px solid #ffffff29;font-size:10px;font-weight:800}.iamccs-nfb-select{position:absolute;right:6px;top:6px;min-width:46px;height:26px!important;padding:0 6px!important;border-radius:7px!important;background:#080c12db!important;color:#708096!important;border:1px solid #ffffff26!important;font-size:9px!important;font-weight:800!important}.iamccs-nfb-card.chosen .iamccs-nfb-select{background:#16413ddd!important;color:#7ff5e8!important;border-color:#4ee1d28c!important}.iamccs-nfb-card.anchor .iamccs-nfb-select{background:#24655f!important;color:white!important}.iamccs-nfb-card-body{padding:7px}.iamccs-nfb-card-prompt{height:30px;overflow:hidden;color:#aeb9c7;font-size:10px;line-height:1.45}.iamccs-nfb-card-buttons{display:flex;gap:5px;margin-top:6px}.iamccs-nfb-card-buttons button{flex:1;border:1px solid #2b394c;background:#182230;color:#cbd6e3;border-radius:6px;padding:4px;font-size:9px}.iamccs-nfb-card-buttons button:last-child{flex:0 0 25px;color:#f39aa1}.iamccs-nfb-board-empty{grid-column:1/-1;min-height:88px;border:1px dashed #293548;border-radius:10px;display:grid;place-items:center;text-align:center;color:#66758a;font-size:11px}
    .iamccs-nfb.ai-open .iamccs-nfb-grid{max-height:270px}.iamccs-nfb:fullscreen.ai-open .iamccs-nfb-grid{max-height:42vh}
    .iamccs-nfb-drawer{position:absolute;z-index:20;top:0;right:0;width:390px;height:100%;background:#0c1119f7;border-left:1px solid #304058;box-shadow:-22px 0 55px #0009;transform:translateX(102%);transition:transform .2s ease;display:flex;flex-direction:column;backdrop-filter:blur(16px)}.iamccs-nfb.settings-open .iamccs-nfb-drawer{transform:none}.iamccs-nfb-drawer-head{display:flex;justify-content:space-between;align-items:center;padding:16px;border-bottom:1px solid var(--line)}.iamccs-nfb-drawer-head b{font-size:15px}.iamccs-nfb-drawer-body{padding:13px 15px 24px;overflow:auto;user-select:text}.iamccs-nfb-section{border:1px solid #243145;border-radius:11px;margin-bottom:11px;overflow:hidden;background:#101722}.iamccs-nfb-section h4{margin:0;padding:9px 11px;background:#151e2b;border-bottom:1px solid #243145;font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:#b8c5d4}.iamccs-nfb-fields{display:grid;grid-template-columns:1fr 1fr;gap:9px;padding:10px}.iamccs-nfb-field.wide{grid-column:1/-1}.iamccs-nfb-field label{display:block;color:#8997aa;font-size:10px;margin-bottom:4px}.iamccs-nfb-field input,.iamccs-nfb-field select{width:100%;height:33px;border:1px solid #2b394c;border-radius:7px;background:#0a1018;color:#e4ebf3;padding:5px 8px;outline:none;user-select:text}.iamccs-nfb-field input:focus,.iamccs-nfb-field select:focus{border-color:#4ee1d28c}.iamccs-nfb-ref{display:grid;grid-template-columns:1fr auto;gap:6px}.iamccs-nfb-ref button{border:1px solid #33445c;background:#182231;color:#d4deea;border-radius:7px;padding:0 9px}.iamccs-nfb-footnote{font-size:10px;color:#718095;padding:2px 3px 12px;line-height:1.5}
    .iamccs-nfb-idea-message{display:none;margin-top:9px;padding:9px 10px;border:1px solid #3e4960;border-radius:8px;background:#111a27;color:#b8c6d8;font-size:10px;line-height:1.4;white-space:pre-wrap}.iamccs-nfb-idea-message.active{display:block}.iamccs-nfb-idea-message.busy{border-color:#71613e;background:#211b11;color:#ffd28a}.iamccs-nfb-idea-message.error{border-color:#743d49;background:#241116;color:#ff9eaa}.iamccs-nfb-idea-message.success{border-color:#31736d;background:#0e2422;color:#82eee3}
    @media(max-width:980px){.iamccs-nfb{min-width:760px}.iamccs-nfb-grid{grid-template-columns:repeat(3,1fr)}.iamccs-nfb-prompts{grid-template-columns:1fr}.iamccs-nfb-ai-fields{grid-template-columns:1fr 1fr}.iamccs-nfb-idea-body{grid-template-columns:270px 1fr}.iamccs-nfb-idea-results{grid-template-columns:1fr}}
  `;
  document.head.appendChild(style);
}

function setImage(panel, img, filenameEl, path) {
  const url = imageUrl(path, "input");
  if (!url) {
    panel.classList.remove("has-image");
    img.removeAttribute("src");
    filenameEl.textContent = "No image";
    return;
  }
  img.src = url;
  panel.classList.add("has-image");
  filenameEl.textContent = fileParts(path).filename;
  filenameEl.title = String(path);
}

async function uploadFile(file) {
  const form = new FormData();
  form.append("image", file, file.name || `iamccs_source_${Date.now()}.png`);
  form.append("overwrite", "true");
  const response = await api.fetchApi("/upload/image", { method: "POST", body: form });
  if (!response.ok) throw new Error(`Upload failed (${response.status})`);
  const data = await response.json();
  return [data.subfolder, data.name].filter(Boolean).join("/");
}

function downloadJson(filename, payload) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url; anchor.download = filename; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1500);
}

function comboValues(item) {
  const values = item?.options?.values;
  if (Array.isArray(values)) return values;
  if (typeof values === "function") {
    try { return values(); } catch {}
  }
  return [];
}

function makeSettings(node, drawerBody, uploadRef) {
  const datalists = new Map();
  for (const section of SETTINGS_SECTIONS) {
    const box = document.createElement("section");
    box.className = "iamccs-nfb-section";
    box.innerHTML = `<h4>${escapeHtml(section.title)}</h4><div class="iamccs-nfb-fields"></div>`;
    const fields = box.querySelector(".iamccs-nfb-fields");
    for (const [name, label, kind] of section.fields) {
      const item = widget(node, name);
      if (!item) continue;
      const wrap = document.createElement("div");
      wrap.className = `iamccs-nfb-field ${kind === "model" ? "wide" : ""}`;
      const labelEl = document.createElement("label"); labelEl.textContent = label;
      let control;
      if (kind === "select") {
        control = document.createElement("select");
        const values = comboValues(item);
        for (const value of values) {
          const option = document.createElement("option"); option.value = value; option.textContent = value;
          control.appendChild(option);
        }
      } else {
        control = document.createElement("input");
        control.type = kind === "number" ? "number" : "text";
        if (kind === "number") {
          if (item.options?.min != null) control.min = item.options.min;
          if (item.options?.max != null) control.max = item.options.max;
          if (item.options?.step != null) control.step = item.options.step;
        }
        if (kind === "model") {
          const listId = `iamccs-nfb-${node.id}-${name}`;
          const list = document.createElement("datalist"); list.id = listId;
          drawerBody.appendChild(list); control.setAttribute("list", listId); datalists.set(name, list);
        }
      }
      control.value = read(node, name, "");
      control.dataset.widget = name;
      control.addEventListener("change", () => write(node, name, control.type === "number" ? Number(control.value) : control.value));
      wrap.append(labelEl, control); fields.appendChild(wrap);
    }
    drawerBody.appendChild(box);
  }

  const refs = document.createElement("section");
  refs.className = "iamccs-nfb-section";
  refs.innerHTML = `<h4>Optional visual references</h4><div class="iamccs-nfb-fields"></div>`;
  const refFields = refs.querySelector(".iamccs-nfb-fields");
  for (const [name, label] of [["reference_image_2", "Identity / character"], ["reference_image_3", "Style / location"]]) {
    const wrap = document.createElement("div"); wrap.className = "iamccs-nfb-field wide";
    wrap.innerHTML = `<label>${label}</label><div class="iamccs-nfb-ref"><input type="text" data-widget="${name}"><button type="button">Load</button></div>`;
    const input = wrap.querySelector("input"); input.value = read(node, name, "");
    input.addEventListener("change", () => write(node, name, input.value));
    wrap.querySelector("button").addEventListener("click", () => uploadRef(name, input));
    refFields.appendChild(wrap);
  }
  drawerBody.appendChild(refs);
  const note = document.createElement("div"); note.className = "iamccs-nfb-footnote";
  note.textContent = "Defaults mirror the attached Qwen 2511 + Lightning + Next Scene + Light/Shadow workflow. The Scene LoRA slot also accepts a compatible Qwen 2511 Multi-Angle LoRA. Empty names or strength 0 bypass that LoRA.";
  drawerBody.appendChild(note);

  api.fetchApi("/object_info").then((response) => response.json()).then((info) => {
    for (const [field, [classType, inputName]] of Object.entries(MODEL_FIELDS)) {
      const list = datalists.get(field); if (!list) continue;
      const spec = info?.[classType]?.input?.required?.[inputName] || info?.[classType]?.input?.optional?.[inputName];
      const values = Array.isArray(spec?.[0]) ? spec[0] : [];
      list.replaceChildren(...values.map((value) => {
        const option = document.createElement("option"); option.value = value; return option;
      }));
    }
  }).catch((error) => console.debug("[IAMCCS NextFrameBuilder] model list unavailable", error));
  return () => {
    for (const control of drawerBody.querySelectorAll("[data-widget]")) {
      control.value = read(node, control.dataset.widget, "");
    }
  };
}

function mount(node) {
  if (!node || typeof node.addDOMWidget !== "function") return;
  if (node._iamccsNextFrameUiVersion === UI_VERSION) {
    node._iamccsNextFrameEnsureHidden?.();
    node.setSize([Math.max(980, Number(node.size?.[0] || 0)), FIXED_UI_HEIGHT + 44]);
    return;
  }
  node._iamccsNextFrameUiVersion = UI_VERSION;
  installStyle();
  (node.widgets || []).forEach(hideWidget);
  const backendWidgets = [...(node.widgets || [])];

  const root = document.createElement("div"); root.className = "iamccs-nfb";
  root.innerHTML = `
    <header class="iamccs-nfb-head">
      <div class="iamccs-nfb-brand"><div class="iamccs-nfb-mark">N›</div><div><div class="iamccs-nfb-title">IAMCCS NextFrameBuilder</div><div class="iamccs-nfb-sub">Qwen 2511 · iterative scene continuity · storyboard workspace</div></div></div>
      <div class="iamccs-nfb-head-actions"><div class="iamccs-nfb-status"><i></i><span>Ready</span></div><button class="iamccs-nfb-ghost" data-action="open-editor" title="Open the complete UI on this monitor">⛶ Open editor</button><button class="iamccs-nfb-icon" data-action="settings" title="Model and generation settings">⚙</button></div>
    </header>
    <main class="iamccs-nfb-main">
      <div class="iamccs-nfb-compare">
        <section class="iamccs-nfb-preview" data-preview="source"><img><div class="iamccs-nfb-empty"><div><div style="font-size:28px">＋</div><b>Load the current frame</b><span>Click, paste or drop an image here</span></div></div><div class="iamccs-nfb-preview-top"><span class="iamccs-nfb-chip">Current frame</span></div><div class="iamccs-nfb-preview-actions"><button data-action="load-source">Load image</button><span class="iamccs-nfb-file">No image</span></div></section>
        <div class="iamccs-nfb-arrow"><div>→</div></div>
        <section class="iamccs-nfb-preview" data-preview="result"><img><div class="iamccs-nfb-empty"><div><div style="font-size:25px">◇</div><b>Next frame preview</b><span>Your generated scene will appear here</span></div></div><div class="iamccs-nfb-preview-top"><span class="iamccs-nfb-chip">Generated scene</span></div><div class="iamccs-nfb-preview-actions"><span class="iamccs-nfb-file">Waiting</span><button data-action="use-result-mini" disabled>Use as next →</button></div></section>
      </div>
      <section class="iamccs-nfb-reference-deck">
        <div class="iamccs-nfb-reference-head"><b>Optional scene references</b><span>Image 1 remains the current-frame continuity anchor</span></div>
        <div class="iamccs-nfb-reference-grid">
          <article class="iamccs-nfb-reference-card" data-reference="reference_image_2">
            <div class="iamccs-nfb-reference-thumb"><img><div class="empty">Drop a character, object or environment</div><span class="iamccs-nfb-reference-badge">IMAGE 2</span></div>
            <div class="iamccs-nfb-reference-info"><div class="iamccs-nfb-reference-top"><b>Reference A</b><label class="iamccs-nfb-reference-toggle"><input type="checkbox" data-ref-enabled checked> Use</label></div><select class="iamccs-nfb-reference-role" data-ref-role><option value="second_character">Second character</option><option value="object">Object / prop</option><option value="environment">Environment / location</option><option value="style">Style / wardrobe</option><option value="lighting">Lighting reference</option></select><div class="iamccs-nfb-reference-bottom"><span class="iamccs-nfb-reference-file">No image</span><button type="button" data-ref-load>Load</button><button type="button" class="remove" data-ref-remove title="Remove reference">×</button></div></div>
          </article>
          <article class="iamccs-nfb-reference-card" data-reference="reference_image_3">
            <div class="iamccs-nfb-reference-thumb"><img><div class="empty">Drop an additional visual reference</div><span class="iamccs-nfb-reference-badge">IMAGE 3</span></div>
            <div class="iamccs-nfb-reference-info"><div class="iamccs-nfb-reference-top"><b>Reference B</b><label class="iamccs-nfb-reference-toggle"><input type="checkbox" data-ref-enabled checked> Use</label></div><select class="iamccs-nfb-reference-role" data-ref-role><option value="environment">Environment / location</option><option value="second_character">Second character</option><option value="object">Object / prop</option><option value="style">Style / wardrobe</option><option value="lighting">Lighting reference</option></select><div class="iamccs-nfb-reference-bottom"><span class="iamccs-nfb-reference-file">No image</span><button type="button" data-ref-load>Load</button><button type="button" class="remove" data-ref-remove title="Remove reference">×</button></div></div>
          </article>
        </div>
      </section>
      <div class="iamccs-nfb-prompts">
        <section class="iamccs-nfb-prompt"><div class="iamccs-nfb-prompt-head"><span>Direction for the next scene</span><div class="iamccs-nfb-prompt-tools"><button type="button" data-action="prompt-template">Director template</button><button type="button" class="ai" data-action="idea-open">✦ Idea AI</button><button type="button" class="ai" data-action="ai-toggle">✦ AI Assistance</button></div></div><textarea data-role="positive" placeholder="Write a rough direction or a complete Next Scene prompt..."></textarea></section>
        <section class="iamccs-nfb-prompt negative"><div class="iamccs-nfb-prompt-head"><span>Avoid in the result</span><span>Negative prompt</span></div><textarea data-role="negative" placeholder="Artifacts, unwanted text, anatomy errors..."></textarea></section>
      </div>
      <section class="iamccs-nfb-ai"><div class="iamccs-nfb-ai-head"><b>✦ Qwen 2511 Prompt Director</b><span>API keys are used once and never saved</span></div><div class="iamccs-nfb-ai-fields"><select data-ai="provider"><option value="ollama">Ollama · local</option><option value="openai_compatible">OpenAI / compatible</option><option value="anthropic">Claude / Anthropic</option></select><input data-ai="base-url" placeholder="Provider base URL"><input data-ai="model" list="iamccs-nfb-ai-models" placeholder="Model"><button class="refresh" data-action="ai-models" title="Refresh Ollama models">↻</button><input data-ai="api-key" type="password" autocomplete="off" placeholder="API key (not stored)" style="grid-column:2/4"><datalist id="iamccs-nfb-ai-models"></datalist></div><div class="iamccs-nfb-ai-foot"><span class="iamccs-nfb-ai-note">Turns the current text into a concise, continuity-safe prompt with the exact Next Scene trigger.</span><button class="iamccs-nfb-ai-run" data-action="ai-run">Optimize prompt</button></div><div class="iamccs-nfb-ai-progress" data-ai="progress"><i></i><span>AI Prompt Director is working…</span><b data-ai="seconds">0 s</b></div></section>
      <div class="iamccs-nfb-actions"><button class="iamccs-nfb-use" data-action="use-result" disabled>Use generated frame as next source</button><button class="iamccs-nfb-primary" data-action="generate">Generate next scene</button></div>
    </main>
    <section class="iamccs-nfb-idea-overlay" aria-hidden="true"><div class="iamccs-nfb-idea-dialog"><div class="iamccs-nfb-idea-head"><div><h3>✦ Idea AI · Story Continuation Lab</h3><p>Generate several coherent next-frame alternatives from your logline and visual references.</p></div><button type="button" class="iamccs-nfb-idea-close" data-action="idea-close">Close</button></div><div class="iamccs-nfb-idea-body"><div class="iamccs-nfb-idea-form"><label>Story logline · general direction</label><textarea data-idea="logline" placeholder="Example: A disillusioned courier discovers that the package she is carrying contains a living memory, while an unknown pursuer closes in."></textarea><div class="iamccs-nfb-idea-controls"><div><label>Ideas</label><select data-idea="count"><option value="3">3 scenes</option><option value="4" selected>4 scenes</option><option value="5">5 scenes</option><option value="6">6 scenes</option></select></div><div><label>Surprise</label><select data-idea="temperature"><option value="0.72">Controlled</option><option value="0.9" selected>Creative</option><option value="1.08">Bold</option><option value="1.18">Wild</option></select></div></div><div class="iamccs-nfb-idea-provider">Uses <b data-idea="provider-label">Ollama</b> and the model configured in AI Assistance. Vision-capable models can inspect Image 1 and the enabled references.</div><button type="button" class="iamccs-nfb-idea-generate" data-action="idea-generate">Invent next scenes</button><div class="iamccs-nfb-idea-progress" data-idea="progress"><i></i><span>Idea AI is exploring the story…</span><b data-idea="seconds">0 s</b></div><div class="iamccs-nfb-idea-warning">For external providers, the current frame and enabled reference images are sent to the selected API for this request. API keys are never stored.</div></div><div class="iamccs-nfb-idea-results" data-idea="results"><div class="iamccs-nfb-idea-empty">Write the general story logline, then let Idea AI propose several immediate visual advances.<br><br>Each result can be copied or used directly as the next-frame prompt.</div></div></div></div></section>
    <section class="iamccs-nfb-board"><div class="iamccs-nfb-board-head"><div class="iamccs-nfb-board-title">Storyboard <span class="iamccs-nfb-count">0 frames</span></div><div class="iamccs-nfb-board-tools"><button data-action="save-project">Save project</button><button data-action="import-project">Import project</button><button data-action="export">Export board</button><button data-action="clear">Clear</button></div></div><div class="iamccs-nfb-injectbar"><span class="iamccs-nfb-injectbar-label">Inject from START into</span><button class="target" data-target="minimax">MiniMax H3</button><button class="target" data-target="v3">Shotboard V3</button><button class="target" data-target="bridge">H3 Bridge</button><button class="inject" data-action="inject">Inject sequence →</button></div><div class="iamccs-nfb-grid"></div></section>
    <aside class="iamccs-nfb-drawer"><div class="iamccs-nfb-drawer-head"><div><b>Workflow settings</b><div class="iamccs-nfb-sub">Reference backend defaults</div></div><button class="iamccs-nfb-icon" data-action="close-settings">×</button></div><div class="iamccs-nfb-drawer-body"></div></aside>
  `;

  const sourcePanel = root.querySelector('[data-preview="source"]');
  const resultPanel = root.querySelector('[data-preview="result"]');
  const sourceImg = sourcePanel.querySelector("img"); const resultImg = resultPanel.querySelector("img");
  const sourceFile = sourcePanel.querySelector(".iamccs-nfb-file"); const resultFile = resultPanel.querySelector(".iamccs-nfb-file");
  const promptArea = root.querySelector('textarea[data-role="positive"]');
  const negativeArea = root.querySelector('textarea[data-role="negative"]');
  const grid = root.querySelector(".iamccs-nfb-grid");
  const count = root.querySelector(".iamccs-nfb-count"); const status = root.querySelector(".iamccs-nfb-status");
  const generate = root.querySelector('[data-action="generate"]');
  const openEditor = root.querySelector('[data-action="open-editor"]');
  const useButtons = [...root.querySelectorAll('[data-action^="use-result"]')];
  const aiProvider = root.querySelector('[data-ai="provider"]');
  const aiBaseUrl = root.querySelector('[data-ai="base-url"]');
  const aiModel = root.querySelector('[data-ai="model"]');
  const aiApiKey = root.querySelector('[data-ai="api-key"]');
  const aiModelList = root.querySelector("#iamccs-nfb-ai-models");
  const aiRun = root.querySelector('[data-action="ai-run"]');
  const aiProgress = root.querySelector('[data-ai="progress"]');
  const aiSeconds = root.querySelector('[data-ai="seconds"]');
  const ideaOverlay = root.querySelector(".iamccs-nfb-idea-overlay");
  const ideaLogline = root.querySelector('[data-idea="logline"]');
  const ideaCount = root.querySelector('[data-idea="count"]');
  const ideaTemperature = root.querySelector('[data-idea="temperature"]');
  const ideaProviderLabel = root.querySelector('[data-idea="provider-label"]');
  const ideaGenerate = root.querySelector('[data-action="idea-generate"]');
  const ideaMessage = document.createElement("div");
  ideaMessage.className = "iamccs-nfb-idea-message";
  ideaMessage.setAttribute("role", "status"); ideaMessage.setAttribute("aria-live", "polite");
  ideaGenerate.insertAdjacentElement("afterend", ideaMessage);
  const ideaProgress = root.querySelector('[data-idea="progress"]');
  const ideaSeconds = root.querySelector('[data-idea="seconds"]');
  const ideaResults = root.querySelector('[data-idea="results"]');
  const sourcePicker = document.createElement("input"); sourcePicker.type = "file"; sourcePicker.accept = "image/*"; sourcePicker.hidden = true;
  const refPicker = document.createElement("input"); refPicker.type = "file"; refPicker.accept = "image/*"; refPicker.hidden = true;
  const projectPicker = document.createElement("input"); projectPicker.type = "file"; projectPicker.accept = ".json,application/json"; projectPicker.hidden = true;
  root.append(sourcePicker, refPicker, projectPicker);

  let board = parseBoard(read(node, "storyboard_json", "{}"));
  board.inject_targets = injectTargetsOf(board);
  function normalizeAnchor() {
    const frames = framesOf(board);
    let index = frames.findIndex((frame) => String(frame?.id || "") === String(board.inject_anchor_id || ""));
    if (index < 0 && frames.length) index = Math.max(0, frames.findIndex((frame) => frame?.selected !== false));
    if (frames.length) {
      board.inject_anchor_id = String(frames[index]?.id || "");
      board.inject_anchor_index = index;
      frames.forEach((frame, frameIndex) => { frame.selected = frameIndex >= index; });
    } else {
      board.inject_anchor_id = "";
      board.inject_anchor_index = 0;
    }
  }
  normalizeAnchor();
  const aiSaved = node.properties?.iamccs_nextframe_ai || {};
  aiProvider.value = String(aiSaved.provider || "ollama");
  aiBaseUrl.value = String(aiSaved.base_url || "http://127.0.0.1:11434");
  aiModel.value = String(aiSaved.model || "");
  const ideaSaved = node.properties?.iamccs_nextframe_idea || {};
  ideaLogline.value = String(ideaSaved.logline || "");
  ideaCount.value = ["3", "4", "5", "6"].includes(String(ideaSaved.count)) ? String(ideaSaved.count) : "4";
  ideaTemperature.value = ["0.72", "0.9", "1.08", "1.18"].includes(String(ideaSaved.temperature)) ? String(ideaSaved.temperature) : "0.9";
  let ideaItems = Array.isArray(ideaSaved.ideas) ? ideaSaved.ideas.slice(0, 6) : [];
  let generatedFilename = String(read(node, "generated_filename", ""));
  let pendingRef = null;
  let aiTimer = null;
  let ideaTimer = null;
  let refreshSettingsControls = () => {};
  let requestContentResize = () => node.setDirtyCanvas?.(true, true);

  function setStatus(text, kind = "ready") {
    status.className = `iamccs-nfb-status ${kind === "ready" ? "" : kind}`;
    status.querySelector("span").textContent = text;
    root.classList.toggle("busy", kind === "busy"); generate.disabled = kind === "busy";
  }

  function referenceSettings() {
    node.properties = node.properties || {};
    const saved = node.properties.iamccs_nextframe_references || {};
    const normalized = {};
    for (const [name, defaults] of Object.entries(REFERENCE_DEFAULTS)) {
      const item = saved?.[name] || {};
      normalized[name] = {
        enabled: item.enabled !== false,
        role: Object.prototype.hasOwnProperty.call(REFERENCE_ROLE_PROMPTS, item.role) ? item.role : defaults.role,
      };
    }
    node.properties.iamccs_nextframe_references = normalized;
    return normalized;
  }

  let referenceState = referenceSettings();
  const referenceUi = new Map([...root.querySelectorAll("[data-reference]")].map((card) => [card.dataset.reference, {
    card,
    img: card.querySelector("img"),
    file: card.querySelector(".iamccs-nfb-reference-file"),
    enabled: card.querySelector("[data-ref-enabled]"),
    role: card.querySelector("[data-ref-role]"),
  }]));

  function persistReferenceState() {
    node.properties = node.properties || {};
    node.properties.iamccs_nextframe_references = referenceState;
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
  }

  function refreshReferenceDeck() {
    referenceState = referenceSettings();
    for (const [name, ui] of referenceUi) {
      const state = referenceState[name] || REFERENCE_DEFAULTS[name];
      ui.enabled.checked = state.enabled !== false;
      ui.role.value = state.role;
      ui.card.classList.toggle("disabled", state.enabled === false);
      setImage(ui.card, ui.img, ui.file, read(node, name, ""));
    }
  }

  function activeReferences() {
    return [...referenceUi.keys()].flatMap((name, index) => {
      const state = referenceState[name] || REFERENCE_DEFAULTS[name];
      const path = String(read(node, name, "") || "").trim();
      return state.enabled !== false && path ? [{ name, imageNumber: index + 2, role: state.role, path }] : [];
    });
  }

  function promptWithReferenceRoles(prompt) {
    const active = activeReferences();
    if (!active.length) return String(prompt || "");
    const mapping = active.map((item) => `Image ${item.imageNumber} is ${REFERENCE_ROLE_PROMPTS[item.role]}.`).join(" ");
    return `${String(prompt || "").trim()} Reference map: Image 1 is the current scene and primary continuity anchor. ${mapping} Integrate only the requested elements from the additional images; do not replace Image 1 or copy unwanted reference backgrounds.`;
  }

  async function receiveReference(name, file) {
    if (!file?.type?.startsWith("image/")) return;
    try {
      setStatus("Uploading visual reference…", "busy");
      const path = await uploadFile(file);
      write(node, name, path);
      refreshReferenceDeck(); refreshSettingsControls();
      setStatus(`${name === "reference_image_2" ? "Image 2" : "Image 3"} reference ready`);
    } catch (error) {
      console.error(error); setStatus(error.message || "Reference upload failed", "error");
    }
  }

  function repairBackendWidgets() {
    let corrupted = false;
    for (const name of NUMERIC_WIDGETS) {
      const value = read(node, name, NaN);
      if (value === "" || !Number.isFinite(Number(value))) corrupted = true;
    }
    for (const name of ["seed_mode", "sampler_name", "scheduler", "reference_method"]) {
      const item = widget(node, name); const allowed = comboValues(item).map(String);
      if (allowed.length && !allowed.includes(String(item?.value ?? ""))) corrupted = true;
    }
    if (corrupted) {
      for (const [name, value] of Object.entries(SAFE_GENERATION_DEFAULTS)) write(node, name, value);
      console.warn("[IAMCCS NextFrameBuilder] repaired shifted legacy widget values");
      return true;
    }
    for (const name of NUMERIC_WIDGETS) {
      const item = widget(node, name);
      if (item && typeof item.value !== "number") item.value = Number(item.value);
    }
    return false;
  }

  const repairedLegacyWidgets = repairBackendWidgets();

  function setResult(path) {
    generatedFilename = String(path || "");
    setImage(resultPanel, resultImg, resultFile, generatedFilename);
    useButtons.forEach((button) => { button.disabled = !generatedFilename; });
  }

  function useAsSource(path = generatedFilename) {
    if (!path) return;
    write(node, "source_image", path); setImage(sourcePanel, sourceImg, sourceFile, path);
    setStatus(`Frame ${fileParts(path).filename} is now the source`);
  }

  function persistBoard() {
    normalizeAnchor();
    board.schema = "iamccs.next_frame_builder.storyboard.v1";
    write(node, "storyboard_json", JSON.stringify(board)); renderBoard();
  }

  function persistAiSettings() {
    node.properties = node.properties || {};
    node.properties.iamccs_nextframe_ai = {
      provider: aiProvider.value,
      base_url: aiBaseUrl.value.trim(),
      model: aiModel.value.trim(),
    };
    app.graph?.setDirtyCanvas?.(true, true);
  }

  function startAiTimer() {
    if (aiTimer) clearInterval(aiTimer);
    const startedAt = Date.now();
    const update = () => { aiSeconds.textContent = `${Math.floor((Date.now() - startedAt) / 1000)} s`; };
    update(); aiProgress.classList.add("active");
    aiTimer = setInterval(update, 1000);
  }

  function stopAiTimer() {
    if (aiTimer) clearInterval(aiTimer);
    aiTimer = null; aiProgress.classList.remove("active");
  }

  function persistIdeaSettings() {
    node.properties = node.properties || {};
    node.properties.iamccs_nextframe_idea = { logline: ideaLogline.value.trim(), count: Number(ideaCount.value) || 4, temperature: Number(ideaTemperature.value) || 0.9, ideas: ideaItems.slice(0, 6) };
    app.graph?.setDirtyCanvas?.(true, true);
  }
  function startIdeaTimer() {
    if (ideaTimer) clearInterval(ideaTimer);
    const startedAt = Date.now(); const update = () => { ideaSeconds.textContent = `${Math.floor((Date.now() - startedAt) / 1000)} s`; };
    update(); ideaProgress.classList.add("active"); ideaTimer = setInterval(update, 1000);
  }
  function stopIdeaTimer() { if (ideaTimer) clearInterval(ideaTimer); ideaTimer = null; ideaProgress.classList.remove("active"); }
  function setIdeaMessage(text = "", kind = "") { ideaMessage.textContent = String(text || ""); ideaMessage.className = `iamccs-nfb-idea-message${text ? " active" : ""}${kind ? ` ${kind}` : ""}`; }
  function refreshIdeaProviderLabel() {
    const providerName = { ollama: "Ollama", openai_compatible: "OpenAI / compatible", anthropic: "Claude / Anthropic" }[aiProvider.value] || aiProvider.value;
    ideaProviderLabel.textContent = `${providerName} · ${aiModel.value.trim() || "model not selected"}`;
  }
  async function copyIdeaPrompt(text) {
    try { await navigator.clipboard.writeText(text); }
    catch { const helper = document.createElement("textarea"); helper.value = text; helper.style.position = "fixed"; helper.style.opacity = "0"; document.body.appendChild(helper); helper.select(); document.execCommand("copy"); helper.remove(); }
    setStatus("Scene prompt copied");
  }
  function renderIdeaResults() {
    ideaResults.replaceChildren();
    if (!ideaItems.length) { const empty = document.createElement("div"); empty.className = "iamccs-nfb-idea-empty"; empty.innerHTML = "Write the general story logline, then let Idea AI propose several immediate visual advances.<br><br>Each result can be copied or used directly as the next-frame prompt."; ideaResults.appendChild(empty); return; }
    ideaItems.forEach((idea, index) => {
      const card = document.createElement("article"); card.className = "iamccs-nfb-idea-card";
      card.innerHTML = `<div class="iamccs-nfb-idea-card-head"><span class="iamccs-nfb-idea-card-num">${index + 1}</span><b title="${escapeHtml(idea.title)}">${escapeHtml(idea.title || `Scene idea ${index + 1}`)}</b></div><div class="iamccs-nfb-idea-card-body"><div class="iamccs-nfb-idea-beat">${escapeHtml(idea.beat || "Immediate story advance")}</div><div class="iamccs-nfb-idea-prompt">${escapeHtml(idea.prompt || "")}</div><div class="iamccs-nfb-idea-card-actions"><button type="button" data-copy>Copy</button><button type="button" class="use" data-use>Use as next prompt</button></div></div>`;
      card.querySelector("[data-copy]").addEventListener("click", () => copyIdeaPrompt(String(idea.prompt || "")));
      card.querySelector("[data-use]").addEventListener("click", () => { promptArea.value = String(idea.prompt || ""); write(node, "prompt_text", promptArea.value); root.classList.remove("idea-open"); ideaOverlay.setAttribute("aria-hidden", "true"); setStatus(`Idea ${index + 1} loaded as the next-frame prompt`); promptArea.focus(); });
      ideaResults.appendChild(card);
    }); persistIdeaSettings();
  }
  function ideaReferenceContext() {
    const lines = ["Image 1 is the current frame and primary continuity authority."];
    for (const item of activeReferences()) lines.push(`Image ${item.imageNumber} is ${REFERENCE_ROLE_PROMPTS[item.role]}.`);
    return lines.join(" ");
  }
  function dataUrlFromBlob(blob) { return new Promise((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(String(reader.result || "")); reader.onerror = () => reject(reader.error || new Error("Could not read a visual reference")); reader.readAsDataURL(blob); }); }
  async function fetchIdeaImage(path, name) {
    let lastError = null;
    for (const type of ["input", "output", "temp"]) {
      try { const response = await fetch(imageUrl(path, type)); if (!response.ok) { lastError = new Error(`${name} was not found in ${type}`); continue; } const blob = await response.blob(); if (blob.size > 16 * 1024 * 1024) throw new Error(`${name} is larger than 16 MB for AI vision`); if (!blob.type.startsWith("image/")) { lastError = new Error(`${name} is not a readable image`); continue; } return { data: await dataUrlFromBlob(blob), mime_type: blob.type || "image/png" }; }
      catch (error) { lastError = error; }
    }
    throw lastError || new Error(`Could not read ${name}`);
  }
  async function collectIdeaImages() {
    const roleMap = { second_character: "identity", object: "reference", environment: "composition", style: "style", lighting: "style" }; const specs = [];
    const sourcePath = String(read(node, "source_image", "") || "").trim();
    if (sourcePath) specs.push({ path: sourcePath, name: "Image 1 · current frame", role: "opening", slot: "1" });
    for (const item of activeReferences()) specs.push({ path: item.path, name: `Image ${item.imageNumber}`, role: roleMap[item.role] || "reference", slot: String(item.imageNumber) });
    const settled = await Promise.allSettled(specs.slice(0, 3).map(async (item) => ({ ...(await fetchIdeaImage(item.path, item.name)), name: item.name, role: item.role, slot: item.slot })));
    return { images: settled.filter((item) => item.status === "fulfilled").map((item) => item.value), skipped: settled.filter((item) => item.status === "rejected").map((item) => item.reason?.message || "Unreadable reference") };
  }
  async function generateSceneIdeas() {
    const logline = ideaLogline.value.trim();
    if (!logline) { ideaLogline.focus(); setIdeaMessage("Write the story logline before generating ideas.", "error"); setStatus("Write the story logline first", "error"); return; }
    if (!aiModel.value.trim()) { setIdeaMessage("No AI model is selected. Close this panel, open AI Assistance, select or refresh an Ollama model, then try again.", "error"); setStatus("Select an AI model in AI Assistance", "error"); return; }
    try {
      ideaGenerate.disabled = true; startIdeaTimer(); persistIdeaSettings(); persistAiSettings(); setIdeaMessage("Reading the logline and visual references…", "busy"); setStatus("Idea AI is reading the story and visual references…", "busy");
      const collected = await collectIdeaImages(); if (collected.skipped.length) setIdeaMessage(`Generating without ${collected.skipped.length} unreadable reference image${collected.skipped.length === 1 ? "" : "s"}.`, "busy");
      const controller = new AbortController(); const requestTimeout = setTimeout(() => controller.abort(), 155000);
      const response = await fetch(api.apiURL("/iamccs/nextframe/ideas"), { method: "POST", headers: { "Content-Type": "application/json" }, signal: controller.signal, body: JSON.stringify({ provider: aiProvider.value, base_url: aiBaseUrl.value.trim(), model: aiModel.value.trim(), api_key: aiApiKey.value, logline, current_prompt: promptArea.value.trim(), reference_context: ideaReferenceContext(), count: Number(ideaCount.value) || 4, temperature: Number(ideaTemperature.value) || 0.9, timeout: 150, nonce: `${Date.now()}-${Math.random().toString(36).slice(2)}`, images: collected.images }) });
      clearTimeout(requestTimeout); const responseText = await response.text(); let payload = null; try { payload = JSON.parse(responseText); } catch { throw new Error(responseText.slice(0, 240) || `Idea AI HTTP ${response.status}`); } if (!response.ok || !payload.ok) throw new Error(payload.error || "Idea AI failed");
      ideaItems = Array.isArray(payload.ideas) ? payload.ideas.slice(0, 6) : []; if (!ideaItems.length) throw new Error("Idea AI returned no scene ideas");
      aiApiKey.value = ""; renderIdeaResults(); setIdeaMessage(`${ideaItems.length} next-scene ideas are ready.`, "success"); setStatus(`${ideaItems.length} next-scene ideas created with ${payload.report?.model || aiModel.value}`);
    } catch (error) { const message = error?.name === "AbortError" ? "Idea AI timed out after 155 seconds. Check the selected model and provider." : (error.message || "Idea AI failed"); console.error(error); setIdeaMessage(message, "error"); setStatus(message, "error"); }
    finally { stopIdeaTimer(); ideaGenerate.disabled = false; }
  }

  function applyProviderDefaults(force = false) {
    const defaults = {
      ollama: ["http://127.0.0.1:11434", ""],
      openai_compatible: ["https://api.openai.com/v1", "gpt-4.1-mini"],
      anthropic: ["https://api.anthropic.com/v1", "claude-sonnet-4-5"],
    };
    const [baseUrl, model] = defaults[aiProvider.value] || defaults.ollama;
    if (force || !aiBaseUrl.value.trim()) aiBaseUrl.value = baseUrl;
    if (force || !aiModel.value.trim()) aiModel.value = model;
    aiApiKey.style.display = aiProvider.value === "ollama" ? "none" : "block";
    persistAiSettings(); refreshIdeaProviderLabel();
  }

  async function refreshAiModels() {
    if (aiProvider.value !== "ollama") { setStatus("Model discovery is available for Ollama", "error"); return; }
    try {
      setStatus("Reading local Ollama models…", "busy");
      const url = api.apiURL(`/iamccs/prompter/ollama/models?base_url=${encodeURIComponent(aiBaseUrl.value.trim())}`);
      const response = await fetch(url);
      const payload = await response.json();
      if (!response.ok || !payload.ok) throw new Error(payload.error || "Ollama model discovery failed");
      const models = Array.isArray(payload.models) ? payload.models : [];
      aiModelList.replaceChildren(...models.map((item) => {
        const option = document.createElement("option"); option.value = String(item.name || ""); return option;
      }));
      if (!aiModel.value.trim() && models[0]?.name) aiModel.value = String(models[0].name);
      persistAiSettings(); refreshIdeaProviderLabel(); setStatus(`${models.length} Ollama model${models.length === 1 ? "" : "s"} available`);
    } catch (error) { setStatus(error.message || "Ollama is unavailable", "error"); }
  }

  async function runAiAssistance() {
    const rough = promptArea.value.trim();
    if (!rough) { promptArea.focus(); setStatus("Write a rough scene direction first", "error"); return; }
    if (!aiModel.value.trim()) { aiModel.focus(); setStatus("Select an AI assistant model", "error"); return; }
    try {
      aiRun.disabled = true; startAiTimer(); setStatus("AI is directing the Qwen prompt…", "busy"); persistAiSettings();
      const response = await fetch(api.apiURL("/iamccs/nextframe/assist"), {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider: aiProvider.value, base_url: aiBaseUrl.value.trim(), model: aiModel.value.trim(),
          api_key: aiApiKey.value, user_prompt: rough, current_prompt: rough, temperature: 0.25, timeout: 120,
        }),
      });
      const payload = await response.json();
      if (!response.ok || !payload.ok) throw new Error(payload.error || "AI Assistance failed");
      promptArea.value = String(payload.prompt || rough); write(node, "prompt_text", promptArea.value);
      aiApiKey.value = ""; setStatus(`Prompt optimized with ${payload.report?.model || aiModel.value}`);
    } catch (error) { setStatus(error.message || "AI Assistance failed", "error"); }
    finally { stopAiTimer(); aiRun.disabled = false; }
  }

  function saveProject() {
    write(node, "prompt_text", promptArea.value);
    write(node, "negative_prompt", negativeArea.value);
    normalizeAnchor(); persistAiSettings(); persistIdeaSettings();
    const widgets = Object.fromEntries(PROJECT_WIDGET_FIELDS.map((name) => [name, read(node, name, "")]));
    const project = {
      schema: PROJECT_SCHEMA,
      schema_version: 1,
      ui_version: UI_VERSION,
      saved_at: new Date().toISOString(),
      name: String(read(node, "session_id", "storyboard") || "storyboard"),
      widgets,
      storyboard: board,
      generated_filename: generatedFilename,
      ai_settings: { ...(node.properties?.iamccs_nextframe_ai || {}) },
      idea_settings: { ...(node.properties?.iamccs_nextframe_idea || {}) },
      reference_settings: { ...referenceState },
      security: { api_keys_saved: false },
    };
    const safeName = project.name.replace(/[^A-Za-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "") || "storyboard";
    downloadJson(`IAMCCS_NextFrameBuilder_Project_${safeName}.json`, project);
    setStatus("Project saved");
  }

  async function importProject(file) {
    if (!file) return;
    if (Number(file.size || 0) > 25 * 1024 * 1024) throw new Error("Project file is larger than 25 MB");
    const payload = JSON.parse(await file.text());
    if (!payload || typeof payload !== "object" || Array.isArray(payload)) throw new Error("Invalid project JSON");
    const isProject = payload.schema === PROJECT_SCHEMA;
    const isLegacyBoard = String(payload.schema || "").startsWith("iamccs.next_frame_builder.storyboard");
    if (!isProject && !isLegacyBoard) throw new Error("This is not an IAMCCS NextFrameBuilder project");

    const values = isProject && payload.widgets && typeof payload.widgets === "object" ? payload.widgets : {};
    for (const name of PROJECT_WIDGET_FIELDS) {
      if (Object.prototype.hasOwnProperty.call(values, name)) write(node, name, values[name]);
    }
    const repairedImport = repairBackendWidgets();
    const rawBoard = isLegacyBoard ? payload : payload.storyboard;
    board = parseBoard(typeof rawBoard === "string" ? rawBoard : JSON.stringify(rawBoard || {}));
    board.inject_targets = injectTargetsOf(board); normalizeAnchor();

    if (isProject && payload.ai_settings && typeof payload.ai_settings === "object") {
      const importedAi = payload.ai_settings;
      node.properties = node.properties || {};
      node.properties.iamccs_nextframe_ai = {
        provider: ["ollama", "openai_compatible", "anthropic"].includes(importedAi.provider) ? importedAi.provider : "ollama",
        base_url: String(importedAi.base_url || "http://127.0.0.1:11434"),
        model: String(importedAi.model || ""),
      };
      aiProvider.value = node.properties.iamccs_nextframe_ai.provider;
      aiBaseUrl.value = node.properties.iamccs_nextframe_ai.base_url;
      aiModel.value = node.properties.iamccs_nextframe_ai.model;
      aiApiKey.value = ""; applyProviderDefaults(false);
    }
    if (isProject && payload.idea_settings && typeof payload.idea_settings === "object") {
      const importedIdea = payload.idea_settings;
      ideaLogline.value = String(importedIdea.logline || "");
      ideaCount.value = ["3", "4", "5", "6"].includes(String(importedIdea.count)) ? String(importedIdea.count) : "4";
      ideaTemperature.value = ["0.72", "0.9", "1.08", "1.18"].includes(String(importedIdea.temperature)) ? String(importedIdea.temperature) : "0.9";
      ideaItems = Array.isArray(importedIdea.ideas) ? importedIdea.ideas.slice(0, 6).map((item) => ({ title: String(item?.title || "Scene idea"), beat: String(item?.beat || ""), prompt: String(item?.prompt || "") })).filter((item) => item.prompt) : [];
      persistIdeaSettings(); renderIdeaResults();
    }
    if (isProject && payload.reference_settings && typeof payload.reference_settings === "object") {
      node.properties = node.properties || {};
      node.properties.iamccs_nextframe_references = payload.reference_settings;
      referenceState = referenceSettings();
      persistReferenceState();
    }

    promptArea.value = String(read(node, "prompt_text", ""));
    negativeArea.value = String(read(node, "negative_prompt", ""));
    setImage(sourcePanel, sourceImg, sourceFile, read(node, "source_image", ""));
    refreshReferenceDeck();
    setResult(isProject ? String(payload.generated_filename || "") : "");
    persistBoard(); refreshTargetButtons(); refreshSettingsControls();
    setStatus(`${repairedImport ? "Project imported with safe generation defaults" : "Project imported"} · ${framesOf(board).length} frame${framesOf(board).length === 1 ? "" : "s"}`);
  }

  async function toggleFullEditor() {
    try {
      if (document.fullscreenElement === root) await document.exitFullscreen();
      else await root.requestFullscreen({ navigationUI: "hide" });
    } catch (error) {
      setStatus(error?.message || "Fullscreen editor is unavailable", "error");
    }
  }

  function refreshTargetButtons() {
    for (const button of root.querySelectorAll("[data-target]")) {
      button.classList.toggle("on", Boolean(board.inject_targets?.[button.dataset.target]));
    }
  }

  function renderBoard() {
    const frames = framesOf(board); count.textContent = `${frames.length} frame${frames.length === 1 ? "" : "s"}`;
    grid.replaceChildren();
    if (!frames.length) {
      const empty = document.createElement("div"); empty.className = "iamccs-nfb-board-empty";
      empty.innerHTML = "Generated scenes will populate this production board.<br>Each frame remains reusable as a new starting point."; grid.appendChild(empty); requestContentResize(); return;
    }
    const { anchorIndex } = injectionSelection(board);
    frames.forEach((frame, index) => {
      const card = document.createElement("article");
      const chosen = index >= anchorIndex; const isAnchor = index === anchorIndex;
      card.className = `iamccs-nfb-card ${chosen ? "chosen" : ""} ${isAnchor ? "anchor" : ""} ${String(read(node, "source_image")) === String(frame.filename) ? "active" : ""}`;
      card.innerHTML = `<div class="iamccs-nfb-thumb"><img src="${imageUrl(frame.filename, "input")}"><span class="iamccs-nfb-num">${frame.number || index + 1}</span><button class="iamccs-nfb-select" data-select title="Start incremental injection from this frame">${isAnchor ? "START" : chosen ? "→" : "SET"}</button></div><div class="iamccs-nfb-card-body"><div class="iamccs-nfb-card-prompt" title="${escapeHtml(frame.prompt)}">${escapeHtml(frame.prompt || "Generated next scene")}</div><div class="iamccs-nfb-card-buttons"><button data-use>Use as source</button><button data-remove title="Remove from board">×</button></div></div>`;
      card.querySelector("[data-select]").addEventListener("click", () => { board.inject_anchor_id = String(frame.id || ""); board.inject_anchor_index = index; persistBoard(); });
      card.querySelector("[data-use]").addEventListener("click", () => { useAsSource(frame.filename); renderBoard(); });
      card.querySelector("[data-remove]").addEventListener("click", () => { board.frames = frames.filter((item) => item.id !== frame.id); persistBoard(); });
      grid.appendChild(card);
    });
    requestContentResize();
  }

  function connectCineLinx(target) {
    const outputSlot = typeof node.findOutputSlot === "function" ? node.findOutputSlot("cine_linx") : 3;
    const inputSlot = typeof target.findInputSlot === "function" ? target.findInputSlot("cine_linx") : -1;
    if (outputSlot < 0 || inputSlot < 0) return false;
    const existing = target.inputs?.[inputSlot]?.link;
    if (existing != null) target.graph?.removeLink?.(existing);
    node.connect(outputSlot, target, inputSlot);
    return true;
  }

  function injectSelected() {
    const selection = injectionSelection(board);
    const frames = selection.frames.filter((frame) => String(frame?.filename || "").trim());
    if (!frames.length) { setStatus("Choose a START frame in the storyboard", "error"); return; }
    board.inject_anchor_index = selection.anchorIndex;
    persistBoard();
    const timeline = buildInjectionTimeline(
      frames, read(node, "inject_slot_seconds", 5), promptArea.value, negativeArea.value, selection.anchorIndex,
    );
    const selectedPaths = frames.map((frame) => String(frame.filename));
    const enabledTypes = new Map(
      Object.entries(INJECT_TARGETS)
        .filter(([key]) => board.inject_targets?.[key])
        .map(([key, value]) => [value.type, key]),
    );
    let connected = 0;
    const touched = [];
    for (const target of node.graph?._nodes || []) {
      const targetType = nodeType(target);
      const targetKey = enabledTypes.get(targetType);
      if (!targetKey || target === node) continue;
      if (targetKey !== "bridge") {
        const merged = mergeInjection(target, timeline, selectedPaths, selection.anchorIndex);
        write(target, "timeline_data", JSON.stringify(merged.timeline));
        write(target, "image_paths", JSON.stringify(merged.paths));
        write(target, "duration_seconds", merged.timeline.duration_seconds);
        write(target, "frame_rate", timeline.frame_rate);
        if (!String(read(target, "global_prompt", "")).trim()) write(target, "global_prompt", promptArea.value.trim());
      }
      if (connectCineLinx(target)) {
        connected += 1;
        touched.push(INJECT_TARGETS[targetKey].label);
      }
    }
    app.graph?.setDirtyCanvas?.(true, true);
    if (!connected) {
      setStatus("No enabled H3/V3/Bridge target found on canvas", "error");
      return;
    }
    setStatus(`Injected slots ${selection.anchorIndex + 1}-${selection.anchorIndex + frames.length} → ${[...new Set(touched)].join(", ")}`);
  }

  async function receiveSource(file) {
    if (!file?.type?.startsWith("image/")) return;
    try { setStatus("Uploading source…", "busy"); const path = await uploadFile(file); write(node, "source_image", path); setImage(sourcePanel, sourceImg, sourceFile, path); setStatus("Source ready"); }
    catch (error) { console.error(error); setStatus(error.message || "Upload failed", "error"); }
  }

  async function queueBuilder() {
    if (!String(read(node, "source_image", "")).trim()) { sourcePicker.click(); setStatus("Choose a source frame", "error"); return; }
    if (!promptArea.value.trim()) { promptArea.focus(); setStatus("Write the next-scene direction", "error"); return; }
    write(node, "prompt_text", promptArea.value.trim());
    write(node, "negative_prompt", negativeArea.value.trim());
    if (repairBackendWidgets()) {
      refreshSettingsControls();
      setStatus("Corrupted legacy settings repaired; generating with safe defaults", "busy");
    }
    const mode = String(read(node, "seed_mode", "randomize"));
    if (mode === "randomize") write(node, "seed", Math.floor(Math.random() * Number.MAX_SAFE_INTEGER));
    else if (mode === "increment") write(node, "seed", Number(read(node, "seed", 0)) + 1);
    write(node, "run_token", `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`);
    setStatus("Building next scene…", "busy");
    try {
      const comfyApp = node.graph?.comfyApp || window.app || app;
      const full = await comfyApp.graphToPrompt();
      const id = String(node.id); const promptNode = full?.output?.[id] || full?.prompt?.[id];
      if (!promptNode) throw new Error("NextFrameBuilder is not available in the prompt graph");
      promptNode.inputs = promptNode.inputs || {};
      promptNode.inputs.prompt_text = promptWithReferenceRoles(promptArea.value.trim());
      for (const name of referenceUi.keys()) {
        if (referenceState[name]?.enabled === false) promptNode.inputs[name] = "";
      }
      await api.queuePrompt(0, { output: { [id]: promptNode }, workflow: full.workflow || app.graph.serialize() });
    } catch (error) {
      console.error("[IAMCCS NextFrameBuilder] queue failed", error); setStatus(error?.message || "Queue failed", "error");
    }
  }

  promptArea.value = String(read(node, "prompt_text", ""));
  promptArea.addEventListener("input", () => write(node, "prompt_text", promptArea.value));
  negativeArea.value = String(read(node, "negative_prompt", ""));
  negativeArea.addEventListener("input", () => write(node, "negative_prompt", negativeArea.value));
  applyProviderDefaults(false); renderIdeaResults();
  setImage(sourcePanel, sourceImg, sourceFile, read(node, "source_image", "")); setResult(generatedFilename); refreshReferenceDeck(); refreshTargetButtons(); renderBoard();

  root.querySelector('[data-action="prompt-template"]').addEventListener("click", () => {
    promptArea.value = DIRECTOR_TEMPLATE;
    write(node, "prompt_text", DIRECTOR_TEMPLATE);
    promptArea.focus();
    setStatus("Director prompt structure loaded");
  });
  root.querySelector('[data-action="ai-toggle"]').addEventListener("click", () => {
    root.classList.toggle("ai-open"); requestContentResize();
    if (root.classList.contains("ai-open") && aiProvider.value === "ollama" && !aiModel.value.trim()) refreshAiModels();
  });
  root.querySelector('[data-action="idea-open"]').addEventListener("click", () => {
    refreshIdeaProviderLabel(); renderIdeaResults(); root.classList.add("idea-open"); ideaOverlay.setAttribute("aria-hidden", "false");
    if (aiProvider.value === "ollama" && !aiModel.value.trim()) refreshAiModels();
    setTimeout(() => ideaLogline.focus(), 0);
  });
  const closeIdeaOverlay = () => { root.classList.remove("idea-open"); ideaOverlay.setAttribute("aria-hidden", "true"); persistIdeaSettings(); };
  root.querySelector('[data-action="idea-close"]').addEventListener("click", closeIdeaOverlay);
  ideaOverlay.addEventListener("click", (event) => { if (event.target === ideaOverlay) closeIdeaOverlay(); });
  const onIdeaKeydown = (event) => { if (event.key === "Escape" && root.classList.contains("idea-open")) closeIdeaOverlay(); };
  document.addEventListener("keydown", onIdeaKeydown);
  ideaGenerate.addEventListener("click", generateSceneIdeas);
  ideaLogline.addEventListener("input", persistIdeaSettings);
  ideaCount.addEventListener("change", persistIdeaSettings); ideaTemperature.addEventListener("change", persistIdeaSettings);
  root.querySelector('[data-action="ai-models"]').addEventListener("click", refreshAiModels);
  aiRun.addEventListener("click", runAiAssistance);
  aiProvider.addEventListener("change", () => { applyProviderDefaults(true); if (aiProvider.value === "ollama") refreshAiModels(); });
  aiBaseUrl.addEventListener("change", persistAiSettings); aiModel.addEventListener("change", () => { persistAiSettings(); refreshIdeaProviderLabel(); });
  root.querySelector('[data-action="load-source"]').addEventListener("click", () => sourcePicker.click());
  sourcePanel.addEventListener("dblclick", () => sourcePicker.click());
  sourcePicker.addEventListener("change", () => { receiveSource(sourcePicker.files?.[0]); sourcePicker.value = ""; });
  for (const event of ["dragenter", "dragover"]) sourcePanel.addEventListener(event, (e) => { e.preventDefault(); sourcePanel.classList.add("drag"); });
  for (const event of ["dragleave", "drop"]) sourcePanel.addEventListener(event, (e) => { e.preventDefault(); sourcePanel.classList.remove("drag"); });
  sourcePanel.addEventListener("drop", (e) => receiveSource(e.dataTransfer?.files?.[0]));
  root.addEventListener("paste", (e) => { const image = [...(e.clipboardData?.files || [])].find((file) => file.type.startsWith("image/")); if (image) receiveSource(image); });
  for (const [name, ui] of referenceUi) {
    ui.enabled.addEventListener("change", () => {
      referenceState[name].enabled = ui.enabled.checked;
      persistReferenceState(); refreshReferenceDeck();
      setStatus(`${name === "reference_image_2" ? "Image 2" : "Image 3"} ${ui.enabled.checked ? "enabled" : "disabled"}`);
    });
    ui.role.addEventListener("change", () => {
      referenceState[name].role = ui.role.value;
      persistReferenceState();
      setStatus(`${name === "reference_image_2" ? "Image 2" : "Image 3"} role updated`);
    });
    ui.card.querySelector("[data-ref-load]").addEventListener("click", () => { pendingRef = { name, input: null }; refPicker.click(); });
    ui.card.querySelector("[data-ref-remove]").addEventListener("click", () => {
      write(node, name, ""); refreshReferenceDeck(); refreshSettingsControls();
      setStatus(`${name === "reference_image_2" ? "Image 2" : "Image 3"} removed`);
    });
    ui.card.addEventListener("dblclick", (event) => {
      if (event.target.closest("button,select,input,label")) return;
      pendingRef = { name, input: null }; refPicker.click();
    });
    for (const eventName of ["dragenter", "dragover"]) ui.card.addEventListener(eventName, (event) => { event.preventDefault(); ui.card.classList.add("drag"); });
    for (const eventName of ["dragleave", "drop"]) ui.card.addEventListener(eventName, (event) => { event.preventDefault(); ui.card.classList.remove("drag"); });
    ui.card.addEventListener("drop", (event) => receiveReference(name, event.dataTransfer?.files?.[0]));
  }
  generate.addEventListener("click", queueBuilder); useButtons.forEach((button) => button.addEventListener("click", () => { useAsSource(); renderBoard(); }));
  openEditor.addEventListener("click", toggleFullEditor);
  const onFullscreenChange = () => {
    const active = document.fullscreenElement === root;
    openEditor.textContent = active ? "⛶ Exit editor" : "⛶ Open editor";
    openEditor.title = active ? "Exit fullscreen editor (Esc)" : "Open the complete UI on this monitor";
  };
  document.addEventListener("fullscreenchange", onFullscreenChange);
  root.querySelector('[data-action="settings"]').addEventListener("click", () => root.classList.add("settings-open"));
  root.querySelector('[data-action="close-settings"]').addEventListener("click", () => { root.classList.remove("settings-open"); refreshReferenceDeck(); });
  root.addEventListener("change", (event) => {
    if (["reference_image_2", "reference_image_3"].includes(event.target?.dataset?.widget)) queueMicrotask(refreshReferenceDeck);
  });
  root.querySelector('[data-action="save-project"]').addEventListener("click", saveProject);
  root.querySelector('[data-action="import-project"]').addEventListener("click", () => projectPicker.click());
  root.querySelector('[data-action="export"]').addEventListener("click", () => downloadJson(`IAMCCS_NextFrameBuilder_${read(node, "session_id", "storyboard")}.json`, board));
  root.querySelector('[data-action="clear"]').addEventListener("click", () => { board.frames = []; board.inject_anchor_id = ""; board.inject_anchor_index = 0; persistBoard(); setResult(""); setStatus("Storyboard cleared"); });
  root.querySelector('[data-action="inject"]').addEventListener("click", injectSelected);
  for (const button of root.querySelectorAll("[data-target]")) {
    button.addEventListener("click", () => {
      board.inject_targets[button.dataset.target] = !board.inject_targets[button.dataset.target];
      write(node, "storyboard_json", JSON.stringify(board)); refreshTargetButtons();
    });
  }

  refPicker.addEventListener("change", async () => {
    const target = pendingRef; const file = refPicker.files?.[0]; refPicker.value = ""; if (!target || !file) return;
    await receiveReference(target.name, file);
    if (target.input) target.input.value = read(node, target.name, "");
  });
  projectPicker.addEventListener("change", async () => {
    const file = projectPicker.files?.[0]; projectPicker.value = "";
    if (!file) return;
    try { await importProject(file); }
    catch (error) { setStatus(error?.message || "Project import failed", "error"); }
  });
  refreshSettingsControls = makeSettings(node, root.querySelector(".iamccs-nfb-drawer-body"), (name, input) => { pendingRef = { name, input }; refPicker.click(); });
  if (repairedLegacyWidgets) {
    refreshSettingsControls();
    setStatus("Legacy widget values repaired with safe generation defaults");
  }

  const onExecuted = (event) => {
    const detail = event?.detail || event || {};
    if (String(detail.display_node ?? detail.node ?? "") !== String(node.id)) return;
    const output = detail.output || {};
    const boardJson = firstValue(output.storyboard_json, ""); const filename = firstValue(output.generated_filename, "");
    if (boardJson) { board = parseBoard(boardJson); board.inject_targets = injectTargetsOf(board); normalizeAnchor(); write(node, "storyboard_json", JSON.stringify(board)); }
    if (filename) setResult(filename);
    const message = firstValue(output.message, filename ? "Next scene ready" : "Generation complete");
    setStatus(message); renderBoard();
  };
  const onError = (event) => {
    const detail = event?.detail || event || {};
    if (String(detail.display_node_id ?? detail.node_id ?? detail.node ?? "") !== String(node.id)) return;
    setStatus(detail?.exception_message || "Generation failed", "error");
  };
  api.addEventListener("executed", onExecuted); api.addEventListener("execution_error", onError);

  const domWidget = node.addDOMWidget("IAMCCS NextFrameBuilder", "iamccs_next_frame_builder", root, { serialize: false, hideOnZoom: false });
  domWidget.computeSize = (width) => [Math.max(940, Number(width || 940)), FIXED_UI_HEIGHT];
  const ensureBackendWidgetsHidden = () => backendWidgets.forEach(hideWidget);
  node._iamccsNextFrameEnsureHidden = ensureBackendWidgetsHidden;
  const previousDrawForeground = node.onDrawForeground;
  node.onDrawForeground = function (...args) {
    ensureBackendWidgetsHidden();
    return previousDrawForeground?.apply(this, args);
  };
  requestContentResize = () => node.setDirtyCanvas?.(true, true);
  node.setSize([Math.max(980, Number(node.size?.[0] || 0)), FIXED_UI_HEIGHT + 44]);
  for (const delay of [0, 50, 250, 1000]) setTimeout(ensureBackendWidgetsHidden, delay);
  node.resizable = true;

  const previousRemoved = node.onRemoved;
  node.onRemoved = function (...args) {
    api.removeEventListener("executed", onExecuted); api.removeEventListener("execution_error", onError);
    document.removeEventListener("fullscreenchange", onFullscreenChange);
    document.removeEventListener("keydown", onIdeaKeydown);
    stopAiTimer(); stopIdeaTimer();
    delete node._iamccsNextFrameEnsureHidden;
    try { root.remove(); } catch {}
    return previousRemoved?.apply(this, args);
  };
}

app.registerExtension({
  name: "iamccs.next_frame_builder.ui",
  async beforeRegisterNodeDef(nodeClass, nodeData) {
    if (String(nodeData?.name || "") !== TYPE) return;
    const previousCreated = nodeClass.prototype.onNodeCreated;
    nodeClass.prototype.onNodeCreated = function (...args) {
      const result = previousCreated?.apply(this, args);
      queueMicrotask(() => mount(this));
      return result;
    };
    const previousConfigured = nodeClass.prototype.onConfigure;
    nodeClass.prototype.onConfigure = function (...args) {
      const result = previousConfigured?.apply(this, args);
      queueMicrotask(() => mount(this));
      return result;
    };
  },
  nodeCreated(node) {
    if (nodeType(node) === TYPE) queueMicrotask(() => mount(node));
  },
});
