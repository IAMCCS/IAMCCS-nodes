import { app } from "../../scripts/app.js";

// By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
// By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
// By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
// By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
// By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com

const TYPE = "IAMCCS_StoryboardFrameDesigner";
const STYLE_ID = "iamccs-ideogram-storyboard-frame-style";
const IDEOBOARD_SCHEMA = "iamccs.ideoboard.package";

const PRESETS = {
  storyboard: {
    label: "Storyboard",
    summary: "Wide cinematic shot design with readable blocking and annotation-friendly balance.",
    canvas: { width: 1536, height: 864, aspect_label: "16:9 Storyboard" },
    scene: {
      high_level_description: "A storyboard-ready cinematic frame with strong blocking, readable staging, and production-minded composition.",
      aesthetics: "film storyboard realism, readable silhouettes, practical art direction, shot-design clarity",
      lighting: "controlled cinematic lighting with clear value separation and readable focal hierarchy",
      photo: "",
      medium: "storyboard concept art",
      color_palette: ["#1C2430", "#B86A3B", "#E9D7B9", "#6B7A8F"],
      background: "Production-aware environment blocking that supports shot continuity and leaves room for annotations or title elements.",
    },
  },
  poster: {
    label: "Poster",
    summary: "Vertical key art for title, billing, and dominant hero composition.",
    canvas: { width: 1024, height: 1536, aspect_label: "2:3 Poster" },
    scene: {
      high_level_description: "A striking cinematic poster image with a dominant focal subject, bold typography zones, and premium visual hierarchy.",
      aesthetics: "premium theatrical poster design, dramatic scale, iconic silhouette, polished key art",
      lighting: "high-contrast dramatic key art lighting with controlled glow, depth, and premium finish",
      photo: "",
      medium: "cinematic poster illustration",
      color_palette: ["#111111", "#C84E2F", "#F5E6C8", "#4E6FAE"],
      background: "Graphic poster backdrop with atmospheric depth and clear negative space for billing, taglines, and title treatment.",
    },
  },
  signage: {
    label: "Signage",
    summary: "Readable environmental graphics and in-world text surfaces.",
    canvas: { width: 1536, height: 864, aspect_label: "16:9 Signage" },
    scene: {
      high_level_description: "A cinematic environment built around readable signage, branded surfaces, and strong in-world typography.",
      aesthetics: "designed environmental graphics, high readability, urban production design, premium prop styling",
      lighting: "motivated practical lighting that supports legibility on signs, surfaces, and surrounding space",
      photo: "",
      medium: "environment concept art",
      color_palette: ["#0E1A24", "#19A7CE", "#F6F1D1", "#D65A31"],
      background: "Architectural or environmental context that supports the sign as the hero graphic while keeping text readable.",
    },
  },
  screen_ui: {
    label: "Screen UI",
    summary: "Diegetic screens, monitors, dashboards, and interface-led compositions.",
    canvas: { width: 1536, height: 864, aspect_label: "16:9 Screen UI" },
    scene: {
      high_level_description: "A diegetic screen composition with readable interface panels, cinematic reflections, and believable display design.",
      aesthetics: "futuristic screen graphics, clean UI hierarchy, believable diegetic display design, premium sci-fi interface art",
      lighting: "monitor-emissive lighting with controlled reflections and clean contrast for readable interface elements",
      photo: "",
      medium: "screen interface design",
      color_palette: ["#08121C", "#37D5D6", "#D9F3FF", "#F2A65A"],
      background: "Physical screen housing or surrounding set elements that support the interface without obscuring key text zones.",
    },
  },
  title_card: {
    label: "Title Card",
    summary: "Minimal, typography-forward composition for opening cards and chapter slates.",
    canvas: { width: 1536, height: 864, aspect_label: "16:9 Title Card" },
    scene: {
      high_level_description: "A cinematic title card frame designed around elegant typography, strong negative space, and mood-driven art direction.",
      aesthetics: "title card design, clean hierarchy, premium typography composition, deliberate negative space",
      lighting: "controlled atmosphere with subtle gradients and composition that preserves crisp title readability",
      photo: "",
      medium: "title card artwork",
      color_palette: ["#141414", "#8C1C13", "#F6E7CB", "#6C8EA3"],
      background: "Minimal or atmospheric background treatment that elevates the title and avoids visual clutter.",
    },
  },
};

function widget(node, name) {
  return (node.widgets || []).find((entry) => entry?.name === name);
}

function setWidget(node, name, value) {
  const entry = widget(node, name);
  if (!entry) return;
  entry.value = value;
  try { entry.callback?.(value); } catch {}
  node.setDirtyCanvas?.(true, true);
  app.graph?.setDirtyCanvas?.(true, true);
}

function hideWidget(entry) {
  if (!entry) return;
  entry.hidden = true;
  entry.type = "hidden";
  entry.computeSize = () => [0, -4];
  entry.draw = () => {};
  entry.options = { ...(entry.options || {}), hidden: true };
}

function cloneValue(value, fallback = null) {
  try {
    return JSON.parse(JSON.stringify(value));
  } catch {
    return fallback;
  }
}

function defaultData() {
  return {
    schema: "iamccs.ideogram_storyboard_frame_designer",
    schema_version: 1,
    preset_key: "storyboard",
    canvas: { width: 1536, height: 864, aspect_label: "16:9 Storyboard" },
    scene: {
      high_level_description: "A storyboard-ready cinematic frame with strong blocking, readable staging, and production-minded composition.",
      aesthetics: "film storyboard realism, readable silhouettes, practical art direction, shot-design clarity",
      lighting: "controlled cinematic lighting with clear value separation and readable focal hierarchy",
      photo: "",
      medium: "storyboard concept art",
      color_palette: ["#1C2430", "#B86A3B", "#E9D7B9", "#6B7A8F"],
      background: "Production-aware environment blocking that supports shot continuity and leaves room for annotations or title elements.",
    },
    items: [
      {
        id: "item_001",
        kind: "obj",
        label: "Primary subject",
        text: "",
        x: 160,
        y: 170,
        w: 480,
        h: 520,
        desc: "Primary visual subject in the mid-frame with strong silhouette readability, production-design detail, and clear cinematic emphasis.",
        color_palette: ["#8B4513", "#FFE4B5", "#1A1A2E"],
      },
      {
        id: "item_002",
        kind: "text",
        label: "Title block",
        text: "TITLE",
        x: 650,
        y: 120,
        w: 220,
        h: 120,
        desc: "Readable in-frame text block with deliberate typography, good contrast, and clean spatial separation from the main subject.",
        color_palette: ["#FF6B35", "#FFE4B5"],
      },
    ],
  };
}

function presetData(key) {
  const preset = PRESETS[key] || PRESETS.storyboard;
  const base = defaultData();
  return {
    ...base,
    preset_key: PRESETS[key] ? key : "storyboard",
    canvas: { ...base.canvas, ...(preset.canvas || {}) },
    scene: { ...base.scene, ...(preset.scene || {}), color_palette: [...(preset.scene?.color_palette || base.scene.color_palette)] },
  };
}

function cleanText(value) {
  return String(value || "").trim();
}

function clampInt(value, min, max, fallback) {
  const number = Number.isFinite(Number(value)) ? Math.round(Number(value)) : fallback;
  return Math.max(min, Math.min(max, number));
}

function normalizeHex(value) {
  let text = String(value || "").trim().toUpperCase();
  if (!text) return "";
  if (!text.startsWith("#")) text = `#${text}`;
  if (text.length === 4) text = `#${text[1]}${text[1]}${text[2]}${text[2]}${text[3]}${text[3]}`;
  if (!/^#[0-9A-F]{6}$/.test(text)) return "";
  return text;
}

function paletteList(value, fallback = []) {
  const source = Array.isArray(value) ? value : String(value || "").replace(/;/g, ",").split(",");
  const out = [];
  source.forEach((entry) => {
    const color = normalizeHex(entry);
    if (color && !out.includes(color)) out.push(color);
  });
  return out.length ? out : [...fallback];
}

function itemFromPrompt(entry, index) {
  const bbox = Array.isArray(entry?.bbox) && entry.bbox.length === 4 ? entry.bbox : [120, 120, 760, 760];
  const x = clampInt(bbox[0], 0, 999, 120);
  const y = clampInt(bbox[1], 0, 999, 120);
  const x2 = clampInt(bbox[2], x + 1, 1000, 760);
  const y2 = clampInt(bbox[3], y + 1, 1000, 760);
  const kind = String(entry?.type || "obj").toLowerCase() === "text" ? "text" : "obj";
  return {
    id: `item_${String(index + 1).padStart(3, "0")}`,
    kind,
    label: cleanText(entry?.label || entry?.text || entry?.type || `Element ${index + 1}`),
    text: kind === "text" ? cleanText(entry?.text) : "",
    x,
    y,
    w: Math.max(20, x2 - x),
    h: Math.max(20, y2 - y),
    desc: cleanText(entry?.desc || entry?.description || `Element ${index + 1}`),
    color_palette: paletteList(entry?.color_palette, ["#FFE4B5", "#1A1A2E"]),
  };
}

function promptToDesign(raw) {
  const base = defaultData();
  const style = raw?.style_description || {};
  const comp = raw?.compositional_deconstruction || {};
  return {
    ...base,
    scene: {
      high_level_description: cleanText(raw?.high_level_description) || base.scene.high_level_description,
      aesthetics: cleanText(style?.aesthetics) || base.scene.aesthetics,
      lighting: cleanText(style?.lighting) || base.scene.lighting,
      photo: cleanText(style?.photo),
      medium: cleanText(style?.medium) || base.scene.medium,
      color_palette: paletteList(style?.color_palette, base.scene.color_palette),
      background: cleanText(comp?.background) || base.scene.background,
    },
    items: Array.isArray(comp?.elements) && comp.elements.length ? comp.elements.map(itemFromPrompt) : cloneValue(base.items, []),
  };
}

function normalizeItem(entry, index) {
  const kind = String(entry?.kind || entry?.type || "obj").toLowerCase() === "text" ? "text" : "obj";
  const x = clampInt(entry?.x, 0, 980, 120 + index * 40);
  const y = clampInt(entry?.y, 0, 980, 120 + index * 30);
  return {
    id: cleanText(entry?.id) || `item_${String(index + 1).padStart(3, "0")}`,
    kind,
    label: cleanText(entry?.label || entry?.name || `Element ${index + 1}`),
    text: kind === "text" ? cleanText(entry?.text) : "",
    x,
    y,
    w: Math.min(1000 - x, clampInt(entry?.w, 20, 1000, 260)),
    h: Math.min(1000 - y, clampInt(entry?.h, 20, 1000, 180)),
    desc: cleanText(entry?.desc || entry?.description || `Element ${index + 1}`),
    color_palette: paletteList(entry?.color_palette, ["#FFE4B5", "#1A1A2E"]),
  };
}

function normalizeDesignObject(raw, fallback = defaultData()) {
  if (!raw || typeof raw !== "object") return cloneValue(fallback, fallback);
  return {
    ...fallback,
    preset_key: PRESETS[raw.preset_key] ? raw.preset_key : fallback.preset_key,
    canvas: { ...fallback.canvas, ...(raw.canvas || {}) },
    scene: {
      ...fallback.scene,
      ...(raw.scene || {}),
      color_palette: paletteList(raw?.scene?.color_palette, fallback.scene.color_palette),
    },
    items: Array.isArray(raw.items) && raw.items.length ? raw.items.map(normalizeItem) : cloneValue(fallback.items, fallback.items),
  };
}

function parseData(node) {
  const fallback = defaultData();
  try {
    const parsed = JSON.parse(String(widget(node, "design_data")?.value || ""));
    if (parsed && typeof parsed === "object") {
      if (parsed.boards && typeof parsed.boards === "object") {
        const activeKey = PRESETS[parsed.active_preset_key] ? parsed.active_preset_key : (PRESETS[parsed.preset_key] ? parsed.preset_key : fallback.preset_key);
        const activeBoard = parsed.boards[activeKey];
        if (activeBoard && typeof activeBoard === "object") {
          return normalizeDesignObject({ ...activeBoard, preset_key: activeKey }, fallback);
        }
      }
      if (parsed.high_level_description && parsed.style_description && parsed.compositional_deconstruction) {
        return promptToDesign(parsed);
      }
      return normalizeDesignObject(parsed, fallback);
    }
  } catch {}
  return fallback;
}

function toPrompt(data) {
  const items = Array.isArray(data?.items) ? data.items.map(normalizeItem) : [];
  return {
    high_level_description: cleanText(data?.scene?.high_level_description),
    style_description: {
      aesthetics: cleanText(data?.scene?.aesthetics),
      lighting: cleanText(data?.scene?.lighting),
      photo: cleanText(data?.scene?.photo),
      medium: cleanText(data?.scene?.medium),
      color_palette: paletteList(data?.scene?.color_palette, ["#1A1A2E", "#FFE4B5"]),
    },
    compositional_deconstruction: {
      background: cleanText(data?.scene?.background),
      elements: items.map((item) => ({
        type: item.kind === "text" ? "text" : "obj",
        bbox: [item.x, item.y, Math.min(1000, item.x + item.w), Math.min(1000, item.y + item.h)],
        desc: item.desc,
        color_palette: paletteList(item.color_palette, ["#FFE4B5", "#1A1A2E"]),
        ...(item.kind === "text" ? { text: item.text || item.label } : {}),
      })),
    },
  };
}

function safeFilename(value, fallback = "IAMCCS_Ideoboard") {
  const cleaned = cleanText(value).replace(/[\\/:*?"<>|]+/g, "_").replace(/\s+/g, " ").trim();
  return cleaned || fallback;
}

function escapeXml(value) {
  return String(value || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

function boardPreviewSvgDataUri(data) {
  const palette = paletteList(data?.scene?.color_palette, ["#18202A", "#0D1118", "#F6E7CB"]);
  const items = Array.isArray(data?.items) ? data.items.map(normalizeItem) : [];
  const body = items.map((item) => {
    const fill = item.kind === "text" ? (item.color_palette?.[0] || "#FFC078") : (item.color_palette?.[0] || "#48C2BA");
    const label = escapeXml(item.kind === "text" ? (item.text || item.label) : (item.label || item.id));
    return `\n  <g>\n    <rect x="${item.x}" y="${item.y}" width="${item.w}" height="${item.h}" rx="18" ry="18" fill="${fill}" fill-opacity="0.18" stroke="${fill}" stroke-width="4"/>\n    <text x="${item.x + 16}" y="${item.y + 40}" fill="#F7FBFB" font-size="26" font-family="Segoe UI, Arial, sans-serif">${label}</text>\n  </g>`;
  }).join("");
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000"><defs><linearGradient id="bg" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="${palette[0]}"/><stop offset="100%" stop-color="${palette[1] || palette[0]}"/></linearGradient></defs><rect width="1000" height="1000" fill="url(#bg)"/><text x="24" y="46" fill="#FFF6E7" font-size="28" font-family="Segoe UI, Arial, sans-serif">${escapeXml(data?.scene?.high_level_description || data?.preset_key || "Ideoboard")}</text>${body}\n</svg>`;
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

function saveDialogWindow() {
  try {
    if (window.top && typeof window.top.showSaveFilePicker === 'function') return window.top;
  } catch {}
  return typeof window.showSaveFilePicker === 'function' ? window : null;
}

function buildIdeoboardPackage(data, boardName = "IAMCCS_Ideoboard") {
  const normalized = normalizeDesignObject(data, defaultData());
  const presetKey = PRESETS[normalized.preset_key] ? normalized.preset_key : "storyboard";
  return {
    schema: IDEOBOARD_SCHEMA,
    schema_version: 1,
    board_name: cleanText(boardName) || "IAMCCS_Ideoboard",
    active_preset_key: presetKey,
    boards: {
      [presetKey]: cloneValue({ ...normalized, preset_key: presetKey }, normalized),
    },
    metadata: {
      exported_at: new Date().toISOString(),
      source_node: TYPE,
      width: normalized.canvas?.width || 0,
      height: normalized.canvas?.height || 0,
    },
    assets: {
      preview_svg: boardPreviewSvgDataUri(normalized),
      prompt_json: toPrompt(normalized),
    },
  };
}

function designFromImportedBoard(raw) {
  if (raw && typeof raw === "object" && raw.high_level_description && raw.style_description && raw.compositional_deconstruction) {
    return promptToDesign(raw);
  }
  if (raw && typeof raw === "object" && raw.boards && typeof raw.boards === "object") {
    const activeKey = PRESETS[raw.active_preset_key] ? raw.active_preset_key : (PRESETS[raw.preset_key] ? raw.preset_key : "storyboard");
    const selected = raw.boards[activeKey];
    if (selected && typeof selected === "object") {
      return normalizeDesignObject({ ...selected, preset_key: activeKey }, defaultData());
    }
  }
  return normalizeDesignObject(raw, defaultData());
}

function writeData(node, data) {
  setWidget(node, "design_data", JSON.stringify(data, null, 2));
}

function writeInputSignature(node, signature) {
  setWidget(node, "ideoboard_input_signature", String(signature || ""));
}

function ensureStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
    .iamccs-isf-mount { box-sizing:border-box; width:100%; height:100%; min-width:0; min-height:0; padding:8px; overflow:hidden; }
    .iamccs-isf { box-sizing:border-box; width:100%; min-width:0; height:100%; color:#eef4ef; border:1px solid #1f4648; border-radius:16px; overflow:hidden; background:linear-gradient(180deg,#0d1417,#121f24 48%,#0a1013); box-shadow:0 18px 40px rgba(0,0,0,.35); font:12px/1.35 "Segoe UI",system-ui,sans-serif; display:grid; grid-template-rows:auto minmax(0,1fr) auto; }
    .iamccs-isf * { box-sizing:border-box; }
    .iamccs-isf.iamccs-isf-fullscreen { position:fixed; inset:18px; width:auto; height:auto; z-index:2147483640; border-radius:20px; box-shadow:0 30px 80px rgba(0,0,0,.55); }
    .iamccs-isf-head { display:flex; align-items:center; justify-content:space-between; gap:12px; padding:14px 16px; border-bottom:1px solid rgba(120,200,198,.18); background:radial-gradient(circle at top left,rgba(255,107,53,.15),transparent 28%),linear-gradient(180deg,#122026,#0c1418); }
    .iamccs-isf-title { font-size:16px; font-weight:800; letter-spacing:.02em; color:#fff6e7; }
    .iamccs-isf-sub { color:#9cb8b5; margin-top:2px; max-width:760px; }
    .iamccs-isf-toolbar { display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; }
    .iamccs-isf-btn { border:1px solid #2f676b; background:#122b2f; color:#f0fbfa; border-radius:10px; padding:8px 12px; font-weight:700; cursor:pointer; }
    .iamccs-isf-btn.primary { background:linear-gradient(180deg,#ffb65d,#f28b45); color:#24170b; border-color:#ffcb86; }
    .iamccs-isf-body { display:grid; grid-template-columns:320px minmax(0,1fr) 340px; min-height:0; height:800px; }
    .iamccs-isf.iamccs-isf-fullscreen .iamccs-isf-body { height:calc(100vh - 160px); min-height:820px; }
    .iamccs-isf-pane { padding:14px; border-right:1px solid rgba(120,200,198,.14); background:rgba(8,14,16,.45); overflow-y:auto; min-height:0; }
    .iamccs-isf-pane:last-child { border-right:none; border-left:1px solid rgba(120,200,198,.14); }
    .iamccs-isf-panel { border:1px solid rgba(100,188,184,.16); border-radius:14px; background:linear-gradient(180deg,rgba(18,30,34,.92),rgba(10,17,19,.94)); padding:12px; margin-bottom:12px; }
    .iamccs-isf-panel h4 { margin:0 0 10px; font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:#ffcf8f; }
    .iamccs-isf-preset-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }
    .iamccs-isf-preset { text-align:left; border:1px solid rgba(97,177,173,.18); background:#0b1417; color:#eaf6f4; border-radius:12px; padding:10px; cursor:pointer; }
    .iamccs-isf-preset.selected { border-color:#f4b868; box-shadow:0 0 0 1px rgba(244,184,104,.35); background:linear-gradient(180deg,#18242a,#0b1417); }
    .iamccs-isf-preset strong { display:block; color:#fff6e7; margin-bottom:4px; }
    .iamccs-isf-preset span { display:block; color:#9cb8b5; }
    .iamccs-isf-field { margin-bottom:10px; }
    .iamccs-isf-field label { display:block; margin-bottom:5px; color:#dbe9e7; font-weight:700; }
    .iamccs-isf-field input, .iamccs-isf-field textarea, .iamccs-isf-field select { width:100%; border:1px solid #2b4d50; background:#091215; color:#f7fbfb; border-radius:10px; padding:8px 10px; outline:none; }
    .iamccs-isf-field textarea { min-height:86px; resize:vertical; }
    .iamccs-isf-field.small textarea { min-height:60px; }
    .iamccs-isf-stage-wrap { padding:18px; display:flex; flex-direction:column; gap:12px; background:radial-gradient(circle at top,rgba(255,107,53,.08),transparent 25%),linear-gradient(180deg,#0b1316,#091014); overflow:auto; min-height:0; }
    .iamccs-isf-stage-meta { display:flex; justify-content:space-between; align-items:center; color:#9eb8b7; }
    .iamccs-isf-stage-meta strong { color:#fff; font-size:13px; }
    .iamccs-isf-stage { position:relative; width:100%; aspect-ratio:1/1; border:1px solid rgba(255,255,255,.08); border-radius:18px; overflow:hidden; background:linear-gradient(180deg,#16222d,#0c1118 58%,#11161b); box-shadow:inset 0 0 0 1px rgba(255,255,255,.03),0 22px 44px rgba(0,0,0,.34); }
    .iamccs-isf-grid, .iamccs-isf-overlay { position:absolute; inset:0; pointer-events:none; }
    .iamccs-isf-grid::before, .iamccs-isf-grid::after, .iamccs-isf-overlay::before, .iamccs-isf-overlay::after { content:""; position:absolute; }
    .iamccs-isf-grid::before { inset:0; background-image:linear-gradient(to right,rgba(255,255,255,.05) 1px,transparent 1px),linear-gradient(to bottom,rgba(255,255,255,.05) 1px,transparent 1px); background-size:10% 10%; }
    .iamccs-isf-grid::after { left:33.333%; top:0; width:33.333%; height:100%; border-left:1px dashed rgba(255,215,160,.18); border-right:1px dashed rgba(255,215,160,.18); }
    .iamccs-isf-overlay::before { top:33.333%; left:0; width:100%; border-top:1px dashed rgba(255,215,160,.18); }
    .iamccs-isf-overlay::after { top:66.666%; left:0; width:100%; border-top:1px dashed rgba(255,215,160,.18); }
    .iamccs-isf-artboard { position:absolute; inset:0; }
    .iamccs-isf-item { position:absolute; border-radius:14px; border:2px solid rgba(188,244,240,.72); background:rgba(70,184,176,.12); box-shadow:0 8px 22px rgba(0,0,0,.22); cursor:grab; overflow:hidden; }
    .iamccs-isf-item.text { border-color:rgba(255,192,120,.86); background:rgba(255,161,74,.14); }
    .iamccs-isf-item.selected { box-shadow:0 0 0 2px rgba(255,255,255,.18),0 0 0 5px rgba(72,194,186,.28),0 14px 28px rgba(0,0,0,.3); }
    .iamccs-isf-item-head { display:flex; justify-content:space-between; align-items:center; gap:8px; padding:6px 8px; background:rgba(5,8,10,.48); font-weight:800; color:#fff8ea; text-shadow:0 1px 0 rgba(0,0,0,.35); }
    .iamccs-isf-item-kind { font-size:10px; letter-spacing:.08em; text-transform:uppercase; color:#96e8e2; }
    .iamccs-isf-item.text .iamccs-isf-item-kind { color:#ffd39e; }
    .iamccs-isf-item-body { padding:8px; color:#d9ece9; font-size:11px; }
    .iamccs-isf-handle { position:absolute; width:12px; height:12px; right:6px; bottom:6px; border-radius:50%; border:1px solid rgba(255,255,255,.7); background:#fff; cursor:nwse-resize; }
    .iamccs-isf-list { display:flex; flex-direction:column; gap:8px; min-height:360px; max-height:520px; overflow:auto; }
    .iamccs-isf-card { border:1px solid rgba(97,177,173,.18); background:#0b1417; border-radius:12px; padding:10px; cursor:pointer; }
    .iamccs-isf-card.selected { border-color:#f4b868; box-shadow:0 0 0 1px rgba(244,184,104,.35); }
    .iamccs-isf-card-title { display:flex; justify-content:space-between; gap:8px; font-weight:800; color:#fff; }
    .iamccs-isf-chip { display:inline-flex; align-items:center; padding:2px 8px; border-radius:999px; background:#143438; color:#9fece5; font-size:10px; text-transform:uppercase; letter-spacing:.08em; }
    .iamccs-isf-chip.text { background:#3a2614; color:#ffd8a5; }
    .iamccs-isf-card p { margin:6px 0 0; color:#a7c2bf; }
    .iamccs-isf-palette { display:flex; gap:6px; flex-wrap:wrap; margin-top:8px; }
    .iamccs-isf-swatch { width:18px; height:18px; border-radius:50%; border:1px solid rgba(255,255,255,.24); box-shadow:0 2px 6px rgba(0,0,0,.25); }
    .iamccs-isf-posgrid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }
    .iamccs-isf-preview { width:100%; min-height:180px; border:1px solid #2b4d50; background:#091215; color:#d8eeeb; border-radius:10px; padding:8px 10px; resize:vertical; font:11px/1.4 Consolas,monospace; }
    .iamccs-isf-foot { display:flex; justify-content:space-between; gap:10px; padding:12px 16px; border-top:1px solid rgba(120,200,198,.12); color:#95b3af; background:#0a1012; }
  `;
  document.head.appendChild(style);
}

function install(node) {
  if (!node || node._iamccsIdeogramStoryboardReady || typeof node.addDOMWidget !== "function") return;
  node._iamccsIdeogramStoryboardReady = true;
  ensureStyles();
  hideWidget(widget(node, "design_data"));
  hideWidget(widget(node, "ideoboard_input_signature"));

  const state = {
    data: parseData(node),
    selectedId: null,
    drag: null,
  };
  state.selectedId = state.data.items?.[0]?.id || null;
  writeData(node, state.data);

  const mountHost = document.createElement("div");
  mountHost.className = "iamccs-isf-mount";

  const root = document.createElement("div");
  root.className = "iamccs-isf";
  root.innerHTML = `
    <div class="iamccs-isf-head">
      <div>
        <div class="iamccs-isf-title">IAMCCS StoryboardFrame + TextInFrame Director</div>
        <div class="iamccs-isf-sub">Visual layout canvas for Ideogram-style shot design, prop text, signage, title zones, and art-directed frame composition.</div>
      </div>
      <div class="iamccs-isf-toolbar">
        <button class="iamccs-isf-btn" data-action="save-ideoboard">Save Ideoboard</button>
        <button class="iamccs-isf-btn" data-action="import-ideoboard">Import Ideoboard</button>
        <button class="iamccs-isf-btn" data-action="toggle-fullscreen">Open Editor</button>
        <button class="iamccs-isf-btn" data-action="copy-json">Copy Prompt JSON</button>
        <button class="iamccs-isf-btn" data-action="add-object">Add Object</button>
        <button class="iamccs-isf-btn" data-action="add-text">Add Text</button>
        <button class="iamccs-isf-btn" data-action="duplicate">Duplicate</button>
        <button class="iamccs-isf-btn" data-action="delete">Delete</button>
        <button class="iamccs-isf-btn primary" data-action="reset">Reset Layout</button>
      </div>
    </div>
    <div class="iamccs-isf-body">
      <div class="iamccs-isf-pane" data-pane="scene"></div>
      <div class="iamccs-isf-stage-wrap">
        <div class="iamccs-isf-stage-meta"><strong>Frame Canvas</strong><span data-stage-size></span></div>
        <div class="iamccs-isf-stage">
          <div class="iamccs-isf-grid"></div>
          <div class="iamccs-isf-overlay"></div>
          <div class="iamccs-isf-artboard"></div>
        </div>
      </div>
      <div class="iamccs-isf-pane" data-pane="inspector"></div>
    </div>
    <div class="iamccs-isf-foot"><span>patreon.com/IAMCCS</span><span data-foot-status></span></div>
    <input type="file" accept=".ideoboard.json,.json" data-role="ideoboard-import" style="display:none" />
  `;

  const scenePane = root.querySelector('[data-pane="scene"]');
  const inspectorPane = root.querySelector('[data-pane="inspector"]');
  const artboard = root.querySelector('.iamccs-isf-artboard');
  const footStatus = root.querySelector('[data-foot-status]');
  const stageSize = root.querySelector('[data-stage-size]');
  const fullscreenButton = root.querySelector('[data-action="toggle-fullscreen"]');
  const importInput = root.querySelector('[data-role="ideoboard-import"]');

  const sceneFields = {};
  const itemFields = {};
  let previewField = null;
  let layerListHost = null;
  let fullscreenHost = null;

  function applyNodeSize() {
    node.resizable = false;
    node.size = [1580, 1020];
    node.setDirtyCanvas?.(true, true);
  }

  function currentItem() {
    return (state.data.items || []).find((entry) => entry.id === state.selectedId) || null;
  }

  function syncSelectionState() {
    requestAnimationFrame(() => {
      const selectedCard = layerListHost?.querySelector(`[data-item-id="${state.selectedId}"]`);
      selectedCard?.scrollIntoView({ block: 'nearest' });
    });
  }

  function toggleFullscreen(force) {
    const next = typeof force === 'boolean' ? force : !root.classList.contains('iamccs-isf-fullscreen');
    if (next) {
      if (!fullscreenHost) {
        fullscreenHost = document.createElement('div');
        fullscreenHost.style.position = 'fixed';
        fullscreenHost.style.inset = '0';
        fullscreenHost.style.zIndex = '2147483639';
        fullscreenHost.style.background = 'rgba(3, 6, 8, 0.72)';
      }
      document.body.appendChild(fullscreenHost);
      fullscreenHost.appendChild(root);
    } else if (fullscreenHost?.parentNode) {
      fullscreenHost.remove();
      mountHost.appendChild(root);
      root.style.display = '';
      root.getBoundingClientRect();
      requestAnimationFrame(() => {
        applyNodeSize();
        node.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas?.(true, true);
      });
    }
    root.classList.toggle('iamccs-isf-fullscreen', next);
    if (fullscreenButton) {
      fullscreenButton.textContent = next ? 'Close Editor' : 'Open Editor';
    }
  }

  async function downloadIdeoboard() {
    const suggested = `${safeFilename(`IAMCCS_${state.data.preset_key}_board`)}.ideoboard.json`;
    const payloadData = buildIdeoboardPackage(state.data, suggested.replace(/\.ideoboard\.json$/i, ''));
    const payload = JSON.stringify(payloadData, null, 2);
    footStatus.textContent = 'Preparing Ideoboard save...';
    const pickerWindow = saveDialogWindow();
    if (pickerWindow) {
      try {
        const handle = await pickerWindow.showSaveFilePicker({
          suggestedName: suggested,
          types: [{
            description: 'IAMCCS Ideoboard JSON',
            accept: { 'application/json': ['.json'] },
          }],
        });
        const writable = await handle.createWritable();
        await writable.write(payload);
        await writable.close();
        footStatus.textContent = `Ideoboard saved to ${handle.name}`;
        return;
      } catch (error) {
        if (error?.name === 'AbortError') {
          footStatus.textContent = 'Ideoboard save cancelled';
          return;
        }
        footStatus.textContent = `Save As unavailable, using browser download (${error?.name || 'runtime error'})`;
      }
    }
    try {
      const filename = suggested;
      const blob = new Blob([payload], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      link.rel = 'noopener';
      link.style.display = 'none';
      (document.body || root).appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      footStatus.textContent = `Ideoboard downloaded as ${filename}`;
    } catch (error) {
      footStatus.textContent = `Ideoboard save failed (${error?.name || 'runtime error'})`;
    }
  }

  function importIdeoboardFile(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        state.data = designFromImportedBoard(JSON.parse(String(reader.result || '')));
        state.selectedId = state.data.items[0]?.id || null;
        writeInputSignature(node, "");
        persist();
        render();
        footStatus.textContent = `Imported ideoboard ${file.name}`;
      } catch {
        footStatus.textContent = `Import failed for ${file.name}`;
      }
    };
    reader.readAsText(file);
  }

  function applyRuntimeIdeoboard(payload, signature, usedIncoming) {
    if (!payload) return;
    try {
      const raw = typeof payload === 'string' ? JSON.parse(payload) : payload;
      const next = designFromImportedBoard(raw);
      state.data = next;
      state.selectedId = state.data.items?.[0]?.id || null;
      writeData(node, state.data);
      writeInputSignature(node, signature || "");
      render();
      const mode = usedIncoming ? 'loaded from IdeoTranslate input' : 'using edited canvas working copy';
      footStatus.textContent = `Runtime ideoboard ${mode} (${state.data.canvas.width}x${state.data.canvas.height})`;
    } catch (error) {
      footStatus.textContent = `Runtime ideoboard sync failed (${error?.name || 'invalid JSON'})`;
    }
  }

  function persist() {
    writeData(node, state.data);
    footStatus.textContent = `Saved ${state.data.items.length} layers to design_data`;
    if (previewField) previewField.value = JSON.stringify(toPrompt(state.data), null, 2);
  }

  function makeField(container, label, key, opts = {}) {
    const wrap = document.createElement('div');
    wrap.className = `iamccs-isf-field ${opts.small ? 'small' : ''}`;
    const lbl = document.createElement('label');
    lbl.textContent = label;
    const input = opts.multiline ? document.createElement('textarea') : document.createElement('input');
    if (!opts.multiline) {
      input.type = opts.type || 'text';
    }
    wrap.append(lbl, input);
    container.appendChild(wrap);
    return input;
  }

  function buildScenePane() {
    scenePane.innerHTML = '';
    const presets = document.createElement('div');
    presets.className = 'iamccs-isf-panel';
    presets.innerHTML = '<h4>IAMCCS Presets</h4><div class="iamccs-isf-preset-grid" data-preset-grid></div>';
    scenePane.appendChild(presets);

    const presetGrid = presets.querySelector('[data-preset-grid]');
    Object.entries(PRESETS).forEach(([key, preset]) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = `iamccs-isf-preset ${state.data.preset_key === key ? 'selected' : ''}`;
      button.innerHTML = `<strong>${preset.label}</strong><span>${preset.summary}</span>`;
      button.addEventListener('click', () => {
        const applied = presetData(key);
        state.data = {
          ...applied,
          items: key === 'title_card'
            ? [{ ...createItem('text'), id: 'item_001', label: 'Title Block', text: 'TITLE', x: 180, y: 310, w: 640, h: 150, desc: 'Primary title zone with strong readability and elegant negative space.' }]
            : state.data.items?.length ? state.data.items.map(normalizeItem) : applied.items,
        };
        state.selectedId = state.data.items[0]?.id || null;
        persist();
        render();
      });
      presetGrid.appendChild(button);
    });

    const canvas = document.createElement('div');
    canvas.className = 'iamccs-isf-panel';
    canvas.innerHTML = '<h4>Canvas Controls</h4>';
    scenePane.appendChild(canvas);
    const canvasGrid = document.createElement('div');
    canvasGrid.className = 'iamccs-isf-posgrid';
    canvas.appendChild(canvasGrid);
    sceneFields.width = makeField(canvasGrid, 'Width', 'width', { type: 'number' });
    sceneFields.height = makeField(canvasGrid, 'Height', 'height', { type: 'number' });
    sceneFields.aspect = makeField(canvas, 'Aspect Label', 'aspect');

    const summary = document.createElement('div');
    summary.className = 'iamccs-isf-panel';
    summary.innerHTML = '<h4>Scene Direction</h4>';
    scenePane.appendChild(summary);
    sceneFields.high = makeField(summary, 'High-Level Description', 'high', { multiline: true });
    sceneFields.background = makeField(summary, 'Background', 'background', { multiline: true, small: true });

    const style = document.createElement('div');
    style.className = 'iamccs-isf-panel';
    style.innerHTML = '<h4>Art Direction</h4>';
    scenePane.appendChild(style);
    sceneFields.aesthetics = makeField(style, 'Aesthetics', 'aesthetics', { multiline: true, small: true });
    sceneFields.lighting = makeField(style, 'Lighting', 'lighting', { multiline: true, small: true });
    sceneFields.medium = makeField(style, 'Medium', 'medium');
    sceneFields.photo = makeField(style, 'Photo / Lens Notes', 'photo', { multiline: true, small: true });
    sceneFields.palette = makeField(style, 'Global Palette', 'palette');

    Object.entries(sceneFields).forEach(([key, input]) => {
      input.addEventListener('input', () => {
        if (key === 'width') state.data.canvas.width = clampInt(input.value, 256, 4096, state.data.canvas.width || 1024);
        else if (key === 'height') state.data.canvas.height = clampInt(input.value, 256, 4096, state.data.canvas.height || 1024);
        else if (key === 'aspect') state.data.canvas.aspect_label = input.value;
        else if (key === 'high') state.data.scene.high_level_description = input.value;
        else if (key === 'background') state.data.scene.background = input.value;
        else if (key === 'aesthetics') state.data.scene.aesthetics = input.value;
        else if (key === 'lighting') state.data.scene.lighting = input.value;
        else if (key === 'medium') state.data.scene.medium = input.value;
        else if (key === 'photo') state.data.scene.photo = input.value;
        else if (key === 'palette') state.data.scene.color_palette = paletteList(input.value, state.data.scene.color_palette);
        persist();
        render();
      });
    });
  }

  function buildInspectorPane() {
    inspectorPane.innerHTML = '';
    const layers = document.createElement('div');
    layers.className = 'iamccs-isf-panel';
    layers.innerHTML = '<h4>Frame Layers</h4><div class="iamccs-isf-list" data-layer-list></div>';
    inspectorPane.appendChild(layers);
    layerListHost = layers.querySelector('[data-layer-list]');

    const itemPanel = document.createElement('div');
    itemPanel.className = 'iamccs-isf-panel';
    itemPanel.innerHTML = '<h4>Selected Layer</h4>';
    inspectorPane.appendChild(itemPanel);
    itemFields.label = makeField(itemPanel, 'Layer Label', 'label');
    itemFields.text = makeField(itemPanel, 'Rendered Text', 'text', { multiline: true, small: true });
    itemFields.desc = makeField(itemPanel, 'Visual Description', 'desc', { multiline: true });
    itemFields.palette = makeField(itemPanel, 'Layer Palette', 'palette');
    const pos = document.createElement('div');
    pos.className = 'iamccs-isf-posgrid';
    itemPanel.appendChild(pos);
    itemFields.x = makeField(pos, 'X', 'x', { type: 'number' });
    itemFields.y = makeField(pos, 'Y', 'y', { type: 'number' });
    itemFields.w = makeField(pos, 'Width', 'w', { type: 'number' });
    itemFields.h = makeField(pos, 'Height', 'h', { type: 'number' });

    const previewPanel = document.createElement('div');
    previewPanel.className = 'iamccs-isf-panel';
    previewPanel.innerHTML = '<h4>Prompt JSON Export</h4>';
    previewField = document.createElement('textarea');
    previewField.className = 'iamccs-isf-preview';
    previewField.readOnly = true;
    previewPanel.appendChild(previewField);
    inspectorPane.appendChild(previewPanel);

    Object.entries(itemFields).forEach(([key, input]) => {
      input.addEventListener('input', () => {
        const item = currentItem();
        if (!item) return;
        if (['x', 'y', 'w', 'h'].includes(key)) {
          item[key] = clampInt(input.value, key === 'w' || key === 'h' ? 20 : 0, 1000, item[key]);
          if (key === 'x') item.w = Math.min(item.w, 1000 - item.x);
          if (key === 'y') item.h = Math.min(item.h, 1000 - item.y);
          if (key === 'w') item.w = Math.min(item.w, 1000 - item.x);
          if (key === 'h') item.h = Math.min(item.h, 1000 - item.y);
        } else if (key === 'palette') {
          item.color_palette = paletteList(input.value, item.color_palette);
        } else {
          item[key] = input.value;
        }
        persist();
        render();
      });
    });
  }

  function createItem(kind) {
    const next = (state.data.items?.length || 0) + 1;
    return {
      id: `item_${String(next).padStart(3, '0')}`,
      kind,
      label: kind === 'text' ? `Text ${next}` : `Object ${next}`,
      text: kind === 'text' ? 'TEXT' : '',
      x: 170 + (next * 12),
      y: 150 + (next * 10),
      w: kind === 'text' ? 240 : 280,
      h: kind === 'text' ? 110 : 240,
      desc: kind === 'text' ? 'Readable in-frame text element with deliberate styling and placement.' : 'Visual subject block placed for clear silhouette and cinematic balance.',
      color_palette: kind === 'text' ? ['#FFB65D', '#FFE4B5'] : ['#8B4513', '#1A1A2E', '#FFE4B5'],
    };
  }

  function syncSceneFields() {
    sceneFields.width.value = state.data.canvas.width || 1024;
    sceneFields.height.value = state.data.canvas.height || 1024;
    sceneFields.aspect.value = state.data.canvas.aspect_label || '';
    sceneFields.high.value = state.data.scene.high_level_description || '';
    sceneFields.background.value = state.data.scene.background || '';
    sceneFields.aesthetics.value = state.data.scene.aesthetics || '';
    sceneFields.lighting.value = state.data.scene.lighting || '';
    sceneFields.medium.value = state.data.scene.medium || '';
    sceneFields.photo.value = state.data.scene.photo || '';
    sceneFields.palette.value = (state.data.scene.color_palette || []).join(', ');
    stageSize.textContent = `${state.data.canvas.width} × ${state.data.canvas.height} • ${state.data.canvas.aspect_label || 'Canvas'}`;
    const presetGrid = scenePane.querySelector('[data-preset-grid]');
    presetGrid?.querySelectorAll('.iamccs-isf-preset').forEach((button, index) => {
      const key = Object.keys(PRESETS)[index];
      button.classList.toggle('selected', key === state.data.preset_key);
    });
  }

  function syncItemFields() {
    const item = currentItem();
    const disabled = !item;
    Object.values(itemFields).forEach((input) => { input.disabled = disabled; });
    if (!item) {
      Object.values(itemFields).forEach((input) => { input.value = ''; });
      return;
    }
    itemFields.label.value = item.label || '';
    itemFields.text.value = item.text || '';
    itemFields.text.disabled = item.kind !== 'text';
    itemFields.desc.value = item.desc || '';
    itemFields.palette.value = (item.color_palette || []).join(', ');
    itemFields.x.value = item.x;
    itemFields.y.value = item.y;
    itemFields.w.value = item.w;
    itemFields.h.value = item.h;
  }

  function renderLayerList() {
    const host = layerListHost;
    if (!host) return;
    host.innerHTML = '';
    state.data.items.forEach((item) => {
      const card = document.createElement('button');
      card.type = 'button';
      card.dataset.itemId = item.id;
      card.className = `iamccs-isf-card ${item.id === state.selectedId ? 'selected' : ''}`;
      card.innerHTML = `
        <div class="iamccs-isf-card-title">
          <span>${item.label || item.id}</span>
          <span class="iamccs-isf-chip ${item.kind === 'text' ? 'text' : ''}">${item.kind}</span>
        </div>
        <p>${item.desc || ''}</p>
        <div class="iamccs-isf-palette">${(item.color_palette || []).map((color) => `<span class="iamccs-isf-swatch" style="background:${color}"></span>`).join('')}</div>
      `;
      card.addEventListener('click', () => {
        state.selectedId = item.id;
        render();
        syncSelectionState();
      });
      host.appendChild(card);
    });
  }

  function startDrag(event, item, mode) {
    const rect = artboard.getBoundingClientRect();
    state.drag = {
      mode,
      id: item.id,
      startX: event.clientX,
      startY: event.clientY,
      origin: { x: item.x, y: item.y, w: item.w, h: item.h },
      rect,
    };
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', stopDrag, { once: true });
  }

  function onPointerMove(event) {
    if (!state.drag) return;
    const item = state.data.items.find((entry) => entry.id === state.drag.id);
    if (!item) return;
    const dx = ((event.clientX - state.drag.startX) / state.drag.rect.width) * 1000;
    const dy = ((event.clientY - state.drag.startY) / state.drag.rect.height) * 1000;
    if (state.drag.mode === 'move') {
      item.x = clampInt(state.drag.origin.x + dx, 0, 1000 - item.w, item.x);
      item.y = clampInt(state.drag.origin.y + dy, 0, 1000 - item.h, item.y);
    } else {
      item.w = clampInt(state.drag.origin.w + dx, 20, 1000 - item.x, item.w);
      item.h = clampInt(state.drag.origin.h + dy, 20, 1000 - item.y, item.h);
    }
    persist();
    render();
  }

  function stopDrag() {
    state.drag = null;
    window.removeEventListener('pointermove', onPointerMove);
  }

  function renderArtboard() {
    artboard.innerHTML = '';
    const palette = state.data.scene.color_palette || [];
    const bgA = palette[0] || '#18202A';
    const bgB = palette[1] || '#0D1118';
    artboard.style.background = `radial-gradient(circle at 20% 15%, ${bgA}55, transparent 25%), linear-gradient(180deg, ${bgA}, ${bgB})`;
    state.data.items.forEach((item) => {
      const box = document.createElement('div');
      box.dataset.itemId = item.id;
      box.className = `iamccs-isf-item ${item.kind === 'text' ? 'text' : ''} ${item.id === state.selectedId ? 'selected' : ''}`;
      box.style.left = `${item.x / 10}%`;
      box.style.top = `${item.y / 10}%`;
      box.style.width = `${item.w / 10}%`;
      box.style.height = `${item.h / 10}%`;
      box.innerHTML = `
        <div class="iamccs-isf-item-head">
          <span>${item.label || item.id}</span>
          <span class="iamccs-isf-item-kind">${item.kind}</span>
        </div>
        <div class="iamccs-isf-item-body">${item.kind === 'text' ? (item.text || 'Text block') : (item.desc || 'Object block')}</div>
        <span class="iamccs-isf-handle"></span>
      `;
      box.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();
        state.selectedId = item.id;
        render();
        syncSelectionState();
      });
      box.addEventListener('pointerdown', (event) => {
        event.preventDefault();
        state.selectedId = item.id;
        const handle = event.target.closest('.iamccs-isf-handle');
        startDrag(event, item, handle ? 'resize' : 'move');
        render();
        syncSelectionState();
      });
      artboard.appendChild(box);
    });
  }

  function render() {
    syncSceneFields();
    syncItemFields();
    renderLayerList();
    renderArtboard();
  }

  importInput?.addEventListener('change', (event) => {
    const file = event.target.files?.[0];
    importIdeoboardFile(file);
    event.target.value = '';
  });

  root.querySelectorAll('[data-action]').forEach((button) => {
    button.addEventListener('click', () => {
      const action = button.dataset.action;
      if (action === 'save-ideoboard') {
        downloadIdeoboard();
        return;
      } else if (action === 'import-ideoboard') {
        importInput?.click();
        return;
      } else if (action === 'add-object') {
        const item = createItem('obj');
        state.data.items.push(item);
        state.selectedId = item.id;
      } else if (action === 'add-text') {
        const item = createItem('text');
        state.data.items.push(item);
        state.selectedId = item.id;
      } else if (action === 'duplicate') {
        const item = currentItem();
        if (!item) return;
        const clone = normalizeItem({ ...item, id: '', x: item.x + 24, y: item.y + 24, label: `${item.label} copy` }, state.data.items.length);
        state.data.items.push(clone);
        state.selectedId = clone.id;
      } else if (action === 'delete') {
        if (!state.selectedId) return;
        state.data.items = state.data.items.filter((entry) => entry.id !== state.selectedId);
        state.selectedId = state.data.items[0]?.id || null;
      } else if (action === 'copy-json') {
        const payload = JSON.stringify(toPrompt(state.data), null, 2);
        previewField.value = payload;
        navigator.clipboard?.writeText(payload).then(() => {
          footStatus.textContent = 'Prompt JSON copied to clipboard';
        }).catch(() => {
          previewField.focus();
          previewField.select();
          footStatus.textContent = 'Prompt JSON selected for manual copy';
        });
        return;
      } else if (action === 'toggle-fullscreen') {
        toggleFullscreen();
        return;
      } else if (action === 'reset') {
        state.data = defaultData();
        state.selectedId = state.data.items[0]?.id || null;
        writeInputSignature(node, "");
      }
      persist();
      render();
    });
  });

  buildScenePane();
  buildInspectorPane();
  mountHost.appendChild(root);

  const domWidget = node.addDOMWidget('IAMCCS StoryboardFrame Director', 'iamccs_storyboard_frame_designer', mountHost, { serialize: false });
  domWidget.computeSize = () => [1540, 940];
  const originalOnResize = node.onResize;
  node.onResize = function () {
    applyNodeSize();
    return originalOnResize?.apply(this, arguments);
  };
  const originalOnExecuted = node.onExecuted;
  node.onExecuted = function (message) {
    const result = originalOnExecuted?.apply(this, arguments);
    const pick = (value) => Array.isArray(value) ? value[0] : value;
    const payload = pick(message?.design_data);
    if (payload) {
      applyRuntimeIdeoboard(
        payload,
        pick(message?.ideoboard_input_signature),
        Boolean(pick(message?.used_ideoboard_input))
      );
    }
    return result;
  };
  applyNodeSize();
  window.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && root.classList.contains('iamccs-isf-fullscreen')) {
      toggleFullscreen(false);
    }
  });
  render();
}

app.registerExtension({
  name: 'IAMCCS.StoryboardFrameDesigner',
  nodeCreated(node) {
    const type = node?.comfyClass || node?.type || node?.constructor?.type || '';
    if (type === TYPE || node?.type === TYPE) [0, 180, 600].forEach((delay) => setTimeout(() => install(node), delay));
  },
  loadedGraphNode(node) {
    const type = node?.comfyClass || node?.type || node?.constructor?.type || '';
    if (type === TYPE || node?.type === TYPE) [0, 180, 600].forEach((delay) => setTimeout(() => install(node), delay));
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== TYPE) return;
    const original = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      original?.apply(this, arguments);
      [0, 180, 600].forEach((delay) => setTimeout(() => install(this), delay));
    };
  },
});