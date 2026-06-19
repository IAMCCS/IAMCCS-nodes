import { app } from "../../scripts/app.js";

const TYPE = "IAMCCS_IdeogramStoryboardSheet";
const STYLE_ID = "iamccs-ideogram-storyboard-sheet-style";

const DEFAULT_PANELS = [
  ["1. Corridor", "extreme wide action shot", "the protagonist runs ankle-deep through a flooded palace corridor of impossible doors", "low wide lens, off-center, the character small in frame"],
  ["2. Key", "medium-wide action shot", "the protagonist crawls under a rotten dinner table reaching for a tarnished brass key", "profile view, eyes on the key, no camera gaze"],
  ["3. Stairs", "low-angle full-body action shot", "the protagonist slips down a chessboard staircase toward a sunken theatre", "body in motion, one hand scraping the filthy wall"],
  ["4. Mirror", "tense mirror action shot", "the protagonist is pulled sideways through a cracked oval mirror", "twisted profile, reflection delayed and misaligned"],
  ["5. Saucer", "overhead action shot", "the protagonist crawls across a cracked tea saucer floating in black oily water", "top-down composition, red thread labyrinth around the body"],
  ["6. Door", "final wide action shot", "the protagonist forces open a black door into a rotten winter garden under a false moon", "seen from behind three-quarter angle, not looking at camera"],
];

const PRESETS = {
  sick_alice: {
    label: "Sick Alice",
    title: "IAMCCS Dark Alice Storyboard",
    character_bible: "same very thin adult Alice-like woman, dirty tangled red hair, pale sick skin, torn dark blue velvet coat, stained white collar, frayed red ribbon, tarnished brass key; she is inside the action, not posing, not looking into camera",
    world_bible: "dark surreal fantasy palace-world: flooded corridors, impossible doors, cracked mirrors, chessboard marble, red velvet, winter garden glass, fog, mold, stains, rust, mud, sickness, tactile photorealistic decay",
    style_bible: "extremely photorealistic dark surrealist fairytale cinema, Hungarian arthouse stillness, Russian poetic realism, Belgian symbolist production design, unsettling diseased Alice in Wonderland atmosphere",
    lighting_bible: "cold moonlit interiors, wet reflective floors, dim candle practicals, soft volumetric fog, deep blacks, restrained red accents, brass highlights, realistic sickly skin and dirty fabric texture",
    camera_bible: "ARRI Alexa 65 cinematic stills, anamorphic landscape 16:9 panels, action blocking, no front-facing portrait staging, no beauty-shot closeups, no centered foreground protagonist",
    negative_bible: "no portrait poster, no vertical sheet, no protagonist staring at viewer, no large foreground face, no clean beauty fashion image, no cheerful fairytale, no second protagonist",
    color_palette: ["#0A0A0D", "#1C2230", "#443348", "#7B1E2B", "#B9A66D", "#D8D1C0"],
  },
  tarkovsky_sci_fi: {
    label: "Tarkovsky Sci-Fi",
    title: "IAMCCS Contaminated Station Storyboard",
    character_bible: "same exhausted cosmonaut woman, thin body, shaved copper hair, cracked helmet under one arm, stained pressure suit, tired eyes; she moves through the action and avoids camera gaze",
    world_bible: "abandoned orbital monastery, condensation, rusted prayer machines, floating dust, flooded corridors, Soviet panels, Belgian glass, impossible gravity shifts",
    style_bible: "photorealistic philosophical science fiction, Tarkovsky patience, Eastern European surrealism, quiet dread, industrial sacred architecture",
    lighting_bible: "cold window light, dim emergency strips, reflective water, milky fog, oxidized metal, pale skin tones, deep black corridors",
    camera_bible: "wide 16:9 cinematic panels, restrained camera, slow observational compositions, protagonist small or mid-distance inside the action",
    negative_bible: "no hero poster, no clean spaceship glamour, no centered portrait, no smiling, no front-facing beauty closeup",
    color_palette: ["#080A0B", "#263238", "#53605C", "#8B1E2D", "#B7A46B", "#D6D0C2"],
  },
  dirty_fantasy: {
    label: "Dirty Fantasy",
    title: "IAMCCS Rotten Kingdom Storyboard",
    character_bible: "same very thin runaway princess, dirty copper hair, mud-stained wool dress, bruised hands, blackened silver crown hidden in cloth; always acting inside the scene",
    world_bible: "rotting medieval kingdom, flooded chapels, black roots through marble, diseased tapestries, broken icons, fog, wet stone, carrion colors",
    style_bible: "photorealistic dark fantasy cinema, surreal Belgian symbolist detail, Russian fairytale dread, Hungarian arthouse restraint",
    lighting_bible: "clouded moonlight, torch smoke, cold wet highlights, muddy shadows, candle glints on metal and glass",
    camera_bible: "landscape 16:9 story panels, action blocking, varied wide and medium-wide staging, no portrait poster logic",
    negative_bible: "no clean fantasy illustration, no glamour portrait, no smiling princess, no large face foreground, no vertical poster",
    color_palette: ["#070606", "#1C2022", "#41413A", "#6D1D24", "#9C814B", "#C8BBA0"],
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

function clean(value) {
  return String(value || "").trim();
}

function clampInt(value, min, max, fallback) {
  const number = Number.isFinite(Number(value)) ? Math.round(Number(value)) : fallback;
  return Math.max(min, Math.min(max, number));
}

function clone(value) {
  try { return JSON.parse(JSON.stringify(value)); } catch { return null; }
}

function panelFromTuple(tuple, index) {
  return {
    title: tuple?.[0] || `Panel ${index + 1}`,
    shot: tuple?.[1] || "wide action shot",
    action: tuple?.[2] || "the protagonist moves through the scene",
    camera: tuple?.[3] || "cinematic landscape framing, no camera gaze",
  };
}

function defaultData() {
  const base = PRESETS.sick_alice;
  return {
    schema: "iamccs.ideogram.storyboard_sheet",
    schema_version: 1,
    title: base.title,
    layout: { columns: 2, rows: 3, panel_width: 1024, panel_height: 576, gap: 0, orientation_guard: "wide landscape contact sheet, not portrait, not vertical" },
    character_bible: base.character_bible,
    world_bible: base.world_bible,
    style_bible: base.style_bible,
    lighting_bible: base.lighting_bible,
    camera_bible: base.camera_bible,
    negative_bible: base.negative_bible,
    color_palette: [...base.color_palette],
    panels: DEFAULT_PANELS.map(panelFromTuple),
  };
}

function normalizeData(raw) {
  const fallback = defaultData();
  let data = raw && typeof raw === "object" ? raw : fallback;
  const layout = { ...fallback.layout, ...(data.layout || {}) };
  layout.columns = clampInt(layout.columns, 1, 6, 2);
  layout.rows = clampInt(layout.rows, 1, 6, 3);
  layout.panel_width = clampInt(layout.panel_width, 256, 2048, 1024);
  layout.panel_height = clampInt(layout.panel_height, 256, 2048, 576);
  layout.gap = clampInt(layout.gap, 0, 128, 0);
  const count = layout.columns * layout.rows;
  const sourcePanels = Array.isArray(data.panels) && data.panels.length ? data.panels : fallback.panels;
  const panels = Array.from({ length: count }, (_, index) => ({
    title: clean(sourcePanels[index]?.title) || `Panel ${index + 1}`,
    shot: clean(sourcePanels[index]?.shot) || "wide action shot",
    action: clean(sourcePanels[index]?.action) || "the protagonist moves through the scene",
    camera: clean(sourcePanels[index]?.camera) || "cinematic landscape framing, no camera gaze",
  }));
  return {
    ...fallback,
    ...data,
    layout,
    color_palette: Array.isArray(data.color_palette) && data.color_palette.length ? data.color_palette : fallback.color_palette,
    panels,
  };
}

function parseData(node) {
  try {
    return normalizeData(JSON.parse(String(widget(node, "storyboard_data")?.value || "")));
  } catch {
    return defaultData();
  }
}

function installStyle() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
.iamccs-iss{box-sizing:border-box;width:100%;min-width:1180px;height:900px;background:#090a0b;color:#f3eee4;border:1px solid #25282d;border-radius:8px;overflow:hidden;font-family:Inter,Segoe UI,Arial,sans-serif;display:grid;grid-template-rows:auto 1fr auto}
.iamccs-iss *{box-sizing:border-box;letter-spacing:0}
.iamccs-iss-top{display:flex;align-items:center;gap:10px;padding:12px 14px;border-bottom:1px solid #25282d;background:#111317}
.iamccs-iss-brand{font-weight:800;font-size:14px;color:#f5dfb2}
.iamccs-iss-sub{font-size:12px;color:#9ca3af;margin-left:auto}
.iamccs-iss-main{display:grid;grid-template-columns:310px 1fr 360px;min-height:0}
.iamccs-iss-left,.iamccs-iss-mid,.iamccs-iss-right{min-height:0;overflow:auto;padding:12px}
.iamccs-iss-left{border-right:1px solid #25282d;background:#0c0d10}
.iamccs-iss-right{border-left:1px solid #25282d;background:#0c0d10}
.iamccs-iss-section{border:1px solid #25282d;border-radius:8px;background:#121418;margin-bottom:10px;padding:10px}
.iamccs-iss-section h4{margin:0 0 8px;font-size:12px;text-transform:uppercase;color:#cdbf9e;font-weight:800}
.iamccs-iss-row{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.iamccs-iss-field{display:flex;flex-direction:column;gap:4px;margin-bottom:8px}
.iamccs-iss-field label{font-size:11px;color:#9ca3af}
.iamccs-iss input,.iamccs-iss textarea,.iamccs-iss select{width:100%;background:#07080a;border:1px solid #30343a;border-radius:6px;color:#f3eee4;padding:7px 8px;font-size:12px;outline:none}
.iamccs-iss textarea{min-height:68px;resize:vertical;line-height:1.35}
.iamccs-iss input:focus,.iamccs-iss textarea:focus,.iamccs-iss select:focus{border-color:#b9a66d;box-shadow:0 0 0 1px #b9a66d44}
.iamccs-iss-btn{height:30px;border:1px solid #343941;background:#171a1f;color:#e9e3d7;border-radius:6px;font-size:12px;font-weight:700;cursor:pointer}
.iamccs-iss-btn:hover{border-color:#b9a66d;color:#fff4d1}
.iamccs-iss-btn.primary{background:#e7dec6;color:#08090a;border-color:#e7dec6}
.iamccs-iss-presets{display:grid;grid-template-columns:1fr;gap:6px}
.iamccs-iss-panel-card{border:1px solid #30343a;background:#15181d;border-radius:8px;padding:9px;margin-bottom:8px;cursor:pointer}
.iamccs-iss-panel-card.selected{border-color:#b9a66d;background:#1b1a16}
.iamccs-iss-panel-card strong{display:block;font-size:12px;color:#f1e7c8;margin-bottom:4px}
.iamccs-iss-panel-card span{display:block;font-size:11px;color:#aab0b8;line-height:1.25}
.iamccs-iss-preview{height:330px;border:1px solid #30343a;border-radius:8px;background:#050607;position:relative;overflow:hidden;margin-bottom:10px}
.iamccs-iss-grid{position:absolute;inset:0;display:grid;gap:var(--gap,0px);grid-template-columns:repeat(var(--cols),1fr);grid-template-rows:repeat(var(--rows),1fr)}
.iamccs-iss-cell{position:relative;border:1px solid #3c4149;background:linear-gradient(135deg,#15191f,#0b0c0f);padding:8px;overflow:hidden;display:grid;grid-template-rows:minmax(18px,auto) 1fr;gap:6px}
.iamccs-iss-cell.selected{outline:2px solid #b9a66d;z-index:2}
.iamccs-iss-cell-title{font-size:10px;font-weight:800;color:#f4dfb5;line-height:1.15;max-height:34px;overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical}
.iamccs-iss-cell-action{position:static;font-size:9px;color:#cfd3da;line-height:1.2;overflow:hidden;display:-webkit-box;-webkit-line-clamp:6;-webkit-box-orient:vertical}
.iamccs-iss-swatches{display:flex;gap:5px;flex-wrap:wrap}
.iamccs-iss-swatch{width:20px;height:20px;border-radius:5px;border:1px solid #000}
.iamccs-iss-bottom{display:flex;align-items:center;gap:10px;padding:8px 12px;border-top:1px solid #25282d;background:#111317;color:#9ca3af;font-size:12px}
.iamccs-iss-code{height:155px;font-family:Consolas,monospace;font-size:11px}
`;
  document.head.appendChild(style);
}

function install(node) {
  if (!node || node._iamccsIdeogramSheetReady || typeof node.addDOMWidget !== "function") return;
  node._iamccsIdeogramSheetReady = true;
  installStyle();

  ["storyboard_data"].forEach((name) => hideWidget(widget(node, name)));

  const state = { data: parseData(node), selected: 0 };
  const host = document.createElement("div");
  const root = document.createElement("div");
  root.className = "iamccs-iss";
  root.innerHTML = `
    <div class="iamccs-iss-top">
      <div class="iamccs-iss-brand">IAMCCS Ideogram Storyboard Sheet</div>
      <button class="iamccs-iss-btn primary" data-action="apply">Apply</button>
      <button class="iamccs-iss-btn" data-action="copy">Copy Prompt</button>
      <div class="iamccs-iss-sub" data-status>Ready</div>
    </div>
    <div class="iamccs-iss-main">
      <div class="iamccs-iss-left">
        <div class="iamccs-iss-section">
          <h4>Presets</h4>
          <div class="iamccs-iss-presets" data-presets></div>
        </div>
        <div class="iamccs-iss-section">
          <h4>Layout</h4>
          <div class="iamccs-iss-row">
            <div class="iamccs-iss-field"><label>Columns</label><input data-field="columns" type="number" min="1" max="6"></div>
            <div class="iamccs-iss-field"><label>Rows</label><input data-field="rows" type="number" min="1" max="6"></div>
            <div class="iamccs-iss-field"><label>Panel W</label><input data-field="panel_width" type="number" min="256" max="2048" step="8"></div>
            <div class="iamccs-iss-field"><label>Panel H</label><input data-field="panel_height" type="number" min="256" max="2048" step="8"></div>
          </div>
          <div class="iamccs-iss-field"><label>Grid Gap</label><input data-field="gap" type="number" min="0" max="128"></div>
        </div>
        <div class="iamccs-iss-section">
          <h4>Panels</h4>
          <div data-panel-list></div>
        </div>
      </div>
      <div class="iamccs-iss-mid">
        <div class="iamccs-iss-preview"><div class="iamccs-iss-grid" data-grid></div></div>
        <div class="iamccs-iss-section">
          <h4>Selected Panel</h4>
          <div class="iamccs-iss-field"><label>Title</label><input data-panel="title"></div>
          <div class="iamccs-iss-field"><label>Shot</label><input data-panel="shot"></div>
          <div class="iamccs-iss-field"><label>Action</label><textarea data-panel="action"></textarea></div>
          <div class="iamccs-iss-field"><label>Camera / Blocking</label><textarea data-panel="camera"></textarea></div>
        </div>
      </div>
      <div class="iamccs-iss-right">
        <div class="iamccs-iss-section">
          <h4>Story Bible</h4>
          <div class="iamccs-iss-field"><label>Title</label><input data-root="title"></div>
          <div class="iamccs-iss-field"><label>Character</label><textarea data-root="character_bible"></textarea></div>
          <div class="iamccs-iss-field"><label>World</label><textarea data-root="world_bible"></textarea></div>
          <div class="iamccs-iss-field"><label>Style</label><textarea data-root="style_bible"></textarea></div>
          <div class="iamccs-iss-field"><label>Lighting</label><textarea data-root="lighting_bible"></textarea></div>
          <div class="iamccs-iss-field"><label>Camera Rules</label><textarea data-root="camera_bible"></textarea></div>
          <div class="iamccs-iss-field"><label>Negative Rules</label><textarea data-root="negative_bible"></textarea></div>
          <div class="iamccs-iss-field"><label>Palette</label><input data-root="color_palette"></div>
          <div class="iamccs-iss-swatches" data-swatches></div>
        </div>
        <div class="iamccs-iss-section">
          <h4>Prompt Preview</h4>
          <textarea class="iamccs-iss-code" data-preview readonly></textarea>
        </div>
      </div>
    </div>
    <div class="iamccs-iss-bottom">
      <span data-size></span>
      <span>Outputs: prompt_text, prompt_json, width, height, crop manifest, panel prompts</span>
    </div>
  `;
  host.appendChild(root);

  const q = (selector) => root.querySelector(selector);
  const qa = (selector) => Array.from(root.querySelectorAll(selector));
  const status = q("[data-status]");
  const grid = q("[data-grid]");
  const panelList = q("[data-panel-list]");
  const preview = q("[data-preview]");
  const sizeLabel = q("[data-size]");

  function persist() {
    const data = normalizeData(state.data);
    state.data = data;
    setWidget(node, "columns", data.layout.columns);
    setWidget(node, "rows", data.layout.rows);
    setWidget(node, "panel_width", data.layout.panel_width);
    setWidget(node, "panel_height", data.layout.panel_height);
    setWidget(node, "grid_gap", data.layout.gap);
    setWidget(node, "storyboard_data", JSON.stringify(data, null, 2));
  }

  function promptPreview() {
    const data = state.data;
    return [
      `${data.title}`,
      `${data.layout.columns}x${data.layout.rows} contact sheet, panel ${data.layout.panel_width}x${data.layout.panel_height}`,
      `Character: ${data.character_bible}`,
      `World: ${data.world_bible}`,
      `Style: ${data.style_bible}`,
      `Negative: ${data.negative_bible}`,
      "",
      ...data.panels.map((panel, index) => `${index + 1}. ${panel.title}: ${panel.shot}; ${panel.action}; ${panel.camera}`),
    ].join("\n");
  }

  function renderPresets() {
    const host = q("[data-presets]");
    host.innerHTML = "";
    Object.entries(PRESETS).forEach(([key, preset]) => {
      const button = document.createElement("button");
      button.className = "iamccs-iss-btn";
      button.type = "button";
      button.textContent = preset.label;
      button.addEventListener("click", () => {
        state.data = normalizeData({ ...state.data, ...clone(preset), panels: state.data.panels });
        persist();
        render();
        status.textContent = `${preset.label} preset applied`;
      });
      host.appendChild(button);
    });
  }

  function renderLayoutFields() {
    const layout = state.data.layout;
    qa("[data-field]").forEach((input) => { input.value = layout[input.dataset.field] ?? ""; });
    qa("[data-root]").forEach((input) => {
      const key = input.dataset.root;
      input.value = key === "color_palette" ? (state.data.color_palette || []).join(", ") : (state.data[key] || "");
    });
    const swatches = q("[data-swatches]");
    swatches.innerHTML = "";
    (state.data.color_palette || []).forEach((color) => {
      const swatch = document.createElement("span");
      swatch.className = "iamccs-iss-swatch";
      swatch.style.background = color;
      swatches.appendChild(swatch);
    });
  }

  function renderPanels() {
    panelList.innerHTML = "";
    state.data.panels.forEach((panel, index) => {
      const card = document.createElement("div");
      card.className = `iamccs-iss-panel-card ${index === state.selected ? "selected" : ""}`;
      card.innerHTML = `<strong>${panel.title}</strong><span>${panel.shot}</span><span>${panel.action}</span>`;
      card.addEventListener("click", () => {
        state.selected = index;
        render();
      });
      panelList.appendChild(card);
    });
    const panel = state.data.panels[state.selected] || state.data.panels[0];
    qa("[data-panel]").forEach((input) => { input.value = panel?.[input.dataset.panel] || ""; });
  }

  function renderGrid() {
    const layout = state.data.layout;
    grid.style.setProperty("--cols", layout.columns);
    grid.style.setProperty("--rows", layout.rows);
    grid.style.setProperty("--gap", `${Math.min(18, layout.gap)}px`);
    grid.innerHTML = "";
    state.data.panels.forEach((panel, index) => {
      const cell = document.createElement("div");
      cell.className = `iamccs-iss-cell ${index === state.selected ? "selected" : ""}`;
      const shortTitle = String(panel.title || `Panel ${index + 1}`).replace(/^\\s*\\d+\\.?\\s*Panel\\s+\\d+[:,]?\\s*/i, `${index + 1}. `);
      const shortAction = String(panel.action || "").replace(/^\\s*Panel\\s+\\d+[:,]?\\s*/i, "");
      cell.innerHTML = `<div class="iamccs-iss-cell-title"></div><div class="iamccs-iss-cell-action"></div>`;
      cell.querySelector(".iamccs-iss-cell-title").textContent = shortTitle;
      cell.querySelector(".iamccs-iss-cell-action").textContent = shortAction;
      cell.addEventListener("click", () => {
        state.selected = index;
        render();
      });
      grid.appendChild(cell);
    });
    const width = layout.columns * layout.panel_width + Math.max(0, layout.columns - 1) * layout.gap;
    const height = layout.rows * layout.panel_height + Math.max(0, layout.rows - 1) * layout.gap;
    sizeLabel.textContent = `${width} x ${height} sheet`;
  }

  function render() {
    state.data = normalizeData(state.data);
    state.selected = clampInt(state.selected, 0, Math.max(0, state.data.panels.length - 1), 0);
    renderLayoutFields();
    renderPanels();
    renderGrid();
    preview.value = promptPreview();
  }

  qa("[data-field]").forEach((input) => {
    input.addEventListener("input", () => {
      state.data.layout[input.dataset.field] = Number(input.value);
      const oldCount = state.data.panels.length;
      state.data = normalizeData(state.data);
      if (state.data.panels.length > oldCount) {
        for (let i = oldCount; i < state.data.panels.length; i++) {
          state.data.panels[i] = panelFromTuple(DEFAULT_PANELS[i % DEFAULT_PANELS.length], i);
        }
      }
      persist();
      render();
    });
  });

  qa("[data-root]").forEach((input) => {
    input.addEventListener("input", () => {
      const key = input.dataset.root;
      if (key === "color_palette") {
        state.data.color_palette = input.value.split(",").map((entry) => clean(entry)).filter(Boolean);
      } else {
        state.data[key] = input.value;
      }
      persist();
      renderGrid();
      preview.value = promptPreview();
    });
  });

  qa("[data-panel]").forEach((input) => {
    input.addEventListener("input", () => {
      const panel = state.data.panels[state.selected];
      if (!panel) return;
      panel[input.dataset.panel] = input.value;
      persist();
      renderPanels();
      renderGrid();
      preview.value = promptPreview();
    });
  });

  root.querySelectorAll("[data-action]").forEach((button) => {
    button.addEventListener("click", () => {
      if (button.dataset.action === "copy") {
        navigator.clipboard?.writeText(preview.value);
        status.textContent = "Prompt preview copied";
        return;
      }
      persist();
      render();
      status.textContent = "Storyboard sheet data applied";
    });
  });

  renderPresets();
  persist();
  render();

  const domWidget = node.addDOMWidget("IAMCCS Ideogram Storyboard Sheet", "iamccs_ideogram_storyboard_sheet", host, { serialize: false });
  domWidget.computeSize = () => [1240, 930];

  const originalOnExecuted = node.onExecuted;
  node.onExecuted = function (message) {
    const result = originalOnExecuted?.apply(this, arguments);
    const payload = Array.isArray(message?.storyboard_data) ? message.storyboard_data[0] : message?.storyboard_data;
    if (payload) {
      try {
        state.data = normalizeData(JSON.parse(payload));
        persist();
        render();
      } catch {}
    }
    return result;
  };
}

app.registerExtension({
  name: "IAMCCS.IdeogramStoryboardSheet",
  nodeCreated(node) {
    const type = node?.comfyClass || node?.type || node?.constructor?.type || "";
    if (type === TYPE || node?.type === TYPE) [0, 150, 500].forEach((delay) => setTimeout(() => install(node), delay));
  },
  loadedGraphNode(node) {
    const type = node?.comfyClass || node?.type || node?.constructor?.type || "";
    if (type === TYPE || node?.type === TYPE) [0, 150, 500].forEach((delay) => setTimeout(() => install(node), delay));
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== TYPE) return;
    const original = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      original?.apply(this, arguments);
      [0, 150, 500].forEach((delay) => setTimeout(() => install(this), delay));
    };
  },
});
