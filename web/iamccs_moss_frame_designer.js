import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { upstreamDesigners, layoutSignature, latestSavedRender } from "./moss_frame_routes.js";
import { LIGHT_PRESETS, CAMERA_PRESETS, STYLE_PRESETS } from "./moss_frame_presets.js";
import { importedMboards, portableDesign, cloneProjectData } from "./moss_mboard_library.js";

const TYPE = "IAMCCS_MOSSFrameDesigner";
const SCHEMA = "iamccs.moss.frame_designer";
const COLORS = { character: "#eaa853", object: "#66bcb2", text: "#ca9dea" };
const EDITOR_WIDTH = 1120;
const EDITOR_HEIGHT = 880;
const RESOLUTIONS = [
  ["512 × 512 · 0.25 MP", 512, 512], ["1024 × 1024 · 1 MP", 1024, 1024],
  ["1280 × 720 · HD", 1280, 720], ["720 × 1280 · Portrait", 720, 1280],
  ["1536 × 1024 · 1.5 MP", 1536, 1024], ["1024 × 1536 · Portrait", 1024, 1536],
  ["1920 × 1088 · 2 MP", 1920, 1088], ["1088 × 1920 · Portrait", 1088, 1920],
  ["2048 × 864 · Cinema scope", 2048, 864], ["2048 × 1024 · 2 MP", 2048, 1024],
  ["2048 × 2048 · 4 MP extrapolation", 2048, 2048],
  ["2560 × 1440 · 3.7 MP extrapolation", 2560, 1440],
  ["1440 × 2560 · Portrait extrapolation", 1440, 2560],
  ["3072 × 1296 · Scope 4 MP extrapolation", 3072, 1296],
];

let libraryDB;
function openLibrary() {
  libraryDB ||= new Promise((resolve, reject) => {
    const request = indexedDB.open("IAMCCS_MOSS_FrameDesigner", 1);
    request.onupgradeneeded = () => { request.result.createObjectStore("boards", { keyPath: "id" }); request.result.createObjectStore("renders", { keyPath: "id" }); };
    request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error);
  });
  return libraryDB;
}
async function libraryRows(store) {
  const db = await openLibrary();
  return new Promise((resolve, reject) => { const request = db.transaction(store).objectStore(store).getAll(); request.onsuccess = () => resolve(request.result.sort((a, b) => b.time - a.time)); request.onerror = () => reject(request.error); });
}
async function libraryPut(store, value) {
  const db = await openLibrary();
  await new Promise((resolve, reject) => { const tx = db.transaction(store, "readwrite"); tx.objectStore(store).put(value); tx.oncomplete = resolve; tx.onerror = () => reject(tx.error); });
  if (store === "renders") {
    const rows = await libraryRows(store);
    if (rows.length > 6) await new Promise((resolve, reject) => { const tx = db.transaction(store, "readwrite"); for (const row of rows.slice(6)) tx.objectStore(store).delete(row.id); tx.oncomplete = resolve; tx.onerror = () => reject(tx.error); });
  }
}

function normalize(raw) {
  if (raw?.format === "IAMCCS.Mboard") {
    if (raw.edition !== "MOSS" || raw.version !== 1) throw new Error("Unsupported Mboard edition or version.");
    raw = { ...raw.design, project_metadata: { ...raw.design.project_metadata, workflow: raw.workflow || raw.design.project_metadata?.workflow } };
  }
  if (!raw || typeof raw !== "object" || !raw.canvas || !Array.isArray(raw.items)) {
    throw new Error("Import a Frame Designer V2 or MOSS project with canvas, scene and items.");
  }
  const data = cloneProjectData(raw);
  data.schema = SCHEMA;
  data.schema_version = 1;
  data.project_id ||= crypto.randomUUID();
  data.scene ||= {};
  data.items = data.items.map((item, i) => ({
    ...item,
    id: item.id || `region_${i + 1}`,
    kind: item.kind || "obj",
    role: item.kind === "text" ? "text" : item.role || "object",
    label: item.label || `Region ${i + 1}`,
    desc: item.desc || "",
    text: item.text || "",
    style: item.style || "",
    text_mode: item.text_mode || "native",
    color_palette: item.color_palette || [],
  }));
  return data;
}

function imagePath(file) {
  return `/view?${new URLSearchParams({ filename: file.filename, subfolder: file.subfolder || "", type: file.type || "output" })}`;
}

async function imageDataURL(response) {
  if (!response.ok) throw new Error(`Saved image could not be read (${response.status}).`);
  const blob = await response.blob();
  return new Promise((resolve, reject) => {
    const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.onerror = reject; reader.readAsDataURL(blob);
  });
}

function install(node) {
  if (node._iamccsMossEditor) return;
  const storage = node.widgets?.find(w => w.name === "design_data");
  if (!storage) return;
  node._iamccsMossEditor = true;
  storage.type = "converted-widget";
  storage.computeSize = () => [0, -4];
  if (storage.inputEl) storage.inputEl.style.display = "none";

  const root = document.createElement("div");
  root.className = "iamccs-moss-editor";
  root.dataset.editorVersion = "R4.1";
  const style = document.createElement("style");
  style.textContent = `
    .iamccs-moss-editor{box-sizing:border-box;background:#15121d;color:#eee;font:13px/1.4 system-ui,sans-serif;width:100%;height:100%;min-width:0;overflow:hidden;display:flex;flex-direction:column;border:1px solid #61417a;border-radius:8px}
    .iamccs-moss-editor *{box-sizing:border-box}.iamccs-moss-editor header{padding:14px;background:#542579;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
    .iamccs-moss-editor h2{font:700 22px 'Courier New',monospace;color:white;margin:0 18px 0 0}.iamccs-moss-editor button,.iamccs-moss-editor select{background:#33243f;color:#fff;border:1px solid #76558b;border-radius:5px;padding:6px 9px;cursor:pointer}
    .iamccs-moss-editor .moss-body{display:grid;grid-template-columns:240px 1fr 260px;gap:12px;padding:12px;flex:1;min-height:0}.iamccs-moss-editor .moss-body>section{min-height:0;overflow:auto}.iamccs-moss-editor header,.iamccs-moss-editor>details,.iamccs-moss-editor .moss-status{flex-shrink:0}.iamccs-moss-editor label{display:block;color:#d8c5e9;margin:9px 0 3px}
    .iamccs-moss-editor .moss-body{grid-template-columns:240px minmax(0,1fr) 260px}.iamccs-moss-editor .moss-body>section{min-width:0;overflow-x:hidden;padding-right:4px}.iamccs-moss-editor select{min-width:0;max-width:100%}.iamccs-moss-editor .moss-grid{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}.iamccs-moss-editor .moss-h3{padding:10px 0;margin:8px 0;border-top:1px solid #4d3a5b;border-bottom:1px solid #4d3a5b}.iamccs-moss-editor .moss-h3-controls{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:8px;margin-top:10px}.iamccs-moss-editor .moss-h3-controls select:last-child{grid-column:1/-1}.iamccs-moss-editor .moss-h3 summary{cursor:pointer}.iamccs-moss-editor .moss-tools{column-gap:8px;row-gap:8px}.iamccs-moss-editor .moss-resolution{display:block;width:100%;text-overflow:ellipsis}
    .iamccs-moss-editor input,.iamccs-moss-editor textarea{width:100%;background:#201a29;color:#fff;border:1px solid #574264;border-radius:4px;padding:6px;font:12px/1.4 system-ui}
    .iamccs-moss-editor textarea{min-height:64px;resize:vertical}.iamccs-moss-editor .moss-stage{background:#08070c;border:1px solid #4d3a5b;min-height:0;flex:1;overflow:hidden;display:flex;align-items:center;justify-content:center}
    .iamccs-moss-editor canvas{display:block;width:100%;touch-action:none;cursor:crosshair}.iamccs-moss-editor .moss-surface{position:relative;width:100%}.iamccs-moss-editor .moss-box-fields{position:absolute;inset:0;pointer-events:none;overflow:hidden}.iamccs-moss-editor .moss-box-editor{position:absolute;pointer-events:none;padding:30px 5px 16px;display:flex;flex-direction:column;gap:3px;overflow:hidden}.iamccs-moss-editor .moss-box-editor textarea{pointer-events:auto;min-height:25px;height:100%;resize:none;background:#161020b8;border-color:#ffffff55;font-size:12px}.iamccs-moss-editor .moss-box-editor textarea:focus{background:#1b1228f2;outline:1px solid #ffe5a4}.iamccs-moss-editor .moss-box-editor .moss-literal{height:40%;color:#f0d6ff}.iamccs-moss-editor .moss-file{position:relative}.iamccs-moss-editor .moss-file-menu{position:absolute;left:0;top:100%;z-index:5;background:#23192f;border:1px solid #8d64a8;padding:6px;min-width:190px;box-shadow:0 8px 24px #0009}.iamccs-moss-editor .moss-file-menu button{display:block;width:100%;text-align:left;margin:3px 0}.iamccs-moss-editor .moss-tools{display:flex;gap:5px;flex-wrap:wrap;margin-bottom:8px}
    .iamccs-moss-editor .moss-regions{display:flex;flex-direction:column;gap:5px;margin-top:10px}.iamccs-moss-editor .moss-regions button{text-align:left;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    .iamccs-moss-editor .moss-regions .selected{border-color:#f9d47f;background:#4a3548}.iamccs-moss-editor .moss-grid{display:grid;grid-template-columns:1fr 1fr;gap:7px}
    .iamccs-moss-editor .moss-note{font-size:11px;color:#bcb1c5;margin:9px 0}.iamccs-moss-editor .moss-status{padding:8px 14px;border-top:1px solid #4d3a5b;white-space:pre-wrap}
    .iamccs-moss-editor details{padding:0 12px 12px}.iamccs-moss-editor pre{white-space:pre-wrap;word-break:break-word;background:#0f0c14;padding:10px;max-height:240px;overflow:auto;font-size:11px}
    .iamccs-moss-editor.moss-fullscreen{position:fixed;inset:18px;width:auto!important;height:auto!important;z-index:100000;box-shadow:0 0 0 30px #000b}
    .iamccs-moss-editor .moss-body>.moss-center{display:flex;flex-direction:column;overflow:hidden;height:100%}.iamccs-moss-editor .moss-center>details{padding:0;margin:3px 0;flex-shrink:0}.iamccs-moss-editor .moss-center>.moss-note{margin:4px 0;flex-shrink:0}.iamccs-moss-editor .moss-center>.moss-boards-title{margin:5px 0}.iamccs-moss-editor .moss-boards-list{height:100px;max-height:100px;flex-shrink:0;overflow-y:auto;display:flex;flex-direction:column;gap:5px}.iamccs-moss-editor .moss-boards-list button{text-align:left;white-space:normal}.iamccs-moss-editor .moss-gallery{display:grid;grid-template-columns:1fr 1fr;gap:6px}.iamccs-moss-editor .moss-gallery img{width:100%;height:74px;object-fit:contain;background:#09070d}.iamccs-moss-editor .moss-gallery button{padding:3px;font-size:10px}.iamccs-moss-editor .moss-boards-title{font:700 15px 'Courier New',monospace;margin:14px 0 7px}.moss-image-view{position:fixed;inset:0;background:#050408;z-index:200000;display:flex;align-items:center;justify-content:center}.moss-image-view img{width:100%;height:100%;object-fit:contain}.moss-image-view button{position:absolute;right:20px;top:20px;background:#271c36dd;color:white;border:1px solid #9977b4;padding:10px;border-radius:5px;cursor:pointer}
    .iamccs-moss-editor .moss-save-dialog{width:min(720px,90vw);max-height:85vh;background:#21172d;color:#eee;border:1px solid #9470ae;border-radius:8px;padding:22px}.iamccs-moss-editor .moss-save-dialog::backdrop{background:#000b}.iamccs-moss-editor .moss-folder-list{height:280px;overflow:auto;display:flex;flex-direction:column;gap:4px;margin:10px 0}.iamccs-moss-editor .moss-folder-list button{text-align:left}.iamccs-moss-editor .moss-save-dialog input[type=checkbox]{width:auto}
    .iamccs-moss-editor .moss-body .moss-grid{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}.iamccs-moss-editor .moss-body .moss-tools{gap:8px}.iamccs-moss-editor .moss-body .moss-grid>div{min-width:0}
    .iamccs-moss-editor .moss-resolution{width:174px;max-width:100%;font-size:12px}
    .iamccs-moss-editor .moss-preset{display:block;width:100%;font-size:12px;text-overflow:ellipsis}
    .iamccs-moss-editor .moss-regions-title{font:700 13px system-ui;margin:0 0 7px}.iamccs-moss-editor .moss-regions-frame{display:block;width:100%;height:120px;border:1px solid #574264;border-radius:4px;margin-bottom:10px}
    .iamccs-moss-editor header:after{content:'R4.1';color:#d7c4e4;font:11px monospace;margin-left:auto}
    .iamccs-moss-editor .moss-mboard{min-width:0;border:1px solid #574264;border-radius:5px;padding:9px;margin-bottom:12px}
    .iamccs-moss-editor .moss-mboard h3{font:700 14px 'Courier New',monospace;margin:0 0 7px}
    .iamccs-moss-editor .moss-mboard select{width:100%;font-size:12px;text-overflow:ellipsis}
    .iamccs-moss-editor .moss-mboard p{font-size:11px;color:#bcb1c5;margin:7px 0 0}
    dialog.moss-image-view{margin:auto;width:96vw;height:94vh;max-width:none;max-height:none;padding:0;border:1px solid #76558b;overflow:hidden}
    dialog.moss-image-view::backdrop{background:#000d}dialog.moss-image-view:fullscreen{width:100vw;height:100vh;border:0}
    .moss-image-surface{position:relative;display:flex;align-items:center;justify-content:center;width:100%;height:100%;background:#050408}
    dialog.moss-image-view.moss-full-monitor{inset:0;margin:0;width:100vw;height:100vh;border:0}
  `;
  root.append(style);
  const header = document.createElement("header");
  const title = document.createElement("h2"); title.textContent = "IAMCCS FRAME DESIGNER · MOSS EDITION"; title.title = "MOSS editor R4.1";
  const fileMenu = document.createElement("div"); fileMenu.className = "moss-file";
  const fileActions = document.createElement("div"); fileActions.className = "moss-file-menu"; fileActions.hidden = true;
  button(fileMenu, "File ▾", () => fileActions.hidden = !fileActions.hidden).setAttribute("aria-expanded", "false");
  fileMenu.append(fileActions); header.append(fileMenu, title); root.append(header);
  const body = document.createElement("div"); body.className = "moss-body"; root.append(body);
  const scenePane = document.createElement("section");
  const center = document.createElement("section"); center.className = "moss-center";
  const inspectorPane = document.createElement("section");
  const inspector = document.createElement("div");
  body.append(scenePane, center, inspectorPane);
  const tools = document.createElement("div"); tools.className = "moss-tools"; center.append(tools);
  const stage = document.createElement("div"); stage.className = "moss-stage";
  const surface = document.createElement("div"); surface.className = "moss-surface";
  const canvas = document.createElement("canvas"); canvas.setAttribute("aria-label", "MOSS frame canvas");
  const boxFields = document.createElement("div"); boxFields.className = "moss-box-fields";
  surface.append(canvas, boxFields); stage.append(surface); center.append(stage);
  const hint = document.createElement("p"); hint.className = "moss-note";
  hint.textContent = "Type prompts directly inside the boxes. Drag a border to move; drag the lower-right handle to resize. The latest generated image appears behind the boxes. Layout guides a new image; it does not lock pixels or identity.";
  center.append(hint);
  const regionTitle = document.createElement("h3"); regionTitle.className = "moss-regions-title"; regionTitle.textContent = "Regions · box selection";
  const regionsFrame = document.createElement("iframe"); regionsFrame.title = "Regions · box selection";
  regionsFrame.className = "moss-regions-frame"; regionsFrame.setAttribute("sandbox", "allow-same-origin");
  regionsFrame.srcdoc = `<!doctype html><html lang="en"><meta charset="utf-8"><style>html,body{margin:0;background:#15121d;color:#ddd;font:12px/1.4 system-ui;overflow-x:hidden}main{display:flex;flex-direction:column;gap:5px;padding:3px}button{box-sizing:border-box;width:100%;text-align:left;overflow-wrap:anywhere;background:#33243f;color:white;border:1px solid #76558b;border-radius:4px;padding:7px;cursor:pointer}.selected{border-color:#f9d47f;background:#4a3548}</style><main aria-label="Regions"></main></html>`;
  let list;
  regionsFrame.addEventListener("load", () => { list = regionsFrame.contentDocument?.querySelector("main"); renderList(); });
  const mboardPanel = document.createElement("div"); mboardPanel.className = "moss-mboard";
  const mboardTitle = document.createElement("h3"); mboardTitle.textContent = "Mboard";
  const mboardSelect = document.createElement("select"); mboardSelect.setAttribute("aria-label", "Imported Mboards");
  const mboardInfo = document.createElement("p");
  mboardPanel.append(mboardTitle, mboardSelect, mboardInfo);
  inspectorPane.append(mboardPanel, regionTitle, regionsFrame, inspector);
  const details = document.createElement("details");
  const summary = document.createElement("summary"); summary.textContent = "Compiled MOSS layout JSON";
  const output = document.createElement("pre"); details.append(summary, output); root.append(details);
  const guide = document.createElement("details");
  const guideTitle = document.createElement("summary"); guideTitle.textContent = "MOSS photographic prompting tips";
  const guideText = document.createElement("p"); guideText.className = "moss-note";
  guideText.textContent = "Write the finished visible photograph in English: subject count, action, setting, framing, actual light direction and material appearance. Put the overall scene in the scene fields and each region's content in its description. Use exact wording for text regions. There is no minimum word count. Avoid redundant quality slogans and conflicting lighting or focus instructions. Layout JSON goes directly to the encoder; a free prose rewrite can discard boxes. Native layout is experimental and does not guarantee photorealism.";
  const guideLink = document.createElement("a"); guideLink.href = "https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md";
  guideLink.textContent = "OpenMOSS User Guide"; guideLink.target = "_blank"; guideLink.rel = "noopener noreferrer"; guideLink.style.color = "#d6b9f0";
  guide.append(guideTitle, guideText, guideLink); root.append(guide);
  const status = document.createElement("div"); status.className = "moss-status"; root.append(status);
  const file = document.createElement("input"); file.type = "file"; file.accept = ".Mboard,.mboard,.json,application/json"; file.hidden = true; root.append(file);
  const host = document.createElement("div");
  host.style.width = `${EDITOR_WIDTH}px`; host.style.height = `${EDITOR_HEIGHT}px`; host.append(root);
  let data = normalize(JSON.parse(storage.value));
  let selected = data.items[0]?.id;
  let drag = null;
  let importedBoards = cloneProjectData(node.properties?.mossImportedBoards || []);
  let activeBoardKey = node.properties?.mossActiveBoardKey || "";
  let renderedImage = null;
  let renderedSource = "";
  let showBoxes = node.properties?.mossShowBoxes !== false;
  let imageViewer = null;
  let awaitingRender = false;
  let latestRenderKey = "";
  let recentOpen = true;
  let h3Open = false;
  let recoveryTimer;
  let pendingImageKey = "";
  let lastLegacyPrompt;
  let dirtyFrame;
  const localProjectSave = ["127.0.0.1", "localhost", "[::1]"].includes(location.hostname);
  let lastSaveDirectory = "";

  function button(parent, text, fn) {
    const el = document.createElement("button"); el.textContent = text;
    el.addEventListener("click", event => { event.stopPropagation(); Promise.resolve(fn()).catch(error => message(error.message)); }); parent.append(el); return el;
  }
  function message(text) { status.textContent = text; }
  function persist() {
    const stored = { ...data };
    if (data.generated) { stored.generated = { ...data.generated }; delete stored.generated.data_url; }
    storage.value = JSON.stringify(stored);
    if (!dirtyFrame) dirtyFrame = requestAnimationFrame(() => {
      dirtyFrame = null;
      node.setDirtyCanvas(true, true);
      app.graph?.setDirtyCanvas(true, true);
    });
  }
  function current() { return data.items.find(i => i.id === selected); }
  function preview() {
    const width = +data.canvas.width, height = +data.canvas.height;
    return { summary: { caption: ["high_level_description", "background", "aesthetics", "lighting", "photo", "medium", "art_style"].map(k => data.scene[k]).filter(Boolean).join(" ") },
      items: data.items.filter(i => !["image", "mask"].includes(i.kind) && !(i.kind === "text" && i.text_mode === "overlay")).map(i => {
        const item = { label: i.kind === "text" ? "text" : i.role || "object", caption: [i.desc, i.style && `Rendering treatment for this region: ${i.style}`].filter(Boolean).join(" "),
          bbox: [Math.floor(i.y * height / 1000) - Math.floor(height / 2), Math.floor(i.x * width / 1000) - Math.floor(width / 2),
            Math.ceil(Math.min(1000, i.y + i.h) * height / 1000) - Math.floor(height / 2), Math.ceil(Math.min(1000, i.x + i.w) * width / 1000) - Math.floor(width / 2)] };
        if (item.label === "text") item.text = i.text || "";
        if (i.color_palette?.length) item.palette = i.color_palette;
        return item;
      }), bbox_coord_space: "abs_centralized", bbox_format: "yxyx" };
  }
  function draw() {
    if (stage.clientWidth && stage.clientHeight) surface.style.width = `${Math.min(stage.clientWidth, stage.clientHeight * data.canvas.width / data.canvas.height)}px`;
    canvas.width = 1000;
    canvas.height = Math.round(1000 * data.canvas.height / data.canvas.width);
    const c = canvas.getContext("2d"), H = canvas.height;
    c.fillStyle = "#17141f"; c.fillRect(0, 0, 1000, H);
    if (renderedImage) {
      const fit = Math.min(1000 / renderedImage.width, H / renderedImage.height);
      const w = renderedImage.width * fit, h = renderedImage.height * fit;
      c.drawImage(renderedImage, (1000 - w) / 2, (H - h) / 2, w, h);
    }
    c.save(); c.globalAlpha = !showBoxes && renderedImage ? 0 : renderedImage ? 0.12 : 1;
    c.strokeStyle = "#322b3e"; c.lineWidth = 1;
    for (let t = 0; t <= 1000; t += 100) {
      c.beginPath(); c.moveTo(t, 0); c.lineTo(t, H); c.stroke();
      c.beginPath(); c.moveTo(0, t * H / 1000); c.lineTo(1000, t * H / 1000); c.stroke();
    }
    c.restore();
    if (showBoxes) data.items.forEach((i, index) => {
      const role = i.kind === "text" ? "text" : i.role || "object";
      const color = COLORS[role] || "#888";
      const x = i.x, y = i.y * H / 1000, w = i.w, h = i.h * H / 1000;
      c.fillStyle = color + "22"; c.fillRect(x, y, w, h);
      c.strokeStyle = i.id === selected ? "#fff2bd" : color; c.lineWidth = i.id === selected ? 4 : 2; c.strokeRect(x, y, w, h);
      c.fillStyle = color; c.fillRect(x, y, Math.min(w, 330), 30);
      c.fillStyle = "#100d16"; c.font = "bold 18px sans-serif"; c.fillText(`${index + 1}. ${i.label}`, x + 6, y + 22, Math.max(1, w - 10));
      if (i.id === selected) { c.fillStyle = "#fff2bd"; c.fillRect(x + w - 14, y + h - 14, 14, 14); }
    });
    syncBoxFields();
    output.textContent = JSON.stringify(preview(), null, 2);
  }
  function syncBoxFields() {
    boxFields.hidden = !showBoxes;
    const ids = new Set(data.items.map(item => item.id));
    for (const el of [...boxFields.children]) if (!ids.has(el.dataset.id)) el.remove();
    for (const item of data.items) {
      let box = [...boxFields.children].find(el => el.dataset.id === item.id);
      if (!box) { box = document.createElement("div"); box.className = "moss-box-editor"; box.dataset.id = item.id; boxFields.append(box); }
      Object.assign(box.style, { left: `${item.x / 10}%`, top: `${item.y / 10}%`, width: `${item.w / 10}%`, height: `${item.h / 10}%` });
      const keys = item.kind === "text" ? ["text", "desc"] : ["desc"];
      if (box.children.length !== keys.length) box.replaceChildren();
      keys.forEach((key, index) => {
        let input = box.children[index];
        if (!input) {
          input = document.createElement("textarea"); input.dataset.key = key;
          input.addEventListener("focus", () => {
            if (selected !== item.id) { selected = item.id; renderList(); renderInspector(); }
          });
          input.addEventListener("input", () => {
            const target = data.items.find(region => region.id === box.dataset.id);
            if (!target) return;
            target[key] = input.value; persist(); updatePromptPreview();
            const mirror = inspector.querySelector(`[data-region-key="${key}"]`);
            if (selected === item.id && mirror && document.activeElement !== mirror) mirror.value = input.value;
          });
          input.addEventListener("pointerdown", event => event.stopPropagation()); box.append(input);
        }
        input.className = key === "text" ? "moss-literal" : "";
        input.setAttribute("aria-label", `${key === "text" ? "Visible text" : "Box prompt"}: ${item.label}`);
        input.placeholder = key === "text" ? "Exact words to render…" : "Describe this region…";
        if (document.activeElement !== input) input.value = item[key] || "";
      });
    }
  }
  function loadRenderedImage() {
    if (awaitingRender) { renderedImage = null; draw(); return; }
    const file = data.generated?.files?.[0];
    const source = data.generated?.data_url || (file ? api.apiURL(imagePath(file)) : "");
    if (source === renderedSource) return;
    renderedSource = source; renderedImage = null;
    if (!source) { draw(); return; }
    const image = new Image();
    image.onload = () => { if (renderedSource === source) { renderedImage = image; draw(); } };
    image.onerror = () => message("The embedded generated image could not be opened.");
    image.src = source;
  }
  function field(parent, label, value, change, type = "text") {
    const elLabel = document.createElement("label"); elLabel.textContent = label; parent.append(elLabel);
    const input = document.createElement(type === "textarea" ? "textarea" : "input");
    input.setAttribute("aria-label", label);
    if (type !== "textarea") input.type = type;
    input.value = value ?? "";
    input.addEventListener("input", () => {
      change(input.value); persist();
      // Prose editing does not rebuild the inspector or repaint the image canvas.
      if (type === "number" || label === "Region name") draw();
      else { syncBoxFields(); updatePromptPreview(); }
    });
    parent.append(input); return input;
  }
  function updatePromptPreview() { output.textContent = JSON.stringify(preview(), null, 2); }
  function preset(parent, label, entries, applyPreset) {
    const heading = document.createElement("label"); heading.textContent = label; parent.append(heading);
    const select = document.createElement("select"); select.className = "moss-preset"; select.setAttribute("aria-label", label);
    select.append(new Option("Choose preset…", ""));
    entries.forEach(([name], index) => select.append(new Option(name, String(index))));
    select.addEventListener("change", () => {
      if (select.value === "") return;
      const [name, text] = entries[+select.value]; applyPreset(text); persist(); draw();
      message(`${name} applied. The field remains editable.`);
    });
    parent.append(select); return select;
  }
  function renderInspector() {
    inspector.replaceChildren();
    const item = current();
    if (!item) { inspector.textContent = "Select a region or add one."; return; }
    field(inspector, "Region name", item.label, v => { item.label = v; renderList(); });
    const label = document.createElement("label"); label.textContent = "Region type"; inspector.append(label);
    const role = document.createElement("select");
    role.setAttribute("aria-label", "Region type");
    for (const value of ["character", "object", "text"]) { const option = new Option(value, value); role.append(option); }
    role.value = item.kind === "text" ? "text" : item.role;
    role.addEventListener("change", () => { item.role = role.value; item.kind = role.value === "text" ? "text" : "obj"; persist(); renderInspector(); draw(); }); inspector.append(role);
    field(inspector, "Region description", item.desc, v => item.desc = v, "textarea").dataset.regionKey = "desc";
    preset(inspector, "Region style preset", STYLE_PRESETS, text => { item.style = text; renderInspector(); });
    field(inspector, "Region rendering treatment (optional)", item.style, v => item.style = v, "textarea");
    if (item.kind === "text") {
      field(inspector, "Exact visible text", item.text, v => item.text = v, "textarea").dataset.regionKey = "text";
      const label = document.createElement("label"); label.textContent = "Text rendering"; inspector.append(label);
      const mode = document.createElement("select"); mode.setAttribute("aria-label", "Text rendering");
      mode.append(new Option("MOSS native (experimental)", "native"), new Option("Exact overlay · MOSS Result", "overlay"));
      mode.value = item.text_mode;
      mode.onchange = () => { item.text_mode = mode.value; persist(); draw(); renderInspector(); };
      inspector.append(mode);
      if (item.text_mode === "overlay") {
        field(inspector, "Text size in pixels (0 = auto)", item.font_size || 0, v => item.font_size = Math.max(0, +v || 0), "number");
        const note = document.createElement("p"); note.className = "moss-note";
        note.textContent = "Exact lettering is composited after diffusion by MOSS Result. It is not generated by MOSS. White text with a dark outline; size fits the box."; inspector.append(note);
      }
    }
    field(inspector, "Palette (hex colors, comma-separated)", item.color_palette.join(", "), v => item.color_palette = v.split(",").map(s => s.trim()).filter(Boolean));
    const dims = document.createElement("div"); dims.className = "moss-grid"; inspector.append(dims);
    for (const [key, text] of [["x", "Left"], ["y", "Top"], ["w", "Width"], ["h", "Height"]]) {
      const cell = document.createElement("div"); dims.append(cell);
      field(cell, text + " / 1000", item[key], v => {
        const number = Number(v); if (!Number.isFinite(number)) return;
        item[key] = Math.max(key === "w" || key === "h" ? 1 : 0, Math.min(1000, number));
      }, "number");
    }
    const actions = document.createElement("div"); actions.className = "moss-tools"; inspector.append(actions);
    button(actions, "Duplicate", () => { const copy = cloneProjectData(item); copy.id = crypto.randomUUID(); copy.label += " copy"; data.items.push(copy); selected = copy.id; persist(); renderList(); renderInspector(); draw(); });
    button(actions, "Delete", () => { data.items = data.items.filter(i => i.id !== selected); selected = data.items[0]?.id; persist(); renderList(); renderInspector(); draw(); });
    for (const [text, delta] of [["Earlier", -1], ["Later", 1]]) button(actions, text, () => {
      const index = data.items.indexOf(item), other = index + delta;
      if (other < 0 || other >= data.items.length) return;
      [data.items[index], data.items[other]] = [data.items[other], data.items[index]];
      persist(); renderList(); draw();
    });
    const recent = document.createElement("details"); recent.open = recentOpen;
    recent.addEventListener("toggle", () => recentOpen = recent.open);
    const recentTitle = document.createElement("summary"); recentTitle.textContent = "Recent renders · 6";
    const gallery = document.createElement("div"); gallery.className = "moss-gallery"; gallery.setAttribute("aria-label", "Six most recent MOSS renders");
    recent.append(recentTitle, gallery); inspector.append(recent); refreshRecent(gallery).catch(error => message(error.message));
  }
  async function refreshRecent(gallery) {
    const rows = await libraryRows("renders");
    gallery.replaceChildren();
    if (!rows.length) { gallery.textContent = "Generated MOSS frames will appear here."; return; }
    for (const row of rows) {
      const entry = button(gallery, `${row.width} × ${row.height}`, () => viewImage(row.data_url));
      const image = document.createElement("img"); image.src = row.data_url; image.alt = `MOSS render ${row.width} × ${row.height}`; entry.prepend(image);
    }
  }
  async function rememberBoard(project, name) {
    await libraryPut("boards", { id: `${name}|${project.design.project_id}`, name, time: Date.now(), project: cloneProjectData(project) });
  }
  function viewImage(source = renderedSource, fullMonitor = false) {
    if (!source) { message("Generate or recall an image first."); return; }
    imageViewer?.close();
    const viewer = document.createElement("dialog"); viewer.className = "moss-image-view"; viewer.setAttribute("aria-label", "MOSS image preview");
    viewer.classList.toggle("moss-full-monitor", fullMonitor);
    const surface = document.createElement("div"); surface.className = "moss-image-surface"; viewer.append(surface);
    const image = document.createElement("img"); image.src = source; image.alt = "MOSS generated frame · full-screen preview"; surface.append(image);
    const close = () => { if (document.fullscreenElement === surface) document.exitFullscreen().catch(() => {}); viewer.close(); };
    viewer.addEventListener("close", () => { viewer.remove(); if (imageViewer === viewer) imageViewer = null; }, { once: true });
    button(surface, "Close image · Esc", close);
    viewer.addEventListener("keydown", event => { event.stopPropagation(); if (event.key === "Escape") close(); });
    document.body.append(viewer); imageViewer = viewer; viewer.showModal();
    if (fullMonitor) surface.requestFullscreen?.({ navigationUI: "hide" }).catch(() => message("Image view fills the browser window. Press Esc or Close image to return."));
  }
  function renderList() {
    if (!list) return;
    list.replaceChildren();
    data.items.forEach((i, index) => {
      const b = button(list, `${index + 1}. ${i.label}`, () => { selected = i.id; renderList(); renderInspector(); draw(); });
      if (i.id === selected) b.className = "selected";
    });
  }
  function renderScene() {
    scenePane.replaceChildren();
    const resolutionLabel = document.createElement("label"); resolutionLabel.textContent = "Output resolution"; scenePane.append(resolutionLabel);
    const resolution = document.createElement("select"); resolution.className = "moss-resolution"; resolution.setAttribute("aria-label", "Output resolution");
    RESOLUTIONS.forEach(([, w, h]) => resolution.append(new Option(`${w} × ${h}`, `${w}x${h}`)));
    resolution.append(new Option("Custom…", "custom"));
    const match = `${data.canvas.width}x${data.canvas.height}`;
    resolution.value = data.canvas.resolution_mode !== "custom" && RESOLUTIONS.some(([, w, h]) => `${w}x${h}` === match) ? match : "custom";
    resolution.addEventListener("change", () => {
      if (resolution.value !== "custom") [data.canvas.width, data.canvas.height] = resolution.value.split("x").map(Number);
      data.canvas.resolution_mode = resolution.value; persist(); renderScene(); draw();
    }); scenePane.append(resolution);
    const size = document.createElement("div"); size.className = "moss-grid"; scenePane.append(size);
    for (const key of ["width", "height"]) {
      const cell = document.createElement("div"); size.append(cell);
      const input = field(cell, key === "width" ? "Canvas width" : "Canvas height", data.canvas[key], v => {
        if (+v > 0) data.canvas[key] = +v;
        resolution.value = "custom"; data.canvas.resolution_mode = "custom";
      }, "number"); input.step = "16"; input.min = "16";
    }
    const resolutionNote = document.createElement("p"); resolutionNote.className = "moss-note";
    resolutionNote.textContent = "MOSS: 1–2 MP recommended; up to ~4 MP extrapolation. Use multiples of 16. Custom above this range is experimental and requires more memory.";
    scenePane.append(resolutionNote);
    data.h3 ||= { slot: 1, role: "composition" };
    const h3 = document.createElement("details"); h3.className = "moss-h3"; h3.open = h3Open;
    h3.addEventListener("toggle", () => h3Open = h3.open);
    const h3Label = document.createElement("summary"); h3Label.textContent = "H3 / cine_linx export";
    const h3Controls = document.createElement("div"); h3Controls.className = "moss-h3-controls"; h3.append(h3Label, h3Controls); scenePane.append(h3);
    const slot = document.createElement("select"); slot.setAttribute("aria-label", "H3 picture slot");
    for (let n = 1; n <= 4; n++) slot.append(new Option(`Picture ${n}`, n));
    slot.value = data.h3.slot; slot.onchange = () => { data.h3.slot = +slot.value; persist(); }; h3Controls.append(slot);
    const h3Role = document.createElement("select"); h3Role.setAttribute("aria-label", "H3 reference role");
    for (const role of ["composition", "subject_identity", "style", "keyframe"]) h3Role.append(new Option(role, role));
    h3Role.value = data.h3.role; h3Role.onchange = () => { data.h3.role = h3Role.value; persist(); }; h3Controls.append(h3Role);
    const target = document.createElement("select"); target.setAttribute("aria-label", "H3 export target");
    for (const [value, label] of [["pictures", "H3 reference pictures"], ["swap_subject", "R49 SwapChar · subject"], ["swap_environment", "R49 SwapChar · environment"]]) target.append(new Option(label, value));
    target.value = data.h3.target || "pictures"; target.onchange = () => { data.h3.target = target.value; persist(); }; h3Controls.append(target);
    for (const [key, text] of [["high_level_description", "Scene / global caption"], ["background", "Setting / background"], ["lighting", "Lighting"], ["photo", "Camera / photographic treatment"], ["aesthetics", "Aesthetics"]]) {
      const entries = key === "lighting" ? LIGHT_PRESETS : key === "photo" ? CAMERA_PRESETS : key === "aesthetics" ? STYLE_PRESETS : null;
      if (entries) preset(scenePane, key === "lighting" ? "Lighting preset" : key === "photo" ? "Camera preset" : "Style preset", entries, value => { data.scene[key] = value; renderScene(); });
      field(scenePane, text, data.scene[key], v => data.scene[key] = v, "textarea");
    }
  }
  function refresh() { renderScene(); renderList(); renderInspector(); loadRenderedImage(); draw(); }
  function storeImportedBoards() {
    node.properties ||= {};
    node.properties.mossImportedBoards = importedBoards;
    node.properties.mossActiveBoardKey = activeBoardKey;
  }
  function renderMboards() {
    mboardSelect.replaceChildren(new Option(importedBoards.length ? "Choose imported Mboard…" : "Import an Mboard JSON pack…", ""));
    for (const entry of importedBoards) mboardSelect.append(new Option(entry.name, entry.key));
    mboardSelect.value = activeBoardKey;
    mboardSelect.disabled = importedBoards.length === 0;
    mboardInfo.textContent = importedBoards.length ? `${importedBoards.length} imported Mboards · File → Import Mboard…` : "File → Import Mboard… accepts a JSON pack or a single .Mboard.";
  }
  mboardSelect.addEventListener("change", () => {
    const entry = importedBoards.find(board => board.key === mboardSelect.value);
    if (!entry) return;
    const previous = importedBoards.find(board => board.key === activeBoardKey);
    if (previous) previous.design = portableDesign(data);
    activeBoardKey = entry.key; storeImportedBoards(); apply(entry.design);
    message(`Mboard loaded: ${entry.name}`);
  });
  function apply(raw) { data = normalize(raw); awaitingRender = false; selected = data.items[0]?.id; persist(); refresh(); message("Project loaded. Image and mask layers are editor data only in Layout mode."); }
  function download(name, text) {
    const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
    const a = document.createElement("a"); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function chooseProjectFile(project) {
    return new Promise((resolve, reject) => {
      const dialog = document.createElement("dialog"); dialog.className = "moss-save-dialog";
      dialog.setAttribute("aria-label", "Save Mboard project as");
      const heading = document.createElement("h2"); heading.textContent = "Save Mboard project as"; dialog.append(heading);
      const address = field(dialog, "Save folder", lastSaveDirectory, () => {});
      const nav = document.createElement("div"); nav.className = "moss-tools"; dialog.append(nav);
      const folders = document.createElement("div"); folders.className = "moss-folder-list"; folders.setAttribute("aria-label", "Project folders"); dialog.append(folders);
      const filename = field(dialog, "Mboard filename", "IAMCCS_MOSS_FRAME.Mboard", () => {});
      const replaceLabel = document.createElement("label"); const replace = document.createElement("input"); replace.type = "checkbox"; replace.setAttribute("aria-label", "Replace existing file"); replaceLabel.append(replace, " Replace existing file"); dialog.append(replaceLabel);
      const feedback = document.createElement("p"); dialog.append(feedback);
      const actions = document.createElement("div"); actions.className = "moss-tools"; dialog.append(actions);
      let parent = "";
      async function browse(path) {
        const response = await api.fetchApi(`/iamccs/moss/project-folders?${new URLSearchParams({ path })}`);
        const result = await response.json();
        if (!response.ok) { feedback.textContent = result.error; return; }
        address.value = result.path; parent = result.parent; folders.replaceChildren();
        for (const folder of result.folders) button(folders, `▸ ${folder.name}`, () => browse(folder.path));
        feedback.textContent = result.path || "Choose a drive, then a destination folder.";
      }
      button(nav, "Drives / roots", () => browse("")); button(nav, "Parent folder", () => browse(parent)); button(nav, "Open folder", () => browse(address.value));
      address.addEventListener("keydown", event => { if (event.key === "Enter") { event.preventDefault(); browse(address.value).catch(error => feedback.textContent = error.message); } });
      const close = value => { dialog.close(); dialog.remove(); resolve(value); };
      button(actions, "Cancel", () => close(null));
      const save = button(actions, "Save Mboard", async () => {
        save.disabled = true;
        try {
          const response = await api.fetchApi("/iamccs/moss/save-mboard", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ directory: address.value, name: filename.value, overwrite: replace.checked, project }) });
          const result = await response.json();
          if (!response.ok) { feedback.textContent = result.error; return; }
          lastSaveDirectory = address.value; close({ path: result.saved, name: filename.value });
        } catch (error) { feedback.textContent = error.message; }
        finally { save.disabled = false; }
      });
      dialog.addEventListener("cancel", event => { event.preventDefault(); close(null); });
      root.append(dialog); dialog.showModal(); browse(lastSaveDirectory).catch(error => { dialog.remove(); reject(error); });
    });
  }
  async function saveProjectAs() {
    message("Preparing portable Mboard project…");
    fileActions.hidden = true;
    const picker = !localProjectSave && typeof window.showSaveFilePicker === "function" ? window : null;
    let handle = null;
    if (picker) {
      try { handle = await picker.showSaveFilePicker({ suggestedName: "IAMCCS_MOSS_FRAME.Mboard", types: [{ description: "IAMCCS MOSS Board", accept: { "application/json": [".Mboard"] } }] }); }
      catch (error) { if (error.name === "AbortError") { message("Save cancelled."); return; } throw error; }
    }
    const project = { format: "IAMCCS.Mboard", version: 1, edition: "MOSS", saved_at: new Date().toISOString(), design: cloneProjectData(data) };
    if (data.generated && !data.generated.data_url && renderedSource) {
      const response = await fetch(renderedSource);
      if (!response.ok) throw new Error("Cannot embed the generated image. Check its local output file.");
      project.design.generated.data_url = await new Promise((resolve, reject) => {
        const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.onerror = reject; response.blob().then(blob => reader.readAsDataURL(blob), reject);
      });
    }
    project.workflow = app.graph.serialize();
    for (const n of project.workflow.nodes || []) if (n.type === TYPE && n.widgets_values?.[0]) {
      const clean = JSON.parse(n.widgets_values[0]); delete clean.generated;
      if (clean.project_metadata) delete clean.project_metadata.workflow;
      n.widgets_values[0] = JSON.stringify(clean);
    }
    if (project.design.project_metadata) delete project.design.project_metadata.workflow;
    const json = JSON.stringify(project);
    if (handle) { const writable = await handle.createWritable(); await writable.write(json); await writable.close(); await rememberBoard(project, handle.name); message(`Saved ${handle.name} with prompts, boxes, resolution, image and metadata.`); }
    else if (localProjectSave) {
      const saved = await chooseProjectFile(project);
      if (saved) await rememberBoard(project, saved.name);
      message(saved ? `Saved ${saved.path} with embedded image and metadata.` : "Save cancelled.");
    }
    else { download("IAMCCS_MOSS_FRAME.Mboard", json); await rememberBoard(project, "IAMCCS_MOSS_FRAME.Mboard"); message("Mboard downloaded with embedded image and metadata. Enable your browser's 'Ask where to save' setting to choose a folder."); }
  }
  button(fileActions, "Save prj as…", saveProjectAs);
  button(fileActions, "Import Mboard…", () => { fileActions.hidden = true; file.click(); });
  button(fileActions, "Import legacy V2 / gallery…", () => { fileActions.hidden = true; file.click(); });
  button(fileActions, "Export MOSS prompt", () => { fileActions.hidden = true; download("MOSS_LAYOUT_PROMPT.json", JSON.stringify(preview(), null, 2)); });
  function fullscreen(enabled) {
    root.classList.toggle("moss-fullscreen", enabled);
    if (enabled) document.body.append(root);
    else host.append(root);
  }
  button(header, "Fullscreen", () => fullscreen(!root.classList.contains("moss-fullscreen")));
  const boxToggle = button(header, showBoxes ? "Image only" : "Edit boxes", () => {
    showBoxes = !showBoxes; node.properties ||= {}; node.properties.mossShowBoxes = showBoxes;
    boxToggle.textContent = showBoxes ? "Image only" : "Edit boxes";
    boxToggle.setAttribute("aria-pressed", String(!showBoxes)); draw();
  });
  boxToggle.setAttribute("aria-pressed", String(!showBoxes));
  button(header, "View image fullscreen", () => viewImage(renderedSource, true));
  button(header, "Recover last image", () => recoverSavedImage(true));
  button(header, "Connect result node", () => {
    const existing = app.graph._nodes.find(n => n.type === "IAMCCS_MOSSFrameResult" && n.inputs?.find(i => i.name === "project_data")?.link && app.graph.links[n.inputs.find(i => i.name === "project_data").link]?.origin_id === node.id);
    const result = existing || LiteGraph.createNode("IAMCCS_MOSSFrameResult");
    if (!existing) { result.pos = [node.pos[0] + node.size[0] + 80, node.pos[1] + 800]; app.graph.add(result); }
    for (const [name, type] of [["cine_linx", "IAMCCS_SUPERNODE_LINX"], ["project_data", "STRING"]]) if (!node.outputs.some(output => output.name === name)) node.addOutput(name, type);
    node.connect(node.outputs.findIndex(o => o.name === "project_data"), result, result.inputs.findIndex(i => i.name === "project_data"));
    const sink = app.graph._nodes.find(n => ["SaveImage", "PreviewImage"].includes(n.comfyClass || n.type) && upstreamDesigners(app.graph, n.id).some(designer => designer.id === node.id));
    const link = sink?.getInputLink(sink.inputs.findIndex(input => input.name === "images"));
    if (link) app.graph.getNodeById(link.origin_id).connect(link.origin_slot, result, result.inputs.findIndex(input => input.name === "images"));
    message(link ? "MOSS Result connected to the saved image and project. Its cine_linx exports this execution's image." : "Project connected. Connect the final decoded IMAGE to MOSS Result. Standard SaveImage also updates this canvas automatically.");
  });
  file.addEventListener("change", async () => {
    try {
      const raw = JSON.parse(await file.files[0].text());
      // Validate all boards before changing the current scene or the dropdown.
      importedMboards(raw, normalize, file.files[0].name);
      const incoming = raw.format === "IAMCCS.Mboard" ? raw.design : raw;
      if (incoming?.generated?.data_url) {
        const image = await fetch(incoming.generated.data_url).then(response => response.blob());
        const form = new FormData(); form.append("image", image, `Mboard_${crypto.randomUUID()}.png`);
        form.append("subfolder", "IAMCCS_MOSS_Mboard"); form.append("type", "input");
        const uploaded = await api.fetchApi("/upload/image", { method: "POST", body: form });
        if (!uploaded.ok) throw new Error("Could not restore the embedded image into ComfyUI input.");
        const saved = await uploaded.json();
        incoming.generated.files = [{ filename: saved.name, subfolder: saved.subfolder, type: saved.type }];
      }
      const collection = importedMboards(raw, normalize, file.files[0].name);
      if (collection) {
        importedBoards = collection; activeBoardKey = collection[0].key;
        storeImportedBoards(); renderMboards(); apply(collection[0].design);
        message(`Imported ${collection.length} Mboards. Choose a scene in Mboard above Regions.`);
      } else if (Array.isArray(raw.nodes)) {
        const original = raw.nodes.find(n => n.type === TYPE || n.type === "IAMCCS_StoryboardFrameDesignerV2");
        if (!original) throw new Error("No Frame Designer project found in this workflow.");
        apply(JSON.parse(original.widgets_values[0]));
      } else {
        apply(raw);
        if (raw.format === "IAMCCS.Mboard") await rememberBoard(raw, file.files[0].name);
      }
    } catch (error) { message(`Import failed: ${error.message}`); }
    finally { file.value = ""; }
  });
  for (const role of ["character", "object", "text"]) button(tools, `+ ${role}`, () => {
    const item = { id: crypto.randomUUID(), kind: role === "text" ? "text" : "obj", role,
      label: `${role} ${data.items.length + 1}`, x: 300, y: 200, w: 300, h: 500, desc: "", text: "", color_palette: [] };
    data.items.push(item); selected = item.id; persist(); renderList(); renderInspector(); draw();
  });
  function point(event) { const r = canvas.getBoundingClientRect(); return { x: (event.clientX - r.left) / r.width * 1000, y: (event.clientY - r.top) / r.height * 1000 }; }
  canvas.addEventListener("pointerdown", e => {
    if (!showBoxes) return;
    e.preventDefault(); e.stopPropagation(); const p = point(e);
    const item = [...data.items].reverse().find(i => p.x >= i.x && p.x <= i.x + i.w && p.y >= i.y && p.y <= i.y + i.h);
    if (!item) return;
    selected = item.id;
    const r = canvas.getBoundingClientRect();
    drag = { start: p, original: cloneProjectData(item), resize: (item.x + item.w - p.x) * r.width / 1000 < 16 && (item.y + item.h - p.y) * r.height / 1000 < 16 };
    canvas.setPointerCapture(e.pointerId); renderList(); renderInspector(); draw();
  });
  canvas.addEventListener("pointermove", e => {
    if (!drag) return; e.preventDefault(); const p = point(e), i = current(), o = drag.original;
    if (drag.resize) {
      i.w = Math.round(Math.max(10, Math.min(1000 - o.x, o.w + p.x - drag.start.x)));
      i.h = Math.round(Math.max(10, Math.min(1000 - o.y, o.h + p.y - drag.start.y)));
    } else {
      i.x = Math.round(Math.max(0, Math.min(1000 - o.w, o.x + p.x - drag.start.x)));
      i.y = Math.round(Math.max(0, Math.min(1000 - o.h, o.y + p.y - drag.start.y)));
    }
    persist(); draw();
  });
  canvas.addEventListener("pointerup", () => { drag = null; renderInspector(); });
  canvas.addEventListener("pointercancel", () => { drag = null; });
  root.addEventListener("pointerdown", e => e.stopPropagation());
  root.addEventListener("wheel", e => e.stopPropagation());
  root.addEventListener("keydown", e => { e.stopPropagation(); if (e.key === "Escape") fullscreen(false); });
  const widget = node.addDOMWidget("MOSS Frame Designer", "iamccs_moss_editor", host, {
    serialize: false, getMinHeight: () => EDITOR_HEIGHT,
    getHeight: () => EDITOR_HEIGHT,
    afterResize: () => { host.style.width = "100%"; host.style.height = "100%"; },
  });
  const stageResize = new ResizeObserver(() => draw()); stageResize.observe(stage);
  widget.computeSize = () => [EDITOR_WIDTH, EDITOR_HEIGHT];
  widget.computeLayoutSize = () => ({ minWidth: EDITOR_WIDTH, minHeight: EDITOR_HEIGHT, maxHeight: Infinity });
  const minimumHeight = node.computeSize()[1] + 20;
  const previousResize = node.onResize;
  node.onResize = function () {
    this.size[0] = Math.max(EDITOR_WIDTH + 40, this.size[0]); this.size[1] = Math.max(minimumHeight, this.size[1]);
    previousResize?.apply(this, arguments);
    host.style.width = "100%"; host.style.height = "100%";
  };
  const previousConfigure = node.onConfigure;
  node.onConfigure = function () {
    const result = previousConfigure?.apply(this, arguments);
    showBoxes = this.properties?.mossShowBoxes !== false;
    boxToggle.textContent = showBoxes ? "Image only" : "Edit boxes";
    boxToggle.setAttribute("aria-pressed", String(!showBoxes));
    importedBoards = cloneProjectData(this.properties?.mossImportedBoards || []);
    activeBoardKey = this.properties?.mossActiveBoardKey || ""; renderMboards();
    try { data = normalize(JSON.parse(storage.value)); selected = data.items[0]?.id; refresh(); }
    catch (e) { message(e.message); }
    this.onResize(); scheduleRecovery();
    return result;
  };
  const previousExecuted = node.onExecuted;
  node.onExecuted = function (result) {
    previousExecuted?.apply(this, arguments);
    if (result?.moss_prompt?.[0]) output.textContent = result.moss_prompt[0];
    if (result?.moss_report?.[0]) {
      const report = JSON.parse(result.moss_report[0]);
      message(`Layout compiled · ${report.regions} regions · Waiting for the saved image.`);
    }
  };
  const previousRemoved = node.onRemoved;
  const onRender = event => {
    if (String(event.detail.designer_id) !== String(node.id)) return;
    if (event.detail.project_id && event.detail.project_id !== data.project_id) return;
    data.generated = event.detail.generated; awaitingRender = false; persist(); loadRenderedImage();
    stage.dataset.previewState = "rendered";
    const frame = data.generated;
    const key = frame.files?.map(f => `${f.subfolder}/${f.filename}`).join("|") || frame.data_url?.slice(-120);
    if (frame.data_url && key !== latestRenderKey) {
      latestRenderKey = key;
      libraryPut("renders", { id: key, time: Date.now(), data_url: frame.data_url, width: frame.width, height: frame.height }).then(() => {
        const recentGallery = inspector.querySelector(".moss-gallery");
        if (recentGallery) return refreshRecent(recentGallery);
      }).catch(error => message(error.message));
    }
    message(`Generated ${data.generated.width} × ${data.generated.height} · Ready. Save prj as… includes the image and render metadata.`);
  };
  node._iamccsMossApplyResult = generated => onRender({ detail: { designer_id: node.id, project_id: data.project_id, generated } });
  node._iamccsMossReceiveImages = async (images, metadata = {}) => {
    if (!images?.length) return;
    const key = images.map(file => `${file.type}/${file.subfolder}/${file.filename}`).join("|");
    if (pendingImageKey === key) return;
    pendingImageKey = key;
    try {
      const dataURL = await imageDataURL(await api.fetchApi(imagePath(images[0])));
      const image = new Image(); image.src = dataURL; await image.decode();
      node._iamccsMossApplyResult({ data_url: dataURL, files: images, width: image.naturalWidth, height: image.naturalHeight, metadata });
    } catch (error) { message(`Could not display the saved image: ${error.message}`); }
    finally { pendingImageKey = ""; }
  };
  async function recoverSavedImage(explicit = false) {
    if (!explicit && (data.generated || awaitingRender)) return;
    const signature = layoutSignature(data);
    const response = await api.fetchApi("/history?max_items=8");
    if (!response.ok) throw new Error("Could not read ComfyUI image history.");
    const saved = latestSavedRender(await response.json(), data);
    if (signature !== layoutSignature(data)) return;
    if (saved) await node._iamccsMossReceiveImages(saved.images, { source: "ComfyUI completed render", prompt_id: saved.prompt_id });
    else if (explicit) message("No completed image matches these box prompts and this resolution.");
  }
  function scheduleRecovery() {
    clearTimeout(recoveryTimer);
    recoveryTimer = setTimeout(() => recoverSavedImage().catch(error => message(error.message)), 250);
  }
  const onStart = event => {
    if (String(event.detail.designer_id) !== String(node.id) || (event.detail.project_id && event.detail.project_id !== data.project_id)) return;
    awaitingRender = true; renderedImage = null; renderedSource = "";
    stage.dataset.previewState = "waiting"; stage.dataset.clearCount = String(+(stage.dataset.clearCount || 0) + 1);
    imageViewer?.close();
    draw(); message("New MOSS render queued. The previous preview is cleared; saved Mboards and recent renders remain available.");
  };
  node._iamccsMossLegacyStart = promptId => {
    if (lastLegacyPrompt === promptId) return;
    lastLegacyPrompt = promptId;
    onStart({ detail: { designer_id: node.id, project_id: data.project_id } });
  };
  api.addEventListener("iamccs-moss-render", onRender);
  api.addEventListener("iamccs-moss-start", onStart);
  node.onRemoved = function () { if (dirtyFrame) cancelAnimationFrame(dirtyFrame); imageViewer?.close(); clearTimeout(recoveryTimer); stageResize.disconnect(); api.removeEventListener("iamccs-moss-render", onRender); api.removeEventListener("iamccs-moss-start", onStart); root.remove(); return previousRemoved?.apply(this, arguments); };
  node.setSize([Math.max(node.size[0], EDITOR_WIDTH + 40), Math.max(node.size[1], minimumHeight)]); node.onResize(); renderMboards(); refresh(); persist(); scheduleRecovery(); message("MOSS native layout · Standard SaveImage and MOSS Result both update this canvas.");
}

app.registerExtension({
  name: "IAMCCS.MOSSFrameDesigner",
  setup() {
    let currentPrompt;
    api.addEventListener("execution_start", event => { currentPrompt = event.detail.prompt_id; });
    api.addEventListener("executed", event => {
      const saved = app.graph.getNodeById(event.detail.node);
      if (!saved || !["SaveImage", "PreviewImage"].includes(saved.comfyClass || saved.type)) return;
      for (const designer of upstreamDesigners(app.graph, saved.id)) designer._iamccsMossReceiveImages?.(event.detail.output?.images, { source: saved.comfyClass || saved.type, prompt_id: event.detail.prompt_id });
    });
    api.addEventListener("executing", event => {
      const id = event.detail?.node ?? event.detail;
      if (id == null) return;
      for (const designer of upstreamDesigners(app.graph, id)) designer._iamccsMossLegacyStart?.(event.detail?.prompt_id || currentPrompt || "running");
    });
  },
  nodeCreated(node) {
    if ((node.comfyClass || node.type) === TYPE) install(node);
    if ((node.comfyClass || node.type) === "IAMCCS_MOSSFrameResult") {
      const previous = node.onExecuted;
      node.onExecuted = function (result) {
        previous?.apply(this, arguments);
        if (result?.moss_project?.[0]) {
          const link = app.graph.links[node.inputs.find(i => i.name === "project_data")?.link];
          app.graph.getNodeById(link?.origin_id)?._iamccsMossApplyResult?.(JSON.parse(result.moss_project[0]).design.generated);
        }
      };
    }
  },
  loadedGraphNode(node) { if ((node.comfyClass || node.type) === TYPE) install(node); },
});
