import { app } from "../../scripts/app.js";

const TYPE = "IAMCCS_VideoColorCorrectionControl";
const STYLE_ID = "iamccs-video-color-correction-style-20260706-layout";
const NODE_SIZE = [900, 590];
const CHROME_HEIGHT = 108;

function nodeType(node) {
  return String(node?.comfyClass || node?.type || node?.constructor?.type || "");
}

function widget(node, name) {
  return (node.widgets || []).find((item) => item?.name === name);
}

function read(node, name, fallback) {
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
  item.hidden = true;
  item.type = "hidden";
  item.computeSize = () => [0, 0];
  item.draw = () => {};
  item.options = { ...(item.options || {}), hidden: true };
  if (item.inputEl) item.inputEl.style.display = "none";
}

function installStyle() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
    .iamccs-vcc { width:100%; height:${NODE_SIZE[1] - CHROME_HEIGHT}px; box-sizing:border-box; background:#141719; color:#e3e8e5; border:1px solid #5b6668; display:grid; grid-template-rows:44px minmax(0,1fr) 28px; gap:8px; padding:10px; overflow:hidden; font-family:Arial,sans-serif; }
    .iamccs-vcc * { box-sizing:border-box; border-radius:0 !important; letter-spacing:0; }
    .iamccs-vcc button, .iamccs-vcc input { border:1px solid #626d70; background:#22292c; color:#eef3ef; height:26px; font-size:11px; font-weight:800; }
    .iamccs-vcc button.on { background:#d5a64e; color:#111; border-color:#f2cf7d; }
    .iamccs-vcc .top { display:grid; grid-template-columns:230px 1fr 230px; gap:10px; align-items:center; border:1px solid #313b3e; background:#0f1517; padding:6px; min-width:0; }
    .iamccs-vcc h3 { margin:0; color:#ffe1a1; font-size:13px; }
    .iamccs-vcc .sub { color:#8ea4a4; font-size:10px; margin-top:2px; }
    .iamccs-vcc .look { min-width:0; display:flex; gap:8px; align-items:center; justify-content:center; font-weight:900; color:#fff1b8; }
    .iamccs-vcc .look input { width:170px; min-width:0; padding:0 7px; background:#091011; color:#f3ecd7; }
    .iamccs-vcc .main { min-height:0; display:grid; grid-template-columns:minmax(0,1.08fr) minmax(280px,.92fr); gap:8px; }
    .iamccs-vcc .panel { min-height:0; border:1px solid #354246; background:#0b1011; overflow:hidden; display:grid; grid-template-rows:24px minmax(0,1fr); }
    .iamccs-vcc .title { padding:5px 7px; color:#ffe1a1; font-size:10px; font-weight:900; background:#151d20; border-bottom:1px solid #354246; }
    .iamccs-vcc .controls { height:100%; min-height:0; display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; padding:10px; overflow:hidden; }
    .iamccs-vcc .strip { min-height:0; border:1px solid #394a4e; background:linear-gradient(180deg,#202426,#101315); padding:8px 8px 9px; display:grid; grid-template-rows:18px 78px repeat(3,34px); gap:7px; min-width:0; overflow:hidden; }
    .iamccs-vcc .strip-name { color:#d6bc78; font-size:10px; line-height:18px; font-weight:900; text-align:center; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .iamccs-vcc .wheel { width:78px; height:78px; margin:0 auto; border:2px solid #687579; background:radial-gradient(circle at 50% 50%,#253034 0,#151b1e 54%,#070909 100%); position:relative; box-shadow:inset 0 0 22px rgba(255,255,255,.08); }
    .iamccs-vcc .wheel:before { content:""; position:absolute; inset:8px; background:conic-gradient(#f35,#fc4,#4e7,#3cf,#85f,#f35); opacity:.55; }
    .iamccs-vcc .wheel:after { content:""; position:absolute; left:50%; top:50%; width:2px; height:34px; background:#f0d27a; transform-origin:bottom center; transform:translate(-50%,-100%) rotate(var(--angle,0deg)); box-shadow:0 0 6px #f0d27a; }
    .iamccs-vcc label { min-width:0; display:grid; grid-template-columns:54px minmax(30px,1fr) 34px; gap:4px; align-items:center; color:#c7d7d2; font-size:9px; font-weight:800; }
    .iamccs-vcc input[type=range] { width:100%; min-width:0; height:18px; accent-color:#d5a64e; }
    .iamccs-vcc input[type=number] { width:34px; min-width:34px; padding:0 2px; text-align:center; font-family:Consolas,monospace; background:#071011; color:#eaffea; font-size:10px; }
    .iamccs-vcc .scopes { display:grid; grid-template-rows:1fr 1fr; gap:8px; padding:10px; min-height:0; }
    .iamccs-vcc canvas { width:100%; height:100%; background:#020606; border:1px solid #26383b; display:block; }
    .iamccs-vcc .bottom { display:flex; align-items:center; justify-content:space-between; border:1px solid #313b3e; background:#0b1011; padding:0 8px; font-family:Consolas,monospace; color:#9edfb6; font-size:10px; }
  `;
  document.head.appendChild(style);
}

function setNodeSize(node) {
  if (!node) return;
  const width = Math.max(Number(node.size?.[0] || 0), NODE_SIZE[0]);
  const height = Math.max(Number(node.size?.[1] || 0), NODE_SIZE[1]);
  node.size = [width, height];
  node.min_size = [...NODE_SIZE];
  try { node.setSize?.([width, height]); } catch {}
}

function drawScope(canvas, kind, values) {
  if (!canvas?.isConnected) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (!Number.isFinite(rect.width) || !Number.isFinite(rect.height) || rect.width < 4 || rect.height < 4) return;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.scale(dpr, dpr);
  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "rgba(110,140,145,.25)";
  ctx.lineWidth = 1;
  for (let x = 0; x <= w; x += w / 8) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
  for (let y = 0; y <= h; y += h / 4) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
  if (kind === "waveform") {
    const exp = Number(values.exposure || 0);
    const con = Number(values.contrast || 1);
    const sat = Number(values.saturation || 1);
    [["#ff5a5a", exp], ["#62ff87", con - 1], ["#5fb9ff", sat - 1]].forEach(([color, offset], channel) => {
      ctx.strokeStyle = color;
      ctx.globalAlpha = .82;
      ctx.beginPath();
      for (let x = 0; x <= w; x += 4) {
        const y = h * .5 - Math.sin((x / w) * Math.PI * 3 + channel) * h * .18 - Number(offset) * h * .16;
        if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    });
    ctx.globalAlpha = 1;
  } else {
    const temp = Number(values.temperature || 0);
    const tint = Number(values.tint || 0);
    const radius = Math.min(w, h) * .38;
    const cx = w / 2 + tint * radius;
    const cy = h / 2 - temp * radius;
    ctx.strokeStyle = "#d5a64e";
    ctx.beginPath();
    ctx.arc(w / 2, h / 2, radius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = "#e7c86d";
    ctx.beginPath();
    ctx.arc(cx, cy, 6, 0, Math.PI * 2);
    ctx.fill();
  }
}

function installNode(node) {
  if (nodeType(node) !== TYPE || node._iamccsVccReady || typeof node.addDOMWidget !== "function") return;
  node._iamccsVccReady = "installing";
  installStyle();
  setNodeSize(node);

  const root = document.createElement("div");
  root.className = "iamccs-vcc";
  const values = () => ({
    enabled: Boolean(read(node, "enabled", true)),
    exposure: Number(read(node, "exposure", 0)),
    contrast: Number(read(node, "contrast", 1)),
    saturation: Number(read(node, "saturation", 1)),
    gamma: Number(read(node, "gamma", 1)),
    temperature: Number(read(node, "temperature", 0)),
    tint: Number(read(node, "tint", 0)),
    vignette: Number(read(node, "vignette", 0)),
    look_name: String(read(node, "look_name", "neutral") || "neutral"),
  });

  let waveformCanvas = null;
  let vectorscopeCanvas = null;

  const refresh = () => {
    const current = values();
    root.querySelector("[data-enabled]")?.classList.toggle("on", current.enabled);
    root.querySelectorAll("[data-wheel]").forEach((el) => {
      const key = el.dataset.wheel;
      const angle = key === "temperature" ? current.temperature * 50 : key === "tint" ? current.tint * 50 : (current.exposure || 0) * 22;
      el.style.setProperty("--angle", `${angle}deg`);
    });
    try {
      if (waveformCanvas) drawScope(waveformCanvas, "waveform", current);
      if (vectorscopeCanvas) drawScope(vectorscopeCanvas, "vectorscope", current);
    } catch (error) {
      console.warn("[IAMCCS Video Color] scope draw skipped", error);
    }
    const status = root.querySelector(".bottom span");
    if (status) status.textContent = `${current.enabled ? "ON" : "BYPASS"} | look ${current.look_name} | exp ${current.exposure.toFixed(2)} | con ${current.contrast.toFixed(2)} | sat ${current.saturation.toFixed(2)}`;
  };

  const slider = (label, name, min, max, step) => {
    const wrap = document.createElement("label");
    const range = document.createElement("input");
    const num = document.createElement("input");
    range.type = "range";
    range.min = min;
    range.max = max;
    range.step = step;
    num.type = "number";
    num.min = min;
    num.max = max;
    num.step = step;
    const sync = (value) => {
      const next = Math.max(Number(min), Math.min(Number(max), Number(value) || 0));
      range.value = String(next);
      num.value = String(next);
      write(node, name, next);
      refresh();
    };
    range.value = String(read(node, name, 0));
    num.value = String(read(node, name, 0));
    range.addEventListener("input", () => sync(range.value));
    num.addEventListener("change", () => sync(num.value));
    wrap.append(label, range, num);
    return wrap;
  };

  const strip = (title, wheelKey, controls) => {
    const el = document.createElement("div");
    el.className = "strip";
    const name = document.createElement("div");
    name.className = "strip-name";
    name.textContent = title;
    const wheelEl = document.createElement("div");
    wheelEl.className = "wheel";
    wheelEl.dataset.wheel = wheelKey;
    el.append(name, wheelEl, ...controls);
    return el;
  };

  const top = document.createElement("div");
  top.className = "top";
  const title = document.createElement("div");
  title.innerHTML = `<h3>IAMCCS Color Correction</h3><div class="sub">editor grade control / cine_linx metadata</div>`;
  const look = document.createElement("div");
  look.className = "look";
  const lookInput = document.createElement("input");
  lookInput.value = values().look_name;
  lookInput.addEventListener("change", () => { write(node, "look_name", lookInput.value || "neutral"); refresh(); });
  look.append("LOOK", lookInput);
  const enabled = document.createElement("button");
  enabled.textContent = "ACTIVE";
  enabled.dataset.enabled = "1";
  enabled.addEventListener("click", () => { write(node, "enabled", !values().enabled); refresh(); });
  top.append(title, look, enabled);

  const main = document.createElement("div");
  main.className = "main";
  const controlPanel = document.createElement("div");
  controlPanel.className = "panel";
  controlPanel.innerHTML = `<div class="title">PRIMARY CORRECTION</div>`;
  const controls = document.createElement("div");
  controls.className = "controls";
  controls.append(
    strip("LIFT / EXPOSURE", "exposure", [
      slider("Exposure", "exposure", -4, 4, .01),
      slider("Gamma", "gamma", .1, 4, .01),
    ]),
    strip("BALANCE", "temperature", [
      slider("Temp", "temperature", -1, 1, .01),
      slider("Tint", "tint", -1, 1, .01),
    ]),
    strip("LOOK", "tint", [
      slider("Contrast", "contrast", 0, 3, .01),
      slider("Saturation", "saturation", 0, 3, .01),
      slider("Vignette", "vignette", 0, 1, .01),
    ]),
  );
  controlPanel.appendChild(controls);
  const scopes = document.createElement("div");
  scopes.className = "panel";
  scopes.innerHTML = `<div class="title">SCOPES / PREVIEW</div>`;
  const scopeBody = document.createElement("div");
  scopeBody.className = "scopes";
  waveformCanvas = document.createElement("canvas");
  vectorscopeCanvas = document.createElement("canvas");
  scopeBody.append(waveformCanvas, vectorscopeCanvas);
  scopes.appendChild(scopeBody);
  main.append(controlPanel, scopes);

  const bottom = document.createElement("div");
  bottom.className = "bottom";
  bottom.innerHTML = `<span>READY</span><span>Attach cine_linx before Video Editor / render</span>`;
  root.append(top, main, bottom);
  const domWidget = node.addDOMWidget("IAMCCS Color Correction", "iamccs_video_color_correction", root, { serialize: false });
  domWidget.computeSize = () => [NODE_SIZE[0] - 24, NODE_SIZE[1] - CHROME_HEIGHT];
  ["enabled", "exposure", "contrast", "saturation", "gamma", "temperature", "tint", "vignette", "look_name"].forEach((name) => hideWidget(widget(node, name)));
  node._iamccsVccReady = true;
  refresh();
  setTimeout(refresh, 80);
}

function scheduleInstall(node) {
  if (nodeType(node) !== TYPE || node?._iamccsVccReady) return;
  const run = () => {
    try {
      installNode(node);
    } catch (error) {
      node._iamccsVccReady = false;
      console.error("[IAMCCS Video Color] UI disabled after safe install failure", error);
    }
  };
  if (typeof window.requestIdleCallback === "function") {
    window.requestIdleCallback(run, { timeout: 500 });
  } else {
    setTimeout(run, 40);
  }
}

app.registerExtension({
  name: "IAMCCS.VideoColorCorrectionControl.UI",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== TYPE) return;
    if (nodeType.prototype.__iamccsVccWrapped) return;
    nodeType.prototype.__iamccsVccWrapped = true;
    const original = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const ret = original?.apply(this, args);
      scheduleInstall(this);
      return ret;
    };
    const originalConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (...args) {
      const ret = originalConfigure?.apply(this, args);
      scheduleInstall(this);
      return ret;
    };
  },
});
