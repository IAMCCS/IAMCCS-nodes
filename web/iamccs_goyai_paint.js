import { app } from "../../scripts/app.js";

const BUILD = "IAMCCS-GOYAI-PAINT-20260619-A";
console.info("[IAMCCS GoyAIcanvas Paint] loaded", BUILD);

function findWidget(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

function setWidget(node, name, value) {
  const widget = findWidget(node, name);
  if (!widget) return;
  widget.value = value;
  widget.callback?.(value);
}

function hideWidget(widget) {
  if (!widget) return;
  widget.type = "hidden";
  widget.computeSize = () => [0, -4];
}

function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }

function installGoyaiPaint(node) {
  if (node._iamccsGoyaiPaintReady) return;
  node._iamccsGoyaiPaintReady = true;

  const paintWidget = findWidget(node, "paint_data");
  hideWidget(paintWidget);

  const root = document.createElement("div");
  root.className = "iamccs-goyai-paint";
  root.innerHTML = `
    <style>
      .iamccs-goyai-paint{box-sizing:border-box;width:100%;min-width:520px;background:#081012;color:#e8f4f2;border:1px solid #24515a;border-radius:8px;padding:10px;font-family:Inter,Arial,sans-serif;}
      .igp-head{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px;}
      .igp-title{font-weight:800;color:#ffe0a3;font-size:13px;letter-spacing:.02em;}
      .igp-build{font-size:10px;color:#82c6ca;opacity:.8;}
      .igp-tools{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:6px;margin-bottom:8px;}
      .igp-btn{border:1px solid #2c646d;border-radius:6px;background:#10242a;color:#f3fbfb;font-weight:700;font-size:11px;padding:7px 8px;cursor:pointer;}
      .igp-btn.primary{background:#164b2f;border-color:#39a96b;}
      .igp-btn.warn{background:#5a2b10;border-color:#c7832b;}
      .igp-btn.danger{background:#551719;border-color:#b43c45;}
      .igp-btn.active{outline:2px solid #ff445f;background:#3d1420;}
      .igp-size{display:flex;align-items:center;gap:6px;background:#071519;border:1px solid #1f4650;border-radius:6px;padding:4px 6px;font-size:11px;}
      .igp-size input{width:100%;}
      .igp-stage{position:relative;display:flex;align-items:center;justify-content:center;min-height:280px;max-height:520px;background:#05080a;border:1px solid #28545d;border-radius:8px;overflow:auto;touch-action:none;}
      .igp-stack{position:relative;line-height:0;}
      .igp-stack canvas{display:block;max-width:100%;height:auto;image-rendering:auto;}
      .igp-img{background:#111;}
      .igp-mask{position:absolute;left:0;top:0;cursor:crosshair;touch-action:none;}
      .igp-status{margin-top:7px;white-space:pre-wrap;font:11px/1.35 Consolas,monospace;color:#b9eff2;background:#061114;border:1px solid #173841;border-radius:6px;padding:6px;max-height:72px;overflow:auto;}
      .igp-empty{padding:42px 24px;text-align:center;color:#9ec9cb;font-size:13px;}
    </style>
    <div class="igp-head"><div><div class="igp-title">GoyAIcanvas Paint - Image + Mask</div><div class="igp-build">${BUILD}</div></div><input data-file type="file" accept="image/*" style="display:none"></div>
    <div class="igp-tools">
      <button class="igp-btn primary" data-action="load" type="button">Load Image</button>
      <button class="igp-btn active" data-tool="paint" type="button">Brush</button>
      <button class="igp-btn" data-tool="erase" type="button">Erase</button>
      <button class="igp-btn warn" data-action="undo" type="button">Undo</button>
      <button class="igp-btn danger" data-action="clear" type="button">Clear Mask</button>
      <label class="igp-size">Size <input data-size type="range" min="1" max="256" value="64"><span data-size-label>64</span></label>
    </div>
    <div class="igp-stage" data-stage><div class="igp-empty">Load an image here, then paint the red inpaint mask.</div></div>
    <div class="igp-status" data-status>Ready.</div>
  `;

  const state = {
    imageData: "",
    imageName: "",
    width: 1024,
    height: 576,
    tool: "paint",
    brushSize: 64,
    strokes: [],
    drawing: null,
    img: null,
  };

  const fileInput = root.querySelector("[data-file]");
  const stage = root.querySelector("[data-stage]");
  const status = root.querySelector("[data-status]");
  const sizeInput = root.querySelector("[data-size]");
  const sizeLabel = root.querySelector("[data-size-label]");

  let imgCanvas = null;
  let maskCanvas = null;
  let imgCtx = null;
  let maskCtx = null;

  function log(message) {
    status.textContent = `${new Date().toLocaleTimeString()} ${message}\n${status.textContent || ""}`.slice(0, 1600);
  }

  function payload() {
    return JSON.stringify({
      version: 1,
      build: BUILD,
      image_name: state.imageName,
      image_data: state.imageData,
      width: state.width,
      height: state.height,
      brush_size: state.brushSize,
      strokes: state.strokes,
    });
  }

  function persist() {
    setWidget(node, "paint_data", payload());
    setWidget(node, "width", state.width);
    setWidget(node, "height", state.height);
  }

  function drawStroke(ctx, stroke) {
    const pts = stroke.points || [];
    if (!pts.length) return;
    const size = Math.max(1, Number(stroke.size || state.brushSize || 64));
    const r = size / 2;
    ctx.save();
    ctx.globalCompositeOperation = stroke.mode === "erase" ? "destination-out" : "source-over";
    ctx.strokeStyle = "rgba(255,32,64,.72)";
    ctx.fillStyle = "rgba(255,32,64,.72)";
    ctx.lineWidth = size;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    const cap = (p) => { ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, Math.PI * 2); ctx.fill(); };
    if (pts.length === 1) cap(pts[0]);
    else {
      for (let i = 1; i < pts.length; i++) {
        ctx.beginPath();
        ctx.moveTo(pts[i - 1][0], pts[i - 1][1]);
        ctx.lineTo(pts[i][0], pts[i][1]);
        ctx.stroke();
        if (i % 8 === 0) cap(pts[i]);
      }
      cap(pts[0]); cap(pts[pts.length - 1]);
    }
    ctx.restore();
  }

  function redrawMask() {
    if (!maskCtx || !maskCanvas) return;
    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    for (const stroke of state.strokes) drawStroke(maskCtx, stroke);
    if (state.drawing?.stroke) drawStroke(maskCtx, state.drawing.stroke);
  }

  function redrawImage() {
    if (!state.img || !imgCtx) return;
    imgCtx.clearRect(0, 0, imgCanvas.width, imgCanvas.height);
    imgCtx.drawImage(state.img, 0, 0, imgCanvas.width, imgCanvas.height);
  }

  function buildCanvases(img) {
    state.img = img;
    state.width = img.naturalWidth || img.width || 1024;
    state.height = img.naturalHeight || img.height || 576;
    stage.innerHTML = `<div class="igp-stack"><canvas class="igp-img"></canvas><canvas class="igp-mask"></canvas></div>`;
    imgCanvas = stage.querySelector(".igp-img");
    maskCanvas = stage.querySelector(".igp-mask");
    imgCanvas.width = state.width;
    imgCanvas.height = state.height;
    maskCanvas.width = state.width;
    maskCanvas.height = state.height;
    imgCtx = imgCanvas.getContext("2d", { willReadFrequently: true });
    maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true });
    redrawImage();
    redrawMask();
    bindMaskEvents();
    persist();
    node.setSize?.([Math.max(node.size?.[0] || 650, 650), Math.max(node.size?.[1] || 520, 520)]);
    log(`Loaded ${state.imageName || "image"} ${state.width}x${state.height}`);
  }

  function pointFromEvent(event) {
    const rect = maskCanvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    const x = (event.clientX - rect.left) * (maskCanvas.width / rect.width);
    const y = (event.clientY - rect.top) * (maskCanvas.height / rect.height);
    return [clamp(Math.round(x), 0, maskCanvas.width - 1), clamp(Math.round(y), 0, maskCanvas.height - 1)];
  }

  function stop(event) {
    event?.preventDefault?.();
    event?.stopPropagation?.();
    event?.stopImmediatePropagation?.();
  }

  function finish(event) {
    if (!state.drawing) return;
    stop(event);
    window.removeEventListener("pointermove", move, true);
    window.removeEventListener("pointerup", finish, true);
    window.removeEventListener("pointercancel", finish, true);
    const stroke = state.drawing.stroke;
    state.drawing = null;
    if ((stroke.points || []).length) state.strokes.push(stroke);
    redrawMask();
    persist();
    log(`stroke end mode=${stroke.mode} points=${stroke.points.length} strokes=${state.strokes.length}`);
  }

  function move(event) {
    if (!state.drawing) return;
    stop(event);
    const p = pointFromEvent(event);
    if (!p) return;
    const pts = state.drawing.stroke.points;
    const last = pts[pts.length - 1];
    if (!last || Math.hypot(p[0] - last[0], p[1] - last[1]) >= 1.5) pts.push(p);
    redrawMask();
  }

  function start(event) {
    if (!maskCanvas || event.button !== 0) return;
    stop(event);
    const p = pointFromEvent(event);
    if (!p) return;
    state.drawing = { stroke: { mode: state.tool, size: state.brushSize, points: [p] } };
    window.addEventListener("pointermove", move, true);
    window.addEventListener("pointerup", finish, true);
    window.addEventListener("pointercancel", finish, true);
    redrawMask();
    log(`stroke start mode=${state.tool} at=${p[0]},${p[1]}`);
  }

  function bindMaskEvents() {
    if (!maskCanvas) return;
    maskCanvas.addEventListener("pointerdown", start, true);
    maskCanvas.addEventListener("contextmenu", stop, true);
  }

  root.querySelector("[data-action='load']")?.addEventListener("click", (e) => { stop(e); fileInput.click(); });
  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      state.imageData = String(reader.result || "");
      state.imageName = file.name || "image";
      state.strokes = [];
      const img = new Image();
      img.onload = () => buildCanvases(img);
      img.src = state.imageData;
    };
    reader.readAsDataURL(file);
  });

  root.querySelectorAll("[data-tool]").forEach((button) => {
    button.addEventListener("click", (event) => {
      stop(event);
      state.tool = button.dataset.tool || "paint";
      root.querySelectorAll("[data-tool]").forEach((b) => b.classList.toggle("active", b === button));
      log(`tool=${state.tool}`);
    });
  });

  root.querySelector("[data-action='undo']")?.addEventListener("click", (event) => {
    stop(event);
    state.strokes.pop();
    redrawMask();
    persist();
    log(`undo strokes=${state.strokes.length}`);
  });

  root.querySelector("[data-action='clear']")?.addEventListener("click", (event) => {
    stop(event);
    state.strokes = [];
    redrawMask();
    persist();
    log("mask cleared");
  });

  sizeInput.addEventListener("input", () => {
    state.brushSize = Math.max(1, Math.round(Number(sizeInput.value) || 64));
    sizeLabel.textContent = String(state.brushSize);
    persist();
  });

  try {
    const initial = JSON.parse(paintWidget?.value || "{}");
    if (initial && initial.image_data) {
      state.imageData = initial.image_data;
      state.imageName = initial.image_name || "image";
      state.strokes = Array.isArray(initial.strokes) ? initial.strokes : [];
      state.brushSize = Number(initial.brush_size || state.brushSize) || state.brushSize;
      sizeInput.value = String(state.brushSize);
      sizeLabel.textContent = String(state.brushSize);
      const img = new Image();
      img.onload = () => buildCanvases(img);
      img.src = state.imageData;
    }
  } catch {}

  node.addDOMWidget("goyai_paint_ui", "GoyAIcanvas Paint", root, { serialize: false });
  node.setSize?.([Math.max(node.size?.[0] || 650, 650), Math.max(node.size?.[1] || 520, 520)]);
  persist();
}

app.registerExtension({
  name: "iamccs.goyai.paint",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "IAMCCS_GoyAICanvasPaint") return;
    const original = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      original?.apply(this, arguments);
      installGoyaiPaint(this);
    };
  },
});
