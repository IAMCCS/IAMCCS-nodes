import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TYPE = "IAMCCS_shotboarder_aud+vid_exporter_PRO";
const STYLE_ID = "iamccs-shotboarder-exporter-pro-style";
const UI_VERSION = "20260721-exporter-audio-edl-layout";
const NODE_SIZE = [680, 940];
const HIDDEN_FIELDS = [
  "filename_prefix",
  "audio_source_mode",
  "master_audio_file",
  "video_profile",
  "audio_profile",
  "audio_sync",
  "frame_rate_override",
  "video_quality",
  "embed_metadata",
  "write_sidecar",
  "metadata_json",
  "editor_audio_edl_json",
  "rtx_enabled",
  "rtx_mode",
  "rtx_resize_type",
  "rtx_scale",
  "rtx_megapixels",
  "rtx_width",
  "rtx_height",
  "rtx_divisible_by",
  "rtx_device",
  "rtx_ratio_preset",
  "rtx_resize_method",
];
const VIDEO_OPTIONS = [
  ["h264_mp4", "H.264 / MP4 / delivery"],
  ["h265_mp4", "H.265 / HEVC / MP4 / 10-bit"],
  ["prores_422_mov", "Apple ProRes 422 / MOV"],
  ["prores_422_hq_mov", "Apple ProRes 422 HQ / MOV"],
  ["prores_4444_mov", "Apple ProRes 4444 / MOV"],
  ["dnxhr_hqx_mov", "Avid DNxHR HQX / MOV / 10-bit"],
  ["v210_mov", "v210 10-bit 4:2:2 / MOV"],
  ["ffv1_mkv", "FFV1 lossless / MKV / archive"],
];
const AUDIO_OPTIONS = [
  ["copy_source", "Source audio / direct copy"],
  ["aac_320", "AAC 320 kb/s"],
  ["aac_192", "AAC 192 kb/s"],
  ["pcm_s16le", "PCM signed 16-bit"],
  ["pcm_s24le", "PCM signed 24-bit"],
  ["pcm_s32le", "PCM signed 32-bit"],
  ["flac", "FLAC lossless"],
  ["alac", "Apple Lossless / ALAC"],
];
const RTX_MODE_OPTIONS = [
  ["VSR Medium", "VSR Medium"],
  ["VSR High", "VSR High"],
  ["VSR Low", "VSR Low"],
  ["VSR Ultra", "VSR Ultra"],
  ["High Bitrate Medium", "High Bitrate Medium"],
  ["High Bitrate High", "High Bitrate High"],
  ["High Bitrate Low", "High Bitrate Low"],
  ["High Bitrate Ultra", "High Bitrate Ultra"],
  ["Denoise Medium", "Denoise Medium"],
  ["Denoise High", "Denoise High"],
  ["Denoise Low", "Denoise Low"],
  ["Denoise Ultra", "Denoise Ultra"],
  ["Deblur Medium", "Deblur Medium"],
  ["Deblur High", "Deblur High"],
  ["Deblur Low", "Deblur Low"],
  ["Deblur Ultra", "Deblur Ultra"],
];
const RTX_RESIZE_TYPE_OPTIONS = [
  ["Scale", "Scale"],
  ["Keep Ratio", "Keep Ratio"],
  ["Manual", "Manual"],
  ["Preset Ratio", "Preset Ratio"],
  ["Same Size", "Same Size"],
];
const RTX_DIVISIBLE_OPTIONS = ["1", "8", "16", "32", "64", "128"].map((value) => [value, value]);
const RTX_RATIO_OPTIONS = ["1:1", "4:5", "5:4", "3:4", "4:3", "2:3", "3:2", "16:9", "9:16", "16:10", "10:16", "21:9", "9:21"].map((value) => [value, value]);
const RTX_RESIZE_METHOD_OPTIONS = [
  ["Center Crop (Fill)", "Center Crop (Fill)"],
  ["Fit (Letterbox/Pillarbox)", "Fit (Letterbox/Pillarbox)"],
];

function nodeType(node) {
  return String(node?.comfyClass || node?.type || node?.constructor?.type || "");
}

function widget(node, name) {
  return (node.widgets || []).find((item) => item?.name === name || item?.label === name);
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
    .iamccs-export-pro { width:100%; height:902px; box-sizing:border-box; padding:10px; display:grid; grid-template-rows:40px 28px minmax(270px,1fr) 174px 170px 24px; gap:8px; overflow:hidden; background:#111719; color:#e8eeee; border:1px solid #465255; font-family:Arial,sans-serif; }
    .iamccs-export-pro * { box-sizing:border-box; min-width:0; border-radius:0 !important; letter-spacing:0; }
    .iamccs-export-pro .head { display:flex; align-items:center; justify-content:space-between; gap:10px; padding:6px 9px; background:#1a2427; border:1px solid #3d4c50; }
    .iamccs-export-pro .brand { color:#f1d88e; font-size:13px; font-weight:900; line-height:15px; }
    .iamccs-export-pro .sub { color:#96a9aa; font-size:9px; font-weight:700; margin-top:2px; }
    .iamccs-export-pro .badge { padding:5px 7px; border:1px solid #c4a557; color:#f6dda0; background:#252313; font:900 10px Consolas,monospace; white-space:nowrap; }
    .iamccs-export-pro .signal { display:flex; align-items:center; gap:7px; padding:5px 8px; background:#0c1112; border:1px solid #2d3a3d; color:#b9c8c7; font:800 10px Consolas,monospace; overflow:hidden; white-space:nowrap; }
    .iamccs-export-pro .dot { width:8px; height:8px; flex:0 0 8px; background:#62dc8a; border:1px solid #b9ffd0; box-shadow:0 0 7px #62dc8a; }
    .iamccs-export-pro .signal .tail { margin-left:auto; color:#7e9997; overflow:hidden; text-overflow:ellipsis; }
    .iamccs-export-pro .grid { min-height:0; display:grid; grid-template-columns:1fr 1fr; grid-template-rows:repeat(3,minmax(0,1fr)); gap:8px; overflow:hidden; }
    .iamccs-export-pro .control { display:grid; grid-template-rows:15px minmax(0,1fr); gap:4px; padding:7px 8px; background:#171f21; border:1px solid #354447; overflow:hidden; }
    .iamccs-export-pro .control.wide { grid-column:1 / -1; }
    .iamccs-export-pro .control label { color:#abbdbc; font:900 9px Consolas,monospace; text-transform:uppercase; }
    .iamccs-export-pro select, .iamccs-export-pro input { width:100%; height:27px; padding:0 7px; border:1px solid #556568; background:#0a0f10; color:#edf3f0; font:700 11px Arial,sans-serif; outline:none; }
    .iamccs-export-pro select:focus, .iamccs-export-pro input:focus { border-color:#d2b767; box-shadow:0 0 0 1px #8c7738; }
    .iamccs-export-pro .row { display:grid; grid-template-columns:minmax(0,1fr) 86px; gap:8px; align-items:center; }
    .iamccs-export-pro .checkrow { display:flex; align-items:center; gap:7px; height:27px; color:#d2dddd; font-size:10px; font-weight:800; }
    .iamccs-export-pro .checkrow input { width:15px; height:15px; accent-color:#d0b366; }
    .iamccs-export-pro .meta { display:flex; align-items:center; gap:7px; color:#8fa5a3; font:700 9px Consolas,monospace; overflow:hidden; white-space:nowrap; }
    .iamccs-export-pro .meta strong { color:#e8d18a; font-weight:900; }
    .iamccs-export-pro .rtx-panel { min-height:0; display:grid; grid-template-rows:24px minmax(0,1fr); gap:6px; padding:7px 8px; background:#111d1f; border:1px solid #6e5630; overflow:hidden; }
    .iamccs-export-pro .rtx-head { display:flex; align-items:center; justify-content:space-between; gap:8px; color:#f1d88e; font:900 10px Consolas,monospace; }
    .iamccs-export-pro .rtx-toggle { display:flex; align-items:center; gap:6px; color:#d5e1dd; font:800 10px Arial,sans-serif; }
    .iamccs-export-pro .rtx-toggle input { width:15px; height:15px; accent-color:#d0b366; }
    .iamccs-export-pro .rtx-head .rtx-note { color:#8da6a2; font-size:9px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .iamccs-export-pro .rtx-controls { min-height:0; display:grid; grid-template-columns:1.25fr 1.1fr .7fr .85fr; grid-template-rows:repeat(3,39px); align-content:start; gap:5px 7px; overflow:hidden; }
    .iamccs-export-pro .rtx-field { min-width:0; display:grid; grid-template-rows:11px 25px; gap:3px; }
    .iamccs-export-pro .rtx-field label { color:#a8bcb7; font:900 8px Consolas,monospace; text-transform:uppercase; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; }
    .iamccs-export-pro .rtx-controls select, .iamccs-export-pro .rtx-controls input { height:25px; font-size:10px; padding:0 5px; }
    .iamccs-export-pro .rtx-controls input:disabled, .iamccs-export-pro .rtx-controls select:disabled { opacity:.45; cursor:not-allowed; }
    .iamccs-export-pro .preview { position:relative; min-height:0; overflow:hidden; background:#050708; border:1px solid #354447; }
    .iamccs-export-pro .preview video { display:block; width:100%; height:100%; object-fit:contain; background:#000; }
    .iamccs-export-pro .preview .preview-label { position:absolute; left:8px; top:7px; padding:4px 6px; background:rgba(7,12,13,.82); border:1px solid #657477; color:#c9d7d4; font:900 9px Consolas,monospace; pointer-events:none; }
    .iamccs-export-pro .preview .preview-status { position:absolute; inset:0; display:grid; place-items:center; padding:12px; color:#829493; font:700 10px Consolas,monospace; text-align:center; pointer-events:none; }
    .iamccs-export-pro .preview .preview-status[hidden] { display:none; }
    .iamccs-export-pro .footer { display:flex; align-items:center; justify-content:space-between; gap:8px; padding:4px 8px; background:#0a0f10; border:1px solid #2d3a3d; color:#8fa7a4; font:700 9px Consolas,monospace; overflow:hidden; }
    .iamccs-export-pro .footer .status { color:#9bd8af; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  `;
  document.head.appendChild(style);
}

function setNodeSize(node) {
  node.min_size = [...NODE_SIZE];
  try { node.setSize([...NODE_SIZE]); } catch {}
}

function removeStaleUiWidgets(node) {
  const stale = (node.widgets || []).filter((item) => (
    item?.type === "iamccs_shotboarder_exporter_pro" || item?.name === "Shotboarder Pro Export"
  ));
  if (!stale.length) return;
  for (const item of stale) {
    try { item.element?.remove?.(); } catch {}
  }
  node.widgets = (node.widgets || []).filter((item) => !stale.includes(item));
}

function addOptions(select, options) {
  for (const [value, label] of options) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    select.appendChild(option);
  }
}

function optionLabel(options, value) {
  return options.find(([key]) => key === value)?.[1] || String(value || "");
}

function optionValueIsValid(options, value) {
  return options.some(([key]) => key === String(value || ""));
}

function migrateLegacyRtxWidgets(node) {
  // RTX fields were added after the first exporter workflows. In those saved
  // graphs the old metadata JSON can occupy rtx_mode by widget position.
  // Repair in-memory before Comfy serializes/queues the node.
  const defaults = [
    ["rtx_mode", RTX_MODE_OPTIONS, "VSR Medium"],
    ["rtx_resize_type", RTX_RESIZE_TYPE_OPTIONS, "Keep Ratio"],
    ["rtx_divisible_by", RTX_DIVISIBLE_OPTIONS, "1"],
    ["rtx_ratio_preset", RTX_RATIO_OPTIONS, "16:9"],
    ["rtx_resize_method", RTX_RESIZE_METHOD_OPTIONS, "Center Crop (Fill)"],
  ];
  let repaired = false;
  for (const [name, options, fallback] of defaults) {
    const item = widget(node, name);
    if (!item || optionValueIsValid(options, item.value)) continue;
    item.value = fallback;
    try { item.callback?.(fallback); } catch {}
    repaired = true;
  }
  if (repaired) {
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
    console.info("[IAMCCS Shotboarder Exporter PRO] migrated legacy RTX widget values");
  }
  return repaired;
}

function install(node) {
  if (nodeType(node) !== TYPE || node._iamccsShotboarderExporterUiVersion === UI_VERSION || typeof node.addDOMWidget !== "function") return;
  if (node._iamccsShotboarderExporterReady) removeStaleUiWidgets(node);
  node._iamccsShotboarderExporterUiVersion = UI_VERSION;
  node._iamccsShotboarderExporterReady = true;
  installStyle();
  migrateLegacyRtxWidgets(node);
  HIDDEN_FIELDS.forEach((name) => hideWidget(widget(node, name)));
  node.resizable = false;
  node.resizeable = false;
  node.flags = { ...(node.flags || {}), resizable: false };

  const root = document.createElement("div");
  root.className = "iamccs-export-pro";
  root.innerHTML = `
    <div class="head"><div><div class="brand">IAMCCS / SHOTBOARDER</div><div class="sub">AUDIO + VIDEO EXPORTER</div></div><div class="badge">PRO DELIVERY</div></div>
    <div class="signal"><span class="dot"></span><span>MASTER MUX</span><span class="tail" data-source-status>explicit AUDIO input / single pass</span></div>
    <div class="grid">
      <div class="control"><label>Video profile</label><select data-field="video_profile"></select></div>
      <div class="control"><label>Audio profile</label><select data-field="audio_profile"></select></div>
      <div class="control wide"><label>Audio source</label><select data-field="audio_source_mode"><option value="audio_input">Connected AUDIO waveform</option><option value="master_file_direct">Master file / direct stream copy</option></select></div>
      <div class="control"><label>Audio sync</label><select data-field="audio_sync"><option value="trim_to_video">Trim / pad to video</option><option value="shortest">Shortest stream</option></select></div>
      <div class="control"><label>Frame rate / quality</label><div class="row"><input data-field="frame_rate_override" type="number" min="0" max="240" step="0.01" placeholder="source fps"><input data-field="video_quality" type="number" min="0" max="51" step="1" title="CRF for H.264/H.265"></div></div>
      <div class="control wide"><label>Output prefix</label><input data-field="filename_prefix" type="text" spellcheck="false"></div>
      <div class="control wide"><label>Metadata</label><div class="checkrow"><label><input data-field="embed_metadata" type="checkbox"> embed</label><label><input data-field="write_sidecar" type="checkbox"> sidecar .metadata.json</label><span class="meta" data-meta></span></div></div>
    </div>
    <div class="rtx-panel">
      <div class="rtx-head"><label class="rtx-toggle"><input data-field="rtx_enabled" type="checkbox"><span>ENABLE RTX VIDEO SUPER RESOLUTION</span></label><span class="rtx-note" data-rtx-status>native optional module / bypassed</span></div>
      <div class="rtx-controls">
        <div class="rtx-field"><label>Quality mode</label><select data-field="rtx_mode"></select></div>
        <div class="rtx-field"><label>Resize type</label><select data-field="rtx_resize_type"></select></div>
        <div class="rtx-field"><label>Scale</label><input data-field="rtx_scale" type="number" min="1" max="4" step="0.05"></div>
        <div class="rtx-field"><label>Megapixels</label><input data-field="rtx_megapixels" type="number" min="0.01" max="64" step="0.01"></div>
        <div class="rtx-field"><label>Target width</label><input data-field="rtx_width" type="number" min="64" max="8192" step="8"></div>
        <div class="rtx-field"><label>Target height</label><input data-field="rtx_height" type="number" min="64" max="8192" step="8"></div>
        <div class="rtx-field"><label>Divisible by</label><select data-field="rtx_divisible_by"></select></div>
        <div class="rtx-field"><label>CUDA device</label><input data-field="rtx_device" type="number" min="0" max="16" step="1"></div>
        <div class="rtx-field"><label>Preset ratio</label><select data-field="rtx_ratio_preset"></select></div>
        <div class="rtx-field"><label>Aspect handling</label><select data-field="rtx_resize_method"></select></div>
      </div>
    </div>
    <div class="preview" data-preview><video data-preview-video controls playsinline preload="metadata"></video><div class="preview-label">EXPORTED VIDEO PREVIEW</div><div class="preview-status" data-preview-status>Run the exporter to load the encoded video.</div></div>
    <div class="footer"><span class="status" data-status>READY</span><span>FFMPEG / RTX FRAMES OUT</span></div>
  `;

  const videoSelect = root.querySelector('[data-field="video_profile"]');
  const audioSelect = root.querySelector('[data-field="audio_profile"]');
  addOptions(videoSelect, VIDEO_OPTIONS);
  addOptions(audioSelect, AUDIO_OPTIONS);
  addOptions(root.querySelector('[data-field="rtx_mode"]'), RTX_MODE_OPTIONS);
  addOptions(root.querySelector('[data-field="rtx_resize_type"]'), RTX_RESIZE_TYPE_OPTIONS);
  addOptions(root.querySelector('[data-field="rtx_divisible_by"]'), RTX_DIVISIBLE_OPTIONS);
  addOptions(root.querySelector('[data-field="rtx_ratio_preset"]'), RTX_RATIO_OPTIONS);
  addOptions(root.querySelector('[data-field="rtx_resize_method"]'), RTX_RESIZE_METHOD_OPTIONS);

  const sync = (name, element, parse = (value) => value) => {
    const value = read(node, name, element.type === "checkbox" ? false : "");
    if (element.type === "checkbox") element.checked = Boolean(value);
    else element.value = String(value ?? "");
    element.addEventListener("change", () => { write(node, name, parse(element.type === "checkbox" ? element.checked : element.value)); refresh(); });
    element.addEventListener("input", () => { if (element.type !== "checkbox") { write(node, name, parse(element.value)); refresh(); } });
  };

  sync("video_profile", videoSelect);
  sync("audio_profile", audioSelect);
  sync("audio_source_mode", root.querySelector('[data-field="audio_source_mode"]'));
  sync("audio_sync", root.querySelector('[data-field="audio_sync"]'));
  sync("frame_rate_override", root.querySelector('[data-field="frame_rate_override"]'), (value) => Number(value) || 0);
  sync("video_quality", root.querySelector('[data-field="video_quality"]'), (value) => Math.max(0, Math.min(51, Math.round(Number(value) || 0))));
  sync("filename_prefix", root.querySelector('[data-field="filename_prefix"]'));
  sync("embed_metadata", root.querySelector('[data-field="embed_metadata"]'), Boolean);
  sync("write_sidecar", root.querySelector('[data-field="write_sidecar"]'), Boolean);
  sync("rtx_enabled", root.querySelector('[data-field="rtx_enabled"]'), Boolean);
  sync("rtx_mode", root.querySelector('[data-field="rtx_mode"]'));
  sync("rtx_resize_type", root.querySelector('[data-field="rtx_resize_type"]'));
  sync("rtx_scale", root.querySelector('[data-field="rtx_scale"]'), (value) => Math.max(1, Math.min(4, Number(value) || 1)));
  sync("rtx_megapixels", root.querySelector('[data-field="rtx_megapixels"]'), (value) => Math.max(0.01, Math.min(64, Number(value) || 0.01)));
  sync("rtx_width", root.querySelector('[data-field="rtx_width"]'), (value) => Math.max(64, Math.min(8192, Math.round(Number(value) || 64))));
  sync("rtx_height", root.querySelector('[data-field="rtx_height"]'), (value) => Math.max(64, Math.min(8192, Math.round(Number(value) || 64))));
  sync("rtx_divisible_by", root.querySelector('[data-field="rtx_divisible_by"]'));
  sync("rtx_device", root.querySelector('[data-field="rtx_device"]'), (value) => Math.max(0, Math.min(16, Math.round(Number(value) || 0))));
  sync("rtx_ratio_preset", root.querySelector('[data-field="rtx_ratio_preset"]'));
  sync("rtx_resize_method", root.querySelector('[data-field="rtx_resize_method"]'));

  const sourceSelect = root.querySelector('[data-field="audio_source_mode"]');
  sourceSelect?.addEventListener("change", () => {
    if (String(sourceSelect.value || "") === "master_file_direct" && String(read(node, "audio_profile", "")) === "pcm_s16le") {
      write(node, "audio_profile", "copy_source");
    }
    refresh();
  });

  function refresh() {
    const videoValue = String(read(node, "video_profile", "prores_422_hq_mov"));
    const audioValue = String(read(node, "audio_profile", "pcm_s16le"));
    const sourceValue = String(read(node, "audio_source_mode", "audio_input"));
    if (videoSelect.value !== videoValue) videoSelect.value = videoValue;
    if (audioSelect.value !== audioValue) audioSelect.value = audioValue;
    const sourceSelect = root.querySelector('[data-field="audio_source_mode"]');
    if (sourceSelect && sourceSelect.value !== sourceValue) sourceSelect.value = sourceValue;
    const sourceStatus = root.querySelector("[data-source-status]");
    if (sourceStatus) sourceStatus.textContent = sourceValue === "master_file_direct"
      ? "master asset from cine_linx / one direct audio stream"
      : "connected AUDIO input / one mux pass";
    const rtxEnabled = Boolean(read(node, "rtx_enabled", false));
    const rtxDefaults = {
      rtx_mode: "VSR Medium",
      rtx_resize_type: "Keep Ratio",
      rtx_scale: 2,
      rtx_megapixels: 2,
      rtx_width: 1920,
      rtx_height: 1080,
      rtx_divisible_by: "1",
      rtx_device: 0,
      rtx_ratio_preset: "16:9",
      rtx_resize_method: "Center Crop (Fill)",
    };
    for (const [name, fallback] of Object.entries(rtxDefaults)) {
      const field = root.querySelector(`[data-field="${name}"]`);
      if (!field) continue;
      const value = read(node, name, fallback);
      if (String(field.value) !== String(value)) field.value = String(value);
      field.disabled = !rtxEnabled;
    }
    const rtxStatus = root.querySelector("[data-rtx-status]");
    if (rtxStatus) rtxStatus.textContent = rtxEnabled
      ? `native RTX active / ${String(read(node, "rtx_mode", "VSR Medium"))}`
      : "native optional module / bypassed";
    const losslessVideo = videoValue === "ffv1_mkv";
    const losslessAudio = ["pcm_s16le", "pcm_s24le", "pcm_s32le", "flac", "alac"].includes(audioValue);
    const meta = root.querySelector("[data-meta]");
    if (meta) meta.innerHTML = `<strong>${losslessVideo ? "VIDEO LOSSLESS" : optionLabel(VIDEO_OPTIONS, videoValue)}</strong> <span>|</span> <strong>${losslessAudio ? "AUDIO LOSSLESS" : optionLabel(AUDIO_OPTIONS, audioValue)}</strong>`;
    const status = root.querySelector("[data-status]");
    if (status) status.textContent = `${optionLabel(VIDEO_OPTIONS, videoValue)} / ${optionLabel(AUDIO_OPTIONS, audioValue)}`;
  }

  const previewVideo = root.querySelector("[data-preview-video]");
  const previewStatus = root.querySelector("[data-preview-status]");
  node.__iamccsExporterPreview = { video: previewVideo, status: previewStatus };
  const uiWidget = node.addDOMWidget("Shotboarder Pro Export", "iamccs_shotboarder_exporter_pro", root, { serialize: false });
  uiWidget.computeSize = () => [NODE_SIZE[0] - 18, 902];
  HIDDEN_FIELDS.forEach((name) => hideWidget(widget(node, name)));
  setNodeSize(node);
  refresh();
  node.setDirtyCanvas?.(true, true);
}

function exporterPreviewMeta(output) {
  const list = output?.iamccs_exporter_preview || output?.gifs || output?.videos;
  return Array.isArray(list) ? list[0] : null;
}

function updateExporterPreview(node, output) {
  const state = node?.__iamccsExporterPreview;
  const meta = exporterPreviewMeta(output);
  if (!state?.video || !meta?.filename) return;
  const params = new URLSearchParams({
    filename: String(meta.filename),
    subfolder: String(meta.subfolder || ""),
    type: String(meta.type || "output"),
    rand: String(Date.now()),
  });
  const src = api.apiURL("/view?" + params.toString());
  state.video.src = src;
  state.video.load();
  if (state.status) {
    state.status.hidden = false;
    state.status.textContent = "Encoded preview loaded. Press play to view.";
  }
  state.video.addEventListener("loadeddata", () => {
    if (state.status) state.status.hidden = true;
  }, { once: true });
}

app.registerExtension({
  name: "iamccs.shotboarder_exporter_pro",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== TYPE || nodeType.prototype.__iamccsExporterPreviewHook) return;
    nodeType.prototype.__iamccsExporterPreviewHook = true;
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (output) {
      const result = onExecuted?.apply(this, arguments);
      try { updateExporterPreview(this, output || {}); } catch {}
      return result;
    };
  },
  async nodeCreated(node) {
    install(node);
  },
});
