import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TYPE = "IAMCCS_ShotboardVideoEditorV1";
const RENDER_TYPE = "IAMCCS_ShotboardVideoEditorRenderV1";
const STYLE_ID = "iamccs-shotboard-video-editor-v1-style";
const UI_VERSION = "20260710-replace-prj-selection-waveforms";
const NODE_SIZE = [1600, 1560];
const CHROME_HEIGHT = 156;
const WIDGET_HEIGHT = NODE_SIZE[1] - CHROME_HEIGHT;
const LANE_HEAD_WIDTH = 92;
const MIN_TIMELINE_PX_PER_SECOND = 44;
const TIMELINE_TAIL_PAD_SECONDS = 4;

function nodeType(node) {
  return String(node?.comfyClass || node?.type || node?.constructor?.type || "");
}

function isEditor(node) {
  return nodeType(node) === TYPE;
}

function widget(node, name) {
  return (node.widgets || []).find((item) => item?.name === name);
}

function setWidget(node, name, value) {
  const item = widget(node, name);
  if (!item) return false;
  item.value = value;
  try { item.callback?.(value); } catch {}
  try { node.setDirtyCanvas?.(true, true); } catch {}
  try { app.graph?.setDirtyCanvas?.(true, true); } catch {}
  return true;
}

function findSlotIndex(node, slotKind, name) {
  const slots = slotKind === "input" ? (node?.inputs || []) : (node?.outputs || []);
  const wanted = String(name || "").toLowerCase();
  return slots.findIndex((slot) => String(slot?.name || "").toLowerCase() === wanted);
}

function setNodeWidgetValue(node, name, value) {
  const item = widget(node, name);
  if (!item) return false;
  item.value = value;
  try { item.callback?.(value); } catch {}
  return true;
}

function forceNodeWidgetValue(node, name, index, value) {
  if (setNodeWidgetValue(node, name, value)) return true;
  const item = node?.widgets?.[index];
  if (!item) return false;
  item.value = value;
  try { item.callback?.(value); } catch {}
  return true;
}

function connectBySlotName(source, outputName, target, inputName) {
  if (!source || !target) return false;
  const outIndex = findSlotIndex(source, "output", outputName);
  const inIndex = findSlotIndex(target, "input", inputName);
  if (outIndex < 0 || inIndex < 0) return false;
  try {
    target.disconnectInput?.(inIndex);
  } catch {}
  try {
    source.connect(outIndex, target, inIndex);
    return true;
  } catch {
    return false;
  }
}

function disconnectInputBySlotName(node, inputName) {
  if (!node) return false;
  const inIndex = findSlotIndex(node, "input", inputName);
  if (inIndex < 0) return false;
  try {
    node.disconnectInput?.(inIndex);
    return true;
  } catch {
    return false;
  }
}

function graphLinksArray(graph) {
  const links = graph?.links;
  if (!links) return [];
  return Array.isArray(links) ? links.filter(Boolean) : Object.values(links).filter(Boolean);
}

function linkOriginId(link) {
  return link?.origin_id ?? link?.originId ?? (Array.isArray(link) ? link[1] : undefined);
}

function linkTargetId(link) {
  return link?.target_id ?? link?.targetId ?? (Array.isArray(link) ? link[3] : undefined);
}

function nodeTypeMatches(node, type) {
  return String(node?.comfyClass || node?.type || node?.constructor?.type || "") === type;
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

function hideRawWidgets(node) {
  ["session_key", "collect_policy", "append_mode", "fps", "editor_manifest_json", "take_package_json"].forEach((name) => hideWidget(widget(node, name)));
}

function parseJson(text, fallback = null) {
  try {
    const parsed = JSON.parse(String(text || "").trim());
    return parsed && typeof parsed === "object" ? parsed : fallback;
  } catch {
    return fallback;
  }
}

function fmtTime(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  const hh = Math.floor(value / 3600);
  const mm = Math.floor((value % 3600) / 60);
  const ss = Math.floor(value % 60);
  const ff = Math.floor((value - Math.floor(value)) * 100);
  return `${String(hh).padStart(2, "0")}:${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")}.${String(ff).padStart(2, "0")}`;
}

const EDITOR_TRACKS = [
  ...Array.from({ length: 5 }, (_, index) => ({ id: `V${index + 1}`, name: `V${index + 1}`, kind: "video" })),
  ...Array.from({ length: 5 }, (_, index) => ({ id: `A${index + 1}`, name: `A${index + 1}`, kind: "audio" })),
  { id: "AM", name: "MASTER AUDIO", kind: "master_audio" },
];

function normalizeManifestTracks(manifest) {
  const data = manifest && typeof manifest === "object" ? manifest : {};
  const tracks = Array.isArray(data.tracks) ? data.tracks.filter((track) => track && typeof track === "object") : [];
  const byId = new Map(tracks.map((track) => [String(track.id || ""), track]));
  const normalized = [];
  for (const base of EDITOR_TRACKS) {
    normalized.push({ ...base, ...(byId.get(base.id) || {}) });
    byId.delete(base.id);
  }
  for (const track of byId.values()) normalized.push(track);
  data.tracks = normalized;
  return data;
}

function pairedTimelineTracks(manifest) {
  const tracks = normalizeManifestTracks(manifest).tracks || [];
  const byId = new Map(tracks.map((track) => [String(track.id || ""), track]));
  const ordered = [];
  const used = new Set();
  for (let index = 1; index <= 5; index++) {
    for (const id of [`V${index}`, `A${index}`]) {
      const track = byId.get(id);
      if (track && !isMasterTrack(track)) {
        ordered.push(track);
        used.add(id);
      }
    }
  }
  for (const track of tracks) {
    const id = String(track.id || "");
    if (!used.has(id) && !isMasterTrack(track)) ordered.push(track);
  }
  return ordered;
}

function manifestFromNode(node) {
  const raw = widget(node, "editor_manifest_json")?.value || node.properties?.iamccs_video_editor_manifest || "";
  const parsed = parseJson(raw, null);
  if (parsed?.schema === "iamccs.shotboard_video_editor.v1") return ensureMasterAudioLaneClip(normalizeManifestTracks(parsed));
  return ensureMasterAudioLaneClip(normalizeManifestTracks({
    schema: "iamccs.shotboard_video_editor.v1",
    schema_version: 1,
    fps: Number(widget(node, "fps")?.value || 24),
    assets: {},
    clips: [],
    tracks: EDITOR_TRACKS.map((track) => ({ ...track })),
    duration_seconds: 20,
    assembly_order: [],
    ui_state: { playhead: 0, zoom_seconds: 20 },
  }));
}

function saveManifest(node, manifest) {
  ensureMasterAudioLaneClip(manifest);
  manifest.updated_at = Date.now() / 1000;
  const text = JSON.stringify(manifest, null, 2);
  node.properties = node.properties || {};
  node.properties.iamccs_video_editor_manifest = text;
  setWidget(node, "editor_manifest_json", text);
  repairEditorHiddenWidgets(node, text);
}

function videoTakeIndexes(manifest) {
  const clips = Array.isArray(manifest?.clips) ? manifest.clips : [];
  return new Set(clips
    .filter((clip) => String(clip?.type || "") === "video")
    .map((clip) => Math.max(0, Math.round(Number(clip?.takeIndex || 0))))
    .filter((take) => take > 0));
}

function incomingAppendTakes(current, incoming) {
  const currentTakes = videoTakeIndexes(current);
  const incomingTakes = videoTakeIndexes(incoming);
  return Array.from(incomingTakes).filter((take) => !currentTakes.has(take)).sort((a, b) => a - b);
}

function mergeIncomingAppendManifest(current, incoming) {
  const appendTakes = incomingAppendTakes(current, incoming);
  if (!appendTakes.length) return null;
  const appendSet = new Set(appendTakes);
  const out = normalizeManifestTracks(JSON.parse(JSON.stringify(current || {})));
  const source = normalizeManifestTracks(JSON.parse(JSON.stringify(incoming || {})));
  out.assets = out.assets && typeof out.assets === "object" ? out.assets : {};
  out.clips = Array.isArray(out.clips) ? out.clips : [];
  const incomingAssets = source.assets && typeof source.assets === "object" ? source.assets : {};
  for (const [assetId, asset] of Object.entries(incomingAssets)) {
    const take = Math.max(0, Math.round(Number(asset?.takeIndex || 0)));
    const role = String(asset?.role || assetId || "");
    if (appendSet.has(take) || role === "master_audio" || role === "master_excerpt") {
      out.assets[assetId] = JSON.parse(JSON.stringify(asset));
    }
  }
  const existingClipIds = new Set(out.clips.map((clip) => String(clip?.id || "")));
  for (const clip of source.clips || []) {
    const take = Math.max(0, Math.round(Number(clip?.takeIndex || 0)));
    if (!appendSet.has(take)) continue;
    const id = String(clip?.id || "");
    if (id && existingClipIds.has(id)) continue;
    out.clips.push(JSON.parse(JSON.stringify(clip)));
  }
  const trackMap = new Map((out.tracks || []).map((track) => [String(track?.id || ""), track]));
  for (const track of source.tracks || []) {
    const id = String(track?.id || "");
    if (id && !trackMap.has(id)) {
      out.tracks.push(JSON.parse(JSON.stringify(track)));
      trackMap.set(id, track);
    }
  }
  out.duration_seconds = Math.max(Number(out.duration_seconds || 0), Number(source.duration_seconds || 0), manifestEndSeconds(out));
  out.assembly_order = Array.from(new Set([
    ...(Array.isArray(out.assembly_order) ? out.assembly_order : []),
    ...(Array.isArray(source.assembly_order) ? source.assembly_order : []),
  ]));
  out.ui_state = out.ui_state && typeof out.ui_state === "object" ? out.ui_state : {};
  return out;
}

function editorRenderAudioPolicy(manifest) {
  const policy = String(manifest?.render_audio_policy || manifest?.audio_policy || "concat_clip_audio");
  return policy === "use_master_audio" ? "use_master_audio" : "concat_clip_audio";
}

function syncMasterAudioClipPolicy(manifest) {
  const clips = Array.isArray(manifest?.clips) ? manifest.clips : [];
  const masterClip = clips.find((clip) => isMasterClip(clip));
  if (!masterClip) return null;
  // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
  masterClip.muted = editorRenderAudioPolicy(manifest) !== "use_master_audio";
  return masterClip;
}

function setEditorRenderAudioPolicy(manifest, policy) {
  manifest.render_audio_policy = String(policy) === "use_master_audio" ? "use_master_audio" : "concat_clip_audio";
  manifest.audio_policy = manifest.render_audio_policy;
  syncMasterAudioClipPolicy(manifest);
  return manifest.render_audio_policy;
}

function manifestMasterAudioAsset(manifest) {
  if (manifest?.master_excerpt && typeof manifest.master_excerpt === "object") return manifest.master_excerpt;
  if (manifest?.master_audio && typeof manifest.master_audio === "object") return manifest.master_audio;
  const assets = manifest?.assets && typeof manifest.assets === "object" ? manifest.assets : {};
  return assets.master_excerpt || assets.master_audio || null;
}

function hasActiveMasterAudioClip(manifest) {
  const clips = Array.isArray(manifest?.clips) ? manifest.clips : [];
  return clips.some((clip) => isMasterClip(clip) && !clip?.muted);
}

function copyMasterAudioBundle(target, source) {
  if (!target || !source || typeof target !== "object" || typeof source !== "object") return false;
  let changed = false;
  target.assets = target.assets && typeof target.assets === "object" ? target.assets : {};
  const sourceAssets = source.assets && typeof source.assets === "object" ? source.assets : {};
  for (const key of ["master_excerpt", "master_audio"]) {
    if (!target[key] && source[key] && typeof source[key] === "object") {
      target[key] = JSON.parse(JSON.stringify(source[key]));
      changed = true;
    }
    if (!target.assets[key] && sourceAssets[key] && typeof sourceAssets[key] === "object") {
      target.assets[key] = JSON.parse(JSON.stringify(sourceAssets[key]));
      changed = true;
    }
  }
  const fallbackAsset = manifestMasterAudioAsset(source);
  if (!manifestMasterAudioAsset(target) && fallbackAsset) {
    target.master_excerpt = JSON.parse(JSON.stringify({ ...fallbackAsset, id: "master_excerpt", role: "master_excerpt" }));
    target.assets.master_excerpt = JSON.parse(JSON.stringify({ ...fallbackAsset, id: "master_excerpt", role: "master_excerpt" }));
    changed = true;
  }
  const sourceClips = Array.isArray(source.clips) ? source.clips : [];
  const targetClips = Array.isArray(target.clips) ? target.clips : [];
  if (!targetClips.some((clip) => isMasterClip(clip))) {
    for (const clip of sourceClips) {
      if (isMasterClip(clip)) {
        target.clips = targetClips.concat([JSON.parse(JSON.stringify(clip))]);
        changed = true;
        break;
      }
    }
  }
  if (changed) ensureMasterAudioLaneClip(target);
  return changed;
}

function manifestFromRenderNode(renderNode) {
  if (!renderNode) return null;
  const direct =
    widget(renderNode, "editor_manifest_json")?.value ||
    renderNode.properties?.iamccs_video_editor_manifest ||
    renderNode.properties?.editor_manifest_json ||
    "";
  let parsed = parseJson(direct, null);
  if (parsed?.schema === "iamccs.shotboard_video_editor.v1") return parsed;
  for (const item of renderNode.widgets || []) {
    if (typeof item?.value !== "string" || !isManifestJsonText(item.value)) continue;
    parsed = parseJson(item.value, null);
    if (parsed?.schema === "iamccs.shotboard_video_editor.v1") return parsed;
  }
  return null;
}

function ensureMasterAudioLaneClip(manifest) {
  const data = normalizeManifestTracks(manifest);
  const asset = manifestMasterAudioAsset(data);
  if (!asset || typeof asset !== "object") return data;
  data.assets = data.assets && typeof data.assets === "object" ? data.assets : {};
  data.clips = Array.isArray(data.clips) ? data.clips : [];
  const assetId = data.assets.master_excerpt || asset.role === "master_excerpt" || data.master_excerpt
    ? "master_excerpt"
    : "master_audio";
  data.assets[assetId] = {
    id: assetId,
    type: "audio",
    takeIndex: 0,
    timelineId: "MASTER",
    audioLane: "MASTER",
    role: assetId,
    ...asset,
  };
  if (assetId === "master_excerpt") data.master_excerpt = data.master_excerpt || { ...asset, id: "master_excerpt", role: "master_excerpt" };
  if (assetId === "master_audio") data.master_audio = data.master_audio || { ...asset, id: "master_audio", role: "master_audio" };
  const duration = Math.max(
    0,
    Number(asset.duration || 0),
    Number(asset.duration_seconds || 0),
    Number(asset.timeline_duration || 0),
    Number(asset.source_duration || 0),
    Number(asset.trimEnd || 0)
  );
  if (duration <= 0) return data;
  const existing = data.clips.find((clip) => clip && (clip.id === "clip_MASTER_AUDIO" || isMasterClip(clip)));
  const masterClip = existing || {
    id: "clip_MASTER_AUDIO",
    assetId,
    type: "audio",
    takeIndex: 0,
    timelineId: "MASTER",
    audioLane: "MASTER",
    linkedClipIds: [],
    role: assetId,
  };
  masterClip.id = "clip_MASTER_AUDIO";
  masterClip.assetId = assetId;
  masterClip.type = "audio";
  masterClip.takeIndex = 0;
  masterClip.timelineId = "MASTER";
  masterClip.audioLane = "MASTER";
  masterClip.startTime = Math.max(0, Number(masterClip.startTime || 0));
  masterClip.duration = Math.max(0.1, Number(masterClip.duration || duration));
  masterClip.trimStart = Math.max(0, Number(masterClip.trimStart || 0));
  masterClip.trimEnd = Math.max(masterClip.trimStart + 0.1, Number(masterClip.trimEnd || duration));
  masterClip.trackId = "AM";
  masterClip.trackIndex = 10;
  masterClip.muted = Boolean(masterClip.muted);
  masterClip.volume = Number.isFinite(Number(masterClip.volume)) ? Number(masterClip.volume) : 1;
  masterClip.linkedClipIds = Array.isArray(masterClip.linkedClipIds) ? masterClip.linkedClipIds : [];
  masterClip.role = assetId;
  data.clips = data.clips.filter((clip) => clip === masterClip || !isMasterClip(clip));
  if (!data.clips.includes(masterClip)) data.clips.push(masterClip);
  data.duration_seconds = Math.max(Number(data.duration_seconds || 0), clipEndSeconds(masterClip));
  // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
  syncMasterAudioClipPolicy(data);
  return data;
}

function isManifestJsonText(value) {
  if (typeof value !== "string") return false;
  const trimmed = value.trim();
  return trimmed.startsWith("{") && trimmed.includes('"schema"') && trimmed.includes("iamccs.shotboard_video_editor.v1");
}

function repairEditorHiddenWidgets(node, manifestText = "") {
  if (!Array.isArray(node?.widgets)) return;
  const validCollect = new Set(["append_sequence", "replace_same_take", "append_always"]);
  const validAppend = new Set(["append_sequence", "timeline_origin"]);
  const defaults = {
    session_key: "shotboard_video_editor_v1",
    collect_policy: "append_sequence",
    append_mode: "append_sequence",
    fps: 24,
    take_package_json: "",
  };
  const manifestValue = manifestText || node.properties?.iamccs_video_editor_manifest || "";
  for (const item of node.widgets) {
    if (!item || item.type === "iamccs_shotboard_video_editor_v1") continue;
    const name = String(item.name || "");
    if (name !== "editor_manifest_json" && isManifestJsonText(item.value)) {
      if (Object.prototype.hasOwnProperty.call(defaults, name)) item.value = defaults[name];
      else item.value = "";
    }
    if (name === "session_key" && !String(item.value || "").trim()) item.value = defaults.session_key;
    if (name === "collect_policy" && !validCollect.has(String(item.value || ""))) item.value = defaults.collect_policy;
    if (name === "append_mode" && !validAppend.has(String(item.value || ""))) item.value = defaults.append_mode;
    if (name === "fps" && (!Number.isFinite(Number(item.value)) || Number(item.value) <= 0)) item.value = defaults.fps;
    if (name === "take_package_json" && isManifestJsonText(item.value)) item.value = "";
    if (name === "editor_manifest_json" && manifestValue) item.value = manifestValue;
  }
}

function previewUrl(item) {
  if (!item?.filename) return "";
  const filename = encodeURIComponent(item.filename);
  const type = encodeURIComponent(item.type || "output");
  const subfolder = encodeURIComponent(item.subfolder || "");
  return api.apiURL(`/view?filename=${filename}&type=${type}&subfolder=${subfolder}`);
}

function uniqueNonEmpty(values) {
  const seen = new Set();
  const out = [];
  for (const value of values || []) {
    const text = String(value || "").trim();
    if (!text || seen.has(text)) continue;
    seen.add(text);
    out.push(text);
  }
  return out;
}

function assetForClip(manifest, clip) {
  return manifest?.assets?.[clip?.assetId] || null;
}

function imageForClip(manifest, clip, localSeconds) {
  const asset = assetForClip(manifest, clip);
  const strip = Array.isArray(asset?.preview_strip) ? asset.preview_strip : [];
  if (!strip.length) return "";
  const sourceDuration = clipSourceDuration(manifest, clip);
  const trimStart = Math.max(0, Number(clip?.trimStart || 0));
  const editableDuration = clipEditableDuration(manifest, clip);
  const rawTrimEnd = Math.max(trimStart + 0.001, Number(clip?.trimEnd || trimStart + Number(clip?.duration || 0)));
  const trimEnd = Math.max(trimStart + 0.001, Math.min(editableDuration, rawTrimEnd));
  const sourceSeconds = Math.max(trimStart, Math.min(trimEnd - 0.001, trimStart + Math.max(0, Number(localSeconds || 0))));
  const ratio = Math.max(0, Math.min(0.999, sourceSeconds / Math.max(0.001, sourceDuration)));
  const index = Math.min(strip.length - 1, Math.floor(ratio * strip.length));
  return previewUrl(strip[index]);
}

function audioPeaksForClip(manifest, clip) {
  const asset = assetForClip(manifest, clip);
  const raw = Array.isArray(asset?.waveform_peaks)
    ? asset.waveform_peaks
    : Array.isArray(asset?.waveformPeaks)
      ? asset.waveformPeaks
      : Array.isArray(asset?.peaks)
        ? asset.peaks
        : Array.isArray(clip?.waveform_peaks)
          ? clip.waveform_peaks
          : Array.isArray(clip?.waveformPeaks)
            ? clip.waveformPeaks
            : [];
  return raw
    .map((peak) => {
      if (peak && typeof peak === "object") {
        const max = Number.isFinite(Number(peak.max)) ? Math.abs(Number(peak.max)) : Math.abs(Number(peak.min || 0));
        const rms = Number.isFinite(Number(peak.rms)) ? Math.abs(Number(peak.rms)) : max * 0.55;
        return { max: Math.max(0, Math.min(1, max)), rms: Math.max(0, Math.min(1, rms)) };
      }
      const value = Math.max(0, Math.min(1, Math.abs(Number(peak || 0))));
      return { max: value, rms: value * 0.55 };
    })
    .filter((peak) => peak.max > 0 || peak.rms > 0);
}

function shouldRefreshRealWaveform(manifest, clip) {
  const asset = assetForClip(manifest, clip);
  if (audioPeaksForClip(manifest, clip).length && (asset?.waveformReal !== false || asset?.collected_runtime)) return false;
  const urls = audioCandidateUrlsForClip(manifest, clip);
  return Boolean(asset && urls.length && (!asset.waveformReal || !urls.includes(String(asset.waveform_source_url || ""))));
}

function audioCandidateUrlsForClip(manifest, clip) {
  const asset = assetForClip(manifest, clip);
  if (!asset) return [];
  const urls = [];
  const pushDirect = (url) => {
    const text = String(url || "").trim();
    if (!text) return;
    urls.push(/^https?:\/\//i.test(text) || text.startsWith("blob:") ? text : api.apiURL(text));
  };
  const pushPreview = (filename, declaredType, subfolder = "") => {
    const cleanName = String(filename || "").trim();
    if (!cleanName) return;
    const types = uniqueNonEmpty([declaredType, "input", "output", "temp"]);
    for (const type of types) urls.push(previewUrl({ filename: cleanName, type, subfolder }));
  };
  const pushPath = (rawPath, declaredType) => {
    const path = String(rawPath || "").replace(/\\/g, "/").trim();
    if (!path) return;
    const parts = path.split("/").filter(Boolean);
    const filename = parts.pop() || "";
    const subfolder = parts.join("/");
    pushPreview(filename, declaredType, subfolder);
    if (subfolder) pushPreview(filename, declaredType, "");
  };
  pushDirect(asset.url || asset.audioUrl || asset.preview_url);
  if (asset.audio_preview_file) {
    pushPreview(asset.audio_preview_file, asset.audio_preview_type || asset.audioUploadType || asset.preview_type || "input", asset.audio_preview_subfolder || "");
  }
  if (asset.filename) {
    pushPreview(asset.filename, asset.preview_type || asset.file_type || asset.audioUploadType || "input", asset.preview_subfolder || asset.subfolder || "");
  }
  pushPath(asset.path, asset.audioUploadType || asset.preview_type || asset.file_type || "input");
  pushPath(asset.audioFile, asset.audioUploadType || asset.preview_type || asset.file_type || "input");
  pushPath(asset.packagePath, asset.audioUploadType || asset.preview_type || asset.file_type || "input");
  return uniqueNonEmpty(urls);
}

function audioUrlForClip(manifest, clip) {
  return audioCandidateUrlsForClip(manifest, clip)[0] || "";
}

async function fetchArrayBufferWithTimeout(url, timeoutMs = 7000) {
  const controller = typeof AbortController !== "undefined" ? new AbortController() : null;
  const timer = controller ? window.setTimeout(() => controller.abort(), timeoutMs) : 0;
  try {
    const response = await fetch(url, controller ? { signal: controller.signal } : undefined);
    if (!response.ok) throw new Error(`audio fetch failed ${response.status}`);
    return await response.arrayBuffer();
  } finally {
    if (timer) window.clearTimeout(timer);
  }
}

function clipSourceDuration(manifest, clip) {
  const asset = assetForClip(manifest, clip);
  return Math.max(
    0.1,
    Number(clip?.sourceDuration || 0),
    Number(asset?.duration || 0),
    Number(clip?.trimEnd || 0),
    Number(clip?.duration || 0)
  );
}

function clipEditableDuration(manifest, clip) {
  const asset = assetForClip(manifest, clip);
  const sourceDuration = clipSourceDuration(manifest, clip);
  const explicitLimit = Math.max(
    0,
    Number(clip?.sourceDurationLimit || 0),
    Number(clip?.timeline_duration || 0),
    Number(asset?.timeline_duration || 0),
    Number(asset?.target_duration || 0)
  );
  if (explicitLimit > 0) return Math.max(0.1, Math.min(sourceDuration, explicitLimit));
  const fallback = Math.max(0, Number(clip?.trimEnd || 0), Number(clip?.duration || 0));
  return Math.max(0.1, Math.min(sourceDuration, fallback || sourceDuration));
}

function clampClipToEditableDuration(manifest, clip) {
  if (!clip || (clip.type !== "video" && clip.type !== "audio")) return false;
  const before = JSON.stringify({
    duration: clip.duration,
    trimStart: clip.trimStart,
    trimEnd: clip.trimEnd,
  });
  const limit = clipEditableDuration(manifest, clip);
  const trimStart = Math.max(0, Math.min(limit - 0.1, Number(clip.trimStart || 0)));
  const maxDuration = Math.max(0.1, limit - trimStart);
  clip.trimStart = trimStart;
  clip.duration = Math.max(0.1, Math.min(Number(clip.duration || maxDuration), maxDuration));
  clip.trimEnd = Math.max(trimStart + 0.1, Math.min(limit, Number(clip.trimEnd || trimStart + clip.duration), trimStart + clip.duration));
  clip.duration = Math.max(0.1, Math.min(clip.duration, clip.trimEnd - trimStart));
  const after = JSON.stringify({
    duration: clip.duration,
    trimStart: clip.trimStart,
    trimEnd: clip.trimEnd,
  });
  return before !== after;
}

function isMasterTrack(track) {
  const id = String(track?.id || "").trim().toUpperCase();
  const kind = String(track?.kind || "").trim().toLowerCase();
  const name = String(track?.name || "").trim().toLowerCase();
  return id === "AM" || id === "MASTER" || kind === "master_audio" || name === "master audio";
}

function isMasterClip(clip) {
  const trackId = String(clip?.trackId || "").trim().toUpperCase();
  const role = String(clip?.role || "").trim().toLowerCase();
  const lane = String(clip?.audioLane || "").trim().toUpperCase();
  return trackId === "AM" || trackId === "MASTER" || role === "master_audio" || role === "master_excerpt" || lane === "MASTER";
}

function clipEndSeconds(clip) {
  return Math.max(0, Number(clip?.startTime || 0) + Number(clip?.duration || 0));
}

function manifestEndSeconds(manifest) {
  const clips = Array.isArray(manifest?.clips) ? manifest.clips : [];
  return Math.max(0, ...clips.map(clipEndSeconds));
}

function installStyle() {
  let style = document.getElementById(STYLE_ID);
  if (!style) {
    style = document.createElement("style");
    style.id = STYLE_ID;
    document.head.appendChild(style);
  }
  style.textContent = `
    .iamccs-sve { width:100%; height:${WIDGET_HEIGHT}px; box-sizing:border-box; background:#131719; color:#d8e1df; border:1px solid #566368; overflow:hidden; font-family:Arial, sans-serif; display:grid; grid-template-rows:70px 48px 554px minmax(0,1fr) 30px; gap:8px; padding:10px; }
    .iamccs-sve * { box-sizing:border-box; border-radius:0 !important; letter-spacing:0; }
    .iamccs-sve button, .iamccs-sve select, .iamccs-sve input { height:28px; border:1px solid #5f6f74; background:#222c30; color:#e9f2ef; font-size:11px; font-weight:800; padding:0 10px; box-shadow:inset 0 1px 0 rgba(255,255,255,.08); }
    .iamccs-sve button:hover { border-color:#d2bd78; background:#2c383c; }
    .iamccs-sve button.on { background:#d6aa55; color:#10100d; border-color:#f0ce83; box-shadow:inset 0 0 0 2px rgba(255,255,255,.18); }
    .iamccs-sve button.tap-feedback { transform:translateY(1px); border-color:#fff0b2 !important; box-shadow:inset 0 0 0 2px rgba(255,255,255,.26),0 0 0 2px rgba(214,170,85,.42),0 0 14px rgba(214,170,85,.38) !important; filter:brightness(1.16); }
    .iamccs-sve button.gold { background:#c79a4a; color:#111; border-color:#edcf89; }
    .iamccs-sve .top { display:grid; grid-template-columns:168px 398px minmax(0,1fr); gap:8px; min-height:0; padding:6px; border:1px solid #30393b; background:#101618; }
    .iamccs-sve .brand { border:1px solid #344145; padding:0 8px; background:#10191b; display:flex; align-items:center; }
    .iamccs-sve .brand h3 { margin:0; color:#ffe4a2; font-size:13px; line-height:1.25; }
    .iamccs-sve .brand p { display:none; }
    .iamccs-sve .transport { display:flex; align-items:center; justify-content:center; gap:4px; background:#0b1314; border:1px solid #263639; padding:5px; min-width:0; overflow:hidden; }
    .iamccs-sve .transport button { height:32px; min-width:42px; padding:0 8px; white-space:nowrap; flex:0 0 auto; }
    .iamccs-sve .transport .wide { min-width:50px; }
    .iamccs-sve .clock { margin-left:auto; min-width:270px; background:#edffe8; color:#00851d; border:2px solid #8cb889; font-family:Consolas,monospace; font-size:17px; font-weight:900; display:flex; align-items:center; justify-content:center; padding:0 12px; }
    .iamccs-sve .audio-meter { width:118px; height:30px; border:1px solid #4f6267; background:#061010; padding:4px; display:flex; align-items:center; gap:4px; }
    .iamccs-sve .audio-meter span { font-family:Consolas,monospace; font-size:9px; color:#9ce8ad; min-width:24px; }
    .iamccs-sve .audio-meter .meter-shell { position:relative; height:12px; flex:1; background:linear-gradient(90deg,#063b24,#384000 65%,#4b0f0f); border:1px solid #1e3133; overflow:hidden; }
    .iamccs-sve .audio-meter .meter-fill { position:absolute; left:0; top:0; bottom:0; width:0%; background:linear-gradient(90deg,#17ff75,#ffe66d 72%,#ff4a4a); box-shadow:0 0 8px rgba(35,255,130,.5); }
    .iamccs-sve .tools { display:flex; gap:4px; align-items:center; flex-wrap:nowrap; justify-content:flex-start; align-content:center; padding:5px; border:1px solid #263639; background:#0b1314; overflow:hidden; min-width:0; }
    .iamccs-sve .tools button { height:26px; white-space:nowrap; flex:1 1 auto; min-width:0; padding:0 6px; font-size:10px; overflow:hidden; text-overflow:ellipsis; }
    .iamccs-sve .takes { display:flex; gap:8px; align-items:center; min-height:0; padding:7px; border:1px solid #30393b; background:#172023; overflow:hidden; }
    .iamccs-sve .takes-label { color:#ffe4a2; font-size:10px; font-weight:900; min-width:68px; text-align:center; }
    .iamccs-sve .takes-scroll { display:flex; gap:4px; align-items:center; overflow-x:auto; overflow-y:hidden; flex:0 0 184px; max-width:184px; padding-bottom:2px; scrollbar-width:thin; }
    .iamccs-sve .takes-scroll button { flex:0 0 86px; }
    .iamccs-sve .takes-scroll::-webkit-scrollbar { height:7px; }
    .iamccs-sve .takes-scroll::-webkit-scrollbar-track { background:#081012; }
    .iamccs-sve .takes-scroll::-webkit-scrollbar-thumb { background:#526a6e; border:1px solid #7b9296; }
    .iamccs-sve .takes-actions { margin-left:auto; display:flex; gap:6px; align-items:center; }
    .iamccs-sve-purge-confirm { position:fixed; z-index:2147483600; width:300px; max-width:calc(100vw - 16px); padding:10px; border:1px solid #e0a85d; border-radius:6px; background:linear-gradient(180deg,#263033,#0d1315); color:#f4f8e9; box-shadow:0 12px 28px rgba(0,0,0,.58),0 0 0 1px rgba(255,226,168,.12) inset; font:11px/1.35 Inter,system-ui,sans-serif; }
    .iamccs-sve-purge-confirm strong { display:block; margin-bottom:5px; color:#ffe4a2; font-weight:900; }
    .iamccs-sve-purge-confirm .detail { display:block; margin-bottom:9px; color:#bfd0ce; font-size:10px; }
    .iamccs-sve-purge-confirm .actions { display:flex; justify-content:flex-end; gap:6px; }
    .iamccs-sve-purge-confirm button { min-width:76px; height:26px; padding:0 9px; border:1px solid #687b7e; border-radius:4px; background:#182326; color:#e7f1e8; cursor:pointer; font-size:10px; font-weight:900; }
    .iamccs-sve-purge-confirm button.confirm { border-color:#e0a85d; background:linear-gradient(180deg,#70451f,#3b2415); color:#ffe8bd; }
    .iamccs-sve .render-controls { display:flex; gap:6px; align-items:center; border-left:1px solid #344145; padding-left:8px; margin-left:2px; }
    .iamccs-sve .mini-field { display:flex; align-items:center; gap:4px; height:28px; border:1px solid #4d5d61; background:#0b1314; padding:0 6px; color:#ffe4a2; font-size:10px; font-weight:900; white-space:nowrap; }
    .iamccs-sve .mini-field input { width:52px; height:22px; padding:0 5px; text-align:center; font-family:Consolas,monospace; background:#071011; color:#edffe8; }
    .iamccs-sve .main { display:grid; grid-template-columns:240px 1fr; min-height:0; gap:10px; padding:8px; background:#111719; border:1px solid #30393b; }
    .iamccs-sve .pool, .iamccs-sve .monitor { border:1px solid #344145; background:#0b1011; overflow:hidden; min-height:0; }
    .iamccs-sve .monitor { display:grid; grid-template-rows:22px minmax(0,1fr); }
    .iamccs-sve .panel-title { height:22px; padding:5px 7px; color:#ffe4a2; background:#111b1d; font-size:10px; font-weight:800; border-bottom:1px solid #344145; }
    .iamccs-sve .pool-list { height:calc(100% - 22px); overflow:auto; padding:5px; }
    .iamccs-sve .pool-item { border:1px solid #33484d; background:#172226; margin-bottom:5px; padding:5px; font-size:10px; cursor:pointer; }
    .iamccs-sve .pool-item.on { border-color:#e7bd67; background:#302816; }
    .iamccs-sve .monitor-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; height:100%; min-height:0; }
    .iamccs-sve .screen { position:relative; border:1px solid #344145; background:#020606; height:100%; min-height:0; overflow:hidden; display:flex; align-items:center; justify-content:center; }
    .iamccs-sve .screen img { width:100%; height:100%; object-fit:contain; display:none; background:#050909; }
    .iamccs-sve .screen .empty { position:absolute; inset:20px; border:1px dashed #2e4145; display:flex; align-items:center; justify-content:center; color:#61777a; font-size:12px; }
    .iamccs-sve .screen .tag { position:absolute; left:8px; bottom:8px; background:#050909; border:1px solid #2a3c40; padding:4px 7px; font-family:Consolas,monospace; font-size:10px; }
    .iamccs-sve .source-transport { position:absolute; left:8px; right:8px; top:30px; height:30px; display:flex; gap:4px; justify-content:center; pointer-events:auto; opacity:.92; }
    .iamccs-sve .source-transport button { height:24px; min-width:38px; padding:0 7px; background:#172629; }
    .iamccs-sve .timeline-wrap { min-height:0; margin:0; border:1px solid #45565a; background:#080d0e; overflow:hidden; display:grid; grid-template-rows:40px minmax(0,1fr) 84px; }
    .iamccs-sve .meter { height:38px; display:grid; grid-template-columns:${LANE_HEAD_WIDTH}px 1fr; border-bottom:1px solid #344145; background:#10191b; cursor:crosshair; overflow:hidden; user-select:none; }
    .iamccs-sve .meter-label { color:#ffe4a2; font-size:10px; display:flex; align-items:center; justify-content:center; font-weight:800; border-right:1px solid #344145; }
    .iamccs-sve .meter-ruler { position:relative; overflow:hidden; background:#0c1416; user-select:none; }
    .iamccs-sve .ruler-track { position:absolute; inset:0 auto 0 0; height:100%; will-change:transform; pointer-events:none; user-select:none; }
    .iamccs-sve .tick { position:absolute; top:0; bottom:0; border-left:1px solid #344145; color:#c7d7d2; font-family:Consolas,monospace; font-size:10px; padding-left:3px; user-select:none; }
    .iamccs-sve .tick.minor { border-left-color:#233236; color:#537175; font-size:8px; padding-left:2px; opacity:.75; }
    .iamccs-sve .tick .frame { display:block; margin-top:13px; color:#82a3a7; font-size:9px; }
    .iamccs-sve .tick .half { display:block; margin-top:18px; }
    .iamccs-sve .lane-scroll { min-height:0; overflow:auto; position:relative; background:#070b0c; }
    .iamccs-sve .lanes { position:relative; min-height:100%; width:max-content; min-width:100%; }
    .iamccs-sve .master-lane-fixed { height:84px; overflow:hidden; border-top:2px solid #d6aa55; background:#14100a; position:relative; z-index:25; flex-shrink:0; }
    .iamccs-sve .lane { display:grid; grid-template-columns:${LANE_HEAD_WIDTH}px 1fr; min-height:72px; border-bottom:1px solid #314044; width:max-content; min-width:100%; }
    .iamccs-sve .lane.master-audio { min-height:82px; border-top:0; border-bottom:0; }
    .iamccs-sve .lane-head { position:sticky; left:0; z-index:6; background:#11191b; border-right:1px solid #344145; padding:7px 6px; color:#dce8e5; font-weight:900; box-shadow:4px 0 0 rgba(0,0,0,.2); }
    .iamccs-sve .lane.master-audio .lane-head { background:#2a2414; color:#ffe4a2; }
    .iamccs-sve .lane-head .kind { display:block; font-size:9px; color:#80a5a8; font-weight:600; margin-top:2px; }
    .iamccs-sve .lane.is-selected { outline:2px solid rgba(255,226,168,.62); outline-offset:-2px; box-shadow:inset 0 0 0 1px rgba(255,226,168,.25), inset 0 0 24px rgba(214,170,85,.10); }
    .iamccs-sve .lane.is-selected .lane-head { background:linear-gradient(180deg,#4a3719,#1a1711); color:#ffe4a2; border-right-color:#f0cf79; box-shadow:4px 0 0 rgba(214,170,85,.22), inset 0 0 0 1px rgba(255,226,168,.24); }
    .iamccs-sve .lane.is-selected .lane-body { background-color:rgba(214,170,85,.055); }
    .iamccs-sve .lane-body { position:relative; min-height:72px; background-image:linear-gradient(90deg, rgba(255,255,255,.07) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px), linear-gradient(180deg, rgba(255,255,255,.04), transparent); background-size:var(--major-grid, 62px) 100%, var(--half-grid, 31px) 100%, 100% 100%; }
    .iamccs-sve .master-lane-fixed .lane-body { min-height:82px; will-change:transform; }
    .iamccs-sve .clip { position:absolute; top:10px; height:50px; border:2px solid #f0cf79; background:#24414a; color:#fff; overflow:hidden; cursor:grab; touch-action:none; }
    .iamccs-sve .clip.audio { background:#1b4e78; border-color:#9bd6ff; }
    .iamccs-sve .clip.master_audio { background:#4b3915; border-color:#f0cf79; }
    .iamccs-sve .clip.selected { outline:2px solid #ffefb0; }
    .iamccs-sve .clip.is-dragging { z-index:18; opacity:.94; cursor:grabbing; box-shadow:0 0 0 2px rgba(255,239,176,.45),0 6px 18px rgba(0,0,0,.52); }
    .iamccs-sve .clip-title { position:absolute; left:6px; top:3px; font-size:10px; text-shadow:0 1px 2px #000; z-index:2; }
    .iamccs-sve .film { position:absolute; inset:0; display:flex; opacity:.85; }
    .iamccs-sve .film img { height:100%; width:54px; object-fit:cover; border-right:1px solid rgba(255,255,255,.18); }
    .iamccs-sve .wave { position:absolute; inset:0; opacity:.9; }
    .iamccs-sve .handle { position:absolute; top:0; width:14px; height:100%; background:linear-gradient(90deg,rgba(255,225,139,.92),rgba(255,225,139,.32)); cursor:ew-resize; z-index:6; touch-action:none; }
    .iamccs-sve .handle.left { left:0; }
    .iamccs-sve .handle.right { right:0; }
    .iamccs-sve .handle::after { content:""; position:absolute; top:9px; bottom:9px; width:2px; left:6px; background:rgba(20,16,8,.56); box-shadow:4px 0 0 rgba(20,16,8,.34); }
    .iamccs-sve .clip-preview-label { position:absolute; right:4px; bottom:3px; z-index:8; height:16px; padding:1px 5px; border:1px solid rgba(255,240,199,.72); background:rgba(5,8,8,.74); color:#fff3bf; font:9px/12px Consolas,monospace; pointer-events:none; }
    .iamccs-sve .playhead { position:absolute; top:0; bottom:0; width:2px; background:#ffe06b; z-index:20; pointer-events:none; }
    .iamccs-sve .ruler-render-controls { position:absolute; right:8px; top:4px; height:30px; z-index:35; display:flex; gap:6px; align-items:center; background:rgba(7,16,17,.94); border:1px solid #506066; padding:2px 5px; pointer-events:auto; }
    .iamccs-sve .ruler-render-controls .mini-field { height:24px; padding:0 5px; }
    .iamccs-sve .ruler-render-controls .mini-field input { height:19px; width:42px; }
    .iamccs-sve .ruler-render-controls button { height:24px; min-width:72px; }
    .iamccs-sve .render-main-button { min-width:86px; border-color:#f2d78d !important; background:#d6aa55 !important; color:#10100d !important; box-shadow:inset 0 0 0 1px rgba(255,255,255,.18),0 0 10px rgba(214,170,85,.22); }
    .iamccs-sve .status { min-height:0; display:flex; align-items:center; padding:0 8px; background:#060a0b; border-top:1px solid #344145; color:#a8c7c4; font-family:Consolas,monospace; font-size:10px; }
    .iamccs-sve-fullscreen-overlay { position:fixed; inset:0; z-index:2147483640; background:rgba(4,8,10,.90); display:flex; flex-direction:column; gap:10px; padding:14px; box-sizing:border-box; pointer-events:auto; }
    .iamccs-sve-fullscreen-bar { height:42px; flex:0 0 42px; display:flex; align-items:center; justify-content:space-between; border:1px solid #566368; background:#101719; padding:0 10px; color:#ffe4a2; font-weight:900; font-size:13px; }
    .iamccs-sve-fullscreen-bar button { height:28px; border:1px solid #5f6f74; background:#c79a4a; color:#111; font-size:11px; font-weight:900; padding:0 12px; border-radius:0; }
    .iamccs-sve-fullscreen-panel { flex:1 1 auto; min-height:0; border:1px solid #566368; background:#090f11; overflow:hidden; }
    .iamccs-sve.is-fullscreen { width:100%; height:100%; grid-template-rows:62px 42px minmax(274px,calc(40vh - 50px)) minmax(0,1fr) 30px; gap:8px; padding:10px; }
    .iamccs-sve.is-fullscreen .top { grid-template-columns:168px 398px minmax(0,1fr); padding:6px; }
    .iamccs-sve.is-fullscreen .main { grid-template-columns:260px 1fr; padding:7px; }
    .iamccs-sve.is-fullscreen .brand { padding:0 8px; display:flex; align-items:center; }
    .iamccs-sve.is-fullscreen .transport,
    .iamccs-sve.is-fullscreen .tools { padding:5px; }
    .iamccs-sve.is-fullscreen .transport { min-width:0; }
    .iamccs-sve.is-fullscreen .transport button { height:32px; padding:0 8px; white-space:nowrap; flex:0 0 auto; }
    .iamccs-sve.is-fullscreen .tools button { height:26px; padding:0 6px; white-space:nowrap; flex:1 1 auto; min-width:0; overflow:hidden; text-overflow:ellipsis; }
    .iamccs-sve.is-fullscreen .audio-meter { width:118px; height:30px; }
    .iamccs-sve.is-fullscreen .tools { flex-wrap:nowrap; justify-content:flex-start; overflow:hidden; align-content:center; }
    .iamccs-sve.is-fullscreen .takes { padding:5px 7px; }
    .iamccs-sve.is-fullscreen .monitor-grid { gap:12px; }
    .iamccs-sve.is-fullscreen .screen .empty { inset:26px; }
    .iamccs-sve.is-fullscreen .timeline-wrap { grid-template-rows:40px minmax(0,1fr) 32px; }
    .iamccs-sve.is-fullscreen .master-lane-fixed { height:32px; }
    .iamccs-sve.is-fullscreen .lane.master-audio { min-height:30px; }
    .iamccs-sve.is-fullscreen .lane.master-audio .lane-head { padding:3px 6px; font-size:10px; }
    .iamccs-sve.is-fullscreen .lane.master-audio .lane-head .kind { display:none; }
    .iamccs-sve.is-fullscreen .master-lane-fixed .lane-body { min-height:30px; }
    .iamccs-sve.is-fullscreen .lane.master-audio .clip { top:4px; height:22px; }
    .iamccs-sve.is-fullscreen .lane.master-audio .clip-title,
    .iamccs-sve.is-fullscreen .lane.master-audio .clip-preview-label { display:none; }
  `;
}

function makeButton(label, fn, cls = "") {
  const b = document.createElement("button");
  b.textContent = label;
  if (cls) b.className = cls;
  b.addEventListener("click", (event) => {
    event.preventDefault();
    b.classList.add("on");
    b.classList.remove("tap-feedback");
    void b.offsetWidth;
    b.classList.add("tap-feedback");
    setTimeout(() => {
      if (!b.dataset.stickyOn) b.classList.remove("on");
      b.classList.remove("tap-feedback");
    }, 180);
    try {
      const result = fn?.(b);
      if (result?.catch) result.catch((error) => {
        console.warn("[IAMCCS ShotboardVideoEditorV1] button action failed", error);
      });
    } catch (error) {
      console.warn("[IAMCCS ShotboardVideoEditorV1] button action failed", error);
    }
  });
  return b;
}

async function uploadEditorAudioFile(file, options = {}) {
  const body = new FormData();
  body.append("image", file);
  if (options.subfolder) body.append("subfolder", String(options.subfolder));
  body.append("type", String(options.type || "input"));
  body.append("overwrite", "false");
  const resp = await api.fetchApi("/upload/image", { method: "POST", body });
  if (!resp || resp.status !== 200) throw new Error(`upload failed: ${resp?.status || "no response"}`);
  const data = await resp.json();
  const filename = data?.name || file.name;
  const subfolder = data?.subfolder || "";
  return {
    path: subfolder ? `${subfolder}/${filename}` : filename,
    type: data?.type || "input",
    filename,
    subfolder,
  };
}

function audioFileDuration(file) {
  return new Promise((resolve) => {
    if (!file) {
      resolve(0);
      return;
    }
    const audio = document.createElement("audio");
    const url = URL.createObjectURL(file);
    const done = (value) => {
      try { URL.revokeObjectURL(url); } catch {}
      resolve(Math.max(0, Number(value) || 0));
    };
    audio.preload = "metadata";
    audio.onloadedmetadata = () => done(audio.duration);
    audio.onerror = () => done(0);
    audio.src = url;
  });
}

function pickEditorAudioFile() {
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "audio/*,.wav,.mp3,.m4a,.flac,.ogg,.aac,.aif,.aiff,.wma";
    input.style.display = "none";
    const finish = (file) => {
      try { input.remove(); } catch {}
      resolve(file || null);
    };
    input.addEventListener("change", () => finish(input.files?.[0] || null), { once: true });
    document.body.appendChild(input);
    input.click();
  });
}

function setNodeSize(node) {
  node.size = [...NODE_SIZE];
  node.min_size = [...NODE_SIZE];
  node.max_size = [...NODE_SIZE];
  node.resizable = false;
  try { node.setSize?.([...NODE_SIZE]); } catch {}
}

function removeExistingEditorDom(node) {
  if (!Array.isArray(node?.widgets)) return;
  const keep = [];
  for (const item of node.widgets) {
    const element = item?.element || item?.inputEl || item?.domElement || null;
    const isEditorWidget =
      item?.name === "Shotboard Video Editor V1" ||
      item?.type === "iamccs_shotboard_video_editor_v1" ||
      element?.classList?.contains?.("iamccs-sve") ||
      Boolean(element?.querySelector?.(".iamccs-sve"));
    if (isEditorWidget) {
      try { element?.remove?.(); } catch {}
      continue;
    }
    keep.push(item);
  }
  node.widgets = keep;
}

function installEditor(node, reason = "install") {
  if (!isEditor(node) || typeof node.addDOMWidget !== "function") return;
  if (node._iamccsSveReady && node._iamccsSveVersion === UI_VERSION) return;
  removeExistingEditorDom(node);
  node._iamccsSveReady = true;
  node._iamccsSveVersion = UI_VERSION;
  installStyle();
  hideRawWidgets(node);
  repairEditorHiddenWidgets(node);
  setNodeSize(node);

  const root = document.createElement("div");
  root.className = "iamccs-sve";
  root.dataset.iamccsSveVersion = UI_VERSION;

  let manifest = manifestFromNode(node);
  let playhead = Number(manifest.ui_state?.playhead || 0);
  let playing = false;
  let timer = null;
  let raf = 0;
  let playStartMs = 0;
  let playStartHead = 0;
  let sourceHead = 0;
  let sourcePlaying = false;
  let sourceRaf = 0;
  let sourceStartMs = 0;
  let sourceStartHead = 0;
  let currentSourceClipId = "";
  let currentAudioClipId = "";
  let currentAudioUrl = "";
  const waveformPeakJobs = new Set();
  let selectedClipId = "";
  let selectedTrackId = String(manifest.ui_state?.selected_track_id || "");
  let replaceProjectOnInject = false;
  let status = null;
  let fullscreenState = null;
  const audioEl = document.createElement("audio");
  audioEl.preload = "auto";
  audioEl.style.display = "none";

  const persist = () => {
    manifest.ui_state = manifest.ui_state || {};
    manifest.ui_state.playhead = playhead;
    manifest.ui_state.selected_clip_id = selectedClipId;
    manifest.ui_state.selected_track_id = selectedTrackId;
    manifest.ui_state.replace_project_on_inject = false;
    node.properties = node.properties || {};
    node.properties.iamccs_video_editor_replace_project = false;
    saveManifest(node, manifest);
  };

  const visibleLaneBodyWidth = () => {
    const ruler = root.querySelector(".meter-ruler");
    if (ruler?.clientWidth) return Math.max(240, ruler.clientWidth);
    const laneScroll = root.querySelector(".lane-scroll");
    const timelineWrap = root.querySelector(".timeline-wrap");
    const candidates = [
      Number(laneScroll?.clientWidth || 0) - LANE_HEAD_WIDTH,
      Number(timelineWrap?.clientWidth || 0) - LANE_HEAD_WIDTH,
      Number(root?.clientWidth || 0) - LANE_HEAD_WIDTH - 24,
      Number(node?.size?.[0] || 0) - LANE_HEAD_WIDTH - 48,
      1000,
    ];
    return Math.max(240, ...candidates.filter((value) => Number.isFinite(value) && value > 0));
  };

  const visibleTimelineScrollLeft = () => {
    const lanes = root.querySelector(".lane-scroll");
    return Math.max(0, Number(lanes?.scrollLeft || 0));
  };

  const elementScreenToCssScale = (element) => {
    if (!element) return 1;
    const rect = element.getBoundingClientRect?.();
    const cssWidth = Number(element.clientWidth || element.offsetWidth || 0);
    const screenWidth = Number(rect?.width || 0);
    if (!cssWidth || !screenWidth) return 1;
    return Math.max(0.01, screenWidth / cssWidth);
  };

  const timelineScreenToCssScale = () => {
    const ruler = root.querySelector(".meter-ruler");
    const lanes = root.querySelector(".lane-scroll");
    return elementScreenToCssScale(ruler || lanes);
  };

  const clientDeltaToTimelineCssPx = (deltaClientX) => {
    return Number(deltaClientX || 0) / timelineScreenToCssScale();
  };

  const timelineDuration = () => Math.max(
    20,
    Number(manifest.duration_seconds) || 0,
    manifestEndSeconds(manifest),
    Number(manifest.ui_state?.playhead || 0),
    Number(playhead || 0)
  );

  const timelineRulerDuration = () => {
    const base = timelineDuration();
    const viewportSecondsAtMinimumScale = visibleLaneBodyWidth() / Math.max(1, MIN_TIMELINE_PX_PER_SECOND);
    const span = Math.max(base, viewportSecondsAtMinimumScale);
    return Math.max(20, Math.ceil((span + TIMELINE_TAIL_PAD_SECONDS) * 2) / 2);
  };

  const pxPerSecond = () => Math.max(
    visibleLaneBodyWidth() / Math.max(1, timelineRulerDuration()),
    MIN_TIMELINE_PX_PER_SECOND
  );

  const timeToPx = (seconds, scale = pxPerSecond()) => Number(seconds || 0) * scale;

  const timelineInnerWidth = () => Math.max(
    visibleLaneBodyWidth(),
    Math.ceil(timelineRulerDuration() * pxPerSecond())
  );

  const syncRulerScroll = () => {
    const lanes = root.querySelector(".lane-scroll");
    const track = root.querySelector(".ruler-track");
    const masterBody = root.querySelector(".master-lane-fixed .lane-body");
    const offset = visibleTimelineScrollLeft();
    if (track) track.style.transform = `translateX(${-offset}px)`;
    if (masterBody) masterBody.style.transform = `translateX(${-offset}px)`;
  };

  const clipAtTime = (kind = "video") => {
    return (manifest.clips || [])
      .filter((clip) => clip.type === kind && playhead >= Number(clip.startTime || 0) && playhead < Number(clip.startTime || 0) + Number(clip.duration || 0))
      .sort((a, b) => Number(b.trackIndex || 0) - Number(a.trackIndex || 0))[0] || null;
  };

  const audioClipAtTime = (time = playhead) => {
    const clips = (manifest.clips || [])
      .filter((clip) => (
        clip.type === "audio" &&
        !clip.muted &&
        time >= Number(clip.startTime || 0) &&
        time < Number(clip.startTime || 0) + Number(clip.duration || 0) &&
        audioUrlForClip(manifest, clip)
      ))
      .sort((a, b) => {
        const am = isMasterClip(a) ? 1 : 0;
        const bm = isMasterClip(b) ? 1 : 0;
        if (am !== bm) return bm - am;
        return Number(b.trackIndex || 0) - Number(a.trackIndex || 0);
      });
    return clips[0] || null;
  };

  const updateAudioMeter = () => {
    const fill = root.querySelector(".audio-meter .meter-fill");
    const label = root.querySelector(".audio-meter .meter-label");
    if (!fill || !label) return;
    const clip = audioClipAtTime(playhead);
    let level = 0;
    if (clip) {
      const peaks = audioPeaksForClip(manifest, clip);
      if (peaks.length) {
        const local = Math.max(0, playhead - Number(clip.startTime || 0) + Number(clip.trimStart || 0));
        const sourceDuration = clipSourceDuration(manifest, clip);
        const index = Math.max(0, Math.min(peaks.length - 1, Math.floor((local / Math.max(0.001, sourceDuration)) * peaks.length)));
        level = Math.max(Number(peaks[index]?.rms || 0), Number(peaks[index]?.max || 0) * 0.65);
      } else if (!audioEl.paused) {
        level = 0.28;
      }
    }
    level = Math.max(0, Math.min(1, level));
    fill.style.width = `${Math.round(level * 100)}%`;
    label.textContent = level > 0.86 ? "RED" : level > 0.62 ? "YEL" : level > 0.02 ? "GRN" : "PK";
  };

  const syncAudioPlayback = (force = false) => {
    const clip = audioClipAtTime(playhead);
    if (!clip) {
      if (!audioEl.paused) audioEl.pause();
      currentAudioClipId = "";
      currentAudioUrl = "";
      return;
    }
    const url = audioUrlForClip(manifest, clip);
    const desiredTime = Math.max(0, Number(clip.trimStart || 0) + (playhead - Number(clip.startTime || 0)));
    const changed = currentAudioClipId !== clip.id || currentAudioUrl !== url;
    if (changed) {
      currentAudioClipId = clip.id;
      currentAudioUrl = url;
      audioEl.src = url;
    }
    audioEl.volume = Math.max(0, Math.min(1, Number(clip.volume ?? 1)));
    if (force || changed || Math.abs((audioEl.currentTime || 0) - desiredTime) > 0.18) {
      try { audioEl.currentTime = desiredTime; } catch {}
    }
    if (playing && audioEl.paused) {
      audioEl.play().catch((err) => {
        if (status) status.textContent = `Audio preview blocked: ${err?.message || err}`;
      });
    }
  };

  const selectedClip = () => (manifest.clips || []).find((clip) => clip.id === selectedClipId) || null;
  const trackIdForClip = (clip) => isMasterClip(clip) ? "AM" : String(clip?.trackId || "");
  const selectedLaneId = () => {
    const clip = selectedClip();
    return clip ? trackIdForClip(clip) : String(selectedTrackId || "");
  };
  const selectTrack = (track) => {
    selectedTrackId = String(track?.id || "");
    selectedClipId = "";
    persist();
    renderTimeline();
    if (status) status.textContent = `Selected track ${track?.name || track?.id || ""}.`;
  };

  const updateMonitor = () => {
    const sourceImg = root.querySelector(".source-img");
    const programImg = root.querySelector(".program-img");
    const sourceEmpty = root.querySelector(".source-empty");
    const programEmpty = root.querySelector(".program-empty");
    const selectedForSource = selectedClipId ? (manifest.clips || []).find((item) => item.id === selectedClipId) : null;
    const sourceClip = selectedForSource?.type === "video" ? selectedForSource : clipAtTime("video");
    const programClip = clipAtTime("video");
    if (!selectedClipId) currentSourceClipId = sourceClip?.id || "";
    const sourceDuration = sourceClip ? Number(sourceClip.duration || 0) : 0;
    if (selectedClipId && sourceClip?.id && currentSourceClipId !== sourceClip.id) {
      currentSourceClipId = sourceClip.id;
      sourceHead = 0;
    }
    const sourceLocal = sourceClip
      ? (selectedClipId ? Math.max(0, Math.min(sourceDuration, sourceHead)) : Math.max(0, playhead - Number(sourceClip.startTime || 0)))
      : 0;
    const sourceUrl = sourceClip ? imageForClip(manifest, sourceClip, sourceLocal) : "";
    const programUrl = programClip ? imageForClip(manifest, programClip, Math.max(0, playhead - Number(programClip.startTime || 0))) : "";
    const setMonitorImage = (img, empty, url) => {
      if (!img) return;
      if (!url) {
        img.removeAttribute("src");
        img.style.display = "none";
        if (empty) empty.style.display = "flex";
        return;
      }
      img.onerror = () => {
        img.removeAttribute("src");
        img.style.display = "none";
        if (empty) empty.style.display = "flex";
      };
      img.onload = () => {
        img.style.display = "block";
        if (empty) empty.style.display = "none";
      };
      img.src = url;
    };
    setMonitorImage(sourceImg, sourceEmpty, sourceUrl);
    setMonitorImage(programImg, programEmpty, programUrl);
    const clock = root.querySelector(".clock");
    if (clock) clock.textContent = `${fmtTime(playhead)} / ${fmtTime(timelineDuration())}`;
    const rulerHead = root.querySelector(".ruler-track > .playhead");
    const lanesHead = root.querySelector(".lanes > .playhead");
    const masterHead = root.querySelector(".master-lane-fixed > .playhead");
    const scrollLeft = visibleTimelineScrollLeft();
    const x = timeToPx(playhead);
    if (rulerHead) rulerHead.style.left = `${x}px`;
    if (lanesHead) lanesHead.style.left = `${LANE_HEAD_WIDTH + x}px`;
    if (masterHead) masterHead.style.left = `${LANE_HEAD_WIDTH + x - scrollLeft}px`;
    syncRulerScroll();
    updateAudioMeter();
  };

  const renderPool = () => {
    const list = root.querySelector(".pool-list");
    if (!list) return;
    list.innerHTML = "";
    const assets = Object.values(manifest.assets || {}).sort((a, b) => Number(a.takeIndex || 0) - Number(b.takeIndex || 0));
    for (const asset of assets) {
      if (asset.type !== "video") continue;
      const item = document.createElement("div");
      item.className = "pool-item";
      item.textContent = `${asset.timelineId || "T??"} / ${asset.audioLane || "A?"}  ${fmtTime(asset.duration || 0)}  ${asset.frames || 0}f`;
      item.addEventListener("click", () => {
        selectedClipId = `clip_T${String(asset.takeIndex || 1).padStart(2, "0")}_V`;
        selectedTrackId = "V1";
        currentSourceClipId = selectedClipId;
        sourceHead = 0;
        stopSourcePlayback();
        root.querySelectorAll(".pool-item").forEach((el) => el.classList.remove("on"));
        item.classList.add("on");
        persist();
        renderTimeline();
      });
      list.appendChild(item);
    }
    if (!assets.length) {
      const empty = document.createElement("div");
      empty.className = "pool-item";
      empty.textContent = "No rendered takes collected yet.";
      list.appendChild(empty);
    }
  };

  function drawWaveCanvas(canvas, peaks, errorText = "") {
    const width = 700;
    const height = 46;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "rgba(29,92,139,.55)";
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = "rgba(234,247,255,.72)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();
    if (!Array.isArray(peaks) || !peaks.length) {
      ctx.fillStyle = "rgba(255,255,255,.55)";
      ctx.font = "10px Arial";
      ctx.fillText(errorText ? "waveform decode failed" : "loading real waveform", 8, 28);
      return;
    }
    const barWidth = Math.max(1, width / peaks.length);
    ctx.fillStyle = "rgba(191,233,255,.9)";
    ctx.strokeStyle = "rgba(255,255,255,.62)";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    peaks.forEach((peak, index) => {
      const max = Math.max(0.01, Math.min(1, Number(peak.max || 0)));
      const rms = Math.max(0.005, Math.min(1, Number(peak.rms || max * 0.55)));
      const x = index * barWidth;
      const amp = max * 20;
      ctx.fillRect(x, (height / 2) - amp, Math.max(1, barWidth * 0.82), amp * 2);
      const y = (height / 2) - (rms * 13);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  async function generateRealWaveformPeaks(clip) {
    const asset = assetForClip(manifest, clip);
    const urls = audioCandidateUrlsForClip(manifest, clip);
    const jobKey = String(clip?.assetId || clip?.id || urls.join("|") || "");
    if (!asset || !urls.length || !jobKey || waveformPeakJobs.has(jobKey)) return;
    waveformPeakJobs.add(jobKey);
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) throw new Error("WebAudio unavailable");
      let lastError = null;
      for (const url of urls) {
        try {
          const bytes = await fetchArrayBufferWithTimeout(url);
          const ctx = new AudioCtx();
          const buffer = await ctx.decodeAudioData(bytes.slice(0));
          try { await ctx.close?.(); } catch {}
          const bins = 420;
          const channels = Array.from({ length: buffer.numberOfChannels }, (_, index) => buffer.getChannelData(index));
          const samples = buffer.length;
          const peaks = [];
          for (let bin = 0; bin < bins; bin += 1) {
            const start = Math.floor((bin / bins) * samples);
            const end = Math.max(start + 1, Math.floor(((bin + 1) / bins) * samples));
            let max = 0;
            let sum = 0;
            let count = 0;
            for (let i = start; i < end; i += 1) {
              let value = 0;
              for (const channel of channels) value += Math.abs(channel[i] || 0);
              value /= Math.max(1, channels.length);
              max = Math.max(max, value);
              sum += value * value;
              count += 1;
            }
            peaks.push({ max, rms: Math.sqrt(sum / Math.max(1, count)) });
          }
          asset.waveform_peaks = peaks;
          asset.waveformReal = true;
          asset.waveform_source = "webaudio_decode";
          asset.waveform_source_url = url;
          delete asset.waveform_error;
          persist();
          renderTimeline();
          return;
        } catch (error) {
          lastError = error;
        }
      }
      throw lastError || new Error("no readable audio preview url");
    } catch (error) {
      console.warn("[IAMCCS ShotboardVideoEditorV1] real waveform decode failed", error);
      if (asset) asset.waveform_error = String(error?.message || error || "decode failed");
      renderTimeline();
    } finally {
      waveformPeakJobs.delete(jobKey);
    }
  }

  const renderWave = (clip) => {
    const peaks = audioPeaksForClip(manifest, clip);
    const asset = assetForClip(manifest, clip);
    const canvas = document.createElement("canvas");
    canvas.className = "wave real-wave";
    drawWaveCanvas(canvas, peaks, asset?.waveform_error || "");
    if ((!peaks.length && !asset?.waveform_error) || (peaks.length && shouldRefreshRealWaveform(manifest, clip))) generateRealWaveformPeaks(clip);
    return canvas;
  };

  const renderClip = (laneBody, clip) => {
    const scale = pxPerSecond();
    const el = document.createElement("div");
    const isMasterAudio = isMasterClip(clip);
    const sourceDuration = clipEditableDuration(manifest, clip);
    el.className = `clip ${clip.type === "audio" ? "audio" : ""}${isMasterAudio ? " master_audio" : ""}${selectedClipId === clip.id ? " selected" : ""}`;
    el.dataset.clipId = String(clip.id || "");
    el.dataset.audioLane = String(clip.audioLane || clip.trackId || "");
    el.style.left = `${Number(clip.startTime || 0) * scale}px`;
    el.style.width = `${Math.max(32, Number(clip.duration || 1) * scale)}px`;
    el.title = `Drag center to move. Drag edges to trim. Source ${fmtTime(sourceDuration)}.`;
    const title = document.createElement("div");
    title.className = "clip-title";
    title.textContent = isMasterAudio
      ? "MASTER AUDIO"
      : clip.type === "audio"
        ? `${clip.timelineId || ""} ${clip.audioLane || clip.trackId || "A?"} AUDIO`
        : `${clip.timelineId || ""} ${clip.type}`;
    el.appendChild(title);
    if (clip.type === "video") {
      const film = document.createElement("div");
      film.className = "film";
      const count = Math.max(1, Math.ceil((Number(clip.duration || 1) * scale) / 54));
      for (let i = 0; i < count; i++) {
        const img = document.createElement("img");
        const local = (i / Math.max(1, count)) * Math.max(0.001, Number(clip.duration || 1));
        img.src = imageForClip(manifest, clip, local);
        film.appendChild(img);
      }
      el.appendChild(film);
    } else {
      el.appendChild(renderWave(clip));
    }
    ["left", "right"].forEach((edge) => {
      const h = document.createElement("div");
      h.className = `handle ${edge}`;
      h.addEventListener("pointerdown", (event) => startTrim(event, clip, edge));
      el.appendChild(h);
    });
    const previewLabel = document.createElement("div");
    previewLabel.className = "clip-preview-label";
    previewLabel.textContent = clip.type === "audio" && !isMasterAudio
      ? `${clip.audioLane || clip.trackId || "A?"}  ${fmtTime(Number(clip.startTime || 0))} -> ${fmtTime(Number(clip.startTime || 0) + Number(clip.duration || 0))}`
      : `${fmtTime(Number(clip.startTime || 0))} -> ${fmtTime(Number(clip.startTime || 0) + Number(clip.duration || 0))}`;
    el.appendChild(previewLabel);
    el.addEventListener("pointerdown", (event) => startDrag(event, clip));
    laneBody.appendChild(el);
  };

  function clipElement(clip) {
    const id = String(clip?.id || "");
    if (!id) return null;
    try {
      return root.querySelector(`.clip[data-clip-id="${CSS.escape(id)}"]`);
    } catch {
      return Array.from(root.querySelectorAll(".clip")).find((item) => item.dataset.clipId === id) || null;
    }
  }

  function updateClipElementPreview(clip, mode = "") {
    const el = clipElement(clip);
    if (!el) return;
    const scale = pxPerSecond();
    el.style.left = `${Number(clip.startTime || 0) * scale}px`;
    el.style.width = `${Math.max(32, Number(clip.duration || 0.1) * scale)}px`;
    el.classList.add("is-dragging");
    const label = el.querySelector(".clip-preview-label");
    if (label) {
      const start = Number(clip.startTime || 0);
      const end = start + Number(clip.duration || 0);
      const sourceIn = Number(clip.trimStart || 0);
      const sourceOut = Number(clip.trimEnd || sourceIn + Number(clip.duration || 0));
      label.textContent = `${mode ? `${mode} ` : ""}${fmtTime(start)}-${fmtTime(end)} / src ${fmtTime(sourceIn)}-${fmtTime(sourceOut)}`;
    }
  }

  function finishClipElementPreview(clip) {
    const el = clipElement(clip);
    if (!el) return;
    el.classList.remove("is-dragging");
  }

  function startDrag(event, clip) {
    if (event.target?.classList?.contains("handle")) return;
    event.preventDefault();
    event.stopPropagation();
    selectedClipId = clip.id;
    selectedTrackId = trackIdForClip(clip);
    const startX = event.clientX;
    const startTime = Number(clip.startTime || 0);
    const pointerTarget = event.currentTarget || event.target;
    try { pointerTarget?.setPointerCapture?.(event.pointerId); } catch {}
    const move = (ev) => {
      ev.preventDefault?.();
      const dt = clientDeltaToTimelineCssPx(ev.clientX - startX) / pxPerSecond();
      clip.startTime = Math.max(0, startTime + dt);
      playhead = clip.startTime;
      updateClipElementPreview(clip, "MOVE");
      updateMonitor();
      if (status) status.textContent = `Move clip: ${fmtTime(clip.startTime)} -> ${fmtTime(clip.startTime + Number(clip.duration || 0))}.`;
    };
    const up = (ev) => {
      try { pointerTarget?.releasePointerCapture?.(event.pointerId); } catch {}
      window.removeEventListener("pointermove", move, true);
      window.removeEventListener("pointerup", up, true);
      window.removeEventListener("pointercancel", up, true);
      finishClipElementPreview(clip);
      persist();
      renderTimeline();
    };
    window.addEventListener("pointermove", move, { passive: false, capture: true });
    window.addEventListener("pointerup", up, { passive: false, capture: true });
    window.addEventListener("pointercancel", up, { passive: false, capture: true });
  }

  function startTrim(event, clip, edge) {
    event.preventDefault();
    event.stopPropagation();
    selectedClipId = clip.id;
    selectedTrackId = trackIdForClip(clip);
    const startX = event.clientX;
    const startStart = Number(clip.startTime || 0);
    const startDur = Number(clip.duration || 1);
    const startTrimIn = Math.max(0, Number(clip.trimStart || 0));
    const startTrimOut = Math.max(startTrimIn + 0.1, Number(clip.trimEnd || startTrimIn + startDur));
    const sourceDuration = clipEditableDuration(manifest, clip);
    const fixedTimelineEnd = startStart + startDur;
    const pointerTarget = event.currentTarget || event.target;
    try { pointerTarget?.setPointerCapture?.(event.pointerId); } catch {}
    const move = (ev) => {
      ev.preventDefault?.();
      const dt = clientDeltaToTimelineCssPx(ev.clientX - startX) / pxPerSecond();
      if (edge === "left") {
        const maxStart = fixedTimelineEnd - 0.1;
        const minStart = Math.max(0, startStart - startTrimIn);
        const nextStart = Math.max(minStart, Math.min(maxStart, startStart + dt));
        const nextTrimIn = Math.max(0, Math.min(startTrimOut - 0.1, startTrimIn + (nextStart - startStart)));
        clip.startTime = nextStart;
        clip.trimStart = nextTrimIn;
        clip.duration = Math.max(0.1, fixedTimelineEnd - nextStart);
        clip.trimEnd = Math.max(nextTrimIn + 0.1, Math.min(sourceDuration, startTrimOut));
      } else {
        const maxDuration = Math.max(0.1, sourceDuration - startTrimIn);
        clip.duration = Math.max(0.1, Math.min(maxDuration, startDur + dt));
        clip.trimStart = startTrimIn;
        clip.trimEnd = Math.max(startTrimIn + 0.1, Math.min(sourceDuration, startTrimIn + clip.duration));
      }
      playhead = edge === "left"
        ? clip.startTime
        : Math.max(clip.startTime, clip.startTime + clip.duration - 0.001);
      updateClipElementPreview(clip, edge === "left" ? "TRIM IN" : "TRIM OUT");
      updateMonitor();
      if (status) status.textContent = `${edge === "left" ? "Trim in" : "Trim out"}: ${fmtTime(clip.startTime)} -> ${fmtTime(clip.startTime + clip.duration)}.`;
    };
    const up = () => {
      try { pointerTarget?.releasePointerCapture?.(event.pointerId); } catch {}
      window.removeEventListener("pointermove", move, true);
      window.removeEventListener("pointerup", up, true);
      window.removeEventListener("pointercancel", up, true);
      finishClipElementPreview(clip);
      persist();
      renderTimeline();
    };
    window.addEventListener("pointermove", move, { passive: false, capture: true });
    window.addEventListener("pointerup", up, { passive: false, capture: true });
    window.addEventListener("pointercancel", up, { passive: false, capture: true });
  }

  const renderTimeline = () => {
    manifest = manifestFromNode(node);
    let clamped = false;
    for (const clip of manifest.clips || []) {
      clamped = clampClipToEditableDuration(manifest, clip) || clamped;
    }
    if (clamped) saveManifest(node, manifest);
    const laneScroll = root.querySelector(".lane-scroll");
    const lanes = root.querySelector(".lanes");
    const masterLane = root.querySelector(".master-lane-fixed");
    const ruler = root.querySelector(".meter-ruler");
    if (!laneScroll || !lanes || !masterLane || !ruler) return;
    const scale = pxPerSecond();
    const innerWidth = timelineInnerWidth();
    const duration = Math.max(timelineRulerDuration(), innerWidth / Math.max(1, scale));
    lanes.style.width = `${LANE_HEAD_WIDTH + innerWidth}px`;
    lanes.style.minWidth = `${LANE_HEAD_WIDTH + innerWidth}px`;
    masterLane.style.width = "100%";
    ruler.innerHTML = "";
    const rulerTrack = document.createElement("div");
    rulerTrack.className = "ruler-track";
    rulerTrack.style.width = `${innerWidth}px`;
    const tickCount = Math.ceil(duration * 2);
    for (let i = 0; i <= tickCount; i++) {
      const seconds = i / 2;
      const isMajor = i % 2 === 0;
      const tick = document.createElement("div");
      tick.className = `tick ${isMajor ? "major" : "minor"}`;
      tick.style.left = `${timeToPx(seconds, scale)}px`;
      tick.innerHTML = isMajor
        ? `${seconds}s<span class="frame">${Math.round(seconds * Number(manifest.fps || widget(node, "fps")?.value || 24))}f</span>`
        : `<span class="half">${seconds.toFixed(1)}s</span>`;
      rulerTrack.appendChild(tick);
    }
    const ph = document.createElement("div");
    ph.className = "playhead";
    rulerTrack.appendChild(ph);
    ruler.appendChild(rulerTrack);
    lanes.innerHTML = "";
    masterLane.innerHTML = "";
    const renderLane = (track, target, clipFilter = null) => {
      const lane = document.createElement("div");
      const laneId = isMasterTrack(track) ? "AM" : String(track.id || "");
      const isSelectedLane = String(selectedLaneId() || "") === laneId;
      lane.className = `lane ${isMasterTrack(track) ? "master-audio" : ""}${isSelectedLane ? " is-selected" : ""}`;
      lane.dataset.trackId = laneId;
      const head = document.createElement("div");
      head.className = "lane-head";
      head.innerHTML = `${track.name || track.id}<span class="kind">${track.kind || ""} lane</span>`;
      head.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        selectTrack(track);
      });
      const body = document.createElement("div");
      body.className = "lane-body";
      body.style.width = `${innerWidth}px`;
      body.style.minWidth = `${innerWidth}px`;
      body.style.setProperty("--major-grid", `${scale}px`);
      body.style.setProperty("--half-grid", `${scale / 2}px`);
      body.style.backgroundPosition = "0 0, 0 0, 0 0";
      const clips = (manifest.clips || []).filter((clip) => (
        typeof clipFilter === "function" ? clipFilter(clip) : clip.trackId === track.id && !isMasterClip(clip)
      ));
      body.addEventListener("click", (event) => {
        if (event.target?.closest?.(".clip")) return;
        selectTrack(track);
      });
      for (const clip of clips) renderClip(body, clip);
      lane.appendChild(head);
      lane.appendChild(body);
      target.appendChild(lane);
    };
    for (const track of pairedTimelineTracks(manifest)) {
      renderLane(track, lanes);
    }
    const lanePh = document.createElement("div");
    lanePh.className = "playhead";
    lanes.appendChild(lanePh);
    const masterTrack = normalizeManifestTracks(manifest).tracks.find((track) => isMasterTrack(track)) || { id: "AM", name: "MASTER AUDIO", kind: "master_audio" };
    renderLane(masterTrack, masterLane, (clip) => isMasterClip(clip));
    const masterPh = document.createElement("div");
    masterPh.className = "playhead";
    masterLane.appendChild(masterPh);
    laneScroll.onscroll = syncRulerScroll;
    updateMonitor();
    renderPool();
    if (status) status.textContent = `Ready. Clips ${manifest.clips?.length || 0} / duration ${fmtTime(manifest.duration_seconds || 0)}.`;
  };

  const clientXToTimelineTime = (clientX, origin = "ruler", clientY = null, eventTarget = null) => {
    const lanes = root.querySelector(".lane-scroll");
    const ruler = root.querySelector(".meter-ruler");
    const scrollLeft = visibleTimelineScrollLeft();
    let localX = 0;
    if (origin === "lanes" && lanes) {
      const rect = lanes.getBoundingClientRect();
      const scale = elementScreenToCssScale(lanes);
      localX = ((Number(clientX || 0) - Number(rect?.left || 0)) / scale) - LANE_HEAD_WIDTH + scrollLeft;
    } else {
      const rect = ruler?.getBoundingClientRect();
      const scale = elementScreenToCssScale(ruler);
      localX = ((Number(clientX || 0) - Number(rect?.left || 0)) / scale) + scrollLeft;
    }
    const x = Math.max(0, localX);
    const duration = timelineRulerDuration();
    return Math.max(0, Math.min(duration, x / pxPerSecond()));
  };

  const seekFromPointer = (event, origin = "ruler") => {
    event?.preventDefault?.();
    playhead = clientXToTimelineTime(event.clientX, origin, event.clientY, event.target);
    persist();
    syncAudioPlayback(true);
    updateMonitor();
    if (status) status.textContent = `Scrub ${fmtTime(playhead)} / ${fmtTime(timelineRulerDuration())}.`;
  };

  function bindScrubber(element, origin) {
    if (!element) return;
    element.addEventListener("pointerdown", (event) => {
      if (event.target?.closest?.(".clip,.lane-head,.ruler-render-controls")) return;
      const target = event.currentTarget;
      seekFromPointer(event, origin);
      const move = (ev) => seekFromPointer(ev, origin);
      const up = () => {
        try { target?.releasePointerCapture?.(event.pointerId); } catch {}
        window.removeEventListener("pointermove", move, true);
        window.removeEventListener("pointerup", up, true);
        window.removeEventListener("pointercancel", up, true);
        document.body.style.userSelect = "";
      };
      document.body.style.userSelect = "none";
      try { target?.setPointerCapture?.(event.pointerId); } catch {}
      window.addEventListener("pointermove", move, { passive: false, capture: true });
      window.addEventListener("pointerup", up, { passive: false, capture: true });
      window.addEventListener("pointercancel", up, { passive: false, capture: true });
    });
  }

  function play() {
    if (playing) return;
    playing = true;
    root.querySelector(".play-btn")?.classList.add("on");
    playStartMs = performance.now();
    playStartHead = playhead;
    syncAudioPlayback(true);
    const tick = (now) => {
      if (!playing) return;
      playhead = playStartHead + ((now - playStartMs) / 1000);
      const end = timelineDuration();
      if (playhead >= end) {
        playhead = end;
        stop();
        return;
      }
      syncAudioPlayback(false);
      updateMonitor();
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
  }

  function stop() {
    playing = false;
    root.querySelector(".play-btn")?.classList.remove("on");
    if (timer) clearInterval(timer);
    timer = null;
    if (raf) cancelAnimationFrame(raf);
    raf = 0;
    if (!audioEl.paused) audioEl.pause();
    persist();
    updateMonitor();
  }

  function nudge(delta) {
    playhead = Math.max(0, Math.min(timelineDuration(), playhead + delta));
    persist();
    syncAudioPlayback(true);
    updateMonitor();
  }

  function addManual(kind) {
    const path = window.prompt(`Path ${kind.toUpperCase()} da aggiungere alla timeline:`);
    if (!path) return;
    const index = (manifest.clips || []).length + 1;
    const assetId = `manual_${kind}_${Date.now()}`;
    manifest.assets[assetId] = { id: assetId, type: kind, path, duration: 5, manual: true };
    manifest.clips.push({
      id: `clip_manual_${index}`,
      assetId,
      type: kind,
      startTime: Number(manifest.duration_seconds || 0),
      duration: 5,
      trimStart: 0,
      trimEnd: 5,
      trackId: kind === "video" ? "V1" : "A1",
      trackIndex: kind === "video" ? 0 : 5,
      manual: true,
      muted: false,
      volume: 1,
    });
    manifest.duration_seconds = Number(manifest.duration_seconds || 0) + 5;
    selectedTrackId = kind === "video" ? "V1" : "A1";
    selectedClipId = `clip_manual_${index}`;
    persist();
    renderTimeline();
  }

  async function runVideoEditorParkingPurge(button, sessionKey) {
    if (button) {
      button.disabled = true;
      button.textContent = "Purging...";
    }
    try {
      const response = await api.fetchApi("/api/iamccs/cine/video_editor/purge_parking", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_key: sessionKey, all_sessions: true }),
      });
      const result = await response.json().catch(() => ({}));
      if (!response.ok || result?.error) {
        if (response.status === 404) throw new Error("Purge non caricato dal server: riavvia ComfyUI e ricarica il frontend.");
        throw new Error(result?.error || `HTTP ${response.status}`);
      }
      const files = Number(result.deleted_files || 0);
      const gb = Number(result.deleted_bytes || 0) / (1024 ** 3);
      const failed = Number(result.failed_files || 0);
      const remaining = Number(result.remaining_files || 0);
      if (status) status.textContent = failed || remaining
        ? `Parking partially purged: ${files} deleted / ${failed} failed / ${remaining} remaining. Folder: ${result.folder || "parking"}.`
        : `Parking purged: ${files} files / ${gb.toFixed(2)} GB. Folder: ${result.folder || "parking"}.`;
    } catch (error) {
      if (status) status.textContent = `Parking purge failed: ${error?.message || error}`;
      console.warn("[IAMCCS ShotboardVideoEditorV1] parking purge failed", error);
    } finally {
      if (button) {
        button.disabled = false;
        button.textContent = "Purge";
      }
    }
  }

  function purgeVideoEditorParking(button) {
    const sessionKey = String(
      manifest?.session_key
      || widget(node, "session_key")?.value
      || "shotboard_video_editor_v1"
    ).trim() || "shotboard_video_editor_v1";
    root._iamccsPurgeConfirm?.remove?.();
    const panel = document.createElement("div");
    panel.className = "iamccs-sve-purge-confirm";
    panel.innerHTML = `<strong>Purge parked generations?</strong><span class="detail">All IAMCCS parking sessions will be deleted. Existing takes may stop previewing/rendering until regenerated.</span>`;
    const actions = document.createElement("div");
    actions.className = "actions";
    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.textContent = "Cancel";
    const confirm = document.createElement("button");
    confirm.type = "button";
    confirm.className = "confirm";
    confirm.textContent = "Confirm purge";
    actions.append(cancel, confirm);
    panel.appendChild(actions);
    document.body.appendChild(panel);
    root._iamccsPurgeConfirm = panel;
    const close = () => {
      panel.remove();
      if (root._iamccsPurgeConfirm === panel) root._iamccsPurgeConfirm = null;
    };
    cancel.onclick = close;
    confirm.onclick = () => {
      close();
      runVideoEditorParkingPurge(button, sessionKey);
    };
    const rect = button?.getBoundingClientRect?.();
    const place = () => {
      if (!panel.isConnected) return;
      const width = panel.offsetWidth || 300;
      const left = Math.max(8, Math.min(Number(rect?.left || 8), window.innerWidth - width - 8));
      const top = Math.max(8, Math.min(Number(rect?.bottom || 8) + 7, window.innerHeight - panel.offsetHeight - 8));
      panel.style.left = `${left}px`;
      panel.style.top = `${top}px`;
    };
    place();
    requestAnimationFrame(place);
  }

  async function addMasterAudioClip() {
    const file = await pickEditorAudioFile();
    if (!file) {
      if (status) status.textContent = "Add MA annullato: nessun file selezionato.";
      return;
    }
    if (status) status.textContent = `Uploading MA ${file.name || "audio"}...`;
    const [uploaded, measuredDuration] = await Promise.all([
      uploadEditorAudioFile(file, { subfolder: "IAMCCS_video_editor_audio", type: "input" }),
      audioFileDuration(file),
    ]);
    const renderNode = findLinkedRenderNode();
    manifest.assets = manifest.assets && typeof manifest.assets === "object" ? manifest.assets : {};
    manifest.clips = Array.isArray(manifest.clips) ? manifest.clips : [];
    const duration = Number(measuredDuration || 0) > 0
      ? Math.max(0.1, Number(measuredDuration || 0))
      : Math.max(0.1, Number(timelineDuration() || 0), 20);
    const masterAsset = {
      id: "master_excerpt",
      type: "audio",
      role: "master_excerpt",
      takeIndex: 0,
      timelineId: "MASTER",
      audioLane: "MASTER",
      path: uploaded.path,
      audioFile: uploaded.path,
      audioUploadType: uploaded.type || "input",
      audio_preview_file: uploaded.filename || file.name,
      audio_preview_subfolder: uploaded.subfolder || "",
      audio_preview_type: uploaded.type || "input",
      preview_type: uploaded.type || "input",
      preview_subfolder: uploaded.subfolder || "",
      file_type: uploaded.type || "input",
      fileName: uploaded.filename || file.name,
      filename: uploaded.filename || file.name,
      mime: file.type || "audio/wav",
      size: Number(file.size || 0),
      duration,
      duration_seconds: duration,
      manual: true,
    };
    manifest.master_excerpt = JSON.parse(JSON.stringify(masterAsset));
    manifest.assets.master_excerpt = JSON.parse(JSON.stringify(masterAsset));
    manifest.clips = manifest.clips.filter((clip) => !isMasterClip(clip));
    manifest.clips.push({
      id: "clip_MASTER_AUDIO",
      assetId: "master_excerpt",
      type: "audio",
      takeIndex: 0,
      timelineId: "MASTER",
      audioLane: "MASTER",
      startTime: 0,
      duration,
      trimStart: 0,
      trimEnd: duration,
      trackId: "AM",
      trackIndex: 10,
      muted: false,
      volume: 1,
      linkedClipIds: [],
      role: "master_excerpt",
      manual: Boolean(masterAsset.manual),
    });
    setEditorRenderAudioPolicy(manifest, "use_master_audio");
    ensureMasterAudioLaneClip(manifest);
    selectedClipId = "clip_MASTER_AUDIO";
    selectedTrackId = "AM";
    playhead = 0;
    persist();
    renderTimeline();
    if (renderNode) {
      forceNodeWidgetValue(renderNode, "editor_manifest_json", 0, JSON.stringify(manifest, null, 2));
      forceNodeWidgetValue(renderNode, "audio_policy", 1, "use_master_audio");
    }
    if (status) status.textContent = `MA added. Master lane has priority over T-lane audio chunks. Duration ${fmtTime(duration)}.`;
  }

  function deleteSelectedClip() {
    const clip = selectedClip();
    if (!clip) {
      if (status) status.textContent = "Select a clip before deleting.";
      return;
    }
    const wasMaster = isMasterClip(clip);
    const removedId = String(clip.id || "");
    const assetId = String(clip.assetId || "");
    manifest.clips = (manifest.clips || []).filter((item) => item && item.id !== clip.id);
    if (wasMaster) {
      delete manifest.master_excerpt;
      delete manifest.master_audio;
      if (manifest.assets && typeof manifest.assets === "object") {
        delete manifest.assets.master_excerpt;
        delete manifest.assets.master_audio;
      }
      setEditorRenderAudioPolicy(manifest, "concat_clip_audio");
    } else if (assetId && manifest.assets && typeof manifest.assets === "object") {
      const stillUsed = (manifest.clips || []).some((item) => String(item?.assetId || "") === assetId);
      if (!stillUsed && manifest.assets[assetId]?.manual) delete manifest.assets[assetId];
    }
    selectedClipId = "";
    selectedTrackId = "";
    currentSourceClipId = "";
    currentAudioClipId = "";
    persist();
    renderTimeline();
    if (status) status.textContent = `Deleted selected clip ${removedId || ""}${wasMaster ? ". MA priority disabled." : "."}`;
  }

  function splitSelectedClip() {
    const clip = selectedClip();
    if (!clip) {
      if (status) status.textContent = "Select a clip before cutting.";
      return;
    }
    const start = Number(clip.startTime || 0);
    const duration = Number(clip.duration || 0);
    const end = start + duration;
    if (playhead <= start + 0.05 || playhead >= end - 0.05) {
      if (status) status.textContent = "Move playhead inside the selected clip before cutting.";
      return;
    }
    const leftDuration = playhead - start;
    const rightDuration = end - playhead;
    const right = JSON.parse(JSON.stringify(clip));
    right.id = `${clip.id}_cut_${Date.now()}`;
    right.startTime = playhead;
    right.duration = rightDuration;
    right.trimStart = Number(clip.trimStart || 0) + leftDuration;
    right.trimEnd = Number(clip.trimEnd || Number(clip.trimStart || 0) + duration);
    clip.duration = leftDuration;
    clip.trimEnd = Number(clip.trimStart || 0) + leftDuration;
    manifest.clips.push(right);
    selectedClipId = right.id;
    selectedTrackId = trackIdForClip(right);
    persist();
    renderTimeline();
    if (status) status.textContent = `Cut created at ${fmtTime(playhead)}.`;
  }

  function trimSelectedToPlayhead(edge = "out") {
    const clip = selectedClip();
    if (!clip) {
      if (status) status.textContent = "Select a clip before trimming.";
      return;
    }
    const start = Number(clip.startTime || 0);
    const end = start + Number(clip.duration || 0);
    if (playhead <= start || playhead >= end) {
      if (status) status.textContent = "Playhead must be inside selected clip.";
      return;
    }
    if (edge === "in") {
      const delta = playhead - start;
      clip.startTime = playhead;
      clip.duration = Math.max(0.1, end - playhead);
      clip.trimStart = Math.max(0, Number(clip.trimStart || 0) + delta);
    } else {
      clip.duration = Math.max(0.1, playhead - start);
      clip.trimEnd = Number(clip.trimStart || 0) + clip.duration;
    }
    persist();
    renderTimeline();
    if (status) status.textContent = `${edge === "in" ? "Trim in" : "Trim out"} applied at ${fmtTime(playhead)}.`;
  }

  function findLinkedRenderNode() {
    const graph = node.graph || app.graph;
    if (!graph) return null;
    const storedId = node.properties?.iamccs_video_editor_render_node_id;
    if (storedId != null) {
      const found = graph.getNodeById?.(storedId) || (graph._nodes || []).find((item) => String(item?.id) === String(storedId));
      if (found && nodeTypeMatches(found, RENDER_TYPE)) return found;
    }
    const nodes = graph._nodes || [];
    return nodes.find((item) => {
      if (!nodeTypeMatches(item, RENDER_TYPE)) return false;
      const inputIndex = findSlotIndex(item, "input", "editor_manifest_json");
      const linkId = item.inputs?.[inputIndex]?.link;
      const link = linkId != null ? graph.links?.[linkId] : null;
      return String(link?.origin_id || "") === String(node.id || "");
    }) || null;
  }

  function createRenderNode() {
    const graph = node.graph || app.graph;
    const lite = window.LiteGraph || globalThis.LiteGraph;
    if (!graph || !lite?.createNode) return null;
    let renderNode = findLinkedRenderNode();
    if (!renderNode) {
      renderNode = lite.createNode(RENDER_TYPE);
      if (!renderNode) return null;
      renderNode.pos = [Number(node.pos?.[0] || 0) + Number(node.size?.[0] || NODE_SIZE[0]) + 80, Number(node.pos?.[1] || 0) + 80];
      graph.add(renderNode);
    }
    node.properties = node.properties || {};
    node.properties.iamccs_video_editor_render_node_id = renderNode.id;
    connectBySlotName(node, "editor_manifest_json", renderNode, "editor_manifest_json");
    return renderNode;
  }

  function downstreamNodeIds(startNode) {
    const graph = node.graph || app.graph;
    const wanted = new Set([String(startNode?.id ?? "")]);
    if (!graph || !startNode) return wanted;
    let changed = true;
    while (changed) {
      changed = false;
      for (const link of graphLinksArray(graph)) {
        const origin = String(linkOriginId(link) ?? "");
        const target = String(linkTargetId(link) ?? "");
        if (origin && target && wanted.has(origin) && !wanted.has(target)) {
          wanted.add(target);
          changed = true;
        }
      }
    }
    return wanted;
  }

  function manualRenderPrompt(renderNode) {
    if (!manifestMasterAudioAsset(manifest)) {
      copyMasterAudioBundle(manifest, manifestFromRenderNode(renderNode));
    }
    ensureMasterAudioLaneClip(manifest);
    const audioPolicy = editorRenderAudioPolicy(manifest);
    return {
      class_type: RENDER_TYPE,
      inputs: {
        editor_manifest_json: JSON.stringify(manifest, null, 2),
        audio_policy: audioPolicy,
        fps_mode: "from_manifest",
        override_fps: Number(manifest.fps || widget(node, "fps")?.value || 24),
        tail_trim_frames_per_clip: Number(manifest.render_tail_trim_frames || 0),
      },
    };
  }

  async function queueRenderBranch(renderNode) {
    if (!renderNode || typeof api?.queuePrompt !== "function") return false;
    try {
      const comfyApp = node.graph?.comfyApp || window.app || app;
      const renderId = String(renderNode.id);
      let output = { [renderId]: manualRenderPrompt(renderNode) };
      let workflow = typeof app.graph?.serialize === "function"
        ? app.graph.serialize()
        : { nodes: [renderNode.serialize?.() || { id: renderNode.id, type: RENDER_TYPE }] };
      try {
        if (typeof comfyApp?.graphToPrompt === "function") {
          const full = await comfyApp.graphToPrompt();
          const fullOutput = full?.output || full?.prompt || {};
          const allowed = downstreamNodeIds(renderNode);
          const branchOutput = {};
          for (const [id, promptNode] of Object.entries(fullOutput || {})) {
            if (allowed.has(String(id))) branchOutput[id] = promptNode;
          }
          branchOutput[renderId] = manualRenderPrompt(renderNode);
          if (Object.keys(branchOutput).length > 1) {
            output = branchOutput;
            workflow = full?.workflow || workflow;
          }
        }
      } catch (branchError) {
        console.warn("[IAMCCS ShotboardVideoEditorV1] downstream render branch fallback to render-only", branchError);
      }
      await api.queuePrompt(0, { output, workflow });
      return true;
    } catch (error) {
      console.warn("[IAMCCS ShotboardVideoEditorV1] isolated render queue failed", error);
      if (status) status.textContent = `Render node prepared, queue failed: ${error?.message || error}`;
      return false;
    }
  }

  async function markRenderReady() {
    manifest.render_ready_at = Date.now() / 1000;
    manifest.render_tail_trim_frames = Math.max(0, Math.min(12, Number(manifest.render_tail_trim_frames || 0)));
    setEditorRenderAudioPolicy(manifest, editorRenderAudioPolicy(manifest));
    const renderNode = createRenderNode();
    if (!manifestMasterAudioAsset(manifest)) {
      copyMasterAudioBundle(manifest, manifestFromRenderNode(renderNode));
    }
    ensureMasterAudioLaneClip(manifest);
    persist();
    if (renderNode) {
      forceNodeWidgetValue(renderNode, "editor_manifest_json", 0, JSON.stringify(manifest, null, 2));
      forceNodeWidgetValue(renderNode, "audio_policy", 1, editorRenderAudioPolicy(manifest));
      forceNodeWidgetValue(renderNode, "fps_mode", 2, "from_manifest");
      forceNodeWidgetValue(renderNode, "override_fps", 3, Number(manifest.fps || widget(node, "fps")?.value || 24));
      forceNodeWidgetValue(renderNode, "tail_trim_frames_per_clip", 4, Number(manifest.render_tail_trim_frames || 0));
      try { renderNode.setDirtyCanvas?.(true, true); } catch {}
      try { app.graph?.setDirtyCanvas?.(true, true); } catch {}
    }
    root.querySelectorAll("[data-render-ready]").forEach((button) => button.classList.add("on"));
    if (status) {
      status.textContent = renderNode
        ? `Render node ready. Audio ${editorRenderAudioPolicy(manifest)}. Tail trim ${manifest.render_tail_trim_frames || 0}f. Queueing render branch only...`
        : `Render plan written but render node could not be created. Audio ${editorRenderAudioPolicy(manifest)}. Tail trim ${manifest.render_tail_trim_frames || 0}f.`;
    }
    if (renderNode) {
      const queued = await queueRenderBranch(renderNode);
      if (queued && status) status.textContent = `Render branch queued only. Audio ${editorRenderAudioPolicy(manifest)}. Upstream generation is not part of this prompt. Tail trim ${manifest.render_tail_trim_frames || 0}f.`;
    }
  }

  function setTailTrimFrames(value) {
    const parsed = Math.max(0, Math.min(12, Math.round(Number(value) || 0)));
    manifest.render_tail_trim_frames = parsed;
    persist();
    if (status) status.textContent = `Final-frame trim set to ${parsed} frame${parsed === 1 ? "" : "s"} per clip.`;
    return parsed;
  }

  function createRenderControls(extraClass = "") {
    const controls = document.createElement("div");
    controls.className = `render-controls ${extraClass}`.trim();
    const tailTrimField = document.createElement("label");
    tailTrimField.className = "mini-field";
    const tailTrimInput = document.createElement("input");
    tailTrimInput.type = "number";
    tailTrimInput.min = "0";
    tailTrimInput.max = "12";
    tailTrimInput.step = "1";
    tailTrimInput.value = String(Math.max(0, Math.min(12, Number(manifest.render_tail_trim_frames || 0))));
    tailTrimInput.title = "Trim this many frames from the end of every rendered take during final assembly.";
    tailTrimInput.addEventListener("mousedown", (event) => event.stopPropagation());
    tailTrimInput.addEventListener("change", () => {
      tailTrimInput.value = String(setTailTrimFrames(tailTrimInput.value));
    });
    tailTrimField.append("TAIL", tailTrimInput, "f");
    const masterAudioButton = makeButton("Master Audio", (button) => {
      const next = editorRenderAudioPolicy(manifest) === "use_master_audio" ? "concat_clip_audio" : "use_master_audio";
      setEditorRenderAudioPolicy(manifest, next);
      button.dataset.stickyOn = next === "use_master_audio" ? "1" : "";
      button.classList.toggle("on", next === "use_master_audio");
      persist();
      const renderNode = findLinkedRenderNode();
      if (renderNode) forceNodeWidgetValue(renderNode, "audio_policy", 1, next);
      if (status) {
        const asset = manifestMasterAudioAsset(manifest);
        status.textContent = next === "use_master_audio"
          ? `Render audio set to MASTER AUDIO${asset ? "" : " (no master asset in manifest yet)"}.`
          : "Render audio set to concatenated take audio.";
      }
    });
    masterAudioButton.title = "Use the single master excerpt on AM for final render instead of concatenating T-lane audio chunks.";
    if (editorRenderAudioPolicy(manifest) === "use_master_audio") {
      masterAudioButton.dataset.stickyOn = "1";
      masterAudioButton.classList.add("on");
    }
    const renderButton = makeRenderButton("Render");
    controls.append(tailTrimField, masterAudioButton, renderButton);
    return controls;
  }

  function makeRenderButton(label = "RENDER") {
    const button = makeButton(label, () => markRenderReady(), "gold render-main-button");
    button.dataset.renderReady = "1";
    button.dataset.stickyOn = "1";
    button.title = "Render the current edited timeline through IAMCCS_ShotboardVideoEditorRenderV1";
    if (manifest.render_ready_at) button.classList.add("on");
    return button;
  }

  function updateOpenEditorButtons(open) {
    root.querySelectorAll("[data-open-editor]").forEach((button) => {
      button.textContent = open ? "Close Editor" : "Open Editor";
      button.classList.toggle("on", Boolean(open));
    });
  }

  function closeFullscreenEditor() {
    if (!fullscreenState) return;
    const state = fullscreenState;
    fullscreenState = null;
    window.removeEventListener("keydown", state.keyHandler);
    root.classList.remove("is-fullscreen");
    state.placeholder.replaceWith(root);
    state.overlay.remove();
    updateOpenEditorButtons(false);
    renderTimeline();
    updateMonitor();
    if (status) status.textContent = "Full-frame editor closed.";
  }

  function openFullscreenEditor() {
    if (fullscreenState) {
      closeFullscreenEditor();
      return;
    }
    const parent = root.parentNode;
    if (!parent) return;
    const placeholder = document.createElement("div");
    placeholder.style.width = "100%";
    placeholder.style.height = `${WIDGET_HEIGHT}px`;
    parent.insertBefore(placeholder, root);

    const overlay = document.createElement("div");
    overlay.className = "iamccs-sve-fullscreen-overlay";
    const bar = document.createElement("div");
    bar.className = "iamccs-sve-fullscreen-bar";
    const title = document.createElement("div");
    title.textContent = "Shotboard Video Editor V1 - Full Frame Monitor";
    const close = document.createElement("button");
    close.textContent = "Close Editor";
    close.addEventListener("click", () => closeFullscreenEditor());
    bar.append(title, close);
    const panel = document.createElement("div");
    panel.className = "iamccs-sve-fullscreen-panel";
    panel.appendChild(root);
    overlay.append(bar, panel);
    const keyHandler = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closeFullscreenEditor();
      }
    };
    fullscreenState = { overlay, placeholder, keyHandler };
    root.classList.add("is-fullscreen");
    document.body.appendChild(overlay);
    window.addEventListener("keydown", keyHandler);
    updateOpenEditorButtons(true);
    renderTimeline();
    updateMonitor();
    if (status) status.textContent = "Full-frame editor open. Press Esc to close.";
  }

  function selectedVideoClip() {
    const clip = selectedClip();
    if (clip?.type === "video") return clip;
    return clipAtTime("video");
  }

  function stopSourcePlayback() {
    sourcePlaying = false;
    if (sourceRaf) cancelAnimationFrame(sourceRaf);
    sourceRaf = 0;
    root.querySelector(".source-play-btn")?.classList.remove("on");
  }

  function setSourceHeadForClip(clip, value) {
    if (!clip) return;
    const duration = Math.max(0, Number(clip.duration || 0));
    selectedClipId = clip.id;
    currentSourceClipId = clip.id;
    sourceHead = Math.max(0, Math.min(duration, Number(value || 0)));
    persist();
    updateMonitor();
  }

  function sourceJump(edge = "in") {
    const clip = selectedVideoClip();
    if (!clip) return;
    const duration = Math.max(0, Number(clip.duration || 0));
    setSourceHeadForClip(clip, edge === "out" ? Math.max(0, duration - (1 / Math.max(1, Number(manifest.fps || 24)))) : 0);
  }

  function sourceNudge(delta) {
    const clip = selectedVideoClip();
    if (!clip) return;
    if (currentSourceClipId !== clip.id) {
      sourceHead = Math.max(0, playhead - Number(clip.startTime || 0));
    }
    setSourceHeadForClip(clip, sourceHead + delta);
  }

  function sourcePlay(button = null) {
    const clip = selectedVideoClip();
    if (!clip) return;
    if (sourcePlaying) {
      stopSourcePlayback();
      return;
    }
    selectedClipId = clip.id;
    currentSourceClipId = clip.id;
    const duration = Math.max(0, Number(clip.duration || 0));
    if (sourceHead >= duration || sourceHead < 0) sourceHead = 0;
    sourcePlaying = true;
    sourceStartMs = performance.now();
    sourceStartHead = sourceHead;
    const playButton = button || root.querySelector(".source-play-btn");
    if (playButton) {
      playButton.dataset.stickyOn = "1";
      playButton.classList.add("on");
    }
    const tick = (now) => {
      if (!sourcePlaying) return;
      sourceHead = sourceStartHead + ((now - sourceStartMs) / 1000);
      if (sourceHead >= duration) {
        sourceHead = Math.max(0, duration - (1 / Math.max(1, Number(manifest.fps || 24))));
        stopSourcePlayback();
        updateMonitor();
        return;
      }
      updateMonitor();
      sourceRaf = requestAnimationFrame(tick);
    };
    sourceRaf = requestAnimationFrame(tick);
  }

  function saveProjectFile() {
    manifest.saved_as_project_at = Date.now() / 1000;
    persist();
    const blob = new Blob([JSON.stringify(manifest, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `iamccs_video_editor_${Date.now()}.iamedit.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    if (status) status.textContent = "Video editor project exported.";
  }

  function openProjectFile(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const parsed = parseJson(reader.result, null);
      if (!parsed || parsed.schema !== "iamccs.shotboard_video_editor.v1") {
        if (status) status.textContent = "Invalid IAMCCS video editor project.";
        return;
      }
      manifest = normalizeManifestTracks(parsed);
      playhead = Math.min(Number(manifest.ui_state?.playhead || 0), timelineDuration());
      selectedClipId = String(manifest.ui_state?.selected_clip_id || "");
      selectedTrackId = String(manifest.ui_state?.selected_track_id || "");
      replaceProjectOnInject = Boolean(manifest.ui_state?.replace_project_on_inject || node.properties?.iamccs_video_editor_replace_project);
      persist();
      renderTimeline();
      if (status) status.textContent = "Video editor project imported.";
    };
    reader.readAsText(file);
  }

  function resetEditorProject(reason = "clear") {
    stop();
    stopSourcePlayback();
    manifest = normalizeManifestTracks({
      schema: "iamccs.shotboard_video_editor.v1",
      schema_version: 1,
      fps: Number(widget(node, "fps")?.value || 24),
      assets: {},
      clips: [],
      tracks: EDITOR_TRACKS.map((track) => ({ ...track })),
      duration_seconds: 20,
      assembly_order: [],
      ui_state: { playhead: 0 },
      cleared_at: Date.now() / 1000,
    });
    playhead = 0;
    sourceHead = 0;
    selectedClipId = "";
    selectedTrackId = "";
    currentSourceClipId = "";
    persist();
    renderTimeline();
    if (status) status.textContent = reason === "new" ? "Video editor project reset." : "Video editor cleared for a new generation.";
  }

  const top = document.createElement("div");
  top.className = "top";
  const brand = document.createElement("div");
  brand.className = "brand";
  brand.innerHTML = `<h3>Shotboard Video Editor</h3>`;
  const transport = document.createElement("div");
  transport.className = "transport";
  const audioMeter = document.createElement("div");
  audioMeter.className = "audio-meter";
  audioMeter.innerHTML = `<span class="meter-label">PK</span><div class="meter-shell"><div class="meter-fill"></div></div>`;
  transport.append(
    makeButton("|<", () => { playhead = 0; persist(); updateMonitor(); }),
    makeButton("<<", () => nudge(-1)),
    makeButton("Play", () => playing ? stop() : play(), "gold wide play-btn"),
    makeButton(">>", () => nudge(1)),
    makeButton(">|", () => { playhead = timelineDuration(); persist(); syncAudioPlayback(true); updateMonitor(); }),
    makeButton("Stop", () => stop(), "wide"),
    audioMeter,
  );
  const tools = document.createElement("div");
  tools.className = "tools";
  const openProjectInput = document.createElement("input");
  openProjectInput.type = "file";
  openProjectInput.accept = ".json,.iamedit.json,application/json";
  openProjectInput.style.display = "none";
  openProjectInput.addEventListener("change", () => {
    openProjectFile(openProjectInput.files?.[0]);
    openProjectInput.value = "";
  });
  tools.append(
    makeButton("New", () => resetEditorProject("new")),
    makeButton("Open", () => openProjectInput.click()),
    makeButton("Save Project", () => saveProjectFile()),
    makeButton("Cut", () => splitSelectedClip()),
    makeButton("Trim In", () => trimSelectedToPlayhead("in")),
    makeButton("Trim Out", () => trimSelectedToPlayhead("out")),
    makeButton("Clear", () => resetEditorProject("clear"), "gold"),
    makeButton("Razor", () => splitSelectedClip()),
    makeButton("Snap", (b) => b.classList.toggle("on")),
    (() => {
      const button = makeButton("Open Editor", () => openFullscreenEditor(), "gold");
      button.dataset.openEditor = "1";
      return button;
    })(),
    createRenderControls("toolbar-render-controls"),
    openProjectInput,
  );
  top.append(brand, transport, tools);

  const takes = document.createElement("div");
  takes.className = "takes";
  const takesLabel = document.createElement("div");
  takesLabel.className = "takes-label";
  takesLabel.textContent = "T/A TAKES";
  const takesScroll = document.createElement("div");
  takesScroll.className = "takes-scroll";
  for (let i = 1; i <= 5; i++) {
    takesScroll.appendChild(makeButton(`T${i}/A${i}`, () => {
      const clip = (manifest.clips || []).find((item) => Number(item.takeIndex) === i && item.type === "video");
      if (clip) {
        selectedClipId = clip.id;
        selectedTrackId = trackIdForClip(clip);
        currentSourceClipId = clip.id;
        sourceHead = 0;
        stopSourcePlayback();
        playhead = Number(clip.startTime || 0);
        persist();
        renderTimeline();
      }
    }, i === 1 ? "gold" : ""));
  }
  const clock = document.createElement("div");
  clock.className = "clock";
  clock.textContent = `${fmtTime(0)} / ${fmtTime(manifest.duration_seconds || 0)}`;
  const takesActions = document.createElement("div");
  takesActions.className = "takes-actions";
  takesActions.append(
    makeButton("Add Video", () => addManual("video"), "gold"),
    makeButton("Add Audio", () => addManual("audio"), "gold"),
    makeButton("Add MA", () => addMasterAudioClip(), "gold"),
    makeButton("Purge", purgeVideoEditorParking, "danger"),
    makeButton("Delete Selected", () => deleteSelectedClip()),
  );
  takes.append(takesLabel, takesScroll, clock, takesActions);

  const main = document.createElement("div");
  main.className = "main";
  const pool = document.createElement("div");
  pool.className = "pool";
  pool.innerHTML = `<div class="panel-title">MEDIA POOL</div><div class="pool-list"></div>`;
  const monitors = document.createElement("div");
  monitors.className = "monitor-grid";
  monitors.innerHTML = `
    <div class="monitor"><div class="panel-title">SOURCE MONITOR</div><div class="screen"><img class="source-img"/><div class="source-transport"></div><div class="empty source-empty">select a take</div><div class="tag">Source</div></div></div>
    <div class="monitor"><div class="panel-title">PROGRAM MONITOR</div><div class="screen"><img class="program-img"/><div class="empty program-empty">timeline output</div><div class="tag">Program</div></div></div>
  `;
  main.append(pool, monitors);
  const sourceTransport = monitors.querySelector(".source-transport");
  sourceTransport?.append(
    makeButton("|<", () => sourceJump("in")),
    makeButton("<<", () => sourceNudge(-0.25)),
    makeButton("Play", (button) => sourcePlay(button), "gold source-play-btn"),
    makeButton(">>", () => sourceNudge(0.25)),
    makeButton(">|", () => sourceJump("out")),
  );

  const timeline = document.createElement("div");
  timeline.className = "timeline-wrap";
  timeline.innerHTML = `<div class="meter"><div class="meter-label">TIME</div><div class="meter-ruler"></div></div><div class="lane-scroll"><div class="lanes"></div></div><div class="master-lane-fixed"></div>`;
  bindScrubber(timeline.querySelector(".meter-ruler"), "ruler");
  bindScrubber(timeline.querySelector(".lane-scroll"), "lanes");
  bindScrubber(timeline.querySelector(".master-lane-fixed"), "lanes");

  status = document.createElement("div");
  status.className = "status";
  status.textContent = "Ready.";

  root.append(top, takes, main, timeline, status, audioEl);
  const uiWidget = node.addDOMWidget("Shotboard Video Editor V1", "iamccs_shotboard_video_editor_v1", root, { serialize: false });
  uiWidget.iamccsSveVersion = UI_VERSION;
  uiWidget.computeSize = () => [NODE_SIZE[0] - 18, WIDGET_HEIGHT];
  try {
    if (node._iamccsSveResizeObserver) node._iamccsSveResizeObserver.disconnect();
    let resizeTimer = 0;
    node._iamccsSveResizeObserver = new ResizeObserver(() => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        try {
          renderTimeline();
          syncRulerScroll();
        } catch (error) {
          console.warn("[IAMCCS ShotboardVideoEditorV1 UI] resize render skipped", error);
        }
      }, 80);
    });
    node._iamccsSveResizeObserver.observe(root);
  } catch {}

  const originalExecuted = node.onExecuted;
  node.onExecuted = function(message) {
    try { originalExecuted?.apply(this, arguments); } catch {}
    const fromUi = message?.iamccs_video_editor_manifest?.[0];
    if (fromUi) {
      const incoming = parseJson(fromUi, null);
      if (!incoming || incoming.schema !== "iamccs.shotboard_video_editor.v1") return;
      const hasExistingProject = Boolean((manifest.clips || []).length || Object.keys(manifest.assets || {}).length);
      const appendedManifest = hasExistingProject && !replaceProjectOnInject ? mergeIncomingAppendManifest(manifest, incoming) : null;
      const appendedTakes = appendedManifest ? incomingAppendTakes(manifest, incoming) : [];
      if (hasExistingProject && !replaceProjectOnInject) {
        if (!appendedManifest) {
          saveManifest(node, manifest);
          if (status) status.textContent = "Backend project replace blocked. Press Clear before writing a new project.";
          return;
        }
      }
      node.properties = node.properties || {};
      const nextManifest = appendedManifest || incoming;
      nextManifest.ui_state = nextManifest.ui_state && typeof nextManifest.ui_state === "object" ? nextManifest.ui_state : {};
      nextManifest.ui_state.replace_project_on_inject = false;
      const nextText = JSON.stringify(nextManifest, null, 2);
      node.properties.iamccs_video_editor_manifest = nextText;
      setWidget(node, "editor_manifest_json", nextText);
      manifest = manifestFromNode(node);
      playhead = Math.min(playhead, Number(manifest.duration_seconds || 0));
      selectedClipId = String(manifest.ui_state?.selected_clip_id || "");
      selectedTrackId = String(manifest.ui_state?.selected_track_id || "");
      renderTimeline();
      if (status) status.textContent = appendedManifest
        ? `Backend generation appended T${appendedTakes.map((take) => String(take).padStart(2, "0")).join(", T") || "??"}.`
        : "Backend generation injected into empty editor project.";
    }
  };

  renderTimeline();
  requestAnimationFrame(() => {
    try {
      renderTimeline();
      syncRulerScroll();
    } catch {}
  });
  console.info("[IAMCCS ShotboardVideoEditorV1 UI] installed", { nodeId: node?.id, reason });
}

app.registerExtension({
  name: "IAMCCS.ShotboardVideoEditorV1",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== TYPE) return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const ret = onNodeCreated?.apply(this, arguments);
      setTimeout(() => installEditor(this, "created"), 50);
      return ret;
    };
  },
  nodeCreated(node) {
    if (isEditor(node)) setTimeout(() => installEditor(node, "nodeCreated"), 50);
  },
});
