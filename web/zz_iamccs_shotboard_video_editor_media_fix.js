import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// IAMCCS Shotboard Video Editor manual-media repair
// 2026-09-17
// Fixes:
// 1) manual video imports now expose the uploaded media as a real browser video
//    source instead of only a static JPEG preview strip;
// 2) embedded browser-decodable audio is extracted to WAV and inserted as a
//    linked companion clip on the matching A lane.

const TYPE = "IAMCCS_ShotboardVideoEditorV1";
const FIX_VERSION = "20260917-manual-video-playback-embedded-audio-v1";

function nodeType(node) {
  return String(node?.comfyClass || node?.type || node?.constructor?.type || "");
}

function widget(node, name) {
  return (node?.widgets || []).find((item) => item?.name === name);
}

function parseJson(value, fallback = null) {
  if (value && typeof value === "object") return value;
  if (typeof value !== "string" || !value.trim()) return fallback;
  try { return JSON.parse(value); } catch { return fallback; }
}

function editorManifest(node) {
  const candidates = [
    widget(node, "editor_manifest_json")?.value,
    node?.properties?.iamccs_video_editor_manifest,
    node?.properties?.editor_manifest_json,
  ];
  for (const value of candidates) {
    const parsed = parseJson(value, null);
    if (parsed?.schema === "iamccs.shotboard_video_editor.v1") return parsed;
  }
  return {
    schema: "iamccs.shotboard_video_editor.v1",
    schema_version: 1,
    fps: 24,
    assets: {},
    clips: [],
    tracks: [],
    duration_seconds: 20,
    assembly_order: [],
    ui_state: { playhead: 0, selected_clip_id: "", selected_track_id: "" },
  };
}

function saveSerializedManifest(node, manifest) {
  const text = JSON.stringify(manifest, null, 2);
  node.properties = node.properties || {};
  node.properties.iamccs_video_editor_manifest = text;
  node.properties.editor_manifest_json = text;
  const w = widget(node, "editor_manifest_json");
  if (w) {
    w.value = text;
    try { w.callback?.(text); } catch {}
  }
  try { node.setDirtyCanvas?.(true, true); } catch {}
  try { app.graph?.setDirtyCanvas?.(true, true); } catch {}
  return text;
}

function splitUploadedPath(path, fallbackName = "") {
  const normalized = String(path || "").replace(/\\/g, "/").replace(/^\/+/, "");
  const parts = normalized.split("/").filter(Boolean);
  const filename = parts.pop() || String(fallbackName || "");
  return { filename, subfolder: parts.join("/") };
}

function repairManualVideoPreviewMetadata(manifest) {
  let changed = false;
  const assets = manifest?.assets && typeof manifest.assets === "object" ? manifest.assets : {};
  for (const asset of Object.values(assets)) {
    if (!asset || asset.type !== "video" || !asset.manual) continue;
    if (asset.preview_video || asset.previewVideo || asset.preview_video_file) continue;
    const source = splitUploadedPath(
      asset.videoFile || asset.mediaPath || asset.media_path || asset.path,
      asset.filename || asset.fileName,
    );
    if (!source.filename) continue;
    const type = String(asset.videoUploadType || asset.uploadType || asset.file_type || "input");
    asset.preview_video = { filename: source.filename, subfolder: source.subfolder, type };
    asset.preview_video_file = source.filename;
    asset.preview_video_subfolder = source.subfolder;
    asset.preview_video_type = type;
    asset.preview_video_fps = Number(asset.fps || manifest.fps || 24);
    asset.preview_video_schema = 2;
    asset.preview_video_codec = "source_browser_native";
    changed = true;
  }
  return changed;
}

function repairSerializedManualPreview(node) {
  const manifest = editorManifest(node);
  if (!manifest || manifest.schema !== "iamccs.shotboard_video_editor.v1") return false;
  if (!repairManualVideoPreviewMetadata(manifest)) return false;
  saveSerializedManifest(node, manifest);
  return true;
}

async function uploadMedia(file, subfolder) {
  const body = new FormData();
  body.append("image", file);
  if (subfolder) body.append("subfolder", String(subfolder));
  body.append("type", "input");
  body.append("overwrite", "false");
  const response = await api.fetchApi("/upload/image", { method: "POST", body });
  if (!response || response.status !== 200) {
    throw new Error(`upload failed: ${response?.status || "no response"}`);
  }
  const data = await response.json();
  const filename = data?.name || file.name;
  const returnedSubfolder = data?.subfolder || subfolder || "";
  return {
    filename,
    subfolder: returnedSubfolder,
    type: data?.type || "input",
    path: [returnedSubfolder, filename].filter(Boolean).join("/"),
  };
}

function waitForVideoSeek(video, time) {
  return new Promise((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      video.removeEventListener("seeked", finish);
      resolve();
    };
    video.addEventListener("seeked", finish, { once: true });
    setTimeout(finish, 900);
    try { video.currentTime = Math.max(0, Number(time) || 0); } catch { finish(); }
  });
}

async function inspectVideo(file) {
  const video = document.createElement("video");
  const url = URL.createObjectURL(file);
  video.preload = "metadata";
  video.muted = true;
  video.playsInline = true;
  video.src = url;
  try {
    await new Promise((resolve, reject) => {
      video.onloadedmetadata = resolve;
      video.onerror = () => reject(new Error("video metadata could not be read by Chromium"));
      video.load();
    });
    const duration = Number.isFinite(video.duration) ? Math.max(0, video.duration) : 0;
    const preview_strip = [];
    if (duration > 0 && video.videoWidth > 0 && video.videoHeight > 0) {
      const width = 320;
      const height = Math.max(64, Math.round(width * video.videoHeight / video.videoWidth));
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        const count = Math.min(10, Math.max(4, Math.ceil(duration / 3)));
        for (let i = 0; i < count; i += 1) {
          const at = count <= 1 ? 0 : Math.min(duration - 0.001, (i / (count - 1)) * duration);
          await waitForVideoSeek(video, at);
          ctx.fillStyle = "#050909";
          ctx.fillRect(0, 0, width, height);
          try { ctx.drawImage(video, 0, 0, width, height); } catch {}
          preview_strip.push(canvas.toDataURL("image/jpeg", 0.78));
        }
      }
    }
    return { duration, preview_strip, width: video.videoWidth || 0, height: video.videoHeight || 0 };
  } finally {
    try { video.pause(); } catch {}
    try { URL.revokeObjectURL(url); } catch {}
  }
}

function waveformPeaks(buffer, bins = 420) {
  if (!buffer || !buffer.length || !buffer.numberOfChannels) return [];
  const channels = Array.from({ length: buffer.numberOfChannels }, (_, i) => buffer.getChannelData(i));
  const count = Math.max(64, Math.min(bins, buffer.length));
  const peaks = [];
  for (let bin = 0; bin < count; bin += 1) {
    const start = Math.floor((bin / count) * buffer.length);
    const end = Math.max(start + 1, Math.floor(((bin + 1) / count) * buffer.length));
    let max = 0;
    let sum = 0;
    let samples = 0;
    for (let s = start; s < end; s += 1) {
      let value = 0;
      for (const channel of channels) value += Math.abs(channel[s] || 0);
      value /= Math.max(1, channels.length);
      max = Math.max(max, value);
      sum += value * value;
      samples += 1;
    }
    peaks.push({ min: -max, max, rms: Math.sqrt(sum / Math.max(1, samples)) });
  }
  return peaks;
}

function writeWav16(buffer) {
  const channels = Math.max(1, Math.min(2, buffer.numberOfChannels || 1));
  const sampleRate = Math.max(8000, Math.round(buffer.sampleRate || 48000));
  const frames = buffer.length;
  const bytesPerSample = 2;
  const blockAlign = channels * bytesPerSample;
  const dataSize = frames * blockAlign;
  const out = new ArrayBuffer(44 + dataSize);
  const view = new DataView(out);
  const writeText = (offset, text) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };
  writeText(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeText(8, "WAVE");
  writeText(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeText(36, "data");
  view.setUint32(40, dataSize, true);
  const source = Array.from({ length: channels }, (_, i) => buffer.getChannelData(Math.min(i, buffer.numberOfChannels - 1)));
  let offset = 44;
  for (let frame = 0; frame < frames; frame += 1) {
    for (let ch = 0; ch < channels; ch += 1) {
      const sample = Math.max(-1, Math.min(1, source[ch][frame] || 0));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      offset += 2;
    }
  }
  return new Blob([out], { type: "audio/wav" });
}

async function extractEmbeddedAudio(file) {
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtx) return null;
  let ctx = null;
  try {
    ctx = new AudioCtx();
    const bytes = await file.arrayBuffer();
    const buffer = await ctx.decodeAudioData(bytes.slice(0));
    if (!buffer || !(buffer.duration > 0) || !buffer.numberOfChannels) return null;
    const base = String(file.name || "video").replace(/\.[^.]+$/, "");
    const wav = writeWav16(buffer);
    return {
      file: new File([wav], `${base}_embedded_audio.wav`, { type: "audio/wav" }),
      duration: buffer.duration,
      sampleRate: buffer.sampleRate,
      channels: Math.min(2, buffer.numberOfChannels),
      waveform_peaks: waveformPeaks(buffer),
    };
  } catch (error) {
    console.info("[IAMCCS VideoEditor Media Fix] no browser-decodable embedded audio", error);
    return null;
  } finally {
    try { await ctx?.close?.(); } catch {}
  }
}

function trackOccupied(manifest, trackId, kind, startTime, duration) {
  const start = Math.max(0, Number(startTime || 0));
  const end = start + Math.max(0.001, Number(duration || 0));
  return (manifest.clips || []).some((clip) => {
    if (!clip || clip.type !== kind) return false;
    if (String(clip.trackId || "") !== String(trackId || "")) return false;
    const cs = Number(clip.startTime || 0);
    const ce = cs + Math.max(0.001, Number(clip.duration || 0));
    return start < ce && end > cs;
  });
}

function selectVideoTrack(manifest, startTime, duration) {
  const selected = String(manifest?.ui_state?.selected_track_id || "").toUpperCase();
  if (selected && !/^V[1-5]$/.test(selected)) {
    throw new Error(`Selected lane ${selected} cannot contain video. Select V1-V5.`);
  }
  if (selected) {
    if (trackOccupied(manifest, selected, "video", startTime, duration)) {
      throw new Error(`Selected lane ${selected} is occupied at the insertion point.`);
    }
    return selected;
  }
  for (let i = 1; i <= 5; i += 1) {
    const id = `V${i}`;
    if (!trackOccupied(manifest, id, "video", startTime, duration)) return id;
  }
  throw new Error("All V1-V5 lanes are occupied at the insertion point.");
}

function companionAudioTrack(manifest, videoTrack, startTime, duration) {
  const number = Number(String(videoTrack || "V1").replace(/\D/g, "")) || 1;
  const preferred = `A${Math.max(1, Math.min(5, number))}`;
  if (!trackOccupied(manifest, preferred, "audio", startTime, duration)) return preferred;
  for (let i = 1; i <= 5; i += 1) {
    const id = `A${i}`;
    if (!trackOccupied(manifest, id, "audio", startTime, duration)) return id;
  }
  throw new Error("Embedded audio found, but all A1-A5 lanes are occupied at the insertion point.");
}

function manifestForImport(node) {
  const manifest = JSON.parse(JSON.stringify(editorManifest(node)));
  manifest.assets = manifest.assets && typeof manifest.assets === "object" ? manifest.assets : {};
  manifest.clips = Array.isArray(manifest.clips) ? manifest.clips : [];
  manifest.ui_state = manifest.ui_state && typeof manifest.ui_state === "object" ? manifest.ui_state : {};
  manifest.fps = Math.max(1, Number(manifest.fps || 24));
  return manifest;
}

function injectManifest(node, manifest) {
  const text = saveSerializedManifest(node, manifest);
  try {
    node.onExecuted?.({ iamccs_video_editor_manifest: [text] });
  } catch (error) {
    console.warn("[IAMCCS VideoEditor Media Fix] immediate UI refresh failed", error);
  }
}

function statusForRoot(root, text) {
  const status = root?.querySelector?.(".status");
  if (status) status.textContent = text;
}

async function importVideo(node, root, file) {
  statusForRoot(root, `Reading video ${file.name || "file"}...`);
  const [details, embeddedAudio] = await Promise.all([
    inspectVideo(file),
    extractEmbeddedAudio(file),
  ]);
  const duration = Math.max(0, Number(details.duration || 0));
  if (!(duration > 0)) throw new Error("Could not read video duration.");

  const manifest = manifestForImport(node);
  const startTime = Math.max(0, Number(manifest.ui_state?.playhead || 0));
  const videoTrack = selectVideoTrack(manifest, startTime, duration);
  const lane = Number(videoTrack.slice(1)) || 1;
  const takeIndex = Math.max(1, Math.min(5, lane));
  const timelineId = `T${String(takeIndex).padStart(2, "0")}`;
  const audioLane = `A${takeIndex}`;

  statusForRoot(root, `Uploading video ${file.name || "file"}...`);
  const uploadedVideo = await uploadMedia(file, "IAMCCS_video_editor_manual");
  const stamp = Date.now();
  const videoAssetId = `manual_video_${stamp}`;
  const videoClipId = `clip_manual_video_${stamp}`;
  const videoAsset = {
    id: videoAssetId,
    type: "video",
    takeIndex,
    timelineId,
    audioLane,
    path: uploadedVideo.path,
    mediaPath: uploadedVideo.path,
    media_path: uploadedVideo.path,
    videoFile: uploadedVideo.path,
    videoUploadType: uploadedVideo.type,
    uploadType: uploadedVideo.type,
    fileName: uploadedVideo.filename,
    filename: uploadedVideo.filename,
    duration,
    source_duration: duration,
    timeline_duration: duration,
    fps: manifest.fps,
    width: details.width || 0,
    height: details.height || 0,
    preview_strip: Array.isArray(details.preview_strip) ? details.preview_strip : [],
    preview_video: {
      filename: uploadedVideo.filename,
      subfolder: uploadedVideo.subfolder,
      type: uploadedVideo.type,
    },
    preview_video_file: uploadedVideo.filename,
    preview_video_subfolder: uploadedVideo.subfolder,
    preview_video_type: uploadedVideo.type,
    preview_video_fps: manifest.fps,
    preview_video_schema: 2,
    preview_video_codec: "source_browser_native",
    manual: true,
  };
  manifest.assets[videoAssetId] = videoAsset;

  const videoClip = {
    id: videoClipId,
    assetId: videoAssetId,
    type: "video",
    takeIndex,
    timelineId,
    audioLane,
    startTime,
    duration,
    sourceDuration: duration,
    sourceDurationLimit: duration,
    trimStart: 0,
    trimEnd: duration,
    trackId: videoTrack,
    trackIndex: takeIndex - 1,
    manual: true,
    muted: false,
    volume: 1,
    linked: false,
    linkedClipIds: [],
  };
  manifest.clips.push(videoClip);

  let audioInserted = false;
  if (embeddedAudio?.file) {
    const audioDuration = Math.min(duration, Math.max(0, Number(embeddedAudio.duration || duration)));
    const audioTrack = companionAudioTrack(manifest, videoTrack, startTime, audioDuration);
    statusForRoot(root, `Uploading embedded audio from ${file.name || "video"}...`);
    const uploadedAudio = await uploadMedia(embeddedAudio.file, "IAMCCS_video_editor_manual_audio");
    const audioAssetId = `manual_audio_${stamp}`;
    const audioClipId = `clip_manual_audio_${stamp}`;
    const audioTrackIndex = 5 + Math.max(0, Number(audioTrack.slice(1)) - 1);
    manifest.assets[audioAssetId] = {
      id: audioAssetId,
      type: "audio",
      takeIndex,
      timelineId,
      audioLane: audioTrack,
      path: uploadedAudio.path,
      audioFile: uploadedAudio.path,
      audioUploadType: uploadedAudio.type,
      uploadType: uploadedAudio.type,
      fileName: uploadedAudio.filename,
      filename: uploadedAudio.filename,
      audio_preview_file: uploadedAudio.filename,
      audio_preview_subfolder: uploadedAudio.subfolder,
      audio_preview_type: uploadedAudio.type,
      preview_type: uploadedAudio.type,
      preview_subfolder: uploadedAudio.subfolder,
      duration: audioDuration,
      duration_seconds: audioDuration,
      source_duration: audioDuration,
      sample_rate: embeddedAudio.sampleRate,
      channels: embeddedAudio.channels,
      waveform_peaks: embeddedAudio.waveform_peaks,
      waveformPeaks: embeddedAudio.waveform_peaks,
      waveformReal: true,
      waveform_source: "embedded_video_audio_decode",
      manual: true,
    };
    const audioClip = {
      id: audioClipId,
      assetId: audioAssetId,
      type: "audio",
      takeIndex,
      timelineId,
      audioLane: audioTrack,
      startTime,
      duration: audioDuration,
      sourceDuration: audioDuration,
      sourceDurationLimit: audioDuration,
      trimStart: 0,
      trimEnd: audioDuration,
      trackId: audioTrack,
      trackIndex: audioTrackIndex,
      manual: true,
      muted: false,
      volume: 1,
      linked: true,
      linkedClipIds: [videoClipId],
    };
    videoClip.linked = true;
    videoClip.linkedClipIds = [audioClipId];
    manifest.clips.push(audioClip);
    audioInserted = true;
  }

  manifest.duration_seconds = Math.max(Number(manifest.duration_seconds || 0), startTime + duration);
  manifest.ui_state.playhead = startTime;
  manifest.ui_state.selected_clip_id = videoClipId;
  manifest.ui_state.selected_track_id = videoTrack;
  manifest.updated_at = Date.now() / 1000;
  repairManualVideoPreviewMetadata(manifest);
  injectManifest(node, manifest);
  setTimeout(() => {
    statusForRoot(
      root,
      audioInserted
        ? `Video + embedded audio added: ${videoTrack}/${videoClip.audioLane}. Playback source active.`
        : `Video added to ${videoTrack}. No browser-decodable embedded audio track was found. Playback source active.`,
    );
  }, 80);
}

function installFixedAddVideo(node) {
  if (!node || nodeType(node) !== TYPE) return false;
  const widgetRoot = (node.widgets || [])
    .map((w) => w?.element || w?.inputEl || w?.domElement)
    .find((el) => el?.querySelector?.(".iamccs-sve"))?.querySelector?.(".iamccs-sve")
    || (node.widgets || []).map((w) => w?.element).find((el) => el?.classList?.contains?.("iamccs-sve"));
  const root = widgetRoot;
  if (!root) return false;
  const buttons = Array.from(root.querySelectorAll("button"));
  const old = buttons.find((button) => String(button.textContent || "").trim() === "Add Video");
  if (!old) return false;
  if (old.dataset.iamccsManualMediaFix === FIX_VERSION) return true;

  const replacement = old.cloneNode(true);
  replacement.dataset.iamccsManualMediaFix = FIX_VERSION;
  replacement.title = "Add a video. Browser-compatible playback is enabled and embedded audio is automatically added to the matching A lane.";
  old.replaceWith(replacement);
  replacement.addEventListener("click", () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "video/*,.mp4,.mov,.mkv,.webm,.avi,.m4v";
    input.style.display = "none";
    input.addEventListener("change", async () => {
      const file = input.files?.[0];
      input.remove();
      if (!file) return;
      try {
        await importVideo(node, root, file);
      } catch (error) {
        console.warn("[IAMCCS VideoEditor Media Fix] Add Video failed", error);
        statusForRoot(root, `Add Video failed: ${error?.message || error}`);
      }
    }, { once: true });
    document.body.appendChild(input);
    input.click();
  });
  console.info("[IAMCCS VideoEditor Media Fix] Add Video repaired", { nodeId: node.id, version: FIX_VERSION });
  return true;
}

function schedulePatch(node) {
  let attempts = 0;
  const tick = () => {
    attempts += 1;
    repairSerializedManualPreview(node);
    if (installFixedAddVideo(node) || attempts >= 30) return;
    setTimeout(tick, 100);
  };
  setTimeout(tick, 70);
}

app.registerExtension({
  name: "IAMCCS.ShotboardVideoEditorV1.ManualMediaFix",
  beforeRegisterNodeDef(nodeTypeDef, nodeData) {
    if (nodeData?.name !== TYPE) return;
    const originalCreated = nodeTypeDef.prototype.onNodeCreated;
    nodeTypeDef.prototype.onNodeCreated = function () {
      try { repairSerializedManualPreview(this); } catch {}
      const result = originalCreated?.apply(this, arguments);
      schedulePatch(this);
      return result;
    };
  },
  loadedGraphNode(node) {
    if (nodeType(node) !== TYPE) return;
    try { repairSerializedManualPreview(node); } catch {}
    schedulePatch(node);
  },
});

console.info("[IAMCCS VideoEditor Media Fix] loaded", FIX_VERSION);
