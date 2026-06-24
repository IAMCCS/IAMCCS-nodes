import { app } from "../../scripts/app.js";

const STYLE_ID = "iamccs-dialogue-tag-editor-style";
const TYPE = "IAMCCS_DialogueTagEditor";
const EMOTIONS = ["calm", "tense", "fearful", "serious", "sad", "angry", "excited", "surprised", "empathy", "coldness", "admiration", "relief"];
const STYLES = ["low", "whisper", "serious", "warm", "dry", "comfort", "authority", "chat", "story", "gentle", "murmur", "shout"];
const PARAS = ["Breathing", "Laughter", "Sigh", "Uhm", "Surprise-oh", "Question-ei", "Confirmation-en", "pause:0.3"];
const OVERLAP_PRESETS = [
  ["0", "No overlap"],
  ["0.06", "Tight handoff"],
  ["0.12", "Natural cross"],
  ["0.22", "Interrupt"],
  ["0.35", "Heavy overlap"],
];

function typeOf(node) { return String(node?.comfyClass || node?.type || node?.constructor?.type || ""); }
function isEditor(node) { return typeOf(node) === TYPE; }
function widget(node, name) { return (node.widgets || []).find((item) => item?.name === name); }
function nodeType(node) { return String(node?.comfyClass || node?.type || node?.constructor?.type || ""); }
function setWidget(node, name, value) {
  const item = widget(node, name);
  if (!item) return;
  item.value = value;
  try { item.callback?.(value); } catch {}
  node.setDirtyCanvas?.(true, true);
  app.graph?.setDirtyCanvas?.(true, true);
}
function hideWidget(item) {
  if (!item) return;
  item.hidden = true;
  item.type = "hidden";
  item.computeSize = () => [0, -4];
  item.draw = () => {};
  item.options = { ...(item.options || {}), hidden: true };
}
function defaultData() {
  return {
    schema: "iamccs.dialogue_tag_editor",
    schema_version: 2,
    global_prompt: "cinematic field and reverse-field dialogue, hard cut coverage, one dominant speaking face per shot, visible mouth movement, natural audio-driven performance, silent listener reaction, stable identities, coherent eyelines",
    settings: { engine_profile: "", output_mode: "speaker_stems_for_overlap", tts_generation_mode: "double_stem_ab", speaker_stems_zero_start: false, speaker_stem_srt_local_zero: true, inline_edit_mode: "metadata_only", emotion_routing: "clean_metadata", default_gap_seconds: 0.12, text_theme: "light_boxes", font_zoom: 1 },
    speakers: [
      { id: "A", name: "Man A", voice: "speaker_a_low_tense", reference_text: "Keep your voice low. We do not know who is listening.", language: "en" },
      { id: "B", name: "Man B", voice: "speaker_b_controlled_whisper", reference_text: "Good. Now we finally have something worth protecting.", language: "en" },
    ],
    lines: [
      { id: "line_001", speaker: "A", text: "You said the signal was dead. Then why is that receiver still blinking?", emotion: "tense", style: "low", paralinguistic: "Breathing", overlap_after: 0.18, ref: 1, track: 0, local_prompt: "hard cut, Man A close-up, Man A speaks clearly, visible mouth movement, tense controlled delivery, Man B listens quietly" },
      { id: "line_002", speaker: "B", text: "Because someone on the other side wants us to think we are alone.", emotion: "serious", style: "whisper", paralinguistic: "none", overlap_after: 0.12, ref: 2, track: 1, local_prompt: "hard cut, Man B close-up, Man B speaks clearly, visible mouth movement, guarded quiet answer, Man A listens quietly" },
      { id: "line_003", speaker: "A", text: "If we open that door, we may be giving them exactly what they came for.", emotion: "fearful", style: "dry", paralinguistic: "Sigh", overlap_after: 0.1, ref: 1, track: 0, local_prompt: "hard cut, Man A tighter close-up, Man A speaks clearly, visible mouth movement, fear held under discipline" },
      { id: "line_004", speaker: "B", text: "Then we do not open it. We make them knock twice.", emotion: "coldness", style: "authority", paralinguistic: "none", overlap_after: 0, ref: 2, track: 1, local_prompt: "hard cut, Man B close-up, Man B speaks clearly, visible mouth movement, decisive controlled authority" },
    ],
  };
}
const DIALOGUE_TEMPLATES = [
  ["dialogue_abab", "Dialogue A/B/A/B"],
  ["simple_ab", "Simple dialogue A/B"],
  ["monologue_a", "Monologue A"],
  ["monologue_a_simple", "Monologue A simple"],
];
function templateGlobalPrompt(kind) {
  if (kind === "monologue_a_simple") {
    return "cinematic single-speaker monologue, one dominant speaking face, visible mouth movement, quiet emotional focus, stable identity";
  }
  if (kind === "monologue_a") {
    return "cinematic single-speaker monologue, Speaker A alone in frame, controlled emotional progression across hard-cut close-ups, visible mouth movement, stable identity";
  }
  if (kind === "simple_ab") {
    return "cinematic simple two-person dialogue, clean field and reverse-field coverage, one short A/B exchange, visible mouth movement on the speaking face, coherent eyelines";
  }
  return "cinematic reverse-shot dialogue scene, two people facing each other in a quiet tense space, strict field and reverse-field coverage based on shotboard references, hard cuts only, one dominant speaking face per shot, coherent eyelines";
}
function templateLines(kind) {
  if (kind === "monologue_a_simple") {
    return [
      { id: "line_001", speaker: "A", text: "I know exactly what I have to do now.", emotion: "resolved", style: "low, steady", paralinguistic: "none", overlap_after: 0, ref: 1, track: 0, local_prompt: "hard cut, Speaker A close-up, Speaker A speaks clearly, visible mouth movement, calm resolved delivery" },
    ];
  }
  if (kind === "monologue_a") {
    return [
      { id: "line_001", speaker: "A", text: "I have carried this silence longer than I should.", emotion: "reflective", style: "soft, intimate", paralinguistic: "none", overlap_after: 0, ref: 1, track: 0, local_prompt: "hard cut, Speaker A intimate close-up, Speaker A speaks clearly, visible mouth movement, quiet confession" },
      { id: "line_002", speaker: "A", text: "If I say it now, everything in this room changes.", emotion: "uncertain", style: "quiet, close", paralinguistic: "Breathing", overlap_after: 0, ref: 1, track: 0, local_prompt: "hard cut, Speaker A close-up, Speaker A speaks clearly after a small breath, controlled hesitation, stable identity and coherent eyeline" },
      { id: "line_003", speaker: "A", text: "So listen carefully, because I will only say it once.", emotion: "resolved", style: "low, steady", paralinguistic: "none", overlap_after: 0, ref: 1, track: 0, local_prompt: "hard cut, Speaker A tight close-up, Speaker A speaks clearly, visible mouth movement, resolved final phrase" },
    ];
  }
  if (kind === "simple_ab") {
    return [
      { id: "line_001", speaker: "A", text: "Are you ready?", emotion: "calm", style: "natural", paralinguistic: "none", overlap_after: 0, ref: 1, track: 0, local_prompt: "hard cut, Speaker A close-up, Speaker A speaks clearly, visible mouth movement, clean question" },
      { id: "line_002", speaker: "B", text: "Yes. Let's begin.", emotion: "calm", style: "natural", paralinguistic: "none", overlap_after: 0, ref: 2, track: 1, local_prompt: "hard cut, Speaker B close-up, Speaker B speaks clearly, visible mouth movement, coherent eyeline" },
    ];
  }
  return [
    { id: "line_001", speaker: "A", text: "I have not seen you in such a long time.", emotion: "wistful", style: "soft, intimate", paralinguistic: "none", overlap_after: 0.12, ref: 1, track: 0, local_prompt: "hard cut, Speaker A intimate close-up, Speaker A speaks clearly, visible mouth movement, controlled expression" },
    { id: "line_002", speaker: "B", text: "No, that is not true. You are mistaken.", emotion: "calm denial", style: "quiet, certain", paralinguistic: "none", overlap_after: 0.12, ref: 2, track: 1, local_prompt: "hard cut, Speaker B close-up, Speaker B speaks clearly, visible mouth movement, steady gaze" },
    { id: "line_003", speaker: "A", text: "Maybe I saw you in my dreams.", emotion: "uncertain, haunted", style: "low, reflective", paralinguistic: "Breathing", overlap_after: 0.1, ref: 1, track: 0, local_prompt: "hard cut, Speaker A close-up, Speaker A speaks clearly, visible mouth movement, haunted controlled emotion" },
    { id: "line_004", speaker: "B", text: "No. You saw me in mine.", emotion: "mysterious", style: "slow, unsettling", paralinguistic: "none", overlap_after: 0, ref: 2, track: 1, local_prompt: "hard cut, Speaker B close-up, Speaker B speaks clearly, visible mouth movement, quiet mystery" },
  ];
}
function parseData(node) {
  try {
    const parsed = JSON.parse(String(widget(node, "dialogue_data")?.value || ""));
    if (parsed && typeof parsed === "object") return { ...defaultData(), ...parsed, settings: { ...defaultData().settings, ...(parsed.settings || {}) } };
  } catch {}
  return defaultData();
}
function modeFromData(data) {
  const value = data?.settings?.tts_generation_mode || data?.settings?.output_mode || "speaker_stems_for_overlap";
  return value === "tts_master_unico" || value === "flatten_for_single_track" ? "tts_master_unico" : "double_stem_ab";
}
function outputMode(mode) { return mode === "tts_master_unico" ? "flatten_for_single_track" : "speaker_stems_for_overlap"; }
function writeData(node, data) {
  data.schema = "iamccs.dialogue_tag_editor";
  data.schema_version = 2;
  data.settings = data.settings || {};
  data.settings.tts_generation_mode = modeFromData(data);
  data.settings.output_mode = outputMode(data.settings.tts_generation_mode);
  setWidget(node, "output_mode", data.settings.output_mode);
  setWidget(node, "inline_edit_mode", data.settings.inline_edit_mode || "metadata_only");
  setWidget(node, "dialogue_data", JSON.stringify(data, null, 2));
}
function linesToText(data) {
  return (data.lines || []).map((line) => {
    const attrs = [line.speaker || "A"];
    if (line.emotion && line.emotion !== "none") attrs.push("emotion:" + line.emotion);
    if (line.style && line.style !== "none") attrs.push("style:" + line.style);
    if (line.paralinguistic && line.paralinguistic !== "none") attrs.push("para:" + line.paralinguistic);
    (Array.isArray(line.extra_tags) ? line.extra_tags : []).forEach((tag) => {
      const value = String(tag || "").trim().replace(/^\|+/, "");
      if (value && !attrs.some((attr) => attr.split(":")[0] === value.split(":")[0])) attrs.push(value);
    });
    if (Number(line.overlap_after || 0) > 0) attrs.push("overlap:" + Number(line.overlap_after).toFixed(2));
    if (Number(line.duration || 0) > 0) attrs.push("duration:" + Number(line.duration).toFixed(2));
    if (Number(line.ref || 0) > 0) attrs.push("ref:" + line.ref);
    return "[" + attrs.join("|") + "] " + (line.text || "");
  }).join("\n");
}
function attrValue(value) {
  return String(value || "natural").trim().replace(/[|\]\[]/g, " ").replace(/\s+/g, "_");
}
function ttsExtraTagsForLine(line, mode) {
  const speaker = "speaker_" + speakerIndex(line);
  if (mode === "index") return [];
  if (mode === "longcat") return ["voice:" + speaker];
  if (mode === "chatterbox") {
    const hot = /angry|fear|tense|excited|surprised|haunted|sad|panic|rage/i.test(String(line.emotion || ""));
    return ["exaggeration:" + (hot ? "0.85" : "0.55")];
  }
  return [];
}
function applyTTSFormat(data, mode) {
  data.settings ||= {};
  data.settings.tts_export_mode = mode;
  data.settings.engine_profile = "tts_" + mode;
  data.lines = (data.lines || []).map((line) => ({
    ...line,
    tts_model: mode,
    extra_tags: ttsExtraTagsForLine(line, mode),
  }));
  return data;
}
function modelAwareTagToken(kind, value, mode) {
  const base = "|" + kind + ":" + value;
  if (kind === "emotion" && mode === "chatterbox") return base + "|exaggeration:" + (/angry|fear|tense|excited|surprised|haunted|sad|panic|rage/i.test(value) ? "0.85" : "0.55");
  return base;
}
function cleanDialogueText(value) {
  return stripFormattingTokens(String(value || "")).replace(/\s+/g, " ").trim();
}
function speakerIndex(line) {
  return String(line?.speaker || "A").toUpperCase().startsWith("B") ? 2 : 1;
}
function lineDirection(line) {
  const bits = [];
  if (line.emotion && line.emotion !== "none") bits.push("emotion " + line.emotion);
  if (line.style && line.style !== "none") bits.push("style " + line.style);
  if (line.paralinguistic && line.paralinguistic !== "none") bits.push("paralinguistic " + line.paralinguistic);
  return bits.join(", ") || "natural";
}
function ttsExportText(data, mode) {
  const lines = Array.isArray(data?.lines) ? data.lines : [];
  if (mode === "longcat") {
    return lines.map((line) => "[speaker_" + speakerIndex(line) + "]: " + cleanDialogueText(line.text)).join("\n");
  }
  if (mode === "index") {
    return lines.map((line) => {
      const text = cleanDialogueText(line.text);
      return "[" + String(line.speaker || "A") + "] " + text + "\n# IndexTTS-2 emotion/style hint: " + lineDirection(line);
    }).join("\n\n");
  }
  if (mode === "chatterbox") {
    const emotionLoad = lines.some((line) => /angry|fear|tense|excited|surprised|haunted|sad/i.test(String(line.emotion || ""))) ? "0.75-0.95" : "0.45-0.65";
    return "# Chatterbox: keep tags out of the speech text. Use exaggeration " + emotionLoad + " for this pass.\n\n" +
      lines.map((line) => String(line.speaker || "A") + ": " + cleanDialogueText(line.text)).join("\n");
  }
  return "Voice instruction: Preserve speaker identity. Perform each line with the listed emotion and style; do not speak the bracket metadata.\n\n" +
    lines.map((line) => {
      return String(line.speaker || "A") + " (" + lineDirection(line) + "): " + cleanDialogueText(line.text);
    }).join("\n");
}
function srtTime(seconds) {
  const total = Math.max(0, Number(seconds || 0));
  const whole = Math.floor(total);
  const millis = Math.min(999, Math.max(0, Math.round((total - whole) * 1000)));
  const h = Math.floor(whole / 3600);
  const m = Math.floor((whole % 3600) / 60);
  const s = whole % 60;
  return String(h).padStart(2, "0") + ":" + String(m).padStart(2, "0") + ":" + String(s).padStart(2, "0") + "," + String(millis).padStart(3, "0");
}
function speakerLines(data, speakerIndex = 0) {
  const speakerIds = Array.from(new Set((data?.lines || []).map((line) => String(line?.speaker || "A").toUpperCase()))).filter(Boolean);
  const speaker = speakerIds[speakerIndex] || (speakerIndex === 0 ? "A" : "B");
  return (Array.isArray(data?.lines) ? data.lines : []).filter((line) => String(line?.speaker || "A").toUpperCase() === speaker);
}
function speakerPlainText(data, speakerIndex = 0) {
  return speakerLines(data, speakerIndex)
    .map((line) => cleanDialogueText(line?.text || ""))
    .filter(Boolean)
    .join("\n");
}
function speakerSrtText(data, speakerIndex = 0) {
  const speakerIds = Array.from(new Set((data?.lines || []).map((line) => String(line?.speaker || "A").toUpperCase()))).filter(Boolean);
  const selectedSpeaker = speakerIds[speakerIndex] || (speakerIndex === 0 ? "A" : "B");
  const allLines = Array.isArray(data?.lines) ? data.lines : [];
  if (!allLines.length) return "";
  let cursor = 0;
  const selected = [];
  allLines.forEach((line) => {
    const duration = Number(line?.duration || 0) > 0 ? Number(line.duration) : estimateSeconds(line?.text);
    const absoluteStart = line?.start !== undefined && line?.start !== "" ? Number(line.start || 0) : cursor;
    if (String(line?.speaker || "A").toUpperCase() === selectedSpeaker) {
      selected.push({ line, duration, absoluteStart });
    }
    cursor = Math.max(cursor, absoluteStart + duration + Number(data?.settings?.default_gap_seconds || 0.12) - Math.max(0, Number(line?.overlap_after || 0)));
  });
  if (!selected.length) return "";
  const localSrt = data?.settings?.speaker_stem_srt_local_zero !== false && modeFromData(data) !== "tts_master_unico";
  const offset = localSrt ? Math.min(...selected.map((item) => Number(item.absoluteStart) || 0).filter((value) => Number.isFinite(value))) : 0;
  const parts = [];
  selected.forEach(({ line, duration, absoluteStart }, index) => {
    const start = Math.max(0, absoluteStart - Math.max(0, offset));
    const end = Math.max(start + 0.2, start + duration);
    const text = cleanDialogueText(line?.text || "");
    if (text) parts.push(String(index + 1) + "\n" + srtTime(start) + " --> " + srtTime(end) + "\n" + text + "\n");
  });
  return parts.join("\n").trim();
}
async function copyTextToClipboard(text) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(String(text || ""));
    return true;
  }
  const area = document.createElement("textarea");
  area.value = String(text || "");
  area.style.position = "fixed";
  area.style.left = "-9999px";
  document.body.append(area);
  area.select();
  const ok = document.execCommand("copy");
  area.remove();
  return ok;
}
function liteGraphRegistry() {
  return (window.LiteGraph || globalThis.LiteGraph)?.registered_node_types || {};
}
function isLikelyTTSNodeType(type) {
  const name = String(type || "");
  if (!/(qwen3tts|qwenemotion|indextts|longcat|chatterbox|vibevoice|higgsaudio|cosyvoice|f5tts|echotts|mosstts)/i.test(name)) return false;
  return !/(unified|emotion|options|vectors|voiceinstruct|voicedesigner|voiceclone|whisper|asr|stt|analyzer|capture|loader|loadvoice|voiceslibrary|image|vl|wan|latent|batch|dataset|training|mouth|viseme|rvc|vocal|voicefixer|mergeaudio|refresh)/i.test(name);
}
function ttsNodeLabel(type) {
  const ctor = liteGraphRegistry()[type];
  return String(ctor?.title || ctor?.display_name || ctor?.nodeData?.display_name || ctor?.nodeData?.name || type);
}
function widgetOptionValues(item) {
  const values = item?.options?.values;
  if (Array.isArray(values)) return values.filter((entry) => entry !== undefined && entry !== null && String(entry).trim() !== "");
  if (Array.isArray(item?.options)) return item.options.filter((entry) => entry !== undefined && entry !== null && String(entry).trim() !== "");
  return [];
}
function isModelSelectorWidget(item) {
  const name = String(item?.name || "").toLowerCase();
  if (!name) return false;
  if (/(text|prompt|instruction|emotion|style|language|seed|temperature|cfg|rate|speed|gap|padding|mode|device|cache|prefix|suffix)/i.test(name)) return false;
  return /(model|checkpoint|voice|speaker|preset|profile)/i.test(name);
}
function ttsNodeHasAvailableModels(type) {
  const lite = window.LiteGraph || globalThis.LiteGraph;
  if (!lite?.createNode) return false;
  try {
    const probe = lite.createNode(type);
    const modelWidgets = (probe?.widgets || []).filter(isModelSelectorWidget);
    if (!modelWidgets.length) return true;
    return modelWidgets.some((item) => {
      const values = widgetOptionValues(item);
      if (values.length) return true;
      const value = String(item?.value ?? "").trim();
      return value && value.toLowerCase() !== "none";
    });
  } catch {
    return false;
  }
}
function isRiggableTTSNodeType(type) {
  const lite = window.LiteGraph || globalThis.LiteGraph;
  const registry = lite?.registered_node_types || {};
  if (!registry[type] || !lite?.createNode) return false;
  try {
    const probe = lite.createNode(type);
    return nodeHasOutputType(probe, "TTS_ENGINE") || nodeHasOutputType(probe, "AUDIO");
  } catch {
    return false;
  }
}
function ttsNodeHelpText(type) {
  const name = String(type || "");
  const label = ttsNodeLabel(name);
  let hint = "Registered ComfyUI TTS node. Add Node creates this exact type.";
  if (/qwen/i.test(name)) hint = "Qwen TTS family: good for instruction-style emotion and voice direction.";
  else if (/index/i.test(name)) hint = "IndexTTS family: good target for explicit emotion/style hint tags.";
  else if (/longcat/i.test(name)) hint = "LongCat family: useful for speaker-tagged dialogue and multi-speaker text.";
  else if (/chatterbox/i.test(name)) hint = "Chatterbox family: prefers clean speech text plus exaggeration/style controls.";
  else if (/vibevoice/i.test(name)) hint = "VibeVoice engine: use as a single engine target when its text input is available.";
  else if (/higgsaudio/i.test(name)) hint = "Higgs Audio engine: use Qwen-like instruction text when text widgets are exposed.";
  else if (/cosyvoice/i.test(name)) hint = "CosyVoice engine: works best with clean text plus separate style/voice settings.";
  else if (/f5tts/i.test(name)) hint = "F5-TTS engine: usually wants clean text and reference voice conditioning.";
  return label + "\n" + name + "\n" + hint;
}
function discoverTTSNodeOptions() {
  return Object.keys(liteGraphRegistry())
    .filter(isLikelyTTSNodeType)
    .filter(isRiggableTTSNodeType)
    .filter(ttsNodeHasAvailableModels)
    .sort((a, b) => ttsNodeLabel(a).localeCompare(ttsNodeLabel(b)))
    .map((type) => [type, ttsNodeLabel(type)]);
}
function setNodeWidgetValue(targetNode, names, value) {
  const wanted = new Set(names.map((name) => String(name).toLowerCase()));
  (targetNode.widgets || []).forEach((item) => {
    if (!wanted.has(String(item?.name || "").toLowerCase())) return;
    item.value = value;
    try { item.callback?.(value); } catch {}
  });
}
function firstSlotIndex(slots, names = [], typeHint = "") {
  const list = Array.isArray(slots) ? slots : [];
  const wanted = names.map((name) => String(name || "").toLowerCase());
  const exact = list.findIndex((slot) => wanted.includes(String(slot?.name || "").toLowerCase()));
  if (exact >= 0) return exact;
  const normalized = list.findIndex((slot) => wanted.includes(String(slot?.name || "").toLowerCase().replace(/[^a-z0-9]+/g, "_")));
  if (normalized >= 0) return normalized;
  if (typeHint) {
    const typed = list.findIndex((slot) => String(slot?.type || "").toLowerCase() === String(typeHint).toLowerCase());
    if (typed >= 0) return typed;
  }
  return -1;
}
function connectBySlotName(source, outputNames, target, inputNames, typeHint = "") {
  const out = firstSlotIndex(source?.outputs, outputNames, typeHint);
  const input = firstSlotIndex(target?.inputs, inputNames, typeHint);
  if (out < 0 || input < 0) {
    console.warn("[IAMCCS DialogueTagEditor] rig slot not found", {
      source: source?.title || source?.type,
      target: target?.title || target?.type,
      outputNames,
      inputNames,
      sourceOutputs: (source?.outputs || []).map((slot) => slot?.name || slot?.type),
      targetInputs: (target?.inputs || []).map((slot) => slot?.name || slot?.type),
    });
    return false;
  }
  try {
    if (target?.inputs?.[input]?.link != null) {
      target.disconnectInput?.(input);
    }
    source.connect(out, target, input);
    console.info("[IAMCCS DialogueTagEditor] rig connected", {
      from: source?.title || source?.type,
      output: source?.outputs?.[out]?.name,
      to: target?.title || target?.type,
      input: target?.inputs?.[input]?.name,
    });
    return true;
  } catch (err) {
    console.warn("[IAMCCS DialogueTagEditor] rig connect failed", outputNames, inputNames, err);
    return false;
  }
}
function linkOriginNode(linkId) {
  const link = app?.graph?.links?.[linkId];
  if (!link) return null;
  const originId = link.origin_id ?? link[1];
  return originId != null ? app.graph.getNodeById?.(originId) : null;
}
function linkTargetNode(linkId) {
  const link = app?.graph?.links?.[linkId];
  if (!link) return null;
  const targetId = link.target_id ?? link[3];
  return targetId != null ? app.graph.getNodeById?.(targetId) : null;
}
function nodeHasOutputType(node, type) {
  return Array.isArray(node?.outputs) && node.outputs.some((slot) => String(slot?.type || "").toLowerCase() === String(type).toLowerCase());
}
function nodeHasInputType(node, type) {
  return Array.isArray(node?.inputs) && node.inputs.some((slot) => String(slot?.type || "").toLowerCase() === String(type).toLowerCase());
}
function createGraphNode(lite, graph, type, pos, title = "") {
  const node = lite?.createNode?.(type);
  if (!node) return null;
  node.pos = pos;
  if (title) node.title = title;
  graph.add(node);
  return node;
}
function configureRigTTSNode(node, type, speakerIndex = 0, data = null) {
  const lower = String(type || "").toLowerCase();
  const routing = emotionRoutingValue(data || {});
  if (lower.includes("chatterbox")) {
    setNodeWidgetValue(node, ["language"], "English");
    setNodeWidgetValue(node, ["device"], "auto");
    setNodeWidgetValue(node, ["model_version"], routing === "chatterbox_v2_tokens" ? "v2" : "v1");
    setNodeWidgetValue(node, ["exaggeration"], routing === "chatterbox_v2_tokens" ? 0.78 : (speakerIndex === 0 ? 0.5 : 0.35));
    setNodeWidgetValue(node, ["temperature"], speakerIndex === 0 ? 0.75 : 0.88);
    setNodeWidgetValue(node, ["cfg_weight"], speakerIndex === 0 ? 0.5 : 0.45);
    setNodeWidgetValue(node, ["crash_protection_template"], "hmm ,, {seg} hmm ,,");
  }
  if (lower.includes("index")) {
    const useTextEmotion = routing === "index_tts_text_emotion";
    setNodeWidgetValue(node, ["use_emotion_text", "enable_emotion_text", "text_emotion_enabled"], useTextEmotion);
    setNodeWidgetValue(node, ["emotion_text", "text_emotion", "emotion_prompt"], useTextEmotion ? "{seg}" : "");
    setNodeWidgetValue(node, ["emotion_alpha", "emotion_strength", "emotion_weight"], useTextEmotion ? 1.0 : 0.75);
  }
}
function configureRigSRTNode(node, speakerIndex = 0) {
  setNodeWidgetValue(node, ["narrator_voice"], speakerIndex === 0 ? "voices_examples/Clint_Eastwood CC3 (enhanced2).wav" : "voices_examples/David_Attenborough CC3.wav");
  setNodeWidgetValue(node, ["seed"], speakerIndex === 0 ? 1284582220 : 1284582221);
  setNodeWidgetValue(node, ["timing_mode"], "pad_with_silence");
  setNodeWidgetValue(node, ["enable_audio_cache"], false);
  setNodeWidgetValue(node, ["fade_for_StretchToFit"], 0.01);
  setNodeWidgetValue(node, ["max_stretch_ratio"], 1);
  setNodeWidgetValue(node, ["min_stretch_ratio"], 0.8);
  setNodeWidgetValue(node, ["timing_tolerance"], 2);
  setNodeWidgetValue(node, ["batch_size"], 0);
}
function widgetValue(targetNode, names = []) {
  const wanted = new Set(names.map((name) => String(name).toLowerCase()));
  const item = (targetNode?.widgets || []).find((entry) => wanted.has(String(entry?.name || "").toLowerCase()));
  return item?.value;
}
function isCineAudioExportNode(targetNode) {
  if (nodeType(targetNode) !== "IAMCCS_CineAudioInfo") return false;
  const mode = String(widgetValue(targetNode, ["mode"]) || "").toLowerCase();
  const title = String(targetNode?.title || "").toLowerCase();
  return mode.startsWith("export") || title.includes("export");
}
function isCineAudioInjectNode(targetNode) {
  if (nodeType(targetNode) !== "IAMCCS_CineAudioInfo") return false;
  const mode = String(widgetValue(targetNode, ["mode"]) || "").toLowerCase();
  const title = String(targetNode?.title || "").toLowerCase();
  return mode.startsWith("inject") || title.includes("inject") || title.includes("save tts");
}
function findExistingCineAudioRigNodes(editorNode) {
  const nodes = Array.isArray(app?.graph?._nodes) ? app.graph._nodes : [];
  const linked = new Set(downstream(editorNode, (candidate) => nodeType(candidate) === "IAMCCS_CineAudioInfo").map((candidate) => candidate.id));
  const cineNodes = nodes.filter((candidate) => nodeType(candidate) === "IAMCCS_CineAudioInfo");
  const exportInfo = cineNodes.find((candidate) => linked.has(candidate.id) && isCineAudioExportNode(candidate)) || cineNodes.find(isCineAudioExportNode) || null;
  const injectInfo = cineNodes.find((candidate) => linked.has(candidate.id) && isCineAudioInjectNode(candidate)) || cineNodes.find(isCineAudioInjectNode) || null;
  const bridge = nodes.find((candidate) => nodeType(candidate) === "IAMCCS_DialogueAudioBoardBridge") || null;
  return { exportInfo, injectInfo, bridge };
}
function isReplaceableAudioRigNode(targetNode) {
  const type = nodeType(targetNode);
  const title = String(targetNode?.title || "").toLowerCase();
  if (!targetNode || type === "IAMCCS_CineAudioInfo" || type === "IAMCCS_DialogueAudioBoardBridge" || isEditor(targetNode)) return false;
  if (/^direct tts\b|iamccs unified srt/.test(title)) return true;
  if (type === "UnifiedTTSSRTNode" || type === "UnifiedTTSTextNode") return true;
  return isLikelyTTSNodeType(type) && isRiggableTTSNodeType(type);
}
function collectUpstreamRigNodes(targetNode, result) {
  (targetNode?.inputs || []).forEach((input) => {
    const origin = input?.link != null ? linkOriginNode(input.link) : null;
    if (!origin || result.has(origin.id)) return;
    if (!isReplaceableAudioRigNode(origin)) return;
    result.add(origin.id);
    collectUpstreamRigNodes(origin, result);
  });
}
function collectDownstreamRigNodes(sourceNode, result) {
  (sourceNode?.outputs || []).forEach((output) => {
    (output?.links || []).forEach((linkId) => {
      const target = linkTargetNode(linkId);
      if (!target || result.has(target.id)) return;
      if (!isReplaceableAudioRigNode(target)) return;
      result.add(target.id);
      collectDownstreamRigNodes(target, result);
      collectUpstreamRigNodes(target, result);
    });
  });
}
function removeNodeSafely(graph, targetNode) {
  if (!targetNode || !graph) return false;
  try {
    (targetNode.inputs || []).forEach((_, index) => {
      try { targetNode.disconnectInput?.(index); } catch {}
    });
    (targetNode.outputs || []).forEach((_, index) => {
      try { targetNode.disconnectOutput?.(index); } catch {}
    });
    graph.remove(targetNode);
    return true;
  } catch (err) {
    console.warn("[IAMCCS DialogueTagEditor] could not remove old rig node", targetNode?.type, targetNode?.id, err);
    return false;
  }
}
function removeExistingAudioRigNodes(exportInfo, injectInfo) {
  const graph = app.graph;
  const nodes = Array.isArray(graph?._nodes) ? graph._nodes : [];
  const removeIds = new Set();
  collectDownstreamRigNodes(exportInfo, removeIds);
  collectUpstreamRigNodes(injectInfo, removeIds);
  nodes.forEach((targetNode) => {
    if (isReplaceableAudioRigNode(targetNode) && /^direct tts\b|iamccs unified srt/.test(String(targetNode.title || "").toLowerCase())) {
      removeIds.add(targetNode.id);
    }
  });
  [exportInfo?.id, injectInfo?.id].forEach((id) => removeIds.delete(id));
  let removed = 0;
  Array.from(removeIds)
    .map((id) => graph.getNodeById?.(id))
    .filter(Boolean)
    .forEach((targetNode) => {
      if (removeNodeSafely(graph, targetNode)) removed += 1;
  });
  return removed;
}
function repairExistingDialogueRigLinks(editorNode, reason = "repair") {
  if (!isEditor(editorNode)) return 0;
  const graph = app?.graph;
  const nodes = Array.isArray(graph?._nodes) ? graph._nodes : [];
  const rigNodes = findExistingCineAudioRigNodes(editorNode);
  const exportInfo = rigNodes.exportInfo;
  if (!exportInfo) return 0;
  const data = parseData(editorNode);
  let fixed = 0;
  nodes.forEach((targetNode) => {
    if (nodeType(targetNode) !== "UnifiedTTSSRTNode") return;
    const title = String(targetNode?.title || "").toLowerCase();
    if (!title.includes("direct tts")) return;
    let speakerIndex = -1;
    if (/\bb\b|man b|speaker b/.test(title)) speakerIndex = 1;
    else if (/\ba\b|man a|speaker a/.test(title)) speakerIndex = 0;
    if (speakerIndex < 0) return;
    const sourceName = speakerIndex === 0 ? "speaker_a_srt" : "speaker_b_srt";
    const inputIndex = firstSlotIndex(targetNode.inputs, ["srt_content"], "");
    const outputIndex = firstSlotIndex(exportInfo.outputs, [sourceName], "");
    if (inputIndex < 0 || outputIndex < 0) {
      console.warn("[IAMCCS DialogueTagEditor] existing A/B rig cannot be repaired", {
        reason,
        target: targetNode?.title || targetNode?.type,
        wantedOutput: sourceName,
      });
      return;
    }
    const currentLink = targetNode.inputs?.[inputIndex]?.link;
    const link = currentLink != null ? app?.graph?.links?.[currentLink] : null;
    const currentOrigin = link ? graph.getNodeById?.(link.origin_id ?? link[1]) : null;
    const currentOutputIndex = link ? (link.origin_slot ?? link[2]) : -1;
    const currentOutputName = currentOrigin?.outputs?.[currentOutputIndex]?.name || "";
    const alreadyCorrect = currentOrigin?.id === exportInfo.id && String(currentOutputName).toLowerCase() === sourceName;
    if (!alreadyCorrect) {
      if (connectBySlotName(exportInfo, [sourceName], targetNode, ["srt_content"], "")) fixed += 1;
    }
    setNodeWidgetValue(targetNode, ["enable_audio_cache"], false);
    setNodeWidgetValue(targetNode, ["srt_content", "text", "dialogue", "prompt"], speakerSrtText(data, speakerIndex));
  });
  if (fixed) {
    console.info("[IAMCCS DialogueTagEditor] repaired existing A/B SRT rig links", { reason, fixed });
    graph?.setDirtyCanvas?.(true, true);
  }
  return fixed;
}
function addAvailableTTSStarterNodes(data, statusTarget, selectedTypes = []) {
  const graph = app.graph;
  const lite = window.LiteGraph || globalThis.LiteGraph;
  const registry = lite?.registered_node_types || {};
  if (!graph || !lite?.createNode) {
    if (statusTarget) statusTarget.textContent = "Graph API not ready.";
    return 0;
  }
  const requested = Array.from(new Set((selectedTypes || []).map((type) => String(type || "")).filter((type) => registry[type])));
  if (!requested.length) {
    if (statusTarget) statusTarget.textContent = "Select one or more registered TTS nodes first.";
    return 0;
  }
  const existing = graph._nodes || graph.nodes || [];
  const bottom = existing.reduce((maxY, item) => Math.max(maxY, Number(item?.pos?.[1] || 0) + Number(item?.size?.[1] || 160)), 0);
  const baseX = Number(existing.find((item) => isEditor(item))?.pos?.[0] || 80);
  const baseY = bottom + 120;
  const qwenText = ttsExportText(data, "qwen");
  const longcatText = ttsExportText(data, "longcat");
  const indexText = ttsExportText(data, "index");
  let created = 0;
  requested.forEach((type) => {
    const label = ttsNodeLabel(type);
    const next = lite.createNode(type);
    if (!next) return;
    next.pos = [baseX + (created % 4) * 330, baseY + Math.floor(created / 4) * 220];
    next.title = "IAMCCS " + label;
    graph.add(next);
    const lower = type.toLowerCase();
    if (lower.includes("longcat")) setNodeWidgetValue(next, ["text", "dialogue", "prompt"], longcatText);
    else if (lower.includes("index")) setNodeWidgetValue(next, ["text", "dialogue", "prompt"], indexText);
    else if (lower.includes("qwen") || lower.includes("higgs") || lower.includes("cosy") || lower.includes("vibe")) setNodeWidgetValue(next, ["text", "dialogue", "prompt", "custom_instruct", "instruction"], qwenText);
    else setNodeWidgetValue(next, ["text", "dialogue", "prompt"], ttsExportText(data, "chatterbox"));
    created += 1;
  });
  graph.setDirtyCanvas?.(true, true);
  app.graph?.setDirtyCanvas?.(true, true);
  if (statusTarget) statusTarget.textContent = created ? "Created " + created + " selected TTS node" + (created === 1 ? "" : "s") + " at graph bottom." : "Selected TTS nodes could not be created.";
  return created;
}
function createCineAudioRig(editorNode, data, statusTarget, selectedType) {
  const graph = app.graph;
  const lite = window.LiteGraph || globalThis.LiteGraph;
  const registry = lite?.registered_node_types || {};
  if (!graph || !lite?.createNode) {
    if (statusTarget) statusTarget.textContent = "Graph API not ready.";
    return 0;
  }
  if (!registry.IAMCCS_CineAudioInfo) {
    if (statusTarget) statusTarget.textContent = "IAMCCS_CineAudioInfo is not registered.";
    return 0;
  }
  if (!selectedType || !registry[selectedType]) {
    if (statusTarget) statusTarget.textContent = "Select a TTS model node before creating the rig.";
    return 0;
  }
  const compatibilityProbe = lite.createNode(selectedType);
  const canRigSelected = nodeHasOutputType(compatibilityProbe, "TTS_ENGINE") || nodeHasOutputType(compatibilityProbe, "AUDIO");
  if (!canRigSelected) {
    if (statusTarget) statusTarget.textContent = ttsNodeLabel(selectedType) + " is not a rig generator. Select a TTS engine or AUDIO-producing TTS node.";
    return 0;
  }
  const existing = graph._nodes || graph.nodes || [];
  const rigNodes = findExistingCineAudioRigNodes(editorNode);
  const reusableExport = rigNodes.exportInfo || null;
  const reusableInject = rigNodes.injectInfo || null;
  const bottom = existing.reduce((maxY, item) => Math.max(maxY, Number(item?.pos?.[1] || 0) + Number(item?.size?.[1] || 160)), 0);
  const baseX = Number(reusableExport?.pos?.[0] ?? editorNode?.pos?.[0] ?? 80);
  const baseY = Number(reusableExport?.pos?.[1] ?? bottom + 140);
  const speakerIds = Array.from(new Set((data.lines || []).map((line) => String(line.speaker || "A").toUpperCase()))).filter(Boolean);
  const isMono = modeFromData(data) === "tts_master_unico" || speakerIds.length <= 1;
  const created = [];
  const exportInfo = reusableExport || createGraphNode(lite, graph, "IAMCCS_CineAudioInfo", [baseX, baseY], "CineAudioInfo 1 - Export SRT A/B from dialogue");
  const injectInfo = reusableInject || createGraphNode(lite, graph, "IAMCCS_CineAudioInfo", [baseX + 980, baseY], "CineAudioInfo 2 - Save TTS WAVs + inject stems");
  if (!exportInfo || !injectInfo) {
    if (statusTarget) statusTarget.textContent = "Could not create CineAudioInfo nodes.";
    return 0;
  }
  if (!reusableExport) created.push(exportInfo);
  if (!reusableInject) created.push(injectInfo);
  const cineTtsTextMode = cineAudioTextModeFromEmotionRouting(data);
  setNodeWidgetValue(exportInfo, ["mode"], isMono ? "export_tts_srt" : "export_speaker_stems");
  setNodeWidgetValue(exportInfo, ["tts_text_mode"], cineTtsTextMode);
  setNodeWidgetValue(exportInfo, ["frame_rate"], Number(widget(editorNode, "frame_rate")?.value || 24));
  setNodeWidgetValue(exportInfo, ["lane_injection_mode"], "speaker_full_timeline_clips");
  setNodeWidgetValue(exportInfo, ["file_prefix"], "dialogue_tts_export");
  setNodeWidgetValue(injectInfo, ["mode"], isMono ? "inject_generated_audio" : "inject_speaker_stems");
  setNodeWidgetValue(injectInfo, ["tts_text_mode"], cineTtsTextMode);
  setNodeWidgetValue(injectInfo, ["lane_injection_mode"], isMono ? "single_master_clip" : "speaker_full_timeline_clips");
  setNodeWidgetValue(injectInfo, ["frame_rate"], Number(widget(editorNode, "frame_rate")?.value || 24));
  setNodeWidgetValue(injectInfo, ["file_prefix"], isMono ? "dialogue_tts_single_master" : "dialogue_tts_stem");
  connectBySlotName(editorNode, ["cine_linx"], exportInfo, ["cine_linx"], "IAMCCS_SUPERNODE_LINX");
  if (registry.IAMCCS_DialogueAudioBoardBridge) {
    const bridge = rigNodes.bridge || createGraphNode(lite, graph, "IAMCCS_DialogueAudioBoardBridge", [baseX, baseY + 245], "Dialogue AudioBoard Bridge - placeholder lanes");
    if (bridge) {
      if (!rigNodes.bridge) created.push(bridge);
      connectBySlotName(editorNode, ["cine_linx"], bridge, ["cine_linx"], "IAMCCS_SUPERNODE_LINX");
      connectBySlotName(bridge, ["cine_linx"], injectInfo, ["cine_linx"], "IAMCCS_SUPERNODE_LINX");
    }
  } else {
    connectBySlotName(editorNode, ["cine_linx"], injectInfo, ["cine_linx"], "IAMCCS_SUPERNODE_LINX");
  }
  const removedOldRigNodes = removeExistingAudioRigNodes(exportInfo, injectInfo);

  const makeSelected = (index, label) => {
    const node = createGraphNode(lite, graph, selectedType, [baseX + 300, baseY + index * 245], "DIRECT TTS " + label + " - " + ttsNodeLabel(selectedType));
    if (!node) return null;
    created.push(node);
    configureRigTTSNode(node, selectedType, index, data);
    const mode = String(data.settings?.tts_export_mode || "qwen");
    setNodeWidgetValue(node, ["text", "dialogue", "prompt", "target_text", "srt_content"], isMono ? ttsExportText(data, mode) : speakerPlainText(data, index));
    setNodeWidgetValue(node, ["instruct", "custom_instruct", "instruction", "system_prompt"], isMono ? ttsExportText(data, "qwen") : speakerPlainText(data, index));
    return node;
  };
  const sourceFor = (speakerIndex) => {
    if (isMono) return ["tts_srt", "tts_text"];
    return speakerIndex === 0 ? ["speaker_a_srt"] : ["speaker_b_srt"];
  };
  const targetAudioInput = (speakerIndex) => isMono || speakerIndex === 0 ? ["generated_audio"] : ["generated_audio_b"];

  const middleY = reusableInject ? Math.min(Number(reusableInject.pos?.[1] || baseY), baseY) : baseY;
  const probe = createGraphNode(lite, graph, selectedType, [baseX + 300, middleY], "DIRECT TTS A - " + ttsNodeLabel(selectedType));
  if (!probe) {
    if (statusTarget) statusTarget.textContent = "Could not create selected TTS node.";
    return created.length;
  }
  created.push(probe);
  configureRigTTSNode(probe, selectedType, 0, data);
  setNodeWidgetValue(probe, ["text", "dialogue", "prompt", "target_text", "srt_content"], isMono ? ttsExportText(data, String(data.settings?.tts_export_mode || "qwen")) : speakerPlainText(data, 0));
  setNodeWidgetValue(probe, ["instruct", "custom_instruct", "instruction", "system_prompt"], isMono ? ttsExportText(data, "qwen") : speakerPlainText(data, 0));
  const selectedIsEngine = nodeHasOutputType(probe, "TTS_ENGINE");
  const selectedIsAudio = nodeHasOutputType(probe, "AUDIO");
  if (selectedIsEngine && registry.UnifiedTTSSRTNode) {
    setNodeWidgetValue(probe, ["instruct", "custom_instruct", "instruction", "system_prompt"], isMono ? ttsExportText(data, "qwen") : speakerPlainText(data, 0));
    const engineNodes = [probe];
    if (!isMono) {
      const secondEngine = makeSelected(1, "B");
      if (secondEngine) engineNodes.push(secondEngine);
    }
    const count = isMono ? 1 : 2;
    for (let i = 0; i < count; i++) {
      const srt = createGraphNode(lite, graph, "UnifiedTTSSRTNode", [baseX + 600, middleY + i * 245], "DIRECT TTS " + (i === 0 ? "A" : "B") + " - " + (i === 0 ? "Man A" : "Man B") + " SRT to audio");
      if (!srt) continue;
      created.push(srt);
      configureRigSRTNode(srt, i);
      setNodeWidgetValue(srt, ["srt_content", "text", "dialogue", "prompt"], isMono ? ttsExportText(data, "chatterbox") : speakerSrtText(data, i));
      connectBySlotName(engineNodes[i] || probe, ["TTS_engine", "tts_engine"], srt, ["TTS_engine", "tts_engine"], "TTS_ENGINE");
      connectBySlotName(exportInfo, sourceFor(i), srt, ["srt_content"], isMono ? "STRING" : "");
      connectBySlotName(srt, ["audio"], injectInfo, targetAudioInput(i), "AUDIO");
      connectBySlotName(srt, ["Adjusted_SRT", "adjusted_srt"], injectInfo, isMono || i === 0 ? ["adjusted_srt"] : ["adjusted_srt_b"], "STRING");
      if (i === 0) connectBySlotName(srt, ["timing_report"], injectInfo, ["timing_report"], "STRING");
    }
  } else if (selectedIsAudio || nodeHasInputType(probe, "STRING")) {
    const nodes = [probe];
    if (!isMono) {
      const second = makeSelected(1, "B");
      if (second) nodes.push(second);
    }
    nodes.forEach((ttsNode, index) => {
      connectBySlotName(exportInfo, sourceFor(index), ttsNode, ["text", "dialogue", "prompt", "target_text", "srt_content"], isMono ? "STRING" : "");
      connectBySlotName(ttsNode, ["audio", "AUDIO"], injectInfo, targetAudioInput(index), "AUDIO");
    });
  } else {
    if (statusTarget) statusTarget.textContent = "Rig created with CineAudioInfo, but selected node exposes neither AUDIO nor TTS_ENGINE.";
  }
  graph.setDirtyCanvas?.(true, true);
  app.graph?.setDirtyCanvas?.(true, true);
  if (statusTarget) {
    const reuseText = reusableExport || reusableInject ? "Reused existing CineAudioInfo nodes. " : "";
    const removeText = removedOldRigNodes ? "Removed " + removedOldRigNodes + " old TTS rig node" + (removedOldRigNodes === 1 ? ". " : "s. ") : "";
    statusTarget.textContent = reuseText + removeText + "Created " + (isMono ? "mono" : "A/B") + " rig for " + ttsNodeLabel(selectedType) + ". Emotion routing: " + emotionRoutingLabel(emotionRoutingValue(data)) + ".";
  }
  return created.length;
}
function parseScript(text, data) {
  const old = data.lines || [];
  const lines = [];
  String(text || "").split(/\n+/).forEach((raw, index) => {
    const value = raw.trim();
    if (!value) return;
    const match = value.match(/^\[([^\]]+)\]\s*(.*)$/);
    const atSpeakerMatch = value.match(/^@([A-Za-z])#\s*(.*)$/);
    const line = { ...(old[index] || {}), id: old[index]?.id || "line_" + String(index + 1).padStart(3, "0"), speaker: index % 2 ? "B" : "A", text: value, emotion: "none", style: "none", paralinguistic: "none", tts_model: "", extra_tags: [], overlap_after: 0, ref: index % 2 ? 2 : 1, track: index % 2 ? 1 : 0 };
    if (match) {
      line.text = match[2].trim();
      match[1].split("|").map((part) => part.trim()).filter(Boolean).forEach((part, partIndex) => {
        const [rawKey, ...rest] = part.split(":");
        const key = rawKey.trim();
        const val = rest.join(":").trim();
        if (partIndex === 0 && !rest.length) line.speaker = key.slice(0, 1).toUpperCase();
        else if (key === "tts") line.tts_model = val || "";
        else if (key === "emotion") line.emotion = val || "none";
        else if (key === "style") line.style = val || "none";
        else if (key === "para") line.paralinguistic = val || "none";
        else if (key === "overlap") line.overlap_after = Math.max(0, Number(val || 0));
        else if (key === "duration") line.duration = Math.max(0, Number(val || 0));
        else if (key === "ref") line.ref = Math.max(1, Number(val || 1));
        else if (rest.length) {
          line.extra_tags.push(part);
          if (/emotion|mood/i.test(key) && line.emotion === "none") line.emotion = val || "none";
          if (/style/i.test(key) && line.style === "none") line.style = val || "none";
        }
      });
    } else if (atSpeakerMatch) {
      line.speaker = String(atSpeakerMatch[1] || "A").toUpperCase();
      line.text = String(atSpeakerMatch[2] || "").trim();
    }
    line.track = line.speaker === "B" ? 1 : 0;
    line.local_prompt ||= line.speaker === "B" ? "hard cut, Man B close-up, Man B speaks clearly, visible mouth movement, Man A listens quietly" : "hard cut, Man A close-up, Man A speaks clearly, visible mouth movement, Man B listens quietly";
    lines.push(line);
  });
  return { ...data, lines };
}

// By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
// By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
// By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
// By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
// By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
function overlapPresetValue(value) {
  const numeric = Math.max(0, Number(value || 0));
  for (const [preset] of OVERLAP_PRESETS) {
    if (Math.abs(Number(preset) - numeric) < 0.005) return preset;
  }
  return "custom";
}

function promptPreviewTokens(text) {
  return String(text || "")
    .split(/[,|]/)
    .map((token) => token.trim())
    .filter(Boolean)
    .slice(0, 6);
}

function lineTokenMeta(line) {
  return [
    { kind: "speaker", label: String(line.speaker || "A") },
    { kind: "emotion", label: "emotion: " + String(line.emotion || "none") },
    { kind: "style", label: "style: " + String(line.style || "none") },
    { kind: "para", label: "para: " + String(line.paralinguistic || "none") },
    { kind: "overlap", label: "overlap: " + Number(line.overlap_after || 0).toFixed(2) + "s" },
    { kind: "ref", label: "ref: " + String(line.ref || 1) },
  ];
}

function linePreview(node) {
  const wrap = document.createElement("div");
  wrap.className = "iamccs-token-row";
  lineTokenMeta(node).forEach((entry) => {
    const chip = document.createElement("span");
    chip.className = "iamccs-token " + entry.kind;
    chip.textContent = entry.label;
    wrap.append(chip);
  });
  return wrap;
}

function promptPreview(text) {
  const wrap = document.createElement("div");
  wrap.className = "iamccs-token-row iamccs-token-row-prompt";
  const tokens = promptPreviewTokens(text);
  if (!tokens.length) {
    const empty = document.createElement("span");
    empty.className = "iamccs-token prompt";
    empty.textContent = "local prompt preview";
    wrap.append(empty);
    return wrap;
  }
  tokens.forEach((token) => {
    const chip = document.createElement("span");
    chip.className = "iamccs-token prompt";
    chip.textContent = token;
    wrap.append(chip);
  });
  return wrap;
}

function stripFormattingTokens(text) {
  return String(text || "").replace(/\*\*(.*?)\*\*/g, "$1");
}

function applyOverlapToTaggedLine(sourceText, caret, value) {
  const text = String(sourceText || "");
  const numeric = Math.max(0, Number(value || 0));
  const formatted = numeric.toFixed(2);
  const token = "overlap:" + formatted;
  const safeCaret = Math.max(0, Math.min(Number(caret || 0), text.length));
  const lineStart = text.lastIndexOf("\n", Math.max(0, safeCaret - 1)) + 1;
  const nextNewline = text.indexOf("\n", safeCaret);
  const lineEnd = nextNewline === -1 ? text.length : nextNewline;
  const line = text.slice(lineStart, lineEnd);
  const tagMatch = line.match(/^\[[^\]]*\]/);
  if (!tagMatch) {
    const inserted = text.slice(0, safeCaret) + "|" + token + text.slice(safeCaret);
    return { text: inserted, caret: safeCaret + token.length + 1 };
  }
  const tag = tagMatch[0];
  const overlapPattern = /\|overlap:[^|\]]+/;
  const nextTag = overlapPattern.test(tag)
    ? tag.replace(overlapPattern, "|" + token)
    : tag.slice(0, -1) + "|" + token + "]";
  const updatedLine = nextTag + line.slice(tag.length);
  const updatedText = text.slice(0, lineStart) + updatedLine + text.slice(lineEnd);
  const overlapIndex = updatedLine.indexOf(token);
  return { text: updatedText, caret: lineStart + overlapIndex + token.length };
}

function applyMetadataToTaggedLine(sourceText, caret, key, rawValue) {
  const text = String(sourceText || "");
  const cleanKey = String(key || "").trim();
  const cleanValue = String(rawValue || "").trim();
  const safeCaret = Math.max(0, Math.min(Number(caret || 0), text.length));
  const lineStart = text.lastIndexOf("\n", Math.max(0, safeCaret - 1)) + 1;
  const nextNewline = text.indexOf("\n", safeCaret);
  const lineEnd = nextNewline === -1 ? text.length : nextNewline;
  const line = text.slice(lineStart, lineEnd);
  const bracketMatch = line.match(/^\[([^\]]*)\](\s*)(.*)$/);
  const atSpeakerMatch = line.match(/^@([A-Za-z])#\s*(.*)$/);
  const speaker = bracketMatch
    ? String(bracketMatch[1] || "A").split("|")[0].trim().slice(0, 1).toUpperCase()
    : atSpeakerMatch
      ? String(atSpeakerMatch[1] || "A").toUpperCase()
      : "A";
  const body = bracketMatch ? (bracketMatch[3] || "") : atSpeakerMatch ? (atSpeakerMatch[2] || "") : line;
  const rawParts = bracketMatch ? String(bracketMatch[1] || "").split("|").map((part) => part.trim()).filter(Boolean) : [speaker];
  const parts = [];
  let found = false;
  rawParts.forEach((part, index) => {
    if (index === 0 && !part.includes(":")) {
      parts.push(speaker || part);
      return;
    }
    const [partKey] = part.split(":");
    if (partKey.trim() === cleanKey) {
      parts.push(cleanKey + ":" + cleanValue);
      found = true;
    } else {
      parts.push(part);
    }
  });
  if (!parts.length || parts[0].includes(":")) parts.unshift(speaker || "A");
  if (!found) parts.push(cleanKey + ":" + cleanValue);
  const updatedLine = "[" + parts.join("|") + "] " + body.trimStart();
  const updatedText = text.slice(0, lineStart) + updatedLine + text.slice(lineEnd);
  const token = cleanKey + ":" + cleanValue;
  const tokenIndex = updatedLine.indexOf(token);
  return { text: updatedText, caret: lineStart + (tokenIndex >= 0 ? tokenIndex + token.length : updatedLine.length) };
}

function lineBoundsForCaret(text, caret) {
  const value = String(text || "");
  const safeCaret = Math.max(0, Math.min(Number(caret || 0), value.length));
  const lineStart = value.lastIndexOf("\n", Math.max(0, safeCaret - 1)) + 1;
  const nextNewline = value.indexOf("\n", safeCaret);
  const lineEnd = nextNewline === -1 ? value.length : nextNewline;
  return { safeCaret, lineStart, lineEnd, line: value.slice(lineStart, lineEnd), localCaret: safeCaret - lineStart };
}

function speakerFromTaggedLine(line) {
  const bracket = String(line || "").match(/^\[([^\]]*)\]/);
  if (bracket) {
    const head = String(bracket[1] || "A").split(/[|:]/)[0].trim();
    if (head) return head.slice(0, 1).toUpperCase();
  }
  const at = String(line || "").match(/^@([A-Za-z])#/);
  if (at) return String(at[1] || "A").toUpperCase();
  return "A";
}

function normalizeInlineTagValue(value) {
  return String(value || "").trim().replace(/[\[\]<>|]/g, " ").replace(/\s+/g, "_");
}

function chatterboxTokenFor(kind, value) {
  const raw = String(value || "").trim();
  const key = raw.toLowerCase();
  const map = {
    breathing: "inhale", laughter: "laughter", sigh: "sigh", uhm: "UM", "surprise-oh": "gasp",
    whisper: "whisper", murmur: "mumble", low: "mumble", shout: "gasp", excited: "gasp", surprised: "gasp", fearful: "gasp", sad: "sigh", angry: "groan"
  };
  const token = map[key] || (kind === "para" ? raw : "");
  return token ? "<" + normalizeInlineTagValue(token) + "> " : "";
}

function performanceTagTokenFor(kind, value, routing, speaker) {
  const clean = normalizeInlineTagValue(value);
  const route = String(routing || "clean_metadata");
  const who = normalizeInlineTagValue(speaker || "A") || "A";
  if (!clean) return "";
  if (route === "index_tts_character_tags") return "[" + who + ":" + clean + "] ";
  if (route === "index_tts_text_emotion") return "[" + who + ":" + clean + "] ";
  if (route === "chatterbox_v2_tokens") return chatterboxTokenFor(kind, clean) || "<" + clean + "> ";
  if (route === "step_editx_tags") return kind === "para" ? "<" + clean + "> " : "<" + kind + ":" + clean + "> ";
  return kind === "para" ? "<" + clean + "> " : "[" + who + ":" + clean + "] ";
}

function insertPerformanceTagAtTaggedCursor(sourceText, caret, kind, rawValue, routing) {
  const text = String(sourceText || "");
  const bounds = lineBoundsForCaret(text, caret);
  const speaker = speakerFromTaggedLine(bounds.line);
  const token = performanceTagTokenFor(kind, rawValue, routing, speaker);
  if (!token) return { text, caret: bounds.safeCaret };
  const structuralMatch = bounds.line.match(/^\[[^\]]*\]/);
  const structuralEnd = structuralMatch ? structuralMatch[0].length : 0;
  const tagRegex = /(\[[^\]]+:[^\]]+\]|<[^>]+>)/g;
  let match;
  while ((match = tagRegex.exec(bounds.line))) {
    const start = match.index;
    const end = start + match[0].length;
    if (start === 0 && end <= structuralEnd && bounds.line.slice(0, end).includes("|")) continue;
    if (bounds.localCaret >= start && bounds.localCaret <= end) {
      const from = bounds.lineStart + start;
      const to = bounds.lineStart + end;
      const nextText = text.slice(0, from) + token + text.slice(to).replace(/^\s+/, "");
      return { text: nextText, caret: from + token.length };
    }
  }
  let insertAt = bounds.safeCaret;
  if (structuralEnd && bounds.localCaret <= structuralEnd) insertAt = bounds.lineStart + structuralEnd + (bounds.line.charAt(structuralEnd) === " " ? 1 : 0);
  const before = text.slice(0, insertAt);
  const after = text.slice(insertAt);
  const leftPad = before && !/[\s]$/.test(before) ? " " : "";
  const rightTrimmed = after.replace(/^\s+/, "");
  const nextText = before + leftPad + token + rightTrimmed;
  return { text: nextText, caret: before.length + leftPad.length + token.length };
}

function escapeHtml(value) {
  return String(value || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;");
}

function renderScriptHighlight(text) {
  return String(text || "")
    .split("\n")
    .map((line) => {
      const value = String(line || "");
      const match = value.match(/^(\[[^\]]*\])(\s*)(.*)$/);
      if (!match) return '<div class="iamccs-script-line">' + escapeHtml(value || " ") + "</div>";
      const prefix = escapeHtml(match[1])
        .replace(/(emotion:[^|\]]+)/g, '<span class="iamccs-script-emotion">$1</span>')
        .replace(/(style:[^|\]]+)/g, '<span class="iamccs-script-style">$1</span>')
        .replace(/(para:[^|\]]+)/g, '<span class="iamccs-script-para">$1</span>')
        .replace(/(overlap:[^|\]]+)/g, '<span class="iamccs-script-overlap">$1</span>')
        .replace(/(ref:[^|\]]+)/g, '<span class="iamccs-script-ref">$1</span>')
        .replace(/^\[A(?=\||\])/g, '<span class="iamccs-script-speaker-a">[A</span>')
        .replace(/^\[B(?=\||\])/g, '<span class="iamccs-script-speaker-b">[B</span>');
      const body = escapeHtml(match[3] || "").replace(/\*\*(.*?)\*\*/g, '<strong class="iamccs-script-bold">$1</strong>');
      return '<div class="iamccs-script-line"><span class="iamccs-script-prefix">' + prefix + '</span>' + escapeHtml(match[2] || "") + '<span class="iamccs-script-text">' + body + '</span></div>';
    })
    .join("");
}

function wrapSelection(target, before, after = before) {
  if (target instanceof HTMLTextAreaElement || target instanceof HTMLInputElement) {
    const start = target.selectionStart ?? target.value.length;
    const end = target.selectionEnd ?? start;
    const selected = target.value.slice(start, end);
    target.value = target.value.slice(0, start) + before + selected + after + target.value.slice(end);
    target.focus();
    const caret = start + before.length + selected.length;
    target.selectionStart = caret;
    target.selectionEnd = caret;
    target.dispatchEvent(new Event("input", { bubbles: true }));
    return;
  }
  if (target?.isContentEditable) {
    target.focus();
    const selection = window.getSelection?.();
    if (!selection || !selection.rangeCount) {
      insertAt(target, before + after);
      return;
    }
    const range = selection.getRangeAt(0);
    if (!target.contains(range.startContainer)) {
      insertAt(target, before + after);
      return;
    }
    const selected = range.toString();
    range.deleteContents();
    const node = document.createTextNode(before + selected + after);
    range.insertNode(node);
    const next = document.createRange();
    const caret = before.length + selected.length;
    next.setStart(node, caret);
    next.collapse(true);
    selection.removeAllRanges();
    selection.addRange(next);
    target.dispatchEvent(new Event("input", { bubbles: true }));
  }
}

function editorPlainText(element) {
  return String(element?.innerText || "")
    .replace(/\u00a0/g, " ")
    .replace(/\r/g, "")
    .replace(/\n$/, "");
}

function caretOffset(root) {
  const selection = window.getSelection?.();
  if (!selection || !selection.rangeCount) return null;
  const range = selection.getRangeAt(0);
  if (!root.contains(range.startContainer)) return null;
  const probe = range.cloneRange();
  probe.selectNodeContents(root);
  probe.setEnd(range.startContainer, range.startOffset);
  return probe.toString().length;
}

function setCaretOffset(root, offset) {
  if (offset == null) return;
  const selection = window.getSelection?.();
  if (!selection) return;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let remaining = offset;
  let node = walker.nextNode();
  while (node) {
    const length = node.textContent?.length || 0;
    if (remaining <= length) {
      const range = document.createRange();
      range.setStart(node, Math.max(0, remaining));
      range.collapse(true);
      selection.removeAllRanges();
      selection.addRange(range);
      return;
    }
    remaining -= length;
    node = walker.nextNode();
  }
  const range = document.createRange();
  range.selectNodeContents(root);
  range.collapse(false);
  selection.removeAllRanges();
  selection.addRange(range);
}

function ensureStyle() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
  .iamccs-dte{--ink:#242018;--paper:#f7f1e4;--paper2:#efe5d0;--desk:#181713;--brass:#b8893b;--brass2:#d6b66f;--green:#5e7d72;--blue:#516f8f;--tag:#7c5a91;--line:#d7c8ab;width:100%;height:100%;box-sizing:border-box;background:linear-gradient(180deg,#24231d 0%,#171713 100%);color:#efe8d8;border:1px solid rgba(214,182,111,.42);border-radius:8px;overflow:hidden;font:12px/1.45 Inter,ui-sans-serif,system-ui,sans-serif;display:grid;grid-template-rows:58px 1fr 34px;box-shadow:inset 0 0 0 1px rgba(255,255,255,.03)}
  .iamccs-dte *{box-sizing:border-box}.iamccs-dte-head{display:flex;align-items:center;justify-content:space-between;padding:12px 16px;background:#2b2a22;border-bottom:1px solid rgba(214,182,111,.36)}
  .iamccs-dte-title{font-family:Georgia,serif;font-size:18px;font-weight:900;color:#f7ecd0;letter-spacing:.01em}.iamccs-dte-sub{font-size:10px;color:#c9bea4;margin-top:2px}.iamccs-dte-actions{display:flex;gap:8px;align-items:center}
  .iamccs-dte-main{display:grid;grid-template-columns:236px 1fr 286px;min-height:0;padding:12px;gap:12px;background:#191813;overflow:hidden}.iamccs-dte-side,.iamccs-dte-tags,.iamccs-dte-center{min-width:0;border:1px solid rgba(214,182,111,.26);border-radius:8px;box-shadow:0 1px 0 rgba(255,255,255,.04)}
  .iamccs-dte-side,.iamccs-dte-tags{background:#211f18;padding:14px;min-height:0;overflow:auto;overscroll-behavior:contain}.iamccs-dte-center{display:grid;grid-template-rows:auto auto auto minmax(260px,1fr);min-height:0;background:#ebe0c8;color:var(--ink);overflow:auto}
  .iamccs-dte-toolbar{display:flex;align-items:center;gap:8px;padding:8px 10px;border-bottom:1px solid var(--line);background:#d9c9a7}.iamccs-dte-section{padding:14px 16px;border-bottom:1px solid var(--line);min-height:0;background:#f2ead8}.iamccs-dte-section.fill{border-bottom:0;overflow:hidden;display:grid;grid-template-rows:auto 1fr}
  .iamccs-dte-label{display:flex;align-items:center;justify-content:space-between;margin:0 0 10px;padding:6px 8px;border-left:4px solid var(--brass);background:rgba(184,137,59,.1);color:#5c431c;text-transform:uppercase;font-size:10px;font-weight:950;letter-spacing:.055em}.iamccs-dte-label span:last-child{color:#7b705d;text-transform:none;font-weight:750;letter-spacing:0}
  textarea.iamccs-box{display:block;width:100%;height:auto;min-height:54px;max-width:100%;padding:12px 14px;resize:vertical;overflow:auto;background:#0a0c10;color:#f4f7fb;border:1px solid #434b57;border-radius:6px;font:13px/1.55 "Courier New",ui-monospace,SFMono-Regular,Consolas,monospace;outline:none;box-shadow:inset 0 1px 2px rgba(0,0,0,.32)}
  textarea.iamccs-box::placeholder{color:#9aa5b4}
  textarea.iamccs-box:focus{border-color:#6d8db7;box-shadow:0 0 0 2px rgba(109,141,183,.22),inset 0 1px 2px rgba(0,0,0,.28)}.iamccs-dte.light textarea.iamccs-box{background:#fff;color:#111;border-color:#b7a37c;box-shadow:inset 0 1px 2px rgba(42,31,12,.08)}
  .iamccs-dte.light textarea.iamccs-box::placeholder{color:#77694f}
  textarea.iamccs-global{color:#d9f3ea;background:#0b1210}textarea.iamccs-script{color:#dbe7ff;background:#0a0d14}textarea.iamccs-dialogue{color:#f6f3ea;background:#0b0907}textarea.iamccs-local{color:#f0dcff;background:#0d0a12}
  .iamccs-dte.light textarea.iamccs-global{color:#314f45;background:#fbf7ec}.iamccs-dte.light textarea.iamccs-script{color:#2d3548;background:#fffdf7}.iamccs-dte.light textarea.iamccs-dialogue{color:#21180f;background:#fffaf0}.iamccs-dte.light textarea.iamccs-local{color:#5b316c;background:#fbf7ff}
  .iamccs-dte-script{min-height:152px}.iamccs-rich-editor{width:100%;min-height:152px;max-height:520px;resize:vertical;overflow:auto;padding:12px 14px;background:#0a0d14;color:#dbe7ff;border:1px solid #434b57;border-radius:6px;font:13px/1.55 "Courier New",ui-monospace,SFMono-Regular,Consolas,monospace;outline:none;white-space:pre-wrap;word-break:break-word}.iamccs-rich-editor:focus{border-color:#6d8db7;box-shadow:0 0 0 2px rgba(109,141,183,.22),inset 0 1px 2px rgba(0,0,0,.28)}.iamccs-dte.light .iamccs-rich-editor{background:#fffdf7;color:#2d3548;border-color:#b7a37c}
  .iamccs-script-line{white-space:pre-wrap;word-break:break-word}.iamccs-script-prefix{color:#c8d2e6}.iamccs-script-text{color:#f4f7fb}.iamccs-dte.light .iamccs-script-text{color:#1f1b14}.iamccs-script-bold{font-weight:900;color:inherit}.iamccs-script-speaker-a{color:#6fc0ff;font-weight:900}.iamccs-script-speaker-b{color:#ffb36f;font-weight:900}.iamccs-script-emotion{color:#ff7f9d}.iamccs-script-style{color:#c6a0ff}.iamccs-script-para{color:#62d8bf}.iamccs-script-overlap{color:#ffd36f}.iamccs-script-ref{color:#b7bec8}
  .iamccs-dte-grid{display:grid;grid-template-columns:1fr 1fr;grid-auto-rows:max-content;gap:12px;align-content:start;align-items:start;overflow:auto;padding-right:4px}
  .iamccs-dte-grid > *{min-width:0;min-height:0}
  .iamccs-dte-card{display:flex;flex-direction:column;align-items:stretch;gap:10px;padding:12px;background:#f8f0dd;border:1px solid #d2bd8e;border-radius:7px;min-height:0;overflow:hidden}.iamccs-dte-card-head{display:flex;align-items:center;justify-content:space-between;color:#4a3210;font-weight:950;font-size:12px}.iamccs-dte-card-meta{color:#77694f;font-size:10px}
  .iamccs-dte-card-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}.iamccs-dte-card-grid.tight{grid-template-columns:1.2fr .8fr 1fr}.iamccs-dte-card-field{display:grid;gap:4px}.iamccs-dte-card-field label{color:#65563f;font-size:10px;font-weight:900;text-transform:uppercase;letter-spacing:.05em}
  .iamccs-token-row{display:flex;flex-wrap:wrap;gap:6px;align-items:center}.iamccs-token{display:inline-flex;align-items:center;min-height:24px;padding:0 9px;border-radius:999px;font-size:10px;font-weight:900;letter-spacing:.02em;border:1px solid transparent}
  .iamccs-token.speaker{background:#18354f;color:#d9eeff;border-color:#5686ba}.iamccs-token.emotion{background:#4d1c28;color:#ffd9e2;border-color:#b6647a}.iamccs-token.style{background:#2d234e;color:#eadcff;border-color:#8f75c7}.iamccs-token.para{background:#163e38;color:#ddfff5;border-color:#4b9b89}.iamccs-token.overlap{background:#4d3413;color:#ffe7b5;border-color:#ba8a3d}.iamccs-token.ref{background:#3b3b3b;color:#f3f3f3;border-color:#8d8d8d}.iamccs-token.prompt{background:#2c3140;color:#e0ebff;border-color:#647493}
  .iamccs-overlap-help{color:#7a6a51;font-size:10px;font-weight:700;line-height:1.4}.iamccs-dte.light .iamccs-overlap-help{color:#7a6a51}
  .iamccs-dte:not(.light) .iamccs-dte-center{background:#12151b;color:#f4f7fb}.iamccs-dte:not(.light) .iamccs-dte-toolbar{background:#171c25;border-bottom-color:#2d3542}.iamccs-dte:not(.light) .iamccs-dte-section{background:#12151b;border-bottom-color:#29303b}.iamccs-dte:not(.light) .iamccs-dte-label{background:rgba(109,141,183,.12);border-left-color:#6d8db7;color:#dbe7ff}.iamccs-dte:not(.light) .iamccs-dte-label span:last-child{color:#a8b4c6}.iamccs-dte:not(.light) .iamccs-dte-card{background:#171b23;border-color:#313949}.iamccs-dte:not(.light) .iamccs-dte-card-head{color:#eef4ff}.iamccs-dte:not(.light) .iamccs-dte-card-meta{color:#aab5c6}.iamccs-dte:not(.light) .iamccs-overlap-help{color:#a8b4c6}
  .iamccs-dte h4{margin:0 0 10px;padding:5px 7px;border-left:4px solid var(--brass);background:rgba(214,182,111,.12);color:#f4dfad;text-transform:uppercase;font-size:10px;letter-spacing:.05em}.iamccs-dte button,.iamccs-dte select,.iamccs-dte input{font:inherit;color:#f6ead1;background:#2a2b23;border:1px solid rgba(214,182,111,.38);border-radius:6px}
  .iamccs-dte button{height:28px;padding:0 10px;font-weight:900;cursor:pointer;transition:transform .12s ease,box-shadow .18s ease,background .18s ease,color .18s ease}.iamccs-dte button:hover{border-color:#f1ce78;color:#fff6d8;background:#343426}.iamccs-dte .gold{background:#d6b66f;border-color:#f0d892;color:#21180b}.iamccs-dte .active,.iamccs-dte button[aria-pressed="true"]{outline:2px solid rgba(133,235,145,.72);background:linear-gradient(180deg,#b9e98e,#69aa61);border-color:#d8f5b7;color:#10210f;box-shadow:0 0 0 3px rgba(105,170,97,.20),0 0 16px rgba(105,220,117,.30);transform:translateY(1px)}.iamccs-dte button[aria-pressed="true"]::after{content:" ON";font-size:8px;font-weight:950;color:#173719}
  .iamccs-dte .iamccs-inject-btn{min-width:110px;box-shadow:0 0 0 0 rgba(214,182,111,0)}.iamccs-dte .iamccs-inject-btn.is-busy{transform:translateY(1px) scale(.985);box-shadow:0 0 0 2px rgba(214,182,111,.28),0 0 18px rgba(214,182,111,.22);filter:saturate(1.12)}.iamccs-dte .iamccs-inject-btn.is-done{background:#84b86b;border-color:#ccecad;color:#10200e;box-shadow:0 0 0 2px rgba(132,184,107,.24),0 0 16px rgba(132,184,107,.16)}
  .iamccs-dte select,.iamccs-dte input{height:30px;padding:4px 8px;width:100%;margin-bottom:10px}.iamccs-dte-speaker{padding:12px;margin-bottom:12px;background:#26251d;border:1px solid rgba(214,182,111,.34);border-radius:7px}
  .iamccs-overlap-panel{padding:12px;margin-bottom:12px;background:#26251d;border:1px solid rgba(214,182,111,.34);border-radius:7px}.iamccs-overlap-row{display:grid;grid-template-columns:1fr auto;gap:8px;align-items:end}.iamccs-overlap-row input{margin-bottom:0}.iamccs-overlap-preset-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}.iamccs-overlap-preset-grid button{width:100%;height:30px;padding:0 7px;font-size:10px;text-align:center;white-space:normal;line-height:1.05;overflow:hidden}.iamccs-tts-convert-grid{grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.iamccs-tts-convert-grid button{height:36px;font-size:9px}.iamccs-dte button.iamccs-convert-success{background:#5fb56f;border-color:#a9e6b5;color:#071c0d;box-shadow:0 0 0 2px rgba(95,181,111,.2)}.iamccs-dte button.iamccs-convert-active{outline:2px solid rgba(95,181,111,.46)}.iamccs-tts-node-select{height:30px!important;margin:8px 0 12px}.iamccs-node-actions{display:grid;gap:10px;justify-items:center;margin:10px 0 4px}.iamccs-node-actions button{width:100%;max-width:220px;min-height:38px;height:auto;padding:8px 12px;line-height:1.15}.iamccs-overlap-hint{margin-top:10px;color:#c9bea4;font-size:10px;line-height:1.4}.iamccs-dte.light .iamccs-overlap-hint{color:#6f675b}
  .iamccs-tag-table{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-bottom:16px}.iamccs-tag-table button{width:100%;height:30px;padding:0 7px;font-size:10px;text-align:center}.iamccs-tag-table button:nth-child(3n+1){border-color:rgba(126,90,145,.65);color:#ead9f3}.iamccs-tag-table button:nth-child(3n+2){border-color:rgba(81,111,143,.65);color:#dbeaff}.iamccs-tag-table button:nth-child(3n){border-color:rgba(94,125,114,.65);color:#ddf4ea}
  .iamccs-dte-foot{display:flex;align-items:center;justify-content:space-between;padding:8px 14px;background:#24231d;border-top:1px solid rgba(214,182,111,.22);color:#cabd9f;font-size:10px}.iamccs-dte-foot.iamccs-inject-status{background:#2a2517;color:#f7ecc9;border-top-color:rgba(214,182,111,.34)}.iamccs-dte-foot.iamccs-inject-status-success{background:#1f2b1c;color:#e5f6d7;border-top-color:rgba(132,184,107,.36)}.iamccs-field-wrap{display:flex;flex-direction:column;align-items:stretch;gap:8px;min-height:0;overflow:hidden}.iamccs-field-wrap .iamccs-dte-label{margin-bottom:0;padding:7px 9px}
  
  `;
  document.head.appendChild(style);
}
function insertAt(target, token) {
  if (target instanceof HTMLTextAreaElement || target instanceof HTMLInputElement) {
    const start = target.selectionStart ?? target.value.length;
    const end = target.selectionEnd ?? start;
    target.value = target.value.slice(0, start) + token + target.value.slice(end);
    target.focus();
    target.selectionStart = target.selectionEnd = start + token.length;
    target.dispatchEvent(new Event("input", { bubbles: true }));
    return;
  }
  if (target?.isContentEditable) {
    target.focus();
    const selection = window.getSelection?.();
    if (selection && selection.rangeCount) {
      const range = selection.getRangeAt(0);
      if (target.contains(range.startContainer)) {
        range.deleteContents();
        const node = document.createTextNode(token);
        range.insertNode(node);
        range.setStartAfter(node);
        range.collapse(true);
        selection.removeAllRanges();
        selection.addRange(range);
      } else {
        target.append(document.createTextNode(token));
      }
    } else {
      target.append(document.createTextNode(token));
    }
    target.dispatchEvent(new Event("input", { bubbles: true }));
  }
}
function downstream(start, accept) {
  const out = [];
  const seen = new Set();
  const queue = [start];
  while (queue.length) {
    const current = queue.shift();
    if (!current || seen.has(current.id)) continue;
    seen.add(current.id);
    for (const output of current.outputs || []) {
      for (const linkId of output.links || []) {
        const link = app.graph?.links?.[linkId];
        const next = link ? app.graph.getNodeById(link.target_id) : null;
        if (!next || seen.has(next.id)) continue;
        if (accept(next)) out.push(next);
        queue.push(next);
      }
    }
  }
  return out;
}
function estimateSeconds(text) {
  const words = String(text || "").trim().split(/\s+/).filter(Boolean).length;
  return Math.max(0.8, words ? (words / 130) * 60 + 0.22 : 2.4);
}
function buildInjectionPayload(data, fps = 24) {
  let cursor = 0;
  const segments = [];
  const audioSegments = [];
  const localPrompts = [];
  const lengths = [];
  (data.lines || []).forEach((line, index) => {
    const duration = Number(line.duration || 0) > 0 ? Number(line.duration) : estimateSeconds(line.text);
    const startSeconds = line.start !== undefined && line.start !== "" ? Number(line.start || 0) : cursor;
    const speaker = String(line.speaker || (index % 2 ? "B" : "A"));
    const start = Math.max(0, Math.round(startSeconds * fps));
    const length = Math.max(1, Math.round(duration * fps));
    const prompt = String(line.local_prompt || "");
    localPrompts.push(prompt);
    lengths.push(String(length));
    const cleanText = stripFormattingTokens(line.text || "");
    segments.push({ id: "shot_" + String(index + 1).padStart(3, "0") + "_" + speaker.toLowerCase(), type: "text", label: speaker === "B" ? "controcampo_B_" + String(index + 1).padStart(2, "0") : "campo_A_" + String(index + 1).padStart(2, "0"), start, length, ref: Number(line.ref || (speaker === "B" ? 2 : 1)), prompt, dialogue: speaker + ': "' + cleanText + '"', audio_or_dialogue: speaker + ': "' + cleanText + '"', emotion: String(line.emotion || "none"), style: String(line.style || "none"), use_prompt: true, use_guide: false, force: 0, guide_strength: 0, transition: index === 0 ? "opening_field" : "hard_cut" });
    audioSegments.push({ id: "dlg_" + String(index + 1).padStart(3, "0") + "_" + speaker.toLowerCase(), type: "audio", name: speaker + " " + String(index + 1).padStart(2, "0"), track: modeFromData(data) === "tts_master_unico" ? 0 : (speaker === "B" ? 1 : 0), start, length, audioDurationFrames: length, gain: 1, pan: 0, purpose: "dialogue_pending_tts", speaker, dialogueText: cleanText, pendingTTS: true, source: "IAMCCS_DialogueTagEditor_UI_Inject" });
    cursor = Math.max(cursor, startSeconds + duration + Number(data.settings?.default_gap_seconds || 0.12) - Math.max(0, Number(line.overlap_after || 0)));
  });
  const zeroStartStems = Boolean(data.settings?.speaker_stems_zero_start) && modeFromData(data) !== "tts_master_unico";
  if (zeroStartStems) {
    const firstBySpeaker = new Map();
    audioSegments.forEach((seg) => {
      const key = String(seg.speaker || seg.track || "A");
      firstBySpeaker.set(key, Math.min(firstBySpeaker.get(key) ?? Number(seg.start || 0), Number(seg.start || 0)));
    });
    audioSegments.forEach((seg) => {
      const key = String(seg.speaker || seg.track || "A");
      seg.start = Math.max(0, Number(seg.start || 0) - Number(firstBySpeaker.get(key) || 0));
      seg.speakerStemsZeroStart = true;
    });
  }
  const durationFrames = Math.max(0, ...segments.map((s) => s.start + s.length), ...audioSegments.map((s) => s.start + s.length));
  const audioTrackCount = modeFromData(data) === "tts_master_unico" ? 1 : 2;
  const audioBoard = { schema: "iamccs.audio_board_arranger", schema_version: 1, audioSegments, audioTrackCount, audioSyncMode: "timeline_audio", duration_seconds: durationFrames / fps, frame_rate: fps, masterAudioGain: 1, masterAudioNormalize: false, speakerStemsZeroStart: zeroStartStems, speakerStemSrtLocalZero: data.settings?.speaker_stem_srt_local_zero !== false, bridgeStatus: { source: "DialogueTagEditor UI Inject", pending_tts: true } };
  const timeline = { schema: "iamccs.cine.filmmaker_timeline", schema_version: 2, global_prompt: data.global_prompt || "", prompt: data.global_prompt || "", promptrelay_enabled: true, use_custom_audio: false, audioSyncMode: "timeline_audio", duration_seconds: durationFrames / fps, frame_rate: fps, director_local_prompts: localPrompts.join(" | "), local_prompts: localPrompts.join(" | "), director_segment_lengths: lengths.join(","), segment_lengths: lengths.join(","), segments, audioSegments, audioTrackCount, dialogue: data };
  return { audioBoard, timeline };
}
function safeJsonParse(text, fallback = {}) {
  try {
    const parsed = JSON.parse(String(text || ""));
    return parsed && typeof parsed === "object" ? parsed : fallback;
  } catch {
    return fallback;
  }
}
function isVisualShotboardSegment(segment) {
  return segment && String(segment.type || "image").toLowerCase() !== "audio" && !segment.placeholder;
}
function isVisualShotboardRow(row) {
  return row && row.placeholder !== true;
}
function mergeDialoguePromptsIntoShotboard(existingTimeline, data, fps = 24) {
  const merged = existingTimeline && typeof existingTimeline === "object" ? JSON.parse(JSON.stringify(existingTimeline)) : {};
  const promptEntries = (data.lines || []).map((line) => String(line.local_prompt || "").trim()).filter(Boolean);
  const visualSegments = Array.isArray(merged.segments) ? merged.segments.filter(isVisualShotboardSegment) : [];
  const visualRows = Array.isArray(merged.rows) ? merged.rows.filter(isVisualShotboardRow) : [];
  const appliedPromptCount = Math.min(promptEntries.length, visualSegments.length || visualRows.length || 0);
  const appliedPrompts = promptEntries.slice(0, appliedPromptCount);
  const segmentLengths = visualSegments.slice(0, appliedPromptCount).map((segment) => String(Math.max(1, Number(segment.length || 1))));

  merged.schema = merged.schema || "iamccs.cine.filmmaker_timeline";
  merged.schema_version = Math.max(2, Number(merged.schema_version || 2));
  merged.global_prompt = String(data.global_prompt || "");
  merged.prompt = String(data.global_prompt || "");
  merged.frame_rate = Math.max(1, Number(merged.frame_rate || fps || 24));
  merged.dialogue = data;

  visualSegments.slice(0, appliedPromptCount).forEach((segment, index) => {
    const prompt = appliedPrompts[index];
    if (!prompt) return;
    segment.prompt = prompt;
    segment.relay_prompt = prompt;
    segment.local_prompt = prompt;
    segment.use_prompt = true;
    segment.relay_manual_off = false;
    segment.promptrelay_manual_off = false;
  });
  visualRows.slice(0, appliedPromptCount).forEach((row, index) => {
    const prompt = appliedPrompts[index];
    if (!prompt) return;
    row.relay_prompt = prompt;
    row.local_prompt = prompt;
    row.use_prompt = true;
  });

  if (appliedPrompts.length) {
    merged.director_local_prompts = appliedPrompts.join(" | ");
    merged.local_prompts = appliedPrompts.join(" | ");
    merged.director_segment_lengths = segmentLengths.join(",");
    merged.segment_lengths = segmentLengths.join(",");
    merged.promptrelay_enabled = true;
  }
  return { timeline: merged, appliedPromptCount, preservedVisualSegments: visualSegments.length };
}
function fieldLabel(text, extra = "") {
  const label = document.createElement("div");
  label.className = "iamccs-dte-label";
  label.innerHTML = "<span>" + text + "</span><span>" + extra + "</span>";
  return label;
}
function emotionRoutingValue(data) {
  const value = String(data?.settings?.emotion_routing || "clean_metadata");
  if (value === "index_tts_text_emotion") return "index_tts_text_emotion";
  if (value === "index_tts_character_tags") return "index_tts_character_tags";
  if (value === "chatterbox_v2_tokens") return "chatterbox_v2_tokens";
  if (value === "step_editx_tags") return "step_editx_tags";
  return "clean_metadata";
}
function cineAudioTextModeFromEmotionRouting(data) {
  const value = emotionRoutingValue(data);
  if (value === "step_editx_tags") return "tts_audio_suite_tags";
  if (value === "index_tts_text_emotion") return "index_tts_text_emotion";
  if (value === "index_tts_character_tags") return "index_tts_character_tags";
  if (value === "chatterbox_v2_tokens") return "chatterbox_v2_tokens";
  return "plain_dialogue";
}
function emotionRoutingLabel(value) {
  const key = String(value || "");
  if (key === "index_tts_text_emotion") return "IndexTTS text emotion";
  if (key === "index_tts_character_tags") return "IndexTTS character tags";
  if (key === "chatterbox_v2_tokens") return "ChatterBox v2 tokens";
  if (key === "step_editx_tags") return "Step EditX tags";
  return "Clean metadata";
}
function installDialogueLowZoomOverlay(node) {
  if (node) node._iamccsDialogueLowZoomOverlay = true;
  return;
  if (!node || node._iamccsDialogueLowZoomOverlay) return;
  const previous = node.onDrawForeground;
  node.onDrawForeground = function(ctx) {
    if (typeof previous === "function") previous.apply(this, arguments);
    const scale = Math.max(0.12, Number(app?.canvas?.ds?.scale || 1));
    if (!ctx || scale >= 0.62) return;
    let data = {};
    try { data = parseData(this); } catch {}
    const lines = Array.isArray(data.lines) ? data.lines : [];
    const speakers = Array.from(new Set(lines.map((line) => String(line?.speaker || "A").toUpperCase()))).filter(Boolean);
    const mode = modeFromData(data);
    const routing = emotionRoutingLabel(data.settings?.emotion_routing || "clean_metadata");
    const drawLines = [
      "Dialogue Tag Editor mini view",
      `${lines.length} lines / ${speakers.length || 1} speaker${speakers.length === 1 ? "" : "s"}`,
      mode === "tts_master_unico" ? "single master mode" : "A/B speaker stems mode",
      `emotion routing: ${routing}`,
    ];
    const nodeW = Math.max(340, Number(this.size?.[0] || 420));
    const nodeH = Math.max(180, Number(this.size?.[1] || 240));
    const boost = Math.max(1.2, Math.min(3.4, 0.72 / scale));
    const pad = 12 * boost;
    const lineH = 18 * boost;
    const titleFont = Math.round(13 * boost);
    const bodyFont = Math.round(11 * boost);
    const w = Math.max(240, Math.min(nodeW - pad * 2, 700 * boost));
    const h = 34 * boost + drawLines.length * lineH;
    const x = pad;
    const y = Math.min(Math.max(64, 52 * boost), Math.max(46, nodeH - h - pad));
    ctx.save();
    ctx.globalAlpha = 0.96;
    ctx.fillStyle = "rgba(7,17,18,.93)";
    ctx.strokeStyle = "rgba(143,208,204,.72)";
    ctx.lineWidth = Math.max(1.5, 1.2 * boost);
    if (typeof ctx.roundRect === "function") {
      ctx.beginPath();
      ctx.roundRect(x, y, w, h, 8 * boost);
      ctx.fill();
      ctx.stroke();
    } else {
      ctx.fillRect(x, y, w, h);
      ctx.strokeRect(x, y, w, h);
    }
    ctx.fillStyle = "rgba(239,204,139,.95)";
    ctx.fillRect(x, y, Math.max(4, 3 * boost), h);
    ctx.fillStyle = "#F4D49E";
    ctx.font = `900 ${titleFont}px sans-serif`;
    ctx.fillText(drawLines[0], x + 12 * boost, y + 21 * boost);
    ctx.fillStyle = "#BFD7D5";
    ctx.font = `800 ${bodyFont}px sans-serif`;
    for (let i = 1; i < drawLines.length; i += 1) ctx.fillText(drawLines[i], x + 12 * boost, y + 21 * boost + i * lineH);
    ctx.restore();
  };
  node._iamccsDialogueLowZoomOverlay = true;
}
function install(node, reason = "install") {
  if (!isEditor(node) || node._iamccsDialogueTagEditorReady) return;
  ensureStyle();
  node._iamccsDialogueTagEditorReady = true;
  installDialogueLowZoomOverlay(node);
  ["dialogue_data", "frame_rate", "speech_wpm", "min_line_seconds", "default_gap_seconds", "output_mode", "inline_edit_mode"].forEach((name) => hideWidget(widget(node, name)));
  let data = parseData(node);
  setTimeout(() => repairExistingDialogueRigLinks(node, reason), 0);
  let scriptText = linesToText(data);
  let zoom = Number(data.settings?.font_zoom || 1);
  let light = String(data.settings?.text_theme || "light_boxes") === "light_boxes";
  const root = document.createElement("div");
  root.className = "iamccs-dte";
  const save = () => {
    data.settings ||= {};
    data.settings.font_zoom = zoom;
    data.settings.text_theme = light ? "light_boxes" : "dark_boxes";
    writeData(node, data);
  };
  const render = () => {
    root.replaceChildren();
    root.classList.toggle("light", light);
    const head = document.createElement("div");
    head.className = "iamccs-dte-head";
    const title = document.createElement("div");
    title.innerHTML = '<div class="iamccs-dte-title">IAMCCS Dialogue Tag Editor</div><div class="iamccs-dte-sub">global prompt / local prompts / dialogue lanes / cine_linx</div>';
    const actions = document.createElement("div");
    actions.className = "iamccs-dte-actions";
    const addA = button("Add A", "gold");
    const addB = button("Add B", "gold");
    const injectBtn = button("Inject UI", "gold iamccs-inject-btn");
    const zeroStartBtn = button("A+B @ 0", data.settings?.speaker_stems_zero_start ? "active" : "");
    zeroStartBtn.title = "When active, Speaker A and Speaker B stems both begin at frame 0 while preserving timing inside each speaker lane.";
    zeroStartBtn.setAttribute("aria-pressed", String(Boolean(data.settings?.speaker_stems_zero_start)));
    const boldBtn = button("Bold", "");
    const lightBtn = button(light ? "Dark" : "Light", light ? "" : "active");
    const zOut = button("A-", "");
    const zIn = button("A+", "");
    actions.append(addA, addB, zeroStartBtn, boldBtn, injectBtn, lightBtn, zOut, zIn);
    head.append(title, actions);

    const main = document.createElement("div");
    main.className = "iamccs-dte-main";
    const side = document.createElement("div");
    side.className = "iamccs-dte-side";
    const templateSelect = select(DIALOGUE_TEMPLATES, data.settings?.dialogue_template || "dialogue_abab");
    templateSelect.onchange = () => {
      data.settings ||= {};
      data.settings.dialogue_template = templateSelect.value;
      data.lines = templateLines(templateSelect.value);
      data.global_prompt = templateGlobalPrompt(templateSelect.value);
      scriptText = linesToText(data);
      save();
      render();
    };
    side.append(fieldLabel("Template", "writes example"), templateSelect);
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    side.append(fieldLabel("TTS Mode"));
    const ttsMode = select([["double_stem_ab", "Double stem A/B"], ["tts_master_unico", "TTS single master"]], modeFromData(data));
    ttsMode.onchange = () => { data.settings.tts_generation_mode = ttsMode.value; data.settings.output_mode = outputMode(ttsMode.value); save(); render(); };
    side.append(ttsMode, fieldLabel("Tag Mode"));
    const tagMode = select([["metadata_only", "Metadata only"], ["tts_audio_suite_inline_tags", "Inline tags"]], data.settings.inline_edit_mode || "metadata_only");
    tagMode.onchange = () => { data.settings.inline_edit_mode = tagMode.value; save(); };
    side.append(tagMode);
    side.append(fieldLabel("Emotion Routing", "working paths"));
    const emotionRoute = select([
      ["clean_metadata", "Clean metadata"],
      ["index_tts_text_emotion", "IndexTTS text emotion"],
      ["index_tts_character_tags", "IndexTTS character tags"],
      ["chatterbox_v2_tokens", "ChatterBox v2 tokens"],
      ["step_editx_tags", "Step EditX tags"],
    ], emotionRoutingValue(data));
    emotionRoute.title = "Clean metadata strips tags. IndexTTS text emotion sets emotion text on compatible IndexTTS rigs. IndexTTS character tags exports [Speaker:emotion]. ChatterBox v2 tokens are experimental. Step EditX tags are for the Step EditX inline path.";
    emotionRoute.onchange = () => {
      data.settings ||= {};
      data.settings.emotion_routing = emotionRoute.value;
      if (emotionRoute.value.startsWith("index_tts")) data.settings.tts_export_mode = "index";
      if (emotionRoute.value === "chatterbox_v2_tokens") data.settings.tts_export_mode = "chatterbox";
      save();
      render();
    };
    side.append(emotionRoute);
    const speakersTitle = document.createElement("h4");
    speakersTitle.textContent = "Speakers";
    side.append(speakersTitle);
    (data.speakers || []).forEach((speaker, index) => {
      const card = document.createElement("div");
      card.className = "iamccs-dte-speaker";
      const id = input(speaker.id || String.fromCharCode(65 + index));
      const name = input(speaker.name || speaker.id || "");
      const voice = input(speaker.voice || "", "voice alias/path");
      [id, name, voice].forEach((el) => el.onchange = () => {
        speaker.id = id.value || String.fromCharCode(65 + index);
        speaker.name = name.value || speaker.id;
        speaker.voice = voice.value || "";
        save();
        render();
      });
      card.append(fieldLabel("Speaker " + (index + 1)), id, name, voice);
      side.append(card);
    });
    const conversionPanel = document.createElement("div");
    conversionPanel.className = "iamccs-overlap-panel";
    const conversionGrid = document.createElement("div");
    conversionGrid.className = "iamccs-overlap-preset-grid iamccs-tts-convert-grid";
    const conversionStatus = document.createElement("div");
    conversionStatus.className = "iamccs-overlap-hint";
    conversionStatus.textContent = "Select a TTS tag style: it rewrites the tags directly in Dialogue Script.";
    let activeTtsMode = String(data.settings?.tts_export_mode || "qwen");
    const convertButtons = [];
    const ttsFormatStatus = document.createElement("div");
    ttsFormatStatus.className = "iamccs-overlap-hint";
    const refreshTtsFormatStatus = () => {
      ttsFormatStatus.textContent = "Selected TTS profile: " + activeTtsMode + " | Emotion routing: " + emotionRoutingLabel(emotionRoutingValue(data)) + " / " + cineAudioTextModeFromEmotionRouting(data) + ".";
    };
    let applyTtsModeToScript = null;
    const paintConvertButtons = (successMode = "") => {
      convertButtons.forEach(({ button: item, mode }) => {
        item.classList.toggle("iamccs-convert-active", mode === activeTtsMode);
        item.classList.toggle("iamccs-convert-success", mode === successMode);
      });
    };
    [
      ["Qwen", "qwen"],
      ["IndexTTS", "index"],
      ["LongCat", "longcat"],
      ["Chatterbox", "chatterbox"],
    ].forEach(([label, mode]) => {
      const convertBtn = button(label, "");
      convertBtn.title = "Apply " + label + " tag format directly to Dialogue Script";
      convertBtn.onclick = () => {
        activeTtsMode = mode;
        data.settings ||= {};
        data.settings.tts_export_mode = activeTtsMode;
        if (typeof applyTtsModeToScript === "function") applyTtsModeToScript(mode, label);
        else save();
        refreshTtsFormatStatus();
        paintConvertButtons("");
        conversionStatus.textContent = label + " tags applied in Dialogue Script.";
        paintConvertButtons(mode);
        window.setTimeout(() => paintConvertButtons(""), 1300);
      };
      convertButtons.push({ button: convertBtn, mode });
      conversionGrid.append(convertBtn);
    });
    refreshTtsFormatStatus();
    const ttsNodeOptions = discoverTTSNodeOptions();
    const ttsNodeSelect = select(ttsNodeOptions.length ? ttsNodeOptions : [["", "No registered ready TTS nodes found"]], ttsNodeOptions[0]?.[0] || "");
    ttsNodeSelect.classList.add("iamccs-tts-node-select");
    ttsNodeSelect.disabled = !ttsNodeOptions.length;
    ttsNodeSelect.title = "Select a registered TTS model node with a usable model/voice selection. Add Rig wires it through CineAudioInfo like the reference workflow.";
    const addTtsNodes = button("Add Selected Node", "gold");
    addTtsNodes.title = "Create only the selected registered TTS node types at graph bottom, unlinked";
    addTtsNodes.onclick = () => {
      const selectedTypes = [ttsNodeSelect.value].filter(Boolean);
      addAvailableTTSStarterNodes(data, conversionStatus, selectedTypes);
    };
    const addRig = button("Add Rig", "gold");
    addRig.title = "Create CineAudioInfo export, selected TTS generator(s), Unified SRT when needed, and CineAudioInfo inject";
    addRig.onclick = () => {
      const selectedType = ttsNodeSelect.value;
      createCineAudioRig(node, data, conversionStatus, selectedType);
    };
    const nodeActions = document.createElement("div");
    nodeActions.className = "iamccs-node-actions";
    nodeActions.append(addTtsNodes, addRig);
    paintConvertButtons("");
    conversionPanel.append(fieldLabel("TTS Convert", "write tags"), conversionGrid, ttsFormatStatus, fieldLabel("Add Node", "registered"), ttsNodeSelect, nodeActions, conversionStatus);
    side.append(conversionPanel);

    const center = document.createElement("div");
    center.className = "iamccs-dte-center";
    const toolbar = document.createElement("div");
    toolbar.className = "iamccs-dte-toolbar";
    const scriptArea = document.createElement("div");
    toolbarButtons(toolbar, scriptArea);
    const globalSection = document.createElement("div");
    globalSection.className = "iamccs-dte-section";
    const global = textarea(data.global_prompt || "", "Global prompt");
    global.style.fontSize = Math.round(13 * zoom) + "px";
    global.oninput = () => { data.global_prompt = global.value; save(); };
    global.classList.add("iamccs-global");
    globalSection.append(fieldLabel("Global Prompt", "sent to Shotboard"), global);
    const scriptSection = document.createElement("div");
    scriptSection.className = "iamccs-dte-section";
    scriptArea.className = "iamccs-dte-script iamccs-rich-editor";
    scriptArea.contentEditable = "true";
    scriptArea.spellcheck = false;
    scriptArea.style.fontSize = Math.round(14 * zoom) + "px";
    const savedScriptHeight = Math.max(0, Number(node.properties?.iamccs_dte_script_height || 0));
    if (savedScriptHeight) scriptArea.style.height = Math.max(152, Math.min(520, savedScriptHeight)) + "px";
    const persistScriptHeight = () => {
      const height = Math.round(scriptArea.getBoundingClientRect?.().height || 0);
      if (height < 120) return;
      node.properties = node.properties || {};
      node.properties.iamccs_dte_script_height = Math.max(152, Math.min(520, height));
      node.setDirtyCanvas?.(true, true);
      app.graph?.setDirtyCanvas?.(true, true);
    };
    scriptArea.addEventListener("pointerup", persistScriptHeight);
    scriptArea.addEventListener("blur", persistScriptHeight);
    const statusText = () => {
      const overlapCount = (data.lines || []).filter((line) => Number(line.overlap_after || 0) > 0).length;
      return data.lines.length + " lines | " + (modeFromData(data) === "tts_master_unico" ? "TTS single master" : "Double stem A/B") + " | overlap lines " + overlapCount;
    };
    const overlapValue = document.createElement("input");
    overlapValue.type = "number";
    overlapValue.min = "0";
    overlapValue.max = "2";
    overlapValue.step = "0.01";
    overlapValue.value = "0.12";
    const paintScriptEditor = (preserveCaret = true) => {
      const offset = preserveCaret ? caretOffset(scriptArea) : null;
      scriptArea.innerHTML = renderScriptHighlight(scriptText);
      if (preserveCaret) setCaretOffset(scriptArea, offset);
    };
    paintScriptEditor(false);
    const syncScriptFromData = () => {
      scriptText = linesToText(data);
      paintScriptEditor(false);
      status.textContent = statusText();
    };
    applyTtsModeToScript = (mode, label = mode) => {
      data = applyTTSFormat(data, mode);
      activeTtsMode = String(mode || activeTtsMode);
      scriptText = linesToText(data);
      paintScriptEditor(false);
      save();
      status.textContent = statusText();
      renderCards(cardsGrid);
      scriptArea.focus();
      refreshTtsFormatStatus();
      conversionStatus.textContent = String(label || mode) + " tags applied in Dialogue Script.";
    };
    scriptArea.oninput = () => {
      scriptText = editorPlainText(scriptArea);
      data = parseScript(scriptText, data);
      save();
      status.textContent = statusText();
      paintScriptEditor(true);
      renderCards(cardsGrid);
    };
    const applyOverlapAtCursor = (rawValue) => {
      const currentText = editorPlainText(scriptArea);
      const offset = caretOffset(scriptArea);
      const updated = applyOverlapToTaggedLine(currentText, offset, rawValue);
      scriptText = updated.text;
      data = parseScript(scriptText, data);
      save();
      status.textContent = statusText();
      paintScriptEditor(false);
      setCaretOffset(scriptArea, updated.caret);
      renderCards(cardsGrid);
      scriptArea.focus();
    };
    const insertPerformanceAtCursor = (key, rawValue) => {
      const currentText = editorPlainText(scriptArea);
      const offset = caretOffset(scriptArea);
      const updated = insertPerformanceTagAtTaggedCursor(currentText, offset, key, rawValue, emotionRoutingValue(data));
      scriptText = updated.text;
      data = parseScript(scriptText, data);
      save();
      status.textContent = statusText();
      paintScriptEditor(false);
      setCaretOffset(scriptArea, updated.caret);
      renderCards(cardsGrid);
      scriptArea.focus();
    };
    scriptSection.append(fieldLabel("Dialogue Script", "editor colorato"), scriptArea);
    const cardsSection = document.createElement("div");
    cardsSection.className = "iamccs-dte-section fill";
    const cardsGrid = document.createElement("div");
    cardsGrid.className = "iamccs-dte-grid";
    cardsSection.append(fieldLabel("Dialogue + Local Prompts", "editable per line"), cardsGrid);
    const status = document.createElement("div");
    status.className = "iamccs-dte-foot";
    status.textContent = statusText();
    center.append(toolbar, globalSection, scriptSection, cardsSection, status);

    const overlapPanel = document.createElement("div");
    overlapPanel.className = "iamccs-overlap-panel";
    const overlapRow = document.createElement("div");
    overlapRow.className = "iamccs-overlap-row";
    const overlapApply = button("Apply", "gold");
    overlapApply.onclick = () => applyOverlapAtCursor(overlapValue.value);
    overlapRow.append(overlapValue, overlapApply);
    const overlapPresetGrid = document.createElement("div");
    overlapPresetGrid.className = "iamccs-overlap-preset-grid";
    [0.12, 0.18, 0.22].forEach((preset) => {
      const presetBtn = button(preset.toFixed(2), "");
      presetBtn.onclick = () => {
        overlapValue.value = preset.toFixed(2);
        applyOverlapAtCursor(preset);
      };
      overlapPresetGrid.append(presetBtn);
    });
    const overlapHint = document.createElement("div");
    overlapHint.className = "iamccs-overlap-hint";
    overlapHint.textContent = "Agisce sulla riga dove si trova il cursore nel Dialogue Script: inserisce o sostituisce solo overlap:valore.";
    overlapPanel.append(fieldLabel("Overlap", "cursor line"), overlapRow, overlapPresetGrid, overlapHint);
    side.append(overlapPanel);

    const tags = document.createElement("div");
    tags.className = "iamccs-dte-tags";
    // By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    tags.append(
      tagPanel("Emotion", EMOTIONS, (name) => modelAwareTagToken("emotion", name, activeTtsMode), () => scriptArea, (name) => insertPerformanceAtCursor("emotion", name)),
      tagPanel("Style", STYLES, (name) => modelAwareTagToken("style", name, activeTtsMode), () => scriptArea, (name) => insertPerformanceAtCursor("style", name)),
      tagPanel("Inline", PARAS, (name) => name.startsWith("pause") ? " [" + name + "]" : " <" + name + ">", () => scriptArea, (name) => {
        if (name.startsWith("pause")) insertAt(scriptArea, " [" + name + "] ");
        else insertPerformanceAtCursor("para", name);
      })
    );
    main.append(side, center, tags);
    const foot = document.createElement("div");
    foot.className = "iamccs-dte-foot";
    foot.innerHTML = "<span>Inject UI writes visible Shotboard/AudioBoard widgets and temporary dialogue placeholders. Publish real audio replaces those placeholders.</span><span>No &lt;speech1&gt; required.</span>";
    root.append(head, main, foot);
    renderCards(cardsGrid);

    addA.onclick = () => insertAt(scriptArea, (scriptText.trim() ? "\n" : "") + "[A|emotion:calm|style:low|overlap:0.15|ref:1] New question.");
    addB.onclick = () => insertAt(scriptArea, (scriptText.trim() ? "\n" : "") + "[B|emotion:tense|style:whisper|ref:2] New answer.");
    boldBtn.onclick = () => wrapSelection(scriptArea, "**", "**");
    lightBtn.onclick = () => { light = !light; save(); render(); };
    zOut.onclick = () => { zoom = Math.max(0.8, zoom - 0.1); save(); render(); };
    zIn.onclick = () => { zoom = Math.min(1.8, zoom + 0.1); save(); render(); };
    injectBtn.onclick = () => injectVisibleWidgets(node, data, status, injectBtn, foot);
    zeroStartBtn.onclick = () => {
      data.settings ||= {};
      data.settings.speaker_stems_zero_start = !data.settings.speaker_stems_zero_start;
      save();
      render();
    };

    function renderCards(container) {
      container.replaceChildren();
      (data.lines || []).forEach((line, index) => {
        const card = document.createElement("div");
        card.className = "iamccs-dte-card";
        const head = document.createElement("div");
        head.className = "iamccs-dte-card-head";
        head.innerHTML = "<span>Dialogue " + (line.speaker || "A") + (index + 1) + "</span><span class='iamccs-dte-card-meta'>ref " + (line.ref || (line.speaker === "B" ? 2 : 1)) + "</span>";
        const textBox = textarea(line.text || "", "Dialogue text");
        const promptBox = textarea(line.local_prompt || "", "Local prompt");
        textBox.classList.add("iamccs-dialogue");
        promptBox.classList.add("iamccs-local");
        textBox.style.fontSize = Math.round(12 * zoom) + "px";
        promptBox.style.fontSize = Math.round(12 * zoom) + "px";

        textBox.oninput = () => {
          line.text = textBox.value;
          syncScriptFromData();
          save();
        };
        promptBox.oninput = () => {
          line.local_prompt = promptBox.value;
          save();
        };

        card.append(
          head,
          labelledBox("Text", textBox),
          labelledBox("Local Prompt", promptBox)
        );
        container.append(card);
      });
    }
    save();
  };
  try {
    render();
  } catch (err) {
    console.error("[IAMCCS DialogueTagEditor] render failed", err);
    root.innerHTML = '<div style="padding:12px;color:#ffd7d7;background:#2a1111;border:1px solid #844;border-radius:8px;font:12px system-ui">Dialogue Tag Editor UI error: ' + (err?.message || String(err)) + "</div>";
  }
  const dom = node.addDOMWidget("Dialogue Tag Editor", "iamccs_dialogue_tag_editor", root, { serialize: false });
  dom.computeSize = (width) => [Math.max(1280, width), 1040];
  node.size = [1360, 1140];
  node.setDirtyCanvas?.(true, true);
  console.info("[IAMCCS DialogueTagEditor] installed", { nodeId: node.id, reason });
}
function button(text, cls) {
  const btn = document.createElement("button");
  btn.textContent = text;
  if (cls) btn.className = cls;
  return btn;
}
function input(value, placeholder = "") {
  const el = document.createElement("input");
  el.value = value || "";
  el.placeholder = placeholder;
  return el;
}
function select(options, value) {
  const el = document.createElement("select");
  options.forEach(([val, label]) => {
    const opt = document.createElement("option");
    opt.value = val;
    opt.textContent = label;
    opt.selected = val === value;
    el.append(opt);
  });
  return el;
}
function textarea(value, placeholder) {
  const el = document.createElement("textarea");
  el.className = "iamccs-box";
  el.value = value || "";
  el.placeholder = placeholder || "";
  return el;
}
function labelledBox(label, box) {
  const wrap = document.createElement("div");
  wrap.className = "iamccs-field-wrap";
  wrap.style.minHeight = "0";
  wrap.append(fieldLabel(label), box);
  return wrap;
}
function cardField(label, control) {
  const wrap = document.createElement("div");
  wrap.className = "iamccs-dte-card-field";
  const title = document.createElement("label");
  title.textContent = label;
  wrap.append(title, control);
  return wrap;
}
function toolbarButtons(toolbar, target) {
  [["A line", "\n[A|emotion:calm|style:low|overlap:0.15|ref:1] "], ["B line", "\n[B|emotion:tense|style:whisper|ref:2] "], ["Pause", " [pause:0.3]"], ["Breath", " <Breathing>"]].forEach(([label, token]) => {
    const btn = button(label, "");
    btn.onclick = () => insertAt(target, token);
    toolbar.append(btn);
  });
}
// By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
function tagPanel(title, values, makeToken, getFallbackTarget = null, onPick = null) {
  const panel = document.createElement("div");
  const h = document.createElement("h4");
  h.textContent = title;
  const grid = document.createElement("div");
  grid.className = "iamccs-tag-table";
  values.forEach((value) => {
    const btn = button(value, "");
    btn.onclick = () => {
      if (typeof onPick === "function") {
        onPick(value);
        return;
      }
      const active = document.activeElement;
      const focused = active?.tagName === "TEXTAREA" || active?.isContentEditable ? active : null;
      const fallback = typeof getFallbackTarget === "function" ? getFallbackTarget() : null;
      const target = focused || fallback;
      if (!target) return;
      insertAt(target, makeToken(value));
    };
    grid.append(btn);
  });
  panel.append(h, grid);
  return panel;
}
// By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
// By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
function injectVisibleWidgets(node, data, status, injectBtn, foot) {
  const fps = Number(widget(node, "frame_rate")?.value || 24);
  const { audioBoard, timeline } = buildInjectionPayload(data, fps);
  let shotboardCount = 0;
  let audioBoardCount = 0;
  const pendingCount = Array.isArray(timeline?.audioSegments) ? timeline.audioSegments.filter((seg) => seg?.pendingTTS === true || String(seg?.purpose || "") === "dialogue_pending_tts").length : 0;
  injectBtn?.classList.remove("is-done");
  injectBtn?.classList.add("is-busy");
  if (injectBtn) {
    injectBtn.disabled = true;
    injectBtn.textContent = "Injecting...";
  }
  status?.classList.add("iamccs-inject-status");
  foot?.classList.add("iamccs-inject-status");
  status.textContent = "Inject UI in progress...";
  downstream(node, (candidate) => nodeType(candidate) === "IAMCCS_CineShotboardPlannerV3").forEach((shotboard) => {
    const gp = widget(shotboard, "global_prompt");
    const td = widget(shotboard, "timeline_data");
    if (gp) { gp.value = data.global_prompt || ""; try { gp.callback?.(gp.value); } catch {} }
    if (td) {
      const existingTimeline = safeJsonParse(td.value, {});
      const merged = mergeDialoguePromptsIntoShotboard(existingTimeline, data, fps);
      if (typeof shotboard._iamccsCineShotboardV3ApplyExternalTimeline === "function") {
        shotboard._iamccsCineShotboardV3ApplyExternalTimeline(merged.timeline);
      } else {
        td.value = JSON.stringify(merged.timeline, null, 2);
        try { td.callback?.(td.value); } catch {}
      }
    }
    shotboard.setDirtyCanvas?.(true, true);
    shotboardCount++;
  });
  downstream(node, (candidate) => nodeType(candidate) === "IAMCCS_AudioBoardArranger").forEach((board) => {
    const aw = widget(board, "arranger_data");
    if (aw) { aw.value = JSON.stringify(audioBoard, null, 2); try { aw.callback?.(aw.value); } catch {} }
    board.setDirtyCanvas?.(true, true);
    audioBoardCount++;
  });
  const stamp = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  status.textContent = "Injected " + shotboardCount + " Shotboard / " + audioBoardCount + " AudioBoard at " + stamp + ". " + (pendingCount > 0 ? pendingCount + " pending dialogue placeholder lane" + (pendingCount === 1 ? "" : "s") + " sent to AudioBoard until Publish real audio." : "Visual UI synced.");
  status?.classList.remove("iamccs-inject-status");
  status?.classList.add("iamccs-inject-status-success");
  foot?.classList.remove("iamccs-inject-status");
  foot?.classList.add("iamccs-inject-status-success");
  if (injectBtn) {
    injectBtn.classList.remove("is-busy");
    injectBtn.classList.add("is-done");
    injectBtn.disabled = false;
    injectBtn.textContent = "Injected";
    if (injectBtn._iamccsResetTimer) window.clearTimeout(injectBtn._iamccsResetTimer);
    injectBtn._iamccsResetTimer = window.setTimeout(() => {
      injectBtn.classList.remove("is-done");
      injectBtn.textContent = "Inject UI";
      status?.classList.remove("iamccs-inject-status-success");
      foot?.classList.remove("iamccs-inject-status-success");
    }, 1800);
  }
  app.graph?.setDirtyCanvas?.(true, true);
}
app.registerExtension({
  name: "IAMCCS.DialogueTagEditor",
  setup() {
    [300, 1200, 2500].forEach((delay) => setTimeout(() => {
      const nodes = Array.isArray(app?.graph?._nodes) ? app.graph._nodes : [];
      nodes.forEach((node) => {
        install(node, "scan+" + delay);
        if (isEditor(node)) repairExistingDialogueRigLinks(node, "scan+" + delay);
      });
    }, delay));
  },
  nodeCreated(node) { [0, 180, 600].forEach((delay) => setTimeout(() => install(node, "nodeCreated+" + delay), delay)); },
  loadedGraphNode(node) { [0, 180, 600].forEach((delay) => setTimeout(() => install(node, "loadedGraphNode+" + delay), delay)); },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== TYPE) return;
    const original = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      original?.apply(this, arguments);
      setTimeout(() => install(this, "prototype.onNodeCreated"), 0);
    };
  },
});
