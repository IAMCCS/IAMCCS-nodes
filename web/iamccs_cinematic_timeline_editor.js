import { app } from "../../scripts/app.js";

const EDITOR_VERSION = "2026-04-30-1";
const CINE_FILM_LAB = {
    header: "#2E2A24",
    nodeBg: "#171512",
    relay: "#5FA8C7",
};

const CUT_MODES = ["hard_cut", "continuity_cut", "soft_cut", "match_cut"];
const AUDIO_VALUES = ["0", "1", "2", "3", "4", "5", "6", "7", "8"];
const SOURCE_VALUES = ["0", "1", "2", "3", "4", "5", "6", "7", "8"];
const REF_VALUES = ["1", "2", "3", "4", "5", "6", "7", "8", "1>2", "2>1", "1+2", "2+3", "3+1", "1>4", "2>4", "3>4"];
const AUDIO_OPTIONS = [
    { value: "0", label: "No audio (0)" },
    { value: "1", label: "Audio A / speaker A (1)" },
    { value: "2", label: "Audio B / speaker B (2)" },
    { value: "3", label: "Audio C / return A (3)" },
    { value: "4", label: "Ambience / FX (4)" },
    { value: "5", label: "Audio 5" },
    { value: "6", label: "Audio 6" },
    { value: "7", label: "Audio 7" },
    { value: "8", label: "Audio 8" },
];
const SOURCE_OPTIONS = [
    { value: "0", label: "No source (0)" },
    { value: "1", label: "Source A / camera A (1)" },
    { value: "2", label: "Source B / camera B (2)" },
    { value: "3", label: "Source C (3)" },
    { value: "4", label: "Source D / wide (4)" },
    { value: "5", label: "Source 5" },
    { value: "6", label: "Source 6" },
    { value: "7", label: "Source 7" },
    { value: "8", label: "Source 8" },
];
const REF_OPTIONS = [
    { value: "1", label: "Reference A / character A (1)" },
    { value: "2", label: "Reference B / character B (2)" },
    { value: "3", label: "Environment / insert (3)" },
    { value: "4", label: "Wide / final reveal (4)" },
    { value: "5", label: "Reference 5" },
    { value: "6", label: "Reference 6" },
    { value: "7", label: "Reference 7" },
    { value: "8", label: "Reference 8" },
    { value: "1>2", label: "A to B continuity (1>2)" },
    { value: "2>1", label: "B to A continuity (2>1)" },
    { value: "1+2", label: "A + B same shot (1+2)" },
    { value: "2+3", label: "B + environment (2+3)" },
    { value: "3+1", label: "Environment + A (3+1)" },
    { value: "1>4", label: "A to wide/final (1>4)" },
    { value: "2>4", label: "B to wide/final (2>4)" },
    { value: "3>4", label: "Insert/environment to wide (3>4)" },
];
const SOURCE_RANGE_OPTIONS = ["all", "0+96", "24+84", "48+120", "tail120", "0-96", "24-108"];
const V2V_MODES = [
    "v2v_source_plus_reference",
    "v2v_source_context",
    "i2v_from_reference",
    "two_segments_if_long",
    "loop_if_long",
];

const DEFAULT_ASSET_LABELS = {
    ref1: "Reference A / character A",
    ref2: "Reference B / character B",
    ref3: "Environment / insert",
    ref4: "Wide / final reveal",
    ref5: "Reference 5",
    ref6: "Reference 6",
    ref7: "Reference 7",
    ref8: "Reference 8",
    audio1: "Audio A / speaker A",
    audio2: "Audio B / speaker B",
    audio3: "Audio C / return A",
    audio4: "Ambience / FX",
    source1: "Source A / camera A",
    source2: "Source B / camera B",
    source3: "Source C",
    source4: "Source D / wide",
};

const FRAMING_OPTIONS = [
    "custom",
    "wide establishing shot",
    "medium shot",
    "medium close-up",
    "close-up",
    "extreme close-up",
    "over-the-shoulder",
    "reverse angle",
    "insert detail",
    "two-shot",
    "tracking shot",
    "V2V close-up",
    "V2V reverse angle",
    "V2V wide shot",
];

const CONFIGS = {
    IAMCCS_CineMultiGenDirector: {
        kind: "multigen",
        widgetName: "shot_lines",
        title: "IAMCCS Cine Multi-Generation Director",
        lineFields: ["seconds", "cut", "ref", "audio", "label", "prompt", "dialogue", "voice"],
    },
    IAMCCS_CineV2VTimelineDirector: {
        kind: "v2v",
        widgetName: "timeline_lines",
        title: "IAMCCS Cine V2V Timeline Director",
        lineFields: ["seconds", "cut", "source", "sourceRange", "ref", "audio", "label", "prompt", "dialogue", "voice", "v2vMode"],
    },
    IAMCCS_LTX2_CinematicMultiGenPlanner: {
        kind: "multigen",
        widgetName: "shot_lines",
        title: "IAMCCS Cine Multi-Generation Director",
        lineFields: ["seconds", "cut", "ref", "audio", "label", "prompt", "dialogue", "voice"],
    },
    IAMCCS_LTX2_CinematicV2VTimelinePlanner: {
        kind: "v2v",
        widgetName: "timeline_lines",
        title: "IAMCCS Cine V2V Timeline Director",
        lineFields: ["seconds", "cut", "source", "sourceRange", "ref", "audio", "label", "prompt", "dialogue", "voice", "v2vMode"],
    },
};

function nodeClassName(node) {
    return String(node?.comfyClass || node?.type || node?.constructor?.comfyClass || "");
}

function getConfigForNodeData(nodeData) {
    const name = String(nodeData?.name || "");
    return CONFIGS[name] || null;
}

function getConfigForNode(node) {
    return CONFIGS[nodeClassName(node)] || null;
}

function getWidget(node, name) {
    return node?.widgets?.find((widget) => widget?.name === name || widget?.label === name) || null;
}

function setWidgetValue(node, widgetName, value) {
    const widget = getWidget(node, widgetName);
    if (!widget) return false;
    widget.value = value;
    try {
        widget.callback?.(value, app.canvas, node);
    } catch {
        // ignore UI callback issues
    }
    try {
        node.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas?.(true, true);
    } catch {
        // ignore
    }
    return true;
}

function splitPipeLine(line, expectedCount) {
    const parts = String(line || "").split("|").map((part) => part.trim());
    while (parts.length < expectedCount) parts.push("");
    if (parts.length > expectedCount) {
        const head = parts.slice(0, expectedCount - 1);
        const tail = parts.slice(expectedCount - 1).join(" | ");
        return [...head, tail];
    }
    return parts;
}

function cleanLinePart(value) {
    return String(value ?? "").replace(/\s+/g, " ").trim();
}

function joinLine(values) {
    return values.map(cleanLinePart).join(" | ");
}

function parseRows(text, config) {
    const rows = [];
    const fields = config.lineFields;
    for (const rawLine of String(text || "").split(/\r?\n/)) {
        const line = rawLine.trim();
        if (!line || line.startsWith("#")) continue;
        const parts = splitPipeLine(line, fields.length);
        const row = {};
        fields.forEach((field, index) => {
            row[field] = parts[index] || "";
        });
        row.framing = "custom";
        rows.push(normalizeRow(row, config.kind));
    }
    if (!rows.length) {
        return config.kind === "v2v" ? presetRows("v2v_two_camera") : presetRows("multigen_dialogue_4");
    }
    return rows;
}

function normalizeRow(row, kind) {
    const out = { ...row };
    out.seconds = out.seconds || "4.0";
    out.cut = CUT_MODES.includes(out.cut) ? out.cut : "hard_cut";
    out.ref = out.ref || "1";
    out.audio = AUDIO_VALUES.includes(String(out.audio)) ? String(out.audio) : "0";
    out.label = out.label || "shot";
    out.prompt = out.prompt || (kind === "v2v" ? "V2V cinematic shot, keep source motion" : "cinematic shot, natural acting");
    out.dialogue = out.dialogue || "";
    out.voice = out.voice || "";
    out.framing = out.framing || "custom";
    if (kind === "v2v") {
        out.source = SOURCE_VALUES.includes(String(out.source)) ? String(out.source) : "1";
        out.sourceRange = out.sourceRange || "all";
        out.v2vMode = V2V_MODES.includes(out.v2vMode) ? out.v2vMode : "v2v_source_plus_reference";
    }
    return out;
}

function rowToLine(row, config) {
    const prompt = buildPromptWithFraming(row);
    if (config.kind === "v2v") {
        return joinLine([
            row.seconds,
            row.cut,
            row.source,
            row.sourceRange,
            row.ref,
            row.audio,
            row.label,
            prompt,
            row.dialogue,
            row.voice,
            row.v2vMode,
        ]);
    }
    return joinLine([
        row.seconds,
        row.cut,
        row.ref,
        row.audio,
        row.label,
        prompt,
        row.dialogue,
        row.voice,
    ]);
}

function rowsToText(rows, config) {
    return rows.map((row) => rowToLine(row, config)).join("\n");
}

function valueFromOption(option) {
    return typeof option === "object" ? String(option.value) : String(option);
}

function labelFromOption(option) {
    return typeof option === "object" ? String(option.label ?? option.value) : String(option);
}

function labelForValue(options, value) {
    const raw = String(value ?? "");
    const found = options.find((option) => valueFromOption(option) === raw);
    return found ? labelFromOption(found) : raw;
}

function getAssetLabels(node) {
    return { ...DEFAULT_ASSET_LABELS, ...(node?.properties?.iamccs_cine_asset_labels || {}) };
}

function setAssetLabel(node, key, value) {
    node.properties = node.properties || {};
    node.properties.iamccs_cine_asset_labels = {
        ...(node.properties.iamccs_cine_asset_labels || {}),
        [key]: cleanLinePart(value),
    };
}

function stripIndexSuffix(value) {
    return String(value || "").replace(/\s*\([^)]*\)\s*$/, "");
}

function namedIndexedValue(prefix, value, labels, options) {
    const raw = String(value ?? "");
    if (raw === "0") return prefix === "audio" ? "No audio" : "No source";
    const key = `${prefix}${raw}`;
    return labels?.[key] || stripIndexSuffix(labelForValue(options, raw));
}

function humanRef(value, labels = DEFAULT_ASSET_LABELS) {
    const raw = String(value ?? "");
    if (raw.includes(">")) {
        return raw.split(">").map((part) => namedIndexedValue("ref", part.trim(), labels, REF_OPTIONS)).join(" -> ");
    }
    if (raw.includes("+")) {
        return raw.split("+").map((part) => namedIndexedValue("ref", part.trim(), labels, REF_OPTIONS)).join(" + ");
    }
    return namedIndexedValue("ref", raw, labels, REF_OPTIONS);
}

function humanAudio(value, labels = DEFAULT_ASSET_LABELS) {
    return namedIndexedValue("audio", value, labels, AUDIO_OPTIONS);
}

function humanSource(value, labels = DEFAULT_ASSET_LABELS) {
    return namedIndexedValue("source", value, labels, SOURCE_OPTIONS);
}

function rowSummary(row, index, config, labels = DEFAULT_ASSET_LABELS) {
    const seconds = cleanLinePart(row.seconds || "?");
    const label = cleanLinePart(row.label || `shot_${index + 1}`);
    const framing = cleanLinePart(row.framing || "custom");
    const prompt = cleanLinePart(row.prompt || "");
    const shortPrompt = prompt.length > 72 ? `${prompt.slice(0, 72)}...` : prompt;
    if (config.kind === "v2v") {
        return `Segment ${index + 1}: ${label} | ${seconds}s | ${row.cut} | ${humanSource(row.source, labels)} | range ${row.sourceRange || "all"} | ${humanRef(row.ref, labels)} | ${humanAudio(row.audio, labels)} | ${framing} | ${shortPrompt}`;
    }
    return `Shot ${index + 1}: ${label} | ${seconds}s | ${row.cut} | ${humanRef(row.ref, labels)} | ${humanAudio(row.audio, labels)} | ${framing} | ${shortPrompt}`;
}

function computeWarnings(rows, config) {
    const warnings = [];
    if (!rows.length) warnings.push("timeline vuota");
    rows.forEach((row, index) => {
        const label = row.label || `row ${index + 1}`;
        const seconds = Number(row.seconds);
        if (!Number.isFinite(seconds) || seconds <= 0) warnings.push(`${label}: durata non valida`);
        if (String(row.audio || "0") !== "0" && !cleanLinePart(row.dialogue)) {
            warnings.push(`${label}: audio selezionato ma dialogue vuoto`);
        }
        if (cleanLinePart(row.dialogue) && String(row.audio || "0") === "0") {
            warnings.push(`${label}: dialogue presente ma audio = 0`);
        }
        if (String(row.cut) === "continuity_cut" && !String(row.ref || "").includes(">")) {
            warnings.push(`${label}: continuity_cut senza ref tipo 1>2`);
        }
        if (String(row.framing || "").toLowerCase().includes("two-shot") && !String(row.ref || "").includes("+")) {
            warnings.push(`${label}: two-shot senza ref combinata tipo 1+2`);
        }
        if (config.kind === "v2v") {
            if (String(row.source || "0") === "0") warnings.push(`${label}: source video = 0`);
            if (!cleanLinePart(row.sourceRange)) warnings.push(`${label}: source range vuoto`);
        }
    });
    return warnings;
}

function buildPromptWithFraming(row) {
    const framing = cleanLinePart(row.framing);
    const prompt = cleanLinePart(row.prompt);
    if (!framing || framing === "custom") return prompt;
    if (prompt.toLowerCase().startsWith(framing.toLowerCase())) return prompt;
    return `${framing}. ${prompt}`;
}

function presetRows(key) {
    const presets = {
        multigen_dialogue_4: [
            { seconds: "4.0", cut: "hard_cut", ref: "1", audio: "1", label: "campo_mara", framing: "medium close-up", prompt: "Mara near the kitchen window, she speaks softly but firmly", dialogue: "You left the city without saying goodbye.", voice: "controlled hurt voice" },
            { seconds: "3.5", cut: "hard_cut", ref: "2", audio: "2", label: "controcampo_elio", framing: "reverse angle", prompt: "Elio at the doorway, he answers without stepping in", dialogue: "I thought silence would protect you.", voice: "low ashamed voice" },
            { seconds: "2.5", cut: "hard_cut", ref: "3", audio: "0", label: "insert_key", framing: "insert detail", prompt: "old brass key on the table, rain reflected on metal", dialogue: "", voice: "" },
            { seconds: "4.5", cut: "continuity_cut", ref: "1>2", audio: "1", label: "ritorno_mara", framing: "close-up", prompt: "Mara returns to silence, slow push-in, visible hesitation", dialogue: "It protected only you.", voice: "almost whispered voice" },
        ],
        multigen_dialogue_2: [
            { seconds: "4.0", cut: "hard_cut", ref: "1", audio: "1", label: "campo_A", framing: "medium close-up", prompt: "character A speaks with restrained emotion, coherent eyeline", dialogue: "I knew you would come back.", voice: "tired intimate low voice" },
            { seconds: "3.5", cut: "hard_cut", ref: "2", audio: "2", label: "controcampo_B", framing: "reverse angle", prompt: "character B listens, then answers, natural reaction timing", dialogue: "I never really left.", voice: "calm low voice" },
        ],
        multigen_monologue: [
            { seconds: "6.0", cut: "hard_cut", ref: "1", audio: "1", label: "monologo_closeup", framing: "close-up", prompt: "single character speaking quietly, subtle facial movement, stable identity", dialogue: "I need you to hear the whole story.", voice: "low confessional voice" },
        ],
        v2v_two_camera: [
            { seconds: "4.0", cut: "hard_cut", source: "1", sourceRange: "0+96", ref: "1", audio: "1", label: "camera_A_mara", framing: "V2V close-up", prompt: "Mara speaking, keep source head movement and natural eye line", dialogue: "You left the city without saying goodbye.", voice: "controlled hurt voice", v2vMode: "v2v_source_plus_reference" },
            { seconds: "3.5", cut: "hard_cut", source: "2", sourceRange: "24+84", ref: "2", audio: "2", label: "camera_B_elio", framing: "V2V reverse angle", prompt: "Elio answering, keep source performance and natural eye line", dialogue: "I thought silence would protect you.", voice: "low ashamed voice", v2vMode: "v2v_source_plus_reference" },
            { seconds: "5.0", cut: "continuity_cut", source: "1", sourceRange: "tail120", ref: "1>2", audio: "0", label: "return_camera_A", framing: "V2V close-up", prompt: "continue from camera A, slow push-in, preserve gesture continuity", dialogue: "", voice: "", v2vMode: "two_segments_if_long" },
        ],
        v2v_single: [
            { seconds: "4.0", cut: "hard_cut", source: "1", sourceRange: "all", ref: "1", audio: "1", label: "v2v_single_take", framing: "V2V close-up", prompt: "preserve original motion, improve cinematic light and texture", dialogue: "", voice: "", v2vMode: "v2v_source_plus_reference" },
        ],
        v2v_long: [
            { seconds: "8.0", cut: "continuity_cut", source: "1", sourceRange: "0+192", ref: "1", audio: "1", label: "long_part_1", framing: "V2V tracking shot", prompt: "continuous movement, preserve direction and speed", dialogue: "", voice: "", v2vMode: "two_segments_if_long" },
            { seconds: "8.0", cut: "continuity_cut", source: "1", sourceRange: "tail192", ref: "1", audio: "1", label: "long_part_2", framing: "V2V tracking shot", prompt: "continue same movement, preserve gesture continuity", dialogue: "", voice: "", v2vMode: "two_segments_if_long" },
        ],
        multigen_zoom_drone: [
            { seconds: "3.0", cut: "continuity_cut", ref: "1>2", audio: "0", label: "face_to_eye", framing: "close-up", prompt: "slow forward zoom from the man's face toward his eye, stable identity, cinematic macro detail", dialogue: "", voice: "" },
            { seconds: "3.0", cut: "continuity_cut", ref: "2>3", audio: "0", label: "eye_to_mountain", framing: "extreme close-up", prompt: "camera travels through the pupil, the dark pupil becomes a mountain landscape, seamless surreal transition", dialogue: "", voice: "" },
            { seconds: "4.0", cut: "continuity_cut", ref: "3>4", audio: "0", label: "mountain_to_sea", framing: "tracking shot", prompt: "drone flight over the mountain, descending toward a vast sea, continuous forward motion", dialogue: "", voice: "" },
            { seconds: "3.0", cut: "continuity_cut", ref: "4", audio: "0", label: "sea_wide", framing: "wide establishing shot", prompt: "wide cinematic drone shot over a calm sea, slow forward movement", dialogue: "", voice: "" },
        ],
        multigen_astronaut_action: [
            { seconds: "4.0", cut: "hard_cut", ref: "1", audio: "1", label: "astronaut_speaks", framing: "medium close-up", prompt: "astronaut holding a laser pistol, speaking directly to camera, tense sci-fi mood", dialogue: "We are not alone here.", voice: "calm but alarmed voice" },
            { seconds: "3.0", cut: "hard_cut", ref: "2", audio: "0", label: "laser_shot_zoomout", framing: "medium shot", prompt: "astronaut fires the laser pistol, slight zoom out, bright laser flash, dust and sparks", dialogue: "", voice: "" },
            { seconds: "4.0", cut: "hard_cut", ref: "3>4", audio: "0", label: "back_view_alien", framing: "over-the-shoulder", prompt: "rear view of the astronaut firing at an alien facing him, laser beam crossing the frame", dialogue: "", voice: "" },
        ],
        multigen_bar_dialogue: [
            { seconds: "4.0", cut: "hard_cut", ref: "1", audio: "1", label: "bar_campo_A", framing: "close-up", prompt: "person A sitting at a bar table, speaking quietly, city lights reflected in the window", dialogue: "I waited here every night.", voice: "tired intimate voice" },
            { seconds: "3.5", cut: "hard_cut", ref: "2", audio: "2", label: "bar_controcampo_B", framing: "reverse angle", prompt: "person B sitting across the table, answering with a small pause, warm bar light, coherent eyeline", dialogue: "I know. I was afraid to come in.", voice: "low regretful voice" },
            { seconds: "3.0", cut: "hard_cut", ref: "1", audio: "3", label: "bar_ritorno_A", framing: "close-up", prompt: "return to person A, slight expression change, glass and neon reflections in the foreground", dialogue: "Then why are you here now?", voice: "quiet but direct voice" },
            { seconds: "4.0", cut: "hard_cut", ref: "3", audio: "0", label: "bar_largo_citta", framing: "wide establishing shot", prompt: "wide shot of both people seated at a small bar table, city environment visible outside", dialogue: "", voice: "" },
        ],
        multigen_product_reveal: [
            { seconds: "2.5", cut: "hard_cut", ref: "1", audio: "0", label: "macro_object", framing: "insert detail", prompt: "premium macro detail of the mysterious object, glossy reflections, cinematic commercial lighting", dialogue: "", voice: "" },
            { seconds: "3.0", cut: "hard_cut", ref: "2", audio: "0", label: "hand_reveal", framing: "medium close-up", prompt: "a hand picks up the object, controlled camera movement, elegant product reveal", dialogue: "", voice: "" },
            { seconds: "3.0", cut: "hard_cut", ref: "3", audio: "0", label: "table_insert", framing: "insert detail", prompt: "object on the table, light catching its edges, laboratory atmosphere", dialogue: "", voice: "" },
            { seconds: "4.0", cut: "hard_cut", ref: "4", audio: "0", label: "wide_lab", framing: "wide establishing shot", prompt: "wide cinematic view of the laboratory or room where the object is revealed", dialogue: "", voice: "" },
        ],
        multigen_music_clip: [
            { seconds: "3.0", cut: "hard_cut", ref: "1", audio: "1", label: "singer_closeup", framing: "close-up", prompt: "singer performing into microphone, emotional music video lighting", dialogue: "", voice: "" },
            { seconds: "2.5", cut: "hard_cut", ref: "2", audio: "1", label: "hands_microphone", framing: "insert detail", prompt: "hands gripping the microphone, reflections and rhythmic movement", dialogue: "", voice: "" },
            { seconds: "3.0", cut: "hard_cut", ref: "3", audio: "1", label: "crowd_lights", framing: "wide establishing shot", prompt: "crowd silhouettes and moving stage lights in a cinematic music clip", dialogue: "", voice: "" },
            { seconds: "4.0", cut: "hard_cut", ref: "4", audio: "1", label: "night_city", framing: "wide establishing shot", prompt: "night city lights matching the song mood, atmospheric ending", dialogue: "", voice: "" },
        ],
        v2v_bar_dialogue: [
            { seconds: "4.0", cut: "hard_cut", source: "1", sourceRange: "all", ref: "1", audio: "0", label: "v2v_bar_A", framing: "V2V close-up", prompt: "preserve source performance, person A at bar speaking, improve cinematic night lighting", dialogue: "", voice: "", v2vMode: "v2v_source_plus_reference" },
            { seconds: "3.5", cut: "hard_cut", source: "2", sourceRange: "all", ref: "2", audio: "0", label: "v2v_bar_B", framing: "V2V reverse angle", prompt: "reverse angle on person B answering, preserve source audio performance", dialogue: "", voice: "", v2vMode: "v2v_source_plus_reference" },
            { seconds: "3.0", cut: "hard_cut", source: "3", sourceRange: "all", ref: "1", audio: "0", label: "v2v_bar_A_return", framing: "V2V close-up", prompt: "return close-up on person A, preserve source timing and expression", dialogue: "", voice: "", v2vMode: "v2v_source_plus_reference" },
            { seconds: "4.0", cut: "hard_cut", source: "4", sourceRange: "all", ref: "3", audio: "0", label: "v2v_bar_wide_city", framing: "V2V wide shot", prompt: "wide shot of both people in the bar, city visible outside, preserve source motion", dialogue: "", voice: "", v2vMode: "v2v_source_plus_reference" },
        ],
        v2v_flashback: [
            { seconds: "4.0", cut: "hard_cut", source: "1", sourceRange: "all", ref: "1", audio: "0", label: "corridor_memory", framing: "V2V tracking shot", prompt: "dreamlike memory corridor walking, preserve source motion, soft halation", dialogue: "", voice: "", v2vMode: "v2v_source_plus_reference" },
            { seconds: "3.5", cut: "hard_cut", source: "2", sourceRange: "all", ref: "1", audio: "0", label: "face_memory", framing: "V2V close-up", prompt: "close-up face, emotional memory, soft light leak", dialogue: "", voice: "", v2vMode: "v2v_source_plus_reference" },
            { seconds: "3.0", cut: "hard_cut", source: "3", sourceRange: "all", ref: "2", audio: "0", label: "door_hand", framing: "V2V insert detail", prompt: "hand touching door, symbolic detail, dreamlike color", dialogue: "", voice: "", v2vMode: "v2v_source_plus_reference" },
            { seconds: "4.0", cut: "hard_cut", source: "4", sourceRange: "all", ref: "3", audio: "0", label: "outside_light", framing: "V2V wide shot", prompt: "bright exterior light, memory resolves into light", dialogue: "", voice: "", v2vMode: "v2v_source_plus_reference" },
        ],
    };
    return (presets[key] || presets.multigen_dialogue_4).map((row) => normalizeRow(row, key.startsWith("v2v") ? "v2v" : "multigen"));
}

function ensureStyles() {
    if (document.getElementById("iamccs-cinematic-timeline-editor-style")) return;
    const style = document.createElement("style");
    style.id = "iamccs-cinematic-timeline-editor-style";
    style.textContent = `
        .iamccs-cine-overlay {
            position: fixed;
            inset: 0;
            z-index: 100000;
            background: rgba(12, 9, 6, 0.76);
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: Inter, Arial, sans-serif;
            color: #F1E8D2;
        }
        .iamccs-cine-modal {
            width: min(96vw, 1500px);
            height: min(92vh, 920px);
            background: #171512;
            border: 1px solid #4A4032;
            box-shadow: 0 24px 80px rgba(0,0,0,0.55);
            display: grid;
            grid-template-rows: auto auto 1fr auto;
            overflow: hidden;
        }
        .iamccs-cine-modal.director {
            grid-template-rows: auto auto auto 1fr auto;
        }
        .iamccs-cine-header,
        .iamccs-cine-toolbar,
        .iamccs-cine-assets,
        .iamccs-cine-footer {
            padding: 12px 16px;
            border-bottom: 1px solid #4A4032;
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
        }
        .iamccs-cine-footer {
            border-top: 1px solid #4A4032;
            border-bottom: 0;
            justify-content: space-between;
        }
        .iamccs-cine-title {
            font-size: 18px;
            font-weight: 700;
            letter-spacing: 0;
            margin-right: auto;
        }
        .iamccs-cine-help {
            color: #B8A98E;
            font-size: 12px;
        }
        .iamccs-cine-steps {
            display: grid;
            grid-template-columns: repeat(4, minmax(120px, 1fr));
            gap: 8px;
            width: 100%;
        }
        .iamccs-cine-step {
            background: #201D18;
            border: 1px solid #4A4032;
            border-radius: 6px;
            padding: 8px;
            min-height: 52px;
        }
        .iamccs-cine-step strong {
            display: block;
            color: #F1E8D2;
            font-size: 12px;
            margin-bottom: 4px;
        }
        .iamccs-cine-step span {
            color: #B8A98E;
            font-size: 11px;
            line-height: 1.25;
        }
        .iamccs-cine-assets {
            background: #201D18;
            align-items: flex-start;
        }
        .iamccs-cine-assets-title {
            min-width: 160px;
            color: #F1E8D2;
            font-size: 12px;
            font-weight: 700;
            padding-top: 7px;
        }
        .iamccs-cine-asset-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(170px, 1fr));
            gap: 8px;
            flex: 1;
        }
        .iamccs-cine-asset-field {
            display: grid;
            gap: 4px;
        }
        .iamccs-cine-asset-field label {
            color: #B8A98E;
            font-size: 11px;
        }
        .iamccs-cine-btn,
        .iamccs-cine-select,
        .iamccs-cine-input,
        .iamccs-cine-textarea {
            background: #11100D;
            color: #F1E8D2;
            border: 1px solid #4A4032;
            border-radius: 6px;
            font-size: 12px;
            font-family: Inter, Arial, sans-serif;
        }
        .iamccs-cine-btn {
            padding: 8px 10px;
            cursor: pointer;
            white-space: nowrap;
        }
        .iamccs-cine-btn:hover { background: #4A4032; }
        .iamccs-cine-btn.primary { background: #5A4424; border-color: #D6A85A; }
        .iamccs-cine-btn.danger { background: #6E2D31; border-color: #B84A4A; }
        .iamccs-cine-select,
        .iamccs-cine-input {
            height: 30px;
            padding: 4px 6px;
        }
        .iamccs-cine-textarea {
            min-height: 54px;
            padding: 6px;
            resize: vertical;
            line-height: 1.25;
        }
        .iamccs-cine-body {
            overflow: auto;
            padding: 12px 16px;
        }
        .iamccs-cine-table {
            border-collapse: separate;
            border-spacing: 0;
            min-width: 1250px;
            width: 100%;
        }
        .iamccs-cine-table th {
            position: sticky;
            top: 0;
            background: #201D18;
            z-index: 1;
            border-bottom: 1px solid #4A4032;
            color: #B8A98E;
            font-size: 11px;
            text-align: left;
            padding: 7px 6px;
        }
        .iamccs-cine-table td {
            border-bottom: 1px solid #3A332A;
            padding: 6px;
            vertical-align: top;
        }
        .iamccs-cine-row-index {
            color: #B8A98E;
            font-weight: 700;
            text-align: center;
            padding-top: 12px;
        }
        .iamccs-cine-row-actions {
            display: grid;
            grid-template-columns: repeat(2, 30px);
            gap: 4px;
        }
        .iamccs-cine-row-actions button {
            height: 28px;
            padding: 0;
        }
        .iamccs-cine-preview {
            width: min(720px, 44vw);
            min-height: 86px;
            max-height: 130px;
            color: #F1E8D2;
            background: #11100D;
        }
        .iamccs-cine-status {
            color: #B8A98E;
            font-size: 12px;
            max-width: 360px;
            white-space: pre-line;
        }
        .iamccs-cine-summary {
            width: min(420px, 30vw);
            max-height: 130px;
            overflow: auto;
            background: #11100D;
            border: 1px solid #4A4032;
            color: #F1E8D2;
            padding: 8px;
            font-size: 12px;
            line-height: 1.35;
            white-space: pre-line;
        }
        @media (max-width: 900px) {
            .iamccs-cine-steps,
            .iamccs-cine-asset-grid {
                grid-template-columns: 1fr;
            }
            .iamccs-cine-summary,
            .iamccs-cine-preview {
                width: 100%;
            }
        }
    `;
    document.head.appendChild(style);
}

function makeOption(value, label = value) {
    const option = document.createElement("option");
    option.value = valueFromOption(value);
    option.textContent = label === value ? labelFromOption(value) : label;
    return option;
}

function makeSelect(value, options, onChange) {
    const select = document.createElement("select");
    select.className = "iamccs-cine-select";
    for (const option of options) select.appendChild(makeOption(option));
    select.value = String(value || valueFromOption(options[0]) || "");
    const optionValues = options.map(valueFromOption);
    if (!optionValues.includes(select.value) && options.length) {
        select.appendChild(makeOption(select.value, select.value));
    }
    select.addEventListener("change", () => onChange(select.value));
    return select;
}

function makeInput(value, onChange, attrs = {}) {
    const input = document.createElement("input");
    input.className = "iamccs-cine-input";
    input.value = String(value ?? "");
    for (const [key, val] of Object.entries(attrs)) input.setAttribute(key, val);
    input.addEventListener("input", () => onChange(input.value));
    return input;
}

function makeTextarea(value, onChange) {
    const textarea = document.createElement("textarea");
    textarea.className = "iamccs-cine-textarea";
    textarea.value = String(value ?? "");
    textarea.addEventListener("input", () => onChange(textarea.value));
    return textarea;
}

function stopCanvasKeys(element) {
    const stop = (event) => event.stopPropagation();
    for (const eventName of ["keydown", "keyup", "keypress", "pointerdown", "pointerup", "wheel"]) {
        element.addEventListener(eventName, stop, { capture: true });
    }
}

function createDatalist(id, values) {
    let list = document.getElementById(id);
    if (list) return;
    list = document.createElement("datalist");
    list.id = id;
    for (const value of values) list.appendChild(makeOption(value));
    document.body.appendChild(list);
}

function addCell(rowEl, child, width = null) {
    const td = document.createElement("td");
    if (width) td.style.width = width;
    td.appendChild(child);
    rowEl.appendChild(td);
    return td;
}

function openTimelineEditor(node, config, options = {}) {
    ensureStyles();
    createDatalist("iamccs-cine-ref-options", REF_OPTIONS);
    createDatalist("iamccs-cine-range-options", SOURCE_RANGE_OPTIONS);

    const widget = getWidget(node, config.widgetName);
    const rows = parseRows(widget?.value || "", config);
    let selectedPreset = config.kind === "v2v" ? "v2v_two_camera" : "multigen_dialogue_4";
    const directorMode = options.directorMode === true;
    let assetLabels = getAssetLabels(node);

    const overlay = document.createElement("div");
    overlay.className = "iamccs-cine-overlay";
    const modal = document.createElement("div");
    modal.className = "iamccs-cine-modal";
    if (directorMode) modal.classList.add("director");
    overlay.appendChild(modal);
    stopCanvasKeys(overlay);

    const header = document.createElement("div");
    header.className = "iamccs-cine-header";
    const title = document.createElement("div");
    title.className = "iamccs-cine-title";
    title.textContent = options.title || config.title;
    const help = document.createElement("div");
    help.className = "iamccs-cine-help";
    help.textContent = directorMode
        ? "Director App: scegli preset, dai nomi umani agli asset, costruisci la timeline e applica al nodo."
        : config.kind === "v2v"
        ? "Edit V2V source, range, reference, audio and prompt. Apply writes timeline_lines."
        : "Edit shots with dropdowns and prompt boxes. Apply writes shot_lines.";
    header.append(title, help);

    const toolbar = document.createElement("div");
    toolbar.className = "iamccs-cine-toolbar";
    if (directorMode) {
        const steps = document.createElement("div");
        steps.className = "iamccs-cine-steps";
        steps.append(
            stepCard("1. Preset", "Parti da una scena tipo, poi cambiala."),
            stepCard("2. Asset", "Nomina immagini, video e audio come personaggi."),
            stepCard("3. Timeline", "Una riga = uno shot o segmento."),
            stepCard("4. Applica", "Apply to node, poi Queue in ComfyUI."),
        );
        toolbar.appendChild(steps);
    }

    const presetSelect = makeSelect(selectedPreset, config.kind === "v2v"
        ? [
            { value: "v2v_two_camera", label: "V2V two cameras dialogue" },
            { value: "v2v_single", label: "V2V single source restyle" },
            { value: "v2v_long", label: "V2V long continuity" },
            { value: "v2v_bar_dialogue", label: "Bar dialogue with source audio" },
            { value: "v2v_flashback", label: "Flashback / memory sequence" },
        ]
        : [
            { value: "multigen_dialogue_4", label: "Dialogue 4 shots" },
            { value: "multigen_dialogue_2", label: "Dialogue 2 shots" },
            { value: "multigen_monologue", label: "Monologue" },
            { value: "multigen_zoom_drone", label: "Zoom drone face to sea" },
            { value: "multigen_astronaut_action", label: "Astronaut laser action" },
            { value: "multigen_bar_dialogue", label: "Bar dialogue city wide" },
            { value: "multigen_product_reveal", label: "Product reveal" },
            { value: "multigen_music_clip", label: "Music clip" },
        ], (value) => {
            selectedPreset = value;
        });
    const loadPreset = button("Load preset", () => {
        rows.splice(0, rows.length, ...presetRows(selectedPreset));
        renderRows();
        updatePreview();
    });
    const addRow = button("Add row", () => {
        const base = rows[rows.length - 1] || (config.kind === "v2v" ? presetRows("v2v_single")[0] : presetRows("multigen_monologue")[0]);
        rows.push({ ...base, label: `${base.label || "shot"}_${rows.length + 1}` });
        renderRows();
        updatePreview();
    });
    const clearRows = button("Clear", () => {
        rows.splice(0, rows.length);
        renderRows();
        updatePreview();
    }, "danger");
    toolbar.append(labelText("Preset"), presetSelect, loadPreset, addRow, clearRows);

    const assetPanel = document.createElement("div");
    assetPanel.className = "iamccs-cine-assets";
    if (directorMode) {
        renderAssetPanel();
    }

    const body = document.createElement("div");
    body.className = "iamccs-cine-body";
    const table = document.createElement("table");
    table.className = "iamccs-cine-table";
    body.appendChild(table);

    const footer = document.createElement("div");
    footer.className = "iamccs-cine-footer";
    const summary = document.createElement("div");
    summary.className = "iamccs-cine-summary";
    const status = document.createElement("div");
    status.className = "iamccs-cine-status";
    const preview = document.createElement("textarea");
    preview.className = "iamccs-cine-textarea iamccs-cine-preview";
    preview.readOnly = true;
    const actions = document.createElement("div");
    actions.style.display = "flex";
    actions.style.gap = "8px";
    actions.style.alignItems = "center";
    actions.append(
        button("Copy text", async () => {
            try {
                await navigator.clipboard?.writeText(preview.value);
                status.textContent = "Copied generated lines.";
            } catch {
                status.textContent = "Copy failed; select preview text manually.";
            }
        }),
        button("Cancel", () => close()),
        button("Apply to node", () => {
            setWidgetValue(node, config.widgetName, rowsToText(rows, config));
            status.textContent = `Applied ${rows.length} rows to ${config.widgetName}.`;
            close();
        }, "primary"),
    );
    footer.append(summary, preview, status, actions);

    modal.append(header, toolbar);
    if (directorMode) modal.appendChild(assetPanel);
    modal.append(body, footer);
    document.body.appendChild(overlay);

    overlay.addEventListener("pointerdown", (event) => {
        if (event.target === overlay) close();
    });
    document.addEventListener("keydown", onDocumentKeydown, { capture: true });

    function onDocumentKeydown(event) {
        if (event.key === "Escape") {
            event.preventDefault();
            event.stopPropagation();
            close();
        }
    }

    function close() {
        document.removeEventListener("keydown", onDocumentKeydown, { capture: true });
        overlay.remove();
    }

    function renderRows() {
        table.textContent = "";
        const thead = document.createElement("thead");
        const headerRow = document.createElement("tr");
        const columns = config.kind === "v2v"
            ? ["#", "seconds", "cut", "source", "range", "ref", "audio", "label", "framing", "prompt", "dialogue", "voice", "mode", "actions"]
            : ["#", "seconds", "cut", "ref", "audio", "label", "framing", "prompt", "dialogue", "voice", "actions"];
        for (const col of columns) {
            const th = document.createElement("th");
            th.textContent = col;
            headerRow.appendChild(th);
        }
        thead.appendChild(headerRow);
        table.appendChild(thead);

        const tbody = document.createElement("tbody");
        rows.forEach((row, index) => {
            const tr = document.createElement("tr");
            addIndexCell(tr, index);
            addCell(tr, makeInput(row.seconds, (v) => { row.seconds = v; updatePreview(); }, { type: "number", step: "0.1", min: "0.01" }), "70px");
            addCell(tr, makeSelect(row.cut, CUT_MODES, (v) => { row.cut = v; updatePreview(); }), "135px");
            if (config.kind === "v2v") {
                addCell(tr, makeSelect(row.source, SOURCE_OPTIONS, (v) => { row.source = v; updatePreview(); }), "70px");
                addCell(tr, makeInput(row.sourceRange, (v) => { row.sourceRange = v; updatePreview(); }, { list: "iamccs-cine-range-options" }), "115px");
            }
            addCell(tr, makeSelect(row.ref, REF_OPTIONS, (v) => { row.ref = v; updatePreview(); }), "145px");
            addCell(tr, makeSelect(row.audio, AUDIO_OPTIONS, (v) => { row.audio = v; updatePreview(); }), "135px");
            addCell(tr, makeInput(row.label, (v) => { row.label = v; updatePreview(); }), "140px");
            addCell(tr, makeSelect(row.framing, FRAMING_OPTIONS, (v) => { row.framing = v; updatePreview(); }), "160px");
            addCell(tr, makeTextarea(row.prompt, (v) => { row.prompt = v; updatePreview(); }), "260px");
            addCell(tr, makeTextarea(row.dialogue, (v) => { row.dialogue = v; updatePreview(); }), "220px");
            addCell(tr, makeTextarea(row.voice, (v) => { row.voice = v; updatePreview(); }), "170px");
            if (config.kind === "v2v") {
                addCell(tr, makeSelect(row.v2vMode, V2V_MODES, (v) => { row.v2vMode = v; updatePreview(); }), "190px");
            }
            addCell(tr, rowActions(index), "74px");
            tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        updatePreview();
    }

    function addIndexCell(tr, index) {
        const div = document.createElement("div");
        div.className = "iamccs-cine-row-index";
        div.textContent = String(index + 1);
        addCell(tr, div, "34px");
    }

    function rowActions(index) {
        const box = document.createElement("div");
        box.className = "iamccs-cine-row-actions";
        box.append(
            button("up", () => {
                if (index <= 0) return;
                const [row] = rows.splice(index, 1);
                rows.splice(index - 1, 0, row);
                renderRows();
            }),
            button("down", () => {
                if (index >= rows.length - 1) return;
                const [row] = rows.splice(index, 1);
                rows.splice(index + 1, 0, row);
                renderRows();
            }),
            button("dup", () => {
                rows.splice(index + 1, 0, { ...rows[index], label: `${rows[index].label || "shot"}_copy` });
                renderRows();
            }),
            button("del", () => {
                rows.splice(index, 1);
                renderRows();
            }, "danger"),
        );
        return box;
    }

    function updatePreview() {
        const text = rowsToText(rows, config);
        preview.value = text;
        summary.textContent = rows.map((row, index) => rowSummary(row, index, config, assetLabels)).join("\n");
        const warnings = computeWarnings(rows, config);
        status.style.color = warnings.length ? "#ffd36a" : "#9eb2c4";
        status.textContent = warnings.length
            ? `Warnings:\n- ${warnings.slice(0, 6).join("\n- ")}${warnings.length > 6 ? "\n- ..." : ""}`
            : `${rows.length} row(s). ${config.widgetName} will receive ${text.length} characters.`;
    }

    function renderAssetPanel() {
        assetPanel.textContent = "";
        const titleEl = document.createElement("div");
        titleEl.className = "iamccs-cine-assets-title";
        titleEl.textContent = "Asset names";
        const grid = document.createElement("div");
        grid.className = "iamccs-cine-asset-grid";
        const fields = config.kind === "v2v"
            ? [
                ["source1", "Source 1"], ["source2", "Source 2"], ["source3", "Source 3"], ["source4", "Source 4"],
                ["ref1", "Reference 1"], ["ref2", "Reference 2"], ["ref3", "Reference 3"], ["ref4", "Reference 4"],
                ["audio1", "Audio 1"], ["audio2", "Audio 2"], ["audio3", "Audio 3"], ["audio4", "Audio 4"],
            ]
            : [
                ["ref1", "Reference 1"], ["ref2", "Reference 2"], ["ref3", "Reference 3"], ["ref4", "Reference 4"],
                ["ref5", "Reference 5"], ["ref6", "Reference 6"], ["audio1", "Audio 1"], ["audio2", "Audio 2"],
                ["audio3", "Audio 3"], ["audio4", "Audio 4"],
            ];
        for (const [key, label] of fields) {
            const field = document.createElement("div");
            field.className = "iamccs-cine-asset-field";
            const lab = document.createElement("label");
            lab.textContent = label;
            field.append(
                lab,
                makeInput(assetLabels[key] || "", (value) => {
                    setAssetLabel(node, key, value);
                    assetLabels = getAssetLabels(node);
                    updatePreview();
                }),
            );
            grid.appendChild(field);
        }
        assetPanel.append(titleEl, grid);
    }

    renderRows();
}

function stepCard(title, text) {
    const card = document.createElement("div");
    card.className = "iamccs-cine-step";
    const strong = document.createElement("strong");
    strong.textContent = title;
    const span = document.createElement("span");
    span.textContent = text;
    card.append(strong, span);
    return card;
}

function button(text, onClick, variant = "") {
    const btn = document.createElement("button");
    btn.className = `iamccs-cine-btn ${variant || ""}`.trim();
    btn.type = "button";
    btn.textContent = text;
    btn.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        onClick();
    });
    return btn;
}

function labelText(text) {
    const span = document.createElement("span");
    span.className = "iamccs-cine-help";
    span.textContent = text;
    return span;
}

function installEditorButton(node, config) {
    if (!node || node.__iamccsTimelineEditorInstalled) return;
    node.__iamccsTimelineEditorInstalled = true;
    node.properties = node.properties || {};
    node.color = CINE_FILM_LAB.header;
    node.bgcolor = CINE_FILM_LAB.nodeBg;
    node.boxcolor = CINE_FILM_LAB.relay;

    const rawWidget = getWidget(node, config.widgetName);
    if (rawWidget) {
        rawWidget.label = config.kind === "v2v" ? "timeline_lines raw text" : "shot_lines raw text";
    }

    const directorButton = node.addWidget("button", "Open Director App", null, () => {
        openTimelineEditor(node, config, {
            directorMode: true,
            title: config.kind === "v2v" ? "IAMCCS V2V Director App" : "IAMCCS Cinematic Director App",
        });
    }, { serialize: false });
    directorButton.serialize = false;
    directorButton.options = { ...(directorButton.options || {}), serialize: false };

    const editorButton = node.addWidget("button", "Open Timeline Editor", null, () => {
        openTimelineEditor(node, config);
    }, { serialize: false });
    editorButton.serialize = false;
    editorButton.options = { ...(editorButton.options || {}), serialize: false };

    const presetButton = node.addWidget("button", "Load Default Timeline", null, () => {
        const key = config.kind === "v2v" ? "v2v_two_camera" : "multigen_dialogue_4";
        const rows = presetRows(key);
        setWidgetValue(node, config.widgetName, rowsToText(rows, config));
    }, { serialize: false });
    presetButton.serialize = false;
    presetButton.options = { ...(presetButton.options || {}), serialize: false };

    try {
        const size = node.computeSize?.() || node.size;
        node.setSize?.([Math.max(node.size?.[0] || 0, 560), Math.max(node.size?.[1] || 0, size?.[1] || 520)]);
        node.setDirtyCanvas?.(true, true);
    } catch {
        // ignore
    }
}

app.registerExtension({
    name: "iamccs.cinematic.timeline_editor",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        const config = getConfigForNodeData(nodeData);
        if (!config) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            installEditorButton(this, config);
            return result;
        };
    },

    async loadedGraphNode(node) {
        const config = getConfigForNode(node);
        if (config) installEditorButton(node, config);
    },
});

console.log(`[IAMCCS Cinematic] Timeline editor loaded v=${EDITOR_VERSION}`);
